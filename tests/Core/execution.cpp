// Tests of the execution policies and the primitives every count-sized loop runs through:
// the ExecutionPolicy concept, forEachIndex(), forEachChunk(), reduceRanges(), transformReduce(),
// cumulativeSum(), gather(), copyIfIndex(), and fill() in EmrysFilter/Core/Execution.h, under
// Sequential, the standard policies, and policies of the user's own, and the
// associative LogSumExp, logSumExp(), and logTotalWeight() under a policy in Core/Weights.h.
// Roadmap step 18; the evidence for [execution-policies]: a for-each matches Sequential under every policy, and a
// reduction or scan under a parallel policy is the same at every run and matches Sequential to
// rounding.

#include <gtest/gtest.h>

#include <cmath>
#include <compare>
#include <execution>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Weights.h>

#include "standard_policies.h"
#include "user_policies.h"

namespace execution = Emrys::execution;

namespace
{

using EmrysTest::ReversedPolicy;
using EmrysTest::ThreadsPolicy;
using Index = Eigen::Index;

// Enough elements for several chunks and a short last one. It and mixedValues() serve only the
// tests of the standard policies, which some standard libraries leave out.
[[maybe_unused]] constexpr Index manyElements = 5 * Emrys::internal::chunkSize + 37;

// Values of mixed magnitude, so that the order of a floating-point sum shows in its rounding.
[[maybe_unused]] std::vector<double>
mixedValues(Index count)
{
    auto rng = std::mt19937{7};
    auto exponent = std::uniform_real_distribution<double>{-8.0, 8.0};
    auto values = std::vector<double>(static_cast<std::size_t>(count));
    for (double & v : values)
    {
        v = std::exp(exponent(rng));
    }
    return values;
}

// Not policies: a forEachIndex() that isn't const, and one that takes no body.
struct MutableForEach
{
    template <typename F>
    void
    forEachIndex(Index, F const &)
    {}
};

struct NoBody
{
    void
    forEachIndex(Index) const
    {}
};

}  // namespace

// The concept admits Sequential and the standard policies, and nothing a filter step takes after
// the policy: a generator, a time step, or an integer.
TEST(ExecutionPolicy, AdmitsSequentialAndTheStandardPolicies)
{
    static_assert(Emrys::ExecutionPolicy<execution::Sequential>);
    static_assert(Emrys::ExecutionPolicy<execution::Sequential const &>);
#if EMRYS_FILTER_TEST_STANDARD_POLICIES
    static_assert(Emrys::ExecutionPolicy<decltype(std::execution::seq)>);
    static_assert(Emrys::ExecutionPolicy<decltype(std::execution::par) const &>);
    static_assert(Emrys::ExecutionPolicy<decltype(std::execution::par_unseq)>);
    static_assert(Emrys::ExecutionPolicy<decltype(std::execution::unseq)>);
    static_assert(!Emrys::internal::isSequential<decltype(std::execution::par)>);
#endif
    static_assert(!Emrys::ExecutionPolicy<std::mt19937 &>);
    static_assert(!Emrys::ExecutionPolicy<double>);
    static_assert(!Emrys::ExecutionPolicy<int>);
    static_assert(Emrys::internal::isSequential<execution::Sequential const &>);
    SUCCEED();
}

// A policy of the user's own is a type with a const forEachIndex(count, f), and it leads an
// argument list as the standard policies do; one whose forEachIndex() isn't const, or takes no
// body, is no policy.
TEST(ExecutionPolicy, AdmitsAUsersPolicy)
{
    static_assert(Emrys::ExecutionPolicy<ReversedPolicy>);
    static_assert(Emrys::ExecutionPolicy<ThreadsPolicy const &>);
    static_assert(!Emrys::ExecutionPolicy<MutableForEach>);
    static_assert(!Emrys::ExecutionPolicy<NoBody>);
    static_assert(!Emrys::internal::isSequential<ReversedPolicy>);
    static_assert(Emrys::internal::leadsWithPolicy<ReversedPolicy, double>);
    static_assert(!Emrys::internal::leadsWithPolicy<NoBody, double>);
    SUCCEED();
}

#if EMRYS_FILTER_TEST_STANDARD_POLICIES

// Every index is visited exactly once, under each policy, including an empty range.
TEST(ExecutionPrimitives, ForEachIndexVisitsEveryIndexOnce)
{
    auto const check = [](auto const & policy) {
        auto visits = std::vector<int>(static_cast<std::size_t>(manyElements), 0);
        Emrys::internal::forEachIndex(policy, manyElements, [&](Index i) {
            ++visits[std::size_t(i)];
        });
        for (int const v : visits)
        {
            EXPECT_EQ(v, 1);
        }
        Emrys::internal::forEachIndex(policy, 0, [](Index) {
            ADD_FAILURE() << "an empty range has no index";
        });
    };
    check(execution::sequential);
    check(std::execution::seq);
    check(std::execution::par);
    check(std::execution::par_unseq);
}

// Under Sequential a non-empty range is one chunk; otherwise the chunks are chunkSize long, the
// last one short, and together cover the range once.
TEST(ExecutionPrimitives, ForEachChunkCoversTheRange)
{
    auto ranges = std::vector<std::pair<Index, Index>>{};
    Emrys::internal::forEachChunk(execution::sequential, manyElements, [&](Index begin, Index length) {
        ranges.emplace_back(begin, length);
    });
    ASSERT_EQ(ranges.size(), 1U);
    EXPECT_EQ(ranges[0], std::make_pair(Index{0}, manyElements));

    auto covered = std::vector<int>(static_cast<std::size_t>(manyElements), 0);
    auto lengths = std::vector<Index>(static_cast<std::size_t>(Emrys::internal::chunkCount(manyElements)), 0);
    Emrys::internal::forEachChunk(std::execution::par, manyElements, [&](Index begin, Index length) {
        EXPECT_EQ(begin % Emrys::internal::chunkSize, 0);
        lengths[std::size_t(begin / Emrys::internal::chunkSize)] = length;
        for (Index i = begin; i < begin + length; ++i)
        {
            ++covered[std::size_t(i)];
        }
    });
    for (int const c : covered)
    {
        EXPECT_EQ(c, 1);
    }
    ASSERT_EQ(lengths.size(), 6U);
    EXPECT_EQ(lengths.front(), Emrys::internal::chunkSize);
    EXPECT_EQ(lengths.back(), 37);
}

// An integer sum is exact, so it is the same under every policy; a floating-point sum under a
// parallel policy is the sum of each chunk, added in chunk order, which a plain loop reproduces
// bitwise; and under Sequential it is the whole range in order.
TEST(ExecutionPrimitives, ReductionsCombineChunksInOrder)
{
    auto const intSum = [](auto const & policy) {
        return Emrys::internal::transformReduce(policy, manyElements, Index{0}, std::plus<>{}, [](Index i) {
            return i;
        });
    };
    Index const expected = manyElements * (manyElements - 1) / 2;
    EXPECT_EQ(intSum(execution::sequential), expected);
    EXPECT_EQ(intSum(std::execution::par), expected);
    EXPECT_EQ(intSum(std::execution::par_unseq), expected);

    std::vector<double> const values = mixedValues(manyElements);
    auto const sum = [&](auto const & policy) {
        return Emrys::internal::reduceRanges(
            policy,
            manyElements,
            [&](Index begin, Index length) {
                double s = 0;
                for (Index i = begin; i < begin + length; ++i)
                {
                    s += values[std::size_t(i)];
                }
                return s;
            },
            std::plus<>{}
        );
    };
    double sequential = 0;
    for (double const v : values)
    {
        sequential += v;
    }
    double chunked = 0;
    for (Index begin = 0; begin < manyElements; begin += Emrys::internal::chunkSize)
    {
        double s = 0;
        for (Index i = begin; i < std::min(begin + Emrys::internal::chunkSize, manyElements); ++i)
        {
            s += values[std::size_t(i)];
        }
        chunked = begin == 0 ? s : chunked + s;
    }
    EXPECT_EQ(sum(execution::sequential), sequential);
    EXPECT_EQ(sum(std::execution::par), chunked);
    EXPECT_EQ(sum(std::execution::par_unseq), chunked);
    EXPECT_EQ(sum(std::execution::par), sum(std::execution::par));
    EXPECT_NEAR(sum(std::execution::par), sequential, 1e-12 * sequential);
}

// A reduction over a single chunk is Sequential's, bitwise.
TEST(ExecutionPrimitives, OneChunkReducesAsSequential)
{
    std::vector<double> const values = mixedValues(Emrys::internal::chunkSize);
    auto const sum = [&](auto const & policy) {
        return Emrys::internal::transformReduce(policy, Index(values.size()), 0.0, std::plus<>{}, [&](Index i) {
            return values[std::size_t(i)];
        });
    };
    EXPECT_EQ(sum(std::execution::par), sum(execution::sequential));
}

// A scan of a single chunk under a parallel policy is Sequential's, bitwise, with no offsets.
TEST(ExecutionPrimitives, OneChunkScansAsSequential)
{
    std::vector<double> const values = mixedValues(Emrys::internal::chunkSize);
    auto sequential = std::vector<double>(values.size());
    auto threaded = std::vector<double>(values.size());
    Emrys::internal::cumulativeSum(
        execution::sequential,
        std::span<double const>{values},
        std::span<double>{sequential}
    );
    Emrys::internal::cumulativeSum(ThreadsPolicy{}, std::span<double const>{values}, std::span<double>{threaded});
    EXPECT_EQ(threaded, sequential);
}

// The iterators that the standard algorithms run over move and compare as their indices do, in
// each of the operations a random-access iterator has.
TEST(ExecutionPrimitives, IteratorsMoveAsTheirIndices)
{
    using Emrys::internal::CountingIterator;
    auto it = CountingIterator{5};
    EXPECT_EQ(*it++, 5);
    EXPECT_EQ(*it, 6);
    EXPECT_EQ(*it--, 6);
    EXPECT_EQ(*--it, 4);
    it += 3;
    EXPECT_EQ(*it, 7);
    it -= 2;
    EXPECT_EQ(*it, 5);
    EXPECT_EQ(*(2 + it), 7);
    EXPECT_EQ(*(it - 2), 3);
    EXPECT_EQ(it[1], 6);

    struct Squares
    {
        using value_type = Index;

        Index
        operator[](Index i) const
        {
            return i * i;
        }
    };

    auto const squares = Squares{};
    using Iterator = Emrys::internal::ElementIterator<Squares>;
    static_assert(std::random_access_iterator<Iterator>);
    auto const first = Iterator{squares, 1};
    auto const third = Iterator{squares, 3};
    EXPECT_EQ(*(2 + first), 9);
    EXPECT_EQ(third - first, 2);
    EXPECT_LT(first, third);
    EXPECT_EQ(first <=> third, std::strong_ordering::less);
    EXPECT_GE(third, first);
}

// The scan matches std::inclusive_scan exactly on integers under every policy, also in place,
// and on floating point matches Sequential to rounding; Sequential matches std::partial_sum.
TEST(ExecutionPrimitives, CumulativeSum)
{
    auto counts = std::vector<Index>(static_cast<std::size_t>(manyElements));
    for (std::size_t i = 0; i < counts.size(); ++i)
    {
        counts[i] = Index(i % 3);
    }
    auto expected = std::vector<Index>(counts.size());
    std::partial_sum(counts.begin(), counts.end(), expected.begin());
    for (auto const scanIntegers : {0, 1})
    {
        auto out = std::vector<Index>(counts.size());
        std::vector<Index> inPlace = counts;
        if (scanIntegers == 0)
        {
            Emrys::internal::cumulativeSum(
                execution::sequential,
                std::span<Index const>{counts},
                std::span<Index>{out}
            );
            Emrys::internal::cumulativeSum(
                execution::sequential,
                std::span<Index const>{inPlace},
                std::span<Index>{inPlace}
            );
        }
        else
        {
            Emrys::internal::cumulativeSum(std::execution::par, std::span<Index const>{counts}, std::span<Index>{out});
            Emrys::internal::cumulativeSum(
                std::execution::par,
                std::span<Index const>{inPlace},
                std::span<Index>{inPlace}
            );
        }
        EXPECT_EQ(out, expected);
        EXPECT_EQ(inPlace, expected);
    }

    std::vector<double> const values = mixedValues(manyElements);
    auto sequential = std::vector<double>(values.size());
    auto parallel = std::vector<double>(values.size());
    auto partial = std::vector<double>(values.size());
    Emrys::internal::cumulativeSum(
        execution::sequential,
        std::span<double const>{values},
        std::span<double>{sequential}
    );
    Emrys::internal::cumulativeSum(std::execution::par, std::span<double const>{values}, std::span<double>{parallel});
    std::partial_sum(values.begin(), values.end(), partial.begin());
    EXPECT_EQ(sequential, partial);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_NEAR(parallel[i], sequential[i], 1e-12 * sequential[i]);
        if (i < std::size_t(Emrys::internal::chunkSize))
        {
            EXPECT_EQ(parallel[i], sequential[i]);
        }
    }

    auto empty = std::vector<double>{};
    Emrys::internal::cumulativeSum(std::execution::par, std::span<double const>{empty}, std::span<double>{empty});
}

// The compaction keeps the indices for which keep() holds, in order, in place under Sequential
// and into a separate buffer under a parallel policy.
TEST(ExecutionPrimitives, CopyIfIndexCompactsInOrder)
{
    auto const keep = [](Index i) {
        return i % 3 != 1;
    };
    auto expected = std::vector<Index>{};
    for (Index i = 0; i < manyElements; ++i)
    {
        if (keep(i))
        {
            expected.push_back(i);
        }
    }

    auto inPlace = std::vector<Index>(static_cast<std::size_t>(manyElements));
    std::iota(inPlace.begin(), inPlace.end(), Index{0});
    Index const kept = Emrys::internal::copyIfIndex(execution::sequential, manyElements, keep, [&](Index i, Index j) {
        inPlace[std::size_t(j)] = inPlace[std::size_t(i)];
    });
    inPlace.resize(std::size_t(kept));
    EXPECT_EQ(inPlace, expected);

    auto out = std::vector<Index>(static_cast<std::size_t>(manyElements), -1);
    Index const keptInParallel =
        Emrys::internal::copyIfIndex(std::execution::par, manyElements, keep, [&](Index i, Index j) {
            out[std::size_t(j)] = i;
        });
    out.resize(std::size_t(keptInParallel));
    EXPECT_EQ(out, expected);
    EXPECT_EQ(Emrys::internal::copyIfIndex(std::execution::par, 0, keep, [](Index, Index) {}), 0);
}

// Under a user's policy each primitive gives std::execution::par's result bitwise, whether the
// policy visits the indices in reverse or on threads of its own: the for-each visits every index
// once, and the reduction, the scan, and the compaction combine the same chunks in the same order.
TEST(ExecutionPrimitives, UsersPoliciesMatchPar)
{
    std::vector<double> const values = mixedValues(manyElements);
    auto const run = [&](auto const & policy) {
        auto visits = std::vector<int>(values.size(), 0);
        Emrys::internal::forEachIndex(policy, manyElements, [&](Index i) {
            ++visits[std::size_t(i)];
        });
        double const sum = Emrys::internal::transformReduce(policy, manyElements, 0.0, std::plus<>{}, [&](Index i) {
            return values[std::size_t(i)];
        });
        auto cumulative = std::vector<double>(values.size());
        Emrys::internal::cumulativeSum(policy, std::span<double const>{values}, std::span<double>{cumulative});
        auto kept = std::vector<Index>(values.size(), -1);
        auto const keep = [](Index i) {
            return i % 3 != 1;
        };
        Index const count = Emrys::internal::copyIfIndex(policy, manyElements, keep, [&](Index i, Index j) {
            kept[std::size_t(j)] = i;
        });
        kept.resize(std::size_t(count));
        return std::tuple{visits, sum, cumulative, kept};
    };
    auto const expected = run(std::execution::par);
    EXPECT_EQ(std::get<0>(expected), std::vector<int>(values.size(), 1));
    EXPECT_EQ(run(ReversedPolicy{}), expected);
    EXPECT_EQ(run(ThreadsPolicy{}), expected);
}

// gather() copies the indexed columns, repeats included; fill() sets every element.
TEST(ExecutionPrimitives, GatherAndFill)
{
    auto from = Eigen::Matrix<double, 2, Eigen::Dynamic>{2, 4};
    from << 1, 2, 3, 4, 5, 6, 7, 8;
    auto const indices = std::vector<Index>{3, 0, 0, 2, 1};
    for (bool const parallel : {false, true})
    {
        auto to = Eigen::Matrix<double, 2, Eigen::Dynamic>{2, 5};
        if (parallel)
        {
            Emrys::internal::gather(std::execution::par, std::span<Index const>{indices}, from, to);
        }
        else
        {
            Emrys::internal::gather(execution::sequential, std::span<Index const>{indices}, from, to);
        }
        for (std::size_t j = 0; j < indices.size(); ++j)
        {
            EXPECT_EQ(to.col(Index(j)), from.col(indices[j]));
        }
    }

    auto values = std::vector<double>(static_cast<std::size_t>(manyElements), 0.0);
    Emrys::internal::fill(std::execution::par, std::span<double>{values}, 2.5);
    for (double const v : values)
    {
        EXPECT_EQ(v, 2.5);
    }
    Emrys::internal::fill(execution::sequential, std::span<double>{values}, -1.0);
    EXPECT_EQ(values.front(), -1.0);
    EXPECT_EQ(values.back(), -1.0);
}

#endif  // EMRYS_FILTER_TEST_STANDARD_POLICIES

// Adding whole sums agrees with adding the inputs one at a time, to rounding, in either order and
// with the empty sum on either side; -inf inputs add nothing.
TEST(LogSumExp, AddingSumsIsAssociative)
{
    auto const inputs = std::vector<double>{-3.0, 700.0, -std::numeric_limits<double>::infinity(), 699.5, 1.0, -1000.0};
    auto streamed = Emrys::LogSumExp<double>{};
    for (double const x : inputs)
    {
        streamed.add(x);
    }
    for (std::size_t split = 0; split <= inputs.size(); ++split)
    {
        auto first = Emrys::LogSumExp<double>{};
        auto second = Emrys::LogSumExp<double>{};
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            (i < split ? first : second).add(inputs[i]);
        }
        auto forward = Emrys::LogSumExp<double>{first};
        forward.add(second);
        auto backward = Emrys::LogSumExp<double>{second};
        backward.add(first);
        EXPECT_NEAR(forward.value(), streamed.value(), 1e-12);
        EXPECT_NEAR(backward.value(), streamed.value(), 1e-12);
    }
    auto empty = Emrys::LogSumExp<double>{};
    empty.add(Emrys::LogSumExp<double>{});
    EXPECT_EQ(empty.value(), -std::numeric_limits<double>::infinity());
}

// Adding an array whole agrees with adding its elements one at a time, to rounding, on top of an
// empty or a nonempty sum; an empty array or one of -inf adds nothing, a NaN gives NaN, and +inf
// inputs give +inf.
TEST(LogSumExp, AddingAnArrayMatchesAddingItsElements)
{
    double const inf = std::numeric_limits<double>::infinity();
    auto const inputs = Eigen::ArrayXd{{-3.0, 700.0, -inf, 699.5, 1.0, -1000.0}};
    for (double const before : {-inf, 0.0, 800.0})
    {
        auto streamed = Emrys::LogSumExp<double>{};
        auto whole = Emrys::LogSumExp<double>{};
        streamed.add(before);
        whole.add(before);
        for (double const x : inputs)
        {
            streamed.add(x);
        }
        whole.add(inputs);
        EXPECT_NEAR(whole.value(), streamed.value(), 1e-12) << before;
    }

    auto sum = Emrys::LogSumExp<double>{};
    sum.add(Eigen::ArrayXd{});
    sum.add(Eigen::ArrayXd::Constant(3, -inf));
    EXPECT_EQ(sum.value(), -inf);
    sum.add(Eigen::ArrayXd{{1.0, std::numeric_limits<double>::quiet_NaN()}});
    EXPECT_TRUE(std::isnan(sum.value()));
    auto infinite = Emrys::LogSumExp<double>{};
    infinite.add(Eigen::ArrayXd{{1.0, inf, inf}});
    EXPECT_EQ(infinite.value(), inf);
}

#if EMRYS_FILTER_TEST_STANDARD_POLICIES

// logTotalWeight() under a policy: Sequential's is the range form's and the array logSumExp()'s
// exactly, and the streaming logSumExp()'s to rounding; a parallel one's agrees to rounding and is
// the same every run, and each error is found in any chunk.
TEST(LogSumExp, LogTotalWeightUnderAPolicy)
{
    std::vector<double> logWeights = mixedValues(manyElements);
    for (double & w : logWeights)
    {
        w = std::log(w) - 20.0;
    }
    auto const weights = std::span<double const>{logWeights};
    auto const expected = Emrys::internal::logTotalWeight<double>(weights);
    ASSERT_TRUE(expected);
    EXPECT_EQ(Emrys::internal::logTotalWeight<double>(execution::sequential, weights).value(), *expected);
    auto const parallel = Emrys::internal::logTotalWeight<double>(std::execution::par, weights);
    ASSERT_TRUE(parallel);
    EXPECT_NEAR(*parallel, *expected, 1e-12);
    EXPECT_EQ(*parallel, Emrys::internal::logTotalWeight<double>(std::execution::par, weights).value());
    auto const asArray = Eigen::Map<Eigen::ArrayXd const>{logWeights.data(), manyElements};
    EXPECT_EQ(Emrys::logSumExp(execution::sequential, asArray), *expected);
    EXPECT_NEAR(Emrys::logSumExp(std::execution::par, asArray), *expected, 1e-12);
    EXPECT_NEAR(
        Emrys::logSumExp<double>(
            execution::sequential,
            manyElements,
            [&](Index i) {
                return logWeights[std::size_t(i)];
            }
        ),
        *expected,
        1e-12
    );

    auto const empty = std::vector<double>{};
    EXPECT_EQ(
        Emrys::internal::logTotalWeight<double>(std::execution::par, std::span<double const>{empty}).error(),
        Emrys::WeightError::Empty
    );
    auto zeros = std::vector<double>(static_cast<std::size_t>(manyElements), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(
        Emrys::internal::logTotalWeight<double>(std::execution::par, std::span<double const>{zeros}).error(),
        Emrys::WeightError::ZeroSum
    );
    logWeights[std::size_t(manyElements - 3)] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(
        Emrys::internal::logTotalWeight<double>(std::execution::par, weights).error(),
        Emrys::WeightError::NonFinite
    );
    logWeights[std::size_t(manyElements - 3)] = std::numeric_limits<double>::infinity();
    EXPECT_EQ(
        Emrys::internal::logTotalWeight<double>(std::execution::par, weights).error(),
        Emrys::WeightError::NonFinite
    );
}

#endif  // EMRYS_FILTER_TEST_STANDARD_POLICIES
