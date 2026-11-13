// The resamplers of Core/Resampling.h (step 7, [upkeep]): Multinomial, Stratified, Systematic, and
// Residual draw offspring counts in proportion to the weights, never choose a zero-weight particle,
// and the low-variance ones keep each count within one of its expectation. Each typed test runs
// over all four.

#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Core/Weights.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include "user_policies.h"

namespace resamplers = Emrys::resamplers;

namespace
{

static_assert(Emrys::Resampler<resamplers::Systematic, double, std::mt19937_64>);
static_assert(Emrys::Resampler<resamplers::Residual<resamplers::Systematic>, float, std::mt19937>);

// The resamplers that need storage sized by the counts take it as scratch; the others don't.
static_assert(Emrys::ResamplerWithScratch<resamplers::Multinomial, double, std::mt19937_64>);
static_assert(Emrys::ResamplerWithScratch<resamplers::Residual<>, double, std::mt19937_64>);
static_assert(Emrys::ResamplerWithScratch<resamplers::Residual<resamplers::Systematic>, float, std::mt19937>);
static_assert(!Emrys::ResamplerWithScratch<resamplers::Stratified, double, std::mt19937_64>);
static_assert(!Emrys::ResamplerWithScratch<resamplers::Systematic, double, std::mt19937_64>);

// Unnormalized on purpose; index 2 and the last index have zero weight.
std::vector<double> const weights{0.8, 0.4, 0.0, 1.6, 0.2, 0.0};

constexpr double totalWeight = 3.0;

template <typename R>
std::vector<std::size_t>
countOffspring(R const & resampler, std::size_t m, std::uint64_t seed)
{
    auto rng = std::mt19937_64{seed};
    auto ancestors = std::vector<Eigen::Index>(m);
    resampler(std::span<double const>{weights}, std::span<Eigen::Index>{ancestors}, rng);
    auto counts = std::vector<std::size_t>(weights.size(), 0);
    for (auto const a : ancestors)
    {
        EXPECT_GE(a, 0);
        EXPECT_LT(a, Eigen::Index(weights.size()));
        if (a >= 0 && a < Eigen::Index(weights.size()))
        {
            ++counts[std::size_t(a)];
        }
    }
    return counts;
}

template <typename R>
class ResamplerTest : public ::testing::Test
{};

using Resamplers = ::testing::Types<
    resamplers::Multinomial,
    resamplers::Stratified,
    resamplers::Systematic,
    resamplers::Residual<>,
    resamplers::Residual<resamplers::Systematic>>;

}  // namespace

TYPED_TEST_SUITE(ResamplerTest, Resamplers);

// A resampler that takes scratch draws, from the same seed, the ancestors its three-argument form
// draws: the scratch holds what that form allocates and nothing else.
TYPED_TEST(ResamplerTest, ScratchFormDrawsTheSameAncestors)
{
    if constexpr (Emrys::ResamplerWithScratch<TypeParam, double, std::mt19937_64>)
    {
        constexpr std::size_t m = 7;
        auto const resampler = TypeParam{};
        auto plain = std::vector<Eigen::Index>(m);
        auto withScratch = std::vector<Eigen::Index>(m);
        auto scratch = std::vector<double>(weights.size() + m);
        auto rng = std::mt19937_64{11};
        resampler(std::span<double const>{weights}, std::span<Eigen::Index>{plain}, rng);
        rng = std::mt19937_64{11};
        resampler(
            std::span<double const>{weights},
            std::span<Eigen::Index>{withScratch},
            rng,
            std::span<double>{scratch}
        );
        EXPECT_EQ(plain, withScratch);
    }
}

// Over fifty seeds, no ancestor index has zero weight.
TYPED_TEST(ResamplerTest, NeverChoosesZeroWeight)
{
    for (std::uint64_t seed = 0; seed < 50; ++seed)
    {
        auto const counts = countOffspring(TypeParam{}, 7, seed);
        EXPECT_EQ(counts[2], 0u);
        EXPECT_EQ(counts[5], 0u);
    }
}

// With 30,000 draws, each index's offspring count is within five binomial standard deviations of
// its expectation m p, plus one for rounding.
TYPED_TEST(ResamplerTest, OffspringProportionalToWeight)
{
    constexpr std::size_t m = 30000;
    auto const counts = countOffspring(TypeParam{}, m, 1);
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        double const p = weights[i] / totalWeight;
        double const sigma = std::sqrt(double(m) * p * (1 - p));
        EXPECT_NEAR(double(counts[i]), double(m) * p, 5 * sigma + 1) << "index " << i;
    }
}

// With one nonzero weight, every ancestor is that index.
TYPED_TEST(ResamplerTest, SingleNonzeroWeight)
{
    auto const oneHot = std::vector<double>{0.0, 0.0, 2.0};
    auto rng = std::mt19937_64{3};
    auto ancestors = std::vector<Eigen::Index>(5);
    TypeParam{}(std::span<double const>{oneHot}, std::span<Eigen::Index>{ancestors}, rng);
    for (auto const a : ancestors)
    {
        EXPECT_EQ(a, 2);
    }
}

// Systematic resampling gives each index either the floor or the ceiling of its expected count, for
// every seed.
TEST(SystematicResampler, CountsAreFloorOrCeil)
{
    constexpr std::size_t m = 17;
    for (std::uint64_t seed = 0; seed < 50; ++seed)
    {
        auto const counts = countOffspring(resamplers::systematic(), m, seed);
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            double const expected = double(m) * weights[i] / totalWeight;
            EXPECT_GE(double(counts[i]), std::floor(expected)) << "index " << i;
            EXPECT_LE(double(counts[i]), std::ceil(expected)) << "index " << i;
        }
    }
}

// Residual resampling deterministically assigns the floor of each expected count before drawing the
// remainder, so no count falls below it.
TEST(ResidualResampler, CountsAtLeastFloor)
{
    constexpr std::size_t m = 17;
    for (std::uint64_t seed = 0; seed < 50; ++seed)
    {
        auto const counts = countOffspring(resamplers::residual(), m, seed);
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            EXPECT_GE(double(counts[i]), std::floor(double(m) * weights[i] / totalWeight)) << "index " << i;
        }
    }
}

// Under a parallel policy the inversion caps each ancestor at the last particle of positive
// weight, found chunk by chunk; with the weight all in the first of three chunks, the later
// chunks find none, and no ancestor lands past the first.
TEST(MultinomialResampler, UnderAPolicyNeverChoosesAZeroWeight)
{
    auto spread = std::vector<double>(std::size_t(3 * Emrys::internal::chunkSize), 0.0);
    for (std::size_t i = 0; i < 10; ++i)
    {
        spread[i] = double(i + 1);
    }
    auto ancestors = std::vector<Eigen::Index>(500);
    auto rng = std::mt19937_64{3};
    resamplers::multinomial()(
        EmrysTest::ThreadsPolicy{},
        std::span<double const>{spread},
        std::span<Eigen::Index>{ancestors},
        rng
    );
    for (Eigen::Index const a : ancestors)
    {
        EXPECT_GE(a, 0);
        EXPECT_LT(a, 10);
    }
}

// A single draw by its weights takes the last index of positive weight when the draw falls past
// their sum, as rounding can leave a normalized sum short of one; here the weights sum to a half
// and the draw is the largest the generator makes.
TEST(DrawIndex, PastTheSumTakesTheLastPositiveWeight)
{
    struct Largest
    {
        using result_type = std::uint64_t;

        static constexpr result_type
        min()
        {
            return 0;
        }

        static constexpr result_type
        max()
        {
            return ~result_type{0};
        }

        result_type
        operator()()
        {
            return max();
        }
    };

    std::vector<double> const halves{0.25, 0.25, 0.0};
    auto rng = Largest{};
    EXPECT_EQ(
        Emrys::internal::drawIndex<double>(
            3,
            [&](Eigen::Index i) {
                return halves[std::size_t(i)];
            },
            rng
        ),
        1
    );
}
