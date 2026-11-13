// The ParticleSet (step 7; [upkeep], [allocation], [sizes-and-dimensions]): moments and effective sample size
// against references, resampling that keeps the states, normalizes, and is counted (resampleCount(), for step 14's
// ParticleHistory; [smoothing]), weight validation and failure that leaves the set unchanged, the editing operations
// the filters and upkeep use (remove, forEach, bulk updates, removeIf, append, resize), and the check a
// KernelDensity keeps, with the same death tests Gaussian has [view-stamp].

#include <EmrysFilter/Distributions/ParticleSet.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "user_policies.h"

using Emrys::Linear;
using Emrys::ParticleSet;
using Emrys::WeightError;

namespace resamplers = Emrys::resamplers;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

// Lets std::vector move rather than copy particle sets on reallocation.
static_assert(std::is_nothrow_move_constructible_v<ParticleSet<Linear<Eigen::Dynamic>>>);

static_assert(std::is_nothrow_move_constructible_v<ParticleSet<Linear<3>>>);

// Each config names a ParticleSet type and the state dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config
{
    using Type = ParticleSet<Linear<Size, MaxSize>, Scalar>;
    static constexpr Eigen::Index n = N;
};

using Configs = ::testing::Types<
    Config<double, 1, 1, 1>,
    Config<double, 3, 3, 3>,
    Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
    Config<double, Eigen::Dynamic, 6, 4>,
    Config<float, 2, 2, 2>>;

template <typename C>
class ParticleSetTest : public ::testing::Test
{
protected:

    using ParticleSetType = typename C::Type;
    using Scalar = typename ParticleSetType::Scalar;
    using SampleSetType = typename ParticleSetType::SampleSetType;
    using WeightArrayType = typename ParticleSetType::WeightArrayType;
    static constexpr Eigen::Index n = C::n;
    static constexpr Eigen::Index count = 7;

    static Scalar
    tolerance()
    {
        return std::is_same_v<Scalar, float> ? Scalar(1e-5) : Scalar(1e-12);
    }

    static SampleSetType
    makeStates()
    {
        return SampleSetType::NullaryExpr(n, count, [](Eigen::Index i, Eigen::Index j) {
            return Scalar(std::sin(1.3 * double(i) + 0.7 * double(j)));
        });
    }

    // Unnormalized weights 1..count.
    static WeightArrayType
    makeLogWeights()
    {
        return WeightArrayType::NullaryExpr(count, [](Eigen::Index j) {
            return Scalar(std::log(double(j + 1)));
        });
    }

    static ParticleSetType
    make()
    {
        return ParticleSetType{makeStates(), makeLogWeights()};
    }
};

}  // namespace

TYPED_TEST_SUITE(ParticleSetTest, Configs);

// moments() is the weighted sample mean and covariance computed with plain Eigen, the covariance is
// exactly symmetric, and logTotalWeight() is the log of the weights' sum.
TYPED_TEST(ParticleSetTest, MomentsMatchReference)
{
    auto const set = this->make();
    auto const moments = set.moments();
    ASSERT_TRUE(moments.has_value());

    Eigen::MatrixXd const x = set.states().template cast<double>();
    Eigen::VectorXd const w = set.logWeights().template cast<double>().exp().matrix();
    Eigen::VectorXd const wn = w / w.sum();
    Eigen::VectorXd const mean = x * wn;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(this->n, this->n);
    for (Eigen::Index j = 0; j < x.cols(); ++j)
    {
        cov += wn(j) * (x.col(j) - mean) * (x.col(j) - mean).transpose();
    }

    EXPECT_LT((moments->mean().template cast<double>() - mean).norm(), this->tolerance() * 10);
    EXPECT_LT((moments->covarianceLower().template cast<double>() - cov).norm(), this->tolerance() * 10);
    EXPECT_EQ(moments->covarianceLower(), moments->covarianceLower().transpose());
    EXPECT_NEAR(set.logTotalWeight().value(), std::log(28.0), this->tolerance() * 10);
}

// Equal weights give an effective sample size of the count, and one nonzero weight gives exactly
// one.
TYPED_TEST(ParticleSetTest, Ess)
{
    using Scalar = typename TestFixture::Scalar;
    auto set = this->make();
    set.setLogWeights(TestFixture::WeightArrayType::Constant(this->count, Scalar(3)));
    EXPECT_NEAR(set.ess().value(), Scalar(this->count), this->tolerance() * 10);

    set.setLogWeights(TestFixture::WeightArrayType::Constant(this->count, -Eigen::NumTraits<Scalar>::infinity()));
    set.setLogWeight(2, Scalar(0));
    EXPECT_EQ(set.ess().value(), Scalar(1));
}

// resample(rng, m) draws m particles whose states are the ancestors' states, with equal log-weights
// -log m and total weight one.
TYPED_TEST(ParticleSetTest, ResampleKeepsStatesAndNormalizes)
{
    using Scalar = typename TestFixture::Scalar;
    auto set = this->make();
    auto const original = set.states();
    auto rng = std::mt19937_64{42};

    ASSERT_TRUE(set.resample(rng, 11).has_value());
    EXPECT_EQ(set.size(), 11);
    ASSERT_EQ(set.ancestors().size(), 11u);
    for (Eigen::Index j = 0; j < set.size(); ++j)
    {
        EXPECT_EQ(set.state(j), original.col(set.ancestors()[std::size_t(j)]));
        EXPECT_EQ(set.logWeight(j), -std::log(Scalar(11)));
    }
    EXPECT_NEAR(set.logTotalWeight().value(), Scalar(0), this->tolerance() * 10);
}

// resampleCount() counts the successful resamples, not a failed one. A move carries it with the
// ancestors it describes ([smoothing]'s ParticleHistory reads the two together); a copy is a new set with
// neither, and copying over a resampled set clears its record.
TYPED_TEST(ParticleSetTest, ResampleCountCountsSuccessesAndMovesWithTheSet)
{
    using Scalar = typename TestFixture::Scalar;
    auto set = this->make();
    auto rng = std::mt19937_64{42};
    EXPECT_EQ(set.resampleCount(), 0);

    ASSERT_TRUE(set.resample(rng).has_value());
    ASSERT_TRUE(set.resample(rng, 9).has_value());
    EXPECT_EQ(set.resampleCount(), 2);
    auto const ancestors = std::vector<Eigen::Index>(set.ancestors().begin(), set.ancestors().end());

    set.setLogWeights(TestFixture::WeightArrayType::Constant(9, -Eigen::NumTraits<Scalar>::infinity()));
    EXPECT_FALSE(set.resample(rng).has_value());
    EXPECT_EQ(set.resampleCount(), 2);
    EXPECT_TRUE(std::ranges::equal(set.ancestors(), ancestors));

    auto const copy = set;
    EXPECT_EQ(copy.resampleCount(), 0);
    EXPECT_TRUE(copy.ancestors().empty());
    auto assigned = this->make();
    ASSERT_TRUE(assigned.resample(rng).has_value());
    ASSERT_EQ(assigned.resampleCount(), 1);
    assigned = copy;
    EXPECT_EQ(assigned.resampleCount(), 0);
    EXPECT_TRUE(assigned.ancestors().empty());

    auto moved = std::move(set);
    EXPECT_EQ(moved.resampleCount(), 2);
    EXPECT_TRUE(std::ranges::equal(moved.ancestors(), ancestors));
    auto movedAssigned = this->make();
    movedAssigned = std::move(moved);
    EXPECT_EQ(movedAssigned.resampleCount(), 2);
    EXPECT_TRUE(std::ranges::equal(movedAssigned.ancestors(), ancestors));
}

namespace
{

using Particles2 = ParticleSet<Linear<2>>;

// A 2-D set with the given weights; particle i is at (i, -i).
Particles2
makeWeighted(std::vector<double> const & weights)
{
    auto set = Particles2{};
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        set.add(std::log(weights[i]), Eigen::Vector2d{double(i), -double(i)});
    }
    return set;
}

// Expects the set's weights, in any order, to match the ascending list \a expected.
void
expectWeights(Particles2 const & set, std::vector<double> const & expected)
{
    auto weights = std::vector<double>{};
    for (Eigen::Index i = 0; i < set.size(); ++i)
    {
        weights.push_back(std::exp(set.logWeight(i)));
    }
    std::ranges::sort(weights);
    ASSERT_EQ(weights.size(), expected.size());
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        EXPECT_NEAR(weights[i], expected[i], 1e-12) << "index " << i;
    }
}

}  // namespace

// Constructing from states alone gives equal normalized weights.
TEST(ParticleSet, ConstructFromStatesHasEqualWeights)
{
    auto const set = Particles2{Eigen::Matrix<double, 2, 4>::Random()};
    EXPECT_EQ(set.size(), 4);
    expectWeights(set, {0.25, 0.25, 0.25, 0.25});
}

// normalizedWeights() divides by the total.
TEST(ParticleSet, NormalizedWeights)
{
    auto const set = makeWeighted({1.0, 3.0});
    auto const weights = set.normalizedWeights();
    ASSERT_TRUE(weights.has_value());
    EXPECT_NEAR((*weights)(0), 0.25, 1e-15);
    EXPECT_NEAR((*weights)(1), 0.75, 1e-15);
}

// An empty set reports Empty from every weight-dependent operation, all-zero weights ZeroSum, and a
// NaN or infinite weight NonFinite.
TEST(ParticleSet, WeightErrors)
{
    Particles2 const empty;
    EXPECT_EQ(empty.moments().error(), WeightError::Empty);
    EXPECT_EQ(empty.ess().error(), WeightError::Empty);
    EXPECT_EQ(empty.normalizedWeights().error(), WeightError::Empty);
    EXPECT_EQ(Emrys::silvermanBandwidth(empty).error(), WeightError::Empty);
    EXPECT_EQ(empty.kdeBandwidth().error(), WeightError::Empty);
    EXPECT_EQ(empty.toDensity().error(), WeightError::Empty);

    auto set = makeWeighted({1.0, 1.0});
    set.scaleWeights(-inf);
    EXPECT_EQ(set.moments().error(), WeightError::ZeroSum);

    set.setLogWeight(0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(set.moments().error(), WeightError::NonFinite);
    set.setLogWeight(0, inf);
    EXPECT_EQ(set.moments().error(), WeightError::NonFinite);
}

// A resample that fails on the weights leaves the states as they were.
TEST(ParticleSet, ResampleFailureLeavesSetUnchanged)
{
    auto set = makeWeighted({1.0, 2.0});
    set.scaleWeights(-inf);
    auto const before = set.states();
    auto rng = std::mt19937_64{1};
    EXPECT_EQ(set.resample(rng).error(), WeightError::ZeroSum);
    EXPECT_EQ(set.states(), before);
}

// resample() accepts a resampler; with one nonzero weight every particle becomes that one.
TEST(ParticleSet, ResampleWithChosenResampler)
{
    auto set = makeWeighted({0.0, 1.0, 0.0});
    auto rng = std::mt19937_64{1};
    ASSERT_TRUE(set.resample(rng, resamplers::stratified()).has_value());
    EXPECT_EQ(set.size(), 3);
    for (Eigen::Index j = 0; j < set.size(); ++j)
    {
        EXPECT_EQ(set.state(j), Eigen::Vector2d(1.0, -1.0));
    }
}

// remove(i) swaps the last particle into place i, so the order is not preserved.
TEST(ParticleSet, RemoveMovesLastIntoPlace)
{
    auto set = makeWeighted({1.0, 2.0, 3.0});
    set.remove(0);
    EXPECT_EQ(set.size(), 2);
    EXPECT_EQ(set.state(0), Eigen::Vector2d(2.0, -2.0));
    set.remove(1);
    EXPECT_EQ(set.size(), 1);
    EXPECT_EQ(set.state(0), Eigen::Vector2d(2.0, -2.0));
}

// forEach() lets the callback read each particle's weight and change its state, and reweight()
// also scales each weight by the log-factor it returns; forEach() leaves the weights alone.
TEST(ParticleSet, ForEachScalesWeightsAndModifiesParticles)
{
    auto set = makeWeighted({1.0, 2.0});
    set.reweight([](auto & p) {
        p.setState(2.0 * p.state());
        return std::log(3.0);
    });
    expectWeights(set, {3.0, 6.0});
    EXPECT_EQ(set.state(1), Eigen::Vector2d(2.0, -2.0));

    set.forEach([](auto & p) {
        EXPECT_DOUBLE_EQ(p.logWeight(), std::log(3.0 * double(p.index() + 1)));
        p.setState(Eigen::Vector2d::Zero());
    });
    expectWeights(set, {3.0, 6.0});
    EXPECT_EQ(set.states(), Eigen::Matrix2d::Zero());
}

// setStates() replaces every state, scaleWeights(array) scales each weight by its own factor, and
// scaleWeight(i) one weight.
TEST(ParticleSet, BulkStateAndWeightUpdates)
{
    auto set = makeWeighted({1.0, 2.0, 4.0});
    auto const a = Eigen::Matrix2d{{0.0, 1.0}, {1.0, 0.0}};
    set.setStates(a * set.states());
    EXPECT_EQ(set.state(2), Eigen::Vector2d(-2.0, 2.0));

    set.scaleWeights(Eigen::Array3d{std::log(4.0), std::log(2.0), 0.0});
    expectWeights(set, {4.0, 4.0, 4.0});
    set.scaleWeight(1, std::log(0.5));
    expectWeights(set, {2.0, 4.0, 4.0});
    EXPECT_DOUBLE_EQ(set.logWeight(1), std::log(2.0));
}

// removeIf(predicate) drops the particles the predicate selects by log-weight or state, and
// removeIf<true>() also returns the log of the weight fraction removed.
TEST(ParticleSet, RemoveIf)
{
    auto set = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0});
    auto const light = [](double logWeight, auto const &) {
        return logWeight < std::log(2.5);
    };

    auto copy = set;
    copy.removeIf(light);
    expectWeights(copy, {3.0, 4.0, 5.0});

    auto const removed = set.removeIf<true>(light);
    ASSERT_TRUE(removed.has_value());
    EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
    expectWeights(set, {3.0, 4.0, 5.0});

    set.removeIf([](double, auto const & state) {
        return state(0) == 4.0;
    });
    expectWeights(set, {4.0, 5.0});
}

// removeIf<true>() on all-zero weights fails with ZeroSum and removes nothing.
TEST(ParticleSet, RemoveIfReturningWeightFailsWithoutRemoving)
{
    auto set = makeWeighted({1.0, 1.0});
    set.scaleWeights(-inf);
    EXPECT_EQ(
        set.removeIf<true>(
               [](double, auto const &) {
                   return true;
               }
        ).error(),
        WeightError::ZeroSum
    );
    EXPECT_EQ(set.size(), 2);
}

// normalize() scales the weights to sum to one, and append() adds another set's particles with
// their weights.
TEST(ParticleSet, NormalizeAndAppend)
{
    auto set = makeWeighted({1.0, 3.0});
    ASSERT_TRUE(set.normalize().has_value());
    expectWeights(set, {0.25, 0.75});

    set.append(set);
    EXPECT_EQ(set.size(), 4);
    EXPECT_EQ(set.state(3), Eigen::Vector2d(1.0, -1.0));
    expectWeights(set, {0.25, 0.25, 0.75, 0.75});
}

// normalize() and resample() record the log total weight as zero, exactly, which copies and moves
// carry and ess() reads, and every change to the weights or the particles forgets it, so that
// logTotalWeight() sums the weights again (expected values by hand).
TEST(ParticleSet, RecordsTheLogTotalWeightItNormalizedTo)
{
    auto set = makeWeighted({1.0, 3.0});
    EXPECT_NEAR(set.logTotalWeight().value(), std::log(4.0), 1e-15);
    ASSERT_TRUE(set.normalize().has_value());
    EXPECT_EQ(set.logTotalWeight().value(), 0.0);
    EXPECT_NEAR(set.ess().value(), 1.0 / (0.25 * 0.25 + 0.75 * 0.75), 1e-12);

    auto const copy = set;
    EXPECT_EQ(copy.logTotalWeight().value(), 0.0);
    auto moved = Particles2{set};
    auto const target = std::move(moved);
    EXPECT_EQ(target.logTotalWeight().value(), 0.0);

    set.setLogWeight(0, std::log(0.5));
    EXPECT_NEAR(set.logTotalWeight().value(), std::log(1.25), 1e-15);
    ASSERT_TRUE(set.normalize().has_value());
    set.scaleWeights(1.0);
    EXPECT_NEAR(set.logTotalWeight().value(), 1.0, 1e-15);
    ASSERT_TRUE(set.normalize().has_value());
    set.reweight([](auto &) {
        return std::log(2.0);
    });
    EXPECT_NEAR(set.logTotalWeight().value(), std::log(2.0), 1e-15);
    ASSERT_TRUE(set.normalize().has_value());
    set.add(0.0, Eigen::Vector2d::Zero());
    EXPECT_NEAR(set.logTotalWeight().value(), std::log(2.0), 1e-15);

    auto rng = std::mt19937_64{3};
    ASSERT_TRUE(set.resample(rng).has_value());
    EXPECT_EQ(set.logTotalWeight().value(), 0.0);
}

// resizeParticles(n) changes a dynamic set's state dimension and keeps the count.
TEST(ParticleSet, ResizeParticlesDynamic)
{
    auto set = ParticleSet<Linear<Eigen::Dynamic>>{Eigen::MatrixXd::Zero(2, 3)};
    set.resizeParticles(4);
    EXPECT_EQ(set.valueDimension(), 4);
    EXPECT_EQ(set.size(), 3);
    set.setStates(Eigen::MatrixXd::Ones(4, 3));
    EXPECT_EQ(set.moments()->mean(), Eigen::Vector4d::Ones());
}

// A set with a capacity reserves for its states at its own size when its space has no bound,
// clear() empties it but keeps the space, moving a set onto itself changes nothing, and
// normalize() and a kernel density report weights that fail to define a distribution.
TEST(ParticleSet, CapacityClearingAndFailures)
{
    auto dynamic = ParticleSet<Linear<Eigen::Dynamic>>{Eigen::MatrixXd::Zero(2, 3)};
    dynamic.setCapacity(8);
    EXPECT_EQ(dynamic.size(), 3);
    dynamic.clear();
    EXPECT_EQ(dynamic.size(), 0);
    EXPECT_EQ(dynamic.valueDimension(), 2);

    auto set = makeWeighted({1.0, 3.0});
    auto & alias = set;
    set = std::move(alias);
    expectWeights(set, {1.0, 3.0});

    set.scaleWeights(-inf);
    EXPECT_EQ(set.normalize().error(), WeightError::ZeroSum);
    EXPECT_EQ(set.toDensity(Eigen::Matrix2d::Identity()).error(), WeightError::ZeroSum);
}

// An empty set takes its value dimension from a length, its space, or both, and reserves for a
// capacity; an allocator-extended move into another resource keeps the particles.
TEST(ParticleSet, ConstructsEmptyInEachForm)
{
    using Dynamic = Linear<Eigen::Dynamic>;
    using Pmr = std::pmr::polymorphic_allocator<>;
    auto first = std::pmr::monotonic_buffer_resource{};
    auto second = std::pmr::monotonic_buffer_resource{};
    auto const byLength = ParticleSet<Dynamic, double, Pmr>{3, Pmr{&first}};
    EXPECT_EQ(byLength.valueDimension(), 3);
    EXPECT_EQ(byLength.size(), 0);
    auto const inSpace = ParticleSet<Linear<2>, double, Pmr>{Linear<2>{}, Pmr{&first}};
    EXPECT_EQ(inSpace.valueDimension(), 2);
    EXPECT_EQ(inSpace.get_allocator(), Pmr{&first});
    auto const reserved = ParticleSet<Linear<2>, double, Pmr>{Linear<2>{}, Emrys::Capacity{8}, Pmr{&first}};
    EXPECT_EQ(reserved.capacity(), 8);
    EXPECT_EQ(reserved.size(), 0);
    auto const reservedFixed = ParticleSet<Linear<2>, double, Pmr>{Emrys::Capacity{8}, Pmr{&first}};
    EXPECT_EQ(reservedFixed.capacity(), 8);
    EXPECT_EQ(reservedFixed.valueDimension(), 2);
    EXPECT_EQ(reservedFixed.get_allocator(), Pmr{&first});
    static_assert(!std::is_constructible_v<ParticleSet<Dynamic>, Emrys::Capacity>);
    auto const both = ParticleSet<Dynamic, double, Pmr>{Dynamic{3}, 3, Pmr{&first}};
    EXPECT_EQ(both.valueDimension(), 3);

    auto source = ParticleSet<Linear<2>, double, Pmr>{Linear<2>{}, Eigen::Matrix2d::Identity(), Pmr{&first}};
    auto const moved = ParticleSet<Linear<2>, double, Pmr>{std::move(source), Pmr{&second}};
    EXPECT_EQ(moved.get_allocator(), Pmr{&second});
    EXPECT_EQ(moved.states(), Eigen::Matrix2d::Identity());
}

// removeIf() under a parallel policy compacts the particles kept, as Sequential does.
TEST(ParticleSet, RemoveIfUnderAPolicy)
{
    auto set = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0});
    set.removeIf(EmrysTest::ThreadsPolicy{}, [](double logWeight, auto const &) {
        return logWeight < std::log(2.5);
    });
    expectWeights(set, {3.0, 4.0, 5.0});
}

// Copies and moves of a KernelDensity read the same estimate, and neither they nor a finished
// forEach() keep the set from changing once they are no longer read [view-stamp].
TEST(ParticleSet, KernelDensitiesCopyAndMoveFreely)
{
    auto set = makeWeighted({1.0, 2.0});
    {
        auto kde = set.toDensity(Eigen::Matrix2d::Identity()).value();
        auto const copy = kde;
        auto const moved = std::move(kde);
        EXPECT_EQ(copy.logDensity(Eigen::Vector2d::Zero()), moved.logDensity(Eigen::Vector2d::Zero()));
    }
    set.scaleWeights(1.0);
    set.forEach([](auto &) {});
    set.scaleWeights(1.0);
    EXPECT_TRUE(set.toDensity(Eigen::Matrix2d::Identity()).has_value());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// Reading a KernelDensity after any mutation of its set dies [view-stamp].
TEST(ParticleSetDeathTest, ReadAfterChange)
{
    auto set = makeWeighted({1.0, 2.0});
    auto const kde = set.toDensity(Eigen::Matrix2d::Identity());
    auto rng = std::mt19937_64{1};
    Eigen::Vector2d const x = Eigen::Vector2d::Zero();
    std::string const message = "changed while a view";
    EXPECT_DEATH(
        {
            set.add(0.0, x);
            std::ignore = kde->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            set.remove(0);
            std::ignore = kde->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            set.setState(0, x);
            std::ignore = kde->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            set.scaleWeights(1.0);
            std::ignore = kde->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            set.forEach([](auto &) {});
            std::ignore = kde->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            std::ignore = set.resample(rng);
            std::ignore = kde->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            set.clear();
            std::ignore = kde->size();
        },
        message
    );
}

// Changing the set from inside its own forEach() or removeIf() callback dies when the callback
// returns.
TEST(ParticleSetDeathTest, ChangeSetInsideCallback)
{
    auto set = makeWeighted({1.0, 2.0});
    EXPECT_DEATH(
        set.forEach([&](auto &) {
            set.scaleWeights(1.0);
        }),
        "changed inside their own callback"
    );
    EXPECT_DEATH(
        set.removeIf([&](double, auto const &) {
            set.scaleWeights(1.0);
            return false;
        }),
        "changed inside their own callback"
    );
}

// Reading a KernelDensity of a set that was moved from dies.
TEST(ParticleSetDeathTest, ReadAfterMove)
{
    auto set = makeWeighted({1.0, 2.0});
    auto const kde = set.toDensity(Eigen::Matrix2d::Identity());
    EXPECT_DEATH(
        {
            auto stolen = std::move(set);
            std::ignore = kde->logDensity(Eigen::Vector2d::Zero());
        },
        "changed while a view"
    );
}

// A log-weight read past the last particle dies, as every other read by index does.
TEST(ParticleSetDeathTest, LogWeightOutOfRange)
{
    auto const set = makeWeighted({1.0, 2.0});
    EXPECT_DEATH(std::ignore = set.logWeight(2), "particle index out of range");
}

#endif
