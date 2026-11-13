// The upkeep function objects of Distributions/Upkeep.h (step 7, [upkeep]): Resample,
// WhenEss, Roughen, and WhenSize applied to a ParticleSet or a GaussianMixtureParams, including across
// a periodic seam and with a caller's own action, comparator, or predicate (When); FloorModes on a hybrid's mode
// probabilities, on a categorical's, and on each component's of a mixture of hybrids (step 17c,
// [autonomous-multiple-model]); PruneBelow, PruneTail, and PruneToSize, and the methods
// they call, on each state with weighted elements: ParticleSet, ParticleGaussianParams, MixtureParams, and
// GridGaussianParams, which choose what to remove by the same rules (Core/Weights.h); RepairCovariance on a
// Gaussian, a mixture, and a hybrid ([covariance-repair]); and Sequence and None, the upkeep a filter holds.

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::CategoricalParams;
using Emrys::GaussianError;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::HybridGaussianParams;
using Emrys::Linear;
using Emrys::Named;
using Emrys::ParticleSet;
using Emrys::RadiansPi;
using Emrys::WeightError;

namespace upkeep = Emrys::upkeep;

namespace
{

constexpr double pi = std::numbers::pi;

using Matrix1 = Eigen::Matrix<double, 1, 1>;

struct First
{};

struct Second
{};

struct Third
{};

struct Three : Named<First, Second, Third>
{};

// A hybrid over three modes with probabilities p, the same Gaussian in each.
HybridGaussianParams<Linear<1>, Three>
makeHybrid(Eigen::Array3d const & p)
{
    return {CategoricalParams<Three>::fromWeights(Three{}, p), GaussianParams<Linear<1>>{Matrix1{0.0}, Matrix1{1.0}}};
}

void
expectModes(HybridGaussianParams<Linear<1>, Three> const & hybrid, Eigen::Array3d const & expected)
{
    for (Eigen::Index m = 0; m < 3; ++m)
    {
        EXPECT_NEAR(hybrid.modes().toDensity().value().probability(m), expected(m), 1e-15) << "mode " << m;
    }
}

template <typename G>
ParticleSet<Linear<2>>
drawParticles(Eigen::Index count, G & rng)
{
    auto const gaussian =
        GaussianParams<Linear<2>>{Eigen::Vector2d{1.0, -2.0}, Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}};
    return ParticleSet<Linear<2>>{Emrys::drawSamples(gaussian.toDensity().value(), count, rng)};
}

// A caller's action that records that it ran, with the value of a counter it captures and advances,
// as an action captures what it needs beside the state.
struct Record
{
    std::vector<std::string> * log;
    int * counter;
    std::string name;
    bool changes = true;

    template <typename State>
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State &) const
    {
        log->push_back(name + std::to_string(*counter));
        ++*counter;
        return changes;
    }
};

// A one-dimensional mixture with components (weight, mean, variance).
GaussianMixtureParams<Linear<1>>
makeMixture(std::vector<std::array<double, 3>> const & components)
{
    auto mixture = GaussianMixtureParams<Linear<1>>{1};
    for (auto const & [weight, mean, variance] : components)
    {
        mixture.add(std::log(weight), Matrix1{mean}, Matrix1{variance});
    }
    return mixture;
}

}  // namespace

// Resample reports a change, keeps the particle count, sets every log-weight to -log n, and records
// ancestors in which the heaviest particle appears most.
TEST(Upkeep, ResampleEqualizesTheWeightsAndKeepsTheSize)
{
    auto rng = std::mt19937_64{1};
    auto particles = drawParticles(100, rng);
    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(100, -5.0, 0.0));
    auto const result = upkeep::resample(rng)(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(particles.size(), 100);
    EXPECT_LT((particles.logWeights() + std::log(100.0)).abs().maxCoeff(), 1e-12);
    // The heaviest particle is drawn most often.
    auto const ancestors = particles.ancestors();
    EXPECT_GT(std::ranges::count(ancestors, 99), std::ranges::count(ancestors, 0));
}

// Resample given a size draws that many particles, growing or shrinking the set, with or without a
// policy and with any resampler, and sets every log-weight to -log of the new size. An integer
// names the size, never a resampler.
TEST(Upkeep, ResampleToASizeChangesTheCount)
{
    auto rng = std::mt19937_64{11};
    using Generator = std::mt19937_64;
    static_assert(std::same_as<decltype(upkeep::resample(rng, 250)), upkeep::Resample<Generator>>);
    static_assert(std::same_as<
                  decltype(upkeep::resample(rng, 40, Emrys::resamplers::multinomial())),
                  upkeep::Resample<Generator, Emrys::resamplers::Multinomial>>);
    EXPECT_FALSE(upkeep::resample(rng).size());
    EXPECT_EQ(upkeep::resample(rng, 250).size(), 250);

    auto particles = drawParticles(100, rng);
    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(100, -5.0, 0.0));
    auto const grown = upkeep::resample(rng, 250)(particles);
    ASSERT_TRUE(grown);
    EXPECT_TRUE(*grown);
    EXPECT_EQ(particles.size(), 250);
    EXPECT_EQ(particles.ancestors().size(), 250u);
    EXPECT_LT((particles.logWeights() + std::log(250.0)).abs().maxCoeff(), 1e-12);
    EXPECT_TRUE(std::ranges::all_of(particles.ancestors(), [](Eigen::Index a) {
        return a >= 0 && a < 100;
    }));

    auto const shrunk =
        upkeep::resample(Emrys::execution::sequential, rng, 40, Emrys::resamplers::multinomial())(particles);
    ASSERT_TRUE(shrunk);
    EXPECT_TRUE(*shrunk);
    EXPECT_EQ(particles.size(), 40);
    EXPECT_LT((particles.logWeights() + std::log(40.0)).abs().maxCoeff(), 1e-12);
    EXPECT_TRUE(std::ranges::all_of(particles.ancestors(), [](Eigen::Index a) {
        return a >= 0 && a < 250;
    }));

    // Under a trigger, as an adaptive count would run.
    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(40, -20.0, 0.0));
    auto const restored = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng, 100))(particles);
    ASSERT_TRUE(restored);
    EXPECT_TRUE(*restored);
    EXPECT_EQ(particles.size(), 100);
}

// WhenEss does nothing at equal weights, runs its actions in order on the state when the
// effective sample size drops below the fraction, reports a change if any action did, and
// passes a weight error through without running anything.
TEST(Upkeep, WhenEssRunsItsActionsOnlyBelowTheThreshold)
{
    auto rng = std::mt19937_64{2};
    auto particles = drawParticles(10, rng);
    auto log = std::vector<std::string>{};
    int counter = 0;
    auto const trigger =
        upkeep::whenEss(std::less{}, 0.5, Record{&log, &counter, "a"}, Record{&log, &counter, "b", false});

    // Equal weights: the effective sample size is the size.
    auto result = trigger(particles);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    EXPECT_TRUE(log.empty());

    // One particle with almost all the weight: the actions run in order.
    Eigen::ArrayXd logWeights = Eigen::ArrayXd::Constant(10, -10.0);
    logWeights(3) = 0.0;
    particles.setLogWeights(logWeights);
    result = trigger(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(log, (std::vector<std::string>{"a0", "b1"}));
    EXPECT_EQ(counter, 2);

    // Invalid weights are an error, and nothing runs.
    particles.setLogWeights(Eigen::ArrayXd::Constant(10, -std::numeric_limits<double>::infinity()));
    result = trigger(particles);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), WeightError::ZeroSum);
    EXPECT_EQ(log.size(), 2u);
}

// Resample and Roughen hold a reference to their generator, so a generator made in the call, which
// would dangle, does not compile, with a policy or without one; and the operations are called with
// the state alone.
template <typename Generator>
concept ResampleTakes = requires(Generator && rng) { upkeep::resample(std::forward<Generator>(rng)); };
template <typename Generator>
concept RoughenTakes =
    requires(Generator && rng) { upkeep::roughen(Emrys::execution::sequential, std::forward<Generator>(rng), 0.5); };

static_assert(ResampleTakes<std::mt19937_64 &> && !ResampleTakes<std::mt19937_64>);
static_assert(RoughenTakes<std::mt19937_64 &> && !RoughenTakes<std::mt19937_64>);
static_assert(!std::invocable<
              upkeep::WhenEss<std::less<>, double, upkeep::Resample<std::mt19937_64>> const &,
              ParticleSet<Linear<1>> &,
              std::mt19937_64 &>);

// upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng)) on a set whose effective sample size is under half restores
// it to the full count.
TEST(Upkeep, WhenEssResamplesAParticleSet)
{
    auto rng = std::mt19937_64{3};
    auto particles = drawParticles(200, rng);
    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(200, -8.0, 0.0));
    ASSERT_LT(particles.ess().value(), 100.0);
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    auto const result = resampling(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_NEAR(particles.ess().value(), 200.0, 1e-9);
}

// The actions that hold a number hold it in the type it is given, so a float filter's upkeep
// holds no double; a float fraction still tests a double set, in the set's scalar.
TEST(Upkeep, ActionsHoldTheScalarTheyAreGiven)
{
    using Generator = std::mt19937_64;
    auto rng = Generator{3};
    static_assert(std::is_same_v<
                  decltype(upkeep::whenEss(std::less{}, 0.5f, upkeep::resample(rng))),
                  upkeep::WhenEss<std::less<>, float, upkeep::Resample<Generator>>>);
    static_assert(std::is_same_v<
                  decltype(upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))),
                  upkeep::WhenEss<std::less<>, double, upkeep::Resample<Generator>>>);
    static_assert(std::is_same_v<decltype(upkeep::roughen(rng, 0.25f)), upkeep::Roughen<Generator, float>>);
    static_assert(std::is_same_v<decltype(upkeep::roughen(rng)), upkeep::Roughen<Generator, double>>);
    static_assert(std::is_same_v<decltype(upkeep::pruneBelow(1e-4f)), upkeep::PruneBelow<float>>);
    static_assert(std::is_same_v<decltype(upkeep::pruneTail(1e-3f)), upkeep::PruneTail<float>>);
    static_assert(std::is_same_v<decltype(upkeep::floorModes(1e-3f)), upkeep::FloorModes<float>>);

    auto particles = drawParticles(200, rng);
    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(200, -8.0, 0.0));
    auto const result = upkeep::whenEss(std::less{}, 0.5f, upkeep::resample(rng))(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
}

// upkeep::roughen(s) adds jitter whose sample covariance is s times the stored bandwidth, within
// five standard errors over 40,000 particles, without moving the mean or changing the stored
// bandwidth.
TEST(Upkeep, RoughenAddsTheScaledBandwidthToTheCovariance)
{
    constexpr Eigen::Index count = 40000;
    auto rng = std::mt19937_64{4};
    auto particles = drawParticles(count, rng);
    auto const bandwidth = Eigen::Matrix2d{{0.3, -0.1}, {-0.1, 0.2}};
    particles.setBandwidth(bandwidth);
    auto const before = particles.moments().value();

    auto const roughened = upkeep::roughen(rng, 0.5)(particles);
    ASSERT_TRUE(roughened);
    EXPECT_TRUE(*roughened);
    auto const after = particles.moments().value();
    Eigen::Matrix2d const added = 0.5 * bandwidth;
    Eigen::Matrix2d const jitter = (after.covarianceLower() - before.covarianceLower());
    Eigen::Vector2d const shift = after.mean() - before.mean();
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        EXPECT_NEAR(shift(i), 0.0, 5 * std::sqrt(added(i, i) / count));
        for (Eigen::Index j = 0; j < 2; ++j)
        {
            // The jitter's own sample covariance, plus its sample cross-covariance with the states.
            double const standardError = std::sqrt(
                (added(i, i) * added(j, j) + added(i, j) * added(i, j)) / count
                + 2 * (before.covarianceLower()(i, i) * added(j, j)) / count
            );
            EXPECT_NEAR(jitter(i, j), added(i, j), 5 * standardError) << i << ", " << j;
        }
    }
    EXPECT_EQ(particles.bandwidth().value(), bandwidth);
}

// Jittering particles that straddle pi keeps them in [-pi, pi), some having wrapped from just
// below pi to just above -pi.
TEST(Upkeep, RoughenWrapsAPeriodicCoordinate)
{
    struct Heading : RadiansPi<1>
    {};

    auto rng = std::mt19937_64{5};
    Eigen::RowVectorXd const straddling = Eigen::RowVectorXd::LinSpaced(2000, pi - 0.02, pi + 0.02);
    auto particles = ParticleSet<Heading>{Heading{}, straddling};
    Eigen::RowVectorXd const before = particles.states();
    ASSERT_TRUE(upkeep::roughen(rng)(particles));
    EXPECT_GE(particles.states().minCoeff(), -pi);
    EXPECT_LT(particles.states().maxCoeff(), pi);
    EXPECT_GT(((before.array() > 0) && (particles.states().array() < 0)).count(), 10);
}

// WhenSize runs a caller's action as it runs the library's: here one that merges the first two
// components until four remain, which does nothing at nine components and reduces to four once a
// tenth arrives.
TEST(Upkeep, WhenSizeRunsACallersActionInBatches)
{
    auto components = std::vector<std::array<double, 3>>{};
    for (int i = 0; i < 9; ++i)
    {
        components.push_back({1.0, double(i), 1.0});
    }
    auto mixture = makeMixture(components);
    auto const reduce = upkeep::whenSize(
        std::greater{},
        9,
        [](GaussianMixtureParams<Linear<1>> & state) -> std::expected<bool, WeightError> {
            while (state.size() > 4)
            {
                static_cast<void>(state.merge(0, 1));
            }
            return true;
        }
    );
    auto result = reduce(mixture);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    EXPECT_EQ(mixture.size(), 9);

    mixture.add(0.0, Matrix1{9.0}, Matrix1{1.0});
    result = reduce(mixture);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(mixture.size(), 4);
}

// upkeep::when() runs its actions, in order, when a caller's predicate of the state holds, and
// nothing otherwise. A predicate that returns std::expected stops the trigger with its error.
TEST(Upkeep, WhenRunsItsActionsWhenItsPredicateHolds)
{
    auto rng = std::mt19937_64{6};
    auto particles = drawParticles(10, rng);
    auto log = std::vector<std::string>{};
    int counter = 0;

    auto const more = [](auto const & state) {
        return state.size() > 5;
    };
    auto result = upkeep::when(more, Record{&log, &counter, "a"}, Record{&log, &counter, "b", false})(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(log, (std::vector<std::string>{"a0", "b1"}));

    auto const fewer = [](auto const & state) {
        return state.size() < 5;
    };
    result = upkeep::when(fewer, Record{&log, &counter, "c"})(particles);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    EXPECT_EQ(log.size(), 2u);

    auto const failing = [](auto const &) -> std::expected<bool, WeightError> {
        return std::unexpected{WeightError::NonFinite};
    };
    result = upkeep::when(failing, Record{&log, &counter, "d"})(particles);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), WeightError::NonFinite);
    EXPECT_EQ(log.size(), 2u);

    auto const resampled = upkeep::when(more, upkeep::resample(rng))(particles);
    ASSERT_TRUE(resampled);
    EXPECT_TRUE(*resampled);
}

// The comparator of WhenSize and WhenEss is the caller's: std::less_equal runs a size trigger at its
// limit, where std::less doesn't, and a caller's lambda, here one that runs while the effective sample
// size is above the threshold, runs WhenEss at equal weights, where std::less doesn't.
TEST(Upkeep, TriggersTakeTheCallersComparator)
{
    auto rng = std::mt19937_64{7};
    auto particles = drawParticles(10, rng);
    auto log = std::vector<std::string>{};
    int counter = 0;

    auto result = upkeep::whenSize(std::less{}, 10, Record{&log, &counter, "a"})(particles);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    result = upkeep::whenSize(std::less_equal{}, 10, Record{&log, &counter, "b"})(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);

    result = upkeep::whenEss(std::less{}, 0.5, Record{&log, &counter, "c"})(particles);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    auto const above = [](double ess, double threshold) {
        return ess > threshold;
    };
    result = upkeep::whenEss(above, 0.5, Record{&log, &counter, "d"})(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(log, (std::vector<std::string>{"b0", "d1"}));
}

// upkeep::sequence() runs every action, in order, on the state at every call; returns whether any
// changed the state; and stops at the first error, keeping what ran before it. It is a When whose
// predicate always holds.
TEST(Upkeep, SequenceRunsEveryActionInOrder)
{
    auto rng = std::mt19937_64{8};
    auto particles = drawParticles(10, rng);
    auto log = std::vector<std::string>{};
    int counter = 0;

    auto const both = upkeep::sequence(Record{&log, &counter, "a", false}, Record{&log, &counter, "b"});
    static_assert(
        std::is_same_v<std::remove_const_t<decltype(both)>, upkeep::When<Emrys::internal::Always, Record, Record>>
    );
    auto result = both(particles);
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    result = both(particles);
    ASSERT_TRUE(result);
    EXPECT_EQ(log, (std::vector<std::string>{"a0", "b1", "a2", "b3"}));

    result = upkeep::sequence(Record{&log, &counter, "c", false})(particles);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);

    auto const failing = [](auto &) -> std::expected<bool, WeightError> {
        return std::unexpected{WeightError::ZeroSum};
    };
    result = upkeep::sequence(Record{&log, &counter, "d"}, failing, Record{&log, &counter, "e"})(particles);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), WeightError::ZeroSum);
    EXPECT_EQ(log, (std::vector<std::string>{"a0", "b1", "a2", "b3", "c4", "d5"}));

    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(10, -10.0, 0.0));
    auto const resampling = upkeep::sequence(upkeep::resample(rng), upkeep::roughen(rng, 0.0));
    auto const resampled = resampling(particles);
    ASSERT_TRUE(resampled);
    EXPECT_TRUE(*resampled);
    EXPECT_NEAR(particles.ess().value(), double(particles.size()), 1e-9);
    EXPECT_EQ(particles.size(), 10);
}

// upkeep::none() takes any state, changes nothing, and returns false.
TEST(Upkeep, NoneChangesNothing)
{
    auto rng = std::mt19937_64{9};
    auto particles = drawParticles(10, rng);
    Eigen::MatrixXd const before = particles.states();
    static_assert(std::is_same_v<decltype(upkeep::none()), upkeep::None>);
    auto const result = upkeep::none()(particles);
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    EXPECT_TRUE(particles.states() == before);
    static_assert(Emrys::UpkeepOperation<upkeep::None, ParticleSet<Linear<2>>>);
}

namespace
{

// The three prunes apply to every state with weighted elements, and choose alike: each state below
// holds six elements tagged 0 to 5, with weights 1, 5, 2, 4, 3, and 1, summing to 16, so the first
// and last tie. Each prune keeps the elements in order, normalizes their weights, and returns the
// log of the fraction of the weight it removed.
constexpr std::array<double, 6> pruneWeights{1, 5, 2, 4, 3, 1};

struct Along : Linear<1>
{};

struct Offset : Linear<1>
{};

using Pair = Emrys::Composite<Along, Offset>;

// A particle set whose particle i sits at i.
struct ParticleStates
{
    using State = ParticleSet<Linear<1>>;

    static State
    make()
    {
        auto state = State{Eigen::RowVectorXd::LinSpaced(6, 0.0, 5.0)};
        state.setLogWeights(Eigen::Map<Eigen::Array<double, 6, 1> const>{pruneWeights.data()}.log());
        return state;
    }

    static double
    tag(State const & state, Eigen::Index i)
    {
        return state.state(i)(0);
    }

    static double
    logWeight(State const & state, Eigen::Index i)
    {
        return state.logWeight(i);
    }
};

// A Rao-Blackwellized particle state whose particle i sits at i, with a Gaussian of mean i over the
// rest, which a prune must keep with its particle.
struct RbParticleStates
{
    using State = Emrys::ParticleGaussianParams<Pair, Along>;

    static State
    make()
    {
        Eigen::RowVectorXd const at = Eigen::RowVectorXd::LinSpaced(6, 0.0, 5.0);
        auto particles = State::ParticlesType{at};
        particles.setLogWeights(Eigen::Map<Eigen::Array<double, 6, 1> const>{pruneWeights.data()}.log());
        return State{Pair{}, std::move(particles), at, Matrix1{1.0}};
    }

    static double
    tag(State const & state, Eigen::Index i)
    {
        EXPECT_EQ(state.conditional(i).mean()(0), state.particle(i)(0)) << "particle " << i;
        return state.particle(i)(0);
    }

    static double
    logWeight(State const & state, Eigen::Index i)
    {
        return state.logWeight(i);
    }
};

// A mixture whose component i has mean i.
struct MixtureStates
{
    using State = GaussianMixtureParams<Linear<1>>;

    static State
    make()
    {
        auto state = State{1};
        for (std::size_t i = 0; i < pruneWeights.size(); ++i)
        {
            state.add(std::log(pruneWeights[i]), Matrix1{double(i)}, Matrix1{1.0});
        }
        return state;
    }

    static double
    tag(State const & state, Eigen::Index i)
    {
        return state[i].component.mean()(0);
    }

    static double
    logWeight(State const & state, Eigen::Index i)
    {
        return state[i].logWeight;
    }
};

// A grid of six cells along one axis, cell i centered at i.
auto
makeGrid()
{
    return Emrys::regularGrid<Pair, Along>(Matrix1{0.0}, Matrix1{1.0}, {6});
}

// A grid state over makeGrid()'s cells.
struct GridStates
{
    using Grid = decltype(makeGrid());
    using State = Emrys::GridGaussianParams<Grid>;

    static State
    make()
    {
        auto const grid = makeGrid();
        auto state = State{grid, GaussianParams<typename Grid::RestSpace>{Matrix1{0.0}, Matrix1{1.0}}};
        Eigen::ArrayXd const weights = Eigen::Map<Eigen::Array<double, 6, 1> const>{pruneWeights.data()};
        EXPECT_TRUE(state.setWeights(CategoricalParams{grid.cells(), Eigen::ArrayXd{weights.log()}}));
        return state;
    }

    static double
    tag(State const & state, Eigen::Index i)
    {
        return state.center(i)(0);
    }

    static double
    logWeight(State const & state, Eigen::Index i)
    {
        return state.logWeight(i);
    }
};

// The log of the removed weight a prune returns: a grid's prune can't fail, and the others' must
// not here.
template <typename Removed>
double
removedBy(Removed const & removed)
{
    if constexpr (Emrys::internal::isExpected<Removed>)
    {
        EXPECT_TRUE(removed.has_value());
        return removed.value_or(std::numeric_limits<double>::quiet_NaN());
    }
    else
    {
        return removed;
    }
}

template <typename States>
class PruneRules : public testing::Test
{
protected:

    using State = typename States::State;

    // Expects the state to hold the elements tagged kept, in order, with their weights normalized,
    // and removed to be the log of the fraction of the 16 that the others held.
    static void
    expectPruned(State const & state, double removed, std::vector<int> const & kept)
    {
        double keptWeight = 0;
        for (int const tag : kept)
        {
            keptWeight += pruneWeights[std::size_t(tag)];
        }
        if (kept.size() == pruneWeights.size())
        {
            EXPECT_EQ(removed, -std::numeric_limits<double>::infinity());
        }
        else
        {
            EXPECT_NEAR(removed, std::log((16.0 - keptWeight) / 16.0), 1e-12);
        }
        ASSERT_EQ(state.size(), Eigen::Index(kept.size()));
        for (Eigen::Index i = 0; i < state.size(); ++i)
        {
            int const tag = kept[std::size_t(i)];
            EXPECT_EQ(States::tag(state, i), double(tag));
            EXPECT_NEAR(States::logWeight(state, i), std::log(pruneWeights[std::size_t(tag)] / keptWeight), 1e-12)
                << "element " << tag;
        }
    }
};

using PrunedStates = testing::Types<ParticleStates, RbParticleStates, MixtureStates, GridStates>;

}  // namespace

TYPED_TEST_SUITE(PruneRules, PrunedStates);

// pruneBelow(log 0.1) removes the two elements of probability 1/16; at 0.5, above every
// probability, it keeps only the most probable, of 5/16.
TYPED_TEST(PruneRules, PruneBelowRemovesEachElementBelowTheThreshold)
{
    auto some = TypeParam::make();
    this->expectPruned(some, removedBy(some.pruneBelow(std::log(0.1))), {1, 2, 3, 4});

    auto all = TypeParam::make();
    this->expectPruned(all, removedBy(all.pruneBelow(std::log(0.5))), {1});
}

// pruneTail(log f) removes the lightest while their total stays below f: at 0.3 the elements of 1,
// 1, and 2, a total of 0.25, since adding 3 would reach 0.44; at 0.1 one of the two of 1/16, the
// later, since both reach 0.125; and at 0.05 none.
TYPED_TEST(PruneRules, PruneTailRemovesTheLightestWhileTheirTotalStaysBelow)
{
    auto three = TypeParam::make();
    this->expectPruned(three, removedBy(three.pruneTail(std::log(0.3))), {1, 3, 4});

    auto tie = TypeParam::make();
    this->expectPruned(tie, removedBy(tie.pruneTail(std::log(0.1))), {0, 1, 2, 3, 4});

    auto none = TypeParam::make();
    this->expectPruned(none, removedBy(none.pruneTail(std::log(0.05))), {0, 1, 2, 3, 4, 5});
}

// pruneToSize(n) keeps the n heaviest: at five it removes the later of the two of 1/16, at three
// the lightest three, and at six nothing.
TYPED_TEST(PruneRules, PruneToSizeKeepsTheHeaviest)
{
    auto tie = TypeParam::make();
    this->expectPruned(tie, removedBy(tie.pruneToSize(5)), {0, 1, 2, 3, 4});

    auto three = TypeParam::make();
    this->expectPruned(three, removedBy(three.pruneToSize(3)), {1, 3, 4});

    auto none = TypeParam::make();
    this->expectPruned(none, removedBy(none.pruneToSize(6)), {0, 1, 2, 3, 4, 5});
}

// The upkeep actions call the prunes and say whether they removed anything, alone and under a size
// trigger, which runs pruneToSize(2) on more than three elements.
TYPED_TEST(PruneRules, TheActionsSayWhetherTheyPruned)
{
    auto state = TypeParam::make();
    auto const below = upkeep::pruneBelow(0.1)(state);
    ASSERT_TRUE(below);
    EXPECT_TRUE(*below);
    EXPECT_EQ(state.size(), 4);

    auto const none = upkeep::pruneTail(0.05)(state);
    ASSERT_TRUE(none);
    EXPECT_FALSE(*none);
    EXPECT_EQ(state.size(), 4);

    auto const bounded = upkeep::whenSize(std::greater{}, 3, upkeep::pruneToSize(2))(state);
    ASSERT_TRUE(bounded);
    EXPECT_TRUE(*bounded);
    this->expectPruned(state, std::log(7.0 / 16.0), {1, 3});
}

// The prunes of a mixture read only the weights, so a mixture of hybrids prunes too, and an empty
// mixture stops each action with its error.
TEST(Upkeep, PrunesApplyToAMixtureOfAnyComponent)
{
    auto hybrids = Emrys::MixtureParams<Emrys::HybridGaussian<Linear<1>, Three>>{Linear<1>{}};
    hybrids.add(std::log(0.98), makeHybrid(Eigen::Array3d{0.5, 0.3, 0.2}));
    hybrids.add(std::log(0.02), makeHybrid(Eigen::Array3d{0.2, 0.3, 0.5}));
    auto const prunedHybrids = upkeep::pruneBelow(0.05)(hybrids);
    ASSERT_TRUE(prunedHybrids);
    EXPECT_TRUE(*prunedHybrids);
    ASSERT_EQ(hybrids.size(), 1);
    expectModes(hybrids[0].component, Eigen::Array3d{0.5, 0.3, 0.2});

    auto empty = GaussianMixtureParams<Linear<1>>{1};
    EXPECT_EQ(upkeep::pruneBelow(0.05)(empty).error(), WeightError::Empty);
    EXPECT_EQ(upkeep::pruneTail(0.05)(empty).error(), WeightError::Empty);
    EXPECT_EQ(upkeep::pruneToSize(1)(empty).error(), WeightError::Empty);
}

namespace
{

using Rng = std::mt19937_64;

// Whether each action applies to a state of type S: Resample, Roughen, the three prunes (all three or
// none), Subdivide, FloorModes, and RepairCovariance, in the order of the Upkeep module page's table.
template <typename S>
constexpr std::array<bool, 6> actionsApplying{
    Emrys::UpkeepOperation<upkeep::Resample<Rng>, S>,
    Emrys::UpkeepOperation<upkeep::Roughen<Rng>, S>,
    Emrys::UpkeepOperation<upkeep::PruneBelow<>, S> && Emrys::UpkeepOperation<upkeep::PruneTail<>, S> &&
        Emrys::UpkeepOperation<upkeep::PruneToSize, S>,
    Emrys::UpkeepOperation<upkeep::Subdivide, S>,
    Emrys::UpkeepOperation<upkeep::FloorModes<>, S>,
    Emrys::UpkeepOperation<upkeep::RepairCovariance<>, S>,
};

// Whether the upkeep operation of type U applies to the state of type S.
template <typename U, typename S>
constexpr bool applies = Emrys::UpkeepOperation<U, S>;

// Whether no prune applies to a state of type S, so that the column of the prunes reads false only when
// none of the three applies.
template <typename S>
constexpr bool prunesNone =
    !Emrys::UpkeepOperation<upkeep::PruneBelow<>, S> && !Emrys::UpkeepOperation<upkeep::PruneTail<>, S>
    && !Emrys::UpkeepOperation<upkeep::PruneToSize, S>;

}  // namespace

// Every action is constrained on the members of the state it calls, and a trigger on its predicate and its
// actions, so UpkeepOperation answers whether an operation applies to a state (decisions.md [upkeep]). The expected
// answers are the states' members: a set of particles resamples, roughens, and prunes; a Gaussian mixture
// prunes and repairs; a regular grid prunes and subdivides; a grid of points prunes only; a hybrid floors
// its modes and repairs; a mixture of hybrids prunes, floors each hybrid's modes, and repairs; a categorical
// distribution floors; and a Gaussian repairs only.
TEST(Upkeep, OperationsSayWhichStatesTheyApplyTo)
{
    using Particles = ParticleSet<Linear<1>>;
    using RbParticles = Emrys::ParticleGaussianParams<Pair, Along>;
    using Mixture = GaussianMixtureParams<Linear<1>>;
    using Grid = GridStates::State;
    using Points = Emrys::GridGaussianParams<Emrys::PointGrid<Pair, Along, Emrys::Finite<6>>>;
    using Hybrid = HybridGaussianParams<Linear<1>, Three>;
    using Hybrids = Emrys::MixtureParams<Emrys::HybridGaussian<Linear<1>, Three>>;
    using Modes = CategoricalParams<Three>;
    using Gaussian = GaussianParams<Linear<1>>;

    //                                       Res    Rgh    Prn    Sub    Flr    Rep
    static_assert(actionsApplying<Particles> == std::array{true, true, true, false, false, false});
    static_assert(actionsApplying<RbParticles> == std::array{true, true, true, false, false, false});
    static_assert(actionsApplying<Mixture> == std::array{false, false, true, false, false, true});
    static_assert(actionsApplying<Grid> == std::array{false, false, true, true, false, false});
    static_assert(actionsApplying<Points> == std::array{false, false, true, false, false, false});
    static_assert(actionsApplying<Hybrid> == std::array{false, false, false, false, true, true});
    static_assert(actionsApplying<Hybrids> == std::array{false, false, true, false, true, true});
    static_assert(actionsApplying<Modes> == std::array{false, false, false, false, true, false});
    static_assert(actionsApplying<Gaussian> == std::array{false, false, false, false, false, true});
    static_assert(prunesNone<Hybrid> && prunesNone<Modes> && prunesNone<Gaussian>);

    using Resample = upkeep::Resample<Rng>;
    using Roughen = upkeep::Roughen<Rng>;
    using PruneBelow = upkeep::PruneBelow<>;
    using Subdivide = upkeep::Subdivide;
    using Repair = upkeep::RepairCovariance<>;

    // A trigger applies where its predicate tests the state and each of its actions applies.
    using ResampleWhenEss = upkeep::WhenEss<std::less<>, double, Resample, Roughen>;
    using SubdivideWhenFew = upkeep::WhenSize<std::less<>, Subdivide>;
    static_assert(applies<ResampleWhenEss, Particles> && !applies<ResampleWhenEss, Grid>);
    static_assert(applies<SubdivideWhenFew, Grid> && !applies<SubdivideWhenFew, Particles>);
    static_assert(!applies<SubdivideWhenFew, Points>);
    static_assert(applies<upkeep::Sequence<PruneBelow, SubdivideWhenFew>, Grid>);
    static_assert(!applies<upkeep::Sequence<PruneBelow, SubdivideWhenFew>, Particles>);
    static_assert(!applies<upkeep::Sequence<PruneBelow, Repair>, Particles>);
    static_assert(applies<upkeep::Sequence<PruneBelow, Repair>, Mixture>);
    static_assert(applies<upkeep::None, Points> && applies<upkeep::Sequence<>, Modes>);
}

// A mode below the floor, here one of probability zero, is raised to it, and the others share the
// rest in proportion: by hand, 0.9 and 0.1 scaled by 0.99.
TEST(Upkeep, FloorModesRaisesAModeAndRescalesTheOthers)
{
    auto hybrid = makeHybrid(Eigen::Array3d{0.9, 0.1, 0.0});
    auto const changed = upkeep::floorModes(0.01)(hybrid);
    ASSERT_TRUE(changed);
    EXPECT_TRUE(*changed);
    expectModes(hybrid, Eigen::Array3d{0.891, 0.099, 0.01});
}

// A mode that the rescaling takes below the floor is floored too: by hand, flooring 0.05 alone
// scales 0.1 by 0.9 / 0.95 to 0.0947, so it is floored as well and 0.85 takes the remaining 0.8.
TEST(Upkeep, FloorModesFloorsAModeTheRescalingTakesBelow)
{
    auto hybrid = makeHybrid(Eigen::Array3d{0.85, 0.1, 0.05});
    ASSERT_TRUE(upkeep::floorModes(0.1)(hybrid));
    expectModes(hybrid, Eigen::Array3d{0.8, 0.1, 0.1});
}

// Modes at or above the floor are left exactly as they were, and the action says so.
TEST(Upkeep, FloorModesLeavesModesAboveTheFloor)
{
    auto hybrid = makeHybrid(Eigen::Array3d{0.5, 0.3, 0.2});
    auto const before = hybrid.modes().toDensity().value().logProbabilities();
    auto const changed = upkeep::floorModes(0.15)(hybrid);
    ASSERT_TRUE(changed);
    EXPECT_FALSE(*changed);
    EXPECT_TRUE((hybrid.modes().toDensity().value().logProbabilities() == before).all());
}

// A CategoricalParams, the state of a hidden Markov filter, is floored as a hybrid's modes are: the
// cascade of FloorModesFloorsAModeTheRescalingTakesBelow, and nothing below the floor left as it was.
TEST(Upkeep, FloorModesFloorsACategorical)
{
    auto categorical = CategoricalParams<Three>::fromWeights(0.85, 0.1, 0.05);
    auto const changed = upkeep::floorModes(0.1)(categorical);
    ASSERT_TRUE(changed);
    EXPECT_TRUE(*changed);
    Eigen::Array3d const expected = Eigen::Array3d{0.8, 0.1, 0.1};
    for (Eigen::Index m = 0; m < 3; ++m)
    {
        EXPECT_NEAR(categorical.toDensity().value().probability(m), expected(m), 1e-15) << "value " << m;
    }

    auto above = CategoricalParams<Three>::fromWeights(0.5, 0.3, 0.2);
    Eigen::Array3d const before = above.logWeights();
    auto const unchanged = upkeep::floorModes(0.15)(above);
    ASSERT_TRUE(unchanged);
    EXPECT_FALSE(*unchanged);
    EXPECT_TRUE((above.logWeights() == before).all());

    auto zero = CategoricalParams<Three>::fromWeights(0.0, 0.0, 0.0);
    EXPECT_EQ(upkeep::floorModes(0.1)(zero).error(), WeightError::ZeroSum);
}

// A mixture of hybrids is floored component by component, as each hybrid is alone, and its
// components' weights are left as they were; a mixture with no mode below the floor is untouched,
// and one whose component's mode weights fail stops with their error before any component changes.
TEST(Upkeep, FloorModesFloorsEachComponentOfAMixtureOfHybrids)
{
    auto hybrids = Emrys::MixtureParams<Emrys::HybridGaussian<Linear<1>, Three>>{Linear<1>{}};
    hybrids.add(std::log(0.7), makeHybrid(Eigen::Array3d{0.9, 0.1, 0.0}));
    hybrids.add(std::log(0.3), makeHybrid(Eigen::Array3d{1.0, 0.0, 0.0}));
    auto const changed = upkeep::floorModes(0.01)(hybrids);
    ASSERT_TRUE(changed);
    EXPECT_TRUE(*changed);
    expectModes(hybrids[0].component, Eigen::Array3d{0.891, 0.099, 0.01});
    expectModes(hybrids[1].component, Eigen::Array3d{0.98, 0.01, 0.01});
    EXPECT_EQ(hybrids[0].logWeight, std::log(0.7));
    EXPECT_EQ(hybrids[1].logWeight, std::log(0.3));

    auto const again = upkeep::floorModes(0.01)(hybrids);
    ASSERT_TRUE(again);
    EXPECT_FALSE(*again);

    auto failing = Emrys::MixtureParams<Emrys::HybridGaussian<Linear<1>, Three>>{Linear<1>{}};
    failing.add(std::log(0.5), makeHybrid(Eigen::Array3d{0.9, 0.1, 0.0}));
    failing.add(std::log(0.5), makeHybrid(Eigen::Array3d{0.0, 0.0, 0.0}));
    EXPECT_EQ(upkeep::floorModes(0.01)(failing).error(), WeightError::ZeroSum);
    expectModes(failing[0].component, Eigen::Array3d{0.9, 0.1, 0.0});
}

// upkeep::roughen(rng, 0) turns roughening off: it returns false, leaves the states as they were, and
// draws nothing, with and without a policy, so the generator's next value is the one it would have
// been without it; with weights that fail it returns their error. It returns before any loop, so
// no policy takes another path.
TEST(Upkeep, RoughenByZeroChangesNothing)
{
    auto rng = std::mt19937_64{5};
    auto particles = drawParticles(100, rng);
    // A copy: states() is a map over the set's storage, which Roughen writes.
    Eigen::Matrix2Xd const states = particles.states();
    auto untouched = rng;

    auto const sequential = upkeep::roughen(rng, 0)(particles);
    ASSERT_TRUE(sequential);
    EXPECT_FALSE(*sequential);
    auto const withPolicy = upkeep::roughen(Emrys::execution::sequential, rng, 0.0f)(particles);
    ASSERT_TRUE(withPolicy);
    EXPECT_FALSE(*withPolicy);
    EXPECT_EQ(particles.states(), states);
    EXPECT_EQ(rng(), untouched());

    particles.scaleWeights(-std::numeric_limits<double>::infinity());
    EXPECT_EQ(upkeep::roughen(rng, 0)(particles).error(), WeightError::ZeroSum);
}

// Weights that fail to define a distribution stop every action that reads them, with their
// error, and the actions after a failure don't run: a size trigger runs Resample, which fails, and
// skips Roughen; WhenEss and Roughen made with a policy report it too, and FloorModes. A size
// trigger leaves a state within its limit alone.
TEST(Upkeep, ActionsReportWeightsThatFail)
{
    auto rng = std::mt19937_64{107};
    auto particles = drawParticles(5, rng);
    particles.scaleWeights(-std::numeric_limits<double>::infinity());
    auto const before = particles.states();
    auto const trigger = upkeep::whenSize(std::greater{}, 2, upkeep::resample(rng), upkeep::roughen(rng, 0.5));
    EXPECT_EQ(trigger(particles).error(), WeightError::ZeroSum);
    EXPECT_EQ(particles.states(), before);
    auto const whenEss = upkeep::whenEss(Emrys::execution::sequential, std::less{}, 0.5, upkeep::resample(rng));
    EXPECT_EQ(whenEss(particles).error(), WeightError::ZeroSum);
    EXPECT_EQ(upkeep::roughen(Emrys::execution::sequential, rng, 0.5)(particles).error(), WeightError::ZeroSum);
    auto const untouched = upkeep::whenSize(std::greater{}, 10, upkeep::resample(rng))(particles);
    ASSERT_TRUE(untouched);
    EXPECT_FALSE(*untouched);

    auto hybrid = makeHybrid(Eigen::Array3d{0.0, 0.0, 0.0});
    EXPECT_EQ(upkeep::floorModes(0.1)(hybrid).error(), WeightError::ZeroSum);
}

// RepairCovariance ridges a covariance that is not positive definite, as factorize(roots::ridge()) does: the
// repaired covariance is the indefinite one plus a ridge on the diagonal, and its density can be read. Called
// again, now on a positive definite covariance, it changes nothing and says so.
TEST(Upkeep, RepairCovarianceRidgesACovarianceThatIsNotPositiveDefinite)
{
    auto const indefinite = Eigen::Matrix2d{{0.5, 0.0}, {0.0, -0.5}};
    auto gaussian = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), indefinite};
    ASSERT_EQ(gaussian.toDensity().error(), GaussianError::NotPositiveDefinite);

    auto const repair = upkeep::repairCovariance();
    auto const repaired = repair(gaussian);
    ASSERT_TRUE(repaired);
    EXPECT_TRUE(*repaired);
    Eigen::Matrix2d const ridge = gaussian.covarianceLower() - indefinite;
    EXPECT_GT(ridge(1, 1), 0.5);
    EXPECT_DOUBLE_EQ(ridge(0, 0), ridge(1, 1));
    EXPECT_EQ(ridge(1, 0), 0.0);
    EXPECT_TRUE(gaussian.toDensity().has_value());

    Eigen::Matrix2d const kept = gaussian.covarianceLower();
    auto const again = repair(gaussian);
    ASSERT_TRUE(again);
    EXPECT_FALSE(*again);
    EXPECT_EQ(gaussian.covarianceLower(), kept);
}

// RepairCovariance returns what it cannot repair and leaves the covariance as it was: a zero covariance, which the
// ridge leaves zero and a ridge with an absolute floor repairs, and one holding NaN, which fails to factorize before
// any repair is tried.
TEST(Upkeep, RepairCovarianceReportsWhatItCannotRepair)
{
    auto zero = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    zero.setCovariance(Eigen::Matrix2d::Zero());
    EXPECT_EQ(upkeep::repairCovariance()(zero).error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(zero.covarianceLower().isZero(0));
    auto const floored = upkeep::repairCovariance(Emrys::roots::ridge<1.0, 1e-12>())(zero);
    ASSERT_TRUE(floored);
    EXPECT_TRUE(*floored);
    EXPECT_TRUE(zero.toDensity().has_value());

    Eigen::Matrix2d nan = -Eigen::Matrix2d::Identity();
    nan(1, 0) = std::numeric_limits<double>::quiet_NaN();
    auto broken = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    broken.setCovariance(nan);
    EXPECT_EQ(upkeep::repairCovariance()(broken).error(), GaussianError::NonFinite);
    EXPECT_EQ(broken.covarianceLower()(0, 0), -1.0);
    EXPECT_TRUE(std::isnan(broken.covarianceLower()(1, 0)));
}

// On a mixture RepairCovariance repairs the failing component and leaves the others, and leaves a mixture with a
// component it can't repair as it was, returning the component's error without its index; weights that fail stop
// it with their error; and on a hybrid it repairs each mode's covariance, or none.
TEST(Upkeep, RepairCovarianceRepairsMixturesAndHybrids)
{
    auto mixture = makeMixture({{0.5, 0.0, 1.0}, {0.5, 1.0, -1.0}});
    auto const repaired = upkeep::repairCovariance()(mixture);
    ASSERT_TRUE(repaired);
    EXPECT_TRUE(*repaired);
    EXPECT_EQ(mixture[0].component.covarianceLower()(0, 0), 1.0);
    EXPECT_GT(mixture[1].component.covarianceLower()(0, 0), 0.0);
    EXPECT_TRUE(mixture.toDensity().has_value());

    auto unrepairable = makeMixture({{0.5, 0.0, -1.0}, {0.5, 1.0, 1.0}});
    unrepairable.setCovariance(1, Matrix1::Zero());
    EXPECT_EQ(upkeep::repairCovariance()(unrepairable).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(unrepairable[0].component.covarianceLower()(0, 0), -1.0);

    auto empty = GaussianMixtureParams<Linear<1>>{1};
    EXPECT_EQ(upkeep::repairCovariance()(empty).error(), WeightError::Empty);

    auto hybrid = HybridGaussianParams<Linear<1>, Three>{
        CategoricalParams<Three>::fromWeights(Three{}, Eigen::Array3d{0.5, 0.3, 0.2}),
        GaussianParams<Linear<1>>{Matrix1{0.0}, Matrix1{-1.0}}
    };
    auto const repairedModes = upkeep::repairCovariance()(hybrid);
    ASSERT_TRUE(repairedModes);
    EXPECT_TRUE(*repairedModes);
    EXPECT_TRUE(hybrid.toDensity().has_value());

    // A hybrid, as a mixture, repairs every mode or none (decisions.md [covariance-repair]): the first two modes'
    // covariances could be repaired, but the last's is zero, so the first two keep their own.
    auto zero = GaussianParams<Linear<1>>{Matrix1{0.0}, Matrix1{1.0}};
    zero.setCovariance(Matrix1::Zero());
    auto unrepairableModes = HybridGaussianParams<Linear<1>, Three>{
        CategoricalParams<Three>::fromWeights(Three{}, Eigen::Array3d{0.5, 0.3, 0.2}),
        GaussianParams<Linear<1>>{Matrix1{0.0}, Matrix1{-1.0}}
    };
    unrepairableModes.setConditional(2, zero);
    EXPECT_EQ(upkeep::repairCovariance()(unrepairableModes).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(unrepairableModes.conditional(0).covarianceLower()(0, 0), -1.0);
    EXPECT_EQ(unrepairableModes.conditional(1).covarianceLower()(0, 0), -1.0);
}
