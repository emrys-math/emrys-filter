// ModeSequence, overModeSequences(), GpbRecursion, and GpbFilter (roadmap step 17zn;
// [switching-models], [by-mode], [filters-and-recursions], [update-outcomes]): the space of the last modes is a
// product with the current mode fastest, a hybrid over the current mode spreads over the sequences, one mode is a
// Kalman filter, sequences of one, two, and three modes match the generalized pseudo-Bayesian estimator of that order
// written by hand as the source gives it, with noise by a mode axis read from each sequence's current mode, the filter
// is the exact posterior for as many steps as a sequence holds modes, sequences of two modes are the IMM on the chain
// of pairs of modes, a square-root form matches the covariance form, rejected sequences lose their weight, a
// maneuver is followed more closely with longer sequences on average, and the filter deduces its upkeep and prints.
// Headers: Spaces/ModeSequence.h, Distributions/Hybrid.h, Filters/ImmFilter.h (HybridUpdateSteps),
// Filters/GpbFilter.h.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/GpbFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/ModeSequence.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::byMode;
using Emrys::CategoricalParams;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::ContinuousTimeMarkovChain;
using Emrys::Current;
using Emrys::DiscreteProduct;
using Emrys::FixedStepModel;
using Emrys::Gaussian;
using Emrys::GaussianParams;
using Emrys::GpbFilter;
using Emrys::HybridGaussianParams;
using Emrys::HybridParams;
using Emrys::ImmFilter;
using Emrys::KalmanFilter;
using Emrys::Lagged;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::ModeSequence;
using Emrys::Named;
using Emrys::overModeSequences;
using Emrys::TagPath;
using Emrys::when;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

struct Cruise
{};

struct Maneuvering
{};

struct Maneuver : Named<Cruise, Maneuvering>
{};

struct Inlier
{};

struct Outlier
{};

struct Sensor : Named<Inlier, Outlier>
{};

using Mode = DiscreteProduct<Maneuver, Sensor>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double timeStep = 1.0;
constexpr double quiet = 0.01;
constexpr double agile = 4.0;
constexpr double outlierScale = 100;

Eigen::Matrix<double, 2, 4> const H{{1, 0, 0, 0}, {0, 1, 0, 0}};
Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

auto
makeModel(double density)
{
    return ConstantVelocity<Position, Velocity>{density};
}

// The motion: the quiet model while cruising and the agile one while maneuvering.
auto
makeMotion()
{
    return byMode<Maneuver>(when<Cruise>(makeModel(quiet)), when<Maneuvering>(makeModel(agile)));
}

GaussianParams<State>
makeGaussian(Vector4 const & mean, Vector4 const & variances)
{
    return {State{}, mean, Matrix4{variances.asDiagonal()}};
}

Matrix4
transitionMatrix(double dt)
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = dt * Eigen::Matrix2d::Identity();
    return f;
}

Matrix4
processNoise(double density, double dt)
{
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    auto q = Matrix4{};
    q << std::pow(dt, 3) / 3 * i2, dt * dt / 2 * i2, dt * dt / 2 * i2, dt * i2;
    return density * q;
}

// A position fix with the noise policy \a noise.
template <typename NoisePolicy>
auto
observeFix(Eigen::Vector2d const & z, NoisePolicy noise)
{
    return observe(
        methods::linearized(),
        Emrys::measurementModel(Eigen::Matrix2d::Identity(), Emrys::reads(Emrys::path<Position>), std::move(noise)),
        z
    );
}

// A fix with noise R.
auto
makeFix(Eigen::Vector2d const & z)
{
    return observeFix(z, additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R}));
}

// A fix with noise by the mode axis Sensor: R in the inlier modes and 100 R in the outlier ones.
auto
makeMarkovFix(Eigen::Vector2d const & z)
{
    return observeFix(
        z,
        additiveNoise(
            byMode<Sensor>(
                when<Outlier>(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), outlierScale * R}).weight(0.1),
                when<Inlier>(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R}).weight(0.9)
            )
        )
    );
}

// A Gaussian in plain Eigen.
struct HandGaussian
{
    Vector4 mean;
    Matrix4 cov;
};

// The Gaussian with the mean and covariance of the mixture of \a parts with the weights \a weights, which need not
// sum to one; of no weight at all, the first part.
HandGaussian
mergeByHand(std::vector<double> const & weights, std::vector<HandGaussian> const & parts)
{
    double total = 0;
    for (double const w : weights)
    {
        total += w;
    }
    if (!(total > 0))
    {
        return parts.front();
    }
    auto merged = HandGaussian{Vector4::Zero(), Matrix4::Zero()};
    for (std::size_t i = 0; i < parts.size(); ++i)
    {
        merged.mean += weights[i] / total * parts[i].mean;
    }
    for (std::size_t i = 0; i < parts.size(); ++i)
    {
        Vector4 const d = parts[i].mean - merged.mean;
        merged.cov += weights[i] / total * (parts[i].cov + d * d.transpose());
    }
    return merged;
}

// One Kalman step in plain Eigen: \a prior moved over \a dt with process noise density \a density and updated by the
// fix \a z with noise \a noiseScale R. Returns the posterior and the likelihood of z.
std::pair<HandGaussian, double>
stepByHand(HandGaussian const & prior, double density, double noiseScale, Eigen::Vector2d const & z, double dt)
{
    Matrix4 const f = transitionMatrix(dt);
    Vector4 const mean = f * prior.mean;
    Matrix4 const cov = f * prior.cov * f.transpose() + processNoise(density, dt);
    Eigen::Matrix2d const s = H * cov * H.transpose() + noiseScale * R;
    Eigen::Vector2d const innovation = z - H * mean;
    Eigen::Matrix<double, 4, 2> const k = cov * H.transpose() * s.inverse();
    double const likelihood =
        std::exp(-0.5 * innovation.dot(s.inverse() * innovation)) / (2 * pi * std::sqrt(s.determinant()));
    return {HandGaussian{mean + k * innovation, (Matrix4::Identity() - k * H) * cov}, likelihood};
}

// The generalized pseudo-Bayesian estimator of order `order` as the source gives it, in plain Eigen: after each
// update it keeps a Gaussian and a probability for each sequence of the last order - 1 modes, and for order 1 one
// Gaussian and the probabilities of the current mode. A step runs a Kalman filter from each kept Gaussian through
// each mode, and merges by moments the results that agree in their last order - 1 modes.
struct HandEstimator
{
    int order = 1;
    Eigen::MatrixXd transitions;      // the chain over a step
    std::vector<double> densities;    // each mode's process noise density
    std::vector<double> noiseScales;  // each mode's measurement noise, as a multiple of R
    std::vector<double> weights;      // a probability per kept sequence, the current mode fastest
    std::vector<HandGaussian> kept;   // a Gaussian per kept sequence
    std::vector<double> current;      // the probabilities of the current mode

    // Returns the likelihood of z given the fixes before it.
    double
    step(Eigen::Vector2d const & z, double dt)
    {
        std::size_t const modes = densities.size();
        std::size_t const count = kept.size();
        // What a kept sequence keeps of itself when a mode is appended: all but its oldest mode.
        std::size_t const tail = order >= 2 ? count / modes : 1;
        auto joint = std::vector<double>(count * modes);
        auto outputs = std::vector<HandGaussian>(count * modes);
        double evidence = 0;
        for (std::size_t r = 0; r < count; ++r)
        {
            for (std::size_t j = 0; j < modes; ++j)
            {
                double prior = 0;
                if (order == 1)
                {
                    for (std::size_t i = 0; i < modes; ++i)
                    {
                        prior += current[i] * transitions(Eigen::Index(i), Eigen::Index(j));
                    }
                }
                else
                {
                    prior = weights[r] * transitions(Eigen::Index(r % modes), Eigen::Index(j));
                }
                auto const [posterior, likelihood] = stepByHand(kept[r], densities[j], noiseScales[j], z, dt);
                outputs[r * modes + j] = posterior;
                joint[r * modes + j] = prior * likelihood;
                evidence += prior * likelihood;
            }
        }
        std::size_t const next = order >= 2 ? count : 1;
        auto nextWeights = std::vector<double>(next, 0.0);
        auto nextKept = std::vector<HandGaussian>(next);
        for (std::size_t target = 0; target < next; ++target)
        {
            auto mergedWeights = std::vector<double>{};
            auto merged = std::vector<HandGaussian>{};
            for (std::size_t r = 0; r < count; ++r)
            {
                for (std::size_t j = 0; j < modes; ++j)
                {
                    std::size_t const lands = order >= 2 ? (r % tail) * modes + j : 0;
                    if (lands == target)
                    {
                        mergedWeights.push_back(joint[r * modes + j]);
                        merged.push_back(outputs[r * modes + j]);
                        nextWeights[target] += joint[r * modes + j] / evidence;
                    }
                }
            }
            nextKept[target] = mergeByHand(mergedWeights, merged);
        }
        current.assign(modes, 0.0);
        for (std::size_t r = 0; r < count; ++r)
        {
            for (std::size_t j = 0; j < modes; ++j)
            {
                current[j] += joint[r * modes + j] / evidence;
            }
        }
        weights = nextWeights;
        kept = nextKept;
        return evidence;
    }
};

// A hybrid over the sequences of \a Order modes of \a ModeSpace whose components and weights all differ.
template <typename ModeSpace, std::size_t Order>
HybridGaussianParams<State, ModeSequence<ModeSpace, Order>>
makeDistinctState()
{
    using Sequences = ModeSequence<ModeSpace, Order>;
    Eigen::Index const count = Sequences{}.cardinality();
    auto conditionals = std::vector<GaussianParams<State>>{};
    Eigen::ArrayXd weights(count);
    for (Eigen::Index s = 0; s < count; ++s)
    {
        double const x = double(s);
        conditionals.push_back(makeGaussian(
            Vector4{0.3 * x, -0.2 * x, 1 + 0.1 * x, 0.05 * x},
            Vector4{4 + 0.5 * x, 3 + 0.25 * x, 1 + 0.1 * x, 2}
        ));
        weights(s) = 1 + std::fmod(3 * x, 5.0);
    }
    return {CategoricalParams<Sequences>::fromWeights(Sequences{}, weights), conditionals};
}

// What the source's estimator keeps of the hybrid over sequences \a state: for each sequence less its oldest mode,
// the probability and the merged Gaussian of the sequences that end in it, and the probabilities of the current
// mode, computed in plain Eigen from the state's components and probabilities.
template <typename StateType>
void
keepByHand(StateType const & state, HandEstimator & hand)
{
    auto const density = state.modes().toDensity().value();
    std::size_t const modes = hand.densities.size();
    std::size_t const count = std::size_t(state.size());
    std::size_t const kept = count / modes;
    hand.weights.assign(kept, 0.0);
    hand.kept.assign(kept, HandGaussian{});
    hand.current.assign(modes, 0.0);
    for (std::size_t rest = 0; rest < kept; ++rest)
    {
        auto weights = std::vector<double>{};
        auto parts = std::vector<HandGaussian>{};
        for (std::size_t oldest = 0; oldest < modes; ++oldest)
        {
            auto const s = Eigen::Index(oldest * kept + rest);
            weights.push_back(density.probability(s));
            parts.push_back(HandGaussian{state.conditional(s).mean(), state.conditional(s).covariance().value()});
            hand.weights[rest] += density.probability(s);
        }
        hand.kept[rest] = mergeByHand(weights, parts);
    }
    for (std::size_t s = 0; s < count; ++s)
    {
        hand.current[s % modes] += density.probability(Eigen::Index(s));
    }
    if (hand.order == 1)
    {
        hand.weights.assign(1, 1.0);
    }
}

// Runs the filter of sequences of \a Order modes of \a ModeSpace beside the source's estimator of that order over
// five fixes, from a state whose components all differ, and expects after every update the same likelihood, the same
// probabilities of the current mode, and, of what the estimator keeps, the same probabilities and Gaussians to 1e-10
// (the two merge in different orders and at different times, so they agree to rounding, not bitwise).
template <typename ModeSpace, std::size_t Order, typename Chain, typename Motion, typename MakeFix>
void
expectMatchesTheSource(
    Chain const & chain,
    Motion const & motion,
    MakeFix const & makeMeasurement,
    std::vector<double> const & densities,
    std::vector<double> const & noiseScales
)
{
    auto filter = GpbFilter{chain, motion, makeDistinctState<ModeSpace, Order>()};
    static_assert(std::remove_cvref_t<decltype(filter)>::Order == Order);
    auto hand = HandEstimator{int(Order), chain.transition(timeStep).matrix(), densities, noiseScales, {}, {}, {}};
    keepByHand(filter.state(), hand);

    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {2.0, 0.5}, {9.0, -4.0}, {3.9, 0.9}, {5.5, 1.5}};
    for (Eigen::Vector2d const & z : fixes)
    {
        filter.predict(timeStep);
        auto const update = filter.update(makeMeasurement(z));
        ASSERT_TRUE(update);
        ASSERT_TRUE(update->accepted);
        EXPECT_EQ(Eigen::Index(update->modes.size()), filter.state().size());
        double const evidence = hand.step(z, timeStep);
        EXPECT_NEAR(update->logLikelihood(), std::log(evidence), 1e-10);

        auto fromFilter = hand;
        keepByHand(filter.state(), fromFilter);
        for (std::size_t j = 0; j < hand.current.size(); ++j)
        {
            EXPECT_NEAR(fromFilter.current[j], hand.current[j], 1e-10) << "mode " << j;
        }
        ASSERT_EQ(fromFilter.kept.size(), hand.kept.size());
        for (std::size_t r = 0; r < hand.kept.size(); ++r)
        {
            EXPECT_NEAR(fromFilter.weights[r], hand.weights[r], 1e-10) << "kept " << r;
            EXPECT_TRUE(fromFilter.kept[r].mean.isApprox(hand.kept[r].mean, 1e-10)) << "kept " << r;
            EXPECT_TRUE(fromFilter.kept[r].cov.isApprox(hand.kept[r].cov, 1e-10)) << "kept " << r;
        }
    }
}

// What operator<< writes, as a string.
template <typename T>
std::string
printed(T const & value)
{
    auto stream = std::ostringstream{};
    stream << value;
    return stream.str();
}

}  // namespace

// The sequences of the last two values of a mode space are the product of two tagged copies of it, the current mode
// varying fastest: a value is named through the axis of its lag, the current mode of a sequence is its flat index
// modulo the number of modes, the sequence that follows drops the oldest mode and appends the new one, and a
// categorical over the sequences has the current mode's marginal. Over a product of two axes the copies nest, and
// an axis of the current mode is named through Current.
TEST(ModeSequence, IsTheProductOfTheLastModesWithTheCurrentOneFastest)
{
    using Pairs = ModeSequence<Maneuver, 2>;
    static_assert(Emrys::DiscreteSpace<Pairs>);
    static_assert(Emrys::DiscreteSpace<ModeSequence<Mode, 3>>);
    static_assert(Pairs::Cardinality == 4);
    static_assert(ModeSequence<Mode, 3>::Cardinality == 64);
    static_assert(ModeSequence<Maneuver, 1>::Cardinality == 2);
    static_assert(Pairs::Length == 2);
    static_assert(std::is_same_v<Current<Maneuver>, Lagged<Maneuver, 0>>);
    static_assert(std::is_same_v<Pairs::ProductType, DiscreteProduct<Lagged<Maneuver, 1>, Lagged<Maneuver, 0>>>);

    auto const pairs = Pairs{};
    EXPECT_EQ(pairs.cardinality(), 4);
    EXPECT_EQ(pairs.current().cardinality(), 2);
    EXPECT_EQ(
        (Emrys::valueOf<TagPath<Lagged<Maneuver, 1>, Cruise>, TagPath<Current<Maneuver>, Maneuvering>>(pairs)),
        1
    );
    EXPECT_EQ(
        (Emrys::valueOf<TagPath<Lagged<Maneuver, 1>, Maneuvering>, TagPath<Current<Maneuver>, Cruise>>(pairs)),
        2
    );
    for (Eigen::Index s = 0; s < 4; ++s)
    {
        EXPECT_EQ(pairs.currentOf(s), s % 2);
        for (Eigen::Index next = 0; next < 2; ++next)
        {
            EXPECT_EQ(pairs.following(s, next), 2 * (s % 2) + next);
        }
    }
    EXPECT_EQ(pairs, Pairs{Maneuver{}});

    auto const weights = CategoricalParams<Pairs>::fromWeights(pairs, Eigen::Array4d{0.1, 0.2, 0.3, 0.4});
    auto const marginal = weights.marginal<Current<Maneuver>>();
    auto const now = marginal.toDensity().value();
    EXPECT_NEAR(now.probability<Cruise>(), 0.4, 1e-15);
    EXPECT_NEAR(now.probability<Maneuvering>(), 0.6, 1e-15);
    EXPECT_NEAR((weights.toDensity().value().probability<TagPath<Lagged<Maneuver, 1>, Cruise>>()), 0.3, 1e-15);

    // Three modes of a product of two axes: 64 sequences, the current mode the last 4.
    auto const triples = ModeSequence<Mode, 3>{};
    EXPECT_EQ(triples.currentOf(37), 37 % 4);
    EXPECT_EQ(triples.following(37, 2), (37 % 16) * 4 + 2);
    EXPECT_EQ((Emrys::internal::locateAxisByPath<TagPath<Current<Mode>, Sensor>>(triples).first), 1);
    EXPECT_EQ((Emrys::internal::locateAxisByPath<TagPath<Lagged<Mode, 2>, Maneuver>>(triples).first), 32);

    // A sequence of one mode has the mode space's values, and any other space is its own current mode.
    EXPECT_EQ(Emrys::internal::currentMode(ModeSequence<Maneuver, 1>{}, 1), 1);
    EXPECT_EQ(Emrys::internal::currentMode(Mode{}, 3), 3);
    static_assert(std::is_same_v<Emrys::internal::CurrentModeSpace<Pairs>, Maneuver>);
    static_assert(std::is_same_v<Emrys::internal::CurrentModeSpace<Mode>, Mode>);
}

// overModeSequences() gives each sequence its current mode's component and an equal share of that mode's weight, so
// the current mode's marginal and the mean and covariance over every sequence are the hybrid's own.
TEST(HybridGaussian, SpreadsOverModeSequences)
{
    auto const conditionals = std::vector<GaussianParams<State>>{
        makeGaussian(Vector4{0, 0, 1, 0}, Vector4{4, 4, 1, 1}),
        makeGaussian(Vector4{1, -1, 0.5, 0.5}, Vector4{2, 3, 1, 2})
    };
    auto const prior =
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{0.7, 0.3}), conditionals};
    auto const spread = overModeSequences<3>(prior);
    static_assert(std::is_same_v<decltype(spread), HybridGaussianParams<State, ModeSequence<Maneuver, 3>> const>);
    ASSERT_EQ(spread.size(), 8);
    auto const probabilities = spread.modes().toDensity().value();
    for (Eigen::Index s = 0; s < 8; ++s)
    {
        EXPECT_NEAR(probabilities.probability(s), (s % 2 == 0 ? 0.7 : 0.3) / 4, 1e-15);
        EXPECT_EQ(spread.conditional(s).mean(), conditionals[std::size_t(s % 2)].mean());
    }
    auto const marginal = spread.modes().marginal<Current<Maneuver>>();
    auto const now = marginal.toDensity().value();
    EXPECT_NEAR(now.probability<Maneuvering>(), 0.3, 1e-15);
    auto const collapsed = spread.toDensity().value().collapsed();
    auto const expected = prior.toDensity().value().collapsed();
    EXPECT_TRUE(collapsed.mean().isApprox(expected.mean(), 1e-14));
    EXPECT_TRUE(collapsed.covariance().value().isApprox(expected.covariance().value(), 1e-14));

    // Grouped by the current mode, the sequences give back the hybrid over it.
    auto const grouped = spread.toDensity().value().collapse<Current<Maneuver>>();
    EXPECT_TRUE(grouped.conditional(1).mean().isApprox(conditionals[1].mean(), 1e-14));
    EXPECT_NEAR(grouped.modes().toDensity().value().probability<Maneuvering>(), 0.3, 1e-15);
}

// With a single mode, the filter of sequences of one and of two modes matches a KalmanFilter with the same method
// step for step in state, log-likelihood, and NIS; its model and the mode's are reachable through it.
TEST(GpbFilter, OneModeIsAKalmanFilter)
{
    struct Only
    {};

    struct Single : Named<Only>
    {};

    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 1, 0.5}, Vector4{4, 4, 1, 1});
    auto const prior = HybridParams{CategoricalParams<Single>{}, initial};
    auto const expectKalman = [&](auto filter) {
        auto kalman = KalmanFilter{makeModel(0.1), initial, methods::unscented()};
        EXPECT_EQ(&filter.model().motion(), &std::as_const(filter).model().motion());
        EXPECT_EQ(&filter.recursion().model(), &filter.model());
        EXPECT_TRUE(filter.recursion().partitions(State{}));
        EXPECT_EQ(filter.recursion().sequenceSpace(), filter.state().modeSpace());
        for (int k = 1; k <= 3; ++k)
        {
            filter.predict(timeStep);
            kalman.predict(timeStep);
            auto const fix = makeFix(Eigen::Vector2d{1.1 * k, 0.4 * k});
            auto const update = filter.update(fix);
            auto const kalmanUpdate = kalman.update(fix);
            ASSERT_TRUE(update && kalmanUpdate);
            ASSERT_TRUE(update->accepted);
            EXPECT_NEAR(update->logLikelihood(), kalmanUpdate->logLikelihood(), 1e-12);
            EXPECT_NEAR(update->modes[0].branches[0].nis(), kalmanUpdate->branches[0].nis(), 1e-12);
            EXPECT_DOUBLE_EQ(filter.state().modes().toDensity().value().probability(0), 1.0);
            EXPECT_TRUE(filter.state().conditional(0).mean().isApprox(kalman.state().mean(), 1e-12));
            EXPECT_TRUE(
                filter.state().conditional(0).covarianceLower().isApprox(kalman.state().covarianceLower(), 1e-12)
            );
        }
    };
    expectKalman(
        GpbFilter{
            ContinuousTimeMarkovChain<Single>{},
            makeModel(0.1),
            overModeSequences<1>(prior),
            methods::unscented()
        }
    );
    expectKalman(
        GpbFilter{
            ContinuousTimeMarkovChain<Single>{},
            makeModel(0.1),
            overModeSequences<2>(prior),
            methods::unscented()
        }
    );
}

// Sequences of one, two, and three maneuver modes match the source's estimator of orders one, two, and three (see
// expectMatchesTheSource()).
TEST(GpbFilter, MatchesTheSourceAtOrdersOneToThree)
{
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, 0.05}, {0.1, 0}}).value();
    auto const densities = std::vector<double>{quiet, agile};
    auto const noiseScales = std::vector<double>{1.0, 1.0};
    auto const fix = [](Eigen::Vector2d const & z) {
        return makeFix(z);
    };
    expectMatchesTheSource<Maneuver, 1>(chain, makeMotion(), fix, densities, noiseScales);
    expectMatchesTheSource<Maneuver, 2>(chain, makeMotion(), fix, densities, noiseScales);
    expectMatchesTheSource<Maneuver, 3>(chain, makeMotion(), fix, densities, noiseScales);
}

// Over the four modes of a maneuver and a sensor state, a fix whose noise is by the Sensor axis updates each
// sequence by the noise of its current mode's sensor value, whatever the sequence's length: sequences of one and of
// two modes match the source's estimator in which each mode has its own measurement noise. The branch of each
// sequence's update is labeled with the sequence and with the noise component its current mode names.
TEST(GpbFilter, UpdatesEachSequenceByItsCurrentMode)
{
    auto const maneuverChain =
        ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, 0.05}, {0.1, 0}}).value();
    auto const sensorChain = ContinuousTimeMarkovChain<Sensor>::create(Eigen::Matrix2d{{0, 0.1}, {0.4, 0}}).value();
    auto const chain = ContinuousTimeMarkovChain<Mode>::independent(maneuverChain, sensorChain);
    auto const densities = std::vector<double>{quiet, quiet, agile, agile};
    auto const noiseScales = std::vector<double>{1.0, outlierScale, 1.0, outlierScale};
    auto const fix = [](Eigen::Vector2d const & z) {
        return makeMarkovFix(z);
    };
    expectMatchesTheSource<Mode, 1>(chain, makeMotion(), fix, densities, noiseScales);
    expectMatchesTheSource<Mode, 2>(chain, makeMotion(), fix, densities, noiseScales);

    auto filter = GpbFilter{chain, makeMotion(), makeDistinctState<Mode, 2>()};
    filter.predict(timeStep);
    auto const update = filter.update(makeMarkovFix(Eigen::Vector2d{1.2, 0.1}));
    ASSERT_TRUE(update);
    ASSERT_EQ(update->modes.size(), 16u);
    for (std::size_t s = 0; s < 16; ++s)
    {
        ASSERT_EQ(update->modes[s].branches.size(), 1u);
        EXPECT_EQ(update->modes[s].branches[0].mode, Eigen::Index(s));
        EXPECT_EQ(update->modes[s].branches[0].component, Eigen::Index(s % 2));
    }
}

// From one Gaussian, the filter of sequences of N modes is the exact posterior for its first N steps. The exact
// posterior after k steps is the mixture over every mode history m_0, ..., m_k, enumerated here in plain Eigen with a
// Kalman filter along each history. For each k up to N, the filter's likelihood of the fix is the exact one, and at
// step N each sequence's Gaussian is that of the histories m_1, ..., m_N it names and its probability the sum of
// theirs over m_0. Checked for N = 2 and N = 3 to 1e-10.
TEST(GpbFilter, IsExactForAsManyStepsAsASequenceHoldsModes)
{
    struct History
    {
        std::vector<std::size_t> modes;  // m_0, ..., m_k
        double weight;                   // p(m_0, ..., m_k, z_1, ..., z_k)
        HandGaussian gaussian;
    };

    auto const chain = ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, 0.3}, {0.5, 0}}).value();
    Eigen::Matrix2d const transitions = chain.transition(timeStep).matrix();
    auto const densities = std::vector<double>{quiet, agile};
    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 1, 0.5}, Vector4{4, 4, 1, 1});
    auto const prior =
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{0.7, 0.3}), initial};
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {3.5, -1.5}, {4.0, 0.9}};

    auto const expectExact = [&](auto filter) {
        std::size_t const length = std::remove_cvref_t<decltype(filter)>::Order;
        auto histories = std::vector<History>{
            {{0}, 0.7, {initial.mean(), initial.covariance().value()}},
            {{1}, 0.3, {initial.mean(), initial.covariance().value()}}
        };
        for (std::size_t k = 1; k <= length; ++k)
        {
            Eigen::Vector2d const & z = fixes[k - 1];
            auto next = std::vector<History>{};
            double before = 0;
            double after = 0;
            for (History const & history : histories)
            {
                before += history.weight;
                for (std::size_t j = 0; j < 2; ++j)
                {
                    auto const [posterior, likelihood] = stepByHand(history.gaussian, densities[j], 1.0, z, timeStep);
                    auto modes = history.modes;
                    modes.push_back(j);
                    double const weight =
                        history.weight * transitions(Eigen::Index(history.modes.back()), Eigen::Index(j)) * likelihood;
                    after += weight;
                    next.push_back({modes, weight, posterior});
                }
            }
            histories = next;

            filter.predict(timeStep);
            auto const update = filter.update(makeFix(z));
            ASSERT_TRUE(update);
            EXPECT_NEAR(update->logLikelihood(), std::log(after / before), 1e-10) << "step " << k;
            if (k < length)
            {
                continue;
            }
            auto const probabilities = filter.state().modes().toDensity().value();
            for (Eigen::Index s = 0; s < filter.state().size(); ++s)
            {
                // The sequence's modes m_1, ..., m_N, the oldest first, are the digits of s in base 2.
                double probability = 0;
                for (History const & history : histories)
                {
                    std::size_t named = 0;
                    for (std::size_t i = 1; i <= length; ++i)
                    {
                        named = 2 * named + history.modes[i];
                    }
                    if (named != std::size_t(s))
                    {
                        continue;
                    }
                    probability += history.weight / after;
                    EXPECT_TRUE(filter.state().conditional(s).mean().isApprox(history.gaussian.mean, 1e-10));
                    EXPECT_TRUE(
                        filter.state().conditional(s).covariance().value().isApprox(history.gaussian.cov, 1e-10)
                    );
                }
                EXPECT_NEAR(probabilities.probability(s), probability, 1e-10) << "sequence " << s;
            }
        }
    };
    expectExact(GpbFilter{chain, makeMotion(), overModeSequences<2>(prior)});
    expectExact(GpbFilter{chain, makeMotion(), overModeSequences<3>(prior)});
}

// The filter of sequences of two modes is the interacting multiple model filter on the chain of pairs of modes: an
// ImmFilter whose mode space is the earlier mode and the current one, whose chain moves the pair (a, i) to (i, j)
// with the probability of i moving to j, and whose models follow the current mode. Its mixing for the pair (i, j)
// weights the pairs (a, i) by their probabilities times that of i moving to j, which is the same for each of them,
// so it merges what this filter merges. The two match in every component and probability to 1e-12 over five steps,
// with fixed-step models, which a MarkovChain takes; predict(Steps{2}) is two predictions.
TEST(GpbFilter, SequencesOfTwoModesAreTheImmOnTheChainOfPairs)
{
    struct WasCruise
    {};

    struct WasManeuvering
    {};

    struct Before : Named<WasCruise, WasManeuvering>
    {};

    using Pair = DiscreteProduct<Before, Maneuver>;

    auto const chain = MarkovChain<Maneuver>::create(Eigen::Matrix2d{{0.9, 0.1}, {0.25, 0.75}}).value();
    Eigen::Matrix4d pairs = Eigen::Matrix4d::Zero();
    for (Eigen::Index a = 0; a < 2; ++a)
    {
        for (Eigen::Index i = 0; i < 2; ++i)
        {
            for (Eigen::Index j = 0; j < 2; ++j)
            {
                pairs(2 * a + i, 2 * i + j) = chain.matrix()(i, j);
            }
        }
    }
    auto const pairChain = MarkovChain<Pair>::create(pairs).value();
    auto const motion = byMode<Maneuver>(
        when<Cruise>(FixedStepModel{makeModel(quiet), State{}, timeStep}),
        when<Maneuvering>(FixedStepModel{makeModel(agile), State{}, timeStep})
    );

    auto const start = makeDistinctState<Maneuver, 2>();
    auto conditionals = std::vector<GaussianParams<State>>{};
    for (Eigen::Index s = 0; s < 4; ++s)
    {
        conditionals.push_back(start.conditional(s));
    }
    auto sequences = GpbFilter{chain, motion, start};
    static_assert(Emrys::isFixedStep<decltype(sequences)::ModelType>);
    static_assert(std::is_same_v<decltype(sequences)::MethodType, Emrys::ModelsMethod>);
    auto imm = ImmFilter{
        pairChain,
        motion,
        HybridParams{CategoricalParams<Pair>{Pair{}, start.modes().logWeights()}, conditionals}
    };

    auto const expectSame = [&] {
        auto const mine = sequences.state().modes().toDensity().value();
        auto const theirs = imm.state().modes().toDensity().value();
        for (Eigen::Index s = 0; s < 4; ++s)
        {
            EXPECT_NEAR(mine.probability(s), theirs.probability(s), 1e-12);
            EXPECT_TRUE(sequences.state().conditional(s).mean().isApprox(imm.state().conditional(s).mean(), 1e-12));
            EXPECT_TRUE(sequences.state().conditional(s).covarianceLower().isApprox(
                imm.state().conditional(s).covarianceLower(),
                1e-12
            ));
        }
    };
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {2.0, 0.5}, {9.0, -4.0}, {3.9, 0.9}, {5.5, 1.5}};
    for (Eigen::Vector2d const & z : fixes)
    {
        sequences.predict();
        imm.predict();
        expectSame();
        auto const mine = sequences.update(makeFix(z));
        auto const theirs = imm.update(makeFix(z));
        ASSERT_TRUE(mine && theirs);
        EXPECT_NEAR(mine->logLikelihood(), theirs->logLikelihood(), 1e-12);
        expectSame();
    }
    sequences.predict(Emrys::Steps{2});
    imm.predict(Emrys::Steps{2});
    expectSame();
}

// A filter of square-root Gaussians takes the same steps as one of Gaussians in covariance form: over four steps
// their probabilities, means, and covariances agree to 1e-9, the predictions returning no error.
TEST(GpbFilter, ASquareRootFormMatchesTheCovarianceForm)
{
    using Emrys::SqrtGaussian;
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, 0.05}, {0.1, 0}}).value();
    auto const start = makeDistinctState<Maneuver, 2>();
    auto roots = std::vector<Emrys::SqrtGaussianParams<State>>{};
    for (Eigen::Index s = 0; s < 4; ++s)
    {
        roots.push_back(Emrys::toSqrt(start.conditional(s)).value());
    }
    auto covariance = GpbFilter{chain, makeMotion(), start};
    auto root = GpbFilter{
        chain,
        makeMotion(),
        HybridParams<SqrtGaussian<State>, ModeSequence<Maneuver, 2>>{start.modes(), roots}
    };
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {2.0, 0.5}, {9.0, -4.0}, {3.9, 0.9}};
    for (Eigen::Vector2d const & z : fixes)
    {
        covariance.predict(timeStep);
        ASSERT_TRUE(root.predict(timeStep));
        auto const expected = covariance.update(makeFix(z));
        auto const updated = root.update(makeFix(z));
        ASSERT_TRUE(expected && updated);
        EXPECT_NEAR(updated->logLikelihood(), expected->logLikelihood(), 1e-9);
        auto const mine = root.state().modes().toDensity().value();
        auto const theirs = covariance.state().modes().toDensity().value();
        for (Eigen::Index s = 0; s < 4; ++s)
        {
            EXPECT_NEAR(mine.probability(s), theirs.probability(s), 1e-9);
            auto const gaussian = Emrys::toGaussian(root.state().conditional(s)).value();
            EXPECT_TRUE(gaussian.mean().isApprox(covariance.state().conditional(s).mean(), 1e-9));
            EXPECT_TRUE(
                gaussian.covariance().value().isApprox(covariance.state().conditional(s).covariance().value(), 1e-9)
            );
        }
    }
}

// Under a Gate, a sequence whose update is rejected keeps its Gaussian and gets weight zero; a fix rejected in
// every sequence leaves the state unchanged; refilling one result, made empty by the filter's emptyUpdate(), gives
// the state that the update returning its result gives; and setState() takes a state back.
TEST(GpbFilter, RejectedSequencesLoseTheirWeight)
{
    ContinuousTimeMarkovChain const chain = ContinuousTimeMarkovChain<Maneuver>::persistent(0.1);
    // Sequences that end in Cruise are confident near the origin and those that end in Maneuvering are vague: a fix
    // 30 m away fails the gate only in the first.
    auto const conditionals = std::vector<GaussianParams<State>>{
        makeGaussian(Vector4::Zero(), Vector4{0.5, 0.5, 0.1, 0.1}),
        makeGaussian(Vector4::Zero(), Vector4{400, 400, 1, 1})
    };
    auto const prior = overModeSequences<2>(HybridParams{CategoricalParams<Maneuver>{}, conditionals});
    auto filter = GpbFilter{chain, makeMotion(), prior};

    auto const update = filter.update(makeFix(Eigen::Vector2d{30, 0}), reductions::gate(0.997));
    ASSERT_TRUE(update);
    ASSERT_TRUE(update->accepted);
    for (std::size_t s = 0; s < 4; ++s)
    {
        EXPECT_EQ(update->modes[s].accepted, s % 2 == 1);
    }
    auto const marginal = filter.state().modes().marginal<Current<Maneuver>>();
    auto const now = marginal.toDensity().value();
    EXPECT_EQ(now.probability<Maneuvering>(), 1.0);
    EXPECT_EQ(filter.state().conditional(0).mean(), prior.conditional(0).mean());
    EXPECT_GT(filter.state().conditional(1).mean()(0), 25.0);

    // A fix that fails in every sequence is rejected and leaves the state unchanged.
    auto const state = filter.state();
    auto const far = filter.update(makeFix(Eigen::Vector2d{5000, 0}), reductions::gate(0.997, reductions::select()));
    ASSERT_TRUE(far);
    EXPECT_FALSE(far->accepted);
    EXPECT_TRUE((filter.state().modes().logWeights() == state.modes().logWeights()).all());
    EXPECT_EQ(filter.state().conditional(1).mean(), state.conditional(1).mean());

    // The update into a result that is reused.
    auto refilled = GpbFilter{chain, makeMotion(), prior};
    auto returned = GpbFilter{chain, makeMotion(), prior};
    auto const fix = makeFix(Eigen::Vector2d{1, 0.5});
    auto result = refilled.emptyUpdate<decltype(fix)>();
    for (int k = 0; k < 2; ++k)
    {
        refilled.predict(timeStep);
        returned.predict(timeStep);
        ASSERT_TRUE(refilled.update(fix, reductions::collapse(), result));
        auto const expected = returned.update(fix);
        ASSERT_TRUE(expected);
        EXPECT_EQ(result.logLikelihood(), expected->logLikelihood());
        EXPECT_EQ(result.modes.size(), 4u);
    }
    EXPECT_EQ(refilled.state().conditional(3).mean(), returned.state().conditional(3).mean());

    refilled.setState(prior);
    EXPECT_EQ(refilled.state().conditional(1).mean(), prior.conditional(1).mean());
    EXPECT_EQ(refilled.get_allocator(), prior.get_allocator());
}

// The filter deduces an upkeep given after the state or after the method, and maintain() runs it on the state: a
// floor of 0.05 on the weights of the four sequences raises a sequence of weight zero to it. The filter prints its
// name and its state nested under it, a hybrid of four modes.
TEST(GpbFilter, HoldsAnUpkeepAndPrints)
{
    namespace upkeep = Emrys::upkeep;
    ContinuousTimeMarkovChain const chain = ContinuousTimeMarkovChain<Maneuver>::persistent(0.1);
    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 1, 0.5}, Vector4{4, 4, 1, 1});
    auto prior = overModeSequences<2>(HybridParams{CategoricalParams<Maneuver>{}, initial});
    auto weights = prior.modes();
    weights.setWeight(0, 0.0);
    prior.setModes(weights);

    auto floored = GpbFilter{chain, makeMotion(), prior, upkeep::floorModes(0.05)};
    static_assert(!std::is_same_v<decltype(floored)::UpkeepType, upkeep::None>);
    auto const maintained = floored.maintain();
    ASSERT_TRUE(maintained);
    EXPECT_TRUE(*maintained);
    EXPECT_NEAR(floored.state().modes().toDensity().value().probability(0), 0.05, 1e-15);

    auto both = GpbFilter{chain, makeMotion(), prior, methods::unscented(), upkeep::floorModes(0.05)};
    static_assert(std::is_same_v<decltype(both)::UpkeepType, decltype(floored)::UpkeepType>);
    static_assert(!std::is_same_v<decltype(both)::MethodType, decltype(floored)::MethodType>);
    ASSERT_TRUE(both.maintain());

    // The switching model itself, in place of the chain and the motion.
    auto const model = Emrys::SwitchingModel{chain, makeMotion()};
    auto const fromModel = GpbFilter{model, prior};
    static_assert(std::is_same_v<decltype(fromModel)::StateType, decltype(prior)>);

    std::string const text = printed(floored);
    EXPECT_EQ(text.rfind("generalized pseudo-Bayesian filter\n", 0), 0u) << text;
    EXPECT_NE(text.find("hybrid of 4 modes"), std::string::npos) << text;
}

// On the ImmFilter test's maneuvering target, a cruise, a 10-step sideways acceleration of 2 m/s^2, and another
// cruise, the position error over steps 11 to 60, as a root mean square over 200 runs that give every filter the same
// truth and the same noise, is larger with sequences of one mode than the IMM's and smaller with sequences of two, by
// the margins asserted: the order Bar-Shalom, Li, and Kirubarajan (2001) report. Over 2000 runs the four errors were
// 1.148 (one mode), 0.975 (IMM), 0.948 (two modes), and 0.939 (three). Nothing orders them on one run, and within
// the ten steps of the maneuver they differed by less than the sampling error, so the test claims neither.
TEST(GpbFilter, RanksWithTheImmOnAManeuverOnAverage)
{
    auto const noiseFactor = Eigen::LLT<Eigen::Matrix2d>{R};
    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 10, 0}, Vector4{10, 10, 4, 4});
    auto const prior = HybridParams{CategoricalParams<Maneuver>{}, initial};
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::persistent(0.05);
    double oneError = 0;
    double immError = 0;
    double twoError = 0;
    int const runs = 200;
    for (int run = 0; run < runs; ++run)
    {
        auto rng = std::mt19937_64{std::uint64_t(run) + 1};
        auto normal = std::normal_distribution<double>{};
        auto one = GpbFilter{chain, makeMotion(), overModeSequences<1>(prior)};
        auto imm = ImmFilter{chain, makeMotion(), prior};
        auto two = GpbFilter{chain, makeMotion(), overModeSequences<2>(prior)};
        auto truth = Vector4{0, 0, 10, 0};
        for (int k = 1; k <= 60; ++k)
        {
            double const accel = k > 30 && k <= 40 ? 2.0 : 0.0;
            truth.head<2>() += truth.tail<2>() * timeStep + Eigen::Vector2d{0, 0.5 * accel};
            truth(3) += accel * timeStep;
            Eigen::Vector2d const z =
                truth.head<2>() + noiseFactor.matrixL() * Eigen::Vector2d{normal(rng), normal(rng)};
            auto const squaredError = [&](auto & filter) {
                filter.predict(timeStep);
                EXPECT_TRUE(filter.update(makeFix(z)));
                Vector4 const mean = filter.state().toDensity().value().collapsed().mean();
                return k > 10 ? (mean.head<2>() - truth.head<2>()).squaredNorm() : 0.0;
            };
            oneError += squaredError(one);
            immError += squaredError(imm);
            twoError += squaredError(two);
        }
    }
    double const oneRms = std::sqrt(oneError / (50 * runs));
    double const immRms = std::sqrt(immError / (50 * runs));
    double const twoRms = std::sqrt(twoError / (50 * runs));
    EXPECT_GT(oneRms, 1.1 * immRms) << oneRms << " " << immRms;
    EXPECT_LT(twoRms, 0.99 * immRms) << twoRms << " " << immRms;
}
