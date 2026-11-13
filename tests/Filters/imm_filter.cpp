// ImmFilter and HybridGaussian (steps 5b, 13d; [switching-models], [imm], [fixed-step-models]): collapsing
// over modes and by dimension, one mode is a Kalman filter, a hand-computed IMM step with Markov
// outlier states, noise by mode axis read as memoryless outliers by a Kalman filter and matched by an
// IMM with a memoryless chain, a measurement per value of a mode axis, fixed-step models fix the step
// and the method, each mode predicts by its own method, rejected modes lose their probability, a
// maneuver is detected, and a burst of outliers is ridden out.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::DiscreteProduct;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianParams;
using Emrys::HybridParams;
using Emrys::ImmFilter;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::ScalarOf;
using Emrys::switchingMeasurement;
using Emrys::SwitchingModel;
using Emrys::switchingModel;
using Emrys::valueOf;

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

auto
makeModel(double density)
{
    return ConstantVelocity<Position, Velocity>{density};
}

GaussianParams<State>
makeGaussian(Vector4 const & mean, Vector4 const & variances)
{
    return {State{}, mean, Matrix4{variances.asDiagonal()}};
}

Matrix4
transitionMatrix()
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

Matrix4
processNoise(double density)
{
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    auto q = Matrix4{};
    q << std::pow(timeStep, 3) / 3 * i2, timeStep * timeStep / 2 * i2, timeStep * timeStep / 2 * i2, timeStep * i2;
    return density * q;
}

Eigen::Matrix<double, 2, 4> const H{{1, 0, 0, 0}, {0, 1, 0, 0}};
Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};
constexpr double outlierScale = 100;

// Constant velocity slowed by a drag quadratic in each velocity coordinate, v' = v - c dt v * v: a
// model that isn't affine, so that Linearized and Unscented predict it differently.
auto
makeDragModel(double density)
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            Emrys::ValueVector<Velocity, T> const v = Emrys::block<Velocity>(in, x);
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Position>(out, y) = Emrys::block<Position>(in, x) + T(dt) * v;
            Emrys::block<Velocity>(out, y) = v - T(0.2 * dt) * v.cwiseProduct(v);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(density))
    );
}

// A position fix with the noise policy \a noise.
template <typename NoisePolicy>
auto
observeFix(Eigen::Vector2d const & z, NoisePolicy noise)
{
    return observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(Eigen::Matrix2d::Identity(), Emrys::reads(Emrys::path<Position>), std::move(noise)),
        z
    );
}

template <typename Noise>
auto
makeFix(Eigen::Vector2d const & z, Noise noise)
{
    return observeFix(z, additiveNoise(std::move(noise)));
}

auto
makeFix(Eigen::Vector2d const & z)
{
    return makeFix(z, GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R});
}

// Inlier noise R and outlier noise 100 R; the weights matter only to filters that read the noise
// as a memoryless mixture.
GaussianMixtureParams<Linear<2>>
robustNoise()
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), outlierScale * R);
    return noise;
}

// A fix with the robust noise, memoryless.
auto
makeRobustFix(Eigen::Vector2d const & z)
{
    return makeFix(z, robustNoise());
}

// A fix with the robust noise by the mode axis Sensor: inlier and outlier are its values.
auto
makeMarkovFix(Eigen::Vector2d const & z)
{
    return observeFix(z, additiveNoise(robustNoise()).byMode<Sensor>());
}

// A hand-written IMM over the four modes of Mode, in plain Eigen.
struct HandImm
{
    std::vector<double> probabilities;
    std::vector<Vector4> means;
    std::vector<Matrix4> covs;
};

double
densityOf(std::size_t mode)
{
    return mode / 2 == 0 ? quiet : agile;
}

void
predictByHand(HandImm & imm, Eigen::Matrix4d const & transitions)
{
    std::size_t const n = imm.probabilities.size();
    auto next = HandImm{std::vector<double>(n, 0.0), std::vector<Vector4>(n), std::vector<Matrix4>(n)};
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            next.probabilities[j] += transitions(Eigen::Index(i), Eigen::Index(j)) * imm.probabilities[i];
        }
        Vector4 mean = Vector4::Zero();
        for (std::size_t i = 0; i < n; ++i)
        {
            mean += transitions(Eigen::Index(i), Eigen::Index(j)) * imm.probabilities[i] / next.probabilities[j]
                  * imm.means[i];
        }
        Matrix4 cov = Matrix4::Zero();
        for (std::size_t i = 0; i < n; ++i)
        {
            double const mixing =
                transitions(Eigen::Index(i), Eigen::Index(j)) * imm.probabilities[i] / next.probabilities[j];
            Vector4 const d = imm.means[i] - mean;
            cov += mixing * (imm.covs[i] + d * d.transpose());
        }
        Matrix4 const f = transitionMatrix();
        next.means[j] = f * mean;
        next.covs[j] = f * cov * f.transpose() + processNoise(densityOf(j));
    }
    imm = next;
}

// Updates each mode with noise scale noiseScale(mode) and the predicted measurement offset by
// offset(mode); returns log p(z).
template <typename NoiseScale, typename Offset>
double
updateByHand(HandImm & imm, Eigen::Vector2d const & z, NoiseScale noiseScale, Offset offset)
{
    std::size_t const n = imm.probabilities.size();
    auto joint = std::vector<double>(n);
    double evidence = 0;
    for (std::size_t m = 0; m < n; ++m)
    {
        Eigen::Matrix2d const s = H * imm.covs[m] * H.transpose() + noiseScale(m) * R;
        Eigen::Vector2d const innovation = z - H * imm.means[m] - offset(m);
        Eigen::Matrix<double, 4, 2> const k = imm.covs[m] * H.transpose() * s.inverse();
        double const logLikelihood =
            -0.5 * (2 * std::log(2 * pi) + std::log(s.determinant()) + innovation.dot(s.inverse() * innovation));
        imm.means[m] += k * innovation;
        imm.covs[m] = (Matrix4::Identity() - k * H) * imm.covs[m];
        joint[m] = imm.probabilities[m] * std::exp(logLikelihood);
        evidence += joint[m];
    }
    for (std::size_t m = 0; m < n; ++m)
    {
        imm.probabilities[m] = joint[m] / evidence;
    }
    return std::log(evidence);
}

// Updates each mode with noise scale noiseScale(mode); returns log p(z).
template <typename NoiseScale>
double
updateByHand(HandImm & imm, Eigen::Vector2d const & z, NoiseScale noiseScale)
{
    return updateByHand(imm, z, noiseScale, [](std::size_t) {
        return Eigen::Vector2d::Zero().eval();
    });
}

// A fix through a function of its own, not a matrix: the position offset by bias, with noise
// scale times R.
auto
makeBiasedFix(Eigen::Vector2d const & z, Eigen::Vector2d const & bias, double scale)
{
    return observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            [bias](auto const & in, auto const &, auto const & x) {
                auto y = Eigen::Matrix<ScalarOf<decltype(x)>, 2, 1>{Emrys::block<Position>(in, x)};
                y(0) += bias(0);
                y(1) += bias(1);
                return y;
            },
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), scale * R})
        ),
        z
    );
}

template <typename Filter>
void
expectMatches(Filter const & filter, HandImm const & imm, double tolerance)
{
    auto const & state = filter.state();
    for (Eigen::Index m = 0; m < state.size(); ++m)
    {
        std::size_t const i = std::size_t(m);
        EXPECT_NEAR(state.modes().toDensity().value().probability(m), imm.probabilities[i], tolerance) << "mode " << m;
        EXPECT_TRUE(state.conditional(m).mean().isApprox(imm.means[i], tolerance)) << "mode " << m;
        EXPECT_TRUE(state.conditional(m).covarianceLower().isApprox(imm.covs[i], tolerance)) << "mode " << m;
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

// A HybridGaussian over a two-axis mode deduces its type, addresses a conditional by mode names,
// collapses over all modes to the hand moment-matched Gaussian, converts to a mixture, and
// collapse<Axis>() sums out the other axis with the right marginal probabilities and conditionals.
TEST(HybridGaussian, CollapsesOverModesAndByDimension)
{
    auto const means = std::vector<Vector4>{{0, 0, 1, 0}, {1, 2, 0, 1}, {4, -1, 1, 1}, {-2, 3, 0, 0}};
    auto const variances = std::vector<Vector4>{{1, 1, 1, 1}, {2, 1, 1, 1}, {1, 3, 1, 1}, {1, 1, 2, 2}};
    auto conditionals = std::vector<GaussianParams<State>>{};
    for (std::size_t m = 0; m < 4; ++m)
    {
        conditionals.push_back(makeGaussian(means[m], variances[m]));
    }
    auto const p = Eigen::Array4d{0.4, 0.1, 0.3, 0.2};
    auto const state = HybridParams{CategoricalParams<Mode>::fromWeights(Mode{}, p), conditionals};
    static_assert(std::is_same_v<decltype(state), HybridGaussianParams<State, Mode> const>);

    EXPECT_EQ((&state.conditional<Maneuvering, Inlier>()), &state.conditional(2));

    auto const collapse = [&](std::vector<std::size_t> const & modes) {
        double total = 0;
        Vector4 mean = Vector4::Zero();
        for (std::size_t const m : modes)
        {
            total += p(Eigen::Index(m));
        }
        for (std::size_t const m : modes)
        {
            mean += p(Eigen::Index(m)) / total * means[m];
        }
        Matrix4 cov = Matrix4::Zero();
        for (std::size_t const m : modes)
        {
            Vector4 const d = means[m] - mean;
            cov += p(Eigen::Index(m)) / total * (Matrix4{variances[m].asDiagonal()} + d * d.transpose());
        }
        return std::pair{mean, cov};
    };

    auto const [mean, cov] = collapse({0, 1, 2, 3});
    auto const density = state.toDensity();
    ASSERT_TRUE(density);
    EXPECT_TRUE(density->collapsed().mean().isApprox(mean, 1e-14));
    EXPECT_TRUE(density->collapsed().covarianceLower().isApprox(cov, 1e-14));
    EXPECT_TRUE(density->moments().mean().isApprox(mean, 1e-14));
    auto const mixture = state.toMixture();
    EXPECT_EQ(mixture.size(), 4);

    HybridGaussianParams<State, Maneuver> const byManeuver = density->collapse<Maneuver>();
    EXPECT_NEAR(byManeuver.modes().toDensity().value().probability<Maneuvering>(), 0.5, 1e-15);
    auto const [cruiseMean, cruiseCov] = collapse({0, 1});
    EXPECT_TRUE(byManeuver.conditional<Cruise>().mean().isApprox(cruiseMean, 1e-14));
    EXPECT_TRUE(byManeuver.conditional<Cruise>().covarianceLower().isApprox(cruiseCov, 1e-14));
    HybridGaussianParams<State, Sensor> const bySensor = density->collapse<Sensor>();
    auto const [outlierMean, outlierCov] = collapse({1, 3});
    EXPECT_TRUE(bySensor.conditional<Outlier>().mean().isApprox(outlierMean, 1e-14));
    EXPECT_TRUE(bySensor.conditional<Outlier>().covarianceLower().isApprox(outlierCov, 1e-14));

    // A value of probability zero keeps its modes' plain average, which nothing weights.
    auto const neverManeuvering =
        HybridParams{CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array4d{0.5, 0.5, 0, 0}), conditionals};
    HybridGaussianParams<State, Maneuver> const cruising = neverManeuvering.toDensity()->collapse<Maneuver>();
    EXPECT_EQ(cruising.modes().toDensity().value().probability<Maneuvering>(), 0.0);
    EXPECT_TRUE(cruising.conditional<Maneuvering>().mean().isApprox((means[2] + means[3]) / 2, 1e-14));
}

// With a single mode, the IMM matches a KalmanFilter with the same method step for step in state,
// log-likelihood, and NIS; its model and the mode's are reachable through it.
TEST(ImmFilter, OneModeIsAKalmanFilter)
{
    struct Only
    {};

    struct Single : Named<Only>
    {};

    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 1, 0.5}, Vector4{4, 4, 1, 1});
    auto imm = ImmFilter{
        SwitchingModel{MarkovChain<Single>{}, makeModel(0.1)},
        HybridParams{CategoricalParams<Single>{}, initial},
        methods::unscented()
    };
    auto kalman = KalmanFilter{makeModel(0.1), initial, methods::unscented()};
    EXPECT_EQ(&imm.model().model<0>(), &std::as_const(imm).model().model<0>());
    EXPECT_EQ(&imm.recursion().model(), &imm.model());
    EXPECT_TRUE(imm.recursion().partitions(State{}));
    for (int k = 1; k <= 3; ++k)
    {
        imm.predict(timeStep);
        kalman.predict(timeStep);
        auto const fix = makeFix(Eigen::Vector2d{1.1 * k, 0.4 * k});
        auto const immUpdate = imm.update(fix);
        auto const kalmanUpdate = kalman.update(fix);
        ASSERT_TRUE(immUpdate && kalmanUpdate);
        ASSERT_TRUE(immUpdate->accepted);
        EXPECT_NEAR(immUpdate->logLikelihood(), kalmanUpdate->logLikelihood(), 1e-12);
        EXPECT_NEAR(immUpdate->modes[0].branches[0].nis(), kalmanUpdate->branches[0].nis(), 1e-12);
        EXPECT_DOUBLE_EQ(imm.state().modes().toDensity().value().probability(0), 1.0);
        EXPECT_TRUE(imm.state().conditional(0).mean().isApprox(kalman.state().mean(), 1e-12));
        EXPECT_TRUE(imm.state().conditional(0).covarianceLower().isApprox(kalman.state().covarianceLower(), 1e-12));
    }
}

// Over four modes (maneuver times sensor state), predict() matches the hand IMM mixing and
// prediction, an update with inlier/outlier noise by the Sensor axis matches the hand update with
// each mode's noise component and labels its branches, the outlier modes take the probability after a jump, and
// a plain Gaussian fix updates every mode alike.
TEST(ImmFilter, MatchesAHandComputedImmWithMarkovOutliers)
{
    auto const maneuverChain = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.1, 0.9}}};
    auto const sensorChain = MarkovChain<Sensor>{Eigen::Matrix2d{{0.9, 0.1}, {0.4, 0.6}}};
    auto const chain = MarkovChain<Mode>{maneuverChain, sensorChain};

    auto const means = std::vector<Vector4>{{0, 0, 1, 0}, {0.5, 0, 1, 0.2}, {0, 1, 0.5, 0.5}, {1, -1, 1, 0}};
    auto const variances = std::vector<Vector4>{{4, 4, 1, 1}, {2, 3, 1, 2}, {5, 4, 2, 1}, {4, 2, 1, 1}};
    auto conditionals = std::vector<GaussianParams<State>>{};
    auto hand = HandImm{{0.4, 0.1, 0.3, 0.2}, {}, {}};
    for (std::size_t m = 0; m < 4; ++m)
    {
        conditionals.push_back(makeGaussian(means[m], variances[m]));
        hand.means.push_back(means[m]);
        hand.covs.push_back(Matrix4{variances[m].asDiagonal()});
    }
    auto filter = ImmFilter{
        switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile)),
        HybridParams{CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array4d{0.4, 0.1, 0.3, 0.2}), conditionals}
    };
    static_assert(std::is_same_v<decltype(filter)::ModeSpaceType, Mode>);
    // The abbreviation deduces the same filter.
    auto const abbreviated =
        Emrys::ImmFilter{switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile)), filter.state()};
    static_assert(std::is_same_v<decltype(abbreviated), const decltype(filter)>);

    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {2.0, 0.5}, {9.0, -4.0}, {3.9, 0.9}};
    for (Eigen::Vector2d const & z : fixes)
    {
        filter.predict(timeStep);
        predictByHand(hand, chain.matrix());
        expectMatches(filter, hand, 1e-12);

        auto const update = filter.update(makeMarkovFix(z));
        double const logEvidence = updateByHand(hand, z, [](std::size_t m) {
            return m % 2 == 0 ? 1.0 : outlierScale;
        });
        ASSERT_TRUE(update);
        ASSERT_TRUE(update->accepted);
        EXPECT_NEAR(update->logLikelihood(), logEvidence, 1e-12);
        for (std::size_t m = 0; m < 4; ++m)
        {
            ASSERT_EQ(update->modes[m].branches.size(), 1u);
            EXPECT_EQ(update->modes[m].branches[0].mode, Eigen::Index(m));
            EXPECT_EQ(update->modes[m].branches[0].component, Eigen::Index(m % 2));
        }
        expectMatches(filter, hand, 1e-12);
        // The third fix jumps 6 m, and the outlier modes take most of the probability.
        if (&z == &fixes[2])
        {
            EXPECT_GT(filter.state().modes().toDensity().value().probability<Outlier>(), 0.7);
        }
    }

    // A fix with ordinary Gaussian noise updates every mode the same way.
    filter.predict(timeStep);
    predictByHand(hand, chain.matrix());
    auto const plain = filter.update(makeFix(Eigen::Vector2d{5.0, 1.0}));
    ASSERT_TRUE(plain);
    EXPECT_NEAR(
        plain->logLikelihood(),
        updateByHand(
            hand,
            Eigen::Vector2d{5.0, 1.0},
            [](std::size_t) {
                return 1.0;
            }
        ),
        1e-12
    );
    expectMatches(filter, hand, 1e-12);
}

// Noise by mode axis ([imm]) is the memoryless outlier model where no mode space has its axis: a
// Kalman filter given the Markov fix updates exactly as given the robust fix, which carries the same
// mixture unmarked. An IMM over Sensor alone whose chain is memoryless, each row the noise's
// weights, is the Markov model of that sensor and matches the Kalman filter to rounding (1e-12; the
// guide measured 1e-16): mixing with identical rows starts every mode from the collapsed Gaussian
// that Collapse hands the Kalman filter, and each mode's likelihood is its component's.
TEST(ImmFilter, NoiseByModeIsMemorylessWhereNoModeSpaceHasItsAxis)
{
    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 10, 0}, Vector4{10, 10, 4, 4});
    auto const rate = CategoricalParams<Sensor>::fromWeights(Sensor{}, Eigen::Vector2d{0.9, 0.1});
    auto const memoryless = MarkovChain<Sensor>::memoryless(rate.toDensity().value());
    auto imm = ImmFilter{SwitchingModel{memoryless, makeModel(quiet)}, HybridParams{rate, initial}};
    auto kalman = KalmanFilter{makeModel(quiet), initial};
    auto robust = KalmanFilter{makeModel(quiet), initial};
    for (int k = 1; k <= 6; ++k)
    {
        // A target at 10 m/s along x, with exact fixes but the fourth, 25 m off.
        Eigen::Vector2d const z{10.0 * k * timeStep, k == 4 ? 25.0 : 0.0};
        imm.predict(timeStep);
        kalman.predict(timeStep);
        robust.predict(timeStep);
        auto const markov = imm.update(makeMarkovFix(z));
        auto const read = kalman.update(makeMarkovFix(z));
        auto const unmarked = robust.update(makeRobustFix(z));
        ASSERT_TRUE(markov && read && unmarked);
        ASSERT_EQ(read->branches.size(), 2u);
        EXPECT_EQ(read->logLikelihood(), unmarked->logLikelihood());
        EXPECT_TRUE(kalman.state().mean() == robust.state().mean());
        EXPECT_TRUE(kalman.state().covarianceLower() == robust.state().covarianceLower());

        auto const collapsed = imm.state().toDensity()->collapsed();
        EXPECT_LE((collapsed.mean() - kalman.state().mean()).norm(), 1e-12) << k;
        EXPECT_LE((collapsed.covarianceLower() - kalman.state().covarianceLower()).norm(), 1e-12) << k;
        EXPECT_NEAR(markov->logLikelihood(), read->logLikelihood(), 1e-12) << k;
    }
}

// Over four modes, an update by switchingMeasurement<Sensor>() matches the hand update in which the
// Inlier modes see the plain fix and the Outlier modes a fix of another model type, biased and
// with four times the noise; switchingMeasurement<Maneuver>() selects by the other axis, whose
// stride in the flat index is two; and refilling one result gives the same state as the update
// that returns its result.
TEST(ImmFilter, UpdatesEachModeByTheMeasurementItsValueSelects)
{
    auto const maneuverChain = MarkovChain{Maneuver{}, Eigen::Matrix2d{{0.95, 0.05}, {0.1, 0.9}}};
    auto const sensorChain = MarkovChain{Sensor{}, Eigen::Matrix2d{{0.9, 0.1}, {0.4, 0.6}}};
    auto const chain = MarkovChain<Mode>{Mode{}, maneuverChain, sensorChain};
    auto const means = std::vector<Vector4>{{0, 0, 1, 0}, {0.5, 0, 1, 0.2}, {0, 1, 0.5, 0.5}, {1, -1, 1, 0}};
    auto const variances = std::vector<Vector4>{{4, 4, 1, 1}, {2, 3, 1, 2}, {5, 4, 2, 1}, {4, 2, 1, 1}};
    auto conditionals = std::vector<GaussianParams<State>>{};
    auto hand = HandImm{{0.4, 0.1, 0.3, 0.2}, {}, {}};
    for (std::size_t m = 0; m < 4; ++m)
    {
        conditionals.push_back(makeGaussian(means[m], variances[m]));
        hand.means.push_back(means[m]);
        hand.covs.push_back(Matrix4{variances[m].asDiagonal()});
    }
    auto const initial =
        HybridParams{CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array4d{0.4, 0.1, 0.3, 0.2}), conditionals};
    auto filter = ImmFilter{switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile)), initial};
    auto refilled = filter;
    auto const handStart = hand;

    Eigen::Vector2d const bias{3.0, -1.0};
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {2.0, 0.5}, {9.0, -4.0}, {3.9, 0.9}};
    auto const switchingFix = [&](Eigen::Vector2d const & z) {
        return switchingMeasurement<Sensor>(makeFix(z), makeBiasedFix(z, bias, 4.0));
    };
    using Switching = decltype(switchingFix(fixes.front()));
    auto result = decltype(filter)::RecursionType::SwitchingResultType<Switching>{};
    for (Eigen::Vector2d const & z : fixes)
    {
        filter.predict(timeStep);
        refilled.predict(timeStep);
        predictByHand(hand, chain.matrix());

        auto const update = filter.update(switchingFix(z));
        double const logEvidence = updateByHand(
            hand,
            z,
            [](std::size_t m) {
                return m % 2 == 0 ? 1.0 : 4.0;
            },
            [&](std::size_t m) {
                return m % 2 == 0 ? Eigen::Vector2d::Zero().eval() : bias;
            }
        );
        ASSERT_TRUE(update);
        ASSERT_TRUE(update->accepted);
        EXPECT_NEAR(update->logLikelihood(), logEvidence, 1e-12);
        for (std::size_t m = 0; m < 4; ++m)
        {
            ASSERT_EQ(update->modes[m].branches.size(), 1u);
            EXPECT_EQ(update->modes[m].branches[0].mode, Eigen::Index(m));
        }
        expectMatches(filter, hand, 1e-12);

        ASSERT_TRUE(refilled.update(switchingFix(z), reductions::collapse(), result));
        EXPECT_EQ(result.logLikelihood(), update->logLikelihood());
        for (Eigen::Index m = 0; m < 4; ++m)
        {
            EXPECT_EQ(refilled.state().conditional(m).mean(), filter.state().conditional(m).mean());
        }
    }

    // Selected by Maneuver, the bias belongs to the maneuvering modes, two and three.
    auto byManeuver = ImmFilter{switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile)), initial};
    hand = handStart;
    for (Eigen::Vector2d const & z : fixes)
    {
        byManeuver.predict(timeStep);
        predictByHand(hand, chain.matrix());
        auto const update = byManeuver.update(switchingMeasurement<Maneuver>(makeFix(z), makeBiasedFix(z, bias, 4.0)));
        double const logEvidence = updateByHand(
            hand,
            z,
            [](std::size_t m) {
                return m / 2 == 0 ? 1.0 : 4.0;
            },
            [&](std::size_t m) {
                return m / 2 == 0 ? Eigen::Vector2d::Zero().eval() : bias;
            }
        );
        ASSERT_TRUE(update);
        EXPECT_NEAR(update->logLikelihood(), logEvidence, 1e-12);
        expectMatches(byManeuver, hand, 1e-12);
    }
}

// With fixed-step models [fixed-step-models], the IMM over four modes deduces through the abbreviation, takes
// ModelsMethod as its method, each mode predicting by its model's, and predicts with no step. Over four fixes with
// Markov outliers it matches the same filter predicting over the step: to rounding under Linearized, where each mode
// predicts by its cached transition, and exactly under Unscented, which takes the same path with the same noise, also
// when the update refills one result.
TEST(ImmFilter, FixedStepModelsFixTheStepAndTheMethod)
{
    using Emrys::FixedStepModel;

    auto const maneuverChain = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.1, 0.9}}};
    auto const sensorChain = MarkovChain<Sensor>{Eigen::Matrix2d{{0.9, 0.1}, {0.4, 0.6}}};
    auto const chain = MarkovChain<Mode>{maneuverChain, sensorChain};
    auto conditionals = std::vector<GaussianParams<State>>{};
    for (int m = 0; m < 4; ++m)
    {
        conditionals.push_back(makeGaussian(Vector4{0.5 * m, -0.5 * m, 1, 0.2 * m}, Vector4{4, 3, 1, 2}));
    }
    auto const initial =
        HybridParams{CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array4d{0.4, 0.1, 0.3, 0.2}), conditionals};

    auto cached = Emrys::ImmFilter{
        switchingModel<Maneuver>(
            chain,
            FixedStepModel{makeModel(quiet), State{}, timeStep},
            FixedStepModel{makeModel(agile), State{}, timeStep}
        ),
        initial
    };
    static_assert(Emrys::isFixedStep<decltype(cached)::ModelType>);
    static_assert(std::is_same_v<decltype(cached)::MethodType, Emrys::ModelsMethod>);
    auto unscented = Emrys::ImmFilter{
        switchingModel<Maneuver>(
            chain,
            FixedStepModel{makeModel(quiet), State{}, timeStep, methods::unscented<0.5>()},
            FixedStepModel{makeModel(agile), State{}, timeStep, methods::unscented<0.5>()}
        ),
        initial
    };
    static_assert(std::is_same_v<decltype(unscented)::MethodType, Emrys::ModelsMethod>);
    EXPECT_EQ(unscented.model().model<1>().method().alpha, 0.5);
    auto continuousCached = ImmFilter{switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile)), initial};
    auto continuousUnscented = ImmFilter{
        switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile)),
        initial,
        methods::unscented<0.5>()
    };

    auto const expectSame = [](auto const & filter, auto const & expected, double tolerance) {
        EXPECT_LE(
            (filter.state().modes().toDensity().value().probabilities()
             - expected.state().modes().toDensity().value().probabilities())
                .matrix()
                .norm(),
            tolerance
        );
        for (Eigen::Index m = 0; m < 4; ++m)
        {
            EXPECT_LE(
                (filter.state().conditional(m).mean() - expected.state().conditional(m).mean()).norm(),
                tolerance
            );
            EXPECT_LE(
                (filter.state().conditional(m).covarianceLower() - expected.state().conditional(m).covarianceLower())
                    .norm(),
                tolerance
            );
        }
    };
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.1}, {2.0, 0.5}, {9.0, -4.0}, {3.9, 0.9}};
    using AxisUpdate = decltype(continuousUnscented.update(makeMarkovFix(fixes.front())))::value_type;
    auto refilled = AxisUpdate{};
    for (Eigen::Vector2d const & z : fixes)
    {
        cached.predict();
        continuousCached.predict(timeStep);
        unscented.predict();
        continuousUnscented.predict(timeStep);
        expectSame(cached, continuousCached, 1e-12);
        expectSame(unscented, continuousUnscented, 0.0);
        ASSERT_TRUE(cached.update(makeMarkovFix(z)));
        ASSERT_TRUE(continuousCached.update(makeMarkovFix(z)));
        ASSERT_TRUE(unscented.update(makeMarkovFix(z)));
        ASSERT_TRUE(continuousUnscented.update(makeMarkovFix(z), Emrys::reductions::collapse(), refilled));
        EXPECT_TRUE(refilled.accepted);
        expectSame(cached, continuousCached, 1e-12);
        expectSame(unscented, continuousUnscented, 0.0);
    }
}

// With a method per mode [switching-models] (imm-filter.md Q7 and Q14), each mode predicts by its
// model's own method where it carries one and by the filter's otherwise. On the identity chain the
// modes don't mix, so over four fixes each matches a Kalman filter with its model and method, to
// rounding: a constant-velocity mode under the filter's Linearized, and a drag mode given Unscented
// by withMethod(). The drag mode's Kalman filter under Linearized is 0.2 away in each velocity coordinate,
// so the match shows the mode took its own method. Fixed-step models with different methods, which a
// filter refused before Q14, match the same Kalman filters, and the filter's method is ModelsMethod.
// Under a chain that mixes, from one Gaussian shared by both modes, whose mixing gives that Gaussian
// back, the first prediction matches too.
TEST(ImmFilter, EachModePredictsByItsOwnMethod)
{
    using Emrys::FixedStepModel;
    using Emrys::withMethod;

    auto const start = makeGaussian(Vector4{0, 0, 1, -1}, Vector4{2, 2, 1, 1});
    auto const initial = HybridParams{CategoricalParams<Maneuver>{}, start};
    auto const identity = MarkovChain<Maneuver>{};

    auto filter = ImmFilter{
        SwitchingModel{identity, makeModel(quiet), withMethod(makeDragModel(agile), methods::unscented())},
        initial
    };
    static_assert(std::is_same_v<decltype(filter)::MethodType, methods::Linearized>);
    auto fixed = ImmFilter{
        SwitchingModel{
            identity,
            FixedStepModel{makeModel(quiet), State{}, timeStep},
            FixedStepModel{makeDragModel(agile), State{}, timeStep, methods::unscented()}
        },
        initial
    };
    static_assert(std::is_same_v<decltype(fixed)::MethodType, Emrys::ModelsMethod>);
    auto mixing = ImmFilter{
        SwitchingModel{
            MarkovChain<Maneuver>::persistent(0.9),
            makeModel(quiet),
            withMethod(makeDragModel(agile), methods::unscented())
        },
        initial
    };

    auto cruise = KalmanFilter{makeModel(quiet), start};
    auto drag = KalmanFilter{makeDragModel(agile), start, methods::unscented()};
    auto linearizedDrag = KalmanFilter{makeDragModel(agile), start};

    auto const expectMatches = [](auto const & imm, auto const & mode0, auto const & mode1, int step) {
        EXPECT_LE((imm.state().conditional(0).mean() - mode0.state().mean()).norm(), 1e-12) << step;
        EXPECT_LE((imm.state().conditional(0).covarianceLower() - mode0.state().covarianceLower()).norm(), 1e-12)
            << step;
        EXPECT_LE((imm.state().conditional(1).mean() - mode1.state().mean()).norm(), 1e-12) << step;
        EXPECT_LE((imm.state().conditional(1).covarianceLower() - mode1.state().covarianceLower()).norm(), 1e-12)
            << step;
    };

    mixing.predict(timeStep);
    filter.predict(timeStep);
    fixed.predict();
    cruise.predict(timeStep);
    drag.predict(timeStep);
    linearizedDrag.predict(timeStep);
    expectMatches(mixing, cruise, drag, 0);
    EXPECT_GE((linearizedDrag.state().mean() - drag.state().mean()).tail<2>().cwiseAbs().minCoeff(), 0.19);

    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, -0.9}, {2.0, -1.5}, {2.4, -2.5}, {3.3, -3.0}};
    for (int k = 0; k < int(fixes.size()); ++k)
    {
        if (k > 0)
        {
            filter.predict(timeStep);
            fixed.predict();
            cruise.predict(timeStep);
            drag.predict(timeStep);
        }
        expectMatches(filter, cruise, drag, k);
        expectMatches(fixed, cruise, drag, k);
        ASSERT_TRUE(filter.update(makeFix(fixes[std::size_t(k)])));
        ASSERT_TRUE(fixed.update(makeFix(fixes[std::size_t(k)])));
        ASSERT_TRUE(cruise.update(makeFix(fixes[std::size_t(k)])));
        ASSERT_TRUE(drag.update(makeFix(fixes[std::size_t(k)])));
        expectMatches(filter, cruise, drag, k);
        expectMatches(fixed, cruise, drag, k);
    }
}

// Mode probabilities that define no distribution, as a caller can leave them through the mutable
// state(), fail the update with their WeightError before any mode is updated ([update-outcomes]).
TEST(ImmFilter, ReturnsTheErrorOfModeProbabilitiesThatCannotBeNormalized)
{
    MarkovChain const chain = MarkovChain<Maneuver>::persistent(0.9);
    auto const conditionals = std::vector<GaussianParams<State>>{
        makeGaussian(Vector4::Zero(), Vector4{0.5, 0.5, 0.1, 0.1}),
        makeGaussian(Vector4::Zero(), Vector4{400, 400, 1, 1})
    };
    auto filter = ImmFilter{
        SwitchingModel{chain, makeModel(quiet), makeModel(agile)},
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{0.0, 0.0}), conditionals}
    };
    auto const update = filter.update(makeFix(Eigen::Vector2d{1, 0}));
    ASSERT_FALSE(update);
    EXPECT_EQ(update.error(), Emrys::WeightError::ZeroSum);
    EXPECT_EQ(filter.state().conditional<Cruise>().mean(), conditionals[0].mean());
    EXPECT_EQ(filter.state().conditional<Maneuvering>().mean(), conditionals[1].mean());
}

// Under a Gate, a mode whose update is rejected keeps its Gaussian and gets probability zero, the
// chain brings it back at the next prediction, and a fix rejected in every mode leaves the state
// unchanged.
TEST(ImmFilter, RejectedModesLoseTheirProbability)
{
    // A confident mode near the origin and a vague one: a fix 30 m away fails the gate only in the first.
    MarkovChain const chain = MarkovChain<Maneuver>::persistent(0.9);
    auto const conditionals = std::vector<GaussianParams<State>>{
        makeGaussian(Vector4::Zero(), Vector4{0.5, 0.5, 0.1, 0.1}),
        makeGaussian(Vector4::Zero(), Vector4{400, 400, 1, 1})
    };
    auto filter = ImmFilter{
        SwitchingModel{chain, makeModel(quiet), makeModel(agile)},
        HybridParams{CategoricalParams<Maneuver>{}, conditionals}
    };
    auto const before = filter.state();

    auto const update = filter.update(makeFix(Eigen::Vector2d{30, 0}), reductions::gate(0.997));
    ASSERT_TRUE(update);
    ASSERT_TRUE(update->accepted);
    EXPECT_FALSE(update->modes[0].accepted);
    EXPECT_TRUE(update->modes[1].accepted);
    EXPECT_EQ(filter.state().modes().toDensity().value().probability<Maneuvering>(), 1.0);
    EXPECT_EQ(filter.state().conditional<Cruise>().mean(), before.conditional<Cruise>().mean());
    EXPECT_GT(filter.state().conditional<Maneuvering>().mean()(0), 25.0);

    // The chain brings the rejected mode back at the next prediction.
    filter.predict(timeStep);
    EXPECT_NEAR(filter.state().modes().toDensity().value().probability<Cruise>(), 0.1, 1e-15);

    // A fix that fails in every mode is rejected and leaves the state unchanged.
    auto const state = filter.state();
    auto const far = filter.update(makeFix(Eigen::Vector2d{5000, 0}), reductions::gate(0.997, reductions::select()));
    ASSERT_TRUE(far);
    EXPECT_FALSE(far->accepted);
    EXPECT_TRUE((filter.state().modes().toDensity().value().logProbabilities()
                 == state.modes().toDensity().value().logProbabilities())
                    .all());
    EXPECT_EQ(filter.state().conditional(1).mean(), state.conditional(1).mean());
}

// Over a cruise, a 10-step sideways acceleration, and another cruise, the maneuvering mode's
// probability averages below 0.3 while cruising and above 0.8 while maneuvering, and the IMM's
// position error during the maneuver is under half a cruise-only Kalman filter's.
TEST(ImmFilter, DetectsAManeuver)
{
    // Truth cruises for 30 steps, accelerates sideways at 2 m/s^2 for 10, then cruises again.
    auto rng = std::mt19937_64{7};
    auto normal = std::normal_distribution<double>{};
    auto const noiseFactor = Eigen::LLT<Eigen::Matrix2d>{R};

    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 10, 0}, Vector4{10, 10, 4, 4});
    auto imm = ImmFilter{
        SwitchingModel{MarkovChain<Maneuver>::persistent(0.95), makeModel(quiet), makeModel(agile)},
        HybridParams{CategoricalParams<Maneuver>{}, initial}
    };
    auto cruiseOnly = KalmanFilter{makeModel(quiet), initial};

    auto truth = Vector4{0, 0, 10, 0};
    double quietManeuverProbability = 0;
    double activeManeuverProbability = 0;
    double immError = 0;
    double kalmanError = 0;
    for (int k = 1; k <= 60; ++k)
    {
        double const accel = k > 30 && k <= 40 ? 2.0 : 0.0;
        truth.head<2>() += truth.tail<2>() * timeStep + Eigen::Vector2d{0, 0.5 * accel};
        truth(3) += accel * timeStep;
        Eigen::Vector2d const z = truth.head<2>() + noiseFactor.matrixL() * Eigen::Vector2d{normal(rng), normal(rng)};

        imm.predict(timeStep);
        cruiseOnly.predict(timeStep);
        ASSERT_TRUE(imm.update(makeFix(z)));
        ASSERT_TRUE(cruiseOnly.update(makeFix(z)));

        double const maneuvering = imm.state().modes().toDensity().value().probability<Maneuvering>();
        if (k > 10 && k <= 30)
        {
            quietManeuverProbability += maneuvering / 20;
        }
        if (k > 32 && k <= 40)
        {
            activeManeuverProbability += maneuvering / 8;
            immError += (imm.state().toDensity()->collapsed().mean().head<2>() - truth.head<2>()).squaredNorm() / 8;
            kalmanError += (cruiseOnly.state().mean().head<2>() - truth.head<2>()).squaredNorm() / 8;
        }
    }
    EXPECT_LT(quietManeuverProbability, 0.3);
    EXPECT_GT(activeManeuverProbability, 0.8);
    EXPECT_LT(std::sqrt(immError), 0.5 * std::sqrt(kalmanError))
        << std::sqrt(immError) << " " << std::sqrt(kalmanError);
}

// With the sensor state as the mode and a 50-unit bias on five consecutive fixes, the outlier
// probability exceeds 0.99 during the burst, the IMM drifts under 10 units where a plain Kalman
// filter follows the bias past 30, recovers to within 3 afterwards, and the outlier probability
// falls back below 0.2.
TEST(ImmFilter, RidesOutABurstOfOutliers)
{
    // One model for every mode; the mode is the sensor's state, which persists. A 50 m bias
    // corrupts five fixes in a row. Outlier noise of 100 R still lets a biased fix pull the
    // estimate a little, so the IMM drifts by a few meters during the burst, where a Kalman
    // filter without an outlier model follows the bias.
    auto rng = std::mt19937_64{11};
    auto normal = std::normal_distribution<double>{};
    auto const noiseFactor = Eigen::LLT<Eigen::Matrix2d>{R};
    GaussianParams<State> const initial = makeGaussian(Vector4{0, 0, 10, 0}, Vector4{10, 10, 4, 4});
    auto filter = ImmFilter{
        SwitchingModel{MarkovChain<Sensor>::persistent(0.9), makeModel(quiet)},
        HybridParams{CategoricalParams<Sensor>::fromWeights(Sensor{}, Eigen::Vector2d{0.95, 0.05}), initial}
    };
    auto plain = KalmanFilter{makeModel(quiet), initial};

    auto truth = Vector4{0, 0, 10, 0};
    double worst = 0;
    double plainWorst = 0;
    double recovered = 0;
    for (int k = 1; k <= 40; ++k)
    {
        truth.head<2>() += truth.tail<2>() * timeStep;
        bool const burst = k > 20 && k <= 25;
        Eigen::Vector2d const z = truth.head<2>() + noiseFactor.matrixL() * Eigen::Vector2d{normal(rng), normal(rng)}
                                + (burst ? Eigen::Vector2d{0, 50} : Eigen::Vector2d::Zero());
        filter.predict(timeStep);
        ASSERT_TRUE(filter.update(makeMarkovFix(z)));
        plain.predict(timeStep);
        ASSERT_TRUE(plain.update(makeFix(z)));
        if (burst)
        {
            EXPECT_GT(filter.state().modes().toDensity().value().probability<Outlier>(), 0.99) << k;
        }
        double const error = (filter.state().toDensity()->collapsed().mean().head<2>() - truth.head<2>()).norm();
        if (k > 10)
        {
            worst = std::max(worst, error);
        }
        if (k > 30)
        {
            recovered = std::max(recovered, error);
        }
        plainWorst = std::max(plainWorst, (plain.state().mean().head<2>() - truth.head<2>()).norm());
    }
    EXPECT_LT(worst, 10.0);
    EXPECT_GT(plainWorst, 30.0);
    EXPECT_LT(recovered, 3.0);
    EXPECT_LT(filter.state().modes().toDensity().value().probability<Outlier>(), 0.2);
}

// A hybrid prints the moments over every mode, then each mode's Gaussian, nested, with the
// mode's probability, as the IMM filter's state.
TEST(HybridGaussian, PrintsEachModeNested)
{
    auto const conditionals = std::vector<GaussianParams<State>>{
        makeGaussian({0, 0, 1, 0}, {1, 1, 1, 1}),
        makeGaussian({1, 2, 0, 1}, {2, 1, 1, 1}),
        makeGaussian({4, -1, 1, 1}, {1, 3, 1, 1}),
        makeGaussian({-2, 3, 0, 0}, {1, 1, 2, 2})
    };
    auto const state =
        HybridParams{CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array4d{0.4, 0.1, 0.3, 0.2}), conditionals};
    std::string const text = printed(state);
    EXPECT_EQ(text.substr(0, text.find('\n')), "hybrid of 4 modes of dimension 4");
    std::string const third = "\n  mode 2, probability 0.3:\n    Gaussian of dimension 4\n      mean: 4 -1 1 1\n";
    EXPECT_NE(text.find(third), std::string::npos) << text;
    std::string const collapsed = printed(state.toDensity()->collapsed());
    std::string const moments =
        collapsed.substr(collapsed.find('\n'), collapsed.find("\n  mode") - collapsed.find('\n'));
    EXPECT_EQ(text.substr(text.find('\n'), moments.size()), moments);
}
