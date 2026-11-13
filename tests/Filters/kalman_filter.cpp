// KalmanFilter (step 2, 2b, 13d; [reduction-policies], [templated-measurements], [filters-and-recursions],
// [composed-features], [fixed-step-models], [borrowed-intermediates], [upkeep]): deduction and accessors, predict and
// update match the closed form under every method, a fixed-step model fixes the step and the method, the in-place steps
// read only the covariance's lower triangle, upkeep repairs its covariance, measurements of different types and methods
// through one filter, Collapse
// matches the moment-matched mixture of the branches (including across a periodic seam), mixture noise and gates reject
// injected outliers, Gate per noise component, NisMonitor flags mistuned measurement noise, and NeesMonitor flags
// mistuned process noise on a target drawn by simulate().

#include <EmrysFilter/Models/Consistency.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/Simulation.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Upkeep.h>

#include "user_recursion.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::Gaussian;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::NeesMonitor;
using Emrys::NisMonitor;
using Emrys::NisThreshold;
using Emrys::RadiansPi;
using Emrys::ScalarOf;

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

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double accelDensity = 0.1;  // white-noise acceleration spectral density
constexpr double timeStep = 1.0;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{accelDensity};
}

GaussianParams<State>
makeInitial()
{
    auto const variances = Vector4{4.0, 4.0, 1.0, 1.0};
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, Matrix4{variances.asDiagonal()}};
}

Matrix4
transitionMatrix()
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

Matrix4
processNoise()
{
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    auto q = Matrix4{};
    q << std::pow(timeStep, 3) / 3 * i2, timeStep * timeStep / 2 * i2, timeStep * timeStep / 2 * i2, timeStep * i2;
    return accelDensity * q;
}

// A position fix: z = p + v.
Eigen::Matrix<double, 2, 4> const H{{1, 0, 0, 0}, {0, 1, 0, 0}};

Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

// A reduction output that holds one component, as a Kalman filter's does: what a policy kept of
// some branches, and its log-weight among them [reduction-policies].
struct KeptOne
{
    using ComponentType = GaussianParams<State>;

    std::optional<ComponentType> component;
    double logWeight = 0;

    void
    add(double weight, ComponentType kept)
    {
        component = std::move(kept);
        logWeight = weight;
    }
};

// What \a policy keeps of \a update's branches.
template <typename Policy, typename Update>
KeptOne
keptBy(Policy const & policy, Update const & update)
{
    auto out = KeptOne{};
    policy(update.branches, out);
    return out;
}

template <typename Method, typename Noise>
auto
makeFix(Eigen::Vector2d const & z, Method const & method, Noise noise)
{
    auto const position = [](auto const & in, auto const &, auto const & x) {
        return Emrys::block<Position>(in, x);
    };
    return observe(
        method,
        Emrys::measurementModel(position, Emrys::reads(Emrys::path<Position>), additiveNoise(std::move(noise))),
        z
    );
}

template <typename Method>
auto
makeFix(Eigen::Vector2d const & z, Method const & method)
{
    return makeFix(z, method, GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R});
}

// Range and bearing from the origin, with a periodic bearing.
struct Range : Linear<1>
{};

struct Bearing : RadiansPi<1>
{};

using RangeBearingSpace = Composite<Range, Bearing>;
Eigen::Matrix2d const rangeBearingNoise{{0.04, 0.0}, {0.0, 0.0009}};

template <typename Method>
auto
makeRangeBearing(Eigen::Vector2d const & z, Method const & method)
{
    auto const rangeBearing = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = Emrys::block<Position>(in, x);
            return Eigen::Matrix<T, 2, 1>{
                Eigen::numext::sqrt(T(p.squaredNorm())),
                Eigen::numext::atan2(T(p(1)), T(p(0)))
            };
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::outputSpace(RangeBearingSpace{}),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), rangeBearingNoise})
    );
    return observe(method, rangeBearing, z);
}

struct Moments4
{
    Vector4 mean;
    Matrix4 cov;
};

Moments4
predictByHand(Moments4 const & x)
{
    Matrix4 const f = transitionMatrix();
    return {f * x.mean, f * x.cov * f.transpose() + processNoise()};
}

// The Kalman update with a linear(ized) model: innovation r, Jacobian h, noise r.
Moments4
updateByHand(
    Moments4 const & x,
    Eigen::Vector2d const & innovation,
    Eigen::Matrix<double, 2, 4> const & h,
    Eigen::Matrix2d const & noise
)
{
    Eigen::Matrix2d const s = h * x.cov * h.transpose() + noise;
    Eigen::Matrix<double, 4, 2> const k = x.cov * h.transpose() * s.inverse();
    return {x.mean + k * innovation, (Matrix4::Identity() - k * h) * x.cov};
}

Moments4
rangeBearingByHand(Moments4 const & x, Eigen::Vector2d const & z)
{
    Eigen::Vector2d const p = x.mean.head<2>();
    double const r2 = p.squaredNorm();
    Eigen::Matrix<double, 2, 4> h = Eigen::Matrix<double, 2, 4>::Zero();
    h.leftCols<2>() << p(0) / std::sqrt(r2), p(1) / std::sqrt(r2), -p(1) / r2, p(0) / r2;
    auto innovation = Eigen::Vector2d{z(0) - std::sqrt(r2), z(1) - std::atan2(p(1), p(0))};
    innovation(1) = std::remainder(innovation(1), 2 * pi);
    return updateByHand(x, innovation, h, rangeBearingNoise);
}

template <typename Filter>
void
expectState(Filter const & filter, Moments4 const & expected, double tolerance)
{
    EXPECT_LT((filter.state().mean() - expected.mean).norm(), tolerance);
    EXPECT_LT((filter.state().covarianceLower() - expected.cov).norm(), tolerance);
}

template <typename Method>
class KalmanFilterMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<methods::Linearized, methods::Unscented<>>;

}  // namespace

TYPED_TEST_SUITE(KalmanFilterMethodTest, Methods);

// The deduction guides pick the space, model, and method, and name the filter by the Gaussian it
// evaluates while it holds the parameters [params-and-density]; state(), model(), and method() are
// accessible; the model's parameters can change between steps; setState() replaces the state; and
// a mutable state() changes it where it is.
TEST(KalmanFilter, DeducesItsTypeAndHoldsItsParts)
{
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    static_assert(std::is_same_v<decltype(filter), KalmanFilter<Gaussian<State>, decltype(makeModel())>>);
    static_assert(std::is_same_v<decltype(filter)::StateType, GaussianParams<State>>);
    static_assert(!Emrys::KalmanState<GaussianParams<State>>);
    auto const unscented = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(unscented.method())>, methods::Unscented<>>);
    EXPECT_EQ(filter.state().mean(), makeInitial().mean());
    EXPECT_EQ(filter.model().spectralDensity(), accelDensity * Eigen::Matrix2d::Identity());
    EXPECT_EQ(&std::as_const(filter).model(), &filter.model());
    EXPECT_EQ(&filter.recursion().model(), &filter.model());
    EXPECT_TRUE(filter.recursion().partitions(State{}));

    // The model's parameters can change between steps.
    filter.model() = ConstantVelocity<Position, Velocity>{2 * accelDensity};
    filter.predict(timeStep);
    Matrix4 const f = transitionMatrix();
    EXPECT_LT(
        (filter.state().covarianceLower() - (f * makeInitial().covarianceLower() * f.transpose() + 2 * processNoise()))
            .norm(),
        1e-12
    );

    filter.setState(makeInitial());
    EXPECT_EQ(filter.state().covarianceLower(), makeInitial().covarianceLower());

    // The state is changed where it is: a covariance inflated after a gap, with the mean kept.
    EXPECT_EQ(&filter.state(), &std::as_const(filter).state());
    Matrix4 const inflated = 4.0 * makeInitial().covarianceLower();
    filter.state().setCovariance(inflated);
    EXPECT_EQ(filter.state().covarianceLower(), inflated);
    EXPECT_EQ(filter.state().mean(), makeInitial().mean());
    filter.predict(timeStep);
    EXPECT_LT((filter.state().covarianceLower() - (f * inflated * f.transpose() + 2 * processNoise())).norm(), 1e-12);
}

// Over three fixes, every method's predict() and update() match the hand Kalman recursion in state,
// log-likelihood, and NIS, with one branch and the update accepted.
TYPED_TEST(KalmanFilterMethodTest, PredictAndUpdateMatchClosedForm)
{
    // Every method is exact for an affine model, in prediction and in update.
    auto filter = KalmanFilter{makeModel(), makeInitial(), TypeParam{}};
    auto expected = Moments4{makeInitial().mean(), makeInitial().covarianceLower()};
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.4}, {1.9, 1.1}, {3.2, 1.4}};
    for (Eigen::Vector2d const & z : fixes)
    {
        filter.predict(timeStep);
        auto const updated = filter.update(makeFix(z, TypeParam{}));
        ASSERT_TRUE(updated);
        static_assert(std::is_same_v<
                      std::remove_cvref_t<decltype(updated->branches)>,
                      std::array<Emrys::FilterBranch<Linear<2>>, 1>>);

        Moments4 const predicted = predictByHand(expected);
        Eigen::Matrix2d const s = H * predicted.cov * H.transpose() + R;
        Eigen::Vector2d const innovation = z - H * predicted.mean;
        double const logLikelihood =
            -0.5 * (2 * std::log(2 * pi) + std::log(s.determinant()) + innovation.dot(s.inverse() * innovation));
        expected = updateByHand(predicted, innovation, H, R);
        expectState(filter, expected, 1e-10);
        EXPECT_NEAR(updated->logLikelihood(), logLikelihood, 1e-10);
        EXPECT_NEAR(updated->branches[0].nis(), innovation.dot(s.inverse() * innovation), 1e-10);
        EXPECT_TRUE(updated->accepted);
    }
}

// With a FixedStepModel [fixed-step-models] the filter deduces the model's method, with its parameters, and
// predict() takes no step. Over three fixes it matches the hand Kalman recursion under Linearized,
// which predicts by the cached transition, and matches a filter predicting over the step exactly
// under Unscented, which takes the same path with the same noise.
TEST(KalmanFilter, AFixedStepModelFixesTheStepAndTheMethod)
{
    using Emrys::FixedStepModel;

    auto const cachedModel = FixedStepModel{makeModel(), State{}, timeStep};
    auto cached = KalmanFilter{cachedModel, makeInitial()};
    static_assert(std::is_same_v<
                  decltype(cached),
                  KalmanFilter<Gaussian<State>, std::remove_const_t<decltype(cachedModel)>, methods::Linearized>>);
    auto unscented =
        KalmanFilter{FixedStepModel{makeModel(), State{}, timeStep, methods::unscented<0.5>()}, makeInitial()};
    static_assert(std::is_same_v<decltype(unscented)::MethodType, methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);
    auto continuous = KalmanFilter{makeModel(), makeInitial(), methods::unscented<0.5>()};

    auto expected = Moments4{makeInitial().mean(), makeInitial().covarianceLower()};
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.4}, {1.9, 1.1}, {3.2, 1.4}};
    for (Eigen::Vector2d const & z : fixes)
    {
        cached.predict();
        ASSERT_TRUE(cached.update(makeFix(z, methods::linearized())));
        Moments4 const predicted = predictByHand(expected);
        expected = updateByHand(predicted, z - H * predicted.mean, H, R);
        expectState(cached, expected, 1e-10);

        unscented.predict();
        ASSERT_TRUE(unscented.update(makeFix(z, methods::unscented())));
        continuous.predict(timeStep);
        ASSERT_TRUE(continuous.update(makeFix(z, methods::unscented())));
        EXPECT_EQ(unscented.state().mean(), continuous.state().mean());
        EXPECT_EQ(unscented.state().covarianceLower(), continuous.state().covarianceLower());
    }
}

// The filter predicts and updates its state where it is [borrowed-intermediates] reading only the lower triangle of
// the covariance: from a correlated state whose strict upper triangle holds garbage, an update and
// a predict each match the hand recursion on the true covariance, and leave it symmetric.
TEST(KalmanFilter, StepsInPlaceFromTheLowerTriangle)
{
    auto const mean = Vector4{0.0, 0.0, 1.0, 0.5};
    Matrix4 cov = makeInitial().covarianceLower();
    cov(2, 0) = cov(0, 2) = 0.5;
    cov(3, 1) = cov(1, 3) = -0.3;
    Matrix4 garbage = cov;
    garbage.triangularView<Eigen::StrictlyUpper>().setConstant(99.0);

    // Updating first conditions the state as given, with no predict to have mirrored it.
    auto updatedFirst = KalmanFilter{makeModel(), GaussianParams<State>{State{}, mean, garbage}};
    auto const z = Eigen::Vector2d{1.2, 0.4};
    ASSERT_TRUE(updatedFirst.update(makeFix(z, methods::linearized())));
    Moments4 const expectedUpdate = updateByHand({mean, cov}, z - H * mean, H, R);
    expectState(updatedFirst, expectedUpdate, 1e-12);
    Matrix4 const updatedCov = updatedFirst.state().covarianceLower();
    EXPECT_LT((updatedCov - updatedCov.transpose()).norm(), 1e-14);

    auto predictedFirst = KalmanFilter{makeModel(), GaussianParams<State>{State{}, mean, garbage}};
    predictedFirst.predict(timeStep);
    expectState(predictedFirst, predictByHand({mean, cov}), 1e-12);
    Matrix4 const predictedCov = predictedFirst.state().covarianceLower();
    EXPECT_LT((predictedCov - predictedCov.transpose()).norm(), 1e-14);
}

// Upkeep applies to a Kalman filter's state() between steps [upkeep]: a covariance that has lost positive
// definiteness, here a velocity variance set below zero in place, is repaired by upkeep::repairCovariance() with
// a ridge on the diagonal, and the filter then predicts and updates with a state whose density can be read.
TEST(KalmanFilter, UpkeepRepairsItsCovariance)
{
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    Matrix4 indefinite = makeInitial().covarianceLower();
    indefinite(3, 3) = -1e-9;
    filter.state().setCovariance(indefinite);
    ASSERT_EQ(filter.state().toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);

    auto const repaired = Emrys::upkeep::repairCovariance()(filter.state());
    ASSERT_TRUE(repaired);
    EXPECT_TRUE(*repaired);
    Matrix4 const ridge = filter.state().covarianceLower() - indefinite;
    EXPECT_TRUE(ridge.triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0));
    EXPECT_TRUE((ridge.diagonal().array() > 0.0).all());
    EXPECT_GT(filter.state().covarianceLower()(3, 3), 0.0);
    EXPECT_TRUE(filter.state().toDensity().has_value());

    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeFix(Eigen::Vector2d{1.2, 0.4}, methods::linearized())));
    EXPECT_TRUE(filter.state().toDensity().has_value());
}

// One filter takes an affine fix and a linearized range-bearing in turn [templated-measurements], the same stream
// dispatched from a variant, and an unscented filter with the same measurements, all matching the
// hand EKF.
TEST(KalmanFilter, TakesMeasurementsOfDifferentTypesAndMethods)
{
    // A fix by an affine update, then range and bearing by an extended update, on one filter.
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    auto const fix = Eigen::Vector2d{1.2, 0.4};
    auto const rangeBearing = Eigen::Vector2d{2.4, 0.4};

    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeFix(fix, methods::linearized())));
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeRangeBearing(rangeBearing, methods::linearized())));

    Moments4 expected = predictByHand({makeInitial().mean(), makeInitial().covarianceLower()});
    expected = updateByHand(expected, fix - H * expected.mean, H, R);
    expected = rangeBearingByHand(predictByHand(expected), rangeBearing);
    expectState(filter, expected, 1e-10);

    // The same stream, mixed in a variant and dispatched by the caller.
    using Fix = decltype(makeFix(fix, methods::linearized()));
    using RangeBearing = decltype(makeRangeBearing(rangeBearing, methods::linearized()));
    auto const stream = std::vector<std::variant<Fix, RangeBearing>>{
        makeFix(fix, methods::linearized()),
        makeRangeBearing(rangeBearing, methods::linearized())
    };
    auto visited = KalmanFilter{makeModel(), makeInitial()};
    for (auto const & measurement : stream)
    {
        visited.predict(timeStep);
        std::visit(
            [&](auto const & m) {
                EXPECT_TRUE(visited.update(m));
            },
            measurement
        );
    }
    expectState(visited, expected, 1e-10);

    // An unscented fix (exact, since the fix is affine) between unscented predictions, then the
    // extended range and bearing.
    auto unscented = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    unscented.predict(timeStep);
    ASSERT_TRUE(unscented.update(makeFix(fix, methods::unscented())));
    unscented.predict(timeStep);
    ASSERT_TRUE(unscented.update(makeRangeBearing(rangeBearing, methods::linearized())));
    expectState(unscented, expected, 1e-10);
}

// Collapse returns the moment-matched Gaussian of the branches' posterior mixture, the filter
// collapses by default, and Select keeps the more probable branch.
TEST(Reduction, CollapseMatchesTheMixtureOfBranches)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.8), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.2), Eigen::Vector2d{2.0, -1.0}, 16 * R);
    auto prior = makeInitial();
    auto const updated = Emrys::update(prior, makeFix(Eigen::Vector2d{2.5, -0.5}, methods::linearized(), noise));
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 2u);

    auto mixture = GaussianMixtureParams<State>{State{}};
    for (auto const & branch : updated->branches)
    {
        mixture.add(branch.logWeight, branch.posterior.mean(), branch.posterior.covarianceLower());
    }
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    auto const expected = density->moments();
    auto const collapsed = *reductions::collapse()(updated->branches);
    EXPECT_LT((collapsed.mean() - expected.mean()).norm(), 1e-12);
    EXPECT_LT((collapsed.covarianceLower() - expected.covarianceLower()).norm(), 1e-12);

    // The filter collapses by default.
    auto filter = KalmanFilter{makeModel(), prior};
    auto const result = filter.update(makeFix(Eigen::Vector2d{2.5, -0.5}, methods::linearized(), noise));
    ASSERT_TRUE(result);
    EXPECT_EQ(result->branches.size(), 2u);
    EXPECT_LT((filter.state().mean() - expected.mean()).norm(), 1e-12);

    // Select keeps the more probable posterior.
    std::size_t const best = updated->branches[0].logWeight > updated->branches[1].logWeight ? 0u : 1u;
    auto const selected = *reductions::select()(updated->branches);
    EXPECT_EQ(selected.mean(), updated->branches[best].posterior.mean());
    EXPECT_EQ(selected.covarianceLower(), updated->branches[best].posterior.covarianceLower());
}

// Collapsing branches on either side of the heading seam gives mean pi and variance within plus
// between, measured around the circle; the linear block averages linearly.
TEST(Reduction, CollapseAveragesAcrossAPeriodicSeam)
{
    struct Heading : RadiansPi<1>
    {};

    using Pose = Composite<Position, Heading>;
    using Vector3 = Eigen::Vector3d;
    Eigen::Matrix3d const cov = 0.01 * Eigen::Matrix3d::Identity();
    auto update = Emrys::MeasurementUpdate<Pose, Linear<1>>{};
    auto const innovation =
        Emrys::Innovation<Linear<1>>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>::Identity()};
    update.branches.push_back(
        {std::log(0.5), GaussianParams<Pose>{Pose{}, Vector3{1.0, 0.0, pi - 0.1}, cov}, innovation, 0, 0}
    );
    update.branches.push_back(
        {std::log(0.5), GaussianParams<Pose>{Pose{}, Vector3{3.0, 0.0, -pi + 0.1}, cov}, innovation, 0, 1}
    );
    auto const collapsed = *reductions::collapse()(update.branches);
    EXPECT_NEAR(collapsed.mean()(0), 2.0, 1e-12);
    EXPECT_NEAR(std::abs(collapsed.mean()(2)), pi, 1e-12);
    EXPECT_NEAR(collapsed.covarianceLower()(0, 0), 0.01 + 1.0, 1e-12);
    EXPECT_NEAR(collapsed.covarianceLower()(2, 2), 0.01 + 0.01, 1e-12);
}

// Over 70 steps with a 60-unit outlier every seventh, a plain filter strays by more than 15 while
// Collapse and Select over inlier/outlier noise and a chi-squared Gate each stay within 5, with
// every outlier assigned to the outlier component or rejected.
TEST(KalmanFilter, MixturesAndGatesRejectInjectedOutliers)
{
    // A target at constant velocity, fixed every step; every seventh fix is off by 60 m. Mixture
    // noise rejects them by comparing components, and a chi-squared gate by their NIS alone.
    constexpr int steps = 70;
    constexpr double sigma = 1.0;
    Eigen::Matrix2d const inlierCov = sigma * sigma * Eigen::Matrix2d::Identity();
    auto robustNoise = GaussianMixtureParams<Linear<2>>{2};
    robustNoise.add(std::log(0.95), Eigen::Vector2d::Zero(), inlierCov);
    robustNoise.add(std::log(0.05), Eigen::Vector2d::Zero(), 1e4 * inlierCov);
    auto const plainNoise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), inlierCov};

    auto plain = KalmanFilter{makeModel(), makeInitial()};
    auto collapsing = KalmanFilter{makeModel(), makeInitial()};
    auto selecting = KalmanFilter{makeModel(), makeInitial()};
    auto gated = KalmanFilter{makeModel(), makeInitial()};

    auto rng = std::mt19937_64{17};
    auto gaussian = std::normal_distribution<double>{0.0, sigma};
    Vector4 truth = makeInitial().mean();
    double plainError = 0;
    double collapsingError = 0;
    double selectingError = 0;
    double gatedError = 0;
    int rejected = 0;
    for (int step = 1; step <= steps; ++step)
    {
        truth = transitionMatrix() * truth;
        bool const outlier = step % 7 == 0;
        Eigen::Vector2d const z = truth.head<2>() + Eigen::Vector2d{gaussian(rng), gaussian(rng)}
                                + (outlier ? Eigen::Vector2d{60.0, 0.0} : Eigen::Vector2d::Zero());

        plain.predict(timeStep);
        collapsing.predict(timeStep);
        selecting.predict(timeStep);
        gated.predict(timeStep);
        ASSERT_TRUE(plain.update(makeFix(z, methods::linearized(), plainNoise)));
        auto const soft = collapsing.update(makeFix(z, methods::linearized(), robustNoise));
        auto const hard = selecting.update(makeFix(z, methods::linearized(), robustNoise), reductions::select());
        auto const tested = gated.update(makeFix(z, methods::linearized(), plainNoise), reductions::gate(0.9999));
        ASSERT_TRUE(soft);
        ASSERT_TRUE(hard);
        ASSERT_TRUE(tested);
        if (outlier)
        {
            EXPECT_FALSE(tested->accepted) << "step " << step;
            EXPECT_GT(std::exp(soft->branches[1].logWeight), 0.999) << "step " << step;
            EXPECT_GT(hard->branches[1].logWeight, hard->branches[0].logWeight) << "step " << step;
            ++rejected;
        }
        if (step > 5)
        {
            plainError = std::max(plainError, (plain.state().mean().head<2>() - truth.head<2>()).norm());
            collapsingError = std::max(collapsingError, (collapsing.state().mean().head<2>() - truth.head<2>()).norm());
            selectingError = std::max(selectingError, (selecting.state().mean().head<2>() - truth.head<2>()).norm());
            gatedError = std::max(gatedError, (gated.state().mean().head<2>() - truth.head<2>()).norm());
        }
    }
    EXPECT_EQ(rejected, steps / 7);
    EXPECT_GT(plainError, 15.0);
    EXPECT_LT(collapsingError, 5.0);
    EXPECT_LT(selectingError, 5.0);
    EXPECT_LT(gatedError, 5.0);
}

// reductions::gate(p).threshold(k) is the chi-squared quantile and NisThreshold gives it directly; a near fix
// is accepted and updates as without the gate, a far one is rejected and leaves the prior, and a
// huge threshold accepts anything.
TEST(Gate, RejectsAnImprobableInnovation)
{
    auto const initial = KalmanFilter{makeModel(), makeInitial()};
    Eigen::Vector2d const expected = H * makeInitial().mean();
    EXPECT_NEAR(reductions::gate(0.99).threshold(2), -2 * std::log(0.01), 1e-12);
    EXPECT_EQ(reductions::gate(NisThreshold{4.0}).threshold(2), 4.0);
    // A gate given a float holds a float and computes its threshold in float unless asked.
    static_assert(std::is_same_v<decltype(reductions::gate(0.99f)), reductions::Gate<reductions::Collapse, float>>);
    static_assert(std::is_same_v<
                  decltype(reductions::gate(NisThreshold{4.0f}, reductions::select())),
                  reductions::Gate<reductions::Select, float>>);
    // The gate hands out the policy it reduces the passing branches by.
    auto const selecting = reductions::gate(0.99, reductions::select());
    [[maybe_unused]] reductions::Select const & inner = selecting.inner();
    static_assert(std::is_same_v<decltype(reductions::gate(0.99f).threshold(2)), float>);
    EXPECT_NEAR(reductions::gate(0.99f).threshold(2), -2 * std::log(0.01f), 1e-5f);
    EXPECT_NEAR(reductions::gate(0.99).threshold<float>(2), -2 * std::log(0.01f), 1e-5f);

    // Near the prediction: accepted, and the same update as without the gate.
    auto near = initial;
    auto const accepted =
        near.update(makeFix(expected + Eigen::Vector2d{1.0, -0.5}, methods::linearized()), reductions::gate(0.99));
    ASSERT_TRUE(accepted);
    EXPECT_TRUE(accepted->accepted);
    auto ungated = initial;
    ASSERT_TRUE(ungated.update(makeFix(expected + Eigen::Vector2d{1.0, -0.5}, methods::linearized())));
    EXPECT_EQ(near.state().mean(), ungated.state().mean());

    // Far from it: rejected, and the state keeps its prior.
    auto far = initial;
    auto const rejected =
        far.update(makeFix(expected + Eigen::Vector2d{30.0, 0.0}, methods::linearized()), reductions::gate(0.99));
    ASSERT_TRUE(rejected);
    EXPECT_FALSE(rejected->accepted);
    EXPECT_GT(rejected->branches[0].nis(), reductions::gate(0.99).threshold(2));
    EXPECT_EQ(far.state().mean(), makeInitial().mean());
    EXPECT_EQ(far.state().covarianceLower(), makeInitial().covarianceLower());

    // A threshold given directly.
    auto lenient = initial;
    EXPECT_TRUE(lenient
                    .update(
                        makeFix(expected + Eigen::Vector2d{30.0, 0.0}, methods::linearized()),
                        reductions::gate(NisThreshold{1e6})
                    )
                    .value()
                    .accepted);
}

// A gate over mixture noise tests each branch: both passing is the inner policy alone, only the
// wide component passing keeps it whatever the inner policy, with the weight of that branch alone,
// and none passing keeps nothing [reduction-policies].
TEST(Gate, GatesEachNoiseComponent)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), 400 * R);
    auto const prior = makeInitial();
    Eigen::Vector2d const expected = H * prior.mean();
    auto const fixAt = [&](Eigen::Vector2d const & offset) {
        return makeFix(expected + offset, methods::linearized(), noise);
    };

    // Both components pass: the same as the inner policy alone.
    auto const both = Emrys::update(prior, fixAt({0.5, 0.5}));
    ASSERT_TRUE(both);
    ASSERT_LE(both->branches[0].nis(), reductions::gate(0.997).threshold(2));
    auto const gatedBoth = keptBy(reductions::gate(0.997), *both);
    ASSERT_TRUE(gatedBoth.component);
    EXPECT_LT((gatedBoth.component->mean() - reductions::collapse()(both->branches)->mean()).norm(), 1e-15);
    EXPECT_NEAR(gatedBoth.logWeight, 0.0, 1e-15);

    // Only the wide component passes, so it alone is kept, whatever the inner policy.
    auto const one = Emrys::update(prior, fixAt({40.0, 0.0}));
    ASSERT_TRUE(one);
    double const limit = reductions::gate(0.997).threshold(2);
    ASSERT_GT(one->branches[0].nis(), limit);
    ASSERT_LE(one->branches[1].nis(), limit);
    auto const gatedOne = keptBy(reductions::gate(0.997), *one);
    ASSERT_TRUE(gatedOne.component);
    EXPECT_EQ(gatedOne.component->mean(), one->branches[1].posterior.mean());
    EXPECT_EQ(gatedOne.logWeight, one->branches[1].logWeight);
    auto const selectedOne = keptBy(reductions::Gate(0.997, reductions::select()), *one);
    ASSERT_TRUE(selectedOne.component);
    EXPECT_EQ(selectedOne.component->covarianceLower(), one->branches[1].posterior.covarianceLower());
    EXPECT_EQ(selectedOne.logWeight, one->branches[1].logWeight);

    // Neither passes: nothing is kept.
    EXPECT_FALSE(keptBy(reductions::gate(0.997), Emrys::update(prior, fixAt({4000.0, 0.0})).value()).component);
}

// A policy of the user's own may add what it keeps to the filter's output rather than return it:
// the filter keeps what it added, and rejects the measurement when it adds nothing; and
// reduceToOne(), by which a recursion of the user's own runs a policy, runs either form
// [reduction-policies], [user-extension-points].
TEST(Reduction, APolicyMayAddToTheOutput)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), 400 * R);
    auto const prior = makeInitial();
    auto const fix = makeFix(H * prior.mean() + Eigen::Vector2d{1.0, -0.5}, methods::linearized(), noise);
    auto const updated = Emrys::update(prior, fix);
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 2u);

    auto const keepWide = [](auto && branches, auto & out) {
        out.add(branches[1].logWeight, std::move(branches[1].posterior));
    };
    auto filter = KalmanFilter{makeModel(), prior};
    auto const result = filter.update(fix, keepWide);
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->accepted);
    EXPECT_EQ(filter.state().mean(), updated->branches[1].posterior.mean());
    EXPECT_EQ(filter.state().covarianceLower(), updated->branches[1].posterior.covarianceLower());

    auto const keepNone = [](auto &&, auto &) {};
    auto rejecting = KalmanFilter{makeModel(), prior};
    auto const rejected = rejecting.update(fix, keepNone);
    ASSERT_TRUE(rejected);
    EXPECT_FALSE(rejected->accepted);
    EXPECT_EQ(rejecting.state().mean(), prior.mean());

    using Component = GaussianParams<State>;
    auto const wide = Emrys::reduceToOne<Component>(keepWide, updated->branches);
    ASSERT_TRUE(wide);
    EXPECT_EQ(wide->mean(), updated->branches[1].posterior.mean());
    auto const collapsed = Emrys::reduceToOne<Component>(reductions::collapse(), updated->branches);
    ASSERT_TRUE(collapsed);
    EXPECT_EQ(collapsed->mean(), reductions::collapse()(updated->branches)->mean());
    EXPECT_FALSE(Emrys::reduceToOne<Component>(keepNone, updated->branches));
}

// Over 200 steps of a target driven by the model's noise, the monitor of the correctly tuned filter
// is consistent at 99.9% while one assuming 1% of R is overconfident (sum above the bound) and one
// assuming 100 R is underconfident (sum below).
TEST(NisMonitor, FlagsMistunedMeasurementNoise)
{
    // A target driven by the model's own process noise, with fixes whose noise is R. A filter
    // that assumes R is consistent; one that assumes much less or much more is not.
    auto rng = std::mt19937_64{23};
    auto standard = std::normal_distribution<double>{0.0, 1.0};
    auto const draw = [&]<int N>(Eigen::Matrix<double, N, N> const & cov) {
        auto z = Eigen::Matrix<double, N, 1>{};
        for (int i = 0; i < N; ++i)
        {
            z(i) = standard(rng);
        }
        return Eigen::Matrix<double, N, 1>{cov.llt().matrixL() * z};
    };
    auto const scales = std::array<double, 3>{1.0, 0.01, 100.0};
    std::vector<KalmanFilter<Gaussian<State>, decltype(makeModel())>> filters(
        3,
        KalmanFilter{makeModel(), makeInitial()}
    );
    auto monitors = std::array<NisMonitor<50>, 3>{};

    Vector4 truth = makeInitial().mean();
    for (int step = 0; step < 200; ++step)
    {
        truth = transitionMatrix() * truth + draw(processNoise());
        Eigen::Vector2d const z = truth.head<2>() + draw(R);
        for (std::size_t i = 0; i < 3; ++i)
        {
            filters[i].predict(timeStep);
            monitors[i].add(
                filters[i]
                    .update(makeFix(
                        z,
                        methods::linearized(),
                        GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), scales[i] * R}
                    ))
                    .value()
            );
        }
    }
    EXPECT_EQ(monitors[0].degreesOfFreedom(), 100);
    EXPECT_EQ(monitors[0].verdict(0.999), Emrys::Consistency::Consistent) << monitors[0].average().value();
    EXPECT_EQ(monitors[1].verdict(0.999), Emrys::Consistency::Overconfident);
    EXPECT_EQ(monitors[2].verdict(0.999), Emrys::Consistency::Underconfident);
}

// Over 200 independent runs of 20 steps, each a target drawn by simulate() from the model with fixes
// drawn from R, the NEES of each run's last estimate, summed by a Cumulative monitor (the Monte Carlo
// test), is consistent at 99.9% for the filter assuming that model, while one assuming 1% of its
// process noise is overconfident and one assuming 100 times it is underconfident. The position fixes
// constrain the position whatever the process noise, so most of what the NEES sees is the velocity,
// which no fix measures. The runs are independent because one run's errors are correlated from step
// to step, which a sum over its steps would take for chi-squared.
TEST(NeesMonitor, FlagsMistunedProcessNoise)
{
    auto rng = std::mt19937_64{29};
    auto const model = makeModel();
    auto const fixNoise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R};
    auto const fixDensity = fixNoise.toDensity().value();
    auto const initial = makeInitial();
    auto const scales = std::array<double, 3>{1.0, 0.01, 100.0};
    auto monitors = std::array<NeesMonitor<Emrys::Cumulative>, 3>{};
    for (int run = 0; run < 200; ++run)
    {
        auto filters = std::vector<KalmanFilter<Gaussian<State>, decltype(makeModel())>>{};
        for (double const scale : scales)
        {
            filters.emplace_back(ConstantVelocity<Position, Velocity>{scale * accelDensity}, initial);
        }
        Vector4 truth = Emrys::drawSamples(initial.toDensity().value(), 1, rng);
        for (int step = 0; step < 20; ++step)
        {
            truth = Emrys::simulate(State{}, truth, model, rng, timeStep);
            Eigen::Vector2d const z = truth.head<2>() + Emrys::drawSamples(fixDensity, 1, rng);
            for (auto & filter : filters)
            {
                filter.predict(timeStep);
                EXPECT_TRUE(filter.update(makeFix(z, methods::linearized())).value().accepted);
            }
        }
        for (std::size_t i = 0; i < 3; ++i)
        {
            monitors[i].add(truth, filters[i].state());
        }
    }
    EXPECT_EQ(monitors[0].degreesOfFreedom(), 800);
    EXPECT_EQ(monitors[0].verdict(0.999), Emrys::Consistency::Consistent) << monitors[0].average().value();
    EXPECT_EQ(monitors[1].verdict(0.999), Emrys::Consistency::Overconfident);
    EXPECT_EQ(monitors[2].verdict(0.999), Emrys::Consistency::Underconfident);
}

// A recursion of the user's own, FadingMemory over a KalmanRecursion (tests/user_recursion.h), meets Recursion by its
// interface alone, is deduced from the constructor as the filter's last argument, and runs the filter's steps: after a
// prediction the covariance is the plain filter's times alpha squared, and the update is the plain filter's on that
// prior [filters-and-recursions], [user-extension-points].
TEST(KalmanFilter, UserRecursionRunsTheSteps)
{
    using Model = decltype(makeModel());
    using Inner = Emrys::KalmanRecursion<Gaussian<State>, Model>;
    using Fading = EmrysTest::FadingMemory<Inner>;
    static_assert(Emrys::Recursion<Inner>);
    static_assert(Emrys::Recursion<Fading>);
    static_assert(Emrys::RecursionOver<Fading, Gaussian<State>>);
    static_assert(!Emrys::RecursionOver<Fading, Gaussian<Position>>);
    static_assert(!Emrys::Recursion<Model>);
    static_assert(!Emrys::Recursion<GaussianParams<State>>);

    constexpr double alpha = 1.1;
    auto filter = KalmanFilter{EmrysTest::FadingMemory{Inner{makeModel()}, alpha}, makeInitial()};
    static_assert(std::is_same_v<decltype(filter), KalmanFilter<Gaussian<State>, Model, methods::Linearized, Fading>>);
    static_assert(std::is_same_v<decltype(filter)::RecursionType, Fading>);
    EXPECT_DOUBLE_EQ(filter.recursion().alpha(), alpha);
    auto plain = KalmanFilter{makeModel(), makeInitial()};
    static_assert(std::is_same_v<decltype(plain), KalmanFilter<Gaussian<State>, Model>>);

    filter.predict(timeStep);
    plain.predict(timeStep);
    EXPECT_LT((filter.state().mean() - plain.state().mean()).norm(), 1e-12);
    EXPECT_LT((filter.state().covarianceLower() - alpha * alpha * plain.state().covarianceLower()).norm(), 1e-12);

    plain.setState(filter.state());
    auto const z = Eigen::Vector2d{1.2, 0.4};
    auto const updated = filter.update(makeFix(z, methods::linearized()));
    auto const reference = plain.update(makeFix(z, methods::linearized()));
    ASSERT_TRUE(updated && reference);
    EXPECT_TRUE(updated->accepted);
    EXPECT_DOUBLE_EQ(updated->logLikelihood(), reference->logLikelihood());
    EXPECT_LT((filter.state().mean() - plain.state().mean()).norm(), 1e-12);
    EXPECT_LT((filter.state().covarianceLower() - plain.state().covarianceLower()).norm(), 1e-12);
}

// The policies meet ReductionPolicy, the contract the recursion's update() requires of its policy: Collapse, Select,
// and a Gate over them reduce the branches to one GaussianParams; Keep adds every branch to an output that holds
// several, which only a mixture filter's does (MixtureReductionPolicy); a policy of the user's own that adds to the
// output meets both; and a callable of the wrong result, or none, meets neither [reduction-policies],
// [user-extension-points].
TEST(KalmanFilter, PoliciesMeetReductionPolicy)
{
    [[maybe_unused]] auto const prior = makeInitial();
    [[maybe_unused]] auto const fix = makeFix(Eigen::Vector2d{1.2, 0.4}, methods::linearized());
    using Measured = decltype(Emrys::update(prior, fix))::value_type;
    using Component = GaussianParams<State>;
    static_assert(Emrys::ReductionPolicy<reductions::Collapse, Measured, Component>);
    static_assert(Emrys::ReductionPolicy<reductions::Select, Measured, Component>);
    static_assert(Emrys::ReductionPolicy<reductions::Gate<reductions::Select>, Measured, Component>);
    static_assert(!Emrys::ReductionPolicy<Emrys::reductions::Keep, Measured, Component>);
    static_assert(!Emrys::ReductionPolicy<reductions::Collapse, Measured, GaussianParams<Position>>);
    static_assert(Emrys::MixtureReductionPolicy<reductions::Collapse, Measured, Gaussian<State>>);
    static_assert(Emrys::MixtureReductionPolicy<Emrys::reductions::Keep, Measured, Gaussian<State>>);
    static_assert(Emrys::MixtureReductionPolicy<reductions::Gate<Emrys::reductions::Keep>, Measured, Gaussian<State>>);
    auto const selecting = [](auto const & branches) {
        return reductions::select()(branches);
    };
    static_assert(Emrys::ReductionPolicy<decltype(selecting), Measured, Component>);
    auto const keepFirst = [](auto && branches, auto & out) {
        out.add(branches[0].logWeight, branches[0].posterior);
    };
    static_assert(Emrys::ReductionPolicy<decltype(keepFirst), Measured, Component>);
    static_assert(Emrys::MixtureReductionPolicy<decltype(keepFirst), Measured, Gaussian<State>>);
    auto const mean = [](auto const & branches) {
        return branches[0].posterior.mean();
    };
    static_assert(!Emrys::ReductionPolicy<decltype(mean), Measured, Component>);
    static_assert(!Emrys::MixtureReductionPolicy<decltype(mean), Measured, Gaussian<State>>);
    static_assert(!Emrys::ReductionPolicy<int, Measured, Component>);
    SUCCEED();
}
