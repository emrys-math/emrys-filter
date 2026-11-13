// FixedStepModel (step 13d; [fixed-step-models], amending [continuous-time-noise]; [returned-jacobians]): the cached
// noise and transition are the closed form over the step, predict(prior, model) matches predict(method, prior, model,
// dt) under every method, over many steps and with inputs, a model caches no transition where the method or the model
// says not to, and a cached transition handles a periodic block across its seam. Headers under test:
// Models/ProcessModel.h.

#include <EmrysFilter/Models/ProcessModel.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <type_traits>

using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantAcceleration;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::OrnsteinUhlenbeck;
using Emrys::RadiansPi;
using Emrys::StationaryVariance;
using Emrys::TimeConstant;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Acceleration : Linear<2>
{};

struct Bias : Linear<1>
{};

using State = Composite<Position, Velocity, Acceleration, Bias>;

using Vector7 = Eigen::Matrix<double, 7, 1>;
using Matrix7 = Eigen::Matrix<double, 7, 7>;

constexpr double jerkDensity = 0.3;  // jerk spectral density
constexpr double tau = 20.0;         // bias time constant
constexpr double biasVar = 0.04;     // bias stationary variance
constexpr double timeStep = 0.5;

Matrix7
makeCov()
{
    auto a = Matrix7{};
    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            a(i, j) = 0.1 * (i + 1) - 0.07 * j + (i == j ? 1.0 : 0.0);
        }
    }
    return a * a.transpose() / 4 + 0.1 * Matrix7::Identity();
}

GaussianParams<State>
makeState()
{
    return {State{}, Vector7{10.0, -5.0, 1.0, 2.0, 0.1, -0.2, 0.3}, makeCov()};
}

auto
makeModel()
{
    return compose(
        ConstantAcceleration<Position, Velocity, Acceleration>{jerkDensity},
        OrnsteinUhlenbeck<Bias>{TimeConstant{tau}, StationaryVariance{biasVar}}
    );
}

// The closed-form transition and process noise of makeModel() over timeStep.
Matrix7
transitionMatrix()
{
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    Matrix7 f = Matrix7::Zero();
    f.block<2, 2>(0, 0) = i2;
    f.block<2, 2>(0, 2) = timeStep * i2;
    f.block<2, 2>(0, 4) = timeStep * timeStep / 2 * i2;
    f.block<2, 2>(2, 2) = i2;
    f.block<2, 2>(2, 4) = timeStep * i2;
    f.block<2, 2>(4, 4) = i2;
    f(6, 6) = std::exp(-timeStep / tau);
    return f;
}

Matrix7
processNoise()
{
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    auto const chain = Eigen::Matrix3d{
        {std::pow(timeStep, 5) / 20, std::pow(timeStep, 4) / 8, std::pow(timeStep, 3) / 6},
        {std::pow(timeStep, 4) / 8, std::pow(timeStep, 3) / 3, timeStep * timeStep / 2},
        {std::pow(timeStep, 3) / 6, timeStep * timeStep / 2, timeStep}
    };
    Matrix7 noise = Matrix7::Zero();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            noise.block<2, 2>(2 * i, 2 * j) = jerkDensity * chain(i, j) * i2;
        }
    }
    double const phi = std::exp(-timeStep / tau);
    noise(6, 6) = biasVar * (1 - phi * phi);
    return noise;
}

// The specific force an accelerometer reads, an uncertain input of the call, and gravity, a sure one.
struct SpecificForce : Linear<2>
{};

struct Gravity : Linear<2>
{};

// Velocity driven by a measured acceleration plus a known gravity vector; the accelerometer's noise
// is the uncertainty, so the model adds none.
template <typename P, typename V>
auto
accelerometerDriven()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Eigen::Matrix<T, 2, 1> const a =
                Emrys::block<SpecificForce>(in, x) + Emrys::block<Gravity>(in, x).template cast<T>();
            Emrys::block<P>(out, y) = Emrys::block<P>(in, x) + T(dt) * Emrys::block<V>(in, x) + T(dt * dt / 2) * a;
            Emrys::block<V>(out, y) = Emrys::block<V>(in, x) + T(dt) * a;
            return y;
        },
        Emrys::reads(Emrys::path<P>, Emrys::path<V>, Emrys::path<SpecificForce>, Emrys::path<Gravity>),
        Emrys::writes(Emrys::path<P>, Emrys::path<V>)
    );
}

template <typename Method>
class FixedStepMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<methods::Linearized, methods::Unscented<>>;

}  // namespace

TYPED_TEST_SUITE(FixedStepMethodTest, Methods);

// The model deduces its type, keeps its step, model, and method, and caches the closed-form noise
// and, under Linearized, the closed-form transition from the model's transition().
TEST(FixedStepModel, CachesTheClosedFormNoiseAndTransition)
{
    auto const model = FixedStepModel{makeModel(), State{}, timeStep};
    static_assert(
        std::is_same_v<decltype(model), FixedStepModel<State, decltype(makeModel()), methods::Linearized> const>
    );
    static_assert(
        Emrys::isFixedStep<std::remove_const_t<decltype(model)>> && !Emrys::isFixedStep<decltype(makeModel())>
    );
    static_assert(decltype(model)::CachesTransition);
    EXPECT_EQ(model.timeStep(), timeStep);
    EXPECT_LT((model.noise() - processNoise()).norm(), 1e-14);
    EXPECT_LT((model.transition() - transitionMatrix()).norm(), 1e-14);
    EXPECT_TRUE(model.partitions(State{}));

    Vector7 const x = makeState().mean();
    EXPECT_EQ(model.propagate(State{}, x), Emrys::propagate(State{}, x, makeModel(), timeStep));

    auto const unscented = FixedStepModel{makeModel(), State{}, timeStep, methods::unscented<0.5>()};
    static_assert(!decltype(unscented)::CachesTransition);
    static_assert(std::is_same_v<decltype(unscented)::MethodType, methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);
    EXPECT_LT((unscented.noise() - processNoise()).norm(), 1e-14);
}

// A filter's method parameter defaults to DefaultMethodFor its model, the model's own for a fixed
// step, and its constructor takes a MethodFor, which is ModelsMethod for a fixed step, so that a
// filter's type and a function forwarding its constructor's arguments can be spelled.
using Continuous = decltype(makeModel());
using FixedUnscented = FixedStepModel<State, Continuous, methods::Unscented<0.5>>;
static_assert(std::is_same_v<Emrys::DefaultMethodFor<Continuous>, methods::Linearized>);
static_assert(std::is_same_v<Emrys::DefaultMethodFor<FixedUnscented>, methods::Unscented<0.5>>);
static_assert(std::is_same_v<Emrys::MethodFor<Continuous, methods::Unscented<>>, methods::Unscented<>>);
static_assert(std::is_same_v<Emrys::MethodFor<FixedUnscented, methods::Unscented<0.5>>, Emrys::ModelsMethod>);

// Over twenty steps, predicting by the fixed-step model matches predicting by the method over the
// step. Linearized takes its cached transition, as the continuous prediction takes the model's
// transition(), so the two agree to rounding; the other methods take the same path with the same noise.
TYPED_TEST(FixedStepMethodTest, PredictionMatchesTheContinuousModel)
{
    auto const model = FixedStepModel{makeModel(), State{}, timeStep, TypeParam{}};
    GaussianParams<State> fixed = makeState();
    GaussianParams<State> continuous = makeState();
    for (int k = 0; k < 20; ++k)
    {
        fixed = Emrys::predict(fixed, model);
        continuous = Emrys::predict(TypeParam{}, continuous, makeModel(), timeStep);
    }
    double const scale = continuous.covarianceLower().norm();
    EXPECT_LT((fixed.mean() - continuous.mean()).norm(), 1e-12 * continuous.mean().norm());
    EXPECT_LT((fixed.covarianceLower() - continuous.covarianceLower()).norm(), 1e-12 * scale);
    EXPECT_EQ(fixed.covarianceLower(), fixed.covarianceLower().transpose());
}

// With an uncertain input (an accelerometer reading) and a known one (gravity), the prediction goes
// through the transform under every method, with the cached noise, and matches the continuous one
// exactly, since the arithmetic is the same.
TYPED_TEST(FixedStepMethodTest, PredictionTakesUncertainAndKnownInputs)
{
    using Inertial = Composite<Position, Velocity, Bias>;
    using Vector5 = Eigen::Matrix<double, 5, 1>;
    using Matrix5 = Eigen::Matrix<double, 5, 5>;
    auto const x = GaussianParams<Inertial>{
        Inertial{},
        Vector5{10.0, -5.0, 1.0, 2.0, 0.3},
        Matrix5{makeCov().topLeftCorner<5, 5>()}
    };
    auto const accelerometer =
        GaussianParams<SpecificForce>{Eigen::Vector2d{0.4, 9.6}, Eigen::Matrix2d{{0.02, 0.005}, {0.005, 0.03}}};
    auto const gravity = Emrys::sure<Gravity>(Eigen::Vector2d{0.0, -9.8});
    auto const continuous = compose(
        accelerometerDriven<Position, Velocity>(),
        OrnsteinUhlenbeck<Bias>{TimeConstant{tau}, StationaryVariance{biasVar}}
    );
    auto const model = FixedStepModel{continuous, Inertial{}, timeStep, TypeParam{}};

    auto const fixed = Emrys::predict(x, model, accelerometer, gravity);
    auto const expected = Emrys::predict(TypeParam{}, x, continuous, timeStep, accelerometer, gravity);
    EXPECT_EQ(fixed.mean(), expected.mean());
    EXPECT_EQ(fixed.covarianceLower(), expected.covarianceLower());

    auto const step = Emrys::fixedStep(continuous, timeStep, TypeParam{});
    auto const unbound = Emrys::predict(x, step, accelerometer, gravity);
    EXPECT_EQ(unbound.mean(), fixed.mean());
    EXPECT_EQ(unbound.covarianceLower(), fixed.covarianceLower());
}

// predict() and predictJoint() given fixedStep()'s result bind it to the prior's space and scalar,
// as a filter does, and predict as the FixedStepModel over them built by hand, bitwise; a float
// prior binds a step given as a double in float.
TEST(FixedStepModel, AnUnboundStepBindsToThePrior)
{
    auto const prior = makeState();
    auto const step = Emrys::fixedStep(makeModel(), timeStep);
    auto const byHand = FixedStepModel{makeModel(), State{}, timeStep};
    auto const predicted = Emrys::predict(prior, step);
    auto const expected = Emrys::predict(prior, byHand);
    EXPECT_EQ(predicted.mean(), expected.mean());
    EXPECT_EQ(predicted.covarianceLower(), expected.covarianceLower());

    auto const joint = Emrys::predictJoint(prior, step);
    auto const expectedJoint = Emrys::predictJoint(prior, byHand);
    EXPECT_EQ(joint.predicted.mean(), expectedJoint.predicted.mean());
    EXPECT_EQ(joint.cross, expectedJoint.cross);

    auto const single = compose(
        ConstantAcceleration<Position, Velocity, Acceleration, float>{static_cast<float>(jerkDensity)},
        Emrys::Constant<Bias>{}
    );
    auto const priorf = GaussianParams<State, float>{State{}, prior.mean().cast<float>(), makeCov().cast<float>()};
    auto const predictedf = Emrys::predict(priorf, Emrys::fixedStep(single, timeStep));
    static_assert(std::is_same_v<decltype(predictedf), GaussianParams<State, float> const>);
    auto const expectedf = Emrys::predict(
        priorf,
        FixedStepModel<State, decltype(single), methods::Linearized, float>{single, State{}, timeStep}
    );
    EXPECT_EQ(predictedf.mean(), expectedf.mean());
    EXPECT_EQ(predictedf.covarianceLower(), expectedf.covarianceLower());
}

// A constant-rate heading on a periodic block: the cached transition, from the model's transition(), carries
// a mean across the seam at pi as the transform at the mean does, and the mean comes back wrapped.
TEST(FixedStepModel, ACachedTransitionWrapsAPeriodicBlock)
{
    struct Heading : RadiansPi<1>
    {};

    struct Rate : Linear<1>
    {};

    using Turning = Composite<Heading, Rate>;
    auto const continuous = ConstantVelocity<Heading, Rate>{0.01};
    auto const model = FixedStepModel{continuous, Turning{}, timeStep};
    static_assert(decltype(model)::CachesTransition);
    auto const x =
        GaussianParams<Turning>{Turning{}, Eigen::Vector2d{3.0, 0.6}, Eigen::Matrix2d{{0.01, 0.002}, {0.002, 0.04}}};

    auto const fixed = Emrys::predict(x, model);
    auto const expected = Emrys::predict(methods::linearized(), x, continuous, timeStep);
    EXPECT_NEAR(fixed.mean()(0), 3.3 - 2 * std::numbers::pi, 1e-14);
    EXPECT_LT((fixed.mean() - expected.mean()).norm(), 1e-14);
    EXPECT_LT((fixed.covarianceLower() - expected.covarianceLower()).norm(), 1e-14);
}
