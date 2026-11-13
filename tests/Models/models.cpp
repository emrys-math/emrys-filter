// ProcessModel and Measurement (steps 1, 2; [templated-measurements], [method-per-measurement], [additive-noise],
// [weighted-branches], [continuous-time-noise], and [prediction-inputs]): blocks compose the transition and the noise
// and must partition the state, a constant-velocity block takes a density matrix, each method's prediction and
// hand-driven predict/update match the closed form, a partial measurement updates the full state, noise through the
// function matches additive noise where it adds, range-bearing on a tagged subspace, mixture noise gives one branch
// per component and favors the outlier component for an outlier, the point-wise log-likelihood, and a square-root
// prior.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory_resource>
#include <numbers>
#include <span>
#include <type_traits>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::ConstantAcceleration;
using Emrys::ConstantVelocity;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::logLikelihood;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::OrnsteinUhlenbeck;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::RandomWalk;
using Emrys::reads;
using Emrys::ScalarOf;
using Emrys::StationaryVariance;
using Emrys::TagPath;
using Emrys::TagPaths;
using Emrys::TimeConstant;

namespace methods = Emrys::methods;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Acceleration : Linear<2>
{};

struct Bias : Linear<1>
{};

// The noise of a position fix that passes through the function, an input of the call.
struct FixNoise : Linear<2>
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

// A position fix with a common bias on both axes: z = p + b [1 1]^T + v.
Eigen::Matrix<double, 2, 7> const H{{1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 0, 1}};

Eigen::Matrix2d const R{{0.5, 0.1}, {0.1, 0.4}};

GaussianParams<Linear<2>>
makeNoise()
{
    return {Eigen::Vector2d::Zero(), R};
}

// The fix's measurement function, h(p, b) = p + b [1 1]^T, of its input's Position and Bias. A test
// calls it from a function of its own, so it deduces its scalar rather than taking it as a template
// parameter.
auto const biasedPosition = [](auto const & in, auto const &, auto const & x) {
    using T = ScalarOf<decltype(x)>;
    return Eigen::Matrix<T, 2, 1>{block<Position>(in, x) + Eigen::Matrix<T, 2, 1>::Constant(block<Bias>(in, x)(0))};
};

// The fix as a model with additive noise.
auto
makeFixModel()
{
    return measurementModel(biasedPosition, reads(path<Position>, path<Bias>), additiveNoise(makeNoise()));
}

template <typename Method>
auto
makeFix(Eigen::Vector2d const & z, Method const & method)
{
    return observe(method, makeFixModel(), z);
}

struct KalmanStep
{
    Vector7 mean;
    Matrix7 cov;
    double logLikelihood;
    double nis;
};

KalmanStep
kalman(
    Vector7 const & mean,
    Matrix7 const & cov,
    Eigen::Vector2d const & z,
    Eigen::Matrix2d const & r,
    Eigen::Vector2d const & offset = Eigen::Vector2d::Zero()
)
{
    Eigen::Matrix2d const s = H * cov * H.transpose() + r;
    Eigen::Matrix<double, 7, 2> const k = cov * H.transpose() * s.inverse();
    Eigen::Vector2d const innovation = z - H * mean - offset;
    double const nis = innovation.dot(s.inverse() * innovation);
    return {
        mean + k * innovation,
        (Matrix7::Identity() - k * H) * cov,
        -0.5 * (2 * std::log(2 * pi) + std::log(s.determinant()) + nis),
        nis
    };
}

// The bias part of makeModel() with no Jacobian of its own and its noise over timeStep fixed, so that
// a model with it is differentiated.
auto
untransitioned()
{
    double const phi = std::exp(-timeStep / tau);
    return Emrys::processModel(
        []<typename T>(auto const &, auto const &, auto const & x, double dt) {
            return Eigen::Matrix<T, 1, 1>{T(std::exp(-dt / tau)) * x};
        },
        reads(path<Bias>),
        Emrys::writes(path<Bias>),
        additiveNoise(
            GaussianParams<Linear<1>>{
                Eigen::Matrix<double, 1, 1>::Zero(),
                Eigen::Matrix<double, 1, 1>{biasVar * (1 - phi * phi)}
            }
        )
    );
}

// Whether a fixed step over a process model caches its transition matrix in State.
template <typename Model>
constexpr bool hasTransition = Emrys::FixedStepModel<State, Model>::CachesTransition;

template <typename Method>
class ModelMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<methods::Linearized, methods::Unscented<>>;

}  // namespace

TYPED_TEST_SUITE(ModelMethodTest, Methods);

// A composition of a constant-acceleration model and an Ornstein-Uhlenbeck bias steps by the
// closed-form transition and has the closed-form discretized noise, and its parts write the whole
// state.
TEST(ProcessModel, PartsComposeTheTransitionAndNoise)
{
    auto const model = makeModel();
    Vector7 const x = makeState().mean();
    EXPECT_LT((Emrys::propagate(State{}, x, model, timeStep) - transitionMatrix() * x).norm(), 1e-14);
    auto const fixed = Emrys::FixedStepModel{model, State{}, timeStep};
    EXPECT_LT((fixed.noise() - processNoise()).norm(), 1e-14);
    EXPECT_TRUE(model.partitions(State{}));
}

// A fixed step caches the closed-form transition matrix when every part is affine and returns its
// Jacobian, and a model with a part that gives none caches none, so its Linearized prediction
// differentiates the model.
TEST(ProcessModel, TransitionMatrixFromTheParts)
{
    EXPECT_LT((Emrys::FixedStepModel{makeModel(), State{}, timeStep}.transition() - transitionMatrix()).norm(), 1e-15);

    auto const evaluated =
        compose(ConstantAcceleration<Position, Velocity, Acceleration>{jerkDensity}, untransitioned());
    static_assert(hasTransition<decltype(makeModel())> && !hasTransition<decltype(evaluated)>);
    auto const x = makeState();
    auto const predicted = Emrys::predict(methods::linearized(), x, evaluated, timeStep);
    Matrix7 const f = transitionMatrix();
    EXPECT_LT((predicted.mean() - f * x.mean()).norm(), 1e-12);
    EXPECT_LT((predicted.covarianceLower() - (f * x.covarianceLower() * f.transpose() + processNoise())).norm(), 1e-12);
}

// partitions() is false when a block is missing; two parts writing one block is a compile error
// (tests/Models/compile_errors.cpp, COMPOSE_WRITES_TWICE).
TEST(ProcessModel, PartsMustPartitionTheState)
{
    auto const missingBias = ConstantAcceleration<Position, Velocity, Acceleration>{jerkDensity};
    EXPECT_FALSE(missingBias.partitions(State{}));
}

// A model over a fixed-size space is checked from its types, so that a prediction with a missing
// block fails to compile in every build; a resizable space leaves the check to run time.
TEST(ProcessModel, PartitionKnownFromTheTypes)
{
    using Emrys::internal::StaticPartition;
    using Full = decltype(makeModel());
    using Missing = ConstantAcceleration<Position, Velocity, Acceleration>;
    using Evaluated =
        decltype(compose(ConstantAcceleration<Position, Velocity, Acceleration>{jerkDensity}, untransitioned()));
    static_assert(Full::staticPartition<State>() == StaticPartition::holds);
    static_assert(Missing::staticPartition<State>() == StaticPartition::fails);
    static_assert(Evaluated::staticPartition<State>() == StaticPartition::holds);

    struct Extra : Linear<Eigen::Dynamic, 4>
    {};

    using Resizable = Composite<Position, Velocity, Acceleration, Bias, Extra>;
    auto const allConstant = compose(
        Constant<Position>{},
        Constant<Velocity>{},
        Constant<Acceleration>{},
        Constant<Bias>{},
        Constant<Extra>{}
    );
    static_assert(decltype(allConstant)::staticPartition<Resizable>() == StaticPartition::unknown);
    EXPECT_TRUE(allConstant.partitions(Resizable{}));
}

// A composition's scalar is the one its parts state, through a nested composition too; a part that
// states none, such as Constant, joins any, and a composition of parts that state none has double.
TEST(ProcessModel, ACompositionHasItsPartsScalar)
{
    using Single = decltype(compose(
        compose(Constant<Bias>{}, RandomWalk<Acceleration, float>{0.1f}),
        ConstantVelocity<Position, Velocity, float>{0.1f}
    ));
    static_assert(std::is_same_v<Single::Scalar, float>);

    using Unstated = decltype(compose(Constant<Position>{}, Constant<Velocity>{}));
    static_assert(std::is_same_v<Unstated::Scalar, double>);
    auto const joined = compose(
        compose(Constant<Position>{}, Constant<Velocity>{}),
        RandomWalk<Acceleration, float>{0.1f},
        Constant<Bias>{}
    );
    static_assert(std::is_same_v<decltype(joined)::Scalar, float>);
    EXPECT_TRUE(joined.partitions(State{}));
}

// ConstantVelocity takes a full acceleration spectral density matrix and nested tag paths, giving
// the Q = [T^3/3, T^2/2; T^2/2, T] times density blocks and the right propagation.
TEST(ProcessModel, ConstantVelocityWithADensityMatrix)
{
    // White-noise acceleration correlated across axes, on a nested path.
    struct Platform : Composite<Position, Velocity>
    {};

    using Nested = Composite<Bias, Platform>;
    auto const density = Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}};
    auto const model = compose(
        ConstantVelocity<TagPath<Platform, Position>, TagPath<Platform, Velocity>>{density},
        RandomWalk<Bias>{0.01}
    );
    ASSERT_TRUE(model.partitions(Nested{}));
    auto const noise = Emrys::FixedStepModel{model, Nested{}, timeStep}.noise();
    EXPECT_NEAR(noise(0, 0), 0.01 * timeStep, 1e-15);
    EXPECT_LT((noise.block<2, 2>(1, 1) - std::pow(timeStep, 3) / 3 * density).norm(), 1e-15);
    EXPECT_LT((noise.block<2, 2>(1, 3) - timeStep * timeStep / 2 * density).norm(), 1e-15);
    EXPECT_LT((noise.block<2, 2>(3, 3) - timeStep * density).norm(), 1e-15);
    auto const x = Eigen::Matrix<double, 5, 1>{0.5, 1.0, 2.0, -1.0, 3.0};
    Eigen::Matrix<double, 5, 1> const moved = Emrys::propagate(Nested{}, x, model, timeStep);
    EXPECT_EQ(moved, (Eigen::Matrix<double, 5, 1>{0.5, 1.0 - timeStep, 2.0 + 3.0 * timeStep, -1.0, 3.0}));
}

// A random walk takes a block whose differences have another size than its values
// ([continuous-time-noise]): on a quaternion attitude it holds the four values and adds Q_c T to the three rotation
// vector rows, so the linearized and unscented predictions keep the mean and add the noise.
TEST(ProcessModel, RandomWalkOnAnAttitude)
{
    struct Attitude : Emrys::UnitQuaternion
    {};

    using Oriented = Composite<Attitude, Bias>;
    auto const density = Eigen::Matrix3d{{0.2, 0.05, 0.0}, {0.05, 0.1, 0.02}, {0.0, 0.02, 0.3}};
    auto const model = compose(RandomWalk<Attitude>{density}, RandomWalk<Bias>{0.01});
    static_assert(RandomWalk<Attitude>::Dimension == 3);
    static_assert(decltype(model)::staticPartition<Oriented>() == Emrys::internal::StaticPartition::holds);

    Eigen::Matrix4d expectedNoise = Eigen::Matrix4d::Zero();
    expectedNoise.topLeftCorner<3, 3>() = timeStep * density;
    expectedNoise(3, 3) = 0.01 * timeStep;
    auto const fixed = Emrys::FixedStepModel{model, Oriented{}, timeStep};
    EXPECT_LT((fixed.noise() - expectedNoise).norm(), 1e-15);
    EXPECT_EQ(fixed.transition(), Eigen::Matrix4d::Identity());

    auto const mean = Eigen::Matrix<double, 5, 1>{0.5, 0.5, 0.5, 0.5, 0.2};
    Eigen::Matrix4d const cov =
        Eigen::Matrix4d{{0.04, 0.01, 0.0, 0.0}, {0.01, 0.05, 0.0, 0.0}, {0.0, 0.0, 0.03, 0.0}, {0.0, 0.0, 0.0, 0.1}};
    auto const x = GaussianParams<Oriented>{mean, cov};
    auto const linearized = Emrys::predict(methods::linearized(), x, model, timeStep);
    EXPECT_EQ(linearized.mean(), mean);
    EXPECT_LT((linearized.covarianceLower() - (cov + expectedNoise)).norm(), 1e-15);
    auto const unscented = Emrys::predict(methods::unscented(), x, model, timeStep);
    EXPECT_LT((unscented.mean() - mean).norm(), 1e-12);
    EXPECT_LT((unscented.covarianceLower() - (cov + expectedNoise)).norm(), 1e-12);
}

// predict(method, x, model, T) matches F x and F P F' + Q for every method, since the model is
// linear.
TYPED_TEST(ModelMethodTest, PredictMatchesClosedForm)
{
    auto const x = makeState();
    auto const predicted = Emrys::predict(TypeParam{}, x, makeModel(), timeStep);
    static_assert(std::is_same_v<decltype(predicted), GaussianParams<State> const>);
    Matrix7 const f = transitionMatrix();
    EXPECT_LT((predicted.mean() - f * x.mean()).norm(), 1e-12);
    EXPECT_LT((predicted.covarianceLower() - (f * x.covarianceLower() * f.transpose() + processNoise())).norm(), 1e-12);
}

// predict() then update() with a position-plus-bias fix match a hand Kalman step: posterior,
// log-likelihood, NIS, innovation size, and a single branch of weight one.
TYPED_TEST(ModelMethodTest, HandDrivenPredictAndUpdateMatchClosedForm)
{
    // CA kinematics plus an OU bias, then a fix that reads only position and bias.
    auto const x = makeState();
    auto const predicted = Emrys::predict(TypeParam{}, x, makeModel(), timeStep);
    auto const z = Eigen::Vector2d{11.2, -3.7};
    auto const updated = Emrys::update(predicted, makeFix(z, TypeParam{}));
    ASSERT_TRUE(updated);
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(updated->branches)>,
                  std::array<Emrys::UpdateBranch<State, Linear<2>>, 1>>);

    Matrix7 const f = transitionMatrix();
    KalmanStep const expected = kalman(f * x.mean(), f * x.covarianceLower() * f.transpose() + processNoise(), z, R);
    auto const & branch = updated->branches[0];
    EXPECT_LT((branch.posterior.mean() - expected.mean).norm(), 1e-10);
    EXPECT_LT((branch.posterior.covarianceLower() - expected.cov).norm(), 1e-10);
    EXPECT_NEAR(updated->logLikelihood(), expected.logLikelihood, 1e-10);
    EXPECT_NEAR(branch.nis(), expected.nis, 1e-10);
    EXPECT_EQ(updated->innovationSize(), 2);
    EXPECT_EQ(branch.logWeight, 0.0);
    EXPECT_EQ(branch.logLikelihood(), updated->logLikelihood());
}

// A square-root form is an uncertain input by its root ([gaussian-forms]): predict() and update() of a
// SqrtGaussianParams prior match those of its covariance form for every method, and give the covariance form.
TYPED_TEST(ModelMethodTest, ASquareRootPriorMatchesItsCovarianceForm)
{
    auto const x = makeState();
    auto const sqrt = Emrys::toSqrt(x).value();
    auto const predicted = Emrys::predict(TypeParam{}, sqrt, makeModel(), timeStep);
    static_assert(std::is_same_v<decltype(predicted), GaussianParams<State> const>);
    auto const expected = Emrys::predict(TypeParam{}, x, makeModel(), timeStep);
    EXPECT_LT((predicted.mean() - expected.mean()).norm(), 1e-12);
    EXPECT_LT((predicted.covarianceLower() - expected.covarianceLower()).norm(), 1e-12);

    auto const z = Eigen::Vector2d{11.2, -3.7};
    auto const updated = Emrys::update(sqrt, makeFix(z, TypeParam{}));
    auto const reference = Emrys::update(x, makeFix(z, TypeParam{}));
    ASSERT_TRUE(updated);
    ASSERT_TRUE(reference);
    auto const & branch = updated->branches[0];
    EXPECT_LT((branch.posterior.mean() - reference->branches[0].posterior.mean()).norm(), 1e-12);
    EXPECT_LT((branch.posterior.covarianceLower() - reference->branches[0].posterior.covarianceLower()).norm(), 1e-12);
    EXPECT_NEAR(updated->logLikelihood(), reference->logLikelihood(), 1e-12);
}

// A measurement reading only Position and Bias updates the whole state identically to a
// transform linearized over the whole state.
TEST(Measurement, PartialUpdateMatchesAFullStateUpdate)
{
    // Linearizing only the blocks read gives the same update as linearizing over the whole state.
    auto const x = makeState();
    auto const z = Eigen::Vector2d{11.2, -3.7};
    auto const partial = Emrys::update(x, makeFix(z, methods::linearized()));

    auto const fullMeasurement = []<typename T>(auto const & s) {
        return Eigen::Matrix<T, 2, 1>{s.template head<2>() + Eigen::Matrix<T, 2, 1>::Constant(s(6))};
    };
    auto const full = Emrys::condition(
        x,
        Emrys::transform(
            Emrys::methods::linearized(),
            fullMeasurement,
            Emrys::additiveNoise(makeNoise()),
            Emrys::withCross(x)
        ),
        z
    );
    ASSERT_TRUE(partial);
    ASSERT_TRUE(full);
    EXPECT_LT((partial->branches[0].posterior.mean() - full->posterior.mean()).norm(), 1e-12);
    EXPECT_LT((partial->branches[0].posterior.covarianceLower() - full->posterior.covarianceLower()).norm(), 1e-12);
    EXPECT_NEAR(partial->logLikelihood(), full->logLikelihood(), 1e-12);
}

// additiveNoise(std::cref(noise)) refers to the noise rather than copying it, and
// observe(method, std::cref(model), z) to the model, and each updates exactly as the copy does.
TEST(Measurement, NoiseByReferenceMatchesNoiseByCopy)
{
    using Emrys::internal::PolicyKind;
    auto const x = makeState();
    auto const z = Eigen::Vector2d{11.2, -3.7};
    GaussianParams<Linear<2>> const noise = makeNoise();
    auto const byReference = observe(
        methods::linearized(),
        measurementModel(biasedPosition, reads(path<Position>, path<Bias>), additiveNoise(std::cref(noise))),
        z
    );
    auto const & referredNoise = byReference.model().template policy<PolicyKind::additiveNoise>().distribution();
    static_assert(std::is_same_v<decltype(referredNoise), GaussianParams<Linear<2>> const &>);
    EXPECT_EQ(&referredNoise, &noise);
    auto const referred = Emrys::update(x, byReference);
    auto const copied = Emrys::update(x, makeFix(z, methods::linearized()));
    ASSERT_TRUE(referred);
    ASSERT_TRUE(copied);
    EXPECT_EQ(referred->branches[0].posterior.mean(), copied->branches[0].posterior.mean());
    EXPECT_EQ(referred->branches[0].posterior.covarianceLower(), copied->branches[0].posterior.covarianceLower());
    EXPECT_EQ(referred->logLikelihood(), copied->logLikelihood());

    auto const model = makeFixModel();
    auto const byModelReference = observe(methods::linearized(), std::cref(model), z);
    EXPECT_EQ(&byModelReference.model(), &model);
    auto const modelReferred = Emrys::update(x, byModelReference);
    ASSERT_TRUE(modelReferred);
    EXPECT_EQ(modelReferred->branches[0].posterior.mean(), copied->branches[0].posterior.mean());
    EXPECT_EQ(modelReferred->logLikelihood(), copied->logLikelihood());
}

// An update computes its NIS and log-likelihood only when asked for [innovation-statistics]: computed later from the
// innovation they equal those an update that reads them computes with its gain, to rounding, and
// the innovation holds the residual and covariance that a hand Kalman step gives.
TEST(Measurement, StatisticsOnDemandMatchStatisticsWithTheGain)
{
    auto const x = makeState();
    auto const z = Eigen::Vector2d{11.2, -3.7};
    auto const later = Emrys::update(x, makeFix(z, methods::linearized()));
    auto const withGain = Emrys::internal::updateBranches<true>(x, makeFix(z, methods::linearized()));
    ASSERT_TRUE(later);
    ASSERT_TRUE(withGain);
    EXPECT_EQ(later->branches[0].posterior.mean(), withGain->branches[0].posterior.mean());
    EXPECT_NEAR(later->branches[0].nis(), withGain->branches[0].nis(), 1e-12);
    EXPECT_NEAR(later->logLikelihood(), withGain->logLikelihood(), 1e-12);

    KalmanStep const expected = kalman(x.mean(), x.covarianceLower(), z, R);
    Eigen::Matrix2d const s = H * x.covarianceLower() * H.transpose() + R;
    EXPECT_LT((later->branches[0].innovation.residual() - (z - H * x.mean())).norm(), 1e-12);
    EXPECT_LT((later->branches[0].innovation.covarianceLower() - s).norm(), 1e-12);
    EXPECT_NEAR(later->branches[0].nis(), expected.nis, 1e-10);
    EXPECT_NEAR(later->logLikelihood(), expected.logLikelihood, 1e-10);
}

// Noise the function reads as an input of the call, by its tag, gives the same update as
// additiveNoise() when the function adds it.
TEST(Measurement, InputNoiseMatchesAdditiveNoiseWhenItAdds)
{
    auto const x = makeState();
    auto const z = Eigen::Vector2d{11.2, -3.7};
    auto const throughModel = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 2, 1>{
                block<Position>(in, x) + Eigen::Matrix<T, 2, 1>::Constant(block<Bias>(in, x)(0))
                + block<FixNoise>(in, x)
            };
        },
        reads(path<Position>, path<Bias>, path<FixNoise>)
    );
    auto const noise = GaussianParams<FixNoise>{Eigen::Vector2d::Zero(), R};
    auto const byInput = Emrys::update(x, observe(methods::unscented(), throughModel, z, noise));
    auto const byAddition = Emrys::update(x, makeFix(z, methods::unscented()));
    ASSERT_TRUE(byInput);
    ASSERT_TRUE(byAddition);
    EXPECT_LT((byInput->branches[0].posterior.mean() - byAddition->branches[0].posterior.mean()).norm(), 1e-10);
    EXPECT_LT(
        (byInput->branches[0].posterior.covarianceLower() - byAddition->branches[0].posterior.covarianceLower()).norm(),
        1e-10
    );
    EXPECT_NEAR(byInput->logLikelihood(), byAddition->logLikelihood(), 1e-10);
}

// A measurement function that returns its Jacobian with its value (withJacobian(), [returned-jacobians]), one
// matrix over its input's differences, the two blocks read and then the noise it reads from the call, updates as
// the same function differentiated automatically does. The noise's scale depends on the position, so each block
// matters.
TEST(Measurement, AReturnedJacobianMatchesAutoDiff)
{
    auto const x = makeState();
    auto const z = Eigen::Vector2d{11.2, -3.7};
    // The function that returns its Jacobian calls this one, so it deduces its scalar.
    auto const measure = [](auto const & in, auto const &, auto const & x) {
        using T = ScalarOf<decltype(x)>;
        auto const p = block<Position>(in, x);
        return Eigen::Matrix<T, 2, 1>{
            p + Eigen::Matrix<T, 2, 1>::Constant(block<Bias>(in, x)(0)) + T(0.1) * p(0) * block<FixNoise>(in, x)
        };
    };
    auto const returning = [&](auto const & in, auto const & out, Eigen::Matrix<double, 5, 1> const & x) {
        Eigen::Vector2d const p = block<Position>(in, x);
        Eigen::Vector2d const v = block<FixNoise>(in, x);
        Eigen::Matrix<double, 2, 5> jacobian = Eigen::Matrix<double, 2, 5>::Zero();
        jacobian.leftCols<2>() = Eigen::Matrix2d::Identity();
        jacobian.col(0) += 0.1 * v;
        jacobian.col(2) = Eigen::Vector2d::Ones();
        jacobian.rightCols<2>() = 0.1 * p(0) * Eigen::Matrix2d::Identity();
        return Emrys::withJacobian(measure(in, out, x), jacobian);
    };
    auto const read = reads(path<Position>, path<Bias>, path<FixNoise>);
    auto const noise = GaussianParams<FixNoise>{Eigen::Vector2d::Zero(), R};
    auto const automatic = Emrys::update(x, observe(methods::linearized(), measurementModel(measure, read), z, noise));
    auto const supplied = Emrys::update(x, observe(methods::linearized(), measurementModel(returning, read), z, noise));
    ASSERT_TRUE(automatic);
    ASSERT_TRUE(supplied);
    EXPECT_LT((supplied->branches[0].posterior.mean() - automatic->branches[0].posterior.mean()).norm(), 1e-12);
    EXPECT_LT(
        (supplied->branches[0].posterior.covarianceLower() - automatic->branches[0].posterior.covarianceLower()).norm(),
        1e-12
    );
    EXPECT_NEAR(supplied->logLikelihood(), automatic->logLikelihood(), 1e-12);
}

// A range-bearing measurement on a nested tag path with a periodic bearing matches the hand EKF
// with the bearing residual wrapped, for an observation just across the seam, and its likelihood
// at a point is the noise's density at that wrapped residual, though the noise is over Linear<2>
// ([sensors]).
TEST(Measurement, RangeBearingOnATaggedSubspace)
{
    // Range and bearing to the platform's position, through a nested tag path, with a periodic bearing.
    struct Platform : Composite<Position, Velocity>
    {};

    struct Range : Linear<1>
    {};

    struct Bearing : RadiansPi<1>
    {};

    using Nested = Composite<Platform, Bias>;
    using RangeBearing = Composite<Range, Bearing>;
    Eigen::Matrix<double, 5, 5> const p0 = makeCov().topLeftCorner<5, 5>();
    auto const x = GaussianParams<Nested>{Nested{}, Eigen::Matrix<double, 5, 1>{-3.0, -0.1, 1.0, 0.5, 0.2}, p0};

    auto const z = Eigen::Vector2d{3.1, pi - 0.02};  // just across the seam from the predicted bearing
    auto const r = Eigen::Matrix2d{{0.01, 0.0}, {0.0, 0.0004}};
    auto const rangeBearing = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            return Eigen::Matrix<T, 2, 1>{
                Eigen::numext::sqrt(T(p.squaredNorm())),
                Eigen::numext::atan2(T(p(1)), T(p(0)))
            };
        },
        reads(path<Platform, Position>),
        outputSpace(RangeBearing{}),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), r})
    );
    auto const updated = Emrys::update(x, observe(methods::linearized(), rangeBearing, z));
    ASSERT_TRUE(updated);

    // The EKF by hand, with the bearing residual wrapped.
    Eigen::Vector2d const p = x.mean().head<2>();
    double const r2 = p.squaredNorm();
    Eigen::Matrix<double, 2, 5> h = Eigen::Matrix<double, 2, 5>::Zero();
    h.leftCols<2>() << p(0) / std::sqrt(r2), p(1) / std::sqrt(r2), -p(1) / r2, p(0) / r2;
    auto innovation = Eigen::Vector2d{z(0) - std::sqrt(r2), z(1) - std::atan2(p(1), p(0))};
    innovation(1) = std::remainder(innovation(1), 2 * pi);
    Eigen::Matrix2d const s = h * p0 * h.transpose() + r;
    Eigen::Matrix<double, 5, 2> const k = p0 * h.transpose() * s.inverse();
    ASSERT_LT(std::abs(innovation(1)), 0.1);
    EXPECT_LT((updated->branches[0].posterior.mean() - (x.mean() + k * innovation)).norm(), 1e-10);
    EXPECT_LT((updated->branches[0].posterior.covarianceLower() - (p0 - k * h * p0)).norm(), 1e-10);
    EXPECT_NEAR(updated->branches[0].nis(), innovation.dot(s.inverse() * innovation), 1e-10);

    double const atMean = logLikelihood(observe(rangeBearing, z), Nested{}, x.mean());
    double const expected =
        -0.5 * (2 * std::log(2 * pi) + std::log(r.determinant()) + innovation.dot(r.inverse() * innovation));
    EXPECT_NEAR(atMean, expected, 1e-10);
}

// Mixture noise [additive-noise], [weighted-branches] gives one branch per component from a single transform (the
// function is called once per sigma point), each with the Kalman posterior for its component and the posterior
// log-weight prior times likelihood over the evidence.
TEST(Measurement, MixtureNoiseGivesOneBranchPerComponent)
{
    // A narrow inlier and a wide outlier with an offset; one transform serves both.
    auto const x = makeState();
    auto const z = Eigen::Vector2d{11.2, -3.7};
    auto const offset = Eigen::Vector2d{1.5, -0.5};
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), offset, 25 * R);

    int calls = 0;
    auto const counted = [&calls](auto const & in, auto const & out, auto const & x) {
        ++calls;
        return biasedPosition(in, out, x);
    };
    auto const updated = Emrys::update(
        x,
        observe(
            methods::unscented(),
            measurementModel(counted, reads(path<Position>, path<Bias>), additiveNoise(noise)),
            z
        )
    );
    ASSERT_TRUE(updated);
    EXPECT_EQ(calls, 2 * 3 + 1);  // sigma points over the three coordinates read, once
    ASSERT_EQ(updated->branches.size(), 2u);

    KalmanStep const inlier = kalman(x.mean(), x.covarianceLower(), z, R);
    KalmanStep const outlier = kalman(x.mean(), x.covarianceLower(), z, 25 * R, offset);
    double const evidence = std::log(0.9 * std::exp(inlier.logLikelihood) + 0.1 * std::exp(outlier.logLikelihood));
    EXPECT_NEAR(updated->logLikelihood(), evidence, 1e-10);
    auto const expected = std::array<KalmanStep, 2>{inlier, outlier};
    auto const priorWeights = std::array<double, 2>{0.9, 0.1};
    for (std::size_t k = 0; k < 2; ++k)
    {
        auto const & branch = updated->branches[k];
        EXPECT_EQ(branch.component, Eigen::Index(k));
        EXPECT_EQ(branch.mode, 0);
        EXPECT_LT((branch.posterior.mean() - expected[k].mean).norm(), 1e-10);
        EXPECT_LT((branch.posterior.covarianceLower() - expected[k].cov).norm(), 1e-10);
        EXPECT_NEAR(branch.logLikelihood(), expected[k].logLikelihood, 1e-10);
        EXPECT_NEAR(branch.nis(), expected[k].nis, 1e-10);
        EXPECT_NEAR(branch.logWeight, std::log(priorWeights[k]) + expected[k].logLikelihood - evidence, 1e-10);
    }
}

// Branch weights sum to one; an observation near the prediction favors the inlier component and one
// far away the outlier component.
TEST(Measurement, AnOutlierFavorsTheOutlierComponent)
{
    auto const x = makeState();
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.95), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.05), Eigen::Vector2d::Zero(), 400 * R);
    Eigen::Vector2d const expected = H * x.mean();
    auto const fix = measurementModel(
        [](auto const & in, auto const &, Eigen::Vector3d const & x) {
            Eigen::Matrix<double, 2, 3> jacobian;
            jacobian << Eigen::Matrix2d::Identity(), Eigen::Vector2d::Ones();
            return Emrys::withJacobian(
                Eigen::Vector2d{block<Position>(in, x) + Eigen::Vector2d::Constant(block<Bias>(in, x)(0))},
                jacobian
            );
        },
        reads(path<Position>, path<Bias>),
        additiveNoise(noise)
    );
    auto const weights = [&](Eigen::Vector2d const & z) {
        auto const updated = Emrys::update(x, observe(methods::linearized(), std::cref(fix), z)).value();
        return Eigen::Vector2d{std::exp(updated.branches[0].logWeight), std::exp(updated.branches[1].logWeight)};
    };
    Eigen::Vector2d const near = weights(expected + Eigen::Vector2d{0.3, -0.2});
    Eigen::Vector2d const far = weights(expected + Eigen::Vector2d{40.0, 30.0});
    EXPECT_NEAR(near.sum(), 1.0, 1e-12);
    EXPECT_GT(near(0), 0.9);
    EXPECT_GT(far(1), 0.999);

    // With noise of one component the update's one branch is held in place, so the update has no
    // allocator but the default.
    auto const single = makeFixModel();
    auto const updated = Emrys::update(x, observe(methods::linearized(), std::cref(single), expected));
    ASSERT_TRUE(updated);
    EXPECT_EQ(updated->get_allocator(), Emrys::DefaultAllocator{});
}

// A result reused across updates moves like the update it is, keeping the allocator it was made
// with when the allocators are equal.
TEST(Measurement, AReusableUpdateMoves)
{
    using Result = Emrys::ReusableFilterUpdate<State, Linear<2>>;
    auto result = Result{};
    result.accepted = true;
    auto moved = Result{};
    moved = std::move(result);
    EXPECT_TRUE(moved.accepted);
    EXPECT_TRUE(moved.branches.empty());
}

// A result reused across updates copies into storage from another allocator with its fields.
TEST(Measurement, AReusableUpdateCopiesIntoAnotherAllocator)
{
    using Pmr = std::pmr::polymorphic_allocator<>;
    using Result = Emrys::ReusableFilterUpdate<State, Linear<2>, double, Pmr>;
    auto first = std::pmr::monotonic_buffer_resource{};
    auto second = std::pmr::monotonic_buffer_resource{};
    auto result = Result{Pmr{&first}};
    result.accepted = true;
    Emrys::internal::MixtureEvidenceAccess::setMixtureLogLikelihood(result, -1.5);
    auto const copy = Result{result, Pmr{&second}};
    EXPECT_EQ(result.get_allocator(), Pmr{&first});
    EXPECT_EQ(copy.get_allocator(), Pmr{&second});
    EXPECT_TRUE(copy.accepted);
    EXPECT_EQ(copy.logLikelihood(), -1.5);
}

// A process model's blocks cover the space's coordinates exactly once; a block whose rows fall
// outside the space fails the check, whatever the rest cover.
TEST(ProcessModel, RowsOutsideTheSpaceFailThePartition)
{
    auto coverage = Emrys::internal::RowCoverage<Linear<3>>{Linear<3>{}};
    coverage.push_back(Emrys::BlockRows{0, 3});
    EXPECT_TRUE(coverage.partitioned());
    coverage.push_back(Emrys::BlockRows{2, 2});
    EXPECT_FALSE(coverage.partitioned());
}

// Under Iterated each noise component's branch is its own iterated update, and a component whose
// innovation covariance can't be factorized fails the whole update.
TEST(Measurement, IteratedMixtureNoiseFailsWithAComponent)
{
    auto const x = makeState();
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), -1000 * R);
    auto const fix = measurementModel(biasedPosition, reads(path<Position>, path<Bias>), additiveNoise(noise));
    EXPECT_EQ(
        Emrys::update(x, observe(methods::iterated(methods::linearized()), std::cref(fix), Eigen::Vector2d{1.0, 2.0}))
            .error(),
        Emrys::GaussianError::NotPositiveDefinite
    );
}

namespace
{

// The range to the origin plus the bias, h(p, b) = |p| + b.
auto const biasedRange = []<typename T>(auto const & in, auto const &, auto const & x) {
    return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(block<Position>(in, x).squaredNorm())) + block<Bias>(in, x)(0)};
};

GaussianParams<Linear<1>>
rangeNoise()
{
    return {Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.25}};
}

}  // namespace

// logLikelihood(measurement, space, point) is the additive noise's density at the residual, or the
// model's own logLikelihood(g) when it has one, whatever else the call was given, and a model with
// its own likelihood updates like any other.
TEST(Measurement, LogLikelihoodAtAStatePoint)
{
    auto const point = Vector7{3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.5};
    auto const z = Eigen::Vector2d{3.5, 4.0};

    // From the additive noise's density at the residual.
    auto const fix = makeFix(z, methods::linearized());
    Eigen::Vector2d const residual = z - H * point;
    double const gaussian =
        -0.5 * (2 * std::log(2 * pi) + std::log(R.determinant()) + residual.dot(R.inverse() * residual));
    EXPECT_NEAR(logLikelihood(fix, State{}, point), gaussian, 1e-12);

    // The model's own, when it has one: Laplace noise with the Gaussian's scale, for filters that weight
    // state points.
    auto const laplace = measurementModel(
        biasedRange,
        reads(path<Position>, path<Bias>),
        additiveNoise(rangeNoise()),
        logLikelihood([](auto const & in, auto const & x, auto const & observed) {
            return -std::log(2 * 0.5)
                 - std::abs(observed(0) - block<Position>(in, x).norm() - block<Bias>(in, x)(0)) / 0.5;
        })
    );
    auto const range = observe(methods::linearized(), laplace, Eigen::Matrix<double, 1, 1>{6.0});
    EXPECT_NEAR(logLikelihood(range, State{}, point), -std::log(1.0) - 0.5 / 0.5, 1e-12);
    auto const joseph =
        observe(methods::linearized(), laplace, Eigen::Matrix<double, 1, 1>{6.0}, Emrys::updates::joseph());
    EXPECT_NEAR(logLikelihood(joseph, State{}, point), -std::log(1.0) - 0.5 / 0.5, 1e-12);

    // A model with its own likelihood updates like any other, by its additive noise.
    auto const updated = Emrys::update(makeState(), range);
    ASSERT_TRUE(updated);
    EXPECT_EQ(updated->branches.size(), 1u);
    EXPECT_TRUE(std::isfinite(updated->logLikelihood()));
}

// logLikelihoods() of a measurement whose observation holds its model by value, a temporary at the
// call, evaluates the noise it holds. The expected values are the 1-D Gaussian density with variance
// 0.25 at each residual.
TEST(Measurement, LogLikelihoodsOfAModelHeldByTheObservation)
{
    auto points = Eigen::Matrix<double, 7, 2>{};
    points.col(0) << 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.5;
    points.col(1) << 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, -1.0;
    Eigen::ArrayXd const result = Emrys::logLikelihoods(
        observe(
            measurementModel(biasedRange, reads(path<Position>, path<Bias>), additiveNoise(rangeNoise())),
            Eigen::Matrix<double, 1, 1>{6.0}
        ),
        State{},
        points
    );
    ASSERT_EQ(result.size(), 2);
    for (Eigen::Index j = 0; j < 2; ++j)
    {
        double const residual = 6.0 - points.col(j).head<2>().norm() - points(6, j);
        double const expected = -0.5 * (std::log(2 * pi) + std::log(0.25) + residual * residual / 0.25);
        EXPECT_NEAR(result(j), expected, 1e-12) << j;
    }
}

namespace
{

// The specific force an accelerometer reads, an uncertain input of the call, and gravity, a sure one.
struct SpecificForce : Linear<2>
{};

struct Gravity : Linear<2>
{};

// A model written for a vehicle: velocity is driven by a measured acceleration plus known gravity,
// rather than modeled. The accelerometer's noise is the uncertainty, so the model adds none.
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

}  // namespace

// predict() forwards an uncertain input (an accelerometer reading) and a known one (gravity) to a
// hand-written block [prediction-inputs], and the result matches the closed form with the input's covariance
// mapped through G.
TYPED_TEST(ModelMethodTest, PredictTakesUncertainAndKnownInputs)
{
    using Inertial = Composite<Position, Velocity, Bias>;
    using Vector5 = Eigen::Matrix<double, 5, 1>;
    using Matrix5 = Eigen::Matrix<double, 5, 5>;
    auto const p0 = Matrix5{makeCov().topLeftCorner<5, 5>()};
    auto const x = GaussianParams<Inertial>{Inertial{}, Vector5{10.0, -5.0, 1.0, 2.0, 0.3}, p0};
    auto const accelerometer =
        GaussianParams<SpecificForce>{Eigen::Vector2d{0.4, 9.6}, Eigen::Matrix2d{{0.02, 0.005}, {0.005, 0.03}}};
    auto const gravity = Eigen::Vector2d{0.0, -9.8};
    auto const model = compose(
        accelerometerDriven<Position, Velocity>(),
        OrnsteinUhlenbeck<Bias>{TimeConstant{tau}, StationaryVariance{biasVar}}
    );

    auto const predicted =
        Emrys::predict(TypeParam{}, x, model, timeStep, accelerometer, Emrys::sure<Gravity>(gravity));

    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    double const phi = std::exp(-timeStep / tau);
    Matrix5 f = Matrix5::Identity();
    f.block<2, 2>(0, 2) = timeStep * i2;
    f(4, 4) = phi;
    Eigen::Matrix<double, 5, 2> g = Eigen::Matrix<double, 5, 2>::Zero();
    g.topRows<2>() = timeStep * timeStep / 2 * i2;
    g.middleRows<2>(2) = timeStep * i2;
    Matrix5 q = Matrix5::Zero();
    q(4, 4) = biasVar * (1 - phi * phi);
    EXPECT_LT((predicted.mean() - (f * x.mean() + g * (accelerometer.mean() + gravity))).norm(), 1e-12);
    EXPECT_LT(
        (predicted.covarianceLower()
         - (f * p0 * f.transpose() + g * accelerometer.covarianceLower() * g.transpose() + q))
            .norm(),
        1e-12
    );
}
