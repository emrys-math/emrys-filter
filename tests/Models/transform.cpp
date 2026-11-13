// transform() and the methods Linearized and Unscented of Models/Transform.h (steps 1, 2, and 17zb;
// [method-per-measurement], [returned-jacobians], [transform-policies], [whole-state-models]): a linear step matches
// the closed form, with its inputs matched to reads() by tag and a sure input; the cross-covariance with a marked
// input, which may be any input; condition() is the Kalman update and takes an observation; autodiff Jacobians match
// one a function returns over its whole input; the unscented transform is exact for a quadratic and averages across a
// periodic seam; the output space is outputSpace()'s, the Composite of writes(), the noise's, or a Linear sized by the
// function; additiveNoise() adds to the output; a measurement model's predicted measurement is transform() of it, with
// affineIn() and additiveNoise() as observe() takes them; and densities, marginals, and the square-root form are
// accepted as inputs.

#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <tuple>
#include <type_traits>
#include <utility>

using Emrys::additiveNoise;
using Emrys::affineIn;
using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::reads;
using Emrys::ScalarOf;
using Emrys::sure;
using Emrys::TransformMethod;
using Emrys::withCross;
using Emrys::writes;

namespace methods = Emrys::methods;

namespace
{

// The methods a sensor can take, and what it can't.
static_assert(
    TransformMethod<methods::Linearized> && TransformMethod<methods::Unscented<>>
    && TransformMethod<Emrys::methods::Cubature>
);
static_assert(TransformMethod<Emrys::methods::GaussHermite<3>> && TransformMethod<Emrys::methods::Iterated<>>);
static_assert(TransformMethod<methods::Linearized const &>);
static_assert(!TransformMethod<double> && !TransformMethod<GaussianParams<Linear<1>>>);

// A mark of a temporary would dangle, so withCross() takes lvalues only.
template <typename Input>
concept MarkableAsIs = requires(Input && input) { withCross(std::forward<Input>(input)); };
static_assert(MarkableAsIs<GaussianParams<Linear<1>> const &> && !MarkableAsIs<GaussianParams<Linear<1>>>);

// The marks have public names, so a function that forwards one to transform() can spell its type.
static_assert(std::is_same_v<
              decltype(withCross(std::declval<GaussianParams<Linear<1>> const &>())),
              Emrys::CrossMarked<GaussianParams<Linear<1>>>>);

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct State : Composite<Position, Velocity>
{};

struct ProcessNoise : Linear<2>
{};

struct Control : Linear<4>
{};

struct Fix : Linear<2>
{};

struct SensorPosition : Linear<2>
{};

struct ScaleNoise : Linear<1>
{};

struct Range : Linear<1>
{};

Eigen::Matrix4d
makeCov()
{
    return Eigen::Matrix4d{{1.0, 0.2, 0.1, 0.0}, {0.2, 2.0, -0.1, 0.3}, {0.1, -0.1, 0.5, 0.05}, {0.0, 0.3, 0.05, 0.8}};
}

GaussianParams<State>
makeState()
{
    return {State{}, Eigen::Vector4d{1.0, -2.0, 0.5, 0.25}, makeCov()};
}

// A constant-velocity step with a control input and process noise on the velocity.
Eigen::Matrix4d const F{{1.0, 0.0, 0.5, 0.0}, {0.0, 1.0, 0.0, 0.5}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};

Eigen::Matrix<double, 4, 2> const G{{0.125, 0.0}, {0.0, 0.125}, {0.5, 0.0}, {0.0, 0.5}};

// The step reads the state, the noise, and the control by tag, f(in, x).
auto const process = []<typename T>(auto const & in, auto const & x) {
    return Eigen::Matrix<T, 4, 1>{
        F * Emrys::block<State>(in, x) + G * Emrys::block<ProcessNoise>(in, x) + Emrys::block<Control>(in, x)
    };
};

// Position measurement, H x, with its noise added.
Eigen::Matrix<double, 2, 4> const H{{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}};

auto const fix = []<typename T>(auto const & x) {
    return Eigen::Matrix<T, 2, 1>{H * x};
};

Eigen::Matrix2d
noiseCov()
{
    return Eigen::Matrix2d{{0.3, 0.1}, {0.1, 0.2}};
}

GaussianParams<ProcessNoise>
makeProcessNoise()
{
    return {ProcessNoise{}, Eigen::Vector2d::Zero(), noiseCov()};
}

GaussianParams<Fix>
makeFixNoise()
{
    return {Fix{}, Eigen::Vector2d::Zero(), noiseCov()};
}

template <typename Method>
class TransformMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<methods::Linearized, methods::Unscented<>>;

}  // namespace

TYPED_TEST_SUITE(TransformMethodTest, Methods);

// Every method transforms a constant-velocity step with noise and a known control, each matched to
// reads() by tag, to the closed-form mean F x + u and covariance F P F' + G Q G', in the output space
// outputSpace() names.
TYPED_TEST(TransformMethodTest, LinearProcessMatchesClosedForm)
{
    auto const x = makeState();
    auto const w = makeProcessNoise();
    auto const u = Eigen::Vector4d{0.1, 0.2, 0.0, -0.1};
    auto const predicted = Emrys::transform(
        TypeParam{},
        process,
        reads(path<State>, path<ProcessNoise>, path<Control>),
        outputSpace(State{}),
        x,
        w,
        sure<Control>(u)
    );
    static_assert(std::is_same_v<decltype(predicted), GaussianParams<State> const>);
    EXPECT_LT((predicted.mean() - (F * x.mean() + u)).norm(), 1e-12);
    Eigen::Matrix4d const expected = F * makeCov() * F.transpose() + G * w.covarianceLower() * G.transpose();
    EXPECT_LT((predicted.covarianceLower() - expected).norm(), 1e-12);

    // The inputs are matched by tag, so their order after the policies is free; it orders the joint
    // input, and so the sums, which agree to rounding.
    auto const reordered = Emrys::transform(
        TypeParam{},
        process,
        reads(path<State>, path<ProcessNoise>, path<Control>),
        outputSpace(State{}),
        sure<Control>(u),
        w,
        x
    );
    EXPECT_LT((reordered.mean() - predicted.mean()).norm(), 1e-14);
    EXPECT_LT((reordered.covarianceLower() - predicted.covarianceLower()).norm(), 1e-14);
}

// withCross(x) makes the result carry the cross-covariance P F' between the marked input and the
// output, and the output keeps its space.
TYPED_TEST(TransformMethodTest, CrossWithTheMarkedInput)
{
    auto const x = makeState();
    auto const w = makeProcessNoise();
    auto const predicted = Emrys::transform(
        TypeParam{},
        process,
        reads(path<State>, path<ProcessNoise>, path<Control>),
        outputSpace(State{}),
        withCross(x),
        w,
        sure<Control>(Eigen::Vector4d::Zero())
    );
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(predicted.crossCovariance())>, Eigen::Matrix4d>);
    static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(predicted.output())>::SpaceType, State>);
    EXPECT_LT((predicted.crossCovariance() - makeCov() * F.transpose()).norm(), 1e-12);
    Eigen::Matrix4d const expected = F * makeCov() * F.transpose() + G * w.covarianceLower() * G.transpose();
    EXPECT_LT((predicted.output().covarianceLower() - expected).norm(), 1e-12);
}

// A square-root form is an uncertain input by its root ([gaussian-forms]), marked or not: every method transforms a
// SqrtGaussianParams state and noise as it does their covariance forms, and condition() of the square-root prior
// matches the covariance form's. The information forms are not inputs, since their mean can fail.
TYPED_TEST(TransformMethodTest, ASquareRootFormIsAnInput)
{
    using Sqrt = Emrys::SqrtGaussianParams<State>;
    static_assert(Emrys::UncertainInput<Sqrt> && Emrys::UncertainInput<Emrys::SqrtGaussian<State>>);
    static_assert(!Emrys::UncertainInput<Emrys::InfoGaussianParams<State>>);
    static_assert(!Emrys::UncertainInput<Emrys::SqrtInfoGaussianParams<State>>);
    auto const x = makeState();
    auto const w = makeProcessNoise();
    auto const sqrtX = Emrys::toSqrt(x).value();
    auto const sqrtW = Emrys::toSqrt(w).value();
    auto const read = reads(path<State>, path<ProcessNoise>, path<Control>);
    auto const u = sure<Control>(Eigen::Vector4d{0.1, 0.2, 0.0, -0.1});
    auto const predicted =
        Emrys::transform(TypeParam{}, process, read, outputSpace(State{}), withCross(sqrtX), sqrtW, u);
    auto const expected = Emrys::transform(TypeParam{}, process, read, outputSpace(State{}), withCross(x), w, u);
    EXPECT_LT((predicted.output().mean() - expected.output().mean()).norm(), 1e-12);
    EXPECT_LT((predicted.output().covarianceLower() - expected.output().covarianceLower()).norm(), 1e-12);
    EXPECT_LT((predicted.crossCovariance() - expected.crossCovariance()).norm(), 1e-12);

    auto const z = Eigen::Vector2d{1.5, -1.0};
    auto const conditioned =
        Emrys::condition(sqrtX, Emrys::transform(TypeParam{}, fix, additiveNoise(makeFixNoise()), withCross(sqrtX)), z);
    auto const reference =
        Emrys::condition(x, Emrys::transform(TypeParam{}, fix, additiveNoise(makeFixNoise()), withCross(x)), z);
    ASSERT_TRUE(conditioned);
    ASSERT_TRUE(reference);
    EXPECT_LT((conditioned->posterior.mean() - reference->posterior.mean()).norm(), 1e-12);
    EXPECT_LT((conditioned->posterior.covarianceLower() - reference->posterior.covarianceLower()).norm(), 1e-12);
}

// An innovation covariance that is not positive definite fails the update, as it fails a filter's.
TEST(TransformUpdate, FailsOnAnIndefiniteInnovation)
{
    auto const x = makeState();
    auto noise = makeFixNoise();
    noise.setCovariance(Eigen::Matrix2d{-100.0 * Eigen::Matrix2d::Identity()});
    auto const predicted = Emrys::transform(methods::linearized(), fix, additiveNoise(noise), withCross(x));
    EXPECT_EQ(
        Emrys::condition(x, predicted, Eigen::Vector2d{1.4, -1.5}).error(),
        Emrys::GaussianError::NotPositiveDefinite
    );
}

// condition(x, predicted measurement, z) gives the Kalman posterior (gain P H' S^-1) and the
// innovation: the residual r = z - H x, the covariance S, and from them the NIS and the Gaussian
// log-likelihood, for every method, with the noise given by additiveNoise() and its space the output's.
TYPED_TEST(TransformMethodTest, UpdateIsTheKalmanUpdate)
{
    auto const x = makeState();
    auto const v = makeFixNoise();
    auto const z = Eigen::Vector2d{1.4, -1.5};
    auto const predicted = Emrys::transform(TypeParam{}, fix, additiveNoise(v), withCross(x));
    static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(predicted.output())>::SpaceType, Fix>);
    auto const updated = Emrys::condition(x, predicted, z);
    ASSERT_TRUE(updated);
    auto const & posterior = updated->posterior;
    static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(posterior)>::SpaceType, State>);

    Eigen::Matrix4d const P = makeCov();
    Eigen::Matrix2d const S = H * P * H.transpose() + v.covarianceLower();
    Eigen::Matrix<double, 4, 2> const K = P * H.transpose() * S.inverse();
    Eigen::Vector2d const r = z - H * x.mean();
    EXPECT_LT((posterior.mean() - (x.mean() + K * r)).norm(), 1e-10);
    EXPECT_LT((posterior.covarianceLower() - (Eigen::Matrix4d::Identity() - K * H) * P).norm(), 1e-10);
    EXPECT_LT((updated->innovation.residual() - r).norm(), 1e-10);
    EXPECT_LT((updated->innovation.covarianceLower() - S).norm(), 1e-10);
    double const nis = r.dot(S.inverse() * r);
    EXPECT_NEAR(updated->innovation.nis(), nis, 1e-10);
    double const expected = -0.5 * (2 * std::log(2 * pi) + std::log(S.determinant()) + nis);
    EXPECT_NEAR(updated->logLikelihood(), expected, 1e-10);
}

// methods::linearized() differentiating automatically and methods::linearized() taking the hand
// Jacobian a function returns with its value over its whole input (withJacobian(), [returned-jacobians])
// agree on a range-bearing transform with an uncertain sensor position, the sensor's uncertainty
// widens the output, and only the marked input's cross-covariance is returned.
TEST(Transform, AutoDiffMatchesAReturnedJacobian)
{
    // Range and bearing to the state's position from an uncertain sensor position, which is not kept.
    // The test calls it too, so it deduces its scalar rather than taking it as a template parameter.
    auto const rangeBearing = [](auto const & in, auto const & x) {
        using T = ScalarOf<decltype(x)>;
        using std::atan2;
        using std::sqrt;
        Eigen::Matrix<T, 2, 1> const d =
            Emrys::block<State>(in, x).template head<2>() - Emrys::block<SensorPosition>(in, x);
        return Eigen::Matrix<T, 2, 1>{sqrt(d.squaredNorm()), atan2(d(1), d(0))};
    };
    auto const jacobian = [](Eigen::Vector4d const & x, Eigen::Vector2d const & sensor) {
        Eigen::Vector2d const d = x.head<2>() - sensor;
        double const r2 = d.squaredNorm();
        double const r = std::sqrt(r2);
        Eigen::Matrix<double, 2, 4> jx = Eigen::Matrix<double, 2, 4>::Zero();
        jx.leftCols<2>() << d(0) / r, d(1) / r, -d(1) / r2, d(0) / r2;
        return std::tuple{jx, Eigen::Matrix2d{-jx.leftCols<2>()}};
    };
    auto const x = makeState();
    auto const sensor = GaussianParams<SensorPosition>{
        SensorPosition{},
        Eigen::Vector2d{-1.0, 1.0},
        0.04 * Eigen::Matrix2d::Identity()
    };
    auto const withJacobian = [&](auto const & in, Eigen::Matrix<double, 6, 1> const & joint) {
        Eigen::Vector4d const state = Emrys::block<State>(in, joint);
        Eigen::Vector2d const position = Emrys::block<SensorPosition>(in, joint);
        auto const [jx, js] = jacobian(state, position);
        Eigen::Matrix<double, 2, 6> whole;
        whole << jx, js;
        return Emrys::withJacobian(rangeBearing(in, joint), whole);
    };
    auto const read = reads(path<State>, path<SensorPosition>);
    auto const automatic = Emrys::transform(methods::linearized(), rangeBearing, read, withCross(x), sensor);
    auto const supplied = Emrys::transform(methods::linearized(), withJacobian, read, withCross(x), sensor);
    EXPECT_LT((automatic.output().mean() - supplied.output().mean()).norm(), 1e-14);
    EXPECT_LT((automatic.output().covarianceLower() - supplied.output().covarianceLower()).norm(), 1e-12);
    EXPECT_LT((automatic.crossCovariance() - supplied.crossCovariance()).norm(), 1e-12);

    // The sensor's uncertainty widens the output; only the state's cross-covariance is returned.
    auto const [jx, js] = jacobian(x.mean(), sensor.mean());
    Eigen::Matrix2d const expected = jx * makeCov() * jx.transpose() + js * sensor.covarianceLower() * js.transpose();
    EXPECT_LT((automatic.output().covarianceLower() - expected).norm(), 1e-12);
    EXPECT_LT((automatic.crossCovariance() - makeCov() * jx.transpose()).norm(), 1e-12);
}

// For a square of a scalar Gaussian the unscented transform gives the exact mean and variance (mu^2
// + sigma^2 and 4 mu^2 sigma^2 + 2 sigma^4) where linearization misses the sigma terms.
TEST(Transform, UnscentedIsExactForASquare)
{
    // For x ~ N(μ, σ²), E[x²] = μ² + σ² and Var[x²] = 4μ²σ² + 2σ⁴; linearizing misses the σ terms.
    double const mu = 1.5;
    double const variance = 0.25;
    auto const x = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{mu}, Eigen::Matrix<double, 1, 1>{variance}};
    auto const square = []<typename T>(auto const & s) {
        return Eigen::Matrix<T, 1, 1>{s(0) * s(0)};
    };

    auto const sigma = Emrys::transform(Emrys::methods::unscented(), square, x);
    EXPECT_NEAR(sigma.mean()(0), mu * mu + variance, 1e-12);
    EXPECT_NEAR(sigma.covarianceLower()(0, 0), 4 * mu * mu * variance + 2 * variance * variance, 1e-12);

    auto const tangent = Emrys::transform(Emrys::methods::linearized(), square, x);
    EXPECT_NEAR(tangent.mean()(0), mu * mu, 1e-12);
    EXPECT_NEAR(tangent.covarianceLower()(0, 0), 4 * mu * mu * variance, 1e-12);
}

// With an output space given, the unscented mean of sigma points straddling pi is the wrapped mean
// with the right variance; without it, averaging in a linear space is wrong by more than one.
TEST(Transform, UnscentedAveragesAcrossTheSeam)
{
    // The function wraps its own output, so its sigma points straddle ±π.
    auto const heading =
        GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{pi - 0.05}, Eigen::Matrix<double, 1, 1>{0.01}};
    auto const turn = []<typename T>(auto const & theta) {
        using std::atan2;
        using std::cos;
        using std::sin;
        return Eigen::Matrix<T, 1, 1>{atan2(sin(theta(0) + T(0.1)), cos(theta(0) + T(0.1)))};
    };
    auto const turned = Emrys::transform(Emrys::methods::unscented(), turn, outputSpace(RadiansPi<>{}), heading);
    EXPECT_NEAR(turned.mean()(0), -pi + 0.05, 1e-12);
    EXPECT_NEAR(turned.covarianceLower()(0, 0), 0.01, 1e-12);

    auto const unwrapped = Emrys::transform(Emrys::methods::unscented(), turn, heading);
    EXPECT_GT(std::abs(unwrapped.covarianceLower()(0, 0) - 0.01), 1.0);  // averaging across the seam in a linear space
}

// A transform without an output space returns a Gaussian over Linear of the function's output size.
TEST(Transform, OutputSpaceDefaultsToLinear)
{
    auto const x = makeState();
    auto const positionAndSpeed = []<typename T>(auto const & s) {
        return Eigen::Matrix<T, 3, 1>{s(0), s(1), s.template tail<2>().norm()};
    };
    auto const result = Emrys::transform(Emrys::methods::linearized(), positionAndSpeed, x);
    static_assert(std::is_same_v<decltype(result)::SpaceType, Linear<3>>);
    EXPECT_EQ(result.mean().head<2>(), x.mean().head<2>());
}

// The output space is outputSpace()'s, else the Composite of writes(), else the noise's: the three
// give one result, each in its space, and the noise adds its covariance whichever names the space.
TEST(Transform, OutputSpaceFromItsPolicies)
{
    auto const x = makeState();
    auto const v = makeFixNoise();
    auto const named = Emrys::transform(methods::unscented(), fix, outputSpace(Fix{}), withCross(x));
    auto const written = Emrys::transform(methods::unscented(), fix, writes(path<Fix>), withCross(x));
    auto const noisy = Emrys::transform(methods::unscented(), fix, additiveNoise(v), withCross(x));
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(named.output())>::SpaceType, Fix>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(written.output())>::SpaceType, Composite<Fix>>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(noisy.output())>::SpaceType, Fix>);
    EXPECT_EQ(named.output().mean(), written.output().mean());
    EXPECT_EQ(named.output().mean(), noisy.output().mean());
    EXPECT_EQ(named.crossCovariance(), noisy.crossCovariance());
    EXPECT_LT((noisy.output().covarianceLower() - (named.output().covarianceLower() + noiseCov())).norm(), 1e-14);
    EXPECT_LT((noisy.inputsCovariance() - (named.inputsCovariance() + noiseCov())).norm(), 1e-14);

    auto const linear = Emrys::transform(methods::linearized(), fix, outputSpace(Fix{}), additiveNoise(v), x);
    static_assert(std::is_same_v<decltype(linear), GaussianParams<Fix> const>);
    EXPECT_LT((linear.covarianceLower() - (H * makeCov() * H.transpose() + noiseCov())).norm(), 1e-14);
}

// A function that returns an Eigen expression, A * s, which refers to the argument the library
// builds for the call, is evaluated while the argument is alive: the mean is
// A m = {5, -2} under every method, where it read freed memory before.
TEST(Transform, EvaluatesAReturnedExpression)
{
    Eigen::Matrix2d const A{{1.0, 2.0}, {0.0, -1.0}};
    auto const x = GaussianParams<Linear<2>>{Eigen::Vector2d{1.0, 2.0}, Eigen::Matrix2d::Identity()};
    auto const expression = [&]<typename T>(auto const & s) {
        return A.template cast<T>() * s;
    };
    EXPECT_LT(
        (Emrys::transform(Emrys::methods::unscented(), expression, x).mean() - Eigen::Vector2d{5.0, -2.0}).norm(),
        1e-12
    );
    EXPECT_LT(
        (Emrys::transform(Emrys::methods::linearized(), expression, x).mean() - Eigen::Vector2d{5.0, -2.0}).norm(),
        1e-12
    );
}

// The marked input may be any input, and the cross-covariance is with that input alone.
TEST(Transform, CrossWithAnyArgument)
{
    struct A : Linear<2>
    {};

    struct B : Linear<2>
    {};

    auto const a = GaussianParams<A>{A{}, Eigen::Vector2d::Zero(), noiseCov()};
    auto const b = GaussianParams<B>{B{}, Eigen::Vector2d{1.0, 2.0}, 3.0 * Eigen::Matrix2d::Identity()};
    auto const sum = []<typename T>(auto const & in, auto const & x) {
        return Eigen::Matrix<T, 2, 1>{Emrys::block<A>(in, x) + Emrys::block<B>(in, x)};
    };
    auto const predicted =
        Emrys::transform(Emrys::methods::linearized(), sum, reads(path<A>, path<B>), a, withCross(b));
    EXPECT_EQ(predicted.crossCovariance(), b.covarianceLower());
    EXPECT_EQ(predicted.output().mean(), a.mean() + b.mean());
    EXPECT_EQ(predicted.output().covarianceLower(), a.covarianceLower() + b.covarianceLower());
}

// A Gaussian view and a GaussianMarginal are accepted as inputs and give the same result as
// their parameters.
TEST(Transform, AcceptsDensitiesAndMarginals)
{
    auto const x = makeState();
    auto const density = x.toDensity();
    auto const speed = [](auto const & velocity) {
        return Eigen::Matrix<ScalarOf<decltype(velocity)>, 1, 1>{velocity.norm()};
    };
    auto const fromMarginal = Emrys::transform(Emrys::methods::linearized(), speed, x.marginal<Velocity>());
    auto const fromDensity = Emrys::transform(
        Emrys::methods::linearized(),
        [&]<typename T>(auto const & s) {
            return speed(Eigen::Matrix<T, 2, 1>{s.template tail<2>()});
        },
        *density
    );
    EXPECT_NEAR(fromMarginal.mean()(0), fromDensity.mean()(0), 1e-15);
    EXPECT_NEAR(fromMarginal.covarianceLower()(0, 0), fromDensity.covarianceLower()(0, 0), 1e-15);
}

// The unscented transform factors a semidefinite noise covariance (no Cholesky factor) and matches
// the affine result.
TEST(Transform, UnscentedHandlesSemidefiniteNoise)
{
    // Noise only on the first coordinate: its covariance has no Cholesky factor.
    auto const w =
        GaussianParams<ProcessNoise>{ProcessNoise{}, Eigen::Vector2d::Zero(), Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.0}}};
    auto const x = makeState();
    auto const u = sure<Control>(Eigen::Vector4d::Zero());
    auto const read = reads(path<State>, path<ProcessNoise>, path<Control>);
    auto const sigma = Emrys::transform(Emrys::methods::unscented(), process, read, outputSpace(State{}), x, w, u);
    auto const exact = Emrys::transform(Emrys::methods::linearized(), process, read, outputSpace(State{}), x, w, u);
    EXPECT_LT((sigma.mean() - exact.mean()).norm(), 1e-12);
    EXPECT_LT((sigma.covarianceLower() - exact.covarianceLower()).norm(), 1e-12);
}

// A measurement model is a transform with its inputs left unbound: transform(method, model, inputs)
// of a whole-state model at the state, and of a model with reads() at the blocks it reads, is the
// predicted measurement an update conditions on, its mean z - r and its covariance S.
TEST(Transform, MeasurementModelPredictsAsATransform)
{
    auto const x = makeState();
    auto const z = Eigen::Vector2d{1.4, -1.5};
    auto const whole = Emrys::measurementModel(fix, additiveNoise(makeFixNoise()));
    auto const tagged = Emrys::measurementModel(
        [](auto const & in, auto const & x) {
            return Emrys::block<Position>(in, x).eval();
        },
        reads(path<Position>),
        additiveNoise(makeFixNoise())
    );
    auto const check = [&](auto const & method, auto const & model, auto const & input) {
        auto const updated = Emrys::update(x, Emrys::observe(method, model, z));
        ASSERT_TRUE(updated);
        auto const & innovation = updated->branches.front().innovation;
        auto const predicted = Emrys::transform(method, model, input);
        EXPECT_LT((predicted.mean() - (z - innovation.residual())).norm(), 1e-12);
        EXPECT_LT((predicted.covarianceLower() - innovation.covarianceLower()).norm(), 1e-12);
    };
    check(methods::linearized(), whole, x);
    check(methods::unscented(), whole, x);
    check(methods::linearized(), tagged, x.marginal<Position>());
    check(methods::cubature(), tagged, x.marginal<Position>());
}

// A model affine in a noise that scales its range, affineIn(), with additive noise besides, gives
// through transform() the predicted measurement observe() conditions on, under a sigma-point method
// that factors the noise out and under Linearized, which ignores the declaration.
TEST(Transform, AffineInAndAdditiveNoiseMatchObserve)
{
    auto const x = makeState();
    auto const scale =
        GaussianParams<ScaleNoise>{ScaleNoise{}, Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.04}};
    auto const rangeNoise =
        GaussianParams<Range>{Range{}, Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.09}};
    auto const range = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            return T{(1 + Emrys::block<ScaleNoise>(in, x)(0)) * Emrys::block<Position>(in, x).norm()};
        },
        reads(path<Position>, path<ScaleNoise>),
        affineIn(path<ScaleNoise>),
        additiveNoise(rangeNoise)
    );
    auto const z = Eigen::Matrix<double, 1, 1>{2.5};
    auto const check = [&](auto const & method) {
        auto const updated = Emrys::update(x, Emrys::observe(method, range, z, scale));
        ASSERT_TRUE(updated);
        auto const & innovation = updated->branches.front().innovation;
        auto const predicted = Emrys::transform(method, range, x.marginal<Position>(), scale);
        static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(predicted)>::SpaceType, Range>);
        EXPECT_LT((predicted.mean() - (z - innovation.residual())).norm(), 1e-12);
        EXPECT_LT((predicted.covarianceLower() - innovation.covarianceLower()).norm(), 1e-12);
    };
    check(methods::cubature());
    check(methods::unscented());
    check(methods::linearized());
}

namespace
{

template <typename T>
concept Markable = requires(T const & input) { withCross(input); };

static_assert(Markable<GaussianParams<Linear<2>>>);
static_assert(!Markable<Eigen::Vector2d>, "a known input has no cross-covariance");

// Declared only: the concept below never calls it.
struct Sum
{
    template <typename P>
    Eigen::Matrix<ScalarOf<P>, 2, 1>
    operator()(P const & p) const;
};

template <typename T>
concept TwoMarkable =
    requires(T const & a, T const & b) { Emrys::transform(methods::linearized(), Sum{}, withCross(a), withCross(b)); };

static_assert(!TwoMarkable<GaussianParams<Linear<2>>>, "at most one input is marked");

}  // namespace
