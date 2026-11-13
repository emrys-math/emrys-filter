// The linearization forms of a transform's result, OutputWithCross in Models/Linearization.h (roadmap step 17u,
// [linearization-forms]; [covariance-update-forms]): every method's result holds H P H^T, N, and Omega apart and
// sums them to the output covariance, the root form's Psi splits by input and gives H by a solve with its
// root, Omega comes from the residuals at the points where a subtraction would lose its digits, a
// semidefinite input's zero root column keeps Psi^T Psi the explained part, and the low-level update's
// Joseph form takes its noise from the prediction rather than recovering it by subtraction.

#include <EmrysFilter/Models/Transform.h>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

using Emrys::additiveNoise;
using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::path;
using Emrys::reads;
using Emrys::ScalarOf;
using Emrys::withCross;

namespace methods = Emrys::methods;
namespace updates = Emrys::updates;

namespace
{

struct Scale : Linear<1>
{};

struct Velocity : Linear<2>
{};

struct Point : Linear<3>
{};

struct Noise : Linear<2>
{};

using Vector1 = Eigen::Matrix<double, 1, 1>;

// A state of three coordinates and a nonlinear function of it and of a noise of two, whose noise
// enters through the state, so that neither part of the output covariance is affine.
GaussianParams<Point>
makeState()
{
    return {
        Point{},
        Eigen::Vector3d{0.4, -1.0, 0.7},
        Eigen::Matrix3d{{0.3, 0.05, -0.02}, {0.05, 0.2, 0.04}, {-0.02, 0.04, 0.25}}
    };
}

GaussianParams<Noise>
makeNoise()
{
    return {Noise{}, Eigen::Vector2d{0.1, 0.0}, Eigen::Matrix2d{{0.04, 0.01}, {0.01, 0.09}}};
}

// The function of the state and the noise, read by tag, and the reads() it is transformed with.
auto const nonlinear = []<typename T>(auto const & in, auto const & joint) {
    auto const x = Emrys::block<Point>(in, joint);
    auto const w = Emrys::block<Noise>(in, joint);
    return Eigen::Matrix<T, 2, 1>{
        Eigen::numext::sin(T(x(0))) * x(1) + w(0) * x(2),
        x(0) * x(0) + Eigen::numext::exp(T(0.5 * x(2))) + w(1) + w(0) * w(1)
    };
};

auto const pointAndNoise = reads(path<Point>, path<Noise>);

}  // namespace

// For every method of the Transforms module page's table, with the noise an unmarked input:
// the parts sum to the output covariance exactly and to the unmarked transform's to rounding, the
// explained part is H P H^T for the H the form gives, and the cross-covariance is P H^T. The root form's
// root is a root of P and its Psi is (H S)^T, the marked input's rows of the joint Psi.
TEST(LinearizationForms, ThePartsSumToTheOutputCovariance)
{
    auto const x = makeState();
    auto const w = makeNoise();
    Eigen::Matrix3d const p = x.covarianceLower();
    auto const check = [&](auto const & method, char const * name) {
        auto const plain = Emrys::transform(method, nonlinear, pointAndNoise, x, w);
        auto const predicted = Emrys::transform(method, nonlinear, pointAndNoise, withCross(x), w);
        Eigen::Matrix2d const sum =
            predicted.explainedCovariance() + predicted.inputsCovariance() + predicted.unexplainedCovariance();
        EXPECT_EQ(predicted.output().covarianceLower(), sum) << name;
        EXPECT_LT((sum - plain.covarianceLower()).norm(), 1e-14 * plain.covarianceLower().norm()) << name;
        EXPECT_EQ(predicted.output().mean(), plain.mean()) << name;
        Eigen::Matrix<double, 2, 3> const h = predicted.jacobian();
        EXPECT_LT((predicted.explainedCovariance() - h * p * h.transpose()).norm(), 1e-13) << name;
        EXPECT_LT((predicted.crossCovariance() - p * h.transpose()).norm(), 1e-13) << name;
        using Predicted = std::remove_cvref_t<decltype(predicted)>;
        if constexpr (std::is_same_v<typename Predicted::FormType, Emrys::RootForm>)
        {
            EXPECT_LT((predicted.root() * predicted.root().transpose() - p).norm(), 1e-14) << name;
            EXPECT_LT((predicted.whitenedCross() - (h * predicted.root()).transpose()).norm(), 1e-13) << name;
        }
        else
        {
            EXPECT_TRUE(predicted.unexplainedCovariance().isZero()) << name;
        }
    };
    check(methods::linearized(), "linearized");
    check(methods::unscented(), "unscented");
    check(methods::unscented<1.0, 2.0, 2.0>(), "unscented<1, 2, 2>");
    check(methods::fourthMomentUnscented(), "fourthMomentUnscented");
    check(methods::cubature(), "cubature");
    check(methods::sphericalSimplex(), "sphericalSimplex");
    check(methods::gaussHermite<2>(), "gaussHermite<2>");
    check(methods::gaussHermite<3>(), "gaussHermite<3>");
}

// The factored transform a model's affineIn() makes (internal::FactoredAt), with the velocity and the noise affine:
// y = s v + w for a scale s correlated with the velocity v. The affine coordinates get no points but have their
// root, so the result is in the root form, its parts sum to the output covariance, the curvature the Jacobian's
// change with s leaves is in Omega, and H, P H^T, and the root are those of the prior.
TEST(LinearizationForms, TheFactoredTransformKeepsTheAffinePartsApart)
{
    struct State : Composite<Scale, Velocity>
    {};

    auto const x = GaussianParams<State>{
        State{},
        Eigen::Vector3d{1.5, 0.5, -1.0},
        Eigen::Matrix3d{{0.2, 0.05, -0.03}, {0.05, 0.4, 0.1}, {-0.03, 0.1, 0.3}}
    };
    auto const w = GaussianParams<Noise>{Noise{}, Eigen::Vector2d::Zero(), Eigen::Matrix2d{{0.01, 0.0}, {0.0, 0.02}}};
    auto const f = []<typename T>(auto const & in, auto const & joint) {
        auto const state = Emrys::block<State>(in, joint);
        return Eigen::Matrix<T, 2, 1>{state(0) * state.template tail<2>() + Emrys::block<Noise>(in, joint)};
    };
    auto const method = Emrys::internal::FactoredAt<4, 4, methods::Unscented<1.0, 0.0, 2.0>>{
        methods::unscented<1.0, 0.0, 2.0>(),
        Eigen::Matrix<Eigen::Index, 4, 1>{1, 2, 3, 4}
    };
    auto const predicted = Emrys::transform(method, f, reads(path<State>, path<Noise>), withCross(x), w);
    static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(predicted)>::FormType, Emrys::RootForm>);
    Eigen::Matrix3d const p = x.covarianceLower();
    Eigen::Matrix2d const sum =
        predicted.explainedCovariance() + predicted.inputsCovariance() + predicted.unexplainedCovariance();
    EXPECT_EQ(predicted.output().covarianceLower(), sum);
    EXPECT_LT((predicted.inputsCovariance() - w.covarianceLower()).norm(), 1e-15);
    EXPECT_GT(predicted.unexplainedCovariance().norm(), 0.01);
    EXPECT_LT((predicted.root() * predicted.root().transpose() - p).norm(), 1e-14);
    Eigen::Matrix<double, 2, 3> const h = predicted.jacobian();
    EXPECT_LT((predicted.crossCovariance() - p * h.transpose()).norm(), 1e-13);
    EXPECT_LT((predicted.explainedCovariance() - h * p * h.transpose()).norm(), 1e-13);

    // Isserlis' theorem, as AffineCoordinates.EverySigmaPointMethodGivesIsserlisMomentsOnABilinearModel, plus the
    // noise.
    Eigen::Vector3d const mean = x.mean();
    double const mu = mean(0);
    double const s2 = p(0, 0);
    Eigen::Vector2d const m = mean.tail<2>();
    Eigen::Vector2d const c = p.bottomLeftCorner<2, 1>();
    Eigen::Matrix2d const pv = p.bottomRightCorner<2, 2>();
    Eigen::Matrix2d const expected = (mu * mu + s2) * pv + mu * (c * m.transpose() + m * c.transpose())
                                   + s2 * m * m.transpose() + c * c.transpose() + w.covarianceLower();
    EXPECT_LT((sum - expected).norm(), 1e-12);
}

// y = x + eps x^2 with x ~ N(0, 1) and eps = 1e-8: H = 1, and what H leaves unexplained is the closed
// form 2 eps^2 sigma^4 = 2e-16, which the unscented transform and a third-order Gauss-Hermite rule are
// exact for. Sigma_yy - H P H^T subtracts numbers of size one to get it and keeps no digit of it; the
// residuals at the points, of size eps, give it to a part in a million.
TEST(LinearizationForms, OmegaComesFromTheResidualsWhereASubtractionLosesIt)
{
    double const eps = 1e-8;
    auto const x = GaussianParams<Linear<1>>{Vector1{0.0}, Vector1{1.0}};
    auto const f = [&]<typename T>(auto const & s) {
        return Eigen::Matrix<T, 1, 1>{s(0) + T(eps) * s(0) * s(0)};
    };
    double const exact = 2 * eps * eps;
    auto const check = [&](auto const & method, char const * name) {
        auto const predicted = Emrys::transform(method, f, withCross(x));
        EXPECT_NEAR(predicted.jacobian()(0, 0), 1.0, 1e-14) << name;
        EXPECT_NEAR(predicted.unexplainedCovariance()(0, 0), exact, 1e-6 * exact) << name;
        auto const [jacobian, errorCov] = Emrys::statisticalLinearization(predicted);
        EXPECT_NEAR(errorCov(0, 0), exact, 1e-6 * exact) << name;
    };
    check(methods::unscented(), "unscented");
    check(methods::gaussHermite<3>(), "gaussHermite<3>");
}

// A semidefinite input, a coordinate known exactly, under the spherical simplex, whose points are not
// symmetric: the default root has a zero column there, Psi's row at it is zero, so Psi^T Psi is still
// H P H^T, and the cross-covariance's row at the known coordinate is zero.
TEST(LinearizationForms, ASemidefiniteInputKeepsTheExplainedPart)
{
    auto const x = GaussianParams<Linear<2>>{Eigen::Vector2d{0.5, 1.0}, Eigen::Matrix2d{{0.3, 0.0}, {0.0, 0.0}}};
    auto const f = []<typename T>(auto const & s) {
        return Eigen::Matrix<T, 2, 1>{s(0) * s(0) * s(1), Eigen::numext::sin(T(s(0))) + s(1) * s(1)};
    };
    auto const predicted = Emrys::transform(methods::sphericalSimplex(), f, withCross(x));
    Eigen::Matrix2d const root = predicted.root();
    EXPECT_TRUE(root.col(1).isZero());
    EXPECT_TRUE(predicted.whitenedCross().row(1).isZero());
    Eigen::Matrix2d const h = predicted.jacobian();
    Eigen::Matrix2d const p = x.covarianceLower();
    EXPECT_LT((predicted.explainedCovariance() - h * p * h.transpose()).norm(), 1e-14);
    EXPECT_TRUE(predicted.crossCovariance().row(1).isZero());
    Eigen::Matrix2d const sum =
        predicted.explainedCovariance() + predicted.inputsCovariance() + predicted.unexplainedCovariance();
    auto const plain = Emrys::transform(methods::sphericalSimplex(), f, x);
    EXPECT_LT((sum - plain.covarianceLower()).norm(), 1e-14);
}

// [covariance-update-forms]'s fix through the low-level update(): a fix of the first coordinate with additive noise
// of variance 1e-12, which the transform adds to N, against a prior of variance 1e6 correlated 0.999999 with
// the second. updates::joseph() takes R as the prediction's N, a product, rather than recovering it from
// Sigma_yy by subtraction, so the precise coordinate's posterior variance is R P / (P + R), 1e-12, to a
// part in a thousand, under the linearized and the cubature methods.
TEST(LinearizationForms, JosephKeepsAPreciseNoisePassedToTheTransform)
{
    double const big = 1e6;
    double const rho = 0.999999;
    double const precise = 1e-12;
    auto const prior =
        GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), big * Eigen::Matrix2d{{1.0, rho}, {rho, 1.0}}};
    auto const noise = GaussianParams<Linear<1>>{Vector1{0.0}, Vector1{precise}};
    auto const fix = []<typename T>(auto const & s) {
        return Eigen::Matrix<T, 1, 1>{s(0)};
    };
    auto const z = Vector1{1.0};
    auto const check = [&](auto const & method, char const * name) {
        auto const predicted = Emrys::transform(method, fix, additiveNoise(noise), withCross(prior));
        EXPECT_NEAR(predicted.inputsCovariance()(0, 0), precise, 1e-6 * precise) << name;
        auto const updated = Emrys::condition(prior, predicted, z, updates::joseph());
        ASSERT_TRUE(updated) << name;
        double const variance = updated->posterior.covarianceLower()(0, 0);
        EXPECT_NEAR(variance, precise * big / (big + precise), 1e-3 * precise) << name;
        EXPECT_NEAR(updated->posterior.covarianceLower()(1, 1), big * (1 - rho * rho), 1e-6 * big * (1 - rho * rho))
            << name;
    };
    check(methods::linearized(), "linearized");
    check(methods::cubature(), "cubature");
}

// In the root form updates::joseph() forms (S - K Psi^T)(S - K Psi^T)^T + K R K^T with no solve for H:
// on the nonlinear function and noise above, under the cubature rule and the unscented transform, it is
// (I - K H) P (I - K H)^T + K R K^T computed by hand from the same prediction, with H = Psi^T S^{-1} and
// R = N + Omega, to rounding, and it is exactly symmetric.
TEST(LinearizationForms, JosephInTheRootFormIsTheJosephForm)
{
    auto const x = makeState();
    auto const w = makeNoise();
    Eigen::Matrix3d const p = x.covarianceLower();
    auto const z = Eigen::Vector2d{0.2, 2.1};
    auto const check = [&](auto const & method, char const * name) {
        auto const predicted = Emrys::transform(method, nonlinear, pointAndNoise, withCross(x), w);
        auto const updated = Emrys::condition(x, predicted, z, updates::joseph());
        ASSERT_TRUE(updated) << name;
        Eigen::Matrix<double, 2, 3> const h = predicted.jacobian();
        Eigen::Matrix2d const r = predicted.inputsCovariance() + predicted.unexplainedCovariance();
        Eigen::Matrix<double, 3, 2> const k =
            predicted.crossCovariance() * predicted.output().covarianceLower().inverse();
        Eigen::Matrix3d const a = Eigen::Matrix3d::Identity() - k * h;
        Eigen::Matrix3d const expected = a * p * a.transpose() + k * r * k.transpose();
        Eigen::Matrix3d const posterior = updated->posterior.covarianceLower();
        EXPECT_LT((posterior - expected).norm(), 1e-14) << name;
        EXPECT_EQ(posterior, posterior.transpose()) << name;
        auto const standard = Emrys::condition(x, predicted, z);
        ASSERT_TRUE(standard) << name;
        EXPECT_LT((posterior - standard->posterior.covarianceLower()).norm(), 1e-14) << name;
    };
    check(methods::cubature(), "cubature");
    check(methods::unscented(), "unscented");
}
