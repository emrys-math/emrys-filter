// statisticalLinearization() and the updates::Standard, updates::Symmetric, and updates::Joseph covariance
// updates of Models/Condition.h, and the measurement's choice of one as an argument of observe() in
// Models/MeasurementModel.h (step 13c; [covariance-update-forms]): the linearization recovers an
// affine function and its noise and gives a square's regression line, the three forms agree with
// the hand formulas on an affine update, Joseph stays positive semidefinite for a measurement far
// more precise than the prior and upkeep repairs what the standard form leaves of it ([upkeep]), and a
// measurement's form reaches every branch of its update.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Transform.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::Composite;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::path;
using Emrys::reads;
using Emrys::ScalarOf;
using Emrys::withCross;

namespace methods = Emrys::methods;
namespace updates = Emrys::updates;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

using Vector1 = Eigen::Matrix<double, 1, 1>;

GaussianParams<State>
makeState()
{
    auto const cov =
        Eigen::Matrix4d{{1.0, 0.2, 0.1, 0.0}, {0.2, 2.0, -0.1, 0.3}, {0.1, -0.1, 0.5, 0.05}, {0.0, 0.3, 0.05, 0.8}};
    return {State{}, Eigen::Vector4d{1.0, -2.0, 0.5, 0.25}, cov};
}

// A position fix, h(p) = p, with additive noise, observed with the covariance update form given, if any.
template <typename... Form>
auto
makeFix(Eigen::Vector2d const & z, GaussianParams<Linear<2>> noise, Form const &... form)
{
    auto const position = [](auto const & in, auto const &, auto const & x) {
        return block<Position>(in, x);
    };
    return observe(
        methods::linearized(),
        measurementModel(position, reads(path<Position>), additiveNoise(std::move(noise))),
        z,
        form...
    );
}

// A range from the origin, h(p) = |p|, which is nonlinear in the position, by the unscented
// transform, with the covariance update form given, if any.
template <typename Noise, typename... Form>
auto
makeRange(double z, Noise noise, Form const &... form)
{
    auto const range = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(block<Position>(in, x).squaredNorm()))};
        },
        reads(path<Position>),
        additiveNoise(std::move(noise))
    );
    return observe(methods::unscented(), range, Vector1{z}, form...);
}

// Whether a matrix equals its transpose exactly, as the symmetric and Joseph forms make it.
bool
isExactlySymmetric(Eigen::MatrixXd const & m)
{
    return m == m.transpose();
}

constexpr double preciseVariance = 1e-12;  // the variance of the precise fix
constexpr double priorVariance = 1e6;      // the prior variance of each position coordinate
constexpr double priorCorrelation = 0.999999;

// A prior whose position coordinates have variance 1e6 and are almost perfectly correlated.
GaussianParams<State>
makePrecisePrior()
{
    Eigen::Matrix4d cov = Eigen::Matrix4d::Identity();
    cov.topLeftCorner<2, 2>() = priorVariance * Eigen::Matrix2d{{1.0, priorCorrelation}, {priorCorrelation, 1.0}};
    return {State{}, Eigen::Vector4d::Zero(), cov};
}

// A fix of the first position coordinate with variance 1e-12, observed with the covariance update form given, if any.
template <typename... Form>
auto
makePreciseFix(Form const &... form)
{
    auto const first = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{block<Position>(in, x)(0)};
        },
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<1>>{Vector1{0.0}, Vector1{preciseVariance}})
    );
    return observe(methods::linearized(), first, Vector1{1.0}, form...);
}

}  // namespace

// For y = A x + B w + c, with w an unmarked input, the linearization in x is A, and what it leaves
// unexplained is B Q B^T: exact for every method, since each is exact for an affine function.
TEST(StatisticalLinearization, RecoversAnAffineFunctionAndItsNoise)
{
    struct Input : Linear<2>
    {};

    struct InputNoise : Linear<1>
    {};

    auto const x = GaussianParams<Input>{Input{}, Eigen::Vector2d{1.0, -0.5}, Eigen::Matrix2d{{2.0, 0.3}, {0.3, 0.7}}};
    auto const w = GaussianParams<InputNoise>{InputNoise{}, Vector1{0.0}, Vector1{0.4}};
    auto const a = Eigen::Matrix<double, 3, 2>{{1.0, 2.0}, {-0.5, 0.0}, {0.25, 3.0}};
    auto const b = Eigen::Vector3d{1.0, -2.0, 0.5};
    auto const f = [&]<typename T>(auto const & in, auto const & joint) {
        return Eigen::Matrix<T, 3, 1>{
            a.cast<T>() * Emrys::block<Input>(in, joint) + b.cast<T>() * Emrys::block<InputNoise>(in, joint)(0)
            + Eigen::Vector3d{0.1, 0.2, 0.3}.cast<T>()
        };
    };
    Eigen::Matrix3d const noise = 0.4 * b * b.transpose();

    auto const check = [&](auto const & method) {
        auto const predicted =
            Emrys::transform(method, f, Emrys::reads(Emrys::path<Input>, Emrys::path<InputNoise>), withCross(x), w);
        auto const [jacobian, errorCov] = Emrys::statisticalLinearization(predicted);
        EXPECT_TRUE(jacobian.isApprox(a, 1e-12));
        EXPECT_LT((errorCov - noise).norm(), 1e-12);
    };
    check(methods::linearized());
    check(methods::unscented());
    check(methods::cubature());
}

// For x ~ N(mu, sigma^2) and y = x^2, Cov(x, y) = 2 mu sigma^2 and Var(y) = 4 mu^2 sigma^2 +
// 2 sigma^4, so the regression line has slope 2 mu and leaves 2 sigma^4 unexplained; the unscented
// transform is exact for a square, and linearizing at the mean sees no error at all.
TEST(StatisticalLinearization, OfASquareIsItsRegressionLine)
{
    double const mu = 1.5;
    double const variance = 0.25;
    auto const x = GaussianParams<Linear<1>>{Vector1{mu}, Vector1{variance}};
    auto const square = []<typename T>(auto const & s) {
        return Eigen::Matrix<T, 1, 1>{s(0) * s(0)};
    };

    auto const sigma =
        Emrys::statisticalLinearization(Emrys::transform(Emrys::methods::unscented(), square, withCross(x)));
    EXPECT_NEAR(sigma.jacobian(0, 0), 2 * mu, 1e-12);
    EXPECT_NEAR(sigma.errorCov(0, 0), 2 * variance * variance, 1e-12);

    auto const tangent =
        Emrys::statisticalLinearization(Emrys::transform(Emrys::methods::linearized(), square, withCross(x)));
    EXPECT_NEAR(tangent.jacobian(0, 0), 2 * mu, 1e-12);
    EXPECT_NEAR(tangent.errorCov(0, 0), 0.0, 1e-12);
}

// On an affine fix of the position with additive noise, the three forms
// give the hand Kalman update to rounding: P - K H P, P - K S K^T, and
// (I - K H) P (I - K H)^T + K R K^T. The mean and log-likelihood don't depend on the form, and the
// symmetric and Joseph forms are exactly symmetric.
TEST(CovarianceUpdate, FormsAgreeWithTheHandFormulas)
{
    GaussianParams<State> const prior = makeState();
    auto const r = Eigen::Matrix2d{{0.3, 0.05}, {0.05, 0.2}};
    auto const noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), r};
    auto const z = Eigen::Vector2d{1.4, -1.7};
    auto const fix = [&]<typename T>(auto const & s) {
        return Eigen::Matrix<T, 2, 1>{s.template head<2>()};
    };
    auto const predicted =
        Emrys::transform(Emrys::methods::linearized(), fix, Emrys::additiveNoise(noise), withCross(prior));

    Eigen::Matrix<double, 2, 4> h = Eigen::Matrix<double, 2, 4>::Zero();
    h.leftCols<2>() = Eigen::Matrix2d::Identity();
    Eigen::Matrix4d const p = prior.covarianceLower();
    Eigen::Matrix2d const s = h * p * h.transpose() + r;
    Eigen::Matrix<double, 4, 2> const k = p * h.transpose() * s.inverse();
    Eigen::Matrix4d const a = Eigen::Matrix4d::Identity() - k * h;
    Eigen::Vector4d const mean = prior.mean() + k * (z - h * prior.mean());

    auto const standard = Emrys::condition(prior, predicted, z);
    auto const symmetric = Emrys::condition(prior, predicted, z, updates::symmetric());
    auto const joseph = Emrys::condition(prior, predicted, z, updates::joseph());
    ASSERT_TRUE(standard);
    ASSERT_TRUE(symmetric);
    ASSERT_TRUE(joseph);
    EXPECT_TRUE(standard->posterior.mean().isApprox(mean, 1e-12));
    EXPECT_LT((standard->posterior.covarianceLower() - (p - k * h * p)).norm(), 1e-12);
    EXPECT_LT((symmetric->posterior.covarianceLower() - (p - k * s * k.transpose())).norm(), 1e-12);
    EXPECT_LT((joseph->posterior.covarianceLower() - (a * p * a.transpose() + k * r * k.transpose())).norm(), 1e-12);
    EXPECT_EQ(symmetric->posterior.mean(), standard->posterior.mean());
    EXPECT_EQ(joseph->posterior.mean(), standard->posterior.mean());
    EXPECT_EQ(symmetric->logLikelihood(), standard->logLikelihood());
    EXPECT_EQ(joseph->logLikelihood(), standard->logLikelihood());
    EXPECT_TRUE(isExactlySymmetric(symmetric->posterior.covarianceLower()));
    EXPECT_TRUE(isExactlySymmetric(joseph->posterior.covarianceLower()));
}

// A fix of the first position coordinate with variance 1e-12 against a prior of variance 1e6, almost
// perfectly correlated with the second: the posterior variance of the first is R P / (P + R),
// eighteen decimal orders below the prior's, and of the second about P (1 - rho^2). With the noise
// kept apart from the innovation covariance, the Joseph form is a sum of positive semidefinite
// terms, so its eigenvalues are nonnegative and the precise coordinate's variance is right to a part
// in a thousand. The standard and symmetric forms subtract numbers of size P to get it and, on
// Clang at least, return exactly zero: a singular posterior. That is rounding, so it is not asserted.
TEST(CovarianceUpdate, JosephStaysPositiveSemidefiniteForAPreciseMeasurement)
{
    double const big = priorVariance;
    double const rho = priorCorrelation;
    double const precise = preciseVariance;
    auto const updated = Emrys::update(makePrecisePrior(), makePreciseFix(updates::joseph()));
    ASSERT_TRUE(updated);
    Eigen::Matrix4d const posterior = updated->branches[0].posterior.covarianceLower();
    EXPECT_TRUE(isExactlySymmetric(posterior));
    auto const eigen = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>{posterior};
    EXPECT_GE(eigen.eigenvalues().minCoeff(), 0.0);
    EXPECT_NEAR(posterior(0, 0), precise * big / (big + precise), 1e-3 * precise);
    EXPECT_NEAR(posterior(1, 1), big * (1 - rho * rho), 1e-6 * big * (1 - rho * rho));
}

// The standard form may leave the posterior of the same precise fix not positive definite, by rounding, which the
// test above does not assert. upkeep::repairCovariance() applied to the Kalman filter's state() afterward [upkeep]
// repairs it exactly when its density fails, and reports whether it did, so the state's density can then be read.
TEST(CovarianceUpdate, UpkeepRepairsWhatTheStandardFormLeaves)
{
    auto filter = Emrys::KalmanFilter{Emrys::ConstantVelocity<Position, Velocity>{0.1}, makePrecisePrior()};
    ASSERT_TRUE(filter.update(makePreciseFix()));
    bool const failed = !filter.state().toDensity().has_value();
    auto const repaired = Emrys::upkeep::repairCovariance()(filter.state());
    ASSERT_TRUE(repaired);
    EXPECT_EQ(*repaired, failed);
    EXPECT_TRUE(filter.state().toDensity().has_value());
}

// A position fix observed with a form is a measurement whose form that is, and without one takes
// updates::Standard; through a KalmanFilter each form gives the standard posterior to rounding, and
// the symmetric and Joseph posteriors are exactly symmetric.
TEST(CovarianceUpdate, AKalmanFilterUsesTheMeasurementsForm)
{
    auto const noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{0.3, 0.05}, {0.05, 0.2}}};
    auto const z = Eigen::Vector2d{1.4, -1.7};
    auto const fix = makeFix(z, noise);
    static_assert(Emrys::Measurement<decltype(fix)>);
    static_assert(std::is_same_v<decltype(fix)::FormType, updates::Standard>);
    static_assert(std::is_same_v<decltype(makeFix(z, noise, updates::joseph()))::FormType, updates::Joseph>);

    auto const model = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto standard = Emrys::KalmanFilter{model, makeState()};
    auto symmetric = Emrys::KalmanFilter{model, makeState()};
    auto joseph = Emrys::KalmanFilter{model, makeState()};
    ASSERT_TRUE(standard.update(fix));
    ASSERT_TRUE(symmetric.update(makeFix(z, noise, updates::symmetric())));
    ASSERT_TRUE(joseph.update(makeFix(z, noise, updates::joseph())));
    EXPECT_LT((symmetric.state().covarianceLower() - standard.state().covarianceLower()).norm(), 1e-12);
    EXPECT_LT((joseph.state().covarianceLower() - standard.state().covarianceLower()).norm(), 1e-12);
    EXPECT_TRUE(isExactlySymmetric(symmetric.state().covarianceLower()));
    EXPECT_TRUE(isExactlySymmetric(joseph.state().covarianceLower()));
}

// An unscented range with an inlier and an outlier noise component: Joseph takes the linearization
// error from the sigma points and adds each component's own R, which is the standard update in
// exact arithmetic, so every branch matches the standard one to rounding and is exactly symmetric.
TEST(CovarianceUpdate, EveryBranchOfAMixtureUpdateUsesTheForm)
{
    auto noise = GaussianMixtureParams<Linear<1>>{1};
    noise.add(std::log(0.9), Vector1{0.0}, Vector1{0.05});
    noise.add(std::log(0.1), Vector1{0.3}, Vector1{4.0});
    GaussianParams<State> const prior = makeState();

    auto const standard = Emrys::update(prior, makeRange(2.4, noise));
    auto const joseph = Emrys::update(prior, makeRange(2.4, noise, updates::joseph()));
    ASSERT_TRUE(standard);
    ASSERT_TRUE(joseph);
    ASSERT_EQ(joseph->branches.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i)
    {
        Eigen::Matrix4d const cov = joseph->branches[i].posterior.covarianceLower();
        EXPECT_LT((cov - standard->branches[i].posterior.covarianceLower()).norm(), 1e-12) << i;
        EXPECT_EQ(joseph->branches[i].posterior.mean(), standard->branches[i].posterior.mean()) << i;
        EXPECT_TRUE(isExactlySymmetric(cov)) << i;
    }
    EXPECT_EQ(joseph->logLikelihood(), standard->logLikelihood());
}
