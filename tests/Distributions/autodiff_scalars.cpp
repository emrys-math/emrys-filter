// The continuous densities over Eigen's AutoDiffScalar, a scalar that RealScalar admits and that
// is not floating point: Gaussian, StudentT, and Gamma evaluate their log-densities
// in it, and the Gaussian its confidence through Core/ChiSquared.h, with the value of the double
// evaluation and the closed-form gradient. A follow-up to
// roadmap step 13b's review findings, which found two headers that did not compile for such a
// scalar; the constants and log-gamma they need come from Core/Math.h.

#include <EmrysFilter/Distributions/Gamma.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/StudentT.h>

#include <gtest/gtest.h>

#include <unsupported/Eigen/AutoDiff>

using Emrys::Gamma;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::StudentT;

namespace
{

// Three variables: the two coordinates of the point, and a parameter (the Student-t degrees of
// freedom) where a test needs one.
using Dual = Eigen::AutoDiffScalar<Eigen::Vector3d>;

using DualVector = Eigen::Matrix<Dual, 2, 1>;
using DualMatrix = Eigen::Matrix<Dual, 2, 2>;

static_assert(Emrys::RealScalar<Dual>);

// The point x as the first two variables.
DualVector
variables(Eigen::Vector2d const & x)
{
    return DualVector{Dual{x(0), 3, 0}, Dual{x(1), 3, 1}};
}

Eigen::Vector2d const mu{1.0, -1.0};
Eigen::Matrix2d const sigma{{2.0, 0.5}, {0.5, 1.0}};
Eigen::Vector2d const x{2.5, 0.5};

}  // namespace

// The Gaussian log-density in Dual has the double evaluation's value and the gradient
// -Sigma^-1 (x - mu) in the point, zero in the unused third variable.
TEST(AutoDiffScalars, GaussianLogDensityAndItsGradient)
{
    auto const gaussian = GaussianParams<Linear<2>, Dual>{DualVector{mu.cast<Dual>()}, DualMatrix{sigma.cast<Dual>()}};
    auto const density = gaussian.toDensity();
    ASSERT_TRUE(density.has_value());
    Dual const logDensity = density->logDensity(variables(x));

    auto const reference = GaussianParams<Linear<2>>{mu, sigma};
    EXPECT_NEAR(logDensity.value(), reference.toDensity()->logDensity(x), 1e-13);
    Eigen::Vector2d const gradient = -sigma.ldlt().solve(x - mu);
    EXPECT_NEAR(logDensity.derivatives()(0), gradient(0), 1e-13);
    EXPECT_NEAR(logDensity.derivatives()(1), gradient(1), 1e-13);
    EXPECT_EQ(logDensity.derivatives()(2), 0.0);
}

// The confidence in Dual has the double evaluation's value and, in two dimensions, where it is
// 1 - exp(-q / 2) of the squared Mahalanobis distance q, the gradient exp(-q / 2) Sigma^-1 (x - mu)
// in the point: the chi-squared CDF carries the derivative of the squared Mahalanobis distance rather than dropping it.
// The tolerance allows for the series stopping at the value's epsilon, not the derivative's.
TEST(AutoDiffScalars, GaussianConfidenceAndItsGradient)
{
    auto const gaussian = GaussianParams<Linear<2>, Dual>{DualVector{mu.cast<Dual>()}, DualMatrix{sigma.cast<Dual>()}};
    auto const density = gaussian.toDensity();
    ASSERT_TRUE(density.has_value());
    Dual const confidence = density->confidence(variables(x));

    auto const reference = GaussianParams<Linear<2>>{mu, sigma};
    EXPECT_NEAR(confidence.value(), reference.toDensity()->confidence(x), 1e-14);
    Eigen::Vector2d const solved = sigma.ldlt().solve(x - mu);
    Eigen::Vector2d const gradient = std::exp(-(x - mu).dot(solved) / 2) * solved;
    EXPECT_NEAR(confidence.derivatives()(0), gradient(0), 1e-12);
    EXPECT_NEAR(confidence.derivatives()(1), gradient(1), 1e-12);
    EXPECT_EQ(confidence.derivatives()(2), 0.0);
}

// The Student-t log-density in Dual has the double evaluation's value, the gradient
// -(nu + d) / (nu + q) Sigma^-1 (x - mu) in the point, with q the squared Mahalanobis distance,
// and a derivative in nu, which runs through the log-gamma normalizer, matching a central
// difference of the double density with step 1e-5 (truncation error about 1e-11).
TEST(AutoDiffScalars, StudentTLogDensityAndItsGradient)
{
    double const nu = 3.0;
    auto const t =
        StudentT<Linear<2>, Dual>::create(DualVector{mu.cast<Dual>()}, DualMatrix{sigma.cast<Dual>()}, Dual{nu, 3, 2});
    ASSERT_TRUE(t.has_value());
    Dual const logDensity = t->logDensity(variables(x));

    auto const reference = [&](double dof) {
        return StudentT<Linear<2>>::create(mu, sigma, dof)->logDensity(x);
    };
    EXPECT_NEAR(logDensity.value(), reference(nu), 1e-13);
    Eigen::Vector2d const solved = sigma.ldlt().solve(x - mu);
    double const q = (x - mu).dot(solved);
    Eigen::Vector2d const gradient = -(nu + 2) / (nu + q) * solved;
    EXPECT_NEAR(logDensity.derivatives()(0), gradient(0), 1e-13);
    EXPECT_NEAR(logDensity.derivatives()(1), gradient(1), 1e-13);
    double const h = 1e-5;
    EXPECT_NEAR(logDensity.derivatives()(2), (reference(nu + h) - reference(nu - h)) / (2 * h), 1e-8);
}

// The gamma log-density in Dual has the double evaluation's value and the gradient
// (k_i - 1) / x_i - 1 / theta_i in the point.
TEST(AutoDiffScalars, GammaLogDensityAndItsGradient)
{
    auto const shape = Eigen::Vector2d{2.5, 1.0};
    auto const scale = Eigen::Vector2d{1.5, 3.0};
    auto const point = Eigen::Vector2d{2.0, 0.5};
    auto const gamma =
        Gamma<Linear<2>, Dual>::create(DualVector{shape.cast<Dual>()}, DualVector{scale.cast<Dual>()}).value();
    Dual const logDensity = gamma.logDensity(variables(point));

    auto const reference = Gamma<Linear<2>>::create(shape, scale).value();
    EXPECT_NEAR(logDensity.value(), reference.logDensity(point), 1e-13);
    Eigen::Array2d const gradient = (shape.array() - 1) / point.array() - 1 / scale.array();
    EXPECT_NEAR(logDensity.derivatives()(0), gradient(0), 1e-13);
    EXPECT_NEAR(logDensity.derivatives()(1), gradient(1), 1e-13);
    EXPECT_EQ(logDensity.derivatives()(2), 0.0);
}
