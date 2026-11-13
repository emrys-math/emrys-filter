// The factor operations of the information forms (roadmap step 16; decisions.md [gaussian-forms]): adding
// information against the Kalman update and the batch least-squares estimate, a prior with no
// information, moving the reference on a linear space and, by the reset Jacobian [covariance-reset], on a space
// with a unit quaternion, multiply() in both information forms, and marginal<Path>() and
// conditional<Path>() by tag path against the covariance form. Headers under test:
// Distributions/InfoGaussian.h and SqrtInfoGaussian.h. The accessors and
// conversions every form shares are in gaussian_forms.cpp.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Spaces/Reset.h>

#include <gtest/gtest.h>

#include "Spaces/unit_quaternion.h"

#include <Eigen/LU>

#include <limits>
#include <utility>

using Emrys::Composite;
using Emrys::GaussianError;
using Emrys::GaussianParams;
using Emrys::InfoGaussianParams;
using Emrys::Linear;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::resetJacobian;
using Emrys::SqrtInfoGaussianParams;
using EmrysTest::exponential;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Heading : RadiansPi<>
{};

struct Attitude : EmrysTest::UnitQuaternion
{};

struct Offset : Linear<3>
{};

using Track = Composite<Position, Velocity>;
using Pose = Composite<Position, Heading>;
using Rigid = Composite<Attitude, Offset>;

Eigen::Matrix4d
makeCov()
{
    return Eigen::Matrix4d{{2.0, 0.3, 0.5, 0.0}, {0.3, 1.5, 0.1, 0.4}, {0.5, 0.1, 1.0, 0.2}, {0.0, 0.4, 0.2, 0.8}};
}

Eigen::Vector4d const mean4{1.0, -1.0, 0.5, 2.0};

// The Kalman update of (mean, cov) by z = H x + v, v ~ N(0, R), in plain Eigen.
struct Posterior
{
    Eigen::Vector4d mean;
    Eigen::Matrix4d cov;
};

Posterior
kalmanUpdate(
    Eigen::Vector4d const & mean,
    Eigen::Matrix4d const & cov,
    Eigen::Matrix<double, 2, 4> const & h,
    Eigen::Matrix2d const & r,
    Eigen::Vector2d const & z
)
{
    Eigen::Matrix2d const s = h * cov * h.transpose() + r;
    Eigen::Matrix<double, 4, 2> const gain = cov * h.transpose() * s.inverse();
    return Posterior{mean + gain * (z - h * mean), cov - gain * s * gain.transpose()};
}

// Adds the information of z = H x + v to g, linearized at its reference.
void
addMeasurement(
    InfoGaussianParams<Track> & g,
    Eigen::Matrix<double, 2, 4> const & h,
    Eigen::Matrix2d const & r,
    Eigen::Vector2d const & z
)
{
    Eigen::Matrix2d const rInverse = r.inverse();
    Eigen::Matrix4d const matrix = h.transpose() * rInverse * h;
    Eigen::Vector4d const vector = h.transpose() * rInverse * (z - h * g.reference());
    g.addInfo(matrix, vector);
}

Eigen::Matrix<double, 2, 4> const positionRows{{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}};
Eigen::Matrix<double, 2, 4> const mixedRows{{0.0, 1.0, 1.0, 0.0}, {1.0, 0.0, 0.0, -1.0}};
Eigen::Matrix2d const noise{{0.5, 0.1}, {0.1, 0.3}};

// A Gaussian over an attitude and an offset: a quaternion mean away from the identity and a
// covariance over the six differences.
GaussianParams<Rigid>
rigid()
{
    auto mean = Eigen::Matrix<double, 7, 1>{};
    mean << exponential(Eigen::Vector3d{0.3, -0.2, 0.1}), 1.0, 2.0, 3.0;
    Eigen::Matrix<double, 6, 6> const a = Eigen::Matrix<double, 6, 6>::NullaryExpr([](Eigen::Index i, Eigen::Index j) {
        return std::sin(1.7 * double(i) + 0.9 * double(j) + 0.5);
    });
    Eigen::Matrix<double, 6, 6> const cov = a * a.transpose() + 0.5 * Eigen::Matrix<double, 6, 6>::Identity();
    return GaussianParams<Rigid>{mean, cov};
}

}  // namespace

// Two measurements added as information give the sequential Kalman updates' posterior, in either
// order. The reference is the plain-Eigen Kalman update above.
TEST(InfoGaussianParams, AddedInfoEqualsKalmanUpdates)
{
    auto const first = Eigen::Vector2d{1.2, -0.7};
    auto const second = Eigen::Vector2d{-0.4, -1.5};
    Posterior const once = kalmanUpdate(mean4, makeCov(), positionRows, noise, first);
    Posterior const expected = kalmanUpdate(once.mean, once.cov, mixedRows, noise, second);

    auto forward = Emrys::toInfo(GaussianParams<Track>{mean4, makeCov()});
    auto backward = forward;
    ASSERT_TRUE(forward && backward);
    addMeasurement(*forward, positionRows, noise, first);
    addMeasurement(*forward, mixedRows, noise, second);
    addMeasurement(*backward, mixedRows, noise, second);
    addMeasurement(*backward, positionRows, noise, first);
    for (auto const * g : {&*forward, &*backward})
    {
        EXPECT_TRUE(g->mean()->isApprox(expected.mean, 1e-12));
        EXPECT_TRUE(g->covariance()->isApprox(expected.cov, 1e-12));
    }
}

// From no information at all (Λ = 0 about a reference), the mean is Singular until the
// measurements determine every direction, and then it is the batch least-squares estimate
// x_0 + (Σ H^T R^{-1} H)^{-1} Σ H^T R^{-1} (z - H x_0), computed here in plain Eigen.
TEST(InfoGaussianParams, NoInfoBecomesLeastSquares)
{
    auto const reference = Eigen::Vector4d{0.1, 0.2, 0.3, 0.4};
    auto g = InfoGaussianParams<Track>{reference, Eigen::Matrix4d::Zero(), Eigen::Vector4d::Zero()};
    auto const first = Eigen::Vector2d{1.2, -0.7};
    auto const second = Eigen::Vector2d{-0.4, -1.5};

    addMeasurement(g, positionRows, noise, first);
    EXPECT_EQ(g.mean().error(), GaussianError::Singular);
    addMeasurement(g, mixedRows, noise, second);

    auto h = Eigen::Matrix<double, 4, 4>{};
    h << positionRows, mixedRows;
    Eigen::Matrix4d r = Eigen::Matrix4d::Zero();
    r.topLeftCorner<2, 2>() = noise;
    r.bottomRightCorner<2, 2>() = noise;
    auto z = Eigen::Vector4d{};
    z << first, second;
    Eigen::Matrix4d const normal = h.transpose() * r.inverse() * h;
    Eigen::Vector4d const expected = reference + normal.inverse() * h.transpose() * r.inverse() * (z - h * reference);
    ASSERT_TRUE(g.mean());
    EXPECT_TRUE(g.mean()->isApprox(expected, 1e-12));
    EXPECT_TRUE(g.covariance()->isApprox(normal.inverse(), 1e-12));
}

// Moving the reference re-expresses η and leaves the distribution unchanged on a linear space,
// in both information forms, and keeps the information form's factorization.
TEST(InfoGaussianParams, MovingTheReferenceKeepsTheDistribution)
{
    auto const g = GaussianParams<Track>{mean4, makeCov()};
    auto info = Emrys::toInfo(g);
    auto rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(info && rootInfo);
    ASSERT_TRUE(info->factorize());
    auto const reference = Eigen::Vector4d{5.0, -3.0, 0.0, 1.0};
    info->setReference(reference);
    rootInfo->setReference(reference);
    EXPECT_EQ(info->reference(), reference);
    EXPECT_EQ(rootInfo->reference(), reference);
    EXPECT_TRUE(info->mean()->isApprox(mean4, 1e-12));
    EXPECT_TRUE(info->covariance()->isApprox(makeCov(), 1e-12));
    EXPECT_TRUE(rootInfo->mean()->isApprox(mean4, 1e-12));
    EXPECT_TRUE(rootInfo->covariance()->isApprox(makeCov(), 1e-12));
    EXPECT_TRUE(rootInfo->infoRoot().isApprox(Eigen::Matrix4d{makeCov().inverse().llt().matrixU()}, 1e-12));
}

// On a space with a unit quaternion the reference moves by the reset Jacobian [covariance-reset]: with G the
// Jacobian at the old reference and the step d, the precision becomes G^{-T} Λ G^{-1} and the
// information vector G^{-T} (η - Λ d), in both information forms, and the covariance is then
// G Σ G^T, what resetCovariance() gives the covariance form. Moved from the mean, the mean stays
// put exactly: the offset becomes -G d, and the right Jacobian of SO(3) has G d = d, since the
// exponential map is a one-parameter subgroup along d. Moved again from there, with a nonzero
// information vector, the transport is first order and the mean moves a little. The reference
// values are resetJacobian() and plain Eigen.
TEST(InfoGaussianParams, MovingTheReferenceOnAQuaternionUsesTheResetJacobian)
{
    GaussianParams<Rigid> const g = rigid();
    Eigen::Matrix<double, 6, 6> const lambda = g.covarianceLower().inverse();
    auto const d = Eigen::Matrix<double, 6, 1>{0.2, -0.1, 0.15, 0.5, 0.0, -0.3};
    Eigen::Matrix<double, 7, 1> const moved = plus(Rigid{}, g.mean(), d);
    Eigen::Matrix<double, 6, 6> const jacobian = resetJacobian(Rigid{}, g.mean(), d);
    Eigen::Matrix<double, 6, 6> const inverse = jacobian.inverse();
    Eigen::Matrix<double, 6, 6> const expectedLambda = inverse.transpose() * lambda * inverse;
    Eigen::Matrix<double, 6, 1> const expectedEta = inverse.transpose() * (-lambda * d);

    auto info = Emrys::toInfo(g);
    auto rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(info && rootInfo);
    info->setReference(moved);
    rootInfo->setReference(moved);
    auto const check = [&](auto const & form) {
        EXPECT_TRUE(form.reference().isApprox(moved, 1e-15));
        EXPECT_TRUE(form.precision()->isApprox(expectedLambda, 1e-10));
        EXPECT_TRUE(form.infoVector().isApprox(expectedEta, 1e-10));
        EXPECT_TRUE(form.covariance()->isApprox(jacobian * g.covarianceLower() * jacobian.transpose(), 1e-10));
    };
    check(*info);
    check(*rootInfo);
    EXPECT_TRUE(rootInfo->infoRoot().triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));

    EXPECT_LT(minus(Rigid{}, info->mean().value(), g.mean()).norm(), 1e-12);
    EXPECT_LT(minus(Rigid{}, rootInfo->mean().value(), g.mean()).norm(), 1e-12);

    auto const again = Eigen::Matrix<double, 6, 1>{-0.1, 0.2, 0.1, 0.0, 0.4, 0.2};
    info->setReference(plus(Rigid{}, moved, again));
    double const drift = minus(Rigid{}, info->mean().value(), g.mean()).norm();
    EXPECT_GT(drift, 1e-6);
    EXPECT_LT(drift, 5e-2);
}

// multiply() of two Gaussians about different references gives their normalized product,
// Σ = (Σ1^{-1} + Σ2^{-1})^{-1} and μ = Σ (Σ1^{-1} μ1 + Σ2^{-1} μ2), in both information forms.
// The reference is the closed form in plain Eigen.
TEST(InfoGaussianParams, MultiplyIsTheProductOfDensities)
{
    Eigen::Matrix4d const firstCov = makeCov();
    Eigen::Matrix4d const secondCov = Eigen::Vector4d{0.5, 1.0, 2.0, 0.25}.asDiagonal();
    auto const secondMean = Eigen::Vector4d{0.0, 1.0, 1.0, -1.0};
    Eigen::Matrix4d const cov = (firstCov.inverse() + secondCov.inverse()).inverse();
    Eigen::Vector4d const mean = cov * (firstCov.inverse() * mean4 + secondCov.inverse() * secondMean);

    auto const first = GaussianParams<Track>{mean4, firstCov};
    auto const second = GaussianParams<Track>{secondMean, secondCov};
    auto const product = Emrys::multiply(Emrys::toInfo(first).value(), Emrys::toInfo(second).value());
    EXPECT_EQ(product.reference(), mean4);
    EXPECT_TRUE(product.mean()->isApprox(mean, 1e-12));
    EXPECT_TRUE(product.covariance()->isApprox(cov, 1e-12));

    auto const rootProduct = Emrys::multiply(Emrys::toSqrtInfo(first).value(), Emrys::toSqrtInfo(second).value());
    EXPECT_TRUE(rootProduct.infoRoot().triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
    EXPECT_TRUE(rootProduct.mean()->isApprox(mean, 1e-12));
    EXPECT_TRUE(rootProduct.covariance()->isApprox(cov, 1e-12));
}

// In the square-root information form a prior with no information (R = 0) multiplied by a
// Gaussian gives that Gaussian, where the information filter's prediction could not start
// ([gaussian-forms]'s table).
TEST(InfoGaussianParams, SqrtInfoFromNoInfo)
{
    auto const prior =
        SqrtInfoGaussianParams<Track>{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Zero(), Eigen::Vector4d::Zero()};
    auto const measured = Emrys::toSqrtInfo(GaussianParams<Track>{mean4, makeCov()});
    ASSERT_TRUE(measured);
    auto const product = Emrys::multiply(prior, *measured);
    EXPECT_TRUE(product.mean()->isApprox(mean4, 1e-12));
    EXPECT_TRUE(product.covariance()->isApprox(makeCov(), 1e-12));
}

// marginal<Path>() equals the covariance form's block in both information forms, for one block
// and for the other, on a space with a periodic coordinate, where the reference carries the
// marginal's mean, and for one repetition of a repeated block.
TEST(InfoGaussianParams, MarginalMatchesTheCovarianceForm)
{
    auto const g = GaussianParams<Track>{mean4, makeCov()};
    auto info = Emrys::toInfo(g);
    auto rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(info && rootInfo);
    info->setReference(Eigen::Vector4d{0.0, 0.0, 1.0, 1.0});
    rootInfo->setReference(Eigen::Vector4d{0.0, 0.0, 1.0, 1.0});

    auto const checkBlocks = [&](auto const & form) {
        auto const position = form.template marginal<Position>();
        ASSERT_TRUE(position);
        auto const expectedPosition = g.marginal<Position>().copy();
        EXPECT_TRUE(position->mean()->isApprox(expectedPosition.mean(), 1e-12));
        EXPECT_TRUE(position->covariance()->isApprox(expectedPosition.covarianceLower(), 1e-12));

        auto const velocity = form.template marginal<Velocity>();
        ASSERT_TRUE(velocity);
        auto const expectedVelocity = g.marginal<Velocity>().copy();
        EXPECT_TRUE(velocity->mean()->isApprox(expectedVelocity.mean(), 1e-12));
        EXPECT_TRUE(velocity->covariance()->isApprox(expectedVelocity.covarianceLower(), 1e-12));
    };
    checkBlocks(*info);
    checkBlocks(*rootInfo);
    auto const rootPosition = rootInfo->marginal<Position>();
    ASSERT_TRUE(rootPosition);
    EXPECT_TRUE(rootPosition->infoRoot().triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));

    Eigen::Matrix3d const poseCov = makeCov().topLeftCorner<3, 3>();
    auto const pose = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, poseCov};
    auto const heading = Emrys::toInfo(pose)->marginal<Heading>();
    auto const rootHeading = Emrys::toSqrtInfo(pose)->marginal<Heading>();
    ASSERT_TRUE(heading && rootHeading);
    EXPECT_NEAR(heading->mean().value()(0), pose.mean()(2), 1e-12);
    EXPECT_NEAR(heading->covariance().value()(0, 0), poseCov(2, 2), 1e-12);
    EXPECT_NEAR(rootHeading->mean().value()(0), pose.mean()(2), 1e-12);
    EXPECT_NEAR(rootHeading->covariance().value()(0, 0), poseCov(2, 2), 1e-12);

    struct Fleet : Repeated<Position, 3>
    {};

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.block<4, 4>(2, 2) = makeCov();
    auto const fleet = GaussianParams<Fleet>{Fleet{}, Eigen::Matrix<double, 6, 1>::LinSpaced(6, 0.0, 5.0), cov};
    auto const second = Emrys::toInfo(fleet)->marginal<Fleet>(1);
    auto const rootSecond = Emrys::toSqrtInfo(fleet)->marginal<Fleet>(1);
    ASSERT_TRUE(second && rootSecond);
    auto const expectedSecond = fleet.marginal<Fleet>(1).copy();
    EXPECT_TRUE(second->mean()->isApprox(expectedSecond.mean(), 1e-12));
    EXPECT_TRUE(second->covariance()->isApprox(expectedSecond.covarianceLower(), 1e-12));
    EXPECT_TRUE(rootSecond->mean()->isApprox(expectedSecond.mean(), 1e-12));
    EXPECT_TRUE(rootSecond->covariance()->isApprox(expectedSecond.covarianceLower(), 1e-12));
}

// Marginalizing out a block with no information is Singular in both information forms: the Schur
// complement needs the rest's precision to be nonsingular. Marginalizing the block that has
// information succeeds.
TEST(InfoGaussianParams, MarginalOverNoInfoIsSingular)
{
    Eigen::Matrix4d precision = Eigen::Matrix4d::Zero();
    precision.topLeftCorner<2, 2>() = Eigen::Matrix2d::Identity();
    auto const g = InfoGaussianParams<Track>{mean4, precision, Eigen::Vector4d::Zero()};
    EXPECT_EQ(g.marginal<Position>().error(), GaussianError::Singular);
    EXPECT_TRUE(g.marginal<Velocity>());

    Eigen::Matrix4d factor = Eigen::Matrix4d::Zero();
    factor.topLeftCorner<2, 2>() = Eigen::Matrix2d::Identity();
    auto const h = SqrtInfoGaussianParams<Track>{mean4, factor, Eigen::Vector4d::Zero()};
    EXPECT_EQ(h.marginal<Position>().error(), GaussianError::Singular);
    EXPECT_TRUE(h.marginal<Velocity>());

    // A NaN in the block summed out passes the factorization's test and fails as NonFinite.
    Eigen::Matrix4d nan = Eigen::Matrix4d::Identity();
    nan(3, 3) = std::numeric_limits<double>::quiet_NaN();
    auto const n = InfoGaussianParams<Track>{mean4, nan, Eigen::Vector4d::Zero()};
    EXPECT_EQ(n.marginal<Position>().error(), GaussianError::NonFinite);
}

// Moving or copying parameters keeps their factorization when it is valid, so the copy evaluates
// without factorizing again, and gives the same log-density. setInfo() replaces the information
// about the reference, which a factorization then evaluates.
TEST(InfoGaussianParams, MovesAndCopiesKeepAValidFactorization)
{
    auto info = InfoGaussianParams<Track>{mean4, Eigen::Matrix4d::Identity() * 2, Eigen::Vector4d::Zero()};
    ASSERT_TRUE(info.factorize());
    double const logDensity = info.toDensity()->logDensity(mean4);
    auto moved = InfoGaussianParams<Track>{std::move(info)};
    EXPECT_EQ(moved.toDensity()->logDensity(mean4), logDensity);
    auto assigned = InfoGaussianParams<Track>{mean4, Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero()};
    assigned = std::move(moved);
    EXPECT_EQ(assigned.toDensity()->logDensity(mean4), logDensity);
    auto infoCopy = InfoGaussianParams<Track>{mean4, Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero()};
    infoCopy = assigned;
    EXPECT_EQ(infoCopy.toDensity()->logDensity(mean4), logDensity);
    auto reset = InfoGaussianParams<Track>{mean4, Eigen::Matrix4d::Identity(), Eigen::Vector4d::Ones()};
    reset.setInfo(Eigen::Matrix4d{Eigen::Matrix4d::Identity() * 2}, Eigen::Vector4d{Eigen::Vector4d::Zero()});
    ASSERT_TRUE(reset.factorize());
    EXPECT_NEAR(reset.toDensity()->logDensity(mean4), logDensity, 1e-14);

    auto gaussian = GaussianParams<Track>{mean4, Eigen::Matrix4d::Identity() * 3};
    ASSERT_TRUE(gaussian.factorize());
    auto copied = GaussianParams<Track>{mean4, Eigen::Matrix4d::Identity()};
    copied = gaussian;
    EXPECT_EQ(copied.toDensity()->logDensity(mean4), gaussian.toDensity()->logDensity(mean4));
}

// conditional<Path>(values) takes a block of Λ and shifts η, in the square-root form by a
// triangularization, and equals the covariance form's conditional, with a reference away from the
// mean.
TEST(InfoGaussianParams, ConditionalMatchesTheCovarianceForm)
{
    auto const g = GaussianParams<Track>{mean4, makeCov()};
    auto info = Emrys::toInfo(g);
    auto rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(info && rootInfo);
    info->setReference(Eigen::Vector4d{0.0, 0.0, 1.0, 1.0});
    rootInfo->setReference(Eigen::Vector4d{0.0, 0.0, 1.0, 1.0});
    auto const values = Eigen::Vector2d{0.3, 0.9};
    auto const expected = g.conditional<Velocity>(values);

    auto const actual = info->conditional<Velocity>(values);
    static_assert(std::is_same_v<decltype(actual), InfoGaussianParams<Position> const>);
    EXPECT_TRUE(actual.mean()->isApprox(expected.mean(), 1e-12));
    EXPECT_TRUE(actual.covariance()->isApprox(expected.covarianceLower(), 1e-12));

    auto const rootActual = rootInfo->conditional<Velocity>(values);
    static_assert(std::is_same_v<decltype(rootActual), SqrtInfoGaussianParams<Position> const>);
    EXPECT_TRUE(rootActual.infoRoot().triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
    EXPECT_TRUE(rootActual.mean()->isApprox(expected.mean(), 1e-12));
    EXPECT_TRUE(rootActual.covariance()->isApprox(expected.covarianceLower(), 1e-12));

    auto const givenPosition = rootInfo->conditional<Position>(values);
    auto const expectedGivenPosition = g.conditional<Position>(values);
    EXPECT_TRUE(givenPosition.mean()->isApprox(expectedGivenPosition.mean(), 1e-12));
    EXPECT_TRUE(givenPosition.covariance()->isApprox(expectedGivenPosition.covarianceLower(), 1e-12));
}
