// GaussianParams and the Gaussian view (baseline; [sizes-and-dimensions], [density-names], [forwarded-matrices]): the
// log-density against a reference, lazy Cholesky factorization and refactorization, failure on indefinite or non-finite
// covariances and recovery, the Cholesky root after roots::ridge() and the repair by a Repair, the ridge's
// or one of the user's own ([covariance-repair]), confidence scaling against the chi-squared quantile, the check a view
// keeps of its parameters [view-stamp]: no density of a temporary, and the death tests for reading a density after its
// parameters were changed or moved. The typed tests run over fixed and dynamic sizes.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "user_axes.h"

using Emrys::GaussianError;
using Emrys::GaussianParams;
using Emrys::Linear;

namespace roots = Emrys::roots;

namespace
{

// Lets std::vector move rather than copy on reallocation.
static_assert(std::is_nothrow_move_constructible_v<GaussianParams<Linear<Eigen::Dynamic>>>);

static_assert(std::is_nothrow_move_constructible_v<GaussianParams<Linear<3>>>);

// Each config names a Gaussian type and the runtime dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config
{
    using Type = GaussianParams<Linear<Size, MaxSize>, Scalar>;
    static constexpr Eigen::Index n = N;
};

using Configs = ::testing::Types<
    Config<double, 1, 1, 1>,
    Config<double, 3, 3, 3>,
    Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
    Config<double, Eigen::Dynamic, 6, 4>,
    Config<float, 2, 2, 2>>;

template <typename C>
class GaussianTest : public ::testing::Test
{
protected:

    using MG = typename C::Type;
    using Scalar = typename MG::Scalar;
    using VectorType = typename MG::VectorType;
    using MatrixType = typename MG::MatrixType;
    static constexpr Eigen::Index n = C::n;

    static Scalar
    tolerance()
    {
        return std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10);
    }

    static VectorType
    makeVector(double phase)
    {
        return VectorType::NullaryExpr(n, [phase](Eigen::Index i) {
            return Scalar(std::sin(1.3 * i + phase));
        });
    }

    // Deterministic symmetric positive definite matrix: A·Aᵀ + n·I.
    static MatrixType
    makeSpd()
    {
        MatrixType const a = MatrixType::NullaryExpr(n, n, [](Eigen::Index i, Eigen::Index j) {
            return Scalar(std::sin(7.0 * i + 3.0 * j + 1.0));
        });
        MatrixType p = a * a.transpose();
        p.diagonal().array() += Scalar(n);
        return p;
    }

    // Direct evaluation through determinant and inverse, in double precision.
    static double
    referenceLogDensity(VectorType const & mean, MatrixType const & cov, VectorType const & x)
    {
        Eigen::MatrixXd const p = cov.template cast<double>();
        Eigen::VectorXd const r = (x - mean).template cast<double>();
        double const squaredDistance = r.dot(p.inverse() * r);
        return -0.5 * (double(n) * std::log(2.0 * std::numbers::pi) + std::log(p.determinant()) + squaredDistance);
    }

    MG
    make() const
    {
        return MG{makeVector(0.0), makeSpd()};
    }
};

}  // namespace

TYPED_TEST_SUITE(GaussianTest, Configs);

// logDensity() matches a direct evaluation through the determinant and inverse in double, density()
// is its exponential, and logDeterminant() matches the determinant's log, for every configured size
// and scalar.
TYPED_TEST(GaussianTest, LogDensityMatchesReference)
{
    auto const mg = this->make();
    auto const g = mg.toDensity();
    ASSERT_TRUE(g.has_value());

    auto const x = this->makeVector(0.7);
    double const expected = this->referenceLogDensity(mg.mean(), mg.covarianceLower(), x);
    EXPECT_NEAR(g->logDensity(x), expected, this->tolerance() * std::abs(expected));
    EXPECT_NEAR(g->density(x), std::exp(g->logDensity(x)), this->tolerance());
    EXPECT_NEAR(
        g->logDeterminant(),
        std::log(mg.covarianceLower().template cast<double>().determinant()),
        this->tolerance() * 10
    );
}

// The squared Mahalanobis distance at the mean is exactly zero and the log-density there is logPeak().
TYPED_TEST(GaussianTest, PeakIsAtMean)
{
    auto const mg = this->make();
    auto const g = mg.toDensity();
    ASSERT_TRUE(g.has_value());
    EXPECT_EQ(g->squaredMahalanobisDistance(mg.mean()), 0);
    EXPECT_EQ(g->logDensity(mg.mean()), g->logPeak());
}

// Overwriting the strict upper triangle of the covariance changes nothing: only the lower triangle
// is read.
TYPED_TEST(GaussianTest, ReadsOnlyLowerTriangle)
{
    auto mg = this->make();
    auto const x = this->makeVector(0.7);
    auto const expected = mg.toDensity()->logDensity(x);

    auto cov = mg.covarianceLower();
    cov.template triangularView<Eigen::StrictlyUpper>().setConstant(12345);
    mg.setCovariance(cov);
    EXPECT_EQ(mg.toDensity()->logDensity(x), expected);
}

// setMean() keeps the Cholesky factorization, so the peak is unchanged and sits at the new mean.
TYPED_TEST(GaussianTest, MeanChangeKeepsFactorization)
{
    auto mg = this->make();
    auto const logPeak = mg.toDensity()->logPeak();

    mg.setMean(this->makeVector(2.0));
    auto const g = mg.toDensity();
    EXPECT_EQ(g->logPeak(), logPeak);
    EXPECT_EQ(g->logDensity(this->makeVector(2.0)), logPeak);
}

// setCovariance() refactorizes: scaling the covariance by 4 raises the log-determinant by n log 4.
TYPED_TEST(GaussianTest, CovChangeRefactorizes)
{
    using Scalar = typename TestFixture::Scalar;
    auto mg = this->make();
    auto const logDet = mg.toDensity()->logDeterminant();

    mg.setCovariance(mg.covarianceLower() * Scalar(4));
    auto const expected = logDet + Scalar(this->n) * std::log(Scalar(4));
    EXPECT_NEAR(mg.toDensity()->logDeterminant(), expected, this->tolerance() * 10);
}

// toDensity() fails with NotPositiveDefinite for a negative covariance.
TYPED_TEST(GaussianTest, NotPositiveDefinite)
{
    using MatrixType = typename TestFixture::MatrixType;
    auto mg = this->make();
    mg.setCovariance(-MatrixType::Identity(this->n, this->n));
    auto const g = mg.toDensity();
    ASSERT_FALSE(g.has_value());
    EXPECT_EQ(g.error(), GaussianError::NotPositiveDefinite);
}

// toDensity() fails with NonFinite when the covariance holds a NaN.
TYPED_TEST(GaussianTest, NonFinite)
{
    using Scalar = typename TestFixture::Scalar;
    auto mg = this->make();
    auto cov = mg.covarianceLower();
    cov(this->n - 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    mg.setCovariance(cov);
    auto const g = mg.toDensity();
    ASSERT_FALSE(g.has_value());
    EXPECT_EQ(g.error(), GaussianError::NonFinite);
}

// After a failed factorization, a valid covariance factorizes again.
TYPED_TEST(GaussianTest, RecoversAfterFailure)
{
    using MatrixType = typename TestFixture::MatrixType;
    auto mg = this->make();
    mg.setCovariance(-MatrixType::Identity(this->n, this->n));
    ASSERT_FALSE(mg.toDensity().has_value());
    mg.setCovariance(this->makeSpd());
    EXPECT_TRUE(mg.toDensity().has_value());
}

// factorize() checks the covariance without changing what a view reads, so a density made before
// it stays readable, and it reports the same error toDensity() would.
TYPED_TEST(GaussianTest, FactorizeChangesNothingAViewReads)
{
    using MatrixType = typename TestFixture::MatrixType;
    auto mg = this->make();
    auto const g = mg.toDensity();
    EXPECT_TRUE(mg.factorize().has_value());
    EXPECT_EQ(g->logPeak(), mg.toDensity()->logPeak());  // would assert if factorize() changed the stamp

    mg.setCovariance(-MatrixType::Identity(this->n, this->n));
    auto const result = mg.factorize();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GaussianError::NotPositiveDefinite);
}

// The Cholesky root after the ridge, the one a repaired density factors, is a root of A + δI, with δ at
// least the shift the smallest eigenvalue needs.
TYPED_TEST(GaussianTest, RidgedCholeskyFactorsIndefiniteCovariance)
{
    using Scalar = typename TestFixture::Scalar;
    using MatrixType = typename TestFixture::MatrixType;
    MatrixType cov = this->makeSpd();
    cov(0, 0) = -Scalar(1);
    Eigen::Index const n = this->n;
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge()), cov);
    ASSERT_TRUE(root.has_value());

    MatrixType const ridge = *root * root->transpose() - cov;
    Scalar const delta = ridge(0, 0);
    Scalar const scale = cov.cwiseAbs().maxCoeff();
    EXPECT_TRUE(ridge.isApprox(delta * MatrixType::Identity(n, n), this->tolerance() * scale));
    auto const eigen = Eigen::SelfAdjointEigenSolver<MatrixType>{cov, Eigen::EigenvaluesOnly};
    EXPECT_GT(delta, -eigen.eigenvalues()(0));
}

// The ridge fails with NonFinite on a NaN entry.
TYPED_TEST(GaussianTest, RidgedCholeskyRejectsNonFinite)
{
    using Scalar = typename TestFixture::Scalar;
    auto cov = this->makeSpd();
    cov(this->n - 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge()), cov);
    ASSERT_FALSE(root.has_value());
    EXPECT_EQ(root.error(), GaussianError::NonFinite);
}

// The Gershgorin bound is exact here: [[1, 2], [2, 1]] has eigenvalues -1 and 3.
TEST(RidgedCholesky, ShiftsByExactBoundForTwoByTwo)
{
    auto const cov = Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}};
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge()), cov);
    ASSERT_TRUE(root.has_value());
    double const delta = 1.0 + std::sqrt(std::numeric_limits<double>::epsilon());
    Eigen::Matrix2d const reconstructed = *root * root->transpose();
    EXPECT_TRUE(reconstructed.isApprox(cov + delta * Eigen::Matrix2d::Identity(), 1e-14));
}

// A singular positive semidefinite matrix gets only the margin.
TEST(RidgedCholesky, FactorsSingularSemidefinite)
{
    auto const v = Eigen::Vector3d{1.0, -2.0, 0.5};
    Eigen::Matrix3d const cov = v * v.transpose();
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge()), cov);
    ASSERT_TRUE(root.has_value());
    Eigen::Matrix3d const ridge = *root * root->transpose() - cov;
    EXPECT_GT(ridge(0, 0), 0.0);
    EXPECT_TRUE(ridge.isDiagonal(1e-12));
}

// The ridge of a zero matrix is zero, so its Cholesky root fails with NotPositiveDefinite: there is no
// scale to set a ridge against.
TEST(RidgedCholesky, RejectsZeroMatrix)
{
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge()), Eigen::MatrixXd::Zero(3, 3));
    ASSERT_FALSE(root.has_value());
    EXPECT_EQ(root.error(), GaussianError::NotPositiveDefinite);
}

// The ridge fails with NonFinite on an infinite entry.
TEST(RidgedCholesky, RejectsInfinity)
{
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
    cov(1, 0) = std::numeric_limits<double>::infinity();
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge()), cov);
    ASSERT_FALSE(root.has_value());
    EXPECT_EQ(root.error(), GaussianError::NonFinite);
}

// The repair adds the same ridge to every diagonal element and leaves the rest alone.
TYPED_TEST(GaussianTest, RidgeRepairUpdatesCovariance)
{
    using Scalar = typename TestFixture::Scalar;
    using MatrixType = typename TestFixture::MatrixType;
    MatrixType cov = this->makeSpd();
    cov(0, 0) = -Scalar(1);
    auto mg = this->make();
    mg.setCovariance(cov);
    EXPECT_EQ(mg.toDensity().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(mg.covarianceLower(), cov);

    auto const g = mg.toDensity(roots::ridge());
    ASSERT_TRUE(g.has_value());
    MatrixType const ridge = mg.covarianceLower() - cov;
    EXPECT_GT(ridge(0, 0), Scalar(0));
    EXPECT_TRUE(ridge.diagonal().isConstant(ridge(0, 0), this->tolerance()));
    EXPECT_TRUE(ridge.template triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0));
    EXPECT_TRUE(Eigen::numext::isfinite(g->logDensity(mg.mean())));
}

// factorize(roots::ridge()) on a positive definite covariance leaves it exactly as it was.
TYPED_TEST(GaussianTest, RidgeRepairLeavesPositiveDefiniteCovariance)
{
    auto mg = this->make();
    auto const cov = mg.covarianceLower();
    ASSERT_TRUE(mg.factorize(roots::ridge()).has_value());
    EXPECT_EQ(mg.covarianceLower(), cov);
}

// A failed repair leaves the covariance as it was.
TYPED_TEST(GaussianTest, RidgeRepairFailureLeavesCovariance)
{
    using Scalar = typename TestFixture::Scalar;
    using MatrixType = typename TestFixture::MatrixType;
    auto mg = this->make();
    mg.setCovariance(MatrixType::Zero(this->n, this->n));
    EXPECT_EQ(mg.factorize(roots::ridge()).error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(mg.covarianceLower().isZero(0));

    MatrixType cov = -MatrixType::Identity(this->n, this->n);
    cov(this->n - 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    mg.setCovariance(cov);
    EXPECT_EQ(mg.toDensity(roots::ridge()).error(), GaussianError::NonFinite);
    EXPECT_TRUE((mg.covarianceLower().array() == cov.array() || cov.array().isNaN()).all());
}

// A repair of the user's own, an eigenvalue floor, is a Repair: the density repaired by it holds the
// covariance it returned, here diag(0.5, -0.5) raised to the identity, exactly since the eigenvectors of a
// diagonal matrix are the axes.
TEST(GaussianRepair, TakesARepairOfTheUsersOwn)
{
    auto const indefinite = Eigen::Matrix2d{{0.5, 0.0}, {0.0, -0.5}};
    auto g = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), indefinite};
    EXPECT_EQ(g.toDensity().error(), GaussianError::NotPositiveDefinite);
    ASSERT_TRUE(g.toDensity(EmrysTest::EigenvalueFloor<1.0>{}).has_value());
    Eigen::Matrix2d const repaired = g.covarianceLower();
    EXPECT_TRUE(repaired.isApprox(Eigen::Matrix2d::Identity(), 1e-15));
}

// scaleToConfidence(p) multiplies the covariance by the chi-squared quantile of p at n degrees of
// freedom, keeps the mean, and refactorizes so the peak drops by n log q / 2.
TYPED_TEST(GaussianTest, ScaleToConfidenceMultipliesByChiSquaredQuantile)
{
    using Scalar = typename TestFixture::Scalar;
    auto mg = this->make();
    auto const mean = mg.mean();
    auto const cov = mg.covarianceLower();
    Scalar const logDensity = mg.toDensity()->logDensity(mean);
    mg.scaleToConfidence(Scalar(0.95));
    Scalar const quantile = Emrys::chiSquaredQuantile<Scalar>(Scalar(0.95), Scalar(this->n));
    EXPECT_TRUE(mg.covarianceLower().isApprox(quantile * cov, this->tolerance()));
    EXPECT_EQ(mg.mean(), mean);
    // The factorization is redone: the peak density drops by half the log-determinant change.
    Scalar const expected = logDensity - Scalar(this->n) * Eigen::numext::log(quantile) / 2;
    EXPECT_NEAR(mg.toDensity()->logDensity(mean), expected, this->tolerance() * 10);
}

// In two dimensions the quantile is -2 log(1 - p), so the 1 - e^{-1/2} region is the
// one-standard-deviation ellipse and leaves the covariance as it was.
TEST(GaussianConfidence, TwoDimensionalQuantileIsClosedForm)
{
    auto g = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}};
    Eigen::Matrix2d const cov = g.covarianceLower();
    g.scaleToConfidence(1 - std::exp(-0.5));
    EXPECT_TRUE(g.covarianceLower().isApprox(cov, 1e-12));
    g.scaleToConfidence(0.99);
    EXPECT_TRUE(g.covarianceLower().isApprox(-2 * std::log(0.01) * cov, 1e-12));
}

// In two dimensions the confidence of a value at Mahalanobis distance r is 1 - e^{-r^2 / 2}.
TEST(GaussianConfidence, TwoDimensionalConfidenceIsClosedForm)
{
    auto g = GaussianParams<Linear<2>>{Eigen::Vector2d{1.0, -1.0}, Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}};
    auto const density = g.toDensity();
    ASSERT_TRUE(density.has_value());
    Eigen::Vector2d const x = g.mean() + density->llt().matrixL() * Eigen::Vector2d{0.0, std::sqrt(2.0)};
    EXPECT_NEAR(density->confidence(x), 1 - std::exp(-1.0), 1e-12);
    EXPECT_TRUE(density->contains(x, 0.64));
    EXPECT_FALSE(density->contains(x, 0.63));
}

// contains() is the unit ellipsoid of the Gaussian that scaleToConfidence() makes.
TEST(GaussianConfidence, ContainsMatchesScaledUnitEllipsoid)
{
    auto g = GaussianParams<Linear<3>>{
        Eigen::Vector3d::Zero(),
        Eigen::Matrix3d{{2.0, 0.5, 0.0}, {0.5, 1.0, 0.2}, {0.0, 0.2, 3.0}}
    };
    GaussianParams<Linear<3>> scaled = g;
    scaled.scaleToConfidence(0.9);
    auto const density = g.toDensity();
    auto const scaledDensity = scaled.toDensity();
    ASSERT_TRUE(density.has_value() && scaledDensity.has_value());
    for (double const r : {0.5, 1.0, 1.5, 2.0, 2.5, 3.0})
    {
        Eigen::Vector3d const x = r * Eigen::Vector3d{1.0, -0.5, 0.3}.normalized();
        EXPECT_EQ(density->contains(x, 0.9), scaledDensity->squaredMahalanobisDistance(x) <= 1.0) << r;
    }
}

// confidence(x) is the chi-squared CDF of the squared Mahalanobis distance at n degrees of freedom, zero at the mean,
// and contains(x, level) is true just above that confidence and false just below. Both compute in the Gaussian's
// scalar, so the float configuration matches the double CDF to the fixture's float tolerance.
TYPED_TEST(GaussianTest, ConfidenceIsChiSquaredCdfOfSquaredMahalanobisDistance)
{
    using Scalar = typename TestFixture::Scalar;
    using VectorType = typename TestFixture::VectorType;
    auto const mg = this->make();
    auto const g = mg.toDensity();
    ASSERT_TRUE(g.has_value());
    EXPECT_EQ(g->confidence(mg.mean()), 0.0);
    EXPECT_TRUE(g->contains(mg.mean(), Scalar(0.01)));

    VectorType const x = mg.mean() + this->makeVector(0.5);
    Scalar const level = g->confidence(x);
    static_assert(std::is_same_v<decltype(g->confidence(x)), Scalar>);
    EXPECT_NEAR(
        level,
        Emrys::chiSquaredCdf<double>(double(g->squaredMahalanobisDistance(x)), double(this->n)),
        this->tolerance()
    );
    ASSERT_GT(level, Scalar(0));
    ASSERT_LT(level, Scalar(1));
    EXPECT_TRUE(g->contains(x, std::min(Scalar(1.01) * level, (1 + level) / 2)));
    EXPECT_FALSE(g->contains(x, Scalar(0.99) * level));
}

// Copies and moves of a view read the same density, and views that are not read again leave the
// parameters free to change [view-stamp].
TYPED_TEST(GaussianTest, ViewsCopyAndMoveFreely)
{
    auto mg = this->make();
    {
        auto g = mg.toDensity().value();
        auto const copy = g;
        auto const moved = std::move(g);
        EXPECT_EQ(copy.logPeak(), moved.logPeak());
    }
    mg.setCovariance(mg.covarianceLower());
    mg.resize(this->n);
    EXPECT_TRUE(mg.toDensity().has_value());
}

// A copy of parameters is new parameters: changing it leaves the source's density readable.
TYPED_TEST(GaussianTest, ChangingACopyLeavesTheSourceDensity)
{
    auto mg = this->make();
    auto const g = mg.toDensity();
    double const peak = g->logPeak();
    auto copy = mg;
    copy.setCovariance(copy.covarianceLower() * 4.0);
    EXPECT_EQ(g->logPeak(), peak);  // would assert if the copy shared the source's stamp
    EXPECT_NE(copy.toDensity()->logPeak(), peak);
}

/// Whether \a G, with its value category, can make a density.
template <typename G>
concept MakesDensity = requires(G && g) { std::forward<G>(g).toDensity(); };

// A density of a temporary would dangle at the end of the statement, so toDensity() on an rvalue
// does not compile, while a named Gaussian's does [view-stamp].
TEST(GaussianParams, NoDensityOfATemporary)
{
    using MG = GaussianParams<Linear<2>>;
    static_assert(MakesDensity<MG &>);
    static_assert(MakesDensity<MG const &>);
    static_assert(!MakesDensity<MG>);
    static_assert(!MakesDensity<MG const>);
}

// A dynamic Gaussian resizes and factorizes again at the new dimension.
TEST(Gaussian, ResizeDynamic)
{
    auto mg = GaussianParams<Linear<Eigen::Dynamic>>{2};
    mg.setMean(Eigen::Vector2d::Zero());
    mg.setCovariance(Eigen::Matrix2d::Identity());
    ASSERT_TRUE(mg.toDensity().has_value());

    mg.resize(3);
    EXPECT_EQ(mg.valueDimension(), 3);
    mg.setMean(Eigen::Vector3d::Zero());
    mg.setCovariance(Eigen::Matrix3d::Identity() * 2.0);
    auto const g = mg.toDensity();
    ASSERT_TRUE(g.has_value());
    EXPECT_EQ(g->differenceDimension(), 3);
    EXPECT_NEAR(g->logDeterminant(), 3.0 * std::log(2.0), 1e-12);
}

// The constructor, setMean(), and setCovariance() move an rvalue of the stored type (same data pointer)
// and copy lvalues and expressions [forwarded-matrices]; the constructor from moments sizes the
// space to them.
TEST(GaussianParams, MovesRvalueMeanAndCov)
{
    Eigen::VectorXd mean = Eigen::Vector2d{1.0, 2.0};
    Eigen::MatrixXd cov = Eigen::Matrix2d::Identity();
    double const * meanData = mean.data();
    double const * covData = cov.data();
    auto g = GaussianParams<Linear<Eigen::Dynamic>>{std::move(mean), std::move(cov)};
    EXPECT_EQ(g.mean().data(), meanData);
    EXPECT_EQ(g.covarianceLower().data(), covData);

    Eigen::VectorXd newMean = Eigen::Vector2d{3.0, 4.0};
    Eigen::MatrixXd newCov = 2.0 * Eigen::Matrix2d::Identity();
    meanData = newMean.data();
    covData = newCov.data();
    g.setMean(std::move(newMean));
    g.setCovariance(std::move(newCov));
    EXPECT_EQ(g.mean().data(), meanData);
    EXPECT_EQ(g.covarianceLower().data(), covData);

    // Lvalues and expressions are still copied.
    Eigen::VectorXd const kept = Eigen::Vector2d{5.0, 6.0};
    g.setMean(kept);
    EXPECT_NE(g.mean().data(), kept.data());
    EXPECT_EQ(g.mean(), kept);
    g.setCovariance(g.covarianceLower() * 2.0);
    EXPECT_EQ(g.covarianceLower()(0, 0), 4.0);

    auto const fromMoments = GaussianParams<Linear<Eigen::Dynamic>>{g.moments()};
    EXPECT_EQ(fromMoments.valueDimension(), 2);
    EXPECT_EQ(fromMoments.mean(), kept);
    EXPECT_EQ(fromMoments.covarianceLower(), g.covarianceLower());
}

// A covariance may be given as a diagonal, its variances, to the constructors and to
// setCovariance(), which store the dense matrix it is.
TEST(GaussianParams, TakesADiagonalCovariance)
{
    auto const variances = Eigen::Vector3d{1.0, 4.0, 9.0};
    Eigen::Matrix3d const dense = variances.asDiagonal();
    auto g = GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), variances.asDiagonal()};
    EXPECT_EQ(g.covarianceLower(), dense);
    auto const inSpace = GaussianParams<Linear<3>>{
        Linear<3>{},
        Eigen::Vector3d::Zero(),
        Eigen::DiagonalMatrix<double, 3>{1.0, 4.0, 9.0}
    };
    EXPECT_EQ(inSpace.covarianceLower(), dense);
    g.setCovariance((2.0 * variances).asDiagonal());
    EXPECT_EQ(g.covarianceLower(), (2.0 * dense).eval());
    EXPECT_TRUE(g.factorize());
}

// A Gaussian built without values, fixed-size or sized, starts at zero rather than with
// whatever the memory held, so that one never filled in fails factorize() the same way every time.
TEST(GaussianParams, ConstructedWithoutValuesIsZero)
{
    auto fixed = GaussianParams<Linear<3>>{};
    EXPECT_TRUE(fixed.mean().isZero());
    EXPECT_TRUE(fixed.covarianceLower().isZero());
    EXPECT_FALSE(fixed.factorize());
    auto const sized = GaussianParams<Linear<Eigen::Dynamic>>{4};
    EXPECT_TRUE(sized.mean().isZero());
    EXPECT_TRUE(sized.covarianceLower().isZero());
    auto const inSpace = GaussianParams<Linear<2>>{Linear<2>{}};
    EXPECT_TRUE(inSpace.covarianceLower().isZero());
    EXPECT_TRUE(Emrys::SqrtGaussianParams<Linear<2>>{}.root().isZero());
    EXPECT_TRUE(Emrys::InfoGaussianParams<Linear<2>>{}.infoMatrix().isZero());
    EXPECT_TRUE(Emrys::SqrtInfoGaussianParams<Linear<2>>{}.infoRoot().isZero());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

class GaussianDeathTest : public ::testing::Test
{
protected:

    using MG = GaussianParams<Linear<2>>;

    static MG
    make()
    {
        return MG{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    }
};

// Reading a density after setCovariance, setMean, or resize changed its parameters dies [view-stamp]; the
// mutation itself does not.
TEST_F(GaussianDeathTest, ReadAfterChange)
{
    auto mg = make();
    auto const g = mg.toDensity();
    EXPECT_DEATH(
        {
            mg.setCovariance(Eigen::Matrix2d::Identity());
            std::ignore = g->logPeak();
        },
        "changed while a view"
    );
    EXPECT_DEATH(
        {
            mg.setMean(Eigen::Vector2d::Zero());
            std::ignore = g->logDensity(Eigen::Vector2d::Zero());
        },
        "changed while a view"
    );
    EXPECT_DEATH(
        {
            mg.resize(2);
            std::ignore = g->mean();
        },
        "changed while a view"
    );
}

// A copy of a density checks the same stamp after the original is gone.
TEST_F(GaussianDeathTest, CopyOfViewChecks)
{
    auto mg = make();
    auto copy = std::optional<MG::DensityType>{};
    {
        auto const g = mg.toDensity();
        copy.emplace(*g);
    }
    mg.setCovariance(Eigen::Matrix2d::Identity());
    EXPECT_DEATH(std::ignore = copy->logPeak(), "changed while a view");
}

// Reading a density of parameters that were moved from dies, as does one of parameters assigned to.
TEST_F(GaussianDeathTest, ReadAfterMove)
{
    auto mg = make();
    auto const g = mg.toDensity();
    EXPECT_DEATH(
        {
            auto stolen = std::move(mg);
            std::ignore = g->logPeak();
        },
        "changed while a view"
    );
    EXPECT_DEATH(
        {
            mg = make();
            std::ignore = g->logPeak();
        },
        "changed while a view"
    );
}

#endif
