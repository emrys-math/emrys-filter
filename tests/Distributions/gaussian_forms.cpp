// The four forms of the Gaussian (roadmap steps 16 and 17; decisions.md [gaussian-forms]): the GaussianForm
// concept and its traits, the accessors every form shares (covariance(), precision(),
// covarianceRoot(), precisionRoot()) with the failures in [gaussian-forms]'s table, the conversions between
// forms, the density, whitening, moments, marginals, and draws of FormGaussian against Gaussian
// [form-densities], the square-root form's marginals, and the check a density keeps. Headers under test:
// Distributions/GaussianForm.h, Gaussian.h, SqrtGaussian.h, InfoGaussian.h, and
// SqrtInfoGaussian.h. The factor operations of the information forms are in
// info_gaussian.cpp.

#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <cmath>
#include <expected>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Distributions/draw_stream.h"

using Emrys::Composite;
using Emrys::GaussianError;
using Emrys::GaussianForm;
using Emrys::GaussianParams;
using Emrys::InfoGaussian;
using Emrys::InfoGaussianParams;
using Emrys::Linear;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::SqrtGaussian;
using Emrys::SqrtGaussianParams;
using Emrys::SqrtInfoGaussianParams;
using Emrys::TagPath;

namespace
{

struct Position : Linear<2>
{};

struct Heading : RadiansPi<>
{};

using Pose = Composite<Position, Heading>;

static_assert(GaussianForm<GaussianParams<Linear<3>>>);
static_assert(GaussianForm<GaussianParams<Pose, float>>);
static_assert(GaussianForm<SqrtGaussianParams<Linear<Eigen::Dynamic>>>);
static_assert(!GaussianForm<Emrys::Moments<Linear<3>>>);
static_assert(!Emrys::isInfoForm<GaussianParams<Linear<3>>> && !Emrys::isSqrtForm<GaussianParams<Linear<3>>>);
static_assert(!Emrys::isInfoForm<SqrtGaussianParams<Linear<3>>>);
static_assert(Emrys::isSqrtForm<SqrtGaussianParams<Linear<3>>>);
static_assert(std::is_nothrow_move_constructible_v<SqrtGaussianParams<Linear<3>>>);
static_assert(GaussianForm<InfoGaussianParams<Pose>> && GaussianForm<SqrtInfoGaussianParams<Pose>>);
static_assert(Emrys::isInfoForm<InfoGaussianParams<Linear<3>>>);
static_assert(!Emrys::isSqrtForm<InfoGaussianParams<Linear<3>>>);
static_assert(Emrys::isInfoForm<SqrtInfoGaussianParams<Linear<3>>>);
static_assert(Emrys::isSqrtForm<SqrtInfoGaussianParams<Linear<3>>>);
static_assert(std::is_nothrow_move_constructible_v<InfoGaussianParams<Linear<3>>>);
static_assert(std::is_nothrow_move_constructible_v<SqrtInfoGaussianParams<Linear<3>>>);

// The covariance form's marginal, of parameters or of a density, is a GaussianForm, as the other
// forms' marginals are parameters that are [form-densities].
static_assert(GaussianForm<decltype(std::declval<GaussianParams<Pose> const &>().marginal<Position>())>);
static_assert(GaussianForm<decltype(std::declval<Emrys::Gaussian<Pose> const &>().marginal<Heading>())>);
static_assert(!Emrys::isInfoForm<decltype(std::declval<GaussianParams<Pose> const &>().marginal<Position>())>);

// The square-root form is a Component and the information forms are not [component-first], [gaussian-forms]: a
// mixture needs a collapse and a density of every element, which a prior with no information lacks.
static_assert(Emrys::Component<SqrtGaussian<Linear<3>>>);

static_assert(!Emrys::Component<InfoGaussian<Linear<3>>>);

Eigen::Matrix3d
makeCov()
{
    return Eigen::Matrix3d{{2.0, 0.3, -0.4}, {0.3, 1.0, 0.2}, {-0.4, 0.2, 0.5}};
}

Eigen::Vector3d const mean3{1.0, -2.0, 0.5};

}  // namespace

// Gaussian's accessors: the covariance whole, the precision its inverse, the covariance factor
// the lower Cholesky factor, and the precision factor its inverse, lower triangular with
// R^T R = Λ. The references are Eigen's inverse() and LLT.
TEST(GaussianForms, GaussianAccessors)
{
    Eigen::Matrix3d const cov = makeCov();
    Eigen::Matrix3d lowerOnly = cov;
    lowerOnly.triangularView<Eigen::StrictlyUpper>().setConstant(99.0);
    auto const g = GaussianParams<Linear<3>>{mean3, lowerOnly};

    auto const covariance = g.covariance();
    ASSERT_TRUE(covariance);
    EXPECT_TRUE(covariance->isApprox(cov, 1e-15));

    auto const precision = g.precision();
    ASSERT_TRUE(precision);
    EXPECT_TRUE(precision->isApprox(cov.inverse(), 1e-12));

    auto const factor = g.covarianceRoot();
    ASSERT_TRUE(factor);
    Eigen::Matrix3d const cholesky = cov.llt().matrixL();
    EXPECT_TRUE(factor->isApprox(cholesky, 1e-15));
    EXPECT_TRUE(factor->triangularView<Eigen::StrictlyUpper>().toDenseMatrix().isZero(0.0));

    auto const precisionRoot = g.precisionRoot();
    ASSERT_TRUE(precisionRoot);
    EXPECT_TRUE(precisionRoot->triangularView<Eigen::StrictlyUpper>().toDenseMatrix().isZero(0.0));
    Eigen::Matrix3d const product = precisionRoot->transpose() * *precisionRoot;
    EXPECT_TRUE(product.isApprox(cov.inverse(), 1e-12));
}

// An indefinite covariance: covariance() still succeeds, as it never fails in the covariance form,
// and the three accessors that need a factorization return NotPositiveDefinite.
TEST(GaussianForms, GaussianAccessorsFailOnAnIndefiniteCovariance)
{
    Eigen::Matrix3d cov = makeCov();
    cov(2, 2) = -1.0;
    auto const g = GaussianParams<Linear<3>>{mean3, cov};
    EXPECT_TRUE(g.covariance());
    EXPECT_EQ(g.precision().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(g.covarianceRoot().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(g.precisionRoot().error(), GaussianError::NotPositiveDefinite);
}

// A GaussianMarginal's accessors are those of the GaussianParams copy() gives, and convert as any
// form does; a block that is not positive definite fails as GaussianParams does, while a block
// that is still converts, though the whole covariance is not.
TEST(GaussianForms, GaussianMarginalAccessors)
{
    auto const g = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()};
    auto const position = g.marginal<Position>();
    auto const copy = position.copy();
    EXPECT_EQ(position.covariance().value(), copy.covariance().value());
    EXPECT_TRUE(position.precision()->isApprox(copy.precision().value(), 1e-14));
    EXPECT_TRUE(position.covarianceRoot()->isApprox(copy.covarianceRoot().value(), 1e-15));
    EXPECT_TRUE(position.precisionRoot()->isApprox(copy.precisionRoot().value(), 1e-14));
    auto const root = Emrys::toSqrt(position);
    auto const info = Emrys::toInfo(position);
    ASSERT_TRUE(root && info);
    EXPECT_TRUE(root->root().isApprox(copy.covarianceRoot().value(), 1e-15));
    EXPECT_EQ(info->reference(), copy.mean());

    Eigen::Matrix3d cov = makeCov();
    cov(2, 2) = -1.0;
    auto const indefinite = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, cov};
    EXPECT_EQ(indefinite.marginal<Heading>().precision().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(indefinite.marginal<Heading>().covarianceRoot().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(indefinite.marginal<Heading>().precisionRoot().error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(Emrys::toInfo(indefinite.marginal<Position>()));

    cov = makeCov();
    cov(2, 2) = std::numeric_limits<double>::quiet_NaN();
    auto const nonFinite = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, cov};
    EXPECT_EQ(nonFinite.marginal<Heading>().precisionRoot().error(), GaussianError::NonFinite);
}

namespace
{

// A Gaussian form of the user's own, a covariance form whose mean and matrices fail as told, for
// the conversions, which pass on the first failure they meet.
struct Faulty
{
    using SpaceType = Linear<3>;
    using Scalar = double;
    using VectorType = Eigen::Vector3d;
    using MatrixType = Eigen::Matrix3d;
    static constexpr bool IsInfoForm = false;
    static constexpr bool IsSqrtForm = false;

    bool meanFails = false;
    bool matricesFail = false;
    SpaceType m_space{};

    [[nodiscard]] SpaceType const &
    space() const
    {
        return m_space;
    }

    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return 3;
    }

    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return 3;
    }

    [[nodiscard]] std::expected<VectorType, GaussianError>
    mean() const
    {
        if (meanFails)
        {
            return std::unexpected{GaussianError::Singular};
        }
        return mean3;
    }

    [[nodiscard]] std::expected<MatrixType, GaussianError>
    matrix() const
    {
        if (matricesFail)
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        return MatrixType::Identity();
    }

    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        return matrix();
    }

    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precision() const
    {
        return matrix();
    }

    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covarianceRoot() const
    {
        return matrix();
    }

    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precisionRoot() const
    {
        return matrix();
    }
};

static_assert(GaussianForm<Faulty>);

}  // namespace

// Each conversion returns the error of the first accessor of its source that fails, whichever it
// reads first.
TEST(GaussianForms, ConversionsPassOnTheFirstFailure)
{
    auto const noMean = Faulty{.meanFails = true};
    auto const noMatrices = Faulty{.matricesFail = true};
    EXPECT_EQ(Emrys::toGaussian(noMatrices).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(Emrys::toInfo(noMean).error(), GaussianError::Singular);
    EXPECT_EQ(Emrys::toInfo(noMatrices).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(Emrys::toSqrt(noMean).error(), GaussianError::Singular);
    EXPECT_EQ(Emrys::toSqrtInfo(noMean).error(), GaussianError::Singular);
    EXPECT_EQ(Emrys::toSqrtInfo(noMatrices).error(), GaussianError::NotPositiveDefinite);

    // A NaN in the square-root information factor fails moments() as it fails the mean.
    Eigen::Matrix3d factor = makeCov().inverse().llt().matrixU();
    factor(0, 2) = std::numeric_limits<double>::quiet_NaN();
    auto const nonFinite = SqrtInfoGaussianParams<Linear<3>>{mean3, factor, Eigen::Vector3d::Zero()};
    EXPECT_EQ(nonFinite.moments().error(), GaussianError::NonFinite);
}

// In dimension zero every value is the mean: its confidence is 0, every confidence region holds
// it, and scaling to a confidence changes nothing, in the covariance form and in the others.
TEST(GaussianForms, DimensionZero)
{
    using None = Linear<Eigen::Dynamic>;
    auto const empty = Eigen::VectorXd{};
    auto g = GaussianParams<None>{None{0}, empty, Eigen::MatrixXd{}};
    g.scaleToConfidence(0.9);
    EXPECT_EQ(g.differenceDimension(), 0);
    auto const density = g.toDensity();
    ASSERT_TRUE(density);
    EXPECT_EQ(density->confidence(empty), 0.0);
    EXPECT_TRUE(density->contains(empty, 0.5));

    auto const root = SqrtGaussianParams<None>{None{0}, empty, Eigen::MatrixXd{}};
    auto const rootDensity = root.toDensity();
    ASSERT_TRUE(rootDensity);
    EXPECT_EQ(rootDensity->confidence(empty), 0.0);
    EXPECT_TRUE(rootDensity->contains(empty, 0.5));
}

// SqrtGaussianParams keeps only the lower triangle of its factor, forms the covariance S S^T and
// the moments, and gives the precision and its lower factor S^{-1}. References: products and
// Eigen's inverse().
TEST(GaussianForms, SqrtAccessors)
{
    Eigen::Matrix3d const cholesky = makeCov().llt().matrixL();
    Eigen::Matrix3d withUpper = cholesky;
    withUpper.triangularView<Eigen::StrictlyUpper>().setConstant(7.0);
    auto const g = SqrtGaussianParams<Linear<3>>{mean3, withUpper};

    EXPECT_EQ(g.root(), cholesky);
    EXPECT_EQ(g.covarianceRoot().value(), cholesky);
    EXPECT_TRUE(g.covariance()->isApprox(makeCov(), 1e-15));
    EXPECT_TRUE(g.moments().covarianceLower().isApprox(makeCov(), 1e-15));
    EXPECT_EQ(g.moments().mean(), mean3);
    EXPECT_TRUE(g.precision()->isApprox(makeCov().inverse(), 1e-12));
    Eigen::Matrix3d const inverse = cholesky.inverse();
    EXPECT_TRUE(g.precisionRoot()->isApprox(inverse, 1e-13));
}

// A factor whose diagonal has negative entries is as good as the Cholesky factor: flipping a
// column's sign leaves S S^T, the density, and the precision unchanged.
TEST(GaussianForms, SqrtFactorSignsDoNotMatter)
{
    Eigen::Matrix3d const cholesky = makeCov().llt().matrixL();
    Eigen::Matrix3d flipped = cholesky;
    flipped.col(1) *= -1.0;
    auto const g = SqrtGaussianParams<Linear<3>>{mean3, cholesky};
    auto const h = SqrtGaussianParams<Linear<3>>{mean3, flipped};

    EXPECT_TRUE(h.covariance()->isApprox(g.covariance().value(), 1e-15));
    EXPECT_TRUE(h.precision()->isApprox(g.precision().value(), 1e-13));
    auto const x = Eigen::Vector3d{0.3, -1.0, 2.0};
    EXPECT_NEAR(h.toDensity()->logDensity(x), g.toDensity()->logDensity(x), 1e-13);
    EXPECT_NEAR(h.toDensity()->density(x), std::exp(g.toDensity()->logDensity(x)), 1e-15);
}

// A zero on the factor's diagonal makes the covariance singular: the covariance and its factor
// are still there, and the precision, its factor, and the density fail with NotPositiveDefinite.
// A NaN fails with NonFinite.
TEST(GaussianForms, SqrtFailures)
{
    Eigen::Matrix3d singular = makeCov().llt().matrixL();
    singular(2, 2) = 0.0;
    auto const g = SqrtGaussianParams<Linear<3>>{mean3, singular};
    EXPECT_TRUE(g.covariance());
    EXPECT_TRUE(g.covarianceRoot());
    EXPECT_EQ(g.precision().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(g.precisionRoot().error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(g.toDensity().error(), GaussianError::NotPositiveDefinite);

    Eigen::Matrix3d nonFinite = makeCov().llt().matrixL();
    nonFinite(1, 0) = std::numeric_limits<double>::quiet_NaN();
    auto const h = SqrtGaussianParams<Linear<3>>{mean3, nonFinite};
    EXPECT_EQ(h.precision().error(), GaussianError::NonFinite);
    EXPECT_EQ(h.toDensity().error(), GaussianError::NonFinite);
}

// The square-root form's density, squared Mahalanobis distance, confidence, and log-determinant equal Gaussian's
// for the same distribution, on a space with a periodic coordinate, where the residual takes the
// short way around.
TEST(GaussianForms, SqrtDensityMatchesGaussian)
{
    auto const mean = Eigen::Vector3d{1.0, 2.0, 4.0};
    auto const g = GaussianParams<Pose>{mean, makeCov()};
    auto const root = Emrys::toSqrt(g);
    ASSERT_TRUE(root);
    EXPECT_NEAR(root->mean()(2), mean(2) - 2.0 * std::numbers::pi, 1e-15);

    auto const expected = g.toDensity();
    auto const actual = root->toDensity();
    ASSERT_TRUE(expected && actual);
    EXPECT_NEAR(actual->logDeterminant(), expected->logDeterminant(), 1e-13);
    for (Eigen::Vector3d const & x :
         {Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.5, 2.5, -3.0}, Eigen::Vector3d{2.0, 1.0, 0.1}})
    {
        EXPECT_NEAR(actual->logDensity(x), expected->logDensity(x), 1e-12);
        EXPECT_NEAR(actual->squaredMahalanobisDistance(x), expected->squaredMahalanobisDistance(x), 1e-12);
        EXPECT_NEAR(actual->confidence(x), expected->confidence(x), 1e-12);
        EXPECT_EQ(actual->contains(x, 0.9), expected->contains(x, 0.9));
    }
}

// A Gaussian converted to the square-root form and back is the same Gaussian, and the factor of
// toSqrt() is the Gaussian's own Cholesky factor.
TEST(GaussianForms, SqrtRoundTrip)
{
    auto const g = GaussianParams<Linear<3>>{mean3, makeCov()};
    auto const root = Emrys::toSqrt(g);
    ASSERT_TRUE(root);
    EXPECT_EQ(root->root(), g.covarianceRoot().value());

    auto const back = Emrys::toGaussian(*root);
    ASSERT_TRUE(back);
    EXPECT_EQ(back->mean(), g.mean());
    EXPECT_TRUE(back->covarianceLower().isApprox(makeCov(), 1e-15));

    Eigen::Matrix3d cov = makeCov();
    cov(0, 0) = -1.0;
    auto const indefinite = GaussianParams<Linear<3>>{mean3, cov};
    EXPECT_EQ(Emrys::toSqrt(indefinite).error(), GaussianError::NotPositiveDefinite);
}

// Draws from the square-root form converted from a Gaussian equal Gaussian's draws with
// the same seed, in a batch and one drawSample() at a time, since both color the same standard
// normals by the same Cholesky factor; and drawSampleWithLogDensity() draws what drawSample() draws,
// with the form's log-density at each, exactly off a linear space and to rounding on one.
TEST(GaussianForms, SqrtDrawsMatchGaussian)
{
    auto const g = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()};
    auto const root = Emrys::toSqrt(g);
    auto first = std::mt19937{7};
    auto second = std::mt19937{7};
    auto const expected = Emrys::drawSamples(g.toDensity().value(), 50, first);
    auto const actual = Emrys::drawSamples(root->toDensity().value(), 50, second);
    EXPECT_TRUE(actual.isApprox(expected, 1e-14));
    for (int k = 0; k < 10; ++k)
    {
        Eigen::Vector3d const one = Emrys::drawSample(g.toDensity().value(), first);
        EXPECT_TRUE(Emrys::drawSample(root->toDensity().value(), second).isApprox(one, 1e-14));
    }
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(root->toDensity().value(), 20, 8, 0.0);
    auto const linear = Emrys::toSqrt(GaussianParams<Linear<3>>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()});
    ASSERT_TRUE(linear.has_value());
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(linear->toDensity().value(), 20, 9, 1e-12);
}

// The square-root form's marginal by tag path is parameters whose factor is the lower Cholesky
// factor of the covariance form's marginal block, for a leading block, a trailing one, and one
// repetition of a repeated block.
TEST(GaussianForms, SqrtMarginalMatchesGaussian)
{
    auto const g = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()};
    auto const root = Emrys::toSqrt(g);
    ASSERT_TRUE(root);

    auto const position = root->marginal<Position>();
    static_assert(std::is_same_v<decltype(position), SqrtGaussianParams<Position> const>);
    auto const expectedPosition = g.marginal<Position>().copy();
    EXPECT_EQ(position.mean(), expectedPosition.mean());
    EXPECT_TRUE(position.root().isApprox(Eigen::Matrix2d{expectedPosition.covarianceLower().llt().matrixL()}, 1e-14));
    EXPECT_TRUE(position.covariance()->isApprox(expectedPosition.covarianceLower(), 1e-14));

    auto const heading = root->marginal<Heading>();
    EXPECT_NEAR(heading.mean()(0), g.mean()(2), 1e-15);
    EXPECT_NEAR(heading.root()(0, 0), std::sqrt(makeCov()(2, 2)), 1e-14);

    struct Fleet : Repeated<Pose, 2>
    {};

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.block<3, 3>(3, 3) = makeCov();
    cov.block<3, 3>(0, 3) = 0.1 * Eigen::Matrix3d::Ones();
    cov.block<3, 3>(3, 0) = 0.1 * Eigen::Matrix3d::Ones();
    auto const fleet = GaussianParams<Fleet>{Fleet{}, Eigen::Matrix<double, 6, 1>::LinSpaced(6, 0.0, 5.0), cov};
    auto const second = Emrys::toSqrt(fleet)->marginal<Fleet>(1);
    static_assert(std::is_same_v<decltype(second), SqrtGaussianParams<Pose> const>);
    EXPECT_EQ(second.mean(), fleet.marginal<Fleet>(1).copy().mean());
    EXPECT_TRUE(second.covariance()->isApprox(makeCov(), 1e-13));
    auto const secondHeading = Emrys::toSqrt(fleet)->marginal<TagPath<Fleet, Heading>>(1);
    EXPECT_NEAR(secondHeading.mean()(0), fleet.mean()(5), 1e-15);
}

// The information form's accessors: the covariance, mean, and moments by solves, the precision as
// stored, and upper-triangular factors, R the Cholesky factor of Λ and R^{-1} a factor of Σ. A
// reference away from the mean carries the offset in η = Λ δ. References: Eigen's inverse() and LLT.
TEST(GaussianForms, InfoAccessors)
{
    Eigen::Matrix3d const precision = makeCov().inverse();
    auto const reference = Eigen::Vector3d{0.0, 1.0, -1.0};
    Eigen::Vector3d const eta = precision * (mean3 - reference);
    auto const g = InfoGaussianParams<Linear<3>>{reference, precision, eta};

    EXPECT_TRUE(g.mean()->isApprox(mean3, 1e-13));
    EXPECT_TRUE(g.moments()->mean().isApprox(mean3, 1e-13));
    EXPECT_TRUE(g.moments()->covarianceLower().isApprox(makeCov(), 1e-13));
    EXPECT_TRUE(g.covariance()->isApprox(makeCov(), 1e-13));
    EXPECT_TRUE(g.precision()->isApprox(precision, 1e-15));
    Eigen::Matrix3d const upper = precision.llt().matrixU();
    EXPECT_TRUE(g.precisionRoot()->isApprox(upper, 1e-14));
    auto const factor = g.covarianceRoot();
    ASSERT_TRUE(factor);
    EXPECT_TRUE(factor->triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
    Eigen::Matrix3d const product = *factor * factor->transpose();
    EXPECT_TRUE(product.isApprox(makeCov(), 1e-13));
}

// The square-root information form's accessors: R stored and kept upper, the precision R^T R,
// the information vector R^T r, and the mean, moments, covariance, and R^{-1} by triangular solves.
TEST(GaussianForms, SqrtInfoAccessors)
{
    Eigen::Matrix3d const precision = makeCov().inverse();
    Eigen::Matrix3d const upper = precision.llt().matrixU();
    Eigen::Matrix3d withLower = upper;
    withLower.triangularView<Eigen::StrictlyLower>().setConstant(5.0);
    auto const reference = Eigen::Vector3d{0.0, 1.0, -1.0};
    Eigen::Vector3d const whitened = upper * (mean3 - reference);
    auto const g = SqrtInfoGaussianParams<Linear<3>>{reference, withLower, whitened};

    EXPECT_EQ(g.infoRoot(), upper);
    EXPECT_EQ(g.precisionRoot().value(), upper);
    EXPECT_TRUE(g.precision()->isApprox(precision, 1e-14));
    EXPECT_TRUE(g.infoVector().isApprox(precision * (mean3 - reference), 1e-13));
    EXPECT_TRUE(g.mean()->isApprox(mean3, 1e-13));
    EXPECT_TRUE(g.moments()->mean().isApprox(mean3, 1e-13));
    EXPECT_TRUE(g.moments()->covarianceLower().isApprox(makeCov(), 1e-13));
    EXPECT_TRUE(g.covariance()->isApprox(makeCov(), 1e-13));
    Eigen::Matrix3d const inverse = upper.inverse();
    EXPECT_TRUE(g.covarianceRoot()->isApprox(inverse, 1e-13));
}

// A prior with no information, Λ = 0 and R = 0, is valid parameters ([gaussian-forms]'s first row): the
// precision and, in the square-root form, its factor are there, and everything that needs the mean
// or the covariance returns Singular. A NaN in the precision is NonFinite.
TEST(GaussianForms, InfoFormsWithNoInfo)
{
    Eigen::Vector3d const reference = Eigen::Vector3d::Zero();
    auto const g = InfoGaussianParams<Linear<3>>{reference, Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero()};
    EXPECT_TRUE(g.precision()->isZero(0.0));
    EXPECT_EQ(g.mean().error(), GaussianError::Singular);
    EXPECT_EQ(g.moments().error(), GaussianError::Singular);
    EXPECT_EQ(g.covariance().error(), GaussianError::Singular);
    EXPECT_EQ(g.covarianceRoot().error(), GaussianError::Singular);
    EXPECT_EQ(g.precisionRoot().error(), GaussianError::Singular);
    EXPECT_EQ(g.toDensity().error(), GaussianError::Singular);
    EXPECT_EQ(Emrys::toGaussian(g).error(), GaussianError::Singular);

    auto const h = SqrtInfoGaussianParams<Linear<3>>{reference, Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero()};
    EXPECT_TRUE(h.precision()->isZero(0.0));
    EXPECT_TRUE(h.precisionRoot()->isZero(0.0));
    EXPECT_EQ(h.mean().error(), GaussianError::Singular);
    EXPECT_EQ(h.covariance().error(), GaussianError::Singular);
    EXPECT_EQ(h.covarianceRoot().error(), GaussianError::Singular);
    EXPECT_EQ(h.toDensity().error(), GaussianError::Singular);

    Eigen::Matrix3d nonFinite = makeCov().inverse();
    nonFinite(2, 1) = std::numeric_limits<double>::quiet_NaN();
    auto const bad = InfoGaussianParams<Linear<3>>{reference, nonFinite, Eigen::Vector3d::Zero()};
    EXPECT_EQ(bad.mean().error(), GaussianError::NonFinite);
}

// Every conversion from a Gaussian and back gives the same Gaussian, on a space with a periodic
// coordinate. The factors of the conversions are the unique ones with a positive diagonal: the
// lower Cholesky factor of Σ for the square-root form and the upper Cholesky factor of Λ for the
// square-root information form, whatever form they come from.
TEST(GaussianForms, ConversionsRoundTrip)
{
    auto const g = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()};
    Eigen::Matrix3d const lower = makeCov().llt().matrixL();
    Eigen::Matrix3d const upper = makeCov().inverse().llt().matrixU();

    auto const check = [&](auto const & form) {
        auto const back = Emrys::toGaussian(form);
        ASSERT_TRUE(back);
        EXPECT_TRUE(back->mean().isApprox(g.mean(), 1e-13));
        EXPECT_TRUE(back->covarianceLower().isApprox(g.covarianceLower(), 1e-13));
        EXPECT_TRUE(Emrys::toSqrt(form)->root().isApprox(lower, 1e-13));
        EXPECT_TRUE(Emrys::toSqrtInfo(form)->infoRoot().isApprox(upper, 1e-12));
        EXPECT_TRUE(Emrys::toInfo(form)->precision()->isApprox(makeCov().inverse(), 1e-12));
    };
    check(g);
    auto const root = Emrys::toSqrt(g);
    auto const info = Emrys::toInfo(g);
    auto const rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(root && info && rootInfo);
    check(*root);
    check(*info);
    check(*rootInfo);
    check(Emrys::toInfo(*rootInfo).value());
    check(Emrys::toSqrtInfo(*info).value());
}

// The information forms' densities equal Gaussian's, with a reference away from the mean,
// on a space with a periodic coordinate.
TEST(GaussianForms, InfoDensitiesMatchGaussian)
{
    auto const g = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()};
    auto info = Emrys::toInfo(g);
    auto rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(info && rootInfo);
    info->setReference(Eigen::Vector3d{0.5, 2.5, -3.0});
    rootInfo->setReference(Eigen::Vector3d{0.5, 2.5, -3.0});

    auto const expected = g.toDensity();
    auto const first = info->toDensity();
    auto const second = rootInfo->toDensity();
    ASSERT_TRUE(expected && first && second);
    EXPECT_TRUE(first->mean().isApprox(expected->mean(), 1e-13));
    EXPECT_TRUE(second->mean().isApprox(expected->mean(), 1e-13));
    EXPECT_NEAR(first->logDeterminant(), expected->logDeterminant(), 1e-12);
    EXPECT_NEAR(second->logDeterminant(), expected->logDeterminant(), 1e-12);
    for (Eigen::Vector3d const & x :
         {Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.5, 2.5, -3.0}, Eigen::Vector3d{2.0, 1.0, 0.1}})
    {
        EXPECT_NEAR(first->logDensity(x), expected->logDensity(x), 1e-12);
        EXPECT_NEAR(second->logDensity(x), expected->logDensity(x), 1e-12);
        EXPECT_NEAR(first->squaredMahalanobisDistance(x), expected->squaredMahalanobisDistance(x), 1e-12);
        EXPECT_NEAR(second->squaredMahalanobisDistance(x), expected->squaredMahalanobisDistance(x), 1e-12);
    }
}

// Draws from the two information forms are equal, since both color by the inverse of the same
// upper Cholesky factor of Λ, and their sample mean and covariance match the Gaussian's. The
// tolerance is about four standard errors of 20000 draws.
TEST(GaussianForms, InfoDraws)
{
    auto const g = GaussianParams<Linear<3>>{mean3, makeCov()};
    auto const info = Emrys::toInfo(g);
    auto const rootInfo = Emrys::toSqrtInfo(g);
    auto first = std::mt19937{11};
    auto second = std::mt19937{11};
    auto const draws = Emrys::drawSamples(info->toDensity().value(), 20000, first);
    auto const same = Emrys::drawSamples(rootInfo->toDensity().value(), 20000, second);
    EXPECT_TRUE(same.isApprox(draws, 1e-13));

    Eigen::Vector3d const sampleMean = draws.rowwise().mean();
    Eigen::Matrix<double, 3, Eigen::Dynamic> const centered = draws.colwise() - sampleMean;
    Eigen::Matrix3d const sampleCov = centered * centered.transpose() / double(draws.cols() - 1);
    EXPECT_TRUE((sampleMean - mean3).cwiseAbs().maxCoeff() < 0.05);
    EXPECT_TRUE((sampleCov - makeCov()).cwiseAbs().maxCoeff() < 0.08);
}

// The three forms' densities have the moments, marginals, and draws with log-densities that a
// Gaussian has, and they match the Gaussian's to rounding [form-densities], so code generic over the forms
// reads all four alike: every density's marginal is a GaussianForm that toGaussian() takes. The
// information forms are moved to a reference away from the mean, so that their moments solve and
// carry the covariance; the heading's mean wraps from that reference.
TEST(GaussianForms, FormDensitiesReadAsTheGaussian)
{
    auto const g = GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()};
    auto const root = Emrys::toSqrt(g);
    auto info = Emrys::toInfo(g);
    auto rootInfo = Emrys::toSqrtInfo(g);
    ASSERT_TRUE(root && info && rootInfo);
    info->setReference(Eigen::Vector3d{0.5, 2.5, -3.0});
    rootInfo->setReference(Eigen::Vector3d{0.5, 2.5, -3.0});
    auto const expected = g.toDensity();
    ASSERT_TRUE(expected);
    auto const position = g.marginal<Position>().copy();

    auto const check = [&](auto const & density) {
        auto const moments = density.moments();
        static_assert(std::is_same_v<std::remove_const_t<decltype(moments)>, Emrys::Moments<Pose>>);
        EXPECT_TRUE(moments.mean().isApprox(expected->mean(), 1e-13));
        EXPECT_TRUE(moments.covarianceLower().isApprox(expected->covarianceLower(), 1e-13));

        auto const marginal = Emrys::toGaussian(density.template marginal<Position>());
        ASSERT_TRUE(marginal);
        EXPECT_TRUE(marginal->mean().isApprox(position.mean(), 1e-13));
        EXPECT_TRUE(marginal->covarianceLower().isApprox(position.covarianceLower(), 1e-13));
        auto const heading = Emrys::toGaussian(density.template marginal<Heading>());
        ASSERT_TRUE(heading);
        EXPECT_NEAR(heading->mean()(0), 3.0, 1e-13);
        EXPECT_NEAR(heading->covarianceLower()(0, 0), makeCov()(2, 2), 1e-13);

        auto first = std::mt19937{5};
        auto second = std::mt19937{5};
        auto const drawn = Emrys::drawSamplesWithLogDensity(density, 50, first);
        EXPECT_EQ(drawn.samples, Emrys::drawSamples(density, 50, second));
        for (Eigen::Index j = 0; j < drawn.samples.cols(); ++j)
        {
            EXPECT_NEAR(drawn.logDensity(j), expected->logDensity(drawn.samples.col(j)), 1e-12);
        }
    };
    check(expected.value());
    check(root->toDensity().value());
    check(info->toDensity().value());
    check(rootInfo->toDensity().value());
}

// The information form's density marginalizes from its cached factor by triangularization, which
// gives the parameters' Schur complement, and a repetition's marginal of a repeated block matches
// the covariance form's. In a linear space a draw's log-density is read from its normals, which
// matches logDensity() of the draw.
TEST(GaussianForms, InfoDensityMarginalIsTheSchurComplement)
{
    auto info = Emrys::toInfo(GaussianParams<Pose>{Eigen::Vector3d{1.0, 2.0, 3.0}, makeCov()});
    ASSERT_TRUE(info);
    info->setReference(Eigen::Vector3d{0.5, 2.5, -3.0});
    auto const schur = info->marginal<Position>();
    ASSERT_TRUE(schur);
    auto const density = info->toDensity();
    ASSERT_TRUE(density);
    auto const factored = density->marginal<Position>();
    static_assert(std::is_same_v<decltype(factored), InfoGaussianParams<Position> const>);
    EXPECT_EQ(factored.reference(), schur->reference());
    EXPECT_TRUE(factored.infoMatrix().isApprox(schur->infoMatrix(), 1e-13));
    EXPECT_TRUE(factored.infoVector().isApprox(schur->infoVector(), 1e-13));

    struct Fleet : Repeated<Pose, 2>
    {};

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.block<3, 3>(3, 3) = makeCov();
    cov.block<3, 3>(0, 3) = 0.1 * Eigen::Matrix3d::Ones();
    cov.block<3, 3>(3, 0) = 0.1 * Eigen::Matrix3d::Ones();
    auto const fleet = GaussianParams<Fleet>{Fleet{}, Eigen::Matrix<double, 6, 1>::LinSpaced(6, 0.0, 5.0), cov};
    auto const fleetInfo = Emrys::toInfo(fleet);
    ASSERT_TRUE(fleetInfo);
    auto const fleetDensity = fleetInfo->toDensity();
    ASSERT_TRUE(fleetDensity);
    auto const second = Emrys::toGaussian(fleetDensity->marginal<Fleet>(1));
    ASSERT_TRUE(second);
    EXPECT_TRUE(second->mean().isApprox(fleet.marginal<Fleet>(1).copy().mean(), 1e-13));
    EXPECT_TRUE(second->covarianceLower().isApprox(makeCov(), 1e-13));

    auto const linear = Emrys::toInfo(GaussianParams<Linear<3>>{mean3, makeCov()});
    ASSERT_TRUE(linear);
    auto const linearDensity = linear->toDensity();
    ASSERT_TRUE(linearDensity);
    auto rng = std::mt19937{7};
    auto const drawn = Emrys::drawSamplesWithLogDensity(linearDensity.value(), 20, rng);
    for (Eigen::Index j = 0; j < drawn.samples.cols(); ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), linearDensity->logDensity(drawn.samples.col(j)), 1e-12);
    }
}

// The forms work in a dynamic space with a bound and in float, and a copy of an information form
// carries its factorization.
TEST(GaussianForms, DynamicAndFloatSpaces)
{
    using Bounded = Linear<Eigen::Dynamic, 6>;
    auto const g = GaussianParams<Bounded>{Bounded{3}, mean3, makeCov()};
    auto const root = Emrys::toSqrt(g);
    auto const info = Emrys::toInfo(g);
    ASSERT_TRUE(root && info);
    EXPECT_EQ(root->differenceDimension(), 3);
    EXPECT_TRUE(root->covariance()->isApprox(makeCov(), 1e-14));
    ASSERT_TRUE(info->factorize());
    InfoGaussianParams<Bounded> const copy = *info;
    EXPECT_TRUE(copy.mean()->isApprox(mean3, 1e-13));

    auto const f = GaussianParams<Linear<3>, float>{mean3.cast<float>(), makeCov().cast<float>()};
    auto const rootF = Emrys::toSqrtInfo(f);
    ASSERT_TRUE(rootF);
    EXPECT_TRUE(rootF->mean()->isApprox(mean3.cast<float>(), 1e-5f));
    EXPECT_TRUE(rootF->covariance()->isApprox(makeCov().cast<float>(), 1e-4f));
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// A FormGaussian checks its parameters: reading it after any change of them dies [view-stamp].
TEST(GaussianFormsDeathTest, DensityChecksItsParams)
{
    Eigen::Vector3d const x = Eigen::Vector3d::Zero();
    std::string const message = "changed while a view";

    auto g = SqrtGaussianParams<Linear<3>>{mean3, Eigen::Matrix3d::Identity()};
    auto const density = g.toDensity();
    ASSERT_TRUE(density);
    EXPECT_DEATH(
        {
            g.setMean(x);
            std::ignore = density->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            g.setRoot(Eigen::Matrix3d::Identity());
            std::ignore = density->logDensity(x);
        },
        message
    );

    auto h = InfoGaussianParams<Linear<3>>{mean3, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
    auto const hDensity = h.toDensity();
    ASSERT_TRUE(hDensity);
    EXPECT_DEATH(
        {
            h.setReference(x);
            std::ignore = hDensity->mean();
        },
        message
    );
    EXPECT_DEATH(
        {
            h.addInfo(Eigen::Matrix3d::Identity(), x);
            std::ignore = hDensity->logDensity(x);
        },
        message
    );

    auto k = SqrtInfoGaussianParams<Linear<3>>{mean3, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
    auto const kDensity = k.toDensity();
    ASSERT_TRUE(kDensity);
    EXPECT_DEATH(
        {
            k.setRoot(Eigen::Matrix3d::Identity(), x);
            std::ignore = kDensity->logDensity(x);
        },
        message
    );
}

#endif
