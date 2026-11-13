// The roots of a covariance (roadmap 17t, [sigma-point-axes]): each of the library's axes gives a root that
// multiplies back to its covariance, and the same root from a factor as from the covariance; Cholesky is the lower
// factor and Symmetric the symmetric one; Principal and Symmetric handle a semidefinite covariance where Cholesky
// fails; Along puts its first column along the direction, one standard deviation long; a fallback list takes the first
// entry that succeeds, a ridge with a positive floor repairs a zero block, and a list that fails reports each entry
// tried; from a factor a list falls back as from a covariance; the axes and repair of the user's own in
// tests/user_axes.h run as the library's do; and the roots of a bounded covariance have its size.

#include <EmrysFilter/Distributions/Roots.h>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>

#include <cmath>
#include <limits>
#include <type_traits>
#include <variant>

#include "user_axes.h"

namespace roots = Emrys::roots;

namespace
{

// A positive definite covariance with every coordinate correlated.
[[nodiscard]] Eigen::Matrix4d
positiveDefinite()
{
    Eigen::Matrix4d a;
    a << 2.0, 0.3, -0.4, 0.1, 0.5, 1.5, 0.2, -0.3, -0.1, 0.4, 1.2, 0.6, 0.2, -0.2, 0.3, 0.9;
    return a * a.transpose();
}

// A covariance of rank two in three coordinates, v v^T + w w^T, whose last Cholesky pivot is exactly zero.
[[nodiscard]] Eigen::Matrix3d
semidefinite()
{
    Eigen::Vector3d const v{1.0, 1.0, 0.0};
    Eigen::Vector3d const w{0.0, 1.0, 1.0};
    return v * v.transpose() + w * w.transpose();
}

template <typename Root, typename Cov>
void
expectRootOf(Root const & root, Cov const & cov, double tolerance = 1e-12)
{
    ASSERT_EQ(root.rows(), cov.rows());
    ASSERT_EQ(root.cols(), cov.cols());
    EXPECT_LE((root * root.transpose() - cov).cwiseAbs().maxCoeff(), tolerance * cov.cwiseAbs().maxCoeff());
}

// A factor of the covariance that is neither triangular nor square, as a square-root form's stack is.
template <typename Cov>
[[nodiscard]] Eigen::MatrixXd
wideFactorOf(Cov const & cov)
{
    Eigen::MatrixXd const lower = Eigen::LLT<Eigen::MatrixXd>{cov}.matrixL();
    Eigen::Index const n = cov.rows();
    // [L Q1, 0] for a rotation Q1 and a zero column: the same F F^T.
    Eigen::MatrixXd const rotation =
        Eigen::HouseholderQR<Eigen::MatrixXd>{Eigen::MatrixXd::Random(n, n)}.householderQ();
    Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(n, n + 1);
    factor.leftCols(n) = lower * rotation;
    return factor;
}

TEST(Roots, EachAxesMultipliesBackToItsCovariance)
{
    auto const cov = positiveDefinite();
    expectRootOf(roots::squareRoot(roots::cholesky(), cov).value(), cov);
    expectRootOf(roots::squareRoot(roots::cholesky(roots::pivoted()), cov), cov);
    expectRootOf(roots::squareRoot(roots::pivoted(), cov), cov);
    expectRootOf(roots::squareRoot(roots::principal(), cov), cov);
    expectRootOf(roots::squareRoot(roots::symmetric(), cov), cov);
    expectRootOf(roots::squareRoot(roots::along(Eigen::Vector4d{1.0, -2.0, 0.5, 3.0}), cov).value(), cov);
}

TEST(Roots, CholeskyIsTheLowerFactorAndSymmetricIsSymmetric)
{
    auto const cov = positiveDefinite();
    Eigen::Matrix4d const lower = Eigen::LLT<Eigen::Matrix4d>{cov}.matrixL();
    auto const cholesky = roots::squareRoot(roots::cholesky(), cov).value();
    EXPECT_EQ(cholesky, lower);
    auto const symmetric = roots::squareRoot(roots::symmetric(), cov);
    EXPECT_LE((symmetric - symmetric.transpose()).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>{symmetric}.eigenvalues().minCoeff(), 0.0);
    // Principal's columns are orthogonal, largest first.
    auto const principal = roots::squareRoot(roots::principal(), cov);
    Eigen::Matrix4d const gram = principal.transpose() * principal;
    EXPECT_LE((gram - Eigen::Matrix4d{gram.diagonal().asDiagonal()}).cwiseAbs().maxCoeff(), 1e-12);
    for (Eigen::Index j = 1; j < 4; ++j)
    {
        EXPECT_GE(gram(j - 1, j - 1), gram(j, j));
    }
}

TEST(Roots, AFactorGivesTheSameRootAsItsCovariance)
{
    auto const cov = positiveDefinite();
    Eigen::MatrixXd const covariance = cov;
    Eigen::MatrixXd const factor = wideFactorOf(cov);
    auto const same = [&](auto const & axes) {
        Eigen::MatrixXd const fromCov = Emrys::internal::asExpected<Eigen::MatrixXd>(axes.root(covariance)).value();
        Eigen::MatrixXd const fromFactor =
            Emrys::internal::asExpected<Eigen::MatrixXd>(Emrys::internal::rootFromFactor(axes, factor)).value();
        EXPECT_LE((fromCov - fromFactor).cwiseAbs().maxCoeff(), 1e-12) << fromCov << "\n\n" << fromFactor;
    };
    same(roots::cholesky());
    same(roots::cholesky(roots::pivoted()));
    same(roots::principal());
    same(roots::symmetric());
    same(roots::along(Eigen::Vector4d{1.0, -2.0, 0.5, 3.0}));
}

TEST(Roots, PrincipalAndSymmetricHandleASemidefiniteCovariance)
{
    auto const cov = semidefinite();
    EXPECT_EQ(roots::squareRoot(roots::cholesky(), cov).error(), Emrys::GaussianError::NotPositiveDefinite);
    expectRootOf(roots::squareRoot(roots::cholesky(roots::pivoted()), cov), cov);
    auto const principal = roots::squareRoot(roots::principal(), cov);
    expectRootOf(principal, cov);
    EXPECT_LE(principal.col(2).norm(), 1e-7);
    expectRootOf(roots::squareRoot(roots::symmetric(), cov), cov);
    // From a factor, Cholesky triangularizes, which a semidefinite covariance's factor survives.
    Eigen::Matrix<double, 3, 2> factor;
    factor << 1.0, 0.0, 1.0, 1.0, 0.0, 1.0;
    expectRootOf(Emrys::internal::rootFromFactor(roots::cholesky(), factor).value(), cov);
    expectRootOf(Emrys::internal::rootFromFactor(roots::principal(), factor), cov);
    expectRootOf(Emrys::internal::rootFromFactor(roots::symmetric(), factor), cov);
}

TEST(Roots, AlongPutsItsFirstColumnAlongTheDirection)
{
    auto const cov = positiveDefinite();
    for (Eigen::Vector4d const & direction :
         {Eigen::Vector4d{1.0, -2.0, 0.5, 3.0},
          Eigen::Vector4d{-1.0, 0.0, 0.0, 0.0},
          Eigen::Vector4d{0.0, 0.0, 0.0, 1.0}})
    {
        auto const root = roots::squareRoot(roots::along(direction), cov).value();
        expectRootOf(root, cov);
        double const length = std::sqrt(direction.dot(cov.llt().solve(direction)));
        EXPECT_LE((root.col(0) - direction / length).cwiseAbs().maxCoeff(), 1e-12) << direction.transpose();
    }
    // The direction of the Cholesky factor's own first column gives that factor back.
    Eigen::Matrix4d const lower = cov.llt().matrixL();
    auto const first = roots::squareRoot(roots::along(lower.col(0)), cov).value();
    EXPECT_LE((first - lower).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(Roots, AlongFailsOutsideTheRangeAndFallsBack)
{
    auto const cov = semidefinite();
    Eigen::Vector3d const direction{1.0, 0.0, 0.0};
    EXPECT_FALSE(roots::squareRoot(roots::along(direction), cov).has_value());
    EXPECT_FALSE(
        roots::squareRoot(roots::along(Eigen::Vector3d::Zero().eval()), Eigen::Matrix3d::Identity().eval()).has_value()
    );
    auto const root = roots::squareRoot(roots::along(direction, roots::pivoted()), cov);
    EXPECT_EQ(root, roots::squareRoot(roots::pivoted(), cov));
}

TEST(Roots, AListTakesTheFirstEntryThatSucceeds)
{
    // Indefinite by rounding: Cholesky fails, the ridge repairs it, and Pivoted is not reached.
    Eigen::Matrix2d cov{{1.0, 1.0}, {1.0, 1.0 - 1e-15}};
    auto const report = roots::squareRootWithReport(roots::cholesky(roots::ridge(), roots::pivoted()), cov);
    ASSERT_EQ(report.report.index(), 1U);
    double const ridge = std::get<1>(report.report).ridge;
    EXPECT_GT(ridge, 0.0);
    expectRootOf(report.root, (cov + ridge * Eigen::Matrix2d::Identity()).eval());
    // Without the ridge, Pivoted takes it, clamping the negative pivot.
    auto const pivoted = roots::squareRootWithReport(roots::cholesky(roots::pivoted()), cov);
    ASSERT_EQ(pivoted.report.index(), 1U);
    EXPECT_EQ(std::get<1>(pivoted.report).clamped, 1);
    // A positive definite one never leaves Cholesky.
    auto const first =
        roots::squareRootWithReport(roots::cholesky(roots::ridge(), roots::pivoted()), positiveDefinite());
    EXPECT_EQ(first.report.index(), 0U);
}

TEST(Roots, ARidgeWithAPositiveFloorRepairsAZeroBlock)
{
    Eigen::Matrix2d const zero = Eigen::Matrix2d::Zero();
    EXPECT_EQ(
        roots::squareRoot(roots::cholesky(roots::ridge()), zero).error(),
        Emrys::GaussianError::NotPositiveDefinite
    );
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge<1.0, 1e-12>()), zero).value();
    expectRootOf(root, (1e-12 * Eigen::Matrix2d::Identity()).eval());
}

TEST(Roots, AListThatFailsReportsEachEntryTried)
{
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
    cov(1, 0) = std::numeric_limits<double>::quiet_NaN();
    auto const report =
        roots::squareRootWithReport(roots::cholesky(roots::ridge(), roots::along(Eigen::Vector2d{1.0, 0.0})), cov);
    ASSERT_FALSE(report.has_value());
    EXPECT_EQ(report.error().error, Emrys::GaussianError::NonFinite);
    auto const & [cholesky, ridge, along] = report.error().report;
    EXPECT_TRUE(cholesky.has_value());
    EXPECT_TRUE(ridge.has_value());
    EXPECT_TRUE(along.has_value());
    // A ridge that adds nothing, to a zero matrix, says so.
    auto const zero = roots::squareRootWithReport(roots::cholesky(roots::ridge()), Eigen::Matrix2d::Zero().eval());
    ASSERT_FALSE(zero.has_value());
    EXPECT_EQ(std::get<1>(zero.error().report)->ridge, 0.0);
}

// From a factor, a list falls back as it does from a covariance: a factor with NaN fails Cholesky with
// NonFinite, a ridge then fails on the covariance formed from it, and Pivoted takes it; Along fails when its
// direction leaves the factor's range and its fallback takes it. A square factor that is not lower
// triangular with a nonnegative diagonal is triangularized, and one that is passes through. The ridge
// repairs without a sink and over no coordinates, and a list that can fail reports its success.
TEST(Roots, AFactorFallsBackAsACovarianceDoes)
{
    Eigen::Matrix2d nan = Eigen::Matrix2d::Identity();
    nan(1, 0) = std::numeric_limits<double>::quiet_NaN();
    auto const pivotedOfNan = Emrys::internal::rootFromFactor(roots::cholesky(roots::pivoted()), nan);
    EXPECT_TRUE(pivotedOfNan.hasNaN());
    auto const repairedOfNan = Emrys::internal::rootFromFactor(roots::cholesky(roots::ridge(), roots::pivoted()), nan);
    EXPECT_TRUE(repairedOfNan.hasNaN());

    Eigen::Matrix3d singular = Eigen::Matrix3d::Zero();
    singular.leftCols<2>() = Eigen::LLT<Eigen::Matrix3d>{semidefinite() + Eigen::Matrix3d::Identity()}
                                 .matrixL()
                                 .toDenseMatrix()
                                 .leftCols<2>();
    Eigen::Vector3d const outside{0.0, 0.0, 1.0};
    EXPECT_FALSE(Emrys::internal::rootFromFactor(roots::along(outside), singular).has_value());
    auto const fallback = Emrys::internal::rootFromFactor(roots::along(outside, roots::pivoted()), singular);
    expectRootOf(fallback, (singular * singular.transpose()).eval());
    // A ridge with a floor repairs the singular covariance for Along, so Pivoted after it is not reached;
    // an Along that fails again passes the error on, to Pivoted or, last in the list, out of it.
    auto const ridged =
        Emrys::internal::rootFromFactor(roots::along(outside, roots::ridge<1.0, 1e-6>(), roots::pivoted()), singular);
    EXPECT_GT(ridged.col(0).norm(), 0.0);
    EXPECT_NEAR(ridged.col(0).normalized().dot(outside), 1.0, 1e-12);
    auto const twice =
        Emrys::internal::rootFromFactor(roots::along(outside, roots::along(outside), roots::pivoted()), singular);
    expectRootOf(twice, (singular * singular.transpose()).eval());
    EXPECT_FALSE(Emrys::internal::rootFromFactor(roots::along(outside, roots::along(outside)), singular).has_value());
    EXPECT_EQ(
        Emrys::internal::rootFromFactor(roots::along(Eigen::Vector2d{1.0, 0.0}), nan).error(),
        Emrys::GaussianError::NonFinite
    );

    Eigen::Matrix3d const full = Eigen::Matrix3d{{1.0, 0.5, 0.0}, {0.0, 1.0, 0.25}, {0.0, 0.0, 1.0}};
    auto const fromFull = Emrys::internal::rootFromFactor(roots::cholesky(), full).value();
    expectRootOf(fromFull, (full * full.transpose()).eval());
    EXPECT_TRUE(fromFull.isLowerTriangular());
    Eigen::Matrix3d negative = Eigen::Matrix3d::Identity();
    negative(1, 1) = -2.0;
    auto const fromNegative = Emrys::internal::rootFromFactor(roots::cholesky(), negative).value();
    expectRootOf(fromNegative, (negative * negative.transpose()).eval());
    EXPECT_GE(fromNegative.diagonal().minCoeff(), 0.0);

    auto const cov = positiveDefinite();
    auto const repaired = roots::ridge().repaired(cov).value();
    EXPECT_GT((repaired - cov).diagonal().minCoeff(), 0.0);
    EXPECT_EQ(roots::ridge().repaired(Eigen::MatrixXd(0, 0)).value().size(), 0);
    auto const reported = roots::squareRootWithReport(roots::cholesky(), cov);
    ASSERT_TRUE(reported.has_value());
    expectRootOf(reported->root, cov);
}

TEST(Roots, TheUsersAxesAndRepairRunAsTheLibrarys)
{
    static_assert(Emrys::Axes<EmrysTest::ReversedCholesky>);
    static_assert(Emrys::Repair<EmrysTest::EigenvalueFloor<1e-9>>);
    static_assert(!Emrys::Axes<roots::Ridge<>> && !Emrys::Repair<roots::Cholesky>);
    auto const cov = positiveDefinite();
    auto const upper = roots::squareRoot(EmrysTest::ReversedCholesky{}, cov).value();
    expectRootOf(upper, cov);
    EXPECT_TRUE(upper.isUpperTriangular(0));
    // In a list, with the user's repair retrying the user's axes on a semidefinite covariance.
    auto const repaired = roots::squareRootWithReport(
        roots::WithFallbacks<EmrysTest::ReversedCholesky, EmrysTest::EigenvalueFloor<1e-9>, roots::Pivoted>{},
        semidefinite()
    );
    ASSERT_EQ(repaired.report.index(), 1U);
    static_assert(std::is_same_v<std::variant_alternative_t<1, decltype(repaired.report)>, roots::NoReport>);
    expectRootOf(repaired.root, semidefinite(), 1e-8);
    // From a factor, the user's axes take the covariance formed.
    expectRootOf(Emrys::internal::rootFromFactor(EmrysTest::ReversedCholesky{}, wideFactorOf(cov)).value(), cov);
}

TEST(Roots, ABoundedCovarianceGivesABoundedRoot)
{
    using Bounded = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 4>;
    Bounded const cov = positiveDefinite();
    auto const root = roots::squareRoot(roots::cholesky(roots::pivoted()), cov);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(root)>, Bounded>);
    auto const principal = roots::squareRoot(roots::principal(), cov);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(principal)>, Bounded>);
    expectRootOf(root, cov);
    expectRootOf(principal, cov);
    static_assert(std::is_same_v<roots::RootMatrix<Eigen::Matrix<float, 3, 5>>, Eigen::Matrix3f>);
}

TEST(Roots, AFloatCovarianceTakesFloatRoots)
{
    Eigen::Matrix4f const cov = positiveDefinite().cast<float>();
    auto const root = roots::squareRoot(roots::cholesky(roots::ridge(), roots::pivoted()), cov);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(root)>, Eigen::Matrix4f>);
    EXPECT_LE((root * root.transpose() - cov).cwiseAbs().maxCoeff(), 1e-5F * cov.cwiseAbs().maxCoeff());
    auto const along = roots::squareRoot(roots::along(Eigen::Vector4f{1.0F, 0.0F, 0.0F, 1.0F}, roots::pivoted()), cov);
    EXPECT_LE((along * along.transpose() - cov).cwiseAbs().maxCoeff(), 1e-5F * cov.cwiseAbs().maxCoeff());
}

}  // namespace
