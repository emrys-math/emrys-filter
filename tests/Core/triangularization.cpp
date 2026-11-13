// triangularize() and internal::lowerTriangularRoot() from Core/Triangularization.h, the
// triangularization primitive of roadmap step 17 that the square-root and information forms of the
// Gaussian (decisions.md [gaussian-forms]) are built on: the factor it leaves against a Cholesky factor of the
// product it stands for, columns carried along beside it, rank deficiency, and work in a block
// of a larger buffer.

#include <EmrysFilter/Core/Triangularization.h>

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include <utility>

namespace
{

// A writable matrix or a temporary block is taken; a const matrix is rejected rather than written
// through.
template <typename M>
concept Triangularizable = requires(M && m) { Emrys::triangularize(std::forward<M>(m)); };

using Block = decltype(std::declval<Eigen::MatrixXd &>().block(0, 0, 2, 2));
using ConstBlock = decltype(std::declval<Eigen::MatrixXd const &>().block(0, 0, 2, 2));
static_assert(Triangularizable<Eigen::Matrix3d &> && Triangularizable<Block>);
static_assert(!Triangularizable<Eigen::Matrix3d const &> && !Triangularizable<ConstBlock>);

// Triangularizing a tall M leaves an upper-triangular R with a nonnegative diagonal and
// R^T R = M^T M, which makes R the transpose of the Cholesky factor of M^T M. The reference is
// Eigen's LLT of the formed product, so the tolerance is the product's rounding.
TEST(Triangularization, FactorsTheGramMatrix)
{
    Eigen::Matrix<double, 7, 4> const m = Eigen::Matrix<double, 7, 4>::Random();
    Eigen::Matrix<double, 7, 4> r = m;
    Emrys::triangularize(r);

    EXPECT_TRUE(r.bottomRows(3).isZero(0.0));
    EXPECT_TRUE(r.topRows(4).triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
    EXPECT_TRUE((r.diagonal().array() >= 0.0).all());
    Eigen::Matrix4d const gram = m.transpose() * m;
    Eigen::Matrix4d const upper = gram.llt().matrixU();
    EXPECT_TRUE(r.topRows(4).isApprox(upper, 1e-12));
}

// A column appended to A is carried along as Q^T b, so the triangular solve R x = (Q^T b)'s top
// rows gives the least-squares solution of A x = b. The reference is Eigen's Householder QR solve.
TEST(Triangularization, CarriesAppendedColumns)
{
    Eigen::Matrix<double, 6, 3> const a = Eigen::Matrix<double, 6, 3>::Random();
    Eigen::Matrix<double, 6, 1> const b = Eigen::Matrix<double, 6, 1>::Random();
    auto augmented = Eigen::Matrix<double, 6, 4>{};
    augmented << a, b;
    Emrys::triangularize(augmented);

    Eigen::Vector3d const x =
        augmented.topLeftCorner<3, 3>().triangularView<Eigen::Upper>().solve(augmented.col(3).head<3>());
    Eigen::Vector3d const expected = a.householderQr().solve(b);
    EXPECT_TRUE(x.isApprox(expected, 1e-12));
}

// A rank-deficient M leaves a zero on the diagonal, not NaN, and still R^T R = M^T M.
TEST(Triangularization, ToleratesRankDeficiency)
{
    Eigen::Matrix<double, 5, 3> m = Eigen::Matrix<double, 5, 3>::Random();
    m.col(1) = 2.0 * m.col(0);
    Eigen::Matrix<double, 5, 3> r = m;
    Emrys::triangularize(r);

    EXPECT_TRUE(r.allFinite());
    EXPECT_NEAR(r(1, 1), 0.0, 1e-14);
    Eigen::Matrix3d const gram = m.transpose() * m;
    Eigen::Matrix3d const product = r.transpose() * r;
    EXPECT_TRUE(product.isApprox(gram, 1e-12));
}

// A wide matrix (fewer rows than columns) and a block of a dynamic buffer are triangularized in
// place, leaving the rest of the buffer alone.
TEST(Triangularization, WorksInABlock)
{
    Eigen::MatrixXd buffer = Eigen::MatrixXd::Random(6, 8);
    Eigen::MatrixXd const original = buffer;
    Emrys::triangularize(buffer.block(1, 2, 3, 5));

    Eigen::MatrixXd const block = original.block(1, 2, 3, 5);
    Eigen::MatrixXd const r = buffer.block(1, 2, 3, 5);
    EXPECT_TRUE(r.triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
    Eigen::MatrixXd const gram = block.transpose() * block;
    Eigen::MatrixXd const product = r.transpose() * r;
    EXPECT_TRUE(product.isApprox(gram, 1e-12));
    EXPECT_EQ(buffer.topRows(1), original.topRows(1));
    EXPECT_EQ(buffer.leftCols(2), original.leftCols(2));
}

// A single row, which the loop reaches with nothing below the diagonal, is only sign-flipped, so
// a one-dimensional data array keeps R^T R.
TEST(Triangularization, HandlesASingleRow)
{
    auto row = Eigen::Matrix<double, 1, 2, Eigen::RowMajor>{-2.0, 3.0};
    Emrys::triangularize(row);
    EXPECT_EQ(row(0, 0), 2.0);
    EXPECT_EQ(row(0, 1), -3.0);
}

// lowerTriangularRoot() of a wide A is the lower Cholesky factor of A A^T, and with fewer columns
// than rows its trailing columns are zero while L L^T still equals A A^T.
TEST(Triangularization, LowerFactorOfAWideMatrix)
{
    Eigen::Matrix<double, 3, 5> const a = Eigen::Matrix<double, 3, 5>::Random();
    Eigen::Matrix3d const l = Emrys::internal::lowerTriangularRoot(a);
    Eigen::Matrix3d const gram = a * a.transpose();
    Eigen::Matrix3d const cholesky = gram.llt().matrixL();
    EXPECT_TRUE(l.isApprox(cholesky, 1e-12));

    Eigen::Matrix<double, 3, 2> const narrow = Eigen::Matrix<double, 3, 2>::Random();
    Eigen::Matrix3d const thin = Emrys::internal::lowerTriangularRoot(narrow);
    EXPECT_TRUE(thin.col(2).isZero(0.0));
    Eigen::Matrix3d const product = thin * thin.transpose();
    Eigen::Matrix3d const expected = narrow * narrow.transpose();
    EXPECT_TRUE(product.isApprox(expected, 1e-12));
}

}  // namespace
