// internal::transposedSolve() from Core/Solve.h, the solve by a factorization that gives the
// Kalman gain (decisions.md [innovation-cholesky]) and the other transposed solves of the library: column by column
// for a fixed size of at most eight, against Eigen's solve of the whole right-hand side, for a
// Cholesky factorization and for a pivoted LDLT of a singular matrix, and the whole solve
// unchanged for any other size.

#include <EmrysFilter/Core/Solve.h>

#include <gtest/gtest.h>

#include <Eigen/Cholesky>

namespace
{

template <int Size>
Eigen::Matrix<double, Size, Size>
positiveDefinite(Eigen::Index size = Size)
{
    Eigen::Matrix<double, Size, Size> const spread = Eigen::Matrix<double, Size, Size>::Random(size, size);
    Eigen::Matrix<double, Size, Size> result = spread * spread.transpose();
    result.diagonal().array() += 0.5;
    return result;
}

// A 3×3 Cholesky factorization solves a 3×6 right-hand side one column at a time. The reference is
// Eigen's solve of the whole matrix; the two differ only in the order of the rounding.
TEST(Solve, SolvesASmallFixedSizeByColumns)
{
    Eigen::Matrix3d const a = positiveDefinite<3>();
    Eigen::Matrix<double, 3, 6> const b = Eigen::Matrix<double, 3, 6>::Random();
    auto const llt = Eigen::LLT<Eigen::Matrix3d>{a};

    auto const x = Emrys::internal::transposedSolve<Eigen::Matrix<double, 6, 3>>(llt, b);

    Eigen::Matrix<double, 6, 3> const expected = llt.solve(b).transpose();
    EXPECT_TRUE(x.isApprox(expected, 1e-14));
    EXPECT_TRUE((a * x.transpose()).isApprox(b, 1e-12));
}

// A pivoted LDLT of a 4×4 with a coordinate known exactly (decisions.md [gaussian-forms]), a zero row and
// column, skips the zero pivot in each column's solve as it does in the whole solve: the solution
// of a right-hand side in the range agrees with Eigen's and is zero in that coordinate.
TEST(Solve, KeepsThePivotedLdltOfASingularMatrix)
{
    Eigen::Matrix4d a = positiveDefinite<4>();
    a.row(2).setZero();
    a.col(2).setZero();
    Eigen::Matrix<double, 4, 5> b = Eigen::Matrix<double, 4, 5>::Random();
    b.row(2).setZero();
    auto const ldlt = Eigen::LDLT<Eigen::Matrix4d>{a};

    auto const x = Emrys::internal::transposedSolve<Eigen::Matrix<double, 5, 4>>(ldlt, b);

    Eigen::Matrix<double, 5, 4> const expected = ldlt.solve(b).transpose();
    EXPECT_TRUE(x.isApprox(expected, 1e-14));
    EXPECT_TRUE(x.col(2).isZero(0.0));
    EXPECT_TRUE((a * x.transpose()).isApprox(b, 1e-12));
}

// A dynamic size, a bounded one, and a fixed size above eight are solved whole, exactly as Eigen
// solves them.
TEST(Solve, SolvesOtherSizesWhole)
{
    Eigen::MatrixXd const dynamicA = positiveDefinite<Eigen::Dynamic>(3);
    Eigen::MatrixXd const dynamicB = Eigen::MatrixXd::Random(3, 4);
    auto const dynamicLlt = Eigen::LLT<Eigen::MatrixXd>{dynamicA};
    Eigen::MatrixXd const dynamicExpected = dynamicLlt.solve(dynamicB).transpose();
    EXPECT_EQ(Emrys::internal::transposedSolve<Eigen::MatrixXd>(dynamicLlt, dynamicB), dynamicExpected);

    using Bounded = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 4, 4>;
    Bounded const boundedA = dynamicA;
    Bounded const boundedB = dynamicB.leftCols(3);
    auto const boundedLlt = Eigen::LLT<Bounded>{boundedA};
    Bounded const boundedExpected = boundedLlt.solve(boundedB).transpose();
    EXPECT_EQ(Emrys::internal::transposedSolve<Bounded>(boundedLlt, boundedB), boundedExpected);

    using Large = Eigen::Matrix<double, 12, 12>;
    Large const largeA = positiveDefinite<12>();
    Eigen::Matrix<double, 12, 2> const largeB = Eigen::Matrix<double, 12, 2>::Random();
    auto const largeLlt = Eigen::LLT<Large>{largeA};
    Eigen::Matrix<double, 2, 12> const largeExpected = largeLlt.solve(largeB).transpose();
    EXPECT_EQ((Emrys::internal::transposedSolve<Eigen::Matrix<double, 2, 12>>(largeLlt, largeB)), largeExpected);
}

}  // namespace
