// internal::transposedSolve() from Core/Solve.h, the solve by a factorization that gives the gain
// of a conditional and the other transposed solves of the library through one of Eigen's
// factorizations (decisions.md [innovation-cholesky]): column by column through an LLT or a pivoted LDLT of a
// fixed size of at most eight, against Eigen's solve of the whole right-hand side, and the whole
// solve unchanged for any other size. The library's own factorization has its own transposed
// solve, which tests/Core/cholesky.cpp covers.

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

// An Eigen::LLT of a fixed size of at most eight is solved a column at a time, with Eigen's
// arithmetic for each column: the result agrees with Eigen's solve of the whole right-hand side
// to rounding and solves the system, with a fixed number of right-hand sides and with one known
// only at run time.
TEST(Solve, SolvesASmallFixedLltAColumnAtATime)
{
    Eigen::Matrix4d const a = positiveDefinite<4>();
    Eigen::Matrix<double, 4, 6> const b = Eigen::Matrix<double, 4, 6>::Random();
    auto const llt = Eigen::LLT<Eigen::Matrix4d>{a};

    auto const x = Emrys::internal::transposedSolve<Eigen::Matrix<double, 6, 4>>(llt, b);

    Eigen::Matrix<double, 6, 4> const expected = llt.solve(b).transpose();
    EXPECT_TRUE(x.isApprox(expected, 1e-14));
    EXPECT_TRUE((a * x.transpose()).isApprox(b, 1e-12));

    Eigen::Matrix<double, 4, Eigen::Dynamic> const dynamicB = Eigen::Matrix<double, 4, Eigen::Dynamic>::Random(4, 5);
    auto const dynamicX = Emrys::internal::transposedSolve<Eigen::Matrix<double, Eigen::Dynamic, 4>>(llt, dynamicB);
    ASSERT_EQ(dynamicX.rows(), 5);
    Eigen::Matrix<double, Eigen::Dynamic, 4> const dynamicExpected = llt.solve(dynamicB).transpose();
    EXPECT_TRUE(dynamicX.isApprox(dynamicExpected, 1e-14));
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

    using Large = Eigen::Matrix<double, 25, 25>;
    Large const largeA = positiveDefinite<25>();
    Eigen::Matrix<double, 25, 2> const largeB = Eigen::Matrix<double, 25, 2>::Random();
    auto const largeLlt = Eigen::LLT<Large>{largeA};
    Eigen::Matrix<double, 2, 25> const largeExpected = largeLlt.solve(largeB).transpose();
    EXPECT_EQ((Emrys::internal::transposedSolve<Eigen::Matrix<double, 2, 25>>(largeLlt, largeB)), largeExpected);

    // A fixed size above eight through a pivoted LDLT.
    using Middle = Eigen::Matrix<double, 12, 12>;
    Middle const middleA = positiveDefinite<12>();
    Eigen::Matrix<double, 12, 2> const middleB = Eigen::Matrix<double, 12, 2>::Random();
    auto const middleLdlt = Eigen::LDLT<Middle>{middleA};
    Eigen::Matrix<double, 2, 12> const middleExpected = middleLdlt.solve(middleB).transpose();
    EXPECT_EQ((Emrys::internal::transposedSolve<Eigen::Matrix<double, 2, 12>>(middleLdlt, middleB)), middleExpected);
}

}  // namespace
