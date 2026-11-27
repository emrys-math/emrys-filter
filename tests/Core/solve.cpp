// internal::transposedSolve() from Core/Solve.h, the solve by a factorization that gives the gain
// of a conditional and the other transposed solves of the library through one of Eigen's
// factorizations (decisions.md [innovation-cholesky]): column by column through an LLT or a pivoted LDLT of a
// fixed size of at most eight, against Eigen's solve of the whole right-hand side, and the whole
// solve unchanged for any other size. The library's own factorization has its own transposed
// solve, which tests/Core/cholesky.cpp covers. internal::solveInPlace() through a triangular view
// of a small fixed size substitutes itself, against Eigen's triangular solve.

#include <EmrysFilter/Core/Solve.h>

#include <gtest/gtest.h>

#include <Eigen/Cholesky>

#include <type_traits>

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

// A lower triangle, an upper one, and the transpose of a lower one, of a fixed size, are
// substituted through by the library's own substitution: for one right-hand side, for a fixed
// number, and for a number known only at run time, the result agrees with Eigen's triangular solve
// to rounding and solves the system.
template <typename Scalar, int Size>
void
expectSubstitutionAgrees(Scalar tolerance)
{
    using Matrix = Eigen::Matrix<Scalar, Size, Size>;
    Matrix root = Matrix::Random();
    root.diagonal().array() = root.diagonal().array().abs() + Scalar(Size);
    auto const check = [&](auto const & view, auto const & b) {
        static_assert(Emrys::internal::substitutesTriangular<std::remove_cvref_t<decltype(view)>>);
        auto x = b.eval();
        Emrys::internal::solveInPlace(view, x);
        auto const expected = view.solve(b).eval();
        EXPECT_TRUE(x.isApprox(expected, tolerance));
        EXPECT_TRUE((Matrix{view} * x).isApprox(b, tolerance));
        EXPECT_TRUE(Emrys::internal::solved(view, b).isApprox(expected, tolerance));
    };
    using Vector = Eigen::Matrix<Scalar, Size, 1>;
    using Fixed = Eigen::Matrix<Scalar, Size, 5>;
    using Dynamic = Eigen::Matrix<Scalar, Size, Eigen::Dynamic>;
    Vector const vector = Vector::Random();
    Fixed const fixed = Fixed::Random();
    Dynamic const dynamic = Dynamic::Random(Size, 3);
    auto const lower = root.template triangularView<Eigen::Lower>();
    auto const upper = root.template triangularView<Eigen::Upper>();
    auto const transposed = root.template triangularView<Eigen::Lower>().transpose();
    check(lower, vector);
    check(lower, fixed);
    check(lower, dynamic);
    check(upper, vector);
    check(upper, fixed);
    check(upper, dynamic);
    check(transposed, vector);
    check(transposed, fixed);
    check(transposed, dynamic);
}

TEST(Solve, SubstitutesThroughASmallFixedTriangle)
{
    expectSubstitutionAgrees<double, 1>(1e-14);
    expectSubstitutionAgrees<double, 4>(1e-12);
    expectSubstitutionAgrees<double, 24>(1e-10);
    expectSubstitutionAgrees<float, 6>(1e-4F);
}

// A triangular view of a dynamic size, or a fixed size above 24, goes to Eigen's triangular
// solve, exactly.
TEST(Solve, LeavesOtherTrianglesToEigen)
{
    Eigen::MatrixXd root = Eigen::MatrixXd::Random(5, 5);
    root.diagonal().array() += 3.0;
    auto const view = root.triangularView<Eigen::Lower>();
    static_assert(!Emrys::internal::substitutesTriangular<std::remove_cvref_t<decltype(view)>>);
    Eigen::MatrixXd const b = Eigen::MatrixXd::Random(5, 2);
    EXPECT_EQ(Emrys::internal::solved(view, b), view.solve(b));

    using Large = Eigen::Matrix<double, 25, 25>;
    Large large = Large::Random();
    large.diagonal().array() += 3.0;
    auto const largeView = large.triangularView<Eigen::Upper>();
    static_assert(!Emrys::internal::substitutesTriangular<std::remove_cvref_t<decltype(largeView)>>);
    Eigen::Matrix<double, 25, 1> const largeB = Eigen::Matrix<double, 25, 1>::Random();
    EXPECT_EQ(Emrys::internal::solved(largeView, largeB), largeView.solve(largeB));
}

}  // namespace
