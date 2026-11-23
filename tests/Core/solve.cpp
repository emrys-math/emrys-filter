// internal::transposedSolve() from Core/Solve.h, the solve by a factorization that gives the
// Kalman gain (decisions.md [innovation-cholesky]) and the other transposed solves of the library: by its own
// substitution through a Cholesky factorization of a fixed size of at most 24, and column by column
// through a pivoted LDLT of a fixed size of at most eight, each against Eigen's solve of the whole
// right-hand side, and the whole solve unchanged for any other size.

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

/// Expects transposedSolve() through the Cholesky factorization of a \a Size × \a Size matrix in
/// \a Scalar, with \a Count right-hand sides, to agree with Eigen's solve of the whole right-hand
/// side and to solve the system. The two differ by rounding: the substitution multiplies by the
/// reciprocals of the factor's diagonal where Eigen divides by it.
template <int Size, int Count, typename Scalar = double>
void
expectSubstitutionSolves()
{
    using Matrix = Eigen::Matrix<Scalar, Size, Size>;
    using Rhs = Eigen::Matrix<Scalar, Size, Count>;
    using Result = Eigen::Matrix<Scalar, Count, Size>;
    Matrix const spread = Matrix::Random();
    Matrix a = spread * spread.transpose();
    a.diagonal().array() += Scalar(0.5);
    Rhs const b = Rhs::Random();
    auto const llt = Eigen::LLT<Matrix>{a};
    ASSERT_TRUE(Emrys::internal::solvesBySubstitution<Eigen::LLT<Matrix>>);

    auto const x = Emrys::internal::transposedSolve<Result>(llt, b);

    Result const expected = llt.solve(b).transpose();
    Scalar const rounding = std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-10);
    EXPECT_TRUE(x.isApprox(expected, rounding)) << "size " << Size << ", " << Count << " right-hand sides";
    EXPECT_TRUE((a * x.transpose()).isApprox(b, rounding)) << "size " << Size << ", " << Count << " right-hand sides";
}

// A Cholesky factorization of a fixed size of at most 24 is solved by the substitution written out
// in transposedSolve(), every right-hand side together: from 1×1 to the largest size, with one
// right-hand side, a few, and more of them than the size.
TEST(Solve, SubstitutesThroughASmallFixedCholeskyFactorization)
{
    expectSubstitutionSolves<1, 1>();
    expectSubstitutionSolves<1, 4>();
    expectSubstitutionSolves<3, 6>();
    expectSubstitutionSolves<4, 4>();
    expectSubstitutionSolves<8, 4>();
    expectSubstitutionSolves<12, 6>();
    expectSubstitutionSolves<24, 6>();
    expectSubstitutionSolves<24, 1>();
}

// The columns of the substitution are padded with zeros to whole packets of 16 bytes, and the
// padding does not reach the result: numbers of right-hand sides that are not a multiple of a
// packet, 1, 3, 5, and 7 doubles and 1, 3, 5, 6, 7, and 9 floats, beside those that are.
TEST(Solve, PadsTheRightHandSidesToWholePackets)
{
    EXPECT_GE(Emrys::internal::smallestPacketSize<float>, Emrys::internal::smallestPacketSize<double>);
    expectSubstitutionSolves<4, 1>();
    expectSubstitutionSolves<4, 3>();
    expectSubstitutionSolves<6, 5>();
    expectSubstitutionSolves<3, 7>();
    expectSubstitutionSolves<4, 1, float>();
    expectSubstitutionSolves<4, 3, float>();
    expectSubstitutionSolves<4, 4, float>();
    expectSubstitutionSolves<6, 5, float>();
    expectSubstitutionSolves<6, 6, float>();
    expectSubstitutionSolves<12, 6, float>();
    expectSubstitutionSolves<3, 7, float>();
    expectSubstitutionSolves<8, 8, float>();
    expectSubstitutionSolves<5, 9, float>();
    expectSubstitutionSolves<4, 6, long double>();
}

// The substitution writes into a result whose number of right-hand sides is known only at run
// time, as the gain of a state of dynamic size is.
TEST(Solve, SubstitutesIntoADynamicNumberOfRightHandSides)
{
    Eigen::Matrix3d const a = positiveDefinite<3>();
    Eigen::Matrix<double, 3, Eigen::Dynamic> const b = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 5);
    auto const llt = Eigen::LLT<Eigen::Matrix3d>{a};

    auto const x = Emrys::internal::transposedSolve<Eigen::Matrix<double, Eigen::Dynamic, 3>>(llt, b);

    ASSERT_EQ(x.rows(), 5);
    Eigen::Matrix<double, Eigen::Dynamic, 3> const expected = llt.solve(b).transpose();
    EXPECT_TRUE(x.isApprox(expected, 1e-12));
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

// A dynamic size, a bounded one, and a fixed size above 24 are solved whole, exactly as Eigen
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

    // A fixed size above eight through a pivoted LDLT, which has no substitution here.
    using Middle = Eigen::Matrix<double, 12, 12>;
    Middle const middleA = positiveDefinite<12>();
    Eigen::Matrix<double, 12, 2> const middleB = Eigen::Matrix<double, 12, 2>::Random();
    auto const middleLdlt = Eigen::LDLT<Middle>{middleA};
    Eigen::Matrix<double, 2, 12> const middleExpected = middleLdlt.solve(middleB).transpose();
    EXPECT_EQ((Emrys::internal::transposedSolve<Eigen::Matrix<double, 2, 12>>(middleLdlt, middleB)), middleExpected);
}

}  // namespace
