// CholeskyFactorization from Core/Cholesky.h, the Cholesky factorization that the
// innovation covariance's gain, NIS, and log-determinant are taken from (decisions.md [own-cholesky],
// [innovation-cholesky]): its own factorization without square roots, by whole columns, of a real
// matrix of a fixed size of at most 24, against Eigen's LLT and against the matrix itself; that
// only the lower triangle's values are used; what it reports for a matrix that is not positive
// definite; its solves and its whitening; and the Eigen::LLT it holds for any other matrix. It is
// public: Gaussian::cholesky() returns one.

#include <EmrysFilter/Core/Cholesky.h>
#include <EmrysFilter/Core/Solve.h>

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <unsupported/Eigen/AutoDiff>

#include <complex>
#include <cstring>
#include <limits>
#include <type_traits>

namespace
{

using Emrys::CholeskyFactorization;

/// A symmetric positive definite matrix of \a Matrix's type, \a size × \a size where that is not
/// fixed.
template <typename Matrix>
Matrix
positiveDefinite(Eigen::Index size = Matrix::RowsAtCompileTime)
{
    using Scalar = typename Matrix::Scalar;
    Matrix const spread = Matrix::Random(size, size);
    Matrix result = spread * spread.transpose();
    result.diagonal().array() += Scalar(0.5);
    return result;
}

/// The relative tolerance for a result in \a Scalar that differs from Eigen's by rounding.
template <typename Scalar>
constexpr Scalar rounding = std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-10);

/// Expects the factorization of a \a Size × \a Size matrix in \a Scalar to be the one made here,
/// to agree with Eigen's LLT in its factor and its log-determinant, and to give back the matrix.
/// The two differ by rounding: the factor here is the unit factor times the roots of the pivots,
/// and the log-determinant the sum of the pivots' logarithms.
template <int Size, typename Scalar = double>
void
expectFactorizes()
{
    using Matrix = Eigen::Matrix<Scalar, Size, Size>;
    ASSERT_TRUE(Emrys::internal::factorizesByColumns<Matrix>);
    Matrix const a = positiveDefinite<Matrix>();

    auto const cholesky = CholeskyFactorization<Matrix>{a};

    ASSERT_TRUE(cholesky.isPositiveDefinite()) << "size " << Size;
    EXPECT_EQ(cholesky.rows(), Size);
    auto const llt = Eigen::LLT<Matrix>{a};
    Matrix const factor = cholesky.lower();
    Matrix const expected = llt.matrixL();
    EXPECT_TRUE(factor.isApprox(expected, rounding<Scalar>)) << "size " << Size;
    EXPECT_TRUE(factor.template triangularView<Eigen::StrictlyUpper>().toDenseMatrix().isZero(Scalar(0)));
    EXPECT_TRUE((factor * factor.transpose()).isApprox(a, rounding<Scalar>)) << "size " << Size;
    Scalar const logDeterminant = Scalar(2) * expected.diagonal().array().log().sum();
    EXPECT_NEAR(double(cholesky.logDeterminant()), double(logDeterminant), double(rounding<Scalar>) * Size)
        << "size " << Size;
}

// A real matrix of a fixed size from 1 to 24 is factorized here, by whole columns: sizes that are
// a multiple of a packet of 16 bytes and sizes that are not, whose columns are padded, in double,
// float, and long double, whose columns have no packets.
TEST(Cholesky, FactorizesAFixedSizeOfAtMost24ByColumns)
{
    expectFactorizes<1>();
    expectFactorizes<2>();
    expectFactorizes<3>();
    expectFactorizes<4>();
    expectFactorizes<5>();
    expectFactorizes<6>();
    expectFactorizes<8>();
    expectFactorizes<12>();
    expectFactorizes<24>();
    expectFactorizes<1, float>();
    expectFactorizes<3, float>();
    expectFactorizes<4, float>();
    expectFactorizes<6, float>();
    expectFactorizes<7, float>();
    expectFactorizes<8, float>();
    expectFactorizes<24, float>();
    expectFactorizes<4, long double>();
}

// The elements above the diagonal are loaded with their columns and nothing is taken from them:
// with NaN there, with values too large to square, and with other finite values, the factor, the
// log-determinant, and the solutions are the symmetric matrix's, bit for bit.
TEST(Cholesky, UsesOnlyTheLowerTriangle)
{
    using Matrix = Eigen::Matrix<double, 6, 6>;
    using Transposed = Eigen::Matrix<double, 3, 6>;
    Matrix const a = positiveDefinite<Matrix>();
    Eigen::Matrix<double, 6, 3> const b = Eigen::Matrix<double, 6, 3>::Random();
    auto const whole = CholeskyFactorization<Matrix>{a};
    ASSERT_TRUE(whole.isPositiveDefinite());
    Matrix const expected = whole.lower();
    Transposed const expectedSolution = whole.transposedSolve<Transposed>(b);

    for (double const above : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max(), 3.0})
    {
        Matrix lower = a;
        lower.triangularView<Eigen::StrictlyUpper>().setConstant(above);

        auto const cholesky = CholeskyFactorization<Matrix>{lower};

        ASSERT_TRUE(cholesky.isPositiveDefinite()) << "with " << above << " above the diagonal";
        Matrix const factor = cholesky.lower();
        EXPECT_EQ(std::memcmp(factor.data(), expected.data(), sizeof(double) * 36), 0)
            << "with " << above << " above the diagonal";
        EXPECT_EQ(cholesky.logDeterminant(), whole.logDeterminant()) << "with " << above << " above the diagonal";
        Transposed const solution = cholesky.transposedSolve<Transposed>(b);
        EXPECT_EQ(std::memcmp(solution.data(), expectedSolution.data(), sizeof(double) * 18), 0)
            << "with " << above << " above the diagonal";
    }
}

/// Expects a \a Matrix, \a size × \a size where that is not fixed, to be reported positive definite
/// only when it is: not with a negative eigenvalue, not when singular, and not with a NaN in its
/// lower triangle, which Eigen's LLT alone reports as a success.
template <typename Matrix>
void
expectReportsWhatIsNotPositiveDefinite(Eigen::Index size = Matrix::RowsAtCompileTime)
{
    using Scalar = typename Matrix::Scalar;
    Matrix const a = positiveDefinite<Matrix>(size);
    EXPECT_TRUE(CholeskyFactorization<Matrix>{a}.isPositiveDefinite()) << "size " << size;

    Matrix indefinite = a;
    indefinite(size - 1, size - 1) = Scalar(-1);
    EXPECT_FALSE(CholeskyFactorization<Matrix>{indefinite}.isPositiveDefinite()) << "size " << size;

    Matrix const singular = Matrix::Zero(size, size);
    EXPECT_FALSE(CholeskyFactorization<Matrix>{singular}.isPositiveDefinite()) << "size " << size;

    Matrix notANumber = a;
    notANumber(size - 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    EXPECT_FALSE(CholeskyFactorization<Matrix>{notANumber}.isPositiveDefinite()) << "size " << size;
}

// A matrix that is not positive definite is reported at every size, by the factorization made
// here and by the one Eigen makes: a fixed size up to 24, a larger one, a dynamic one, and a
// bounded one.
TEST(Cholesky, ReportsAMatrixThatIsNotPositiveDefinite)
{
    expectReportsWhatIsNotPositiveDefinite<Eigen::Matrix<double, 1, 1>>();
    expectReportsWhatIsNotPositiveDefinite<Eigen::Matrix<double, 4, 4>>();
    expectReportsWhatIsNotPositiveDefinite<Eigen::Matrix<float, 6, 6>>();
    expectReportsWhatIsNotPositiveDefinite<Eigen::Matrix<double, 24, 24>>();
    expectReportsWhatIsNotPositiveDefinite<Eigen::Matrix<double, 25, 25>>();
    expectReportsWhatIsNotPositiveDefinite<Eigen::MatrixXd>(5);
    expectReportsWhatIsNotPositiveDefinite<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6>>(4);
}

// A factorization made of nothing reports no positive definite matrix, and compute() replaces a
// factorization that failed, as the repair of an innovation covariance does.
TEST(Cholesky, ComputesAgainAfterAMatrixThatIsNotPositiveDefinite)
{
    auto cholesky = CholeskyFactorization<Eigen::Matrix3d>{};
    EXPECT_FALSE(cholesky.isPositiveDefinite());
    EXPECT_EQ(cholesky.rows(), 3);

    Eigen::Matrix3d const a = positiveDefinite<Eigen::Matrix3d>();
    cholesky.compute(Eigen::Matrix3d{-a});
    EXPECT_FALSE(cholesky.isPositiveDefinite());

    cholesky.compute(a);
    ASSERT_TRUE(cholesky.isPositiveDefinite());
    Eigen::Matrix3d const factor = cholesky.lower();
    EXPECT_TRUE((factor * factor.transpose()).isApprox(a, 1e-12));
}

/// Expects the factorization of a \a Matrix, \a size × \a size where that is not fixed, to be
/// Eigen's: the same factor, log-determinant, quadratic form, solve, and transposed solve, exactly.
template <typename Matrix>
void
expectFactorizesByEigen(Eigen::Index size = Matrix::RowsAtCompileTime)
{
    using Scalar = typename Matrix::Scalar;
    using Rhs = Eigen::Matrix<Scalar, Matrix::RowsAtCompileTime, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Matrix::RowsAtCompileTime, 1>;
    using Transposed = Eigen::Matrix<Scalar, Eigen::Dynamic, Matrix::RowsAtCompileTime>;
    ASSERT_FALSE(Emrys::internal::factorizesByColumns<Matrix>);
    Matrix const a = positiveDefinite<Matrix>(size);
    Rhs const b = Rhs::Random(size, 3);

    auto const cholesky = CholeskyFactorization<Matrix>{a};

    ASSERT_TRUE(cholesky.isPositiveDefinite());
    EXPECT_EQ(cholesky.rows(), size);
    auto const llt = Eigen::LLT<Matrix>{a};
    EXPECT_EQ(cholesky.lower(), Matrix{llt.matrixL()});
    EXPECT_EQ(cholesky.logDeterminant(), Scalar(2) * llt.matrixLLT().diagonal().array().log().sum());
    Vector whitened = b.col(0);
    llt.matrixL().solveInPlace(whitened);
    EXPECT_EQ(cholesky.inverseQuadraticForm(b.col(0)), whitened.squaredNorm());
    Rhs const solved = llt.solve(b);
    EXPECT_EQ(cholesky.solve(b), solved);
    Rhs inPlace = b;
    cholesky.solveInPlace(inPlace);
    EXPECT_EQ(inPlace, solved);
    Transposed const transposed = solved.transpose();
    EXPECT_EQ(Emrys::internal::transposedSolve<Transposed>(cholesky, b), transposed);
    Rhs whitenedByEigen = b;
    llt.matrixL().solveInPlace(whitenedByEigen);
    Rhs whitenedHere = b;
    cholesky.whitenInPlace(whitenedHere);
    EXPECT_EQ(whitenedHere, whitenedByEigen);
}

// A fixed size above 24, a dynamic size, and a bounded one are factorized and solved by the
// Eigen::LLT that the factorization holds, as before it was the library's own; so are a complex
// matrix and an empty one.
TEST(Cholesky, FactorizesOtherSizesByEigen)
{
    expectFactorizesByEigen<Eigen::Matrix<double, 25, 25>>();
    expectFactorizesByEigen<Eigen::MatrixXd>(5);
    expectFactorizesByEigen<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6>>(4);
    EXPECT_FALSE((Emrys::internal::factorizesByColumns<Eigen::Matrix<std::complex<double>, 3, 3>>));
    EXPECT_FALSE((Emrys::internal::factorizesByColumns<Eigen::Matrix<double, 0, 0>>));
}

/// Expects the solves through the factorization of a \a Size × \a Size matrix in \a Scalar to
/// agree with Eigen's LLT and to solve the system: the transposed solve with \a Count right-hand
/// sides, solve() and solveInPlace(), which are its transpose, the quadratic form of the
/// inverse, which the NIS is, and the whitening, whose columns have that form as their squared
/// length.
template <int Size, int Count, typename Scalar = double>
void
expectSolves()
{
    using Matrix = Eigen::Matrix<Scalar, Size, Size>;
    using Rhs = Eigen::Matrix<Scalar, Size, Count>;
    using Vector = Eigen::Matrix<Scalar, Size, 1>;
    using Transposed = Eigen::Matrix<Scalar, Count, Size>;
    Matrix const a = positiveDefinite<Matrix>();
    Rhs const b = Rhs::Random();
    auto const cholesky = CholeskyFactorization<Matrix>{a};
    ASSERT_TRUE(cholesky.isPositiveDefinite());
    auto const llt = Eigen::LLT<Matrix>{a};
    Rhs const expected = llt.solve(b);

    Transposed const transposed = cholesky.template transposedSolve<Transposed>(b);
    EXPECT_TRUE(transposed.isApprox(expected.transpose(), rounding<Scalar>)) << "size " << Size;
    EXPECT_TRUE((a * transposed.transpose()).isApprox(b, rounding<Scalar>)) << "size " << Size;
    EXPECT_EQ(Emrys::internal::transposedSolve<Transposed>(cholesky, b), transposed) << "size " << Size;

    EXPECT_TRUE(cholesky.solve(b).isApprox(expected, rounding<Scalar>)) << "size " << Size;
    Rhs inPlace = b;
    cholesky.solveInPlace(inPlace);
    EXPECT_EQ(inPlace, cholesky.solve(b)) << "size " << Size;
    Vector const first = b.col(0);
    EXPECT_TRUE(cholesky.solve(first).isApprox(expected.col(0), rounding<Scalar>)) << "size " << Size;

    Scalar const form = first.dot(llt.solve(first));
    double const formTolerance = double(rounding<Scalar> * form);
    EXPECT_NEAR(double(cholesky.inverseQuadraticForm(first)), double(form), formTolerance) << "size " << Size;
    EXPECT_NEAR(double(cholesky.inverseQuadraticForm(b.col(0))), double(form), formTolerance) << "size " << Size;

    Rhs whitened = b;
    cholesky.whitenInPlace(whitened);
    Rhs expectedWhitened = b;
    llt.matrixL().solveInPlace(expectedWhitened);
    EXPECT_TRUE(whitened.isApprox(expectedWhitened, rounding<Scalar>)) << "size " << Size;
    EXPECT_NEAR(double(whitened.col(0).squaredNorm()), double(form), formTolerance) << "size " << Size;
    Vector whitenedVector = first;
    cholesky.whitenInPlace(whitenedVector);
    EXPECT_EQ(whitenedVector, Vector{whitened.col(0)}) << "size " << Size;
}

// The solves through a factorization made here agree with Eigen's, with one right-hand side, a
// few, a number that is not a multiple of a packet, and more of them than the size, in double, in
// float, and in long double.
TEST(Cholesky, SolvesThroughTheFactorization)
{
    expectSolves<1, 1>();
    expectSolves<1, 4>();
    expectSolves<3, 1>();
    expectSolves<3, 7>();
    expectSolves<4, 4>();
    expectSolves<6, 5>();
    expectSolves<12, 6>();
    expectSolves<24, 3>();
    expectSolves<4, 3, float>();
    expectSolves<4, 4, float>();
    expectSolves<6, 6, float>();
    expectSolves<8, 3, float>();
    expectSolves<5, 9, float>();
    expectSolves<4, 6, long double>();
}

// The transposed solve writes into a result whose number of right-hand sides is known only at
// run time, as the gain of a state of dynamic size is, and solveInPlace() into such a matrix.
TEST(Cholesky, SolvesWithADynamicNumberOfRightHandSides)
{
    Eigen::Matrix3d const a = positiveDefinite<Eigen::Matrix3d>();
    Eigen::Matrix<double, 3, Eigen::Dynamic> const b = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 5);
    auto const cholesky = CholeskyFactorization<Eigen::Matrix3d>{a};
    ASSERT_TRUE(cholesky.isPositiveDefinite());

    auto const x = Emrys::internal::transposedSolve<Eigen::Matrix<double, Eigen::Dynamic, 3>>(cholesky, b);

    ASSERT_EQ(x.rows(), 5);
    Eigen::Matrix<double, 3, Eigen::Dynamic> const expected = Eigen::LLT<Eigen::Matrix3d>{a}.solve(b);
    EXPECT_TRUE(x.isApprox(expected.transpose(), 1e-12));
    Eigen::Matrix<double, 3, Eigen::Dynamic> inPlace = b;
    cholesky.solveInPlace(inPlace);
    EXPECT_TRUE(inPlace.isApprox(expected, 1e-12));
    Eigen::Matrix<double, 3, Eigen::Dynamic> whitened = b;
    cholesky.whitenInPlace(whitened);
    EXPECT_TRUE((whitened.transpose() * whitened).isApprox(b.transpose() * expected, 1e-12));
}

// A matrix given as an expression, here a sum, is evaluated once and factorized, and a block of a
// larger matrix, which has its coefficients in memory, is read where it is: both give the factor
// of the same matrix held as a plain one.
TEST(Cholesky, FactorizesAnExpressionAndABlock)
{
    using Matrix = Eigen::Matrix<double, 4, 4>;
    Matrix const a = positiveDefinite<Matrix>();
    Matrix const noise = Matrix::Identity();
    Matrix const sum = a + noise;
    auto const expected = CholeskyFactorization<Matrix>{sum};
    ASSERT_TRUE(expected.isPositiveDefinite());

    auto const fromExpression = CholeskyFactorization<Matrix>{a + noise};
    ASSERT_TRUE(fromExpression.isPositiveDefinite());
    EXPECT_EQ(fromExpression.lower(), expected.lower());

    Eigen::Matrix<double, 6, 6> larger = Eigen::Matrix<double, 6, 6>::Zero();
    larger.block<4, 4>(1, 1) = sum;
    auto const fromBlock = CholeskyFactorization<Matrix>{larger.block<4, 4>(1, 1)};
    ASSERT_TRUE(fromBlock.isPositiveDefinite());
    EXPECT_EQ(fromBlock.lower(), expected.lower());
}

// The factorization made here runs in an automatic-differentiation scalar, whose columns have no
// packets: the factor's values are the double factorization's, the derivative of its first
// diagonal element, sqrt(a00), with respect to a00 is 1 / (2 sqrt(a00)), and the derivative of the
// log-determinant with respect to a00 is the first diagonal element of the inverse.
TEST(Cholesky, FactorizesInAnAutoDiffScalar)
{
    using Dual = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
    using DualMatrix = Eigen::Matrix<Dual, 3, 3>;
    Eigen::Matrix3d const a = positiveDefinite<Eigen::Matrix3d>();
    DualMatrix dual = a.cast<Dual>();
    for (int i = 0; i < 9; ++i)
    {
        dual(i).derivatives().setZero();
    }
    dual(0, 0).derivatives()(0) = 1.0;

    auto const cholesky = CholeskyFactorization<DualMatrix>{dual};

    ASSERT_TRUE(cholesky.isPositiveDefinite());
    Eigen::Matrix3d const expected = CholeskyFactorization<Eigen::Matrix3d>{a}.lower();
    DualMatrix const factor = cholesky.lower();
    for (int j = 0; j < 3; ++j)
    {
        for (int i = j; i < 3; ++i)
        {
            EXPECT_NEAR(factor(i, j).value(), expected(i, j), 1e-14);
        }
    }
    EXPECT_NEAR(factor(0, 0).derivatives()(0), 0.5 / expected(0, 0), 1e-12);
    Dual const logDeterminant = cholesky.logDeterminant();
    Eigen::Matrix3d const inverse = a.inverse();
    EXPECT_NEAR(logDeterminant.derivatives()(0), inverse(0, 0), 1e-10);
}

}  // namespace
