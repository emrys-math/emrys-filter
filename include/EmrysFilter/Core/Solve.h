#pragma once

/** \file
 * Solving by a factorization of a small matrix: transposedSolve() gives
 * \f$ (A^{-1} B)^T \f$, the shape of a Kalman gain and of a statistical linearization, from a
 * factorization of \f$ A \f$, and solveInPlace() overwrites \f$ B \f$ with
 * \f$ A^{-1} B \f$, and inverseFrom() gives \f$ A^{-1} \f$, all one column of \f$ B \f$ at a time
 * where \f$ A \f$ has a small fixed size, which Eigen's solve with a matrix right-hand side
 * does not do; from a Cholesky factorization of a small fixed size, transposedSolve() substitutes
 * through the factor itself (Cholesky.h). Multiply.h does the same for products. Everything here
 * is in `internal`. Read Stamp.h next.
 */

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <EmrysFilter/Core/Cholesky.h>
#include <EmrysFilter/Core/Multiply.h>

namespace Emrys::internal
{

/** The largest fixed size of a factorization that transposedSolve() solves one column at a time:
 * Eigen unrolls a triangular solve whose right-hand side is a vector of at most this fixed size,
 * and has no fixed-size solve beyond it.
 */
inline constexpr int unrolledSolveSize = 8;

/** Whether \a Decomposition is a CholeskyFactorization, which transposedSolve() asks for its own
 * transposed solve.
 */
template <typename Decomposition>
inline constexpr bool isCholeskyFactorization = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Matrix>
inline constexpr bool isCholeskyFactorization<CholeskyFactorization<Matrix>> = true;
/// \endcond

/** Whether transposedSolve() substitutes through \a Decomposition itself: an `Eigen::LLT` of the
 * lower triangle of a matrix that a CholeskyFactorization would factorize itself
 * (factorizesByColumns).
 */
template <typename Decomposition>
inline constexpr bool solvesBySubstitution = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Matrix>
inline constexpr bool solvesBySubstitution<Eigen::LLT<Matrix, Eigen::Lower>> = factorizesByColumns<Matrix>;

/// \endcond

/** \returns \f$ (A^{-1} B)^T \f$ as a \a Result, for \f$ A = L L^T \f$ of a fixed size, with
 * \f$ L \f$ the lower triangle of \a factor, \a reciprocal the reciprocals of its diagonal, and
 * \a rhs the right-hand side \f$ B \f$: forward and backward substitution, every right-hand side
 * together.
 *
 * Column \f$ r \f$ of the result holds unknown \f$ r \f$ of each right-hand side, so each step of
 * the substitution is one operation on whole columns. It multiplies by the reciprocals of the
 * diagonal, taken once, where Eigen's solve divides by the diagonal at each step, so the two
 * differ by rounding.
 *
 * Where the number of right-hand sides is fixed, the columns are padded with zeros to a multiple
 * of smallestPacketSize. Eigen vectorizes a column that is not, six `float`s say, from wherever
 * its address lets a packet start, so one column's stores are not the next operation's loads,
 * and the processor waits on each: unpadded, the gain of a 6-state filter in `float` took four
 * to six times as long as in `double` [benchmarks].
 */
template <typename Result, typename Factor, typename Reciprocal, typename Rhs>
[[nodiscard]] Result
substituteTransposed(Factor const & factor, Reciprocal const & reciprocal, Eigen::MatrixBase<Rhs> const & rhs)
{
    using Scalar = typename Factor::Scalar;
    constexpr int size = Factor::RowsAtCompileTime;
    constexpr int count = Result::RowsAtCompileTime;
    if constexpr (count != Eigen::Dynamic)
    {
        using Column = Eigen::Matrix<Scalar, paddedSize<Scalar, count>, 1>;
        auto work = Eigen::Matrix<Scalar, paddedSize<Scalar, count>, size>{};
        // L Y = B: unknown r of every right-hand side from those before it.
        for (int r = 0; r < size; ++r)
        {
            Column unknown = Column::Zero();
            unknown.template head<count>() = rhs.row(r).transpose();
            for (int c = 0; c < r; ++c)
            {
                unknown -= factor(r, c) * work.col(c);
            }
            work.col(r) = unknown * reciprocal(r);
        }
        // L^T X = Y: unknown r of every right-hand side from those after it.
        for (int r = size - 1; r >= 0; --r)
        {
            Column unknown = work.col(r);
            for (int c = r + 1; c < size; ++c)
            {
                unknown -= factor(c, r) * work.col(c);
            }
            work.col(r) = unknown * reciprocal(r);
        }
        Result result = work.template topRows<count>();
        return result;
    }
    else
    {
        // The same two passes in the result itself, which no fixed padding fits.
        auto result = Result{};
        result.resize(rhs.cols(), rhs.rows());
        for (int r = 0; r < size; ++r)
        {
            result.col(r) = rhs.row(r).transpose();
            for (int c = 0; c < r; ++c)
            {
                result.col(r) -= factor(r, c) * result.col(c);
            }
            result.col(r) *= reciprocal(r);
        }
        for (int r = size - 1; r >= 0; --r)
        {
            for (int c = r + 1; c < size; ++c)
            {
                result.col(r) -= factor(c, r) * result.col(c);
            }
            result.col(r) *= reciprocal(r);
        }
        return result;
    }
}

/** \returns \f$ (A^{-1} B)^T \f$ as a \a Result, for the factorization \a decomposition of
 * \f$ A \f$ (a CholeskyFactorization, an `Eigen::LLT` or `Eigen::LDLT`, or any with `solve()`)
 * and the right-hand side \a rhs, \f$ B \f$.
 *
 * Eigen sends a triangular solve with a matrix right-hand side to its blocked solver whatever
 * the matrix's size, which for a 3×3 \f$ A \f$ costs more than the products around it, while it
 * unrolls a solve with a vector of a small fixed size. So where \f$ A \f$ has a fixed size of at
 * most unrolledSolveSize, each column of \f$ B \f$ is solved alone, with the same factorization
 * and the same arithmetic per column; any other size is solved whole, as `solve()` does.
 *
 * Eigen's unrolled solve is slow too, though: for the gain of a 4×4 innovation covariance it
 * takes 208 instructions and six times as long as substituteTransposed() does [benchmarks]. So a
 * CholeskyFactorization gives the result itself, by that substitution where it factorized the
 * matrix itself, and through an `Eigen::LLT` of such a matrix (solvesBySubstitution) the result
 * is formed by the same substitution, from the reciprocals of its factor's diagonal.
 */
template <typename Result, typename Decomposition, typename Rhs>
[[nodiscard]] Result
transposedSolve(Decomposition const & decomposition, Eigen::MatrixBase<Rhs> const & rhs)
{
    constexpr int size = Decomposition::MatrixType::RowsAtCompileTime;
    if constexpr (isCholeskyFactorization<Decomposition>)
    {
        return decomposition.template transposedSolve<Result>(rhs);
    }
    else if constexpr (solvesBySubstitution<Decomposition>)
    {
        using Scalar = typename Decomposition::MatrixType::Scalar;
        Eigen::Matrix<Scalar, size, 1> const reciprocal = decomposition.matrixLLT().diagonal().cwiseInverse();
        return substituteTransposed<Result>(decomposition.matrixLLT(), reciprocal, rhs);
    }
    else if constexpr (size != Eigen::Dynamic && size <= unrolledSolveSize)
    {
        using Column = Eigen::Matrix<typename Rhs::Scalar, size, 1>;
        auto result = Result{};
        result.resize(rhs.cols(), rhs.rows());
        for (Eigen::Index j = 0; j < rhs.cols(); ++j)
        {
            Column const column = rhs.col(j);
            result.row(j) = decomposition.solve(column).transpose();
        }
        return result;
    }
    else
    {
        Result result = decomposition.solve(rhs).transpose();
        return result;
    }
}

/** Overwrites \a matrix, \f$ B \f$, with \f$ A^{-1} B \f$, for \a solver, anything with a
 * `solveInPlace()` for \f$ A \f$: an `Eigen::LLT` or `Eigen::LDLT`, or a triangular view.
 *
 * As in transposedSolve(), where \f$ B \f$ has a fixed number of rows of at most
 * unrolledSolveSize and more than one column, each column is solved alone, so that Eigen unrolls
 * it rather than calling its blocked solver; any other shape is solved whole. An inverse is this
 * on the identity.
 */
template <typename Solver, typename Derived>
void
solveInPlace(Solver const & solver, Eigen::MatrixBase<Derived> & matrix)
{
    constexpr int size = Derived::RowsAtCompileTime;
    if constexpr (size != Eigen::Dynamic && size <= unrolledSolveSize && Derived::ColsAtCompileTime != 1)
    {
        using Column = Eigen::Matrix<typename Derived::Scalar, size, 1>;
        for (Eigen::Index j = 0; j < matrix.cols(); ++j)
        {
            Column column = matrix.col(j);
            solver.solveInPlace(column);
            matrix.col(j) = column;
        }
    }
    else
    {
        solver.solveInPlace(matrix);
    }
}

/** \returns \f$ A^{-1} \f$ as a \a Result, for the factorization \a decomposition of \f$ A \f$
 * (an `Eigen::LLT`, `Eigen::LDLT`, `Eigen::PartialPivLU`, or `Eigen::FullPivLU`, or any with
 * `solve()` and `rows()`).
 *
 * Eigen's `inverse()` of a factorization solves the identity whole, by its blocked solver. Here,
 * as in transposedSolve(), where \f$ A \f$ has a fixed size of at most unrolledSolveSize each
 * column of the identity is solved alone; any other size is solved whole.
 */
template <typename Result, typename Decomposition>
[[nodiscard]] Result
inverseFrom(Decomposition const & decomposition)
{
    using Matrix = typename Decomposition::MatrixType;
    constexpr int size = Matrix::RowsAtCompileTime;
    Eigen::Index const n = decomposition.rows();
    if constexpr (size != Eigen::Dynamic && size <= unrolledSolveSize)
    {
        using Column = Eigen::Matrix<typename Matrix::Scalar, size, 1>;
        auto result = Result{};
        result.resize(n, n);
        for (Eigen::Index j = 0; j < n; ++j)
        {
            result.col(j) = decomposition.solve(Column::Unit(n, j));
        }
        return result;
    }
    else
    {
        Result result = decomposition.solve(Result::Identity(n, n));
        return result;
    }
}

}  // namespace Emrys::internal
