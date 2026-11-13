#pragma once

/** \file
 * Solving by a factorization of a small matrix: transposedSolve() gives
 * \f$ (A^{-1} B)^T \f$, the shape of a Kalman gain and of a statistical linearization, from an
 * Eigen factorization of \f$ A \f$, and solveInPlace() overwrites \f$ B \f$ with
 * \f$ A^{-1} B \f$, and inverseFrom() gives \f$ A^{-1} \f$, all one column of \f$ B \f$ at a time
 * where \f$ A \f$ has a small fixed size, which Eigen's solve with a matrix right-hand side
 * does not do. Multiply.h does the same for products. Everything here is in `internal`. Read Stamp.h
 * next.
 */

#include <Eigen/Core>

#include <EmrysFilter/Core/Multiply.h>

namespace Emrys::internal
{

/** The largest fixed size of a factorization that transposedSolve() solves one column at a time:
 * Eigen unrolls a triangular solve whose right-hand side is a vector of at most this fixed size,
 * and has no fixed-size solve beyond it.
 */
inline constexpr int unrolledSolveSize = 8;

/** \returns \f$ (A^{-1} B)^T \f$ as a \a Result, for the factorization \a decomposition of
 * \f$ A \f$ (an `Eigen::LLT` or `Eigen::LDLT`, or any with `solve()`) and the right-hand side
 * \a rhs, \f$ B \f$.
 *
 * Eigen sends a triangular solve with a matrix right-hand side to its blocked solver whatever
 * the matrix's size, which for a 3×3 \f$ A \f$ costs more than the products around it, while it
 * unrolls a solve with a vector of a small fixed size. So where \f$ A \f$ has a fixed size of at
 * most unrolledSolveSize, each column of \f$ B \f$ is solved alone, with the same factorization
 * and the same arithmetic per column; any other size is solved whole, as `solve()` does.
 */
template <typename Result, typename Decomposition, typename Rhs>
[[nodiscard]] Result
transposedSolve(Decomposition const & decomposition, Eigen::MatrixBase<Rhs> const & rhs)
{
    constexpr int size = Decomposition::MatrixType::RowsAtCompileTime;
    if constexpr (size != Eigen::Dynamic && size <= unrolledSolveSize)
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
