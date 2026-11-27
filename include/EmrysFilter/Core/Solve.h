#pragma once

/** \file
 * Solving by a factorization of a small matrix: transposedSolve() gives
 * \f$ (A^{-1} B)^T \f$, the shape of a Kalman gain and of a statistical linearization, from a
 * factorization of \f$ A \f$, and solveInPlace() overwrites \f$ B \f$ with
 * \f$ A^{-1} B \f$, and inverseFrom() gives \f$ A^{-1} \f$, all one column of \f$ B \f$ at a time
 * where \f$ A \f$ has a small fixed size, which Eigen's solve with a matrix right-hand side
 * does not do; a CholeskyFactorization (Cholesky.h) gives its own transposed solve, and
 * solveInPlace() substitutes through a small fixed triangular view itself. Multiply.h does the
 * same for products. Everything here is in `internal`. Read Stamp.h next.
 */

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <EmrysFilter/Core/Cholesky.h>
#include <EmrysFilter/Core/Macros.h>
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
 * takes 208 instructions and six times as long as a substitution written out [benchmarks]. So a
 * CholeskyFactorization gives the result itself, by its own substitution where it factorized the
 * matrix itself; what is solved a column at a time here is one of Eigen's factorizations, a
 * pivoted `Eigen::LDLT` or an `Eigen::LLT`.
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

/** Whether \a Solver is a lower or upper triangular view of a real matrix of a fixed size from 1
 * to columnCholeskySize, which solveInPlace() substitutes through itself.
 */
template <typename Solver>
inline constexpr bool substitutesTriangular = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Nested, unsigned int Mode>
inline constexpr bool substitutesTriangular<Eigen::TriangularView<Nested, Mode>> =
    (Mode == Eigen::Lower || Mode == Eigen::Upper) && Nested::RowsAtCompileTime != Eigen::Dynamic
    && Nested::RowsAtCompileTime >= 1 && Nested::RowsAtCompileTime <= columnCholeskySize
    && !Eigen::NumTraits<typename Nested::Scalar>::IsComplex;

/// \endcond

/** Overwrites \a matrix, \f$ B \f$, with \f$ T^{-1} B \f$ for the triangular view \a view,
 * \f$ T \f$, of a fixed size (substitutesTriangular), by substitution: unknown \f$ r \f$ of every
 * right-hand side together, from the unknowns before it in the order of the substitution, as
 * CholeskyFactorization::transposedSolve() does. Where the number of right-hand sides is fixed
 * and more than one, the rows of \f$ B \f$ are worked on as columns padded to whole packets, for
 * the same reason as there.
 */
template <typename Nested, unsigned int Mode, typename Derived>
void
substituteInPlace(Eigen::TriangularView<Nested, Mode> const & view, Eigen::MatrixBase<Derived> & matrix)
{
    using Scalar = typename Derived::Scalar;
    constexpr int size = Nested::RowsAtCompileTime;
    EMRYS_FILTER_ASSERT(matrix.rows() == size);
    auto const & t = view.nestedExpression();
    // The k-th row substituted: from the top for a lower triangle, from the bottom for an upper.
    auto const row = [](int k) {
        return Mode == Eigen::Lower ? k : size - 1 - k;
    };
    if constexpr (Derived::ColsAtCompileTime == 1)
    {
        for (int k = 0; k < size; ++k)
        {
            int const r = row(k);
            Scalar unknown = matrix.coeff(r, 0);
            for (int l = 0; l < k; ++l)
            {
                int const c = row(l);
                unknown -= t.coeff(r, c) * matrix.coeff(c, 0);
            }
            matrix.coeffRef(r, 0) = unknown / t.coeff(r, r);
        }
    }
    else if constexpr (Derived::ColsAtCompileTime != Eigen::Dynamic)
    {
        constexpr int count = Derived::ColsAtCompileTime;
        constexpr int padded = paddedSize<Scalar, count>;
        using Column = Eigen::Matrix<Scalar, padded, 1>;
        // Where columns are padded, the right-hand sides first, with the padding zeroed, in one pass,
        // as CholeskyFactorization::transposedSolve() takes them; otherwise not initialized, since
        // each column is written whole before the substitution reads it.
        Eigen::Matrix<Scalar, padded, size> work;
        if constexpr (padded != count)
        {
            work.template topRows<count>() = matrix.transpose();
            work.template bottomRows<padded - count>().setZero();
        }
        for (int k = 0; k < size; ++k)
        {
            int const r = row(k);
            Column unknown;
            if constexpr (padded != count)
            {
                unknown = work.col(r);
            }
            else
            {
                unknown = matrix.row(r).transpose();
            }
            for (int l = 0; l < k; ++l)
            {
                int const c = row(l);
                unknown -= t.coeff(r, c) * work.col(c);
            }
            work.col(r) = unknown / t.coeff(r, r);
        }
        matrix = work.template topRows<count>().transpose();
    }
    else
    {
        for (int k = 0; k < size; ++k)
        {
            int const r = row(k);
            for (int l = 0; l < k; ++l)
            {
                int const c = row(l);
                matrix.row(r) -= t.coeff(r, c) * matrix.row(c);
            }
            matrix.row(r) /= t.coeff(r, r);
        }
    }
}

/** Overwrites \a matrix, \f$ B \f$, with \f$ A^{-1} B \f$, for \a solver, anything with a
 * `solveInPlace()` for \f$ A \f$: an `Eigen::LLT` or `Eigen::LDLT`, or a triangular view.
 *
 * A triangular view of a real matrix of a fixed size of at most columnCholeskySize is
 * substituted through here (substituteInPlace()), as a CholeskyFactorization substitutes through
 * its own factor: Eigen's unrolled solve of a small fixed size is slow, as transposedSolve() says.
 * Otherwise, as in transposedSolve(), where \f$ B \f$ has a fixed number of rows of at most
 * unrolledSolveSize and more than one column, each column is solved alone, so that Eigen unrolls
 * it rather than calling its blocked solver; any other shape is solved whole. An inverse is this
 * on the identity.
 */
template <typename Solver, typename Derived>
void
solveInPlace(Solver const & solver, Eigen::MatrixBase<Derived> & matrix)
{
    constexpr int size = Derived::RowsAtCompileTime;
    if constexpr (substitutesTriangular<Solver>)
    {
        substituteInPlace(solver, matrix);
    }
    else if constexpr (size != Eigen::Dynamic && size <= unrolledSolveSize && Derived::ColsAtCompileTime != 1)
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

/** \returns \f$ A^{-1} B \f$ for \a solver and the right-hand side \a rhs, \f$ B \f$, by
 * solveInPlace() on a copy of \a rhs.
 */
template <typename Solver, typename Rhs>
[[nodiscard]] typename Rhs::PlainObject
solved(Solver const & solver, Eigen::MatrixBase<Rhs> const & rhs)
{
    typename Rhs::PlainObject result = rhs;
    solveInPlace(solver, result);
    return result;
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
