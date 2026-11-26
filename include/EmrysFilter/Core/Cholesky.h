#pragma once

/** \file
 * The Cholesky factorization the library takes its gains, the quadratic forms of an inverse,
 * and log-determinants from: CholeskyFactorization, which factorizes a real matrix of a small
 * fixed size itself, a whole column at a time and without square roots, and any other matrix by
 * `Eigen::LLT` (decisions.md [own-cholesky]). The constants and traits beside it are in `internal`.
 * Read Solve.h next.
 */

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <type_traits>

#include <EmrysFilter/Core/Macros.h>

namespace Emrys::internal
{

/** The largest fixed size that CholeskyFactorization factorizes and solves itself: the largest at
 * which its factorization and its substitution were measured against Eigen's [benchmarks], and
 * took less time at every size up to it.
 */
inline constexpr int columnCholeskySize = 24;

/** The number of \a Scalar in 16 bytes, the smallest packet Eigen vectorizes with, or 1 where it
 * does not vectorize \a Scalar. A column of a multiple of this many scalars in a fixed-size matrix
 * starts on a packet boundary, wherever it lies in the matrix.
 */
template <typename Scalar>
inline constexpr int smallestPacketSize =
    std::is_arithmetic_v<Scalar> && EIGEN_MAX_STATIC_ALIGN_BYTES >= 16 && sizeof(Scalar) <= 16
        ? int(16 / sizeof(Scalar))
        : 1;

/** \a Count rounded up to a multiple of smallestPacketSize: the length of a column of \a Count
 * scalars padded to whole packets.
 */
template <typename Scalar, int Count>
inline constexpr int paddedSize =
    (Count + smallestPacketSize<Scalar> - 1) / smallestPacketSize<Scalar> * smallestPacketSize<Scalar>;

/** Whether CholeskyFactorization factorizes a \a Matrix itself: a real matrix of a fixed size
 * from 1 to columnCholeskySize.
 */
template <typename Matrix>
inline constexpr bool factorizesByColumns =
    Matrix::RowsAtCompileTime != Eigen::Dynamic && Matrix::RowsAtCompileTime >= 1
    && Matrix::RowsAtCompileTime <= columnCholeskySize && !Eigen::NumTraits<typename Matrix::Scalar>::IsComplex;

}  // namespace Emrys::internal

namespace Emrys
{

/** \ingroup Core
 * \brief The Cholesky factorization of a symmetric positive definite \a Matrix \f$ A \f$, of which
 * only the lower triangle's values are used (decisions.md [own-cholesky]).
 *
 * It is what a %Gaussian keeps of its covariance (Gaussian::cholesky()) and what an update takes
 * its gain from.
 *
 * A real matrix of a fixed size of at most internal::columnCholeskySize
 * (internal::factorizesByColumns) is factorized
 * here, in the form without square roots, \f$ A = \tilde L D \tilde L^T \f$ with
 * \f$ \tilde L \f$ unit lower triangular and \f$ D \f$ the diagonal matrix of the pivots, so that
 * the Cholesky factor is \f$ L = \tilde L D^{1/2} \f$
 * (\ref Golub2013 "Golub and Van Loan (2013)", as given in
 * [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition)).
 * It is not Eigen's `LDLT`, which pivots and accepts a semidefinite matrix: the pivots here are
 * the squares of the diagonal of \f$ L \f$, taken in order, and a pivot that is not greater than
 * zero ends the factorization as it ends a Cholesky factorization.
 *
 * Column \f$ k \f$ of \f$ \tilde L D \f$ is column \f$ k \f$ of \f$ A \f$ less
 * \f$ \tilde L_{kj} \f$ times each column \f$ j \f$ before it, and its element on the diagonal
 * is the pivot. Three things about how that is done are for speed at these sizes [benchmarks]:
 *
 * - A filter's step waits on its factorization, and each column waits on the one before. With
 *   square roots a column waits on a root and then a division; here it waits on one division.
 *   The substitutions then scale nothing inside a pass: transposedSolve() divides by the pivots
 *   once, between its forward and its backward pass.
 * - Every operation is on a whole column, padded with zeros to a multiple of
 *   internal::smallestPacketSize,
 *   so the elements above the diagonal are loaded and computed with too. Nothing reads what that
 *   leaves there, so the upper triangle of the matrix given may hold any values, but it must be
 *   initialized.
 * - No norm is taken for a condition estimate, which `Eigen::LLT` takes whether or not one is
 *   asked for.
 *
 * Any other matrix, of a dynamic or a larger size or complex, is factorized by an `Eigen::LLT`
 * that this holds, and solved by it. The results of the two differ by rounding.
 *
 * A matrix is positive definite here when every pivot is greater than zero, so one with a NaN in
 * its lower triangle is not, at any size, where `Eigen::LLT` alone reports success for it.
 */
template <typename Matrix>
class CholeskyFactorization
{
public:

    using MatrixType = Matrix;               ///< the matrix factorized
    using Scalar = typename Matrix::Scalar;  ///< its scalar

    /** Makes a factorization of nothing, for compute() to fill; isPositiveDefinite() is false. */
    CholeskyFactorization() = default;

    /** Factorizes \a matrix, as compute() does. */
    template <typename Derived>
    explicit CholeskyFactorization(Eigen::MatrixBase<Derived> const & matrix)
    {
        compute(matrix);
    }

    /** Factorizes \a matrix, of which only the lower triangle's values are used. The upper
     * triangle is loaded all the same, so it must be initialized. Afterward isPositiveDefinite()
     * says whether \a matrix was, and the members that read the factorization may be called only
     * if it was.
     */
    template <typename Derived>
    void
    compute(Eigen::MatrixBase<Derived> const & matrix)
    {
        EMRYS_FILTER_ASSERT(matrix.rows() == matrix.cols());
        if constexpr (!ByColumns)
        {
            using Real = typename Eigen::NumTraits<Scalar>::Real;
            m_storage.compute(matrix);
            m_positiveDefinite =
                m_storage.info() == Eigen::Success && (m_storage.matrixLLT().diagonal().real().array() > Real(0)).all();
        }
        else if constexpr ((Eigen::internal::traits<Derived>::Flags & Eigen::DirectAccessBit) != 0)
        {
            factorize(matrix.derived());
        }
        else
        {
            // An expression is evaluated once, not once for each column read from it.
            Matrix const evaluated = matrix;
            factorize(evaluated);
        }
    }

    /** \returns whether the matrix given to compute() was positive definite: every pivot greater
     *           than zero
     */
    [[nodiscard]] bool
    isPositiveDefinite() const
    {
        return m_positiveDefinite;
    }

    /** \returns the size of the matrix factorized */
    [[nodiscard]] Eigen::Index
    rows() const
    {
        if constexpr (ByColumns)
        {
            return Matrix::RowsAtCompileTime;
        }
        else
        {
            return m_storage.rows();
        }
    }

    /** \returns the Cholesky factor \f$ L \f$, lower triangular with zeros above its diagonal.
     *           Where the matrix was factorized here it is made at each call, as
     *           \f$ \tilde L D^{1/2} \f$.
     */
    [[nodiscard]] Matrix
    lower() const
    {
        EMRYS_FILTER_ASSERT(m_positiveDefinite && "the matrix was not positive definite, or none was factorized");
        if constexpr (ByColumns)
        {
            constexpr int size = Matrix::RowsAtCompileTime;
            Matrix result = Matrix::Zero();
            for (int j = 0; j < size; ++j)
            {
                Scalar const root = Eigen::numext::sqrt(m_storage.pivot(j));
                result(j, j) = root;
                for (int i = j + 1; i < size; ++i)
                {
                    result(i, j) = m_storage.unit(i, j) * root;
                }
            }
            return result;
        }
        else
        {
            Matrix result = m_storage.matrixL();
            return result;
        }
    }

    /** \returns \f$ \log |A| \f$: the sum of the logarithms of the pivots, which is
     *           \f$ 2 \sum_i \log L_{ii} \f$
     */
    [[nodiscard]] Scalar
    logDeterminant() const
    {
        EMRYS_FILTER_ASSERT(m_positiveDefinite && "the matrix was not positive definite, or none was factorized");
        if constexpr (ByColumns)
        {
            return m_storage.pivot.array().log().sum();
        }
        else
        {
            return Scalar(2) * m_storage.matrixLLT().diagonal().array().log().sum();
        }
    }

    /** \returns \f$ x^T A^{-1} x \f$ for the vector \a x, which is \f$ |L^{-1} x|^2 \f$: one
     *           forward substitution where a solve takes two. It is the squared Mahalanobis
     *           distance of a difference \a x under the covariance \f$ A \f$.
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    inverseQuadraticForm(Eigen::MatrixBase<Derived> const & x) const
    {
        EMRYS_FILTER_ASSERT(m_positiveDefinite && "the matrix was not positive definite, or none was factorized");
        EMRYS_FILTER_ASSERT(x.cols() == 1 && x.rows() == rows());
        if constexpr (ByColumns)
        {
            // y = L~^-1 x, and the form is y^T D^-1 y.
            constexpr int size = Matrix::RowsAtCompileTime;
            Eigen::Matrix<Scalar, size, 1> y = x;
            Scalar result = Scalar(0);
            for (int r = 0; r < size; ++r)
            {
                Scalar unknown = y(r);
                for (int c = 0; c < r; ++c)
                {
                    unknown -= m_storage.unit(r, c) * y(c);
                }
                y(r) = unknown;
                result += unknown * unknown * m_storage.reciprocal(r);
            }
            return result;
        }
        else
        {
            typename Derived::PlainObject whitened = x;
            m_storage.matrixL().solveInPlace(whitened);
            return whitened.squaredNorm();
        }
    }

    /** \returns \f$ (A^{-1} B)^T \f$ as a \a Result, for the right-hand side \a rhs, \f$ B \f$:
     *           the shape of a Kalman gain.
     *
     * Where the matrix was factorized here, it is a forward substitution through
     * \f$ \tilde L \f$, a division by the pivots, as a multiplication by their reciprocals, and
     * a backward substitution through \f$ \tilde L^T \f$, every right-hand side together: column
     * \f$ r \f$ of the result holds unknown \f$ r \f$ of each right-hand side, so each step is one
     * operation on whole columns. Where the number of right-hand sides is fixed, the columns are
     * padded with zeros to a multiple of internal::smallestPacketSize: Eigen vectorizes a column that is
     * not, six `float`s say, from wherever its address lets a packet start, so one column's
     * stores are not the next operation's loads, and the processor waits on each [benchmarks].
     * Otherwise it is Eigen's solve.
     */
    template <typename Result, typename Rhs>
    [[nodiscard]] Result
    transposedSolve(Eigen::MatrixBase<Rhs> const & rhs) const
    {
        EMRYS_FILTER_ASSERT(m_positiveDefinite && "the matrix was not positive definite, or none was factorized");
        if constexpr (!ByColumns)
        {
            Result result = m_storage.solve(rhs).transpose();
            return result;
        }
        else if constexpr (Result::RowsAtCompileTime != Eigen::Dynamic)
        {
            constexpr int size = Matrix::RowsAtCompileTime;
            constexpr int count = Result::RowsAtCompileTime;
            using Column = Eigen::Matrix<Scalar, internal::paddedSize<Scalar, count>, 1>;
            // Not initialized, since each column is written whole before a pass reads it.
            Eigen::Matrix<Scalar, internal::paddedSize<Scalar, count>, size> work;
            // L~ Y = B: unknown r of every right-hand side from those before it.
            for (int r = 0; r < size; ++r)
            {
                Column unknown = Column::Zero();
                unknown.template head<count>() = rhs.row(r).transpose();
                for (int c = 0; c < r; ++c)
                {
                    unknown -= m_storage.unit(r, c) * work.col(c);
                }
                work.col(r) = unknown;
            }
            // L~^T X = D^-1 Y: unknown r of every right-hand side from those after it.
            for (int r = size - 1; r >= 0; --r)
            {
                Column unknown = work.col(r) * m_storage.reciprocal(r);
                for (int c = r + 1; c < size; ++c)
                {
                    unknown -= m_storage.unit(c, r) * work.col(c);
                }
                work.col(r) = unknown;
            }
            Result result = work.template topRows<count>();
            return result;
        }
        else
        {
            // The same two passes in the result itself, which no fixed padding fits.
            constexpr int size = Matrix::RowsAtCompileTime;
            auto result = Result{};
            result.resize(rhs.cols(), rhs.rows());
            for (int r = 0; r < size; ++r)
            {
                result.col(r) = rhs.row(r).transpose();
                for (int c = 0; c < r; ++c)
                {
                    result.col(r) -= m_storage.unit(r, c) * result.col(c);
                }
            }
            for (int r = size - 1; r >= 0; --r)
            {
                result.col(r) *= m_storage.reciprocal(r);
                for (int c = r + 1; c < size; ++c)
                {
                    result.col(r) -= m_storage.unit(c, r) * result.col(c);
                }
            }
            return result;
        }
    }

    /** Overwrites \a matrix, \f$ B \f$, with \f$ A^{-1} B \f$: the transpose of
     * transposedSolve() where the matrix was factorized here, and Eigen's solve otherwise.
     */
    template <typename Derived>
    void
    solveInPlace(Eigen::MatrixBase<Derived> & matrix) const
    {
        if constexpr (ByColumns)
        {
            // Eigen stores a matrix of one row by rows.
            constexpr int options = Derived::MaxColsAtCompileTime == 1 && Derived::MaxRowsAtCompileTime != 1
                                      ? Eigen::RowMajor
                                      : Eigen::ColMajor;
            using Transposed = Eigen::Matrix<
                Scalar,
                Derived::ColsAtCompileTime,
                Derived::RowsAtCompileTime,
                options,
                Derived::MaxColsAtCompileTime,
                Derived::MaxRowsAtCompileTime>;
            matrix.derived() = transposedSolve<Transposed>(matrix).transpose();
        }
        else
        {
            EMRYS_FILTER_ASSERT(m_positiveDefinite && "the matrix was not positive definite, or none was factorized");
            m_storage.solveInPlace(matrix);
        }
    }

    /** \returns \f$ A^{-1} B \f$ for the right-hand side \a rhs, \f$ B \f$, as solveInPlace()
     *           gives it
     */
    template <typename Rhs>
    [[nodiscard]] typename Rhs::PlainObject
    solve(Eigen::MatrixBase<Rhs> const & rhs) const
    {
        typename Rhs::PlainObject result = rhs;
        solveInPlace(result);
        return result;
    }

    /** Overwrites \a matrix, \f$ B \f$, with \f$ L^{-1} B \f$, whose columns are those of
     * \f$ B \f$ whitened: \f$ (L^{-1} B)^T (L^{-1} B) = B^T A^{-1} B \f$. Where the matrix was
     * factorized here, each column is substituted through \f$ \tilde L \f$ and divided by the
     * square roots of the pivots, which are taken at each call; inverseQuadraticForm() gives the
     * squared length of one whitened vector without them. It divides by the roots that lower()
     * multiplies by, so whitening lower() gives the identity exactly, and a divergence of a
     * %Gaussian from itself is exactly zero.
     */
    template <typename Derived>
    void
    whitenInPlace(Eigen::MatrixBase<Derived> & matrix) const
    {
        EMRYS_FILTER_ASSERT(m_positiveDefinite && "the matrix was not positive definite, or none was factorized");
        EMRYS_FILTER_ASSERT(matrix.rows() == rows());
        if constexpr (ByColumns)
        {
            constexpr int size = Matrix::RowsAtCompileTime;
            Eigen::Matrix<Scalar, size, 1> const root = m_storage.pivot.cwiseSqrt();
            auto & whitened = matrix.derived();
            for (Eigen::Index j = 0; j < whitened.cols(); ++j)
            {
                // L~ y = b in place, then y over the roots of D.
                for (int r = 1; r < size; ++r)
                {
                    Scalar unknown = whitened(r, j);
                    for (int c = 0; c < r; ++c)
                    {
                        unknown -= m_storage.unit(r, c) * whitened(c, j);
                    }
                    whitened(r, j) = unknown;
                }
                for (int r = 0; r < size; ++r)
                {
                    whitened(r, j) /= root(r);
                }
            }
        }
        else
        {
            m_storage.matrixL().solveInPlace(matrix);
        }
    }

private:

    /// whether this factorizes the matrix itself
    static constexpr bool ByColumns = internal::factorizesByColumns<Matrix>;

    /** What a factorization made here keeps. */
    struct Columns
    {
        /// a vector of the matrix's size
        using Diagonal = Eigen::Matrix<Scalar, Matrix::RowsAtCompileTime, 1>;

        /// \f$ \tilde L \f$ below its diagonal, in the padded columns the factorization wrote, so
        /// that the substitutions, which a filter's step waits on, wait on no copy of it; the rest
        /// is unspecified
        Eigen::Matrix<Scalar, internal::paddedSize<Scalar, Matrix::RowsAtCompileTime>, Matrix::RowsAtCompileTime> unit;

        Diagonal pivot;       ///< the pivots, the diagonal of \f$ D \f$
        Diagonal reciprocal;  ///< the reciprocals of the pivots
    };

    /** Factorizes \a matrix, which has direct access to its coefficients, by whole columns. */
    template <typename Derived>
    void
    factorize(Derived const & matrix)
    {
        constexpr int size = Matrix::RowsAtCompileTime;
        using Column = Eigen::Matrix<Scalar, internal::paddedSize<Scalar, size>, 1>;
        // The columns of L~ D, which the columns after them are reduced by. Not initialized, since
        // column k is written whole before a column after it reads it: an initialized one, of
        // 4.6 KB at a size of 24 in double, was zeroed at every factorization.
        Eigen::Matrix<Scalar, internal::paddedSize<Scalar, size>, size> unscaled;
        m_positiveDefinite = false;
        for (int k = 0; k < size; ++k)
        {
            Column column = Column::Zero();
            column.template head<size>() = matrix.col(k);
            for (int j = 0; j < k; ++j)
            {
                // L~(k, j), from the scalars it waits on, not from the column stored below.
                column -= (unscaled(k, j) * m_storage.reciprocal(j)) * unscaled.col(j);
            }
            Scalar const pivot = column(k);
            if (!(pivot > Scalar(0)))
            {
                return;
            }
            Scalar const reciprocal = Scalar(1) / pivot;
            m_storage.pivot(k) = pivot;
            m_storage.reciprocal(k) = reciprocal;
            unscaled.col(k) = column;
            m_storage.unit.col(k) = column * reciprocal;
        }
        m_positiveDefinite = true;
    }

    /// the unit factor and the pivots, or Eigen's factorization
    std::conditional_t<ByColumns, Columns, Eigen::LLT<Matrix>> m_storage;

    bool m_positiveDefinite = false;  ///< whether a positive definite matrix was factorized
};

}  // namespace Emrys
