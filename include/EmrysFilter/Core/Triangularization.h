#pragma once

/** \file
 * Triangularization: the triangular factor of a stacked matrix by Householder reflections, in
 * place and without forming the orthogonal factor. The square-root forms of the %Gaussian
 * (decisions.md [gaussian-forms]) use it wherever the covariance form adds matrices: a sum
 * \f$ A A^T + B B^T \f$ is \f$ L L^T \f$ for the triangular factor \f$ L \f$ of
 * \f$ [\,A \;\; B\,] \f$, so no covariance is formed. Read Multiply.h next.
 */

#include <algorithm>
#include <concepts>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Householder>

#include <EmrysFilter/Core/Macros.h>

namespace Emrys
{

/** \ingroup Core
 * Reduces \a m to upper-triangular form in place by
 * [Householder reflections](https://en.wikipedia.org/wiki/Householder_transformation) from the
 * left, with a nonnegative diagonal.
 *
 * On return \a m holds \f$ Q^T M \f$ for the input \f$ M \f$ and an orthogonal \f$ Q \f$ that is
 * never formed, with zeros below the diagonal. Since \f$ Q \f$ is orthogonal,
 * \f$ (Q^T M)^T (Q^T M) = M^T M \f$: the leading square block \f$ R \f$ of the result is an
 * upper-triangular factor of \f$ M^T M \f$, which is its Cholesky factor when the diagonal is
 * positive, without \f$ M^T M \f$ ever being formed. Columns appended to \f$ M \f$ are carried
 * along: triangularizing \f$ [\,A \;\; b\,] \f$ gives \f$ [\,R \;\; Q^T b\,] \f$, which is how the
 * square-root information form keeps its vector beside its factor. A rank-deficient \f$ M \f$
 * gives zeros on the diagonal rather than failing.
 *
 * This is Householder QR (\ref Golub2013 "Golub and Van Loan (2013)", as given in
 * [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections)),
 * with the reflections applied as they are made and then discarded, so the work is in \a m
 * and a row of scratch the width of \a m, sized by its compile-time bound: a matrix with a
 * bounded size allocates nothing [allocation]. It departs from the algorithm only in flipping the
 * sign of each row whose diagonal element is negative, which keeps \f$ R^T R \f$ and makes the
 * factor of a full-rank \f$ M \f$ unique.
 *
 * \a m may be any writable Eigen matrix or expression, such as a block of a larger buffer passed
 * as a temporary; a <tt>const</tt> one is rejected rather than written through.
 */
template <typename M, typename Derived = std::remove_cvref_t<M>>
    requires std::derived_from<Derived, Eigen::MatrixBase<Derived>> && (!std::is_const_v<std::remove_reference_t<M>>)
          && ((Derived::Flags & Eigen::LvalueBit) != 0)
void
triangularize(M && m)
{
    Derived & matrix = m;
    using Scalar = typename Derived::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Workspace =
        Eigen::Matrix<Scalar, Derived::ColsAtCompileTime, 1, Eigen::ColMajor, Derived::MaxColsAtCompileTime, 1>;
    Eigen::Index const rows = matrix.rows();
    Eigen::Index const cols = matrix.cols();
    Workspace workspace(cols);
    for (Eigen::Index k = 0; k < std::min(rows, cols); ++k)
    {
        Eigen::Index const below = rows - k - 1;
        Eigen::Index const right = cols - k - 1;
        if (below > 0)
        {
            auto tau = Scalar{};
            auto beta = RealScalar{};
            matrix.col(k).tail(below + 1).makeHouseholderInPlace(tau, beta);
            if (right > 0)
            {
                matrix.bottomRightCorner(below + 1, right)
                    .applyHouseholderOnTheLeft(matrix.col(k).tail(below), tau, workspace.data());
            }
            matrix(k, k) = beta;
            matrix.col(k).tail(below).setZero();
        }
        if (matrix(k, k) < Scalar(0))
        {
            matrix.row(k).tail(cols - k) *= Scalar(-1);
        }
    }
}

namespace internal
{

/** \returns the lower-triangular \f$ L \f$, square with \a a's rows, with
 * \f$ L L^T = A A^T \f$ for the wide (or square) \a a: the transpose of the triangular factor of
 * \f$ A^T \f$ (triangularize()). With fewer columns than rows the trailing columns of \f$ L \f$
 * are zero. Copies \a a into a buffer of its transposed size, so a bounded \a a allocates nothing.
 */
template <typename Derived>
[[nodiscard]] Eigen::Matrix<
    typename Derived::Scalar,
    Derived::RowsAtCompileTime,
    Derived::RowsAtCompileTime,
    Eigen::ColMajor,
    Derived::MaxRowsAtCompileTime,
    Derived::MaxRowsAtCompileTime>
lowerTriangularRoot(Eigen::MatrixBase<Derived> const & a)
{
    using Scalar = typename Derived::Scalar;
    using Transposed = Eigen::Matrix<
        Scalar,
        Derived::ColsAtCompileTime,
        Derived::RowsAtCompileTime,
        Eigen::ColMajor,
        Derived::MaxColsAtCompileTime,
        Derived::MaxRowsAtCompileTime>;
    using Result = Eigen::Matrix<
        Scalar,
        Derived::RowsAtCompileTime,
        Derived::RowsAtCompileTime,
        Eigen::ColMajor,
        Derived::MaxRowsAtCompileTime,
        Derived::MaxRowsAtCompileTime>;
    Transposed buffer = a.transpose();
    triangularize(buffer);
    Eigen::Index const n = a.rows();
    Eigen::Index const k = std::min(a.rows(), a.cols());
    Result factor = Result::Zero(n, n);
    factor.leftCols(k) = buffer.topRows(k).transpose();
    return factor;
}

}  // namespace internal

}  // namespace Emrys
