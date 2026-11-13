#pragma once

/** \file
 * Products of fixed-size matrices by Eigen's fixed-size product at every size: multiply() is
 * \f$ A B \f$ by Eigen's coefficient-based product wherever both operands have a fixed size, which
 * Eigen itself chooses only while no dimension reaches eight, and asTriangular() is a triangular
 * factor as a dense matrix to multiply by, which a triangular view never is. Everything here is
 * in `internal`. Read Solve.h next.
 */

#include <concepts>

#include <Eigen/Core>

namespace Emrys::internal
{

/** The fixed dimension from which Eigen counts a product as large and sends it to its run-time
 * matrix-vector or blocked matrix product, `EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD`.
 */
inline constexpr int largeProductSize = EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD;

/** Whether a product of a \a rows by \a depth matrix and a \a depth by \a cols one, each size fixed
 * at compile time or `Eigen::Dynamic`, has the shape that isLargeFixedProduct describes.
 */
[[nodiscard]] consteval bool
isLargeFixedShape(int rows, int depth, int cols)
{
    if (rows == Eigen::Dynamic || depth == Eigen::Dynamic || cols == Eigen::Dynamic)
    {
        return false;
    }
    return depth > 1 && !(rows == 1 && cols == 1)
        && (depth >= largeProductSize || (rows >= largeProductSize && cols >= largeProductSize));
}

/** Whether the product of \a Lhs and \a Rhs has every dimension fixed at compile time and a
 * shape that Eigen sends to its run-time matrix-vector or blocked matrix product: an inner
 * dimension of at least largeProductSize, or two outer ones of at least that, and neither an
 * inner product (one row by one column) nor an outer product (an inner dimension of one), whose
 * own kernels keep the sizes fixed (Eigen's `product_type_selector`).
 */
template <typename Lhs, typename Rhs>
inline constexpr bool isLargeFixedProduct =
    isLargeFixedShape(Lhs::RowsAtCompileTime, Lhs::ColsAtCompileTime, Rhs::ColsAtCompileTime);

/** \returns the product \f$ A B \f$ of \a lhs and \a rhs, Eigen expressions such as dense
 * matrices or triangular views, as an expression to assign.
 *
 * Eigen's product picks its kernel by the operands' sizes at compile time: while every dimension
 * is fixed and below largeProductSize it multiplies coefficient by coefficient with the sizes
 * fixed, and from there it calls a matrix-vector or blocked matrix kernel whose loops are sized at
 * run time, fixed sizes or not. Where both are dense and every dimension is fixed this is Eigen's
 * `lazyProduct()`, its fixed-size product, whatever the size; anything else, a triangular view
 * among them (see asTriangular()), is the product `*` makes.
 *
 * Like `lazyProduct()`, and unlike `*`, the product of two fixed operands is not evaluated into a
 * temporary first, so it must not be assigned to a matrix it reads.
 */
template <typename Lhs, typename Rhs>
[[nodiscard]] auto
multiply(Eigen::EigenBase<Lhs> const & lhs, Eigen::EigenBase<Rhs> const & rhs)
{
    if constexpr (std::derived_from<Lhs, Eigen::MatrixBase<Lhs>> && std::derived_from<Rhs, Eigen::MatrixBase<Rhs>>
                  && isLargeFixedProduct<Lhs, Rhs>)
    {
        return lhs.derived().lazyProduct(rhs.derived());
    }
    else
    {
        return lhs.derived() * rhs.derived();
    }
}

/** \returns the \a Mode triangle of \a matrix (`Eigen::Lower` or `Eigen::Upper`), to multiply
 * by: a copy with zeros in the other triangle where \a matrix has a fixed size, and its triangular
 * view otherwise.
 *
 * Eigen sends a product with a triangular view to an out-of-line matrix-vector kernel or to its
 * blocked matrix product whatever the size, while a product of fixed-size dense matrices has its
 * fixed-size path (multiply()); the zeros of the copy change no value. \a matrix must outlive the
 * product, which the view refers to.
 */
template <int Mode, typename Derived>
[[nodiscard]] auto
asTriangular(Eigen::MatrixBase<Derived> const & matrix)
{
    if constexpr (Derived::RowsAtCompileTime != Eigen::Dynamic && Derived::ColsAtCompileTime != Eigen::Dynamic)
    {
        typename Derived::PlainObject result = matrix.template triangularView<Mode>();
        return result;
    }
    else
    {
        return matrix.template triangularView<Mode>();
    }
}

}  // namespace Emrys::internal
