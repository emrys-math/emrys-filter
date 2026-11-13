#pragma once

/** \file
 * How a function that stores a matrix argument takes it (decisions.md [forwarded-matrices]): the concept
 * MatrixExpression, which constrains its forwarding reference so that an rvalue is
 * moved, and internal::castScalar(), which forwards a matrix that already has the scalar
 * rather than copying it; and internal::forwardLike(), std::forward_like for every standard
 * library. Read Allocation.h next.
 */

#include <concepts>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

namespace Emrys
{

namespace internal
{

/** The CRTP type of a dense matrix or matrix expression \a T, however qualified, from derived(),
 * so that a <tt>MatrixBase<Derived></tt> passed on from a parameter of that type has it too.
 */
template <typename T>
using CrtpDerived = std::remove_cvref_t<decltype(std::declval<std::remove_cvref_t<T> &>().derived())>;

}  // namespace internal

/** \ingroup Core
 * A dense matrix or matrix expression, possibly cv- or reference-qualified: what a
 * <tt>const Eigen::MatrixBase<Derived>&</tt> parameter accepts. It constrains the forwarding
 * references of functions that store their argument, so that a plain matrix passed as an rvalue
 * is moved rather than copied (decisions.md [forwarded-matrices]); a user's function that stores
 * a matrix takes one the same way.
 */
template <typename T>
concept MatrixExpression = requires { typename internal::CrtpDerived<T>; }
                        && std::derived_from<std::remove_cvref_t<T>, Eigen::MatrixBase<internal::CrtpDerived<T>>>;

/** \ingroup Core
 * A diagonal matrix or diagonal expression, possibly cv- or reference-qualified, such as
 * <tt>v.asDiagonal()</tt> or an <tt>Eigen::DiagonalMatrix</tt>: what a
 * <tt>const Eigen::DiagonalBase<Derived>&</tt> parameter accepts.
 */
template <typename T>
concept DiagonalExpression = std::derived_from<std::remove_cvref_t<T>, Eigen::DiagonalBase<std::remove_cvref_t<T>>>;

/** \ingroup Core
 * A covariance as a function that stores one takes it: a MatrixExpression, or a
 * DiagonalExpression, whose diagonal holds the variances and whose other entries are zero, so that
 * <tt>GaussianParams<State>{mean, variances.asDiagonal()}</tt> needs no <tt>toDenseMatrix()</tt>.
 */
template <typename T>
concept CovarianceExpression = MatrixExpression<T> || DiagonalExpression<T>;

namespace internal
{

/** \a m cast to \a Scalar, or \a m itself, forwarded, if it already has that scalar: Eigen's
 * cast() to the same scalar returns a reference, which a sink would copy rather than move.
 */
template <typename Scalar, MatrixExpression M>
[[nodiscard]] decltype(auto)
castScalar(M && m)
{
    if constexpr (std::is_same_v<typename std::remove_cvref_t<M>::Scalar, Scalar>)
    {
        return std::forward<M>(m);
    }
    else
    {
        return m.template cast<Scalar>();
    }
}

/** \a U's referent, const if \a T's is. */
template <typename T, typename U>
using ConstLike = std::conditional_t<
    std::is_const_v<std::remove_reference_t<T>>,
    std::remove_reference_t<U> const,
    std::remove_reference_t<U>>;

/** ConstLike referenced as \a T is: what std::forward_like returns. */
template <typename T, typename U>
using ForwardedLike = std::conditional_t<std::is_lvalue_reference_v<T>, ConstLike<T, U> &, ConstLike<T, U> &&>;

/** std::forward_like with its return type spelled out: libstdc++ 14.2 deduces it, and Clang 21
 * then rejects a call it reaches before instantiating the definition.
 */
template <typename T, typename U>
[[nodiscard]] constexpr ForwardedLike<T, U>
forwardLike(U && u) noexcept
{
    return static_cast<ForwardedLike<T, U>>(u);
}

}  // namespace internal

}  // namespace Emrys
