#pragma once

/** \file
 * Mathematical constants and scalar functions that work in every scalar type the library
 * accepts, where the standard library and Eigen serve only some of them. Library code calls these
 * rather than their \c std:: or \c Eigen::numext:: counterparts, and they are public so that a
 * user's function evaluated in such a scalar, such as a measurement under \c methods::linearized(), can
 * too: pi(), log2Pi(), logGamma(), logOnePlus(), isFinite(), isNaN(), and allFinite().
 *
 * <b>Why this header exists.</b> Every distribution, model, and filter takes a \c Scalar, and
 * RealScalar admits any real number type, not only \c float, \c double, and <tt>long
 * double</tt>. Two kinds of custom scalar matter:
 *
 * - <b>Automatic differentiation.</b> \c methods::linearized(), for a function that does not return its Jacobian,
 * evaluates the user's function in Eigen's \c AutoDiffScalar (Transform.h), so a function that builds or evaluates a
 * library density inside it, such as a robust measurement's log-likelihood, runs the library in that scalar.
 * Differentiating a filter's log-likelihood with respect to its noise parameters, to tune or learn them, would run the
 * whole library in it; that is a goal this header serves, not something the tests yet show.
 * - <b>Multiprecision.</b> A type such as MPFR's \c mpreal gives a reference to check the
 *   double-precision results against.
 *
 * In such a scalar the obvious code fails, mostly at compile time, sometimes silently:
 *
 * - \c std::numbers::pi_v<Scalar> is ill-formed for anything but a floating-point type
 *   ([math.constants]).
 * - \c EIGEN_PI is a macro (the project uses none) and a <tt>long double</tt> literal, so
 *   <tt>Scalar(EIGEN_PI)</tt> compiles for a multiprecision scalar but rounds π to 64 bits, or
 *   to 53 on MSVC, where <tt>long double</tt> is \c double. Every normalizer then carries that
 *   error with nothing to show for it.
 * - \c Eigen::numext::lgamma has implementations for \c float and \c double only; anything
 *   else, <tt>long double</tt> included, is a static assertion.
 * - \c AutoDiffScalar has no \c lgamma, \c log1p, \c floor, \c isfinite, or \c isnan, so the \c numext
 *   versions, \c std versions found through a <tt>using std::f;</tt> line, and Eigen's own
 *   <tt>DenseBase::allFinite()</tt> fail to compile for it.
 * - Arithmetic on \c AutoDiffScalar values yields expression types
 *   (<tt>AutoDiffScalar<CwiseBinaryOp<...>></tt>). A \c numext function deduces its type from
 *   such an argument and then cannot convert its result back, so <tt>numext::log(a * b)</tt>
 *   fails where <tt>numext::log(Scalar(a * b))</tt> compiles.
 *
 * <b>What each function does.</b> Each dispatches at compile time to the standard function for
 * a floating-point scalar, to the chain rule on the value and the derivatives for an
 * \c AutoDiffScalar, and otherwise to Eigen's or the scalar's own function, which a
 * multiprecision type provides. Those returning a scalar take it as an explicit template argument
 * (<tt>logGamma<Scalar>(x)</tt>), not a deduced one, so that an autodiff expression converts to
 * the plain scalar first. For \c double they compile to the standard call.
 *
 * <b>Adding one.</b> When library code needs another function that the standard library, Eigen,
 * or \c AutoDiffScalar lacks for some scalar, it goes here with the same three branches, a test
 * of each branch in \c tests/Core/math.cpp, and, if a density uses it, a case in
 * \c tests/Distributions/autodiff_scalars.cpp, which evaluates the densities in
 * \c AutoDiffScalar and so catches the next gap. Functions that \c AutoDiffScalar and Eigen
 * both support (\c log, \c exp, \c sqrt, \c atan2) go through \c Eigen::numext::
 * directly, with a plain \a Scalar argument.
 *
 * Read Forwarding.h next.
 */

#include <cmath>
#include <concepts>
#include <numbers>
#include <type_traits>

#include <EmrysFilter/Core/Macros.h>

#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/SpecialFunctions>

namespace Emrys
{

/** \ingroup Core
 * \brief A type usable as the scalar of a distribution: a real number type such as \c float,
 * \c double, or an automatic-differentiation scalar.
 *
 * Means, covariances, factorizations, and log-weights need real arithmetic, so integer and
 * complex types are excluded. Integer-valued quantities are values in a discrete space, not
 * scalars.
 */
template <typename T>
concept RealScalar =
    !Eigen::NumTraits<T>::IsInteger && !Eigen::NumTraits<T>::IsComplex && requires(T const & a, T const & b) {
        a + b;
        a - b;
        a * b;
        a / b;
        a < b;
    };

namespace internal
{

/** Whether \a T is an \c Eigen::AutoDiffScalar, plain or an expression.
 *
 * The functions here special-case it because it is the autodiff scalar the library itself uses
 * (in \c methods::linearized()) and because it lacks functions other custom scalars provide. Another
 * autodiff type that defines its own \c lgamma, \c log1p, \c isfinite, and \c isnan takes the
 * general branch.
 */
template <typename T>
inline constexpr bool isAutoDiffScalar = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Derivatives>
inline constexpr bool isAutoDiffScalar<Eigen::AutoDiffScalar<Derivatives>>{true};

/// \endcond

/** The scalar deduced from the arithmetic values \a Values given for the parameters of an object
 * that holds them, such as a Gate or a Roughen: the common type of those that are floating
 * point, so that <tt>reductions::gate(0.997f)</tt> holds a \c float, or \c double
 * when every one is an integer, so that <tt>upkeep::roughen(rng, 0)</tt> holds a \c double as
 * <tt>upkeep::roughen(rng, 0.0)</tt> does. An integer stands for no type of its own, here as \c float, the
 * least precise standard floating type, which never raises the common type.
 */
template <typename... Values>
    requires(std::is_arithmetic_v<Values> && ...)
struct DeducedScalarOf
{
    /// The common floating type, or \c double
    using type = std::conditional_t<
        (std::integral<Values> && ...),
        double,
        std::common_type_t<std::conditional_t<std::integral<Values>, float, Values>...>>;
};

/** The scalar deduced from the arithmetic values \a Values (DeducedScalarOf). */
template <typename... Values>
using DeducedScalar = typename DeducedScalarOf<Values...>::type;

}  // namespace internal

/** \ingroup Core
 * \returns \f$ \pi \f$ as a \a Scalar.
 *
 * In place of <tt>std::numbers::pi_v<Scalar></tt>, which is ill-formed for a scalar that is
 * not floating point, and \c EIGEN_PI, which rounds through <tt>long double</tt>. A
 * floating-point \a Scalar takes \c std::numbers::pi_v. A scalar whose
 * <tt>Eigen::NumTraits<Scalar>::Literal</tt> is floating point, such as Eigen's
 * \c AutoDiffScalar, converts \c pi_v of that literal type, so its derivatives are zero. Any
 * other scalar, such as a multiprecision type whose literal is itself, computes
 * \f$ 4 \arctan 1 \f$ in its own precision through \c Eigen::numext::atan.
 */
template <typename Scalar>
[[nodiscard]] Scalar
pi()
{
    using Literal = typename Eigen::NumTraits<Scalar>::Literal;
    if constexpr (std::floating_point<Scalar>)
    {
        return std::numbers::pi_v<Scalar>;
    }
    else if constexpr (std::floating_point<Literal>)
    {
        return Scalar(std::numbers::pi_v<Literal>);
    }
    else
    {
        return Scalar(4) * Eigen::numext::atan(Scalar(1));
    }
}

/** \ingroup Core
 * \returns \f$ \log 2\pi \f$ as a \a Scalar, the constant in every %Gaussian normalizer.
 *
 * A floating-point \a Scalar takes a literal, correctly rounded for \c float, \c double, and an
 * 80-bit <tt>long double</tt>, so it is a constant on every compiler; <tt>std::log</tt> is not
 * \c constexpr before C++26, and <tt>std::log(2 * pi)</tt> would be one ulp low in \c double,
 * since \f$ \pi \f$ is rounded before the log. A scalar whose
 * <tt>Eigen::NumTraits<Scalar>::Literal</tt> is floating point, such as Eigen's
 * \c AutoDiffScalar, converts that literal type's, so its derivatives are zero. Any other scalar
 * takes the log of twice pi() in its own precision, which a multiprecision type may set at run
 * time, so nothing is cached; the product is held in a \a Scalar before the log because an
 * expression type is one \c numext::log cannot take.
 */
template <typename Scalar>
[[nodiscard]] Scalar
log2Pi()
{
    using Literal = typename Eigen::NumTraits<Scalar>::Literal;
    if constexpr (std::floating_point<Scalar>)
    {
        return Scalar(1.83787706640934548356065947281123527972L);
    }
    else if constexpr (std::floating_point<Literal>)
    {
        return Scalar(log2Pi<Literal>());
    }
    else
    {
        Scalar const twoPi = Scalar(2) * pi<Scalar>();
        return Eigen::numext::log(twoPi);
    }
}

/** \ingroup Core
 * \returns \f$ \log \Gamma(x) \f$, the log of the
 * [gamma function](https://en.wikipedia.org/wiki/Gamma_function), as a \a Scalar.
 *
 * In place of \c Eigen::numext::lgamma, which supports only \c float and \c double, and
 * \c std::lgamma, which \c AutoDiffScalar does not overload. A floating-point \a Scalar takes
 * \c std::lgamma, <tt>long double</tt> included. An \c Eigen::AutoDiffScalar takes its
 * value's and carries the derivatives by \f$ \tfrac{d}{dx} \log \Gamma(x) = \psi(x) \f$, with
 * the [digamma function](https://en.wikipedia.org/wiki/Digamma_function) \f$ \psi \f$ from
 * \c Eigen::numext::digamma, so its value type must be \c float or \c double. Any other scalar
 * calls its own \c lgamma, found by argument-dependent lookup. The Gamma and Student-t
 * normalizers use it; their derivative with respect to a shape or the degrees of freedom runs
 * through the digamma term.
 *
 * \a Scalar is not deduced: \c std::type_identity_t keeps the argument out of deduction, so a
 * call names it, as in <tt>logGamma<Scalar>(a * b)</tt>. An argument that is an expression,
 * as \c AutoDiffScalar arithmetic yields, then converts to a plain \a Scalar rather than
 * becoming the return type.
 */
template <typename Scalar>
[[nodiscard]] Scalar
logGamma(std::type_identity_t<Scalar> const & x)
{
    if constexpr (std::floating_point<Scalar>)
    {
        return std::lgamma(x);
    }
    else if constexpr (internal::isAutoDiffScalar<Scalar>)
    {
        using Real = typename Scalar::Real;
        return Scalar{logGamma<Real>(x.value()), Eigen::numext::digamma(x.value()) * x.derivatives()};
    }
    else
    {
        return lgamma(x);
    }
}

/** \ingroup Core
 * \returns \f$ \log(1 + x) \f$ as a \a Scalar, accurate for small \a x.
 *
 * In place of \c Eigen::numext::log1p, which fails for \c AutoDiffScalar for want of its
 * \c log1p. A floating-point \a Scalar takes \c std::log1p. An \c Eigen::AutoDiffScalar takes
 * its value's and carries the derivatives by \f$ \tfrac{d}{dx} \log(1 + x) = 1 / (1 + x) \f$,
 * keeping the small-\a x accuracy that <tt>log(1 + x)</tt> would lose. Any other scalar goes
 * through \c Eigen::numext::log1p.
 *
 * \a Scalar is not deduced, as for logGamma(): a call names it, as in
 * <tt>logOnePlus<Scalar>(x)</tt>.
 */
template <typename Scalar>
[[nodiscard]] Scalar
logOnePlus(std::type_identity_t<Scalar> const & x)
{
    if constexpr (std::floating_point<Scalar>)
    {
        return std::log1p(x);
    }
    else if constexpr (internal::isAutoDiffScalar<Scalar>)
    {
        using Real = typename Scalar::Real;
        return Scalar{logOnePlus<Real>(x.value()), x.derivatives() / (Real(1) + x.value())};
    }
    else
    {
        return Eigen::numext::log1p(x);
    }
}

/** \ingroup Core
 * \returns \f$ \lfloor x \rfloor \f$, the largest integer not above \a x, as a \a Scalar.
 *
 * In place of \c Eigen::numext::floor, which fails for \c AutoDiffScalar for want of its
 * \c floor. A floating-point \a Scalar takes \c std::floor. An \c Eigen::AutoDiffScalar takes its
 * value's, with zero derivatives: the floor is constant between integers, and a wrap such as
 * Periodic's subtracts a whole number of periods, so its derivative is the identity's. Any other
 * scalar goes through \c Eigen::numext::floor.
 *
 * \a Scalar is not deduced, as for logGamma(): a call names it, as in <tt>floorOf<Scalar>(x)</tt>.
 */
template <typename Scalar>
[[nodiscard]] Scalar
floorOf(std::type_identity_t<Scalar> const & x)
{
    if constexpr (std::floating_point<Scalar>)
    {
        return std::floor(x);
    }
    else if constexpr (internal::isAutoDiffScalar<Scalar>)
    {
        using Real = typename Scalar::Real;
        return Scalar{floorOf<Real>(x.value()), Scalar::DerType::Zero(x.derivatives().size())};
    }
    else
    {
        return Eigen::numext::floor(x);
    }
}

/** \ingroup Core
 * \returns whether \a x is finite, neither infinite nor NaN.
 *
 * In place of \c Eigen::numext::isfinite, which fails for \c AutoDiffScalar. A floating-point
 * scalar takes \c std::isfinite, an \c Eigen::AutoDiffScalar (or an expression of them) asks of
 * its value, since finiteness is a property of the value and not of the derivatives, and any
 * other scalar goes through \c Eigen::numext::isfinite. It returns \c bool, so the argument is
 * deduced.
 */
template <typename T>
[[nodiscard]] bool
isFinite(T const & x)
{
    if constexpr (std::floating_point<T>)
    {
        return std::isfinite(x);
    }
    else if constexpr (internal::isAutoDiffScalar<T>)
    {
        return isFinite(x.value());
    }
    else
    {
        return Eigen::numext::isfinite(x);
    }
}

/** \ingroup Core
 * \returns whether \a x is NaN.
 *
 * In place of \c Eigen::numext::isnan, which fails for \c AutoDiffScalar; the dispatch is
 * isFinite()'s.
 */
template <typename T>
[[nodiscard]] bool
isNaN(T const & x)
{
    if constexpr (std::floating_point<T>)
    {
        return std::isnan(x);
    }
    else if constexpr (internal::isAutoDiffScalar<T>)
    {
        return isNaN(x.value());
    }
    else
    {
        return Eigen::numext::isnan(x);
    }
}

/** \ingroup Core
 * \returns whether every coefficient of \a x is finite.
 *
 * In place of <tt>x.allFinite()</tt>, which applies \c numext::isfinite to each coefficient and
 * so fails for a matrix of \c AutoDiffScalar. A floating-point matrix takes \c allFinite(),
 * Eigen's vectorized test; any other tests each coefficient with isFinite().
 */
template <typename Derived>
[[nodiscard]] bool
allFinite(Eigen::DenseBase<Derived> const & x)
{
    if constexpr (std::floating_point<typename Derived::Scalar>)
    {
        return x.allFinite();
    }
    else
    {
        for (Eigen::Index j = 0; j < x.cols(); ++j)
        {
            for (Eigen::Index i = 0; i < x.rows(); ++i)
            {
                if (!isFinite(x.coeff(i, j)))
                {
                    return false;
                }
            }
        }
        return true;
    }
}

namespace internal
{

/** \returns the count \a count, such as a dimension or a number of particles, as a \a Scalar.
 *
 * In place of <tt>Scalar(count)</tt>, which for an \c AutoDiffScalar converts the integer to
 * \c double inside Eigen's constructor, a narrowing MSVC warns of (C4244) at the cast. Like pi(),
 * it goes through <tt>Eigen::NumTraits<Scalar>::Literal</tt> when that is floating point, so the
 * conversion is a cast the code asks for; any other scalar converts the integer itself.
 */
template <typename Scalar>
[[nodiscard]] Scalar
scalarFromCount(std::integral auto count)
{
    using Literal = typename Eigen::NumTraits<Scalar>::Literal;
    if constexpr (std::floating_point<Literal>)
    {
        return Scalar(Literal(count));
    }
    else
    {
        return Scalar(count);
    }
}

/** \returns \f$ \sqrt{x} \f$ of \a x at least 0, by Newton's method from above, for constants
 * computed at compile time, such as a method's weights (Methods.h, Transform.h): \c std::sqrt is not
 * \c constexpr before C++26. Newton's iterates from above decrease to the root, and it stops at
 * the first that does not, within an ulp of it.
 */
[[nodiscard]] consteval double
compileTimeSqrt(double x)
{
    if (x <= 0)
    {
        return 0;
    }
    double root = x > 1 ? x : 1;
    while (true)
    {
        double const next = (root + x / root) / 2;
        if (!(next < root))
        {
            return root;
        }
        root = next;
    }
}

}  // namespace internal

}  // namespace Emrys
