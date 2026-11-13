#pragma once

/** \file
 * The transform methods, as policies made by their factories (decisions.md [policy-factories],
 * [method-per-measurement], [affine-coordinates], [returned-jacobians], [iterated-update],
 * [sigma-point-sets], [user-extension-points]): methods::linearized(), unscented(),
 * fourthMomentUnscented(), cubature(), sphericalSimplex(), gaussHermite(), spread(), and
 * iterated(); the UnscentedRule and PointSet concepts a rule or a point set of the user's own meets; the
 * TransformMethod concept observe() requires of its method; and withJacobian() for a function that knows
 * its Jacobian. What each method computes is in Transform.h. Read TransformArgs.h next.
 */

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Eigenvalues>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Roots.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Transforms
 * What a function returns when it knows its Jacobian: its value, and the Jacobian of its output
 * differences in its uncertain inputs' differences, at the same point (decisions.md [returned-jacobians]). Made by
 * withJacobian().
 *
 * \a Jacobian is one matrix when the function has one uncertain input, and otherwise a
 * \c std::tuple of blocks, one per uncertain input in order, each with a row per output
 * difference coordinate and a column per difference coordinate of its input. A model's function
 * returns one matrix over its whole input instead, rows in its output's differences and columns in
 * its input's (decisions.md [returned-jacobians]).
 */
template <typename Value, typename Jacobian>
struct WithJacobian
{
    Value value;        ///< the function's value
    Jacobian jacobian;  ///< its Jacobian, one matrix or a tuple of blocks
};

namespace internal
{

/** The unqualified \a T as a function stores it beside its Jacobian: a matrix expression
 * evaluated to a plain matrix, a tuple of them a tuple of plain matrices.
 */
template <typename T>
struct PlainResultOf
{
    /** \a T itself. */
    using type = T;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <MatrixExpression T>
struct PlainResultOf<T>
{
    using type = typename T::PlainObject;
};

template <typename... Blocks>
struct PlainResultOf<std::tuple<Blocks...>>
{
    using type = std::tuple<typename PlainResultOf<std::remove_cvref_t<Blocks>>::type...>;
};

/// \endcond

/** The unqualified \a T as a function stores it beside its Jacobian. */
template <typename T>
using PlainResult = typename PlainResultOf<T>::type;

/** \returns \a f called with \a args, with a returned Eigen expression evaluated into a plain
 * matrix while the arguments are alive: an expression such as <tt>H * s</tt> refers to its
 * arguments, which the library builds as temporaries for the call, so read after the call it would
 * read freed memory. A plain value, a WithJacobian, or anything else is returned
 * as it is.
 */
template <typename F, typename... Args>
[[nodiscard]] decltype(auto)
callEvaluated(F & f, Args &&... args)
{
    using Returned = std::remove_cvref_t<decltype(f(std::forward<Args>(args)...))>;
    if constexpr (MatrixExpression<Returned> && !std::same_as<PlainResult<Returned>, Returned>)
    {
        return PlainResult<Returned>{f(std::forward<Args>(args)...)};
    }
    else
    {
        return f(std::forward<Args>(args)...);
    }
}

/** Whether \a F, called with \a Args, takes \a Scalar as its first template argument, given
 * explicitly: a function written <tt>[]\<typename T\>(auto const & x)</tt>, whose \c T no argument
 * deduces (decisions.md [function-scalars]). A generic function without a template parameter list
 * doesn't: \a Scalar would name its first argument's type, which a space or a vector is not.
 */
template <typename F, typename Scalar, typename... Args>
concept TakesScalar = requires(F & f, Args &&... args) { f.template operator()<Scalar>(std::forward<Args>(args)...); };

/** The function \a F, by reference, called with \a Scalar as its explicit template argument where it
 * takes one (TakesScalar) and as it is otherwise (decisions.md [function-scalars]). It is invocable
 * exactly where one of the two calls is, so the form a function takes is still told by overload
 * resolution (functionForm).
 */
template <typename Scalar, typename F>
struct WithScalar
{
    F & f;  ///< the function

    /** \returns \a f called with \a args, and with \a Scalar where it takes it */
    template <typename... Args>
        requires(TakesScalar<F, Scalar, Args...> || std::invocable<F &, Args...>)
    [[nodiscard]] decltype(auto)
    operator()(Args &&... args) const
    {
        if constexpr (TakesScalar<F, Scalar, Args...>)
        {
            return f.template operator()<Scalar>(std::forward<Args>(args)...);
        }
        else
        {
            return f(std::forward<Args>(args)...);
        }
    }
};

/** \returns \a f called with \a args as WithScalar calls it: with \a Scalar as its explicit template
 * argument where it takes one, a returned expression evaluated as callEvaluated() does
 */
template <typename Scalar, typename F, typename... Args>
[[nodiscard]] decltype(auto)
callWithScalar(F & f, Args &&... args)
{
    auto const supplied = WithScalar<Scalar, F>{f};
    return callEvaluated(supplied, std::forward<Args>(args)...);
}

/** The form a model's function \a F takes (decisions.md [model-policies]), told by the count of its
 * arguments: 3 for the full form <tt>f(in, out, x, rest...)</tt>, 2 for <tt>f(in, x, rest...)</tt>,
 * 1 for <tt>f(x, rest...)</tt>, and 0 when it is invocable in none. \a In and \a Out are the input and
 * output spaces, \a X the input vector, and \a Rest what follows it: a process model's step, a
 * batched function's output, or nothing. Each form is tried with the scalar of \a X as the explicit
 * template argument too (WithScalar). A count that does not match fails overload resolution before a
 * generic function's body is instantiated, so only the matching form's body is.
 */
template <typename F, typename In, typename Out, typename X, typename... Rest>
inline constexpr int functionForm = [] {
    using G = WithScalar<typename X::Scalar, F const> const &;
    return std::invocable<G, In const &, Out const &, X const &, Rest...> ? 3
         : std::invocable<G, In const &, X const &, Rest...>              ? 2
         : std::invocable<G, X const &, Rest...>                          ? 1
                                                                          : 0;
}();

/** \returns \a f called in the form it takes (functionForm): with the input space \a in, the output
 * space \a out, the input vector \a x, and \a rest; with \a in, \a x, and \a rest; or with \a x and
 * \a rest; and in each with the scalar of \a x as its explicit template argument where it takes one
 * (WithScalar), a returned expression evaluated as callEvaluated() does
 */
template <typename F, typename In, typename Out, typename X, typename... Rest>
[[nodiscard]] decltype(auto)
callForm(F const & f, In const & in, [[maybe_unused]] Out const & out, X const & x, Rest &&... rest)
{
    constexpr int Form = functionForm<F, In, Out, X, Rest &&...>;
    static_assert(
        Form != 0,
        "a model's function takes h(x), h(in, x), or h(in, out, x), and a process model's f(x, dt), f(in, x, dt), or "
        "f(in, out, x, dt), with in the input space, out the output space, and x a vector in the input space; a "
        "batched function takes its output Y last; and each may take x's scalar as its one template parameter, "
        "[]<typename T>(auto const & x)"
    );
    using Scalar = typename X::Scalar;
    if constexpr (Form == 3)
    {
        return callWithScalar<Scalar>(f, in, out, x, std::forward<Rest>(rest)...);
    }
    else if constexpr (Form == 2)
    {
        return callWithScalar<Scalar>(f, in, x, std::forward<Rest>(rest)...);
    }
    else
    {
        return callWithScalar<Scalar>(f, x, std::forward<Rest>(rest)...);
    }
}

}  // namespace internal

/** \ingroup Transforms
 * \returns \a value with \a jacobian beside it, for a function to return when it knows its
 * Jacobian (see WithJacobian). Expressions are evaluated, and plain matrices passed as rvalues
 * are moved.
 *
 * Linearized then takes the Jacobian as it is and calls the function once; the other methods
 * use the value alone, so a function used only by them has no reason to return one.
 *
 * \code
 * // Ranges to anchors, and their Jacobian: each anchor's unit direction from p, negated.
 * auto const ranges = [&](Eigen::Vector2d const & p) {
 *   Eigen::Matrix<double, 2, 4> const toAnchor = anchors.colwise() - p;
 *   Eigen::Vector4d const r = toAnchor.colwise().norm().transpose();
 *   return Emrys::withJacobian(r, -(toAnchor.array().rowwise() / r.transpose().array()).matrix().transpose());
 * };
 * \endcode
 */
template <typename Value, typename Jacobian>
[[nodiscard]] WithJacobian<
    internal::PlainResult<std::remove_cvref_t<Value>>,
    internal::PlainResult<std::remove_cvref_t<Jacobian>>>
withJacobian(Value && value, Jacobian && jacobian)
{
    return {std::forward<Value>(value), std::forward<Jacobian>(jacobian)};
}

/** The most points a PointSet may have over a fixed number of coordinates, which bounds the
 * compile-time size of its matrices: a transform by a set of more is refused at compile time,
 * and a GaussHermite rule of more by an assertion over a dynamic number of coordinates too.
 */
inline constexpr int MaxPointCount = 4096;

namespace internal
{

/** Whether \a Scalar takes a method's weights as constants computed at compile time: \c float or
 * \c double, to which a \c double computed by the compiler converts without loss of what the
 * scalar can hold. A method's parameters are compile-time constants (decisions.md [sigma-point-sets]), so the
 * scalars its weights reduce to over a fixed number of coordinates (Unscented's weights and
 * spread, the cubature radius and weight) are too; a \c float filter then holds them as \c float
 * constants and computes none of them. Any other scalar, such as an autodiff or a multiprecision
 * one, computes them at run time in its own arithmetic, at its own precision. What is sized by a
 * count, the spherical simplex's steps and the Gauss-Hermite rule, is computed at run time on
 * every transform (roadmap, Before going public).
 */
template <typename Scalar>
inline constexpr bool takesDoubleConstants = std::same_as<Scalar, float> || std::same_as<Scalar, double>;

/** \returns the weights and the one-dimensional points of the Gauss-Hermite rule of \a Order points
 * (GaussHermite): the roots of the probabilists' Hermite polynomial \f$ H_p \f$, as the
 * eigenvalues of its symmetric tridiagonal recursion matrix, and the weights
 * \f$ p! / (p^2 H_{p-1}(\xi_i)^2) \f$, \ref Sarkka2013 "Särkkä (2013)", (6.14) and (6.17).
 */
template <typename Scalar, int Order>
[[nodiscard]] std::pair<Eigen::Matrix<Scalar, Order, 1>, Eigen::Matrix<Scalar, Order, 1>>
hermiteRule()
{
    using Vector = Eigen::Matrix<Scalar, Order, 1>;
    using Matrix = Eigen::Matrix<Scalar, Order, Order>;
    Matrix recursion = Matrix::Zero();
    for (int k = 1; k < Order; ++k)
    {
        recursion(k - 1, k) = Eigen::numext::sqrt(Scalar(k));
        recursion(k, k - 1) = recursion(k - 1, k);
    }
    auto const solver = Eigen::SelfAdjointEigenSolver<Matrix>{recursion, Eigen::EigenvaluesOnly};
    Vector const nodes = solver.eigenvalues();
    Scalar factorial = 1;
    for (int k = 2; k <= Order; ++k)
    {
        factorial *= Scalar(k);
    }
    auto weights = Vector{};
    for (int i = 0; i < Order; ++i)
    {
        // H_{p-1}(x) by H_{k+1} = x H_k - k H_{k-1}, from H_0 = 1 and H_1 = x.
        Scalar previous = 1;
        Scalar current = nodes(i);
        for (int k = 1; k < Order - 1; ++k)
        {
            Scalar const next = nodes(i) * current - Scalar(k) * previous;
            previous = current;
            current = next;
        }
        Scalar const value = Order == 1 ? Scalar(1) : current;
        weights(i) = factorial / (Scalar(Order) * Scalar(Order) * value * value);
    }
    return {nodes, weights};
}

}  // namespace internal

/** \ingroup Transforms
 * \brief The parameters of the scaled unscented transform over a number of coordinates, in
 * \a Scalar, as an UnscentedRule gives them: \f$ \lambda = \alpha^2 (L + \kappa) - L \f$.
 */
template <typename Scalar>
struct UnscentedParams
{
    Scalar alpha;  ///< the spread of the points about the mean
    Scalar beta;   ///< the center point's extra covariance weight
    Scalar kappa;  ///< the secondary scaling
};

/** \ingroup Transforms
 * \brief The transform methods and the sigma-point rules, each made by its factory (decisions.md
 * [policy-factories], [method-per-measurement]): linearized(), unscented(), cubature(),
 * sphericalSimplex(), gaussHermite(), and iterated(). The types beside them name a
 * method where a type is needed, as a filter's template argument.
 */
namespace methods
{

/** \ingroup Transforms
 * Linearized method: \f$ \bar y = f(\mu) \f$ and \f$ \Sigma_{yy} = \sum_i J_i \Sigma_i J_i^T \f$,
 * with the Jacobians \f$ J_i \f$ of \a f at the input means. Exact for an affine \a f.
 *
 * An \a f that returns its Jacobian with its value (withJacobian()) is called once, at the means,
 * and that Jacobian used. Any other \a f is differentiated by forward-mode
 * [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), with
 * Eigen's \c AutoDiffScalar, and must be generic in its scalar type; one that returns plain
 * scalars when given dual numbers fails with a static_assert that says so. There is no numerical
 * differentiation by fixed steps: no one step size suits every input, and the covariance, which
 * should set it, is what the unscented methods already spread their points by (decisions.md [returned-jacobians]).
 */
struct Linearized
{};

/** \ingroup Transforms
 * \returns the Linearized method
 */
[[nodiscard]] constexpr Linearized
linearized()
{
    return {};
}

/** \ingroup Transforms
 * Unscented method: \f$ 2L + 1 \f$ sigma points over the joint input space of dimension
 * \f$ L \f$, spread by \f$ \lambda = \alpha^2 (L + \kappa) - L \f$.
 *
 * The defaults \f$ \alpha = 1, \beta = 2, \kappa = 0 \f$ give a zero center mean weight and
 * positive weights otherwise, so the covariance is positive semidefinite by construction;
 * \f$ \beta = 2 \f$ is optimal for %Gaussian inputs. The other points sit at \f$ \sqrt{L} \f$
 * standard deviations, 2 over four coordinates and 3.9 over fifteen, so over a large input \a f is
 * evaluated far into the tails. Sigma points are
 * normalized in their inputs' spaces before \a f sees them, and the output mean is a
 * tangent-space mean in the output space.
 *
 * The scaled unscented transform of \ref Julier2004 "Julier and Uhlmann (2004)", as given in
 * \ref Sarkka2013 "Särkkä (2013)", §5.5, equations (5.74) to (5.77), with
 * their \f$ \alpha \f$, \f$ \beta \f$, and \f$ \kappa \f$; the weights are in
 * internal::sigmaWeights().
 *
 * The parameters are compile-time constants, as a Periodic coordinate's interval is
 * (<tt>unscented<0.5>()</tt>), so the method is an empty tag and a \c float filter converts them to
 * \c float constants where it computes the weights, holding no \c double. They are the same over
 * every number of coordinates; FourthMomentUnscented, or an UnscentedRule of the user's own, chooses
 * them by it.
 *
 * \tparam Alpha  the spread of the points about the mean, in \f$ (0, 1] \f$
 * \tparam Beta   the center point's extra covariance weight; 2 for %Gaussian inputs
 * \tparam Kappa  the secondary scaling, usually 0
 */
template <double Alpha = 1.0, double Beta = 2.0, double Kappa = 0.0>
struct Unscented
{
    static_assert(Alpha > 0, "the unscented transform's alpha must be positive");

    static constexpr double alpha = Alpha;  ///< \a Alpha
    static constexpr double beta = Beta;    ///< \a Beta
    static constexpr double kappa = Kappa;  ///< \a Kappa

    /** \returns \a Alpha, \a Beta, and \a Kappa in \a Scalar, over any number of coordinates */
    template <typename Scalar>
    [[nodiscard]] static constexpr UnscentedParams<Scalar>
    params(Eigen::Index /*n*/)
    {
        return {Scalar(Alpha), Scalar(Beta), Scalar(Kappa)};
    }
};

/** \ingroup Transforms
 * \returns the Unscented method with the parameters \a Alpha, \a Beta, and \a Kappa
 * (<tt>unscented<0.5>()</tt>)
 */
template <double Alpha = 1.0, double Beta = 2.0, double Kappa = 0.0>
[[nodiscard]] constexpr Unscented<Alpha, Beta, Kappa>
unscented()
{
    return {};
}

/** \ingroup Transforms
 * Unscented method with \f$ \alpha = 1 \f$, \f$ \beta = 0 \f$, and \f$ \kappa = 3 - L \f$ over
 * \f$ L \f$ coordinates, so that \f$ L + \lambda = 3 \f$: the points sit at \f$ \sqrt 3 \f$
 * standard deviations along each column of the root whatever \f$ L \f$ is, and their fourth moment
 * along each column is a %Gaussian's, 3.
 *
 * The choice of \f$ \kappa \f$ for a %Gaussian input of \ref Julier2004 "Julier and Uhlmann (2004)"
 * (\ref Sarkka2013 "Särkkä (2013)", §6.5, at equation (6.45)).
 * The center point's weight is \f$ 1 - L / 3 \f$, negative over more than three coordinates, so the
 * covariance is a difference of positive terms there and can be indefinite; a square-root form
 * downdates by the center's column. \f$ \beta = 0 \f$ because the points already carry the fourth
 * moment that \f$ \beta = 2 \f$ makes up for. An UnscentedRule, as Unscented is, whose
 * parameters depend on the number of coordinates: over a fixed number, the compiler computes its
 * weights as it does Unscented's.
 */
struct FourthMomentUnscented
{
    /** \returns \f$ \alpha = 1 \f$, \f$ \beta = 0 \f$, and \f$ \kappa = 3 - n \f$ in \a Scalar */
    template <typename Scalar>
    [[nodiscard]] static constexpr UnscentedParams<Scalar>
    params(Eigen::Index n)
    {
        return {Scalar(1), Scalar(0), Scalar(3) - Scalar(n)};
    }
};

/** \ingroup Transforms
 * \returns the FourthMomentUnscented method
 */
[[nodiscard]] constexpr FourthMomentUnscented
fourthMomentUnscented()
{
    return {};
}

/** \ingroup Transforms
 * Cubature method: the \f$ 2L \f$ points \f$ \mu \boxplus \pm \sqrt{L} \, S_j \f$ over the joint
 * input space of dimension \f$ L \f$, for each column \f$ S_j \f$ of a root of the joint
 * covariance, equally weighted by \f$ 1 / (2L) \f$, with no center point and no parameter. Exact
 * for polynomials of degree three in a %Gaussian input. Over no coordinates it is the mean alone.
 *
 * The third-degree spherical-radial cubature rule of the cubature Kalman filter of Arasaratnam
 * and Haykin (2009), as \ref Arasaratnam2010 "Arasaratnam, Haykin, and Hurd (2010)", §IV-A, state
 * it. It is Unscented with \f$ \alpha = 1, \beta = 0, \kappa = 0 \f$, whose center point has
 * weight zero, without evaluating the function there. The output mean is a tangent-space mean
 * about the first point's output (decisions.md [sigma-point-sets]).
 */
struct Cubature
{
    /** \returns the points over \a n coordinates, \f$ 2n \f$ */
    [[nodiscard]] static constexpr Eigen::Index
    pointCount(Eigen::Index n)
    {
        return 2 * n;
    }

    /** Fills \a points, \f$ n \times 2n \f$, with \f$ \pm \sqrt{n} \, e_i \f$, the plus points first, and
     * \a weights with \f$ 1 / (2n) \f$ each, over \f$ n > 0 \f$ coordinates; over a fixed \a n in
     * \c float or \c double the constants are the compiler's (internal::takesDoubleConstants).
     *
     * \returns the point the output mean is taken about, the first
     */
    template <typename Points, typename Weights>
    static Eigen::Index
    unitPoints(Eigen::Index n, Points & points, Weights & weights)
    {
        using Scalar = typename Points::Scalar;
        constexpr int N = Points::RowsAtCompileTime;
        points.setZero();
        if constexpr (N != Eigen::Dynamic && N > 0 && internal::takesDoubleConstants<Scalar>)
        {
            constexpr Scalar radius = Scalar(internal::compileTimeSqrt(double(N)));
            points.diagonal().setConstant(radius);
            points.rightCols(N).diagonal().setConstant(-radius);
            weights.setConstant(Scalar(1.0 / (2.0 * N)));
        }
        else
        {
            Scalar const radius = Eigen::numext::sqrt(Scalar(n));
            for (Eigen::Index i = 0; i < n; ++i)
            {
                points(i, i) = radius;
                points(i, n + i) = -radius;
            }
            weights.setConstant(Scalar(1) / Scalar(2 * n));
        }
        return 0;
    }
};

/** \ingroup Transforms
 * \returns the Cubature method
 */
[[nodiscard]] constexpr Cubature
cubature()
{
    return {};
}

/** \ingroup Transforms
 * Spherical simplex method: \f$ L + 2 \f$ points over the joint input space of dimension \f$ L \f$,
 * the mean with weight \a centerWeight \f$ W_0 \f$ and \f$ L + 1 \f$ points of weight
 * \f$ (1 - W_0) / (L + 1) \f$ on a sphere of radius \f$ \sqrt{L / (1 - W_0)} \f$ about it, in
 * the coordinates of a root of the joint covariance. The fewest points that match a mean and a
 * covariance, plus the center, for a function too costly to evaluate \f$ 2L + 1 \f$ times. The
 * points are not symmetric about the mean, so odd moments of the input are not matched as the
 * symmetric sets match them.
 *
 * The spherical simplex set of \ref Julier2003 "Julier (2003)", by the construction of \ref Julier2004 "Julier and
 * Uhlmann (2004)", which places all but the center at one distance from it,
 * \f$ \sqrt{L / (1 - W_0)} \f$. No free description of the construction has been read, and the
 * original has not been. The output mean is a tangent-space mean about the center's output
 * (decisions.md [sigma-point-sets]). Its parameter is a compile-time constant, as Unscented's are.
 *
 * \tparam CenterWeight  \f$ W_0 \f$, the center point's weight, in \f$ [0, 1) \f$
 */
template <double CenterWeight = 0.0>
struct SphericalSimplex
{
    static_assert(CenterWeight >= 0 && CenterWeight < 1, "a spherical simplex's center weight must be in [0, 1)");

    static constexpr double centerWeight = CenterWeight;  ///< \a CenterWeight

    /** \returns the points over \a n coordinates, \f$ n + 2 \f$ */
    [[nodiscard]] static constexpr Eigen::Index
    pointCount(Eigen::Index n)
    {
        return n + 2;
    }

    /** Fills \a points, \f$ n \times (n + 2) \f$, with the center, then the \f$ n + 1 \f$ points of the
     * recursion of \ref Julier2004 "Julier and Uhlmann (2004)", whose coordinate
     * \f$ j \f$ is \f$ -1 / \sqrt{j (j + 1) W_1} \f$ at points 1 to \f$ j \f$,
     * \f$ j / \sqrt{j (j + 1) W_1} \f$ at point \f$ j + 1 \f$, and zero after, and \a weights with
     * \f$ W_0 \f$ and \f$ W_1 = (1 - W_0) / (n + 1) \f$, over \f$ n > 0 \f$ coordinates.
     *
     * \returns the point the output mean is taken about, the center
     */
    template <typename Points, typename Weights>
    static Eigen::Index
    unitPoints(Eigen::Index n, Points & points, Weights & weights)
    {
        using Scalar = typename Points::Scalar;
        points.setZero();
        Scalar const center = Scalar(CenterWeight);
        Scalar const pointWeight = (Scalar(1) - center) / Scalar(n + 1);
        weights(0) = center;
        weights.tail(n + 1).setConstant(pointWeight);
        for (Eigen::Index j = 1; j <= n; ++j)
        {
            Scalar const step = Scalar(1) / Eigen::numext::sqrt(Scalar(j * (j + 1)) * pointWeight);
            points.row(j - 1).segment(1, j).setConstant(-step);
            points(j - 1, j + 1) = Scalar(j) * step;
        }
        return 0;
    }
};

/** \ingroup Transforms
 * \returns the SphericalSimplex method with the center weight \a CenterWeight
 * (<tt>sphericalSimplex<0.5>()</tt>)
 */
template <double CenterWeight = 0.0>
[[nodiscard]] constexpr SphericalSimplex<CenterWeight>
sphericalSimplex()
{
    return {};
}

/** \ingroup Transforms
 * Gauss-Hermite method: the product rule of \a Order points per coordinate over the joint input
 * space of dimension \f$ L \f$, \f$ \mathrm{Order}^L \f$ points in the coordinates of a root of
 * the joint covariance. Each coordinate takes the roots of the probabilists' Hermite polynomial
 * of degree \a Order, and a point's weight is the product of its coordinates' weights. Exact for
 * polynomials of degree up to \f$ 2 \, \mathrm{Order} - 1 \f$ in each coordinate of a %Gaussian
 * input. The count grows exponentially in \f$ L \f$, so the method is for small inputs, such as
 * the coordinates a model's affineIn() leaves nonlinear, and a joint input of more than MaxPointCount points
 * is refused: at compile time where its size is fixed, by an assertion otherwise.
 *
 * The Gauss-Hermite Kalman filter of \ref Ito2000 "Ito and Xiong (2000)" as \ref Sarkka2013 "Särkkä (2013)",
 * algorithms 6.5 and 6.6, give it: the one-dimensional points as the roots of \f$ H_p \f$, computed
 * as that book advises as the eigenvalues of the symmetric tridiagonal matrix of the recursion
 * (6.14), with \f$ \sqrt{k} \f$ beside the diagonal, and the weights by equation (6.17),
 * \f$ W_i = p! / (p^2 H_{p-1}(\xi_i)^2) \f$. The output mean is a tangent-space mean about the
 * output of the point of greatest weight, the center when \a Order is odd (decisions.md [sigma-point-sets]).
 *
 * \tparam Order  the points per coordinate, at least 1; 3 is exact for a quadratic function's
 *                mean and covariance
 */
template <int Order = 3>
struct GaussHermite
{
    static_assert(Order >= 1, "a Gauss-Hermite rule has at least one point per coordinate");
    static constexpr int PointsPerCoordinate = Order;  ///< \a Order

    /** \returns the points over \a n coordinates, \f$ \mathrm{Order}^n \f$, or one more than
     * MaxPointCount once past it, so that the count never overflows
     */
    [[nodiscard]] static constexpr Eigen::Index
    pointCount(Eigen::Index n)
    {
        Eigen::Index count = 1;
        for (Eigen::Index i = 0; i < n && count <= MaxPointCount; ++i)
        {
            count *= Order;
        }
        return std::min<Eigen::Index>(count, MaxPointCount + 1);
    }

    /** Fills \a points, \f$ n \times \mathrm{Order}^n \f$, with every combination of the one-dimensional
     * points of internal::hermiteRule(), the first coordinate varying fastest, and \a weights with the
     * product of each point's coordinates' weights, over \f$ n > 0 \f$ coordinates; asserts that the
     * points are within MaxPointCount.
     *
     * \returns the point the output mean is taken about: the one of greatest weight, the first of any tie
     */
    template <typename Points, typename Weights>
    static Eigen::Index
    unitPoints(Eigen::Index n, Points & points, Weights & weights)
    {
        using Scalar = typename Points::Scalar;
        Eigen::Index const count = weights.size();
        EMRYS_FILTER_ASSERT(
            count <= MaxPointCount && "a Gauss-Hermite rule over this many coordinates has too many points"
        );
        auto const [nodes, nodeWeights] = internal::hermiteRule<Scalar, Order>();
        for (Eigen::Index k = 0; k < count; ++k)
        {
            Eigen::Index digits = k;
            Scalar weight = 1;
            for (Eigen::Index i = 0; i < n; ++i)
            {
                Eigen::Index const digit = digits % Order;
                digits /= Order;
                points(i, k) = nodes(digit);
                weight *= nodeWeights(digit);
            }
            weights(k) = weight;
        }
        Eigen::Index reference = 0;
        weights.maxCoeff(&reference);
        return reference;
    }
};

/** \ingroup Transforms
 * \returns the GaussHermite method of \a Order points per coordinate (<tt>gaussHermite<3>()</tt>)
 */
template <int Order = 3>
[[nodiscard]] constexpr GaussHermite<Order>
gaussHermite()
{
    return {};
}

}  // namespace methods

/** \ingroup Transforms
 * \brief A rule for the scaled unscented transform's parameters by the number of coordinates it
 * spreads its points over: a method that places Unscented's \f$ 2L + 1 \f$ points with Unscented's
 * weights, at the \f$ \alpha \f$, \f$ \beta \f$, and \f$ \kappa \f$ it gives for each \f$ L \f$.
 *
 * Unscented and FourthMomentUnscented are the library's. One of the user's own is any type with
 * the static member below, with nothing of the library's specialized, and is then a
 * TransformMethod: a measurement's method and a filter's prediction method, which spreads its
 * points over the coordinates a model's affineIn() does not name (decisions.md
 * [user-extension-points], [affine-coordinates]). usage.md, "Choosing the unscented parameters by the
 * dimension", writes one.
 *
 * - <tt>template \<typename Scalar\> static constexpr UnscentedParams\<Scalar\> params(Eigen::Index n)</tt>:
 *   the parameters over \f$ n > 0 \f$ coordinates, computed in \a Scalar, so that a \c float filter
 *   computes no \c double [embedded-builds]. It is a constant expression for a constant \a n, so
 *   that the compiler computes the weights of a fixed-size input in \c float or \c double, and there
 *   it must give \f$ \alpha^2 (n + \kappa) > 0 \f$ or the transform fails to compile; elsewhere the
 *   transform asserts it.
 *
 * Over no coordinates the transform is the mean alone and calls nothing. A type meets this
 * concept or PointSet, not both.
 */
template <typename T>
concept UnscentedRule = requires(Eigen::Index n) {
    { T::template params<double>(n) } -> std::same_as<UnscentedParams<double>>;
};

/** \ingroup Transforms
 * \brief A set of sigma points other than Unscented's: a method that evaluates a function at
 * weighted points given in the coordinates of a root of the covariance, with one weight per point
 * for the mean and the covariance alike.
 *
 * Cubature, SphericalSimplex, and GaussHermite are the library's (decisions.md [sigma-point-sets]).
 * One of the user's own is any type with the two static members below, with nothing of the library's
 * specialized, and is then a TransformMethod: a measurement's method and a filter's prediction
 * method (decisions.md [user-extension-points]). usage.md, "Writing a
 * sigma-point set", writes one.
 *
 * - <tt>static constexpr Eigen::Index pointCount(Eigen::Index n)</tt>: the number of points over
 *   \f$ n > 0 \f$ coordinates, a constant expression, so that the points of a fixed-size input are
 *   sized at compile time; at most MaxPointCount over a fixed-size input.
 * - <tt>template <typename Points, typename Weights> static Eigen::Index unitPoints(Eigen::Index n,
 *   Points & points, Weights & weights)</tt>: fills \a points, an \f$ n \times \mathrm{pointCount}(n) \f$
 *   matrix in the transform's scalar (its \c Scalar), with the points \f$ \xi_j \f$ as its columns,
 *   and \a weights, a vector of as many, with weights that sum to one and give the points mean zero
 *   and identity covariance, and returns the index of the point the output mean is taken about as a
 *   tangent-space mean, the center or the heaviest point. Both arrive sized, with fixed sizes where
 *   the input's are, so a set may compute its constants at compile time, as Cubature does.
 *
 * Over no coordinates every set is the mean alone, which the transform handles before calling either.
 */
template <typename T>
concept PointSet = requires(Eigen::Index n, Eigen::MatrixXd & points, Eigen::VectorXd & weights) {
    { T::pointCount(n) } -> std::convertible_to<Eigen::Index>;
    { T::unitPoints(n, points, weights) } -> std::same_as<Eigen::Index>;
};

namespace methods
{

/** \ingroup Transforms
 * A sigma-point rule spread by axes of its own (decisions.md [sigma-point-axes]): \a Rule's unit points and
 * weights, an UnscentedRule or a PointSet, carried into the input by the root \a A takes (Axes), in
 * place of the default, roots::cholesky(roots::pivoted()). The axes apply to the input marked by
 * withCross(), or to the first uncertain input when none is marked; the other inputs take the
 * default, and an input that holds a root of its own, as a square-root form's blocks do, gives it to
 * the axes as a factor. Wherever a rule is a method, a spread rule is: a measurement's method, a filter's
 * prediction method, and the method an Iterated relinearizes.
 *
 * A transform returns no \c std::expected, so axes that can fail are refused there at compile time,
 * with a message naming roots::pivoted() to append and roots::squareRoot() for a caller who wants the
 * \c std::expected.
 *
 * \tparam Rule  the sigma-point rule
 * \tparam A     the axes
 *
 * \sa spread()
 */
template <typename Rule, typename A>
struct Spread
{
    static_assert(
        UnscentedRule<Rule> || PointSet<Rule>,
        "spread() takes a sigma-point rule, an UnscentedRule or a PointSet, not linearized(), iterated(), or a "
        "rule spread already"
    );
    static_assert(Axes<A>, "spread() takes axes, such as roots::principal(), after its rule");

    Rule rule{};  ///< the unit points and weights
    A axes{};     ///< the axes they are spread along
};

/** \ingroup Transforms
 * \returns the sigma-point rule \a rule spread by the axes \a axes (Spread):
 * <tt>spread(cubature(), roots::principal())</tt>, or
 * <tt>spread(sphericalSimplex(), roots::along(d, roots::pivoted()))</tt>
 */
template <typename Rule, typename A>
[[nodiscard]] constexpr Spread<Rule, A>
spread(Rule rule, A axes)
{
    return {std::move(rule), std::move(axes)};
}

}  // namespace methods

namespace internal
{

/** Whether \a T is a PointSet, a set of sigma points other than Unscented's: the library's or a
 * user's, by its interface.
 */
template <typename T>
inline constexpr bool isPointSet = PointSet<T>;

/** Whether \a T places Unscented's points: an UnscentedRule, the library's or a user's, by its
 * interface.
 */
template <typename T>
inline constexpr bool isUnscented = UnscentedRule<T>;

/** Whether \a T is a rule spread by axes of its own (methods::Spread). */
template <typename T>
inline constexpr bool isSpread = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Rule, typename A>
inline constexpr bool isSpread<methods::Spread<Rule, A>>{true};

/// \endcond

/** Whether \a T is a sigma-point method, which an Iterated may hold and which factors a model's
 * affine coordinates out: an
 * UnscentedRule or a point set, spread by the default axes or by its own.
 */
template <typename T>
inline constexpr bool isSigmaPointMethod = isUnscented<T> || isPointSet<T> || isSpread<T>;

/** \returns the rule of the sigma-point method \a method: its rule if it is spread by axes of its
 * own, and itself otherwise.
 */
template <typename Method>
[[nodiscard]] constexpr decltype(auto)
ruleOf(Method const & method)
{
    if constexpr (isSpread<Method>)
    {
        return (method.rule);
    }
    else
    {
        return (method);
    }
}

/** \returns the axes the sigma-point method \a method spreads by: its own if it has them, and
 * roots::DefaultAxes otherwise.
 */
template <typename Method>
[[nodiscard]] constexpr decltype(auto)
axesOf(Method const & method)
{
    if constexpr (isSpread<Method>)
    {
        return (method.axes);
    }
    else
    {
        return roots::DefaultAxes{};
    }
}

/** The rule of the sigma-point method \a Method (ruleOf()). */
template <typename Method>
using RuleOf = std::remove_cvref_t<decltype(ruleOf(std::declval<Method const &>()))>;

/** The axes of the sigma-point method \a Method (axesOf()). */
template <typename Method>
using AxesOf = std::remove_cvref_t<decltype(axesOf(std::declval<Method const &>()))>;

/** Refuses the axes \a A of a sigma-point spread in a transform when they can fail, since a transform
 * returns no \c std::expected (decisions.md [sigma-point-axes]).
 */
template <typename A>
consteval void
refuseFailingAxes()
{
    static_assert(
        !A::canFail,
        "a transform returns no std::expected, so its axes cannot fail: append roots::pivoted() to the list, or take "
        "the root with roots::squareRoot(), which returns one"
    );
}

/** \returns the root in \a result, which axes returned: the root itself, or the value of a
 * \c std::expected, which a transform's axes, refused when they can fail, always hold.
 */
template <typename Result>
[[nodiscard]] decltype(auto)
rootValue(Result && result)
{
    if constexpr (requires { result.has_value(); })
    {
        EMRYS_FILTER_ASSERT(result.has_value() && "a transform's axes cannot fail");
        return *std::forward<Result>(result);
    }
    else
    {
        return std::forward<Result>(result);
    }
}

/** Whether \a T is a method that linearizes once at the means, as Linearized does, so that the
 * output is affine in the inputs as far as the method sees: \f$ \Sigma_{xy} = \Sigma J^T \f$.
 */
template <typename T>
inline constexpr bool isLinearMethod = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <>
inline constexpr bool isLinearMethod<methods::Linearized> = true;

/// \endcond

}  // namespace internal

/** \ingroup Transforms
 * \brief The run-time settings of an Iterated method, given to methods::iterated() as
 * <tt>iterated(method, {.tolerance = 1e-6})</tt>: what stops the iteration, in the scalar the
 * method holds (decisions.md [policy-factories]).
 *
 * \tparam Scalar  the scalar they are held in: methods::iterated()'s, \c double unless named
 */
template <RealScalar Scalar = double>
struct IteratedOptions
{
    Scalar tolerance = Scalar(1e-3);  ///< the largest step that stops the iteration, in prior standard deviations
    std::size_t maxIterations = 10;   ///< the most linearizations it makes, at least 1
};

namespace methods
{

/** \ingroup Transforms
 * \brief Iterated method, for a measurement update: the update of \a Method relinearized with
 * respect to its own posterior, with the prior held fixed, until the mean stops moving. With
 * Linearized it is the iterated extended Kalman filter's update, which is Gauss-Newton on the
 * update's least-squares problem and so finds the posterior's mode where one linearization at the
 * prior mean falls short, as with a precise measurement of a poorly known state. With a
 * sigma-point method (an UnscentedRule or a PointSet) it is the iterated
 * posterior linearization filter's update (decisions.md [iterated-update], [sigma-point-sets]).
 *
 * Each iteration linearizes the measurement about the iterate \f$ x_i = \mu \boxplus \delta_i \f$,
 * with \f$ \delta_0 = 0 \f$, and takes the Kalman update from the prior \f$ (\mu, P) \f$ under that
 * linearization. Linearized takes the Jacobian at \f$ x_i \f$; a sigma-point method takes the
 * statistical linearization \f$ (G_i, \Omega_i) \f$ of the measurement over the iterate's
 * posterior, its points spread about \f$ x_i \f$ by the posterior covariance of the blocks read,
 * \f$ P_i = P - K_{i-1} S_{i-1} K_{i-1}^T \f$, carried to \f$ x_i \f$'s chart by the reset Jacobian,
 * and its \f$ \Omega_i \f$ added to the noise. With \f$ h \f$ the measurement function and
 * \f$ G_i \f$ its Jacobian in \f$ \delta \f$ at \f$ \delta_i \f$, the predicted measurement is
 * \f$ h(x_i) \boxplus (-G_i \delta_i) \f$, the residual \f$ r_i \f$ is the observation's from it,
 * and the step is \f$ \delta_{i+1} = K_i r_i \f$. On a space with its own \f$ \boxplus \f$ the Jacobian in
 * \f$ \delta \f$ is the Jacobian at \f$ x_i \f$ times the reset Jacobian (resetJacobian()) of the
 * prior mean and \f$ \delta_i \f$, so the iteration is Gauss-Newton in the prior's own chart. The
 * iteration stops at the linearization after a step that moved no coordinate by
 * more than \a tolerance of its prior standard deviation, \f$ |\delta_i - \delta_{i-1}| \le
 * \mathrm{tolerance} \cdot \sqrt{P_{jj}} \f$, or at linearization number \a maxIterations, and
 * that linearization's update is the result, covariance and Innovation included. Each branch of
 * the update reports which (UpdateBranch::iterations and UpdateBranch::converged). One iteration is
 * \a Method's update exactly.
 *
 * A sensor takes it as its method [method-per-measurement]; a transform or a prediction has no posterior to
 * relinearize at and refuses it, as do the updates given a block of the Rao-Blackwellized filters.
 * With mixture noise each component's branch iterates on its own, since its posterior is its own.
 *
 * The update of \ref Bell1993 "Bell and Cathey (1993)", who show the iterated Kalman filter's
 * update to be the Gauss-Newton method for the maximum a posteriori estimate, on a vector state.
 * On a manifold, \ref Hertzberg2013 "Hertzberg et al. (2013)", §4.1 and table 1, take each
 * Gauss-Newton step in the chart at the iterate, \f$ x_{i+1} = x_i \boxplus \Delta \f$; here every
 * step is in the prior's chart instead, where the prior's own residual is \f$ \delta \f$ with
 * covariance \f$ P \f$ as it stands, and the reset Jacobian carries the measurement's Jacobian
 * there. Bell and Cathey leave the stopping rule open; this one is scaled by the prior so that one
 * tolerance serves coordinates in any unit.
 *
 * With a sigma-point method it is algorithm 1 of \ref GarciaFernandez2015 "García-Fernández,
 * Svensson, Morelande, and Särkkä (2015)", the iterated posterior linearization filter, whose
 * statistical linearization with respect to the current posterior approximation they show to
 * minimize the expected divergence of the linearized from the true measurement density, and
 * which they show to be the iterated extended Kalman filter when the moments are taken by
 * linearization at the mean. They stop when the Kullback-Leibler divergence between successive
 * posteriors falls below a threshold; this stops by the step in the mean, as above, so that one
 * rule and one tolerance serve every method. The first iteration is the sigma-point update itself.
 *
 * \code
 * auto const ranger = measurementModel(rangeBearing, reads(path<Position>), additiveNoise(noise));
 * auto const result = filter.update(observe(methods::iterated(methods::linearized(), {.tolerance = 1e-6}), ranger, z));
 * // result.branches[0].iterations, .converged
 * auto const plf = observe(methods::iterated(methods::cubature()), ranger, z);
 * \endcode
 *
 * The tolerance and the iteration cap are run-time values, given to iterated() as an
 * IteratedOptions, since they set how much work is done rather than which algorithm runs
 * (decisions.md [policy-factories]). They are held in \a Scalar, \c double unless iterated() names
 * it, so a \c float filter writes <tt>iterated<float>(method, {.tolerance = 1e-6f})</tt> and holds no
 * \c double [embedded-builds].
 *
 * \tparam Method  the method relinearized: Linearized, or a sigma-point method
 * \tparam Scalar  the scalar of the tolerance
 */
template <typename Method = Linearized, RealScalar Scalar = double>
struct Iterated
{
    static_assert(
        internal::isLinearMethod<Method> || internal::isSigmaPointMethod<Method>,
        "Iterated relinearizes Linearized or a sigma-point method (an UnscentedRule such as Unscented, or a PointSet "
        "such as Cubature, SphericalSimplex, GaussHermite, or the user's own) (decisions.md [iterated-update], "
        "[sigma-point-sets])"
    );

    Method method{};                  ///< the method relinearized at each iterate
    Scalar tolerance = Scalar(1e-3);  ///< the largest step that stops it, in prior standard deviations
    std::size_t maxIterations = 10;   ///< the most linearizations it makes, at least 1
};

/** \ingroup Transforms
 * \returns an Iterated method relinearizing \a method (Linearized by default) until a step is within
 * \a options.tolerance or \a options.maxIterations is reached, holding them in \a Scalar: \c double
 * unless named, since a braced argument deduces nothing, so <tt>iterated(method, {.tolerance =
 * 1e-6})</tt> holds a \c double and <tt>iterated<float>(method, {.tolerance = 1e-6f})</tt> a
 * \c float. Asserts that the tolerance is not negative and that at least one linearization is
 * made.
 */
template <RealScalar Scalar = double, typename Method = Linearized>
[[nodiscard]] Iterated<Method, Scalar>
iterated(Method method = {}, IteratedOptions<Scalar> options = {})
{
    EMRYS_FILTER_ASSERT(options.tolerance >= Scalar(0) && "an iteration's tolerance must not be negative");
    EMRYS_FILTER_ASSERT(options.maxIterations >= 1 && "an Iterated method makes at least one linearization");
    return {std::move(method), options.tolerance, options.maxIterations};
}

}  // namespace methods

namespace internal
{

/** Whether \a T is an Iterated method. */
template <typename T>
inline constexpr bool isIterated = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Method, typename Scalar>
inline constexpr bool isIterated<methods::Iterated<Method, Scalar>>{true};

/// \endcond

}  // namespace internal

/** \ingroup Transforms
 * \brief A method observe() can take: Linearized, an UnscentedRule (Unscented,
 * FourthMomentUnscented, or one of the user's own), a PointSet (Cubature, SphericalSimplex,
 * GaussHermite, or one of the user's own), or an Iterated one of them.
 *
 * observe() requires it of its method, so a measurement made with something else, such as the
 * method's template unspecialized or an unrelated type, fails where it is made rather than deep
 * inside an update.
 */
template <typename M>
concept TransformMethod =
    internal::isLinearMethod<std::remove_cvref_t<M>> || internal::isSigmaPointMethod<std::remove_cvref_t<M>>
    || internal::isIterated<std::remove_cvref_t<M>>;

}  // namespace Emrys
