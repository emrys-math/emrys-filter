#pragma once

/** \file
 * The chi-squared distribution's CDF and quantile, with no dependency outside Eigen, for the
 * gates and consistency checks of Consistency.h and the confidence regions of Gaussian.h. The
 * quantile is the inverse regularized incomplete gamma that Eigen lacks (roadmap, "Upstream to
 * Eigen"). Both run in the caller's scalar, so a \c float filter computes them in \c float and an
 * autodiff scalar carries its derivatives through them, with the log-gamma of Math.h. Read
 * Triangularization.h next.
 */

#include <limits>
#include <type_traits>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>

namespace Emrys
{

namespace internal
{

/** The regularized lower incomplete gamma function \f$ P(a, x) \f$, by its series below
 * \f$ x = a + 1 \f$ and by a continued fraction for \f$ Q = 1 - P \f$ above.
 *
 * The series and the continued fraction, and the switch between them at \f$ x = a + 1 \f$,
 * follow \ref Press2007 "Press et al. (2007)" (the series and fraction as given in
 * [Incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function#Evaluation_formulae);
 * no free source read gives the switch); the continued fraction is evaluated by
 * the modified Lentz method (\ref Lentz1976 "Lentz (1976)",
 * \ref Thompson1986 "Thompson and Barnett (1986)", as given in
 * [Lentz's algorithm](https://en.wikipedia.org/wiki/Lentz%27s_algorithm#Thompson_and_Barnett_modification)),
 * which rescales a vanishing term to \c tiny instead of dividing by zero. Both stop at the epsilon
 * of \a Scalar, where Press et al. fix it for \c double. See the
 * [incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).
 */
template <typename Scalar>
[[nodiscard]] Scalar
regularizedGammaP(std::type_identity_t<Scalar> const & a, std::type_identity_t<Scalar> const & x)
{
    if (x <= Scalar(0))
    {
        return Scalar(0);
    }
    // Both expansions take the logarithm of x, which an infinite x makes NaN.
    if (!(x < Eigen::NumTraits<Scalar>::infinity()))
    {
        return isNaN(x) ? x : Scalar(1);
    }
    using Literal = typename Eigen::NumTraits<Scalar>::Literal;
    Scalar const epsilon = Scalar(Eigen::NumTraits<Scalar>::epsilon());
    // Where x is near a, the series' terms fall as about exp(-n^2 / 2a), so it takes some
    // sqrt(2a ln(1/epsilon)) of them, past any fixed limit once a is in the tens of thousands, as
    // a cumulative consistency monitor's is; both loops are allowed twice that.
    Scalar const maxIterations =
        Scalar(1000) + Scalar(2) * Eigen::numext::sqrt(Scalar(Scalar(-2) * a * Eigen::numext::log(epsilon)));
    Scalar const logPrefactor = Scalar(-x + a * Eigen::numext::log(x)) - logGamma<Scalar>(a);
    if (x < a + Scalar(1))
    {
        Scalar term = Scalar(1) / a;
        Scalar sum = term;
        for (int n = 1;
             scalarFromCount<Scalar>(n) < maxIterations && Eigen::numext::abs(term) > Eigen::numext::abs(sum) * epsilon;
             ++n)
        {
            term *= x / (a + Scalar(n));
            sum += term;
        }
        Scalar const p = sum * Eigen::numext::exp(logPrefactor);
        return p < Scalar(1) ? p : Scalar(1);
    }
    // Modified Lentz's method.
    Scalar const tiny = Scalar((std::numeric_limits<Literal>::min)()) / epsilon;
    Scalar b = x + Scalar(1) - a;
    Scalar c = Scalar(1) / tiny;
    Scalar d = Scalar(1) / b;
    Scalar h = d;
    for (int i = 1; scalarFromCount<Scalar>(i) < maxIterations; ++i)
    {
        Scalar const an = -Scalar(i) * (Scalar(i) - a);
        b += Scalar(2);
        d = an * d + b;
        if (Eigen::numext::abs(d) < tiny)
        {
            d = tiny;
        }
        c = b + an / c;
        if (Eigen::numext::abs(c) < tiny)
        {
            c = tiny;
        }
        d = Scalar(1) / d;
        Scalar const delta = d * c;
        h *= delta;
        if (Eigen::numext::abs(Scalar(delta - Scalar(1))) <= epsilon)
        {
            break;
        }
    }
    Scalar const p = Scalar(1) - Eigen::numext::exp(logPrefactor) * h;
    return Scalar(0) < p ? p : Scalar(0);
}

}  // namespace internal

/** \ingroup Core
 * \returns the chi-squared cumulative distribution function with \a dof degrees of freedom at
 * \a x, \f$ P(\tfrac{k}{2}, \tfrac{x}{2}) \f$
 *
 * The [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution) is
 * that of a sum of \a dof squared standard normals, which is what a normalized innovation
 * squared is for a consistent filter (\ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan
 * (2001)"; \ref Chen2023 "Chen et al. (2023)", §III-A).
 *
 * \a Scalar is not deduced, as for logGamma(): a call names it, as in
 * <tt>chiSquaredCdf<Scalar>(squaredDistance, k)</tt>, and computes in it throughout. In an autodiff scalar
 * the derivative in \a x is the chi-squared density, to the precision of the series or the
 * continued fraction. It is 0 at zero and below, 1 at infinity, and NaN at NaN.
 */
template <typename Scalar>
[[nodiscard]] Scalar
chiSquaredCdf(std::type_identity_t<Scalar> const & x, std::type_identity_t<Scalar> const & dof)
{
    EMRYS_FILTER_ASSERT(dof > Scalar(0) && "the degrees of freedom must be positive");
    return internal::regularizedGammaP<Scalar>(dof / Scalar(2), x / Scalar(2));
}

/** \ingroup Core
 * \returns the \a probability quantile of the chi-squared distribution with \a dof degrees of
 * freedom: the NIS that a consistent filter's innovations stay under with that probability.
 *
 * Solved by safeguarded Newton iteration on chiSquaredCdf(), to within four epsilons of
 * \a Scalar: the root is bracketed by doubling, each Newton step uses the chi-squared density as
 * the derivative, and a step that leaves the bracket, or shrinks less than half as fast as the
 * step before last, is replaced by bisection (\ref Press2007 "Press et al. (2007)", rtsafe).
 * A quantile far below one, as a small probability with a degree of freedom or less gives, is
 * bracketed from below by squaring a ratio and bisected geometrically, so that it takes as few
 * steps as one near one; a quantile below the smallest normal \a Scalar comes out at most that.
 * \a Scalar is not deduced, as for chiSquaredCdf().
 *
 * A probability outside (0, 1) is a programmer error. Where asserts are off it still returns,
 * as the limits do: zero for a probability of zero or less, infinity for one or more, and NaN for
 * a NaN probability or degrees of freedom that are not positive.
 */
template <typename Scalar>
[[nodiscard]] Scalar
chiSquaredQuantile(std::type_identity_t<Scalar> const & probability, std::type_identity_t<Scalar> const & dof)
{
    EMRYS_FILTER_ASSERT(probability > Scalar(0) && probability < Scalar(1) && "the probability must be in (0, 1)");
    EMRYS_FILTER_ASSERT(dof > Scalar(0) && "the degrees of freedom must be positive");
    if (!(dof > Scalar(0)) || isNaN(probability))
    {
        return Eigen::NumTraits<Scalar>::quiet_NaN();
    }
    if (probability <= Scalar(0))
    {
        return Scalar(0);
    }
    if (probability >= Scalar(1))
    {
        return Eigen::NumTraits<Scalar>::infinity();
    }
    Scalar const epsilon = Scalar(Eigen::NumTraits<Scalar>::epsilon());
    Scalar const a = dof / Scalar(2);
    Scalar const logNormalizer = Scalar(a * Eigen::numext::log(Scalar(2))) + logGamma<Scalar>(a);
    Scalar lower = Scalar(0);
    Scalar upper = dof < Scalar(1) ? Scalar(1) : dof;
    while (chiSquaredCdf<Scalar>(upper, dof) < probability)
    {
        lower = upper;
        upper *= Scalar(2);
        // A probability that rounds to one below the cdf's last representable step has no finite
        // bracket.
        if (!isFinite(upper))
        {
            return upper;
        }
    }
    // A small probability with few degrees of freedom puts the quantile many orders of magnitude
    // below one, so the bracket closes from below by a ratio squared at each step, and bisects
    // geometrically while its ends differ by more than a factor of two.
    if (lower == Scalar(0))
    {
        Scalar ratio = Scalar(0.5);
        Scalar next = upper * ratio;
        while (Scalar(0) < next && !(chiSquaredCdf<Scalar>(next, dof) < probability))
        {
            upper = next;
            ratio *= ratio;
            next = upper * ratio;
        }
        lower = next;
    }
    auto const middle = [&] {
        return Scalar(2) * lower < upper && Scalar(0) < lower
                 ? Scalar(Eigen::numext::sqrt(lower) * Eigen::numext::sqrt(upper))
                 : Scalar((lower + upper) / Scalar(2));
    };
    Scalar x = middle();
    Scalar step = upper - lower;
    Scalar stepBefore = step;
    for (int i = 0; i < 200; ++i)
    {
        Scalar const error = chiSquaredCdf<Scalar>(x, dof) - probability;
        if (error < Scalar(0))
        {
            lower = x;
        }
        else
        {
            upper = x;
        }
        Scalar const exponent = (a - Scalar(1)) * Eigen::numext::log(x) - x / Scalar(2) - logNormalizer;
        Scalar const density = Eigen::numext::exp(exponent);
        // Newton's step, unless it leaves the bracket or is more than half the step before last,
        // as it is far down a tail, where it shrinks the bracket less than bisection would.
        Scalar next = x - error / density;
        if (!(next > lower && next < upper)
            || Eigen::numext::abs(Scalar(Scalar(2) * error)) > Eigen::numext::abs(Scalar(stepBefore * density)))
        {
            next = middle();
        }
        stepBefore = step;
        step = next - x;
        bool const converged = Eigen::numext::abs(step) <= Scalar(4) * epsilon * x;
        x = next;
        if (converged)
        {
            break;
        }
    }
    return x;
}

}  // namespace Emrys
