#pragma once

/** \file
 * NumericError, the reasons a factorization or a normalization can fail, and its two names,
 * GaussianError and WeightError (decisions.md [update-outcomes]). It is one enumeration so that a
 * step that can fail both ways, as a mixture filter's update fails on a component's innovation
 * covariance or on the mixture's weights, returns one error type that every comparison of either
 * name reads; and internal::isExpected, which tells a std::expected from a value, for code that
 * calls a function that may fail and one that may not alike. Read Weights.h next.
 */

#include <expected>

namespace Emrys
{

/** \ingroup Core
 * Reasons a computation from parameters the caller gave can fail: a matrix that does not factorize
 * or a set of log-weights that does not normalize. A function returns the ones its documentation
 * names, through either name below; a step that can fail both ways, such as a mixture filter's
 * update, may return any of them (decisions.md [update-outcomes]).
 *
 * A %Gaussian's errors name the matrix that failed: a covariance that is singular or indefinite, as
 * a square-root factor with a zero on its diagonal gives, is NotPositiveDefinite; a precision that
 * is, as a prior with no information has, is Singular.
 *
 * \sa GaussianError, WeightError
 */
enum class NumericError
{
    NotPositiveDefinite,  ///< the covariance is not positive definite, so it has no Cholesky factor or inverse
    Singular,             ///< the precision is not positive definite, so the mean and covariance are undefined
    NonFinite,            ///< a matrix, a factor, or a log-weight holds NaN or infinity (+inf, for a log-weight)
    Empty,                ///< there are no weights
    ZeroSum,              ///< every weight is zero (every log-weight is -inf)
};

/** \ingroup Distributions
 * Reasons a %Gaussian, in any of its forms (decisions.md [gaussian-forms]), can fail to give a
 * density, a mean, or one of its matrices, and a root of a covariance can fail to be taken:
 * NotPositiveDefinite, Singular, or NonFinite of NumericError.
 *
 * \sa GaussianParams::toDensity(), GaussianForm, roots::squareRoot()
 */
using GaussianError = NumericError;

/** \ingroup Core
 * Reasons a collection of log-weights cannot be normalized: Empty, ZeroSum, or NonFinite of
 * NumericError.
 *
 * \sa GaussianMixtureParams::logTotalWeight(), ParticleSet::logTotalWeight()
 */
using WeightError = NumericError;

namespace internal
{

/** Whether \a T is a std::expected, as a function that may fail returns. */
template <typename T>
inline constexpr bool isExpected = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename T, typename E>
inline constexpr bool isExpected<std::expected<T, E>>{true};
/// \endcond

}  // namespace internal

}  // namespace Emrys
