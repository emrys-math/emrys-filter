#pragma once

/** \file
 * Params and Density: the traits that pair a distribution's parameters with its density, the
 * type that evaluates it (decisions.md [params-and-density]). GaussianParams holds a %Gaussian's parameters and
 * Gaussian evaluates its density, so Params<Gaussian<Space>> is GaussianParams<Space> and
 * Density<GaussianParams<Space>> is Gaussian<Space>; a type whose values are always valid, such
 * as Uniform, is both. Filters, mixtures, and hybrids are named by the density and store its
 * Params. Why a distribution has two types is on the page of the Distributions group, under
 * \ref ParamsAndDensities "Parameters and densities". Read Moments.h next.
 */

namespace Emrys
{

/** \ingroup Distributions
 * The parameters of the density \a D, as its member \c ParamsType: the type it borrows from, or
 * what a marginal's copy() returns. A type with no \c ParamsType is its own parameters
 * (decisions.md [params-and-density]). Use the alias Params.
 */
template <typename D>
struct ParamsOf
{
    using type = D;  ///< the parameters: \a D itself
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename D>
    requires requires { typename D::ParamsType; }
struct ParamsOf<D>
{
    using type = typename D::ParamsType;
};

/// \endcond

/** \ingroup Distributions
 * The parameters of the density \a D: GaussianParams for a Gaussian, and \a D itself for a
 * distribution whose values are always valid (decisions.md [params-and-density]).
 */
template <typename D>
using Params = typename ParamsOf<D>::type;

/** \ingroup Distributions
 * The density of the parameters \a P, as its member \c DensityType: what its toDensity()
 * returns. A type with no \c DensityType is its own density (decisions.md [params-and-density]). Use the alias
 * Density.
 */
template <typename P>
struct DensityOf
{
    using type = P;  ///< the density: \a P itself
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename P>
    requires requires { typename P::DensityType; }
struct DensityOf<P>
{
    using type = typename P::DensityType;
};

/// \endcond

/** \ingroup Distributions
 * The density of the parameters \a P: Gaussian for GaussianParams, and \a P itself for a type
 * that is its own parameters (decisions.md [params-and-density]). Deduction guides use it to name a filter or
 * mixture by the density when they are given its parameters.
 */
template <typename P>
using Density = typename DensityOf<P>::type;

}  // namespace Emrys
