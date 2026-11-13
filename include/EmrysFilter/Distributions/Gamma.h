#pragma once

/** \file
 * Gamma: independent Gamma distributions on the coordinates, with the exponential and
 * chi-squared as special cases and a parameterization by mean and standard deviation, and its
 * draws; and GammaError, why parameters fail to define one. An immutable value, its own
 * parameters, built only by factories that check them (decisions.md [params-and-density], [checked-factories]). Read
 * StudentT.h next.
 */

#include <cmath>
#include <concepts>
#include <expected>
#include <limits>
#include <optional>
#include <random>
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Distributions
 * Reasons a shape and scale fail to define a Gamma.
 */
enum class GammaError
{
    NotPositive,  ///< a shape or scale is zero or negative
    NonFinite,    ///< a shape or scale is NaN or infinite
};

namespace internal
{

template <typename S>
consteval bool
holdsHalfLine();

/** Whether every one of the blocks in the tuple type \a Blocks holds <tt>[0, ∞)</tt>. */
template <typename Blocks>
struct AllHoldHalfLine;

/** Whether every one of \a Blocks holds <tt>[0, ∞)</tt>. */
template <typename... Blocks>
struct AllHoldHalfLine<std::tuple<Blocks...>> : std::bool_constant<(holdsHalfLine<Blocks>() && ...)>
{};

/** \returns whether every leaf block of \a S contains <tt>[0, ∞)</tt> without folding it: a
 * Linear one does, a Periodic one doesn't, and a non-linear space written from scratch, whose
 * range the library can't see, doesn't either.
 */
template <typename S>
consteval bool
holdsHalfLine()
{
    if constexpr (IsComposite<S>)
    {
        return AllHoldHalfLine<CompositeBlocks<S>>::value;
    }
    else if constexpr (IsRepeated<S>)
    {
        return holdsHalfLine<RepeatedBlock<S>>();
    }
    else
    {
        constexpr auto support = leafSupport<S>();
        return support && !support->periodic && support->lower <= 0
            && support->upper == std::numeric_limits<double>::infinity();
    }
}

}  // namespace internal

/** \ingroup Distributions
 * \brief Independent Gamma distributions on the coordinates, for positive quantities such
 * as ranges, speeds, and rates.
 *
 * The [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) in its shape and
 * scale parameterization. Coordinate \f$ i \f$ has shape \f$ k_i \f$ and scale \f$ \theta_i \f$:
 * \f[ p(x_i) = \frac{x_i^{k_i - 1} e^{-x_i / \theta_i}}{\Gamma(k_i)\, \theta_i^{k_i}}, \quad x_i \ge 0, \f]
 * with mean \f$ k_i \theta_i \f$ and variance \f$ k_i \theta_i^2 \f$. A shape of 1 is the
 * exponential distribution (see exponential()); shape \f$ \nu / 2 \f$ with scale 2 is
 * \f$ \chi^2_\nu \f$ (see chiSquared()). A shape above 1 puts the mode at
 * \f$ (k - 1) \theta \f$, away from zero; fromMeanAndStdDev() picks the shape and scale for a
 * given mean and spread.
 *
 * \code
 * auto const speed = Gamma<Speed>::create(4.0, 0.5);           // or a GammaError
 * auto const rate = Gamma<Rate>::fromMeanAndStdDev(2.0, 0.5);  // shape 16, scale 0.125
 * \endcode
 *
 * Every coordinate's space must contain all of <tt>[0, ∞)</tt> without wrapping it, as Linear
 * does. The support is the distribution's own: the density is zero below zero, and no space
 * keeps a value from going there [no-bounded-coordinates].
 *
 * A Gamma is an immutable value that owns its parameters. Its factories check them, since a
 * shape or scale is often computed at run time, and return a GammaError if one is not positive
 * and finite; it has no public constructor [checked-factories].
 *
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 *
 * \sa drawSamples(Gamma const &, Eigen::Index, G &)
 */
template <Space Space_, RealScalar Scalar_ = double>
class Gamma
{
public:

    using Scalar = Scalar_;                               ///< the scalar type
    using Index = Eigen::Index;                           ///< Eigen's index type
    using SpaceType = Space_;                             ///< the space the values live in
    using MomentsType = Moments<Space_, Scalar_>;         ///< a mean and covariance together
    using VectorType = typename MomentsType::VectorType;  ///< a value
    using MatrixType = typename MomentsType::MatrixType;  ///< a matrix over differences

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static_assert(
        internal::hasEqualSizes<Space_>,
        "a gamma distribution needs a space whose values and differences have the same coordinates"
    );
    static_assert(
        internal::holdsHalfLine<Space_>(),
        "a gamma distribution needs every coordinate to contain [0, inf) without folding it"
    );

    /** \returns the Gamma with shape \a shape and scale \a scale in \a space, or a GammaError if a
     * shape or scale is not positive and finite. Each is a vector with one entry per coordinate, or
     * a scalar shared by every coordinate. A space with one resizable block is resized to a vector
     * parameter's size; with only scalars it keeps its own size.
     */
    template <typename Shape, typename Scale>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    create(Space_ space, Shape const & shape, Scale const & scale)
    {
        Index const n = parameterSize(shape, scale, Emrys::valueDimension(space));
        internal::fitValueDimension(space, n);
        VectorType shapes = broadcast(shape, n);
        VectorType scales = broadcast(scale, n);
        if (auto const error = check(shapes, scales))
        {
            return std::unexpected{*error};
        }
        return Gamma{std::move(space), std::move(shapes), std::move(scales)};
    }

    /** create() in the default space. */
    template <typename Shape, typename Scale>
        requires internal::SizedByLength<Space_>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    create(Shape const & shape, Scale const & scale)
    {
        return create(Space_{}, shape, scale);
    }

    /** \returns the exponential distribution with mean \a scale (rate \f$ 1 / \theta \f$) in
     * \a space, or a GammaError if a scale is not positive and finite
     */
    template <typename Scale>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    exponential(Space_ space, Scale const & scale)
    {
        return create(std::move(space), Scalar(1), scale);
    }

    /** exponential() in the default space. */
    template <typename Scale>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    exponential(Scale const & scale)
    {
        return exponential(Space_{}, scale);
    }

    /** \returns \f$ \chi^2 \f$ with \a dof degrees of freedom in \a space: shape \f$ \nu / 2 \f$,
     * scale 2; or a GammaError if a number of degrees of freedom is not positive and finite
     */
    template <typename Dof>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    chiSquared(Space_ space, Dof const & dof)
    {
        Index const n = parameterSize(dof, Scalar(2), Emrys::valueDimension(space));
        return create(std::move(space), VectorType{Scalar(0.5) * broadcast(dof, n)}, Scalar(2));
    }

    /** chiSquared() in the default space. */
    template <typename Dof>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    chiSquared(Dof const & dof)
    {
        return chiSquared(Space_{}, dof);
    }

    /** \returns the Gamma distribution with mean \a mean and standard deviation \a stdDev in
     * \a space: shape \f$ (\mu / \sigma)^2 \f$ and scale \f$ \sigma^2 / \mu \f$. A mean or standard
     * deviation that is not positive and finite gives a shape or scale that is not either, and so a
     * GammaError.
     */
    template <typename Mean, typename StdDev>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    fromMeanAndStdDev(Space_ space, Mean const & mean, StdDev const & stdDev)
    {
        Index const n = parameterSize(mean, stdDev, Emrys::valueDimension(space));
        VectorType const mu = broadcast(mean, n);
        VectorType const sigma = broadcast(stdDev, n);
        if (!allFinite(mu) || !allFinite(sigma))
        {
            return std::unexpected{GammaError::NonFinite};
        }
        if (!(mu.array() > Scalar(0)).all() || !(sigma.array() > Scalar(0)).all())
        {
            return std::unexpected{GammaError::NotPositive};
        }
        return create(
            std::move(space),
            VectorType{(mu.array() / sigma.array()).square().matrix()},
            VectorType{(sigma.array().square() / mu.array()).matrix()}
        );
    }

    /** fromMeanAndStdDev() in the default space. */
    template <typename Mean, typename StdDev>
    [[nodiscard]] static std::expected<Gamma, GammaError>
    fromMeanAndStdDev(Mean const & mean, StdDev const & stdDev)
    {
        return fromMeanAndStdDev(Space_{}, mean, stdDev);
    }

    /** \returns the number of coordinates */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_shape.size();
    }

    /** \returns the same: the coordinates have plain differences */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return m_shape.size();
    }

    /** \returns the space the coordinates are in, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the shape \f$ k \f$ of each coordinate */
    [[nodiscard]] VectorType const &
    shape() const
    {
        return m_shape;
    }

    /** \returns the scale \f$ \theta \f$ of each coordinate */
    [[nodiscard]] VectorType const &
    scale() const
    {
        return m_scale;
    }

    /** \returns the log of the density at \a x, \f$ -\infty \f$ if any coordinate is negative */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "logDensity() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        Scalar result = m_logNormalizer;
        for (Index i = 0; i < valueDimension(); ++i)
        {
            Scalar const xi = x(i);
            if (xi < Scalar(0))
            {
                return -std::numeric_limits<Scalar>::infinity();
            }
            // Skip the power term for shape 1, where it is 0 · log 0 at the origin.
            if (m_shape(i) != Scalar(1))
            {
                result += (m_shape(i) - Scalar(1)) * Eigen::numext::log(xi);
            }
            result -= xi / m_scale(i);
        }
        return result;
    }

    /** \returns the density at \a x */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** \returns the mean \f$ k \theta \f$ and the diagonal covariance \f$ k \theta^2 \f$ */
    [[nodiscard]] MomentsType
    moments() const
    {
        auto result = MomentsType{valueDimension(), differenceDimension()};
        result.mean() = m_shape.cwiseProduct(m_scale);
        result.covarianceLower() = result.mean().cwiseProduct(m_scale).asDiagonal();
        return result;
    }

    /** marginal() of the subspace named by the tag path \a Path (see idx()), in that subspace. */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::marginalOverPath<Path>(m_space, [&](auto const & target, auto space) {
            return restricted(target.values, std::move(space));
        });
    }

    /** marginal() of one repetition of a repeated block; see GaussianParams::marginal(element). */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const
    {
        return internal::marginalOverElement<Path>(m_space, element, [&](auto const & target, auto space) {
            return restricted(target.values, std::move(space));
        });
    }

    /** \returns the distribution of the other coordinates given that the subspace named by the
     * path \a GivenPath equals \a values, in the space of the remaining blocks (see
     * GaussianParams::conditional()): since the coordinates are independent, their marginal.
     * Negative values have zero density, so strictly the conditional is undefined there; this
     * returns the marginal anyway.
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditional<GivenPath>(
            m_space,
            [&](auto const & t, auto const & g, auto space, auto const &) {
                EMRYS_FILTER_ASSERT(values.size() == g.values.size());
                return restricted(t.values, std::move(space));
            }
        );
    }

private:

    /** Marginals build another instantiation from checked parts. */
    template <Space, RealScalar>
    friend class Gamma;

    /** The Gamma with the checked shapes \a shape and scales \a scale in \a space, which has
     * their size; only create() and the marginals call this.
     */
    Gamma(Space_ space, VectorType shape, VectorType scale)
        : m_space{std::move(space)}
        , m_shape{std::move(shape)}
        , m_scale{std::move(scale)}
    {
        for (Index i = 0; i < m_shape.size(); ++i)
        {
            m_logNormalizer -= logGamma<Scalar>(m_shape(i)) + m_shape(i) * Eigen::numext::log(m_scale(i));
        }
    }

    /** \returns the GammaError of \a shape and \a scale, or nothing if they are positive and finite */
    [[nodiscard]] static std::optional<GammaError>
    check(VectorType const & shape, VectorType const & scale)
    {
        if (!allFinite(shape) || !allFinite(scale))
        {
            return GammaError::NonFinite;
        }
        if (!(shape.array() > Scalar(0)).all() || !(scale.array() > Scalar(0)).all())
        {
            return GammaError::NotPositive;
        }
        return std::nullopt;
    }

    /** \returns the shapes and scales of the coordinates \a target, which form \a space */
    template <typename Target, typename SubSpace>
    [[nodiscard]] Gamma<SubSpace, Scalar>
    restricted(Target const & target, SubSpace space) const
    {
        using ResultVector = typename Gamma<SubSpace, Scalar>::VectorType;
        return Gamma<SubSpace, Scalar>{std::move(space), ResultVector{m_shape(target)}, ResultVector{m_scale(target)}};
    }

    /** Whether a parameter \a P is one number shared by every coordinate rather than a vector. */
    template <typename P>
    static constexpr bool isScalar = std::is_arithmetic_v<P>;

    /** \returns the size of whichever parameter is a vector, or \a spaceSize if both are scalars */
    template <typename A, typename B>
    [[nodiscard]] static Index
    parameterSize(A const & a, B const & b, Index spaceSize)
    {
        if constexpr (!isScalar<A>)
        {
            if constexpr (!isScalar<B>)
            {
                EMRYS_FILTER_ASSERT(a.size() == b.size() && "the parameters differ in size");
            }
            return a.size();
        }
        else if constexpr (!isScalar<B>)
        {
            return b.size();
        }
        else
        {
            return spaceSize;
        }
    }

    /** \returns the parameter \a p as a vector of \a n entries: repeated if it is one number */
    template <typename P>
    [[nodiscard]] static VectorType
    broadcast(P const & p, Index n)
    {
        if constexpr (isScalar<P>)
        {
            return VectorType::Constant(n, Scalar(p));
        }
        else
        {
            EMRYS_FILTER_ASSERT(p.size() == n);
            return VectorType{p};
        }
    }

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space the coordinates are in
    VectorType m_shape;                             ///< each coordinate's shape \f$ k \f$
    VectorType m_scale;                             ///< each coordinate's scale \f$ \theta \f$
    Scalar m_logNormalizer = 0;                     ///< \f$ -\sum_i (\log\Gamma(k_i) + k_i \log\theta_i) \f$
};

/** \ingroup Distributions
 * \returns \a n independent draws from \a gamma, one per column, by <tt>std::gamma_distribution</tt>
 * on each coordinate.
 *
 * \sa drawSamplesWithLogDensity(Gamma const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Space, Scalar>
drawSamples(Gamma<Space, Scalar> const & gamma, Eigen::Index n, G & rng)
{
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    EMRYS_FILTER_ASSERT(n >= 0);
    using Distribution = std::gamma_distribution<Scalar>;
    constexpr int maxDimension = Gamma<Space, Scalar>::MaxValueDimensionAtCompileTime;
    // One distribution per coordinate, inline for a bounded space (see \ref Allocation).
    using Distributions = std::conditional_t<
        maxDimension == Eigen::Dynamic,
        std::vector<Distribution>,
        std::array<Distribution, std::size_t(maxDimension == Eigen::Dynamic ? 0 : maxDimension)>>;
    Eigen::Index const d = gamma.valueDimension();
    auto distributions = Distributions();
    if constexpr (maxDimension == Eigen::Dynamic)
    {
        distributions.resize(std::size_t(d));
    }
    for (Eigen::Index i = 0; i < d; ++i)
    {
        distributions[std::size_t(i)] = Distribution{gamma.shape()(i), gamma.scale()(i)};
    }
    auto samples = SampleSet<Space, Scalar>{d, n};
    for (Eigen::Index j = 0; j < n; ++j)
    {
        for (Eigen::Index i = 0; i < d; ++i)
        {
            samples(i, j) = distributions[std::size_t(i)](rng);
        }
    }
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a gamma, by <tt>std::gamma_distribution</tt> on each coordinate, in a
 * vector sized by the space: the stream of draws that drawSample() of a %Gaussian is. It makes each
 * coordinate's distribution afresh, where drawSamples() keeps one per coordinate, with whatever
 * state it carries, from draw to draw.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(Gamma<Space, Scalar> const & gamma, G & rng)
{
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    auto sample = typename Moments<Space, Scalar>::VectorType{};
    sample.resize(gamma.valueDimension());
    for (Eigen::Index i = 0; i < sample.size(); ++i)
    {
        sample(i) = std::gamma_distribution<Scalar>{gamma.shape()(i), gamma.scale()(i)}(rng);
    }
    return sample;
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a gamma, one per column, with the log-density at each.
 *
 * \sa drawSamples(Gamma const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(Gamma<Space, Scalar> const & gamma, Eigen::Index n, G & rng)
{
    auto result = SamplesWithLogDensity<Space, Scalar>{drawSamples(gamma, n, rng), {}};
    result.logDensity.resize(n);
    for (Eigen::Index j = 0; j < n; ++j)
    {
        result.logDensity(j) = gamma.logDensity(result.samples.col(j));
    }
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a gamma, as drawSample() draws it, with the log-density at it, in a
 * vector sized by the space: the stream of draws that drawSampleWithLogDensity() of a %Gaussian is.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(Gamma<Space, Scalar> const & gamma, G & rng)
{
    auto result = SampleWithLogDensity<Space, Scalar>{drawSample(gamma, rng), 0};
    result.logDensity = gamma.logDensity(result.sample);
    return result;
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes a GammaError in words. */
inline void
describe(std::ostream & stream, GammaError error, Indent indent)
{
    stream << indent;
    switch (error)
    {
        case GammaError::NotPositive:
            stream << "a shape or scale is not positive";
            break;
        case GammaError::NonFinite:
            stream << "a shape or scale is not finite";
            break;
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
