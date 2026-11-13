#pragma once

/** \file
 * Uniform: the uniform distribution over a box, with the box taken from a space of
 * Periodic coordinates or given to create(), which checks it, and its draws; and UniformError,
 * why bounds fail to define one. An immutable value, its own parameters (decisions.md [params-and-density],
 * [checked-factories]). Read Gamma.h next.
 */

#include <cmath>
#include <expected>
#include <limits>
#include <optional>
#include <random>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
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
 * Reasons bounds fail to define a Uniform.
 */
enum class UniformError
{
    Empty,            ///< an upper bound is not above its lower bound
    NonFinite,        ///< a bound is NaN or infinite
    WiderThanPeriod,  ///< an interval on a Periodic coordinate spans more than one period
};

/** \ingroup Distributions
 * \brief The uniform distribution over a box in a space.
 *
 * Each coordinate \f$ i \f$ is
 * [uniform](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) on an interval of
 * width \f$ w_i \f$ starting at \f$ \ell_i \f$, independently, so the density is
 * \f$ 1 / \prod_i w_i \f$ inside the box and zero outside.
 *
 * The space supplies the default box where it can: a Periodic coordinate is uniform over one
 * whole period, and that constructor cannot fail. A Linear coordinate has no default and needs a
 * box, which create() checks, since bounds are often computed at run time.
 * A Periodic coordinate's interval may cross the seam: <tt>[3, 3.5]</tt> on an angle covers
 * <tt>[3, π)</tt> and <tt>[-π, 3.5 - 2π]</tt>.
 *
 * \code
 * auto const heading = Uniform<Heading>{};                                          // one whole period
 * auto const box = Uniform<Position>::create(Eigen::Vector2d{0, 0}, Eigen::Vector2d{10, 5});  // or a UniformError
 * \endcode
 *
 * A Uniform is an immutable value that owns its box [checked-factories].
 *
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 * \tparam Scalar_  scalar type
 *
 * \sa drawSamples(Uniform const &, Eigen::Index, G &)
 */
template <Space Space_, RealScalar Scalar_ = double>
class Uniform
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
        "a uniform box needs a space whose values and differences have the same coordinates"
    );

    /** Uniform over the whole of \a space. Every coordinate must be Periodic. */
    explicit Uniform(Space_ space = {})
        requires(!isLinear<Space_>)
        : m_space{std::move(space)}
    {
        Index const n = Emrys::valueDimension(m_space);
        m_lower.resize(n);
        m_width.resize(n);
        m_period.resize(n);
        internal::forEachLeaf(m_space, [&](Index offset, auto const & leaf) {
            using Leaf = std::remove_cvref_t<decltype(leaf)>;
            static_assert(hasDefaultInterval<Leaf>(), "a coordinate has no default interval; give the box explicitly");
            constexpr auto found = internal::leafSupport<Leaf>();
            constexpr internal::LeafSupport support = *found;
            constexpr double width = support.upper - support.lower;
            auto const rows = Eigen::seqN(offset, leaf.valueDimension());
            m_lower(rows).setConstant(Scalar(support.lower));
            m_width(rows).setConstant(Scalar(width));
            // Only a Periodic coordinate has a default interval.
            static_assert(support.periodic);
            m_period(rows).setConstant(Scalar(width));
        });
        m_logVolume = m_width.array().log().sum();
    }

    /** \returns the uniform distribution over the box from \a lower to \a upper in \a space, moving
     * \a lower if it is an rvalue of the stored type, or a UniformError if a bound is not finite,
     * an upper bound is not above its lower bound, or an interval on a Periodic coordinate is
     * wider than one period. A space with one resizable block is resized to fit.
     */
    template <MatrixExpression Lower, typename UpperDerived>
    [[nodiscard]] static std::expected<Uniform, UniformError>
    create(Space_ space, Lower && lower, Eigen::MatrixBase<UpperDerived> const & upper)
    {
        VectorType start = std::forward<Lower>(lower);
        EMRYS_FILTER_ASSERT(upper.size() == start.size() && "give one upper bound per lower bound");
        VectorType width = upper - start;
        internal::fitValueDimension(space, start.size());
        if (!allFinite(start) || !allFinite(width))
        {
            return std::unexpected{UniformError::NonFinite};
        }
        if (!(width.array() > Scalar(0)).all())
        {
            return std::unexpected{UniformError::Empty};
        }
        auto period = VectorType{VectorType::Zero(start.size())};
        bool fits = true;
        internal::forEachLeaf(space, [&](Index offset, auto const & leaf) {
            constexpr auto support = internal::leafSupport<std::remove_cvref_t<decltype(leaf)>>();
            if constexpr (support.has_value() && support->periodic)
            {
                constexpr double leafPeriod = support->upper - support->lower;
                for (Index i = offset; i < offset + leaf.valueDimension(); ++i)
                {
                    fits = fits && width(i) <= Scalar(leafPeriod);
                    period(i) = Scalar(leafPeriod);
                }
            }
        });
        if (!fits)
        {
            return std::unexpected{UniformError::WiderThanPeriod};
        }
        return Uniform{std::move(space), std::move(start), std::move(width), std::move(period)};
    }

    /** create() in the default space. */
    template <MatrixExpression Lower, typename UpperDerived>
        requires internal::SizedByLength<Space_>
    [[nodiscard]] static std::expected<Uniform, UniformError>
    create(Lower && lower, Eigen::MatrixBase<UpperDerived> const & upper)
    {
        return create(Space_{}, std::forward<Lower>(lower), upper);
    }

    /** \returns the number of coordinates of the box */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_lower.size();
    }

    /** \returns the same: the box has plain differences */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return m_lower.size();
    }

    /** \returns the space the box is in, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the start \f$ \ell \f$ of each coordinate's interval, as given */
    [[nodiscard]] VectorType const &
    lower() const
    {
        return m_lower;
    }

    /** \returns the end \f$ \ell + w \f$ of each coordinate's interval, not normalized: on a
     * Periodic coordinate it may lie past the space's range
     */
    [[nodiscard]] VectorType
    upper() const
    {
        return m_lower + m_width;
    }

    /** \returns the width \f$ w \f$ of each coordinate's interval */
    [[nodiscard]] VectorType const &
    width() const
    {
        return m_width;
    }

    /** \returns the log of the box's volume, \f$ \sum_i \log w_i \f$ */
    [[nodiscard]] Scalar
    logVolume() const
    {
        return m_logVolume;
    }

    /** \returns whether \a x lies in the box, taking a Periodic coordinate's interval around the circle */
    template <typename Derived>
    [[nodiscard]] bool
    contains(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "contains() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        for (Index i = 0; i < valueDimension(); ++i)
        {
            Scalar offset = Scalar(x(i)) - m_lower(i);
            if (m_period(i) > Scalar(0))
            {
                offset -= m_period(i) * floorOf<Scalar>(offset / m_period(i));
            }
            if (!(offset >= Scalar(0) && offset <= m_width(i)))
            {
                return false;
            }
        }
        return true;
    }

    /** \returns the log of the density at \a x: \f$ -\log \prod_i w_i \f$ inside the box, and
     * \f$ -\infty \f$ outside
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        return contains(x) ? -m_logVolume : -std::numeric_limits<Scalar>::infinity();
    }

    /** \returns the density at \a x */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** \returns the mean and covariance: the box's center, normalized, and
     * \f$ \mathrm{diag}(w_i^2 / 12) \f$, the variance of a uniform interval of width \f$ w_i \f$.
     *
     * For a Periodic coordinate these are tangent-space moments about the interval's center.
     * Over a whole period every point is equally a center, so the reported mean is arbitrary
     * and the variance \f$ p^2 / 12 \f$ only says the coordinate is unknown.
     */
    [[nodiscard]] MomentsType
    moments() const
    {
        auto result = MomentsType{valueDimension(), differenceDimension()};
        result.mean() = m_lower + Scalar(0.5) * m_width;
        normalizeValues(m_space, result.mean());
        result.covarianceLower() = (m_width.array().square() / Scalar(12)).matrix().asDiagonal();
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
     * GaussianParams::conditional()): since the coordinates are independent, their marginal. Values
     * outside the box have zero density, so strictly the conditional is undefined there; this
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
    friend class Uniform;

    /** Whether every coordinate of the leaf block \a Leaf has an interval of its own: one
     * period of a Periodic block.
     */
    template <typename Leaf>
    static consteval bool
    hasDefaultInterval()
    {
        constexpr auto support = internal::leafSupport<Leaf>();
        return support && support->lower > -std::numeric_limits<double>::infinity()
            && support->upper < std::numeric_limits<double>::infinity();
    }

    /** The box with the given intervals and periods, already checked; only create() and the
     * marginals call this.
     */
    Uniform(Space_ space, VectorType lower, VectorType width, VectorType period)
        : m_space{std::move(space)}
        , m_lower{std::move(lower)}
        , m_width{std::move(width)}
        , m_period{std::move(period)}
        , m_logVolume{m_width.array().log().sum()}
    {}

    /** \returns this box's intervals on the coordinates \a target, which form \a space */
    template <typename Target, typename SubSpace>
    [[nodiscard]] Uniform<SubSpace, Scalar>
    restricted(Target const & target, SubSpace space) const
    {
        using Result = Uniform<SubSpace, Scalar>;
        using ResultVector = typename Result::VectorType;
        return Result{
            std::move(space),
            ResultVector{m_lower(target)},
            ResultVector{m_width(target)},
            ResultVector{m_period(target)}
        };
    }

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space the box is in
    VectorType m_lower;                             ///< each interval's start \f$ \ell \f$
    VectorType m_width;                             ///< each interval's width \f$ w \f$
    VectorType m_period;                            ///< a Periodic coordinate's period, or zero
    Scalar m_logVolume{};                           ///< \f$ \sum_i \log w_i \f$
};

/** \ingroup Distributions
 * \returns \a n independent draws from \a uniform, one per column: \f$ \ell + w \circ u \f$
 * with \f$ u \f$ uniform on <tt>[0, 1)</tt>, normalized in the space.
 *
 * \sa drawSamplesWithLogDensity(Uniform const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Space, Scalar>
drawSamples(Uniform<Space, Scalar> const & uniform, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using SampleSetType = SampleSet<Space, Scalar>;
    SampleSetType samples =
        uniform.width().asDiagonal() * internal::standardUniform<SampleSetType>(uniform.valueDimension(), n, rng);
    samples.colwise() += uniform.lower();
    normalizeValues(uniform.space(), samples);
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a uniform, \f$ \ell + w \circ u \f$ as drawSamples() draws each, in a
 * vector sized by the space: the stream of draws that drawSample() of a %Gaussian is. Its
 * coordinates take the uniform draws in the order drawSamples() gives each column.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(Uniform<Space, Scalar> const & uniform, G & rng)
{
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    using Vector = typename Moments<Space, Scalar>::VectorType;
    auto draw = std::uniform_real_distribution<Scalar>{};
    auto sample = Vector{};
    sample.resize(uniform.valueDimension());
    for (Eigen::Index i = 0; i < sample.size(); ++i)
    {
        sample(i) = uniform.lower()(i) + uniform.width()(i) * draw(rng);
    }
    normalizeValues(uniform.space(), sample);
    return sample;
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a uniform, one per column, with the log-density at
 * each, which is the same everywhere in the box.
 *
 * \sa drawSamples(Uniform const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(Uniform<Space, Scalar> const & uniform, Eigen::Index n, G & rng)
{
    return {drawSamples(uniform, n, rng), Eigen::Array<Scalar, Eigen::Dynamic, 1>::Constant(n, -uniform.logVolume())};
}

/** \ingroup Distributions
 * \returns one draw from \a uniform, as drawSample() draws it, with the log-density at it, the
 * same everywhere in the box, in a vector sized by the space: the stream of draws that
 * drawSampleWithLogDensity() of a %Gaussian is.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(Uniform<Space, Scalar> const & uniform, G & rng)
{
    return {drawSample(uniform, rng), -uniform.logVolume()};
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes a UniformError in words. */
inline void
describe(std::ostream & stream, UniformError error, Indent indent)
{
    stream << indent;
    switch (error)
    {
        case UniformError::Empty:
            stream << "an upper bound is not above its lower bound";
            break;
        case UniformError::NonFinite:
            stream << "a bound is not finite";
            break;
        case UniformError::WiderThanPeriod:
            stream << "an interval is wider than its coordinate's period";
            break;
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
