#pragma once

/** \file
 * StudentT: the multivariate Student-t, a %Gaussian with heavier tails, built by create()
 * from a location, scale matrix, and degrees of freedom, with its marginals, conditionals, and
 * draws; and StudentTError, why those fail to define one. An immutable value, its own parameters
 * (decisions.md [params-and-density], [checked-factories]), that holds a factorized %Gaussian. Read Product.h next.
 */

#include <cmath>
#include <concepts>
#include <expected>
#include <optional>
#include <random>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>

namespace Emrys
{

/** \ingroup Distributions
 * Reasons a location, scale matrix, and degrees of freedom fail to define a StudentT.
 */
enum class StudentTError
{
    NotPositiveDefinite,          ///< the scale matrix is not positive definite
    NonFiniteScale,               ///< the scale matrix is not finite
    NotPositiveDegreesOfFreedom,  ///< the degrees of freedom are zero or negative
    NonFiniteDegreesOfFreedom,    ///< the degrees of freedom are NaN or infinite
};

/** \ingroup Distributions
 * \brief The multivariate Student-t distribution: a %Gaussian with heavier tails.
 *
 * The [multivariate t-distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution).
 * With location \f$ \mu \f$, scale matrix \f$ \Sigma \f$, and \f$ \nu \f$ degrees of freedom in
 * \f$ d \f$ dimensions,
 * \f[ p(x) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) (\nu\pi)^{d/2}
 * |\Sigma|^{1/2}}
 *     \left(1 + \frac{\delta^2}{\nu}\right)^{-\frac{\nu + d}{2}}, \f]
 * where \f$ \delta^2 = (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu) \f$. It is a scale
 * mixture of Gaussians: \f$ x = \mu + L z / \sqrt{u / \nu} \f$ with \f$ z \sim \mathcal{N}(0, I) \f$,
 * \f$ u \sim \chi^2_\nu \f$, and \f$ \Sigma = L L^T \f$. As \f$ \nu \to \infty \f$ it approaches
 * \f$ \mathcal{N}(\mu, \Sigma) \f$; \f$ \nu = 1 \f$ is the Cauchy distribution.
 *
 * The scale matrix is not the covariance: the covariance is \f$ \frac{\nu}{\nu - 2} \Sigma \f$,
 * and only exists for \f$ \nu > 2 \f$ (see moments()). Densities use the nearest image, as
 * a %Gaussian's do.
 *
 * Created by create(), which checks the degrees of freedom and factorizes the scale matrix, and
 * returns a StudentTError if either fails; it has no public constructor [checked-factories]. A StudentT is an
 * immutable value that owns its parameters.
 *
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 * \tparam Scalar_  scalar type
 *
 * \sa drawSamples(StudentT const &, Eigen::Index, G &)
 */
template <Space Space_, RealScalar Scalar_ = double>
class StudentT
{
public:

    /** \f$ \mathcal{N}(\mu, \Sigma) \f$, which holds the parameters */
    using ScaleGaussianType = GaussianParams<Space_, Scalar_>;

    using MomentsType = typename ScaleGaussianType::MomentsType;  ///< a mean and covariance together
    using Scalar = Scalar_;                                       ///< the scalar type
    using Index = Eigen::Index;                                   ///< Eigen's index type
    using SpaceType = Space_;                                     ///< the space the values live in
    using VectorType = typename MomentsType::VectorType;          ///< a value
    using MatrixType = typename MomentsType::MatrixType;          ///< a matrix over differences

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the scale matrix, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    /** \returns a Student-t with location \a location, scale matrix \a scale, and \a dof degrees of
     * freedom in \a space, or a StudentTError if \a dof is not positive and finite, or \a scale is
     * not positive definite or not finite. Only the lower triangle of \a scale is read.
     */
    template <MatrixExpression Location, typename ScaleDerived>
    [[nodiscard]] static std::expected<StudentT, StudentTError>
    create(Space_ space, Location && location, Eigen::MatrixBase<ScaleDerived> const & scale, Scalar dof)
    {
        if (auto const error = checkDegreesOfFreedom(dof))
        {
            return std::unexpected{*error};
        }
        auto gaussian = ScaleGaussianType{
            std::move(space),
            std::forward<Location>(location),
            MatrixType{scale.template selfadjointView<Eigen::Lower>()}
        };
        if (auto result = gaussian.factorize(); !result)
        {
            return std::unexpected{
                result.error() == GaussianError::NonFinite ? StudentTError::NonFiniteScale
                                                           : StudentTError::NotPositiveDefinite
            };
        }
        return StudentT{std::move(gaussian), dof};
    }

    /** create() in the default space. */
    template <MatrixExpression Location, typename ScaleDerived>
    [[nodiscard]] static std::expected<StudentT, StudentTError>
    create(Location && location, Eigen::MatrixBase<ScaleDerived> const & scale, Scalar dof)
    {
        return create(Space_{}, std::forward<Location>(location), scale, dof);
    }

    /** create() from the location and scale of \a gaussian, which keeps its space. */
    [[nodiscard]] static std::expected<StudentT, StudentTError>
    create(ScaleGaussianType const & gaussian, Scalar dof)
    {
        return create(gaussian.space(), gaussian.mean(), gaussian.covarianceLower(), dof);
    }

    /** \returns the number of coordinates of the location */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_gaussian.valueDimension();
    }

    /** \returns the number of rows and columns of the scale matrix */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return m_gaussian.differenceDimension();
    }

    /** \returns the space the values live in, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_gaussian.space();
    }

    /** \returns the location \f$ \mu \f$: the mode and median, and the mean when \f$ \nu > 1 \f$ */
    [[nodiscard]] VectorType const &
    location() const
    {
        return m_gaussian.mean();
    }

    /** \returns the scale matrix \f$ \Sigma \f$, symmetric */
    [[nodiscard]] MatrixType const &
    scale() const
    {
        return m_gaussian.covarianceLower();
    }

    /** \returns the degrees of freedom \f$ \nu \f$ */
    [[nodiscard]] Scalar
    dof() const
    {
        return m_dof;
    }

    /** \returns a view of \f$ \mathcal{N}(\mu, \Sigma) \f$, whose factorization this distribution
     * shares. It must not outlive this distribution.
     */
    [[nodiscard]] typename ScaleGaussianType::DensityType
    scaleDensity() const
    {
        auto const density = m_gaussian.toDensity();
        return *density;
    }

    /** \returns the log of the density at \a x
     *
     * \sa density()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        return logDensityAtSquaredDistance(scaleDensity().squaredMahalanobisDistance(x));
    }

    /** \returns the density at \a x
     *
     * \warning Underflows easily in high dimensions; prefer logDensity().
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** \returns the mean \f$ \mu \f$ and covariance \f$ \frac{\nu}{\nu - 2} \Sigma \f$, or
     * MomentsError::Infinite if \f$ \nu \le 2 \f$, when the covariance is infinite
     */
    [[nodiscard]] std::expected<MomentsType, MomentsError>
    moments() const
    {
        if (!(m_dof > Scalar(2)))
        {
            return std::unexpected{MomentsError::Infinite};
        }
        return MomentsType{location(), MatrixType{(m_dof / (m_dof - Scalar(2))) * scale()}};
    }

    /** \returns the log-density at a point whose squared Mahalanobis distance under the scale
     * matrix is \a squaredDistance
     */
    [[nodiscard]] Scalar
    logDensityAtSquaredDistance(Scalar squaredDistance) const
    {
        return m_logNormalizer
             - Scalar(0.5) * (m_dof + internal::scalarFromCount<Scalar>(differenceDimension()))
                   * logOnePlus<Scalar>(squaredDistance / m_dof);
    }

    /** marginal() of the subspace named by the tag path \a Path (see idx()), in that subspace. */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::marginalOverPath<Path>(space(), [&](auto const & target, auto subspace) {
            return restricted(target, std::move(subspace));
        });
    }

    /** marginal() of one repetition of a repeated block; see GaussianParams::marginal(element). */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const
    {
        return internal::marginalOverElement<Path>(space(), element, [&](auto const & target, auto subspace) {
            return restricted(target, std::move(subspace));
        });
    }

    /** \returns the distribution of the other coordinates given that the subspace named by the
     * path \a GivenPath equals \a values, in the space of the remaining blocks (see
     * GaussianParams::conditional()): again a Student-t, or a StudentTError if its scale fails to
     * factorize, as it may for a numerically singular source.
     *
     * With \f$ d_b \f$ given difference coordinates at squared Mahalanobis distance \f$ \delta_b^2 \f$ from their
     * location, the conditional has \f$ \nu + d_b \f$ degrees of freedom, the %Gaussian
     * conditional's location \f$ \mu_a \boxplus K (v \boxminus \mu_b) \f$, and the %Gaussian
     * conditional's scale \f$ \Sigma_{aa} - K \Sigma_{ba} \f$ times \f$ (\nu + \delta_b^2) / (\nu + d_b) \f$,
     * with \f$ K = \Sigma_{ab} \Sigma_{bb}^{-1} \f$. So values far from the location widen it. The
     * result is \ref Ding2016 "Ding (2016)", Theorem 1.
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditional<GivenPath>(
            space(),
            [&](auto const & t, auto const & g, auto subspace, auto const & givenSpace) {
                return conditioned(t, g, values, std::move(subspace), givenSpace);
            }
        );
    }

private:

    /** \returns the StudentTError of \a dof, or nothing if it is positive and finite */
    [[nodiscard]] static std::optional<StudentTError>
    checkDegreesOfFreedom(Scalar dof)
    {
        if (!isFinite(dof))
        {
            return StudentTError::NonFiniteDegreesOfFreedom;
        }
        if (!(dof > Scalar(0)))
        {
            return StudentTError::NotPositiveDegreesOfFreedom;
        }
        return std::nullopt;
    }

    /** \returns the marginal on the coordinates \a target, which form \a subspace */
    template <typename Target, typename SubSpace>
    [[nodiscard]] StudentT<SubSpace, Scalar>
    restricted(Target const & target, SubSpace subspace) const
    {
        using Result = StudentT<SubSpace, Scalar>;
        auto result = Result::create(
            std::move(subspace),
            location()(target.values),
            typename Result::MatrixType{internal::lowerSymmetricBlock(scale(), target.differences, target.differences)},
            m_dof
        );
        EMRYS_FILTER_ASSERT(result.has_value() && "a principal block of a positive definite scale should factorize");
        return std::move(*result);
    }

    /** \returns the conditional on \a given, which form \a givenSpace, equal to \a values, for the
     * coordinates \a target, which form \a subspace
     */
    template <typename Target, typename Given, typename Derived, typename SubSpace, typename GivenSpace>
    [[nodiscard]] std::expected<StudentT<SubSpace, Scalar>, StudentTError>
    conditioned(
        Target const & target,
        Given const & given,
        Eigen::MatrixBase<Derived> const & values,
        SubSpace subspace,
        GivenSpace const & givenSpace
    ) const
    {
        using Result = StudentT<SubSpace, Scalar>;
        auto gaussian =
            internal::conditionMomentsWithEvidence(location(), scale(), target, given, values, givenSpace, subspace);
        if (!gaussian)
        {
            return std::expected<Result, StudentTError>{std::unexpect, StudentTError::NotPositiveDefinite};
        }
        Scalar const givenSize = Scalar(given.differences.size());
        Scalar const stretch = (m_dof + gaussian->squaredDistance) / (m_dof + givenSize);
        return Result::create(
            std::move(subspace),
            std::move(gaussian->moments.mean),
            typename Result::MatrixType{stretch * gaussian->moments.cov},
            m_dof + givenSize
        );
    }

    /** Takes the factorized \a gaussian and \a dof, and computes the log-normalizer
     * \f$ \log\Gamma(\tfrac{\nu + d}{2}) - \log\Gamma(\tfrac{\nu}{2}) - \tfrac{d}{2} \log(\nu\pi)
     * - \tfrac{1}{2} \log|\Sigma| \f$.
     */
    StudentT(ScaleGaussianType gaussian, Scalar dof)
        : m_gaussian{std::move(gaussian)}
        , m_dof{dof}
    {
        Scalar const d = internal::scalarFromCount<Scalar>(differenceDimension());
        Scalar const logDeterminant = scaleDensity().logDeterminant();
        m_logNormalizer = logGamma<Scalar>(Scalar(0.5) * (m_dof + d)) - logGamma<Scalar>(Scalar(0.5) * m_dof)
                        - Scalar(0.5) * d * Eigen::numext::log(Scalar(m_dof * pi<Scalar>()))
                        - Scalar(0.5) * logDeterminant;
    }

    ScaleGaussianType m_gaussian;  ///< \f$ \mathcal{N}(\mu, \Sigma) \f$, factorized
    Scalar m_dof;                  ///< the degrees of freedom \f$ \nu \f$
    Scalar m_logNormalizer{};      ///< the log of the density's constant factor
};

namespace internal
{

/** Draws from a Student-t, not yet normalized, with each draw's squared Mahalanobis distance. */
template <typename Space, typename Scalar = double>
struct StudentTDraws
{
    SampleSet<Space, Scalar> samples;  ///< one draw per column
    Eigen::Array<Scalar, Eigen::Dynamic, 1>
        squaredDistance;  ///< each draw's squared Mahalanobis distance under the scale
};

/** \returns \a n draws from \a t as the scale mixture \f$ \mu \boxplus L z \sqrt{\nu / u} \f$,
 * \f$ z \sim \mathcal{N}(0, I) \f$, \f$ u \sim \chi^2_\nu \f$, with each draw's
 * \f$ \|z\|^2 \nu / u \f$
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] StudentTDraws<Space, Scalar>
drawStudentT(StudentT<Space, Scalar> const & t, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using Normals = SampleSet<DifferenceSpace<Space>, Scalar>;
    Normals const z = standardNormal<Normals>(t.differenceDimension(), n, rng);
    auto chiSquared = std::chi_squared_distribution<Scalar>{t.dof()};
    auto stretch = Eigen::Array<Scalar, Eigen::Dynamic, 1>{n};  // sqrt(ν / u) for each draw
    for (Eigen::Index j = 0; j < n; ++j)
    {
        stretch(j) = Eigen::numext::sqrt(t.dof() / chiSquared(rng));
    }
    Normals offsets = multiply(asTriangular<Eigen::Lower>(t.scaleDensity().llt().matrixLLT()), z);
    for (Eigen::Index j = 0; j < n; ++j)
    {
        offsets.col(j) *= stretch(j);
    }
    return {
        Emrys::plus(t.space(), t.location(), offsets),
        z.colwise().squaredNorm().transpose().array() * stretch.square()
    };
}

/** One draw from a Student-t, in a vector sized by the space, with its squared Mahalanobis distance. */
template <typename Space, typename Scalar = double>
struct StudentTDraw
{
    typename Moments<Space, Scalar>::VectorType sample;  ///< the draw
    Scalar squaredDistance = 0;                          ///< its squared Mahalanobis distance under the scale
};

/** \returns one draw from \a t, \f$ \mu \boxplus L z \sqrt{\nu / u} \f$, drawing the normals and
 * then the chi-squared, with its \f$ \|z\|^2 \nu / u \f$
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] StudentTDraw<Space, Scalar>
drawOneStudentT(StudentT<Space, Scalar> const & t, G & rng)
{
    using Difference = DifferenceVector<Space, Scalar>;
    Difference const z = standardNormalVector<Difference>(t.differenceDimension(), rng);
    Scalar const stretch = Eigen::numext::sqrt(t.dof() / std::chi_squared_distribution<Scalar>{t.dof()}(rng));
    Difference offset = multiply(asTriangular<Eigen::Lower>(t.scaleDensity().llt().matrixLLT()), z);
    offset *= stretch;
    return {
        typename Moments<Space, Scalar>::VectorType{Emrys::plus(t.space(), t.location(), offset)},
        z.squaredNorm() * stretch * stretch
    };
}

}  // namespace internal

/** \ingroup Distributions
 * \returns \a n independent draws from \a t, one per column: \f$ \mu \boxplus L z \sqrt{\nu / u} \f$
 * with \f$ z \sim \mathcal{N}(0, I) \f$ and \f$ u \sim \chi^2_\nu \f$, normalized in the space.
 *
 * \sa drawSamplesWithLogDensity(StudentT const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Space, Scalar>
drawSamples(StudentT<Space, Scalar> const & t, Eigen::Index n, G & rng)
{
    auto draws = internal::drawStudentT(t, n, rng);
    return std::move(draws.samples);
}

/** \ingroup Distributions
 * \returns one draw from \a t, \f$ \mu \boxplus L z \sqrt{\nu / u} \f$ as drawSamples() draws each,
 * in a vector sized by the space: the stream of draws that drawSample() of a %Gaussian is. It
 * draws the normals and then the chi-squared, where drawSamples() draws every normal first.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(StudentT<Space, Scalar> const & t, G & rng)
{
    return internal::drawOneStudentT(t, rng).sample;
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a t, one per column, with the log-density at each.
 *
 * In a linear space the log-densities cost almost nothing extra: the squared Mahalanobis
 * distance of a draw is \f$ \|z\|^2 \nu / u \f$. In other spaces each is logDensity() of the
 * stored sample, the nearest-image density.
 *
 * \sa drawSamples(StudentT const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(StudentT<Space, Scalar> const & t, Eigen::Index n, G & rng)
{
    auto draws = internal::drawStudentT(t, n, rng);
    auto result = SamplesWithLogDensity<Space, Scalar>{std::move(draws.samples), {}};
    result.logDensity.resize(n);
    if constexpr (isLinear<Space>)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            result.logDensity(j) = t.logDensityAtSquaredDistance(draws.squaredDistance(j));
        }
    }
    else
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            result.logDensity(j) = t.logDensity(result.samples.col(j));
        }
    }
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a t, as drawSample() draws it, with the log-density at it, in a vector
 * sized by the space: the stream of draws that drawSampleWithLogDensity() of a %Gaussian is, with
 * the log-density computed as drawSamplesWithLogDensity() computes it.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(StudentT<Space, Scalar> const & t, G & rng)
{
    auto draw = internal::drawOneStudentT(t, rng);
    auto result = SampleWithLogDensity<Space, Scalar>{std::move(draw.sample), 0};
    if constexpr (isLinear<Space>)
    {
        result.logDensity = t.logDensityAtSquaredDistance(draw.squaredDistance);
    }
    else
    {
        result.logDensity = t.logDensity(result.sample);
    }
    return result;
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes a StudentTError in words. */
inline void
describe(std::ostream & stream, StudentTError error, Indent indent)
{
    stream << indent;
    switch (error)
    {
        case StudentTError::NotPositiveDefinite:
            stream << "the scale matrix is not positive definite";
            break;
        case StudentTError::NonFiniteScale:
            stream << "the scale matrix is not finite";
            break;
        case StudentTError::NotPositiveDegreesOfFreedom:
            stream << "the degrees of freedom are not positive";
            break;
        case StudentTError::NonFiniteDegreesOfFreedom:
            stream << "the degrees of freedom are not finite";
            break;
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
