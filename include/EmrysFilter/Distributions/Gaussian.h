#pragma once

/** \file
 * The %Gaussian: its parameters GaussianParams, the Gaussian that evaluates the density, read
 * from them by toDensity() (decisions.md [params-and-density]), and the GaussianMarginal view from marginal<Path>()
 * (decisions.md [density-names]), with conditional<Path>(), insert() and erase() on a resizable block, the
 * repair of a covariance that fails to factorize by a Repair such as roots::ridge(), and the exact confidence
 * regions. Every Kalman-family
 * filter holds its parameters. It is the covariance form of the four in decisions.md [gaussian-forms], with the
 * accessors every form shares. GaussianError and the repairs are in Roots.h. Read Sampling.h next.
 */

#include <expected>
#include <optional>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/ChiSquared.h>
#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Roots.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

template <Space Space_, RealScalar Scalar_>
class Gaussian;

template <Space Space_, RealScalar Scalar_>
class GaussianParams;

template <typename Source, typename Selection, typename SubSpace>
class GaussianMarginal;

/** \ingroup Distributions
 * \brief The result of GaussianParams::conditionalWithEvidence(): the conditional distribution
 * and the log-evidence of the given values.
 *
 * \tparam Space_   the space of the remaining blocks
 * \tparam Scalar_  scalar type of the moments
 */
template <Space Space_, RealScalar Scalar_ = double>
struct GaussianConditional
{
    GaussianParams<Space_, Scalar_> conditional;  ///< the distribution of the other blocks given the values
    Scalar_ logEvidence;  ///< \f$ \log p(b = v) \f$, the given values' log-density under their marginal
};

namespace internal
{

/** Write access to a %Gaussian's mean and covariance where they are, for an update that conditions
 * the %Gaussian in place (decisions.md [borrowed-intermediates]).
 */
struct GaussianAccess
{
    /** \returns \a gaussian's moments to change in place, its factorization invalidated. Its
     * marginals may be read afterward, as values may change under them; a density may not.
     */
    template <typename G>
    [[nodiscard]] static typename G::MomentsType &
    moments(G & gaussian)
    {
        gaussian.m_stamp.touchValues();
        gaussian.m_factorValid = false;
        return gaussian.m_moments;
    }
};

/** \brief The density of a %Gaussian whose covariance is factorized, read without a view, for an
 * container whose own view already checks that the %Gaussian has not changed: a GaussianMixture
 * reading its components, and a KernelDensity its kernel. It skips the view's check and the
 * \c std::expected that toDensity() returns, which a loop over components would pay for each one.
 * Gaussian::logPeak() and Gaussian::squaredMahalanobisDistance() are these.
 */
struct FactorizedGaussian
{
    /** \returns \f$ \log|\Sigma| \f$ of \a gaussian, as Gaussian::logDeterminant() */
    template <typename Space, typename Scalar>
    [[nodiscard]] static Scalar
    logDeterminant(GaussianParams<Space, Scalar> const & gaussian)
    {
        EMRYS_FILTER_ASSERT(gaussian.m_factorValid && "the Gaussian must be factorized");
        return gaussian.m_logDeterminant;
    }

    /** \returns \f$ \log p \f$ at the mean of \a gaussian, as Gaussian::logPeak() */
    template <typename Space, typename Scalar>
    [[nodiscard]] static Scalar
    logPeak(GaussianParams<Space, Scalar> const & gaussian)
    {
        EMRYS_FILTER_ASSERT(gaussian.m_factorValid && "the Gaussian must be factorized");
        Scalar const log2Pi = Emrys::log2Pi<Scalar>();
        return Scalar(-0.5)
             * (internal::scalarFromCount<Scalar>(gaussian.differenceDimension()) * log2Pi + gaussian.m_logDeterminant);
    }

    /** \returns the squared Mahalanobis distance of \a x from \a gaussian, as Gaussian::squaredMahalanobisDistance() */
    template <typename Space, typename Scalar, typename Derived>
    [[nodiscard]] static Scalar
    squaredMahalanobisDistance(GaussianParams<Space, Scalar> const & gaussian, Eigen::MatrixBase<Derived> const & x)
    {
        EMRYS_FILTER_ASSERT(gaussian.m_factorValid && "the Gaussian must be factorized");
        typename GaussianParams<Space, Scalar>::DifferenceVectorType residual =
            Emrys::minus(gaussian.space(), x, gaussian.mean());
        gaussian.m_llt->matrixL().solveInPlace(residual);
        return residual.squaredNorm();
    }

    /** \returns the log-density of \a gaussian at \a x, as Gaussian::logDensity() */
    template <typename Space, typename Scalar, typename Derived>
    [[nodiscard]] static Scalar
    logDensity(GaussianParams<Space, Scalar> const & gaussian, Eigen::MatrixBase<Derived> const & x)
    {
        return logPeak(gaussian) - Scalar(0.5) * squaredMahalanobisDistance(gaussian, x);
    }

    /** \returns the Cholesky factorization of \a gaussian's covariance, as Gaussian::llt() */
    template <typename Space, typename Scalar>
    [[nodiscard]] static Eigen::LLT<typename GaussianParams<Space, Scalar>::MatrixType> const &
    llt(GaussianParams<Space, Scalar> const & gaussian)
    {
        EMRYS_FILTER_ASSERT(gaussian.m_factorValid && "the Gaussian must be factorized");
        auto const & llt = gaussian.m_llt;
        return *llt;
    }
};

template <template <typename, typename, typename> typename Marginal, typename Path, typename Source>
[[nodiscard]] auto
tagMarginal(Source const & source);

template <template <typename, typename, typename> typename Marginal, typename Path, typename Source>
[[nodiscard]] auto
elementMarginal(Source const & source, Eigen::Index element);

/** Shared implementation of GaussianParams::conditional() and Gaussian::conditional(), with
 * resolved indices and the spaces of the target and the given entries, and of their
 * conditionalWithEvidence() when \a WithEvidence is set.
 */
template <
    bool WithEvidence,
    typename Source,
    typename Target,
    typename Given,
    typename Derived,
    typename TargetSpace,
    typename GivenSpace>
[[nodiscard]] auto
conditionalGaussianIn(
    Source const & source,
    Target const & target,
    Given const & given,
    Eigen::MatrixBase<Derived> const & values,
    TargetSpace const & targetSpace,
    GivenSpace const & givenSpace
)
{
    using Scalar = typename Source::Scalar;
    EMRYS_FILTER_ASSERT(distinctIndices(target.values, given.values) && "target and given indices must be distinct");
    if constexpr (WithEvidence)
    {
        using Result = std::expected<GaussianConditional<TargetSpace, Scalar>, GaussianError>;
        auto result = conditionMomentsWithEvidence(
            source.mean(),
            source.covarianceLower(),
            target,
            given,
            values,
            givenSpace,
            targetSpace
        );
        if (!result)
        {
            return Result{std::unexpect, GaussianError::NotPositiveDefinite};
        }
        return Result{GaussianConditional<TargetSpace, Scalar>{
            {targetSpace, std::move(result->moments.mean), std::move(result->moments.cov)},
            result->logLikelihood
        }};
    }
    else
    {
        auto result =
            conditionMoments(source.mean(), source.covarianceLower(), target, given, values, givenSpace, targetSpace);
        return GaussianParams<TargetSpace, Scalar>{targetSpace, std::move(result.mean), std::move(result.cov)};
    }
}

/** Shared implementation of conditional<GivenPath>() and conditionalWithEvidence<GivenPath>();
 * see tagConditional().
 */
template <bool WithEvidence, typename GivenPath, typename Source, typename Derived>
[[nodiscard]] auto
tagConditionalGaussian(Source const & source, Eigen::MatrixBase<Derived> const & values)
{
    return tagConditional<GivenPath>(
        source.space(),
        [&](auto const & target, auto const & given, auto const & targetSpace, auto const & givenSpace) {
            return conditionalGaussianIn<WithEvidence>(source, target, given, values, targetSpace, givenSpace);
        }
    );
}

}  // namespace internal

/** \ingroup Distributions
 * \brief A %Gaussian distribution: a mean and covariance in a space.
 *
 * The [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution),
 * with the space's \f$ \boxplus \f$ and \f$ \boxminus \f$ in place of addition and
 * subtraction where its coordinates are not plain.
 *
 * toDensity() factorizes the covariance (if it has changed since the last
 * factorization) and returns a Gaussian, which evaluates the density. The density checks this
 * object for changes: reading it after this object has changed in any way is a programmer error
 * caught by \c EMRYS_FILTER_ASSERT (decisions.md [view-stamp]). The check detects the change and prevents
 * nothing: it does not synchronize threads or block the mutators (see \ref Distributions). Share
 * the Gaussian between threads, or a copy of moments(), rather than this object: its \c const
 * members factorize into a cache (see factorize()).
 *
 * marginal() returns a read-only view of a subset of the entries, which checks only this object's
 * size and structure: its values may change under the marginal, which sees them, but reading the
 * marginal after a resize, an insert() or erase(), or an assignment is a programmer error. Neither
 * view may outlive this object, and neither may be made from a temporary.
 *
 * The space's value dimension always matches the mean's length. insert() and erase() grow and
 * shrink the space's resizable block in place, such as adding or dropping a landmark, and
 * resize() resizes it whole.
 *
 * The space describes how each coordinate behaves (see \ref Spaces). The mean is
 * normalized whenever it is set, and residuals are normalized before they are used, so
 * a periodic coordinate's mean stays in range and its residuals take the short way
 * around. Densities use the nearest image only (see \ref Spaces).
 *
 * Only the lower triangle of the covariance is read by the factorization.
 * Changing the mean does not invalidate the factorization.
 *
 * \tparam Space_   space of the coordinates, which sets the dimension and the storage
 *                  bound (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 *
 * \sa class Gaussian, class Moments
 */
template <Space Space_, RealScalar Scalar_ = double>
class GaussianParams
{
public:

    using MomentsType = Moments<Space_, Scalar_>;                             ///< the mean and covariance together
    using Scalar = typename MomentsType::Scalar;                              ///< the scalar type
    using Index = typename MomentsType::Index;                                ///< Eigen's index type
    using VectorType = typename MomentsType::VectorType;                      ///< a value, such as the mean
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;  ///< a difference, such as a residual
    using MatrixType = typename MomentsType::MatrixType;                      ///< a matrix over differences
    using SpaceType = Space_;                                                 ///< the space the values live in
    using DensityType = Gaussian<Space_, Scalar_>;                            ///< the view toDensity() returns

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;     ///< parameters, which views check
    static constexpr bool IsInfoForm = false;  ///< holds the covariance, not the precision [gaussian-forms]
    static constexpr bool IsSqrtForm = false;  ///< holds the covariance whole, not a factor [gaussian-forms]

    /** Constructs a %Gaussian in a default space, with a zero mean and covariance, which
     * factorize() rejects until a covariance is set.
     */
    GaussianParams()
        : GaussianParams{Space_{}}
    {}

    /** Constructs a %Gaussian with \a n value coordinates, with a zero mean and covariance. A space
     * with one resizable block is resized to \a n; any other must already have \a n coordinates.
     * A space with several resizable blocks must be given, with its counts.
     */
    explicit GaussianParams(Index n)
        requires internal::SizedByLength<Space_>
    {
        fitValueDimension(n);
        zeroMoments();
    }

    /** Constructs a %Gaussian from \a moments, in a default space resized to fit them. */
    explicit GaussianParams(MomentsType moments)
        requires internal::SizedByLength<Space_>
        : m_moments{std::move(moments)}
    {
        fitSpace();
    }

    /** Constructs a %Gaussian from \a mean and \a cov, moving either one passed as an rvalue of the
     * stored type. \a cov may be a diagonal, its variances:
     * <tt>GaussianParams<State>{mean, variances.asDiagonal()}</tt>.
     */
    template <MatrixExpression Mean, CovarianceExpression Cov>
        requires internal::SizedByLength<Space_>
    GaussianParams(Mean && mean, Cov && cov)
        : m_moments{std::forward<Mean>(mean), std::forward<Cov>(cov)}
    {
        fitSpace();
    }

    /** Constructs a %Gaussian in \a space, with a zero mean and covariance. */
    explicit GaussianParams(Space_ space)
        : m_moments{space}
        , m_space{std::move(space)}
    {
        zeroMoments();
        fitSpace();
    }

    /** Constructs a %Gaussian with \a n value coordinates in \a space, with a zero mean and
     * covariance. A space with one resizable block is resized to \a n; any other must already
     * have \a n coordinates.
     */
    GaussianParams(Space_ space, Index n)
        : m_space{std::move(space)}
    {
        fitValueDimension(n);
        zeroMoments();
    }

    /** Constructs a %Gaussian from \a moments in \a space, which is resized to fit them if it has
     * one resizable block.
     */
    GaussianParams(Space_ space, MomentsType moments)
        : m_moments{std::move(moments)}
        , m_space{std::move(space)}
    {
        fitSpace();
    }

    /** GaussianParams(mean, cov) in \a space. */
    template <MatrixExpression Mean, CovarianceExpression Cov>
    GaussianParams(Space_ space, Mean && mean, Cov && cov)
        : m_moments{std::forward<Mean>(mean), std::forward<Cov>(cov)}
        , m_space{std::move(space)}
    {
        fitSpace();
    }

    /** Copies the data and any valid factorization. Views of \a other do not see the copy. */
    GaussianParams(GaussianParams const & other)
        : m_moments{other.m_moments}
        , m_space{other.m_space}
        , m_llt{other.m_factorValid ? other.m_llt : std::nullopt}
        , m_factorValid{other.m_factorValid}
    {
        if (m_factorValid)
        {
            m_logDeterminant = other.m_logDeterminant;
        }
    }

    /** Moves the data and any valid factorization; a view of \a other may not be read afterward. */
    GaussianParams(GaussianParams && other) noexcept
        : m_moments{std::move(other.m_moments)}
        , m_space{std::move(other.m_space)}
        , m_llt{other.m_factorValid ? std::move(other.m_llt) : std::nullopt}
        , m_factorValid{other.m_factorValid}
    {
        if (m_factorValid)
        {
            m_logDeterminant = other.m_logDeterminant;
        }
        other.m_stamp.touchShape();
        other.m_factorValid = false;
    }

    /** Copies the data and any valid factorization; a view of this object may not be read
     * afterward.
     */
    GaussianParams &
    operator=(GaussianParams const & other)
    {
        m_stamp.touchShape();
        m_moments = other.m_moments;
        m_space = other.m_space;
        if (other.m_factorValid)
        {
            m_llt = other.m_llt;
            m_logDeterminant = other.m_logDeterminant;
        }
        m_factorValid = other.m_factorValid;
        return *this;
    }

    /** Moves the data and any valid factorization; a view of either object may not be read
     * afterward.
     */
    GaussianParams &
    operator=(GaussianParams && other) noexcept
    {
        m_stamp.touchShape();
        other.m_stamp.touchShape();
        m_moments = std::move(other.m_moments);
        m_space = std::move(other.m_space);
        if (other.m_factorValid)
        {
            m_llt = std::move(other.m_llt);
            m_logDeterminant = other.m_logDeterminant;
        }
        m_factorValid = other.m_factorValid;
        other.m_factorValid = false;
        return *this;
    }

    /** \returns the number of coordinates of the mean (see \ref Spaces) */
    [[nodiscard]] Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the number of rows and columns of the covariance (see \ref Spaces) */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

    /** Resizes mean, covariance, and factorization, and the space's resizable block, to \a n
     * value coordinates. Contents are unspecified afterward. To grow or shrink the block element
     * by element, keeping the contents, use insert() and erase(), as a space with several
     * resizable blocks must.
     */
    void
    resize(Index n)
        requires internal::SizedByLength<Space_>
    {
        m_stamp.touchShape();
        fitValueDimension(n);
        m_factorValid = false;
    }

    /** Inserts the entries \a mean into the resizable block named by the tag path \a Path,
     * before its element \a element, with covariance \a cov and cross-covariance \a cross
     * with the existing entries (a row per difference coordinate of the new entries and a
     * column per existing one, in order). The space's block grows to match.
     *
     * An element is one repetition of a Repeated block, or one coordinate of a resizable
     * Linear or Periodic block; \a mean holds one or more whole elements.
     */
    template <typename Path, typename MeanDerived, typename CovDerived, typename CrossDerived>
    void
    insert(
        Index element,
        Eigen::MatrixBase<MeanDerived> const & mean,
        Eigen::MatrixBase<CovDerived> const & cov,
        Eigen::MatrixBase<CrossDerived> const & cross
    )
    {
        m_stamp.touchShape();
        EMRYS_FILTER_ASSERT(cross.cols() == differenceDimension());
        auto const rows = internal::growBlock<Path>(m_space, element, mean.size());
        EMRYS_FILTER_ASSERT(
            cov.rows() == rows.differenceCount && cov.cols() == rows.differenceCount
            && cross.rows() == rows.differenceCount
        );
        internal::insertRows(m_moments.mean(), rows.start, mean);
        internal::insertSymmetric(m_moments.covarianceLower(), rows.differenceStart, cov, cross);
        normalizeValues(m_space, m_moments.mean());
        m_factorValid = false;
    }

    /** insert() of entries independent of the existing ones. */
    template <typename Path, typename MeanDerived, typename CovDerived>
    void
    insert(Index element, Eigen::MatrixBase<MeanDerived> const & mean, Eigen::MatrixBase<CovDerived> const & cov)
    {
        using Cross = internal::MatrixOf<
            Scalar,
            Eigen::Dynamic,
            Eigen::Dynamic,
            CovDerived::MaxRowsAtCompileTime,
            MomentsType::MatrixType::MaxColsAtCompileTime>;
        insert<Path>(element, mean, cov, Cross::Zero(cov.rows(), differenceDimension()));
    }

    /** Removes \a count elements of the resizable block named by the tag path \a Path,
     * starting at its element \a element: the marginal of the rest. The space's block shrinks
     * to match.
     */
    template <typename Path>
    void
    erase(Index element, Index count = 1)
    {
        m_stamp.touchShape();
        auto const rows = internal::shrinkBlock<Path>(m_space, element, count);
        internal::eraseRows(m_moments.mean(), rows.start, rows.count);
        internal::eraseSymmetric(m_moments.covarianceLower(), rows.differenceStart, rows.differenceCount);
        m_factorValid = false;
    }

    /** \returns the space, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the mean and covariance together */
    [[nodiscard]] MomentsType const &
    moments() const
    {
        return m_moments;
    }

    /** \returns the mean, a value in the space */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return m_moments.mean();
    }

    /** \returns the stored covariance, over differences, of which only the lower triangle is
     * guaranteed to be read: a reference, with no copy and no failure. covariance() returns it
     * with both triangles filled, and covariance<Path>() a block of it.
     */
    [[nodiscard]] MatrixType const &
    covarianceLower() const
    {
        return m_moments.covarianceLower();
    }

    /** \returns the entries of the mean in the block named by the path \a Path (a tag or a
     * TagPath): <tt>g.mean<Position>()</tt>. A lazy expression that keeps the block's fixed size,
     * or a coefficient for a single coordinate.
     */
    template <typename Path>
    [[nodiscard]] auto
    mean() const
    {
        return mean()(idx<Path>(m_space));
    }

    /** mean<Path>() of repetition \a element of the repeated block \a Path starts at (see
     * marginal<Path>(element)).
     */
    template <typename Path>
    [[nodiscard]] auto
    mean(Index element) const
    {
        return mean()(idx<Path>(m_space, element));
    }

    /** \returns the covariance of the block named by the path \a Path, at its rows of
     * differences (see diffIdx()), read from the lower triangle: a lazy expression that
     * must not outlive this object, or a variance for a single coordinate.
     */
    template <typename Path>
    [[nodiscard]] auto
    covariance() const
    {
        return internal::lowerSymmetricSelection<DifferenceDimensionAtCompileTime, MaxDifferenceDimensionAtCompileTime>(
            covarianceLower(),
            diffIdx<Path>(m_space)
        );
    }

    /** covariance<Path>() of repetition \a element of the repeated block \a Path starts at. */
    template <typename Path>
    [[nodiscard]] auto
    covariance(Index element) const
    {
        return internal::lowerSymmetricSelection<DifferenceDimensionAtCompileTime, MaxDifferenceDimensionAtCompileTime>(
            covarianceLower(),
            diffIdx<Path>(m_space, element)
        );
    }

    /** \returns the cross-covariance of the blocks named by the paths \a RowPath and \a ColPath,
     * read from the lower triangle: a matrix with a row per difference coordinate of the first and
     * a column per difference coordinate of the second, or a coefficient if each is a single
     * coordinate.
     */
    template <typename RowPath, typename ColPath>
    [[nodiscard]] auto
    covariance() const
    {
        return internal::lowerSymmetricCross<DifferenceDimensionAtCompileTime, MaxDifferenceDimensionAtCompileTime>(
            covarianceLower(),
            diffIdx<RowPath>(m_space),
            diffIdx<ColPath>(m_space)
        );
    }

    /** Sets the mean, normalized by the space, moving \a mean if it is an rvalue of the stored
     * type. Does not invalidate the factorization.
     */
    template <MatrixExpression Mean>
    void
    setMean(Mean && mean)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(mean.size() == valueDimension());
        m_moments.mean() = std::forward<Mean>(mean);
        normalizeValues(m_space, m_moments.mean());
    }

    /** Sets the covariance, moving \a cov if it is an rvalue of the stored type, and invalidates
     * the factorization. Only its lower triangle is read, so \a cov need not be symmetric: an
     * upper triangle that disagrees is ignored, not an error. \a cov may be a diagonal, its
     * variances.
     */
    template <CovarianceExpression Cov>
    void
    setCovariance(Cov && cov)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(cov.rows() == differenceDimension() && cov.cols() == differenceDimension());
        m_moments.covarianceLower() = std::forward<Cov>(cov);
        m_factorValid = false;
    }

    /** Scales the covariance so that its one-standard-deviation ellipsoid is the \a probability
     * confidence region, and invalidates the factorization.
     *
     * The [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) squared
     * \f$ (x - \mu)^T \Sigma^{-1} (x - \mu) \f$ of a draw is
     * chi-squared with \f$ k \f$ degrees of freedom, the difference dimension, so the region
     * holding \a probability of the mass is the unit ellipsoid of
     * \f$ \chi^{2\,-1}_k(p) \, \Sigma \f$, the covariance this sets. Does nothing to a %Gaussian of
     * dimension zero.
     *
     * \param probability  the confidence, in \f$ (0, 1) \f$
     *
     * \sa chiSquaredQuantile()
     */
    void
    scaleToConfidence(Scalar probability)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(probability > 0 && probability < 1 && "the probability must be in (0, 1)");
        if (differenceDimension() == 0)
        {
            return;
        }
        m_moments.covarianceLower() *=
            chiSquaredQuantile<Scalar>(probability, internal::scalarFromCount<Scalar>(differenceDimension()));
        m_factorValid = false;
    }

    /** \returns the covariance, with both triangles filled from the lower one. Never fails: it
     * returns \c std::expected so that every form of the %Gaussian has the same four accessors
     * (decisions.md [gaussian-forms]), which generic code reaches through the GaussianForm concept.
     * covarianceLower() reads the stored matrix without a copy.
     *
     * \sa precision(), covarianceRoot(), precisionRoot()
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        MatrixType result = covarianceLower().template selfadjointView<Eigen::Lower>();
        return result;
    }

    /** \returns the precision \f$ \Lambda = \Sigma^{-1} \f$, solved with the cached factorization
     * (see factorize()), or its GaussianError.
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precision() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(*m_llt, result);
        return result;
    }

    /** \returns the lower-triangular Cholesky factor \f$ L \f$, with \f$ L L^T = \Sigma \f$ and zeros
     * above the diagonal, from the cached factorization (see factorize()), or its GaussianError.
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covarianceRoot() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        MatrixType result = m_llt->matrixL();
        return result;
    }

    /** \returns \f$ L^{-1} \f$, the triangular factor of the precision that this form gives
     * cheapest: \f$ (L^{-1})^T L^{-1} = \Sigma^{-1} \f$. It is lower triangular, as the covariance
     * forms' factors are, where the information forms give an upper-triangular one [gaussian-forms]. Or the
     * GaussianError of factorize().
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precisionRoot() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(m_llt->matrixL(), result);
        return result;
    }

    /** \returns a read-only view of the subspace named by the path \a Path (a tag or a TagPath),
     * whose space is that subspace with its parameters. It checks this object's shape.
     *
     * \sa class GaussianMarginal
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const &
    {
        return internal::tagMarginal<GaussianMarginal, Path>(*this);
    }

    /** A marginal of a temporary would outlive it; copy the marginal of a named %Gaussian instead. */
    template <typename Path>
    void
    marginal() const && = delete;

    /** \returns a read-only view of repetition \a element of a repeated block (see idx()), such
     * as one vehicle of a fleet. \a Path starts at the repeated block, and may go on inside it:
     * <tt>marginal<TagPath<Fleet, Heading>>(k)</tt> is vehicle \a k's heading. Its space is that of
     * one repetition, or of the block the path ends at.
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const &
    {
        return internal::elementMarginal<GaussianMarginal, Path>(*this, element);
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal(Index element) const && = delete;

    /** \returns a copy of the distribution of the other coordinates given that the subspace
     * named by the path \a GivenPath (a tag or a TagPath) equals \a values, which are in that
     * subspace. The result is in the space of the remaining blocks: a Composite that loses a
     * block is replaced by the blocks it has left.
     *
     * \f$ \mu_{a|b} = \mu_a \boxplus K (v \boxminus \mu_b) \f$ and
     * \f$ \Sigma_{a|b} = \Sigma_{aa} - K \Sigma_{ba} \f$, with \f$ K = \Sigma_{ab} \Sigma_{bb}^{-1} \f$
     * (internal::conditionMoments(), which gives the source). Uses a pivoted LDLT, so it always
     * produces a result, exact for values on the support when the given block is known exactly
     * (decisions.md [innovation-cholesky]); for a numerically singular source the result may fail toDensity().
     * conditionalWithEvidence() also returns the log-density of \a values.
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditionalGaussian<false, GivenPath>(*this, values);
    }

    /** \returns conditional() and the log-evidence of \a values, as a GaussianConditional.
     *
     * The log-evidence is \f$ \log p(b = v) \f$, the log-density of \a values under the marginal
     * of the given subspace, which conditioning computes anyway. A mixture's conditional keeps it
     * in its weights; a single %Gaussian has no weight, so it returns it beside the result. A
     * Rao-Blackwellized update weights each particle or grid point by it (decisions.md
     * [conditional-evidence]).
     *
     * \returns the conditional and its evidence, or GaussianError::NotPositiveDefinite if the
     *          given block's covariance \f$ \Sigma_{bb} \f$ is not positive definite: a block
     *          known exactly has no density, so its values have no evidence (decisions.md [innovation-cholesky])
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditionalWithEvidence(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditionalGaussian<true, GivenPath>(*this, values);
    }

    /** Factorizes the covariance if it has changed since the last factorization.
     *
     * Calling this ahead of time moves the cost of factorization out of a later
     * toDensity() call.
     *
     * It is \c const because it changes nothing a caller can see: it writes only the cached
     * factorization, which toDensity(), precision(), covarianceRoot(), and precisionRoot()
     * also fill when it is missing. So none of them is safe to call from two threads at once on
     * the same object until a factorization has succeeded; after that they only read. Threads
     * should share the Gaussian that toDensity() returns instead (see \ref Distributions).
     *
     * \returns nothing on success, or a GaussianError if the covariance is not
     *          positive definite or not finite.
     *
     * \sa toDensity(), GaussianError
     */
    [[nodiscard]] std::expected<void, GaussianError>
    factorize() const
    {
        if (m_factorValid)
        {
            return {};
        }
        return factorizeCovariance(m_moments.covarianceLower());
    }

    /** factorize(), repairing a covariance that is not positive definite by \a repair, a Repair:
     * <tt>g.factorize(roots::ridge())</tt>.
     *
     * If the factorization fails with GaussianError::NotPositiveDefinite, it is retried on the
     * matrix \a repair returns, which for roots::ridge() is the covariance with a ridge on its
     * diagonal, and if that succeeds, the repaired matrix becomes the covariance. Otherwise the
     * covariance is unchanged. It is not \c const, since replacing the covariance is a change that
     * views of this object see. Axes are refused: a density's factor is the Cholesky factor, and
     * axes that cannot fail would hide a covariance that is not positive definite
     * (decisions.md [covariance-repair]).
     *
     * \returns nothing on success, GaussianError::NonFinite if the covariance is not finite, the
     *          error of \a repair if it fails, or GaussianError::NotPositiveDefinite if the
     *          repaired covariance still fails to factorize, as a zero covariance does with
     *          roots::ridge()
     *
     * \sa roots::Ridge, Repair
     */
    template <typename R>
    [[nodiscard]] std::expected<void, GaussianError>
    factorize(R const & repair)
    {
        static_assert(
            Repair<R>,
            "a density takes a repair, such as roots::ridge(), not axes: its factor is the Cholesky factor "
            "(decisions.md [covariance-repair])"
        );
        auto result = std::as_const(*this).factorize();
        if (result || result.error() != GaussianError::NotPositiveDefinite)
        {
            return result;
        }
        auto repaired = repair.repaired(m_moments.covarianceLower());
        if (!repaired)
        {
            return std::unexpected{repaired.error()};
        }
        result = factorizeCovariance(*repaired);
        if (result)
        {
            m_stamp.touchValues();
            m_moments.covarianceLower() = std::move(*repaired);
        }
        return result;
    }

    /** Factorizes the covariance if needed (see factorize()), which writes the cache and so is
     * not safe on one object from two threads at once.
     *
     * \returns a Gaussian that checks this object, or a GaussianError if the covariance is
     *          not positive definite or not finite.
     *
     * \sa factorize(), class Gaussian, GaussianError
     */
    [[nodiscard]] std::expected<DensityType, GaussianError>
    toDensity() const &
    {
        if (auto result = factorize(); !result)
        {
            return std::unexpected{result.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

    /** toDensity(), repairing a covariance that is not positive definite by \a repair (see
     * factorize(R const &)): <tt>g.toDensity(roots::ridge())</tt>.
     *
     * \returns a Gaussian that checks this object, or the GaussianError from factorize(R const &)
     */
    template <typename R>
    [[nodiscard]] std::expected<DensityType, GaussianError>
    toDensity(R const & repair) &
    {
        if (auto result = factorize(repair); !result)
        {
            return std::unexpected{result.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    template <typename R>
    void
    toDensity(R const &) && = delete;

private:

    friend DensityType;                          ///< the view reads the factorization cache
    friend struct internal::FactorizedGaussian;  ///< reads the factorization cache without a view

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** A marginal checks the stamp. */
    template <typename>
    friend class internal::ShapeCheck;

    friend struct internal::GaussianAccess;

    /** Factorizes \a cov into the cache, which is then valid if this succeeds. */
    [[nodiscard]] std::expected<void, GaussianError>
    factorizeCovariance(MatrixType const & cov) const
    {
        if (!m_llt)
        {
            m_llt.emplace(cov.rows());
        }
        m_llt->compute(cov);
        if (m_llt->info() != Eigen::Success)
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        // NaN slips through LLT's positivity test but always reaches the diagonal of
        // L, so checking the log-determinant catches any non-finite lower-triangle input.
        Scalar const logDet = Scalar(2) * m_llt->matrixLLT().diagonal().array().log().sum();
        if (!isFinite(logDet))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        m_logDeterminant = logDet;
        m_factorValid = true;
        return {};
    }

    /** Resizes a space with one resizable block to the mean, checks that the sizes match, and
     * normalizes the mean.
     */
    void
    fitSpace()
    {
        internal::fitValueDimension(m_space, m_moments.valueDimension());
        EMRYS_FILTER_ASSERT(
            m_moments.differenceDimension() == differenceDimension()
            && "the covariance must have a row per difference coordinate of the space"
        );
        normalizeValues(m_space, m_moments.mean());
    }

    /** Resizes a space with one resizable block to \a n value coordinates, and the moments to fit. */
    void
    fitValueDimension(Index n)
    {
        internal::fitValueDimension(m_space, n);
        m_moments.resize(m_space);
    }

    /** Sets the mean and covariance to zero, for a constructor given neither. */
    void
    zeroMoments()
    {
        m_moments.mean().setZero();
        m_moments.covarianceLower().setZero();
    }

    MomentsType m_moments;                          ///< the mean and covariance
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space, kept in step with the mean's size

    // Factorization cache, filled lazily by the const factorize(), and so written through const
    // members (see factorize()).
    /// the Cholesky factorization of the covariance, absent until one is made, and copied only
    /// while valid
    mutable std::optional<Eigen::LLT<MatrixType>> m_llt;

    mutable Scalar m_logDeterminant{};  ///< \f$ \log|\Sigma| \f$ from the factorization
    mutable bool m_factorValid{false};  ///< whether the cache matches the covariance
    internal::Stamp m_stamp;            ///< what views check [view-stamp]
};

/** \ingroup Distributions
 * \brief The density of a %Gaussian whose covariance has been factorized: a read-only view.
 *
 * Represents the density
 * \f[ p(x) = (2\pi)^{-n/2} |\Sigma|^{-1/2} \exp\left(-\tfrac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right). \f]
 *
 * Obtained from GaussianParams::toDensity(). It checks its parameters for changes, as its copies
 * do: reading it after they have changed is a programmer error (decisions.md [view-stamp]). A Gaussian
 * must not outlive its source.
 *
 * \sa class GaussianParams
 */
template <Space Space_, RealScalar Scalar_ = double>
class Gaussian
{
public:

    using ParamsType = GaussianParams<Space_, Scalar_>;                      ///< the parameters it reads
    using SpaceType = Space_;                                                ///< the space the values live in
    using MomentsType = typename ParamsType::MomentsType;                    ///< the mean and covariance together
    using Scalar = typename ParamsType::Scalar;                              ///< the scalar type
    using Index = typename ParamsType::Index;                                ///< Eigen's index type
    using VectorType = typename ParamsType::VectorType;                      ///< a value, such as the mean
    using DifferenceVectorType = typename ParamsType::DifferenceVectorType;  ///< a difference, such as a residual
    using MatrixType = typename ParamsType::MatrixType;                      ///< a matrix over differences
    using LltType = Eigen::LLT<MatrixType>;                                  ///< the Cholesky factorization

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = ParamsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = ParamsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = ParamsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = ParamsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = false;  ///< a view: it checks its parameters

    /** \returns the number of coordinates of the mean */
    [[nodiscard]] Index
    valueDimension() const
    {
        return source().valueDimension();
    }

    /** \returns the number of rows and columns of the covariance */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return source().differenceDimension();
    }

    /** \returns the source's space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return source().space();
    }

    /** \returns the source's mean and covariance together */
    [[nodiscard]] MomentsType const &
    moments() const
    {
        return source().moments();
    }

    /** \returns the source's mean */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return source().mean();
    }

    /** \returns the source's covariance */
    [[nodiscard]] MatrixType const &
    covarianceLower() const
    {
        return source().covarianceLower();
    }

    /** \returns the mean of the block named by \a Path; see GaussianParams::mean<Path>(). */
    template <typename Path>
    [[nodiscard]] auto
    mean() const
    {
        return source().template mean<Path>();
    }

    /** mean<Path>() of one repetition; see GaussianParams::mean<Path>(element). */
    template <typename Path>
    [[nodiscard]] auto
    mean(Index element) const
    {
        return source().template mean<Path>(element);
    }

    /** \returns the covariance of the block named by \a Path; see GaussianParams::covariance<Path>(). */
    template <typename Path>
    [[nodiscard]] auto
    covariance() const
    {
        return source().template covariance<Path>();
    }

    /** covariance<Path>() of one repetition; see GaussianParams::covariance<Path>(element). */
    template <typename Path>
    [[nodiscard]] auto
    covariance(Index element) const
    {
        return source().template covariance<Path>(element);
    }

    /** \returns the cross-covariance of two blocks; see GaussianParams::covariance<RowPath, ColPath>(). */
    template <typename RowPath, typename ColPath>
    [[nodiscard]] auto
    covariance() const
    {
        return source().template covariance<RowPath, ColPath>();
    }

    /** \returns the Cholesky factorization \f$ \Sigma = L L^T \f$ of the covariance */
    [[nodiscard]] LltType const &
    llt() const
    {
        auto const & llt = source().m_llt;
        return *llt;
    }

    /** \returns the log-determinant \f$ \log|\Sigma| \f$ of the covariance */
    [[nodiscard]] Scalar
    logDeterminant() const
    {
        return source().m_logDeterminant;
    }

    /** \returns the log of the density at the mean, \f$ -\frac{1}{2}(d \log 2\pi + \log|\Sigma|) \f$,
     * with \f$ d \f$ the difference dimension
     *
     * \sa logDensity()
     */
    [[nodiscard]] Scalar
    logPeak() const
    {
        return internal::FactorizedGaussian::logPeak(source());
    }

    /** \returns the squared Mahalanobis distance of \a x,
     * \f$ d_M^2(x) = (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu) \f$
     *
     * The NIS is this for a measurement under its predicted density, and the NEES (nees()) for
     * the true state under an estimate.
     *
     * \sa logDensity()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    squaredMahalanobisDistance(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "squaredMahalanobisDistance() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        return internal::FactorizedGaussian::squaredMahalanobisDistance(source(), x);
    }

    /** \returns the confidence of \a x: the probability of the smallest confidence region that
     * contains it, \f$ F_{\chi^2_k}(d_M^2(x)) \f$, with \f$ k \f$ the difference dimension
     *
     * The density falls as squaredMahalanobisDistance() grows, and the squaredMahalanobisDistance() of a draw is
     * chi-squared with
     * \f$ k \f$ degrees of freedom, so this is also the probability that a draw is denser than
     * \a x. The confidence of a draw is uniform on \f$ [0, 1] \f$. The mean has confidence 0, as
     * does every value in dimension zero.
     *
     * \sa contains(), squaredMahalanobisDistance(), chiSquaredCdf()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    confidence(Eigen::MatrixBase<Derived> const & x) const
    {
        if (differenceDimension() == 0)
        {
            return Scalar(0);
        }
        return chiSquaredCdf<Scalar>(
            squaredMahalanobisDistance(x),
            internal::scalarFromCount<Scalar>(differenceDimension())
        );
    }

    /** \returns whether \a x lies in the \a probability confidence region, the ellipsoid
     * \f$ d_M^2(x) \le \chi^{2\,-1}_k(p) \f$ that holds \a probability of the mass
     *
     * Agrees with <tt>confidence(x) <= probability</tt>, but compares squaredMahalanobisDistance() with the quantile
     * instead of evaluating the chi-squared CDF. The region is the one-standard-deviation
     * ellipsoid of the %Gaussian that GaussianParams::scaleToConfidence() makes. In dimension zero
     * every value is inside.
     *
     * \param x            a value
     * \param probability  the confidence, in \f$ (0, 1) \f$
     *
     * \sa confidence(), GaussianParams::scaleToConfidence(), chiSquaredQuantile()
     */
    template <typename Derived>
    [[nodiscard]] bool
    contains(Eigen::MatrixBase<Derived> const & x, Scalar probability) const
    {
        EMRYS_FILTER_ASSERT(probability > 0 && probability < 1 && "the probability must be in (0, 1)");
        if (differenceDimension() == 0)
        {
            return true;
        }
        return squaredMahalanobisDistance(x)
            <= chiSquaredQuantile<Scalar>(probability, internal::scalarFromCount<Scalar>(differenceDimension()));
    }

    /** \returns the log of the density at \a x, \f$ \log p(x) = \mathrm{logPeak} - \frac{1}{2}d_M^2(x) \f$
     *
     * On a space that is not linear this is the density in the tangent space at the mean, with the
     * residual \f$ x \boxminus \mu \f$; the wrapped normal is not computed [tangent-gaussians].
     *
     * \sa logPeak(), squaredMahalanobisDistance(), density()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        return logPeak() - Scalar(0.5) * squaredMahalanobisDistance(x);
    }

    /** \returns the density \f$ p(x) \f$ at \a x
     *
     * \warning Underflows easily in high dimensions; prefer logDensity().
     *
     * \sa logDensity()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** marginal() of the subspace named by the path \a Path; see GaussianParams::marginal(). It holds a
     * copy of this view, and so of its check of the parameters.
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::tagMarginal<GaussianMarginal, Path>(*this);
    }

    /** marginal() of one repetition of a repeated block; see GaussianParams::marginal(element). */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const
    {
        return internal::elementMarginal<GaussianMarginal, Path>(*this, element);
    }

    /** \returns a copy of the conditional distribution given the subspace named by the path
     * \a GivenPath; see GaussianParams::conditional().
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditionalGaussian<false, GivenPath>(*this, values);
    }

    /** \returns conditional() and the log-evidence of \a values; see
     * GaussianParams::conditionalWithEvidence().
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditionalWithEvidence(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditionalGaussian<true, GivenPath>(*this, values);
    }

private:

    friend ParamsType;  ///< the parameters construct views

    /** Views \a source, whose covariance is factorized, and records its stamp; only
     * GaussianParams::toDensity() calls this.
     */
    explicit Gaussian(ParamsType const & source)
        : m_check{source}
    {}

    /** \returns the source; a programmer error if it has changed since toDensity() */
    [[nodiscard]] ParamsType const &
    source() const
    {
        return m_check.get();
    }

    internal::ValuesCheck<ParamsType> m_check;  ///< the check of the viewed parameters
};

/** \ingroup Distributions
 * \brief Read-only view of a subset of the entries of a GaussianParams or Gaussian.
 *
 * Obtained from GaussianParams::marginal() or Gaussian::marginal(), and held by \c auto: its
 * Selection is internal, so its type is never spelled, and a function that takes one takes it as a
 * template parameter. Of GaussianParams it
 * checks their shape (values may change underneath it, size may not); of a Gaussian it
 * holds a copy of the view, and so the view's check. A marginal of a mixture or a hybrid gives one
 * per component, which borrows the selection that marginal resolved once, and must not outlive it.
 *
 * mean() and covarianceLower() return lazy Eigen expressions that must not outlive the marginal.
 * covarianceLower() reads only the source's lower triangle. The dimensions are known at compile time
 * when the selected block's sizes are. It is a GaussianForm, with the four fallible accessors of
 * every form, so code generic over the forms takes it as it takes the other forms' marginals,
 * which are parameters (decisions.md [form-densities]).
 *
 * \tparam Source     GaussianParams or Gaussian
 * \tparam Selection  an internal::Coordinates of the index sets of the selected block, from
 *                    idx() for the mean and diffIdx() for the covariance, or an
 *                    internal::SharedSelection of a selection resolved already
 * \tparam SubSpace   space of the selected block
 */
template <typename Source, typename Selection, typename SubSpace>
class GaussianMarginal
{
public:

    using Scalar = typename Source::Scalar;  ///< the scalar type
    using Index = Eigen::Index;              ///< Eigen's index type
    using SpaceType = SubSpace;              ///< the selected block's space

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = valueDimensionAtCompileTime<SubSpace>;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = maxValueDimensionAtCompileTime<SubSpace>;

    /// the length of a difference and the rows of the covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = valueDimensionAtCompileTime<DifferenceSpace<SubSpace>>;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime =
        maxValueDimensionAtCompileTime<DifferenceSpace<SubSpace>>;

    using ParamsType = GaussianParams<SubSpace, Scalar>;   ///< what copy() returns
    using MomentsType = typename ParamsType::MomentsType;  ///< the mean and covariance together
    using VectorType = typename ParamsType::VectorType;    ///< a value, such as the mean
    using MatrixType = typename ParamsType::MatrixType;    ///< a matrix over differences

    static constexpr bool IsInfoForm = false;  ///< a block of the covariance, not of the precision [gaussian-forms]
    static constexpr bool IsSqrtForm = false;  ///< a block of the covariance whole, not of a factor [gaussian-forms]

    /// the selection, resolved
    using ResolvedType = typename internal::SelectionHolder<Source, Selection, SubSpace>::ResolvedType;

    /** The selected mean entries' indices. */
    using IndexVectorType = decltype(ResolvedType::indices);

    /** The selected covariance rows' indices. */
    using DifferenceIndexVectorType = decltype(ResolvedType::differenceIndices);

    /** Selects \a selection's coordinates, which form \a space. */
    GaussianMarginal(Source const & source, Selection const & selection, SubSpace space)
        requires(!internal::isSharedSelection<Selection>)
        : m_holder{source}
        , m_selection{source, selection, std::move(space)}
    {}

    /** Selects the coordinates of \a source that \a shared resolved, borrowing them. */
    GaussianMarginal(Source const & source, Selection shared)
        requires(internal::isSharedSelection<Selection>)
        : m_holder{source}
        , m_selection{shared}
    {}

    /** \returns the number of selected mean entries */
    [[nodiscard]] Index
    valueDimension() const
    {
        return indices().size();
    }

    /** \returns the number of selected covariance rows */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return differenceIndices().size();
    }

    /** \returns the selected block's space, with its parameters and size */
    [[nodiscard]] SubSpace const &
    space() const
    {
        return m_selection.get().space;
    }

    /** \returns the source's mean entries this marginal selects */
    [[nodiscard]] IndexVectorType const &
    indices() const
    {
        return m_selection.get().indices;
    }

    /** \returns the source's covariance rows this marginal selects */
    [[nodiscard]] DifferenceIndexVectorType const &
    differenceIndices() const
    {
        return m_selection.get().differenceIndices;
    }

    /** \returns the selected entries of the source's mean, as a lazy expression */
    [[nodiscard]] auto
    mean() const
    {
        return source().mean()(indices());
    }

    /** \returns the selected block of the source's covariance, read from its lower triangle, as a
     * lazy expression
     */
    [[nodiscard]] auto
    covarianceLower() const
    {
        return internal::lowerSymmetricBlock(source().covarianceLower(), differenceIndices(), differenceIndices());
    }

    /** \returns the selected mean and covariance, evaluated */
    [[nodiscard]] MomentsType
    moments() const
    {
        return MomentsType{mean(), covarianceLower()};
    }

    /** \returns an independent %Gaussian holding this marginal, in its space */
    [[nodiscard]] ParamsType
    copy() const
    {
        return ParamsType{space(), moments()};
    }

    /** \returns the selected block of the covariance, evaluated. Never fails: it returns
     * \c std::expected, as GaussianParams::covariance() does, so that a marginal is a GaussianForm
     * (decisions.md [form-densities]).
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        MatrixType result = covarianceLower();
        return result;
    }

    /** \returns the precision of the selected block, the inverse of its covariance, or the
     * GaussianError of its factorization, which a view does not cache (decisions.md [view-stamp]) and so
     * makes on each call
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precision() const
    {
        auto const llt = factorization();
        if (!llt)
        {
            return std::unexpected{llt.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(*llt, result);
        return result;
    }

    /** \returns the lower-triangular Cholesky factor \f$ L \f$ of the selected block's covariance,
     * as GaussianParams::covarianceRoot() gives it, or the GaussianError of its factorization
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covarianceRoot() const
    {
        auto const llt = factorization();
        if (!llt)
        {
            return std::unexpected{llt.error()};
        }
        MatrixType result = llt->matrixL();
        return result;
    }

    /** \returns \f$ L^{-1} \f$, lower triangular, as GaussianParams::precisionRoot() gives it, or
     * the GaussianError of the factorization
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precisionRoot() const
    {
        auto const llt = factorization();
        if (!llt)
        {
            return std::unexpected{llt.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(llt->matrixL(), result);
        return result;
    }

private:

    /** \returns the viewed source */
    [[nodiscard]] Source const &
    source() const
    {
        return m_holder.get();
    }

    /** \returns the Cholesky factorization of the selected block's covariance, or
     * GaussianError::NotPositiveDefinite or GaussianError::NonFinite as GaussianParams::factorize()
     * returns them
     */
    [[nodiscard]] std::expected<Eigen::LLT<MatrixType>, GaussianError>
    factorization() const
    {
        auto llt = Eigen::LLT<MatrixType>{MatrixType{covarianceLower()}};
        if (llt.info() != Eigen::Success)
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        // As in GaussianParams::factorize(): NaN passes LLT's test but reaches the diagonal.
        if (!allFinite(llt.matrixLLT().diagonal()))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        return llt;
    }

    internal::HolderFor<Source> m_holder;  ///< the check of parameters, or a copy of a view
    internal::SelectionHolder<Source, Selection, SubSpace> m_selection;  ///< the selected coordinates and their space
};

namespace internal
{

/** Shared implementation of marginal<Path>(): the marginal over idx() and diffIdx() of
 * \a Path, whose space is the named subspace.
 */
template <template <typename, typename, typename> typename Marginal, typename Path, typename Source>
[[nodiscard]] auto
tagMarginal(Source const & source)
{
    auto const selection =
        Coordinates{asIndexSet(idx<Path>(source.space())), asIndexSet(diffIdx<Path>(source.space()))};
    auto space = pathSubspace<Path>(source.space());
    return Marginal<Source, std::remove_cvref_t<decltype(selection)>, decltype(space)>{
        source,
        selection,
        std::move(space)
    };
}

/** Shared implementation of marginal<Path>(element): the marginal over idx() and
 * diffIdx() of \a element, whose space is that of one repetition, or of the rest of the
 * path within it.
 */
template <template <typename, typename, typename> typename Marginal, typename Path, typename Source>
[[nodiscard]] auto
elementMarginal(Source const & source, Eigen::Index element)
{
    auto const selection =
        Coordinates{asIndexSet(idx<Path>(source.space(), element)), asIndexSet(diffIdx<Path>(source.space(), element))};
    auto space = pathElementSubspace<Path>(source.space());
    return Marginal<Source, std::remove_cvref_t<decltype(selection)>, decltype(space)>{
        source,
        selection,
        std::move(space)
    };
}

}  // namespace internal

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a gaussian: its dimension, mean, and covariance. */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, GaussianParams<Space_, Scalar_> const & gaussian, Indent indent)
{
    stream << indent << "Gaussian";
    printDimension(stream, gaussian.valueDimension(), gaussian.differenceDimension());
    printMoments(stream, indent.deeper(), gaussian.moments());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
