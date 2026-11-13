#pragma once

/** \file
 * The square-root form of the %Gaussian (decisions.md [gaussian-forms]): its parameters SqrtGaussianParams, which
 * holds a mean and a lower-triangular root of the covariance, its marginal<Path>() by
 * triangularization, and toSqrt(), which converts any form to it. Its density is a
 * SqrtGaussian, a FormGaussian. It is a Component, by ComponentTraits in Component.h, with
 * internal::SqrtComponentMarginal, the marginal of one component. Read InfoGaussian.h next.
 */

#include <expected>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Triangularization.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

template <typename C>
struct ComponentTraits;

namespace internal
{

template <typename Source, typename Selection, typename SubSpace>
class SqrtComponentMarginal;

}  // namespace internal

/** \ingroup Distributions
 * \brief A %Gaussian kept as a mean and a lower-triangular root of its covariance.
 *
 * The [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
 * with covariance \f$ \Sigma = S S^T \f$, where the lower-triangular \f$ S \f$ is what is stored
 * and the covariance is formed only on request (decisions.md [gaussian-forms]). A root is positive
 * semidefinite by construction and represents the covariance at twice the precision of the
 * covariance itself, so a filter that keeps it, updating it by triangularize() rather than by
 * adding and subtracting covariances, never needs the repair that a Repair makes after the
 * fact. The density, its whitening, and draws come from \f$ S \f$ directly, where Gaussian has
 * to factorize first.
 *
 * Only the lower triangle of a root passed in is kept; the upper is set to zero. The diagonal
 * may have either sign, since flipping the sign of a column of \f$ S \f$ leaves \f$ S S^T \f$
 * unchanged. A zero on the diagonal makes the covariance singular: covarianceRoot() and
 * covariance() still succeed, and precision(), precisionRoot(), and toDensity() return
 * GaussianError::NotPositiveDefinite.
 *
 * The space and dimension contract is Gaussian's [sizes-and-dimensions]: the mean is a value with
 * valueDimension() coordinates, normalized whenever it is set, and the root is square over its
 * differences. toDensity() returns a FormGaussian, which checks this object for changes as
 * Gaussian checks a GaussianParams. marginal<Path>() returns parameters, since its
 * root is computed rather than selected; a conditional, insert(), and erase() go through
 * toGaussian() and back.
 *
 * It is a Component (decisions.md [component-first]), so a MixtureParams or a Hybrid holds it as it
 * holds GaussianParams: factorize() checks the root and caches its log-determinant, and the
 * collapse of several is one triangularization, so a mixture filter or an IMM over the
 * square-root form keeps roots throughout.
 *
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 *
 * \sa toSqrt(), class GaussianParams, GaussianForm
 */
template <Space Space_, RealScalar Scalar_ = double>
class SqrtGaussianParams
{
public:

    using MomentsType = Moments<Space_, Scalar_>;                             ///< the mean and covariance together
    using Scalar = typename MomentsType::Scalar;                              ///< the scalar type
    using Index = typename MomentsType::Index;                                ///< Eigen's index type
    using VectorType = typename MomentsType::VectorType;                      ///< a value, such as the mean
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;  ///< a difference, such as a residual
    using MatrixType = typename MomentsType::MatrixType;                      ///< a matrix over differences
    using SpaceType = Space_;                                                 ///< the space the values live in
    using DensityType = FormGaussian<SqrtGaussianParams>;                     ///< the view toDensity() returns

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the root, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;     ///< parameters, which views check
    static constexpr bool IsInfoForm = false;  ///< holds a root of the covariance, not the precision
    static constexpr bool IsSqrtForm = true;   ///< holds a triangular root

    /** Constructs a SqrtGaussianParams in a default space, with a zero mean and root. */
    SqrtGaussianParams()
        : SqrtGaussianParams{Space_{}}
    {}

    /** Constructs a SqrtGaussianParams in \a space, with a zero mean and root. */
    explicit SqrtGaussianParams(Space_ space)
        : m_space{std::move(space)}
    {
        m_mean.setZero(Emrys::valueDimension(m_space));
        m_root.setZero(differenceDimension(), differenceDimension());
    }

    /** Constructs a SqrtGaussianParams from \a mean and the lower triangle of \a root, moving
     * either one passed as an rvalue of the stored type, in a default space resized to fit them.
     */
    template <MatrixExpression Mean, MatrixExpression Root>
        requires internal::SizedByLength<Space_>
    SqrtGaussianParams(Mean && mean, Root && root)
        : m_mean{std::forward<Mean>(mean)}
        , m_root{std::forward<Root>(root)}
    {
        fitSpace();
    }

    /** SqrtGaussianParams(mean, root) in \a space, which is resized to fit them if it has one
     * resizable block.
     */
    template <MatrixExpression Mean, MatrixExpression Root>
    SqrtGaussianParams(Space_ space, Mean && mean, Root && root)
        : m_mean{std::forward<Mean>(mean)}
        , m_root{std::forward<Root>(root)}
        , m_space{std::move(space)}
    {
        fitSpace();
    }

    /** \returns the number of coordinates of the mean (see \ref Spaces) */
    [[nodiscard]] Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the number of rows and columns of the root (see \ref Spaces) */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

    /** \returns the space, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the mean, a value in the space */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return m_mean;
    }

    /** \returns the stored lower-triangular root \f$ S \f$, \f$ S S^T = \Sigma \f$, with zeros above the
     * diagonal [roots-and-factors]
     */
    [[nodiscard]] MatrixType const &
    root() const
    {
        return m_root;
    }

    /** \returns the mean and the covariance \f$ S S^T \f$ together, formed on request */
    [[nodiscard]] MomentsType
    moments() const
    {
        return MomentsType{m_mean, *(covariance())};
    }

    /** Sets the mean, normalized by the space, moving \a mean if it is an rvalue of the stored
     * type.
     */
    template <MatrixExpression Mean>
    void
    setMean(Mean && mean)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(mean.size() == valueDimension());
        m_mean = std::forward<Mean>(mean);
        normalizeValues(m_space, m_mean);
    }

    /** Sets the root to the lower triangle of \a root, moving it if it is an rvalue of the
     * stored type.
     */
    template <MatrixExpression Root>
    void
    setRoot(Root && root)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(root.rows() == differenceDimension() && root.cols() == differenceDimension());
        m_root = std::forward<Root>(root);
        m_root.template triangularView<Eigen::StrictlyUpper>().setZero();
    }

    /** \returns the covariance \f$ S S^T \f$, formed on request. Never fails. */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        MatrixType result = internal::multiply(internal::asTriangular<Eigen::Lower>(m_root), m_root.transpose());
        return result;
    }

    /** \returns the precision \f$ S^{-T} S^{-1} \f$, or GaussianError::NotPositiveDefinite if the
     * root has a zero on its diagonal, or GaussianError::NonFinite if it is not finite.
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precision() const
    {
        auto inverse = precisionRoot();
        if (!inverse)
        {
            return std::unexpected{inverse.error()};
        }
        MatrixType result = internal::multiply(inverse->transpose(), *inverse);
        return result;
    }

    /** \returns the stored root \f$ S \f$. Never fails. */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covarianceRoot() const
    {
        return m_root;
    }

    /** \returns \f$ S^{-1} \f$, lower triangular, with \f$ (S^{-1})^T S^{-1} = \Sigma^{-1} \f$, or the
     * GaussianError of precision()
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precisionRoot() const
    {
        if (auto checked = checkRoot(); !checked)
        {
            return std::unexpected{checked.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(m_root.template triangularView<Eigen::Lower>(), result);
        return result;
    }

    /** \returns the marginal of the subspace named by the path \a Path (a tag or a TagPath), in
     * that subspace, as a SqrtGaussianParams: the mean's entries, and the lower-triangular root
     * of \f$ S_k S_k^T \f$ for the root's rows \f$ S_k \f$ at the block's differences, by
     * triangularization (internal::lowerTriangularRoot()), so the block's covariance is never
     * formed. Never fails. Unlike GaussianParams::marginal(), it returns parameters, since the
     * root is computed rather than selected.
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::marginalOverPath<Path>(m_space, [&](auto const & kept, auto const & keptSpace) {
            return marginalAt(kept, keptSpace);
        });
    }

    /** marginal<Path>() of repetition \a element of the repeated block \a Path starts at (see
     * GaussianParams::marginal(element)).
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const
    {
        return internal::marginalOverElement<Path>(m_space, element, [&](auto const & kept, auto const & keptSpace) {
            return marginalAt(kept, keptSpace);
        });
    }

    /** Checks the root and computes \f$ \log|\Sigma| = 2 \sum_i \log|S_{ii}| \f$, writing it to
     * a cache when it has changed, so that a container of components can read their densities
     * (see Component).
     *
     * \returns nothing, or GaussianError::NonFinite or GaussianError::NotPositiveDefinite as
     *          precision() returns them
     */
    [[nodiscard]] std::expected<void, GaussianError>
    factorize() const
    {
        if (auto checked = checkRoot(); !checked)
        {
            return std::unexpected{checked.error()};
        }
        // Unchanged, it writes nothing, so factorizing what is factorized only reads.
        Scalar const logDeterminant = Scalar(2) * m_root.diagonal().array().abs().log().sum();
        if (m_logDeterminant != logDeterminant)
        {
            m_logDeterminant = logDeterminant;
        }
        return {};
    }

    /** Refused: a root needs no repair, being positive semidefinite by construction, so a mixture of
     * square-root components given a repair does not compile.
     */
    template <typename R>
    [[nodiscard]] std::expected<void, GaussianError>
    factorize(R const & /*repair*/) const
    {
        static_assert(
            sizeof(R) == 0,
            "a square-root form's root is positive semidefinite by construction and has no repair; repairing a "
            "covariance is the covariance form's (decisions.md [gaussian-forms])"
        );
        return factorize();
    }

    /** factorize(), so that the cache it writes is not safe on one object from two threads at
     * once while the root changes. Threads should share the density instead (see
     * \ref Distributions).
     *
     * \returns a FormGaussian that checks this object, or the GaussianError of factorize()
     */
    [[nodiscard]] std::expected<DensityType, GaussianError>
    toDensity() const &
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

private:

    friend DensityType;  ///< the view reads the mean, the log-determinant, and the root

    /** A container's density reads the log-determinant without a view. */
    friend struct ComponentTraits<SqrtGaussianParams>;

    /** A container's marginal makes each component's from a resolved selection. */
    template <typename, typename, typename>
    friend class internal::SqrtComponentMarginal;

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** \returns nothing if the root is finite with no zero on its diagonal, otherwise the
     *           GaussianError that says which
     */
    [[nodiscard]] std::expected<void, GaussianError>
    checkRoot() const
    {
        if (!allFinite(m_root))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        if ((m_root.diagonal().array() == Scalar(0)).any())
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        return {};
    }

    /** \returns the marginal over the resolved coordinates \a kept, which form \a keptSpace: the
     * mean's entries and the lower-triangular root of the selected rows of \f$ S \f$ (see
     * marginal<Path>())
     */
    template <typename Kept, typename KeptSpace>
    [[nodiscard]] SqrtGaussianParams<KeptSpace, Scalar>
    marginalAt(Kept const & kept, KeptSpace const & keptSpace) const
    {
        using Result = SqrtGaussianParams<KeptSpace, Scalar>;
        using Rows = internal::MatrixOf<
            Scalar,
            Result::DifferenceDimensionAtCompileTime,
            DifferenceDimensionAtCompileTime,
            Result::MaxDifferenceDimensionAtCompileTime,
            MaxDifferenceDimensionAtCompileTime>;
        typename Result::VectorType mean = m_mean(kept.values);
        Rows const rows = m_root(kept.differences, Eigen::placeholders::all);
        return Result{keptSpace, std::move(mean), internal::lowerTriangularRoot(rows)};
    }

    /** \returns marginalAt(kept, keptSpace), for FormGaussian, since it never fails */
    template <typename Kept, typename KeptSpace>
    [[nodiscard]] SqrtGaussianParams<KeptSpace, Scalar>
    densityMarginalAt(Kept const & kept, KeptSpace const & keptSpace) const
    {
        return marginalAt(kept, keptSpace);
    }

    /** \returns moments(), for FormGaussian */
    [[nodiscard]] MomentsType
    densityMoments() const
    {
        return moments();
    }

    /** \returns the mean, for FormGaussian */
    [[nodiscard]] VectorType const &
    densityMean() const
    {
        return m_mean;
    }

    /** Whitens \a residual in place: \f$ S^{-1} r \f$. */
    void
    whitenInPlace(DifferenceVectorType & residual) const
    {
        m_root.template triangularView<Eigen::Lower>().solveInPlace(residual);
    }

    /** Colors the standard normals in the columns of \a normals in place: \f$ S z \f$. */
    void
    colorInto(auto & colored, auto const & normals) const
    {
        colored.noalias() = internal::multiply(internal::asTriangular<Eigen::Lower>(m_root), normals);
    }

    /** Resizes a space with one resizable block to the mean, checks the sizes, normalizes the
     * mean, and clears the root's upper triangle.
     */
    void
    fitSpace()
    {
        internal::fitValueDimension(m_space, m_mean.size());
        EMRYS_FILTER_ASSERT(
            m_root.rows() == differenceDimension() && m_root.cols() == differenceDimension()
            && "the root must have a row and a column per difference coordinate of the space"
        );
        normalizeValues(m_space, m_mean);
        m_root.template triangularView<Eigen::StrictlyUpper>().setZero();
    }

    VectorType m_mean;                              ///< the mean
    MatrixType m_root;                              ///< the lower-triangular root \f$ S \f$
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space, kept in step with the mean's size
    mutable Scalar m_logDeterminant{};              ///< \f$ \log|\Sigma| \f$, set by toDensity()
    internal::Stamp m_stamp;                        ///< what views check [view-stamp]
};

namespace internal
{

/** The marginal of one SqrtGaussianParams component of a mixture or a hybrid, over the coordinates
 * the container's marginal resolved once (an internal::SharedSelection, \a Selection), which form
 * \a SubSpace: parameters, as SqrtGaussianParams::marginal<Path>() returns, since the root of the
 * selected rows is computed by triangularization rather than selected (decisions.md
 * [element-marginals]).
 */
template <typename Source, typename Selection, typename SubSpace>
class SqrtComponentMarginal : public SqrtGaussianParams<SubSpace, typename Source::Scalar>
{
public:

    using ParamsType = SqrtGaussianParams<SubSpace, typename Source::Scalar>;  ///< what copy() returns

    /** The marginal of \a source over the coordinates \a shared resolved. */
    SqrtComponentMarginal(Source const & source, Selection shared)
        : ParamsType{source.marginalAt(
              Coordinates<
                  decltype(shared.resolved->indices) const &,
                  decltype(shared.resolved->differenceIndices) const &>{
                  shared.resolved->indices,
                  shared.resolved->differenceIndices
              },
              shared.resolved->space
          )}
    {}

    /** \returns the marginal's parameters */
    [[nodiscard]] ParamsType
    copy() const
    {
        return *this;
    }
};

}  // namespace internal

/** \ingroup Distributions
 * \brief The %Gaussian in square-root form, which evaluates its density: what
 * SqrtGaussianParams::toDensity() returns (decisions.md [gaussian-forms], [params-and-density]). It reads its
 * parameters' cached factorization and must not outlive them [view-stamp].
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 */
template <Space Space_, RealScalar Scalar_ = double>
using SqrtGaussian = FormGaussian<SqrtGaussianParams<Space_, Scalar_>>;

/** \ingroup Distributions
 * \returns \a form in the square-root form: its mean and a lower-triangular root of its
 * covariance in a SqrtGaussianParams in the same space, or the GaussianError of either.
 *
 * The root is covarianceRoot() itself in the covariance forms, which have it lower
 * triangular already. An information form gives \f$ R^{-1} \f$, upper triangular, which
 * triangularize() turns into the lower-triangular root of the same covariance without forming
 * it; and since that covariance lives at the form's reference, on a space with a reset Jacobian
 * \f$ G \f$ the root is carried to the mean as \f$ \mathrm{tri}(G S) \f$ (decisions.md [covariance-reset], as
 * toGaussian() carries the covariance).
 *
 * \sa toGaussian(), toInfo(), toSqrtInfo()
 */
template <GaussianForm G>
[[nodiscard]] std::expected<SqrtGaussianParams<typename G::SpaceType, typename G::Scalar>, GaussianError>
toSqrt(G const & form)
{
    using Result = SqrtGaussianParams<typename G::SpaceType, typename G::Scalar>;
    std::expected<typename G::VectorType, GaussianError> mean = form.mean();
    if (!mean)
    {
        return std::unexpected{mean.error()};
    }
    auto root = form.covarianceRoot();
    if (!root)
    {
        return std::unexpected{root.error()};
    }
    if constexpr (isInfoForm<G>)
    {
        typename Result::MatrixType lower = internal::lowerTriangularRoot(*root);
        if constexpr (!hasIdentityReset<typename G::SpaceType>)
        {
            auto const step = Emrys::minus(form.space(), *mean, form.reference());
            auto const g = resetJacobian(form.space(), form.reference(), step);
            lower = internal::lowerTriangularRoot(typename Result::MatrixType{internal::multiply(g, lower)});
        }
        return Result{form.space(), std::move(*mean), std::move(lower)};
    }
    else
    {
        return Result{form.space(), std::move(*mean), std::move(*root)};
    }
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a gaussian: its dimension, mean, and covariance. */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, SqrtGaussianParams<Space_, Scalar_> const & gaussian, Indent indent)
{
    stream << indent << "square-root Gaussian";
    printDimension(stream, gaussian.valueDimension(), gaussian.differenceDimension());
    printMoments(stream, indent.deeper(), gaussian.moments());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
