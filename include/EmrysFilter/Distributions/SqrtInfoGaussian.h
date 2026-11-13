#pragma once

/** \file
 * The square-root information form of the %Gaussian (decisions.md [gaussian-forms]): its parameters
 * SqrtInfoGaussianParams, which holds an upper-triangular root of the precision and the
 * whitened offset of the mean from a reference value; multiply(), marginal<Path>(), and
 * conditional<Path>(), each a triangularization of a stacked data array; and
 * toSqrtInfo(), which converts any form to it. Its density is a SqrtInfoGaussian, a FormGaussian.
 * Read Uniform.h next.
 */

#include <expected>
#include <type_traits>
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

/** \ingroup Distributions
 * \brief A %Gaussian kept as an upper-triangular root of its precision and a whitened offset,
 * about a reference value.
 *
 * The square-root information form of \ref Bierman1977 "Bierman (1977)"
 * (\ref Ke2024 "Ke et al. (2024)", §II): an upper-triangular \f$ R \f$ with \f$ R^T R = \Lambda \f$, and
 * \f$ r = R \delta \f$, where the mean is \f$ \mu = x_0 \boxplus \delta \f$ for the stored
 * reference value \f$ x_0 \f$. The pair is the data array \f$ [\,R \;\; z\,] \f$, with the state
 * solving \f$ R x = z \f$; here \f$ x \f$ is the offset from the reference, as in InfoGaussianParams, so
 * that the information lives in the differences at \f$ x_0 \f$ on a space with \f$ \boxplus \f$,
 * and moving the reference carries it by the reset Jacobian of decisions.md [covariance-reset]
 * (setReference()).
 *
 * It combines the information form's strengths with the square root's: independent information
 * is combined by stacking whitened rows under \f$ [\,R \;\; r\,] \f$ and triangularizing
 * (triangularize(), multiply()), which never fails and never forms \f$ \Lambda \f$, and a prior
 * with no information is \f$ R = 0 \f$. The mean, the covariance, and the density are triangular
 * solves with \f$ R \f$, which return GaussianError::Singular while it has a zero on its diagonal.
 *
 * Only the upper triangle of a root passed in is kept; the lower is set to zero. The diagonal
 * may have either sign, since flipping a row of \f$ [\,R \;\; r\,] \f$ leaves \f$ R^T R \f$ and
 * \f$ \delta \f$ unchanged. The matrices the accessors give are in the differences at the
 * reference; moments() and the conversions to a covariance form carry the covariance to the
 * differences at the mean [covariance-reset]. toDensity() returns a FormGaussian, which checks this object for
 * changes as Gaussian checks a GaussianParams. marginal<Path>() and conditional<Path>() return
 * parameters, since their roots are computed rather than selected.
 *
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 *
 * \sa toSqrtInfo(), multiply(), class InfoGaussianParams, GaussianForm
 */
template <Space Space_, RealScalar Scalar_ = double>
class SqrtInfoGaussianParams
{
public:

    using MomentsType = Moments<Space_, Scalar_>;                             ///< the mean and covariance together
    using Scalar = typename MomentsType::Scalar;                              ///< the scalar type
    using Index = typename MomentsType::Index;                                ///< Eigen's index type
    using VectorType = typename MomentsType::VectorType;                      ///< a value, such as the reference
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;  ///< a difference, or the whitened offset
    using MatrixType = typename MomentsType::MatrixType;                      ///< a matrix over differences
    using SpaceType = Space_;                                                 ///< the space the values live in
    using DensityType = FormGaussian<SqrtInfoGaussianParams>;                 ///< the view toDensity() returns

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the root, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;    ///< parameters, which views check
    static constexpr bool IsInfoForm = true;  ///< holds a root of the precision
    static constexpr bool IsSqrtForm = true;  ///< holds a triangular root

    /** Constructs a SqrtInfoGaussianParams in a default space, with a zero reference and no
     * information.
     */
    SqrtInfoGaussianParams()
        : SqrtInfoGaussianParams{Space_{}}
    {}

    /** Constructs a SqrtInfoGaussianParams in \a space, with a zero reference and no information. */
    explicit SqrtInfoGaussianParams(Space_ space)
        : m_space{std::move(space)}
    {
        m_reference.setZero(Emrys::valueDimension(m_space));
        m_root.setZero(differenceDimension(), differenceDimension());
        m_whitenedOffset.setZero(differenceDimension());
        m_densityMean.setZero(valueDimension());
    }

    /** Constructs a SqrtInfoGaussianParams about \a reference, with the upper triangle of
     * \a root as \f$ R \f$ and \a whitenedOffset as \f$ r \f$, moving any passed as an rvalue of
     * the stored type, in a default space resized to fit them.
     */
    template <MatrixExpression Reference, MatrixExpression Root, MatrixExpression Offset>
        requires internal::SizedByLength<Space_>
    SqrtInfoGaussianParams(Reference && reference, Root && root, Offset && whitenedOffset)
        : m_reference{std::forward<Reference>(reference)}
        , m_root{std::forward<Root>(root)}
        , m_whitenedOffset{std::forward<Offset>(whitenedOffset)}
    {
        fitSpace();
    }

    /** SqrtInfoGaussianParams(reference, root, whitenedOffset) in \a space, which is
     * resized to fit them if it has one resizable block.
     */
    template <MatrixExpression Reference, MatrixExpression Root, MatrixExpression Offset>
    SqrtInfoGaussianParams(Space_ space, Reference && reference, Root && root, Offset && whitenedOffset)
        : m_reference{std::forward<Reference>(reference)}
        , m_root{std::forward<Root>(root)}
        , m_whitenedOffset{std::forward<Offset>(whitenedOffset)}
        , m_space{std::move(space)}
    {
        fitSpace();
    }

    /** \returns the number of coordinates of a value (see \ref Spaces) */
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

    /** \returns the reference value \f$ x_0 \f$, at whose differences the information lives */
    [[nodiscard]] VectorType const &
    reference() const
    {
        return m_reference;
    }

    /** \returns the stored upper-triangular root \f$ R \f$ of the precision, \f$ R^T R = \Lambda \f$, with
     * zeros below the diagonal: named for the information form, since \f$ R \f$ is no root of the
     * covariance [roots-and-factors]
     */
    [[nodiscard]] MatrixType const &
    infoRoot() const
    {
        return m_root;
    }

    /** \returns the whitened offset \f$ r = R \delta \f$ of the mean from the reference */
    [[nodiscard]] DifferenceVectorType const &
    whitenedOffset() const
    {
        return m_whitenedOffset;
    }

    /** \returns the information vector \f$ \eta = \Lambda \delta = R^T r \f$ about the reference */
    [[nodiscard]] DifferenceVectorType
    infoVector() const
    {
        DifferenceVectorType result =
            internal::multiply(internal::asTriangular<Eigen::Upper>(m_root).transpose(), m_whitenedOffset);
        return result;
    }

    /** Moves the reference to \a reference, normalized by the space, and re-expresses the data
     * array about it. With \f$ d = x_0' \boxminus x_0 \f$ the step and \f$ G \f$ the reset Jacobian
     * of the space at \f$ x_0 \f$ and \f$ d \f$ (decisions.md [covariance-reset], resetJacobian()), the new
     * array is the triangularization of \f$ [\,R G^{-1} \;\; r - R d\,] \f$, which is
     * InfoGaussianParams::setReference()'s \f$ \Lambda' = G^{-T} \Lambda G^{-1} \f$ and
     * \f$ \eta' = G^{-T}(\eta - \Lambda d) \f$ in factored form. On a space with an identity reset
     * \f$ G \f$ is the identity and only \f$ r \f$ changes, exactly.
     */
    template <typename Derived>
    void
    setReference(Eigen::MatrixBase<Derived> const & reference)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(reference.size() == valueDimension());
        VectorType moved = reference;
        normalizeValues(m_space, moved);
        DifferenceVectorType const shift = Emrys::minus(m_space, moved, m_reference);
        m_whitenedOffset.noalias() -= internal::multiply(internal::asTriangular<Eigen::Upper>(m_root), shift);
        if constexpr (!hasIdentityReset<Space_>)
        {
            Eigen::Index const n = differenceDimension();
            AugmentedType stacked(n, n + 1);
            stacked.leftCols(n) = m_root;
            internal::forEachInverseResetJacobian(
                m_space,
                m_reference,
                shift,
                moved,
                [&](Eigen::Index differences, auto const & gInverse) {
                    Eigen::Index const k = gInverse.rows();
                    stacked.middleCols(differences, k) = m_root.middleCols(differences, k) * gInverse;
                }
            );
            stacked.col(n) = m_whitenedOffset;
            triangularize(stacked);
            m_root = stacked.leftCols(n);
            m_whitenedOffset = stacked.col(n);
        }
        m_reference = std::move(moved);
    }

    /** Sets \f$ R \f$ to the upper triangle of \a root and \f$ r \f$ to \a whitenedOffset, about
     * the current reference, moving either one passed as an rvalue of the stored type.
     */
    template <MatrixExpression Root, MatrixExpression Offset>
    void
    setRoot(Root && root, Offset && whitenedOffset)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(
            root.rows() == differenceDimension() && root.cols() == differenceDimension()
            && whitenedOffset.size() == differenceDimension()
        );
        m_root = std::forward<Root>(root);
        m_root.template triangularView<Eigen::StrictlyLower>().setZero();
        m_whitenedOffset = std::forward<Offset>(whitenedOffset);
    }

    /** \returns the mean \f$ x_0 \boxplus R^{-1} r \f$, or GaussianError::Singular if \f$ R \f$ has a
     * zero on its diagonal, or GaussianError::NonFinite if it is not finite
     */
    [[nodiscard]] std::expected<VectorType, GaussianError>
    mean() const
    {
        if (auto checked = checkRoot(); !checked)
        {
            return std::unexpected{checked.error()};
        }
        return solvedMean();
    }

    /** \returns the mean and the covariance together, by triangular solves, the covariance carried
     * to the differences at the mean as toGaussian() carries it, or the GaussianError of mean()
     */
    [[nodiscard]] std::expected<MomentsType, GaussianError>
    moments() const
    {
        if (auto checked = checkRoot(); !checked)
        {
            return std::unexpected{checked.error()};
        }
        return momentsWithMean(solvedMean());
    }

    /** \returns the covariance \f$ R^{-1} R^{-T} \f$, in the differences at the reference, or the
     * GaussianError of mean()
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        auto inverse = covarianceRoot();
        if (!inverse)
        {
            return std::unexpected{inverse.error()};
        }
        MatrixType result = internal::multiply(*inverse, inverse->transpose());
        return result;
    }

    /** \returns the precision \f$ R^T R \f$, formed on request. Never fails. */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precision() const
    {
        MatrixType result = internal::multiply(m_root.transpose(), internal::asTriangular<Eigen::Upper>(m_root));
        return result;
    }

    /** \returns \f$ R^{-1} \f$, upper triangular, with \f$ R^{-1} R^{-T} = \Sigma \f$, or the
     * GaussianError of mean()
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covarianceRoot() const
    {
        if (auto checked = checkRoot(); !checked)
        {
            return std::unexpected{checked.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(m_root.template triangularView<Eigen::Upper>(), result);
        return result;
    }

    /** \returns the stored root \f$ R \f$. Never fails. */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precisionRoot() const
    {
        return m_root;
    }

    /** Checks the root, solves for the mean, and computes
     * \f$ \log|\Sigma| = -2 \sum_i \log|R_{ii}| \f$, writing both to a cache each time, so it is
     * not safe on one object from two threads at once. Threads should share the density instead
     * (see \ref Distributions).
     *
     * \returns a FormGaussian that checks this object, or the GaussianError of mean()
     */
    [[nodiscard]] std::expected<DensityType, GaussianError>
    toDensity() const &
    {
        if (auto checked = checkRoot(); !checked)
        {
            return std::unexpected{checked.error()};
        }
        m_densityMean = solvedMean();
        m_logDeterminant = Scalar(-2) * m_root.diagonal().array().abs().log().sum();
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

    /** \returns the marginal of the subspace named by the path \a Path (a tag or a TagPath), in
     * that subspace, as a SqrtInfoGaussianParams about the matching entries of the
     * reference, or GaussianError::Singular if the rest's root is singular.
     *
     * The data array's columns are reordered with the rest \f$ r \f$ first and the kept block
     * \f$ k \f$ after it, and triangularized:
     * \f[ Q^T \begin{bmatrix} R_{:r} & R_{:k} & r \end{bmatrix}
     *     = \begin{bmatrix} R_{rr} & R_{rk} & r_r \\ 0 & R_{kk} & r_k \end{bmatrix}, \f]
     * whose trailing block is the marginal, \f$ R_{kk}^T R_{kk} \f$ being the
     * [Schur complement](https://en.wikipedia.org/wiki/Schur_complement) of
     * \f$ \Lambda_{rr} \f$ in \f$ \Lambda \f$ and \f$ R_{kk}^T r_k \f$ the matching information
     * vector, when \f$ R_{rr} \f$ is nonsingular. This is the orthogonal elimination of variables
     * from a data array of \ref Bierman1977 "Bierman (1977)", as \ref Ke2024 "Ke et al. (2024)",
     * §III-D, apply it to marginalize states; a time update eliminates the process noise, and here
     * it eliminates the rest of the state. Nothing is inverted, and a bounded space allocates nothing.
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

    /** \returns the distribution of the other coordinates given that the subspace named by the
     * path \a GivenPath (a tag or a TagPath) equals \a values, which are in that subspace: a
     * SqrtInfoGaussianParams in the space of the remaining blocks, about the matching
     * entries of the reference.
     *
     * With \f$ \delta_g = v \boxminus x_{0,g} \f$ the given offset, the target's data array is the
     * triangularization of \f$ [\,R_{:t} \;\; r - R_{:g} \delta_g\,] \f$, whose top rows give
     * \f$ R_{tt}^T R_{tt} = \Lambda_{tt} \f$ and \f$ R_{tt}^T r_t = \eta_t - \Lambda_{tg} \delta_g \f$,
     * the conditional of InfoGaussianParams::conditional() in factored form. Never fails, and has
     * no conditionalWithEvidence(), as InfoGaussianParams has none.
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditional<GivenPath>(
            m_space,
            [&](auto const & target, auto const & given, auto const & targetSpace, auto const & givenSpace) {
                using GivenSpace = std::remove_cvref_t<decltype(givenSpace)>;
                using Result = SqrtInfoGaussianParams<std::remove_cvref_t<decltype(targetSpace)>, Scalar>;
                constexpr int GivenSize = valueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>;
                constexpr int GivenMaxSize = maxValueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>;
                constexpr int TargetSize = Result::DifferenceDimensionAtCompileTime;
                constexpr int TargetMaxSize = Result::MaxDifferenceDimensionAtCompileTime;
                using GivenVector = Eigen::Matrix<Scalar, GivenSize, 1, Eigen::ColMajor, GivenMaxSize, 1>;
                using Stacked = internal::MatrixOf<
                    Scalar,
                    DifferenceDimensionAtCompileTime,
                    TargetSize == Eigen::Dynamic ? Eigen::Dynamic : TargetSize + 1,
                    MaxDifferenceDimensionAtCompileTime,
                    TargetMaxSize == Eigen::Dynamic ? Eigen::Dynamic : TargetMaxSize + 1>;
                EMRYS_FILTER_ASSERT(values.size() == given.values.size());

                Eigen::Index const n = differenceDimension();
                Eigen::Index const t = target.differences.size();
                typename Result::VectorType reference = m_reference(target.values);
                auto root = typename Result::MatrixType{};
                root.resize(t, t);
                auto whitened = typename Result::DifferenceVectorType{};
                whitened.resize(t);
                // MSVC can't form the empty products of a conditional on everything or on nothing.
                if constexpr (TargetSize != 0 && GivenSize != 0)
                {
                    GivenVector const offset = Emrys::minus(givenSpace, values, m_reference(given.values));
                    Stacked stacked(n, t + 1);
                    stacked.leftCols(t) = m_root(Eigen::placeholders::all, target.differences);
                    stacked.col(t) = m_whitenedOffset
                                   - internal::multiply(m_root(Eigen::placeholders::all, given.differences), offset);
                    triangularize(stacked);
                    root = stacked.topLeftCorner(t, t);
                    whitened = stacked.col(t).head(t);
                }
                else
                {
                    root = m_root(target.differences, target.differences);
                    whitened = m_whitenedOffset(target.differences);
                }
                return Result{targetSpace, std::move(reference), std::move(root), std::move(whitened)};
            }
        );
    }

private:

    friend DensityType;  ///< the view reads the mean, the log-determinant, and the root

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** The data array \f$ [\,R \;\; r\,] \f$, a column wider than the root. */
    using AugmentedType = internal::MatrixOf<
        Scalar,
        DifferenceDimensionAtCompileTime,
        DifferenceDimensionAtCompileTime == Eigen::Dynamic ? Eigen::Dynamic : DifferenceDimensionAtCompileTime + 1,
        MaxDifferenceDimensionAtCompileTime,
        MaxDifferenceDimensionAtCompileTime == Eigen::Dynamic ? Eigen::Dynamic
                                                              : MaxDifferenceDimensionAtCompileTime + 1>;

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
            return std::unexpected{GaussianError::Singular};
        }
        return {};
    }

    /** \returns the marginal over the resolved coordinates \a kept, which form \a keptSpace, by
     * triangularizing the data array with the rest's columns first (see marginal<Path>())
     */
    template <typename Kept, typename KeptSpace>
    [[nodiscard]] std::expected<SqrtInfoGaussianParams<KeptSpace, Scalar>, GaussianError>
    marginalAt(Kept const & kept, KeptSpace const & keptSpace) const
    {
        using Result = SqrtInfoGaussianParams<KeptSpace, Scalar>;
        using Expected = std::expected<Result, GaussianError>;
        constexpr int D = DifferenceDimensionAtCompileTime;
        constexpr int DMax = MaxDifferenceDimensionAtCompileTime;
        Eigen::Index const n = differenceDimension();
        Eigen::Index const k = kept.differences.size();
        auto const rest =
            internal::resolveIndices<D, DMax>(n, internal::complementIndices<D, DMax>(n, kept.differences));
        Eigen::Index const r = rest.size();

        typename Result::VectorType reference = m_reference(kept.values);
        AugmentedType stacked(n, n + 1);
        stacked.leftCols(r) = m_root(Eigen::placeholders::all, rest);
        stacked.middleCols(r, k) = m_root(Eigen::placeholders::all, kept.differences);
        stacked.col(n) = m_whitenedOffset;
        triangularize(stacked);
        if ((stacked.diagonal().head(r).array() == Scalar(0)).any())
        {
            return Expected{std::unexpected{GaussianError::Singular}};
        }
        typename Result::MatrixType root = stacked.block(r, r, k, k);
        typename Result::DifferenceVectorType whitened = stacked.col(n).tail(k);
        return Expected{Result{keptSpace, std::move(reference), std::move(root), std::move(whitened)}};
    }

    /** \returns marginalAt(kept, keptSpace), for FormGaussian: the rest's columns of a root
     * with no zero on its diagonal are independent, so their triangle has none either
     */
    template <typename Kept, typename KeptSpace>
    [[nodiscard]] SqrtInfoGaussianParams<KeptSpace, Scalar>
    densityMarginalAt(Kept const & kept, KeptSpace const & keptSpace) const
    {
        auto result = marginalAt(kept, keptSpace);
        EMRYS_FILTER_ASSERT(result.has_value() && "a checked root's marginal cannot be singular");
        return std::move(*result);
    }

    /** \returns the mean and the covariance together, with the mean \a mean solved from the
     * checked root and the covariance carried to the differences at it
     */
    [[nodiscard]] MomentsType
    momentsWithMean(VectorType mean) const
    {
        auto result = MomentsType{std::move(mean), *(covariance())};
        if constexpr (!hasIdentityReset<Space_>)
        {
            DifferenceVectorType const step = Emrys::minus(m_space, result.mean(), m_reference);
            resetCovariance(m_space, m_reference, step, result.covarianceLower());
        }
        return result;
    }

    /** \returns the moments with the mean toDensity() solved for, for FormGaussian */
    [[nodiscard]] MomentsType
    densityMoments() const
    {
        return momentsWithMean(m_densityMean);
    }

    /** \returns the mean from the checked root */
    [[nodiscard]] VectorType
    solvedMean() const
    {
        DifferenceVectorType const offset = m_root.template triangularView<Eigen::Upper>().solve(m_whitenedOffset);
        VectorType result = Emrys::plus(m_space, m_reference, offset);
        return result;
    }

    /** \returns the mean toDensity() solved for, for FormGaussian */
    [[nodiscard]] VectorType const &
    densityMean() const
    {
        return m_densityMean;
    }

    /** Whitens \a residual in place: \f$ R r \f$, so that its squared norm is \f$ r^T \Lambda r \f$. */
    void
    whitenInPlace(DifferenceVectorType & residual) const
    {
        residual = internal::multiply(internal::asTriangular<Eigen::Upper>(m_root), residual).eval();
    }

    /** Colors the standard normals in the columns of \a normals in place: \f$ R^{-1} z \f$. */
    void
    colorInto(auto & colored, auto const & normals) const
    {
        colored = normals;
        internal::solveInPlace(m_root.template triangularView<Eigen::Upper>(), colored);
    }

    /** Resizes a space with one resizable block to the reference, checks the sizes, normalizes
     * the reference, clears the root's lower triangle, and zeroes the density mean.
     */
    void
    fitSpace()
    {
        internal::fitValueDimension(m_space, m_reference.size());
        EMRYS_FILTER_ASSERT(
            m_root.rows() == differenceDimension() && m_root.cols() == differenceDimension()
            && m_whitenedOffset.size() == differenceDimension()
            && "the root and whitened offset must have a row per difference coordinate of the space"
        );
        normalizeValues(m_space, m_reference);
        m_root.template triangularView<Eigen::StrictlyLower>().setZero();
        m_densityMean.setZero(valueDimension());
    }

    VectorType m_reference;                         ///< the reference value \f$ x_0 \f$
    MatrixType m_root;                              ///< the upper-triangular root \f$ R \f$
    DifferenceVectorType m_whitenedOffset;          ///< the whitened offset \f$ r = R \delta \f$
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space, kept in step with the reference's size
    mutable Scalar m_logDeterminant{};              ///< \f$ \log|\Sigma| \f$, set by toDensity()
    mutable VectorType m_densityMean;               ///< the mean toDensity() solved, or zero
    internal::Stamp m_stamp;                        ///< what views check [view-stamp]
};

/** \ingroup Distributions
 * \brief The %Gaussian in square-root information form, which evaluates its density: what
 * SqrtInfoGaussianParams::toDensity() returns (decisions.md [gaussian-forms], [params-and-density]). It reads its
 * parameters' cached factorization and must not outlive them [view-stamp].
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 */
template <Space Space_, RealScalar Scalar_ = double>
using SqrtInfoGaussian = FormGaussian<SqrtInfoGaussianParams<Space_, Scalar_>>;

/** \ingroup Distributions
 * \returns the normalized product of the densities \a first and \a second, about \a first's
 * reference: \a second is moved there and the two data arrays are stacked and triangularized,
 * \f[ Q^T \begin{bmatrix} R_1 & r_1 \\ R_2 & r_2 \end{bmatrix} = \begin{bmatrix} R & r \\ 0 & e \end{bmatrix}, \f]
 * which gives \f$ R^T R = R_1^T R_1 + R_2^T R_2 \f$ without forming either precision. This is
 * the measurement update of \ref Bierman1977 "Bierman (1977)" (\ref Ke2024 "Ke et al. (2024)",
 * §II), with the second factor as the whitened measurement; \f$ e \f$, the residual, is discarded.
 * The stacked array lives in a buffer sized by the difference dimension, so a bounded space
 * allocates nothing.
 */
template <Space Space_, RealScalar Scalar_>
[[nodiscard]] SqrtInfoGaussianParams<Space_, Scalar_>
multiply(SqrtInfoGaussianParams<Space_, Scalar_> const & first, SqrtInfoGaussianParams<Space_, Scalar_> const & second)
{
    using Result = SqrtInfoGaussianParams<Space_, Scalar_>;
    constexpr int D = Result::DifferenceDimensionAtCompileTime;
    constexpr int DMax = Result::MaxDifferenceDimensionAtCompileTime;
    using Stacked = internal::MatrixOf<
        Scalar_,
        D == Eigen::Dynamic ? Eigen::Dynamic : 2 * D,
        D == Eigen::Dynamic ? Eigen::Dynamic : D + 1,
        DMax == Eigen::Dynamic ? Eigen::Dynamic : 2 * DMax,
        DMax == Eigen::Dynamic ? Eigen::Dynamic : DMax + 1>;
    EMRYS_FILTER_ASSERT(first.differenceDimension() == second.differenceDimension());
    Eigen::Index const n = first.differenceDimension();

    Result moved = second;
    moved.setReference(first.reference());
    Stacked stacked(2 * n, n + 1);
    stacked.topLeftCorner(n, n) = first.infoRoot();
    stacked.topRightCorner(n, 1) = first.whitenedOffset();
    stacked.bottomLeftCorner(n, n) = moved.infoRoot();
    stacked.bottomRightCorner(n, 1) = moved.whitenedOffset();
    triangularize(stacked);

    // Built from parts rather than as a copy of first, whose root and offset would be overwritten.
    auto root = typename Result::MatrixType{stacked.topLeftCorner(n, n)};
    root.template triangularView<Eigen::StrictlyLower>().setZero();
    return Result{
        first.space(),
        first.reference(),
        std::move(root),
        typename Result::DifferenceVectorType{stacked.topRightCorner(n, 1)}
    };
}

/** \ingroup Distributions
 * \returns \a form in the square-root information form, or the GaussianError of the accessors
 * it reads.
 *
 * A covariance form gives the form about its mean with \f$ r = 0 \f$ and \f$ R \f$ from
 * triangularizing its precisionRoot(), which is lower triangular there. InfoGaussianParams
 * keeps its reference and gives its upper-triangular precisionRoot() and
 * \f$ r = R^{-T} \eta \f$, so it fails while its precision is singular, though the target form
 * could hold that.
 *
 * \sa toGaussian(), toSqrt(), toInfo()
 */
template <GaussianForm G>
[[nodiscard]] std::expected<SqrtInfoGaussianParams<typename G::SpaceType, typename G::Scalar>, GaussianError>
toSqrtInfo(G const & form)
{
    using Result = SqrtInfoGaussianParams<typename G::SpaceType, typename G::Scalar>;
    if constexpr (std::is_same_v<G, Result>)
    {
        return form;
    }
    else
    {
        auto root = form.precisionRoot();
        if (!root)
        {
            return std::unexpected{root.error()};
        }
        if constexpr (isInfoForm<G>)
        {
            typename Result::DifferenceVectorType whitened =
                root->transpose().template triangularView<Eigen::Lower>().solve(form.infoVector());
            return Result{form.space(), form.reference(), std::move(*root), std::move(whitened)};
        }
        else
        {
            std::expected<typename G::VectorType, GaussianError> mean = form.mean();
            if (!mean)
            {
                return std::unexpected{mean.error()};
            }
            triangularize(*root);
            return Result{
                form.space(),
                std::move(*mean),
                std::move(*root),
                Result::DifferenceVectorType::Zero(form.differenceDimension())
            };
        }
    }
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a gaussian: its dimension, and its mean and covariance where its root is
 * nonsingular.
 */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, SqrtInfoGaussianParams<Space_, Scalar_> const & gaussian, Indent indent)
{
    stream << indent << "square-root information Gaussian";
    printDimension(stream, gaussian.valueDimension(), gaussian.differenceDimension());
    printMoments(stream, indent.deeper(), gaussian.moments());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
