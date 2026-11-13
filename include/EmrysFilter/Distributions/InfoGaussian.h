#pragma once

/** \file
 * The information form of the %Gaussian (decisions.md [gaussian-forms]): its parameters InfoGaussianParams, which
 * holds the precision and the information vector about a reference value; its factor
 * operations, addInfo(), multiply(), marginal<Path>(), and conditional<Path>(); and
 * toInfo(), which converts any form to it. Its density is an InfoGaussian, a FormGaussian. Read
 * SqrtInfoGaussian.h next.
 */

#include <expected>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

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
 * \brief A %Gaussian kept as its precision and information vector, about a reference value.
 *
 * The [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
 * in [canonical (information)
 * form](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Canonical_form): the
 * precision \f$ \Lambda = \Sigma^{-1} \f$ and the information vector \f$ \eta = \Lambda \delta \f$,
 * where the mean is \f$ \mu = x_0 \boxplus \delta \f$ for the stored reference value \f$ x_0 \f$.
 * On a linear space the reference is only an origin; on a space with \f$ \boxplus \f$ the
 * information lives in the differences at it, as a covariance lives in the differences at a mean,
 * and moving the reference carries it by the reset Jacobian of decisions.md [covariance-reset] (setReference()).
 *
 * The form's free operation is the one the covariance form pays for: combining independent
 * information about the same state, from a measurement or another factor, adds precisions and
 * information vectors (addInfo(), multiply()). Its cost is the mean, a solve with
 * \f$ \Lambda \f$. And it states what the covariance form cannot: a prior with no information at
 * all, \f$ \Lambda = 0 \f$, which is well defined here until something asks for the mean, the
 * covariance, or the density, each of which returns GaussianError::Singular while
 * \f$ \Lambda \f$ is not positive definite [gaussian-forms].
 *
 * Only the lower triangle of the precision is read, as only that of a %Gaussian's covariance is.
 * The Cholesky factorization of the precision is cached, as Gaussian caches its covariance's, and
 * survives changes to the information vector. The matrices the accessors give, the precision, the
 * covariance, and their factors, are in the differences at the reference; moments() and the
 * conversions to a covariance form carry the covariance to the differences at the mean [covariance-reset].
 * toDensity() returns an InfoGaussian, which checks this object for changes as Gaussian
 * checks a GaussianParams.
 * marginal<Path>() and conditional<Path>() return parameters, since their matrices are computed
 * rather than selected. The density's marginal<Path>() cannot fail, as this one's can
 * (decisions.md [form-densities]).
 *
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 *
 * \sa toInfo(), multiply(), class GaussianParams, GaussianForm
 */
template <Space Space_, RealScalar Scalar_ = double>
class InfoGaussianParams
{
public:

    using MomentsType = Moments<Space_, Scalar_>;         ///< the mean and covariance together
    using Scalar = typename MomentsType::Scalar;          ///< the scalar type
    using Index = typename MomentsType::Index;            ///< Eigen's index type
    using VectorType = typename MomentsType::VectorType;  ///< a value, such as the reference

    /** A difference, or the information vector. */
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;

    using MatrixType = typename MomentsType::MatrixType;   ///< a matrix over differences
    using SpaceType = Space_;                              ///< the space the values live in
    using DensityType = FormGaussian<InfoGaussianParams>;  ///< the view toDensity() returns

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the precision, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;     ///< parameters, which views check
    static constexpr bool IsInfoForm = true;   ///< holds the precision
    static constexpr bool IsSqrtForm = false;  ///< holds the precision whole, not a factor

    /** Constructs an InfoGaussianParams in a default space, with a zero reference and no
     * information.
     */
    InfoGaussianParams()
        : InfoGaussianParams{Space_{}}
    {}

    /** Constructs an InfoGaussianParams in \a space, with a zero reference and no information. */
    explicit InfoGaussianParams(Space_ space)
        : m_space{std::move(space)}
    {
        m_reference.setZero(Emrys::valueDimension(m_space));
        m_infoMatrix.setZero(differenceDimension(), differenceDimension());
        m_infoVector.setZero(differenceDimension());
    }

    /** Constructs an InfoGaussianParams about \a reference, with the lower triangle of
     * \a infoMatrix as the precision and \a infoVector as \f$ \eta \f$, moving any
     * passed as an rvalue of the stored type, in a default space resized to fit them.
     */
    template <MatrixExpression Reference, MatrixExpression Matrix, MatrixExpression Vector>
        requires internal::SizedByLength<Space_>
    InfoGaussianParams(Reference && reference, Matrix && infoMatrix, Vector && infoVector)
        : m_reference{std::forward<Reference>(reference)}
        , m_infoMatrix{std::forward<Matrix>(infoMatrix)}
        , m_infoVector{std::forward<Vector>(infoVector)}
    {
        fitSpace();
    }

    /** InfoGaussianParams(reference, infoMatrix, infoVector) in \a space, which is
     * resized to fit them if it has one resizable block.
     */
    template <MatrixExpression Reference, MatrixExpression Matrix, MatrixExpression Vector>
    InfoGaussianParams(Space_ space, Reference && reference, Matrix && infoMatrix, Vector && infoVector)
        : m_reference{std::forward<Reference>(reference)}
        , m_infoMatrix{std::forward<Matrix>(infoMatrix)}
        , m_infoVector{std::forward<Vector>(infoVector)}
        , m_space{std::move(space)}
    {
        fitSpace();
    }

    /** Copies the data and any valid factorization. Views of \a other do not see the copy. */
    InfoGaussianParams(InfoGaussianParams const & other)
        : m_reference{other.m_reference}
        , m_infoMatrix{other.m_infoMatrix}
        , m_infoVector{other.m_infoVector}
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
    InfoGaussianParams(InfoGaussianParams && other) noexcept
        : m_reference{std::move(other.m_reference)}
        , m_infoMatrix{std::move(other.m_infoMatrix)}
        , m_infoVector{std::move(other.m_infoVector)}
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
    InfoGaussianParams &
    operator=(InfoGaussianParams const & other)
    {
        m_stamp.touchShape();
        m_reference = other.m_reference;
        m_infoMatrix = other.m_infoMatrix;
        m_infoVector = other.m_infoVector;
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
    InfoGaussianParams &
    operator=(InfoGaussianParams && other) noexcept
    {
        m_stamp.touchShape();
        other.m_stamp.touchShape();
        m_reference = std::move(other.m_reference);
        m_infoMatrix = std::move(other.m_infoMatrix);
        m_infoVector = std::move(other.m_infoVector);
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

    /** \returns the number of coordinates of a value (see \ref Spaces) */
    [[nodiscard]] Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the number of rows and columns of the precision (see \ref Spaces) */
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

    /** \returns the stored precision \f$ \Lambda \f$; only its lower triangle is guaranteed to be read */
    [[nodiscard]] MatrixType const &
    infoMatrix() const
    {
        return m_infoMatrix;
    }

    /** \returns the information vector \f$ \eta = \Lambda \delta \f$ about the reference */
    [[nodiscard]] DifferenceVectorType const &
    infoVector() const
    {
        return m_infoVector;
    }

    /** Moves the reference to \a reference, normalized by the space, and re-expresses the
     * information about it. With \f$ d = x_0' \boxminus x_0 \f$ the step and \f$ G \f$ the reset
     * Jacobian of the space at \f$ x_0 \f$ and \f$ d \f$ (decisions.md [covariance-reset], resetJacobian()),
     * \f[ \Lambda' = G^{-T} \Lambda G^{-1}, \qquad \eta' = G^{-T} (\eta - \Lambda d), \f]
     * which is the quadratic form of the density written in the differences at the new reference,
     * where an error \f$ \epsilon \f$ at \f$ x_0 \f$ is \f$ G (\epsilon - d) \f$. On a space with
     * an identity reset \f$ G \f$ is the identity, the distribution is unchanged exactly, and the
     * factorization survives; elsewhere it is the first-order transport of [covariance-reset] and the
     * factorization is invalidated.
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
        // The product and the reset read the precision whole, so mirror the lower triangle first:
        // a dense product keeps fixed sizes fixed (internal::multiply()), where a self-adjoint view's never does.
        m_infoMatrix.template triangularView<Eigen::StrictlyUpper>() = m_infoMatrix.transpose();
        m_infoVector.noalias() -= internal::multiply(m_infoMatrix, shift);
        if constexpr (!hasIdentityReset<Space_>)
        {
            internal::resetInfoAt(m_space, m_reference, shift, moved, m_infoMatrix, m_infoVector);
            m_factorValid = false;
        }
        m_reference = std::move(moved);
    }

    /** Sets the precision to the lower triangle of \a infoMatrix and the information vector
     * to \a infoVector, about the current reference, moving either one passed as an rvalue
     * of the stored type, and invalidates the factorization.
     */
    template <MatrixExpression Matrix, MatrixExpression Vector>
    void
    setInfo(Matrix && infoMatrix, Vector && infoVector)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(
            infoMatrix.rows() == differenceDimension() && infoMatrix.cols() == differenceDimension()
            && infoVector.size() == differenceDimension()
        );
        m_infoMatrix = std::forward<Matrix>(infoMatrix);
        m_infoVector = std::forward<Vector>(infoVector);
        m_factorValid = false;
    }

    /** Adds \a infoMatrix (its lower triangle) to the precision and \a infoVector to
     * the information vector, both about the current reference: independent information, such as
     * \f$ H^T R^{-1} H \f$ and \f$ H^T R^{-1} (z - h(x_0)) \f$ from a measurement linearized at the
     * reference. Exact, and independent of the order in which information arrives. Invalidates
     * the factorization.
     */
    template <typename MatrixDerived, typename VectorDerived>
    void
    addInfo(Eigen::MatrixBase<MatrixDerived> const & infoMatrix, Eigen::MatrixBase<VectorDerived> const & infoVector)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(
            infoMatrix.rows() == differenceDimension() && infoMatrix.cols() == differenceDimension()
            && infoVector.size() == differenceDimension()
        );
        m_infoMatrix.template triangularView<Eigen::Lower>() += infoMatrix;
        m_infoVector += infoVector;
        m_factorValid = false;
    }

    /** \returns the mean \f$ x_0 \boxplus \Lambda^{-1} \eta \f$, solved with the cached
     * factorization, or the GaussianError of factorize().
     */
    [[nodiscard]] std::expected<VectorType, GaussianError>
    mean() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        return solvedMean();
    }

    /** \returns the mean and the covariance together, both solved with the cached factorization,
     * the covariance carried to the differences at the mean as toGaussian() carries it, or the
     * GaussianError of factorize()
     */
    [[nodiscard]] std::expected<MomentsType, GaussianError>
    moments() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        return momentsWithMean(solvedMean());
    }

    /** \returns the covariance \f$ \Lambda^{-1} \f$, in the differences at the reference, or the
     * GaussianError of factorize()
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(*m_llt, result);
        return result;
    }

    /** \returns the precision, with both triangles filled from the lower one. Never fails. */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precision() const
    {
        MatrixType result = m_infoMatrix.template selfadjointView<Eigen::Lower>();
        return result;
    }

    /** \returns \f$ R^{-1} \f$ for the upper-triangular Cholesky factor \f$ R \f$ of the precision:
     * upper triangular, with \f$ R^{-1} R^{-T} = \Sigma \f$. Or the GaussianError of factorize().
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covarianceRoot() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        MatrixType result = MatrixType::Identity(differenceDimension(), differenceDimension());
        internal::solveInPlace(m_llt->matrixU(), result);
        return result;
    }

    /** \returns the upper-triangular Cholesky factor \f$ R \f$ of the precision, with
     * \f$ R^T R = \Lambda \f$, or the GaussianError of factorize()
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    precisionRoot() const
    {
        if (auto factorized = factorize(); !factorized)
        {
            return std::unexpected{factorized.error()};
        }
        MatrixType result = m_llt->matrixU();
        return result;
    }

    /** Factorizes the precision if it has changed since the last factorization.
     *
     * It is \c const because it writes only the cached factorization, which the accessors that
     * need it also fill, so none of them is safe on one object from two threads at once until a
     * factorization has succeeded. toDensity() writes the solved mean each time. Threads should
     * share the density instead (see \ref Distributions).
     *
     * \returns nothing on success, GaussianError::Singular if the precision is not positive
     *          definite, as it is not while some direction has no information, or
     *          GaussianError::NonFinite if it is not finite.
     */
    [[nodiscard]] std::expected<void, GaussianError>
    factorize() const
    {
        if (m_factorValid)
        {
            return {};
        }
        if (!m_llt)
        {
            m_llt.emplace(differenceDimension());
        }
        m_llt->compute(m_infoMatrix);
        if (m_llt->info() != Eigen::Success)
        {
            return std::unexpected{GaussianError::Singular};
        }
        // As in GaussianParams::factorize(): NaN passes LLT's test but reaches the factor's
        // diagonal.
        Scalar const logDeterminant = Scalar(-2) * m_llt->matrixLLT().diagonal().array().log().sum();
        if (!isFinite(logDeterminant))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        m_logDeterminant = logDeterminant;
        m_factorValid = true;
        return {};
    }

    /** Factorizes the precision if needed (see factorize()) and solves for the mean, writing
     * both to the cache, so it is not safe on one object from two threads at once.
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
        m_densityMean = solvedMean();
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

    /** \returns the marginal of the subspace named by the path \a Path (a tag or a TagPath), in
     * that subspace, as an InfoGaussianParams about the matching entries of the reference, or
     * GaussianError::Singular if the precision of the rest is not positive definite, or
     * GaussianError::NonFinite if it is not finite.
     *
     * Marginalizing in the information form is the
     * [Schur complement](https://en.wikipedia.org/wiki/Schur_complement) of the rest \f$ r \f$
     * in the precision, \f$ \Lambda_{kk} - \Lambda_{kr} \Lambda_{rr}^{-1} \Lambda_{rk} \f$, with
     * \f$ \eta_k - \Lambda_{kr} \Lambda_{rr}^{-1} \eta_r \f$: a solve, where the covariance form
     * takes a block. Unlike GaussianParams::marginal(), it returns parameters, since the result is
     * computed rather than selected.
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
     * path \a GivenPath (a tag or a TagPath) equals \a values, which are in that subspace: an
     * InfoGaussianParams in the space of the remaining blocks, about the matching entries of the
     * reference.
     *
     * Conditioning in the information form takes a block, where the covariance form solves:
     * the precision is \f$ \Lambda_{tt} \f$ and the information vector
     * \f$ \eta_t - \Lambda_{tg} (v \boxminus x_{0,g}) \f$, from the
     * [canonical form](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Canonical_form)
     * of the joint with the given offset fixed. Never fails. There is no conditionalWithEvidence():
     * the evidence needs the given block's marginal covariance, which this form does not hold
     * (decisions.md [conditional-evidence]).
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditional<GivenPath>(
            m_space,
            [&](auto const & target, auto const & given, auto const & targetSpace, auto const & givenSpace) {
                using GivenSpace = std::remove_cvref_t<decltype(givenSpace)>;
                using Result = InfoGaussianParams<std::remove_cvref_t<decltype(targetSpace)>, Scalar>;
                constexpr int GivenSize = valueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>;
                constexpr int GivenMaxSize = maxValueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>;
                constexpr int TargetSize = Result::DifferenceDimensionAtCompileTime;
                constexpr int TargetMaxSize = Result::MaxDifferenceDimensionAtCompileTime;
                using GivenVector = Eigen::Matrix<Scalar, GivenSize, 1, Eigen::ColMajor, GivenMaxSize, 1>;
                using CrossMatrix = internal::MatrixOf<Scalar, TargetSize, GivenSize, TargetMaxSize, GivenMaxSize>;
                EMRYS_FILTER_ASSERT(values.size() == given.values.size());

                typename Result::VectorType reference = m_reference(target.values);
                typename Result::MatrixType matrix =
                    internal::lowerSymmetricBlock(m_infoMatrix, target.differences, target.differences);
                typename Result::DifferenceVectorType vector = m_infoVector(target.differences);
                // MSVC can't form the empty products of a conditional on everything or on nothing.
                if constexpr (TargetSize != 0 && GivenSize != 0)
                {
                    GivenVector const offset = Emrys::minus(givenSpace, values, m_reference(given.values));
                    CrossMatrix const cross =
                        internal::lowerSymmetricBlock(m_infoMatrix, target.differences, given.differences);
                    vector.noalias() -= internal::multiply(cross, offset);
                }
                return Result{targetSpace, std::move(reference), std::move(matrix), std::move(vector)};
            }
        );
    }

private:

    friend DensityType;  ///< the view reads the mean, the log-determinant, and the factorization

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** The data array \f$ [\,R \;\; R \delta\,] \f$, a column wider than the precision. */
    using AugmentedType = internal::MatrixOf<
        Scalar,
        DifferenceDimensionAtCompileTime,
        DifferenceDimensionAtCompileTime == Eigen::Dynamic ? Eigen::Dynamic : DifferenceDimensionAtCompileTime + 1,
        MaxDifferenceDimensionAtCompileTime,
        MaxDifferenceDimensionAtCompileTime == Eigen::Dynamic ? Eigen::Dynamic
                                                              : MaxDifferenceDimensionAtCompileTime + 1>;

    /** \returns the marginal over the resolved coordinates \a kept, which form \a keptSpace, by
     * the Schur complement of every other coordinate (see marginal<Path>())
     */
    template <typename Kept, typename KeptSpace>
    [[nodiscard]] std::expected<InfoGaussianParams<KeptSpace, Scalar>, GaussianError>
    marginalAt(Kept const & kept, KeptSpace const & keptSpace) const
    {
        using Result = InfoGaussianParams<KeptSpace, Scalar>;
        using Expected = std::expected<Result, GaussianError>;
        constexpr int D = DifferenceDimensionAtCompileTime;
        constexpr int DMax = MaxDifferenceDimensionAtCompileTime;
        constexpr int KeptSize = Result::DifferenceDimensionAtCompileTime;
        constexpr int KeptMaxSize = Result::MaxDifferenceDimensionAtCompileTime;
        Eigen::Index const n = differenceDimension();
        auto const rest =
            internal::resolveIndices<D, DMax>(n, internal::complementIndices<D, DMax>(n, kept.differences));
        using RestIndices = std::remove_cvref_t<decltype(rest)>;
        constexpr int RestSize = RestIndices::RowsAtCompileTime;
        constexpr int RestMaxSize = RestIndices::MaxRowsAtCompileTime;
        using RestMatrix = internal::MatrixOf<Scalar, RestSize, RestSize, RestMaxSize, RestMaxSize>;
        using RestVector = Eigen::Matrix<Scalar, RestSize, 1, Eigen::ColMajor, RestMaxSize, 1>;
        using CrossMatrix = internal::MatrixOf<Scalar, RestSize, KeptSize, RestMaxSize, KeptMaxSize>;

        typename Result::VectorType reference = m_reference(kept.values);
        typename Result::MatrixType matrix =
            internal::lowerSymmetricBlock(m_infoMatrix, kept.differences, kept.differences);
        typename Result::DifferenceVectorType vector = m_infoVector(kept.differences);
        // MSVC can't form the empty products of a marginal over the whole space or over nothing.
        if constexpr (RestSize != 0 && KeptSize != 0)
        {
            if (rest.size() > 0 && kept.differences.size() > 0)
            {
                auto const llt =
                    Eigen::LLT<RestMatrix>{RestMatrix{internal::lowerSymmetricBlock(m_infoMatrix, rest, rest)}};
                if (llt.info() != Eigen::Success)
                {
                    return Expected{std::unexpected{GaussianError::Singular}};
                }
                if (!allFinite(llt.matrixLLT().diagonal()))
                {
                    return Expected{std::unexpected{GaussianError::NonFinite}};
                }
                CrossMatrix const cross = internal::lowerSymmetricBlock(m_infoMatrix, rest, kept.differences);
                CrossMatrix solved = cross;
                internal::solveInPlace(llt, solved);
                RestVector const restVector = m_infoVector(rest);
                matrix.noalias() -= internal::multiply(cross.transpose(), solved);
                vector.noalias() -= internal::multiply(solved.transpose(), restVector);
            }
        }
        return Expected{Result{keptSpace, std::move(reference), std::move(matrix), std::move(vector)}};
    }

    /** \returns the marginal over the resolved coordinates \a kept, which form \a keptSpace, from
     * the valid factorization, for FormGaussian. With \f$ \Lambda = R^T R \f$ the data array
     * \f$ [\,R \;\; R \delta\,] \f$, where \f$ R \delta = R^{-T} \eta \f$, is triangularized with
     * the rest's columns first, as SqrtInfoGaussianParams::marginal<Path>() triangularizes its
     * own, and the trailing block \f$ [\,T \;\; t\,] \f$ gives the precision \f$ T^T T \f$ and the
     * information vector \f$ T^T t \f$. Nothing is factorized, so nothing fails.
     */
    template <typename Kept, typename KeptSpace>
    [[nodiscard]] InfoGaussianParams<KeptSpace, Scalar>
    densityMarginalAt(Kept const & kept, KeptSpace const & keptSpace) const
    {
        using Result = InfoGaussianParams<KeptSpace, Scalar>;
        constexpr int D = DifferenceDimensionAtCompileTime;
        constexpr int DMax = MaxDifferenceDimensionAtCompileTime;
        Eigen::Index const n = differenceDimension();
        Eigen::Index const k = kept.differences.size();
        auto const rest =
            internal::resolveIndices<D, DMax>(n, internal::complementIndices<D, DMax>(n, kept.differences));
        Eigen::Index const r = rest.size();

        MatrixType const factor = m_llt->matrixU();
        DifferenceVectorType whitened = m_infoVector;
        internal::solveInPlace(m_llt->matrixL(), whitened);
        AugmentedType stacked(n, n + 1);
        stacked.leftCols(r) = factor(Eigen::placeholders::all, rest);
        stacked.middleCols(r, k) = factor(Eigen::placeholders::all, kept.differences);
        stacked.col(n) = whitened;
        triangularize(stacked);

        typename Result::VectorType reference = m_reference(kept.values);
        auto matrix = typename Result::MatrixType{};
        matrix.resize(k, k);
        auto vector = typename Result::DifferenceVectorType{};
        vector.resize(k);
        // MSVC can't form the empty products of a marginal over nothing.
        if constexpr (Result::DifferenceDimensionAtCompileTime != 0)
        {
            typename Result::MatrixType const trailing = stacked.block(r, r, k, k);
            typename Result::DifferenceVectorType const tail = stacked.col(n).tail(k);
            matrix.noalias() = internal::multiply(trailing.transpose(), trailing);
            vector.noalias() = internal::multiply(trailing.transpose(), tail);
        }
        return Result{keptSpace, std::move(reference), std::move(matrix), std::move(vector)};
    }

    /** \returns the mean and the covariance together, with the mean \a mean solved with the valid
     * factorization and the covariance carried to the differences at it
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

    /** \returns the mean from the valid factorization */
    [[nodiscard]] VectorType
    solvedMean() const
    {
        DifferenceVectorType const offset = m_llt->solve(m_infoVector);
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
        residual =
            internal::multiply(internal::asTriangular<Eigen::Lower>(m_llt->matrixLLT()).transpose(), residual).eval();
    }

    /** Colors the standard normals in the columns of \a normals in place: \f$ R^{-1} z \f$, whose
     * covariance is \f$ R^{-1} R^{-T} = \Sigma \f$.
     */
    void
    colorInto(auto & colored, auto const & normals) const
    {
        colored = normals;
        internal::solveInPlace(m_llt->matrixU(), colored);
    }

    /** Resizes a space with one resizable block to the reference, checks the sizes, and normalizes
     * the reference.
     */
    void
    fitSpace()
    {
        internal::fitValueDimension(m_space, m_reference.size());
        EMRYS_FILTER_ASSERT(
            m_infoMatrix.rows() == differenceDimension() && m_infoMatrix.cols() == differenceDimension()
            && m_infoVector.size() == differenceDimension()
            && "the precision and information vector must have a row per difference coordinate of the space"
        );
        normalizeValues(m_space, m_reference);
    }

    VectorType m_reference;                         ///< the reference value \f$ x_0 \f$
    MatrixType m_infoMatrix;                        ///< the precision \f$ \Lambda \f$, lower triangle
    DifferenceVectorType m_infoVector;              ///< the information vector \f$ \eta \f$
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space, kept in step with the reference's size

    // Factorization cache, filled lazily by factorize().
    /// the Cholesky factorization \f$ \Lambda = L L^T \f$, \f$ R = L^T \f$, absent until one is made
    mutable std::optional<Eigen::LLT<MatrixType>> m_llt;

    mutable Scalar m_logDeterminant{};  ///< \f$ \log|\Sigma| = -\log|\Lambda| \f$ from the factorization
    mutable bool m_factorValid{false};  ///< whether the cache matches the precision
    mutable VectorType m_densityMean;   ///< the mean, solved by toDensity()
    internal::Stamp m_stamp;            ///< what views check [view-stamp]
};

/** \ingroup Distributions
 * \brief The %Gaussian in information form, which evaluates its density: what
 * InfoGaussianParams::toDensity() returns (decisions.md [gaussian-forms], [params-and-density]). It reads its
 * parameters' cached factorization and must not outlive them [view-stamp].
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 */
template <Space Space_, RealScalar Scalar_ = double>
using InfoGaussian = FormGaussian<InfoGaussianParams<Space_, Scalar_>>;

/** \ingroup Distributions
 * \returns the normalized product of the densities \a first and \a second, about \a first's
 * reference: \a second is moved there (InfoGaussianParams::setReference()) and their precisions
 * and information vectors are added. This is how independent information about one state
 * combines, from two sensors, two factors of a graph, or a prior and a likelihood.
 */
template <Space Space_, RealScalar Scalar_>
[[nodiscard]] InfoGaussianParams<Space_, Scalar_>
multiply(InfoGaussianParams<Space_, Scalar_> const & first, InfoGaussianParams<Space_, Scalar_> const & second)
{
    using Result = InfoGaussianParams<Space_, Scalar_>;
    EMRYS_FILTER_ASSERT(first.differenceDimension() == second.differenceDimension());
    Result moved = second;
    moved.setReference(first.reference());
    // Built from parts rather than as a copy of first, whose cached factorization would be copied
    // only to be invalidated. Only the lower triangles are reliable, as in every stored matrix.
    return Result{
        first.space(),
        first.reference(),
        typename Result::MatrixType{first.infoMatrix() + moved.infoMatrix()},
        typename Result::DifferenceVectorType{first.infoVector() + moved.infoVector()}
    };
}

/** \ingroup Distributions
 * \returns \a form in the information form, or the GaussianError of the accessors it reads.
 *
 * A covariance form gives an InfoGaussianParams about its mean, with \f$ \eta = 0 \f$ and its
 * precision(). An information form keeps its reference and gives its precision() and its
 * infoVector().
 *
 * \sa toGaussian(), toSqrt(), toSqrtInfo()
 */
template <GaussianForm G>
[[nodiscard]] std::expected<InfoGaussianParams<typename G::SpaceType, typename G::Scalar>, GaussianError>
toInfo(G const & form)
{
    using Result = InfoGaussianParams<typename G::SpaceType, typename G::Scalar>;
    auto precision = form.precision();
    if (!precision)
    {
        return std::unexpected{precision.error()};
    }
    if constexpr (isInfoForm<G>)
    {
        return Result{form.space(), form.reference(), std::move(*precision), form.infoVector()};
    }
    else
    {
        std::expected<typename G::VectorType, GaussianError> mean = form.mean();
        if (!mean)
        {
            return std::unexpected{mean.error()};
        }
        return Result{
            form.space(),
            std::move(*mean),
            std::move(*precision),
            Result::DifferenceVectorType::Zero(form.differenceDimension())
        };
    }
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a gaussian: its dimension, and its mean and covariance where its precision is
 * positive definite.
 */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, InfoGaussianParams<Space_, Scalar_> const & gaussian, Indent indent)
{
    stream << indent << "information Gaussian";
    printDimension(stream, gaussian.valueDimension(), gaussian.differenceDimension());
    printMoments(stream, indent.deeper(), gaussian.moments());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
