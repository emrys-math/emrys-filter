#pragma once

/** \file
 * What the four forms of the %Gaussian share (decisions.md [gaussian-forms], [form-densities]): the GaussianForm
 * concept and the isInfoForm<> and isSqrtForm<> traits; FormGaussian, the read-only density view that the square-root
 * and information forms' parameters return from toDensity(), with its moments(), marginals, drawSamples(), and
 * drawSamplesWithLogDensity(); and toGaussian(), which converts any form to the covariance form. Gaussian.h holds the
 * covariance form, and SqrtGaussian.h, InfoGaussian.h, and SqrtInfoGaussian.h the other three, each with the conversion
 * to it. Read SqrtGaussian.h next.
 */

#include <concepts>
#include <expected>
#include <random>
#include <utility>

#include <EmrysFilter/Core/ChiSquared.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Distributions
 * A %Gaussian in any of its four forms (decisions.md [gaussian-forms]): GaussianParams, SqrtGaussianParams,
 * InfoGaussianParams, or SqrtInfoGaussianParams, or a GaussianMarginal, the covariance form's
 * marginal, so that a marginal of any form's density is a GaussianForm [form-densities].
 *
 * Every form has the same space and dimension contract [sizes-and-dimensions] and the same four fallible
 * accessors, each computed by the cheapest route its form has: covariance(), precision(),
 * covarianceRoot() (a triangular \f$ S \f$ with \f$ S S^T = \Sigma \f$), and precisionRoot()
 * (a triangular \f$ R \f$ with \f$ R^T R = \Lambda \f$). The roots are lower triangular in the
 * covariance forms and upper triangular in the information forms, the triangle each gets without
 * a second factorization. mean() returns the mean, a reference or, in a marginal, an expression,
 * or a \c std::expected of it in the information forms, where it is a solve that fails while the
 * precision is singular.
 */
template <typename G>
concept GaussianForm = requires(G const & g) {
    typename G::SpaceType;
    typename G::Scalar;
    typename G::VectorType;
    typename G::MatrixType;
    { G::IsInfoForm } -> std::convertible_to<bool>;
    { G::IsSqrtForm } -> std::convertible_to<bool>;
    { g.space() } -> std::same_as<typename G::SpaceType const &>;
    { g.valueDimension() } -> std::same_as<Eigen::Index>;
    { g.differenceDimension() } -> std::same_as<Eigen::Index>;
    { g.mean() } -> std::convertible_to<std::expected<typename G::VectorType, GaussianError>>;
    { g.covariance() } -> std::same_as<std::expected<typename G::MatrixType, GaussianError>>;
    { g.precision() } -> std::same_as<std::expected<typename G::MatrixType, GaussianError>>;
    { g.covarianceRoot() } -> std::same_as<std::expected<typename G::MatrixType, GaussianError>>;
    { g.precisionRoot() } -> std::same_as<std::expected<typename G::MatrixType, GaussianError>>;
};

/** \ingroup Distributions
 * Whether the GaussianForm \a G holds the precision, as InfoGaussianParams and
 * SqrtInfoGaussianParams do, rather than the covariance. Its roots are then upper
 * triangular, and its mean is a solve that fails while the precision is singular.
 */
template <GaussianForm G>
inline constexpr bool isInfoForm = G::IsInfoForm;

/** \ingroup Distributions
 * Whether the GaussianForm \a G holds a triangular root, as SqrtGaussianParams and
 * SqrtInfoGaussianParams do, rather than a whole matrix.
 */
template <GaussianForm G>
inline constexpr bool isSqrtForm = G::IsSqrtForm;

/** \ingroup Distributions
 * \returns \a form in the covariance form: its mean and covariance() in a %Gaussian in the same
 * space, or the GaussianError of either.
 *
 * An information form's covariance lives in the differences at its reference, a %Gaussian's at
 * its mean, so on a space with a reset Jacobian the covariance is carried from the one to the
 * other by resetCovariance() (decisions.md [covariance-reset]); where the reset is the identity, as on a space of
 * plain, periodic, and bounded blocks, nothing moves.
 *
 * \sa toSqrt(), toInfo(), toSqrtInfo()
 */
template <GaussianForm G>
[[nodiscard]] std::expected<GaussianParams<typename G::SpaceType, typename G::Scalar>, GaussianError>
toGaussian(G const & form)
{
    std::expected<typename G::VectorType, GaussianError> mean = form.mean();
    if (!mean)
    {
        return std::unexpected{mean.error()};
    }
    auto covariance = form.covariance();
    if (!covariance)
    {
        return std::unexpected{covariance.error()};
    }
    if constexpr (isInfoForm<G> && !hasIdentityReset<typename G::SpaceType>)
    {
        auto const step = Emrys::minus(form.space(), *mean, form.reference());
        resetCovariance(form.space(), form.reference(), step, *covariance);
    }
    return GaussianParams<typename G::SpaceType, typename G::Scalar>{
        form.space(),
        std::move(*mean),
        std::move(*covariance)
    };
}

/** \ingroup Distributions
 * \brief The density of a %Gaussian in a square-root or information form: a read-only view.
 *
 * Represents the same density as Gaussian, evaluated from the root its parameters hold or
 * have cached, never from a covariance:
 * \f[ p(x) = (2\pi)^{-n/2} |\Sigma|^{-1/2} \exp\left(-\tfrac{1}{2} w^T w\right), \f]
 * where the parameters whiten the residual \f$ r = x \boxminus \mu \f$ to \f$ w \f$ with
 * \f$ w^T w = r^T \Sigma^{-1} r \f$, and colors standard normals by a root of \f$ \Sigma \f$,
 * each with a triangular product or solve.
 *
 * Obtained from the parameters' toDensity(), which checks the root and, in the information forms,
 * solves for the mean. What can fail in the parameters, a mean, moments, or a marginal of a
 * precision that is not positive definite, cannot fail here, so the four forms' densities are
 * read alike (decisions.md [form-densities]). It checks the parameters for changes, as Gaussian checks a
 * GaussianParams: reading it after they have changed is a programmer error (decisions.md [view-stamp]). It must not
 * outlive its parameters. SqrtGaussian, InfoGaussian, and SqrtInfoGaussian name it [params-and-density].
 *
 * \tparam Params_  SqrtGaussianParams, InfoGaussianParams, or SqrtInfoGaussianParams
 *
 * \sa SqrtGaussianParams::toDensity(), InfoGaussianParams::toDensity(),
 *     SqrtInfoGaussianParams::toDensity()
 */
template <typename Params_>
class FormGaussian
{
public:

    using ParamsType = Params_;                                           ///< the parameters it reads
    using SpaceType = typename Params_::SpaceType;                        ///< the space the values live in
    using Scalar = typename Params_::Scalar;                              ///< the scalar type
    using Index = Eigen::Index;                                           ///< Eigen's index type
    using VectorType = typename Params_::VectorType;                      ///< a value, such as the mean
    using DifferenceVectorType = typename Params_::DifferenceVectorType;  ///< a difference, such as a residual
    using MatrixType = typename Params_::MatrixType;                      ///< a matrix over differences
    using MomentsType = typename Params_::MomentsType;                    ///< the mean and covariance together

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = Params_::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = Params_::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = Params_::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = Params_::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = false;  ///< a view: it checks its parameters

    /** \returns the number of coordinates of the mean */
    [[nodiscard]] Index
    valueDimension() const
    {
        return params().valueDimension();
    }

    /** \returns the number of rows and columns of the covariance */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return params().differenceDimension();
    }

    /** \returns the parameters' space */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return params().space();
    }

    /** \returns the mean, which toDensity() solved for in the information forms */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return params().densityMean();
    }

    /** \returns the mean and the covariance together, the covariance formed from the root and,
     * in the information forms, carried to the differences at the mean as toGaussian() carries it.
     * Never fails, since toDensity() has checked the root.
     */
    [[nodiscard]] MomentsType
    moments() const
    {
        return params().densityMoments();
    }

    /** \returns the marginal of the subspace named by the path \a Path (a tag or a TagPath), in
     * that subspace, as parameters of the same form: a value, since its matrix is computed
     * (decisions.md [element-marginals]). Never fails: the parameters' own marginal<Path>() fails only where the
     * precision of the rest is singular, which toDensity() has ruled out. The information form's
     * is made from the cached root, as the square-root information form's is (see
     * SqrtInfoGaussianParams::marginal<Path>()), so nothing is factorized again.
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::marginalOverPath<Path>(space(), [&](auto const & kept, auto const & keptSpace) {
            return params().densityMarginalAt(kept, keptSpace);
        });
    }

    /** marginal<Path>() of repetition \a element of the repeated block \a Path starts at (see
     * GaussianParams::marginal(element)).
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const
    {
        return internal::marginalOverElement<Path>(space(), element, [&](auto const & kept, auto const & keptSpace) {
            return params().densityMarginalAt(kept, keptSpace);
        });
    }

    /** \returns the log-determinant \f$ \log|\Sigma| \f$ of the covariance, from the root's diagonal */
    [[nodiscard]] Scalar
    logDeterminant() const
    {
        return params().m_logDeterminant;
    }

    /** \returns the square-root form's lower-triangular root \f$ S \f$, \f$ S S^T = \Sigma \f$,
     * which the closed-form divergences between two of them read (see kullbackLeibler())
     */
    [[nodiscard]] MatrixType const &
    root() const
        requires(Params_::IsSqrtForm && !Params_::IsInfoForm)
    {
        return params().root();
    }

    /** \returns the log of the density at the mean, \f$ -\frac{1}{2}(d \log 2\pi + \log|\Sigma|) \f$,
     * with \f$ d \f$ the difference dimension
     */
    [[nodiscard]] Scalar
    logPeak() const
    {
        Scalar const log2Pi = Emrys::log2Pi<Scalar>();
        return Scalar(-0.5) * (internal::scalarFromCount<Scalar>(differenceDimension()) * log2Pi + logDeterminant());
    }

    /** \returns the squared Mahalanobis distance of \a x,
     * \f$ d_M^2(x) = (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu) \f$, the squared norm of the
     * whitened residual; see Gaussian::squaredMahalanobisDistance()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    squaredMahalanobisDistance(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "squaredMahalanobisDistance() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        DifferenceVectorType residual = Emrys::minus(space(), x, mean());
        params().whitenInPlace(residual);
        return residual.squaredNorm();
    }

    /** \returns the confidence of \a x, \f$ F_{\chi^2_k}(d_M^2(x)) \f$; see
     * Gaussian::confidence()
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

    /** \returns whether \a x lies in the \a probability confidence region; see
     * Gaussian::contains()
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

    /** \returns the log of the density at \a x, \f$ \mathrm{logPeak} - \frac{1}{2}d_M^2(x) \f$ */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        return logPeak() - Scalar(0.5) * squaredMahalanobisDistance(x);
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

    /** \returns \f$ A z \f$ for a root \f$ A \f$ of the covariance, \f$ A A^T = \Sigma \f$: standard
     * normal draws in the columns of \a z become draws of the residual, which drawSamples() adds
     * to the mean. \f$ A \f$ is the stored root in the square-root form and the inverse of the
     * precision's root in the information forms, applied by a triangular solve.
     */
    template <typename Derived>
    [[nodiscard]] typename Derived::PlainObject
    colorNormals(Eigen::MatrixBase<Derived> const & z) const
    {
        EMRYS_FILTER_ASSERT(z.rows() == differenceDimension());
        auto result = typename Derived::PlainObject{};
        result.resize(z.rows(), z.cols());
        params().colorInto(result, z);
        return result;
    }

private:

    friend Params_;  ///< the parameters construct views

    /** Views \a params, whose root toDensity() has checked, and records their stamp. */
    explicit FormGaussian(Params_ const & params)
        : m_check{params}
    {}

    /** \returns the viewed parameters */
    [[nodiscard]] Params_ const &
    params() const
    {
        return m_check.get();
    }

    internal::ValuesCheck<Params_> m_check;  ///< the check of the parameters
};

/** \ingroup Distributions
 * \returns \a n independent draws from \a gaussian, one per column: \f$ \mu \boxplus A z \f$ with
 * \f$ z \sim \mathcal{N}(0, I) \f$ and \f$ A \f$ the root of
 * FormGaussian::colorNormals(), as drawSamples() of a %Gaussian draws with its
 * Cholesky factor.
 */
template <typename Params_, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<typename Params_::SpaceType, typename Params_::Scalar>
drawSamples(FormGaussian<Params_> const & gaussian, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using Space = typename Params_::SpaceType;
    using Scalar = typename Params_::Scalar;
    using Normals = SampleSet<DifferenceSpace<Space>, Scalar>;
    Normals const z = internal::standardNormal<Normals>(gaussian.differenceDimension(), n, rng);
    return SampleSet<Space, Scalar>{Emrys::plus(gaussian.space(), gaussian.mean(), gaussian.colorNormals(z))};
}

/** \ingroup Distributions
 * \returns one draw from \a gaussian, \f$ \mu \boxplus A z \f$ as drawSamples() draws each, in a
 * vector sized by the space: the stream of draws that drawSample() of a %Gaussian is.
 */
template <typename Params_, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<typename Params_::SpaceType, typename Params_::Scalar>::VectorType
drawSample(FormGaussian<Params_> const & gaussian, G & rng)
{
    using MomentsType = Moments<typename Params_::SpaceType, typename Params_::Scalar>;
    using Normals = DifferenceVector<typename Params_::SpaceType, typename Params_::Scalar>;
    Normals const z = internal::standardNormalVector<Normals>(gaussian.differenceDimension(), rng);
    return typename MomentsType::VectorType{Emrys::plus(gaussian.space(), gaussian.mean(), gaussian.colorNormals(z))};
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a gaussian, one per column, as drawSamples() draws them,
 * with the log-density of \a gaussian at each.
 *
 * The log-densities cost almost nothing extra in a linear space: the whitened residual of
 * \f$ x = \mu + A z \f$ is \f$ z \f$, so the squared Mahalanobis distance is \f$ \|z\|^2 \f$. In
 * other spaces each is FormGaussian::logDensity() of the stored sample, as for a %Gaussian (see
 * drawSamplesWithLogDensity(Gaussian const &, Eigen::Index, G &)).
 */
template <typename Params_, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogDensity<typename Params_::SpaceType, typename Params_::Scalar>
drawSamplesWithLogDensity(FormGaussian<Params_> const & gaussian, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using Space = typename Params_::SpaceType;
    using Scalar = typename Params_::Scalar;
    using Normals = SampleSet<DifferenceSpace<Space>, Scalar>;
    Normals const z = internal::standardNormal<Normals>(gaussian.differenceDimension(), n, rng);
    auto result = SamplesWithLogDensity<Space, Scalar>{
        Emrys::plus(gaussian.space(), gaussian.mean(), gaussian.colorNormals(z)),
        {}
    };
    if constexpr (isLinear<Space>)
    {
        result.logDensity = gaussian.logPeak() - Scalar(0.5) * z.colwise().squaredNorm().transpose().array();
    }
    else
    {
        result.logDensity.resize(n);
        for (Eigen::Index j = 0; j < n; ++j)
        {
            result.logDensity(j) = gaussian.logDensity(result.samples.col(j));
        }
    }
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a gaussian, as drawSample() draws it, with the log-density of
 * \a gaussian at it, in a vector sized by the space: the stream of draws that
 * drawSampleWithLogDensity() of a %Gaussian is, with the log-density computed as
 * drawSamplesWithLogDensity() computes it.
 */
template <typename Params_, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<typename Params_::SpaceType, typename Params_::Scalar>
drawSampleWithLogDensity(FormGaussian<Params_> const & gaussian, G & rng)
{
    using Space = typename Params_::SpaceType;
    using Scalar = typename Params_::Scalar;
    using MomentsType = Moments<Space, Scalar>;
    using Normals = DifferenceVector<Space, Scalar>;
    Normals const z = internal::standardNormalVector<Normals>(gaussian.differenceDimension(), rng);
    auto result = SampleWithLogDensity<Space, Scalar>{
        typename MomentsType::VectorType{Emrys::plus(gaussian.space(), gaussian.mean(), gaussian.colorNormals(z))},
        0
    };
    if constexpr (isLinear<Space>)
    {
        result.logDensity = gaussian.logPeak() - Scalar(0.5) * z.squaredNorm();
    }
    else
    {
        result.logDensity = gaussian.logDensity(result.sample);
    }
    return result;
}

}  // namespace Emrys
