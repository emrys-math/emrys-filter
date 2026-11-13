#pragma once

/** \file
 * The Kalman conditioning of a marked input on an observation of a transform's output (decisions.md
 * [covariance-update-forms], [innovation-cholesky], [linearization-forms]): condition() with the
 * updates::Standard, updates::Symmetric, and updates::Joseph covariance forms and the CovarianceUpdate
 * concept, its result Conditioned, and statisticalLinearization() of a result in its marked input. It reads
 * only what Linearization.h defines, so an update needs none of the transform's machinery. Read Methods.h
 * next.
 */

#include <array>
#include <cstddef>
#include <expected>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Linearization.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Transforms
 * The result of condition(): the posterior, and the innovation, from which the log-likelihood of the
 * observation follows when asked for.
 *
 * \tparam Space        space of the input
 * \tparam OutputSpace  space of the output
 */
template <typename Space, typename OutputSpace, typename Scalar = double>
struct Conditioned
{
    GaussianParams<Space, Scalar> posterior;     ///< the input conditioned on the observation
    Innovation<OutputSpace, Scalar> innovation;  ///< the observation's residual and its covariance

    /** \returns the log-likelihood of the observation, \f$ \log \mathcal{N}(y; \bar y, \Sigma_{yy}) \f$ */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return innovation.logLikelihood();
    }
};

/** \ingroup Transforms
 * \brief The covariance update forms, each made by its factory (decisions.md [policy-factories],
 * [covariance-update-forms]): standard(), symmetric(), and joseph(). The types beside them name a
 * form where a type is needed.
 */
namespace updates
{

/** \ingroup Transforms
 * The standard covariance update of condition(): \f$ P - K \Sigma_{xy}^T \f$, which is
 * \f$ (I - K H) P \f$ written without \f$ H \f$. The cheapest form, and the default. Rounding
 * leaves it slightly asymmetric, and with a measurement much more precise than the prior it can
 * lose positive definiteness.
 *
 * The form of \ref Kalman1960 "Kalman (1960)", written as the conditioning of the joint %Gaussian
 * of input and output (\ref Sarkka2013 "Särkkä (2013)", §6.1).
 *
 * \sa Symmetric, Joseph
 */
struct Standard
{};

/** \ingroup Transforms
 * \returns the Standard covariance update, the default
 */
[[nodiscard]] constexpr Standard
standard()
{
    return {};
}

/** \ingroup Transforms
 * The symmetric covariance update: \f$ P - K \Sigma_{yy} K^T \f$, equal to the standard form
 * when \f$ K \f$ is the optimal gain, and made exactly symmetric. Positive definiteness can
 * still be lost as it can in the standard form.
 *
 * \sa Standard, Joseph
 */
struct Symmetric
{};

/** \ingroup Transforms
 * \returns the Symmetric covariance update
 */
[[nodiscard]] constexpr Symmetric
symmetric()
{
    return {};
}

/** \ingroup Transforms
 * The Joseph covariance update:
 * \f$ (I - K H) P (I - K H)^T + K R K^T \f$, with \f$ H \f$ and \f$ R \f$ from the statistical
 * linearization of the output in the input (see statisticalLinearization()). It is correct for
 * any gain, an error in the gain changes it only at second order, and it is a sum of two
 * positive semidefinite terms, so it stays positive semidefinite whenever \f$ R \f$ is. It costs
 * a solve against the prior covariance and products of the state's size cubed.
 *
 * \f$ R \f$ is everything in \f$ \Sigma_{yy} \f$ that \f$ H \f$ does not explain: input noise,
 * and the error of the linearization. A measurement's additive noise is kept apart and added
 * to \f$ R \f$ rather than being subtracted back out of \f$ \Sigma_{yy} \f$, which would cancel
 * the digits that make a precise measurement hard. With a negative sigma-point weight the
 * linearization error can come out indefinite, and then so can the result.
 *
 * The Joseph stabilized form (\ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)"),
 * with the measurement matrix and noise replaced by the statistical linearization's, so that it
 * applies to every transform method.
 *
 * \sa Standard, Symmetric
 */
struct Joseph
{};

/** \ingroup Transforms
 * \returns the Joseph covariance update
 */
[[nodiscard]] constexpr Joseph
joseph()
{
    return {};
}

}  // namespace updates

/** \ingroup Transforms
 * One of the covariance updates: updates::Standard, updates::Symmetric, or updates::Joseph.
 */
template <typename T>
concept CovarianceUpdate =
    std::is_same_v<T, updates::Standard> || std::is_same_v<T, updates::Symmetric> || std::is_same_v<T, updates::Joseph>;

/** \ingroup Transforms
 * The statistical linearization of an output in an input, from statisticalLinearization(): the
 * affine function of the input that best predicts the output in mean square,
 * \f$ y \approx \bar y \boxplus H (x \boxminus \mu) \f$, and the covariance \f$ \Omega \f$ of
 * what it leaves unexplained.
 *
 * \tparam InputSpace   space of the input
 * \tparam OutputSpace  space of the output
 */
template <typename InputSpace, typename OutputSpace, typename Scalar = double>
struct StatisticalLinearization
{
    /** The Jacobian's type: output differences by input differences. */
    using JacobianMatrix = internal::MatrixOf<
        Scalar,
        valueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        valueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<InputSpace>>>;

    /** The error covariance's type: over the output's differences. */
    using ErrorMatrix = internal::MatrixOf<
        Scalar,
        valueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        valueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>>;

    /** \f$ H = \Sigma_{xy}^T P^{-1} \f$: a row per difference coordinate of the output, a column
     * per difference coordinate of the input
     */
    JacobianMatrix jacobian;

    /** \f$ \Omega = \Sigma_{yy} - H P H^T \f$: the covariance of the output given the input under
     * the affine fit, from the other uncertain inputs and from the function's curvature
     */
    ErrorMatrix errorCov;
};

namespace internal
{

/** What updates::Joseph needs beyond the gain: \f$ H \f$, and the \f$ R \f$ it adds back. */
template <typename Jacobian, typename Noise>
struct JosephTerms
{
    Jacobian jacobian;  ///< \f$ H \f$, output differences by input differences
    Noise noise;        ///< \f$ R \f$, over the output's differences
};

/** What updates::Joseph needs beyond the gain from a prediction in the root form (decisions.md [linearization-forms]):
 * the root \f$ S \f$ of the prior's covariance that spread the points, \f$ \Psi = (H S)^T \f$, and the
 * \f$ R \f$ it adds back, so that \f$ (I - K H) S = S - K \Psi^T \f$ is formed with no solve.
 */
template <typename Root, typename Whitened, typename Noise>
struct JosephRootTerms
{
    Root root;          ///< \f$ S \f$, over the input's differences
    Whitened whitened;  ///< \f$ \Psi \f$, input differences by output differences
    Noise noise;        ///< \f$ R \f$, over the output's differences
};

/** \returns the posterior covariance of the updates::Standard, \f$ P - K \Sigma_{xy}^T \f$, from the
 * symmetric prior covariance \a cov, the gain \a gain, and the cross-covariance \a cross.
 */
template <typename Matrix, typename Gain, typename Cross, typename OutputCov, typename Linearize>
[[nodiscard]] Matrix
posteriorCov(
    updates::Standard,
    Matrix const & cov,
    Gain const & gain,
    Cross const & cross,
    OutputCov const & /*outputCov*/,
    Linearize const & /*linearize*/
)
{
    return Matrix{cov - multiply(gain, cross.transpose())};
}

/** \returns the posterior covariance of the updates::Symmetric, \f$ P - K \Sigma_{yy} K^T \f$, made
 * symmetric, with \a outputCov the innovation covariance \f$ \Sigma_{yy} \f$, of which only the
 * lower triangle is read.
 */
template <typename Matrix, typename Gain, typename Cross, typename OutputCov, typename Linearize>
[[nodiscard]] Matrix
posteriorCov(
    updates::Symmetric,
    Matrix const & cov,
    Gain const & gain,
    Cross const & /*cross*/,
    OutputCov const & outputCov,
    Linearize const & /*linearize*/
)
{
    using Scalar = typename Matrix::Scalar;
    OutputCov const symmetric = outputCov.template selfadjointView<Eigen::Lower>();
    Matrix result = cov - multiply(multiply(gain, symmetric), gain.transpose());
    return Matrix{Scalar(0.5) * (result + result.transpose())};
}

/** \returns the posterior covariance of the updates::Joseph,
 * \f$ (I - K H) P (I - K H)^T + K R K^T \f$, made symmetric, with \f$ H \f$ and \f$ R \f$ from
 * \a linearize, a function returning JosephTerms that only this form calls. Given JosephRootTerms,
 * \f$ (S - K \Psi^T)(S - K \Psi^T)^T + K R K^T \f$, the same for \f$ S S^T = P \f$ and
 * \f$ \Psi^T = H S \f$, with one product of the input's size cubed where the other takes two, and
 * still a sum of two positive semidefinite terms (decisions.md [linearization-forms]).
 */
template <typename Matrix, typename Gain, typename Cross, typename OutputCov, typename Linearize>
[[nodiscard]] Matrix
posteriorCov(
    updates::Joseph,
    Matrix const & cov,
    Gain const & gain,
    Cross const & /*cross*/,
    OutputCov const & /*outputCov*/,
    Linearize const & linearize
)
{
    using Scalar = typename Matrix::Scalar;
    auto const terms = linearize();
    Matrix result = multiply(multiply(gain, terms.noise), gain.transpose());
    if constexpr (requires { terms.root; })
    {
        Matrix const factor = terms.root - multiply(gain, terms.whitened.transpose());
        result.noalias() += multiply(factor, factor.transpose());
    }
    else
    {
        Matrix const factor = Matrix::Identity(cov.rows(), cov.cols()) - multiply(gain, terms.jacobian);
        result.noalias() += multiply(multiply(factor, cov), factor.transpose());
    }
    return Matrix{Scalar(0.5) * (result + result.transpose())};
}

/** The rows of the state's differences that a Kalman update considers, one BlockRows per block
 * that a consider() names, resolved against the state's space (decisions.md [consider-states]).
 * NoConsidered, of no blocks, is the full update.
 */
template <std::size_t K>
using ConsideredRows = std::array<BlockRows, K>;

/** The considered rows of a full update: none. */
using NoConsidered = ConsideredRows<0>;

/** Zeroes the rows \a considered of \a gain, so that the step leaves the considered blocks' mean
 * where it was: the gain \f$ [K_x; 0] \f$ of the Schmidt-Kalman update.
 */
template <typename Gain, std::size_t K>
void
zeroConsideredRows(Gain & gain, ConsideredRows<K> const & considered)
{
    for (BlockRows const rows : considered)
    {
        gain.middleRows(rows.start, rows.size).setZero();
    }
}

/** Writes \a prior's covariance of the blocks \a considered with each other into \a cov, both
 * whole matrices, so that the posterior keeps \f$ P_{cc} \f$ as it was.
 */
template <typename Matrix, std::size_t K>
void
keepConsideredCovariance(Matrix & cov, Matrix const & prior, ConsideredRows<K> const & considered)
{
    for (BlockRows const a : considered)
    {
        for (BlockRows const b : considered)
        {
            cov.block(a.start, b.start, a.size, b.size) = prior.block(a.start, b.start, a.size, b.size);
        }
    }
}

/** A Kalman update's posterior, and the statistics of its innovation when they were asked for. */
template <typename Space, typename Scalar>
struct KalmanConditioned
{
    GaussianParams<Space, Scalar> posterior;                 ///< the conditioned input
    std::optional<InnovationStatistics<Scalar>> statistics;  ///< the NIS and log-determinant, if computed
};

/** What kalmanConditionInPlace() did: the innovation's statistics, when computed, and whether the
 * acceptor let it write the state.
 */
template <typename Scalar>
struct ConditionedInPlace
{
    std::optional<InnovationStatistics<Scalar>> statistics;  ///< the NIS and log-determinant, if computed
    bool accepted = true;                                    ///< false when the state was left as it was
};

/** Conditions \a state in place on an observation whose residual from the predicted output is
 * \a residual, normalized in the output space, with output covariance \a outputCov, of which only
 * the lower triangle is read, and cross-covariance \a cross with \a state; its covariance by
 * \a form, with \a linearize giving updates::Joseph its JosephTerms (decisions.md [borrowed-intermediates]). The
 * updates::Standard mirrors the covariance's lower triangle and subtracts
 * \f$ K \Sigma_{xy}^T \f$ where it is; the other forms read the symmetric prior while forming
 * theirs. The covariance, formed in the tangent space at the prior mean, is then carried to the
 * one at the posterior mean by resetCovariance() [covariance-reset], which does nothing on a space with an
 * identity reset. \a residual, \a outputCov, and \a cross must not refer to \a state.
 *
 * An \a outputCov that is not positive definite is repaired as \a Repair says and replaced by the
 * repaired matrix (see innovationCholesky()), \a Repair being a Repair or NoRepair, which then gives the gain, the
 * statistics, and the standard and symmetric forms' covariance. updates::Joseph's \f$ R \f$ stays the measurement's
 * own, since that form is right for any gain.
 *
 * The blocks at the rows \a considered keep their mean and their covariance with each other, and
 * the rest of the state and its covariance with them are conditioned as the full update conditions
 * them: the Schmidt-Kalman update (decisions.md [consider-states]). The gain's rows there are zeroed
 * before the step. updates::Standard and updates::Symmetric, right only for the optimal gain, form
 * the full update's covariance, whose blocks outside \f$ P_{cc} \f$ are the consider update's; and
 * updates::Joseph, right for any gain, takes the zeroed one, so the result stays a sum of positive
 * semidefinite terms. Every form then writes the prior's \f$ P_{cc} \f$ back, which Joseph's root
 * form would otherwise reproduce only to rounding. The reset sees no step in the considered blocks.
 *
 * \a accept, an acceptor (see AcceptEvery), is asked once \a outputCov is factored, before \a state
 * is written, and an innovation it refuses leaves \a state as it was (decisions.md [update-outcomes]).
 *
 * \returns the innovation's NIS and log-determinant, with \a Statistics or an acceptor that reads
 *          them, from the Cholesky factorization of \a outputCov that gives the gain, and whether
 *          the state was written; or the error of innovationCholesky(), with \a state unchanged, if
 *          \a outputCov does not factorize (decisions.md [innovation-cholesky])
 */
template <
    bool Statistics,
    typename Repair,
    typename Space,
    typename Scalar,
    typename Residual,
    typename OutputMatrix,
    typename Cross,
    typename Form,
    typename Linearize,
    std::size_t K,
    typename Accept = AcceptEvery>
[[nodiscard]] std::expected<ConditionedInPlace<Scalar>, GaussianError>
kalmanConditionInPlace(
    GaussianParams<Space, Scalar> & state,
    Eigen::MatrixBase<Residual> const & residual,
    OutputMatrix & outputCov,
    Cross const & cross,
    Form const & form,
    Linearize const & linearize,
    [[maybe_unused]] ConsideredRows<K> const & considered,
    Accept const & accept = {}
)
{
    using Matrix = typename GaussianParams<Space, Scalar>::MatrixType;
    [[maybe_unused]] Eigen::Index const nd = state.differenceDimension();
    [[maybe_unused]] Eigen::Index const md = outputCov.rows();
    EMRYS_FILTER_ASSERT(cross.rows() == nd && cross.cols() == md && "the prediction was not made from this prior");
    EMRYS_FILTER_ASSERT(residual.size() == md);

    auto const factored = innovationCholesky<Repair>(outputCov);
    if (!factored)
    {
        return std::unexpected{factored.error()};
    }
    auto const & llt = *factored;
    auto result = ConditionedInPlace<Scalar>{};
    if constexpr (Statistics || acceptsByNis<Accept>)
    {
        result.statistics = innovationStatistics(llt, residual);
    }
    if constexpr (acceptsByNis<Accept>)
    {
        if (!accept(result.statistics->nis, md))
        {
            result.accepted = false;
            return result;
        }
    }

    // With nothing to condition, only the likelihood remains; MSVC can't form an empty gain.
    if constexpr (valueDimensionAtCompileTime<DifferenceSpace<Space>> != 0)
    {
        auto gain = transposedSolve<typename Cross::PlainObject>(llt, cross.transpose());
        auto & moments = GaussianAccess::moments(state);
        Matrix & cov = moments.covarianceLower();
        if constexpr (K > 0)
        {
            auto const symmetric = symmetricFromLower<Matrix>(cov);
            if constexpr (std::is_same_v<Form, updates::Joseph>)
            {
                zeroConsideredRows(gain, considered);
                cov = posteriorCov(form, symmetric, gain, cross, outputCov, linearize);
            }
            else
            {
                cov = posteriorCov(form, symmetric, gain, cross, outputCov, linearize);
                zeroConsideredRows(gain, considered);
            }
            keepConsideredCovariance(cov, symmetric, considered);
        }
        else if constexpr (std::is_same_v<Form, updates::Standard>)
        {
            cov.template triangularView<Eigen::StrictlyUpper>() = cov.transpose();
            cov.noalias() -= multiply(gain, cross.transpose());
        }
        else
        {
            auto const symmetric = symmetricFromLower<Matrix>(cov);
            cov = posteriorCov(form, symmetric, gain, cross, outputCov, linearize);
        }
        if constexpr (hasIdentityReset<Space>)
        {
            moments.mean() = Emrys::plus(state.space(), moments.mean(), multiply(gain, residual));
        }
        else
        {
            using Vector = typename GaussianParams<Space, Scalar>::VectorType;
            using Difference = typename GaussianParams<Space, Scalar>::DifferenceVectorType;
            Vector const start = moments.mean();
            auto step = Difference{};
            step.resize(nd);
            step.noalias() = multiply(gain, residual);
            moments.mean() = Emrys::plus(state.space(), start, step);
            resetCovarianceAt(state.space(), start, step, moments.mean(), cov);
        }
    }
    return result;
}

/** The condition() of \a prior by an observation whose residual from the predicted output is
 * \a residual, normalized in the output space, with output covariance \a outputCov, of which only
 * the lower triangle is read, and cross-covariance \a cross; its covariance by \a form, with
 * \a linearize giving updates::Joseph its JosephTerms. With \a Statistics, the innovation's NIS and
 * log-determinant too, from the Cholesky factorization of \a outputCov that gives the gain, which
 * replaces \a outputCov when \a Repair repairs it, keeping the blocks at the rows \a considered as
 * they were. The posterior starts as a copy of \a prior and is conditioned in place (see
 * kalmanConditionInPlace(), whose error it returns).
 */
template <
    bool Statistics,
    typename Repair,
    typename Prior,
    typename Residual,
    typename OutputMatrix,
    typename Cross,
    typename Form,
    typename Linearize,
    std::size_t K>
[[nodiscard]] std::expected<KalmanConditioned<typename Prior::SpaceType, typename Prior::Scalar>, GaussianError>
kalmanCondition(
    Prior const & prior,
    Eigen::MatrixBase<Residual> const & residual,
    OutputMatrix & outputCov,
    Cross const & cross,
    Form const & form,
    Linearize const & linearize,
    ConsideredRows<K> const & considered
)
{
    using Posterior = GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>;
    auto result = KalmanConditioned<typename Prior::SpaceType, typename Prior::Scalar>{
        Posterior{
            prior.space(),
            typename Posterior::VectorType{prior.mean()},
            typename Posterior::MatrixType{covarianceLowerOf(prior)}
        },
        std::nullopt
    };
    auto conditioned = kalmanConditionInPlace<Statistics, Repair>(
        result.posterior,
        residual,
        outputCov,
        cross,
        form,
        linearize,
        considered
    );
    if (!conditioned)
    {
        return std::unexpected{conditioned.error()};
    }
    result.statistics = conditioned->statistics;
    return result;
}

/** The condition() of \a prior by an observation \a observed of \a predicted's output, its
 * covariance by \a form, with \a linearize giving updates::Joseph its JosephTerms.
 */
template <
    typename Prior,
    typename OutputSpace,
    typename LinearizationForm,
    typename Derived,
    typename Form,
    typename Linearize>
[[nodiscard]] std::expected<Conditioned<typename Prior::SpaceType, OutputSpace, typename Prior::Scalar>, GaussianError>
kalmanUpdate(
    Prior const & prior,
    OutputWithCross<typename Prior::SpaceType, OutputSpace, LinearizationForm, typename Prior::Scalar> const &
        predicted,
    Eigen::MatrixBase<Derived> const & observed,
    Form const & form,
    Linearize const & linearize
)
{
    using Output = GaussianParams<OutputSpace, typename Prior::Scalar>;
    Output const & output = predicted.output();
    EMRYS_FILTER_ASSERT(observed.size() == output.valueDimension());
    typename Output::DifferenceVectorType residual = Emrys::minus(output.space(), observed, output.mean());
    auto const & cross = predicted.crossCovariance();
    auto conditioned = kalmanCondition<false, NoRepair>(
        prior,
        residual,
        output.covarianceLower(),
        cross,
        form,
        linearize,
        NoConsidered{}
    );
    if (!conditioned)
    {
        return std::unexpected{conditioned.error()};
    }
    return Conditioned<typename Prior::SpaceType, OutputSpace, typename Prior::Scalar>{
        std::move(conditioned->posterior),
        {std::move(residual), output.covarianceLower()}
    };
}

/** \returns the JosephTerms of \a predicted when \a Form is updates::Joseph, and an empty struct
 * otherwise, so that the other forms do no work: \f$ H \f$, and \f$ N + \Omega \f$ as \f$ R \f$, from
 * the prediction's own parts (see statisticalLinearization()), so that no noise is recovered by
 * subtraction (decisions.md [linearization-forms]). A prediction in the root form gives JosephRootTerms instead, its
 * root and \f$ \Psi \f$, so that \f$ H \f$ is never solved for.
 */
template <typename Form, typename InputSpace, typename OutputSpace, typename LinearizationForm, typename Scalar>
[[nodiscard]] auto
linearizeFor(OutputWithCross<InputSpace, OutputSpace, LinearizationForm, Scalar> const & predicted)
{
    if constexpr (std::is_same_v<Form, updates::Joseph>)
    {
        using Linearization = StatisticalLinearization<InputSpace, OutputSpace, Scalar>;
        using Predicted = OutputWithCross<InputSpace, OutputSpace, LinearizationForm, Scalar>;
        typename Linearization::ErrorMatrix noise = predicted.inputsCovariance() + predicted.unexplainedCovariance();
        if constexpr (std::is_same_v<LinearizationForm, RootForm>)
        {
            using Terms = JosephRootTerms<
                typename Predicted::RootMatrix const &,
                typename Predicted::CrossMatrix const &,
                typename Linearization::ErrorMatrix>;
            return Terms{predicted.root(), predicted.whitenedCross(), std::move(noise)};
        }
        else
        {
            return JosephTerms<typename Linearization::JacobianMatrix const &, typename Linearization::ErrorMatrix>{
                predicted.jacobian(),
                std::move(noise)
            };
        }
    }
    else
    {
        return std::tuple<>{};
    }
}

}  // namespace internal

/** \ingroup Transforms
 * \returns the statistical linearization of the output of \a predicted, which came from transform()
 * with an input marked by withCross(), in that input: \f$ H \f$, the Jacobian of the affine fit, and
 * \f$ N + \Omega \f$, the covariance of its error (see StatisticalLinearization), read from the
 * prediction's parts (see OutputWithCross), so that neither needs a factorization of the input's
 * covariance or a subtraction (decisions.md [linearization-forms]).
 *
 * The error covariance holds whatever \f$ H \f$ does not explain: the other uncertain inputs, such as
 * noise passed to the function, and the function's curvature over the input's spread. For the
 * Linearized method it is the other inputs' part alone, and \f$ H \f$ is the Jacobian at the mean.
 * For a sigma-point method \f$ H = \Psi^T S^{-1} \f$, and the curvature is the covariance of the
 * fit's residuals at the points, which can be indefinite when a sigma-point weight is negative. It
 * is returned as the method computed it, never projected onto the positive semidefinite cone: a
 * caller that needs it semidefinite makes it so. updates::Joseph takes its \f$ H \f$ and \f$ R \f$
 * from here.
 *
 * \code
 * auto const predicted = Emrys::transform<RangeBearing>(Emrys::methods::unscented(), rangeBearing, withCross(state));
 * auto const [jacobian, errorCov] = Emrys::statisticalLinearization(predicted);
 * \endcode
 *
 * The statistical linear regression through the transform's moments of
 * \ref Lefebvre2002 "Lefebvre, Bruyninckx, and De Schutter (2002)", who read the unscented
 * transform as one; the same formulas hold for every transform method here.
 */
template <typename InputSpace, typename OutputSpace, typename Form, typename Scalar>
[[nodiscard]] StatisticalLinearization<InputSpace, OutputSpace, Scalar>
statisticalLinearization(OutputWithCross<InputSpace, OutputSpace, Form, Scalar> const & predicted)
{
    using Result = StatisticalLinearization<InputSpace, OutputSpace, Scalar>;
    typename Result::ErrorMatrix errorCov = predicted.inputsCovariance() + predicted.unexplainedCovariance();
    return Result{typename Result::JacobianMatrix{predicted.jacobian()}, std::move(errorCov)};
}

/** \ingroup Transforms
 * \returns \a prior conditioned on an observation \a observed of the output of \a predicted,
 * which came from transform() with \a prior marked by withCross(), and the log-likelihood of
 * the observation, with the covariance updated by \a form (updates::Standard by default,
 * updates::Symmetric, or updates::Joseph).
 *
 * The covariance form's fold of one prediction (decisions.md [linearization-forms]), which a measurement's update,
 * update(prior, measurement), ends in for each of its branches. It has a name of its own because
 * it conditions the very input the prediction was made from, with no noise of its own and one
 * posterior, where update() corrects a whole state through the blocks a measurement reads and
 * branches on its noise.
 *
 * The Kalman update: with gain \f$ K = \Sigma_{xy} \Sigma_{yy}^{-1} \f$ and residual
 * \f$ r = y \boxminus \bar y \f$, normalized in the output space, the posterior mean is
 * \f$ \mu \boxplus K r \f$, normalized in the prior's space, and its covariance is
 * \f$ \Sigma - K \Sigma_{xy}^T \f$ in the standard form. Solves with the Cholesky factor of
 * \f$ \Sigma_{yy} \f$. updates::Joseph takes \f$ H \f$ and \f$ R = N + \Omega \f$ from the
 * prediction's parts (statisticalLinearization()), so noise passed to the transform as an uncertain
 * input is never recovered by subtraction.
 *
 * This is the update of \ref Kalman1960 "Kalman (1960)" written as the conditioning of the
 * joint %Gaussian of input and output, the form that holds for every transform method alike
 * (\ref Sarkka2013 "Särkkä (2013)", §6.1).
 *
 * \returns the Conditioned posterior and innovation, or GaussianError::NotPositiveDefinite if
 *          \f$ \Sigma_{yy} \f$ is not positive definite (decisions.md [innovation-cholesky])
 */
template <
    UncertainInput Prior,
    typename OutputSpace,
    typename LinearizationForm,
    typename Derived,
    CovarianceUpdate Form = updates::Standard>
[[nodiscard]] auto
condition(
    Prior const & prior,
    OutputWithCross<typename Prior::SpaceType, OutputSpace, LinearizationForm, typename Prior::Scalar> const &
        predicted,
    Eigen::MatrixBase<Derived> const & observed,
    Form const & form = {}
)
{
    return internal::kalmanUpdate(prior, predicted, observed, form, [&] {
        return internal::linearizeFor<Form>(predicted);
    });
}

}  // namespace Emrys
