#pragma once

/** \file
 * The update that the Rao-Blackwellized filters share (decisions.md [rb-point-mass], [rao-blackwellized-core],
 * [rb-particle-filter]): every point's %Gaussian conditioned on a measurement given the point, with one transform for
 * all points under a shared covariance or one Kalman update each, into the reused PointLikelihoods. Everything here is
 * in `internal`. Read RbPointMassFilter.h and RbParticleFilter.h next.
 */

#include <cmath>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Distributions/Conditionals.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Models/Likelihood.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Models/Reduction.h>

namespace Emrys
{

namespace internal
{

/** \returns the mean of \a input if it is uncertain, or \a input itself if it is known. */
template <typename Input>
[[nodiscard]] decltype(auto)
meanOf(Input const & input)
{
    if constexpr (UncertainInput<Input>)
    {
        return input.mean();
    }
    else
    {
        return (input);
    }
}

/** Each point's log-likelihood and NIS from an update given each point, in storage reused across
 * updates. A point the policy rejected has log-likelihood \f$ -\infty \f$ and NIS NaN; a point
 * whose update failed also has its error (decisions.md [innovation-cholesky]).
 */
template <typename Scalar, typename Allocator = DefaultAllocator>
struct PointLikelihoods
{
    /** Empty storage that allocates through \a allocator. */
    explicit PointLikelihoods(Allocator const & allocator = {})
        : logLikelihoodData(ReBound<Allocator, Scalar>(allocator))
        , nisData(ReBound<Allocator, Scalar>(allocator))
        , errorData(ReBound<Allocator, std::optional<GaussianError>>(allocator))
        , batchData(ReBound<Allocator, Scalar>(allocator))
    {}

    /** Makes every one of \a count points rejected, for an innovation of \a size difference coordinates. */
    void
    reject(Eigen::Index count, Eigen::Index size)
    {
        logLikelihoodData.assign(std::size_t(count), -Eigen::NumTraits<Scalar>::infinity());
        nisData.assign(std::size_t(count), Eigen::NumTraits<Scalar>::quiet_NaN());
        errorData.assign(std::size_t(count), std::nullopt);
        innovationSize = size;
    }

    /** Reserves room for \a count points. */
    void
    reserve(Eigen::Index count)
    {
        logLikelihoodData.reserve(std::size_t(count));
        nisData.reserve(std::size_t(count));
        errorData.reserve(std::size_t(count));
    }

    /** \returns nothing, or the error of the first point whose update failed */
    [[nodiscard]] std::expected<void, GaussianError>
    firstError() const
    {
        for (auto const & error : errorData)
        {
            if (error)
            {
                return std::unexpected{*error};
            }
        }
        return {};
    }

    /** \returns the number of points */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return Eigen::Index(logLikelihoodData.size());
    }

    /** \returns each point's log-likelihood, as a writable map */
    [[nodiscard]] auto
    logLikelihoods()
    {
        return Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>>{logLikelihoodData.data(), size()};
    }

    /** \returns each point's log-likelihood */
    [[nodiscard]] auto
    logLikelihoods() const
    {
        return Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1> const>{logLikelihoodData.data(), size()};
    }

    /** \returns each point's NIS, as a writable map */
    [[nodiscard]] auto
    nis()
    {
        return Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>>{nisData.data(), size()};
    }

    /** \returns each point's NIS */
    [[nodiscard]] auto
    nis() const
    {
        return Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1> const>{nisData.data(), size()};
    }

    Buffer<Scalar, Allocator> logLikelihoodData;                ///< each point's log-likelihood
    Buffer<Scalar, Allocator> nisData;                          ///< each point's NIS
    Buffer<std::optional<GaussianError>, Allocator> errorData;  ///< each point's error, if its update failed
    Buffer<Scalar, Allocator> batchData;                        ///< a batched likelihood's input, a column per point
    Eigen::Index innovationSize = 0;                            ///< the innovation's difference dimension
};

/** Reaches the PointLikelihoods a Rao-Blackwellized filter's update result keeps, for the filter
 * that fills it.
 */
struct PointLikelihoodsAccess
{
    /** \returns \a result's per-point storage */
    template <typename Result>
    [[nodiscard]] static auto &
    points(Result & result)
    {
        return result.m_points;
    }

    /** \returns the cells \a result records, for a point-mass filter's */
    template <typename Result>
    [[nodiscard]] static auto &
    cells(Result & result)
    {
        return result.m_cells;
    }
};

/** Conditions every point's %Gaussian on \a measurement given the point, with one covariance for
 * all of them: one transform at point \a reference gives the innovation covariance and gain,
 * and each point costs an evaluation of the model's function and a mean update.
 *
 * Writes each point's log-likelihood and NIS to \a result and the posterior Gaussians to
 * \a posterior, reusing their storage. The points are a for-each under \a policy.
 *
 * The measurement update of the marginalized particle filter in the case
 * \ref Schon2005 "Schön, Gustafsson, and Nordlund (2005)" call model 1, where the linear
 * substate's gain is the same for every particle, so the Kalman update's covariance is computed
 * once (their §IV.B). Each point's mean moves by its own step, so the shared covariance is not
 * reset to the tangent space at each (decisions.md [covariance-reset]): the reset is part of the shared
 * approximation, exact where the conditional space has an identity reset, and
 * PerPointCovariance applies it per point.
 *
 * \returns nothing, or the error of innovationCholesky(), with \a posterior unspecified, if the
 *          shared innovation covariance does not factorize after the measurement's repair
 *          (decisions.md [innovation-cholesky])
 */
template <ExecutionPolicy P, typename Points, typename Store, Measurement M, typename Likelihoods>
[[nodiscard]] std::expected<void, GaussianError>
updateGivenShared(
    P && policy,
    Points const & points,
    Store const & conditionals,
    Eigen::Index reference,
    M const & measurement,
    Likelihoods & result,
    Store & posterior
)
{
    using Scalar = typename Store::Scalar;
    static_assert(Store::Shared);
    auto const f =
        PinnedMeasurement<typename Points::SpaceType, typename Points::PathType, M>{points.space(), measurement};
    auto const & resolved = f.resolved();
    using R = std::remove_cvref_t<decltype(resolved)>;
    static_assert(
        R::HasMethod,
        "a Kalman update pushes the prior through the model by a method: observe(method, model, z)"
    );
    static_assert(
        R::HasAdditiveNoise && !R::MixtureNoise && R::InputCount == 0,
        "a shared covariance needs additive Gaussian noise and no uncertain input; use PerPointCovariance for mixture "
        "noise or noise through the function"
    );
    static_assert(!isIterated<typename R::Method>, "an update given a block does not support Iterated yet");
    static_assert(
        !hasConsider<R>,
        "an update given a block, as a Rao-Blackwellized filter's, does not consider blocks yet (decisions.md "
        "[consider-states])"
    );
    EMRYS_FILTER_ASSERT(
        conditionals.size() == points.size() && conditionals.size() > 0
        && "give one Gaussian per point, and at least one point"
    );
    auto const prior = conditionals.conditional(reference);
    auto const predicted = withPoint(points, reference, [&](auto const & pinned) {
        return transformPrediction(resolved, resolved.method(), f, withCross(prior), pinned);
    });
    auto const & y = predicted.output();
    auto const & cross = predicted.crossCovariance();
    auto const & noise = resolved.noise().distribution();
    using InnovationMatrix = typename std::remove_cvref_t<decltype(y)>::MatrixType;
    using InnovationVector = typename std::remove_cvref_t<decltype(y)>::DifferenceVectorType;
    InnovationMatrix s = y.covarianceLower() + noise.covarianceLower();
    auto const factored = innovationCholesky<typename R::InnovationRepair>(s);
    if (!factored)
    {
        return std::unexpected{factored.error()};
    }
    auto const & llt = *factored;
    // With nothing left Gaussian, only the likelihoods remain; MSVC can't form an empty gain.
    constexpr bool NothingLeft = std::remove_cvref_t<decltype(cross)>::RowsAtCompileTime == 0;
    auto const gain = [&] {
        if constexpr (NothingLeft)
        {
            return cross;
        }
        else
        {
            return transposedSolve<std::remove_cvref_t<decltype(predicted.crossCovariance())>>(
                llt,
                predicted.crossCovariance().transpose()
            );
        }
    }();
    Eigen::Index const m = y.differenceDimension();
    Scalar const logNormalizer =
        Scalar(-0.5) * (Scalar(m) * log2Pi<Scalar>() + Scalar(2) * llt.matrixLLT().diagonal().array().log().sum());

    result.reject(points.size(), m);
    // Each posterior mean is written once, from the prior's, rather than copied and then moved.
    posterior.resizeLike(conditionals);
    auto posteriorMeans = posterior.means();
    forEachIndex(std::forward<P>(policy), points.size(), [&](Eigen::Index i) {
        // The point's own predicted measurement, moved by the noise's mean, and the observation's
        // residual from it.
        InnovationVector residual = observedResidual<Scalar>(
            resolved,
            y.space(),
            Emrys::plus(y.space(), valueOf(f(conditionals.mean(i), points.center(i))), noise.mean())
        );
        Scalar const nis = residual.dot(llt.solve(residual));
        result.nisData[std::size_t(i)] = nis;
        result.logLikelihoodData[std::size_t(i)] = logNormalizer - Scalar(0.5) * nis;
        if constexpr (!NothingLeft)
        {
            posteriorMeans.col(i) = Emrys::plus(conditionals.space(), conditionals.mean(i), multiply(gain, residual));
        }
    });
    if constexpr (!NothingLeft)
    {
        // The covariance update in the measurement's form; updates::Joseph's R is the noise plus the
        // linearization error of the measurement before it, N + Omega from the prediction's parts.
        using Form = std::remove_cvref_t<decltype(resolved.covarianceUpdate())>;
        using CovMatrix = std::remove_cvref_t<decltype(posterior.sharedCovariance())>;
        CovMatrix symmetric = prior.covarianceLower();
        symmetric.template triangularView<Eigen::StrictlyUpper>() = symmetric.transpose();
        posterior.sharedCovariance() = posteriorCov(resolved.covarianceUpdate(), symmetric, gain, cross, s, [&] {
            if constexpr (std::is_same_v<Form, updates::Joseph>)
            {
                auto linearization = statisticalLinearization(predicted);
                using Linearization = decltype(linearization);
                return JosephTerms<typename Linearization::JacobianMatrix, InnovationMatrix>{
                    std::move(linearization.jacobian),
                    InnovationMatrix{linearization.errorCov + noise.covarianceLower()}
                };
            }
            else
            {
                return std::tuple<>{};
            }
        });
    }
    return {};
}

/** Conditions each point's %Gaussian on \a measurement given the point, each as its own Kalman
 * filter, and reduces its branches with \a policy, which must keep one %Gaussian. A point the
 * policy rejects keeps its %Gaussian and gets log-likelihood \f$ -\infty \f$.
 *
 * Writes each point's log-likelihood and NIS (its most probable branch's) to \a result and the
 * posterior Gaussians to \a posterior, reusing their storage. The points are a for-each under
 * \a executionPolicy, which then runs \a policy for several points at once.
 *
 * The general case of \ref Doucet2000 "Doucet et al. (2000)": one Kalman update per particle,
 * through update(prior, measurement, given).
 *
 * \returns nothing, or the error of the first point whose update failed, with \a posterior
 *          unspecified (decisions.md [innovation-cholesky])
 */
template <ExecutionPolicy P, typename Points, typename Store, Measurement M, typename Policy, typename Likelihoods>
[[nodiscard]] std::expected<void, GaussianError>
updateGivenEach(
    P && executionPolicy,
    Points const & points,
    Store const & conditionals,
    M const & measurement,
    Policy const & policy,
    Likelihoods & result,
    Store & posterior
)
{
    using ConditionalType = typename Store::ConditionalType;
    static_assert(!Store::Shared);
    EMRYS_FILTER_ASSERT(
        conditionals.size() == points.size() && conditionals.size() > 0
        && "give one Gaussian per point, and at least one point"
    );
    // Each posterior is written once: the policy's, or the prior's where it rejected the point.
    posterior.resizeLike(conditionals);
    // Every point starts rejected; the first point's update gives the innovation's size.
    result.reject(points.size(), 0);
    // The measurement over the rest and a point, resolved once for every point.
    auto const f =
        PinnedMeasurement<typename Points::SpaceType, typename Points::PathType, M>{points.space(), measurement};
    forEachIndex(std::forward<P>(executionPolicy), points.size(), [&](Eigen::Index i) {
        // MSVC loses i in a nested generic lambda that captures it implicitly, so it's named.
        auto updated = withPoint(points, i, [&, i](auto const & pinned) {
            return updateBranchesWith<true>(conditionals.conditional(i), f, pinned);
        });
        if (!updated)
        {
            result.errorData[std::size_t(i)] = updated.error();
            return;
        }
        auto & measured = *updated;
        if (i == 0)
        {
            result.innovationSize = measured.innovationSize();
        }
        static_assert(
            ReductionPolicy<Policy, std::remove_cvref_t<decltype(measured)>, ConditionalType>,
            "a Rao-Blackwellized filter keeps one Gaussian per point, so its policy must keep one: "
            "Collapse, Select, or a Gate over them"
        );
        // The policy writes what it keeps to the point's posterior.
        using Scalar = typename std::remove_cvref_t<decltype(measured)>::Scalar;
        auto out = OneComponentOutput<ConditionalType, Scalar>{posterior.conditional(i)};
        reduceInto(policy, std::move(measured.branches), out);
        if (!out.added)
        {
            posterior.conditional(i) = conditionals.conditional(i);
            return;
        }
        std::size_t best = 0;
        for (std::size_t k = 1; k < measured.branches.size(); ++k)
        {
            if (measured.branches[k].logWeight > measured.branches[best].logWeight)
            {
                best = k;
            }
        }
        result.nisData[std::size_t(i)] = measured.branches[best].nis();
        // The point is weighted by the likelihood of the branches what it kept stands for.
        result.logLikelihoodData[std::size_t(i)] = measured.logLikelihood() + out.logWeight;
    });
    return result.firstError();
}

/** A compile-time error unless the block named by the path \a Read lies in the pinned block named
 * by \a Pinned; the instantiation names both.
 */
template <typename Pinned, typename Read>
consteval void
assertReadsPinned()
{
    static_assert(
        pathLiesIn<Read, Pinned>,
        "logLikelihood() may read only the gridded or sampled block: with the rest conditioned on it, "
        "the rest would no longer be Gaussian"
    );
}

/** Weights every point by the model's own logLikelihood(g) at the point's center, every block it
 * reads lying in the pinned block, and leaves every point's %Gaussian as it was: a likelihood that
 * reads none of the rest tells it nothing.
 *
 * Writes each point's log-likelihood to \a result, with NIS NaN and an innovation of size 0, and
 * nothing else, so that the filter reweights the points and keeps their Gaussians where they are
 * (decisions.md [update-outcomes]). The points are a for-each under \a policy. A point's kernel covariance is
 * ignored: the likelihood is evaluated at its center. A batched likelihood is called once over the
 * points under Sequential and once per chunk otherwise (decisions.md [batched-functions]).
 *
 * The update of the point-mass filter of \ref Bucy1971 "Bucy and Senne (1971)", each cell's
 * probability multiplied by the likelihood at its center, as \ref Bergman1999 "Bergman (1999)",
 * chapter 5, writes it for a terrain map; and the weighting of
 * \ref Arulampalam2002 "Arulampalam et al. (2002)", equation (48), for particles.
 */
template <ExecutionPolicy P, typename Points, typename Store, WeighsPoints M, typename Likelihoods>
void
updateGivenLikelihood(
    P && policy,
    Points const & points,
    Store const & conditionals,
    M const & measurement,
    Likelihoods & result
)
{
    using Scalar = typename Store::Scalar;
    using Pinned = typename Points::PathType;
    auto const resolved = resolveMeasurement(measurement, points.space());
    using R = std::remove_cvref_t<decltype(resolved)>;
    static_assert(R::HasLogLikelihood, "a measurement weighted by its likelihood alone has logLikelihood(g)");
    static_assert(
        !hasConsider<R>,
        "consider() is a Kalman update's; a filter that weights points moves no block it does not condition "
        "(decisions.md [consider-states])"
    );
    [&]<typename... Paths>(TagPaths<Paths...> *) {
        (assertReadsPinned<Pinned, Paths>(), ...);
    }(static_cast<typename R::StatePaths *>(nullptr));
    EMRYS_FILTER_ASSERT(
        conditionals.size() == points.size() && conditionals.size() > 0
        && "give one Gaussian per point, and at least one point"
    );
    Eigen::Index const start = points.pinnedRows().start;
    result.reject(points.size(), 0);
    if constexpr (R::BatchedLikelihood)
    {
        fillBatchedLogLikelihoods(
            std::forward<P>(policy),
            resolved,
            points.size(),
            [&](Eigen::Index i) {
                return points.center(i);
            },
            start,
            result.logLikelihoodData.data(),
            result.batchData
        );
    }
    else
    {
        forEachIndex(std::forward<P>(policy), points.size(), [&](Eigen::Index i) {
            result.logLikelihoodData[std::size_t(i)] = Scalar(resolved.logLikelihoodAtRows(points.center(i), start));
        });
    }
}

/** Updates each point's hybrid in the per-point store \a hybrids where it is, by
 * \a updateOne(hybrid, pinned, oneResult), with the point's value or its kernel as \a pinned (see
 * withPoint()) and \a oneResult, an ImmFilterUpdate, reused from point to point, and records in
 * \a likelihoods each point's log-likelihood, the IMM's evidence, and the NIS of its most probable
 * mode after the update, of that mode's most probable branch. A point no mode accepts keeps its
 * hybrid, which the recursion leaves as it was, and gets likelihood zero. Nothing is copied to undo a
 * failure (decisions.md [update-outcomes]): a point whose update fails keeps its hybrid, and the
 * points before it keep theirs updated.
 *
 * \returns nothing, or the first failed point's error
 */
template <typename Points, typename Store, typename Likelihoods, typename OneResult, typename UpdateOne>
[[nodiscard]] std::expected<void, NumericError>
updateHybridsGiven(
    Points const & points,
    Store & hybrids,
    Likelihoods & likelihoods,
    OneResult & oneResult,
    UpdateOne const & updateOne
)
{
    likelihoods.reject(points.size(), 0);
    for (Eigen::Index i = 0; i < points.size(); ++i)
    {
        auto & hybrid = hybrids.conditional(i);
        auto const updated = withPoint(points, i, [&](auto const & pinned) {
            return updateOne(hybrid, pinned, oneResult);
        });
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        if (i == 0)
        {
            likelihoods.innovationSize = oneResult.modes.front().innovationSize();
        }
        if (!oneResult.accepted)
        {
            continue;
        }
        likelihoods.logLikelihoodData[std::size_t(i)] = oneResult.logLikelihood();
        // The NIS of the most probable mode after the update, and of its most probable branch.
        Eigen::Index mode = 0;
        hybrid.modes().logWeights().maxCoeff(&mode);
        auto const & branches = oneResult.modes[std::size_t(mode)].branches;
        std::size_t best = 0;
        for (std::size_t k = 1; k < branches.size(); ++k)
        {
            if (branches[k].logWeight > branches[best].logWeight)
            {
                best = k;
            }
        }
        likelihoods.nisData[std::size_t(i)] = branches[best].nis();
    }
    return {};
}

/** Whether \a Model is a SwitchingModel, under which each point of a Rao-Blackwellized filter holds
 * a hybrid.
 */
template <typename Model>
inline constexpr bool isSwitchingPointModel = requires { typename Model::ModeSpaceType; };

/** The covariance form a Rao-Blackwellized filter with \a Model takes by default: SharedCovariance,
 * or PerPointCovariance for a switching model, whose mixing gives each point its own.
 */
template <typename Model>
using PointCovarianceFor = std::conditional_t<isSwitchingPointModel<Model>, PerPointCovariance, SharedCovariance>;

/** A Kalman filter's steps for every point: the process model and the prediction method. */
template <typename Model_, typename Method_>
class PointSteps
{
public:

    /** No model, for a grid that covers the whole state. */
    PointSteps() = default;

    /** \a model and \a method; the allocator is the IMM's, which keeps scratch. */
    template <typename Allocator>
    PointSteps(Model_ model, MethodFor<Model_, Method_> method, Allocator const &)
        : m_model{std::move(model)}
        , m_method{predictionMethod(m_model, std::move(method))}
    {}

    /** \returns the process model */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_model;
    }

    /** \returns the process model, to change its parameters */
    [[nodiscard]] Model_ &
    model()
    {
        return m_model;
    }

    /** \returns the prediction method */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_method;
    }

private:

    Model_ m_model;                                   ///< the process model
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method_ m_method;  ///< the prediction method
};

}  // namespace internal

}  // namespace Emrys
