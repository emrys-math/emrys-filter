#pragma once

/** \file
 * The update of a %Gaussian by a measurement (decisions.md [templated-measurements], [weighted-branches],
 * [covariance-update-forms], [innovation-statistics], [linearization-reference], [rb-point-mass]): update()
 * of a state by a measurement that reads part of it, with one UpdateBranch per noise component, and
 * update() given a block from outside the %Gaussian. It includes what it builds on: MeasurementModel.h
 * (the models and the observations observe() makes of them), MeasurementUpdate.h (what an update returns),
 * MeasurementReads.h and IteratedUpdate.h (how an update reads a measurement and iterates), and
 * Likelihood.h (the logLikelihoods() of sample-based filters). Read SwitchingModel.h, Reduction.h, and
 * Consistency.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Models/IteratedUpdate.h>
#include <EmrysFilter/Models/Likelihood.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/MeasurementReads.h>
#include <EmrysFilter/Models/MeasurementUpdate.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The body of updateInto(): the update of \a prior by \a resolved, its measurement resolved against
 * the prior's space (ResolvedMeasurement), with the posteriors put where \a sink says, each branch's
 * NIS and log-likelihood computed with the gain when \a Statistics (decisions.md
 * [innovation-statistics]), linearized about \a reference, a linearizedAbout() reference, or at the
 * prior as usual where it is NoReference [linearization-reference].
 */
template <bool Statistics, UncertainInput Prior, typename R, typename Sink, typename Reference = NoReference>
[[nodiscard]] decltype(auto)
updateResolved(Prior const & prior, R const & resolved, Sink sink, Reference const & reference = {})
{
    using Scalar = typename Prior::Scalar;
    using Space = typename Prior::SpaceType;
    using Reads = typename R::Reads;
    constexpr int N = valueDimensionAtCompileTime<DifferenceSpace<Space>>;
    constexpr int NMax = maxValueDimensionAtCompileTime<DifferenceSpace<Space>>;

    static_assert(
        R::HasMethod,
        "a Kalman update pushes the prior through the model by a method: observe(method, model, z)"
    );
    static_assert(
        R::HasFunction,
        "a Kalman update needs the model's function; a model known by its likelihood alone serves the filters that "
        "weight points"
    );
    auto const & reads = resolved.reads();
    auto const read = GaussianMarginal<Prior, typename Reads::Selection, typename Reads::JointSpace>{
        prior,
        reads.selection(),
        reads.space()
    };
    auto const f = ResolvedFunction{resolved};
    auto const & method = resolved.method();
    using Method = std::remove_cvref_t<decltype(method)>;
    auto const all = internal::consecutiveIndices<N, NMax>(0, prior.differenceDimension());
    // The covariance of the whole state with the blocks read, P_xS.
    using StateReadMatrix = internal::MatrixOf<Scalar, N, Reads::DifferenceSize, NMax, Reads::DifferenceMaxSize>;
    StateReadMatrix const stateRead =
        internal::lowerSymmetricBlock(covarianceLowerOf(prior), all, reads.differenceIndices());

    if constexpr (isLinearizedAbout<Reference>)
    {
        // One linearization about the reference serves every noise component, which conditionOnPrediction()
        // then branches on as it does for one at the prior.
        static_assert(checkReference<Method, Reference, Space>());
        using Vector = typename GaussianParams<Space, Scalar>::VectorType;
        using Difference = typename GaussianParams<Space, Scalar>::DifferenceVectorType;
        using ReadVector = typename MovedInput<std::remove_cvref_t<decltype(read)>>::VectorType;
        using ReadDifference = typename GaussianParams<typename Reads::JointSpace, Scalar>::DifferenceVectorType;
        Vector const point = reference.mean();
        EMRYS_FILTER_ASSERT(
            point.size() == prior.valueDimension() && "a point to linearize about must have the prior's value dimension"
        );
        Difference const step = Emrys::minus(prior.space(), point, prior.mean());
        ReadVector const readMean = read.mean();
        ReadDifference const readStep = step(reads.differenceIndices());
        auto const condition = [&](auto const & y, auto const & cross, auto const & readJacobian) -> decltype(auto) {
            return conditionOnPrediction<Statistics>(prior, resolved, y, cross, readJacobian, sink);
        };
        if constexpr (isLinearMethod<Method>)
        {
            return linearizeReadsAt<Space, Scalar>(
                resolved,
                method,
                reads,
                read,
                f,
                stateRead,
                readMean,
                true,
                readStep,
                std::monostate{},
                condition
            );
        }
        else if constexpr (Reference::IsDistribution)
        {
            using ReadMatrix = internal::MatrixOf<
                Scalar,
                Reads::DifferenceSize,
                Reads::DifferenceSize,
                Reads::DifferenceMaxSize,
                Reads::DifferenceMaxSize>;
            // The reference's spread over the blocks read, in its own chart, as an iterate's is in the iterate's.
            ReadMatrix const spreadCov = internal::lowerSymmetricBlock(
                reference.reference.covarianceLower(),
                reads.differenceIndices(),
                reads.differenceIndices()
            );
            return linearizeReadsAt<Space, Scalar>(
                resolved,
                method,
                reads,
                read,
                f,
                stateRead,
                readMean,
                true,
                readStep,
                spreadCov,
                condition
            );
        }
    }
    else if constexpr (isIterated<Method>)
    {
        if constexpr (R::MixtureNoise)
        {
            // Each component's branch is its own Iterated update, weighted as conditionOnPrediction()
            // weights the branches of one linearization.
            static_assert(!isIntoPrior<Sink>, "conditioning in place takes noise of one component");
            auto const & noise = resolved.noise().distribution();
            using Component = std::remove_cvref_t<decltype(noise.begin()->component)>;
            using One = WithNoiseComponent<typename R::ObservedType, Component>;
            using Single =
                typename decltype(updateInto<true>(prior, std::declval<One const &>(), NewPosteriors{}))::value_type;
            using MeasurementSpace = typename Single::MeasurementSpaceType;
            auto const total = noise.logTotalWeight();
            EMRYS_FILTER_ASSERT(total && "the noise mixture's weights must be valid");
            decltype(auto) result = mixtureUpdateFor<Space, MeasurementSpace, Scalar>(sink);
            using Outcome = MixtureOutcome<Sink, std::remove_cvref_t<decltype(result)>>;
            result.branches.reserve(std::size_t(noise.size()));
            auto evidence = LogSumExp<Scalar>{};
            Eigen::Index k = 0;
            for (auto const & entry : noise)
            {
                auto const one = One{resolved.observed(), entry.component};
                auto single = updateInto<true>(prior, one, NewPosteriors{});
                if (!single)
                {
                    return Outcome{std::unexpected{single.error()}};
                }
                auto & branch = single->branches.front();
                branch.component = k++;
                branch.logWeight = entry.logWeight - *total + branch.logLikelihood();
                evidence.add(branch.logWeight);
                result.branches.push_back(std::move(branch));
            }
            auto const logEvidence = evidence.value();
            MixtureEvidenceAccess::setMixtureLogLikelihood(result, logEvidence);
            for (auto & branch : result.branches)
            {
                branch.logWeight -= logEvidence;
            }
            return mixtureOutcome<Sink>(result);
        }
        else
        {
            return iteratedUpdate<Statistics>(prior, resolved, reads, read, f, stateRead, sink);
        }
    }
    else if constexpr (internal::isLinearMethod<Method>)
    {
        // The method sees the measurement as affine in the blocks read, J_S (S - mu_S) plus terms
        // independent of the state, so its cross-covariance with the whole state is P_xS J_S^T,
        // with no solve against P_SS.
        auto const condition = [&](auto const & problem, auto const & linearized) -> decltype(auto) {
            using Problem = std::remove_cvref_t<decltype(problem)>;
            using Predicted = OutputWithCross<Space, typename Problem::OutputSpaceType, JacobianForm, Scalar>;
            auto const & jacobian = linearized.jacobian;
            Eigen::Index const readSize = reads.differenceIndices().size();
            auto const readJacobian = ReadLinearization{
                jacobian.template middleCols<Reads::DifferenceSize>(problem.differenceOffset(0), readSize),
                reads.differenceIndices(),
                [&] {
                    using Error = typename Problem::OutputMatrix;
                    return internal::inputsCovariance<Error>(resolved, problem, jacobian);
                }
            };
            auto cross = typename Predicted::CrossMatrix{};
            cross.noalias() = internal::multiply(stateRead, readJacobian.jacobian.transpose());
            if constexpr (R::InputCount > 0)
            {
                typename Problem::CrossMatrix const joint =
                    internal::multiply(problem.covarianceLower(), jacobian.transpose());
                auto const y = internal::PredictionMoments{problem.outputSpace(), linearized.mean, jacobian, joint};
                return internal::conditionOnPrediction<Statistics>(prior, resolved, y, cross, readJacobian, sink);
            }
            else
            {
                // The blocks read are the only uncertain input, so P_SS J^T is the rows of
                // P_xS J^T at them, borrowed rather than formed again (decisions.md [borrowed-intermediates]).
                auto const joint = std::as_const(cross)(reads.differenceIndices(), Eigen::placeholders::all);
                auto const y = internal::PredictionMoments{problem.outputSpace(), linearized.mean, jacobian, joint};
                return internal::conditionOnPrediction<Statistics>(prior, resolved, y, cross, readJacobian, sink);
            }
        };
        return internal::linearizeMeasurement(resolved, method, f, read, condition);
    }
    else
    {
        // Lift the cross-covariance from the blocks read to the full state: P_xS P_SS^-1 Sigma_Sy, which
        // is P_xS S^-T Psi with S the root that spread the points, so P_SS is not factorized again
        // (decisions.md [linearization-forms]).
        auto const predicted = internal::transformMeasurement(resolved, method, f, read);
        using Predicted = std::remove_cvref_t<decltype(predicted)>;
        using Solved = typename Predicted::CrossMatrix;
        Solved const solved = internal::solveWithRoot<Solved>(predicted.root(), predicted.whitenedCross());
        using MeasurementSpace = std::remove_cvref_t<decltype(predicted.output().space())>;
        typename OutputWithCross<Space, MeasurementSpace, JacobianForm, Scalar>::CrossMatrix const cross =
            internal::multiply(stateRead, solved);
        // The measurement's Jacobian in the blocks read, Sigma_Sy^T P_SS^-1, and N + Omega, for updates::Joseph.
        auto const readJacobian =
            internal::ReadLinearization{solved.transpose(), reads.differenceIndices(), [&] {
                                            using Error = typename Predicted::OutputMatrix;
                                            return Error{
                                                predicted.inputsCovariance() + predicted.unexplainedCovariance()
                                            };
                                        }};
        return internal::conditionOnPrediction<Statistics>(
            prior,
            resolved,
            predicted.output(),
            cross,
            readJacobian,
            sink
        );
    }
}

/** update(prior, measurement) with the posteriors put where \a sink says, and each branch's NIS
 * and log-likelihood computed with the gain when \a Statistics, for a caller that will read them
 * (decisions.md [innovation-statistics]), linearized about \a reference, a linearizedAbout() reference, or
 * at the prior as usual where it is NoReference [linearization-reference]. The measurement is
 * resolved against the prior's space once, here (ResolvedMeasurement).
 */
template <bool Statistics, UncertainInput Prior, Measurement M, typename Sink, typename Reference = NoReference>
[[nodiscard]] decltype(auto)
updateInto(Prior const & prior, M const & measurement, Sink sink, Reference const & reference = {})
{
    auto const resolved = resolveMeasurement(measurement, prior.space());
    return updateResolved<Statistics>(prior, resolved, std::move(sink), reference);
}

/** update(prior, measurement), with each branch's NIS and log-likelihood computed with the gain
 * when \a Statistics, for a caller that will read them (decisions.md [innovation-statistics]), linearized
 * about \a reference where it is a linearizedAbout() reference [linearization-reference].
 */
template <bool Statistics, UncertainInput Prior, Measurement M, typename Reference = NoReference>
[[nodiscard]] auto
updateBranches(Prior const & prior, M const & measurement, Reference const & reference = {})
{
    return updateInto<Statistics>(prior, measurement, NewPosteriors{}, reference);
}

/** The MeasurementUpdate of \a Prior by \a M: what updateBranches() returns inside its
 * \c std::expected, and what a reduction policy is given (see ReductionPolicy).
 */
template <UncertainInput Prior, Measurement M>
using MeasuredUpdate =
    typename decltype(updateBranches<false>(std::declval<Prior const &>(), std::declval<M const &>()))::value_type;

/** updateBranches() filling \a into, a MeasurementUpdate of a run-time number of branches, where
 * it is: cleared and refilled within its storage, so that a caller who keeps it allocates nothing
 * once it has held as many branches (see \ref Allocation).
 *
 * \returns nothing, or GaussianError::NotPositiveDefinite if a branch's innovation covariance is
 *          not positive definite (decisions.md [innovation-cholesky]), with \a into partly filled
 */
template <
    bool Statistics,
    UncertainInput Prior,
    Measurement M,
    typename Space,
    typename MeasurementSpace,
    typename Scalar,
    typename Allocator,
    typename Reference = NoReference>
[[nodiscard]] std::expected<void, GaussianError>
updateBranchesInto(
    Prior const & prior,
    M const & measurement,
    MeasurementUpdate<Space, MeasurementSpace, std::dynamic_extent, Scalar, Allocator> & into,
    Reference const & reference = {}
)
{
    using Update = MeasurementUpdate<Space, MeasurementSpace, std::dynamic_extent, Scalar, Allocator>;
    return updateInto<Statistics>(prior, measurement, ReusedUpdate<Update>{&into}, reference);
}

/** \a state conditioned in place on \a measurement, whose noise has one component, as a filter
 * whose state is the update's one branch does it (decisions.md [borrowed-intermediates]).
 *
 * Linearized about \a reference where it is a linearizedAbout() reference [linearization-reference].
 * \a accept, an acceptor (see AcceptEvery), is asked on the NIS before \a state is written, as a
 * Gate on the one branch asks (decisions.md [update-outcomes]).
 *
 * \returns the update without its posterior (see FilterUpdate), accepted unless \a accept refused
 *          it, with \a state unchanged, and with the NIS and log-likelihood computed with the gain
 *          when \a Statistics or \a accept reads them; or GaussianError::NotPositiveDefinite, with
 *          \a state unchanged, if the innovation covariance is not positive definite (decisions.md
 *          [innovation-cholesky])
 */
template <
    bool Statistics,
    typename Space,
    typename Scalar,
    Measurement M,
    typename Reference = NoReference,
    typename Accept = AcceptEvery>
[[nodiscard]] auto
updateInPlace(
    GaussianParams<Space, Scalar> & state,
    M const & measurement,
    Reference const & reference = {},
    Accept const & accept = {}
)
{
    return updateInto<Statistics>(
        std::as_const(state),
        measurement,
        IntoPrior<GaussianParams<Space, Scalar>, Accept>{state, accept},
        reference
    );
}

}  // namespace internal

/** \ingroup Models
 * \returns \a prior conditioned on \a measurement, as weighted branches (see MeasurementUpdate).
 *
 * Only the blocks the measurement's model reads, \f$ S \f$, go through its transform. The full state
 * is then corrected through the cross-covariance
 * \f[ \Sigma_{xy} = P_{xS} \, P_{SS}^{-1} \, \Sigma_{Sy}, \f]
 * which is exact under a %Gaussian prior: \f$ E[x \mid S] \f$ is affine in \f$ S \f$, and the
 * remainder is independent of \f$ S \f$ and so of \f$ h(S) \f$. A linear method's
 * \f$ P_{SS}^{-1} \Sigma_{Sy} \f$ is its Jacobian, and a sigma-point method's
 * \f$ R^{-T} \Psi \f$, a solve with the root \f$ R \f$ that spread its points (decisions.md [linearization-forms]),
 * whose least-norm solution keeps a block known exactly fine.
 *
 * The innovation covariance must be positive definite, since each branch's gain and likelihood
 * come from its Cholesky factor (decisions.md [innovation-cholesky]). Measurement noise that is positive definite
 * makes it so; a measurement with no noise of a quantity the prior knows exactly does not, and
 * neither can rounding or an unscented transform with a negative center weight. Such an update
 * returns GaussianError::NotPositiveDefinite, whichever branch it was in. A constraint that a
 * measurement with no noise would impose is better kept by a space (see \ref Spaces) or a small
 * positive noise.
 *
 * The measurement \f$ \hat z \f$ is predicted in the measurement space, and the residual of the
 * observation from it, \f$ z \boxminus \hat z \f$, is formed there, so a bearing wraps and a
 * rotation's residual is its rotation vector. With additive noise, the measurement function is
 * transformed once, whatever the number of noise components. Component \f$ k \f$, with weight
 * \f$ w_k \f$, mean \f$ \mu_k \f$, and covariance \f$ R_k \f$, predicts the measurement
 * \f$ \hat z \boxplus \mu_k \f$ with covariance \f$ \Sigma_{zz} + R_k \f$, and its branch is the
 * Kalman update (see update()) by the observation. Its weight is
 * \f$ w_k \, p(z \mid k) / p(z) \f$. An uncertain input of the call goes through the function with
 * the blocks read, and widens \f$ \Sigma_{zz} \f$.
 *
 * The covariance is updated in the form the measurement was given (see observe()),
 * updates::Standard by default. updates::Joseph linearizes the measurement before
 * additive noise once (see statisticalLinearization()), and each component adds its own
 * \f$ R_k \f$ to the linearization error, so a precise measurement's noise is never recovered
 * by subtraction.
 *
 * The branches are the hypotheses of a %Gaussian-sum update
 * (\ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)"); with one inlier and
 * one outlier component and the Collapse policy, the result is the probabilistic data
 * association update of \ref BarShalom1975 "Bar-Shalom and Tse (1975)".
 *
 * \returns the branches, or GaussianError::NotPositiveDefinite
 *
 * \sa observe(), Measurement, \ref Models
 */
template <UncertainInput Prior, Measurement M>
[[nodiscard]] auto
update(Prior const & prior, M const & measurement)
{
    return internal::updateBranches<false>(internal::withCovariance(prior), measurement);
}

/** \ingroup Models
 * \returns \a prior conditioned on \a measurement, as update(prior, measurement) does, with the
 * measurement linearized about \a about rather than at the prior (decisions.md [linearization-reference]):
 * by Linearized at the reference's point, or by a sigma-point method over its distribution, with
 * the Jacobian carried to the prior's chart and the predicted measurement moved back along it to the
 * prior mean, and the Kalman update of the prior under that linearization. About the prior's own
 * mean, Linearized gives update(prior, measurement) to rounding; about the posterior of an update,
 * it is the next iteration of an Iterated one.
 *
 * Every noise component branches from the one linearization. An Iterated measurement, which chooses
 * its own points, refuses a reference at compile time, as does a sigma-point method given a point.
 *
 * \returns the branches, or GaussianError::NotPositiveDefinite
 *
 * \sa linearizedAbout(), \ref Models
 */
template <UncertainInput Prior, Measurement M, typename Reference>
[[nodiscard]] auto
update(Prior const & prior, M const & measurement, LinearizedAbout<Reference> const & about)
{
    return internal::updateBranches<false>(internal::withCovariance(prior), measurement, about);
}

namespace internal
{

/** The update of \a prior by \a resolved, a measurement resolved against the prior's space, once its
 * measurement is predicted: \a y, with cross-covariance \a cross between \a prior and \a y, and
 * \a readJacobian, a ReadLinearization, for updates::Joseph. One branch per noise component, as
 * update(prior, measurement) describes, with the posteriors put where \a sink says: a
 * MeasurementUpdate with NewPosteriors, and a FilterUpdate with IntoPrior, which takes noise of
 * one component.
 */
template <
    bool Statistics,
    typename Prior,
    typename R,
    typename Output,
    typename Cross,
    typename Jacobian,
    typename Sink>
[[nodiscard]] decltype(auto)
conditionOnPrediction(
    Prior const & prior,
    R const & resolved,
    Output const & y,
    Cross const & cross,
    [[maybe_unused]] Jacobian const & readJacobian,
    Sink && sink
)
{
    using Scalar = typename Prior::Scalar;
    using Space = typename Prior::SpaceType;
    using MeasurementSpace = std::remove_cvref_t<decltype(y.space())>;
    using MeasurementVector = typename GaussianParams<MeasurementSpace, Scalar>::VectorType;
    using InnovationVector = typename GaussianParams<MeasurementSpace, Scalar>::DifferenceVectorType;
    using InnovationMatrix = typename GaussianParams<MeasurementSpace, Scalar>::MatrixType;
    MeasurementVector const observed = observedValue<Scalar>(resolved, y.space());

    // updates::Joseph's H and linearization error, once for every noise component: of the measurement
    // before additive noise, which each component then adds back as its own R. H is the measurement's
    // Jacobian in the columns its ReadLinearization names, and the error the N + Omega it gives, a
    // product of the prediction's, never Sigma_yy less H P H^T (decisions.md [linearization-forms]).
    auto const & form = resolved.covarianceUpdate();
    using Form = std::remove_cvref_t<decltype(form)>;
    auto const linearization = [&] {
        if constexpr (std::is_same_v<Form, updates::Joseph>)
        {
            using Result = StatisticalLinearization<Space, MeasurementSpace, Scalar>;
            typename Result::JacobianMatrix jacobian =
                Result::JacobianMatrix::Zero(y.differenceDimension(), prior.differenceDimension());
            jacobian(Eigen::placeholders::all, readJacobian.indices) = readJacobian.jacobian;
            typename Result::ErrorMatrix errorCov = readJacobian.error();
            return Result{std::move(jacobian), std::move(errorCov)};
        }
        else
        {
            return std::tuple<>{};
        }
    }();

    constexpr bool MixtureParams = R::MixtureNoise;
    static_assert(
        !(MixtureParams && isIntoPrior<std::remove_cvref_t<Sink>>),
        "conditioning in place takes noise of one component"
    );
    using Branch = UpdateBranch<Space, MeasurementSpace, Scalar>;
    constexpr bool InPlace = isIntoPrior<std::remove_cvref_t<Sink>>;
    constexpr bool BranchStatistics = Statistics || MixtureParams;
    using BranchRepair = typename R::InnovationRepair;
    // The blocks considered, which every branch keeps as the prior has them [consider-states].
    auto const considered = consideredRows<R>(prior.space());

    // What the updates take to make updates::Joseph's terms for a noise component whose covariance is
    // noise: a callable that makes them when the form asks, and nothing for the forms that never
    // ask. noise must outlive the callable.
    auto const termsFor = [&]([[maybe_unused]] InnovationMatrix const & noise) {
        if constexpr (std::is_same_v<Form, updates::Joseph>)
        {
            return [&] {
                // The Jacobian is the same for every component, so the terms borrow it.
                using Linearization = StatisticalLinearization<Space, MeasurementSpace, Scalar>;
                return JosephTerms<typename Linearization::JacobianMatrix const &, InnovationMatrix>{
                    linearization.jacobian,
                    InnovationMatrix{linearization.errorCov + noise}
                };
            };
        }
        else
        {
            return std::tuple<>{};
        }
    };

    // The branch whose predicted measurement has mean mean and the covariance makeCov() returns,
    // conditioned on the residual z - mean, with its own posterior, or the error of a covariance
    // that is not positive definite (decisions.md [innovation-cholesky]). Its NIS and log-likelihood come from the
    // gain's factorization when they will be read, as weighting a mixture's branches reads them.
    auto const condition = [&](MeasurementVector const & mean,
                               auto const & makeCov,
                               InnovationMatrix const & noise,
                               Eigen::Index component) -> std::expected<Branch, GaussianError> {
        InnovationVector residual = Emrys::minus(y.space(), observed, mean);
        InnovationMatrix cov = makeCov();
        auto conditioned = kalmanCondition<BranchStatistics, BranchRepair>(
            prior,
            residual,
            cov,
            cross,
            form,
            termsFor(noise),
            considered
        );
        if (!conditioned)
        {
            return std::unexpected{conditioned.error()};
        }
        return Branch{
            Scalar(0),
            std::move(conditioned->posterior),
            {std::move(residual), std::move(cov), conditioned->statistics},
            0,
            component
        };
    };

    if constexpr (MixtureParams)
    {
        auto const & noise = resolved.noise().distribution();
        auto const total = noise.logTotalWeight();
        EMRYS_FILTER_ASSERT(total && "the noise mixture's weights must be valid");
        decltype(auto) result = mixtureUpdateFor<Space, MeasurementSpace, Scalar>(sink);
        using Outcome = MixtureOutcome<std::remove_cvref_t<Sink>, std::remove_cvref_t<decltype(result)>>;
        result.branches.reserve(std::size_t(noise.size()));
        auto evidence = LogSumExp<Scalar>{};
        Eigen::Index k = 0;
        // A component whose covariance is not positive definite fails the whole update [innovation-cholesky].
        for (auto const & entry : noise)
        {
            auto branch = condition(
                Emrys::plus(y.space(), y.mean(), entry.component.mean()),
                [&] {
                    return innovationCovariancePlus<InnovationMatrix>(y, entry.component.covarianceLower());
                },
                entry.component.covarianceLower(),
                k++
            );
            if (!branch)
            {
                return Outcome{std::unexpected{branch.error()}};
            }
            branch->logWeight = entry.logWeight - *total + branch->logLikelihood();
            evidence.add(branch->logWeight);
            result.branches.push_back(std::move(*branch));
        }
        auto const logEvidence = evidence.value();
        MixtureEvidenceAccess::setMixtureLogLikelihood(result, logEvidence);
        for (auto & branch : result.branches)
        {
            branch.logWeight -= logEvidence;
        }
        return mixtureOutcome<std::remove_cvref_t<Sink>>(result);
    }
    else
    {
        // The one branch conditioned in place: S is made in the Innovation that keeps it, and the
        // state conditioned on it there; a failed conditioning, or one the sink's acceptor refuses
        // on its NIS, leaves the state as it was (decisions.md [update-outcomes]). The update is
        // built where it is returned, since each move of the branch would copy the innovation's
        // fixed-size residual and S: transform() initializes its value from the prvalue, and the
        // one return lets the compiler construct it in the caller's object, which Clang does in a
        // generic lambda only when its return type is written out. into is the sink, an IntoPrior.
        using InPlaceUpdate = FilterUpdate<MeasurementSpace, 1, Scalar>;
        auto const inPlace = [&](auto & into,
                                 MeasurementVector const & mean,
                                 auto const & makeCov,
                                 InnovationMatrix const & noise) -> std::expected<InPlaceUpdate, GaussianError> {
            auto failure = std::optional<GaussianError>{};
            bool accepted = true;
            auto const terms = termsFor(noise);
            auto result = std::expected<void, GaussianError>{}.transform([&] {
                return InPlaceUpdate{
                    {FilterBranch<MeasurementSpace, Scalar>{
                        Scalar(0),
                        Innovation<MeasurementSpace, Scalar>{
                            InnovationInPlace{},
                            InnovationVector{Emrys::minus(y.space(), observed, mean)},
                            makeCov,
                            [&](auto const & residual, auto & cov) {
                                auto conditioned = kalmanConditionInPlace<Statistics, BranchRepair>(
                                    into.state,
                                    residual,
                                    cov,
                                    cross,
                                    form,
                                    terms,
                                    considered,
                                    into.accept
                                );
                                if (!conditioned)
                                {
                                    failure = conditioned.error();
                                    return std::optional<InnovationStatistics<Scalar>>{};
                                }
                                accepted = conditioned->accepted;
                                return conditioned->statistics;
                            }
                        },
                        0,
                        0
                    }},
                    Scalar(0),
                    true
                };
            });
            if (failure)
            {
                result = std::unexpected{*failure};
            }
            else
            {
                result->accepted = accepted;
            }
            return result;
        };

        // The one branch, or the error of its conditioning.
        auto const single = [&](MeasurementVector const & mean, auto const & makeCov, InnovationMatrix const & noise) {
            if constexpr (InPlace)
            {
                return inPlace(sink, mean, makeCov, noise);
            }
            else
            {
                using One = MeasurementUpdate<Space, MeasurementSpace, 1, Scalar>;
                auto branch = condition(mean, makeCov, noise, 0);
                if (!branch)
                {
                    return std::expected<One, GaussianError>{std::unexpected{branch.error()}};
                }
                return std::expected<One, GaussianError>{One{{std::move(*branch)}}};
            }
        };
        if constexpr (!R::HasAdditiveNoise)
        {
            return single(
                y.mean(),
                [&] {
                    return InnovationMatrix{y.covarianceLower()};
                },
                InnovationMatrix::Zero(y.differenceDimension(), y.differenceDimension())
            );
        }
        else
        {
            auto const & noise = resolved.noise().distribution();
            return single(
                Emrys::plus(y.space(), y.mean(), noise.mean()),
                [&] {
                    return innovationCovariancePlus<InnovationMatrix>(y, noise.covarianceLower());
                },
                noise.covarianceLower()
            );
        }
    }
}

}  // namespace internal

/** \ingroup Models
 * A block of a state given from outside the %Gaussian being updated, as given() makes it, for
 * update(prior, measurement, given): the block named by the path \a P of a state in \a Space,
 * given as \a Value. The space is a copy, since a space is empty or holds its sizes, so only the
 * value is borrowed.
 */
template <typename Space, typename P, typename Value>
struct GivenBlock
{
    Space space;          ///< the whole state space
    Value const & value;  ///< the block's value, known or an uncertain input
};

namespace internal
{

/** A measurement's model as a function of the rest of the state and the block named by the path
 * \a P, which together make up a state in \a Space: <tt>(x, p, inputs...)</tt>, with the call's
 * uncertain inputs after them. Holds the measurement \a M resolved against \a Space.
 */
template <typename Space, typename P, typename M>
class PinnedMeasurement
{
public:

    /** The measurement, resolved against the whole state's space. */
    using ResolvedType = ResolvedMeasurement<M, Space>;

private:

    using Reads = typename ResolvedType::Reads;                         ///< the blocks the model reads
    static constexpr int N = valueDimensionAtCompileTime<Space>;        ///< the whole state's value dimension
    static constexpr int NMax = maxValueDimensionAtCompileTime<Space>;  ///< a bound on it
    static constexpr int D =
        valueDimensionAtCompileTime<DifferenceSpace<Space>>;  ///< the whole state's difference dimension
    static constexpr int DMax = maxValueDimensionAtCompileTime<DifferenceSpace<Space>>;  ///< a bound on it

    /** The given block's difference dimension. */
    static constexpr int PinnedD = valueDimensionAtCompileTime<DifferenceSpace<PathTag<P>>>;

    /** The rest's difference dimension. */
    static constexpr int RestD = D == Eigen::Dynamic || PinnedD == Eigen::Dynamic ? Eigen::Dynamic : D - PinnedD;

    static constexpr int RestDMax = RestD == Eigen::Dynamic ? DMax : RestD;  ///< a bound on it

    /** The given block's value dimension. */
    static constexpr int PinnedN = valueDimensionAtCompileTime<PathTag<P>>;

    /** The rest's value dimension. */
    static constexpr int RestN = N == Eigen::Dynamic || PinnedN == Eigen::Dynamic ? Eigen::Dynamic : N - PinnedN;

    static constexpr int RestNMax = RestN == Eigen::Dynamic ? NMax : RestN;  ///< a bound on it

public:

    /** Resolves \a measurement against \a space and the given block's rows and the rest's in it;
     * the measurement outlives this.
     */
    PinnedMeasurement(Space const & space, M const & measurement)
        : m_resolved{measurement, space}
        , m_rows{blockRows<P>(space)}
        , m_differenceRows{differenceBlockRows<P>(space)}
        , m_size{valueDimension(space)}
        , m_differenceSize{differenceDimension(space)}
    {
        m_rest.resize(m_size - m_rows.size);
        Eigen::Index k = 0;
        for (Eigen::Index i = 0; i < m_size; ++i)
        {
            if (i < m_rows.start || i >= m_rows.start + m_rows.size)
            {
                m_rest(k++) = i;
            }
        }
    }

    /** \returns the measurement, resolved */
    [[nodiscard]] ResolvedType const &
    resolved() const
    {
        return m_resolved;
    }

    /** \returns the number of value coordinates outside the given block */
    [[nodiscard]] Eigen::Index
    restSize() const
    {
        return m_rest.size();
    }

    /** \returns the predicted measurement at the whole state assembled from the rest \a x and
     * the given block \a p, with the uncertain inputs \a inputs passed on. A Jacobian the model's
     * function returns comes back as one block per argument (see ArgumentJacobians): the rest's,
     * the given block's, and each input's (decisions.md [returned-jacobians]).
     */
    template <typename X, typename Pinned, typename... Inputs>
    [[nodiscard]] auto
    operator()(X const & x, Pinned const & p, Inputs const &... inputs) const
    {
        using T = ScalarOf<X>;
        EMRYS_FILTER_ASSERT(x.size() == m_rest.size() && p.size() == m_rows.size && "the blocks do not fit the space");
        auto state = Eigen::Matrix<T, N, 1, Eigen::ColMajor, NMax, 1>{};
        state.resize(m_size);
        state(m_rest) = x;
        middleRowsOf<PinnedN>(state, m_rows.start, m_rows.size) = p.template cast<T>();
        using Returned = std::remove_cvref_t<decltype(m_resolved.evaluateAtState(state, inputs...))>;
        if constexpr (!isWithJacobian<Returned>)
        {
            return m_resolved.evaluateAtState(state, inputs...);
        }
        else
        {
            auto returned = m_resolved.evaluateAtState(state, inputs...);
            return withJacobian(
                std::move(returned.value),
                splitJacobian(returned.jacobian, std::integral_constant<std::size_t, sizeof...(Inputs)>{})
            );
        }
    }

    /// whether the transforms evaluate a batch of points in one call, evaluateBatch() (decisions.md
    /// [batched-functions])
    static constexpr bool EvaluatesBatches = ResolvedType::BatchedFunction;

    /** Writes into \a y the predicted measurement at a batch of points by one call of the batched
     * function: point \a j at the whole state assembled from column \a j of \a xs, the rest, and of
     * \a p, the given block, or \a p itself where it has one column, as a block known exactly does,
     * with column \a j of each of \a inputs
     */
    template <typename Y, typename Xs, typename Pinned, typename... Inputs>
    void
    evaluateBatch(Y & y, Xs const & xs, Pinned const & p, Inputs const &... inputs) const
        requires EvaluatesBatches
    {
        using T = typename Xs::Scalar;
        using States = MatrixOf<T, N, Y::ColsAtCompileTime, NMax, Y::MaxColsAtCompileTime>;
        Eigen::Index const count = xs.cols();
        EMRYS_FILTER_ASSERT(
            xs.rows() == m_rest.size() && p.rows() == m_rows.size && (p.cols() == 1 || p.cols() == count)
            && "the blocks do not fit the space"
        );
        auto states = States{};
        states.resize(m_size, count);
        for (Eigen::Index j = 0; j < count; ++j)
        {
            auto state = states.col(j);
            state(m_rest) = xs.col(j);
            middleRowsOf<PinnedN>(state, m_rows.start, m_rows.size) = p.col(p.cols() == 1 ? 0 : j).template cast<T>();
        }
        m_resolved.evaluateStateColumns(y, states, inputs...);
    }

private:

    /** \returns the Jacobian \a blocks the function returned, one over the whole state's differences
     * and then one per uncertain input, as ArgumentJacobians over <tt>(x, p, inputs...)</tt>: the
     * whole state's split into the rest's columns and the given block's
     */
    template <typename Blocks, std::size_t InputCount>
    [[nodiscard]] auto
    splitJacobian(Blocks const & blocks, std::integral_constant<std::size_t, InputCount>) const
    {
        auto const & whole = jacobianBlock<0>(blocks);
        using Whole = std::remove_cvref_t<decltype(whole)>;
        using Scalar = typename Whole::Scalar;
        constexpr int Rows = Whole::RowsAtCompileTime;
        constexpr int MaxRows = Whole::MaxRowsAtCompileTime;
        using RestJacobian = MatrixOf<Scalar, Rows, RestD, MaxRows, RestDMax>;
        using GivenJacobian =
            MatrixOf<Scalar, Rows, PinnedD, MaxRows, maxValueDimensionAtCompileTime<DifferenceSpace<PathTag<P>>>>;
        Eigen::Index const before = m_differenceRows.start;
        Eigen::Index const after = m_differenceSize - m_differenceRows.start - m_differenceRows.size;
        RestJacobian rest(whole.rows(), before + after);
        rest.leftCols(before) = whole.leftCols(before);
        rest.rightCols(after) = whole.rightCols(after);
        GivenJacobian given = middleColsOf<PinnedD>(whole, m_differenceRows.start, m_differenceRows.size);
        return [&]<std::size_t... K>(std::index_sequence<K...>) {
            using Tuple =
                std::tuple<RestJacobian, GivenJacobian, std::remove_cvref_t<decltype(jacobianBlock<1 + K>(blocks))>...>;
            return ArgumentJacobians<Tuple>{Tuple{std::move(rest), std::move(given), jacobianBlock<1 + K>(blocks)...}};
        }(std::make_index_sequence<InputCount>{});
    }

    ResolvedType m_resolved;              ///< the measurement, resolved
    BlockRows m_rows;                     ///< the given block's rows
    BlockRows m_differenceRows;           ///< the given block's rows among the differences
    Eigen::Index m_size;                  ///< the whole state's value dimension
    Eigen::Index m_differenceSize;        ///< the whole state's difference dimension
    IndexVector<RestN, RestNMax> m_rest;  ///< the rows outside the given block
};

}  // namespace internal

/** \ingroup Models
 * \returns the block named by the path \a P (a tag or a TagPath) of a state in \a space, given
 * as \a value from outside the %Gaussian being updated, for update(prior, measurement, given):
 * a vector when the block is known exactly, or an uncertain input such as a %Gaussian,
 * independent of the rest of the state. The mark copies the space and refers to \a value, so it
 * is valid while \a value is; a temporary value is rejected.
 */
template <typename P, typename Space, typename Value>
[[nodiscard]] GivenBlock<Space, P, Value>
given(Space const & space, Value const & value)
{
    return {space, value};
}

/** \ingroup Models
 * given() of a temporary value is deleted: the mark refers to the value, which would be gone at
 * the end of the full expression, so a mark kept in a variable would dangle. Store the value
 * first.
 */
template <typename P, typename Space, typename Value>
void
given(Space const & space, Value const && value) = delete;

namespace internal
{

/** update(prior, measurement, pinned) with the model over the rest and the given block already
 * resolved as \a f, so that a caller updating many Gaussians, each given its own value of the
 * block, as a Rao-Blackwellized filter does, resolves it once; \a value is the given block's value.
 */
template <bool Statistics, UncertainInput Prior, typename Space, typename P, typename M, typename Value>
[[nodiscard]] auto
updateBranchesWith(Prior const & prior, PinnedMeasurement<Space, P, M> const & f, Value const & value)
{
    auto const & resolved = f.resolved();
    using R = std::remove_cvref_t<decltype(resolved)>;
    static_assert(
        R::HasMethod,
        "a Kalman update pushes the prior through the model by a method: observe(method, model, z)"
    );
    static_assert(!internal::isIterated<typename R::Method>, "an update given a block does not support Iterated yet");
    static_assert(
        !hasConsider<R>,
        "an update given a block, as a Rao-Blackwellized filter's, does not consider blocks yet (decisions.md "
        "[consider-states])"
    );
    EMRYS_FILTER_ASSERT(
        prior.valueDimension() == f.restSize() && "the prior must cover the state outside the given block"
    );
    auto const & method = resolved.method();
    auto const predicted = std::apply(
        [&](auto const &... inputs) {
            return internal::transformPrediction(resolved, method, f, withCross(prior), value, inputs...);
        },
        resolved.inputs()
    );
    // The measurement reads the whole prior, so its linearization is the prediction's own.
    constexpr int N = valueDimensionAtCompileTime<DifferenceSpace<typename Prior::SpaceType>>;
    constexpr int NMax = maxValueDimensionAtCompileTime<DifferenceSpace<typename Prior::SpaceType>>;
    auto const all = internal::consecutiveIndices<N, NMax>(0, prior.differenceDimension());
    using Predicted = std::remove_cvref_t<decltype(predicted)>;
    auto const & cross = predicted.crossCovariance();
    return internal::conditionOnPrediction<Statistics>(
        prior,
        resolved,
        predicted.output(),
        cross,
        internal::ReadLinearization{
            typename Predicted::JacobianMatrix{predicted.jacobian()},
            all,
            [&] {
                using Error = typename Predicted::OutputMatrix;
                return Error{predicted.inputsCovariance() + predicted.unexplainedCovariance()};
            }
        },
        NewPosteriors{}
    );
}

/** Whether \a T is a GivenBlock, which an update taking a policy in its place refuses. */
template <typename T>
inline constexpr bool isGivenBlock = false;

/** A GivenBlock is one. */
template <typename Space, typename P, typename Value>
inline constexpr bool isGivenBlock<GivenBlock<Space, P, Value>> = true;

/** A measurement with a block of the state given, as a step that updates several Gaussians, each
 * given the same value, passes it on (an IMM's modes, given a cell's center): \a F, a
 * PinnedMeasurement or a reference to one, and the block's value.
 */
template <typename F, typename Value>
struct GivenMeasurement
{
    F pinned;             ///< the measurement over the rest and the given block, resolved
    Value const & value;  ///< the given block's value, known or an uncertain input
};

/** A measurement with a block given, as GivenMeasurement holds one: its \c pinned, a
 * PinnedMeasurement, and its \c value.
 */
template <typename G>
concept GivenMeasurementLike = requires(G const & given) {
    given.pinned.resolved();
    given.value;
};

/** update(prior, measurement, pinned), with each branch's NIS and log-likelihood computed with
 * the gain when \a Statistics, for a caller that will read them (decisions.md [innovation-statistics]).
 */
template <bool Statistics, UncertainInput Prior, Measurement M, typename Space, typename P, typename Value>
[[nodiscard]] auto
updateBranches(Prior const & prior, M const & measurement, GivenBlock<Space, P, Value> const & pinned)
{
    auto const f = PinnedMeasurement<Space, P, M>{pinned.space, measurement};
    return updateBranchesWith<Statistics>(prior, f, pinned.value);
}

}  // namespace internal

/** \ingroup Models
 * \returns \a prior, a %Gaussian over the rest of a state, conditioned on \a measurement, with
 * the block that \a pinned names given separately: weighted branches over the rest, or the error,
 * as update(prior, measurement) returns them.
 *
 * The measurement's model reads blocks of the whole state by tag path as usual; the given block
 * comes from \a pinned, and the others from \a prior, whose space is the rest of the state in order.
 * A given value is known exactly, as a grid point or a particle is. A given %Gaussian is an
 * independent uncertain input: its uncertainty widens the predicted measurement, but it is not
 * conditioned on the measurement, so a later update treats it as independent again. This is
 * the update of a Rao-Blackwellized filter's conditional %Gaussian given its point.
 *
 * \code
 * // prior over Composite<Velocity, Bias>, given Position of Composite<Position, Velocity, Bias>
 * auto const updated = update(prior, observe(methods::unscented(), rangePlusBias, z), given<Position>(State{},
 * cellCenter));
 * \endcode
 *
 * \sa given(), RbPointMassFilter
 */
template <UncertainInput Prior, Measurement M, typename Space, typename P, typename Value>
[[nodiscard]] auto
update(Prior const & prior, M const & measurement, GivenBlock<Space, P, Value> const & pinned)
{
    return internal::updateBranches<false>(internal::withCovariance(prior), measurement, pinned);
}

}  // namespace Emrys
