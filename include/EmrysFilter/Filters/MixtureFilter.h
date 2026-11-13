#pragma once

/** \file
 * MixtureFilter (decisions.md [mixture-filter], [filters-and-recursions]): a Mixture state whose components are each
 * predicted and updated by the recursion their type names, a KalmanRecursion for Gaussians and an ImmRecursion for
 * hybrid states, or by one of the user's own (Recursion.h), and reweighted by their likelihoods, with
 * MixtureFilterUpdate; GaussianMixturePredictedJoint, what
 * predictJoint() records for the mixture smoother [smoothing] in GaussianMixtureHistory.h; and GaussianMixtureFilter,
 * the spelling for the mixture of Gaussians. Read ImmFilter.h next.
 */

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief What a mixture filter's update did: each prior component's update before reduction,
 * the observation's log-likelihood under the whole mixture, and whether any component
 * accepted the measurement.
 *
 * \c components[i] is the update of the prior's component \a i by its recursion: a FilterUpdate
 * for a %Gaussian, with its own log-likelihood \f$ \log p(z \mid i) \f$, its noise components'
 * posterior probabilities, and their NIS; an ImmFilterUpdate for a hybrid
 * state. Its \c accepted is false when the recursion, such as a Gate's policy, dropped that
 * component. When \c accepted is false here, no component with a nonzero weight accepted the
 * measurement and the state kept its prior (decisions.md [update-outcomes]). Prior weights that
 * cannot be normalized are an error of the update, not a result. posteriorIndices(i) gives the
 * components of the state that prior component \a i became.
 *
 * The components and the posterior indices come from \a Allocator, the indices reserved by the
 * state's capacity where it has one. Passing the same result, made from the allocator, to each
 * MixtureFilter::update(measurement, policy, result) reuses their storage.
 *
 * \tparam ComponentUpdate  what the recursion's update returns for one component
 * \tparam Scalar           scalar type, \c double by default
 * \tparam Allocator        allocator for the components' updates (see \ref Allocation)
 */
template <typename ComponentUpdate, typename Scalar = double, typename Allocator = DefaultAllocator>
struct MixtureFilterUpdate
{
    using ComponentUpdateType = ComponentUpdate;                          ///< one component's update
    using ComponentsType = internal::Buffer<ComponentUpdate, Allocator>;  ///< the components' container

    MixtureFilterUpdate() = default;  ///< an empty result, from a default-constructed allocator

    /** An empty result whose components come from \a allocator. */
    explicit MixtureFilterUpdate(Allocator const & allocator)
        : components(internal::ReBound<Allocator, ComponentUpdate>(allocator))
        , m_posteriorEnds(internal::ReBound<Allocator, Eigen::Index>(allocator))
    {}

    /** \returns the allocator the components come from */
    [[nodiscard]] Allocator
    get_allocator() const
    {
        return Allocator(components.get_allocator());
    }

    /** \returns \f$ \log p(z) = \log \sum_i w_i \, p(z \mid i) \f$, with the prior weights normalized */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return m_logLikelihood;
    }

    /** \returns the indices in the filter's state() of the components that prior component \a i
     * became, consecutive and in order: none if its policy dropped it, one if the policy reduced
     * it to one component, and one per kept branch if the policy kept several. The components of
     * the priors before \a i come first, so with Collapse and nothing dropped prior \a i is
     * posterior \a i. When \c accepted is false the state kept its prior, and prior \a i is still
     * component \a i. The indices hold until the state next changes, as by a reduction through
     * state(); the state's lineage follows the components through those too (MixtureLineage).
     */
    [[nodiscard]] std::ranges::iota_view<Eigen::Index, Eigen::Index>
    posteriorIndices(Eigen::Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < Eigen::Index(m_posteriorEnds.size()) && "prior component index out of range");
        Eigen::Index const first = i == 0 ? 0 : m_posteriorEnds[std::size_t(i - 1)];
        return std::views::iota(first, m_posteriorEnds[std::size_t(i)]);
    }

    ComponentsType components;  ///< one per prior component, in order
    bool accepted = false;      ///< whether any component accepted the measurement

private:

    friend struct internal::UpdateAccess;

    Scalar m_logLikelihood = 0;  ///< the observation's log-likelihood under the mixture

    /// One past the last posterior index of each prior component, in order
    internal::Buffer<Eigen::Index, Allocator> m_posteriorEnds;
};

/** \ingroup Filters
 * \brief One prediction of a mixture of Gaussians as its smoother needs it: the weights the step
 * started from, each component's prediction as a PredictedJoint, and how the components descend
 * from those of the record before.
 *
 * MixtureFilter::predictJoint() fills it before predicting, and a GaussianMixtureHistory records it
 * (decisions.md [smoothing]). \c joints[n] is component \a n's step, so its \c filtered is the component
 * before the step. \c lineage is the state's lineage at that moment: component \a n's share from
 * each component the state had at the previous predictJoint(), through every update and reduction
 * between the two (see MixtureLineage).
 *
 * The weights, joints, and lineage come from \a Allocator_. Passing the same joint to each
 * MixtureFilter::predictJoint(dt, joint, inputs...) reuses their storage.
 *
 * \tparam Space_      the state space
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the weights, joints, and lineage (see \ref Allocation)
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
struct GaussianMixturePredictedJoint
{
    using Scalar = Scalar_;                                        ///< the scalar type
    using SpaceType = Space_;                                      ///< the state space
    using JointType = PredictedJoint<Space_, Scalar_>;             ///< one component's joint
    using JointsType = internal::Buffer<JointType, Allocator_>;    ///< the components' joints
    using LogWeightsType = internal::Buffer<Scalar_, Allocator_>;  ///< the components' log-weights
    using LineageType = MixtureLineage<Scalar_, Allocator_>;       ///< the components' descent

    LogWeightsType logWeights;  ///< each component's log-weight before the step, as the state had it
    JointsType joints;          ///< one per component, in the state's order
    LineageType lineage;        ///< how the components descend from those of the previous record
};

/** \ingroup Filters
 * \brief A mixture filter: a Mixture state, a process model, and the method that predicts
 * through it, each component run by the recursion its type names, or by the one given.
 *
 * Each component is predicted and updated as a filter over one of it would be, and an update
 * multiplies each component's weight by the observation's likelihood under it,
 * \f$ p(z \mid i) \f$. With %Gaussian components (GaussianMixtureFilter) each is a KalmanFilter's
 * step, and a mixture can carry a multimodal state, such as an ambiguity that later measurements
 * resolve, or many narrow components that follow a nonlinear model more closely than one wide
 * one, as uniformComponents() gives over a box where only the box is known (UniformMixture.h).
 * With hybrid components (a Mixture of HybridGaussian) each is an
 * ImmFilter's step, mixed through its switching model and updated per mode,
 * and reweighted by the IMM's likelihood \f$ \log \sum_m \mu_m \, p(z \mid m) \f$: a mixture of
 * IMM filters (decisions.md [filters-and-recursions]).
 *
 * The %Gaussian sum filter of \ref Alspach1972 "Alspach and Sorenson (1972)", with the component
 * reduction left to the caller (\ref Upkeep); with hybrid components, its components are
 * mode-matched as the %Gaussian mixture filter for jump Markov systems of
 * \ref Pasha2009 "Pasha, Vo, Tuan, and Ma (2009)" carries each target's.
 *
 * The policy reduces each component's branches, one per measurement-noise component. Collapse
 * (the default), Select, and Gate keep one component per component, so the mixture keeps its
 * size; Keep, for %Gaussian components, keeps every branch, so with mixture noise the number of
 * components grows by that factor with each update. Reduce the mixture between steps through
 * state(): pruneBelow(), pruneTail(), pruneToSize(), merge(), or fitMixture(). With %Gaussian components,
 * predictJoint() records each step for a GaussianMixtureHistory to smooth, and the state's
 * lineage (MixtureParams::lineage()) follows the components through the updates and reductions
 * between (decisions.md [smoothing]).
 *
 * Keep gains over Collapse when the components disagree for longer than one update, as about a
 * multimodal state or an outlier too near to tell from an inlier at once. When each fix is plainly
 * one or the other, every component's branches agree on it and Collapse loses little.
 * kullbackLeiblerCollapseBounds() of the state's density says whether its moments describe it
 * (see \ref Divergences), and of a component's branches whether collapsing them would lose much,
 * so a policy of the user's own can keep the branches only where they disagree. A Gate commits:
 * a component it drops is gone, where mixture noise without a gate would only lower its weight,
 * so a bad fix that another component happens to explain removes the right one.
 *
 * Which single estimate to read from the mixture depends on the question. The moments of the
 * state's density, state().toDensity()->moments(), are the minimum mean-squared-error estimate,
 * and can sit between the modes, where no component is. The most probable component,
 * state()[state().mostProbable()].component, is the estimate a display shows of a multimodal
 * state.
 *
 * A component whose update the recursion rejects, as a Gate does when no branch passes, is
 * dropped. When every component is rejected, the measurement is rejected and the state keeps its
 * prior, so with one component the filter behaves as the filter over that component. A component
 * whose update fails, as one whose innovation covariance is not positive definite does, fails the
 * whole update, which returns the error. The components are updated in place, so those before the
 * failed one stay updated, with the weights as they were; nothing is copied to undo a failure
 * (decisions.md [innovation-cholesky]). A policy that keeps several branches builds the posterior
 * apart, and its failure leaves the state as it was.
 *
 * \code
 * auto filter = GaussianMixtureFilter{ConstantVelocity<Position, Velocity>{0.1}, initial};
 * filter.predict(dt);
 * auto const updated = filter.update(fix);  // each component updated and reweighted
 * auto const kept = filter.update(robustFix, reductions::keep());  // an inlier and an outlier branch per component
 * auto const pruned = filter.state().pruneToSize(16);  // the 16 heaviest components kept
 *
 * auto bank = MixtureFilter{switchingModel<Maneuver>(chain, cruise, turn), hybrids};  // a mixture of IMMs
 * bank.predict(dt);
 * auto const robust = bank.update(robustFix);  // noise byMode<Sensor>(): each IMM's Markov outlier axis
 * \endcode
 *
 * The recursion updates the state's components in place. A policy that keeps several branches
 * builds the posterior in a second mixture from the state's allocator, with the state's capacity,
 * exchanged with the state. An update allocates nothing once the mixture has been as large.
 * update(measurement, policy, result) also reuses the result's
 * storage (see \ref Allocation). A recursion with scratch, the IMM's, runs the components in
 * order, so a policy other than Sequential is a compile-time error with hybrid components.
 *
 * A recursion of the user's own, one that meets Recursion over the component, runs on each
 * component in place of the one the component names: <tt>MixtureFilter{recursion, mixture}</tt>
 * deduces it as the last argument (decisions.md [user-extension-points]).
 *
 * \tparam Component_  the components' type (see Component): a Gaussian over the state space (see
 *                     \ref Spaces), or a Hybrid of them
 * \tparam Model_      the process model, a ProcessModel or a FixedStepModel, which
 *                     partition the space, or a SwitchingModel for hybrid components; with a
 *                     FixedStepModel, predict() takes no time step
 * \tparam Method_     the prediction method; a FixedStepModel's own
 * \tparam Allocator_  the state's allocator (see \ref Allocation)
 * \tparam Recursion_  the recursion each component's steps run (see Recursion): the one the component
 *                     names unless given one
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one
 *
 * \sa KalmanFilter, ImmFilter, Recursion, \ref Filters
 */
template <
    Component Component_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator,
    RecursionOver<Component_> Recursion_ = internal::RecursionForType<Component_, Model_, Method_, Allocator_>,
    typename Upkeep_ = upkeep::None>
    requires std::same_as<typename Recursion_::ModelType, Model_>
          && std::same_as<typename Recursion_::MethodType, Method_>
class MixtureFilter
    : public internal::HeldUpkeep<MixtureFilter<Component_, Model_, Method_, Allocator_, Recursion_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<MixtureFilter, Upkeep_>;  ///< what holds the upkeep

    static_assert(
        internal::predictsBy<Model_, Method_>,
        "a filter with a fixed-step model predicts by the model's method"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

public:

    using Scalar = typename Params<Component_>::Scalar;        ///< the scalar type
    using SpaceType = typename Params<Component_>::SpaceType;  ///< the state space
    using ModelType = Model_;                                  ///< the process model
    using MethodType = Method_;                                ///< the prediction method
    using StateType = MixtureParams<Component_, Allocator_>;   ///< the state the filter owns
    using ComponentType = Params<Component_>;                  ///< one component of the state
    using allocator_type = Allocator_;                         ///< the state's allocator

    /** the steps each component runs (see Recursion) */
    using RecursionType = Recursion_;

    /** what predictJoint() returns, for %Gaussian components */
    using JointType = GaussianMixturePredictedJoint<SpaceType, Scalar, Allocator_>;

    /** What update() with \a M fills when given a result: one component's update per component,
     * each with the filter's allocator for what in it is sized by a run-time count.
     */
    template <Measurement M>
    using ResultType = MixtureFilterUpdate<
        internal::RebindUpdateAllocator<typename RecursionType::template UpdateType<M>::value_type, Allocator_>,
        Scalar,
        Allocator_>;

    /** What update() with \a M returns: the ResultType in a \c std::expected (decisions.md [innovation-cholesky]). */
    template <Measurement M>
    using UpdateType = std::expected<ResultType<M>, NumericError>;

    /** A filter with process model \a model, initial state \a state, prediction method \a method,
     * which a fixed-step model carries instead, and upkeep \a upkeep, which maintain() runs. \a state
     * must have at least one component.
     */
    MixtureFilter(
        Model_ model,
        StateType state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : UpkeepBase{std::move(upkeep)}
        , m_state{std::move(state)}
        , m_recursion{makeRecursion(std::move(model), std::move(method), m_state)}
        , m_posterior{m_state.space(), m_state.get_allocator()}
    {
        checkState();
    }

    /** A filter with process model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    MixtureFilter(Model_ model, StateType state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : MixtureFilter{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}

    /** A filter that runs \a recursion, the library's or one of the user's own (see Recursion), on
     * each component of the initial state \a state, which must have at least one, with upkeep
     * \a upkeep, which maintain() runs. A recursion with scratch, as the IMM's, must have been given
     * the components' allocator.
     */
    MixtureFilter(Recursion_ recursion, StateType state, std::type_identity_t<Upkeep_> upkeep = {})
        : UpkeepBase{std::move(upkeep)}
        , m_state{std::move(state)}
        , m_recursion{std::move(recursion)}
        , m_posterior{m_state.space(), m_state.get_allocator()}
    {
        checkState();
    }

    /** \returns the allocator of the state and its posterior */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_state.get_allocator();
    }

    /** \returns the current state estimate */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the current state estimate, to change where it is between steps: to reduce it with
     * pruneBelow(), pruneTail(), pruneToSize(), or merge(), or to assign it a mixture in the same space, as
     * fitMixture()'s. It must keep its space, its allocator, and at least one component; setState()
     * replaces it with another, and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state estimate, as when reinitializing, with one that has at least one
     * component and whose space the process model's writes partition, as the constructor's must.
     * The scratch follows the new state's space, allocator, and capacity. A recursion with scratch
     * is given the new components' allocator through its resetScratch(), as the IMM's is; one
     * without that member must have it already.
     */
    void
    setState(StateType state)
    {
        EMRYS_FILTER_ASSERT(!state.empty() && "the mixture must have a component");
        m_state = std::move(state);
        if (m_posterior.get_allocator() != m_state.get_allocator() || m_posterior.space() != m_state.space()
            || m_posterior.capacity() != m_state.capacity())
        {
            // The assignment carried these in, so the scratch is made again from them.
            m_posterior = StateType{m_state.space(), m_state.get_allocator()};
        }
        if constexpr (requires { m_recursion.resetScratch(componentAllocator()); })
        {
            if (m_recursion.get_allocator() != componentAllocator())
            {
                m_recursion.resetScratch(componentAllocator());
            }
        }
        checkState();
    }

    /** \returns the recursion: the model, method, and any scratch the components' steps run with */
    [[nodiscard]] RecursionType const &
    recursion() const
    {
        return m_recursion;
    }

    /** \returns the process model */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_recursion.model();
    }

    /** \returns the process model, to change its parameters */
    [[nodiscard]] Model_ &
    model()
    {
        return m_recursion.model();
    }

    /** \returns the prediction method */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_recursion.method();
    }

    /** Moves each component forward by \a dt, any TimeStep (see \ref Time), through the process
     * model, with \a inputs passed to its blocks as in Emrys::predict(). The weights are unchanged.
     *
     * \returns nothing for Gaussians and their hybrids; for square-root forms and their hybrids, a
     *          \c std::expected that is empty when a component could not take the step, as its
     *          recursion's predict() reports, a component that could not take it keeps its prior while the others
     *          move (decisions.md [gaussian-forms])
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        return predict(execution::sequential, dt, inputs...);
    }

    /** predict(dt, inputs...) under \a policy, a for-each over the components, so the result is
     * Sequential's (see \ref Execution). A mixture of a few components is best left Sequential,
     * and a recursion with scratch, the IMM's, takes nothing else.
     */
    template <ExecutionPolicy P, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(P && policy, TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        assertRunsUnder<P>();
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the mixture must have a component");
        Scalar const step = internal::toScalarTime<Scalar>(dt);
        return predictComponents(std::forward<P>(policy), [&](ComponentType & component) {
            return m_recursion.predict(component, step, inputs...);
        });
    }

    /** Moves each component forward by the fixed-step model's step, with \a inputs passed as in
     * Emrys::predict(). The weights are unchanged.
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithPolicy<Inputs...> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] auto
    predict(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        return predict(execution::sequential, inputs...);
    }

    /** predict(inputs...) under \a policy, as predict(policy, dt, inputs...) runs. */
    template <ExecutionPolicy P, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] auto
    predict(P && policy, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        assertRunsUnder<P>();
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the mixture must have a component");
        return predictComponents(std::forward<P>(policy), [&](ComponentType & component) {
            return m_recursion.predict(component, inputs...);
        });
    }

    /** Moves each component forward by \a steps of the fixed-step model, as that many calls of
     * predict(inputs...) do, with \a inputs at every step, returning as predict(inputs...) does.
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(Steps steps, Inputs const &... inputs)
    {
        return predict(execution::sequential, steps, inputs...);
    }

    /** predict(steps, inputs...) under \a policy, as that many calls of predict(policy, inputs...)
     * run, stopping at the first that reports an error.
     */
    template <ExecutionPolicy P, typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(P && policy, Steps steps, Inputs const &... inputs)
    {
        EMRYS_FILTER_ASSERT(steps.count >= 0 && "the number of steps must be nonnegative");
        using Result = decltype(predict(policy, inputs...));
        if constexpr (std::is_void_v<Result>)
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                predict(policy, inputs...);
            }
        }
        else
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                if (Result result = predict(policy, inputs...); !result)
                {
                    return result;
                }
            }
            return Result{};
        }
    }

    /** Moves each %Gaussian component forward as predict(dt, inputs...) does, and starts the
     * state's lineage over (MixtureParams::startLineage()).
     *
     * \returns the step as a GaussianMixturePredictedJoint, from the state's allocator, for a
     * GaussianMixtureHistory to record: the weights, each component's PredictedJoint, and the
     * state's lineage before the step
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_> && isGaussian<ComponentType>)
    [[nodiscard]] JointType
    predictJoint(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        auto joint = emptyJoint();
        predictJoint(execution::sequential, dt, joint, inputs...);
        return joint;
    }

    /** As predictJoint(dt, inputs...), but into \a joint, whose storage is reused: with the same
     * joint at every step, recording allocates nothing once it has been as large.
     */
    template <typename JointAllocator, typename... Inputs>
        requires(!isFixedStep<Model_> && isGaussian<ComponentType>)
    void
    predictJoint(
        TimeStep<Scalar> auto const & dt,
        GaussianMixturePredictedJoint<SpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        predictJoint(execution::sequential, dt, joint, inputs...);
    }

    /** predictJoint(dt, joint, inputs...) under \a policy, a for-each over the components into
     * the joint's slots, so the joint and the state are Sequential's (see \ref Execution).
     */
    template <ExecutionPolicy P, typename JointAllocator, typename... Inputs>
        requires(!isFixedStep<Model_> && isGaussian<ComponentType>)
    void
    predictJoint(
        P && policy,
        TimeStep<Scalar> auto const & dt,
        GaussianMixturePredictedJoint<SpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        Scalar const step = internal::toScalarTime<Scalar>(dt);
        recordJoint(std::forward<P>(policy), joint, [&](ComponentType & component) {
            return m_recursion.predictJoint(component, step, inputs...);
        });
    }

    /** Moves each %Gaussian component forward as predict(inputs...) does with a fixed-step model,
     * and starts the state's lineage over.
     *
     * \returns the step as a GaussianMixturePredictedJoint, as predictJoint(dt, inputs...) does
     */
    template <typename... Inputs>
        requires(
            isFixedStep<Model_> && isGaussian<ComponentType> && !internal::leadsWithPolicy<Inputs...>
            && !internal::leadsWithSteps<Inputs...>
        )
    [[nodiscard]] JointType
    predictJoint(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        auto joint = emptyJoint();
        predictJoint(execution::sequential, joint, inputs...);
        return joint;
    }

    /** As predictJoint(inputs...), but into \a joint, whose storage is reused. */
    template <typename JointAllocator, typename... Inputs>
        requires(isFixedStep<Model_> && isGaussian<ComponentType>)
    void
    predictJoint(GaussianMixturePredictedJoint<SpaceType, Scalar, JointAllocator> & joint, Inputs const &... inputs)
    {
        predictJoint(execution::sequential, joint, inputs...);
    }

    /** predictJoint(joint, inputs...) under \a policy, as predictJoint(policy, dt, joint, inputs...)
     * runs.
     */
    template <ExecutionPolicy P, typename JointAllocator, typename... Inputs>
        requires(isFixedStep<Model_> && isGaussian<ComponentType>)
    void
    predictJoint(
        P && policy,
        GaussianMixturePredictedJoint<SpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        recordJoint(std::forward<P>(policy), joint, [&](ComponentType & component) {
            return m_recursion.predictJoint(component, inputs...);
        });
    }

    /** Conditions each component on \a measurement, by the measurement's own method, reduces its
     * branches with \a policy, a MixtureReductionPolicy over the component (Collapse, Select, Keep,
     * or a Gate), and multiplies the weight of what the policy keeps by \f$ p(z \mid i) \f$
     * times its share among the branches: for a Gate the passing branches' share, so that a
     * branch the gate fails counts as impossible. The weights are then normalized, and the
     * observation's log-likelihood counts every branch. Components the recursion rejects are
     * dropped; if it rejects them all, the state is unchanged.
     *
     * \returns each component's update before reduction, the observation's log-likelihood under
     * the mixture, and whether the measurement was accepted (see MixtureFilterUpdate), with the
     * components from the state's allocator; or the WeightError of prior weights that cannot be
     * normalized, with the state unchanged; or, if any component's update failed, as one whose
     * innovation covariance is not positive definite does, its error, with the components before
     * it updated and the weights as they were (see the class comment, and decisions.md
     * [innovation-cholesky], [update-outcomes])
     */
    template <Measurement M, typename Policy = reductions::Collapse>
        requires(
            !isGaussian<ComponentType>
            || MixtureReductionPolicy<Policy, internal::MeasuredUpdate<ComponentType, M>, Component_>
        )
    [[nodiscard]] UpdateType<M>
    update(M const & measurement, Policy const & policy = {})
    {
        return update(execution::sequential, measurement, policy);
    }

    /** update(measurement, policy) under \a executionPolicy: each component's update is a
     * for-each into slots made first, and the reweighting, the evidence, and the posterior then
     * follow in order, so the result is Sequential's (see \ref Execution). The slots allocate.
     */
    template <ExecutionPolicy P, Measurement M, typename Policy = reductions::Collapse>
        requires(
            !isGaussian<ComponentType>
            || MixtureReductionPolicy<Policy, internal::MeasuredUpdate<ComponentType, M>, Component_>
        )
    [[nodiscard]] UpdateType<M>
    update(P && executionPolicy, M const & measurement, Policy const & policy = {})
    {
        auto result = emptyResult<ResultType<M>>();
        auto const updated = update(std::forward<P>(executionPolicy), measurement, policy, result);
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        return result;
    }

    /** As update(measurement, policy), but into \a result, whose storage is reused: with the same
     * result at every step, an update allocates nothing once it has been as large.
     *
     * \returns nothing, or the error update(measurement, policy) would return, with the state
     *          unchanged and \a result unspecified
     */
    template <Measurement M, typename Policy, typename ResultAllocator>
        requires(
            !isGaussian<ComponentType>
            || MixtureReductionPolicy<Policy, internal::MeasuredUpdate<ComponentType, M>, Component_>
        )
    [[nodiscard]] std::expected<void, NumericError>
    update(
        M const & measurement,
        Policy const & policy,
        MixtureFilterUpdate<typename ResultType<M>::ComponentUpdateType, Scalar, ResultAllocator> & result
    )
    {
        return update(execution::sequential, measurement, policy, result);
    }

    /** update(measurement, policy, result) under \a executionPolicy, as
     * update(executionPolicy, measurement, policy) runs.
     */
    template <ExecutionPolicy P, Measurement M, typename Policy, typename ResultAllocator>
        requires(
            !isGaussian<ComponentType>
            || MixtureReductionPolicy<Policy, internal::MeasuredUpdate<ComponentType, M>, Component_>
        )
    [[nodiscard]] std::expected<void, NumericError>
    update(
        P && executionPolicy,
        M const & measurement,
        Policy const & policy,
        MixtureFilterUpdate<typename ResultType<M>::ComponentUpdateType, Scalar, ResultAllocator> & result
    )
    {
        if constexpr (keepsBranches<M, Policy>())
        {
            return updateBranches(std::forward<P>(executionPolicy), measurement, policy, result);
        }
        else
        {
            // The reweighting reads each component's likelihood, so the recursion computes it
            // from the gain's factorization (decisions.md [innovation-statistics]).
            auto const reading = internal::WithStatistics<Policy>{policy};
            return updateComponents(
                std::forward<P>(executionPolicy),
                result,
                [&]<typename Update>(ComponentType & component, Update & slot) {
                    return m_recursion.update(component, measurement, reading, slot);
                },
                [&](ComponentType & component) {
                    return m_recursion.update(component, measurement, reading);
                }
            );
        }
    }

private:

    /** Checks the initial state and gives the posterior scratch its room: the body of every
     * constructor.
     */
    void
    checkState()
    {
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the initial mixture must have a component");
        internal::assertPartitions(m_recursion.model(), m_state.space());
        if (auto const capacity = m_state.capacity())
        {
            m_posterior.setCapacity(*capacity);
        }
        else
        {
            m_posterior.reserve(m_state.size());
        }
    }

    /** A compile-time error unless the recursion's steps may run at once or \a P is Sequential. */
    template <typename P>
    static constexpr void
    assertRunsUnder()
    {
        static_assert(
            internal::isSequential<P> || RecursionType::Reentrant,
            "a recursion with scratch, such as the IMM's, runs the components in order: pass Sequential"
        );
    }

    /** \returns the recursion for \a model and \a method: with the components' allocator when it
     * keeps scratch sized by their elements, as the IMM's does
     */
    static RecursionType
    makeRecursion(Model_ model, MethodFor<Model_, Method_> method, StateType const & state)
    {
        if constexpr (std::is_constructible_v<
                          RecursionType,
                          Model_,
                          MethodFor<Model_, Method_>,
                          decltype(componentAllocator(state))>)
        {
            return RecursionType{std::move(model), std::move(method), componentAllocator(state)};
        }
        else
        {
            return RecursionType{std::move(model), std::move(method)};
        }
    }

    /** \returns the allocator the components' own storage comes from, for a hybrid component, or
     * the mixture's for a component that has none
     */
    [[nodiscard]] static auto
    componentAllocator(StateType const & state)
    {
        if constexpr (requires(ComponentType const & c) { c.get_allocator(); })
        {
            EMRYS_FILTER_ASSERT(!state.empty() && "the initial mixture must have a component");
            return state.begin()->component.get_allocator();
        }
        else
        {
            return state.get_allocator();
        }
    }

    /** componentAllocator() of the state. */
    [[nodiscard]] auto
    componentAllocator() const
    {
        return componentAllocator(m_state);
    }

    /** Whether \a Policy may keep several of a %Gaussian component's branches, as Keep does, so
     * that one prior component becomes several posterior ones and the update goes through the
     * branches rather than through the recursion: any policy but one that returns the component it
     * keeps, or a Gate over one (see internal::keepsOne).
     */
    template <Measurement M, typename Policy>
    [[nodiscard]] static consteval bool
    keepsBranches()
    {
        if constexpr (isGaussian<ComponentType>)
        {
            using Measured = typename decltype(internal::updateBranches<true>(
                std::declval<ComponentType const &>(),
                std::declval<M const &>()
            ))::value_type;
            return !internal::keepsOne<Policy, typename Measured::Container, ComponentType>;
        }
        else
        {
            return false;
        }
    }

    /** Moves each component in place by \a predictOne, which returns what the recursion's
     * predict() does, under \a policy: the body of every predict().
     *
     * \returns nothing, or, when \a predictOne returns a \c std::expected, the error of a component
     *          that could not take the step, which keeps its prior; under a parallel policy, which
     *          one's error is not fixed
     */
    template <ExecutionPolicy P, typename PredictOne>
    [[nodiscard]] auto
    predictComponents(P && policy, PredictOne const & predictOne)
    {
        using Moved = std::invoke_result_t<PredictOne const &, ComponentType &>;
        if constexpr (std::is_void_v<Moved>)
        {
            m_state.forEach(std::forward<P>(policy), [&](auto & entry) {
                entry.updateComponent(predictOne);
            });
        }
        else
        {
            // The first error any component reports, written once under any policy.
            auto failed = std::atomic_flag{};
            auto error = typename Moved::error_type{};
            m_state.forEach(std::forward<P>(policy), [&](auto & entry) {
                Moved const moved = entry.updateComponent(predictOne);
                if (!moved && !failed.test_and_set())
                {
                    error = moved.error();
                }
            });
            return failed.test() ? Moved{std::unexpected{error}} : Moved{};
        }
    }

    /** \returns an empty joint from the state's allocator */
    [[nodiscard]] JointType
    emptyJoint() const
    {
        return {
            internal::bufferOf<Scalar>(get_allocator()),
            internal::bufferOf<PredictedJoint<SpaceType, Scalar>>(get_allocator()),
            typename JointType::LineageType(get_allocator())
        };
    }

    /** Fills \a joint with the state's weights and lineage, then moves each component by
     * \a predictOne, which returns its PredictedJoint, into the joint's slot for it, and starts the
     * state's lineage over. The body of every predictJoint().
     */
    template <ExecutionPolicy P, typename JointAllocator, typename PredictOne>
    void
    recordJoint(
        P && policy,
        GaussianMixturePredictedJoint<SpaceType, Scalar, JointAllocator> & joint,
        PredictOne const & predictOne
    )
    {
        assertRunsUnder<P>();
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the mixture must have a component");
        std::size_t const count = std::size_t(m_state.size());
        internal::MixtureLineageAccess::assign(joint.lineage, m_state.lineage());
        joint.logWeights.clear();
        for (auto const & entry : m_state)
        {
            joint.logWeights.push_back(entry.logWeight);
        }
        joint.joints.resize(count);
        m_state.forEach(std::forward<P>(policy), [&](auto & entry) {
            joint.joints[std::size_t(entry.index())] = entry.updateComponent([&](ComponentType & component) {
                return predictOne(component);
            });
        });
        m_state.startLineage();
    }

    /** \returns an empty \a Result with room for every component, from the state's allocator */
    template <typename Result>
    [[nodiscard]] Result
    emptyResult() const
    {
        auto result = Result{get_allocator()};
        result.components.reserve(std::size_t(m_state.size()));
        internal::UpdateAccess::posteriorEnds(result).reserve(std::size_t(m_state.size()));
        return result;
    }

    /** \returns the posterior ends of \a result, one per component of the state, reserved to the
     * state's capacity where it has one, so that a result reused under a capacity grows only once
     * (decisions.md [allocation])
     */
    template <typename Result>
    [[nodiscard]] auto &
    posteriorEndsFor(Result & result) const
    {
        auto & ends = internal::UpdateAccess::posteriorEnds(result);
        if (auto const capacity = m_state.capacity())
        {
            ends.reserve(std::size_t(*capacity));
        }
        ends.resize(std::size_t(m_state.size()));
        return ends;
    }

    /** Sets the posterior ends of \a result for a rejected measurement, after which the state is
     * its prior and each prior component is where it was.
     */
    template <typename Result>
    void
    keepPriorIndices(Result & result) const
    {
        auto & ends = internal::UpdateAccess::posteriorEnds(result);
        std::iota(ends.begin(), ends.end(), Eigen::Index{1});
    }

    /** The update through the recursion: the recursion runs on each component of the state in
     * place, by \a updateInto into the slot of \a result the component had at the last update or
     * by \a updateNew into a new one, and the components are reweighted by their likelihoods, in
     * order, with the rejected ones dropped. A component the recursion rejects keeps its prior, so
     * a rejected measurement leaves the state as it was; an update that fails returns its error
     * with the components before it updated and the weights as they were, since nothing is copied
     * to undo it (decisions.md [innovation-cholesky]). A component is reweighted by the likelihood of the branches its
     * policy kept, its update's log-likelihood plus the kept weight the recursion recorded in it
     * (see logKeptWeightOf()); the evidence counts every branch.
     *
     * \returns nothing, or the error of the first component whose update failed
     */
    template <ExecutionPolicy P, typename Result, typename UpdateInto, typename UpdateNew>
    [[nodiscard]] std::expected<void, NumericError>
    updateComponents(P && executionPolicy, Result & result, UpdateInto const & updateInto, UpdateNew const & updateNew)
    {
        assertRunsUnder<P>();
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the mixture must have a component");
        // Weights the caller left undefined fail the update before anything is written.
        auto const logTotal = m_state.logTotalWeight();
        if (!logTotal)
        {
            return std::unexpected{logTotal.error()};
        }
        Eigen::Index const count = m_state.size();
        using Update = typename Result::ComponentUpdateType;

        if constexpr (internal::isSequential<P>)
        {
            // The slots a result holds from the last update are reused, with the storage a
            // recursion's result keeps; the components beyond them get new ones, from the filter's
            // allocator.
            Eigen::Index const held = std::min(Eigen::Index(result.components.size()), count);
            result.components.erase(result.components.begin() + held, result.components.end());
            auto failure = std::optional<GaussianError>{};
            m_state.forEach(execution::sequential, [&](auto & entry) {
                if (failure)
                {
                    return;
                }
                auto const updated = [&]() -> std::expected<void, NumericError> {
                    if (entry.index() < held)
                    {
                        return entry.updateComponent([&](ComponentType & component) {
                            return updateInto(component, result.components[std::size_t(entry.index())]);
                        });
                    }
                    else if constexpr (internal::EmptyFromAllocator<Update, Allocator_>)
                    {
                        result.components.push_back(internal::emptyUpdate<Update>(get_allocator()));
                        return entry.updateComponent([&](ComponentType & component) {
                            return updateInto(component, result.components.back());
                        });
                    }
                    else
                    {
                        auto update = entry.updateComponent([&](ComponentType & component) {
                            return updateNew(component);
                        });
                        if (!update)
                        {
                            return std::unexpected{update.error()};
                        }
                        result.components.push_back(std::move(*update));
                        return {};
                    }
                }();
                if (!updated)
                {
                    failure = updated.error();
                }
            });
            if (failure)
            {
                return std::unexpected{*failure};
            }
        }
        else
        {
            using Updated = std::remove_cvref_t<decltype(updateNew(std::declval<ComponentType &>()))>;
            auto slots = internal::bufferOf<std::optional<Updated>>(get_allocator());
            slots.resize(std::size_t(count));
            m_state.forEach(std::forward<P>(executionPolicy), [&](auto & entry) {
                slots[std::size_t(entry.index())].emplace(entry.updateComponent([&](ComponentType & component) {
                    return updateNew(component);
                }));
            });
            result.components.clear();
            for (auto & slot : slots)
            {
                if (!*slot)
                {
                    return std::unexpected{slot->error()};
                }
                result.components.push_back(std::move(**slot));
            }
        }

        // Component i's weight times its likelihood, and that of the branches its policy kept.
        auto const logWeightOf = [&](Eigen::Index i) {
            return m_state[i].logWeight - *logTotal + result.components[std::size_t(i)].logLikelihood();
        };
        auto const logKeptOf = [&](Eigen::Index i) {
            return logWeightOf(i) + logKeptWeightOf(result.components[std::size_t(i)]);
        };
        // Whether the measurement is accepted is known before a weight is written: some component
        // accepted it and keeps a nonzero weight (decisions.md [update-outcomes]).
        auto evidence = LogSumExp<Scalar>{};
        bool accepted = false;
        for (Eigen::Index i = 0; i < count; ++i)
        {
            evidence.add(logWeightOf(i));
            accepted =
                accepted
                || (result.components[std::size_t(i)].accepted && logKeptOf(i) > -Eigen::NumTraits<Scalar>::infinity());
        }
        internal::UpdateAccess::logLikelihood(result) = evidence.value();
        result.accepted = accepted;
        auto & ends = posteriorEndsFor(result);
        if (!accepted)
        {
            // Only components of weight zero accepted it, if any, so the mixture is as it was.
            for (auto & component : result.components)
            {
                component.accepted = false;
            }
            keepPriorIndices(result);
            return {};
        }
        // The accepted components keep their order, so prior i ends where the count of those up
        // to it does. The scan starts from an Index: without a start, libc++ sums in the
        // components' own type.
        std::transform_inclusive_scan(
            result.components.begin(),
            result.components.end(),
            ends.begin(),
            std::plus<>{},
            [](auto const & component) {
                return Eigen::Index(component.accepted);
            },
            Eigen::Index{0}
        );
        for (Eigen::Index i = 0; i < count; ++i)
        {
            if (result.components[std::size_t(i)].accepted)
            {
                m_state.setLogWeight(i, logKeptOf(i));
            }
        }
        m_state.removeIf([&](Eigen::Index i, auto const &) {
            return !result.components[std::size_t(i)].accepted;
        });
        [[maybe_unused]] auto const normalized = m_state.normalize();
        EMRYS_FILTER_ASSERT(normalized && "an accepted component keeps a nonzero weight");
        return {};
    }

    /** The update through the branches, for %Gaussian components and a policy that may keep
     * several of them: the policy adds what it keeps of each component's branches to a
     * PosteriorOutput, which places each in the posterior with the component's weight times its
     * likelihood. The slots of \a result are reused as updateComponents() reuses them, each
     * measurement's branches filled in its slot's scratch.
     *
     * \returns nothing, or the error of the first component whose update failed, with the state
     *          unchanged (decisions.md [innovation-cholesky])
     */
    template <ExecutionPolicy P, Measurement M, typename Policy, typename Result>
    [[nodiscard]] std::expected<void, NumericError>
    updateBranches(P && executionPolicy, M const & measurement, Policy const & policy, Result & result)
    {
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the mixture must have a component");
        auto const logTotal = m_state.logTotalWeight();
        if (!logTotal)
        {
            return std::unexpected{logTotal.error()};
        }
        Eigen::Index const count = m_state.size();
        using Update = typename Result::ComponentUpdateType;

        m_posterior.clear();
        internal::MixtureLineageAccess::startFrom(m_posterior, m_state);
        auto & ends = posteriorEndsFor(result);
        auto evidence = LogSumExp<Scalar>{};
        // Reduces component i's measured branches into the posterior, each kept one weighted by the
        // likelihood of the branches it stands for; the evidence counts them all.
        // \returns whether any was kept
        auto const reduce = [&](Eigen::Index i, auto & measured) {
            Scalar const logWeight = m_state[i].logWeight - *logTotal + measured.logLikelihood();
            evidence.add(logWeight);
            auto out = PosteriorOutput{*this, i, logWeight};
            internal::reduceInto(policy, std::move(measured.branches), out);
            // The components are reduced in order, each into the end of the posterior.
            ends[std::size_t(i)] = m_posterior.size();
            return out.added;
        };
        if constexpr (internal::isSequential<P>)
        {
            Eigen::Index const held = std::min(Eigen::Index(result.components.size()), count);
            result.components.erase(result.components.begin() + held, result.components.end());
            for (Eigen::Index i = 0; i < count; ++i)
            {
                // The reweighting reads each component's likelihood, so the update computes it with its gain.
                auto const updated = internal::updateSlot<true>(
                    result.components,
                    std::size_t(i),
                    get_allocator(),
                    m_state[i].component,
                    measurement,
                    [&](auto & measured) {
                        return reduce(i, measured);
                    }
                );
                if (!updated)
                {
                    return updated;
                }
            }
        }
        else
        {
            using Measured =
                decltype(internal::updateBranches<true>(std::declval<ComponentType const &>(), measurement));
            auto measuredAll = internal::bufferOf<std::optional<Measured>>(get_allocator());
            measuredAll.resize(std::size_t(count));
            internal::forEachIndex(std::forward<P>(executionPolicy), count, [&](Eigen::Index i) {
                measuredAll[std::size_t(i)].emplace(internal::updateBranches<true>(m_state[i].component, measurement));
            });
            for (auto const & measured : measuredAll)
            {
                if (!*measured)
                {
                    return std::unexpected{measured->error()};
                }
            }
            result.components.clear();
            for (Eigen::Index i = 0; i < count; ++i)
            {
                auto & measured = **measuredAll[std::size_t(i)];
                bool const accepted = reduce(i, measured);
                if constexpr (internal::EmptyFromAllocator<Update, Allocator_>)
                {
                    result.components.push_back(internal::emptyUpdate<Update>(get_allocator()));
                    internal::filterUpdateInto(measured, accepted, result.components.back());
                }
                else
                {
                    result.components.push_back(internal::filterUpdate(std::move(measured), accepted));
                }
            }
        }
        internal::UpdateAccess::logLikelihood(result) = evidence.value();
        // The posterior is built apart, so a measurement that only components of weight zero accept
        // leaves the state as it was (decisions.md [update-outcomes]).
        result.accepted = !m_posterior.empty() && m_posterior.normalize().has_value();
        if (!result.accepted)
        {
            for (auto & component : result.components)
            {
                component.accepted = false;
            }
            keepPriorIndices(result);
            return {};
        }
        m_state.swap(m_posterior);
        return {};
    }

    /** The reduction output of the update through the branches (see ReductionPolicy): each
     * component a policy adds of the state's component \c prior joins the posterior with \c logWeight,
     * the prior's weight times its likelihood, times the component's own weight among the branches,
     * and with the prior's lineage.
     */
    struct PosteriorOutput
    {
        using ComponentType = Params<Component_>;   ///< the component a policy adds
        static constexpr bool HoldsSeveral = true;  ///< a policy may add any number

        MixtureFilter & filter;  ///< the filter whose posterior is filled
        Eigen::Index prior;      ///< the state's component the branches came from
        Scalar logWeight;        ///< its normalized log-weight plus its update's log-likelihood
        bool added = false;      ///< whether the policy added a component

        /** Adds \a component, moved, with log-weight \a weight among the branches. */
        void
        add(Scalar weight, ComponentType && component)
        {
            filter.m_posterior.add(logWeight + weight, std::move(component));
            internal::MixtureLineageAccess::inherit(filter.m_posterior, filter.m_state, prior);
            added = true;
        }

        /** Adds a copy of \a component with log-weight \a weight among the branches. */
        void
        add(Scalar weight, ComponentType const & component)
        {
            add(weight, ComponentType(component));
        }
    };

    /** \returns the log of the posterior probability of the branches the component \a update
     * kept stands for, as its recursion recorded it, or zero for an update that records none, such
     * as an IMM's
     */
    template <typename Update>
    [[nodiscard]] static Scalar
    logKeptWeightOf(Update const & update)
    {
        if constexpr (internal::isFilterUpdate<Update>)
        {
            return Scalar(internal::MixtureEvidenceAccess::logKeptWeight(update));
        }
        else
        {
            return Scalar(0);
        }
    }

    StateType m_state;          ///< the state estimate
    RecursionType m_recursion;  ///< the model, method, and any scratch the components' steps run with
    StateType m_posterior;      ///< scratch for an update that keeps branches, exchanged with the state
};

/// \cond DEDUCTION_GUIDES
template <Component C, typename Model, typename Method = DefaultMethodFor<Model>, typename Allocator>
    requires(!Recursion<Model> && !internal::UpkeepArgument<Method>)
MixtureFilter(Model, MixtureParams<C, Allocator>, Method = {}) -> MixtureFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Method,
    Allocator>;

template <Component C, RecursionOver<C> R, typename Allocator>
MixtureFilter(R, MixtureParams<C, Allocator>)
    -> MixtureFilter<C, typename R::ModelType, typename R::MethodType, Allocator, R>;

template <Component C, typename Model, typename Allocator, typename U>
    requires(!Recursion<Model> && internal::UpkeepArgument<U>)
MixtureFilter(Model, MixtureParams<C, Allocator>, U) -> MixtureFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    DefaultMethodFor<Model>,
    Allocator,
    internal::RecursionForType<
        C,
        internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
        DefaultMethodFor<Model>,
        Allocator>,
    U>;

template <Component C, typename Model, typename Method, typename Allocator, typename U>
    requires(!Recursion<Model>)
MixtureFilter(Model, MixtureParams<C, Allocator>, Method, U) -> MixtureFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Method,
    Allocator,
    internal::RecursionForType<
        C,
        internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
        Method,
        Allocator>,
    U>;

template <Component C, RecursionOver<C> R, typename Allocator, typename U>
    requires internal::UpkeepArgument<U>
MixtureFilter(R, MixtureParams<C, Allocator>, U)
    -> MixtureFilter<C, typename R::ModelType, typename R::MethodType, Allocator, R, U>;

/// \endcond

/** \ingroup Filters
 * \brief The MixtureFilter over a GaussianMixtureParams, the spelling for the common case
 * (decisions.md [filters-and-recursions]): each component a KalmanFilter's step, reweighted by its likelihood. Its
 * arguments are deduced the same way: <tt>GaussianMixtureFilter filter{model, initial}</tt>.
 *
 * It is the MixtureFilter it names with nothing added, as a derived class rather than an alias so
 * that its arguments deduce on every compiler: MSVC does not deduce through an alias whose
 * parameters sit inside a nested template argument, here \c GaussianParams<Space_, Scalar_> [filters-and-recursions].
 * It runs the KalmanRecursion; a filter with a recursion of the user's own is a MixtureFilter.
 *
 * \tparam Space_      the state space (see \ref Spaces)
 * \tparam Model_      the process model, a ProcessModel or a FixedStepModel
 * \tparam Method_     the prediction method; a FixedStepModel's own
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  the state's allocator (see \ref Allocation)
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one
 */
template <
    Space Space_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    RealScalar Scalar_ = double,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class GaussianMixtureFilter
    : public MixtureFilter<
          Gaussian<Space_, Scalar_>,
          Model_,
          Method_,
          Allocator_,
          internal::RecursionForType<Gaussian<Space_, Scalar_>, Model_, Method_, Allocator_>,
          Upkeep_>
{
public:

    /// the filter this is
    using Base = MixtureFilter<
        Gaussian<Space_, Scalar_>,
        Model_,
        Method_,
        Allocator_,
        internal::RecursionForType<Gaussian<Space_, Scalar_>, Model_, Method_, Allocator_>,
        Upkeep_>;

    /** The filter MixtureFilter's constructor makes from \a args. It forwards rather than
     * inheriting the constructors, since MSVC 19.51 crashes deducing the class's arguments when
     * C++23 lets the inherited constructors take part; the deduction guides below deduce them.
     */
    template <typename... Args>
        requires(
            std::constructible_from<Base, Args...>
            && !(sizeof...(Args) == 1 && (std::is_same_v<std::remove_cvref_t<Args>, GaussianMixtureFilter> && ...))
        )
    GaussianMixtureFilter(Args &&... args)
        : Base(std::forward<Args>(args)...)
    {}
};

/// \cond DEDUCTION_GUIDES
// A guide with a defaulted method, its parameters in the class's order, and two with an upkeep.
template <
    typename Space,
    typename Model,
    typename Method = DefaultMethodFor<Model>,
    typename Scalar,
    typename Allocator>
    requires(!internal::UpkeepArgument<Method>)
GaussianMixtureFilter(Model, MixtureParams<Gaussian<Space, Scalar>, Allocator>, Method = {})
    -> GaussianMixtureFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar, Allocator>;

template <typename Space, typename Model, typename Scalar, typename Allocator, typename U>
    requires internal::UpkeepArgument<U>
GaussianMixtureFilter(Model, MixtureParams<Gaussian<Space, Scalar>, Allocator>, U) -> GaussianMixtureFilter<
    Space,
    internal::BoundStep<Model, Space, Scalar>,
    DefaultMethodFor<Model>,
    Scalar,
    Allocator,
    U>;

template <typename Space, typename Model, typename Method, typename Scalar, typename Allocator, typename U>
GaussianMixtureFilter(Model, MixtureParams<Gaussian<Space, Scalar>, Allocator>, Method, U)
    -> GaussianMixtureFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar, Allocator, U>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <
    typename Component_,
    typename Model_,
    typename Method_,
    typename Allocator_,
    typename Recursion_,
    typename Upkeep_>
void
describe(
    std::ostream & stream,
    MixtureFilter<Component_, Model_, Method_, Allocator_, Recursion_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "mixture filter", filter.state(), indent);
}

/** Describes \a filter: its name, and its state nested under it. */
template <typename Space_, typename Model_, typename Method_, typename Scalar_, typename Allocator_, typename Upkeep_>
void
describe(
    std::ostream & stream,
    GaussianMixtureFilter<Space_, Model_, Method_, Scalar_, Allocator_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "Gaussian mixture filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
