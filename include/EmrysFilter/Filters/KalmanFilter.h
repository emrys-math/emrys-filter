#pragma once

/** \file
 * KalmanRecursion and KalmanFilter (decisions.md [templated-measurements] and [filters-and-recursions]): the Kalman
 * steps over one component as a recursion, a process model and a prediction method acting on a state passed in, and the
 * filter that owns a component and runs that recursion on it, with updates by any measurement reduced to one component
 * by a policy. The shape every other filter follows, and internal::RecursionFor, which names the recursion a filter
 * over a component runs unless it is given one of the user's own (Recursion.h). Its predictJoint() records a step
 * for the smoother in GaussianHistory.h [smoothing]. Over a
 * square-root or information form of the %Gaussian the same recursion runs that form's steps (FormSteps.h,
 * [gaussian-forms]), which InfoFilter.h, SqrtKalmanFilter.h, and SqrtInfoFilter.h spell. Read MixtureFilter.h next.
 */

#include <concepts>
#include <expected>
#include <optional>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Models/FormSteps.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Core/Print.h>

namespace Emrys
{

namespace internal
{

/** Whether \a C is a Component or evaluates a GaussianForm: what KalmanState says, as a variable
 * so that the concept's definition is one name.
 */
template <typename C>
inline constexpr bool isKalmanState = Component<C> || (GaussianForm<Params<C>> && std::same_as<Density<Params<C>>, C>);

}  // namespace internal

/** \ingroup Filters
 * What a KalmanFilter holds and a KalmanRecursion acts on: a Component, such as a Gaussian, or a
 * square-root or information form of the %Gaussian (see GaussianForm, decisions.md [gaussian-forms]), whose
 * steps FormSteps.h supplies.
 */
template <typename C>
concept KalmanState = requires { requires internal::isKalmanState<C>; };

namespace internal
{

/** \returns the type a KalmanRecursion's update() of \a C by \a M returns, as a type_identity: the
 * FilterUpdate of the covariance form's branches, or what the form's updateInPlace() returns, in
 * a \c std::expected (decisions.md [innovation-cholesky])
 */
template <typename C, typename M>
[[nodiscard]] consteval auto
kalmanUpdateType()
{
    if constexpr (isOtherGaussianForm<C>)
    {
        return std::type_identity<decltype(updateInPlace<false>(std::declval<C &>(), std::declval<M const &>()))>{};
    }
    else
    {
        using Measured =
            typename decltype(Emrys::update(std::declval<C const &>(), std::declval<M const &>()))::value_type;
        return std::type_identity<std::expected<
            FilterUpdateWithScratch<
                typename C::SpaceType,
                typename Measured::MeasurementSpaceType,
                Measured::BranchCount,
                typename C::Scalar,
                DefaultAllocator>,
            GaussianError>>{};
    }
}

}  // namespace internal

/** \ingroup Filters
 * \brief The Kalman filter's steps over one component, without the state: a process model and a
 * prediction method, acting on a component passed in (decisions.md [filters-and-recursions]).
 *
 * KalmanFilter owns a component and runs this on it; MixtureFilter runs it on each %Gaussian of a
 * mixture, and ImmFilter mixes around it. It holds no scratch, so its steps
 * on different components may run at once (\c Reentrant).
 *
 * The filter of \ref Kalman1960 "Kalman (1960)"; with the Linearized method it is the extended
 * Kalman filter and with Unscented the unscented Kalman filter of
 * \ref Julier2004 "Julier and Uhlmann (2004)" (\ref Sarkka2013 "Särkkä (2013)", chapters 4 to
 * 6). Mixture noise with Collapse is the probabilistic data association filter of
 * \ref BarShalom1975 "Bar-Shalom and Tse (1975)".
 *
 * update() returns a \c std::expected, empty when the innovation covariance is not positive
 * definite, with the state unchanged (decisions.md [innovation-cholesky]). An update of one branch
 * under Collapse, Select, or a Gate over them conditions the state where it is, the gate deciding on
 * the NIS before the state is written, so a rejected measurement copies nothing [update-outcomes].
 * Over a square-root or information form of the %Gaussian [gaussian-forms] the steps are the form's
 * own (FormSteps.h): predict() then returns a \c std::expected that says whether the form could take
 * the step, update() takes measurement noise of one component and those policies alone, and there is
 * no predictJoint(), since the smoother's history holds the covariance form.
 *
 * \tparam Component_  the component the steps act on (see Component), such as a Gaussian, or a
 *                     form of the %Gaussian (see GaussianForm)
 * \tparam Model_      the process model, a ProcessModel or a FixedStepModel, which
 *                     partition the component's space
 * \tparam Method_     the prediction method; a FixedStepModel's own
 *
 * \sa KalmanFilter
 */
template <KalmanState Component_, typename Model_, typename Method_ = DefaultMethodFor<Model_>>
class KalmanRecursion
{
    static_assert(
        internal::predictsBy<Model_, Method_>,
        "a filter with a fixed-step model predicts by the model's method"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

    /// whether the state is a square-root or information form, whose steps are FormSteps.h's
    static constexpr bool IsForm = internal::isOtherGaussianForm<Params<Component_>>;

public:

    using Scalar = typename Params<Component_>::Scalar;        ///< the scalar type
    using SpaceType = typename Params<Component_>::SpaceType;  ///< the state space
    using ModelType = Model_;                                  ///< the process model
    using MethodType = Method_;                                ///< the prediction method
    using StateType = Params<Component_>;                      ///< the state the steps act on
    static constexpr bool Reentrant = true;                    ///< the steps hold no scratch

    /** What update() with \a M returns: a FilterUpdate in a \c std::expected (decisions.md [innovation-cholesky]). */
    template <Measurement M>
    using UpdateType = typename decltype(internal::kalmanUpdateType<Params<Component_>, M>())::type;

    /** What update() with \a M fills when given a result: the FilterUpdate of UpdateType. */
    template <Measurement M>
    using ResultType = typename UpdateType<M>::value_type;

    /** The recursion with process model \a model and prediction method \a method; a fixed-step
     * model carries its own, so there is none to give.
     */
    explicit KalmanRecursion(Model_ model, MethodFor<Model_, Method_> method = {})
        : m_model{std::move(model)}
        , m_method{internal::predictionMethod(m_model, std::move(method))}
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

    /** \returns whether the model's writes partition \a space, which a state must be in */
    [[nodiscard]] bool
    partitions(SpaceType const & space) const
    {
        return m_model.partitions(space);
    }

    /** Moves \a state forward by \a dt, any TimeStep (see \ref Time), through the process model,
     * with \a inputs passed to its model as in Emrys::predict(): an uncertain input carries its
     * uncertainty through the dynamics, and anything else is known.
     *
     * \returns nothing for a Gaussian; for a form of the %Gaussian, a \c std::expected that is
     *          empty when the form could not take the step (decisions.md [gaussian-forms]), such as an
     *          information form with no information asked to predict through its moments
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(Params<Component_> & state, TimeStep<Scalar> auto const & dt, Inputs const &... inputs) const
    {
        return internal::predictInPlace(m_method, state, m_model, internal::toScalarTime<Scalar>(dt), inputs...);
    }

    /** Moves \a state forward by the fixed-step model's step, with \a inputs passed as in
     * Emrys::predict(), returning as predict(state, dt, inputs...) does.
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(Params<Component_> & state, Inputs const &... inputs) const
    {
        return internal::predictInPlace(state, m_model, inputs...);
    }

    /** Moves \a state forward as predict(state, dt, inputs...) does.
     *
     * \returns the step as a PredictedJoint: the state before it, the state after it, and their
     * cross-covariance, for a GaussianHistory to record
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_> && !IsForm)
    [[nodiscard]] PredictedJoint<SpaceType, Scalar>
    predictJoint(Params<Component_> & state, TimeStep<Scalar> auto const & dt, Inputs const &... inputs) const
    {
        auto joint = Emrys::predictJoint(m_method, state, m_model, dt, inputs...);
        state = joint.predicted;
        return joint;
    }

    /** Moves \a state forward as predict(state, inputs...) does with a fixed-step model.
     *
     * \returns the step as a PredictedJoint, as predictJoint(state, dt, inputs...) does
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !IsForm)
    [[nodiscard]] PredictedJoint<SpaceType, Scalar>
    predictJoint(Params<Component_> & state, Inputs const &... inputs) const
    {
        auto joint = Emrys::predictJoint(state, m_model, inputs...);
        state = joint.predicted;
        return joint;
    }

    /** Conditions \a state on \a measurement, by the measurement's own method, and reduces the
     * branches to the new state with \a policy, a ReductionPolicy to the component (Collapse,
     * Select, or a Gate). If the policy keeps nothing, it rejects the measurement and the state is
     * unchanged.
     *
     * \returns the update before reduction, without its posteriors, and whether it was accepted
     * (see FilterUpdate): each noise component's posterior probability, such as the probability
     * that the measurement is an outlier, and its innovation, from which the observation's
     * log-likelihood and the NIS follow when asked for (decisions.md [innovation-statistics]). The update is in a
     * \c std::expected, empty, with the state unchanged, when an innovation covariance is not
     * positive definite [innovation-cholesky] or a form could not take the update [gaussian-forms].
     */
    template <Measurement M, typename Policy = reductions::Collapse>
        requires(
            (IsForm || ReductionPolicy<Policy, internal::MeasuredUpdate<Params<Component_>, M>, Params<Component_>>)
            && !internal::isLinearizedAbout<Policy>
        )
    [[nodiscard]] UpdateType<M>
    update(Params<Component_> & state, M const & measurement, Policy const & policy = {}) const
    {
        return updateAbout(state, measurement, internal::NoReference{}, policy);
    }

    /** Conditions \a state on \a measurement as update(state, measurement, policy) does, with the
     * measurement linearized about \a about, a point or a distribution (see linearizedAbout()), rather
     * than at the state (decisions.md [linearization-reference]): a shadow filter's update about the main
     * filter's state. An Iterated measurement refuses a reference at compile time.
     *
     * \returns as update(state, measurement, policy) does
     */
    template <Measurement M, typename Reference, typename Policy = reductions::Collapse>
        requires(IsForm || ReductionPolicy<Policy, internal::MeasuredUpdate<Params<Component_>, M>, Params<Component_>>)
    [[nodiscard]] UpdateType<M>
    update(
        Params<Component_> & state,
        M const & measurement,
        LinearizedAbout<Reference> const & about,
        Policy const & policy = {}
    ) const
    {
        return updateAbout(state, measurement, about, policy);
    }

    /** update(state, measurement, policy) into \a result, a ResultType: the form every recursion
     * has, so that a MixtureFilter reuses one slot per component. A result of a run-time number of
     * branches (a ReusableFilterUpdate) is refilled within its storage, so with the same result at
     * every step an update allocates nothing once it has been as large (see \ref Allocation).
     *
     * \returns nothing, or the error update(state, measurement, policy) would return, with
     *          \a state unchanged and \a result unspecified
     */
    template <Measurement M, typename Policy, typename Result>
        requires(
            (IsForm || ReductionPolicy<Policy, internal::MeasuredUpdate<Params<Component_>, M>, Params<Component_>>)
            && !internal::isLinearizedAbout<Policy>
        )
    [[nodiscard]] std::expected<void, GaussianError>
    update(Params<Component_> & state, M const & measurement, Policy const & policy, Result & result) const
    {
        return updateAboutInto(state, measurement, internal::NoReference{}, policy, result);
    }

    /** update(state, measurement, about, policy) into \a result, as
     * update(state, measurement, policy, result) fills it.
     *
     * \returns nothing, or the error update(state, measurement, about, policy) would return, with
     *          \a state unchanged and \a result unspecified
     */
    template <Measurement M, typename Reference, typename Policy, typename Result>
        requires(IsForm || ReductionPolicy<Policy, internal::MeasuredUpdate<Params<Component_>, M>, Params<Component_>>)
    [[nodiscard]] std::expected<void, GaussianError>
    update(
        Params<Component_> & state,
        M const & measurement,
        LinearizedAbout<Reference> const & about,
        Policy const & policy,
        Result & result
    ) const
    {
        return updateAboutInto(state, measurement, about, policy, result);
    }

private:

    /** The body of update(state, measurement, policy), linearized about \a reference, a
     * LinearizedAbout or internal::NoReference.
     */
    template <Measurement M, typename Reference, typename Policy>
    [[nodiscard]] UpdateType<M>
    updateAbout(
        Params<Component_> & state,
        M const & measurement,
        Reference const & reference,
        Policy const & policy
    ) const
    {
        constexpr bool Statistics = internal::policyReadsStatistics<Policy>;
        if constexpr (IsForm)
        {
            static_assert(
                internal::isOneBranchInPlacePolicy<Policy>,
                "a square-root or information form has one branch and takes Collapse, Select, or a Gate over them; "
                "a policy of the user's own is the covariance form's (decisions.md [gaussian-forms])"
            );
            return internal::updateInPlace<Statistics>(
                state,
                measurement,
                reference,
                internal::oneBranchAcceptor(policy)
            );
        }
        else
        {
            using Measured = typename decltype(internal::updateBranches<Statistics>(state, measurement))::value_type;
            if constexpr (internal::isOneBranchInPlacePolicy<Policy> && Measured::BranchCount == 1)
            {
                // One branch is its own collapse, and is the new state, conditioned where it is
                // (decisions.md [borrowed-intermediates]); a gate decides on its NIS before the state
                // is written, so a rejection copies nothing [update-outcomes].
                return internal::updateInPlace<Statistics>(
                    state,
                    measurement,
                    reference,
                    internal::oneBranchAcceptor(policy)
                );
            }
            else if constexpr (Measured::BranchCount == std::dynamic_extent)
            {
                auto result = UpdateType<M>{};
                auto const reduced = reduceBranches(state, measurement, reference, policy, *result);
                if (!reduced)
                {
                    return std::unexpected{reduced.error()};
                }
                return result;
            }
            else
            {
                auto measured = internal::updateBranches<Statistics>(state, measurement, reference);
                if (!measured)
                {
                    return std::unexpected{measured.error()};
                }
                // The policy writes what it keeps to the state, which it never reads.
                auto out = internal::OneComponentOutput<Params<Component_>, Scalar>{state};
                internal::reduceInto(policy, std::move(measured->branches), out);
                auto result = internal::filterUpdate(std::move(*measured), out.added);
                internal::MixtureEvidenceAccess::setLogKeptWeight(result, out.logWeight);
                return result;
            }
        }
    }

    /** The body of update(state, measurement, policy, result), linearized about \a reference, a
     * LinearizedAbout or internal::NoReference.
     */
    template <Measurement M, typename Reference, typename Policy, typename Result>
    [[nodiscard]] std::expected<void, GaussianError>
    updateAboutInto(
        Params<Component_> & state,
        M const & measurement,
        Reference const & reference,
        Policy const & policy,
        Result & result
    ) const
    {
        if constexpr (!IsForm && Result::BranchCount == std::dynamic_extent)
        {
            return reduceBranches(state, measurement, reference, policy, result);
        }
        else
        {
            auto updated = updateAbout(state, measurement, reference, policy);
            if (!updated)
            {
                return std::unexpected{updated.error()};
            }
            result = std::move(*updated);
            return {};
        }
    }

    /** The update of \a state by \a measurement, linearized about \a reference, whose noise has a
     * run-time number of components, reduced by \a policy into \a result, whose scratch holds the
     * branches between calls.
     *
     * \returns nothing, or the update's error, with \a state unchanged
     */
    template <Measurement M, typename Reference, typename Policy, typename Result>
    [[nodiscard]] std::expected<void, GaussianError>
    reduceBranches(
        Params<Component_> & state,
        M const & measurement,
        Reference const & reference,
        Policy const & policy,
        Result & result
    ) const
    {
        constexpr bool Statistics = internal::policyReadsStatistics<Policy>;
        auto const measured = internal::updateBranchesInto<Statistics>(state, measurement, result.measured, reference);
        if (!measured)
        {
            return measured;
        }
        // The policy writes what it keeps to the state, which it never reads.
        auto out = internal::OneComponentOutput<Params<Component_>, Scalar>{state};
        internal::reduceInto(policy, std::move(result.measured.branches), out);
        internal::filterUpdateInto(result.measured, out.added, result);
        internal::MixtureEvidenceAccess::setLogKeptWeight(result, out.logWeight);
        return {};
    }

    Model_ m_model;                                   ///< the process model
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method_ m_method;  ///< the prediction method
};

namespace internal
{

/** The recursion a filter over \a Component_ runs its steps with (decisions.md [filters-and-recursions]): the
 * KalmanRecursion for a Gaussian and any other single component, and, by the specialization in
 * ImmFilter.h, the ImmRecursion for a Hybrid. \a Allocator is the state's,
 * for a recursion whose scratch is sized by the elements.
 */
template <KalmanState Component_, typename Model, typename Method, typename Allocator>
struct RecursionFor
{
    using type = KalmanRecursion<Component_, Model, Method>;  ///< the recursion
};

/** The recursion RecursionFor names. */
template <KalmanState Component_, typename Model, typename Method, typename Allocator>
using RecursionForType = typename RecursionFor<Component_, Model, Method, Allocator>::type;

}  // namespace internal

/** \ingroup Filters
 * \brief A Kalman filter: a %Gaussian state and the KalmanRecursion that predicts it through a
 * process model and conditions it on measurements.
 *
 * Prediction uses \a Method (Linearized or a sigma-point method; Linearized is exact for the
 * built-in models). Each update uses its measurement's own method, and its branches are
 * reduced to one %Gaussian by a policy, Collapse by default. With %Gaussian measurement noise
 * there is one branch and the policy has nothing to do; with mixture noise, such as an
 * inlier and an outlier component, the filter is outlier-robust.
 *
 * The filter of \ref Kalman1960 "Kalman (1960)"; with the Linearized method it is the extended
 * Kalman filter and with Unscented the unscented Kalman filter of
 * \ref Julier2004 "Julier and Uhlmann (2004)" (\ref Sarkka2013 "Särkkä (2013)", chapters 4 to
 * 6). Mixture noise with Collapse is the probabilistic data association filter of
 * \ref BarShalom1975 "Bar-Shalom and Tse (1975)".
 *
 * With a FixedStepModel the filter predicts over the model's step, by the model's method, and
 * predict() takes no time step. Given fixedStep(model, dt), it binds the step to its state's space:
 *
 * \code
 * auto filter = KalmanFilter{fixedStep(ConstantVelocity<Position, Velocity>{0.1}, 0.01), initial};
 * filter.predict();
 * \endcode
 *
 * The filter adds nothing to the free update() of its state but storage, so what it doesn't do is
 * done on state() and stored with setState(). The free update() returns the MeasurementUpdate,
 * posteriors included, which is how a caller looks before it commits, as nearest-neighbor data
 * association does, keeping the most likely of several detections:
 *
 * \code
 * auto best = std::optional<std::remove_cvref_t<decltype(*update(filter.state(), observe(method, gps, zs[0])))>>{};
 * for (auto const & z : zs)
 * {
 *     auto candidate = update(filter.state(), observe(method, gps, z));
 *     if (candidate && (!best || candidate->logLikelihood() > best->logLikelihood())) { best = std::move(*candidate); }
 * }
 * if (best) { filter.setState(*reductions::collapse()(*std::move(best))); }
 * \endcode
 *
 * Upkeep (see \ref Upkeep) applies to state() between steps, as to any filter's: a covariance that
 * rounding has left not positive definite, as the standard update can after a measurement far more
 * precise than the prior, is repaired by
 *
 * \code
 * auto const repair = upkeep::repairCovariance();  // roots::ridge() unless given another repair
 * auto const repaired = repair(filter.state());    // true if it was not positive definite
 * \endcode
 *
 * or by the filter's own upkeep, given last to the constructor and run by maintain() where the caller
 * calls it, as every filter holds one:
 *
 * \code
 * auto filter = KalmanFilter{model, initial, upkeep::repairCovariance()};
 * filter.predict(dt);
 * (void)filter.update(fix);
 * auto const repaired = filter.maintain();
 * \endcode
 *
 * Over a square-root or information form of the %Gaussian (decisions.md [gaussian-forms]) it is the
 * square-root, information, or square-root information filter, which SqrtKalmanFilter,
 * InfoFilter, and SqrtInfoFilter spell space first; see KalmanRecursion for
 * what changes in the steps.
 *
 * \tparam Component_  the state's type (see Component), a Gaussian over the state space (see
 *                     \ref Spaces), or a form of the %Gaussian (see GaussianForm)
 * \tparam Model_      the process model, a ProcessModel or a FixedStepModel, which
 *                     partition the space
 * \tparam Method_     the prediction method; a FixedStepModel's own
 * \tparam Recursion_  the recursion that runs the steps (see Recursion): the KalmanRecursion over the
 *                     component unless given one, as <tt>KalmanFilter{recursion, state}</tt> deduces
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one
 *
 * \sa predict(), update(), KalmanRecursion, Recursion, \ref Filters
 */
template <
    KalmanState Component_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    RecursionOver<Component_> Recursion_ = KalmanRecursion<Component_, Model_, Method_>,
    typename Upkeep_ = upkeep::None>
    requires std::same_as<typename Recursion_::ModelType, Model_>
          && std::same_as<typename Recursion_::MethodType, Method_>
class KalmanFilter
    : public internal::HeldUpkeep<KalmanFilter<Component_, Model_, Method_, Recursion_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<KalmanFilter, Upkeep_>;  ///< what holds the upkeep

    /// whether the state is a square-root or information form, whose steps are FormSteps.h's
    static constexpr bool IsForm = internal::isOtherGaussianForm<Params<Component_>>;

public:

    using Scalar = typename Params<Component_>::Scalar;        ///< the scalar type
    using SpaceType = typename Params<Component_>::SpaceType;  ///< the state space
    using ModelType = Model_;                                  ///< the process model
    using MethodType = Method_;                                ///< the prediction method
    using StateType = Params<Component_>;                      ///< the state the filter owns
    using RecursionType = Recursion_;                          ///< the steps it runs (see Recursion)

    /** A filter with process model \a model, initial state \a state, prediction method \a method, and
     * upkeep \a upkeep, which maintain() runs; a fixed-step model carries its own method, so there is
     * none to give.
     */
    KalmanFilter(
        Model_ model,
        Params<Component_> state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : UpkeepBase{std::move(upkeep)}
        , m_recursion{std::move(model), std::move(method)}
        , m_state{std::move(state)}
    {
        internal::assertPartitions(m_recursion.model(), m_state.space());
    }

    /** A filter with process model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    KalmanFilter(Model_ model, Params<Component_> state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : KalmanFilter{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}

    /** A filter that runs \a recursion, the library's or one of the user's own (see Recursion), on
     * the initial state \a state, whose space the recursion's model must partition, with upkeep
     * \a upkeep, which maintain() runs.
     */
    KalmanFilter(Recursion_ recursion, StateType state, std::type_identity_t<Upkeep_> upkeep = {})
        : UpkeepBase{std::move(upkeep)}
        , m_recursion{std::move(recursion)}
        , m_state{std::move(state)}
    {
        internal::assertPartitions(m_recursion.model(), m_state.space());
    }

    /** \returns the current state estimate */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the current state estimate, to change where it is between steps, as when inflating
     * its covariance after a gap in the measurements. It must keep its space; setState() replaces it
     * with another, and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state estimate, as when reinitializing, with one whose space the process model's
     * writes partition, as the constructor's must.
     */
    void
    setState(StateType state)
    {
        internal::assertPartitions(m_recursion.model(), state.space());
        m_state = std::move(state);
    }

    /** \returns the recursion: the model and method the steps run with */
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

    /** Moves the state forward by \a dt, any TimeStep (see \ref Time), through the process model,
     * with \a inputs passed to its model as in Emrys::predict(): an uncertain input carries its
     * uncertainty through the dynamics, and anything else is known. A linearizedAbout() reference
     * first among \a inputs linearizes the step about it rather than at the state
     * (decisions.md [linearization-reference]), as a shadow filter's step is linearized about another
     * filter's state.
     *
     * \returns nothing for a Gaussian; for a form of the %Gaussian, a \c std::expected that is
     *          empty when the form could not take the step (see KalmanRecursion::predict())
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        return m_recursion.predict(m_state, internal::toScalarTime<Scalar>(dt), inputs...);
    }

    /** Moves the state forward by the fixed-step model's step, with \a inputs passed as in
     * Emrys::predict(), returning as predict(dt, inputs...) does.
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] auto
    predict(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        return m_recursion.predict(m_state, inputs...);
    }

    /** Moves the state forward by \a steps of the fixed-step model, as that many calls of
     * predict(inputs...) do, with \a inputs at every step.
     *
     * \returns as predict(inputs...) does; for a form, the first step the form could not take,
     *          which leaves the state as that step left it
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(Steps steps, Inputs const &... inputs)
    {
        EMRYS_FILTER_ASSERT(steps.count >= 0 && "the number of steps must be nonnegative");
        if constexpr (IsForm)
        {
            using Result = decltype(m_recursion.predict(m_state, inputs...));
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                if (Result result = m_recursion.predict(m_state, inputs...); !result)
                {
                    return result;
                }
            }
            return Result{};
        }
        else
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                m_recursion.predict(m_state, inputs...);
            }
        }
    }

    /** Moves the state forward as predict(dt, inputs...) does.
     *
     * \returns the step as a PredictedJoint: the state before it, the state after it, and their
     * cross-covariance, for a GaussianHistory to record
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_> && !IsForm)
    [[nodiscard]] PredictedJoint<SpaceType, Scalar>
    predictJoint(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        return m_recursion.predictJoint(m_state, dt, inputs...);
    }

    /** Moves the state forward as predict(inputs...) does with a fixed-step model.
     *
     * \returns the step as a PredictedJoint, as predictJoint(dt, inputs...) does
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !IsForm && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] PredictedJoint<SpaceType, Scalar>
    predictJoint(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        return m_recursion.predictJoint(m_state, inputs...);
    }

    /** Conditions the state on \a measurement, by the measurement's own method, and reduces the
     * branches to the new state with \a policy, a ReductionPolicy to the component (Collapse,
     * Select, or a Gate). If the policy
     * rejects the measurement, the state is unchanged.
     *
     * \returns the update before reduction, without its posteriors, and whether it was accepted
     * (see FilterUpdate): each noise component's posterior probability, such as the probability
     * that the measurement is an outlier, and its innovation, from which the observation's
     * log-likelihood and the NIS follow when asked for (decisions.md [innovation-statistics]), in a \c std::expected
     * that is empty, with the state unchanged, when the update could not be taken (see
     * KalmanRecursion::update())
     */
    template <Measurement M, typename Policy = reductions::Collapse>
        requires(!internal::isLinearizedAbout<Policy>)
    [[nodiscard]] auto
    update(M const & measurement, Policy const & policy = {})
    {
        return m_recursion.update(m_state, measurement, policy);
    }

    /** Conditions the state on \a measurement as update(measurement, policy) does, with the
     * measurement linearized about \a about, a point or a distribution (see linearizedAbout()),
     * rather than at the state (decisions.md [linearization-reference]): a shadow filter's update about
     * another filter's state, which with predict(dt, about) runs the shadow's whole step about it.
     *
     * \returns as update(measurement, policy) does
     */
    template <Measurement M, typename Reference, typename Policy = reductions::Collapse>
    [[nodiscard]] auto
    update(M const & measurement, LinearizedAbout<Reference> const & about, Policy const & policy = {})
    {
        return m_recursion.update(m_state, measurement, about, policy);
    }

    /** As update(measurement, policy), but into \a result, the update inside the
     * \c std::expected that update() returns, whose storage is reused: with the same result at
     * every step, an update under mixture noise, whose branches are a run-time count, allocates
     * nothing once the result has held as many (see \ref Allocation).
     *
     * \returns nothing, or the error update(measurement, policy) would return
     */
    template <Measurement M, typename Policy, typename Result>
        requires(!internal::isLinearizedAbout<Policy>)
    [[nodiscard]] std::expected<void, GaussianError>
    update(M const & measurement, Policy const & policy, Result & result)
    {
        return m_recursion.update(m_state, measurement, policy, result);
    }

    /** As update(measurement, about, policy), but into \a result, as update(measurement, policy, result)
     * fills it.
     *
     * \returns nothing, or the error update(measurement, about, policy) would return
     */
    template <Measurement M, typename Reference, typename Policy, typename Result>
    [[nodiscard]] std::expected<void, GaussianError>
    update(M const & measurement, LinearizedAbout<Reference> const & about, Policy const & policy, Result & result)
    {
        return m_recursion.update(m_state, measurement, about, policy, result);
    }

private:

    RecursionType m_recursion;  ///< the model and method the steps run with
    StateType m_state;          ///< the state estimate
};

/// \cond DEDUCTION_GUIDES
template <typename P, typename Model>
    requires KalmanState<Density<P>> && (!Recursion<Model>)
KalmanFilter(Model, P) -> KalmanFilter<
    Density<P>,
    internal::BoundStep<Model, typename P::SpaceType, typename P::Scalar>,
    DefaultMethodFor<Model>>;

template <typename P, Recursion R>
    requires KalmanState<Density<P>> && std::same_as<typename R::StateType, P>
KalmanFilter(R, P) -> KalmanFilter<Density<P>, typename R::ModelType, typename R::MethodType, R>;

// A third argument is a method or an upkeep, told apart by its type.
template <typename P, typename Model, typename X>
    requires KalmanState<Density<P>> && (!Recursion<Model>)
KalmanFilter(Model, P, X) -> KalmanFilter<
    Density<P>,
    internal::BoundStep<Model, typename P::SpaceType, typename P::Scalar>,
    internal::MethodArgumentFor<Model, X>,
    KalmanRecursion<
        Density<P>,
        internal::BoundStep<Model, typename P::SpaceType, typename P::Scalar>,
        internal::MethodArgumentFor<Model, X>>,
    internal::UpkeepArgumentFor<X>>;

template <typename P, typename Model, typename Method, typename U>
    requires KalmanState<Density<P>>
KalmanFilter(Model, P, Method, U) -> KalmanFilter<
    Density<P>,
    internal::BoundStep<Model, typename P::SpaceType, typename P::Scalar>,
    Method,
    KalmanRecursion<Density<P>, internal::BoundStep<Model, typename P::SpaceType, typename P::Scalar>, Method>,
    U>;

template <typename P, Recursion R, typename U>
    requires KalmanState<Density<P>> && std::same_as<typename R::StateType, P> && internal::UpkeepArgument<U>
KalmanFilter(R, P, U) -> KalmanFilter<Density<P>, typename R::ModelType, typename R::MethodType, R, U>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <typename Component_, typename Model_, typename Method_, typename Recursion_, typename Upkeep_>
void
describe(
    std::ostream & stream,
    KalmanFilter<Component_, Model_, Method_, Recursion_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "Kalman filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
