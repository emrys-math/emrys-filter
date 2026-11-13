#pragma once

/** \file
 * The Recursion concept (decisions.md [filters-and-recursions], [user-extension-points]): what a KalmanFilter or a
 * MixtureFilter asks of the object that runs its steps, spelled so that a recursion of the user's own, a method the
 * library doesn't ship, runs inside either filter; and internal::UpdateAccess, through which the filters write the
 * log-likelihood their results hold [update-outcomes]. Read KalmanFilter.h next, whose KalmanRecursion is the first
 * model of one.
 */

#include <concepts>

#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief A recursion: a filter without its state, acting in place on a state passed in
 * (decisions.md [filters-and-recursions]).
 *
 * A filter holds a state and a recursion: the process model, the prediction method, and any scratch
 * its steps need. KalmanRecursion runs the Kalman filter's steps over one component and
 * ImmRecursion the interacting multiple model's over a Hybrid; KalmanFilter and MixtureFilter take
 * theirs as their last template parameter, defaulting to the one the component names, and deduce it
 * from a recursion passed to the constructor, <tt>MixtureFilter{recursion, mixture}</tt>. So a
 * recursion of the user's own, a method the library doesn't take
 * (decisions.md [user-extension-points]), runs over one state or over each component of a mixture
 * with nothing of the library's changed; usage.md, "Writing a recursion", shows one that wraps a
 * KalmanRecursion.
 *
 * The concept checks what it can without a measurement: the type members \c Scalar, a RealScalar;
 * \c SpaceType, a Space; \c StateType, the parameters the steps act on (a KalmanFilter's or a
 * mixture's component, \c Params of the filter's component); \c ModelType and \c MethodType, the
 * filter's model and method; <tt>static constexpr bool Reentrant</tt>, whether steps on different
 * states may run at once, which is false for a recursion with scratch, and which a MixtureFilter
 * reads to refuse an execution policy other than Sequential; and \c model() and \c method(),
 * returning the two by constant reference. The rest of the contract is what the filters call, each
 * instantiated only where it is used:
 *
 * - <tt>predict(state, dt, inputs...)</tt>, or <tt>predict(state, inputs...)</tt> for a fixed-step
 *   model (see FixedStepModel), moving \a state forward in place and returning nothing, or a
 *   \c std::expected<void, E> that is empty when the state could not take the step.
 * - <tt>template \<MeasurementModel M\> using UpdateType</tt>, the \c std::expected<ResultType<M>,
 *   NumericError> (GaussianError and WeightError are its names) that <tt>update(state, measurement, policy)</tt>
 * returns, and <tt>template \<MeasurementModel M\> using ResultType</tt>, its value: a type with a <tt>bool
 * accepted</tt> and the observation's log-likelihood under the prior state as
 *   \c logLikelihood(), by which a mixture reweights the component, as every filter's result has
 *   them (decisions.md [update-outcomes]). An update that returns \c accepted false, because the
 *   policy rejected the measurement or the model gives it no density, has not written \a state, so
 *   that no filter copies its state to undo one; an update that returns an error leaves \a state
 *   valid. The policy is any callable the filter was given, wrapped so that its type is the
 *   filter's: run it on the branches by reduceToOne(), which returns the state it keeps, or
 *   nothing (see ReductionPolicy).
 * - <tt>update(state, measurement, policy, result)</tt>, the same update into a \c ResultType<M>
 *   whose storage is reused, returning \c std::expected<void, GaussianError>. A MixtureFilter keeps
 *   one result per component and refills them; a result constructible from the filter's allocator
 *   is made empty with it and filled this way, and any other is moved from
 *   <tt>update(state, measurement, policy)</tt> into its slot.
 * - <tt>predictJoint(state, dt, inputs...)</tt>, or without \a dt, returning the step as a
 *   PredictedJoint for a smoother's history, only where predictJoint() is called on the filter.
 *
 * A MixtureFilter runs a %Gaussian component's update through the measurement's branches rather than
 * the recursion when the policy may keep several of them, adding them to the filter's output as
 * Keep does, so a recursion over Gaussians decides the update only under a policy that returns the
 * component it keeps, such as Collapse or Select, or a Gate over one.
 *
 * \sa RecursionOver, KalmanRecursion, ImmRecursion, KalmanFilter, MixtureFilter
 */
template <typename R>
concept Recursion = RealScalar<typename R::Scalar> && Space<typename R::SpaceType> && requires(R const & recursion) {
    typename R::StateType;
    typename R::ModelType;
    typename R::MethodType;
    { R::Reentrant } -> std::convertible_to<bool>;
    { recursion.model() } -> std::same_as<typename R::ModelType const &>;
    { recursion.method() } -> std::same_as<typename R::MethodType const &>;
};

/** \ingroup Filters
 * \brief A Recursion whose steps act on the parameters of the component \a Component_, the
 * constraint a filter over that component puts on its recursion.
 */
template <typename R, typename Component_>
concept RecursionOver = Recursion<R> && std::same_as<typename R::StateType, Params<Component_>>;

namespace internal
{

/** Writes what a filter's result shows only through accessors, its log-likelihood, through
 * \c logLikelihood() (decisions.md [update-outcomes]), and a mixture filter's posterior indices,
 * through \c posteriorIndices(): each result names this struct its friend, and the filter that
 * fills the result writes through it.
 */
struct UpdateAccess
{
    /** \returns the log-likelihood \a update holds, to write */
    template <typename Update>
    [[nodiscard]] static auto &
    logLikelihood(Update & update)
    {
        return update.m_logLikelihood;
    }

    /** \returns the end of each prior component's posterior indices that \a update holds, to write */
    template <typename Update>
    [[nodiscard]] static auto &
    posteriorEnds(Update & update)
    {
        return update.m_posteriorEnds;
    }
};

}  // namespace internal

}  // namespace Emrys
