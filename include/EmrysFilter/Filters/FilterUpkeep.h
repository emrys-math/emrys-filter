#pragma once

/** \file
 * The upkeep a filter holds (decisions.md [upkeep]): internal::HeldUpkeep, the base that gives every
 * filter its upkeep() and maintain(). Read KalmanFilter.h next.
 */

#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/ProcessModel.h>

namespace Emrys
{

namespace internal
{

/** Whether \a U, given to a filter's constructor where a prediction method may stand, is the filter's
 * upkeep instead: anything but a transform method, or what a fixed-step model takes in place of one.
 */
template <typename U>
concept UpkeepArgument = !TransformMethod<U> && !std::same_as<std::remove_cvref_t<U>, ModelsMethod>;

/** The prediction method a filter with process model \a Model is given by an argument \a X that is a
 * method or an upkeep: \a X, or the model's default. One deduction guide takes both, rather than one
 * guide each, since MSVC forms the type a guide names before it checks the guide's constraints.
 */
template <typename Model, typename X>
using MethodArgumentFor = std::conditional_t<UpkeepArgument<X>, DefaultMethodFor<Model>, X>;

/** The upkeep a filter is given by an argument \a X that is a method or an upkeep: \a X, or None. */
template <typename X>
using UpkeepArgumentFor = std::conditional_t<UpkeepArgument<X>, X, upkeep::None>;

/** \brief The upkeep a filter holds, and the members that give it out and run it on the filter's
 * state (decisions.md [upkeep]).
 *
 * \a Derived is the filter, whose state() the upkeep runs on, and \a Upkeep_ the upkeep, an action,
 * a trigger, a Sequence of them, or upkeep::None. The filter never runs it on its own: the caller
 * calls maintain() where it fits the problem, as it would call the upkeep on state().
 *
 * maintain() exists only when the upkeep applies to the state: a filter holding one that doesn't, such
 * as a KalmanFilter holding a Resample, constructs, since the constructor tells an upkeep from a method
 * by its type alone, and has no maintain(). A held Resample or Roughen holds a reference to its
 * generator, so a copy of the filter draws from the same generator.
 */
template <typename Derived, typename Upkeep_>
class HeldUpkeep
{
public:

    /** the upkeep the filter holds (see \ref Upkeep) */
    using UpkeepType = Upkeep_;

    /** \returns the upkeep the filter holds, upkeep::None if it was given none */
    [[nodiscard]] Upkeep_ const &
    upkeep() const
    {
        return m_upkeep;
    }

    /** Runs the upkeep on the filter's state: the same as <tt>filter.upkeep()(filter.state())</tt>. The
     * upkeep holds what else it needs, such as a random generator or an execution policy.
     * \returns whether it changed the state, or the NumericError that stopped it
     */
    [[nodiscard]] std::expected<bool, NumericError>
    maintain()
        requires UpkeepOperation<Upkeep_, typename Derived::StateType>
    {
        return m_upkeep(static_cast<Derived &>(*this).state());
    }

protected:

    /** Holds \a upkeep. */
    explicit HeldUpkeep(Upkeep_ upkeep)
        : m_upkeep{std::move(upkeep)}
    {}

private:

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Upkeep_ m_upkeep;  ///< the upkeep maintain() runs
};

}  // namespace internal

}  // namespace Emrys
