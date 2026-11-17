#pragma once

/** \file
 * AmmFilter (decisions.md [autonomous-multiple-model]): the ImmFilter on the
 * identity chain, whose constructor builds that chain from the state's modes, and the AmmFilter
 * alias. The floor that keeps its modes recoverable is FloorModes, in Upkeep.h. Read
 * ParticleFilter.h next; the design review is docs/guide/imm-filter.md.
 */

#include <concepts>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/SwitchingModel.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief The autonomous multiple model (AMM) estimator: a bank of filters, one per mode, each
 * running on its own, and the probability that each mode is the one in force, by Bayes' rule over
 * their measurement likelihoods.
 *
 * The mode is fixed for all time but unknown, so it is a parameter to identify rather than a state
 * that switches: the ImmFilter with the identity chain, which this class
 * builds from the state's modes and gives its base. The two are one algorithm, with
 * \f$ \Pi = I \f$, so every IMM step works here unchanged. Mixing through the identity gives each
 * mode its own component, and the IMM's predict() sees the identity and skips it: each mode's
 * component moves through its own process model as a KalmanFilter's would, and the mode
 * probabilities carry over. update() updates each component and multiplies each mode's
 * probability by \f$ p(z \mid m) \f$. The combined estimate is the collapsed() of the state's density.
 *
 * \code
 * auto filter = AmmFilter{byMode<Regime>(when<Slow>(slow), when<Fast>(fast)),   // a model per regime
 *                         HybridGaussianParams<State, Regime>{CategoricalParams<Regime>{}, initial}};
 * filter.predict(dt);
 * auto const updated = filter.update(fix);
 * upkeep::floorModes(1e-3)(filter.state());   // keep every mode recoverable
 * filter.state().modes().toDensity()->probability<Fast>();
 * \endcode
 *
 * The method and the upkeep come after the state, as the other filters take them, and maintain()
 * runs the upkeep:
 *
 * \code
 * auto floored = AmmFilter{motion, prior, upkeep::floorModes(1e-3)};
 * auto const updated = floored.update(fix);
 * (void)floored.maintain();
 * \endcode
 *
 * Its known failure is a mode probability that falls to zero, or so near it that no run of
 * measurements brings it back: the estimator then cannot follow a mode that changes after all,
 * nor recover from a mode that was wrongly ruled out. FloorModes, applied after each update,
 * bounds the probabilities below. The chain stays the base's to change through model(); a chain
 * that is no longer the identity makes the filter the IMM it is built on.
 *
 * The multiple model estimator that \ref Athans1977 "Athans et al. (1977)", §V.B–C, credit to
 * Magill (1965): a Kalman filter per model and the probabilities of equation (5.5). Theirs are
 * steady-state filters with constant innovation covariances; here each mode's component is a full
 * filter's, so its likelihood is exact under the mode. \ref BarShalom2001 "Bar-Shalom, Li, and
 * Kirubarajan (2001)" set it beside the IMM.
 *
 * \tparam Component_  the component given one mode (see Component), a Gaussian over the state
 *                     space (see \ref Spaces)
 * \tparam Model_      a SwitchingModel over the mode space: its motion is one process model, or
 *                     a byMode() of them per value of a mode axis [by-mode]
 * \tparam Method_     the prediction method of every mode whose model carries none of its own
 *                    (see withMethod()); ModelsMethod with fixed-step models, which all carry one
 * \tparam Allocator_  the state's allocator (see \ref Allocation)
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one, such as FloorModes
 *
 * It runs the ImmRecursion. A recursion of the user's own runs in an ImmFilter instead, whose
 * switching model it holds is given the chain that never moves, as this class gives its own.
 *
 * \sa ImmFilter, FloorModes, \ref Filters
 */
template <
    Component Component_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class AmmFilter
    : public ImmFilter<
          Component_,
          Model_,
          Method_,
          Allocator_,
          ImmRecursion<Component_, Model_, Method_, Allocator_>,
          Upkeep_>
{
public:

    /** the filter this is: the ImmFilter that runs the ImmRecursion */
    using Base = ImmFilter<
        Component_,
        Model_,
        Method_,
        Allocator_,
        ImmRecursion<Component_, Model_, Method_, Allocator_>,
        Upkeep_>;

    /** A filter whose modes follow the motion \a motion, one process model or a byMode() of them
     * [by-mode], from the initial state \a state, with prediction method \a method, which fixed-step
     * models carry instead, and upkeep \a upkeep, which maintain() runs. The chain is the identity on
     * the state's modes.
     */
    template <typename Motion>
        requires std::
            constructible_from<Model_, SwitchingModel<typename Model_::ModeSpaceType, typename Model_::Scalar, Motion>>
        AmmFilter(
            Motion motion,
            typename Base::StateType state,
            MethodFor<Model_, Method_> method = {},
            std::type_identity_t<Upkeep_> upkeep = {}
        )
        // The braces sequence the arguments, so the chain copies the modes before the state moves.
        : Base{
              // The motion's fixed-step models are bound to the state's space as the model converts.
              Model_(
                  SwitchingModel<typename Model_::ModeSpaceType, typename Model_::Scalar, Motion>{
                      typename Model_::ChainType{state.modeSpace()},
                      std::move(motion)
                  }
              ),
              std::move(state),
              std::move(method),
              std::move(upkeep)
          }
    {}

    /** As AmmFilter(motion, state, method, upkeep), with the default prediction method. */
    template <typename Motion>
        requires(
            !std::same_as<Upkeep_, Emrys::upkeep::None>
            && std::constructible_from<
                Model_,
                SwitchingModel<typename Model_::ModeSpaceType, typename Model_::Scalar, Motion>>
        )
    AmmFilter(Motion motion, typename Base::StateType state, std::type_identity_t<Upkeep_> upkeep)
        : AmmFilter{std::move(motion), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}
};

/// \cond DEDUCTION_GUIDES
template <
    typename Motion,
    Component C,
    typename ModeSpace,
    typename Allocator,
    typename Method = DefaultMethodFor<SwitchingModel<ModeSpace, typename C::Scalar, Motion>>>
    requires(!internal::UpkeepArgument<Method>)
AmmFilter(Motion, HybridParams<C, ModeSpace, Allocator>, Method = {}) -> AmmFilter<
    C,
    internal::BoundStep<
        SwitchingModel<ModeSpace, typename C::Scalar, Motion>,
        typename Params<C>::SpaceType,
        typename Params<C>::Scalar>,
    Method,
    Allocator>;

template <typename Motion, Component C, typename ModeSpace, typename Allocator, typename U>
    requires internal::UpkeepArgument<U>
AmmFilter(Motion, HybridParams<C, ModeSpace, Allocator>, U) -> AmmFilter<
    C,
    internal::BoundStep<
        SwitchingModel<ModeSpace, typename C::Scalar, Motion>,
        typename Params<C>::SpaceType,
        typename Params<C>::Scalar>,
    DefaultMethodFor<SwitchingModel<ModeSpace, typename C::Scalar, Motion>>,
    Allocator,
    U>;

template <typename Motion, Component C, typename ModeSpace, typename Allocator, typename Method, typename U>
AmmFilter(Motion, HybridParams<C, ModeSpace, Allocator>, Method, U) -> AmmFilter<
    C,
    internal::BoundStep<
        SwitchingModel<ModeSpace, typename C::Scalar, Motion>,
        typename Params<C>::SpaceType,
        typename Params<C>::Scalar>,
    Method,
    Allocator,
    U>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <typename Component_, typename Model_, typename Method_, typename Allocator_, typename Upkeep_>
void
describe(
    std::ostream & stream,
    AmmFilter<Component_, Model_, Method_, Allocator_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "autonomous multiple model filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
