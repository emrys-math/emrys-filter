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
 * auto filter = AmmFilter{HybridGaussianParams<State, Regime>{CategoricalParams<Regime>{}, initial},
 *                                             slow, fast};   // one process model per regime
 * filter.predict(dt);
 * auto const updated = filter.update(fix);
 * upkeep::floorModes(1e-3)(filter.state());   // keep every mode recoverable
 * filter.state().modes().toDensity()->probability<Fast>();
 * \endcode
 *
 * The models are a pack, so the filter's upkeep comes before the state, after the method if one is
 * given, and maintain() runs it:
 *
 * \code
 * auto floored = AmmFilter{upkeep::floorModes(1e-3), HybridGaussianParams<State, Regime>{CategoricalParams<Regime>{},
 * initial}, slow, fast}; auto const updated = floored.update(fix); (void)floored.maintain();
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
 * \tparam Model_      a SwitchingModel over the mode space, with the mode space as its own axis:
 *                     one process model per mode, or one shared
 * \tparam Method_     the prediction method of every mode whose model carries none of its own
 *                    (see withMethod()); ModelsMethod with fixed-step models, which all carry one
 * \tparam Allocator_  the state's allocator (see \ref Allocation)
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one, such as FloorModes
 *
 * \sa ImmFilter, FloorModes, \ref Filters
 */
template <
    Component Component_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class AmmFilter : public ImmFilter<Component_, Model_, Method_, Allocator_, Upkeep_>
{
public:

    using Base = ImmFilter<Component_, Model_, Method_, Allocator_, Upkeep_>;  ///< the filter this is

    /** A filter with initial state \a state whose modes follow \a models, one per mode in
     * flat-index order or one shared by every mode, each predicting by its models' default method.
     */
    template <typename... Models>
        requires std::constructible_from<Model_, typename Model_::ChainType, Models...>
    AmmFilter(typename Base::StateType state, Models... models)
        // The braces sequence the arguments, so the chain copies the modes before the state moves.
        : Base{Model_{typename Model_::ChainType{state.modeSpace()}, std::move(models)...}, std::move(state)}
    {}

    /** As AmmFilter(state, models...), predicting by \a method. The method
     * comes first because the models are a pack; fixed-step models carry their own, so there is
     * none to give.
     */
    template <typename... Models>
        requires(!isFixedStep<Model_> && std::constructible_from<Model_, typename Model_::ChainType, Models...>)
    AmmFilter(Method_ method, typename Base::StateType state, Models... models)
        : Base{
              Model_{typename Model_::ChainType{state.modeSpace()}, std::move(models)...},
              std::move(state),
              std::move(method)
          }
    {}

    /** As AmmFilter(state, models...), with upkeep \a upkeep, which maintain() runs. The upkeep
     * comes before the state because the models are a pack.
     */
    template <typename... Models>
        requires(
            !std::same_as<Upkeep_, Emrys::upkeep::None>
            && std::constructible_from<Model_, typename Model_::ChainType, Models...>
        )
    AmmFilter(std::type_identity_t<Upkeep_> upkeep, typename Base::StateType state, Models... models)
        : Base{
              Model_{typename Model_::ChainType{state.modeSpace()}, std::move(models)...},
              std::move(state),
              MethodFor<Model_, Method_>{},
              std::move(upkeep)
          }
    {}

    /** As AmmFilter(method, state, models...), with upkeep \a upkeep, which maintain() runs. */
    template <typename... Models>
        requires(
            !isFixedStep<Model_> && !std::same_as<Upkeep_, Emrys::upkeep::None>
            && std::constructible_from<Model_, typename Model_::ChainType, Models...>
        )
    AmmFilter(Method_ method, std::type_identity_t<Upkeep_> upkeep, typename Base::StateType state, Models... models)
        : Base{
              Model_{typename Model_::ChainType{state.modeSpace()}, std::move(models)...},
              std::move(state),
              std::move(method),
              std::move(upkeep)
          }
    {}
};

/// \cond DEDUCTION_GUIDES
template <Component C, typename ModeSpace, typename Allocator, typename... Models>
AmmFilter(HybridParams<C, ModeSpace, Allocator>, Models...) -> AmmFilter<
    C,
    SwitchingModel<
        ModeSpace,
        typename C::Scalar,
        ModeSpace,
        internal::BoundStep<Models, typename Params<C>::SpaceType, typename Params<C>::Scalar>...>,
    DefaultMethodFor<SwitchingModel<ModeSpace, typename C::Scalar, ModeSpace, Models...>>,
    Allocator>;

template <typename Method, Component C, typename ModeSpace, typename Allocator, typename... Models>
    requires(!internal::UpkeepArgument<Method>)
AmmFilter(Method, HybridParams<C, ModeSpace, Allocator>, Models...) -> AmmFilter<
    C,
    SwitchingModel<
        ModeSpace,
        typename C::Scalar,
        ModeSpace,
        internal::BoundStep<Models, typename Params<C>::SpaceType, typename Params<C>::Scalar>...>,
    Method,
    Allocator>;

template <typename U, Component C, typename ModeSpace, typename Allocator, typename... Models>
    requires internal::UpkeepArgument<U>
AmmFilter(U, HybridParams<C, ModeSpace, Allocator>, Models...) -> AmmFilter<
    C,
    SwitchingModel<
        ModeSpace,
        typename C::Scalar,
        ModeSpace,
        internal::BoundStep<Models, typename Params<C>::SpaceType, typename Params<C>::Scalar>...>,
    DefaultMethodFor<SwitchingModel<ModeSpace, typename C::Scalar, ModeSpace, Models...>>,
    Allocator,
    U>;

template <typename Method, typename U, Component C, typename ModeSpace, typename Allocator, typename... Models>
AmmFilter(Method, U, HybridParams<C, ModeSpace, Allocator>, Models...) -> AmmFilter<
    C,
    SwitchingModel<
        ModeSpace,
        typename C::Scalar,
        ModeSpace,
        internal::BoundStep<Models, typename Params<C>::SpaceType, typename Params<C>::Scalar>...>,
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
