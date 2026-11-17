#pragma once

/** \file
 * SwitchingModel (decisions.md [switching-models], [imm], [by-mode]): a Markov chain on a discrete mode space,
 * continuous-time unless the models are fixed-step, and the motion, one process model every mode follows or a
 * byMode() of them per value of a mode axis, nested where a model depends on several axes, and over a fixed step when
 * every model is a FixedStepModel [fixed-step-models]. The interacting multiple model filter's process model, which a
 * filter over a Hybrid builds from its first two arguments, the chain and the motion. withMethod() gives one mode's
 * model a prediction method of its own, ModelWithMethod. A measurement per value of a mode axis is a ByMode of
 * measurements, which observe() also makes of a ByMode of models. Read Reduction.h next; the design review is
 * docs/guide/imm-filter.md.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Models/ByMode.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>

namespace Emrys
{

/** \ingroup Models
 * \brief A process model and the prediction method that one mode of a SwitchingModel predicts it
 * by, in place of the filter's [switching-models].
 *
 * A switching model's modes predict by the filter's method unless a mode's model carries its own,
 * as a FixedStepModel always does and a model given to withMethod() does: in
 * <tt>byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(withMethod(turn, methods::unscented())))</tt> the cruise mode
 * predicts by the filter's method, Linearized by default, and the turn mode by Unscented. It is a
 * switching model's argument only: SwitchingModel::model() returns the model it holds, and no other
 * filter takes it.
 *
 * \tparam Model_   the process model, which is not fixed-step
 * \tparam Method_  the prediction method (see TransformMethod), not an Iterated one
 *
 * \sa withMethod(), SwitchingModel
 */
template <typename Model_, typename Method_>
class ModelWithMethod
{
    static_assert(
        !isFixedStep<Model_> && !isUnboundStep<Model_>,
        "a fixed-step model carries its own method; give the method to fixedStep() in place of withMethod()"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

public:

    using ModelType = Model_;    ///< the process model
    using MethodType = Method_;  ///< the prediction method

    /** \a model, predicted by \a method. */
    ModelWithMethod(Model_ model, Method_ method)
        : m_model{std::move(model)}
        , m_method{std::move(method)}
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

namespace internal
{

/** Whether \a Model is a ModelWithMethod. */
template <typename Model>
inline constexpr bool isModelWithMethod = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Model_, typename Method_>
inline constexpr bool isModelWithMethod<ModelWithMethod<Model_, Method_>> = true;

/// \endcond

/** The process model a switching model's entry \a Model holds: the model a ModelWithMethod
 * holds, or \a Model itself.
 */
template <typename Model>
struct ModeModelOf
{
    using type = Model;  ///< the process model
};

/** A ModelWithMethod holds its model. */
template <typename Model_, typename Method_>
struct ModeModelOf<ModelWithMethod<Model_, Method_>>
{
    using type = Model_;  ///< the process model
};

/** The process model a switching model's entry \a Model holds (see ModeModelOf). */
template <typename Model>
using ModeModel = typename ModeModelOf<Model>::type;

/** \returns the process model that the switching model's entry \a entry holds */
template <typename Entry>
[[nodiscard]] auto &
modeModel(Entry & entry)
{
    if constexpr (isModelWithMethod<std::remove_const_t<Entry>>)
    {
        return entry.model();
    }
    else
    {
        return entry;
    }
}

/** \returns the method that the switching model's entry \a entry predicts by: its own, if it is
 * a ModelWithMethod or a FixedStepModel, and \a fallback, the filter's, otherwise
 */
template <typename Entry, typename Fallback>
[[nodiscard]] auto const &
modeMethod(Entry const & entry, Fallback const & fallback)
{
    if constexpr (isModelWithMethod<Entry> || isFixedStep<Entry>)
    {
        return entry.method();
    }
    else
    {
        return fallback;
    }
}

}  // namespace internal

/** \ingroup Models
 * \returns \a model, which a mode of a SwitchingModel predicts by \a method in place of the
 * filter's method: <tt>byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(withMethod(turn, methods::unscented())))</tt>
 * [switching-models]. A fixed-step model takes its method from fixedStep() instead.
 */
template <typename Model, TransformMethod Method>
[[nodiscard]] ModelWithMethod<Model, Method>
withMethod(Model model, Method method)
{
    return {std::move(model), std::move(method)};
}

namespace internal
{

/** The process models that the motion \a Motion holds, as a tuple type: \a Motion itself, or for a
 * ByMode, its alternatives' models in turn, nested ones included.
 */
template <typename Motion>
struct MotionLeavesOf
{
    using type = std::tuple<Motion>;  ///< the one model
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename P, typename... Alternatives>
struct MotionLeavesOf<ByMode<P, Alternatives...>>
{
    /// every alternative's models
    using type =
        decltype(std::tuple_cat(std::declval<typename MotionLeavesOf<typename Alternatives::PartType>::type>()...));
};

/// \endcond

/** The process models that the motion \a Motion holds (see MotionLeavesOf). */
template <typename Motion>
using MotionLeaves = typename MotionLeavesOf<Motion>::type;

/** The first process model of the motion \a Motion. */
template <typename Motion>
using FirstMotionLeaf = std::tuple_element_t<0, MotionLeaves<Motion>>;

/** Calls \a f on each process model of the motion \a motion, nested alternatives included. */
template <typename Motion, typename F>
void
forEachMotionLeaf(Motion const & motion, F const & f)
{
    if constexpr (isByModePart<Motion>)
    {
        std::apply(
            [&](auto const &... alternative) {
                (forEachMotionLeaf(alternative.part(), f), ...);
            },
            motion.alternatives()
        );
    }
    else
    {
        f(motion);
    }
}

/** \returns true, after checking at compile time that the motion \a Motion chooses by each mode
 * axis once along a nesting, the axes \a Used having chosen already, and that every axis it names
 * is one of \a ModeSpace and takes no weights
 */
template <typename Motion, typename ModeSpace, typename... Used>
consteval bool
checkMotion()
{
    if constexpr (isByModePart<Motion>)
    {
        using Axis = PathTag<typename Motion::PathType>;
        static_assert(
            pathResolves<typename Motion::PathType, ModeSpace>(),
            "a process model by mode axis, byMode<Axis>(), chooses by an axis that the mode space does not have"
        );
        static_assert(
            !(std::is_same_v<Axis, Used> || ...),
            "a process model chosen by a mode axis chooses again by the same axis, which would only ever take one "
            "alternative: give each value of the axis its model once"
        );
        static_assert(
            !Motion::Weighted,
            "a process model per value of a mode axis takes no weight: the chain gives the modes' probabilities"
        );
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return (checkMotion<typename Motion::template PartType<I>, ModeSpace, Used..., Axis>() && ...);
        }(std::make_index_sequence<Motion::Count>{});
    }
    else
    {
        return true;
    }
}

/** \returns \a f applied to the process model of the motion \a motion that the mode with flat
 * index \a mode of \a modeSpace follows: \a motion itself, or the alternative its value names, in turn
 */
template <typename Motion, typename ModeSpace, typename F>
decltype(auto)
visitMotion(Motion const & motion, ModeSpace const & modeSpace, Eigen::Index mode, F const & f)
{
    if constexpr (isByModePart<Motion>)
    {
        auto const located = locateAxisByPath<typename Motion::PathType>(modeSpace);
        Eigen::Index const values = located.second->cardinality();
        return motion.visit((mode / located.first) % values, [&](auto const & part) -> decltype(auto) {
            return visitMotion(part, modeSpace, mode, f);
        });
    }
    else
    {
        return f(motion);
    }
}

/** \returns whether every ByMode of the motion \a motion has one alternative per value of its axis in
 * \a modeSpace, as an axis sized at run time can show only then
 */
template <typename Motion, typename ModeSpace>
[[nodiscard]] bool
servesEveryValue(Motion const & motion, ModeSpace const & modeSpace)
{
    if constexpr (isByModePart<Motion>)
    {
        auto const located = locateAxisByPath<typename Motion::PathType>(modeSpace);
        return located.second->cardinality() == motion.values()
            && std::apply(
                   [&](auto const &... alternative) {
                       return (servesEveryValue(alternative.part(), modeSpace) && ...);
                   },
                   motion.alternatives()
            );
    }
    else
    {
        return true;
    }
}

/** The motion \a Motion with each process model bound to \a Space and \a Scalar (see BoundStep). */
template <typename Motion, typename Space, typename Scalar>
struct BoundMotionOf
{
    using type = BoundStep<Motion, Space, Scalar>;  ///< the model, bound
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename P, typename... Values, typename... Parts, typename... Weights, typename Space, typename Scalar>
struct BoundMotionOf<ByMode<P, When<Values, Parts, Weights>...>, Space, Scalar>
{
    /// each alternative's models, bound
    using type = ByMode<P, When<Values, typename BoundMotionOf<Parts, Space, Scalar>::type, Weights>...>;
};

/// \endcond

/** \returns \a motion converted to \a Target, a motion of the same shape whose process models each
 * convert from its: how a filter binds fixed-step models to its state's space
 */
template <typename Target, typename Motion>
[[nodiscard]] Target
convertMotion(Motion const & motion)
{
    if constexpr (isByModePart<Motion>)
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return Target{typename Target::template AlternativeType<I>{
                convertMotion<typename Target::template PartType<I>>(motion.template alternative<I>().part())
            }...};
        }(std::make_index_sequence<Motion::Count>{});
    }
    else
    {
        return Target(motion);
    }
}

}  // namespace internal

/** \ingroup Models
 * \brief A process model that switches between modes: a Markov chain on a discrete mode space, and
 * the motion, the process model each mode follows (decisions.md [switching-models], [by-mode]).
 *
 * The motion is one process model, which every mode follows, or
 * <tt>byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(turn))</tt>, a process model per value of a
 * mode axis, each labeled by the value it serves, in any order (see ByMode). Alternatives nest, each
 * reading the axis it depends on, so a mode space <tt>DiscreteProduct<Maneuver, Payload></tt> takes
 * <tt>byMode<Maneuver>(when<Cruise>(byMode<Payload>(...)), when<Turn>(...))</tt>, and an axis left out,
 * such as a sensor's inlier/outlier state, doesn't affect the dynamics. A model per mode of a
 * product, in flat-index order, is <tt>byMode<Mode>(m0, m1, ...)</tt> by the product itself, whose
 * values have no names.
 *
 * Each model is a process model over the whole state (see ProcessModel), and every mode shares the
 * state space. A mode's model writes even a block the mode doesn't use: with Constant or RandomWalk
 * where the quantity persists through the mode, such as a sensor bias, and with Zero where it is zero
 * in the mode, such as the acceleration of a constant-velocity mode beside a constant-acceleration
 * one (decisions.md [unused-blocks]).
 *
 * The chain moves the mode over the same step as the models, so its kind follows theirs
 * [switching-models]:
 *  - with models stepped by a time step, the chain is a ContinuousTimeMarkovChain, and each
 *    prediction over \f$ \Delta t \f$ moves the mode by \f$ \Pi = e^{Q \Delta t} \f$
 *    (see transition()). A MarkovChain there is a compile-time error, since it would move the mode
 *    one step whatever the time step;
 *  - with fixed-step models, the chain is a MarkovChain, which moves the mode once per prediction.
 *    A ContinuousTimeMarkovChain given there is taken over the models' step, once.
 *
 * Each mode predicts by the filter's method, unless its model carries a method of its own: a model
 * given to withMethod() does, so that a constant-velocity mode, affine and exact under Linearized,
 * need not run through the sigma points a coordinated-turn mode beside it takes.
 *
 * The models are all FixedStepModel over one step or none are. With fixed-step models the switching
 * model is fixed-step too (isFixedStep), its filter's predict() takes no time step and no method, and
 * each mode predicts by its own model's method, which may differ from mode to mode. Models made by
 * fixedStep(model, dt), not yet bound to a space, are bound to the state's space by the filter given
 * the switching model.
 *
 * A filter over a Hybrid takes the chain and the motion as its first two arguments,
 * <tt>ImmFilter{chain, motion, prior}</tt>, and builds this from them.
 *
 * The Markovian switching of \ref Blom1988 "Blom and Bar-Shalom (1988)": a jump Markov linear
 * system whose mode selects the dynamics.
 *
 * \tparam ModeSpace_  the discrete mode space (see \ref Discrete)
 * \tparam Scalar_     scalar type of the chain's probabilities
 * \tparam Motion_     the motion: a process model, a ModelWithMethod, or a ByMode of them
 *
 * \sa ImmFilter, ContinuousTimeMarkovChain, MarkovChain, byMode(), withMethod()
 */
template <DiscreteSpace ModeSpace_, RealScalar Scalar_, typename Motion_>
class SwitchingModel
{
    /** the process models the motion holds */
    using Leaves = internal::MotionLeaves<Motion_>;

    /** whether each model is fixed-step, bound to a space or not yet */
    static constexpr auto LeavesFixed = []<typename... Ms>(std::tuple<Ms...> *) {
        return std::array<bool, sizeof...(Ms)>{(isFixedStep<Ms> || isUnboundStep<Ms>)...};
    }(static_cast<Leaves *>(nullptr));

    static_assert(
        std::ranges::all_of(
            LeavesFixed,
            [](bool fixed) {
                return fixed;
            }
        )
            || std::ranges::none_of(
                LeavesFixed,
                [](bool fixed) {
                    return fixed;
                }
            ),
        "a switching model's process models are all fixed-step or none are"
    );

    static_assert(internal::checkMotion<Motion_, ModeSpace_>());

public:

    /** Whether the models are fixed-step, bound to a space or not yet: then the chain is a
     * MarkovChain over their step, and otherwise a ContinuousTimeMarkovChain.
     */
    static constexpr bool FixedSteps = LeavesFixed[0];

    using Scalar = Scalar_;            ///< the scalar type of the chain's probabilities or rates
    using ModeSpaceType = ModeSpace_;  ///< the discrete mode space
    using MotionType = Motion_;        ///< the motion

    /** The chain's type: a MarkovChain with fixed-step models, a ContinuousTimeMarkovChain otherwise. */
    using ChainType = std::
        conditional_t<FixedSteps, MarkovChain<ModeSpace_, Scalar_>, ContinuousTimeMarkovChain<ModeSpace_, Scalar_>>;

    /** Whether one model serves every mode: a motion that is not a ByMode. */
    static constexpr bool Shared = !internal::isByModePart<Motion_>;

    /** The chain \a chain on the modes, and the motion \a motion, one process model or a ByMode of them. */
    SwitchingModel(ChainType chain, Motion_ motion)
        : m_chain{std::move(chain)}
        , m_motion{std::move(motion)}
    {
        if constexpr (FixedSteps)
        {
            EMRYS_FILTER_ASSERT(sharesOneStep() && "a switching model's fixed-step models must share one step");
        }
        EMRYS_FILTER_ASSERT(
            internal::servesEveryValue(m_motion, modeSpace()) && "give one process model per value of the axis that "
                                                                  "selects it"
        );
    }

    /** The continuous-time chain \a chain taken over the step of the fixed-step models of \a motion,
     * \f$ \Pi = e^{Q \Delta t} \f$, and \a motion, as the other constructor takes them.
     */
    SwitchingModel(ContinuousTimeMarkovChain<ModeSpace_, Scalar_> const & chain, Motion_ motion)
        requires FixedSteps
        : SwitchingModel{chain.transition(firstStep(motion)), std::move(motion)}
    {}

    /** Refuses a MarkovChain with models stepped by a time step, which would move the mode one step
     * of the chain whatever the time step.
     */
    template <typename Chain>
        requires(!FixedSteps && std::is_same_v<Chain, MarkovChain<ModeSpace_, Scalar_>>)
    SwitchingModel(Chain const & chain, Motion_ const & motion)
        : m_chain{chain.space()}
        , m_motion{motion}
    {
        static_assert(
            sizeof(Chain) == 0,
            "a MarkovChain moves the mode one step whatever the time step: give models stepped by a time step a "
            "ContinuousTimeMarkovChain, or make them fixed-step with fixedStep(model, dt)"
        );
    }

    /** \a other with each of its models converted to this one's: how a filter binds a switching
     * model of fixedStep() results to its state's space. Implicit, as FixedStepModel's is, so that a
     * filter's constructor takes the switching model where its deduction guide names this type.
     */
    template <typename Other>
        requires(!std::is_same_v<Other, Motion_>)
    SwitchingModel(SwitchingModel<ModeSpace_, Scalar_, Other> const & other)
        : SwitchingModel{other.chain(), internal::convertMotion<Motion_>(other.motion())}
    {}

    /** \returns the Markov chain on the modes */
    [[nodiscard]] ChainType const &
    chain() const
    {
        return m_chain;
    }

    /** \returns the Markov chain on the modes, to change it */
    [[nodiscard]] ChainType &
    chain()
    {
        return m_chain;
    }

    /** \returns the chain over a prediction of \a dt, any TimeStep (see \ref Time):
     * \f$ \Pi = e^{Q \Delta t} \f$ of the continuous-time chain
     */
    [[nodiscard]] MarkovChain<ModeSpace_, Scalar_>
    transition(TimeStep<Scalar_> auto const & dt) const
        requires(!FixedSteps)
    {
        return m_chain.transition(dt);
    }

    /** \returns the chain over one prediction of the fixed-step models: the chain itself */
    [[nodiscard]] MarkovChain<ModeSpace_, Scalar_> const &
    transition() const
        requires FixedSteps
    {
        return m_chain;
    }

    /** \returns the mode space */
    [[nodiscard]] ModeSpace_ const &
    modeSpace() const
    {
        return m_chain.space();
    }

    /** \returns the motion: the process model, or the ByMode of them */
    [[nodiscard]] Motion_ const &
    motion() const
    {
        return m_motion;
    }

    /** \returns the motion, to change its models' parameters */
    [[nodiscard]] Motion_ &
    motion()
    {
        return m_motion;
    }

    /** \returns \a f applied to the process model that the mode with flat index \a mode follows; for
     * a model given to withMethod(), the model it was given
     */
    template <typename F>
    decltype(auto)
    visit(Eigen::Index mode, F && f) const
    {
        EMRYS_FILTER_ASSERT(mode >= 0 && mode < modeSpace().cardinality() && "the mode is out of range");
        return internal::visitMotion(m_motion, modeSpace(), mode, [&](auto const & entry) -> decltype(auto) {
            return f(internal::modeModel(entry));
        });
    }

    /** \returns \a f applied to the process model that the mode with flat index \a mode follows
     * and to the method it predicts by: its own, if it was given to withMethod() or is a
     * FixedStepModel, and \a fallback, the filter's, otherwise
     */
    template <typename Fallback, typename F>
    decltype(auto)
    visit(Eigen::Index mode, Fallback const & fallback, F && f) const
    {
        EMRYS_FILTER_ASSERT(mode >= 0 && mode < modeSpace().cardinality() && "the mode is out of range");
        return internal::visitMotion(m_motion, modeSpace(), mode, [&](auto const & entry) -> decltype(auto) {
            return f(internal::modeModel(entry), internal::modeMethod(entry, fallback));
        });
    }

    /** \returns what the models' types say about whether their writes partition \a Space: that they
     * fail if one model's do, that they hold if every model's do, and otherwise unknown
     */
    template <typename Space>
    [[nodiscard]] static consteval internal::StaticPartition
    staticPartition()
    {
        using internal::StaticPartition;
        constexpr auto each = []<typename... Ms>(std::tuple<Ms...> *) {
            return std::array<StaticPartition, sizeof...(Ms)>{
                internal::staticPartition<internal::ModeModel<Ms>, Space>()...
            };
        }(static_cast<Leaves *>(nullptr));
        if (std::ranges::find(each, StaticPartition::fails) != each.end())
        {
            return StaticPartition::fails;
        }
        return std::ranges::all_of(
                   each,
                   [](StaticPartition p) {
                       return p == StaticPartition::holds;
                   }
               )
                 ? StaticPartition::holds
                 : StaticPartition::unknown;
    }

    /** \returns whether every model's blocks cover every coordinate of \a space exactly once */
    template <typename Space>
    [[nodiscard]] bool
    partitions(Space const & space) const
    {
        bool all = true;
        internal::forEachMotionLeaf(m_motion, [&](auto const & entry) {
            all = all && internal::modeModel(entry).partitions(space);
        });
        return all;
    }

private:

    /** \returns whether the fixed-step models all share one step */
    [[nodiscard]] bool
    sharesOneStep() const
    {
        bool shared = true;
        auto const first = firstStep(m_motion);
        internal::forEachMotionLeaf(m_motion, [&](auto const & entry) {
            shared = shared && entry.timeStep() == first;
        });
        return shared;
    }

    /** \returns the step of the first of the fixed-step models of \a motion, which they all share */
    [[nodiscard]] static auto
    firstStep(Motion_ const & motion)
    {
        using First = internal::FirstMotionLeaf<Motion_>;
        auto first = std::optional<std::remove_cvref_t<decltype(std::declval<First const &>().timeStep())>>{};
        internal::forEachMotionLeaf(motion, [&](auto const & entry) {
            if (!first)
            {
                first = entry.timeStep();
            }
        });
        return *first;
    }

    ChainType m_chain;  ///< the chain on the modes
    Motion_ m_motion;   ///< the motion
};

/** \ingroup Models
 * A SwitchingModel whose process models are all FixedStepModel is fixed-step too.
 */
template <typename ModeSpace_, typename Scalar_, typename Motion_>
inline constexpr bool isFixedStep<SwitchingModel<ModeSpace_, Scalar_, Motion_>>{
    []<typename... Ms>(std::tuple<Ms...> *) {
        return (isFixedStep<Ms> && ...);
    }(static_cast<internal::MotionLeaves<Motion_> *>(nullptr))
};

namespace internal
{

/** The method a SwitchingModel's filter predicts the modes by that carry none of their own:
 * Linearized, or, with fixed-step models, which each carry their own, ModelsMethod, there being no
 * such mode.
 */
template <typename ModeSpace_, typename Scalar_, typename Motion_>
struct DefaultMethodOf<SwitchingModel<ModeSpace_, Scalar_, Motion_>>
{
    /// the method
    using type = std::conditional_t<
        isFixedStep<FirstMotionLeaf<Motion_>> || isUnboundStep<FirstMotionLeaf<Motion_>>,
        ModelsMethod,
        methods::Linearized>;
};

/** \returns ModelsMethod, the method of a filter whose switching model's modes are fixed-step and
 * each predict by their own
 */
template <typename ModeSpace_, typename Scalar_, typename Motion_>
[[nodiscard]] ModelsMethod
predictionMethod(SwitchingModel<ModeSpace_, Scalar_, Motion_> const &, ModelsMethod method)
{
    return method;
}

/** A SwitchingModel's filter over \a Space and \a Scalar binds each of its models. */
template <typename ModeSpace_, typename Scalar_, typename Motion_, typename Space, typename Scalar>
struct BoundStepOf<SwitchingModel<ModeSpace_, Scalar_, Motion_>, Space, Scalar>
{
    /// the model
    using type = SwitchingModel<ModeSpace_, Scalar_, typename BoundMotionOf<Motion_, Space, Scalar>::type>;
};

/** A filter predicts a SwitchingModel by any method, which serves the modes that carry none of their
 * own, unless the models are fixed-step: then every mode carries its own, and the filter's method is
 * ModelsMethod.
 */
template <typename ModeSpace_, typename Scalar_, typename Motion_, typename Method>
inline constexpr bool predictsBy<SwitchingModel<ModeSpace_, Scalar_, Motion_>, Method>{
    !isFixedStep<SwitchingModel<ModeSpace_, Scalar_, Motion_>> || std::is_same_v<Method, ModelsMethod>
};

/** Whether \a Chain is a chain on a mode space, the first argument of a filter over a Hybrid. */
template <typename Chain>
inline constexpr bool isModeChain = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename ModeSpace, typename Scalar>
inline constexpr bool isModeChain<MarkovChain<ModeSpace, Scalar>> = true;

template <typename ModeSpace, typename Scalar>
inline constexpr bool isModeChain<ContinuousTimeMarkovChain<ModeSpace, Scalar>> = true;

/// \endcond

/** The SwitchingModel that a filter over a Hybrid builds from the chain \a Chain and the motion \a Motion. */
template <typename Chain, typename Motion>
using SwitchingModelOf = SwitchingModel<typename Chain::SpaceType, typename Chain::Scalar, Motion>;

}  // namespace internal

/// \cond DEDUCTION_GUIDES
template <typename ModeSpace, typename Scalar, typename Motion>
SwitchingModel(MarkovChain<ModeSpace, Scalar>, Motion) -> SwitchingModel<ModeSpace, Scalar, Motion>;

template <typename ModeSpace, typename Scalar, typename Motion>
SwitchingModel(ContinuousTimeMarkovChain<ModeSpace, Scalar>, Motion) -> SwitchingModel<ModeSpace, Scalar, Motion>;

/// \endcond

namespace internal
{

/** Whether \a M is a measurement a filter over a Hybrid updates its modes by: a Measurement, whose noise
 * may be by mode axis (ModeNoise), or a ByMode of them, a measurement per value of a mode axis [by-mode].
 */
template <typename M>
inline constexpr bool isModeMeasurement = Measurement<M>;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename P, typename... Alternatives>
inline constexpr bool isModeMeasurement<ByMode<P, Alternatives...>> =
    (isModeMeasurement<std::remove_cvref_t<typename Alternatives::PartType>> && ...);

/// \endcond

/** A measurement a filter over a Hybrid updates its modes by (see isModeMeasurement). */
template <typename M>
concept ModeMeasurement = isModeMeasurement<std::remove_cvref_t<M>>;

/** A measurement the filters weighting points take (see WeighsPoints), or a ByMode of them, which those
 * whose points each hold a hybrid take [by-mode].
 */
template <typename M>
concept WeighsPointsOrModes = WeighsPoints<M> || (isByModePart<M> && ModeMeasurement<M>);

/** Whether \a M is a ByMode of measurements, a measurement per value of a mode axis. */
template <typename M>
inline constexpr bool isMeasurementByMode = isByModePart<M> && ModeMeasurement<M>;

}  // namespace internal

/** \ingroup Models
 * \returns a measurement per value of the mode axis of \a models, a ByMode of measurement models:
 * each model observed by \a method with the observation \a observation and the call's \a args, as
 * observe() observes one, and serving the values its alternative served [by-mode].
 *
 * \code
 * auto const sensor = byMode<SensorKind>(when<Radar>(radar), when<Lidar>(lidar));
 * auto const updated = filter.update(observe(methods::unscented(), sensor, z));
 * \endcode
 */
template <TransformMethod Method, typename P, typename... Alternatives, MatrixExpression Z, typename... Args>
[[nodiscard]] auto
observe(Method method, ByMode<P, Alternatives...> const & models, Z const & observation, Args const &... args)
{
    return models.map([&](auto const & model) {
        return observe(method, model, observation, args...);
    });
}

/** \ingroup Models
 * \returns observe(method, models, observation, args...) with the method each model's kind gives
 * [by-mode]: <tt>observe(byMode<SensorKind>(when<Radar>(radar), when<Lidar>(lidar)), z)</tt>
 */
template <typename P, typename... Alternatives, MatrixExpression Z, typename... Args>
[[nodiscard]] auto
observe(ByMode<P, Alternatives...> const & models, Z const & observation, Args const &... args)
{
    return models.map([&](auto const & model) {
        return observe(model, observation, args...);
    });
}

}  // namespace Emrys
