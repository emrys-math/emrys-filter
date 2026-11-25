#pragma once

/** \file
 * GpbRecursion and GpbFilter (roadmap step 17zn; decisions.md [switching-models],
 * [filters-and-recursions]): the generalized pseudo-Bayesian (GPB) filter of any order, as a Hybrid with a component
 * per sequence of the last few modes (ModeSequence), predicted by merging the sequences that differ only in their
 * oldest mode and moving each merged component through every mode's model, and updated per sequence by
 * HybridUpdateSteps, which the ImmRecursion shares. Read AmmFilter.h next.
 */

#include <concepts>
#include <cstddef>
#include <expected>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/ModeSequence.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief The generalized pseudo-Bayesian (GPB) filter's steps of order \a Order_, without the state:
 * a switching process model, the method that predicts the modes whose models carry none of their
 * own, and the scratch the components are predicted and updated into, acting on a Hybrid passed in
 * that holds a component per sequence of the last \a Order_ modes, a ModeSequence (decisions.md
 * [filters-and-recursions]).
 *
 * With \f$ M \f$ modes the state holds \f$ M^N \f$ components, one per sequence
 * \f$ (m_{k-N+1}, \dots, m_k) \f$ of the last \f$ N \f$ modes, each the state's distribution given
 * that sequence, and a weight per sequence. predict() first merges by moments the \f$ M \f$
 * components whose sequences differ only in their oldest mode, weighted by their sequences' weights,
 * which leaves \f$ M^{N-1} \f$; it then moves each merged component through every mode's process
 * model, by the model's own method where it carries one (withMethod(), FixedStepModel) and by the
 * recursion's otherwise [switching-models], which gives the \f$ M^N \f$ components of the sequences
 * one step on. The sequence that drops \f$ m_{k-N+1} \f$ and appends \f$ j \f$ gets the weight
 * \f$ \sum_{m_{k-N+1}} w(m_{k-N+1}, \dots, m_k) \, \Pi_{m_k j} \f$, with \f$ \Pi \f$ the chain over
 * the step. update(), HybridUpdateSteps's, updates each sequence's component by the measurement of
 * the sequence's current mode and multiplies the sequence's weight by its likelihood.
 *
 * The components may be in square-root form (decisions.md [gaussian-forms]), as an ImmRecursion's
 * may: the merge collapses factors by triangularization (see ComponentTraits), each merged copy is
 * predicted in place, so predict() returns a \c std::expected, and the policy is Collapse alone.
 *
 * The generalized pseudo-Bayesian algorithm of order \f$ N \f$, GPB(N), of
 * \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", as given in
 * \ref Hashimzade2024 "Hashimzade et al. (2024)", §2.2.2, Algorithm 1: \f$ M^N \f$ forecasts, their
 * updates, and the merge of the \f$ M^N \f$ outputs into \f$ M^{N-1} \f$. It departs in when it
 * merges. The source merges after each update and keeps the \f$ M^{N-1} \f$ merged components. This
 * recursion keeps the \f$ M^N \f$ outputs and merges them at the start of the next prediction, so
 * the state between a prediction and an update is the predicted distribution, as every filter's is,
 * and a second update before the next prediction conditions each sequence's component again. With
 * one update between predictions the two give the same components at every prediction and the same
 * mean and covariance over all sequences after every update, since merging by moments in two stages
 * keeps what merging at once keeps.
 *
 * \tparam Component_  the component given one sequence (see Component), such as a Gaussian
 * \tparam Model_      a SwitchingModel over the mode space, of process models whose blocks
 *                     partition the state space; with fixed-step models, predict() takes no time step
 * \tparam Order_      the order \f$ N \f$ of the algorithm, the number of modes in a sequence
 * \tparam Method_     the prediction method of every mode whose model carries none of its own
 *                    (see withMethod()); ModelsMethod with fixed-step models, which all carry one
 * \tparam Allocator_  the hybrid's allocator (see \ref Allocation)
 *
 * \sa GpbFilter, ModeSequence, HybridUpdateSteps, ImmRecursion
 */
template <
    Component Component_,
    typename Model_,
    std::size_t Order_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator>
class GpbRecursion
    : public HybridUpdateSteps<Component_, ModeSequence<typename Model_::ModeSpaceType, Order_>, Allocator_>
{
    static_assert(
        internal::predictsBy<Model_, Method_>,
        "a filter with fixed-step models predicts each mode by its model's own method, so it takes "
        "ModelsMethod, the default, as its method"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

    /** the update, which it shares with the other recursions over a hybrid */
    using UpdateSteps = HybridUpdateSteps<Component_, ModeSequence<typename Model_::ModeSpaceType, Order_>, Allocator_>;

    /// whether the components are a square-root form, whose steps are FormSteps.h's
    static constexpr bool IsForm = internal::isOtherGaussianForm<Params<Component_>>;

public:

    using Scalar = typename Params<Component_>::Scalar;             ///< the scalar type
    using SpaceType = typename Params<Component_>::SpaceType;       ///< the state space
    using ModelType = Model_;                                       ///< the switching model
    using MethodType = Method_;                                     ///< the prediction method
    using ModeSpaceType = typename Model_::ModeSpaceType;           ///< the discrete mode space
    using SequenceSpaceType = ModeSequence<ModeSpaceType, Order_>;  ///< the sequences of the last modes
    using ComponentType = Params<Component_>;                       ///< one sequence's component
    using allocator_type = Allocator_;                              ///< the scratch's allocator
    static constexpr std::size_t Order = Order_;                    ///< the order: the modes in a sequence

    /** the state the steps act on */
    using StateType = HybridParams<Component_, SequenceSpaceType, Allocator_>;

    static constexpr bool Reentrant = false;  ///< the steps write the scratch, so they run one at a time

    /** The recursion with switching model \a model and prediction method \a method, which the
     * modes whose models carry none of their own predict by, with scratch for every sequence from
     * \a allocator. Fixed-step models carry their own method, so there is none to give.
     */
    explicit GpbRecursion(Model_ model, MethodFor<Model_, Method_> method = {}, Allocator_ const & allocator = {})
        : UpdateSteps{allocator}
        , m_model{std::move(model)}
        , m_method{internal::predictionMethod(m_model, std::move(method))}
        , m_predicted(ComponentAllocator(allocator))
    {
        reserveScratch();
    }

    /** \returns the allocator of the scratch */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_predicted.get_allocator());
    }

    /** Replaces the scratch with storage from \a allocator, when a state from another allocator
     * arrives.
     */
    void
    resetScratch(Allocator_ const & allocator)
    {
        m_predicted = ConditionalsType(ComponentAllocator(allocator));
        UpdateSteps::resetPosterior(allocator);
        reserveScratch();
    }

    /** \returns the switching model */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_model;
    }

    /** \returns the switching model, to change its chain or its models' parameters */
    [[nodiscard]] Model_ &
    model()
    {
        return m_model;
    }

    /** \returns the prediction method of the modes whose models carry none of their own;
     * ModelsMethod with fixed-step models
     */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_method;
    }

    /** \returns the discrete mode space, the model's */
    [[nodiscard]] ModeSpaceType const &
    modeSpace() const
    {
        return m_model.modeSpace();
    }

    /** \returns the sequences of the last modes, the mode space of the states the steps act on */
    [[nodiscard]] SequenceSpaceType
    sequenceSpace() const
    {
        return SequenceSpaceType{m_model.modeSpace()};
    }

    /** \returns whether every model's writes partition \a space, which a state must be in */
    [[nodiscard]] bool
    partitions(SpaceType const & space) const
    {
        return m_model.partitions(space);
    }

    /** Merges \a state's components whose sequences differ only in their oldest mode, moves each
     * merged component by \a dt, any TimeStep (see \ref Time), through every mode's process model,
     * with \a inputs passed as in Emrys::predict(), and gives each sequence one step on its weight by
     * the chain over \a dt, \f$ \Pi = e^{Q \Delta t} \f$.
     *
     * \returns nothing for Gaussians; for a square-root form, a \c std::expected with the error of
     *          the first sequence that could not take the step, which keeps the merged component it
     *          was to move from while the others move
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(StateType & state, TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        Scalar const step = internal::toScalarTime<Scalar>(dt);
        auto const chainOverStep = m_model.transition(step);
        if constexpr (IsForm)
        {
            return moveSequences(
                state,
                chainOverStep,
                [&](ComponentType & moved, auto const & model, auto const & method) {
                    return internal::predictInPlace(method, moved, model, step, inputs...);
                }
            );
        }
        else
        {
            moveSequences(
                state,
                chainOverStep,
                [&](ComponentType const & merged, auto const & model, auto const & method) {
                    return Emrys::predict(method, merged, model, step, inputs...);
                }
            );
        }
    }

    /** Merges \a state's components as predict(state, dt, inputs...) does, moves each merged
     * component by every mode's fixed-step model's step, by that model's method, with \a inputs
     * passed as in Emrys::predict(), and gives each sequence one step on its weight by one step of
     * the chain, returning as predict(state, dt, inputs...) does.
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(StateType & state, Inputs const &... inputs)
    {
        auto const & chainOverStep = m_model.transition();
        if constexpr (IsForm)
        {
            return moveSequences(state, chainOverStep, [&](ComponentType & moved, auto const & model, auto const &) {
                return internal::predictInPlace(moved, model, inputs...);
            });
        }
        else
        {
            moveSequences(state, chainOverStep, [&](ComponentType const & merged, auto const & model, auto const &) {
                return Emrys::predict(merged, model, inputs...);
            });
        }
    }

private:

    using ConditionalsType = typename StateType::ConditionalsType;         ///< the sequences' components, as stored
    using ComponentAllocator = typename ConditionalsType::allocator_type;  ///< allocates them

    /** Reserves the scratch for every sequence. */
    void
    reserveScratch()
    {
        auto const sequences = std::size_t(sequenceSpace().cardinality());
        m_predicted.reserve(sequences);
        UpdateSteps::reservePosterior(sequences);
    }

    /** The body of both forms of predict(): merges the components of \a state whose sequences differ
     * only in their oldest mode, moves each merged component by \a predictOne with every mode's
     * process model and method, and sets the weights of the sequences one step on by \a chain, the
     * chain over the step. For Gaussians \a predictOne returns the predicted component. For a
     * square-root form it moves a copy of the merged component in place, in the scratch, and returns
     * a \c std::expected; a sequence it fails on keeps the merged component.
     *
     * The sequences with flat index \f$ o M^{N-1} + r \f$, for each oldest mode \f$ o \f$, differ
     * only in it, and the sequence that follows them into the mode \f$ j \f$ has the flat index
     * \f$ r M + j \f$, so the predicted components are built in flat-index order.
     *
     * \returns nothing, or for a square-root form the first failed sequence's error
     */
    template <typename Chain, typename PredictOne>
    [[nodiscard]] auto
    moveSequences(StateType & state, Chain const & chain, PredictOne const & predictOne)
    {
        SequenceSpaceType const sequences = state.modeSpace();
        Eigen::Index const count = state.size();
        Eigen::Index const modes = sequences.current().cardinality();
        Eigen::Index const kept = count / modes;
        auto const & transitions = chain.matrix();
        typename StateType::ModesType::ArrayType const logPrior = state.modes().logWeights();
        auto logPredicted = typename StateType::ModesType::ArrayType{};
        logPredicted.resize(count);
        [[maybe_unused]] auto failure = std::expected<void, NumericError>{};
        m_predicted.clear();
        for (Eigen::Index rest = 0; rest < kept; ++rest)
        {
            // The sequences that end in the modes of rest and differ in their oldest mode.
            auto const member = [&](Eigen::Index oldest) {
                return oldest * kept + rest;
            };
            auto total = LogSumExp<Scalar>{};
            for (Eigen::Index oldest = 0; oldest < modes; ++oldest)
            {
                total.add(logPrior(member(oldest)));
            }
            // Sequences of no weight merge into their components' plain average, which is never weighted.
            bool const weighted = total.value() > -Eigen::NumTraits<Scalar>::infinity();
            ComponentType const merged = ComponentTraits<ComponentType>::collapse(
                modes,
                [&](Eigen::Index oldest) -> ComponentType const & {
                    return state.conditional(member(oldest));
                },
                [&](Eigen::Index oldest) {
                    return weighted ? logPrior(member(oldest)) : Scalar(0);
                }
            );
            for (Eigen::Index next = 0; next < modes; ++next)
            {
                auto reached = LogSumExp<Scalar>{};
                for (Eigen::Index oldest = 0; oldest < modes; ++oldest)
                {
                    Eigen::Index const from = member(oldest);
                    reached.add(logPrior(from) + Eigen::numext::log(transitions(sequences.currentOf(from), next)));
                }
                logPredicted(sequences.following(member(0), next)) = reached.value();
                m_model.visit(next, m_method, [&](auto const & model, auto const & method) {
                    if constexpr (IsForm)
                    {
                        m_predicted.push_back(merged);
                        if (auto const moved = predictOne(m_predicted.back(), model, method); !moved && failure)
                        {
                            failure = moved;
                        }
                    }
                    else
                    {
                        m_predicted.push_back(predictOne(merged, model, method));
                    }
                });
            }
        }
        state.swapConditionals(m_predicted);
        state.setModes(typename StateType::ModesType{sequences, logPredicted});
        if constexpr (IsForm)
        {
            return failure;
        }
    }

    Model_ m_model;                                   ///< the switching model
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method_ m_method;  ///< the prediction method

    // Scratch, exchanged with the state's components; the base holds the posterior ones.
    ConditionalsType m_predicted;  ///< the merged and predicted components
};

/** \ingroup Filters
 * \brief A generalized pseudo-Bayesian (GPB) filter of order \a Order_: a hybrid state with a
 * component per sequence of the last \a Order_ modes, a ModeSequence, and the GpbRecursion that
 * predicts it through a switching process model and updates it per sequence. Of order two it is
 * GPB2, which econometrics knows as Kim's filter, and of order one GPB1.
 *
 * A system whose mode switches has, after \f$ k \f$ steps, a %Gaussian for each of the \f$ M^k \f$
 * sequences of modes it may have taken, and the exact filter keeps them all. This filter keeps the
 * \f$ M^N \f$ that the last \f$ N \f$ modes distinguish, and replaces the sequences that agree in
 * them by one %Gaussian with their mean and covariance. A longer sequence loses less and costs
 * \f$ M \f$ times as many predictions and updates per mode added: \f$ M^N \f$ of each per step.
 *
 * predict() merges the components whose sequences differ only in their oldest mode and moves each
 * merged component through every mode's process model, and update() updates each sequence's
 * component by the measurement of its current mode and multiplies the sequence's weight by its
 * likelihood (see GpbRecursion). A measurement may depend on the mode as an ImmFilter's
 * may, by noise by mode axis or as a measurement per value of a mode axis [by-mode]; it names the
 * axes of the mode space and reads the current mode of each sequence.
 *
 * \code
 * auto const prior = HybridGaussianParams<State, Maneuver>{CategoricalParams<Maneuver>{}, initial};
 * auto filter = GpbFilter{chain, byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(turn)),
 *                         overModeSequences<2>(prior)};            // the last two maneuvers: GPB2
 * filter.predict(dt);
 * auto const updated = filter.update(fix);
 * filter.state().toDensity()->collapsed();                         // the estimate as one GaussianParams
 * filter.state().modes().marginal<Current<Maneuver>>();            // the weights of the mode now
 * filter.state().toDensity()->collapse<Current<Maneuver>>();       // a Gaussian per mode now, as an IMM keeps
 * \endcode
 *
 * The state is a Hybrid, so it is read as an ImmFilter's is. Its mode space is the product of
 * \a Order_ copies of the model's, the axis of the mode \a k steps back tagged
 * <tt>Lagged<ModeSpace, k></tt>, so a value of the current mode is named through
 * <tt>Current<ModeSpace></tt>. overModeSequences() makes the initial state from a hybrid over the
 * current mode.
 *
 * It stands beside the ImmFilter as follows. Both merge components by moments. This filter merges
 * the components that differ in the oldest mode of a sequence, weighted by the sequences' weights.
 * The IMM keeps one component per mode and merges all of them once for each mode the chain may
 * move to, weighted by the weights times the chain's probability of moving there. With sequences of
 * one mode this filter merges all the components into one, whatever mode comes next, which costs
 * what the IMM costs and uses less: the IMM is the better filter of that size. With sequences of two
 * or more modes the chain's probability of the next mode is the same for every component merged,
 * since they share their current mode, so merging once for each next mode would change nothing.
 *
 * A sequence whose update the policy rejects keeps its prior component and gets weight zero, as an
 * IMM's mode does. The predicted and posterior components are built in scratch from the state's
 * allocator and exchanged with the state's, and update(measurement, policy, result) reuses the
 * result's storage (see \ref Allocation). The steps take no execution policy. No smoother takes its
 * steps: it has no predictJoint().
 *
 * The algorithm is that of \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", as given in
 * \ref Hashimzade2024 "Hashimzade et al. (2024)", §2.2.2, Algorithm 1, who show in §2.2 that the
 * filter of \ref Kim1994 "Kim (1994)" is its order 2. GpbRecursion says where the code
 * departs from it.
 *
 * \tparam Component_  the component given one sequence (see Component), a Gaussian over the state
 *                     space (see \ref Spaces)
 * \tparam Model_      a SwitchingModel over the mode space, of process models whose blocks
 *                     partition the state space; with fixed-step models, predict() takes no time step
 * \tparam Order_      the order of the algorithm, the number of modes in a sequence
 * \tparam Method_     the prediction method of every mode whose model carries none of its own
 *                    (see withMethod()); ModelsMethod with fixed-step models, which all carry one
 * \tparam Allocator_  the state's allocator (see \ref Allocation)
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one, such as RepairCovariance
 *
 * \sa ModeSequence, GpbRecursion, overModeSequences(), ImmFilter, SwitchingModel, \ref Filters
 */
template <
    Component Component_,
    typename Model_,
    std::size_t Order_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class GpbFilter
    : public internal::HeldUpkeep<GpbFilter<Component_, Model_, Order_, Method_, Allocator_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<GpbFilter, Upkeep_>;  ///< what holds the upkeep

public:

    /** the steps it runs (see Recursion) */
    using RecursionType = GpbRecursion<Component_, Model_, Order_, Method_, Allocator_>;

    using Scalar = typename RecursionType::Scalar;                        ///< the scalar type
    using SpaceType = typename RecursionType::SpaceType;                  ///< the state space
    using ModelType = Model_;                                             ///< the process model
    using MethodType = Method_;                                           ///< the prediction method
    using ModeSpaceType = typename Model_::ModeSpaceType;                 ///< the discrete mode space
    using SequenceSpaceType = typename RecursionType::SequenceSpaceType;  ///< the sequences of the last modes
    using StateType = typename RecursionType::StateType;                  ///< the state the filter owns
    using ComponentType = Params<Component_>;                             ///< one component of the state
    using allocator_type = Allocator_;                                    ///< the state's allocator
    static constexpr std::size_t Order = Order_;                          ///< the order: the modes in a sequence

    /** A filter with switching model \a model, initial state \a state, prediction method \a method,
     * and upkeep \a upkeep, which maintain() runs. Fixed-step models carry their own method, so there
     * is none to give; each mode predicts by its own model's. The scratch comes from the state's
     * allocator.
     */
    GpbFilter(
        Model_ model,
        StateType state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : UpkeepBase{std::move(upkeep)}
        , m_state{std::move(state)}
        , m_recursion{std::move(model), std::move(method), m_state.get_allocator()}
    {
        checkState(m_state);
    }

    /** A filter with switching model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    GpbFilter(Model_ model, StateType state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : GpbFilter{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}

    /** As the constructor that takes a switching model, with the SwitchingModel of the chain \a chain on
     * the modes and the motion \a motion, one process model or a byMode() of them [by-mode], as an
     * ImmFilter takes them. The arguments after them are that constructor's.
     */
    template <typename Chain, typename Motion, typename... Rest>
        requires internal::isModeChain<Chain>
    GpbFilter(Chain chain, Motion motion, StateType state, Rest &&... rest)
        : GpbFilter{
              Model_(SwitchingModel{std::move(chain), std::move(motion)}),
              std::move(state),
              std::forward<Rest>(rest)...
          }
    {}

    /** \returns the allocator of the state and scratch */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_state.get_allocator();
    }

    /** \returns the current state: the sequences' weights and each sequence's component */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the current state, to change where it is between steps, as upkeep does. It must keep
     * its space, its sequences, and its allocator; setState() replaces it with another, and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state, as when reinitializing, with one over the sequences of the model's modes
     * whose space every process model's writes partition, as the constructor's must. The scratch
     * follows the new state's allocator.
     */
    void
    setState(StateType state)
    {
        checkState(state);
        m_state = std::move(state);
        if (m_state.get_allocator() != m_recursion.get_allocator())
        {
            // The assignment carried the new state's allocator in, so the scratch follows it.
            m_recursion.resetScratch(m_state.get_allocator());
        }
    }

    /** \returns the recursion: the model, method, and scratch the steps run with */
    [[nodiscard]] RecursionType const &
    recursion() const
    {
        return m_recursion;
    }

    /** \returns the switching model */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_recursion.model();
    }

    /** \returns the switching model, to change its chain or its models' parameters */
    [[nodiscard]] Model_ &
    model()
    {
        return m_recursion.model();
    }

    /** \returns the prediction method of the modes whose models carry none of their own;
     * ModelsMethod with fixed-step models
     */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_recursion.method();
    }

    /** Merges the components whose sequences differ only in their oldest mode, moves each merged
     * component by \a dt, any TimeStep (see \ref Time), through every mode's process model, with
     * \a inputs passed as in Emrys::predict(), and gives each sequence one step on its weight by the
     * chain over \a dt.
     *
     * \returns nothing for Gaussians; for a square-root form, a \c std::expected that is empty
     *          when a sequence could not take the step (see GpbRecursion::predict())
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        return m_recursion.predict(m_state, internal::toScalarTime<Scalar>(dt), inputs...);
    }

    /** Merges the components as predict(dt, inputs...) does and moves each merged component by every
     * mode's fixed-step model's step, by that model's method, with \a inputs passed as in
     * Emrys::predict(), the weights moving one step of the chain, returning as predict(dt, inputs...)
     * does.
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] auto
    predict(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        return m_recursion.predict(m_state, inputs...);
    }

    /** Moves the state forward by \a steps of the fixed-step models, as that many calls of
     * predict(inputs...) do, with \a inputs at every step. For a square-root form it stops at the
     * first step a sequence could not take, returning its error.
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(Steps steps, Inputs const &... inputs)
    {
        EMRYS_FILTER_ASSERT(steps.count >= 0 && "the number of steps must be nonnegative");
        using Result = decltype(m_recursion.predict(m_state, inputs...));
        if constexpr (std::is_void_v<Result>)
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                m_recursion.predict(m_state, inputs...);
            }
        }
        else
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                if (Result result = m_recursion.predict(m_state, inputs...); !result)
                {
                    return result;
                }
            }
            return Result{};
        }
    }

    /** Conditions each sequence's component on \a measurement, by the measurement's own method,
     * reduces its branches with \a policy (Collapse, Select, or a Gate), and multiplies each
     * sequence's weight by its likelihood. The measurement may depend on the mode, by its noise or as
     * a measurement per value of a mode axis, a ByMode: each sequence reads its current mode
     * (see HybridUpdateSteps::update()) [by-mode].
     *
     * \returns each sequence's update before reduction, in \c modes, the observation's
     * log-likelihood, and whether it was accepted (see ImmFilterUpdate), with the updates from the
     * state's allocator, in a \c std::expected that is empty, with the state unchanged, when a
     * sequence's update failed
     */
    template <internal::ModeMeasurement M, typename Policy = reductions::Collapse>
    [[nodiscard]] auto
    update(M const & measurement, Policy const & policy = {})
    {
        return m_recursion.update(m_state, measurement, policy);
    }

    /** As update(measurement, policy), but into \a result, whose storage is reused: with the same
     * result at every step, an update allocates nothing once it has been as large.
     *
     * \returns nothing, or the error update(measurement, policy) would return
     */
    template <internal::ModeMeasurement M, typename Policy, typename ModeUpdate, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(M const & measurement, Policy const & policy, ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result)
    {
        return m_recursion.update(m_state, measurement, policy, result);
    }

    /** \returns an empty result for update(measurement, policy, result) with a measurement of type
     * \a M, \c const and references aside, a Measurement or a ByMode of them: the update inside the
     * \c std::expected that update(measurement) returns, from the state's allocator, with room for
     * an update of every sequence
     */
    template <internal::ModeMeasurement M>
    [[nodiscard]] auto
    emptyUpdate() const
    {
        return m_recursion.template emptyUpdate<std::remove_cvref_t<M>>(get_allocator(), m_state.size());
    }

private:

    /** Checks that \a state is over the sequences of the model's modes and that every process
     * model's writes partition its space: the check of every constructor and of setState().
     */
    void
    checkState(StateType const & state) const
    {
        EMRYS_FILTER_ASSERT(
            m_recursion.model().modeSpace() == state.modeSpace().current()
            && "the model and the state have different modes"
        );
        internal::assertPartitions(
            m_recursion.model(),
            state.space(),
            "every process model must write every coordinate of the state space once"
        );
    }

    StateType m_state;          ///< the state estimate
    RecursionType m_recursion;  ///< the model, method, and scratch the steps run with
};

/// \cond DEDUCTION_GUIDES
// Each guide lists its parameters in the class's order, as MSVC needs to deduce through them.
template <
    Component C,
    typename Model,
    std::size_t Order,
    typename Method = DefaultMethodFor<Model>,
    typename Allocator,
    typename ModeSpace>
    requires(!internal::isModeChain<Model> && !internal::UpkeepArgument<Method>)
GpbFilter(Model, HybridParams<C, ModeSequence<ModeSpace, Order>, Allocator>, Method = {}) -> GpbFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Order,
    Method,
    Allocator>;

template <
    Component C,
    typename Chain,
    typename Motion,
    std::size_t Order,
    typename Method = DefaultMethodFor<internal::SwitchingModelOf<Chain, Motion>>,
    typename Allocator,
    typename ModeSpace>
    requires(internal::isModeChain<Chain> && !internal::UpkeepArgument<Method>)
GpbFilter(Chain, Motion, HybridParams<C, ModeSequence<ModeSpace, Order>, Allocator>, Method = {}) -> GpbFilter<
    C,
    internal::
        BoundStep<internal::SwitchingModelOf<Chain, Motion>, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Order,
    Method,
    Allocator>;

template <Component C, typename Model, std::size_t Order, typename Allocator, typename ModeSpace, typename U>
    requires(!internal::isModeChain<Model> && internal::UpkeepArgument<U>)
GpbFilter(Model, HybridParams<C, ModeSequence<ModeSpace, Order>, Allocator>, U) -> GpbFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Order,
    DefaultMethodFor<Model>,
    Allocator,
    U>;

template <
    Component C,
    typename Chain,
    typename Motion,
    std::size_t Order,
    typename Allocator,
    typename ModeSpace,
    typename U>
    requires(internal::isModeChain<Chain> && internal::UpkeepArgument<U>)
GpbFilter(Chain, Motion, HybridParams<C, ModeSequence<ModeSpace, Order>, Allocator>, U) -> GpbFilter<
    C,
    internal::
        BoundStep<internal::SwitchingModelOf<Chain, Motion>, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Order,
    DefaultMethodFor<internal::SwitchingModelOf<Chain, Motion>>,
    Allocator,
    U>;

template <
    Component C,
    typename Model,
    std::size_t Order,
    typename Method,
    typename Allocator,
    typename ModeSpace,
    typename U>
    requires(!internal::isModeChain<Model>)
GpbFilter(Model, HybridParams<C, ModeSequence<ModeSpace, Order>, Allocator>, Method, U) -> GpbFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Order,
    Method,
    Allocator,
    U>;

template <
    Component C,
    typename Chain,
    typename Motion,
    std::size_t Order,
    typename Method,
    typename Allocator,
    typename ModeSpace,
    typename U>
    requires internal::isModeChain<Chain>
GpbFilter(Chain, Motion, HybridParams<C, ModeSequence<ModeSpace, Order>, Allocator>, Method, U) -> GpbFilter<
    C,
    internal::
        BoundStep<internal::SwitchingModelOf<Chain, Motion>, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Order,
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
template <
    typename Component_,
    typename Model_,
    std::size_t Order_,
    typename Method_,
    typename Allocator_,
    typename Upkeep_>
void
describe(
    std::ostream & stream,
    GpbFilter<Component_, Model_, Order_, Method_, Allocator_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "generalized pseudo-Bayesian filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
