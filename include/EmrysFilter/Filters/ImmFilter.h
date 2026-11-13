#pragma once

/** \file
 * ImmRecursion and ImmFilter (decisions.md [switching-models],
 * [imm], [filters-and-recursions]): a Hybrid state mixed through a Markov chain, predicted through each mode's model,
 * and updated per mode, with Markov outlier states when a measurement's noise is by mode axis (ModeNoise) and a
 * measurement per mode through a SwitchingMeasurement; the recursion that does this to a hybrid
 * passed in, keeping the scratch it mixes and updates into, the filter that owns a hybrid and runs the recursion on it.
 * predictJoint() records a step as a HybridPredictedJoint for the IMM smoother in HybridGaussianHistory.h [smoothing],
 * and a MixtureFilter of hybrids runs the recursion on each of its components. Read AmmFilter.h next; the design review
 * is docs/guide/imm-filter.md.
 */

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief What an IMM filter's update did: each mode's update before reduction, the
 * observation's log-likelihood over every mode, and whether any mode accepted the measurement.
 *
 * \c modes[m] is the update of mode \a m's component (see FilterUpdate), with its
 * log-likelihood \f$ \log p(z \mid m) \f$ and its branches labeled with \c mode = \a m. Its
 * \c accepted is false when the policy, such as a Gate, rejected the measurement in that mode.
 * When \c accepted is false here, every mode rejected it and the state kept its prior.
 *
 * The modes' updates come from \a Allocator. Passing the same result, made from the allocator, to each
 * ImmFilter::update(measurement, policy, result) reuses their storage, and a mode's update of
 * mixture noise (a ReusableFilterUpdate) keeps its scratch, so an update allocates nothing once
 * the result has been as large (see \ref Allocation).
 *
 * \tparam ModeUpdate  one mode's update: a FilterUpdate, or a ReusableFilterUpdate for mixture noise
 * \tparam Scalar_     the scalar type
 * \tparam Allocator   where the modes' updates come from
 */
template <typename ModeUpdate, typename Scalar_ = double, typename Allocator = DefaultAllocator>
struct ImmFilterUpdate
{
    using Scalar = Scalar_;                                           ///< the scalar type
    using ModeUpdateType = ModeUpdate;                                ///< one mode's update
    using ModeUpdatesType = internal::Buffer<ModeUpdate, Allocator>;  ///< the modes' container

    ImmFilterUpdate() = default;  ///< an empty result, from a default-constructed allocator

    /** An empty result whose modes' updates come from \a allocator. */
    explicit ImmFilterUpdate(Allocator const & allocator)
        : modes(internal::ReBound<Allocator, ModeUpdate>(allocator))
    {}

    /** \returns the allocator the modes' updates come from */
    [[nodiscard]] Allocator
    get_allocator() const
    {
        return Allocator(modes.get_allocator());
    }

    /** \returns \f$ \log p(z) = \log \sum_m P(m) \, p(z \mid m) \f$, with the predicted mode probabilities */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return m_logLikelihood;
    }

    ModeUpdatesType modes;  ///< one per mode, in flat-index order
    bool accepted = false;  ///< whether any mode accepted the measurement

private:

    friend struct internal::UpdateAccess;

    Scalar m_logLikelihood = 0;  ///< the observation's log-likelihood over every mode
};

namespace internal
{

/// \cond INTERNAL_SPECIALIZATIONS
template <typename ModeUpdate, typename Scalar, typename Old, typename Allocator>
struct RebindUpdateAllocatorOf<ImmFilterUpdate<ModeUpdate, Scalar, Old>, Allocator>
{
    using type = ImmFilterUpdate<RebindUpdateAllocator<ModeUpdate, Allocator>, Scalar, Allocator>;
};

/// \endcond

}  // namespace internal

/** \ingroup Filters
 * \brief One IMM prediction as its smoother needs it: the mode probabilities the step started
 * from, each mode's %Gaussian predicted through its own model as a PredictedJoint, and the chain's
 * transition matrix.
 *
 * ImmFilter::predictJoint() fills it before predicting, and a
 * HybridGaussianHistory records it (decisions.md [smoothing]). \c joints[j] is mode \a j's filtered
 * %Gaussian \f$ p(x_t \mid r_t = j, z_{1:t}) \f$ moved through mode \a j's process model, with
 * its cross-covariance, so that its \c filtered is the mode's %Gaussian before the step; it is
 * not the IMM's own prediction of mode \a j, which starts from the mixed %Gaussian. This is the
 * mode-matched prediction the IMM smoother's backward step is written against (see
 * HybridGaussianHistory). \c throughOtherModels holds mode \a j's filtered %Gaussian moved through
 * each other mode \a i's model, \f$ p(x_{t+1} \mid r_t = j, r_{t+1} = i, z_{1:t}) \f$, from which
 * the smoother weighs the modes (see predictedThrough()).
 *
 * The joints come from \a Allocator_. Passing the same joint to each
 * ImmFilter::predictJoint(dt, joint, inputs...) reuses their storage.
 *
 * \tparam Space_      the continuous state space
 * \tparam ModeSpace_  the discrete mode space
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the joints (see \ref Allocation)
 */
template <Space Space_, DiscreteSpace ModeSpace_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
struct HybridPredictedJoint
{
    using Scalar = Scalar_;                                                         ///< the scalar type
    using SpaceType = Space_;                                                       ///< the continuous state space
    using ModeSpaceType = ModeSpace_;                                               ///< the discrete mode space
    using ModesType = CategoricalParams<ModeSpace_, Scalar_>;                       ///< the mode weights' type
    using JointType = PredictedJoint<Space_, Scalar_>;                              ///< one mode's joint
    using JointsType = internal::Buffer<JointType, Allocator_>;                     ///< the modes' joints
    using TransitionsType = typename MarkovChain<ModeSpace_, Scalar_>::MatrixType;  ///< the chain's matrix
    using PredictionsType = internal::Buffer<GaussianParams<Space_, Scalar_>, Allocator_>;  ///< predictions alone

    ModesType modes;              ///< \f$ \mu_{t \mid t} \f$, the mode probabilities before the step
    JointsType joints;            ///< one per mode, in flat-index order
    TransitionsType transitions;  ///< \f$ \Pi \f$, with \f$ \Pi_{ij} = P(r_{t+1} = j \mid r_t = i) \f$

    /** Mode \a j's filtered %Gaussian moved through mode \a i's model for every \a i other than
     * \a j, \a j major and \a i minor (see predictedThrough())
     */
    PredictionsType throughOtherModels;

    /** \returns mode \a from's filtered %Gaussian moved through mode \a model's process model:
     * \c joints[from].predicted when they are the same mode, and otherwise its entry of
     * \c throughOtherModels
     */
    [[nodiscard]] GaussianParams<Space_, Scalar_> const &
    predictedThrough(Eigen::Index from, Eigen::Index model) const
    {
        if (from == model)
        {
            return joints[std::size_t(from)].predicted;
        }
        Eigen::Index const count = Eigen::Index(joints.size());
        return throughOtherModels[std::size_t(from * (count - 1) + (model < from ? model : model - 1))];
    }
};

namespace internal
{

/** The type of one mode's update of \a C by \a M: the MeasurementUpdate of the covariance form's
 * branches.
 */
template <typename C, typename M, bool = isOtherGaussianForm<C>>
struct ImmMeasured
{
    /// the update's type
    using type = typename decltype(Emrys::update(std::declval<C const &>(), std::declval<M const &>()))::value_type;
};

/** The type of one mode's update of the square-root form \a C by \a M: the FilterUpdate its
 * updateInPlace() fills.
 */
template <typename C, typename M>
struct ImmMeasured<C, M, true>
{
    /// the update's type
    using type = typename decltype(updateInPlace<true>(std::declval<C &>(), std::declval<M const &>()))::value_type;
};

/** \a Lazy::type, the return type of an update that takes the reduction policy \a Policy, formed
 * only where \a Policy is no GivenBlock: MSVC substitutes a return type before it checks the
 * requires-clause that refuses a GivenBlock, and the plain update's type of a measurement that
 * reads the given block fails a \c static_assert.
 */
template <typename Lazy, typename Policy>
struct UnlessGivenBlock
{
    using type = typename Lazy::type;  ///< the update's type
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Lazy, typename Space, typename P, typename Value>
struct UnlessGivenBlock<Lazy, GivenBlock<Space, P, Value>>
{};

/// \endcond

/** The path to the mode axis whose values the noise components of the measurement \a M are, as
 * \c type: the axis of its model's ModeNoise, or \c void.
 */
template <typename M>
struct MeasurementModeAxisOf
{
    using type = void;  ///< no mode axis
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename M>
    requires(M::ModelType::template has<PolicyKind::additiveNoise>)
struct MeasurementModeAxisOf<M>
{
    /// the noise's mode axis, or \c void
    using type = NoiseModeAxis<typename M::ModelType::template Policy<PolicyKind::additiveNoise>>;
};

/// \endcond

/** Whether the measurement \a M has noise by mode axis, <tt>additiveNoise(mixture).byMode<Axis>()</tt>. */
template <typename M>
inline constexpr bool isByMode = !std::is_void_v<typename MeasurementModeAxisOf<M>::type>;

/** One mode's measurement in an update by mode axis given a block (see GivenMeasurement): \a One,
 * the measurement with the mode's noise component, and the model over the rest and the block
 * resolved against it, which refers to it and to the space held here, so a copy resolves its own.
 */
template <typename Space, typename P, typename One, typename Value>
struct GivenComponent
{
    /** \a one, resolved against \a whole, the state space, with the block named by \a P given as \a given. */
    GivenComponent(One one, Space whole, Value const & given)
        : measurement{std::move(one)}
        , space{std::move(whole)}
        , pinned{space, measurement}
        , value{given}
    {}

    /** A copy of \a other, resolved against its own measurement and space. */
    GivenComponent(GivenComponent const & other)
        : GivenComponent{other.measurement, other.space, other.value}
    {}

    One measurement;                          ///< the measurement with the mode's noise component
    Space space;                              ///< the whole state space
    PinnedMeasurement<Space, P, One> pinned;  ///< the measurement, over the rest and the given block
    Value const & value;                      ///< the given block's value
};

}  // namespace internal

/** \ingroup Filters
 * \brief The interacting multiple model (IMM) filter's steps over a hybrid state, without the
 * state: a switching process model, the method that predicts the modes whose models carry none of
 * their own, and the scratch the modes are mixed and updated into, acting on a Hybrid passed in (decisions.md
 * [filters-and-recursions]).
 *
 * predict() first mixes: the component that mode \a j starts from is the collapse of every
 * mode's component, weighted by the probability \f$ \mu_{i \mid j} \propto \Pi_{ij} \mu_i \f$
 * that the target was in mode \a i given that it is now in \a j. Each mixed component is then
 * predicted through its mode's process model, by the model's own method where it carries one
 * (withMethod(), FixedStepModel) and by the recursion's otherwise [switching-models], and the
 * mode probabilities move one step of the chain. When the chain is the identity, the mixing would give each mode its
 * own component, so predict() moves each as it is and leaves the probabilities. update() updates each mode's component
 * as a KalmanRecursion would and multiplies each mode's probability by
 * \f$ p(z \mid m) \f$.
 *
 * The components may be in square-root form (decisions.md [gaussian-forms]): the mixing collapses
 * factors by triangularization (see ComponentTraits), each mode predicts and updates in
 * place as a KalmanRecursion over the form does, so predict() returns a \c std::expected, and the
 * policy is Collapse alone, the form's update having one branch. predictJoint() is the covariance
 * form's.
 *
 * The interacting multiple model algorithm of \ref Blom1988 "Blom and Bar-Shalom (1988)", as
 * given in \ref Hashimzade2024 "Hashimzade et al. (2024)", §2.2.3, Algorithm 2 with one lag:
 * mixing, mode-matched filtering, mode probability update, and the combined estimate as the state's
 * collapsed().
 *
 * ImmFilter owns a hybrid and runs this on it; MixtureFilter runs it on each
 * hybrid of a mixture of hybrids, in order, since the scratch is one per recursion (not
 * \c Reentrant). The scratch comes from the allocator the recursion is given, which must equal
 * the hybrids' (see \ref Allocation).
 *
 * \tparam Component_  the component given one mode (see Component), such as a Gaussian
 * \tparam Model_      a SwitchingModel over the mode space, of process models whose blocks
 *                     partition the state space; with fixed-step models, predict() takes no time step
 * \tparam Method_     the prediction method of every mode whose model carries none of its own
 *                    (see withMethod()); ModelsMethod with fixed-step models, which all carry one
 * \tparam Allocator_  the hybrid's allocator (see \ref Allocation)
 *
 * \sa ImmFilter, class Hybrid
 */
template <
    Component Component_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator>
class ImmRecursion
{
    static_assert(
        internal::predictsBy<Model_, Method_>,
        "a filter with fixed-step models predicts each mode by its model's own method, so it takes "
        "ModelsMethod, the default, as its method"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

    /// whether the modes' components are a square-root form, whose steps are FormSteps.h's
    static constexpr bool IsForm = internal::isOtherGaussianForm<Params<Component_>>;

public:

    using Scalar = typename Params<Component_>::Scalar;                     ///< the scalar type
    using SpaceType = typename Params<Component_>::SpaceType;               ///< the state space
    using ModelType = Model_;                                               ///< the switching model
    using MethodType = Method_;                                             ///< the prediction method
    using ModeSpaceType = typename Model_::ModeSpaceType;                   ///< the discrete mode space
    using ComponentType = Params<Component_>;                               ///< one mode's component
    using StateType = HybridParams<Component_, ModeSpaceType, Allocator_>;  ///< the state the steps act on
    using allocator_type = Allocator_;                                      ///< the scratch's allocator

    /** what predictJoint() records */
    using JointType = HybridPredictedJoint<SpaceType, ModeSpaceType, Scalar, Allocator_>;

    static constexpr bool Reentrant = false;  ///< the steps write the scratch, so they run one at a time

    /** The MeasurementUpdate of \a M on one mode's component, or, for a square-root form, the
     * FilterUpdate of its one branch.
     */
    template <Measurement M>
    using MeasuredType = typename internal::ImmMeasured<ComponentType, M>::type;

    /** What update() with \a M fills when given a result: one branch per mode for noise by mode
     * axis, every noise component's branch otherwise.
     */
    template <Measurement M>
    using ResultType = ImmFilterUpdate<
        std::conditional_t<
            internal::isByMode<M>,
            FilterUpdate<typename MeasuredType<M>::MeasurementSpaceType, 1, Scalar>,
            std::conditional_t<
                IsForm,
                MeasuredType<M>,
                internal::FilterUpdateWithScratch<
                    SpaceType,
                    typename MeasuredType<M>::MeasurementSpaceType,
                    MeasuredType<M>::BranchCount,
                    Scalar,
                    Allocator_>>>,
        Scalar,
        Allocator_>;

    /** What update() with \a M returns: the ResultType in a \c std::expected (decisions.md [innovation-cholesky]). */
    template <Measurement M>
    using UpdateType = std::expected<ResultType<M>, NumericError>;

    /** UpdateType<M>, named without being formed (see internal::UnlessGivenBlock). */
    template <Measurement M>
    struct UpdateTypeOf
    {
        using type = UpdateType<M>;  ///< the update's type
    };

    /** What update() with the SwitchingMeasurement \a S fills when given a result: the ResultType
     * of its first measurement, which every one of its measurements shares.
     */
    template <typename S>
    using SwitchingResultType = ResultType<typename S::template MeasurementType<0>>;

    /** What update() with the SwitchingMeasurement \a S returns: the SwitchingResultType in a
     * \c std::expected.
     */
    template <typename S>
    using SwitchingUpdateType = std::expected<SwitchingResultType<S>, NumericError>;

    /** The MeasurementUpdate of \a M on one mode's %Gaussian, with the block named by the path \a P
     * of a state in \a Space given (see update(state, measurement, pinned, policy)).
     */
    template <Measurement M, typename Space, typename P>
    using GivenMeasuredType = typename decltype(internal::updateBranchesWith<true>(
        std::declval<ComponentType const &>(),
        std::declval<internal::PinnedMeasurement<Space, P, M> const &>(),
        std::declval<typename Moments<internal::PathTag<P>, Scalar>::VectorType const &>()
    ))::value_type;

    /** What update(state, measurement, pinned, policy) with \a M fills when given a result: one
     * branch per mode for noise by mode axis, as ResultType has.
     */
    template <Measurement M, typename Space, typename P>
    using GivenResultType = ImmFilterUpdate<
        std::conditional_t<
            internal::isByMode<M>,
            FilterUpdate<typename GivenMeasuredType<M, Space, P>::MeasurementSpaceType, 1, Scalar>,
            internal::FilterUpdateWithScratch<
                SpaceType,
                typename GivenMeasuredType<M, Space, P>::MeasurementSpaceType,
                GivenMeasuredType<M, Space, P>::BranchCount,
                Scalar,
                Allocator_>>,
        Scalar,
        Allocator_>;

    /** What update(state, measurement, pinned, policy) with \a M returns. */
    template <Measurement M, typename Space, typename P>
    using GivenUpdateType = std::expected<GivenResultType<M, Space, P>, NumericError>;

    /** The recursion with switching model \a model and prediction method \a method, which the
     * modes whose models carry none of their own predict by, with scratch for every mode from
     * \a allocator. Fixed-step models carry their own method, so there is none to give.
     */
    explicit ImmRecursion(Model_ model, MethodFor<Model_, Method_> method = {}, Allocator_ const & allocator = {})
        : m_model{std::move(model)}
        , m_method{internal::predictionMethod(m_model, std::move(method))}
        , m_predicted(ComponentAllocator(allocator))
        , m_posterior(ComponentAllocator(allocator))
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
        m_posterior = ConditionalsType(ComponentAllocator(allocator));
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

    /** \returns the discrete mode space */
    [[nodiscard]] ModeSpaceType const &
    modeSpace() const
    {
        return m_model.modeSpace();
    }

    /** \returns whether every model's writes partition \a space, which a state must be in */
    [[nodiscard]] bool
    partitions(SpaceType const & space) const
    {
        return m_model.partitions(space);
    }

    /** Mixes \a state's components through the chain, moves each by \a dt, any TimeStep (see
     * \ref Time), through its mode's process model, with \a inputs passed as in Emrys::predict(),
     * and moves the mode probabilities one step of the chain.
     *
     * \returns nothing for Gaussians; for a square-root form, a \c std::expected with the error of
     *          the first mode that could not take the step (see KalmanRecursion::predict()), which
     *          keeps the component it was to move from while the others move
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(StateType & state, TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        Scalar const step = internal::toScalarTime<Scalar>(dt);
        if constexpr (IsForm)
        {
            return moveModes(state, [&](ComponentType & moved, auto const & model, auto const & method) {
                return internal::predictInPlace(method, moved, model, step, inputs...);
            });
        }
        else
        {
            moveModes(state, [&](ComponentType const & mixed, auto const & model, auto const & method) {
                return Emrys::predict(method, mixed, model, step, inputs...);
            });
        }
    }

    /** Mixes \a state's components through the chain, moves each by its mode's fixed-step model's
     * step, by that model's method, with \a inputs passed as in Emrys::predict(), and moves the mode
     * probabilities one step of the chain, returning as predict(state, dt, inputs...) does.
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(StateType & state, Inputs const &... inputs)
    {
        if constexpr (IsForm)
        {
            return moveModes(state, [&](ComponentType & moved, auto const & model, auto const &) {
                return internal::predictInPlace(moved, model, inputs...);
            });
        }
        else
        {
            moveModes(state, [&](ComponentType const & mixed, auto const & model, auto const &) {
                return Emrys::predict(mixed, model, inputs...);
            });
        }
    }

    /** Moves \a state forward as predict(state, dt, inputs...) does.
     *
     * \returns the step as a HybridPredictedJoint, from the state's allocator, for a
     * HybridGaussianHistory to record
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_> && !IsForm)
    [[nodiscard]] JointType
    predictJoint(StateType & state, TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        auto joint = emptyJoint(state);
        predictJoint(state, dt, joint, inputs...);
        return joint;
    }

    /** As predictJoint(state, dt, inputs...), but into \a joint, whose storage is reused: with the
     * same joint at every step, recording allocates nothing once it has been filled.
     */
    template <typename JointAllocator, typename... Inputs>
        requires(!isFixedStep<Model_> && !IsForm)
    void
    predictJoint(
        StateType & state,
        TimeStep<Scalar> auto const & dt,
        HybridPredictedJoint<SpaceType, ModeSpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        Scalar const step = internal::toScalarTime<Scalar>(dt);
        recordJoint(
            state,
            joint,
            [&](ComponentType const & filtered, auto const & model, auto const & method) {
                return Emrys::predictJoint(method, filtered, model, step, inputs...);
            },
            [&](ComponentType const & filtered, auto const & model, auto const & method) {
                return Emrys::predict(method, filtered, model, step, inputs...);
            }
        );
        predict(state, step, inputs...);
    }

    /** Moves \a state forward as predict(state, inputs...) does with fixed-step models.
     *
     * \returns the step as a HybridPredictedJoint, as predictJoint(state, dt, inputs...) does
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !IsForm)
    [[nodiscard]] JointType
    predictJoint(StateType & state, Inputs const &... inputs)
    {
        auto joint = emptyJoint(state);
        predictJoint(state, joint, inputs...);
        return joint;
    }

    /** As predictJoint(state, inputs...), but into \a joint, whose storage is reused. */
    template <typename JointAllocator, typename... Inputs>
        requires(isFixedStep<Model_> && !IsForm)
    void
    predictJoint(
        StateType & state,
        HybridPredictedJoint<SpaceType, ModeSpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        recordJoint(
            state,
            joint,
            [&](ComponentType const & filtered, auto const & model, auto const &) {
                return Emrys::predictJoint(filtered, model, inputs...);
            },
            [&](ComponentType const & filtered, auto const & model, auto const &) {
                return Emrys::predict(filtered, model, inputs...);
            }
        );
        predict(state, inputs...);
    }

    /** Conditions each of \a state's components on \a measurement, by the measurement's own
     * method, reduces its branches with \a policy (Collapse, Select, or a Gate), and multiplies
     * each mode's probability by \f$ p(z \mid m) \f$.
     *
     * When the measurement's noise is by mode axis, <tt>additiveNoise(mixture).byMode<Axis>()</tt>
     * (see ModeNoise), its components are the values of the mode axis \c Axis: in each mode, only
     * the component that the mode's value in that axis names is used, and its own likelihood
     * reweights the mode. The components' weights are not used; the chain gives the probabilities
     * instead. The mode space must have the axis, and the noise one component per value of it.
     *
     * \returns each mode's update before reduction, the observation's log-likelihood, and whether
     * it was accepted (see ImmFilterUpdate), with the modes' updates from the
     * scratch's allocator; or, if any mode's update failed, as one whose innovation covariance is
     * not positive definite does, its error, with \a state unchanged (decisions.md [innovation-cholesky])
     */
    template <Measurement M, typename Policy = reductions::Collapse>
        requires(!internal::isGivenBlock<Policy>)
    [[nodiscard]] typename internal::UnlessGivenBlock<UpdateTypeOf<M>, Policy>::type
    update(StateType & state, M const & measurement, Policy const & policy = {})
    {
        auto result = emptyUpdate<M>(get_allocator(), state.size());
        auto const updated = update(state, measurement, policy, result);
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        return result;
    }

    /** As update(state, measurement, policy), but into \a result, whose storage is reused: with
     * the same result at every step, an update allocates nothing once it has been as large.
     *
     * \returns nothing, or the error update(state, measurement, policy) would return, with
     *          \a state unchanged and \a result unspecified
     */
    template <Measurement M, typename Policy, typename ModeUpdate, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(
        StateType & state,
        M const & measurement,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result
    )
    {
        if constexpr (internal::isByMode<M>)
        {
            return updateByAxis(state, measurement, policy, result, [](auto one) {
                return one;
            });
        }
        else
        {
            return updateModes(
                state,
                policy,
                result,
                [&](Eigen::Index, auto const & updateMode) {
                    return updateMode(measurement);
                },
                [](auto &, Eigen::Index) {}
            );
        }
    }

    /** As update(state, measurement, policy), but with a measurement per value of the mode axis
     * that \a measurement names (see SwitchingMeasurement): each mode is updated by the measurement
     * that its value in that axis selects, and its own likelihood \f$ p(z \mid m) \f$ reweights it.
     * The measurements need one ResultType, so that every mode's update has one type.
     */
    template <typename P, Measurement... Ms, typename Policy = reductions::Collapse>
    [[nodiscard]] SwitchingUpdateType<SwitchingMeasurement<P, Ms...>>
    update(StateType & state, SwitchingMeasurement<P, Ms...> const & measurement, Policy const & policy = {})
    {
        auto result = SwitchingResultType<SwitchingMeasurement<P, Ms...>>{get_allocator()};
        result.modes.reserve(std::size_t(state.size()));
        auto const updated = update(state, measurement, policy, result);
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        return result;
    }

    /** As update(state, measurement, policy) with a SwitchingMeasurement, but into \a result,
     * whose storage is reused.
     *
     * \returns nothing, or the error update(state, measurement, policy) would return
     */
    template <typename P, Measurement... Ms, typename Policy, typename ModeUpdate, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(
        StateType & state,
        SwitchingMeasurement<P, Ms...> const & measurement,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result
    )
    {
        static_assert(
            internal::pathResolves<P, ModeSpaceType>(),
            "a switching measurement is selected by a mode axis that the filter's mode space does not have"
        );
        static_assert(
            !(internal::isByMode<Ms> || ...),
            "a measurement of a switching measurement has noise by mode axis, byMode<Axis>(): give each value's "
            "measurement the noise it has in the modes with that value"
        );
        static_assert(
            (std::is_same_v<ResultType<Ms>, SwitchingResultType<SwitchingMeasurement<P, Ms...>>> && ...),
            "the measurements of a switching measurement update a mode into one type: give them one measurement "
            "space, and noise that is Gaussian in all of them or a mixture in all of them"
        );
        if constexpr (internal::pathResolves<P, ModeSpaceType>())
        {
            auto const [stride, axis] = internal::locateAxisByPath<P>(state.modeSpace());
            Eigen::Index const values = axis->cardinality();
            EMRYS_FILTER_ASSERT(
                values == Eigen::Index(sizeof...(Ms)) && "give one measurement per value of the axis that selects it"
            );
            return updateModes(
                state,
                policy,
                result,
                [&, stride](Eigen::Index m, auto const & updateMode) {
                    return measurement.visit(std::size_t((m / stride) % values), updateMode);
                },
                [](auto &, Eigen::Index) {}
            );
        }
        else
        {
            return std::expected<void, NumericError>{};
        }
    }

private:

    /** The update of a measurement whose noise is by mode axis, common to update(state,
     * measurement, policy) and its form given a block: mode \a m is updated by \a wrap(one), where
     * \a one is \a measurement with the noise component that the mode's value in the noise's axis
     * names.
     */
    template <Measurement M, typename Policy, typename ModeUpdate, typename ResultAllocator, typename Wrap>
    [[nodiscard]] std::expected<void, NumericError>
    updateByAxis(
        StateType & state,
        M const & measurement,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result,
        Wrap const & wrap
    )
    {
        using Axis = typename internal::MeasurementModeAxisOf<M>::type;
        static_assert(
            internal::pathResolves<Axis, ModeSpaceType>(),
            "the measurement's noise is by a mode axis, byMode<Axis>(), that the filter's mode space does not have"
        );
        if constexpr (internal::pathResolves<Axis, ModeSpaceType>())
        {
            return updateByAxisIn<Axis>(state, measurement, policy, result, wrap);
        }
        else
        {
            return std::expected<void, NumericError>{};
        }
    }

    /** updateByAxis() with \a Axis, the noise's mode axis, found in the mode space. */
    template <
        typename Axis,
        Measurement M,
        typename Policy,
        typename ModeUpdate,
        typename ResultAllocator,
        typename Wrap>
    [[nodiscard]] std::expected<void, NumericError>
    updateByAxisIn(
        StateType & state,
        M const & measurement,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result,
        Wrap const & wrap
    )
    {
        auto const [stride, axis] = internal::locateAxisByPath<Axis>(state.modeSpace());
        Eigen::Index const values = axis->cardinality();
        auto const & noise = measurement.model().template policy<internal::PolicyKind::additiveNoise>().distribution();
        EMRYS_FILTER_ASSERT(noise.size() == values && "the noise needs one component per value of the mode axis");
        using Component = std::remove_cvref_t<decltype(noise.begin()->component)>;
        using One = internal::WithNoiseComponent<M, Component>;
        // Each mode is updated with the one component its value names, not with every component.
        auto const componentOf = [=](Eigen::Index m) {
            return (m / stride) % values;
        };
        return updateModes(
            state,
            policy,
            result,
            [&](Eigen::Index m, auto const & updateMode) {
                return updateMode(wrap(One{measurement, noise[componentOf(m)].component}));
            },
            [&](auto & measured, Eigen::Index m) {
                auto & branch = measured.branches.front();
                branch.component = componentOf(m);
                branch.logWeight = Scalar(0);
            }
        );
    }

public:

    /** As update(state, measurement, policy), but with \a state a hybrid over the rest of a state
     * and the block that \a pinned names given separately, as given() makes it: each mode's
     * %Gaussian is conditioned as update(prior, measurement, given) conditions one. This is the
     * step of a Rao-Blackwellized point-mass filter whose cells each hold a hybrid
     * (RbPointMassFilter), each cell's center given.
     *
     * \returns as update(state, measurement, policy) does
     */
    template <Measurement M, typename Space, typename P, typename Value, typename Policy = reductions::Collapse>
        requires(!IsForm)
    [[nodiscard]] GivenUpdateType<M, Space, P>
    update(
        StateType & state,
        M const & measurement,
        GivenBlock<Space, P, Value> const & pinned,
        Policy const & policy = {}
    )
    {
        auto result = emptyGivenUpdate<M, Space, P>(get_allocator(), state.size());
        auto const updated = update(state, measurement, pinned, policy, result);
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        return result;
    }

    /** As update(state, measurement, pinned, policy), but into \a result, whose storage is reused.
     *
     * \returns nothing, or the error update(state, measurement, pinned, policy) would return
     */
    template <
        Measurement M,
        typename Space,
        typename P,
        typename Value,
        typename Policy,
        typename ModeUpdate,
        typename ResultAllocator>
        requires(!IsForm)
    [[nodiscard]] std::expected<void, NumericError>
    update(
        StateType & state,
        M const & measurement,
        GivenBlock<Space, P, Value> const & pinned,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result
    )
    {
        if constexpr (internal::isByMode<M>)
        {
            // Each mode's measurement holds its own noise component, so each resolves its own.
            return updateByAxis(state, measurement, policy, result, [&]<typename One>(One one) {
                return internal::GivenComponent<Space, P, One, Value>{std::move(one), pinned.space, pinned.value};
            });
        }
        else
        {
            auto const f = internal::PinnedMeasurement<Space, P, M>{pinned.space, measurement};
            return updateGiven(state, f, pinned.value, policy, result);
        }
    }

    /** As update(state, measurement, pinned, policy, result), with the measurement over the rest and
     * the given block already resolved as \a f and the block's value \a value: a caller that
     * updates many hybrids, each given its own value, resolves the measurement once. A measurement
     * whose noise is by mode axis takes update(state, measurement, pinned, policy, result), which
     * resolves each mode's component.
     */
    template <
        typename Space,
        typename P,
        typename M,
        typename Value,
        typename Policy,
        typename ModeUpdate,
        typename ResultAllocator>
        requires(!IsForm && !internal::isByMode<M>)
    [[nodiscard]] std::expected<void, NumericError>
    updateGiven(
        StateType & state,
        internal::PinnedMeasurement<Space, P, M> const & f,
        Value const & value,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result
    )
    {
        using Given = internal::GivenMeasurement<internal::PinnedMeasurement<Space, P, M> const &, Value>;
        return updateModes(
            state,
            policy,
            result,
            [&](Eigen::Index, auto const & updateMode) {
                return updateMode(Given{f, value});
            },
            [](auto &, Eigen::Index) {}
        );
    }

    /** \returns an empty ResultType<M> with room for \a modes modes, from \a allocator, for update()
     * to fill
     */
    template <Measurement M>
    [[nodiscard]] ResultType<M>
    emptyUpdate(Allocator_ const & allocator, Eigen::Index modes) const
    {
        using Result = ResultType<M>;
        auto result = Result{allocator};
        result.modes.reserve(std::size_t(modes));
        return result;
    }

    /** \returns an empty GivenResultType with room for \a modes modes, from \a allocator, for
     * update(state, measurement, pinned, policy, result) to fill
     */
    template <Measurement M, typename Space, typename P>
    [[nodiscard]] GivenResultType<M, Space, P>
    emptyGivenUpdate(Allocator_ const & allocator, Eigen::Index modes) const
    {
        auto result = GivenResultType<M, Space, P>{allocator};
        result.modes.reserve(std::size_t(modes));
        return result;
    }

    /** \returns an empty joint with room for every mode of \a state, from the state's allocator */
    [[nodiscard]] JointType
    emptyJoint(StateType const & state) const
    {
        auto joint = JointType{
            state.modes(),
            typename JointType::JointsType(state.get_allocator()),
            m_model.chain().matrix(),
            typename JointType::PredictionsType(state.get_allocator())
        };
        joint.joints.reserve(std::size_t(state.size()));
        joint.throughOtherModels.reserve(std::size_t(state.size() * (state.size() - 1)));
        return joint;
    }

private:

    using ConditionalsType = typename StateType::ConditionalsType;         ///< the modes' components, as stored
    using ComponentAllocator = typename ConditionalsType::allocator_type;  ///< allocates them

    /** Reserves the scratch for every mode. */
    void
    reserveScratch()
    {
        auto const modes = std::size_t(m_model.modeSpace().cardinality());
        m_predicted.reserve(modes);
        if constexpr (!IsForm)
        {
            m_posterior.reserve(modes);
        }
    }

    /** Mixes \a state's components through the chain, moves each mode's mixed component by
     * \a predictMode with its mode's process model and method, and moves the mode probabilities one step of
     * the chain: the body of both forms of predict(). For Gaussians \a predictMode returns the
     * predicted component. For a square-root form it moves the component in place, in the state
     * when the chain mixes nothing and in the scratch the mixed one is built in otherwise, and
     * returns a \c std::expected; a mode it fails on keeps the component it was to move from.
     *
     * \returns nothing, or for a square-root form the first failed mode's error
     */
    template <typename PredictMode>
    [[nodiscard]] auto
    moveModes(StateType & state, PredictMode const & predictMode)
    {
        if constexpr (IsForm)
        {
            auto failure = std::expected<void, NumericError>{};
            auto const keepFirst = [&](std::expected<void, NumericError> const & moved) {
                if (!moved && failure)
                {
                    failure = moved;
                }
            };
            moveModesBy(
                state,
                [&](Eigen::Index j, auto const & model, auto const & method) {
                    keepFirst(state.updateConditional(j, [&](ComponentType & own) {
                        return predictMode(own, model, method);
                    }));
                },
                [&](ComponentType && mixed, auto const & model, auto const & method) {
                    m_predicted.push_back(std::move(mixed));
                    keepFirst(predictMode(m_predicted.back(), model, method));
                }
            );
            return failure;
        }
        else
        {
            moveModesBy(
                state,
                [&](Eigen::Index j, auto const & model, auto const & method) {
                    m_predicted.push_back(predictMode(state.conditional(j), model, method));
                },
                [&](ComponentType && mixed, auto const & model, auto const & method) {
                    m_predicted.push_back(predictMode(mixed, model, method));
                }
            );
        }
    }

    /** moveModes() with \a moveOwn, which moves mode \a j's own component when the chain mixes
     * nothing, and \a moveMixed, which moves mode \a j's mixed component otherwise. Both push to
     * the scratch for Gaussians; a square-root form's \a moveOwn moves the state's component.
     */
    template <typename MoveOwn, typename MoveMixed>
    void
    moveModesBy(StateType & state, MoveOwn const & moveOwn, MoveMixed const & moveMixed)
    {
        Eigen::Index const count = state.size();
        auto const & logPrior = state.modes().logWeights();
        auto const & transitions = m_model.chain().matrix();
        m_predicted.clear();
        if (transitions.isIdentity(Scalar(0)))
        {
            // The autonomous multiple model estimator: each mode's mixing puts all its weight on
            // its own component, so each is predicted as it is, and the probabilities carry over.
            for (Eigen::Index j = 0; j < count; ++j)
            {
                m_model.visit(j, m_method, [&](auto const & model, auto const & method) {
                    moveOwn(j, model, method);
                });
            }
            if constexpr (!IsForm)
            {
                state.swapConditionals(m_predicted);
            }
            return;
        }
        CategoricalParams<ModeSpaceType, Scalar> const predictedModes = Emrys::predict(state.modes(), m_model.chain());

        auto const conditional = [&](Eigen::Index i) -> ComponentType const & {
            return state.conditional(i);
        };
        auto logMixing = typename StateType::ModesType::ArrayType{};
        logMixing.resize(count);
        for (Eigen::Index j = 0; j < count; ++j)
        {
            // A mode the chain can't reach starts from its own component, which is never weighted.
            bool const reachable = predictedModes.logWeight(j) > -Eigen::NumTraits<Scalar>::infinity();
            for (Eigen::Index i = 0; i < count; ++i)
            {
                logMixing(i) = reachable ? Eigen::numext::log(transitions(i, j)) + logPrior(i)
                                         : (i == j ? Scalar(0) : -Eigen::NumTraits<Scalar>::infinity());
            }
            ComponentType mixed = ComponentTraits<ComponentType>::collapse(count, conditional, [&](Eigen::Index i) {
                return logMixing(i);
            });
            m_model.visit(j, m_method, [&](auto const & model, auto const & method) {
                moveMixed(std::move(mixed), model, method);
            });
        }
        state.swapConditionals(m_predicted);
        state.setModes(predictedModes);
    }

    /** Fills \a joint with \a state before a prediction: the mode probabilities, the chain's
     * matrix, each mode's component moved by \a predictMode through its own process model, as a
     * PredictedJoint, and by \a predictThrough through every other mode's. The body of both forms
     * of predictJoint(), before they predict.
     */
    template <typename JointAllocator, typename PredictMode, typename PredictThrough>
    void
    recordJoint(
        StateType const & state,
        HybridPredictedJoint<SpaceType, ModeSpaceType, Scalar, JointAllocator> & joint,
        PredictMode const & predictMode,
        PredictThrough const & predictThrough
    ) const
    {
        Eigen::Index const count = state.size();
        joint.modes = state.modes();
        joint.transitions = m_model.chain().matrix();
        joint.joints.clear();
        joint.joints.reserve(std::size_t(count));
        joint.throughOtherModels.clear();
        joint.throughOtherModels.reserve(std::size_t(count * (count - 1)));
        for (Eigen::Index j = 0; j < count; ++j)
        {
            m_model.visit(j, m_method, [&](auto const & model, auto const & method) {
                joint.joints.push_back(predictMode(state.conditional(j), model, method));
            });
            for (Eigen::Index i = 0; i < count; ++i)
            {
                if (i != j)
                {
                    m_model.visit(i, m_method, [&](auto const & model, auto const & method) {
                        joint.throughOtherModels.push_back(predictThrough(state.conditional(j), model, method));
                    });
                }
            }
        }
    }

    /** The update common to every form of update(): mode \a m's %Gaussian is updated by its
     * measurement, which \a withMeasurement(m, updateMode) passes to \a updateMode, so that the
     * modes' measurements may have different types, as a SwitchingMeasurement's do; its branches
     * are labeled by \a relabel(measured, m) and reduced by \a policy, and the modes are
     * reweighted by their log-likelihoods, with \a result filled in.
     * The reweighting reads each mode's likelihood, so the update computes it with its gain. The
     * slots a result holds from the last update are reused, each mode measured in its slot's
     * scratch where it has one, and the modes beyond them get new slots from the result's
     * allocator (see internal::updateSlot()). For Gaussians the posterior components are built
     * apart and exchanged with the state's at the end, so a mode whose update fails, or a
     * measurement no mode explains, leaves \a state as it was. A square-root form updates each
     * mode's component in place: a mode whose update fails keeps its component and stops the
     * update, the modes before it updated, with the mode probabilities as they were.
     *
     * \returns nothing, or the WeightError of mode probabilities that define no distribution, before
     *          any mode is updated, or the error of the first mode whose update failed (decisions.md
     *          [innovation-cholesky], [update-outcomes])
     */
    template <
        typename Policy,
        typename ModeUpdate,
        typename ResultAllocator,
        typename WithMeasurement,
        typename Relabel>
    [[nodiscard]] std::expected<void, NumericError>
    updateModes(
        StateType & state,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result,
        WithMeasurement const & withMeasurement,
        Relabel const & relabel
    )
    {
        // Mode probabilities the caller left undefined fail the update before any mode is written.
        auto const logTotal = state.modes().logTotalWeight();
        if (!logTotal)
        {
            return std::unexpected{logTotal.error()};
        }
        Eigen::Index const count = state.size();
        Eigen::Index const held = std::min(Eigen::Index(result.modes.size()), count);
        result.modes.erase(result.modes.begin() + held, result.modes.end());
        internal::UpdateAccess::logLikelihood(result) = Scalar(0);
        result.accepted = false;
        m_posterior.clear();
        auto logLikelihoods = typename StateType::ModesType::ArrayType{};
        logLikelihoods.resize(count);
        for (Eigen::Index m = 0; m < count; ++m)
        {
            auto const updateMode = [&](auto const & measurement) -> std::expected<void, NumericError> {
                if constexpr (IsForm)
                {
                    static_assert(
                        internal::isCollapsePolicy<Policy>,
                        "a square-root form has one branch and takes the Collapse policy alone; gating and selection "
                        "among noise components are the covariance form's (decisions.md [gaussian-forms])"
                    );
                    auto updated = state.updateConditional(m, [&](ComponentType & conditional) {
                        return internal::updateInPlace<true>(conditional, measurement);
                    });
                    if (!updated)
                    {
                        result.accepted = false;
                        return std::unexpected{updated.error()};
                    }
                    relabel(*updated, m);
                    updated->branches.front().mode = m;
                    logLikelihoods(m) = updated->logLikelihood();
                    result.accepted = true;
                    if (std::size_t(m) == result.modes.size())
                    {
                        result.modes.push_back(std::move(*updated));
                    }
                    else
                    {
                        result.modes[std::size_t(m)] = std::move(*updated);
                    }
                }
                else
                {
                    auto const updated = internal::updateSlot<true>(
                        result.modes,
                        std::size_t(m),
                        ResultAllocator(result.modes.get_allocator()),
                        state.conditional(m),
                        measurement,
                        [&](auto & measured) {
                            relabel(measured, m);
                            for (auto & branch : measured.branches)
                            {
                                branch.mode = m;
                            }
                            static_assert(
                                ReductionPolicy<Policy, std::remove_cvref_t<decltype(measured)>, ComponentType>,
                                "an IMM filter keeps one component per mode, so its policy must keep one: Collapse, "
                                "Select, or a Gate over them"
                            );
                            auto kept = std::optional<ComponentType>{};
                            auto out =
                                internal::OneComponentOutput<ComponentType, Scalar, std::optional<ComponentType>>{kept};
                            internal::reduceInto(policy, std::move(measured.branches), out);
                            // The mode is weighted by the likelihood of the branches what it kept stands for.
                            logLikelihoods(m) = out.added ? measured.logLikelihood() + out.logWeight
                                                          : -Eigen::NumTraits<Scalar>::infinity();
                            m_posterior.push_back(out.added ? std::move(*kept) : state.conditional(m));
                            result.accepted = result.accepted || out.added;
                            return out.added;
                        }
                    );
                    if (!updated)
                    {
                        result.accepted = false;
                        return updated;
                    }
                }
                return {};
            };
            if (auto const updated = withMeasurement(m, updateMode); !updated)
            {
                return updated;
            }
        }
        auto evidence = LogSumExp<Scalar>{};
        for (Eigen::Index m = 0; m < count; ++m)
        {
            evidence.add(state.modes().logWeight(m) + result.modes[std::size_t(m)].logLikelihood());
        }
        internal::UpdateAccess::logLikelihood(result) = evidence.value() - *logTotal;
        if (!result.accepted)
        {
            return {};
        }

        CategoricalParams<ModeSpaceType, Scalar> modes = state.modes();
        if (!modes.update(logLikelihoods))
        {
            // Every accepting mode has probability zero: the measurement has no density under the
            // modes that could hold it, so it is rejected [update-outcomes].
            result.accepted = false;
            for (auto & mode : result.modes)
            {
                mode.accepted = false;
            }
            return {};
        }
        if constexpr (!IsForm)
        {
            state.swapConditionals(m_posterior);
        }
        state.setModes(std::move(modes));
        return {};
    }

    Model_ m_model;                                   ///< the switching model
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method_ m_method;  ///< the prediction method

    // Scratch, exchanged with the state's components: the predicted ones and the posterior ones.
    ConditionalsType m_predicted;  ///< the mixed and predicted components

    ConditionalsType m_posterior;  ///< the updated components
};

namespace internal
{

/// \cond INTERNAL_SPECIALIZATIONS
template <
    Component Component_,
    DiscreteSpace ModeSpace_,
    typename HybridAllocator,
    typename Model,
    typename Method,
    typename Allocator>
struct RecursionFor<Hybrid<Component_, ModeSpace_, HybridAllocator>, Model, Method, Allocator>
{
    using type = ImmRecursion<Component_, Model, Method, HybridAllocator>;
};

/// \endcond

}  // namespace internal

/** \ingroup Filters
 * \brief An interacting multiple model (IMM) filter: a hybrid state with one component per mode,
 * and the ImmRecursion that mixes it through a switching process model and
 * updates it per mode.
 *
 * predict() first mixes: the component that mode \a j starts from is the collapse of every
 * mode's component, weighted by the probability \f$ \mu_{i \mid j} \propto \Pi_{ij} \mu_i \f$
 * that the target was in mode \a i given that it is now in \a j. Each mixed component is then
 * predicted through its mode's process model, by the model's own method where it carries one
 * (withMethod(), FixedStepModel) and by the filter's otherwise [switching-models], and the
 * mode probabilities move one step of the chain. update() updates each mode's component as a
 * KalmanFilter would and multiplies each mode's probability by \f$ p(z \mid m) \f$.
 *
 * The interacting multiple model algorithm of \ref Blom1988 "Blom and Bar-Shalom (1988)", as
 * given in \ref Hashimzade2024 "Hashimzade et al. (2024)", §2.2.3, Algorithm 2 with one lag:
 * mixing, mode-matched filtering, mode probability update, and the combined estimate as
 * collapsed().
 *
 * The mode space may be a DiscreteProduct, such as a maneuver regime and a sensor's
 * inlier/outlier state. A measurement's noise components can then be the values of a mode
 * axis, so that the outlier state is a Markov chain (bursts of outliers, a stuck sensor)
 * rather than independent at each measurement: a measurement whose noise is
 * <tt>additiveNoise(mixture).byMode<Sensor>()</tt> (see ModeNoise) uses, in each mode, only the
 * noise component that the mode's \c Sensor value names. A measurement that depends on the mode
 * in more than its noise, such as a sensor whose bias the mode sets, is one measurement per value
 * of an axis: update(switchingMeasurement<Sensor>(nominal, biased)) updates each mode by the one
 * that its \c Sensor value names (see SwitchingMeasurement).
 *
 * \code
 * using Mode = DiscreteProduct<Maneuver, Sensor>;
 * auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95),
 *                                      MarkovChain<Sensor>::persistent(0.8)};
 * auto filter = ImmFilter{switchingModel<Maneuver>(chain, cruise, turn),
 *                  Hybrid{CategoricalParams<Mode>{}, initial}};
 * filter.predict(dt);
 * auto const updated = filter.update(fix);  // every mode, with the fix's own noise
 * auto const robust = filter.update(robustFix);  // noise byMode<Sensor>(): inlier and outlier are its values
 * auto const biased = filter.update(switchingMeasurement<Sensor>(fix, biasedFix));  // a measurement per value
 * filter.state().modes().toDensity()->probability<Turn>();
 * filter.state().toDensity()->collapsed();  // the estimate as one GaussianParams
 * \endcode
 *
 * A mode whose update the policy rejects keeps its prior component and gets probability zero,
 * from which the chain can bring it back at the next prediction. When every mode rejects the
 * measurement, it is rejected and the state is unchanged. With one mode the filter is a
 * KalmanFilter. With the identity chain, <tt>MarkovChain<Mode>{}</tt>, it is the autonomous
 * multiple model estimator, which AmmFilter spells: each mode's mixing
 * weights put everything on its own component, so predict() skips the mixing and leaves the
 * mode probabilities as they are.
 *
 * The predicted and posterior components are built in scratch from the state's allocator and
 * exchanged with the state's, so on a bounded space its steps allocate nothing;
 * update(measurement, policy, result) also reuses the result's storage (see \ref Allocation).
 *
 * \tparam Component_  the component given one mode (see Component), a Gaussian over the state
 *                     space (see \ref Spaces)
 * \tparam Model_      a SwitchingModel over the mode space, of process models whose blocks
 *                     partition the state space; with fixed-step models, predict() takes no time step
 * \tparam Method_     the prediction method of every mode whose model carries none of its own
 *                    (see withMethod()); ModelsMethod with fixed-step models, which all carry one
 * \tparam Allocator_  the state's allocator (see \ref Allocation)
 * \tparam Upkeep_     the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                     one, such as FloorModes or RepairCovariance
 *
 * \sa Hybrid, SwitchingModel, ImmRecursion, \ref Filters
 */
template <
    Component Component_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class ImmFilter : public internal::HeldUpkeep<ImmFilter<Component_, Model_, Method_, Allocator_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<ImmFilter, Upkeep_>;  ///< what holds the upkeep

public:

    /** the steps it runs */
    using RecursionType = ImmRecursion<Component_, Model_, Method_, Allocator_>;

    using Scalar = typename RecursionType::Scalar;                ///< the scalar type
    using SpaceType = typename RecursionType::SpaceType;          ///< the state space
    using ModelType = Model_;                                     ///< the process model
    using MethodType = Method_;                                   ///< the prediction method
    using ModeSpaceType = typename RecursionType::ModeSpaceType;  ///< the discrete mode space
    using StateType = typename RecursionType::StateType;          ///< the state the filter owns
    using ComponentType = Params<Component_>;                     ///< one component of the state
    using allocator_type = Allocator_;                            ///< the state's allocator
    using JointType = typename RecursionType::JointType;          ///< what predictJoint() returns

    /** A filter with switching model \a model, initial state \a state, prediction method \a method,
     * and upkeep \a upkeep, which maintain() runs. Fixed-step models carry their own method, so there
     * is none to give; each mode predicts by its own model's.
     */
    ImmFilter(
        Model_ model,
        StateType state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : UpkeepBase{std::move(upkeep)}
        , m_state{std::move(state)}
        , m_recursion{std::move(model), std::move(method), m_state.get_allocator()}
    {
        EMRYS_FILTER_ASSERT(
            m_recursion.modeSpace() == m_state.modeSpace() && "the model and the state have different modes"
        );
        internal::assertPartitions(
            m_recursion.model(),
            m_state.space(),
            "every process model must write every coordinate of the state space once"
        );
    }

    /** A filter with switching model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    ImmFilter(Model_ model, StateType state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : ImmFilter{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}

    /** \returns the allocator of the state and scratch */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_state.get_allocator();
    }

    /** \returns the current state: the mode probabilities and each mode's component */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the current state, to change where it is between steps, as upkeep such as
     * FloorModes and RepairCovariance does. It must keep its space, mode space, and allocator; setState() replaces it
     * with another, and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state, as when reinitializing, with one that has the model's modes and whose
     * space every process model's writes partition, as the constructor's must. The scratch follows
     * the new state's allocator.
     */
    void
    setState(StateType state)
    {
        EMRYS_FILTER_ASSERT(state.modeSpace() == m_recursion.modeSpace() && "the state has different modes");
        internal::assertPartitions(
            m_recursion.model(),
            state.space(),
            "every process model must write every coordinate of the state space once"
        );
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

    /** Mixes the modes' components through the chain, moves each by \a dt, any TimeStep (see
     * \ref Time), through its mode's process model, with \a inputs passed as in Emrys::predict(),
     * and moves the mode probabilities one step of the chain.
     *
     * \returns nothing for Gaussians; for a square-root form, a \c std::expected that is empty
     *          when a mode could not take the step (see ImmRecursion::predict())
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        return m_recursion.predict(m_state, internal::toScalarTime<Scalar>(dt), inputs...);
    }

    /** Mixes the modes' components through the chain, moves each by its mode's fixed-step model's
     * step, by that model's method, with \a inputs passed as in Emrys::predict(), and moves the mode
     * probabilities one step of the chain, returning as predict(dt, inputs...) does.
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
     * predict(inputs...) do, with \a inputs at every step: the chain takes a step each time. For a
     * square-root form it stops at the first step a mode could not take, returning its error.
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

    /** Moves the state forward as predict(dt, inputs...) does.
     *
     * \returns the step as a HybridPredictedJoint, from the state's allocator, for a
     * HybridGaussianHistory to record
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] JointType
    predictJoint(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        return m_recursion.predictJoint(m_state, internal::toScalarTime<Scalar>(dt), inputs...);
    }

    /** As predictJoint(dt, inputs...), but into \a joint, whose storage is reused: with the same
     * joint at every step, recording allocates nothing once it has been filled.
     */
    template <typename JointAllocator, typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predictJoint(
        TimeStep<Scalar> auto const & dt,
        HybridPredictedJoint<SpaceType, ModeSpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        m_recursion.predictJoint(m_state, internal::toScalarTime<Scalar>(dt), joint, inputs...);
    }

    /** Moves the state forward as predict(inputs...) does with fixed-step models.
     *
     * \returns the step as a HybridPredictedJoint, as predictJoint(dt, inputs...) does
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] JointType
    predictJoint(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        return m_recursion.predictJoint(m_state, inputs...);
    }

    /** As predictJoint(inputs...), but into \a joint, whose storage is reused. */
    template <typename JointAllocator, typename... Inputs>
        requires isFixedStep<Model_>
    void
    predictJoint(
        HybridPredictedJoint<SpaceType, ModeSpaceType, Scalar, JointAllocator> & joint,
        Inputs const &... inputs
    )
    {
        m_recursion.predictJoint(m_state, joint, inputs...);
    }

    /** Conditions each mode's component on \a measurement, by the measurement's own method,
     * reduces its branches with \a policy (Collapse, Select, or a Gate), and multiplies each mode's
     * probability by \f$ p(z \mid m) \f$.
     *
     * \returns each mode's update before reduction, the observation's log-likelihood, and whether
     * it was accepted (see ImmFilterUpdate), with the modes' updates from the
     * state's allocator, in a \c std::expected that is empty, with the state unchanged, when a
     * mode's update failed (see ImmRecursion::update())
     */
    template <Measurement M, typename Policy = reductions::Collapse>
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
    template <Measurement M, typename Policy, typename ModeUpdate, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(M const & measurement, Policy const & policy, ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result)
    {
        return m_recursion.update(m_state, measurement, policy, result);
    }

    /** As update(measurement, policy), but with a measurement per value of the mode axis that
     * \a measurement names (see SwitchingMeasurement): each mode is updated by the measurement that
     * its value in that axis selects, and its own likelihood \f$ p(z \mid m) \f$ reweights it.
     */
    template <typename P, Measurement... Ms, typename Policy = reductions::Collapse>
    [[nodiscard]] auto
    update(SwitchingMeasurement<P, Ms...> const & measurement, Policy const & policy = {})
    {
        return m_recursion.update(m_state, measurement, policy);
    }

    /** As update(measurement, policy) with a SwitchingMeasurement, but into \a result, whose
     * storage is reused.
     *
     * \returns nothing, or the error update(measurement, policy) would return
     */
    template <typename P, Measurement... Ms, typename Policy, typename ModeUpdate, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(
        SwitchingMeasurement<P, Ms...> const & measurement,
        Policy const & policy,
        ImmFilterUpdate<ModeUpdate, Scalar, ResultAllocator> & result
    )
    {
        return m_recursion.update(m_state, measurement, policy, result);
    }

private:

    StateType m_state;          ///< the state estimate
    RecursionType m_recursion;  ///< the model, method, and scratch the steps run with
};

/// \cond DEDUCTION_GUIDES
// One guide with a defaulted method, its parameters in the class's order.
template <
    Component C,
    typename Model,
    typename Method = DefaultMethodFor<Model>,
    typename Allocator,
    typename ModeSpace>
    requires(!internal::UpkeepArgument<Method>)
ImmFilter(Model, HybridParams<C, ModeSpace, Allocator>, Method = {}) -> ImmFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    Method,
    Allocator>;

template <Component C, typename Model, typename Allocator, typename ModeSpace, typename U>
    requires internal::UpkeepArgument<U>
ImmFilter(Model, HybridParams<C, ModeSpace, Allocator>, U) -> ImmFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
    DefaultMethodFor<Model>,
    Allocator,
    U>;

template <Component C, typename Model, typename Method, typename Allocator, typename ModeSpace, typename U>
ImmFilter(Model, HybridParams<C, ModeSpace, Allocator>, Method, U) -> ImmFilter<
    C,
    internal::BoundStep<Model, typename Params<C>::SpaceType, typename Params<C>::Scalar>,
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
    ImmFilter<Component_, Model_, Method_, Allocator_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "interacting multiple model filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
