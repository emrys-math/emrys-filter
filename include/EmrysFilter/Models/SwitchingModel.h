#pragma once

/** \file
 * SwitchingModel (decisions.md [switching-models], [imm]): a Markov chain on a discrete mode space and the process
 * model each mode follows, one per mode, one per value of a mode axis through switchingModel(),
 * or one shared, and over a fixed step when every model is a FixedStepModel [fixed-step-models]. The
 * interacting multiple model filter's process model. withMethod() gives one mode's model a prediction
 * method of its own, ModelWithMethod. SwitchingMeasurement is the model's counterpart for an update:
 * one measurement per value of a mode axis, through switchingMeasurement(). Read Reduction.h next;
 * the design review is docs/guide/imm-filter.md.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Categorical.h>
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
 * <tt>SwitchingModel{chain, cruise, withMethod(turn, methods::unscented())}</tt> the cruise mode
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
 * filter's method: <tt>SwitchingModel{chain, cruise, withMethod(turn, methods::unscented())}</tt>
 * [switching-models]. A fixed-step model takes its method from fixedStep() instead.
 */
template <typename Model, TransformMethod Method>
[[nodiscard]] ModelWithMethod<Model, Method>
withMethod(Model model, Method method)
{
    return {std::move(model), std::move(method)};
}

/** \ingroup Models
 * \brief A process model that switches between modes: a Markov chain on a discrete mode space,
 * and the process model each mode follows [switching-models].
 *
 * Each model is a process model over the whole state (see ProcessModel), and every mode
 * shares the state space; a mode that doesn't use a block holds it. There are three ways to
 * assign models to modes:
 *  - one model per mode, in flat-index order: <tt>SwitchingModel{chain, cruise, turn}</tt>;
 *  - one model per value of one axis of the mode space, when the others (such as a
 *    sensor's inlier/outlier state) don't affect the dynamics:
 *    <tt>switchingModel<Maneuver>(chain, cruise, turn)</tt>;
 *  - one model shared by every mode, when only the measurements depend on the mode:
 *    <tt>SwitchingModel{chain, model}</tt>.
 *
 * The chain moves the mode once per prediction, whatever its time step.
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
 * The Markovian switching of \ref Blom1988 "Blom and Bar-Shalom (1988)": a jump Markov linear
 * system whose mode selects the dynamics.
 *
 * \tparam ModeSpace_  the discrete mode space (see \ref Discrete)
 * \tparam Scalar_     scalar type of the chain's probabilities
 * \tparam Path_       the mode axis that selects the model, as a tag or TagPath; the whole
 *                     mode space when it is \a ModeSpace_ itself
 * \tparam Models      the process models, in the order of the axis's values, each of which may be a
 *                     ModelWithMethod
 *
 * \sa ImmFilter, MarkovChain, processModel(), withMethod()
 */
template <DiscreteSpace ModeSpace_, RealScalar Scalar_, typename Path_, typename... Models>
class SwitchingModel
{
    static_assert(sizeof...(Models) > 0, "a switching model needs a process model");

    static_assert(
        ((isFixedStep<Models> || isUnboundStep<Models>) && ...)
            || ((!isFixedStep<Models> && !isUnboundStep<Models>) && ...),
        "a switching model's process models are all fixed-step or none are"
    );

public:

    using Scalar = Scalar_;                              ///< the scalar type of the chain's probabilities
    using ModeSpaceType = ModeSpace_;                    ///< the discrete mode space
    using ChainType = MarkovChain<ModeSpace_, Scalar_>;  ///< the chain's type

    /** Whether one model serves every mode: one model and no path, the path being the whole mode
     * space. One model with a path that names an axis is one per value of that axis, and its count
     * is checked, so that a forgotten model is an error rather than dynamics every mode shares.
     */
    static constexpr bool Shared = sizeof...(Models) == 1 && std::is_same_v<Path_, ModeSpace_>;

    /** The chain \a chain on the modes, and \a models, one per value of the axis \a Path_
     * (one shared by every mode if there is only one and \a Path_ is the mode space).
     */
    SwitchingModel(ChainType chain, Models... models)
        : m_chain{std::move(chain)}
        , m_models{std::move(models)...}
    {
        if constexpr ((isFixedStep<Models> && ...))
        {
            EMRYS_FILTER_ASSERT(
                std::apply(
                    [](auto const & first, auto const &... rest) {
                        return ((rest.timeStep() == first.timeStep()) && ...);
                    },
                    m_models
                )
                && "a switching model's fixed-step models must share one step"
            );
        }
        if constexpr (!Shared)
        {
            using Axis = internal::PathTag<Path_>;
            static_assert(
                Axis::Cardinality == Eigen::Dynamic || Axis::Cardinality == int(sizeof...(Models)),
                "give one process model per value of the axis that selects it, "
                "or no path for one model shared by every mode"
            );
            EMRYS_FILTER_ASSERT(
                selector().cardinality == Eigen::Index(sizeof...(Models))
                && "give one process model per value of the axis that selects it, "
                   "or no path for one model shared by every mode"
            );
        }
    }

    /** \a other with each of its models converted to this one's: how a filter binds a switching
     * model of fixedStep() results to its state's space. Implicit, as FixedStepModel's is, so that a
     * filter's constructor takes the switching model where its deduction guide names this type.
     */
    template <typename... Others>
        requires(
            sizeof...(Others) == sizeof...(Models) && !(std::is_same_v<Others, Models> && ...)
            && (std::is_convertible_v<Others const &, Models> && ...)
        )
    SwitchingModel(SwitchingModel<ModeSpace_, Scalar_, Path_, Others...> const & other)
        : SwitchingModel{other, std::index_sequence_for<Models...>{}}
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

    /** \returns the mode space */
    [[nodiscard]] ModeSpace_ const &
    modeSpace() const
    {
        return m_chain.space();
    }

    /** \returns process model \a I; for a model given to withMethod(), the model it was given */
    template <std::size_t I>
    [[nodiscard]] auto const &
    model() const
    {
        return internal::modeModel(std::get<I>(m_models));
    }

    /** \returns process model \a I, to change its parameters */
    template <std::size_t I>
    [[nodiscard]] auto &
    model()
    {
        return internal::modeModel(std::get<I>(m_models));
    }

    /** \returns the index of the process model that the mode with flat index \a mode follows */
    [[nodiscard]] std::size_t
    modelIndex(Eigen::Index mode) const
    {
        EMRYS_FILTER_ASSERT(mode >= 0 && mode < modeSpace().cardinality() && "the mode is out of range");
        if constexpr (Shared)
        {
            return 0;
        }
        else
        {
            auto const location = selector();
            return std::size_t((mode / location.stride) % location.cardinality);
        }
    }

    /** \returns \a f applied to the process model that the mode with flat index \a mode follows */
    template <typename F>
    decltype(auto)
    visit(Eigen::Index mode, F && f) const
    {
        return visitEntry(mode, [&](auto const & entry) -> decltype(auto) {
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
        return visitEntry(mode, [&](auto const & entry) -> decltype(auto) {
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
        constexpr auto each = std::array<StaticPartition, sizeof...(Models)>{
            internal::staticPartition<internal::ModeModel<Models>, Space>()...
        };
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
        return std::apply(
            [&](auto const &... models) {
                return (internal::modeModel(models).partitions(space) && ...);
            },
            m_models
        );
    }

private:

    /** \returns \a f applied to the entry, the model or ModelWithMethod, that the mode with flat
     * index \a mode follows
     */
    template <typename F>
    decltype(auto)
    visitEntry(Eigen::Index mode, F const & f) const
    {
        std::size_t const index = modelIndex(mode);
        return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto) {
            using Result = decltype(f(std::get<0>(m_models)));
            if constexpr (std::is_void_v<Result>)
            {
                ((I == index ? (void)f(std::get<I>(m_models)) : void()), ...);
            }
            else
            {
                auto result = Result{};
                ((I == index ? (void)(result = f(std::get<I>(m_models))) : void()), ...);
                return result;
            }
        }(std::index_sequence_for<Models...>{});
    }

    /** \a other with each of its models, at indices \a I, converted to this one's */
    template <typename Other, std::size_t... I>
    SwitchingModel(Other const & other, std::index_sequence<I...>)
        : SwitchingModel{other.chain(), Models(other.template model<I>())...}
    {}

    /** Where the selecting axis sits in a mode's flat index. */
    struct Selector
    {
        Eigen::Index stride;       ///< the axis's stride
        Eigen::Index cardinality;  ///< the axis's number of values
    };

    /** \returns the stride and cardinality of the axis named by \a Path_ */
    [[nodiscard]] Selector
    selector() const
    {
        auto const [stride, axis] = internal::locateAxisByPath<Path_>(modeSpace());
        return {stride, axis->cardinality()};
    }

    ChainType m_chain;               ///< the chain on the modes
    std::tuple<Models...> m_models;  ///< the process models, in the axis's order
};

/** \ingroup Models
 * A SwitchingModel whose process models are all FixedStepModel is fixed-step too.
 */
template <typename ModeSpace_, typename Scalar_, typename Path_, typename... Models>
inline constexpr bool isFixedStep<SwitchingModel<ModeSpace_, Scalar_, Path_, Models...>>{(isFixedStep<Models> && ...)};

namespace internal
{

/** The method a SwitchingModel's filter predicts the modes by that carry none of their own:
 * Linearized, or, with fixed-step models, which each carry their own, ModelsMethod, there being no
 * such mode.
 */
template <typename ModeSpace_, typename Scalar_, typename Path_, typename First, typename... Rest>
struct DefaultMethodOf<SwitchingModel<ModeSpace_, Scalar_, Path_, First, Rest...>>
{
    /// the method
    using type = std::conditional_t<isFixedStep<First> || isUnboundStep<First>, ModelsMethod, methods::Linearized>;
};

/** \returns ModelsMethod, the method of a filter whose switching model's modes are fixed-step and
 * each predict by their own
 */
template <typename ModeSpace_, typename Scalar_, typename Path_, typename... Models>
[[nodiscard]] ModelsMethod
predictionMethod(SwitchingModel<ModeSpace_, Scalar_, Path_, Models...> const &, ModelsMethod method)
{
    return method;
}

/** A SwitchingModel's filter over \a Space and \a Scalar binds each of its models. */
template <typename ModeSpace_, typename Scalar_, typename Path_, typename... Models, typename Space, typename Scalar>
struct BoundStepOf<SwitchingModel<ModeSpace_, Scalar_, Path_, Models...>, Space, Scalar>
{
    using type = SwitchingModel<ModeSpace_, Scalar_, Path_, BoundStep<Models, Space, Scalar>...>;  ///< the model
};

/** A filter predicts a SwitchingModel by any method, which serves the modes that carry none of their
 * own, unless the models are fixed-step: then every mode carries its own, and the filter's method is
 * ModelsMethod.
 */
template <typename ModeSpace_, typename Scalar_, typename Path_, typename... Models, typename Method>
inline constexpr bool predictsBy<SwitchingModel<ModeSpace_, Scalar_, Path_, Models...>, Method>{
    !isFixedStep<SwitchingModel<ModeSpace_, Scalar_, Path_, Models...>> || std::is_same_v<Method, ModelsMethod>
};

}  // namespace internal

/// \cond DEDUCTION_GUIDES
template <typename ModeSpace, typename Scalar, typename... Models>
SwitchingModel(MarkovChain<ModeSpace, Scalar>, Models...) -> SwitchingModel<ModeSpace, Scalar, ModeSpace, Models...>;

/// \endcond

/** \ingroup Models
 * \returns a SwitchingModel on the modes of \a chain in which the mode's value in the axis
 * named by the path \a P (a tag or a TagPath) selects the process model: \a models has one per
 * value of that axis, in order. <tt>switchingModel<Maneuver>(chain, cruise, turn)</tt> on
 * a mode space <tt>DiscreteProduct<Maneuver, Sensor></tt> gives both sensor states of a maneuver
 * the same dynamics.
 */
template <typename P, typename ModeSpace, typename Scalar, typename... Models>
[[nodiscard]] SwitchingModel<ModeSpace, Scalar, P, Models...>
switchingModel(MarkovChain<ModeSpace, Scalar> chain, Models... models)
{
    return {std::move(chain), std::move(models)...};
}

/** \ingroup Models
 * \brief A measurement that switches between modes: one Measurement per value of a mode axis, so
 * that a filter over a Hybrid updates each mode by the measurement that its value in that axis
 * names [imm].
 *
 * A sensor may see the state differently in each mode: with a bias or a scale factor that the mode
 * sets, through a function that holds in one regime only, or not at all, when a blocked sensor
 * reads clutter. <tt>switchingMeasurement<Sensor>(nominal, biased)</tt> on a mode space
 * <tt>DiscreteProduct<Maneuver, Sensor></tt> updates each mode whose \c Sensor value is the first by
 * \c nominal and each other mode by \c biased, and each mode's own likelihood
 * \f$ p(z \mid m) \f$ reweights it. The measurements may differ in model, function, noise, and
 * method, as long as every mode's update has one type: one measurement space, and noise that is
 * %Gaussian in all of them or a mixture in all of them. None of them has noise by mode axis.
 *
 * It is the counterpart of SwitchingModel for an update: where switchingModel<Maneuver>() selects
 * the dynamics by an axis, switchingMeasurement<Sensor>() selects the measurement. Noise by mode
 * axis, <tt>additiveNoise(mixture).byMode<Sensor>()</tt> (see ModeNoise), is the case in which the
 * measurements differ in their noise alone, the components of one measurement's mixture noise
 * being the axis's values.
 *
 * \tparam Path_         the mode axis that selects the measurement, as a tag or TagPath
 * \tparam Measurements  the measurements, in the order of the axis's values
 *
 * \sa switchingMeasurement(), SwitchingModel, ImmFilter
 */
template <typename Path_, Measurement... Measurements>
class SwitchingMeasurement
{
    static_assert(sizeof...(Measurements) > 0, "a switching measurement needs a measurement");

public:

    using PathType = Path_;  ///< the mode axis that selects the measurement

    static constexpr std::size_t Count = sizeof...(Measurements);  ///< the number of measurements

    /** Measurement \a I, for value \a I of the axis. */
    template <std::size_t I>
    using MeasurementType = std::tuple_element_t<I, std::tuple<Measurements...>>;

    /** \a measurements, one per value of the axis \a Path_, in order. */
    explicit SwitchingMeasurement(Measurements... measurements)
        : m_measurements{std::move(measurements)...}
    {
        using Axis = internal::PathTag<Path_>;
        static_assert(
            Axis::Cardinality == Eigen::Dynamic || Axis::Cardinality == int(sizeof...(Measurements)),
            "give one measurement per value of the axis that selects it"
        );
    }

    /** \returns measurement \a I, for value \a I of the axis */
    template <std::size_t I>
    [[nodiscard]] MeasurementType<I> const &
    measurement() const
    {
        return std::get<I>(m_measurements);
    }

    /** \returns \a f applied to the measurement for value \a value of the axis, which is less than
     * Count; \a f returns one type for every measurement
     */
    template <typename F>
    decltype(auto)
    visit(std::size_t value, F && f) const
    {
        EMRYS_FILTER_ASSERT(value < Count && "the switching measurement has no measurement for this value of the axis");
        return visitFrom<0>(value, f);
    }

private:

    /** \returns \a f applied to the measurement for \a value, looked for from measurement \a I on */
    template <std::size_t I, typename F>
    decltype(auto)
    visitFrom(std::size_t value, F & f) const
    {
        if constexpr (I + 1 == Count)
        {
            return f(std::get<I>(m_measurements));
        }
        else
        {
            if (value == I)
            {
                return f(std::get<I>(m_measurements));
            }
            return visitFrom<I + 1>(value, f);
        }
    }

    std::tuple<Measurements...> m_measurements;  ///< the measurements, in the axis's order
};

/** \ingroup Models
 * \returns a SwitchingMeasurement in which the mode's value in the axis named by the path \a P (a
 * tag or a TagPath) selects the measurement: \a measurements has one per value of that axis, in
 * order. <tt>filter.update(switchingMeasurement<Sensor>(nominal, biased))</tt> updates the modes of
 * an ImmFilter over <tt>DiscreteProduct<Maneuver, Sensor></tt> whose \c Sensor value is the first by
 * \c nominal, and the others by \c biased.
 */
template <typename P, Measurement... Measurements>
[[nodiscard]] SwitchingMeasurement<P, Measurements...>
switchingMeasurement(Measurements... measurements)
{
    return SwitchingMeasurement<P, Measurements...>{std::move(measurements)...};
}

namespace internal
{

/** Whether \a T is a SwitchingMeasurement. */
template <typename T>
inline constexpr bool isSwitchingMeasurement = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Path, typename... Measurements>
inline constexpr bool isSwitchingMeasurement<SwitchingMeasurement<Path, Measurements...>> = true;

/// \endcond

}  // namespace internal

}  // namespace Emrys
