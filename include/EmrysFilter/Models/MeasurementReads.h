#pragma once

/** \file
 * How an update reads a measurement, all internal (decisions.md [model-policies], [measurement-space-functions],
 * [additive-noise], [method-per-measurement]): ResolvedMeasurement, an observation resolved against a state space,
 * which builds on ResolvedReads.h's resolution of the model's reads() against the state and the call's arguments,
 * builds the output space, places the additive noise, calls the function with its input gathered once per evaluation,
 * and splits a returned Jacobian by input; the predicted measurement's moments; and the covariance the uncertain
 * inputs carry through the function. Read IteratedUpdate.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/MeasurementUpdate.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/ResolvedReads.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** What a ResolvedMeasurement holds in place of additive noise when its model has none. */
struct NoNoise
{};

/** An observation \a M (an Observed) resolved against the state space \a Space (decisions.md
 * [model-policies]): its model's reads() resolved against the state and the call's arguments
 * (ResolvedReads), its output space, the additive noise placed in the output space, and the
 * function called with its input gathered once per evaluation, a returned Jacobian split by input.
 * An update resolves its measurement once, at its start, and reads this. Valid while the
 * observation and the space are.
 *
 * \a M may also wrap an observation with noise of its own (WithNoiseComponent), marked by \c OwnNoise,
 * whose noise() is then read in place of the model's policy.
 */
template <typename M, typename Space>
class ResolvedMeasurement : public ResolvedReads<typename M::ModelType, typename M::ArgumentsType, Space>
{
    /** The reads() resolved, which this builds on. */
    using Base = ResolvedReads<typename M::ModelType, typename M::ArgumentsType, Space>;

public:

    using ObservedType = M;                         ///< the observation
    using Model = typename M::ModelType;            ///< the model
    using Function = typename Model::FunctionType;  ///< its function, or NoFunction
    using Method = typename M::MethodType;          ///< the method, or NoMethod

    static constexpr bool HasFunction = Model::HasFunction;      ///< whether the model has a function
    static constexpr bool HasMethod = M::HasMethod;              ///< whether a method was given
    static constexpr bool HasObservation = M::HasObservation;    ///< whether an observed value was given
    static constexpr bool OwnNoise = requires { M::OwnNoise; };  ///< whether \a M brings its own noise

    /// whether the measurement has additive noise, from the model's policy or its own
    static constexpr bool HasAdditiveNoise = OwnNoise || Model::template has<PolicyKind::additiveNoise>;

    /// whether the model has a likelihood of its own, logLikelihood(g)
    static constexpr bool HasLogLikelihood = Model::template has<PolicyKind::logLikelihood>;

    /// whether the model has a draw of its observation, sample(g)
    static constexpr bool HasSample = Model::template has<PolicyKind::sample>;

    /// whether that likelihood is batched, logLikelihood(batched(g)) (decisions.md [batched-functions])
    static constexpr bool BatchedLikelihood = [] {
        if constexpr (HasLogLikelihood)
        {
            return isBatched<decltype(Model::template Policy<PolicyKind::logLikelihood>::function)>;
        }
        else
        {
            return false;
        }
    }();

    using Base::ArgumentCount;
    using Base::InputCount;
    using Base::ItemCount;
    using Base::StateCount;
    using typename Base::In;
    using typename Base::Reads;
    using typename Base::StatePaths;

    /** Item \a I of reads(). */
    template <std::size_t I>
    using Item = typename Base::template Item<I>;

    /** Argument \a J of the call. */
    template <std::size_t J>
    using Argument = typename Base::template Argument<J>;

    /** A vector in the input space, in \a T. */
    template <typename T>
    using InVector = typename Base::template InVector<T>;

private:

    using Base::argumentIsUncertain;
    using Base::argumentOf;
    using Base::inputIndexOf;
    using Base::inputItems;
    using Base::inState;
    using Base::m_differenceRows;
    using Base::m_in;
    using Base::m_reads;
    using Base::m_stateDifferenceSize;
    using Base::m_valueRows;
    using Base::stateIndexOf;
    using Base::stateItems;

    /** The scalar the output space is probed with. */
    using ProbeScalar = double;

    /// whether the output space is the space of the model's one unplaced additive noise
    static constexpr bool OutputFromNoise = [] {
        if constexpr (Model::template has<PolicyKind::outputSpace> || Model::template has<PolicyKind::writes>)
        {
            return false;
        }
        else if constexpr (Model::template has<PolicyKind::additiveNoise>)
        {
            return Model::template Policy<PolicyKind::additiveNoise>::HasOwnSpace;
        }
        else
        {
            return false;
        }
    }();

    /** \returns the output space's type, wrapped (decisions.md [sensors]): from outputSpace(), from
     * writes(), the space of the one unplaced additive noise, from a function that names it, or a
     * Linear sized by what the function returns; NoOutputSpace for a model with no function
     */
    [[nodiscard]] static consteval auto
    outputType()
    {
        if constexpr (Model::template has<PolicyKind::outputSpace>)
        {
            using Policy = typename Model::template Policy<PolicyKind::outputSpace>;
            return std::type_identity<decltype(std::declval<Policy const &>().in(std::declval<Space const &>()))>{};
        }
        else if constexpr (Model::template has<PolicyKind::writes>)
        {
            return std::type_identity<typename Model::template Policy<PolicyKind::writes>::OutputSpace>{};
        }
        else if constexpr (OutputFromNoise)
        {
            return std::type_identity<
                typename Model::template Policy<PolicyKind::additiveNoise>::Distribution::SpaceType>{};
        }
        else if constexpr (requires { typename Function::OutputSpaceType; })
        {
            return std::type_identity<typename Function::OutputSpaceType>{};
        }
        else if constexpr (HasFunction && functionForm<Function, In, NoOutputSpace, InVector<ProbeScalar>> == 0)
        {
            // A function in no form: instantiating callForm() gives its message, and a space of one
            // coordinate stands in for the output's, so that what follows adds no errors about it.
            using Refused [[maybe_unused]] = decltype(callForm(
                std::declval<Function const &>(),
                std::declval<In const &>(),
                std::declval<NoOutputSpace const &>(),
                std::declval<InVector<ProbeScalar> const &>()
            ));
            return std::type_identity<Linear<1>>{};
        }
        else if constexpr (HasFunction)
        {
            using Returned = decltype(vectorReturned(callForm(
                std::declval<Function const &>(),
                std::declval<In const &>(),
                std::declval<NoOutputSpace const &>(),
                std::declval<InVector<ProbeScalar> const &>()
            )));
            using Evaluated = std::remove_cvref_t<decltype(valueOf(std::declval<Returned>()).eval())>;
            static_assert(Evaluated::ColsAtCompileTime == 1, "a measurement model's function returns a column vector");
            return std::type_identity<Linear<Evaluated::RowsAtCompileTime, Evaluated::MaxRowsAtCompileTime>>{};
        }
        else
        {
            return std::type_identity<NoOutputSpace>{};
        }
    }

public:

    /** The output space. */
    using Out = typename decltype(outputType())::type;

    static constexpr bool HasOutputSpace = !std::is_same_v<Out, NoOutputSpace>;  ///< whether there is one

    /** The additive noise, placed in the output space (placeNoise()), the observation's own, or NoNoise. */
    using NoiseType = typename decltype([] {
        if constexpr (OwnNoise)
        {
            return std::type_identity<typename M::NoiseType>{};
        }
        else if constexpr (HasAdditiveNoise)
        {
            static_assert(
                HasOutputSpace,
                "additive noise needs the model's output space, from writes(), outputSpace(), or its function's result"
            );
            return std::type_identity<decltype(placeNoise(
                std::declval<Out const &>(),
                std::declval<Model const &>().template policy<PolicyKind::additiveNoise>()
            ))>{};
        }
        else
        {
            return std::type_identity<NoNoise>{};
        }
    }())::type;

    /// whether the additive noise is a mixture, so an update has a branch per component
    static constexpr bool MixtureNoise = [] {
        if constexpr (HasAdditiveNoise)
        {
            return isGaussianMixture<typename NoiseType::Distribution>;
        }
        else
        {
            return false;
        }
    }();

    /** The repair of an innovation covariance that is not positive definite, as the observation
     * names it (decisions.md [innovation-cholesky]).
     */
    using InnovationRepair = typename M::InnovationRepair;

    /** The blocks the update considers, as the observation names them (decisions.md [consider-states]). */
    using ConsiderType = typename M::ConsiderType;

    /** Resolves \a measurement against \a space, which both outlive this. */
    ResolvedMeasurement(M const & measurement, Space const & space)
        : Base{measurement.arguments(), space}
        , m_measurement{measurement}
        , m_out{makeOut(measurement, space)}
        , m_noise{makeNoise(measurement)}
    {}

    /** \returns the observation */
    [[nodiscard]] M const &
    observed() const
    {
        return m_measurement;
    }

    /** \returns the model */
    [[nodiscard]] Model const &
    model() const
    {
        return m_measurement.model();
    }

    /** \returns the method */
    [[nodiscard]] Method const &
    method() const
    {
        return m_measurement.method();
    }

    /** \returns the observed value */
    [[nodiscard]] decltype(auto)
    observation() const
        requires HasObservation
    {
        return m_measurement.observation();
    }

    /** \returns the covariance update form */
    [[nodiscard]] decltype(auto)
    covarianceUpdate() const
    {
        return m_measurement.covarianceUpdate();
    }

    /** \returns the output space instance, the measurement space */
    [[nodiscard]] Out const &
    space() const
        requires HasOutputSpace
    {
        return m_out;
    }

    /** \returns the additive noise, placed in the output space */
    [[nodiscard]] NoiseType const &
    noise() const
        requires HasAdditiveNoise
    {
        return m_noise;
    }

    /** \returns the function at the blocks read, given as \a joint, a vector over the joint space of
     * reads(), and at the uncertain inputs \a inputs, one vector each in the order of inputs(): its
     * value, or a WithJacobian whose blocks are the joint's and then each input's (decisions.md
     * [returned-jacobians])
     */
    template <typename Joint, typename... Inputs>
    [[nodiscard]] auto
    evaluate(Joint const & joint, Inputs const &... inputs) const
    {
        static_assert(sizeof...(Inputs) == InputCount, "the function takes one vector per uncertain input");
        using T = typename Joint::Scalar;
        return split<false>(call(this->template gather<T>(joint, std::tuple<Inputs const &...>{inputs...})));
    }

    /** \returns the function at the whole state \a state, a vector in the state space, and at the
     * uncertain inputs \a inputs, one vector each in the order of inputs(), or at their means when
     * none is given: its value, or a WithJacobian whose blocks are one over the whole state's
     * differences, zero outside the blocks read, and then each input's
     */
    template <typename State, typename... Inputs>
    [[nodiscard]] auto
    evaluateAtState(State const & state, Inputs const &... inputs) const
    {
        static_assert(
            sizeof...(Inputs) == InputCount || sizeof...(Inputs) == 0,
            "the function takes one vector per uncertain input, or none to read each at its mean"
        );
        using T = typename State::Scalar;
        return split<true>(
            call(this->template gatherFromRows<T>(state, Eigen::Index{0}, std::tuple<Inputs const &...>{inputs...}))
        );
    }

    /** \returns the model's own \f$ \log p(z \mid x) \f$ at the whole state \a state, from its
     * logLikelihood() policy, <tt>g(in, x, z)</tt>, with each uncertain input at its mean
     */
    template <typename State>
    [[nodiscard]] auto
    logLikelihoodAt(State const & state) const
        requires(HasLogLikelihood && HasObservation)
    {
        return logLikelihoodAtRows(state, Eigen::Index{0});
    }

    /** \returns the model's own \f$ \log p(z \mid x) \f$ at the rows \a rows of a state from row
     * \a start on, which must hold every block read, with each uncertain input at its mean
     */
    template <typename Rows>
    [[nodiscard]] auto
    logLikelihoodAtRows(Rows const & rows, Eigen::Index start) const
        requires(HasLogLikelihood && HasObservation)
    {
        using T = typename Rows::Scalar;
        if constexpr (BatchedLikelihood)
        {
            // A batch of one column.
            auto x = InMatrix<T>{};
            x.resize(valueDimension(m_in), 1);
            this->template gatherColumnsFromRows<T>(
                x,
                [&](Eigen::Index) -> Rows const & {
                    return rows;
                },
                start
            );
            auto result = Eigen::Matrix<T, 1, 1>{T(0)};
            batchedLogLikelihoods(x, result);
            return result(0);
        }
        else
        {
            return callWithScalar<T>(
                model().template policy<PolicyKind::logLikelihood>().function,
                m_in,
                this->template gatherFromRows<T>(rows, start, std::tuple<>{}),
                m_measurement.observation()
            );
        }
    }

    /** \returns a draw of the observation at the whole state \a state from \a rng, by the model's
     * sample() policy, <tt>g(in, x, rng)</tt>, with the uncertain inputs at the vectors \a inputs,
     * one per input, or at their means when none is given
     */
    template <typename State, typename G, typename... Inputs>
    [[nodiscard]] auto
    sampleAtState(State const & state, G & rng, Inputs const &... inputs) const
        requires HasSample
    {
        static_assert(
            sizeof...(Inputs) == InputCount || sizeof...(Inputs) == 0,
            "the draw takes one vector per uncertain input, or none to read each at its mean"
        );
        using T = typename State::Scalar;
        return callWithScalar<T>(
            model().template policy<PolicyKind::sample>().function,
            m_in,
            this->template gatherFromRows<T>(state, Eigen::Index{0}, std::tuple<Inputs const &...>{inputs...}),
            rng
        );
    }

    /** A matrix whose columns are vectors in the input space, in \a T: a batched function's input. */
    template <typename T>
    using InMatrix = MatrixOf<
        T,
        valueDimensionAtCompileTime<In>,
        Eigen::Dynamic,
        maxValueDimensionAtCompileTime<In>,
        Eigen::Dynamic>;

    /// whether the function is batched, batched(h) (decisions.md [batched-functions])
    static constexpr bool BatchedFunction = takesBatches<Function>;

    /** A matrix whose columns are vectors in the output space, in \a T: a batched function's output. */
    template <typename T, typename S = Out>
    using OutMatrix =
        MatrixOf<T, valueDimensionAtCompileTime<S>, Eigen::Dynamic, maxValueDimensionAtCompileTime<S>, Eigen::Dynamic>;

    /** Calls the batched function once over \a x, the inputs of a batch, a column per point, writing
     * into \a y a column per point in the output space, with no Jacobian wanted (decisions.md
     * [batched-functions]).
     */
    template <typename X, typename Y>
    void
    evaluateBatch(X const & x, Y && y) const
        requires BatchedFunction
    {
        callBatched(m_measurement.model().function(), m_in, m_out, x, y);
    }

    /** Writes into \a y the function at \a x, the inputs of a batch, a column per point, by one call
     * of the batched function, and into \a jacobians its Jacobian over the uncertain inputs, in the
     * order of inputs(), a block of columns per point (internal::callBatchedWithJacobians())
     */
    template <typename X, typename Y, typename J>
    void
    evaluateBatchWithInputJacobians(X const & x, Y & y, J & jacobians) const
        requires BatchedFunction
    {
        auto wanted = std::array<bool, std::tuple_size_v<InputBlocks<In>>>{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((wanted[I] = argumentIsUncertain(I)), ...);
        }(std::make_index_sequence<ItemCount>{});
        callBatchedWithJacobians(m_measurement.model().function(), m_in, m_out, x, wanted, y, jacobians);
    }

    /** \returns the inputs of a batch, a column per point in the input space, as many as \a Y has:
     * column \a j gathered from column \a j of \a joint, the blocks read, and of each of \a inputs,
     * the uncertain inputs in the order of inputs()
     */
    template <typename Y, typename Joint, typename... Inputs>
    [[nodiscard]] auto
    gatherColumns(Joint const & joint, Inputs const &... inputs) const
    {
        static_assert(sizeof...(Inputs) == InputCount);
        using T = typename Joint::Scalar;
        using X = MatrixOf<
            T,
            valueDimensionAtCompileTime<In>,
            Y::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<In>,
            Y::MaxColsAtCompileTime>;
        auto x = X{};
        x.resize(valueDimension(m_in), joint.cols());
        for (Eigen::Index j = 0; j < joint.cols(); ++j)
        {
            x.col(j) = this->template gather<T>(joint.col(j), std::tuple{inputs.col(j)...});
        }
        return x;
    }

    /** Writes into \a y the function at a batch of points, a column each in the output space, by one
     * call of the batched function: column \a j at column \a j of \a joint, the blocks read as
     * evaluate() takes them, and of each of \a inputs, the uncertain inputs in the order of inputs()
     * (decisions.md [batched-functions]).
     */
    template <typename Y, typename Joint, typename... Inputs>
    void
    evaluateColumns(Y & y, Joint const & joint, Inputs const &... inputs) const
        requires BatchedFunction
    {
        callBatched(m_measurement.model().function(), m_in, m_out, gatherColumns<Y>(joint, inputs...), y);
    }

    /** Writes into \a y the function at a batch of points by one call of the batched function, as
     * evaluateAtState() gives it at each: column \a j at column \a j of \a states, a column each in
     * the state space, and of each of \a inputs, the uncertain inputs in the order of inputs()
     */
    template <typename Y, typename States, typename... Inputs>
    void
    evaluateStateColumns(Y & y, States const & states, Inputs const &... inputs) const
        requires BatchedFunction
    {
        static_assert(sizeof...(Inputs) == InputCount);
        using T = typename States::Scalar;
        using X = MatrixOf<
            T,
            valueDimensionAtCompileTime<In>,
            Y::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<In>,
            Y::MaxColsAtCompileTime>;
        auto x = X{};
        x.resize(valueDimension(m_in), states.cols());
        for (Eigen::Index j = 0; j < states.cols(); ++j)
        {
            x.col(j) = this->template gatherFromRows<T>(states.col(j), Eigen::Index{0}, std::tuple{inputs.col(j)...});
        }
        callBatched(m_measurement.model().function(), m_in, m_out, x, y);
    }

    /** evaluateColumns() with the function's Jacobian over the items the model's affineIn() factors
     * out written into \a jacobians, a block of columns per point in the order of reads(), the
     * coordinates affineCoordinates() gives (internal::callBatchedWithJacobians())
     */
    template <typename Y, typename J, typename Joint, typename... Inputs>
    void
    evaluateColumnsWithAffineJacobians(Y & y, J & jacobians, Joint const & joint, Inputs const &... inputs) const
        requires BatchedFunction
    {
        auto wanted = std::array<bool, std::tuple_size_v<InputBlocks<In>>>{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((wanted[I] = this->template itemFactored<I>()), ...);
        }(std::make_index_sequence<ItemCount>{});
        auto const x = gatherColumns<Y>(joint, inputs...);
        callBatchedWithJacobians(m_measurement.model().function(), m_in, m_out, x, wanted, y, jacobians);
    }

    /** Writes the model's own \f$ \log p(z \mid x_j) \f$ into \a result, a row with an entry per
     * column \f$ x_j \f$ of \a x, the inputs of a batch, by one call of its batched
     * logLikelihood(), <tt>g(in, X, z, Y)</tt> (decisions.md [batched-functions]).
     */
    template <typename X, typename Result>
    void
    batchedLogLikelihoods(X const & x, Result && result) const
        requires(BatchedLikelihood && HasObservation)
    {
        using T = typename X::Scalar;
        using Likelihood = typename Model::template Policy<PolicyKind::logLikelihood>;
        using Output = BatchedOutput<In, Linear<1>, T, typename decltype(Likelihood::function)::MarkType>;
        EMRYS_FILTER_ASSERT(result.size() == x.cols() && "give one entry per column of the batch");
        auto y = Output{
            typename Output::ValuesType{result.data(), 1, x.cols()},
            typename Output::JacobiansType{nullptr, 1, 0},
            m_in,
            {}
        };
        callWithScalar<T>(
            model().template policy<PolicyKind::logLikelihood>().function.function,
            m_in,
            x,
            m_measurement.observation(),
            y
        );
    }

private:

    /** \returns the output space instance in \a space: from outputSpace(), the noise's space, the
     * empty instance of a fixed-size space, or a Linear sized by a function that knows its rows
     */
    [[nodiscard]] static Out
    makeOut([[maybe_unused]] M const & measurement, [[maybe_unused]] Space const & space)
    {
        if constexpr (Model::template has<PolicyKind::outputSpace>)
        {
            return measurement.model().template policy<PolicyKind::outputSpace>().in(space);
        }
        else if constexpr (OutputFromNoise)
        {
            return measurement.model().template policy<PolicyKind::additiveNoise>().distribution().space();
        }
        else if constexpr (FixedSizeSpace<Out> || !HasOutputSpace)
        {
            return Out{};
        }
        else
        {
            static_assert(
                requires(Function const & f) { f.outputRows(); },
                "a function that returns a vector of run-time size names its output with writes() or outputSpace()"
            );
            return Out{measurement.model().function().outputRows()};
        }
    }

    /** \returns the additive noise: the observation's own, the model's placed in the output space,
     * or NoNoise
     */
    [[nodiscard]] NoiseType
    makeNoise([[maybe_unused]] M const & measurement) const
    {
        if constexpr (OwnNoise)
        {
            return measurement.noise();
        }
        else if constexpr (HasAdditiveNoise)
        {
            return placeNoise(m_out, measurement.model().template policy<PolicyKind::additiveNoise>());
        }
        else
        {
            return {};
        }
    }

    /** \returns the function called at the input vector \a x, an expression it returned evaluated
     * and a scalar wrapped as a vector of one coordinate
     */
    template <typename X>
    [[nodiscard]] auto
    call(X const & x) const
    {
        static_assert(HasFunction, "a measurement model known by its likelihood alone has no function to evaluate");
        return vectorReturned(callForm(m_measurement.model().function(), m_in, m_out, x));
    }

    /** \returns what the function returned, \a returned, with a Jacobian it returned over the whole
     * input split into a block for the blocks read and a block per uncertain input, as the transform
     * takes them: the blocks read in their joint difference coordinates, or when \a Whole over the
     * whole state's differences, zero outside them
     */
    template <bool Whole, typename R>
    [[nodiscard]] auto
    split(R && returned) const
    {
        using Plain = std::remove_cvref_t<R>;
        if constexpr (!isWithJacobian<Plain>)
        {
            return std::forward<R>(returned);
        }
        else
        {
            using J = std::remove_cvref_t<decltype(returned.jacobian)>;
            static_assert(
                MatrixExpression<J>,
                "a measurement model's function returns its Jacobian as one matrix over its input's differences, "
                "withJacobian(y, J)"
            );
            auto const & jacobian = returned.jacobian;
            EMRYS_FILTER_ASSERT(
                jacobian.cols() == differenceDimension(m_in)
                && "a returned Jacobian has a column per difference coordinate of the model's input"
            );
            using Scalar = typename J::Scalar;
            constexpr int Rows = J::RowsAtCompileTime;
            constexpr int MaxRows = J::MaxRowsAtCompileTime;
            constexpr int Cols = Whole ? valueDimensionAtCompileTime<DifferenceSpace<Space>> : Reads::DifferenceSize;
            constexpr int MaxCols =
                Whole ? maxValueDimensionAtCompileTime<DifferenceSpace<Space>> : Reads::DifferenceMaxSize;
            using StateBlock = MatrixOf<Scalar, Rows, Cols, MaxRows, MaxCols>;
            StateBlock stateBlock =
                StateBlock::Zero(jacobian.rows(), Whole ? m_stateDifferenceSize : m_reads.differenceIndices().size());
            [&]<std::size_t... K>(std::index_sequence<K...>) {
                ((stateBlock.middleCols(
                      Whole ? m_reads.differenceRows(K).start : m_reads.jointDifferenceStart(K),
                      m_differenceRows[stateItems(K)].size
                  ) = jacobian.middleCols(m_differenceRows[stateItems(K)].start, m_differenceRows[stateItems(K)].size)),
                 ...);
            }(std::make_index_sequence<StateCount>{});
            if constexpr (InputCount == 0)
            {
                return withJacobian(std::move(returned.value), std::move(stateBlock));
            }
            else
            {
                auto blocks = [&]<std::size_t... P>(std::index_sequence<P...>) {
                    return std::tuple{std::move(stateBlock), inputBlock<P>(jacobian)...};
                }(std::make_index_sequence<InputCount>{});
                return withJacobian(std::move(returned.value), std::move(blocks));
            }
        }
    }

    /** \returns the columns of \a jacobian for uncertain input \a P */
    template <std::size_t P, typename J>
    [[nodiscard]] auto
    inputBlock(J const & jacobian) const
    {
        using Input =
            std::remove_cvref_t<decltype(inputOf(std::declval<Argument<argumentOf(inputItems(P))> const &>()))>;
        using Block = MatrixOf<
            typename J::Scalar,
            J::RowsAtCompileTime,
            Input::DifferenceDimensionAtCompileTime,
            J::MaxRowsAtCompileTime,
            Input::MaxDifferenceDimensionAtCompileTime>;
        BlockRows const & rows = m_differenceRows[inputItems(P)];
        Block block = jacobian.middleCols(rows.start, rows.size);
        return block;
    }

    M const & m_measurement;                           ///< the observation
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Out m_out;          ///< the output space instance
    EMRYS_FILTER_NO_UNIQUE_ADDRESS NoiseType m_noise;  ///< the additive noise, placed
};

/** The function of the resolved measurement \a R as the transforms call it: at the blocks read, a
 * vector over the joint space of reads(), and a vector per uncertain input (ResolvedMeasurement::evaluate()),
 * and for a batched function also at a batch of points in one call (decisions.md
 * [batched-functions]). Valid while the resolved measurement is.
 */
template <typename R>
class ResolvedFunction
{
public:

    /// whether the transforms evaluate a batch of points in one call, evaluateBatch()
    static constexpr bool EvaluatesBatches = R::BatchedFunction;

    /** The function of \a resolved, which outlives this. */
    explicit ResolvedFunction(R const & resolved)
        : m_resolved{resolved}
    {}

    /** \returns the function at the blocks read, \a joint, and at the uncertain inputs \a inputs */
    template <typename Joint, typename... Inputs>
    [[nodiscard]] auto
    operator()(Joint const & joint, Inputs const &... inputs) const
    {
        return m_resolved.evaluate(joint, inputs...);
    }

    /** Writes into \a y the function at the points that are the columns of \a joint and of each of
     * \a inputs, by one call of the batched function
     */
    template <typename Y, typename Joint, typename... Inputs>
    void
    evaluateBatch(Y & y, Joint const & joint, Inputs const &... inputs) const
        requires EvaluatesBatches
    {
        m_resolved.evaluateColumns(y, joint, inputs...);
    }

    /** evaluateBatch() with the Jacobian over the coordinates the factored transform takes as affine
     * written into \a jacobians, a block of columns per point
     */
    template <typename Y, typename J, typename Joint, typename... Inputs>
    void
    evaluateBatchWithAffineJacobians(Y & y, J & jacobians, Joint const & joint, Inputs const &... inputs) const
        requires EvaluatesBatches
    {
        m_resolved.evaluateColumnsWithAffineJacobians(y, jacobians, joint, inputs...);
    }

private:

    R const & m_resolved;  ///< the resolved measurement
};

/** \returns \a measurement resolved against the state space \a space (see ResolvedMeasurement) */
template <typename Space, typename M>
[[nodiscard]] ResolvedMeasurement<M, Space>
resolveMeasurement(M const & measurement, Space const & space)
{
    return ResolvedMeasurement<M, Space>{measurement, space};
}

/** \returns the observed value of \a resolved in \a space, its measurement space, in the scalar
 * \a Scalar and normalized there, as values from outside are. Every Kalman update reads its
 * observation here, so a NaN or infinite one, which would make the posterior NaN with the update
 * accepted, is asserted against here.
 */
template <typename Scalar, typename R, typename Space>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
observedValue(R const & resolved, Space const & space)
{
    using Vector = typename Moments<Space, Scalar>::VectorType;
    auto const & observation = resolved.observation();
    EMRYS_FILTER_ASSERT(
        observation.size() == valueDimension(space) && "the observation must be a value in the measurement space"
    );
    Vector result = observation.template cast<Scalar>();
    EMRYS_FILTER_ASSERT(result.allFinite() && "the observation must be finite");
    normalizeValues(space, result);
    return result;
}

/** \returns the residual \f$ z \boxminus \hat z \f$ of the observation of \a resolved from
 * \a predicted, a value in \a space, its measurement space
 */
template <typename Scalar, typename R, typename Space, typename Predicted>
[[nodiscard]] typename Moments<Space, Scalar>::DifferenceVectorType
observedResidual(R const & resolved, Space const & space, Eigen::MatrixBase<Predicted> const & predicted)
{
    using Difference = typename Moments<Space, Scalar>::DifferenceVectorType;
    return Difference{Emrys::minus(space, observedValue<Scalar>(resolved, space), predicted)};
}

/** The positional transform by \a method of the model's function \a f into the measurement space of \a resolved,
 * over \a args.
 */
template <typename R, typename Method, typename F, typename... Args>
[[nodiscard]] auto
transformPrediction(R const & resolved, Method const & method, F const & f, Args const &... args)
{
    return transformBy(method, f, resolved.space(), args...);
}

/** transformPrediction() of the blocks read, \a read, marked for the cross-covariance, and of the
 * call's uncertain inputs after it, with the coordinates the model's affineIn() names factored out
 * of a sigma-point method (decisions.md [affine-coordinates]).
 */
template <typename R, typename Method, typename F, typename Read>
[[nodiscard]] auto
transformMeasurement(R const & resolved, Method const & method, F const & f, Read const & read)
{
    return std::apply(
        [&](auto const &... inputs) {
            if constexpr (isSigmaPointMethod<Method> && R::factorsAffine())
            {
                auto const factored = FactoredAt<R::affineSize(), R::affineMaxSize(), Method>{
                    method,
                    resolved.template affineCoordinates<false>()
                };
                return transformPrediction(resolved, factored, f, withCross(read), inputs...);
            }
            else
            {
                return transformPrediction(resolved, method, f, withCross(read), inputs...);
            }
        },
        resolved.inputs()
    );
}

/** Calls \a g with the Transform of the model's function \a f over \a args, into the measurement
 * space of \a resolved, and that Transform's linearizeAt() by the linear \a method.
 */
template <typename R, typename Method, typename F, typename G, typename... Args>
[[nodiscard]] decltype(auto)
linearizePrediction(R const & resolved, Method const & method, F const & f, G const & g, Args const &... args)
{
    using Problem = Transform<F const, typename R::Out, Args...>;
    auto const problem = Problem{f, resolved.space(), args...};
    return g(problem, linearizeAt(method, problem));
}

/** linearizePrediction() of the blocks read, \a read, marked for the cross-covariance, and of the
 * call's uncertain inputs after it, as transformMeasurement() lays them out.
 */
template <typename R, typename Method, typename F, typename Read, typename G>
[[nodiscard]] decltype(auto)
linearizeMeasurement(R const & resolved, Method const & method, F const & f, Read const & read, G const & g)
{
    return std::apply(
        [&](auto const &... inputs) -> decltype(auto) {
            return linearizePrediction(resolved, method, f, g, withCross(read), inputs...);
        },
        resolved.inputs()
    );
}

/** A linearized predicted measurement's space, mean, and covariance \f$ J \Sigma_{xy} \f$ as its two factors,
 * which conditionOnPrediction() reads as it would a %Gaussian's, without one being assembled. The
 * covariance is formed only when asked for, so that covariancePlus() can accumulate it into the noise.
 * Valid while its parts are.
 */
template <typename Space, typename Vector, typename Jacobian, typename Cross>
class PredictionMoments
{
public:

    /** The predicted measurement in \a space with mean \a mean and covariance \a jacobian times
     * \a cross, the Jacobian in the joint input and that input's cross-covariance with the measurement.
     */
    PredictionMoments(Space const & space, Vector const & mean, Jacobian const & jacobian, Cross const & cross)
        : m_space{space}
        , m_mean{mean}
        , m_jacobian{jacobian}
        , m_cross{cross}
    {}

    /** \returns the measurement space */
    [[nodiscard]] Space const &
    space() const
    {
        return m_space;
    }

    /** \returns the predicted mean */
    [[nodiscard]] Vector const &
    mean() const
    {
        return m_mean;
    }

    /** \returns the predicted covariance, as an expression */
    [[nodiscard]] auto
    covarianceLower() const
    {
        return multiply(m_jacobian, m_cross);
    }

    /** \returns the predicted covariance plus \a noise, as a \a Matrix, with the product accumulated
     * into a copy of \a noise
     */
    template <typename Matrix, typename Noise>
    [[nodiscard]] Matrix
    covariancePlus(Noise const & noise) const
    {
        Matrix result = noise;
        result.noalias() += multiply(m_jacobian, m_cross);
        return result;
    }

    /** \returns the length of the mean */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the rows of the covariance */
    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

private:

    Space const & m_space;        ///< the measurement space
    Vector const & m_mean;        ///< the predicted mean
    Jacobian const & m_jacobian;  ///< the Jacobian in the joint input
    Cross const & m_cross;        ///< the joint input's cross-covariance with the measurement
};

/** \returns the covariance of the predicted measurement \a y plus the noise covariance \a noise, as
 * a \a Matrix: by PredictionMoments::covariancePlus() when \a y has it, and otherwise as \a y's
 * covariance added to a copy of \a noise.
 */
template <typename Matrix, typename Output, typename Noise>
[[nodiscard]] Matrix
innovationCovariancePlus(Output const & y, Noise const & noise)
{
    if constexpr (requires { y.template covariancePlus<Matrix>(noise); })
    {
        return y.template covariancePlus<Matrix>(noise);
    }
    else
    {
        Matrix result = noise;
        result += y.covarianceLower();
        return result;
    }
}

/** Where an update puts each branch's posterior: a new %Gaussian in the branch, as update()
 * returns them in a MeasurementUpdate.
 */
struct NewPosteriors
{};

/** Where an update puts its one branch's posterior: into \a state, the prior itself, conditioned
 * in place, the update returning a FilterUpdate without posteriors (decisions.md [borrowed-intermediates]).
 * \a accept, an acceptor such as AcceptEvery, decides on the NIS before the state is written, and
 * an innovation it refuses leaves the state as it was and the update not accepted
 * [update-outcomes].
 */
template <typename G, typename Accept = AcceptEvery>
struct IntoPrior
{
    G & state;      ///< the prior, conditioned where it is
    Accept accept;  ///< whether to write the state, from the NIS and its dimension
};

/** Whether \a Sink is an IntoPrior. */
template <typename Sink>
inline constexpr bool isIntoPrior = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename G, typename Accept>
inline constexpr bool isIntoPrior<IntoPrior<G, Accept>>{true};

/// \endcond

/** Where an update of mixture noise puts its branches: into \a update, a MeasurementUpdate of a
 * run-time number of branches that is cleared and refilled where it is, and returned by reference,
 * so that a filter refilling a result allocates nothing once the result has held as many branches
 * (see \ref Allocation). The update is filled in place rather than moved out and back, since a
 * standard library with iterator debugging allocates on every move of a container. An update of
 * one branch ignores the sink.
 */
template <typename Update>
struct ReusedUpdate
{
    Update * update;  ///< the update whose storage is reused
};

/** Whether \a Sink is a ReusedUpdate. */
template <typename Sink>
inline constexpr bool isReusedUpdate = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Update>
inline constexpr bool isReusedUpdate<ReusedUpdate<Update>>{true};

/// \endcond

/** \returns the empty MeasurementUpdate of a run-time number of branches, over \a Space with
 * innovations in \a MeasurementSpace, that an update of mixture noise fills: a reference to
 * \a sink's own, cleared, where it reuses one (ReusedUpdate), and a new one by value otherwise
 */
template <typename Space, typename MeasurementSpace, typename Scalar, typename Sink>
[[nodiscard]] decltype(auto)
mixtureUpdateFor(Sink & sink)
{
    if constexpr (isReusedUpdate<Sink>)
    {
        using Update = std::remove_cvref_t<decltype(*sink.update)>;
        static_assert(
            std::is_same_v<typename Update::SpaceType, Space>
                && std::is_same_v<typename Update::MeasurementSpaceType, MeasurementSpace>
                && Update::BranchCount == std::dynamic_extent,
            "the update reused must be over the state and the innovation, with a run-time number of branches"
        );
        Update & result = *sink.update;
        result.branches.clear();
        return result;
    }
    else
    {
        return MeasurementUpdate<Space, MeasurementSpace, std::dynamic_extent, Scalar>{};
    }
}

/** What an update of mixture noise into \a Sink returns: only whether it succeeded where it
 * refilled the sink's own update (ReusedUpdate), and otherwise the new \a Update; in either case
 * the error when a component's covariance is not positive definite (decisions.md [innovation-cholesky]).
 */
template <typename Sink, typename Update>
using MixtureOutcome =
    std::conditional_t<isReusedUpdate<Sink>, std::expected<void, GaussianError>, std::expected<Update, GaussianError>>;

/** \returns the successful MixtureOutcome of \a update, which mixtureUpdateFor() gave: success
 * alone where \a Sink reuses the update, and \a update, moved, otherwise
 */
template <typename Sink, typename Update>
[[nodiscard]] MixtureOutcome<Sink, Update>
mixtureOutcome(Update & update)
{
    if constexpr (isReusedUpdate<Sink>)
    {
        return {};
    }
    else
    {
        return std::move(update);
    }
}

/** What updates::Joseph takes from a measurement's linearization (decisions.md [linearization-forms]): its Jacobian
 * \f$ H_S \f$ in the prior's difference coordinates \a indices, zero outside them, and \a error, a
 * callable returning \f$ N + \Omega \f$, the covariance the linearization leaves unexplained before
 * additive noise, which only that form calls. Each method gives both from its prediction's parts,
 * as products, so that the update needs no factorization of the prior and recovers no noise by
 * subtraction: the linear methods their Jacobian and the inputs' \f$ \sum_k J_k C_k J_k^T \f$, the
 * sigma-point methods \f$ \Psi^T S^{-1} \f$ and their \f$ N + \Omega \f$. The indices are the
 * blocks read, or every coordinate where the measurement reads the whole prior, as an update given a
 * block does.
 */
template <typename Jacobian, typename Indices, typename Error>
struct ReadLinearization
{
    Jacobian jacobian;        ///< \f$ H_S \f$, output differences by the coordinates \a indices
    Indices const & indices;  ///< the prior's difference coordinates that the Jacobian's columns are
    Error error;              ///< returns \f$ N + \Omega \f$, over the output's differences
};

/// \cond DEDUCTION_GUIDES
template <typename Jacobian, typename Indices, typename Error>
ReadLinearization(Jacobian, Indices const &, Error) -> ReadLinearization<Jacobian, Indices, Error>;

/// \endcond

/** \returns \f$ N \f$ of a linear method's prediction of \a resolved, whose joint Jacobian over the
 * Transform \a problem is \a jacobian, as a \a Matrix: \f$ \sum_k J_k C_k J_k^T \f$ over the call's
 * uncertain inputs, which follow the blocks read among the problem's inputs, and zero without them
 */
template <typename Matrix, typename R, typename Problem, typename Jacobian>
[[nodiscard]] Matrix
inputsCovariance(R const & resolved, Problem const & problem, Jacobian const & jacobian)
{
    using Scalar = typename Matrix::Scalar;
    Eigen::Index const m = jacobian.rows();
    Matrix result = Matrix::Zero(m, m);
    auto const inputs = resolved.inputs();
    [&]<std::size_t... P>(std::index_sequence<P...>) {
        [[maybe_unused]] auto const add = [&]<std::size_t K>(std::integral_constant<std::size_t, K>) {
            auto const & input = inputOf(std::get<K>(inputs));
            using Input = std::remove_cvref_t<decltype(input)>;
            using NoiseMatrix = MatrixOf<
                Scalar,
                Input::DifferenceDimensionAtCompileTime,
                Input::DifferenceDimensionAtCompileTime,
                Input::MaxDifferenceDimensionAtCompileTime,
                Input::MaxDifferenceDimensionAtCompileTime>;
            using InputJacobian = MatrixOf<
                Scalar,
                Matrix::RowsAtCompileTime,
                Input::DifferenceDimensionAtCompileTime,
                Matrix::MaxRowsAtCompileTime,
                Input::MaxDifferenceDimensionAtCompileTime>;
            Eigen::Index const d = input.differenceDimension();
            InputJacobian const block = jacobian.middleCols(problem.differenceOffset(K + 1), d);
            NoiseMatrix const cov = symmetricFromLower<NoiseMatrix>(covarianceLowerOf(input));
            result.noalias() += multiply(multiply(block, cov), block.transpose());
        };
        (add(std::integral_constant<std::size_t, P>{}), ...);
    }(std::make_index_sequence<R::InputCount>{});
    if constexpr (R::InputCount > 0)
    {
        result = (Scalar(0.5) * (result + result.transpose())).eval();
    }
    return result;
}

/** Whether the block that the path \a P names lies in none of the blocks \a Qs name but itself. */
template <typename P, typename... Qs>
inline constexpr bool liesInNoOther = ((std::is_same_v<P, Qs> || !pathLiesIn<P, Qs>) && ...);

/** The rows of the blocks that the Consider \a C names, resolved against a state's space. */
template <typename C>
struct ConsideredRowsOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Paths>
struct ConsideredRowsOf<Consider<Paths...>>
{
    /** \returns the rows of the blocks named, in \a space, the state's space */
    template <typename Space>
    [[nodiscard]] static ConsideredRows<sizeof...(Paths)>
    resolve([[maybe_unused]] Space const & space)
    {
        static_assert(
            ([]<typename First, typename... Rest>(TagPath<First, Rest...> *) {
                return countTag<First, Space>() > 0;
            }(static_cast<Paths *>(nullptr))
             && ...),
            "a path of consider() names no block of the state"
        );
        static_assert((liesInNoOther<Paths, Paths...> && ...), "a block of consider() lies in another it names");
        return {differenceBlockRows<Paths>(space)...};
    }
};

/// \endcond

/** \returns the rows of the blocks that the measurement \a R considers, resolved against \a space,
 * the state's space (decisions.md [consider-states]): none for a full update
 */
template <typename R, typename Space>
[[nodiscard]] auto
consideredRows(Space const & space)
{
    return ConsideredRowsOf<typename R::ConsiderType>::resolve(space);
}

template <
    bool Statistics,
    typename Prior,
    typename R,
    typename Output,
    typename Cross,
    typename Jacobian,
    typename Sink>
[[nodiscard]] decltype(auto)
conditionOnPrediction(
    Prior const & prior,
    R const & resolved,
    Output const & y,
    Cross const & cross,
    Jacobian const & readJacobian,
    Sink && sink
);

}  // namespace internal

}  // namespace Emrys
