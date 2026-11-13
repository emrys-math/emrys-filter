#pragma once

/** \file
 * Process models as functions with reads and writes (decisions.md [model-policies], [continuous-time-noise],
 * [prediction-inputs], [time-steps]): the ProcessFunction that processModel() builds from a function and its policies;
 * the built-in models RandomWalk, the Kinematic chains ConstantVelocity and ConstantAcceleration, OrnsteinUhlenbeck,
 * and Constant, each the same object; compose(), which puts process models side by side; the ProcessModel concept;
 * predict(), which moves a %Gaussian through one by any transform method with the discretized process noise added,
 * and predictJoint(), which also keeps the %Gaussian it started from and its cross-covariance with the prediction, a
 * PredictedJoint, which is what a smoother records [smoothing]; propagate(), the noise-free step; and
 * FixedStepModel, a model over one step with the work that depends only on the step done once. How a prediction
 * reads a model is internal::ResolvedProcess, built on ResolvedReads.h. Read Measurement.h next.
 */

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/LU>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/ResolvedReads.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** Counts how many of a process model's written blocks own each value coordinate of a space, as
 * the blocks' rows are pushed, in storage sized by the space: a bounded space's needs no heap, so
 * the partition check that predict() asserts allocates nothing.
 */
template <typename Space>
class RowCoverage
{
public:

    /** Starts with every coordinate of \a space unowned. */
    explicit RowCoverage(Space const & space)
        : m_counts{CountsType::Zero(valueDimension(space))}
    {}

    /** Counts a block that owns \a rows; rows outside the space fail the check. */
    void
    push_back(BlockRows const & rows)
    {
        if (rows.start < 0 || rows.size < 0 || rows.start + rows.size > m_counts.size())
        {
            m_inRange = false;
        }
        else
        {
            m_counts.segment(rows.start, rows.size) += 1;
        }
    }

    /** \returns whether every coordinate was owned exactly once */
    [[nodiscard]] bool
    partitioned() const
    {
        return m_inRange && (m_counts == 1).all();
    }

private:

    /** One count per coordinate, in the space's own storage bound. */
    using CountsType = Eigen::
        Array<int, valueDimensionAtCompileTime<Space>, 1, Eigen::ColMajor, maxValueDimensionAtCompileTime<Space>, 1>;

    CountsType m_counts;    ///< how many blocks own each coordinate
    bool m_inRange = true;  ///< whether every block's rows lay within the space
};

/** What the types of a process model and a space say about whether the model writes every
 * coordinate of the space exactly once: that it does, that it doesn't, or nothing, so that the
 * check is left to run time.
 */
enum class StaticPartition
{
    unknown,  ///< the space is resizable, or the model's types do not decide it
    holds,    ///< every coordinate is written exactly once
    fails     ///< some coordinate is written twice or not at all
};

/** \returns whether the paths of \a Lists, each a TagPaths of the blocks a model writes, own every
 * value coordinate of the fixed-size \a Space exactly once
 */
template <typename Space, typename... Lists>
consteval bool
pathsPartition()
{
    auto counts = std::array<int, std::size_t(Space::FixedValueDimension)>{};
    [[maybe_unused]] auto const count = [&]<typename... Paths>(TagPaths<Paths...> *) {  // unused for no Lists
        (
            [&] {
                constexpr ResolvedIndex rows = ResolvePath<Space, Paths>::value;
                for (int i = rows.offset; i < rows.offset + rows.size; ++i)
                {
                    ++counts[std::size_t(i)];
                }
            }(),
            ...
        );
    };
    (count(static_cast<Lists *>(nullptr)), ...);
    for (int const owners : counts)
    {
        if (owners != 1)
        {
            return false;
        }
    }
    return true;
}

/** \returns what the type \a Model, through its staticPartition() where it has one, says about
 * whether it writes every coordinate of \a Space exactly once
 */
template <typename Model, typename Space>
consteval StaticPartition
staticPartition()
{
    if constexpr (requires { Model::template staticPartition<Space>(); })
    {
        return Model::template staticPartition<Space>();
    }
    else
    {
        return StaticPartition::unknown;
    }
}

/** Checks that \a model writes every coordinate of \a space exactly once: at compile time where the
 * types decide it (decisions.md [continuous-time-noise]), in every build, and otherwise by \a model.partitions() in an
 * assertion with \a message.
 */
template <typename Model, typename Space>
void
assertPartitions(
    [[maybe_unused]] Model const & model,
    [[maybe_unused]] Space const & space,
    [[maybe_unused]] char const * message = "the process model must write every coordinate of the state space once"
)
{
    constexpr StaticPartition known = staticPartition<Model, Space>();
    static_assert(
        known != StaticPartition::fails,
        "the process model must write every coordinate of the state space exactly once: a block it writes twice, or "
        "one no model writes, is not a partition"
    );
    if constexpr (known == StaticPartition::unknown)
    {
        EMRYS_FILTER_ASSERT(model.partitions(space) && message);
    }
}

/** \returns whether the paths of \a Paths, a TagPaths, write every coordinate of \a Space once, as
 * the types say, for a model whose writes are \a Paths
 */
template <typename Space, typename Paths>
consteval StaticPartition
staticPartitionOfWrites()
{
    if constexpr (Space::ValueDimensionStep == 0)
    {
        return pathsPartition<Space, Paths>() ? StaticPartition::holds : StaticPartition::fails;
    }
    else
    {
        return StaticPartition::unknown;
    }
}

/** \returns whether the blocks of \a Paths, a TagPaths, cover every coordinate of \a space once */
template <typename Space, typename... Paths>
[[nodiscard]] bool
writesPartition(Space const & space, TagPaths<Paths...> *)
{
    auto coverage = RowCoverage<Space>{space};
    (coverage.push_back(blockRows<Paths>(space)), ...);
    return coverage.partitioned();
}

/** The paths of the TagPaths \a P as a tuple type. */
template <typename P>
struct PathsOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Paths>
struct PathsOf<TagPaths<Paths...>>
{
    using type = std::tuple<Paths...>;
};

/// \endcond

/** Whether the process model \a Model writes the whole state: it was given no writes(), and so
 * no reads() either (decisions.md [whole-state-models]).
 */
template <typename Model>
inline constexpr bool writesWholeState = !Model::template has<PolicyKind::writes>;

/** The paths a model \a Model writes, a TagPaths, from its writes() policy. */
template <typename Model>
using WritePathsOf = typename Model::template Policy<PolicyKind::writes>::PathList;

/** The output space of a model \a Model, from its writes() policy: the Composite of the blocks'
 * names, one per path written.
 */
template <typename Model>
using OutputSpaceOf = typename Model::template Policy<PolicyKind::writes>::OutputSpace;

/** The path written at position \a K by \a Model. */
template <typename Model, std::size_t K>
using WritePathAt = std::tuple_element_t<K, typename PathsOf<WritePathsOf<Model>>::type>;

/** The name of the block written at position \a K by \a Model: its block of the output space. */
template <typename Model, std::size_t K>
using WriteNameAt = std::tuple_element_t<K, CompositeBlocks<OutputSpaceOf<Model>>>;

/** How many blocks \a Model writes. */
template <typename Model>
inline constexpr std::size_t writeCount = std::tuple_size_v<typename PathsOf<WritePathsOf<Model>>::type>;

/** The output space of the process model \a Model in the state space \a Space: the Composite of the
 * blocks written, or the state's space for a whole-state model.
 */
template <typename Model, typename Space>
using OutputSpaceIn = typename decltype([] {
    if constexpr (writesWholeState<Model>)
    {
        return std::type_identity<Space>{};
    }
    else
    {
        return std::type_identity<OutputSpaceOf<Model>>{};
    }
}())::type;

/** The output space instance of the process model \a Model in the state space \a space: a block per
 * path written, each the state's block there under the output's name for it, or the state's space
 * itself for a whole-state model.
 */
template <typename Model, typename Space>
[[nodiscard]] OutputSpaceIn<Model, Space>
outputOf(Space const & space)
{
    if constexpr (writesWholeState<Model>)
    {
        return space;
    }
    else
    {
        return [&]<std::size_t... K>(std::index_sequence<K...>) {
            return OutputSpaceOf<Model>{nameOf<WriteNameAt<Model, K>>(pathBlock<WritePathAt<Model, K>>(space))...};
        }(std::make_index_sequence<writeCount<Model>>{});
    }
}

/** \returns the instance of the Composite \a Target from \a source, a Composite instance with a block
 * per block of \a Target in order and the same leaves: the same blocks under other names, as a
 * placed model's function gives the model it places
 */
template <typename Target, typename Source>
[[nodiscard]] Target
renamedInstance(Source const & source)
{
    if constexpr (std::is_same_v<Target, Source>)
    {
        return source;
    }
    else
    {
        return []<typename... Names, typename... Blocks>(
                   Source const & from,
                   std::tuple<Names...> *,
                   std::tuple<Blocks...> *
               ) {
            return Target{nameOf<Names>(pathBlock<Blocks>(from))...};
        }(source, static_cast<CompositeBlocks<Target> *>(nullptr), static_cast<CompositeBlocks<Source> *>(nullptr));
    }
}

/** Whether \a T is an OnPath, a value placed on one block by on<Path>(). */
template <typename T>
inline constexpr bool isOnPath = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename Path>
inline constexpr bool isOnPath<OnPath<X, Path>>{true};

/// \endcond

/** Adds \a value, a spectral density over a block's differences, times \a dt to the diagonal block
 * of \a q at \a rows: a scalar on every axis, or a matrix.
 */
template <typename Matrix, typename Value, typename Scalar>
void
addSpectralDensity(Matrix & q, BlockRows const & rows, Value const & value, Scalar dt)
{
    if constexpr (MatrixExpression<Value>)
    {
        EMRYS_FILTER_ASSERT(
            value.rows() == rows.size && value.cols() == rows.size
            && "a spectral density matrix is over the block's differences"
        );
        q.block(rows.start, rows.start, rows.size, rows.size) += dt * value.template cast<Scalar>();
    }
    else
    {
        q.block(rows.start, rows.start, rows.size, rows.size).diagonal().array() += dt * Scalar(value);
    }
}

/** Adds the part \a part of continuousNoise(), discretized over \a dt as \f$ Q_c \, \Delta t \f$, to
 * \a q, a matrix over the differences of the output space \a out: on the block its path names, or
 * on the whole space for an unplaced part.
 */
template <typename Out, typename Matrix, typename Part, typename Scalar>
void
addContinuousPart(Out const & out, Matrix & q, Part const & part, Scalar dt)
{
    if constexpr (isOnPath<Part>)
    {
        [&]<typename X, typename Path>(OnPath<X, Path> const *) {
            addSpectralDensity(q, differenceBlockRows<Path>(out), part.value, dt);
        }(static_cast<Part const *>(nullptr));
    }
    else if constexpr (isPlacedByPath<Part>)
    {
        static_assert(
            std::is_void_v<Part>,
            "continuousNoise() takes no onEach<Path>(q); place the model on each repetition with onEach<Path>(model), "
            "and "
            "give it continuousNoise(q) for one repetition"
        );
    }
    else
    {
        addSpectralDensity(q, BlockRows{0, differenceDimension(out)}, part, dt);
    }
}

/** Checks the policies \a Set of a process model: reads() and writes() both or neither; no
 * logLikelihood() or outputSpace(), which are a measurement model's; one kind of noise; and no
 * affineIn() without reads(), since a whole-state model's input has no names (decisions.md
 * [whole-state-models]).
 *
 * \returns true, when it compiles
 */
template <typename Set>
[[nodiscard]] consteval bool
checkProcessPolicies()
{
    static_assert(
        Set::template has<PolicyKind::reads> || !Set::template has<PolicyKind::writes>,
        "a process model that writes blocks with writes() says what it reads with reads(); give neither to read and "
        "write the whole state"
    );
    static_assert(
        Set::template has<PolicyKind::writes> || !Set::template has<PolicyKind::reads>,
        "a process model that reads blocks with reads() says which blocks of the state it writes with writes(); give "
        "neither to read and write the whole state"
    );
    static_assert(
        Set::template has<PolicyKind::reads> || !Set::template has<PolicyKind::affineIn>,
        "affineIn() names items of reads(), and a model without reads() reads the whole state under no names"
    );
    static_assert(
        !Set::template has<PolicyKind::logLikelihood>,
        "logLikelihood(g) is a measurement model's; a process model has no observation to weight points by"
    );
    static_assert(
        !Set::template has<PolicyKind::sample>,
        "sample(g) is a measurement model's; a process model's draw is its step and its noise"
    );
    static_assert(
        !Set::template has<PolicyKind::outputSpace>,
        "a process model's output is the blocks of the state it writes, whose spaces the state has, so it takes no "
        "outputSpace()"
    );
    static_assert(
        !(Set::template has<PolicyKind::continuousNoise> && Set::template has<PolicyKind::additiveNoise>),
        "a process model's noise is continuousNoise(), discretized over the step, or additiveNoise(), a fixed Q, not "
        "both"
    );
    return true;
}

/** The function of a linear whole-state process model, \f$ x_{k+1} = F x_k \f$ for the matrix
 * \a Matrix over the state (decisions.md [whole-state-models]): its value with \f$ F \f$ beside it as
 * its Jacobian, so that nothing is differentiated and every method predicts it exactly.
 */
template <typename Matrix>
class LinearStep
{
public:

    static constexpr bool IsAffine = true;  ///< affine in the whole state, so its Jacobian is the same everywhere

    /** The function of \a matrix. */
    explicit LinearStep(Matrix matrix)
        : m_matrix{std::move(matrix)}
    {}

    /** \returns \f$ F \f$ */
    [[nodiscard]] Matrix const &
    matrix() const
    {
        return m_matrix;
    }

    /** \returns \f$ F x \f$ for \a x, a vector in the state space \a in, with \f$ F \f$ as its
     * Jacobian; the step is not read, and the state's coordinates must add their values
     */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()([[maybe_unused]] In const & in, Out const &, X const & x, TimeScalar) const
    {
        static_assert(
            blocksAddValues<In>,
            "a matrix process model moves a state whose coordinates add their values; a state with a block such as a "
            "unit quaternion takes a function"
        );
        static_assert(
            Matrix::ColsAtCompileTime == Eigen::Dynamic || valueDimensionAtCompileTime<In> == Eigen::Dynamic
                || (Matrix::ColsAtCompileTime == valueDimensionAtCompileTime<In>
                    && Matrix::RowsAtCompileTime == valueDimensionAtCompileTime<In>),
            "a matrix process model's matrix is square, with a row and a column per coordinate of the state"
        );
        EMRYS_FILTER_ASSERT(
            m_matrix.rows() == x.size() && m_matrix.cols() == x.size()
            && "a matrix process model's matrix is square, with a row and a column per coordinate of the state"
        );
        using T = typename X::Scalar;
        using Value = Eigen::Matrix<T, Matrix::RowsAtCompileTime, 1, Eigen::ColMajor, Matrix::MaxRowsAtCompileTime, 1>;
        Value value = m_matrix.template cast<T>() * x;
        return WithJacobian<Value, Matrix const &>{std::move(value), m_matrix};
    }

private:

    Matrix m_matrix;  ///< \f$ F \f$
};

/** What a process model writes, from its writes() policy \a Writes: its paths and the Composite
 * of its blocks. A specialization rather than a branch, so that a whole-state model's NoPolicy is
 * never asked for either.
 */
template <bool Whole, typename Writes>
struct ProcessWrites
{
    /** the paths of the blocks written */
    using Paths = typename Writes::PathList;
    /** the Composite of the blocks written */
    using OutputSpace = typename Writes::OutputSpace;
};

/** A whole-state model writes the whole space. */
template <typename Writes>
struct ProcessWrites<true, Writes>
{
    /** the whole space */
    using Paths = TagPaths<WholeSpace>;
    /** the state's space, named by the filter */
    using OutputSpace = WholeSpace;
};

}  // namespace internal

/** \ingroup Models
 * \brief A process model: a function \f$ f \f$ of its input and the step, with the policies that say
 * what it reads, what it writes, and what it is affine in, and its noise (decisions.md
 * [model-policies], [continuous-time-noise]).
 *
 * processModel() builds one, and the built-in models are ones; a user never spells the type. The
 * function is called as <tt>f(in, out, x, dt)</tt>, <tt>f(in, x, dt)</tt>, or <tt>f(x, dt)</tt>,
 * whichever it takes, with \a in the instance of the input space that reads() defines, \a out the
 * instance of the output space that writes() defines, built from the state's blocks, \a x a vector
 * in \a in, which it reads with <tt>block<Name>(in, x)</tt>, and \a dt the step in the filter's
 * scalar (see \ref Time); it returns a vector in \a out, which it may fill block by block from
 * <tt>zeros<T>(out)</tt>, either generic in the scalar of \a x or with its Jacobian beside it
 * (withJacobian(), decisions.md [returned-jacobians]), rows in \a out's differences and columns in
 * \a in's. A function generic in the scalar may take it as its one template parameter,
 * <tt>[]<typename T>(auto const & in, auto const & out, auto const & x, double dt)</tt>, which the
 * call supplies (decisions.md [function-scalars]). The next state is the state with the written blocks replaced. A
 * model without reads() and writes() reads and writes the whole state of the filter it is given to, and both spaces are
 * the state's (decisions.md [whole-state-models]).
 *
 * \tparam F         the function
 * \tparam Policies  the policies, each at most once
 *
 * \sa processModel(), predict(), ProcessModel
 */
template <typename F, typename... Policies>
class ProcessFunction
{
    /** The policies, found by kind. */
    using Set = internal::PolicySet<Policies...>;

public:

    /** the function */
    using FunctionType = F;

    static constexpr bool IsModel = true;      ///< a model, for internal::isModel
    static constexpr bool HasFunction = true;  ///< a process model always has a function

    /** Whether the policy of kind \a K was given. */
    template <internal::PolicyKind K>
    static constexpr bool has = Set::template has<K>;

    /** The policy of kind \a K, or internal::NoPolicy. */
    template <internal::PolicyKind K>
    using Policy = typename Set::template Type<K>;

    // processModel() and the placements give both reads() and writes(), or neither.
    static_assert(has<internal::PolicyKind::reads> == has<internal::PolicyKind::writes>);

    /// whether the model reads and writes the whole state (decisions.md [whole-state-models])
    static constexpr bool Whole = !has<internal::PolicyKind::writes>;

    /** The paths of the blocks written, a TagPaths; the whole space for a whole-state model. */
    using WritePaths = typename internal::ProcessWrites<Whole, Policy<internal::PolicyKind::writes>>::Paths;

    /** The output space: the Composite of the blocks written, in order; for a whole-state model,
     * which has its state's space as its output, internal::WholeSpace.
     */
    using OutputSpace = typename internal::ProcessWrites<Whole, Policy<internal::PolicyKind::writes>>::OutputSpace;

    /** The model of \a function with \a policies. */
    explicit ProcessFunction(F function, Policies... policies)
        : m_function{std::move(function)}
        , m_policies{std::move(policies)...}
    {}

    /** \returns the function */
    [[nodiscard]] F const &
    function() const
    {
        return m_function;
    }

    /** \returns the policy of kind \a K, which the model must have */
    template <internal::PolicyKind K>
    [[nodiscard]] Policy<K> const &
    policy() const
    {
        return Set::template get<K>(m_policies);
    }

    /** \returns every policy, as given */
    [[nodiscard]] std::tuple<Policies...> const &
    policies() const
    {
        return m_policies;
    }

    /** \returns the process noise \f$ Q \f$ over the step \a dt, a matrix in \a Scalar over the
     * differences of the output space instance \a out: a built-in model's closed form, the spectral
     * densities of continuousNoise() times the step, a fixed \f$ Q \f$ from additiveNoise(), or zero
     */
    template <typename Scalar, typename Out>
    [[nodiscard]] typename Moments<Out, Scalar>::MatrixType
    noiseOver(Out const & out, [[maybe_unused]] Scalar dt) const
    {
        using Matrix = typename Moments<Out, Scalar>::MatrixType;
        Eigen::Index const n = differenceDimension(out);
        Matrix q = Matrix::Zero(n, n);
        if constexpr (requires { m_function.template noiseOver<Scalar>(out, dt); })
        {
            q = m_function.template noiseOver<Scalar>(out, dt);
        }
        else if constexpr (has<internal::PolicyKind::discreteNoise>)
        {
            q = policy<internal::PolicyKind::discreteNoise>().function(out, dt);
        }
        else if constexpr (has<internal::PolicyKind::continuousNoise>)
        {
            std::apply(
                [&](auto const &... parts) {
                    (internal::addContinuousPart(out, q, parts, dt), ...);
                },
                policy<internal::PolicyKind::continuousNoise>().parts()
            );
        }
        else if constexpr (has<internal::PolicyKind::additiveNoise>)
        {
            auto const placed = internal::placeNoise(out, policy<internal::PolicyKind::additiveNoise>());
            using Placed = std::remove_cvref_t<decltype(placed)>;
            static_assert(
                !internal::isGaussianMixture<typename Placed::Distribution>,
                "a process model's additive noise is one Gaussian; a mixture over the step is a mixture argument of "
                "predict() (roadmap 17z)"
            );
            auto const & distribution = placed.distribution();
            q = internal::symmetricFromLower<Matrix>(distribution.covarianceLower().template cast<Scalar>());
        }
        return q;
    }

    /** \returns what the types say about whether the model writes every coordinate of \a Space
     * exactly once (internal::StaticPartition)
     */
    template <typename Space>
    [[nodiscard]] static consteval internal::StaticPartition
    staticPartition()
    {
        if constexpr (Whole)
        {
            return internal::StaticPartition::holds;
        }
        else
        {
            return internal::staticPartitionOfWrites<Space, WritePaths>();
        }
    }

    /** \returns whether the model writes every coordinate of \a space exactly once, as a whole-state
     * model always does
     */
    template <typename Space>
    [[nodiscard]] bool
    partitions([[maybe_unused]] Space const & space) const
    {
        if constexpr (Whole)
        {
            return true;
        }
        else
        {
            return internal::writesPartition(space, static_cast<WritePaths *>(nullptr));
        }
    }

private:

    EMRYS_FILTER_NO_UNIQUE_ADDRESS F m_function;  ///< the function
    std::tuple<Policies...> m_policies;           ///< the policies
};

/** \ingroup Models
 * \returns the process model whose function is \a function, <tt>f(in, out, x, dt)</tt>,
 * <tt>f(in, x, dt)</tt>, or <tt>f(x, dt)</tt>, told apart by the count of its arguments, with
 * \a policies, each a call such as reads(), writes(), affineIn(), continuousNoise(), or
 * additiveNoise(), in any order and each at most once (decisions.md [model-policies]). A model reads
 * what reads() names and writes the blocks of the state that writes() names; a model with neither
 * reads and writes the whole state of the filter it is given to, its noise over the whole state, and
 * takes no inputs of the call, nor is it composed or placed (decisions.md [whole-state-models]). A
 * model with no noise policy adds no noise, as one driven by a measured input does.
 *
 * \code
 * // Constant velocity on a Linear<4> state of position and velocity, as a whole-state model.
 * auto const cv = processModel(
 *     [](auto const & x, double dt) {
 *       auto y = x.eval();
 *       y.template head<2>() += dt * x.template tail<2>();
 *       return y;
 *     },
 *     continuousNoise(Q));
 * \endcode
 *
 * \code
 * auto const kinematics = processModel(
 *     []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
 *       auto const v = block<Velocity>(in, x);
 *       auto const a = less<SpecificForce>(block<SpecificForce>(in, x), block<AccelBias>(in, x));
 *       auto y = zeros<T>(out);
 *       block<Position>(out, y) = block<Position>(in, x) + dt * v + (dt * dt / 2) * a;
 *       block<Velocity>(out, y) = v + dt * a;
 *       return y;
 *     },
 *     reads(path<Position>, path<Velocity>, path<AccelBias>, path<SpecificForce>),
 *     writes(path<Position>, path<Velocity>),
 *     continuousNoise(on<Velocity>(q)));
 * \endcode
 */
template <typename F, typename... Policies>
    requires(!MatrixExpression<F> && !internal::ModelPolicy<F>)
[[nodiscard]] auto
processModel(F && function, Policies &&... policies)
{
    using Set = internal::PolicySet<std::remove_cvref_t<Policies>...>;
    static_assert(internal::checkProcessPolicies<Set>());
    return ProcessFunction<std::decay_t<F>, std::remove_cvref_t<Policies>...>{
        std::forward<F>(function),
        std::forward<Policies>(policies)...
    };
}

/** \ingroup Models
 * \returns the linear whole-state process model \f$ x_{k+1} = F x_k \f$ of the matrix \a matrix over
 * the state's coordinates, with \a policies, its noise (decisions.md [whole-state-models]): its
 * function returns \f$ F x \f$ with \f$ F \f$ as its Jacobian, so every method predicts it exactly,
 * as the matrix measurement model updates exactly. The state's coordinates must add their values, as
 * a Linear state's do, and \f$ F \f$ is the same over every step; a model whose matrix depends on the
 * step is a function.
 *
 * \code
 * auto const cv = processModel(F, additiveNoise(Q));  // F and Q are 4 by 4
 * \endcode
 */
template <MatrixExpression Matrix, typename... Policies>
[[nodiscard]] auto
processModel(Matrix && matrix, Policies &&... policies)
{
    using Plain = internal::PlainResult<std::remove_cvref_t<Matrix>>;
    using Set = internal::PolicySet<std::remove_cvref_t<Policies>...>;
    static_assert(
        !Set::template has<internal::PolicyKind::reads> && !Set::template has<internal::PolicyKind::writes>,
        "a matrix process model is over the whole state, so it takes no reads() or writes(); give a function for a "
        "model of some blocks"
    );
    static_assert(internal::checkProcessPolicies<Set>());
    return ProcessFunction<internal::LinearStep<Plain>, std::remove_cvref_t<Policies>...>{
        internal::LinearStep<Plain>{Plain{std::forward<Matrix>(matrix)}},
        std::forward<Policies>(policies)...
    };
}

namespace internal
{

/** Declared, never defined: true for a pointer to a ProcessFunction or a type derived from one. */
template <typename F, typename... Policies>
std::true_type
processFunctionBase(ProcessFunction<F, Policies...> const *);

/** The fallback for anything else. */
std::false_type
processFunctionBase(void const *);

/** The reads() policy of one block, the path \a Path, read under its last tag. */
template <typename Path>
using ReadsOne = Reads<ReadItem<AsTagPath<Path>, PathTag<Path>>>;

/** The reads() policy of the blocks \a Paths, each read under its last tag. */
template <typename... Paths>
using ReadsOf = Reads<ReadItem<AsTagPath<Paths>, PathTag<Paths>>...>;

/** The writes() policy of the blocks \a Paths. */
template <typename... Paths>
using WritesOf = Writes<AsTagPath<Paths>...>;

/** The function of a model that leaves its blocks as they are, \f$ y = x \f$, with the identity as
 * its Jacobian: RandomWalk's and Constant's.
 */
struct IdentityStep
{
    /** \returns \a x in \a out, with the identity Jacobian over its differences */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()(In const &, Out const & out, X const & x, TimeScalar) const
    {
        using T = ScalarOf<X>;
        using JacobianMatrix = typename Moments<Out, TimeScalar>::MatrixType;
        Eigen::Index const n = differenceDimension(out);
        return withJacobian(typename Moments<Out, T>::VectorType{x}, JacobianMatrix{JacobianMatrix::Identity(n, n)});
    }
};

/** The noise of a RandomWalk over a step: \f$ Q_c \, \Delta t \f$ over the block's differences. */
template <int Dimension, typename Scalar>
struct RandomWalkNoise
{
    Eigen::Matrix<Scalar, Dimension, Dimension> spectralDensity;  ///< \f$ Q_c \f$

    /** \returns \f$ Q_c \, \Delta t \f$ */
    template <typename Out, typename TimeScalar>
    [[nodiscard]] typename Moments<Out, TimeScalar>::MatrixType
    operator()(Out const &, TimeScalar dt) const
    {
        return typename Moments<Out, TimeScalar>::MatrixType{dt * spectralDensity.template cast<TimeScalar>()};
    }
};

/** \returns \f$ n! \f$, with \f$ 0! = 1 \f$, an integer, so that it converts exactly to any scalar */
[[nodiscard]] constexpr int
factorial(int n)
{
    return n <= 1 ? 1 : n * factorial(n - 1);
}

/** \returns \f$ \Delta t^k \f$ for each \f$ k < N \f$, by multiplication, which is cheaper than
 * \c std::pow for the small powers of a chain
 */
template <int N, typename TimeScalar>
[[nodiscard]] std::array<TimeScalar, N>
powersOf(TimeScalar dt)
{
    auto result = std::array<TimeScalar, N>{};
    result[0] = TimeScalar(1);
    for (int k = 1; k < N; ++k)
    {
        result[k] = result[k - 1] * dt;
    }
    return result;
}

/** The function of a Kinematic chain over the tags \a Tags, each block of dimension \a Dimension:
 * block \f$ i \f$ gains \f$ \Delta t^{j-i} / (j-i)! \f$ times each later block \f$ j \f$, with
 * the same coefficients on the identity as its Jacobian.
 */
template <int Dimension, typename... Tags>
struct KinematicStep
{
    static constexpr int Order = int(sizeof...(Tags));  ///< the number of blocks in the chain

    /** \returns the chain's step of \a x in \a out, with its Jacobian over the differences */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt) const
    {
        using T = ScalarOf<X>;
        using Vector = typename Moments<Out, T>::VectorType;
        using JacobianMatrix = typename Moments<Out, TimeScalar>::MatrixType;
        auto const starts = std::array<Eigen::Index, Order>{blockRows<Tags>(in).start...};
        auto const differenceStarts = std::array<Eigen::Index, Order>{differenceBlockRows<Tags>(out).start...};
        auto const dtPowers = powersOf<Order>(dt);
        Eigen::Index const n = differenceDimension(out);
        Vector y{x};
        JacobianMatrix jacobian = JacobianMatrix::Identity(n, n);
        for (int i = 0; i < Order; ++i)
        {
            for (int j = i + 1; j < Order; ++j)
            {
                TimeScalar const scale = dtPowers[j - i] / TimeScalar(factorial(j - i));
                y.template segment<Dimension>(starts[i]) += T(scale) * x.template segment<Dimension>(starts[j]);
                jacobian.template block<Dimension, Dimension>(differenceStarts[i], differenceStarts[j])
                    .diagonal()
                    .setConstant(scale);
            }
        }
        return withJacobian(std::move(y), std::move(jacobian));
    }
};

/** The noise of a Kinematic chain over a step: the exact discretization of white noise with
 * spectral density \f$ Q_c \f$ on the last derivative,
 * \f$ Q_{ij} = \Delta t^k / (k \, (N-1-i)! \, (N-1-j)!) \, Q_c \f$ with \f$ k = 2N - 1 - i - j \f$.
 */
template <int Dimension, typename Scalar, typename... Tags>
struct KinematicNoise
{
    static constexpr int Order = int(sizeof...(Tags));  ///< the number of blocks in the chain

    Eigen::Matrix<Scalar, Dimension, Dimension> spectralDensity;  ///< \f$ Q_c \f$

    /** \returns the denominator of the coefficient of \f$ \Delta t^k Q_c \f$ in noise block
     * \f$ Q_{ij} \f$, \f$ k \, (N-1-i)! \, (N-1-j)! \f$ with \f$ k = 2N - 1 - i - j \f$
     */
    [[nodiscard]] static constexpr int
    denominator(int i, int j)
    {
        return (2 * Order - 1 - i - j) * factorial(Order - 1 - i) * factorial(Order - 1 - j);
    }

    /** \returns \f$ Q \f$ over \a dt, over the differences of \a out */
    template <typename Out, typename TimeScalar>
    [[nodiscard]] typename Moments<Out, TimeScalar>::MatrixType
    operator()(Out const & out, TimeScalar dt) const
    {
        using Matrix = typename Moments<Out, TimeScalar>::MatrixType;
        auto const starts = std::array<Eigen::Index, Order>{differenceBlockRows<Tags>(out).start...};
        auto const dtPowers = powersOf<2 * Order>(dt);
        Eigen::Index const n = differenceDimension(out);
        Matrix q = Matrix::Zero(n, n);
        for (int i = 0; i < Order; ++i)
        {
            for (int j = 0; j < Order; ++j)
            {
                TimeScalar const scale = dtPowers[2 * Order - 1 - i - j] / TimeScalar(denominator(i, j));
                q.template block<Dimension, Dimension>(starts[i], starts[j]) =
                    scale * spectralDensity.template cast<TimeScalar>();
            }
        }
        return q;
    }
};

/** The function of an OrnsteinUhlenbeck model: \f$ y = \phi x \f$ with \f$ \phi = e^{-\Delta t / \tau} \f$,
 * and \f$ \phi \f$ on the diagonal as its Jacobian.
 */
template <typename Scalar>
struct OrnsteinUhlenbeckStep
{
    Scalar timeConstant;  ///< \f$ \tau \f$

    /** \returns \f$ \phi = e^{-\Delta t / \tau} \f$, in the scalar of \a dt */
    template <typename TimeScalar>
    [[nodiscard]] TimeScalar
    decay(TimeScalar dt) const
    {
        return Eigen::numext::exp(TimeScalar(-dt / TimeScalar(timeConstant)));
    }

    /** \returns the block decayed, with \f$ \phi I \f$ as its Jacobian */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()(In const &, Out const & out, X const & x, TimeScalar dt) const
    {
        using T = ScalarOf<X>;
        using JacobianMatrix = typename Moments<Out, TimeScalar>::MatrixType;
        TimeScalar const phi = decay(dt);
        Eigen::Index const n = differenceDimension(out);
        return withJacobian(
            typename Moments<Out, T>::VectorType{T(phi) * x},
            JacobianMatrix{phi * JacobianMatrix::Identity(n, n)}
        );
    }
};

/** The noise of an OrnsteinUhlenbeck model over a step: \f$ \sigma^2 (1 - \phi^2) \f$ on each
 * coordinate.
 */
template <typename Scalar>
struct OrnsteinUhlenbeckNoise
{
    Scalar timeConstant;  ///< \f$ \tau \f$
    Scalar variance;      ///< \f$ \sigma^2 \f$

    /** \returns \f$ \sigma^2 (1 - \phi^2) I \f$ over the differences of \a out */
    template <typename Out, typename TimeScalar>
    [[nodiscard]] typename Moments<Out, TimeScalar>::MatrixType
    operator()(Out const & out, TimeScalar dt) const
    {
        using Matrix = typename Moments<Out, TimeScalar>::MatrixType;
        TimeScalar const phi = Eigen::numext::exp(TimeScalar(-dt / TimeScalar(timeConstant)));
        Eigen::Index const n = differenceDimension(out);
        return Matrix{TimeScalar(variance) * (1 - phi * phi) * Matrix::Identity(n, n)};
    }
};

}  // namespace internal

/** \ingroup Models
 * \brief A model driven by white noise alone: \f$ x' = x \f$, with noise \f$ Q_c \, \Delta t \f$
 * on its differences.
 *
 * \a Path is a tag or a TagPath to a fixed-size block, such as a sensor bias. The spectral
 * density \f$ Q_c \f$ is a \f$ d \times d \f$ matrix over the block's differences, so it applies
 * as well to a block whose differences have another size than its values: a random walk on a
 * UnitQuaternion attitude holds the attitude and adds the noise to its rotation vector
 * (decisions.md [continuous-time-noise]). It is the exact discretization of a Wiener process. The model is affine in
 * its block and returns its Jacobian, the identity, so Linearized predicts it exactly without differentiating.
 *
 * \a Scalar_ is the type of \f$ Q_c \f$. The step computes in the scalar of the step it is given,
 * the filter's, converting \f$ Q_c \f$ to it where the two differ.
 */
template <typename Path, RealScalar Scalar_ = double>
class RandomWalk
    : public ProcessFunction<
          internal::IdentityStep,
          internal::ReadsOne<Path>,
          internal::WritesOf<Path>,
          AffineIn<internal::PathTag<Path>>,
          internal::DiscreteNoise<internal::RandomWalkNoise<
              internal::DifferenceSpaceOf<internal::PathTag<Path>>::type::FixedValueDimension,
              Scalar_>>>
{
    /** The space of the block's differences, which sets the dimension. */
    using Differences = typename internal::DifferenceSpaceOf<internal::PathTag<Path>>::type;

    static_assert(Differences::ValueDimensionStep == 0, "the block of a random walk must have a fixed size");

    /** The noise function. */
    using Noise = internal::RandomWalkNoise<Differences::FixedValueDimension, Scalar_>;

    /** The model this is. */
    using Base = ProcessFunction<
        internal::IdentityStep,
        internal::ReadsOne<Path>,
        internal::WritesOf<Path>,
        AffineIn<internal::PathTag<Path>>,
        internal::DiscreteNoise<Noise>>;

public:

    /** Dimension of the block's differences. */
    static constexpr int Dimension = Differences::FixedValueDimension;

    /** the scalar type of the spectral density */
    using Scalar = Scalar_;

    /** The spectral density's type: \f$ d \times d \f$ over the block's differences. */
    using SpectralDensityMatrix = Eigen::Matrix<Scalar_, Dimension, Dimension>;

    /** Spectral density \a spectralDensity on every axis, independently. */
    explicit RandomWalk(Scalar_ spectralDensity)
        : RandomWalk{SpectralDensityMatrix{spectralDensity * SpectralDensityMatrix::Identity()}}
    {}

    /** Spectral density matrix \a spectralDensity. */
    explicit RandomWalk(SpectralDensityMatrix const & spectralDensity)
        : Base{
              internal::IdentityStep{},
              internal::ReadsOne<Path>{},
              internal::WritesOf<Path>{},
              AffineIn<internal::PathTag<Path>>{},
              internal::DiscreteNoise<Noise>{Noise{spectralDensity}}
          }
    {}

    /** \returns the spectral density \f$ Q_c \f$ of the white noise on the block */
    [[nodiscard]] SpectralDensityMatrix const &
    spectralDensity() const
    {
        return this->template policy<internal::PolicyKind::discreteNoise>().function.spectralDensity;
    }
};

/** \ingroup Models
 * \brief A chain of two or more derivatives whose last one is driven by white noise: constant
 * velocity and constant acceleration.
 *
 * \a Paths name the chain in order, such as position and then velocity. Each is a tag or a
 * TagPath to a fixed-size block, and all have the same dimension \f$ d \f$. A chain adds one
 * block's value to another's, so each block's differences must have the size of its values. A
 * single block is a RandomWalk, which adds nothing and takes any block. Over a step
 * \f$ \Delta t \f$, block \f$ i \f$ of an \f$ N \f$-block chain moves by
 * \f$ x_i' = \sum_{j \ge i} \frac{\Delta t^{j-i}}{(j-i)!} x_j \f$, and the noise is
 * \f[ Q_{ij} = \frac{\Delta t^{k}}{k \, (N-1-i)! \, (N-1-j)!} \, Q_c, \qquad k = 2N - 1 - i - j, \f]
 * the exact discretization of white noise with spectral density \f$ Q_c \f$ (a \f$ d \times d \f$
 * matrix) on the last derivative. The model is affine in its blocks and returns its Jacobian, so
 * Linearized predicts it exactly without differentiating.
 *
 * These are the continuous white noise acceleration and jerk models of
 * \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", for any number of derivatives:
 * the closed form of the exact discretization (as given in
 * [Discretization](https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models)),
 * \f$ \int_0^{\Delta t} e^{A s} G Q_c G^T e^{A^T s}\,ds \f$ for the chain's
 * nilpotent \f$ A \f$.
 *
 * \a Scalar_ is the type of \f$ Q_c \f$, and comes before the paths, as a type ending in a pack
 * takes it; the aliases take it last, \c double by default. The step computes in the scalar of the
 * step it is given, the filter's, converting \f$ Q_c \f$ to it where the two differ.
 *
 * \sa RandomWalk, ConstantVelocity, ConstantAcceleration
 */
template <RealScalar Scalar_, typename... Paths>
class Kinematic
    : public ProcessFunction<
          internal::KinematicStep<
              internal::PathTag<std::tuple_element_t<0, std::tuple<Paths...>>>::FixedValueDimension,
              internal::PathTag<Paths>...>,
          internal::ReadsOf<Paths...>,
          internal::WritesOf<Paths...>,
          AffineIn<internal::PathTag<Paths>...>,
          internal::DiscreteNoise<internal::KinematicNoise<
              internal::PathTag<std::tuple_element_t<0, std::tuple<Paths...>>>::FixedValueDimension,
              Scalar_,
              internal::PathTag<Paths>...>>>
{
    static_assert(sizeof...(Paths) > 1, "a kinematic chain needs at least two blocks; one block is a RandomWalk");

    /** The first block's space, which sets the dimension. */
    using First = internal::PathTag<std::tuple_element_t<0, std::tuple<Paths...>>>;

public:

    /** Number of blocks in the chain. */
    static constexpr int Order = int(sizeof...(Paths));

    /** Dimension of each block. */
    static constexpr int Dimension = First::FixedValueDimension;

    static_assert(
        ((internal::PathTag<Paths>::ValueDimensionStep == 0
          && internal::PathTag<Paths>::FixedValueDimension == Dimension)
         && ...),
        "the blocks of a kinematic chain must have the same fixed size"
    );

    static_assert(
        (internal::hasEqualSizes<internal::PathTag<Paths>> && ...),
        "the blocks of a kinematic chain must have differences of the same size as their values"
    );

    /** the scalar type of the spectral density */
    using Scalar = Scalar_;

    /** The spectral density's type: \f$ d \times d \f$ over the last derivative's coordinates. */
    using SpectralDensityMatrix = Eigen::Matrix<Scalar_, Dimension, Dimension>;

private:

    /** The step function. */
    using Step = internal::KinematicStep<Dimension, internal::PathTag<Paths>...>;

    /** The noise function. */
    using Noise = internal::KinematicNoise<Dimension, Scalar_, internal::PathTag<Paths>...>;

    /** The model this is. */
    using Base = ProcessFunction<
        Step,
        internal::ReadsOf<Paths...>,
        internal::WritesOf<Paths...>,
        AffineIn<internal::PathTag<Paths>...>,
        internal::DiscreteNoise<Noise>>;

public:

    /** Spectral density \a spectralDensity on every axis, independently. */
    explicit Kinematic(Scalar_ spectralDensity)
        : Kinematic{SpectralDensityMatrix{spectralDensity * SpectralDensityMatrix::Identity()}}
    {}

    /** Spectral density matrix \a spectralDensity. */
    explicit Kinematic(SpectralDensityMatrix const & spectralDensity)
        : Base{
              Step{},
              internal::ReadsOf<Paths...>{},
              internal::WritesOf<Paths...>{},
              AffineIn<internal::PathTag<Paths>...>{},
              internal::DiscreteNoise<Noise>{Noise{spectralDensity}}
          }
    {}

    /** \returns the spectral density \f$ Q_c \f$ of the white noise on the last derivative */
    [[nodiscard]] SpectralDensityMatrix const &
    spectralDensity() const
    {
        return this->template policy<internal::PolicyKind::discreteNoise>().function.spectralDensity;
    }
};

/** \ingroup Models
 * Constant velocity (white-noise acceleration): position moves by velocity times \f$ \Delta t \f$.
 */
template <typename Position, typename Velocity, RealScalar Scalar = double>
using ConstantVelocity = Kinematic<Scalar, Position, Velocity>;

/** \ingroup Models
 * Constant acceleration (white-noise jerk).
 */
template <typename Position, typename Velocity, typename Acceleration, RealScalar Scalar = double>
using ConstantAcceleration = Kinematic<Scalar, Position, Velocity, Acceleration>;

/** \ingroup Models
 * \brief The time constant \f$ \tau \f$ of an OrnsteinUhlenbeck model, named so that it cannot
 * be given in the place of its variance.
 */
template <RealScalar Scalar = double>
struct TimeConstant
{
    Scalar value;  ///< \f$ \tau \f$, positive, in the unit of the time steps
};

/** \ingroup Models
 * \brief The stationary variance \f$ \sigma^2 \f$ of an OrnsteinUhlenbeck model, named so that
 * it cannot be given in the place of its time constant.
 */
template <RealScalar Scalar = double>
struct StationaryVariance
{
    Scalar value;  ///< \f$ \sigma^2 \f$, nonnegative
};

/// \cond DEDUCTION_GUIDES
template <typename Value>
    requires std::is_arithmetic_v<Value>
TimeConstant(Value) -> TimeConstant<internal::DeducedScalar<Value>>;

template <typename Value>
    requires std::is_arithmetic_v<Value>
StationaryVariance(Value) -> StationaryVariance<internal::DeducedScalar<Value>>;

/// \endcond

/** \ingroup Models
 * \brief A zero-mean Ornstein-Uhlenbeck (first-order Gauss-Markov) process on each coordinate
 * of the block named by \a Path, such as a slowly varying sensor bias.
 *
 * With time constant \f$ \tau \f$ and stationary variance \f$ \sigma^2 \f$, a step
 * \f$ \Delta t \f$ gives \f$ x' = \phi x \f$ with \f$ \phi = e^{-\Delta t / \tau} \f$ and noise
 * \f$ \sigma^2 (1 - \phi^2) \f$ on each coordinate, the exact discretization of the
 * [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process),
 * the first-order Gauss-Markov process of \ref Gelb1974 "Gelb (1974)", that models slowly
 * varying sensor errors. The model is affine in its block and returns its Jacobian.
 *
 * \a Scalar_ is the type of \f$ \tau \f$ and \f$ \sigma^2 \f$. The step computes in the scalar of
 * the step it is given, as Kinematic's does. The two are given by name,
 * <tt>OrnsteinUhlenbeck<Bias>{TimeConstant{60.0}, StationaryVariance{0.25}}</tt>, since as two
 * numbers of one type they would swap silently. Each coordinate of the block takes the same
 * \f$ \tau \f$ and \f$ \sigma^2 \f$; coordinates that differ are blocks of their own (decisions.md
 * [continuous-time-noise]).
 */
template <typename Path, RealScalar Scalar_ = double>
class OrnsteinUhlenbeck
    : public ProcessFunction<
          internal::OrnsteinUhlenbeckStep<Scalar_>,
          internal::ReadsOne<Path>,
          internal::WritesOf<Path>,
          AffineIn<internal::PathTag<Path>>,
          internal::DiscreteNoise<internal::OrnsteinUhlenbeckNoise<Scalar_>>>
{
    /** The model this is. */
    using Base = ProcessFunction<
        internal::OrnsteinUhlenbeckStep<Scalar_>,
        internal::ReadsOne<Path>,
        internal::WritesOf<Path>,
        AffineIn<internal::PathTag<Path>>,
        internal::DiscreteNoise<internal::OrnsteinUhlenbeckNoise<Scalar_>>>;

public:

    /** the scalar type of the parameters */
    using Scalar = Scalar_;

    /** Time constant \a timeConstant and stationary variance \a variance. */
    OrnsteinUhlenbeck(TimeConstant<Scalar_> timeConstant, StationaryVariance<Scalar_> variance)
        : Base{
              internal::OrnsteinUhlenbeckStep<Scalar_>{timeConstant.value},
              internal::ReadsOne<Path>{},
              internal::WritesOf<Path>{},
              AffineIn<internal::PathTag<Path>>{},
              internal::DiscreteNoise<internal::OrnsteinUhlenbeckNoise<Scalar_>>{
                  internal::OrnsteinUhlenbeckNoise<Scalar_>{timeConstant.value, variance.value}
              }
          }
    {
        EMRYS_FILTER_ASSERT(timeConstant.value > 0 && "the time constant must be positive");
        EMRYS_FILTER_ASSERT(variance.value >= 0 && "the variance must be nonnegative");
    }

    /** \returns the time constant \f$ \tau \f$ */
    [[nodiscard]] Scalar_
    timeConstant() const
    {
        return this->function().timeConstant;
    }

    /** \returns the stationary variance \f$ \sigma^2 \f$ */
    [[nodiscard]] Scalar_
    variance() const
    {
        return this->template policy<internal::PolicyKind::discreteNoise>().function.variance;
    }
};

/** \ingroup Models
 * \brief The block named by \a Path does not change, and has no process noise, such as a
 * fixed calibration parameter. The model is affine in its block and returns its Jacobian, the
 * identity.
 */
template <typename Path>
class Constant
    : public ProcessFunction<
          internal::IdentityStep,
          internal::ReadsOne<Path>,
          internal::WritesOf<Path>,
          AffineIn<internal::PathTag<Path>>>
{
    /** The model this is. */
    using Base = ProcessFunction<
        internal::IdentityStep,
        internal::ReadsOne<Path>,
        internal::WritesOf<Path>,
        AffineIn<internal::PathTag<Path>>>;

public:

    /** The model. */
    Constant()
        : Base{
              internal::IdentityStep{},
              internal::ReadsOne<Path>{},
              internal::WritesOf<Path>{},
              AffineIn<internal::PathTag<Path>>{}
          }
    {}
};

namespace internal
{

/** The items of the reads() policy \a R as a tuple type. */
template <typename R>
struct ItemsOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Items>
struct ItemsOf<Reads<Items...>>
{
    using type = std::tuple<Items...>;
};

/// \endcond

/** The union of read items: \a Acc, a Reads of the items kept so far, with each of \a Items appended
 * unless one of its name is already there, which must then read the same path.
 */
template <typename Acc, typename... Items>
struct ReadUnion
{
    using type = Acc;  ///< the union
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Kept, typename First, typename... Rest>
struct ReadUnion<Reads<Kept...>, First, Rest...>
{
    static constexpr bool present = (std::is_same_v<typename Kept::NameType, typename First::NameType> || ...);

    static_assert(
        ((!std::is_same_v<typename Kept::NameType, typename First::NameType>
          || std::is_same_v<typename Kept::PathType, typename First::PathType>)
         && ...),
        "two parts of a composition read different blocks under one name; read one under another name with "
        "on<Name>(path) in its reads()"
    );

    using type = typename ReadUnion<std::conditional_t<present, Reads<Kept...>, Reads<Kept..., First>>, Rest...>::type;
};

/// \endcond

/** The union of the items in the tuple type \a Tuple, in order of first appearance. */
template <typename Tuple>
struct ReadUnionOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Items>
struct ReadUnionOf<std::tuple<Items...>>
{
    using type = typename ReadUnion<Reads<>, Items...>::type;
};

/// \endcond

/** The writes of a composition: the paths in the tuple type \a Paths under the names in the tuple
 * type \a Names, the parts' own, as a WritesAs.
 */
template <typename Paths, typename Names>
struct WritesOfTuple;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Paths, typename... Names>
struct WritesOfTuple<std::tuple<Paths...>, std::tuple<Names...>>
{
    static_assert(distinctTypes<Paths...>(), "two parts of a composition write the same block");
    static_assert(
        distinctTypes<Names...>(),
        "two parts of a composition write blocks under one name, their paths' last tag; place each model on its "
        "block with on<Path>(model)"
    );
    using type = WritesAs<TagPaths<Paths...>, Composite<Names...>>;
};

/// \endcond

/** Whether one of the items in the tuple type \a Items reads under the name \a Name. */
template <typename Items, typename Name>
inline constexpr bool itemsName = false;

/// \cond INTERNAL_SPECIALIZATIONS
// A partial specialization rather than a lambda invoked in the initializer, which MSVC 19.50
// fails on with an internal compiler error.
template <typename... Items, typename Name>
inline constexpr bool itemsName<std::tuple<Items...>, Name> = (std::is_same_v<typename Items::NameType, Name> || ...);

/// \endcond

/** Whether the model \a Model reads the name \a Name. */
template <typename Model, typename Name>
inline constexpr bool modelReads =
    itemsName<typename ItemsOf<typename Model::template Policy<PolicyKind::reads>>::type, Name>;

/** Whether the model \a Model names \a Name in its affineIn(), or does not read it. */
template <typename Model, typename Name>
inline constexpr bool modelAffineIn =
    !modelReads<Model, Name> || affineNames<typename Model::template Policy<PolicyKind::affineIn>, Name>;

/** The names among \a Names that every one of \a Models reading them declares affine, as an
 * AffineIn: the affine set of a composition (decisions.md [affine-coordinates]).
 */
template <typename Acc, typename Models, typename... Names>
struct AffineUnion
{
    using type = Acc;  ///< the affine names
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Kept, typename... Models, typename First, typename... Rest>
struct AffineUnion<AffineIn<Kept...>, std::tuple<Models...>, First, Rest...>
{
    static constexpr bool affine = (modelAffineIn<Models, First> && ...);

    using type = typename AffineUnion<
        std::conditional_t<affine, AffineIn<Kept..., First>, AffineIn<Kept...>>,
        std::tuple<Models...>,
        Rest...>::type;
};

/// \endcond

/** The affine set of \a Models over the union reads \a Union, a Reads. */
template <typename Union, typename Models>
struct AffineOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Items, typename Models>
struct AffineOf<Reads<Items...>, Models>
{
    using type = typename AffineUnion<AffineIn<>, Models, typename Items::NameType...>::type;
};

/// \endcond

/** Whether the model \a Model's function returns its Jacobian when called with a vector of plain
 * scalars \a Scalar and \a Extra after it, the step for a process model and nothing for a
 * measurement model.
 */
template <typename Model, typename Scalar, typename... Extra>
inline constexpr bool returnsJacobianWith = isWithJacobian<std::remove_cvref_t<decltype(callForm(
    std::declval<typename Model::FunctionType const &>(),
    std::declval<typename Model::template Policy<PolicyKind::reads>::InputSpace const &>(),
    std::declval<OutputSpaceOf<Model> const &>(),
    std::declval<
        typename Moments<typename Model::template Policy<PolicyKind::reads>::InputSpace, Scalar>::VectorType const &>(),
    std::declval<Extra>()...
))>>;

/** Whether the process model \a Model's function returns its Jacobian when called with plain
 * scalars \a Scalar.
 */
template <typename Model, typename Scalar>
inline constexpr bool returnsStepJacobian = returnsJacobianWith<Model, Scalar, Scalar>;

/** Whether the process model \a Model's function returns its Jacobian when called with plain
 * scalars \a Scalar in the state space \a Space, which a whole-state model reads and writes.
 */
template <typename Model, typename Space, typename Scalar>
inline constexpr bool returnsStepJacobianIn = [] {
    if constexpr (readsWholeState<Model>)
    {
        return isWithJacobian<std::remove_cvref_t<decltype(callForm(
            std::declval<typename Model::FunctionType const &>(),
            std::declval<Space const &>(),
            std::declval<Space const &>(),
            std::declval<typename Moments<Space, Scalar>::VectorType const &>(),
            std::declval<Scalar>()
        ))>>;
    }
    else
    {
        return returnsStepJacobian<Model, Scalar>;
    }
}();

/** The items of the model \a Model's reads(), as a tuple type. */
template <typename Model>
using ReadItemsOf = typename ItemsOf<typename Model::template Policy<PolicyKind::reads>>::type;

/** \returns the input space instance of \a Part, a part of a composition: its names' blocks of the
 * whole's input \a in
 */
template <typename Part, typename In>
[[nodiscard]] auto
partIn(In const & in)
{
    using PartIn = typename Part::template Policy<PolicyKind::reads>::InputSpace;
    return [&]<typename... Items>(std::tuple<Items...> *) {
        return PartIn{pathBlock<typename Items::NameType>(in)...};
    }(static_cast<ReadItemsOf<Part> *>(nullptr));
}

/** \returns the input vector of \a Part, over its input \a pin: its names' blocks of \a x, a
 * vector in the whole's input \a in
 */
template <typename Part, typename In, typename PartIn, typename X>
[[nodiscard]] auto
partVector(In const & in, PartIn const & pin, X const & x)
{
    auto px = zeros<typename X::Scalar>(pin);
    [&]<typename... Items>(std::tuple<Items...> *) {
        ((block<typename Items::NameType>(pin, px) = block<typename Items::NameType>(in, x)), ...);
    }(static_cast<ReadItemsOf<Part> *>(nullptr));
    return px;
}

/** \returns the output space instance of \a Part, a part of a composition: its names' blocks of
 * the whole's output \a out
 */
template <typename Part, typename Out>
[[nodiscard]] OutputSpaceOf<Part>
partOut(Out const & out)
{
    return [&]<typename... Names>(std::tuple<Names...> *) {
        return OutputSpaceOf<Part>{pathBlock<Names>(out)...};
    }(static_cast<CompositeBlocks<OutputSpaceOf<Part>> *>(nullptr));
}

/** Writes the row block \a Row of a part's Jacobian \a partJacobian, over its output \a pout and
 * input \a pin, into the whole's \a jacobian over \a out and \a in, one column block per item the
 * part reads.
 */
template <
    typename Row,
    typename Out,
    typename In,
    typename J,
    typename POut,
    typename PIn,
    typename PJ,
    typename... Items>
void
partJacobianRow(
    Out const & out,
    In const & in,
    J & jacobian,
    POut const & pout,
    PIn const & pin,
    PJ const & partJacobian,
    std::tuple<Items...> *
)
{
    ((diffBlock<Row, typename Items::NameType>(out, in, jacobian) =
          diffBlock<Row, typename Items::NameType>(pout, pin, partJacobian)),
     ...);
}

/** Steps part \a I of \a parts: gathers its input from \a x by name, calls its function with
 * \a extra after the input, writes its result into \a y by block, and its Jacobian into
 * \a jacobian where the whole assembles one.
 */
template <
    std::size_t I,
    typename Parts,
    typename In,
    typename Out,
    typename X,
    typename Y,
    typename J,
    typename... Extra>
void
stepPart(
    Parts const & parts,
    In const & in,
    Out const & out,
    X const & x,
    Y & y,
    [[maybe_unused]] J & jacobian,
    Extra const &... extra
)
{
    using Part = std::tuple_element_t<I, Parts>;
    auto const pin = partIn<Part>(in);
    auto const pout = partOut<Part>(out);
    auto const px = partVector<Part>(in, pin, x);
    auto result = vectorReturned(callForm(std::get<I>(parts).function(), pin, pout, px, extra...));
    auto const & value = valueOf(result);
    using Names = CompositeBlocks<OutputSpaceOf<Part>>;
    [&]<typename... Rows>(std::tuple<Rows...> *) {
        ((block<Rows>(out, y) = block<Rows>(pout, value)), ...);
        if constexpr (!std::is_same_v<J, NoPolicy>)
        {
            (partJacobianRow<
                 Rows>(out, in, jacobian, pout, pin, result.jacobian, static_cast<ReadItemsOf<Part> *>(nullptr)),
             ...);
        }
    }(static_cast<Names *>(nullptr));
}

/** \returns the models \a parts side by side at \a x, a vector in the whole's input space \a in,
 * into the whole's output space \a out, each called with \a extra after its input: each part's
 * function on its own input, its result written by block, and with the Jacobian, in
 * \a JacobianScalar, assembled from the parts' when every part returns one
 */
template <typename JacobianScalar, typename... Models, typename In, typename Out, typename X, typename... Extra>
[[nodiscard]] auto
stepParts(std::tuple<Models...> const & parts, In const & in, Out const & out, X const & x, Extra const &... extra)
{
    constexpr bool AllJacobians = (returnsJacobianWith<Models, JacobianScalar, Extra...> && ...);
    auto y = zeros<typename X::Scalar>(out);
    auto jacobian = [&] {
        if constexpr (AllJacobians)
        {
            using JacobianMatrix = MatrixOf<
                JacobianScalar,
                valueDimensionAtCompileTime<DifferenceSpace<Out>>,
                valueDimensionAtCompileTime<DifferenceSpace<In>>,
                maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
                maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
            return JacobianMatrix{JacobianMatrix::Zero(differenceDimension(out), differenceDimension(in))};
        }
        else
        {
            return NoPolicy{};
        }
    }();
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (stepPart<I>(parts, in, out, x, y, jacobian, extra...), ...);
    }(std::index_sequence_for<Models...>{});
    if constexpr (AllJacobians)
    {
        return withJacobian(std::move(y), std::move(jacobian));
    }
    else
    {
        return y;
    }
}

/** Whether the function of any of the models \a Models takes batches (decisions.md
 * [batched-functions]), so that the models side by side do.
 */
template <typename... Models>
inline constexpr bool anyTakesBatches =
    (takesBatches<std::remove_cvref_t<decltype(std::declval<Models const &>().function())>> || ...);

/** Writes into \a y, the BatchedOutput of the whole, the models \a parts side by side at a batch
 * \a x, a column per point in the whole's input space \a in, into the whole's output space \a out,
 * each called with \a extra after its input (decisions.md [batched-functions]): each part's input
 * gathered column by column, a part whose function takes batches called once over its columns and
 * any other at each column, and its values written by block.
 */
template <typename... Models, typename In, typename Out, typename X, typename Y, typename... Extra>
void
stepPartsBatch(
    std::tuple<Models...> const & parts,
    In const & in,
    Out const & out,
    X const & x,
    Y & y,
    Extra const &... extra
)
{
    using T = typename X::Scalar;
    constexpr int Cols = X::ColsAtCompileTime;
    constexpr int MaxCols = X::MaxColsAtCompileTime;
    Eigen::Index const count = x.cols();
    auto & values = y.values();
    values.setZero();
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (
            [&] {
                using Part = std::tuple_element_t<I, std::tuple<Models...>>;
                auto const pin = partIn<Part>(in);
                auto const pout = partOut<Part>(out);
                using PIn = std::remove_cvref_t<decltype(pin)>;
                using POut = OutputSpaceOf<Part>;
                using PX =
                    MatrixOf<T, valueDimensionAtCompileTime<PIn>, Cols, maxValueDimensionAtCompileTime<PIn>, MaxCols>;
                using PY =
                    MatrixOf<T, valueDimensionAtCompileTime<POut>, Cols, maxValueDimensionAtCompileTime<POut>, MaxCols>;
                auto px = PX{};
                px.resize(valueDimension(pin), count);
                for (Eigen::Index j = 0; j < count; ++j)
                {
                    px.col(j) = partVector<Part>(in, pin, x.col(j));
                }
                auto py = PY{};
                py.resize(valueDimension(pout), count);
                auto const & function = std::get<I>(parts).function();
                if constexpr (takesBatches<std::remove_cvref_t<decltype(function)>>)
                {
                    callBatched(function, pin, pout, px, py, extra...);
                }
                else
                {
                    for (Eigen::Index j = 0; j < count; ++j)
                    {
                        typename Moments<PIn, T>::VectorType const column = px.col(j);
                        py.col(j) = valueOf(vectorReturned(callForm(function, pin, pout, column, extra...)));
                    }
                }
                [&]<typename... Rows>(std::tuple<Rows...> *) {
                    ((block<Rows>(out, values) = block<Rows>(pout, py)), ...);
                }(static_cast<CompositeBlocks<POut> *>(nullptr));
            }(),
            ...
        );
    }(std::index_sequence_for<Models...>{});
}

}  // namespace internal

template <typename... Models>
class Composition;

namespace internal
{

/** Whether \a Model is a Composition. */
template <typename Model>
inline constexpr bool isComposition = false;

/// \cond INTERNAL_SPECIALIZATIONS
/** A Composition is one. */
template <typename... Models>
inline constexpr bool isComposition<Composition<Models...>> = true;

/// \endcond

/** The scalars a process model's parameters state, as a \c std::tuple: none for a model without a
 * \c Scalar, such as Constant or one of processModel()'s.
 */
template <typename Model>
struct DeclaredScalars
{
    using type = std::tuple<>;  ///< no scalar
};

/// \cond INTERNAL_SPECIALIZATIONS
/** A model with a \c Scalar, other than a composition, states that one. */
template <typename Model>
    requires(!isComposition<Model> && requires { typename Model::Scalar; })
struct DeclaredScalars<Model>
{
    using type = std::tuple<typename Model::Scalar>;  ///< the model's scalar
};

/** A composition states its parts' scalars, not the \c double it takes when they state none. */
template <typename... Models>
struct DeclaredScalars<Composition<Models...>>
{
    using type = decltype(std::tuple_cat(std::declval<typename DeclaredScalars<Models>::type>()...));  ///< theirs
};

/// \endcond

/** The scalars the process models \a Models state, together. */
template <typename... Models>
using DeclaredScalarsOf = decltype(std::tuple_cat(std::declval<typename DeclaredScalars<Models>::type>()...));

/** \returns whether the process models \a Models state one scalar at most among them */
template <typename... Models>
[[nodiscard]] consteval bool
oneDeclaredScalar()
{
    return []<typename... Scalars>(std::tuple<Scalars...> *) {
        if constexpr (sizeof...(Scalars) == 0)
        {
            return true;
        }
        else
        {
            using First = std::tuple_element_t<0, std::tuple<Scalars...>>;
            return (std::is_same_v<First, Scalars> && ...);
        }
    }(static_cast<DeclaredScalarsOf<Models...> *>(nullptr));
}

/** The first of the scalars \a Scalars, a \c std::tuple, or \c double where it is empty. */
template <typename Scalars>
struct FirstScalarOr
{
    using type = double;  ///< the library's default scalar
};

/// \cond INTERNAL_SPECIALIZATIONS
/** A tuple with a first scalar gives it. */
template <typename First, typename... Rest>
struct FirstScalarOr<std::tuple<First, Rest...>>
{
    using type = First;  ///< the first scalar
};

/// \endcond

}  // namespace internal

/** \ingroup Models
 * \brief Process models side by side, as compose() puts them (decisions.md [model-policies],
 * [model-placement]): one model whose reads are the union of the parts', gathered once, whose writes
 * are the parts' together, which must write every block of the state once, whose noise is
 * block-diagonal, and whose affine set is the names every part reading them declares affine.
 *
 * Each part's function is called on its own input, gathered from the whole's by name, and its result
 * is written into the whole's output by block. When every part returns its Jacobian the composition
 * does too, assembled from theirs; otherwise the whole is differentiated. The parts that state a
 * scalar, as the built-in models do, state the same one, which is the composition's; a composition
 * of parts that state none, such as Constant, has \c double.
 *
 * \tparam Models  the parts, each a ProcessModel
 *
 * \sa compose(), processModel()
 */
template <typename... Models>
class Composition
{
    static_assert(sizeof...(Models) > 0, "a composition needs a process model");
    static_assert(
        internal::oneDeclaredScalar<Models...>(),
        "the parts of a composition must have one scalar: give every built-in model the same Scalar"
    );

    /** The parts' reads(), concatenated. */
    using AllItems = decltype(std::tuple_cat(
        std::declval<
            typename internal::ItemsOf<typename Models::template Policy<internal::PolicyKind::reads>>::type>()...
    ));

    /** The parts' writes(), concatenated. */
    using AllPaths =
        decltype(std::tuple_cat(std::declval<typename internal::PathsOf<typename Models::WritePaths>::type>()...));

    /** The names of the parts' written blocks, concatenated. */
    using AllNames =
        decltype(std::tuple_cat(std::declval<internal::CompositeBlocks<typename Models::OutputSpace>>()...));

public:

    /// the scalar type: the one its parts state, or \c double where none states one
    using Scalar = typename internal::FirstScalarOr<internal::DeclaredScalarsOf<Models...>>::type;

    using FunctionType = Composition;  ///< the function: the composition itself

    static constexpr bool IsModel = true;      ///< a model, for internal::isModel
    static constexpr bool HasFunction = true;  ///< a process model always has a function

    /** The reads() of the whole: the union of the parts', in order of first appearance. */
    using ReadsPolicy = typename internal::ReadUnionOf<AllItems>::type;

    /** The writes() of the whole: the parts', in order. */
    using WritesPolicy = typename internal::WritesOfTuple<AllPaths, AllNames>::type;

    /** The affineIn() of the whole. */
    using AffinePolicy = typename internal::AffineOf<ReadsPolicy, std::tuple<Models...>>::type;

    /** Whether the policy of kind \a K is one the whole has: reads(), writes(), and affineIn(). */
    template <internal::PolicyKind K>
    static constexpr bool has =
        K == internal::PolicyKind::reads || K == internal::PolicyKind::writes || K == internal::PolicyKind::affineIn;

    /** The policy of kind \a K, or internal::NoPolicy. */
    template <internal::PolicyKind K>
    using Policy = std::conditional_t<
        K == internal::PolicyKind::reads,
        ReadsPolicy,
        std::conditional_t<
            K == internal::PolicyKind::writes,
            WritesPolicy,
            std::conditional_t<K == internal::PolicyKind::affineIn, AffinePolicy, internal::NoPolicy>>>;

    /** The paths of the blocks written, a TagPaths. */
    using WritePaths = typename WritesPolicy::PathList;

    /** The output space: the Composite of the blocks written, in order. */
    using OutputSpace = typename WritesPolicy::OutputSpace;

    /** The composition of \a models. */
    explicit Composition(Models... models)
        : m_parts{std::move(models)...}
    {}

    /** \returns the function: the composition itself */
    [[nodiscard]] Composition const &
    function() const
    {
        return *this;
    }

    /** \returns the policy of kind \a K, which the whole must have */
    template <internal::PolicyKind K>
    [[nodiscard]] Policy<K> const &
    policy() const
    {
        static_assert(has<K>, "the composition was not given this policy");
        static constexpr Policy<K> value{};
        return value;
    }

    /** \returns the parts, in order */
    [[nodiscard]] std::tuple<Models...> const &
    parts() const
    {
        return m_parts;
    }

    /** \returns the whole's step of \a x, a vector in the input space \a in, into the output space
     * \a out over \a dt: each part's function on its own input, its result written by block, and the
     * Jacobian assembled from the parts' when every part returns one
     */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt) const
        requires MatrixExpression<X>
    {
        return internal::stepParts<TimeScalar>(m_parts, in, out, x, dt);
    }

    /// whether the whole takes a batch: a part's function does (decisions.md [batched-functions])
    static constexpr bool TakesBatches = internal::anyTakesBatches<Models...>;

    /** The whole writes no Jacobian in a batch: a method that needs one differentiates it. */
    using MarkType = internal::NoJacobians;

    /** Writes into \a y the whole's step of a batch \a x, a column per point in the input space
     * \a in, into the output space \a out over \a dt: each batched part called once over its
     * columns, and each other part at each column (internal::stepPartsBatch())
     */
    template <typename In, typename Out, typename X, typename TimeScalar, typename Y>
    void
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt, Y & y) const
        requires TakesBatches
    {
        internal::stepPartsBatch(m_parts, in, out, x, y, dt);
    }

    /** \returns the process noise \f$ Q \f$ over \a dt, over the differences of the output space
     * instance \a out: block-diagonal, each part's over its own blocks
     */
    template <typename Scalar, typename Out>
    [[nodiscard]] typename Moments<Out, Scalar>::MatrixType
    noiseOver(Out const & out, Scalar dt) const
    {
        using Matrix = typename Moments<Out, Scalar>::MatrixType;
        Eigen::Index const n = differenceDimension(out);
        Matrix q = Matrix::Zero(n, n);
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (noisePart<I, Scalar>(out, dt, q), ...);
        }(std::index_sequence_for<Models...>{});
        return q;
    }

    /** \returns what the types say about whether the whole writes every coordinate of \a Space
     * exactly once (internal::StaticPartition)
     */
    template <typename Space>
    [[nodiscard]] static consteval internal::StaticPartition
    staticPartition()
    {
        return internal::staticPartitionOfWrites<Space, WritePaths>();
    }

    /** \returns whether the whole writes every coordinate of \a space exactly once */
    template <typename Space>
    [[nodiscard]] bool
    partitions(Space const & space) const
    {
        return internal::writesPartition(space, static_cast<WritePaths *>(nullptr));
    }

private:

    /** Part \a I's type. */
    template <std::size_t I>
    using Part = std::tuple_element_t<I, std::tuple<Models...>>;

    /** Adds part \a I's noise over \a dt to \a q, on its blocks of \a out. */
    template <std::size_t I, typename Scalar, typename Out, typename Q>
    void
    noisePart(Out const & out, Scalar dt, Q & q) const
    {
        auto const pout = internal::partOut<Part<I>>(out);
        auto const partNoise = std::get<I>(m_parts).template noiseOver<Scalar>(pout, dt);
        using Names = internal::CompositeBlocks<typename Part<I>::OutputSpace>;
        [&]<typename... Rows>(std::tuple<Rows...> *) {
            (noiseRow<Rows>(out, q, pout, partNoise, static_cast<Names *>(nullptr)), ...);
        }(static_cast<Names *>(nullptr));
    }

    /** Writes the row block \a Row of a part's noise \a partNoise over its output \a pout into the
     * whole's \a q over \a out, one column block per block the part writes.
     */
    template <typename Row, typename Out, typename Q, typename POut, typename PQ, typename... Columns>
    static void
    noiseRow(Out const & out, Q & q, POut const & pout, PQ const & partNoise, std::tuple<Columns...> *)
    {
        ((diffBlock<Row, Columns>(out, out, q) = diffBlock<Row, Columns>(pout, pout, partNoise)), ...);
    }

    std::tuple<Models...> m_parts;  ///< the parts, in order
};

namespace internal
{

/** Declared, never defined: true for a pointer to a Composition. */
template <typename... Models>
std::true_type
compositionBase(Composition<Models...> const *);

/** The fallback for anything else. */
std::false_type
compositionBase(void const *);

}  // namespace internal

/** \ingroup Models
 * \brief A process model: what processModel() returns, a built-in model, or a composition of
 * process models (decisions.md [model-policies]). A filter owns one, and predict() moves a
 * %Gaussian through one.
 */
template <typename M>
concept ProcessModel = decltype(internal::processFunctionBase(std::declval<std::remove_cvref_t<M> const *>()))::value
                    || decltype(internal::compositionBase(std::declval<std::remove_cvref_t<M> const *>()))::value;

/** \ingroup Models
 * \brief The process model of a state with nothing to move: what a PointMassFilter, whose grid
 * covers the whole state, holds in place of one. It writes nothing, so it partitions only an empty
 * space.
 */
struct NoProcessModel
{
    /** \returns whether the empty writes partition \a Space: only an empty fixed-size space */
    template <typename Space>
    [[nodiscard]] static consteval internal::StaticPartition
    staticPartition()
    {
        if constexpr (Space::ValueDimensionStep == 0)
        {
            return Space::FixedValueDimension == 0 ? internal::StaticPartition::holds
                                                   : internal::StaticPartition::fails;
        }
        else
        {
            return internal::StaticPartition::unknown;
        }
    }

    /** \returns whether \a space is empty, the only space a model that writes nothing partitions */
    template <typename Space>
    [[nodiscard]] bool
    partitions(Space const & space) const
    {
        return valueDimension(space) == 0;
    }
};

/** \ingroup Models
 * \returns the process model of \a models side by side (see Composition): its reads are the union
 * of theirs, its writes theirs together, and its noise block-diagonal.
 *
 * \code
 * auto const vehicle = compose(kinematics, RandomWalk<AccelBias>{qb}, RandomWalk<Heading>{qh});
 * \endcode
 */
template <ProcessModel... Models>
[[nodiscard]] Composition<std::remove_cvref_t<Models>...>
compose(Models &&... models)
{
    (internal::refuseWholeState<std::remove_cvref_t<Models>>(), ...);
    return Composition<std::remove_cvref_t<Models>...>{std::forward<Models>(models)...};
}

namespace internal
{

/** \returns whether every item of \a Model's reads() names a block of the state space \a Space, so
 * that the model needs no input of the call
 */
template <typename Model, typename Space>
[[nodiscard]] consteval bool
readsOnlyState()
{
    if constexpr (readsWholeState<Model>)
    {
        return true;
    }
    else
    {
        return []<typename... Items>(std::tuple<Items...> *) {
            return (
                []<typename First, typename... Rest>(TagPath<First, Rest...> *) {
                    return countTag<First, Space>() > 0;
                }(static_cast<typename Items::PathType *>(nullptr))
                && ... && true
            );
        }(static_cast<typename ItemsOf<typename Model::template Policy<PolicyKind::reads>>::type *>(nullptr));
    }
}

/** processNoise() of a model that writes blocks: its noise over its output, placed on them. */
template <typename Scalar, ProcessModel Model, Space S>
[[nodiscard]] typename Moments<S, Scalar>::MatrixType
processNoiseOfBlocks(Model const & model, S const & space, Scalar dt)
{
    using Matrix = typename Moments<S, Scalar>::MatrixType;
    auto const out = outputOf<Model>(space);
    auto const partial = model.template noiseOver<Scalar>(out, dt);
    Eigen::Index const n = differenceDimension(space);
    Matrix q = Matrix::Zero(n, n);
    [&]<std::size_t... K>(std::index_sequence<K...>) {
        std::array<BlockRows, sizeof...(K)> const rows{differenceBlockRows<WritePathAt<Model, K>>(space)...};
        std::array<BlockRows, sizeof...(K)> const outRows{differenceBlockRows<WriteNameAt<Model, K>>(out)...};
        for (std::size_t k = 0; k < sizeof...(K); ++k)
        {
            for (std::size_t l = 0; l < sizeof...(K); ++l)
            {
                q.block(rows[k].start, rows[l].start, rows[k].size, rows[l].size) =
                    partial.block(outRows[k].start, outRows[l].start, outRows[k].size, outRows[l].size);
            }
        }
    }(std::make_index_sequence<writeCount<Model>>{});
    return q;
}

/** \returns the process noise \f$ Q \f$ of \a model over \a dt in \a space, a matrix in \a Scalar
 * over the space's differences: the model's noise over its output, placed on the blocks it writes,
 * or the whole of it for a whole-state model
 */
template <typename Scalar, ProcessModel Model, Space S>
[[nodiscard]] typename Moments<S, Scalar>::MatrixType
processNoise(Model const & model, S const & space, Scalar dt)
{
    using Matrix = typename Moments<S, Scalar>::MatrixType;
    if constexpr (writesWholeState<Model>)
    {
        return Matrix{model.template noiseOver<Scalar>(space, dt)};
    }
    else
    {
        return processNoiseOfBlocks<Scalar>(model, space, dt);
    }
}

/** A call of a process model \a Call (an Observed of the model and the call's arguments, with no
 * method or observation) resolved against the state space \a Space (decisions.md [model-policies]):
 * its reads() resolved against the state and the arguments (ResolvedReads), its output space
 * built from the state's blocks, and its function called on the whole state, with its input
 * gathered once per evaluation and its result written into the next state by block, a returned
 * Jacobian expanded to the whole state's differences and split by uncertain input. A prediction
 * resolves its model once, at its start, and reads this. Valid while the call and the space are.
 */
template <typename Call, typename Space>
class ResolvedProcess : public ResolvedReads<typename Call::ModelType, typename Call::ArgumentsType, Space>
{
    /** The reads() resolved, which this builds on. */
    using Base = ResolvedReads<typename Call::ModelType, typename Call::ArgumentsType, Space>;

public:

    using CallType = Call;                          ///< the call
    using Model = typename Call::ModelType;         ///< the model
    using Function = typename Model::FunctionType;  ///< its function

    using Base::InputCount;
    using Base::ItemCount;
    using Base::StateCount;
    using typename Base::In;
    using typename Base::Reads;

    /** Argument \a J of the call. */
    template <std::size_t J>
    using Argument = typename Base::template Argument<J>;

    /** A vector in the input space, in \a T. */
    template <typename T>
    using InVector = typename Base::template InVector<T>;

    /** The paths of the blocks written, a TagPaths. */
    using WritePaths = typename Model::WritePaths;

    static constexpr std::size_t WriteCount = std::tuple_size_v<typename PathsOf<WritePaths>::type>;  ///< how many

    /// whether the model reads and writes the whole state (decisions.md [whole-state-models])
    static constexpr bool Whole = writesWholeState<Model>;

    /** The output space: the Composite of the blocks written, or the state's space for a whole-state
     * model.
     */
    using Out = OutputSpaceIn<Model, Space>;

    /// whether the function is affine in its whole input, so its Jacobian is the same at every point
    static constexpr bool Affine = affineInEverything<Model>();

    /// whether the function returns its Jacobian with its value
    static constexpr bool ReturnsJacobian = returnsStepJacobianIn<Model, Space, double>;

private:

    using Base::argumentOf;
    using Base::inputItems;
    using Base::m_differenceRows;
    using Base::m_in;
    using Base::m_reads;
    using Base::m_stateDifferenceSize;
    using Base::stateIndexOf;
    using Base::stateItems;

public:

    /** Resolves \a call against \a space, which both outlive this. */
    ResolvedProcess(Call const & call, Space const & space)
        : Base{call.arguments(), space}
        , m_call{call}
        , m_out{outputOf<Model>(space)}
        , m_stateSize{valueDimension(space)}
    {
        if constexpr (Whole)
        {
            m_writeRows[0] = m_outRows[0] = BlockRows{0, m_stateSize};
            m_writeDifferenceRows[0] = m_outDifferenceRows[0] = BlockRows{0, differenceDimension(space)};
        }
        else
        {
            [&]<std::size_t... K>(std::index_sequence<K...>) {
                ((m_writeRows[K] = blockRows<WritePathAt<Model, K>>(space),
                  m_writeDifferenceRows[K] = differenceBlockRows<WritePathAt<Model, K>>(space),
                  m_outRows[K] = blockRows<WriteNameAt<Model, K>>(m_out),
                  m_outDifferenceRows[K] = differenceBlockRows<WriteNameAt<Model, K>>(m_out)),
                 ...);
            }(std::make_index_sequence<WriteCount>{});
        }
    }

    /** \returns the model */
    [[nodiscard]] Model const &
    model() const
    {
        return m_call.model();
    }

    /** \returns the output space instance, the written blocks' from the state */
    [[nodiscard]] Out const &
    out() const
    {
        return m_out;
    }

    /** \returns the process noise \f$ Q \f$ over \a dt, in \a Scalar over the state's differences */
    template <typename Scalar>
    [[nodiscard]] typename Moments<Space, Scalar>::MatrixType
    noise(Scalar dt) const
    {
        using Matrix = typename Moments<Space, Scalar>::MatrixType;
        auto const partial = model().template noiseOver<Scalar>(m_out, dt);
        Matrix q = Matrix::Zero(m_stateDifferenceSize, m_stateDifferenceSize);
        for (std::size_t k = 0; k < WriteCount; ++k)
        {
            for (std::size_t l = 0; l < WriteCount; ++l)
            {
                q.block(
                    m_writeDifferenceRows[k].start,
                    m_writeDifferenceRows[l].start,
                    m_writeDifferenceRows[k].size,
                    m_writeDifferenceRows[l].size
                ) =
                    partial.block(
                        m_outDifferenceRows[k].start,
                        m_outDifferenceRows[l].start,
                        m_outDifferenceRows[k].size,
                        m_outDifferenceRows[l].size
                    );
            }
        }
        return q;
    }

    /** \returns the state \a state moved forward by \a dt: the function at the blocks read and at the
     * uncertain inputs \a inputs, one vector each in the order of inputs(), or at their means when
     * none is given, written into a copy of the state by block. Not normalized. Where the function
     * returns its Jacobian, a WithJacobian whose Jacobian is one matrix over the whole state's
     * differences, or with uncertain inputs a tuple of that and one block per input, as the
     * transform takes them (decisions.md [returned-jacobians]).
     */
    template <typename State, typename TimeScalar, typename... Inputs>
    [[nodiscard]] auto
    propagate(State const & state, TimeScalar dt, Inputs const &... inputs) const
    {
        static_assert(
            sizeof...(Inputs) == InputCount || sizeof...(Inputs) == 0,
            "the step takes one vector per uncertain input, or none to read each at its mean"
        );
        using T = ScalarOf<State>;
        using Vector = typename Moments<Space, T>::VectorType;
        EMRYS_FILTER_ASSERT(state.size() == m_stateSize && "the state does not fit the space");
        auto const x =
            this->template gatherFromRows<T>(state, Eigen::Index{0}, std::tuple<Inputs const &...>{inputs...});
        auto result = vectorReturned(callForm(model().function(), m_in, m_out, x, dt));
        using Result = std::remove_cvref_t<decltype(result)>;
        Vector y{state};
        scatter(y, valueOf(result));
        if constexpr (isWithJacobian<Result>)
        {
            return withJacobian(std::move(y), expand(result.jacobian));
        }
        else
        {
            return y;
        }
    }

    /// whether the function is batched, batched(f) (decisions.md [batched-functions])
    static constexpr bool BatchedFunction = takesBatches<Function>;

    /** A matrix whose columns are vectors in the space \a S, in \a T: a batch's input or output. */
    template <typename S, typename T>
    using Columns =
        MatrixOf<T, valueDimensionAtCompileTime<S>, Eigen::Dynamic, maxValueDimensionAtCompileTime<S>, Eigen::Dynamic>;

    /** Writes into \a column the input of the point \a state, a vector in the state space, at the
     * uncertain inputs \a inputs, one vector each in the order of inputs(): a column of a batch.
     */
    template <typename T, typename Column, typename State, typename... Inputs>
    void
    gatherColumn(Column && column, State const & state, Inputs const &... inputs) const
    {
        static_assert(sizeof...(Inputs) == InputCount, "the step takes one vector per uncertain input");
        column = this->template gatherFromRows<T>(state, Eigen::Index{0}, std::tuple<Inputs const &...>{inputs...});
    }

    /** Calls the batched function once over \a x, the inputs of a batch, a column per point, over
     * \a dt, writing into \a y a column per point in the output space, with no Jacobian wanted
     * (decisions.md [batched-functions]).
     */
    template <typename X, typename TimeScalar, typename Y>
    void
    stepBatch(X const & x, TimeScalar dt, Y && y) const
        requires BatchedFunction
    {
        callBatched(model().function(), m_in, m_out, x, y, dt);
    }

    /** Writes into \a y the states \a states, a column each in the state space, moved forward by
     * \a dt by one call of the batched function, column \a j at column \a j of each of \a inputs,
     * the uncertain inputs in the order of inputs(): propagate() at each column, not normalized
     * (decisions.md [batched-functions]).
     */
    template <typename Y, typename States, typename TimeScalar, typename... Inputs>
    void
    propagateColumns(Y & y, States const & states, TimeScalar dt, Inputs const &... inputs) const
        requires BatchedFunction
    {
        auto const x = gatherStates<Y>(states, inputs...);
        auto values = ValuesOf<Y, typename States::Scalar>{};
        values.resize(valueDimension(m_out), states.cols());
        stepBatch(x, dt, values);
        scatterColumns(y, states, values);
    }

    /** propagateColumns() with the step's Jacobian over the items the model's affineIn() factors out
     * written into \a jacobians, a block of columns per point in the order of reads(), the coordinates
     * affineCoordinates() gives, its rows the whole state's differences: the identity on a state block
     * factored out and not written, which the step carries, and the function's rows on the blocks
     * written (internal::callBatchedWithJacobians())
     */
    template <typename Y, typename J, typename States, typename TimeScalar, typename... Inputs>
    void
    propagateColumnsWithAffineJacobians(
        Y & y,
        J & jacobians,
        States const & states,
        TimeScalar dt,
        Inputs const &... inputs
    ) const
        requires BatchedFunction
    {
        using T = typename States::Scalar;
        auto const x = gatherStates<Y>(states, inputs...);
        Eigen::Index const count = states.cols();
        auto values = ValuesOf<Y, T>{};
        values.resize(valueDimension(m_out), count);
        auto wanted = std::array<bool, std::tuple_size_v<InputBlocks<In>>>{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((wanted[I] = this->template itemFactored<I>()), ...);
        }(std::make_index_sequence<ItemCount>{});
        Eigen::Index const l = jacobians.cols() / std::max<Eigen::Index>(count, 1);
        using OutJacobians = MatrixOf<
            T,
            valueDimensionAtCompileTime<DifferenceSpace<Out>>,
            J::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
            J::MaxColsAtCompileTime>;
        auto outJacobians = OutJacobians{};
        outJacobians.resize(differenceDimension(m_out), l * count);
        callBatchedWithJacobians(model().function(), m_in, m_out, x, wanted, values, outJacobians, dt);
        scatterColumns(y, states, values);
        if constexpr (Whole)
        {
            jacobians = outJacobians;
        }
        else
        {
            jacobians.setZero();
            Eigen::Index offset = 0;
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                (
                    [&] {
                        if constexpr (Base::template itemFactored<I>())
                        {
                            if constexpr (Base::inState(I))
                            {
                                BlockRows const rows = m_reads.differenceRows(stateIndexOf(I));
                                for (Eigen::Index j = 0; j < count; ++j)
                                {
                                    for (Eigen::Index r = 0; r < rows.size; ++r)
                                    {
                                        jacobians(rows.start + r, j * l + offset + r) = T(1);
                                    }
                                }
                            }
                            offset += m_differenceRows[I].size;
                        }
                    }(),
                    ...
                );
            }(std::make_index_sequence<ItemCount>{});
            for (std::size_t k = 0; k < WriteCount; ++k)
            {
                BlockRows const & written = m_writeDifferenceRows[k];
                BlockRows const & rows = m_outDifferenceRows[k];
                jacobians.middleRows(written.start, written.size) = outJacobians.middleRows(rows.start, rows.size);
            }
        }
    }

    /** \returns the inputs of a batch, a column per point in the input space, as many as \a Y has:
     * column \a j gathered from column \a j of \a states, a column each in the state space, and of each
     * of \a inputs, the uncertain inputs in the order of inputs()
     */
    template <typename Y, typename States, typename... Inputs>
    [[nodiscard]] auto
    gatherStates(States const & states, Inputs const &... inputs) const
    {
        static_assert(sizeof...(Inputs) == InputCount);
        using T = typename States::Scalar;
        using X = MatrixOf<
            T,
            valueDimensionAtCompileTime<In>,
            Y::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<In>,
            Y::MaxColsAtCompileTime>;
        EMRYS_FILTER_ASSERT(states.rows() == m_stateSize && "the states do not fit the space");
        auto x = X{};
        x.resize(valueDimension(m_in), states.cols());
        for (Eigen::Index j = 0; j < states.cols(); ++j)
        {
            gatherColumn<T>(x.col(j), states.col(j), inputs.col(j)...);
        }
        return x;
    }

    /** The function's values for a batch, a column per point in the output space, in \a T, as many
     * columns as \a Y has.
     */
    template <typename Y, typename T>
    using ValuesOf = MatrixOf<
        T,
        valueDimensionAtCompileTime<Out>,
        Y::ColsAtCompileTime,
        maxValueDimensionAtCompileTime<Out>,
        Y::MaxColsAtCompileTime>;

    /** Writes into \a y the states \a states with the blocks written replaced by the columns of
     * \a values, a column each in the output space: the next states, as propagate() gives them
     */
    template <typename Y, typename States, typename Values>
    void
    scatterColumns(Y & y, States const & states, Values const & values) const
    {
        y = states;
        for (Eigen::Index j = 0; j < states.cols(); ++j)
        {
            auto column = y.col(j);
            scatter(column, values.col(j));
        }
    }

    /** \returns \a state, a vector in the state space, with the blocks written replaced by \a value,
     * a vector in the output space: the next state, not normalized, as propagate() gives it
     */
    template <typename State, typename Value>
    [[nodiscard]] auto
    scattered(State const & state, Value const & value) const
    {
        using Vector = typename Moments<Space, ScalarOf<State>>::VectorType;
        Vector y{state};
        scatter(y, value);
        return y;
    }

    /** \returns the Jacobian \f$ F \f$ of the step over \a dt in the state's differences, the same
     * at every state since the function is affine in everything it reads and returns its Jacobian,
     * taken at the zero vector
     */
    template <typename Scalar>
    [[nodiscard]] typename Moments<Space, Scalar>::MatrixType
    transition(Scalar dt) const
        requires(Affine && ReturnsJacobian && InputCount == 0)
    {
        using Vector = typename Moments<Space, Scalar>::VectorType;
        auto const zero = Vector{Vector::Zero(m_stateSize)};
        return propagate(zero, dt).jacobian;
    }

private:

    /** Writes the function's result \a value, a vector in the output space, into the written blocks
     * of the state \a y.
     */
    template <typename Y, typename Value>
    void
    scatter(Y & y, Value const & value) const
    {
        EMRYS_FILTER_ASSERT(
            value.size() == valueDimension(m_out) && "a process model's function returns a vector in its output space"
        );
        if constexpr (Whole)
        {
            y = value;
        }
        else
        {
            scatterBlocks(y, value);
        }
    }

    /** scatter() of a model that writes blocks: each from its rows of \a value into its rows of \a y. */
    template <typename Y, typename Value>
    void
    scatterBlocks(Y & y, Value const & value) const
    {
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            ((middleRowsOf<
                  valueDimensionAtCompileTime<PathTag<std::tuple_element_t<K, typename PathsOf<WritePaths>::type>>>>(
                  y,
                  m_writeRows[K].start,
                  m_writeRows[K].size
              ) =
                  middleRowsOf<valueDimensionAtCompileTime<
                      PathTag<std::tuple_element_t<K, typename PathsOf<WritePaths>::type>>>>(
                      value,
                      m_outRows[K].start,
                      m_outRows[K].size
                  )),
             ...);
        }(std::make_index_sequence<WriteCount>{});
    }

    /** \returns the Jacobian \a jacobian of the function, rows in the output's differences and
     * columns in the input's, expanded to the whole state: the identity on the blocks not written,
     * the function's rows on the blocks written, in the columns of the blocks read; and with
     * uncertain inputs a tuple of that and one block per input
     */
    template <typename J>
    [[nodiscard]] auto
    expand(J const & jacobian) const
    {
        static_assert(
            MatrixExpression<J>,
            "a process model's function returns its Jacobian as one matrix over its input's differences, "
            "withJacobian(y, J)"
        );
        EMRYS_FILTER_ASSERT(
            jacobian.rows() == differenceDimension(m_out) && jacobian.cols() == differenceDimension(m_in)
            && "a returned Jacobian has a row per difference coordinate of the output and a column per one of the input"
        );
        using Scalar = typename J::Scalar;
        using StateMatrix = typename Moments<Space, Scalar>::MatrixType;
        StateMatrix f = StateMatrix::Identity(m_stateDifferenceSize, m_stateDifferenceSize);
        for (std::size_t k = 0; k < WriteCount; ++k)
        {
            f.middleRows(m_writeDifferenceRows[k].start, m_writeDifferenceRows[k].size).setZero();
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((f.block(
                      m_writeDifferenceRows[k].start,
                      m_reads.differenceRows(I).start,
                      m_writeDifferenceRows[k].size,
                      m_reads.differenceRows(I).size
                  ) =
                      jacobian.block(
                          m_outDifferenceRows[k].start,
                          m_differenceRows[stateItems(I)].start,
                          m_outDifferenceRows[k].size,
                          m_differenceRows[stateItems(I)].size
                      )),
                 ...);
            }(std::make_index_sequence<StateCount>{});
        }
        if constexpr (InputCount == 0)
        {
            return f;
        }
        else
        {
            return [&]<std::size_t... P>(std::index_sequence<P...>) {
                return std::tuple{std::move(f), inputBlock<P>(jacobian)...};
            }(std::make_index_sequence<InputCount>{});
        }
    }

    /** \returns the Jacobian's columns for uncertain input \a P, expanded to the whole state's rows */
    template <std::size_t P, typename J>
    [[nodiscard]] auto
    inputBlock(J const & jacobian) const
    {
        using Input =
            std::remove_cvref_t<decltype(inputOf(std::declval<Argument<argumentOf(inputItems(P))> const &>()))>;
        using Block = MatrixOf<
            typename J::Scalar,
            valueDimensionAtCompileTime<DifferenceSpace<Space>>,
            Input::DifferenceDimensionAtCompileTime,
            maxValueDimensionAtCompileTime<DifferenceSpace<Space>>,
            Input::MaxDifferenceDimensionAtCompileTime>;
        BlockRows const & columns = m_differenceRows[inputItems(P)];
        Block block = Block::Zero(m_stateDifferenceSize, columns.size);
        for (std::size_t k = 0; k < WriteCount; ++k)
        {
            block.middleRows(m_writeDifferenceRows[k].start, m_writeDifferenceRows[k].size) =
                jacobian.block(m_outDifferenceRows[k].start, columns.start, m_outDifferenceRows[k].size, columns.size);
        }
        return block;
    }

    Call const & m_call;                                        ///< the call
    Out m_out;                                                  ///< the output space instance
    Eigen::Index m_stateSize;                                   ///< the state's value dimension
    std::array<BlockRows, WriteCount> m_writeRows{};            ///< each written block's rows in the state's values
    std::array<BlockRows, WriteCount> m_writeDifferenceRows{};  ///< and in its differences
    std::array<BlockRows, WriteCount> m_outRows{};              ///< each written block's rows in the output's values
    std::array<BlockRows, WriteCount> m_outDifferenceRows{};    ///< and in its differences
};

/** The step of the resolved process model \a R over a fixed time step as the transforms call it:
 * at a state and a vector per uncertain input (ResolvedProcess::propagate()), and for a batched
 * function also at a batch of points in one call (decisions.md [batched-functions]). Valid while the
 * resolved model is.
 */
template <typename R, typename TimeScalar>
class ProcessStep
{
public:

    /// whether the transforms evaluate a batch of points in one call, evaluateBatch()
    static constexpr bool EvaluatesBatches = R::BatchedFunction;

    /** The step of \a resolved over \a dt; \a resolved outlives this. */
    ProcessStep(R const & resolved, TimeScalar dt)
        : m_resolved{resolved}
        , m_dt{dt}
    {}

    /** \returns the state \a x moved forward, at the uncertain inputs \a inputs (see
     * ResolvedProcess::propagate())
     */
    template <typename X, typename... Inputs>
    [[nodiscard]] auto
    operator()(X const & x, Inputs const &... inputs) const
    {
        return m_resolved.propagate(x, m_dt, inputs...);
    }

    /** Writes into \a y the states that are the columns of \a states moved forward, at the columns of
     * each of \a inputs, by one call of the batched function
     */
    template <typename Y, typename States, typename... Inputs>
    void
    evaluateBatch(Y & y, States const & states, Inputs const &... inputs) const
        requires EvaluatesBatches
    {
        m_resolved.propagateColumns(y, states, m_dt, inputs...);
    }

    /** evaluateBatch() with the step's Jacobian over the coordinates the factored transform takes as
     * affine written into \a jacobians, a block of columns per point
     */
    template <typename Y, typename J, typename States, typename... Inputs>
    void
    evaluateBatchWithAffineJacobians(Y & y, J & jacobians, States const & states, Inputs const &... inputs) const
        requires EvaluatesBatches
    {
        m_resolved.propagateColumnsWithAffineJacobians(y, jacobians, states, m_dt, inputs...);
    }

private:

    R const & m_resolved;  ///< the resolved model
    TimeScalar m_dt;       ///< the time step
};

/** A ProcessStep whose call at one state gives the next state's value alone, without a Jacobian the
 * function returns: the step a filter applies to a whole state assembled from parts, whose Jacobian
 * is not over the transform's input.
 */
template <typename R, typename TimeScalar>
class ProcessStepValues : public ProcessStep<R, TimeScalar>
{
public:

    using ProcessStep<R, TimeScalar>::ProcessStep;

    /** \returns the state \a x moved forward, at the uncertain inputs \a inputs: its value alone */
    template <typename X, typename... Inputs>
    [[nodiscard]] auto
    operator()(X const & x, Inputs const &... inputs) const
    {
        return valueOf(ProcessStep<R, TimeScalar>::operator()(x, inputs...));
    }
};

/** \returns the process model call \a call resolved against the state space \a space (see
 * ResolvedProcess)
 */
template <typename Space, typename Call>
[[nodiscard]] ResolvedProcess<Call, Space>
resolveProcess(Call const & call, Space const & space)
{
    return ResolvedProcess<Call, Space>{call, space};
}

/** Whether any of \a Args but the first is a linearizedAbout() reference. */
template <typename... Args>
inline constexpr bool referenceAfterFirst = []<std::size_t... I>(std::index_sequence<I...>) {
    return ((I > 0 && isLinearizedAbout<std::tuple_element_t<I, std::tuple<Args...>>>) || ... || false);
}(std::index_sequence_for<Args...>{});

/** \returns \a f called with the leading linearizedAbout() reference of \a args, if one leads, and
 * the rest, or with NoReference and all of \a args
 */
template <typename F, typename... Args>
[[nodiscard]] decltype(auto)
withReference(F && f, Args const &... args)
{
    static_assert(
        !referenceAfterFirst<Args...>,
        "a prediction's linearizedAbout() reference comes first among the arguments after the step, once"
    );
    if constexpr (leadsWithReference<Args...>)
    {
        return std::forward<F>(f)(args...);
    }
    else
    {
        return std::forward<F>(f)(NoReference{}, args...);
    }
}

}  // namespace internal

/** \ingroup Models
 * \returns the state \a x in \a space moved forward by \a dt through the process model \a model with
 * the call's \a args, each a sure input or an uncertain one read at its mean, before process noise:
 * the function at the blocks read, written into the state by block. Not normalized. What a
 * simulation or a test reads of a model's step.
 */
template <Space Space_, typename Derived, ProcessModel Model, typename... Args>
[[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
propagate(
    Space_ const & space,
    Eigen::MatrixBase<Derived> const & x,
    Model const & model,
    TimeStep<typename Derived::Scalar> auto const & dt,
    Args const &... args
)
{
    using Scalar = typename Derived::Scalar;
    EMRYS_FILTER_ASSERT(x.size() == valueDimension(space) && "the state does not fit the space");
    auto const call = internal::modelWith(std::cref(model), args...);
    auto const resolved = internal::resolveProcess(call, space);
    return internal::valueOf(resolved.propagate(x.derived(), internal::toScalarTime<Scalar>(dt)));
}

template <Space Space_, typename Model_, typename Method_, RealScalar Scalar_>
class FixedStepModel;

/** \ingroup Models
 * Whether \a Model is a FixedStepModel, whose predictions take no time step.
 */
template <typename Model>
inline constexpr bool isFixedStep = false;

/** \ingroup Models
 * A FixedStepModel is one.
 */
template <typename Space_, typename Model_, typename Method_, typename Scalar_>
inline constexpr bool isFixedStep<FixedStepModel<Space_, Model_, Method_, Scalar_>>{true};

template <ProcessModel Model_, typename Method_, typename Time_>
class UnboundFixedStep;

/** \ingroup Models
 * Whether \a Model is a fixed step not yet bound to a state space: an UnboundFixedStep, which
 * fixedStep() returns and a filter binds to its state's space.
 */
template <typename Model>
inline constexpr bool isUnboundStep = false;

/** \ingroup Models
 * An UnboundFixedStep is one.
 */
template <typename Model_, typename Method_, typename Time_>
inline constexpr bool isUnboundStep<UnboundFixedStep<Model_, Method_, Time_>>{true};

/** \ingroup Models
 * \brief One prediction step as a joint: the %Gaussian the step started from, the %Gaussian it
 * predicted, and the cross-covariance between them.
 *
 * predictJoint() returns it, and a filter's predictJoint() moves its state as predict() does and
 * returns it too. With \f$ x_k \f$ the state before the step and \f$ x_{k+1} \f$ after it, given
 * the measurements \f$ z_{1:k} \f$ so far, it holds \f$ p(x_k \mid z_{1:k}) \f$,
 * \f$ p(x_{k+1} \mid z_{1:k}) \f$, and \f$ C_k = \mathrm{Cov}(x_k, x_{k+1} \mid z_{1:k}) \f$ in
 * the two tangent spaces, which is everything a Rauch-Tung-Striebel backward step needs from the
 * forward pass (see smoothStep() and GaussianHistory, decisions.md [smoothing]). For an affine step
 * \f$ x_{k+1} = F x_k + w \f$, \f$ C_k = P_{k \mid k} F^T \f$.
 *
 * \tparam Space_   the state space
 * \tparam Scalar_  scalar type, \c double by default
 */
template <Space Space_, RealScalar Scalar_ = double>
struct PredictedJoint
{
    using Scalar = Scalar_;                                ///< the scalar type
    using SpaceType = Space_;                              ///< the state space
    using GaussianType = GaussianParams<Space_, Scalar_>;  ///< the type of either %Gaussian
    using MatrixType = typename GaussianType::MatrixType;  ///< the cross-covariance's type, over differences

    GaussianType filtered;   ///< \f$ p(x_k \mid z_{1:k}) \f$, the state the step started from
    GaussianType predicted;  ///< \f$ p(x_{k+1} \mid z_{1:k}) \f$, process noise included

    /** \f$ C_k \f$: a row per difference coordinate of \f$ x_k \f$, a column per one of
     * \f$ x_{k+1} \f$
     */
    MatrixType cross;
};

namespace internal
{

/** Whether predict() by \a Method through the resolved model \a R moves a state by the Jacobian of
 * the model's step at the mean, which the model's function returns: the Linearized method, no
 * uncertain inputs, and a function that returns its Jacobian [returned-jacobians].
 */
template <typename Method, typename R>
inline constexpr bool predictsByJacobian =
    std::is_same_v<Method, methods::Linearized> && R::InputCount == 0 && R::ReturnsJacobian;

/** \returns \a prior, an UncertainInput or one marked by withCross(), moved forward by \a dt
 * through the resolved model \a resolved by \a method in \a space, before process noise, with the
 * call's uncertain inputs after it: a %Gaussian, or for a marked prior an OutputWithCross.
 */
template <typename Method, typename SpaceType, typename Prior, typename R, typename TimeScalar>
[[nodiscard]] auto
transformStep(Method const & method, SpaceType const & space, Prior const & prior, R const & resolved, TimeScalar dt)
{
    auto const f = ProcessStep{resolved, dt};
    return std::apply(
        [&](auto const &... inputs) {
            if constexpr (isSigmaPointMethod<Method> && R::factorsAffine())
            {
                auto const factored = FactoredAt<R::affineSize(), R::affineMaxSize(), Method>{
                    method,
                    resolved.template affineCoordinates<true>()
                };
                return transformBy(factored, f, space, prior, inputs...);
            }
            else
            {
                return transformBy(method, f, space, prior, inputs...);
            }
        },
        resolved.inputs()
    );
}

/** \returns \a prior copied into a %Gaussian, as PredictedJoint::filtered holds it */
template <UncertainInput Prior>
[[nodiscard]] GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>
gaussianOf(Prior const & prior)
{
    using Result = GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>;
    if constexpr (std::is_same_v<Prior, Result>)
    {
        return prior;
    }
    else
    {
        return {
            prior.space(),
            typename Result::VectorType{prior.mean()},
            typename Result::MatrixType{covarianceLowerOf(prior)}
        };
    }
}

/** \returns the PredictedJoint of \a prior and \a step, the step's output with \a noise added to its
 * covariance, and its cross-covariance
 */
template <UncertainInput Prior, typename Form, typename Noise>
[[nodiscard]] PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>
jointOf(
    Prior const & prior,
    OutputWithCross<typename Prior::SpaceType, typename Prior::SpaceType, Form, typename Prior::Scalar> step,
    Noise const & noise
)
{
    using Matrix = typename PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>::MatrixType;
    Matrix cross = step.crossCovariance();
    auto predicted = std::move(step).output();
    predicted.setCovariance(predicted.covarianceLower() + noise);
    return {gaussianOf(prior), std::move(predicted), std::move(cross)};
}

/** \returns the PredictedJoint of \a prior and \a step, whose output already holds the process noise */
template <UncertainInput Prior, typename Form>
[[nodiscard]] PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>
jointOf(
    Prior const & prior,
    OutputWithCross<typename Prior::SpaceType, typename Prior::SpaceType, Form, typename Prior::Scalar> step
)
{
    using Matrix = typename PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>::MatrixType;
    Matrix cross = step.crossCovariance();
    return {gaussianOf(prior), std::move(step).output(), std::move(cross)};
}

/** \returns the step of the resolved model \a resolved over \a dt at \a x with its Jacobian at
 * \a x over the state's differences, as a WithJacobian, where predictsByJacobian says the function
 * gives it
 */
template <typename R, typename Derived, typename TimeScalar>
[[nodiscard]] auto
stepWithJacobian(R const & resolved, Eigen::MatrixBase<Derived> const & x, TimeScalar dt)
{
    return resolved.propagate(x.derived(), dt);
}

/** Moves \a state forward by \a dt through the resolved model \a resolved by the Jacobian \f$ F \f$ of
 * its step at the mean, which the function gives (see predictsByJacobian): \f$ x' = f(x) \f$,
 * \f$ P' = F P F^T + Q \f$, with the covariance's lower triangle mirrored first (decisions.md
 * [borrowed-intermediates]), and \f$ Q \f$ added by \a addNoise, called with \f$ F P F^T \f$: the
 * model's own over \a dt for a step given at each call, the cached one for a fixed step.
 */
template <typename Space, typename Scalar, typename R, typename AddNoise>
void
predictInPlaceByJacobianWithNoise(
    GaussianParams<Space, Scalar> & state,
    R const & resolved,
    Scalar dt,
    AddNoise const & addNoise
)
{
    using Matrix = typename GaussianParams<Space, Scalar>::MatrixType;
    auto const & space = state.space();
    auto step = stepWithJacobian(resolved, state.mean(), dt);
    auto & moments = GaussianAccess::moments(state);
    Matrix & cov = moments.covarianceLower();
    cov.template triangularView<Eigen::StrictlyUpper>() = cov.transpose();
    auto propagated = Matrix{};
    propagated.noalias() = multiply(step.jacobian, cov);
    cov.noalias() = multiply(propagated, step.jacobian.transpose());
    addNoise(cov);
    moments.mean() = std::move(step.value);
    normalizeValues(space, moments.mean());
}

/** predictInPlaceByJacobianWithNoise() of \a state over \a dt through \a resolved, with the model's
 * own process noise over \a dt
 */
template <typename Space, typename Scalar, typename R>
void
predictInPlaceByJacobian(GaussianParams<Space, Scalar> & state, R const & resolved, Scalar dt)
{
    auto const addNoise = [&](typename GaussianParams<Space, Scalar>::MatrixType & cov) {
        cov += resolved.template noise<Scalar>(dt);
    };
    predictInPlaceByJacobianWithNoise(state, resolved, dt, addNoise);
}

/** \returns the LinearizationAbout of the resolved model \a resolved's step over \a dt in \a space
 * by \a method, about \a about in the chart of \a base (decisions.md [linearization-reference]):
 * where predictsByJacobian says the function gives the Jacobian of its step, that Jacobian and the
 * step at the reference's point, which leave nothing unexplained; otherwise linearizeAbout() of the
 * step, with the call's uncertain inputs after the state.
 */
template <typename Method, typename R, typename Space, typename Base, typename Reference>
[[nodiscard]] LinearizationAbout<Space, Space, typename Base::Scalar>
linearizeStepAbout(
    Method const & method,
    R const & resolved,
    Space const & space,
    typename Base::Scalar dt,
    Eigen::MatrixBase<Base> const & base,
    LinearizedAbout<Reference> const & about
)
{
    using Scalar = typename Base::Scalar;
    using Result = LinearizationAbout<Space, Space, Scalar>;
    if constexpr (predictsByJacobian<Method, R>)
    {
        typename Moments<Space, Scalar>::VectorType const point = about.mean();
        auto step = stepWithJacobian(resolved, point, dt);
        Eigen::Index const n = differenceDimension(space);
        auto result = Result{
            space,
            typename Result::OutputVector{std::move(step.value)},
            typename Result::JacobianMatrix{std::move(step.jacobian)},
            Result::ErrorMatrix::Zero(n, n),
            Result::ErrorMatrix::Zero(n, n)
        };
        normalizeValues(space, result.value);
        toChartOf(space, base, point, result);
        return result;
    }
    else
    {
        auto const f = ProcessStep{resolved, dt};
        return std::apply(
            [&](auto const &... inputs) {
                if constexpr (isSigmaPointMethod<Method> && R::factorsAffine())
                {
                    auto const factored = FactoredAt<R::affineSize(), R::affineMaxSize(), Method>{
                        method,
                        resolved.template affineCoordinates<true>()
                    };
                    return linearizeAbout(factored, f, space, space, base, about, inputs...);
                }
                else
                {
                    return linearizeAbout(method, f, space, space, base, about, inputs...);
                }
            },
            resolved.inputs()
        );
    }
}

/** \returns \a prior moved forward by \a dt through the resolved model \a resolved by \a method
 * linearized about \a about (see linearizedAbout()), with the process noise \a noise added, and the
 * prior's cross-covariance with the prediction (decisions.md [linearization-reference]): from the
 * LinearizationAbout \f$ (v, F, \Omega) \f$ of the step in the prior's chart
 * (linearizeStepAbout()), the mean \f$ v \f$, the covariance \f$ F P F^T + N + \Omega \f$, with
 * \f$ Q \f$ in \f$ N \f$, and \f$ C = P F^T \f$, in the Jacobian form.
 */
template <typename Method, UncertainInput Prior, typename R, typename Derived, typename Reference>
[[nodiscard]] OutputWithCross<
    typename Prior::SpaceType,
    typename Prior::SpaceType,
    JacobianForm,
    typename Prior::Scalar>
predictAbout(
    Method const & method,
    Prior const & prior,
    R const & resolved,
    typename Prior::Scalar dt,
    Eigen::MatrixBase<Derived> const & noise,
    LinearizedAbout<Reference> const & about
)
{
    using Space = typename Prior::SpaceType;
    using Scalar = typename Prior::Scalar;
    using Result = OutputWithCross<Space, Space, JacobianForm, Scalar>;
    using Matrix = typename GaussianParams<Space, Scalar>::MatrixType;
    auto const & space = prior.space();
    auto linearization = linearizeStepAbout(method, resolved, space, dt, prior.mean(), about);
    Matrix const cov = symmetricFromLower<Matrix>(covarianceLowerOf(prior));
    typename Result::CrossMatrix cross = multiply(cov, linearization.jacobian.transpose());
    Matrix explained = multiply(linearization.jacobian, cross);
    explained = (Scalar(0.5) * (explained + explained.transpose())).eval();
    // The process noise is one more input the step does not explain.
    Matrix others = linearization.inputs + symmetricFromLower<Matrix>(noise);
    return Result{
        space,
        std::move(linearization.value),
        std::move(linearization.jacobian),
        std::move(cross),
        std::move(explained),
        std::move(others),
        std::move(linearization.unexplained)
    };
}

/** \returns \a prior moved forward by \a dt through the resolved model \a resolved by \a method, with
 * the process noise \a noise added: the body of both forms of predict(). With a reference other
 * than NoReference, the step is linearized about it (predictAbout()).
 */
template <typename Method, UncertainInput Prior, typename R, typename Derived, typename Reference = NoReference>
[[nodiscard]] GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>
predictWithNoise(
    Method const & method,
    Prior const & prior,
    R const & resolved,
    typename Prior::Scalar dt,
    Eigen::MatrixBase<Derived> const & noise,
    Reference const & reference = {}
)
{
    if constexpr (isLinearizedAbout<Reference>)
    {
        return predictAbout(method, prior, resolved, dt, noise, reference).output();
    }
    else
    {
        auto predicted = transformStep(method, prior.space(), prior, resolved, dt);
        predicted.setCovariance(predicted.covarianceLower() + noise);
        return predicted;
    }
}

/** \returns predictWithNoise() as a PredictedJoint: the body of both forms of predictJoint() */
template <typename Method, UncertainInput Prior, typename R, typename Derived, typename Reference = NoReference>
[[nodiscard]] PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>
predictJointWithNoise(
    Method const & method,
    Prior const & prior,
    R const & resolved,
    typename Prior::Scalar dt,
    Eigen::MatrixBase<Derived> const & noise,
    Reference const & reference = {}
)
{
    if constexpr (isLinearizedAbout<Reference>)
    {
        return jointOf(prior, predictAbout(method, prior, resolved, dt, noise, reference));
    }
    else
    {
        return jointOf(prior, transformStep(method, prior.space(), withCross(prior), resolved, dt), noise);
    }
}

/** \returns \a prior moved forward by \a dt through the resolved model \a resolved by \a method with
 * the process noise \a noise added: by the function's Jacobian at the mean where predictsByJacobian
 * says it gives one and no reference is given, and otherwise through the transform
 */
template <typename Method, UncertainInput Prior, typename R, typename Derived, typename Reference>
[[nodiscard]] GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>
predictResolved(
    Method const & method,
    Prior const & prior,
    R const & resolved,
    typename Prior::Scalar dt,
    Eigen::MatrixBase<Derived> const & noise,
    Reference const & reference
)
{
    using Result = GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>;
    if constexpr (!isLinearizedAbout<Reference> && predictsByJacobian<Method, R>)
    {
        // The function gives its Jacobian at the mean, so the Linearized prediction needs no
        // differentiation: x' = f(x), P' = F P F^T + Q.
        auto result = Result{
            prior.space(),
            typename Result::VectorType{prior.mean()},
            typename Result::MatrixType{covarianceLowerOf(prior)}
        };
        predictInPlaceByJacobianWithNoise(result, resolved, dt, [&](typename Result::MatrixType & cov) {
            cov += noise;
        });
        return result;
    }
    else
    {
        return predictWithNoise(method, prior, resolved, dt, noise, reference);
    }
}

/** Whether a filter with process model \a Model may predict by \a Method: any method, unless the
 * model is a FixedStepModel, which carries its own.
 */
template <typename Model, typename Method>
inline constexpr bool predictsBy = true;

/** A FixedStepModel predicts by its own method only. */
template <typename Space_, typename Model_, typename ModelMethod, typename Scalar_, typename Method>
inline constexpr bool predictsBy<FixedStepModel<Space_, Model_, ModelMethod, Scalar_>, Method>{
    std::is_same_v<ModelMethod, Method>
};

/** Refuses an Iterated prediction method at compile time, where a filter or a fixed-step model
 * names it rather than at its first prediction: a prediction has no posterior to relinearize at
 * (decisions.md [iterated-update]). Each class that takes a prediction method calls it inside a
 * static_assert, so that the class's type is enough to refuse it.
 *
 * \returns true, when it compiles
 */
template <typename Method>
[[nodiscard]] consteval bool
checkPredictionMethod()
{
    static_assert(
        !isIterated<Method>,
        "Iterated is a measurement update's method: a prediction has no posterior to relinearize at, so give the "
        "filter or the fixed-step model its inner method"
    );
    return true;
}

/** The prediction method of a filter with process model \a Model, unless it is given one:
 * Linearized.
 */
template <typename Model>
struct DefaultMethodOf
{
    using type = methods::Linearized;  ///< the method
};

/** A FixedStepModel's own method. */
template <typename Space_, typename Model_, typename Method_, typename Scalar_>
struct DefaultMethodOf<FixedStepModel<Space_, Model_, Method_, Scalar_>>
{
    using type = Method_;  ///< the method
};

}  // namespace internal

/** \ingroup Models
 * The prediction method of a filter with process model \a Model, unless it is given one:
 * Linearized, a FixedStepModel's own, or, for a SwitchingModel of fixed-step models, which each carry
 * their own, ModelsMethod. The filters default their method parameter to it, so their deduction guides need no form of
 * their own for a fixed-step model, and a filter's type is spelled with it (<tt>KalmanFilter<Gaussian<State>, Model,
 * DefaultMethodFor<Model>></tt>).
 */
template <typename Model>
using DefaultMethodFor = typename internal::DefaultMethodOf<Model>::type;

/** \ingroup Models
 * What a filter with a fixed-step model takes in place of a prediction method: nothing, since
 * the model carries its own.
 */
struct ModelsMethod
{};

/** \ingroup Models
 * The type of a filter constructor's method argument: \a Method, or ModelsMethod for a
 * fixed-step \a Model, so that a fixed-step filter can't be given a method. One constructor serves
 * both kinds of model, so the deduction guides, through the abbreviation aliases too, have one
 * implicit candidate to rank against.
 */
template <typename Model, typename Method>
using MethodFor = std::conditional_t<isFixedStep<Model>, ModelsMethod, Method>;

namespace internal
{

/** \returns the method \a method a filter with a process model other than a fixed-step one was
 * given
 */
template <typename Model, typename Method>
    requires(!std::is_same_v<Method, ModelsMethod>)
[[nodiscard]] Method
predictionMethod(Model const &, Method method)
{
    return method;
}

/** \returns a FixedStepModel's own method */
template <typename Space_, typename Model_, typename Method_, typename Scalar_>
[[nodiscard]] Method_
predictionMethod(FixedStepModel<Space_, Model_, Method_, Scalar_> const & model, ModelsMethod)
{
    return model.method();
}

/** Whether \a Model caches its transition: a FixedStepModel whose CachesTransition is true. */
template <typename Model>
inline constexpr bool cachesTransition = false;

/** A FixedStepModel caches it as its CachesTransition says. */
template <typename Space_, typename Model_, typename Method_, typename Scalar_>
inline constexpr bool cachesTransition<FixedStepModel<Space_, Model_, Method_, Scalar_>> =
    FixedStepModel<Space_, Model_, Method_, Scalar_>::CachesTransition;

/** What a FixedStepModel holds in place of a transition it doesn't cache. */
struct NoTransition
{};

}  // namespace internal

/** \ingroup Models
 * \returns \a prior moved forward by \a dt through \a model, computed by \a method
 * (Linearized or a sigma-point method; Linearized is exact for the built-in models), with the
 * process noise added: \f$ P' = F P F^T + \sum_i G_i \Sigma_i G_i^T + Q \f$ in the linear case,
 * with \f$ G_i \f$ the Jacobian in uncertain input \f$ i \f$.
 *
 * \a args are the call's inputs, in any order, each matched to an item of the model's reads() that
 * the state lacks by its tag (decisions.md [prediction-inputs], [model-policies]): an uncertain
 * input (see UncertainInput) over a tagged space, such as an inertial measurement with its noise,
 * whose uncertainty is carried through the dynamics; or <tt>sure<Tag>(value)</tt> for one known
 * exactly, such as a control. An item the state lacks must be given, and an argument the model does
 * not read is refused, both at compile time. A sigma-point method spreads its points over the
 * coordinates the model's affineIn() does not name, and carries the ones it names, of linear blocks
 * and uncertain inputs, exactly by their %Gaussian given the rest (decisions.md [affine-coordinates]).
 *
 * \a dt is any TimeStep, converted once to the prior's scalar, which is what the model's function
 * receives (see \ref Time).
 *
 * Under the Linearized method with no uncertain inputs, a model whose function returns its Jacobian
 * [returned-jacobians], as the built-in models do, moves the covariance by that Jacobian at the
 * mean, \f$ F P F^T + Q \f$, with no differentiation.
 *
 * A linearizedAbout() reference first among \a args linearizes the step about it rather than at
 * the prior (decisions.md [linearization-reference]): the step's Jacobian \f$ F \f$ and value at
 * the reference's point, or its statistical linearization over the reference's distribution, are
 * carried to the prior's chart, and the prediction is \f$ v \f$, the affine model's value at the
 * prior mean, with covariance \f$ F P F^T + \Omega + Q \f$. A model that gives its Jacobian gives
 * it at the reference.
 *
 * The prediction step of \ref Kalman1960 "Kalman (1960)", with the transform standing in for
 * the transition matrix where the dynamics are not affine (\ref Sarkka2013 "Särkkä (2013)",
 * §4.3, §5.2, and §5.5 for the three methods).
 *
 * \sa \ref Models, transform(), FixedStepModel, linearizedAbout()
 */
template <typename Method, UncertainInput Prior, ProcessModel Model, typename... Args>
[[nodiscard]] GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>
predict(
    Method const & method,
    Prior const & prior,
    Model const & model,
    TimeStep<typename Prior::Scalar> auto const & time,
    Args const &... args
)
{
    if constexpr (!requires { prior.covarianceLower(); })
    {
        return predict(method, internal::withCovariance(prior), model, time, args...);
    }
    else
    {
        using Scalar = typename Prior::Scalar;
        auto const & space = prior.space();
        Scalar const dt = internal::toScalarTime<Scalar>(time);
        internal::assertPartitions(model, space);
        return internal::withReference(
            [&](auto const & reference, auto const &... inputs) {
                auto const call = internal::modelWith(std::cref(model), inputs...);
                auto const resolved = internal::resolveProcess(call, space);
                return internal::predictResolved(
                    method,
                    prior,
                    resolved,
                    dt,
                    resolved.template noise<Scalar>(dt),
                    reference
                );
            },
            args...
        );
    }
}

/** \ingroup Models
 * \returns predict(method, prior, model, dt, args...) as a PredictedJoint: \a prior, the
 * prediction, and their cross-covariance \f$ C_k \f$, which transform() gives with \a prior marked
 * by withCross(), for every method. The process noise and any uncertain inputs are independent of
 * \a prior, so they add to the prediction's covariance and not to \f$ C_k \f$.
 *
 * The forward half of the %Gaussian smoother of \ref Sarkka2013 "Särkkä (2013)", chapter 10, which
 * writes the Rauch-Tung-Striebel gain in terms of this cross-covariance so that it needs no
 * Jacobian (see smoothStep()).
 *
 * \sa PredictedJoint, \ref Models
 */
template <typename Method, UncertainInput Prior, ProcessModel Model, typename... Args>
[[nodiscard]] PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>
predictJoint(
    Method const & method,
    Prior const & prior,
    Model const & model,
    TimeStep<typename Prior::Scalar> auto const & time,
    Args const &... args
)
{
    if constexpr (!requires { prior.covarianceLower(); })
    {
        return predictJoint(method, internal::withCovariance(prior), model, time, args...);
    }
    else
    {
        using Scalar = typename Prior::Scalar;
        auto const & space = prior.space();
        Scalar const dt = internal::toScalarTime<Scalar>(time);
        internal::assertPartitions(model, space);
        return internal::withReference(
            [&](auto const & reference, auto const &... inputs) {
                auto const call = internal::modelWith(std::cref(model), inputs...);
                auto const resolved = internal::resolveProcess(call, space);
                return internal::predictJointWithNoise(
                    method,
                    prior,
                    resolved,
                    dt,
                    resolved.template noise<Scalar>(dt),
                    reference
                );
            },
            args...
        );
    }
}

/** \ingroup Models
 * \brief A process model taken over one fixed time step, not yet bound to a state space: what
 * fixedStep() returns.
 *
 * A filter given one binds it to its state's space and scalar, as the FixedStepModel over them,
 * which does the work that depends only on the step once, when the filter is built; so the space
 * is named once, by the filter's state. simulate() and PosteriorCramerRaoBound take one too, over
 * the space they are given, as the model and its step.
 *
 * \tparam Model_   the process model
 * \tparam Method_  the prediction method
 * \tparam Time_    the time step's type, any TimeStep (see \ref Time)
 *
 * \sa fixedStep(), FixedStepModel, \ref Models
 */
template <ProcessModel Model_, typename Method_, typename Time_>
class UnboundFixedStep
{
public:

    using ModelType = Model_;    ///< the process model taken over the step
    using MethodType = Method_;  ///< the prediction method
    using TimeType = Time_;      ///< the time step's type

    /** \a model taken over the step \a dt, predicting by \a method */
    UnboundFixedStep(Model_ model, Time_ dt, Method_ method)
        : m_model{std::move(model)}
        , m_method{std::move(method)}
        , m_timeStep{std::move(dt)}
    {}

    /** \returns the process model taken over the step */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_model;
    }

    /** \returns the prediction method */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_method;
    }

    /** \returns the time step \f$ \Delta t \f$, as it was given */
    [[nodiscard]] Time_ const &
    timeStep() const
    {
        return m_timeStep;
    }

    /** \returns whether the process model writes every coordinate of \a space once */
    template <typename Space>
    [[nodiscard]] bool
    partitions(Space const & space) const
    {
        return m_model.partitions(space);
    }

    /** \returns what the process model's types say about whether it writes every coordinate of
     * \a Space once, as its staticPartition()
     */
    template <typename Space>
    [[nodiscard]] static consteval internal::StaticPartition
    staticPartition()
    {
        return internal::staticPartition<Model_, Space>();
    }

private:

    Model_ m_model;                                   ///< the process model taken over the step
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method_ m_method;  ///< the prediction method
    Time_ m_timeStep;                                 ///< \f$ \Delta t \f$, as it was given
};

namespace internal
{

/** \returns the one instance of the fixed-size space \a Space_, to which a filter binds an
 * UnboundFixedStep
 */
template <typename Space_>
[[nodiscard]] Space_
boundSpace()
{
    static_assert(
        FixedSizeSpace<Space_> && std::default_initializable<Space_>,
        "fixedStep(model, dt) binds to a fixed-size state space; over a space sized at run time, give "
        "FixedStepModel{model, space, dt} the space"
    );
    return Space_{};
}

}  // namespace internal

/** \ingroup Models
 * \brief A process model over one fixed time step, with the work that depends only on the step
 * done once.
 *
 * \a Model_, a ProcessModel, is taken over the step \a dt, by \a Method_, in \a Space_. The process
 * noise \f$ Q \f$ is computed at construction. Under the Linearized method the transition \f$ F \f$
 * is too, when the model is affine in everything it reads, returns its Jacobian, and reads no input
 * of the call, as the built-in models are. Otherwise, and under the other methods, which depend on
 * the state, the model caches only \f$ Q \f$.
 *
 * Its predictions take no time step: predict(prior, model, args...), and a filter that owns one
 * has <tt>predict(args...)</tt> in place of <tt>predict(dt, args...)</tt>. The model carries
 * the prediction method, so a filter that owns one takes its method from it. The model doesn't
 * change after construction; to change its step or its parameters, assign a new one.
 *
 * A filter given fixedStep(model, dt) makes the one over its state's space, so the space is named
 * by the state alone:
 *
 * \code
 * auto filter = KalmanFilter{fixedStep(ConstantVelocity<Position, Velocity>{0.1}, 0.01), initial};
 * filter.predict();
 * \endcode
 *
 * Built directly, <tt>FixedStepModel{model, space, dt}</tt>, it gives its cached matrices to read.
 *
 * With \f$ F \f$ and \f$ Q \f$ fixed, an affine model's prediction is the time-invariant case of
 * \ref Kalman1960 "Kalman (1960)": \f$ \mu' = F \mu \f$ and \f$ P' = F P F^T + Q \f$.
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Model_   the process model, which writes every block of the space
 * \tparam Method_  the prediction method, Linearized by default
 * \tparam Scalar_  scalar type of the cached matrices, \c double by default
 *
 * \sa fixedStep(), UnboundFixedStep, processModel(), isFixedStep, \ref Models
 */
template <Space Space_, typename Model_, typename Method_ = methods::Linearized, RealScalar Scalar_ = double>
class FixedStepModel
{
    static_assert(ProcessModel<Model_>, "a fixed-step model is a process model taken over one step");
    static_assert(internal::checkPredictionMethod<Method_>());

public:

    using Scalar = Scalar_;                                            ///< the scalar type
    using SpaceType = Space_;                                          ///< the state space
    using ModelType = Model_;                                          ///< the process model taken over the step
    using MethodType = Method_;                                        ///< the prediction method
    using MatrixType = typename Moments<Space_, Scalar_>::MatrixType;  ///< \f$ F \f$ and \f$ Q \f$, over differences

    /// whether the model reads blocks of the state alone, so a step needs no input of the call
    static constexpr bool ReadsOnlyState = internal::readsOnlyState<Model_, Space_>();

    /** Whether the model caches the transition: under Linearized, for a model that is affine in
     * everything it reads, returns its Jacobian, and reads no input of the call, so that the
     * Jacobian of its step is the same at every state.
     */
    static constexpr bool CachesTransition = std::is_same_v<Method_, methods::Linearized> && ReadsOnlyState
                                          && internal::returnsStepJacobianIn<Model_, Space_, Scalar_>
                                          && internal::affineInEverything<Model_>();

    /** \a model taken over the step \a dt in \a space, predicting by \a method. The step is any
     * TimeStep, converted once to \a Scalar_ (see \ref Time).
     */
    FixedStepModel(Model_ model, Space_ const & space, TimeStep<Scalar_> auto const & dt, Method_ method = {})
        : m_model{std::move(model)}
        , m_method{std::move(method)}
        , m_timeStep{internal::toScalarTime<Scalar_>(dt)}
        , m_noise{internal::processNoise<Scalar_>(m_model, space, m_timeStep)}
        , m_noiseRoot{internal::defaultRoot(m_noise)}
        , m_transition{transitionOf(m_model, space, m_timeStep)}
        , m_transitionInverse{inverseOf(m_transition)}
    {
        internal::assertPartitions(m_model, space);
    }

    /** \a step bound to the space \a Space_, which is fixed-size and so has one instance: how a filter
     * given fixedStep(model, dt) makes its model. Implicit, so that a filter's constructor takes
     * the step where its deduction guide names this type.
     */
    template <TimeStep<Scalar_> Time>
    FixedStepModel(UnboundFixedStep<Model_, Method_, Time> const & step)
        : FixedStepModel{step.model(), internal::boundSpace<Space_>(), step.timeStep(), step.method()}
    {}

    /** \returns what the process model's types say about whether it writes every coordinate of
     * \a Space once, as its staticPartition()
     */
    template <typename Space>
    [[nodiscard]] static consteval internal::StaticPartition
    staticPartition()
    {
        return internal::staticPartition<Model_, Space>();
    }

    /** \returns the process model taken over the step */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_model;
    }

    /** \returns the prediction method */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_method;
    }

    /** \returns the time step \f$ \Delta t \f$, in the filter's scalar (see \ref Time) */
    [[nodiscard]] Scalar_
    timeStep() const
    {
        return m_timeStep;
    }

    /** \returns the process noise \f$ Q \f$ over the step, over the space's differences */
    [[nodiscard]] MatrixType const &
    noise() const
    {
        return m_noise;
    }

    /** \returns the transition \f$ F \f$ over the step, over the space's differences */
    [[nodiscard]] MatrixType const &
    transition() const
        requires CachesTransition
    {
        return m_transition;
    }

    /** \returns a factor \f$ G \f$ of the noise, \f$ G G^T = Q \f$, computed once (see
     * internal::defaultRoot()): what the square-root forms add the noise through, so a
     * fixed step factorizes it once rather than on every prediction
     */
    [[nodiscard]] MatrixType const &
    noiseRoot() const
    {
        return m_noiseRoot;
    }

    /** \returns the inverse of the transition, computed once, or nothing if the transition is
     * singular: what the square-root information form predicts through
     */
    [[nodiscard]] std::optional<MatrixType> const &
    transitionInverse() const
        requires CachesTransition
    {
        return m_transitionInverse;
    }

    /** \returns the state \a x in \a space moved forward by the step, for a model that reads blocks
     * of the state alone. Not normalized. With inputs of the call, Emrys::propagate() over model()
     * and timeStep().
     */
    template <typename Derived>
    [[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
    propagate(Space_ const & space, Eigen::MatrixBase<Derived> const & x) const
        requires ReadsOnlyState
    {
        return Emrys::propagate(space, x, m_model, m_timeStep);
    }

    /** \returns whether the model writes every coordinate of \a space once and its matrices fit it */
    [[nodiscard]] bool
    partitions(Space_ const & space) const
    {
        return m_model.partitions(space) && m_noise.rows() == differenceDimension(space);
    }

private:

    /** \returns the model's transition matrix over the step \a dt in \a space, or nothing where the
     * model doesn't cache one
     */
    [[nodiscard]] static auto
    transitionOf(
        [[maybe_unused]] Model_ const & model,
        [[maybe_unused]] Space_ const & space,
        [[maybe_unused]] Scalar_ dt
    )
    {
        if constexpr (CachesTransition)
        {
            auto const call = internal::modelWith(std::cref(model));
            auto const resolved = internal::resolveProcess(call, space);
            return MatrixType{resolved.template transition<Scalar_>(dt)};
        }
        else
        {
            return internal::NoTransition{};
        }
    }

    /** \returns the inverse of \a transition, or nothing where it is singular or not cached */
    [[nodiscard]] static auto
    inverseOf([[maybe_unused]] auto const & transition)
    {
        if constexpr (CachesTransition)
        {
            auto const lu = Eigen::FullPivLU<MatrixType>{transition};
            return lu.isInvertible() ? std::optional<MatrixType>{internal::inverseFrom<MatrixType>(lu)} : std::nullopt;
        }
        else
        {
            return internal::NoTransition{};
        }
    }

    Model_ m_model;                                   ///< the process model taken over the step
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method_ m_method;  ///< the prediction method
    Scalar_ m_timeStep;                               ///< \f$ \Delta t \f$, as a scalar
    MatrixType m_noise;                               ///< \f$ Q \f$
    MatrixType m_noiseRoot;                           ///< \f$ G \f$ with \f$ G G^T = Q \f$

    /** \f$ F \f$, where the model caches it */
    EMRYS_FILTER_NO_UNIQUE_ADDRESS std::conditional_t<CachesTransition, MatrixType, internal::NoTransition>
        m_transition;

    /** \f$ F^{-1} \f$, where the model caches the transition and it is invertible */
    EMRYS_FILTER_NO_UNIQUE_ADDRESS
    std::conditional_t<CachesTransition, std::optional<MatrixType>, internal::NoTransition>
        m_transitionInverse;
};

/// \cond DEDUCTION_GUIDES
template <typename Model, typename Space, typename Time>
FixedStepModel(Model, Space, Time) -> FixedStepModel<Space, Model, methods::Linearized, double>;

template <typename Model, typename Space, typename Time, typename Method>
FixedStepModel(Model, Space, Time, Method) -> FixedStepModel<Space, Model, Method, double>;

/// \endcond

/** \ingroup Models
 * \returns \a model taken over the step \a dt, predicting by \a method: the UnboundFixedStep
 * that a filter binds to its state's space, <tt>KalmanFilter{fixedStep(model, dt), initial}</tt>,
 * as the FixedStepModel over that space and the state's scalar.
 */
template <ProcessModel Model, typename Time, typename Method = methods::Linearized>
    requires TimeStep<Time, double>
[[nodiscard]] UnboundFixedStep<Model, Method, Time>
fixedStep(Model model, Time dt, Method method = {})
{
    return UnboundFixedStep<Model, Method, Time>{std::move(model), std::move(dt), std::move(method)};
}

namespace internal
{

/** The process model a filter over \a Space and \a Scalar keeps of \a Model: \a Model itself. */
template <typename Model, typename Space, typename Scalar>
struct BoundStepOf
{
    using type = Model;  ///< the model
};

/** An UnboundFixedStep's is the FixedStepModel over \a Space and \a Scalar. */
template <typename Model_, typename Method_, typename Time_, typename Space, typename Scalar>
struct BoundStepOf<UnboundFixedStep<Model_, Method_, Time_>, Space, Scalar>
{
    using type = FixedStepModel<Space, Model_, Method_, Scalar>;  ///< the model
};

/** The process model a filter over \a Space and \a Scalar keeps of \a Model, which its deduction
 * guides name: a FixedStepModel for fixedStep()'s result, and otherwise \a Model.
 */
template <typename Model, typename Space, typename Scalar>
using BoundStep = typename BoundStepOf<Model, Space, Scalar>::type;

/** An UnboundFixedStep's method is its own. */
template <typename Model_, typename Method_, typename Time_>
struct DefaultMethodOf<UnboundFixedStep<Model_, Method_, Time_>>
{
    using type = Method_;  ///< the method
};

}  // namespace internal

namespace internal
{

/** \returns \a prior moved forward by \a model's cached transition \f$ F \f$, with its noise
 * \f$ Q \f$ added, and the cross-covariance \f$ P F^T \f$: the body of predict() and
 * predictJoint() for a FixedStepModel that caches its transition, given no inputs.
 */
template <UncertainInput Prior, typename Space_, typename Model_, typename Method_, typename Scalar_>
    requires FixedStepModel<Space_, Model_, Method_, Scalar_>::CachesTransition
[[nodiscard]] OutputWithCross<Space_, Space_, JacobianForm, Scalar_>
predictByTransition(Prior const & prior, FixedStepModel<Space_, Model_, Method_, Scalar_> const & model)
{
    using MatrixType = typename FixedStepModel<Space_, Model_, Method_, Scalar_>::MatrixType;
    auto const & space = prior.space();
    auto const & f = model.transition();
    typename Moments<Space_, Scalar_>::VectorType mean{model.propagate(space, prior.mean())};
    normalizeValues(space, mean);
    MatrixType cross = multiply(covarianceLowerOf(prior), f.transpose());
    MatrixType explained = multiply(f, cross);
    explained = (Scalar_(0.5) * (explained + explained.transpose())).eval();
    Eigen::Index const n = explained.rows();
    // The process noise is one more input the transition does not explain.
    return OutputWithCross<Space_, Space_, JacobianForm, Scalar_>{
        space,
        std::move(mean),
        MatrixType{f},
        std::move(cross),
        std::move(explained),
        MatrixType{model.noise()},
        MatrixType::Zero(n, n)
    };
}

}  // namespace internal

/** \ingroup Models
 * \returns \a prior moved forward by \a model's step, by its method, with its process noise
 * added, as predict(method, prior, model, dt, args...) would.
 *
 * Under a cached transition (FixedStepModel::CachesTransition) and with no arguments, the mean is
 * the propagated mean and the covariance \f$ F P F^T + Q \f$, with no transform. Otherwise it goes
 * as predict() goes, with the cached \f$ Q \f$: by the function's Jacobian at the mean where it
 * returns one and no input is uncertain, and through the transform where an uncertain input needs
 * its own Jacobian.
 *
 * \sa FixedStepModel, \ref Models
 */
template <UncertainInput Prior, typename Space_, typename Model_, typename Method_, typename Scalar_, typename... Args>
[[nodiscard]] GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>
predict(Prior const & prior, FixedStepModel<Space_, Model_, Method_, Scalar_> const & model, Args const &... args)
{
    static_assert(std::is_same_v<typename Prior::SpaceType, Space_>, "the model must be over the prior's space");
    static_assert(std::is_same_v<typename Prior::Scalar, Scalar_>, "the model's scalar must be the prior's");
    using Model = FixedStepModel<Space_, Model_, Method_, Scalar_>;
    [[maybe_unused]] auto const & space = prior.space();
    internal::assertPartitions(model, space);
    if constexpr (Model::CachesTransition && sizeof...(Args) == 0)
    {
        return internal::predictByTransition(prior, model).output();
    }
    else
    {
        return internal::withReference(
            [&](auto const & reference, auto const &... inputs) {
                auto const call = internal::modelWith(std::cref(model.model()), inputs...);
                auto const resolved = internal::resolveProcess(call, space);
                return internal::predictResolved(
                    model.method(),
                    prior,
                    resolved,
                    model.timeStep(),
                    model.noise(),
                    reference
                );
            },
            args...
        );
    }
}

/** \ingroup Models
 * \returns predict(prior, model, args...) as a PredictedJoint, as
 * predictJoint(method, prior, model, dt, args...) gives one. Under a cached transition and with
 * no arguments, \f$ C_k = P F^T \f$, which the prediction computes on the way to \f$ F P F^T \f$.
 *
 * \sa FixedStepModel, PredictedJoint, \ref Models
 */
template <UncertainInput Prior, typename Space_, typename Model_, typename Method_, typename Scalar_, typename... Args>
[[nodiscard]] PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>
predictJoint(Prior const & prior, FixedStepModel<Space_, Model_, Method_, Scalar_> const & model, Args const &... args)
{
    static_assert(std::is_same_v<typename Prior::SpaceType, Space_>, "the model must be over the prior's space");
    static_assert(std::is_same_v<typename Prior::Scalar, Scalar_>, "the model's scalar must be the prior's");
    using Model = FixedStepModel<Space_, Model_, Method_, Scalar_>;
    [[maybe_unused]] auto const & space = prior.space();
    internal::assertPartitions(model, space);
    if constexpr (Model::CachesTransition && sizeof...(Args) == 0)
    {
        return internal::jointOf(prior, internal::predictByTransition(prior, model));
    }
    else
    {
        return internal::withReference(
            [&](auto const & reference, auto const &... inputs) {
                auto const call = internal::modelWith(std::cref(model.model()), inputs...);
                auto const resolved = internal::resolveProcess(call, space);
                return internal::predictJointWithNoise(
                    model.method(),
                    prior,
                    resolved,
                    model.timeStep(),
                    model.noise(),
                    reference
                );
            },
            args...
        );
    }
}

namespace internal
{

/** \returns \a step bound to \a prior's space and scalar: the FixedStepModel over them */
template <typename Prior, typename Model_, typename Method_, typename Time_>
[[nodiscard]] FixedStepModel<typename Prior::SpaceType, Model_, Method_, typename Prior::Scalar>
boundToPrior(Prior const & prior, UnboundFixedStep<Model_, Method_, Time_> const & step)
{
    return FixedStepModel<typename Prior::SpaceType, Model_, Method_, typename Prior::Scalar>{
        step.model(),
        prior.space(),
        step.timeStep(),
        step.method()
    };
}

}  // namespace internal

/** \ingroup Models
 * \returns predict(prior, model, args...) over the FixedStepModel that \a step, fixedStep()'s
 * result, makes over \a prior's space and scalar, as a filter given it binds it:
 * <tt>predict(prior, fixedStep(model, dt))</tt>. The model is made on every call, so its
 * \f$ Q \f$ is computed each time; to predict over the step again and again, keep a FixedStepModel
 * or give the step to a filter.
 *
 * \sa fixedStep(), FixedStepModel, \ref Models
 */
template <UncertainInput Prior, typename Model_, typename Method_, typename Time_, typename... Args>
[[nodiscard]] GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>
predict(Prior const & prior, UnboundFixedStep<Model_, Method_, Time_> const & step, Args const &... args)
{
    return Emrys::predict(prior, internal::boundToPrior(prior, step), args...);
}

/** \ingroup Models
 * \returns predictJoint(prior, model, args...) over the FixedStepModel that \a step, fixedStep()'s
 * result, makes over \a prior's space and scalar, as predict(prior, step, args...) binds it.
 *
 * \sa fixedStep(), FixedStepModel, PredictedJoint, \ref Models
 */
template <UncertainInput Prior, typename Model_, typename Method_, typename Time_, typename... Args>
[[nodiscard]] PredictedJoint<typename Prior::SpaceType, typename Prior::Scalar>
predictJoint(Prior const & prior, UnboundFixedStep<Model_, Method_, Time_> const & step, Args const &... args)
{
    return Emrys::predictJoint(prior, internal::boundToPrior(prior, step), args...);
}

namespace internal
{

/** predict() of \a state into itself: where it is when the model's function gives the Jacobian of
 * its step (see predictInPlaceByJacobian()), and otherwise by assignment.
 */
template <typename Method, typename Space, typename Scalar, ProcessModel Model, typename... Args>
void
predictInPlace(
    Method const & method,
    GaussianParams<Space, Scalar> & state,
    Model const & model,
    Scalar dt,
    Args const &... args
)
{
    withReference(
        [&](auto const & reference, auto const &... inputs) {
            auto const call = modelWith(std::cref(model), inputs...);
            auto const resolved = resolveProcess(call, state.space());
            using R = std::remove_cvref_t<decltype(resolved)>;
            if constexpr (!isLinearizedAbout<std::remove_cvref_t<decltype(reference)>> && predictsByJacobian<Method, R>)
            {
                assertPartitions(model, state.space());
                predictInPlaceByJacobian(state, resolved, dt);
            }
            else
            {
                state = Emrys::predict(method, state, model, dt, args...);
            }
        },
        args...
    );
}

/** predict() of \a state into itself over \a model's fixed step: where it is when the model's
 * function gives the Jacobian of its step and the transition is not cached, and otherwise through
 * the copy predict() returns
 */
template <
    typename Space,
    typename Scalar,
    typename Space_,
    typename Model_,
    typename Method_,
    typename Scalar_,
    typename... Args>
void
predictInPlace(
    GaussianParams<Space, Scalar> & state,
    FixedStepModel<Space_, Model_, Method_, Scalar_> const & model,
    Args const &... args
)
{
    using Model = FixedStepModel<Space_, Model_, Method_, Scalar_>;
    static_assert(std::is_same_v<Space, Space_>);
    static_assert(std::is_same_v<Scalar, Scalar_>);
    if constexpr (Model::CachesTransition && sizeof...(Args) == 0)
    {
        state = Emrys::predict(state, model);
    }
    else
    {
        withReference(
            [&](auto const & reference, auto const &... inputs) {
                auto const call = modelWith(std::cref(model.model()), inputs...);
                auto const resolved = resolveProcess(call, state.space());
                using R = std::remove_cvref_t<decltype(resolved)>;
                if constexpr (!isLinearizedAbout<std::remove_cvref_t<decltype(reference)>>
                              && predictsByJacobian<Method_, R>)
                {
                    assertPartitions(model, state.space());
                    predictInPlaceByJacobianWithNoise(state, resolved, model.timeStep(), [&](auto & cov) {
                        cov += model.noise();
                    });
                }
                else
                {
                    state = Emrys::predict(state, model, args...);
                }
            },
            args...
        );
    }
}

}  // namespace internal

}  // namespace Emrys
