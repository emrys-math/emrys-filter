#pragma once

/** \file
 * Measurement models and their observations (decisions.md [model-policies], [sensors], [linear-sensors],
 * [measurement-space-functions], [templated-measurements], [method-per-measurement]): the MeasurementFunction that
 * measurementModel() builds from a function, a matrix, or Columns and its policies, or from a likelihood alone;
 * CountFrom, a measurement space that repeats with a block of the state; the MeasurementModel concept; Observed,
 * what observe() makes of a model, a method, an observation, and the call's inputs, and the Measurement concept a
 * filter's update() takes. How an update reads one is in MeasurementReads.h. Read Measurement.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Roots.h>
#include <EmrysFilter/Models/Condition.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/TransformArgs.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** What a measurement model known by its likelihood alone holds in place of a function. */
struct NoFunction
{};

/** What a model whose output space is sized by its function's result holds in place of one: the
 * placeholder its function is given as \c out, so that a function that reads its output space by
 * block declares writes().
 */
struct NoOutputSpace
{};

/** The number of repetitions of the one resizable block of \a S at most, or Eigen::Dynamic if
 * unbounded.
 */
template <typename S>
inline constexpr int maxCountOf =
    S::ValueDimensionStep == 0 ? 0
    : maxValueDimensionAtCompileTime<S> == Eigen::Dynamic
        ? Eigen::Dynamic
        : (maxValueDimensionAtCompileTime<S> - S::FixedValueDimension) / S::ValueDimensionStep;

/** Whether every block of the space \a In adds its values, so that a matrix acts on its values as on
 * its differences: none has a plus() of its own, and each block's differences have the size of its
 * values. A space that is not a Composite is one block, as a whole-state model's input is.
 */
template <typename In>
inline constexpr bool blocksAddValues = [] {
    if constexpr (IsComposite<In>)
    {
        return []<typename... Blocks>(std::tuple<Blocks...> *) {
            return ((!HasPlus<Blocks> && hasEqualSizes<Blocks>) && ...);
        }(static_cast<CompositeBlocks<In> *>(nullptr));
    }
    else
    {
        return !HasPlus<In> && hasEqualSizes<In>;
    }
}();

}  // namespace internal

/** \ingroup Models
 * \brief The measurement space function of a model whose measurement has a block of
 * \a MeasurementSpace per repetition of the state's block \a Path, given to outputSpace()
 * (decisions.md [measurement-space-functions]).
 *
 * Called with the state's space, it returns a \a MeasurementSpace whose one resizable block has
 * the count of the one resizable block of the block \a Path (a tag or a TagPath) names: a
 * Repeated block's count(), or a resizable coordinate's coordinates. The model need not read the
 * block. A count is never inferred from the blocks a model reads (decisions.md
 * [several-resizable-blocks]); a measurement sized otherwise takes a function of its own.
 *
 * \code
 * struct RangeBearings : Repeated<RangeBearing, 32> {};
 * auto const ranging = measurementModel(h, reads(path<Landmarks>, path<Pose>),
 *                                       outputSpace(CountFrom<Landmarks, RangeBearings>{}),
 *                                       additiveNoise(onEach<RangeBearings>(r)));
 * \endcode
 *
 * \tparam Path              the path to the state's block whose count the measurement takes
 * \tparam MeasurementSpace  the measurement space, with one resizable block
 */
template <typename Path, Space MeasurementSpace>
struct CountFrom
{
    static_assert(
        internal::resizableBlockCount<MeasurementSpace> == 1,
        "CountFrom's measurement space must have exactly one resizable block"
    );

    /** \returns the measurement space for the state's space \a space */
    template <Space StateSpace>
    [[nodiscard]] MeasurementSpace
    operator()(StateSpace const & space) const
    {
        using Block = internal::PathSubspaceType<StateSpace, Path>;
        static_assert(
            internal::resizableBlockCount<Block> == 1,
            "CountFrom's block of the state must have exactly one resizable block"
        );
        static_assert(
            internal::maxCountOf<MeasurementSpace> == Eigen::Dynamic
                || (internal::maxCountOf<Block> != Eigen::Dynamic
                    && internal::maxCountOf<MeasurementSpace> >= internal::maxCountOf<Block>),
            "CountFrom's measurement space must hold a block per repetition of the state's block at its largest"
        );
        Eigen::Index const count = (valueDimension(internal::pathSubspace<Path>(space)) - Block::FixedValueDimension)
                                 / Block::ValueDimensionStep;
        auto result = MeasurementSpace{};
        result.resize(MeasurementSpace::FixedValueDimension + count * MeasurementSpace::ValueDimensionStep);
        return result;
    }
};

/** \ingroup Models
 * \brief A measurement model: a function \f$ h \f$ of its input, with the policies that say what it
 * reads, what it writes, and what it is affine in, and its noise (decisions.md [model-policies]).
 *
 * measurementModel() builds one; a user never spells the type. The function is called as
 * <tt>h(in, out, x)</tt>, <tt>h(in, x)</tt>, or <tt>h(x)</tt>, whichever it takes, with \a in the
 * instance of the input space that reads() defines, or the state's space without reads(), \a out the
 * instance of the output space, and \a x a vector in \a in, which it reads with
 * <tt>block<Name>(in, x)</tt>; it returns a vector in \a out, which it may fill block by block from
 * <tt>zeros<T>(out)</tt>, either generic in its scalar or with its Jacobian beside it
 * (withJacobian(), decisions.md [returned-jacobians]). A function generic in its scalar may take it
 * as its one template parameter, <tt>[]<typename T>(auto const & in, auto const & out, auto const &
 * x)</tt>, which the call supplies (decisions.md [function-scalars]). A model with a likelihood alone
 * (logLikelihood()) and no function serves the filters that weight points.
 *
 * \tparam F         the function, or internal::NoFunction
 * \tparam Policies  the policies, each at most once
 *
 * \sa measurementModel(), observe(), MeasurementModel
 */
template <typename F, typename... Policies>
class MeasurementFunction
{
    /** The policies, found by kind. */
    using Set = internal::PolicySet<Policies...>;

public:

    using FunctionType = F;  ///< the function

    static constexpr bool IsModel = true;  ///< a model, for internal::isModel

    /// whether the model has a function; a model known by its likelihood alone has none
    static constexpr bool HasFunction = !std::is_same_v<F, internal::NoFunction>;

    /** Whether the policy of kind \a K was given. */
    template <internal::PolicyKind K>
    static constexpr bool has = Set::template has<K>;

    /** The policy of kind \a K, or internal::NoPolicy. */
    template <internal::PolicyKind K>
    using Policy = typename Set::template Type<K>;

    /** The model of \a function with \a policies. */
    explicit MeasurementFunction(F function, Policies... policies)
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

private:

    EMRYS_FILTER_NO_UNIQUE_ADDRESS F m_function;  ///< the function
    std::tuple<Policies...> m_policies;           ///< the policies
};

namespace internal
{

/** Whether \a T is a MeasurementFunction. */
template <typename T>
inline constexpr bool isMeasurementFunction = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename F, typename... Policies>
inline constexpr bool isMeasurementFunction<MeasurementFunction<F, Policies...>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Models
 * \brief A measurement model: what measurementModel() returns, or a composition of such models
 * (decisions.md [model-policies]). observe() makes the Measurement a filter takes from one, an
 * observation, and a method.
 */
template <typename M>
concept MeasurementModel = internal::isMeasurementFunction<std::remove_cvref_t<M>>;

namespace internal
{

/** The function of a linear measurement model, \f$ z = H x \f$ for the matrix \a Matrix over the
 * model's input (decisions.md [linear-sensors]): its value with \f$ H \f$ beside it as its Jacobian,
 * so that nothing is differentiated and the update is the Kalman filter's exact step.
 */
template <typename Matrix>
class LinearFunction
{
public:

    /** The output space where the model names none: Linear, with the matrix's rows. */
    using OutputSpaceType = Linear<Matrix::RowsAtCompileTime, Matrix::MaxRowsAtCompileTime>;

    /** The function of \a matrix. */
    explicit LinearFunction(Matrix matrix)
        : m_matrix{std::move(matrix)}
    {}

    /** \returns \f$ H \f$ */
    [[nodiscard]] Matrix const &
    matrix() const
    {
        return m_matrix;
    }

    /** \returns the rows of \f$ H \f$, the output's size where the model names no output space */
    [[nodiscard]] Eigen::Index
    outputRows() const
    {
        return m_matrix.rows();
    }

    /** \returns \f$ H x \f$ for \a x, a vector in the input space \a in, with \f$ H \f$ as its
     * Jacobian; the blocks of \a in must add their values, as a Linear or Periodic one does
     */
    template <typename In, typename Out, typename X>
    [[nodiscard]] auto
    operator()([[maybe_unused]] In const & in, Out const &, X const & x) const
    {
        static_assert(
            blocksAddValues<In>,
            "a linear measurement model reads blocks with no plus() of their own; a block such as a unit quaternion "
            "has no linear measurement, so give measurementModel() a function"
        );
        static_assert(
            Matrix::ColsAtCompileTime == Eigen::Dynamic || valueDimensionAtCompileTime<In> == Eigen::Dynamic
                || Matrix::ColsAtCompileTime == valueDimensionAtCompileTime<In>,
            "a linear measurement model's matrix must have a column per coordinate of the blocks it reads"
        );
        EMRYS_FILTER_ASSERT(
            m_matrix.cols() == x.size()
            && "a linear measurement model's matrix must have a column per coordinate of the blocks it reads"
        );
        using T = typename X::Scalar;
        using Value = Eigen::Matrix<T, Matrix::RowsAtCompileTime, 1, Eigen::ColMajor, Matrix::MaxRowsAtCompileTime, 1>;
        Value value = m_matrix.template cast<T>() * x;
        return WithJacobian<Value, Matrix const &>{std::move(value), m_matrix};
    }

private:

    Matrix m_matrix;  ///< \f$ H \f$
};

/** What a block of a model's Columns says of itself, given as \a T: nothing, for anything but a
 * matrix placed by on() or onEach() on a block of the state.
 */
template <typename T>
struct ColumnsOf
{
    static constexpr bool IsPlaced = false;      ///< whether \a T is a block of columns
    static constexpr bool Each = false;          ///< whether it is read per repetition
    static constexpr int Rows = Eigen::Dynamic;  ///< the rows of its matrix, or Eigen::Dynamic
    using Path = void;                           ///< the path to the block read
};

/** A matrix read from the block that \a P names. */
template <MatrixExpression M, typename P>
struct ColumnsOf<OnPath<M, P>>
{
    static constexpr bool IsPlaced = true;             ///< a block of columns
    static constexpr bool Each = false;                ///< read whole
    static constexpr int Rows = M::RowsAtCompileTime;  ///< the rows of its matrix
    using Path = P;                                    ///< the path to the block read
};

/** A matrix read from each repetition of the repeated block that \a P names. */
template <MatrixExpression M, typename P>
    requires(!std::is_same_v<P, WholeSpace>)
struct ColumnsOf<OnEach<M, P>>
{
    static constexpr bool IsPlaced = true;             ///< a block of columns
    static constexpr bool Each = true;                 ///< read per repetition
    static constexpr int Rows = M::RowsAtCompileTime;  ///< the rows of its matrix
    using Path = P;                                    ///< the path to the block read
};

/** The fixed one of \a Rows, Eigen::Dynamic if none is, or 0 if two fixed ones differ. */
template <int... Rows>
inline constexpr int agreedRows = [] {
    int result = Eigen::Dynamic;
    for (int const rows : std::array<int, sizeof...(Rows)>{Rows...})
    {
        if (rows == Eigen::Dynamic)
        {
            continue;
        }
        if (result != Eigen::Dynamic && rows != result)
        {
            return 0;
        }
        result = rows;
    }
    return result;
}();

/** The path of the first of \a Placed read per repetition, or void if none is. */
template <typename... Placed>
struct FirstEachPathOf
{
    /** None. */
    using type = void;
};

/** The first's, if it is read per repetition, or else the rest's. */
template <typename First, typename... Rest>
struct FirstEachPathOf<First, Rest...>
{
    /** The path. */
    using type = std::
        conditional_t<ColumnsOf<First>::Each, typename ColumnsOf<First>::Path, typename FirstEachPathOf<Rest...>::type>;
};

}  // namespace internal

/** \ingroup Models
 * \brief The matrix \f$ H \f$ of a linear measurement model by its columns: a matrix per block of
 * the state read, each placed by tag path with on() or onEach() (decisions.md [linear-sensors]).
 *
 * The measurement is \f$ z = \sum_k H_k x_k \boxplus v \f$ over the blocks read, in the order
 * given, which is what the model reads. A matrix given as <tt>on<Path>(H)</tt> has a column per
 * coordinate of the block. Given as <tt>onEach<Path>(H)</tt>, it has a column per coordinate of one
 * repetition of a repeated block, and the measurement repeats with the block: repetition \f$ i \f$
 * of the measurement reads repetition \f$ i \f$ of the block, so the block's columns of \f$ H \f$
 * are \f$ I \otimes H \f$, and an <tt>on<Path>()</tt> matrix is read into every repetition. Every
 * matrix has the rows of one repetition of the measurement, or of the whole of it if nothing
 * repeats.
 *
 * \code
 * // each landmark's position relative to the vehicle's, z_i = l_i - p
 * auto const relative = measurementModel(Columns{onEach<Landmarks>(Eigen::Matrix2d::Identity()),
 *                                                on<Position>(-Eigen::Matrix2d::Identity())},
 *                                        additiveNoise(onEach<>(R)));
 * \endcode
 *
 * \tparam Placed  the matrices, as on() and onEach() return them, in the order read
 *
 * \sa measurementModel()
 */
template <typename... Placed>
class Columns
{
    static_assert(sizeof...(Placed) > 0, "a model's matrix needs at least one block of columns");
    static_assert(
        (internal::ColumnsOf<Placed>::IsPlaced && ...),
        "each block of a model's columns is a matrix placed on a block of the state, with on<Path>() or "
        "onEach<Path>()"
    );
    static_assert(
        !(internal::ColumnsOf<Placed>::IsPlaced && ...)
            || internal::distinctTypes<internal::AsTagPath<typename internal::ColumnsOf<Placed>::Path>...>(),
        "a model's columns name each block of the state once"
    );
    static_assert(
        internal::agreedRows<internal::ColumnsOf<Placed>::Rows...> != 0,
        "every block of a model's columns has the rows of one repetition of the measurement"
    );

public:

    /// whether the measurement repeats, with a block of the state read per repetition by onEach()
    static constexpr bool Repeats = (internal::ColumnsOf<Placed>::Each || ...);

    /// the rows of one repetition of the measurement, or Eigen::Dynamic
    static constexpr int Rows = internal::agreedRows<internal::ColumnsOf<Placed>::Rows...>;

    /** The paths read, in order, as a TagPaths. */
    using Paths = TagPaths<typename internal::ColumnsOf<Placed>::Path...>;

    /** The path of the first block read per repetition, or void. */
    using FirstEachPath = typename internal::FirstEachPathOf<Placed...>::type;

    /** The matrices \a placed. */
    explicit Columns(Placed... placed)
        : m_placed{std::move(placed)...}
    {
        EMRYS_FILTER_ASSERT(
            std::apply(
                [&](auto const &... each) {
                    return ((each.value.rows() == rows()) && ...);
                },
                m_placed
            )
            && "every block of a model's columns has the rows of one repetition of the measurement"
        );
    }

    /** \returns the matrices, as placed by path */
    [[nodiscard]] std::tuple<Placed...> const &
    placed() const
    {
        return m_placed;
    }

    /** \returns the rows of one repetition of the measurement */
    [[nodiscard]] Eigen::Index
    rows() const
    {
        return std::get<0>(m_placed).value.rows();
    }

private:

    std::tuple<Placed...> m_placed;  ///< the matrices, as placed by path
};

namespace internal
{

/** Whether \a T is a Columns. */
template <typename T>
inline constexpr bool isColumns = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Placed>
inline constexpr bool isColumns<Columns<Placed...>>{true};

/// \endcond

/** The measurement space function of a linear model whose Columns repeat and that was given no
 * space: a Repeated block of \a Rows Linear coordinates per repetition of the state's block
 * \a Path, up to its count at most.
 */
template <typename Path, int Rows>
struct LinearPerRepetition
{
    /** \returns the measurement space in the state's space \a space */
    template <typename StateSpace>
    [[nodiscard]] auto
    operator()(StateSpace const & space) const
    {
        using Block = PathSubspaceType<StateSpace, Path>;
        static_assert(
            Block::FixedValueDimension == 0 && Block::ValueDimensionStep > 0,
            "a model's onEach<Path>() repeats over a resizable block of the state, and the block named is fixed-size"
        );
        return CountFrom<Path, Repeated<Linear<Rows>, maxCountOf<Block>>>{}(space);
    }
};

/** The function of a linear measurement model given its matrix by Columns (decisions.md
 * [linear-sensors]): \f$ \sum_k H_k x_k \f$ over the blocks read, which are the blocks of its input
 * in the columns' order, with \f$ H \f$ over the whole input beside it as its Jacobian: the
 * block-diagonal \f$ I \otimes H_k \f$ for an onEach() block and \f$ H_k \f$ stacked once per
 * repetition for an on() block.
 */
template <typename... Placed>
class ColumnsFunction
{
    /// the scalar of the matrices
    using Scalar = typename std::remove_cvref_t<
        decltype(std::get<0>(std::declval<std::tuple<Placed...> const &>()).value)>::Scalar;

public:

    /** The output space where the model names none and nothing repeats: Linear, with the matrices'
     * rows.
     */
    using OutputSpaceType = Linear<Columns<Placed...>::Rows>;

    /** The function of \a columns. */
    explicit ColumnsFunction(Columns<Placed...> columns)
        : m_columns{std::move(columns)}
    {}

    /** \returns \f$ H \f$, by its columns */
    [[nodiscard]] Columns<Placed...> const &
    columns() const
    {
        return m_columns;
    }

    /** \returns the rows of the measurement where nothing repeats: the matrices' rows, the
     * output's size where the model names no output space
     */
    [[nodiscard]] Eigen::Index
    outputRows() const
        requires(!Columns<Placed...>::Repeats)
    {
        return m_columns.rows();
    }

    /** \returns \f$ H x \f$ for \a x, a vector in the input space \a in, into the output space
     * \a out, with \f$ H \f$ over the input as its Jacobian
     */
    template <typename In, typename Out, typename X>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x) const
    {
        static_assert(
            blocksAddValues<In>,
            "a linear measurement model reads blocks with no plus() of their own; a block such as a unit quaternion "
            "has no linear measurement, so give measurementModel() a function"
        );
        using T = typename X::Scalar;
        constexpr int OutRows = Columns<Placed...>::Repeats ? Eigen::Dynamic : Columns<Placed...>::Rows;
        constexpr int OutMaxRows =
            Columns<Placed...>::Repeats ? maxValueDimensionAtCompileTime<Out> : Columns<Placed...>::Rows;
        using Value = Eigen::Matrix<T, OutRows, 1, Eigen::ColMajor, OutMaxRows, 1>;
        using Jacobian =
            MatrixOf<Scalar, OutRows, valueDimensionAtCompileTime<In>, OutMaxRows, maxValueDimensionAtCompileTime<In>>;
        Eigen::Index const rows = m_columns.rows();
        Eigen::Index const count = [&] {
            if constexpr (Columns<Placed...>::Repeats)
            {
                Eigen::Index const repetitions = valueDimension(out) / rows;
                EMRYS_FILTER_ASSERT(
                    repetitions * rows == valueDimension(out)
                    && "a linear model's matrices have a row per coordinate of one repetition of the measurement"
                );
                return repetitions;
            }
            else
            {
                return Eigen::Index{1};
            }
        }();
        Jacobian jacobian = Jacobian::Zero(count * rows, valueDimension(in));
        Eigen::Index column = 0;
        std::apply(
            [&](auto const &... placed) {
                (place(placed, in, count, column, jacobian), ...);
            },
            m_columns.placed()
        );
        EMRYS_FILTER_ASSERT(
            column == valueDimension(in)
            && "a linear model's matrices must have a column per coordinate of the blocks they read"
        );
        Value value = jacobian.template cast<T>() * x;
        return WithJacobian<Value, Jacobian>{std::move(value), std::move(jacobian)};
    }

private:

    /** Writes the columns of \a placed, a matrix on a block of the input \a in, into \a jacobian
     * from column \a column on, over \a count repetitions of the measurement, and moves \a column
     * past them.
     */
    template <typename P, typename In, typename Jacobian>
    static void
    place(P const & placed, In const & in, Eigen::Index count, Eigen::Index & column, Jacobian & jacobian)
    {
        using C = ColumnsOf<P>;
        using Name = PathTag<typename C::Path>;
        auto const & matrix = placed.value;
        Eigen::Index const rows = matrix.rows();
        Eigen::Index const width = valueDimension(pathSubspace<Name>(in));
        if constexpr (C::Each)
        {
            using Block = PathSubspaceType<In, Name>;
            static_assert(
                Block::FixedValueDimension == 0 && Block::ValueDimensionStep > 0,
                "a model's onEach<Path>() names a repeated block of the state"
            );
            constexpr int Cols = std::remove_cvref_t<decltype(matrix)>::ColsAtCompileTime;
            static_assert(
                Cols == Eigen::Dynamic || Cols == Block::ValueDimensionStep,
                "a model's onEach<Path>() matrix has a column per coordinate of one repetition of the block"
            );
            EMRYS_FILTER_ASSERT(
                matrix.cols() * count == width
                && "a model's onEach<Path>() blocks have the measurement's count of repetitions"
            );
            for (Eigen::Index i = 0; i < count; ++i)
            {
                jacobian.block(i * rows, column + i * matrix.cols(), rows, matrix.cols()) = matrix;
            }
        }
        else
        {
            EMRYS_FILTER_ASSERT(
                matrix.cols() == width
                && "a linear model's matrix must have a column per coordinate of the block it reads"
            );
            for (Eigen::Index i = 0; i < count; ++i)
            {
                jacobian.block(i * rows, column, rows, matrix.cols()) = matrix;
            }
        }
        column += width;
    }

    Columns<Placed...> m_columns;  ///< \f$ H \f$, by its columns
};

/** Whether any of \a Policies is of the kind \a K. */
template <PolicyKind K, typename... Policies>
inline constexpr bool hasPolicy = ((policyKind<std::remove_cvref_t<Policies>> == K) || ...);

/** \returns the reads() policy of the paths \a Paths, a TagPaths: what a Columns reads */
template <typename... Paths>
[[nodiscard]] auto
readsOf(TagPaths<Paths...> *)
{
    return reads(path<Paths>...);
}

/** Checks the policies \a Set of a measurement model: no continuousNoise(), which is a process
 * model's, and no affineIn() without reads(), since a whole-state model's input has no names
 * (decisions.md [whole-state-models]).
 *
 * \returns true, when it compiles
 */
template <typename Set>
[[nodiscard]] consteval bool
checkMeasurementPolicies()
{
    static_assert(
        !Set::template has<PolicyKind::continuousNoise>,
        "continuousNoise() is a process model's; a measurement model's noise is additiveNoise()"
    );
    static_assert(
        !Set::template has<PolicyKind::affineIn> || Set::template has<PolicyKind::reads>,
        "affineIn() names items of reads(), and a model without reads() reads the whole state under no names"
    );
    return true;
}

/** The number of parameters of \a F's call, for a function pointer or a function object whose
 * call operator is not a template and not overloaded, or -1 where it is: such a count is read
 * from the type alone, with nothing instantiated, so a generic function is never probed with a
 * scalar it may not take.
 */
template <typename F>
struct CallArityOf
{
    static constexpr int value = -1;  ///< unknown
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename F>
    requires requires { &F::operator(); }
struct CallArityOf<F> : CallArityOf<decltype(&F::operator())>
{};

// Each signature is spelled with and without noexcept, since MSVC does not deduce a noexcept(B)
// in a partial specialization (C2764).
template <typename... A>
struct ArityOfParameters
{
    static constexpr int value = sizeof...(A);
};

template <typename R, typename... A>
struct CallArityOf<R (*)(A...)> : ArityOfParameters<A...>
{};

template <typename R, typename... A>
struct CallArityOf<R (*)(A...) noexcept> : ArityOfParameters<A...>
{};

template <typename R, typename C, typename... A>
struct CallArityOf<R (C::*)(A...)> : ArityOfParameters<A...>
{};

template <typename R, typename C, typename... A>
struct CallArityOf<R (C::*)(A...) noexcept> : ArityOfParameters<A...>
{};

template <typename R, typename C, typename... A>
struct CallArityOf<R (C::*)(A...) const> : ArityOfParameters<A...>
{};

template <typename R, typename C, typename... A>
struct CallArityOf<R (C::*)(A...) const noexcept> : ArityOfParameters<A...>
{};

template <typename R, typename C, typename... A>
struct CallArityOf<R (C::*)(A...) const &> : ArityOfParameters<A...>
{};

template <typename R, typename C, typename... A>
struct CallArityOf<R (C::*)(A...) const & noexcept> : ArityOfParameters<A...>
{};

/// \endcond

/** The output space a measurement model of the function \a F and the policies \a Set has before it
 * meets a state, wrapped: the Composite of writes(), or the space a matrix's rows give; void where
 * it is outputSpace()'s, which takes the state's space, or the noise's, or the function's result.
 */
template <typename F, typename Set>
[[nodiscard]] consteval auto
knownOutputType()
{
    if constexpr (Set::template has<PolicyKind::outputSpace>)
    {
        return std::type_identity<void>{};
    }
    else if constexpr (Set::template has<PolicyKind::writes>)
    {
        return std::type_identity<typename Set::template Type<PolicyKind::writes>::OutputSpace>{};
    }
    else if constexpr (requires { typename F::OutputSpaceType; })
    {
        return std::type_identity<typename F::OutputSpaceType>{};
    }
    else
    {
        return std::type_identity<void>{};
    }
}

/** The difference coordinates of \a S at compile time, or Eigen::Dynamic if it is resizable. */
template <typename S>
inline constexpr int differencesAtCompileTime = valueDimensionAtCompileTime<DifferenceSpace<S>>;

/** \returns the rows of \a X, a part of additiveNoise() as held, at compile time: a covariance's
 * rows or a distribution's difference coordinates, or Eigen::Dynamic where only a run time knows
 */
template <typename X>
[[nodiscard]] consteval int
noiseRowsAtCompileTime()
{
    if constexpr (MatrixExpression<X>)
    {
        return X::RowsAtCompileTime;
    }
    else if constexpr (requires { typename X::SpaceType; })
    {
        return differencesAtCompileTime<typename X::SpaceType>;
    }
    else
    {
        return Eigen::Dynamic;
    }
}

/** \returns whether noise of \a rows rows covers \a coordinates difference coordinates, either
 * Eigen::Dynamic where only a run time knows
 */
[[nodiscard]] consteval bool
noiseRowsFit(int rows, int coordinates)
{
    return rows == Eigen::Dynamic || coordinates == Eigen::Dynamic || rows == coordinates;
}

/** \returns whether the unplaced part \a X of additiveNoise() covers the output space \a Out */
template <typename Out, typename X>
[[nodiscard]] consteval bool
noisePartFits(X const *)
{
    return noiseRowsFit(noiseRowsAtCompileTime<X>(), differencesAtCompileTime<Out>);
}

/** \returns whether \a X, placed on the block of \a Out that \a Path names, covers the block */
template <typename Out, typename X, typename Path>
[[nodiscard]] consteval bool
noisePartFits(OnPath<X, Path> const *)
{
    using Block = typename EachBlockOf<Out, Path>::type;
    return noiseRowsFit(noiseRowsAtCompileTime<std::remove_cvref_t<X>>(), differencesAtCompileTime<Block>);
}

/** \returns whether \a X, placed on each repetition of the block of \a Out that \a Path names,
 * covers one repetition
 */
template <typename Out, typename X, typename Path>
[[nodiscard]] consteval bool
noisePartFits(OnEach<X, Path> const *)
{
    using Block = typename EachBlockOf<Out, Path>::type;
    if constexpr (requires { typename Block::BlockType; })
    {
        return noiseRowsFit(
            noiseRowsAtCompileTime<std::remove_cvref_t<X>>(),
            differencesAtCompileTime<typename Block::BlockType>
        );
    }
    else
    {
        return true;
    }
}

/** \returns whether every part of the additive noise \a Noise covers the output space \a Out or
 * the block of it the part is placed on
 */
template <typename Out, typename... Parts>
[[nodiscard]] consteval bool
noiseFits(AdditiveNoise<Parts...> const *)
{
    return (noisePartFits<Out>(static_cast<std::remove_cvref_t<Parts> const *>(nullptr)) && ...);
}

/** Checks what a measurement model of the function \a F and the policies \a Set can be checked for
 * before it meets a state: that a function whose parameters its type gives takes one of the three
 * forms, and that additive noise has a row for each difference coordinate of an output space known
 * without the state, the Composite of writes() or the space of a matrix's rows, a part placed with
 * on() or onEach() matching its block or one repetition of it. Neither check calls the function. A
 * generic function is checked where an update calls it, since only the state gives its input's
 * scalar: the one a check here could guess is \c double, and calling the function in it would
 * instantiate a body that may take only \c float or only dual numbers, so a valid model would fail
 * to compile. observe() knows the method but not the state, so it could check no more.
 *
 * \returns true, when it compiles
 */
template <typename F, typename Set>
[[nodiscard]] consteval bool
checkMeasurementFunction()
{
    constexpr int Arity = CallArityOf<F>::value;
    static_assert(
        Arity < 0 || (Arity >= 1 && Arity <= 3),
        "a measurement model's function takes one to three arguments, h(x), h(in, x), or h(in, out, x), with in the "
        "input space, out the output space, and x a vector in the input space"
    );
    static_assert(
        !Set::template has<PolicyKind::sample>,
        "a measurement model with a function draws its observation through the function and its noise, so sample(g) "
        "is for a model known by its likelihood alone"
    );
    using Known = typename decltype(knownOutputType<F, Set>())::type;
    if constexpr (!std::is_void_v<Known> && Set::template has<PolicyKind::additiveNoise>)
    {
        static_assert(
            noiseFits<Known>(static_cast<typename Set::template Type<PolicyKind::additiveNoise> const *>(nullptr)),
            "a measurement model's additive noise has a row for each difference coordinate of the output space, from "
            "writes() or the matrix's rows, or of the block it is placed on"
        );
    }
    return true;
}

}  // namespace internal

/** \ingroup Models
 * \returns the measurement model whose function is \a function, with \a policies, each a call such
 * as reads(), writes(), affineIn(), additiveNoise(), logLikelihood(), or outputSpace(), in any order
 * and each at most once (decisions.md [model-policies]). The function takes <tt>h(x)</tt>,
 * <tt>h(in, x)</tt>, or <tt>h(in, out, x)</tt>, told apart by the count of its arguments. A model
 * reads what reads() names; a model without reads() reads the whole state of the filter it is given
 * to, its input space the state's (decisions.md [whole-state-models]), and is not composed or placed.
 * The output space is outputSpace()'s, else the Composite of writes(), else the space of the one
 * unplaced additive noise, else a Linear sized by what the function returns (decisions.md [sensors]).
 *
 * \code
 * // A position fix on a Linear<4> state of position and velocity, as a whole-state model.
 * auto const fix = measurementModel([](auto const & x) { return x.template head<2>().eval(); },
 *                                   additiveNoise(R));
 * \endcode
 *
 * \code
 * auto const range = measurementModel(
 *     []<typename T>(auto const & in, auto const & out, auto const & x) {
 *       auto const d = minus<Position>(block<Position>(in, x), block<SensorPosition>(in, x));
 *       return T{(1 + block<ScaleNoise>(in, x)(0)) * d.norm()};
 *     },
 *     reads(on<SensorPosition>(path<Ownship, Position>), path<Target, Position>, path<ScaleNoise>),
 *     writes(path<Range>),
 *     affineIn(path<ScaleNoise>),
 *     additiveNoise(on<Range>(r)));
 * \endcode
 */
template <typename F, typename... Policies>
    requires(!MatrixExpression<F> && !internal::ModelPolicy<F> && !internal::isColumns<std::remove_cvref_t<F>>)
[[nodiscard]] auto
measurementModel(F && function, Policies &&... policies)
{
    using Set = internal::PolicySet<std::remove_cvref_t<Policies>...>;
    static_assert(internal::checkMeasurementPolicies<Set>());
    static_assert(internal::checkMeasurementFunction<std::decay_t<F>, Set>());
    return MeasurementFunction<std::decay_t<F>, std::remove_cvref_t<Policies>...>{
        std::forward<F>(function),
        std::forward<Policies>(policies)...
    };
}

/** \ingroup Models
 * \returns the linear measurement model \f$ z = H x \boxplus v \f$ of the matrix \a matrix over the
 * blocks that reads() names, stacked in that order as \f$ x \f$, or over the whole state without
 * reads() (decisions.md [linear-sensors], [whole-state-models]), with \a policies as
 * measurementModel(function, policies...) takes them. The blocks must add their values, as Linear
 * and Periodic ones do, and the model's update is the Kalman filter's exact step, with nothing
 * differentiated.
 *
 * \code
 * auto const gps = measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(r));
 * auto const fix = measurementModel(H, additiveNoise(R));  // H is 2 by 4, R 2 by 2
 * \endcode
 */
template <MatrixExpression Matrix, typename... Policies>
[[nodiscard]] auto
measurementModel(Matrix && matrix, Policies &&... policies)
{
    using Plain = internal::PlainResult<std::remove_cvref_t<Matrix>>;
    using Set = internal::PolicySet<std::remove_cvref_t<Policies>...>;
    static_assert(internal::checkMeasurementPolicies<Set>());
    static_assert(internal::checkMeasurementFunction<internal::LinearFunction<Plain>, Set>());
    return MeasurementFunction<internal::LinearFunction<Plain>, std::remove_cvref_t<Policies>...>{
        internal::LinearFunction<Plain>{Plain{std::forward<Matrix>(matrix)}},
        std::forward<Policies>(policies)...
    };
}

/** \ingroup Models
 * \returns the linear measurement model with \f$ H \f$ given by \a columns, a matrix per block read
 * in the order given, which is what the model reads, so it takes no reads() (decisions.md
 * [linear-sensors]). If the columns repeat, with onEach(), and no outputSpace() is given, the
 * measurement space is a Repeated block of Linear coordinates with the block's count at each update.
 */
template <typename... Placed, typename... Policies>
[[nodiscard]] auto
measurementModel(Columns<Placed...> columns, Policies &&... policies)
{
    using ColumnsType = Columns<Placed...>;
    static_assert(
        !internal::hasPolicy<internal::PolicyKind::reads, Policies...>,
        "a model given Columns reads the blocks its columns are placed on, so it takes no reads()"
    );
    auto const read = internal::readsOf(static_cast<typename ColumnsType::Paths *>(nullptr));
    using Function = internal::ColumnsFunction<Placed...>;
    if constexpr (ColumnsType::Repeats && !internal::hasPolicy<internal::PolicyKind::outputSpace, Policies...>)
    {
        static_assert(
            ColumnsType::Rows != Eigen::Dynamic,
            "a model whose columns repeat infers its measurement space from its matrices' rows, so fix them at "
            "compile time, or give it outputSpace() with a function such as CountFrom"
        );
        auto const space =
            outputSpace(internal::LinearPerRepetition<typename ColumnsType::FirstEachPath, ColumnsType::Rows>{});
        using Set = internal::PolicySet<decltype(read), decltype(space), std::remove_cvref_t<Policies>...>;
        static_assert(internal::checkMeasurementFunction<Function, Set>());
        return MeasurementFunction<Function, decltype(read), decltype(space), std::remove_cvref_t<Policies>...>{
            Function{std::move(columns)},
            read,
            space,
            std::forward<Policies>(policies)...
        };
    }
    else
    {
        using Set = internal::PolicySet<decltype(read), std::remove_cvref_t<Policies>...>;
        static_assert(internal::checkMeasurementFunction<Function, Set>());
        return MeasurementFunction<Function, decltype(read), std::remove_cvref_t<Policies>...>{
            Function{std::move(columns)},
            read,
            std::forward<Policies>(policies)...
        };
    }
}

/** \ingroup Models
 * \returns the measurement model known by its likelihood alone, <tt>measurementModel(reads(...),
 * logLikelihood(g))</tt>, for the filters that weight points (decisions.md
 * [likelihood-measurements]): it has no function, so no Kalman update, and its error distribution,
 * such as a quantization convolved with %Gaussian noise, is in the likelihood alone. Given
 * sample(g) too, a draw of the observation at a state, simulate() draws its observations.
 */
template <internal::ModelPolicy... Policies>
[[nodiscard]] auto
measurementModel(Policies &&... policies)
{
    using Set = internal::PolicySet<std::remove_cvref_t<Policies>...>;
    static_assert(internal::checkMeasurementPolicies<Set>());
    static_assert(
        Set::template has<internal::PolicyKind::logLikelihood>,
        "a measurement model without a function is known by its likelihood alone, so it takes logLikelihood(g)"
    );
    static_assert(
        !Set::template has<internal::PolicyKind::additiveNoise>,
        "a measurement model known by its likelihood alone has no function to add noise to"
    );
    return MeasurementFunction<internal::NoFunction, std::remove_cvref_t<Policies>...>{
        internal::NoFunction{},
        std::forward<Policies>(policies)...
    };
}

namespace internal
{

/** What an observation holds in place of a method, where a model is read without one: by
 * PosteriorCramerRaoBound, which linearizes, and by simulate(), which draws.
 */
struct NoMethod
{};

/** What an observation holds in place of an observed value, where a model is read without one. */
struct NoObservation
{};

/** The kinds of argument observe() and predict() take after their fixed ones. */
enum class ArgumentKind
{
    uncertain,  ///< an uncertain input, a distribution over a tagged space
    sure,       ///< a sure input, sure<Tag>(value)
    form,       ///< a covariance update form, updates::...
    repair,     ///< a repair of the innovation covariance, roots::ridge()
    consider,   ///< the blocks a Kalman update considers, consider()
    reference,  ///< a linearizedAbout() reference, which a prediction's step linearizes about
    other       ///< none of these
};

/** \returns the uncertain input \a argument holds: itself */
template <typename T>
[[nodiscard]] T const &
inputOf(T const & argument)
{
    return argument;
}

/** The kind of an argument of type \a T. */
template <typename T>
inline constexpr ArgumentKind argumentKind = [] {
    using Plain = std::remove_cvref_t<T>;
    if constexpr (isSure<Plain>)
    {
        return ArgumentKind::sure;
    }
    else if constexpr (UncertainInput<Plain>)
    {
        return ArgumentKind::uncertain;
    }
    else if constexpr (isLinearizedAbout<Plain>)
    {
        return ArgumentKind::reference;
    }
    else if constexpr (CovarianceUpdate<Plain>)
    {
        return ArgumentKind::form;
    }
    else if constexpr (Repair<Plain>)
    {
        return ArgumentKind::repair;
    }
    else if constexpr (isConsider<Plain>)
    {
        return ArgumentKind::consider;
    }
    else
    {
        return ArgumentKind::other;
    }
}();

/** The tag of the input argument of type \a T: the space of an uncertain or a sure input, void for
 * any other argument.
 */
template <typename T>
struct InputTagOf
{
    using type = void;  ///< not an input
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename T>
    requires(argumentKind<T> == ArgumentKind::uncertain || argumentKind<T> == ArgumentKind::sure)
struct InputTagOf<T>
{
    using type =
        typename std::remove_cvref_t<decltype(inputOf(std::declval<std::remove_cvref_t<T> const &>()))>::SpaceType;
};

/// \endcond

/** The tag of the input argument of type \a T (InputTagOf). */
template <typename T>
using InputTag = typename InputTagOf<T>::type;

/** Checks the arguments \a Args of observe() or predict() after their fixed ones: each an uncertain
 * input over a tagged space, a sure input, a covariance update form, a repair, a consider(), or a
 * linearizedAbout() reference; one tag at most once; and one form, one repair, one consider(), and
 * one reference at most.
 *
 * \returns true, when it compiles
 */
template <typename... Args>
[[nodiscard]] consteval bool
checkArguments()
{
    static_assert(
        ((argumentKind<Args> != ArgumentKind::other) && ...),
        "an argument after the observation is an uncertain input over a tagged space, sure<Tag>(value), a covariance "
        "update form such as updates::joseph(), a repair of the innovation covariance such as roots::ridge(), or the "
        "blocks to consider, consider(path<Bias>); after a prediction's step, an uncertain or a sure input, or a "
        "linearizedAbout() reference first"
    );
    static_assert(
        (std::size_t{argumentKind<Args> == ArgumentKind::reference} + ... + 0) <= 1,
        "give the linearizedAbout() reference once"
    );
    static_assert(
        ((argumentKind<Args> != ArgumentKind::uncertain || !IsUntagged<InputTag<Args>>) && ...),
        "an uncertain input is named by the tag of its space, so give it over a tag such as struct SpecificForce : "
        "Linear<3> {}"
    );
    static_assert(
        (std::size_t{argumentKind<Args> == ArgumentKind::form} + ... + 0) <= 1,
        "give the covariance update form once"
    );
    static_assert(
        (std::size_t{argumentKind<Args> == ArgumentKind::repair} + ... + 0) <= 1,
        "give the repair of the innovation covariance once"
    );
    static_assert(
        (std::size_t{argumentKind<Args> == ArgumentKind::consider} + ... + 0) <= 1,
        "give the blocks to consider in one consider()"
    );
    static_assert(
        distinctTypes<
            std::conditional_t<std::is_void_v<InputTag<Args>>, std::type_identity<Args>, InputTag<Args>>...>(),
        "two inputs of the call have one tag; an input's tag names it once"
    );
    return true;
}

/** The first of \a Args of the kind \a K, or \a Default if none is. */
template <ArgumentKind K, typename Default, typename... Args>
struct ArgumentOfKindOf
{
    using type = Default;  ///< none
};

/// \cond INTERNAL_SPECIALIZATIONS
template <ArgumentKind K, typename Default, typename First, typename... Rest>
struct ArgumentOfKindOf<K, Default, First, Rest...>
{
    using type = std::conditional_t<
        argumentKind<First> == K,
        std::remove_cvref_t<First>,
        typename ArgumentOfKindOf<K, Default, Rest...>::type>;
};

/// \endcond

/** The first of \a Args of the kind \a K, or \a Default. */
template <ArgumentKind K, typename Default, typename... Args>
using ArgumentOfKind = typename ArgumentOfKindOf<K, Default, Args...>::type;

/** \returns the first of \a args of the kind \a K, or \a fallback if none is */
template <ArgumentKind K, typename Default, typename... Args>
[[nodiscard]] auto const &
argumentOfKind(std::tuple<Args...> const & args, Default const & fallback)
{
    using Result = ArgumentOfKind<K, Default, Args...>;
    constexpr std::size_t position = [] {
        constexpr auto kinds = std::array<ArgumentKind, sizeof...(Args)>{argumentKind<Args>...};
        for (std::size_t i = 0; i < kinds.size(); ++i)
        {
            if (kinds[i] == K)
            {
                return i;
            }
        }
        return sizeof...(Args);
    }();
    if constexpr (position < sizeof...(Args))
    {
        return static_cast<Result const &>(std::get<position>(args));
    }
    else
    {
        return fallback;
    }
}

}  // namespace internal

/** \ingroup Models
 * \brief An observation of a measurement model: the observed value, the model, the method that
 * pushes the state through it, and the inputs of the call, as observe() makes it (decisions.md
 * [templated-measurements], [method-per-measurement], [model-policies]). What a filter's update()
 * takes (see Measurement).
 *
 * \tparam Method       the transform method, or internal::NoMethod
 * \tparam Model        the model, a MeasurementModel, or a const reference to one that outlives the
 *                      observation
 * \tparam Observation  the observed value's type, a plain vector, or internal::NoObservation
 * \tparam Args         the call's arguments after the observation: uncertain and sure inputs, the
 *                      covariance update form, and the repair, each held by value
 *
 * \sa observe(), update()
 */
template <typename Method, typename Model, typename Observation, typename... Args>
class Observed
{
public:

    using MethodType = Method;                                                      ///< the method
    using ModelType = std::remove_cvref_t<Model>;                                   ///< the model's type
    using ObservationType = Observation;                                            ///< the observed value's type
    using ArgumentsType = std::tuple<Args...>;                                      ///< the arguments, as held
    static constexpr bool HasMethod = !std::is_same_v<Method, internal::NoMethod>;  ///< whether a method was given

    /// whether an observed value was given
    static constexpr bool HasObservation = !std::is_same_v<Observation, internal::NoObservation>;

    /** The covariance update form given, or updates::Standard. */
    using FormType = internal::ArgumentOfKind<internal::ArgumentKind::form, updates::Standard, Args...>;

    /** The repair of an innovation covariance that is not positive definite, or internal::NoRepair
     * (decisions.md [innovation-cholesky]).
     */
    using InnovationRepair = internal::ArgumentOfKind<internal::ArgumentKind::repair, internal::NoRepair, Args...>;

    /** The blocks the update considers rather than estimates, or Consider<> (decisions.md [consider-states]). */
    using ConsiderType = internal::ArgumentOfKind<internal::ArgumentKind::consider, Consider<>, Args...>;

    static_assert(internal::isModel<ModelType>, "an observation is of a model");
    static_assert(internal::checkArguments<Args...>());

    /** The observation \a observation of \a model by \a method, with the call's \a args. */
    template <typename M, typename Z>
    Observed(Method method, M && model, Z && observation, Args... args)
        : m_method{std::move(method)}
        , m_model(std::forward<M>(model))
        , m_observation{std::forward<Z>(observation)}
        , m_arguments{std::move(args)...}
    {
        if constexpr (HasObservation)
        {
            EMRYS_FILTER_ASSERT(m_observation.cols() == 1 && "an observation is a column vector");
        }
    }

    /** \returns the method */
    [[nodiscard]] Method const &
    method() const
    {
        return m_method;
    }

    /** \returns the model */
    [[nodiscard]] ModelType const &
    model() const
    {
        return m_model;
    }

    /** \returns the observed value \f$ z \f$ */
    [[nodiscard]] Observation const &
    observation() const
    {
        return m_observation;
    }

    /** \returns the call's arguments, as held */
    [[nodiscard]] ArgumentsType const &
    arguments() const
    {
        return m_arguments;
    }

    /** \returns the covariance update form: the one given, or updates::Standard */
    [[nodiscard]] FormType const &
    covarianceUpdate() const
    {
        return internal::argumentOfKind<internal::ArgumentKind::form>(m_arguments, m_standard);
    }

private:

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Method m_method;               ///< the method
    Model m_model;                                                ///< the model, or a reference to it
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Observation m_observation;     ///< the observed value
    ArgumentsType m_arguments;                                    ///< the arguments
    EMRYS_FILTER_NO_UNIQUE_ADDRESS updates::Standard m_standard;  ///< the default form
};

namespace internal
{

/** Whether \a T is an Observed. */
template <typename T>
inline constexpr bool isObserved = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Method, typename Model, typename Observation, typename... Args>
inline constexpr bool isObserved<Observed<Method, Model, Observation, Args...>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Models
 * \brief A measurement: an observed value with its model and the call's inputs, and the method a
 * Kalman update pushes the state through, as observe() makes it; what update() and every filter's
 * update() take (decisions.md [templated-measurements]). A filter that weights points needs no
 * method, and a Kalman update refuses a measurement without one at compile time.
 */
template <typename M>
concept Measurement = internal::isObserved<std::remove_cvref_t<M>> && std::remove_cvref_t<M>::HasObservation;

namespace internal
{

/** A measurement that the filters weighting points take: a Measurement, with or without a method. */
template <typename M>
concept WeighsPoints = Measurement<M>;

/** Whether the measurement \a M's model has a likelihood of its own, logLikelihood(g), which the
 * filters that weight points use whatever its noise says (decisions.md [likelihood-measurements]).
 */
template <typename M>
inline constexpr bool hasOwnLikelihood = std::remove_cvref_t<M>::ModelType::template has<PolicyKind::logLikelihood>;

/** Whether the measurement \a M repairs an innovation covariance that is not positive definite. */
template <typename M>
inline constexpr bool hasInnovationRepair =
    !std::is_same_v<typename std::remove_cvref_t<M>::InnovationRepair, NoRepair>;

/** Whether the measurement \a M considers a block of the state (decisions.md [consider-states]). */
template <typename M>
inline constexpr bool hasConsider = std::remove_cvref_t<M>::ConsiderType::Count > 0;

/** Whether \a F is the function of a model given its matrix, which every method takes exactly. */
template <typename F>
inline constexpr bool isLinearFunction = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Matrix>
inline constexpr bool isLinearFunction<LinearFunction<Matrix>>{true};

template <typename... Placed>
inline constexpr bool isLinearFunction<ColumnsFunction<Placed...>>{true};
/// \endcond

/** The method observe() gives a measurement it is given none for: Linearized for a model given its
 * matrix, whose update it makes exactly, and otherwise NoMethod (decisions.md [linear-sensors]).
 */
template <typename Model>
using MethodWithoutOne =
    std::conditional_t<isLinearFunction<typename Model::FunctionType>, methods::Linearized, NoMethod>;

}  // namespace internal

/** \ingroup Models
 * \returns the Measurement of \a observation, a value in the measurement space, by \a model,
 * pushed through by \a method, with the call's \a args after it in any order: an uncertain input
 * over a tagged space for each item of the model's reads() that the state lacks, or
 * <tt>sure<Tag>(value)</tt> for one known exactly; a covariance update form (updates::joseph()); and
 * a repair of an innovation covariance that is not positive definite (roots::ridge()). The model is
 * held by copy, or for <tt>std::cref(model)</tt> by reference to one that outlives the measurement
 * (decisions.md [method-per-measurement], [model-policies]).
 *
 * \code
 * filter.update(observe(methods::unscented(), range, z, GaussianParams<ScaleNoise>{zero, s}));
 * filter.update(observe(methods::linearized(), std::cref(gps), fix, updates::joseph()));
 * \endcode
 */
template <TransformMethod Method, typename M, MatrixExpression Z, typename... Args>
    requires MeasurementModel<std::unwrap_ref_decay_t<M>>
[[nodiscard]] auto
observe(Method method, M && model, Z && observation, Args &&... args)
{
    using Stored = std::unwrap_ref_decay_t<M>;
    using Observation = typename internal::CrtpDerived<Z>::PlainObject;
    return Observed<Method, Stored, Observation, std::unwrap_ref_decay_t<Args>...>{
        std::move(method),
        std::forward<M>(model),
        std::forward<Z>(observation),
        std::forward<Args>(args)...
    };
}

/** \ingroup Models
 * \returns the Measurement of \a observation by \a model with the call's \a args, as
 * observe(method, model, observation, args...) makes it, with no method given. A model given its
 * matrix, <tt>measurementModel(H, noise)</tt>, is linear, and its measurement is pushed through by
 * Linearized, which is exact for it, so a linear Kalman filter names no method. Any other model's
 * measurement has no method, for the filters that weight points by the likelihood, which push no
 * %Gaussian through the model, and a Kalman update refuses it at compile time (decisions.md
 * [linear-sensors]).
 *
 * \code
 * filter.update(observe(fix, z));        // fix = measurementModel(H, additiveNoise(R))
 * particles.update(observe(range, z));
 * \endcode
 */
template <typename M, MatrixExpression Z, typename... Args>
    requires MeasurementModel<std::unwrap_ref_decay_t<M>>
[[nodiscard]] auto
observe(M && model, Z && observation, Args &&... args)
{
    using Stored = std::unwrap_ref_decay_t<M>;
    using Observation = typename internal::CrtpDerived<Z>::PlainObject;
    using Method = internal::MethodWithoutOne<Stored>;
    return Observed<Method, Stored, Observation, std::unwrap_ref_decay_t<Args>...>{
        Method{},
        std::forward<M>(model),
        std::forward<Z>(observation),
        std::forward<Args>(args)...
    };
}

namespace internal
{

/** \returns \a model, a measurement or a process model, with the call's \a args and no method or
 * observed value, as the bound, a simulation, and a prediction read a model: an Observed with
 * NoMethod and NoObservation
 */
template <typename M, typename... Args>
    requires isModel<std::remove_cvref_t<std::unwrap_ref_decay_t<M>>>
[[nodiscard]] auto
modelWith(M && model, Args &&... args)
{
    using Stored = std::unwrap_ref_decay_t<M>;
    return Observed<NoMethod, Stored, NoObservation, std::unwrap_ref_decay_t<Args>...>{
        NoMethod{},
        std::forward<M>(model),
        NoObservation{},
        std::forward<Args>(args)...
    };
}

}  // namespace internal

}  // namespace Emrys
