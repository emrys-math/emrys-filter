#pragma once

/** \file
 * Batched functions (decisions.md [batched-functions]): batched(), the mark that a function takes every point at
 * once, as the columns of a matrix, and writes one output; the marks of the Jacobians it writes,
 * jacobians::perColumn() and jacobians::shared(); and BatchedOutput, the output the library provides, sized once
 * for the batch and for the blocks whose Jacobians it uses. Read Policies.h next.
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <unsupported/Eigen/AutoDiff>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Models
 * \brief The marks of the Jacobians a batched function writes beside its values (decisions.md
 * [batched-functions], [policy-factories]): perColumn() and shared(), given as batched(f, mark).
 */
namespace jacobians
{

/** \ingroup Models
 * \brief The mark that a batched function writes one Jacobian per column, by block of its input:
 * <tt>Y.jacobian<Name>(j)</tt>, made by perColumn().
 */
struct PerColumn
{};

/** \ingroup Models
 * \brief The mark that a batched function writes one Jacobian for every column, by block of its
 * input: <tt>Y.jacobian<Name>()</tt>, for a function linear in its input, made by shared().
 */
struct Shared
{};

/** \ingroup Models
 * \returns the mark that a batched function writes one Jacobian per column (see PerColumn)
 */
[[nodiscard]] constexpr PerColumn
perColumn()
{
    return {};
}

/** \ingroup Models
 * \returns the mark that a batched function writes one Jacobian for every column, which is the
 * user's contract that the function is linear in its input (see Shared)
 */
[[nodiscard]] constexpr Shared
shared()
{
    return {};
}

}  // namespace jacobians

namespace internal
{

/** The mark of a batched function that writes no Jacobian: batched(f). */
struct NoJacobians
{};

/** Whether \a T is a mark of the Jacobians a batched function writes, NoJacobians among them. */
template <typename T>
inline constexpr bool isBatchedMark = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <>
inline constexpr bool isBatchedMark<NoJacobians>{true};

template <>
inline constexpr bool isBatchedMark<jacobians::PerColumn>{true};

template <>
inline constexpr bool isBatchedMark<jacobians::Shared>{true};

/// \endcond

}  // namespace internal

/** \ingroup Models
 * Whether \a T is a mark of the Jacobians a batched function writes, made by a factory in
 * Emrys::jacobians (decisions.md [batched-functions]).
 */
template <typename T>
concept JacobianMark =
    internal::isBatchedMark<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, internal::NoJacobians>;

/** \ingroup Models
 * \brief A function that takes every point at once, as batched() marks it (decisions.md
 * [batched-functions]): it receives the points as the columns of a matrix \c X and writes into
 * one output \c Y, a BatchedOutput the library provides, last.
 *
 * \tparam F     the function
 * \tparam Mark  the Jacobians it writes: internal::NoJacobians, jacobians::PerColumn, or
 *               jacobians::Shared
 *
 * \sa batched()
 */
template <typename F, typename Mark = internal::NoJacobians>
struct Batched
{
    static_assert(internal::isBatchedMark<Mark>);

    /** the function */
    using FunctionType = F;
    /** the Jacobians it writes */
    using MarkType = Mark;

    /// whether the function writes Jacobians
    static constexpr bool GivesJacobians = !std::is_same_v<Mark, internal::NoJacobians>;

    /// whether it writes one Jacobian per column
    static constexpr bool PerColumn = std::is_same_v<Mark, jacobians::PerColumn>;

    F function;  ///< the function

    /** \returns the function at one point, \a x, a vector in its input space \a in, as a batch of one
     * column, with \a rest, the step of a process model or nothing, before its output: the value, a
     * vector in the output space \a out, and with a mark that gives Jacobians a WithJacobian of it and
     * the Jacobian over the input's differences, every block wanted. This is how Linearized and the
     * iterated updates call a batched function, and how a function differentiated in the autodiff
     * scalar is called (decisions.md [batched-functions]).
     */
    template <typename In, typename Out, MatrixExpression X, typename... Rest>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, Rest const &... rest) const;
};

namespace internal
{

/** Whether \a T is a Batched. */
template <typename T>
inline constexpr bool isBatched = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename F, typename Mark>
inline constexpr bool isBatched<Batched<F, Mark>>{true};

/// \endcond

/** Whether the function \a F takes a batch of points in one call (decisions.md [batched-functions]):
 * a Batched, or the library's function of models placed or composed (Placement.h, Composition),
 * which says so with \c TakesBatches where a model it holds is batched, and takes the batch as a
 * Batched's function does, its output last.
 */
template <typename F>
inline constexpr bool takesBatches = isBatched<F> || requires { requires F::TakesBatches; };

/** The mark of the Jacobians the function \a F that takes batches writes: a Batched's own, or the
 * \c MarkType of a function of placed or composed models.
 */
template <typename F>
struct BatchMarkOf
{
    using type = typename F::MarkType;  ///< the function's mark
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename F, typename Mark>
struct BatchMarkOf<Batched<F, Mark>>
{
    using type = Mark;  ///< the mark batched() was given
};

/// \endcond

/** The mark of the Jacobians the function \a F that takes batches writes (BatchMarkOf). */
template <typename F>
using BatchMark = typename BatchMarkOf<F>::type;

/** The mark of the Jacobians \a F writes in a batch where it takes batches, and NoJacobians
 * otherwise: the mark of a function that places or composes \a F.
 */
template <typename F, bool = takesBatches<F>>
struct BatchMarkOrNoneOf
{
    using type = NoJacobians;  ///< none
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename F>
struct BatchMarkOrNoneOf<F, true>
{
    using type = BatchMark<F>;  ///< its mark
};

/// \endcond

/** The mark of the Jacobians \a F writes in a batch, or NoJacobians (BatchMarkOrNoneOf). */
template <typename F>
using BatchMarkOrNone = typename BatchMarkOrNoneOf<F>::type;

/** \returns what is called with a batch of the function \a f that takes batches: a Batched's
 * function, or \a f itself
 */
template <typename F>
[[nodiscard]] auto const &
batchBody(F const & f)
{
    if constexpr (isBatched<F>)
    {
        return f.function;
    }
    else
    {
        return f;
    }
}

/** The type batchBody() gives for \a F. */
template <typename F>
using BatchBody = std::remove_cvref_t<decltype(batchBody(std::declval<F const &>()))>;

}  // namespace internal

/** \ingroup Models
 * \returns \a function marked as taking every point at once, with no Jacobian (see Batched): a
 * measurement model's <tt>h(in, out, X, Y)</tt>, a process model's <tt>f(in, out, X, dt, Y)</tt>, or
 * a log-likelihood's <tt>g(in, X, z, Y)</tt>, which writes one log-likelihood per column. Column
 * \f$ j \f$ of what it writes depends only on column \f$ j \f$ of \c X, which is the user's
 * contract. A method that needs a Jacobian differentiates the function in the autodiff scalar.
 *
 * \code
 * auto const range = Emrys::logLikelihood(Emrys::batched([](auto const & in, auto const & X, auto const & z, auto & Y)
 * { Y.values() = -0.5 * (block<Position>(in, X).colwise().norm().array() - z(0)).square().matrix() / variance;
 * }));
 * \endcode
 */
template <typename F>
    requires(!internal::isBatched<std::remove_cvref_t<F>> && !MatrixExpression<F>)
[[nodiscard]] Batched<std::remove_cvref_t<F>>
batched(F && function)
{
    return {std::forward<F>(function)};
}

/** \ingroup Models
 * \returns \a function marked as taking every point at once and writing the Jacobians \a mark
 * says, jacobians::perColumn() or jacobians::shared() (see Batched)
 */
template <typename F, JacobianMark Mark>
    requires(!internal::isBatched<std::remove_cvref_t<F>> && !MatrixExpression<F>)
[[nodiscard]] Batched<std::remove_cvref_t<F>, std::remove_cvref_t<Mark>>
batched(F && function, Mark)
{
    return {std::forward<F>(function)};
}

namespace internal
{

/** The position of \a Name among the blocks of the tuple type \a Blocks, or its size if it is not one. */
template <typename Name, typename Blocks>
inline constexpr std::size_t blockPosition = 0;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Name, typename... Blocks>
inline constexpr std::size_t blockPosition<Name, std::tuple<Blocks...>> = [] {
    constexpr std::array<bool, sizeof...(Blocks) + 1> same{std::is_same_v<Name, Blocks>..., true};
    std::size_t k = 0;
    while (!same[k])
    {
        ++k;
    }
    return k;
}();

/// \endcond

/** The blocks of a batched function's input space \a In, as a tuple type: a Composite's blocks, or
 * the space itself as one block, as a whole-state model's input is (decisions.md [whole-state-models]).
 */
template <typename In>
using InputBlocks = typename decltype([] {
    if constexpr (IsComposite<In>)
    {
        return std::type_identity<CompositeBlocks<In>>{};
    }
    else
    {
        return std::type_identity<std::tuple<In>>{};
    }
}())::type;

/** \returns the difference rows of the block \a Name of \a in, a batched function's input space:
 * its rows in a Composite, or every row of a space that is its own one block
 */
template <typename Name, typename In>
[[nodiscard]] BlockRows
inputBlockRows(In const & in)
{
    if constexpr (IsComposite<In>)
    {
        return differenceBlockRows<Name>(in);
    }
    else
    {
        return BlockRows{0, differenceDimension(in)};
    }
}

}  // namespace internal

/** \ingroup Models
 * \brief The output of a batched function (decisions.md [batched-functions]): its values, a column per
 * point, and the Jacobians the library uses, by block of the function's input, in storage the library
 * provides and sizes once before the call.
 *
 * The function writes values() whole. Where its mark gives Jacobians, it writes
 * <tt>jacobian<Name>(j)</tt> (jacobians::perColumn()) or <tt>jacobian<Name>()</tt>
 * (jacobians::shared()) for each block \c Name of its input for which <tt>wantsJacobian<Name>()</tt>
 * is true on this call: the columns of the Jacobian for the block that reads() names \c Name. The
 * storage holds only those blocks, so a block not wanted is not written. A function of a model
 * without tags writes <tt>jacobian(j)</tt> or <tt>jacobian()</tt>, the Jacobian over its whole input,
 * when wantsJacobians() says every block is wanted (decisions.md [whole-state-models]).
 *
 * \tparam In       the function's input space, a Composite of the names reads() gives, or the
 *                  state's space for a whole-state model, which is its own one block
 * \tparam Out      its output space: the values have a row per value coordinate of it, and the
 *                  Jacobians a row per difference coordinate
 * \tparam Scalar_  the scalar the function writes
 * \tparam Mark     the Jacobians the function writes
 */
template <typename In, typename Out, typename Scalar_, typename Mark>
class BatchedOutput
{
    /** The blocks of the input, as a tuple type. */
    using Blocks = internal::InputBlocks<In>;

    static constexpr std::size_t BlockCount = std::tuple_size_v<Blocks>;  ///< how many

public:

    /** the scalar type */
    using Scalar = Scalar_;

    /** The values, a column per point. */
    using ValuesType = Eigen::Map<internal::MatrixOf<
        Scalar_,
        valueDimensionAtCompileTime<Out>,
        Eigen::Dynamic,
        maxValueDimensionAtCompileTime<Out>,
        Eigen::Dynamic>>;

    /** The Jacobians' storage: the blocks wanted side by side, for each column or once. */
    using JacobiansType = Eigen::Map<internal::MatrixOf<
        Scalar_,
        valueDimensionAtCompileTime<DifferenceSpace<Out>>,
        Eigen::Dynamic,
        maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
        Eigen::Dynamic>>;

    /** The output over the values \a values and the Jacobians' storage \a jacobians, of the
     * function's input \a in, with the Jacobian of block \a k wanted where \a wanted says. The
     * storage holds the wanted blocks' columns side by side, once per column of the values under
     * jacobians::perColumn() and once under jacobians::shared(). \a in outlives this.
     */
    BatchedOutput(
        ValuesType values,
        JacobiansType jacobians,
        In const & in,
        std::array<bool, BlockCount> const & wanted
    )
        : m_values{values}
        , m_jacobians{jacobians}
        , m_in{in}
    {
        Eigen::Index offset = 0;
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            ((m_offsets[K] = wanted[K] ? offset : -1,
              offset += wanted[K] ? internal::inputBlockRows<std::tuple_element_t<K, Blocks>>(in).size : 0),
             ...);
        }(std::make_index_sequence<BlockCount>{});
        m_wantedColumns = offset;
        EMRYS_FILTER_ASSERT(
            m_jacobians.cols() == (std::is_same_v<Mark, jacobians::PerColumn> ? values.cols() : 1) * m_wantedColumns
            && "the Jacobians' storage holds the wanted blocks' columns for each column, or once"
        );
    }

    /** \returns the values, a column per point, for the function to write */
    [[nodiscard]] ValuesType &
    values()
    {
        return m_values;
    }

    /** \returns the values, a column per point */
    [[nodiscard]] ValuesType const &
    values() const
    {
        return m_values;
    }

    /** \returns whether the library uses the Jacobian of the block named \a Name on this call, which
     * the function then writes; always false for a function marked without Jacobians
     */
    template <typename Name>
    [[nodiscard]] bool
    wantsJacobian() const
    {
        checkName<Name>();
        return m_offsets[internal::blockPosition<Name, Blocks>] >= 0;
    }

    /** \returns whether the library uses the Jacobian of every block of the input on this call, the
     * whole of it, which a function of a model without tags writes with jacobian(j) or jacobian()
     */
    [[nodiscard]] bool
    wantsJacobians() const
    {
        return !std::is_same_v<Mark, internal::NoJacobians> && m_wantedColumns == differenceDimension(m_in);
    }

    /** \returns the Jacobian of column \a j over the whole input, writable, under jacobians::perColumn(),
     * when wantsJacobians() says every block is wanted: what a function of a model without tags writes
     */
    [[nodiscard]] auto
    jacobian(Eigen::Index j)
    {
        checkWhole();
        static_assert(
            !std::is_same_v<Mark, jacobians::Shared>,
            "under jacobians::shared(), a batched function writes its whole Jacobian once for every column, jacobian()"
        );
        EMRYS_FILTER_ASSERT(j >= 0 && j < m_values.cols() && "the column must be one of the batch");
        return wholeColumns(j * m_wantedColumns);
    }

    /** \returns the Jacobian of every column over the whole input, writable, under jacobians::shared(),
     * when wantsJacobians() says every block is wanted
     */
    [[nodiscard]] auto
    jacobian()
    {
        checkWhole();
        static_assert(
            !std::is_same_v<Mark, jacobians::PerColumn>,
            "under jacobians::perColumn(), a batched function writes its whole Jacobian once per column, jacobian(j)"
        );
        return wholeColumns(0);
    }

    /** \returns the Jacobian of column \a j in the block named \a Name, writable: its rows are the
     * output's differences and its columns the block's, under jacobians::perColumn(), for a block
     * wantsJacobian() says is wanted
     */
    template <typename Name>
    [[nodiscard]] auto
    jacobian(Eigen::Index j)
    {
        checkJacobian<Name>();
        static_assert(
            !std::is_same_v<Mark, jacobians::Shared>,
            "a batched function marked jacobians::shared() writes one Jacobian for every column, jacobian<Name>()"
        );
        EMRYS_FILTER_ASSERT(j >= 0 && j < m_values.cols() && "the column must be one of the batch");
        return columnsOf<Name>(j * m_wantedColumns);
    }

    /** \returns the Jacobian of every column in the block named \a Name, writable, under
     * jacobians::shared(), for a block wantsJacobian() says is wanted
     */
    template <typename Name>
    [[nodiscard]] auto
    jacobian()
    {
        checkJacobian<Name>();
        static_assert(
            !std::is_same_v<Mark, jacobians::PerColumn>,
            "a batched function marked jacobians::perColumn() writes one Jacobian per column, jacobian<Name>(j)"
        );
        return columnsOf<Name>(0);
    }

    /** \returns this output over the same storage for a function whose input space \a in and output
     * space \a Out2 name the same blocks in the same order and sizes, under names of their own: what
     * a placed model's function gives the model it places, which reads and writes the blocks by its
     * own names (decisions.md [model-placement]). \a in outlives the result.
     */
    template <typename Out2, typename In2>
    [[nodiscard]] BatchedOutput<In2, Out2, Scalar_, Mark>
    renamed(In2 const & in)
    {
        using Renamed = BatchedOutput<In2, Out2, Scalar_, Mark>;
        static_assert(std::tuple_size_v<internal::InputBlocks<In2>> == BlockCount);
        auto wanted = std::array<bool, BlockCount>{};
        for (std::size_t k = 0; k < BlockCount; ++k)
        {
            wanted[k] = m_offsets[k] >= 0;
        }
        return Renamed{
            typename Renamed::ValuesType{m_values.data(), m_values.rows(), m_values.cols()},
            typename Renamed::JacobiansType{m_jacobians.data(), m_jacobians.rows(), m_jacobians.cols()},
            in,
            wanted
        };
    }

private:

    /** A compile-time error unless \a Name names a block of the function's input. */
    template <typename Name>
    static consteval void
    checkName()
    {
        static_assert(
            internal::blockPosition<Name, Blocks> < BlockCount,
            "a batched function writes a Jacobian only for a block of its input, by the name reads() gives it"
        );
    }

    /** A compile-time error unless the function's mark gives Jacobians and \a Name names a block of
     * its input.
     */
    template <typename Name>
    static consteval void
    checkJacobian()
    {
        static_assert(
            !std::is_same_v<Mark, internal::NoJacobians>,
            "a batched function marked without Jacobians writes none: mark it batched(f, jacobians::perColumn()) "
            "or batched(f, jacobians::shared())"
        );
        checkName<Name>();
    }

    /** \returns the columns of the block named \a Name in the Jacobians' storage, from column
     * \a start on
     */
    template <typename Name>
    [[nodiscard]] auto
    columnsOf(Eigen::Index start)
    {
        Eigen::Index const offset = m_offsets[internal::blockPosition<Name, Blocks>];
        EMRYS_FILTER_ASSERT(offset >= 0 && "write only the Jacobians wantsJacobian() says are wanted");
        constexpr int Cols = valueDimensionAtCompileTime<DifferenceSpace<Name>>;
        return internal::middleColsOf<Cols>(m_jacobians, start + offset, internal::inputBlockRows<Name>(m_in).size);
    }

    /** A compile-time error unless the function's mark gives Jacobians. */
    static consteval void
    checkWhole()
    {
        static_assert(
            !std::is_same_v<Mark, internal::NoJacobians>,
            "a batched function marked without Jacobians writes none: mark it batched(f, jacobians::perColumn()) "
            "or batched(f, jacobians::shared())"
        );
    }

    /** \returns the columns of the whole input in the Jacobians' storage, from column \a start on */
    [[nodiscard]] auto
    wholeColumns(Eigen::Index start)
    {
        EMRYS_FILTER_ASSERT(
            m_wantedColumns == differenceDimension(m_in)
            && "write the whole Jacobian only when wantsJacobians() says every block is wanted"
        );
        constexpr int Cols = valueDimensionAtCompileTime<DifferenceSpace<In>>;
        return internal::middleColsOf<Cols>(m_jacobians, start, m_wantedColumns);
    }

    ValuesType m_values;                               ///< the values
    JacobiansType m_jacobians;                         ///< the wanted blocks' Jacobians
    In const & m_in;                                   ///< the function's input space
    std::array<Eigen::Index, BlockCount> m_offsets{};  ///< each block's first column among the wanted, or -1
    Eigen::Index m_wantedColumns = 0;                  ///< the columns of the wanted blocks, side by side
};

template <typename F, typename Mark>
template <typename In, typename Out, MatrixExpression X, typename... Rest>
auto
Batched<F, Mark>::operator()(In const & in, Out const & out, X const & x, Rest const &... rest) const
{
    static_assert(
        Space<Out>,
        "a batched function returns nothing to size its output by, so its model names the output with writes() or "
        "outputSpace()"
    );
    using T = typename X::Scalar;
    using Output = BatchedOutput<In, Out, T, Mark>;
    using Value = internal::MatrixOf<T, valueDimensionAtCompileTime<Out>, 1, maxValueDimensionAtCompileTime<Out>, 1>;
    using Jacobian = internal::MatrixOf<
        T,
        valueDimensionAtCompileTime<DifferenceSpace<Out>>,
        valueDimensionAtCompileTime<DifferenceSpace<In>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
    EMRYS_FILTER_ASSERT(x.rows() == valueDimension(in) && x.cols() == 1 && "the point is a vector in the input space");
    auto value = Value{Value::Zero(valueDimension(out))};
    auto jacobian = Jacobian{};
    auto wanted = std::array<bool, std::tuple_size_v<internal::InputBlocks<In>>>{};
    if constexpr (GivesJacobians)
    {
        jacobian.setZero(differenceDimension(out), differenceDimension(in));
        wanted.fill(true);
    }
    auto y = Output{
        typename Output::ValuesType{value.data(), value.rows(), 1},
        typename Output::JacobiansType{jacobian.data(), differenceDimension(out), GivesJacobians ? jacobian.cols() : 0},
        in,
        wanted
    };
    internal::callForm(function, in, out, x, rest..., y);
    if constexpr (GivesJacobians)
    {
        return withJacobian(std::move(value), std::move(jacobian));
    }
    else
    {
        return value;
    }
}

namespace internal
{

/** Calls the batched function \a batched once over \a x, the inputs of a batch, a column per point
 * in the input space \a in, with \a rest, the step of a process model or nothing, before its output,
 * writing into \a y a column per point in the output space \a out, with no Jacobian wanted
 * (decisions.md [batched-functions]). \a y is a plain matrix, whose columns are contiguous.
 */
template <typename F, typename In, typename Out, typename X, typename Y, typename... Rest>
void
callBatched(F const & batched, In const & in, Out const & out, X const & x, Y & y, Rest const &... rest)
{
    static_assert(takesBatches<F>);
    using T = typename X::Scalar;
    using Output = BatchedOutput<In, Out, T, BatchMark<F>>;
    EMRYS_FILTER_ASSERT(
        x.rows() == valueDimension(in) && y.rows() == valueDimension(out) && y.cols() == x.cols()
        && "the batch writes a column in the output space per column in the input space"
    );
    auto output = Output{
        typename Output::ValuesType{y.data(), y.rows(), y.cols()},
        typename Output::JacobiansType{nullptr, differenceDimension(out), 0},
        in,
        {}
    };
    callForm(batchBody(batched), in, out, x, rest..., output);
}

/** \returns the value rows of the block \a Name of \a in, a batched function's input space, as
 * inputBlockRows() gives its difference rows
 */
template <typename Name, typename In>
[[nodiscard]] BlockRows
inputValueBlockRows(In const & in)
{
    if constexpr (IsComposite<In>)
    {
        return blockRows<Name>(in);
    }
    else
    {
        return BlockRows{0, valueDimension(in)};
    }
}

/** Whether the batched function \a F, without a mark of Jacobians, can be called in the dual
 * numbers \a Dual over the input space \a In into the output space \a Out, with \a Rest before its
 * output, as automatic differentiation of a batch calls it: it takes its input as <tt>auto const &</tt>.
 */
template <typename F, typename In, typename Out, typename Dual, typename... Rest>
inline constexpr bool batchTakesDuals = functionForm<
                                            F,
                                            In,
                                            Out,
                                            MatrixOf<
                                                Dual,
                                                valueDimensionAtCompileTime<In>,
                                                Eigen::Dynamic,
                                                maxValueDimensionAtCompileTime<In>,
                                                Eigen::Dynamic>,
                                            Rest const &...,
                                            BatchedOutput<In, Out, Dual, NoJacobians> &>
                                     != 0;

/** Calls the batched function \a batched once over \a x, the inputs of a batch, a column per point
 * in the input space \a in, with \a rest, the step of a process model or nothing, before its output,
 * writing into \a y its values, a column per point in the output space \a out, and into
 * \a jacobians its Jacobian over the blocks of its input that \a wanted marks, in their order
 * (decisions.md [batched-functions]): for column \f$ j \f$, columns \f$ j w \f$ to
 * \f$ j w + w - 1 \f$, with \f$ w \f$ the wanted blocks' difference columns side by side, and a row
 * per difference coordinate of the output. \a y and \a jacobians are plain matrices, whose columns
 * are contiguous.
 *
 * A function marked jacobians::perColumn() writes them in place, and one marked jacobians::shared()
 * writes one that is copied to every column. A function without a mark is differentiated: it is
 * called once in dual numbers whose derivatives are seeded on the wanted blocks, which must be
 * linear, and its Jacobian is that of its output's difference from its value, as Linearized takes it.
 */
template <typename F, typename In, typename Out, typename X, std::size_t K, typename Y, typename J, typename... Rest>
void
callBatchedWithJacobians(
    F const & batched,
    In const & in,
    Out const & out,
    X const & x,
    std::array<bool, K> const & wanted,
    Y & y,
    J & jacobians,
    Rest const &... rest
)
{
    static_assert(takesBatches<F>);
    using Mark = BatchMark<F>;
    using T = typename X::Scalar;
    using Blocks = InputBlocks<In>;
    static_assert(K == std::tuple_size_v<Blocks>);
    Eigen::Index const count = x.cols();
    Eigen::Index const m = differenceDimension(out);
    Eigen::Index w = 0;
    [&]<std::size_t... B>(std::index_sequence<B...>) {
        ((w += wanted[B] ? inputBlockRows<std::tuple_element_t<B, Blocks>>(in).size : 0), ...);
    }(std::make_index_sequence<K>{});
    EMRYS_FILTER_ASSERT(
        x.rows() == valueDimension(in) && y.rows() == valueDimension(out) && y.cols() == count && jacobians.rows() == m
        && jacobians.cols() == w * count
        && "the batch writes a value and a Jacobian over the wanted blocks per column of its input"
    );
    if constexpr (!std::is_same_v<Mark, NoJacobians>)
    {
        using Output = BatchedOutput<In, Out, T, Mark>;
        constexpr bool PerColumn = std::is_same_v<Mark, jacobians::PerColumn>;
        // A shared Jacobian is written once, beside the values, and then copied to every column.
        using Shared = MatrixOf<
            T,
            valueDimensionAtCompileTime<DifferenceSpace<Out>>,
            Eigen::Dynamic,
            maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
        auto shared = Shared{};
        if constexpr (!PerColumn)
        {
            shared.resize(m, w);
        }
        auto output = Output{
            typename Output::ValuesType{y.data(), y.rows(), count},
            typename Output::JacobiansType{PerColumn ? jacobians.data() : shared.data(), m, PerColumn ? w * count : w},
            in,
            wanted
        };
        callForm(batchBody(batched), in, out, x, rest..., output);
        if constexpr (!PerColumn)
        {
            for (Eigen::Index j = 0; j < count; ++j)
            {
                jacobians.middleCols(j * w, w) = shared;
            }
        }
    }
    else
    {
        using Derivatives = Eigen::
            Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, maxValueDimensionAtCompileTime<DifferenceSpace<In>>, 1>;
        using Dual = Eigen::AutoDiffScalar<Derivatives>;
        static_assert(
            batchTakesDuals<BatchBody<F>, In, Out, Dual, Rest...>,
            "a batched function whose Jacobian a method uses is differentiated in dual numbers, so it takes X as "
            "auto const & and writes Y in X's scalar; or it writes its Jacobians, batched(f, jacobians::perColumn()) "
            "or batched(f, jacobians::shared())"
        );
        using DualX = MatrixOf<
            Dual,
            valueDimensionAtCompileTime<In>,
            X::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<In>,
            X::MaxColsAtCompileTime>;
        using DualY = MatrixOf<
            Dual,
            valueDimensionAtCompileTime<Out>,
            Y::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<Out>,
            Y::MaxColsAtCompileTime>;
        auto xd = DualX{};
        xd.resize(x.rows(), count);
        for (Eigen::Index j = 0; j < count; ++j)
        {
            for (Eigen::Index r = 0; r < x.rows(); ++r)
            {
                xd(r, j) = Dual{x(r, j), Derivatives::Zero(w)};
            }
        }
        Eigen::Index offset = 0;
        [&]<std::size_t... B>(std::index_sequence<B...>) {
            (
                [&] {
                    using Name = std::tuple_element_t<B, Blocks>;
                    if (wanted[B])
                    {
                        BlockRows const values = inputValueBlockRows<Name>(in);
                        BlockRows const differences = inputBlockRows<Name>(in);
                        EMRYS_FILTER_ASSERT(
                            isLinear<Name> && values.size == differences.size
                            && "a block differentiated in a batch is linear, its values its differences"
                        );
                        for (Eigen::Index r = 0; r < values.size; ++r)
                        {
                            for (Eigen::Index j = 0; j < count; ++j)
                            {
                                xd(values.start + r, j).derivatives()(offset + r) = T(1);
                            }
                        }
                        offset += differences.size;
                    }
                }(),
                ...
            );
        }(std::make_index_sequence<K>{});
        auto yd = DualY{};
        yd.resize(y.rows(), count);
        callBatched(batched, in, out, xd, yd, rest...);
        using Deviation = MatrixOf<
            Dual,
            valueDimensionAtCompileTime<DifferenceSpace<Out>>,
            1,
            maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
            1>;
        for (Eigen::Index j = 0; j < count; ++j)
        {
            for (Eigen::Index r = 0; r < y.rows(); ++r)
            {
                y(r, j) = yd(r, j).value();
            }
            Deviation const deviation = [&] {
                if constexpr (HasMinus<Out>)
                {
                    return Deviation{Emrys::minus(out, yd.col(j), y.col(j).template cast<Dual>())};
                }
                else
                {
                    return Deviation{yd.col(j)};
                }
            }();
            for (Eigen::Index r = 0; r < m; ++r)
            {
                // An output that does not depend on the wanted blocks may carry no derivatives.
                if (deviation(r).derivatives().size() > 0)
                {
                    jacobians.block(r, j * w, 1, w) = deviation(r).derivatives().transpose();
                }
                else
                {
                    jacobians.block(r, j * w, 1, w).setZero();
                }
            }
        }
    }
}

}  // namespace internal

}  // namespace Emrys
