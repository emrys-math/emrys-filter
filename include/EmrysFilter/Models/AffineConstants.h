#pragma once

/** \file
 * The constant affine part of a model's function (decisions.md [affine-coordinates]): the items
 * <tt>on<Name>(H)</tt> and <tt>offset(...)</tt> of affineIn(), which measurementModel() and
 * processModel() turn into a function that adds \f$ \sum_k H_k x_k + c \f$ to the function's value
 * and the columns \f$ H_k \f$ to its Jacobian, with each \f$ x_k \f$ read and factored as any affine
 * input is.
 */

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/TransformArgs.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The name a constant item <tt>on<Name>(H)</tt> gives its input. */
template <typename Constant>
using ConstantName = PathTag<typename OnPathParts<Constant>::PathType>;

/** The block of the output an offset part <tt>on<Block>(c)</tt> is on. */
template <typename Part>
using OffsetBlock = PathTag<typename OnPathParts<Part>::PathType>;

/** Whether the output space \a Out has the block \a Block among its blocks. */
template <typename Out, typename Block>
inline constexpr bool outputHasBlock = [] {
    if constexpr (IsComposite<Out>)
    {
        return []<typename... Blocks>(std::tuple<Blocks...> *) {
            return (std::is_same_v<Blocks, Block> || ... || false);
        }(static_cast<CompositeBlocks<Out> *>(nullptr));
    }
    else
    {
        return false;
    }
}();

/** Adds the part \a part of an offset() to \a delta, a difference of the output \a out */
template <typename T, typename Out, typename Delta, typename Part>
void
addOffsetPart(Out const & out, Delta & delta, Part const & part)
{
    if constexpr (isConstantItem<Part>)
    {
        using Block = OffsetBlock<Part>;
        static_assert(
            outputHasBlock<Out, Block>,
            "offset(on<Block>(c)) places its part on a block of the model's output, so the output names Block"
        );
        BlockRows const rows = differenceBlockRows<Block>(out);
        EMRYS_FILTER_ASSERT(
            part.value.size() == rows.size && "a part of offset() has a coordinate per difference of its block"
        );
        delta.middleRows(rows.start, rows.size) += part.value.template cast<T>();
    }
    else
    {
        EMRYS_FILTER_ASSERT(
            part.value.size() == delta.size() && "offset(c) has a coordinate per difference of the output"
        );
        delta += part.value.template cast<T>();
    }
}

/** Adds the offset() \a offset to \a delta, a difference of the output \a out */
template <typename T, typename Out, typename Delta, typename... Parts>
void
addOffset(Out const & out, Delta & delta, AffineOffset<Parts...> const & offset)
{
    std::apply(
        [&](auto const &... parts) {
            (addOffsetPart<T>(out, delta, parts), ...);
        },
        offset.parts
    );
}

/** The output type's \a OutputSpaceType, where the function \a F names one, so that the function
 * that adds a constant part keeps the output its function's model has.
 */
template <typename F>
struct OutputSpaceOfFunction
{};

/** A function that names its output space. */
template <typename F>
    requires requires { typename F::OutputSpaceType; }
struct OutputSpaceOfFunction<F>
{
    using OutputSpaceType = typename F::OutputSpaceType;  ///< the function's output space
};

/** What a model without a function has in its place: its map is affineIn() alone, \f$ y = c \boxplus
 * \sum_k H_k x_k \f$ with \f$ c \f$ a point of the output, zero where offset() gives none.
 */
struct NoAffineFunction
{};

/** The function of a model with constant items in affineIn(): the function \a F called on the
 * input \a InnerIn, the items of reads() it was given, and \f$ \sum_k H_k x_k + c \f$ added to its
 * value by plus() on the output, where \f$ x_k \f$ is the input named by the constant item
 * <tt>on<Name_k>(H_k)</tt> and \f$ c \f$ the offset \a Offset (decisions.md [affine-coordinates]).
 * Where \a F returns its Jacobian, the function returns it with the columns \f$ H_k \f$ in its
 * inputs' places; otherwise a method differentiates it, and those columns are \f$ H_k \f$ exactly.
 * The model holds one step where it is a process model's (internal::holdsOneStep), since its
 * constant part follows none.
 *
 * \tparam F          the function given, which reads \a InnerIn
 * \tparam InnerIn    the input the function reads, the Composite of reads()' names
 * \tparam Offset     the offset, an AffineOffset
 * \tparam Constants  the constant items, each an OnPath of a matrix on one tag
 */
template <typename F, typename InnerIn, typename Offset, typename... Constants>
class ConstantAffineFunction : public OutputSpaceOfFunction<F>
{
    static_assert(
        !takesBatches<F>,
        "a batched function takes no constant items in affineIn() yet; give the constant part with path<Name> and "
        "write its columns"
    );

public:

    /// a process model with a constant part holds one step, so that only fixedStep() takes it
    static constexpr bool HoldsOneStep = true;

    /** \a function with the constant items \a constants and the offset \a offset. */
    ConstantAffineFunction(F function, std::tuple<Constants...> constants, Offset offset)
        : m_function{std::move(function)}
        , m_constants{std::move(constants)}
        , m_offset{std::move(offset)}
    {}

    /** \returns the function given, which reads its own items alone */
    [[nodiscard]] F const &
    function() const
    {
        return m_function;
    }

    /** \returns the function's value at \a x, a vector in the input space \a in, with the constant
     * part added in the output space \a out, and its Jacobian with the constant columns where the
     * function returns one; \a rest is a process model's step
     */
    template <typename In, typename Out, typename X, typename... Rest>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, Rest &&... rest) const
        requires MatrixExpression<X>
    {
        static_assert(
            !HasPlus<Out>,
            "a model with constant items in affineIn() adds their term to its function's value as a difference, so "
            "its output's plus() is the plain one, as a Linear or a Periodic output's is"
        );
        using T = typename X::Scalar;
        using Value = typename Moments<Out, T>::VectorType;
        if constexpr (std::is_same_v<F, NoAffineFunction>)
        {
            // The map is the constant part alone, its value the offset moved by H x, its Jacobian the
            // matrices, so nothing is differentiated.
            using Scalar = std::conditional_t<std::is_floating_point_v<T>, T, double>;
            Value const zero = Value::Zero(valueDimension(out));
            Value value = plus(out, zero, constantPart<T>(in, out, x));
            return withJacobian(std::move(value), constantJacobian<Scalar>(in, out));
        }
        else
        {
            return withFunction<T, Value>(in, out, x, std::forward<Rest>(rest)...);
        }
    }

private:

    /** operator()() of a model with a function */
    template <typename T, typename Value, typename In, typename Out, typename X, typename... Rest>
    [[nodiscard]] auto
    withFunction(In const & in, Out const & out, X const & x, Rest &&... rest) const
    {
        InnerIn const inner = innerOf(in, static_cast<CompositeBlocks<InnerIn> *>(nullptr));
        auto const innerX = gatherInner<T>(in, inner, x, static_cast<CompositeBlocks<InnerIn> *>(nullptr));
        auto const returned = callForm(m_function, inner, out, innerX, std::forward<Rest>(rest)...);
        auto const delta = constantPart<T>(in, out, x);
        if constexpr (isWithJacobian<std::remove_cvref_t<decltype(returned)>>)
        {
            Value value = plus(out, returned.value, delta);
            return withJacobian(std::move(value), jacobianOf(in, inner, out, returned.jacobian));
        }
        else
        {
            Value value = plus(out, returned, delta);
            return value;
        }
    }

    /** \returns the instance of \a InnerIn whose blocks are those of \a in with the same names */
    template <typename... Names, typename In>
    [[nodiscard]] static InnerIn
    innerOf(In const & in, std::tuple<Names...> *)
    {
        return InnerIn{pathBlock<Names>(in)...};
    }

    /** \returns the function's own input: the blocks of \a x, in \a in, that \a inner names */
    template <typename T, typename... Names, typename In, typename X>
    [[nodiscard]] static typename Moments<InnerIn, T>::VectorType
    gatherInner(In const & in, InnerIn const & inner, X const & x, std::tuple<Names...> *)
    {
        typename Moments<InnerIn, T>::VectorType result(valueDimension(inner));
        ((result.middleRows(blockRows<Names>(inner).start, blockRows<Names>(inner).size) =
              x.middleRows(blockRows<Names>(in).start, blockRows<Names>(in).size)),
         ...);
        return result;
    }

    /** \returns \f$ \sum_k H_k x_k + c \f$, a difference of the output \a out, in \a T */
    template <typename T, typename In, typename Out, typename X>
    [[nodiscard]] auto
    constantPart(In const & in, Out const & out, X const & x) const
    {
        using Difference = typename Moments<DifferenceSpace<Out>, T>::VectorType;
        Eigen::Index const n = differenceDimension(out);
        Difference delta = Difference::Zero(n);
        std::apply(
            [&](auto const &... constants) {
                (addConstant<T>(in, out, x, delta, constants), ...);
            },
            m_constants
        );
        addOffset<T>(out, delta, m_offset);
        return delta;
    }

    /** Adds \f$ H x \f$ of the constant item \a constant to \a delta, a difference of the output \a out */
    template <typename T, typename In, typename Out, typename X, typename Delta, typename Constant>
    static void
    addConstant(In const & in, Out const & out, X const & x, Delta & delta, Constant const & constant)
    {
        using Name = ConstantName<Constant>;
        using H = typename OnPathParts<Constant>::Value;
        constexpr int Rows = valueDimensionAtCompileTime<DifferenceSpace<Out>>;
        constexpr int Cols = valueDimensionAtCompileTime<DifferenceSpace<Name>>;
        static_assert(
            (int(H::RowsAtCompileTime) == Eigen::Dynamic || Rows == Eigen::Dynamic || int(H::RowsAtCompileTime) == Rows)
                && (int(H::ColsAtCompileTime) == Eigen::Dynamic || Cols == Eigen::Dynamic
                    || int(H::ColsAtCompileTime) == Cols),
            "the matrix of on<Name>(H) in affineIn() has a row per difference of the output and a column per "
            "difference of the input it names"
        );
        Eigen::Index const n = differenceDimension(out);
        BlockRows const rows = blockRows<Name>(in);
        EMRYS_FILTER_ASSERT(
            constant.value.rows() == n && constant.value.cols() == rows.size
            && "the matrix of on<Name>(H) has a row per difference of the output and a column per one of its input"
        );
        delta.noalias() += constant.value.template cast<T>() * x.middleRows(rows.start, rows.size);
    }

    /** \returns the Jacobian over the input \a in: the function's \a jacobian, over \a inner, in its
     * blocks' columns, and each constant item's matrix in its input's
     */
    template <typename In, typename Out, typename J>
    [[nodiscard]] auto
    jacobianOf(In const & in, InnerIn const & inner, Out const & out, J const & jacobian) const
    {
        static_assert(
            MatrixExpression<J>,
            "a function with constant items in affineIn() returns its Jacobian as one matrix over its input's "
            "differences, withJacobian(y, J)"
        );
        using Scalar = typename J::Scalar;
        using Matrix = MatrixOf<
            Scalar,
            valueDimensionAtCompileTime<DifferenceSpace<Out>>,
            valueDimensionAtCompileTime<DifferenceSpace<In>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
        Matrix result = constantJacobian<Scalar>(in, out);
        [&]<typename... Names>(std::tuple<Names...> *) {
            ((result.middleCols(differenceBlockRows<Names>(in).start, differenceBlockRows<Names>(in).size) =
                  jacobian.middleCols(differenceBlockRows<Names>(inner).start, differenceBlockRows<Names>(inner).size)),
             ...);
        }(static_cast<CompositeBlocks<InnerIn> *>(nullptr));
        return result;
    }

    /** \returns the Jacobian over the input \a in of the constant part alone, in \a Scalar: each
     * constant item's matrix in its input's columns, zero elsewhere
     */
    template <typename Scalar, typename In, typename Out>
    [[nodiscard]] auto
    constantJacobian(In const & in, Out const & out) const
    {
        using Matrix = MatrixOf<
            Scalar,
            valueDimensionAtCompileTime<DifferenceSpace<Out>>,
            valueDimensionAtCompileTime<DifferenceSpace<In>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
        Matrix result = Matrix::Zero(differenceDimension(out), differenceDimension(in));
        std::apply(
            [&](auto const &... constants) {
                ((result.middleCols(
                      differenceBlockRows<ConstantName<std::remove_cvref_t<decltype(constants)>>>(in).start,
                      differenceBlockRows<ConstantName<std::remove_cvref_t<decltype(constants)>>>(in).size
                  ) = constants.value.template cast<Scalar>()),
                 ...);
            },
            m_constants
        );
        return result;
    }

    F m_function;                          ///< the function given
    std::tuple<Constants...> m_constants;  ///< the constant items
    Offset m_offset;                       ///< the offset
};

/** Whether the policy \a T is an AffineConstants. */
template <typename T>
inline constexpr bool isAffineConstants = false;

/** An AffineConstants is one. */
template <typename Points, typename Offset, typename... Constants>
inline constexpr bool isAffineConstants<AffineConstants<Points, Offset, Constants...>> = true;

/** Whether one of \a Policies is affineIn() with constant items. */
template <typename... Policies>
inline constexpr bool hasAffineConstants = (isAffineConstants<std::remove_cvref_t<Policies>> || ... || false);

/** Whether \a T is a reads() policy. */
template <typename T>
inline constexpr bool isReadsPolicy = false;

/** A Reads is one. */
template <typename... Items>
inline constexpr bool isReadsPolicy<Reads<Items...>> = true;

/** Whether \a Name is the name of one of the read items \a Items. */
template <typename Name, typename... Items>
inline constexpr bool isReadName = (std::is_same_v<Name, typename Items::NameType> || ... || false);

/** \returns the reads() policy \a reads with an item for each constant name \a Names, each read by
 * its one tag
 */
template <typename... Names, typename... Items>
[[nodiscard]] auto
readsWithConstants(Reads<Items...> const & reads)
{
    static_assert(
        (!isReadName<Names, Items...> && ...),
        "a constant item on<Name>(H) of affineIn() names an input that the function does not read, so reads() "
        "does not name it too"
    );
    using Result = Reads<Items..., ReadItem<TagPath<Names>, Names>...>;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return Result{
            std::in_place,
            reads.template selection<I>()...,
            (static_cast<void>(sizeof(Names)), NoSelection{})...
        };
    }(std::index_sequence_for<Items...>{});
}

/** \returns the policy \a policy as the model with constant items takes it: reads() with the
 * constant names added, affineIn() with them as names, any other as it is
 */
template <typename AffinePolicy, typename Policy>
[[nodiscard]] auto
rewrittenPolicy(Policy const & policy)
{
    using Names = typename AffinePolicy::ConstantNames;
    if constexpr (isReadsPolicy<Policy>)
    {
        return [&]<typename... N>(std::tuple<N...> *) {
            return readsWithConstants<N...>(policy);
        }(static_cast<Names *>(nullptr));
    }
    else if constexpr (isAffineConstants<Policy>)
    {
        return typename Policy::AllNames{};
    }
    else
    {
        return policy;
    }
}

/** \returns the reads() policy among \a policies */
template <typename... Policies>
[[nodiscard]] auto const &
readsAmong(Policies const &... policies)
{
    auto const all = std::tie(policies...);
    constexpr std::size_t I = [] {
        constexpr bool found[] = {isReadsPolicy<Policies>...};
        std::size_t i = 0;
        while (!found[i])
        {
            ++i;
        }
        return i;
    }();
    return std::get<I>(all);
}

/** \returns the affineIn() policy with constant items among \a policies */
template <typename... Policies>
[[nodiscard]] auto const &
affineConstantsAmong(Policies const &... policies)
{
    auto const all = std::tie(policies...);
    constexpr std::size_t I = [] {
        constexpr bool found[] = {isAffineConstants<Policies>...};
        std::size_t i = 0;
        while (!found[i])
        {
            ++i;
        }
        return i;
    }();
    return std::get<I>(all);
}

/** \returns \a make called with the function of \a function and \a policies as a model with
 * constant items in affineIn() takes them: the ConstantAffineFunction of \a function, and the
 * policies with the constant names in reads() and affineIn()
 */
template <typename Make, typename F, typename... Policies>
[[nodiscard]] auto
withAffineConstants(Make const & make, F && function, Policies const &... policies)
{
    static_assert(
        (isReadsPolicy<Policies> || ...),
        "a model with constant items in affineIn() names what its function reads with reads()"
    );
    auto const & affine = affineConstantsAmong(policies...);
    using Affine = std::remove_cvref_t<decltype(affine)>;
    static_assert(
        Affine::PlainConstants,
        "a model with a function takes on<Name>(H) of one input in affineIn(); onEach() and a selection of "
        "repetitions are a model's without one"
    );
    using InnerIn = typename std::remove_cvref_t<decltype(readsAmong(policies...))>::InputSpace;
    using Function = typename Affine::template FunctionOf<std::decay_t<F>, InnerIn>;
    return make(
        Function{std::forward<F>(function), affine.constants, affine.offset},
        rewrittenPolicy<Affine>(policies)...
    );
}

/** Whether \a T is an AffineMatrix. */
template <typename T>
inline constexpr bool isAffineMatrix = false;

/** An AffineMatrix is one. */
template <typename Matrix>
inline constexpr bool isAffineMatrix<AffineMatrix<Matrix>> = true;

/** Whether one of \a Policies is an affineIn() that a model without a function takes as its map: of
 * constant items, or of a matrix.
 */
template <typename... Policies>
inline constexpr bool hasAffineMap =
    hasAffineConstants<Policies...> || (isAffineMatrix<std::remove_cvref_t<Policies>> || ... || false);

/** \returns the policies among \a policies other than affineIn(), as a tuple of copies */
template <typename... Policies>
[[nodiscard]] auto
othersThanAffine(Policies const &... policies)
{
    return std::tuple_cat([&]() {
        if constexpr (isAffineConstants<Policies> || isAffineMatrix<Policies>)
        {
            return std::tuple<>{};
        }
        else
        {
            return std::tuple<Policies>{policies};
        }
    }()...);
}

/** \returns the affineIn() that is a model's map among \a policies */
template <typename... Policies>
[[nodiscard]] auto const &
affineMapAmong(Policies const &... policies)
{
    auto const all = std::tie(policies...);
    constexpr std::size_t I = [] {
        constexpr bool found[] = {(isAffineConstants<Policies> || isAffineMatrix<Policies>)...};
        std::size_t i = 0;
        while (!found[i])
        {
            ++i;
        }
        return i;
    }();
    return std::get<I>(all);
}

/** \returns \a make called with the function and the policies of the model without a function whose
 * map is the affineIn() \a affine, the constant items on one input each and an offset: the
 * ConstantAffineFunction of no function, reads() of the constant names, and affineIn() of them, with
 * \a others, the model's other policies, after them
 */
template <typename Make, typename Affine, typename... Others>
[[nodiscard]] auto
modelOfAffineMap(Make const & make, Affine const & affine, std::tuple<Others...> const & others)
{
    static_assert(
        Affine::NoPoints,
        "a model without a function has no coefficient to give at each point, so its affineIn() takes on<Name>(H), "
        "not path<Name>"
    );
    static_assert(
        Affine::PlainConstants,
        "a model without a function whose affineIn() has an offset() takes on<Name>(H) of one input each; onEach() "
        "and a selection of repetitions take no offset() yet"
    );
    using Function = typename Affine::template FunctionOf<NoAffineFunction, NoAffineFunction>;
    return [&]<typename... Names>(std::tuple<Names...> *) {
        return std::apply(
            [&](auto const &... other) {
                return make(
                    Function{NoAffineFunction{}, affine.constants, affine.offset},
                    Reads<ReadItem<TagPath<Names>, Names>...>{},
                    typename Affine::AllNames{},
                    other...
                );
            },
            others
        );
    }(static_cast<typename Affine::ConstantNames *>(nullptr));
}

}  // namespace internal

}  // namespace Emrys
