#pragma once

/** \file
 * Models placed in a larger state and measurement models side by side (decisions.md [model-placement],
 * [model-policies]): on<Path>(model), which places a process or a measurement model on the block that a path names,
 * prefixing every path into the state it reads and writes; onEach<Path>(model), which places one copy of a process
 * model on every repetition of a repeated block; and compose() of measurement models, whose output is their outputs
 * side by side. compose() of process models is in ProcessModel.h. Read Measurement.h next.
 */

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/ResolvedReads.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The name a placed model gives the block \a Name it reads or writes in the state, under the
 * prefix \a Prefix, a TagPath: a tag of its own with \a Name's leaf beneath it, so that two
 * placements of one model read and write blocks a composition tells apart.
 */
template <typename Prefix, typename Name>
struct PlacedName : Name
{
    using Name::Name;
};

/** The TagPath of \a Prefix's tags followed by \a Path's. */
template <typename Prefix, typename Path>
struct JoinPathsOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Prefix, typename... Tags>
struct JoinPathsOf<TagPath<Prefix...>, TagPath<Tags...>>
{
    using type = TagPath<Prefix..., Tags...>;
};

/// \endcond

/** The TagPath of \a Prefix's tags followed by \a Path's. */
template <typename Prefix, typename Path>
using JoinPaths = typename JoinPathsOf<Prefix, Path>::type;

/** The first tag of the TagPath \a Path. */
template <typename Path>
struct FirstTagOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename First, typename... Rest>
struct FirstTagOf<TagPath<First, Rest...>>
{
    using type = First;
};

/// \endcond

/** Whether the TagPath \a Path starts in the space \a S: its first tag is a block of \a S. */
template <typename Path, typename S>
inline constexpr bool startsIn = countTag<typename FirstTagOf<Path>::type, S>() > 0;

/** Whether \a T renames an input of a placed model: <tt>on<Name>(path<Tag>)</tt>. */
template <typename T>
inline constexpr bool isRename = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <PathValue X, typename Name>
inline constexpr bool isRename<OnPath<X, Name>> = isTagPath<PathOf<X>> || PathValue<X>;

/// \endcond

/** The name the renames \a Renames give the input \a Name: the \a New of the first
 * <tt>on<New>(path<Name>)</tt> among them, or \a Name itself.
 */
template <typename Name, typename... Renames>
struct RenamedOf
{
    using type = Name;  ///< no rename names it
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Name, typename X, typename New, typename... Rest>
struct RenamedOf<Name, OnPath<X, New>, Rest...>
{
    using type =
        std::conditional_t<std::is_same_v<PathOf<X>, TagPath<Name>>, New, typename RenamedOf<Name, Rest...>::type>;
};

/// \endcond

/** The name \a Name renamed by \a Renames. */
template <typename Name, typename... Renames>
using Renamed = typename RenamedOf<Name, Renames...>::type;

/** Whether one of the items in the tuple type \a Items is an input, not a block of \a S, read under
 * the name the rename \a Rename renames: a class rather than a lambda, which MSVC evaluates wrongly
 * in a class template's static_assert.
 */
template <typename Rename, typename S, typename Items>
struct RenamesAnInput
{
    static constexpr bool value = false;  ///< not a rename
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename New, typename S, typename... Items>
struct RenamesAnInput<OnPath<X, New>, S, std::tuple<Items...>>
{
    static constexpr bool value =
        ((std::is_same_v<PathOf<X>, TagPath<typename Items::NameType>> && !startsIn<typename Items::PathType, S>) || ...
         || false);
};

/// \endcond

/** The names of the AffineIn \a A as a tuple type, or none for NoPolicy. */
template <typename A>
struct AffineNamesOf
{
    using type = std::tuple<>;  ///< none
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Names>
struct AffineNamesOf<AffineIn<Names...>>
{
    using type = std::tuple<Names...>;
};

/// \endcond

/** The affine names of the model \a Model, as a tuple type. */
template <typename Model>
using AffineNamesOfModel = typename AffineNamesOf<typename Model::template Policy<PolicyKind::affineIn>>::type;

/** Whether the model \a Model names \a Name in its affineIn(). */
template <typename Model, typename Name>
inline constexpr bool declaresAffine = []<typename... Names>(std::tuple<Names...> *) {
    return (std::is_same_v<Names, Name> || ...);
}(static_cast<AffineNamesOfModel<Model> *>(nullptr));

/** \returns the position of \a Name among \a Names, or their count if it is not one of them */
template <typename Name, typename... Names>
[[nodiscard]] consteval std::size_t
nameIndex()
{
    constexpr std::array<bool, sizeof...(Names) + 1> same{std::is_same_v<Names, Name>..., true};
    std::size_t i = 0;
    while (!same[i])
    {
        ++i;
    }
    return i;
}

/** How the model \a Model, placed on the block that the TagPath \a Prefix names with the renames
 * \a Renames, reads and writes: each item of its reads() that starts in the block is a block of the
 * state, its path prefixed and its name a PlacedName of its own; each other item is an input of the
 * call, under its name or the one a rename gives it; and each block written is prefixed.
 */
template <typename Prefix, typename Model, typename... Renames>
struct Placement
{
    using PrefixSpace = PathTag<Prefix>;                                    ///< the block the model is placed on
    using InnerReads = typename Model::template Policy<PolicyKind::reads>;  ///< the model's own reads()
    using InnerItems = typename ItemsOf<InnerReads>::type;                  ///< and its items

    static_assert(
        (isRename<Renames> && ...),
        "on<Path>(model, renames...) renames an input of the model with on<Name>(path<Tag>)"
    );
    static_assert(
        (RenamesAnInput<Renames, PrefixSpace, InnerItems>::value && ... && true),
        "a rename in on<Path>(model, on<Name>(path<Tag>)) names an input the model reads, a tag that is not a block of "
        "the block Path names"
    );

    /** The item the model's item \a Item becomes. */
    template <typename Item>
    struct OuterItemOf
    {
        static constexpr bool State = startsIn<typename Item::PathType, PrefixSpace>;  ///< a block of the state

        using New = Renamed<typename Item::NameType, Renames...>;  ///< an input's name after the renames

        static_assert(
            State || sameLeaf<New, typename Item::NameType>,
            "a renamed input has the same leaf beneath it as the name the model reads it under"
        );

        /** The item. */
        using type = std::conditional_t<
            State,
            ReadItem<JoinPaths<Prefix, typename Item::PathType>, PlacedName<Prefix, typename Item::NameType>>,
            ReadItem<TagPath<New>, New>>;
    };

    /** The placed reads(). */
    using ReadsPolicy = typename decltype([]<typename... Items>(std::tuple<Items...> *) {
        return std::type_identity<Reads<typename OuterItemOf<Items>::type...>>{};
    }(static_cast<InnerItems *>(nullptr)))::type;

    /** The name the placed model reads the model's name \a Name under. */
    template <typename Name>
    using OuterName = typename decltype([]<typename... Items>(std::tuple<Items...> *) {
        return std::type_identity<std::tuple_element_t<
            nameIndex<Name, typename Items::NameType...>(),
            std::tuple<typename OuterItemOf<Items>::type::NameType...>>>{};
    }(static_cast<InnerItems *>(nullptr)))::type;

    /** The placed affineIn(). */
    using AffinePolicy = typename decltype([]<typename... Names>(std::tuple<Names...> *) {
        return std::type_identity<AffineIn<OuterName<Names>...>>{};
    }(static_cast<AffineNamesOfModel<Model> *>(nullptr)))::type;
};

/** The function of a process model placed on a block (see on()): the model's own function, called
 * with the model's input and output spaces under the names it reads and writes them by, on the
 * input vector as it is, since the placed model reads the same blocks in the same order. Its
 * noise is the model's.
 *
 * \tparam Model  the model placed
 */
template <typename Model>
class PlacedStep
{
public:

    using InnerIn = typename Model::template Policy<PolicyKind::reads>::InputSpace;  ///< the model's input
    using InnerOut = OutputSpaceOf<Model>;                                           ///< and its output

    /** The function of \a model placed. */
    explicit PlacedStep(Model model)
        : m_model{std::move(model)}
    {}

    /** \returns the model placed */
    [[nodiscard]] Model const &
    model() const
    {
        return m_model;
    }

    /** \returns the model placed, to change its parameters */
    [[nodiscard]] Model &
    model()
    {
        return m_model;
    }

    /** \returns the model's step of \a x, a vector in the placed input space \a in, into the placed
     * output space \a out over \a dt
     */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] decltype(auto)
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt) const
        requires MatrixExpression<X>
    {
        return callForm(m_model.function(), renamedInstance<InnerIn>(in), renamedInstance<InnerOut>(out), x, dt);
    }

    /** The model's function. */
    using Inner = std::remove_cvref_t<decltype(std::declval<Model const &>().function())>;

    /// whether the placed function takes a batch, as the model's does (decisions.md [batched-functions])
    static constexpr bool TakesBatches = takesBatches<Inner>;

    /** The Jacobians the placed function writes in a batch: the model's, by the same blocks. */
    using MarkType = BatchMarkOrNone<Inner>;

    /** Writes into \a y the model's step of a batch \a x, a column per point in the placed input
     * space \a in, into the placed output space \a out over \a dt, by one call of the model's batched
     * function, which writes \a y under its own names for the same blocks
     */
    template <typename In, typename Out, typename X, typename TimeScalar, typename Y>
    void
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt, Y & y) const
        requires TakesBatches
    {
        auto const innerIn = renamedInstance<InnerIn>(in);
        auto const innerOut = renamedInstance<InnerOut>(out);
        auto inner = y.template renamed<InnerOut>(innerIn);
        callForm(batchBody(m_model.function()), innerIn, innerOut, x, dt, inner);
    }

    /** \returns the model's process noise over \a dt, over the differences of the placed output \a out */
    template <typename Scalar, typename Out>
    [[nodiscard]] auto
    noiseOver(Out const & out, Scalar dt) const
    {
        return m_model.template noiseOver<Scalar>(renamedInstance<InnerOut>(out), dt);
    }

private:

    Model m_model;  ///< the model placed
};

/** The placed writes() of the process model \a Model on the block that \a Prefix names: each path
 * prefixed, each name a PlacedName.
 */
template <typename Prefix, typename Model>
using PlacedWrites = typename decltype([]<std::size_t... K>(std::index_sequence<K...>) {
    static_assert(
        (startsIn<WritePathAt<Model, K>, PathTag<Prefix>> && ...),
        "on<Path>(model) places a model whose writes() are blocks of the block Path names"
    );
    return std::type_identity<WritesAs<
        TagPaths<JoinPaths<Prefix, WritePathAt<Model, K>>...>,
        Composite<PlacedName<Prefix, WriteNameAt<Model, K>>...>>>{};
}(std::make_index_sequence<writeCount<Model>>{}))::type;

/** The kinds of item of a process model placed on every repetition of a repeated block. */
enum class EachKind
{
    state,     ///< a block of one repetition
    shared,    ///< an input every repetition reads
    repeated,  ///< an input with a value per repetition, renamed to a Repeated tag
};

/** How the process model \a Model, placed on every repetition of the repeated block that the
 * TagPath \a Prefix names with the renames \a Renames, reads: the whole repeated block once, under
 * its tag, in place of the items that start in one repetition; each input renamed to a Repeated tag
 * as one argument with a value per repetition; and each other input as it is, shared by every
 * repetition (decisions.md [model-placement]).
 */
template <typename Prefix, typename Model, typename... Renames>
struct EachPlacement
{
    using RepeatedTag = PathTag<Prefix>;  ///< the repeated block

    static_assert(IsRepeated<RepeatedTag>, "onEach<Path>(model) places a model on a repeated block of the state");

    using Block = RepeatedBlock<RepeatedTag>;                               ///< one repetition
    using InnerReads = typename Model::template Policy<PolicyKind::reads>;  ///< the model's own reads()
    using InnerItems = typename ItemsOf<InnerReads>::type;                  ///< and its items
    using InnerIn = typename InnerReads::InputSpace;                        ///< the model's input
    using InnerOut = OutputSpaceOf<Model>;                                  ///< and its output

    static constexpr std::size_t ItemCount = std::tuple_size_v<InnerItems>;  ///< the model's items

    static_assert(
        (isRename<Renames> && ...),
        "onEach<Path>(model, renames...) renames an input of the model with on<Name>(path<Tag>)"
    );
    static_assert(
        (RenamesAnInput<Renames, Block, InnerItems>::value && ... && true),
        "a rename in onEach<Path>(model, on<Name>(path<Tag>)) names an input the model reads, a tag that is not a "
        "block of one repetition of the block Path names"
    );
    static_assert(
        FixedSizeSpace<InnerIn> && FixedSizeSpace<InnerOut>,
        "onEach<Path>(model) repeats a model whose input and output are fixed-size"
    );
    static_assert(
        pathsPartition<Block, WritePathsOf<Model>>(),
        "onEach<Path>(model) places a model that writes every coordinate of one repetition of the block exactly once"
    );

    /** Item \a K of the model. */
    template <std::size_t K>
    using InnerItem = std::tuple_element_t<K, InnerItems>;

    /** The name item \a K's input is given under, after the renames. */
    template <std::size_t K>
    using NewName = Renamed<typename InnerItem<K>::NameType, Renames...>;

    /** \returns the kind of item \a K */
    template <std::size_t K>
    [[nodiscard]] static consteval EachKind
    kind()
    {
        if constexpr (startsIn<typename InnerItem<K>::PathType, Block>)
        {
            return EachKind::state;
        }
        else if constexpr (IsRepeated<NewName<K>>)
        {
            static_assert(
                sameLeaf<RepeatedBlock<NewName<K>>, typename InnerItem<K>::NameType>,
                "an input renamed to a Repeated tag repeats the leaf the model reads it under"
            );
            return EachKind::repeated;
        }
        else
        {
            static_assert(
                sameLeaf<NewName<K>, typename InnerItem<K>::NameType>,
                "a renamed input has the same leaf beneath it as the name the model reads it under"
            );
            return EachKind::shared;
        }
    }

    /** The placed reads(): the repeated block, then each input. */
    using ReadsPolicy = typename decltype([]<std::size_t... K>(std::index_sequence<K...>) {
        return std::type_identity<decltype(std::tuple_cat(
            std::tuple<ReadItem<Prefix, RepeatedTag>>{},
            std::conditional_t<
                kind<K>() == EachKind::state,
                std::tuple<>,
                std::tuple<ReadItem<TagPath<NewName<K>>, NewName<K>>>>{}...
        ))>{};
    }(std::make_index_sequence<ItemCount>{}))::type;

    /** The placed reads(), as a Reads. */
    using ReadsOfEach = typename decltype([]<typename... Items>(std::tuple<Items...> *) {
        return std::type_identity<Reads<Items...>>{};
    }(static_cast<ReadsPolicy *>(nullptr)))::type;

    /** The placed affineIn(): the repeated block where the model is affine in every block of one
     * repetition it reads, and each input the model is affine in.
     */
    using AffinePolicy = typename decltype([]<std::size_t... K>(std::index_sequence<K...>) {
        constexpr bool blockAffine =
            ((kind<K>() != EachKind::state || declaresAffine<Model, typename InnerItem<K>::NameType>) && ...);
        return std::type_identity<decltype(std::tuple_cat(
            std::conditional_t<blockAffine, std::tuple<RepeatedTag>, std::tuple<>>{},
            std::conditional_t < kind<K>() != EachKind::state && declaresAffine<Model, typename InnerItem<K>::NameType>,
            std::tuple<NewName<K>>,
            std::tuple < >> {}...
        ))>{};
    }(std::make_index_sequence<ItemCount>{}))::type;

    /** The placed affineIn(), as an AffineIn. */
    using AffineOfEach = typename decltype([]<typename... Names>(std::tuple<Names...> *) {
        return std::type_identity<AffineIn<Names...>>{};
    }(static_cast<AffinePolicy *>(nullptr)))::type;
};

/** The function of a process model placed on every repetition of a repeated block (see onEach()):
 * the model's own function called once per repetition, on that repetition's blocks and its value
 * of each repeated input, with the shared inputs as they are, its results written into the
 * repetitions, and its Jacobian, where it returns one, assembled into the whole's. Its noise is the
 * model's on every repetition, block-diagonal.
 *
 * \tparam Prefix   the path to the repeated block, a TagPath
 * \tparam Model    the model placed
 * \tparam Renames  the renames of its inputs
 */
template <typename Prefix, typename Model, typename... Renames>
class EachStep
{
    /** how the model is placed */
    using P = EachPlacement<Prefix, Model, Renames...>;
    /** the tag of the repeated block */
    using RepeatedTag = typename P::RepeatedTag;
    /** one repetition's block */
    using Block = typename P::Block;
    /** the input space of one repetition's model */
    using InnerIn = typename P::InnerIn;
    /** the output space of one repetition's model */
    using InnerOut = typename P::InnerOut;

    static constexpr int BlockSize = valueDimensionAtCompileTime<Block>;  ///< one repetition's values
    static constexpr int BlockDifferenceSize = valueDimensionAtCompileTime<DifferenceSpace<Block>>;  ///< differences

public:

    /** The function of \a model on every repetition. */
    explicit EachStep(Model model)
        : m_model{std::move(model)}
    {}

    /** \returns the model placed */
    [[nodiscard]] Model const &
    model() const
    {
        return m_model;
    }

    /** \returns the model placed, to change its parameters */
    [[nodiscard]] Model &
    model()
    {
        return m_model;
    }

    /** \returns the step of \a x, a vector in the placed input space \a in, into the placed output
     * space \a out over \a dt: the model on each repetition
     */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt) const
        requires MatrixExpression<X>
    {
        using T = ScalarOf<X>;
        using InnerVector = typename Moments<InnerIn, T>::VectorType;
        constexpr bool ReturnsJacobian = returnsJacobianWith<Model, TimeScalar, TimeScalar>;
        Eigen::Index const count = pathBlock<RepeatedTag>(in).count();
        BlockRows const inRows = blockRows<RepeatedTag>(in);
        BlockRows const outRows = blockRows<RepeatedTag>(out);
        auto y = zeros<T>(out);
        auto jacobian = [&] {
            if constexpr (ReturnsJacobian)
            {
                using Matrix = MatrixOf<
                    TimeScalar,
                    valueDimensionAtCompileTime<DifferenceSpace<Out>>,
                    valueDimensionAtCompileTime<DifferenceSpace<In>>,
                    maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
                    maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
                return Matrix{Matrix::Zero(differenceDimension(out), differenceDimension(in))};
            }
            else
            {
                return NoPolicy{};
            }
        }();
        InnerIn const innerIn{};
        InnerOut const innerOut{};
        for (Eigen::Index i = 0; i < count; ++i)
        {
            InnerVector xi{InnerVector::Zero(valueDimension(innerIn))};
            [&]<std::size_t... K>(std::index_sequence<K...>) {
                (gather<K>(in, x, i, inRows, innerIn, xi), ...);
            }(std::make_index_sequence<P::ItemCount>{});
            auto result = vectorReturned(callForm(m_model.function(), innerIn, innerOut, xi, dt));
            auto const & value = valueOf(result);
            [&]<std::size_t... W>(std::index_sequence<W...>) {
                ((y.segment(
                      outRows.start + i * BlockSize + blockRows<WritePathAt<Model, W>>(Block{}).start,
                      blockRows<WritePathAt<Model, W>>(Block{}).size
                  ) =
                      value.segment(
                          blockRows<WriteNameAt<Model, W>>(innerOut).start,
                          blockRows<WriteNameAt<Model, W>>(innerOut).size
                      )),
                 ...);
            }(std::make_index_sequence<writeCount<Model>>{});
            if constexpr (ReturnsJacobian)
            {
                [&]<std::size_t... W>(std::index_sequence<W...>) {
                    (jacobianRows<W>(in, out, i, innerIn, innerOut, result.jacobian, jacobian), ...);
                }(std::make_index_sequence<writeCount<Model>>{});
            }
        }
        if constexpr (ReturnsJacobian)
        {
            return withJacobian(std::move(y), std::move(jacobian));
        }
        else
        {
            return y;
        }
    }

    /** The model's function. */
    using Inner = std::remove_cvref_t<decltype(std::declval<Model const &>().function())>;

    /// whether the placed function takes a batch, as the model's does (decisions.md [batched-functions])
    static constexpr bool TakesBatches = takesBatches<Inner>;

    /** The placed function writes no Jacobian in a batch: a method that needs one differentiates it. */
    using MarkType = NoJacobians;

    /** Writes into \a y the step of a batch \a x, a column per point in the placed input space
     * \a in, into the placed output space \a out over \a dt: the model's batched function called once
     * per repetition over that repetition's columns, as the decision on batched functions has it
     */
    template <typename In, typename Out, typename X, typename TimeScalar, typename Y>
    void
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt, Y & y) const
        requires TakesBatches
    {
        using T = typename X::Scalar;
        using InnerVector = typename Moments<InnerIn, T>::VectorType;
        using XI = MatrixOf<
            T,
            valueDimensionAtCompileTime<InnerIn>,
            X::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<InnerIn>,
            X::MaxColsAtCompileTime>;
        using YI = MatrixOf<
            T,
            valueDimensionAtCompileTime<InnerOut>,
            X::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<InnerOut>,
            X::MaxColsAtCompileTime>;
        Eigen::Index const repetitions = pathBlock<RepeatedTag>(in).count();
        Eigen::Index const points = x.cols();
        BlockRows const inRows = blockRows<RepeatedTag>(in);
        BlockRows const outRows = blockRows<RepeatedTag>(out);
        auto & values = y.values();
        values.setZero();
        InnerIn const innerIn{};
        InnerOut const innerOut{};
        auto xi = XI{};
        xi.resize(valueDimension(innerIn), points);
        auto yi = YI{};
        yi.resize(valueDimension(innerOut), points);
        auto column = InnerVector{};
        column.resize(valueDimension(innerIn));
        for (Eigen::Index i = 0; i < repetitions; ++i)
        {
            for (Eigen::Index j = 0; j < points; ++j)
            {
                column.setZero();
                [&]<std::size_t... K>(std::index_sequence<K...>) {
                    (gather<K>(in, x.col(j), i, inRows, innerIn, column), ...);
                }(std::make_index_sequence<P::ItemCount>{});
                xi.col(j) = column;
            }
            callBatched(m_model.function(), innerIn, innerOut, xi, yi, dt);
            [&]<std::size_t... W>(std::index_sequence<W...>) {
                ((values.middleRows(
                      outRows.start + i * BlockSize + blockRows<WritePathAt<Model, W>>(Block{}).start,
                      blockRows<WritePathAt<Model, W>>(Block{}).size
                  ) =
                      yi.middleRows(
                          blockRows<WriteNameAt<Model, W>>(innerOut).start,
                          blockRows<WriteNameAt<Model, W>>(innerOut).size
                      )),
                 ...);
            }(std::make_index_sequence<writeCount<Model>>{});
        }
    }

    /** \returns the process noise over \a dt, over the differences of the placed output \a out: the
     * model's on every repetition
     */
    template <typename Scalar, typename Out>
    [[nodiscard]] typename Moments<Out, Scalar>::MatrixType
    noiseOver(Out const & out, Scalar dt) const
    {
        using Matrix = typename Moments<Out, Scalar>::MatrixType;
        Eigen::Index const n = differenceDimension(out);
        Matrix q = Matrix::Zero(n, n);
        InnerOut const innerOut{};
        auto const inner = m_model.template noiseOver<Scalar>(innerOut, dt);
        Eigen::Index const count = pathBlock<RepeatedTag>(out).count();
        Eigen::Index const start = differenceBlockRows<RepeatedTag>(out).start;
        for (Eigen::Index i = 0; i < count; ++i)
        {
            [&]<std::size_t... W>(std::index_sequence<W...>) {
                (noiseRows<W>(q, start + i * BlockDifferenceSize, innerOut, inner), ...);
            }(std::make_index_sequence<writeCount<Model>>{});
        }
        return q;
    }

private:

    /** Writes item \a K of repetition \a i's input into \a xi, from \a x over \a in. */
    template <std::size_t K, typename In, typename X, typename XI>
    static void
    gather(In const & in, X const & x, Eigen::Index i, BlockRows const & inRows, InnerIn const & innerIn, XI & xi)
    {
        using Item = typename P::template InnerItem<K>;
        BlockRows const rows = blockRows<typename Item::NameType>(innerIn);
        if constexpr (P::template kind<K>() == EachKind::state)
        {
            BlockRows const within = blockRows<typename Item::PathType>(Block{});
            xi.segment(rows.start, rows.size) = x.segment(inRows.start + i * BlockSize + within.start, within.size);
        }
        else if constexpr (P::template kind<K>() == EachKind::repeated)
        {
            xi.segment(rows.start, rows.size) =
                block<typename P::template NewName<K>>(in, x).segment(i * rows.size, rows.size);
        }
        else
        {
            xi.segment(rows.start, rows.size) = block<typename P::template NewName<K>>(in, x);
        }
    }

    /** Writes the rows of the block written at position \a W of repetition \a i's Jacobian
     * \a inner into the whole's \a jacobian, one column block per item.
     */
    template <std::size_t W, typename In, typename Out, typename J, typename Whole>
    static void
    jacobianRows(
        In const & in,
        Out const & out,
        Eigen::Index i,
        InnerIn const & innerIn,
        InnerOut const & innerOut,
        J const & inner,
        Whole & jacobian
    )
    {
        BlockRows const innerRows = differenceBlockRows<WriteNameAt<Model, W>>(innerOut);
        Eigen::Index const row = differenceBlockRows<RepeatedTag>(out).start + i * BlockDifferenceSize
                               + differenceBlockRows<WritePathAt<Model, W>>(Block{}).start;
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            (
                [&] {
                    using Item = typename P::template InnerItem<K>;
                    BlockRows const innerColumns = differenceBlockRows<typename Item::NameType>(innerIn);
                    Eigen::Index column = 0;
                    if constexpr (P::template kind<K>() == EachKind::state)
                    {
                        column = differenceBlockRows<RepeatedTag>(in).start + i * BlockDifferenceSize
                               + differenceBlockRows<typename Item::PathType>(Block{}).start;
                    }
                    else if constexpr (P::template kind<K>() == EachKind::repeated)
                    {
                        column = differenceBlockRows<typename P::template NewName<K>>(in).start + i * innerColumns.size;
                    }
                    else
                    {
                        column = differenceBlockRows<typename P::template NewName<K>>(in).start;
                    }
                    jacobian.block(row, column, innerRows.size, innerColumns.size) =
                        inner.block(innerRows.start, innerColumns.start, innerRows.size, innerColumns.size);
                }(),
                ...
            );
        }(std::make_index_sequence<P::ItemCount>{});
    }

    /** Writes the row block of the block written at position \a W of one repetition's noise
     * \a inner into \a q, from the repetition's first difference row \a start.
     */
    template <std::size_t W, typename Q, typename Inner>
    static void
    noiseRows(Q & q, Eigen::Index start, InnerOut const & innerOut, Inner const & inner)
    {
        BlockRows const innerRows = differenceBlockRows<WriteNameAt<Model, W>>(innerOut);
        Eigen::Index const row = start + differenceBlockRows<WritePathAt<Model, W>>(Block{}).start;
        [&]<std::size_t... V>(std::index_sequence<V...>) {
            (
                [&] {
                    BlockRows const innerColumns = differenceBlockRows<WriteNameAt<Model, V>>(innerOut);
                    Eigen::Index const column = start + differenceBlockRows<WritePathAt<Model, V>>(Block{}).start;
                    q.block(row, column, innerRows.size, innerColumns.size) =
                        inner.block(innerRows.start, innerColumns.start, innerRows.size, innerColumns.size);
                }(),
                ...
            );
        }(std::make_index_sequence<writeCount<Model>>{});
    }

    Model m_model;  ///< the model placed
};

/** The function of a measurement model placed on a block (see on()): the model's own function,
 * called with its input space under the names it reads by and the output as it is, since a
 * measurement's output is not part of the state.
 *
 * \tparam F        the model's function
 * \tparam InnerIn  the model's input space
 */
template <typename F, typename InnerIn>
class PlacedMeasure
{
public:

    /** The function \a function placed. */
    explicit PlacedMeasure(F function)
        : m_function{std::move(function)}
    {}

    /** \returns the measurement of \a x, a vector in the placed input space \a in, in \a out */
    template <typename In, typename Out, typename X>
    [[nodiscard]] decltype(auto)
    operator()(In const & in, Out const & out, X const & x) const
        requires MatrixExpression<X>
    {
        return callForm(m_function, renamedInstance<InnerIn>(in), out, x);
    }

    /// whether the placed function takes a batch, as the function does (decisions.md [batched-functions])
    static constexpr bool TakesBatches = takesBatches<F>;

    /** The Jacobians the placed function writes in a batch: the function's, by the same blocks. */
    using MarkType = BatchMarkOrNone<F>;

    /** Writes into \a y the measurement of a batch \a x, a column per point in the placed input space
     * \a in, in \a out, by one call of the batched function, which reads its input by its own names
     */
    template <typename In, typename Out, typename X, typename Y>
    void
    operator()(In const & in, Out const & out, X const & x, Y & y) const
        requires TakesBatches
    {
        auto const innerIn = renamedInstance<InnerIn>(in);
        auto inner = y.template renamed<Out>(innerIn);
        callForm(batchBody(m_function), innerIn, out, x, inner);
    }

    /** \returns the rows of the function's result, where it knows them */
    [[nodiscard]] Eigen::Index
    outputRows() const
        requires requires(F const & f) { f.outputRows(); }
    {
        return m_function.outputRows();
    }

private:

    EMRYS_FILTER_NO_UNIQUE_ADDRESS F m_function;  ///< the function
};

/** The likelihood of a measurement model placed on a block: the model's own, called with its input
 * space under the names it reads by.
 *
 * \tparam G        the model's likelihood
 * \tparam InnerIn  the model's input space
 */
template <typename G, typename InnerIn>
struct PlacedLikelihood
{
    G function;  ///< the likelihood

    /** \returns \f$ \log p(z \mid x) \f$ at \a x, a vector in the placed input \a in */
    template <typename In, typename X, typename Z>
    [[nodiscard]] auto
    operator()(In const & in, X const & x, Z const & z) const
    {
        return function(renamedInstance<InnerIn>(in), x, z);
    }
};

/** The draw of an observation of a measurement model placed on a block: the model's own, called
 * with its input space under the names it reads by.
 *
 * \tparam G        the model's draw
 * \tparam InnerIn  the model's input space
 */
template <typename G, typename InnerIn>
struct PlacedSample
{
    G function;  ///< the draw

    /** \returns a draw of the observation at \a x, a vector in the placed input \a in, from \a rng */
    template <typename In, typename X, typename R>
    [[nodiscard]] auto
    operator()(In const & in, X const & x, R & rng) const
    {
        return function(renamedInstance<InnerIn>(in), x, rng);
    }
};

/** The output space function of a measurement model placed on the block that \a Prefix names: the
 * model's own, called with the space of that block, since the model was written for it.
 */
template <typename Prefix, typename G>
struct PlacedOutputSpace
{
    G function;  ///< the model's output space function

    /** \returns the output space in the state's space \a space */
    template <typename StateSpace>
    [[nodiscard]] auto
    operator()(StateSpace const & space) const
    {
        return function(pathSubspace<Prefix>(space));
    }
};

/** \returns the policy \a policy of a measurement model placed on the block that \a Prefix names,
 * with reads \a ReadsPolicy and affine names \a AffinePolicy: the reads() and affineIn() placed, a
 * likelihood, a draw, or an output space function given the placed input or the block's space, and anything
 * else as it is
 */
template <typename Prefix, typename InnerIn, typename ReadsPolicy, typename AffinePolicy, typename Policy>
[[nodiscard]] auto
placedPolicy(Policy const & policy)
{
    constexpr std::optional<PolicyKind> kind = policyKind<Policy>;
    if constexpr (kind == PolicyKind::reads)
    {
        return ReadsPolicy{};
    }
    else if constexpr (kind == PolicyKind::affineIn)
    {
        return AffinePolicy{};
    }
    else if constexpr (kind == PolicyKind::logLikelihood)
    {
        using G = std::remove_cvref_t<decltype(policy.function)>;
        return LogLikelihood<PlacedLikelihood<G, InnerIn>>{PlacedLikelihood<G, InnerIn>{policy.function}};
    }
    else if constexpr (kind == PolicyKind::sample)
    {
        using G = std::remove_cvref_t<decltype(policy.function)>;
        return Sample<PlacedSample<G, InnerIn>>{PlacedSample<G, InnerIn>{policy.function}};
    }
    else if constexpr (kind == PolicyKind::outputSpace)
    {
        if constexpr (Policy::Computed)
        {
            using G = std::remove_cvref_t<decltype(policy.value)>;
            return OutputSpace<PlacedOutputSpace<Prefix, G>>{PlacedOutputSpace<Prefix, G>{policy.value}};
        }
        else
        {
            return policy;
        }
    }
    else
    {
        return policy;
    }
}

/** The function of measurement models side by side (see compose()): each model's function on its
 * own input, gathered from the whole's by name, its result written into the whole's output by
 * block, and the Jacobian assembled from the models' where every one returns one.
 *
 * \tparam Models  the models, each a MeasurementModel with writes()
 */
template <typename... Models>
class MeasurementParts
{
public:

    /** The function of \a models side by side. */
    explicit MeasurementParts(Models... models)
        : m_parts{std::move(models)...}
    {}

    /** \returns the models, in order */
    [[nodiscard]] std::tuple<Models...> const &
    parts() const
    {
        return m_parts;
    }

    /** \returns the measurement of \a x, a vector in the whole's input space \a in, in \a out */
    template <typename In, typename Out, typename X>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x) const
        requires MatrixExpression<X>
    {
        return stepParts<ScalarOf<X>>(m_parts, in, out, x);
    }

    /// whether the whole takes a batch: a model's function does (decisions.md [batched-functions])
    static constexpr bool TakesBatches = anyTakesBatches<Models...>;

    /** The whole writes no Jacobian in a batch: a method that needs one differentiates it. */
    using MarkType = NoJacobians;

    /** Writes into \a y the measurement of a batch \a x, a column per point in the whole's input space
     * \a in, in \a out: each batched model called once over its columns, and each other model at each
     * column (stepPartsBatch())
     */
    template <typename In, typename Out, typename X, typename Y>
    void
    operator()(In const & in, Out const & out, X const & x, Y & y) const
        requires TakesBatches
    {
        stepPartsBatch(m_parts, in, out, x, y);
    }

private:

    std::tuple<Models...> m_parts;  ///< the models
};

/** The likelihood of measurement models side by side, each with its own: the sum of theirs, each at
 * its own input and its blocks of the observation.
 *
 * \tparam Out     the whole's output space, fixed-size
 * \tparam Models  the models, each with logLikelihood()
 */
template <typename Out, typename... Models>
struct SumOfLikelihoods
{
    std::tuple<Models...> parts;  ///< the models

    /** \returns \f$ \sum_i \log p(z_i \mid x) \f$ at \a x, a vector in the whole's input \a in */
    template <typename In, typename X, typename Z>
    [[nodiscard]] auto
    operator()(In const & in, X const & x, Z const & z) const
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return (likelihoodOf<I>(in, x, z) + ...);
        }(std::index_sequence_for<Models...>{});
    }

private:

    /** \returns model \a I's log-likelihood at its blocks of \a x and \a z */
    template <std::size_t I, typename In, typename X, typename Z>
    [[nodiscard]] auto
    likelihoodOf(In const & in, X const & x, Z const & z) const
    {
        using Part = std::tuple_element_t<I, std::tuple<Models...>>;
        using PartOut = OutputSpaceOf<Part>;
        auto const pin = partIn<Part>(in);
        auto const px = partVector<Part>(in, pin, x);
        Out const out{};
        PartOut const pout{};
        auto pz = zeros<typename Z::Scalar>(pout);
        [&]<typename... Names>(std::tuple<Names...> *) {
            ((block<Names>(pout, pz) = block<Names>(out, z)), ...);
        }(static_cast<CompositeBlocks<PartOut> *>(nullptr));
        return callWithScalar<typename X::Scalar>(
            std::get<I>(parts).template policy<PolicyKind::logLikelihood>().function,
            pin,
            px,
            pz
        );
    }
};

/** \returns the additive noise parts of the measurement model \a model, a part of a composition,
 * as a tuple of parts placed by tag path: its own placed parts, one unplaced distribution placed on
 * the one block it writes, or none
 */
template <typename Model>
[[nodiscard]] auto
placedNoiseParts(Model const & model)
{
    if constexpr (!Model::template has<PolicyKind::additiveNoise>)
    {
        return std::tuple<>{};
    }
    else
    {
        using Noise = typename Model::template Policy<PolicyKind::additiveNoise>;
        if constexpr (Noise::Placed)
        {
            return model.template policy<PolicyKind::additiveNoise>().parts();
        }
        else
        {
            static_assert(
                writeCount<Model> == 1,
                "a part of a composition whose additiveNoise() covers its whole output writes one block; place each "
                "part of its noise with on<Path>() to write several"
            );
            return std::tuple{on<WritePathAt<Model, 0>>(
                typename Noise::Distribution{model.template policy<PolicyKind::additiveNoise>().distribution()}
            )};
        }
    }
}

}  // namespace internal

/** \ingroup Models
 * \returns the process model \a model placed on the block of the state that \a Path (a tag or a
 * TagPath) names (decisions.md [model-placement]): every path into the state it reads and writes
 * is prefixed by \a Path, and its function is untouched. An input of the call keeps its tag, so
 * two placements of one model share it, unless \a renames rename it, each
 * <tt>on<Name>(path<Tag>)</tt>, which gives the input \a Tag the name \a Name.
 *
 * \code
 * auto const pair = compose(on<Ownship>(vehicle), on<Target>(vehicle, on<TargetForce>(path<SpecificForce>)));
 * \endcode
 */
template <typename Path, ProcessModel Model, typename... Renames>
[[nodiscard]] auto
on(Model const & model, Renames const &...)
{
    internal::refuseWholeState<Model>();
    using Prefix = internal::AsTagPath<Path>;
    using Placement = internal::Placement<Prefix, Model, Renames...>;
    using Step = internal::PlacedStep<Model>;
    return ProcessFunction<
        Step,
        typename Placement::ReadsPolicy,
        internal::PlacedWrites<Prefix, Model>,
        typename Placement::AffinePolicy>{
        Step{model},
        typename Placement::ReadsPolicy{},
        internal::PlacedWrites<Prefix, Model>{},
        typename Placement::AffinePolicy{}
    };
}

/** \ingroup Models
 * \returns the measurement model \a model placed on the block of the state that \a Path (a tag or
 * a TagPath) names (decisions.md [model-placement]): every path into the state it reads is prefixed
 * by \a Path, and an input of the call keeps its tag unless \a renames rename it, as on() of a
 * process model does. Its output is its own, since a measurement is not part of the state.
 */
template <typename Path, MeasurementModel Model, typename... Renames>
[[nodiscard]] auto
on(Model const & model, Renames const &...)
{
    internal::refuseWholeState<Model>();
    using Prefix = internal::AsTagPath<Path>;
    using Placement = internal::Placement<Prefix, Model, Renames...>;
    using InnerIn = typename Model::template Policy<internal::PolicyKind::reads>::InputSpace;
    using ReadsPolicy = typename Placement::ReadsPolicy;
    using AffinePolicy = typename Placement::AffinePolicy;
    auto const function = [&] {
        if constexpr (Model::HasFunction)
        {
            return internal::PlacedMeasure<typename Model::FunctionType, InnerIn>{model.function()};
        }
        else
        {
            return internal::NoFunction{};
        }
    }();
    return std::apply(
        [&](auto const &... policies) {
            return MeasurementFunction<
                std::remove_cvref_t<decltype(function)>,
                decltype(internal::placedPolicy<Prefix, InnerIn, ReadsPolicy, AffinePolicy>(policies))...>{
                function,
                internal::placedPolicy<Prefix, InnerIn, ReadsPolicy, AffinePolicy>(policies)...
            };
        },
        model.policies()
    );
}

/** \ingroup Models
 * \returns the process model \a model placed on every repetition of the repeated block of the
 * state that \a Path (a tag or a TagPath) names, as many as the state has (decisions.md
 * [model-placement]): each repetition moves by the model on its own blocks, with its own noise. An
 * input renamed by \a renames to a Repeated tag, <tt>on<Forces>(path<SpecificForce>)</tt> with
 * <tt>struct Forces : Repeated<SpecificForce, N></tt>, is one argument with a value per
 * repetition; any other input is shared by every repetition.
 *
 * \code
 * auto const fleet = compose(on<Ownship>(vehicle), onEach<Targets>(vehicle, on<TargetForces>(path<SpecificForce>)));
 * \endcode
 */
template <typename Path, ProcessModel Model, typename... Renames>
[[nodiscard]] auto
onEach(Model const & model, Renames const &...)
{
    internal::refuseWholeState<Model>();
    using Prefix = internal::AsTagPath<Path>;
    using Each = internal::EachPlacement<Prefix, Model, Renames...>;
    using Step = internal::EachStep<Prefix, Model, Renames...>;
    return ProcessFunction<Step, typename Each::ReadsOfEach, Writes<Prefix>, typename Each::AffineOfEach>{
        Step{model},
        typename Each::ReadsOfEach{},
        Writes<Prefix>{},
        typename Each::AffineOfEach{}
    };
}

namespace internal
{

/** \returns the measurement model of \a models side by side (see compose()) */
template <typename... Models>
[[nodiscard]] auto
composeMeasurements(Models... models)
{
    (refuseWholeState<Models>(), ...);
    static_assert(
        (Models::template has<PolicyKind::writes> && ...),
        "compose() takes measurement models that name their output with writes()"
    );
    static_assert(
        (Models::HasFunction && ...),
        "compose() takes measurement models with a function; one known by its likelihood alone has no output"
    );
    static_assert(
        (!Models::template has<PolicyKind::outputSpace> && ...),
        "compose() takes measurement models whose output writes() fixes, without outputSpace()"
    );
    constexpr std::size_t likelihoods = (std::size_t{Models::template has<PolicyKind::logLikelihood>} + ...);
    static_assert(
        likelihoods == 0 || likelihoods == sizeof...(Models),
        "in a composition of measurement models, every model gives logLikelihood() or none does"
    );
    using AllItems = decltype(std::tuple_cat(std::declval<ReadItemsOf<Models>>()...));
    using ReadsPolicy = typename ReadUnionOf<AllItems>::type;
    using WritesPolicy = typename WritesOfTuple<
        decltype(std::tuple_cat(std::declval<typename PathsOf<WritePathsOf<Models>>::type>()...)),
        decltype(std::tuple_cat(std::declval<CompositeBlocks<OutputSpaceOf<Models>>>()...))>::type;
    using AffinePolicy = typename AffineOf<ReadsPolicy, std::tuple<Models...>>::type;
    using Function = MeasurementParts<Models...>;
    auto const noise = std::tuple_cat(placedNoiseParts(models)...);
    auto const base = std::tuple{ReadsPolicy{}, WritesPolicy{}, AffinePolicy{}};
    auto const withNoise = [&] {
        if constexpr (std::tuple_size_v<std::remove_cvref_t<decltype(noise)>> > 0)
        {
            return std::tuple_cat(
                base,
                std::tuple{std::apply(
                    [](auto const &... parts) {
                        return AdditiveNoise<std::remove_cvref_t<decltype(parts)>...>{parts...};
                    },
                    noise
                )}
            );
        }
        else
        {
            return base;
        }
    }();
    auto const policies = [&] {
        if constexpr (likelihoods > 0)
        {
            using Sum = SumOfLikelihoods<typename WritesPolicy::OutputSpace, Models...>;
            return std::tuple_cat(withNoise, std::tuple{LogLikelihood<Sum>{Sum{std::tuple{models...}}}});
        }
        else
        {
            return withNoise;
        }
    }();
    return std::apply(
        [&](auto const &... each) {
            return MeasurementFunction<Function, std::remove_cvref_t<decltype(each)>...>{
                Function{std::move(models)...},
                each...
            };
        },
        policies
    );
}

}  // namespace internal

/** \ingroup Models
 * \returns the measurement model of \a models side by side (decisions.md [model-placement]): its
 * reads are the union of theirs, gathered once, its output the blocks they write together, which
 * must be distinct, its noise their noise on their blocks, and its log-likelihood, where every one
 * has its own, their sum. Each must name its output with writes().
 *
 * \code
 * auto const radar = compose(range, bearing);  // measures Composite<Range, Bearing>
 * \endcode
 */
template <MeasurementModel... Models>
    requires(sizeof...(Models) > 0)
[[nodiscard]] auto
compose(Models &&... models)
{
    return internal::composeMeasurements(std::remove_cvref_t<Models>{std::forward<Models>(models)}...);
}

}  // namespace Emrys
