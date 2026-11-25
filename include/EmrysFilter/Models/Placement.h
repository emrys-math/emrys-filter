#pragma once

/** \file
 * Models placed in a larger state and measurement models side by side (decisions.md [model-placement],
 * [model-policies], [composed-selections]): on<Path>(model), which places a process or a measurement model on the block
 * that a path names, prefixing every path into the state it reads and writes, each with the selection of repetitions it
 * holds; onEach<Path>(model), which places one copy of a process model on every repetition of a repeated block; and
 * compose() of measurement models, whose output is their outputs side by side. compose() of process models is in
 * ProcessModel.h. Read Measurement.h next.
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

/** The names of the SharedJacobians \a S as a tuple type, or none for NoPolicy. */
template <typename S>
struct SharedNamesOf
{
    using type = std::tuple<>;  ///< none
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Names>
struct SharedNamesOf<SharedJacobians<Names...>>
{
    using type = std::tuple<Names...>;
};

/// \endcond

/** The names of the model \a Model whose Jacobian is shared, as a tuple type. */
template <typename Model>
using SharedNamesOfModel = typename SharedNamesOf<typename Model::template Policy<PolicyKind::sharedJacobians>>::type;

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

/** The block a model placed on the TagPath \a Prefix with the selection \a Selection is placed on,
 * as \c type: the block the path names, or the block that is repeated where the path names a
 * repeated block and \a Selection selects one of its repetitions.
 */
template <typename Prefix, typename Selection>
struct PlacedBlockOf
{
    using type = PathTag<Prefix>;  ///< the block the path names
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Prefix, typename Selection>
    requires(!std::is_same_v<Selection, NoSelection> && IsRepeated<PathTag<Prefix>>)
struct PlacedBlockOf<Prefix, Selection>
{
    using type = RepeatedBlock<PathTag<Prefix>>;
};

/// \endcond

/** PlacedBlockOf, as a type. */
template <typename Prefix, typename Selection>
using PlacedBlock = typename PlacedBlockOf<Prefix, Selection>::type;

/** \returns the space of the block in the state's \a space that a model placed on the TagPath
 * \a Prefix with the selection \a Selection is placed on: the block the path names, or, for one
 * repetition, its space inside the block that is repeated, which every repetition shares
 */
template <typename Prefix, typename Selection, typename Space>
[[nodiscard]] auto
placedSubspace(Space const & space)
{
    if constexpr (std::is_same_v<Selection, NoSelection>)
    {
        return pathSubspace<Prefix>(space);
    }
    else
    {
        return pathElementSubspace<Prefix>(space);
    }
}

/** How the model \a Model, placed on the block that the TagPath \a Prefix names with the renames
 * \a Renames, reads and writes: each item of its reads() that starts in the block is a block of the
 * state, its path prefixed, its name a PlacedName of its own, and its selection of repetitions, where
 * it has one, the model's; each other item is kept as it is, path, name, and selection, so it is read
 * from the state where the state has it and is an input of the call by its one tag otherwise
 * (decisions.md [model-placement]); an item a rename renames is the new name's, by that tag;
 * and each block written is prefixed. Where \a Selection is an integer, the model is placed on one
 * repetition of the repeated block \a Prefix names, or on the block \a Prefix names inside it, and
 * every item in the block selects that repetition.
 */
template <typename Prefix, typename Selection, typename Model, typename... Renames>
struct Placement
{
    /// whether the model is placed on one repetition, which every item in its block then selects
    static constexpr bool Selected = !std::is_same_v<Selection, NoSelection>;

    using PrefixSpace = PlacedBlock<Prefix, Selection>;                     ///< the block the model is placed on
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

        /// whether a rename renames the item, which is then read by the new name's tag
        static constexpr bool IsRenamed = !State && !std::is_same_v<New, typename Item::NameType>;

        static_assert(
            State || sameLeaf<New, typename Item::NameType>,
            "a renamed input has the same leaf beneath it as the name the model reads it under"
        );
        static_assert(
            !IsRenamed || !Item::Selected,
            "on<Path>(model, on<Name>(path<Tag>)) renames an input of the call, which has no repetitions to select, "
            "so the item it renames selects none"
        );

        /** The item. */
        using type = std::conditional_t<
            State,
            ReadItem<
                JoinPaths<Prefix, typename Item::PathType>,
                PlacedName<Prefix, typename Item::NameType>,
                std::conditional_t<Selected, Selection, typename Item::SelectionType>>,
            std::conditional_t<
                IsRenamed,
                ReadItem<TagPath<New>, New>,
                ReadItem<typename Item::PathType, typename Item::NameType, typename Item::SelectionType>>>;
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

    /** The placed names whose Jacobian is shared, renamed as the affine ones are. */
    using SharedPolicy = typename decltype([]<typename... Names>(std::tuple<Names...> *) {
        return std::type_identity<SharedJacobians<OuterName<Names>...>>{};
    }(static_cast<SharedNamesOfModel<Model> *>(nullptr)))::type;
};

/** \returns the reads() policy \a Target of a model placed on a block, from the model's \a reads:
 * each item holds the model's selection, or, where the model is placed on one repetition, \a where,
 * the repetition, if the item is in it, which is where the model's item selects none and the placed
 * one does
 */
template <typename Target, typename Selection, typename... Items>
[[nodiscard]] Target
placedReads(Reads<Items...> const & reads, [[maybe_unused]] Selection const & where)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        auto const selectionOf = [&]<std::size_t K>(std::integral_constant<std::size_t, K>) ->
            typename Target::template Item<K>::SelectionType {
                if constexpr (!std::is_same_v<Selection, NoSelection> && Target::template Item<K>::Selected
                              && !Reads<Items...>::template Item<K>::Selected)
                {
                    return where;
                }
                else
                {
                    return reads.template selection<K>();
                }
            };
        return Target{std::in_place, selectionOf(std::integral_constant<std::size_t, I>{})...};
    }(std::index_sequence_for<Items...>{});
}

/** The function of a process model placed on a block (see on()): the model's own function, called
 * with the model's input and output spaces under the names it reads and writes them by, on the
 * input vector as it is, since the placed model reads the same blocks in the same order. Its
 * noise is the model's.
 *
 * \tparam Prefix     the path to the block the model is placed on, a TagPath
 * \tparam Model      the model placed
 * \tparam Selection  NoSelection, or an integer type where the model is placed on one repetition
 */
template <typename Prefix, typename Model, typename Selection = NoSelection>
class PlacedStep
{
public:

    /// whether the model placed holds one step (holdsOneStep)
    static constexpr bool HoldsOneStep = holdsOneStep<Model>;

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

    /** \returns the model's step of \a x, the value alone, the model's function asked for its value
     * alone (callValueForm())
     */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    valueAt(In const & in, Out const & out, X const & x, TimeScalar dt) const
        requires MatrixExpression<X>
    {
        return callValueForm(m_model.function(), renamedInstance<InnerIn>(in), renamedInstance<InnerOut>(out), x, dt);
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

    /** \returns whether the parts of the model placed, where it is a composition, select the same
     * repetitions under one name in its block of the state's \a space (internal::selectionsAgree())
     */
    template <typename Space>
    [[nodiscard]] bool
    selectionsAgree(Space const & space) const
    {
        return internal::selectionsAgree(m_model, placedSubspace<Prefix, Selection>(space));
    }

private:

    Model m_model;  ///< the model placed
};

/** The placed writes() of the process model \a Model on the block that \a Prefix names: each path
 * prefixed, each name a PlacedName, and each selection of repetitions the model's, or, where the
 * model is placed on one repetition (\a Selection an integer type), that repetition.
 */
template <typename Prefix, typename Model, typename Selection = NoSelection>
using PlacedWrites = typename decltype([]<std::size_t... K>(std::index_sequence<K...>) {
    static_assert(
        (startsIn<WritePathAt<Model, K>, PlacedBlock<Prefix, Selection>> && ...),
        "on<Path>(model) places a model whose writes() are blocks of the block Path names"
    );
    if constexpr (!std::is_same_v<Selection, NoSelection>)
    {
        return std::type_identity<WritesItems<ReadItem<
            JoinPaths<Prefix, WritePathAt<Model, K>>,
            PlacedName<Prefix, WriteNameAt<Model, K>>,
            Selection>...>>{};
    }
    else if constexpr (writesSelected<Model>)
    {
        return std::type_identity<WritesItems<ReadItem<
            JoinPaths<Prefix, WritePathAt<Model, K>>,
            PlacedName<Prefix, WriteNameAt<Model, K>>,
            typename WriteItemAt<Model, K>::SelectionType>...>>{};
    }
    else
    {
        return std::type_identity<WritesAs<
            TagPaths<JoinPaths<Prefix, WritePathAt<Model, K>>...>,
            Composite<PlacedName<Prefix, WriteNameAt<Model, K>>...>>>{};
    }
}(std::make_index_sequence<writeCount<Model>>{}))::type;

/** \returns the placed writes() of \a model on the block that \a Prefix names (PlacedWrites), with
 * the selections that the model's writes() holds, or with \a where, the repetition the model is
 * placed on
 */
template <typename Prefix, typename Model, typename Selection = NoSelection>
[[nodiscard]] PlacedWrites<Prefix, Model, Selection>
placedWrites(Model const & model, [[maybe_unused]] Selection const & where = {})
{
    if constexpr (!std::is_same_v<Selection, NoSelection>)
    {
        return [&]<std::size_t... K>(std::index_sequence<K...>) {
            return PlacedWrites<Prefix, Model, Selection>{std::in_place, ((void)K, where)...};
        }(std::make_index_sequence<writeCount<Model>>{});
    }
    else if constexpr (writesSelected<Model>)
    {
        return std::apply(
            [](auto const &... selections) {
                return PlacedWrites<Prefix, Model>{std::in_place, selections...};
            },
            writeSelectionsOf(model)
        );
    }
    else
    {
        return {};
    }
}

/** The kinds of item of a process model placed on every repetition of a repeated block. */
enum class EachKind
{
    state,     ///< a block of one repetition
    shared,    ///< an item outside one repetition, which every repetition reads
    repeated,  ///< an input with a value per repetition, renamed to a Repeated tag
};

/** How the process model \a Model, placed on every repetition of the repeated block that the
 * TagPath \a Prefix names with the renames \a Renames, reads: the whole repeated block once, under
 * its tag, in place of the items that start in one repetition; each input renamed to a Repeated tag
 * as one argument with a value per repetition; and each other item as it is, path and selection,
 * read from the state where the state has it, shared by every repetition (decisions.md
 * [model-placement]). Where \a Selection is a sequence or a list, the model
 * is placed on each of the repetitions it selects, read and written as a repeated block of their
 * count under the block's tag.
 */
template <typename Prefix, typename Model, typename Selection, typename... Renames>
struct EachPlacement
{
    /// whether the model is placed on selected repetitions, not on every one
    static constexpr bool Selected = !std::is_same_v<Selection, NoSelection>;

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
        !writesSelected<Model>,
        "onEach<Path>(model) places a model on one repetition of a repeated block, which holds no repeated block, so "
        "the model's writes() select no repetitions"
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
            static_assert(
                !InnerItem<K>::Selected,
                "onEach<Path>(model, on<Name>(path<Tag>)) renames an input of the call, which has no repetitions to "
                "select, so the item it renames selects none"
            );
            return EachKind::repeated;
        }
        else
        {
            static_assert(
                sameLeaf<NewName<K>, typename InnerItem<K>::NameType>,
                "a renamed input has the same leaf beneath it as the name the model reads it under"
            );
            static_assert(
                std::is_same_v<NewName<K>, typename InnerItem<K>::NameType> || !InnerItem<K>::Selected,
                "onEach<Path>(model, on<Name>(path<Tag>)) renames an input of the call, which has no repetitions to "
                "select, so the item it renames selects none"
            );
            return EachKind::shared;
        }
    }

    /** Item \a K of the model outside one repetition, as the placed reads() holds it: as it is, path,
     * name, and selection, read from the state where the state has it (decisions.md
     * [model-placement]), or, renamed, by the new name's tag.
     */
    template <std::size_t K>
    using OuterItem = std::conditional_t<
        std::is_same_v<NewName<K>, typename InnerItem<K>::NameType>,
        ReadItem<typename InnerItem<K>::PathType, NewName<K>, typename InnerItem<K>::SelectionType>,
        ReadItem<TagPath<NewName<K>>, NewName<K>>>;

    /** The placed reads(): the repeated block, or the repetitions selected, then each item outside
     * one repetition.
     */
    using ReadsPolicy = typename decltype([]<std::size_t... K>(std::index_sequence<K...>) {
        return std::type_identity<decltype(std::tuple_cat(
            std::tuple<ReadItem<Prefix, RepeatedTag, Selection>>{},
            std::conditional_t<kind<K>() == EachKind::state, std::tuple<>, std::tuple<OuterItem<K>>>{}...
        ))>{};
    }(std::make_index_sequence<ItemCount>{}))::type;

    /** The placed reads(), as a Reads. */
    using ReadsOfEach = typename decltype([]<typename... Items>(std::tuple<Items...> *) {
        return std::type_identity<Reads<Items...>>{};
    }(static_cast<ReadsPolicy *>(nullptr)))::type;

    /** The placed writes(): the repeated block, or the repetitions selected, under its tag. */
    using WritesOfEach =
        std::conditional_t<Selected, WritesItems<ReadItem<Prefix, RepeatedTag, Selection>>, Writes<Prefix>>;

    /** \returns the placed reads(), whose first item holds \a where, the repetitions selected, and
     * each item outside one repetition its selection in the model's \a reads
     */
    [[nodiscard]] static ReadsOfEach
    makeReads(InnerReads const & reads, [[maybe_unused]] Selection const & where)
    {
        if constexpr (!ReadsOfEach::Selected)
        {
            return {};
        }
        else
        {
            return [&]<std::size_t... K>(std::index_sequence<K...>) {
                return std::apply(
                    [&](auto const &... selections) {
                        return ReadsOfEach{std::in_place, where, selections...};
                    },
                    std::tuple_cat(outerSelection<K>(reads)...)
                );
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    /** \returns the selection the placed reads() holds for item \a K of the model's \a reads, as a
     * tuple: none for an item in one repetition, the model's for an item outside it, and NoSelection
     * for an item renamed
     */
    template <std::size_t K>
    [[nodiscard]] static auto
    outerSelection([[maybe_unused]] InnerReads const & reads)
    {
        if constexpr (kind<K>() == EachKind::state)
        {
            return std::tuple<>{};
        }
        else if constexpr (std::is_same_v<NewName<K>, typename InnerItem<K>::NameType>)
        {
            return std::tuple{reads.template selection<K>()};
        }
        else
        {
            return std::tuple{NoSelection{}};
        }
    }

    /** \returns the placed writes(), which holds \a where, the repetitions selected */
    [[nodiscard]] static WritesOfEach
    makeWrites([[maybe_unused]] Selection const & where)
    {
        if constexpr (Selected)
        {
            return WritesOfEach{std::in_place, where};
        }
        else
        {
            return {};
        }
    }

    static_assert(
        std::tuple_size_v<SharedNamesOfModel<Model>> == 0,
        "onEach() places a model whose affineIn() shares no Jacobian, shared(path<Name>), yet; give the block with "
        "path<Name>"
    );

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
 * \tparam Prefix     the path to the repeated block, a TagPath
 * \tparam Model      the model placed
 * \tparam Selection  NoSelection, or the sequence or list of repetitions it is placed on
 * \tparam Renames    the renames of its inputs
 */
template <typename Prefix, typename Model, typename Selection, typename... Renames>
class EachStep
{
public:

    /// whether the model placed holds one step (holdsOneStep)
    static constexpr bool HoldsOneStep = holdsOneStep<Model>;

private:

    /** how the model is placed */
    using P = EachPlacement<Prefix, Model, Selection, Renames...>;
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
        auto & values = Emrys::values(y);
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
 * \tparam Prefix     the path to the block the model is placed on, a TagPath
 * \tparam F          the model's function
 * \tparam InnerIn    the model's input space
 * \tparam Selection  NoSelection, or an integer type where the model is placed on one repetition
 */
template <typename Prefix, typename F, typename InnerIn, typename Selection = NoSelection>
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

    /** \returns whether the parts of the function, where it is a composition's, select the same
     * repetitions under one name in its block of the state's \a space (internal::selectionsAgree())
     */
    template <typename Space>
    [[nodiscard]] bool
    selectionsAgree(Space const & space) const
        requires requires(F const & f) { f.selectionsAgree(placedSubspace<Prefix, Selection>(space)); }
    {
        return m_function.selectionsAgree(placedSubspace<Prefix, Selection>(space));
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
 * model's own, called with the space of that block, since the model was written for it; for one
 * repetition (\a Selection an integer type), the space inside the block that is repeated.
 */
template <typename Prefix, typename G, typename Selection = NoSelection>
struct PlacedOutputSpace
{
    G function;  ///< the model's output space function

    /** \returns the output space in the state's space \a space */
    template <typename StateSpace>
    [[nodiscard]] auto
    operator()(StateSpace const & space) const
    {
        return function(placedSubspace<Prefix, Selection>(space));
    }
};

/** \returns the policy \a policy of a measurement model placed on the block that \a Prefix names,
 * with reads \a ReadsPolicy and affine names \a AffinePolicy: the reads() placed, holding the
 * selections that the model's holds or \a where, the repetition it is placed on (placedReads()), the
 * affineIn() placed, a likelihood, a draw, or an output space function given the placed input or the
 * block's space, and anything else as it is
 */
template <
    typename Prefix,
    typename InnerIn,
    typename ReadsPolicy,
    typename AffinePolicy,
    typename SharedPolicy,
    typename Policy,
    typename Selection>
[[nodiscard]] auto
placedPolicy(Policy const & policy, [[maybe_unused]] Selection const & where)
{
    constexpr std::optional<PolicyKind> kind = policyKind<Policy>;
    if constexpr (kind == PolicyKind::reads)
    {
        return placedReads<ReadsPolicy>(policy, where);
    }
    else if constexpr (kind == PolicyKind::affineIn)
    {
        return AffinePolicy{};
    }
    else if constexpr (kind == PolicyKind::sharedJacobians)
    {
        return SharedPolicy{};
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
            return OutputSpace<PlacedOutputSpace<Prefix, G, Selection>>{
                PlacedOutputSpace<Prefix, G, Selection>{policy.value}
            };
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

    /** \returns whether the models that read repetitions of a block of \a space under one name select
     * the same ones (partsSelectionsAgree()), which the whole asserts when it meets a state
     */
    template <typename Space>
    [[nodiscard]] bool
    selectionsAgree(Space const & space) const
    {
        return partsSelectionsAgree(space, m_parts);
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

/** \returns the process model \a model placed on the block that \a Prefix names with the renames
 * \a renames (see on()), or on its repetition \a where where that is an integer
 */
template <typename Prefix, typename Selection, typename Model, typename... Renames>
[[nodiscard]] auto
placeProcess(Selection const & where, Model const & model, Renames const &...)
{
    static_assert(wholeStateRefused<Model>);
    using Placement = internal::Placement<Prefix, Selection, Model, Renames...>;
    using Step = PlacedStep<Prefix, Model, Selection>;
    return ProcessFunction<
        Step,
        typename Placement::ReadsPolicy,
        PlacedWrites<Prefix, Model, Selection>,
        typename Placement::AffinePolicy,
        typename Placement::SharedPolicy>{
        Step{model},
        placedReads<typename Placement::ReadsPolicy>(model.template policy<PolicyKind::reads>(), where),
        placedWrites<Prefix>(model, where),
        typename Placement::AffinePolicy{},
        typename Placement::SharedPolicy{}
    };
}

/** \returns the measurement model \a model placed on the block that \a Prefix names with the
 * renames \a renames (see on()), or on its repetition \a where where that is an integer
 */
template <typename Prefix, typename Selection, typename Model, typename... Renames>
[[nodiscard]] auto
placeMeasurement(Selection const & where, Model const & model, Renames const &...)
{
    static_assert(wholeStateRefused<Model>);
    using Placement = internal::Placement<Prefix, Selection, Model, Renames...>;
    using InnerIn = typename Model::template Policy<PolicyKind::reads>::InputSpace;
    using ReadsPolicy = typename Placement::ReadsPolicy;
    using AffinePolicy = typename Placement::AffinePolicy;
    using SharedPolicy = typename Placement::SharedPolicy;
    auto const function = [&] {
        if constexpr (Model::HasFunction)
        {
            return PlacedMeasure<Prefix, typename Model::FunctionType, InnerIn, Selection>{model.function()};
        }
        else
        {
            return NoFunction{};
        }
    }();
    return std::apply(
        [&](auto const &... policies) {
            return MeasurementFunction<
                std::remove_cvref_t<decltype(function)>,
                decltype(placedPolicy<Prefix, InnerIn, ReadsPolicy, AffinePolicy, SharedPolicy>(policies, where))...>{
                function,
                placedPolicy<Prefix, InnerIn, ReadsPolicy, AffinePolicy, SharedPolicy>(policies, where)...
            };
        },
        model.policies()
    );
}

/** \returns the process model \a model placed on every repetition of the repeated block that
 * \a Prefix names with the renames \a renames (see onEach()), or on each that \a where selects
 */
template <typename Prefix, typename Selection, typename Model, typename... Renames>
[[nodiscard]] auto
placeEach(Selection const & where, Model const & model, Renames const &...)
{
    static_assert(wholeStateRefused<Model>);
    using Each = EachPlacement<Prefix, Model, Selection, Renames...>;
    using Step = EachStep<Prefix, Model, Selection, Renames...>;
    return ProcessFunction<Step, typename Each::ReadsOfEach, typename Each::WritesOfEach, typename Each::AffineOfEach>{
        Step{model},
        Each::makeReads(model.template policy<PolicyKind::reads>(), where),
        Each::makeWrites(where),
        typename Each::AffineOfEach{}
    };
}

}  // namespace internal

/** \ingroup Models
 * \returns the process model \a model placed on the block of the state that \a Path (a tag or a
 * TagPath) names (decisions.md [model-placement]): every path it reads that starts in that block,
 * and every path it writes, is prefixed by \a Path, and its function is untouched. An item that
 * selects repetitions of a repeated block in the model's block selects the same ones where it is
 * placed. Any other item keeps its path and selection, read from the state where the state has it,
 * <tt>path<Fleet>[0]</tt>, and otherwise an input of the call by its tag, so two placements of one
 * model share it, unless \a renames rename it, each <tt>on<Name>(path<Tag>)</tt>, which gives the
 * input \a Tag the name \a Name.
 *
 * \code
 * auto const pair = compose(on<Ownship>(vehicle), on<Target>(vehicle, on<TargetForce>(path<SpecificForce>)));
 * \endcode
 */
template <typename Path, ProcessModel Model, typename... Renames>
[[nodiscard]] auto
on(Model const & model, Renames const &... renames)
{
    return internal::placeProcess<internal::AsTagPath<Path>>(internal::NoSelection{}, model, renames...);
}

/** \ingroup Models
 * \returns the process model \a model placed on one repetition of a repeated block of the state,
 * the one that \a where selects, <tt>path<Fleet>[k]</tt>, or on a block inside it,
 * <tt>path<Fleet, Body>[k]</tt> (decisions.md [model-placement], [placed-repetitions]): as
 * on<Path>(model) places it, each path into the state it reads and writes running through the
 * repetition. Its items in that block select no repetitions; one outside it is kept, as on<Path>(model)
 * keeps it, so the model may read another repetition, <tt>path<Fleet>[j]</tt>. Two placements of models
 * that read or write blocks under one name do so under one name here too, so a composition of them
 * reads one repetition under it; onEach(path<Fleet>[repetitions], model) places a model on several.
 *
 * \code
 * auto const fleet = compose(on(path<Fleet>[k], steered), onEach(path<Fleet>[others], cruising));
 * \endcode
 */
template <typename Path, typename Selection, ProcessModel Model, typename... Renames>
[[nodiscard]] auto
on(SelectedPath<Path, Selection> const & where, Model const & model, Renames const &... renames)
{
    static_assert(
        internal::RepetitionIndex<Selection>,
        "on(path<Block>[k], model) places a model on one repetition; onEach(path<Block>[repetitions], model) places "
        "it on each of several"
    );
    return internal::placeProcess<Path>(where.selection, model, renames...);
}

/** \ingroup Models
 * \returns the measurement model \a model placed on the block of the state that \a Path (a tag or
 * a TagPath) names (decisions.md [model-placement]): every path into the state it reads is prefixed
 * by \a Path, and any other item keeps its path and selection, read from the state where the state
 * has it and otherwise an input of the call by its tag unless \a renames rename it, as on() of a
 * process model does. Its output is its own, since a measurement is not part of the state.
 */
template <typename Path, MeasurementModel Model, typename... Renames>
[[nodiscard]] auto
on(Model const & model, Renames const &... renames)
{
    return internal::placeMeasurement<internal::AsTagPath<Path>>(internal::NoSelection{}, model, renames...);
}

/** \ingroup Models
 * \returns the measurement model \a model placed on one repetition of a repeated block of the
 * state, the one that \a where selects, <tt>path<Landmarks>[k]</tt>, or on a block inside it
 * (decisions.md [model-placement], [placed-repetitions]): as on<Path>(model) places it, each path
 * into the state it reads running through the repetition. Its items in that block select no
 * repetitions; one outside it is kept, as on<Path>(model) keeps it.
 */
template <typename Path, typename Selection, MeasurementModel Model, typename... Renames>
[[nodiscard]] auto
on(SelectedPath<Path, Selection> const & where, Model const & model, Renames const &... renames)
{
    static_assert(
        internal::RepetitionIndex<Selection>,
        "on(path<Block>[k], model) places a model on one repetition; a measurement of several is a composition of "
        "placements, or a model whose reads() selects them"
    );
    return internal::placeMeasurement<Path>(where.selection, model, renames...);
}

/** \ingroup Models
 * \returns the process model \a model placed on every repetition of the repeated block of the
 * state that \a Path (a tag or a TagPath) names, as many as the state has (decisions.md
 * [model-placement]): each repetition moves by the model on its own blocks, with its own noise. An
 * input renamed by \a renames to a Repeated tag, <tt>on<Forces>(path<SpecificForce>)</tt> with
 * <tt>struct Forces : Repeated<SpecificForce, N></tt>, is one argument with a value per
 * repetition; any other item outside one repetition keeps its path and selection, read from the
 * state where the state has it, and is shared by every repetition.
 *
 * \code
 * auto const fleet = compose(on<Ownship>(vehicle), onEach<Targets>(vehicle, on<TargetForces>(path<SpecificForce>)));
 * \endcode
 */
template <typename Path, ProcessModel Model, typename... Renames>
[[nodiscard]] auto
onEach(Model const & model, Renames const &... renames)
{
    return internal::placeEach<internal::AsTagPath<Path>>(internal::NoSelection{}, model, renames...);
}

/** \ingroup Models
 * \returns the process model \a model placed on each repetition that \a where selects of a repeated
 * block of the state, a sequence or a list, <tt>path<Fleet>[moving]</tt> (decisions.md
 * [model-placement], [placed-repetitions]): as onEach<Path>(model) places it on every repetition,
 * the repetitions read and written as a repeated block of their count under the block's tag, so that
 * an input renamed to a Repeated tag has a value per repetition selected, in their order. The
 * repetitions not selected are left to other parts of a composition.
 *
 * \code
 * auto const fleet = compose(on(path<Fleet>[k], steered), onEach(path<Fleet>[others], cruising));
 * \endcode
 */
template <typename Path, typename Selection, ProcessModel Model, typename... Renames>
[[nodiscard]] auto
onEach(SelectedPath<Path, Selection> const & where, Model const & model, Renames const &... renames)
{
    static_assert(
        !internal::RepetitionIndex<Selection>,
        "onEach(path<Block>[repetitions], model) places a model on each of several repetitions, a sequence or a "
        "list; on(path<Block>[k], model) places it on one"
    );
    static_assert(
        internal::IsRepeated<internal::PathTag<Path>>,
        "onEach(path<Block>[repetitions], model) places a model on repetitions of the repeated block the path names, "
        "so the path ends at it"
    );
    return internal::placeEach<Path>(where.selection, model, renames...);
}

namespace internal
{

/** \returns the measurement model of \a models side by side (see compose()) */
template <typename... Models>
[[nodiscard]] auto
composeMeasurements(Models... models)
{
    static_assert((wholeStateRefused<Models> && ...));
    // A whole-state model has no writes() either, which its own refusal says.
    static_assert(
        ((Models::template has<PolicyKind::writes> || readsWholeState<Models>) && ...),
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
    using SharedPolicy = typename SharedOf<ReadsPolicy, std::tuple<Models...>>::type;
    using Function = MeasurementParts<Models...>;
    auto const noise = std::tuple_cat(placedNoiseParts(models)...);
    auto const base = std::tuple{unionReads<ReadsPolicy>(models...), WritesPolicy{}, AffinePolicy{}, SharedPolicy{}};
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
 * reads are the union of theirs, gathered once, with the selection of repetitions of the first model
 * that reads each name, which every model reading that name must hold, its output the blocks they
 * write together, which
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
