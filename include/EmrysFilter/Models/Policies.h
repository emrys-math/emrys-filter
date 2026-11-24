#pragma once

/** \file
 * The policies a model is built from, and the marks on a step's arguments (decisions.md [model-policies],
 * [tag-addressing]): reads(), writes(), affineIn(), additiveNoise(), continuousNoise(), logLikelihood(), and
 * outputSpace(), each a call returning an object of its own type, which processModel() and measurementModel() take
 * after the function in any order; the internal PolicySet, which finds each in an argument pack; the placing of
 * additive noise in an output space; and sure(), which marks a known input of predict() or observe() by the tag of
 * its space. Read MeasurementModel.h next.
 */

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Models/ByMode.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Selection.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** One item of reads(): the block that the path \a Path (a TagPath) names, or the repetitions
 * \a Selection selects of the repeated block it runs through (decisions.md [repetition-selections]),
 * read under the name \a Name, a tag, which is the path's last tag unless on<Name>() gave another.
 */
template <typename Path, typename Name, typename Selection = NoSelection>
struct ReadItem
{
    using PathType = Path;            ///< the path read, a TagPath
    using NameType = Name;            ///< the name it is read under, a tag
    using SelectionType = Selection;  ///< the repetitions selected, as a SelectedPath holds them, or NoSelection

    /// whether the item reads selected repetitions of a repeated block, not a whole block
    static constexpr bool Selected = !std::is_same_v<Selection, NoSelection>;

    /// whether the path has one tag and no selection, the only form that can name an input of the call
    static constexpr bool SingleTag = !Selected && std::is_same_v<Path, TagPath<Name>>;
};

/** What a selected path has no tag to be read under, so that it needs on<Name>(). */
struct NoName
{};

/** The tag that the selection \a Selection of the path \a Path is read under where on<Name>() gives
 * none, as \c type: the tag of one repetition, or the path's last tag inside it, for an integer;
 * the repeated block's own tag for several repetitions of it; and NoName where that space has no
 * tag, as a repetition declared as a library space, or a block inside each of several repetitions.
 */
template <typename Path, typename Selection, typename Last = PathTag<Path>>
struct SelectedNameOf
{
    /** The path's last tag, for one repetition's block inside it; none for that block of several. */
    using type = std::conditional_t<RepetitionIndex<Selection>, Last, NoName>;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Path, typename Selection, IsRepeated Last>
struct SelectedNameOf<Path, Selection, Last>
{
    using type = std::conditional_t<
        !RepetitionIndex<Selection>,
        Last,
        std::conditional_t<IsUntagged<RepeatedBlock<Last>>, NoName, RepeatedBlock<Last>>>;
};

/// \endcond

/** What an argument of reads() of type \a T denotes: nothing, for anything but a path value, a
 * selected path, or an on<Name>() of one.
 */
template <typename T>
struct ReadItemOf
{
    static constexpr bool valid = false;  ///< not an item
};

/** A path value: the path, read under its last tag. */
template <PathValue T>
struct ReadItemOf<T>
{
    static constexpr bool valid = true;                    ///< an item
    using type = ReadItem<PathOf<T>, PathTag<PathOf<T>>>;  ///< the item
};

/** on<Name>(path): the path, read under \a Name. */
template <PathValue X, typename Name>
struct ReadItemOf<OnPath<X, Name>>
{
    static constexpr bool valid = true;      ///< an item
    using type = ReadItem<PathOf<X>, Name>;  ///< the item
};

/** A selected path, <tt>path<Landmarks>[k]</tt>: the repetitions it selects, read under the tag of
 * their space (SelectedNameOf).
 */
template <typename Path, typename Selection>
struct ReadItemOf<SelectedPath<Path, Selection>>
{
    static constexpr bool valid = true;                                                      ///< an item
    using type = ReadItem<Path, typename SelectedNameOf<Path, Selection>::type, Selection>;  ///< the item
};

/** on<Name>() of a selected path: the repetitions it selects, read under \a Name. */
template <typename Path, typename Selection, typename Name>
struct ReadItemOf<OnPath<SelectedPath<Path, Selection>, Name>>
{
    static constexpr bool valid = true;            ///< an item
    using type = ReadItem<Path, Name, Selection>;  ///< the item
};

/** \returns the selection that \a item, an argument of reads(), brings: a selected path's, or none */
template <typename T>
[[nodiscard]] constexpr NoSelection
readSelection(T const &)
{
    return {};
}

/** A selected path's selection. */
template <typename Path, typename Selection>
[[nodiscard]] Selection const &
readSelection(SelectedPath<Path, Selection> const & item)
{
    return item.selection;
}

/** The selection of a selected path given a name. */
template <typename Path, typename Selection, typename Name>
[[nodiscard]] Selection const &
readSelection(OnPath<SelectedPath<Path, Selection>, Name> const & item)
{
    return item.value.selection;
}

/** The selection of the item at position \a I of a reads(), held in a base of its own, so that an
 * item without one adds nothing to the policy's size.
 */
template <std::size_t I, typename Selection>
struct ItemSelection
{
    /** Holds \a selection. */
    explicit ItemSelection(Selection selection)
        : value{std::move(selection)}
    {}

    Selection value;  ///< the repetitions selected
};

/** An item that reads its whole block holds nothing. */
template <std::size_t I>
struct ItemSelection<I, NoSelection>
{
    constexpr ItemSelection() = default;

    /** Holds nothing. */
    constexpr explicit ItemSelection(NoSelection) {}

    static constexpr NoSelection value{};  ///< no selection
};

/** The selections of the items \a Items of a reads(), each in a base at its position, \a I. */
template <typename Positions, typename... Items>
struct ReadSelections;

/// \cond INTERNAL_SPECIALIZATIONS
template <std::size_t... I, typename... Items>
struct ReadSelections<std::index_sequence<I...>, Items...> : ItemSelection<I, typename Items::SelectionType>...
{
    constexpr ReadSelections() = default;

    /** Holds each item's selection. */
    explicit ReadSelections(std::in_place_t, typename Items::SelectionType const &... selections)
        : ItemSelection<I, typename Items::SelectionType>{selections}...
    {}
};

/// \endcond

/** Whether the types \a Ts are pairwise distinct. */
template <typename... Ts>
consteval bool
distinctTypes()
{
    if constexpr (sizeof...(Ts) < 2)
    {
        return true;
    }
    else
    {
        return []<typename First, typename... Rest>(std::type_identity<First>, std::type_identity<Rest>...) {
            return (!std::is_same_v<First, Rest> && ...) && distinctTypes<Rest...>();
        }(std::type_identity<Ts>{}...);
    }
}

/** The library space that the tag \a T derives from, reconstructed from what it exposes: a Linear or
 * Periodic of its sizes, a Composite of its blocks, a Repeated of its block, UnitQuaternion for a tag
 * derived from it, and the type itself for a space written from scratch. Two tags with one library
 * space beneath them retract the same way, so one may stand for the other in a model's input.
 */
template <typename T>
struct LibraryBaseOf
{
    /** The type itself. */
    using type = T;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <IsComposite T>
struct LibraryBaseOf<T>
{
    template <typename... Blocks>
    static Composite<Blocks...> *
    of(std::tuple<Blocks...> *);

    using type = std::remove_pointer_t<decltype(of(static_cast<CompositeBlocks<T> *>(nullptr)))>;
};

template <IsRepeated T>
struct LibraryBaseOf<T>
{
    using type = Repeated<RepeatedBlock<T>, T::MaxCount>;
};

template <IsUniformLeaf T>
    requires(!IsPeriodic<T>)
struct LibraryBaseOf<T>
{
    using type = Linear<T::ValueDimensionStep == 0 ? T::FixedValueDimension : Eigen::Dynamic, T::MaxValueDimension>;
};

template <IsPeriodic T>
struct LibraryBaseOf<T>
{
    template <PeriodicInterval Interval, int Dimension, int MaxDimension>
    static Periodic<Interval, Dimension, MaxDimension> *
    of(Periodic<Interval, Dimension, MaxDimension> const *);

    using type = std::remove_pointer_t<decltype(of(static_cast<T const *>(nullptr)))>;
};

/// \endcond

/** The library space beneath the tag \a T (LibraryBaseOf). */
template <typename T>
using LibraryBase = typename LibraryBaseOf<T>::type;

/** Whether the tags \a Name and \a Block have the same leaf beneath them, so that \a Name can stand
 * for \a Block in a model's input: one derives from the other, or both derive from one library space.
 * Two tags over a space written from scratch, which exposes nothing to rebuild it from, match only
 * where one derives from the other.
 */
template <typename Name, typename Block>
inline constexpr bool sameLeaf = std::derived_from<Name, Block> || std::derived_from<Block, Name>
                              || std::is_same_v<LibraryBase<Name>, LibraryBase<Block>>;

}  // namespace internal

/** \ingroup Models
 * \brief What a model reads, as reads() declares it: the items, each a path into the state or the tag
 * of an input of the call, under a name (decisions.md [model-policies]).
 *
 * The model's input space is the Composite of the items' names, in the order given, and the
 * function reads its input vector with <tt>block<Name>(in, x)</tt>. An item is a path value,
 * <tt>path<Ownship, Position></tt> or a bare tag's value, read under its last tag, or
 * <tt>on<Name>(path<...>)</tt>, read under \a Name, so that two items whose paths end in one tag
 * can be told apart. A path of one tag that the state lacks names an input of the call, given as an
 * uncertain input over that tag's space or as sure<Tag>(). A path into the state may select
 * repetitions of the repeated block it runs through, <tt>path<Landmarks>[k]</tt> (SelectedPath):
 * one repetition is read under the repetition's tag, as the block that is repeated, and a sequence
 * or a list of them under the repeated block's own tag, as that block with their count (decisions.md
 * [selected-reads]). The policy holds each selection by value.
 *
 * \tparam Items  the items, as internal::ReadItem
 *
 * \sa reads(), writes()
 */
template <typename... Items>
struct Reads : internal::ReadSelections<std::index_sequence_for<Items...>, Items...>
{
    static constexpr std::size_t Count = sizeof...(Items);  ///< the number of items

    /// whether an item reads selected repetitions of a repeated block
    static constexpr bool Selected = (Items::Selected || ... || false);

    /** The input space: the Composite of the names, in order. */
    using InputSpace = Composite<typename Items::NameType...>;

    /** The item at position \a I. */
    template <std::size_t I>
    using Item = std::tuple_element_t<I, std::tuple<Items...>>;

    /** The policy of items that read whole blocks; one with a selection is made from its selections. */
    constexpr Reads()
        requires(!Selected)
    = default;

    /** The policy whose items hold \a selections, one for each in order, internal::NoSelection for
     * an item that reads its whole block.
     */
    explicit Reads(std::in_place_t, typename Items::SelectionType const &... selections)
        : internal::ReadSelections<std::index_sequence_for<Items...>, Items...>{std::in_place, selections...}
    {}

    /** \returns the selection of the item at position \a I, or internal::NoSelection */
    template <std::size_t I>
    [[nodiscard]] auto const &
    selection() const
    {
        return static_cast<internal::ItemSelection<I, typename Item<I>::SelectionType> const &>(*this).value;
    }
};

/** \ingroup Models
 * \returns the reads() policy of \a items: each a path value, <tt>path<Tags...></tt> or a tag's
 * value, read under its last tag, a selected path, <tt>path<Landmarks>[k]</tt>, read under the tag
 * of the repetitions' space, or <tt>on<Name>(path)</tt> of either, read under \a Name. Two items
 * that would share a name are refused, with a message that points at on<Name>(), and so is a
 * selection whose space has no tag.
 *
 * \code
 * reads(path<Position>, path<Velocity>, path<SpecificForce>)
 * reads(on<SensorPosition>(path<Ownship, Position>), path<Target, Position>, path<ScaleNoise>)
 * reads(path<Position>, path<Heading>, path<Landmarks>[k])
 * reads(path<Pose>, on<Sighted>(path<Landmarks>[seen]))
 * \endcode
 */
template <typename... Items>
[[nodiscard]] auto
reads(Items const &... items)
{
    static_assert(
        (internal::ReadItemOf<std::remove_cvref_t<Items>>::valid && ...),
        "reads() takes paths, path<Tags...> or a tag's value, and on<Name>(path) for a path read under another "
        "name"
    );
    using Result = Reads<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type...>;
    static_assert(
        (!std::is_same_v<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type::NameType, internal::NoName>
         && ...),
        "the repetitions this item of reads() selects have a space without a tag, so name it: "
        "on<Name>(path<...>[...])"
    );
    static_assert(
        internal::distinctTypes<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type::NameType...>(),
        "two items of reads() share a name, their paths' last tag; read one under another name with on<Name>(path)"
    );
    return Result{std::in_place, internal::readSelection(items)...};
}

/** \ingroup Models
 * \brief What a model writes, as writes() declares it: the blocks its function's result fills
 * (decisions.md [model-policies]). For a process model they are paths into the state, for a
 * measurement model blocks of the measurement space, and the output space is their Composite, in
 * the order given.
 *
 * \tparam Paths  the paths, each a TagPath
 *
 * \sa writes(), reads(), WritesItems
 */
template <typename... Paths>
struct Writes
{
    static constexpr std::size_t Count = sizeof...(Paths);  ///< the number of blocks written

    /** The output space: the Composite of the blocks, in order. */
    using OutputSpace = Composite<internal::PathTag<Paths>...>;

    /** The paths, as a TagPaths. */
    using PathList = TagPaths<Paths...>;

    /** The path at position \a I. */
    template <std::size_t I>
    using Path = std::tuple_element_t<I, std::tuple<Paths...>>;
};

namespace internal
{

/** What a composition or a placed model writes: the paths \a Paths_, a TagPaths, into the state,
 * under the names of the output space \a Out_, a Composite with a block per path in order. A block's
 * name is its path's last tag, as in writes(), except where a placement gives it one of its own,
 * so that two placements of one model write blocks the output tells apart.
 *
 * \tparam Paths_  the paths, a TagPaths
 * \tparam Out_    the output space
 */
template <typename Paths_, typename Out_>
struct WritesAs
{
    using PathList = Paths_;   ///< the paths, as a TagPaths
    using OutputSpace = Out_;  ///< the output space: the Composite of the blocks' names, in order
};

}  // namespace internal

/** \ingroup Models
 * \brief What a process model writes where an item of its writes() selects repetitions of a repeated
 * block, or is written under a name of its own (decisions.md [selected-writes]): the items, each a
 * path into the state under a name, with the selections held by value as Reads holds those of what
 * a model reads.
 *
 * The output space is the Composite of the items' names, in the order given, and the function
 * fills its result with <tt>block<Name>(out, y)</tt>. One repetition, <tt>path<Fleet>[k]</tt>, is
 * written under the repetition's tag, as the block that is repeated, and a sequence or a list of
 * them under the repeated block's own tag, as that block with their count. writes() of whole blocks
 * under their own tags is a Writes, which holds nothing.
 *
 * \tparam Items  the items, as internal::ReadItem
 *
 * \sa writes(), Writes
 */
template <typename... Items>
struct WritesItems : internal::ReadSelections<std::index_sequence_for<Items...>, Items...>
{
    static constexpr std::size_t Count = sizeof...(Items);  ///< the number of blocks written

    /// whether an item writes selected repetitions of a repeated block
    static constexpr bool Selected = (Items::Selected || ... || false);

    /** The output space: the Composite of the names, in order. */
    using OutputSpace = Composite<typename Items::NameType...>;

    /** The paths, as a TagPaths; a path that an item selects repetitions of names more than is written. */
    using PathList = TagPaths<typename Items::PathType...>;

    /** The item at position \a I. */
    template <std::size_t I>
    using Item = std::tuple_element_t<I, std::tuple<Items...>>;

    /** The policy whose items hold \a selections, one for each in order, internal::NoSelection for
     * an item that writes its whole block.
     */
    explicit WritesItems(std::in_place_t, typename Items::SelectionType const &... selections)
        : internal::ReadSelections<std::index_sequence_for<Items...>, Items...>{std::in_place, selections...}
    {}

    /** \returns the selection of the item at position \a I, or internal::NoSelection */
    template <std::size_t I>
    [[nodiscard]] auto const &
    selection() const
    {
        return static_cast<internal::ItemSelection<I, typename Item<I>::SelectionType> const &>(*this).value;
    }
};

namespace internal
{

/** \returns the position of the first of the types \a Ts that is \a T, or their count if none is */
template <typename T, typename... Ts>
[[nodiscard]] consteval std::size_t
typePosition()
{
    constexpr std::array<bool, sizeof...(Ts) + 1> same{std::is_same_v<Ts, T>..., true};
    std::size_t i = 0;
    while (!same[i])
    {
        ++i;
    }
    return i;
}

/** \returns whether no two of the items \a Items that name whole blocks name one path. Two items
 * that select repetitions of one block may select different ones, which is known only where the
 * model meets a state.
 */
template <typename... Items>
[[nodiscard]] consteval bool
wholeBlocksDistinct()
{
    constexpr std::size_t count = sizeof...(Items);
    constexpr std::array<bool, count + 1> selected{Items::Selected..., false};
    // For each item, the position of the first item with its path.
    constexpr std::array<std::size_t, count + 1> first{
        typePosition<typename Items::PathType, typename Items::PathType...>()...,
        0
    };
    for (std::size_t i = 0; i < count; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            if (!selected[i] && !selected[j] && first[i] == first[j])
            {
                return false;
            }
        }
    }
    return true;
}

}  // namespace internal

/** \ingroup Models
 * \returns the writes() policy of \a items, the blocks of the state that a process model's function
 * fills, or the blocks of its measurement space for a measurement model: each a path value,
 * <tt>path<Tags...></tt> or a tag's value, written under its last tag. For a process model an item
 * may also be a selected path, <tt>path<Fleet>[k]</tt>, written under the tag of the repetitions'
 * space, or <tt>on<Name>(path)</tt> of either, written under \a Name (decisions.md
 * [selected-writes]). A block named twice is refused, and so are two items that would share a name
 * and a selection whose space has no tag.
 *
 * \code
 * writes(path<Position>, path<Velocity>)
 * writes(path<Fleet>[k])
 * writes(on<Lead>(path<Fleet>[0]), on<Wing>(path<Fleet>[1]))
 * \endcode
 */
template <typename... Items>
[[nodiscard]] auto
writes(Items const &... items)
{
    static_assert(sizeof...(Items) > 0, "writes() names at least one block");
    static_assert(
        (internal::ReadItemOf<std::remove_cvref_t<Items>>::valid && ...),
        "writes() takes paths, path<Tags...> or a tag's value, and on<Name>(path) for a path written under another "
        "name"
    );
    if constexpr ((PathValue<Items> && ...))
    {
        static_assert(internal::distinctTypes<internal::PathOf<Items>...>(), "writes() names each block once");
        return Writes<internal::PathOf<Items>...>{};
    }
    else
    {
        using Result = WritesItems<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type...>;
        static_assert(
            internal::wholeBlocksDistinct<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type...>(),
            "writes() names each block once"
        );
        static_assert(
            (!std::
                 is_same_v<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type::NameType, internal::NoName>
             && ...),
            "the repetitions this item of writes() selects have a space without a tag, so name it: "
            "on<Name>(path<...>[...])"
        );
        static_assert(
            internal::distinctTypes<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type::NameType...>(),
            "two items of writes() share a name, their paths' last tag; write one under another name with "
            "on<Name>(path)"
        );
        return Result{std::in_place, internal::readSelection(items)...};
    }
}

/** \ingroup Models
 * \brief The input names a model's function is affine in, given the rest, as affineIn() declares
 * them (decisions.md [model-policies], [affine-coordinates]): a sigma-point method factors those
 * coordinates out and spreads its points over the others alone.
 *
 * \tparam Names  the names, each a tag of the model's reads()
 */
template <typename... Names>
struct AffineIn
{
    static constexpr std::size_t Count = sizeof...(Names);  ///< the number of names
};

/** \ingroup Models
 * \returns the affineIn() policy of \a names, each the one tag an item of reads() is read under, as
 * a path value of one tag
 */
template <PathValue... Names>
[[nodiscard]] auto
affineIn(Names const &...)
{
    static_assert(
        (std::is_same_v<internal::PathOf<Names>, TagPath<internal::PathTag<internal::PathOf<Names>>>> && ...),
        "affineIn() names an input of the model by the one tag it is read under in reads(), never by a path into the "
        "state"
    );
    static_assert(
        internal::distinctTypes<internal::PathTag<internal::PathOf<Names>>...>(),
        "affineIn() names each input once"
    );
    return AffineIn<internal::PathTag<internal::PathOf<Names>>...>{};
}

/** \ingroup Models
 * \brief The constant term \f$ c \f$ of a whole output, given to offset() as a vector: a difference
 * of the output, added to its function's value, or a point of it where the model has no function.
 */
template <typename Value>
struct WholeOffset
{
    Value value;  ///< \f$ c \f$
};

/** \ingroup Models
 * \brief The constant term \f$ c \f$ of a model's output that affineIn() takes, offset(), by block
 * of the output, each <tt>on<Block>(c)</tt>, or for the whole output, a WholeOffset; an output
 * block without a part has none (decisions.md [affine-coordinates]).
 *
 * \tparam Parts  the parts, each an OnPath of a vector on one block of the output, or one WholeOffset
 */
template <typename... Parts>
struct AffineOffset
{
    std::tuple<Parts...> parts;  ///< the parts
};

namespace internal
{

template <typename F, typename InnerIn, typename Offset, typename... Constants>
class ConstantAffineFunction;

/** The value and the path of \a T, a value placed on one block by on() or on each repetition by
 * onEach().
 */
template <typename T>
struct OnPathParts
{};

/** An OnPath. */
template <typename X, typename Path, typename Selection>
struct OnPathParts<OnPath<X, Path, Selection>>
{
    using Value = X;        ///< the value placed
    using PathType = Path;  ///< the path it is placed on, a tag or a TagPath
    /// whether it is <tt>on<Name>(H)</tt> of one block and no selection, which a model with a function takes
    static constexpr bool Plain = std::is_same_v<Selection, NoSelection>;
};

/** An OnEach. */
template <typename X, typename Path, typename Selection>
struct OnPathParts<OnEach<X, Path, Selection>>
{
    using Value = X;                      ///< the value placed
    using PathType = Path;                ///< the path of the repeated block, a tag or a TagPath
    static constexpr bool Plain = false;  ///< only a model without a function takes it
};

/** Whether \a T is a constant item of affineIn(), <tt>on<Name>(H)</tt> or <tt>onEach<Block>(H)</tt>,
 * or either of a selection: a matrix placed on an input.
 */
template <typename T>
inline constexpr bool isConstantItem = false;

/** An OnPath of a matrix is one. */
template <typename X, typename Path, typename Selection>
inline constexpr bool isConstantItem<OnPath<X, Path, Selection>> = MatrixExpression<X>;

/** An OnEach of a matrix is one. */
template <typename X, typename Path, typename Selection>
inline constexpr bool isConstantItem<OnEach<X, Path, Selection>> = MatrixExpression<X>;

/** Whether \a T is an offset() part of a block, <tt>on<Block>(c)</tt>. */
template <typename T>
inline constexpr bool isOffsetPart = [] {
    if constexpr (isConstantItem<T>)
    {
        return OnPathParts<T>::Plain;
    }
    else
    {
        return false;
    }
}();

/** Whether \a T is an AffineOffset. */
template <typename T>
inline constexpr bool isAffineOffset = false;

/** An AffineOffset is one. */
template <typename... Parts>
inline constexpr bool isAffineOffset<AffineOffset<Parts...>> = true;

/** \returns \a item as the tuple of the names it gives affineIn() at each point: its tag for a
 * path value, and nothing for a constant item or an offset
 */
template <typename Item>
[[nodiscard]] auto
pointNameOf(Item const *)
{
    if constexpr (PathValue<Item>)
    {
        return std::tuple<PathTag<PathOf<Item>>>{};
    }
    else
    {
        return std::tuple<>{};
    }
}

/** \returns \a item as the tuple of the constant items it gives affineIn() */
template <typename Item>
[[nodiscard]] auto
constantOf(Item const & item)
{
    if constexpr (isConstantItem<Item>)
    {
        return std::tuple<Item>{item};
    }
    else
    {
        return std::tuple<>{};
    }
}

/** \returns \a item as the tuple of the offsets it gives affineIn() */
template <typename Item>
[[nodiscard]] auto
offsetOf(Item const & item)
{
    if constexpr (isAffineOffset<Item>)
    {
        return std::tuple<Item>{item};
    }
    else
    {
        return std::tuple<>{};
    }
}

/** Whether the item \a Item of affineIn() names its input by one tag: a path value of one tag, or a
 * constant item or an offset part placed on one tag.
 */
template <typename Item>
[[nodiscard]] consteval bool
namedByOneTag()
{
    if constexpr (PathValue<Item>)
    {
        return std::is_same_v<PathOf<Item>, TagPath<PathTag<PathOf<Item>>>>;
    }
    else if constexpr (isConstantItem<Item>)
    {
        using Path = typename OnPathParts<Item>::PathType;
        return std::is_same_v<AsTagPath<Path>, TagPath<PathTag<Path>>>;
    }
    else
    {
        return true;
    }
}

/** The AffineIn of the names in the tuple \a Names. */
template <typename Names>
struct AffineInOf;

/** Of a tuple of names. */
template <typename... Names>
struct AffineInOf<std::tuple<Names...>>
{
    using type = AffineIn<Names...>;  ///< the policy
};

}  // namespace internal

/** \ingroup Models
 * \brief affineIn() with constant items (decisions.md [affine-coordinates]): the names \a Points,
 * whose coefficient the function gives at each point, and the constant items \a Constants, each
 * <tt>on<Name>(H)</tt>, a matrix over every row of the output for the input \a Name, which the
 * function does not read, with the offset \a Offset. measurementModel() and processModel() read
 * each constant item's input as an affine one and add \f$ \sum_k H_k x_k + c \f$ to the function's
 * value (internal::ConstantAffineFunction).
 */
template <typename Points, typename Offset, typename... Constants>
struct AffineConstants;

/** Of the names \a Points. */
template <typename... Points, typename Offset, typename... Constants>
struct AffineConstants<AffineIn<Points...>, Offset, Constants...>
{
    /// the constant items' names, as a tuple
    using ConstantNames = std::tuple<internal::PathTag<typename internal::OnPathParts<Constants>::PathType>...>;

    /// the affineIn() of the model the factories make: every name, the constant ones last
    using AllNames = AffineIn<Points..., internal::PathTag<typename internal::OnPathParts<Constants>::PathType>...>;

    /// whether every constant item is <tt>on<Name>(H)</tt> of one input, which a model with a function takes
    static constexpr bool PlainConstants = (internal::OnPathParts<Constants>::Plain && ... && true);

    /// whether no item is a path value, so that a model without a function takes it
    static constexpr bool NoPoints = sizeof...(Points) == 0;

    /// whether the offset has no part
    static constexpr bool NoOffset = std::is_same_v<Offset, AffineOffset<>>;

    /// the function the factories make of \a F, which reads \a InnerIn
    template <typename F, typename InnerIn>
    using FunctionOf = internal::ConstantAffineFunction<F, InnerIn, Offset, Constants...>;

    std::tuple<Constants...> constants;  ///< the constant items
    Offset offset;                       ///< the offset
};

/** \ingroup Models
 * \brief affineIn() of one matrix, \f$ F \f$ or \f$ H \f$ over the whole state, which a model
 * without a function and without tags takes as its whole map (decisions.md [whole-state-models]).
 *
 * \tparam Matrix  the matrix, a plain Eigen matrix
 */
template <typename Matrix>
struct AffineMatrix
{
    Matrix matrix;  ///< the matrix
};

/** \ingroup Models
 * \returns the affineIn() of the matrix \a matrix over the whole state, for a model without a
 * function: <tt>processModel(affineIn(F), additiveNoise(Q))</tt> or
 * <tt>measurementModel(affineIn(H), additiveNoise(R))</tt>
 */
template <MatrixExpression Matrix>
[[nodiscard]] AffineMatrix<typename Matrix::PlainObject>
affineIn(Matrix const & matrix)
{
    return {matrix};
}

/** \ingroup Models
 * \returns the offset() of affineIn(): the constant term \f$ c \f$ by block of the output, each part
 * <tt>on<Block>(c)</tt> a vector over the block's differences
 */
template <typename... Parts>
    requires(sizeof...(Parts) > 0 && (internal::isOffsetPart<Parts> && ...))
[[nodiscard]] AffineOffset<Parts...>
offset(Parts const &... parts)
{
    static_assert(
        (internal::namedByOneTag<Parts>() && ...),
        "offset() places each part on a block of the output by its one tag, on<Block>(c)"
    );
    static_assert(
        internal::distinctTypes<internal::PathTag<typename internal::OnPathParts<Parts>::PathType>...>(),
        "offset() gives each block of the output one part"
    );
    return {{parts...}};
}

/** \ingroup Models
 * \returns the offset() of affineIn() for the whole output: the vector \a c over its differences
 */
template <MatrixExpression C>
[[nodiscard]] AffineOffset<WholeOffset<typename C::PlainObject>>
offset(C const & c)
{
    return {{WholeOffset<typename C::PlainObject>{c}}};
}

/** \ingroup Models
 * \returns the affineIn() policy of \a items with constant items among them (decisions.md
 * [affine-coordinates]): each path value <tt>path<Name></tt> an input whose coefficient the function
 * gives at each point, each <tt>on<Name>(H)</tt> an input the function does not read, whose columns
 * are the matrix \f$ H \f$ over every row of the output, and at most one offset(), the constant term.
 *
 * \code
 * affineIn(path<Velocity>, on<Bias>(Hb), offset(on<Range>(cr), on<Bearing>(cb)))
 * \endcode
 */
template <typename... Items>
    requires(
        !(PathValue<Items> && ...)
        && ((PathValue<Items> || internal::isConstantItem<Items> || internal::isAffineOffset<Items>) && ...)
    )
[[nodiscard]] auto
affineIn(Items const &... items)
{
    using PointNames = decltype(std::tuple_cat(internal::pointNameOf(static_cast<Items const *>(nullptr))...));
    auto constants = std::tuple_cat(internal::constantOf(items)...);
    auto offsets = std::tuple_cat(internal::offsetOf(items)...);
    static_assert(std::tuple_size_v<decltype(offsets)> <= 1, "affineIn() takes one offset()");
    static_assert(
        (internal::namedByOneTag<Items>() && ...),
        "affineIn() names an input of the model by the one tag it is read under in reads(), never by a path into the "
        "state"
    );
    static_assert(
        []<typename... Points, typename... Constants>(std::tuple<Points...> *, std::tuple<Constants...> *) {
            return internal::
                distinctTypes<Points..., internal::PathTag<typename internal::OnPathParts<Constants>::PathType>...>();
        }(static_cast<PointNames *>(nullptr), static_cast<decltype(constants) *>(nullptr)),
        "affineIn() gives each input one item, path<Name> or on<Name>(H)"
    );
    using Points = typename internal::AffineInOf<PointNames>::type;
    auto offset = [&] {
        if constexpr (std::tuple_size_v<decltype(offsets)> == 0)
        {
            return AffineOffset<>{};
        }
        else
        {
            return std::get<0>(offsets);
        }
    }();
    return std::apply(
        [&](auto &... c) {
            return AffineConstants<Points, decltype(offset), std::remove_cvref_t<decltype(c)>...>{
                {std::move(c)...},
                std::move(offset)
            };
        },
        constants
    );
}

namespace internal
{

/** Whether \a T is a GaussianMixtureParams, so additive noise has several components. */
template <typename T>
inline constexpr bool isGaussianMixture = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Scalar, typename Space, typename Allocator>
inline constexpr bool isGaussianMixture<GaussianMixtureParams<Space, Scalar, Allocator>>{true};
/// \endcond

/** Whether \a T, a part of additiveNoise() as held, is a covariance alone: zero-mean %Gaussian
 * noise over the differences of what it covers, whose space the model's output gives.
 */
template <typename T>
inline constexpr bool isNoiseCovariance = MatrixExpression<T>;

/** Whether \a T places a covariance alone by tag path, <tt>on<Path>(R)</tt> or <tt>onEach<Path>(R)</tt>. */
template <typename T>
inline constexpr bool placesCovariance = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename Path, typename Selection>
inline constexpr bool placesCovariance<OnPath<X, Path, Selection>> = isNoiseCovariance<X>;

template <typename X, typename Path, typename Selection>
inline constexpr bool placesCovariance<OnEach<X, Path, Selection>> = isNoiseCovariance<X>;
/// \endcond

/** Whether \a T, a part of additiveNoise(), is placed by tag path with on() or onEach(): a
 * distribution or a covariance.
 */
template <typename T>
inline constexpr bool isPlacedNoise =
    isPlacedFactor<std::remove_cvref_t<T>> || placesCovariance<std::remove_cvref_t<T>>;

/** Whether the part \a T of additiveNoise() places a mixture, which a product of factors cannot
 * hold.
 */
template <typename T>
inline constexpr bool placesMixture = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename Path, typename Selection>
inline constexpr bool placesMixture<OnPath<X, Path, Selection>> = isGaussianMixture<std::remove_cvref_t<X>>;

template <typename X, typename Path, typename Selection>
inline constexpr bool placesMixture<OnEach<X, Path, Selection>> = isGaussianMixture<std::remove_cvref_t<X>>;
/// \endcond

}  // namespace internal

/** \ingroup Models
 * \brief Noise added to a model's output, \f$ z = h(x) \boxplus v \f$, as additiveNoise() declares
 * it (decisions.md [additive-noise], [model-policies]): the function is transformed once, and the
 * noise only widens and shifts the result.
 *
 * One unplaced part covers the whole output space: a covariance matrix \f$ R \f$ over the output's
 * differences, zero-mean %Gaussian noise, as additive noise nearly always is; a GaussianParams or a
 * GaussianMixtureParams over them, such as a biased sensor's or a narrow inlier and a wide outlier
 * component; or a const reference to one that outlives the model. Parts placed by tag path,
 * <tt>on<Range>(r)</tt> and <tt>onEach<Offsets>(r)</tt>, each a covariance or a GaussianParams, are
 * independent Gaussians on blocks of the output, placed in its space instance when the model meets
 * a state, as the factors of a Product are (decisions.md [product-factors]); a placed mixture is
 * refused. A covariance has no space of its own, so a model whose output space would come from its
 * noise measures in a Linear sized by its function's result instead.
 *
 * \tparam Parts  the parts, as given
 *
 * \sa additiveNoise(), ContinuousNoise
 */
template <typename... Parts>
class AdditiveNoise
{
    static_assert(sizeof...(Parts) > 0, "additiveNoise() takes a distribution, or parts placed by tag path");
    static_assert(
        (internal::isPlacedNoise<Parts> && ...) || (sizeof...(Parts) == 1 && !(internal::isPlacedNoise<Parts> || ...)),
        "additiveNoise() with no placement covers the whole output space, so it takes one distribution alone; place "
        "each part with on<Path>() or onEach<Path>() to give several"
    );
    static_assert(
        (!internal::placesMixture<std::remove_cvref_t<Parts>> && ...),
        "a mixture is additive noise over the whole output space, unplaced; a placed part is one Gaussian"
    );

public:

    /// whether the parts are placed by tag path, and so become one Gaussian in the output space
    static constexpr bool Placed = (internal::isPlacedNoise<Parts> || ...);

    /** The one unplaced distribution's type, unqualified, or the covariance's. */
    using Distribution = std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Parts...>>>;

    /// whether the one unplaced part is a covariance alone, zero-mean noise with no space of its own
    static constexpr bool ZeroMean = !Placed && internal::isNoiseCovariance<Distribution>;

    /// whether the one unplaced part is a distribution, whose space a model's output may take
    static constexpr bool HasOwnSpace = !Placed && !ZeroMean;

    /** The noise of \a parts. */
    explicit AdditiveNoise(Parts... parts)
        : m_parts{std::forward<Parts>(parts)...}
    {}

    /** \returns the parts, as given */
    [[nodiscard]] std::tuple<Parts...> const &
    parts() const
    {
        return m_parts;
    }

    /** \returns the one unplaced distribution, or covariance, over the whole output space */
    [[nodiscard]] Distribution const &
    distribution() const &
        requires(!Placed)
    {
        return std::get<0>(m_parts);
    }

    /** \returns the one unplaced distribution of a temporary noise: moved out where it is held by
     * copy, the reference where it is held by one, so that a reference bound to
     * <tt>model.noise().distribution()</tt> lives as long as the result and never dangles
     */
    [[nodiscard]] std::tuple_element_t<0, std::tuple<Parts...>>
    distribution() &&
        requires(!Placed)
    {
        return std::get<0>(std::move(m_parts));
    }

private:

    std::tuple<Parts...> m_parts;  ///< the parts
};

/** \ingroup Models
 * \brief Additive noise that depends on a mode axis, one %Gaussian per value of it, as
 * <tt>additiveNoise(byMode<Axis>(when<Value>(gaussian), ...))</tt> declares it (decisions.md
 * [by-mode], [imm]): the Markov outlier model, in which a sensor's outlier state is a mode of the
 * target that persists as the mode chain says.
 *
 * A filter whose state has the mode axis \a Axis, such as an ImmFilter whose mode space has it,
 * updates each mode with the %Gaussian its value in \a Axis names, components(), and reweights the
 * mode by that %Gaussian's likelihood; the chain gives the probabilities. A filter whose state has no
 * mode space, such as a KalmanFilter or a GaussianMixtureFilter, reads it as the mixture of the
 * %Gaussians with the weights given to the alternatives, an outlier rate independent from one
 * measurement to the next: the memoryless outlier model. Given no weights, such a filter refuses it
 * at compile time. A filter whose state has a mode space without \a Axis refuses it at compile time.
 * So one measurement serves a filter with mixture noise and the IMM. The weights should match the
 * chain's stationary probabilities on \a Axis, for the two readings to describe one sensor.
 *
 * \tparam Axis      the path to the mode axis, a tag or a TagPath
 * \tparam Part      the %Gaussians as a mixture, in the order of the axis's values
 * \tparam Weighted  whether the alternatives were given weights
 *
 * \sa byMode(), additiveNoise(), ImmFilter
 */
template <typename Axis, typename Part, bool Weighted>
class ModeNoise : public AdditiveNoise<Part>
{
public:

    using ModeAxisType = Axis;  ///< the path to the mode axis whose values the components are

    /** \a components, one %Gaussian per value of the mode axis, in order, as a mixture. */
    explicit ModeNoise(Part components)
        : AdditiveNoise<Part>{std::move(components)}
    {}

    /** \returns the %Gaussians, one per value of the mode axis, in order, as a mixture whose weights are
     * the alternatives' (equal, if they were given none)
     */
    [[nodiscard]] Part const &
    components() const
    {
        return AdditiveNoise<Part>::distribution();
    }

    /// whether the alternatives were given weights, which a filter without the mode axis reads them by
    static constexpr bool HasWeights = Weighted;
};

namespace internal
{

/** Whether \a T is a GaussianParams, which noise by mode axis takes per value. */
template <typename T>
inline constexpr bool isModeNoiseGaussian = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Space_, typename Scalar_>
inline constexpr bool isModeNoiseGaussian<GaussianParams<Space_, Scalar_>> = true;

/// \endcond

/** \returns the alternatives of \a noise, one %Gaussian per value of its axis, as a mixture in the axis's
 * order, with their weights, or equal ones
 */
template <typename P, typename... Alternatives>
[[nodiscard]] auto
modeNoiseMixture(ByMode<P, Alternatives...> const & noise)
{
    using NoiseGaussian = std::remove_cvref_t<typename ByMode<P, Alternatives...>::template PartType<0>>;
    static_assert(
        (std::is_same_v<std::remove_cvref_t<typename Alternatives::PartType>, NoiseGaussian> && ...),
        "additiveNoise(byMode<Axis>(...)) takes one GaussianParams type for every value of the axis"
    );
    static_assert(
        isModeNoiseGaussian<NoiseGaussian>,
        "additiveNoise(byMode<Axis>(...)) takes a GaussianParams per value of the axis, the noise in its modes"
    );
    using Scalar = typename NoiseGaussian::Scalar;
    auto mixture = GaussianMixtureParams<typename NoiseGaussian::SpaceType, Scalar>{
        std::get<0>(noise.alternatives()).part().space()
    };
    Eigen::Index const values = noise.values();
    mixture.reserve(values);
    for (Eigen::Index v = 0; v < values; ++v)
    {
        Scalar logWeight = -std::log(Scalar(values));
        if constexpr (ByMode<P, Alternatives...>::Weighted)
        {
            logWeight = std::log(Scalar(noise.weightFor(v)));
        }
        noise.visit(v, [&](NoiseGaussian const & component) {
            mixture.add(logWeight, component);
        });
    }
    return mixture;
}

}  // namespace internal

/** \ingroup Models
 * \returns the additive noise by mode axis of \a noise, one %Gaussian per value of the axis (see
 * ModeNoise): <tt>additiveNoise(byMode<Sensor>(when<Inlier>(inlier).weight(0.9),
 * when<Outlier>(outlier).weight(0.1)))</tt> [by-mode]
 */
template <typename P, typename... Alternatives>
[[nodiscard]] auto
additiveNoise(ByMode<P, Alternatives...> const & noise)
{
    using NoiseMixture = decltype(internal::modeNoiseMixture(noise));
    return ModeNoise<P, NoiseMixture, ByMode<P, Alternatives...>::Weighted>{internal::modeNoiseMixture(noise)};
}

/** \ingroup Models
 * \returns the additiveNoise() policy of \a parts (see AdditiveNoise): one covariance, zero-mean
 * noise, or one distribution, either held by copy, or for <tt>std::cref(noise)</tt> by reference to
 * one that must outlive the model, or parts placed by tag path with on<Path>() and onEach<Path>()
 *
 * \code
 * additiveNoise(R)  // zero-mean, R over the output's differences
 * additiveNoise(on<Range>(r), on<Bearing>(b))
 * additiveNoise(onEach<Offsets>(r))
 * additiveNoise(GaussianParams<Linear<2>>{bias, R})
 * additiveNoise(std::cref(mixture))
 * \endcode
 */
template <typename... Parts>
    requires(!internal::isByModePart<Parts> && ...)
[[nodiscard]] AdditiveNoise<typename internal::HeldOf<std::unwrap_ref_decay_t<Parts>>::type...>
additiveNoise(Parts &&... parts)
{
    return AdditiveNoise<typename internal::HeldOf<std::unwrap_ref_decay_t<Parts>>::type...>{
        std::forward<Parts>(parts)...
    };
}

/** \ingroup Models
 * \brief A process model's white-noise spectral density, as continuousNoise() declares it, placed
 * by tag path on blocks of the output and discretized over the step by the model (decisions.md
 * [continuous-time-noise], [model-policies]): \f$ Q_c \, \Delta t \f$ on each block's differences.
 *
 * Each part is <tt>on<Velocity>(q)</tt> or <tt>onEach<Fleet>(q)</tt>, with \f$ Q_c \f$ a scalar,
 * the density of every axis, or a matrix over the block's differences; one unplaced \f$ Q_c \f$
 * covers the whole output space.
 *
 * \tparam Parts  the parts, as given
 *
 * \sa continuousNoise(), AdditiveNoise
 */
template <typename... Parts>
class ContinuousNoise
{
    static_assert(sizeof...(Parts) > 0, "continuousNoise() takes a spectral density, or parts placed by tag path");
    static_assert(
        (internal::isPlacedByPath<std::remove_cvref_t<Parts>> && ...)
            || (sizeof...(Parts) == 1 && !(internal::isPlacedByPath<std::remove_cvref_t<Parts>> || ...)),
        "continuousNoise() with no placement covers the whole output space, so it takes one spectral density alone; "
        "place each part with on<Path>() or onEach<Path>() to give several"
    );

public:

    /// whether the parts are placed by tag path
    static constexpr bool Placed = (internal::isPlacedByPath<std::remove_cvref_t<Parts>> || ...);

    /** The noise of \a parts. */
    explicit ContinuousNoise(Parts... parts)
        : m_parts{std::move(parts)...}
    {}

    /** \returns the parts, as given */
    [[nodiscard]] std::tuple<Parts...> const &
    parts() const
    {
        return m_parts;
    }

private:

    std::tuple<Parts...> m_parts;  ///< the parts
};

/** \ingroup Models
 * \returns the continuousNoise() policy of \a parts (see ContinuousNoise): spectral densities placed
 * by tag path, <tt>continuousNoise(on<Velocity>(q))</tt>, or one for the whole output
 */
template <typename... Parts>
[[nodiscard]] ContinuousNoise<std::remove_cvref_t<Parts>...>
continuousNoise(Parts &&... parts)
{
    return ContinuousNoise<std::remove_cvref_t<Parts>...>{std::forward<Parts>(parts)...};
}

namespace internal
{

/** A built-in process model's noise over a step, as its own closed form gives it: a function
 * <tt>g(out, dt)</tt> of the output space instance and the step, returning \f$ Q \f$ over the
 * output's differences, where continuousNoise() would give only \f$ Q_c \, \Delta t \f$. Internal:
 * the built-in models carry one, and processModel() builds none.
 *
 * \tparam G  the function
 */
template <typename G>
struct DiscreteNoise
{
    G function;  ///< \f$ Q \f$ over the step, as <tt>g(out, dt)</tt>
};

/** Whether \a T is a model, a measurement or a process model: what observe(), predict(), and the
 * calls that read a model with arguments take. A model type declares <tt>static constexpr bool
 * IsModel = true</tt>.
 */
template <typename T>
inline constexpr bool isModel = requires { requires std::remove_cvref_t<T>::IsModel; };

}  // namespace internal

/** \ingroup Models
 * \brief A measurement model's own \f$ \log p(z \mid x) \f$ at a state point, as logLikelihood()
 * declares it, for the filters that weight points (decisions.md [likelihood-measurements],
 * [model-policies]): a function <tt>g(in, x, z)</tt> of the model's input space instance, its
 * input vector, and the observation.
 *
 * \tparam G  the function
 *
 * \sa logLikelihood(G)
 */
template <typename G>
struct LogLikelihood
{
    G function;  ///< \f$ \log p(z \mid x) \f$ as <tt>g(in, x, z)</tt>
};

/** \ingroup Models
 * \returns the logLikelihood() policy of \a function, <tt>g(in, x, z)</tt> (see LogLikelihood):
 * what a particle or grid filter weights by when the model's noise is not additive, or is not
 * %Gaussian
 */
template <typename G>
    requires(!MatrixExpression<G> && !Space<std::remove_cvref_t<G>>)
[[nodiscard]] LogLikelihood<std::remove_cvref_t<G>>
logLikelihood(G && function)
{
    return {std::forward<G>(function)};
}

/** \ingroup Models
 * \brief A draw of the observation that a measurement model known by its likelihood alone makes of
 * a state, as sample() declares it, for simulate() (decisions.md [likelihood-measurements],
 * [model-policies]): a function <tt>g(in, x, rng)</tt> of the model's input space instance, its
 * input vector, and the caller's generator, returning the observation.
 *
 * \tparam G  the function
 *
 * \sa sample(G)
 */
template <typename G>
struct Sample
{
    G function;  ///< a draw of \f$ z \sim p(z \mid x) \f$ as <tt>g(in, x, rng)</tt>
};

/** \ingroup Models
 * \returns the sample() policy of \a function, <tt>g(in, x, rng)</tt> (see Sample): what simulate()
 * draws an observation by for a model known by its likelihood alone, whose logLikelihood() it must
 * agree with
 */
template <typename G>
    requires(!MatrixExpression<G> && !Space<std::remove_cvref_t<G>>)
[[nodiscard]] Sample<std::remove_cvref_t<G>>
sample(G && function)
{
    return {std::forward<G>(function)};
}

/** \ingroup Models
 * \brief A model's output space given by its instance, or computed from the state's space, as
 * outputSpace() declares it (decisions.md [measurement-space-functions], [model-policies]): for an
 * output with a resizable block, whose size the type does not fix.
 *
 * \tparam X  a Space instance, or a function from the state's space instance to the output's
 *
 * \sa outputSpace(), CountFrom
 */
template <typename X>
struct OutputSpace
{
    /// whether the output space is a function of the state's space
    static constexpr bool Computed = !Space<X>;

    X value;  ///< the instance, or the function

    /** \returns the output space in the state's space \a stateSpace: the instance, or the function's
     * result there
     */
    template <typename StateSpace>
    [[nodiscard]] auto
    in([[maybe_unused]] StateSpace const & stateSpace) const
    {
        if constexpr (Computed)
        {
            static_assert(
                std::invocable<X const &, StateSpace const &>,
                "a model's outputSpace() function takes the state's space instance"
            );
            using Result = std::remove_cvref_t<std::invoke_result_t<X const &, StateSpace const &>>;
            static_assert(Space<Result>, "a model's outputSpace() function returns the output space instance");
            return value(stateSpace);
        }
        else
        {
            return value;
        }
    }
};

/** \ingroup Models
 * \returns the outputSpace() policy of \a x (see OutputSpace): the output space instance, or a
 * function of the state's space instance that computes it, such as CountFrom
 */
template <typename X>
    requires(!MatrixExpression<X>)
[[nodiscard]] OutputSpace<std::remove_cvref_t<X>>
outputSpace(X && x)
{
    return {std::forward<X>(x)};
}

namespace internal
{

/** The kinds of policy a model takes. */
enum class PolicyKind
{
    reads,
    writes,
    affineIn,
    additiveNoise,
    continuousNoise,
    logLikelihood,
    sample,
    outputSpace,
    discreteNoise  ///< internal: a built-in process model's closed-form noise over the step
};

/** The kind of the policy type \a T, or nothing for a type that is no policy. */
template <typename T>
inline constexpr std::optional<PolicyKind> policyKind = std::nullopt;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Items>
inline constexpr std::optional<PolicyKind> policyKind<Reads<Items...>>{PolicyKind::reads};

template <typename... Paths>
inline constexpr std::optional<PolicyKind> policyKind<Writes<Paths...>>{PolicyKind::writes};

template <typename Paths, typename Out>
inline constexpr std::optional<PolicyKind> policyKind<WritesAs<Paths, Out>>{PolicyKind::writes};

template <typename... Items>
inline constexpr std::optional<PolicyKind> policyKind<WritesItems<Items...>>{PolicyKind::writes};
/// \endcond

/** Whether \a T is a WritesItems, the writes() of a process model with a selection or a name of its own. */
template <typename T>
inline constexpr bool isWritesItems = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Items>
inline constexpr bool isWritesItems<WritesItems<Items...>> = true;

template <typename... Names>
inline constexpr std::optional<PolicyKind> policyKind<AffineIn<Names...>>{PolicyKind::affineIn};

template <typename Points, typename Offset, typename... Constants>
inline constexpr std::optional<PolicyKind> policyKind<AffineConstants<Points, Offset, Constants...>>{
    PolicyKind::affineIn
};

template <typename Matrix>
inline constexpr std::optional<PolicyKind> policyKind<AffineMatrix<Matrix>>{PolicyKind::affineIn};

template <typename... Parts>
inline constexpr std::optional<PolicyKind> policyKind<AdditiveNoise<Parts...>>{PolicyKind::additiveNoise};

template <typename Axis, typename Part, bool Weighted>
inline constexpr std::optional<PolicyKind> policyKind<ModeNoise<Axis, Part, Weighted>>{PolicyKind::additiveNoise};

template <typename... Parts>
inline constexpr std::optional<PolicyKind> policyKind<ContinuousNoise<Parts...>>{PolicyKind::continuousNoise};

template <typename G>
inline constexpr std::optional<PolicyKind> policyKind<LogLikelihood<G>>{PolicyKind::logLikelihood};

template <typename G>
inline constexpr std::optional<PolicyKind> policyKind<Sample<G>>{PolicyKind::sample};

template <typename X>
inline constexpr std::optional<PolicyKind> policyKind<OutputSpace<X>>{PolicyKind::outputSpace};

template <typename G>
inline constexpr std::optional<PolicyKind> policyKind<DiscreteNoise<G>>{PolicyKind::discreteNoise};
/// \endcond

/** Whether \a T is a policy a model takes. */
template <typename T>
concept ModelPolicy = policyKind<std::remove_cvref_t<T>>.has_value();

/** Whether \a T is an AdditiveNoise. */
template <typename T>
inline constexpr bool isAdditiveNoise = policyKind<std::remove_cvref_t<T>> == PolicyKind::additiveNoise;

/** The path to the mode axis whose values the components of the additive noise \a Noise are, as
 * \c type: a ModeNoise's axis, or \c void for other noise.
 */
template <typename Noise>
struct NoiseModeAxisOf
{
    /** no mode axis */
    using type = void;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Axis, typename Part, bool Weighted>
struct NoiseModeAxisOf<ModeNoise<Axis, Part, Weighted>>
{
    using type = Axis;  ///< the noise's mode axis
};

/// \endcond

/** Whether the additive noise \a Noise is by mode axis with alternatives given no weights, which only a
 * filter that tracks the axis can read.
 */
template <typename Noise>
inline constexpr bool isUnweightedModeNoise = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Axis, typename Part>
inline constexpr bool isUnweightedModeNoise<ModeNoise<Axis, Part, false>> = true;

/// \endcond

/** The path to the mode axis whose values the components of the additive noise \a Noise are, or
 * \c void (see NoiseModeAxisOf).
 */
template <typename Noise>
using NoiseModeAxis = typename NoiseModeAxisOf<Noise>::type;

/** What holds a policy's place in a PolicySet that lacks it. */
struct NoPolicy
{};

/** The policies \a Policies given to a model, each at most once, found by kind. */
template <typename... Policies>
struct PolicySet
{
    static_assert(
        (ModelPolicy<Policies> && ...),
        "a model takes its function and then its policies, in any order: reads(), writes(), affineIn(), "
        "additiveNoise(), continuousNoise(), logLikelihood(), and outputSpace()"
    );

    /** The kinds given, in order. */
    static constexpr std::array<PolicyKind, sizeof...(Policies)> kinds{*policyKind<std::remove_cvref_t<Policies>>...};

    /** \returns the position of the policy of kind \a kind, or the number of policies if none is */
    [[nodiscard]] static consteval std::size_t
    position(PolicyKind kind)
    {
        for (std::size_t i = 0; i < kinds.size(); ++i)
        {
            if (kinds[i] == kind)
            {
                return i;
            }
        }
        return kinds.size();
    }

    /** \returns whether every kind is given at most once */
    [[nodiscard]] static consteval bool
    distinct()
    {
        for (std::size_t i = 0; i < kinds.size(); ++i)
        {
            for (std::size_t j = i + 1; j < kinds.size(); ++j)
            {
                if (kinds[i] == kinds[j])
                {
                    return false;
                }
            }
        }
        return true;
    }

    static_assert(distinct(), "a model takes each policy at most once");

    /** Whether the policy of kind \a K is given. */
    template <PolicyKind K>
    static constexpr bool has = position(K) < sizeof...(Policies);

    /** The policy of kind \a K, or NoPolicy. */
    template <PolicyKind K>
    using Type = std::conditional_t<
        has<K>,
        std::tuple_element_t<has<K> ? position(K) : 0, std::tuple<Policies..., NoPolicy>>,
        NoPolicy>;

    /** \returns the policy of kind \a K in \a policies, a tuple of the pack, which must hold it */
    template <PolicyKind K, typename Tuple>
    [[nodiscard]] static auto const &
    get(Tuple const & policies)
    {
        static_assert(has<K>, "the model was not given this policy");
        return std::get<has<K> ? position(K) : 0>(policies);
    }
};

/** \returns the placed part \a part of additive noise as a factor of a Product over \a space:
 * itself, a distribution placed by tag path
 */
template <typename Space, typename Part>
[[nodiscard]] Part const &
noiseFactor(Space const &, Part const & part)
{
    // NOLINTNEXTLINE(bugprone-return-const-ref-from-parameter): the identity case of the overloads
    return part;
}

/** The zero mean of a covariance of type \a X, as a column vector. */
template <typename X>
using ZeroMeanOf =
    Eigen::Matrix<typename X::Scalar, X::RowsAtCompileTime, 1, Eigen::ColMajor, X::MaxRowsAtCompileTime, 1>;

/** \returns the covariance \a part placed on a block as a zero-mean %Gaussian in the block's space */
template <typename Space, MatrixExpression X, typename Path>
[[nodiscard]] auto
noiseFactor(Space const & space, OnPath<X, Path> const & part)
{
    auto block = pathSubspace<Path>(space);
    using Noise = GaussianParams<decltype(block), typename X::Scalar>;
    return OnPath<Noise, Path>{Noise{std::move(block), ZeroMeanOf<X>::Zero(part.value.rows()), part.value}};
}

/** \returns the covariance \a part placed on the repetitions it selects as a zero-mean %Gaussian in
 * their space: the block that is repeated for one, or a repeated block of their count for several
 */
template <typename Space, MatrixExpression X, typename Path, typename Selection>
[[nodiscard]] auto
noiseFactor(Space const & space, OnPath<X, Path, Selection> const & part)
{
    auto block = pathChoiceSubspace<Path>(space, part.selection);
    using Noise = GaussianParams<decltype(block), typename X::Scalar>;
    return OnPath<Noise, Path, Selection>{
        Noise{std::move(block), ZeroMeanOf<X>::Zero(part.value.rows()), part.value},
        part.selection
    };
}

/** \returns the covariance \a part placed on each repetition it selects as a zero-mean %Gaussian in
 * one repetition's space
 */
template <typename Space, MatrixExpression X, typename Path, typename Selection>
[[nodiscard]] auto
noiseFactor(Space const &, OnEach<X, Path, Selection> const & part)
{
    using Block = decltype(pathRepetitionsSubspace<Path>(std::declval<Space const &>(), Eigen::Index{}));
    using Noise = GaussianParams<typename Block::BlockType, typename X::Scalar>;
    return OnEach<Noise, Path, Selection>{Noise{ZeroMeanOf<X>::Zero(part.value.rows()), part.value}, part.selection};
}

/** \returns the covariance \a part placed on each repetition of a block as a zero-mean %Gaussian in
 * one repetition's space
 */
template <typename Space, MatrixExpression X, typename Path>
[[nodiscard]] auto
noiseFactor(Space const &, OnEach<X, Path> const & part)
{
    using Block = typename EachBlockOf<Space, Path>::type;
    if constexpr (requires { typename Block::BlockType; })
    {
        using Noise = GaussianParams<typename Block::BlockType, typename X::Scalar>;
        return OnEach<Noise, Path>{Noise{ZeroMeanOf<X>::Zero(part.value.rows()), part.value}};
    }
    else
    {
        static_assert(
            std::is_void_v<X>,
            "onEach<Path>(R) in additiveNoise() places R on each repetition of a repeated block"
        );
    }
}

/** \returns the additive noise \a noise placed in \a space, the output space instance, as one
 * AdditiveNoise over the whole space: the unplaced distribution itself, by reference; a covariance
 * alone as a zero-mean %Gaussian over the space's differences in a Linear space; or the Product of
 * the placed parts, each a distribution or a zero-mean %Gaussian of a covariance in its block's
 * space, as one %Gaussian over the space's differences in a Linear space
 */
template <typename Space, typename... Parts>
[[nodiscard]] auto
placeNoise(Space const & space, AdditiveNoise<Parts...> const & noise)
{
    if constexpr (AdditiveNoise<Parts...>::ZeroMean)
    {
        auto const & cov = noise.distribution();
        using Cov = typename AdditiveNoise<Parts...>::Distribution;
        using Differences = DifferenceSpace<Space>;
        constexpr int Rows = valueDimensionAtCompileTime<Differences>;
        static_assert(
            Rows == Eigen::Dynamic || Cov::RowsAtCompileTime == Eigen::Dynamic || Cov::RowsAtCompileTime == Rows,
            "additiveNoise(R) is over the output's differences, so R has a row for each"
        );
        EMRYS_FILTER_ASSERT(
            cov.rows() == differenceDimension(space) && cov.cols() == cov.rows()
            && "additiveNoise(R) is over the output's differences, so R is square with a row for each"
        );
        using Noise = GaussianParams<Linear<Rows, maxValueDimensionAtCompileTime<Differences>>, typename Cov::Scalar>;
        return AdditiveNoise<Noise>{Noise{ZeroMeanOf<Cov>::Zero(cov.rows()), cov}};
    }
    else if constexpr (AdditiveNoise<Parts...>::Placed)
    {
        auto const product = std::apply(
            [&](auto const &... parts) {
                return Product{space, noiseFactor(space, parts)...};
            },
            noise.parts()
        );
        using Scalar = typename decltype(product)::Scalar;
        using Noise =
            GaussianParams<Linear<valueDimensionAtCompileTime<Space>, maxValueDimensionAtCompileTime<Space>>, Scalar>;
        auto const moments = product.moments();
        if constexpr (isExpected<std::remove_cvref_t<decltype(moments)>>)
        {
            EMRYS_FILTER_ASSERT(moments && "the noise's parts must have moments");
            return AdditiveNoise<Noise>{Noise{moments->mean(), moments->covarianceLower()}};
        }
        else
        {
            return AdditiveNoise<Noise>{Noise{moments.mean(), moments.covarianceLower()}};
        }
    }
    else
    {
        return AdditiveNoise<typename AdditiveNoise<Parts...>::Distribution const &>{noise.distribution()};
    }
}

/** The space of what the item \a Item of a reads() reads from a state in \a Space, as \c type: the
 * block its path names; for one selected repetition, the block that is repeated, or the block the
 * path names inside it; and for several, a repeated block of their count (decisions.md
 * [repetition-selections]).
 */
template <typename Space, typename Item>
struct ReadSpaceOf
{
    /** The tag of the block the path names. */
    using type = PathTag<typename Item::PathType>;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Space, typename Item>
    requires(Item::Selected && RepetitionIndex<typename Item::SelectionType>)
struct ReadSpaceOf<Space, Item>
{
    using type = decltype(pathElementSubspace<typename Item::PathType>(std::declval<Space const &>()));
};

template <typename Space, typename Item>
    requires(Item::Selected && !RepetitionIndex<typename Item::SelectionType>)
struct ReadSpaceOf<Space, Item>
{
    using type =
        decltype(pathRepetitionsSubspace<typename Item::PathType>(std::declval<Space const &>(), Eigen::Index{}));
};

/// \endcond

/** ReadSpaceOf, as a type. */
template <typename Space, typename Item>
using ReadSpace = typename ReadSpaceOf<Space, Item>::type;

/** Whether the tag \a Name can stand in a model's input for what an item reads, whose space is
 * \a Block: the two have the same leaf (sameLeaf), or, for several repetitions, both are repeated
 * blocks whose repetitions have, so that a name may bound their count more tightly than the state's
 * block does.
 */
template <typename Name, typename Block, bool Several>
inline constexpr bool nameFits = [] {
    if constexpr (sameLeaf<Name, Block>)
    {
        return true;
    }
    else if constexpr (Several && IsRepeated<Name> && IsRepeated<Block>)
    {
        return sameLeaf<RepeatedBlock<Name>, RepeatedBlock<Block>>;
    }
    else
    {
        return false;
    }
}();

/** Checks that each item of \a R, a Reads, that names a block of the state space \a Space is read
 * under a name with the same leaf as the block (sameLeaf), so that the model's input retracts as the
 * state does there. An item the state lacks names an input of the call, whose own space is its name's.
 *
 * \returns true, when it compiles
 */
template <typename R, typename Space>
[[nodiscard]] consteval bool
checkReadNames()
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        auto const check = []<typename Item>(std::type_identity<Item>) {
            using Path = typename Item::PathType;
            using Name = typename Item::NameType;
            constexpr bool inState = [] {
                return []<typename First, typename... Rest>(TagPath<First, Rest...> *) {
                    return countTag<First, Space>() > 0;
                }(static_cast<Path *>(nullptr));
            }();
            if constexpr (inState)
            {
                static_assert(
                    nameFits < Name,
                    ReadSpace<Space, Item>,
                    Item::Selected && !RepetitionIndex < typename Item::SelectionType >>,
                    "a name given by on<Name>() in reads() must have the same leaf space beneath it as the block it "
                    "names, such as Linear<2> for a Position : Linear<2>"
                );
            }
            return true;
        };
        return (check(std::type_identity<typename R::template Item<I>>{}) && ...);
    }(std::make_index_sequence<R::Count>{});
}

}  // namespace internal

/** \ingroup Models
 * \brief A sure input of predict() or observe(): a known value in the tagged space \a S, as sure()
 * marks it (decisions.md [prediction-inputs], [tag-addressing]). It names the item of the model's
 * reads() that the state lacks and has the tag of \a S, and reaches the function as a point, with no
 * spread.
 *
 * \tparam S       the input's space, a tag
 * \tparam Vector  the value's plain vector type
 */
template <Space S, typename Vector>
struct Sure
{
    /** the input's space, a tag */
    using SpaceType = S;
    /** the value's type */
    using VectorType = Vector;

    EMRYS_FILTER_NO_UNIQUE_ADDRESS S space;  ///< the space, with its sizes
    Vector value;                            ///< the value, a point of the space
};

/** \ingroup Models
 * \returns \a value, a vector, as the sure input in the fixed-size tagged space \a S:
 * <tt>sure<Gravity>(g)</tt>. Its size is checked against the space, at compile time where the
 * vector's is fixed.
 */
template <FixedSizeSpace S, MatrixExpression V>
    requires Space<S>
[[nodiscard]] auto
sure(V && value)
{
    using Plain = typename internal::CrtpDerived<V>::PlainObject;
    static_assert(!internal::IsUntagged<S>, "a sure input is named by a tag, a type derived from a space");
    static_assert(Plain::ColsAtCompileTime == 1, "a sure input is a column vector");
    static_assert(
        Plain::RowsAtCompileTime == Eigen::Dynamic || Plain::RowsAtCompileTime == valueDimensionAtCompileTime<S>,
        "a sure input's rows do not match its space's dimension at compile time"
    );
    EMRYS_FILTER_ASSERT(value.rows() == valueDimension(S{}) && "a sure input must be a value in its space");
    return Sure<S, Plain>{S{}, Plain{std::forward<V>(value)}};
}

/** \ingroup Models
 * \returns \a value as the sure input in \a space, a tagged space instance, for a resizable space,
 * whose size is checked against the instance
 */
template <Space S, MatrixExpression V>
[[nodiscard]] auto
sure(S space, V && value)
{
    using Plain = typename internal::CrtpDerived<V>::PlainObject;
    static_assert(!internal::IsUntagged<S>, "a sure input is named by a tag, a type derived from a space");
    static_assert(Plain::ColsAtCompileTime == 1, "a sure input is a column vector");
    EMRYS_FILTER_ASSERT(value.rows() == valueDimension(space) && "a sure input must be a value in its space");
    return Sure<S, Plain>{std::move(space), Plain{std::forward<V>(value)}};
}

namespace internal
{

/** Whether \a T is a Sure. */
template <typename T>
inline constexpr bool isSure = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename S, typename Vector>
inline constexpr bool isSure<Sure<S, Vector>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Models
 * \brief The blocks of the state that a Kalman update considers rather than estimates, as consider()
 * names them (decisions.md [consider-states]): the Schmidt-Kalman update of
 * \ref Schmidt1966 "Schmidt (1966)". The update keeps their mean and covariance block as the prior's,
 * and conditions the rest of the state, and its covariance with them, as the full update would.
 * <tt>Consider<></tt>, naming no block, is the full update, what an observation without a consider()
 * argument does.
 *
 * \tparam Paths  the blocks considered, each a TagPath looked up in the state's space
 */
template <typename... Paths>
struct Consider
{
    static constexpr std::size_t Count = sizeof...(Paths);  ///< the number of blocks considered
};

/** \ingroup Models
 * \returns the blocks named by \a paths, each a path value such as <tt>path<GyroBias></tt>, as the
 * argument of observe() that considers them rather than estimating them:
 * <tt>observe(methods::linearized(), model, z, consider(path<GyroBias>))</tt> (decisions.md
 * [consider-states]). Each path names a block of the state, which need not be one the model reads,
 * and no block lies in another.
 */
template <PathValue... Paths>
[[nodiscard]] constexpr Consider<internal::PathOf<Paths>...>
consider(Paths const &...)
{
    static_assert(sizeof...(Paths) > 0, "consider() names at least one block of the state");
    static_assert(internal::distinctTypes<internal::PathOf<Paths>...>(), "consider() names each block once");
    return {};
}

/** \ingroup Models
 * \brief The blocks of the state that a Kalman update considers, as consider() names them when one
 * of them selects repetitions of a repeated block (decisions.md [consider-states],
 * [repetition-selections]): each item a block, or the repetitions it selects, held by value.
 *
 * \tparam Items  the items, as internal::ReadItem
 */
template <typename... Items>
struct ConsiderItems : internal::ReadSelections<std::index_sequence_for<Items...>, Items...>
{
    static constexpr std::size_t Count = sizeof...(Items);  ///< the number of items

    /** The item at position \a I. */
    template <std::size_t I>
    using Item = std::tuple_element_t<I, std::tuple<Items...>>;

    /** The items holding \a selections, one for each in order. */
    explicit ConsiderItems(std::in_place_t, typename Items::SelectionType const &... selections)
        : internal::ReadSelections<std::index_sequence_for<Items...>, Items...>{std::in_place, selections...}
    {}

    /** \returns the selection of the item at position \a I, or internal::NoSelection */
    template <std::size_t I>
    [[nodiscard]] auto const &
    selection() const
    {
        return static_cast<internal::ItemSelection<I, typename Item<I>::SelectionType> const &>(*this).value;
    }
};

/** \ingroup Models
 * \returns the blocks named by \a items, each a path value or a selected path such as
 * <tt>path<Landmarks>[k]</tt>, as the argument of observe() that considers them: one repetition,
 * or each of a sequence or a list (decisions.md [consider-states], [placed-selected-values]). The
 * selections are held by value, and no coordinate is considered twice, which is asserted where the
 * update meets the state.
 */
template <typename... Items>
    requires(
        sizeof...(Items) > 0 && !(PathValue<Items> && ...)
        && (internal::ReadItemOf<std::remove_cvref_t<Items>>::valid && ...)
    )
[[nodiscard]] ConsiderItems<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type...>
consider(Items const &... items)
{
    return ConsiderItems<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type...>{
        std::in_place,
        internal::readSelection(items)...
    };
}

namespace internal
{

/** Whether \a T is a Consider. */
template <typename T>
inline constexpr bool isConsider = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Paths>
inline constexpr bool isConsider<Consider<Paths...>>{true};

template <typename... Items>
inline constexpr bool isConsider<ConsiderItems<Items...>>{true};
/// \endcond

}  // namespace internal

}  // namespace Emrys
