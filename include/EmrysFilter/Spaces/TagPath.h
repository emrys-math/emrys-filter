#pragma once

/** \file
 * Tag paths and the index utilities (decisions.md [tag-addressing], [index-names]): TagPath and TagPaths, and path,
 * a path as a value; on() and onEach(), which place a factor or a matrix by path in a space not yet known
 * [product-factors], or name a path differently; idx() and diffIdx(), which turn a path into Eigen
 * indices from a space's run-time sizes or as compile-time constants; block() and diffBlock(), the block an Eigen
 * vector or matrix has for a path; and zeros(), the vector a function fills block by block. The internal
 * namespace finds a tag in a space and resolves a path through composites and repeated blocks. Read Subspace.h
 * next.
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** A library space type itself rather than a tag derived from one, which a path can't name. */
template <typename T>
inline constexpr bool IsUntagged = false;
/// \cond INTERNAL_SPECIALIZATIONS
template <int Dimension, int MaxDimension>
inline constexpr bool IsUntagged<Linear<Dimension, MaxDimension>>{true};

template <PeriodicInterval Interval, int Dimension, int MaxDimension>
inline constexpr bool IsUntagged<Periodic<Interval, Dimension, MaxDimension>>{true};

template <typename... Blocks>
inline constexpr bool IsUntagged<Composite<Blocks...>>{true};

template <typename Block, int MaxCount>
inline constexpr bool IsUntagged<Repeated<Block, MaxCount>>{true};
/// \endcond

/** Number of times \a Tag appears in \a Space, not looking inside repeated blocks. */
template <typename Tag, typename Space>
consteval int
countTag();

/** countTag() summed over \a Blocks, the blocks of a Composite. */
template <typename Tag, typename... Blocks>
consteval int
countTagIn(std::tuple<Blocks...> *)
{
    return (countTag<Tag, Blocks>() + ... + 0);
}

template <typename Tag, typename Space>
consteval int
countTag()
{
    if constexpr (std::is_same_v<Tag, Space>)
    {
        return 1;
    }
    else if constexpr (IsComposite<Space>)
    {
        return countTagIn<Tag>(static_cast<CompositeBlocks<Space> *>(nullptr));
    }
    else
    {
        return 0;
    }
}

/** Where a tag sits in a space, as far as is known at compile time. */
struct TagLocation
{
    int offset = -1;            ///< offset, counting fixed sizes only
    bool staticOffset = false;  ///< no resizable block precedes the tag, so \c offset is exact
    bool tail = false;          ///< the tag's block ends the space
};

/** \a S, or its difference space if \a Difference: the space whose sizes an operand of values or
 * of differences takes.
 */
template <bool Difference, typename S>
struct SizedAsOf
{
    /** The space itself, for an operand of values. */
    using type = S;
};

/** The difference space of \a S, for an operand of differences. */
template <typename S>
struct SizedAsOf<true, S>
{
    /** The difference space. */
    using type = typename DifferenceSpaceOf<S>::type;
};

/** SizedAsOf, as a type. */
template <bool Difference, typename S>
using SizedAs = typename SizedAsOf<Difference, S>::type;

template <bool Difference, typename Tag, typename Space>
consteval TagLocation
locateTag();

/** locateTag() over \a Blocks, the blocks of a Composite, adding up the sizes before the match. */
template <bool Difference, typename Tag, typename... Blocks>
consteval TagLocation
locateTagIn(std::tuple<Blocks...> *)
{
    constexpr std::size_t count = sizeof...(Blocks);
    constexpr auto counts = std::array<int, count>{countTag<Tag, Blocks>()...};
    constexpr auto inner = std::array<TagLocation, count>{locateTag<Difference, Tag, Blocks>()...};
    constexpr auto sizes = std::array<int, count>{SizedAs<Difference, Blocks>::FixedValueDimension...};
    constexpr auto fixed = std::array<bool, count>{(Blocks::ValueDimensionStep == 0)...};
    int offset = 0;
    bool staticOffset = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (counts[i] > 0)
        {
            return {offset + inner[i].offset, staticOffset && inner[i].staticOffset, inner[i].tail && i + 1 == count};
        }
        offset += sizes[i];
        staticOffset = staticOffset && fixed[i];
    }
    return {};
}

/** Location of the first \a Tag within \a Space, counting value coordinates, or difference
 * coordinates if \a Difference; only consulted when countTag() finds it.
 */
template <bool Difference, typename Tag, typename Space>
consteval TagLocation
locateTag()
{
    if constexpr (std::is_same_v<Tag, Space>)
    {
        return {0, true, true};
    }
    else if constexpr (IsComposite<Space>)
    {
        return locateTagIn<Difference, Tag>(static_cast<CompositeBlocks<Space> *>(nullptr));
    }
    else
    {
        return {};
    }
}

/** Where a path of tags resolves to, as far as is known at compile time. */
struct ResolvedIndex
{
    int offset = 0;            ///< first coordinate, if \c staticOffset
    int size = 0;              ///< fixed size of the resolved block (0 if resizable)
    bool resizable = false;    ///< the resolved block is resizable
    int stride = 0;            ///< nonzero: one coordinate of every repetition of a repeated block
    bool staticOffset = true;  ///< no resizable block precedes it
    bool tail = true;          ///< it ends the space
};

/** resolveTags() counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename Space, typename Tag, typename... Rest>
consteval ResolvedIndex
resolveTagsIn()
{
    static_assert(!IsUntagged<Tag>, "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}");
    static_assert(countTag<Tag, Space>() > 0, "the space does not contain this tag");
    static_assert(countTag<Tag, Space>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
    using Sized = SizedAs<Difference, Tag>;
    constexpr TagLocation location = locateTag<Difference, Tag, Space>();
    if constexpr (sizeof...(Rest) == 0)
    {
        return {
            location.offset,
            Sized::FixedValueDimension,
            Tag::ValueDimensionStep != 0,
            0,
            location.staticOffset,
            location.tail
        };
    }
    else if constexpr (IsRepeated<Tag>)
    {
        constexpr ResolvedIndex inner = resolveTagsIn<Difference, RepeatedBlock<Tag>, Rest...>();
        static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
        return {
            location.offset + inner.offset,
            1,
            false,
            Sized::ValueDimensionStep,
            location.staticOffset,
            location.tail
        };
    }
    else
    {
        constexpr ResolvedIndex inner = resolveTagsIn<Difference, Tag, Rest...>();
        return {
            location.offset + inner.offset,
            inner.size,
            inner.resizable,
            inner.stride,
            location.staticOffset && inner.staticOffset,
            location.tail && inner.tail
        };
    }
}

/** Where the path \a Tag, \a Rest resolves to within \a Space, counting value coordinates. */
template <typename Space, typename Tag, typename... Rest>
consteval ResolvedIndex
resolveTags()
{
    return resolveTagsIn<false, Space, Tag, Rest...>();
}

/** \returns the Eigen index that the compile-time resolution \a R denotes: a single index, a
 * fixed \c seqN, or a \c seq to \c last for a resizable or repeated tail block; see idx().
 */
template <ResolvedIndex R>
constexpr auto
makeIndex()
{
    static_assert(R.staticOffset, "this path's offset depends on the size of a resizable block; use idx(space)");
    if constexpr (R.stride != 0)
    {
        static_assert(R.tail, "this repeated block is not at the end of the space; use idx(space)");
        return Eigen::seq(Eigen::fix<R.offset>, Eigen::placeholders::last, Eigen::fix<R.stride>);
    }
    else if constexpr (R.resizable)
    {
        static_assert(R.tail, "this resizable block is not at the end of the space; use idx(space)");
        return Eigen::seq(Eigen::fix<R.offset>, Eigen::placeholders::last);
    }
    else if constexpr (R.size == 1)
    {
        return Eigen::Index{R.offset};
    }
    else
    {
        return Eigen::seqN(Eigen::fix<R.offset>, Eigen::fix<R.size>);
    }
}

/** \returns the offset of the unique block of type \a Tag within \a space, counting value
 * coordinates, or difference coordinates if \a Difference, and a pointer to that block (const
 * if \a Space is).
 */
template <typename Tag, bool Difference = false, typename Space>
[[nodiscard]] auto
locate(Space & space)
{
    using Plain = std::remove_const_t<Space>;
    using TagPointer = std::conditional_t<std::is_const_v<Space>, Tag const *, Tag *>;
    if constexpr (std::is_same_v<Tag, Plain>)
    {
        return std::pair<Eigen::Index, TagPointer>{0, &space};
    }
    else
    {
        static_assert(
            !IsUntagged<Tag>,
            "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}"
        );
        static_assert(countTag<Tag, Plain>() > 0, "the space does not contain this tag");
        static_assert(countTag<Tag, Plain>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
        auto found = std::pair<Eigen::Index, TagPointer>{0, nullptr};
        Eigen::Index offset = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    using Block = std::tuple_element_t<I, CompositeBlocks<Plain>>;
                    auto & block = space.template block<I>();
                    if constexpr (countTag<Tag, Block>() > 0)
                    {
                        auto const inner = locate<Tag, Difference>(block);
                        found = {offset + inner.first, inner.second};
                    }
                    offset += rowsOf<Difference>(block);
                }(),
                ...
            );
        }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<Plain>>>{});
        return found;
    }
}

/** locate() along a path of tags, none of them inside a repeated block. */
template <bool Difference, typename Tag, typename... Rest, typename Space>
[[nodiscard]] auto
locatePathIn(Space & space)
{
    auto const [offset, tag] = locate<Tag, Difference>(space);
    if constexpr (sizeof...(Rest) == 0)
    {
        return std::pair{offset, tag};
    }
    else
    {
        static_assert(!IsRepeated<Tag>, "this path continues inside a repeated block");
        auto const [innerOffset, inner] = locatePathIn<Difference, Rest...>(*tag);
        return std::pair{offset + innerOffset, inner};
    }
}

/** locatePathIn() counting value coordinates. */
template <typename Tag, typename... Rest, typename Space>
[[nodiscard]] auto
locatePath(Space & space)
{
    return locatePathIn<false, Tag, Rest...>(space);
}

/** \returns \a index shifted by \a offset, keeping any compile-time size and increment. */
template <typename Index>
[[nodiscard]] auto
shiftIndex(Index const & index, Eigen::Index offset)
{
    if constexpr (std::is_integral_v<Index>)
    {
        return Eigen::Index{index + offset};
    }
    else
    {
        return Eigen::seqN(index.first() + offset, index.sizeObject(), index.incrObject());
    }
}

}  // namespace internal

/** \ingroup Spaces
 * A path of tags, each naming a block inside the one before: <tt>TagPath<Ownship, Position></tt>
 * is the Position block of Ownship. A bare tag is a path of one tag. Every template that takes
 * a path takes it as one argument, and one that takes several reads its arguments as a list
 * of paths, so a path of more than one tag is always spelled as a TagPath.
 *
 * Each tag must appear once where it is looked up, and lookup does not enter a Repeated block:
 * a tag inside one is reached only through the repeated block's tag, as in
 * <tt>TagPath<Fleet, Position></tt>, and is not counted against the same tag outside it. So in
 * a space of an Ownship and a Fleet of vehicles, each with a Position, the bare Position is the
 * ownship's.
 */
template <typename... Tags>
struct TagPath
{
    static_assert(sizeof...(Tags) > 0, "a tag path needs at least one tag");
};

/** \ingroup Spaces
 * A list of tag paths, each a tag or a TagPath, such as the blocks a measurement reads
 * (<tt>TagPaths<Position, TagPath<Sensor, Bias>></tt>).
 */
template <typename... Paths>
struct TagPaths
{};

/** \ingroup Spaces
 * The path <tt>TagPath<Tags...></tt> as a value, for a call that takes its paths as arguments rather
 * than as template arguments: <tt>reads(path<Position>, path<Ownship, Velocity>)</tt>. A bare tag's
 * own value, <tt>Position{}</tt>, is a path of one tag wherever a path value is taken (PathValue),
 * since a tag is an empty type.
 */
template <typename... Tags>
inline constexpr TagPath<Tags...> path{};

namespace internal
{

/** The path of onEach<>() given none: the whole space, which is repeated. */
struct WholeSpace
{};

/** What on() and onEach() hold of \a X: a matrix expression evaluated to its plain matrix, so that
 * nothing it refers to need outlive it, and anything else as it is.
 */
template <typename X>
struct HeldOf
{
    /** \a X itself. */
    using type = X;
};

/** A matrix expression, evaluated. */
template <MatrixExpression X>
struct HeldOf<X>
{
    /** The plain matrix. */
    using type = typename CrtpDerived<X>::PlainObject;
};

/** A diagonal, such as a covariance given by its variances, held as the dense matrix it is. */
template <DiagonalExpression X>
struct HeldOf<X>
{
    /** The plain dense matrix. */
    using type = typename X::DenseMatrixType;
};

/** \a value placed on the block that the path \a Path names, in a space not yet known (see on()). */
template <typename X, typename Path>
struct OnPath
{
    X value;  ///< the factor or matrix placed
};

/** \a value placed on every repetition of the repeated block that the path \a Path names, or of
 * the whole space if \a Path is WholeSpace, in a space not yet known (see onEach()).
 */
template <typename X, typename Path>
struct OnEach
{
    X value;  ///< the factor or matrix of one repetition
};

/** Whether \a X is a model, which declares <tt>static constexpr bool IsModel = true</tt>: on() and
 * onEach() of a model place the model, in Models/Placement.h, not a value.
 */
template <typename X>
concept PlacesModel = requires { requires std::remove_cvref_t<X>::IsModel; };

/** Whether \a T places a value by tag path, as on() and onEach() do. */
template <typename T>
inline constexpr bool isPlacedByPath = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename Path>
inline constexpr bool isPlacedByPath<OnPath<X, Path>>{true};

template <typename X, typename Path>
inline constexpr bool isPlacedByPath<OnEach<X, Path>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Spaces
 * \returns \a x placed on the block that the path \a Path (a tag or a TagPath) names, in a space
 * that is given later: a factor of a Product or of Factors, such as <tt>on<Heading>(uniform)</tt>,
 * or a sensor's matrix columns for the block's coordinates, in Columns (decisions.md
 * [product-factors]). A matrix expression is evaluated here. Given a path value, it names that path
 * by the tag \a Path where a list of paths takes each under a name:
 * <tt>on<SensorPosition>(path<Ownship, Position>)</tt> is the ownship's position under the name
 * SensorPosition (decisions.md [tag-addressing]).
 */
template <typename Path, typename X>
    requires(!internal::PlacesModel<X>)
[[nodiscard]] internal::OnPath<typename internal::HeldOf<X>::type, Path>
on(X x)
{
    return {std::move(x)};
}

/** \ingroup Spaces
 * \returns \a x, given for one repetition, placed on every repetition of the repeated block that
 * the path \a Path names, as many as the space given later has: a factor of Factors, such as
 * <tt>onEach<Landmarks>(GaussianParams<Landmark>{zero, P})</tt>, which is \f$ I \otimes P \f$, or
 * a sensor's matrix in Columns, which reads each repetition into a repetition of the measurement.
 * Without a path, <tt>onEach<>(x)</tt>, it is placed on every repetition of the whole space, which
 * is then repeated, such as a measurement space inferred by a sensor (decisions.md
 * [product-factors]).
 */
template <typename Path = internal::WholeSpace, typename X>
    requires(!internal::PlacesModel<X>)
[[nodiscard]] internal::OnEach<typename internal::HeldOf<X>::type, Path>
onEach(X x)
{
    return {std::move(x)};
}

namespace internal
{

/** The path \a P as a TagPath: a bare tag becomes a path of one tag. */
template <typename P>
struct AsTagPathImpl
{
    /** The path. */
    using type = TagPath<P>;
};

/** A TagPath is already one. */
template <typename... Tags>
struct AsTagPathImpl<TagPath<Tags...>>
{
    /** The path. */
    using type = TagPath<Tags...>;
};

/** The path \a P as a TagPath: a bare tag becomes a path of one tag. */
template <typename P>
using AsTagPath = typename AsTagPathImpl<P>::type;

/** Whether \a T is a TagPath. */
template <typename T>
inline constexpr bool isTagPath = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Tags>
inline constexpr bool isTagPath<TagPath<Tags...>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Spaces
 * A path given as a value: a TagPath, such as <tt>path<Ownship, Position></tt>, or a tag's own
 * value, such as <tt>Position{}</tt>, which is a path of one tag. A library space type, such as
 * <tt>Linear<2></tt>, is not one, since a path names blocks by tag (decisions.md [tag-addressing]).
 */
template <typename T>
concept PathValue = internal::isTagPath<std::remove_cvref_t<T>>
                 || (Space<std::remove_cvref_t<T>> && !internal::IsUntagged<std::remove_cvref_t<T>>);

namespace internal
{

/** The TagPath that a path value of type \a T denotes. */
template <PathValue T>
using PathOf = AsTagPath<std::remove_cvref_t<T>>;

/** \returns whether the block that the tags \a R, \a Rs... name lies in the one that \a W, \a Ws...
 * name, both looked up in one space: it is that block or inside it. A tag is unique where it is
 * looked up, so a tag found inside another is the one a lookup reaches.
 */
template <typename R, typename... Rs, typename W, typename... Ws>
consteval bool
tagsLieIn(TagPath<R, Rs...> *, TagPath<W, Ws...> *)
{
    if constexpr (countTag<R, W>() > 0)
    {
        // R is W, or inside it.
        if constexpr (sizeof...(Ws) == 0)
        {
            return true;
        }
        else if constexpr (!std::is_same_v<R, W>)
        {
            return tagsLieIn(static_cast<TagPath<R, Rs...> *>(nullptr), static_cast<TagPath<Ws...> *>(nullptr));
        }
        else if constexpr (sizeof...(Rs) == 0)
        {
            return false;
        }
        else
        {
            return tagsLieIn(static_cast<TagPath<Rs...> *>(nullptr), static_cast<TagPath<Ws...> *>(nullptr));
        }
    }
    else if constexpr (countTag<W, R>() > 0 && sizeof...(Rs) > 0)
    {
        // W is inside R, and the read path goes on inside R.
        return tagsLieIn(static_cast<TagPath<Rs...> *>(nullptr), static_cast<TagPath<W, Ws...> *>(nullptr));
    }
    else
    {
        return false;
    }
}

/** Whether the block named by the path \a Read lies in the block named by the path \a Within, each
 * a tag or a TagPath looked up in one space: it is that block or inside it.
 */
template <typename Read, typename Within>
inline constexpr bool pathLiesIn =
    tagsLieIn(static_cast<AsTagPath<Read> *>(nullptr), static_cast<AsTagPath<Within> *>(nullptr));

/** \returns \a f.template operator()<Tags...>() for the tags of the path \a P (a tag or a TagPath). */
template <typename P, typename F>
[[nodiscard]] constexpr decltype(auto)
withPathTags(F && f)
{
    return [&]<typename... Tags>(TagPath<Tags...> *) -> decltype(auto) {
        return f.template operator()<Tags...>();
    }(static_cast<AsTagPath<P> *>(nullptr));
}

/** resolveTagsIn() for the path \a P, a tag or a TagPath. */
template <typename Space, typename P, bool Difference = false>
struct ResolvePath
{
    static constexpr ResolvedIndex value = resolveTagsIn<Difference, Space, P>();  ///< the resolution
};

/** resolveTagsIn() for the tags of a TagPath, in order. */
template <typename Space, typename... Tags, bool Difference>
struct ResolvePath<Space, TagPath<Tags...>, Difference>
{
    static constexpr ResolvedIndex value = resolveTagsIn<Difference, Space, Tags...>();  ///< the resolution
};

/** idx() along the tags of a path, counting value coordinates, or difference coordinates if
 * \a Difference.
 */
template <bool Difference, typename Tag, typename... Rest, typename S>
[[nodiscard]] auto
pathIdx(S const & space)
{
    using Sized = SizedAs<Difference, Tag>;
    auto const [offset, tag] = locate<Tag, Difference>(space);
    if constexpr (sizeof...(Rest) == 0)
    {
        if constexpr (Tag::ValueDimensionStep != 0)
        {
            return Eigen::seqN(offset, rowsOf<Difference>(*tag));
        }
        else if constexpr (Sized::FixedValueDimension == 1)
        {
            return Eigen::Index{offset};
        }
        else
        {
            return Eigen::seqN(offset, Eigen::fix<Sized::FixedValueDimension>);
        }
    }
    else if constexpr (IsRepeated<Tag>)
    {
        constexpr ResolvedIndex inner = resolveTagsIn<Difference, RepeatedBlock<Tag>, Rest...>();
        static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
        return Eigen::seqN(offset + inner.offset, tag->count(), Eigen::fix<Sized::ValueDimensionStep>);
    }
    else
    {
        return shiftIndex(pathIdx<Difference, Rest...>(*tag), offset);
    }
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the index of the subspace named by the path \a Path (a tag or a TagPath) within the
 * space type \a S, from the type alone, for direct use in Eigen indexing:
 *  - a single fixed coordinate: an \c Eigen::Index, as a constant expression
 *    (<tt>static_assert(idx<State, Heading>() == 3)</tt>);
 *  - a fixed block: \c seqN(fix<offset>, fix<size>);
 *  - a resizable block at the end of the space: \c seq(fix<offset>, last);
 *  - a path through a repeated block at the end of the space to one of its coordinates,
 *    such as <tt>idx<World, TagPath<Vehicles, Altitude>>()</tt>: that coordinate of every
 *    repetition, \c seq(fix<offset>, last, fix<stride>).
 *
 * Only available when no resizable block precedes the path's block, and a resizable or
 * repeated block ends the space; otherwise idx<Path>(space), which works for every path. A tag
 * that appears more than once must be qualified by an enclosing tag; one inside a repeated block
 * is reached only through it, and is not counted against the same tag outside (see TagPath).
 *
 * The overloads of idx() are told apart by their arguments: idx<S, Path>() from the type,
 * idx<Path>(space) from a space's sizes, and idx<Path>(space, element) and
 * idx<S, Path>(element) for one repetition. Each indexes values, such as a mean; diffIdx(),
 * with the same overloads, indexes differences, such as the rows of a covariance.
 */
template <Space S, typename Path>
[[nodiscard]] constexpr auto
idx()
{
    return internal::makeIndex<internal::ResolvePath<S, Path>::value>();
}

/** \ingroup Spaces
 * idx<S, Path>() of the differences of the subspace named by \a Path: its rows of a covariance.
 * It differs from idx() where a block's differences have other sizes than its values.
 */
template <Space S, typename Path>
[[nodiscard]] constexpr auto
diffIdx()
{
    return internal::makeIndex<internal::ResolvePath<S, Path, true>::value>();
}

/** \ingroup Spaces
 * \returns the index of the subspace named by the path \a Path (a tag or a TagPath) within
 * \a space, from its runtime sizes: an \c Eigen::Index for a single fixed coordinate;
 * \c seqN(offset, fix<size>) for a fixed block; \c seqN(offset, size) for a resizable one; and
 * for a path through a repeated block to one of its coordinates, \c seqN(offset, count, fix<stride>).
 * It works for every path, where idx<S, Path>() needs every offset fixed by the type.
 */
template <typename Path>
[[nodiscard]] auto
idx(Space auto const & space)
{
    return internal::withPathTags<Path>([&]<typename... Tags>() {
        return internal::pathIdx<false, Tags...>(space);
    });
}

/** \ingroup Spaces
 * idx<Path>(space) of the differences of the subspace named by \a Path: its rows of a covariance.
 */
template <typename Path>
[[nodiscard]] auto
diffIdx(Space auto const & space)
{
    return internal::withPathTags<Path>([&]<typename... Tags>() {
        return internal::pathIdx<true, Tags...>(space);
    });
}

namespace internal
{

/** Index within one repetition of \a RepeatedTag of the coordinates \a Inner (all of it if none),
 * counting value coordinates, or difference coordinates if \a Difference.
 */
template <bool Difference, typename RepeatedTag, typename... Inner>
consteval ResolvedIndex
repetitionIndex()
{
    static_assert(IsRepeated<RepeatedTag>, "the element form takes a path that starts at a repeated block");
    if constexpr (sizeof...(Inner) == 0)
    {
        return ResolvedIndex{0, SizedAs<Difference, RepeatedTag>::ValueDimensionStep, false, 0};
    }
    else
    {
        return resolveTagsIn<Difference, RepeatedBlock<RepeatedTag>, Inner...>();
    }
}

/** \returns the index of the coordinates \a Inner of one repetition that starts at \a start: a
 * single index, or a fixed \c seqN
 */
template <ResolvedIndex Inner>
[[nodiscard]] auto
repetitionIndexAt(Eigen::Index start)
{
    if constexpr (Inner.size == 1)
    {
        return start;
    }
    else
    {
        return Eigen::seqN(start, Eigen::fix<Inner.size>);
    }
}

/** idx(space, element), counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename Path, typename S>
[[nodiscard]] auto
elementIdx(S const & space, Eigen::Index element)
{
    return withPathTags<Path>([&]<typename RepeatedTag, typename... Inner>() {
        auto const [offset, tag] = locate<RepeatedTag, Difference>(space);
        EMRYS_FILTER_ASSERT(element >= 0 && element < tag->count() && "repetition out of range");
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        return repetitionIndexAt<inner>(
            offset + element * SizedAs<Difference, RepeatedTag>::ValueDimensionStep + inner.offset
        );
    });
}

/** idx<S, Path>(element), counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename S, typename Path>
[[nodiscard]] auto
staticElementIdx(Eigen::Index element)
{
    return withPathTags<Path>([&]<typename RepeatedTag, typename... Inner>() {
        constexpr ResolvedIndex outer = resolveTagsIn<Difference, S, RepeatedTag>();
        static_assert(
            outer.staticOffset,
            "this block's offset depends on the size of a resizable block; use idx(space, element)"
        );
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        return repetitionIndexAt<inner>(
            outer.offset + element * SizedAs<Difference, RepeatedTag>::ValueDimensionStep + inner.offset
        );
    });
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the index of repetition \a element of a repeated block within \a space, from the
 * space's runtime sizes. \a Path starts at the repeated block: <tt>idx<Fleet>(space, k)</tt>
 * is vehicle \a k, and <tt>idx<TagPath<Fleet, Heading>>(space, k)</tt> its heading. The
 * result is an \c Eigen::Index for one coordinate, otherwise \c seqN(start, fix<size>).
 * \a element must be one of its repetitions.
 */
template <typename Path>
[[nodiscard]] auto
idx(Space auto const & space, Eigen::Index element)
{
    return internal::elementIdx<false, Path>(space, element);
}

/** \ingroup Spaces
 * idx<Path>(space, element) of the differences of one repetition: its rows of a covariance.
 */
template <typename Path>
[[nodiscard]] auto
diffIdx(Space auto const & space, Eigen::Index element)
{
    return internal::elementIdx<true, Path>(space, element);
}

/** \ingroup Spaces
 * idx<Path>(space, element) for the space type \a S, without a space object. Only available
 * when no resizable block precedes the repeated block, and \a element is not range-checked, since
 * the repetition count is unknown; otherwise use idx<Path>(space, element).
 */
template <Space S, typename Path>
[[nodiscard]] auto
idx(Eigen::Index element)
{
    return internal::staticElementIdx<false, S, Path>(element);
}

/** \ingroup Spaces
 * idx<S, Path>(element) of the differences of one repetition.
 */
template <Space S, typename Path>
[[nodiscard]] auto
diffIdx(Eigen::Index element)
{
    return internal::staticElementIdx<true, S, Path>(element);
}

/** \ingroup Spaces
 * Rows of a contiguous block of a space.
 */
struct BlockRows
{
    Eigen::Index start;  ///< first row
    Eigen::Index size;   ///< number of rows
};

namespace internal
{

/** \returns the rows of the block named by the path \a P (a tag or a TagPath) in \a space, from
 * its runtime sizes, in an operand of values such as a mean. The path must not continue inside
 * a repeated block, so the rows are contiguous.
 *
 * \sa differenceBlockRows()
 */
template <typename P, typename Space>
[[nodiscard]] BlockRows
blockRows(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        auto const [offset, tag] = locatePathIn<false, Tags...>(space);
        return BlockRows{offset, valueDimension(*tag)};
    });
}

/** blockRows() in an operand of differences, such as the rows of a covariance.
 */
template <typename P, typename Space>
[[nodiscard]] BlockRows
differenceBlockRows(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        auto const [offset, tag] = locatePathIn<true, Tags...>(space);
        return BlockRows{offset, differenceDimension(*tag)};
    });
}

/** Checks at compile time that \a X, an Eigen expression whose rows are the coordinates of a space
 * with \a Rows rows at compile time, has that many rows where both are fixed, and that it is not a
 * temporary plain matrix, whose block would dangle.
 */
template <int Rows, typename X>
consteval void
checkBlockRows()
{
    using Derived = CrtpDerived<X>;
    static_assert(
        Derived::RowsAtCompileTime == Eigen::Dynamic || Rows == Eigen::Dynamic || Derived::RowsAtCompileTime == Rows,
        "the rows do not match the space's dimension at compile time"
    );
    static_assert(
        std::is_lvalue_reference_v<X> || !std::derived_from<std::remove_cvref_t<X>, Eigen::PlainObjectBase<Derived>>,
        "a block of a temporary matrix would dangle; store it first"
    );
}

/** checkBlockRows() for the columns of \a M, a matrix whose columns are the coordinates of a space
 * with \a Columns columns at compile time.
 */
template <int Columns, typename M>
consteval void
checkBlockColumns()
{
    using Derived = CrtpDerived<M>;
    static_assert(
        Derived::ColsAtCompileTime == Eigen::Dynamic || Columns == Eigen::Dynamic
            || Derived::ColsAtCompileTime == Columns,
        "the columns do not match the space's dimension at compile time"
    );
}

/** \returns the rows of \a x for the block named by the path \a Path in \a space, where the rows of
 * \a x are the coordinates of \a space in values, or in differences if \a Difference: a block of as
 * many rows at compile time as the block's type fixes. See block().
 */
template <bool Difference, typename Path, typename S, typename X>
[[nodiscard]] auto
rowsByPath(S const & space, X && x)
{
    checkBlockRows<rowsAtCompileTime<Difference, S>, X>();
    EMRYS_FILTER_ASSERT(x.rows() == rowsOf<Difference>(space) && "the rows do not match the space");
    return withPathTags<Path>([&]<typename... Tags>() {
        auto const [offset, tag] = locatePathIn<Difference, Tags...>(space);
        using Block = std::remove_cvref_t<decltype(*tag)>;
        return middleRowsOf<rowsAtCompileTime<Difference, Block>>(x, offset, rowsOf<Difference>(*tag));
    });
}

/** rowsByPath() for repetition \a element of the repeated block that \a Path starts at, or of the
 * block the rest of the path names within it. See block(space, x, element).
 */
template <bool Difference, typename Path, typename S, typename X>
[[nodiscard]] auto
repetitionRowsByPath(S const & space, X && x, Eigen::Index element)
{
    checkBlockRows<rowsAtCompileTime<Difference, S>, X>();
    EMRYS_FILTER_ASSERT(x.rows() == rowsOf<Difference>(space) && "the rows do not match the space");
    return withPathTags<Path>([&]<typename RepeatedTag, typename... Inner>() {
        auto const [offset, tag] = locate<RepeatedTag, Difference>(space);
        EMRYS_FILTER_ASSERT(element >= 0 && element < tag->count() && "repetition out of range");
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        return middleRowsOf<inner.size>(
            x,
            offset + element * SizedAs<Difference, RepeatedTag>::ValueDimensionStep + inner.offset,
            inner.size
        );
    });
}

/** \returns the block of \a m at the rows of the path \a Row in \a rowSpace and the columns of the
 * path \a Column in \a columnSpace, both in differences: a block of as many rows and columns at
 * compile time as the blocks' types fix. See diffBlock().
 */
template <typename Row, typename Column, typename RowSpace, typename ColumnSpace, typename M>
[[nodiscard]] auto
differenceBlockByPaths(RowSpace const & rowSpace, ColumnSpace const & columnSpace, M && m)
{
    checkBlockRows<rowsAtCompileTime<true, RowSpace>, M>();
    checkBlockColumns<rowsAtCompileTime<true, ColumnSpace>, M>();
    EMRYS_FILTER_ASSERT(
        m.rows() == rowsOf<true>(rowSpace) && m.cols() == rowsOf<true>(columnSpace)
        && "the matrix's rows and columns do not match the spaces"
    );
    auto const [rowOffset, rowTag] = withPathTags<Row>([&]<typename... Tags>() {
        return locatePathIn<true, Tags...>(rowSpace);
    });
    auto const [columnOffset, columnTag] = withPathTags<Column>([&]<typename... Tags>() {
        return locatePathIn<true, Tags...>(columnSpace);
    });
    using RowBlock = std::remove_cvref_t<decltype(*rowTag)>;
    using ColumnBlock = std::remove_cvref_t<decltype(*columnTag)>;
    return blockOf<rowsAtCompileTime<true, RowBlock>, rowsAtCompileTime<true, ColumnBlock>>(
        m,
        rowOffset,
        columnOffset,
        rowsOf<true>(*rowTag),
        rowsOf<true>(*columnTag)
    );
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the block named by the path \a Path (a tag or a TagPath) of \a x, an Eigen vector of
 * values in \a space, or a matrix whose columns are such values: Eigen's block of its rows, with as
 * many rows at compile time as the block's type fixes, so that it indexes as a hand-written segment
 * does, and writable where \a x is (<tt>block<Position>(out, y) = ...</tt>). The rows of \a x are
 * checked against \a space, at compile time where both are fixed: which space a vector is in is
 * never inferred from its size (decisions.md [tag-addressing]). The path must not continue inside a
 * Repeated block, whose repetitions block(space, x, element) reaches one at a time; a temporary
 * plain matrix is refused, since its block would dangle. For an operand of differences, such as a
 * covariance's rows, see diffBlock(); for a fixed-size space named by type, block<S, Path>(x).
 */
template <typename Path, MatrixExpression X>
[[nodiscard]] auto
block(Space auto const & space, X && x)
{
    return internal::rowsByPath<false, Path>(space, std::forward<X>(x));
}

/** \ingroup Spaces
 * \returns block<Path>(space, x) for repetition \a element of the Repeated block that \a Path starts
 * at, or of the block the rest of the path names within it: <tt>block<Fleet>(space, x, k)</tt> is
 * vehicle \a k, and <tt>block<TagPath<Fleet, Heading>>(space, x, k)</tt> its heading. \a element
 * must be one of its repetitions.
 */
template <typename Path, MatrixExpression X>
[[nodiscard]] auto
block(Space auto const & space, X && x, Eigen::Index element)
{
    return internal::repetitionRowsByPath<false, Path>(space, std::forward<X>(x), element);
}

/** \ingroup Spaces
 * \returns block<Path>(space, x) in the fixed-size space \a S, named as a template argument:
 * <tt>block<State, Position>(x)</tt> (decisions.md [space-storage]).
 */
template <FixedSizeSpace S, typename Path, MatrixExpression X>
    requires Space<S>
[[nodiscard]] auto
block(X && x)
{
    return internal::rowsByPath<false, Path>(S{}, std::forward<X>(x));
}

/** \ingroup Spaces
 * \returns the block named by the path \a Path of \a d, an Eigen vector of differences in \a space,
 * or a matrix whose columns are such differences, as block() does for values: its rows are the
 * block's rows of a covariance, which differ from its value rows where the block's differences have
 * another size than its values, as a unit quaternion's do.
 */
template <typename Path, MatrixExpression D>
[[nodiscard]] auto
diffBlock(Space auto const & space, D && d)
{
    return internal::rowsByPath<true, Path>(space, std::forward<D>(d));
}

/** \ingroup Spaces
 * \returns diffBlock<Path>(space, d) for repetition \a element of the Repeated block that \a Path
 * starts at, as block(space, x, element) does for values.
 */
template <typename Path, MatrixExpression D>
[[nodiscard]] auto
diffBlock(Space auto const & space, D && d, Eigen::Index element)
{
    return internal::repetitionRowsByPath<true, Path>(space, std::forward<D>(d), element);
}

/** \ingroup Spaces
 * \returns diffBlock<Path>(space, d) in the fixed-size space \a S, named as a template argument:
 * <tt>diffBlock<State, Attitude>(d)</tt>.
 */
template <FixedSizeSpace S, typename Path, MatrixExpression D>
    requires Space<S>
[[nodiscard]] auto
diffBlock(D && d)
{
    return internal::rowsByPath<true, Path>(S{}, std::forward<D>(d));
}

/** \ingroup Spaces
 * \returns the block of \a m, a matrix over differences with its rows in \a rowSpace and its columns
 * in \a columnSpace, such as a Jacobian or a cross-covariance, at the rows of the path \a Row and
 * the columns of the path \a Column: Eigen's block with as many rows and columns at compile time as
 * the two blocks' types fix, writable where \a m is. Both sizes of \a m are checked against the
 * spaces, at compile time where they are fixed.
 */
template <typename Row, typename Column, MatrixExpression M>
[[nodiscard]] auto
diffBlock(Space auto const & rowSpace, Space auto const & columnSpace, M && m)
{
    return internal::differenceBlockByPaths<Row, Column>(rowSpace, columnSpace, std::forward<M>(m));
}

/** \ingroup Spaces
 * \returns diffBlock<Row, Column>(space, space, m): the block of a matrix over the differences of
 * one space on both axes, such as a covariance.
 */
template <typename Row, typename Column, MatrixExpression M>
[[nodiscard]] auto
diffBlock(Space auto const & space, M && m)
{
    return internal::differenceBlockByPaths<Row, Column>(space, space, std::forward<M>(m));
}

/** \ingroup Spaces
 * \returns diffBlock<Row, Column>(space, m) in the fixed-size space \a S, named as a template
 * argument: <tt>diffBlock<State, Position, Velocity>(cov)</tt>.
 */
template <FixedSizeSpace S, typename Row, typename Column, MatrixExpression M>
    requires Space<S>
[[nodiscard]] auto
diffBlock(M && m)
{
    return internal::differenceBlockByPaths<Row, Column>(S{}, S{}, std::forward<M>(m));
}

/** \ingroup Spaces
 * \returns a zero Eigen vector of \a Scalar with the value dimension of \a space: the vector a
 * function fills block by block, <tt>auto y = zeros<T>(out); block<Position>(out, y) = ...;</tt>,
 * with \c T the scalar the function is called in (decisions.md [function-scalars]), so that a
 * function differentiated automatically builds its result in dual numbers. It is zero rather than
 * uninitialized, so that a coordinate the function leaves unwritten reads as zero and no compiler
 * warns of a read before a write. In a fixed-size space it is <tt>ValueVector<S, T>::Zero()</tt>.
 */
template <typename Scalar, Space S>
[[nodiscard]] ValueVector<S, Scalar>
zeros(S const & space)
{
    return ValueVector<S, Scalar>::Zero(valueDimension(space));
}

}  // namespace Emrys
