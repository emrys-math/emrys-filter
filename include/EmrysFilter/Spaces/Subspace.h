#pragma once

/** \file
 * Subspaces, and growing and shrinking blocks (decisions.md [several-resizable-blocks]). Everything here is in
 * `internal`: the space a path names and the space of the coordinates outside it, for marginals and conditionals, and
 * the rows that insert() and erase() add to or remove from an elementwise block. Read Reset.h next, for the covariance
 * reset.
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The last tag of the path \a P: a bare tag is its own. */
template <typename P>
struct PathTagOf
{
    /** The tag. */
    using type = P;
};

/** The last tag of a TagPath. */
template <typename... Tags>
struct PathTagOf<TagPath<Tags...>>
{
    /** The tag. */
    using type = std::tuple_element_t<sizeof...(Tags) - 1, std::tuple<Tags...>>;
};

/** The type of the block named by the path \a P: its last tag. */
template <typename P>
using PathTag = typename PathTagOf<P>::type;

/** Whether the path \a P runs through a Repeated block: a tag before its last is repeated, so
 * idx<P>(space) is a strided index over every repetition rather than one block. A bare tag runs
 * through none.
 */
template <typename P>
inline constexpr bool RunsThroughRepeated = false;
/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Tags>
inline constexpr bool RunsThroughRepeated<TagPath<Tags...>> = [] {
    constexpr bool repeated[] = {IsRepeated<Tags>...};
    for (std::size_t i = 0; i + 1 < sizeof...(Tags); ++i)
    {
        if (repeated[i])
        {
            return true;
        }
    }
    return false;
}();

/// \endcond

/** \returns the block named by the path \a P in \a space, with its parameters and size. */
template <typename P, typename Space>
[[nodiscard]] PathTag<P> const &
pathBlock(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() -> PathTag<P> const & {
        auto const * block = locatePath<Tags...>(space).second;
        return *block;
    });
}

/** \returns the unique block of type \a Tag within \a space, with its parameters and size. */
template <typename Tag, typename Space>
[[nodiscard]] Tag const &
findTag(Space const & space)
{
    Tag const * block = locate<Tag>(space).second;
    return *block;
}

/** \returns the space of the subspace named by \a Tag and \a Rest within \a space: the
 * tag's own block, or for a path through a repeated block, the selected coordinate
 * repeated as many times, up to the same maximum count.
 */
template <typename Space, typename Tag, typename... Rest>
[[nodiscard]] auto
subspace(Space const & space)
{
    Tag const & tag = findTag<Tag>(space);
    if constexpr (sizeof...(Rest) == 0)
    {
        return tag;
    }
    else if constexpr (IsRepeated<Tag>)
    {
        auto inner = subspace<RepeatedBlock<Tag>, Rest...>(tag.block());
        return Repeated<decltype(inner), Tag::MaxCount>{std::move(inner), tag.count()};
    }
    else
    {
        return subspace<Tag, Rest...>(tag);
    }
}

/** The space of the subspace named by \a Path within \a Space, as subspace() returns it. */
template <typename Space, typename... Path>
using SubspaceType = decltype(subspace<Space, Path...>(std::declval<Space const &>()));

/** \returns the space of one repetition of the repeated block \a RepeatedTag within \a space,
 * or of the subspace named by \a Inner within it.
 */
template <typename RepeatedTag, typename... Inner, typename Space>
[[nodiscard]] auto
elementSubspace(Space const & space)
{
    static_assert(IsRepeated<RepeatedTag>, "this takes a repeated block");
    auto const & block = findTag<RepeatedTag>(space).block();
    if constexpr (sizeof...(Inner) == 0)
    {
        return block;
    }
    else
    {
        return subspace<RepeatedBlock<RepeatedTag>, Inner...>(block);
    }
}

/** \returns \a blocks as one space: the block itself if there is one, Linear<0> if there are none,
 * otherwise their Composite.
 */
template <typename... Blocks>
[[nodiscard]] auto
joinBlocks(std::tuple<Blocks...> blocks)
{
    if constexpr (sizeof...(Blocks) == 0)
    {
        return Linear<0>{};
    }
    else if constexpr (sizeof...(Blocks) == 1)
    {
        return std::get<0>(std::move(blocks));
    }
    else
    {
        return std::apply(
            [](Blocks &... b) {
                return Composite<Blocks...>{std::move(b)...};
            },
            blocks
        );
    }
}

template <typename Tag, typename... Rest, typename Space>
[[nodiscard]] auto
blocksWithout(Space const & space);  // defined below; blockWithout() recurses into it

/** blocksWithout() for one block of a Composite: the block itself if the path's first tag is
 * not in it.
 */
template <typename Tag, typename... Rest, typename Block>
[[nodiscard]] auto
blockWithout(Block const & block)
{
    if constexpr (countTag<Tag, Block>() > 0)
    {
        return blocksWithout<Tag, Rest...>(block);
    }
    else
    {
        return std::tuple<Block>{block};
    }
}

/** \returns the blocks of \a space left when the subspace named by the path \a Tag, \a Rest is
 * removed, in order, as a tuple: empty if the path names all of \a space.
 *
 * A Composite that loses a block is replaced by its remaining blocks, since it no longer
 * holds what its tag names. A Repeated block that loses the same coordinates from every
 * repetition stays a Repeated block of what is left.
 */
template <typename Tag, typename... Rest, typename Space>
[[nodiscard]] auto
blocksWithout(Space const & space)
{
    if constexpr (std::is_same_v<Tag, Space>)
    {
        if constexpr (sizeof...(Rest) == 0)
        {
            return std::tuple<>{};
        }
        else if constexpr (IsRepeated<Tag>)
        {
            auto rest = blocksWithout<Rest...>(space.block());
            if constexpr (std::tuple_size_v<decltype(rest)> == 0)
            {
                return std::tuple<>{};
            }
            else
            {
                auto block = joinBlocks(std::move(rest));
                return std::tuple<Repeated<decltype(block), Tag::MaxCount>>{
                    Repeated<decltype(block), Tag::MaxCount>{std::move(block), space.count()}
                };
            }
        }
        else
        {
            return blocksWithout<Rest...>(space);
        }
    }
    else
    {
        static_assert(
            !IsUntagged<Tag>,
            "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}"
        );
        static_assert(IsComposite<Space>, "the space does not contain this tag");
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::tuple_cat(blockWithout<Tag, Rest...>(space.template block<I>())...);
        }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<Space>>>{});
    }
}

/** \returns the space of the coordinates of \a space outside the subspace named by \a Given,
 * in ascending order: the remaining blocks, joined by joinBlocks().
 */
template <typename... Given, typename Space>
[[nodiscard]] auto
complementSubspace(Space const & space)
{
    return joinBlocks(blocksWithout<Given...>(space));
}

/** subspace() of the path \a P, a tag or a TagPath. */
template <typename P, typename Space>
[[nodiscard]] auto
pathSubspace(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        return subspace<Space, Tags...>(space);
    });
}

/** The space of the subspace named by the path \a P within \a Space. */
template <typename Space, typename P>
using PathSubspaceType = decltype(pathSubspace<P>(std::declval<Space const &>()));

/** elementSubspace() of the path \a P, which starts at a repeated block. */
template <typename P, typename Space>
[[nodiscard]] auto
pathElementSubspace(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        return elementSubspace<Tags...>(space);
    });
}

/** complementSubspace() of the path \a P. */
template <typename P, typename Space>
[[nodiscard]] auto
pathComplement(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        return complementSubspace<Tags...>(space);
    });
}

/** A block that insert() and erase() can grow and shrink element by element: a Repeated
 * block (elements are repetitions) or a resizable Linear or Periodic block (elements are
 * coordinates).
 */
template <typename Block>
concept IsElementwise = IsRepeated<Block> || (IsUniformLeaf<Block> && Block::ValueDimensionStep != 0);

/** \returns the number of elements of \a block: repetitions, or coordinates of a resizable leaf */
template <IsElementwise Block>
[[nodiscard]] Eigen::Index
elementCount(Block const & block)
{
    if constexpr (IsRepeated<Block>)
    {
        return block.count();
    }
    else
    {
        return block.valueDimension();
    }
}

/** Sets the number of elements of \a block to \a count; see elementCount(). */
template <IsElementwise Block>
void
setElementCount(Block & block, Eigen::Index count)
{
    if constexpr (IsRepeated<Block>)
    {
        block.setCount(count);
    }
    else
    {
        block.resize(count);
    }
}

/** Rows of a block of elements in the full space. */
struct ElementRows
{
    Eigen::Index start;            ///< first row of values
    Eigen::Index count;            ///< number of rows of values
    Eigen::Index differenceStart;  ///< first row of differences
    Eigen::Index differenceCount;  ///< number of rows of differences
};

/** \returns the offsets of the elementwise block named by \a Path in \a space, in values and in
 * differences, and a pointer to it.
 */
template <typename Path, typename Space>
[[nodiscard]] auto
locateElementwise(Space & space)
{
    auto [offset, block] = withPathTags<Path>([&]<typename... Tags>() {
        return locatePathIn<false, Tags...>(space);
    });
    Eigen::Index const differenceOffset = withPathTags<Path>([&]<typename... Tags>() {
        return locatePathIn<true, Tags...>(space).first;
    });
    using Block = std::remove_pointer_t<decltype(block)>;
    static_assert(
        IsElementwise<Block>,
        "insert() and erase() take a Repeated block or a resizable Linear or Periodic block"
    );
    return std::tuple{offset, differenceOffset, block};
}

/** Grows the elementwise block named by \a Path by \a rows rows of values, inserted before its
 * element \a element. \returns the rows they occupy, in values and in differences.
 */
template <typename Path, typename Space>
[[nodiscard]] ElementRows
growBlock(Space & space, Eigen::Index element, Eigen::Index rows)
{
    auto [offset, differenceOffset, block] = locateElementwise<Path>(space);
    using Block = std::remove_pointer_t<decltype(block)>;
    constexpr Eigen::Index elementSize = Block::ValueDimensionStep;
    constexpr Eigen::Index differenceElementSize = SizedAs<true, Block>::ValueDimensionStep;
    Eigen::Index const count = elementCount(*block);
    EMRYS_FILTER_ASSERT(element >= 0 && element <= count && "insertion point out of range");
    EMRYS_FILTER_ASSERT(rows % elementSize == 0 && "insert whole elements");
    setElementCount(*block, count + rows / elementSize);
    return {
        offset + element * elementSize,
        rows,
        differenceOffset + element * differenceElementSize,
        rows / elementSize * differenceElementSize
    };
}

/** Shrinks the elementwise block named by \a Path by \a elements elements starting at its
 * element \a element. \returns the rows they occupied, in values and in differences.
 */
template <typename Path, typename Space>
[[nodiscard]] ElementRows
shrinkBlock(Space & space, Eigen::Index element, Eigen::Index elements)
{
    auto [offset, differenceOffset, block] = locateElementwise<Path>(space);
    using Block = std::remove_pointer_t<decltype(block)>;
    constexpr Eigen::Index elementSize = Block::ValueDimensionStep;
    constexpr Eigen::Index differenceElementSize = SizedAs<true, Block>::ValueDimensionStep;
    Eigen::Index const count = elementCount(*block);
    EMRYS_FILTER_ASSERT(element >= 0 && elements >= 0 && element + elements <= count && "erased elements out of range");
    setElementCount(*block, count - elements);
    return {
        offset + element * elementSize,
        elements * elementSize,
        differenceOffset + element * differenceElementSize,
        elements * differenceElementSize
    };
}

/** An index set for marginal() and conditional(): a single index becomes a one-element set. */
template <typename Indices>
[[nodiscard]] auto
asIndexSet(Indices const & indices)
{
    if constexpr (std::is_integral_v<Indices>)
    {
        return std::array<Eigen::Index, 1>{indices};
    }
    else
    {
        return indices;
    }
}

}  // namespace internal

}  // namespace Emrys
