#pragma once

/** \file
 * Sequences and lists of the repetitions of a repeated block (decisions.md [repetition-selections], roadmap 17zu):
 * idx(), diffIdx(), block(), and diffBlock() given several repetitions where those of TagPath.h take one. The internal
 * namespace resolves a selection against a block's count, for these and for what reads a SelectedPath. Read Discrete.h
 * next.
 */

#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** Whether \a T is an Eigen arithmetic sequence, which <tt>Eigen::seq()</tt>, <tt>Eigen::seqN()</tt>, and
 * <tt>Eigen::placeholders::lastN()</tt> return.
 */
template <typename T>
inline constexpr bool isArithmeticSequence = false;

/** Whether the bounds and the increment of the sequence \a T are integers, so that it lists its
 * indices without the size of what it indexes; one that counts from the end, as lastN() does, has
 * symbols for bounds.
 */
template <typename T>
inline constexpr bool hasIntegerBounds = false;

/** Whether the repetitions \a T selects are consecutive whatever its values: a sequence whose
 * increment its type fixes at one.
 */
template <typename T>
inline constexpr bool isConsecutive = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename First, typename Size, typename Increment>
inline constexpr bool isArithmeticSequence<Eigen::ArithmeticSequence<First, Size, Increment>>{true};

template <typename First, typename Size, typename Increment>
inline constexpr bool hasIntegerBounds<Eigen::ArithmeticSequence<First, Size, Increment>> =
    std::is_convertible_v<First, Eigen::Index> && std::is_convertible_v<Size, Eigen::Index>
    && std::is_convertible_v<Increment, Eigen::Index>;

template <typename First, typename Size, typename Increment>
inline constexpr bool isConsecutive<Eigen::ArithmeticSequence<First, Size, Increment>> =
    int(Eigen::ArithmeticSequence<First, Size, Increment>::IncrAtCompileTime) == 1;
/// \endcond

/** Several repetitions of a repeated block: an Eigen arithmetic sequence, or a list of their
 * indices (RepetitionList). One repetition is an integer.
 */
template <typename T>
concept RepetitionSelection = isArithmeticSequence<std::remove_cvref_t<T>> || RepetitionList<std::remove_cvref_t<T>>;

/** One repetition of a repeated block, an integer, or several of them, a RepetitionSelection: what
 * an accessor of repetitions takes, such as marginal<Path>(repetitions).
 */
template <typename T>
concept RepetitionChoice = RepetitionIndex<T> || RepetitionSelection<T>;

/** The column of indices that the selection \a Selection resolves to in a repeated block of at most
 * \a MaxCount repetitions: of fixed size where the selection's type fixes its size, and otherwise
 * bounded by the list's own bound or by \a MaxCount, since a selection lists each repetition once.
 */
template <int MaxCount, typename Selection>
struct ResolvedRepetitionsOf
{
    /** The column, for a list. */
    using type = IndexVector<
        repetitionListSize<Selection>(),
        repetitionListMaxSize<Selection>() != Eigen::Dynamic ? repetitionListMaxSize<Selection>() : MaxCount>;
};

/** The column a sequence resolves to, sized as Eigen sizes a vector indexed by it. */
template <int MaxCount, typename First, typename Size, typename Increment>
struct ResolvedRepetitionsOf<MaxCount, Eigen::ArithmeticSequence<First, Size, Increment>>
{
    /** The column, for a sequence. */
    using type = IndexedSize<Eigen::ArithmeticSequence<First, Size, Increment>, Eigen::Dynamic, MaxCount>;
};

/** ResolvedRepetitionsOf, as a type. */
template <int MaxCount, typename Selection>
using ResolvedRepetitions = typename ResolvedRepetitionsOf<MaxCount, Selection>::type;

/** \returns the number of repetitions that \a selection, a list or a sequence with integer bounds, names */
template <typename Selection>
[[nodiscard]] Eigen::Index
selectionLength(Selection const & selection)
{
    if constexpr (isArithmeticSequence<Selection>)
    {
        return selection.size();
    }
    else
    {
        return repetitionListLength(selection);
    }
}

/** \returns the indices that \a selection, a list or a sequence with integer bounds, names, in order
 * and unchecked, in the column of a block of at most \a MaxCount repetitions
 */
template <int MaxCount, typename Selection>
[[nodiscard]] ResolvedRepetitions<MaxCount, Selection>
listRepetitions(Selection const & selection)
{
    auto repetitions = ResolvedRepetitions<MaxCount, Selection>{};
    repetitions.resize(selectionLength(selection));
    if constexpr (isArithmeticSequence<Selection>)
    {
        for (Eigen::Index i = 0; i < repetitions.size(); ++i)
        {
            repetitions[i] = selection[i];
        }
    }
    else
    {
        copyRepetitionListInto(selection, repetitions);
    }
    return repetitions;
}

/** \returns the indices of the repetitions that \a selection names in a repeated block of \a count
 * repetitions, at most \a MaxCount, in the order it names them. Each is one of the block's
 * repetitions, and none is named twice. A sequence that counts from the block's end is resolved
 * against \a count.
 */
template <int MaxCount, typename Selection>
[[nodiscard]] ResolvedRepetitions<MaxCount, Selection>
resolveRepetitions(Eigen::Index count, Selection const & selection)
{
    auto repetitions = ResolvedRepetitions<MaxCount, Selection>{};
    if constexpr (isArithmeticSequence<Selection> && !hasIntegerBounds<Selection>)
    {
        // Eigen resolves the symbols against the count, and asserts the range of what they give.
        repetitions = resolveIndices<Eigen::Dynamic, MaxCount>(count, selection);
    }
    else
    {
        EMRYS_FILTER_ASSERT(
            selectionLength(selection) >= 0 && selectionLength(selection) <= count
            && "a selection names more repetitions than the block has"
        );
        repetitions = listRepetitions<MaxCount>(selection);
    }
    EMRYS_FILTER_ASSERT(
        (repetitions.array() >= 0).all() && (repetitions.array() < count).all() && "repetition out of range"
    );
    EMRYS_FILTER_ASSERT(distinctIndices(repetitions) && "a selection names a repetition twice");
    return repetitions;
}

/** \returns the coordinates of the repetitions \a repetitions of a repeated block, in order: \a Size
 * coordinates of each, from the one at \a start in repetition 0, where a repetition has \a step
 * coordinates
 */
template <int Size, typename Repetitions>
[[nodiscard]] auto
repetitionCoordinates(Repetitions const & repetitions, Eigen::Index start, Eigen::Index step)
{
    constexpr int count = Repetitions::RowsAtCompileTime;
    constexpr int maxCount = Repetitions::MaxRowsAtCompileTime;
    auto coordinates = IndexVector<
        count == Eigen::Dynamic ? Eigen::Dynamic : count * Size,
        maxCount == Eigen::Dynamic ? Eigen::Dynamic : maxCount * Size>{};
    coordinates.resize(repetitions.size() * Size);
    for (Eigen::Index r = 0; r < repetitions.size(); ++r)
    {
        for (Eigen::Index c = 0; c < Size; ++c)
        {
            coordinates[r * Size + c] = start + repetitions[r] * step + c;
        }
    }
    return coordinates;
}

/** \returns the index of \a repetitions, consecutive repetitions of \a Step coordinates each in a
 * block that starts at \a offset: one \c seqN, of fixed size where their number is fixed
 */
template <int Step, typename Repetitions>
[[nodiscard]] auto
consecutiveRepetitionsIdx(Repetitions const & repetitions, Eigen::Index offset)
{
    Eigen::Index const first = repetitions.size() > 0 ? repetitions[0] : Eigen::Index{0};
    if constexpr (Repetitions::RowsAtCompileTime == Eigen::Dynamic)
    {
        return Eigen::seqN(offset + first * Step, repetitions.size() * Step);
    }
    else
    {
        return Eigen::seqN(offset + first * Step, Eigen::fix<Repetitions::RowsAtCompileTime * Step>);
    }
}

/** idx(space, selection), counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename Path, typename S, typename Selection>
[[nodiscard]] auto
selectionIdx(S const & space, Selection const & selection)
{
    return withRepeatedSplit<Path>([&]<typename Outer, typename RepeatedTag, typename... Inner>() {
        auto const [offset, tag] = locateRepeated<Difference, Outer>(space);
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        constexpr int step = SizedAs<Difference, RepeatedTag>::ValueDimensionStep;
        auto const repetitions = resolveRepetitions<RepeatedTag::MaxCount>(tag->count(), selection);
        if constexpr (isConsecutive<Selection> && inner.size == step)
        {
            return consecutiveRepetitionsIdx<step>(repetitions, offset);
        }
        else
        {
            return repetitionCoordinates<inner.size>(repetitions, offset + inner.offset, step);
        }
    });
}

/** \returns the coordinates in \a space of the repetitions that \a selection names of the repeated
 * block that \a Path runs through, or of the block the rest of the path names within each, as a
 * column, repetition after repetition: value coordinates, or difference coordinates if \a Difference
 */
template <bool Difference, typename Path, typename S, typename Selection>
[[nodiscard]] auto
selectionCoordinates(S const & space, Selection const & selection)
{
    return withRepeatedSplit<Path>([&]<typename Outer, typename RepeatedTag, typename... Inner>() {
        auto const [offset, tag] = locateRepeated<Difference, Outer>(space);
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        constexpr int step = SizedAs<Difference, RepeatedTag>::ValueDimensionStep;
        auto const repetitions = resolveRepetitions<RepeatedTag::MaxCount>(tag->count(), selection);
        return repetitionCoordinates<inner.size>(repetitions, offset + inner.offset, step);
    });
}

/** \returns the number of repetitions that \a selection names of the repeated block that \a Path
 * runs through in \a space
 */
template <typename Path, typename S, typename Selection>
[[nodiscard]] Eigen::Index
selectedCount(S const & space, Selection const & selection)
{
    return withRepeatedSplit<Path>([&]<typename Outer, typename RepeatedTag, typename... Inner>() {
        auto const * tag = locateRepeated<false, Outer>(space).second;
        return Eigen::Index{resolveRepetitions<RepeatedTag::MaxCount>(tag->count(), selection).size()};
    });
}

/** Whether the rows of the repetitions that \a Selection selects through the path \a Path are one run
 * whatever its values: whole repetitions of a sequence whose increment is one.
 */
template <typename Path, typename Selection>
inline constexpr bool selectsOneRun = isConsecutive<Selection> && IsRepeated<PathTag<Path>>;

/** idx<S, Path>(selection), counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename S, typename Path, typename Selection>
[[nodiscard]] auto
staticSelectionIdx(Selection const & selection)
{
    static_assert(
        !isArithmeticSequence<Selection> || hasIntegerBounds<Selection>,
        "a sequence that counts from the block's end needs the block's count; use idx(space, selection)"
    );
    return withRepeatedSplit<Path>([&]<typename Outer, typename RepeatedTag, typename... Inner>() {
        constexpr ResolvedIndex outer = ResolvePath<S, Outer, Difference>::value;
        static_assert(
            outer.staticOffset,
            "this block's offset depends on the size of a resizable block; use idx(space, selection)"
        );
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        constexpr int step = SizedAs<Difference, RepeatedTag>::ValueDimensionStep;
        auto const repetitions = listRepetitions<RepeatedTag::MaxCount>(selection);
        if constexpr (isConsecutive<Selection> && inner.size == step)
        {
            return consecutiveRepetitionsIdx<step>(repetitions, Eigen::Index{outer.offset});
        }
        else
        {
            return repetitionCoordinates<inner.size>(repetitions, outer.offset + inner.offset, step);
        }
    });
}

/** rowsByPath() for the repetitions \a selection names of the repeated block that \a Path runs
 * through, or of the block the rest of the path names within each. See block(space, x, selection).
 */
template <bool Difference, typename Path, typename S, typename X, typename Selection>
[[nodiscard]] auto
selectionRowsByPath(S const & space, X && x, Selection const & selection)
{
    checkBlockRows<rowsAtCompileTime<Difference, S>, X>();
    EMRYS_FILTER_ASSERT(x.rows() == rowsOf<Difference>(space) && "the rows do not match the space");
    return withRepeatedSplit<Path>([&]<typename Outer, typename RepeatedTag, typename... Inner>() {
        auto const [offset, tag] = locateRepeated<Difference, Outer>(space);
        constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
        constexpr int step = SizedAs<Difference, RepeatedTag>::ValueDimensionStep;
        auto const repetitions = resolveRepetitions<RepeatedTag::MaxCount>(tag->count(), selection);
        using Repetitions = std::remove_cvref_t<decltype(repetitions)>;
        if constexpr (isConsecutive<Selection> && inner.size == step)
        {
            constexpr int rows = Repetitions::RowsAtCompileTime == Eigen::Dynamic
                                   ? Eigen::Dynamic
                                   : Repetitions::RowsAtCompileTime * step;
            Eigen::Index const first = repetitions.size() > 0 ? repetitions[0] : Eigen::Index{0};
            return middleRowsOf<rows>(x, offset + first * step, repetitions.size() * step);
        }
        else
        {
            return x(
                repetitionCoordinates<inner.size>(repetitions, offset + inner.offset, step),
                Eigen::placeholders::all
            );
        }
    });
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the index of the repetitions that \a selection names of a repeated block within
 * \a space, as idx<Path>(space, element) gives the index of one: \a selection is an Eigen sequence,
 * such as <tt>Eigen::seqN(first, n)</tt> or <tt>Eigen::placeholders::lastN(n)</tt>, or a list of
 * repetition indices, a <tt>std::array</tt>, a <tt>std::vector</tt>, a <tt>std::span</tt>, or an
 * Eigen vector of integers, read in the order given. The result is one \c seqN for whole
 * repetitions of a sequence whose increment is one, and otherwise a column of the coordinates'
 * indices, repetition after repetition. Each repetition named must be one the block has, and none
 * is named twice.
 *
 * \code
 * x(idx<Landmarks>(space, Eigen::seqN(2, 3)));                 // landmarks 2, 3, and 4
 * x(idx<TagPath<Fleet, Heading>>(space, std::array{3, 0}));    // the headings of vehicles 3 and 0
 * \endcode
 */
template <typename Path, internal::RepetitionSelection Selection>
[[nodiscard]] auto
idx(Space auto const & space, Selection const & selection)
{
    return internal::selectionIdx<false, Path>(space, selection);
}

/** \ingroup Spaces
 * idx<Path>(space, selection) of the differences of the repetitions: their rows of a covariance.
 */
template <typename Path, internal::RepetitionSelection Selection>
[[nodiscard]] auto
diffIdx(Space auto const & space, Selection const & selection)
{
    return internal::selectionIdx<true, Path>(space, selection);
}

/** \ingroup Spaces
 * idx<Path>(space, selection) for the space type \a S, without a space object. Only available
 * when no resizable block precedes the repeated block, and for a selection that does not count
 * from the block's end; the repetitions are not checked against the block's count, which is
 * unknown.
 */
template <Space S, typename Path, internal::RepetitionSelection Selection>
[[nodiscard]] auto
idx(Selection const & selection)
{
    return internal::staticSelectionIdx<false, S, Path>(selection);
}

/** \ingroup Spaces
 * idx<S, Path>(selection) of the differences of the repetitions.
 */
template <Space S, typename Path, internal::RepetitionSelection Selection>
[[nodiscard]] auto
diffIdx(Selection const & selection)
{
    return internal::staticSelectionIdx<true, S, Path>(selection);
}

/** \ingroup Spaces
 * \returns block<Path>(space, x) for the repetitions that \a selection names of the Repeated block
 * that \a Path runs through, as block(space, x, element) gives one: \a selection is an Eigen
 * sequence or a list of repetition indices, as idx(space, selection) takes. Whole repetitions of a
 * sequence whose increment is one are Eigen's block of their rows; any other selection is Eigen's
 * indexed view of them, repetition after repetition. Both are writable where \a x is.
 */
template <typename Path, MatrixExpression X, internal::RepetitionSelection Selection>
[[nodiscard]] auto
block(Space auto const & space, X && x, Selection const & selection)
{
    return internal::selectionRowsByPath<false, Path>(space, std::forward<X>(x), selection);
}

/** \ingroup Spaces
 * \returns diffBlock<Path>(space, d) for the repetitions that \a selection names, as
 * block(space, x, selection) does for values.
 */
template <typename Path, MatrixExpression D, internal::RepetitionSelection Selection>
[[nodiscard]] auto
diffBlock(Space auto const & space, D && d, Selection const & selection)
{
    return internal::selectionRowsByPath<true, Path>(space, std::forward<D>(d), selection);
}

namespace internal
{

/** \returns the space of the repetitions that \a choice names of the repeated block that \a Path
 * runs through in \a space, or of the block the rest of the path names within them: for one
 * repetition that block itself, and for several a repeated block of their count
 */
template <typename Path, typename S, typename Choice>
[[nodiscard]] auto
pathChoiceSubspace(S const & space, [[maybe_unused]] Choice const & choice)
{
    if constexpr (RepetitionIndex<Choice>)
    {
        return pathElementSubspace<Path>(space);
    }
    else
    {
        return pathRepetitionsSubspace<Path>(space, selectedCount<Path>(space, choice));
    }
}

/** Shared implementation of marginal<Path>(repetitions) for distributions that own their
 * parameters, where \a Path runs through a repeated block and \a repetitions is one repetition's
 * index, or a sequence or a list of them.
 *
 * \returns \a make(target, targetSpace): the Coordinates of the repetitions (or of the rest of the
 * path within each), and their space (pathChoiceSubspace())
 */
template <typename Path, typename Space, typename Choice, typename Make>
[[nodiscard]] auto
marginalOverElement(Space const & space, Choice const & repetitions, Make make)
{
    return make(
        resolveCoordinates(space, idx<Path>(space, repetitions), diffIdx<Path>(space, repetitions)),
        pathChoiceSubspace<Path>(space, repetitions)
    );
}

}  // namespace internal

}  // namespace Emrys
