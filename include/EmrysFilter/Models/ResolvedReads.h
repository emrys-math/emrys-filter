#pragma once

/** \file
 * How a model's reads() meet a state and a call, all internal (decisions.md [model-policies], [tag-addressing]):
 * ReadBlocks, the blocks of the state a model reads; and ResolvedReads, a model's reads() resolved against a state
 * space and the call's arguments, which sorts the items into blocks of the state and inputs of the call, builds the
 * model's input space from both, and gathers the function's input vector once per evaluation. MeasurementReads.h and
 * ProcessModel.h build their resolvers on it. Read MeasurementReads.h next.
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <unsupported/Eigen/AutoDiff>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Selection.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The blocks a model reads from a state in \a Space: those of \a Items, a tuple of the items of its
 * reads() that name blocks of the state (ReadItem).
 */
template <typename Space, typename Items>
class ReadBlocks;

/** Whether the rows of a state that the item \a Item reads are one run whatever its selection's
 * values: a block, one repetition, or whole repetitions of a sequence whose increment is one. The
 * rows of any other selection are read through the blocks' indices.
 */
template <typename Item>
inline constexpr bool readsOneRun = !Item::Selected || RepetitionIndex<typename Item::SelectionType>
                                 || selectsOneRun<typename Item::PathType, typename Item::SelectionType>;

/** \returns the space of what the item \a Item, of a reads() or a writes(), names in \a space with
 * \a selection: the block its path names, the block that is repeated for one repetition, or a
 * repeated block of the count selected
 */
template <typename Item, typename Space>
[[nodiscard]] ReadSpace<Space, Item>
itemSpace(Space const & space, [[maybe_unused]] typename Item::SelectionType const & selection)
{
    using Path = typename Item::PathType;
    if constexpr (!Item::Selected)
    {
        return pathBlock<Path>(space);
    }
    else if constexpr (RepetitionIndex<typename Item::SelectionType>)
    {
        return pathElementSubspace<Path>(space);
    }
    else
    {
        return pathRepetitionsSubspace<Path>(space, selectedCount<Path>(space, selection));
    }
}

/** \returns the rows of \a space that the item \a Item names with \a selection where they are one
 * run (readsOneRun), in values, or in differences if \a Difference
 */
template <typename Item, bool Difference, typename Space>
[[nodiscard]] BlockRows
itemRun(Space const & space, [[maybe_unused]] typename Item::SelectionType const & selection)
{
    static_assert(readsOneRun<Item>);
    using Path = typename Item::PathType;
    if constexpr (!Item::Selected)
    {
        if constexpr (Difference)
        {
            return differenceBlockRows<Path>(space);
        }
        else
        {
            return blockRows<Path>(space);
        }
    }
    else if constexpr (RepetitionIndex<typename Item::SelectionType>)
    {
        return repetitionBlockRows<Difference, Path>(space, selection);
    }
    else
    {
        auto const coordinates = selectionCoordinates<Difference, Path>(space, selection);
        return BlockRows{coordinates.size() > 0 ? coordinates(0) : Eigen::Index{0}, coordinates.size()};
    }
}

/** Writes the rows of \a space that the item \a Item names with \a selection into \a rows from row
 * \a at on, in values, or in differences if \a Difference.
 *
 * \returns their number, after their first, which is where they start when they are one run
 */
template <typename Item, bool Difference, typename Space, typename Rows>
[[nodiscard]] BlockRows
writeItemRows(
    Space const & space,
    [[maybe_unused]] typename Item::SelectionType const & selection,
    Rows & rows,
    Eigen::Index at
)
{
    if constexpr (Item::Selected && !RepetitionIndex<typename Item::SelectionType>)
    {
        auto const coordinates = selectionCoordinates<Difference, typename Item::PathType>(space, selection);
        rows.segment(at, coordinates.size()) = coordinates;
        return BlockRows{coordinates.size() > 0 ? coordinates(0) : Eigen::Index{0}, coordinates.size()};
    }
    else
    {
        BlockRows const block = itemRun<Item, Difference>(space, selection);
        for (Eigen::Index j = 0; j < block.size; ++j)
        {
            rows(at + j) = block.start + j;
        }
        return block;
    }
}

/** The blocks that the items \a Items read, resolved in \a Space: their rows in the state, in values
 * and in differences, and the joint space they form, in the order given.
 */
template <typename Space, typename... Items>
class ReadBlocks<Space, std::tuple<Items...>>
{
    static_assert(sizeof...(Items) > 0, "a model reads at least one block of the state");

    /** The first item. */
    using First = std::tuple_element_t<0, std::tuple<Items...>>;

public:

    static constexpr std::size_t Count = sizeof...(Items);  ///< the number of blocks read

    /** The space of the blocks read: the one block's, or the Composite of them. */
    using JointSpace = std::conditional_t<Count == 1, ReadSpace<Space, First>, Composite<ReadSpace<Space, Items>...>>;

    static constexpr int Size = valueDimensionAtCompileTime<JointSpace>;        ///< the joint value dimension
    static constexpr int MaxSize = maxValueDimensionAtCompileTime<JointSpace>;  ///< a bound on it
    static constexpr int DifferenceSize =
        valueDimensionAtCompileTime<DifferenceSpace<JointSpace>>;  ///< the joint difference dimension
    static constexpr int DifferenceMaxSize =
        maxValueDimensionAtCompileTime<DifferenceSpace<JointSpace>>;           ///< a bound on it
    using Indices = IndexVector<Size, MaxSize>;                                ///< the value coordinates read
    using DifferenceIndices = IndexVector<DifferenceSize, DifferenceMaxSize>;  ///< the difference coordinates read
    using Selection = Coordinates<Indices, DifferenceIndices>;                 ///< both, for a GaussianMarginal

    /** Whether block \a K's rows in a state are one run (readsOneRun), so that rows() and
     * differenceRows() give them.
     */
    template <std::size_t K>
    static constexpr bool OneRun = readsOneRun<std::tuple_element_t<K, std::tuple<Items...>>>;

    /** Resolves the blocks' rows in \a space and the joint space they form, each item's with its
     * selection of \a selections, which is NoSelection for an item that reads its whole block.
     */
    explicit ReadBlocks(Space const & space, typename Items::SelectionType const &... selections)
        : m_space{makeSpace(space, selections...)}
    {
        m_indices.resize(valueDimension(m_space));
        m_differenceIndices.resize(differenceDimension(m_space));
        Eigen::Index offset = 0;
        Eigen::Index differenceOffset = 0;
        std::size_t k = 0;
        (
            [&] {
                m_rows[k] = writeItemRows<Items, false>(space, selections, m_indices, offset);
                m_differenceRows[k] =
                    writeItemRows<Items, true>(space, selections, m_differenceIndices, differenceOffset);
                m_jointStarts[k] = offset;
                m_jointDifferenceStarts[k] = differenceOffset;
                offset += m_rows[k].size;
                differenceOffset += m_differenceRows[k].size;
                ++k;
            }(),
            ...
        );
        EMRYS_FILTER_ASSERT(offset == m_indices.size() && differenceOffset == m_differenceIndices.size());
        EMRYS_FILTER_ASSERT(distinctIndices(m_indices) && "a model must not read a coordinate twice");
    }

    /** \returns the joint space of the blocks read, with their parameters */
    [[nodiscard]] JointSpace const &
    space() const
    {
        return m_space;
    }

    /** \returns the space of block \a K, with its parameters */
    template <std::size_t K>
    [[nodiscard]] auto const &
    blockSpace() const
    {
        if constexpr (Count == 1)
        {
            return m_space;
        }
        else
        {
            return m_space.template block<K>();
        }
    }

    /** \returns the state's value coordinates read, block after block */
    [[nodiscard]] Indices const &
    indices() const
    {
        return m_indices;
    }

    /** \returns the state's difference coordinates read, block after block: rows of its covariance */
    [[nodiscard]] DifferenceIndices const &
    differenceIndices() const
    {
        return m_differenceIndices;
    }

    /** \returns the coordinates read, for a GaussianMarginal */
    [[nodiscard]] Selection
    selection() const
    {
        return {m_indices, m_differenceIndices};
    }

    /** \returns block \a k's rows in a state's values, where they are one run (OneRun); of any other
     * block, only their number, its rows being indices() from jointStart()
     */
    [[nodiscard]] BlockRows const &
    rows(std::size_t k) const
    {
        return m_rows[k];
    }

    /** \returns block \a k's rows in a state's differences, as rows() gives those of its values */
    [[nodiscard]] BlockRows const &
    differenceRows(std::size_t k) const
    {
        return m_differenceRows[k];
    }

    /** \returns block \a k's first row in the joint vector of values */
    [[nodiscard]] Eigen::Index
    jointStart(std::size_t k) const
    {
        return m_jointStarts[k];
    }

    /** \returns block \a k's first row in the joint vector of differences */
    [[nodiscard]] Eigen::Index
    jointDifferenceStart(std::size_t k) const
    {
        return m_jointDifferenceStarts[k];
    }

private:

    /** \returns the blocks' spaces from \a space, as one block or their Composite */
    [[nodiscard]] static JointSpace
    makeSpace(Space const & space, typename Items::SelectionType const &... selections)
    {
        if constexpr (Count == 1)
        {
            return itemSpace<First>(space, selections...);
        }
        else
        {
            return JointSpace{itemSpace<Items>(space, selections)...};
        }
    }

    JointSpace m_space;                                         ///< the joint space
    std::array<BlockRows, Count> m_rows{};                      ///< each block's rows in a state's values
    std::array<BlockRows, Count> m_differenceRows{};            ///< each block's rows in a state's differences
    std::array<Eigen::Index, Count> m_jointStarts{};            ///< each block's first row in the joint values
    std::array<Eigen::Index, Count> m_jointDifferenceStarts{};  ///< each block's first row in the joint differences
    Indices m_indices;                                          ///< the value coordinates read
    DifferenceIndices m_differenceIndices;                      ///< the difference coordinates read
};

/** The blocks a whole-state model reads (decisions.md [whole-state-models]): the whole state in
 * \a Space as one block, its rows every row, with the same interface as the blocks named by paths.
 */
template <typename Space>
class ReadBlocks<Space, std::tuple<ReadItem<WholeSpace, Space>>>
{
public:

    static constexpr std::size_t Count = 1;  ///< one block, the whole state

    /** The whole state is one run of rows. */
    template <std::size_t K>
    static constexpr bool OneRun = true;

    /** The space of the block read: the state's. */
    using JointSpace = Space;

    static constexpr int Size = valueDimensionAtCompileTime<Space>;        ///< the value dimension
    static constexpr int MaxSize = maxValueDimensionAtCompileTime<Space>;  ///< a bound on it
    static constexpr int DifferenceSize =
        valueDimensionAtCompileTime<DifferenceSpace<Space>>;  ///< the difference dimension
    static constexpr int DifferenceMaxSize = maxValueDimensionAtCompileTime<DifferenceSpace<Space>>;  ///< a bound on it
    using Indices = IndexVector<Size, MaxSize>;                                ///< the value coordinates read
    using DifferenceIndices = IndexVector<DifferenceSize, DifferenceMaxSize>;  ///< the difference coordinates read
    using Selection = Coordinates<Indices, DifferenceIndices>;                 ///< both, for a GaussianMarginal

    /** The whole of \a space. */
    explicit ReadBlocks(Space const & space)
        : m_rows{0, valueDimension(space)}
        , m_differenceRows{0, differenceDimension(space)}
        , m_indices{Indices::LinSpaced(m_rows.size, 0, m_rows.size - 1)}
        , m_differenceIndices{DifferenceIndices::LinSpaced(m_differenceRows.size, 0, m_differenceRows.size - 1)}
        , m_space{space}
    {}

    /** \returns the state's space */
    [[nodiscard]] JointSpace const &
    space() const
    {
        return m_space;
    }

    /** \returns the state's space, the one block's */
    template <std::size_t K>
    [[nodiscard]] JointSpace const &
    blockSpace() const
    {
        return m_space;
    }

    /** \returns every value coordinate */
    [[nodiscard]] Indices const &
    indices() const
    {
        return m_indices;
    }

    /** \returns every difference coordinate */
    [[nodiscard]] DifferenceIndices const &
    differenceIndices() const
    {
        return m_differenceIndices;
    }

    /** \returns the coordinates read, for a GaussianMarginal */
    [[nodiscard]] Selection
    selection() const
    {
        return {m_indices, m_differenceIndices};
    }

    /** \returns the rows of the state's values */
    [[nodiscard]] BlockRows const &
    rows(std::size_t) const
    {
        return m_rows;
    }

    /** \returns the rows of the state's differences */
    [[nodiscard]] BlockRows const &
    differenceRows(std::size_t) const
    {
        return m_differenceRows;
    }

    /** \returns 0, the block's first row in the joint vector of values */
    [[nodiscard]] Eigen::Index
    jointStart(std::size_t) const
    {
        return 0;
    }

    /** \returns 0, the block's first row in the joint vector of differences */
    [[nodiscard]] Eigen::Index
    jointDifferenceStart(std::size_t) const
    {
        return 0;
    }

private:

    BlockRows m_rows;                       ///< every row of a state's values
    BlockRows m_differenceRows;             ///< every row of a state's differences
    Indices m_indices;                      ///< the value coordinates
    DifferenceIndices m_differenceIndices;  ///< the difference coordinates
    Space m_space;                          ///< the state's space
};

/** What a whole-state model reads in place of reads() (decisions.md [whole-state-models]): the
 * whole state in \a Space as one item, read under the state's own space, which is then the model's
 * input space.
 */
template <typename Space>
struct ReadsWhole
{
    static constexpr std::size_t Count = 1;  ///< one item

    /** The input space: the state's. */
    using InputSpace = Space;

    /** The one item: the whole state. */
    template <std::size_t I>
    using Item = ReadItem<WholeSpace, Space>;
};

/** Whether \a Model reads the whole state: it was given no reads() (decisions.md [whole-state-models]). */
template <typename Model>
inline constexpr bool readsWholeState = !Model::template has<PolicyKind::reads>;

/** Refuses the whole-state model \a Model where models are placed or composed, by on(), onEach(),
 * and compose(), which work through the paths of reads() and writes() (decisions.md
 * [whole-state-models]). The assertion is a class's, which fires where wholeStateRefused is read,
 * before the caller's own assertions: a consteval function's waits for its body, and MSVC 19.51
 * reported compose()'s first.
 */
template <typename Model>
struct WholeStateRefusal
{
    static_assert(
        !readsWholeState<Model>,
        "a model without reads() and writes() reads and writes the whole state, so it is not placed by on() or "
        "onEach() or composed by compose(): give it reads() and writes() to name its blocks"
    );

    static constexpr bool value = true;  ///< read to make the assertion
};

/** True once WholeStateRefusal has refused a whole-state \a Model: assert it where models are placed
 * or composed.
 */
template <typename Model>
inline constexpr bool wholeStateRefused = WholeStateRefusal<Model>::value;

/** \returns whether the parts of \a model that read repetitions of a block of \a space under one name
 * select the same ones (decisions.md [composed-selections]): a composition holds the selection of the
 * first part that reads a name, and every part's function reads its input by that name. A
 * composition's function and a placed model's answer for their parts, with \c selectionsAgree(space);
 * any other model has no parts, and one whose reads() selects nothing has nothing to compare.
 */
template <typename Model, typename Space>
[[nodiscard]] bool
selectionsAgree([[maybe_unused]] Model const & model, [[maybe_unused]] Space const & space)
{
    if constexpr (readsWholeState<Model>)
    {
        return true;
    }
    else if constexpr (!Model::template Policy<PolicyKind::reads>::Selected)
    {
        return true;
    }
    else if constexpr (requires { model.function().selectionsAgree(space); })
    {
        return model.function().selectionsAgree(space);
    }
    else
    {
        return true;
    }
}

/** What \a Model reads in the state space \a Space: its reads(), or ReadsWhole for a whole-state model. */
template <typename Model, typename Space>
using ReadsIn =
    std::conditional_t<readsWholeState<Model>, ReadsWhole<Space>, typename Model::template Policy<PolicyKind::reads>>;

/** A TagPaths of the paths in the tuple type \a Tuple. */
template <typename Tuple>
struct TagPathsOfTuple;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Paths>
struct TagPathsOfTuple<std::tuple<Paths...>>
{
    using type = TagPaths<Paths...>;
};

/// \endcond

/** Whether the affineIn() policy \a Policy names \a Name; false for NoPolicy. */
template <typename Policy, typename Name>
inline constexpr bool affineNames = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Names, typename Name>
inline constexpr bool affineNames<AffineIn<Names...>, Name> = (std::is_same_v<Names, Name> || ...);

/// \endcond

/** \returns \a Name's instance for the block \a block it stands for in a model's input: the block
 * itself where \a Name is its type, the empty instance where \a Name is a fixed-size space, and
 * otherwise a default instance fitted to the block's value dimension
 */
template <typename Name, typename Block>
[[nodiscard]] Name
nameOf(Block const & block)
{
    if constexpr (std::is_same_v<Name, Block>)
    {
        return block;
    }
    else if constexpr (FixedSizeSpace<Name>)
    {
        return Name{};
    }
    else
    {
        auto name = Name{};
        fitValueDimension(name, valueDimension(block));
        return name;
    }
}

/** The scalar type \a T as a vector stores it: itself, or for an automatic-differentiation scalar
 * whose derivatives are an expression, the one whose derivatives are a plain vector.
 */
template <typename T>
struct PlainScalarOf
{
    using type = T;  ///< the scalar itself
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Derivatives>
struct PlainScalarOf<Eigen::AutoDiffScalar<Derivatives>>
{
    using type = Eigen::AutoDiffScalar<typename std::remove_cvref_t<Derivatives>::PlainObject>;
};

/// \endcond

/** \returns \a returned, what a model's function returned, with a scalar wrapped as a vector of one
 * coordinate, so that a function of one output may return its value alone, in its input's scalar
 * type: an automatic-differentiation scalar whose derivatives are still an expression refers to
 * temporaries of the function, which are gone by now, so it is refused
 */
template <typename R>
[[nodiscard]] auto
vectorReturned(R && returned)
{
    using Plain = std::remove_cvref_t<R>;
    if constexpr (!MatrixExpression<Plain> && !isWithJacobian<Plain>)
    {
        static_assert(
            std::is_same_v<Plain, typename PlainScalarOf<Plain>::type>,
            "a function that returns a scalar returns it as its input's scalar type, T{...}: an expression of "
            "automatic-differentiation scalars refers to temporaries of the function; or return a vector of one "
            "coordinate"
        );
        return Eigen::Matrix<Plain, 1, 1>{std::forward<R>(returned)};
    }
    else
    {
        return std::forward<R>(returned);
    }
}

/** \returns whether every item of \a Model's reads() is named in its affineIn(), or for a whole-state
 * model whether its function says it is affine, as a matrix model's does: the function is then
 * affine in its whole input, so its Jacobian is the same at every point
 */
template <typename Model>
[[nodiscard]] consteval bool
affineInEverything()
{
    if constexpr (readsWholeState<Model>)
    {
        return requires { requires Model::FunctionType::IsAffine; };
    }
    else
    {
        using Affine = typename Model::template Policy<PolicyKind::affineIn>;
        return []<typename... Items>(Reads<Items...> const *) {
            return (affineNames<Affine, typename Items::NameType> && ... && true);
        }(static_cast<typename Model::template Policy<PolicyKind::reads> const *>(nullptr));
    }
}

/** A model \a Model's reads() resolved against the state space \a Space and a call's arguments
 * \a Arguments, a tuple of the arguments as the call holds them (decisions.md [model-policies]): which
 * items name blocks of the state and which the call's arguments, the model's input space built from
 * both, the layout of the function's input vector, and the gathering of that vector once per
 * evaluation from a joint vector of the blocks read or from the whole state, and from the arguments.
 * The measurement and process resolvers build on it. Valid while the arguments and the space are.
 */
template <typename Model, typename Arguments, typename Space>
class ResolvedReads
{
public:

    using ModelType = Model;  ///< the model
    using SpaceType = Space;  ///< the state's space

    /// whether the model reads the whole state, with no reads() (decisions.md [whole-state-models])
    static constexpr bool Whole = readsWholeState<Model>;

    /** The reads() policy, or ReadsWhole for a whole-state model. */
    using ReadsPolicy = ReadsIn<Model, Space>;

    static constexpr std::size_t ItemCount = ReadsPolicy::Count;  ///< the items of reads()

    /** Item \a I of reads(). */
    template <std::size_t I>
    using Item = typename ReadsPolicy::template Item<I>;

    /** The input space: the Composite of the items' names, in order. */
    using In = typename ReadsPolicy::InputSpace;

    /** The call's arguments, as the call holds them. */
    using ArgumentsType = Arguments;

    static constexpr std::size_t ArgumentCount = std::tuple_size_v<Arguments>;  ///< how many

    /** Argument \a J. */
    template <std::size_t J>
    using Argument = std::tuple_element_t<J, Arguments>;

protected:

    /** \returns whether item \a I names a block of the state: its first tag is in the space */
    template <std::size_t I>
    [[nodiscard]] static consteval bool
    itemInState()
    {
        if constexpr (Whole)
        {
            return true;
        }
        else
        {
            return []<typename First, typename... Rest>(TagPath<First, Rest...> *) {
                return countTag<First, Space>() > 0;
            }(static_cast<typename Item<I>::PathType *>(nullptr));
        }
    }

    /** \returns the argument that item \a I, which the state lacks, names by its tag, or
     * ArgumentCount if none does, after checking the item is one tag
     */
    template <std::size_t I>
    [[nodiscard]] static consteval std::size_t
    argumentOfItem()
    {
        if constexpr (itemInState<I>())
        {
            return ArgumentCount;
        }
        else
        {
            static_assert(
                Item<I>::SingleTag,
                "an item of reads() that names no block of the state names an input of the call by one tag, so a "
                "path of several tags must be in the state"
            );
            static_assert(
                !Item<I>::Selected,
                "a selected item of reads() names a block of the state, since an item the state lacks is an input "
                "of the call, which has no repetitions to select"
            );
            using Tag = PathTag<typename Item<I>::PathType>;
            constexpr auto matches = []<std::size_t... J>(std::index_sequence<J...>) {
                return std::array<bool, ArgumentCount + 1>{std::is_same_v<InputTag<Argument<J>>, Tag>..., false};
            }(std::make_index_sequence<ArgumentCount>{});
            for (std::size_t j = 0; j < ArgumentCount; ++j)
            {
                if (matches[j])
                {
                    return j;
                }
            }
            return ArgumentCount;
        }
    }

    // These are consteval functions of an index rather than static constexpr arrays: MSVC reads a
    // class template's constexpr array members in a class-scope static_assert before it has
    // initialized them ("read of an uninitialized symbol"), and a function has nothing to read.

    /** \returns whether item \a i names a block of the state */
    [[nodiscard]] static consteval bool
    inState(std::size_t i)
    {
        return []<std::size_t... I>(std::size_t item, std::index_sequence<I...>) {
            return ((I == item && itemInState<I>()) || ... || false);
        }(i, std::make_index_sequence<ItemCount>{});
    }

    /** \returns the argument item \a i names, or ArgumentCount */
    [[nodiscard]] static consteval std::size_t
    argumentOf(std::size_t i)
    {
        return []<std::size_t... I>(std::size_t item, std::index_sequence<I...>) {
            std::size_t found = ArgumentCount;
            ((I == item ? (found = argumentOfItem<I>(), 0) : 0), ...);
            return found;
        }(i, std::make_index_sequence<ItemCount>{});
    }

    static_assert(
        []<std::size_t... I>(std::index_sequence<I...>) {
            return ((itemInState<I>() || argumentOfItem<I>() < ArgumentCount) && ...);
        }(std::make_index_sequence<ItemCount>{}),
        "the model reads an input that the state lacks, which the call must give as an uncertain input over the "
        "tag's space or as sure<Tag>(value)"
    );

    /** \returns whether argument \a J, if it is an input, is read by some item */
    template <std::size_t J>
    [[nodiscard]] static consteval bool
    argumentRead()
    {
        if constexpr (std::is_void_v<InputTag<Argument<J>>>)
        {
            return true;
        }
        else
        {
            return []<std::size_t... I>(std::index_sequence<I...>) {
                return ((argumentOfItem<I>() == J) || ... || false);
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    static_assert(
        []<std::size_t... J>(std::index_sequence<J...>) {
            return (argumentRead<J>() && ... && true);
        }(std::make_index_sequence<ArgumentCount>{}),
        "the call was given an input the model does not read"
    );

    /** \returns whether argument \a j is an uncertain input; false for ArgumentCount, no argument */
    [[nodiscard]] static consteval bool
    uncertainArgument(std::size_t j)
    {
        return []<std::size_t... J>(std::size_t argument, std::index_sequence<J...>) {
            return ((J == argument && argumentKind<Argument<J>> == ArgumentKind::uncertain) || ... || false);
        }(j, std::make_index_sequence<ArgumentCount>{});
    }

    /** \returns whether item \a i names an uncertain input */
    [[nodiscard]] static consteval bool
    argumentIsUncertain(std::size_t i)
    {
        return !inState(i) && uncertainArgument(argumentOf(i));
    }

    /** \returns the number of items that name blocks of the state (\a state), or uncertain inputs */
    [[nodiscard]] static consteval std::size_t
    countItems(bool state)
    {
        std::size_t count = 0;
        for (std::size_t i = 0; i < ItemCount; ++i)
        {
            if (state ? inState(i) : argumentIsUncertain(i))
            {
                ++count;
            }
        }
        return count;
    }

public:

    static constexpr std::size_t StateCount = countItems(true);   ///< the items that name blocks of the state
    static constexpr std::size_t InputCount = countItems(false);  ///< the items that name uncertain inputs

protected:

    /** \returns the \a n th item (from 0) that names a block of the state (\a state), or an uncertain
     * input, or ItemCount when there are fewer
     */
    [[nodiscard]] static consteval std::size_t
    itemWhere(bool state, std::size_t n)
    {
        std::size_t seen = 0;
        for (std::size_t i = 0; i < ItemCount; ++i)
        {
            if (state ? inState(i) : argumentIsUncertain(i))
            {
                if (seen == n)
                {
                    return i;
                }
                ++seen;
            }
        }
        return ItemCount;
    }

    /** \returns the \a k th item that names a block of the state */
    [[nodiscard]] static consteval std::size_t
    stateItems(std::size_t k)
    {
        return itemWhere(true, k);
    }

    /** \returns the \a p th item that names an uncertain input */
    [[nodiscard]] static consteval std::size_t
    inputItems(std::size_t p)
    {
        return itemWhere(false, p);
    }

    /** \returns the position of item \a i among the state items */
    [[nodiscard]] static consteval std::size_t
    stateIndexOf(std::size_t i)
    {
        for (std::size_t k = 0; k < StateCount; ++k)
        {
            if (stateItems(k) == i)
            {
                return k;
            }
        }
        return StateCount;
    }

    /** \returns the position of item \a i among the uncertain input items */
    [[nodiscard]] static consteval std::size_t
    inputIndexOf(std::size_t i)
    {
        for (std::size_t p = 0; p < InputCount; ++p)
        {
            if (inputItems(p) == i)
            {
                return p;
            }
        }
        return InputCount;
    }

public:

    /** The paths of the blocks of the state read, as a TagPaths, in the order of reads(). */
    using StatePaths = typename TagPathsOfTuple<decltype([]<std::size_t... K>(std::index_sequence<K...>) {
        return std::tuple<typename Item<stateItems(K)>::PathType...>{};
    }(std::make_index_sequence<StateCount>{}))>::type;

    /** The items that name blocks of the state, as a tuple, in the order of reads(). */
    using StateItems = decltype([]<std::size_t... K>(std::index_sequence<K...>) {
        return std::tuple<Item<stateItems(K)>...>{};
    }(std::make_index_sequence<StateCount>{}));

    /** The blocks of the state the model reads, resolved. */
    using Reads = ReadBlocks<Space, StateItems>;

    /** A vector in the input space, in \a T. */
    template <typename T>
    using InVector = typename Moments<In, T>::VectorType;

    /** Resolves the reads() of \a model, which holds each item's selection, against \a space and the
     * call's \a arguments, which both outlive this.
     */
    ResolvedReads(Arguments const & arguments, Space const & space, Model const & model)
        : m_arguments{arguments}
        , m_reads{makeReads(space, model)}
        , m_in{makeIn(arguments)}
        , m_stateDifferenceSize{differenceDimension(space)}
    {
        if constexpr (Whole)
        {
            m_valueRows[0] = BlockRows{0, valueDimension(m_in)};
            m_differenceRows[0] = BlockRows{0, differenceDimension(m_in)};
        }
        else
        {
            static_assert(checkReadNames<ReadsPolicy, Space>());
            EMRYS_FILTER_ASSERT(
                selectionsAgree(model, space)
                && "parts of a composition that read repetitions of a block under one name select the same ones"
            );
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((m_valueRows[I] = blockRows<typename Item<I>::NameType>(m_in),
                  m_differenceRows[I] = differenceBlockRows<typename Item<I>::NameType>(m_in)),
                 ...);
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    /** \returns the call's arguments */
    [[nodiscard]] Arguments const &
    arguments() const
    {
        return m_arguments;
    }

    /** \returns the blocks of the state read, resolved */
    [[nodiscard]] Reads const &
    reads() const
    {
        return m_reads;
    }

    /** \returns the model's input space instance */
    [[nodiscard]] In const &
    in() const
    {
        return m_in;
    }

    /** \returns the state's difference dimension */
    [[nodiscard]] Eigen::Index
    stateDifferenceSize() const
    {
        return m_stateDifferenceSize;
    }

    /** \returns item \a i's rows in the input's values */
    [[nodiscard]] BlockRows const &
    valueRows(std::size_t i) const
    {
        return m_valueRows[i];
    }

    /** \returns item \a i's rows in the input's differences */
    [[nodiscard]] BlockRows const &
    differenceRows(std::size_t i) const
    {
        return m_differenceRows[i];
    }

    /** \returns the uncertain inputs of the call, in the order the model reads them */
    [[nodiscard]] auto
    inputs() const
    {
        return [&]<std::size_t... P>(std::index_sequence<P...>) {
            return std::tuple<Argument<argumentOf(inputItems(P))> const &...>{
                std::get<argumentOf(inputItems(P))>(m_arguments)...
            };
        }(std::make_index_sequence<InputCount>{});
    }

    /** \returns whether every uncertain input is named in the model's affineIn() */
    [[nodiscard]] static consteval bool
    inputsAffine()
    {
        using Affine = typename Model::template Policy<PolicyKind::affineIn>;
        return []<std::size_t... P>(std::index_sequence<P...>) {
            return (affineNames<Affine, typename Item<inputItems(P)>::NameType> && ... && true);
        }(std::make_index_sequence<InputCount>{});
    }

    /** \returns whether item \a I is one the method factors out: named in the model's affineIn(), in a
     * linear space, and a block of the state or an uncertain input, since a sure input has no spread
     */
    template <std::size_t I>
    [[nodiscard]] static consteval bool
    itemFactored()
    {
        using Affine = typename Model::template Policy<PolicyKind::affineIn>;
        using Name = typename Item<I>::NameType;
        return affineNames<Affine, Name> && isLinear<Name> && (inState(I) || argumentIsUncertain(I));
    }

    /** \returns which blocks of the function's input the factored transform reads the Jacobian of,
     * one flag per block in order: the items factored out (itemFactored()); every block of a
     * whole-state model's input, which is its one block
     */
    template <std::size_t Blocks>
    [[nodiscard]] static constexpr std::array<bool, Blocks>
    factoredBlocks()
    {
        std::array<bool, Blocks> wanted{};
        if constexpr (Blocks == ItemCount)
        {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((wanted[I] = itemFactored<I>()), ...);
            }(std::make_index_sequence<ItemCount>{});
        }
        else
        {
            wanted.fill(true);
        }
        return wanted;
    }

    /** \returns the difference dimension of the items factored out, or Eigen::Dynamic: a function rather
     * than a constant, since MSVC reads a class template's own constexpr member in its member
     * declarations before it has initialized it
     */
    [[nodiscard]] static consteval int
    affineSize()
    {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return sizeSum<
                (itemFactored<I>() ? valueDimensionAtCompileTime<DifferenceSpace<typename Item<I>::NameType>> : 0)...>;
        }(std::make_index_sequence<ItemCount>{});
    }

    /** \returns a bound on affineSize(), or Eigen::Dynamic */
    [[nodiscard]] static consteval int
    affineMaxSize()
    {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return sizeSum<(
                itemFactored<I>() ? maxValueDimensionAtCompileTime<DifferenceSpace<typename Item<I>::NameType>> : 0
            )...>;
        }(std::make_index_sequence<ItemCount>{});
    }

    /** \returns whether any item is factored out, so that a sigma-point method spreads over fewer
     * coordinates
     */
    [[nodiscard]] static consteval bool
    factorsAffine()
    {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return (itemFactored<I>() || ... || false);
        }(std::make_index_sequence<ItemCount>{});
    }

    /** \returns the coordinates of the joint input's differences that the method factors out
     * (itemFactored()), in the order of reads(): the joint is the whole state's differences when
     * \a WholeState, and otherwise the joint of the blocks read, followed by each uncertain input's in
     * the order of inputs()
     */
    template <bool WholeState>
    [[nodiscard]] IndexVector<affineSize(), affineMaxSize()>
    affineCoordinates() const
    {
        std::array<Eigen::Index, InputCount + 1> inputStarts{};
        inputStarts[0] = WholeState ? m_stateDifferenceSize : m_reads.differenceIndices().size();
        std::apply(
            [&](auto const &... input) {
                std::size_t p = 0;
                ((inputStarts[p + 1] = inputStarts[p] + inputOf(input).differenceDimension(), ++p), ...);
            },
            inputs()
        );
        Eigen::Index count = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((count += itemFactored<I>() ? m_differenceRows[I].size : 0), ...);
        }(std::make_index_sequence<ItemCount>{});
        auto result = IndexVector<affineSize(), affineMaxSize()>{};
        result.resize(count);
        Eigen::Index next = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    if constexpr (itemFactored<I>())
                    {
                        if constexpr (inState(I))
                        {
                            // A block's rows of the state are the read indices from its start in the joint.
                            Eigen::Index const start = m_reads.jointDifferenceStart(stateIndexOf(I));
                            for (Eigen::Index r = 0; r < m_differenceRows[I].size; ++r)
                            {
                                result(next++) = WholeState ? m_reads.differenceIndices()(start + r) : start + r;
                            }
                        }
                        else
                        {
                            Eigen::Index const start = inputStarts[inputIndexOf(I)];
                            for (Eigen::Index r = 0; r < m_differenceRows[I].size; ++r)
                            {
                                result(next++) = start + r;
                            }
                        }
                    }
                }(),
                ...
            );
        }(std::make_index_sequence<ItemCount>{});
        return result;
    }

    /** \returns the input vector in \a T gathered from \a joint, a vector over the joint space of the
     * blocks read (reads()), the uncertain inputs \a inputs, a tuple of one vector each in the order
     * of inputs() or empty to read each at its mean, and the sure inputs
     */
    template <typename T, typename Joint, typename Inputs>
    [[nodiscard]] InVector<T>
    gather(Joint const & joint, Inputs const & inputs) const
    {
        auto x = InVector<T>{};
        x.resize(valueDimension(m_in));
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (gatherItem<I, T>(x, joint, inputs), ...);
        }(std::make_index_sequence<ItemCount>{});
        return x;
    }

    /** \returns the input vector in \a T gathered from \a rows, the rows of a state from row \a start
     * on, which hold every block read, and the call's arguments, with the uncertain inputs at the
     * vectors \a inputs, or at their means when the tuple is empty
     */
    template <typename T, typename Rows, typename Inputs>
    [[nodiscard]] InVector<T>
    gatherFromRows(Rows const & rows, Eigen::Index start, Inputs const & inputs) const
    {
        auto x = InVector<T>{};
        x.resize(valueDimension(m_in));
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (gatherItemFromRows<I, T>(x, rows, start, inputs), ...);
        }(std::make_index_sequence<ItemCount>{});
        return x;
    }

    /** Writes the input vectors of a batch into the columns of \a x, a matrix with a row per value
     * coordinate of the input space (decisions.md [batched-functions]): column \a j gathered from
     * <tt>rowsOf(j)</tt>, the rows of a state from row \a start on, which hold every block read, and
     * from the call's arguments, each uncertain input at its mean.
     */
    template <typename T, typename X, typename RowsOf>
    void
    gatherColumnsFromRows(X & x, RowsOf const & rowsOf, Eigen::Index start) const
    {
        EMRYS_FILTER_ASSERT(
            x.rows() == valueDimension(m_in) && "the batch has a row per value coordinate of the input"
        );
        for (Eigen::Index j = 0; j < x.cols(); ++j)
        {
            auto column = x.col(j);
            auto const & rows = rowsOf(j);
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                (gatherItemFromRows<I, T>(column, rows, start, std::tuple<>{}), ...);
            }(std::make_index_sequence<ItemCount>{});
        }
    }

protected:

    /** \returns the blocks of \a space that \a model reads, resolved with its items' selections */
    [[nodiscard]] static Reads
    makeReads(Space const & space, [[maybe_unused]] Model const & model)
    {
        if constexpr (Whole)
        {
            return Reads{space};
        }
        else
        {
            auto const & reads = model.template policy<PolicyKind::reads>();
            return [&]<std::size_t... K>(std::index_sequence<K...>) {
                return Reads{space, reads.template selection<stateItems(K)>()...};
            }(std::make_index_sequence<StateCount>{});
        }
    }

    /** \returns the model's input space instance: each item's name over the block it reads, in the
     * state or in the call's argument, or the state's space for a whole-state model
     */
    [[nodiscard]] In
    makeIn([[maybe_unused]] Arguments const & arguments) const
    {
        if constexpr (Whole)
        {
            return m_reads.space();
        }
        else
        {
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                return In{nameOf<typename Item<I>::NameType>(blockOf<I>(arguments))...};
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    /** \returns the space of the block item \a I reads: the state's block, or the argument's space */
    template <std::size_t I>
    [[nodiscard]] auto
    blockOf([[maybe_unused]] Arguments const & arguments) const
    {
        if constexpr (inState(I))
        {
            return m_reads.template blockSpace<stateIndexOf(I)>();
        }
        else
        {
            auto const & argument = std::get<argumentOf(I)>(arguments);
            if constexpr (isSure<std::remove_cvref_t<decltype(argument)>>)
            {
                return argument.space;
            }
            else
            {
                return inputOf(argument).space();
            }
        }
    }

    /** Writes item \a I's block of the input into \a x: from \a joint for a block of the state, and
     * from the call's arguments otherwise (gatherArgument()).
     */
    template <std::size_t I, typename T, typename Joint, typename Inputs>
    void
    gatherItem(InVector<T> & x, Joint const & joint, Inputs const & inputs) const
    {
        constexpr int Rows = valueDimensionAtCompileTime<typename Item<I>::NameType>;
        auto target = middleRowsOf<Rows>(x, m_valueRows[I].start, m_valueRows[I].size);
        if constexpr (inState(I))
        {
            target = middleRowsOf<Rows>(joint, m_reads.jointStart(stateIndexOf(I)), m_valueRows[I].size);
        }
        else
        {
            gatherArgument<I, T>(target, inputs);
        }
    }

    /** Writes item \a I's block of the input into \a x from the state rows \a rows, which start at
     * row \a start of the state, or from the call's arguments (gatherArgument()).
     */
    template <std::size_t I, typename T, typename X, typename Rows, typename Inputs>
    void
    gatherItemFromRows(X & x, Rows const & rows, Eigen::Index start, Inputs const & inputs) const
    {
        constexpr int Size = valueDimensionAtCompileTime<typename Item<I>::NameType>;
        auto target = middleRowsOf<Size>(x, m_valueRows[I].start, m_valueRows[I].size);
        if constexpr (inState(I))
        {
            // Asked only of an item in the state: no other has a position among the blocks read.
            if constexpr (Reads::template OneRun<stateIndexOf(I)>)
            {
                BlockRows const & block = m_reads.rows(stateIndexOf(I));
                EMRYS_FILTER_ASSERT(
                    block.start >= start && block.start + block.size <= start + rows.size()
                    && "the rows must hold every block read"
                );
                target = middleRowsOf<Size>(rows, block.start - start, block.size).template cast<T>();
            }
            else
            {
                // The rows of a list of repetitions are not one run: read each by its index.
                Eigen::Index const first = m_reads.jointStart(stateIndexOf(I));
                for (Eigen::Index j = 0; j < m_valueRows[I].size; ++j)
                {
                    Eigen::Index const row = m_reads.indices()(first + j) - start;
                    EMRYS_FILTER_ASSERT(row >= 0 && row < rows.size() && "the rows must hold every block read");
                    target(j) = T(rows(row));
                }
            }
        }
        else
        {
            gatherArgument<I, T>(target, inputs);
        }
    }

    /** Writes the value of the argument item \a I names into \a target: a sure input's value, an
     * uncertain input's vector in \a inputs, or its mean when \a inputs is empty.
     */
    template <std::size_t I, typename T, typename Target, typename Inputs>
    void
    gatherArgument(Target & target, [[maybe_unused]] Inputs const & inputs) const
    {
        auto const & argument = std::get<argumentOf(I)>(m_arguments);
        if constexpr (!argumentIsUncertain(I))
        {
            target = argument.value.template cast<T>();
        }
        else if constexpr (std::tuple_size_v<Inputs> > 0)
        {
            target = std::get<inputIndexOf(I)>(inputs);
        }
        else
        {
            target = inputOf(argument).mean().template cast<T>();
        }
    }

    Arguments const & m_arguments;                        ///< the call's arguments
    Reads m_reads;                                        ///< the blocks of the state read
    In m_in;                                              ///< the input space instance
    Eigen::Index m_stateDifferenceSize;                   ///< the state's difference dimension
    std::array<BlockRows, ItemCount> m_valueRows{};       ///< each item's rows in the input's values
    std::array<BlockRows, ItemCount> m_differenceRows{};  ///< each item's rows in the input's differences
};

}  // namespace internal

}  // namespace Emrys
