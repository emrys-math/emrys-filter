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
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The blocks a model reads from a state in \a Space, named by the paths in \a Reads. */
template <typename Space, typename Reads>
class ReadBlocks;

/** The blocks named by \a Paths, resolved in \a Space: their rows in the state, in values and in
 * differences, and the joint space they form, in the order given.
 */
template <typename Space, typename... Paths>
class ReadBlocks<Space, TagPaths<Paths...>>
{
    static_assert(sizeof...(Paths) > 0, "a model reads at least one block of the state");

    /** The first path. */
    using First = std::tuple_element_t<0, std::tuple<Paths...>>;

public:

    static constexpr std::size_t Count = sizeof...(Paths);  ///< the number of blocks read

    /** The space of the blocks read: the one block's, or the Composite of them. */
    using JointSpace = std::conditional_t<Count == 1, PathTag<First>, Composite<PathTag<Paths>...>>;

    static constexpr int Size = valueDimensionAtCompileTime<JointSpace>;        ///< the joint value dimension
    static constexpr int MaxSize = maxValueDimensionAtCompileTime<JointSpace>;  ///< a bound on it
    static constexpr int DifferenceSize =
        valueDimensionAtCompileTime<DifferenceSpace<JointSpace>>;  ///< the joint difference dimension
    static constexpr int DifferenceMaxSize =
        maxValueDimensionAtCompileTime<DifferenceSpace<JointSpace>>;           ///< a bound on it
    using Indices = IndexVector<Size, MaxSize>;                                ///< the value coordinates read
    using DifferenceIndices = IndexVector<DifferenceSize, DifferenceMaxSize>;  ///< the difference coordinates read
    using Selection = Coordinates<Indices, DifferenceIndices>;                 ///< both, for a GaussianMarginal

    /** Resolves the blocks' rows in \a space and the joint space they form. */
    explicit ReadBlocks(Space const & space)
        : m_rows{blockRows<Paths>(space)...}
        , m_differenceRows{differenceBlockRows<Paths>(space)...}
        , m_space{makeSpace(space)}
    {
        Eigen::Index offset = 0;
        Eigen::Index differenceOffset = 0;
        for (std::size_t i = 0; i < Count; ++i)
        {
            m_jointStarts[i] = offset;
            m_jointDifferenceStarts[i] = differenceOffset;
            offset += m_rows[i].size;
            differenceOffset += m_differenceRows[i].size;
        }
        m_indices = concatenate<Indices>(m_rows);
        m_differenceIndices = concatenate<DifferenceIndices>(m_differenceRows);
        EMRYS_FILTER_ASSERT(distinctIndices(m_indices) && "a model must not read a coordinate twice");
    }

    /** \returns the joint space of the blocks read, with their parameters */
    [[nodiscard]] JointSpace const &
    space() const
    {
        return m_space;
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

    /** \returns block \a k's rows in a state's values */
    [[nodiscard]] BlockRows const &
    rows(std::size_t k) const
    {
        return m_rows[k];
    }

    /** \returns block \a k's rows in a state's differences */
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
    makeSpace(Space const & space)
    {
        if constexpr (Count == 1)
        {
            return pathBlock<First>(space);
        }
        else
        {
            return JointSpace{pathBlock<Paths>(space)...};
        }
    }

    /** \returns the rows of \a blocks, block after block, as one index vector */
    template <typename Result>
    [[nodiscard]] static Result
    concatenate(std::array<BlockRows, Count> const & blocks)
    {
        Eigen::Index size = 0;
        for (BlockRows const & rows : blocks)
        {
            size += rows.size;
        }
        auto result = Result{};
        result.resize(size);
        Eigen::Index k = 0;
        for (BlockRows const & rows : blocks)
        {
            for (Eigen::Index j = 0; j < rows.size; ++j)
            {
                result(k++) = rows.start + j;
            }
        }
        return result;
    }

    std::array<BlockRows, Count> m_rows;                        ///< each block's rows in a state's values
    std::array<BlockRows, Count> m_differenceRows;              ///< each block's rows in a state's differences
    std::array<Eigen::Index, Count> m_jointStarts{};            ///< each block's first row in the joint values
    std::array<Eigen::Index, Count> m_jointDifferenceStarts{};  ///< each block's first row in the joint differences
    Indices m_indices;                                          ///< the value coordinates read
    DifferenceIndices m_differenceIndices;                      ///< the difference coordinates read
    JointSpace m_space;                                         ///< the joint space
};

/** The blocks a whole-state model reads (decisions.md [whole-state-models]): the whole state in
 * \a Space as one block, its rows every row, with the same interface as the blocks named by paths.
 */
template <typename Space>
class ReadBlocks<Space, TagPaths<WholeSpace>>
{
public:

    static constexpr std::size_t Count = 1;  ///< one block, the whole state

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

    /** The blocks of the state the model reads, resolved. */
    using Reads = ReadBlocks<Space, StatePaths>;

    /** A vector in the input space, in \a T. */
    template <typename T>
    using InVector = typename Moments<In, T>::VectorType;

    /** Resolves the model's reads() against \a space and the call's \a arguments, which both outlive
     * this.
     */
    ResolvedReads(Arguments const & arguments, Space const & space)
        : m_arguments{arguments}
        , m_reads{space}
        , m_in{makeIn(arguments, space)}
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
                        Eigen::Index start = 0;
                        if constexpr (inState(I))
                        {
                            start = WholeState ? m_reads.differenceRows(stateIndexOf(I)).start
                                               : m_reads.jointDifferenceStart(stateIndexOf(I));
                        }
                        else
                        {
                            start = inputStarts[inputIndexOf(I)];
                        }
                        for (Eigen::Index r = 0; r < m_differenceRows[I].size; ++r)
                        {
                            result(next++) = start + r;
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

    /** \returns the model's input space instance: each item's name over the block it reads, in the
     * state or in the call's argument, or the state's space for a whole-state model
     */
    [[nodiscard]] static In
    makeIn([[maybe_unused]] Arguments const & arguments, Space const & space)
    {
        if constexpr (Whole)
        {
            return space;
        }
        else
        {
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                return In{nameOf<typename Item<I>::NameType>(blockOf<I>(arguments, space))...};
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    /** \returns the space of the block item \a I reads: the state's block, or the argument's space */
    template <std::size_t I>
    [[nodiscard]] static auto
    blockOf([[maybe_unused]] Arguments const & arguments, [[maybe_unused]] Space const & space)
    {
        if constexpr (inState(I))
        {
            return pathSubspace<typename Item<I>::PathType>(space);
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
            BlockRows const & block = m_reads.rows(stateIndexOf(I));
            EMRYS_FILTER_ASSERT(
                block.start >= start && block.start + block.size <= start + rows.size()
                && "the rows must hold every block read"
            );
            target = middleRowsOf<Size>(rows, block.start - start, block.size).template cast<T>();
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
