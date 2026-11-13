#pragma once

/** \file
 * Spaces built from other spaces (decisions.md [sizes-and-dimensions], [several-resizable-blocks]): Composite, which
 * concatenates blocks, and Repeated, which repeats a fixed-size block a run-time number of times. The internal
 * namespace recognizes them and the tags derived from them (IsComposite, IsRepeated, IsUniformLeaf, IsPeriodic), counts
 * a space's resizable blocks for SizedByLength, and visits each leaf block with forEachLeaf(). Read TagPath.h next.
 */

#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** How PartStorage keeps \a Part: in a one-element tuple if it has state, else not at all. */
template <typename Part>
using StoredPart = std::conditional_t<std::is_empty_v<Part>, std::tuple<>, std::tuple<Part>>;

/** The parts with state, held as a member so that \c std is not an associated namespace. */
template <typename Stored>
struct StoredParts
{
    Stored parts;  ///< the parts with state, in order

    /** Equal when the stored parts are. */
    friend bool
    operator==(StoredParts const &, StoredParts const &) = default;
};

/** No parts with state: an empty base. */
template <>
struct StoredParts<std::tuple<>>
{
    StoredParts() = default;

    /** Accepts the empty tuple that PartStorage builds when no part has state. */
    constexpr explicit StoredParts(std::tuple<>) {}

    /** Always equal: nothing is stored. */
    friend bool
    operator==(StoredParts const &, StoredParts const &) = default;
};

/** The parts of a Composite (its blocks) or a DiscreteProduct (its dimensions), in order.
 *
 * Only parts with state are stored. An empty part holds nothing, so it is one shared instance,
 * and the stored parts are a base class, so that when every part is empty the storage is an
 * empty type, and so is a class derived from it.
 */
template <typename... Parts>
class PartStorage : private StoredParts<decltype(std::tuple_cat(std::declval<StoredPart<Parts>>()...))>
{
    /** The tuple of parts with state. */
    using Stored = decltype(std::tuple_cat(std::declval<StoredPart<Parts>>()...));

    /** The base holding it. */
    using Base = StoredParts<Stored>;

    /** Part \a I's type. */
    template <std::size_t I>
    using Part = std::tuple_element_t<I, std::tuple<Parts...>>;

    /** \returns the position of part \a I among the stored parts */
    static consteval std::size_t
    storedIndex(std::size_t i)
    {
        constexpr auto empty = std::array<bool, sizeof...(Parts)>{std::is_empty_v<Parts>...};
        std::size_t result = 0;
        for (std::size_t j = 0; j < i; ++j)
        {
            result += empty[j] ? 0 : 1;
        }
        return result;
    }

    /** \returns \a part in a one-element tuple if it has state, else an empty tuple */
    template <typename P>
    static constexpr StoredPart<P>
    keep(P & part)
    {
        if constexpr (std::is_empty_v<P>)
        {
            return {};
        }
        else
        {
            return StoredPart<P>{std::move(part)};
        }
    }

    /** The one shared instance of an empty part \a I. */
    template <std::size_t I>
    static inline Part<I> instance{};

public:

    PartStorage() = default;

    /** Stores those of \a parts that have state. */
    constexpr explicit PartStorage(Parts... parts)
        : Base{std::tuple_cat(keep(parts)...)}
    {}

    /** \returns part \a I: the stored one, or the shared instance of an empty type */
    template <std::size_t I>
    [[nodiscard]] constexpr Part<I> const &
    part() const
    {
        if constexpr (std::is_empty_v<Part<I>>)
        {
            return instance<I>;
        }
        else
        {
            return std::get<storedIndex(I)>(Base::parts);
        }
    }

    /** \returns part \a I, to change it: the stored one, or the shared instance of an empty type */
    template <std::size_t I>
    [[nodiscard]] Part<I> &
    part()
    {
        if constexpr (std::is_empty_v<Part<I>>)
        {
            return instance<I>;
        }
        else
        {
            return std::get<storedIndex(I)>(Base::parts);
        }
    }

    /** Equal when the parts with state are. */
    friend bool
    operator==(PartStorage const &, PartStorage const &) = default;
};

}  // namespace internal

/** \ingroup Spaces
 * \brief The concatenation of \a Blocks, any number of them resizable.
 *
 * Nesting a named composite names a range of coordinates; storage is always the flat
 * concatenation. Each block's offset is the total size of the blocks before it, so a block after
 * a resizable one has its offset only at run time, from idx<Path>(space). A composite of empty
 * blocks, such as any fixed-size composite of the coordinate types, is itself an empty type, so it
 * costs a distribution nothing.
 *
 * A composite with one resizable block, at any depth, has resize(n). One with several has none,
 * since a length does not say how to split it among them: its blocks are resized through block(),
 * or through a distribution's insert() and erase().
 */
template <Space... Blocks>
class Composite : private internal::PartStorage<Blocks...>
{
    /** The blocks with state. */
    using Storage = internal::PartStorage<Blocks...>;

    static_assert(sizeof...(Blocks) > 0, "a composite needs at least one block");
    static constexpr std::size_t BlockCount = sizeof...(Blocks);  ///< the number of blocks
    static constexpr std::array<bool, BlockCount> Resizable{
        (Blocks::ValueDimensionStep != 0)...
    };  ///< which blocks are resizable

    /** \returns the stride of the composite: the greatest common divisor of its resizable blocks'
     * strides, or zero
     */
    static consteval int
    strides()
    {
        int result = 0;
        ((result = std::gcd(result, Blocks::ValueDimensionStep)), ...);
        return result;
    }

    /** \returns the position of the first resizable block, or \c BlockCount if none */
    static consteval std::size_t
    firstResizable()
    {
        for (std::size_t i = 0; i < BlockCount; ++i)
        {
            if (Resizable[i])
            {
                return i;
            }
        }
        return BlockCount;
    }

public:

    static constexpr int FixedValueDimension =
        (Blocks::FixedValueDimension + ...);  ///< the blocks' fixed coordinates together

    /** Zero if fixed; otherwise the greatest common divisor of its resizable blocks' strides, which
     * every change in its value dimension is a multiple of.
     */
    static constexpr int ValueDimensionStep = strides();

    /** The sum of the blocks' bounds, or \c Eigen::Dynamic if any is unbounded. */
    static constexpr int MaxValueDimension =
        ((Blocks::MaxValueDimension == Eigen::Dynamic) || ...) ? Eigen::Dynamic : (Blocks::MaxValueDimension + ...);

    Composite() = default;

    /** The concatenation of \a blocks, with their parameters and sizes. */
    explicit Composite(Blocks... blocks)
        : Storage{std::move(blocks)...}
    {}

    /** \returns block \a I */
    template <std::size_t I>
    [[nodiscard]] auto const &
    block() const
    {
        return Storage::template part<I>();
    }

    /** \returns block \a I, to resize it */
    template <std::size_t I>
    [[nodiscard]] auto &
    block()
    {
        return Storage::template part<I>();
    }

    /** \returns the number of coordinates: the blocks' sizes together */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return offset<BlockCount>();
    }

    /** \returns the offset of block \a I's first coordinate */
    template <std::size_t I>
    [[nodiscard]] Eigen::Index
    offset() const
    {
        return [&]<std::size_t... J>(std::index_sequence<J...>) {
            return (Eigen::Index{0} + ... + block<J>().valueDimension());
        }(std::make_index_sequence<I>{});
    }

    /** Resizes the composite's only resizable block, at any depth, so the composite has \a n
     * coordinates.
     */
    void
    resize(Eigen::Index n)
        requires((internal::resizableBlockCount<Blocks> + ...) == 1)
    {
        auto & resizable = block<firstResizable()>();
        Eigen::Index const rest = valueDimension() - resizable.valueDimension();
        EMRYS_FILTER_ASSERT(n >= rest && "the dimension does not fit the space");
        resizable.resize(n - rest);
    }

    /** Normalizes each block's rows; a composite of blocks without normalize() has none. */
    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & rows) const
        requires((internal::HasNormalize<Blocks> || ...))
    {
        forEachBlock(rows, [](auto const & block, auto && sub) {
            Emrys::normalizeValues(block, sub);
        });
    }

    /** \f$ x \boxplus \delta \f$ block by block, for a composite with a block that has its own
     * plus(); see Emrys::plus().
     */
    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
        requires((internal::HasPlus<Blocks> || ...))
    {
        forEachBlockOf<false, true, false>(
            x,
            d,
            out,
            [](auto const & block, auto const & xs, auto const & ds, auto const & os) {
                internal::plusInto(block, xs, ds, os);
            }
        );
    }

    /** \f$ a \boxminus b \f$ block by block, for a composite with a block that has its own
     * minus(); see Emrys::minus().
     */
    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
        requires((internal::HasMinus<Blocks> || ...))
    {
        forEachBlockOf<false, false, true>(
            a,
            b,
            out,
            [](auto const & block, auto const & as, auto const & bs, auto const & os) {
                internal::minusInto(block, as, bs, os);
            }
        );
    }

    /** The composite of the blocks' difference spaces, with their sizes. It keeps no tags, the
     * blocks' or its own, even where every block is linear.
     */
    [[nodiscard]] Composite<DifferenceSpace<Blocks>...>
    difference() const
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return Composite<DifferenceSpace<Blocks>...>{differenceSpace(block<I>())...};
        }(std::make_index_sequence<BlockCount>{});
    }

    /** Equal when every stored parameter and size is; a fixed-size Composite is always equal. */
    friend bool
    operator==(Composite const &, Composite const &) = default;

private:

    /** Calls \a f(block, rows) with each block and its rows of \a rows, an operand of values. */
    template <typename Derived, typename F>
    void
    forEachBlock(Eigen::MatrixBase<Derived> const & rows, F f) const
    {
        auto & m = writable(rows);
        EMRYS_FILTER_ASSERT(m.rows() == valueDimension() && "the rows do not match the space");
        Eigen::Index offset = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((f(block<I>(),
                internal::middleRowsOf<valueDimensionAtCompileTime<std::tuple_element_t<I, std::tuple<Blocks...>>>>(
                    m,
                    offset,
                    block<I>().valueDimension()
                )),
              offset += block<I>().valueDimension()),
             ...);
        }(std::make_index_sequence<BlockCount>{});
    }

    /** Calls \a f with each block and its rows of \a first, \a second, and \a out. An operand
     * whose flag is set holds differences, so a block takes its difference space's size there.
     */
    template <
        bool FirstDifference,
        bool SecondDifference,
        bool OutDifference,
        typename First,
        typename Second,
        typename Out,
        typename F>
    void
    forEachBlockOf(
        Eigen::MatrixBase<First> const & first,
        Eigen::MatrixBase<Second> const & second,
        Eigen::MatrixBase<Out> const & out,
        F f
    ) const
    {
        auto & o = writable(out);
        Eigen::Index firstOffset = 0;
        Eigen::Index secondOffset = 0;
        Eigen::Index outOffset = 0;
        auto const step = [&]<typename B>(B const & b) {
            Eigen::Index const firstRows = internal::rowsOf<FirstDifference>(b);
            Eigen::Index const secondRows = internal::rowsOf<SecondDifference>(b);
            Eigen::Index const outRows = internal::rowsOf<OutDifference>(b);
            f(b,
              internal::middleRowsOf<internal::rowsAtCompileTime<FirstDifference, B>>(first, firstOffset, firstRows),
              internal::middleRowsOf<internal::rowsAtCompileTime<SecondDifference, B>>(
                  second,
                  secondOffset,
                  secondRows
              ),
              internal::middleRowsOf<internal::rowsAtCompileTime<OutDifference, B>>(o, outOffset, outRows));
            firstOffset += firstRows;
            secondOffset += secondRows;
            outOffset += outRows;
        };
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (step(block<I>()), ...);
        }(std::make_index_sequence<BlockCount>{});
        EMRYS_FILTER_ASSERT(
            firstOffset == first.rows() && secondOffset == second.rows() && outOffset == o.rows()
            && "the rows do not match the space"
        );
    }
};

/** \ingroup Spaces
 * \brief A fixed-size \a Block repeated count() times, up to \a MaxCount_, such as one
 * block per vehicle.
 *
 * Its size is count() times the block's size. Every repetition shares the block's
 * parameters (periods and bounds). A tag derived from it can inherit its constructors to
 * start with a count: <tt>struct Vehicles : Repeated<Vehicle> { using Repeated::Repeated; };</tt>,
 * then <tt>Vehicles{Vehicle{}, 3}</tt>.
 */
template <Space Block, int MaxCount_ = Eigen::Dynamic>
class Repeated
{
    static_assert(
        Block::ValueDimensionStep == 0 && Block::FixedValueDimension > 0,
        "a repeated block must have a fixed, nonzero size"
    );
    static_assert(MaxCount_ == Eigen::Dynamic || MaxCount_ >= 0);

public:

    /** The repeated block's type. */
    using BlockType = Block;

    /** Maximum number of repetitions, or \c Eigen::Dynamic if unbounded. */
    static constexpr int MaxCount = MaxCount_;

    static constexpr int FixedValueDimension = 0;  ///< nothing is fixed: every coordinate belongs to a repetition
    static constexpr int ValueDimensionStep = Block::FixedValueDimension;  ///< the coordinates of one repetition

    /** \c MaxCount repetitions' worth of coordinates, or \c Eigen::Dynamic if unbounded. */
    static constexpr int MaxValueDimension =
        MaxCount == Eigen::Dynamic ? Eigen::Dynamic : MaxCount * Block::FixedValueDimension;

    Repeated() = default;

    /** \a count repetitions of \a block. */
    explicit Repeated(Block block, Eigen::Index count = 0)
        : m_block{std::move(block)}
    {
        setCount(count);
    }

    /** \returns the repeated block, with its parameters */
    [[nodiscard]] Block const &
    block() const
    {
        return m_block;
    }

    /** \returns the number of repetitions */
    [[nodiscard]] Eigen::Index
    count() const
    {
        return m_count;
    }

    /** \returns the number of coordinates: count() repetitions of \c ValueDimensionStep */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return m_count * ValueDimensionStep;
    }

    /** Sets the number of repetitions to \a count, at most \c MaxCount. */
    void
    setCount(Eigen::Index count)
    {
        EMRYS_FILTER_ASSERT(
            count >= 0 && (MaxCount == Eigen::Dynamic || count <= MaxCount) && "the dimension does not fit the space"
        );
        m_count = count;
    }

    /** Resizes to \a n coordinates, a whole number of repetitions. */
    void
    resize(Eigen::Index n)
    {
        EMRYS_FILTER_ASSERT(n >= 0 && n % ValueDimensionStep == 0 && "the dimension does not fit the space");
        setCount(n / ValueDimensionStep);
    }

    /** Normalizes each repetition; a repeated block without normalize() has none. */
    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & rows) const
        requires(internal::HasNormalize<Block>)
    {
        forEachRepetition(rows, [this](auto && sub) {
            Emrys::normalizeValues(m_block, sub);
        });
    }

    /** \f$ x \boxplus \delta \f$ for each repetition, for a block with its own plus(). */
    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
        requires(internal::HasPlus<Block>)
    {
        forEachRepetitionOf<false, true, false>(x, d, out, [this](auto const & xs, auto const & ds, auto const & os) {
            internal::plusInto(m_block, xs, ds, os);
        });
    }

    /** \f$ a \boxminus b \f$ for each repetition, for a block with its own minus(). */
    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
        requires(internal::HasMinus<Block>)
    {
        forEachRepetitionOf<false, false, true>(a, b, out, [this](auto const & as, auto const & bs, auto const & os) {
            internal::minusInto(m_block, as, bs, os);
        });
    }

    /** The block's difference space, repeated as many times. It keeps no tags. */
    [[nodiscard]] Repeated<DifferenceSpace<Block>, MaxCount_>
    difference() const
    {
        return Repeated<DifferenceSpace<Block>, MaxCount_>{differenceSpace(m_block), m_count};
    }

    /** Equal when every stored parameter and size is; a fixed-size Repeated is always equal. */
    friend bool
    operator==(Repeated const &, Repeated const &) = default;

private:

    /** Calls \a f(rows) with each repetition's rows of \a rows, an operand of values. */
    template <typename Derived, typename F>
    void
    forEachRepetition(Eigen::MatrixBase<Derived> const & rows, F f) const
    {
        auto & m = writable(rows);
        EMRYS_FILTER_ASSERT(m.rows() == valueDimension() && "the rows do not match the space");
        for (Eigen::Index k = 0; k < m_count; ++k)
        {
            f(internal::middleRowsOf<ValueDimensionStep>(m, k * ValueDimensionStep, ValueDimensionStep));
        }
    }

    /** Calls \a f with each repetition's rows of \a first, \a second, and \a out; see
     * Composite::forEachBlockOf().
     */
    template <
        bool FirstDifference,
        bool SecondDifference,
        bool OutDifference,
        typename First,
        typename Second,
        typename Out,
        typename F>
    void
    forEachRepetitionOf(
        Eigen::MatrixBase<First> const & first,
        Eigen::MatrixBase<Second> const & second,
        Eigen::MatrixBase<Out> const & out,
        F f
    ) const
    {
        auto & o = writable(out);
        Eigen::Index const firstRows = internal::rowsOf<FirstDifference>(m_block);
        Eigen::Index const secondRows = internal::rowsOf<SecondDifference>(m_block);
        Eigen::Index const outRows = internal::rowsOf<OutDifference>(m_block);
        EMRYS_FILTER_ASSERT(
            first.rows() == m_count * firstRows && second.rows() == m_count * secondRows
            && o.rows() == m_count * outRows && "the rows do not match the space"
        );
        for (Eigen::Index k = 0; k < m_count; ++k)
        {
            f(internal::middleRowsOf<internal::rowsAtCompileTime<FirstDifference, Block>>(
                  first,
                  k * firstRows,
                  firstRows
              ),
              internal::middleRowsOf<internal::rowsAtCompileTime<SecondDifference, Block>>(
                  second,
                  k * secondRows,
                  secondRows
              ),
              internal::middleRowsOf<internal::rowsAtCompileTime<OutDifference, Block>>(o, k * outRows, outRows));
        }
    }

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Block m_block;  ///< the repeated block, with its parameters
    Eigen::Index m_count = 0;                      ///< the number of repetitions
};

namespace internal
{

/** Declared, never defined: matched by a pointer to a Composite or a tag derived from one, so
 * that IsComposite and CompositeBlocks see through the derivation.
 */
template <typename... Blocks>
std::tuple<Blocks...> *
compositeBlocks(Composite<Blocks...> const *);

/** A Composite, or a tag derived from one. */
template <typename T>
concept IsComposite = requires(T const * t) { compositeBlocks(t); };

/** The blocks of the Composite \a T, as a tuple type. */
template <typename T>
using CompositeBlocks = std::remove_pointer_t<decltype(compositeBlocks(std::declval<T const *>()))>;

/** resizableBlockCount summed over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
inline constexpr int resizableBlockCountIn = 0;

/** resizableBlockCount summed over \a Blocks. */
template <typename... Blocks>
inline constexpr int resizableBlockCountIn<std::tuple<Blocks...>> = (0 + ... + resizableBlockCount<Blocks>);

/** The resizable blocks of a Composite, or of a tag derived from one: its blocks' together. */
template <IsComposite S>
inline constexpr int resizableBlockCount<S> = resizableBlockCountIn<CompositeBlocks<S>>;

/** A space that takes its sizes from a length alone: it has at most one resizable block, at any
 * depth. A distribution can be built in one from a mean or a length without a space, and
 * resized to a length. A space with several takes its counts from the space a caller gives, so
 * that a wrong split of the right total length (2 tracks and 5 landmarks against data for 3 and
 * 3) can't go unnoticed.
 */
template <typename S>
concept SizedByLength = resizableBlockCount<S> <= 1;

/** Declared, never defined: matched by a pointer to a Repeated or a tag derived from one. */
template <typename Block, int MaxCount>
Block *
repeatedBlock(Repeated<Block, MaxCount> const *);

/** A Repeated, or a tag derived from one. */
template <typename T>
concept IsRepeated = requires(T const * t) { repeatedBlock(t); };

/** The repeated block of the Repeated \a T. */
template <typename T>
using RepeatedBlock = std::remove_pointer_t<decltype(repeatedBlock(std::declval<T const *>()))>;

/** Declared, never defined: true for a pointer to a LeafExtent or a type derived from one. */
template <int Dimension, int MaxDimension>
std::true_type
leafExtent(LeafExtent<Dimension, MaxDimension> const *);

/** The fallback for anything else. */
std::false_type
leafExtent(void const *);

/** A Linear or Periodic space (or a tag derived from one): every coordinate alike. */
template <typename T>
concept IsUniformLeaf = decltype(leafExtent(std::declval<T const *>()))::value;

/** Declared, never defined: true for a pointer to a Periodic or a type derived from one. */
template <PeriodicInterval Interval, int Dimension, int MaxDimension>
std::true_type
periodicLeaf(Periodic<Interval, Dimension, MaxDimension> const *);

/** The fallback for anything else. */
std::false_type
periodicLeaf(void const *);

/** A Periodic space, or a tag derived from one. */
template <typename T>
concept IsPeriodic = decltype(periodicLeaf(std::declval<T const *>()))::value;

/** Where the values of a leaf block lie: <tt>[lower, upper)</tt> wrapping with period
 * <tt>upper - lower</tt> if \c periodic, else the whole line.
 */
struct LeafSupport
{
    double lower = -std::numeric_limits<double>::infinity();  ///< the lower end
    double upper = std::numeric_limits<double>::infinity();   ///< the upper end
    bool periodic = false;                                    ///< whether values wrap at the ends
};

/** \returns the support of the leaf block \a Leaf: the whole line for a linear one, its period
 * for a Periodic one, and nothing for a non-linear space written from scratch, whose range the
 * library can't see.
 */
// constexpr rather than consteval: MSVC 14.50 can't apply -> to a consteval call's optional result.
template <typename Leaf>
[[nodiscard]] constexpr std::optional<LeafSupport>
leafSupport()
{
    if constexpr (IsPeriodic<Leaf>)
    {
        return LeafSupport{.lower = Leaf::lower(), .upper = Leaf::upper(), .periodic = true};
    }
    else if constexpr (isLinear<Leaf>)
    {
        return LeafSupport{};
    }
    else
    {
        return std::nullopt;
    }
}

/** Calls \a f(offset, leaf) for each leaf block of \a space, in order, where \a offset is the
 * leaf's first coordinate: each Linear or Periodic block, each repetition of a
 * Repeated block's leaves, and a space written from scratch as a whole.
 */
template <typename Space, typename F>
void
forEachLeaf(Space const & space, F && f, Eigen::Index offset = 0)
{
    if constexpr (IsComposite<Space>)
    {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((forEachLeaf(space.template block<I>(), f, offset), offset += space.template block<I>().valueDimension()),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<Space>>>{});
    }
    else if constexpr (IsRepeated<Space>)
    {
        for (Eigen::Index k = 0; k < space.count(); ++k)
        {
            forEachLeaf(space.block(), f, offset + k * Space::ValueDimensionStep);
        }
    }
    else
    {
        f(offset, space);
    }
}

}  // namespace internal

}  // namespace Emrys
