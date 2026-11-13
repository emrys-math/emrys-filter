#pragma once

/** \file
 * The Space concept and what every continuous space shares (decisions.md [sizes-and-dimensions]): the traits a
 * space exposes, DifferenceSpace, valueDimension(), differenceDimension(), and
 * allowsValueDimension(); ValueVector, DifferenceVector, and DifferenceMatrix, the Eigen types a
 * space sizes; isLinear<> and its parts; writable(); and the free functions plus(),
 * minus(), less(), normalizeValues(), and normalizeDifferences(), which add and subtract where a space
 * has no plus() or minus() of its own [difference-spaces]. The page of the Spaces group, which
 * starts here, describes the continuous spaces as a whole. Read Coordinates.h next, for the types
 * that implement the concept, then Composite.h, TagPath.h, and Subspace.h.
 */

#include <concepts>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>

namespace Emrys
{

namespace internal
{

/** Whether \a S has a normalize() the library can call on an Eigen block. */
template <typename S>
concept HasNormalize = requires(S const & space, Eigen::MatrixXd & block) { space.normalize(block.middleRows(0, 0)); };

/** Whether \a S has its own \f$ x \boxplus \delta \f$: a plus(x, d, out) the library can call on
 * Eigen blocks.
 */
template <typename S>
concept HasPlus = requires(S const & space, Eigen::MatrixXd & block) {
    space.plus(block.middleRows(0, 0), block.middleRows(0, 0), block.middleRows(0, 0));
};

/** Whether \a S has its own \f$ a \boxminus b \f$: a minus(a, b, out) the library can call on
 * Eigen blocks.
 */
template <typename S>
concept HasMinus = requires(S const & space, Eigen::MatrixXd & block) {
    space.minus(block.middleRows(0, 0), block.middleRows(0, 0), block.middleRows(0, 0));
};

/** Whether \a S has a normalize() that HasNormalize rejects but that a near miss of its form would
 * call: a template that is not const, or that takes its block by non-const reference, so that a
 * temporary block does not bind.
 */
template <typename S>
concept NearMissNormalize =
    !HasNormalize<S> && requires(S & space, Eigen::MatrixXd & block) { space.normalize(block); };

/** Whether \a S has a plus() that HasPlus rejects but that a near miss of its form would call, as
 * for NearMissNormalize.
 */
template <typename S>
concept NearMissPlus = !HasPlus<S> && requires(S & space, Eigen::MatrixXd & block) { space.plus(block, block, block); };

/** Whether \a S has a minus() that HasMinus rejects but that a near miss of its form would call, as
 * for NearMissNormalize.
 */
template <typename S>
concept NearMissMinus =
    !HasMinus<S> && requires(S & space, Eigen::MatrixXd & block) { space.minus(block, block, block); };

/** Whether \a S's normalize(), if the library can see one, takes an Eigen block of any number of
 * columns: false for a normalize() that is not a template, so that its address can be taken, and
 * that HasNormalize rejects, such as one written for a single vector, and false for a template
 * that is a near miss of the form (NearMissNormalize). The library would otherwise take it as
 * missing and the values as plain. A member template that rejects every block, or a misspelled
 * name, can't be seen this way and still counts as missing.
 */
template <typename S>
concept NormalizeTakesBlocks = HasNormalize<S> || (!requires { &S::normalize; } && !NearMissNormalize<S>);

/** Whether \a S's plus(), if the library can see one, takes Eigen blocks: as NormalizeTakesBlocks. */
template <typename S>
concept PlusTakesBlocks = HasPlus<S> || (!requires { &S::plus; } && !NearMissPlus<S>);

/** Whether \a S's minus(), if the library can see one, takes Eigen blocks: as NormalizeTakesBlocks. */
template <typename S>
concept MinusTakesBlocks = HasMinus<S> || (!requires { &S::minus; } && !NearMissMinus<S>);

/** Whether \a S names its difference space with a difference() member. */
template <typename S>
concept HasDifference = requires(S const & space) { space.difference(); };

/** The difference space of \a S: \a S itself unless it has a difference() member. */
template <typename S>
struct DifferenceSpaceOf
{
    /** The difference space. */
    using type = S;
};

/** The difference space of an \a S with a difference() member: the type it returns. */
template <HasDifference S>
struct DifferenceSpaceOf<S>
{
    /** The difference space. */
    using type = std::remove_cvref_t<decltype(std::declval<S const &>().difference())>;
};

/** Whether \a D can be the difference space of \a S: a regular type with a value dimension, fixed if \a S
 * is and bounded if \a S is. Its sizes may differ from \a S's. (Not the whole Space concept,
 * which would recurse through a space that is its own difference space.)
 */
template <typename D, typename S>
concept DifferenceSpaceFor = std::regular<D> && requires(D const & d) {
    { d.valueDimension() } -> std::convertible_to<Eigen::Index>;
    typename std::integral_constant<int, D::FixedValueDimension>;
    typename std::integral_constant<int, D::ValueDimensionStep>;
    typename std::integral_constant<int, D::MaxValueDimension>;
    requires(D::ValueDimensionStep == 0) == (S::ValueDimensionStep == 0);
    requires(D::MaxValueDimension == Eigen::Dynamic) == (S::MaxValueDimension == Eigen::Dynamic);
};

/** Whether the values and the differences of \a S have the same sizes. */
template <typename S>
inline constexpr bool hasEqualSizes = DifferenceSpaceOf<S>::type::FixedValueDimension == S::FixedValueDimension
                                   && DifferenceSpaceOf<S>::type::ValueDimensionStep == S::ValueDimensionStep
                                   && DifferenceSpaceOf<S>::type::MaxValueDimension == S::MaxValueDimension;

/** The number of resizable blocks of \a S, at any depth of nested composites: none for a fixed
 * space and one for any other resizable space. A Composite counts its blocks' (the specialization
 * after IsComposite).
 */
template <typename S>
inline constexpr int resizableBlockCount = S::ValueDimensionStep == 0 ? 0 : 1;

}  // namespace internal

/** \ingroup Spaces
 * \brief A type usable as the space of a distribution.
 *
 * Linear, Periodic, Composite, Repeated, and tags derived from them satisfy it. A
 * space written from scratch provides:
 *  - \c FixedValueDimension, \c ValueDimensionStep, and \c MaxValueDimension (\c int) as
 *    constant expressions, consistent with each other: a fixed space (\c ValueDimensionStep
 *    zero) has \c MaxValueDimension equal to \c FixedValueDimension, and a resizable one has
 *    \c MaxValueDimension at least \c FixedValueDimension, or \c Eigen::Dynamic;
 *  - valueDimension(), and resize(n) if it is resizable (a Composite with several resizable
 *    blocks has none);
 *  - if its values need normalizing, a const normalize() that takes any Eigen block by const
 *    reference and writes through it, rows being coordinates and columns samples;
 *  - if its differences are not plain values of the space itself, a const difference()
 *    returning its difference space (the library's spaces all have one, so that the difference
 *    space of a tag is never the tag): a regular type with \c FixedValueDimension,
 *    \c ValueDimensionStep, \c MaxValueDimension, and valueDimension(), fixed if the space is and
 *    bounded if the space is. A space with normalize()
 *    must have one, even if it returns Linear: whether differences wrap as values do is a
 *    choice to state. A space without one is its own difference space. Its sizes may differ
 *    from the space's, as a unit quaternion's differences are three-coordinate rotation
 *    vectors;
 *  - if \f$ x \boxplus \delta \f$ is not <tt>normalize(x + δ)</tt>, a const
 *    <tt>plus(x, d, out)</tt>, and if \f$ a \boxminus b \f$ is not <tt>a - b</tt> normalized in
 *    the difference space, a const <tt>minus(a, b, out)</tt>. Each takes Eigen blocks by const
 *    reference, columns paired, and writes the canonical result through \a out, which does not
 *    alias the inputs. A space has both or neither, since a residual taken by subtraction would
 *    not invert its own update, and with them a difference(). Customization is by these members
 *    only: the library calls Emrys::plus() and Emrys::minus() qualified, so a free plus() or
 *    minus() in the space's namespace is never found;
 *  - optionally, for a space with its own plus(), the reset Jacobian that carries a covariance to
 *    the tangent space at a moved mean (decisions.md [covariance-reset], Reset.h): a const
 *    <tt>resetJacobian(x, d, out)</tt> that writes it in closed form at one value column \a x and
 *    step \a d into the square \a out, or <tt>static constexpr bool identityReset = true;</tt> when
 *    it is the identity, as for a multiplicative \f$ x e^\delta \f$. Without either it is computed
 *    by automatic differentiation of plus() and minus(), which must then take
 *    \c Eigen::AutoDiffScalar coefficients;
 *  - value semantics: default construction, copying, and equality (\c std::regular).
 *
 * A normalize(), plus(), or minus() that is not a template and can't be called on Eigen blocks,
 * such as one written for a single vector, keeps the type from being a Space, whether it is on
 * the space or on its difference space: the library calls each on blocks of many columns, such
 * as every particle's state at once. So does a template that is a near miss of the form, one that
 * is not const or that takes a block by non-const reference, which the library would otherwise
 * take as missing and the space as a plain vector. One the library can't see, such as a
 * misspelled name or a member template that rejects every block, counts as missing.
 *
 * A space written from scratch is a single block: a tag path can name it but not reach
 * inside it.
 */
template <typename S>
concept Space = requires(S const & space) {
    typename std::integral_constant<int, S::FixedValueDimension>;
    typename std::integral_constant<int, S::ValueDimensionStep>;
    typename std::integral_constant<int, S::MaxValueDimension>;
    requires S::FixedValueDimension >= 0 && S::ValueDimensionStep >= 0;
    requires(
        S::ValueDimensionStep == 0
            ? S::MaxValueDimension == S::FixedValueDimension
            : (S::MaxValueDimension == Eigen::Dynamic || S::MaxValueDimension >= S::FixedValueDimension)
    );
    requires std::regular<S>;
    { space.valueDimension() } -> std::convertible_to<Eigen::Index>;
    requires internal::NormalizeTakesBlocks<S> && internal::PlusTakesBlocks<S> && internal::MinusTakesBlocks<S>;
    requires internal::NormalizeTakesBlocks<typename internal::DifferenceSpaceOf<S>::type>;
    requires internal::HasDifference<S>
                 || (!internal::HasNormalize<S> && !internal::HasPlus<S> && !internal::HasMinus<S>);
    requires internal::DifferenceSpaceFor<typename internal::DifferenceSpaceOf<S>::type, S>;
    requires internal::HasPlus<S> == internal::HasMinus<S>;
    requires internal::hasEqualSizes<S> || internal::HasPlus<S>;
    requires S::ValueDimensionStep == 0 || internal::resizableBlockCount<S> > 1
                     || requires(S & resizable, Eigen::Index n) { resizable.resize(n); };
};

/** \ingroup Spaces
 * A space whose instance holds nothing, so that <tt>S{}</tt> is the only one there is: a space
 * of fixed size, continuous or discrete, which is an empty type (decisions.md [space-storage]).
 * The free functions that take a space, such as plus(), minus(), and simulate(), take one of
 * these as a template argument instead: <tt>minus<State>(a, b)</tt>. A fixed-size space that
 * stores state is not one, and is passed as an instance.
 */
template <typename S>
concept FixedSizeSpace = std::is_empty_v<S> && std::default_initializable<S>;

/** \ingroup Spaces
 * The space that differences of values in \a S live in: the type of its difference(), or \a S
 * itself if it has none, as a space written from scratch may not. The library's spaces all have
 * one, and it carries no tag: <tt>DifferenceSpace<Gyro></tt> is <tt>Linear<1></tt> for a tag
 * <tt>struct Gyro : Linear<1> {};</tt>, so a block that is a difference of another quantity, such
 * as a sensor's bias, is declared in it: <tt>struct Bias : DifferenceSpace<Gyro> {};</tt>.
 */
template <Space S>
using DifferenceSpace = typename internal::DifferenceSpaceOf<S>::type;

/** \ingroup Spaces
 * \returns the difference space of \a space, with its sizes: its difference(), or a copy of
 * \a space if it has none. A fixed-size one is an empty type.
 */
template <Space S>
[[nodiscard]] DifferenceSpace<S>
differenceSpace(S const & space)
{
    if constexpr (internal::HasDifference<S>)
    {
        return space.difference();
    }
    else
    {
        return space;
    }
}

/** \ingroup Spaces
 * The value dimension of \a Space at compile time, or \c Eigen::Dynamic if it is resizable.
 */
template <typename Space>
inline constexpr int valueDimensionAtCompileTime =
    Space::ValueDimensionStep == 0 ? Space::FixedValueDimension : Eigen::Dynamic;

/** \ingroup Spaces
 * The largest valid value dimension of \a Space, or \c Eigen::Dynamic if it is unbounded.
 */
template <typename Space>
inline constexpr int maxValueDimensionAtCompileTime =
    Space::ValueDimensionStep == 0 ? Space::FixedValueDimension : Space::MaxValueDimension;

/** \ingroup Spaces
 * \returns the number of coordinates of a value in \a space, the length of a mean or a sample:
 * the space's own valueDimension().
 *
 * It is for run time and is not \c constexpr, even for a fixed space, as Eigen has \c rows() beside
 * \c RowsAtCompileTime: a resizable space's dimension is never a constant, and a user's space would
 * need every member \c constexpr. At compile time use valueDimensionAtCompileTime, which works for
 * every space, and <tt>valueDimensionAtCompileTime<DifferenceSpace<S>></tt> for the differences.
 */
template <Space S>
[[nodiscard]] Eigen::Index
valueDimension(S const & space)
{
    return space.valueDimension();
}

/** \ingroup Spaces
 * \returns the number of coordinates of a difference in \a space, the rows of a covariance, the
 * \f$ d \f$ in a density's normalizer, and the degrees of freedom of a NIS: the value dimension
 * of its difference space.
 */
template <Space S>
[[nodiscard]] Eigen::Index
differenceDimension(S const & space)
{
    if constexpr (internal::HasDifference<S>)
    {
        return space.difference().valueDimension();
    }
    else
    {
        return space.valueDimension();
    }
}

/** \ingroup Spaces
 * Whether the values of \a S are plain: it has neither normalize() nor plus(), so
 * \f$ x \boxplus \delta \f$ is <tt>x + δ</tt>. Periodic's are not; a space that wraps only its
 * differences has plain values.
 */
template <Space S>
inline constexpr bool hasPlainValues = !internal::HasNormalize<S> && !internal::HasPlus<S>;

/** \ingroup Spaces
 * Whether the differences of \a S are plain: its difference space has no normalize(), and it
 * has neither plus() nor minus(), so \f$ a \boxminus b \f$ is <tt>a - b</tt> and
 * \f$ x \boxplus \delta \f$ is <tt>x + δ</tt> up to normalize(). Linear has plain differences,
 * so a weighted mean in it is the plain one; Periodic does not.
 */
template <Space S>
inline constexpr bool hasPlainDifferences =
    !internal::HasNormalize<DifferenceSpace<S>> && !internal::HasPlus<S> && !internal::HasMinus<S>;

/** \ingroup Spaces
 * Whether \a S is linear: its values and its differences are both plain vectors. Linear, and a
 * Composite or Repeated of linear blocks, are linear; Periodic is not. Like hasPlainValues and
 * hasPlainDifferences, it takes only a Space, so a type that is not one does not compile rather
 * than counting as linear.
 */
template <Space S>
inline constexpr bool isLinear = hasPlainValues<S> && hasPlainDifferences<S>;

/** \ingroup Spaces
 * Brings the values in \a block (rows are \a space's coordinates, columns samples) into
 * canonical form, writing through the const reference as the space's normalize() does: for
 * values that come from outside, such as a user's mean or a model's output. Does nothing for a
 * space without normalize(). An update goes through plus() instead.
 */
template <Space S, typename Derived>
void
normalizeValues(S const & space, Eigen::MatrixBase<Derived> const & block)
{
    if constexpr (internal::HasNormalize<S>)
    {
        space.normalize(block);
    }
}

/** \ingroup Spaces
 * Brings the differences in \a block into canonical form: normalizeValues() in the difference
 * space of \a space, for differences that come from outside. Does nothing where that space has
 * no normalize(). A residual goes through minus() instead.
 */
template <Space S, typename Derived>
void
normalizeDifferences(S const & space, Eigen::MatrixBase<Derived> const & block)
{
    if constexpr (internal::HasNormalize<DifferenceSpace<S>>)
    {
        differenceSpace(space).normalize(block);
    }
}

/** \ingroup Spaces
 * normalizeValues() in the fixed-size space \a S, given as a template argument:
 * <tt>normalizeValues<State>(block)</tt>.
 */
template <FixedSizeSpace S, typename Derived>
    requires Space<S>
void
normalizeValues(Eigen::MatrixBase<Derived> const & block)
{
    normalizeValues(S{}, block);
}

/** \ingroup Spaces
 * normalizeDifferences() in the fixed-size space \a S, given as a template argument:
 * <tt>normalizeDifferences<State>(block)</tt>.
 */
template <FixedSizeSpace S, typename Derived>
    requires Space<S>
void
normalizeDifferences(Eigen::MatrixBase<Derived> const & block)
{
    normalizeDifferences(S{}, block);
}

/** \ingroup Spaces
 * \brief \a block, writable: Eigen's idiom for a function that writes through an expression
 * passed as <tt>const&</tt>.
 *
 * A space's normalize(), plus(), and minus() take their output as a
 * <tt>const Eigen::MatrixBase<Derived>&</tt>, so that a temporary block such as
 * <tt>x.segment(i, n)</tt> can be passed, and write through it with this:
 * <tt>writable(block) = block.unaryExpr(f)</tt>.
 */
template <typename Derived>
[[nodiscard]] Eigen::MatrixBase<Derived> &
writable(Eigen::MatrixBase<Derived> const & block)
{
    return const_cast<Eigen::MatrixBase<Derived> &>(block);
}

namespace internal
{
/** \returns \a x wrapped into <tt>[Lower, Lower + Period)</tt>.
 *
 * Values almost always arrive in range or within one period of it (a canonical value plus a small
 * update), so those cases take a comparison or one addition; only values further out pay for the
 * division and floor.
 */
template <double Lower, double Period, typename Scalar>
[[nodiscard]] constexpr Scalar
wrapValue(Scalar x)
{
    constexpr double upper = Lower + Period;
    if (x >= Scalar(Lower) && x < Scalar(upper)) [[likely]]
    {
        return x;
    }
    auto y = Scalar{};
    if (x >= Scalar(Lower - Period) && x < Scalar(upper + Period)) [[likely]]
    {
        // Not a conditional expression, whose branches are distinct expression types for an
        // automatic differentiation scalar.
        if (x < Scalar(Lower))
        {
            y = x + Scalar(Period);
        }
        else
        {
            y = x - Scalar(Period);
        }
    }
    else
    {
        y = x - Scalar(Period) * floorOf<Scalar>((x - Scalar(Lower)) / Scalar(Period));
    }
    // Rounding can leave y an ulp or so below the lower bound: the quotient can round up to the next
    // integer, and Lower - Period is itself rounded. One more period brings it back. Just below the
    // lower bound, that sum, or the one above, can round up to the excluded upper bound. Written so
    // that NaN passes through.
    if (y < Scalar(Lower))
    {
        y += Scalar(Period);
    }
    return y >= Scalar(upper) ? Scalar(Lower) : y;
}

/** A plain Eigen matrix with the storage order Eigen requires for single-row matrices. */
template <typename Scalar, int Rows, int Cols, int MaxRows, int MaxCols>
using MatrixOf = Eigen::
    Matrix<Scalar, Rows, Cols, (MaxRows == 1 && MaxCols != 1 ? Eigen::RowMajor : Eigen::ColMajor), MaxRows, MaxCols>;

/** How the columns of \a L and \a R pair up in plus() and minus(): an operand with one column at
 * compile time is used for every column of the other.
 */
template <typename L, typename R>
struct PairedColumns
{
    /// \a L has one column at compile time and \a R more: use it for every column of \a R
    static constexpr bool broadcastLeft = L::ColsAtCompileTime == 1 && R::ColsAtCompileTime != 1;

    /// \a R has one column at compile time and \a L more: use it for every column of \a L
    static constexpr bool broadcastRight = R::ColsAtCompileTime == 1 && L::ColsAtCompileTime != 1;

    /// the result's columns at compile time
    static constexpr int cols = broadcastLeft ? int(R::ColsAtCompileTime) : int(L::ColsAtCompileTime);

    /// the result's bound on its columns
    static constexpr int maxCols = broadcastLeft ? int(R::MaxColsAtCompileTime) : int(L::MaxColsAtCompileTime);

    /** \returns the number of paired columns of \a left and \a right */
    template <typename Left, typename Right>
    [[nodiscard]] static Eigen::Index
    count(Left const & left, Right const & right)
    {
        EMRYS_FILTER_ASSERT(
            (broadcastLeft || broadcastRight || left.cols() == right.cols())
            && "the operands have different numbers of columns"
        );
        return broadcastLeft ? right.cols() : left.cols();
    }
};

/** \returns \a m with \a cols columns: repeated if \a Broadcast, otherwise \a m itself. */
template <bool Broadcast, typename Derived>
[[nodiscard]] decltype(auto)
withColumns(Eigen::MatrixBase<Derived> const & m, Eigen::Index cols)
{
    if constexpr (Broadcast)
    {
        return m.derived().replicate(1, cols);
    }
    else
    {
        return m.derived();
    }
}

/** \returns \a m itself where its coefficients are in memory (a plain matrix, a map, or a block of
 * one), and otherwise \a m evaluated into a plain matrix: what a space's plus() or minus() reads
 * by block or by column without recomputing an expression such as a product.
 */
template <typename Derived>
[[nodiscard]] decltype(auto)
inMemory(Eigen::MatrixBase<Derived> const & m)
{
    if constexpr ((Eigen::internal::traits<Derived>::Flags & Eigen::DirectAccessBit) != 0)
    {
        return m.derived();
    }
    else
    {
        return m.eval();
    }
}

/** Writes \f$ x \boxplus \delta \f$ through \a out: \a space's plus(), or <tt>x + d</tt>
 * normalized.
 */
template <typename S, typename X, typename D, typename Out>
void
plusInto(
    S const & space,
    Eigen::MatrixBase<X> const & x,
    Eigen::MatrixBase<D> const & d,
    Eigen::MatrixBase<Out> const & out
)
{
    if constexpr (HasPlus<S>)
    {
        space.plus(x, d, out);
    }
    else
    {
        writable(out) = x + d;
        normalizeValues(space, out);
    }
}

/** Writes \f$ a \boxminus b \f$ through \a out: \a space's minus(), or <tt>a - b</tt> normalized
 * in the difference space.
 */
template <typename S, typename A, typename B, typename Out>
void
minusInto(
    S const & space,
    Eigen::MatrixBase<A> const & a,
    Eigen::MatrixBase<B> const & b,
    Eigen::MatrixBase<Out> const & out
)
{
    if constexpr (HasMinus<S>)
    {
        space.minus(a, b, out);
    }
    else
    {
        writable(out) = a - b;
        normalizeDifferences(space, out);
    }
}

/** The number of rows \a space takes in an operand of differences if \a Difference, otherwise of
 * values.
 */
template <bool Difference, typename S>
[[nodiscard]] Eigen::Index
rowsOf(S const & space)
{
    if constexpr (Difference)
    {
        return differenceSpace(space).valueDimension();
    }
    else
    {
        return space.valueDimension();
    }
}

/** The number of rows rowsOf() gives for a space of type \a S at compile time, or
 * \c Eigen::Dynamic.
 */
template <bool Difference, typename S>
inline constexpr int rowsAtCompileTime =
    Difference ? valueDimensionAtCompileTime<DifferenceSpace<S>> : valueDimensionAtCompileTime<S>;

/** \returns the \a rows rows of \a m from \a offset, as a block of \a Rows rows where \a Rows is
 * fixed, so that Eigen unrolls what a block of a fixed-size space does with them, and of
 * \a rows rows otherwise.
 */
template <int Rows, typename M>
[[nodiscard]] auto
middleRowsOf(M & m, Eigen::Index offset, Eigen::Index rows)
{
    if constexpr (Rows == Eigen::Dynamic)
    {
        return m.middleRows(offset, rows);
    }
    else
    {
        EMRYS_FILTER_ASSERT(rows == Rows && "the block's rows must be its compile-time size");
        return m.template middleRows<Rows>(offset);
    }
}

/** \returns the \a cols columns of \a m from \a offset, as a block of \a Cols columns where
 * \a Cols is fixed, as middleRowsOf() gives rows.
 */
template <int Cols, typename M>
[[nodiscard]] auto
middleColsOf(M & m, Eigen::Index offset, Eigen::Index cols)
{
    if constexpr (Cols == Eigen::Dynamic)
    {
        return m.middleCols(offset, cols);
    }
    else
    {
        EMRYS_FILTER_ASSERT(cols == Cols && "the block's columns must be its compile-time size");
        return m.template middleCols<Cols>(offset);
    }
}

/** \returns the \a rows by \a cols block of \a m from row \a row and column \a col, with \a Rows
 * rows and \a Cols columns at compile time where they are fixed, as middleRowsOf() gives rows.
 */
template <int Rows, int Cols, typename M>
[[nodiscard]] auto
blockOf(M & m, Eigen::Index row, Eigen::Index col, Eigen::Index rows, Eigen::Index cols)
{
    EMRYS_FILTER_ASSERT(
        (Rows == Eigen::Dynamic || rows == Rows) && (Cols == Eigen::Dynamic || cols == Cols)
        && "the block's sizes must be its compile-time sizes"
    );
    return m.template block<Rows, Cols>(row, col, rows, cols);
}

}  // namespace internal

/** \ingroup Spaces
 * A value in \a S as an Eigen column vector of \a Scalar: a mean, a sample, or a true state. It
 * has valueDimensionAtCompileTime\<S\> rows, and a resizable space's bound as its maximum, so a
 * bounded space's vector never allocates. A block's own space gives a block's value
 * (<tt>ValueVector<Position></tt>), and \a Scalar may be an automatic-differentiation scalar, as
 * a model's function written generically in its scalar uses.
 */
template <Space S, typename Scalar = double>
using ValueVector =
    Eigen::Matrix<Scalar, valueDimensionAtCompileTime<S>, 1, Eigen::ColMajor, maxValueDimensionAtCompileTime<S>, 1>;

/** \ingroup Spaces
 * A difference in \a S as an Eigen column vector of \a Scalar, the type of minus() and of the
 * offset plus() takes: a residual, an estimation error, a step, or the variances on a covariance's
 * diagonal. It is a value in DifferenceSpace\<S\>, so it has fewer rows than a ValueVector where the
 * space's differences are smaller, as a unit quaternion's 3 are than its 4 coordinates.
 */
template <Space S, typename Scalar = double>
using DifferenceVector = ValueVector<DifferenceSpace<S>, Scalar>;

/** \ingroup Spaces
 * An Eigen matrix of \a Scalar over the differences of \a Rows and \a Cols: a covariance, with one
 * space for both (<tt>DifferenceMatrix<State></tt>), or a Jacobian, a gain, or a linear model's
 * matrix, from \a Cols's differences to \a Rows's (<tt>DifferenceMatrix<Fix, State></tt>). A
 * single row is stored row-major, as Eigen requires.
 */
template <Space Rows, Space Cols = Rows, typename Scalar = double>
using DifferenceMatrix = internal::MatrixOf<
    Scalar,
    valueDimensionAtCompileTime<DifferenceSpace<Rows>>,
    valueDimensionAtCompileTime<DifferenceSpace<Cols>>,
    maxValueDimensionAtCompileTime<DifferenceSpace<Rows>>,
    maxValueDimensionAtCompileTime<DifferenceSpace<Cols>>>;

/** \ingroup Spaces
 * \returns \f$ x \boxplus \delta \f$: the values \a x moved by the differences \a d, in canonical
 * form. Rows are coordinates and columns samples; an operand with one column at compile time is
 * used for every column of the other, so <tt>plus(space, mean, L * z)</tt> draws samples. For
 * most spaces this is <tt>normalizeValues(space, x + d)</tt>, and for a linear one just
 * <tt>x + d</tt>.
 */
template <Space S, typename XDerived, typename DDerived>
[[nodiscard]] auto
plus(S const & space, Eigen::MatrixBase<XDerived> const & x, Eigen::MatrixBase<DDerived> const & d)
{
    using Columns = internal::PairedColumns<XDerived, DDerived>;
    // The result's rows are the space's where it knows them, so that an unbounded operand into a
    // bounded space gives a bounded result.
    constexpr int rows = valueDimensionAtCompileTime<S> != Eigen::Dynamic ? valueDimensionAtCompileTime<S>
                                                                          : int(XDerived::RowsAtCompileTime);
    constexpr int maxRows = maxValueDimensionAtCompileTime<S> != Eigen::Dynamic ? maxValueDimensionAtCompileTime<S>
                                                                                : int(XDerived::MaxRowsAtCompileTime);
    using Result = internal::MatrixOf<typename XDerived::Scalar, rows, Columns::cols, maxRows, Columns::maxCols>;
    Eigen::Index const cols = Columns::count(x, d);
    auto result = Result{};
    result.resize(x.rows(), cols);
    // Evaluated once where they are expressions, so a space's plus() can read them by block or by
    // column without recomputing a product; a broadcast stays a lazy replicate of the evaluated operand.
    auto const & xInMemory = internal::inMemory(x);
    auto const & dInMemory = internal::inMemory(d);
    auto const & paired = internal::withColumns<Columns::broadcastLeft>(xInMemory, cols);
    auto const & offsets = internal::withColumns<Columns::broadcastRight>(dInMemory, cols);
    if constexpr (internal::HasPlus<S>)
    {
        space.plus(paired, offsets, result);
    }
    else
    {
        result = paired + offsets;
        normalizeValues(space, result);
    }
    return result;
}

/** \ingroup Spaces
 * \returns \f$ a \boxminus b \f$: the differences from \a b to \a a, in canonical form. Rows are
 * coordinates and columns samples; an operand with one column at compile time is used for every
 * column of the other, so <tt>minus(space, samples, mean)</tt> gives each sample's deviation.
 * For most spaces this is <tt>normalizeDifferences(space, a - b)</tt>, and for a linear one just
 * <tt>a - b</tt>.
 */
template <Space S, typename ADerived, typename BDerived>
[[nodiscard]] auto
minus(S const & space, Eigen::MatrixBase<ADerived> const & a, Eigen::MatrixBase<BDerived> const & b)
{
    using Columns = internal::PairedColumns<ADerived, BDerived>;
    // The result's rows are the difference space's where it knows them, so that an unbounded
    // operand into a bounded space gives a bounded result, and the operands' rows otherwise, which
    // is right only where the space's differences have the sizes of its values.
    constexpr bool sameRows = internal::hasEqualSizes<S>;
    constexpr int spaceRows = valueDimensionAtCompileTime<DifferenceSpace<S>>;
    constexpr int spaceMaxRows = maxValueDimensionAtCompileTime<DifferenceSpace<S>>;
    constexpr int rows = spaceRows != Eigen::Dynamic ? spaceRows
                       : sameRows                    ? int(ADerived::RowsAtCompileTime)
                                                     : Eigen::Dynamic;
    constexpr int maxRows = spaceMaxRows != Eigen::Dynamic ? spaceMaxRows
                          : sameRows                       ? int(ADerived::MaxRowsAtCompileTime)
                                                           : Eigen::Dynamic;
    using Result = internal::MatrixOf<typename ADerived::Scalar, rows, Columns::cols, maxRows, Columns::maxCols>;
    Eigen::Index const cols = Columns::count(a, b);
    auto result = Result{};
    result.resize(sameRows ? a.rows() : differenceDimension(space), cols);
    auto const & aInMemory = internal::inMemory(a);
    auto const & bInMemory = internal::inMemory(b);
    auto const & left = internal::withColumns<Columns::broadcastLeft>(aInMemory, cols);
    auto const & right = internal::withColumns<Columns::broadcastRight>(bInMemory, cols);
    if constexpr (internal::HasMinus<S>)
    {
        space.minus(left, right, result);
    }
    else
    {
        result = left - right;
        normalizeDifferences(space, result);
    }
    return result;
}

/** \ingroup Spaces
 * \returns \f$ x \boxplus (-\delta) \f$: the values \a x moved back by the differences \a d, in
 * canonical form, as <tt>plus(space, x, -d)</tt>. It takes a difference away from a value, as
 * minus() takes one value from another: a reading less its bias is
 * <tt>less<Gyro>(reading, bias)</tt> for a bias in <tt>DifferenceSpace<Gyro></tt>. Columns pair
 * as in plus().
 *
 * It undoes plus() exactly, <tt>less(space, plus(space, x, d), d) == x</tt> up to rounding, on a
 * space whose plus() is a group's, as on Linear, Periodic, and UnitQuaternion; on a space whose
 * plus() is some other retraction, it undoes it only to first order in \a d.
 *
 * A model need not use it: on a Linear space it is <tt>x - d</tt>, and the filter normalizes the
 * values a model returns, so bare arithmetic is a choice for plainness that is right wherever the
 * space's plus() is addition (docs/usage.md, Conventions).
 */
template <Space S, typename XDerived, typename DDerived>
[[nodiscard]] auto
less(S const & space, Eigen::MatrixBase<XDerived> const & x, Eigen::MatrixBase<DDerived> const & d)
{
    return Emrys::plus(space, x, -d);
}

/** \ingroup Spaces
 * \returns plus() in the fixed-size space \a S, given as a template argument:
 * <tt>plus<State>(x, d)</tt>.
 */
template <FixedSizeSpace S, typename XDerived, typename DDerived>
    requires Space<S>
[[nodiscard]] auto
plus(Eigen::MatrixBase<XDerived> const & x, Eigen::MatrixBase<DDerived> const & d)
{
    return Emrys::plus(S{}, x, d);
}

/** \ingroup Spaces
 * \returns minus() in the fixed-size space \a S, given as a template argument:
 * <tt>minus<State>(a, b)</tt>.
 */
template <FixedSizeSpace S, typename ADerived, typename BDerived>
    requires Space<S>
[[nodiscard]] auto
minus(Eigen::MatrixBase<ADerived> const & a, Eigen::MatrixBase<BDerived> const & b)
{
    return Emrys::minus(S{}, a, b);
}

/** \ingroup Spaces
 * \returns less() in the fixed-size space \a S, given as a template argument:
 * <tt>less<Gyro>(reading, bias)</tt>.
 */
template <FixedSizeSpace S, typename XDerived, typename DDerived>
    requires Space<S>
[[nodiscard]] auto
less(Eigen::MatrixBase<XDerived> const & x, Eigen::MatrixBase<DDerived> const & d)
{
    return Emrys::less(S{}, x, d);
}

/** \ingroup Spaces
 * \returns whether some instance of \a Space has value dimension \a n: at least its
 * \c FixedValueDimension, a whole number of \c ValueDimensionStep beyond it, and at most its
 * \c MaxValueDimension. An instance's own is its valueDimension().
 *
 * The test is exact for a space with at most one resizable block. For one with several it is
 * necessary but not sufficient: a Composite of blocks with strides 2 and 3 has stride 1, and no
 * instance has 1 coordinate.
 */
template <typename Space>
[[nodiscard]] constexpr bool
allowsValueDimension(Eigen::Index n)
{
    if constexpr (Space::ValueDimensionStep == 0)
    {
        return n == Space::FixedValueDimension;
    }
    else
    {
        return n >= Space::FixedValueDimension && (n - Space::FixedValueDimension) % Space::ValueDimensionStep == 0
            && (Space::MaxValueDimension == Eigen::Dynamic || n <= Space::MaxValueDimension);
    }
}

namespace internal
{

/** Resizes \a space to \a n coordinates if it has resize(n), after checking that \a n is a valid
 * value dimension for its type, and checks that it then has it. A space with several resizable
 * blocks has no resize(n), so it must already have \a n.
 */
template <typename Space>
void
fitValueDimension(Space & space, Eigen::Index n)
{
    EMRYS_FILTER_ASSERT(allowsValueDimension<Space>(n) && "the dimension does not fit the space");
    if constexpr (requires { space.resize(n); })
    {
        if (space.valueDimension() != n)
        {
            space.resize(n);
        }
        EMRYS_FILTER_ASSERT(space.valueDimension() == n && "the space's resize() did not give it this dimension");
    }
    else
    {
        EMRYS_FILTER_ASSERT(space.valueDimension() == n && "the dimension does not fit the space");
    }
}

}  // namespace internal

}  // namespace Emrys
