#pragma once

/** \file
 * Grids of points and the state over them (decisions.md [rb-point-mass], [rao-blackwellized-core]): GridAxis and
 * GridCells for a regular grid, PointCells for given points, PointGrid with regularGrid() and pointGrid() for fixed
 * points of one block with optional cell covariances, and GridGaussianParams, the live cells with their probabilities
 * and a %Gaussian or a hybrid of Gaussians over the rest each, which prunes and subdivides, the point-mass filters'
 * state, with its density GridGaussian, which gives the collapse. Read ParticleGaussian.h next; the design review is
 * docs/guide/rao-blackwellized-filters.md.
 */

#include <algorithm>
#include <array>
#include <expected>
#include <cmath>
#include <cstddef>
#include <optional>
#include <expected>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Conditionals.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

template <typename Grid_, typename Model_, typename Covariance_, typename Method_, typename Upkeep_>
class RbPointMassFilter;

template <
    typename Grid_,
    typename Covariance_ = SharedCovariance,
    typename Component_ = Gaussian<typename Grid_::RestSpace, typename Grid_::Scalar>>
class GridGaussian;

/** \ingroup Distributions
 * \brief The axis of a regular grid of points (see regularGrid()) along coordinate \a I of the
 * block \a Block: a run-time number of cells.
 *
 * \a Block is the innermost tagged block that holds the coordinate. A grid over <tt>Motion :
 * Composite<Heading, Speed></tt> has the axes <tt>GridAxis<Heading></tt> and <tt>GridAxis<Speed></tt>,
 * and one over <tt>Position : Linear<2></tt> the axes <tt>GridAxis<Position, 0></tt> and
 * <tt>GridAxis<Position, 1></tt>, so the probability of the heading is
 * <tt>marginal<GridAxis<Heading>>()</tt>.
 */
template <typename Block, int I = 0>
struct GridAxis : Finite<Eigen::Dynamic>
{
    /** A GridAxis is made like a Finite, with its number of cells. */
    using Finite::Finite;
};

namespace internal
{

/** Declared, never defined: the axes of the leaf block \a Block, one per coordinate. */
template <typename Block, std::size_t... I>
std::tuple<GridAxis<Block, int(I)>...> leafGridAxes(std::index_sequence<I...>);

/** The axes of a grid over \a Block, as a tuple type: a leaf block's coordinates in order. */
template <typename Block>
struct GridAxesOf
{
    /** The axes. */
    using type = decltype(leafGridAxes<Block>(std::make_index_sequence<std::size_t(Block::FixedValueDimension)>{}));
};

/** A Composite's axes are those of its blocks, in order. */
template <typename Block>
    requires IsComposite<Block>
struct GridAxesOf<Block>
{
    /** Declared, never defined: the axes of \a Blocks, joined. */
    template <typename... Blocks>
    static auto
    of(std::tuple<Blocks...> *) -> decltype(std::tuple_cat(std::declval<typename GridAxesOf<Blocks>::type>()...));

    /** The axes. */
    using type = decltype(of(static_cast<CompositeBlocks<Block> *>(nullptr)));
};

/** Declared, never defined: the DiscreteProduct of \a Axes. */
template <typename... Axes>
DiscreteProduct<Axes...>
gridProduct(std::tuple<Axes...> *);

/** How many of \a Axes are \a Axis. */
template <typename Axis, typename... Axes>
inline constexpr int axisOccurrences = (0 + ... + int(std::is_same_v<Axis, Axes>));

/** \returns whether \a Axes are distinct types */
template <typename... Axes>
consteval bool
distinctAxes(std::tuple<Axes...> *)
{
    return ((axisOccurrences<Axes, Axes...> == 1) && ...);
}

/** The axes of a grid over the block named by the path \a Path, as a tuple type. */
template <typename Path>
using GridAxes = typename GridAxesOf<PathTag<Path>>::type;

}  // namespace internal

/** \ingroup Distributions
 * \brief The cells of a regular grid over the block named by the path \a Path: the
 * DiscreteProduct of its GridAxis axes, the last varying fastest, under a tag of its own.
 *
 * The tag names the grid's cells in a path wherever they go: <tt>at<GridCells<Position>>(3, 7)</tt>
 * is a cell in valueOf(), and <tt>TagPath<GridCells<Position>, GridAxis<Position, 0>></tt> an axis
 * when two grids share a discrete space.
 */
template <typename Path>
struct GridCells : decltype(internal::gridProduct(static_cast<internal::GridAxes<Path> *>(nullptr)))
{
    static_assert(
        internal::distinctAxes(static_cast<internal::GridAxes<Path> *>(nullptr)),
        "two coordinates of the block lie in blocks with the same tag, so their grid axes would share a name"
    );

    /** The DiscreteProduct of the axes. */
    using Product = decltype(internal::gridProduct(static_cast<internal::GridAxes<Path> *>(nullptr)));

    using Product::Product;
};

/** \ingroup Distributions
 * \brief The cells of a grid of given points (see pointGrid()) for the block named by the path
 * \a Path: one value per point, under a tag of its own.
 */
template <typename Path>
struct PointCells : Finite<Eigen::Dynamic>
{
    /** PointCells are made like a Finite, with their number of points. */
    using Finite::Finite;
};

namespace internal
{

/** Whether \a Cells is a GridCells, so its grid is regular. */
template <typename Cells>
struct IsGridCells : std::false_type
{};

/** The cells of a regular grid. */
template <typename Path>
struct IsGridCells<GridCells<Path>> : std::true_type
{};

}  // namespace internal

/** \ingroup Distributions
 * \brief Fixed points for the block named by the path \a Path_ of a state in \a Space_, one per
 * value of the discrete space \a Cells_, such as the cells of a grid of positions.
 *
 * The grid of the point-mass filter of \ref Bucy1971 "Bucy and Senne (1971)", here for one block
 * of the state only, with the rest %Gaussian [rb-point-mass].
 *
 * Each point is the center of its cell. With \a Cells_ GridCells, the grid is regular: the
 * centers are computed from the first one and the spacing, and the grid can be subdivided (see
 * subdivided()). Otherwise the centers are given and stored.
 *
 * A cell may also have a covariance about its center: by default none, making each cell a point
 * mass. With \a CellCovariance_ SharedCovariance there is at most one, the same for every cell,
 * such as \f$ \mathrm{diag}(s^2 / 12) \f$ for a grid with spacing \f$ s \f$. With
 * PerPointCovariance, each cell has its own, as cells of different sizes do (an irregular set of
 * points, or a latitude-longitude grid). See RbPointMassFilter for what a cell covariance does and
 * doesn't account for.
 *
 * \tparam Space_           the whole state space
 * \tparam Path_            the block the points fix, as a tag or a TagPath
 * \tparam Cells_           the discrete space of cells (see \ref Discrete)
 * \tparam CellCovariance_  SharedCovariance or PerPointCovariance, for the cells' covariances
 * \tparam Scalar_          scalar type, \c double by default
 * \tparam Allocator_       allocator for the stored centers and covariances, and for the
 *                          GridGaussian over the grid (see \ref Allocation)
 *
 * \sa regularGrid(), pointGrid(), GridGaussian, RbPointMassFilter
 */
template <
    Space Space_,
    typename Path_,
    DiscreteSpace Cells_,
    typename CellCovariance_ = SharedCovariance,
    RealScalar Scalar_ = double,
    typename Allocator_ = DefaultAllocator>
class PointGrid
{
    static_assert(
        internal::CovarianceForm<CellCovariance_>,
        "the cell covariance is SharedCovariance or PerPointCovariance"
    );

    static constexpr bool Shared = std::is_same_v<CellCovariance_, SharedCovariance>;  ///< at most one cell covariance

public:

    /** The scalar type. */
    using Scalar = Scalar_;

    /** The whole state space. */
    using SpaceType = Space_;

    /** The discrete space of cells. */
    using CellsType = Cells_;

    /** The path naming the fixed block. */
    using PathType = Path_;

    /** SharedCovariance or PerPointCovariance. */
    using CellCovarianceType = CellCovariance_;

    /** The allocator, re-bound per buffer. */
    using allocator_type = Allocator_;

    /** The space of the fixed block. */
    using PinnedSpace = typename internal::BlockSplit<Space_, Path_>::PinnedSpace;

    /** The space of the rest of the state, in order. */
    using RestSpace = typename internal::BlockSplit<Space_, Path_>::RestSpace;

    /** A plain matrix of centers, one per column, laid out as centers() is. */
    using CentersType = SampleSet<PinnedSpace, Scalar_>;

    /** A value of the block, such as a center. */
    using PinnedVector = typename Moments<PinnedSpace, Scalar_>::VectorType;

    /** A covariance over the block. */
    using PinnedMatrix = typename Moments<PinnedSpace, Scalar_>::MatrixType;

    /** Whether the grid is regular: its cells are GridCells, and its centers are computed. */
    static constexpr bool Regular = internal::IsGridCells<Cells_>::value;

    static_assert(
        !Regular || std::is_same_v<Cells_, GridCells<Path_>>,
        "a regular grid's cells are the GridCells of its own block"
    );

    /** Points in \a space with \a centers, one column per value of \a cells in flat-index order, and
     * \a cellCovariance, every cell's covariance about its center (none by default). The centers are
     * copied into the grid's storage, so a map over a buffer of one's own passes them without
     * allocating.
     */
    template <typename CentersDerived>
    PointGrid(
        Space_ space,
        Cells_ cells,
        Eigen::MatrixBase<CentersDerived> const & centers,
        std::optional<PinnedMatrix> cellCovariance = {},
        Allocator_ const & allocator = {}
    )
        requires(Shared && !Regular)
        : PointGrid{Parts{}, std::move(space), std::move(cells), listOf(cellCovariance), allocator}
    {
        setCenters(centers);
    }

    /** Points in \a space with \a centers, one column per value of \a cells in flat-index order, and
     * \a cellCovariances, each cell's covariance about its center, in the same order, from any
     * contiguous storage.
     */
    template <typename CentersDerived>
    PointGrid(
        Space_ space,
        Cells_ cells,
        Eigen::MatrixBase<CentersDerived> const & centers,
        std::span<PinnedMatrix const> cellCovariances,
        Allocator_ const & allocator = {}
    )
        requires(!Shared && !Regular)
        : PointGrid{Parts{}, std::move(space), std::move(cells), cellCovariances, allocator}
    {
        setCenters(centers);
    }

    /** A regular grid in \a space over \a cells, whose cell with coordinates \f$ c \f$ (see
     * DiscreteProduct::coordinates()) has its center at \f$ \mathit{first} + c \, \mathit{spacing} \f$,
     * each coordinate scaling its axis's spacing, and \a cellCovariance, every cell's covariance
     * about its center (none by default).
     */
    PointGrid(
        Space_ space,
        Cells_ cells,
        PinnedVector first,
        PinnedVector spacing,
        std::optional<PinnedMatrix> cellCovariance = {},
        Allocator_ const & allocator = {}
    )
        requires(Shared && Regular)
        : PointGrid{Parts{}, std::move(space), std::move(cells), listOf(cellCovariance), allocator}
    {
        setGeometry(std::move(first), std::move(spacing));
    }

    /** A regular grid as above, with \a cellCovariances, each cell's covariance about its center in
     * flat-index order.
     */
    PointGrid(
        Space_ space,
        Cells_ cells,
        PinnedVector first,
        PinnedVector spacing,
        std::span<PinnedMatrix const> cellCovariances,
        Allocator_ const & allocator = {}
    )
        requires(!Shared && Regular)
        : PointGrid{Parts{}, std::move(space), std::move(cells), cellCovariances, allocator}
    {
        setGeometry(std::move(first), std::move(spacing));
    }

    /** \returns the allocator the stored centers and covariances come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_cellCovariances.get_allocator());
    }

    /** \returns the whole state space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the discrete space of cells, with its cardinalities */
    [[nodiscard]] Cells_ const &
    cells() const
    {
        return m_cells;
    }

    /** \returns the fixed block's space, with its parameters */
    [[nodiscard]] PinnedSpace const &
    pinnedSpace() const
    {
        return m_split.pinnedSpace();
    }

    /** \returns the space of the rest of the state, in order */
    [[nodiscard]] RestSpace const &
    restSpace() const
    {
        return m_split.restSpace();
    }

    /** \returns the number of cells */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return m_cells.cardinality();
    }

    /** \returns the center of the cell with flat index \a cell */
    [[nodiscard]] PinnedVector
    center(Eigen::Index cell) const
    {
        EMRYS_FILTER_ASSERT(cell >= 0 && cell < size() && "the cell is out of range");
        if constexpr (Regular)
        {
            auto const coordinates = m_cells.coordinates(cell);
            PinnedVector result = m_first;
            for (std::size_t i = 0; i < coordinates.size(); ++i)
            {
                Eigen::Index const row = Eigen::Index(i);
                result(row) += Scalar(coordinates[i]) * m_spacing(row);
            }
            normalizeValues(pinnedSpace(), result);
            return result;
        }
        else
        {
            return PinnedVector{centers().col(cell)};
        }
    }

    /** \returns every cell's center, one per column */
    [[nodiscard]] ConstSampleSetMap<PinnedSpace, Scalar_>
    centers() const
        requires(!Regular)
    {
        return {m_centers.data(), pinnedRows().size, size()};
    }

    /** \returns the center of the first cell of a regular grid */
    [[nodiscard]] PinnedVector const &
    first() const
        requires Regular
    {
        return m_first;
    }

    /** \returns the spacing of a regular grid along each axis */
    [[nodiscard]] PinnedVector const &
    spacing() const
        requires Regular
    {
        return m_spacing;
    }

    /** \returns whether the cells have covariances about their centers, rather than being point masses */
    [[nodiscard]] bool
    hasCellCovariance() const
    {
        return !m_cellCovariances.empty();
    }

    /** \returns the covariance about its center of the cell with flat index \a cell; the cells must have them */
    [[nodiscard]] PinnedMatrix const &
    cellCovariance(Eigen::Index cell) const
    {
        EMRYS_FILTER_ASSERT(hasCellCovariance() && "the cells are point masses");
        EMRYS_FILTER_ASSERT(cell >= 0 && cell < size() && "the cell is out of range");
        return m_cellCovariances[Shared ? 0 : std::size_t(cell)];
    }

    /** \returns the rows of the fixed block in the whole state */
    [[nodiscard]] BlockRows
    pinnedRows() const
    {
        return m_split.pinnedRows();
    }

    /** \returns the rows of the rest of the state in the whole state, in order */
    [[nodiscard]] auto const &
    restIndices() const
    {
        return m_split.restIndices();
    }

    /** \returns the rows of the fixed block in the whole state's differences */
    [[nodiscard]] BlockRows
    pinnedDifferenceRows() const
    {
        return m_split.pinnedDifferenceRows();
    }

    /** \returns the rows of the rest of the state in the whole state's differences, in order */
    [[nodiscard]] auto const &
    restDifferenceIndices() const
    {
        return m_split.restDifferenceIndices();
    }

    /** \returns the regular grid with each cell split into \a factor cells along each axis: the
     * spacing divided by \a factor, the centers spread evenly over each old cell, and the cell
     * covariance divided by \a factor squared. The cell with coordinates \f$ c \f$ becomes the
     * cells with coordinates \f$ \mathit{factor} \, c + a \f$, \f$ 0 \le a_i < \mathit{factor} \f$.
     *
     * With the covariance \f$ \mathrm{diag}(s^2 / 12) \f$ of a cell of width \f$ s \f$, the new
     * cells' spread and covariances together match the old cell's covariance exactly.
     */
    [[nodiscard]] PointGrid
    subdivided(Eigen::Index factor) const
        requires(Shared && Regular)
    {
        EMRYS_FILTER_ASSERT(factor >= 1 && "subdivide each cell into at least one");
        Cells_ cells = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return Cells_{std::tuple_element_t<I, internal::DiscreteProductAxes<Cells_>>{
                m_cells.template axis<I>().cardinality() * factor
            }...};
        }(std::make_index_sequence<Cells_::AxisCount>{});
        PinnedVector spacing = m_spacing / Scalar(factor);
        PinnedVector first = m_first - Scalar(0.5) * Scalar(factor - 1) * spacing;
        auto cellCovariance = std::optional<PinnedMatrix>{};
        if (hasCellCovariance())
        {
            cellCovariance = m_cellCovariances.front() / Scalar(factor * factor);
        }
        return PointGrid{
            m_space,
            std::move(cells),
            std::move(first),
            std::move(spacing),
            std::move(cellCovariance),
            get_allocator()
        };
    }

private:

    /** Tags the constructor that takes the covariances as a list, whatever the form. */
    struct Parts
    {};

    /** \returns \a covariance as a list of zero or one covariances, borrowed from it */
    [[nodiscard]] static std::span<PinnedMatrix const>
    listOf(std::optional<PinnedMatrix> const & covariance)
    {
        return covariance ? std::span<PinnedMatrix const>{&*covariance, 1} : std::span<PinnedMatrix const>{};
    }

    /** Stores \a space, \a cells, and \a cellCovariances, none, one, or one per cell, through
     * \a allocator; the centers or geometry follow.
     */
    PointGrid(
        Parts,
        Space_ space,
        Cells_ cells,
        std::span<PinnedMatrix const> cellCovariances,
        Allocator_ const & allocator
    )
        : m_space{std::move(space)}
        , m_cells{std::move(cells)}
        , m_split{m_space}
        , m_cellCovariances(cellCovariances.begin(), cellCovariances.end(), MatrixAllocator(allocator))
        , m_centers(ScalarAllocator(allocator))
    {
        EMRYS_FILTER_ASSERT(
            (Shared || Eigen::Index(m_cellCovariances.size()) == size()) && "give one covariance per cell"
        );
        for (PinnedMatrix const & covariance : m_cellCovariances)
        {
            checkCovariance(covariance);
        }
    }

    /** Stores \a centers, one per cell, normalized by the block's space. */
    template <typename CentersDerived>
    void
    setCenters(Eigen::MatrixBase<CentersDerived> const & centers)
    {
        EMRYS_FILTER_ASSERT(centers.cols() == m_cells.cardinality() && "give one center per cell");
        EMRYS_FILTER_ASSERT(centers.rows() == pinnedRows().size && "the centers do not fit the block");
        m_centers.resize(std::size_t(centers.size()));
        auto stored = SampleSetMap<PinnedSpace, Scalar_>{m_centers.data(), centers.rows(), centers.cols()};
        stored = centers;
        normalizeValues(pinnedSpace(), stored);
    }

    /** Stores a regular grid's \a first center and \a spacing, one entry per axis. */
    void
    setGeometry(PinnedVector first, PinnedVector spacing)
    {
        EMRYS_FILTER_ASSERT(
            first.size() == pinnedRows().size && spacing.size() == pinnedRows().size
            && pinnedRows().size == Eigen::Index(Cells_::AxisCount)
            && "give the first point and spacing per axis, one axis per coordinate of the block"
        );
        m_first = std::move(first);
        m_spacing = std::move(spacing);
    }

    /** A programmer error unless \a covariance is square over the block's differences. */
    void
    checkCovariance(PinnedMatrix const & covariance) const
    {
        EMRYS_FILTER_ASSERT(
            covariance.rows() == pinnedDifferenceRows().size && covariance.cols() == pinnedDifferenceRows().size
            && "a cell covariance does not fit the block"
        );
    }

    /** Allocates the cell covariances. */
    using MatrixAllocator = internal::ReBound<Allocator_, PinnedMatrix>;

    /** Allocates the stored centers. */
    using ScalarAllocator = internal::ReBound<Allocator_, Scalar_>;

    Space_ m_space;                                                ///< the whole state space
    Cells_ m_cells;                                                ///< the discrete space of cells
    internal::BlockSplit<Space_, Path_> m_split;                   ///< the state divided into block and rest
    internal::Buffer<PinnedMatrix, Allocator_> m_cellCovariances;  ///< none, one shared, or one per cell
    internal::Buffer<Scalar_, Allocator_> m_centers;               ///< not regular: one per cell, column-major
    PinnedVector m_first;                                          ///< regular: the first cell's center
    PinnedVector m_spacing;                                        ///< regular: the spacing along each axis
};

/** \ingroup Distributions
 * \returns points for the block named by the path \a P of a state in \a space at \a centers,
 * one per column, with cells (<tt>PointCells<P></tt>) numbered in column order, and each cell's covariance about its
 * center \a cellCovariance (none by default), stored through \a allocator
 */
template <typename P, Space S, typename Derived, typename Allocator = DefaultAllocator>
[[nodiscard]] auto
pointGrid(
    S space,
    Eigen::MatrixBase<Derived> const & centers,
    std::optional<typename Moments<internal::PathTag<P>, typename Derived::Scalar>::MatrixType> cellCovariance = {},
    Allocator const & allocator = {}
)
{
    using Grid = PointGrid<S, P, PointCells<P>, SharedCovariance, typename Derived::Scalar, Allocator>;
    return Grid{std::move(space), PointCells<P>{centers.cols()}, centers, std::move(cellCovariance), allocator};
}

/** \ingroup Distributions
 * \returns points for the block named by the path \a P of a state in \a space at \a centers,
 * one per column, with cells (<tt>PointCells<P></tt>) numbered in column order, and \a cellCovariances, each cell's
 * covariance about its center, in the same order. The grid needs a PerPointCovariance filter.
 */
template <typename P, Space S, typename Derived, typename Allocator = DefaultAllocator>
[[nodiscard]] auto
pointGrid(
    S space,
    Eigen::MatrixBase<Derived> const & centers,
    std::span<typename Moments<internal::PathTag<P>, typename Derived::Scalar>::MatrixType const> cellCovariances,
    Allocator const & allocator = {}
)
{
    using Grid = PointGrid<S, P, PointCells<P>, PerPointCovariance, typename Derived::Scalar, Allocator>;
    return Grid{std::move(space), PointCells<P>{centers.cols()}, centers, cellCovariances, allocator};
}

/** \ingroup Distributions
 * pointGrid(space, centers, ...) in the FixedSizeSpace \a S, named as the first template argument:
 * <tt>pointGrid<State, Position>(centers)</tt> (decisions.md [space-storage]).
 */
template <FixedSizeSpace S, typename P, typename Derived, typename Allocator = DefaultAllocator>
[[nodiscard]] auto
pointGrid(
    Eigen::MatrixBase<Derived> const & centers,
    std::optional<typename Moments<internal::PathTag<P>, typename Derived::Scalar>::MatrixType> cellCovariance = {},
    Allocator const & allocator = {}
)
{
    return pointGrid<P>(S{}, centers, std::move(cellCovariance), allocator);
}

/** \ingroup Distributions
 * pointGrid(space, centers, cellCovariances, ...) in the FixedSizeSpace \a S, named as the first
 * template argument.
 */
template <FixedSizeSpace S, typename P, typename Derived, typename Allocator = DefaultAllocator>
[[nodiscard]] auto
pointGrid(
    Eigen::MatrixBase<Derived> const & centers,
    std::span<typename Moments<internal::PathTag<P>, typename Derived::Scalar>::MatrixType const> cellCovariances,
    Allocator const & allocator = {}
)
{
    return pointGrid<P>(S{}, centers, cellCovariances, allocator);
}

namespace internal
{

/** The cells of a regular grid with \a counts[i] cells along axis \a i. */
template <typename Cells, std::size_t D>
[[nodiscard]] Cells
regularCells(std::array<Eigen::Index, D> const & counts)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return Cells{std::tuple_element_t<I, DiscreteProductAxes<Cells>>{counts[I]}...};
    }(std::make_index_sequence<D>{});
}

}  // namespace internal

/** \ingroup Distributions
 * \returns points on a regular grid for the block named by the path \a P of a state in \a space:
 * \a counts[i] points along axis \a i, the first at \a first and the rest \a spacing apart, with
 * each cell's covariance about its center \a cellCovariance (none by default). The cells are
 * <tt>GridCells<P></tt>, so the probability of each axis is a marginal
 * (<tt>marginal<GridAxis<Position, 0>>()</tt>). A GridGaussian over the grid allocates through
 * \a allocator.
 */
template <typename P, Space S, typename FirstDerived, typename SpacingDerived, typename Allocator = DefaultAllocator>
[[nodiscard]] auto
regularGrid(
    S space,
    Eigen::MatrixBase<FirstDerived> const & first,
    Eigen::MatrixBase<SpacingDerived> const & spacing,
    std::array<Eigen::Index, std::size_t(internal::PathTag<P>::FixedValueDimension)> const & counts,
    std::optional<typename Moments<internal::PathTag<P>, typename FirstDerived::Scalar>::MatrixType> cellCovariance =
        {},
    Allocator const & allocator = {}
)
{
    static_assert(internal::PathTag<P>::ValueDimensionStep == 0, "a regular grid needs a fixed-size block");
    using Grid = PointGrid<S, P, GridCells<P>, SharedCovariance, typename FirstDerived::Scalar, Allocator>;
    return Grid{
        std::move(space),
        internal::regularCells<typename Grid::CellsType>(counts),
        typename Grid::PinnedVector{first},
        typename Grid::PinnedVector{spacing},
        std::move(cellCovariance),
        allocator
    };
}

/** \ingroup Distributions
 * regularGrid() with \a cellCovariances, each cell's covariance about its center in flat-index
 * order, as on a latitude-longitude grid whose cells narrow toward the poles. The grid needs a
 * PerPointCovariance filter, and can't be subdivided.
 */
template <typename P, Space S, typename FirstDerived, typename SpacingDerived, typename Allocator = DefaultAllocator>
[[nodiscard]] auto
regularGrid(
    S space,
    Eigen::MatrixBase<FirstDerived> const & first,
    Eigen::MatrixBase<SpacingDerived> const & spacing,
    std::array<Eigen::Index, std::size_t(internal::PathTag<P>::FixedValueDimension)> const & counts,
    std::span<typename Moments<internal::PathTag<P>, typename FirstDerived::Scalar>::MatrixType const> cellCovariances,
    Allocator const & allocator = {}
)
{
    static_assert(internal::PathTag<P>::ValueDimensionStep == 0, "a regular grid needs a fixed-size block");
    using Grid = PointGrid<S, P, GridCells<P>, PerPointCovariance, typename FirstDerived::Scalar, Allocator>;
    return Grid{
        std::move(space),
        internal::regularCells<typename Grid::CellsType>(counts),
        typename Grid::PinnedVector{first},
        typename Grid::PinnedVector{spacing},
        cellCovariances,
        allocator
    };
}

/** \ingroup Distributions
 * regularGrid(space, first, spacing, counts, ...) in the FixedSizeSpace \a S, named as the first
 * template argument: <tt>regularGrid<State, Position>(first, spacing, {200, 200})</tt>
 * (decisions.md [space-storage]).
 */
template <
    FixedSizeSpace S,
    typename P,
    typename FirstDerived,
    typename SpacingDerived,
    typename Allocator = DefaultAllocator>
[[nodiscard]] auto
regularGrid(
    Eigen::MatrixBase<FirstDerived> const & first,
    Eigen::MatrixBase<SpacingDerived> const & spacing,
    std::array<Eigen::Index, std::size_t(internal::PathTag<P>::FixedValueDimension)> const & counts,
    std::optional<typename Moments<internal::PathTag<P>, typename FirstDerived::Scalar>::MatrixType> cellCovariance =
        {},
    Allocator const & allocator = {}
)
{
    return regularGrid<P>(S{}, first, spacing, counts, std::move(cellCovariance), allocator);
}

/** \ingroup Distributions
 * regularGrid(space, first, spacing, counts, cellCovariances, ...) in the FixedSizeSpace \a S,
 * named as the first template argument.
 */
template <
    FixedSizeSpace S,
    typename P,
    typename FirstDerived,
    typename SpacingDerived,
    typename Allocator = DefaultAllocator>
[[nodiscard]] auto
regularGrid(
    Eigen::MatrixBase<FirstDerived> const & first,
    Eigen::MatrixBase<SpacingDerived> const & spacing,
    std::array<Eigen::Index, std::size_t(internal::PathTag<P>::FixedValueDimension)> const & counts,
    std::span<typename Moments<internal::PathTag<P>, typename FirstDerived::Scalar>::MatrixType const> cellCovariances,
    Allocator const & allocator = {}
)
{
    return regularGrid<P>(S{}, first, spacing, counts, cellCovariances, allocator);
}

namespace internal
{

/** The live cells of a GridGaussian, as the points of a Rao-Blackwellized state (see withPoint()). */
template <typename Grid>
class GridPoints
{
public:

    /** The scalar type. */
    using Scalar = typename Grid::Scalar;

    /** The whole state space. */
    using SpaceType = typename Grid::SpaceType;

    /** The path naming the fixed block. */
    using PathType = typename Grid::PathType;

    /** The fixed block's space. */
    using PinnedSpace = typename Grid::PinnedSpace;

    /** The rest's space. */
    using RestSpace = typename Grid::RestSpace;

    /** Whether a cell may have a covariance of its own, as kernel() gives it. */
    static constexpr bool HasKernels = true;

    /** Refers to the live \a cells, by flat index, of \a grid, which both outlive this. */
    GridPoints(Grid const & grid, std::span<Eigen::Index const> cells)
        : m_grid{grid}
        , m_cells{cells}
    {}

    /** \returns the whole state space */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return m_grid.space();
    }

    /** \returns the space of the fixed block */
    [[nodiscard]] PinnedSpace const &
    pinnedSpace() const
    {
        return m_grid.pinnedSpace();
    }

    /** \returns the space of the rest of the state, in order */
    [[nodiscard]] RestSpace const &
    restSpace() const
    {
        return m_grid.restSpace();
    }

    /** \returns the fixed block's rows in a whole state's values */
    [[nodiscard]] BlockRows
    pinnedRows() const
    {
        return m_grid.pinnedRows();
    }

    /** \returns the rest's rows in a whole state's values, in order */
    [[nodiscard]] auto const &
    restIndices() const
    {
        return m_grid.restIndices();
    }

    /** \returns the fixed block's rows in a whole state's differences */
    [[nodiscard]] BlockRows
    pinnedDifferenceRows() const
    {
        return m_grid.pinnedDifferenceRows();
    }

    /** \returns the rest's rows in a whole state's differences, in order */
    [[nodiscard]] auto const &
    restDifferenceIndices() const
    {
        return m_grid.restDifferenceIndices();
    }

    /** \returns the number of live cells */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return Eigen::Index(m_cells.size());
    }

    /** \returns the center of the live cell at position \a i */
    [[nodiscard]] auto
    center(Eigen::Index i) const
    {
        return m_grid.center(m_cells[std::size_t(i)]);
    }

    /** \returns the covariance about its center of the live cell at position \a i, or null if the
     * cells are point masses
     */
    [[nodiscard]] typename Grid::PinnedMatrix const *
    kernel(Eigen::Index i) const
    {
        return m_grid.hasCellCovariance() ? &m_grid.cellCovariance(m_cells[std::size_t(i)]) : nullptr;
    }

private:

    Grid const & m_grid;                    ///< the grid
    std::span<Eigen::Index const> m_cells;  ///< the live cells' flat indices
};

}  // namespace internal

/** \ingroup Distributions
 * \brief The parameters of a Rao-Blackwellized point-mass filter's state: the cells of a grid still alive,
 * their probabilities, and for each a %Gaussian, or a hybrid of Gaussians, over the rest of the
 * state.
 *
 * The density is \f$ p(x) = \sum_c P(c) \, \delta(x_p - p_c) \, \mathcal{N}(x_r; \mu_c, \Sigma_c) \f$
 * over the live cells \f$ c \f$, where \f$ p_c \f$ is the cell's center (widened by its
 * covariance about the center, if the grid has one). With \a Component_ a HybridGaussian over the
 * rest, each cell holds an IMM's state instead,
 * \f$ p(x) = \sum_c P(c) \, \delta(x_p - p_c) \sum_m P(m \mid c) \, \mathcal{N}(x_r; \mu_{cm}, \Sigma_{cm}) \f$,
 * whose modes() sums the cells' mode probabilities. Only cells of nonzero probability are
 * stored, in increasing order of their flat index, so a cell is addressed two ways: by its
 * position \a i among the live cells (conditional(i), center(i), logWeight(i)), and by its flat
 * index in the grid's cells (cell(i), find()). A cell whose probability becomes zero, by a
 * gate, by pruning, or by setWeights(), is removed for good, since nothing moves
 * probability between cells.
 *
 * With \a Covariance_ SharedCovariance, every cell has the same covariance; with
 * PerPointCovariance, each has its own (see RbPointMassFilter). A cell that holds a hybrid has
 * PerPointCovariance, since mixing its modes gives each cell covariances of its own.
 *
 * Upkeep (see \ref Upkeep) keeps the number of live cells in bounds: pruneBelow(), pruneTail(),
 * and pruneToSize() remove the improbable cells, and on a regular grid with at most one cell covariance, subdivide()
 * splits every live cell.
 *
 * The live cells, their probabilities, and their Gaussians come from the grid's allocator (see
 * \ref Allocation), and removing cells reuses that storage.
 *
 * \code
 * filter.state().size();                           // live cells
 * filter.state().marginal<GridAxis<Position, 0>>(); // CategoricalParams over the first axis
 * filter.state().toDensity()->collapsed();         // the whole state as one GaussianParams
 * upkeep::pruneBelow(1e-9)(filter.state());
 * upkeep::whenSize(std::less{}, 64, upkeep::subdivide(2))(filter.state());
 * \endcode
 *
 * \tparam Grid_        the grid (see PointGrid)
 * \tparam Covariance_  SharedCovariance or PerPointCovariance
 * \tparam Component_   each cell's distribution over the grid's RestSpace: a Gaussian, by default,
 *                     or a HybridGaussian
 */
template <
    typename Grid_,
    typename Covariance_ = SharedCovariance,
    typename Component_ = Gaussian<typename Grid_::RestSpace, typename Grid_::Scalar>>
class GridGaussianParams
{
    static_assert(internal::CovarianceForm<Covariance_>, "the covariance is SharedCovariance or PerPointCovariance");
    static constexpr bool Shared = std::is_same_v<Covariance_, SharedCovariance>;  ///< one covariance for all cells

    static_assert(
        !Shared || std::is_same_v<typename Grid_::CellCovarianceType, SharedCovariance>,
        "cells with covariances of their own give each cell its own innovation covariance, so the "
        "filter needs PerPointCovariance"
    );

    /** Whether each cell holds a hybrid of Gaussians rather than one %Gaussian. */
    static constexpr bool IsHybrid = internal::isGaussianHybrid<Params<Component_>>;

    static_assert(
        std::is_same_v<
            Params<Component_>,
            GaussianParams<typename Params<Component_>::SpaceType, typename Params<Component_>::Scalar>>
            || IsHybrid,
        "a cell holds a Gaussian or a hybrid of Gaussians"
    );
    static_assert(
        std::is_same_v<typename Params<Component_>::SpaceType, typename Grid_::RestSpace>
            && std::is_same_v<typename Params<Component_>::Scalar, typename Grid_::Scalar>,
        "a cell's distribution is over the grid's RestSpace, in the grid's scalar"
    );
    static_assert(
        !(IsHybrid && Shared),
        "mixing a cell's modes gives each cell covariances of its own, so a hybrid per cell needs PerPointCovariance"
    );

public:

    /** The grid. */
    using GridType = Grid_;

    /** The scalar type. */
    using Scalar = typename Grid_::Scalar;

    /** The whole state space. */
    using SpaceType = typename Grid_::SpaceType;

    /** The discrete space of cells. */
    using CellsType = typename Grid_::CellsType;

    /** The rest's space, in order. */
    using RestSpace = typename Grid_::RestSpace;

    /** The fixed block's space. */
    using PinnedSpace = typename Grid_::PinnedSpace;

    /** SharedCovariance or PerPointCovariance. */
    using CovarianceType = Covariance_;

    /** A cell's distribution over the rest: a %Gaussian, or a hybrid of Gaussians. */
    using ConditionalType = Params<Component_>;

    /** One entry per live cell. */
    using WeightArrayType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

    /** The live cells' log-probabilities. */
    using LogWeightsType = Eigen::Map<WeightArrayType const>;

    /** The grid's allocator. */
    using allocator_type = typename Grid_::allocator_type;

    /** The view toDensity() returns. */
    using DensityType = GridGaussian<Grid_, Covariance_, Component_>;

    static constexpr bool IsParams = true;  ///< parameters, which views check

    /** Every cell of \a grid alive and equally probable, each with \a conditional as its %Gaussian
     * over the rest of the state.
     */
    GridGaussianParams(Grid_ grid, ConditionalType const & conditional, Covariance_ = {})
        : m_grid{std::move(grid)}
        , m_cells(IndexAllocator(m_grid.get_allocator()))
        , m_logWeights(ScalarAllocator(m_grid.get_allocator()))
        , m_conditionals{conditional, m_grid.size(), m_grid.get_allocator()}
        , m_kept(IndexAllocator(m_grid.get_allocator()))
        , m_weightScratch(ScalarAllocator(m_grid.get_allocator()))
        , m_children(ChildAllocator(m_grid.get_allocator()))
    {
        EMRYS_FILTER_ASSERT(
            conditional.space() == m_grid.restSpace() && "the Gaussian must be over the rest of the state"
        );
        m_cells.resize(std::size_t(m_grid.size()));
        for (std::size_t c = 0; c < m_cells.size(); ++c)
        {
            m_cells[c] = Eigen::Index(c);
        }
        m_logWeights.assign(std::size_t(m_grid.size()), -Eigen::numext::log(Scalar(m_grid.size())));
        m_kept.reserve(m_cells.size());
        m_weightScratch.reserve(m_cells.size());
    }

    /** \returns the allocator the live cells come from: the grid's */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_grid.get_allocator();
    }

    /** \returns the grid, with its cells and centers */
    [[nodiscard]] Grid_ const &
    grid() const
    {
        return m_grid;
    }

    /** \returns the number of live cells */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return Eigen::Index(m_cells.size());
    }

    /** \returns the flat index in the grid's cells of the live cell at position \a i */
    [[nodiscard]] Eigen::Index
    cell(Eigen::Index i) const
    {
        assertPosition(i);
        return m_cells[std::size_t(i)];
    }

    /** \returns the flat index of every live cell, in increasing order */
    [[nodiscard]] std::span<Eigen::Index const>
    cells() const
    {
        return {m_cells.data(), m_cells.size()};
    }

    /** \returns the position among the live cells of the cell with flat index \a cell, or nothing if it is dead */
    [[nodiscard]] std::optional<Eigen::Index>
    find(Eigen::Index cell) const
    {
        auto const found = std::lower_bound(m_cells.begin(), m_cells.end(), cell);
        if (found == m_cells.end() || *found != cell)
        {
            return std::nullopt;
        }
        return Eigen::Index(found - m_cells.begin());
    }

    /** \returns the center of the live cell at position \a i */
    [[nodiscard]] auto
    center(Eigen::Index i) const
    {
        return m_grid.center(cell(i));
    }

    /** \returns each live cell's log-probability, normalized */
    [[nodiscard]] LogWeightsType
    logWeights() const
    {
        return {m_logWeights.data(), size()};
    }

    /** \returns the log-probability of the live cell at position \a i */
    [[nodiscard]] Scalar
    logWeight(Eigen::Index i) const
    {
        assertPosition(i);
        return m_logWeights[std::size_t(i)];
    }

    /** \returns the position of the most probable live cell; ties go to the lowest */
    [[nodiscard]] Eigen::Index
    mostProbable() const
    {
        Eigen::Index best = 0;
        logWeights().maxCoeff(&best);
        return best;
    }

    /** \returns the weights over every cell of the grid, dead ones zero, normalized */
    [[nodiscard]] CategoricalParams<CellsType, Scalar>
    toCategorical() const
    {
        using ArrayType = typename CategoricalParams<CellsType, Scalar>::ArrayType;
        ArrayType logP = ArrayType::Constant(m_grid.size(), -Eigen::NumTraits<Scalar>::infinity());
        for (Eigen::Index i = 0; i < size(); ++i)
        {
            logP(m_cells[std::size_t(i)]) = logWeight(i);
        }
        return CategoricalParams<CellsType, Scalar>{m_grid.cells(), logP};
    }

    /** \returns the distribution of the axis of the cells named by the path \a P (a tag or a
     * TagPath), such as an axis of a regular grid (<tt>marginal<GridAxis<Position, 0>>()</tt>), summed over
     * the live cells
     */
    template <typename P>
    [[nodiscard]] auto
    marginal() const
    {
        auto const [stride, axis] = internal::locateAxisByPath<P>(m_grid.cells());
        using Axis = std::remove_cvref_t<decltype(*axis)>;
        using ArrayType = typename CategoricalParams<Axis, Scalar>::ArrayType;
        Eigen::Index const count = axis->cardinality();
        // One pass over the cells per value, so the totals take only the axis's own storage.
        auto logP = ArrayType{};
        logP.resize(count);
        for (Eigen::Index v = 0; v < count; ++v)
        {
            auto total = LogSumExp<Scalar>{};
            for (Eigen::Index i = 0; i < size(); ++i)
            {
                if ((m_cells[std::size_t(i)] / stride) % count == v)
                {
                    total.add(logWeight(i));
                }
            }
            logP(v) = total.value();
        }
        return CategoricalParams<Axis, Scalar>{*axis, logP};
    }

    /** Replaces the live cells' weights with those of \a weights, normalized over the live cells,
     * as for a prior that isn't uniform. Cells it gives weight zero are removed; dead cells stay
     * dead.
     *
     * \returns nothing, or the WeightError of weights that give every live cell weight zero, or one
     *          a weight that is NaN or \f$ +\infty \f$, with the weights as they were, as a filter's
     *          state must always define a distribution (decisions.md [update-outcomes])
     */
    [[nodiscard]] std::expected<void, WeightError>
    setWeights(CategoricalParams<CellsType, Scalar> const & weights)
    {
        EMRYS_FILTER_ASSERT(weights.space() == m_grid.cells() && "the weights are over different cells");
        m_stamp.touchShape();
        m_weightScratch.resize(m_logWeights.size());
        for (Eigen::Index i = 0; i < size(); ++i)
        {
            m_weightScratch[std::size_t(i)] = weights.logWeight(m_cells[std::size_t(i)]);
        }
        auto const total =
            internal::logTotalWeight<Scalar>(std::span<Scalar const>{m_weightScratch.data(), m_weightScratch.size()});
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        m_logWeights.swap(m_weightScratch);
        normalizeAndRemoveDead();
        return {};
    }

    /** \returns the distribution over the rest of the state given the live cell at position \a i:
     * its %Gaussian, or its hybrid
     */
    [[nodiscard]] decltype(auto)
    conditional(Eigen::Index i) const
    {
        assertPosition(i);
        return m_conditionals.conditional(i);
    }

    /** \returns the probability of each mode, summed over the live cells,
     * \f$ P(m) = \sum_c P(c) \, P(m \mid c) \f$, normalized
     */
    [[nodiscard]] auto
    modes() const
        requires IsHybrid
    {
        return internal::summedModes(m_conditionals, logWeights());
    }

    /** \returns the covariance every cell shares */
    [[nodiscard]] typename Moments<RestSpace, Scalar>::MatrixType const &
    sharedCovariance() const
        requires Shared
    {
        return m_conditionals.sharedCovariance();
    }

    /** Factorizes each live cell's %Gaussian, the one they share once, or checks each cell's hybrid,
     * so that the density's collapse reads a distribution; the cells' weights always define one. It is
     * \c const because it writes only the cached factorizations (see \ref Distributions).
     *
     * \returns nothing, or the NumericError of the first cell that fails
     */
    [[nodiscard]] std::expected<void, NumericError>
    factorize() const
    {
        return m_conditionals.factorize();
    }

    /** Checks the cells (factorize()) and views the state.
     *
     * \returns a GridGaussian that checks this object, or the NumericError from factorize()
     */
    [[nodiscard]] std::expected<DensityType, NumericError>
    toDensity() const &
    {
        if (auto result = factorize(); !result)
        {
            return std::unexpected{result.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

    /** Removes every live cell whose log-probability is below \a logThreshold, and renormalizes;
     * the most probable cell is always kept, the first of them if several are. The threshold is
     * absolute: under subdivide(), which shares each cell's probability among its children, a
     * caller who wants the same density cut lowers it by the log of the children per cell.
     * upkeep::pruneBelow() calls it.
     *
     * \returns the log of the probability removed, -inf if nothing was
     *
     * \sa pruneTail(), pruneToSize()
     */
    [[nodiscard]] Scalar
    pruneBelow(Scalar logThreshold)
    {
        return pruneWith(internal::PruneBelowRule<Scalar>{logThreshold});
    }

    /** Removes the least probable live cells, as many as can be while their combined probability
     * stays below \f$ e^{\mathrm{logFraction}} \f$, and renormalizes. The fraction is below one, so
     * a cell is always kept. Of cells equally probable, those later in order go first. It bounds
     * the probability removed however many cells there are. upkeep::pruneTail() calls it.
     *
     * \returns the log of the probability removed, -inf if nothing was
     *
     * \sa pruneBelow(), pruneToSize()
     */
    [[nodiscard]] Scalar
    pruneTail(Scalar logFraction)
    {
        return pruneWith(internal::PruneTailRule<Scalar>{logFraction});
    }

    /** Removes the least probable live cells until at most \a maxSize, at least one, remain, and
     * renormalizes. Of cells equally probable, those later in order go first.
     * upkeep::pruneToSize() calls it.
     *
     * \returns the log of the probability removed, -inf if nothing was
     *
     * \sa pruneBelow(), pruneTail()
     */
    [[nodiscard]] Scalar
    pruneToSize(Eigen::Index maxSize)
    {
        return pruneWith(internal::PruneToSizeRule{maxSize});
    }

    /** Splits every live cell into \a factor cells along each axis (see PointGrid::subdivided()),
     * each with an equal share of its probability and a copy of its %Gaussian. The grid is
     * replaced by the subdivided one, so flat cell indices change.
     *
     * With the cell covariance \f$ \mathrm{diag}(s^2 / 12) \f$ of a cell of width \f$ s \f$, the
     * collapsed mean and covariance don't change. Without a cell covariance the cells are point
     * masses, and the children spread over their parent's cell: the mean doesn't change, and the
     * variance along each axis grows by \f$ \tfrac{s^2}{12} (1 - 1 / k^2) \f$, \f$ k \f$ the factor.
     * For a grid that samples a density at its centers, this brings it closer to that density;
     * where the points mean exactly what they say, don't subdivide.
     */
    void
    subdivide(Eigen::Index factor)
        requires(Grid_::Regular && std::is_same_v<typename Grid_::CellCovarianceType, SharedCovariance>)
    {
        constexpr std::size_t D = CellsType::AxisCount;
        EMRYS_FILTER_ASSERT(factor >= 1 && "subdivide each cell into at least one");
        m_stamp.touchShape();
        Grid_ refined = m_grid.subdivided(factor);
        Eigen::Index offsets = 1;
        for (std::size_t d = 0; d < D; ++d)
        {
            offsets *= factor;
        }
        auto & children = m_children;
        children.clear();
        children.reserve(m_cells.size() * std::size_t(offsets));
        for (Eigen::Index i = 0; i < size(); ++i)
        {
            auto const parent = m_grid.cells().coordinates(m_cells[std::size_t(i)]);
            for (Eigen::Index o = 0; o < offsets; ++o)
            {
                auto child = std::array<Eigen::Index, D>{};
                Eigen::Index rest = o;
                for (std::size_t d = D; d-- > 0;)
                {
                    child[d] = factor * parent[d] + rest % factor;
                    rest /= factor;
                }
                children.emplace_back(refined.cells().value(child), i);
            }
        }
        std::sort(children.begin(), children.end());
        m_kept.clear();
        m_weightScratch.clear();
        Scalar const share = Eigen::numext::log(Scalar(offsets));
        for (std::size_t k = 0; k < children.size(); ++k)
        {
            m_kept.push_back(children[k].second);
            m_weightScratch.push_back(m_logWeights[std::size_t(children[k].second)] - share);
        }
        m_conditionals.select(m_kept);
        m_grid = std::move(refined);
        m_cells.resize(children.size());
        for (std::size_t k = 0; k < children.size(); ++k)
        {
            m_cells[k] = children[k].first;
        }
        m_logWeights.swap(m_weightScratch);
    }

private:

    /** The filter reweights and replaces the Gaussians. */
    template <typename, typename, typename, typename, typename>
    friend class RbPointMassFilter;

    friend DensityType;  ///< the density reads the collapse

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** The cells' distributions' store. */
    using ConditionalsType = internal::Conditionals<RestSpace, Covariance_, Scalar, allocator_type, ConditionalType>;

    /** Allocates the cell indices. */
    using IndexAllocator = internal::ReBound<allocator_type, Eigen::Index>;

    /** Allocates the probabilities. */
    using ScalarAllocator = internal::ReBound<allocator_type, Scalar>;

    /** \returns the live cells as the points of a Rao-Blackwellized state, for collapseGivenPoints() */
    [[nodiscard]] internal::GridPoints<Grid_>
    points() const
    {
        return {m_grid, cells()};
    }

    /** \returns the whole state as one %Gaussian, moment-matched over the live cells: the centers'
     * spread and each cell's covariance about its center for the fixed block, and the cells'
     * Gaussians, each hybrid collapsed over its modes first, for the rest. The cells must have been
     * checked, as a density's check makes sure.
     */
    [[nodiscard]] GaussianParams<SpaceType, Scalar>
    collapseCells() const
    {
        if constexpr (IsHybrid)
        {
            return internal::collapseGivenHybrids(points(), m_conditionals, logWeights(), get_allocator());
        }
        else
        {
            return internal::collapseGivenPoints(points(), m_conditionals, logWeights());
        }
    }

    /** A programmer error unless \a i is a position among the live cells. */
    void
    assertPosition([[maybe_unused]] Eigen::Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < size() && "the position is out of range");
    }

    /** Multiplies the probabilities by \a logLikelihoods, one per live cell, and swaps \a posterior
     * in as the cells' Gaussians, then removes the cells left with probability zero. If every cell
     * has likelihood zero, nothing changes. The evidence is a reduction under \a policy; the
     * removal is sequential. Under Sequential it allocates nothing once the cells have been as many.
     *
     * \returns the log-evidence, \f$ \log \sum_c P(c) \, p(z \mid c) \f$, and whether anything changed
     */
    template <ExecutionPolicy P, typename Derived>
    [[nodiscard]] std::pair<Scalar, bool>
    reweight(P && policy, Eigen::ArrayBase<Derived> const & logLikelihoods, ConditionalsType & posterior)
    {
        return reweightThen(std::forward<P>(policy), logLikelihoods, [&] {
            m_conditionals.swap(posterior);
        });
    }

    /** Multiplies the probabilities by \a logLikelihoods, as reweight(policy, logLikelihoods,
     * posterior) does, and keeps the cells' Gaussians where they are: for an update that changed
     * none, or wrote them in place (decisions.md [update-outcomes]).
     *
     * \returns the log-evidence and whether anything changed
     */
    template <ExecutionPolicy P, typename Derived>
    [[nodiscard]] std::pair<Scalar, bool>
    reweight(P && policy, Eigen::ArrayBase<Derived> const & logLikelihoods)
    {
        return reweightThen(std::forward<P>(policy), logLikelihoods, [] {});
    }

    /** The body of both forms of reweight(): \a place puts the posterior Gaussians in, once the
     * evidence is known to be finite and before the cells of probability zero are removed.
     */
    template <ExecutionPolicy P, typename Derived, typename Place>
    [[nodiscard]] std::pair<Scalar, bool>
    reweightThen(P && policy, Eigen::ArrayBase<Derived> const & logLikelihoods, Place const & place)
    {
        EMRYS_FILTER_ASSERT(logLikelihoods.size() == size() && "give one log-likelihood per live cell");
        m_weightScratch.resize(m_logWeights.size());
        internal::forEachIndex(policy, size(), [&](Eigen::Index i) {
            m_weightScratch[std::size_t(i)] = m_logWeights[std::size_t(i)] + logLikelihoods(i);
        });
        auto const evidence = internal::logTotalWeight<Scalar>(
            policy,
            std::span<Scalar const>{m_weightScratch.data(), m_weightScratch.size()}
        );
        if (!evidence)
        {
            return {
                logSumExp<Scalar>(
                    std::forward<P>(policy),
                    size(),
                    [&](Eigen::Index i) {
                        return m_weightScratch[std::size_t(i)];
                    }
                ),
                false
            };
        }
        m_stamp.touchShape();
        m_logWeights.swap(m_weightScratch);
        place();
        normalizeAndRemoveDead();
        return {*evidence, true};
    }

    /** The body of pruneBelow(), pruneTail(), and pruneToSize(): gives the cells that \a rule's cut
     * removes probability zero, ordering the probabilities in the scratch where it needs to, and
     * removes them. The probabilities are normalized, so their log total is zero.
     *
     * \returns the log of the probability removed
     */
    template <typename Rule>
    [[nodiscard]] Scalar
    pruneWith(Rule const & rule)
    {
        m_stamp.touchShape();
        auto cut = rule(
            size(),
            [&](Eigen::Index i) {
                return m_logWeights[std::size_t(i)];
            },
            Scalar(0),
            m_weightScratch
        );
        auto removed = LogSumExp<Scalar>{};
        for (Scalar & logWeight : m_logWeights)
        {
            if (cut.removes(logWeight))
            {
                removed.add(logWeight);
                logWeight = -Eigen::NumTraits<Scalar>::infinity();
            }
        }
        normalizeAndRemoveDead();
        return removed.value();
    }

    /** Normalizes the probabilities and removes the cells of probability zero, in place. Some live
     * cell must keep a nonzero weight, as every caller ensures: setWeights() by checking first,
     * the prunes by keeping a cell, and reweight() by a finite evidence.
     */
    void
    normalizeAndRemoveDead()
    {
        auto const total = internal::logTotalWeight<Scalar>(m_logWeights);
        EMRYS_FILTER_ASSERT(total && "some live cell must keep a nonzero weight");
        for (Scalar & logWeight : m_logWeights)
        {
            logWeight -= *total;
        }
        m_kept.clear();
        for (Eigen::Index i = 0; i < size(); ++i)
        {
            if (m_logWeights[std::size_t(i)] > -Eigen::NumTraits<Scalar>::infinity())
            {
                m_kept.push_back(i);
            }
        }
        if (m_kept.size() == m_cells.size())
        {
            return;
        }
        for (std::size_t k = 0; k < m_kept.size(); ++k)
        {
            m_cells[k] = m_cells[std::size_t(m_kept[k])];
            m_logWeights[k] = m_logWeights[std::size_t(m_kept[k])];
        }
        m_conditionals.keep(m_kept);
        m_cells.resize(m_kept.size());
        m_logWeights.resize(m_kept.size());
    }

    Grid_ m_grid;                                            ///< the grid
    internal::Buffer<Eigen::Index, allocator_type> m_cells;  ///< flat indices of the live cells, increasing
    internal::Buffer<Scalar, allocator_type> m_logWeights;   ///< one per live cell, normalized
    ConditionalsType m_conditionals;                         ///< one %Gaussian per live cell

    // Scratch, reused: the cells kept, and the reweighted log-probabilities.
    internal::Buffer<Eigen::Index, allocator_type> m_kept;  ///< positions of the cells kept by the last pass

    internal::Buffer<Scalar, allocator_type> m_weightScratch;  ///< log-probabilities being rebuilt

    /** A cell subdivide() makes: its flat index in the refined grid and its parent's position. */
    using Child = std::pair<Eigen::Index, Eigen::Index>;

    /** The allocator of the children. */
    using ChildAllocator = internal::ReBound<allocator_type, Child>;

    internal::Buffer<Child, allocator_type> m_children;  ///< the cells the last subdivide() made, sorted

    internal::Stamp m_stamp;  ///< counts changes, for the density's check [view-stamp]
};

/** \ingroup Distributions
 * \brief The density of a Rao-Blackwellized point-mass filter's state, a checked view of its
 * GridGaussianParams (decisions.md [params-and-density], [view-stamp]):
 * \f$ p(x) = \sum_c P(c) \, \delta(x_p - p_c) \, \mathcal{N}(x_r; \mu_c, \Sigma_c) \f$ over the live
 * cells, or the hybrid of Gaussians in each cell.
 *
 * GridGaussianParams::toDensity() makes one after factorizing each cell's %Gaussian, or checking
 * each cell's hybrid, so the moments and the collapse below read a distribution.
 *
 * \code
 * auto const density = filter.state().toDensity();
 * density->collapsed();   // the whole state as one GaussianParams
 * density->moments();     // its mean and covariance
 * \endcode
 *
 * \sa class GridGaussianParams, RbPointMassFilter
 */
template <typename Grid_, typename Covariance_, typename Component_>
class GridGaussian
{
public:

    using ParamsType = GridGaussianParams<Grid_, Covariance_, Component_>;  ///< the parameters it reads
    using GridType = Grid_;                                                 ///< the grid
    using Scalar = typename Grid_::Scalar;                                  ///< the scalar type
    using SpaceType = typename Grid_::SpaceType;                            ///< the whole state space
    using ConditionalType = typename ParamsType::ConditionalType;           ///< a cell's distribution over the rest
    using MomentsType = Moments<SpaceType, Scalar>;              ///< a mean and covariance of the whole state
    using allocator_type = typename ParamsType::allocator_type;  ///< the parameters' allocator

    static constexpr bool IsParams = false;  ///< a view: it checks its parameters

    /** \returns the grid, with its cells and centers */
    [[nodiscard]] Grid_ const &
    grid() const
    {
        return source().grid();
    }

    /** \returns the number of live cells */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return source().size();
    }

    /** \returns the distribution over the rest of the state given the live cell at position \a i */
    [[nodiscard]] decltype(auto)
    conditional(Eigen::Index i) const
    {
        return source().conditional(i);
    }

    /** \returns the whole state as one %Gaussian, moment-matched over the live cells: the centers'
     * spread and each cell's covariance about its center for the fixed block, and the cells'
     * Gaussians, or each cell's hybrid collapsed over its modes, for the rest
     */
    [[nodiscard]] GaussianParams<SpaceType, Scalar>
    collapsed() const
    {
        return source().collapseCells();
    }

    /** \returns the mean and covariance of the whole state, those of collapsed() */
    [[nodiscard]] MomentsType
    moments() const
    {
        return collapsed().moments();
    }

private:

    friend ParamsType;  ///< the parameters construct views

    /** Views \a source, whose cells are checked, and records its stamp; only
     * GridGaussianParams::toDensity() calls this.
     */
    explicit GridGaussian(ParamsType const & source)
        : m_check{source}
    {}

    /** \returns the source; a programmer error if it has changed since toDensity() */
    [[nodiscard]] ParamsType const &
    source() const
    {
        return m_check.get();
    }

    internal::ValuesCheck<ParamsType> m_check;  ///< the check of the viewed parameters
};

/// \cond DEDUCTION_GUIDES
// A hybrid conditional names the component, and its cells have covariances of their own.
template <typename Grid_, typename Component_, typename ModeSpace_, typename Allocator_>
GridGaussianParams(Grid_, HybridParams<Component_, ModeSpace_, Allocator_>, PerPointCovariance = {})
    -> GridGaussianParams<Grid_, PerPointCovariance, Hybrid<Component_, ModeSpace_, Allocator_>>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a state, a Rao-Blackwellized point-mass filter's: its cell count, and the dimension,
 * mean, and covariance of its collapse; or the error of a cell that fails to factorize in their place.
 */
template <typename Grid_, typename Covariance_, typename Component_>
void
describe(std::ostream & stream, GridGaussianParams<Grid_, Covariance_, Component_> const & state, Indent indent)
{
    stream << indent << "Rao-Blackwellized grid of ";
    printCount(stream, state.size(), "cell");
    // Every step leaves the cells' weights normalized, so only a cell's distribution can fail.
    auto const density = state.toDensity();
    if (!density)
    {
        stream << '\n';
        describe(stream, density.error(), indent.deeper());
        return;
    }
    auto const collapsed = density->collapsed();
    printDimension(stream, collapsed.valueDimension(), collapsed.differenceDimension());
    printMoments(stream, indent.deeper(), collapsed.moments());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
