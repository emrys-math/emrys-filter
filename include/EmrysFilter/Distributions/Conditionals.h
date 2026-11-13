#pragma once

/** \file
 * What the Rao-Blackwellized states share (decisions.md [rao-blackwellized-core], [rb-particle-filter]): the
 * SharedCovariance and PerPointCovariance forms, BlockSplit, which divides a state into the block fixed at points and
 * the %Gaussian rest, Conditionals, the Gaussians over the rest stored in either form, the
 * collapse of points and conditionals into one %Gaussian, and the conditioning of the rest on a
 * drawn block. GridGaussianParams and ParticleGaussianParams are built on it; the update over points is
 * Filters/RbUpdate.h. Read GridGaussian.h next.
 */

#include <cmath>
#include <cstddef>
#include <expected>
#include <numbers>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Distributions
 * One covariance for every point. In a Rao-Blackwellized filter, every point's %Gaussian over the
 * rest of the state shares a covariance: for affine dynamics and measurements whose Jacobian in
 * the rest of the state is the same at every point. In PointGrid, every cell has the same
 * covariance about its center, or none.
 */
struct SharedCovariance
{};

/** \ingroup Distributions
 * A covariance per point. In a Rao-Blackwellized filter, each point's %Gaussian over the rest of
 * the state has its own covariance: for measurements whose Jacobian in the rest of the state
 * depends on the point, such as one nonlinear in it. In PointGrid, each cell has its own
 * covariance about its center, as cells of different sizes do; such a grid needs a
 * PerPointCovariance filter.
 */
struct PerPointCovariance
{};

namespace internal
{

/** SharedCovariance or PerPointCovariance. */
template <typename C>
concept CovarianceForm = std::is_same_v<C, SharedCovariance> || std::is_same_v<C, PerPointCovariance>;

/** Whether \a C, a point's distribution over the rest of a Rao-Blackwellized state, is a hybrid of
 * Gaussians rather than one %Gaussian.
 */
template <typename C>
inline constexpr bool isGaussianHybrid = false;

/** A HybridParams is a point's hybrid of Gaussians when its component is a covariance %Gaussian. */
template <typename Component_, typename ModeSpace_, typename Allocator_>
inline constexpr bool isGaussianHybrid<HybridParams<Component_, ModeSpace_, Allocator_>> = std::is_same_v<
    Params<Component_>,
    GaussianParams<typename Params<Component_>::SpaceType, typename Params<Component_>::Scalar>>;

/** The space of \a Space outside the block named by the bare tag \a P. */
template <typename Space, typename P>
struct ComplementOf
{
    /** The complement's space. */
    using type = decltype(complementSubspace<P>(std::declval<Space const &>()));
};

/** The space of \a Space outside the block named by a TagPath. */
template <typename Space, typename... Tags>
struct ComplementOf<Space, TagPath<Tags...>>
{
    /** The complement's space. */
    using type = decltype(complementSubspace<Tags...>(std::declval<Space const &>()));
};

/** \returns the space of \a space outside the block named by the path \a P, with its sizes */
template <typename P, typename Space>
[[nodiscard]] auto
complementOf(Space const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        return complementSubspace<Tags...>(space);
    });
}

/** A state in \a Space_ split into the block named by the path \a Path_ (a tag or a TagPath),
 * which a Rao-Blackwellized filter fixes at points, and the rest, which it keeps %Gaussian.
 */
template <typename Space_, typename Path_>
class BlockSplit
{
public:

    /** The path naming the fixed block. */
    using PathType = Path_;

    /** The space of the fixed block. */
    using PinnedSpace = PathTag<Path_>;

    /** The space of the rest of the state, in order. */
    using RestSpace = typename ComplementOf<Space_, Path_>::type;

    /** The rest's rows in a whole state's values. */
    using RestIndices = IndexVector<valueDimensionAtCompileTime<RestSpace>, maxValueDimensionAtCompileTime<RestSpace>>;

    /** The rest's rows in a whole state's differences. */
    using RestDifferenceIndices = IndexVector<
        valueDimensionAtCompileTime<DifferenceSpace<RestSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<RestSpace>>>;

    /** Splits \a space, resolving the block's and the rest's rows from its sizes. */
    explicit BlockSplit(Space_ const & space)
        : m_pinned{pathBlock<Path_>(space)}
        , m_rest{complementOf<Path_>(space)}
        , m_rows{blockRows<Path_>(space)}
        , m_differenceRows{differenceBlockRows<Path_>(space)}
        , m_size{valueDimension(space)}
        , m_restIndices{outside<RestIndices>(m_rows, m_size)}
        , m_restDifferenceIndices{outside<RestDifferenceIndices>(m_differenceRows, differenceDimension(space))}
    {}

    /** \returns the space of the fixed block, with its parameters */
    [[nodiscard]] PinnedSpace const &
    pinnedSpace() const
    {
        return m_pinned;
    }

    /** \returns the space of the rest of the state, in order */
    [[nodiscard]] RestSpace const &
    restSpace() const
    {
        return m_rest;
    }

    /** \returns the fixed block's rows in a whole state's values */
    [[nodiscard]] BlockRows
    pinnedRows() const
    {
        return m_rows;
    }

    /** \returns the rest's rows in a whole state's values, in order */
    [[nodiscard]] RestIndices const &
    restIndices() const
    {
        return m_restIndices;
    }

    /** \returns the fixed block's rows in a whole state's differences, such as a covariance */
    [[nodiscard]] BlockRows
    pinnedDifferenceRows() const
    {
        return m_differenceRows;
    }

    /** \returns the rest's rows in a whole state's differences, in order */
    [[nodiscard]] RestDifferenceIndices const &
    restDifferenceIndices() const
    {
        return m_restDifferenceIndices;
    }

    /** \returns the whole state with the rest \a rest and the fixed block \a pinned, in the scalar
     * type of \a rest
     */
    template <typename X, typename P>
    [[nodiscard]] auto
    assemble(X const & rest, P const & pinned) const
    {
        using T = ScalarOf<X>;
        EMRYS_FILTER_ASSERT(
            rest.size() == m_restIndices.size() && pinned.size() == m_rows.size && "the blocks do not fit the space"
        );
        Eigen::Matrix<
            T,
            valueDimensionAtCompileTime<Space_>,
            1,
            Eigen::ColMajor,
            maxValueDimensionAtCompileTime<Space_>,
            1>
            full;
        full.resize(m_size);
        full(m_restIndices) = rest;
        middleRowsOf<valueDimensionAtCompileTime<PinnedSpace>>(full, m_rows.start, m_rows.size) =
            pinned.template cast<T>();
        return full;
    }

private:

    /** \returns the rows of \c 0..n-1 outside \a rows, in order */
    template <typename Indices>
    [[nodiscard]] static Indices
    outside(BlockRows rows, Eigen::Index n)
    {
        auto result = Indices{};
        result.resize(n - rows.size);
        Eigen::Index k = 0;
        for (Eigen::Index i = 0; i < n; ++i)
        {
            if (i < rows.start || i >= rows.start + rows.size)
            {
                result(k++) = i;
            }
        }
        return result;
    }

    PinnedSpace m_pinned;                           ///< the fixed block's space
    RestSpace m_rest;                               ///< the rest's space
    BlockRows m_rows;                               ///< the fixed block's rows in values
    BlockRows m_differenceRows;                     ///< the fixed block's rows in differences
    Eigen::Index m_size;                            ///< the whole state's value dimension
    RestIndices m_restIndices;                      ///< the rest's rows in values
    RestDifferenceIndices m_restDifferenceIndices;  ///< the rest's rows in differences
};

/** The distributions over the rest of a state, one per point of a Rao-Blackwellized filter, stored
 * as \a Covariance says: one mean per point and one covariance for all of them, or one \a Element
 * per point, a %Gaussian by default or a hybrid of Gaussians. The storage comes from \a Allocator,
 * and select() and keep() reuse it, so they allocate nothing once the points have been as many.
 */
template <
    typename Space,
    typename Covariance,
    typename Scalar,
    typename Allocator = DefaultAllocator,
    typename Element = GaussianParams<Space, Scalar>>
class Conditionals;

/** The shared form: one mean per point in a matrix, and one covariance. */
template <typename Space, typename Scalar_, typename Allocator>
class Conditionals<Space, SharedCovariance, Scalar_, Allocator, GaussianParams<Space, Scalar_>>
{
public:

    /** The scalar type. */
    using Scalar = Scalar_;

    /** A point's %Gaussian over the rest. */
    using ConditionalType = GaussianParams<Space, Scalar>;

    /** A covariance over the rest's differences. */
    using MatrixType = typename ConditionalType::MatrixType;

    /** A plain matrix of means, one per column, laid out as means() is. */
    using MeansType = SampleSet<Space, Scalar>;

    static constexpr bool Shared = true;  ///< the points share one covariance

    /** \a count points, each with \a conditional. */
    Conditionals(ConditionalType const & conditional, Eigen::Index count, Allocator const & allocator = {})
        : m_space{conditional.space()}
        , m_rows{conditional.valueDimension()}
        , m_count{count}
        , m_means(ScalarAllocator(allocator))
        , m_scratch(ScalarAllocator(allocator))
        , m_cov{conditional.covarianceLower()}
    {
        m_means.resize(std::size_t(m_rows * m_count));
        means().colwise() = conditional.mean();
    }

    /** Copies \a other's points into storage from \a allocator. */
    Conditionals(Conditionals const & other, Allocator const & allocator)
        : m_space{other.m_space}
        , m_rows{other.m_rows}
        , m_count{other.m_count}
        , m_means(other.m_means, ScalarAllocator(allocator))
        , m_scratch(ScalarAllocator(allocator))
        , m_cov{other.m_cov}
    {}

    /** Copies \a other's points into storage from its allocator (see \ref Allocation). */
    Conditionals(Conditionals const & other)
        : Conditionals{other, other.m_means.get_allocator()}
    {}

    Conditionals(Conditionals &&) noexcept = default;  ///< takes \a other's storage

    /** Copies \a other's points into this storage, which keeps its allocator. */
    Conditionals &
    operator=(Conditionals const & other)
    {
        if (this == &other)
        {
            return *this;
        }
        m_space = other.m_space;
        m_rows = other.m_rows;
        m_count = other.m_count;
        m_means.assign(other.m_means.begin(), other.m_means.end());
        m_cov = other.m_cov;
        return *this;
    }

    /** Takes \a other's storage. */
    Conditionals &
    operator=(Conditionals &&) noexcept = default;

    ~Conditionals() = default;

    /** Gives this store \a other's space and number of points, with every mean left to be written,
     * so that a caller that writes each point once copies nothing (see \ref Allocation). The
     * shared covariance is left as it was.
     */
    void
    resizeLike(Conditionals const & other)
    {
        m_space = other.m_space;
        m_rows = other.m_rows;
        m_count = other.m_count;
        m_means.resize(std::size_t(m_rows * m_count));
    }

    /** Exchanges the points with \a other, which must use an equal allocator. */
    void
    swap(Conditionals & other) noexcept
    {
        std::swap(m_space, other.m_space);
        std::swap(m_rows, other.m_rows);
        std::swap(m_count, other.m_count);
        m_means.swap(other.m_means);
        m_scratch.swap(other.m_scratch);
        std::swap(m_cov, other.m_cov);
    }

    /** \returns the number of points */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return m_count;
    }

    /** \returns the space of the rest of the state */
    [[nodiscard]] Space const &
    space() const
    {
        return m_space;
    }

    /** \returns point \a i's mean, a column of means() */
    [[nodiscard]] auto
    mean(Eigen::Index i) const
    {
        return means().col(i);
    }

    /** \returns the shared covariance, whatever the point */
    [[nodiscard]] MatrixType const &
    covarianceLower(Eigen::Index /*i*/) const
    {
        return m_cov;
    }

    /** \returns point \a i's %Gaussian, assembled from its mean and the shared covariance */
    [[nodiscard]] ConditionalType
    conditional(Eigen::Index i) const
    {
        return ConditionalType{m_space, mean(i), m_cov};
    }

    /** Checks the shared covariance by factorizing it once, for a density's check of its state; a
     * rest of no coordinates, a point-mass filter's, has nothing to check.
     *
     * \returns nothing, or the NumericError of a covariance that fails to factorize
     */
    [[nodiscard]] std::expected<void, NumericError>
    factorize() const
    {
        if constexpr (MatrixType::RowsAtCompileTime == 0)
        {
            return {};
        }
        else
        {
            ConditionalType const first = conditional(0);
            return first.factorize();
        }
    }

    /** \returns every point's mean, one per column */
    [[nodiscard]] ConstSampleSetMap<Space, Scalar>
    means() const
    {
        return {m_means.data(), m_rows, m_count};
    }

    /** \returns every point's mean, one per column, to change them */
    [[nodiscard]] SampleSetMap<Space, Scalar>
    means()
    {
        return {m_means.data(), m_rows, m_count};
    }

    /** \returns the shared covariance */
    [[nodiscard]] MatrixType const &
    sharedCovariance() const
    {
        return m_cov;
    }

    /** \returns the shared covariance, to change it */
    [[nodiscard]] MatrixType &
    sharedCovariance()
    {
        return m_cov;
    }

    /** Reserves room for \a count points. */
    void
    reserve(Eigen::Index count)
    {
        m_means.reserve(std::size_t(m_rows * count));
        m_scratch.reserve(std::size_t(m_rows * count));
    }

    /** Makes point \a j a copy of point \a from[j], for every \a j. */
    void
    select(std::span<Eigen::Index const> from)
    {
        select(execution::sequential, from);
    }

    /** select() under \a policy, a gather of the means. */
    template <ExecutionPolicy P>
    void
    select(P && policy, std::span<Eigen::Index const> from)
    {
        Eigen::Index const count = Eigen::Index(from.size());
        m_scratch.resize(std::size_t(m_rows * count));
        gather(std::forward<P>(policy), from, means(), SampleSetMap<Space, Scalar>{m_scratch.data(), m_rows, count});
        m_means.swap(m_scratch);
        m_count = count;
    }

    /** Keeps only the points \a kept, in increasing order, in place. */
    void
    keep(std::span<Eigen::Index const> kept)
    {
        auto all = means();
        for (std::size_t k = 0; k < kept.size(); ++k)
        {
            if (kept[k] != Eigen::Index(k))
            {
                all.col(Eigen::Index(k)) = all.col(kept[k]);
            }
        }
        m_count = Eigen::Index(kept.size());
        m_means.resize(std::size_t(m_rows * m_count));
    }

private:

    /** Allocates the means and scratch. */
    using ScalarAllocator = ReBound<Allocator, Scalar>;

    Space m_space;                        ///< the rest's space
    Eigen::Index m_rows;                  ///< coordinates per mean
    Eigen::Index m_count;                 ///< the number of points
    Buffer<Scalar, Allocator> m_means;    ///< valueDimension() by size(), column-major
    Buffer<Scalar, Allocator> m_scratch;  ///< means being rebuilt by select(), swapped in when done
    MatrixType m_cov;                     ///< the shared covariance
};

/** The per-point form: one \a Element per point, a %Gaussian or a hybrid of Gaussians. */
template <typename Space, typename Scalar_, typename Allocator, typename Element>
class Conditionals<Space, PerPointCovariance, Scalar_, Allocator, Element>
{
    /** Whether each point holds one %Gaussian, whose mean and covariance the collapse reads. */
    static constexpr bool IsGaussian = std::is_same_v<Element, GaussianParams<Space, Scalar_>>;

public:

    /** The scalar type. */
    using Scalar = Scalar_;

    /** A point's distribution over the rest. */
    using ConditionalType = Element;

    /** A covariance over the rest's differences. */
    using MatrixType = typename Moments<Space, Scalar>::MatrixType;

    static constexpr bool Shared = false;  ///< each point has its own covariance

    /** \a count points, each with \a conditional. */
    Conditionals(ConditionalType const & conditional, Eigen::Index count, Allocator const & allocator = {})
        : m_conditionals(std::size_t(count), conditional, GaussianAllocator(allocator))
        , m_scratch(GaussianAllocator(allocator))
    {}

    /** Copies \a other's points into storage from \a allocator. */
    Conditionals(Conditionals const & other, Allocator const & allocator)
        : m_conditionals(other.m_conditionals, GaussianAllocator(allocator))
        , m_scratch(GaussianAllocator(allocator))
    {}

    /** Copies \a other's points into storage from its allocator (see \ref Allocation). */
    Conditionals(Conditionals const & other)
        : Conditionals{other, other.m_conditionals.get_allocator()}
    {}

    Conditionals(Conditionals &&) noexcept = default;  ///< takes \a other's storage

    /** Copies \a other's points into this storage, which keeps its allocator. */
    Conditionals &
    operator=(Conditionals const & other)
    {
        if (this != &other)
        {
            m_conditionals.assign(other.m_conditionals.begin(), other.m_conditionals.end());
        }
        return *this;
    }

    /** Takes \a other's storage. */
    Conditionals &
    operator=(Conditionals &&) noexcept = default;

    ~Conditionals() = default;

    /** Gives this store \a other's number of points, each %Gaussian left to be written, so that a
     * caller that writes each point once copies nothing (see \ref Allocation).
     */
    void
    resizeLike(Conditionals const & other)
        requires IsGaussian
    {
        m_conditionals.resize(other.m_conditionals.size());
    }

    /** Exchanges the points with \a other, which must use an equal allocator. */
    void
    swap(Conditionals & other) noexcept
    {
        m_conditionals.swap(other.m_conditionals);
        m_scratch.swap(other.m_scratch);
    }

    /** \returns the number of points */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return Eigen::Index(m_conditionals.size());
    }

    /** \returns the space of the rest of the state, from the first point */
    [[nodiscard]] Space const &
    space() const
    {
        return m_conditionals.front().space();
    }

    /** \returns point \a i's mean */
    [[nodiscard]] decltype(auto)
    mean(Eigen::Index i) const
        requires IsGaussian
    {
        return m_conditionals[std::size_t(i)].mean();
    }

    /** \returns point \a i's covariance */
    [[nodiscard]] MatrixType const &
    covarianceLower(Eigen::Index i) const
        requires IsGaussian
    {
        return m_conditionals[std::size_t(i)].covarianceLower();
    }

    /** \returns point \a i's distribution */
    [[nodiscard]] ConditionalType const &
    conditional(Eigen::Index i) const
    {
        return m_conditionals[std::size_t(i)];
    }

    /** \returns point \a i's distribution, to change it */
    [[nodiscard]] ConditionalType &
    conditional(Eigen::Index i)
    {
        return m_conditionals[std::size_t(i)];
    }

    /** Factorizes each point's distribution that has changed, for a density's check of its state:
     * a %Gaussian's covariance, or a hybrid's mode weights and components.
     *
     * \returns nothing, or the NumericError of the first that fails
     */
    [[nodiscard]] std::expected<void, NumericError>
    factorize() const
    {
        if constexpr (MatrixType::RowsAtCompileTime != 0)
        {
            for (auto const & conditional : m_conditionals)
            {
                if (auto result = conditional.factorize(); !result)
                {
                    return std::unexpected{result.error()};
                }
            }
        }
        return {};
    }

    /** Reserves room for \a count points. */
    void
    reserve(Eigen::Index count)
    {
        m_conditionals.reserve(std::size_t(count));
        m_scratch.reserve(std::size_t(count));
    }

    /** Makes point \a j a copy of point \a from[j], for every \a j. */
    void
    select(std::span<Eigen::Index const> from)
    {
        select(execution::sequential, from);
    }

    /** select() under \a policy: a gather of the distributions into slots made first, each a
     * copy of the first point's where the scratch has none yet, since a hybrid has no empty state.
     */
    template <ExecutionPolicy P>
    void
    select(P && policy, std::span<Eigen::Index const> from)
    {
        if (from.size() < m_scratch.size())
        {
            m_scratch.erase(m_scratch.begin() + std::ptrdiff_t(from.size()), m_scratch.end());
        }
        while (m_scratch.size() < from.size())
        {
            m_scratch.push_back(m_conditionals.front());
        }
        if constexpr (isSequential<P>)
        {
            for (std::size_t j = 0; j < from.size(); ++j)
            {
                m_scratch[j] = m_conditionals[std::size_t(from[j])];
            }
        }
        else
        {
            forEachIndex(std::forward<P>(policy), Eigen::Index(from.size()), [&](Eigen::Index j) {
                m_scratch[std::size_t(j)] = m_conditionals[std::size_t(from[std::size_t(j)])];
            });
        }
        m_conditionals.swap(m_scratch);
    }

    /** Keeps only the points \a kept, in increasing order, in place. */
    void
    keep(std::span<Eigen::Index const> kept)
    {
        for (std::size_t k = 0; k < kept.size(); ++k)
        {
            if (kept[k] != Eigen::Index(k))
            {
                m_conditionals[k] = m_conditionals[std::size_t(kept[k])];
            }
        }
        m_conditionals.erase(m_conditionals.begin() + std::ptrdiff_t(kept.size()), m_conditionals.end());
    }

private:

    /** Allocates the distributions. */
    using GaussianAllocator = ReBound<Allocator, ConditionalType>;

    Buffer<ConditionalType, Allocator> m_conditionals;  ///< one distribution per point
    Buffer<ConditionalType, Allocator> m_scratch;  ///< distributions being rebuilt by select(), swapped in when done
};

/** \returns \a f applied to point \a i of \a points: its value as a vector, or as a %Gaussian
 * with its kernel covariance if it has one.
 *
 * \a Points describes the points of a Rao-Blackwellized state: <tt>space()</tt>,
 * <tt>pinnedSpace()</tt>, <tt>restSpace()</tt>, <tt>pinnedRows()</tt>, <tt>restIndices()</tt>,
 * <tt>pinnedDifferenceRows()</tt>, <tt>restDifferenceIndices()</tt>, <tt>size()</tt>,
 * and <tt>center(i)</tt>, with the types \c SpaceType, \c PathType, \c PinnedSpace, and \c Scalar,
 * and \c HasKernels, whether a point may have a kernel covariance. Points that may also have
 * <tt>pinnedSpace()</tt> and <tt>kernel(i)</tt>, a pointer to the point's covariance or null;
 * particles are point masses, known at compile time.
 */
template <typename Points, typename F>
[[nodiscard]] auto
withPoint(Points const & points, Eigen::Index i, F && f)
{
    using Scalar = typename Points::Scalar;
    using PinnedVector = typename Moments<typename Points::PinnedSpace, Scalar>::VectorType;
    PinnedVector center = points.center(i);
    if constexpr (Points::HasKernels)
    {
        if (auto const * kernel = points.kernel(i))
        {
            return f(
                GaussianParams<typename Points::PinnedSpace, Scalar>{points.pinnedSpace(), std::move(center), *kernel}
            );
        }
    }
    return f(center);
}

/** \returns the whole state as one %Gaussian, moment-matched over \a points with \a logWeights:
 * the points' spread and kernels for the given block, and \a conditionals for the rest. Points
 * of weight zero are left out.
 *
 * The mixture moments of internal::mixtureMomentsIn(), with the block's and the rest's
 * contributions placed in their rows of the whole state's covariance; a kernel is a point's
 * covariance about its center, so it adds to the block's rows as a component covariance does.
 */
template <typename Points, typename Store, typename WeightsDerived>
[[nodiscard]] GaussianParams<typename Points::SpaceType, typename Points::Scalar>
collapseGivenPoints(
    Points const & points,
    Store const & conditionals,
    Eigen::ArrayBase<WeightsDerived> const & logWeights
)
{
    using Scalar = typename Points::Scalar;
    using SpaceType = typename Points::SpaceType;
    using MomentsType = Moments<SpaceType, Scalar>;
    EMRYS_FILTER_ASSERT(
        logWeights.size() == points.size() && conditionals.size() == points.size()
        && "give one weight and one Gaussian per point"
    );
    BlockRows const rows = points.pinnedRows();
    auto const & rest = points.restIndices();
    BlockRows const differenceRows = points.pinnedDifferenceRows();
    auto const & restDifferences = points.restDifferenceIndices();
    auto const logTotal = logTotalWeight<Scalar>(logWeights);
    EMRYS_FILTER_ASSERT(logTotal && "the weights must be finite and not all zero");
    // The whole state at point i: its center in the block's rows, its Gaussian's mean in the rest's.
    constexpr int PinnedValues = valueDimensionAtCompileTime<typename Points::PinnedSpace>;
    constexpr int PinnedDifferences = valueDimensionAtCompileTime<DifferenceSpace<typename Points::PinnedSpace>>;
    auto const meanAt = [&](Eigen::Index i) {
        auto mean = typename MomentsType::VectorType{};
        mean.resize(valueDimension(points.space()));
        middleRowsOf<PinnedValues>(mean, rows.start, rows.size) = points.center(i);
        mean(rest) = conditionals.mean(i);
        return mean;
    };
    MomentsType moments = pointMomentsIn<MomentsType>(
        points.space(),
        points.size(),
        meanAt,
        [&](Eigen::Index i) {
            return logWeights.derived()(i);
        },
        *logTotal
    );
    auto & cov = moments.covarianceLower();
    auto pinned = blockOf<PinnedDifferences, PinnedDifferences>(
        cov,
        differenceRows.start,
        differenceRows.start,
        differenceRows.size,
        differenceRows.size
    );
    for (Eigen::Index i = 0; i < points.size(); ++i)
    {
        Scalar const logWeight = logWeights.derived()(i);
        if (logWeight == -Eigen::NumTraits<Scalar>::infinity())
        {
            continue;
        }
        Scalar const weight = Eigen::numext::exp(logWeight - *logTotal);
        if constexpr (Points::HasKernels)
        {
            if (auto const * kernel = points.kernel(i))
            {
                pinned += weight * *kernel;
            }
        }
        if constexpr (!Store::Shared)
        {
            cov(restDifferences, restDifferences) += weight * conditionals.covarianceLower(i);
        }
    }
    if constexpr (Store::Shared)
    {
        cov(restDifferences, restDifferences) += conditionals.sharedCovariance();
    }
    mirrorLower(cov);
    return GaussianParams<SpaceType, Scalar>{points.space(), std::move(moments)};
}

/** \returns the whole state as one %Gaussian, as collapseGivenPoints() gives it, with \a hybrids
 * a per-point store of hybrids of Gaussians over the rest, each collapsed over its modes first into
 * storage from \a allocator. Each hybrid must have been checked, as a density's check makes sure.
 */
template <typename Points, typename Store, typename WeightsDerived, typename Allocator>
[[nodiscard]] GaussianParams<typename Points::SpaceType, typename Points::Scalar>
collapseGivenHybrids(
    Points const & points,
    Store const & hybrids,
    Eigen::ArrayBase<WeightsDerived> const & logWeights,
    Allocator const & allocator
)
{
    using Scalar = typename Points::Scalar;
    using Traits = ComponentTraits<typename Store::ConditionalType>;
    using Gaussians = Conditionals<typename Points::RestSpace, PerPointCovariance, Scalar, Allocator>;
    auto gaussians = Gaussians{Traits::collapseModes(hybrids.conditional(0)), hybrids.size(), allocator};
    for (Eigen::Index i = 1; i < hybrids.size(); ++i)
    {
        gaussians.conditional(i) = Traits::collapseModes(hybrids.conditional(i));
    }
    return collapseGivenPoints(points, gaussians, logWeights);
}

/** \returns the probability of each mode over the points of a per-point store of hybrids
 * \a hybrids, \f$ P(m) = \sum_i w_i \, P(m \mid i) \f$ with the weights \f$ w_i \f$ from
 * \a logWeights, normalized; each hybrid's mode weights must define a distribution, and the
 * weights one too.
 */
template <typename Store, typename WeightsDerived>
[[nodiscard]] auto
summedModes(Store const & hybrids, Eigen::ArrayBase<WeightsDerived> const & logWeights)
{
    using ModesType = typename Store::ConditionalType::ModesType;
    using Scalar = typename ModesType::Scalar;
    auto const & modeSpace = hybrids.conditional(0).modeSpace();
    Eigen::Index const count = modeSpace.cardinality();
    auto logP = typename ModesType::ArrayType{};
    logP.resize(count);
    for (Eigen::Index m = 0; m < count; ++m)
    {
        auto total = LogSumExp<Scalar>{};
        for (Eigen::Index i = 0; i < hybrids.size(); ++i)
        {
            auto const & modes = hybrids.conditional(i).modes();
            // A point's mode weights are proportional to its probabilities, so each is normalized.
            total.add(Scalar(logWeights.derived()(i)) + modes.logWeight(m) - modes.logTotalWeight().value());
        }
        logP(m) = total.value();
    }
    auto modes = ModesType{modeSpace, logP};
    [[maybe_unused]] auto const normalized = modes.normalize();
    EMRYS_FILTER_ASSERT(normalized && "the points' weights must define a distribution");
    return modes;
}

/** How the rest of a state depends on its fixed block under a joint %Gaussian: a factor of the
 * block's covariance, for drawing the block; the gain \f$ G = P_{rp} P_{pp}^{+} \f$, by which a
 * draw's offset from the block's mean moves the rest's mean; and the rest's covariance given the
 * block, \f$ P_{rr} - G P_{pr} \f$.
 */
template <typename Factor, typename Gain, typename Cov>
struct PinnedConditioning
{
    Factor factor;  ///< \f$ F \f$ with \f$ F F^T = P_{pp} \f$, for drawing the block
    Gain gain;      ///< \f$ G = P_{rp} P_{pp}^{+} \f$
    Cov cov;        ///< \f$ P_{rr} - G P_{pr} \f$, the rest's covariance given the block
};

/** \returns the PinnedConditioning of the joint covariance \a cov over the whole state, split by
 * \a split. A singular block covariance, as a noiseless block has, is inverted where it is not
 * zero.
 *
 * The conditional %Gaussian of internal::conditionMoments() with the block as the given
 * coordinates, factored so that the Rao-Blackwellized particle filter's prediction
 * (\ref Schon2005 "Schön, Gustafsson, and Nordlund (2005)", §IV) can draw the block and
 * condition the rest on every particle with one factorization.
 */
template <typename Split, typename CovDerived>
[[nodiscard]] auto
pinnedConditioning(Split const & split, Eigen::MatrixBase<CovDerived> const & cov)
{
    using Scalar = typename CovDerived::Scalar;
    using PinnedMoments = Moments<typename Split::PinnedSpace, Scalar>;
    using RestMoments = Moments<typename Split::RestSpace, Scalar>;
    using PinnedMatrix = typename PinnedMoments::MatrixType;
    using RestMatrix = typename RestMoments::MatrixType;
    // Typed by the blocks' bounds: evaluating an indexed view would give an unbounded matrix.
    using PinnedRestMatrix = MatrixOf<
        Scalar,
        PinnedMoments::DifferenceDimensionAtCompileTime,
        RestMoments::DifferenceDimensionAtCompileTime,
        PinnedMoments::MaxDifferenceDimensionAtCompileTime,
        RestMoments::MaxDifferenceDimensionAtCompileTime>;
    using GainMatrix = MatrixOf<
        Scalar,
        RestMoments::DifferenceDimensionAtCompileTime,
        PinnedMoments::DifferenceDimensionAtCompileTime,
        RestMoments::MaxDifferenceDimensionAtCompileTime,
        PinnedMoments::MaxDifferenceDimensionAtCompileTime>;
    BlockRows const rows = split.pinnedDifferenceRows();
    auto const & rest = split.restDifferenceIndices();
    constexpr int PinnedRows = PinnedMoments::DifferenceDimensionAtCompileTime;
    PinnedMatrix const pinned = blockOf<PinnedRows, PinnedRows>(cov, rows.start, rows.start, rows.size, rows.size);
    PinnedRestMatrix const pinnedRest = cov(Eigen::seqN(rows.start, rows.size), rest);
    auto const ldlt = Eigen::LDLT<PinnedMatrix>{pinned};
    auto gain = transposedSolve<GainMatrix>(ldlt, pinnedRest);
    RestMatrix restCov = cov(rest, rest);
    restCov -= multiply(gain, pinnedRest);
    restCov = (Scalar(0.5) * (restCov + restCov.transpose())).eval();
    return PinnedConditioning{defaultRoot(pinned), std::move(gain), std::move(restCov)};
}

/** Draws the fixed block from a joint %Gaussian with mean \a mean (a whole state) and the
 * conditioning \a conditioning, with the standard normal draw \a z, and conditions the rest on
 * it: writes the block's draw to \a pinned and the rest's mean given it to \a restMean.
 *
 * The rest's covariance is the one conditioning holds for every draw, so it is not reset to the
 * tangent space at each draw's mean (decisions.md [covariance-reset]): the reset is part of that shared
 * approximation, and exact where the rest's space has an identity reset.
 */
template <
    typename Split,
    typename Conditioning,
    typename MeanDerived,
    typename ZDerived,
    typename PinnedOut,
    typename RestOut>
void
drawPinned(
    Split const & split,
    Conditioning const & conditioning,
    Eigen::MatrixBase<MeanDerived> const & mean,
    Eigen::MatrixBase<ZDerived> const & z,
    PinnedOut && pinned,
    RestOut && restMean
)
{
    BlockRows const rows = split.pinnedRows();
    auto const offset = multiply(conditioning.factor, z).eval();
    constexpr int PinnedValues = valueDimensionAtCompileTime<typename Split::PinnedSpace>;
    pinned = Emrys::plus(split.pinnedSpace(), middleRowsOf<PinnedValues>(mean, rows.start, rows.size), offset);
    restMean = Emrys::plus(split.restSpace(), mean(split.restIndices()), multiply(conditioning.gain, offset));
}

}  // namespace internal

}  // namespace Emrys
