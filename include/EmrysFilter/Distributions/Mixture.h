#pragma once

/** \file
 * The mixture: the parameters MixtureParams of weighted components of one type (decisions.md [component-first],
 * [params-and-density]), with its
 * components added, removed, merged, split, truncated, and reweighted, and the MixtureLineage it
 * keeps through those for the mixture smoother [smoothing]; the Mixture that evaluates the
 * density, read from them by toDensity(); the MixtureMarginal view from marginal<Path>(), a range
 * over the components' marginals [element-marginals], which a Hybrid's marginal is too; the draws
 * from a mixture of Gaussians; and the GaussianMixtureParams and GaussianMixture aliases for the
 * mixture of Gaussians, the spelling for the common case. The mixture filter's state [mixture-filter],
 * [filters-and-recursions] and mixture measurement noise [additive-noise] are mixtures. Read UniformMixture.h next.
 */

#include <algorithm>
#include <concepts>
#include <expected>
#include <functional>
#include <iterator>
#include <random>
#include <span>
#include <numeric>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>

namespace Emrys
{

/** \ingroup Distributions
 * A mixture component that could not be factorized.
 */
struct ComponentError
{
    Eigen::Index index;  ///< position of the component in the mixture
    NumericError error;  ///< why factorization failed: a %Gaussian's covariance, or a Hybrid's mode weights

    /** Equal when the position and the error are. */
    friend bool
    operator==(ComponentError const &, ComponentError const &) = default;
};

/** \ingroup Distributions
 * Reasons MixtureParams::toDensity() can fail.
 */
using MixtureError = std::variant<WeightError, ComponentError>;

/** \ingroup Distributions
 * MixtureError by the name it had when every mixture was a GaussianMixtureParams.
 */
using GaussianMixtureError = MixtureError;

template <internal::ComponentOrInfoForm Component_, typename Allocator_>
class Mixture;

template <internal::ComponentOrInfoForm Component_, typename Allocator_>
class MixtureParams;

template <typename Source, typename Selection, typename SubSpace>
class MixtureMarginal;

namespace internal
{
struct MixtureLineageAccess;
}  // namespace internal

/** \ingroup Distributions
 * \brief How each component of a Mixture descends from the components it had at its last
 * startLineage(): each component's share from each of those parents.
 *
 * Row \a i holds component \a i's log-shares, one per parent, which sum to one: the fraction of
 * its weight that came through each parent, \f$ P(\text{parent } n \mid \text{component } i) \f$.
 * MixtureParams::startLineage() gives every component itself as its one parent, and from then on
 * the mixture keeps the rows as its components change: a reweighted, moved, or replaced component
 * keeps its row; a merge takes the weighted sum of the two rows; split() gives each child its
 * parent's row; removal drops rows; and add() and append() give a new component a row of zeros,
 * no parent at all. A filter that splits a component into several updated branches gives each
 * its prior's row. This is the f-graph of
 * \ref Rahmathullah2014 "Rahmathullah, Svensson, and Svensson (2014)": the branches merged into
 * each node and their weights, kept as the normalized share per parent rather than one weight per
 * merged branch, which is what the backward pass of GaussianMixtureHistory reads.
 *
 * count() counts the mixture's startLineage() calls, so a record of the lineage can say how many
 * marks fell between two records. A copy of a mixture is a new mixture and has no lineage, so a
 * filter's state swapped for a copy shows as a count that went back; a move keeps it. Until the
 * first startLineage() there are no rows, and the mixture does no work to keep them.
 *
 * The smoother is one reader. carry() is for another: a value kept per component outside the
 * filter, such as each hypothesis's running NIS, which the rows take from the parents to the
 * components through the updates, merges, and prunes between two marks
 * (usage.md, "Checking an estimator").
 *
 * The rows come from \a Allocator_, one shared buffer of parentCount() log-shares per component.
 *
 * \tparam Scalar_     scalar type of the shares
 * \tparam Allocator_  allocator for the shares (see \ref Allocation)
 *
 * \sa MixtureParams::lineage(), GaussianMixtureHistory
 */
template <RealScalar Scalar_, typename Allocator_ = DefaultAllocator>
class MixtureLineage
{
public:

    using Scalar = Scalar_;             ///< the scalar type
    using allocator_type = Allocator_;  ///< the allocator the shares come from
    using Index = Eigen::Index;         ///< Eigen's index type

    MixtureLineage() = default;

    /** An empty record, with no mark, whose shares come from \a allocator. */
    explicit MixtureLineage(Allocator_ const & allocator)
        : m_logShares(ScalarAllocator(allocator))
    {}

    /** Copies the record into storage from \a other's allocator (see \ref Allocation). */
    MixtureLineage(MixtureLineage const & other)
        : MixtureLineage{other, other.get_allocator()}
    {}

    /** Copies \a other into storage from \a allocator. */
    MixtureLineage(MixtureLineage const & other, Allocator_ const & allocator)
        : m_logShares(other.m_logShares, ScalarAllocator(allocator))
        , m_parentCount{other.m_parentCount}
        , m_count{other.m_count}
    {}

    /** Takes \a other's record. */
    MixtureLineage(MixtureLineage && other) = default;

    /** Moves \a other into storage from \a allocator, taking its storage when the allocators are
     * equal.
     */
    MixtureLineage(MixtureLineage && other, Allocator_ const & allocator)
        : m_logShares(std::move(other.m_logShares), ScalarAllocator(allocator))
        , m_parentCount{other.m_parentCount}
        , m_count{other.m_count}
    {}

    /** Copies \a other's record into this one's storage. */
    MixtureLineage &
    operator=(MixtureLineage const & other) = default;

    /** Takes \a other's record, as \c std::vector's move assignment does. */
    MixtureLineage &
    operator=(MixtureLineage && other) = default;

    ~MixtureLineage() = default;

    /** \returns the allocator the shares come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_logShares.get_allocator());
    }

    /** \returns the number of MixtureParams::startLineage() calls on the mixture since it was made,
     * zero for a mixture with no lineage
     */
    [[nodiscard]] Index
    count() const
    {
        return m_count;
    }

    /** \returns the number of components the mixture had at its last startLineage(), the parents */
    [[nodiscard]] Index
    parentCount() const
    {
        return m_parentCount;
    }

    /** \returns the number of components with a row: the mixture's size(), or zero with no lineage */
    [[nodiscard]] Index
    size() const
    {
        return m_parentCount == 0 ? Index(0) : Index(m_logShares.size()) / m_parentCount;
    }

    /** \returns component \a i's log-share from each parent, \f$ -\infty \f$ for none */
    [[nodiscard]] std::span<Scalar const>
    logShares(Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < size() && "the component has no row");
        return {m_logShares.data() + i * m_parentCount, std::size_t(m_parentCount)};
    }

    /** \returns component \a i's log-share from parent \a parent */
    [[nodiscard]] Scalar
    logShare(Index i, Index parent) const
    {
        EMRYS_FILTER_ASSERT(parent >= 0 && parent < m_parentCount && "no such parent");
        return logShares(i)[std::size_t(parent)];
    }

    /** Carries a value per parent to the components, as a statistic kept per hypothesis follows
     * them through updates, merges, and prunes: \a children(i), one per component, becomes the mean
     * of \a parents over component \a i's parents, weighted by its shares from them, so a component
     * with one parent takes that parent's value and a merge takes the share-weighted mean of two; a
     * component with no parent gets \a orphan. \a parents has parentCount() values and \a children
     * size(), both sized by the caller.
     */
    template <typename Parents, typename Children>
    void
    carry(Eigen::MatrixBase<Parents> const & parents, Eigen::MatrixBase<Children> & children, Scalar orphan = 0) const
    {
        EMRYS_FILTER_ASSERT(
            parents.size() == parentCount() && children.size() == size() && "one value per parent and per component"
        );
        Scalar const none = -Eigen::NumTraits<Scalar>::infinity();
        for (Index i = 0; i < size(); ++i)
        {
            auto const shares = logShares(i);
            Scalar const top = *std::ranges::max_element(shares);
            if (top == none)
            {
                children(i) = orphan;
                continue;
            }
            Scalar sum = 0;
            Scalar total = 0;
            for (Index parent = 0; parent < m_parentCount; ++parent)
            {
                Scalar const share = Eigen::numext::exp(Scalar(shares[std::size_t(parent)] - top));
                sum += share * parents(parent);
                total += share;
            }
            children(i) = sum / total;
        }
    }

private:

    /** The mixture keeps its rows. */
    template <internal::ComponentOrInfoForm, typename>
    friend class MixtureParams;

    friend struct internal::MixtureLineageAccess;  ///< filters carry the rows through their updates

    using ScalarAllocator = internal::ReBound<Allocator_, Scalar>;  ///< allocates the shares

    /** \returns whether there is a lineage to keep */
    [[nodiscard]] bool
    tracking() const
    {
        return m_count > 0;
    }

    /** Marks \a components components as the parents, each its own, and counts the mark. Room for
     * \a capacity rows is reserved if given.
     */
    void
    start(Index components, std::optional<Index> capacity)
    {
        ++m_count;
        m_parentCount = components;
        if (capacity)
        {
            m_logShares.reserve(std::size_t(*capacity * components));
        }
        m_logShares.assign(std::size_t(components * components), -Eigen::NumTraits<Scalar>::infinity());
        for (Index i = 0; i < components; ++i)
        {
            m_logShares[std::size_t(i * components + i)] = Scalar(0);
        }
    }

    /** Forgets the lineage, as a copy of a mixture has none, keeping the storage. */
    void
    reset()
    {
        m_logShares.clear();
        m_parentCount = 0;
        m_count = 0;
    }

    /** Drops every row, keeping the mark, as clear() leaves a mixture with no components. */
    void
    clearRows()
    {
        m_logShares.clear();
    }

    /** Appends a row with no parent. */
    void
    pushOrphan()
    {
        if (tracking())
        {
            m_logShares.resize(m_logShares.size() + std::size_t(m_parentCount), -Eigen::NumTraits<Scalar>::infinity());
        }
    }

    /** Appends a copy of \a row, which has a share per parent. */
    void
    push(std::span<Scalar const> row)
    {
        if (tracking())
        {
            EMRYS_FILTER_ASSERT(Index(row.size()) == m_parentCount && "the row must have a share per parent");
            m_logShares.insert(m_logShares.end(), row.begin(), row.end());
        }
    }

    /** Appends a copy of row \a i. */
    void
    pushCopyOf(Index i)
    {
        if (tracking())
        {
            std::size_t const end = m_logShares.size();
            m_logShares.resize(end + std::size_t(m_parentCount));
            std::copy_n(
                m_logShares.begin() + i * m_parentCount,
                m_parentCount,
                m_logShares.begin() + std::ptrdiff_t(end)
            );
        }
    }

    /** Moves row \a from over row \a to. */
    void
    moveRow(Index from, Index to)
    {
        if (tracking() && from != to)
        {
            std::copy_n(
                m_logShares.begin() + from * m_parentCount,
                m_parentCount,
                m_logShares.begin() + to * m_parentCount
            );
        }
    }

    /** Keeps the first \a rows rows. */
    void
    truncate(Index rows)
    {
        if (tracking())
        {
            m_logShares.resize(std::size_t(rows * m_parentCount));
        }
    }

    /** Removes row \a i by moving the last row into its place, as MixtureParams::removeAt() does. */
    void
    removeAt(Index i)
    {
        if (tracking())
        {
            moveRow(size() - 1, i);
            truncate(size() - 1);
        }
    }

    /** Makes row \a keep the shares of the merge of components \a keep and \a absorb, of
     * log-weights \a logKeep and \a logAbsorb: each parent's share is the weighted sum of the two.
     */
    void
    merge(Index keep, Index absorb, Scalar logKeep, Scalar logAbsorb)
    {
        if (!tracking())
        {
            return;
        }
        auto total = LogSumExp<Scalar>{};
        total.add(logKeep);
        total.add(logAbsorb);
        Scalar const logTotal = total.value();
        if (!isFinite(logTotal))
        {
            return;
        }
        Scalar * const kept = m_logShares.data() + keep * m_parentCount;
        Scalar const * const absorbed = m_logShares.data() + absorb * m_parentCount;
        for (Index n = 0; n < m_parentCount; ++n)
        {
            auto share = LogSumExp<Scalar>{};
            share.add(kept[n] + logKeep - logTotal);
            share.add(absorbed[n] + logAbsorb - logTotal);
            kept[n] = share.value();
        }
    }

    /** Exchanges the record with \a other's. */
    void
    swap(MixtureLineage & other)
    {
        m_logShares.swap(other.m_logShares);
        std::swap(m_parentCount, other.m_parentCount);
        std::swap(m_count, other.m_count);
    }

    internal::Buffer<Scalar, Allocator_> m_logShares;  ///< a row of parentCount() log-shares per component
    Index m_parentCount = 0;                           ///< the components at the last mark
    Index m_count = 0;                                 ///< the marks since the mixture was made
};

namespace internal
{

/** Carries a mixture's lineage into a mixture a filter builds from it, which a copy would drop,
 * and into a record from another allocator.
 */
struct MixtureLineageAccess
{
    /** Copies \a from's record into \a to's storage, whatever their allocators. */
    template <typename Scalar, typename ToAllocator, typename FromAllocator>
    static void
    assign(MixtureLineage<Scalar, ToAllocator> & to, MixtureLineage<Scalar, FromAllocator> const & from)
    {
        to.m_logShares.assign(from.m_logShares.begin(), from.m_logShares.end());
        to.m_parentCount = from.m_parentCount;
        to.m_count = from.m_count;
    }

    /** Gives \a to, a copy of \a from's components, \a from's lineage. */
    template <typename MixtureType>
    static void
    copy(MixtureType & to, MixtureType const & from)
    {
        to.m_lineage = from.m_lineage;
    }

    /** Gives \a to, which has no components, \a from's mark and parents, with no rows yet. */
    template <typename MixtureType>
    static void
    startFrom(MixtureType & to, MixtureType const & from)
    {
        EMRYS_FILTER_ASSERT(to.empty() && "the mixture must have no components");
        to.m_lineage.m_logShares.clear();
        to.m_lineage.m_parentCount = from.m_lineage.m_parentCount;
        to.m_lineage.m_count = from.m_lineage.m_count;
    }

    /** Gives \a to's newest component the row of \a from's component \a i, its prior. */
    template <typename MixtureType>
    static void
    inherit(MixtureType & to, MixtureType const & from, Eigen::Index i)
    {
        if (from.m_lineage.tracking())
        {
            to.m_lineage.truncate(to.size() - 1);
            to.m_lineage.push(from.m_lineage.logShares(i));
        }
    }
};

/** Shared implementation of MixtureParams::conditional() and Mixture::conditional(), with
 * resolved indices and the spaces of the target and the given entries.
 */
template <typename Source, typename Target, typename Given, typename Derived, typename TargetSpace, typename GivenSpace>
[[nodiscard]] auto
conditionalMixtureIn(
    Source const & source,
    Target const & target,
    Given const & given,
    Eigen::MatrixBase<Derived> const & values,
    TargetSpace const & targetSpace,
    GivenSpace const & givenSpace
)
{
    EMRYS_FILTER_ASSERT(distinctIndices(target.values, given.values) && "target and given indices must be distinct");
    using Params = MixtureParams<Gaussian<TargetSpace, typename Source::Scalar>, typename Source::allocator_type>;
    using Result = std::expected<Params, GaussianError>;
    auto result = Params{targetSpace, source.get_allocator()};
    result.reserve(source.size());
    for (auto const & entry : source)
    {
        auto conditioned = conditionMomentsWithEvidence(
            entry.component.mean(),
            entry.component.covarianceLower(),
            target,
            given,
            values,
            givenSpace,
            targetSpace
        );
        if (!conditioned)
        {
            return Result{std::unexpect, GaussianError::NotPositiveDefinite};
        }
        result.add(
            entry.logWeight + conditioned->logLikelihood,
            std::move(conditioned->moments.mean),
            std::move(conditioned->moments.cov)
        );
    }
    return Result{std::move(result)};
}

/** Shared implementation of conditional<GivenPath>(); see tagConditional(). */
template <typename GivenPath, typename Source, typename Derived>
[[nodiscard]] auto
tagConditionalMixture(Source const & source, Eigen::MatrixBase<Derived> const & values)
{
    return tagConditional<GivenPath>(
        source.space(),
        [&](auto const & target, auto const & given, auto const & targetSpace, auto const & givenSpace) {
            return conditionalMixtureIn(source, target, given, values, targetSpace, givenSpace);
        }
    );
}

}  // namespace internal

/** \ingroup Distributions
 * \brief A weighted sum of components of one type, all in one space.
 *
 * A [mixture model](https://en.wikipedia.org/wiki/Mixture_model) with unnormalized weights. Each
 * entry has a log-weight and a component (see Component): a Gaussian, which makes the mixture a
 * %Gaussian mixture, GaussianMixtureParams; or a Hybrid, which makes it a mixture of hybrid states,
 * the state of a mixture of interacting multiple model filters (decisions.md [component-first],
 * [filters-and-recursions]). Weights need not be normalized; a log-weight of -inf is a zero weight. All components
 * share the mixture's space, and so its valueDimension() and differenceDimension().
 *
 * Entries are read by index, `mixture[i]`, or in order through const iterators, and changed only
 * through this object's methods, which take an entry's index, as a ParticleSet's do (decisions.md
 * [mixture-entries]). Adding keeps every index and may invalidate every iterator; removal moves the
 * last entry into the gap, so an index is held only while nothing is removed.
 *
 * toDensity() validates the weights, factorizes every component that has changed, and returns a
 * Mixture, which checks this object for changes: reading it after this object has changed in any
 * way is a programmer error caught by \c EMRYS_FILTER_ASSERT (decisions.md [view-stamp]). So is changing the
 * mixture from inside a forEach() or removeIf() callback.
 *
 * What only a %Gaussian has is offered for %Gaussian components alone, and a mixture of other
 * components says so at the call: the members that take or give a component's moments (add() of
 * moments, setMoments(), setMean(), setCovariance(), merge() with moments, insert(), erase(), and
 * resizeComponents()), the marginal() and conditional() views, and sampling. Every mixture adds
 * and replaces whole components, collapses a pair with merge(keep, absorb), and reweights,
 * truncates, and removes.
 *
 * marginal() returns a read-only view of a subset of every component's entries, which checks only
 * this object's size and structure: values and weights may change under it, but reading it after
 * components are added, removed, or resized, or this object is assigned to, is a programmer error.
 * Neither view may outlive this object, and neither may be made from a temporary.
 *
 * Every component shares the mixture's space (see \ref Spaces): component means are normalized
 * when set, and moments() averages in the space's tangent space.
 *
 * The entries, and the scratch that pruneTail() and pruneToSize() use, come from the allocator
 * (see \ref Allocation). A mixture given a Capacity reserves room for that many components once;
 * holding more is a precondition violation.
 *
 * \tparam Component_  the components' type (see Component), which sets their space, scalar,
 *                     dimensions, and storage bounds
 * \tparam Allocator_  allocator for the entries and scratch (see \ref Allocation)
 *
 * \sa class Mixture, GaussianMixtureParams, class GaussianParams, class Hybrid
 */
template <internal::ComponentOrInfoForm Component_, typename Allocator_ = DefaultAllocator>
class MixtureParams
{
    static_assert(
        Component<Component_>,
        "a Gaussian in an information form is not a mixture component, having no collapse; convert each "
        "with toGaussian() or toSqrt() and hold a Gaussian or a SqrtGaussian (decisions.md [gaussian-forms])"
    );

public:

    using ComponentType = Params<Component_>;                                 ///< a component
    using SpaceType = typename ComponentType::SpaceType;                      ///< the components' space
    using allocator_type = Allocator_;                                        ///< the allocator, re-bound per buffer
    using MomentsType = typename ComponentType::MomentsType;                  ///< a mean and covariance together
    using Scalar = typename ComponentType::Scalar;                            ///< the scalar type
    using Index = Eigen::Index;                                               ///< Eigen's index type
    using VectorType = typename MomentsType::VectorType;                      ///< a value, such as a mean
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;  ///< a difference
    using MatrixType = typename MomentsType::MatrixType;                      ///< a matrix over differences
    using DensityType = Mixture<Component_, Allocator_>;                      ///< the view toDensity() returns
    using LineageType = MixtureLineage<Scalar, Allocator_>;                   ///< what lineage() returns

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = ComponentType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = ComponentType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = ComponentType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = ComponentType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;  ///< parameters, which views check

    /** One weighted component. Only ever exposed as const outside the mixture. An entry whose
     * component has an allocator, such as a Hybrid, has its \c allocator_type, so the mixture's
     * entries build their components from the mixture's allocator when it passes itself on.
     */
    struct Entry : internal::AllocatorTypeOf<ComponentType>
    {
        /** The component \a held with log-weight \a weight. */
        Entry(Scalar weight, ComponentType held)
            : logWeight{weight}
            , component{std::move(held)}
        {}

        /** Copies \a other with its component in storage from \a allocator. */
        template <typename A>
            requires(std::is_constructible_v<ComponentType, ComponentType const &, A const &>)
        Entry(Entry const & other, A const & allocator)
            : logWeight{other.logWeight}
            , component{other.component, allocator}
        {}

        /** Moves \a other with its component into storage from \a allocator. */
        template <typename A>
            requires(std::is_constructible_v<ComponentType, ComponentType &&, A const &>)
        Entry(Entry && other, A const & allocator)
            : logWeight{other.logWeight}
            , component{std::move(other.component), allocator}
        {}

        Entry(Entry const &) = default;  ///< copies the weight and the component
        Entry(Entry &&) = default;       ///< moves the weight and the component

        /** Copies into the component's storage. */
        Entry &
        operator=(Entry const &) = default;

        /** Moves the weight and the component. */
        Entry &
        operator=(Entry &&) = default;

        ~Entry() = default;

        Scalar logWeight;         ///< the log of the component's (unnormalized) weight
        ComponentType component;  ///< the component, in the mixture's space
    };

    using const_iterator = typename internal::Buffer<Entry, Allocator_>::const_iterator;  ///< iterates the entries

    /** Result of removeIf(): the log of the removed fraction of the
     * total weight, or nothing when \a ReturnRemovedWeight is false.
     */
    template <bool ReturnRemovedWeight>
    using RemovalResult = std::conditional_t<ReturnRemovedWeight, std::expected<Scalar, WeightError>, void>;

    /** \brief Writable access to one entry during forEach().
     *
     * Exists only for the duration of the callback; it cannot be copied or moved.
     * The component's dimensions cannot be changed through it.
     */
    class MutableEntry
    {
    public:

        MutableEntry(MutableEntry const &) = delete;  ///< lives only inside the callback

        /** Lives only inside the callback. */
        MutableEntry &
        operator=(MutableEntry const &) = delete;

        /** \returns the entry's log-weight */
        [[nodiscard]] Scalar
        logWeight() const
        {
            return m_entry.logWeight;
        }

        /** \returns the entry's position in the mixture, for a callback that writes a slot per entry */
        [[nodiscard]] Index
        index() const
        {
            return m_index;
        }

        /** \returns the entry's component */
        [[nodiscard]] ComponentType const &
        component() const
        {
            return m_entry.component;
        }

        /** Replaces the entry's component with \a component, which must be in the mixture's space;
         * see MixtureParams::setComponent().
         */
        template <typename C>
            requires std::same_as<std::remove_cvref_t<C>, ComponentType>
        void
        setComponent(C && component)
        {
            EMRYS_FILTER_ASSERT(
                component.space() == m_entry.component.space() && "the component must be in the mixture's space"
            );
            MixtureParams::assignComponent(m_entry, std::forward<C>(component));
        }

        /** \returns \a f called with the entry's component, to change it in place, as a filter's
         * step does; \a f must keep the component's space, which is asserted after
         */
        template <typename F>
            requires std::invocable<F &, ComponentType &>
        decltype(auto)
        updateComponent(F && f)
        {
            return internal::keepingSpace(m_entry.component, f);
        }

        /** \returns the component's mean and covariance together */
        [[nodiscard]] MomentsType const &
        moments() const
            requires(isGaussian<ComponentType>)
        {
            return component().moments();
        }

        /** \returns the component's mean */
        [[nodiscard]] VectorType const &
        mean() const
            requires(isGaussian<ComponentType>)
        {
            return component().mean();
        }

        /** \returns the component's covariance */
        [[nodiscard]] MatrixType const &
        covarianceLower() const
            requires(isGaussian<ComponentType>)
        {
            return component().covarianceLower();
        }

        /** Sets the component's mean; see GaussianParams::setMean(). */
        template <MatrixExpression Mean>
            requires(isGaussian<ComponentType>)
        void
        setMean(Mean && mean)
        {
            m_entry.component.setMean(std::forward<Mean>(mean));
        }

        /** Sets the component's covariance; see GaussianParams::setCovariance(). */
        template <MatrixExpression Cov>
            requires(isGaussian<ComponentType>)
        void
        setCovariance(Cov && cov)
        {
            m_entry.component.setCovariance(std::forward<Cov>(cov));
        }

        /** Sets the component's mean and covariance to \a moments. */
        void
        setMoments(MomentsType moments)
            requires(isGaussian<ComponentType>)
        {
            MixtureParams::assignComponent(m_entry, std::move(moments));
        }

    private:

        friend MixtureParams;  ///< the mixture makes these in forEach()

        /** Wraps \a entry, at position \a index, for the duration of a callback. */
        MutableEntry(Entry & entry, Index index)
            : m_entry{entry}
            , m_index{index}
        {}

        Entry & m_entry;  ///< the wrapped entry
        Index m_index;    ///< its position
    };

    MixtureParams() = default;

    /** Constructs an empty mixture that allocates through \a allocator. */
    explicit MixtureParams(Allocator_ const & allocator)
        : m_entries(EntryAllocator(allocator))
        , m_weightScratch(ScalarAllocator(allocator))
        , m_lineage(allocator)
    {}

    /** Constructs an empty mixture in a FixedSizeSpace that reserves room for \a capacity
     * components. A space that holds values is given: MixtureParams(space, capacity, allocator).
     */
    explicit MixtureParams(Capacity capacity, Allocator_ const & allocator = {})
        requires FixedSizeSpace<SpaceType>
        : MixtureParams{allocator}
    {
        setCapacity(capacity.count);
    }

    /** Constructs an empty mixture whose components will have \a valueDimension value coordinates. */
    explicit MixtureParams(Index valueDimension, Allocator_ const & allocator = {})
        requires internal::SizedByLength<SpaceType>
        : MixtureParams{allocator}
    {
        fitSpace(valueDimension);
    }

    /** Constructs an empty mixture in \a space. */
    explicit MixtureParams(SpaceType space, Allocator_ const & allocator = {})
        : MixtureParams{allocator}
    {
        m_space = std::move(space);
    }

    /** Constructs an empty mixture in \a space that reserves room for \a capacity components. */
    MixtureParams(SpaceType space, Capacity capacity, Allocator_ const & allocator = {})
        : MixtureParams{std::move(space), allocator}
    {
        setCapacity(capacity.count);
    }

    /** Constructs an empty mixture in \a space whose components will have \a valueDimension value
     * coordinates. A space with one resizable block is resized to \a valueDimension; any other
     * must already have that many.
     */
    MixtureParams(SpaceType space, Index valueDimension, Allocator_ const & allocator = {})
        : MixtureParams{std::move(space), allocator}
    {
        fitSpace(valueDimension);
    }

    /** Copies the entries and their factorizations into storage from \a other's allocator, with
     * its capacity (see \ref Allocation). The copy has no lineage: it is a new mixture (see
     * MixtureLineage), which views of \a other do not see.
     */
    MixtureParams(MixtureParams const & other)
        : MixtureParams{other, other.get_allocator()}
    {}

    /** Copies the entries and their factorizations into storage from \a allocator, with
     * \a other's capacity and no lineage.
     */
    MixtureParams(MixtureParams const & other, Allocator_ const & allocator)
        : MixtureParams{allocator}
    {
        m_space = other.m_space;
        if (other.m_capacity)
        {
            setCapacity(*(other.m_capacity));
        }
        m_entries.assign(other.m_entries.begin(), other.m_entries.end());
    }

    /** Takes \a other's entries and storage; a view of \a other may not be read afterward. */
    MixtureParams(MixtureParams && other) noexcept
        : m_entries{std::move(other.m_entries)}
        , m_space{std::move(other.m_space)}
        , m_capacity{other.m_capacity}
        , m_weightScratch{std::move(other.m_weightScratch)}
        , m_lineage{std::move(other.m_lineage)}
    {
        other.m_stamp.touchShape();
    }

    /** Moves \a other's entries into storage from \a allocator, taking its storage when the
     * allocators are equal.
     */
    MixtureParams(MixtureParams && other, Allocator_ const & allocator)
        : m_entries{std::move(other.m_entries), EntryAllocator(allocator)}
        , m_space{std::move(other.m_space)}
        , m_capacity{other.m_capacity}
        , m_weightScratch(ScalarAllocator(allocator))
        , m_lineage{std::move(other.m_lineage), allocator}
    {
        other.m_stamp.touchShape();
    }

    /** Copies \a other's entries and space into this mixture's storage, which keeps its allocator
     * and capacity. An entry that already holds a component is assigned to, so the components'
     * own storage is reused. The lineage is cleared, as a copy has none.
     */
    MixtureParams &
    operator=(MixtureParams const & other)
    {
        m_stamp.touchShape();
        if (this == &other)
        {
            return *this;
        }
        assertCapacity(other.size());
        m_entries.assign(other.m_entries.begin(), other.m_entries.end());
        m_space = other.m_space;
        m_lineage.reset();
        return *this;
    }

    /** Takes \a other's entries, space, and lineage, and its storage when the allocators are equal
     * or propagate; otherwise moves the entries into this mixture's storage, which keeps its
     * capacity. An allocator that propagates and differs brings \a other's scratch with it. A view
     * of either object may not be read afterward.
     */
    MixtureParams &
    operator=(MixtureParams && other) noexcept
    {
        m_stamp.touchShape();
        other.m_stamp.touchShape();
        if (this == &other)
        {
            return *this;
        }
        if (std::allocator_traits<Allocator_>::propagate_on_container_move_assignment::value
            || get_allocator() == other.get_allocator())
        {
            if (get_allocator() != other.get_allocator())
            {
                // The entries take other's allocator, so the scratch does too.
                m_weightScratch = std::move(other.m_weightScratch);
            }
            m_entries = std::move(other.m_entries);
            m_capacity = other.m_capacity;
            m_lineage = std::move(other.m_lineage);
        }
        else
        {
            // Copied into this mixture's storage, which keeps its capacity.
            assertCapacity(other.size());
            m_entries.assign(
                std::make_move_iterator(other.m_entries.begin()),
                std::make_move_iterator(other.m_entries.end())
            );
            m_lineage = other.m_lineage;
        }
        m_space = std::move(other.m_space);
        return *this;
    }

    /** \returns the allocator the entries come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_entries.get_allocator());
    }

    /** \returns the number of components */
    [[nodiscard]] Index
    size() const
    {
        return Index(m_entries.size());
    }

    /** \returns whether there are no components */
    [[nodiscard]] bool
    empty() const
    {
        return m_entries.empty();
    }

    /** \returns the most components the mixture holds, if it has a capacity */
    [[nodiscard]] std::optional<Index>
    capacity() const
    {
        return m_capacity;
    }

    /** Reserves room for \a capacity components, and the scratch for that many, and makes
     * holding more a precondition violation. It must be at least size().
     */
    void
    setCapacity(Index capacity)
    {
        EMRYS_FILTER_ASSERT(capacity >= size() && "the capacity is smaller than the mixture");
        m_capacity = capacity;
        reserveScratch(capacity);
    }

    /** \returns the number of coordinates of every component's mean (see \ref Spaces) */
    [[nodiscard]] Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the number of rows and columns of every component's covariance (see \ref Spaces) */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

    /** \returns the space every component is in, with its sizes */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return m_space;
    }

    /** Resizes every component, and a space with one resizable block, to \a n value coordinates.
     * Component contents are unspecified afterward. To grow or shrink a resizable block element by
     * element, use insert() and erase(), as a space with several resizable blocks must.
     */
    void
    resizeComponents(Index n)
        requires(isGaussian<ComponentType> && internal::SizedByLength<SpaceType>)
    {
        m_stamp.touchShape();
        internal::fitValueDimension(m_space, n);
        for (auto & entry : m_entries)
        {
            entry.component.resize(n);
        }
    }

    /** Inserts the entries \a mean into the resizable block named by the tag path \a Path of
     * every component, before its element \a element, with covariance \a cov and
     * cross-covariance \a cross with each component's existing entries. See
     * GaussianParams::insert().
     */
    template <typename Path, typename MeanDerived, typename CovDerived, typename CrossDerived>
        requires(isGaussian<ComponentType>)
    void
    insert(
        Index element,
        Eigen::MatrixBase<MeanDerived> const & mean,
        Eigen::MatrixBase<CovDerived> const & cov,
        Eigen::MatrixBase<CrossDerived> const & cross
    )
    {
        m_stamp.touchShape();
        static_cast<void>(internal::growBlock<Path>(m_space, element, mean.size()));
        for (auto & entry : m_entries)
        {
            entry.component.template insert<Path>(element, mean, cov, cross);
        }
    }

    /** insert() of entries independent of the existing ones. */
    template <typename Path, typename MeanDerived, typename CovDerived>
        requires(isGaussian<ComponentType>)
    void
    insert(Index element, Eigen::MatrixBase<MeanDerived> const & mean, Eigen::MatrixBase<CovDerived> const & cov)
    {
        using Cross = internal::MatrixOf<
            Scalar,
            Eigen::Dynamic,
            Eigen::Dynamic,
            CovDerived::MaxRowsAtCompileTime,
            MomentsType::MatrixType::MaxColsAtCompileTime>;
        insert<Path>(element, mean, cov, Cross::Zero(cov.rows(), differenceDimension()));
    }

    /** Removes \a count elements of the resizable block named by the tag path \a Path from
     * every component, starting at its element \a element. See GaussianParams::erase().
     */
    template <typename Path>
        requires(isGaussian<ComponentType>)
    void
    erase(Index element, Index count = 1)
    {
        m_stamp.touchShape();
        static_cast<void>(internal::shrinkBlock<Path>(m_space, element, count));
        for (auto & entry : m_entries)
        {
            entry.component.template erase<Path>(element, count);
        }
    }

    /** Reserves room for \a n components without setting a capacity. */
    void
    reserve(Index n)
    {
        m_stamp.touchValues();
        assertCapacity(n);
        m_entries.reserve(std::size_t(n));
        reserveScratch(n);
    }

    /** Removes every component; the space, capacity, scratch, and lineage's mark stay. */
    void
    clear()
    {
        m_stamp.touchShape();
        m_entries.clear();
        m_lineage.clearRows();
    }

    /** Exchanges the entries, space, and lineage with \a other, which must come from an equal
     * allocator.
     * Each mixture keeps its capacity and scratch, so a filter can build a posterior in a second
     * mixture and swap it in without allocating.
     */
    void
    swap(MixtureParams & other)
    {
        m_stamp.touchShape();
        other.m_stamp.touchShape();
        EMRYS_FILTER_ASSERT(get_allocator() == other.get_allocator() && "the mixtures must have equal allocators");
        assertCapacity(other.size());
        other.assertCapacity(size());
        m_entries.swap(other.m_entries);
        std::swap(m_space, other.m_space);
        m_lineage.swap(other.m_lineage);
    }

    /** \returns how each component descends from those the mixture had at its last
     * startLineage(), with no rows before the first (see MixtureLineage)
     */
    [[nodiscard]] LineageType const &
    lineage() const
    {
        return m_lineage;
    }

    /** Makes the current components the parents of the lineage, each its own, and counts the
     * mark. A filter's predictJoint() calls it, so that the lineage of its state at the next
     * record says how each component descends from those it recorded. With a capacity, room for
     * a row per component is reserved, so that keeping the lineage allocates nothing afterward.
     */
    void
    startLineage()
    {
        m_stamp.touchValues();
        m_lineage.start(size(), m_capacity);
    }

    /** \returns an iterator to the first entry */
    [[nodiscard]] const_iterator
    begin() const
    {
        return m_entries.cbegin();
    }

    /** \returns an iterator past the last entry */
    [[nodiscard]] const_iterator
    end() const
    {
        return m_entries.cend();
    }

    /** \returns entry \a i, which must be less than size(). The members that change an entry take
     * its index too, as a particle set's do: an index is what a filter, the lineage, and an update
     * keep, and it stays valid when the mixture grows, where an iterator need not.
     */
    [[nodiscard]] Entry const &
    operator[](Index i) const
    {
        assertIndex(i);
        return at(i);
    }

    /** Appends \a component, which must be in the mixture's space, with log-weight \a logWeight,
     * and no parent in the lineage. \returns the index of its entry, size() - 1.
     */
    Index
    add(Scalar logWeight, ComponentType component)
    {
        m_stamp.touchShape();
        EMRYS_FILTER_ASSERT(component.space() == m_space && "the component must be in the mixture's space");
        assertCapacity(size() + 1);
        m_entries.push_back(Entry{logWeight, std::move(component)});
        m_lineage.pushOrphan();
        return size() - 1;
    }

    /** Appends the %Gaussian with \a moments. \returns the index of its entry. */
    Index
    add(Scalar logWeight, MomentsType moments)
        requires(isGaussian<ComponentType>)
    {
        EMRYS_FILTER_ASSERT(
            moments.valueDimension() == valueDimension() && moments.differenceDimension() == differenceDimension()
        );
        return add(logWeight, ComponentType{m_space, std::move(moments)});
    }

    /** add() of the %Gaussian with mean \a mean and covariance \a cov, moving either one passed as
     * an rvalue of the stored type.
     */
    template <MatrixExpression Mean, MatrixExpression Cov>
        requires(isGaussian<ComponentType>)
    Index
    add(Scalar logWeight, Mean && mean, Cov && cov)
    {
        return add(logWeight, MomentsType{std::forward<Mean>(mean), std::forward<Cov>(cov)});
    }

    /** Removes entry \a i by moving the last entry into its place. */
    void
    remove(Index i)
    {
        assertIndex(i);
        m_stamp.touchShape();
        removeAt(i);
    }

    /** Replaces component \a i with \a component, which must be in the mixture's space. */
    void
    setComponent(Index i, ComponentType component)
    {
        EMRYS_FILTER_ASSERT(component.space() == m_space && "the component must be in the mixture's space");
        assignComponent(mutableAt(i), std::move(component));
    }

    /** Sets the mean and covariance of %Gaussian \a i to \a moments. */
    void
    setMoments(Index i, MomentsType moments)
        requires(isGaussian<ComponentType>)
    {
        assignComponent(mutableAt(i), std::move(moments));
    }

    /** Sets the mean of %Gaussian \a i; see GaussianParams::setMean(). */
    template <MatrixExpression Mean>
        requires(isGaussian<ComponentType>)
    void
    setMean(Index i, Mean && mean)
    {
        mutableAt(i).component.setMean(std::forward<Mean>(mean));
    }

    /** Sets the covariance of %Gaussian \a i; see GaussianParams::setCovariance(). */
    template <MatrixExpression Cov>
        requires(isGaussian<ComponentType>)
    void
    setCovariance(Index i, Cov && cov)
    {
        mutableAt(i).component.setCovariance(std::forward<Cov>(cov));
    }

    /** Sets the log-weight of entry \a i. */
    void
    setLogWeight(Index i, Scalar logWeight)
    {
        mutableAt(i).logWeight = logWeight;
    }

    /** Multiplies the weight of entry \a i by \f$ e^{\mathrm{logScale}} \f$. */
    void
    scaleWeight(Index i, Scalar logScale)
    {
        mutableAt(i).logWeight += logScale;
    }

    /** Multiplies every weight by \f$ e^{\mathrm{logScale}} \f$. */
    void
    scaleWeights(Scalar logScale)
    {
        m_stamp.touchValues();
        for (auto & entry : m_entries)
        {
            entry.logWeight += logScale;
        }
    }

    /** \returns the log of the sum of the weights, or a WeightError if the mixture
     *          is empty, the weights sum to zero, or a log-weight is NaN or +inf.
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    logTotalWeight() const
    {
        return internal::logTotalWeight<Scalar>(m_entries | std::views::transform(&Entry::logWeight));
    }

    /** Scales the weights so they sum to one. On failure the weights are unchanged.
     *
     * \sa logTotalWeight()
     */
    [[nodiscard]] std::expected<void, WeightError>
    normalize()
    {
        m_stamp.touchValues();
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        scaleWeights(-*total);
        return {};
    }

    /** \returns the effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$ of the normalized
     *          weights, between 1 and size(), or the WeightError of logTotalWeight(). It reads
     *          only the weights, which logTotalWeight() checks, so it needs no toDensity(); the
     *          moments and the collapse read the components, which only the factorization
     *          checks, so they are the density's.
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    ess() const
    {
        return logTotalWeight().transform([this](Scalar total) {
            return essGiven(total);
        });
    }

    /** \returns the index of the most probable component, the one with the largest weight; ties
     *          go to the lowest. It is the estimate a display shows of a multimodal mixture, where
     *          the moments can sit between the modes. A programmer error if the mixture is empty;
     *          a NaN weight has no place in the order, so with one the index is unspecified, and
     *          logTotalWeight() reports it.
     */
    [[nodiscard]] Index
    mostProbable() const
    {
        EMRYS_FILTER_ASSERT(!empty() && "the most probable component of an empty mixture");
        return Index(std::ranges::max_element(m_entries, std::ranges::less{}, &Entry::logWeight) - m_entries.begin());
    }

    /** Replaces the components with those in \a components, a range of (log-weight, component)
     * pairs, or of (log-weight, moments) pairs for %Gaussian components, or other
     * structured-bindable pairs, such as uniformComponents()'s. The components have no parent in
     * the lineage, and the space and capacity stay, so a mixture made with a capacity is filled
     * without allocating.
     */
    template <std::ranges::input_range R>
    void
    assign(R && components)
    {
        clear();
        if constexpr (std::ranges::sized_range<R>)
        {
            reserve(Index(std::ranges::size(components)));
        }
        for (auto && [logWeight, component] : components)
        {
            add(logWeight, component);
        }
    }

    /** Appends copies of \a other's entries, which must be in the same space, with no parent in the
     * lineage.
     */
    void
    append(MixtureParams const & other)
    {
        m_stamp.touchShape();
        EMRYS_FILTER_ASSERT(other.m_space == m_space && "cannot append a mixture in a different space");
        std::size_t const n = other.m_entries.size();
        assertCapacity(size() + Index(n));
        m_entries.reserve(m_entries.size() + n);
        for (std::size_t i = 0; i < n; ++i)
        {
            m_entries.push_back(other.m_entries[i]);
            m_lineage.pushOrphan();
        }
    }

    /** Calls \a f on each entry as a \c MutableEntry&, returning nothing; reweight() takes one
     * that returns a log scale factor. \a f may modify the entry it is given but not the
     * mixture, which is a programmer error caught when \a f returns. Views of the mixture may
     * not be read afterward.
     */
    template <typename F>
    void
    forEach(F && f)
    {
        forEach(execution::sequential, std::forward<F>(f));
    }

    /** forEach() under \a policy, a for-each over the entries (see \ref Execution). Under a
     * parallel policy \a f runs for several entries at once, so it must read and write nothing
     * but the entry it is given and what no other call writes.
     */
    template <ExecutionPolicy P, typename F>
    void
    forEach(P && policy, F && f)
    {
        static_assert(
            std::is_void_v<std::invoke_result_t<F &, MutableEntry &>>,
            "forEach() takes a callback that returns nothing; one that returns a log scale factor goes to reweight()"
        );
        visitEntries(std::forward<P>(policy), [&](MutableEntry & handle, Entry &) {
            std::invoke(f, handle);
        });
    }

    /** Calls \a f on each entry as a \c MutableEntry&, as forEach() does, and adds what it
     * returns, a log scale factor in a floating-point type, to that entry's log-weight.
     */
    template <typename F>
    void
    reweight(F && f)
    {
        reweight(execution::sequential, std::forward<F>(f));
    }

    /** reweight() under \a policy, as forEach() under it runs. */
    template <ExecutionPolicy P, typename F>
    void
    reweight(P && policy, F && f)
    {
        static_assert(
            std::floating_point<std::remove_cvref_t<std::invoke_result_t<F &, MutableEntry &>>>,
            "reweight() takes a callback that returns a log scale factor, a floating-point number"
        );
        visitEntries(std::forward<P>(policy), [&](MutableEntry & handle, Entry & entry) {
            entry.logWeight += Scalar(std::invoke(f, handle));
        });
    }

    /** Merges \a absorb into \a keep.
     *
     * \a keep's component becomes the collapse of both, weighted by their weights (see
     * ComponentTraits), and its weight the sum of both weights; \a absorb is then
     * removed, by moving the last entry into its place.
     *
     * \returns the index of the merged entry: \a keep, or \a absorb when \a keep was the last
     */
    Index
    merge(Index keep, Index absorb)
    {
        return merge(keep, absorb, merged(keep, absorb));
    }

    /** \returns the component merge() makes of \a first and \a second, without changing the
     * mixture: the collapse of both, weighted by their weights (see ComponentTraits).
     *
     * The merge's weight is the sum of both weights. What a merge costs is the caller's to
     * measure, from the two components and this one: for Gaussians, the weighted divergences
     * \f$ w_i \, \mathrm{KL}(g_i \parallel g_{ij}) + w_j \, \mathrm{KL}(g_j \parallel g_{ij}) \f$
     * from kullbackLeibler() are the cost of \ref Runnalls2007 "Runnalls (2007)".
     */
    [[nodiscard]] ComponentType
    merged(Index first, Index second) const
    {
        assertIndex(first);
        assertIndex(second);
        Index const pair[2] = {first, second};
        EMRYS_FILTER_ASSERT(pair[0] != pair[1] && "cannot merge a component with itself");
        return ComponentTraits<ComponentType>::collapse(
            Index(2),
            [&](Index i) -> ComponentType const & {
                return at(pair[i]).component;
            },
            [&](Index i) {
                return at(pair[i]).logWeight;
            }
        );
    }

    /** Merges \a absorb into \a keep.
     *
     * \a keep's component becomes \a component and its weight becomes the sum of both
     * weights; \a absorb is then removed, as by merge(keep, absorb).
     *
     * \returns the index of the merged entry
     */
    Index
    merge(Index keep, Index absorb, ComponentType component)
    {
        return mergeInto(keep, absorb, std::move(component));
    }

    /** Merges \a absorb into \a keep.
     *
     * \a keep's moments become \a moments and its weight becomes the sum of both
     * weights; \a absorb is then removed, as by merge(keep, absorb). \a moments may refer to
     * either component.
     *
     * \returns the index of the merged entry
     */
    Index
    merge(Index keep, Index absorb, MomentsType moments)
        requires(isGaussian<ComponentType>)
    {
        return mergeInto(keep, absorb, std::move(moments));
    }

    /** Replaces component \a i with children.
     *
     * \a children is a non-empty range of (log relative weight, component) pairs, or of (log
     * relative weight, moments) pairs for %Gaussian components. The parent's weight is divided
     * among the children in proportion to their relative weights. The first child takes the
     * parent's position; the rest are appended. Each child takes the parent's lineage. \a children
     * must not refer to this mixture's components.
     */
    template <std::ranges::forward_range R>
    void
    split(Index i, R && children)
    {
        assertIndex(i);
        m_stamp.touchShape();
        EMRYS_FILTER_ASSERT(!std::ranges::empty(children));
        auto relativeTotal = LogSumExp<Scalar>{};
        for ([[maybe_unused]] auto && [logRelativeWeight, child] : children)
        {
            relativeTotal.add(logRelativeWeight);
        }
        Scalar const logTotal = relativeTotal.value();
        EMRYS_FILTER_ASSERT(isFinite(logTotal) && "relative weights must have a finite, nonzero sum");
        Scalar const logBase = at(i).logWeight - logTotal;
        bool first = true;
        for (auto && [logRelativeWeight, child] : children)
        {
            if (first)
            {
                at(i).logWeight = logBase + logRelativeWeight;
                assignComponent(at(i), child);
                first = false;
            }
            else
            {
                add(logBase + logRelativeWeight, child);
                m_lineage.truncate(size() - 1);
                m_lineage.pushCopyOf(i);
            }
        }
    }

    /** Removes every entry for which \a pred(Entry const &), or \a pred(Index, Entry const &)
     * with the entry's position, is true.
     *
     * \tparam ReturnRemovedWeight  if true, returns the log of the removed fraction
     *         of the total weight. This requires normalizing the weights; if that
     *         fails, nothing is removed and the WeightError is returned.
     */
    template <bool ReturnRemovedWeight = false, typename Pred>
        requires(!ReturnRemovedWeight)
    void
    removeIf(Pred pred)
    {
        removeEntriesIf<false>(std::move(pred));
    }

    /** As above, returning the log of the removed fraction of the total weight, or the
     * WeightError that left the mixture unchanged.
     */
    template <bool ReturnRemovedWeight, typename Pred>
        requires ReturnRemovedWeight
    [[nodiscard]] std::expected<Scalar, WeightError>
    removeIf(Pred pred)
    {
        return removeEntriesIf<true>(std::move(pred));
    }

    /** Removes every component whose normalized log-weight is below \a logThreshold, and normalizes
     * the weights of the rest; the most probable component is always kept, the first of them if
     * several are. The threshold is absolute: a caller who wants it relative to something, such as
     * the most probable component, makes it so. The components keep their order, and the lineage
     * follows them. It reads only the weights, so it works whatever the component, and
     * upkeep::pruneBelow() calls it.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     *
     * \sa pruneTail(), pruneToSize()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneBelow(Scalar logThreshold)
    {
        return pruneWith(internal::PruneBelowRule<Scalar>{logThreshold});
    }

    /** Removes the lightest components, as many as can be while their combined normalized weight
     * stays below \f$ e^{\mathrm{logFraction}} \f$, and normalizes the weights of the rest. The
     * fraction is below one, so a component is always kept. Of components equally heavy, those
     * later in order go first. upkeep::pruneTail() calls it.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     *
     * \sa pruneBelow(), pruneToSize()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneTail(Scalar logFraction)
    {
        return pruneWith(internal::PruneTailRule<Scalar>{logFraction});
    }

    /** Removes the lightest components until at most \a maxSize, at least one, remain, and
     * normalizes the weights of the rest. Of components equally heavy, those later in order go
     * first. upkeep::pruneToSize() calls it.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     *
     * \sa pruneBelow(), pruneTail()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneToSize(Index maxSize)
    {
        return pruneWith(internal::PruneToSizeRule{maxSize});
    }

    /** \returns a read-only view of the subspace named by the tag path \a Path (see idx()), whose
     * space is that subspace with its parameters: a range over the components that gives each
     * one's log-weight and marginal in turn, copying nothing (decisions.md [element-marginals]). It checks this
     * object's shape.
     *
     * \sa class MixtureMarginal
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const &
    {
        return internal::tagMarginal<MixtureMarginal, Path>(*this);
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal() const && = delete;

    /** marginal() of one repetition of a repeated block; see GaussianParams::marginal(element). */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const &
    {
        return internal::elementMarginal<MixtureMarginal, Path>(*this, element);
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal(Index element) const && = delete;

    /** \returns a copy of the distribution of the other coordinates given that the subspace named
     * by the path \a GivenPath equals \a values, in the space of the remaining blocks (see
     * GaussianParams::conditional()).
     *
     * Each component is conditioned as in GaussianParams::conditionalWithEvidence(), and its
     * log-weight gains the log-likelihood of \a values under that component's marginal. The
     * weights are not normalized, so the result's total weight is this mixture's total weight
     * times the marginal density of \a values.
     *
     * \returns the conditional mixture, or GaussianError::NotPositiveDefinite if any component's
     *          covariance of the given block is not positive definite, since that component's
     *          weight would need the density of a block known exactly (decisions.md [innovation-cholesky])
     */
    template <typename GivenPath, typename Derived>
        requires(isGaussian<ComponentType>)
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditionalMixture<GivenPath>(*this, values);
    }

    /** Validates the weights and factorizes every component that has changed.
     *
     * It is \c const because it writes only caches: the components' factorizations and the total
     * weight that views read. So it, and toDensity(), are not safe on one object from two threads
     * at once until a factorization has succeeded and the weights are unchanged since; after that
     * they only read. Threads should share the GaussianMixture that toDensity() returns instead
     * (see \ref Distributions).
     *
     * \returns nothing on success; a WeightError if the mixture is empty or its
     *          weights cannot be normalized; or a ComponentError naming the first
     *          component that failed to factorize.
     *
     * \sa toDensity()
     */
    [[nodiscard]] std::expected<void, MixtureError>
    factorize() const
    {
        return factorizeComponents(*this, internal::NoRepair{});
    }

    /** factorize(), repairing each covariance that is not positive definite by \a repair (see
     * GaussianParams::factorize(R const &)): <tt>m.factorize(roots::ridge())</tt>.
     *
     * Every repair is tried before any is kept, so a component that can't be repaired leaves the
     * mixture as it was.
     *
     * \returns nothing on success; a WeightError if the mixture is empty or its weights cannot
     *          be normalized; or a ComponentError naming the first component whose covariance
     *          could not be repaired.
     *
     * \sa Repair
     */
    template <typename R>
    [[nodiscard]] std::expected<void, MixtureError>
    factorize(R const & repair)
    {
        m_stamp.touchValues();
        return factorizeComponents(*this, repair);
    }

    /** Factorizes as needed (see factorize()).
     *
     * \returns a Mixture that checks this object, or the MixtureError from factorize().
     */
    [[nodiscard]] std::expected<DensityType, MixtureError>
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

    /** toDensity(), repairing covariances that are not positive definite by \a repair (see
     * factorize(R const &)): <tt>m.toDensity(roots::ridge())</tt>.
     *
     * \returns a Mixture that checks this object, or the MixtureError from factorize(R const &)
     */
    template <typename R>
    [[nodiscard]] std::expected<DensityType, MixtureError>
    toDensity(R const & repair) &
    {
        if (auto result = factorize(repair); !result)
        {
            return std::unexpected{result.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    template <typename R>
    void
    toDensity(R const &) && = delete;

private:

    friend DensityType;  ///< the view reads the total weight and essGiven()

    /** \returns ess() given the log of the total weight, \a total, which is finite */
    [[nodiscard]] Scalar
    essGiven(Scalar total) const
    {
        Scalar sumOfSquares = 0;
        for (auto const & entry : m_entries)
        {
            sumOfSquares += Eigen::numext::abs2(Eigen::numext::exp(entry.logWeight - total));
        }
        return Scalar(1) / sumOfSquares;
    }

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** A marginal checks the stamp. */
    template <typename>
    friend class internal::ShapeCheck;

    friend struct internal::MixtureLineageAccess;  ///< filters carry the lineage through their updates

    /** factorize() for \a self, a const or non-const reference to this object, factorizing each
     * component with its factorize(), or its factorize(R const &) by \a repair unless that is
     * internal::NoRepair.
     */
    template <typename Self, typename R>
    [[nodiscard]] static std::expected<void, MixtureError>
    factorizeComponents(Self & self, R const & repair)
    {
        constexpr bool Repairs = !std::is_same_v<R, internal::NoRepair>;
        auto const factorizeOne = [&](auto & component) {
            if constexpr (Repairs)
            {
                return component.factorize(repair);
            }
            else
            {
                return component.factorize();
            }
        };
        auto const total = self.logTotalWeight();
        if (!total)
        {
            return std::unexpected{MixtureError{total.error()}};
        }
        if constexpr (Repairs)
        {
            // Every repair is tried before any is kept, on a copy of a component that needs one,
            // so that a component that can't be repaired leaves the mixture as it was. Repairs are
            // rare, so the copies are too.
            for (Index i = 0; i < self.size(); ++i)
            {
                auto const & component = std::as_const(self.at(i).component);
                if (!component.factorize())
                {
                    auto trial = ComponentType{component};
                    if (auto result = factorizeOne(trial); !result)
                    {
                        return std::unexpected{MixtureError{ComponentError{i, result.error()}}};
                    }
                }
            }
        }
        for (Index i = 0; i < self.size(); ++i)
        {
            if (auto result = factorizeOne(self.at(i).component); !result)
            {
                return std::unexpected{MixtureError{ComponentError{i, result.error()}}};
            }
        }
        // Unchanged weights write nothing, so factorizing parameters that are already factorized
        // only reads, as toDensity() from several threads needs (decisions.md [execution-policies]).
        if (self.m_logTotalWeight != *total)
        {
            self.m_logTotalWeight = *total;
        }
        return {};
    }

    using EntryAllocator = internal::ReBound<Allocator_, Entry>;    ///< allocates the entries
    using ScalarAllocator = internal::ReBound<Allocator_, Scalar>;  ///< allocates the weight scratch

    /** Resizes a space with one resizable block to \a n, and checks that the sizes match. */
    void
    fitSpace(Index n)
    {
        internal::fitValueDimension(m_space, n);
    }

    /** Copies \a moments into \a entry's %Gaussian, normalizing the mean. */
    static void
    assignComponent(Entry & entry, MomentsType const & moments)
        requires(isGaussian<ComponentType>)
    {
        entry.component.setMean(moments.mean());
        entry.component.setCovariance(moments.covarianceLower());
    }

    /** Moves \a moments into \a entry's %Gaussian, normalizing the mean. */
    static void
    assignComponent(Entry & entry, MomentsType && moments)
        requires(isGaussian<ComponentType>)
    {
        entry.component.setMean(std::move(moments.mean()));
        entry.component.setCovariance(std::move(moments.covarianceLower()));
    }

    /** The body of removeIf(), returning the removed weight if \a ReturnRemovedWeight. */
    template <bool ReturnRemovedWeight, typename Pred>
    RemovalResult<ReturnRemovedWeight>
    removeEntriesIf(Pred pred)
    {
        m_stamp.touchShape();
        auto const guard = internal::CallbackGuard{m_stamp};
        auto const removeAtIndex = [&](Index i) {
            bool remove = false;
            if constexpr (std::is_invocable_v<Pred &, Index, Entry const &>)
            {
                remove = static_cast<bool>(std::invoke(pred, i, std::as_const(at(i))));
            }
            else
            {
                remove = static_cast<bool>(std::invoke(pred, std::as_const(at(i))));
            }
            guard.check();
            return remove;
        };
        if constexpr (ReturnRemovedWeight)
        {
            auto const total = logTotalWeight();
            if (!total)
            {
                return std::unexpected{total.error()};
            }
            return compact(removeAtIndex) - *total;
        }
        else
        {
            compact(removeAtIndex);
        }
    }

    /** The loop of forEach() and reweight(): calls \a visit with each entry's MutableEntry and the
     * entry, under \a policy, checking after each that the callback left the mixture alone.
     */
    template <ExecutionPolicy P, typename Visit>
    void
    visitEntries(P && policy, Visit const & visit)
    {
        m_stamp.touchValues();
        auto const guard = internal::CallbackGuard{m_stamp};
        internal::forEachIndex(std::forward<P>(policy), size(), [&](Index i) {
            auto & entry = m_entries[std::size_t(i)];
            auto handle = MutableEntry{entry, i};
            visit(handle, entry);
            guard.check();
        });
    }

    /** Copies \a component into \a entry. */
    static void
    assignComponent(Entry & entry, ComponentType const & component)
    {
        entry.component = component;
    }

    /** Moves \a component into \a entry. */
    static void
    assignComponent(Entry & entry, ComponentType && component)
    {
        entry.component = std::move(component);
    }

    /** The body of merge(): \a keep's component is assigned \a replacement, a component or
     * moments, its weight becomes the sum, and \a absorb is removed.
     */
    template <typename Replacement>
    Index
    mergeInto(Index k, Index a, Replacement && replacement)
    {
        assertIndex(k);
        assertIndex(a);
        m_stamp.touchShape();
        EMRYS_FILTER_ASSERT(k != a && "cannot merge a component with itself");
        auto weight = LogSumExp<Scalar>{};
        weight.add(at(k).logWeight);
        weight.add(at(a).logWeight);
        m_lineage.merge(k, a, at(k).logWeight, at(a).logWeight);
        at(k).logWeight = weight.value();
        assignComponent(at(k), std::forward<Replacement>(replacement));
        bool const keepWasLast = k == size() - 1;
        removeAt(a);
        return keepWasLast ? a : k;
    }

    /** A programmer error if \a count components would exceed the capacity, when there is one. */
    void
    assertCapacity([[maybe_unused]] Index count) const
    {
        EMRYS_FILTER_ASSERT(
            (!m_capacity || count <= *m_capacity) && "the mixture would hold more components than its capacity"
        );
    }

    /** Reserves the entries and the scratch for \a count components. */
    void
    reserveScratch(Index count)
    {
        auto const n = std::size_t(count);
        m_entries.reserve(n);
        m_weightScratch.reserve(n);
    }

    /** A programmer error unless \a i indexes an entry. */
    void
    assertIndex([[maybe_unused]] Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < size() && "component index out of range");
    }

    /** \returns entry \a i, to change it */
    [[nodiscard]] Entry &
    at(Index i)
    {
        return m_entries[std::size_t(i)];
    }

    /** \returns entry \a i */
    [[nodiscard]] Entry const &
    at(Index i) const
    {
        return m_entries[std::size_t(i)];
    }

    /** \returns entry \a i, to change it, counting the change in the stamp */
    [[nodiscard]] Entry &
    mutableAt(Index i)
    {
        assertIndex(i);
        m_stamp.touchValues();
        return at(i);
    }

    /** Removes entry \a i by moving the last entry into its place. */
    void
    removeAt(Index i)
    {
        if (i != size() - 1)
        {
            at(i) = std::move(m_entries.back());
        }
        m_entries.pop_back();
        m_lineage.removeAt(i);
    }

    /** The body of pruneBelow(), pruneTail(), and pruneToSize(): removes the components that
     * \a rule's cut removes, sorting their weights in the scratch where it needs to, and normalizes
     * the rest. A NaN weight has no place in the order, so logTotalWeight() rejects it first.
     *
     * \returns the log of the removed fraction of the total weight, or the WeightError of
     *          logTotalWeight() with nothing changed
     */
    template <typename Rule>
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneWith(Rule const & rule)
    {
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        m_stamp.touchShape();
        auto cut = rule(
            size(),
            [&](Index i) {
                return at(i).logWeight;
            },
            *total,
            m_weightScratch
        );
        Scalar const removed = compact([&](Index i) {
            return cut.removes(at(i).logWeight);
        });
        // The kept components include one of positive weight, so their total is finite.
        scaleWeights(-*(logTotalWeight()));
        return removed - *total;
    }

    /** Removes entries whose index satisfies \a removeAtIndex, keeping the rest in order.
     *
     * \returns the log of the sum of the removed weights
     */
    template <typename RemoveAtIndex>
    Scalar
    compact(RemoveAtIndex removeAtIndex)
    {
        auto removed = LogSumExp<Scalar>{};
        Index out = 0;
        for (Index i = 0; i < size(); ++i)
        {
            if (removeAtIndex(i))
            {
                removed.add(at(i).logWeight);
                continue;
            }
            if (out != i)
            {
                at(out) = std::move(at(i));
                m_lineage.moveRow(i, out);
            }
            ++out;
        }
        m_entries.erase(m_entries.begin() + out, m_entries.end());
        m_lineage.truncate(out);
        return removed.value();
    }

    internal::Buffer<Entry, Allocator_> m_entries;     ///< the entries, in order
    EMRYS_FILTER_NO_UNIQUE_ADDRESS SpaceType m_space;  ///< the space every component is in
    std::optional<Index> m_capacity;                   ///< the most components allowed, if set

    // Scratch, kept between calls: the weights pruneTail() and pruneToSize() order.
    internal::Buffer<Scalar, Allocator_> m_weightScratch;  ///< a copy of the log-weights being ordered

    LineageType m_lineage;  ///< how the components descend from those at the last startLineage()

    // Filled by the const factorize() for use by views.
    mutable Scalar m_logTotalWeight{};  ///< the log of the total weight at the last factorize()

    internal::Stamp m_stamp;  ///< what views check [view-stamp]
};

/** \ingroup Distributions
 * \brief A Mixture of Gaussians in covariance form, the spelling for the common case
 * (decisions.md [component-first]): \f$ \sum_i w_i \, \mathcal{N}(x; \mu_i, \Sigma_i) \f$.
 *
 * \tparam Space_      space of the components (see \ref Spaces)
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the entries and scratch (see \ref Allocation)
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
using GaussianMixtureParams = MixtureParams<Gaussian<Space_, Scalar_>, Allocator_>;

/** \ingroup Distributions
 * \brief The density of a Mixture whose components have been factorized: a read-only view.
 *
 * Represents the normalized density
 * \f[ p(x) = \frac{\sum_i w_i \, p_i(x)}{\sum_i w_i}, \f]
 * with \f$ p_i \f$ the density of component \f$ i \f$, read from its factorization under this
 * view's check (see ComponentTraits).
 *
 * Obtained from MixtureParams::toDensity(). It checks the source for changes, as its copies do:
 * reading it after the source has changed is a programmer error (decisions.md [view-stamp]). A Mixture must
 * not outlive its source.
 *
 * \sa class MixtureParams
 */
template <internal::ComponentOrInfoForm Component_, typename Allocator_ = DefaultAllocator>
class Mixture
{
public:

    using ParamsType = MixtureParams<Component_, Allocator_>;                ///< the parameters it reads
    using ComponentType = Params<Component_>;                                ///< a component
    using SpaceType = typename ParamsType::SpaceType;                        ///< the components' space
    using allocator_type = Allocator_;                                       ///< the parameters' allocator
    using Entry = typename ParamsType::Entry;                                ///< one weighted component
    using const_iterator = typename ParamsType::const_iterator;              ///< iterates the entries
    using MomentsType = typename ParamsType::MomentsType;                    ///< a mean and covariance together
    using Scalar = typename ParamsType::Scalar;                              ///< the scalar type
    using Index = typename ParamsType::Index;                                ///< Eigen's index type
    using VectorType = typename ParamsType::VectorType;                      ///< a value, such as a mean
    using DifferenceVectorType = typename ParamsType::DifferenceVectorType;  ///< a difference
    using MatrixType = typename ParamsType::MatrixType;                      ///< a matrix over differences

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = ParamsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = ParamsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = ParamsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = ParamsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = false;  ///< a view: it checks its parameters

    /** \returns the number of components */
    [[nodiscard]] Index
    size() const
    {
        return source().size();
    }

    /** \returns the number of coordinates of every component's mean */
    [[nodiscard]] Index
    valueDimension() const
    {
        return source().valueDimension();
    }

    /** \returns the number of rows and columns of every component's covariance */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return source().differenceDimension();
    }

    /** \returns the source's space */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return source().space();
    }

    /** \returns the source's allocator */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return source().get_allocator();
    }

    /** \returns an iterator to the first entry */
    [[nodiscard]] const_iterator
    begin() const
    {
        return source().begin();
    }

    /** \returns an iterator past the last entry */
    [[nodiscard]] const_iterator
    end() const
    {
        return source().end();
    }

    /** \returns entry \a i of the source, which must be less than size() */
    [[nodiscard]] typename ParamsType::Entry const &
    operator[](Index i) const
    {
        return source()[i];
    }

    /** \returns the log of the sum of the (unnormalized) component weights */
    [[nodiscard]] Scalar
    logTotalWeight() const
    {
        return source().m_logTotalWeight;
    }

    /** \returns the effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$ of the normalized
     * weights, between 1 and size()
     */
    [[nodiscard]] Scalar
    ess() const
    {
        return source().essGiven(logTotalWeight());
    }

    /** \returns the index of the most probable component, the one with the largest weight; ties
     * go to the lowest. The factorization checked the weights, so none is NaN.
     *
     * \sa MixtureParams::mostProbable()
     */
    [[nodiscard]] Index
    mostProbable() const
    {
        return source().mostProbable();
    }

    /** \returns the mixture as one component, moment-matched over the components with their
     * normalized weights (see ComponentTraits): for %Gaussian components the %Gaussian
     * with moments(), and for hybrid components the hybrid whose mode probabilities are the
     * weighted sums and whose conditionals are collapsed per mode
     */
    [[nodiscard]] ComponentType
    collapsed() const
    {
        return ComponentTraits<ComponentType>::collapse(
            size(),
            [&](Index i) -> ComponentType const & {
                return (*this)[i].component;
            },
            [&](Index i) {
                return (*this)[i].logWeight;
            }
        );
    }

    /** \returns the mean and covariance of the mixture,
     * \f$ \mu = \sum_i \hat w_i \mu_i \f$ and
     * \f$ \Sigma = \sum_i \hat w_i \left(\Sigma_i + (\mu_i - \mu)(\mu_i - \mu)^T\right) \f$,
     * where \f$ \hat w_i \f$ are the normalized weights. In a non-linear space the mean is
     * a tangent-space mean and the deviations are \f$ \mu_i \boxminus \mu \f$.
     *
     * Only the lower triangle of each component covariance is read; the result is symmetric. It
     * allocates nothing sized by the number of components.
     */
    [[nodiscard]] MomentsType
    moments() const
    {
        return ComponentTraits<ComponentType>::moments(collapsed());
    }

    /** \returns the log of the density at \a x: the log-sum-exp of each component's log-weight plus
     * its log-density, less the log of the total weight (LogSumExp). It reads the
     * components' factorizations under this view's check and makes none of its own, so it may be
     * evaluated at several points at once.
     *
     * \sa density()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "logDensity() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        auto sum = LogSumExp<Scalar>{};
        for (auto const & entry : *this)
        {
            if (entry.logWeight == -Eigen::NumTraits<Scalar>::infinity())
            {
                continue;
            }
            sum.add(entry.logWeight + ComponentTraits<ComponentType>::logDensity(entry.component, x));
        }
        return sum.value() - logTotalWeight();
    }

    /** \returns the density \f$ p(x) \f$ at \a x
     *
     * \warning Underflows easily in high dimensions; prefer logDensity().
     *
     * \sa logDensity()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** marginal() of the subspace named by the tag path \a Path; see MixtureParams::marginal(). */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::tagMarginal<MixtureMarginal, Path>(*this);
    }

    /** marginal() of one repetition of a repeated block; see GaussianParams::marginal(element). */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Index element) const
    {
        return internal::elementMarginal<MixtureMarginal, Path>(*this, element);
    }

    /** \returns a copy of the conditional distribution; see MixtureParams::conditional(). */
    template <typename GivenPath, typename Derived>
        requires(isGaussian<ComponentType>)
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditionalMixture<GivenPath>(*this, values);
    }

private:

    friend ParamsType;  ///< the parameters construct views

    /** Views \a source, whose components are factorized, and records its stamp; only
     * MixtureParams::toDensity() calls this.
     */
    explicit Mixture(ParamsType const & source)
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

/** \ingroup Distributions
 * \brief The Mixture of a GaussianMixtureParams, the spelling for the common case (decisions.md
 * [component-first]).
 *
 * \tparam Space_      space of the components (see \ref Spaces)
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  the parameters' allocator (see \ref Allocation)
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
using GaussianMixture = Mixture<Gaussian<Space_, Scalar_>, Allocator_>;

namespace internal
{

/** How a marginal of a container of components reads the container (decisions.md [element-marginals]): the number
 * of components, each one's log-weight and component, and the container of the components'
 * marginals that the marginal's copy() makes. Specialized for MixtureParams and Mixture here, and
 * for Hybrid in Hybrid.h.
 */
template <typename Source>
struct ElementsOf;

/** ElementsOf of a mixture's parameters, and through the subclass below of its density, which
 * iterates the same entries.
 */
template <Component Component_, typename Allocator_>
struct ElementsOf<MixtureParams<Component_, Allocator_>>
{
    /** \returns the log-weight of entry \a i of \a source */
    template <typename Source>
    [[nodiscard]] static auto
    logWeight(Source const & source, Eigen::Index i)
    {
        return source[i].logWeight;
    }

    /** \returns the component of entry \a i of \a source */
    template <typename Source>
    [[nodiscard]] static auto const &
    component(Source const & source, Eigen::Index i)
    {
        return source[i].component;
    }

    /** The container of components whose parameters are \a ComponentParams, which copy() makes. */
    template <typename ComponentParams>
    using CopyType = MixtureParams<Density<ComponentParams>, Allocator_>;

    /** \returns a mixture of the parameters of each of \a marginal's components, with their
     * log-weights, in storage from \a allocator
     */
    template <typename Source, typename Marginal>
    [[nodiscard]] static auto
    copy(Source const &, Marginal const & marginal, Allocator_ const & allocator)
    {
        auto result = CopyType<typename Marginal::ComponentMarginalType::ParamsType>{marginal.space(), allocator};
        result.reserve(marginal.size());
        for (auto const & element : marginal)
        {
            result.add(element.logWeight, element.component.copy());
        }
        return result;
    }
};

/** ElementsOf of a mixture's density, which reads the entries of its parameters. */
template <Component Component_, typename Allocator_>
struct ElementsOf<Mixture<Component_, Allocator_>> : ElementsOf<MixtureParams<Component_, Allocator_>>
{};

}  // namespace internal

/** \ingroup Distributions
 * \brief Read-only view of a subset of the coordinates of every component of a mixture or a
 * hybrid, which iterates the components and gives each one's marginal in turn (decisions.md [element-marginals]).
 *
 * Obtained from marginal() of MixtureParams, Mixture, or Hybrid, and held by \c auto: its Selection
 * is internal, so its type is never spelled, and a function that takes one takes it as a template
 * parameter. It is a random-access range whose
 * elements, made when read, hold a component's log-weight and the marginal of the component, as the
 * entries of a mixture hold its log-weight and the component: a GaussianMarginal for a %Gaussian, and
 * a MixtureMarginal of the modes for a Hybrid. Nothing is copied, and nothing allocates: the
 * selected coordinates are resolved once, here, and every component's marginal borrows them, so an
 * element must not outlive this view. Of parameters it checks their shape, each time an element is
 * made; of a Mixture it holds a copy of the view, and so the view's check. Weights are those of the
 * source.
 *
 * \code
 * for (auto const & [logWeight, position] : mixture.marginal<Position>())
 * {
 *     position.mean();   // the component's mean position, read in place
 * }
 * \endcode
 *
 * \tparam Source     a MixtureParams, a Mixture, or a Hybrid
 * \tparam Selection  an internal::Coordinates of the index sets of the selected block, from
 *                    idx() for the means and diffIdx() for the covariances, or an
 *                    internal::SharedSelection of a selection resolved already
 * \tparam SubSpace   space of the selected block
 */
template <typename Source, typename Selection, typename SubSpace>
class MixtureMarginal
{
public:

    using Scalar = typename Source::Scalar;  ///< the scalar type
    using Index = Eigen::Index;              ///< Eigen's index type

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = valueDimensionAtCompileTime<SubSpace>;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = maxValueDimensionAtCompileTime<SubSpace>;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = valueDimensionAtCompileTime<DifferenceSpace<SubSpace>>;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime =
        maxValueDimensionAtCompileTime<DifferenceSpace<SubSpace>>;

    using SpaceType = SubSpace;                              ///< the selected block's space
    using allocator_type = typename Source::allocator_type;  ///< the source's allocator
    using ComponentType = typename Source::ComponentType;    ///< a component of the source

    /// the selection, resolved
    using ResolvedType = typename internal::SelectionHolder<Source, Selection, SubSpace>::ResolvedType;

    /** The selected mean entries' indices. */
    using IndexVectorType = decltype(ResolvedType::indices);

    /** The selected covariance rows' indices. */
    using DifferenceIndexVectorType = decltype(ResolvedType::differenceIndices);

    /** The marginal of one component, which borrows this view's selection. */
    using ComponentMarginalType = typename ComponentTraits<
        ComponentType>::template MarginalType<internal::SharedSelection<ResolvedType>, SubSpace>;

    /** One component's log-weight, as in the source, and its marginal. */
    struct Element
    {
        Scalar logWeight;                 ///< the component's log-weight
        ComponentMarginalType component;  ///< the component's marginal
    };

    using value_type = Element;                                         ///< an element, made when read
    using const_iterator = internal::ElementIterator<MixtureMarginal>;  ///< iterates the elements
    using iterator = const_iterator;                                    ///< iterates the elements; they are read-only

    /// what copy() returns: the container of the components' marginals, as parameters
    using ParamsType =
        typename internal::ElementsOf<Source>::template CopyType<typename ComponentMarginalType::ParamsType>;

    /** Selects \a selection's coordinates, which form \a space. */
    MixtureMarginal(Source const & source, Selection const & selection, SubSpace space)
        requires(!internal::isSharedSelection<Selection>)
        : m_holder{source}
        , m_selection{source, selection, std::move(space)}
    {}

    /** Selects the coordinates of \a source that \a shared resolved, borrowing them. */
    MixtureMarginal(Source const & source, Selection shared)
        requires(internal::isSharedSelection<Selection>)
        : m_holder{source}
        , m_selection{shared}
    {}

    /** \returns the selected block's space, with its parameters and size */
    [[nodiscard]] SubSpace const &
    space() const
    {
        return m_selection.get().space;
    }

    /** \returns the source's allocator */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return source().get_allocator();
    }

    /** \returns the number of components */
    [[nodiscard]] Index
    size() const
    {
        return source().size();
    }

    /** \returns the number of coordinates of every marginal component's mean */
    [[nodiscard]] Index
    valueDimension() const
    {
        return indices().size();
    }

    /** \returns the number of rows and columns of every marginal component's covariance */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return differenceIndices().size();
    }

    /** \returns the source's mean entries this marginal selects */
    [[nodiscard]] IndexVectorType const &
    indices() const
    {
        return m_selection.get().indices;
    }

    /** \returns the source's covariance rows this marginal selects */
    [[nodiscard]] DifferenceIndexVectorType const &
    differenceIndices() const
    {
        return m_selection.get().differenceIndices;
    }

    /** \returns component \a i's log-weight and marginal, which must not outlive this view */
    [[nodiscard]] Element
    operator[](Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < size() && "component index out of range");
        Source const & from = source();
        return Element{
            Elements::logWeight(from, i),
            ComponentMarginalType{
                Elements::component(from, i),
                internal::SharedSelection<ResolvedType>{&m_selection.get()}
            }
        };
    }

    /** \returns an iterator to the first component's element */
    [[nodiscard]] const_iterator
    begin() const
    {
        return const_iterator{*this, 0};
    }

    /** \returns an iterator past the last component's element */
    [[nodiscard]] const_iterator
    end() const
    {
        return const_iterator{*this, size()};
    }

    /** \returns an independent container of the parameters of each component's marginal: a
     * MixtureParams with the source's weights, or a Hybrid with its modes, with the source's
     * allocator
     */
    [[nodiscard]] ParamsType
    copy() const
    {
        return copy(get_allocator());
    }

    /** copy() with storage from \a allocator. */
    [[nodiscard]] ParamsType
    copy(allocator_type const & allocator) const
    {
        return Elements::copy(source(), *this, allocator);
    }

private:

    using Elements = internal::ElementsOf<Source>;  ///< how the source's components are read

    /** \returns the viewed source */
    [[nodiscard]] Source const &
    source() const
    {
        return m_holder.get();
    }

    internal::HolderFor<Source> m_holder;  ///< the check of parameters, or a copy of a view
    internal::SelectionHolder<Source, Selection, SubSpace> m_selection;  ///< the selected coordinates and their space
};

/** \ingroup Distributions
 * \returns \a n draws from \a mixture, one per column.
 *
 * Each draw picks a component in proportion to its weight using \a resampler, then
 * draws \f$ \mu_c \boxplus L_c z \f$ from it, with \f$ z \sim \mathcal{N}(0, I) \f$ and
 * \f$ \Sigma_c = L_c L_c^T \f$: [sampling a mixture](https://en.wikipedia.org/wiki/Mixture_distribution#Sampling)
 * by composition. The default resamplers::Multinomial makes the draws
 * independent; a lower-variance resampler such as resamplers::Systematic gives each
 * component a count closer to its expected share.
 *
 * \sa drawSamplesWithLogDensity(GaussianMixture const &, Eigen::Index, G &, R const &), \ref
 * Resamplers
 */
template <
    typename Scalar,
    typename Space,
    typename Allocator,
    std::uniform_random_bit_generator G,
    typename R = resamplers::Multinomial>
    requires Resampler<R, Scalar, G>
[[nodiscard]] SampleSet<Space, Scalar>
drawSamples(
    Mixture<Gaussian<Space, Scalar>, Allocator> const & mixture,
    Eigen::Index n,
    G & rng,
    R const & resampler = {}
)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using SampleSetType = SampleSet<Space, Scalar>;
    auto weights = internal::bufferOf<Scalar>(mixture.get_allocator());
    weights.reserve(std::size_t(mixture.size()));
    for (auto const & entry : mixture)
    {
        weights.push_back(Eigen::numext::exp(entry.logWeight - mixture.logTotalWeight()));
    }
    auto ancestors = internal::bufferOf<Eigen::Index>(mixture.get_allocator());
    ancestors.resize(std::size_t(n));
    resampler(std::span<Scalar const>{weights}, std::span<Eigen::Index>{ancestors}, rng);

    using Normals = SampleSet<DifferenceSpace<Space>, Scalar>;
    Normals const noise = internal::standardNormal<Normals>(mixture.differenceDimension(), n, rng);
    auto samples = SampleSetType{mixture.valueDimension(), n};
    for (Eigen::Index j = 0; j < n; ++j)
    {
        auto const gaussian = mixture[ancestors[std::size_t(j)]].component.toDensity();
        samples.col(j) = Emrys::plus(
            mixture.space(),
            gaussian->mean(),
            internal::multiply(internal::asTriangular<Eigen::Lower>(gaussian->llt().matrixLLT()), noise.col(j))
        );
    }
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a mixture, a component picked in proportion to its weight and then a
 * draw from it, in a vector sized by the space: the stream of draws that drawSample() of a
 * %Gaussian is. Its draws are independent, as resamplers::Multinomial makes those of drawSamples(); a
 * lower-variance resampler spreads a batch over the components, which one draw at a time cannot.
 */
template <typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(Mixture<Gaussian<Space, Scalar>, Allocator> const & mixture, G & rng)
{
    Eigen::Index const picked = internal::drawIndex<Scalar>(
        mixture.size(),
        [&](Eigen::Index k) {
            return Eigen::numext::exp(Scalar(mixture[k].logWeight - mixture.logTotalWeight()));
        },
        rng
    );
    auto const gaussian = mixture[picked].component.toDensity();
    return drawSample(*gaussian, rng);
}

/** \ingroup Distributions
 * \returns \a n draws from \a mixture, one per column, with the log-density of
 * \a mixture at each.
 *
 * Draws exactly as drawSamples() does. Each log-density is a full
 * Mixture::logDensity() evaluation, so the total is
 * \f$ O(n \cdot \mathrm{size}() \cdot d^2) \f$.
 *
 * \sa drawSamples(GaussianMixture const &, Eigen::Index, G &, R const &)
 */
template <
    typename Scalar,
    typename Space,
    typename Allocator,
    std::uniform_random_bit_generator G,
    typename R = resamplers::Multinomial>
    requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(
    Mixture<Gaussian<Space, Scalar>, Allocator> const & mixture,
    Eigen::Index n,
    G & rng,
    R const & resampler = {}
)
{
    return drawSamplesWithLogDensity(execution::sequential, mixture, n, rng, resampler);
}

/** \ingroup Distributions
 * drawSamplesWithLogDensity() under \a policy: the draws are taken as the sequential form takes
 * them, and their log-densities are a for-each over the draws (see \ref Execution), so the result
 * is Sequential's.
 */
template <
    ExecutionPolicy P,
    typename Scalar,
    typename Space,
    typename Allocator,
    std::uniform_random_bit_generator G,
    typename R = resamplers::Multinomial>
    requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(
    P && policy,
    Mixture<Gaussian<Space, Scalar>, Allocator> const & mixture,
    Eigen::Index n,
    G & rng,
    R const & resampler = {}
)
{
    auto result = SamplesWithLogDensity<Space, Scalar>{drawSamples(mixture, n, rng, resampler), {}};
    result.logDensity.resize(n);
    internal::forEachIndex(std::forward<P>(policy), n, [&](Eigen::Index j) {
        result.logDensity(j) = mixture.logDensity(result.samples.col(j));
    });
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a mixture, as drawSample() draws it, with the log-density of \a mixture
 * at it, a full Mixture::logDensity() evaluation, in a vector sized by the space: the stream of
 * draws that drawSampleWithLogDensity() of a %Gaussian is.
 */
template <typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(Mixture<Gaussian<Space, Scalar>, Allocator> const & mixture, G & rng)
{
    auto result = SampleWithLogDensity<Space, Scalar>{drawSample(mixture, rng), 0};
    result.logDensity = mixture.logDensity(result.sample);
    return result;
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a mixture: its count and dimension, the mean and covariance of its collapse, and
 * each component with its normalized weight, or with its log-weight where the weights don't
 * normalize.
 */
template <typename Component_, typename Allocator_>
void
describe(std::ostream & stream, MixtureParams<Component_, Allocator_> const & mixture, Indent indent)
{
    using Scalar = typename MixtureParams<Component_, Allocator_>::Scalar;
    stream << indent << "mixture of ";
    printCount(stream, mixture.size(), "component");
    printDimension(stream, mixture.valueDimension(), mixture.differenceDimension());
    auto const logTotal = mixture.logTotalWeight();
    if (logTotal)
    {
        Params<Component_> const collapsed = ComponentTraits<Params<Component_>>::collapse(
            mixture.size(),
            [&](Eigen::Index i) -> Params<Component_> const & {
                return mixture[i].component;
            },
            [&](Eigen::Index i) {
                return mixture[i].logWeight;
            }
        );
        printMoments(stream, indent.deeper(), ComponentTraits<Params<Component_>>::moments(collapsed));
    }
    else
    {
        stream << '\n';
        describe(stream, logTotal.error(), indent.deeper());
    }
    Eigen::Index i = 0;
    for (auto const & entry : mixture)
    {
        stream << '\n' << indent.deeper() << "component " << i << ", ";
        if (logTotal)
        {
            stream << "weight " << Eigen::numext::exp(Scalar(entry.logWeight - *logTotal));
        }
        else
        {
            stream << "log-weight " << entry.logWeight;
        }
        stream << ":\n";
        describe(stream, entry.component, indent.deeper().deeper());
        ++i;
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
