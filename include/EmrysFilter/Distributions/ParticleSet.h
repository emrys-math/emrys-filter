#pragma once

/** \file
 * The particle set: ParticleSet, the weighted samples, with resampling, upkeep of
 * weights and particles, insert() and erase() on a resizable block, and a kernel bandwidth; its
 * marginal view ParticleSetMarginal; and Silverman's bandwidth. KernelDensity.h completes
 * toDensity() and is included at the end. The particle filter's state [upkeep]. Read
 * KernelDensity.h next.
 */

#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Cholesky>

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
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>

namespace Emrys
{

template <Space Space_, RealScalar Scalar_, typename Allocator_>
class ParticleSet;

template <Space Space_, RealScalar Scalar_, typename Allocator_>
class KernelDensity;

template <typename Source, typename Selection, typename SubSpace>
class ParticleSetMarginal;

namespace internal
{

/** Write access to a particle set's log-weights for an update that leaves them normalized, so that
 * the set records their log total as zero rather than summing them again: ParticleFilter::update().
 */
struct ParticleSetAccess
{
    /** Sets the log-weights of \a set to \a logWeights, whose log total the caller has made zero. */
    template <typename Set, typename Derived>
    static void
    setNormalizedLogWeights(Set & set, Eigen::DenseBase<Derived> const & logWeights)
    {
        set.setLogWeights(logWeights);
        set.m_logTotalWeight = typename Set::Scalar(0);
    }

    /** Prunes \a set by \a rule, as its pruneBelow(), pruneTail(), and pruneToSize() do, calling
     * \a onKept with the position of each particle kept, in order, so that a Rao-Blackwellized
     * state keeps the same particles' Gaussians.
     */
    template <typename Set, typename Rule, typename OnKept>
    [[nodiscard]] static std::expected<typename Set::Scalar, WeightError>
    prune(Set & set, Rule const & rule, OnKept const & onKept)
    {
        return set.pruneParticles(rule, onKept);
    }
};

}  // namespace internal

namespace internal
{

/** Silverman's rule-of-thumb bandwidth for the columns of \a states, from their
 * covariance in \a space. Does not validate the weights; invalid weights give a
 * non-finite result. See Emrys::silvermanBandwidth() for the formula and its source.
 */
template <typename MomentsType, ExecutionPolicy P, typename Space, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] typename MomentsType::MatrixType
silvermanBandwidth(
    P && policy,
    Space const & space,
    Eigen::MatrixBase<StatesDerived> const & states,
    Eigen::ArrayBase<WeightsDerived> const & logWeights
)
{
    using Scalar = typename MomentsType::Scalar;
    Scalar const logTotal = logSumExp<Scalar>(policy, logWeights.size(), [&](Eigen::Index i) {
        return logWeights.derived()(i);
    });
    Scalar const d = Scalar(differenceDimension(space));
    Scalar const factor = Eigen::numext::pow(
        Scalar(4) / (ess(policy, logWeights, logTotal) * (d + Scalar(2))),
        Scalar(2) / (d + Scalar(4))
    );
    return factor * weightedMomentsIn<MomentsType>(policy, space, states, logWeights, logTotal).covarianceLower();
}

/** silvermanBandwidth() under Sequential. */
template <typename MomentsType, typename Space, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] typename MomentsType::MatrixType
silvermanBandwidth(
    Space const & space,
    Eigen::MatrixBase<StatesDerived> const & states,
    Eigen::ArrayBase<WeightsDerived> const & logWeights
)
{
    return silvermanBandwidth<MomentsType>(execution::sequential, space, states, logWeights);
}

/** Shared implementation of the particle set and KernelDensity conditionals.
 *
 * Each kernel \f$ \mathcal{N}(x_i, H) \f$ is conditioned on the entries \a given
 * equal to \a values: its center becomes \f$ x_{a,i} + K (v - x_{b,i}) \f$ with
 * \f$ K = H_{ab} H_{bb}^{-1} \f$, its weight gains \f$ \log \mathcal{N}(v; x_{b,i}, H_{bb}) \f$,
 * and the shared bandwidth becomes \f$ H_{aa} - K H_{ba} \f$. The result takes the source's
 * allocator. Each kernel's conditional is the %Gaussian one of internal::conditionMoments(), with
 * the solve against \f$ H_{bb} \f$ shared by every kernel, but for the covariance reset
 * (decisions.md [covariance-reset]): the kernels' centers move by different steps and share one bandwidth, so the
 * bandwidth is not reset, which is exact where the target space has an identity reset.
 *
 * \returns the conditional set, or GaussianError::NotPositiveDefinite if \f$ H_{bb} \f$ is not
 *          positive definite, as a kernel needs it to be to have a density (decisions.md [innovation-cholesky])
 */
template <
    typename Source,
    typename BandwidthDerived,
    typename Target,
    typename Given,
    typename Derived,
    typename TargetSpace,
    typename GivenSpace>
[[nodiscard]] auto
conditionalParticlesIn(
    Source const & source,
    Eigen::MatrixBase<BandwidthDerived> const & bandwidth,
    Target const & target,
    Given const & given,
    Eigen::MatrixBase<Derived> const & values,
    TargetSpace const & targetSpace,
    GivenSpace const & givenSpace
)
{
    using Scalar = typename Source::Scalar;
    EMRYS_FILTER_ASSERT(distinctIndices(target.values, given.values) && "target and given indices must be distinct");
    EMRYS_FILTER_ASSERT(values.size() == given.values.size());
    constexpr int TargetSize = valueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>;
    constexpr int TargetMaxSize = maxValueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>;
    constexpr int GivenSize = valueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>;
    constexpr int GivenMaxSize = maxValueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>;
    using GivenMatrix = MatrixOf<Scalar, GivenSize, GivenSize, GivenMaxSize, GivenMaxSize>;
    using CrossMatrix = MatrixOf<Scalar, TargetSize, GivenSize, TargetMaxSize, GivenMaxSize>;
    using GivenOffsets = SampleSet<DifferenceSpace<GivenSpace>, Scalar>;
    using TargetStates = SampleSet<TargetSpace, Scalar>;

    using Result = ParticleSet<TargetSpace, Scalar, typename Source::allocator_type>;

    auto const llt =
        Eigen::LLT<GivenMatrix>{GivenMatrix{lowerSymmetricBlock(bandwidth, given.differences, given.differences)}};
    if (llt.info() != Eigen::Success)
    {
        return std::expected<Result, GaussianError>{std::unexpected{GaussianError::NotPositiveDefinite}};
    }
    CrossMatrix const cross = lowerSymmetricBlock(bandwidth, target.differences, given.differences);
    CrossMatrix const gain = transposedSolve<CrossMatrix>(llt, cross.transpose());

    GivenOffsets offsets = Emrys::minus(givenSpace, source.states()(given.values, Eigen::placeholders::all), values);
    GivenOffsets solved = offsets;
    solveInPlace(llt, solved);
    Eigen::Array<Scalar, 1, Eigen::Dynamic> const squaredDistance = (offsets.array() * solved.array()).colwise().sum();
    Scalar const log2Pi = Emrys::log2Pi<Scalar>();
    Scalar const logNormalizer =
        Scalar(-0.5)
        * (Scalar(given.differences.size()) * log2Pi + Scalar(2) * llt.matrixLLT().diagonal().array().log().sum());

    TargetStates const centers =
        Emrys::plus(targetSpace, source.states()(target.values, Eigen::placeholders::all), -(gain * offsets));
    auto result = Result{
        targetSpace,
        centers,
        Eigen::Array<Scalar, Eigen::Dynamic, 1>{
            source.logWeights() + logNormalizer - Scalar(0.5) * squaredDistance.transpose()
        },
        source.get_allocator()
    };
    result.setBandwidth(
        lowerSymmetricBlock(bandwidth, target.differences, target.differences) - multiply(gain, cross.transpose())
    );
    return std::expected<Result, GaussianError>{std::move(result)};
}

/** Shared implementation of conditional<GivenPath>(); see tagConditional(). */
template <typename GivenPath, typename Source, typename BandwidthDerived, typename Derived>
[[nodiscard]] auto
tagConditionalParticles(
    Source const & source,
    Eigen::MatrixBase<BandwidthDerived> const & bandwidth,
    Eigen::MatrixBase<Derived> const & values
)
{
    return tagConditional<GivenPath>(
        source.space(),
        [&](auto const & target, auto const & given, auto const & targetSpace, auto const & givenSpace) {
            return conditionalParticlesIn(source, bandwidth, target, given, values, targetSpace, givenSpace);
        }
    );
}

}  // namespace internal

/** \ingroup Distributions
 * \brief A set of weighted samples: the empirical distribution
 * \f[ \hat p(x) = \sum_i \hat w_i \, \delta(x - x_i), \qquad \hat w_i = w_i / \textstyle\sum_j w_j. \f]
 *
 * The weighted particle approximation of \ref Arulampalam2002 "Arulampalam et al. (2002)", §II,
 * kept as a matrix of states and an array of log-weights.
 *
 * Particle states are the columns of states(), with valueDimension() rows; their log-weights are
 * logWeights(). Both are maps over storage the set keeps through its allocator (see
 * \ref Allocation), valid until the set's particles are added, removed, or resized. Weights
 * need not be normalized; a log-weight of -inf is a zero weight. Particles are addressed by
 * index, and the set has no begin() and end(): loop over the indices, or call forEach(). Removal may reorder the
 * remaining particles. Statistics that need normalized weights, such as moments(), return a WeightError if the weights
 * cannot be normalized (see logTotalWeight()).
 *
 * A set given a Capacity reserves room for that many particles once, along with the scratch that
 * resample() uses, so that resampling and replacing particles at a steady size allocate nothing.
 * Holding more particles than the capacity is a precondition violation.
 *
 * A set may carry a kernel bandwidth: the shared covariance \f$ H \f$ of the
 * %Gaussian kernels in its kernel density estimate, with differenceDimension() rows.
 * kdeBandwidth() is the stored bandwidth if there is one, and Silverman's rule otherwise.
 * Conditionals store their exact bandwidth this way. resample() clears a stored bandwidth, since
 * it describes the cloud before the draw; other changes of the states keep it, so it can go stale;
 * call clearBandwidth() to return to Silverman's rule.
 *
 * A set of weighted points has no density of its own, so toDensity() returns its kernel density
 * estimate: it validates the weights, factorizes a bandwidth, and returns a KernelDensity, which
 * checks this object for changes: reading it after this object has changed in any way is a
 * programmer error caught by \c EMRYS_FILTER_ASSERT (decisions.md [view-stamp]). So is changing the set from
 * inside a forEach() or removeIf() callback.
 *
 * marginal() returns a read-only view of a subset of the state entries, which checks only this
 * object's size and structure: states, weights, and bandwidth may change under it, but reading it
 * after particles are added, removed, resampled, or resized, or this object is assigned to, is a
 * programmer error. Neither view may outlive this object, and neither may be made from a temporary.
 *
 * The space's size always matches the states'. insert() and erase() grow and shrink one
 * resizable block of the space in place, and resizeParticles() resizes a space with a
 * single resizable block.
 *
 * \tparam Space_      space of the states, which sets their dimensions and storage bounds
 *                     (see \ref Spaces): states are normalized when stored, and moments
 *                     are tangent-space moments
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the states, weights, and scratch (see \ref Allocation)
 *
 * \sa class KernelDensity, \ref Resamplers
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class ParticleSet
{
public:

    using MomentsType = Moments<Space_, Scalar_>;                             ///< a mean and covariance together
    using Scalar = typename MomentsType::Scalar;                              ///< the scalar type
    using Index = typename MomentsType::Index;                                ///< Eigen's index type
    using VectorType = typename MomentsType::VectorType;                      ///< a value, such as a state
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;  ///< a difference
    using MatrixType = typename MomentsType::MatrixType;  ///< a matrix over differences, such as a bandwidth

    /** A plain matrix of particle states as columns, laid out as states() is. */
    using SampleSetType = SampleSet<Space_, Scalar_>;

    /** The particle states as columns: a map over the set's storage. */
    using StatesType = ConstSampleSetMap<Space_, Scalar_>;

    /** One particle's state: a column of states(). */
    using ConstStateType = Eigen::Block<StatesType const, SampleSetType::RowsAtCompileTime, 1, true>;

    using WeightArrayType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;  ///< one entry per particle

    /** The log-weights: a map over the set's storage. */
    using LogWeightsType = Eigen::Map<WeightArrayType const>;

    using SpaceType = Space_;                                        ///< the states' space
    using allocator_type = Allocator_;                               ///< the allocator, re-bound per buffer
    using DensityType = KernelDensity<Space_, Scalar_, Allocator_>;  ///< the view toDensity() returns

    /// the length of a state, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a state, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a bandwidth, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;  ///< parameters, which views check

    /** Result of removeIf(): the log of the removed fraction of the total weight,
     * or nothing when \a ReturnRemovedWeight is false.
     */
    template <bool ReturnRemovedWeight>
    using RemovalResult = std::conditional_t<ReturnRemovedWeight, std::expected<Scalar, WeightError>, void>;

    /** \brief Writable access to one particle during forEach().
     *
     * Exists only for the duration of the callback; it cannot be copied or moved.
     */
    class MutableParticle
    {
    public:

        MutableParticle(MutableParticle const &) = delete;  ///< lives only inside the callback

        /** Lives only inside the callback. */
        MutableParticle &
        operator=(MutableParticle const &) = delete;

        /** \returns the particle's index in the set */
        [[nodiscard]] Index
        index() const
        {
            return m_index;
        }

        /** \returns the particle's log-weight */
        [[nodiscard]] Scalar
        logWeight() const
        {
            return m_set.logWeight(m_index);
        }

        /** \returns the particle's state */
        [[nodiscard]] ConstStateType
        state() const
        {
            return m_set.state(m_index);
        }

        /** Sets the particle's state, normalized by the space. */
        template <typename Derived>
        void
        setState(Eigen::MatrixBase<Derived> const & state)
        {
            EMRYS_FILTER_ASSERT(state.size() == m_set.valueDimension());
            m_set.statesMap().col(m_index) = state;
            normalizeValues(m_set.m_space, m_set.statesMap().col(m_index));
        }

    private:

        friend ParticleSet;  ///< the set makes these in forEach()

        /** Refers to particle \a index of \a set for the duration of a callback. */
        MutableParticle(ParticleSet & set, Index index)
            : m_set{set}
            , m_index{index}
        {}

        ParticleSet & m_set;  ///< the set
        Index m_index;        ///< the particle's index
    };

    ParticleSet() = default;

    /** Constructs an empty set that allocates through \a allocator. */
    explicit ParticleSet(Allocator_ const & allocator)
        : m_states(StateAllocator(allocator))
        , m_logWeights(StateAllocator(allocator))
        , m_stateScratch(StateAllocator(allocator))
        , m_weightScratch(StateAllocator(allocator))
        , m_resamplerScratch(StateAllocator(allocator))
        , m_ancestors(IndexAllocator(allocator))
    {
        m_rows = Emrys::valueDimension(m_space);
    }

    /** Constructs an empty set in a FixedSizeSpace that reserves room for \a capacity particles.
     * A space that holds values is given: ParticleSet(space, capacity, allocator).
     */
    explicit ParticleSet(Capacity capacity, Allocator_ const & allocator = {})
        requires FixedSizeSpace<Space_>
        : ParticleSet{allocator}
    {
        setCapacity(capacity.count);
    }

    /** Constructs an empty set whose particles will have \a valueDimension coordinates. */
    explicit ParticleSet(Index valueDimension, Allocator_ const & allocator = {})
        requires internal::SizedByLength<Space_>
        : ParticleSet{allocator}
    {
        m_rows = valueDimension;
        fitSpace();
    }

    /** Constructs a set from the columns of \a states, with equal weights summing to one. */
    template <typename StatesDerived>
        requires internal::SizedByLength<Space_>
    explicit ParticleSet(Eigen::MatrixBase<StatesDerived> const & states, Allocator_ const & allocator = {})
        : ParticleSet{allocator}
    {
        assign(states);
    }

    /** Constructs a set from the columns of \a states with the log-weights \a logWeights. */
    template <typename StatesDerived, typename WeightsDerived>
        requires internal::SizedByLength<Space_>
    ParticleSet(
        Eigen::MatrixBase<StatesDerived> const & states,
        Eigen::DenseBase<WeightsDerived> const & logWeights,
        Allocator_ const & allocator = {}
    )
        : ParticleSet{allocator}
    {
        assign(states, logWeights);
    }

    /** Constructs an empty set in \a space. */
    explicit ParticleSet(Space_ space, Allocator_ const & allocator = {})
        : ParticleSet{allocator}
    {
        m_space = std::move(space);
        m_rows = Emrys::valueDimension(m_space);
    }

    /** Constructs an empty set in \a space that reserves room for \a capacity particles. */
    ParticleSet(Space_ space, Capacity capacity, Allocator_ const & allocator = {})
        : ParticleSet{std::move(space), allocator}
    {
        setCapacity(capacity.count);
    }

    /** Constructs an empty set in \a space whose particles will have \a valueDimension
     * coordinates. A space with one resizable block is resized to \a valueDimension; any other
     * must already have that many.
     */
    ParticleSet(Space_ space, Index valueDimension, Allocator_ const & allocator = {})
        : ParticleSet{allocator}
    {
        m_space = std::move(space);
        m_rows = valueDimension;
        fitSpace();
    }

    /** Constructs a set in \a space from the columns of \a states, with equal weights summing to one. */
    template <typename StatesDerived>
    ParticleSet(Space_ space, Eigen::MatrixBase<StatesDerived> const & states, Allocator_ const & allocator = {})
        : ParticleSet{allocator}
    {
        m_space = std::move(space);
        assign(states);
    }

    /** Constructs a set in \a space from the columns of \a states with the log-weights
     * \a logWeights.
     */
    template <typename StatesDerived, typename WeightsDerived>
    ParticleSet(
        Space_ space,
        Eigen::MatrixBase<StatesDerived> const & states,
        Eigen::DenseBase<WeightsDerived> const & logWeights,
        Allocator_ const & allocator = {}
    )
        : ParticleSet{allocator}
    {
        m_space = std::move(space);
        assign(states, logWeights);
    }

    /** Copies the particles and any stored bandwidth into storage from \a other's allocator, with
     * its capacity (see \ref Allocation). The copy has no resample record: it is a new set, which
     * views of \a other do not see, so its resampleCount() is zero and its ancestors() empty.
     */
    ParticleSet(ParticleSet const & other)
        : ParticleSet{other, other.get_allocator()}
    {}

    /** Copies the particles and any stored bandwidth into storage from \a allocator, with
     * \a other's capacity and no resample record.
     */
    ParticleSet(ParticleSet const & other, Allocator_ const & allocator)
        : ParticleSet{allocator}
    {
        m_capacity = other.m_capacity;
        reserveRows(other.m_rows);
        copyFrom(other);
    }

    /** Takes \a other's particles and storage; a view of \a other may not be read afterward. */
    ParticleSet(ParticleSet && other) noexcept
        : m_rows{other.m_rows}
        , m_states{std::move(other.m_states)}
        , m_logWeights{std::move(other.m_logWeights)}
        , m_space{std::move(other.m_space)}
        , m_bandwidth{std::move(other.m_bandwidth)}
        , m_capacity{other.m_capacity}
        , m_stateScratch{std::move(other.m_stateScratch)}
        , m_weightScratch{std::move(other.m_weightScratch)}
        , m_resamplerScratch{std::move(other.m_resamplerScratch)}
        , m_ancestors{std::move(other.m_ancestors)}
        , m_resampleCount{other.m_resampleCount}
        , m_logTotalWeight{other.m_logTotalWeight}
    {
        other.touchShape();
    }

    /** Moves \a other's particles into storage from \a allocator, taking its storage when the
     * allocators are equal.
     */
    ParticleSet(ParticleSet && other, Allocator_ const & allocator)
        : m_rows{other.m_rows}
        , m_states{std::move(other.m_states), StateAllocator(allocator)}
        , m_logWeights{std::move(other.m_logWeights), StateAllocator(allocator)}
        , m_space{std::move(other.m_space)}
        , m_bandwidth{std::move(other.m_bandwidth)}
        , m_capacity{other.m_capacity}
        , m_stateScratch{std::move(other.m_stateScratch), StateAllocator(allocator)}
        , m_weightScratch{std::move(other.m_weightScratch), StateAllocator(allocator)}
        , m_resamplerScratch{std::move(other.m_resamplerScratch), StateAllocator(allocator)}
        , m_ancestors{std::move(other.m_ancestors), IndexAllocator(allocator)}
        , m_resampleCount{other.m_resampleCount}
        , m_logTotalWeight{other.m_logTotalWeight}
    {
        other.touchShape();
    }

    /** Copies \a other's particles, weights, space, and bandwidth into this set's storage, which
     * keeps its allocator and capacity, and clears its resample record, as a copy has none.
     */
    ParticleSet &
    operator=(ParticleSet const & other)
    {
        touchShape();
        if (this != &other)
        {
            copyFrom(other);
        }
        return *this;
    }

    /** Takes \a other's storage when the allocators are equal or propagate, otherwise copies into
     * this set's storage, which keeps its capacity; either way it takes \a other's resample
     * record. A view of either object may not be read afterward.
     */
    ParticleSet &
    operator=(ParticleSet && other) noexcept
    {
        touchShape();
        other.touchShape();
        if (this == &other)
        {
            return *this;
        }
        if constexpr (std::allocator_traits<Allocator_>::propagate_on_container_move_assignment::value)
        {
            takeStorage(other);
        }
        else
        {
            if (get_allocator() == other.get_allocator())
            {
                takeStorage(other);
            }
            else
            {
                copyFrom(other);
                m_ancestors.assign(other.m_ancestors.begin(), other.m_ancestors.end());
                m_resampleCount = other.m_resampleCount;
            }
        }
        return *this;
    }

    /** \returns the allocator the set's storage comes from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_states.get_allocator());
    }

    /** \returns the number of particles */
    [[nodiscard]] Index
    size() const
    {
        return Index(m_logWeights.size());
    }

    /** \returns whether there are no particles */
    [[nodiscard]] bool
    empty() const
    {
        return size() == 0;
    }

    /** \returns the most particles the set holds without growing its storage, if it has a capacity */
    [[nodiscard]] std::optional<Index>
    capacity() const
    {
        return m_capacity;
    }

    /** Reserves room for \a capacity particles, and for the scratch resample() uses, and makes
     * holding more a precondition violation. It must be at least size().
     */
    void
    setCapacity(Index capacity)
    {
        EMRYS_FILTER_ASSERT(capacity >= size() && "the capacity is smaller than the set");
        m_capacity = capacity;
        reserveRows(m_rows);
    }

    /** \returns the number of coordinates of every particle state (see \ref Spaces) */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_rows;
    }

    /** \returns the number of rows and columns of a bandwidth or covariance (see \ref Spaces) */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

    /** \returns the states' space, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the particle states, one per column */
    [[nodiscard]] StatesType
    states() const
    {
        return StatesType{m_states.data(), m_rows, size()};
    }

    /** \returns the particles' log-weights */
    [[nodiscard]] LogWeightsType
    logWeights() const
    {
        return LogWeightsType{m_logWeights.data(), size()};
    }

    /** \returns particle \a i's state, a column of states() */
    [[nodiscard]] ConstStateType
    state(Index i) const
    {
        StatesType const all = states();
        return ConstStateType{all, i};
    }

    /** \returns particle \a i's log-weight */
    [[nodiscard]] Scalar
    logWeight(Index i) const
    {
        assertIndex(i);
        return m_logWeights[std::size_t(i)];
    }

    /** \returns the stored kernel bandwidth, if any */
    [[nodiscard]] std::optional<MatrixType> const &
    bandwidth() const
    {
        return m_bandwidth;
    }

    /** \returns the bandwidth a kernel density estimate uses: the stored bandwidth, or
     * Silverman's rule if none is stored, which fails with a WeightError if the weights cannot be
     * normalized
     */
    [[nodiscard]] std::expected<MatrixType, WeightError>
    kdeBandwidth() const
    {
        return kdeBandwidth(execution::sequential);
    }

    /** kdeBandwidth() under \a policy: Silverman's rule's total weight, effective sample size, and
     * covariance are reductions over the particles (see \ref Execution).
     */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<MatrixType, WeightError>
    kdeBandwidth(P && policy) const
    {
        if (m_bandwidth)
        {
            return *m_bandwidth;
        }
        return silvermanBandwidth(std::forward<P>(policy), *this);
    }

    /** Stores \a bandwidth as the kernel covariance. Only its lower triangle is read. */
    template <MatrixExpression Bandwidth>
    void
    setBandwidth(Bandwidth && bandwidth)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(bandwidth.rows() == differenceDimension() && bandwidth.cols() == differenceDimension());
        m_bandwidth = MatrixType{std::forward<Bandwidth>(bandwidth)};
    }

    /** Forgets the stored bandwidth, so kdeBandwidth() returns to Silverman's rule. */
    void
    clearBandwidth()
    {
        m_stamp.touchValues();
        m_bandwidth.reset();
    }

    /** Resizes every particle state to \a n coordinates and clears the bandwidth.
     * States are unspecified afterward. A space with several resizable blocks grows and shrinks
     * through insert() and erase() instead.
     */
    void
    resizeParticles(Index n)
        requires internal::SizedByLength<Space_>
    {
        touchShape();
        internal::fitValueDimension(m_space, n);
        m_rows = n;
        reserveRows(m_rows);
        m_states.resize(std::size_t(m_rows * size()));
        m_bandwidth.reset();
    }

    /** Inserts the state entries \a states, one column per particle, into the resizable block
     * named by the tag path \a Path, before its element \a element. The space's block grows to
     * match. A stored bandwidth no longer fits and is cleared, so kdeBandwidth() returns to
     * Silverman's rule. See GaussianParams::insert() for what an element is.
     */
    template <typename Path, typename Derived>
    void
    insert(Index element, Eigen::MatrixBase<Derived> const & states)
    {
        touchShape();
        EMRYS_FILTER_ASSERT(states.cols() == size() && "insert one column per particle");
        auto const rows = internal::growBlock<Path>(m_space, element, states.rows());
        Index const newRows = m_rows + states.rows();
        reserveRows(newRows);
        m_stateScratch.resize(std::size_t(newRows * size()));
        auto grown = SampleSetMap<Space_, Scalar_>{m_stateScratch.data(), newRows, size()};
        grown.topRows(rows.start) = this->states().topRows(rows.start);
        grown.middleRows(rows.start, rows.count) = states;
        grown.bottomRows(m_rows - rows.start) = this->states().bottomRows(m_rows - rows.start);
        m_states.swap(m_stateScratch);
        m_rows = newRows;
        normalizeValues(m_space, statesMap());
        m_bandwidth.reset();
    }

    /** Removes \a count elements of the resizable block named by the tag path \a Path from every
     * particle, starting at its element \a element. A stored bandwidth is marginalized, which
     * keeps the kernel density estimate exact.
     */
    template <typename Path>
    void
    erase(Index element, Index count = 1)
    {
        touchShape();
        auto const rows = internal::shrinkBlock<Path>(m_space, element, count);
        Index const newRows = m_rows - rows.count;
        m_stateScratch.resize(std::size_t(newRows * size()));
        auto shrunk = SampleSetMap<Space_, Scalar_>{m_stateScratch.data(), newRows, size()};
        shrunk.topRows(rows.start) = states().topRows(rows.start);
        shrunk.bottomRows(newRows - rows.start) = states().bottomRows(newRows - rows.start);
        m_states.swap(m_stateScratch);
        m_rows = newRows;
        if (m_bandwidth)
        {
            internal::eraseSymmetric(*m_bandwidth, rows.differenceStart, rows.differenceCount);
        }
    }

    /** Removes every particle; the space, bandwidth, capacity, and scratch stay. */
    void
    clear()
    {
        touchShape();
        m_states.clear();
        m_logWeights.clear();
    }

    /** Replaces the particles with the columns of \a states, with equal weights summing
     * to one. Clears the bandwidth if the number of coordinates changes.
     */
    template <typename StatesDerived>
    void
    assign(Eigen::MatrixBase<StatesDerived> const & states)
    {
        touchShape();
        copyStates(states);
        logWeightsMap().setConstant(-Eigen::numext::log(Scalar(size())));
        fitSpace();
    }

    /** Replaces the particles and weights. Clears the bandwidth if the number of coordinates
     * changes.
     */
    template <typename StatesDerived, typename WeightsDerived>
    void
    assign(Eigen::MatrixBase<StatesDerived> const & states, Eigen::DenseBase<WeightsDerived> const & logWeights)
    {
        touchShape();
        EMRYS_FILTER_ASSERT(logWeights.size() == states.cols());
        copyStates(states);
        logWeightsMap() = logWeights.derived().array();
        fitSpace();
    }

    /** Appends a particle. Without a capacity it may grow the storage, which costs O(size()).
     *
     * \returns the index of the new particle
     */
    template <typename Derived>
    Index
    add(Scalar logWeight, Eigen::MatrixBase<Derived> const & state)
    {
        touchShape();
        EMRYS_FILTER_ASSERT(state.size() == valueDimension());
        // Copied first: growing the storage would move a state read from this set.
        VectorType const value = state;
        Index const i = size();
        setCount(i + 1);
        statesMap().col(i) = value;
        normalizeValues(m_space, statesMap().col(i));
        m_logWeights[std::size_t(i)] = logWeight;
        return i;
    }

    /** Appends copies of \a other's particles, which must be in the same space. */
    template <typename OtherAllocator>
    void
    append(ParticleSet<Space_, Scalar_, OtherAllocator> const & other)
    {
        touchShape();
        EMRYS_FILTER_ASSERT(other.space() == m_space && "cannot append a set in a different space");
        Index const n = size();
        Index const m = other.size();
        // Read after growing, which may move this set's storage when other is this set.
        auto const * source = &other;
        setCount(n + m);
        statesMap().rightCols(m) = source->states().leftCols(m);
        logWeightsMap().tail(m) = source->logWeights().head(m);
    }

    /** Removes particle \a i by moving the last particle into its place. */
    void
    remove(Index i)
    {
        touchShape();
        assertIndex(i);
        Index const last = size() - 1;
        if (i != last)
        {
            statesMap().col(i) = states().col(last);
            m_logWeights[std::size_t(i)] = m_logWeights[std::size_t(last)];
        }
        setCount(last);
    }

    /** Sets particle \a i's state, normalized by the space. */
    template <typename Derived>
    void
    setState(Index i, Eigen::MatrixBase<Derived> const & state)
    {
        m_stamp.touchValues();
        assertIndex(i);
        EMRYS_FILTER_ASSERT(state.size() == valueDimension());
        statesMap().col(i) = state;
        normalizeValues(m_space, statesMap().col(i));
    }

    /** Sets every state at once. \a states must have the current dimensions. */
    template <typename Derived>
    void
    setStates(Eigen::MatrixBase<Derived> const & states)
    {
        m_stamp.touchValues();
        EMRYS_FILTER_ASSERT(states.rows() == valueDimension() && states.cols() == size());
        statesMap() = states;
        normalizeValues(m_space, statesMap());
    }

    /** Sets particle \a i's log-weight. */
    void
    setLogWeight(Index i, Scalar logWeight)
    {
        touchWeights();
        assertIndex(i);
        m_logWeights[std::size_t(i)] = logWeight;
    }

    /** Sets every log-weight at once; \a logWeights has one entry per particle. */
    template <typename Derived>
    void
    setLogWeights(Eigen::DenseBase<Derived> const & logWeights)
    {
        touchWeights();
        EMRYS_FILTER_ASSERT(logWeights.size() == size());
        logWeightsMap() = logWeights.derived().array();
    }

    /** Multiplies the weight of particle \a i by \f$ e^{\mathrm{logScale}} \f$. */
    void
    scaleWeight(Index i, Scalar logScale)
    {
        touchWeights();
        assertIndex(i);
        m_logWeights[std::size_t(i)] += logScale;
    }

    /** Multiplies every weight by \f$ e^{\mathrm{logScale}} \f$. */
    void
    scaleWeights(Scalar logScale)
    {
        touchWeights();
        logWeightsMap() += logScale;
    }

    /** Multiplies weight \c i by \f$ e^{\mathrm{logScales}_i} \f$, e.g. by per-particle likelihoods. */
    template <typename Derived>
    void
    scaleWeights(Eigen::DenseBase<Derived> const & logScales)
    {
        touchWeights();
        EMRYS_FILTER_ASSERT(logScales.size() == size());
        logWeightsMap() += logScales.derived().array();
    }

    /** \returns the log of the sum of the weights, or a WeightError if the set is
     *          empty, the weights sum to zero, or a log-weight is NaN or +inf.
     *
     * After normalize(), resample(), or a ParticleFilter update, and until the weights next
     * change, it is zero, the total those scaled the weights to, without the rounding that summing
     * them again would add; the operations that need the total, such as ess() and moments(), read
     * that rather than taking a pass over the weights.
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    logTotalWeight() const
    {
        return logTotalWeight(execution::sequential);
    }

    /** logTotalWeight() under \a policy: a reduction over the particles (see \ref Execution). */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<Scalar, WeightError>
    logTotalWeight(P && policy) const
    {
        if (m_logTotalWeight)
        {
            return *m_logTotalWeight;
        }
        return internal::logTotalWeight<Scalar>(
            std::forward<P>(policy),
            std::span<Scalar const>{m_logWeights.data(), m_logWeights.size()}
        );
    }

    /** Scales the weights so they sum to one. On failure the weights are unchanged.
     *
     * \sa logTotalWeight()
     */
    [[nodiscard]] std::expected<void, WeightError>
    normalize()
    {
        return normalize(execution::sequential);
    }

    /** normalize() under \a policy: the total is a reduction over the particles, and scaling the
     * weights a whole-array operation (see \ref Execution).
     */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<void, WeightError>
    normalize(P && policy)
    {
        touchWeights();
        auto const total = logTotalWeight(std::forward<P>(policy));
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        scaleWeights(-*total);
        m_logTotalWeight = Scalar(0);
        return {};
    }

    /** \returns the normalized linear weights \f$ \hat w_i \f$, or a WeightError if the
     *          weights cannot be normalized
     */
    [[nodiscard]] std::expected<WeightArrayType, WeightError>
    normalizedWeights() const
    {
        return logTotalWeight().transform([this](Scalar total) {
            return WeightArrayType{(logWeights() - total).exp()};
        });
    }

    /** \returns the effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$, between 1 and
     *          size(), or a WeightError if the weights cannot be normalized; see
     *          internal::ess() for its source
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    ess() const
    {
        return ess(execution::sequential);
    }

    /** ess() under \a policy: two reductions over the particles (see \ref Execution). */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<Scalar, WeightError>
    ess(P && policy) const
    {
        return logTotalWeight(policy).transform([&](Scalar total) {
            return internal::ess(policy, logWeights(), total);
        });
    }

    /** \returns the weighted mean and covariance,
     * \f$ \mu = \sum_i \hat w_i x_i \f$ and \f$ \Sigma = \sum_i \hat w_i (x_i - \mu)(x_i - \mu)^T \f$,
     * or a WeightError if the weights cannot be normalized.
     *
     * The covariance is the plain weighted (biased) estimate and is exactly symmetric.
     * It excludes any kernel bandwidth; see KernelDensity::moments() for that. In a
     * non-linear space these are tangent-space moments.
     */
    [[nodiscard]] std::expected<MomentsType, WeightError>
    moments() const
    {
        return moments(execution::sequential);
    }

    /** moments() under \a policy: reductions over the particles, which match Sequential's to
     * rounding (see \ref Execution).
     */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<MomentsType, WeightError>
    moments(P && policy) const
    {
        return logTotalWeight(policy).transform([&](Scalar total) {
            return internal::weightedMomentsIn<MomentsType>(policy, m_space, states(), logWeights(), total);
        });
    }

    /** Calls \a f on each particle as a \c MutableParticle&, returning nothing; reweight() takes
     * one that returns a log scale factor. \a f may modify the particle it is given but not the
     * set, which is a programmer error caught when \a f returns. Views of the set may not be read
     * afterward.
     */
    template <typename F>
    void
    forEach(F && f)
    {
        forEach(execution::sequential, std::forward<F>(f));
    }

    /** forEach() under \a policy, a for-each over the particles (see \ref Execution). Under a
     * parallel policy \a f runs for several particles at once, so it must read and write nothing but
     * the particle it is given and what no other call writes.
     */
    template <ExecutionPolicy P, typename F>
    void
    forEach(P && policy, F && f)
    {
        static_assert(
            std::is_void_v<std::invoke_result_t<F &, MutableParticle &>>,
            "forEach() takes a callback that returns nothing; one that returns a log scale factor goes to reweight()"
        );
        visitParticles(std::forward<P>(policy), [&](MutableParticle & particle, Index) {
            std::invoke(f, particle);
        });
    }

    /** Calls \a f on each particle as a \c MutableParticle&, as forEach() does, and adds what it
     * returns, a log scale factor in a floating-point type, to that particle's log-weight.
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
            std::floating_point<std::remove_cvref_t<std::invoke_result_t<F &, MutableParticle &>>>,
            "reweight() takes a callback that returns a log scale factor, a floating-point number"
        );
        visitParticles(std::forward<P>(policy), [&](MutableParticle & particle, Index i) {
            m_logWeights[std::size_t(i)] += Scalar(std::invoke(f, particle));
        });
    }

    /** Removes every particle for which \a pred(logWeight, state) is true.
     *
     * \tparam ReturnRemovedWeight  if true, returns the log of the removed fraction
     *         of the total weight. This requires normalizing the weights; if that
     *         fails, nothing is removed and the WeightError is returned.
     */
    template <bool ReturnRemovedWeight = false, typename Pred>
        requires(!ReturnRemovedWeight && !ExecutionPolicy<Pred>)
    void
    removeIf(Pred pred)
    {
        removeParticlesIf<false>(execution::sequential, std::move(pred));
    }

    /** As above, returning the log of the removed fraction of the total weight, or the
     * WeightError that left the set unchanged.
     */
    template <bool ReturnRemovedWeight, typename Pred>
        requires(ReturnRemovedWeight && !ExecutionPolicy<Pred>)
    [[nodiscard]] std::expected<Scalar, WeightError>
    removeIf(Pred pred)
    {
        return removeParticlesIf<true>(execution::sequential, std::move(pred));
    }

    /** removeIf() under \a policy: \a pred runs as a for-each over the particles, and the
     * particles kept are compacted, in order, through the scratch (see \ref Execution). The
     * particles left are Sequential's; the removed weight matches it to rounding. Under a parallel
     * policy \a pred runs for several particles at once, so it must not write shared state.
     */
    template <bool ReturnRemovedWeight = false, ExecutionPolicy P, typename Pred>
        requires(!ReturnRemovedWeight)
    void
    removeIf(P && policy, Pred pred)
    {
        removeParticlesIf<false>(std::forward<P>(policy), std::move(pred));
    }

    /** As above, returning the log of the removed fraction of the total weight, or the
     * WeightError that left the set unchanged.
     */
    template <bool ReturnRemovedWeight, ExecutionPolicy P, typename Pred>
        requires ReturnRemovedWeight
    [[nodiscard]] std::expected<Scalar, WeightError>
    removeIf(P && policy, Pred pred)
    {
        return removeParticlesIf<true>(std::forward<P>(policy), std::move(pred));
    }

    /** Removes every particle whose normalized log-weight is below \a logThreshold, and normalizes
     * the weights of the rest; the heaviest particle is always kept, the first of them if several
     * are. The threshold is absolute: a caller who wants it relative to something, such as the
     * heaviest particle, makes it so. The particles keep their order, and a stored bandwidth is
     * kept, as removeIf() keeps it. upkeep::pruneBelow() calls it.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     *
     * \sa pruneTail(), pruneToSize()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneBelow(Scalar logThreshold)
    {
        return pruneParticles(internal::PruneBelowRule<Scalar>{logThreshold}, [](Index) {});
    }

    /** Removes the lightest particles, as many as can be while their combined normalized weight
     * stays below \f$ e^{\mathrm{logFraction}} \f$, and normalizes the weights of the rest. The
     * fraction is below one, so a particle is always kept. Of particles equally heavy, those later
     * in order go first. upkeep::pruneTail() calls it.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     *
     * \sa pruneBelow(), pruneToSize()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneTail(Scalar logFraction)
    {
        return pruneParticles(internal::PruneTailRule<Scalar>{logFraction}, [](Index) {});
    }

    /** Removes the lightest particles until at most \a maxSize, at least one, remain, and
     * normalizes the weights of the rest. Of particles equally heavy, as after a resample, those
     * later in order go first. upkeep::pruneToSize() calls it.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     *
     * \sa pruneBelow(), pruneTail()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneToSize(Index maxSize)
    {
        return pruneParticles(internal::PruneToSizeRule{maxSize}, [](Index) {});
    }

    /** Replaces the particles with \a newSize draws chosen by \a resampler in
     * proportion to their weights, then sets every weight to \f$ 1/\mathrm{newSize} \f$.
     * A stored bandwidth is cleared, since it describes the particles before the draw.
     *
     * \a rng is used only during this call. Scratch storage is kept between calls, and passed to
     * a \a resampler that takes it (ResamplerWithScratch), so repeated resampling at a steady size
     * does not allocate.
     *
     * \returns nothing on success, or a WeightError (with the set unchanged) if
     *          the weights cannot be normalized.
     *
     * \sa \ref Resamplers
     */
    template <std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
        requires Resampler<R, Scalar, G>
    [[nodiscard]] std::expected<void, WeightError>
    resample(G & rng, Index newSize, R const & resampler = {})
    {
        return resample(execution::sequential, rng, newSize, resampler);
    }

    /** Resamples to the current size(). */
    template <std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
        requires Resampler<R, Scalar, G>
    [[nodiscard]] std::expected<void, WeightError>
    resample(G & rng, R const & resampler = {})
    {
        return resample(execution::sequential, rng, size(), resampler);
    }

    /** resample() under \a policy: the total weight is a reduction, the linear weights a
     * whole-array expression on each chunk, the gather of the ancestors' states a for-each, and
     * \a resampler runs under \a policy if it
     * can (see ResamplerUnder), drawing from \a rng in the order the sequential form does. Its
     * ancestors may differ from Sequential's by rounding (see \ref Resamplers).
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
        requires Resampler<R, Scalar, G>
    [[nodiscard]] std::expected<void, WeightError>
    resample(P && policy, G & rng, Index newSize, R const & resampler = {})
    {
        touchShape();
        EMRYS_FILTER_ASSERT(newSize > 0);
        assertCapacity(newSize);
        auto const total = logTotalWeight(policy);
        if (!total)
        {
            return std::unexpected{total.error()};
        }

        m_weightScratch.resize(std::size_t(size()));
        auto linearWeights = Eigen::Map<WeightArrayType>{m_weightScratch.data(), size()};
        internal::forEachChunk(policy, size(), [&](Index begin, Index length) {
            linearWeights.segment(begin, length) = (logWeights().segment(begin, length) - *total).exp();
        });
        m_ancestors.resize(std::size_t(newSize));
        auto const weights = std::span<Scalar const>{m_weightScratch};
        auto const ancestors = std::span<Index>{m_ancestors};
        if constexpr (ResamplerUnder<R, P, Scalar, G> && !internal::isSequential<P>)
        {
            resampler(policy, weights, ancestors, rng);
        }
        else if constexpr (ResamplerWithScratch<R, Scalar, G>)
        {
            m_resamplerScratch.resize(std::size_t(size() + newSize));
            resampler(weights, ancestors, rng, std::span<Scalar>{m_resamplerScratch});
        }
        else
        {
            resampler(weights, ancestors, rng);
        }

        m_stateScratch.resize(std::size_t(m_rows * newSize));
        internal::gather(
            policy,
            std::span<Index const>{ancestors},
            states(),
            SampleSetMap<Space_, Scalar_>{m_stateScratch.data(), m_rows, newSize}
        );
        m_states.swap(m_stateScratch);
        m_logWeights.resize(std::size_t(newSize));
        internal::fill(std::forward<P>(policy), std::span<Scalar>{m_logWeights}, -Eigen::numext::log(Scalar(newSize)));
        m_logTotalWeight = Scalar(0);
        m_bandwidth.reset();
        ++m_resampleCount;
        return {};
    }

    /** resample() to the current size() under \a policy. */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
        requires Resampler<R, Scalar, G>
    [[nodiscard]] std::expected<void, WeightError>
    resample(P && policy, G & rng, R const & resampler = {})
    {
        return resample(std::forward<P>(policy), rng, size(), resampler);
    }

    /** \returns the ancestor indices chosen by the last successful resample(): particle \a j became
     * a copy of the particle at <tt>ancestors()[j]</tt> before it
     */
    [[nodiscard]] std::span<Index const>
    ancestors() const
    {
        return {m_ancestors.data(), m_ancestors.size()};
    }

    /** \returns how many times resample() has succeeded on this set, including on the sets it was
     * moved from (a copy starts at zero). An observer that saw an earlier count can tell from it
     * whether ancestors() is new since then, as ParticleHistory does (decisions.md [smoothing]).
     */
    [[nodiscard]] Index
    resampleCount() const
    {
        return m_resampleCount;
    }

    /** marginal() of the subspace named by the tag path \a Path (see idx()), whose space is
     * that subspace with its parameters.
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const &
    {
        return internal::tagMarginal<ParticleSetMarginal, Path>(*this);
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
        return internal::elementMarginal<ParticleSetMarginal, Path>(*this, element);
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal(Index element) const && = delete;

    /** \returns a copy of the kernel density estimate's distribution of the other coordinates
     * given that the subspace named by the path \a GivenPath equals \a values, in the space of
     * the remaining blocks (see GaussianParams::conditional()), with this set's allocator.
     *
     * Uses kdeBandwidth(). The conditional of a %Gaussian KDE is again a %Gaussian KDE: each center
     * becomes \f$ x_{a,i} \boxplus K (v \boxminus x_{b,i}) \f$, each weight gains
     * \f$ \mathcal{N}(v; x_{b,i}, H_{bb}) \f$, and the shared bandwidth becomes
     * \f$ H_{aa} - K H_{ba} \f$, with \f$ K = H_{ab} H_{bb}^{-1} \f$. The weights are not normalized,
     * and the result stores the exact conditional bandwidth, so a KDE built from it with its
     * stored bandwidth reproduces this conditional; Silverman's rule would re-smooth it.
     *
     * \returns the conditional set, or GaussianError::NotPositiveDefinite if \f$ H_{bb} \f$ is not
     *          positive definite, as Silverman's rule makes it for particles that coincide in the
     *          given block (decisions.md [innovation-cholesky]), or GaussianError::NonFinite if
     *          no bandwidth is stored and the weights cannot be normalized
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        auto const bandwidth = kdeBandwidth();
        using Result = decltype(internal::tagConditionalParticles<GivenPath>(*this, *bandwidth, values));
        if (!bandwidth)
        {
            return Result{std::unexpected{GaussianError::NonFinite}};
        }
        return internal::tagConditionalParticles<GivenPath>(*this, *bandwidth, values);
    }

    /** Validates the weights and builds a kernel density estimate with kdeBandwidth():
     * the stored bandwidth if there is one, otherwise Silverman's rule. A conditional
     * stores its exact bandwidth, so its density is exact.
     *
     * \returns a KernelDensity that checks this object, or a NumericError if the weights
     *          cannot be normalized or the bandwidth is not positive definite (for example,
     *          Silverman's rule when every particle has the same state).
     */
    [[nodiscard]] std::expected<DensityType, NumericError>
    toDensity() const &
    {
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        // The weights are valid, so the bandwidth is found.
        return kernelDensity(*total, *kdeBandwidth());
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

    /** Validates the weights and builds a kernel density estimate with bandwidth
     * \a bandwidth, of which only the lower triangle is read.
     *
     * \returns a KernelDensity that checks this object, or a NumericError if the weights
     *          cannot be normalized or \a bandwidth is not positive definite.
     */
    template <MatrixExpression Bandwidth>
    [[nodiscard]] std::expected<DensityType, NumericError>
    toDensity(Bandwidth && bandwidth) const &
    {
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        return kernelDensity(*total, std::forward<Bandwidth>(bandwidth));
    }

    /** A density of a temporary would outlive it. */
    template <MatrixExpression Bandwidth>
    void
    toDensity(Bandwidth && bandwidth) const && = delete;

private:

    /** A marginal checks the stamp. */
    template <typename>
    friend class internal::ShapeCheck;

    /** The kernel density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** A filter's update records the log total weight it normalized to. */
    friend struct internal::ParticleSetAccess;

    /** The body of removeIf(), returning the removed weight if \a ReturnRemovedWeight. */
    template <bool ReturnRemovedWeight, ExecutionPolicy P, typename Pred>
    RemovalResult<ReturnRemovedWeight>
    removeParticlesIf(P && policy, Pred pred)
    {
        touchShape();
        [[maybe_unused]] Scalar logTotal = 0;
        if constexpr (ReturnRemovedWeight)
        {
            auto const total = logTotalWeight(policy);
            if (!total)
            {
                return std::unexpected{total.error()};
            }
            logTotal = *total;
        }
        auto const guard = internal::CallbackGuard{m_stamp};
        auto const checked = [&](Scalar logWeight, ConstStateType const & state) {
            bool const remove = static_cast<bool>(std::invoke(pred, logWeight, state));
            guard.check();
            return remove;
        };
        [[maybe_unused]] Scalar removed = 0;
        if constexpr (internal::isSequential<P>)
        {
            removed = compact([&](Index i) {
                return checked(m_logWeights[std::size_t(i)], state(i));
            });
        }
        else
        {
            removed = compact(policy, checked, ReturnRemovedWeight);
        }
        if constexpr (ReturnRemovedWeight)
        {
            return removed - logTotal;
        }
    }

    /** The body of pruneBelow(), pruneTail(), and pruneToSize(): removes the particles that
     * \a rule's cut removes, ordering their weights in the scratch where it needs to, calls
     * \a onKept with the position of each particle kept, in order, and normalizes the rest.
     *
     * \returns the log of the removed fraction of the total weight, or the WeightError of
     *          logTotalWeight() with nothing changed
     */
    template <typename Rule, typename OnKept>
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneParticles(Rule const & rule, OnKept const & onKept)
    {
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        touchShape();
        auto cut = rule(
            size(),
            [&](Index i) {
                return m_logWeights[std::size_t(i)];
            },
            *total,
            m_weightScratch
        );
        Scalar const removed = compact([&](Index i) {
            bool const remove = cut.removes(m_logWeights[std::size_t(i)]);
            if (!remove)
            {
                onKept(i);
            }
            return remove;
        });
        // The kept particles include one of positive weight, so their total is finite.
        static_cast<void>(normalize());
        return removed - *total;
    }

    /** The loop of forEach() and reweight(): calls \a visit with each particle's MutableParticle
     * and its index, under \a policy, checking after each that the callback left the set alone.
     */
    template <ExecutionPolicy P, typename Visit>
    void
    visitParticles(P && policy, Visit const & visit)
    {
        touchWeights();
        auto const guard = internal::CallbackGuard{m_stamp};
        internal::forEachIndex(std::forward<P>(policy), size(), [&](Index i) {
            auto particle = MutableParticle{*this, i};
            visit(particle, i);
            guard.check();
        });
    }

    /** Records a change of the weights, which the recorded log total weight no longer describes. */
    void
    touchWeights()
    {
        m_stamp.touchValues();
        m_logTotalWeight.reset();
    }

    /** Records a change of size or structure, which changes the weights too. */
    void
    touchShape()
    {
        m_stamp.touchShape();
        m_logTotalWeight.reset();
    }

    using StateAllocator = internal::ReBound<Allocator_, Scalar>;  ///< allocates states, weights, and scratch
    using IndexAllocator = internal::ReBound<Allocator_, Index>;   ///< allocates the ancestors

    /** A programmer error if \a count particles would exceed the capacity, when there is one. */
    void
    assertCapacity([[maybe_unused]] Index count) const
    {
        EMRYS_FILTER_ASSERT(
            (!m_capacity || count <= *m_capacity) && "the set would hold more particles than its capacity"
        );
    }

    /** \returns the states as a writable map over the storage */
    [[nodiscard]] SampleSetMap<Space_, Scalar_>
    statesMap()
    {
        return {m_states.data(), m_rows, size()};
    }

    /** \returns the log-weights as a writable map over the storage */
    [[nodiscard]] Eigen::Map<WeightArrayType>
    logWeightsMap()
    {
        return {m_logWeights.data(), size()};
    }

    /** Makes room for \a count particles of the current dimension. */
    void
    setCount(Index count)
    {
        assertCapacity(count);
        m_states.resize(std::size_t(m_rows * count));
        m_logWeights.resize(std::size_t(count));
    }

    /** Reserves the capacity, if any, for states of \a rows coordinates, or of the space's
     * largest size if it is bounded.
     */
    void
    reserveRows(Index rows)
    {
        if (!m_capacity)
        {
            return;
        }
        Index const bound = maxValueDimensionAtCompileTime<Space_> == Eigen::Dynamic
                              ? rows
                              : Index(maxValueDimensionAtCompileTime<Space_>);
        auto const count = std::size_t(*m_capacity);
        m_states.reserve(std::size_t(bound) * count);
        m_stateScratch.reserve(std::size_t(bound) * count);
        m_logWeights.reserve(count);
        m_weightScratch.reserve(count);
        m_resamplerScratch.reserve(2 * count);
        m_ancestors.reserve(count);
    }

    /** Replaces the states with the columns of \a states, clearing the bandwidth if their
     * dimension changes.
     */
    template <typename StatesDerived>
    void
    copyStates(Eigen::MatrixBase<StatesDerived> const & states)
    {
        if (states.rows() != m_rows)
        {
            m_bandwidth.reset();
        }
        m_rows = states.rows();
        reserveRows(m_rows);
        setCount(states.cols());
        statesMap() = states;
    }

    /** Copies \a other's particles, weights, space, and bandwidth into this set's storage, and
     * clears the resample record.
     */
    void
    copyFrom(ParticleSet const & other)
    {
        m_rows = other.m_rows;
        m_space = other.m_space;
        m_bandwidth = other.m_bandwidth;
        setCount(other.size());
        m_states.assign(other.m_states.begin(), other.m_states.end());
        m_logWeights.assign(other.m_logWeights.begin(), other.m_logWeights.end());
        m_logTotalWeight = other.m_logTotalWeight;
        m_ancestors.clear();
        m_resampleCount = 0;
    }

    /** Takes \a other's storage, scratch, and capacity. */
    void
    takeStorage(ParticleSet & other)
    {
        m_rows = other.m_rows;
        m_states = std::move(other.m_states);
        m_logWeights = std::move(other.m_logWeights);
        m_logTotalWeight = other.m_logTotalWeight;
        m_space = std::move(other.m_space);
        m_bandwidth = std::move(other.m_bandwidth);
        m_capacity = other.m_capacity;
        m_stateScratch = std::move(other.m_stateScratch);
        m_weightScratch = std::move(other.m_weightScratch);
        m_resamplerScratch = std::move(other.m_resamplerScratch);
        m_ancestors = std::move(other.m_ancestors);
        m_resampleCount = other.m_resampleCount;
    }

    /** \returns the KernelDensity over this set with \a bandwidth, or its GaussianError as a
     * NumericError
     */
    template <MatrixExpression Bandwidth>
    [[nodiscard]] std::expected<DensityType, NumericError>
    kernelDensity(Scalar logTotalWeight, Bandwidth && bandwidth) const
    {
        auto kde = DensityType::create(*this, logTotalWeight, std::forward<Bandwidth>(bandwidth));
        if (!kde)
        {
            return std::unexpected{kde.error()};
        }
        return std::move(*kde);
    }

    /** A programmer error unless \a i is a particle's index. */
    void
    assertIndex([[maybe_unused]] Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < size() && "particle index out of range");
    }

    /** Resizes a space with one resizable block to the states, checks that the sizes match, and
     * normalizes the states.
     */
    void
    fitSpace()
    {
        internal::fitValueDimension(m_space, valueDimension());
        normalizeValues(m_space, statesMap());
    }

    /** Removes particles whose index satisfies \a removeAtIndex, keeping the rest in order.
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
                removed.add(m_logWeights[std::size_t(i)]);
                continue;
            }
            if (out != i)
            {
                statesMap().col(out) = states().col(i);
                m_logWeights[std::size_t(out)] = m_logWeights[std::size_t(i)];
            }
            ++out;
        }
        setCount(out);
        return removed.value();
    }

    /** compact() under a parallel \a policy, removing the particles for which
     * \a pred(logWeight, state) holds: \a pred runs as a for-each into flags, the kept particles
     * are copied in order into the scratch, which is swapped in, and the removed weight is a
     * reduction, taken only if \a sumRemoved.
     *
     * \returns the log of the sum of the removed weights, or -inf if not \a sumRemoved
     */
    template <ExecutionPolicy P, typename Pred>
    Scalar
    compact(P && policy, Pred const & pred, bool sumRemoved)
    {
        Index const count = size();
        auto remove = internal::scratchVector<char>(policy, static_cast<std::size_t>(count));
        internal::forEachIndex(policy, count, [&](Index i) {
            remove[std::size_t(i)] = static_cast<bool>(std::invoke(pred, m_logWeights[std::size_t(i)], state(i)));
        });
        Scalar const removed = sumRemoved ? logSumExp<Scalar>(
                                                policy,
                                                count,
                                                [&](Index i) {
                                                    return remove[std::size_t(i)]
                                                             ? m_logWeights[std::size_t(i)]
                                                             : -Eigen::NumTraits<Scalar>::infinity();
                                                }
                                            )
                                          : -Eigen::NumTraits<Scalar>::infinity();
        m_stateScratch.resize(m_states.size());
        m_weightScratch.resize(m_logWeights.size());
        auto kept = SampleSetMap<Space_, Scalar_>{m_stateScratch.data(), m_rows, count};
        Index const keptCount = internal::copyIfIndex(
            std::forward<P>(policy),
            count,
            [&](Index i) {
                return remove[std::size_t(i)] == 0;
            },
            [&](Index i, Index j) {
                kept.col(j) = states().col(i);
                m_weightScratch[std::size_t(j)] = m_logWeights[std::size_t(i)];
            }
        );
        m_states.swap(m_stateScratch);
        m_logWeights.swap(m_weightScratch);
        setCount(keptCount);
        return removed;
    }

    Index m_rows = valueDimensionAtCompileTime<Space_> == Eigen::Dynamic
                     ? 0
                     : valueDimensionAtCompileTime<Space_>;  ///< coordinates per state
    internal::Buffer<Scalar, Allocator_> m_states;           ///< valueDimension() by size(), column-major
    internal::Buffer<Scalar, Allocator_> m_logWeights;       ///< one per particle
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;           ///< the states' space, kept in step with m_rows
    std::optional<MatrixType> m_bandwidth;                   ///< the stored kernel bandwidth, if any
    std::optional<Index> m_capacity;                         ///< the most particles allowed, if set

    // Scratch, kept between calls so that steady-size resampling and resizing don't allocate.
    internal::Buffer<Scalar, Allocator_> m_stateScratch;  ///< states being rebuilt, swapped in when done

    internal::Buffer<Scalar, Allocator_> m_weightScratch;     ///< linear weights for the resampler
    internal::Buffer<Scalar, Allocator_> m_resamplerScratch;  ///< scratch for a resampler that takes it
    internal::Buffer<Index, Allocator_> m_ancestors;          ///< the last resample's ancestors
    Index m_resampleCount = 0;                                ///< successful resample() calls
    std::optional<Scalar> m_logTotalWeight;                   ///< the log total weight, where known without a pass
    internal::Stamp m_stamp;                                  ///< what views check [view-stamp]
};

/** \ingroup Distributions
 * \brief Read-only view of a subset of the state entries of a ParticleSet.
 *
 * Obtained from marginal(), and held by \c auto: its Selection is internal, so its type is never
 * spelled, and a function that takes one takes it as a template parameter. It checks the source's
 * shape. Weights are those of the source.
 * A stored bandwidth is marginalized too, which keeps the kernel density estimate
 * exact: the marginal of a KDE is the KDE of the marginal particles with bandwidth
 * \f$ H_{aa} \f$.
 *
 * It is a random-access range over the particles, as the marginal of a mixture is over the
 * components (decisions.md [element-marginals]): each element, made when read, holds a particle's log-weight and
 * the selected entries of its state, read in place, so iterating copies nothing.
 *
 * \code
 * for (auto const & [logWeight, position] : particles.marginal<Position>())
 * {
 *     position.norm();   // the particle's position, read in place
 * }
 * \endcode
 *
 * \tparam Source     ParticleSet
 * \tparam Selection  an internal::Coordinates of the index sets of the selected block, from
 *                    idx() for the states and diffIdx() for the bandwidth
 * \tparam SubSpace   space of the selected block
 */
template <typename Source, typename Selection, typename SubSpace>
class ParticleSetMarginal
{
public:

    using Scalar = typename Source::Scalar;  ///< the scalar type
    using Index = Eigen::Index;              ///< Eigen's index type

    /// the length of a marginal state, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = valueDimensionAtCompileTime<SubSpace>;

    /// a bound on the length of a marginal state, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = maxValueDimensionAtCompileTime<SubSpace>;

    /// the length of a difference and the rows of the marginal bandwidth, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = valueDimensionAtCompileTime<DifferenceSpace<SubSpace>>;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime =
        maxValueDimensionAtCompileTime<DifferenceSpace<SubSpace>>;

    using SpaceType = SubSpace;                                        ///< the selected block's space
    using allocator_type = typename Source::allocator_type;            ///< the source's allocator
    using ParamsType = ParticleSet<SubSpace, Scalar, allocator_type>;  ///< what copy() returns
    using MatrixType = typename ParamsType::MatrixType;                ///< a matrix over the marginal's differences
    using WeightArrayType = typename ParamsType::WeightArrayType;      ///< one entry per particle
    using LogWeightsType = typename Source::LogWeightsType;            ///< the source's log-weights map

    /** The selected state entries' indices. */
    using IndexVectorType = internal::IndexedSize<
        decltype(Selection::values),
        Source::ValueDimensionAtCompileTime,
        Source::MaxValueDimensionAtCompileTime>;

    /** The selected bandwidth rows' indices. */
    using DifferenceIndexVectorType = internal::IndexedSize<
        decltype(Selection::differences),
        Source::DifferenceDimensionAtCompileTime,
        Source::MaxDifferenceDimensionAtCompileTime>;

    /** The selected entries of one particle's state, a lazy expression that reads the source. */
    using StateType = decltype(std::declval<Source const &>().state(Index{})(std::declval<IndexVectorType const &>()));

    /** One particle's log-weight, as in the source, and the selected entries of its state. */
    struct Element
    {
        Scalar logWeight;  ///< the particle's log-weight
        StateType state;   ///< the selected entries of its state, read in place
    };

    using value_type = Element;                                             ///< an element, made when read
    using const_iterator = internal::ElementIterator<ParticleSetMarginal>;  ///< iterates the particles
    using iterator = const_iterator;  ///< iterates the particles; they are read-only

    /** Selects \a selection's coordinates, which form \a space. */
    ParticleSetMarginal(Source const & source, Selection const & selection, SubSpace space)
        : m_holder{source}
        , m_indices{internal::resolveIndices<
              Source::ValueDimensionAtCompileTime,
              Source::MaxValueDimensionAtCompileTime>(source.valueDimension(), selection.values)}
        , m_differenceIndices{internal::resolveIndices<
              Source::DifferenceDimensionAtCompileTime,
              Source::MaxDifferenceDimensionAtCompileTime>(source.differenceDimension(), selection.differences)}
        , m_space{std::move(space)}
    {
        EMRYS_FILTER_ASSERT(
            internal::distinctIndices(m_indices) && internal::distinctIndices(m_differenceIndices)
            && "marginal indices must be distinct"
        );
    }

    /** \returns the selected block's space, with its parameters and size */
    [[nodiscard]] SubSpace const &
    space() const
    {
        return m_space;
    }

    /** \returns the source's allocator */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return source().get_allocator();
    }

    /** \returns the number of particles */
    [[nodiscard]] Index
    size() const
    {
        return source().size();
    }

    /** \returns the number of coordinates of every marginal particle */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_indices.size();
    }

    /** \returns the number of rows and columns of the marginal bandwidth */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return m_differenceIndices.size();
    }

    /** \returns the source's state entries this marginal selects */
    [[nodiscard]] IndexVectorType const &
    indices() const
    {
        return m_indices;
    }

    /** \returns the source's bandwidth rows this marginal selects */
    [[nodiscard]] DifferenceIndexVectorType const &
    differenceIndices() const
    {
        return m_differenceIndices;
    }

    /** \returns the selected rows of the source's states, as a lazy expression */
    [[nodiscard]] auto
    states() const
    {
        return source().states()(m_indices, Eigen::placeholders::all);
    }

    /** \returns the selected entries of particle \a i's state, as a lazy expression */
    [[nodiscard]] auto
    state(Index i) const
    {
        return source().state(i)(m_indices);
    }

    /** \returns the source's log-weights, unchanged by the marginal */
    [[nodiscard]] LogWeightsType
    logWeights() const
    {
        return source().logWeights();
    }

    /** \returns particle \a i's log-weight, as in the source */
    [[nodiscard]] Scalar
    logWeight(Index i) const
    {
        return source().logWeight(i);
    }

    /** \returns particle \a i's log-weight and the selected entries of its state, which must not
     * outlive this view
     */
    [[nodiscard]] Element
    operator[](Index i) const
    {
        return Element{logWeight(i), state(i)};
    }

    /** \returns an iterator to the first particle's element */
    [[nodiscard]] const_iterator
    begin() const
    {
        return const_iterator{*this, 0};
    }

    /** \returns an iterator past the last particle's element */
    [[nodiscard]] const_iterator
    end() const
    {
        return const_iterator{*this, size()};
    }

    /** \returns the marginal of the stored bandwidth, if the source has one */
    [[nodiscard]] std::optional<MatrixType>
    bandwidth() const
    {
        auto const & full = source().bandwidth();
        if (!full)
        {
            return std::nullopt;
        }
        return MatrixType{internal::lowerSymmetricBlock(*full, m_differenceIndices, m_differenceIndices)};
    }

    /** \returns an independent ParticleSet holding this marginal, with the
     * source's weights, the marginal of any stored bandwidth, and the source's allocator
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
        auto result = ParamsType{m_space, states(), logWeights(), allocator};
        if (auto const h = bandwidth())
        {
            result.setBandwidth(*h);
        }
        return result;
    }

private:

    /** \returns the viewed source */
    [[nodiscard]] Source const &
    source() const
    {
        return m_holder.get();
    }

    internal::HolderFor<Source> m_holder;           ///< the check of the particle set
    IndexVectorType m_indices;                      ///< the selected state entries
    DifferenceIndexVectorType m_differenceIndices;  ///< the selected bandwidth rows
    SubSpace m_space;                               ///< the selected block's space
};

/** \ingroup Distributions
 * \returns Silverman's rule-of-thumb bandwidth
 * \f[ H = \left(\frac{4}{N_\mathrm{eff}\,(d + 2)}\right)^{2/(d+4)} \Sigma, \f]
 * where \f$ \Sigma \f$ is the particles' weighted covariance, \f$ d \f$ their
 * dimension, and \f$ N_\mathrm{eff} \f$ their effective sample size; or a WeightError
 * if the weights cannot be normalized.
 *
 * The normal reference rule of \ref Silverman1986 "Silverman (1986)", as given in
 * [Multivariate KDE](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb),
 * for a %Gaussian kernel with the data's covariance as its shape, with the effective sample size
 * in place of the sample count for weighted particles. Near-optimal for roughly %Gaussian
 * particle clouds; scale it down for multimodal ones.
 */
template <typename Scalar, typename Space, typename Allocator>
[[nodiscard]] std::expected<typename ParticleSet<Space, Scalar, Allocator>::MatrixType, WeightError>
silvermanBandwidth(ParticleSet<Space, Scalar, Allocator> const & particles)
{
    return silvermanBandwidth(execution::sequential, particles);
}

/** \ingroup Distributions
 * silvermanBandwidth() under \a policy: the total weight, effective sample size, and covariance
 * are reductions over the particles (see \ref Execution).
 */
template <ExecutionPolicy P, typename Scalar, typename Space, typename Allocator>
[[nodiscard]] std::expected<typename ParticleSet<Space, Scalar, Allocator>::MatrixType, WeightError>
silvermanBandwidth(P && policy, ParticleSet<Space, Scalar, Allocator> const & particles)
{
    using MomentsType = typename ParticleSet<Space, Scalar, Allocator>::MomentsType;
    return particles.logTotalWeight(policy).transform([&](Scalar) {
        return internal::silvermanBandwidth<MomentsType>(
            policy,
            particles.space(),
            particles.states(),
            particles.logWeights()
        );
    });
}

}  // namespace Emrys

// KernelDensity completes ParticleSet::toDensity(); it includes this header first.
#include <EmrysFilter/Distributions/KernelDensity.h>

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a particles: their count, dimension, and effective sample size, and their weighted
 * mean and covariance, not the particles themselves.
 */
template <typename Space_, typename Scalar_, typename Allocator_>
void
describe(std::ostream & stream, ParticleSet<Space_, Scalar_, Allocator_> const & particles, Indent indent)
{
    stream << indent << "particle set of ";
    printCount(stream, particles.size(), "particle");
    printDimension(stream, particles.valueDimension(), particles.differenceDimension());
    if (auto const ess = particles.ess())
    {
        stream << ", effective sample size " << *ess;
    }
    printMoments(stream, indent.deeper(), particles.moments());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
