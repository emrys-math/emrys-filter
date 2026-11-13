#pragma once

/** \file
 * ParticleHistory, the record of a particle filter's particles and their genealogy, smoothed by
 * ancestor tracing (decisions.md [smoothing]). The filter is unchanged: the history records its
 * state. Read ParticleFilter.h first, and CategoricalHistory.h next.
 */

#include <cstddef>
#include <list>
#include <optional>
#include <span>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Smoothers/Lag.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Smoothers
 * \brief The record of a particle filter's forward pass that ancestor tracing runs backward over:
 * the particles at each step, how each descends from the step before, and room for the step's
 * smoothed particles.
 *
 * The caller keeps it, outside the filter (decisions.md [smoothing]): push() records the filter's
 * particles before each prediction, and smooth() takes its current particles and fills every
 * step's smoothed set by following each current particle's genealogy back through the steps.
 * Prediction and update keep each particle in its place, so the only change of lineage between
 * two records is a resample, whose ancestors() the history takes when the set's resampleCount()
 * says one happened. At most one resample may fall between two records, since ancestors() keeps
 * only the last, and nothing else may renumber the particles there (ParticleSet::removeIf(), a
 * prune, insertion, or erasure): each would break the genealogy without the history knowing.
 *
 * \code
 * auto window = ParticleHistory<State>{Lag{10}};
 * for (auto const & z : measurements)
 * {
 *     window.push(filter.state());
 *     filter.predict(rng, dt);
 *     static_cast<void>(filter.update(fix(z)));
 *     static_cast<void>(upkeep(filter.state(), rng));
 *     window.smooth(filter.state());
 *     use(window.front().smoothed.moments());  // x_{k-10|k}
 * }
 * \endcode
 *
 * Each step's smoothed set holds, as its particle \a j, the state at that step of the trajectory
 * that ends in the current particle \a j, with the current particle's weight: a weighted sample of
 * the smoothing distribution \f$ p(x_k \mid z_{1:N}) \f$, whose moments(), kernel density, and
 * marginals are those of any ParticleSet. Read across the steps, particle \a j's states are its
 * trajectory.
 *
 * Ancestor tracing is the smoother of \ref Kitagawa1996 "Kitagawa (1996)", who stores each
 * particle's path and resamples the paths with the particles. The departure is storage: this keeps
 * each step's particles once and the indices that join them, and traces the paths when asked, which
 * costs \f$ O(N L) \f$ index lookups and copies per smooth() rather than copying paths at every
 * resample. Each resample keeps only some lineages, so going back the traced paths pass through
 * fewer and fewer distinct particles and the smoothed sets lose their spread: it is a fixed-lag
 * smoother, for a lag short enough that the paths have not yet collapsed. Backward simulation,
 * which does not degenerate, needs the transition density and is not yet provided [smoothing].
 *
 * As with GaussianHistory, a Lag keeps the newest steps and reuses the oldest step's node once
 * full, copying the particles into its storage, so at a steady particle count fixed-lag smoothing
 * allocates nothing once the window is full. Each step's sets and indices come from the history's
 * allocator, by uses-allocator construction. The recorded sets are copies, so they carry no
 * resample record of their own (see ParticleSet::resampleCount()).
 *
 * \tparam Space_      the state space
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the steps, their particles, and the tracing scratch (see
 *                     \ref Allocation)
 *
 * \sa ParticleFilter, ParticleSet::ancestors(), GaussianHistory, \ref Smoothers
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class ParticleHistory
{
public:

    using Scalar = Scalar_;                                            ///< the scalar type
    using SpaceType = Space_;                                          ///< the state space
    using ParticleSetType = ParticleSet<Space_, Scalar_, Allocator_>;  ///< what each step records
    using allocator_type = Allocator_;                                 ///< the allocator the steps come from
    using Index = Eigen::Index;                                        ///< Eigen's index type

private:

    using IndexBuffer = internal::Buffer<Index, Allocator_>;      ///< parents and lineages
    using IndexAllocator = typename IndexBuffer::allocator_type;  ///< allocates them

public:

    /** One recorded set of particles, its descent from the step before, and its smoothed set,
     * built from the history's allocator.
     */
    class Step
    {
    public:

        using allocator_type = Allocator_;  ///< the allocator its sets and indices come from

        /** A copy of \a recorded in storage from \a allocator, with the recorded particles as the
         * smoothed ones until smooth() runs, and room for a parent per particle.
         */
        explicit Step(ParticleSetType const & recorded, Allocator_ const & allocator = {})
            : particles{recorded, allocator}
            , smoothed{recorded, allocator}
            , parents(IndexAllocator(allocator))
        {
            parents.reserve(std::size_t(recorded.size()));
        }

        /** Copies \a other into storage from \a allocator. */
        Step(Step const & other, Allocator_ const & allocator)
            : particles{other.particles, allocator}
            , smoothed{other.smoothed, allocator}
            , parents(other.parents, IndexAllocator(allocator))
            , resampled{other.resampled}
        {}

        /** Moves \a other into storage from \a allocator, taking its storage when the allocators
         * are equal.
         */
        Step(Step && other, Allocator_ const & allocator)
            : particles{std::move(other.particles), allocator}
            , smoothed{std::move(other.smoothed), allocator}
            , parents(std::move(other.parents), IndexAllocator(allocator))
            , resampled{other.resampled}
        {}

        Step(Step const &) = default;  ///< copies the sets and the parents
        Step(Step &&) = default;       ///< moves the sets and the parents

        /** Copies into this step's storage. */
        Step &
        operator=(Step const &) = default;

        /** Takes the other step's storage, as the members' do. */
        Step &
        operator=(Step &&) = default;

        ~Step() = default;

        /** The particles as push() recorded them, before the step's prediction. */
        ParticleSetType particles;

        /** \f$ p(x_k \mid z_{1:N}) \f$ for the step's \f$ x_k \f$, as of the last smooth(): the
         * traced states with the latest weights; a copy of particles until then
         */
        ParticleSetType smoothed;

        /** If resampled, particle \a i here descends from particle <tt>parents[i]</tt> of the step
         * before; otherwise from particle \a i, and parents is unused
         */
        IndexBuffer parents;

        bool resampled = false;  ///< whether a resample fell between the step before and this one
    };

private:

    using StepAllocator = internal::ReBound<Allocator_, Step>;  ///< allocates the list's nodes
    using ListType = std::list<Step, StepAllocator>;            ///< the steps, oldest first

public:

    using const_iterator = typename ListType::const_iterator;  ///< iterates the steps, oldest first

    /** An empty history that keeps every step. */
    ParticleHistory() = default;

    /** An empty history that keeps every step, with storage from \a allocator. */
    explicit ParticleHistory(Allocator_ const & allocator)
        : m_steps(StepAllocator(allocator))
        , m_lineage(IndexAllocator(allocator))
        , m_traced(internal::bufferOf<Scalar>(allocator))
    {}

    /** An empty history that keeps the newest \a lag steps, with storage from \a allocator. */
    explicit ParticleHistory(Lag lag, Allocator_ const & allocator = {})
        : m_steps(StepAllocator(allocator))
        , m_lineage(IndexAllocator(allocator))
        , m_traced(internal::bufferOf<Scalar>(allocator))
        , m_lag{lag.steps}
    {
        EMRYS_FILTER_ASSERT(lag.steps > 0 && "a fixed-lag history keeps at least one step");
    }

    /** Copies the steps into nodes from \a other's allocator (see \ref Allocation). */
    ParticleHistory(ParticleHistory const & other)
        : ParticleHistory{other, other.get_allocator()}
    {}

    /** Copies \a other's steps into storage from \a allocator. */
    ParticleHistory(ParticleHistory const & other, Allocator_ const & allocator)
        : m_steps(other.m_steps, StepAllocator(allocator))
        , m_lineage(IndexAllocator(allocator))
        , m_traced(internal::bufferOf<Scalar>(allocator))
        , m_lag{other.m_lag}
        , m_resampleCount{other.m_resampleCount}
    {}

    /** Takes \a other's steps. */
    ParticleHistory(ParticleHistory && other) = default;

    /** Moves \a other's steps into storage from \a allocator, taking its storage when the
     * allocators are equal.
     */
    ParticleHistory(ParticleHistory && other, Allocator_ const & allocator)
        : m_steps(std::move(other.m_steps), StepAllocator(allocator))
        , m_lineage(IndexAllocator(allocator))
        , m_traced(internal::bufferOf<Scalar>(allocator))
        , m_lag{other.m_lag}
        , m_resampleCount{other.m_resampleCount}
    {}

    /** Copies \a other's steps and lag. */
    ParticleHistory &
    operator=(ParticleHistory const & other) = default;

    /** Takes \a other's steps and lag, as \c std::list's move assignment does. */
    ParticleHistory &
    operator=(ParticleHistory && other) = default;

    ~ParticleHistory() = default;

    /** \returns the allocator the steps come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_steps.get_allocator());
    }

    /** \returns the most steps a fixed-lag history keeps, or nothing if it keeps every step */
    [[nodiscard]] std::optional<Index>
    lag() const
    {
        return m_lag > 0 ? std::optional<Index>{m_lag} : std::nullopt;
    }

    /** \returns the number of steps recorded, at most lag() */
    [[nodiscard]] Index
    size() const
    {
        return Index(m_steps.size());
    }

    /** \returns whether no step is recorded */
    [[nodiscard]] bool
    empty() const
    {
        return m_steps.empty();
    }

    /** \returns the oldest step kept; the history must not be empty */
    [[nodiscard]] Step const &
    front() const
    {
        EMRYS_FILTER_ASSERT(!empty() && "the history is empty");
        return m_steps.front();
    }

    /** \returns the newest step; the history must not be empty */
    [[nodiscard]] Step const &
    back() const
    {
        EMRYS_FILTER_ASSERT(!empty() && "the history is empty");
        return m_steps.back();
    }

    /** \returns an iterator to the oldest step */
    [[nodiscard]] const_iterator
    begin() const
    {
        return m_steps.begin();
    }

    /** \returns an iterator past the newest step */
    [[nodiscard]] const_iterator
    end() const
    {
        return m_steps.end();
    }

    /** Records \a particles, the filter's state before its next prediction, as the newest step,
     * with their descent from the step before. A full fixed-lag history drops its oldest step and
     * reuses its node.
     */
    void
    push(ParticleSetType const & particles)
    {
        bool const resampled = !empty() && newResample(particles);
        EMRYS_FILTER_ASSERT(
            (empty() || resampled || particles.size() == back().particles.size())
            && "without a resample the particles must be those of the step before"
        );
        if (m_lag > 0 && size() == m_lag)
        {
            m_steps.splice(m_steps.end(), m_steps, m_steps.begin());
            Step & step = m_steps.back();
            step.particles = particles;
            step.smoothed = particles;
        }
        else
        {
            m_steps.emplace_back(particles);
        }
        Step & step = m_steps.back();
        step.resampled = resampled;
        if (resampled)
        {
            std::span<Index const> const ancestors = particles.ancestors();
            step.parents.assign(ancestors.begin(), ancestors.end());
        }
        m_resampleCount = particles.resampleCount();
    }

    /** Fills every step's smoothed set by ancestor tracing, from the newest back, given \a latest,
     * the filter's particles after the newest step. The smoothed sets store no bandwidth, so their
     * kernel densities take Silverman's rule (see ParticleSet::kdeBandwidth()).
     */
    void
    smooth(ParticleSetType const & latest)
    {
        smooth(execution::sequential, latest);
    }

    /** smooth() under \a policy: each step's tracing and gather are for-eaches over the latest
     * particles (see \ref Execution), so the smoothed sets are Sequential's exactly.
     */
    template <ExecutionPolicy P>
    void
    smooth(P && policy, ParticleSetType const & latest)
    {
        if (empty())
        {
            return;
        }
        bool const resampled = newResample(latest);
        EMRYS_FILTER_ASSERT(
            (resampled || latest.size() == back().particles.size())
            && "without a resample the particles must be those of the newest step"
        );
        Index const count = latest.size();
        m_lineage.resize(std::size_t(count));
        std::span<Index const> const ancestors = latest.ancestors();
        internal::forEachIndex(policy, count, [&](Index j) {
            m_lineage[std::size_t(j)] = resampled ? ancestors[std::size_t(j)] : j;
        });
        Step const * next = nullptr;
        for (auto step = m_steps.rbegin(); step != m_steps.rend(); ++step)
        {
            if (next != nullptr && next->resampled)
            {
                internal::forEachIndex(policy, count, [&](Index j) {
                    Index const here = m_lineage[std::size_t(j)];
                    EMRYS_FILTER_ASSERT(here < Index(next->parents.size()) && "a traced index is out of range");
                    m_lineage[std::size_t(j)] = next->parents[std::size_t(here)];
                });
            }
            auto const states = step->particles.states();
            m_traced.resize(std::size_t(states.rows() * count));
            auto traced = SampleSetMap<Space_, Scalar_>{m_traced.data(), states.rows(), count};
            internal::gather(policy, std::span<Index const>{m_lineage}, states, traced);
            step->smoothed.assign(traced, latest.logWeights());
            step->smoothed.clearBandwidth();
            next = &*step;
        }
    }

    /** Drops every step, keeping the lag. */
    void
    clear()
    {
        m_steps.clear();
    }

private:

    /** \returns whether \a particles were resampled since the newest step was recorded, which
     * may have happened at most once
     */
    [[nodiscard]] bool
    newResample(ParticleSetType const & particles) const
    {
        Index const resamples = particles.resampleCount() - m_resampleCount;
        EMRYS_FILTER_ASSERT(
            resamples >= 0 && resamples <= 1 && "at most one resample may fall between two records of the particles"
        );
        return resamples == 1;
    }

    ListType m_steps;                               ///< the steps, oldest first
    IndexBuffer m_lineage;                          ///< each latest particle's index at the step being traced
    internal::Buffer<Scalar, Allocator_> m_traced;  ///< the traced states of the step being smoothed
    Index m_lag = 0;                                ///< the most steps kept, or zero to keep every step
    Index m_resampleCount = 0;                      ///< the resampleCount() of the newest step's particles
};

}  // namespace Emrys
