#pragma once

/** \file
 * GaussianMixtureHistory, the record of a %Gaussian mixture filter's forward pass, whose backward
 * pass follows each component's lineage through merges, with smoothStep() for each component and
 * each parent it descends from (decisions.md [smoothing]). The filter is unchanged but for predictJoint() (see
 * GaussianMixturePredictedJoint in MixtureFilter.h), and the mixture keeps a MixtureLineage
 * (Mixture.h). Read MixtureFilter.h first, and ParticleHistory.h next.
 */

#include <cstddef>
#include <list>
#include <optional>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Smoothers/Lag.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** The scratch of a GaussianMixtureHistory's backward step: the smoothed %Gaussian and log-weight of
 * every (child, parent) pair, grouped by parent, where each group starts, and each parent's
 * smoothed log-weight.
 */
template <typename GaussianType, typename Allocator>
struct MixtureSmoothingScratch
{
    using Scalar = typename GaussianType::Scalar;  ///< the scalar type

    Buffer<GaussianType, Allocator> pairs;     ///< each pair's smoothed %Gaussian, a parent's together
    Buffer<Scalar, Allocator> pairLogWeights;  ///< each pair's log-weight
    Buffer<Eigen::Index, Allocator> offsets;   ///< where each parent's pairs start, and the end
    Buffer<Scalar, Allocator> logWeights;      ///< each parent's smoothed log-weight

    /** Empty scratch from \a allocator. */
    explicit MixtureSmoothingScratch(Allocator const & allocator = {})
        : pairs(ReBound<Allocator, GaussianType>(allocator))
        , pairLogWeights(ReBound<Allocator, Scalar>(allocator))
        , offsets(ReBound<Allocator, Eigen::Index>(allocator))
        , logWeights(ReBound<Allocator, Scalar>(allocator))
    {}
};

/** Fills \a smoothed, the smoothed mixture at the start of \a joint's step, one component per
 * component the step started from, from \a next, the smoothed mixture at its end, and
 * \a nextLineage, how the components at the end descend from those at the start: the backward step
 * of GaussianMixtureHistory::smooth(), which documents it. Allocates nothing for a bounded space
 * once \a scratch has held as many pairs.
 */
template <
    ExecutionPolicy P,
    typename Joint,
    typename Gains,
    typename Next,
    typename NextLineage,
    typename Smoothed,
    typename Scratch>
void
smoothMixtureStep(
    P && policy,
    Joint const & joint,
    Gains const & gains,
    Next const & next,
    NextLineage const & nextLineage,
    Smoothed & smoothed,
    Scratch & scratch
)
{
    using Scalar = typename Joint::Scalar;
    using GaussianType = typename Joint::JointType::GaussianType;
    using Index = Eigen::Index;
    Scalar const minusInfinity = -Eigen::NumTraits<Scalar>::infinity();
    Index const parents = Index(joint.joints.size());
    Index const children = next.size();
    EMRYS_FILTER_ASSERT(smoothed.size() == parents && "the smoothed mixture must have a component per parent");
    EMRYS_FILTER_ASSERT(Index(gains.size()) == parents && "the gains must have one per parent");
    EMRYS_FILTER_ASSERT(
        nextLineage.parentCount() == parents && nextLineage.size() == children
        && "the lineage must have a row per component at the step's end and a share per component at its start"
    );
    auto const logTotal = next.logTotalWeight();
    EMRYS_FILTER_ASSERT(logTotal && "the smoothed mixture at the step's end must have valid weights");
    auto const logChildWeight = [&](Index p) {
        return next[p].logWeight - *logTotal;
    };
    auto const paired = [&](Index p, Index n) {
        return logChildWeight(p) > minusInfinity && nextLineage.logShare(p, n) > minusInfinity;
    };

    // The pairs (p, n) of child p at the end and its parent n at the start, grouped by parent.
    scratch.offsets.assign(std::size_t(parents + 1), 0);
    for (Index n = 0; n < parents; ++n)
    {
        Index count = 0;
        for (Index p = 0; p < children; ++p)
        {
            count += paired(p, n) ? 1 : 0;
        }
        scratch.offsets[std::size_t(n + 1)] = scratch.offsets[std::size_t(n)] + count;
    }
    std::size_t const total = std::size_t(scratch.offsets.back());
    scratch.pairs.resize(total);
    scratch.pairLogWeights.resize(total);
    scratch.logWeights.resize(std::size_t(parents));

    internal::forEachIndex(policy, parents, [&](Index n) {
        auto const & parentJoint = joint.joints[std::size_t(n)];
        Index const first = scratch.offsets[std::size_t(n)];
        Index slot = first;
        for (Index p = 0; p < children; ++p)
        {
            if (paired(p, n))
            {
                // An RTS step from parent n to child p, weighted as (11) with the normalized share of (15).
                scratch.pairs[std::size_t(slot)] = smoothStep(parentJoint, gains[std::size_t(n)], next[p].component);
                scratch.pairLogWeights[std::size_t(slot)] = logChildWeight(p) + nextLineage.logShare(p, n);
                ++slot;
            }
        }
        auto weight = LogSumExp<Scalar>{};
        for (Index i = first; i < slot; ++i)
        {
            weight.add(scratch.pairLogWeights[std::size_t(i)]);
        }
        scratch.logWeights[std::size_t(n)] = weight.value();
    });
    smoothed.forEach(policy, [&](auto & entry) {
        Index const n = entry.index();
        Index const first = scratch.offsets[std::size_t(n)];
        Index const count = scratch.offsets[std::size_t(n + 1)] - first;
        if (count == 0)
        {
            // No child at the end descends from it: it keeps its filtered Gaussian, with no weight.
            entry.setComponent(joint.joints[std::size_t(n)].filtered);
            return;
        }
        // The pairs that share parent n are merged into one, the group merge of Algorithm 2.
        entry.setComponent(
            ComponentTraits<GaussianType>::collapse(
                count,
                [&](Index i) -> GaussianType const & {
                    return scratch.pairs[std::size_t(first + i)];
                },
                [&](Index i) {
                    return scratch.pairLogWeights[std::size_t(first + i)];
                }
            )
        );
    });
    for (Index n = 0; n < parents; ++n)
    {
        smoothed.setLogWeight(n, scratch.logWeights[std::size_t(n)]);
    }
    // A child with weight descends from some parent with a share, so some parent has weight.
    [[maybe_unused]] auto const normalized = smoothed.normalize();
    EMRYS_FILTER_ASSERT(normalized && "nothing at the step's end descends from a component at its start");
}

}  // namespace internal

/** \ingroup Smoothers
 * \brief The record of a %Gaussian mixture filter's forward pass that its smoother runs backward
 * over: one GaussianMixturePredictedJoint per prediction, each with room for its smoothed mixture.
 *
 * It is to a MixtureFilter of Gaussians what GaussianHistory is to a KalmanFilter (decisions.md
 * [smoothing]): push() records the joint the filter's predictJoint() returns, and smooth() takes the
 * filter's current state and fills every step's smoothed mixture, from the newest step back.
 *
 * \code
 * GaussianMixtureHistory<State> history;
 * for (auto const & z : measurements)
 * {
 *     history.push(filter.predictJoint(dt));
 *     static_cast<void>(filter.update(fix(z), reductions::keep()));
 *     static_cast<void>(reduce(filter.state()));  // the caller's merge, as in examples/gaussian_mixture.cpp
 * }
 * history.smooth(filter.state());
 * for (auto const & step : history) { use(step.smoothed); }  // p(x_k | z_{1:N}), oldest first
 * \endcode
 *
 * Between two predictions the filter's updates and reductions branch, reweight, merge, and drop
 * components, and its state's lineage (see MixtureLineage) records how each component descends
 * from those at the last prediction: each component's share from each of them. Each backward step
 * reads it:
 *
 * - every pair of a smoothed component \a p at \a k + 1 and a component \a n at \a k that it has a
 *   share from is smoothed by smoothStep(), from \a n's filtered %Gaussian through its prediction to
 *   \a p's smoothed %Gaussian;
 * - the pair's weight is \a p's smoothed weight times its share from \a n,
 *   \f$ w^s_{k+1,p} \, P(n \mid m) \f$, where \a m is the filtered component at \a k + 1 that
 *   \a p smooths;
 * - the pairs that share a parent \a n are merged into one %Gaussian by moment matching, so the
 *   smoothed mixture at \a k has one component per filtered component at \a k, in the same order,
 *   with the summed weight.
 *
 * A component at \a k that nothing at \a k + 1 descends from, such as one the filter pruned, keeps
 * its filtered %Gaussian with zero weight, so that the smoothed mixture lines up with the filtered
 * one and with the lineage of the step before.
 *
 * The forward-backward smoother on merged %Gaussian mixtures of
 * \ref Rahmathullah2014 "Rahmathullah, Svensson, and Svensson (2014)", §IV and Algorithm 2, whose
 * f-graph is the lineage and whose s-graph is the smoothed mixtures. Four departures. The paper
 * writes each pair's weight as \f$ w^s_{k+1,p} \, w^f_{k+1,m}(r) \f$ in its (16), but its own
 * (15), from which (16) follows, divides the branch weight by the merged node's,
 * \f$ \Pr\{H^f_{k+1,m}\} = \sum_r w^f_{k+1,m}(r) \f$; without it, pairs through heavy nodes are
 * counted twice over, so this takes (15), the share. The lineage keeps the share per parent rather
 * than one weight per merged branch, which is the same once the branches from one parent are summed,
 * as their pairs would be. The merge within each group is the full collapse to one %Gaussian rather
 * than the paper's threshold merging and pruning, which bounds the smoothed mixture by the filtered
 * one and keeps them aligned. And the step is smoothStep(), in the cross-covariance form and the
 * tangent space, so that it serves every prediction method and manifold states. With a policy that
 * keeps one component per component and no merging, each component has one parent, and this is the
 * Rauch-Tung-Striebel smoother per component; with Keep and no merging, on a plain vector space,
 * each smoothed component has the weight and moments of the exact %Gaussian-sum smoother's branches
 * through it, since the RTS step is affine in the moments it smooths toward.
 *
 * Successive records must be one predictJoint() apart, which the history asserts from the lineage's
 * count: a filter state replaced by a copy has no lineage, and one predicted by predict() between
 * records has no mark for it.
 *
 * As with GaussianHistory, a Lag keeps the newest steps and recycles the oldest step's storage
 * once full, so with one joint kept and refilled by predictJoint(dt, joint), at a steady number of
 * components fixed-lag smoothing allocates nothing once the window is full. Each step's joint and
 * smoothed mixture, and the backward step's scratch, come from the history's allocator.
 *
 * \tparam Space_      the state space
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the steps, what they hold, and the scratch (see \ref Allocation)
 *
 * \sa GaussianMixturePredictedJoint, MixtureFilter::predictJoint(), MixtureLineage, GaussianHistory,
 * \ref Smoothers
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class GaussianMixtureHistory
{
public:

    using Scalar = Scalar_;                                                        ///< the scalar type
    using SpaceType = Space_;                                                      ///< the state space
    using JointType = GaussianMixturePredictedJoint<Space_, Scalar_, Allocator_>;  ///< what each step records
    using SmoothedType = GaussianMixtureParams<Space_, Scalar_, Allocator_>;       ///< the smoothed mixture's type
    using allocator_type = Allocator_;  ///< the allocator the steps come from
    using Index = Eigen::Index;         ///< Eigen's index type

    /** One recorded prediction and its smoothed mixture, built from the history's allocator. */
    class Step
    {
    public:

        using allocator_type = Allocator_;  ///< the allocator its joint and mixture come from

        /** One component's Rauch-Tung-Striebel gain. */
        using GainType = typename GaussianParams<Space_, Scalar_>::MatrixType;

        /** The allocator of the gains. */
        using GainAllocator = internal::ReBound<Allocator_, GainType>;

        /** A copy of \a recorded in storage from \a allocator, with the filtered mixture as its
         * smoothed one until smooth() runs.
         */
        explicit Step(JointType const & recorded, Allocator_ const & allocator = {})
            : joint{copyJoint(recorded, allocator)}
            , smoothed{joint.joints.front().filtered.space(), allocator}
            , gains(GainAllocator(allocator))
        {
            smoothed.reserve(Index(joint.joints.size()));
            resetSmoothed();
            computeGains();
        }

        /** \a recorded moved into storage from \a allocator, with the filtered mixture as its
         * smoothed one until smooth() runs.
         */
        explicit Step(JointType && recorded, Allocator_ const & allocator = {})
            : joint{moveJoint(std::move(recorded), allocator)}
            , smoothed{joint.joints.front().filtered.space(), allocator}
            , gains(GainAllocator(allocator))
        {
            smoothed.reserve(Index(joint.joints.size()));
            resetSmoothed();
            computeGains();
        }

        /** Copies \a other into storage from \a allocator. */
        Step(Step const & other, Allocator_ const & allocator)
            : joint{copyJoint(other.joint, allocator)}
            , smoothed{other.smoothed, allocator}
            , gains(other.gains, GainAllocator(allocator))
        {}

        /** Moves \a other into storage from \a allocator, taking its storage when the allocators
         * are equal.
         */
        Step(Step && other, Allocator_ const & allocator)
            : joint{moveJoint(std::move(other.joint), allocator)}
            , smoothed{std::move(other.smoothed), allocator}
            , gains(std::move(other.gains), GainAllocator(allocator))
        {}

        Step(Step const &) = default;  ///< copies the joint and the smoothed mixture
        Step(Step &&) = default;       ///< moves the joint and the smoothed mixture

        /** Copies into this step's storage. */
        Step &
        operator=(Step const &) = default;

        /** Takes the other step's storage, as the members' do. */
        Step &
        operator=(Step &&) = default;

        ~Step() = default;

        /** Replaces the smoothed mixture with the filtered one: the joint's weights and each
         * component's filtered %Gaussian.
         */
        void
        resetSmoothed()
        {
            smoothed.clear();
            for (std::size_t n = 0; n < joint.joints.size(); ++n)
            {
                static_cast<void>(smoothed.add(joint.logWeights[n], joint.joints[n].filtered));
            }
        }

        /** Computes each component's gain from its joint (internal::rtsGain()), once per recording. */
        void
        computeGains()
        {
            gains.resize(joint.joints.size());
            for (std::size_t n = 0; n < gains.size(); ++n)
            {
                gains[n] = internal::rtsGain(joint.joints[n]);
            }
        }

        JointType joint;  ///< the prediction as predictJoint() recorded it

        /** \f$ p(x_k \mid z_{1:N}) \f$, a component per component of the joint, as of the last
         * smooth(); the filtered mixture until then
         */
        SmoothedType smoothed;

        /** Each component's Rauch-Tung-Striebel gain, made once when the joint is recorded. */
        internal::Buffer<GainType, Allocator_> gains;

    private:

        /** \returns \a recorded copied into storage from \a allocator */
        [[nodiscard]] static JointType
        copyJoint(JointType const & recorded, Allocator_ const & allocator)
        {
            return {
                typename JointType::LogWeightsType(recorded.logWeights, allocator),
                typename JointType::JointsType(recorded.joints, allocator),
                typename JointType::LineageType(recorded.lineage, allocator)
            };
        }

        /** \returns \a recorded moved into storage from \a allocator */
        [[nodiscard]] static JointType
        moveJoint(JointType && recorded, Allocator_ const & allocator)
        {
            return {
                typename JointType::LogWeightsType(std::move(recorded.logWeights), allocator),
                typename JointType::JointsType(std::move(recorded.joints), allocator),
                typename JointType::LineageType(std::move(recorded.lineage), allocator)
            };
        }
    };

private:

    using StepAllocator = internal::ReBound<Allocator_, Step>;  ///< allocates the list's nodes
    using ListType = std::list<Step, StepAllocator>;            ///< the steps, oldest first

    /** the backward step's scratch */
    using ScratchType = internal::MixtureSmoothingScratch<GaussianParams<Space_, Scalar_>, Allocator_>;

public:

    using const_iterator = typename ListType::const_iterator;  ///< iterates the steps, oldest first

    /** An empty history that keeps every step. */
    GaussianMixtureHistory() = default;

    /** An empty history that keeps every step, with storage from \a allocator. */
    explicit GaussianMixtureHistory(Allocator_ const & allocator)
        : m_steps(StepAllocator(allocator))
        , m_scratch{allocator}
    {}

    /** An empty history that keeps the newest \a lag steps, with storage from \a allocator. */
    explicit GaussianMixtureHistory(Lag lag, Allocator_ const & allocator = {})
        : m_steps(StepAllocator(allocator))
        , m_scratch{allocator}
        , m_lag{lag.steps}
    {
        EMRYS_FILTER_ASSERT(lag.steps > 0 && "a fixed-lag history keeps at least one step");
    }

    /** Copies the steps into nodes from \a other's allocator (see \ref Allocation). */
    GaussianMixtureHistory(GaussianMixtureHistory const & other)
        : GaussianMixtureHistory{other, other.get_allocator()}
    {}

    /** Copies \a other's steps into storage from \a allocator. */
    GaussianMixtureHistory(GaussianMixtureHistory const & other, Allocator_ const & allocator)
        : m_steps(other.m_steps, StepAllocator(allocator))
        , m_scratch{allocator}
        , m_lag{other.m_lag}
    {}

    /** Takes \a other's steps. */
    GaussianMixtureHistory(GaussianMixtureHistory && other) = default;

    /** Moves \a other's steps into storage from \a allocator, taking its storage when the
     * allocators are equal.
     */
    GaussianMixtureHistory(GaussianMixtureHistory && other, Allocator_ const & allocator)
        : m_steps(std::move(other.m_steps), StepAllocator(allocator))
        , m_scratch{allocator}
        , m_lag{other.m_lag}
    {}

    /** Copies \a other's steps and lag. */
    GaussianMixtureHistory &
    operator=(GaussianMixtureHistory const & other) = default;

    /** Takes \a other's steps and lag, as \c std::list's move assignment does. */
    GaussianMixtureHistory &
    operator=(GaussianMixtureHistory && other) = default;

    ~GaussianMixtureHistory() = default;

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

    /** Records a copy of \a joint as the newest step; it must come from the predictJoint() after
     * the newest step's. A full fixed-lag history drops its oldest step and copies \a joint into
     * its storage.
     */
    void
    push(JointType const & joint)
    {
        record(joint);
    }

    /** Records \a joint as the newest step, taking its storage unless the history is a full
     * fixed-lag one, which copies it into its oldest step's.
     */
    void
    push(JointType && joint)
    {
        record(std::move(joint));
    }

    /** Fills every step's smoothed mixture by the backward step above, from the newest back, given
     * \a latest, the filter's state after the newest step: its posterior
     * \f$ p(x_N \mid z_{1:N}) \f$, which is its own smoothed mixture, with its lineage from the
     * newest step.
     */
    template <typename LatestAllocator>
    void
    smooth(GaussianMixtureParams<Space_, Scalar_, LatestAllocator> const & latest)
    {
        smooth(execution::sequential, latest);
    }

    /** smooth() under \a policy: each backward step's pairs and merges are a for-each over the
     * step's components (see \ref Execution), so the smoothed mixtures are Sequential's exactly.
     */
    template <ExecutionPolicy P, typename LatestAllocator>
    void
    smooth(P && policy, GaussianMixtureParams<Space_, Scalar_, LatestAllocator> const & latest)
    {
        auto step = m_steps.rbegin();
        if (step == m_steps.rend())
        {
            return;
        }
        assertFollows(latest.lineage(), *step);
        internal::smoothMixtureStep(
            policy,
            step->joint,
            step->gains,
            latest,
            latest.lineage(),
            step->smoothed,
            m_scratch
        );
        for (auto next = step++; step != m_steps.rend(); next = step++)
        {
            internal::smoothMixtureStep(
                policy,
                step->joint,
                step->gains,
                next->smoothed,
                next->joint.lineage,
                step->smoothed,
                m_scratch
            );
        }
    }

    /** Drops every step, keeping the lag. */
    void
    clear()
    {
        m_steps.clear();
    }

private:

    /** The body of both push(): records \a joint, a JointType forwarded from push(). */
    template <typename Joint>
    void
    record(Joint && joint)
    {
        EMRYS_FILTER_ASSERT(
            !joint.joints.empty() && joint.logWeights.size() == joint.joints.size() && "the joint has no components"
        );
        if (!empty())
        {
            assertFollows(joint.lineage, back());
        }
        if (m_lag > 0 && size() == m_lag)
        {
            m_steps.splice(m_steps.end(), m_steps, m_steps.begin());
            Step & step = m_steps.back();
            step.joint.logWeights.assign(joint.logWeights.begin(), joint.logWeights.end());
            step.joint.joints.assign(joint.joints.begin(), joint.joints.end());
            step.joint.lineage = joint.lineage;
            step.resetSmoothed();
            step.computeGains();
        }
        else
        {
            m_steps.emplace_back(std::forward<Joint>(joint));
        }
    }

    /** A programmer error unless \a lineage, of the components after \a step, was started by the
     * predictJoint() that recorded \a step: one mark later than \a step's own, with a parent per
     * component \a step recorded.
     */
    template <typename Lineage>
    static void
    assertFollows([[maybe_unused]] Lineage const & lineage, [[maybe_unused]] Step const & step)
    {
        EMRYS_FILTER_ASSERT(
            lineage.count() == step.joint.lineage.count() + 1
            && lineage.parentCount() == Index(step.joint.joints.size())
            && "the lineage must start at the newest record's predictJoint(): one predictJoint() apart, and no copy of "
               "the state between"
        );
    }

    ListType m_steps;       ///< the steps, oldest first
    ScratchType m_scratch;  ///< the backward step's pairs and weights
    Index m_lag = 0;        ///< the most steps kept, or zero to keep every step
};

}  // namespace Emrys
