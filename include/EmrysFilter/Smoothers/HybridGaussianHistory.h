#pragma once

/** \file
 * HybridGaussianHistory, the record of an IMM filter's forward pass, whose backward pass
 * interacts its modes as the IMM does going forward, with smoothStep() for each mode (decisions.md
 * [smoothing]). The filter is unchanged but for predictJoint() (see HybridPredictedJoint in
 * ImmFilter.h). Read ImmFilter.h first, and GaussianMixtureHistory.h next.
 */

#include <cstddef>
#include <list>
#include <optional>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Smoothers/Lag.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** Fills \a smoothed, the smoothed hybrid state at the start of \a joint's step, from \a next, the
 * smoothed hybrid state at its end: the backward step of HybridGaussianHistory::smooth(), which
 * documents it. Allocates nothing for a bounded space and mode space.
 */
template <typename Joint, typename Gains, typename Next, typename Smoothed>
void
smoothHybridStep(Joint const & joint, Gains const & gains, Next const & next, Smoothed & smoothed)
{
    using Scalar = typename Joint::Scalar;
    using ModesType = typename Joint::ModesType;
    using ArrayType = typename ModesType::ArrayType;
    using GaussianType = GaussianParams<typename Joint::SpaceType, Scalar>;
    Scalar const minusInfinity = -Eigen::NumTraits<Scalar>::infinity();
    Eigen::Index const count = joint.modes.cardinality();
    EMRYS_FILTER_ASSERT(
        Eigen::Index(joint.joints.size()) == count && Eigen::Index(gains.size()) == count && next.size() == count
        && smoothed.size() == count
        && "the joint, its gains, the next state, and the smoothed state must have one Gaussian per mode"
    );
    auto const & pi = joint.transitions;
    auto const logFiltered = logProbabilitiesOf(joint.modes);
    auto const logNext = logProbabilitiesOf(next.modes());
    auto const logTransition = [&](Eigen::Index from, Eigen::Index to) {
        return pi(from, to) > Scalar(0) ? Scalar(Eigen::numext::log(pi(from, to))) : minusInfinity;
    };

    // e_i, the predicted probability of mode i at t + 1, normalizes the backward transition
    // b_ij = p_ji mu_t^j / e_i (8.11).
    auto logPredicted = ArrayType{};
    logPredicted.resize(count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        auto total = LogSumExp<Scalar>{};
        for (Eigen::Index l = 0; l < count; ++l)
        {
            total.add(logTransition(l, i) + logFiltered(l));
        }
        logPredicted(i) = total.value();
    }

    auto logMixing = ArrayType{};
    logMixing.resize(count);
    auto logModes = ArrayType{};
    logModes.resize(count);
    for (Eigen::Index j = 0; j < count; ++j)
    {
        // The backward mixing weights mu^{i|j} of (8.12), unnormalized: b_ij mu_{t+1|N}^i.
        for (Eigen::Index i = 0; i < count; ++i)
        {
            logMixing(i) = logPredicted(i) == minusInfinity
                             ? minusInfinity
                             : logTransition(j, i) + logFiltered(j) - logPredicted(i) + logNext(i);
        }
        auto const & modeJoint = joint.joints[std::size_t(j)];
        if (logTotalWeight<Scalar>(logMixing))
        {
            // Mixing (8.13, 8.14) and the mode-matched RTS step (8.15 to 8.17).
            GaussianType const mixed = ComponentTraits<GaussianType>::collapse(
                count,
                [&](Eigen::Index i) -> GaussianType const & {
                    return next.conditional(i);
                },
                [&](Eigen::Index i) {
                    return logMixing(i);
                }
            );
            smoothed.setConditional(j, smoothStep(modeJoint, gains[std::size_t(j)], mixed));
        }
        else
        {
            // Nothing smoothed at t + 1 comes from mode j, so it keeps its filtered Gaussian.
            smoothed.setConditional(j, modeJoint.filtered);
        }
    }

    // The mode probabilities by the expectation correction of Barber (2006), (12) and (13) under
    // its mean approximation: p(r_t = j | r_{t+1} = i, z_{1:N}) is b_ij reweighted by the density
    // of mode i's smoothed mean at t + 1 under mode j's filtered Gaussian moved through mode i's
    // model, normalized over j, and mu_{t|N}^j is its sum over i weighted by mu_{t+1|N}^i. Where no
    // mode's prediction gives that mean a density, b_ij is left as it is.
    auto const logAdd = [](Scalar a, Scalar b) {
        auto sum = LogSumExp<Scalar>{};
        sum.add(a);
        sum.add(b);
        return sum.value();
    };
    logModes.setConstant(minusInfinity);
    auto logPair = ArrayType{};
    logPair.resize(count);
    auto logBackward = ArrayType{};
    logBackward.resize(count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        if (logNext(i) == minusInfinity || logPredicted(i) == minusInfinity)
        {
            continue;
        }
        auto const & smoothedMean = next.conditional(i).mean();
        for (Eigen::Index j = 0; j < count; ++j)
        {
            logBackward(j) = logTransition(j, i) + logFiltered(j) - logPredicted(i);
            logPair(j) = minusInfinity;
            if (logBackward(j) > minusInfinity)
            {
                if (auto const density = joint.predictedThrough(j, i).toDensity())
                {
                    logPair(j) = logBackward(j) + density->logDensity(smoothedMean);
                }
            }
        }
        auto const logNormalizer = logTotalWeight<Scalar>(logPair);
        for (Eigen::Index j = 0; j < count; ++j)
        {
            Scalar const logGiven = logNormalizer ? logPair(j) - *logNormalizer : logBackward(j);
            logModes(j) = logAdd(logModes(j), logNext(i) + logGiven);
        }
    }
    auto modes = ModesType{joint.modes.space(), logModes};
    [[maybe_unused]] auto const normalized = modes.normalize();
    EMRYS_FILTER_ASSERT(normalized && "no mode smoothed at the step's end came from a mode at its start");
    smoothed.setModes(std::move(modes));
}

}  // namespace internal

/** \ingroup Smoothers
 * \brief The record of an IMM filter's forward pass that its smoother runs backward over: one
 * HybridPredictedJoint per prediction, each with room for its smoothed HybridGaussian.
 *
 * It is to an ImmFilter what GaussianHistory is to a KalmanFilter (decisions.md
 * [smoothing]): push() records the joint the filter's predictJoint() fills, and smooth() takes the
 * filter's current state and fills every step's smoothed hybrid state, from the newest step back.
 *
 * \code
 * HybridGaussianHistory<State, Maneuver> history;
 * HybridPredictedJoint<State, Maneuver> joint;
 * for (auto const & z : measurements)
 * {
 *     filter.predictJoint(dt, joint);
 *     history.push(joint);
 *     if (!filter.update(fix(z))) { break; }
 * }
 * history.smooth(filter.state());
 * for (auto const & step : history) { use(step.smoothed.modes(), step.smoothed.toDensity()->collapsed()); }
 * \endcode
 *
 * Each backward step runs the IMM in reverse. With \f$ \mu_{t \mid t} \f$ the filtered mode
 * probabilities, \f$ \Pi \f$ the chain, and the smoothed hybrid state at \f$ t + 1 \f$ given:
 *
 * - the backward transition \f$ b_{ij} = p_{ji} \mu^j_{t \mid t} / e_i \f$, the probability
 *   of mode \a j at \a t given mode \a i at \a t + 1, with
 *   \f$ e_i = \sum_l p_{li} \mu^l_{t \mid t} \f$;
 * - the backward mixing \f$ \mu^{i \mid j} \propto b_{ij} \mu^i_{t+1 \mid N} \f$, the probability
 *   of mode \a i at \a t + 1 given mode \a j at \a t, and for each \a j the moment-matched
 *   mixture of the smoothed Gaussians at \a t + 1 under it;
 * - for each \a j, smoothStep() from mode \a j's filtered %Gaussian, through the joint of its
 *   prediction by mode \a j's own model, to that mixture;
 * - the smoothed mode probabilities \f$ \mu^j_{t \mid N} = \sum_i \mu^i_{t+1 \mid N}\,
 *   p(r_t = j \mid r_{t+1} = i, z_{1:N}) \f$, with
 *   \f$ p(r_t = j \mid r_{t+1} = i, z_{1:N}) \propto b_{ij}\, \mathcal{N}(\hat x^i_{t+1 \mid N};
 *   \hat x^{j \to i}_{t+1 \mid t}, P^{j \to i}_{t+1 \mid t}) \f$ normalized over \a j, where
 *   \f$ \hat x^{j \to i}_{t+1 \mid t}, P^{j \to i}_{t+1 \mid t} \f$ is mode \a j's filtered
 *   %Gaussian moved through mode \a i's model (HybridPredictedJoint::predictedThrough()): the
 *   expectation correction of \ref Barber2006 "Barber (2006)", (12) and (13), under its mean
 *   approximation, which evaluates the average over the smoothed %Gaussian at its mean.
 *
 * The continuous part is the backward IMM smoother of \ref Nadarajah2009 "Nadarajah (2009)",
 * §8.2.1, equations 8.11 to 8.17, published as \ref Nadarajah2012 "Nadarajah, Tharmarasa,
 * McDonald, and Kirubarajan (2012)". It needs no backward-time dynamics, unlike the two-filter IMM
 * smoother, and runs one RTS step per mode, not one per pair of modes. The mode probabilities are
 * not its (8.18, 8.19), which weight each mode by \f$ \Lambda_j \f$, the density of the smoothed
 * means at \a t + 1 under its own prediction, without the smoothed mode probabilities at \a t + 1:
 * that left an autonomous multiple model filter's, which must stay at the final filtered ones,
 * drifting from 0.27 to 0.003 where all were 0.034, and made a maneuver certain where enumeration
 * gives 0.21 to 0.74. The expectation correction keeps the first exactly, since
 * \f$ b_{ij} = \delta_{ij} \f$ for \f$ \Pi = I \f$, and the test
 * \c ImmSmootherModeProbabilitiesTrackEnumeration holds it to the second; it is not the smoother
 * of \ref Kim1994 "Kim (1994)", \f$ \sum_i b_{ij} \mu^i_{t+1 \mid N} \f$, which is exact for
 * \f$ \Pi = I \f$ too but loses what the later measurements say about a maneuver through the
 * continuous state. It costs a prediction per pair of modes in predictJoint(). Three departures
 * in the continuous part. The thesis calls
 * \f$ \hat x^j_{t+1 \mid t}, P^j_{t+1 \mid t} \f$ the predicted mean and covariance of mode \a j;
 * its derivation (8.2) divides by \f$ p(x_{t+1} \mid r_t = j, Z^t) \f$, which is mode \a j's
 * filtered %Gaussian moved through its own model, and its gain (8.17),
 * \f$ P^j_{t \mid t} F_j^T (P^j_{t+1 \mid t})^{-1} \f$, is consistent only with that, so that is
 * what ImmFilter::predictJoint() records, rather than the IMM's prediction
 * from the mixed %Gaussian. The smoothed covariance (8.16) is printed with a minus sign; this is
 * the RTS update, with a plus, which is what the derivation gives. And the step is written with
 * the cross-covariance and in the tangent space, as smoothStep() is, so that it serves every
 * prediction method and manifold states. The weights are kept as logarithms; a mode no smoothed
 * mode at \a t + 1 can have come from keeps its filtered %Gaussian.
 *
 * As with GaussianHistory, a Lag keeps the newest steps and recycles the oldest step's storage
 * once full. A full history's push() copies the joint into that storage, so with one joint kept
 * and refilled by predictJoint(dt, joint), fixed-lag smoothing allocates nothing once the window is
 * full. Each step's joints and smoothed Gaussians come from the history's allocator, by
 * uses-allocator construction.
 *
 * \tparam Space_      the continuous state space
 * \tparam ModeSpace_  the discrete mode space
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the steps and what they hold (see \ref Allocation)
 *
 * \sa HybridPredictedJoint, ImmFilter::predictJoint(), GaussianHistory,
 * \ref Smoothers
 */
template <Space Space_, DiscreteSpace ModeSpace_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class HybridGaussianHistory
{
public:

    using Scalar = Scalar_;                                                              ///< the scalar type
    using SpaceType = Space_;                                                            ///< the continuous state space
    using ModeSpaceType = ModeSpace_;                                                    ///< the discrete mode space
    using JointType = HybridPredictedJoint<Space_, ModeSpace_, Scalar_, Allocator_>;     ///< what each step records
    using SmoothedType = HybridGaussianParams<Space_, ModeSpace_, Scalar_, Allocator_>;  ///< the smoothed state's type
    using allocator_type = Allocator_;  ///< the allocator the steps come from
    using Index = Eigen::Index;         ///< Eigen's index type

    /** One recorded prediction and its smoothed hybrid state, built from the history's allocator. */
    class Step
    {
    public:

        using allocator_type = Allocator_;  ///< the allocator its joints and Gaussians come from

        /** One mode's Rauch-Tung-Striebel gain. */
        using GainType = typename GaussianParams<Space_, Scalar_>::MatrixType;

        /** The allocator of the gains. */
        using GainAllocator = internal::ReBound<Allocator_, GainType>;

        /** A copy of \a recorded in storage from \a allocator, with the filtered state as its
         * smoothed one until smooth() runs.
         */
        explicit Step(JointType const & recorded, Allocator_ const & allocator = {})
            : joint{copyJoint(recorded, allocator)}
            , smoothed{recorded.modes, recorded.joints.front().filtered, allocator}
            , gains(GainAllocator(allocator))
        {
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
            : joint{
                  std::move(other.joint.modes),
                  typename JointType::JointsType(std::move(other.joint.joints), allocator),
                  std::move(other.joint.transitions),
                  typename JointType::PredictionsType(std::move(other.joint.throughOtherModels), allocator)
              }
            , smoothed{std::move(other.smoothed), allocator}
            , gains(std::move(other.gains), GainAllocator(allocator))
        {}

        Step(Step const &) = default;  ///< copies the joint and the smoothed state
        Step(Step &&) = default;       ///< moves the joint and the smoothed state

        /** Copies into this step's storage. */
        Step &
        operator=(Step const &) = default;

        /** Takes the other step's storage, as the members' do. */
        Step &
        operator=(Step &&) = default;

        ~Step() = default;

        /** Replaces the smoothed state with the filtered one: the joint's mode probabilities and
         * each mode's filtered %Gaussian.
         */
        void
        resetSmoothed()
        {
            smoothed.setModes(joint.modes);
            for (Index j = 0; j < smoothed.size(); ++j)
            {
                smoothed.setConditional(j, joint.joints[std::size_t(j)].filtered);
            }
        }

        /** Computes each mode's gain from its joint (internal::rtsGain()), once per recording. */
        void
        computeGains()
        {
            gains.resize(joint.joints.size());
            for (std::size_t j = 0; j < gains.size(); ++j)
            {
                gains[j] = internal::rtsGain(joint.joints[j]);
            }
        }

        JointType joint;  ///< the prediction as predictJoint() recorded it

        /** \f$ p(x_t, r_t \mid z_{1:N}) \f$ as of the last smooth(); the filtered state until then */
        SmoothedType smoothed;

        /** Each mode's Rauch-Tung-Striebel gain, made once when the joint is recorded. */
        internal::Buffer<GainType, Allocator_> gains;

    private:

        /** \returns \a recorded copied into storage from \a allocator */
        [[nodiscard]] static JointType
        copyJoint(JointType const & recorded, Allocator_ const & allocator)
        {
            return {
                recorded.modes,
                typename JointType::JointsType(recorded.joints, allocator),
                recorded.transitions,
                typename JointType::PredictionsType(recorded.throughOtherModels, allocator)
            };
        }
    };

private:

    using StepAllocator = internal::ReBound<Allocator_, Step>;  ///< allocates the list's nodes
    using ListType = std::list<Step, StepAllocator>;            ///< the steps, oldest first

public:

    using const_iterator = typename ListType::const_iterator;  ///< iterates the steps, oldest first

    /** An empty history that keeps every step. */
    HybridGaussianHistory() = default;

    /** An empty history that keeps every step, with storage from \a allocator. */
    explicit HybridGaussianHistory(Allocator_ const & allocator)
        : m_steps(StepAllocator(allocator))
    {}

    /** An empty history that keeps the newest \a lag steps, with storage from \a allocator. */
    explicit HybridGaussianHistory(Lag lag, Allocator_ const & allocator = {})
        : m_steps(StepAllocator(allocator))
        , m_lag{lag.steps}
    {
        EMRYS_FILTER_ASSERT(lag.steps > 0 && "a fixed-lag history keeps at least one step");
    }

    /** Copies the steps into nodes from \a other's allocator (see \ref Allocation). */
    HybridGaussianHistory(HybridGaussianHistory const & other)
        : HybridGaussianHistory{other, other.get_allocator()}
    {}

    /** Copies \a other's steps into storage from \a allocator. */
    HybridGaussianHistory(HybridGaussianHistory const & other, Allocator_ const & allocator)
        : m_steps(other.m_steps, StepAllocator(allocator))
        , m_lag{other.m_lag}
    {}

    /** Takes \a other's steps. */
    HybridGaussianHistory(HybridGaussianHistory && other) = default;

    /** Moves \a other's steps into storage from \a allocator, taking its storage when the
     * allocators are equal.
     */
    HybridGaussianHistory(HybridGaussianHistory && other, Allocator_ const & allocator)
        : m_steps(std::move(other.m_steps), StepAllocator(allocator))
        , m_lag{other.m_lag}
    {}

    /** Copies \a other's steps and lag. */
    HybridGaussianHistory &
    operator=(HybridGaussianHistory const & other) = default;

    /** Takes \a other's steps and lag, as \c std::list's move assignment does. */
    HybridGaussianHistory &
    operator=(HybridGaussianHistory && other) = default;

    ~HybridGaussianHistory() = default;

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

    /** Records a copy of \a joint as the newest step. A full fixed-lag history drops its oldest
     * step and copies \a joint into its storage.
     */
    void
    push(JointType const & joint)
    {
        EMRYS_FILTER_ASSERT(!joint.joints.empty() && "the joint has no modes");
        if (m_lag > 0 && size() == m_lag)
        {
            m_steps.splice(m_steps.end(), m_steps, m_steps.begin());
            Step & step = m_steps.back();
            step.joint.modes = joint.modes;
            step.joint.joints = joint.joints;
            step.joint.transitions = joint.transitions;
            step.joint.throughOtherModels = joint.throughOtherModels;
            step.resetSmoothed();
            step.computeGains();
        }
        else
        {
            m_steps.emplace_back(joint);
        }
    }

    /** Fills every step's smoothed hybrid state by the backward IMM step above, from the newest
     * back, given \a latest, the filter's state after the newest step: its posterior
     * \f$ p(x_N, r_N \mid z_{1:N}) \f$, which is its own smoothed state.
     */
    template <typename LatestAllocator>
    void
    smooth(HybridGaussianParams<Space_, ModeSpace_, Scalar_, LatestAllocator> const & latest)
    {
        auto step = m_steps.rbegin();
        if (step == m_steps.rend())
        {
            return;
        }
        internal::smoothHybridStep(step->joint, step->gains, latest, step->smoothed);
        for (auto next = step++; step != m_steps.rend(); next = step++)
        {
            internal::smoothHybridStep(step->joint, step->gains, next->smoothed, step->smoothed);
        }
    }

    /** Drops every step, keeping the lag. */
    void
    clear()
    {
        m_steps.clear();
    }

private:

    ListType m_steps;  ///< the steps, oldest first
    Index m_lag = 0;   ///< the most steps kept, or zero to keep every step
};

}  // namespace Emrys
