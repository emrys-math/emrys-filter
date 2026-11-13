#pragma once

/** \file
 * Smoothing a HiddenMarkovFilter [emissions]: CategoricalHistory, the record of its forward pass,
 * whose smooth() runs the backward pass over it, fixed-interval or fixed-lag; its smoothStep() and
 * transitionPosterior(); and mostProbableSequence(), the Viterbi path over the same record, with
 * the ValueSequence it returns (decisions.md [smoothing]). The filter is unchanged but for
 * predictJoint() (see CategoricalPredictedJoint in HiddenMarkovFilter.h). Read
 * HiddenMarkovFilter.h first; the design review is docs/guide/smoothing.md.
 */

#include <cstddef>
#include <iterator>
#include <list>
#include <optional>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Filters/HiddenMarkovFilter.h>
#include <EmrysFilter/Smoothers/Lag.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** The largest of \f$ \log s_j - \log \hat p_j \f$ over the values \f$ j \f$ that \a smoothedNext
 * gives any probability, the scale smoothingRatios() divides by.
 */
template <typename Space, typename Scalar>
[[nodiscard]] Scalar
smoothingRatioScale(
    CategoricalPredictedJoint<Space, Scalar> const & joint,
    CategoricalParams<Space, Scalar> const & smoothedNext
)
{
    EMRYS_FILTER_ASSERT(
        smoothedNext.logTotalWeight() && "the smoothed distribution at the step's end must have valid weights"
    );
    Scalar const minusInfinity = -Eigen::NumTraits<Scalar>::infinity();
    auto const logNext = logProbabilitiesOf(smoothedNext);
    auto const logPredicted = logProbabilitiesOf(joint.predicted);
    Scalar scale = minusInfinity;
    for (Eigen::Index j = 0; j < logNext.size(); ++j)
    {
        if (logNext(j) != minusInfinity)
        {
            EMRYS_FILTER_ASSERT(
                logPredicted(j) != minusInfinity && "a value the prediction rules out has smoothed probability"
            );
            scale = Eigen::numext::maxi(scale, Scalar(logNext(j) - logPredicted(j)));
        }
    }
    return scale;
}

/** \returns \f$ s_j / \hat p_j \f$ for each value \f$ j \f$ at the end of \a joint's step, divided by
 * \f$ e \f$ to the smoothingRatioScale(), with \f$ s \f$ the \a smoothedNext distribution: zero
 * where \f$ s_j \f$ is, and at most one.
 */
template <typename Space, typename Scalar>
[[nodiscard]] typename CategoricalParams<Space, Scalar>::ArrayType
smoothingRatios(
    CategoricalPredictedJoint<Space, Scalar> const & joint,
    CategoricalParams<Space, Scalar> const & smoothedNext
)
{
    EMRYS_FILTER_ASSERT(joint.filtered.space() == smoothedNext.space() && "the distributions are on different spaces");
    Scalar const minusInfinity = -Eigen::NumTraits<Scalar>::infinity();
    Scalar const scale = smoothingRatioScale(joint, smoothedNext);
    auto const logNext = logProbabilitiesOf(smoothedNext);
    auto const logPredicted = logProbabilitiesOf(joint.predicted);
    auto ratios = typename CategoricalParams<Space, Scalar>::ArrayType{};
    ratios.resize(logNext.size());
    for (Eigen::Index j = 0; j < logNext.size(); ++j)
    {
        ratios(j) =
            logNext(j) == minusInfinity ? Scalar(0) : Eigen::numext::exp(Scalar(logNext(j) - logPredicted(j) - scale));
    }
    return ratios;
}

}  // namespace internal

/** \ingroup Smoothers
 * \returns the smoothed distribution at the start of \a joint's step, given \a smoothedNext, the
 * smoothed distribution at its end: with \f$ f \f$ the filtered distribution, \f$ \hat p \f$ the
 * prediction, \f$ s \f$ the smoothed one at the end, and \f$ K \f$ the step's transitions,
 *
 * \f[ s'_i = f_i \sum_j K_{ij} \, \frac{s_j}{\hat p_j}. \f]
 *
 * The Bayesian smoothing equation of \ref Sarkka2013 "Särkkä (2013)", chapter 8, on a finite set,
 * where its integral is a sum. On a hidden Markov model it gives the smoothed marginals
 * \f$ \gamma \f$ of the forward-backward algorithm (\ref Rabiner1989 "Rabiner (1989)", §III-A)
 * in the form that needs only the filtered distributions and no likelihoods, as smoothStep() does
 * for a %Gaussian, rather than Rabiner's product of forward and backward variables. The two are
 * the same, since \f$ s_j / \hat p_j \f$ is the backward variable, normalized. The ratio is taken
 * in logs and scaled by its largest entry, so a value improbable beyond a \c double's range at
 * either end still divides correctly. \a smoothedNext must have valid weights, and give no
 * probability to a value the prediction rules out.
 *
 * \sa CategoricalHistory, transitionPosterior()
 */
template <typename Space, typename Scalar>
[[nodiscard]] CategoricalParams<Space, Scalar>
smoothStep(
    CategoricalPredictedJoint<Space, Scalar> const & joint,
    CategoricalParams<Space, Scalar> const & smoothedNext
)
{
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;
    ArrayType const ratios = internal::smoothingRatios(joint, smoothedNext);
    ArrayType const logSmoothed = internal::logProbabilitiesOf(joint.filtered)
                                + internal::multiply(joint.transitions, ratios.matrix()).array().log();
    auto smoothed = CategoricalParams<Space, Scalar>{joint.filtered.space(), logSmoothed};
    // A value with probability at the end was predicted from one with probability at the start.
    [[maybe_unused]] auto const normalized = smoothed.normalize();
    EMRYS_FILTER_ASSERT(normalized && "no value smoothed at the step's end came from a value at its start");
    return smoothed;
}

/** \ingroup Smoothers
 * \returns the posterior probability of each transition of \a joint's step given \a smoothedNext,
 * the smoothed distribution at its end: \f$ \xi_{ij} = P(x_k = i, x_{k+1} = j \mid z_{1:N})
 * = f_i K_{ij} s_j / \hat p_j \f$, a matrix whose entries sum to one, whose row sums are
 * smoothStep() and whose column sums are \a smoothedNext
 *
 * The pairwise posteriors \f$ \xi \f$ of \ref Rabiner1989 "Rabiner (1989)", §III-C, there in
 * terms of the forward and backward variables, here of the filtered and smoothed distributions as
 * smoothStep() takes them. Summed over a history they are the expected transition counts that the
 * Baum-Welch re-estimation of a chain consumes; the re-estimation itself is model identification,
 * outside the library (decisions.md [analysis-project]). \a smoothedNext must meet smoothStep()'s
 * conditions.
 */
template <typename Space, typename Scalar>
[[nodiscard]] typename CategoricalPredictedJoint<Space, Scalar>::MatrixType
transitionPosterior(
    CategoricalPredictedJoint<Space, Scalar> const & joint,
    CategoricalParams<Space, Scalar> const & smoothedNext
)
{
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;
    ArrayType const ratios = internal::smoothingRatios(joint, smoothedNext);
    ArrayType const filtered = internal::logProbabilitiesOf(joint.filtered).exp();
    typename CategoricalPredictedJoint<Space, Scalar>::MatrixType posterior =
        filtered.matrix().asDiagonal() * joint.transitions * ratios.matrix().asDiagonal();
    posterior /= posterior.sum();
    return posterior;
}

/** \ingroup Smoothers
 * \brief The record of a HiddenMarkovFilter's forward pass that a smoother runs backward over: one
 * CategoricalPredictedJoint per prediction, each with room for its smoothed distribution.
 *
 * GaussianHistory for a discrete state (decisions.md [smoothing], [emissions]), with the same interface: push()
 * records the joint a filter's predictJoint() returns, smooth() takes the filter's current state
 * and fills every step's smoothed distribution by smoothStep(), from the newest step back, and a
 * Lag makes it a fixed-lag window that recycles its oldest node. Updates between predictions need
 * no recording. The same record gives the most probable sequence (mostProbableSequence()) and the
 * pairwise transition posteriors (transitionPosterior()).
 *
 * \code
 * CategoricalHistory<Weather> history;
 * for (auto const & z : observations)
 * {
 *     history.push(filter.predictJoint());
 *     static_cast<void>(filter.update(table.observe(z)));
 * }
 * history.smooth(filter.state());
 * for (auto const & step : history) { use(step.smoothed); }       // p(x_k | z_{1:N}), oldest first
 * auto const path = mostProbableSequence(history, filter.state());  // Viterbi
 * \endcode
 *
 * For a bounded space every step is inline in its node, so a fixed-lag history allocates nothing
 * once full.
 *
 * \tparam Space_      the discrete space
 * \tparam Scalar_     scalar type of the probabilities, \c double by default
 * \tparam Allocator_  allocator for the steps, re-bound to the list's nodes (see \ref Allocation)
 *
 * \sa smoothStep(), CategoricalPredictedJoint, HiddenMarkovFilter::predictJoint(), \ref Smoothers
 */
template <DiscreteSpace Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class CategoricalHistory
{
public:

    using Scalar = Scalar_;                                        ///< the scalar type
    using SpaceType = Space_;                                      ///< the discrete space
    using CategoricalType = CategoricalParams<Space_, Scalar_>;    ///< the smoothed distribution's type
    using JointType = CategoricalPredictedJoint<Space_, Scalar_>;  ///< what each step records
    using allocator_type = Allocator_;                             ///< the allocator the steps come from
    using Index = Eigen::Index;                                    ///< Eigen's index type

    /** One recorded prediction and its smoothed distribution. */
    struct Step
    {
        JointType joint;  ///< the prediction as predictJoint() returned it

        /** \f$ p(x_k \mid z_{1:N}) \f$ for the step's \f$ x_k \f$ (joint.filtered), as of the last
         * smooth(); a copy of joint.filtered until then
         */
        CategoricalType smoothed;
    };

private:

    using StepAllocator = internal::ReBound<Allocator_, Step>;  ///< allocates the list's nodes
    using ListType = std::list<Step, StepAllocator>;            ///< the steps, oldest first

public:

    using const_iterator = typename ListType::const_iterator;  ///< iterates the steps, oldest first

    /** An empty history that keeps every step. */
    CategoricalHistory() = default;

    /** An empty history that keeps every step, with nodes from \a allocator. */
    explicit CategoricalHistory(Allocator_ const & allocator)
        : m_steps(StepAllocator(allocator))
    {}

    /** An empty history that keeps the newest \a lag steps, with nodes from \a allocator. */
    explicit CategoricalHistory(Lag lag, Allocator_ const & allocator = {})
        : m_steps(StepAllocator(allocator))
        , m_lag{lag.steps}
    {
        EMRYS_FILTER_ASSERT(lag.steps > 0 && "a fixed-lag history keeps at least one step");
    }

    /** Copies the steps into nodes from \a other's allocator (see \ref Allocation). */
    CategoricalHistory(CategoricalHistory const & other)
        : CategoricalHistory{other, other.get_allocator()}
    {}

    /** Copies \a other's steps into nodes from \a allocator. */
    CategoricalHistory(CategoricalHistory const & other, Allocator_ const & allocator)
        : m_steps(other.m_steps, StepAllocator(allocator))
        , m_lag{other.m_lag}
    {}

    /** Takes \a other's steps. */
    CategoricalHistory(CategoricalHistory && other) = default;

    /** Moves \a other's steps into nodes from \a allocator, taking its nodes when the allocators
     * are equal.
     */
    CategoricalHistory(CategoricalHistory && other, Allocator_ const & allocator)
        : m_steps(std::move(other.m_steps), StepAllocator(allocator))
        , m_lag{other.m_lag}
    {}

    /** Copies \a other's steps and lag. */
    CategoricalHistory &
    operator=(CategoricalHistory const & other) = default;

    /** Takes \a other's steps and lag, as \c std::list's move assignment does. */
    CategoricalHistory &
    operator=(CategoricalHistory && other) = default;

    ~CategoricalHistory() = default;

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

    /** Records \a joint as the newest step. A full fixed-lag history drops its oldest step and
     * reuses its node.
     */
    void
    push(JointType joint)
    {
        if (m_lag > 0 && size() == m_lag)
        {
            m_steps.splice(m_steps.end(), m_steps, m_steps.begin());
            Step & step = m_steps.back();
            step.joint = std::move(joint);
            step.smoothed = step.joint.filtered;
        }
        else
        {
            CategoricalType smoothed = joint.filtered;
            m_steps.push_back(Step{std::move(joint), std::move(smoothed)});
        }
    }

    /** Fills every step's smoothed distribution by smoothStep(), from the newest back, given
     * \a latest, the filter's state after the newest step: its posterior \f$ p(x_N \mid z_{1:N}) \f$,
     * which is its own smoothed distribution.
     */
    void
    smooth(CategoricalType const & latest)
    {
        CategoricalType const * next{&latest};
        for (auto step = m_steps.rbegin(); step != m_steps.rend(); ++step)
        {
            step->smoothed = smoothStep(step->joint, *next);
            next = &step->smoothed;
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

/** \ingroup Smoothers
 * \brief The most probable sequence of values over a CategoricalHistory, and its posterior
 * probability, from mostProbableSequence().
 *
 * \tparam Space      the discrete space
 * \tparam Scalar     scalar type of the probabilities
 * \tparam Allocator  allocator for the values (see \ref Allocation)
 */
template <DiscreteSpace Space, RealScalar Scalar, typename Allocator = DefaultAllocator>
struct ValueSequence
{
    /** The flat index of each value, oldest first: one per step of the history, then the latest. */
    internal::Buffer<Eigen::Index, Allocator> values;

    /** \f$ \log P(x_{0:N} = \mathrm{values} \mid z_{1:N}) \f$, the sequence's posterior log-probability */
    Scalar logProbability;
};

/** \ingroup Smoothers
 * \returns the most probable sequence of values over \a history and \a latest, the filter's state
 * after its newest step, with its posterior log-probability; its values come from the history's
 * allocator
 *
 * The [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) of
 * \ref Viterbi1967 "Viterbi (1967)", in the max-product form of \ref Rabiner1989 "Rabiner (1989)",
 * §III-B: a forward pass of
 * \f$ \delta_{k+1}(j) = \max_i [\delta_k(i) + \log K_{ij}] + \log L_{k+1}(j) \f$ with back-pointers,
 * then a backtrack from the best final value. The departure is where the likelihoods come from:
 * the history keeps no measurements, so \f$ L_{k+1}(j) \f$ is the ratio of the filtered to the
 * predicted probability, \f$ f_{k+1}(j) / \hat p_{k+1}(j) \f$, which is the product of every
 * update's likelihood between the two predictions divided by their evidence. The evidence is the
 * same for every sequence, so the argmax is unchanged, and since \f$ \delta \f$ starts from the
 * oldest step's filtered distribution, the maximum is the posterior log-probability of the
 * sequence rather than a joint density. Over a fixed-lag window the sequence is the most probable
 * one of the window's values, the earlier ones summed out. It is a free function and not a filter
 * member because it is the one place the library takes a maximum instead of a sum.
 *
 * Ties go to the lowest value. It allocates the back-pointers, one index per value per step, and
 * the result.
 */
template <typename Space, typename Scalar, typename Allocator>
[[nodiscard]] ValueSequence<Space, Scalar, Allocator>
mostProbableSequence(
    CategoricalHistory<Space, Scalar, Allocator> const & history,
    CategoricalParams<Space, Scalar> const & latest
)
{
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;
    Scalar const minusInfinity = -Eigen::NumTraits<Scalar>::infinity();
    Eigen::Index const count = latest.cardinality();
    std::size_t const steps = std::size_t(history.size());
    auto backPointers = internal::bufferOf<Eigen::Index>(history.get_allocator());
    backPointers.resize(steps * std::size_t(count));

    ArrayType delta = internal::logProbabilitiesOf(history.empty() ? latest : history.front().joint.filtered);
    auto next = ArrayType{};
    next.resize(count);
    std::size_t k = 0;
    for (auto step = history.begin(); step != history.end(); ++step, ++k)
    {
        auto const following = std::next(step);
        auto const filtered =
            internal::logProbabilitiesOf(following == history.end() ? latest : following->joint.filtered);
        auto const predicted = internal::logProbabilitiesOf(step->joint.predicted);
        for (Eigen::Index j = 0; j < count; ++j)
        {
            Scalar best = minusInfinity;
            Eigen::Index from = 0;
            for (Eigen::Index i = 0; i < count; ++i)
            {
                Scalar const candidate = delta(i) + Eigen::numext::log(step->joint.transitions(i, j));
                if (candidate > best)
                {
                    best = candidate;
                    from = i;
                }
            }
            backPointers[k * std::size_t(count) + std::size_t(j)] = from;
            bool const reachable = best != minusInfinity && predicted(j) != minusInfinity;
            next(j) = reachable ? Scalar(best + filtered(j) - predicted(j)) : minusInfinity;
        }
        delta.swap(next);
    }

    auto result = ValueSequence<Space, Scalar, Allocator>{
        internal::bufferOf<Eigen::Index>(history.get_allocator()),
        minusInfinity
    };
    result.values.resize(steps + 1);
    Eigen::Index value = 0;
    result.logProbability = delta.maxCoeff(&value);
    result.values[steps] = value;
    for (std::size_t s = steps; s > 0; --s)
    {
        value = backPointers[(s - 1) * std::size_t(count) + std::size_t(value)];
        result.values[s - 1] = value;
    }
    return result;
}

}  // namespace Emrys
