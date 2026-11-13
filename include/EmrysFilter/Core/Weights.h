#pragma once

/** \file
 * Log-weights: the WeightError a normalization can fail with (NumericError.h), and the streaming log-sum-exp
 * that totals them without overflow, sequentially or in chunks under an execution policy: LogSumExp and
 * logSumExp(), which are public, so that a user's code normalizes weights or sums likelihoods as the
 * library does. Every distribution with weighted elements (GaussianMixtureParams, ParticleSet,
 * Categorical) normalizes through internal::logTotalWeight(), and a single draw of one of them takes
 * internal::drawIndex(). The states whose elements can be pruned (ParticleSet, ParticleGaussianParams,
 * MixtureParams, GridGaussianParams) choose what pruneBelow(), pruneTail(), and pruneToSize() remove
 * through the same rules, internal::PruneBelowRule, internal::PruneTailRule, and
 * internal::PruneToSizeRule, so ties are broken alike. Read Resampling.h next.
 */

#include <algorithm>
#include <concepts>
#include <expected>
#include <limits>
#include <random>
#include <ranges>
#include <span>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/NumericError.h>

namespace Emrys
{

/** \ingroup Core
 * Streaming log-sum-exp: accumulates \f$ \log \sum_i e^{x_i} \f$ without overflow.
 * Inputs of -inf contribute nothing; an empty sum is -inf.
 *
 * The running maximum \f$ m \f$ and sum \f$ s = \sum_i e^{x_i - m} \f$ are rescaled when a
 * larger input arrives, the one-pass form of the
 * [log-sum-exp trick](https://en.wikipedia.org/wiki/LogSumExp).
 *
 * Adding one input is not associative, but adding a whole sum is: the pairs combine as
 * \f$ (m_1, s_1) \oplus (m_2, s_2) = (m, \; s_1 e^{m_1 - m} + s_2 e^{m_2 - m}) \f$ with
 * \f$ m = \max(m_1, m_2) \f$, whose identity is the empty sum, \f$ (-\infty, 0) \f$. So a
 * reduction streams the inputs of each chunk and combines the chunks' sums (logSumExp()).
 */
template <typename Scalar>
class LogSumExp
{
public:

    /** Adds \a x to the sum, rescaling the running sum if \a x is the new maximum. */
    void
    add(Scalar x)
    {
        if (x == -Eigen::NumTraits<Scalar>::infinity())
        {
            return;
        }
        if (x <= m_max)
        {
            m_sum += Eigen::numext::exp(x - m_max);
        }
        else
        {
            m_sum = m_sum * Eigen::numext::exp(m_max - x) + Scalar(1);
            m_max = x;
        }
    }

    /** Adds every element of \a xs, as if they had been added one by one, up to rounding. For
     * \c float and \c double the array's sum is taken whole, as its maximum and then the sum of
     * its exponentials relative to that, two whole-array expressions that Eigen vectorizes, and
     * joined to this sum as add(LogSumExp) joins two; other scalars add the elements one by one. A
     * NaN gives NaN either way.
     */
    template <typename Derived>
    void
    add(Eigen::ArrayBase<Derived> const & xs)
    {
        if constexpr (std::floating_point<Scalar>)
        {
            if (xs.size() == 0)
            {
                return;
            }
            Scalar const max = xs.template maxCoeff<Eigen::PropagateNaN>();
            if (max == -Eigen::NumTraits<Scalar>::infinity())
            {
                return;
            }
            auto range = LogSumExp{};
            range.m_max = max;
            // Each +inf input is the new maximum, so each counts one relative to it.
            range.m_sum = max == Eigen::NumTraits<Scalar>::infinity() ? Scalar((xs == max).count())
                                                                      : Scalar((xs - max).exp().sum());
            add(range);
        }
        else
        {
            for (Eigen::Index i = 0; i < xs.size(); ++i)
            {
                add(Scalar(xs(i)));
            }
        }
    }

    /** Adds everything added to \a other, as if its inputs had been added one by one, up to
     * rounding. Associative and commutative, with the empty sum as identity.
     */
    void
    add(LogSumExp const & other)
    {
        if (other.m_sum == Scalar(0))
        {
            return;
        }
        if (m_sum == Scalar(0))
        {
            *this = other;
            return;
        }
        if (other.m_max <= m_max)
        {
            m_sum += other.m_sum * Eigen::numext::exp(other.m_max - m_max);
        }
        else
        {
            m_sum = m_sum * Eigen::numext::exp(m_max - other.m_max) + other.m_sum;
            m_max = other.m_max;
        }
    }

    /** \returns \f$ \log \sum_i e^{x_i} \f$ over everything added so far, or -inf for nothing. */
    [[nodiscard]] Scalar
    value() const
    {
        return m_sum == Scalar(0) ? -Eigen::NumTraits<Scalar>::infinity() : m_max + Eigen::numext::log(m_sum);
    }

private:

    Scalar m_max = -Eigen::NumTraits<Scalar>::infinity();  ///< the largest input so far, \f$ m \f$
    Scalar m_sum = 0;                                      ///< \f$ \sum_i e^{x_i - m} \f$
};

/** \ingroup Core
 * \returns \f$ \log \sum_i e^{x_i} \f$ over the \a count inputs \a logAt(i), each converted to
 * \a Scalar, -inf for none, under \a policy: a reduceRanges() that streams each range into a
 * LogSumExp and adds the ranges' sums. Under Sequential it adds the inputs in order, one at a time.
 */
template <typename Scalar, ExecutionPolicy P, typename LogAt>
[[nodiscard]] Scalar
logSumExp(P && policy, Eigen::Index count, LogAt const & logAt)
{
    return internal::reduceRanges(
               std::forward<P>(policy),
               count,
               [&](Eigen::Index begin, Eigen::Index length) {
                   auto sum = LogSumExp<Scalar>{};
                   for (Eigen::Index i = begin; i < begin + length; ++i)
                   {
                       sum.add(Scalar(logAt(i)));
                   }
                   return sum;
               },
               [](LogSumExp<Scalar> a, LogSumExp<Scalar> const & b) {
                   a.add(b);
                   return a;
               }
    ).value();
}

/** \ingroup Core
 * logSumExp(policy, count, logAt) under execution::sequential.
 */
template <typename Scalar, typename LogAt>
[[nodiscard]] Scalar
logSumExp(Eigen::Index count, LogAt const & logAt)
{
    return logSumExp<Scalar>(execution::sequential, count, logAt);
}

/** \ingroup Core
 * \returns \f$ \log \sum_i e^{x_i} \f$ over the elements of \a logs, in their scalar, -inf for
 * none, under \a policy: a reduceRanges() that adds each range whole (LogSumExp::add() of an array)
 * and adds the ranges' sums. Under Sequential the whole array is one range. Normalized weights are
 * then \c (logs - logSumExp(logs)).exp().
 */
template <ExecutionPolicy P, typename Derived>
[[nodiscard]] typename Derived::Scalar
logSumExp(P && policy, Eigen::ArrayBase<Derived> const & logs)
{
    using Scalar = typename Derived::Scalar;
    return internal::reduceRanges(
               std::forward<P>(policy),
               logs.size(),
               [&](Eigen::Index begin, Eigen::Index length) {
                   auto sum = LogSumExp<Scalar>{};
                   sum.add(logs.derived().segment(begin, length));
                   return sum;
               },
               [](LogSumExp<Scalar> a, LogSumExp<Scalar> const & b) {
                   a.add(b);
                   return a;
               }
    ).value();
}

/** \ingroup Core
 * logSumExp(policy, logs) under execution::sequential.
 */
template <typename Derived>
[[nodiscard]] typename Derived::Scalar
logSumExp(Eigen::ArrayBase<Derived> const & logs)
{
    return logSumExp(execution::sequential, logs);
}

namespace internal
{

template <typename Scalar, ExecutionPolicy P>
[[nodiscard]] std::expected<Scalar, WeightError>
logTotalWeight(P && policy, std::span<Scalar const> logWeights);

/** \returns the log of the sum of the weights whose logs are \a logWeights, or a
 *          WeightError if there are none, they sum to zero, or one is NaN or +inf. Weights stored
 *          contiguously are summed whole, as logTotalWeight(execution::sequential, logWeights) sums them.
 */
template <typename Scalar, std::ranges::input_range R>
[[nodiscard]] std::expected<Scalar, WeightError>
logTotalWeight(R && logWeights)
{
    if constexpr (std::ranges::contiguous_range<R>
                  && std::same_as<std::remove_cv_t<std::ranges::range_value_t<R>>, Scalar>)
    {
        return logTotalWeight<Scalar>(execution::sequential, std::span<Scalar const>{logWeights});
    }
    else
    {
        auto total = LogSumExp<Scalar>{};
        bool empty = true;
        for (Scalar const w : logWeights)
        {
            if (isNaN(w) || w == Eigen::NumTraits<Scalar>::infinity())
            {
                return std::unexpected{WeightError::NonFinite};
            }
            total.add(w);
            empty = false;
        }
        if (empty)
        {
            return std::unexpected{WeightError::Empty};
        }
        Scalar const result = total.value();
        if (result == -Eigen::NumTraits<Scalar>::infinity())
        {
            return std::unexpected{WeightError::ZeroSum};
        }
        return result;
    }
}

/** logTotalWeight() of \a logWeights under \a policy, reduced as logSumExp() of an array is, each
 * range checked whole for NaN and +inf. Under Sequential it is logTotalWeight(logWeights) exactly.
 */
template <typename Scalar, ExecutionPolicy P>
[[nodiscard]] std::expected<Scalar, WeightError>
logTotalWeight(P && policy, std::span<Scalar const> logWeights)
{
    // A range's sum, and whether a weight in it is NaN or +inf.
    struct Total
    {
        LogSumExp<Scalar> sum;   ///< the sum of the range's finite weights
        bool nonFinite = false;  ///< whether a weight in the range is NaN or +inf
    };

    if (logWeights.empty())
    {
        return std::unexpected{WeightError::Empty};
    }
    Total const total = reduceRanges(
        std::forward<P>(policy),
        Eigen::Index(logWeights.size()),
        [&](Eigen::Index begin, Eigen::Index length) {
            auto range = Total{};
            auto const weights = logWeights.subspan(std::size_t(begin), std::size_t(length));
            if constexpr (std::floating_point<Scalar>)
            {
                auto const map = Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1> const>{weights.data(), length};
                // NaN and +inf are the weights that are not below +inf.
                range.nonFinite = !(map < Eigen::NumTraits<Scalar>::infinity()).all();
                if (!range.nonFinite)
                {
                    range.sum.add(map);
                }
            }
            else
            {
                for (Scalar const w : weights)
                {
                    if (isNaN(w) || w == Eigen::NumTraits<Scalar>::infinity())
                    {
                        range.nonFinite = true;
                        break;
                    }
                    range.sum.add(w);
                }
            }
            return range;
        },
        [](Total a, Total const & b) {
            a.sum.add(b.sum);
            a.nonFinite = a.nonFinite || b.nonFinite;
            return a;
        }
    );
    if (total.nonFinite)
    {
        return std::unexpected{WeightError::NonFinite};
    }
    Scalar const result = total.sum.value();
    if (result == -Eigen::NumTraits<Scalar>::infinity())
    {
        return std::unexpected{WeightError::ZeroSum};
    }
    return result;
}

/** Which elements a prune removes, decided once their weights are known: every element lighter
 * than \c cutoff, and of those exactly as heavy, all but the first \c keepTies by position.
 * removes() is asked about each element in turn, in order of position, so that the ties kept are
 * the first ones; ties go to the lowest position, as mostProbable() breaks them.
 */
template <typename Scalar>
class PruneCut
{
public:

    /** The cut that removes nothing. */
    PruneCut() = default;

    /** The cut that removes the elements lighter than \a cutoff, and the elements as heavy as it
     * after the first \a keepTies.
     */
    PruneCut(Scalar cutoff, Eigen::Index keepTies)
        : m_cutoff{cutoff}
        , m_keepTies{keepTies}
    {}

    /** \returns whether the next element in order of position, of log-weight \a logWeight, is removed */
    [[nodiscard]] bool
    removes(Scalar logWeight)
    {
        if (logWeight < m_cutoff)
        {
            return true;
        }
        if (logWeight == m_cutoff)
        {
            return m_tiesSeen++ >= m_keepTies;
        }
        return false;
    }

private:

    Scalar m_cutoff = -Eigen::NumTraits<Scalar>::infinity();             ///< lighter elements are removed
    Eigen::Index m_keepTies = std::numeric_limits<Eigen::Index>::max();  ///< the ties kept, the first ones
    Eigen::Index m_tiesSeen = 0;                                         ///< the ties asked about so far
};

/** \returns the cut that removes the \a count lightest of the elements whose log-weights are
 * \a sorted, in increasing order; of the elements as heavy as the last one removed, it keeps those
 * of lowest position
 */
template <typename Scalar>
[[nodiscard]] PruneCut<Scalar>
lightestCut(std::span<Scalar const> sorted, Eigen::Index count)
{
    if (count == 0)
    {
        return {};
    }
    Scalar const cutoff = sorted[std::size_t(count - 1)];
    auto const [first, last] = std::equal_range(sorted.begin(), sorted.end(), cutoff);
    auto const lighter = Eigen::Index(first - sorted.begin());
    auto const ties = Eigen::Index(last - first);
    return {cutoff, ties - (count - lighter)};
}

/** The rule of pruneBelow(): every element whose normalized log-weight is below \c logThreshold
 * is removed, but the most probable is kept, so that the weights still define a distribution.
 */
template <typename Scalar>
struct PruneBelowRule
{
    Scalar logThreshold;  ///< elements whose normalized log-weight is below it are removed

    /** \returns the cut for the \a count elements of log-weights \a logAt(i) and log total \a logTotal */
    template <typename LogAt, typename Scratch>
    [[nodiscard]] PruneCut<Scalar>
    operator()(Eigen::Index count, LogAt const & logAt, Scalar logTotal, Scratch & /*scratch*/) const
    {
        Scalar const cut = logThreshold + logTotal;
        Scalar heaviest = -Eigen::NumTraits<Scalar>::infinity();
        for (Eigen::Index i = 0; i < count; ++i)
        {
            heaviest = std::max(heaviest, Scalar(logAt(i)));
        }
        if (heaviest < cut)
        {
            // Every element is below the threshold: the first of the heaviest stays.
            return {heaviest, 1};
        }
        return {cut, count};
    }
};

/** The rule of pruneTail(): the lightest elements are removed, as many as can be while their
 * combined normalized weight stays below \f$ e^{\mathrm{logFraction}} \f$, which is below one, so
 * that some element stays.
 */
template <typename Scalar>
struct PruneTailRule
{
    Scalar logFraction;  ///< the log of the fraction of the total weight the removed stay below

    /** \returns the cut for the \a count elements of log-weights \a logAt(i) and log total \a logTotal,
     * sorting a copy of the weights in \a scratch
     */
    template <typename LogAt, typename Scratch>
    [[nodiscard]] PruneCut<Scalar>
    operator()(Eigen::Index count, LogAt const & logAt, Scalar logTotal, Scratch & scratch) const
    {
        EMRYS_FILTER_ASSERT(logFraction < Scalar(0) && "the fraction pruned is below one");
        scratch.resize(std::size_t(count));
        for (Eigen::Index i = 0; i < count; ++i)
        {
            scratch[std::size_t(i)] = Scalar(logAt(i));
        }
        std::sort(scratch.begin(), scratch.end());
        Scalar const budget = logFraction + logTotal;
        auto removed = LogSumExp<Scalar>{};
        Eigen::Index taken = 0;
        while (taken < count - 1)
        {
            auto next = removed;
            next.add(scratch[std::size_t(taken)]);
            if (!(next.value() < budget))
            {
                break;
            }
            removed = next;
            ++taken;
        }
        return lightestCut(std::span<Scalar const>{scratch.data(), scratch.size()}, taken);
    }
};

/** The rule of pruneToSize(): the lightest elements are removed until at most \c maxSize, at
 * least one, are left.
 */
struct PruneToSizeRule
{
    Eigen::Index maxSize;  ///< the most elements left

    /** \returns the cut for the \a count elements of log-weights \a logAt(i), finding the lightest
     * in a copy of the weights in \a scratch
     */
    template <typename Scalar, typename LogAt, typename Scratch>
    [[nodiscard]] PruneCut<Scalar>
    operator()(Eigen::Index count, LogAt const & logAt, Scalar /*logTotal*/, Scratch & scratch) const
    {
        EMRYS_FILTER_ASSERT(maxSize >= 1 && "a prune leaves at least one element");
        if (count <= maxSize)
        {
            return {};
        }
        scratch.resize(std::size_t(count));
        for (Eigen::Index i = 0; i < count; ++i)
        {
            scratch[std::size_t(i)] = Scalar(logAt(i));
        }
        // The lightest elements to remove come first; the ties among them are found in a sort of
        // those up to the one at the cut.
        Eigen::Index const removed = count - maxSize;
        auto const at = scratch.begin() + std::ptrdiff_t(removed - 1);
        std::nth_element(scratch.begin(), at, scratch.end());
        Scalar const cutoff = *at;
        auto const lighter = Eigen::Index(std::count_if(scratch.begin(), scratch.end(), [&](Scalar w) {
            return w < cutoff;
        }));
        auto const ties = Eigen::Index(std::count(scratch.begin(), scratch.end(), cutoff));
        return {cutoff, ties - (removed - lighter)};
    }
};

/** \returns an index from zero to \a count, exclusive, drawn with probability \a weight(i), the
 * weights normalized: one uniform draw from \a rng against their running sum, which allocates
 * nothing, as a single draw from a distribution with weighted elements needs. Rounding that leaves
 * the sum short of the draw gives the last index of positive weight.
 */
template <typename Scalar, typename Weight, std::uniform_random_bit_generator G>
[[nodiscard]] Eigen::Index
drawIndex(Eigen::Index count, Weight const & weight, G & rng)
{
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    EMRYS_FILTER_ASSERT(count > 0 && "draw from at least one element");
    Scalar const u = std::uniform_real_distribution<Scalar>{}(rng);
    Scalar sum = 0;
    Eigen::Index last = 0;
    for (Eigen::Index i = 0; i < count; ++i)
    {
        Scalar const w = weight(i);
        if (w > 0)
        {
            sum += w;
            last = i;
            if (u < sum)
            {
                return i;
            }
        }
    }
    return last;
}

}  // namespace internal

}  // namespace Emrys
