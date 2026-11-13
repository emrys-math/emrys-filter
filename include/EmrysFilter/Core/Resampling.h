#pragma once

/** \file
 * The resamplers: function objects that draw ancestors in proportion to weights, which
 * ParticleSet::resample() and the Resample upkeep action call. Multinomial, stratified,
 * systematic, and residual resampling, sequentially or under an execution policy, with the
 * Resampler and ResamplerUnder concepts. Read ChiSquared.h next.
 */

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>

namespace Emrys
{

/** \defgroup Resamplers Resamplers
 * \ingroup Core
 *
 * A resampler is a function object called as \c r(weights, ancestors, rng). It
 * fills every element of \a ancestors with an index into \a weights, drawn so
 * that index \c i is chosen in proportion to \c weights[i]. \a weights are
 * nonnegative linear weights with a positive sum; they need not be normalized.
 * Zero-weight indices are never chosen. \a ancestors may be any length.
 *
 * The four resamplers are the ones \ref Hol2006 "Hol, Schön, and Gustafsson (2006)" compare;
 * \ref Arulampalam2002 "Arulampalam et al. (2002)" describe their place in a particle filter.
 * All four invert the cumulative sum of the weights at a set of positions in [0, 1), and differ
 * only in how the positions are drawn (internal::invertCumulative()).
 *
 * A resampler that needs storage sized by the counts, as resamplers::Multinomial and
 * resamplers::Residual do, also takes it: \c r(weights, ancestors, rng, scratch) with \a scratch a
 * <tt>std::span\<Scalar\></tt> of at least <tt>weights.size() + ancestors.size()</tt> elements it
 * may use freely (ResamplerWithScratch). ParticleSet::resample() keeps that scratch and passes it,
 * so such a resampler allocates nothing there, while its three-argument form allocates what it
 * needs (see \ref Allocation). resamplers::Stratified and resamplers::Systematic need none.
 *
 * Each also runs under an execution policy as \c r(policy, weights, ancestors, rng) (see
 * \ref Execution). The draws stay sequential, in the order the sequential form takes them; the
 * cumulative sum is then a scan under the policy and each ancestor a binary search into it. The
 * sum is the one collective operation that \ref Murray2016 "Murray, Lee, and Jacob (2016)" find
 * these resamplers need in parallel. The ancestors are Sequential's but for rounding at the
 * boundaries between particles, except that resamplers::Multinomial leaves them in the order of
 * its draws rather than sorted, which is the same draw of ancestors.
 *
 * \sa ParticleSet::resample()
 */

/** \ingroup Resamplers
 * Requirements on a resampler for scalar type \a Scalar and random generator \a G.
 */
template <typename R, typename Scalar, typename G>
concept Resampler = requires(R const & r, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) {
    r(weights, ancestors, rng);
};

/** \ingroup Resamplers
 * A Resampler that also takes scratch, called as \c r(weights, ancestors, rng, scratch) with
 * \a scratch a <tt>std::span\<Scalar\></tt> of at least <tt>weights.size() + ancestors.size()</tt>
 * elements. ParticleSet::resample() calls a resampler this way if it can, with scratch it keeps.
 */
template <typename R, typename Scalar, typename G>
concept ResamplerWithScratch = Resampler<R, Scalar, G>
                            && requires(
                                   R const & r,
                                   std::span<Scalar const> weights,
                                   std::span<Eigen::Index> ancestors,
                                   G & rng,
                                   std::span<Scalar> scratch
                            ) { r(weights, ancestors, rng, scratch); };

/** \ingroup Resamplers
 * A Resampler that also runs under the execution policy \a P, called as
 * \c r(policy, weights, ancestors, rng). ParticleSet::resample() under a policy calls a resampler
 * this way if it can, and as a plain Resampler otherwise.
 */
template <typename R, typename P, typename Scalar, typename G>
concept ResamplerUnder =
    ExecutionPolicy<P>
    && requires(R const & r, P && policy, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) {
           r(std::forward<P>(policy), weights, ancestors, rng);
       };

namespace internal
{

/** \returns a uniform draw from [0, 1). Guards against std::uniform_real_distribution returning 1. */
template <std::floating_point Scalar, std::uniform_random_bit_generator G>
[[nodiscard]] Scalar
uniform01(G & rng)
{
    auto uniform = std::uniform_real_distribution<Scalar>{Scalar(0), Scalar(1)};
    return std::min(uniform(rng), std::nextafter(Scalar(1), Scalar(0)));
}

/** Fills \a ancestors by inverting the cumulative weights at positions \c u(j),
 * which must be nondecreasing in [0, 1). Zero-weight indices are never chosen.
 */
template <typename Scalar, typename Positions>
void
invertCumulative(std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, Positions u)
{
    EMRYS_FILTER_ASSERT(!weights.empty());
    Scalar total = 0;
    std::size_t lastPositive = 0;
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        EMRYS_FILTER_ASSERT(weights[i] >= Scalar(0) && "weights must be nonnegative");
        total += weights[i];
        if (weights[i] > Scalar(0))
        {
            lastPositive = i;
        }
    }
    EMRYS_FILTER_ASSERT(total > Scalar(0) && "weights must have a positive sum");

    std::size_t i = 0;
    Scalar cumulative = weights[0];
    for (std::size_t j = 0; j < ancestors.size(); ++j)
    {
        Scalar const position = u(j) * total;
        while (i < lastPositive && position >= cumulative)
        {
            cumulative += weights[++i];
        }
        ancestors[j] = Eigen::Index(i);
    }
}

/** invertCumulative() under \a policy, with \a u(j) a pure function of \c j, since the positions
 * are read in any order. The cumulative weights are a cumulativeSum() and each ancestor the
 * first index whose cumulative weight exceeds its position, which is what the sequential walk
 * finds. Under Sequential it is invertCumulative().
 */
template <ExecutionPolicy P, typename Scalar, typename Positions>
void
invertCumulative(P && policy, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, Positions const & u)
{
    if constexpr (isSequential<P>)
    {
        invertCumulative(weights, ancestors, u);
    }
    else
    {
        EMRYS_FILTER_ASSERT(!weights.empty());
        Eigen::Index const count = Eigen::Index(weights.size());
        auto cumulative = scratchVector<Scalar>(policy, weights.size());
        cumulativeSum(policy, weights, std::span<Scalar>{cumulative});
        // The last positive weight, or -1 for none, as the largest index of each range's.
        Eigen::Index const lastPositive = reduceRanges(
            policy,
            count,
            [&](Eigen::Index begin, Eigen::Index length) {
                Eigen::Index last = -1;
                for (Eigen::Index i = begin; i < begin + length; ++i)
                {
                    EMRYS_FILTER_ASSERT(weights[std::size_t(i)] >= Scalar(0) && "weights must be nonnegative");
                    if (weights[std::size_t(i)] > Scalar(0))
                    {
                        last = i;
                    }
                }
                return last;
            },
            [](Eigen::Index a, Eigen::Index b) {
                return b >= 0 ? b : a;
            }
        );
        Scalar const total = cumulative.back();
        EMRYS_FILTER_ASSERT(lastPositive >= 0 && total > Scalar(0) && "weights must have a positive sum");
        forEachIndex(std::forward<P>(policy), Eigen::Index(ancestors.size()), [&](Eigen::Index j) {
            Scalar const position = u(std::size_t(j)) * total;
            auto const above = std::upper_bound(cumulative.begin(), cumulative.end(), position);
            ancestors[std::size_t(j)] = std::min(Eigen::Index(above - cumulative.begin()), lastPositive);
        });
    }
}

/** \returns \a count uniform draws from [0, 1) with \a rng, in order, as uniform01() draws them,
 * in scratch for \a policy (see scratchVector()).
 */
template <std::floating_point Scalar, typename P, std::uniform_random_bit_generator G>
[[nodiscard]] auto
uniforms01(P const & policy, std::size_t count, G & rng)
{
    auto draws = scratchVector<Scalar>(policy, count);
    for (Scalar & draw : draws)
    {
        draw = uniform01<Scalar>(rng);
    }
    return draws;
}

}  // namespace internal

/** \ingroup Resamplers
 * \brief The resamplers, each made by its factory (decisions.md [policy-factories]): multinomial(),
 * stratified(), systematic(), and residual(). The types beside them name a resampler where a
 * type is needed.
 */
namespace resamplers
{

/** \ingroup Resamplers
 * \brief Draws each ancestor independently. Highest variance; mainly a reference.
 *
 * The resampling of the bootstrap filter of \ref Gordon1993 "Gordon, Salmond, and Smith (1993)":
 * \f$ M \f$ independent uniforms, sorted, invert the cumulative weights.
 */
struct Multinomial
{
    /** Fills \a ancestors with indices into \a weights, each drawn independently in proportion
     * to its weight, using \a rng, sorting the draws in \a scratch, which has at least as many
     * elements as \a ancestors.
     */
    template <std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(
        std::span<Scalar const> weights,
        std::span<Eigen::Index> ancestors,
        G & rng,
        std::span<Scalar> scratch
    ) const
    {
        EMRYS_FILTER_ASSERT(scratch.size() >= ancestors.size() && "the scratch must hold a draw per ancestor");
        auto const positions = scratch.first(ancestors.size());
        for (auto & p : positions)
        {
            p = internal::uniform01<Scalar>(rng);
        }
        std::ranges::sort(positions);
        internal::invertCumulative(weights, ancestors, [&](std::size_t j) {
            return positions[j];
        });
    }

    /** As the form with scratch, allocating the scratch for the draws. */
    template <std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        auto positions = std::vector<Scalar>(ancestors.size());
        (*this)(weights, ancestors, rng, std::span<Scalar>{positions});
    }

    /** As the sequential form, under \a policy: the uniforms are drawn as it draws them but not
     * sorted, so each ancestor is the inversion of its own draw, and the ancestors come in the
     * order of the draws. Under Sequential it is the sequential form.
     */
    template <ExecutionPolicy P, std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(P && policy, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        if constexpr (internal::isSequential<P>)
        {
            (*this)(weights, ancestors, rng);
        }
        else
        {
            auto const positions = internal::uniforms01<Scalar>(policy, ancestors.size(), rng);
            internal::invertCumulative(std::forward<P>(policy), weights, ancestors, [&](std::size_t j) {
                return positions[j];
            });
        }
    }
};

/** \ingroup Resamplers
 * \returns the multinomial resampler, Multinomial
 */
[[nodiscard]] constexpr Multinomial
multinomial()
{
    return {};
}

/** \ingroup Resamplers
 * \brief Draws one uniform in each of \a M equal strata of [0, 1).
 *
 * Stratified sampling as \ref Kitagawa1996 "Kitagawa (1996)" applies it to the Monte Carlo
 * filter, as given in \ref Douc2005 "Douc, Cappé, and Moulines (2005)", §2.3: position \f$ j \f$ is
 * \f$ (j + u_j) / M \f$ with independent \f$ u_j \f$.
 */
struct Stratified
{
    /** Fills \a ancestors with indices into \a weights, one drawn in each stratum of [0, 1),
     * using \a rng.
     */
    template <std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        Scalar const m = Scalar(ancestors.size());
        internal::invertCumulative(weights, ancestors, [&](std::size_t j) {
            return (Scalar(j) + internal::uniform01<Scalar>(rng)) / m;
        });
    }

    /** As the sequential form, under \a policy, with the uniforms drawn first in stratum order,
     * as the sequential form draws them. Under Sequential it is the sequential form.
     */
    template <ExecutionPolicy P, std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(P && policy, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        if constexpr (internal::isSequential<P>)
        {
            (*this)(weights, ancestors, rng);
        }
        else
        {
            Scalar const m = Scalar(ancestors.size());
            auto const offsets = internal::uniforms01<Scalar>(policy, ancestors.size(), rng);
            internal::invertCumulative(std::forward<P>(policy), weights, ancestors, [&](std::size_t j) {
                return (Scalar(j) + offsets[j]) / m;
            });
        }
    }
};

/** \ingroup Resamplers
 * \returns the stratified resampler, Stratified
 */
[[nodiscard]] constexpr Stratified
stratified()
{
    return {};
}

/** \ingroup Resamplers
 * \brief Uses one uniform offset for \a M evenly spaced positions in [0, 1).
 *
 * Index \c i is chosen either floor or ceil of \f$ M w_i / \sum w \f$ times. Low
 * variance and cheap; the usual default.
 *
 * The systematic resampling of \ref Carpenter1999 "Carpenter, Clifford, and Fearnhead (1999)",
 * as given in \ref Douc2005 "Douc, Cappé, and Moulines (2005)", §2.4, who credit Carpenter et al.;
 * \ref Arulampalam2002 "Arulampalam et al. (2002)", §V-A, credit
 * \ref Kitagawa1996 "Kitagawa (1996)" instead: position \f$ j \f$ is \f$ (j + u) / M \f$ with one
 * uniform \f$ u \f$ for all.
 */
struct Systematic
{
    /** Fills \a ancestors with indices into \a weights at \f$ M \f$ evenly spaced positions
     * offset by one uniform draw from \a rng.
     */
    template <std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        Scalar const m = Scalar(ancestors.size());
        Scalar const offset = internal::uniform01<Scalar>(rng);
        internal::invertCumulative(weights, ancestors, [&](std::size_t j) {
            return (Scalar(j) + offset) / m;
        });
    }

    /** As the sequential form, under \a policy. Under Sequential it is the sequential form. */
    template <ExecutionPolicy P, std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(P && policy, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        Scalar const m = Scalar(ancestors.size());
        Scalar const offset = internal::uniform01<Scalar>(rng);
        internal::invertCumulative(std::forward<P>(policy), weights, ancestors, [&](std::size_t j) {
            return (Scalar(j) + offset) / m;
        });
    }
};

/** \ingroup Resamplers
 * \returns the systematic resampler, Systematic, the usual default
 */
[[nodiscard]] constexpr Systematic
systematic()
{
    return {};
}

/** \ingroup Resamplers
 * \brief Chooses index \c i floor(\f$ M w_i / \sum w \f$) times deterministically,
 * then draws the remainder with \a Inner from the fractional parts.
 *
 * Residual resampling (\ref Liu1998 "Liu and Chen (1998)"), as given in
 * \ref Douc2005 "Douc, Cappé, and Moulines (2005)", §2.2, equations (4) and (5): the integer parts are
 * deterministic, and only the \f$ M - \sum_i \lfloor M w_i / \sum w \rfloor \f$ remaining
 * ancestors are random.
 */
template <typename Inner = Multinomial>
struct Residual
{
    Inner fractional{};  ///< the resampler for the remainder, applied to the fractional parts

    /** Fills \a ancestors with floor(\f$ M w_i / \sum w \f$) copies of each index \c i, then
     * draws the rest with #fractional from the fractional parts, using \a rng. The fractional parts
     * are kept in \a scratch, which has at least <tt>weights.size() + ancestors.size()</tt>
     * elements; what follows them is #residual's scratch, if it takes any.
     */
    template <std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(
        std::span<Scalar const> weights,
        std::span<Eigen::Index> ancestors,
        G & rng,
        std::span<Scalar> scratch
    ) const
    {
        EMRYS_FILTER_ASSERT(
            scratch.size() >= weights.size() + ancestors.size()
            && "the scratch must hold a fraction per weight and a draw per ancestor"
        );
        Scalar total = 0;
        for (Scalar const w : weights)
        {
            EMRYS_FILTER_ASSERT(w >= Scalar(0) && "weights must be nonnegative");
            total += w;
        }
        EMRYS_FILTER_ASSERT(total > Scalar(0) && "weights must have a positive sum");

        Scalar const m = Scalar(ancestors.size());
        auto const fractions = scratch.first(weights.size());
        std::size_t out = 0;
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            Scalar const expected = m * weights[i] / total;
            Scalar const copies = std::floor(expected);
            fractions[i] = expected - copies;
            // Converting a negative or NaN count to std::size_t is undefined, so it takes no copies.
            std::size_t const count = copies > Scalar(0) ? std::size_t(copies) : 0;
            for (std::size_t k = 0; k < count && out < ancestors.size(); ++k)
            {
                ancestors[out++] = Eigen::Index(i);
            }
        }
        if (out < ancestors.size())
        {
            if constexpr (ResamplerWithScratch<Inner, Scalar, G>)
            {
                fractional(
                    std::span<Scalar const>{fractions},
                    ancestors.subspan(out),
                    rng,
                    scratch.subspan(weights.size())
                );
            }
            else
            {
                fractional(std::span<Scalar const>{fractions}, ancestors.subspan(out), rng);
            }
        }
    }

    /** As the form with scratch, allocating the scratch. */
    template <std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        auto scratch = std::vector<Scalar>(weights.size() + ancestors.size());
        (*this)(weights, ancestors, rng, std::span<Scalar>{scratch});
    }

    /** As the sequential form, under \a policy: the total is a reduction, each index's copies and
     * fraction a for-each, and the copies are placed by a scan of their counts. The remainder is
     * drawn by #fractional under \a policy if it runs under one. Under Sequential it is the
     * sequential form.
     */
    template <ExecutionPolicy P, std::floating_point Scalar, std::uniform_random_bit_generator G>
    void
    operator()(P && policy, std::span<Scalar const> weights, std::span<Eigen::Index> ancestors, G & rng) const
    {
        if constexpr (internal::isSequential<P>)
        {
            (*this)(weights, ancestors, rng);
        }
        else
        {
            Eigen::Index const count = Eigen::Index(weights.size());
            Scalar const total = internal::reduceRanges(
                policy,
                count,
                [&](Eigen::Index begin, Eigen::Index length) {
                    Scalar sum = 0;
                    for (Scalar const w : weights.subspan(std::size_t(begin), std::size_t(length)))
                    {
                        sum += w;
                    }
                    return sum;
                },
                [](Scalar a, Scalar b) {
                    return a + b;
                }
            );
            EMRYS_FILTER_ASSERT(total > Scalar(0) && "weights must have a positive sum");

            Scalar const m = Scalar(ancestors.size());
            auto fractions = internal::scratchVector<Scalar>(policy, weights.size());
            auto ends = internal::scratchVector<Eigen::Index>(policy, weights.size());
            internal::forEachIndex(policy, count, [&](Eigen::Index i) {
                Scalar const expected = m * weights[std::size_t(i)] / total;
                Scalar const copies = std::floor(expected);
                fractions[std::size_t(i)] = expected - copies;
                ends[std::size_t(i)] = Eigen::Index(copies);
            });
            internal::cumulativeSum(policy, std::span<Eigen::Index const>{ends}, std::span<Eigen::Index>{ends});
            Eigen::Index const size = Eigen::Index(ancestors.size());
            internal::forEachIndex(policy, count, [&](Eigen::Index i) {
                Eigen::Index const begin = i == 0 ? 0 : ends[std::size_t(i - 1)];
                for (Eigen::Index k = begin; k < std::min(ends[std::size_t(i)], size); ++k)
                {
                    ancestors[std::size_t(k)] = i;
                }
            });
            std::size_t const out = std::size_t(std::min(ends.back(), size));
            if (out < ancestors.size())
            {
                auto const remainder = std::span<Scalar const>{fractions};
                if constexpr (ResamplerUnder<Inner, P, Scalar, G>)
                {
                    fractional(std::forward<P>(policy), remainder, ancestors.subspan(out), rng);
                }
                else
                {
                    fractional(remainder, ancestors.subspan(out), rng);
                }
            }
        }
    }
};

/** \ingroup Resamplers
 * \returns the residual resampler, a Residual drawing the remainder with \a inner (Multinomial by
 * default)
 */
template <typename Inner = Multinomial>
[[nodiscard]] constexpr Residual<Inner>
residual(Inner inner = {})
{
    return {std::move(inner)};
}

}  // namespace resamplers

}  // namespace Emrys
