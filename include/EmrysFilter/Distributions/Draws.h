#pragma once

/** \file
 * Lazy draws: Draws, the range drawSamples() and drawSamplesWithLogDensity() return, which draws
 * from the caller's generator as it is read; SamplesWithLogDensity and SampleWithLogDensity, what
 * the draws with their log-densities hold; and the samplers behind the range (decisions.md
 * [random-draws]). Each element of the range is one drawSample() of the density, or one
 * drawSampleWithLogDensity(), so a stream of single draws and the range give the same sequence
 * from the same generator, and reading the range allocates nothing for a bounded space
 * [allocation]. A kernel density's and a mixture's samplers are in their own headers. Read
 * Sampling.h next.
 */

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>

namespace Emrys
{

/** \ingroup Distributions
 * Samples in \a Space together with the log of the density they were drawn from,
 * evaluated at each: what Draws::collect() returns for drawSamplesWithLogDensity().
 */
template <typename Space, typename Scalar = double>
struct SamplesWithLogDensity
{
    SampleSet<Space, Scalar> samples;                    ///< one sample per column
    Eigen::Array<Scalar, Eigen::Dynamic, 1> logDensity;  ///< \c logDensity(i) is the log-density at \c samples.col(i)
};

/** \ingroup Distributions
 * One sample in \a Space, in a vector sized by the space, together with the log of the density it
 * was drawn from, evaluated at it: what drawSampleWithLogDensity() returns and each element of
 * drawSamplesWithLogDensity() is, so that a bounded space allocates nothing [allocation].
 */
template <typename Space, typename Scalar = double>
struct SampleWithLogDensity
{
    typename Moments<Space, Scalar>::VectorType sample;  ///< the sample
    Scalar logDensity = 0;                               ///< the log-density at \c sample
};

namespace internal
{

/** Holds the density that a range of draws reads: the density itself where \a Held is a value,
 * moved in from an rvalue such as <tt>params.toDensity().value()</tt>, so that a range kept in a
 * variable never outlives it.
 */
template <typename Held>
class HeldDensity
{
public:

    /** Holds \a density. */
    explicit HeldDensity(Held density)
        : m_density{std::move(density)}
    {}

    /** \returns the density */
    [[nodiscard]] Held const &
    get() const
    {
        return m_density;
    }

private:

    Held m_density;  ///< the density
};

/** Holds the density that a range of draws reads by reference, where \a Held names an lvalue, as a
 * view of the standard library holds its range: the density must outlive the range.
 */
template <typename Held>
class HeldDensity<Held const &>
{
public:

    /** Refers to \a density. */
    explicit HeldDensity(Held const & density)
        : m_density{&density}
    {}

    /** \returns the density */
    [[nodiscard]] Held const &
    get() const
    {
        return *m_density;
    }

private:

    Held const * m_density;  ///< the density
};

/** The type a range of draws holds for a density given as \a D, a forwarding reference's deduced
 * type: a const reference for an lvalue, the density itself for an rvalue.
 */
template <typename D>
using HeldDensityType =
    std::conditional_t<std::is_lvalue_reference_v<D>, std::remove_reference_t<D> const &, std::remove_cvref_t<D>>;

/** Whether drawSample() of \a D draws from \a G: the densities whose draws Draws reads.
 *
 * The call is unqualified, so that each density's drawSample(), declared in its own header after
 * this one, is found by argument-dependent lookup in \c Emrys.
 */
template <typename D, typename G>
concept DrawsSingly = requires(D const & density, G & rng) { drawSample(density, rng); };

/** The sampler of a density whose draws are its drawSample() and drawSampleWithLogDensity(), one
 * call per element, taking nothing from one draw to the next.
 */
template <typename Held>
class SingleDraws
{
public:

    using DensityType = std::remove_cvref_t<Held>;  ///< the density drawn from

    /** Draws from \a density. */
    explicit SingleDraws(HeldDensity<Held> density)
        : m_density{std::move(density)}
    {}

    /** \returns the density drawn from */
    [[nodiscard]] DensityType const &
    density() const
    {
        return m_density.get();
    }

    /** \returns the next draw, drawSample() of the density */
    template <std::uniform_random_bit_generator G>
    [[nodiscard]] auto
    draw(G & rng)
    {
        return drawSample(density(), rng);
    }

    /** \returns the next draw with its log-density, drawSampleWithLogDensity() of the density */
    template <std::uniform_random_bit_generator G>
    [[nodiscard]] auto
    drawWithLogDensity(G & rng)
    {
        return drawSampleWithLogDensity(density(), rng);
    }

private:

    HeldDensity<Held> m_density;  ///< the density
};

/** The sampler that drawSamples() reads a density held as \a Held with: SingleDraws, unless the
 * density's header specializes it for \a Density, as a kernel density's and a mixture's do.
 */
template <typename Held, typename Density = std::remove_cvref_t<Held>>
struct SamplerOf
{
    using type = SingleDraws<Held>;  ///< the sampler
};

/** Whether \a Sampler's draws with their log-densities are its draws followed by its density's
 * logDensity() at each, so that the log-densities can be evaluated apart, under a policy: true for
 * the samplers that say so with a constant \c LogDensityAfterDraw.
 */
template <typename Sampler>
concept LogDensityAfterDraw = Sampler::LogDensityAfterDraw;

/** The type of one draw of \a Sampler from \a G: its draw()'s */
template <typename Sampler, typename G, bool WithLogDensity>
struct DrawOf
{
    using type = std::remove_cvref_t<decltype(std::declval<Sampler &>().draw(std::declval<G &>()))>;  ///< the draw
};

/** The type of one draw of \a Sampler from \a G with its log-density: its drawWithLogDensity()'s */
template <typename Sampler, typename G>
struct DrawOf<Sampler, G, true>
{
    using type =
        std::remove_cvref_t<decltype(std::declval<Sampler &>().drawWithLogDensity(std::declval<G &>()))>;  ///< the draw
};

}  // namespace internal

/** \ingroup Distributions
 * A range of draws from a density, read once: what drawSamples() returns, and with
 * \a WithLogDensity what drawSamplesWithLogDensity() returns. It draws from the generator as it is
 * read, so its draws are those of the generator's state when it is read, not when it was made.
 *
 * Each element is the next draw of \a Sampler, which for a density's own draws is one drawSample()
 * (or drawSampleWithLogDensity()) per element, so the range and a stream of single draws give the
 * same sequence from the same generator. Reading it holds one draw at a time, in a vector sized by
 * the space, so on a bounded space it allocates nothing. It is a sized input range: begin() may be
 * called once, and the range is moved, never copied, since a copy would draw from the same
 * generator. collect() reads it into a SampleSet, a SamplesWithLogDensity, or, for a categorical
 * distribution, a vector of indices; ParticleSet and ParticleFilter take it whole and draw into
 * their own storage.
 *
 * \tparam Sampler         the sampler of the density, internal to the library
 * \tparam G               the generator, held by reference: it must outlive the range
 * \tparam WithLogDensity  whether each element is a SampleWithLogDensity
 */
template <typename Sampler, std::uniform_random_bit_generator G, bool WithLogDensity = false>
class Draws
{
public:

    using DensityType = typename Sampler::DensityType;     ///< the density drawn from
    using SpaceType = typename DensityType::SpaceType;     ///< the space the draws are in
    using Scalar = typename DensityType::Scalar;           ///< the scalar type
    using GeneratorType = G;                               ///< the generator drawn from
    static constexpr bool HasLogDensity = WithLogDensity;  ///< whether each element has its log-density

    using value_type = typename internal::DrawOf<Sampler, G, WithLogDensity>::type;  ///< the type of one draw

    /** The iterator of a range of draws: an input iterator, whose increment draws the next. */
    class iterator
    {
    public:

        using value_type = Draws::value_type;              ///< the type of one draw
        using difference_type = std::ptrdiff_t;            ///< as for any input iterator
        using iterator_concept = std::input_iterator_tag;  ///< read once

        /** An iterator over no range, which an input iterator must have. */
        iterator() = default;

        /** \returns the current draw */
        [[nodiscard]] value_type const &
        operator*() const
        {
            return m_draws->m_current;
        }

        /** Draws the next element, or ends the range after the last. */
        iterator &
        operator++()
        {
            m_draws->advance();
            return *this;
        }

        /** Draws the next element, as the prefix increment does. */
        void
        operator++(int)
        {
            ++*this;
        }

        /** \returns whether \a it has passed the last draw */
        [[nodiscard]] friend bool
        operator==(iterator const & it, std::default_sentinel_t)
        {
            return it.atEnd();
        }

    private:

        /** \returns whether the range has no draw left: a member, since MSVC gives a friend
         * defined in a nested class none of the nested class's access to the enclosing one
         */
        [[nodiscard]] bool
        atEnd() const
        {
            return m_draws->m_remaining == 0;
        }

        friend Draws;  ///< the range makes its iterator

        /** An iterator over \a draws. */
        explicit iterator(Draws & draws)
            : m_draws{&draws}
        {}

        Draws * m_draws = nullptr;  ///< the range read
    };

    /** \a count draws by \a sampler from \a rng, which must outlive the range. */
    Draws(Sampler sampler, Eigen::Index count, G & rng)
        : m_sampler{std::move(sampler)}
        , m_rng{&rng}
        , m_count{count}
        , m_remaining{count}
    {
        EMRYS_FILTER_ASSERT(count >= 0);
    }

    /** Not copied: a copy would draw from the same generator as the range. */
    Draws(Draws const &) = delete;

    /** Takes over \a other, which must not have been read. */
    Draws(Draws &&) = default;

    /** Not copied, as the copy constructor is not. */
    Draws &
    operator=(Draws const &) = delete;

    /** Takes over \a other, which must not have been read. */
    Draws &
    operator=(Draws &&) = default;

    /** Ends the range, whether read or not. */
    ~Draws() = default;

    /** \returns an iterator at the first draw, which it draws; a range is read once */
    [[nodiscard]] iterator
    begin()
    {
        start();
        return iterator{*this};
    }

    /** \returns the end of the range */
    [[nodiscard]] std::default_sentinel_t
    end() const
    {
        return {};
    }

    /** \returns the number of draws */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return m_count;
    }

    /** \returns the density drawn from */
    [[nodiscard]] DensityType const &
    density() const
    {
        return m_sampler.density();
    }

    /** \returns the space the draws are in */
    [[nodiscard]] decltype(auto)
    space() const
    {
        return density().space();
    }

    /** \returns the number of coordinates of each draw */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return density().valueDimension();
    }

    /** \returns every draw, read in order: a SampleSet, one draw per column; with log-densities a
     * SamplesWithLogDensity; and for a categorical distribution, whose draws are indices, a vector
     * of them
     */
    [[nodiscard]] auto
    collect()
    {
        if constexpr (std::same_as<value_type, Eigen::Index>)
        {
            auto result = std::vector<Eigen::Index>{};
            result.reserve(std::size_t(m_count));
            for (Eigen::Index const index : *this)
            {
                result.push_back(index);
            }
            return result;
        }
        else if constexpr (WithLogDensity)
        {
            auto result = SamplesWithLogDensity<SpaceType, Scalar>{
                SampleSet<SpaceType, Scalar>{valueDimension(), m_count},
                Eigen::Array<Scalar, Eigen::Dynamic, 1>(m_count)
            };
            Eigen::Index j = 0;
            for (auto const & draw : *this)
            {
                result.samples.col(j) = draw.sample;
                result.logDensity(j) = draw.logDensity;
                ++j;
            }
            return result;
        }
        else
        {
            auto result = SampleSet<SpaceType, Scalar>{valueDimension(), m_count};
            Eigen::Index j = 0;
            for (auto const & draw : *this)
            {
                result.col(j) = draw;
                ++j;
            }
            return result;
        }
    }

    /** \returns every draw with its log-density, as collect() does, with the draws taken in order
     * and their log-densities evaluated under \a policy (see \ref Execution), for a density whose
     * log-density at a draw is a full evaluation, as a kernel density's and a mixture's are: the
     * result is collect()'s.
     */
    template <ExecutionPolicy P>
        requires WithLogDensity && internal::LogDensityAfterDraw<Sampler>
    [[nodiscard]] SamplesWithLogDensity<SpaceType, Scalar>
    collect(P && policy)
    {
        EMRYS_FILTER_ASSERT(!m_started && "a range of draws is read once");
        m_started = true;
        prepare();
        auto result = SamplesWithLogDensity<SpaceType, Scalar>{
            SampleSet<SpaceType, Scalar>{valueDimension(), m_count},
            Eigen::Array<Scalar, Eigen::Dynamic, 1>(m_count)
        };
        for (Eigen::Index j = 0; j < m_count; ++j)
        {
            result.samples.col(j) = m_sampler.draw(*m_rng);
        }
        m_remaining = 0;
        DensityType const & source = density();
        internal::forEachIndex(std::forward<P>(policy), m_count, [&](Eigen::Index j) {
            result.logDensity(j) = source.logDensity(result.samples.col(j));
        });
        return result;
    }

private:

    /** Lets the sampler take what it draws for the whole range, as a resampler's ancestors. */
    void
    prepare()
    {
        if constexpr (requires { m_sampler.prepare(m_count, *m_rng); })
        {
            m_sampler.prepare(m_count, *m_rng);
        }
    }

    /** \returns the sampler's next draw */
    [[nodiscard]] value_type
    drawOne()
    {
        if constexpr (WithLogDensity)
        {
            return m_sampler.drawWithLogDensity(*m_rng);
        }
        else
        {
            return m_sampler.draw(*m_rng);
        }
    }

    /** Begins reading: the first draw, unless the range is empty. */
    void
    start()
    {
        EMRYS_FILTER_ASSERT(!m_started && "a range of draws is read once");
        m_started = true;
        if (m_remaining > 0)
        {
            prepare();
            m_current = drawOne();
        }
    }

    /** Moves past the current draw: draws the next, unless it was the last. */
    void
    advance()
    {
        EMRYS_FILTER_ASSERT(m_remaining > 0);
        --m_remaining;
        if (m_remaining > 0)
        {
            m_current = drawOne();
        }
    }

    Sampler m_sampler;         ///< what draws each element
    G * m_rng;                 ///< the generator
    Eigen::Index m_count;      ///< the number of draws
    Eigen::Index m_remaining;  ///< the draws not yet passed, the current one included
    value_type m_current{};    ///< the current draw
    bool m_started = false;    ///< whether the range has been read
};

/** \ingroup Distributions
 * A range of draws, each with its log-density: what drawSamplesWithLogDensity() returns.
 */
template <typename Sampler, std::uniform_random_bit_generator G>
using DrawsWithLogDensity = Draws<Sampler, G, true>;

namespace internal
{

/** Whether \a T is a Draws */
template <typename T>
inline constexpr bool isDraws = false;

/** Whether \a T is a Draws: true */
template <typename Sampler, typename G, bool WithLogDensity>
inline constexpr bool isDraws<Draws<Sampler, G, WithLogDensity>> = true;

/** A range of draws whose elements are values in its space, without log-densities: what
 * ParticleSet and ParticleFilter are built from.
 */
template <typename R>
concept ValueDraws = isDraws<std::remove_cvref_t<R>> && !std::remove_cvref_t<R>::HasLogDensity
                  && !std::same_as<typename std::remove_cvref_t<R>::value_type, Eigen::Index>;

/** Picks elements of a weighted distribution, such as a kernel density's particles or a mixture's
 * components, one per draw and independently, as internal::drawIndex() picks one: it keeps the
 * running sums of the weights, from the density's allocator, and searches them for each uniform
 * draw, so that each pick costs \f$ O(\log n) \f$ and picks as drawIndex() would from the same
 * generator.
 */
template <typename Scalar, typename Allocator>
class IndependentPicks
{
public:

    /** Picks with storage from \a allocator. */
    explicit IndependentPicks(Allocator const & allocator)
        : m_sums(bufferOf<Scalar>(allocator))
    {}

    /** Takes the weights \a weights, which need not be normalized, and keeps their running sums;
     * an element of zero or negative weight is never picked.
     */
    template <std::uniform_random_bit_generator G>
    void
    prepare(Buffer<Scalar, Allocator> weights, Eigen::Index, G &)
    {
        EMRYS_FILTER_ASSERT(!weights.empty() && "draw from at least one element");
        m_sums = std::move(weights);
        Scalar sum = 0;
        for (std::size_t i = 0; i < m_sums.size(); ++i)
        {
            if (m_sums[i] > 0)
            {
                sum += m_sums[i];
                m_last = Eigen::Index(i);
            }
            m_sums[i] = sum;
        }
    }

    /** \returns the next pick: the first element whose running sum exceeds a uniform draw, or the
     * last of positive weight where rounding leaves the sum short of the draw
     */
    template <std::uniform_random_bit_generator G>
    [[nodiscard]] Eigen::Index
    next(G & rng)
    {
        Scalar const u = std::uniform_real_distribution<Scalar>{}(rng);
        auto const found = std::upper_bound(m_sums.begin(), m_sums.end(), u);
        return found == m_sums.end() ? m_last : Eigen::Index(found - m_sums.begin());
    }

private:

    Buffer<Scalar, Allocator> m_sums;  ///< the running sums of the positive weights
    Eigen::Index m_last = 0;           ///< the last element of positive weight
};

/** Picks every element a range of draws needs at the first read, by \a R, a resampler, so that the
 * picks spread over the elements as the resampler spreads its ancestors; the range is then not the
 * stream of single draws.
 */
template <typename Scalar, typename Allocator, typename R>
class ResampledPicks
{
public:

    /** Picks by \a resampler with storage from \a allocator. */
    ResampledPicks(R resampler, Allocator const & allocator)
        : m_resampler{std::move(resampler)}
        , m_ancestors(bufferOf<Eigen::Index>(allocator))
    {}

    /** Picks \a count elements of the weights \a weights by the resampler, drawing from \a rng. */
    template <std::uniform_random_bit_generator G>
    void
    prepare(Buffer<Scalar, Allocator> weights, Eigen::Index count, G & rng)
    {
        m_ancestors.resize(std::size_t(count));
        m_resampler(std::span<Scalar const>{weights}, std::span<Eigen::Index>{m_ancestors}, rng);
        m_next = 0;
    }

    /** \returns the next of the picks */
    template <std::uniform_random_bit_generator G>
    [[nodiscard]] Eigen::Index
    next(G &)
    {
        return m_ancestors[m_next++];
    }

private:

    R m_resampler;                                ///< the resampler
    Buffer<Eigen::Index, Allocator> m_ancestors;  ///< the picks, in order
    std::size_t m_next = 0;                       ///< the next pick to return
};

/** \returns the range of \a count draws from \a density by \a rng that drawSamples() and, with
 * \a WithLogDensity, drawSamplesWithLogDensity() return
 */
template <bool WithLogDensity, typename D, std::uniform_random_bit_generator G>
[[nodiscard]] auto
drawsOf(D && density, Eigen::Index count, G & rng)
{
    using Held = HeldDensityType<D>;
    using Sampler = typename SamplerOf<Held>::type;
    return Draws<Sampler, G, WithLogDensity>{Sampler{HeldDensity<Held>{std::forward<D>(density)}}, count, rng};
}

}  // namespace internal

/** \ingroup Distributions
 * \returns a range of \a n draws from \a density by \a rng, read as it is iterated: each element
 * is the next drawSample() of \a density, so the range and a stream of single draws give the same
 * sequence from the same generator (see Draws). Its collect() gives the draws as a SampleSet;
 * ParticleSet and ParticleFilter take the range itself.
 *
 * The range holds \a density by reference when given an lvalue, which must outlive it, and moves
 * it in when given an rvalue, such as <tt>params.toDensity().value()</tt>. The generator is held
 * by reference.
 *
 * \sa drawSamplesWithLogDensity(), drawSample()
 */
template <typename D, std::uniform_random_bit_generator G>
    requires internal::DrawsSingly<std::remove_cvref_t<D>, G>
[[nodiscard]] auto
drawSamples(D && density, Eigen::Index n, G & rng)
{
    return internal::drawsOf<false>(std::forward<D>(density), n, rng);
}

/** \ingroup Distributions
 * \returns a range of \a n draws from \a density by \a rng, each with the log-density of
 * \a density at it: each element is the next drawSampleWithLogDensity() of \a density, which costs
 * almost nothing extra for a %Gaussian in a linear space. It holds \a density as drawSamples()
 * does.
 *
 * \sa drawSamples(), drawSampleWithLogDensity()
 */
template <typename D, std::uniform_random_bit_generator G>
    requires internal::DrawsSingly<std::remove_cvref_t<D>, G>
[[nodiscard]] auto
drawSamplesWithLogDensity(D && density, Eigen::Index n, G & rng)
{
    return internal::drawsOf<true>(std::forward<D>(density), n, rng);
}

}  // namespace Emrys
