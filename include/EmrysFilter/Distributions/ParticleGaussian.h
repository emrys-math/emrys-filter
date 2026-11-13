#pragma once

/** \file
 * The state of a Rao-Blackwellized particle filter (decisions.md [rb-particle-filter]): ParticleGaussianParams,
 * weighted particles for one block with a %Gaussian or a hybrid of Gaussians over the rest each, built by
 * particleGaussian() or drawn from a joint %Gaussian by drawParticleGaussian(), and its density
 * ParticleGaussian, which gives the collapse. Read \ref Models next; the design
 * review is docs/guide/rao-blackwellized-filters.md.
 */

#include <expected>
#include <random>
#include <span>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Conditionals.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

template <
    Space Space_,
    typename Path_,
    typename Covariance_ = SharedCovariance,
    RealScalar Scalar_ = double,
    typename Allocator_ = DefaultAllocator,
    typename Component_ = Gaussian<typename internal::BlockSplit<Space_, Path_>::RestSpace, Scalar_>>
class ParticleGaussian;

template <
    Space Space_,
    typename Path_,
    typename Model_,
    typename Covariance_,
    typename Method_,
    RealScalar Scalar_,
    typename Allocator_,
    typename Upkeep_>
class RbParticleFilter;

namespace internal
{

/** The particles of a ParticleGaussianParams, as the points of a Rao-Blackwellized state (see withPoint()). */
template <typename Space_, typename Split, typename Particles>
class ParticlePoints
{
public:

    using Scalar = typename Particles::Scalar;        ///< the scalar type
    using SpaceType = Space_;                         ///< the whole state space
    using PathType = typename Split::PathType;        ///< the path naming the sampled block
    using PinnedSpace = typename Split::PinnedSpace;  ///< the sampled block's space
    using RestSpace = typename Split::RestSpace;      ///< the rest's space
    static constexpr bool HasKernels = false;         ///< particles are point masses

    /** Refers to \a particles of the block that \a split names in \a space, which all outlive this. */
    ParticlePoints(Space_ const & space, Split const & split, Particles const & particles)
        : m_space{space}
        , m_split{split}
        , m_particles{particles}
    {}

    /** \returns the whole state space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the space of the rest of the state, in order */
    [[nodiscard]] RestSpace const &
    restSpace() const
    {
        return m_split.restSpace();
    }

    /** \returns the fixed block's rows in a whole state's values */
    [[nodiscard]] BlockRows
    pinnedRows() const
    {
        return m_split.pinnedRows();
    }

    /** \returns the rest's rows in a whole state's values, in order */
    [[nodiscard]] auto const &
    restIndices() const
    {
        return m_split.restIndices();
    }

    /** \returns the fixed block's rows in a whole state's differences */
    [[nodiscard]] BlockRows
    pinnedDifferenceRows() const
    {
        return m_split.pinnedDifferenceRows();
    }

    /** \returns the rest's rows in a whole state's differences, in order */
    [[nodiscard]] auto const &
    restDifferenceIndices() const
    {
        return m_split.restDifferenceIndices();
    }

    /** \returns the number of particles */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return m_particles.size();
    }

    /** \returns particle \a i's value of the block */
    [[nodiscard]] auto
    center(Eigen::Index i) const
    {
        return m_particles.state(i);
    }

private:

    Space_ const & m_space;         ///< the whole state space
    Split const & m_split;          ///< the block split
    Particles const & m_particles;  ///< the particles
};

}  // namespace internal

/** \ingroup Distributions
 * \brief The parameters of a Rao-Blackwellized particle filter's state: weighted particles for one
 * block of the state, each carrying a %Gaussian, or a hybrid of Gaussians, over the rest.
 *
 * The marginalized particle representation of \ref Doucet2000 "Doucet et al. (2000)" and
 * \ref Schon2005 "Schön, Gustafsson, and Nordlund (2005)". The density is \f$ p(x) = \sum_i w_i \, \delta(x_p - p_i) \,
 * \mathcal{N}(x_r; \mu_i, \Sigma_i) \f$: the block \f$ x_p \f$ named by the path \a Path_ is sampled, and the rest \f$
 * x_r \f$ is %Gaussian given each particle. With \a Covariance_ SharedCovariance, every particle's %Gaussian has the
 * same covariance; with PerPointCovariance, each has its own (see RbParticleFilter).
 *
 * With \a Component_ a HybridGaussian over the rest, each particle holds an IMM's state instead,
 * \f$ p(x) = \sum_i w_i \, \delta(x_p - p_i) \sum_m P(m \mid i) \, \mathcal{N}(x_r; \mu_{im}, \Sigma_{im}) \f$,
 * whose modes() sums the particles' mode probabilities. Such a state has PerPointCovariance, since
 * mixing a particle's modes gives each particle covariances of its own.
 *
 * The particles are a ParticleSet over the block, read through particles(). Upkeep (see
 * \ref Upkeep) applies to the state as it does to a ParticleSet: resample() copies each
 * particle's %Gaussian along with it, so <tt>upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))</tt> works
 * unchanged, and Roughen moves the particles while leaving their Gaussians alone.
 *
 * Build one with particleGaussian() from particles and a %Gaussian over the rest, or draw one from
 * a %Gaussian over the whole state with drawParticleGaussian(). Its collapse is its density's, the
 * ParticleGaussian from toDensity(), which checks the weights and the Gaussians first.
 *
 * The particles and their Gaussians come from the allocator (see \ref Allocation). A state given
 * a capacity with setCapacity() reserves room for that many particles, Gaussians, and scratch,
 * so resampling and updating at a steady size allocate nothing.
 *
 * \tparam Space_       the whole state space
 * \tparam Path_        the block the particles sample, as a tag or a TagPath
 * \tparam Covariance_  SharedCovariance or PerPointCovariance
 * \tparam Scalar_      scalar type, \c double by default
 * \tparam Allocator_   allocator for the particles and Gaussians (see \ref Allocation)
 * \tparam Component_   each particle's distribution over the rest: a Gaussian, by default, or a
 *                     HybridGaussian, in \a Scalar_
 *
 * \sa class ParticleGaussian, RbParticleFilter, GridGaussianParams
 */
template <
    Space Space_,
    typename Path_,
    typename Covariance_ = SharedCovariance,
    RealScalar Scalar_ = double,
    typename Allocator_ = DefaultAllocator,
    typename Component_ = Gaussian<typename internal::BlockSplit<Space_, Path_>::RestSpace, Scalar_>>
class ParticleGaussianParams
{
    static_assert(internal::CovarianceForm<Covariance_>, "the covariance is SharedCovariance or PerPointCovariance");
    static constexpr bool Shared = std::is_same_v<Covariance_, SharedCovariance>;  ///< one covariance for all
    using Split = internal::BlockSplit<Space_, Path_>;  ///< the state divided into block and rest

    /** Whether each particle holds a hybrid of Gaussians rather than one %Gaussian. */
    static constexpr bool IsHybrid = internal::isGaussianHybrid<Params<Component_>>;

    static_assert(
        std::is_same_v<
            Params<Component_>,
            GaussianParams<typename Params<Component_>::SpaceType, typename Params<Component_>::Scalar>>
            || IsHybrid,
        "a particle holds a Gaussian or a hybrid of Gaussians"
    );
    static_assert(
        std::is_same_v<typename Params<Component_>::SpaceType, typename Split::RestSpace>
            && std::is_same_v<typename Params<Component_>::Scalar, Scalar_>,
        "a particle's distribution is over the rest of the state, in the state's scalar"
    );
    static_assert(
        !(IsHybrid && Shared),
        "mixing a particle's modes gives each particle covariances of its own, so a hybrid per particle needs "
        "PerPointCovariance"
    );

public:

    using Scalar = Scalar_;                           ///< the scalar type
    using SpaceType = Space_;                         ///< the whole state space
    using PathType = Path_;                           ///< the path naming the sampled block
    using PinnedSpace = typename Split::PinnedSpace;  ///< the sampled block's space
    using RestSpace = typename Split::RestSpace;      ///< the rest's space, in order
    using CovarianceType = Covariance_;               ///< SharedCovariance or PerPointCovariance
    using ConditionalType = Params<Component_>;  ///< a particle's distribution over the rest: a %Gaussian or a hybrid
    using allocator_type = Allocator_;           ///< the allocator, re-bound per buffer
    using ParticlesType = ParticleSet<PinnedSpace, Scalar_, Allocator_>;  ///< the particles of the block
    using WeightArrayType = typename ParticlesType::WeightArrayType;      ///< one entry per particle
    using LogWeightsType = typename ParticlesType::LogWeightsType;        ///< the particles' log-weights map
    using MeansType = SampleSet<RestSpace, Scalar_>;                      ///< means over the rest, one per column
    using DensityType =
        ParticleGaussian<Space_, Path_, Covariance_, Scalar_, Allocator_, Component_>;  ///< what toDensity() returns

    static constexpr bool IsParams = true;  ///< parameters, which views check

    /** \a particles of the block in \a space, each with \a conditional as its %Gaussian, or its
     * hybrid, over the rest.
     */
    ParticleGaussianParams(Space_ space, ParticlesType particles, ConditionalType const & conditional, Covariance_ = {})
        : m_space{std::move(space)}
        , m_split{m_space}
        , m_particles{std::move(particles)}
        , m_kept(IndexAllocator(m_particles.get_allocator()))
        , m_conditionals{conditional, m_particles.size(), m_particles.get_allocator()}
    {
        check();
        EMRYS_FILTER_ASSERT(conditional.space() == restSpace() && "the Gaussian must be over the rest of the state");
    }

    /** \a particles of the block in \a space, particle \a i with the mean \a means.col(i) over the
     * rest and the covariance \a cov, which the particles share or each start from. The means are
     * copied into the state's storage, so a map over a buffer of one's own passes them without
     * allocating.
     */
    template <typename MeansDerived>
    ParticleGaussianParams(
        Space_ space,
        ParticlesType particles,
        Eigen::MatrixBase<MeansDerived> const & means,
        typename Moments<RestSpace, Scalar_>::MatrixType const & cov,
        Covariance_ = {}
    )
        requires(!IsHybrid)
        : m_space{std::move(space)}
        , m_split{m_space}
        , m_particles{std::move(particles)}
        , m_kept(IndexAllocator(m_particles.get_allocator()))
        , m_conditionals{
              ConditionalType{m_split.restSpace(), means.col(0), cov},
              m_particles.size(),
              m_particles.get_allocator()
          }
    {
        check();
        EMRYS_FILTER_ASSERT(
            means.cols() == size() && means.rows() == valueDimension(restSpace())
            && "give one mean over the rest of the state per particle"
        );
        if constexpr (Shared)
        {
            m_conditionals.means() = means;
        }
        else
        {
            for (Eigen::Index i = 1; i < size(); ++i)
            {
                m_conditionals.conditional(i).setMean(means.col(i));
            }
        }
    }

    /** \a particles of the block in \a space, particle \a i with the %Gaussian, or the hybrid,
     * \a conditionals[i] over the rest, from any contiguous storage, such as a vector or an array.
     */
    ParticleGaussianParams(Space_ space, ParticlesType particles, std::span<ConditionalType const> conditionals)
        requires(!Shared)
        : ParticleGaussianParams{std::move(space), std::move(particles), conditionals.front()}
    {
        EMRYS_FILTER_ASSERT(Eigen::Index(conditionals.size()) == size() && "give one Gaussian per particle");
        for (Eigen::Index i = 0; i < size(); ++i)
        {
            EMRYS_FILTER_ASSERT(
                conditionals[std::size_t(i)].space() == restSpace()
                && "the Gaussians must be over the rest of the state"
            );
            m_conditionals.conditional(i) = conditionals[std::size_t(i)];
        }
    }

    /** \returns the whole state space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the sampled block's space, with its parameters */
    [[nodiscard]] PinnedSpace const &
    pinnedSpace() const
    {
        return m_split.pinnedSpace();
    }

    /** \returns the space of the rest of the state, in order */
    [[nodiscard]] RestSpace const &
    restSpace() const
    {
        return m_split.restSpace();
    }

    /** \returns the rows of the sampled block in the whole state */
    [[nodiscard]] BlockRows
    pinnedRows() const
    {
        return m_split.pinnedRows();
    }

    /** \returns the rows of the rest of the state in the whole state, in order */
    [[nodiscard]] auto const &
    restIndices() const
    {
        return m_split.restIndices();
    }

    /** \returns the rows of the sampled block in the whole state's differences */
    [[nodiscard]] BlockRows
    pinnedDifferenceRows() const
    {
        return m_split.pinnedDifferenceRows();
    }

    /** \returns the rows of the rest of the state in the whole state's differences, in order */
    [[nodiscard]] auto const &
    restDifferenceIndices() const
    {
        return m_split.restDifferenceIndices();
    }

    /** \returns the number of particles */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return m_particles.size();
    }

    /** \returns the allocator the particles and Gaussians come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_particles.get_allocator();
    }

    /** Reserves room for \a capacity particles and their Gaussians, and makes holding more a
     * precondition violation. It must be at least size().
     */
    void
    setCapacity(Eigen::Index capacity)
    {
        m_stamp.touchShape();
        m_particles.setCapacity(capacity);
        m_conditionals.reserve(capacity);
        m_kept.reserve(std::size_t(capacity));
    }

    /** \returns the particles of the sampled block, with their weights */
    [[nodiscard]] ParticlesType const &
    particles() const
    {
        return m_particles;
    }

    /** \returns particle \a i's value of the sampled block */
    [[nodiscard]] auto
    particle(Eigen::Index i) const
    {
        return m_particles.state(i);
    }

    /** \returns the particles' log-weights, not necessarily normalized */
    [[nodiscard]] LogWeightsType
    logWeights() const
    {
        return m_particles.logWeights();
    }

    /** \returns particle \a i's log-weight */
    [[nodiscard]] Scalar
    logWeight(Eigen::Index i) const
    {
        return m_particles.logWeight(i);
    }

    /** \returns the log of the weights' sum, or a WeightError if they are invalid */
    [[nodiscard]] std::expected<Scalar, WeightError>
    logTotalWeight() const
    {
        return m_particles.logTotalWeight();
    }

    /** logTotalWeight() under \a policy, a reduction over the particles (see \ref Execution). */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<Scalar, WeightError>
    logTotalWeight(P && policy) const
    {
        return m_particles.logTotalWeight(std::forward<P>(policy));
    }

    /** \returns the effective sample size, or a WeightError if the weights are invalid */
    [[nodiscard]] std::expected<Scalar, WeightError>
    ess() const
    {
        return m_particles.ess();
    }

    /** ess() under \a policy, reductions over the particles (see \ref Execution). */
    template <ExecutionPolicy P>
    [[nodiscard]] std::expected<Scalar, WeightError>
    ess(P && policy) const
    {
        return m_particles.ess(std::forward<P>(policy));
    }

    /** \returns the index of the heaviest particle; ties go to the lowest */
    [[nodiscard]] Eigen::Index
    mostProbable() const
    {
        Eigen::Index best = 0;
        logWeights().maxCoeff(&best);
        return best;
    }

    /** \returns particle \a i's distribution over the rest of the state: its %Gaussian, or its hybrid */
    [[nodiscard]] decltype(auto)
    conditional(Eigen::Index i) const
    {
        EMRYS_FILTER_ASSERT(i >= 0 && i < size() && "the particle is out of range");
        return m_conditionals.conditional(i);
    }

    /** \returns the probability of each mode, summed over the particles,
     * \f$ P(m) = \sum_i \hat w_i \, P(m \mid i) \f$ with the weights normalized; the weights must
     * define a distribution
     */
    [[nodiscard]] auto
    modes() const
        requires IsHybrid
    {
        return internal::summedModes(m_conditionals, logWeights());
    }

    /** \returns the covariance every particle's %Gaussian shares */
    [[nodiscard]] typename Moments<RestSpace, Scalar_>::MatrixType const &
    sharedCovariance() const
        requires Shared
    {
        return m_conditionals.sharedCovariance();
    }

    /** Checks the weights and factorizes each particle's %Gaussian, the one they share once, or
     * checks each particle's hybrid, so that the density's collapse reads a distribution. It is \c const because it
     * writes only the Gaussians' cached factorizations (see \ref Distributions).
     *
     * \returns nothing on success, the NumericError of weights that fail to define a distribution,
     * or that of the first %Gaussian that fails to factorize
     */
    [[nodiscard]] std::expected<void, NumericError>
    factorize() const
    {
        if (auto const total = m_particles.logTotalWeight(); !total)
        {
            return std::unexpected{total.error()};
        }
        return m_conditionals.factorize();
    }

    /** Checks the state (factorize()) and views it.
     *
     * \returns a ParticleGaussian that checks this object, or the NumericError from factorize()
     */
    [[nodiscard]] std::expected<DensityType, NumericError>
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

    /** Resamples the particles to \a newSize in proportion to their weights with \a resampler, as
     * ParticleSet::resample() does, each new particle taking its ancestor's %Gaussian.
     *
     * \returns nothing on success, or a WeightError (with the state unchanged) if the weights
     * cannot be normalized
     */
    template <std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
        requires Resampler<R, Scalar, G>
    [[nodiscard]] std::expected<void, WeightError>
    resample(G & rng, Eigen::Index newSize, R const & resampler = {})
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

    /** resample() under \a policy, as ParticleSet::resample() runs under one, with the ancestors'
     * Gaussians gathered as a for-each (see \ref Execution).
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
        requires Resampler<R, Scalar, G>
    [[nodiscard]] std::expected<void, WeightError>
    resample(P && policy, G & rng, Eigen::Index newSize, R const & resampler = {})
    {
        auto const resampled = m_particles.resample(policy, rng, newSize, resampler);
        if (!resampled)
        {
            return resampled;
        }
        m_stamp.touchShape();
        m_conditionals.select(std::forward<P>(policy), m_particles.ancestors());
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

    /** Removes every particle whose normalized log-weight is below \a logThreshold, with its
     * %Gaussian, as ParticleSet::pruneBelow() does, and normalizes the weights of the rest.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneBelow(Scalar logThreshold)
    {
        return pruneWith(internal::PruneBelowRule<Scalar>{logThreshold});
    }

    /** Removes the lightest particles, with their Gaussians, as many as can be while their
     * combined normalized weight stays below \f$ e^{\mathrm{logFraction}} \f$, as
     * ParticleSet::pruneTail() does, and normalizes the weights of the rest.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneTail(Scalar logFraction)
    {
        return pruneWith(internal::PruneTailRule<Scalar>{logFraction});
    }

    /** Removes the lightest particles, with their Gaussians, until at most \a maxSize, at least
     * one, remain, as ParticleSet::pruneToSize() does, and normalizes the weights of the rest.
     *
     * \returns the log of the removed fraction of the total weight, -inf if nothing was removed, or,
     *          with nothing changed, the WeightError of logTotalWeight()
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneToSize(Eigen::Index maxSize)
    {
        return pruneWith(internal::PruneToSizeRule{maxSize});
    }

    /** Replaces the particles' values of the sampled block with the columns of \a states, one per
     * particle, keeping their weights and Gaussians, as Roughen does.
     */
    template <typename Derived>
    void
    setParticleStates(Eigen::MatrixBase<Derived> const & states)
    {
        EMRYS_FILTER_ASSERT(states.cols() == size() && "give one value per particle");
        m_stamp.touchValues();
        m_particles.setStates(states);
    }

private:

    /** The filter reweights and replaces the Gaussians. */
    template <Space, typename, typename, typename, typename, RealScalar, typename, typename>
    friend class RbParticleFilter;

    friend DensityType;  ///< the density reads the collapse

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** The particles' distributions' store. */
    using ConditionalsType = internal::Conditionals<RestSpace, Covariance_, Scalar_, Allocator_, ConditionalType>;

    /** Allocates the positions of the particles a prune keeps. */
    using IndexAllocator = internal::ReBound<Allocator_, Eigen::Index>;

    /** The body of pruneBelow(), pruneTail(), and pruneToSize(): prunes the particles by \a rule
     * and keeps the Gaussians of those kept.
     */
    template <typename Rule>
    [[nodiscard]] std::expected<Scalar, WeightError>
    pruneWith(Rule const & rule)
    {
        m_kept.clear();
        auto const removed = internal::ParticleSetAccess::prune(m_particles, rule, [&](Eigen::Index i) {
            m_kept.push_back(i);
        });
        if (removed)
        {
            m_stamp.touchShape();
            m_conditionals.keep(std::span<Eigen::Index const>{m_kept.data(), m_kept.size()});
        }
        return removed;
    }

    /** \returns the particles as the points of a Rao-Blackwellized state, for collapseGivenPoints() */
    [[nodiscard]] internal::ParticlePoints<Space_, Split, ParticlesType>
    points() const
    {
        return {m_space, m_split, m_particles};
    }

    /** \returns the whole state as one %Gaussian, moment-matched over the particles: their spread
     * for the sampled block, and their Gaussians, each hybrid collapsed over its modes first, for the
     * rest. The weights must define a distribution and the Gaussians be factorized, as a density's
     * check makes sure.
     */
    [[nodiscard]] GaussianParams<Space_, Scalar>
    collapseParticles() const
    {
        if constexpr (IsHybrid)
        {
            return internal::collapseGivenHybrids(points(), m_conditionals, logWeights(), get_allocator());
        }
        else
        {
            return internal::collapseGivenPoints(points(), m_conditionals, logWeights());
        }
    }

    /** A programmer error unless there is a particle and the particles fit the block. */
    void
    check() const
    {
        EMRYS_FILTER_ASSERT(!m_particles.empty() && "a Rao-Blackwellized state needs a particle");
        EMRYS_FILTER_ASSERT(m_particles.valueDimension() == pinnedRows().size && "the particles do not fit the block");
    }

    /** Multiplies the weights by \a logLikelihoods, one per particle, normalizes them, and swaps
     * \a posterior in as the particles' Gaussians. If every particle has likelihood zero, nothing
     * changes. The prior weights must define a distribution, as RbParticleFilter checks before an
     * update (decisions.md [update-outcomes]); weights that don't fail an assertion, and where it
     * returns change nothing. The totals are reductions under \a policy; under Sequential it
     * allocates nothing.
     *
     * \returns the log-evidence, \f$ \log \sum_i \hat w_i \, p(z \mid i) \f$ with normalized prior
     * weights, and whether anything changed
     */
    template <ExecutionPolicy P, typename Derived>
    [[nodiscard]] std::pair<Scalar, bool>
    reweight(P && policy, Eigen::ArrayBase<Derived> const & logLikelihoods, ConditionalsType & posterior)
    {
        return reweightThen(std::forward<P>(policy), logLikelihoods, [&] {
            m_conditionals.swap(posterior);
        });
    }

    /** Multiplies the weights by \a logLikelihoods and normalizes them, as reweight(policy,
     * logLikelihoods, posterior) does, and keeps the particles' Gaussians where they are: for an
     * update that changed none (decisions.md [update-outcomes]).
     *
     * \returns the log-evidence and whether anything changed
     */
    template <ExecutionPolicy P, typename Derived>
    [[nodiscard]] std::pair<Scalar, bool>
    reweight(P && policy, Eigen::ArrayBase<Derived> const & logLikelihoods)
    {
        return reweightThen(std::forward<P>(policy), logLikelihoods, [] {});
    }

    /** The body of both forms of reweight(): \a place puts the posterior Gaussians in, once the
     * evidence is known to be finite.
     */
    template <ExecutionPolicy P, typename Derived, typename Place>
    [[nodiscard]] std::pair<Scalar, bool>
    reweightThen(P && policy, Eigen::ArrayBase<Derived> const & logLikelihoods, Place const & place)
    {
        EMRYS_FILTER_ASSERT(logLikelihoods.size() == size() && "give one log-likelihood per particle");
        auto const logTotal = m_particles.logTotalWeight(policy);
        EMRYS_FILTER_ASSERT(logTotal && "the particles' weights must be valid");
        if (!logTotal)
        {
            return {Eigen::NumTraits<Scalar>::quiet_NaN(), false};
        }
        auto const logWeights = m_particles.logWeights() - *logTotal + logLikelihoods.derived();
        // A NaN log-likelihood at any particle makes the evidence NaN.
        Scalar const logLikelihood = logSumExp(std::forward<P>(policy), logWeights);
        EMRYS_FILTER_ASSERT(!isNaN(logLikelihood) && "the measurement's log-likelihood must not be NaN");
        if (logLikelihood == -Eigen::NumTraits<Scalar>::infinity())
        {
            return {logLikelihood, false};
        }
        m_stamp.touchShape();
        internal::ParticleSetAccess::setNormalizedLogWeights(m_particles, logWeights - logLikelihood);
        place();
        return {logLikelihood, true};
    }

    Space_ m_space;                                     ///< the whole state space
    Split m_split;                                      ///< the state divided into block and rest
    ParticlesType m_particles;                          ///< the particles of the block, with their weights
    internal::Buffer<Eigen::Index, Allocator_> m_kept;  ///< scratch: the positions of the particles a prune keeps

    ConditionalsType m_conditionals;  ///< each particle's %Gaussian over the rest
    internal::Stamp m_stamp;          ///< counts changes, for the density's check [view-stamp]
};

/** \ingroup Distributions
 * \brief The density of a Rao-Blackwellized particle filter's state, a checked view of its
 * ParticleGaussianParams (decisions.md [params-and-density], [view-stamp]):
 * \f$ p(x) = \sum_i w_i \, \delta(x_p - p_i) \, \mathcal{N}(x_r; \mu_i, \Sigma_i) \f$, or with a
 * hybrid per particle, a mixture over its modes in place of each %Gaussian.
 *
 * ParticleGaussianParams::toDensity() makes one after checking the weights and factorizing each
 * particle's %Gaussian, or checking its hybrid, so the moments and the collapse below read a
 * distribution.
 *
 * \code
 * auto const density = filter.state().toDensity();
 * density->collapsed();   // the whole state as one GaussianParams
 * density->moments();     // its mean and covariance
 * \endcode
 *
 * \sa class ParticleGaussianParams, RbParticleFilter
 */
template <
    Space Space_,
    typename Path_,
    typename Covariance_,
    RealScalar Scalar_,
    typename Allocator_,
    typename Component_>
class ParticleGaussian
{
public:

    using ParamsType =
        ParticleGaussianParams<Space_, Path_, Covariance_, Scalar_, Allocator_, Component_>;  ///< the parameters
    using Scalar = Scalar_;                                                                   ///< the scalar type
    using SpaceType = Space_;                                                                 ///< the whole state space
    using PinnedSpace = typename ParamsType::PinnedSpace;          ///< the sampled block's space
    using RestSpace = typename ParamsType::RestSpace;              ///< the rest's space, in order
    using ConditionalType = typename ParamsType::ConditionalType;  ///< a particle's distribution over the rest
    using ParticlesType = typename ParamsType::ParticlesType;      ///< the particles of the block
    using MomentsType = Moments<Space_, Scalar_>;                  ///< a mean and covariance of the whole state
    using allocator_type = Allocator_;                             ///< the parameters' allocator

    static constexpr bool IsParams = false;  ///< a view: it checks its parameters

    /** \returns the whole state space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return source().space();
    }

    /** \returns the number of particles */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return source().size();
    }

    /** \returns the particles of the sampled block, with their weights, which define a distribution */
    [[nodiscard]] ParticlesType const &
    particles() const
    {
        return source().particles();
    }

    /** \returns particle \a i's distribution over the rest of the state */
    [[nodiscard]] decltype(auto)
    conditional(Eigen::Index i) const
    {
        return source().conditional(i);
    }

    /** \returns the whole state as one %Gaussian, moment-matched over the particles: their spread
     * for the sampled block, and their Gaussians, or each particle's hybrid collapsed over its modes,
     * for the rest
     */
    [[nodiscard]] GaussianParams<Space_, Scalar_>
    collapsed() const
    {
        return source().collapseParticles();
    }

    /** \returns the mean and covariance of the whole state, those of collapsed() */
    [[nodiscard]] MomentsType
    moments() const
    {
        return collapsed().moments();
    }

private:

    friend ParamsType;  ///< the parameters construct views

    /** Views \a source, whose weights and Gaussians are checked, and records its stamp; only
     * ParticleGaussianParams::toDensity() calls this.
     */
    explicit ParticleGaussian(ParamsType const & source)
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
 * \returns the ParticleGaussianParams with \a particles of the block named by the path \a P of a state
 * in \a space, each with \a conditional as its %Gaussian over the rest, its covariance shared or
 * per particle as \a Covariance says
 */
template <typename P, Space S, typename Scalar, typename Allocator, typename Covariance = SharedCovariance>
[[nodiscard]] auto
particleGaussian(
    S space,
    ParticleSet<typename internal::BlockSplit<S, P>::PinnedSpace, Scalar, Allocator> particles,
    GaussianParams<typename internal::BlockSplit<S, P>::RestSpace, Scalar> const & conditional,
    Covariance covariance = {}
)
{
    return ParticleGaussianParams<S, P, Covariance, Scalar, Allocator>{
        std::move(space),
        std::move(particles),
        conditional,
        covariance
    };
}

/** \ingroup Distributions
 * \returns the ParticleGaussianParams with \a particles of the block named by the path \a P of a state
 * in \a space, each with the hybrid \a conditional over the rest, an IMM's state, and so with
 * PerPointCovariance
 */
template <
    typename P,
    Space S,
    typename Scalar,
    typename Allocator,
    typename Component,
    typename ModeSpace,
    typename HybridAllocator>
[[nodiscard]] auto
particleGaussian(
    S space,
    ParticleSet<typename internal::BlockSplit<S, P>::PinnedSpace, Scalar, Allocator> particles,
    HybridParams<Component, ModeSpace, HybridAllocator> const & conditional
)
{
    return ParticleGaussianParams<
        S,
        P,
        PerPointCovariance,
        Scalar,
        Allocator,
        Hybrid<Component, ModeSpace, HybridAllocator>>{std::move(space), std::move(particles), conditional};
}

/** \ingroup Distributions
 * \returns \a count equally weighted particles of the block named by the path \a P drawn from the
 * %Gaussian \a joint over the whole state with \a rng, each with the %Gaussian of the rest given its
 * value: the Rao-Blackwellized form of \a joint. Those Gaussians share one covariance, which a
 * PerPointCovariance state copies to each. The state allocates through \a allocator.
 */
template <
    typename P,
    typename Covariance = SharedCovariance,
    typename S,
    typename Scalar,
    std::uniform_random_bit_generator G,
    typename Allocator = DefaultAllocator>
[[nodiscard]] auto
drawParticleGaussian(
    GaussianParams<S, Scalar> const & joint,
    Eigen::Index count,
    G & rng,
    Covariance = {},
    Allocator const & allocator = {}
)
{
    using Result = ParticleGaussianParams<S, P, Covariance, Scalar, Allocator>;
    using PinnedSamples = SampleSet<typename Result::PinnedSpace, Scalar>;
    using PinnedNormals = SampleSet<DifferenceSpace<typename Result::PinnedSpace>, Scalar>;
    EMRYS_FILTER_ASSERT(count > 0 && "draw at least one particle");
    auto const split = internal::BlockSplit<S, P>{joint.space()};
    auto const conditioning = internal::pinnedConditioning(split, joint.covarianceLower());
    PinnedNormals const z = internal::standardNormal<PinnedNormals>(split.pinnedDifferenceRows().size, count, rng);
    auto pinned = PinnedSamples{split.pinnedRows().size, count};
    auto means = typename Result::MeansType{split.restIndices().size(), count};
    for (Eigen::Index i = 0; i < count; ++i)
    {
        internal::drawPinned(split, conditioning, joint.mean(), z.col(i), pinned.col(i), means.col(i));
    }
    auto particles = typename Result::ParticlesType{split.pinnedSpace(), pinned, allocator};
    return Result{joint.space(), std::move(particles), means, conditioning.cov, Covariance{}};
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a state, a Rao-Blackwellized particle filter's: its particle count, dimension, and
 * effective sample size, and the mean and covariance of its collapse; or the error of the
 * weights or of a %Gaussian that fails to factorize in their place.
 */
template <
    typename Space_,
    typename Path_,
    typename Covariance_,
    typename Scalar_,
    typename Allocator_,
    typename Component_>
void
describe(
    std::ostream & stream,
    ParticleGaussianParams<Space_, Path_, Covariance_, Scalar_, Allocator_, Component_> const & state,
    Indent indent
)
{
    Eigen::Index const dimension = valueDimension(state.space());
    stream << indent << "Rao-Blackwellized particle set of ";
    printCount(stream, state.size(), "particle");
    printDimension(stream, dimension, differenceDimension(state.space()));
    auto const ess = state.ess();
    auto const density = state.toDensity();
    if (ess)
    {
        stream << ", effective sample size " << *ess;
    }
    if (density)
    {
        printMoments(stream, indent.deeper(), density->moments());
    }
    else
    {
        stream << '\n';
        describe(stream, density.error(), indent.deeper());
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
