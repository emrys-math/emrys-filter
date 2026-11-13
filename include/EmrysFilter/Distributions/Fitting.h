#pragma once

/** \file
 * Fitting a %Gaussian mixture by expectation-maximization: fitMixture() over particles, over a
 * mixture's components (hierarchical EM, a mixture reduction), or over particles seeded from a
 * mixture, with FitOptions and the MixtureFit result. Read Upkeep.h next.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <expected>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Core/Print.h>

namespace Emrys
{

/** \ingroup Distributions
 * Reasons fitMixture() can fail.
 */
enum class FitError
{
    Degenerate,      ///< the source has no spread in some dimension, e.g. every particle has the same state
    InvalidWeights,  ///< the particles' weights cannot be normalized (see ParticleSet::logTotalWeight())
};

/** \ingroup Distributions
 * Options for fitMixture().
 */
template <typename Scalar>
struct FitOptions
{
    /** Maximum number of EM iterations (E-steps). */
    int maxIterations = 100;

    /** Stop when an iteration improves the weighted log-likelihood by at most this. */
    Scalar tolerance = Scalar(1e-8);

    /** Covariance floor, relative to the source's variance in each dimension: every
     * component covariance gets \f$ \epsilon \Sigma_{ii} \f$ added to its \f$ i \f$-th variance.
     * Scaling each dimension by its own variance keeps the floor consistent when the
     * dimensions have different units. Keeps components that capture duplicate particles
     * from collapsing.
     */
    Scalar regularization = Scalar(1e-6);

    /** Components whose normalized log-weight falls below this are dropped. */
    Scalar minLogWeight = Scalar(-20);
};

/** \ingroup Distributions
 * Result of fitMixture(), in \a Space, with the source's allocator.
 */
template <typename Space, typename Scalar = double, typename Allocator = DefaultAllocator>
struct MixtureFit
{
    GaussianMixtureParams<Space, Scalar, Allocator> mixture;  ///< fitted mixture, weights normalized to one
    Scalar logLikelihood;  ///< weighted log-likelihood of the source under \c mixture
    int iterations;        ///< E-steps performed
    bool converged;        ///< whether \c tolerance was reached before \c maxIterations
};

namespace internal
{

/** Weighted points to fit, each with an optional covariance (a particle has none). */
template <typename Scalar, typename Space>
struct FitSource
{
    using MatrixType = typename Moments<Space, Scalar>::MatrixType;  ///< a covariance's type
    SampleSet<Space, Scalar> points;                                 ///< the points, one per column
    Eigen::Array<Scalar, Eigen::Dynamic, 1> weights;                 ///< normalized linear weights
    std::vector<MatrixType> covs;                                    ///< per-point covariances; empty for particles
    std::vector<MatrixType> covFactors;                              ///< lower Cholesky factors of \c covs
    Space space;                                                     ///< the points' space
};

/** \returns \a particles as a FitSource with no covariances, or FitError::InvalidWeights */
template <typename Scalar, typename Space, typename Allocator>
[[nodiscard]] std::expected<FitSource<Scalar, Space>, FitError>
fitSource(ParticleSet<Space, Scalar, Allocator> const & particles)
{
    auto weights = particles.normalizedWeights();
    if (!weights)
    {
        return std::unexpected{FitError::InvalidWeights};
    }
    return FitSource<Scalar, Space>{particles.states(), std::move(*weights), {}, {}, particles.space()};
}

/** \returns \a mixture's components as a FitSource: their means as points, with their
 * covariances and Cholesky factors
 */
template <typename Scalar, typename Space, typename Allocator>
[[nodiscard]] FitSource<Scalar, Space>
fitSource(GaussianMixture<Space, Scalar, Allocator> const & mixture)
{
    using MatrixType = typename FitSource<Scalar, Space>::MatrixType;
    auto source = FitSource<Scalar, Space>{
        SampleSet<Space, Scalar>{mixture.valueDimension(), mixture.size()},
        Eigen::Array<Scalar, Eigen::Dynamic, 1>{mixture.size()},
        {},
        {},
        mixture.space()
    };
    source.covs.reserve(std::size_t(mixture.size()));
    source.covFactors.reserve(std::size_t(mixture.size()));
    Eigen::Index i = 0;
    for (auto const & entry : mixture)
    {
        auto const gaussian = entry.component.toDensity();
        source.points.col(i) = gaussian->mean();
        source.weights(i) = Eigen::numext::exp(entry.logWeight - mixture.logTotalWeight());
        source.covs.push_back(MatrixType{entry.component.covarianceLower().template selfadjointView<Eigen::Lower>()});
        source.covFactors.push_back(MatrixType{gaussian->llt().matrixL()});
        ++i;
    }
    return source;
}

/** Weighted EM for a %Gaussian mixture over weighted points that may carry covariances.
 *
 * For points with covariances \f$ S_i \f$ this is hierarchical EM: the E-step uses the
 * expected log-likelihood \f$ \log \mathcal{N}(x_i; \mu_k, \Sigma_k) - \tfrac12 \mathrm{tr}(\Sigma_k^{-1} S_i) \f$,
 * and the M-step moment-matches. For particles (\f$ S_i = 0 \f$) it is ordinary weighted EM.
 * Residuals are normalized by the source's space and means are tangent-space means.
 *
 * The EM algorithm is \ref Dempster1977 "Dempster, Laird, and Rubin (1977)"; its %Gaussian mixture
 * form is the standard one
 * ([expectation-maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture)).
 * Fitting a mixture to a mixture by treating each component as a point with a covariance is the
 * hierarchical EM of \ref Vasconcelos1999 "Vasconcelos and Lippman (1999)"; the covariance term
 * in the E-step is their equation for the expected log-likelihood of a virtual sample. Seeding
 * is the k-means++ of \ref Arthur2007 "Arthur and Vassilvitskii (2007)", with the weights as
 * the points' masses.
 */
template <typename Scalar, typename Space, typename Allocator>
class MixtureFitter
{
public:

    using MomentsType = Moments<Space, Scalar>;                                     ///< a mean and covariance together
    using VectorType = typename MomentsType::VectorType;                            ///< a value
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;        ///< a difference
    using MatrixType = typename MomentsType::MatrixType;                            ///< a covariance
    using Points = SampleSet<Space, Scalar>;                                        ///< points as columns
    using Deviations = SampleSet<DifferenceSpace<Space>, Scalar>;                   ///< differences as columns
    using Weights = Eigen::Array<Scalar, Eigen::Dynamic, 1>;                        ///< one weight per point
    using Responsibilities = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;  ///< components by points
    using Source = FitSource<Scalar, Space>;                                        ///< what is fitted
    using Result = MixtureFit<Space, Scalar, Allocator>;                            ///< what run() returns

    /** Fits \a source with \a options, building the result's mixture from \a allocator. */
    MixtureFitter(Source const & source, FitOptions<Scalar> const & options, Allocator const & allocator)
        : m_source{source}
        , m_options{options}
        , m_allocator{allocator}
    {
        EMRYS_FILTER_ASSERT(options.maxIterations >= 1);
    }

    /** Seeds up to \a k components by weighted k-means++ over \a candidates, then runs EM. */
    template <std::uniform_random_bit_generator G>
    [[nodiscard]] std::expected<Result, FitError>
    run(Points const & candidates, Weights const & candidateWeights, Eigen::Index k, G & rng)
    {
        return run(execution::sequential, candidates, candidateWeights, k, rng);
    }

    /** run() under \a policy: the E-step is a for-each over chunks of the source points and then
     * over the points, and the M-step's moments are reductions over them. The seeding is
     * sequential.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G>
    [[nodiscard]] std::expected<Result, FitError>
    run(P && policy, Points const & candidates, Weights const & candidateWeights, Eigen::Index k, G & rng)
    {
        EMRYS_FILTER_ASSERT(k >= 1);
        if (!prepareWhitening())
        {
            return std::unexpected{FitError::Degenerate};
        }
        Responsibilities responsibilities = initialAssignment(seed(candidates, candidateWeights, k, rng));
        mStep(policy, responsibilities);

        Scalar previous = -Eigen::NumTraits<Scalar>::infinity();
        for (int iteration = 1;; ++iteration)
        {
            auto const logLikelihood = eStep(policy, responsibilities);
            if (!logLikelihood)
            {
                return std::unexpected{logLikelihood.error()};
            }
            bool const converged = *logLikelihood - previous <= m_options.tolerance;
            if (converged || iteration >= m_options.maxIterations)
            {
                return result(*logLikelihood, iteration, converged);
            }
            previous = *logLikelihood;
            mStep(policy, responsibilities);
        }
    }

private:

    /** Factorizes the source's overall covariance, which defines the seeding metric and the floor. */
    [[nodiscard]] bool
    prepareWhitening()
    {
        MomentsType const moments =
            weightedMomentsIn<MomentsType>(m_source.space, m_source.points, m_source.weights.log(), Scalar(0));
        MatrixType total = moments.covarianceLower();
        for (std::size_t i = 0; i < m_source.covs.size(); ++i)
        {
            total += m_source.weights(Eigen::Index(i)) * m_source.covs[i];
        }
        DifferenceVectorType const variances = total.diagonal();
        // A dimension of identical points leaves only rounding noise, of order epsilon² times
        // its second moment; with no spread there is no scale to fit or to floor against. Where
        // differences have other coordinates than values, the mean has no part in that scale.
        DifferenceVectorType const secondMoments = [&] {
            if constexpr (hasEqualSizes<Space>)
            {
                return DifferenceVectorType{variances.array() + moments.mean().array().square()};
            }
            else
            {
                return variances;
            }
        }();
        if (!(variances.array() > Eigen::NumTraits<Scalar>::epsilon() * secondMoments.array()).all()
            || !allFinite(variances))
        {
            return false;
        }
        m_floor = m_options.regularization * variances;
        total.diagonal() += m_floor;
        m_whitening.compute(total);
        return m_whitening.info() == Eigen::Success;
    }

    /** \returns the squared Mahalanobis distance, under the source's overall covariance,
     * of each column of \a points from \a center, with differences normalized by the space.
     */
    template <typename Derived>
    [[nodiscard]] Weights
    distancesFrom(Points const & points, Eigen::MatrixBase<Derived> const & center) const
    {
        Deviations differences = Emrys::minus(m_source.space, points, center);
        solveInPlace(m_whitening.matrixL(), differences);
        return differences.colwise().squaredNorm().transpose().array();
    }

    /** Weighted k-means++: each seed is drawn with probability proportional to weight
     * times squared distance to the nearest seed so far, so points on top of a chosen seed
     * (stacked components, duplicate particles) are essentially never chosen again.
     * Stops early if no remaining candidate is at positive distance.
     *
     * \returns the seed points
     */
    template <typename G>
    [[nodiscard]] Points
    seed(Points const & candidates, Weights const & weights, Eigen::Index k, G & rng) const
    {
        Eigen::Index const m = candidates.cols();
        Weights distances = Weights::Constant(m, Eigen::NumTraits<Scalar>::infinity());
        auto probabilities = std::vector<Scalar>(weights.begin(), weights.end());
        auto chosen = std::vector<Eigen::Index>{};
        while (true)
        {
            auto pick = std::array<Eigen::Index, 1>{};
            Scalar const u = uniform01<Scalar>(rng);
            invertCumulative(std::span<Scalar const>{probabilities}, std::span<Eigen::Index>{pick}, [u](std::size_t) {
                return u;
            });
            chosen.push_back(pick[0]);
            if (Eigen::Index(chosen.size()) == k)
            {
                break;
            }
            distances = distances.min(distancesFrom(candidates, candidates.col(pick[0])));
            Scalar total = 0;
            for (Eigen::Index i = 0; i < m; ++i)
            {
                probabilities[std::size_t(i)] = weights(i) * distances(i);
                total += probabilities[std::size_t(i)];
            }
            if (!(total > Scalar(0)))
            {
                break;
            }
        }
        auto seeds = Points{candidates.rows(), Eigen::Index(chosen.size())};
        for (std::size_t s = 0; s < chosen.size(); ++s)
        {
            seeds.col(Eigen::Index(s)) = candidates.col(chosen[s]);
        }
        return seeds;
    }

    /** \returns hard responsibilities assigning each source point to its nearest seed. */
    [[nodiscard]] Responsibilities
    initialAssignment(Points const & seeds) const
    {
        Eigen::Index const n = m_source.points.cols();
        Weights nearestDistance = Weights::Constant(n, Eigen::NumTraits<Scalar>::infinity());
        Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> nearest = Eigen::Array<Eigen::Index, Eigen::Dynamic, 1>::Zero(n);
        for (Eigen::Index s = 0; s < seeds.cols(); ++s)
        {
            Weights const distances = distancesFrom(m_source.points, seeds.col(s));
            for (Eigen::Index i = 0; i < n; ++i)
            {
                if (distances(i) < nearestDistance(i))
                {
                    nearestDistance(i) = distances(i);
                    nearest(i) = s;
                }
            }
        }
        Responsibilities responsibilities = Responsibilities::Zero(seeds.cols(), n);
        for (Eigen::Index i = 0; i < n; ++i)
        {
            responsibilities(nearest(i), i) = Scalar(1);
        }
        return responsibilities;
    }

    /** Moment-matches each component to its responsibility-weighted points, dropping light ones,
     * with the moments and the points' covariances summed under \a policy.
     */
    template <typename P>
    void
    mStep(P && policy, Responsibilities const & responsibilities)
    {
        m_logWeights.clear();
        m_means.clear();
        m_covs.clear();
        for (Eigen::Index k = 0; k < responsibilities.rows(); ++k)
        {
            Weights const w = m_source.weights * responsibilities.row(k).transpose();
            Scalar const mass = w.sum();
            if (!(mass > Scalar(0)) || Eigen::numext::log(mass) < m_options.minLogWeight)
            {
                continue;
            }
            MomentsType moments = weightedMomentsIn<MomentsType>(
                policy,
                m_source.space,
                m_source.points,
                w.log(),
                Eigen::numext::log(mass)
            );
            // The points' own covariances, as reduceRanges() sums them: in order under Sequential.
            moments.covarianceLower() = reduceRanges(
                policy,
                Eigen::Index(m_source.covs.size()),
                [&](Eigen::Index begin, Eigen::Index length) {
                    MatrixType sum =
                        begin == 0
                            ? moments.covarianceLower()
                            : MatrixType::Zero(moments.covarianceLower().rows(), moments.covarianceLower().cols());
                    for (Eigen::Index i = begin; i < begin + length; ++i)
                    {
                        if (w(i) > Scalar(0))
                        {
                            sum += (w(i) / mass) * m_source.covs[std::size_t(i)];
                        }
                    }
                    return sum;
                },
                [](MatrixType a, MatrixType const & b) {
                    a += b;
                    return a;
                }
            );
            moments.covarianceLower().diagonal() += m_floor;
            m_logWeights.push_back(Eigen::numext::log(mass));
            m_means.push_back(moments.mean());
            m_covs.push_back(moments.covarianceLower());
        }
    }

    /** Computes responsibilities for the current components, dropping any whose covariance
     * does not factorize.
     *
     * \returns the weighted log-likelihood of the source, or FitError::Degenerate if no
     *          component survives
     */
    template <typename P>
    [[nodiscard]] std::expected<Scalar, FitError>
    eStep(P && policy, Responsibilities & responsibilities)
    {
        auto factors = std::vector<Eigen::LLT<MatrixType>>{};
        auto logDeterminants = std::vector<Scalar>{};
        factors.reserve(m_covs.size());
        logDeterminants.reserve(m_covs.size());
        std::size_t out = 0;
        for (std::size_t k = 0; k < m_covs.size(); ++k)
        {
            auto llt = Eigen::LLT<MatrixType>{m_covs[k]};
            Scalar const logDeterminant = Scalar(2) * llt.matrixLLT().diagonal().array().log().sum();
            if (llt.info() != Eigen::Success || !isFinite(logDeterminant))
            {
                continue;
            }
            m_logWeights[out] = m_logWeights[k];
            m_means[out] = m_means[k];
            m_covs[out] = m_covs[k];
            factors.push_back(std::move(llt));
            logDeterminants.push_back(logDeterminant);
            ++out;
        }
        m_logWeights.resize(out);
        m_means.resize(out);
        m_covs.resize(out);
        if (out == 0)
        {
            return std::unexpected{FitError::Degenerate};
        }

        Eigen::Index const n = m_source.points.cols();
        Scalar const log2Pi = Emrys::log2Pi<Scalar>();
        Scalar const dimension = Scalar(differenceDimension(m_source.space));
        auto logTerms = Responsibilities{Eigen::Index(out), n};
        Eigen::Index const withCovariance = Eigen::Index(m_source.covFactors.size());
        // Under Sequential, one chunk of every point.
        forEachChunk(policy, n, [&](Eigen::Index begin, Eigen::Index length) {
            for (std::size_t k = 0; k < out; ++k)
            {
                Deviations residuals =
                    Emrys::minus(m_source.space, m_source.points.middleCols(begin, length), m_means[k]);
                solveInPlace(factors[k].matrixL(), residuals);
                logTerms.row(Eigen::Index(k)).segment(begin, length) =
                    m_logWeights[k] - Scalar(0.5) * (dimension * log2Pi + logDeterminants[k])
                    - Scalar(0.5) * residuals.colwise().squaredNorm().array();
                for (Eigen::Index i = begin; i < std::min(begin + length, withCovariance); ++i)
                {
                    MatrixType solved = m_source.covFactors[std::size_t(i)];
                    solveInPlace(factors[k].matrixL(), solved);
                    logTerms(Eigen::Index(k), i) -= Scalar(0.5) * solved.squaredNorm();
                }
            }
        });

        responsibilities.resize(logTerms.rows(), n);
        auto contributions = Weights{n};
        forEachIndex(policy, n, [&](Eigen::Index i) {
            auto sum = LogSumExp<Scalar>{};
            for (Eigen::Index k = 0; k < logTerms.rows(); ++k)
            {
                sum.add(logTerms(k, i));
            }
            Scalar const logNormalizer = sum.value();
            // A point every component finds impossible (or with no weight) takes no part in the M-step.
            if (!isFinite(logNormalizer) || !(m_source.weights(i) > Scalar(0)))
            {
                responsibilities.col(i).setZero();
                contributions(i) = Scalar(0);
                return;
            }
            responsibilities.col(i) = (logTerms.col(i) - logNormalizer).exp();
            contributions(i) = m_source.weights(i) * logNormalizer;
        });
        // Summed in order, so that the test for convergence is Sequential's under every policy.
        Scalar logLikelihood = 0;
        for (Scalar const contribution : contributions)
        {
            logLikelihood += contribution;
        }
        return logLikelihood;
    }

    /** \returns the current components as a MixtureFit */
    [[nodiscard]] Result
    result(Scalar logLikelihood, int iterations, bool converged) const
    {
        auto mixture = GaussianMixtureParams<Space, Scalar, Allocator>{m_source.space, m_allocator};
        mixture.reserve(Eigen::Index(m_means.size()));
        for (std::size_t k = 0; k < m_means.size(); ++k)
        {
            mixture.add(m_logWeights[k], m_means[k], m_covs[k]);
        }
        return {std::move(mixture), logLikelihood, iterations, converged};
    }

    Source const & m_source;             ///< the points being fitted
    FitOptions<Scalar> m_options;        ///< the options
    Allocator m_allocator;               ///< for the result's mixture
    Eigen::LLT<MatrixType> m_whitening;  ///< the source's overall covariance, factorized, for distances
    DifferenceVectorType m_floor;        ///< the covariance floor added to each component's variances
    std::vector<Scalar> m_logWeights;    ///< the current components' log-weights
    std::vector<VectorType> m_means;     ///< the current components' means
    std::vector<MatrixType> m_covs;      ///< the current components' covariances
};

}  // namespace internal

/** \ingroup Distributions
 * \returns a mixture of at most \a k Gaussians fitted to \a particles by weighted EM, with the
 * particles' allocator.
 *
 * Seeds by weighted k-means++ over the particles, using the Mahalanobis distance under
 * the particles' covariance, so duplicate particles are not seeded twice. Fewer than
 * \a k components result if fewer distinct particles exist or components die (see
 * FitOptions::minLogWeight). The result is in the particles' space. Costs
 * O(iterations · size() · k · d²).
 *
 * \returns the fit; FitError::Degenerate if the particles have no spread in some
 *          dimension; or FitError::InvalidWeights if their weights cannot be normalized
 */
template <typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Space, Scalar, Allocator>, FitError>
fitMixture(
    ParticleSet<Space, Scalar, Allocator> const & particles,
    Eigen::Index k,
    G & rng,
    FitOptions<Scalar> const & options = {}
)
{
    auto const source = internal::fitSource(particles);
    if (!source)
    {
        return std::unexpected{source.error()};
    }
    auto fitter = internal::MixtureFitter<Scalar, Space, Allocator>{*source, options, particles.get_allocator()};
    return fitter.run(source->points, source->weights, k, rng);
}

/** \ingroup Distributions
 * \returns a mixture of at most \a k Gaussians fitted to \a mixture by hierarchical EM, with the
 * mixture's allocator.
 *
 * Reduces the mixture without sampling: each component is a weighted point carrying its
 * covariance. Seeds by weighted k-means++ over the component means, so stacked
 * components are not seeded twice. The result is in the mixture's space. Costs
 * O(iterations · size() · k · d³), with no pairwise pass over the components.
 *
 * \returns the fit, or FitError::Degenerate if the mixture has no spread in some dimension
 */
template <typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Space, Scalar, Allocator>, FitError>
fitMixture(
    GaussianMixture<Space, Scalar, Allocator> const & mixture,
    Eigen::Index k,
    G & rng,
    FitOptions<Scalar> const & options = {}
)
{
    auto const source = internal::fitSource(mixture);
    auto fitter = internal::MixtureFitter<Scalar, Space, Allocator>{source, options, mixture.get_allocator()};
    return fitter.run(source.points, source.weights, k, rng);
}

/** \ingroup Distributions
 * \returns a mixture of at most \a k Gaussians fitted to \a particles by weighted EM,
 * seeded by weighted k-means++ over the component means of \a seedsFrom, which must be
 * in the same space.
 *
 * Useful when the particles were drawn from \a seedsFrom: its components summarize the
 * particles and are far fewer, so seeding is cheaper and starts EM closer to the answer.
 * Fails as fitMixture(ParticleSet const &, Eigen::Index, G &, FitOptions const &) does.
 */
template <
    typename Scalar,
    typename Space,
    typename Allocator,
    typename SeedAllocator,
    std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Space, Scalar, Allocator>, FitError>
fitMixture(
    ParticleSet<Space, Scalar, Allocator> const & particles,
    GaussianMixture<Space, Scalar, SeedAllocator> const & seedsFrom,
    Eigen::Index k,
    G & rng,
    FitOptions<Scalar> const & options = {}
)
{
    EMRYS_FILTER_ASSERT(particles.space() == seedsFrom.space() && "the particles and seeds must share a space");
    auto const source = internal::fitSource(particles);
    if (!source)
    {
        return std::unexpected{source.error()};
    }
    auto const seeds = internal::fitSource(seedsFrom);
    auto fitter = internal::MixtureFitter<Scalar, Space, Allocator>{*source, options, particles.get_allocator()};
    return fitter.run(seeds.points, seeds.weights, k, rng);
}

/** \ingroup Distributions
 * fitMixture() of \a particles under \a policy: each E-step is a for-each over the particles,
 * and each M-step's moments are reductions over them, which match Sequential's to rounding (see
 * \ref Execution). The seeding is sequential, from \a rng as Sequential draws.
 */
template <ExecutionPolicy P, typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Space, Scalar, Allocator>, FitError>
fitMixture(
    P && policy,
    ParticleSet<Space, Scalar, Allocator> const & particles,
    Eigen::Index k,
    G & rng,
    FitOptions<Scalar> const & options = {}
)
{
    auto const source = internal::fitSource(particles);
    if (!source)
    {
        return std::unexpected{source.error()};
    }
    auto fitter = internal::MixtureFitter<Scalar, Space, Allocator>{*source, options, particles.get_allocator()};
    return fitter.run(std::forward<P>(policy), source->points, source->weights, k, rng);
}

/** \ingroup Distributions
 * fitMixture() of \a mixture under \a policy, a for-each and reductions over its components as
 * fitMixture(P &&, ParticleSet const &, Eigen::Index, G &, FitOptions const &) runs over particles.
 */
template <ExecutionPolicy P, typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Space, Scalar, Allocator>, FitError>
fitMixture(
    P && policy,
    GaussianMixture<Space, Scalar, Allocator> const & mixture,
    Eigen::Index k,
    G & rng,
    FitOptions<Scalar> const & options = {}
)
{
    auto const source = internal::fitSource(mixture);
    auto fitter = internal::MixtureFitter<Scalar, Space, Allocator>{source, options, mixture.get_allocator()};
    return fitter.run(std::forward<P>(policy), source.points, source.weights, k, rng);
}

/** \ingroup Distributions
 * fitMixture() of \a particles seeded from \a seedsFrom, under \a policy, as
 * fitMixture(P &&, ParticleSet const &, Eigen::Index, G &, FitOptions const &) runs.
 */
template <
    ExecutionPolicy P,
    typename Scalar,
    typename Space,
    typename Allocator,
    typename SeedAllocator,
    std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Space, Scalar, Allocator>, FitError>
fitMixture(
    P && policy,
    ParticleSet<Space, Scalar, Allocator> const & particles,
    GaussianMixture<Space, Scalar, SeedAllocator> const & seedsFrom,
    Eigen::Index k,
    G & rng,
    FitOptions<Scalar> const & options = {}
)
{
    EMRYS_FILTER_ASSERT(particles.space() == seedsFrom.space() && "the particles and seeds must share a space");
    auto const source = internal::fitSource(particles);
    if (!source)
    {
        return std::unexpected{source.error()};
    }
    auto const seeds = internal::fitSource(seedsFrom);
    auto fitter = internal::MixtureFitter<Scalar, Space, Allocator>{*source, options, particles.get_allocator()};
    return fitter.run(std::forward<P>(policy), seeds.points, seeds.weights, k, rng);
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes a FitError in words. */
inline void
describe(std::ostream & stream, FitError error, Indent indent)
{
    stream << indent;
    switch (error)
    {
        case FitError::Degenerate:
            stream << "the source has no spread in some direction";
            break;
        case FitError::InvalidWeights:
            stream << "the weights cannot be normalized";
            break;
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
