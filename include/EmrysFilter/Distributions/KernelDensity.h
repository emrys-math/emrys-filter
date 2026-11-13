#pragma once

/** \file
 * KernelDensity: the %Gaussian kernel density estimate over a ParticleSet, the read-only view
 * that ParticleSet::toDensity() returns, with its draws. Included by ParticleSet.h at its end;
 * include either. Read Fitting.h next.
 */

#include <cstddef>
#include <expected>
#include <random>
#include <span>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Indexing.h>

namespace Emrys
{

/** \ingroup Distributions
 * \brief %Gaussian kernel density estimate over a ParticleSet.
 *
 * Represents the density
 * \f[ \hat p(x) = \sum_i \hat w_i \, \mathcal{N}(x; x_i, H), \f]
 * where \f$ x_i \f$ are the particles, \f$ \hat w_i \f$ their normalized weights,
 * and \f$ H \f$ the bandwidth (kernel covariance), which is factorized once and
 * shared by every kernel: a weighted
 * [kernel density estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation) with a
 * multivariate %Gaussian kernel (\ref Silverman1986 "Silverman (1986)", as given in
 * [Multivariate KDE](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Definition)).
 *
 * Created by ParticleSet::toDensity(). It checks the source set for changes, as its copies do:
 * reading it after the set has changed is a programmer error (decisions.md [view-stamp]). A KernelDensity
 * must not outlive its source.
 *
 * \sa ParticleSet::toDensity(), silvermanBandwidth()
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class KernelDensity
{
public:

    using SpaceType = Space_;                                     ///< the particles' space
    using ParamsType = ParticleSet<Space_, Scalar_, Allocator_>;  ///< the particles it reads
    using allocator_type = Allocator_;                            ///< the particles' allocator

    /// the length of a state, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = ParamsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a state, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = ParamsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of the bandwidth, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = ParamsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = ParamsType::MaxDifferenceDimensionAtCompileTime;

    /** The zero-mean kernel, in the linear tangent space of \a Space_. */
    using KernelType =
        GaussianParams<Linear<DifferenceDimensionAtCompileTime, MaxDifferenceDimensionAtCompileTime>, Scalar_>;

    using MomentsType = typename ParamsType::MomentsType;                    ///< a mean and covariance together
    using Scalar = typename ParamsType::Scalar;                              ///< the scalar type
    using Index = typename ParamsType::Index;                                ///< Eigen's index type
    using VectorType = typename ParamsType::VectorType;                      ///< a value, such as a state
    using DifferenceVectorType = typename ParamsType::DifferenceVectorType;  ///< a difference
    using MatrixType = typename ParamsType::MatrixType;                      ///< a matrix over differences
    using SampleSetType = typename ParamsType::SampleSetType;                ///< states as columns

    /** \returns the number of particles */
    [[nodiscard]] Index
    size() const
    {
        return particles().size();
    }

    /** \returns the number of coordinates of every particle state */
    [[nodiscard]] Index
    valueDimension() const
    {
        return particles().valueDimension();
    }

    /** \returns the number of rows and columns of the bandwidth */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return particles().differenceDimension();
    }

    /** \returns the particles' space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return particles().space();
    }

    /** \returns the particles; a programmer error if they have changed since toDensity() */
    [[nodiscard]] ParamsType const &
    particles() const
    {
        return m_particles.get();
    }

    /** \returns the log of the sum of the (unnormalized) particle weights */
    [[nodiscard]] Scalar
    logTotalWeight() const
    {
        return m_logTotalWeight;
    }

    /** \returns the kernel covariance \f$ H \f$ */
    [[nodiscard]] MatrixType const &
    bandwidth() const
    {
        return m_kernel.covarianceLower();
    }

    /** \returns a view of the zero-mean kernel \f$ \mathcal{N}(0, H) \f$. It must not outlive this estimate. */
    [[nodiscard]] typename KernelType::DensityType
    kernel() const
    {
        auto const density = m_kernel.toDensity();
        return *density;
    }

    /** \returns the mean and covariance of the estimate: the particles' mean, and
     * their weighted covariance plus \f$ H \f$.
     */
    [[nodiscard]] MomentsType
    moments() const
    {
        return moments(execution::sequential);
    }

    /** moments() under \a policy: the particles' moments are reductions over them (see
     * \ref Execution).
     */
    template <ExecutionPolicy P>
    [[nodiscard]] MomentsType
    moments(P && policy) const
    {
        MomentsType result = internal::weightedMomentsIn<MomentsType>(
            std::forward<P>(policy),
            space(),
            particles().states(),
            particles().logWeights(),
            m_logTotalWeight
        );
        result.covarianceLower() += m_kernel.covarianceLower().template selfadjointView<Eigen::Lower>().toDenseMatrix();
        return result;
    }

    /** \returns the log of the density at \a x
     *
     * Costs one triangular solve per particle, each into a vector sized by the space, so it
     * allocates nothing for a bounded space. It writes nothing, so it may be evaluated at several
     * points at once.
     *
     * \sa density()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        return logDensity(execution::sequential, x);
    }

    /** logDensity() under \a policy: a log-sum-exp over the particles, reduced over chunks of them
     * (see \ref Execution).
     */
    template <ExecutionPolicy P, typename Derived>
    [[nodiscard]] Scalar
    logDensity(P && policy, Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "logDensity() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        auto const & llt = internal::FactorizedGaussian::llt(m_kernel);
        auto const & set = particles();
        LogSumExp<Scalar> const sum = internal::reduceRanges(
            std::forward<P>(policy),
            size(),
            [&](Index begin, Index length) {
                auto range = LogSumExp<Scalar>{};
                for (Index i = begin; i < begin + length; ++i)
                {
                    DifferenceVectorType residual = Emrys::minus(set.space(), set.state(i), x);
                    llt.matrixL().solveInPlace(residual);
                    range.add(set.logWeight(i) - Scalar(0.5) * residual.squaredNorm());
                }
                return range;
            },
            [](LogSumExp<Scalar> a, LogSumExp<Scalar> const & b) {
                a.add(b);
                return a;
            }
        );
        return sum.value() - m_logTotalWeight + internal::FactorizedGaussian::logPeak(m_kernel);
    }

    /** \returns the density \f$ \hat p(x) \f$ at \a x
     *
     * \warning Underflows easily in high dimensions; prefer logDensity().
     *
     * \sa logDensity()
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

private:

    friend ParamsType;  ///< the particles construct views

    /** \returns an estimate over \a particles, whose weights total \f$ e^{\mathrm{logTotalWeight}} \f$,
     *          or a GaussianError if \a bandwidth is not positive definite or not finite.
     *          Only the lower triangle of \a bandwidth is read.
     */
    template <MatrixExpression Bandwidth>
    [[nodiscard]] static std::expected<KernelDensity, GaussianError>
    create(ParamsType const & particles, Scalar logTotalWeight, Bandwidth && bandwidth)
    {
        auto kernel =
            KernelType{DifferenceVectorType::Zero(particles.differenceDimension()), std::forward<Bandwidth>(bandwidth)};
        if (auto result = kernel.factorize(); !result)
        {
            return std::unexpected{result.error()};
        }
        return KernelDensity{particles, logTotalWeight, std::move(kernel)};
    }

    /** Views \a particles and records their stamp, with the factorized \a kernel; only create()
     * calls this.
     */
    KernelDensity(ParamsType const & particles, Scalar logTotalWeight, KernelType kernel)
        : m_particles{particles}
        , m_logTotalWeight{logTotalWeight}
        , m_kernel{std::move(kernel)}
    {}

    internal::ValuesCheck<ParamsType> m_particles;  ///< the check of the particles
    Scalar m_logTotalWeight;                        ///< the log of the total weight when created
    KernelType m_kernel;                            ///< \f$ \mathcal{N}(0, H) \f$, factorized
};

/** \ingroup Distributions
 * \returns \a n draws from \a kde, one per column.
 *
 * Each draw picks a particle in proportion to its weight using \a resampler, then
 * adds kernel noise \f$ L z \f$ with \f$ z \sim \mathcal{N}(0, I) \f$ and
 * \f$ H = L L^T \f$. The default resamplers::Multinomial makes the draws independent;
 * a lower-variance resampler such as resamplers::Systematic spreads them more evenly
 * over the particles, as in the regularized particle filter of
 * \ref Musso2001 "Musso, Oudjane, and Le Gland (2001)".
 *
 * \sa drawSamplesWithLogDensity(KernelDensity const &, Eigen::Index, G &, R const &), \ref Resamplers
 */
template <
    typename Scalar,
    typename Space,
    typename Allocator,
    std::uniform_random_bit_generator G,
    typename R = resamplers::Multinomial>
    requires Resampler<R, Scalar, G>
[[nodiscard]] SampleSet<Space, Scalar>
drawSamples(KernelDensity<Space, Scalar, Allocator> const & kde, Eigen::Index n, G & rng, R const & resampler = {})
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using SampleSetType = SampleSet<Space, Scalar>;
    auto const & particles = kde.particles();
    auto weights = internal::bufferOf<Scalar>(particles.get_allocator());
    weights.resize(std::size_t(particles.size()));
    for (Eigen::Index i = 0; i < particles.size(); ++i)
    {
        weights[std::size_t(i)] = Eigen::numext::exp(particles.logWeight(i) - kde.logTotalWeight());
    }
    auto ancestors = internal::bufferOf<Eigen::Index>(particles.get_allocator());
    ancestors.resize(std::size_t(n));
    resampler(std::span<Scalar const>{weights}, std::span<Eigen::Index>{ancestors}, rng);

    using Normals = SampleSet<DifferenceSpace<Space>, Scalar>;
    Normals const offsets = internal::multiply(
        internal::asTriangular<Eigen::Lower>(kde.kernel().llt().matrixLLT()),
        internal::standardNormal<Normals>(kde.differenceDimension(), n, rng)
    );
    auto samples = SampleSetType{kde.valueDimension(), n};
    for (Eigen::Index j = 0; j < n; ++j)
    {
        samples.col(j) = Emrys::plus(kde.space(), particles.state(ancestors[std::size_t(j)]), offsets.col(j));
    }
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a kde, a particle picked in proportion to its weight plus kernel noise
 * \f$ L z \f$, in a vector sized by the space: the stream of draws that drawSample() of a %Gaussian
 * is. Its draws are independent, as resamplers::Multinomial makes those of drawSamples(); a
 * lower-variance resampler spreads a batch over the particles, which one draw at a time cannot.
 */
template <typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(KernelDensity<Space, Scalar, Allocator> const & kde, G & rng)
{
    using Difference = DifferenceVector<Space, Scalar>;
    auto const & particles = kde.particles();
    Eigen::Index const picked = internal::drawIndex<Scalar>(
        particles.size(),
        [&](Eigen::Index k) {
            return Eigen::numext::exp(Scalar(particles.logWeight(k) - kde.logTotalWeight()));
        },
        rng
    );
    Difference const z = internal::standardNormalVector<Difference>(kde.differenceDimension(), rng);
    return typename Moments<Space, Scalar>::VectorType{Emrys::plus(
        kde.space(),
        particles.state(picked),
        Difference{internal::multiply(internal::asTriangular<Eigen::Lower>(kde.kernel().llt().matrixLLT()), z)}
    )};
}

/** \ingroup Distributions
 * \returns \a n draws from \a kde, one per column, with the log-density of \a kde at each.
 *
 * Draws exactly as drawSamples() does. Unlike the %Gaussian case the log-densities
 * are not free: each costs a full KernelDensity::logDensity() evaluation, so the
 * total is \f$ O(n \cdot \mathrm{size}() \cdot d^2) \f$.
 *
 * \sa drawSamples(KernelDensity const &, Eigen::Index, G &, R const &)
 */
template <
    typename Scalar,
    typename Space,
    typename Allocator,
    std::uniform_random_bit_generator G,
    typename R = resamplers::Multinomial>
    requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(
    KernelDensity<Space, Scalar, Allocator> const & kde,
    Eigen::Index n,
    G & rng,
    R const & resampler = {}
)
{
    return drawSamplesWithLogDensity(execution::sequential, kde, n, rng, resampler);
}

/** \ingroup Distributions
 * drawSamplesWithLogDensity() under \a policy: the draws are taken as the sequential form takes
 * them, and their log-densities are a for-each over the draws (see \ref Execution), so the result
 * is Sequential's.
 */
template <
    ExecutionPolicy P,
    typename Scalar,
    typename Space,
    typename Allocator,
    std::uniform_random_bit_generator G,
    typename R = resamplers::Multinomial>
    requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(
    P && policy,
    KernelDensity<Space, Scalar, Allocator> const & kde,
    Eigen::Index n,
    G & rng,
    R const & resampler = {}
)
{
    auto result = SamplesWithLogDensity<Space, Scalar>{drawSamples(kde, n, rng, resampler), {}};
    result.logDensity.resize(n);
    internal::forEachIndex(std::forward<P>(policy), n, [&](Eigen::Index j) {
        result.logDensity(j) = kde.logDensity(result.samples.col(j));
    });
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a kde, as drawSample() draws it, with the log-density of \a kde at it,
 * a full KernelDensity::logDensity() evaluation, in a vector sized by the space: the stream of
 * draws that drawSampleWithLogDensity() of a %Gaussian is.
 */
template <typename Scalar, typename Space, typename Allocator, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(KernelDensity<Space, Scalar, Allocator> const & kde, G & rng)
{
    auto result = SampleWithLogDensity<Space, Scalar>{drawSample(kde, rng), 0};
    result.logDensity = kde.logDensity(result.sample);
    return result;
}

}  // namespace Emrys
