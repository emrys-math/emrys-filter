#pragma once

/** \file
 * Drawing from a %Gaussian: drawSample() and drawSampleWithLogDensity(), the one-at-a-time draws
 * that allocate nothing for a bounded space [allocation] and that drawSamples() and
 * drawSamplesWithLogDensity() of Draws.h read as a range, and the internal standard normal and
 * uniform draws. A draw's root is the default one of Roots.h. The other distributions' draws are
 * in their own headers. The generator is the caller's and the draws come from the standard
 * distributions, so a seed fixes them within one standard library only (decisions.md
 * [random-draws]). Read GaussianForm.h next.
 */

#include <algorithm>
#include <concepts>
#include <random>

#include <Eigen/Cholesky>

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Distributions/Draws.h>
#include <EmrysFilter/Distributions/SampleSet.h>

namespace Emrys
{

namespace internal
{

/** \returns a \a rows by \a cols matrix of independent standard normal draws,
 * filled in storage order so results depend only on \a rng.
 */
template <typename Matrix, std::uniform_random_bit_generator G>
[[nodiscard]] Matrix
standardNormal(Eigen::Index rows, Eigen::Index cols, G & rng)
{
    using Scalar = typename Matrix::Scalar;
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    auto z = Matrix{rows, cols};
    auto normal = std::normal_distribution<Scalar>{};
    std::generate(z.data(), z.data() + z.size(), [&] {
        return normal(rng);
    });
    return z;
}

/** \returns a vector of \a rows independent standard normal draws, sized as \a Vector is, so that a
 * bounded vector allocates nothing. A fixed vector is not built from \a rows and a column count, as
 * standardNormal() builds its matrix, since Eigen would take those as its coefficients.
 */
template <typename Vector, std::uniform_random_bit_generator G>
[[nodiscard]] Vector
standardNormalVector(Eigen::Index rows, G & rng)
{
    static_assert(std::floating_point<typename Vector::Scalar>, "sampling requires a standard floating-point scalar");
    auto z = Vector{};
    z.resize(rows);
    auto normal = std::normal_distribution<typename Vector::Scalar>{};
    std::generate(z.data(), z.data() + z.size(), [&] {
        return normal(rng);
    });
    return z;
}

/** Fills \a z, contiguous storage such as a map, with independent standard normal draws from
 * \a normal in storage order, without allocating. A caller that fills several blocks passes them
 * all the same \a normal, which may keep a value between draws, so that the draws are those of
 * filling the blocks at once and no value is discarded.
 */
template <typename Derived, std::uniform_random_bit_generator G, typename Scalar>
void
fillStandardNormal(Derived && z, G & rng, std::normal_distribution<Scalar> & normal)
{
    static_assert(std::same_as<typename std::remove_cvref_t<Derived>::Scalar, Scalar>);
    std::generate(z.data(), z.data() + z.size(), [&] {
        return normal(rng);
    });
}

/** Fills \a z, contiguous storage such as a map, with independent standard normal draws in
 * storage order, as standardNormal() does, without allocating.
 */
template <typename Derived, std::uniform_random_bit_generator G>
void
fillStandardNormal(Derived && z, G & rng)
{
    using Scalar = typename std::remove_cvref_t<Derived>::Scalar;
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    auto normal = std::normal_distribution<Scalar>{};
    fillStandardNormal(std::forward<Derived>(z), rng, normal);
}

/** \returns a \a rows by \a cols matrix of independent draws uniform on <tt>[0, 1)</tt>,
 * filled in storage order so results depend only on \a rng.
 */
template <typename Matrix, std::uniform_random_bit_generator G>
[[nodiscard]] Matrix
standardUniform(Eigen::Index rows, Eigen::Index cols, G & rng)
{
    using Scalar = typename Matrix::Scalar;
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    auto u = Matrix{rows, cols};
    auto uniform = std::uniform_real_distribution<Scalar>{};
    std::generate(u.data(), u.data() + u.size(), [&] {
        return uniform(rng);
    });
    return u;
}

}  // namespace internal

/** \ingroup Distributions
 * \returns one draw from \a gaussian in a vector sized by the space, so that a bounded space
 * allocates nothing: \f$ \mu \boxplus L z \f$ with \f$ z \sim \mathcal{N}(0, I) \f$ and
 * \f$ L \f$ the Cholesky factor of the covariance, the standard
 * [affine transformation of standard
 * normals](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution), with
 * the space's \f$ \boxplus \f$ in place of addition.
 *
 * Called again and again, it is a stream of independent draws, the sequence that drawSamples()
 * gives from the same generator, since each element of that range is one call of this.
 *
 * \sa drawSamples()
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(Gaussian<Space, Scalar> const & gaussian, G & rng)
{
    using Difference = DifferenceVector<Space, Scalar>;
    Difference const z = internal::standardNormalVector<Difference>(gaussian.differenceDimension(), rng);
    return typename Moments<Space, Scalar>::VectorType{
        Emrys::plus(gaussian.space(), gaussian.mean(), Difference{internal::multiply(gaussian.covarianceRoot(), z)})
    };
}

/** \ingroup Distributions
 * \returns one draw from \a gaussian, as drawSample() draws it, with the log-density of
 * \a gaussian at it, so that a bounded space allocates nothing: the stream of draws for code
 * that needs each one's density, such as an importance sampler on a system with no heap.
 *
 * The log-density costs almost nothing extra in a linear space, \f$ \|z\|^2 \f$ being the squared
 * Mahalanobis distance of \f$ \mu + L z \f$; in other spaces it is Gaussian::logDensity() of the
 * draw, so that a draw that wraps far around a periodic coordinate gets the nearest-image density.
 *
 * \sa drawSample(), drawSamplesWithLogDensity()
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(Gaussian<Space, Scalar> const & gaussian, G & rng)
{
    using Difference = DifferenceVector<Space, Scalar>;
    Difference const z = internal::standardNormalVector<Difference>(gaussian.differenceDimension(), rng);
    auto result = SampleWithLogDensity<Space, Scalar>{
        typename Moments<Space, Scalar>::VectorType{
            Emrys::plus(gaussian.space(), gaussian.mean(), Difference{internal::multiply(gaussian.covarianceRoot(), z)})
        },
        0
    };
    if constexpr (isLinear<Space>)
    {
        result.logDensity = gaussian.logPeak() - Scalar(0.5) * z.squaredNorm();
    }
    else
    {
        result.logDensity = gaussian.logDensity(result.sample);
    }
    return result;
}

}  // namespace Emrys
