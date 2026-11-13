#pragma once

/** \file
 * The likelihoods that sample-based filters weight by (decisions.md [likelihood-measurements],
 * [execution-policies]): logLikelihoods() of a measurement at the columns of a matrix of states, under an
 * execution policy, and logLikelihood() at one. It needs none of the update, which Measurement.h holds.
 * Read SwitchingModel.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/MeasurementReads.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** \returns the log-density at the residual \a residual of the additive noise \a noise of
 * \a resolved, widened by \a omega, the covariance the call's affine inputs carry through the
 * function (decisions.md [likelihood-measurements]): the %Gaussian's log-density, or for a mixture
 * the log of the weighted sum of its components', each widened by \a omega; NaN where \a omega is
 * not finite, as at a point where the function's Jacobian is not, since the linearization that
 * widens the density is not defined there
 */
template <typename Scalar, typename Noise, typename Residual, typename Omega>
[[nodiscard]] Scalar
widenedLogDensity(Noise const & noise, Residual const & residual, Omega const & omega)
{
    if (!allFinite(omega))
    {
        return Eigen::NumTraits<Scalar>::quiet_NaN();
    }
    auto const component = [&](auto const & gaussian) {
        using Gaussian = std::remove_cvref_t<decltype(gaussian)>;
        using Widened = GaussianParams<typename Gaussian::SpaceType, Scalar>;
        using Matrix = typename Widened::MatrixType;
        Matrix cov = symmetricFromLower<Matrix>(gaussian.covarianceLower());
        cov += omega;
        auto const widened = Widened{
            gaussian.space(),
            typename Widened::VectorType{gaussian.mean().template cast<Scalar>()},
            std::move(cov)
        };
        auto const density = widened.toDensity();
        EMRYS_FILTER_ASSERT(density && "the measurement noise must factorize");
        return Scalar(density->logDensity(residual));
    };
    if constexpr (isGaussianMixture<Noise>)
    {
        auto const total = noise.logTotalWeight();
        EMRYS_FILTER_ASSERT(total && "the noise mixture's weights must be valid");
        auto sum = LogSumExp<Scalar>{};
        for (auto const & entry : noise)
        {
            sum.add(Scalar(entry.logWeight) - Scalar(*total) + component(entry.component));
        }
        return sum.value();
    }
    else
    {
        return component(noise);
    }
}

/** Writes the batched logLikelihood() of \a resolved, a ResolvedMeasurement, at \a count points into
 * \a result, an entry per point, by one call over every point under Sequential and one per chunk of
 * internal::chunkSize points under another \a policy (decisions.md [batched-functions]). Point
 * \a j's rows of a state, from row \a start on, are <tt>rowsOf(j)</tt>; the batch's input is
 * gathered from them into \a scratch, a Buffer resized to a column per point, so that the chunks
 * write disjoint columns of it.
 */
template <ExecutionPolicy P, typename R, typename RowsOf, typename Scalar, typename Scratch>
void
fillBatchedLogLikelihoods(
    P && policy,
    R const & resolved,
    Eigen::Index count,
    RowsOf const & rowsOf,
    Eigen::Index start,
    Scalar * result,
    Scratch & scratch
)
{
    using InMatrix = typename R::template InMatrix<Scalar>;
    Eigen::Index const rows = valueDimension(resolved.in());
    scratch.resize(std::size_t(rows * count));
    forEachChunk(std::forward<P>(policy), count, [&](Eigen::Index begin, Eigen::Index length) {
        auto x = Eigen::Map<InMatrix>{scratch.data() + rows * begin, rows, length};
        resolved.template gatherColumnsFromRows<Scalar>(
            x,
            [&](Eigen::Index j) {
                return rowsOf(begin + j);
            },
            start
        );
        resolved.batchedLogLikelihoods(
            Eigen::Map<InMatrix const>{x.data(), rows, length},
            Eigen::Map<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>{result + begin, length}
        );
    });
}

/** Writes logLikelihoods() of \a measurement at \a states in \a space into \a result, which has
 * one entry per column, without allocating for them, as a for-each over the columns under
 * \a policy: the model's own logLikelihood(g) where it has one, and otherwise the log-density of its
 * additive noise at the residual \f$ z \boxminus h(x_j) \f$, with the covariance the call's affine
 * inputs carry through the function added (decisions.md [likelihood-measurements]). The noise is
 * factorized once, before the loop, where no input widens it.
 *
 * A batched logLikelihood(), or a batched function under additive noise that no input widens, is
 * called once over every column under Sequential and once per chunk of internal::chunkSize columns
 * otherwise (decisions.md [batched-functions]), its input and output in \a scratch, which is resized
 * to hold a column of each per state and reused.
 */
template <ExecutionPolicy P, WeighsPoints M, typename Space, typename Derived, typename Result, typename Scratch>
void
fillLogLikelihoods(
    P && policy,
    M const & measurement,
    Space const & space,
    Eigen::MatrixBase<Derived> const & states,
    Result && result,
    Scratch & scratch
)
{
    using Scalar = typename Derived::Scalar;
    EMRYS_FILTER_ASSERT(states.rows() == valueDimension(space) && "the states do not fit the space");
    EMRYS_FILTER_ASSERT(result.size() == states.cols() && "give one entry per state");
    static_assert(
        !hasConsider<M>,
        "consider() is a Kalman update's; a filter that weights points moves no block it does not condition "
        "(decisions.md [consider-states])"
    );
    auto const resolved = resolveMeasurement(measurement, space);
    using R = std::remove_cvref_t<decltype(resolved)>;
    if constexpr (R::BatchedLikelihood)
    {
        fillBatchedLogLikelihoods(
            std::forward<P>(policy),
            resolved,
            states.cols(),
            [&](Eigen::Index j) {
                return states.col(j);
            },
            Eigen::Index{0},
            result.data(),
            scratch
        );
    }
    else if constexpr (R::HasLogLikelihood)
    {
        forEachIndex(std::forward<P>(policy), states.cols(), [&](Eigen::Index j) {
            result(j) = Scalar(resolved.logLikelihoodAt(states.col(j)));
        });
    }
    else
    {
        static_assert(
            R::HasAdditiveNoise,
            "a measurement model with no additive noise has no likelihood of its own at a point; give it "
            "logLikelihood(g), or name the inputs it is affine in with affineIn() beside additiveNoise()"
        );
        static_assert(
            R::inputsAffine(),
            "the likelihood at a point of noise that passes through the function is exact only where the model is "
            "affine in it, so name the input in affineIn(), or give the model logLikelihood(g) (decisions.md "
            "[likelihood-measurements])"
        );
        auto const & space_ = resolved.space();
        if constexpr (R::InputCount == 0 && R::BatchedFunction)
        {
            // The function once per batch, its outputs after its inputs in the scratch.
            auto const & noise = resolved.noise().distribution();
            auto const density = noise.toDensity();
            EMRYS_FILTER_ASSERT(density && "the measurement noise must factorize");
            using InMatrix = typename R::template InMatrix<Scalar>;
            using OutMatrix = typename R::template OutMatrix<Scalar>;
            Eigen::Index const count = states.cols();
            Eigen::Index const inRows = valueDimension(resolved.in());
            Eigen::Index const outRows = valueDimension(space_);
            scratch.resize(std::size_t((inRows + outRows) * count));
            forEachChunk(std::forward<P>(policy), count, [&](Eigen::Index begin, Eigen::Index length) {
                auto x = Eigen::Map<InMatrix>{scratch.data() + inRows * begin, inRows, length};
                auto y = Eigen::Map<OutMatrix>{scratch.data() + inRows * count + outRows * begin, outRows, length};
                resolved.template gatherColumnsFromRows<Scalar>(
                    x,
                    [&](Eigen::Index j) {
                        return states.col(begin + j);
                    },
                    Eigen::Index{0}
                );
                resolved.evaluateBatch(Eigen::Map<InMatrix const>{x.data(), inRows, length}, y);
                for (Eigen::Index j = 0; j < length; ++j)
                {
                    result(begin + j) = density->logDensity(observedResidual<Scalar>(resolved, space_, y.col(j)));
                }
            });
        }
        else if constexpr (R::InputCount == 0)
        {
            auto const & noise = resolved.noise().distribution();
            auto const density = noise.toDensity();
            EMRYS_FILTER_ASSERT(density && "the measurement noise must factorize");
            forEachIndex(std::forward<P>(policy), states.cols(), [&](Eigen::Index j) {
                auto const predicted = valueOf(resolved.evaluateAtState(states.col(j))).eval();
                result(j) = density->logDensity(observedResidual<Scalar>(resolved, space_, predicted));
            });
        }
        else if constexpr (R::BatchedFunction)
        {
            // The function and its Jacobian in the inputs once per batch, each input at its mean, the
            // inputs' covariance carried through that Jacobian at each point: what the linear method
            // gives below, exact where the function is affine in them.
            using Out = typename R::Out;
            using InMatrix = typename R::template InMatrix<Scalar>;
            using OutMatrix = typename R::template OutMatrix<Scalar>;
            using JacobianMatrix = MatrixOf<
                Scalar,
                valueDimensionAtCompileTime<DifferenceSpace<Out>>,
                Eigen::Dynamic,
                maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
                Eigen::Dynamic>;
            using Omega = typename Moments<Out, Scalar>::MatrixType;
            auto const inputs = resolved.inputs();
            Eigen::Index w = 0;
            std::apply(
                [&](auto const &... input) {
                    ((w += inputOf(input).differenceDimension()), ...);
                },
                inputs
            );
            Eigen::Index const count = states.cols();
            Eigen::Index const inRows = valueDimension(resolved.in());
            Eigen::Index const outRows = valueDimension(space_);
            Eigen::Index const m = differenceDimension(space_);
            scratch.resize(std::size_t((inRows + outRows + m * w) * count));
            Scalar * const outputs = scratch.data() + inRows * count;
            Scalar * const jacobianData = outputs + outRows * count;
            forEachChunk(std::forward<P>(policy), count, [&](Eigen::Index begin, Eigen::Index length) {
                auto x = Eigen::Map<InMatrix>{scratch.data() + inRows * begin, inRows, length};
                auto y = Eigen::Map<OutMatrix>{outputs + outRows * begin, outRows, length};
                auto jacobians = Eigen::Map<JacobianMatrix>{jacobianData + m * w * begin, m, w * length};
                resolved.template gatherColumnsFromRows<Scalar>(
                    x,
                    [&](Eigen::Index j) {
                        return states.col(begin + j);
                    },
                    Eigen::Index{0}
                );
                resolved.evaluateBatchWithInputJacobians(
                    Eigen::Map<InMatrix const>{x.data(), inRows, length},
                    y,
                    jacobians
                );
                for (Eigen::Index j = 0; j < length; ++j)
                {
                    Omega omega = Omega::Zero(m, m);
                    Eigen::Index offset = j * w;
                    std::apply(
                        [&](auto const &... argument) {
                            (
                                [&](auto const & input) {
                                    using Input = std::remove_cvref_t<decltype(input)>;
                                    using Cov = MatrixOf<
                                        Scalar,
                                        Input::DifferenceDimensionAtCompileTime,
                                        Input::DifferenceDimensionAtCompileTime,
                                        Input::MaxDifferenceDimensionAtCompileTime,
                                        Input::MaxDifferenceDimensionAtCompileTime>;
                                    Eigen::Index const d = input.differenceDimension();
                                    auto const block =
                                        middleColsOf<Input::DifferenceDimensionAtCompileTime>(jacobians, offset, d);
                                    Cov const cov = symmetricFromLower<Cov>(covarianceLowerOf(input));
                                    omega.noalias() += multiply(multiply(block, cov), block.transpose());
                                    offset += d;
                                }(inputOf(argument)),
                                ...
                            );
                        },
                        inputs
                    );
                    omega = (Scalar(0.5) * (omega + omega.transpose())).eval();
                    result(begin + j) = widenedLogDensity<Scalar>(
                        resolved.noise().distribution(),
                        observedResidual<Scalar>(resolved, space_, y.col(j)),
                        omega
                    );
                }
            });
        }
        else
        {
            // The inputs' covariance carried through the function's Jacobian at the point, by the linear
            // method, which is exact where the function is affine in them.
            auto const & reads = resolved.reads();
            using Reads = typename R::Reads;
            using ReadPoint = PointInput<typename Reads::JointSpace, Scalar>;
            auto const f = ResolvedFunction{resolved};
            forEachIndex(std::forward<P>(policy), states.cols(), [&](Eigen::Index j) {
                auto const read =
                    ReadPoint{reads.space(), typename ReadPoint::VectorType{states.col(j)(reads.indices())}};
                result(j) = linearizeMeasurement(
                    resolved,
                    methods::linearized(),
                    f,
                    read,
                    [&](auto const & problem, auto const & linearized) {
                        using Problem = std::remove_cvref_t<decltype(problem)>;
                        using Omega = typename Problem::OutputMatrix;
                        Omega const omega = inputsCovariance<Omega>(resolved, problem, linearized.jacobian);
                        return widenedLogDensity<Scalar>(
                            resolved.noise().distribution(),
                            observedResidual<Scalar>(resolved, space_, linearized.mean),
                            omega
                        );
                    }
                );
            });
        }
    }
}

/** fillLogLikelihoods() with scratch of its own, for a caller that keeps none. */
template <ExecutionPolicy P, WeighsPoints M, typename Space, typename Derived, typename Result>
void
fillLogLikelihoods(
    P && policy,
    M const & measurement,
    Space const & space,
    Eigen::MatrixBase<Derived> const & states,
    Result && result
)
{
    auto scratch = Buffer<typename Derived::Scalar, DefaultAllocator>();
    fillLogLikelihoods(std::forward<P>(policy), measurement, space, states, std::forward<Result>(result), scratch);
}

/** fillLogLikelihoods() under Sequential. */
template <WeighsPoints M, typename Space, typename Derived, typename Result>
void
fillLogLikelihoods(
    M const & measurement,
    Space const & space,
    Eigen::MatrixBase<Derived> const & states,
    Result && result
)
{
    fillLogLikelihoods(execution::sequential, measurement, space, states, std::forward<Result>(result));
}

}  // namespace internal

/** \ingroup Models
 * \returns \f$ \log p(z \mid x_j) \f$ for \a measurement at each column \f$ x_j \f$ of \a states
 * in \a space, for filters that weight state points, such as particle and grid filters.
 *
 * \a measurement is an observation of a measurement model (see observe()), with or without a
 * method. Its model's own logLikelihood(g) if it has one, whatever its noise says (decisions.md
 * [likelihood-measurements]); otherwise the log-density of its additive noise at the residual
 * \f$ z \boxminus h(x_j) \f$, formed in the measurement space, since the residual at the true state
 * is the noise. An uncertain input of the call that passes through the function widens that
 * density by its covariance carried through the function's Jacobian at the point, which is exact
 * where the model names the input in affineIn(); otherwise the model must supply its own likelihood,
 * and the compile error says so. The noise must factorize; it is factorized once for all the
 * columns where no input widens it. Where the Jacobian at a column is not finite, as for a range
 * at the sensor's position, the widening is not defined, and that column's log-likelihood is NaN,
 * as it is where the function's value is not finite; a ParticleFilter's update refuses a NaN.
 *
 * This is the likelihood \f$ p(z_k \mid x_k^i) \f$ that weights the particles of
 * \ref Arulampalam2002 "Arulampalam et al. (2002)", equation (48), and the cells of a point-mass
 * filter.
 *
 * \sa logLikelihood()
 */
template <internal::WeighsPoints M, typename Space, typename Derived>
[[nodiscard]] Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>
logLikelihoods(M const & measurement, Space const & space, Eigen::MatrixBase<Derived> const & states)
{
    return logLikelihoods(execution::sequential, measurement, space, states);
}

/** \ingroup Models
 * logLikelihoods() under \a policy, a for-each over the columns (see \ref Execution). A
 * model's own logLikelihood(g) and its function then run for several columns at
 * once, so they must not write shared state.
 */
template <ExecutionPolicy P, internal::WeighsPoints M, typename Space, typename Derived>
[[nodiscard]] Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>
logLikelihoods(P && policy, M const & measurement, Space const & space, Eigen::MatrixBase<Derived> const & states)
{
    auto result = Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>{states.cols()};
    internal::fillLogLikelihoods(std::forward<P>(policy), measurement, space, states, result);
    return result;
}

/** \ingroup Models
 * \returns \f$ \log p(z \mid x) \f$ for \a measurement at the state point \a x, a column vector
 * in \a space, as logLikelihoods() computes it.
 */
template <internal::WeighsPoints M, typename Space, typename Derived>
[[nodiscard]] typename Derived::Scalar
logLikelihood(M const & measurement, Space const & space, Eigen::MatrixBase<Derived> const & x)
{
    EMRYS_FILTER_ASSERT(x.cols() == 1 && "the state point must be a column vector");
    return logLikelihoods(measurement, space, x)(0);
}

/** \ingroup Models
 * logLikelihoods() in the fixed-size state space \a S, given as a template argument:
 * <tt>logLikelihoods<State>(measurement, states)</tt>.
 */
template <FixedSizeSpace S, internal::WeighsPoints M, typename Derived>
    requires Space<S>
[[nodiscard]] Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>
logLikelihoods(M const & measurement, Eigen::MatrixBase<Derived> const & states)
{
    return logLikelihoods(execution::sequential, measurement, S{}, states);
}

/** \ingroup Models
 * logLikelihoods() under \a policy in the fixed-size state space \a S, given as a template
 * argument: <tt>logLikelihoods<State>(policy, measurement, states)</tt>.
 */
template <FixedSizeSpace S, ExecutionPolicy P, internal::WeighsPoints M, typename Derived>
    requires Space<S>
[[nodiscard]] Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>
logLikelihoods(P && policy, M const & measurement, Eigen::MatrixBase<Derived> const & states)
{
    return logLikelihoods(std::forward<P>(policy), measurement, S{}, states);
}

/** \ingroup Models
 * logLikelihood() in the fixed-size state space \a S, given as a template argument:
 * <tt>logLikelihood<State>(measurement, x)</tt>.
 */
template <FixedSizeSpace S, internal::WeighsPoints M, typename Derived>
    requires Space<S>
[[nodiscard]] typename Derived::Scalar
logLikelihood(M const & measurement, Eigen::MatrixBase<Derived> const & x)
{
    return logLikelihood(measurement, S{}, x);
}

}  // namespace Emrys
