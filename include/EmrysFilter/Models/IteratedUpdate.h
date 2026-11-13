#pragma once

/** \file
 * The iterated measurement update, all internal (decisions.md [iterated-update], [sigma-point-sets]): the
 * read blocks moved to an iterate and respread as the prior spreads them, the iteration's scale and its
 * test, the step to the prior's chart, and iteratedUpdate(), which Measurement.h's update() calls for an
 * Iterated method. Read Measurement.h next.
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

/** An UncertainInput with the space and covariance of \a Input at another mean: the blocks a
 * measurement reads, at an iterate of an Iterated update, spread as the prior spreads them
 * (decisions.md [iterated-update]). Valid while the input it moves is.
 */
template <typename Input>
class MovedInput
{
public:

    using Scalar = typename Input::Scalar;        ///< the scalar type
    using SpaceType = typename Input::SpaceType;  ///< the space the values live in

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = Input::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = Input::MaxValueDimensionAtCompileTime;

    /// the length of a difference, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = Input::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = Input::MaxDifferenceDimensionAtCompileTime;

    /// a value
    using VectorType =
        Eigen::Matrix<Scalar, ValueDimensionAtCompileTime, 1, Eigen::ColMajor, MaxValueDimensionAtCompileTime, 1>;

    /** \a input moved to the mean \a mean. */
    MovedInput(Input const & input, VectorType mean)
        : m_input{input}
        , m_mean{std::move(mean)}
    {}

    /** \returns the space */
    [[nodiscard]] decltype(auto)
    space() const
    {
        return m_input.space();
    }

    /** \returns the moved mean */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return m_mean;
    }

    /** \returns the input's covariance */
    [[nodiscard]] decltype(auto)
    covarianceLower() const
    {
        return m_input.covarianceLower();
    }

    /** \returns the number of coordinates of the mean */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return m_input.valueDimension();
    }

    /** \returns the number of difference coordinates */
    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return m_input.differenceDimension();
    }

private:

    Input const & m_input;  ///< the input moved
    VectorType m_mean;      ///< its new mean
};

/** \a Input moved to another mean, as MovedInput, and spread by another covariance \a Cov: the
 * blocks read as an Iterated sigma-point update spreads its points at an iterate, by the
 * posterior covariance there (decisions.md [sigma-point-sets]). Valid while the input and the covariance are.
 */
template <typename Input, typename Cov>
class RespreadInput : public MovedInput<Input>
{
public:

    /** \a input moved to the mean \a mean, with the covariance \a cov. */
    RespreadInput(Input const & input, typename MovedInput<Input>::VectorType mean, Cov const & cov)
        : MovedInput<Input>{input, std::move(mean)}
        , m_cov{cov}
    {}

    /** \returns the covariance it is spread by */
    [[nodiscard]] Cov const &
    covarianceLower() const
    {
        return m_cov;
    }

private:

    Cov const & m_cov;  ///< the covariance
};

/** The observation \a M (an Observed) with its model's additive mixture noise replaced by the one
 * component \a Component, for an Iterated update, whose branches iterate one at a time, and for an
 * IMM filter, whose modes each take one component (decisions.md [iterated-update]). It reads as an
 * Observed, and ResolvedMeasurement reads its noise() in place of the model's policy (\c OwnNoise).
 * Valid while the observation and the component are.
 */
template <typename M, typename Component>
class WithNoiseComponent
{
public:

    using MethodType = typename M::MethodType;              ///< the method, as the observation's
    using ModelType = typename M::ModelType;                ///< the model, as the observation's
    using ArgumentsType = typename M::ArgumentsType;        ///< the call's arguments, as the observation's
    using FormType = typename M::FormType;                  ///< the covariance update form, as the observation's
    using InnovationRepair = typename M::InnovationRepair;  ///< the repair, as the observation's
    using ConsiderType = typename M::ConsiderType;          ///< the blocks considered, as the observation's
    using NoiseType = AdditiveNoise<Component const &>;     ///< the one component, as additive noise

    static constexpr bool HasMethod = M::HasMethod;            ///< as the observation's
    static constexpr bool HasObservation = M::HasObservation;  ///< as the observation's
    static constexpr bool OwnNoise = true;                     ///< the noise is this one's, not the model's

    /** \a measurement with the noise \a component. */
    WithNoiseComponent(M const & measurement, Component const & component)
        : m_measurement{measurement}
        , m_noise{component}
    {}

    /** \returns the method */
    [[nodiscard]] MethodType const &
    method() const
    {
        return m_measurement.method();
    }

    /** \returns the model */
    [[nodiscard]] ModelType const &
    model() const
    {
        return m_measurement.model();
    }

    /** \returns the one component, as additive noise */
    [[nodiscard]] NoiseType const &
    noise() const
    {
        return m_noise;
    }

    /** \returns the observed value */
    [[nodiscard]] decltype(auto)
    observation() const
    {
        return m_measurement.observation();
    }

    /** \returns the call's arguments */
    [[nodiscard]] ArgumentsType const &
    arguments() const
    {
        return m_measurement.arguments();
    }

    /** \returns the covariance update form */
    [[nodiscard]] decltype(auto)
    covarianceUpdate() const
    {
        return m_measurement.covarianceUpdate();
    }

private:

    M const & m_measurement;  ///< the observation
    NoiseType m_noise;        ///< the component, as its noise
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename M, typename Component>
inline constexpr bool isObserved<WithNoiseComponent<M, Component>>{true};

/// \endcond

/** \returns, for each difference coordinate of a prior with covariance \a cov, the largest move
 * of an Iterated update's step that counts as stopped: \a tolerance prior standard deviations
 * (decisions.md [iterated-update]). Only the diagonal of \a cov is read.
 */
template <typename Cov>
[[nodiscard]] auto
iterationScale(Eigen::MatrixBase<Cov> const & cov, typename Cov::Scalar tolerance)
{
    using Scalar = typename Cov::Scalar;
    EMRYS_FILTER_ASSERT(tolerance >= Scalar(0) && "an iteration's tolerance must not be negative");
    using Result = Eigen::Matrix<Scalar, Cov::RowsAtCompileTime, 1, Eigen::ColMajor, Cov::MaxRowsAtCompileTime, 1>;
    Result result = cov.diagonal().cwiseSqrt() * tolerance;
    return result;
}

/** \returns whether the step from \a delta to \a next moved no coordinate by more than its entry
 * of \a scale, from iterationScale()
 */
template <typename Next, typename Delta, typename Scale>
[[nodiscard]] bool
withinScale(
    Eigen::MatrixBase<Next> const & next,
    Eigen::MatrixBase<Delta> const & delta,
    Eigen::MatrixBase<Scale> const & scale
)
{
    return ((next - delta).cwiseAbs().array() <= scale.array()).all();
}

/** Carries \a jacobian, the Jacobian of a measurement in the blocks read taken at the iterate
 * \f$ \mu_S \boxplus \delta_S \f$ in its tangent space, in place to the Jacobian in
 * \f$ \delta_S \f$ at the prior mean \a readMean of the blocks read \a reads, with \a readStep the
 * iterate's \f$ \delta_S \f$: times the reset Jacobian (resetJacobian()), and unchanged where the blocks
 * have an identity reset (decisions.md [covariance-reset], [iterated-update])
 */
template <typename Reads, typename Jacobian, typename ReadMean, typename ReadStep>
void
toPriorChart(
    Reads const & reads,
    Eigen::MatrixBase<Jacobian> const & jacobian,
    ReadMean const & readMean,
    ReadStep const & readStep
)
{
    if constexpr (!hasIdentityReset<typename Reads::JointSpace>)
    {
        using Scalar = typename Jacobian::Scalar;
        using Result = MatrixOf<
            Scalar,
            Jacobian::RowsAtCompileTime,
            Reads::DifferenceSize,
            Jacobian::MaxRowsAtCompileTime,
            Reads::DifferenceMaxSize>;
        Result result(jacobian.rows(), jacobian.cols());
        result.noalias() = multiply(jacobian, resetJacobian(reads.space(), readMean, readStep));
        jacobian.const_cast_derived() = result;
    }
}

/** Calls \a use(y, cross, readJacobian) with \a resolved, a measurement resolved against the
 * prior's space, linearized by \a method, a linear or a sigma-point method, over the blocks read
 * \a reads, given as \a read with prior mean \a readMean, and its moments about the prior: an
 * Iterated update's iterate (decisions.md [iterated-update]) or a linearizedAbout() reference
 * [linearization-reference]. When \a moved, the blocks are at \f$ \mu_S \boxplus \delta_S \f$ with
 * \a readStep the \f$ \delta_S \f$, and otherwise at the prior mean; a sigma-point method spreads its
 * points there by \a spreadCov, in that point's chart, and a linear method reads none. \a f is the
 * model's function over the blocks and the call's inputs and \a stateRead the prior's
 * \f$ P_{xS} \f$, from updateResolved().
 *
 * The moments about the prior are those of the affine model \f$ A_S \delta_S \f$ in the prior's chart:
 * the Jacobian (\f$ \Sigma_{Sy}^T W^{-1} \f$ for a sigma-point method spread by \f$ W \f$) carried
 * to the prior's chart, the mean moved back by it to \f$ \delta = 0 \f$, the innovation covariance
 * \f$ A P_{SS} A^T + \Omega \f$, and the cross-covariance with the whole state \f$ P_{xS} A^T \f$, as
 * updateResolved() forms them for Linearized. Unmoved, these are the method's own update's moments.
 */
template <
    typename Space,
    typename Scalar,
    typename R,
    typename Method,
    typename Reads,
    typename Read,
    typename F,
    typename StateRead,
    typename ReadMean,
    typename ReadStep,
    typename SpreadCov,
    typename Use>
[[nodiscard]] decltype(auto)
linearizeReadsAt(
    R const & resolved,
    Method const & method,
    Reads const & reads,
    Read const & read,
    F const & f,
    StateRead const & stateRead,
    ReadMean const & readMean,
    bool moved,
    ReadStep const & readStep,
    SpreadCov const & spreadCov,
    Use const & use
)
{
    using ReadVector = typename MovedInput<Read>::VectorType;
    if constexpr (!isLinearMethod<Method>)
    {
        using ReadMatrix = SpreadCov;
        auto const input = RespreadInput<Read, ReadMatrix>{
            read,
            moved ? ReadVector{Emrys::plus(reads.space(), readMean, readStep)} : readMean,
            spreadCov
        };
        // The root form's H_S = Psi^T S^-1, a solve with the root that spread the points, and N + Omega,
        // its parts, so that spreadCov is neither factorized again nor subtracted from (decisions.md
        // [linearization-forms]).
        auto const predicted = transformMeasurement(resolved, method, f, input);
        using InnovationGaussian = std::remove_cvref_t<decltype(predicted.output())>;
        using InnovationMatrix = typename InnovationGaussian::MatrixType;
        using Predicted = OutputWithCross<Space, typename InnovationGaussian::SpaceType, JacobianForm, Scalar>;
        using ReadJacobianMatrix = MatrixOf<
            Scalar,
            InnovationGaussian::DifferenceDimensionAtCompileTime,
            Reads::DifferenceSize,
            InnovationGaussian::MaxDifferenceDimensionAtCompileTime,
            Reads::DifferenceMaxSize>;
        ReadJacobianMatrix jacobian = predicted.jacobian();
        InnovationMatrix cov = predicted.inputsCovariance() + predicted.unexplainedCovariance();
        InnovationMatrix const error = cov;
        typename InnovationGaussian::VectorType mean = predicted.output().mean();
        if (moved)
        {
            toPriorChart(reads, jacobian, readMean, readStep);
            auto back = typename InnovationGaussian::DifferenceVectorType{};
            back.noalias() = -multiply(jacobian, readStep);
            mean = Emrys::plus(predicted.output().space(), predicted.output().mean(), back);
        }
        auto cross = typename Predicted::CrossMatrix{};
        cross.noalias() = multiply(stateRead, jacobian.transpose());
        cov.noalias() += multiply(jacobian, std::as_const(cross)(reads.differenceIndices(), Eigen::placeholders::all));
        auto const y = InnovationGaussian{predicted.output().space(), std::move(mean), std::move(cov)};
        return use(y, cross, ReadLinearization{jacobian, reads.differenceIndices(), [&] {
                                                   return error;
                                               }});
    }
    else
    {
        auto const input =
            MovedInput<Read>{read, moved ? ReadVector{Emrys::plus(reads.space(), readMean, readStep)} : readMean};
        return linearizeMeasurement(
            resolved,
            method,
            f,
            input,
            [&](auto const & problem, auto const & linearized) -> decltype(auto) {
                using Problem = std::remove_cvref_t<decltype(problem)>;
                using Predicted = OutputWithCross<Space, typename Problem::OutputSpaceType, JacobianForm, Scalar>;
                Eigen::Index const readSize = reads.differenceIndices().size();
                Eigen::Index const offset = problem.differenceOffset(0);
                typename Problem::JacobianMatrix jacobian = linearized.jacobian;
                typename Problem::template OutputVector<Scalar> mean = linearized.mean;
                auto readJacobian = jacobian.template middleCols<Reads::DifferenceSize>(offset, readSize);
                if (moved)
                {
                    toPriorChart(reads, readJacobian, readMean, readStep);
                    auto back = typename Problem::template OutputDifferenceVector<Scalar>{};
                    back.noalias() = -multiply(readJacobian, readStep);
                    mean = Emrys::plus(problem.outputSpace(), linearized.mean, back);
                }
                auto cross = typename Predicted::CrossMatrix{};
                cross.noalias() = multiply(stateRead, readJacobian.transpose());
                auto const linearization =
                    ReadLinearization{readJacobian, reads.differenceIndices(), [&] {
                                          using Error = typename Problem::OutputMatrix;
                                          return inputsCovariance<Error>(resolved, problem, jacobian);
                                      }};
                if constexpr (R::InputCount > 0)
                {
                    typename Problem::CrossMatrix const joint =
                        multiply(problem.covarianceLower(), jacobian.transpose());
                    auto const y = PredictionMoments{problem.outputSpace(), mean, jacobian, joint};
                    return use(y, cross, linearization);
                }
                else
                {
                    auto const joint = std::as_const(cross)(reads.differenceIndices(), Eigen::placeholders::all);
                    auto const y = PredictionMoments{problem.outputSpace(), mean, jacobian, joint};
                    return use(y, cross, linearization);
                }
            }
        );
    }
}

/** The update of \a prior by \a resolved, a measurement resolved against the prior's space whose
 * method is Iterated and whose noise has one component, with the posterior put where \a sink says
 * (decisions.md [iterated-update]): the inner method's update at each iterate, as Iterated describes,
 * with the blocks read \a reads, given as \a read, the model's function over them \a f, and
 * \a stateRead, the prior's \f$ P_{xS} \f$, from updateResolved(). The last linearization's update
 * is the result, each branch reporting the iterations and whether they stopped by the tolerance.
 * The blocks the measurement considers stay at the prior's mean at every iterate, and keep their
 * covariance block in the spread (decisions.md [consider-states]).
 */
template <
    bool Statistics,
    typename Prior,
    typename R,
    typename Reads,
    typename Read,
    typename F,
    typename StateRead,
    typename Sink>
[[nodiscard]] decltype(auto)
iteratedUpdate(
    Prior const & prior,
    R const & resolved,
    Reads const & reads,
    Read const & read,
    F const & f,
    StateRead const & stateRead,
    Sink && sink
)
{
    using Scalar = typename Prior::Scalar;
    using Space = typename Prior::SpaceType;
    using Difference = typename GaussianParams<Space, Scalar>::DifferenceVectorType;
    using ReadVector = typename MovedInput<Read>::VectorType;
    using ReadDifference = typename GaussianParams<typename Reads::JointSpace, Scalar>::DifferenceVectorType;
    auto const & method = resolved.method();
    auto const scale = iterationScale(covarianceLowerOf(prior), Scalar(method.tolerance));
    ReadVector const readMean = read.mean();
    Difference delta = Difference::Zero(prior.differenceDimension());
    ReadDifference readStep = ReadDifference::Zero(reads.differenceIndices().size());
    std::size_t iteration = 0;
    bool converged = false;
    auto const considered = consideredRows<R>(prior.space());
    constexpr bool Considers = R::ConsiderType::Count > 0;

    // A sigma-point method spreads its points by the posterior covariance of the blocks read at the
    // iterate, in the iterate's chart, the prior's at first [sigma-point-sets]. A linear method reads no
    // covariance, so it keeps none.
    using Inner = std::remove_cvref_t<decltype(method.method)>;
    constexpr bool Spreads = !isLinearMethod<Inner>;
    using ReadMatrix = MatrixOf<
        Scalar,
        Reads::DifferenceSize,
        Reads::DifferenceSize,
        Reads::DifferenceMaxSize,
        Reads::DifferenceMaxSize>;
    using SpreadMatrix = std::conditional_t<Spreads, ReadMatrix, std::monostate>;
    auto priorReadCov = SpreadMatrix{};
    auto spreadCov = SpreadMatrix{};
    if constexpr (Spreads)
    {
        priorReadCov = read.covarianceLower();
        spreadCov = priorReadCov;
    }

    // The linearization at the iterate mu + delta, the prior mean at first, with its moments about the
    // prior (linearizeReadsAt()): by the inner method, Linearized at the iterate or a sigma-point method
    // over its posterior, spread by spreadCov.
    auto const atIterate = [&](auto const & use) -> decltype(auto) {
        return linearizeReadsAt<Space, Scalar>(
            resolved,
            method.method,
            reads,
            read,
            f,
            stateRead,
            readMean,
            iteration > 0,
            readStep,
            spreadCov,
            use
        );
    };

    // The Kalman step from the prior under one linearization: delta = K r, with K and r as
    // conditionOnPrediction() forms them for noise of one component. For a sigma-point method it
    // also sets spreadCov to the posterior covariance of the blocks read under that linearization,
    // P_SS - C_S S^-1 C_S^T with C_S the rows of the cross-covariance at them, carried to the
    // next iterate's chart. An S that is not positive definite is repaired as the measurement says,
    // and one that stays so ends the iteration with its error (decisions.md [innovation-cholesky]).
    auto const step =
        [&](auto const & y, auto const & cross, auto const &) -> std::expected<Difference, GaussianError> {
        using MeasurementSpace = std::remove_cvref_t<decltype(y.space())>;
        using InnovationVector = typename GaussianParams<MeasurementSpace, Scalar>::DifferenceVectorType;
        using InnovationMatrix = typename GaussianParams<MeasurementSpace, Scalar>::MatrixType;
        auto [residual, cov] = [&] {
            if constexpr (!R::HasAdditiveNoise)
            {
                return std::pair{
                    observedResidual<Scalar>(resolved, y.space(), y.mean()),
                    InnovationMatrix{y.covarianceLower()}
                };
            }
            else
            {
                auto const & noise = resolved.noise().distribution();
                return std::pair{
                    InnovationVector{
                        observedResidual<Scalar>(resolved, y.space(), Emrys::plus(y.space(), y.mean(), noise.mean()))
                    },
                    innovationCovariancePlus<InnovationMatrix>(y, noise.covarianceLower())
                };
            }
        }();
        auto const factored = innovationCholesky<typename R::InnovationRepair>(cov);
        if (!factored)
        {
            return std::unexpected{factored.error()};
        }
        auto const & llt = *factored;
        auto next = Difference{};
        next.resize(delta.size());
        next.noalias() = multiply(cross, llt.solve(residual));
        if constexpr (Considers)
        {
            zeroConsideredRows(next, considered);
        }
        if constexpr (Spreads)
        {
            auto const readCross = std::as_const(cross)(reads.differenceIndices(), Eigen::placeholders::all);
            spreadCov = priorReadCov;
            using ReadCross = internal::MatrixOf<
                Scalar,
                Reads::DifferenceSize,
                InnovationMatrix::RowsAtCompileTime,
                Reads::DifferenceMaxSize,
                InnovationMatrix::MaxRowsAtCompileTime>;
            spreadCov.noalias() -=
                multiply(internal::transposedSolve<ReadCross>(llt, readCross.transpose()), readCross.transpose());
            spreadCov = (Scalar(0.5) * (spreadCov + spreadCov.transpose())).eval();
            if constexpr (Considers)
            {
                // The considered blocks the model reads keep their covariance with each other.
                auto const & rows = reads.differenceIndices();
                auto const isConsidered = [&](Eigen::Index row) {
                    for (BlockRows const block : considered)
                    {
                        if (row >= block.start && row < block.start + block.size)
                        {
                            return true;
                        }
                    }
                    return false;
                };
                for (Eigen::Index i = 0; i < rows.size(); ++i)
                {
                    for (Eigen::Index j = 0; j <= i; ++j)
                    {
                        if (isConsidered(rows(i)) && isConsidered(rows(j)))
                        {
                            spreadCov(i, j) = priorReadCov(i, j);
                            spreadCov(j, i) = priorReadCov(i, j);
                        }
                    }
                }
            }
            if constexpr (!hasIdentityReset<typename Reads::JointSpace>)
            {
                ReadDifference const nextReadStep = next(reads.differenceIndices());
                auto const reset = resetJacobian(reads.space(), readMean, nextReadStep);
                spreadCov = multiply(multiply(reset, spreadCov), reset.transpose()).eval();
            }
        }
        return next;
    };

    // The update under the last linearization, reporting the iterations.
    // A step that failed gives its error instead. The error is passed through finish, whose type
    // MSVC can't name with decltype, since atIterate() holds a lambda.
    auto failure = std::optional<GaussianError>{};
    auto const finish = [&](auto const & y, auto const & cross, auto const & readJacobian) {
        using Result = decltype(conditionOnPrediction<Statistics>(prior, resolved, y, cross, readJacobian, sink));
        if (failure)
        {
            return Result{std::unexpected{*failure}};
        }
        auto result = conditionOnPrediction<Statistics>(prior, resolved, y, cross, readJacobian, sink);
        if (result)
        {
            for (auto & branch : result->branches)
            {
                branch.iterations = iteration + 1;
                branch.converged = converged;
            }
        }
        return result;
    };

    // A step that moved too little to count stops the iteration at the linearization after it.
    for (; iteration + 1 < method.maxIterations && !converged; ++iteration)
    {
        auto const next = atIterate(step);
        if (!next)
        {
            failure = next.error();
            break;
        }
        converged = withinScale(*next, delta, scale);
        delta = *next;
        readStep = delta(reads.differenceIndices());
    }
    return atIterate(finish);
}

}  // namespace internal

}  // namespace Emrys
