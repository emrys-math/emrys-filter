#pragma once

/** \file
 * Propagating a distribution through a function (decisions.md [method-per-measurement], [prediction-inputs],
 * [affine-coordinates], [returned-jacobians], [iterated-update], [sigma-point-sets],
 * [linearization-reference], [transform-policies]): transform(), which takes a measurement model's
 * policies or a measurement model itself, the withCross() mark on its arguments, and
 * linearizedAbout() for a step linearized about a point or a distribution the caller gives. The internal
 * namespace holds each method's computation of the output's moments and linearization. It includes what
 * it builds on: Linearization.h (what a transform returns), Condition.h (the Kalman update of a marked
 * input), Methods.h (the methods as policies), and TransformArgs.h (the arguments and the joint input).
 * Read ProcessModel.h and Measurement.h next.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/AutoDiff>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Condition.h>
#include <EmrysFilter/Models/Linearization.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/ResolvedReads.h>
#include <EmrysFilter/Models/TransformArgs.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \defgroup Transforms Transforms
 * \ingroup Models
 *
 * transform() pushes independent uncertain inputs through a function \f$ y = f(x) \boxplus v \f$
 * and returns the distribution of \f$ y \f$ as a %Gaussian (decisions.md [transform-policies]). A
 * process model maps a state to a later state; a measurement model maps a state to a measurement
 * in its measurement space, whose minus() forms the residual of an observation. The function and
 * its policies are a measurement model's: \a f takes <tt>f(x)</tt>, <tt>f(in, x)</tt>, or
 * <tt>f(in, out, x)</tt>, and the policies, each a call before the inputs, are reads(), writes()
 * or outputSpace(), affineIn(), and additiveNoise(), one %Gaussian \f$ v \f$. A measurement model is
 * a transform with its inputs left unbound, and <tt>transform(method, model, inputs...)</tt> takes
 * one in place of the function and its policies.
 *
 * The inputs reach \a f as one vector \f$ x \f$:
 *  - without reads(), \f$ x \f$ is the one uncertain input (see UncertainInput: a GaussianParams,
 *    a Gaussian, a marginal of one, or a SqrtGaussianParams), whole;
 *  - with reads(), each input is an uncertain input over a tagged space or sure<Tag>(value), a
 *    value known exactly, matched to the items of reads() by tag in any order, and \f$ x \f$ is a
 *    vector in the Composite of the items' names, read with <tt>block<Tag>(in, x)</tt>. Uncertain
 *    inputs are independent of each other, so their joint space is the concatenation of their
 *    spaces.
 *
 * The result is a %Gaussian in the output space. Marking one uncertain input with
 * withCross(), usually the state, also returns its cross-covariance with \a y (see
 * OutputWithCross), and condition() then conditions that input on an observation of \a y: the
 * Kalman update. The other uncertain inputs are nuisance parameters, such as noise or an
 * uncertain sensor position, whose correlations are not tracked; anything worth tracking
 * belongs in the state.
 *
 * The output space is outputSpace()'s, an instance or a function of the input space; else the
 * Composite of writes(); else the space of the additive noise; and otherwise Linear, sized by
 * \a f's return type.
 *
 * The method is any TransformMethod, made by a factory in \c methods (decisions.md
 * [policy-factories]). They differ in what they cost and what they ask of \a f; \f$ L \f$ is the
 * difference dimension of the joint uncertain input:
 *
 * | Method | Calls of \a f | Exact for the mean and covariance when \a f is |
 * |---|---|---|
 * | methods::linearized() | 1 | affine |
 * | methods::unscented() | \f$ 2L + 1 \f$ | affine, or with the defaults quadratic |
 * | methods::fourthMomentUnscented(), a user's UnscentedRule | \f$ 2L + 1 \f$ | affine, or one coordinate squared |
 * | methods::cubature() | \f$ 2L \f$ | affine; quadratic, for the mean |
 * | methods::sphericalSimplex() | \f$ L + 2 \f$ | affine; quadratic, for the mean |
 * | methods::gaussHermite\<N\>() | \f$ N^L \f$ | of degree \f$ 2N - 1 \f$ per coordinate, for the mean |
 * | a user's PointSet | its <tt>pointCount(L)</tt> | as its points integrate |
 *
 * Linearized calls \a f once with dual numbers carrying \f$ L \f$ derivatives, or with plain
 * scalars when \a f returns its Jacobian; every other method takes any callable. A point set over a
 * fixed \f$ L \f$ has at most 4096 points. A model's affineIn() factors the coordinates it names out of
 * every sigma-point method (decisions.md [affine-coordinates]), and methods::iterated() is a measurement update's
 * method, which transform() refuses (decisions.md [iterated-update]). For \f$ x^2 \f$ the
 * third-degree rules get the mean and miss the variance, which needs the fourth moment;
 * methods::unscented() gets it through \f$ \beta = 2 \f$ (see statisticalLinearization()). A
 * sigma-point rule is spread by the axes of methods::spread(), or by the default root otherwise
 * (decisions.md [sigma-point-axes]).
 *
 * To be differentiated by Linearized, \a f must be generic in its scalar type: take uncertain
 * input as <tt>const auto&</tt> and build outputs with ScalarOf. A sure input reaches \a f in the
 * input vector's scalar; captured constants may stay \c double, as Eigen mixes \c double with its
 * dual numbers, but a \c float filter holds its constants in \c float. An \a f that knows its
 * Jacobian returns it with its value instead, as withJacobian(), one matrix over the input's
 * differences, and need not be generic; the sigma-point methods take the value and ignore the
 * Jacobian, so one function serves every method.
 *
 * The rules for the inputs: all uncertain inputs share one scalar; at least one input is
 * uncertain; a marginal is an uncertain input in its tag's space. Two marginals of one %Gaussian
 * passed as two inputs are taken as independent, losing their correlation, so a state is passed
 * once and its blocks selected inside \a f. withCross() refuses a temporary, a marginal made in
 * the call included, so that nothing they mark dangles. A function's returned Eigen expression is
 * evaluated at the call, while the arguments it refers to are alive.
 *
 * The methods are the standard ones: the Kalman filter's affine case
 * (\ref Kalman1960 "Kalman (1960)"), the first-order linearization of the extended Kalman filter
 * (\ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)"), and the unscented
 * transform of \ref Julier2004 "Julier and Uhlmann (2004)"; \ref Sarkka2013 "Särkkä (2013)",
 * chapters 5 and 6, gives all three in one notation. The transform's job of carrying several
 * independent inputs and a marked cross-covariance through one function, so that a measurement
 * update is a conditioning of the joint, is how \ref Hertzberg2013 "Hertzberg et al. (2013)"
 * phrase the Kalman update on a manifold.
 *
 * \code
 * auto const rangeBearing = []<typename T>(auto const & state) {
 *   return Eigen::Matrix<T, 2, 1>{
 *     Eigen::numext::sqrt(state.template head<2>().squaredNorm()), Eigen::numext::atan2(state(1), state(0))
 *   };
 * };
 * auto const predicted = Emrys::transform(Emrys::methods::linearized(), rangeBearing,
 *                                         Emrys::outputSpace(RangeBearing{}), withCross(state));
 * auto const updated = Emrys::condition(state, predicted, z);
 * // updated->posterior; updated->logLikelihood() and updated->innovation.nis() when wanted, or
 * // updated.error() when the predicted covariance is not positive definite
 * \endcode
 */

namespace internal
{

/** What a linear method makes of a function: its output at the input means, normalized, and its
 * Jacobian in the joint input differences.
 */
template <typename Problem>
struct LinearizationAt
{
    typename Problem::template OutputVector<typename Problem::Scalar> mean;  ///< the output at the means
    typename Problem::JacobianMatrix jacobian;  ///< output differences by joint input differences
};

/** The linear core's last step: normalizes the output mean of \a result where it is, in the
 * LinearizationAt a linear method returns (decisions.md [borrowed-intermediates]). Its mean and its Jacobian over the
 * joint input then give \f$ \Sigma_{yy} = J \Sigma J^T \f$ and \f$ \Sigma_{xy} = \Sigma J^T \f$.
 */
template <typename Problem>
void
normalizeLinearization(Problem const & problem, LinearizationAt<Problem> & result)
{
    normalizeValues(problem.outputSpace(), result.mean);
}

/** Whether the function of \a Problem returns its Jacobian with its value (see WithJacobian). */
template <typename Problem>
inline constexpr bool returnsJacobian = isWithJacobian<
    std::remove_cvref_t<decltype(std::declval<Problem const &>().template returned<typename Problem::Scalar>(
        std::declval<typename Problem::template InputVector<typename Problem::Scalar> const &>()
    ))>>;

/** The dual number Linearized differentiates the function of \a Problem with: one derivative per
 * joint input difference coordinate, bounded as the input is, so that a bounded input's
 * derivatives stay off the heap.
 */
template <typename Problem>
using DualOf = Eigen::AutoDiffScalar<Eigen::Matrix<
    typename Problem::Scalar,
    Problem::InputDifferenceSize,
    1,
    Eigen::ColMajor,
    Problem::InputDifferenceMaxSize,
    1>>;

/** Whether the function of \a Problem, given dual numbers, returns them, as Linearized needs to
 * differentiate it: false for an \a f that takes a plain vector, which converts its argument, or
 * that builds its result from one.
 */
template <typename Problem, typename Dual = DualOf<Problem>>
inline constexpr bool returnsDuals = requires {
    requires std::same_as<typename std::remove_cvref_t<typename Problem::template Returned<Dual>>::Scalar, Dual>;
};

/** A linear method for a function that returns its Jacobian: one call at the means, its value
 * the mean and its blocks, one per uncertain input (or per argument, from an internal wrapper),
 * placed in the joint Jacobian, which they cover (decisions.md [returned-jacobians]).
 */
template <typename Problem>
[[nodiscard]] LinearizationAt<Problem>
linearizeReturned(Problem const & problem)
{
    using Scalar = typename Problem::Scalar;
    using JacobianMatrix = typename Problem::JacobianMatrix;
    auto returned = problem.template returned<Scalar>(problem.mean());
    using Blocks = decltype(returned.jacobian);
    auto result = LinearizationAt<Problem>{
        std::move(returned.value),
        JacobianMatrix(problem.outputDifferenceSize(), problem.inputDifferenceSize())
    };
    EMRYS_FILTER_ASSERT(
        result.mean.size() == valueDimension(problem.outputSpace()) && "the output does not fit the output space"
    );
    problem.forEachUncertain([&]<std::size_t I>(auto const & input) {
        auto const & block = [&]() -> auto const & {
            if constexpr (isArgumentJacobians<Blocks>)
            {
                return std::get<I>(returned.jacobian.blocks);
            }
            else
            {
                return jacobianBlock<Problem::ArgInfo::ordinal[I]>(returned.jacobian);
            }
        }();
        EMRYS_FILTER_ASSERT(block.rows() == problem.outputDifferenceSize() && block.cols() == input.differenceDimension() &&
                        "a Jacobian block must have a row per output difference coordinate and a column per input "
                        "difference coordinate");
        result.jacobian.middleCols(problem.differenceOffset(I), input.differenceDimension()) = block;
    });
    normalizeLinearization(problem, result);
    return result;
}

/** Differentiates \a f at the means with respect to the joint difference \f$ \delta \f$, at
 * \f$ \mu \boxplus \delta \f$ (see Transform::stepInputs()), and its output as the difference
 * from its value at the means in a space with its own minus().
 */
template <typename Problem>
[[nodiscard]] LinearizationAt<Problem>
linearizeByAutoDiff(Problem const & problem)
{
    using Scalar = typename Problem::Scalar;
    using Dual = DualOf<Problem>;
    using OutputSpace = typename Problem::OutputSpaceType;
    Eigen::Index const n = problem.inputDifferenceSize();
    typename Problem::template InputVector<Scalar> const mu{problem.mean()};
    typename Problem::template InputDifferenceVector<Dual> delta;
    delta.resize(n);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        delta(i) = Dual{Scalar(0), int(n), int(i)};
    }
    typename Problem::template InputVector<Dual> const x{problem.stepInputs(mu.template cast<Dual>(), delta)};
    typename Problem::template OutputVector<Dual> const y{problem.template evaluate<Dual>(x)};

    auto result = LinearizationAt<Problem>{
        Problem::template OutputVector<Scalar>::Zero(y.size()),
        Problem::JacobianMatrix::Zero(problem.outputDifferenceSize(), n)
    };
    for (Eigen::Index r = 0; r < y.size(); ++r)
    {
        result.mean(r) = y(r).value();
    }
    auto const deviation = [&] {
        if constexpr (HasMinus<OutputSpace>)
        {
            return typename Problem::template OutputDifferenceVector<Dual>{
                Emrys::minus(problem.outputSpace(), y, result.mean.template cast<Dual>())
            };
        }
        else
        {
            return y;
        }
    }();
    EMRYS_FILTER_ASSERT(deviation.size() == result.jacobian.rows());
    for (Eigen::Index r = 0; r < deviation.size(); ++r)
    {
        // An output that does not depend on the inputs may carry no derivatives at all.
        if (deviation(r).derivatives().size() > 0)
        {
            result.jacobian.row(r) = deviation(r).derivatives().transpose();
        }
    }
    normalizeLinearization(problem, result);
    return result;
}

/** The Linearized method: the Jacobian \a f returns, or by automatic differentiation. */
template <typename Problem>
[[nodiscard]] LinearizationAt<Problem>
linearizeAt(methods::Linearized const &, Problem const & problem)
{
    if constexpr (returnsJacobian<Problem>)
    {
        return linearizeReturned(problem);
    }
    else
    {
        static_assert(
            returnsDuals<Problem>,
            "methods::linearized() differentiates f automatically, so f must take its uncertain inputs as const auto& "
            "and build its result with their scalar, which it may take as its template parameter, "
            "[]<typename T>(auto const & x), or return its Jacobian with withJacobian()"
        );
        if constexpr (returnsDuals<Problem>)
        {
            return linearizeByAutoDiff(problem);
        }
        else
        {
            std::unreachable();
        }
    }
}

/** A linear method's result, the Jacobian form (decisions.md [linearization-forms]): its mean and its Jacobian
 * over the joint input, from linearizeAt(), which Transform::assemble() splits by input.
 */
template <typename Method, typename Problem>
    requires isLinearMethod<Method>
[[nodiscard]] LinearizationAt<Problem>
transformResult(Method const & method, Problem const & problem)
{
    return linearizeAt(method, problem);
}

/** An Iterated method in a transform, refused at compile time: a transform has no posterior to
 * relinearize at (decisions.md [iterated-update]).
 */
template <typename Method, typename Scalar, typename Problem>
[[nodiscard]] typename Problem::RootResult
transformResult(methods::Iterated<Method, Scalar> const & /*method*/, Problem const & /*problem*/)
{
    static_assert(
        sizeof(Method) == 0,
        "Iterated is a measurement update's method: a transform or a prediction has no posterior to relinearize at, "
        "and an update given a block does not support it yet; use its inner method there"
    );
    return {};
}

/** Weights of the \f$ 2n + 1 \f$ sigma points of an unscented transform over \a n coordinates. */
template <typename Scalar>
struct SigmaWeights
{
    Scalar scale;      ///< \f$ \sqrt{n + \lambda} \f$: the points' distance from the center, in roots
    Scalar center;     ///< mean weight of the center point
    Scalar centerCov;  ///< covariance weight of the center point
    Scalar point;      ///< weight of every other point
};

/** The sigma-point weights of the UnscentedRule \a Rule over \a N coordinates, from its parameters
 * there, computed by the compiler in \c double as sigmaWeights() computes them at run time, and
 * held as \a Scalar constants (takesDoubleConstants).
 */
template <typename Scalar, UnscentedRule Rule, int N>
[[nodiscard]] consteval SigmaWeights<Scalar>
unscentedWeights()
{
    if constexpr (N == 0)
    {
        return SigmaWeights<Scalar>{Scalar(0), Scalar(1), Scalar(1), Scalar(0)};
    }
    else
    {
        constexpr double n = N;
        constexpr UnscentedParams<double> params = Rule::template params<double>(N);
        constexpr double alphaSquared = params.alpha * params.alpha;
        constexpr double lambda = alphaSquared * (n + params.kappa) - n;
        static_assert(n + lambda > 0, "the sigma-point spread must be positive");
        constexpr double center = lambda / (n + lambda);
        return SigmaWeights<Scalar>{
            Scalar(compileTimeSqrt(n + lambda)),
            Scalar(center),
            Scalar(center + 1 - alphaSquared + params.beta),
            Scalar(1 / (2 * (n + lambda)))
        };
    }
}

/** \returns the sigma-point weights of \a method over \a n coordinates; over none, the center
 * alone. The weights are those of \ref Julier2004 "Julier and Uhlmann (2004)", as given in
 * \ref Sarkka2013 "Särkkä (2013)", §5.5, equations (5.74) to (5.77):
 * \f$ W_0^m = \lambda / (n + \lambda) \f$, \f$ W_0^c = W_0^m + 1 - \alpha^2 + \beta \f$, and
 * \f$ W_i = 1 / (2 (n + \lambda)) \f$ otherwise, with the parameters the rule gives over \a n. Over
 * a number of coordinates \a N fixed at compile time they are the constants unscentedWeights() for
 * a scalar that takes them; otherwise the rule's parameters and the weights are computed at run
 * time in \a Scalar.
 */
template <typename Scalar, int N, UnscentedRule Rule>
[[nodiscard]] SigmaWeights<Scalar>
sigmaWeights(Rule const & /*method*/, Eigen::Index n)
{
    if constexpr (N != Eigen::Dynamic && takesDoubleConstants<Scalar>)
    {
        EMRYS_FILTER_ASSERT(n == N);
        return unscentedWeights<Scalar, Rule, N>();
    }
    else
    {
        if (n == 0)
        {
            return {Scalar(0), Scalar(1), Scalar(1), Scalar(0)};
        }
        UnscentedParams<Scalar> const params = Rule::template params<Scalar>(n);
        Scalar const alphaSquared = params.alpha * params.alpha;
        Scalar const lambda = alphaSquared * (Scalar(n) + params.kappa) - Scalar(n);
        EMRYS_FILTER_ASSERT(Scalar(n) + lambda > Scalar(0) && "the sigma-point spread must be positive");
        Scalar const center = lambda / (Scalar(n) + lambda);
        return {
            Eigen::numext::sqrt(Scalar(n) + lambda),
            center,
            center + Scalar(1) - alphaSquared + params.beta,
            Scalar(1) / (Scalar(2) * (Scalar(n) + lambda))
        };
    }
}

/** The whitened cross-covariance \f$ \Psi \f$ of a set of unit points with the output: a row per
 * coordinate of the points, \a Rows of them, at most \a MaxRows, and a column per output difference.
 */
template <typename Problem, int Rows, int MaxRows>
using WhitenedOf =
    MatrixOf<typename Problem::Scalar, Rows, Problem::OutputDifferenceSize, MaxRows, Problem::OutputDifferenceMaxSize>;

/** Output moments of the sigma points as the parts of OutputWithCross, and the deviations about the
 * mean that give them: the square-root forms triangularize the deviations in place of the covariance
 * (decisions.md [gaussian-forms]).
 */
template <typename Problem, typename Deviations>
struct SigmaMoments
{
    typename Problem::template OutputVector<typename Problem::Scalar> mean;  ///< the weighted tangent-space mean

    /// \f$ \Psi \f$, a row per axis of the root
    WhitenedOf<Problem, Deviations::ColsAtCompileTime, Deviations::MaxColsAtCompileTime> whitened;

    typename Problem::OutputMatrix unexplained;  ///< \f$ \Omega \f$, the covariance of the fit's residuals
    Deviations plus;                             ///< the plus points' deviations from the mean
    Deviations minus;                            ///< the minus points' deviations from the mean
    typename Problem::template OutputDifferenceVector<typename Problem::Scalar> center;  ///< the center point's
};

/** \returns the moments of the outputs at the center point, \a center, and at the points either
 * side of it, \a plus and \a minus: a tangent-space mean about the center point, then deviations
 * about that mean, and the parts of their covariance (decisions.md [linearization-forms]). The unit points are
 * \f$ \pm \sqrt{n + \lambda} \, e_j \f$ and the center, so \f$ \Psi \f$'s row \f$ j \f$ is
 * \f$ W \sqrt{n + \lambda} (\Delta_j^+ - \Delta_j^-)^T \f$, and the residual of each pair is its mean
 * deviation \f$ (\Delta_j^+ + \Delta_j^-) / 2 \f$ and of the center its deviation, each of the
 * curvature's size, so that \f$ \Omega \f$ is recovered by no subtraction. A root's zero column
 * moves neither point of its pair, so its row of \f$ \Psi \f$ is zero.
 */
template <typename Problem, typename Samples>
[[nodiscard]] auto
sigmaMoments(
    Problem const & problem,
    SigmaWeights<typename Problem::Scalar> const & weights,
    typename Problem::template OutputVector<typename Problem::Scalar> const & center,
    Samples const & plusPoints,
    Samples const & minusPoints
)
{
    using Scalar = typename Problem::Scalar;
    using OutputVector = typename Problem::template OutputVector<Scalar>;
    using OutputDeviations =
        typename Problem::template OutputDeviations<Samples::ColsAtCompileTime, Samples::MaxColsAtCompileTime>;
    auto const & space = problem.outputSpace();
    OutputDeviations plusDeviations = Emrys::minus(space, plusPoints, center);
    OutputDeviations minusDeviations = Emrys::minus(space, minusPoints, center);
    OutputVector mean =
        Emrys::plus(space, center, weights.point * (plusDeviations.rowwise().sum() + minusDeviations.rowwise().sum()));

    typename Problem::template OutputDifferenceVector<Scalar> const centerDeviation = Emrys::minus(space, center, mean);
    plusDeviations = Emrys::minus(space, plusPoints, mean);
    minusDeviations = Emrys::minus(space, minusPoints, mean);

    using Result = SigmaMoments<Problem, OutputDeviations>;
    OutputDeviations const residual = Scalar(0.5) * (plusDeviations + minusDeviations);
    decltype(Result::whitened) whitened =
        (weights.point * weights.scale) * (plusDeviations - minusDeviations).transpose();
    typename Problem::OutputMatrix unexplained = weights.centerCov * centerDeviation * centerDeviation.transpose();
    if constexpr (Samples::ColsAtCompileTime != 0)
    {
        // A product over no points has an inner size of zero, which MSVC's Eigen refuses to block.
        unexplained.noalias() += (Scalar(2) * weights.point) * multiply(residual, residual.transpose());
    }
    unexplained = (Scalar(0.5) * (unexplained + unexplained.transpose())).eval();
    return Result{
        std::move(mean),
        std::move(whitened),
        std::move(unexplained),
        std::move(plusDeviations),
        std::move(minusDeviations),
        centerDeviation
    };
}

/** What a sigma-point method gives a square-root form (decisions.md [gaussian-forms]): the output mean, and
 * a rectangular root of the output covariance before any noise, the weighted deviations of the
 * \f$ 2L \f$ points either side of the mean and then the center point's, each scaled by the square
 * root of its weight. The center point's covariance weight may be negative, and then its column
 * is scaled by the root of its magnitude and must be subtracted rather than added. A point set's
 * root has a column per point, in the set's order, and no negative weight.
 */
template <
    typename Problem,
    int Points_ =
        Problem::InputDifferenceSize == Eigen::Dynamic ? Eigen::Dynamic : 2 * Problem::InputDifferenceSize + 1,
    int MaxPoints_ =
        Problem::InputDifferenceMaxSize == Eigen::Dynamic ? Eigen::Dynamic : 2 * Problem::InputDifferenceMaxSize + 1>
struct SigmaRoot
{
    /// the deviations' columns: for Unscented, the plus points, the minus points, then the center
    static constexpr int Points = Points_;

    /// a bound on them
    static constexpr int MaxPoints = MaxPoints_;

    /// the root's type
    using RootMatrix = typename Problem::template OutputDeviations<Points, MaxPoints>;

    typename Problem::template OutputVector<typename Problem::Scalar> mean;  ///< the weighted tangent-space mean
    RootMatrix root;  ///< the weighted deviations, \f$ R R^T \f$ the covariance when the center weight is not negative
    bool centerNegative = false;  ///< whether the center's column is to be subtracted
};

/** \a f's outputs at an unscented transform's sigma points: at the center, and at the points either
 * side of it along each column of the spread.
 */
template <typename Problem>
struct SigmaOutputs
{
    typename Problem::template OutputVector<typename Problem::Scalar> center;  ///< at the center
    typename Problem::template OutputSamples<> plus;   ///< at \f$ \mu \boxplus S_j \f$, one per column
    typename Problem::template OutputSamples<> minus;  ///< at \f$ \mu \boxplus (-S_j) \f$, one per column
};

/** \returns \a f's outputs at the mean \a mu and at \f$ \mu \boxplus \pm S_j \f$ for each column of
 * \a spread, \f$ S \f$, in one evaluation of the points (Transform::evaluateEach()): the center
 * first, then the two points of each column in turn, the order in which a function that is not
 * batched is called
 */
template <typename Problem, typename Spread>
[[nodiscard]] SigmaOutputs<Problem>
evaluateSigmaPoints(
    Problem const & problem,
    typename Problem::template InputVector<typename Problem::Scalar> const & mu,
    Spread const & spread
)
{
    using Scalar = typename Problem::Scalar;
    using InputVector = typename Problem::template InputVector<Scalar>;
    constexpr int N = Problem::InputDifferenceSize;
    constexpr int NMax = Problem::InputDifferenceMaxSize;
    using Outputs = typename Problem::template OutputSamples<
        N == Eigen::Dynamic ? Eigen::Dynamic : 2 * N + 1,
        NMax == Eigen::Dynamic ? Eigen::Dynamic : 2 * NMax + 1>;
    Eigen::Index const n = spread.cols();
    auto outputs = Outputs{};
    problem.evaluateEach(
        2 * n + 1,
        [&](Eigen::Index k) -> InputVector {
            if (k == 0)
            {
                return mu;
            }
            Eigen::Index const j = (k - 1) / 2;
            return k % 2 == 1 ? problem.plusInputs(mu, spread.col(j)) : problem.plusInputs(mu, -spread.col(j));
        },
        outputs
    );
    auto result = SigmaOutputs<Problem>{};
    result.center = outputs.col(0);
    // Sized by resize(), since braces would set the coefficients of a fixed-size vector.
    result.plus.resize(outputs.rows(), n);
    result.minus.resize(outputs.rows(), n);
    // Over no coordinates there are no plus and minus points, at compile time.
    if constexpr (N != 0)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            result.plus.col(j) = outputs.col(1 + 2 * j);
            result.minus.col(j) = outputs.col(2 + 2 * j);
        }
    }
    return result;
}

/** An UnscentedRule's sigma points over a root of the joint covariance (Transform::root()),
 * kept as weighted deviations rather than summed into a covariance (see SigmaRoot). The points
 * and the mean are those of transformResult(); the square-root unscented filter of
 * \ref VanDerMerwe2001 "van der Merwe and Wan (2001)" triangularizes them.
 */
template <UnscentedRule Rule, typename Problem, typename A = roots::DefaultAxes>
[[nodiscard]] SigmaRoot<Problem>
transformRoot(Rule const & method, Problem const & problem, A const & axes = {})
{
    using Scalar = typename Problem::Scalar;
    using Result = SigmaRoot<Problem>;
    Eigen::Index const n = problem.inputDifferenceSize();
    SigmaWeights<Scalar> const weights = sigmaWeights<Scalar, Problem::InputDifferenceSize>(method, n);

    typename Problem::template InputVector<Scalar> const mu{problem.mean()};
    typename Problem::InputMatrix const spread = weights.scale * problem.root(axes);
    auto const outputs = evaluateSigmaPoints(problem, mu, spread);

    auto moments = sigmaMoments(problem, weights, outputs.center, outputs.plus, outputs.minus);
    Eigen::Index const m = problem.outputDifferenceSize();
    Scalar const pointRoot = Eigen::numext::sqrt(weights.point);
    Scalar const centerRoot = Eigen::numext::sqrt(Eigen::numext::abs(weights.centerCov));
    typename Result::RootMatrix root(m, 2 * n + 1);
    root.leftCols(n) = pointRoot * moments.plus;
    root.middleCols(n, n) = pointRoot * moments.minus;
    root.col(2 * n) = centerRoot * moments.center;
    return Result{std::move(moments.mean), std::move(root), weights.centerCov < Scalar(0)};
}

/** An UnscentedRule's method, Unscented's: \a f at the mean and at \f$ \mu \boxplus \pm \sqrt{n + \lambda} \, S_j \f$
 * for each column of a root \f$ S \f$ of the joint covariance, then sigmaMoments(), in the root form:
 * \f$ S \f$, \f$ \Psi \f$, and \f$ \Omega \f$.
 */
template <UnscentedRule Rule, typename Problem, typename A = roots::DefaultAxes>
[[nodiscard]] typename Problem::RootResult
transformResult(Rule const & method, Problem const & problem, A const & axes = {})
{
    using Scalar = typename Problem::Scalar;
    Eigen::Index const n = problem.inputDifferenceSize();
    SigmaWeights<Scalar> const weights = sigmaWeights<Scalar, Problem::InputDifferenceSize>(method, n);

    typename Problem::template InputVector<Scalar> const mu{problem.mean()};
    typename Problem::InputMatrix root = problem.root(axes);
    typename Problem::InputMatrix const spread = weights.scale * root;
    auto const outputs = evaluateSigmaPoints(problem, mu, spread);

    auto moments = sigmaMoments(problem, weights, outputs.center, outputs.plus, outputs.minus);
    return {
        std::move(moments.mean),
        std::move(root),
        typename Problem::CrossMatrix{std::move(moments.whitened)},
        std::move(moments.unexplained)
    };
}

/** \returns the number of points of the point set \a Points over \a n coordinates, a compile-time
 * size: \c Eigen::Dynamic for a dynamic \a n, and the set's pointCount() otherwise, capped one
 * past MaxPointCount. Over no coordinates every set is the mean alone.
 */
template <PointSet Points>
[[nodiscard]] consteval int
pointCount(int n)
{
    if (n == Eigen::Dynamic)
    {
        return Eigen::Dynamic;
    }
    if (n == 0)
    {
        return 1;
    }
    return int(std::min<Eigen::Index>(Points::pointCount(n), MaxPointCount + 1));
}

/** \returns the number of points of the point set \a Points over \a n coordinates, at run time */
template <PointSet Points>
[[nodiscard]] Eigen::Index
pointCountAt(Eigen::Index n)
{
    return n == 0 ? Eigen::Index(1) : Points::pointCount(n);
}

/** The points of a point set over \a N coordinates, at most \a NMax, in the coordinates of a root
 * of the covariance, one per column, with their weights, which serve the mean and the covariance
 * alike and sum to one.
 */
template <typename Scalar, typename Points, int N, int NMax>
struct UnitPoints
{
    static constexpr int Count = pointCount<Points>(N);        ///< the points
    static constexpr int MaxCount = pointCount<Points>(NMax);  ///< a bound on them

    static_assert(
        NMax == Eigen::Dynamic || MaxCount <= std::max(2 * NMax + 2, MaxPointCount),
        "a point set over this many coordinates has more than MaxPointCount (4096) points; declare the "
        "coordinates the model is affine in with affineIn(), or use a set with fewer points, such as a Gauss-Hermite "
        "rule of a lower order"
    );

    MatrixOf<Scalar, N, Count, NMax, MaxCount> points;  ///< \f$ \xi_j \f$, one per column
    MatrixOf<Scalar, Count, 1, MaxCount, 1> weights;    ///< \f$ w_j \f$
    Eigen::Index reference;  ///< the point of greatest weight, the first of any tie, which the mean is taken about
};

/** \returns the points of the PointSet \a set over \a n coordinates, with their weights and the
 * point the mean is taken about, as its unitPoints() fills them into matrices sized for \a n; over
 * no coordinates the mean alone, one point of weight one
 */
template <typename Scalar, int N, int NMax, PointSet Points>
[[nodiscard]] UnitPoints<Scalar, Points, N, NMax>
unitPoints(Points const & /*set*/, Eigen::Index n)
{
    using Result = UnitPoints<Scalar, Points, N, NMax>;
    Eigen::Index const count = pointCountAt<Points>(n);
    auto result = Result{};
    result.points.resize(n, count);
    result.weights.resize(count);
    result.reference = 0;
    if (n == 0)
    {
        result.weights(0) = Scalar(1);
        return result;
    }
    result.reference = Points::unitPoints(n, result.points, result.weights);
    EMRYS_FILTER_ASSERT(
        result.reference >= 0 && result.reference < count
        && "the point the mean is taken about must be one of the points"
    );
    return result;
}

/** Output moments of a point set's outputs as the parts of OutputWithCross, and their deviations
 * from the mean.
 */
template <typename Problem, typename Deviations, typename Whitened>
struct PointMoments
{
    typename Problem::template OutputVector<typename Problem::Scalar> mean;  ///< the weighted tangent-space mean
    Whitened whitened;                           ///< \f$ \Psi \f$, a row per coordinate of the unit points
    typename Problem::OutputMatrix unexplained;  ///< \f$ \Omega \f$, the covariance of the fit's residuals
    Deviations deviations;                       ///< each point's output's deviation from the mean, one per column
};

/** \returns the moments of the outputs \a outputs of a point set's unit points \a points, with weights
 * \a weights, spread by the root \a root: a tangent-space mean about the output of point \a reference,
 * then the parts of the weighted covariance of the deviations about that mean (decisions.md [linearization-forms]), the
 * regression's \f$ \Psi = \sum_i w_i \xi_i \Delta y_i^T \f$, and the residuals'
 * \f$ \Omega = \sum_i w_i e_i e_i^T \f$ with \f$ e_i = \Delta y_i - \Psi^T \xi_i \f$, whose sum
 * \f$ \Psi^T \Psi + \Omega \f$ the PointSet contract's mean zero and identity covariance make the
 * covariance, with nothing recovered by subtraction.
 *
 * Where \a root has a zero column, as a semidefinite input's does, the points do not move the input
 * along it, so \f$ \Psi \f$'s row there is set to zero, which keeps \f$ \Psi^T \Psi \f$ the explained
 * part, and the residuals take what the outputs moved with that coordinate of the points: a symmetric
 * set's moved nothing, but a set such as the spherical simplex's can.
 */
template <typename Problem, typename Samples, typename Points, typename Weights, typename Root>
[[nodiscard]] auto
pointMoments(
    Problem const & problem,
    Points const & points,
    Weights const & weights,
    Eigen::Index reference,
    Samples const & outputs,
    Root const & root
)
{
    using Scalar = typename Problem::Scalar;
    using OutputVector = typename Problem::template OutputVector<Scalar>;
    using OutputDeviations =
        typename Problem::template OutputDeviations<Samples::ColsAtCompileTime, Samples::MaxColsAtCompileTime>;
    auto const & space = problem.outputSpace();
    OutputVector const center = outputs.col(reference);
    OutputDeviations deviations = Emrys::minus(space, outputs, center);
    typename Problem::template OutputDifferenceVector<Scalar> step;
    step.noalias() = multiply(deviations, weights);
    OutputVector mean = Emrys::plus(space, center, step);
    deviations = Emrys::minus(space, outputs, mean);
    using Whitened = WhitenedOf<Problem, Points::RowsAtCompileTime, Points::MaxRowsAtCompileTime>;
    Whitened whitened = multiply(points * weights.asDiagonal(), deviations.transpose());
    EMRYS_FILTER_ASSERT(root.cols() == whitened.rows() && "the root must have a column per coordinate of the points");
    for (Eigen::Index j = 0; j < root.cols(); ++j)
    {
        if ((root.col(j).array() == Scalar(0)).all())
        {
            whitened.row(j).setZero();
        }
    }
    OutputDeviations residuals = deviations;
    residuals.noalias() -= multiply(whitened.transpose(), points);
    typename Problem::OutputMatrix unexplained = multiply(residuals * weights.asDiagonal(), residuals.transpose());
    unexplained = (Scalar(0.5) * (unexplained + unexplained.transpose())).eval();
    return PointMoments<Problem, OutputDeviations, Whitened>{
        std::move(mean),
        std::move(whitened),
        std::move(unexplained),
        std::move(deviations)
    };
}

/** \returns \a f's outputs at the points of \a unit spread by \a spread about the means:
 * \f$ \mu \boxplus S \xi_j \f$, one per column, in one evaluation of the points
 * (Transform::evaluateEach())
 */
template <typename Problem, typename Unit, typename Spread>
[[nodiscard]] auto
evaluatePoints(Problem const & problem, Unit const & unit, Spread const & spread)
{
    using Scalar = typename Problem::Scalar;
    using OutputSamples = typename Problem::template OutputSamples<Unit::Count, Unit::MaxCount>;
    typename Problem::template InputVector<Scalar> const mu{problem.mean()};
    auto outputs = OutputSamples{};
    problem.evaluateEach(
        unit.weights.size(),
        [&](Eigen::Index j) {
            return problem.plusInputs(mu, spread.col(j));
        },
        outputs
    );
    return outputs;
}

/** A point set's points over a root of the joint covariance (Transform::root()), kept as weighted
 * deviations rather than summed into a covariance (see SigmaRoot), for the square-root forms:
 * the points and the mean are those of transformResult().
 */
template <typename Points, typename Problem, typename A = roots::DefaultAxes>
    requires isPointSet<Points>
[[nodiscard]] auto
transformRoot(Points const & set, Problem const & problem, A const & axes = {})
{
    using Scalar = typename Problem::Scalar;
    using Unit = UnitPoints<Scalar, Points, Problem::InputDifferenceSize, Problem::InputDifferenceMaxSize>;
    using Result = SigmaRoot<Problem, Unit::Count, Unit::MaxCount>;
    using Spread =
        MatrixOf<Scalar, Problem::InputDifferenceSize, Unit::Count, Problem::InputDifferenceMaxSize, Unit::MaxCount>;
    Unit const unit = unitPoints<Scalar, Problem::InputDifferenceSize, Problem::InputDifferenceMaxSize>(
        set,
        problem.inputDifferenceSize()
    );
    typename Problem::InputMatrix const inputRoot = problem.root(axes);
    auto spread = Spread{};
    spread.resize(unit.points.rows(), unit.points.cols());
    spread.noalias() = multiply(inputRoot, unit.points);
    auto moments = pointMoments(
        problem,
        unit.points,
        unit.weights,
        unit.reference,
        evaluatePoints(problem, unit, spread),
        inputRoot
    );
    typename Result::RootMatrix root = moments.deviations * unit.weights.cwiseSqrt().asDiagonal();
    return Result{std::move(moments.mean), std::move(root), false};
}

/** A point set (Cubature, SphericalSimplex, GaussHermite): \a f at \f$ \mu \boxplus S \xi_j \f$
 * for its points \f$ \xi_j \f$ and a root \f$ S \f$ of the joint covariance, then pointMoments(), in
 * the root form: \f$ S \f$, \f$ \Psi = \sum_j w_j \xi_j \, (y_j \boxminus \bar y)^T \f$, and
 * \f$ \Omega \f$.
 */
template <typename Points, typename Problem, typename A = roots::DefaultAxes>
    requires isPointSet<Points>
[[nodiscard]] typename Problem::RootResult
transformResult(Points const & set, Problem const & problem, A const & axes = {})
{
    using Scalar = typename Problem::Scalar;
    using Unit = UnitPoints<Scalar, Points, Problem::InputDifferenceSize, Problem::InputDifferenceMaxSize>;
    using Spread =
        MatrixOf<Scalar, Problem::InputDifferenceSize, Unit::Count, Problem::InputDifferenceMaxSize, Unit::MaxCount>;
    Unit const unit = unitPoints<Scalar, Problem::InputDifferenceSize, Problem::InputDifferenceMaxSize>(
        set,
        problem.inputDifferenceSize()
    );
    typename Problem::InputMatrix root = problem.root(axes);
    auto spread = Spread{};
    spread.resize(unit.points.rows(), unit.points.cols());
    spread.noalias() = multiply(root, unit.points);
    auto moments =
        pointMoments(problem, unit.points, unit.weights, unit.reference, evaluatePoints(problem, unit, spread), root);
    return {
        std::move(moments.mean),
        std::move(root),
        typename Problem::CrossMatrix{std::move(moments.whitened)},
        std::move(moments.unexplained)
    };
}

/** A rule spread by axes of its own (methods::Spread): its rule's points over the root its axes take. */
template <typename Rule, typename A, typename Problem>
[[nodiscard]] typename Problem::RootResult
transformResult(methods::Spread<Rule, A> const & method, Problem const & problem)
{
    return transformResult(method.rule, problem, method.axes);
}

/** A rule spread by axes of its own, kept as weighted deviations (see SigmaRoot). */
template <typename Rule, typename A, typename Problem>
[[nodiscard]] auto
transformRoot(methods::Spread<Rule, A> const & method, Problem const & problem)
{
    return transformRoot(method.rule, problem, method.axes);
}

/** \returns \a f at \a x, a point of the joint input, with its Jacobian in the joint input's difference
 * coordinates \a affine written into \a columns, a column per coordinate, in one call: the columns
 * of the Jacobian \a f returns, or by automatic differentiation in those coordinates alone
 * (decisions.md [affine-coordinates])
 */
template <typename Problem, typename X, typename Affine, typename Columns>
[[nodiscard]] typename Problem::template OutputVector<typename Problem::Scalar>
valueWithAffineJacobian(Problem const & problem, X const & x, Affine const & affine, Columns && columns)
{
    using Scalar = typename Problem::Scalar;
    using OutputVector = typename Problem::template OutputVector<Scalar>;
    using Index = Eigen::Index;
    Index const l = affine.size();
    if constexpr (returnsJacobian<Problem>)
    {
        auto returned = problem.template returned<Scalar>(x);
        using Blocks = decltype(returned.jacobian);
        problem.forEachUncertain([&]<std::size_t I>(auto const & input) {
            auto const & block = [&]() -> auto const & {
                if constexpr (isArgumentJacobians<Blocks>)
                {
                    return std::get<I>(returned.jacobian.blocks);
                }
                else
                {
                    return jacobianBlock<Problem::ArgInfo::ordinal[I]>(returned.jacobian);
                }
            }();
            Index const start = problem.differenceOffset(I);
            Index const size = input.differenceDimension();
            for (Index j = 0; j < l; ++j)
            {
                if (affine(j) >= start && affine(j) < start + size)
                {
                    columns.col(j) = block.col(affine(j) - start);
                }
            }
        });
        OutputVector y = std::move(returned.value);
        EMRYS_FILTER_ASSERT(
            y.size() == valueDimension(problem.outputSpace()) && "the output does not fit the output space"
        );
        return y;
    }
    else
    {
        using Derivatives =
            Eigen::Matrix<Scalar, Affine::SizeAtCompileTime, 1, Eigen::ColMajor, Affine::MaxSizeAtCompileTime, 1>;
        using Dual = Eigen::AutoDiffScalar<Derivatives>;
        static_assert(
            returnsDuals<Problem, Dual>,
            "a function that a sigma-point method factors affine coordinates out of returns its Jacobian, or is "
            "generic "
            "in its scalar, to be differentiated in those coordinates"
        );
        using OutputSpace = typename Problem::OutputSpaceType;
        Index const total = problem.inputDifferenceSize();
        typename Problem::template InputDifferenceVector<Dual> delta;
        delta.resize(total);
        for (Index i = 0; i < total; ++i)
        {
            delta(i) = Dual{Scalar(0), Derivatives::Zero(l)};
        }
        for (Index j = 0; j < l; ++j)
        {
            delta(affine(j)) = Dual{Scalar(0), int(l), int(j)};
        }
        typename Problem::template InputVector<Dual> const dual{problem.stepInputs(x.template cast<Dual>(), delta)};
        typename Problem::template OutputVector<Dual> const y{problem.template evaluate<Dual>(dual)};
        OutputVector value = OutputVector::Zero(y.size());
        for (Index r = 0; r < y.size(); ++r)
        {
            value(r) = y(r).value();
        }
        auto const deviation = [&] {
            if constexpr (HasMinus<OutputSpace>)
            {
                return typename Problem::template OutputDifferenceVector<Dual>{
                    Emrys::minus(problem.outputSpace(), y, value.template cast<Dual>())
                };
            }
            else
            {
                return y;
            }
        }();
        for (Index r = 0; r < deviation.size(); ++r)
        {
            // An output that does not depend on the affine coordinates may carry no derivatives.
            if (deviation(r).derivatives().size() > 0)
            {
                columns.row(r) = deviation(r).derivatives().transpose();
            }
            else
            {
                columns.row(r).setZero();
            }
        }
        return value;
    }
}

/** Writes \a f's outputs at \a count points of the joint input into the columns of \a outputs, and
 * at point \a k its Jacobian in the joint input's difference coordinates \a affine into columns
 * <tt>k l</tt> to <tt>k l + l - 1</tt> of \a jacobians, with \f$ l \f$ the number of them: point \a k
 * is \f$ \mu \boxplus d_k \f$, the mean \a mu moved by <tt>shiftAt(k)</tt>, a joint difference
 * (decisions.md [affine-coordinates]). Where \a f evaluates batches, it is called once over every
 * point, and gives the Jacobians its function writes or that differentiating it in dual numbers gives
 * (decisions.md [batched-functions]); otherwise valueWithAffineJacobian() is called at each point in
 * turn.
 */
template <typename Problem, typename Affine, typename ShiftAt, typename Outputs, typename Jacobians>
void
evaluateWithAffineJacobians(
    Problem const & problem,
    typename Problem::template InputVector<typename Problem::Scalar> const & mu,
    Affine const & affine,
    Eigen::Index count,
    ShiftAt const & shiftAt,
    Outputs & outputs,
    Jacobians & jacobians
)
{
    using Index = Eigen::Index;
    Index const l = affine.size();
    if constexpr (Problem::EvaluatesBatches)
    {
        EMRYS_FILTER_ASSERT(jacobians.cols() == l * count && "a Jacobian over the affine coordinates per point");
        problem.evaluateEachWithAffineJacobians(
            count,
            [&](Index k) {
                return problem.plusInputs(mu, shiftAt(k));
            },
            outputs,
            jacobians
        );
    }
    else
    {
        for (Index k = 0; k < count; ++k)
        {
            auto const y = valueWithAffineJacobian(
                problem,
                problem.plusInputs(mu, shiftAt(k)),
                affine,
                jacobians.middleCols(k * l, l)
            );
            if (k == 0)
            {
                outputs.resize(y.size(), count);
            }
            outputs.col(k) = y;
        }
    }
}

/** The factored transform (decisions.md [affine-coordinates]): \a method's points over the joint input's
 * coordinates other than its affine ones, \f$ x_n \f$, and at each point the affine ones, \f$ x_a \f$,
 * %Gaussian given \f$ x_n \f$ with mean \f$ \mu_a + P_{an} P_{nn}^{-1} (x_n - \mu_n) \f$ and covariance
 * \f$ P_{a|n} = P_{aa} - P_{an} P_{nn}^{-1} P_{na} \f$, through \a f exactly with its Jacobian \f$ H \f$ in
 * them (valueWithAffineJacobian()): the output adds \f$ H P_{a|n} H^T \f$ at each point. With
 * \ref Julier2004 "Julier and Uhlmann (2004)"'s points over \f$ x_n \f$ this is the marginalized transform of
 * \ref Schon2005 "Schön, Gustafsson, and Nordlund (2005)".
 */
template <typename Problem, int AffineSize, int AffineMaxSize, typename Points>
[[nodiscard]] typename Problem::RootResult
transformResult(FactoredAt<AffineSize, AffineMaxSize, Points> const & method, Problem const & problem)
{
    using Scalar = typename Problem::Scalar;
    using Index = Eigen::Index;
    using InputVector = typename Problem::template InputVector<Scalar>;
    using InputDifferenceVector = typename Problem::template InputDifferenceVector<Scalar>;
    using OutputVector = typename Problem::template OutputVector<Scalar>;
    using OutputMatrix = typename Problem::OutputMatrix;
    constexpr int OutputSize = Problem::OutputDifferenceSize;
    constexpr int OutputMaxSize = Problem::OutputDifferenceMaxSize;
    // The rest of the joint input's differences: the coordinates that get sigma points.
    constexpr int RestSize = Problem::InputDifferenceSize == Eigen::Dynamic || AffineSize == Eigen::Dynamic
                               ? Eigen::Dynamic
                               : Problem::InputDifferenceSize - AffineSize;
    constexpr int RestMaxSize = RestSize == Eigen::Dynamic ? Problem::InputDifferenceMaxSize : RestSize;
    using RestIndices = IndexVector<RestSize, RestMaxSize>;
    using OutputSamples = typename Problem::template OutputSamples<RestSize, RestMaxSize>;
    using RestMatrix = MatrixOf<Scalar, RestSize, RestSize, RestMaxSize, RestMaxSize>;
    using RestAffine = MatrixOf<Scalar, RestSize, AffineSize, RestMaxSize, AffineMaxSize>;
    using RestCross = MatrixOf<Scalar, RestSize, OutputSize, RestMaxSize, OutputMaxSize>;
    using AffineMatrix = MatrixOf<Scalar, AffineSize, AffineSize, AffineMaxSize, AffineMaxSize>;
    using AffineRest = MatrixOf<Scalar, AffineSize, RestSize, AffineMaxSize, RestMaxSize>;
    using AffineCross = MatrixOf<Scalar, AffineSize, OutputSize, AffineMaxSize, OutputMaxSize>;
    using AffineJacobian = MatrixOf<Scalar, OutputSize, AffineSize, OutputMaxSize, AffineMaxSize>;

    auto const & affine = method.affine;
    Index const total = problem.inputDifferenceSize();
    Index const l = affine.size();
    Index const n = total - l;
    EMRYS_FILTER_ASSERT(
        distinctIndices(affine) && (l == 0 || (affine.minCoeff() >= 0 && affine.maxCoeff() < total))
        && "affine coordinates must be distinct coordinates of the joint input"
    );
    auto rest = RestIndices{};
    rest.resize(n);
    for (Index i = 0, k = 0; i < total; ++i)
    {
        if ((affine.array() != i).all())
        {
            rest(k++) = i;
        }
    }

    // x_a given x_n: mean mu_a + gain (x_n - mu_n), covariance conditional; points from a root of P_nn.
    InputVector const mu = problem.mean();
    typename Problem::InputMatrix const cov = problem.covarianceLower();
    AffineMatrix conditional = lowerSymmetricBlock(cov, affine, affine);
    RestMatrix restRoot = RestMatrix::Zero(n, n);
    AffineRest gain = AffineRest::Zero(l, n);
    if constexpr (RestSize != 0)
    {
        if (n > 0)
        {
            auto const ldlt = Eigen::LDLT<RestMatrix>{lowerSymmetricBlock(cov, rest, rest)};
            RestAffine const restAffine = lowerSymmetricBlock(cov, rest, affine);
            gain = transposedSolve<AffineRest>(ldlt, restAffine);
            conditional.noalias() -= multiply(gain, restAffine);
            refuseFailingAxes<AxesOf<Points>>();
            restRoot = rootValue(axesOf(method.points).root(lowerSymmetricBlock(cov, rest, rest)));
        }
    }
    conditional = (Scalar(0.5) * (conditional + conditional.transpose())).eval();
    // L_c, the root of P_a|n, which spreads x_a given x_n as the rest's root spreads x_n.
    AffineMatrix conditionalRoot = AffineMatrix::Zero(l, l);
    if constexpr (AffineSize != 0)
    {
        if (l > 0)
        {
            conditionalRoot = defaultRoot(conditional);
        }
    }

    // f at each point, and its Jacobian in x_a there kept as column block k of jacobians, in one
    // evaluation of the points (evaluateWithAffineJacobians()).
    Index const m = problem.outputDifferenceSize();
    // The affine coordinates' parts (decisions.md [linearization-forms]): with the mean Jacobian H_a = sum_i w_i h_i,
    // Psi_a = L_c^T H_a^T, and Omega_a = sum_i w_i (h_i - H_a) P_a|n (h_i - H_a)^T in a second pass
    // over the Jacobians kept, so that neither is recovered by subtraction.
    auto const affineParts = [&](auto const & jacobians, auto const & pointWeights) {
        Index const count = pointWeights.size();
        AffineJacobian meanJacobian = AffineJacobian::Zero(m, l);
        for (Index k = 0; k < count; ++k)
        {
            meanJacobian += pointWeights(k) * jacobians.middleCols(k * l, l);
        }
        OutputMatrix omega = OutputMatrix::Zero(m, m);
        for (Index k = 0; k < count; ++k)
        {
            AffineJacobian const deviation = jacobians.middleCols(k * l, l) - meanJacobian;
            AffineCross const spread = multiply(conditional, deviation.transpose());
            omega.noalias() += pointWeights(k) * multiply(deviation, spread);
        }
        AffineCross whitened = multiply(conditionalRoot.transpose(), meanJacobian.transpose());
        return std::pair{std::move(whitened), std::move(omega)};
    };

    // The output mean, the rows of x_n in Psi and the points' Omega, and x_a's parts.
    // With every coordinate affine there is nothing to spread: the center alone, by any rule.
    auto [mean, restWhitened, pointOmega, affineWhitened, affineOmega] = [&] {
        if constexpr (isUnscented<RuleOf<Points>> || RestSize == 0)
        {
            constexpr int Count = RestSize == Eigen::Dynamic ? Eigen::Dynamic : 2 * RestSize + 1;
            constexpr int MaxCount = RestMaxSize == Eigen::Dynamic ? Eigen::Dynamic : 2 * RestMaxSize + 1;
            using Jacobians = MatrixOf<
                Scalar,
                OutputSize,
                sizeProduct<AffineSize, Count>,
                OutputMaxSize,
                sizeProduct<AffineMaxSize, MaxCount>>;
            using PointWeights = MatrixOf<Scalar, Count, 1, MaxCount, 1>;
            SigmaWeights<Scalar> const weights = [&] {
                if constexpr (RestSize == 0)
                {
                    return SigmaWeights<Scalar>{Scalar(0), Scalar(1), Scalar(1), Scalar(0)};
                }
                else
                {
                    return sigmaWeights<Scalar, RestSize>(ruleOf(method.points), n);
                }
            }();
            RestMatrix const spread = weights.scale * restRoot;
            AffineRest const affineSpread = [&] {
                if constexpr (RestSize != 0)
                {
                    return AffineRest{multiply(gain, spread)};
                }
                else
                {
                    return AffineRest{};
                }
            }();
            // The center, then the plus and minus points of each column of the spread in turn.
            auto const shiftAt = [&]([[maybe_unused]] Index k) {
                InputDifferenceVector shift = InputDifferenceVector::Zero(total);
                if constexpr (RestSize != 0)
                {
                    if (k > 0)
                    {
                        Index const j = (k - 1) / 2;
                        shift(rest) = spread.col(j);
                        shift(affine) = affineSpread.col(j);
                        if (k % 2 == 0)
                        {
                            shift = -shift;
                        }
                    }
                }
                return shift;
            };
            auto jacobians = Jacobians{};
            jacobians.resize(m, l * (2 * n + 1));
            auto outputs = typename Problem::template OutputSamples<Count, MaxCount>{};
            evaluateWithAffineJacobians(problem, mu, affine, 2 * n + 1, shiftAt, outputs, jacobians);
            auto pointWeights = PointWeights{};
            pointWeights.resize(2 * n + 1);
            pointWeights(0) = weights.center;
            OutputVector const center = outputs.col(0);
            // Sized by resize(), since braces would set the coefficients of a fixed-size vector.
            auto plus = OutputSamples{};
            auto minus = OutputSamples{};
            plus.resize(center.size(), n);
            minus.resize(center.size(), n);
            // With every coordinate affine there are no plus and minus points, at compile time.
            if constexpr (RestSize != 0)
            {
                for (Index j = 0; j < n; ++j)
                {
                    plus.col(j) = outputs.col(1 + 2 * j);
                    minus.col(j) = outputs.col(2 + 2 * j);
                    pointWeights(1 + 2 * j) = weights.point;
                    pointWeights(2 + 2 * j) = weights.point;
                }
            }
            auto sigma = sigmaMoments(problem, weights, center, plus, minus);
            auto [whitened, omega] = affineParts(jacobians, pointWeights);
            return std::tuple{
                std::move(sigma.mean),
                RestCross{std::move(sigma.whitened)},
                std::move(sigma.unexplained),
                std::move(whitened),
                std::move(omega)
            };
        }
        else
        {
            using Unit = UnitPoints<Scalar, RuleOf<Points>, RestSize, RestMaxSize>;
            using RestSpread = MatrixOf<Scalar, RestSize, Unit::Count, RestMaxSize, Unit::MaxCount>;
            using AffineSpread = MatrixOf<Scalar, AffineSize, Unit::Count, AffineMaxSize, Unit::MaxCount>;
            using Jacobians = MatrixOf<
                Scalar,
                OutputSize,
                sizeProduct<AffineSize, Unit::Count>,
                OutputMaxSize,
                sizeProduct<AffineMaxSize, Unit::MaxCount>>;
            Unit const unit = unitPoints<Scalar, RestSize, RestMaxSize>(ruleOf(method.points), n);
            Index const count = unit.weights.size();
            auto spread = RestSpread{};
            spread.resize(n, count);
            spread.noalias() = multiply(restRoot, unit.points);
            auto affineSpread = AffineSpread{};
            affineSpread.resize(l, count);
            affineSpread.noalias() = multiply(gain, spread);
            auto jacobians = Jacobians{};
            jacobians.resize(m, l * count);
            auto outputs = typename Problem::template OutputSamples<Unit::Count, Unit::MaxCount>{};
            auto const shiftAt = [&](Index j) {
                InputDifferenceVector shift = InputDifferenceVector::Zero(total);
                shift(rest) = spread.col(j);
                shift(affine) = affineSpread.col(j);
                return shift;
            };
            evaluateWithAffineJacobians(problem, mu, affine, count, shiftAt, outputs, jacobians);
            auto points = pointMoments(problem, unit.points, unit.weights, unit.reference, outputs, restRoot);
            auto [whitened, omega] = affineParts(jacobians, unit.weights);
            return std::tuple{
                std::move(points.mean),
                RestCross{std::move(points.whitened)},
                std::move(points.unexplained),
                std::move(whitened),
                std::move(omega)
            };
        }
    }();

    // The joint root, with x_n = mu_n + R_n xi_n and x_a = mu_a + G R_n xi_n + L_c xi_a: block-diagonal
    // by input, since G and L_c are, and Psi's rows at x_n from the points and at x_a from L_c.
    typename Problem::InputMatrix root = Problem::InputMatrix::Zero(total, total);
    typename Problem::CrossMatrix whitened = Problem::CrossMatrix::Zero(total, m);
    if constexpr (RestSize != 0)
    {
        AffineRest const affineRoot = multiply(gain, restRoot);
        for (Index k = 0; k < n; ++k)
        {
            whitened.row(rest(k)) = restWhitened.row(k);
            for (Index j = 0; j < n; ++j)
            {
                root(rest(k), rest(j)) = restRoot(k, j);
            }
            for (Index j = 0; j < l; ++j)
            {
                root(affine(j), rest(k)) = affineRoot(j, k);
            }
        }
    }
    for (Index k = 0; k < l; ++k)
    {
        whitened.row(affine(k)) = affineWhitened.row(k);
        for (Index j = 0; j < l; ++j)
        {
            root(affine(k), affine(j)) = conditionalRoot(k, j);
        }
    }
    OutputMatrix unexplained = pointOmega + affineOmega;
    unexplained = (Scalar(0.5) * (unexplained + unexplained.transpose())).eval();
    return {std::move(mean), std::move(root), std::move(whitened), std::move(unexplained)};
}

/** A point of \a Space_ as an UncertainInput with no spread: what the Linearized method takes a
 * Jacobian at, which is all a step needs of an information form that may hold no covariance
 * (decisions.md [gaussian-forms]), or of a point given to linearizedAbout() [linearization-reference].
 * Its covariance reads as zero, which only a method that spreads points would use.
 */
template <Space Space_, RealScalar Scalar_>
class PointInput
{
public:

    using Scalar = Scalar_;                               ///< the scalar type
    using SpaceType = Space_;                             ///< the space the point lives in
    using MomentsType = Moments<Space_, Scalar_>;         ///< supplies the vector and matrix types
    using VectorType = typename MomentsType::VectorType;  ///< a value
    using MatrixType = typename MomentsType::MatrixType;  ///< a matrix over differences

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    /** The point \a point in \a space. */
    PointInput(Space_ space, VectorType point)
        : m_space{std::move(space)}
        , m_point{std::move(point)}
    {}

    /** \returns the space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the point, as the mean */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return m_point;
    }

    /** \returns a zero covariance: the point has no spread */
    [[nodiscard]] MatrixType
    covarianceLower() const
    {
        return MatrixType::Zero(differenceDimension(), differenceDimension());
    }

    /** \returns the number of coordinates of the point */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the number of difference coordinates */
    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

private:

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space
    VectorType m_point;                             ///< the point
};

/** The affine model of a function in an input linearized about a reference, in the chart of the
 * input's base point \f$ b \f$ (decisions.md [linearization-reference]): with \f$ \delta \f$ the input's
 * difference from \f$ b \f$, \f$ y \approx v \boxplus A \delta \f$, with error covariance \f$ N + \Omega \f$,
 * the parts of OutputWithCross kept apart (decisions.md [linearization-forms]). linearizeAbout() makes it, and a
 * transform, a prediction, and a form's step each fold it in.
 *
 * \tparam InputSpace   the input's space
 * \tparam OutputSpace  the output's space
 */
template <typename InputSpace, typename OutputSpace, typename Scalar>
struct LinearizationAbout
{
    /// supplies the matrix types
    using Linearization = StatisticalLinearization<InputSpace, OutputSpace, Scalar>;

    using JacobianMatrix = typename Linearization::JacobianMatrix;                  ///< \f$ A \f$
    using ErrorMatrix = typename Linearization::ErrorMatrix;                        ///< \f$ N \f$ or \f$ \Omega \f$
    using OutputVector = typename GaussianParams<OutputSpace, Scalar>::VectorType;  ///< an output value

    OutputSpace space;        ///< the output's space, sized
    OutputVector value;       ///< \f$ v \f$, the affine model at the base, normalized
    JacobianMatrix jacobian;  ///< \f$ A \f$: output differences by the input's differences at the base
    ErrorMatrix inputs;       ///< \f$ N \f$: what the function's other uncertain inputs explain; symmetric

    /** \f$ \Omega \f$: what no input explains, for a sigma-point method the function's curvature over the
     * reference's spread; symmetric
     */
    ErrorMatrix unexplained;

    /** \returns \f$ N + \Omega \f$, everything \f$ A \f$ leaves unexplained */
    [[nodiscard]] ErrorMatrix
    error() const
    {
        return inputs + unexplained;
    }
};

/** Carries \a linearization, taken about the point \a point of \a space in its own chart, to the
 * chart of \a base (decisions.md [linearization-reference]): the Jacobian times the reset Jacobian of \a base
 * and \f$ \delta_r = \mathrm{point} \boxminus \mathrm{base} \f$ [covariance-reset], as an Iterated
 * update carries an iterate's, and the value moved back along it to \f$ \delta = 0 \f$,
 * \f$ v \boxplus (-A \delta_r) \f$.
 */
template <typename InputSpace, typename Point, typename Base, typename OutputSpace, typename Scalar>
void
toChartOf(
    InputSpace const & space,
    Eigen::MatrixBase<Base> const & base,
    Point const & point,
    LinearizationAbout<InputSpace, OutputSpace, Scalar> & linearization
)
{
    using Linearization = LinearizationAbout<InputSpace, OutputSpace, Scalar>;
    using Difference = typename Moments<InputSpace, Scalar>::DifferenceVectorType;
    using OutputDifference = typename GaussianParams<OutputSpace, Scalar>::DifferenceVectorType;
    Difference const step = Emrys::minus(space, point, base);
    if constexpr (!hasIdentityReset<InputSpace>)
    {
        auto carried = typename Linearization::JacobianMatrix{};
        carried.resize(linearization.jacobian.rows(), linearization.jacobian.cols());
        carried.noalias() = multiply(linearization.jacobian, resetJacobian(space, base, step));
        linearization.jacobian = std::move(carried);
    }
    auto back = OutputDifference{};
    back.resize(linearization.jacobian.rows());
    back.noalias() = -multiply(linearization.jacobian, step);
    linearization.value = Emrys::plus(linearization.space, linearization.value, back);
}

/** Refuses at compile time a reference \a About, a LinearizedAbout, that a step by \a Method over
 * an input in \a InputSpace cannot take (decisions.md [linearization-reference]): any reference given to an
 * Iterated method, which chooses its own points; a distribution in another space; and a point given
 * to a sigma-point method, which needs a spread. Every step that takes a reference calls it first,
 * inside a static_assert, so that the compiler instantiates it there rather than at the end of the
 * translation unit, where MSVC reports what follows from the step's body first.
 *
 * \returns true, when it compiles
 */
template <typename Method, typename About, typename InputSpace>
[[nodiscard]] consteval bool
checkReference()
{
    using Reference = typename About::ReferenceType;
    static_assert(
        !isIterated<Method>,
        "an Iterated update chooses its own points to linearize at, so give linearizedAbout() to a step by its "
        "inner method instead"
    );
    if constexpr (About::IsDistribution)
    {
        static_assert(
            std::is_same_v<typename std::remove_cvref_t<Reference>::SpaceType, InputSpace>,
            "linearizedAbout() takes a point or a distribution in the space of the input it is for"
        );
    }
    static_assert(
        isLinearMethod<Method> || About::IsDistribution,
        "a sigma-point method linearizes over a distribution, so give linearizedAbout() one, such as another "
        "filter's state, rather than a point"
    );
    return true;
}

/** \returns the LinearizationAbout of \a f by \a method in an uncertain input in \a space, the
 * first argument of \a f, linearized about \a about in the chart of the input's \a base, with
 * \a args after it as transform() takes them (decisions.md [linearization-reference]).
 *
 * A linear method (Linearized) takes the Jacobian at the reference's point, a PointInput, and
 * \f$ N = J \Sigma J^T \f$ over \a args, the only spread there is. Any other method transforms
 * the reference's distribution \f$ q \f$, marked as by withCross(), and takes the statistical
 * linearization of the result in it, \f$ A = \Psi^T S_q^{-1} = \Sigma_{qy}^T P_q^{-1} \f$, with
 * \f$ N \f$ over \a args and \f$ \Omega \f$, the curvature over \f$ q \f$, as the root form holds them
 * (decisions.md [linearization-forms]): the statistical linear regression with respect to a density of the caller's
 * choice of \ref GarciaFernandez2015 "García-Fernández, Svensson, Morelande, and Särkkä (2015)". Then
 * toChartOf().
 */
template <
    typename Method,
    typename F,
    typename OutputSpace,
    typename InputSpace,
    typename Base,
    typename Reference,
    typename... Args>
[[nodiscard]] auto
linearizeAbout(
    Method const & method,
    F & f,
    OutputSpace const & outputSpace,
    InputSpace const & space,
    Eigen::MatrixBase<Base> const & base,
    LinearizedAbout<Reference> const & about,
    Args const &... args
)
{
    using Scalar = typename Base::Scalar;
    using About = LinearizedAbout<Reference>;
    using Point = PointInput<InputSpace, Scalar>;
    using Result = LinearizationAbout<InputSpace, OutputSpace, Scalar>;
    static_assert(
        crossCount<Args...> == 0,
        "withCross() may mark only the input linearizedAbout() is for, which is the argument right after it"
    );
    static_assert(checkReference<Method, About, InputSpace>());
    typename Point::VectorType const point = about.mean();
    EMRYS_FILTER_ASSERT(
        point.size() == valueDimension(space) && "a point to linearize about must have the input's value dimension"
    );
    Eigen::Index const n = differenceDimension(space);
    auto result = [&] {
        if constexpr (isLinearMethod<Method>)
        {
            auto const input = Point{space, point};
            using Problem = Transform<F, OutputSpace, Point, Args...>;
            auto const problem = Problem{f, outputSpace, input, args...};
            auto const linearized = linearizeAt(method, problem);
            // The other inputs' share, J Sigma J^T over the joint input, whose block at the point is zero.
            typename Problem::CrossMatrix const cross =
                multiply(problem.covarianceLower(), linearized.jacobian.transpose());
            typename Result::ErrorMatrix inputs = multiply(linearized.jacobian, cross);
            inputs = (Scalar(0.5) * (inputs + inputs.transpose())).eval();
            Eigen::Index const m = inputs.rows();
            return Result{
                problem.outputSpace(),
                typename Result::OutputVector{linearized.mean},
                typename Result::JacobianMatrix{linearized.jacobian.middleCols(problem.differenceOffset(0), n)},
                std::move(inputs),
                Result::ErrorMatrix::Zero(m, m)
            };
        }
        else if constexpr (About::IsDistribution)
        {
            using Distribution = std::remove_cvref_t<Reference>;
            using Marked = CrossMarked<Distribution>;
            using Problem = Transform<F, OutputSpace, Marked, Args...>;
            auto const marked = Marked{about.reference};
            auto const problem = Problem{f, outputSpace, marked, args...};
            auto const predicted = problem.assemble(transformResult(method, problem));
            return Result{
                problem.outputSpace(),
                typename Result::OutputVector{predicted.output().mean()},
                predicted.jacobian(),
                predicted.inputsCovariance(),
                predicted.unexplainedCovariance()
            };
        }
    }();
    toChartOf(space, base, point, result);
    return result;
}

/** transform() of \a x, an uncertain input or one marked by withCross(), linearized about
 * \a about, with \a args after it (see linearizedAbout()): from the LinearizationAbout
 * \f$ (v, A, N, \Omega) \f$ in \a x's chart, the output's mean \f$ v \f$ and covariance
 * \f$ A P A^T + N + \Omega \f$, and, for a marked \a x, the Jacobian form with \f$ H = A \f$ and the
 * cross-covariance \f$ P A^T \f$.
 */
template <typename Method, typename F, typename OutputSpace, typename Reference, typename X, typename... Args>
[[nodiscard]] auto
transformAbout(
    Method const & method,
    F & f,
    OutputSpace const & outputSpace,
    LinearizedAbout<Reference> const & about,
    X const & x,
    Args const &... args
)
{
    static_assert(ArgTraits<X>::isUncertain, "linearizedAbout() is followed by the uncertain input it is for");
    using Input = typename ArgTraits<X>::Input;
    using InputSpace = typename Input::SpaceType;
    using Scalar = typename Input::Scalar;
    using Output = GaussianParams<OutputSpace, Scalar>;
    using Covariance = typename Moments<InputSpace, Scalar>::MatrixType;
    using Cross = CrossMatrixOf<InputSpace, OutputSpace, Scalar>;
    auto const & input = argInput(x);
    auto linearization = linearizeAbout(method, f, outputSpace, input.space(), input.mean(), about, args...);
    Covariance const cov = symmetricFromLower<Covariance>(covarianceLowerOf(input));
    Cross cross = multiply(cov, linearization.jacobian.transpose());
    typename Output::MatrixType explained = multiply(linearization.jacobian, cross);
    explained = (Scalar(0.5) * (explained + explained.transpose())).eval();
    if constexpr (ArgTraits<X>::hasCross)
    {
        return OutputWithCross<InputSpace, OutputSpace, JacobianForm, Scalar>{
            std::move(linearization.space),
            std::move(linearization.value),
            std::move(linearization.jacobian),
            std::move(cross),
            std::move(explained),
            std::move(linearization.inputs),
            std::move(linearization.unexplained)
        };
    }
    else
    {
        typename Output::MatrixType outputCov = explained + linearization.inputs + linearization.unexplained;
        return Output{linearization.space, linearization.value, std::move(outputCov)};
    }
}

}  // namespace internal

/** \ingroup Transforms
 * Marks the uncertain \a input, usually the state, whose cross-covariance with the output
 * transform() also returns (see OutputWithCross). At most one input may be marked.
 */
template <UncertainInput Input>
[[nodiscard]] CrossMarked<Input>
withCross(Input const & input)
{
    return {input};
}

/** \ingroup Transforms
 * withCross() of a temporary is deleted: the mark refers to its input, which would be gone at
 * the end of the full expression, so a mark kept in a variable would dangle.
 */
template <UncertainInput Input>
void
withCross(Input const && input) = delete;

/** \ingroup Transforms
 * \returns the distribution \a distribution as the reference a step linearizes about, in place of
 * its input's own mean and spread (decisions.md [linearization-reference]): every method linearizes over it,
 * Linearized at its mean and a sigma-point method over its spread, and the result is carried to the
 * input's chart and applied to the input's own moments.
 *
 * A transform takes it right before the input it is for, a prediction before its inputs, and an
 * update after its measurement:
 *
 * \code
 * // A shadow filter linearized about the main filter's state, as a tracker's shadow does.
 * auto const about = Emrys::linearizedAbout(main.state());
 * shadow.predict(dt, about);
 * auto const report = shadow.update(measurement, about);
 * \endcode
 *
 * Linearized about another estimate is the linearized Kalman filter of \ref Gelb1974 "Gelb (1974)",
 * about a nominal trajectory, here supplied at each step; about the step's own posterior it is
 * one iteration of the iterated update (see Iterated), and a sigma-point method's is the statistical
 * linear regression with respect to \a distribution of
 * \ref GarciaFernandez2015 "García-Fernández, Svensson, Morelande, and Särkkä (2015)".
 * The mark refers to \a distribution, so it is used as it is made or while
 * \a distribution lives.
 */
template <UncertainInput Distribution>
[[nodiscard]] LinearizedAbout<Distribution const &>
linearizedAbout(Distribution const & distribution)
{
    return {distribution};
}

/** \ingroup Transforms
 * linearizedAbout() of a temporary distribution is deleted, as withCross()'s is: the mark refers to
 * it, and it would be gone at the end of the full expression.
 */
template <UncertainInput Distribution>
void
linearizedAbout(Distribution const && distribution) = delete;

/** \ingroup Transforms
 * \returns the point \a point, a column vector, copied, as the reference a step linearizes at
 * (decisions.md [linearization-reference]). Only a linear method (Linearized) linearizes at a point; a
 * sigma-point method needs a distribution to spread over and refuses one at compile time.
 */
template <typename Derived>
[[nodiscard]] LinearizedAbout<typename Derived::PlainObject>
linearizedAbout(Eigen::MatrixBase<Derived> const & point)
{
    static_assert(Derived::ColsAtCompileTime == 1, "a point to linearize at is a column vector");
    return {typename Derived::PlainObject{point}};
}

namespace internal
{

/** \returns the distribution of \f$ y = f(\ldots) \f$ in \a outputSpace, with \a args passed to
 * \a f in order, computed by \a method (Linearized, or a sigma-point method): a %Gaussian in
 * \a outputSpace, or with an input marked by withCross(), an OutputWithCross. The positional
 * transform that every model's prediction is computed by; transform() builds on it.
 *
 * With a linearizedAbout() reference first among \a args, the input after it, the only one
 * withCross() may then mark, is linearized about the reference rather than over itself
 * (decisions.md [linearization-reference]): the affine model \f$ y \approx v \boxplus A \delta \f$ with
 * error covariance \f$ \Omega \f$ is taken at the reference's point, or over its distribution, in
 * the input's chart, and the result is its mean \f$ v \f$ and covariance
 * \f$ A P A^T + \Omega \f$, with cross-covariance \f$ P A^T \f$. About the input itself this is
 * the transform without a reference, to rounding.
 */
template <typename Method, typename F, typename OutputSpace, typename... Args>
    requires(IsSpace<OutputSpace> && crossCount<Args...> <= 1)
[[nodiscard]] auto
transformBy(Method const & method, F && f, OutputSpace const & outputSpace, Args const &... args)
{
    if constexpr (leadsWithReference<Args...>)
    {
        return transformAbout(method, f, outputSpace, args...);
    }
    else
    {
        using Problem = Transform<std::remove_reference_t<F>, OutputSpace, Args...>;
        Problem const problem{f, outputSpace, args...};
        return problem.assemble(transformResult(method, problem));
    }
}

/** The tag of an input \a T of transform(): an uncertain input's space, a withCross() one's input's,
 * a Sure's space, and void for anything else.
 */
template <typename T>
struct TransformInputTagOf
{
    using type = void;  ///< not an input
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename T>
    requires(ArgTraits<T>::isUncertain)
struct TransformInputTagOf<T>
{
    using type = typename ArgTraits<T>::Input::SpaceType;
};

template <typename S, typename Vector>
struct TransformInputTagOf<Sure<S, Vector>>
{
    using type = S;
};

/// \endcond

/** The tag of an input \a T of transform() (TransformInputTagOf). */
template <typename T>
using TransformInputTag = typename TransformInputTagOf<T>::type;

/** The number of leading arguments of \a Args that are a model's policies: transform()'s policies,
 * before its inputs.
 */
template <typename... Args>
inline constexpr std::size_t leadingPolicies = [] {
    constexpr std::array<bool, sizeof...(Args) + 1> policies{ModelPolicy<Args>..., false};
    std::size_t n = 0;
    while (policies[n])
    {
        ++n;
    }
    return n;
}();

/** Whether a policy of \a Args follows an input. */
template <typename... Args>
inline constexpr bool policyAfterInput = [] {
    constexpr std::array<bool, sizeof...(Args) + 1> policies{ModelPolicy<Args>..., false};
    for (std::size_t i = leadingPolicies<Args...>; i < sizeof...(Args); ++i)
    {
        if (policies[i])
        {
            return true;
        }
    }
    return false;
}();

/** The first of \a Ts that is not void, or void. */
template <typename... Ts>
struct FirstTypeOf
{
    using type = void;  ///< none
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename First, typename... Rest>
struct FirstTypeOf<First, Rest...>
{
    using type = std::conditional_t<std::is_void_v<First>, typename FirstTypeOf<Rest...>::type, First>;
};

/// \endcond

/** The scalar of the vector \a A that the positional transform passes for an uncertain input, or
 * void for a sure one.
 */
template <bool Uncertain, typename A>
struct UncertainScalarOf
{
    using type = void;  ///< a sure input
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename A>
struct UncertainScalarOf<true, A>
{
    using type = typename A::Scalar;
};

/// \endcond

/** \returns, as a std::type_identity, the reads() policy of the measurement model \a Model applied
 * to \a Inputs by transform(), or ReadsWhole of the one input's space for a model without reads():
 * ModelTransform's ReadsPolicy, computed outside the class so that no lambda sits in its body.
 */
template <typename Model, typename... Inputs>
[[nodiscard]] consteval auto
transformReadsPolicy()
{
    if constexpr (readsWholeState<Model>)
    {
        static_assert(
            sizeof...(Inputs) == 1 && (ArgTraits<Inputs>::isUncertain && ...),
            "transform() of a function without reads() takes one uncertain input, the vector the function reads; "
            "name several inputs with reads(), each over a tagged space"
        );
        return std::type_identity<ReadsWhole<TransformInputTag<std::tuple_element_t<0, std::tuple<Inputs...>>>>>{};
    }
    else
    {
        return std::type_identity<typename Model::template Policy<PolicyKind::reads>>{};
    }
}

/** \returns whether the measurement model \a Model's output space is the space of its one unplaced
 * additive noise: ModelTransform's OutputFromNoise, computed outside the class so that no lambda sits
 * in its body.
 */
template <typename Model>
[[nodiscard]] consteval bool
transformOutputFromNoise()
{
    if constexpr (Model::template has<PolicyKind::outputSpace> || Model::template has<PolicyKind::writes>)
    {
        return false;
    }
    else if constexpr (Model::template has<PolicyKind::additiveNoise>)
    {
        return Model::template Policy<PolicyKind::additiveNoise>::HasOwnSpace;
    }
    else
    {
        return false;
    }
}

/** \returns, as a std::type_identity, the output space of the measurement model \a Model over the
 * input space \a In: outputSpace()'s, the Composite of writes(), the noise's space where
 * \a OutputFromNoise, one the function names, or a Linear sized by what the function returns
 * (decisions.md [sensors]). ModelTransform's Out, computed outside the class so that no lambda sits
 * in its body.
 */
template <typename Model, typename In, bool OutputFromNoise>
[[nodiscard]] consteval auto
transformOutputSpace()
{
    using Function = typename Model::FunctionType;
    if constexpr (Model::template has<PolicyKind::outputSpace>)
    {
        using Policy = typename Model::template Policy<PolicyKind::outputSpace>;
        return std::type_identity<decltype(std::declval<Policy const &>().in(std::declval<In const &>()))>{};
    }
    else if constexpr (Model::template has<PolicyKind::writes>)
    {
        return std::type_identity<typename Model::template Policy<PolicyKind::writes>::OutputSpace>{};
    }
    else if constexpr (OutputFromNoise)
    {
        return std::type_identity<
            typename Model::template Policy<PolicyKind::additiveNoise>::Distribution::SpaceType>{};
    }
    else if constexpr (requires { typename Function::OutputSpaceType; })
    {
        return std::type_identity<typename Function::OutputSpaceType>{};
    }
    else
    {
        using Returned = decltype(vectorReturned(callForm(
            std::declval<Function const &>(),
            std::declval<In const &>(),
            std::declval<NoOutputSpace const &>(),
            std::declval<typename Moments<In, double>::VectorType const &>()
        )));
        using Evaluated = std::remove_cvref_t<decltype(valueOf(std::declval<Returned>()).eval())>;
        static_assert(Evaluated::ColsAtCompileTime == 1, "a transformed function must return a column vector");
        return std::type_identity<Linear<Evaluated::RowsAtCompileTime, Evaluated::MaxRowsAtCompileTime>>{};
    }
}

/** A measurement model \a Model applied to the inputs \a Inputs of a transform() (decisions.md
 * [transform-policies]): the model's input space built from the inputs it reads, matched to reads() by
 * tag, or from the one input of a model without reads(); its output space; and the function the
 * positional transform calls, which gathers the model's input vector from the arguments it is given,
 * calls the model's function in the form it takes, and splits a returned Jacobian by uncertain input.
 * Valid while the model and the inputs are.
 */
template <typename Model, typename... Inputs>
class ModelTransform
{
public:

    /// whether the model reads the whole of its one input, with no reads()
    static constexpr bool Whole = readsWholeState<Model>;

    static constexpr std::size_t InputCount = sizeof...(Inputs);  ///< the inputs given

    /** Input \a J's type. */
    template <std::size_t J>
    using InputAt = std::tuple_element_t<J, std::tuple<Inputs...>>;

    /** Input \a J's tag. */
    template <std::size_t J>
    using TagAt = TransformInputTag<InputAt<J>>;

private:

    /** \returns whether input \a j is uncertain */
    [[nodiscard]] static consteval bool
    uncertain(std::size_t j)
    {
        constexpr std::array<bool, InputCount + 1> flags{ArgTraits<Inputs>::isUncertain..., false};
        return flags[j];
    }

    static_assert(
        ((!std::is_void_v<TransformInputTag<Inputs>>) && ...),
        "an input of transform() is an uncertain input, one marked by withCross(), or sure<Tag>(value), after a "
        "linearizedAbout() reference if there is one; its output space is a policy, outputSpace(space), before the "
        "inputs"
    );
    static_assert(
        distinctTypes<TransformInputTag<Inputs>...>(),
        "two inputs of transform() have one tag; an input's tag names it once"
    );

    /** The reads() policy, or ReadsWhole of the one input's space. */
    using ReadsPolicy = typename decltype(transformReadsPolicy<Model, Inputs...>())::type;

public:

    static constexpr std::size_t ItemCount = ReadsPolicy::Count;  ///< the items read

    /** Item \a I of reads(). */
    template <std::size_t I>
    using Item = typename ReadsPolicy::template Item<I>;

    /** The name item \a I is read under. */
    template <std::size_t I>
    using Name = typename Item<I>::NameType;

    /** The model's input space: the Composite of the names, or the one input's space. */
    using In = typename ReadsPolicy::InputSpace;

    /** A vector in the input space, in \a T. */
    template <typename T>
    using InVector = typename Moments<In, T>::VectorType;

private:

    /** \returns the input item \a I reads, by the tag of its one-tag path, or InputCount */
    template <std::size_t I>
    [[nodiscard]] static consteval std::size_t
    inputOfItem()
    {
        if constexpr (Whole)
        {
            return 0;
        }
        else
        {
            using Path = typename Item<I>::PathType;
            static_assert(
                std::is_same_v<Path, TagPath<PathTag<Path>>>,
                "transform() matches each item of reads() to an input by its one tag, path<Tag> or "
                "on<Name>(path<Tag>); "
                "it has no state for a path of several tags to lie in"
            );
            constexpr std::array<bool, InputCount + 1> matches{
                std::is_same_v<TransformInputTag<Inputs>, PathTag<Path>>...,
                false
            };
            std::size_t j = 0;
            while (j < InputCount && !matches[j])
            {
                ++j;
            }
            return j;
        }
    }

    /** \returns the input item \a i reads */
    [[nodiscard]] static consteval std::size_t
    inputOf(std::size_t i)
    {
        return []<std::size_t... I>(std::size_t item, std::index_sequence<I...>) {
            std::size_t found = InputCount;
            ((I == item ? (found = inputOfItem<I>(), 0) : 0), ...);
            return found;
        }(i, std::make_index_sequence<ItemCount>{});
    }

    /** \returns whether every item \a I of reads() names an input given */
    template <std::size_t... I>
    [[nodiscard]] static consteval bool
    itemsGiven(std::index_sequence<I...>)
    {
        return ((inputOfItem<I>() < InputCount) && ...);
    }

public:

    /// whether every item of reads() names an input given, checked by inputsMatch()
    static constexpr bool ItemsGiven = itemsGiven(std::make_index_sequence<ItemCount>{});

private:

    /** \returns whether input \a j is read by an item */
    [[nodiscard]] static consteval bool
    inputRead(std::size_t j)
    {
        for (std::size_t i = 0; i < ItemCount; ++i)
        {
            if (inputOf(i) == j)
            {
                return true;
            }
        }
        return false;
    }

    /** \returns whether an item of reads() reads every input \a J */
    template <std::size_t... J>
    [[nodiscard]] static consteval bool
    inputsRead(std::index_sequence<J...>)
    {
        return (inputRead(J) && ...);
    }

public:

    /// whether an item of reads() reads every input given, checked by inputsMatch()
    static constexpr bool InputsRead = inputsRead(std::make_index_sequence<InputCount>{});

private:

    /** \returns the position of input \a j among the uncertain inputs, the order of the joint input */
    [[nodiscard]] static consteval std::size_t
    uncertainIndex(std::size_t j)
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < j; ++i)
        {
            if (uncertain(i))
            {
                ++k;
            }
        }
        return k;
    }

    /** The number of uncertain inputs. */
    static constexpr std::size_t UncertainCount = (std::size_t{ArgTraits<Inputs>::isUncertain} + ... + 0);

public:

    /** The model's function. */
    using Function = typename Model::FunctionType;

    /// whether the model's output space is the space of its one unplaced additive noise
    static constexpr bool OutputFromNoise = transformOutputFromNoise<Model>();

    /** The output space: outputSpace()'s, the Composite of writes(), the noise's space, one the
     * function names, or a Linear sized by what the function returns (decisions.md [sensors]).
     */
    using Out = typename decltype(transformOutputSpace<Model, In, OutputFromNoise>())::type;

    /** The model \a model applied to \a inputs, which both outlive this. */
    explicit ModelTransform(Model const & model, Inputs const &... inputs)
        : m_model{model}
        , m_inputs{inputs...}
        , m_in{makeIn()}
        , m_out{makeOut()}
    {
        if constexpr (Whole)
        {
            m_valueRows[0] = BlockRows{0, valueDimension(m_in)};
            m_differenceRows[0] = BlockRows{0, differenceDimension(m_in)};
        }
        else
        {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((m_valueRows[I] = blockRows<Name<I>>(m_in), m_differenceRows[I] = differenceBlockRows<Name<I>>(m_in)),
                 ...);
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    /** \returns the output space instance */
    [[nodiscard]] Out const &
    out() const
    {
        return m_out;
    }

    /** \returns the model's function at the inputs \a args, as the positional transform gives them:
     * a vector for each uncertain input and the input itself for a sure one. Its value, or a
     * WithJacobian with a block per uncertain input, in order.
     */
    template <typename... A>
    [[nodiscard]] auto
    operator()(A const &... args) const
    {
        static_assert(sizeof...(A) == InputCount);
        using T = typename FirstTypeOf<typename UncertainScalarOf<ArgTraits<Inputs>::isUncertain, A>::type...>::type;
        auto const arguments = std::tuple<A const &...>{args...};
        auto x = InVector<T>{};
        x.resize(valueDimension(m_in));
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (gatherItem<I, T>(x, std::get<inputOfItem<I>()>(arguments)), ...);
        }(std::make_index_sequence<ItemCount>{});
        return split(vectorReturned(callForm(m_model.function(), m_in, m_out, x)));
    }

    /// whether the positional transform evaluates a batch of points in one call, evaluateBatch():
    /// the model's function is batched (decisions.md [batched-functions])
    static constexpr bool EvaluatesBatches = takesBatches<Function>;

    /** Writes into \a y the model's function at a batch of points, a column each in the output
     * space, by one call of the batched function: point \a j at column \a j of each uncertain input
     * among \a args, as the positional transform gives them, and at each sure input itself.
     */
    template <typename Y, typename... A>
    void
    evaluateBatch(Y & y, A const &... args) const
        requires EvaluatesBatches
    {
        callBatched(m_model.function(), m_in, m_out, gatherColumns<Y>(y.cols(), args...), y);
    }

    /** evaluateBatch() with the function's Jacobian over the items the method factors out written
     * into \a jacobians, a block of columns per point in the order of reads(), the coordinates
     * affineCoordinates() gives (internal::callBatchedWithJacobians())
     */
    template <typename Y, typename J, typename... A>
    void
    evaluateBatchWithAffineJacobians(Y & y, J & jacobians, A const &... args) const
        requires EvaluatesBatches
    {
        auto wanted = std::array<bool, std::tuple_size_v<InputBlocks<In>>>{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((wanted[I] = itemFactored<I>()), ...);
        }(std::make_index_sequence<ItemCount>{});
        auto const x = gatherColumns<Y>(y.cols(), args...);
        callBatchedWithJacobians(m_model.function(), m_in, m_out, x, wanted, y, jacobians);
    }

    /** \returns the inputs of a batch of \a count points, a column each in the input space, as many
     * columns as \a Y has at compile time: point \a j at column \a j of each uncertain input among
     * \a args, as the positional transform gives them, and at each sure input itself
     */
    template <typename Y, typename... A>
    [[nodiscard]] auto
    gatherColumns(Eigen::Index count, A const &... args) const
    {
        static_assert(sizeof...(A) == InputCount);
        using T = typename Y::Scalar;
        using X = MatrixOf<
            T,
            valueDimensionAtCompileTime<In>,
            Y::ColsAtCompileTime,
            maxValueDimensionAtCompileTime<In>,
            Y::MaxColsAtCompileTime>;
        auto const arguments = std::tuple<A const &...>{args...};
        auto x = X{};
        x.resize(valueDimension(m_in), count);
        auto column = InVector<T>{};
        column.resize(valueDimension(m_in));
        for (Eigen::Index j = 0; j < count; ++j)
        {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                (gatherItem<I, T>(column, columnOf<inputOfItem<I>()>(std::get<inputOfItem<I>()>(arguments), j)), ...);
            }(std::make_index_sequence<ItemCount>{});
            x.col(j) = column;
        }
        return x;
    }

    /** \returns whether item \a I is one the method factors out: named in the model's affineIn(), in a
     * linear space, and an uncertain input
     */
    template <std::size_t I>
    [[nodiscard]] static consteval bool
    itemFactored()
    {
        using Affine = typename Model::template Policy<PolicyKind::affineIn>;
        return affineNames<Affine, Name<I>> && isLinear<Name<I>> && uncertain(inputOfItem<I>());
    }

    /** \returns whether any item is factored out */
    [[nodiscard]] static consteval bool
    factorsAffine()
    {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return (itemFactored<I>() || ... || false);
        }(std::make_index_sequence<ItemCount>{});
    }

    /** \returns the difference dimension of the items factored out, or Eigen::Dynamic */
    [[nodiscard]] static consteval int
    affineSize()
    {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return sizeSum<(itemFactored<I>() ? valueDimensionAtCompileTime<DifferenceSpace<Name<I>>> : 0)...>;
        }(std::make_index_sequence<ItemCount>{});
    }

    /** \returns a bound on affineSize(), or Eigen::Dynamic */
    [[nodiscard]] static consteval int
    affineMaxSize()
    {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return sizeSum<(itemFactored<I>() ? maxValueDimensionAtCompileTime<DifferenceSpace<Name<I>>> : 0)...>;
        }(std::make_index_sequence<ItemCount>{});
    }

    /** \returns the coordinates of the joint uncertain input that the method factors out, in the
     * order of reads(): each factored item's input's difference coordinates (decisions.md
     * [affine-coordinates])
     */
    [[nodiscard]] IndexVector<affineSize(), affineMaxSize()>
    affineCoordinates() const
    {
        std::array<Eigen::Index, InputCount + 1> starts{};
        [&]<std::size_t... J>(std::index_sequence<J...>) {
            ((starts[J + 1] = starts[J] + (uncertain(J) ? argInput(std::get<J>(m_inputs)).differenceDimension() : 0)),
             ...);
        }(std::make_index_sequence<InputCount>{});
        Eigen::Index count = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((count += itemFactored<I>() ? m_differenceRows[I].size : 0), ...);
        }(std::make_index_sequence<ItemCount>{});
        auto result = IndexVector<affineSize(), affineMaxSize()>{};
        result.resize(count);
        Eigen::Index next = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    if constexpr (itemFactored<I>())
                    {
                        for (Eigen::Index r = 0; r < m_differenceRows[I].size; ++r)
                        {
                            result(next++) = starts[inputOfItem<I>()] + r;
                        }
                    }
                }(),
                ...
            );
        }(std::make_index_sequence<ItemCount>{});
        return result;
    }

    /** \returns \a result, what the positional transform returned, with the model's additive noise
     * added: its covariance to the output's, as \f$ N \f$ of an OutputWithCross, and its mean moved
     * by the noise's mean
     */
    template <typename Result>
    [[nodiscard]] auto
    withNoise(Result && result) const
    {
        if constexpr (!Model::template has<PolicyKind::additiveNoise>)
        {
            return std::forward<Result>(result);
        }
        else
        {
            auto const placed = placeNoise(m_out, m_model.template policy<PolicyKind::additiveNoise>());
            auto const & noise = placed.distribution();
            static_assert(
                !isGaussianMixture<std::remove_cvref_t<decltype(noise)>>,
                "transform() adds one Gaussian noise; mixture noise gives a branch per component, which update() keeps"
            );
            return plusNoise(result, noise);
        }
    }

private:

    /** \returns the input space instance: each item's name over its input's space, or the one input's */
    [[nodiscard]] In
    makeIn() const
    {
        if constexpr (Whole)
        {
            return argInput(std::get<0>(m_inputs)).space();
        }
        else
        {
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                return In{nameOf<Name<I>>(spaceOf<inputOfItem<I>()>())...};
            }(std::make_index_sequence<ItemCount>{});
        }
    }

    /** \returns input \a J's space */
    template <std::size_t J>
    [[nodiscard]] auto
    spaceOf() const
    {
        auto const & input = std::get<J>(m_inputs);
        if constexpr (isSure<std::remove_cvref_t<decltype(input)>>)
        {
            return input.space;
        }
        else
        {
            return argInput(input).space();
        }
    }

    /** \returns the output space instance: from outputSpace() at the input space, the noise's space,
     * a Linear sized by a function that knows its rows, or the empty instance, which the positional
     * transform sizes from the function's value where it is resizable
     */
    [[nodiscard]] Out
    makeOut() const
    {
        if constexpr (Model::template has<PolicyKind::outputSpace>)
        {
            return m_model.template policy<PolicyKind::outputSpace>().in(m_in);
        }
        else if constexpr (OutputFromNoise)
        {
            return m_model.template policy<PolicyKind::additiveNoise>().distribution().space();
        }
        else if constexpr (!FixedSizeSpace<Out> && requires(Function const & f) { f.outputRows(); })
        {
            return Out{m_model.function().outputRows()};
        }
        else
        {
            return Out{};
        }
    }

    /** Writes item \a I's block of the input vector \a x from \a argument, its input as the positional
     * transform gives it: a vector for an uncertain input, and a sure one's value.
     */
    template <std::size_t I, typename T, typename Argument>
    void
    gatherItem(InVector<T> & x, Argument const & argument) const
    {
        constexpr int Rows = valueDimensionAtCompileTime<Name<I>>;
        auto target = middleRowsOf<Rows>(x, m_valueRows[I].start, m_valueRows[I].size);
        if constexpr (uncertain(inputOfItem<I>()))
        {
            target = argument;
        }
        else
        {
            target = argument.value.template cast<T>();
        }
    }

    /** \returns input \a J at point \a j of a batch, \a argument as evaluateBatch() is given it: its
     * column \a j if the input is uncertain, and itself otherwise
     */
    template <std::size_t J, typename Argument>
    [[nodiscard]] static decltype(auto)
    columnOf(Argument const & argument, Eigen::Index j)
    {
        if constexpr (uncertain(J))
        {
            return argument.col(j);
        }
        else
        {
            return (argument);
        }
    }

    /** \returns what the function returned, \a returned, with a Jacobian over the whole input split
     * into a block per uncertain input, the columns of the items that read it summed
     */
    template <typename R>
    [[nodiscard]] auto
    split(R && returned) const
    {
        using Plain = std::remove_cvref_t<R>;
        if constexpr (!isWithJacobian<Plain> || Whole)
        {
            return std::forward<R>(returned);
        }
        else
        {
            using J = std::remove_cvref_t<decltype(returned.jacobian)>;
            static_assert(
                MatrixExpression<J>,
                "a transformed function returns its Jacobian as one matrix over its input's differences, "
                "withJacobian(y, J)"
            );
            auto const & jacobian = returned.jacobian;
            EMRYS_FILTER_ASSERT(
                jacobian.cols() == differenceDimension(m_in)
                && "a returned Jacobian has a column per difference coordinate of the function's input"
            );
            auto blocks = [&]<std::size_t... J_>(std::index_sequence<J_...>) {
                return std::tuple_cat(inputBlock<J_>(jacobian)...);
            }(std::make_index_sequence<InputCount>{});
            if constexpr (UncertainCount == 1)
            {
                return withJacobian(std::move(returned.value), std::move(std::get<0>(blocks)));
            }
            else
            {
                return withJacobian(std::move(returned.value), std::move(blocks));
            }
        }
    }

    /** \returns a tuple of the columns of \a jacobian for input \a J, the sum of the items that read
     * it, if it is uncertain, and an empty tuple otherwise
     */
    template <std::size_t J, typename Jacobian>
    [[nodiscard]] auto
    inputBlock(Jacobian const & jacobian) const
    {
        if constexpr (!uncertain(J))
        {
            return std::tuple<>{};
        }
        else
        {
            using Traits = ArgTraits<InputAt<J>>;
            using Block = MatrixOf<
                typename Jacobian::Scalar,
                Jacobian::RowsAtCompileTime,
                Traits::differenceSize,
                Jacobian::MaxRowsAtCompileTime,
                Traits::differenceMaxSize>;
            Block block = Block::Zero(jacobian.rows(), argInput(std::get<J>(m_inputs)).differenceDimension());
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                (
                    [&] {
                        if constexpr (inputOfItem<I>() == J)
                        {
                            block += jacobian.middleCols(m_differenceRows[I].start, m_differenceRows[I].size);
                        }
                    }(),
                    ...
                );
            }(std::make_index_sequence<ItemCount>{});
            return std::tuple{std::move(block)};
        }
    }

    Model const & m_model;                                ///< the model
    std::tuple<Inputs const &...> m_inputs;               ///< the inputs
    In m_in;                                              ///< the input space instance
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Out m_out;             ///< the output space instance
    std::array<BlockRows, ItemCount> m_valueRows{};       ///< each item's rows in the input's values
    std::array<BlockRows, ItemCount> m_differenceRows{};  ///< each item's rows in the input's differences
};

/** \returns the %Gaussian \a result with the noise \a noise added: its covariance, and its mean moved
 * by the noise's mean in the output space
 */
template <typename Space, typename Scalar, typename Noise>
[[nodiscard]] GaussianParams<Space, Scalar>
plusNoise(GaussianParams<Space, Scalar> const & result, Noise const & noise)
{
    using Matrix = typename GaussianParams<Space, Scalar>::MatrixType;
    Matrix cov = symmetricFromLower<Matrix>(result.covarianceLower());
    cov += symmetricFromLower<Matrix>(noise.covarianceLower().template cast<Scalar>());
    return GaussianParams<Space, Scalar>{
        result.space(),
        Emrys::plus(result.space(), result.mean(), noise.mean().template cast<Scalar>()),
        std::move(cov)
    };
}

/** \returns the OutputWithCross \a result with the noise \a noise added to \f$ N \f$, the part of the
 * output covariance the other inputs explain, and its mean moved by the noise's mean
 */
template <typename InputSpace, typename Space, typename Form, typename Scalar, typename Noise>
[[nodiscard]] OutputWithCross<InputSpace, Space, Form, Scalar>
plusNoise(OutputWithCross<InputSpace, Space, Form, Scalar> const & result, Noise const & noise)
{
    using Result = OutputWithCross<InputSpace, Space, Form, Scalar>;
    using Matrix = typename Result::OutputMatrix;
    Matrix inputs = result.inputsCovariance();
    inputs += symmetricFromLower<Matrix>(noise.covarianceLower().template cast<Scalar>());
    auto const & space = result.output().space();
    typename Result::OutputVector mean =
        Emrys::plus(space, result.output().mean(), noise.mean().template cast<Scalar>());
    if constexpr (std::is_same_v<Form, RootForm>)
    {
        return Result{
            space,
            std::move(mean),
            typename Result::RootMatrix{result.root()},
            typename Result::CrossMatrix{result.whitenedCross()},
            Matrix{result.explainedCovariance()},
            std::move(inputs),
            Matrix{result.unexplainedCovariance()}
        };
    }
    else
    {
        return Result{
            space,
            std::move(mean),
            typename Result::JacobianMatrix{result.jacobian()},
            typename Result::CrossMatrix{result.crossCovariance()},
            Matrix{result.explainedCovariance()},
            std::move(inputs),
            Matrix{result.unexplainedCovariance()}
        };
    }
}

/** \returns whether the inputs of the ModelTransform \a G match the items of its reads() one to one,
 * refusing at compile time where they don't. A function's check rather than the class's, so that a
 * compiler stops here before pushing mismatched inputs through.
 */
template <typename G>
[[nodiscard]] consteval bool
inputsMatch()
{
    static_assert(
        G::ItemsGiven,
        "the function reads an input that transform() was not given: give each item of reads() as an uncertain "
        "input over its tag's space, or as sure<Tag>(value)"
    );
    static_assert(
        G::InputsRead,
        "transform() was given an input the function does not read: reads() names each input by the tag of its "
        "space, and an untagged input is read only by a function without reads()"
    );
    return G::ItemsGiven && G::InputsRead;
}

/** \returns \a model pushed through by \a method at \a inputs (see transform()): the positional
 * transform of the model's function over the inputs, with its affine coordinates factored out of a
 * sigma-point method, and its additive noise added
 */
template <typename Method, typename Model, typename... Inputs>
[[nodiscard]] auto
transformModel(Method const & method, Model const & model, Inputs const &... inputs)
{
    static_assert(
        Model::HasFunction,
        "a model known by its likelihood alone has no function for transform() to push its inputs through"
    );
    if constexpr (leadsWithReference<Inputs...>)
    {
        return [&]<typename About, typename... Rest>(About const & about, Rest const &... rest) {
            using G = ModelTransform<Model, Rest...>;
            if constexpr (inputsMatch<G>())
            {
                auto const g = G{model, rest...};
                return g.withNoise(transformBy(method, g, g.out(), about, rest...));
            }
        }(inputs...);
    }
    else if constexpr (!inputsMatch<ModelTransform<Model, Inputs...>>())
    {
        return;
    }
    else
    {
        using G = ModelTransform<Model, Inputs...>;
        auto const g = G{model, inputs...};
        if constexpr (isSigmaPointMethod<Method> && G::factorsAffine())
        {
            auto const factored =
                FactoredAt<G::affineSize(), G::affineMaxSize(), Method>{method, g.affineCoordinates()};
            return g.withNoise(transformBy(factored, g, g.out(), inputs...));
        }
        else
        {
            return g.withNoise(transformBy(method, g, g.out(), inputs...));
        }
    }
}

}  // namespace internal

/** \ingroup Transforms
 * \returns the distribution of \f$ y = f(x) \boxplus v \f$ computed by \a method (Linearized, or a
 * sigma-point method), with \a f a function and \a args its policies followed by its inputs
 * (decisions.md [transform-policies]): a %Gaussian in the output space, or with an input marked by
 * withCross(), an OutputWithCross.
 *
 * The function takes <tt>f(x)</tt>, <tt>f(in, x)</tt>, or <tt>f(in, out, x)</tt>, as a measurement
 * model's does, and the policies are a measurement model's, each a call before the inputs:
 * reads(), whose items each name an input by its tag; writes() or outputSpace() for the output space;
 * affineIn(), the inputs the function is affine in, which a sigma-point method factors out
 * (decisions.md [affine-coordinates]); and additiveNoise(), one %Gaussian \f$ v \f$ added to the
 * output. Without reads(), \a f reads the one uncertain input whole, as \f$ x \f$. With it, each
 * input is a distribution over a tagged space, sure<Tag>(value), or one of them marked by
 * withCross(), matched to the items of reads() by tag, and \f$ x \f$ is a vector in their Composite,
 * read with <tt>block<Tag>(in, x)</tt>. The output space is outputSpace()'s, given an instance or a
 * function of the input space; else the Composite of writes(); else the noise's space; else a Linear
 * sized by what \a f returns.
 *
 * With a linearizedAbout() reference first among the inputs, the input after it, the only one
 * withCross() may then mark, is linearized about the reference rather than over itself
 * (decisions.md [linearization-reference]): the affine model \f$ y \approx v \boxplus A \delta \f$ with
 * error covariance \f$ \Omega \f$ is taken at the reference's point, or over its distribution, in
 * the input's chart, and the result is its mean \f$ v \f$ and covariance
 * \f$ A P A^T + \Omega \f$, with cross-covariance \f$ P A^T \f$. About the input itself this is
 * the transform without a reference, to rounding.
 *
 * \code
 * auto const y = transform(methods::unscented(), [](auto const & x) { return polar(x); }, state);
 * auto const range = transform(methods::cubature(), rangeOf, reads(path<Position>, path<ScaleNoise>),
 *                              affineIn(path<ScaleNoise>), withCross(position), scaleNoise);
 * \endcode
 *
 * \sa \ref Transforms, condition(), linearizedAbout()
 */
template <typename Method, typename F, typename... Args>
    requires(!MeasurementModel<F> && !internal::isModel<F> && internal::crossCount<Args...> <= 1)
[[nodiscard]] auto
transform(Method const & method, F const & f, Args const &... args)
{
    static_assert(
        !internal::policyAfterInput<Args...>,
        "transform() takes its policies before its inputs: transform(method, f, reads(...), additiveNoise(...), "
        "inputs...)"
    );
    constexpr std::size_t Policies = internal::leadingPolicies<Args...>;
    auto const all = std::tuple<Args const &...>{args...};
    return [&]<std::size_t... P, std::size_t... I>(std::index_sequence<P...>, std::index_sequence<I...>) {
        auto const model = measurementModel(f, std::get<P>(all)...);
        return internal::transformModel(method, model, std::get<Policies + I>(all)...);
    }(std::make_index_sequence<Policies>{}, std::make_index_sequence<sizeof...(Args) - Policies>{});
}

/** \ingroup Transforms
 * \returns the distribution of the measurement model \a model's output at \a inputs, computed by
 * \a method, as transform(method, f, policies..., inputs...) computes it with the model's function
 * and policies: the model's predicted measurement, with its additive noise, where the inputs are
 * the blocks it reads (decisions.md [transform-policies]). A process model is refused, since its
 * step is predict()'s.
 */
template <typename Method, typename Model, typename... Inputs>
    requires((MeasurementModel<Model> || internal::isModel<Model>) && internal::crossCount<Inputs...> <= 1)
[[nodiscard]] auto
transform(Method const & method, Model const & model, Inputs const &... inputs)
{
    static_assert(
        MeasurementModel<Model>,
        "transform() pushes its inputs through a measurement model or a function; a process model's step is "
        "predict()'s"
    );
    static_assert(
        internal::leadingPolicies<Inputs...> == 0,
        "transform() of a model takes the model's own policies, so give the inputs alone after it"
    );
    return internal::transformModel(method, model, inputs...);
}

}  // namespace Emrys
