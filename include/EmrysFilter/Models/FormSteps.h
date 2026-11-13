#pragma once

/** \file
 * The prediction and update steps of the square-root and information forms of the %Gaussian
 * (decisions.md [gaussian-forms]): internal::predictInPlace() and internal::updateInPlace() overloaded on
 * SqrtGaussianParams, InfoGaussianParams, and SqrtInfoGaussianParams, which
 * KalmanRecursion calls as it calls the covariance form's. The method's job ends at a
 * linearization, internal::linearizeMeasurementAt() for an update and internal::linearizeAt() or
 * internal::transformRoot() for a prediction, and each form folds it in with what it holds. Read
 * Reduction.h next.
 */

#include <cstddef>
#include <expected>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Triangularization.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** Whether \a C is a square-root or information form of the %Gaussian: a GaussianForm other than
 * %Gaussian itself, whose steps this header supplies and which takes measurement noise of one
 * component (decisions.md [gaussian-forms]).
 */
template <typename C>
inline constexpr bool isOtherGaussianForm = GaussianForm<C> && !isGaussian<C>;

/** A %Gaussian over \a Space_ given by its mean and a lower-triangular root of its covariance, as
 * an UncertainInput whose root() the sigma-point methods take as it is (Transform::root()): the
 * blocks a measurement reads from a square-root form, or from an information form through its
 * covarianceRoot() (decisions.md [gaussian-forms]). Its covariance is formed only on request.
 */
template <Space Space_, RealScalar Scalar_>
class RootInput
{
public:

    using Scalar = Scalar_;                               ///< the scalar type
    using SpaceType = Space_;                             ///< the space the values live in
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

    /** The %Gaussian in \a space with mean \a mean and the lower-triangular root \a root of its
     * covariance.
     */
    RootInput(Space_ space, VectorType mean, MatrixType root)
        : m_space{std::move(space)}
        , m_mean{std::move(mean)}
        , m_root{std::move(root)}
    {}

    /** \returns the space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the mean */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return m_mean;
    }

    /** \returns the lower-triangular root \f$ S \f$ of the covariance */
    [[nodiscard]] MatrixType const &
    root() const
    {
        return m_root;
    }

    /** \returns the covariance \f$ S S^T \f$, formed on request */
    [[nodiscard]] MatrixType
    covarianceLower() const
    {
        MatrixType result = multiply(asTriangular<Eigen::Lower>(m_root), m_root.transpose());
        return result;
    }

    /** \returns the number of coordinates of the mean */
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
    VectorType m_mean;                              ///< the mean
    MatrixType m_root;                              ///< the lower-triangular root
};

/** What every form folds in from a measurement (decisions.md [gaussian-forms]): the residual of the
 * observation from the predicted measurement, the Jacobian \f$ H \f$ of the measurement in the
 * whole state's differences at the point the method linearized at, and \f$ S_0 = \Omega + R \f$,
 * everything in the innovation's covariance that \f$ H P H^T \f$ does not explain: the
 * measurement's additive noise, its input noise carried through the function, and the
 * linearization error a sigma-point method sees. The innovation's covariance is then
 * \f$ H P H^T + S_0 \f$.
 *
 * \tparam Space            the state's space
 * \tparam MeasurementSpace  the measurement space
 */
template <typename Space, typename MeasurementSpace, typename Scalar>
struct MeasurementLinearization
{
    using SpaceType = Space;                        ///< the state's space
    using MeasurementSpaceType = MeasurementSpace;  ///< the measurement space

    /// the statistical linearization's types
    using Linearization = StatisticalLinearization<Space, MeasurementSpace, Scalar>;

    /** \f$ H \f$: measurement differences by state differences */
    using JacobianMatrix = typename Linearization::JacobianMatrix;

    using NoiseMatrix = typename Linearization::ErrorMatrix;  ///< \f$ S_0 \f$, over the measurement's differences
    using MeasurementVector = typename GaussianParams<MeasurementSpace, Scalar>::VectorType;  ///< a measurement
    using InnovationVector =
        typename GaussianParams<MeasurementSpace, Scalar>::DifferenceVectorType;  ///< the residual's type

    MeasurementSpace space;       ///< the measurement space
    MeasurementVector predicted;  ///< \f$ \hat z \f$, moved by the additive noise's mean
    InnovationVector residual;    ///< \f$ r = z \boxminus \hat z \f$; empty for a model read without an observation
    JacobianMatrix jacobian;      ///< \f$ H \f$
    NoiseMatrix noise;            ///< \f$ S_0 \f$, symmetric
};

/** Sets the residual of \a linearization, a MeasurementLinearization, to the observation of
 * \a resolved, a measurement resolved against the state's space, less its predicted measurement,
 * if it observes one.
 */
template <typename Scalar, typename R, typename Linearization>
void
observeLinearization(R const & resolved, Linearization & linearization)
{
    if constexpr (R::HasObservation)
    {
        linearization.residual = observedResidual<Scalar>(resolved, linearization.space, linearization.predicted);
    }
}

/** \returns the MeasurementLinearization of \a resolved, a measurement resolved against the
 * state's space \a space, over the blocks \a reads it reads, from the predicted measurement
 * \a predicted in \a measurementSpace, the Jacobian \a readJacobian in the blocks read, and the
 * unexplained covariance \a omega before additive noise: the Jacobian placed at the blocks'
 * difference coordinates, the predicted measurement moved by the additive noise's mean and the
 * observation's residual from it, and \f$ S_0 \f$ with the noise's covariance added.
 */
template <
    typename Scalar,
    typename Space,
    typename Reads,
    typename R,
    typename MeasurementSpace,
    typename Predicted,
    typename ReadJacobian,
    typename Omega>
[[nodiscard]] MeasurementLinearization<Space, MeasurementSpace, Scalar>
assembleLinearization(
    Space const & space,
    Reads const & reads,
    R const & resolved,
    MeasurementSpace const & measurementSpace,
    Predicted const & predicted,
    ReadJacobian const & readJacobian,
    Omega const & omega
)
{
    using Result = MeasurementLinearization<Space, MeasurementSpace, Scalar>;
    using MeasurementVector = typename Result::MeasurementVector;
    using InnovationVector = typename Result::InnovationVector;
    using NoiseMatrix = typename Result::NoiseMatrix;
    Eigen::Index const m = differenceDimension(measurementSpace);
    Eigen::Index const n = differenceDimension(space);
    auto result = Result{
        measurementSpace,
        MeasurementVector{},
        InnovationVector{},
        Result::JacobianMatrix::Zero(m, n),
        NoiseMatrix{}
    };
    result.jacobian(Eigen::placeholders::all, reads.differenceIndices()) = readJacobian;
    if constexpr (!R::HasAdditiveNoise)
    {
        result.predicted = predicted;
        result.noise = omega;
    }
    else
    {
        static_assert(
            !R::MixtureNoise,
            "a square-root or information form takes measurement noise of one component; mixture noise, whose "
            "branches are collapsed by moment matching, is the covariance form's (decisions.md [gaussian-forms])"
        );
        auto const & noise = resolved.noise().distribution();
        result.predicted = Emrys::plus(measurementSpace, predicted, noise.mean());
        result.noise = omega + symmetricFromLower<NoiseMatrix>(noise.covarianceLower());
    }
    result.noise = (Scalar(0.5) * (result.noise + result.noise.transpose())).eval();
    observeLinearization<Scalar>(resolved, result);
    return result;
}

/** \returns the MeasurementLinearization of \a resolved, a measurement resolved against \a space,
 * by \a method, a linear method (Linearized), at the state \a point in \a space: the measurement
 * and its Jacobian at the point, with the call's uncertain inputs carried through the function's
 * Jacobian in them. Needs no covariance, so an information form with no information linearizes at
 * its reference.
 */
template <typename Scalar, typename Space, typename R, typename Method>
[[nodiscard]] auto
linearizeMeasurementAtPoint(
    Space const & space,
    typename Moments<Space, Scalar>::VectorType const & point,
    R const & resolved,
    Method const & method
)
{
    using Reads = typename R::Reads;
    using ReadPoint = PointInput<typename Reads::JointSpace, Scalar>;
    auto const & reads = resolved.reads();
    auto const read = ReadPoint{reads.space(), typename ReadPoint::VectorType{point(reads.indices())}};
    auto const f = ResolvedFunction{resolved};
    static_assert(
        isLinearMethod<Method>,
        "a linearization at a point is for a method that linearizes once at the means"
    );
    return linearizeMeasurement(resolved, method, f, read, [&](auto const & problem, auto const & linearized) {
        using Problem = std::remove_cvref_t<decltype(problem)>;
        using Omega = typename Problem::OutputMatrix;
        Eigen::Index const k = reads.differenceIndices().size();
        auto const readJacobian =
            linearized.jacobian.template middleCols<Reads::DifferenceSize>(problem.differenceOffset(0), k);
        Omega const omega = inputsCovariance<Omega>(resolved, problem, linearized.jacobian);
        return assembleLinearization<Scalar>(
            space,
            reads,
            resolved,
            problem.outputSpace(),
            linearized.mean,
            readJacobian,
            omega
        );
    });
}

/** \returns the MeasurementLinearization of \a resolved, whose method is linear (Linearized), at
 * the state \a point in \a space (see linearizeMeasurementAtPoint() with a method).
 */
template <typename Scalar, typename Space, typename R>
[[nodiscard]] auto
linearizeMeasurementAtPoint(
    Space const & space,
    typename Moments<Space, Scalar>::VectorType const & point,
    R const & resolved
)
{
    return linearizeMeasurementAtPoint<Scalar>(space, point, resolved, resolved.method());
}

/** The last linearization of an Iterated method, about the prior mean, and how the iteration
 * ended (decisions.md [iterated-update]).
 */
template <typename Linearization>
struct IteratedLinearization
{
    Linearization linearization;  ///< the MeasurementLinearization folded in
    std::size_t iterations;       ///< the linearizations made
    bool converged;               ///< whether a step within the tolerance stopped them
};

/** A triangular root \a root of a prior covariance, \f$ P = L L^T \f$ with \f$ L \f$ triangular in
 * \a Mode (\c Eigen::Lower or \c Eigen::Upper) and zero in the other triangle, which an Iterated
 * method steps by without forming \f$ P \f$ (iterateMeasurementLinearization()).
 */
template <typename Root, unsigned int Mode>
struct TriangularRoot
{
    Root const & root;  ///< \f$ L \f$
};

/** \returns iterationScale() of the covariance \a cov */
template <typename Cov>
[[nodiscard]] auto
spreadScale(Eigen::MatrixBase<Cov> const & cov, typename Cov::Scalar tolerance)
{
    return iterationScale(cov, tolerance);
}

/** \returns iterationScale() of the covariance \f$ L L^T \f$ of \a spread, whose diagonal is the
 * squared norms of \f$ L \f$'s rows
 */
template <typename Root, unsigned int Mode>
[[nodiscard]] auto
spreadScale(TriangularRoot<Root, Mode> const & spread, typename Root::Scalar tolerance)
{
    using Scalar = typename Root::Scalar;
    EMRYS_FILTER_ASSERT(tolerance >= Scalar(0) && "an iteration's tolerance must not be negative");
    using Result = Eigen::Matrix<Scalar, Root::RowsAtCompileTime, 1, Eigen::ColMajor, Root::MaxRowsAtCompileTime, 1>;
    Result result = spread.root.rowwise().norm() * tolerance;
    return result;
}

/** \returns the block of the covariance \a cov at the coordinates \a indices */
template <typename Cov, typename Indices>
[[nodiscard]] auto
readBlockCovariance(Eigen::MatrixBase<Cov> const & cov, Indices const & indices)
{
    using Result = MatrixOf<
        typename Cov::Scalar,
        Indices::SizeAtCompileTime,
        Indices::SizeAtCompileTime,
        Indices::MaxRowsAtCompileTime,
        Indices::MaxRowsAtCompileTime>;
    Result result = lowerSymmetricBlock(cov, indices, indices);
    return result;
}

/** \returns the block of the covariance \f$ L L^T \f$ of \a spread at the coordinates \a indices,
 * from \f$ L \f$'s rows there
 */
template <typename Root, unsigned int Mode, typename Indices>
[[nodiscard]] auto
readBlockCovariance(TriangularRoot<Root, Mode> const & spread, Indices const & indices)
{
    using Result = MatrixOf<
        typename Root::Scalar,
        Indices::SizeAtCompileTime,
        Indices::SizeAtCompileTime,
        Indices::MaxRowsAtCompileTime,
        Indices::MaxRowsAtCompileTime>;
    auto const rows = spread.root(indices, Eigen::placeholders::all);
    auto result = Result{};
    result.resize(indices.size(), indices.size());
    result.noalias() = multiply(rows, rows.transpose());
    return result;
}

/** Sets \a crossed to \f$ P H^T \f$ and adds \f$ H P H^T \f$ to \a innovationCov, for the
 * Jacobian \a jacobian and the covariance \a cov
 */
template <typename Cov, typename Jacobian, typename Crossed, typename InnovationCov>
void
addSpread(
    Eigen::MatrixBase<Cov> const & cov,
    Jacobian const & jacobian,
    Crossed & crossed,
    InnovationCov & innovationCov
)
{
    crossed.noalias() = multiply(cov, jacobian.transpose());
    innovationCov.noalias() += multiply(jacobian, crossed);
}

/** Sets \a crossed to \f$ P H^T = L (H L)^T \f$ and adds \f$ (H L) (H L)^T \f$ to
 * \a innovationCov, for the Jacobian \a jacobian and the root of \a spread: two triangular
 * products and one symmetric one, no more work than with \f$ P \f$, and \f$ P \f$ never formed.
 */
template <typename Root, unsigned int Mode, typename Jacobian, typename Crossed, typename InnovationCov>
void
addSpread(
    TriangularRoot<Root, Mode> const & spread,
    Jacobian const & jacobian,
    Crossed & crossed,
    InnovationCov & innovationCov
)
{
    using Scaled = std::remove_cvref_t<decltype(jacobian.eval())>;
    auto scaled = Scaled{};
    scaled.resize(jacobian.rows(), jacobian.cols());
    scaled.noalias() = multiply(jacobian, asTriangular<Mode>(spread.root));
    crossed.noalias() = multiply(asTriangular<Mode>(spread.root), scaled.transpose());
    innovationCov.noalias() += multiply(scaled, scaled.transpose());
}

/** Carries \a linearization, a MeasurementLinearization of \a resolved taken at the point
 * \f$ b \boxplus \delta \f$ in its own chart, to the chart of the base point \f$ b \f$, whose
 * blocks read \a reads are at \a readBase, with \a step the \f$ \delta \f$ (decisions.md
 * [iterated-update], [linearization-reference]): the Jacobian in the blocks read times the reset Jacobian
 * (toPriorChart()), the predicted measurement moved back along it to \f$ \delta = 0 \f$, and the residual
 * formed again from there.
 */
template <typename Scalar, typename Reads, typename R, typename ReadBase, typename Step, typename Linearization>
void
toBaseChart(
    Reads const & reads,
    R const & resolved,
    ReadBase const & readBase,
    Step const & step,
    Linearization & linearization
)
{
    using ReadDifference = typename Moments<typename Reads::JointSpace, Scalar>::DifferenceVectorType;
    using ReadJacobian = MatrixOf<
        Scalar,
        Linearization::JacobianMatrix::RowsAtCompileTime,
        Reads::DifferenceSize,
        Linearization::JacobianMatrix::MaxRowsAtCompileTime,
        Reads::DifferenceMaxSize>;
    ReadDifference const readStep = step(reads.differenceIndices());
    ReadJacobian readJacobian = linearization.jacobian(Eigen::placeholders::all, reads.differenceIndices());
    toPriorChart(reads, readJacobian, readBase, readStep);
    linearization.jacobian(Eigen::placeholders::all, reads.differenceIndices()) = readJacobian;
    auto back = typename Linearization::InnovationVector{};
    back.resize(linearization.jacobian.rows());
    back.noalias() = -multiply(linearization.jacobian, step);
    linearization.predicted = Emrys::plus(linearization.space, linearization.predicted, back);
    observeLinearization<Scalar>(resolved, linearization);
}

/** \returns the MeasurementLinearization of \a resolved, a measurement resolved against \a space,
 * linearized about \a about, a linearizedAbout() reference, in the chart of the state \a base in
 * \a space, which a form folds in as it folds one at its mean (decisions.md
 * [linearization-reference]): by a linear method at the reference's point
 * (linearizeMeasurementAtPoint()), or by a sigma-point method over the reference's distribution,
 * its points spread from the lower-triangular factor of its covariance at the blocks read
 * (linearizeMeasurementAtRoot()), then carried to \a base's chart (toBaseChart()).
 */
template <typename Scalar, typename Space, typename R, typename Reference>
[[nodiscard]] auto
linearizeMeasurementAbout(
    Space const & space,
    typename Moments<Space, Scalar>::VectorType const & base,
    LinearizedAbout<Reference> const & about,
    R const & resolved
)
{
    using About = LinearizedAbout<Reference>;
    using Method = typename R::Method;
    using Reads = typename R::Reads;
    using Vector = typename Moments<Space, Scalar>::VectorType;
    using Difference = typename Moments<Space, Scalar>::DifferenceVectorType;
    static_assert(checkReference<Method, About, Space>());
    auto const & reads = resolved.reads();
    Vector const point = about.mean();
    EMRYS_FILTER_ASSERT(
        point.size() == valueDimension(space) && "a point to linearize about must have the prior's value dimension"
    );
    auto linearization = [&] {
        if constexpr (isLinearMethod<Method>)
        {
            return linearizeMeasurementAtPoint<Scalar>(space, point, resolved);
        }
        else if constexpr (About::IsDistribution)
        {
            using Input = RootInput<typename Reads::JointSpace, Scalar>;
            auto const cov = readBlockCovariance(about.reference.covarianceLower(), reads.differenceIndices());
            auto const input = Input{
                reads.space(),
                typename Input::VectorType{point(reads.indices())},
                lowerTriangularRoot(defaultRoot(cov))
            };
            return linearizeMeasurementAtRoot<Scalar>(space, reads, input, resolved);
        }
    }();
    using ReadVector = typename Moments<typename Reads::JointSpace, Scalar>::VectorType;
    Difference const step = Emrys::minus(space, point, base);
    ReadVector const readBase = base(reads.indices());
    toBaseChart<Scalar>(reads, resolved, readBase, step, linearization);
    return linearization;
}

/** \returns the MeasurementLinearization of \a resolved, a measurement resolved against \a space
 * whose method is Iterated, about the prior mean \a mean in \a space, a form folds in as it folds
 * one by Linearized (decisions.md [iterated-update]): the measurement linearized at each iterate
 * \f$ \mu \boxplus \delta_i \f$, its Jacobian carried to the prior's chart and its prediction to
 * \f$ \delta = 0 \f$, as Iterated describes, with the step
 * \f$ \delta_{i+1} = P H^T (H P H^T + S_0)^{-1} r \f$ from the prior's \a spread: its covariance, or
 * a TriangularRoot of it, which the square-root forms pass so that no covariance is formed. The
 * step is only the next point to linearize at; the form's own fold of the last linearization
 * gives the posterior. Without a spread, as an information form with no information has none,
 * there is nothing to step by, and the one linearization is at \a mean. An iterate's
 * \f$ H P H^T + S_0 \f$ that is not positive definite ends the iteration with
 * GaussianError::NotPositiveDefinite (decisions.md [innovation-cholesky]).
 */
template <typename Scalar, typename Space, typename Spread, typename R>
[[nodiscard]] auto
iterateMeasurementLinearization(
    Space const & space,
    typename Moments<Space, Scalar>::VectorType const & mean,
    std::optional<Spread> const & spread,
    R const & resolved
)
{
    using Reads = typename R::Reads;
    using Vector = typename Moments<Space, Scalar>::VectorType;
    using Difference = typename Moments<Space, Scalar>::DifferenceVectorType;
    using ReadVector = typename Moments<typename Reads::JointSpace, Scalar>::VectorType;
    using ReadDifference = typename Moments<typename Reads::JointSpace, Scalar>::DifferenceVectorType;
    auto const & method = resolved.method();
    auto const & reads = resolved.reads();
    ReadVector const readMean = mean(reads.indices());
    Difference delta = Difference::Zero(differenceDimension(space));
    std::size_t iteration = 0;
    bool converged = false;

    // A sigma-point method spreads its points by a lower-triangular root of the posterior covariance
    // of the blocks read at the iterate, in the iterate's chart, the prior's at first [sigma-point-sets]. A linear
    // method reads no covariance, so it keeps none.
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
    auto spreadRoot = SpreadMatrix{};
    if constexpr (Spreads)
    {
        EMRYS_FILTER_ASSERT(spread && "a sigma-point method needs the prior's covariance to spread its points");
        priorReadCov = readBlockCovariance(*spread, reads.differenceIndices());
        spreadRoot = lowerTriangularRoot(defaultRoot(priorReadCov));
    }

    // The linearization about the iterate mu + delta: by Linearized at it, or by a sigma-point method
    // over its posterior (spreadRoot). Its own lambda, since MSVC rejects a lambda inside one whose
    // call decltype names.
    auto const linearizeAtIterate = [&] {
        if constexpr (Spreads)
        {
            auto const input = RootInput<typename Reads::JointSpace, Scalar>{
                reads.space(),
                iteration == 0 ? readMean
                               : ReadVector{Emrys::plus(reads.space(), readMean, delta(reads.differenceIndices()))},
                spreadRoot
            };
            return linearizeMeasurementAtRoot<Scalar>(space, reads, input, resolved, method.method);
        }
        else
        {
            return linearizeMeasurementAtPoint<Scalar>(
                space,
                iteration == 0 ? mean : Vector{Emrys::plus(space, mean, delta)},
                resolved,
                method.method
            );
        }
    };

    // That linearization about delta = 0 in the prior's chart.
    auto const atIterate = [&] {
        auto result = linearizeAtIterate();
        if (iteration > 0)
        {
            toBaseChart<Scalar>(reads, resolved, readMean, delta, result);
        }
        return result;
    };

    using Linearization = decltype(atIterate());
    using Result = std::expected<IteratedLinearization<Linearization>, GaussianError>;
    if (!spread)
    {
        return Result{IteratedLinearization<Linearization>{atIterate(), 1, false}};
    }
    using Gain = MatrixOf<
        Scalar,
        Linearization::JacobianMatrix::ColsAtCompileTime,
        Linearization::JacobianMatrix::RowsAtCompileTime,
        Linearization::JacobianMatrix::MaxColsAtCompileTime,
        Linearization::JacobianMatrix::MaxRowsAtCompileTime>;
    using NoiseMatrix = typename Linearization::NoiseMatrix;
    auto const scale = spreadScale(*spread, Scalar(method.tolerance));
    // A step that moved too little to count stops the iteration at the linearization after it.
    for (; iteration + 1 < method.maxIterations && !converged; ++iteration)
    {
        Linearization const linearization = atIterate();
        Gain crossed(delta.size(), linearization.residual.size());
        NoiseMatrix innovationCov = linearization.noise;
        addSpread(*spread, linearization.jacobian, crossed, innovationCov);
        auto const llt = Eigen::LLT<NoiseMatrix>{innovationCov};
        if (llt.info() != Eigen::Success)
        {
            return Result{std::unexpected{GaussianError::NotPositiveDefinite}};
        }
        auto next = Difference{};
        next.resize(delta.size());
        next.noalias() = multiply(crossed, llt.solve(linearization.residual));
        if constexpr (Spreads)
        {
            // The posterior covariance of the blocks read, P_SS - C_S S^-1 C_S^T with C_S the rows of
            // P H^T at them, carried to the next iterate's chart.
            auto const readCross = std::as_const(crossed)(reads.differenceIndices(), Eigen::placeholders::all);
            using ReadCross = MatrixOf<
                Scalar,
                Reads::DifferenceSize,
                Gain::ColsAtCompileTime,
                Reads::DifferenceMaxSize,
                Gain::MaxColsAtCompileTime>;
            ReadMatrix posterior = priorReadCov;
            posterior.noalias() -=
                multiply(transposedSolve<ReadCross>(llt, readCross.transpose()), readCross.transpose());
            if constexpr (!hasIdentityReset<typename Reads::JointSpace>)
            {
                ReadDifference const nextReadStep = next(reads.differenceIndices());
                auto const reset = resetJacobian(reads.space(), readMean, nextReadStep);
                posterior = multiply(multiply(reset, posterior), reset.transpose()).eval();
            }
            spreadRoot = lowerTriangularRoot(defaultRoot(posterior));
        }
        converged = withinScale(next, delta, scale);
        delta = next;
    }
    return Result{IteratedLinearization<Linearization>{atIterate(), iteration + 1, converged}};
}

/** \returns \a update, a FilterUpdate or one in a \c std::expected, with each branch reporting the
 * iterations of \a iterated, an IteratedLinearization
 */
template <typename Update, typename Iterated>
[[nodiscard]] Update
withIterations(Update update, Iterated const & iterated)
{
    auto const report = [&](auto & filterUpdate) {
        for (auto & branch : filterUpdate.branches)
        {
            branch.iterations = iterated.iterations;
            branch.converged = iterated.converged;
        }
    };
    if constexpr (requires { update.has_value(); })
    {
        if (update)
        {
            report(*update);
        }
    }
    else
    {
        report(update);
    }
    return update;
}

/** \returns the MeasurementLinearization of \a resolved, a measurement resolved against \a space,
 * by any method, over the blocks \a reads of a state in \a space given as \a read, an
 * UncertainInput over those blocks with a root() of its own or a covariance: the measurement's
 * prediction by the method, then its statistical linearization in the blocks read
 * [covariance-update-forms], \f$ H_S = \Psi^T S^{-1} \f$ for the root \f$ S \f$ that spread the
 * points, or the Jacobian, and \f$ N + \Omega \f$ from the prediction's parts (decisions.md
 * [linearization-forms]). The method is \a byMethod, the measurement's own or, at an iterate of an
 * Iterated method, its inner method (decisions.md [sigma-point-sets]).
 */
template <typename Scalar, typename Space, typename Reads, typename Read, typename R, typename Method>
[[nodiscard]] auto
linearizeMeasurementAtRoot(
    Space const & space,
    Reads const & reads,
    Read const & read,
    R const & resolved,
    Method const & method
)
{
    auto const f = ResolvedFunction{resolved};
    auto const predicted = transformMeasurement(resolved, method, f, read);
    // H_S and N + Omega from the prediction's parts: a solve with the root that spread the points, the
    // read blocks' own factor where they hold one, and no subtraction (decisions.md [linearization-forms]).
    auto const [jacobian, omega] = statisticalLinearization(predicted);
    return assembleLinearization<Scalar>(
        space,
        reads,
        resolved,
        predicted.output().space(),
        predicted.output().mean(),
        jacobian,
        omega
    );
}

/** linearizeMeasurementAtRoot() by the measurement's own method. */
template <typename Scalar, typename Space, typename Reads, typename Read, typename R>
[[nodiscard]] auto
linearizeMeasurementAtRoot(Space const & space, Reads const & reads, Read const & read, R const & resolved)
{
    return linearizeMeasurementAtRoot<Scalar>(space, reads, read, resolved, resolved.method());
}

/** The MeasurementLinearization of the measurement \a M over a state in \a Space, whatever its
 * method: in the measurement space the measurement resolves to there.
 */
template <typename Scalar, typename Space, typename M>
using LinearizationOf = MeasurementLinearization<Space, typename ResolvedMeasurement<M, Space>::Out, Scalar>;

/** \returns the covariance of an innovation under a prior with no information in some direction:
 * infinite on the diagonal, so that the log-likelihood it gives is \f$ -\infty \f$ and the NIS zero
 */
template <typename Matrix>
[[nodiscard]] Matrix
infiniteCovariance(Eigen::Index m)
{
    Matrix result = Matrix::Zero(m, m);
    result.diagonal().setConstant(std::numeric_limits<typename Matrix::Scalar>::infinity());
    return result;
}

/** \returns the FilterUpdate of one branch with the innovation \a innovation, accepted unless
 * \a accepted says otherwise
 */
template <typename MeasurementSpace, typename Scalar>
[[nodiscard]] FilterUpdate<MeasurementSpace, 1, Scalar>
oneBranch(Innovation<MeasurementSpace, Scalar> innovation, bool accepted = true)
{
    return FilterUpdate<MeasurementSpace, 1, Scalar>{
        {FilterBranch<MeasurementSpace, Scalar>{Scalar(0), std::move(innovation), 0, 0}},
        Scalar(0),
        accepted
    };
}

/** Folds the linearization \a linearization of a measurement into the square-root form \a state
 * (decisions.md [gaussian-forms]): the joint of the innovation and the state has the rectangular root
 * \f[ W = \begin{bmatrix} H S & S_0^{1/2} \\ S & 0 \end{bmatrix}, \qquad
 *     W W^T = \begin{bmatrix} H P H^T + S_0 & H P \\ P H^T & P \end{bmatrix}, \f]
 * whose lower-triangular factor \f$ L \f$ holds the innovation's factor \f$ L_{yy} \f$, the gain
 * through \f$ K = L_{xy} L_{yy}^{-1} \f$, and the posterior factor \f$ L_{xx} \f$, since
 * \f$ L_{xx} L_{xx}^T = P - K (H P H^T + S_0) K^T \f$. No covariance is formed, and the posterior
 * factor is positive semidefinite by construction. The mean moves by \f$ K r \f$ and the factor is
 * carried to the tangent space at the new mean by the reset Jacobian [covariance-reset], applied to the factor
 * and triangularized again. This is the array form of the Kalman update of
 * \ref Kaminski1971 "Kaminski, Bryson, and Schmidt (1971)", with \f$ H \f$ and \f$ S_0 \f$ the
 * statistical linearization of whatever method made them.
 *
 * \a accept, an acceptor (see AcceptEvery), is asked on the NIS before \a state is written
 * (decisions.md [update-outcomes]).
 *
 * \returns the update with its one branch, whose innovation has its NIS and log-determinant from
 *          \f$ L_{yy} \f$, not accepted, with \a state unchanged, if \a accept refused it; or
 *          GaussianError::NotPositiveDefinite, with \a state unchanged, if \f$ L_{yy} \f$ has a zero
 *          on its diagonal, which makes the innovation covariance singular (decisions.md
 *          [innovation-cholesky])
 */
template <typename Space, typename Scalar, typename MeasurementSpace, typename Accept = AcceptEvery>
[[nodiscard]] std::expected<FilterUpdate<MeasurementSpace, 1, Scalar>, GaussianError>
foldIntoSqrt(
    SqrtGaussianParams<Space, Scalar> & state,
    MeasurementLinearization<Space, MeasurementSpace, Scalar> const & linearization,
    Accept const & accept = {}
)
{
    using Form = SqrtGaussianParams<Space, Scalar>;
    using InnovationGaussian = GaussianParams<MeasurementSpace, Scalar>;
    constexpr int N = Form::DifferenceDimensionAtCompileTime;
    constexpr int NMax = Form::MaxDifferenceDimensionAtCompileTime;
    constexpr int M = InnovationGaussian::DifferenceDimensionAtCompileTime;
    constexpr int MMax = InnovationGaussian::MaxDifferenceDimensionAtCompileTime;
    using Joint = MatrixOf<Scalar, sizeSum<M, N>, sizeSum<M, N>, sizeSum<MMax, NMax>, sizeSum<MMax, NMax>>;
    using GainFactor = MatrixOf<Scalar, N, M, NMax, MMax>;
    auto const & space = state.space();
    Eigen::Index const n = state.differenceDimension();
    Eigen::Index const m = linearization.residual.size();
    auto const & factor = state.root();

    Joint array(m + n, m + n);
    array.topLeftCorner(m, n).noalias() = multiply(linearization.jacobian, asTriangular<Eigen::Lower>(factor));
    array.topRightCorner(m, m) = defaultRoot(linearization.noise);
    array.bottomLeftCorner(n, n) = factor;
    array.bottomRightCorner(n, m).setZero();
    Joint const lower = lowerTriangularRoot(array);
    auto const innovationFactor = blockOf<M, M>(lower, 0, 0, m, m);
    if ((innovationFactor.diagonal().array() == Scalar(0)).any())
    {
        return std::unexpected{GaussianError::NotPositiveDefinite};
    }
    GainFactor const gainFactor = lower.block(m, 0, n, m);

    typename InnovationGaussian::DifferenceVectorType whitened = linearization.residual;
    innovationFactor.template triangularView<Eigen::Lower>().solveInPlace(whitened);
    auto const statistics = InnovationStatistics<Scalar>{
        whitened.squaredNorm(),
        Scalar(2) * innovationFactor.diagonal().array().abs().log().sum()
    };
    typename InnovationGaussian::MatrixType innovationCov = multiply(innovationFactor, innovationFactor.transpose());
    if constexpr (acceptsByNis<Accept>)
    {
        if (!accept(statistics.nis, m))
        {
            return oneBranch(
                Innovation<MeasurementSpace, Scalar>{linearization.residual, std::move(innovationCov), statistics},
                false
            );
        }
    }

    typename Form::DifferenceVectorType const step = multiply(gainFactor, whitened);
    typename Form::VectorType const start = state.mean();
    typename Form::MatrixType posterior = lower.bottomRightCorner(n, n);
    if constexpr (!hasIdentityReset<Space>)
    {
        auto const g = resetJacobian(space, start, step);
        posterior = lowerTriangularRoot(typename Form::MatrixType{multiply(g, posterior)});
    }
    state.setMean(Emrys::plus(space, start, step));
    state.setRoot(std::move(posterior));
    return oneBranch(
        Innovation<MeasurementSpace, Scalar>{linearization.residual, std::move(innovationCov), statistics}
    );
}

/** The blocks \a reads of the square-root form \a state as a RootInput: the mean's entries and
 * the lower-triangular factor of the factor's rows at the blocks, by triangularization, so that
 * the sigma points of an update spread from the state's own factor (decisions.md [gaussian-forms]).
 */
template <typename Reads, typename Space, typename Scalar>
[[nodiscard]] RootInput<typename Reads::JointSpace, Scalar>
readRoot(Reads const & reads, SqrtGaussianParams<Space, Scalar> const & state)
{
    using Result = RootInput<typename Reads::JointSpace, Scalar>;
    using Rows = MatrixOf<
        Scalar,
        Reads::DifferenceSize,
        SqrtGaussianParams<Space, Scalar>::DifferenceDimensionAtCompileTime,
        Reads::DifferenceMaxSize,
        SqrtGaussianParams<Space, Scalar>::MaxDifferenceDimensionAtCompileTime>;
    Rows const rows = state.root()(reads.differenceIndices(), Eigen::placeholders::all);
    return Result{reads.space(), typename Result::VectorType{state.mean()(reads.indices())}, lowerTriangularRoot(rows)};
}

/** The blocks \a reads of an information form as a RootInput, from its reference \a point and the
 * upper-triangular \a covarianceRoot \f$ R^{-1} \f$ with \f$ R^{-1} R^{-T} = \Sigma \f$ (its
 * covarianceRoot()): the lower-triangular factor of the root's rows at the blocks, by
 * triangularization, so that no covariance is formed (decisions.md [gaussian-forms]).
 */
template <typename Reads, typename Point, typename Root>
[[nodiscard]] RootInput<typename Reads::JointSpace, typename Root::Scalar>
readRootOf(Reads const & reads, Point const & point, Root const & covarianceRoot)
{
    using Scalar = typename Root::Scalar;
    using Result = RootInput<typename Reads::JointSpace, Scalar>;
    using Rows = MatrixOf<
        Scalar,
        Reads::DifferenceSize,
        Root::ColsAtCompileTime,
        Reads::DifferenceMaxSize,
        Root::MaxColsAtCompileTime>;
    Rows const rows = covarianceRoot(reads.differenceIndices(), Eigen::placeholders::all);
    return Result{reads.space(), typename Result::VectorType{point(reads.indices())}, lowerTriangularRoot(rows)};
}

/** \a state, a square-root form, conditioned in place on \a measurement, whose noise has one
 * component, by the measurement's own method (decisions.md [gaussian-forms]): a linear method is linearized
 * at the mean, and a sigma-point method spreads its points from the factor's rows at the blocks
 * read, then foldIntoSqrt(). Given a linearizedAbout() \a reference, the measurement is linearized
 * about it instead, in the mean's chart (linearizeMeasurementAbout(), [linearization-reference]).
 * \a accept, an acceptor (see AcceptEvery), is asked on the NIS before \a state is written
 * (decisions.md [update-outcomes]).
 *
 * \returns the update without its posterior (see FilterUpdate), accepted unless \a accept refused
 *          it, with \a state unchanged; or GaussianError::NotPositiveDefinite, with \a state
 *          unchanged, if the innovation covariance is singular (decisions.md [innovation-cholesky])
 */
template <
    bool Statistics,
    typename Space,
    typename Scalar,
    Measurement M,
    typename Reference = NoReference,
    typename Accept = AcceptEvery>
[[nodiscard]] auto
updateInPlace(
    SqrtGaussianParams<Space, Scalar> & state,
    M const & measurement,
    Reference const & reference = {},
    Accept const & accept = {}
)
{
    static_assert(
        !hasInnovationRepair<M>,
        "repairing the innovation covariance is the covariance form's (decisions.md [gaussian-forms])"
    );
    static_assert(
        !hasConsider<M>,
        "a consider() update is the covariance form's: the square-root and information forms keep no block's "
        "covariance as it was (decisions.md [consider-states]); filter in a Gaussian"
    );
    static_assert(
        M::HasMethod,
        "a Kalman update pushes the prior through the model by a method: observe(method, model, z)"
    );
    using Method = typename M::MethodType;
    auto const & space = state.space();
    auto const resolved = resolveMeasurement(measurement, space);
    if constexpr (isLinearizedAbout<Reference>)
    {
        return foldIntoSqrt(state, linearizeMeasurementAbout<Scalar>(space, state.mean(), reference, resolved), accept);
    }
    else if constexpr (isIterated<Method>)
    {
        // The factor only places the iterates; the fold of the last linearization is the factor's own.
        using Matrix = typename SqrtGaussianParams<Space, Scalar>::MatrixType;
        auto const spread = std::optional<TriangularRoot<Matrix, Eigen::Lower>>{std::in_place, state.root()};
        auto const iterated = iterateMeasurementLinearization<Scalar>(space, state.mean(), spread, resolved);
        using Result = decltype(foldIntoSqrt(state, iterated->linearization, accept));
        if (!iterated)
        {
            return Result{std::unexpected{iterated.error()}};
        }
        return withIterations(foldIntoSqrt(state, iterated->linearization, accept), *iterated);
    }
    else if constexpr (isLinearMethod<Method>)
    {
        return foldIntoSqrt(state, linearizeMeasurementAtPoint<Scalar>(space, state.mean(), resolved), accept);
    }
    else
    {
        auto const & reads = resolved.reads();
        return foldIntoSqrt(
            state,
            linearizeMeasurementAtRoot<Scalar>(space, reads, readRoot(reads, state), resolved),
            accept
        );
    }
}

/** Folds the linearization \a linearization of a measurement into the information form \a state,
 * whose reference is the point the measurement was linearized at (decisions.md [gaussian-forms]):
 * \f$ \Lambda \leftarrow \Lambda + H^T S_0^{-1} H \f$ and
 * \f$ \eta \leftarrow \eta + H^T S_0^{-1} r \f$, the extended information filter's update
 * (\ref Mutambara1998 "Mutambara (1998)"), exact and independent of the order the measurements
 * arrive in. With \a covariance the prior's covariance at the reference, the
 * innovation's covariance is \f$ H \Sigma H^T + S_0 \f$; without one, as a prior with no
 * information has none, it is infinite (infiniteCovariance()), and its NIS zero. \a accept, an
 * acceptor (see AcceptEvery), is asked on the NIS before \a state is written (decisions.md
 * [update-outcomes]), which then factors \f$ H \Sigma H^T + S_0 \f$.
 *
 * \returns the update with its one branch, not accepted, with \a state unchanged, if \a accept
 *          refused it; or GaussianError::NotPositiveDefinite if \f$ S_0 \f$ does not factorize
 */
template <typename Space, typename Scalar, typename MeasurementSpace, typename Accept = AcceptEvery>
[[nodiscard]] std::expected<FilterUpdate<MeasurementSpace, 1, Scalar>, GaussianError>
foldIntoInfo(
    InfoGaussianParams<Space, Scalar> & state,
    MeasurementLinearization<Space, MeasurementSpace, Scalar> const & linearization,
    std::optional<typename InfoGaussianParams<Space, Scalar>::MatrixType> const & covariance,
    Accept const & accept = {}
)
{
    using InnovationMatrix = typename GaussianParams<MeasurementSpace, Scalar>::MatrixType;
    using Lin = MeasurementLinearization<Space, MeasurementSpace, Scalar>;
    Eigen::Index const m = linearization.residual.size();
    auto const llt = Eigen::LLT<InnovationMatrix>{linearization.noise};
    if (llt.info() != Eigen::Success)
    {
        return std::unexpected{GaussianError::NotPositiveDefinite};
    }
    typename Lin::JacobianMatrix whitened = linearization.jacobian;
    solveInPlace(llt, whitened);
    auto added = typename InfoGaussianParams<Space, Scalar>::MatrixType{};
    added.noalias() = multiply(linearization.jacobian.transpose(), whitened);
    auto vector = typename InfoGaussianParams<Space, Scalar>::DifferenceVectorType{};
    vector.noalias() = multiply(linearization.jacobian.transpose(), llt.solve(linearization.residual));

    auto innovation = [&] {
        if (!covariance)
        {
            // The statistics of an infinite covariance, stated rather than computed from it.
            return Innovation<MeasurementSpace, Scalar>{
                linearization.residual,
                infiniteCovariance<InnovationMatrix>(m),
                InnovationStatistics<Scalar>{Scalar(0), std::numeric_limits<Scalar>::infinity()}
            };
        }
        InnovationMatrix innovationCov = linearization.noise;
        innovationCov.noalias() +=
            multiply(multiply(linearization.jacobian, *covariance), linearization.jacobian.transpose());
        return Innovation<MeasurementSpace, Scalar>{linearization.residual, std::move(innovationCov)};
    }();
    if constexpr (acceptsByNis<Accept>)
    {
        if (!accept(innovation.nis(), m))
        {
            return oneBranch(std::move(innovation), false);
        }
    }
    state.addInfo(added, vector);
    return oneBranch(std::move(innovation));
}

/** \a state, an information form, conditioned in place on \a measurement, whose noise has one
 * component, by the measurement's own method (decisions.md [gaussian-forms]). The reference moves to the mean,
 * where the measurement is linearized; a prior with no information keeps its reference and
 * linearizes there, which a linear method can do and a sigma-point method, needing a covariance
 * root, cannot. Then foldIntoInfo(). Given a linearizedAbout() \a reference, the measurement is
 * linearized about it instead, in the chart of the form's reference
 * (linearizeMeasurementAbout(), [linearization-reference]), and a sigma-point method spreads over the
 * reference's distribution, so a prior with no information takes it too. \a accept, an acceptor
 * (see AcceptEvery), is asked on the NIS before the information is added (decisions.md
 * [update-outcomes]); one it refuses leaves the distribution as it was, about its mean.
 *
 * \returns the update without its posterior (see FilterUpdate), accepted unless \a accept refused
 *          it; or GaussianError::Singular when a sigma-point method meets a prior with no
 *          information, or the GaussianError of the precision's factorization
 */
template <
    bool Statistics,
    typename Space,
    typename Scalar,
    Measurement M,
    typename Reference = NoReference,
    typename Accept = AcceptEvery>
[[nodiscard]] auto
updateInPlace(
    InfoGaussianParams<Space, Scalar> & state,
    M const & measurement,
    Reference const & reference = {},
    Accept const & accept = {}
)
{
    static_assert(
        !hasInnovationRepair<M>,
        "repairing the innovation covariance is the covariance form's (decisions.md [gaussian-forms])"
    );
    static_assert(
        !hasConsider<M>,
        "a consider() update is the covariance form's: the square-root and information forms keep no block's "
        "covariance as it was (decisions.md [consider-states]); filter in a Gaussian"
    );
    static_assert(
        M::HasMethod,
        "a Kalman update pushes the prior through the model by a method: observe(method, model, z)"
    );
    using Form = InfoGaussianParams<Space, Scalar>;
    using Method = typename M::MethodType;
    using Lin = LinearizationOf<Scalar, Space, M>;
    using Result = std::expected<FilterUpdate<typename Lin::MeasurementSpaceType, 1, Scalar>, GaussianError>;
    auto const & space = state.space();
    auto const resolved = resolveMeasurement(measurement, space);

    auto const mean = state.mean();
    if (!mean && mean.error() != GaussianError::Singular)
    {
        return Result{std::unexpected{mean.error()}};
    }
    auto covariance = std::optional<typename Form::MatrixType>{};
    auto root = std::optional<typename Form::MatrixType>{};
    if (mean)
    {
        state.setReference(*mean);
        auto factor = state.covarianceRoot();
        if (!factor)
        {
            return Result{std::unexpected{factor.error()}};
        }
        root = std::move(*factor);
        covariance = multiply(*root, root->transpose());
    }
    if constexpr (isLinearizedAbout<Reference>)
    {
        return foldIntoInfo(
            state,
            linearizeMeasurementAbout<Scalar>(space, state.reference(), reference, resolved),
            covariance,
            accept
        );
    }
    else if constexpr (isIterated<Method>)
    {
        if constexpr (!isLinearMethod<std::remove_cvref_t<decltype(measurement.method().method)>>)
        {
            if (!covariance)
            {
                return Result{std::unexpected{GaussianError::Singular}};
            }
        }
        auto const iterated = iterateMeasurementLinearization<Scalar>(space, state.reference(), covariance, resolved);
        if (!iterated)
        {
            return Result{std::unexpected{iterated.error()}};
        }
        return withIterations(foldIntoInfo(state, iterated->linearization, covariance, accept), *iterated);
    }
    else if constexpr (isLinearMethod<Method>)
    {
        return foldIntoInfo(
            state,
            linearizeMeasurementAtPoint<Scalar>(space, state.reference(), resolved),
            covariance,
            accept
        );
    }
    else
    {
        if (!root)
        {
            return Result{std::unexpected{GaussianError::Singular}};
        }
        auto const & reads = resolved.reads();
        return foldIntoInfo(
            state,
            linearizeMeasurementAtRoot<Scalar>(space, reads, readRootOf(reads, state.reference(), *root), resolved),
            covariance,
            accept
        );
    }
}

/** Folds the linearization \a linearization of a measurement into the square-root information
 * form \a state, whose reference is the point the measurement was linearized at (decisions.md
 * [gaussian-forms]): with \f$ L_0 L_0^T = S_0 \f$, the whitened rows \f$ [\,L_0^{-1} H \;\; L_0^{-1} r\,] \f$ are
 * stacked under the data array \f$ [\,R \;\; r\,] \f$ and triangularized,
 * \f[ Q^T \begin{bmatrix} R & r \\ L_0^{-1} H & L_0^{-1} r \end{bmatrix}
 *     = \begin{bmatrix} R^+ & r^+ \\ 0 & e \end{bmatrix}, \f]
 * the measurement update of \ref Bierman1977 "Bierman (1977)", as given in
 * \ref Ke2024 "Ke et al. (2024)", §II, equations (1) to (5). The residual \f$ e \f$ is
 * the increase in the least-squares cost, which is the innovation's NIS, and the innovation's
 * log-determinant follows from the factors' diagonals:
 * \f$ \log|S| = \log|S_0| + 2 \sum_i \log|R^+_{ii}| - 2 \sum_i \log|R_{ii}| \f$. With
 * \a covarianceRoot the prior's \f$ R^{-1} \f$ the innovation's covariance is
 * \f$ H \Sigma H^T + S_0 \f$; without one it is infinite (infiniteCovariance()) and so is its
 * log-determinant. \a accept, an acceptor (see AcceptEvery), is asked on the NIS before \a state is
 * written (decisions.md [update-outcomes]).
 *
 * \returns the update with its one branch, not accepted, with \a state unchanged, if \a accept
 *          refused it; or GaussianError::NotPositiveDefinite if \f$ S_0 \f$ does not factorize
 */
template <typename Space, typename Scalar, typename MeasurementSpace, typename Accept = AcceptEvery>
[[nodiscard]] std::expected<FilterUpdate<MeasurementSpace, 1, Scalar>, GaussianError>
foldIntoSqrtInfo(
    SqrtInfoGaussianParams<Space, Scalar> & state,
    MeasurementLinearization<Space, MeasurementSpace, Scalar> const & linearization,
    std::optional<typename SqrtInfoGaussianParams<Space, Scalar>::MatrixType> const & covarianceRoot,
    Accept const & accept = {}
)
{
    using Form = SqrtInfoGaussianParams<Space, Scalar>;
    using InnovationGaussian = GaussianParams<MeasurementSpace, Scalar>;
    using InnovationMatrix = typename InnovationGaussian::MatrixType;
    constexpr int N = Form::DifferenceDimensionAtCompileTime;
    constexpr int NMax = Form::MaxDifferenceDimensionAtCompileTime;
    constexpr int M = InnovationGaussian::DifferenceDimensionAtCompileTime;
    constexpr int MMax = InnovationGaussian::MaxDifferenceDimensionAtCompileTime;
    using Stacked = MatrixOf<Scalar, sizeSum<N, M>, sizeSum<N, 1>, sizeSum<NMax, MMax>, sizeSum<NMax, 1>>;
    Eigen::Index const n = state.differenceDimension();
    Eigen::Index const m = linearization.residual.size();

    auto const llt = Eigen::LLT<InnovationMatrix>{linearization.noise};
    if (llt.info() != Eigen::Success)
    {
        return std::unexpected{GaussianError::NotPositiveDefinite};
    }
    Stacked stacked(n + m, n + 1);
    stacked.topLeftCorner(n, n) = state.infoRoot();
    stacked.col(n).head(n) = state.whitenedOffset();
    typename MeasurementLinearization<Space, MeasurementSpace, Scalar>::JacobianMatrix whitened =
        linearization.jacobian;
    solveInPlace(llt.matrixL(), whitened);
    stacked.bottomLeftCorner(m, n) = whitened;
    stacked.col(n).tail(m) = llt.matrixL().solve(linearization.residual);
    Scalar const priorLogDet = Scalar(2) * state.infoRoot().diagonal().array().abs().log().sum();
    triangularize(stacked);

    Scalar const nis = stacked.col(n).tail(m).squaredNorm();
    Scalar logDeterminant = std::numeric_limits<Scalar>::infinity();
    InnovationMatrix innovationCov = infiniteCovariance<InnovationMatrix>(m);
    if (covarianceRoot)
    {
        Scalar const posteriorLogDet = Scalar(2) * stacked.diagonal().head(n).array().abs().log().sum();
        logDeterminant = Scalar(2) * llt.matrixLLT().diagonal().array().log().sum() + posteriorLogDet - priorLogDet;
        typename MeasurementLinearization<Space, MeasurementSpace, Scalar>::JacobianMatrix const scaled =
            multiply(linearization.jacobian, *covarianceRoot);
        innovationCov = linearization.noise;
        innovationCov.noalias() += multiply(scaled, scaled.transpose());
    }
    auto innovation = Innovation<MeasurementSpace, Scalar>{
        linearization.residual,
        std::move(innovationCov),
        InnovationStatistics<Scalar>{nis, logDeterminant}
    };
    if constexpr (acceptsByNis<Accept>)
    {
        if (!accept(nis, m))
        {
            return oneBranch(std::move(innovation), false);
        }
    }
    state.setRoot(
        typename Form::MatrixType{stacked.topLeftCorner(n, n)},
        typename Form::DifferenceVectorType{stacked.col(n).head(n)}
    );
    return oneBranch(std::move(innovation));
}

/** \a state, a square-root information form, conditioned in place on \a measurement, whose
 * noise has one component, by the measurement's own method (decisions.md [gaussian-forms]), as
 * updateInPlace() of an InfoGaussianParams does, with or without a linearizedAbout() \a reference,
 * and with \a accept as it takes one, then foldIntoSqrtInfo().
 *
 * \returns the update without its posterior (see FilterUpdate), accepted unless \a accept refused
 *          it; or GaussianError::Singular when a sigma-point method meets a prior with no
 *          information
 */
template <
    bool Statistics,
    typename Space,
    typename Scalar,
    Measurement M,
    typename Reference = NoReference,
    typename Accept = AcceptEvery>
[[nodiscard]] auto
updateInPlace(
    SqrtInfoGaussianParams<Space, Scalar> & state,
    M const & measurement,
    Reference const & reference = {},
    Accept const & accept = {}
)
{
    static_assert(
        !hasInnovationRepair<M>,
        "repairing the innovation covariance is the covariance form's (decisions.md [gaussian-forms])"
    );
    static_assert(
        !hasConsider<M>,
        "a consider() update is the covariance form's: the square-root and information forms keep no block's "
        "covariance as it was (decisions.md [consider-states]); filter in a Gaussian"
    );
    static_assert(
        M::HasMethod,
        "a Kalman update pushes the prior through the model by a method: observe(method, model, z)"
    );
    using Form = SqrtInfoGaussianParams<Space, Scalar>;
    using Method = typename M::MethodType;
    using Lin = LinearizationOf<Scalar, Space, M>;
    using Result = std::expected<FilterUpdate<typename Lin::MeasurementSpaceType, 1, Scalar>, GaussianError>;
    auto const & space = state.space();
    auto const resolved = resolveMeasurement(measurement, space);

    auto const mean = state.mean();
    if (!mean && mean.error() != GaussianError::Singular)
    {
        return Result{std::unexpected{mean.error()}};
    }
    auto root = std::optional<typename Form::MatrixType>{};
    if (mean)
    {
        state.setReference(*mean);
        auto factor = state.covarianceRoot();
        root = std::move(*factor);
    }
    if constexpr (isLinearizedAbout<Reference>)
    {
        return foldIntoSqrtInfo(
            state,
            linearizeMeasurementAbout<Scalar>(space, state.reference(), reference, resolved),
            root,
            accept
        );
    }
    else if constexpr (isIterated<Method>)
    {
        auto spread = std::optional<TriangularRoot<typename Form::MatrixType, Eigen::Upper>>{};
        if (root)
        {
            spread.emplace(*root);
        }
        else if constexpr (!isLinearMethod<std::remove_cvref_t<decltype(measurement.method().method)>>)
        {
            return Result{std::unexpected{GaussianError::Singular}};
        }
        auto const iterated = iterateMeasurementLinearization<Scalar>(space, state.reference(), spread, resolved);
        if (!iterated)
        {
            return Result{std::unexpected{iterated.error()}};
        }
        return withIterations(foldIntoSqrtInfo(state, iterated->linearization, root, accept), *iterated);
    }
    else if constexpr (isLinearMethod<Method>)
    {
        return foldIntoSqrtInfo(
            state,
            linearizeMeasurementAtPoint<Scalar>(space, state.reference(), resolved),
            root,
            accept
        );
    }
    else
    {
        if (!root)
        {
            return Result{std::unexpected{GaussianError::Singular}};
        }
        auto const & reads = resolved.reads();
        return foldIntoSqrtInfo(
            state,
            linearizeMeasurementAtRoot<Scalar>(space, reads, readRootOf(reads, state.reference(), *root), resolved),
            root,
            accept
        );
    }
}

/** Moves the square-root form \a state forward by the affine step with transition \a transition,
 * process noise root \a noiseRoot (\f$ Q = G G^T \f$), and propagated mean \a mean:
 * \f$ S' = \mathrm{tri}([\,F S \;\; G\,]) \f$, the lower-triangular factor of
 * \f$ F P F^T + Q \f$ without forming it (\ref Kaminski1971 "Kaminski, Bryson, and Schmidt
 * (1971)").
 */
template <typename Space, typename Scalar, typename Transition, typename NoiseRoot, typename Mean>
void
propagateRoot(
    SqrtGaussianParams<Space, Scalar> & state,
    Transition const & transition,
    NoiseRoot const & noiseRoot,
    Mean && mean
)
{
    using Form = SqrtGaussianParams<Space, Scalar>;
    constexpr int N = Form::DifferenceDimensionAtCompileTime;
    constexpr int NMax = Form::MaxDifferenceDimensionAtCompileTime;
    using Stacked = MatrixOf<Scalar, N, sizeSum<N, N>, NMax, sizeSum<NMax, NMax>>;
    Eigen::Index const n = state.differenceDimension();
    Stacked stacked(n, 2 * n);
    stacked.leftCols(n).noalias() = multiply(transition, asTriangular<Eigen::Lower>(state.root()));
    stacked.rightCols(n) = noiseRoot;
    state.setMean(std::forward<Mean>(mean));
    state.setRoot(lowerTriangularRoot(stacked));
}

/** Moves the square-root form \a state forward by \a dt through \a model by \a method with the
 * process noise \a noise added (decisions.md [gaussian-forms]): the body of both forms of predictInPlace().
 *
 * A model that gives its Jacobian with no uncertain inputs, under Linearized, is propagated by
 * propagateRoot() with it [returned-jacobians]. Any other Linearized step is differentiated at the means and its
 * Jacobian's blocks times the inputs' roots are stacked with the noise root. A sigma-point method (Unscented, Cubature,
 * SphericalSimplex, GaussHermite) spreads its points from the factor itself (Transform::root()) and triangularizes
 * their weighted deviations with the noise root (\ref VanDerMerwe2001 "van der Merwe and Wan (2001)"); a negative
 * center weight, which only an Unscented with \f$ \alpha < 1 \f$ and a small
 * \f$ \beta \f$ gives, has its term subtracted from the formed covariance, which is then
 * factorized and may fail. Any other method predicts through
 * the covariance and factorizes the result, as does any step given a linearizedAbout() reference
 * first among \a inputs [linearization-reference].
 *
 * \returns nothing, or GaussianError::NotPositiveDefinite when a negative center weight leaves
 *          the predicted covariance indefinite
 */
template <
    typename Method,
    typename Space,
    typename Scalar,
    typename R,
    typename Noise,
    typename Reference = NoReference>
[[nodiscard]] std::expected<void, GaussianError>
predictRootWithNoise(
    Method const & method,
    SqrtGaussianParams<Space, Scalar> & state,
    R const & resolved,
    Scalar dt,
    Eigen::MatrixBase<Noise> const & noise,
    Reference const & about = {}
)
{
    using Form = SqrtGaussianParams<Space, Scalar>;
    using MatrixType = typename Form::MatrixType;
    constexpr int N = Form::DifferenceDimensionAtCompileTime;
    constexpr int NMax = Form::MaxDifferenceDimensionAtCompileTime;
    auto const & space = state.space();
    Eigen::Index const n = state.differenceDimension();
    // Through the covariance, factorized after: for a reference, and for a method with no root form.
    // Generic, so that only a branch that calls it instantiates the prediction through the moments.
    auto const throughMoments = [&]() -> std::expected<void, GaussianError> {
        auto covariance = state.covariance();
        auto const prior = GaussianParams<Space, Scalar>{space, state.mean(), std::move(*covariance)};
        GaussianParams<Space, Scalar> const predicted = predictWithNoise(method, prior, resolved, dt, noise, about);
        state.setMean(predicted.mean());
        // The pivoted root is not triangular, so it is triangularized before the form keeps its lower triangle.
        state.setRoot(lowerTriangularRoot(defaultRoot(symmetricFromLower<MatrixType>(predicted.covarianceLower()))));
        return {};
    };
    if constexpr (isLinearizedAbout<Reference>)
    {
        return throughMoments();
    }
    else if constexpr (predictsByJacobian<Method, R>)
    {
        MatrixType const noiseRoot = defaultRoot(symmetricFromLower<MatrixType>(noise));
        auto step = stepWithJacobian(resolved, state.mean(), dt);
        propagateRoot(state, step.jacobian, noiseRoot, std::move(step.value));
        return {};
    }
    else
    {
        return std::apply(
            [&](auto const &... inputs) -> std::expected<void, GaussianError> {
                MatrixType const noiseRoot = defaultRoot(symmetricFromLower<MatrixType>(noise));
                auto const f = ProcessStep{resolved, dt};
                using Input = RootInput<Space, Scalar>;
                using Problem = Transform<decltype(f) const, Space, Input, std::remove_cvref_t<decltype(inputs)>...>;
                auto const input = Input{space, state.mean(), state.root()};
                if constexpr (isLinearMethod<Method>)
                {
                    auto const problem = Problem{f, space, input, inputs...};
                    auto const linearized = linearizeAt(method, problem);
                    using Stacked = MatrixOf<
                        Scalar,
                        N,
                        sizeSum<Problem::InputDifferenceSize, N>,
                        NMax,
                        sizeSum<Problem::InputDifferenceMaxSize, NMax>>;
                    Stacked stacked(n, problem.inputDifferenceSize() + n);
                    problem.forEachUncertain([&]<std::size_t I>(auto const & in) {
                        Eigen::Index const d = in.differenceDimension();
                        Eigen::Index const offset = problem.differenceOffset(I);
                        using In = std::remove_cvref_t<decltype(in)>;
                        auto const block =
                            middleColsOf<In::DifferenceDimensionAtCompileTime>(linearized.jacobian, offset, d);
                        if constexpr (I == 0)
                        {
                            stacked.middleCols(offset, d).noalias() =
                                multiply(block, asTriangular<Eigen::Lower>(in.root()));
                        }
                        else
                        {
                            using Cov = MatrixOf<
                                Scalar,
                                In::DifferenceDimensionAtCompileTime,
                                In::DifferenceDimensionAtCompileTime,
                                In::MaxDifferenceDimensionAtCompileTime,
                                In::MaxDifferenceDimensionAtCompileTime>;
                            stacked.middleCols(offset, d).noalias() =
                                multiply(block, defaultRoot(symmetricFromLower<Cov>(covarianceLowerOf(in))));
                        }
                    });
                    stacked.rightCols(n) = noiseRoot;
                    state.setMean(typename Form::VectorType{linearized.mean});
                    state.setRoot(lowerTriangularRoot(stacked));
                    return {};
                }
                else if constexpr (isSigmaPointMethod<Method>)
                {
                    auto const problem = Problem{f, space, input, inputs...};
                    auto deviations = transformRoot(method, problem);
                    using Deviations = decltype(deviations);
                    Eigen::Index const points = deviations.root.cols();
                    if (!deviations.centerNegative)
                    {
                        using Stacked = MatrixOf<
                            Scalar,
                            N,
                            sizeSum<Deviations::Points, N>,
                            NMax,
                            sizeSum<Deviations::MaxPoints, NMax>>;
                        Stacked stacked(n, points + n);
                        stacked.leftCols(points) = deviations.root;
                        stacked.rightCols(n) = noiseRoot;
                        state.setMean(typename Form::VectorType{std::move(deviations.mean)});
                        state.setRoot(lowerTriangularRoot(stacked));
                        return {};
                    }
                    auto const spread = deviations.root.leftCols(points - 1);
                    auto const center = deviations.root.col(points - 1);
                    MatrixType cov = symmetricFromLower<MatrixType>(noise);
                    cov.noalias() += multiply(spread, spread.transpose());
                    cov.noalias() -= multiply(center, center.transpose());
                    cov = (Scalar(0.5) * (cov + cov.transpose())).eval();
                    auto const llt = Eigen::LLT<MatrixType>{cov};
                    if (llt.info() != Eigen::Success)
                    {
                        return std::unexpected{GaussianError::NotPositiveDefinite};
                    }
                    state.setMean(typename Form::VectorType{std::move(deviations.mean)});
                    state.setRoot(MatrixType{llt.matrixL()});
                    return {};
                }
                else
                {
                    return throughMoments();
                }
            },
            resolved.inputs()
        );
    }
}

/** predict() of the square-root form \a state into itself, by \a method through \a model over
 * \a dt with the call's \a args (see predictRootWithNoise()).
 */
template <typename Method, typename Space, typename Scalar, ProcessModel Model, typename... Args>
[[nodiscard]] std::expected<void, GaussianError>
predictInPlace(
    Method const & method,
    SqrtGaussianParams<Space, Scalar> & state,
    Model const & model,
    Scalar dt,
    Args const &... args
)
{
    internal::assertPartitions(model, state.space());
    return withReference(
        [&](auto const & about, auto const &... inputs) {
            auto const call = modelWith(std::cref(model), inputs...);
            auto const resolved = resolveProcess(call, state.space());
            return predictRootWithNoise(method, state, resolved, dt, resolved.template noise<Scalar>(dt), about);
        },
        args...
    );
}

/** predict() of the square-root form \a state into itself over \a model's fixed step with
 * \a inputs: by the cached transition and noise where the model caches them and there are no
 * inputs, and otherwise by the model's method over its step (see predictRootWithNoise()).
 */
template <
    typename Space,
    typename Scalar,
    typename Space_,
    typename Model_,
    typename Method_,
    typename Scalar_,
    typename... Args>
[[nodiscard]] std::expected<void, GaussianError>
predictInPlace(
    SqrtGaussianParams<Space, Scalar> & state,
    FixedStepModel<Space_, Model_, Method_, Scalar_> const & model,
    Args const &... args
)
{
    using Model = FixedStepModel<Space_, Model_, Method_, Scalar_>;
    static_assert(std::is_same_v<Space, Space_>, "the model must be over the state's space");
    static_assert(std::is_same_v<Scalar, Scalar_>, "the model's scalar must be the state's");
    internal::assertPartitions(model, state.space());
    if constexpr (Model::CachesTransition && sizeof...(Args) == 0)
    {
        propagateRoot(state, model.transition(), model.noiseRoot(), model.propagate(state.space(), state.mean()));
        return {};
    }
    else
    {
        return withReference(
            [&](auto const & about, auto const &... inputs) {
                auto const call = modelWith(std::cref(model.model()), inputs...);
                auto const resolved = resolveProcess(call, state.space());
                return predictRootWithNoise(model.method(), state, resolved, model.timeStep(), model.noise(), about);
            },
            args...
        );
    }
}

/** predict() of the information form \a state into itself, through its moments (decisions.md
 * [gaussian-forms]): the covariance form's predict() of toGaussian(), converted back, with the reference at
 * the predicted mean.
 *
 * \returns nothing, or GaussianError::Singular while \a state has no information in some
 *          direction, since the moments are then undefined
 */
template <typename Method, typename Space, typename Scalar, typename Model, typename... Inputs>
[[nodiscard]] std::expected<void, GaussianError>
predictInPlace(
    Method const & method,
    InfoGaussianParams<Space, Scalar> & state,
    Model const & model,
    Scalar dt,
    Inputs const &... inputs
)
{
    auto const gaussian = toGaussian(state);
    if (!gaussian)
    {
        return std::unexpected{gaussian.error()};
    }
    auto info = toInfo(Emrys::predict(method, *gaussian, model, dt, inputs...));
    if (!info)
    {
        return std::unexpected{info.error()};
    }
    state = std::move(*info);
    return {};
}

/** predict() of the information form \a state into itself over \a model's fixed step, through
 * its moments (see predictInPlace(method, state, model, dt, inputs...)).
 */
template <
    typename Space,
    typename Scalar,
    typename Space_,
    typename Model_,
    typename Method_,
    typename Scalar_,
    typename... Inputs>
[[nodiscard]] std::expected<void, GaussianError>
predictInPlace(
    InfoGaussianParams<Space, Scalar> & state,
    FixedStepModel<Space_, Model_, Method_, Scalar_> const & model,
    Inputs const &... inputs
)
{
    auto const gaussian = toGaussian(state);
    if (!gaussian)
    {
        return std::unexpected{gaussian.error()};
    }
    auto info = toInfo(Emrys::predict(*gaussian, model, inputs...));
    if (!info)
    {
        return std::unexpected{info.error()};
    }
    state = std::move(*info);
    return {};
}

/** Moves the square-root information form \a state forward by the affine step
 * \f$ x' = F x + w \f$ with \f$ w \sim \mathcal N(0, Q) \f$, \f$ Q = G G^T \f$, whose transition is
 * \a transition, noise root \a noiseRoot, and propagated reference \a propagated: the time update
 * of \ref Bierman1977 "Bierman (1977)". With \f$ w = \nu_w \f$ and
 * \f$ R F^{-1} (x' - G w) = r + \nu \f$ the data array over \f$ (w, x') \f$ is triangularized,
 * \f[ Q^T \begin{bmatrix} I & 0 & 0 \\ -R F^{-1} G & R F^{-1} & r \end{bmatrix}
 *     = \begin{bmatrix} R_{ww} & R_{wx} & r_w \\ 0 & R' & r' \end{bmatrix}, \f]
 * and the trailing block is the predicted array, the marginal over \f$ x' \f$ (see
 * SqrtInfoGaussianParams::marginal()). A prior with no information predicts as any
 * other, which is what the form is for; the step's Jacobian must be invertible instead. Bierman
 * whitens the noise by its own factor; here \f$ w \f$ is standard normal and \f$ G \f$ carries the
 * covariance, so a singular \f$ Q \f$ works too.
 *
 * \returns nothing, or GaussianError::Singular if \a transition is singular
 */
template <typename Space, typename Scalar, typename Transition, typename NoiseRoot>
[[nodiscard]] std::expected<void, GaussianError>
propagateRootInfo(
    SqrtInfoGaussianParams<Space, Scalar> & state,
    Transition const & transition,
    NoiseRoot const & noiseRoot,
    typename SqrtInfoGaussianParams<Space, Scalar>::VectorType propagated
)
{
    using MatrixType = typename SqrtInfoGaussianParams<Space, Scalar>::MatrixType;
    auto const lu = Eigen::FullPivLU<MatrixType>{transition};
    if (!lu.isInvertible())
    {
        return std::unexpected{GaussianError::Singular};
    }
    propagateRootInfoByInverse(state, inverseFrom<MatrixType>(lu), noiseRoot, std::move(propagated));
    return {};
}

/** propagateRootInfo() with the transition's inverse \a inverse already in hand, as a
 * FixedStepModel keeps it, so that a fixed step inverts the transition once rather than on every
 * prediction.
 */
template <typename Space, typename Scalar, typename Inverse, typename NoiseRoot>
void
propagateRootInfoByInverse(
    SqrtInfoGaussianParams<Space, Scalar> & state,
    Inverse const & inverse,
    NoiseRoot const & noiseRoot,
    typename SqrtInfoGaussianParams<Space, Scalar>::VectorType propagated
)
{
    using Form = SqrtInfoGaussianParams<Space, Scalar>;
    using MatrixType = typename Form::MatrixType;
    constexpr int N = Form::DifferenceDimensionAtCompileTime;
    constexpr int NMax = Form::MaxDifferenceDimensionAtCompileTime;
    using Stacked = MatrixOf<Scalar, sizeSum<N, N>, sizeSum<N, N, 1>, sizeSum<NMax, NMax>, sizeSum<NMax, NMax, 1>>;
    Eigen::Index const n = state.differenceDimension();

    MatrixType const scaled = multiply(asTriangular<Eigen::Upper>(state.infoRoot()), inverse);
    Stacked stacked = Stacked::Zero(2 * n, 2 * n + 1);
    stacked.topLeftCorner(n, n).setIdentity();
    stacked.block(n, 0, n, n).noalias() = -multiply(scaled, noiseRoot);
    stacked.block(n, n, n, n) = scaled;
    stacked.col(2 * n).tail(n) = state.whitenedOffset();
    triangularize(stacked);
    state = Form{
        state.space(),
        std::move(propagated),
        MatrixType{stacked.block(n, n, n, n)},
        typename Form::DifferenceVectorType{stacked.col(2 * n).tail(n)}
    };
}

/** Moves the square-root information form \a state forward by \a dt through \a model by
 * \a method with the process noise \a noise added (decisions.md [gaussian-forms]): the body of both forms of
 * predictInPlace().
 *
 * Under Linearized the step is linearized at the mean, to which the reference moves, or at the
 * reference while the form has no information: a model with no uncertain inputs that gives its
 * Jacobian gives it there [returned-jacobians], and any other step is differentiated there, the uncertain inputs' terms
 * added to the noise; then propagateRootInfo(). With a linearizedAbout() reference first among \a inputs, the step
 * is linearized about it and carried to that chart (linearizeStepAbout(), [linearization-reference]), which a
 * form with no information takes too. Any other method predicts through the moments, which a form with no
 * information lacks.
 *
 * \returns nothing, GaussianError::Singular if the step's Jacobian is singular or a sigma-point
 *          method meets a prior with no information, or the GaussianError of the conversion back
 */
template <
    typename Method,
    typename Space,
    typename Scalar,
    typename R,
    typename Noise,
    typename Reference = NoReference>
[[nodiscard]] std::expected<void, GaussianError>
predictRootInfoWithNoise(
    Method const & method,
    SqrtInfoGaussianParams<Space, Scalar> & state,
    R const & resolved,
    Scalar dt,
    Eigen::MatrixBase<Noise> const & noise,
    Reference const & about = {}
)
{
    using Form = SqrtInfoGaussianParams<Space, Scalar>;
    using MatrixType = typename Form::MatrixType;
    using VectorType = typename Form::VectorType;
    auto const & space = state.space();
    Eigen::Index const n = state.differenceDimension();
    if constexpr (isLinearMethod<Method>)
    {
        auto const mean = state.mean();
        if (mean)
        {
            state.setReference(*mean);
        }
        else if (mean.error() != GaussianError::Singular)
        {
            return std::unexpected{mean.error()};
        }
        VectorType const & reference = state.reference();
        MatrixType total = symmetricFromLower<MatrixType>(noise);
        if constexpr (isLinearizedAbout<Reference>)
        {
            // The step linearized about the given reference, in the chart of the form's own.
            auto const linearization = linearizeStepAbout(method, resolved, space, dt, reference, about);
            total += linearization.error();
            return propagateRootInfo(
                state,
                linearization.jacobian,
                defaultRoot(total),
                VectorType{linearization.value}
            );
        }
        else if constexpr (predictsByJacobian<Method, R>)
        {
            auto step = stepWithJacobian(resolved, reference, dt);
            normalizeValues(space, step.value);
            return propagateRootInfo(state, step.jacobian, defaultRoot(total), std::move(step.value));
        }
        else
        {
            return std::apply(
                [&](auto const &... inputs) -> std::expected<void, GaussianError> {
                    auto const f = ProcessStep{resolved, dt};
                    using Point = PointInput<Space, Scalar>;
                    using Problem =
                        Transform<decltype(f) const, Space, Point, std::remove_cvref_t<decltype(inputs)>...>;
                    auto const point = Point{space, reference};
                    auto const problem = Problem{f, space, point, inputs...};
                    auto const linearized = linearizeAt(method, problem);
                    MatrixType const transition = linearized.jacobian.middleCols(problem.differenceOffset(0), n);
                    problem.forEachUncertain([&]<std::size_t I>(auto const & in) {
                        if constexpr (I != 0)
                        {
                            using In = std::remove_cvref_t<decltype(in)>;
                            using Cov = MatrixOf<
                                Scalar,
                                In::DifferenceDimensionAtCompileTime,
                                In::DifferenceDimensionAtCompileTime,
                                In::MaxDifferenceDimensionAtCompileTime,
                                In::MaxDifferenceDimensionAtCompileTime>;
                            Eigen::Index const d = in.differenceDimension();
                            auto const block = middleColsOf<In::DifferenceDimensionAtCompileTime>(
                                linearized.jacobian,
                                problem.differenceOffset(I),
                                d
                            );
                            total.noalias() += multiply(
                                multiply(block, symmetricFromLower<Cov>(covarianceLowerOf(in))),
                                block.transpose()
                            );
                        }
                    });
                    return propagateRootInfo(state, transition, defaultRoot(total), VectorType{linearized.mean});
                },
                resolved.inputs()
            );
        }
    }
    else
    {
        auto const gaussian = toGaussian(state);
        if (!gaussian)
        {
            return std::unexpected{gaussian.error()};
        }
        auto result = toSqrtInfo(predictWithNoise(method, *gaussian, resolved, dt, noise, about));
        if (!result)
        {
            return std::unexpected{result.error()};
        }
        state = std::move(*result);
        return {};
    }
}

/** predict() of the square-root information form \a state into itself, by \a method through
 * \a model over \a dt with \a inputs (see predictRootInfoWithNoise()).
 */
template <typename Method, typename Space, typename Scalar, ProcessModel Model, typename... Args>
[[nodiscard]] std::expected<void, GaussianError>
predictInPlace(
    Method const & method,
    SqrtInfoGaussianParams<Space, Scalar> & state,
    Model const & model,
    Scalar dt,
    Args const &... args
)
{
    internal::assertPartitions(model, state.space());
    return withReference(
        [&](auto const & about, auto const &... inputs) {
            auto const call = modelWith(std::cref(model), inputs...);
            auto const resolved = resolveProcess(call, state.space());
            return predictRootInfoWithNoise(method, state, resolved, dt, resolved.template noise<Scalar>(dt), about);
        },
        args...
    );
}

/** predict() of the square-root information form \a state into itself over \a model's fixed
 * step with \a inputs: by the cached transition and noise where the model caches them and there
 * are no inputs, and otherwise by the model's method over its step (see
 * predictRootInfoWithNoise()).
 */
template <
    typename Space,
    typename Scalar,
    typename Space_,
    typename Model_,
    typename Method_,
    typename Scalar_,
    typename... Args>
[[nodiscard]] std::expected<void, GaussianError>
predictInPlace(
    SqrtInfoGaussianParams<Space, Scalar> & state,
    FixedStepModel<Space_, Model_, Method_, Scalar_> const & model,
    Args const &... args
)
{
    using Model = FixedStepModel<Space_, Model_, Method_, Scalar_>;
    static_assert(std::is_same_v<Space, Space_>, "the model must be over the state's space");
    static_assert(std::is_same_v<Scalar, Scalar_>, "the model's scalar must be the state's");
    internal::assertPartitions(model, state.space());
    if constexpr (Model::CachesTransition && sizeof...(Args) == 0)
    {
        using VectorType = typename SqrtInfoGaussianParams<Space, Scalar>::VectorType;
        auto const mean = state.mean();
        if (mean)
        {
            state.setReference(*mean);
        }
        else if (mean.error() != GaussianError::Singular)
        {
            return std::unexpected{mean.error()};
        }
        VectorType propagated = model.propagate(state.space(), state.reference());
        normalizeValues(state.space(), propagated);
        // The fixed step keeps its transition's inverse and its noise's root [fixed-step-models].
        if (!model.transitionInverse())
        {
            return std::unexpected{GaussianError::Singular};
        }
        propagateRootInfoByInverse(state, *(model.transitionInverse()), model.noiseRoot(), std::move(propagated));
        return {};
    }
    else
    {
        return withReference(
            [&](auto const & about, auto const &... inputs) {
                auto const call = modelWith(std::cref(model.model()), inputs...);
                auto const resolved = resolveProcess(call, state.space());
                return predictRootInfoWithNoise(
                    model.method(),
                    state,
                    resolved,
                    model.timeStep(),
                    model.noise(),
                    about
                );
            },
            args...
        );
    }
}

}  // namespace internal

}  // namespace Emrys
