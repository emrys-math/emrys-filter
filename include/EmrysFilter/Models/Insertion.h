#pragma once

/** \file
 * Growing a state by what a model computes from it (docs/design/groups.md, open question 3; a prototype):
 * insert<Path>(state, element, method, model, inputs...), which inserts into a resizable block of a %Gaussian state
 * the output of a measurement model at the state, with the covariance and the cross-covariance that the method
 * gives it. A new landmark seen from a pose, or a foot placed through a leg's kinematics, is correlated with the
 * pose it was computed from, which GaussianParams::insert() given a mean and a covariance alone cannot say.
 */

#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/MeasurementReads.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/ExtendedPose.h>
#include <EmrysFilter/Spaces/Subspace.h>

namespace Emrys
{

/** \ingroup Models
 * Inserts into the resizable block of \a state named by the tag path \a Path, before its element \a element, the
 * output of the measurement model \a model at the state: its mean, its covariance with the model's additive noise,
 * and its cross-covariance with the existing entries, each as \a method computes it, linearized or by sigma points.
 * The model's output is one or more whole elements of the block, in the block's own space, and \a inputs are the
 * inputs of the call that the model reads, as observe() takes them.
 *
 * Where the block is among the vectors of an ExtendedPose, the model computes the new vector in the world frame, as
 * it would a measurement, and the new rows of the covariance are turned to steps in the pose's body frame by the
 * rotation at the state's mean, as every vector's rows of that pose are.
 *
 * \code
 * // a foot at the end of a leg whose kinematics give its place in the body frame, with their noise
 * auto const placed = measurementModel(
 *     [](auto const & in, auto const & x) {
 *         return (block<Position>(in, x)
 *                 + UnitQuaternion::map(block<Attitude>(in, x)) * block<Leg>(in, x)).eval();
 *     },
 *     reads(path<Attitude>, path<Position>, path<Leg>), outputSpace(Foot{}));
 * insert<path<Body, Feet>>(filter.state(), 0, methods::linearized(), placed, GaussianParams<Leg>{leg, noise});
 * \endcode
 */
template <
    typename Path,
    typename Space_,
    typename Scalar,
    TransformMethod Method,
    MeasurementModel Model,
    typename... Inputs>
void
insert(
    GaussianParams<Space_, Scalar> & state,
    Eigen::Index element,
    Method const & method,
    Model const & model,
    Inputs const &... inputs
)
{
    using Call = Observed<Method, Model const &, internal::NoObservation, Inputs...>;
    auto const call = Call{method, model, internal::NoObservation{}, inputs...};
    auto const & space = state.space();
    auto const resolved = internal::resolveMeasurement(call, space);
    using R = std::remove_cvref_t<decltype(resolved)>;
    using Reads = typename R::Reads;
    constexpr int N = valueDimensionAtCompileTime<DifferenceSpace<Space_>>;
    constexpr int NMax = maxValueDimensionAtCompileTime<DifferenceSpace<Space_>>;
    auto const & reads = resolved.reads();
    auto const read =
        GaussianMarginal<GaussianParams<Space_, Scalar>, typename Reads::Selection, typename Reads::JointSpace>{
            state,
            reads.selection(),
            reads.space()
        };
    auto const f = internal::ResolvedFunction{resolved};
    auto const predicted = internal::transformMeasurement(resolved, method, f, read);

    // The new entries' covariance with the whole state: P_xS H^T, with H the output's Jacobian in the blocks read.
    using Dynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    auto const all = internal::consecutiveIndices<N, NMax>(0, state.differenceDimension());
    Dynamic const stateRead = internal::lowerSymmetricBlock(covarianceLowerOf(state), all, reads.differenceIndices());
    Dynamic const jacobian = predicted.jacobian();
    Dynamic cross = (stateRead * jacobian.transpose()).transpose();
    auto const & output = predicted.output();
    Dynamic cov = output.covarianceLower().template selfadjointView<Eigen::Lower>();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mean = output.mean();
    if constexpr (R::HasAdditiveNoise)
    {
        auto const & noise = resolved.noise().distribution();
        mean = Emrys::plus(output.space(), mean, noise.mean());
        cov += Dynamic{noise.covarianceLower().template selfadjointView<Eigen::Lower>()};
    }

    if constexpr (internal::posePartOf<Space_, Path> == internal::PosePart::vectors)
    {
        using Location = typename internal::PoseLocationOf<Space_, Path>::type;
        Eigen::Index const start = [&] {
            if constexpr (std::is_same_v<typename Location::UnitPath, internal::WholeSpace>)
            {
                return Eigen::Index{0};
            }
            else
            {
                return internal::blockRows<typename Location::UnitPath>(space).start;
            }
        }();
        auto const & pose = [&]() -> auto const & {
            if constexpr (std::is_same_v<typename Location::UnitPath, internal::WholeSpace>)
            {
                return space;
            }
            else
            {
                return internal::pathBlock<typename Location::UnitPath>(space);
            }
        }();
        using Pose = std::remove_cvref_t<decltype(pose)>;
        auto const rotation = state.mean().template middleRows<Pose::RotationValueDimension>(start);
        internal::turnRowsBack(pose, rotation, cross);
        internal::turnRowsBack(pose, rotation, cov);
        internal::turnColumns(pose, rotation, cov);
    }
    state.template insert<Path>(element, mean, cov, cross);
}

}  // namespace Emrys
