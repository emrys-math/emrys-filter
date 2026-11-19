#pragma once

/** \file
 * Models on a Lie group whose Jacobians are formed, not differentiated (docs/design/groups.md, "Jacobians that do
 * not depend on the estimate" and open question 8; a prototype): bodyVelocity(), the motion of a group by a velocity
 * in its body frame, and observedInWorld() and observedInBody(), the two measurements of an ExtendedPose that the
 * invariant extended Kalman filters are built on. Each returns its Jacobian in closed form [returned-jacobians], and
 * each measurement names the carry that suits it, so that a measurement by one of them needs neither a method nor a
 * carry: <tt>observe(sight, z)</tt>.
 */

#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/ExtendedPose.h>
#include <EmrysFilter/Spaces/Group.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** The step of bodyVelocity(): the group block \a Group moved by the velocity \a Velocity over the step. */
template <typename Group, typename Velocity>
struct BodyVelocityStep
{
    /** \returns \f$ X \cdot \mathrm{Exp}(u \, \Delta t) \f$ with its Jacobian over the differences of \a in:
     * \f$ \mathrm{Ad}_{\mathrm{Exp}(-u \Delta t)} \f$ in the group's columns and \f$ \Delta t \, J_r(u \Delta t) \f$
     * in the velocity's, neither of which depends on \f$ X \f$
     */
    template <typename In, typename Out, typename X, typename TimeScalar>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, TimeScalar dt) const
    {
        using T = ScalarOf<X>;
        auto const & group = pathBlock<Group>(in);
        using G = std::remove_cvref_t<decltype(group)>;
        ValueVector<G, T> const value = block<Group>(in, x);
        DifferenceVector<G, T> const step = T(dt) * block<Velocity>(in, x);
        DifferenceVector<G, T> const back = -step;
        using JacobianMatrix = MatrixOf<
            T,
            valueDimensionAtCompileTime<DifferenceSpace<Out>>,
            valueDimensionAtCompileTime<DifferenceSpace<In>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;
        JacobianMatrix jacobian = JacobianMatrix::Zero(differenceDimension(out), differenceDimension(in));
        diffBlock<Group, Group>(out, in, jacobian) = Emrys::adjoint(group, ValueVector<G, T>{Emrys::exp(group, back)});
        diffBlock<Group, Velocity>(out, in, jacobian) = T(dt) * Emrys::rightJacobian(group, step);
        auto y = zeros<T>(out);
        block<Group>(out, y) = Emrys::plus(group, value, step);
        return withJacobian(std::move(y), std::move(jacobian));
    }
};

/** The function of observedInWorld() and observedInBody(): the pose at \a PosePath acting on the point whose part
 * that only turns is \a point and whose weights on the pose's vectors are \a weights, or when \a Inverse the
 * pose's inverse acting on it.
 */
template <typename PosePath, bool Inverse, typename Point, typename... Weights>
struct PoseAction
{
    /** The carry that suits the measurement, which a measurement that names none takes. */
    using DefaultCarry = std::conditional_t<Inverse, carries::Adjoint, carries::None>;

    /** The method of a measurement that names none: the Jacobian is exact and formed here. */
    using DefaultMethod = methods::Linearized;

    Point point;                     ///< the part of the point that only turns, \f$ b_0 \f$
    std::tuple<Weights...> weights;  ///< the weight of each vector named, <tt>on<Position>(1.0)</tt>

    /** \returns \f$ R b_0 + \sum c_i v_i \f$, or when \a Inverse \f$ R^T (b_0 - \sum c_i v_i) \f$, with its
     * Jacobian over the pose's differences: \f$ R \, [G(b_0), \; c_i I] \f$, or
     * \f$ [-G(y), \; -c_i I] \f$ with \f$ y \f$ the value returned, where \f$ G \f$ is the rotation's
     * actionJacobian(), \f$ -[\cdot]_\times \f$ for a UnitQuaternion
     */
    template <typename In, typename X>
    [[nodiscard]] auto
    operator()(In const & in, X const & x) const
    {
        using T = ScalarOf<X>;
        auto const & pose = pathBlock<PosePath>(in);
        using Pose = std::remove_cvref_t<decltype(pose)>;
        static_assert(IsExtendedPose<Pose>, "observedInWorld() and observedInBody() take the path of an ExtendedPose");
        constexpr int N = Pose::VectorDimension;
        constexpr int Rd = Pose::RotationDifferenceDimension;
        using Vector = Eigen::Matrix<T, N, 1>;
        using Turn = Eigen::Matrix<T, N, N>;
        using JacobianMatrix = MatrixOf<
            T,
            N,
            valueDimensionAtCompileTime<DifferenceSpace<Pose>>,
            N,
            maxValueDimensionAtCompileTime<DifferenceSpace<Pose>>>;
        ValueVector<Pose, T> const value = block<PosePath>(in, x);
        Turn turn;
        pose.turn(value, turn);
        Vector sum = Vector::Zero();
        JacobianMatrix jacobian = JacobianMatrix::Zero(N, differenceDimension(pose));
        std::apply(
            [&](auto const &... weight) {
                (addWeighted(pose, value, turn, weight, sum, jacobian), ...);
            },
            weights
        );
        Vector const fixed = point.template cast<T>();
        Eigen::Matrix<T, N, Rd> action;
        Vector y;
        if constexpr (Inverse)
        {
            y = turn.transpose() * (fixed - sum);
            actionJacobianInto(pose.rotation(), y, action);
            jacobian.template leftCols<Rd>() = -action;
        }
        else
        {
            y = turn * fixed + sum;
            actionJacobianInto(pose.rotation(), fixed, action);
            jacobian.template leftCols<Rd>() = turn * action;
        }
        return withJacobian(std::move(y), std::move(jacobian));
    }

private:

    /** Adds the vector that \a weight names, times its weight, to \a sum, and its columns to \a jacobian. */
    template <typename Pose, typename Value, typename Turn, typename Scalar, typename Vector, typename Sum, typename J>
    static void
    addWeighted(
        Pose const & pose,
        Value const & value,
        Turn const & turn,
        OnPath<Scalar, Vector> const & weight,
        Sum & sum,
        J & jacobian
    )
    {
        using T = typename Sum::Scalar;
        constexpr int N = Pose::VectorDimension;
        sum += T(weight.value) * block<Vector>(pose, value);
        Eigen::Index const start = differenceBlockRows<Vector>(pose).start;
        if constexpr (Inverse)
        {
            jacobian.template middleCols<N>(start) = -T(weight.value) * Eigen::Matrix<T, N, N>::Identity();
        }
        else
        {
            jacobian.template middleCols<N>(start) = T(weight.value) * turn;
        }
    }
};

}  // namespace internal

/** \ingroup Models
 * \returns the process model that moves the Lie group block at the tag path \a Group by the velocity it reads under
 * the tag \a Velocity, a block of the group's differences given in its body frame, over each step:
 * \f[ X' = X \cdot \mathrm{Exp}(u \, \Delta t). \f]
 * A UnitQuaternion turned by a gyroscope's rate is one, and a pose moved by a twist. The velocity is an input of
 * the call, <tt>sure<Twist>(u)</tt> or a %Gaussian over it, or a block of the state. \a policies are the model's
 * others, such as its noise.
 *
 * The model returns its Jacobian, \f$ \mathrm{Ad}_{\mathrm{Exp}(-u \Delta t)} \f$ in the group and
 * \f$ \Delta t \, J_r(u \Delta t) \f$ in the velocity, so Linearized never differentiates it, and neither depends
 * on the estimate: the dynamics are group-affine in the sense of \ref Barrau2017 "Barrau and Bonnabel (2017)".
 */
template <typename Group, typename Velocity, typename... Policies>
[[nodiscard]] auto
bodyVelocity(Policies &&... policies)
{
    return processModel(
        internal::BodyVelocityStep<Group, Velocity>{},
        reads(path<Group>, path<Velocity>),
        writes(path<Group>),
        std::forward<Policies>(policies)...
    );
}

/** \ingroup Models
 * \returns the measurement model of a point fixed in the body of the ExtendedPose at the tag path \a Pose, observed
 * in the world: \f$ y = X \cdot b = R \, b_0 + \sum_i c_i v_i \f$, where \a point is \f$ b_0 \f$, a vector in the
 * body frame, and each of \a weights names one of the pose's vectors and its weight \f$ c_i \f$,
 * <tt>on<Position>(1.0)</tt>. A position fix through an antenna's lever arm is
 * <tt>observedInWorld<Body>(arm, std::tuple{on<Position>(1.0)}, additiveNoise(R))</tt>, and a velocity fix names the
 * velocity with a zero point. \a policies are the model's others: its noise, and its output space if not plain.
 *
 * The model returns its Jacobian, \f$ R \, [G(b_0), \; c_i I] \f$ over the pose's body-frame differences, and a
 * measurement by it that names no method is linearized, and one that names no carry takes carries::None: the update
 * of the left-invariant extended Kalman filter of \ref Barrau2017 "Barrau and Bonnabel (2017)".
 */
template <typename Pose, typename Point, typename... Weights, typename... Policies>
[[nodiscard]] auto
observedInWorld(Point const & point, std::tuple<Weights...> weights, Policies &&... policies)
{
    using Held = typename internal::CrtpDerived<Point>::PlainObject;
    return measurementModel(
        internal::PoseAction<Pose, false, Held, Weights...>{point, std::move(weights)},
        reads(path<Pose>),
        std::forward<Policies>(policies)...
    );
}

/** \ingroup Models
 * \returns the measurement model of a point fixed in the world, observed from the body of the ExtendedPose at the
 * tag path \a Pose: \f$ y = X^{-1} \cdot b = R^T (b_0 - \sum_i c_i v_i) \f$, where \a point is \f$ b_0 \f$, a vector
 * in the world frame, and each of \a weights names one of the pose's vectors and its weight \f$ c_i \f$. A landmark
 * seen from the body is <tt>observedInBody<Body>(landmark, std::tuple{on<Position>(1.0)}, additiveNoise(R))</tt>,
 * and a magnetometer or a gravity direction names no vector.
 *
 * The model returns its Jacobian, \f$ [-G(\hat y), \; -c_i I] \f$ over the pose's body-frame differences, which
 * depends on the estimate only through the predicted measurement. A measurement by it that names no method is
 * linearized, and one that names no carry takes carries::Adjoint: the update of the right-invariant extended Kalman
 * filter of \ref Barrau2017 "Barrau and Bonnabel (2017)".
 */
template <typename Pose, typename Point, typename... Weights, typename... Policies>
[[nodiscard]] auto
observedInBody(Point const & point, std::tuple<Weights...> weights, Policies &&... policies)
{
    using Held = typename internal::CrtpDerived<Point>::PlainObject;
    return measurementModel(
        internal::PoseAction<Pose, true, Held, Weights...>{point, std::move(weights)},
        reads(path<Pose>),
        std::forward<Policies>(policies)...
    );
}

}  // namespace Emrys
