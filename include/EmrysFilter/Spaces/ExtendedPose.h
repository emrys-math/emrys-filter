#pragma once

/** \file
 * The family of groups of a rotation and the vectors it turns (docs/design/groups.md; a prototype, not yet a
 * decision): ExtendedPose, which is \f$ SE(3) \f$ with one vector, \f$ SE_2(3) \f$ with two, and \f$ SE_k(3) \f$ with
 * a Repeated block among them, whose count is known at run time; and the Rotation concept, which says what a space
 * provides to head one. UnitQuaternion heads the groups of space, and a Periodic heading those of the plane,
 * \f$ SE_k(2) \f$. Read Reset.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

#include <Eigen/LU>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Group.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** Whether \a R has a rotation's members of its own: \c ActionDimension, the size of the vectors it turns, and
 * rotationMatrix(q, out), integratedRotation(d, out), and actionJacobian(v, out) (see Rotation).
 */
template <typename R>
concept HasRotationMembers = requires(R const & rotation, Eigen::MatrixXd & m) {
    typename std::integral_constant<int, R::ActionDimension>;
    rotation.rotationMatrix(m.col(0), m.middleRows(0, 0));
    rotation.integratedRotation(m.col(0), m.middleRows(0, 0));
    rotation.actionJacobian(m.col(0), m.middleRows(0, 0));
};

/** A Periodic space of one fixed coordinate, or a tag derived from one: a heading, which turns vectors of the plane.
 * Its period is one turn, whatever its unit.
 */
template <typename R>
concept IsPlanarHeading = IsPeriodic<R> && R::FixedValueDimension == 1 && R::ValueDimensionStep == 0;

/** The size of the vectors that the rotation \a R turns. */
template <typename R>
struct ActionDimensionOf
{
    static constexpr int value = 2;  ///< a heading turns vectors of the plane
};

/// \cond INTERNAL_SPECIALIZATIONS
template <HasRotationMembers R>
struct ActionDimensionOf<R>
{
    static constexpr int value = R::ActionDimension;
};

/// \endcond

/** Radians per unit of the heading \a R: a period is one turn. */
template <typename R>
inline constexpr double radiansPerUnit = 2 * std::numbers::pi / R::period();

/** Writes the matrix by which the value \a q of the rotation \a rotation turns a vector through \a out. */
template <typename R, typename Q, typename Out>
void
rotationMatrixInto(R const & rotation, Eigen::MatrixBase<Q> const & q, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasRotationMembers<R>)
    {
        rotation.rotationMatrix(q, out);
    }
    else
    {
        using T = typename Out::Scalar;
        T const angle = T(q(0)) * radiansPerUnit<R>;
        T const c = Eigen::numext::cos(angle);
        T const s = Eigen::numext::sin(angle);
        writable(out) << c, -s, s, c;
    }
}

/** Writes \f$ V(d) = \int_0^1 R(\mathrm{Exp}(s d)) \, ds \f$ of the rotation \a rotation through \a out: the matrix
 * that turns a vector's step into the vector's change over a step \a d of the rotation.
 */
template <typename R, typename D, typename Out>
void
integratedRotationInto(R const & rotation, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasRotationMembers<R>)
    {
        rotation.integratedRotation(d, out);
    }
    else
    {
        using T = typename Out::Scalar;
        T const angle = T(d(0)) * radiansPerUnit<R>;
        T const angle2 = angle * angle;
        T a = T(1) - angle2 / T(6);
        T b = angle / T(2) - angle * angle2 / T(24);
        if (angle2 >= T(1e-8))
        {
            a = Eigen::numext::sin(angle) / angle;
            b = (T(1) - Eigen::numext::cos(angle)) / angle;
        }
        writable(out) << a, -b, b, a;
    }
}

/** Writes through \a out the derivative of \f$ R(\mathrm{Exp}(\phi)) v \f$ in the rotation's step \f$ \phi \f$ at
 * zero: how a small step of the rotation \a rotation moves the vector \a v.
 */
template <typename R, typename V, typename Out>
void
actionJacobianInto(R const & rotation, Eigen::MatrixBase<V> const & v, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasRotationMembers<R>)
    {
        rotation.actionJacobian(v, out);
    }
    else
    {
        using T = typename Out::Scalar;
        writable(out) << T(-v(1)) * radiansPerUnit<R>, T(v(0)) * radiansPerUnit<R>;
    }
}

/** The nodes of the 12-point Gauss-Legendre rule on [0, 1], by which ExtendedPose integrates its right Jacobian. */
inline constexpr std::array<double, 12> unitGaussLegendreNodes{
    0.0092196828766404,  0.04794137181476255, 0.11504866290284765, 0.20634102285669125,
    0.3160842505009099,  0.43738329574426555, 0.56261670425573445, 0.6839157494990901,
    0.79365897714330875, 0.88495133709715235, 0.95205862818523745, 0.9907803171233596
};

/** The weights of the nodes of unitGaussLegendreNodes. */
inline constexpr std::array<double, 12> unitGaussLegendreWeights{
    0.0235876681932559,  0.0534696629976592, 0.0800391642716731, 0.10158371336153295,
    0.1167462682691774,  0.1245735229067014, 0.1245735229067014, 0.1167462682691774,
    0.10158371336153295, 0.0800391642716731, 0.0534696629976592, 0.0235876681932559
};

}  // namespace internal

/** \ingroup Spaces
 * \brief A group that can head an ExtendedPose: a rotation, which turns vectors.
 *
 * UnitQuaternion is one, turning vectors of three coordinates, and so is a Periodic space of one coordinate, a
 * heading in any unit, turning vectors of the plane. A rotation written from scratch is a fixed-size LieGroup with
 *  - <tt>static constexpr int ActionDimension</tt>, the size \f$ n \f$ of the vectors it turns;
 *  - <tt>rotationMatrix(q, out)</tt>, the \f$ n \f$ by \f$ n \f$ matrix \f$ R(q) \f$ by which the value \a q turns
 *    a vector, a homomorphism: \f$ R(a \cdot b) = R(a) R(b) \f$;
 *  - <tt>integratedRotation(d, out)</tt>, \f$ V(d) = \int_0^1 R(\mathrm{Exp}(s d)) \, ds \f$, with its series near
 *    zero;
 *  - <tt>actionJacobian(v, out)</tt>, the \f$ n \f$ by \f$ k \f$ derivative of \f$ R(\mathrm{Exp}(\phi)) v \f$ in
 *    the rotation's step \f$ \phi \f$ of \f$ k \f$ coordinates, at zero,
 *
 * each const, in any scalar, writing through \a out as plus() does.
 */
template <typename R>
concept Rotation = LieGroup<R> && R::ValueDimensionStep == 0
                && (internal::HasRotationMembers<R> || internal::IsPlanarHeading<R>);

/** \ingroup Spaces
 * \brief A rotation followed by vectors that turn with it, as one Lie group: \f$ SE(3) \f$ with one vector,
 * \f$ SE_2(3) \f$ with two, and \f$ SE_k(3) \f$ with any number (docs/design/groups.md).
 *
 * The blocks are the user's tags, as in a Composite, and a Repeated may be among the vectors, so that the group's
 * size is known only at run time:
 * \code
 * struct Attitude : UnitQuaternion {};
 * struct Velocity : Linear<3> {};
 * struct Position : Linear<3> {};
 * struct Foot : Linear<3> {};
 * struct Feet : Repeated<Foot> { using Repeated::Repeated; };
 * struct Body : ExtendedPose<Attitude, Velocity, Position, Feet> { using ExtendedPose::ExtendedPose; };
 * \endcode
 * It holds the values of <tt>Composite<Attitude, Velocity, Position, Feet></tt>, in that order, and names them by
 * the same paths. It differs in its product, a rigid motion after another,
 * \f[ (R_a, v_a) \cdot (R_b, v_b) = (R_a R_b, \; v_a + R_a v_b), \f]
 * applied to every vector alike, and so in its \f$ \boxplus \f$ and \f$ \boxminus \f$: with \f$ V \f$ the rotation's
 * integratedRotation(), which for a UnitQuaternion is the left Jacobian of SO(3),
 * \f[ (R, v_i) \boxplus (\phi, \rho_i) = (R \, \mathrm{Exp}(\phi), \; v_i + R \, V(\phi) \, \rho_i). \f]
 * A vector's difference rows are then a step of that vector in the body frame at the value: to first order
 * \f$ v_i + R \rho_i \f$. The extended Kalman filter on this space is the invariant extended Kalman filter of
 * \ref Barrau2017 "Barrau and Bonnabel (2017)", with the error they call left-invariant.
 *
 * The rotation \a R is a Rotation, and each of \a Vectors a linear block that holds whole vectors of the size the
 * rotation turns: <tt>Linear<3></tt> or a Repeated of it under a UnitQuaternion, <tt>Linear<2></tt> under a heading.
 * The reset Jacobian is the group's right Jacobian, integrated by a 12-point Gauss-Legendre rule over the adjoint
 * (a closed form is still to write).
 */
template <Rotation R, Space... Vectors>
class ExtendedPose : public Composite<R, Vectors...>
{
    /** The Composite that holds the blocks. */
    using Base = Composite<R, Vectors...>;

    static constexpr int N = internal::ActionDimensionOf<R>::value;                 ///< the vectors' size
    static constexpr int Rv = R::FixedValueDimension;                               ///< the rotation's values
    static constexpr int Rd = internal::DifferenceSpaceOf<R>::type::FixedValueDimension;  ///< and differences

    static_assert((isLinear<Vectors> && ...), "the blocks after the rotation are vectors: linear blocks");
    static_assert(
        ((Vectors::FixedValueDimension % N == 0 && Vectors::ValueDimensionStep % N == 0) && ...),
        "each block after the rotation holds whole vectors of the size the rotation turns"
    );

    /** A square matrix of the vectors' size, in \a T. */
    template <typename T>
    using Turn = Eigen::Matrix<T, N, N>;

public:

    using Base::Base;

    ExtendedPose() = default;

    /** \returns the rotation's block */
    [[nodiscard]] R const &
    rotation() const
    {
        return this->template block<0>();
    }

    /** \returns the number of vectors, the repetitions of a Repeated block counted each */
    [[nodiscard]] Eigen::Index
    vectorCount() const
    {
        return (this->valueDimension() - Rv) / N;
    }

    /** Writes \f$ x \boxplus d = x \cdot \mathrm{Exp}(d) \f$ through \a out, column by column. */
    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        Eigen::Index const m = vectorCount();
        for (Eigen::Index j = 0; j < o.cols(); ++j)
        {
            Eigen::Matrix<T, Rv, 1> const q = x.col(j).template head<Rv>().template cast<T>();
            Eigen::Matrix<T, Rd, 1> const phi = d.col(j).template head<Rd>().template cast<T>();
            Eigen::Matrix<T, Rv, 1> moved;
            internal::plusInto(rotation(), q, phi, moved);
            o.col(j).template head<Rv>() = moved;
            Turn<T> turn;
            Turn<T> integrated;
            internal::rotationMatrixInto(rotation(), q, turn);
            internal::integratedRotationInto(rotation(), phi, integrated);
            Turn<T> const step = turn * integrated;
            for (Eigen::Index i = 0; i < m; ++i)
            {
                o.col(j).template segment<N>(Rv + i * N) =
                    x.col(j).template segment<N>(Rv + i * N).template cast<T>()
                    + step * d.col(j).template segment<N>(Rd + i * N).template cast<T>();
            }
        }
    }

    /** Writes \f$ a \boxminus b = \mathrm{Log}(b^{-1} \cdot a) \f$ through \a out, column by column. */
    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        Eigen::Index const m = vectorCount();
        for (Eigen::Index j = 0; j < o.cols(); ++j)
        {
            Eigen::Matrix<T, Rv, 1> const qa = a.col(j).template head<Rv>().template cast<T>();
            Eigen::Matrix<T, Rv, 1> const qb = b.col(j).template head<Rv>().template cast<T>();
            Eigen::Matrix<T, Rd, 1> phi;
            internal::minusInto(rotation(), qa, qb, phi);
            o.col(j).template head<Rd>() = phi;
            Turn<T> turn;
            Turn<T> integrated;
            internal::rotationMatrixInto(rotation(), qb, turn);
            internal::integratedRotationInto(rotation(), phi, integrated);
            Turn<T> const back = integrated.inverse() * turn.transpose();
            for (Eigen::Index i = 0; i < m; ++i)
            {
                o.col(j).template segment<N>(Rd + i * N) =
                    back
                    * (a.col(j).template segment<N>(Rv + i * N).template cast<T>()
                       - b.col(j).template segment<N>(Rv + i * N).template cast<T>());
            }
        }
    }

    /** Writes the product \f$ a \cdot b \f$ through \a out: the rotations' product, and each vector of \a b turned by
     * \a a's rotation and added to \a a's.
     */
    template <typename A, typename B, typename Out>
    void
    compose(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        Eigen::Matrix<T, Rv, 1> const qa = a.template head<Rv>().template cast<T>();
        Eigen::Matrix<T, Rv, 1> const qb = b.template head<Rv>().template cast<T>();
        Eigen::Matrix<T, Rv, 1> q;
        internal::composeInto(rotation(), qa, qb, q);
        o.template head<Rv>() = q;
        Turn<T> turn;
        internal::rotationMatrixInto(rotation(), qa, turn);
        for (Eigen::Index i = 0; i < vectorCount(); ++i)
        {
            o.template segment<N>(Rv + i * N) = a.template segment<N>(Rv + i * N).template cast<T>()
                                              + turn * b.template segment<N>(Rv + i * N).template cast<T>();
        }
    }

    /** Writes the inverse of \a a through \a out: the rotation's inverse, and each vector turned back and negated. */
    template <typename A, typename Out>
    void
    inverse(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        Eigen::Matrix<T, Rv, 1> const qa = a.template head<Rv>().template cast<T>();
        Eigen::Matrix<T, Rv, 1> q;
        internal::inverseInto(rotation(), qa, q);
        o.template head<Rv>() = q;
        Turn<T> turn;
        internal::rotationMatrixInto(rotation(), qa, turn);
        for (Eigen::Index i = 0; i < vectorCount(); ++i)
        {
            o.template segment<N>(Rv + i * N) =
                -(turn.transpose() * a.template segment<N>(Rv + i * N).template cast<T>());
        }
    }

    /** Writes the identity through \a out: the rotation's identity and zero vectors. */
    template <typename Out>
    void
    identity(Eigen::MatrixBase<Out> const & out) const
    {
        auto & o = writable(out);
        o.setZero();
        internal::identityInto(rotation(), o.template head<Rv>());
    }

    /** Writes \f$ \mathrm{Exp}(d) \f$ through \a out: the rotation's exponential, and each vector's step turned by
     * the rotation's integratedRotation().
     */
    template <typename D, typename Out>
    void
    exp(Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        Eigen::Matrix<T, Rd, 1> const phi = d.template head<Rd>().template cast<T>();
        Eigen::Matrix<T, Rv, 1> q;
        internal::expInto(rotation(), phi, q);
        o.template head<Rv>() = q;
        Turn<T> integrated;
        internal::integratedRotationInto(rotation(), phi, integrated);
        for (Eigen::Index i = 0; i < vectorCount(); ++i)
        {
            o.template segment<N>(Rv + i * N) = integrated * d.template segment<N>(Rd + i * N).template cast<T>();
        }
    }

    /** Writes \f$ \mathrm{Log}(a) \f$ through \a out, the inverse of exp(). */
    template <typename A, typename Out>
    void
    log(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        Eigen::Matrix<T, Rv, 1> const q = a.template head<Rv>().template cast<T>();
        Eigen::Matrix<T, Rd, 1> phi;
        internal::logInto(rotation(), q, phi);
        o.template head<Rd>() = phi;
        Turn<T> integrated;
        internal::integratedRotationInto(rotation(), phi, integrated);
        Turn<T> const back = integrated.inverse();
        for (Eigen::Index i = 0; i < vectorCount(); ++i)
        {
            o.template segment<N>(Rd + i * N) = back * a.template segment<N>(Rv + i * N).template cast<T>();
        }
    }

    /** Writes the adjoint \f$ \mathrm{Ad}_a \f$ through \a out, a square matrix over the differences: the rotation's
     * adjoint in its own rows; the rotation matrix in each vector's; and below the rotation's columns, for the vector
     * \f$ v_i \f$, \f$ -G(v_i) \, \mathrm{Ad}_R \f$ with \f$ G \f$ the rotation's actionJacobian(), which for a
     * UnitQuaternion is \f$ [v_i]_\times R \f$.
     */
    template <typename A, typename Out>
    void
    adjoint(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        o.setZero();
        Eigen::Matrix<T, Rv, 1> const q = a.template head<Rv>().template cast<T>();
        Eigen::Matrix<T, Rd, Rd> own;
        internal::adjointInto(rotation(), q, own);
        o.template topLeftCorner<Rd, Rd>() = own;
        Turn<T> turn;
        internal::rotationMatrixInto(rotation(), q, turn);
        for (Eigen::Index i = 0; i < vectorCount(); ++i)
        {
            Eigen::Matrix<T, N, 1> const v = a.template segment<N>(Rv + i * N).template cast<T>();
            Eigen::Matrix<T, N, Rd> action;
            internal::actionJacobianInto(rotation(), v, action);
            o.template block<N, Rd>(Rd + i * N, 0) = -(action * own);
            o.template block<N, N>(Rd + i * N, Rd + i * N) = turn;
        }
    }

    /** Writes the right Jacobian \f$ J_r(d) = \int_0^1 \mathrm{Ad}_{\mathrm{Exp}(-s d)} \, ds \f$ through \a out, by
     * a 12-point Gauss-Legendre rule, which is exact to rounding for steps of a turn or less.
     */
    template <typename D, typename Out>
    void
    rightJacobian(Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        using Value = Eigen::Matrix<T, valueDimensionAtCompileTime<Base>, 1, 0, maxValueDimensionAtCompileTime<Base>, 1>;
        using Difference = Eigen::Matrix<
            T,
            valueDimensionAtCompileTime<DifferenceSpace<Base>>,
            1,
            0,
            maxValueDimensionAtCompileTime<DifferenceSpace<Base>>,
            1>;
        using Matrix = internal::MatrixOf<
            T,
            valueDimensionAtCompileTime<DifferenceSpace<Base>>,
            valueDimensionAtCompileTime<DifferenceSpace<Base>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<Base>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<Base>>>;
        auto & o = writable(out);
        Eigen::Index const n = d.size();
        o.setZero();
        auto value = Value{};
        value.resize(this->valueDimension());
        auto scaled = Difference{};
        scaled.resize(n);
        auto term = Matrix{};
        term.resize(n, n);
        for (std::size_t k = 0; k < internal::unitGaussLegendreNodes.size(); ++k)
        {
            scaled = -internal::unitGaussLegendreNodes[k] * d.template cast<T>();
            exp(scaled, value);
            adjoint(value, term);
            o += internal::unitGaussLegendreWeights[k] * term;
        }
    }

    /** Writes the reset Jacobian at the step \a d through \a out: rightJacobian() of \a d, whatever the value
     * (decisions.md [covariance-reset]).
     */
    template <typename X, typename D, typename Out>
    void
    resetJacobian(
        Eigen::MatrixBase<X> const &,
        Eigen::MatrixBase<D> const & d,
        Eigen::MatrixBase<Out> const & out
    ) const
    {
        rightJacobian(d, out);
    }

    /** Equal when every stored parameter and size is. */
    friend bool
    operator==(ExtendedPose const &, ExtendedPose const &) = default;
};

namespace internal
{

/// \cond INTERNAL_SPECIALIZATIONS
template <typename R, typename... Vectors>
inline constexpr bool IsUntagged<ExtendedPose<R, Vectors...>>{true};
/// \endcond

/** Declared, never defined: matched by a pointer to an ExtendedPose or a tag derived from one. */
template <typename R, typename... Vectors>
std::tuple<R, Vectors...> *
extendedPoseBlocks(ExtendedPose<R, Vectors...> const *);

/** An ExtendedPose, or a tag derived from one. */
template <typename T>
concept IsExtendedPose = requires(T const * t) { extendedPoseBlocks(t); };

}  // namespace internal

}  // namespace Emrys
