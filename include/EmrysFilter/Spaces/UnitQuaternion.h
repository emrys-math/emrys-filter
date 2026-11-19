#pragma once

/** \file
 * The unit quaternion space (decisions.md [unit-quaternion]): UnitQuaternion, an attitude stored in
 * <tt>Eigen::Quaternion</tt>'s own coefficient order with body-frame rotation vectors as its
 * differences, its exponential exp() and logarithm log(), the closed-form reset Jacobian of [covariance-reset], the
 * members that make it a LieGroup (Group.h) and a rotation that can head an ExtendedPose (ExtendedPose.h), and map(),
 * which views an attitude block, such as block<Attitude>(space, x) of a state (TagPath.h), as an
 * <tt>Eigen::Quaternion</tt> without copying it.
 *
 * The four coordinates are (x, y, z, w), the order of <tt>Eigen::Quaternion::coeffs()</tt> and of
 * an <tt>Eigen::Map</tt> over one, not the (w, x, y, z) of Eigen's four-scalar constructor
 * <tt>Quaternion(w, x, y, z)</tt>. Build a value with UnitQuaternion::identity(),
 * UnitQuaternion::coefficientsOf(q), or by writing through map().
 * Reset.h describes the covariance reset this space supplies in closed form. Read Indexing.h next.
 */

#include <concepts>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Spaces
 * \brief An attitude: a [unit quaternion](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
 * of four coordinates whose differences are three-coordinate rotation vectors in the body frame.
 *
 * Values are stored as (x, y, z, w), the order of <tt>Eigen::Quaternion::coeffs()</tt>, so a
 * state's attitude block is exactly what an <tt>Eigen::Quaternion</tt> stores and map() views it
 * as one; the product is Eigen's, Hamilton's. normalize() scales a value to unit norm with
 * \f$ w \ge 0 \f$, choosing one of the two quaternions of each rotation. Differences are rotation
 * vectors in Linear<3>, applied on the right, in the body frame:
 * \f[ q \boxplus \delta = q \otimes \mathrm{Exp}(\delta), \qquad
 *     q \boxminus p = \mathrm{Log}(p^{-1} \otimes q), \f]
 * the right \f$ \oplus \f$ and \f$ \ominus \f$ of \ref Sola2018 "Solà, Deray, and Atchuthan (2018)"
 * on SO(3), the group's \f$ \boxplus \f$ and \f$ \boxminus \f$ in the sense of
 * \ref Hertzberg2013 "Hertzberg, Wagner, Frese, and Schröder (2013)". The logarithm takes the short
 * way round, so a difference has norm at most \f$ \pi \f$. The reset Jacobian is the right Jacobian
 * of SO(3) in closed form (rightJacobian()), so a covariance is carried to a moved mean without
 * automatic differentiation (decisions.md [covariance-reset]).
 *
 * Name the block with a tag, <tt>struct Attitude : UnitQuaternion {};</tt>, and view it in model
 * code with <tt>UnitQuaternion::map(block<Attitude>(space, x))</tt>. Use Repeated for several.
 */
struct UnitQuaternion
{
    static constexpr int FixedValueDimension = 4;  ///< four coordinates, (x, y, z, w)
    static constexpr int ValueDimensionStep = 0;   ///< not resizable
    static constexpr int MaxValueDimension = 4;    ///< the value dimension, as for any fixed space

    /** \returns 4, the number of coordinates of a value */
    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 4;
    }

    /** \returns the difference space: rotation vectors, three plain coordinates */
    [[nodiscard]] Linear<3>
    difference() const
    {
        return {};
    }

    /** Scales each column of \a block, a value (x, y, z, w) per column, to unit norm with
     * \f$ w \ge 0 \f$, writing through the const reference as writable() does.
     */
    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block) const
    {
        using T = typename Derived::Scalar;
        auto & q = writable(block);
        for (Eigen::Index j = 0; j < q.cols(); ++j)
        {
            T const sign = q(3, j) < T(0) ? T(-1) : T(1);
            q.col(j) *= sign / q.col(j).norm();
        }
    }

    /** Writes \f$ x \boxplus d = x \otimes \mathrm{Exp}(d) \f$ through \a out, column by column,
     * normalized.
     */
    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        for (Eigen::Index j = 0; j < o.cols(); ++j)
        {
            o.col(j) = (quaternionOf<T>(x.col(j)) * exp(d.col(j).template cast<T>())).coeffs();
        }
        normalize(o);
    }

    /** Writes \f$ a \boxminus b = \mathrm{Log}(b^{-1} \otimes a) \f$ through \a out, column by
     * column.
     */
    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        for (Eigen::Index j = 0; j < o.cols(); ++j)
        {
            o.col(j) = log(quaternionOf<T>(b.col(j)).conjugate() * quaternionOf<T>(a.col(j)));
        }
    }

    /** Writes the reset Jacobian at the step \a d through \a out: rightJacobian() of \a d, whatever
     * the value. See resetJacobian(space, x, d) for what it is and its sources.
     */
    template <typename X, typename D, typename Out>
    void
    resetJacobian(
        Eigen::MatrixBase<X> const &,
        Eigen::MatrixBase<D> const & d,
        Eigen::MatrixBase<Out> const & out
    ) const
    {
        using T = typename Out::Scalar;
        writable(out) = rightJacobian(d.template cast<T>());
    }

    /** Writes the product \f$ a \otimes b \f$ through \a out, normalized: the group's compose() (Group.h). */
    template <typename A, typename B, typename Out>
    void
    compose(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        o = (quaternionOf<T>(a) * quaternionOf<T>(b)).coeffs();
        normalize(o);
    }

    /** Writes the inverse of \a a, its conjugate, through \a out. */
    template <typename A, typename Out>
    void
    inverse(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        writable(out) = quaternionOf<T>(a).conjugate().coeffs();
    }

    /** Writes the identity rotation, (0, 0, 0, 1), through \a out. */
    template <typename Out>
    void
    identity(Eigen::MatrixBase<Out> const & out) const
    {
        writable(out) = identity<typename Out::Scalar>();
    }

    /** Writes the coordinates of \f$ \mathrm{Exp}(d) \f$ through \a out: exp(d) as the group's member. */
    template <typename D, typename Out>
    void
    exp(Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        writable(out) = exp(d.template cast<T>()).coeffs();
    }

    /** Writes the rotation vector \f$ \mathrm{Log}(a) \f$ through \a out: log(q) as the group's member. */
    template <typename A, typename Out>
    void
    log(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        writable(out) = log(quaternionOf<T>(a));
    }

    /** Writes the adjoint of \a a through \a out: its rotation matrix, which takes a rotation vector in the body
     * frame to the same rotation in the world frame.
     */
    template <typename A, typename Out>
    void
    adjoint(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out) const
    {
        rotationMatrix(a, out);
    }

    /** Writes rightJacobian() of \a d through \a out: the group's member. */
    template <typename D, typename Out>
    void
    rightJacobian(Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        writable(out) = rightJacobian(d.template cast<T>());
    }

    /// the size of the vectors this rotation turns, for a rotation that heads an ExtendedPose
    static constexpr int ActionDimension = 3;

    /** Writes the rotation matrix of \a q through \a out, 3 by 3: what a rotation that heads an ExtendedPose turns
     * its vectors by.
     */
    template <typename Q, typename Out>
    void
    rotationMatrix(Eigen::MatrixBase<Q> const & q, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        writable(out) = quaternionOf<T>(q).toRotationMatrix();
    }

    /** Writes \f$ V(d) = \int_0^1 R(\mathrm{Exp}(s d)) \, ds \f$ through \a out, 3 by 3: the left Jacobian of SO(3),
     * \f$ J_r(-d) \f$, by which an ExtendedPose's exp() turns a vector's step.
     */
    template <typename D, typename Out>
    void
    integratedRotation(Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        Eigen::Matrix<T, 3, 1> const reversed = -d.template cast<T>();
        writable(out) = rightJacobian(reversed);
    }

    /** Writes through \a out, 3 by 3, the derivative of \f$ R(\mathrm{Exp}(\phi)) v \f$ in \f$ \phi \f$ at zero,
     * \f$ -[v]_\times \f$: how a small rotation moves the vector \a v.
     */
    template <typename V, typename Out>
    void
    actionJacobian(Eigen::MatrixBase<V> const & v, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = writable(out);
        o << T(0), T(v(2)), -T(v(1)), -T(v(2)), T(0), T(v(0)), T(v(1)), -T(v(0)), T(0);
    }

    /** \returns the unit quaternion of the rotation vector \a d, the exponential map of SO(3):
     * \f$ \mathrm{Exp}(\delta) = (\hat\delta \sin\tfrac\theta2, \cos\tfrac\theta2) \f$ with
     * \f$ \theta = |\delta| \f$, and its series below a small angle, where it stays
     * differentiable. A process model that integrates a body rate \f$ \omega \f$ over \f$ \Delta t \f$
     * writes <tt>auto q = map(block<Attitude>(out, y)); q = q * exp(omega * dt);</tt>.
     */
    template <typename Derived>
    [[nodiscard]] static Eigen::Quaternion<typename Derived::Scalar>
    exp(Eigen::MatrixBase<Derived> const & d)
    {
        using T = typename Derived::Scalar;
        T const angle2 = d.squaredNorm();
        if (angle2 < T(1e-12))
        {
            T const scale = T(0.5) - angle2 / T(48);
            return Eigen::Quaternion<T>{T(1) - angle2 / T(8), d(0) * scale, d(1) * scale, d(2) * scale};
        }
        T const angle = Eigen::numext::sqrt(angle2);
        T const scale = T(Eigen::numext::sin(T(angle / T(2))) / angle);
        return Eigen::Quaternion<T>{Eigen::numext::cos(T(angle / T(2))), d(0) * scale, d(1) * scale, d(2) * scale};
    }

    /** \returns the rotation vector of the unit quaternion \a q, the logarithm map of SO(3), the
     * short way round: \a q and \f$ -q \f$ give the same vector, of norm at most \f$ \pi \f$.
     */
    template <typename Derived>
    [[nodiscard]] static Eigen::Matrix<typename Derived::Scalar, 3, 1>
    log(Eigen::QuaternionBase<Derived> const & q)
    {
        using T = typename Derived::Scalar;
        T const sign = q.w() < T(0) ? T(-1) : T(1);
        T const w = sign * q.w();
        Eigen::Matrix<T, 3, 1> const v = sign * q.vec();
        T const s2 = v.squaredNorm();
        if (s2 < T(1e-12))
        {
            return v * T(T(2) / w);
        }
        T const s = Eigen::numext::sqrt(s2);
        T const angle = T(2) * Eigen::numext::atan2(s, w);
        return v * T(angle / s);
    }

    /** \returns the right Jacobian of SO(3) at the rotation vector \a d,
     * \f[ J_r(\delta) = I - \frac{1 - \cos\theta}{\theta^2} [\delta]_\times
     *     + \frac{\theta - \sin\theta}{\theta^3} [\delta]_\times^2, \qquad \theta = |\delta|, \f]
     * from \ref Sola2018 "Solà, Deray, and Atchuthan (2018)", eq. (143), with its series
     * \f$ I - \tfrac12 [\delta]_\times + \tfrac16 [\delta]_\times^2 \f$ below a small angle. It takes
     * a change of the step to the change of \f$ q \boxplus \delta \f$ seen from there, so it is
     * both this space's reset Jacobian and the Jacobian of plus() in its step.
     */
    template <typename Derived>
    [[nodiscard]] static Eigen::Matrix<typename Derived::Scalar, 3, 3>
    rightJacobian(Eigen::MatrixBase<Derived> const & d)
    {
        using T = typename Derived::Scalar;
        auto skew = Eigen::Matrix<T, 3, 3>{};
        skew << T(0), -d(2), d(1), d(2), T(0), -d(0), -d(1), d(0), T(0);
        T const angle2 = d.squaredNorm();
        if (angle2 < T(1e-8))
        {
            return Eigen::Matrix<T, 3, 3>::Identity() - T(0.5) * skew + (skew * skew) / T(6);
        }
        T const angle = Eigen::numext::sqrt(angle2);
        return Eigen::Matrix<T, 3, 3>::Identity() - T((T(1) - Eigen::numext::cos(angle)) / angle2) * skew
             + T((angle - Eigen::numext::sin(angle)) / (angle2 * angle)) * (skew * skew);
    }

    /** \returns the coordinates of the identity rotation, (0, 0, 0, 1) in this space's
     * (x, y, z, w) order, where <tt>mean << 1, 0, 0, 0</tt> would be a half turn about x.
     */
    template <typename Scalar = double>
    [[nodiscard]] static Eigen::Matrix<Scalar, 4, 1>
    identity()
    {
        return Eigen::Matrix<Scalar, 4, 1>{Scalar(0), Scalar(0), Scalar(0), Scalar(1)};
    }

    /** \returns the coordinates of \a q in this space's order, (x, y, z, w): its
     * <tt>coeffs()</tt>, for writing an <tt>Eigen::Quaternion</tt> into a mean without counting
     * where w goes.
     */
    template <typename Derived>
    [[nodiscard]] static Eigen::Matrix<typename Derived::Scalar, 4, 1>
    coefficientsOf(Eigen::QuaternionBase<Derived> const & q)
    {
        return q.coeffs();
    }

    /** \returns \a block, four coordinates (x, y, z, w) in one column, as an
     * <tt>Eigen::Map</tt> of an <tt>Eigen::Quaternion</tt> over the same memory: writable where
     * \a block is, otherwise a map of a const quaternion. Nothing is copied, so the map must not
     * outlive the storage, and a non-unit quaternion written through it is canonicalized where the
     * library takes values from outside, by normalizeValues(). Copy it with
     * <tt>Eigen::Quaterniond q = map(block);</tt>. The attitude block of a whole state is
     * <tt>UnitQuaternion::map(block<Attitude>(space, x))</tt>, and that of vehicle \a k in a
     * Repeated block <tt>UnitQuaternion::map(block<Fleet>(space, x, k))</tt> (TagPath.h).
     *
     * \a block must have direct access with unit inner stride, such as a column, a segment, or a
     * whole vector; an unevaluated expression does not compile, and is to be evaluated first. A
     * plain matrix passed as a temporary does not compile either, since the map would dangle; a
     * temporary <tt>Block</tt>, <tt>Map</tt>, or <tt>Ref</tt> views storage that outlives it and is
     * accepted.
     */
    template <typename Block>
    [[nodiscard]] static auto
    map(Block && block)
    {
        using Derived = std::remove_cvref_t<Block>;
        static_assert(MatrixExpression<Block>, "map() takes an Eigen vector of four coordinates");
        static_assert(
            Derived::ColsAtCompileTime == 1
                && (Derived::RowsAtCompileTime == 4 || Derived::RowsAtCompileTime == Eigen::Dynamic),
            "map() takes one column of four coordinates (x, y, z, w)"
        );
        static_assert(
            (Derived::Flags & Eigen::DirectAccessBit) && Derived::InnerStrideAtCompileTime == 1,
            "map() needs coefficients in memory with unit stride; evaluate the expression first"
        );
        static_assert(
            std::is_lvalue_reference_v<Block> || !std::derived_from<Derived, Eigen::PlainObjectBase<Derived>>,
            "map() of a temporary matrix would dangle; store it first"
        );
        EMRYS_FILTER_ASSERT(block.size() == 4 && "a unit quaternion has four coordinates");
        using Scalar = typename Derived::Scalar;
        if constexpr (!std::is_const_v<std::remove_reference_t<Block>> && bool(Derived::Flags & Eigen::LvalueBit))
        {
            return Eigen::Map<Eigen::Quaternion<Scalar>>{block.derived().data()};
        }
        else
        {
            return Eigen::Map<Eigen::Quaternion<Scalar> const>{block.derived().data()};
        }
    }

private:

    /** \returns the quaternion of the coefficients (x, y, z, w) in \a coefficients, one column whose
     * size may be known only at run time, in the scalar type \a T. Eigen builds a quaternion only
     * from a vector whose size is four at compile time.
     */
    template <typename T, typename Derived>
    [[nodiscard]] static Eigen::Quaternion<T>
    quaternionOf(Eigen::MatrixBase<Derived> const & coefficients)
    {
        Eigen::Matrix<T, 4, 1> const fixed = coefficients.template cast<T>();
        return Eigen::Quaternion<T>{fixed};
    }

public:

    /** Spaces compare equal: a UnitQuaternion has no parameters. */
    friend bool
    operator==(UnitQuaternion const &, UnitQuaternion const &) = default;
};

}  // namespace Emrys
