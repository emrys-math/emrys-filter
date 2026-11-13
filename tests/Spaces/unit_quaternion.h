#pragma once

// Unit quaternion helpers for the tests, over the library's Emrys::UnitQuaternion (step 15b, [unit-quaternion]):
// coefficient vectors (x, y, z, w), Eigen's storage order, with the product, conjugate, rotation,
// exponential, logarithm, and right Jacobian of SO(3). The library's space writes its reset
// Jacobian in closed form; AutoDiffQuaternion is the same space without it, so its reset comes from
// automatic differentiation, and the two are the two paths of [covariance-reset]. Used by
// tests/unequal_dimensions.h and its tests, tests/Spaces/reset.cpp, tests/Spaces/unit_quaternion.cpp,
// and tests/allocation.cpp.

#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace EmrysTest
{

using Emrys::UnitQuaternion;

template <typename T>
using Quaternion = Eigen::Matrix<T, 4, 1>;

// The product of the quaternions p and q, each (x, y, z, w).
template <typename P, typename Q>
Quaternion<typename P::Scalar>
multiply(P const & p, Q const & q)
{
    using T = typename P::Scalar;
    Quaternion<T> const a = p.template cast<T>();
    Quaternion<T> const b = q.template cast<T>();
    return (Eigen::Quaternion<T>{a} * Eigen::Quaternion<T>{b}).coeffs();
}

template <typename Q>
Quaternion<typename Q::Scalar>
conjugate(Q const & q)
{
    return Quaternion<typename Q::Scalar>{-q(0), -q(1), -q(2), q(3)};
}

// The unit quaternion of the rotation vector d.
template <typename D>
Quaternion<typename D::Scalar>
exponential(D const & d)
{
    return UnitQuaternion::exponential(d).coeffs();
}

// The rotation vector of the unit quaternion q.
template <typename Q>
Eigen::Matrix<typename Q::Scalar, 3, 1>
logarithm(Q const & q)
{
    Quaternion<typename Q::Scalar> const fixed = q;
    return UnitQuaternion::logarithm(Eigen::Quaternion<typename Q::Scalar>{fixed});
}

// v rotated by the unit quaternion q.
template <typename Q, typename V>
Eigen::Matrix<typename Q::Scalar, 3, 1>
rotate(Q const & q, V const & v)
{
    using T = typename Q::Scalar;
    Quaternion<T> const fixed = q;
    Eigen::Matrix<T, 3, 1> const u = v.template cast<T>();
    return Eigen::Quaternion<T>{fixed} * u;
}

inline Eigen::Matrix3d
rotationMatrix(Quaternion<double> const & q)
{
    return Eigen::Quaterniond{q}.toRotationMatrix();
}

template <typename T>
Eigen::Matrix<T, 3, 3>
skew(Eigen::Matrix<T, 3, 1> const & v)
{
    auto s = Eigen::Matrix<T, 3, 3>{};
    s << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
    return s;
}

// The right Jacobian of SO(3) at the rotation vector d.
template <typename T>
Eigen::Matrix<T, 3, 3>
rightJacobian(Eigen::Matrix<T, 3, 1> const & d)
{
    return UnitQuaternion::rightJacobian(d);
}

// The library's unit quaternion without its closed-form reset Jacobian, so that the reset comes
// from automatic differentiation of plus() and minus().
struct AutoDiffQuaternion
{
    static constexpr int FixedValueDimension = 4;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 4;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 4;
    }

    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block) const
    {
        UnitQuaternion{}.normalize(block);
    }

    [[nodiscard]] Emrys::Linear<3>
    difference() const
    {
        return {};
    }

    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        UnitQuaternion{}.plus(x, d, out);
    }

    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        UnitQuaternion{}.minus(a, b, out);
    }

    friend bool
    operator==(AutoDiffQuaternion const &, AutoDiffQuaternion const &) = default;
};

}  // namespace EmrysTest
