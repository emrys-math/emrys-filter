// Lie groups as spaces (docs/guide/groups.md, a prototype): which spaces satisfy LieGroup; the group members of
// Linear, Periodic, UnitQuaternion, Composite, and Repeated through the free functions of Spaces/Group.h; and
// ExtendedPose, the family of a rotation and the vectors it turns, of fixed size, of run-time size, and in the plane
// under a heading in degrees. checkGroup() checks each group's members against each other; expected values for the
// poses come from Eigen's 4 by 4 homogeneous matrices and its matrix exponential, an implementation independent of
// the space's.

#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/ExtendedPose.h>
#include <EmrysFilter/Spaces/Group.h>
#include <EmrysFilter/Spaces/GroupChecks.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

#include <numbers>

using Emrys::block;
using Emrys::checkGroup;
using Emrys::Composite;
using Emrys::Degrees360;
using Emrys::DifferenceMatrix;
using Emrys::DifferenceVector;
using Emrys::ExtendedPose;
using Emrys::hasIdentityAdjoint;
using Emrys::hasIdentityReset;
using Emrys::LieGroup;
using Emrys::Linear;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::Rotation;
using Emrys::Space;
using Emrys::UnitQuaternion;
using Emrys::ValueVector;

namespace
{

struct Attitude : UnitQuaternion
{};

struct Velocity : Linear<3>
{};

struct Position : Linear<3>
{};

struct Foot : Linear<3>
{};

struct Feet : Repeated<Foot>
{
    using Repeated::Repeated;
};

struct Bias : Linear<3>
{};

struct Heading : RadiansPi<>
{};

struct Compass : Degrees360<>
{};

struct Place : Linear<2>
{};

struct Pose : ExtendedPose<Attitude, Position>
{};

struct Navigation : ExtendedPose<Attitude, Velocity, Position>
{};

struct Body : ExtendedPose<Attitude, Velocity, Position, Feet>
{
    using ExtendedPose::ExtendedPose;
};

struct PlanarPose : ExtendedPose<Compass, Place>
{};

// A space with a plus() of its own that is not a group: a unit direction would be one.
struct NotAGroup
{
    static constexpr int FixedValueDimension = 1;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 1;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 1;
    }

    [[nodiscard]] Linear<1>
    difference() const
    {
        return {};
    }

    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        Emrys::writable(out) = x + d;
    }

    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        Emrys::writable(out) = a - b;
    }

    [[maybe_unused]] friend bool
    operator==(NotAGroup const &, NotAGroup const &) = default;
};

static_assert(LieGroup<Linear<3>>);
static_assert(LieGroup<Heading>);
static_assert(LieGroup<UnitQuaternion>);
static_assert(LieGroup<Attitude>);
static_assert(LieGroup<Composite<Attitude, Bias>>);
static_assert(LieGroup<Repeated<Attitude>>);
static_assert(LieGroup<Pose>);
static_assert(LieGroup<Body>);
static_assert(LieGroup<Composite<Body, Bias>>);
static_assert(LieGroup<PlanarPose>);
static_assert(Space<NotAGroup> && !LieGroup<NotAGroup>);
static_assert(!LieGroup<Composite<Position, NotAGroup>>);

static_assert(Rotation<Attitude>);
static_assert(Rotation<Compass>);
static_assert(!Rotation<Position>);

static_assert(hasIdentityAdjoint<Composite<Position, Heading>>);
static_assert(!hasIdentityAdjoint<Attitude>);
static_assert(!hasIdentityAdjoint<Composite<Pose, Bias>>);

static_assert(Space<Body> && !hasIdentityReset<Body>);
static_assert(std::is_empty_v<Navigation>);
static_assert(Emrys::valueDimensionAtCompileTime<Navigation> == 10);
static_assert(Emrys::valueDimensionAtCompileTime<Emrys::DifferenceSpace<Navigation>> == 9);
static_assert(Emrys::valueDimensionAtCompileTime<Body> == Eigen::Dynamic);

constexpr double tolerance = 1e-12;

// The 4 by 4 homogeneous matrix of a pose value (x, y, z, w, p).
Eigen::Matrix4d
homogeneous(Eigen::Matrix<double, 7, 1> const & x)
{
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.topLeftCorner<3, 3>() = Eigen::Quaterniond{x(3), x(0), x(1), x(2)}.toRotationMatrix();
    result.topRightCorner<3, 1>() = x.tail<3>();
    return result;
}

// The 4 by 4 matrix of the Lie algebra element of a pose difference (phi, rho).
Eigen::Matrix4d
hat(Eigen::Matrix<double, 6, 1> const & d)
{
    Eigen::Matrix4d result = Eigen::Matrix4d::Zero();
    result.topLeftCorner<3, 3>() << 0, -d(2), d(1), d(2), 0, -d(0), -d(1), d(0), 0;
    result.topRightCorner<3, 1>() = d.tail<3>();
    return result;
}

Eigen::Matrix<double, 7, 1>
somePose(double scale)
{
    Eigen::Matrix<double, 7, 1> x;
    x.head<4>() = UnitQuaternion::exp(scale * Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs();
    x.tail<3>() = scale * Eigen::Vector3d{1.5, -2.0, 0.5};
    return x;
}

TEST(Group, AdditiveGroupsComposeByAddition)
{
    Eigen::Vector3d const a{1, 2, 3};
    Eigen::Vector3d const b{0.5, -1, 4};
    EXPECT_TRUE(Emrys::compose<Position>(a, b).isApprox(a + b));
    EXPECT_TRUE(Emrys::inverse<Position>(a).isApprox(-a));
    EXPECT_TRUE(Emrys::identity<Position>().isZero());
    EXPECT_TRUE(Emrys::exp<Position>(a).isApprox(a));
    EXPECT_TRUE(Emrys::adjoint<Position>(a).isIdentity());
    EXPECT_LT(checkGroup(Position{}, a, b, Eigen::Vector3d{0.1, 0.2, -0.3}).worst(), tolerance);

    // A heading wraps: 3 rad and 1 rad compose to 4 - 2 pi.
    Eigen::Matrix<double, 1, 1> const three{3.0};
    Eigen::Matrix<double, 1, 1> const one{1.0};
    EXPECT_NEAR(Emrys::compose<Heading>(three, one)(0), 4.0 - 2 * std::numbers::pi, tolerance);
    EXPECT_LT(checkGroup(Heading{}, three, one, Eigen::Matrix<double, 1, 1>{0.7}).worst(), tolerance);
}

TEST(Group, UnitQuaternionIsAGroup)
{
    Eigen::Vector4d const a = UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs();
    Eigen::Vector4d const b = UnitQuaternion::exp(Eigen::Vector3d{-1.1, 0.3, 0.2}).coeffs();
    Eigen::Vector3d const d{0.3, 0.5, -0.4};
    EXPECT_LT(checkGroup(Attitude{}, a, b, d).worst(), tolerance);
    Eigen::Quaterniond const product = Eigen::Quaterniond{a} * Eigen::Quaterniond{b};
    EXPECT_TRUE(Emrys::compose<Attitude>(a, b).isApprox(product.coeffs()));
    EXPECT_TRUE(Emrys::adjoint<Attitude>(a).isApprox(Eigen::Quaterniond{a}.toRotationMatrix()));
}

TEST(Group, CompositeIsAGroupBlockByBlock)
{
    using State = Composite<Attitude, Bias>;
    Eigen::Matrix<double, 7, 1> a;
    a << UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs(), 1, 2, 3;
    Eigen::Matrix<double, 7, 1> b;
    b << UnitQuaternion::exp(Eigen::Vector3d{-1.1, 0.3, 0.2}).coeffs(), -1, 0.5, 2;
    Eigen::Matrix<double, 6, 1> d;
    d << 0.3, 0.5, -0.4, 0.1, 0.2, 0.3;
    EXPECT_LT(checkGroup(State{}, a, b, d).worst(), tolerance);
    DifferenceMatrix<State> const ad = Emrys::adjoint<State>(a);
    EXPECT_TRUE(ad.bottomRightCorner(3, 3).isIdentity());
    EXPECT_TRUE(ad.topRightCorner(3, 3).isZero());
}

TEST(Group, PoseMatchesHomogeneousMatrices)
{
    auto const a = somePose(1.0);
    auto const b = somePose(-0.6);
    Eigen::Matrix<double, 6, 1> d;
    d << 0.3, 0.5, -0.4, 1.0, -2.0, 0.7;

    EXPECT_TRUE(homogeneous(Emrys::compose<Pose>(a, b)).isApprox(homogeneous(a) * homogeneous(b), 1e-12));
    EXPECT_TRUE(homogeneous(Emrys::inverse<Pose>(a)).isApprox(homogeneous(a).inverse(), 1e-12));
    EXPECT_TRUE(homogeneous(Emrys::exp<Pose>(d)).isApprox(hat(d).exp(), 1e-12));
    EXPECT_TRUE(homogeneous(Emrys::plus<Pose>(a, d)).isApprox(homogeneous(a) * hat(d).exp(), 1e-12));
    // Ad_a d is the element whose matrix is a hat(d) a^-1.
    Eigen::Matrix<double, 6, 1> const turned = Emrys::adjoint<Pose>(a) * d;
    EXPECT_TRUE(hat(turned).isApprox(homogeneous(a) * hat(d) * homogeneous(a).inverse(), 1e-12));
    EXPECT_LT(checkGroup(Pose{}, a, b, d).worst(), 1e-11);

    // A position's rows are a step in the body frame: a step in them alone moves the position by R times it.
    Eigen::Matrix<double, 6, 1> alone = Eigen::Matrix<double, 6, 1>::Zero();
    alone.tail<3>() << 0.1, 0.2, 0.3;
    Eigen::Matrix<double, 7, 1> const moved = Emrys::plus<Pose>(a, alone);
    EXPECT_TRUE(moved.head<4>().isApprox(a.head<4>()));
    EXPECT_TRUE(
        (moved.tail<3>() - a.tail<3>()).isApprox(Eigen::Quaterniond{a(3), a(0), a(1), a(2)} * alone.tail<3>().eval())
    );
}

TEST(Group, ResetJacobianIsTheRightJacobian)
{
    auto const x = somePose(1.0);
    Eigen::Matrix<double, 6, 1> d;
    d << 0.3, 0.5, -0.4, 1.0, -2.0, 0.7;
    // By differences of plus() and minus(), as the reset is defined [covariance-reset].
    Eigen::Matrix<double, 7, 1> const moved = Emrys::plus<Pose>(x, d);
    Eigen::Matrix<double, 6, 6> numeric;
    double const h = 1e-6;
    for (int k = 0; k < 6; ++k)
    {
        Eigen::Matrix<double, 6, 1> up = d;
        Eigen::Matrix<double, 6, 1> down = d;
        up(k) += h;
        down(k) -= h;
        numeric.col(k) = (Emrys::minus<Pose>(Emrys::plus<Pose>(x, up), moved)
                          - Emrys::minus<Pose>(Emrys::plus<Pose>(x, down), moved))
                       / (2 * h);
    }
    Eigen::Matrix<double, 6, 6> const reset = Emrys::resetJacobian(Pose{}, x, d);
    EXPECT_TRUE(reset.isApprox(numeric, 1e-8));
    EXPECT_TRUE(reset.isApprox(Emrys::rightJacobian<Pose>(d)));
    EXPECT_TRUE(Emrys::leftJacobian<Pose>(d).isApprox(Emrys::rightJacobian<Pose>((-d).eval())));
}

TEST(Group, RunTimeSizedGroup)
{
    auto const space = Body{Attitude{}, Velocity{}, Position{}, Feet{Foot{}, 2}};
    ASSERT_EQ(valueDimension(space), 16);
    ASSERT_EQ(differenceDimension(space), 15);
    Eigen::VectorXd a(16);
    a << UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs(), 1, 2, 3, -1, 0.5, 2, 0.3, 0.1, -0.2, 4, 5, 6;
    Eigen::VectorXd b(16);
    b << UnitQuaternion::exp(Eigen::Vector3d{-1.1, 0.3, 0.2}).coeffs(), 0.2, -0.4, 1, 2, 2, -3, 1, 1, 1, -2, 0, 3;
    Eigen::VectorXd const d = Eigen::VectorXd::LinSpaced(15, -0.5, 0.6);
    EXPECT_LT(checkGroup(space, a, b, d).worst(), 1e-11);

    // The second foot by its path, and its rows of a difference.
    EXPECT_TRUE(block<Feet>(space, a, 1).isApprox(Eigen::Vector3d{4, 5, 6}));
    EXPECT_EQ(Emrys::diffIdx<Feet>(space, 1).first(), 12);

    // Each foot is turned as the position is: the pose of the attitude and one foot agrees with the whole.
    struct FootPose : ExtendedPose<Attitude, Foot>
    {};

    Eigen::Matrix<double, 7, 1> footA;
    footA << a.head<4>(), a.tail<3>();
    Eigen::Matrix<double, 6, 1> footD;
    footD << d.head<3>(), d.tail<3>();
    Eigen::VectorXd const moved = Emrys::plus(space, a, d);
    EXPECT_TRUE(moved.tail<3>().isApprox(Emrys::plus<FootPose>(footA, footD).tail<3>()));
}

TEST(Group, PlanarPoseUnderAHeadingInDegrees)
{
    Eigen::Vector3d const a{30.0, 1.0, 2.0};
    Eigen::Vector3d const b{100.0, -0.5, 0.3};
    Eigen::Vector3d const d{20.0, 0.4, -0.2};
    EXPECT_LT(checkGroup(PlanarPose{}, a, b, d).worst(), 1e-11);

    double const radians = 30.0 * std::numbers::pi / 180.0;
    Eigen::Rotation2Dd const turn{radians};
    Eigen::Vector3d const product = Emrys::compose<PlanarPose>(a, b);
    EXPECT_NEAR(product(0), 130.0, tolerance);
    EXPECT_TRUE(product.tail<2>().isApprox(a.tail<2>() + turn * b.tail<2>().eval()));
    // The heading's row of the adjoint is in degrees: a degree of turn at a moves the world-frame step by
    // -[1]x p, per radian.
    DifferenceMatrix<PlanarPose> const ad = Emrys::adjoint<PlanarPose>(a);
    EXPECT_NEAR(ad(1, 0), a(2) * std::numbers::pi / 180.0, tolerance);
    EXPECT_NEAR(ad(2, 0), -a(1) * std::numbers::pi / 180.0, tolerance);
}

}  // namespace
