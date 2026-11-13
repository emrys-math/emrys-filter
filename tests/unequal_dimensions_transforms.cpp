// Transforms of a state holding a unit quaternion, the space of tests/unequal_dimensions.h whose
// values have more coordinates than its differences, differentiated in the tangent space, and a
// unit vector on the sphere, three values and two differences, written here from scratch so that a
// user's own space of unequal sizes stays exercised, updated with its reset Jacobian taken by
// automatic differentiation (step 13e, [covariance-reset]).

#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include "unequal_dimensions.h"

#include <cmath>
#include <type_traits>

namespace methods = Emrys::methods;

using namespace EmrysTest::UnequalDimensions;

namespace
{

// A linearized transform of the pointing direction has the Jacobian -R [v]x in the attitude, the
// unscented transform agrees at a narrow spread and returns a five-row cross-covariance, and a
// transform into the quaternion space itself composes the attitude with a mount and rotates its
// body-frame covariance.
TEST(UnequalDimensions, TransformsDifferentiateInTheTangentSpace)
{
    GaussianParams<Pose> const g = pose();
    auto const forward = Eigen::Vector3d{1.0, 0.0, 0.0};
    auto const pointing = [&](auto const & x) {
        return rotate(x.template head<4>(), forward);
    };

    // R(q exp δ) v ≈ R v - R [v]x δ, so the Jacobian in the attitude is -R [v]x.
    Eigen::Matrix<double, 3, 5> jacobian = Eigen::Matrix<double, 3, 5>::Zero();
    jacobian.leftCols<3>() = -rotationMatrix(turned) * skew(forward);
    Eigen::Matrix3d const expectedCov = jacobian * poseCov() * jacobian.transpose();

    auto const linear = Emrys::transform(Emrys::methods::linearized(), pointing, g);
    EXPECT_TRUE(linear.mean().isApprox(rotate(turned, forward), 1e-12));
    EXPECT_TRUE(linear.covarianceLower().isApprox(expectedCov, 1e-10));

    // At this spread the rotation's curvature shows; at a hundredth of it the sigma points agree.
    auto const narrow = GaussianParams<Pose>{g.mean(), 0.01 * poseCov()};
    auto const sigma = Emrys::transform(Emrys::methods::unscented(), pointing, narrow);
    EXPECT_TRUE(sigma.mean().isApprox(rotate(turned, forward), 1e-3));
    EXPECT_TRUE(sigma.covarianceLower().isApprox(0.01 * expectedCov, 1e-2));
    auto const crossed = Emrys::transform(Emrys::methods::unscented(), pointing, Emrys::withCross(g));
    EXPECT_EQ(crossed.crossCovariance().rows(), 5);
    EXPECT_EQ(crossed.crossCovariance().cols(), 3);

    // Into a space with unequal dimensions: the attitude composed with a fixed rotation.
    Quaternion<double> const mount = exponential(Eigen::Vector3d{0.0, 0.0, 1.0});
    auto const mounted = [&]<typename T>(auto const & x) {
        return multiply(x.template head<4>(), mount.cast<T>());
    };
    auto const sensor = Emrys::transform(methods::unscented(), mounted, Emrys::outputSpace(Orientation{}), g);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(sensor)>, GaussianParams<Orientation>>);
    EXPECT_TRUE(sensor.mean().isApprox(multiply(turned, mount), 1e-9));
    // Right multiplication by R_m rotates the body-frame error: Σ' = R_mᵀ Σ R_m.
    Eigen::Matrix3d const rm = rotationMatrix(mount);
    EXPECT_TRUE(sensor.covarianceLower().isApprox(rm.transpose() * poseCov().topLeftCorner<3, 3>() * rm, 1e-6));
    auto const sensorLinear = Emrys::transform(methods::linearized(), mounted, Emrys::outputSpace(Orientation{}), g);
    EXPECT_TRUE(sensorLinear.covarianceLower().isApprox(rm.transpose() * poseCov().topLeftCorner<3, 3>() * rm, 1e-10));
}

// A direction: a unit vector in three coordinates, whose differences are two coordinates in a
// tangent basis at the base point, written from scratch as a user would. x ⊞ d turns x about the
// tangent rotation vector B(x) d; a ⊟ b is the rotation vector from b to a in the basis at b.
struct Direction
{
    static constexpr int FixedValueDimension = 3;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 3;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 3;
    }

    [[nodiscard]] Linear<2>
    difference() const
    {
        return {};
    }

    // An orthonormal basis of the tangent plane at the unit vector x, away from the z axis by
    // crossing with it and near it by crossing with x.
    template <typename T>
    [[nodiscard]] static Eigen::Matrix<T, 3, 2>
    basis(Eigen::Matrix<T, 3, 1> const & x)
    {
        Eigen::Matrix<T, 3, 1> const pole =
            Eigen::numext::abs(T(x(2))) < T(0.9) ? Eigen::Matrix<T, 3, 1>::UnitZ() : Eigen::Matrix<T, 3, 1>::UnitX();
        Eigen::Matrix<T, 3, 1> const u = pole.cross(x).normalized();
        auto b = Eigen::Matrix<T, 3, 2>{};
        b << u, x.cross(u);
        return b;
    }

    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block) const
    {
        auto & x = Emrys::writable(block);
        for (Eigen::Index j = 0; j < x.cols(); ++j)
        {
            x.col(j).normalize();
        }
    }

    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = Emrys::writable(out);
        for (Eigen::Index j = 0; j < o.cols(); ++j)
        {
            Eigen::Matrix<T, 3, 1> const base = x.col(j).template cast<T>();
            Eigen::Matrix<T, 3, 1> const turn = basis(base) * d.col(j).template cast<T>();
            o.col(j) = EmrysTest::rotate(EmrysTest::exponential(turn), base);
        }
        normalize(o);
    }

    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        using T = typename Out::Scalar;
        auto & o = Emrys::writable(out);
        for (Eigen::Index j = 0; j < o.cols(); ++j)
        {
            Eigen::Matrix<T, 3, 1> const from = b.col(j).template cast<T>();
            Eigen::Matrix<T, 3, 1> const to = a.col(j).template cast<T>();
            Eigen::Matrix<T, 3, 1> const axis = from.cross(to);
            T const sine = axis.norm();
            Eigen::Matrix<T, 3, 1> turn = axis;
            if (sine > T(1e-12))
            {
                turn *= T(Eigen::numext::atan2(sine, T(from.dot(to))) / sine);
            }
            o.col(j) = basis(from).transpose() * turn;
        }
    }

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(Direction const &, Direction const &) = default;
};

static_assert(Space<Direction>);
static_assert(!Emrys::hasIdentityReset<Direction>);

struct Heading : Direction
{};

struct Range : Linear<1>
{};

using Sighting = Composite<Heading, Range>;

// A user's sphere has three values and two differences, its plus() and minus() invert each other,
// and a Linearized update of a Gaussian on it by a measured direction keeps the mean on the sphere
// and moves it toward the measurement, with its reset Jacobian taken by automatic differentiation
// of the user's plus() and minus() [covariance-reset].
TEST(UnequalDimensions, AUserWrittenSphere)
{
    EXPECT_EQ(valueDimension(Sighting{}), 4);
    EXPECT_EQ(differenceDimension(Sighting{}), 3);

    Eigen::Vector3d const x = Eigen::Vector3d{1.0, 2.0, 0.5}.normalized();
    auto const d = Eigen::Vector2d{0.3, -0.2};
    Eigen::Vector3d const moved = plus(Heading{}, x, d);
    EXPECT_NEAR(moved.norm(), 1.0, 1e-15);
    EXPECT_NEAR(std::acos(moved.dot(x)), d.norm(), 1e-12);
    EXPECT_LT((minus(Heading{}, moved, x) - d).norm(), 1e-12);

    auto mean = Eigen::Vector4d{};
    mean << x, 10.0;
    auto const prior = GaussianParams<Sighting>{mean, 0.04 * Eigen::Matrix3d::Identity()};
    Eigen::Vector3d const observed = plus(Heading{}, x, Eigen::Vector2d{0.1, 0.05});
    // The heading is measured in its own space, so the residual is a tangent vector of two
    // coordinates and the noise has two.
    auto const sighting = Emrys::measurementModel(
        [](auto const & in, auto const &, auto const & x) {
            return Emrys::block<Heading>(in, x);
        },
        Emrys::reads(Emrys::path<Heading>),
        Emrys::outputSpace(Heading{}),
        Emrys::additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), 0.01 * Eigen::Matrix2d::Identity()})
    );
    auto const updated = Emrys::update(prior, observe(methods::linearized(), sighting, observed));
    ASSERT_TRUE(updated);
    EXPECT_EQ(updated->branches[0].innovation.residual().size(), 2);
    auto const & posterior = updated->branches[0].posterior;
    Eigen::Vector3d const heading = posterior.mean().head<3>();
    EXPECT_NEAR(heading.norm(), 1.0, 1e-14);
    EXPECT_LT(minus(Heading{}, observed, heading).norm(), 0.5 * minus(Heading{}, observed, x).norm());
    EXPECT_DOUBLE_EQ(posterior.mean()(3), 10.0);
    EXPECT_EQ(posterior.covarianceLower().rows(), 3);
}

}  // namespace
