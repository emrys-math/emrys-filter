// A space whose values have more coordinates than its differences, the unit quaternion of
// tests/unequal_dimensions.h, through spaces, Gaussians, their marginals and conditionals, whose
// covariances are carried to the tangent space at the moved mean (step 13e, [covariance-reset]), samples, particle
// sets, kernel densities, mixtures, and a resizable block of quaternions. The quaternion is the
// library's (step 15b, [unit-quaternion]). The filters are in unequal_dimensions_filters.cpp and the transforms
// and a user-written space in unequal_dimensions_transforms.cpp.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

#include <gtest/gtest.h>

#include "unequal_dimensions.h"

#include <Eigen/LU>

#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

using namespace EmrysTest::UnequalDimensions;

using Emrys::GaussianMixtureParams;
using Emrys::ParticleSet;
using Emrys::Repeated;

namespace
{

// valueDimension() and differenceDimension() differ for the quaternion and its composite, the value
// and difference index utilities place Position at different offsets, and plus() and minus() on the
// quaternion are inverse and keep it unit.
TEST(UnequalDimensions, SpacesNameBothDimensions)
{
    EXPECT_EQ(valueDimension(UnitQuaternion{}), 4);
    EXPECT_EQ(differenceDimension(UnitQuaternion{}), 3);
    EXPECT_EQ(valueDimension(Pose{}), 6);
    EXPECT_EQ(differenceDimension(Pose{}), 5);

    // Position follows four value coordinates but three difference coordinates.
    EXPECT_EQ(idx<Position>(Pose{}).first(), 4);
    EXPECT_EQ(diffIdx<Position>(Pose{}).first(), 3);
    EXPECT_EQ((diffIdx<Pose, Position>().first()), 3);
    EXPECT_EQ(diffIdx<Orientation>(Pose{}).size(), 3);
    BlockRows const rows = differenceBlockRows<Position>(Pose{});
    EXPECT_EQ(rows.start, 3);
    EXPECT_EQ(rows.size, 2);

    Eigen::Vector4d const q = turned;
    auto const d = Eigen::Vector3d{0.1, 0.2, -0.3};
    Eigen::Vector4d const moved = plus(Orientation{}, q, d);
    EXPECT_NEAR(moved.norm(), 1.0, 1e-15);
    EXPECT_TRUE(minus(Orientation{}, moved, q).isApprox(d, 1e-12));
}

// A Gaussian over the pose has a six-coordinate mean and a five-by-five covariance, its normalizer
// counts five degrees of freedom, its squared Mahalanobis distance and density use the tangent-space residual, and the
// path accessors read the mean by value and the covariance by difference indices.
TEST(UnequalDimensions, GaussianEvaluatesInTheTangentSpace)
{
    GaussianParams<Pose> const g = pose();
    EXPECT_EQ(g.valueDimension(), 6);
    EXPECT_EQ(g.differenceDimension(), 5);
    EXPECT_EQ(g.mean().size(), 6);
    EXPECT_EQ(g.covarianceLower().rows(), 5);

    auto const density = g.toDensity();
    ASSERT_TRUE(density);
    // The normalizer counts the five degrees of freedom, not the six coordinates.
    double const log2Pi = std::log(2 * std::numbers::pi);
    EXPECT_NEAR(density->logPeak(), -0.5 * (5 * log2Pi + std::log(poseCov().determinant())), 1e-12);

    auto offset = Eigen::Matrix<double, 5, 1>{};
    offset << 0.05, -0.1, 0.02, 0.5, -1.0;
    auto x = Eigen::Matrix<double, 6, 1>{};
    x << multiply(turned, exponential(offset.head<3>())), Eigen::Vector2d{10.5, -6.0};
    double const squaredDistance = offset.dot(poseCov().ldlt().solve(offset));
    EXPECT_NEAR(density->squaredMahalanobisDistance(x), squaredDistance, 1e-10);
    EXPECT_NEAR(density->logDensity(x), density->logPeak() - 0.5 * squaredDistance, 1e-10);

    // Path accessors read the mean by value coordinates and the covariance by difference ones.
    EXPECT_TRUE(g.mean<Position>().isApprox(Eigen::Vector2d{10.0, -5.0}));
    EXPECT_TRUE(Eigen::Matrix2d{g.covariance<Position>()}.isApprox(poseCov().bottomRightCorner<2, 2>()));
    EXPECT_TRUE(
        (Eigen::Matrix<double, 3, 2>{g.covariance<Orientation, Position>()}).isApprox(poseCov().topRightCorner<3, 2>())
    );
    EXPECT_TRUE(Eigen::Matrix2d{g.moments().covarianceLower<Position>()}.isApprox(poseCov().bottomRightCorner<2, 2>()));
}

// Marginals select the value and difference blocks, the attitude given the position has a mean
// moved by plus() with the gain times the residual and the Schur-complement covariance carried to
// the tangent space there by J_r of that step [covariance-reset], and conditioning on the quaternion reads its
// residual as a rotation vector.
TEST(UnequalDimensions, MarginalsAndConditionals)
{
    GaussianParams<Pose> const g = pose();
    auto const attitude = g.marginal<Orientation>().copy();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(attitude)>, GaussianParams<Orientation>>);
    EXPECT_TRUE(attitude.mean().isApprox(turned));
    EXPECT_TRUE(attitude.covarianceLower().isApprox(poseCov().topLeftCorner<3, 3>()));
    auto const place = g.marginal<Position>().copy();
    EXPECT_TRUE(place.covarianceLower().isApprox(poseCov().bottomRightCorner<2, 2>()));

    // Given the position, the attitude's conditional covariance is the Schur complement, reset.
    auto const at = Eigen::Vector2d{11.0, -4.0};
    GaussianParams<Orientation> const given = g.conditional<Position>(at);
    Eigen::Matrix<double, 3, 2> const cross = poseCov().topRightCorner<3, 2>();
    Eigen::Matrix2d const positionCov = poseCov().bottomRightCorner<2, 2>();
    Eigen::Matrix<double, 3, 2> const gain = cross * positionCov.inverse();
    Eigen::Vector3d const step = gain * (at - Eigen::Vector2d{10.0, -5.0});
    Eigen::Matrix3d const reset = rightJacobian(step);
    Eigen::Matrix3d const schur = poseCov().topLeftCorner<3, 3>() - gain * cross.transpose();
    EXPECT_TRUE(given.covarianceLower().isApprox(reset * schur * reset.transpose(), 1e-12));
    Eigen::Vector4d const expected = multiply(turned, exponential(step));
    EXPECT_TRUE(given.mean().isApprox(expected, 1e-12));

    // Conditioning on the quaternion takes its value and reads its residual as a rotation vector.
    auto const rotation = Eigen::Vector3d{0.1, 0.0, -0.1};
    GaussianParams<Position> const moved = g.conditional<Orientation>(multiply(turned, exponential(rotation)));
    Eigen::Matrix3d const attitudeCov = poseCov().topLeftCorner<3, 3>();
    Eigen::Matrix<double, 2, 3> const back = cross.transpose() * attitudeCov.inverse();
    EXPECT_TRUE(moved.mean().isApprox(Eigen::Vector2d{10.0, -5.0} + back * rotation, 1e-12));
}

// 40,000 samples are unit quaternions whose particle set reports both dimensions, has the
// tangent-space mean and five-by-five covariance of the source, and gives a kernel density with a
// five-by-five bandwidth.
TEST(UnequalDimensions, SamplesAreUnitQuaternionsWithTheRightSpread)
{
    GaussianParams<Pose> const g = pose();
    auto rng = std::mt19937_64{7};
    auto const samples = Emrys::drawSamples(g.toDensity().value(), 40000, rng);
    ASSERT_EQ(samples.rows(), 6);
    for (Eigen::Index j = 0; j < samples.cols(); ++j)
    {
        ASSERT_NEAR(samples.col(j).head<4>().norm(), 1.0, 1e-12);
    }

    auto const particles = ParticleSet<Pose>{Pose{}, samples};
    EXPECT_EQ(particles.valueDimension(), 6);
    EXPECT_EQ(particles.differenceDimension(), 5);
    // The tangent-space mean is within 0.01 rad of the source's and the sample covariance within
    // 5% of its 5-by-5 covariance, both a few standard errors at this count.
    auto const moments = particles.moments();
    ASSERT_TRUE(moments);
    EXPECT_LT(minus(Orientation{}, moments->mean().head<4>(), turned).norm(), 0.01);
    EXPECT_TRUE(moments->covarianceLower().isApprox(poseCov(), 0.05));

    auto const kde = particles.toDensity();
    ASSERT_TRUE(kde);
    EXPECT_EQ(kde->bandwidth().rows(), 5);
    EXPECT_TRUE(std::isfinite(kde->logDensity(g.mean())));
    EXPECT_GT(kde->logDensity(g.mean()), kde->logDensity(samples.col(0)) - 20);
}

// A mixture of two poses rotated either way about the mean has the mean attitude and the
// within-plus-between covariance in the tangent space, and its marginals and conditionals report
// the difference dimension.
TEST(UnequalDimensions, MixturesAverageInTheTangentSpace)
{
    auto mixture = GaussianMixtureParams<Pose>{Pose{}};
    auto const spread = Eigen::Vector3d{0.2, 0.0, 0.0};
    auto a = Eigen::Matrix<double, 6, 1>{};
    a << multiply(turned, exponential(spread)), 10.0, -5.0;
    auto b = Eigen::Matrix<double, 6, 1>{};
    b << multiply(turned, exponential(Eigen::Vector3d{-spread})), 12.0, -5.0;
    mixture.add(std::log(0.5), a, poseCov());
    mixture.add(std::log(0.5), b, poseCov());
    EXPECT_EQ(mixture.valueDimension(), 6);
    EXPECT_EQ(mixture.differenceDimension(), 5);

    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    auto const moments = density->moments();
    EXPECT_TRUE(moments.mean().head<4>().isApprox(turned, 1e-12));
    EXPECT_NEAR(moments.covarianceLower()(0, 0), poseCov()(0, 0) + 0.04, 1e-12);
    EXPECT_NEAR(moments.covarianceLower()(3, 3), poseCov()(3, 3) + 1.0, 1e-12);

    auto const attitude = mixture.marginal<Orientation>().copy();
    EXPECT_EQ(attitude.differenceDimension(), 3);
    auto const given = mixture.conditional<Position>(Eigen::Vector2d{11.0, -5.0});
    ASSERT_TRUE(given);
    EXPECT_EQ(given->size(), 2);
    EXPECT_EQ(given->differenceDimension(), 3);
}

struct Landmark : UnitQuaternion
{};

struct Landmarks : Repeated<Landmark, 4>
{
    using Repeated::Repeated;
};

using Map = Composite<Position, Landmarks>;

// Inserting and erasing quaternion landmarks in a Repeated block grows and shrinks the mean by four
// and the covariance by three per landmark, placing the cross-covariance by difference indices.
TEST(UnequalDimensions, InsertAndEraseAResizableBlock)
{
    auto g = GaussianParams<Map>{
        Map{Position{}, Landmarks{Landmark{}, 0}},
        Eigen::Vector2d{1.0, 2.0},
        Eigen::Matrix2d::Identity()
    };
    EXPECT_EQ(g.valueDimension(), 2);
    EXPECT_EQ(g.differenceDimension(), 2);

    Eigen::Matrix<double, 3, 2> cross = Eigen::Matrix<double, 3, 2>::Zero();
    cross(0, 0) = 0.1;
    g.insert<Landmarks>(0, turned, 0.5 * Eigen::Matrix3d::Identity(), cross);
    g.insert<Landmarks>(1, Quaternion<double>::UnitW(), 0.2 * Eigen::Matrix3d::Identity());
    EXPECT_EQ(g.valueDimension(), 10);
    EXPECT_EQ(g.differenceDimension(), 8);
    EXPECT_EQ(g.covarianceLower().rows(), 8);
    EXPECT_DOUBLE_EQ(g.covarianceLower()(2, 0), 0.1);
    EXPECT_DOUBLE_EQ(g.covarianceLower()(5, 5), 0.2);
    EXPECT_TRUE(g.marginal<Landmarks>(0).copy().mean().isApprox(turned));
    EXPECT_TRUE(g.marginal<Landmarks>(1).copy().covarianceLower().isApprox(0.2 * Eigen::Matrix3d::Identity()));

    g.erase<Landmarks>(0);
    EXPECT_EQ(g.valueDimension(), 6);
    EXPECT_EQ(g.differenceDimension(), 5);
    EXPECT_TRUE((g.covarianceLower().bottomRightCorner<3, 3>().isApprox(0.2 * Eigen::Matrix3d::Identity())));
    EXPECT_DOUBLE_EQ(g.covarianceLower()(2, 0), 0.0);
}

}  // namespace
