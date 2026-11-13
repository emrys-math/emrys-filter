// Moments and SampleSet (step 11, [sizes-and-dimensions]): the space fixes the size and is part of the type, fixed,
// dynamic, and bounded-dynamic sizes, resizing by dimensions or from a space whose differences have
// other sizes than its values, construction from a mean and covariance, mutable access, and mean()
// and covarianceLower() by tag path.

#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <type_traits>

using Emrys::Linear;
using Emrys::Moments;

namespace
{

struct Position : Emrys::Linear<2>
{};

struct Heading : Emrys::RadiansPi<>
{};

using State = Emrys::Composite<Position, Heading>;

}  // namespace

// Moments over a three-coordinate composite has fixed-size vector and matrix types, is not
// convertible to Moments over a plain Linear<3>, and the space's bound is part of the type.
TEST(Moments, SpaceSetsTheSizeAndIsPartOfTheType)
{
    using StateMoments = Moments<State>;
    static_assert(std::is_same_v<StateMoments::SpaceType, State>);
    static_assert(std::is_same_v<StateMoments::VectorType, Eigen::Vector3d>);
    static_assert(std::is_same_v<StateMoments::MatrixType, Eigen::Matrix3d>);
    // Same storage, different space: not interchangeable.
    static_assert(!std::is_convertible_v<StateMoments, Moments<Linear<3>>>);
    static_assert(std::is_same_v<Moments<Linear<Eigen::Dynamic, 6>>, Moments<Emrys::Linear<Eigen::Dynamic, 6>>>);
}

// SampleSet is an Eigen matrix with the space's value dimension as its row count.
TEST(Moments, SampleSetTakesItsSizeFromTheSpace)
{
    static_assert(std::is_same_v<Emrys::SampleSet<State>, Eigen::Matrix<double, 3, Eigen::Dynamic>>);
    static_assert(std::is_same_v<
                  Emrys::SampleSet<Emrys::Linear<Eigen::Dynamic, 5>>,
                  Emrys::SampleSet<Emrys::Linear<Eigen::Dynamic, 5>>>);
}

// A fixed-size Moments reports its dimensions and covariance shape.
TEST(Moments, FixedSize)
{
    auto m = Moments<Linear<3>>{};
    EXPECT_EQ(m.valueDimension(), 3);
    EXPECT_EQ(m.differenceDimension(), 3);
    EXPECT_EQ(m.covarianceLower().rows(), 3);
    EXPECT_EQ(m.covarianceLower().cols(), 3);
}

// A dynamic Moments starts empty and resize() sets its mean and covariance sizes.
TEST(Moments, DynamicSizeAndResize)
{
    auto m = Moments<Linear<Eigen::Dynamic>>{};
    EXPECT_EQ(m.valueDimension(), 0);
    m.resize(4, 4);
    EXPECT_EQ(m.valueDimension(), 4);
    EXPECT_EQ(m.mean().size(), 4);
    EXPECT_EQ(m.covarianceLower().rows(), 4);
    EXPECT_EQ(m.covarianceLower().cols(), 4);
}

// A bounded-dynamic Moments takes its size from the space instance and can resize up to the bound.
TEST(Moments, DynamicWithMaxSize)
{
    auto m = Moments<Linear<Eigen::Dynamic, 6>>{Linear<Eigen::Dynamic, 6>{2}};
    EXPECT_EQ(m.valueDimension(), 2);
    m.resize(6, 6);
    EXPECT_EQ(m.differenceDimension(), 6);
}

// resize() with a space takes each dimension from it: three unit quaternions have 12 value
// coordinates and 9 difference coordinates, so the mean and covariance get different sizes.
TEST(Moments, ResizeTakesItsDimensionsFromASpace)
{
    using Attitudes = Emrys::Repeated<Emrys::UnitQuaternion, 4>;
    auto m = Moments<Attitudes>{Attitudes{Emrys::UnitQuaternion{}, 1}};
    EXPECT_EQ(m.valueDimension(), 4);
    EXPECT_EQ(m.differenceDimension(), 3);
    m.resize(Attitudes{Emrys::UnitQuaternion{}, 3});
    EXPECT_EQ(m.valueDimension(), 12);
    EXPECT_EQ(m.mean().size(), 12);
    EXPECT_EQ(m.differenceDimension(), 9);
    EXPECT_EQ(m.covarianceLower().cols(), 9);
}

// The two-argument constructor stores the mean and covariance given.
TEST(Moments, FromMeanAndCov)
{
    auto const mean = Eigen::Vector2d{1.0, 2.0};
    auto const cov = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    auto m = Moments<Linear<Eigen::Dynamic>>{mean, cov};
    EXPECT_EQ(m.mean(), mean);
    EXPECT_EQ(m.covarianceLower(), cov);
}

// mean() and covarianceLower() return writable references.
TEST(Moments, MutableAccess)
{
    auto m = Moments<Linear<2>>{};
    m.mean().setConstant(3.0);
    m.covarianceLower().setIdentity();
    EXPECT_EQ(m.mean(), Eigen::Vector2d(3.0, 3.0));
    EXPECT_EQ(m.covarianceLower(), Eigen::Matrix2d::Identity());
}

// mean<Path>() and covarianceLower<Path>() select a block's rows, covarianceLower<A, B>() the cross
// block, each exactly as stored (the lower triangle is not mirrored), and both are writable.
TEST(Moments, MeanAndCovTakeAPath)
{
    auto m = Moments<State>{
        Eigen::Vector3d{1.0, 2.0, 3.0},
        Eigen::Matrix3d{{1.0, 4.0, 0.0}, {5.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}
    };
    EXPECT_EQ(Eigen::Vector2d{m.mean<Position>()}, Eigen::Vector2d(1.0, 2.0));
    auto const asStored = Eigen::Matrix2d{{1.0, 4.0}, {5.0, 2.0}};
    EXPECT_EQ(Eigen::Matrix2d{m.covarianceLower<Position>()}, asStored);
    EXPECT_EQ(m.covarianceLower<Heading>(), 3.0);
    EXPECT_EQ((Eigen::Vector2d{m.covarianceLower<Position, Heading>()}), Eigen::Vector2d::Zero());

    m.mean<Heading>() = 0.5;
    m.covarianceLower<Position>() = Eigen::Matrix2d::Identity();
    EXPECT_EQ(m.mean()(2), 0.5);
    EXPECT_EQ(m.covarianceLower()(1, 0), 0.0);
}
