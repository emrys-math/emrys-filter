#pragma once

// The spaces and fixtures shared by the unequal-dimensions tests (tests/unequal_dimensions_*.cpp): a
// pose and a motion state that put a unit quaternion, four values and three differences, ahead of
// linear blocks so that those sit at different value and difference offsets, and a Gaussian over
// the pose.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

#include "Spaces/unit_quaternion.h"

#include <Eigen/Core>

#include <type_traits>

namespace EmrysTest::UnequalDimensions
{

using Emrys::BlockRows;
using Emrys::Composite;
using Emrys::differenceDimension;
using Emrys::DifferenceSpace;
using Emrys::diffIdx;
using Emrys::GaussianParams;
using Emrys::idx;
using Emrys::Linear;
using Emrys::minus;
using Emrys::Moments;
using Emrys::plus;
using Emrys::Space;
using Emrys::valueDimension;
using Emrys::internal::differenceBlockRows;

using EmrysTest::conjugate;
using EmrysTest::exponential;
using EmrysTest::multiply;
using EmrysTest::Quaternion;
using EmrysTest::rightJacobian;
using EmrysTest::rotate;
using EmrysTest::rotationMatrix;
using EmrysTest::skew;
using EmrysTest::UnitQuaternion;

static_assert(Space<UnitQuaternion>);
static_assert(std::is_same_v<DifferenceSpace<UnitQuaternion>, Linear<3>>);

struct Orientation : UnitQuaternion
{};

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

// The quaternion first, so that what follows it sits at different value and difference offsets.
using Pose = Composite<Orientation, Position>;

using Motion = Composite<Orientation, Position, Velocity>;

static_assert(Space<Pose>);
static_assert(
    Emrys::valueDimensionAtCompileTime<Pose> == 6 && Emrys::valueDimensionAtCompileTime<DifferenceSpace<Pose>> == 5
);

// Moments store a value-sized mean and a difference-sized covariance.
static_assert(Moments<Pose>::VectorType::RowsAtCompileTime == 6);

static_assert(Moments<Pose>::MatrixType::RowsAtCompileTime == 5 && Moments<Pose>::MatrixType::ColsAtCompileTime == 5);
static_assert(GaussianParams<Pose>::ValueDimensionAtCompileTime == 6);
static_assert(GaussianParams<Pose>::DifferenceDimensionAtCompileTime == 5);

inline Quaternion<double> const turned = exponential(Eigen::Vector3d{0.3, -0.2, 0.5});

inline Eigen::Matrix<double, 5, 5>
poseCov()
{
    auto a = Eigen::Matrix<double, 5, 5>{};
    a << 0.04, 0.01, 0.00, 0.02, 0.00,  //
        0.01, 0.03, 0.00, 0.00, 0.01,   //
        0.00, 0.00, 0.05, 0.00, 0.00,   //
        0.02, 0.00, 0.00, 1.00, 0.30,   //
        0.00, 0.01, 0.00, 0.30, 2.00;
    return a;
}

inline GaussianParams<Pose>
pose()
{
    auto mean = Eigen::Matrix<double, 6, 1>{};
    mean << turned, 10.0, -5.0;
    return GaussianParams<Pose>{mean, poseCov()};
}

}  // namespace EmrysTest::UnequalDimensions
