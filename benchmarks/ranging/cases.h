#pragma once

// The cases the ranging tests run: a dimension, an anchor count, a linearization, and a scalar, as
// GoogleTest types, shared by the agreement and allocation tests.

#include "scenario.h"

#include <gtest/gtest.h>

namespace Ranging
{

/// A case of the benchmark: a dimension, an anchor count, a linearization, and a scalar.
template <int Dimension_, int AnchorCount_, Jacobian Kind_, typename Scalar_ = double>
struct Case
{
    static constexpr int Dimension = Dimension_;
    static constexpr int AnchorCount = AnchorCount_;
    static constexpr Jacobian Kind{Kind_};
    using Scalar = Scalar_;
};

/// The planar and spatial cases, under both linearizations, in double and in float, the scalar of
/// an embedded target [embedded-builds].
using Cases = ::testing::Types<
    Case<2, 4, Jacobian::Analytic>,
    Case<2, 4, Jacobian::AutoDiff>,
    Case<3, 6, Jacobian::Analytic>,
    Case<3, 6, Jacobian::AutoDiff>,
    Case<2, 4, Jacobian::Analytic, float>,
    Case<2, 4, Jacobian::AutoDiff, float>,
    Case<3, 6, Jacobian::Analytic, float>,
    Case<3, 6, Jacobian::AutoDiff, float>>;

/// A case of the benchmark against mherb/kalman: a dimension, an anchor count, which of the four
/// filters, and a scalar. mherb/kalman has no automatic differentiation, so the Jacobian is
/// analytic.
template <int Dimension_, int AnchorCount_, FilterKind Kind_, typename Scalar_ = double>
struct KindCase
{
    static constexpr int Dimension = Dimension_;
    static constexpr int AnchorCount = AnchorCount_;
    static constexpr FilterKind Kind{Kind_};
    using Scalar = Scalar_;
};

/// Every filter of both libraries in the planar and spatial cases, in double and in float.
using MherbCases = ::testing::Types<
    KindCase<2, 4, FilterKind::Ekf>,
    KindCase<3, 6, FilterKind::Ekf>,
    KindCase<2, 4, FilterKind::Ekf, float>,
    KindCase<3, 6, FilterKind::Ekf, float>,
    KindCase<2, 4, FilterKind::Ukf>,
    KindCase<3, 6, FilterKind::Ukf>,
    KindCase<2, 4, FilterKind::Ukf, float>,
    KindCase<3, 6, FilterKind::Ukf, float>,
    KindCase<2, 4, FilterKind::SqrtEkf>,
    KindCase<3, 6, FilterKind::SqrtEkf>,
    KindCase<2, 4, FilterKind::SqrtEkf, float>,
    KindCase<3, 6, FilterKind::SqrtEkf, float>,
    KindCase<2, 4, FilterKind::SqrtUkf>,
    KindCase<3, 6, FilterKind::SqrtUkf>,
    KindCase<2, 4, FilterKind::SqrtUkf, float>,
    KindCase<3, 6, FilterKind::SqrtUkf, float>>;

/// The cases in which mherb/kalman's filter agrees with EmrysFilter's, and so is timed: the EKF in
/// double, and the square-root EKF in double and in float. The EKF in float does not agree: its
/// explicit inverse of S loses up to a third of a meter of the mean in 2D with 4 anchors, where
/// EmrysFilter's float filter stays within 1e-4 of its double one (mherb_benchmark.cpp prints both).
/// Neither UKF agrees in either scalar: mherb/kalman's updates with the sigma points its prediction
/// moved, which leave out the process noise, rather than with points drawn from the predicted
/// covariance, a difference of 7 to 13 mm in the mean.
using MherbAgreementCases = ::testing::Types<
    KindCase<2, 4, FilterKind::Ekf>,
    KindCase<3, 6, FilterKind::Ekf>,
    KindCase<2, 4, FilterKind::SqrtEkf>,
    KindCase<3, 6, FilterKind::SqrtEkf>,
    KindCase<2, 4, FilterKind::SqrtEkf, float>,
    KindCase<3, 6, FilterKind::SqrtEkf, float>>;

/// The cases of the benchmark against TinyEKF, which has only the EKF: the planar and spatial
/// cases in double and in float.
using TinyEkfCases = ::testing::Types<
    KindCase<2, 4, FilterKind::Ekf>,
    KindCase<3, 6, FilterKind::Ekf>,
    KindCase<2, 4, FilterKind::Ekf, float>,
    KindCase<3, 6, FilterKind::Ekf, float>>;

/// The cases in which TinyEKF's EKF agrees with EmrysFilter's, and so is timed: those in double.
/// In float it strays up to 5e-4 relative in the mean and 2e-3 in the covariance in 2D with 4
/// anchors, where EmrysFilter's float EKF stays within 1e-5 of its double one
/// (tinyekf_benchmark.cpp prints both).
using TinyEkfAgreementCases = ::testing::Types<KindCase<2, 4, FilterKind::Ekf>, KindCase<3, 6, FilterKind::Ekf>>;

/// The cases of the benchmark against OpenCV, which has only the linear filter, run as an EKF: the
/// planar and spatial cases, and the planar case with 8 anchors, from which OpenCV's steps allocate
/// in double, each in double and in float.
using OpenCvCases = ::testing::Types<
    KindCase<2, 4, FilterKind::Ekf>,
    KindCase<2, 8, FilterKind::Ekf>,
    KindCase<3, 6, FilterKind::Ekf>,
    KindCase<2, 4, FilterKind::Ekf, float>,
    KindCase<2, 8, FilterKind::Ekf, float>,
    KindCase<3, 6, FilterKind::Ekf, float>>;

/// The cases in which OpenCV's EKF agrees with EmrysFilter's over the timed run, and so is timed:
/// those in double (opencv_benchmark.cpp prints the float distances and the long run).
using OpenCvAgreementCases = ::testing::Types<KindCase<2, 4, FilterKind::Ekf>, KindCase<3, 6, FilterKind::Ekf>>;

}  // namespace Ranging
