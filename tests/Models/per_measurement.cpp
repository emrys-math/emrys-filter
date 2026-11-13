// Models built per measurement, the general form for what a model built once can't say
// ([measurement-space-functions]): a measurement whose fixed and repeated parts have correlated noise, and whose
// count, sizes, and noise come from the observation. Round-trip ranging from a pivot with time
// differences of arrival at the receivers that detected the dwell, its noise R = D S D^T from each
// timing error's variance, updates as the hand extended Kalman update does, as a model built for the
// dwell whose function captures it; and a linear measurement of a fix and each landmark's offset, with
// noise correlated across them, built per count, as the hand Kalman update.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <array>
#include <functional>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

namespace
{

struct Target : Linear<2>
{};

struct Receiver : Linear<2>
{};

struct Receivers : Repeated<Receiver, 8>
{
    using Repeated::Repeated;
};

using Scene = Composite<Target, Receivers>;

struct RoundTrip : Linear<1>
{};

struct Tdoa : Linear<1>
{};

struct Tdoas : Repeated<Tdoa, 8>
{
    using Repeated::Repeated;
};

using Reading = Composite<RoundTrip, Tdoas>;

using ReadingNoise = GaussianParams<Linear<Eigen::Dynamic, 9>>;

Eigen::Vector2d const pivot{0.0, 0.0};

// A dwell: the receivers that detected it, the round trip and the time differences observed, and
// the signal-to-noise ratio of the pivot and of each receiver that detected it.
struct Dwell
{
    std::array<Eigen::Index, 3> detected;
    Eigen::Vector4d z;
    double pivotSnr;
    Eigen::Vector3d receiverSnr;
};

Dwell const dwell{{0, 2, 3}, Eigen::Vector4d{100.5, 12.0, 25.0, 48.0}, 40.0, Eigen::Vector3d{10.0, 25.0, 4.0}};

// The variance of a timing error, in range units, at a signal-to-noise ratio.
double
timingVariance(double snr)
{
    return 4.0 / snr;
}

// The dwell's noise: v = D e for the independent timing errors e of the pivot and each receiver, so
// R = D S D^T, the round trip seeing the pivot's error twice and each difference minus it once.
ReadingNoise
dwellNoise(Dwell const & d)
{
    Eigen::Index const n = d.receiverSnr.size();
    Eigen::VectorXd variance(n + 1);
    variance(0) = timingVariance(d.pivotSnr);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        variance(i + 1) = timingVariance(d.receiverSnr(i));
    }
    Eigen::MatrixXd map = Eigen::MatrixXd::Zero(n + 1, n + 1);
    map(0, 0) = 2.0;
    map.col(0).tail(n).setConstant(-1.0);
    map.bottomRightCorner(n, n).setIdentity();
    return {Eigen::VectorXd::Zero(n + 1), map * variance.asDiagonal() * map.transpose()};
}

// The round trip to the target from the pivot, then the difference of each detecting receiver's
// range and the pivot's.
auto
dwellFunction(Dwell const & d)
{
    return [detected = d.detected]<typename T>(auto const & in, auto const &, auto const & x) {
        auto const target = block<Target>(in, x);
        auto const receivers = block<Receivers>(in, x);
        auto const range = [&](auto const & from) {
            return T(Eigen::numext::sqrt(T((target - from).squaredNorm())));
        };
        T const fromPivot = range(pivot.cast<T>());
        Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 9, 1> result(1 + Eigen::Index(detected.size()));
        result(0) = T(2.0) * fromPivot;
        for (std::size_t j = 0; j < detected.size(); ++j)
        {
            result(Eigen::Index(j) + 1) = range(receivers.template segment<2>(2 * detected[j])) - fromPivot;
        }
        return result;
    };
}

// The model of a dwell: its function captures the dwell, and its space and noise are sized by the
// receivers that detected it rather than by those the state holds.
auto
dwellModel(Dwell const & d)
{
    return measurementModel(
        dwellFunction(d),
        reads(path<Target>, path<Receivers>),
        outputSpace(Reading{RoundTrip{}, Tdoas{Tdoa{}, Eigen::Index(d.detected.size())}}),
        additiveNoise(dwellNoise(d))
    );
}

// A target and four receivers whose positions are known to a meter or so.
GaussianParams<Scene>
makeScene()
{
    Eigen::Matrix<double, 10, 1> mean;
    mean << 30.0, 40.0, 50.0, 0.0, 0.0, 50.0, -40.0, 10.0, 20.0, -60.0;
    Eigen::Matrix<double, 10, 10> cov = Eigen::Matrix<double, 10, 10>::Identity();
    cov.topLeftCorner<2, 2>() << 25.0, 5.0, 5.0, 16.0;
    return {Scene{Target{}, Receivers{Receiver{}, 4}}, mean, cov};
}

// The extended Kalman update by hand: H the derivative of the ranges at the prior mean.
template <typename Posterior>
void
expectHandUpdate(Posterior const & posterior)
{
    auto const prior = makeScene();
    Eigen::Matrix<double, 10, 1> const x = prior.mean();
    Eigen::Vector2d const t = x.head<2>();
    Eigen::Vector2d const toPivot = (t - pivot).normalized();
    Eigen::Vector4d predicted;
    predicted(0) = 2.0 * (t - pivot).norm();
    Eigen::Matrix<double, 4, 10> h = Eigen::Matrix<double, 4, 10>::Zero();
    h.block<1, 2>(0, 0) = 2.0 * toPivot.transpose();
    for (Eigen::Index j = 0; j < 3; ++j)
    {
        Eigen::Index const k = 2 + 2 * dwell.detected[std::size_t(j)];
        Eigen::Vector2d const r = x.segment<2>(k);
        Eigen::Vector2d const toReceiver = (t - r).normalized();
        predicted(j + 1) = (t - r).norm() - (t - pivot).norm();
        h.block<1, 2>(j + 1, 0) = (toReceiver - toPivot).transpose();
        h.block<1, 2>(j + 1, k) = -toReceiver.transpose();
    }
    Eigen::Matrix<double, 10, 10> const p = prior.covarianceLower().selfadjointView<Eigen::Lower>();
    Eigen::Matrix4d const r = dwellNoise(dwell).covarianceLower().selfadjointView<Eigen::Lower>();
    Eigen::Matrix4d const s = h * p * h.transpose() + r;
    Eigen::Matrix<double, 10, 4> const k = p * h.transpose() * s.inverse();
    EXPECT_LT((posterior.mean() - (x + k * (dwell.z - predicted))).norm(), 1e-9);
    Eigen::Matrix<double, 10, 10> const expected = p - k * h * p;
    Eigen::Matrix<double, 10, 10> const lower = posterior.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lower - Eigen::Matrix<double, 10, 10>{expected.triangularView<Eigen::Lower>()}).norm(), 1e-9);
}

}  // namespace

// The noise couples the round trip with every time difference, and the differences with each
// other, through the pivot's timing error, each variance from its own signal-to-noise ratio.
TEST(PerMeasurement, DwellNoiseCorrelatesThroughThePivot)
{
    Eigen::MatrixXd const r = dwellNoise(dwell).covarianceLower().selfadjointView<Eigen::Lower>();
    double const pivotVariance = timingVariance(dwell.pivotSnr);
    EXPECT_DOUBLE_EQ(r(0, 0), 4.0 * pivotVariance);
    EXPECT_DOUBLE_EQ(r(1, 0), -2.0 * pivotVariance);
    EXPECT_DOUBLE_EQ(r(2, 1), pivotVariance);
    EXPECT_DOUBLE_EQ(r(3, 3), pivotVariance + timingVariance(dwell.receiverSnr(2)));
}

// A model built for the dwell, its space sized by the receivers that detected it rather than by
// those the state holds, updates as the hand extended Kalman update does, whether the observation
// holds the model by copy or by reference.
TEST(PerMeasurement, ModelBuiltForADwellIsTheHandUpdate)
{
    auto const updated = Emrys::update(makeScene(), observe(methods::linearized(), dwellModel(dwell), dwell.z));
    ASSERT_TRUE(updated);
    expectHandUpdate(updated->branches[0].posterior);

    auto const ranging = dwellModel(dwell);
    auto const referred = Emrys::update(makeScene(), observe(methods::linearized(), std::cref(ranging), dwell.z));
    ASSERT_TRUE(referred);
    expectHandUpdate(referred->branches[0].posterior);
}

namespace
{

struct Position : Linear<2>
{};

struct Landmark : Linear<2>
{};

struct Landmarks : Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using Map = Composite<Position, Landmarks>;

struct Fix : Linear<2>
{};

struct Offset : Linear<2>
{};

struct Offsets : Repeated<Offset, 8>
{
    using Repeated::Repeated;
};

using Survey = Composite<Fix, Offsets>;

// A deterministic positive-definite matrix of size n, with every entry nonzero.
Eigen::MatrixXd
dense(Eigen::Index n, double seed)
{
    Eigen::MatrixXd const a = Eigen::MatrixXd::NullaryExpr(n, n, [&](Eigen::Index i, Eigen::Index j) {
        return std::sin(seed + 3.0 * double(i) + 7.0 * double(j));
    });
    return a * a.transpose() + Eigen::MatrixXd::Identity(n, n);
}

}  // namespace

// A linear measurement of a fix and each landmark's offset from the vehicle, with noise correlated
// across the fix and the offsets, is a matrix over the blocks read and a space sized for the count,
// built per measurement, and updates as the hand Kalman update does at each count.
TEST(PerMeasurement, LinearModelWithFixedAndRepeatedPartsIsTheHandUpdate)
{
    for (Eigen::Index const count : {Eigen::Index{2}, Eigen::Index{3}})
    {
        Eigen::Index const n = 2 + 2 * count;
        auto const prior = GaussianParams<Map>{
            Map{Position{}, Landmarks{Landmark{}, count}},
            Eigen::VectorXd::LinSpaced(n, -1.0, 2.0),
            dense(n, 1.0)
        };
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 18, 18> h = Eigen::MatrixXd::Zero(n, n);
        h.topLeftCorner<2, 2>().setIdentity();
        for (Eigen::Index i = 0; i < count; ++i)
        {
            h.block<2, 2>(2 + 2 * i, 0) = -Eigen::Matrix2d::Identity();
            h.block<2, 2>(2 + 2 * i, 2 + 2 * i).setIdentity();
        }
        Eigen::MatrixXd const r = dense(n, 2.0);
        auto const survey = measurementModel(
            h,
            reads(path<Position>, path<Landmarks>),
            outputSpace(Survey{Fix{}, Offsets{Offset{}, count}}),
            additiveNoise(GaussianParams<Linear<Eigen::Dynamic, 18>>{Eigen::VectorXd::Zero(n), r})
        );
        Eigen::VectorXd const observed = Eigen::VectorXd::LinSpaced(n, 0.5, -1.5);
        auto const updated = Emrys::update(prior, observe(methods::linearized(), survey, observed));
        ASSERT_TRUE(updated);

        Eigen::MatrixXd const full = h;
        Eigen::MatrixXd const p = prior.covarianceLower().selfadjointView<Eigen::Lower>();
        Eigen::MatrixXd const k = p * full.transpose() * (full * p * full.transpose() + r).inverse();
        auto const & posterior = updated->branches[0].posterior;
        EXPECT_LT((posterior.mean() - (prior.mean() + k * (observed - full * prior.mean()))).norm(), 1e-12);
        Eigen::MatrixXd const expected = p - k * full * p;
        Eigen::MatrixXd const lower = posterior.covarianceLower().template triangularView<Eigen::Lower>();
        EXPECT_LT((lower - Eigen::MatrixXd{expected.triangularView<Eigen::Lower>()}).norm(), 1e-12);
    }
}
