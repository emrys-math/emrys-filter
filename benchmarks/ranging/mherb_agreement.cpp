// The precondition of the ranging benchmark against mherb/kalman [benchmarks]: before any timing
// counts, each EmrysFilter filter of emrys_ranging_kinds.h and mherb/kalman's of mherb_ranging.h
// must agree in state and covariance, with the analytic Jacobian, the only one mherb/kalman has.
// They agree where cases.h's MherbAgreementCases say. The others are timed with a mark, each
// behind a test here that shows the difference to be mherb/kalman's: its EKF in float strays from
// the double estimate where EmrysFilter's does not, and its two UKFs, which agree with
// EmrysFilter's without process noise, do not reproduce its own EKF on a linear problem with it.
// Each EmrysFilter filter's consistency carries to mherb/kalman's through the agreement;
// agreement.cpp tests the EKF's. Roadmap: the benchmarks against other libraries.

#include "cases.h"
#include "disagreement.h"
#include "distance.h"
#include "emrys_ranging_kinds.h"
#include "mherb_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace
{

using Ranging::EmrysRangingKind;
using Ranging::FilterKind;
using Ranging::MherbRanging;

/// Runs the EmrysFilter and mherb/kalman filters of case \a C over 500 steps of \a scenario's
/// simulated run, and checks at every step that they agree to \a tolerance, relative to the size of
/// the state and of the covariance. The covariance is compared on its lower triangle, the part the
/// library guarantees.
template <typename C>
void
expectAgreement(Ranging::Scenario<C::Dimension, C::AnchorCount> const & scenario, double tolerance)
{
    constexpr int D = C::Dimension;
    constexpr int A = C::AnchorCount;
    using Scalar = typename C::Scalar;
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 500, 1));
    auto const mherbRanges = MherbRanging<D, A, C::Kind, Scalar>::rangesOf(ranges);
    auto emrys = EmrysRangingKind<D, A, C::Kind, Scalar>{scenario};
    auto mherb = MherbRanging<D, A, C::Kind, Scalar>{scenario};
    for (std::size_t k = 0; k < ranges.size(); ++k)
    {
        emrys.predict();
        mherb.predict();
        emrys.update(ranges[k]);
        mherb.update(mherbRanges[k]);
        double const meanScale = std::max(1.0, double(mherb.mean().norm()));
        double const covScale = std::max(1.0, double(mherb.covariance().norm()));
        auto const covDifference = (emrys.covarianceLower() - mherb.covariance()).template cast<double>().eval();
        ASSERT_LT((emrys.mean() - mherb.mean()).template cast<double>().norm(), tolerance * meanScale)
            << "at step " << k;
        ASSERT_LT(covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm(), tolerance * covScale)
            << "at step " << k;
    }
}

template <typename C>
class RangingMherbTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingMherbTest, Ranging::MherbAgreementCases);

// Both filters, driven through the same simulated run, agree at every step to agreementTolerance,
// 1e-9 relative to the size of the state and covariance in double and 1e-4 in float, as the
// plain-Eigen EKF does in agreement.cpp. They compute the same posterior by different arithmetic:
// mherb/kalman's EKF by an explicit inverse of S and P - K H P, its square-root EKF by forming
// P - K H P from the factor and factorizing it again. The worst differences seen are 9e-10 m in the
// mean in double and 6e-5 m in float; a modeling difference is of order one.
TYPED_TEST(RangingMherbTest, ImplementationsAgree)
{
    expectAgreement<TypeParam>(
        Ranging::makeScenario<TypeParam::Dimension, TypeParam::AnchorCount>(),
        Ranging::agreementTolerance<typename TypeParam::Scalar>
    );
}

template <typename C>
class RangingMherbUnscentedTest : public ::testing::Test
{};

using UnscentedCases = ::testing::Types<
    Ranging::KindCase<2, 4, FilterKind::Ukf>,
    Ranging::KindCase<3, 6, FilterKind::Ukf>,
    Ranging::KindCase<2, 4, FilterKind::SqrtUkf>,
    Ranging::KindCase<3, 6, FilterKind::SqrtUkf>>;

TYPED_TEST_SUITE(RangingMherbUnscentedTest, UnscentedCases);

// Without process noise the two UKFs agree over 500 steps, in double, to 1e-5 relative in 2D with 4
// anchors and to 1e-3 in 3D with 6, so they spread the same sigma points with the same weights. With
// process noise they differ by about 7 mm in both, because mherb/kalman updates with the points its
// prediction moved, which leave the process noise out. The difference left without it, at most
// 1.2e-6 m (1.5e-7 relative) in 2D and 1.3 mm (1.2e-4 relative in the mean, 6.7e-4 in the
// covariance) in 3D, over the first steps while the prior is wide, is most likely the other half of
// the same choice: the moved points are F times a Cholesky factor of the previous covariance,
// another square root of the predicted covariance than its own Cholesky factor, which EmrysFilter
// spreads. In 3D that leaves the test a margin of about 5 below the difference with process noise.
TYPED_TEST(RangingMherbUnscentedTest, AgreeWithoutProcessNoise)
{
    auto scenario = Ranging::makeScenario<TypeParam::Dimension, TypeParam::AnchorCount>();
    scenario.accelDensity = 0;
    expectAgreement<TypeParam>(scenario, TypeParam::Dimension == 2 ? 1e-5 : 1e-3);
}

// With process noise the UKF pairs are timed with a mark: over the same 500 steps the two
// libraries' filters differ by 5e-4 to 9e-4 relative in the mean and 2e-3 to 3e-3 in the
// covariance, within disagreementBound, so mherb/kalman's is still a filter of the same problem.
// The next test shows the difference to be its own.
TYPED_TEST(RangingMherbUnscentedTest, StayWithinTheBoundWithProcessNoise)
{
    expectAgreement<TypeParam>(
        Ranging::makeScenario<TypeParam::Dimension, TypeParam::AnchorCount>(),
        Ranging::disagreementBound
    );
}

/// A measurement of the position in \a D dimensions, linear in the state, with independent noise,
/// as a mherb/kalman measurement model.
template <int D>
class PositionMeasurement
    : public Kalman::
          LinearizedMeasurementModel<Kalman::Vector<double, 2 * D>, Kalman::Vector<double, D>, Kalman::StandardBase>
{
public:

    using State = Kalman::Vector<double, 2 * D>;
    using Position = Kalman::Vector<double, D>;

    /// The model whose noise has the standard deviation \a stdDev in each coordinate.
    explicit PositionMeasurement(double stdDev)
    {
        this->H.setZero();
        this->H.template leftCols<D>().setIdentity();
        this->V.setIdentity();
        this->setCovariance(stdDev * stdDev * Kalman::Covariance<Position>::Identity());
    }

    [[nodiscard]] Position
    h(State const & x) const override
    {
        return Position{x.template head<D>()};
    }

    void
    updateJacobians(State const &) override
    {}
};

/// The largest distance of mherb/kalman's UKF from its own EKF over 500 steps of a linear problem:
/// the motion of \a scenario, at its initial estimate, with the position measured in place of the
/// ranges. The measurements are the positions of a straight path with a fixed error added, a sine
/// of the step, so that they are the same with every standard library.
template <int D, int A>
[[nodiscard]] Ranging::Distance
unscentedFromExtendedOnALinearProblem(Ranging::Scenario<D, A> const & scenario)
{
    using State = Kalman::Vector<double, 2 * D>;
    using Position = Kalman::Vector<double, D>;
    auto system =
        Ranging::MherbConstantVelocity<D, double, Kalman::StandardBase>{scenario.timeStep, scenario.accelDensity};
    auto measurement = PositionMeasurement<D>{scenario.rangeStdDev};
    auto ekf = Kalman::ExtendedKalmanFilter<State>{};
    auto ukf = Kalman::UnscentedKalmanFilter<State>{1.0, 2.0, 0.0};
    ekf.init(State{scenario.initialMean});
    ukf.init(State{scenario.initialMean});
    ekf.setCovariance(Kalman::Covariance<State>{scenario.initialCov});
    ukf.setCovariance(Kalman::Covariance<State>{scenario.initialCov});
    auto worst = Ranging::Distance{0, 0};
    Eigen::Matrix<double, 2 * D, 1> truth = scenario.initialMean;
    for (int k = 0; k < 500; ++k)
    {
        truth.template head<D>() += scenario.timeStep * truth.template tail<D>();
        auto z = Position{};
        for (int i = 0; i < D; ++i)
        {
            z(i) = truth(i) + scenario.rangeStdDev * std::sin(1.7 * k + i);
        }
        ekf.predict(system);
        ukf.predict(system);
        ekf.update(measurement, z);
        ukf.update(measurement, z);
        worst = Ranging::largerOf(
            worst,
            Ranging::distance(ukf.getState(), ukf.getCovariance(), ekf.getState(), ekf.getCovariance())
        );
    }
    return worst;
}

// On a linear problem the Kalman filter is exact, and a UKF computes the same moments, since its
// sigma points are exact for a linear function (EmrysFilter's are tested to be, in
// tests/Models/sigma_points.cpp). mherb/kalman's UKF does so without process noise, to 1e-12 of
// its own EKF, and not with it: at the scenario's process noise it differs by 2e-3 relative in the
// covariance, the size of its difference from EmrysFilter's UKF on the ranging problem. Its
// update computes P_xy and P_yy from the sigma points its prediction moved, which were drawn
// before the process noise Q was added, so both leave Q out while its P includes it. EmrysFilter
// is not in this test: the difference is between mherb/kalman's two filters.
TEST(RangingMherbUnscented, DiffersFromItsOwnEkfOnALinearProblemWithProcessNoise)
{
    auto planar = Ranging::makeScenario<2, 4>();
    auto spatial = Ranging::makeScenario<3, 6>();
    EXPECT_GT(unscentedFromExtendedOnALinearProblem(planar).cov, 1e-3);
    EXPECT_GT(unscentedFromExtendedOnALinearProblem(spatial).cov, 1e-3);
    planar.accelDensity = 0;
    spatial.accelDensity = 0;
    Ranging::Distance const planarWithout = unscentedFromExtendedOnALinearProblem(planar);
    Ranging::Distance const spatialWithout = unscentedFromExtendedOnALinearProblem(spatial);
    EXPECT_LT(std::max(planarWithout.mean, planarWithout.cov), 1e-12);
    EXPECT_LT(std::max(spatialWithout.mean, spatialWithout.cov), 1e-12);
}

template <typename C>
class RangingMherbDisagreementTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingMherbDisagreementTest, Ranging::MherbDisagreementCases);

// In float the EKF pair is timed with a mark, since over the same 500 steps only mherb/kalman's
// EKF strays from the double estimate (disagreement.h): by 4e-2 relative in the mean and 9e-2 in
// the covariance in 2D with 4 anchors, where EmrysFilter's float EKF stays within 4e-6. The cause
// is its explicit inverse of S: a copy of its header whose gain is solved by the Cholesky factor
// of S instead, and nothing else changed, stays within 6e-5 at every size.
TYPED_TEST(RangingMherbDisagreementTest, OnlyMherbStraysInFloat)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using MherbFilter = MherbRanging<D, A, FilterKind::Ekf, float>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, 500, 1);
    auto const ranges = MherbFilter::rangesOf(Ranging::rangesIn<float>(run));
    auto mherb = MherbFilter{scenario};
    Ranging::expectOnlyTheOtherStrays(scenario, run, mherb, [&](MherbFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    });
}

}  // namespace
