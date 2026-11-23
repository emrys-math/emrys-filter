// The precondition of the ranging benchmark against mherb/kalman [benchmarks]: before any timing
// counts, each EmrysFilter filter of emrys_ranging_kinds.h and mherb/kalman's of mherb_ranging.h
// must agree in state and covariance, with the analytic Jacobian, the only one mherb/kalman has.
// They agree where cases.h's MherbAgreementCases say, and only those are timed. The two UKFs,
// which differ by mherb/kalman's algorithm, are checked to agree without process noise instead,
// which shows that the pairs are set up alike. Each EmrysFilter filter's consistency carries to
// mherb/kalman's through the agreement; agreement.cpp tests the EKF's. Roadmap: the benchmarks
// against other libraries.

#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "mherb_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <algorithm>
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

// Without process noise the two UKFs agree to 1e-5 relative over 500 steps, in double, so they
// spread the same sigma points with the same weights. With it they differ by 7 to 13 mm, because
// mherb/kalman updates with the points its prediction moved, which leave the process noise out. The
// difference left without it, 1e-6 m at most, is the other half of the same choice: the moved points
// are F times a Cholesky factor of the previous covariance, another square root of the predicted
// covariance than its own Cholesky factor, which EmrysFilter spreads. A wrong parameter, weight, or
// spread would differ by centimeters with or without process noise.
TYPED_TEST(RangingMherbUnscentedTest, AgreeWithoutProcessNoise)
{
    auto scenario = Ranging::makeScenario<TypeParam::Dimension, TypeParam::AnchorCount>();
    scenario.accelDensity = 0;
    expectAgreement<TypeParam>(scenario, 1e-5);
}

}  // namespace
