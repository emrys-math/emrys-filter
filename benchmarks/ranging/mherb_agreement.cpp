// The precondition of the ranging benchmark against mherb/kalman [benchmarks]: before any timing
// counts, the EmrysFilter EKF of emrys_ranging.h and mherb/kalman's of mherb_ranging.h must agree
// in state and covariance, with the analytic Jacobian, the only one mherb/kalman has. They agree in
// double only, so only double is timed (cases.h says why float is not). The EmrysFilter EKF's
// consistency, tested in agreement.cpp, carries to mherb/kalman's through this agreement. Roadmap:
// the benchmarks against other libraries.

#include "cases.h"
#include "emrys_ranging.h"
#include "mherb_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

namespace
{

using Ranging::EmrysRanging;
using Ranging::MherbRanging;

template <typename C>
class RangingMherbTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingMherbTest, Ranging::MherbAgreementCases);

// Both filters, driven through the same simulated run, agree at every step to agreementTolerance,
// 1e-9 relative to the size of the state and covariance in double, as the plain-Eigen EKF does in
// agreement.cpp. They compute the same posterior by different arithmetic, mherb/kalman by an
// explicit inverse of S and the update P - K H P without mirroring a triangle; the worst
// difference seen is 9e-10 m in the mean, in 2D with 4 anchors, and a modeling difference is of
// order one. The covariance is compared on its lower triangle, the part the library guarantees.
TYPED_TEST(RangingMherbTest, ImplementationsAgree)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 500, 1));
    auto const mherbRanges = MherbRanging<D, A, Scalar>::rangesOf(ranges);
    double const tolerance = Ranging::agreementTolerance<Scalar>;
    auto emrys = EmrysRanging<D, A, TypeParam::Kind, Scalar>{scenario};
    auto mherb = MherbRanging<D, A, Scalar>{scenario};
    for (std::size_t k = 0; k < ranges.size(); ++k)
    {
        emrys.predict();
        mherb.predict();
        emrys.update(ranges[k]);
        mherb.update(mherbRanges[k]);
        double const meanScale = std::max(1.0, double(mherb.mean().norm()));
        double const covScale = std::max(1.0, double(mherb.covariance().norm()));
        auto const covDifference =
            (emrys.state().covarianceLower() - mherb.covariance()).template cast<double>().eval();
        ASSERT_LT((emrys.state().mean() - mherb.mean()).template cast<double>().norm(), tolerance * meanScale)
            << "at step " << k;
        ASSERT_LT(covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm(), tolerance * covScale)
            << "at step " << k;
    }
}

}  // namespace
