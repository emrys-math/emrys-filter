// The precondition of the ranging benchmark against FrancoisCarouge/Kalman [benchmarks]: before any
// timing counts, EmrysFilter's EKF of emrys_ranging_kinds.h and the library's of
// carouge_ranging.h must agree in state and covariance, with the analytic Jacobian, the only one
// the library has. They agree in double and in float (cases.h, CarougeCases), so no pair is
// marked. The EmrysFilter EKF's consistency carries to the library's through the agreement;
// agreement.cpp tests it. Roadmap: the benchmarks against other libraries.

#include "carouge_ranging.h"
#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

namespace
{

template <typename C>
class RangingCarougeTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingCarougeTest, Ranging::CarougeCases);

// Both filters, driven through the same simulated run of 500 steps, agree at every step to
// agreementTolerance: 1e-9 relative to the size of the state and covariance in double, and 1e-4 in
// float. They compute the same posterior by different arithmetic: the library solves for the gain
// by a Householder QR decomposition of S with full pivoting and updates by the Joseph form, with no
// triangle mirrored. The worst difference seen in double is 9e-14, over 3000 steps too, at every
// size of the runner. In float, over the runner's 1000 steps at its six sizes, the library's EKF
// stays within 1e-5 of the double EmrysFilter EKF in the mean and 3e-7 in the covariance, where
// EmrysFilter's float EKF stays within 8e-6 and 5e-5 (carouge_benchmark.cpp prints both).
// The covariance is compared on its lower triangle, the part EmrysFilter guarantees.
TYPED_TEST(RangingCarougeTest, ImplementationsAgree)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    double const tolerance = Ranging::agreementTolerance<Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 500, 1));
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>{scenario};
    auto carouge = Ranging::CarougeRanging<D, A, Scalar>{scenario};
    for (std::size_t k = 0; k < ranges.size(); ++k)
    {
        emrys.predict();
        carouge.predict();
        emrys.update(ranges[k]);
        carouge.update(ranges[k]);
        double const meanScale = std::max(1.0, double(carouge.mean().norm()));
        double const covScale = std::max(1.0, double(carouge.covariance().norm()));
        auto const covDifference = (emrys.covarianceLower() - carouge.covariance()).template cast<double>().eval();
        ASSERT_LT((emrys.mean() - carouge.mean()).template cast<double>().norm(), tolerance * meanScale)
            << "at step " << k;
        ASSERT_LT(covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm(), tolerance * covScale)
            << "at step " << k;
    }
}

}  // namespace
