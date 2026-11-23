// The precondition of the ranging benchmark against OpenCV [benchmarks]: before any timing counts,
// EmrysFilter's EKF of emrys_ranging_kinds.h and cv::KalmanFilter run as an EKF in
// opencv_ranging.h must agree in state and covariance, with the analytic Jacobian, the only one
// OpenCV has, over the run that opencv_benchmark.cpp times. They agree in double, and only that is
// timed (cases.h, OpenCvAgreementCases). The EmrysFilter EKF's consistency carries to OpenCV's
// through the agreement; agreement.cpp tests it. Roadmap: the benchmarks against other libraries.

#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "opencv_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

namespace
{

template <typename C>
class RangingOpenCvTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingOpenCvTest, Ranging::OpenCvAgreementCases);

// Both filters, driven through the same simulated run of 300 steps, the timed run, agree at every
// step to agreementTolerance, 1e-9 relative to the size of the state and covariance in double. They
// compute the same posterior by different arithmetic: OpenCV solves for the gain by a singular value
// decomposition of S and updates by P - K (H P), with no triangle mirrored. The covariance is
// compared on its lower triangle, the part EmrysFilter guarantees.
//
// The run is 300 steps, not the 500 of the other pairs, because OpenCV's agreement does not last:
// the asymmetry that rounding leaves in its covariance grows, from 5e-13 to 1e-11 relative at step
// 100 to 1e-3 to 7e-3 at step 1000, and the pair first differs by more than 1e-9 between steps 400
// and 504 of each case. By step 2000 its spatial estimates have diverged, while EmrysFilter's stays
// within 2e-15 of its own estimate in long double; with OpenCV's covariance mirrored after each
// update, the two agree to 2e-15 over 3000 steps. opencv_benchmark.cpp prints the long run.
TYPED_TEST(RangingOpenCvTest, ImplementationsAgree)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    double const tolerance = Ranging::agreementTolerance<Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 300, 1));
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>{scenario};
    auto cv = Ranging::OpenCvRanging<D, A, Scalar>{scenario};
    for (std::size_t k = 0; k < ranges.size(); ++k)
    {
        emrys.predict();
        cv.predict();
        emrys.update(ranges[k]);
        cv.update(ranges[k]);
        double const meanScale = std::max(1.0, double(cv.mean().norm()));
        double const covScale = std::max(1.0, double(cv.covariance().norm()));
        auto const covDifference = (emrys.covarianceLower() - cv.covariance()).template cast<double>().eval();
        ASSERT_LT((emrys.mean() - cv.mean()).template cast<double>().norm(), tolerance * meanScale) << "at step " << k;
        ASSERT_LT(covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm(), tolerance * covScale)
            << "at step " << k;
    }
}

}  // namespace
