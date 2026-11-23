// The precondition of the ranging benchmark against OpenCV [benchmarks]: before any timing counts,
// EmrysFilter's EKF of emrys_ranging_kinds.h and cv::KalmanFilter run as an EKF in
// opencv_ranging.h must agree in state and covariance, with the analytic Jacobian, the only one
// OpenCV has, over the run that opencv_benchmark.cpp times, or the difference must be shown to be
// OpenCV's. They agree in double (cases.h, OpenCvAgreementCases). In float, the default type of
// cv::KalmanFilter, they do not, and the pair is timed with a mark (OpenCvDisagreementCases), which
// the second test here backs. The EmrysFilter EKF's consistency carries to OpenCV's through the
// agreement; agreement.cpp tests it. Roadmap: the benchmarks against other libraries.

#include "cases.h"
#include "disagreement.h"
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

template <typename C>
class RangingOpenCvDisagreementTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingOpenCvDisagreementTest, Ranging::OpenCvDisagreementCases);

// In float the pair is timed with a mark, since over the same 300 steps only OpenCV's EKF strays
// from the double estimate (disagreement.h): by 2e-4 to 2e-3 relative in the mean and 4e-3 to 2e-2
// in the covariance, where EmrysFilter's float EKF stays within 3e-5. The cause is its solve for
// the gain by a singular value decomposition of an innovation covariance that its unmirrored
// covariance leaves asymmetric: a copy of correct()'s arithmetic that solves by Cholesky instead
// stays within 1e-5 at every size, and so does OpenCV's own correct() to 4e-5 when the covariance
// is mirrored after it.
TYPED_TEST(RangingOpenCvDisagreementTest, OnlyOpenCvStraysInFloat)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using CvFilter = Ranging::OpenCvRanging<D, A, float>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, 300, 1);
    auto const ranges = Ranging::rangesIn<float>(run);
    auto cv = CvFilter{scenario};
    Ranging::expectOnlyTheOtherStrays(scenario, run, cv, [&](CvFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    });
}

}  // namespace
