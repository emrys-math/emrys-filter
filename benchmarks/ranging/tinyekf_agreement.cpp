// The precondition of the ranging benchmark against TinyEKF [benchmarks]: before any timing counts,
// EmrysFilter's EKF of emrys_ranging_kinds.h and TinyEKF's of tinyekf_ranging.h must agree in
// state and covariance, with the analytic Jacobian, the only one TinyEKF has. They agree in double,
// and only that is timed (cases.h, TinyEkfAgreementCases). The EmrysFilter EKF's consistency carries
// to TinyEKF's through the agreement; agreement.cpp tests it. Roadmap: the benchmarks against other
// libraries.

#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"
#include "tinyekf_ranging.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

namespace
{

template <typename C>
class RangingTinyEkfTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingTinyEkfTest, Ranging::TinyEkfAgreementCases);

// Both filters, driven through the same simulated run of 500 steps, agree at every step to
// agreementTolerance, 1e-9 relative to the size of the state and covariance in double, as the
// plain-Eigen EKF does in agreement.cpp. They compute the same posterior by different arithmetic:
// TinyEKF forms the gain by an explicit inverse of S from its Cholesky factor and updates by
// (I - G H) P, with no triangle mirrored. The worst difference seen is 6e-11; a modeling difference
// is of order one. The covariance is compared on its lower triangle, the part EmrysFilter
// guarantees.
//
// In float the same three choices cost TinyEKF its accuracy over the first steps, while the prior
// is wide. A copy of its header with all three changed, the gain solved by the Cholesky factor, the
// covariance updated as P - G (H P), and its triangles mirrored, agrees to 4e-5; the solve alone,
// the solve and the form, or the mirroring alone do not.
TYPED_TEST(RangingTinyEkfTest, ImplementationsAgree)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    double const tolerance = Ranging::agreementTolerance<Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 500, 1));
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>{scenario};
    auto tiny = Ranging::TinyEkfRanging<D, A, Scalar>{scenario};
    for (std::size_t k = 0; k < ranges.size(); ++k)
    {
        emrys.predict();
        tiny.predict();
        emrys.update(ranges[k]);
        tiny.update(ranges[k]);
        double const meanScale = std::max(1.0, double(tiny.mean().norm()));
        double const covScale = std::max(1.0, double(tiny.covariance().norm()));
        auto const covDifference = (emrys.covarianceLower() - tiny.covariance()).template cast<double>().eval();
        ASSERT_LT((emrys.mean() - tiny.mean()).template cast<double>().norm(), tolerance * meanScale)
            << "at step " << k;
        ASSERT_LT(covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm(), tolerance * covScale)
            << "at step " << k;
    }
}

}  // namespace
