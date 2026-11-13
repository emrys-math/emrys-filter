// The ranging benchmark's preconditions [benchmarks]: before any timing counts, the EmrysFilter and
// plain-Eigen EKFs of emrys_ranging.h and eigen_ranging.h must agree in state and covariance to
// rounding, under both the analytic and the autodiff Jacobian and in double and in float, and must
// be consistent by an NEES test; a float filter must also stay within float's precision of the
// double one. Covers the planar and spatial constant-velocity cases. Roadmap: Verification,
// Benchmarks.

#include "cases.h"
#include "eigen_ranging.h"
#include "emrys_ranging.h"
#include "scenario.h"

#include <EmrysFilter/Core/ChiSquared.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace
{

using Ranging::EigenRanging;
using Ranging::EmrysRanging;

template <typename C>
class RangingTest : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingTest, Ranging::Cases);

/// The norm of the lower triangle of \a difference, a difference of covariances, cast to double:
/// the part the library guarantees.
template <typename Derived>
[[nodiscard]] double
lowerNorm(Eigen::MatrixBase<Derived> const & difference)
{
    return difference.template cast<double>().template triangularView<Eigen::Lower>().toDenseMatrix().norm();
}

// Both implementations, driven through the same simulated run, agree at every step. They compute
// the same products in nearly the same order, so they differ by rounding; agreementTolerance, 1e-9
// relative to the size of the state and covariance in double and 1e-4 in float, leaves room for
// rounding accumulated over 500 steps and nothing more, since a modeling difference, such as a
// wrong sign or a missed term, is of order one. The worst differences seen are 1e-14 in double and
// 4e-6 in float. The covariance is compared on its lower triangle.
TYPED_TEST(RangingTest, ImplementationsAgree)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 500, 1));
    double const tolerance = Ranging::agreementTolerance<Scalar>;
    auto emrys = EmrysRanging<D, A, TypeParam::Kind, Scalar>{scenario};
    auto eigen = EigenRanging<D, A, TypeParam::Kind, Scalar>{scenario};
    for (std::size_t k = 0; k < ranges.size(); ++k)
    {
        emrys.predict();
        eigen.predict();
        emrys.update(ranges[k]);
        eigen.update(ranges[k]);
        double const meanScale = std::max(1.0, double(eigen.mean().norm()));
        double const covScale = std::max(1.0, double(eigen.covarianceLower().norm()));
        ASSERT_LT((emrys.state().mean() - eigen.mean()).template cast<double>().norm(), tolerance * meanScale)
            << "at step " << k;
        ASSERT_LT(lowerNorm(emrys.state().covarianceLower() - eigen.covarianceLower()), tolerance * covScale)
            << "at step " << k;
    }
}

// A float EKF stays within float's precision of the double one over the same run: 1e-4 relative to
// the size of the state and covariance at every step, where the worst seen is 1e-5, about 150
// times float's epsilon after 500 steps; a modeling or conditioning failure is of order one. The
// agreement test carries the result to the plain-Eigen EKF, which mirrors its covariance's lower
// triangle as the library does: without the mirror its two triangles drift apart, and it strays
// from double ten times as far.
TYPED_TEST(RangingTest, FloatTracksDouble)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    if constexpr (std::is_same_v<Scalar, double>)
    {
        GTEST_SKIP() << "a double filter is its own reference";
    }
    else
    {
        auto const scenario = Ranging::makeScenario<D, A>();
        auto const run = Ranging::simulate(scenario, 500, 1);
        auto const ranges = Ranging::rangesIn<Scalar>(run);
        auto filter = EmrysRanging<D, A, TypeParam::Kind, Scalar>{scenario};
        auto reference = EmrysRanging<D, A, TypeParam::Kind>{scenario};
        for (std::size_t k = 0; k < ranges.size(); ++k)
        {
            filter.predict();
            reference.predict();
            filter.update(ranges[k]);
            reference.update(run.ranges[k]);
            double const meanScale = std::max(1.0, reference.state().mean().norm());
            double const covScale = std::max(1.0, reference.state().covarianceLower().norm());
            ASSERT_LT(
                (filter.state().mean().template cast<double>() - reference.state().mean()).norm(),
                1e-4 * meanScale
            ) << "at step "
              << k;
            ASSERT_LT(
                lowerNorm(
                    filter.state().covarianceLower().template cast<double>() - reference.state().covarianceLower()
                ),
                1e-4 * covScale
            ) << "at step "
              << k;
        }
    }
}

// The EmrysFilter EKF is consistent: over 50 independent runs, the average NEES at each checked
// step lies in the two-sided 99.9% interval of a chi-squared with 50 times the state's degrees of
// freedom, divided by 50 (Bar-Shalom, Li, and Kirubarajan (2001), §5.4). The runs are independent
// and the simulator draws the truth from the filter's own model, so only a wrong filter, or the
// 0.1% chance per checked step, fails. The agreement test carries the result to the plain-Eigen
// EKF.
TYPED_TEST(RangingTest, Consistent)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    constexpr int runs = 50;
    constexpr int steps = 100;
    constexpr int checkEvery = 25;
    auto const scenario = Ranging::makeScenario<D, A>();
    double const dof = runs * 2.0 * D;
    double const lower = Emrys::chiSquaredQuantile<double>(0.0005, dof) / runs;
    double const upper = Emrys::chiSquaredQuantile<double>(0.9995, dof) / runs;

    auto nees = std::array<double, steps / checkEvery>{};
    for (int r = 0; r < runs; ++r)
    {
        auto const run = Ranging::simulate(scenario, steps, 100 + r);
        auto const ranges = Ranging::rangesIn<Scalar>(run);
        auto filter = EmrysRanging<D, A, TypeParam::Kind, Scalar>{scenario};
        for (int k = 0; k < steps; ++k)
        {
            filter.predict();
            filter.update(ranges[k]);
            if ((k + 1) % checkEvery == 0)
            {
                auto const error = (run.truth[k] - filter.state().mean().template cast<double>()).eval();
                auto const cov = filter.state().covarianceLower().template cast<double>().eval();
                nees[(k + 1) / checkEvery - 1] += error.dot(cov.ldlt().solve(error)) / runs;
            }
        }
    }
    for (std::size_t c = 0; c < nees.size(); ++c)
    {
        EXPECT_GT(nees[c], lower) << "at step " << (c + 1) * checkEvery;
        EXPECT_LT(nees[c], upper) << "at step " << (c + 1) * checkEvery;
    }
}

}  // namespace
