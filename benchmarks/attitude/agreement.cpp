// The attitude benchmark's preconditions [benchmarks]: before any timing counts, the EmrysFilter and
// plain-Eigen multiplicative EKFs of emrys_attitude.h and eigen_attitude.h must agree in attitude,
// bias, and covariance to rounding, under both the analytic and the autodiff Jacobian of the
// reference vectors, and must be consistent by an NEES test. Roadmap: step 13e and Verification,
// Benchmarks.

#include "cases.h"
#include "eigen_attitude.h"
#include "emrys_attitude.h"
#include "scenario.h"

#include <EmrysFilter/Core/ChiSquared.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace
{

using Orientation::EigenAttitude;
using Orientation::EmrysAttitude;
using Orientation::Scenario;

template <typename C>
class AttitudeTest : public ::testing::Test
{};

TYPED_TEST_SUITE(AttitudeTest, Orientation::Cases);

/// The angle between the attitude of a state vector (x, y, z, w, bias) and \a attitude.
[[nodiscard]] double
angleBetween(Scenario::StateVector const & state, Eigen::Quaterniond const & attitude)
{
    auto const q = Eigen::Quaterniond{Eigen::Vector4d{state.head<4>()}};
    return Emrys::UnitQuaternion::logarithm(attitude.conjugate() * q).norm();
}

// Both implementations, driven through the same simulated run, agree at every step. They compute
// the same quantities, the library differentiating its process model where the hand-rolled filter
// writes its transition, so they differ by rounding; 1e-9, in radians for the attitude and relative
// to the size of the bias and the covariance, leaves room for rounding accumulated over 1000 steps
// and nothing more, since a modeling difference, such as a wrong sign, a transposed rotation, or a
// missed reset, is of the order of the quantity. The covariance is compared on its lower triangle,
// the part the library guarantees.
TYPED_TEST(AttitudeTest, ImplementationsAgree)
{
    auto const scenario = Orientation::makeScenario();
    auto const run = Orientation::simulate(scenario, 1000, 1);
    auto emrys = EmrysAttitude<TypeParam::Kind>{scenario};
    auto eigen = EigenAttitude<TypeParam::Kind>{scenario};
    for (std::size_t k = 0; k < run.gyro.size(); ++k)
    {
        emrys.predict(run.gyro[k]);
        eigen.predict(run.gyro[k]);
        emrys.update(run.gravity[k], run.magneticField[k]);
        eigen.update(run.gravity[k], run.magneticField[k]);
        Scenario::DifferenceMatrix const covError =
            (emrys.state().covarianceLower() - eigen.covarianceLower()).template triangularView<Eigen::Lower>();
        ASSERT_LT(angleBetween(emrys.state().mean(), eigen.attitude()), 1e-9) << "at step " << k;
        ASSERT_LT((emrys.state().mean().template tail<3>() - eigen.bias()).norm(), 1e-9 * eigen.bias().norm())
            << "at step " << k;
        ASSERT_LT(covError.norm(), 1e-9 * eigen.covarianceLower().norm()) << "at step " << k;
    }
}

// The EmrysFilter EKF is consistent: over 50 independent runs, the average NEES at each checked
// step lies in the two-sided 99.9% interval of a chi-squared with 50 times the state's six degrees
// of freedom, divided by 50 (Bar-Shalom, Li, and Kirubarajan (2001), §5.4), with the error taken
// as truth ⊟ estimate. The simulator turns the attitude by the true rate, which the filter's model
// matches to first order in the step, and the first checked step is 25 steps after a start 0.1 rad
// uncertain, by when the linearization error of the first updates has decayed. The agreement test
// carries the result to the plain-Eigen EKF.
TYPED_TEST(AttitudeTest, Consistent)
{
    constexpr int runs = 50;
    constexpr int steps = 200;
    constexpr int checkEvery = 25;
    auto const scenario = Orientation::makeScenario();
    auto const space = typename EmrysAttitude<TypeParam::Kind>::State{};
    double const dof = runs * 6.0;
    double const lower = Emrys::chiSquaredQuantile<double>(0.0005, dof) / runs;
    double const upper = Emrys::chiSquaredQuantile<double>(0.9995, dof) / runs;

    auto nees = std::array<double, steps / checkEvery>{};
    for (int r = 0; r < runs; ++r)
    {
        auto const run = Orientation::simulate(scenario, steps, 100 + static_cast<unsigned>(r));
        auto filter = EmrysAttitude<TypeParam::Kind>{scenario};
        for (int k = 0; k < steps; ++k)
        {
            filter.predict(run.gyro[k]);
            filter.update(run.gravity[k], run.magneticField[k]);
            if ((k + 1) % checkEvery == 0)
            {
                Eigen::Matrix<double, 6, 1> const error = Emrys::minus(space, run.truth[k], filter.state().mean());
                nees[(k + 1) / checkEvery - 1] +=
                    error.dot(filter.state().covarianceLower().ldlt().solve(error)) / runs;
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
