// The Kalman filter and smoother over a state holding a unit quaternion, the space of
// tests/unequal_dimensions.h whose values have more coordinates than its differences (step 14,
// [smoothing]), with the update's and the smoother's covariance resets (step 13e, [covariance-reset]), and a fixed-step
// model on it, which caches no transition [fixed-step-models].

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

#include <gtest/gtest.h>

#include "unequal_dimensions.h"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <array>
#include <cstddef>

using namespace EmrysTest::UnequalDimensions;

using Emrys::compose;
using Emrys::KalmanFilter;

namespace methods = Emrys::methods;

namespace
{

// Random-walk attitude noise on the rotation vector of the quaternion at Path, which does not move.
template <typename Path>
auto
attitudeWalk(double spectralDensity)
{
    return Emrys::processModel(
        [](auto const &, auto const &, auto const & x, auto) {
            return x;
        },
        Emrys::reads(Emrys::path<Path>),
        Emrys::writes(Emrys::path<Path>),
        Emrys::continuousNoise(Emrys::on<Path>(spectralDensity))
    );
}

// A KalmanFilter over attitude, position, and velocity predicts with a random walk on the rotation
// vector and updates from a measured direction as the hand EKF on the seven-coordinate tangent
// space, applying the step to the quaternion by plus() and resetting the covariance by
// blockdiag(J_r(step), I) [covariance-reset].
TEST(UnequalDimensions, KalmanFilterPredictsAndUpdatesAnAttitude)
{
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << turned, 10.0, -5.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Zero();
    cov.topLeftCorner<5, 5>() = poseCov();
    cov.bottomRightCorner<2, 2>() = 0.25 * Eigen::Matrix2d::Identity();
    auto const initial = GaussianParams<Motion>{mean, cov};

    auto filter = KalmanFilter{
        compose(attitudeWalk<Orientation>(0.01), Emrys::ConstantVelocity<Position, Velocity>{0.1}),
        initial
    };
    filter.predict(1.0);
    auto const & predicted = filter.state();
    EXPECT_TRUE(predicted.mean().head<4>().isApprox(turned));
    EXPECT_TRUE(predicted.mean<Position>().isApprox(Eigen::Vector2d{11.0, -4.5}));
    Eigen::Matrix3d const attitudeCov = poseCov().topLeftCorner<3, 3>() + 0.01 * Eigen::Matrix3d::Identity();
    EXPECT_TRUE((predicted.covarianceLower().topLeftCorner<3, 3>().isApprox(attitudeCov)));
    // Position gains its velocity's variance and the white-noise acceleration's.
    EXPECT_NEAR(predicted.covarianceLower()(3, 3), 1.0 + 0.25 + 0.1 / 3, 1e-12);

    // A measured direction updates the attitude through its rotation vector.
    auto const forward = Eigen::Vector3d{1.0, 0.0, 0.0};
    Eigen::Vector3d const observed = rotate(multiply(turned, exponential(Eigen::Vector3d{0.0, 0.05, 0.05})), forward);
    Eigen::Matrix3d const noise = 0.001 * Eigen::Matrix3d::Identity();
    auto const directionModel = Emrys::measurementModel(
        [&]<typename T>(auto const & in, auto const &, auto const & x) {
            return rotate(Eigen::Matrix<T, 4, 1>{Emrys::block<Orientation>(in, x)}, forward).eval();
        },
        Emrys::reads(Emrys::path<Orientation>),
        Emrys::additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), noise})
    );
    auto const direction = observe(methods::linearized(), directionModel, observed);

    // The extended Kalman filter by hand, on the five-coordinate tangent space.
    GaussianParams<Motion> const prior = filter.state();
    Eigen::Matrix<double, 3, 7> h = Eigen::Matrix<double, 3, 7>::Zero();
    h.leftCols<3>() = rotationMatrix(turned) * skew(forward);  // of the innovation z - R(q exp δ) v
    Eigen::Vector3d const innovation = observed - rotate(turned, forward);
    Eigen::Matrix3d const s = h * prior.covarianceLower() * h.transpose() + noise;
    Eigen::Matrix<double, 7, 3> const gain = prior.covarianceLower() * h.transpose() * s.inverse();
    Eigen::Matrix<double, 7, 1> const step = -gain * innovation;
    Eigen::Matrix<double, 7, 7> reset = Eigen::Matrix<double, 7, 7>::Identity();
    reset.topLeftCorner<3, 3>() = rightJacobian(Eigen::Vector3d{step.head<3>()});
    Eigen::Matrix<double, 7, 7> const posteriorCov =
        reset * (prior.covarianceLower() - gain * h * prior.covarianceLower()) * reset.transpose();

    auto const result = filter.update(direction);
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->accepted);
    auto const & posterior = filter.state();
    EXPECT_NEAR(posterior.mean().head<4>().norm(), 1.0, 1e-14);
    EXPECT_TRUE(posterior.mean().head<4>().isApprox(multiply(turned, exponential(step.head<3>())), 1e-10));
    EXPECT_TRUE(posterior.mean().tail<4>().isApprox(prior.mean().tail<4>() + step.tail<4>(), 1e-10));
    EXPECT_TRUE(posterior.covarianceLower().isApprox(posteriorCov, 1e-10));
    EXPECT_EQ(result->innovationSize(), 3);
}

// On a state with a quaternion, whose attitude block gives no transition() since its Jacobian
// depends on the point, a fixed-step model caches only the noise [fixed-step-models], and its filter predicts
// exactly as one predicting over the step does.
TEST(UnequalDimensions, AFixedStepModelCachesNoTransitionOnAQuaternion)
{
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << turned, 10.0, -5.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Zero();
    cov.topLeftCorner<5, 5>() = poseCov();
    cov.bottomRightCorner<2, 2>() = 0.25 * Eigen::Matrix2d::Identity();
    auto const initial = GaussianParams<Motion>{mean, cov};
    auto const model = compose(attitudeWalk<Orientation>(0.01), Emrys::ConstantVelocity<Position, Velocity>{0.1});

    auto const fixedModel = Emrys::FixedStepModel{model, Motion{}, 1.0};
    static_assert(!decltype(fixedModel)::CachesTransition);
    auto fixed = KalmanFilter{fixedModel, initial};
    auto continuous = KalmanFilter{model, initial};
    fixed.predict();
    continuous.predict(1.0);
    EXPECT_EQ(fixed.state().mean(), continuous.state().mean());
    EXPECT_EQ(fixed.state().covarianceLower(), continuous.state().covarianceLower());
}

// The Kalman smoother over an attitude seen in two alternating directions, noise-free, keeps each
// smoothed quaternion unit and each covariance over the seven difference coordinates, never
// smooths a covariance above its filtered one once both are in the tangent space at the filtered
// mean (the smoothed one carried back by the inverse of the reset Jacobian of the smoothing step,
// [covariance-reset]), and pulls the first attitude, which saw only one direction, closer to the truth than the
// filter left it.
TEST(UnequalDimensions, KalmanSmootherStepsInTheTangentSpace)
{
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << turned, 10.0, -5.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Zero();
    cov.topLeftCorner<5, 5>() = poseCov();
    cov.bottomRightCorner<2, 2>() = 0.25 * Eigen::Matrix2d::Identity();
    auto filter = KalmanFilter{
        compose(attitudeWalk<Orientation>(0.001), Emrys::ConstantVelocity<Position, Velocity>{0.1}),
        GaussianParams<Motion>{mean, cov}
    };

    Quaternion<double> const truth = multiply(turned, exponential(Eigen::Vector3d{0.1, -0.05, 0.08}));
    Eigen::Matrix3d const noise = 0.001 * Eigen::Matrix3d::Identity();
    auto const direction = [&](Eigen::Vector3d const & body) {
        Eigen::Vector3d const observed = rotate(truth, body);
        auto const model = Emrys::measurementModel(
            [body]<typename T>(auto const & in, auto const &, auto const & x) {
                return rotate(Eigen::Matrix<T, 4, 1>{Emrys::block<Orientation>(in, x)}, body).eval();
            },
            Emrys::reads(Emrys::path<Orientation>),
            Emrys::additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), noise})
        );
        return observe(methods::linearized(), model, observed);
    };
    auto const bodies = std::array<Eigen::Vector3d, 2>{Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ()};

    auto history = Emrys::GaussianHistory<Motion>{};
    EXPECT_TRUE(filter.update(direction(bodies[0])));
    for (int k = 1; k <= 6; ++k)
    {
        history.push(filter.predictJoint(1.0));
        EXPECT_TRUE(filter.update(direction(bodies[std::size_t(k % 2)])));
    }
    history.smooth(filter.state());

    static_assert(GaussianParams<Motion>::DifferenceDimensionAtCompileTime == 7);
    static_assert(decltype(history)::JointType::MatrixType::RowsAtCompileTime == 7);
    for (auto const & step : history)
    {
        EXPECT_NEAR(step.smoothed.mean().head<4>().norm(), 1.0, 1e-14);
        Eigen::Matrix<double, 7, 1> const moved = minus(Motion{}, step.smoothed.mean(), step.joint.filtered.mean());
        Eigen::Matrix<double, 7, 7> const back =
            Emrys::resetJacobian(Motion{}, step.joint.filtered.mean(), moved).inverse();
        Eigen::Matrix<double, 7, 7> const reduction =
            step.joint.filtered.covarianceLower() - back * step.smoothed.covarianceLower() * back.transpose();
        auto const eigen = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 7, 7>>{reduction};
        EXPECT_GE(eigen.eigenvalues().minCoeff(), -1e-12);
    }
    auto const attitudeError = [&](GaussianParams<Motion> const & g) {
        return minus(Orientation{}, Eigen::Vector4d{g.mean().head<4>()}, truth).norm();
    };
    double const filtered = attitudeError(history.front().joint.filtered);
    double const smoothed = attitudeError(history.front().smoothed);
    EXPECT_LT(smoothed, 0.5 * filtered);
}

}  // namespace
