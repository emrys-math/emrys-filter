// The consistency of a Kalman filter that considers a sensor's bias (roadmap step 17v; decisions.md
// [consider-states]): consider() of Models/Policies.h through KalmanFilter of Filters/KalmanFilter.h.
// A position sensor reads with a constant bias drawn once per run from its prior. A filter that
// considers the bias is consistent on the position and velocity by the Monte Carlo NEES test of
// Models/Consistency.h, and one that leaves the bias out is overconfident. In a linear run the consider
// filter's covariance is the true error covariance of its gain, the Joseph recursion for a suboptimal
// gain written here in plain Eigen.

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Consistency.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Simulation.h>

#include <gtest/gtest.h>

#include <random>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::Composite;
using Emrys::consider;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::path;
using Emrys::reads;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<1>
{};

struct Velocity : Linear<1>
{};

struct Bias : Linear<1>
{};

using Motion = Composite<Position, Velocity>;
using State = Composite<Position, Velocity, Bias>;

using Vector1 = Eigen::Matrix<double, 1, 1>;

constexpr double timeStep = 0.5;
constexpr double biasVariance = 1.0;
constexpr double noiseVariance = 0.01;

// The motion's prior, and the state's, whose bias is independent of the motion.
GaussianParams<Motion>
makeMotionPrior()
{
    return {Motion{}, Eigen::Vector2d{0.0, 1.0}, Eigen::Matrix2d{{1.0, 0.0}, {0.0, 0.25}}};
}

GaussianParams<State>
makePrior()
{
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    cov.topLeftCorner<2, 2>() = makeMotionPrior().covarianceLower();
    cov(2, 2) = biasVariance;
    return {State{}, Eigen::Vector3d{0.0, 1.0, 0.0}, cov};
}

auto
makeStateModel()
{
    return Emrys::compose(Emrys::ConstantVelocity<Position, Velocity>{0.1}, Emrys::Constant<Bias>{});
}

auto const readingNoise = GaussianParams<Linear<1>>{Vector1{0.0}, Vector1{noiseVariance}};

// The biased reading p + b, of the whole state, and the reading p as a filter that leaves the bias
// out models it.
auto
makeBiasedSensor()
{
    return measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{block<Position>(in, x)(0) + block<Bias>(in, x)(0)};
        },
        reads(path<Position>, path<Bias>),
        additiveNoise(readingNoise)
    );
}

auto
makeUnbiasedSensor()
{
    return measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{block<Position>(in, x)};
        },
        reads(path<Position>),
        additiveNoise(readingNoise)
    );
}

// The position and velocity of a state's Gaussian, its x rows.
GaussianParams<Motion>
motionOf(GaussianParams<State> const & state)
{
    return {
        Motion{},
        Eigen::Vector2d{state.mean().head<2>()},
        Eigen::Matrix2d{Eigen::Matrix3d{state.covarianceLower()}.topLeftCorner<2, 2>()}
    };
}

}  // namespace

// Over 300 runs of 40 steps, each with its own truth and its own bias drawn from the bias's prior and
// held fixed, the consider filter's NEES on the position and velocity at the last step passes the
// two-sided test at 0.99, and the filter that leaves the bias out fails it, overconfident: its
// variance of the position is near R, an error of the bias's size far outside it.
TEST(ConsiderConsistency, ConsideringTheBiasIsConsistentAndDroppingItIsNot)
{
    auto rng = std::mt19937_64{17};
    auto const model = makeStateModel();
    auto const motionModel = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto const prior = makePrior();
    auto const initial = prior.toDensity().value();
    auto considered = Emrys::NeesMonitor<Emrys::Cumulative>{};
    auto dropped = Emrys::NeesMonitor<Emrys::Cumulative>{};
    for (int run = 0; run < 300; ++run)
    {
        Eigen::Vector3d truth = Emrys::drawSample(initial, rng);
        auto considering = Emrys::KalmanFilter{model, prior};
        auto drop = Emrys::KalmanFilter{motionModel, makeMotionPrior()};
        for (int k = 0; k < 40; ++k)
        {
            truth = Emrys::simulate(State{}, truth, model, rng, timeStep);
            Vector1 const z = Emrys::simulate(State{}, truth, makeBiasedSensor(), rng);
            considering.predict(timeStep);
            drop.predict(timeStep);
            ASSERT_TRUE(
                considering.update(observe(methods::linearized(), makeBiasedSensor(), z, consider(path<Bias>)))
            );
            ASSERT_TRUE(drop.update(observe(methods::linearized(), makeUnbiasedSensor(), z)));
            ASSERT_EQ(considering.state().mean()(2), 0.0);
        }
        Eigen::Vector2d const motion = truth.head<2>();
        considered.add(motion, motionOf(considering.state()));
        dropped.add(motion, drop.state());
    }
    EXPECT_EQ(considered.verdict(0.99), Emrys::Consistency::Consistent);
    EXPECT_EQ(dropped.verdict(0.99), Emrys::Consistency::Overconfident);
}

// The consider filter's gain is suboptimal for the whole state, since its bias row is zero, and its
// covariance is the covariance of its true error: P+ = (I - K H) P- (I - K H)^T + K R K^T with
// K = [K_x; 0], K_x the x rows of P- H^T S^-1, and P- = F P F^T + Q, in plain Eigen. F is the constant
// velocity step and Q the model's noise, read as the prediction of a Gaussian with no covariance. A
// linear run's covariances don't depend on the data, so one run is the check; the tolerance is
// rounding over 40 steps.
TEST(ConsiderConsistency, TheCovarianceIsTheTrueErrorCovarianceOfTheGain)
{
    auto const model = makeStateModel();
    auto const prior = makePrior();
    auto const still = GaussianParams<State>{State{}, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d const q = Emrys::predict(methods::linearized(), still, model, timeStep).covarianceLower();
    Eigen::Matrix3d const f{{1.0, timeStep, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    Eigen::RowVector3d const h{1.0, 0.0, 1.0};

    auto filter = Emrys::KalmanFilter{model, prior};
    Eigen::Matrix3d p = prior.covarianceLower();
    for (int k = 0; k < 40; ++k)
    {
        filter.predict(timeStep);
        ASSERT_TRUE(
            filter.update(observe(methods::linearized(), makeBiasedSensor(), Vector1{0.1 * k}, consider(path<Bias>)))
        );
        p = f * p * f.transpose() + q;
        double const s = h * p * h.transpose() + noiseVariance;
        Eigen::Vector3d gain = p * h.transpose() / s;
        gain(2) = 0.0;
        Eigen::Matrix3d const factor = Eigen::Matrix3d::Identity() - gain * h;
        p = factor * p * factor.transpose() + noiseVariance * gain * gain.transpose();
        Eigen::Matrix3d const cov = filter.state().covarianceLower().selfadjointView<Eigen::Lower>();
        EXPECT_LT((cov - p).norm(), 1e-12 * p.norm()) << k;
        EXPECT_EQ(cov(2, 2), biasVariance) << k;
    }
}
