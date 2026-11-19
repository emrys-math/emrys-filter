// A body's attitude, velocity, and position estimated from an inertial unit and landmarks seen from the body,
// started with its heading nearly unknown, by two extended Kalman filters that differ only in the state's space. In
// the first the three are one Lie group, an ExtendedPose, whose differences are steps in the body frame: the
// invariant extended Kalman filter. In the second they are three blocks of a Composite: the conventional filter.
// The process model and the prior are the same objects for both, since a model names blocks by tag and each space
// has them. The program simulates the body and the readings, runs both filters, and reports each one's errors and
// the consistency of its covariance with them. It returns nonzero if the invariant filter's estimates are not
// consistent with their covariances.

#include <EmrysFilter/EmrysFilter.h>

#include <array>
#include <cmath>
#include <numbers>
#include <print>
#include <random>

struct Attitude : Emrys::UnitQuaternion
{};

struct Velocity : Emrys::Linear<3>
{};

struct Position : Emrys::Linear<3>
{};

// The gyroscope's and the accelerometer's readings, inputs of each prediction.
struct Gyro : Emrys::Linear<3>
{};

struct Accel : Emrys::Linear<3>
{};

// The gyroscope's bias, which the filter learns.
struct Bias : Emrys::DifferenceSpace<Gyro>
{};

// The attitude, velocity, and position as one group, SE_2(3): a rotation and two vectors that turn with it. It
// holds the same ten values as Composite<Attitude, Velocity, Position> and names them by the same tags.
struct Body : Emrys::ExtendedPose<Attitude, Velocity, Position>
{};

using GroupState = Emrys::Composite<Body, Bias>;
using BlockState = Emrys::Composite<Attitude, Velocity, Position, Bias>;

// A landmark's position seen from the body.
struct Sighting : Emrys::Linear<3>
{};

Eigen::Vector3d const gravity{0.0, 0.0, -9.81};

// The inertial mechanization: the attitude turns by the gyroscope's reading less its bias, and the velocity and
// position move under the accelerometer's reading turned to the world, plus gravity. The function reads and writes
// the blocks by tag, so the same model serves the group and the Composite. Its noise is the gyroscope's and the
// accelerometer's, both in the body frame, which is the frame of the group's rows.
[[nodiscard]] auto
mechanization(double gyroDensity, double accelDensity)
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            using Emrys::block;
            Eigen::Quaternion<T> const q{Emrys::UnitQuaternion::map(block<Attitude>(in, x))};
            Emrys::DifferenceVector<Attitude, T> const turn =
                dt * Emrys::less<Gyro>(block<Gyro>(in, x), block<Bias>(in, x));
            Eigen::Matrix<T, 3, 1> const acceleration = q * block<Accel>(in, x).template cast<T>().eval() + gravity;
            auto y = Emrys::zeros<T>(out);
            block<Attitude>(out, y) = Emrys::plus<Attitude>(block<Attitude>(in, x), turn);
            block<Velocity>(out, y) = block<Velocity>(in, x) + dt * acceleration;
            block<Position>(out, y) =
                block<Position>(in, x) + dt * block<Velocity>(in, x) + 0.5 * dt * dt * acceleration;
            return y;
        },
        Emrys::reads(
            Emrys::path<Attitude>,
            Emrys::path<Velocity>,
            Emrys::path<Position>,
            Emrys::path<Bias>,
            Emrys::path<Gyro>,
            Emrys::path<Accel>
        ),
        Emrys::writes(Emrys::path<Attitude>, Emrys::path<Velocity>, Emrys::path<Position>),
        Emrys::continuousNoise(Emrys::on<Attitude>(gyroDensity), Emrys::on<Velocity>(accelDensity))
    );
}

// A landmark seen from the body, R^T (l - p), written as a function for the conventional filter.
[[nodiscard]] auto
sightingByFunction(Eigen::Vector3d const & landmark, double variance)
{
    return Emrys::measurementModel(
        [landmark]<typename T>(auto const & in, auto const & x) {
            Eigen::Quaternion<T> const q{Emrys::UnitQuaternion::map(Emrys::block<Attitude>(in, x))};
            return Emrys::ValueVector<Sighting, T>{q.conjugate() * (landmark - Emrys::block<Position>(in, x)).eval()};
        },
        Emrys::reads(Emrys::path<Attitude>, Emrys::path<Position>),
        Emrys::outputSpace(Sighting{}),
        Emrys::additiveNoise(variance * Emrys::DifferenceMatrix<Sighting>::Identity())
    );
}

int
main()
{
    double const dt = 0.01;  // a 100 Hz inertial unit
    int const steps = 1500;  // 15 seconds a run
    int const sightEvery = 10;
    int const runs = 40;

    double const gyroDensity = 1e-5;
    double const accelDensity = 1e-3;
    double const biasDensity = 1e-8;
    double const sightingVariance = 0.01;
    auto const motion =
        Emrys::blockwise(mechanization(gyroDensity, accelDensity), Emrys::RandomWalk<Bias>{biasDensity});

    auto const landmarks =
        std::array{Eigen::Vector3d{10.0, 0.0, 2.0}, Eigen::Vector3d{-4.0, 9.0, -1.0}, Eigen::Vector3d{-3.0, -8.0, 4.0}};

    // The prior: the heading known to a radian, the tilt to 0.2 rad, the velocity to 1 m/s, and the position to
    // 2 m. The same thirteen values and twelve variances serve both spaces; in the group the velocity's and the
    // position's are of steps in the body frame, which for equal variances on the three axes is the same.
    Emrys::ValueVector<GroupState> mean = Emrys::ValueVector<GroupState>::Zero();
    mean.head<4>() = Emrys::UnitQuaternion::identity();
    auto const variances =
        Emrys::DifferenceVector<GroupState>{{0.04, 0.04, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 1e-4, 1e-4, 1e-4}};
    auto const groupPrior = Emrys::GaussianParams<GroupState>{mean, variances.asDiagonal()};
    auto const blockPrior = Emrys::GaussianParams<BlockState>{mean, variances.asDiagonal()};

    auto rng = std::mt19937_64{24};
    auto groupNees = Emrys::NeesMonitor{runs};
    auto blockNees = Emrys::NeesMonitor{runs};
    double groupSquaredError = 0;
    double blockSquaredError = 0;
    for (int run = 0; run < runs; ++run)
    {
        auto group = Emrys::KalmanFilter{motion, groupPrior};
        auto blocks = Emrys::KalmanFilter{motion, blockPrior};
        Emrys::ValueVector<GroupState> truth = Emrys::drawSamples(groupPrior.toDensity().value(), 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            // The readings a body would give while it turns and sways: the gyroscope's as given, and the
            // accelerometer's the specific force of a gentle acceleration in the world.
            double const t = k * dt;
            auto const gyro = Emrys::ValueVector<Gyro>{0.3 * std::sin(0.7 * t), 0.2 * std::cos(0.5 * t), 0.4};
            Eigen::Quaterniond const attitude{Eigen::Vector4d{truth.head<4>()}};
            Eigen::Vector3d const worldAcceleration{std::cos(0.4 * t), std::sin(0.6 * t), 0.3 * std::cos(0.9 * t)};
            Emrys::ValueVector<Accel> const accel = attitude.conjugate() * (worldAcceleration - gravity);
            truth =
                Emrys::simulate<GroupState>(truth, motion, rng, dt, Emrys::sure<Gyro>(gyro), Emrys::sure<Accel>(accel));
            group.predict(dt, Emrys::sure<Gyro>(gyro), Emrys::sure<Accel>(accel));
            blocks.predict(dt, Emrys::sure<Gyro>(gyro), Emrys::sure<Accel>(accel));

            if (k % sightEvery == 0)
            {
                for (Eigen::Vector3d const & landmark : landmarks)
                {
                    // The group's measurement names its form, a point fixed in the world seen from the body, so
                    // its Jacobian is formed, not differentiated, and its update carries the covariance by the
                    // adjoint: the right-invariant filter's update. No method and no carry are named.
                    auto const sight = Emrys::observedInBody<Body>(
                        landmark,
                        std::tuple{Emrys::on<Position>(1.0)},
                        Emrys::outputSpace(Sighting{}),
                        Emrys::additiveNoise(sightingVariance * Emrys::DifferenceMatrix<Sighting>::Identity())
                    );
                    auto const byFunction = sightingByFunction(landmark, sightingVariance);
                    Emrys::ValueVector<Sighting> const z = Emrys::simulate<GroupState>(truth, sight, rng);
                    auto const groupUpdated = group.update(Emrys::observe(sight, z));
                    auto const blocksUpdated =
                        blocks.update(Emrys::observe(Emrys::methods::linearized(), byFunction, z));
                    if (!groupUpdated || !blocksUpdated)
                    {
                        std::println("run {}, step {}: an update failed", run, k);
                        return 1;
                    }
                }
            }
        }
        groupNees.add(truth, group.state());
        blockNees.add(truth, blocks.state());
        Emrys::DifferenceVector<GroupState> const groupError = Emrys::minus<GroupState>(truth, group.state().mean());
        Emrys::DifferenceVector<BlockState> const blockError = Emrys::minus<BlockState>(truth, blocks.state().mean());
        groupSquaredError += Emrys::diffBlock<GroupState, Emrys::TagPath<Body, Attitude>>(groupError).squaredNorm();
        blockSquaredError += Emrys::diffBlock<BlockState, Attitude>(blockError).squaredNorm();
    }

    double const degrees = 180.0 / std::numbers::pi;
    std::println(
        "RMS attitude error after {:g} s: group {:.3f} deg, blocks {:.3f} deg",
        steps * dt,
        degrees * std::sqrt(groupSquaredError / runs),
        degrees * std::sqrt(blockSquaredError / runs)
    );
    std::println(
        "average NEES per degree of freedom (1 is ideal): group {:.2f}, blocks {:.2f}",
        groupNees.average().value(),
        blockNees.average().value()
    );
    bool const consistent = groupNees.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("the invariant filter is {}", consistent ? "consistent" : "NOT consistent");
    return consistent ? 0 : 1;
}
