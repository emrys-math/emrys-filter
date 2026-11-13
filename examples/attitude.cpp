// The attitude of a rotating body, estimated by an extended Kalman filter over a unit quaternion
// and the gyroscope's bias. The gyroscope's reading turns the attitude in the prediction, and an
// accelerometer that sees gravity and a magnetometer that sees the Earth's field correct it. The
// program simulates the body and the readings from the same models the filter assumes, runs the
// filter, and checks that its estimates are consistent with their covariances: the normalized
// estimation error squared (NEES) against the true state, and the normalized innovation squared
// (NIS) of the readings. It returns nonzero if either check fails.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <print>
#include <numbers>
#include <random>

// The attitude takes vectors in the body frame to the world frame. It is stored as Eigen stores a
// quaternion, (x, y, z, w), but its differences are a 3-vector: a small rotation about the body's
// axes, in radians. So the state has 7 values and 6 differences, and its covariance is 6 by 6.
struct Attitude : Emrys::UnitQuaternion
{};

// The gyroscope's reading, in radians per second, an input of each prediction.
struct Gyro : Emrys::Linear<3>
{};

// The gyroscope's bias, in radians per second, which the reading includes and the filter learns:
// an error in the reading, so a difference of readings, declared in their difference space.
struct Bias : Emrys::DifferenceSpace<Gyro>
{};

using State = Emrys::Composite<Attitude, Bias>;

// A reading of a reference vector in the body frame, the space of the accelerometer's and the
// magnetometer's measurements.
using Reading = Emrys::Linear<3>;

// A process model that turns the attitude by the gyroscope's reading less the bias over the step.
// The library has no gyroscope model built in, so the user writes one: the function reads the
// attitude, the bias, and the reading, named in reads(), and writes the attitude, named in
// writes(); continuousNoise() adds the reading's white noise, integrated over the step, to the
// attitude's covariance. The function computes in T, the scalar of x, which the filter supplies, so
// the Linearized prediction finds its Jacobian by automatic differentiation. less() takes the bias
// from the reading. The rate times the step is a rotation vector in the body frame, a difference of
// attitudes, and plus() on the attitude applies it, q Exp(turn).
[[nodiscard]] auto
gyroIntegration(double noiseDensity)  // spectral density of the gyroscope's white noise, rad^2/s
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto const rate = Emrys::less<Gyro>(Emrys::block<Gyro>(in, x), Emrys::block<Bias>(in, x));
            Emrys::DifferenceVector<Attitude, T> const turn = rate * dt;
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Attitude>(out, y) = Emrys::plus<Attitude>(Emrys::block<Attitude>(in, x), turn);
            return y;
        },
        Emrys::reads(Emrys::path<Attitude>, Emrys::path<Bias>, Emrys::path<Gyro>),
        Emrys::writes(Emrys::path<Attitude>),
        Emrys::continuousNoise(Emrys::on<Attitude>(noiseDensity))
    );
}

// The model of a reading of a fixed world-frame unit vector in the body frame, R(q)^T r, with noise
// of standard deviation 0.02 on each coordinate.
[[nodiscard]] auto
referenceModel(Eigen::Vector3d const & reference)
{
    auto const reading = [reference]<typename T>(auto const & in, auto const &, auto const & x) {
        Emrys::ValueVector<Attitude, T> const q = Emrys::block<Attitude>(in, x);
        Emrys::ValueVector<Reading, T> const v = Emrys::UnitQuaternion::map(q).conjugate() * reference;
        return v;
    };
    double const readingVariance = 4e-4;
    return Emrys::measurementModel(
        reading,
        Emrys::reads(Emrys::path<Attitude>),
        Emrys::additiveNoise(readingVariance * Emrys::DifferenceMatrix<Reading>::Identity())
    );
}

int
main()
{
    double const dt = 0.01;  // seconds between readings: a 100 Hz inertial unit
    int const steps = 2000;  // 20 seconds a run
    int const runs = 50;

    // The gyroscope's angle random walk is 1e-6 rad^2/s (0.06 degrees per root second), and its
    // bias wanders as a random walk of 1e-8 rad^2/s^3.
    double const gyroDensity = 1e-6;
    double const biasDensity = 1e-8;
    auto const motion = Emrys::compose(gyroIntegration(gyroDensity), Emrys::RandomWalk<Bias>{biasDensity});

    // Gravity points along the world's vertical z axis; the magnetic field dips 60 degrees below the
    // horizontal. Gravity alone leaves the heading unobserved, which the magnetometer supplies.
    auto const accelerometer = referenceModel(Eigen::Vector3d::UnitZ());
    auto const magnetometer = referenceModel(Eigen::Vector3d{0.5, 0.0, -std::sqrt(0.75)});

    // The prior: an attitude known to about 6 degrees on each axis, and a bias of zero known to
    // 0.01 rad/s (0.6 degrees per second).
    // The mean's attitude is a rotation by the vector (0.3, -0.2, 1.0) radians.
    Eigen::Quaterniond const attitude =
        Emrys::UnitQuaternion::exponential(Emrys::DifferenceVector<Attitude>{0.3, -0.2, 1.0});
    Emrys::ValueVector<State> const mean = (Emrys::ValueVector<State>{} << attitude.coeffs(), 0.0, 0.0, 0.0).finished();
    auto const variances = Emrys::DifferenceVector<State>{{0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4}};
    auto const prior = Emrys::GaussianParams<State>{mean, variances.asDiagonal()};

    // The gyroscope and the reference vectors correct the attitude slowly, so its error and the
    // bias's stay correlated for seconds, and the NEES of one run's successive estimates are far
    // from independent. The NEES test is therefore the Monte Carlo one: the final estimates of
    // independent runs. The innovations of a consistent filter are independent, so the NIS test
    // takes every update.
    auto rng = std::mt19937_64{24};
    auto nees = Emrys::NeesMonitor{runs};
    auto nis = Emrys::NisMonitor{2 * runs * steps};
    double squaredAttitudeError = 0;
    for (int run = 0; run < runs; ++run)
    {
        // The true state is drawn from the prior. The prior's covariance is positive definite by
        // construction, so toDensity() can't fail here.
        auto filter = Emrys::KalmanFilter{motion, prior};
        Emrys::ValueVector<State> truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            // The gyroscope's reading over the step, in rad/s about the body's axes. The simulation
            // takes the readings as given and draws the turn the body made under them, as the
            // filter's model says: the reading less the bias, plus the reading's noise.
            double const t = k * dt;
            auto const gyro = Emrys::ValueVector<Gyro>{
                0.5 * std::sin(0.7 * t),
                0.4 * std::cos(0.5 * t),
                0.2 + 0.3 * std::sin(0.3 * t)
            };
            truth = Emrys::simulate<State>(truth, motion, rng, dt, Emrys::sure<Gyro>(gyro));
            Emrys::ValueVector<Reading> const gravity = Emrys::simulate<State>(truth, accelerometer, rng);
            Emrys::ValueVector<Reading> const field = Emrys::simulate<State>(truth, magnetometer, rng);

            filter.predict(dt, Emrys::sure<Gyro>(gyro));
            auto const method = Emrys::methods::linearized();
            for (auto const & reading :
                 {Emrys::observe(method, accelerometer, gravity), Emrys::observe(method, magnetometer, field)})
            {
                auto const updated = filter.update(reading);
                if (!updated)
                {
                    // The innovation covariance was not positive definite, and the state is unchanged.
                    std::println("run {}, step {}: the update failed", run, k);
                    return 1;
                }
                nis.add(*updated);
            }
        }
        nees.add(truth, filter.state());

        // The error is taken by the space's minus(), so the attitude's is a rotation vector in radians.
        Emrys::DifferenceVector<State> const error = Emrys::minus<State>(truth, filter.state().mean());
        squaredAttitudeError += Emrys::diffBlock<State, Attitude>(error).squaredNorm();
    }

    double const degrees = 180.0 / std::numbers::pi;
    std::println(
        "RMS attitude error after {:g} s: {:.3f} deg",
        steps * dt,
        degrees * std::sqrt(squaredAttitudeError / runs)
    );

    // Each average is the sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES {:.2f}, average NIS {:.2f} per degree of freedom",
        nees.average().value(),
        nis.average().value()
    );

    // A consistent filter's NEES and NIS sums fall within the two-sided chi-squared bounds.
    bool const consistent =
        nees.verdict(0.999) == Emrys::Consistency::Consistent && nis.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("{}", consistent ? "consistent" : "NOT consistent");
    return consistent ? 0 : 1;
}
