// A unicycle, a vehicle that drives along its heading and turns, tracked from range and bearing
// measurements to known landmarks by two nonlinear Kalman filters side by side: the extended
// Kalman filter (the Linearized method) and the unscented Kalman filter (the Unscented method).
// The heading and the bearings are angles, which wrap at plus and minus pi. The program simulates
// the vehicle and the measurements from the same models the filters assume, runs both filters on
// the same data, and checks that each one's estimates are consistent with its covariances (NEES and
// NIS). It returns nonzero if any check fails.

#include <EmrysFilter/EmrysFilter.h>

#include <array>
#include <cmath>
#include <print>
#include <random>

// The state is a position in meters, a heading in radians, a speed in m/s, and a turn rate in
// rad/s. RadiansPi is an angle in [-pi, pi): the filters wrap it and take its differences the
// short way around.
struct Position : Emrys::Linear<2>
{};

struct Heading : Emrys::RadiansPi<1>
{};

struct Speed : Emrys::Linear<1>
{};

struct TurnRate : Emrys::Linear<1>
{};

using State = Emrys::Composite<Position, Heading, Speed, TurnRate>;

// A measurement is a range in meters and a bearing in radians, relative to the heading, so the
// bearing wraps too.
struct Range : Emrys::Linear<1>
{};

struct Bearing : Emrys::RadiansPi<1>
{};

using Measurement = Emrys::Composite<Range, Bearing>;

// The unicycle's motion over a step dt: the position moves dt times the speed along the heading,
// and the heading turns by dt times the turn rate. The library has no model for it, so it is
// written here: a function of the blocks it reads, named in reads(), that fills the blocks it
// writes, named in writes(). It computes in T, the scalar of x, which the filter supplies, since
// the extended Kalman filter differentiates it automatically. Its noise comes in through the speed
// and the turn rate, so it adds none of its own.
[[nodiscard]] auto
unicycle()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            T const heading = Emrys::block<Heading>(in, x)(0);
            T const distance = dt * Emrys::block<Speed>(in, x)(0);
            auto const displacement = Emrys::DifferenceVector<Position, T>{
                distance * Eigen::numext::cos(heading),
                distance * Eigen::numext::sin(heading)
            };
            Emrys::DifferenceVector<Heading, T> const turn = dt * Emrys::block<TurnRate>(in, x);

            // plus() wraps the turned heading back into [-pi, pi).
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Position>(out, y) = Emrys::plus<Position>(Emrys::block<Position>(in, x), displacement);
            Emrys::block<Heading>(out, y) = Emrys::plus<Heading>(Emrys::block<Heading>(in, x), turn);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Heading>, Emrys::path<Speed>, Emrys::path<TurnRate>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Heading>)
    );
}

// Three landmarks at known positions, in meters.
std::array const landmarks{
    Emrys::ValueVector<Position>{60.0, 0.0},
    Emrys::ValueVector<Position>{-30.0, 50.0},
    Emrys::ValueVector<Position>{-30.0, -50.0}
};

// A measurement model for each landmark, the range and bearing to it with noise of 0.5 m standard
// deviation in range and 1 degree (0.0175 rad) in bearing. The method a filter observes it by
// decides how it carries its uncertainty through the function: Linearized differentiates it,
// Unscented evaluates it at sigma points. writes() names the output's blocks, so the output is the
// Composite<Range, Bearing> that Measurement names, and the bearing's residual wraps.
[[nodiscard]] auto
landmarkModels()
{
    double const rangeVariance = 0.25;
    double const bearingVariance = 3e-4;
    Emrys::DifferenceMatrix<Measurement> const noise =
        Emrys::DifferenceVector<Measurement>{rangeVariance, bearingVariance}.asDiagonal();
    auto const toLandmark = [&](Emrys::ValueVector<Position> const & landmark) {
        return Emrys::measurementModel(
            [landmark]<typename T>(auto const & in, auto const &, auto const & x) {
                Emrys::DifferenceVector<Position, T> const offset =
                    Emrys::minus<Position>(landmark.cast<T>(), Emrys::block<Position>(in, x));

                // The bearing is the direction to the landmark less the heading, which minus()
                // takes the short way around, in [-pi, pi).
                auto const direction = Emrys::ValueVector<Heading, T>{Eigen::numext::atan2(offset(1), offset(0))};
                Emrys::DifferenceVector<Heading, T> const bearing =
                    Emrys::minus<Heading>(direction, Emrys::block<Heading>(in, x));
                return Emrys::ValueVector<Measurement, T>{T(offset.norm()), bearing(0)};
            },
            Emrys::reads(Emrys::path<Position>, Emrys::path<Heading>),
            Emrys::writes(Emrys::path<Range>, Emrys::path<Bearing>),
            Emrys::additiveNoise(noise)
        );
    };
    return std::array{toLandmark(landmarks[0]), toLandmark(landmarks[1]), toLandmark(landmarks[2])};
}

int
main()
{
    double const dt = 0.5;  // seconds between steps
    int const runs = 50;
    int const steps = 60;

    // The speed is a random walk of spectral density 0.1 m^2/s^3, and the turn rate decays toward
    // zero with a time constant of 10 s and a stationary variance of 0.01 rad^2/s^2.
    double const speedDensity = 0.1;
    auto const motion = Emrys::compose(
        unicycle(),
        Emrys::RandomWalk<Speed>{speedDensity},
        Emrys::OrnsteinUhlenbeck<TurnRate>{Emrys::TimeConstant{10.0}, Emrys::StationaryVariance{0.01}}
    );
    auto const models = landmarkModels();

    // Both filters start from the same prior: near the origin, heading east at about 3 m/s.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{0.0, 0.0, 0.0, 3.0, 0.0},
        Emrys::DifferenceVector<State>{4.0, 4.0, 0.05, 0.25, 0.01}.asDiagonal()
    };

    // The estimation error is tested over independent runs, one final estimate from each, and the
    // innovations over every update of every run, as in constant_velocity.cpp.
    auto ekfNees = Emrys::NeesMonitor{runs};
    auto ukfNees = Emrys::NeesMonitor{runs};
    auto ekfNis = Emrys::NisMonitor{runs * steps * 3};
    auto ukfNis = Emrys::NisMonitor{runs * steps * 3};
    auto rng = std::mt19937_64{1};
    double ekfSquaredError = 0.0;
    double ukfSquaredError = 0.0;
    for (int run = 0; run < runs; ++run)
    {
        auto ekf = Emrys::KalmanFilter{motion, prior, Emrys::methods::linearized()};
        auto ukf = Emrys::KalmanFilter{motion, prior, Emrys::methods::unscented()};

        // The prior's covariance is positive definite by construction, so toDensity() can't fail here.
        Emrys::ValueVector<State> truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng, dt);
            ekf.predict(dt);
            ukf.predict(dt);
            for (std::size_t i = 0; i < landmarks.size(); ++i)
            {
                // Both filters see the same measurement, drawn once.
                Emrys::ValueVector<Measurement> const z = Emrys::simulate<State>(truth, models[i], rng);
                auto const ekfUpdated = ekf.update(Emrys::observe(Emrys::methods::linearized(), models[i], z));
                auto const ukfUpdated = ukf.update(Emrys::observe(Emrys::methods::unscented(), models[i], z));
                if (!ekfUpdated || !ukfUpdated)
                {
                    // The innovation covariance was not positive definite, and the state is unchanged.
                    std::println("run {}, step {}: an update failed", run, k);
                    return 1;
                }
                ekfNis.add(*ekfUpdated);
                ukfNis.add(*ukfUpdated);
            }
        }
        ekfNees.add(truth, ekf.state());
        ukfNees.add(truth, ukf.state());
        Emrys::DifferenceVector<State> const ekfError = Emrys::minus<State>(truth, ekf.state().mean());
        Emrys::DifferenceVector<State> const ukfError = Emrys::minus<State>(truth, ukf.state().mean());
        ekfSquaredError += Emrys::diffBlock<State, Position>(ekfError).squaredNorm();
        ukfSquaredError += Emrys::diffBlock<State, Position>(ukfError).squaredNorm();
    }

    std::println(
        "RMS position error after {} steps: EKF {:.2f} m, UKF {:.2f} m",
        steps,
        std::sqrt(ekfSquaredError / runs),
        std::sqrt(ukfSquaredError / runs)
    );

    // Each average is the sum divided by its degrees of freedom, so 1 is ideal.
    std::println("average NEES: EKF {:.2f}, UKF {:.2f}", ekfNees.average().value(), ukfNees.average().value());
    std::println("average NIS: EKF {:.2f}, UKF {:.2f}", ekfNis.average().value(), ukfNis.average().value());

    // A consistent filter's NEES and NIS sums fall within the two-sided chi-squared bounds.
    constexpr auto Consistent = Emrys::Consistency::Consistent;
    bool const ekfConsistent = ekfNees.verdict(0.999) == Consistent && ekfNis.verdict(0.999) == Consistent;
    bool const ukfConsistent = ukfNees.verdict(0.999) == Consistent && ukfNis.verdict(0.999) == Consistent;
    std::println(
        "EKF {}, UKF {}",
        ekfConsistent ? "consistent" : "NOT consistent",
        ukfConsistent ? "consistent" : "NOT consistent"
    );
    return ekfConsistent && ukfConsistent ? 0 : 1;
}
