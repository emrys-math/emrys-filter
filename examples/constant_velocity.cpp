// A target moving in the plane at nearly constant velocity, tracked by a Kalman filter from noisy
// position fixes. The program simulates the target and the fixes from the same models the filter
// assumes, runs the filter, and checks that its estimates are consistent with their covariances:
// the normalized estimation error squared (NEES) against the true state, and the normalized
// innovation squared (NIS) of the fixes. It returns nonzero if either check fails.
//
// linear_kalman.cpp is the same filter written as the textbook writes it, on a plain vector with
// its matrices F, H, Q, and R spelled out. It prints the same NEES and NIS.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <print>
#include <random>

// The state is a position and a velocity in the plane. Each block of the state is a type, so the
// process model and the sensors name the blocks they act on.
struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;

int
main()
{
    double const dt = 0.5;  // seconds between fixes
    int const runs = 50;
    int const steps = 100;

    // The position integrates the velocity, which a white-noise acceleration of spectral density
    // 0.1 m^2/s^3 drives. The fixes arrive at a fixed rate, so the model is taken over that one step
    // and its transition and process noise are computed once, here. With irregular timestamps,
    // keep the process model itself and pass each step's length to predict(dt) instead.
    double const accelerationDensity = 0.1;
    auto const motion = Emrys::fixedStep(Emrys::ConstantVelocity<Position, Velocity>{accelerationDensity}, dt);

    // A fix measures the position with noise of 1 m^2 variance on each axis. The measurement is
    // linear, so the model is given by its matrix, and its update is the Kalman filter's exact one,
    // with no method to name.
    double const fixVariance = 1.0;
    auto const gps = Emrys::measurementModel(
        Emrys::DifferenceMatrix<Position>::Identity(),
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(fixVariance * Emrys::DifferenceMatrix<Position>::Identity())
    );

    // The filter starts from a prior over the state: near the origin, moving slowly.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{0.0, 0.0, 1.0, 0.5},
        Emrys::DifferenceVector<State>{10.0, 10.0, 1.0, 1.0}.asDiagonal()
    };

    // The test of a filter is a simulation: draw a true trajectory from the models the filter
    // assumes, filter its fixes, and compare the estimates with the truth. One run's errors are
    // correlated from step to step, so the estimation error is tested over independent runs, one
    // final estimate from each (the Monte Carlo test). The innovations are independent from step
    // to step, so every fix of every run goes into the innovation test.
    auto nees = Emrys::NeesMonitor{runs};
    auto nis = Emrys::NisMonitor{runs * steps};
    auto rng = std::mt19937_64{2};
    double squaredError = 0.0;
    for (int run = 0; run < runs; ++run)
    {
        auto filter = Emrys::KalmanFilter{motion, prior};

        // The true state is drawn from the prior, then moved and measured by the same models. The
        // prior's covariance is positive definite by construction, so toDensity() can't fail here.
        Emrys::ValueVector<State> truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng);
            Emrys::ValueVector<Position> const z = Emrys::simulate<State>(truth, gps, rng);

            filter.predict();
            auto const updated = filter.update(Emrys::observe(gps, z));
            if (!updated)
            {
                // The innovation covariance was not positive definite, and the state is unchanged.
                std::println("run {}, step {}: the update failed", run, k);
                return 1;
            }
            nis.add(*updated);
        }
        nees.add(truth, filter.state());
        Emrys::DifferenceVector<State> const error = Emrys::minus<State>(truth, filter.state().mean());
        squaredError += Emrys::diffBlock<State, Position>(error).squaredNorm();
    }

    std::println("RMS position error after {} fixes: {:.2f} m", steps, std::sqrt(squaredError / runs));

    // Each average is the sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES {:.2f}, average NIS {:.2f} per degree of freedom",
        nees.average().value(),
        nis.average().value()
    );

    // A consistent filter's NEES and NIS sums fall within the two-sided chi-squared bounds. A
    // verdict compares equal to Consistent only when there is one, so an empty monitor fails.
    bool const consistent =
        nees.verdict(0.999) == Emrys::Consistency::Consistent && nis.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("{}", consistent ? "consistent" : "NOT consistent");
    return consistent ? 0 : 1;
}
