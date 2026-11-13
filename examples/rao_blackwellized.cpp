// A target moving in the plane at nearly constant velocity, located only by its ranges to three
// beacons, and tracked by a Rao-Blackwellized particle filter. The program simulates the target
// and the ranges from the same models the filter assumes, runs the filter, and checks that its
// estimates' errors agree with their covariances: the normalized estimation error squared (NEES)
// against the true state. It returns nonzero if the check fails.
//
// The ranges are a nonlinear function of the position, which a single Gaussian fits poorly, so
// particles sample the position. Given one particle's path, though, the velocity is linear and
// Gaussian: it moves the position linearly, and the ranges don't depend on it. So each particle
// carries a Kalman filter over the velocity instead of sampling it too. That is
// Rao-Blackwellization: the particles cover two dimensions rather than four, and each particle's
// velocity distribution is exact rather than a cloud of samples, so fewer particles do what a
// plain particle filter over the whole state would need many more for.

#include <EmrysFilter/EmrysFilter.h>

#include <array>
#include <cmath>
#include <functional>
#include <print>
#include <random>

// The state is a position and a velocity in the plane. The particles sample the Position block;
// each particle carries a Gaussian over the Velocity block.
struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;

// The three ranges to the beacons, the space of the measurement.
using Ranges = Emrys::Linear<3>;

int
main()
{
    double const dt = 1.0;  // seconds between measurements
    int const runs = 50;
    int const steps = 30;

    // With 500 particles the final estimates run a little overconfident, and the test below fails
    // on about 4 seeds in 100; with 1000 it failed on 1 seed in 300, over two standard libraries.
    Eigen::Index const particles = 1000;

    // The position integrates the velocity, which a white-noise acceleration of spectral density
    // 0.1 m^2/s^3 drives. The model covers the whole state, both blocks.
    double const accelerationDensity = 0.1;
    auto const motion = Emrys::ConstantVelocity<Position, Velocity>{accelerationDensity};

    // A measurement is the three ranges to the beacons, each with noise of 1 m^2 variance. The
    // model reads only the Position block. Its function is generic in the scalar type, since the
    // Linearized method differentiates it with automatic-differentiation scalars. The model keeps
    // its own copy of the beacons, so it may outlive them.
    double const rangeVariance = 1.0;
    auto const beacons = std::array{
        Emrys::ValueVector<Position>{-40, -30},
        Emrys::ValueVector<Position>{60, -20},
        Emrys::ValueVector<Position>{10, 70}
    };
    auto const ranges = Emrys::measurementModel(
        [beacons]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const position = Emrys::block<Position>(in, x);
            auto result = Emrys::ValueVector<Ranges, T>{};
            for (int b = 0; b < 3; ++b)
            {
                result(b) = Emrys::minus<Position>(position, beacons[b].template cast<T>()).norm();
            }
            return result;
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(rangeVariance * Emrys::DifferenceMatrix<Ranges>::Identity())
    );

    // The prior over the whole state: within about a meter of the origin, moving at about 1 m/s
    // in an unknown direction. Its covariance is positive definite by construction, so
    // toDensity() can't fail here.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>::Zero(),
        1.0 * Emrys::DifferenceMatrix<State>::Identity()
    };
    auto const priorDensity = prior.toDensity().value();

    // The filter never resamples on its own. After each update, when the effective sample size
    // falls below half the particles, resample them, then jitter the positions a little (Roughen)
    // so that the copies resampling makes spread out again. Both draw from the generator the rest
    // of the program draws from, which they hold by reference.
    auto rng = std::mt19937_64{1};
    auto const resampling =
        Emrys::upkeep::whenEss(std::less{}, 0.5, Emrys::upkeep::resample(rng), Emrys::upkeep::roughen(rng));

    // The test of a filter is a simulation: draw a true path from the models the filter assumes,
    // filter its ranges, and compare the estimates with the truth. One run's errors are correlated
    // from step to step, so the estimation error is tested over independent runs, one final
    // estimate from each (the Monte Carlo test).
    auto nees = Emrys::NeesMonitor{runs};
    double positionError = 0.0;
    double velocityError = 0.0;
    for (int run = 0; run < runs; ++run)
    {
        // drawParticleGaussian<Position>() draws the particles' positions from the prior and gives
        // each particle the Gaussian of the velocity given its position. The true state is one
        // more draw from the prior.
        auto filter = Emrys::RbParticleFilter{motion, Emrys::drawParticleGaussian<Position>(prior, particles, rng)};
        Emrys::ValueVector<State> truth = Emrys::drawSamples(priorDensity, 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng, dt);
            Emrys::ValueVector<Ranges> const z = Emrys::simulate<State>(truth, ranges, rng);

            // Unlike a Kalman filter's, the prediction draws new positions, so it takes the generator.
            filter.predict(rng, dt);
            auto const updated = filter.update(Emrys::observe(Emrys::methods::linearized(), ranges, z));
            if (!updated || !updated->accepted)
            {
                // A particle's innovation covariance was not positive definite, or no particle could
                // explain the ranges. Either way the state is unchanged.
                std::println("run {}, step {}: the update failed", run, k);
                return 1;
            }
            if (!resampling(filter.state()))
            {
                // The weights could not be normalized, which an accepted update rules out.
                std::println("run {}, step {}: resampling failed", run, k);
                return 1;
            }
        }

        // The estimate is the particles collapsed to one Gaussian over the whole state: their
        // spread for the position, and the mixture of their Gaussians for the velocity.
        auto const estimate = filter.state().toDensity()->collapsed();
        Emrys::DifferenceVector<State> const error = Emrys::minus<State>(truth, estimate.mean());
        positionError += Emrys::diffBlock<State, Position>(error).squaredNorm();
        velocityError += Emrys::diffBlock<State, Velocity>(error).squaredNorm();
        nees.add(truth, estimate);
    }

    std::println(
        "RMS error after {} steps: position {:.2f} m, velocity {:.2f} m/s",
        steps,
        std::sqrt(positionError / runs),
        std::sqrt(velocityError / runs)
    );

    // The average is the NEES sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES of the final estimates of {} runs {:.2f} per degree of freedom",
        runs,
        nees.average().value()
    );

    // A consistent filter's NEES sum falls within the two-sided chi-squared bounds.
    bool const consistent = nees.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("{}", consistent ? "consistent" : "NOT consistent");
    return consistent ? 0 : 1;
}
