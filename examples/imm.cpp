// A target in the plane that switches between cruising and maneuvering, tracked from noisy position
// fixes by an interacting multiple model (IMM) filter. The filter runs one Kalman filter per mode,
// mixes them through a Markov chain at each prediction, and weighs the modes by how well each one
// predicted the fix. The program simulates the target's modes and motion from the same switching
// model the filter assumes, prints how well the mode probabilities follow the true mode, and checks
// the combined estimate against its covariance and against a Kalman filter with a single model. It
// returns nonzero if either check fails.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <print>
#include <random>

// The state is a position and a velocity in the plane, as in constant_velocity.cpp.
struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;

// The mode is a discrete space whose values are named by types: Cruising is mode 0, Maneuvering 1.
struct Cruising
{};

struct Maneuvering
{};

struct Maneuver : Emrys::Named<Cruising, Maneuvering>
{};

int
main()
{
    double const dt = 1.0;  // seconds between fixes
    int const runs = 1000;
    int const steps = 100;

    // Both modes move at nearly constant velocity. A cruising target's acceleration is white noise of
    // spectral density 0.01 m^2/s^3; a maneuvering target's is 4 m^2/s^3, so its velocity can change
    // by about 2 m/s in a second. The fixes arrive at a fixed rate, so each model is taken over that
    // one step, as in constant_velocity.cpp.
    double const cruiseDensity = 0.01;
    double const maneuverDensity = 4.0;
    auto const cruise = Emrys::fixedStep(Emrys::ConstantVelocity<Position, Velocity>{cruiseDensity}, dt);
    auto const maneuver = Emrys::fixedStep(Emrys::ConstantVelocity<Position, Velocity>{maneuverDensity}, dt);

    // The mode is a Markov chain. Row i of the transition matrix is the distribution of the next mode
    // after mode i: a cruise lasts about 33 steps, a maneuver about 10. The switching model pairs the
    // chain with one process model per mode, in the order of the mode's values.
    auto const chain = Emrys::MarkovChain<Maneuver>{Eigen::Matrix2d{{0.97, 0.03}, {0.1, 0.9}}};
    auto const switching = Emrys::SwitchingModel{chain, cruise, maneuver};

    // A fix measures the position with noise of 1 m^2 variance on each axis.
    double const fixVariance = 1.0;
    auto const gps = Emrys::measurementModel(
        Emrys::DifferenceMatrix<Position>::Identity(),
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(fixVariance * Emrys::DifferenceMatrix<Position>::Identity())
    );

    // The IMM's state is a hybrid: a probability for each mode, and a Gaussian given each mode. Both
    // modes start from the same Gaussian, and the target most likely starts out cruising.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{0.0, 0.0, 10.0, 0.0},
        Emrys::DifferenceVector<State>{10.0, 10.0, 4.0, 4.0}.asDiagonal()
    };
    auto const modePrior = Emrys::CategoricalParams<Maneuver>::fromWeights(0.9, 0.1);
    Eigen::Index const maneuvering = Emrys::valueOf<Maneuver, Maneuvering>();

    // One run's errors are correlated from step to step, so the estimation error is tested over
    // independent runs, one final estimate from each (the Monte Carlo test of constant_velocity.cpp).
    // The mode tracking and the position errors are counted at every step of every run.
    auto nees = Emrys::NeesMonitor{runs};
    auto rng = std::mt19937_64{7};
    int modeMatches = 0;
    double immSquaredError = 0;
    double singleSquaredError = 0;
    for (int run = 0; run < runs; ++run)
    {
        auto filter = Emrys::ImmFilter{switching, Emrys::HybridParams{modePrior, prior}};

        // For comparison, a Kalman filter that assumes the maneuvering model all the time: the single
        // model that never falls behind a maneuver.
        auto single = Emrys::KalmanFilter{maneuver, prior};

        // The true mode and state are drawn from the priors. The covariance is positive definite and
        // the weights are positive, by construction, so neither toDensity() can fail here.
        Emrys::ValueVector<State> truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        Eigen::Index mode = Emrys::drawSample(modePrior.toDensity().value(), rng);
        for (int k = 0; k < steps; ++k)
        {
            // The true mode takes one step of the chain, a draw from chain.next(mode), and the true
            // state moves by that mode's model: visit() calls the lambda with the process model that
            // the mode follows.
            mode = Emrys::simulate(mode, chain, rng);
            truth = switching.visit(mode, [&](auto const & motion) {
                return Emrys::simulate<State>(truth, motion, rng);
            });
            Emrys::ValueVector<Position> const z = Emrys::simulate<State>(truth, gps, rng);

            filter.predict();
            single.predict();
            auto const fix = Emrys::observe(gps, z);
            if (!filter.update(fix) || !single.update(fix))
            {
                // An innovation covariance was not positive definite, and the state is unchanged.
                std::println("run {}, step {}: the update failed", run, k);
                return 1;
            }

            // The mode probabilities are a CategoricalParams, weights that toDensity() checks define
            // a distribution before it gives the probabilities.
            auto const modes = filter.state().modes().toDensity();
            if (!modes)
            {
                std::println("run {}, step {}: the mode probabilities are not a distribution", run, k);
                return 1;
            }
            bool const saysManeuvering = modes->probability<Maneuvering>() > 0.5;
            modeMatches += saysManeuvering == (mode == maneuvering) ? 1 : 0;

            // collapsed() is the combined estimate: the single Gaussian with the mean and covariance
            // of the mixture over the modes.
            Emrys::DifferenceVector<State> const immError =
                Emrys::minus<State>(filter.state().toDensity().value().collapsed().mean(), truth);
            Emrys::DifferenceVector<State> const singleError = Emrys::minus<State>(single.state().mean(), truth);
            immSquaredError += Emrys::diffBlock<State, Position>(immError).squaredNorm();
            singleSquaredError += Emrys::diffBlock<State, Position>(singleError).squaredNorm();
        }
        nees.add(truth, filter.state().toDensity().value().collapsed());
    }

    double const immRms = std::sqrt(immSquaredError / (runs * steps));
    double const singleRms = std::sqrt(singleSquaredError / (runs * steps));
    std::println("the more probable mode is the true mode at {} of {} steps", modeMatches, runs * steps);
    std::println("position RMS error {:.2f} m, against {:.2f} m for the single-model filter", immRms, singleRms);

    // The average is the sum divided by its degrees of freedom, so 1 is ideal.
    double const average = nees.average().value();
    std::println("average NEES {:.2f} per degree of freedom", average);

    // The combined estimate is the moments of a mixture, not a Gaussian, so its NEES is not
    // chi-squared. The two-sided chi-squared test of constant_velocity.cpp fails here for one seed
    // in three to two in five, even over independent runs. The NEES averages 4 per run, as it
    // should, but varies far more than a chi-squared value: a run that ends just after a maneuver
    // starts can reach a NEES in the hundreds, while a run that ends in a cruise is underconfident.
    // The check asks instead that the average be within a factor of 1.5 of 1, which a maneuvering
    // model with a tenth of the true noise fails, and that the IMM track the position better than
    // the single model does.
    bool const consistent = average > 1 / 1.5 && average < 1.5;
    bool const better = immRms < singleRms;
    std::println("{}, {}", consistent ? "consistent" : "NOT consistent", better ? "better" : "NOT better");
    return consistent && better ? 0 : 1;
}
