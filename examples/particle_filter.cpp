// A drifting target in the plane, such as a buoy, located from its ranges to two beacons by a particle
// filter. Ranges to two beacons can't tell a position from its mirror image across the line through
// them, so the posterior has two peaks, one on each side. A Gaussian, which is all a Kalman filter
// carries, can only sit between them or pick one; the particles keep both, until a coarse position
// fix, which starts partway through, says which side the target is on.
//
// The program simulates the target and the measurements from the same models the filter assumes,
// runs the filter many times, and checks two things. While only ranges arrive, the particles must
// keep weight on both sides, as the true posterior does. Once the fixes have picked a side, the
// posterior has one peak, and a Gaussian fit to the particles at the end of each run must be
// consistent with the truth: its normalized estimation error squared (NEES), over the runs, must
// pass the chi-squared test. The NEES of a fit to two peaks would mean nothing, which is why the
// test waits for the fixes. It returns nonzero if either check fails.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <functional>
#include <print>
#include <random>

// The state is the target's position in the plane.
struct Position : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position>;

// The two ranges to the beacons, the space of the range measurement.
using Ranges = Emrys::Linear<2>;

int
main()
{
    double const dt = 1.0;  // seconds between measurements
    int const runs = 20;
    int const rangeOnlySteps = 20;
    int const steps = 40;
    Eigen::Index const count = 20000;  // particles

    // The target drifts as a random walk, its position spreading by 0.1 m^2 a second on each axis.
    double const driftDensity = 0.1;
    auto const motion = Emrys::RandomWalk<Position>{driftDensity};

    // Each step measures the ranges to beacons at (-10, 0) and (10, 0), with noise of 1 m^2
    // variance. A particle filter only evaluates each measurement's likelihood at each particle, so
    // observe() takes no method here, where a Kalman filter would name the one that linearizes.
    double const rangeVariance = 1.0;
    auto const left = Emrys::ValueVector<Position>{-10.0, 0.0};
    auto const right = Emrys::ValueVector<Position>{10.0, 0.0};
    auto const ranges = Emrys::measurementModel(
        [left, right](auto const & in, auto const &, auto const & x) {
            auto const position = Emrys::block<Position>(in, x);
            return Emrys::ValueVector<Ranges>{
                Emrys::minus<Position>(position, left).norm(),
                Emrys::minus<Position>(position, right).norm()
            };
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(rangeVariance * Emrys::DifferenceMatrix<Ranges>::Identity())
    );

    // After the first 20 steps, a coarse position fix arrives with each pair of ranges, with noise of
    // 4 m^2 variance on each axis.
    double const fixVariance = 4.0;
    auto const fix = Emrys::measurementModel(
        Emrys::DifferenceMatrix<Position>::Identity(),
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(fixVariance * Emrys::DifferenceMatrix<Position>::Identity())
    );

    // The prior is centered on the beacons' line, so ranges alone never favor one side. Its covariance
    // is positive definite by construction, so toDensity() can't fail here.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>::Zero(),
        9.0 * Emrys::DifferenceMatrix<State>::Identity()
    };
    auto const priorDensity = prior.toDensity().value();

    // The filter never resamples on its own. After each step's updates, the particles are resampled
    // when their effective sample size falls below half their number, which drops the particles
    // that explain the measurements badly and copies the ones that explain them well. Resampling
    // draws from the generator the rest of the program draws from, which it holds by reference.
    auto rng = std::mt19937_64{1};
    auto const resampling = Emrys::upkeep::whenEss(std::less{}, 0.5, Emrys::upkeep::resample(rng));

    // The test of a filter is a simulation: draw a true path from the models the filter assumes,
    // filter its measurements, and compare the estimates with the truth. One run's errors are
    // correlated from step to step, so the estimation error is tested over independent runs, one
    // final estimate from each (the Monte Carlo test).
    auto nees = Emrys::NeesMonitor{runs};
    bool keptBothSides = true;
    for (int run = 0; run < runs; ++run)
    {
        // The filter starts from equally weighted particles drawn from the prior, and the true
        // position is one more draw from it.
        auto filter =
            Emrys::ParticleFilter{motion, Emrys::ParticleSet<State>{Emrys::drawSamples(priorDensity, count, rng)}};
        Emrys::ValueVector<State> truth = Emrys::drawSamples(priorDensity, 1, rng);

        // The weight of the particles above the beacons' line, where their y coordinate, row 1 of
        // the states, is positive. The updates keep the weights valid, so normalizedWeights() can't
        // fail here.
        auto const weightAbove = [&] {
            Eigen::ArrayXd const weights = filter.state().normalizedWeights().value();
            return (filter.state().states().row(1).transpose().array() > 0.0).select(weights, 0.0).sum();
        };

        for (int k = 1; k <= steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng, dt);
            filter.predict(rng, dt);

            // An update multiplies each particle's weight by the measurement's likelihood there. It
            // fails if the weights are invalid, and is rejected, leaving the weights alone, if no
            // particle can explain the measurement.
            Emrys::ValueVector<Ranges> const z = Emrys::simulate<State>(truth, ranges, rng);
            auto const rangesUpdated = filter.update(Emrys::observe(ranges, z));
            bool accepted = rangesUpdated && rangesUpdated->accepted;
            if (accepted && k > rangeOnlySteps)
            {
                Emrys::ValueVector<Position> const position = Emrys::simulate<State>(truth, fix, rng);
                auto const fixUpdated = filter.update(Emrys::observe(fix, position));
                accepted = fixUpdated && fixUpdated->accepted;
            }
            if (!accepted)
            {
                std::println("run {}, step {}: an update failed or was rejected", run, k);
                return 1;
            }

            // The true posterior keeps exactly half its weight above the line while only ranges
            // arrive. The particles' split wanders from half, since resampling copies particles at
            // random and the two peaks never exchange them, so the check is only that neither side
            // falls below 10%: that the filter hasn't lost a side.
            if (k <= rangeOnlySteps)
            {
                keptBothSides = keptBothSides && std::abs(weightAbove() - 0.5) < 0.4;
            }

            // With two peaks, the particles' mean lies between them, near neither, which is where a
            // Gaussian filter's estimate would be too.
            if (run == 0 && k == rangeOnlySteps)
            {
                // The weights are valid, as above, so moments() can't fail here.
                std::println(
                    "ranges only: {:.0f}% of the weight above the line; target at y = {:.1f} m, mean at y = {:.1f} m",
                    100.0 * weightAbove(),
                    truth(1),
                    filter.state().moments().value().mean()(1)
                );
            }

            if (!resampling(filter.state()))
            {
                std::println("run {}, step {}: the particles' weights are invalid", run, k);
                return 1;
            }
        }

        // An estimate is read from the particles: their weighted mean and covariance, which fail only
        // if the weights are invalid. The NEES takes a Gaussian with those moments.
        auto const moments = filter.state().moments();
        if (!moments)
        {
            std::println("run {}: the particles' weights are invalid", run);
            return 1;
        }
        if (run == 0)
        {
            std::println(
                "with fixes:  {:.0f}% of the weight above the line; target at y = {:.1f} m, mean at y = {:.1f} m",
                100.0 * weightAbove(),
                truth(1),
                moments->mean()(1)
            );
        }
        nees.add(truth, Emrys::GaussianParams<State>{*moments});
    }

    // The average is the NEES sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES of the final estimates of {} runs {:.2f} per degree of freedom",
        runs,
        nees.average().value()
    );

    bool const passed = keptBothSides && nees.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("{}", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
