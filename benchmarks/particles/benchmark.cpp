// The particle benchmark: times the EmrysFilter bootstrap filter of emrys_particles.h against the
// plain-Eigen one of eigen_particles.h on the ranging problem, over planar and spatial cases,
// growing particle counts, and each likelihood, and reports the ratio of their time per particle
// per predict-update-housekeep step.
//
// Each case first runs both filters, seeded alike, through the same simulated run and checks
// that they hold the same particles with the same weights at its end, as the agreement test
// does, and stops if they don't: a timing of two different filters means nothing. It then times
// batches of steps through the pair driver of harness.h, alternating between the two filters, and
// reports the median, 99th percentile, and worst batch as nanoseconds per particle per step.
//
// With --count, it only runs the agreement pass, one pass of each filter through runSteps over the
// whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_particles_benchmark
// --count
//
// Each instantiation of runSteps then carries the instructions of TrajectoryLength steps of one
// filter in one case. The particle counts of one case share an instantiation, so --particles N
// restricts a run to one count, under Callgrind or when timing.

#include "../harness.h"
#include "eigen_particles.h"
#include "emrys_particles.h"
#include "scenario.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace
{

using Benchmark::Options;
using Particles::EigenParticles;
using Particles::EmrysParticles;
using Particles::Likelihood;

/// The steps in a simulated run; a timed filter starts over from the initial particles at its end.
constexpr std::size_t TrajectoryLength = 200;

/// The width of the table's first column: the dimension, the anchor count, the particle count, and
/// the likelihood.
constexpr int LabelWidth = 23;

/// The name of a likelihood, for the table.
[[nodiscard]] char const *
nameOf(Likelihood kind)
{
    return kind == Likelihood::Noise ? "noise" : kind == Likelihood::Own ? "own" : "batch";
}

/// Benchmarks one case at \a count particles and prints its row, or reports that the filters
/// disagree and returns false.
template <int D, int A, Likelihood K>
[[nodiscard]] bool
benchmarkCase(Options const & options, Eigen::Index count)
{
    auto const scenario = Particles::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1);
    auto const initial = Particles::drawInitialParticles(scenario, count, 2);
    auto const makeEmrys = [&] {
        return EmrysParticles<D, A, K>{scenario, initial, 3};
    };
    auto const makeEigen = [&] {
        return EigenParticles<D, A>{scenario, initial, 3};
    };
    auto const step = [&](auto & filter, std::size_t k) {
        filter.predict();
        filter.update(run.ranges[k]);
        filter.housekeep();
    };

    // The agreement pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto eigen = makeEigen();
        Benchmark::runSteps(emrys, step, 0, TrajectoryLength);
        Benchmark::runSteps(eigen, step, 0, TrajectoryLength);
        double const stateError =
            (emrys.state().states() - eigen.states()).norm() / std::max(1.0, eigen.states().norm());
        double const weightError = (emrys.state().logWeights() - eigen.logWeights()).matrix().norm()
                                 / std::max(1.0, eigen.logWeights().matrix().norm());
        Benchmark::consume(emrys.state().states().sum() + eigen.states().sum());
        if (!(stateError < 1e-9 && weightError < 1e-9))
        {
            std::printf(
                "%dD, %2d anchors, %lld particles, %s: the filters disagree (particles %.3g, weights %.3g relative)\n",
                D,
                A,
                static_cast<long long>(count),
                nameOf(K),
                stateError,
                weightError
            );
            return false;
        }
    }
    if (options.count)
    {
        return true;
    }

    auto emrys = Benchmark::Replay{makeEmrys, step, TrajectoryLength};
    auto eigen = Benchmark::Replay{makeEigen, step, TrajectoryLength};
    Benchmark::PairTiming const perStep = Benchmark::timePair(options, emrys, eigen);
    Benchmark::consume(emrys.filter().state().states().sum() + eigen.filter().states().sum());

    auto const perParticle =
        Benchmark::PairTiming{perStep.emrys / static_cast<double>(count), perStep.eigen / static_cast<double>(count)};
    char label[LabelWidth + 1];
    std::snprintf(label, sizeof label, "%dD %3d  %7lld  %-5s", D, A, static_cast<long long>(count), nameOf(K));
    Benchmark::printRow(label, LabelWidth, perParticle, 1);
    return true;
}

/// Benchmarks a dimension and anchor count at \a count particles under each likelihood.
template <int D, int A>
[[nodiscard]] bool
benchmarkSize(Options const & options, Eigen::Index count)
{
    bool const noise = benchmarkCase<D, A, Likelihood::Noise>(options, count);
    bool const own = benchmarkCase<D, A, Likelihood::Own>(options, count);
    bool const batch = benchmarkCase<D, A, Likelihood::Batched>(options, count);
    return noise && own && batch;
}

}  // namespace

int
main(int argc, char ** argv)
{
    auto options = Options{};
    if (!Benchmark::parse(argc, argv, options))
    {
        Benchmark::printUsage(argv[0]);
        return 2;
    }
    if (!options.count)
    {
        Benchmark::printPreamble("Ranging bootstrap particle filter");
        std::printf(
            "Nanoseconds per particle per predict-update-housekeep step, %d batches of each filter per case.\n\n",
            options.samples
        );
        std::printf(
            "Case: dimension, anchor count, particle count, and whether the likelihood comes from the noise's\n"
            "density, the measurement's own closed form, or the noise's density at a batched function's ranges.\n"
            "Ratio: EmrysFilter's time over Eigen's.\n\n"
        );
        Benchmark::printColumns("case   particles  lik.", LabelWidth);
    }

    auto const counts = {Eigen::Index{1000}, Eigen::Index{10'000}, Eigen::Index{100'000}, Eigen::Index{1'000'000}};
    auto const wanted = [&](Eigen::Index count) {
        return options.particles == 0 || count == options.particles;
    };
    bool ok = true;
    for (Eigen::Index const count : counts)
    {
        if (wanted(count))
        {
            ok = benchmarkSize<2, 4>(options, count) && ok;
        }
    }
    for (Eigen::Index const count : counts)
    {
        if (wanted(count))
        {
            ok = benchmarkSize<3, 6>(options, count) && ok;
        }
    }
    return ok ? 0 : 1;
}
