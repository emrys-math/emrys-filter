// The ranging benchmark against TinyEKF: times EmrysFilter's EKF of emrys_ranging_kinds.h against
// TinyEKF's of tinyekf_ranging.h, over the planar and spatial cases of benchmark.cpp, with the
// analytic Jacobian, the only one TinyEKF has, and reports the ratio of their time per
// predict-update step. TinyEKF is written for microcontrollers, where the claim it backs is
// measured in code size and cycles (docs/design/benchmarks.md, claim 6); this runner is its desktop
// half, beside mherb_benchmark.cpp's table.
//
// Each case first runs both filters through the same simulated run and checks that they agree, as
// tinyekf_agreement.cpp does, and stops if they don't. It then times batches of steps through the
// pair driver of harness.h, alternating between the two filters. Only the pairs in double agree,
// and only they are timed (cases.h, TinyEkfAgreementCases). A table follows in place of the float
// ones: each library's float EKF's largest distance from the double EmrysFilter EKF over the run.
//
// With --count, it only runs the agreement pass of each timed pair, one pass of each filter
// through runSteps over the whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_tinyekf_benchmark --count

#include "../harness.h"
#include "distance.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"
#include "tinyekf_ranging.h"

#include <cstddef>
#include <cstdio>

namespace
{

using Benchmark::Options;
using Ranging::Distance;

/// The steps in a simulated run; a timed filter starts over from the initial estimate at its end.
constexpr std::size_t TrajectoryLength = 1000;

/// The width of each table's first column: the dimension and the anchor count.
constexpr int LabelWidth = 8;

/// Benchmarks one case in double and prints its row, or reports that the filters disagree and
/// returns false.
template <int D, int A>
[[nodiscard]] bool
benchmarkCase(Options const & options)
{
    using EmrysFilter = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf>;
    using TinyFilter = Ranging::TinyEkfRanging<D, A>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<double>(Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1));
    auto const makeEmrys = [&] {
        return EmrysFilter{scenario};
    };
    auto const makeTiny = [&] {
        return TinyFilter{scenario};
    };
    auto const emrysStep = [&](EmrysFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };
    auto const tinyStep = [&](TinyFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };

    // The agreement pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto tiny = makeTiny();
        Benchmark::runSteps(emrys, emrysStep, 0, TrajectoryLength);
        Benchmark::runSteps(tiny, tinyStep, 0, TrajectoryLength);
        Distance const d = Ranging::distance(emrys.mean(), emrys.covarianceLower(), tiny.mean(), tiny.covariance());
        Benchmark::consume(double(emrys.mean().sum() + tiny.mean().sum()));
        double const tolerance = Ranging::agreementTolerance<double>;
        if (!(d.mean < tolerance && d.cov < tolerance))
        {
            std::printf(
                "%dD, %2d anchors: the filters disagree (mean %.3g, covariance %.3g relative)\n",
                D,
                A,
                d.mean,
                d.cov
            );
            return false;
        }
    }
    if (options.count)
    {
        return true;
    }

    auto emrys = Benchmark::Replay{makeEmrys, emrysStep, TrajectoryLength};
    auto tiny = Benchmark::Replay{makeTiny, tinyStep, TrajectoryLength};
    Benchmark::PairTiming const timing = Benchmark::timePair(options, emrys, tiny);
    Benchmark::consume(double(emrys.filter().mean().sum() + tiny.filter().mean().sum()));

    char label[LabelWidth + 1];
    std::snprintf(label, sizeof label, "%dD %3d", D, A);
    Benchmark::printRow(label, LabelWidth, timing);
    return true;
}

/// Prints the row of the float table for one case: the largest distance of each library's float EKF
/// from the double EmrysFilter EKF over the run.
template <int D, int A>
void
floatRow()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1);
    auto const floatRanges = Ranging::rangesIn<float>(run);
    auto reference = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf>{scenario};
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, float>{scenario};
    auto tiny = Ranging::TinyEkfRanging<D, A, float>{scenario};
    auto worstEmrys = Distance{0, 0};
    auto worstTiny = Distance{0, 0};
    for (std::size_t k = 0; k < TrajectoryLength; ++k)
    {
        reference.predict();
        emrys.predict();
        tiny.predict();
        reference.update(run.ranges[k]);
        emrys.update(floatRanges[k]);
        tiny.update(floatRanges[k]);
        auto const refCov = reference.covarianceLower();
        worstEmrys = Ranging::largerOf(
            worstEmrys,
            Ranging::distance(emrys.mean(), emrys.covarianceLower(), reference.mean(), refCov)
        );
        worstTiny =
            Ranging::largerOf(worstTiny, Ranging::distance(tiny.mean(), tiny.covariance(), reference.mean(), refCov));
    }
    std::printf(
        "%dD %3d    %10.2e %10.2e    %10.2e %10.2e\n",
        D,
        A,
        worstEmrys.mean,
        worstEmrys.cov,
        worstTiny.mean,
        worstTiny.cov
    );
    std::fflush(stdout);
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
        Benchmark::printPreamble("Ranging EKF in double, analytic Jacobian", "TinyEKF");
        std::printf("Nanoseconds per predict-update step, %d batches of each filter per case.\n\n", options.samples);
        std::printf("Case: dimension and anchor count. Ratio: EmrysFilter's time over TinyEKF's.\n\n");
        Benchmark::printColumns("case", LabelWidth, "TinyEKF");
    }

    bool ok = true;
    ok = benchmarkCase<2, 4>(options) && ok;
    ok = benchmarkCase<2, 8>(options) && ok;
    ok = benchmarkCase<2, 16>(options) && ok;
    ok = benchmarkCase<3, 6>(options) && ok;
    ok = benchmarkCase<3, 12>(options) && ok;
    ok = benchmarkCase<3, 24>(options) && ok;
    if (options.count)
    {
        return ok ? 0 : 1;
    }

    std::printf(
        "\nThe EKF in float, not timed: the largest distance from the double EmrysFilter EKF over %zu steps, ",
        TrajectoryLength
    );
    std::printf("relative to the size of its mean and of its covariance.\n\n");
    std::printf("%-*s    %-21s    %s\n", LabelWidth, "", "EmrysFilter", "TinyEKF");
    std::printf("%-*s    %10s %10s    %10s %10s\n", LabelWidth, "case", "mean", "covariance", "mean", "covariance");
    floatRow<2, 4>();
    floatRow<2, 8>();
    floatRow<2, 16>();
    floatRow<3, 6>();
    floatRow<3, 12>();
    floatRow<3, 24>();
    return ok ? 0 : 1;
}
