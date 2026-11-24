// The ranging benchmark against FrancoisCarouge/Kalman: times EmrysFilter's EKF of
// emrys_ranging_kinds.h against the library's of carouge_ranging.h, over the planar and spatial
// cases of benchmark.cpp, with the analytic Jacobian, the only one the library has, and reports
// the ratio of their time per predict-update step. The library's filter is in carouge_ranging.cpp,
// a translation unit of its own (carouge_ranging.h says why), so each of its steps makes two calls
// that are not inlined; docs/design/benchmarks.md gives what that costs.
//
// Each case first runs both filters through the same simulated run and compares them, as
// carouge_agreement.cpp does, and stops if the comparison fails. It then times batches of steps
// through the pair driver of harness.h, alternating between the two filters. The pairs agree in
// double and in float (cases.h, CarougeCases), so no row is marked. A table follows: each library's
// float EKF's largest distance from the double EmrysFilter EKF over the run.
//
// With --count, it only runs the comparison pass of each pair, one pass of each filter through
// runSteps over the whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_carouge_benchmark --count

#include "../harness.h"
#include "carouge_ranging.h"
#include "distance.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"

#include <cstddef>
#include <cstdio>

namespace
{

using Benchmark::Options;
using Ranging::Distance;

/// The steps in a simulated run; a timed filter starts over from the initial estimate at its end.
constexpr std::size_t TrajectoryLength = 1000;

/// The width of the distance table's first column: the dimension and the anchor count.
constexpr int LabelWidth = 8;

/// The width of the timing table's first column: the dimension, the anchor count, and the scalar.
constexpr int TimingLabelWidth = 15;

/// Benchmarks one case in \a Scalar and prints its row. Reports that the filters disagree and
/// returns false if they do.
template <int D, int A, typename Scalar>
[[nodiscard]] bool
benchmarkCase(Options const & options)
{
    using EmrysFilter = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>;
    using CarougeFilter = Ranging::CarougeRanging<D, A, Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1));
    auto const makeEmrys = [&] {
        return EmrysFilter{scenario};
    };
    auto const makeCarouge = [&] {
        return CarougeFilter{scenario};
    };
    auto const emrysStep = [&](EmrysFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };
    auto const carougeStep = [&](CarougeFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };

    // The comparison pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto carouge = makeCarouge();
        Benchmark::runSteps(emrys, emrysStep, 0, TrajectoryLength);
        Benchmark::runSteps(carouge, carougeStep, 0, TrajectoryLength);
        Distance const d =
            Ranging::distance(emrys.mean(), emrys.covarianceLower(), carouge.mean(), carouge.covariance());
        Benchmark::consume(double(emrys.mean().sum() + carouge.mean().sum()));
        double const tolerance = Ranging::agreementTolerance<Scalar>;
        if (!(d.mean < tolerance && d.cov < tolerance))
        {
            std::printf(
                "%dD, %2d anchors, %s: the filters disagree (mean %.3g, covariance %.3g relative)\n",
                D,
                A,
                Ranging::scalarName<Scalar>(),
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
    auto carouge = Benchmark::Replay{makeCarouge, carougeStep, TrajectoryLength};
    Benchmark::PairTiming const timing = Benchmark::timePair(options, emrys, carouge);
    Benchmark::consume(double(emrys.filter().mean().sum() + carouge.filter().mean().sum()));

    char label[TimingLabelWidth + 1];
    std::snprintf(label, sizeof label, "%dD %3d   %-6s", D, A, Ranging::scalarName<Scalar>());
    Benchmark::printRow(label, TimingLabelWidth, timing);
    return true;
}

/// Benchmarks the pair at a dimension and anchor count, in double and in float.
template <int D, int A>
[[nodiscard]] bool
benchmarkSize(Options const & options)
{
    bool const inDouble = benchmarkCase<D, A, double>(options);
    bool const inFloat = benchmarkCase<D, A, float>(options);
    return inDouble && inFloat;
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
    auto carouge = Ranging::CarougeRanging<D, A, float>{scenario};
    auto worstEmrys = Distance{0, 0};
    auto worstCarouge = Distance{0, 0};
    for (std::size_t k = 0; k < TrajectoryLength; ++k)
    {
        reference.predict();
        emrys.predict();
        carouge.predict();
        reference.update(run.ranges[k]);
        emrys.update(floatRanges[k]);
        carouge.update(floatRanges[k]);
        auto const refCov = reference.covarianceLower();
        worstEmrys = Ranging::largerOf(
            worstEmrys,
            Ranging::distance(emrys.mean(), emrys.covarianceLower(), reference.mean(), refCov)
        );
        worstCarouge = Ranging::largerOf(
            worstCarouge,
            Ranging::distance(carouge.mean(), carouge.covariance(), reference.mean(), refCov)
        );
    }
    std::printf(
        "%dD %3d    %10.2e %10.2e    %10.2e %10.2e\n",
        D,
        A,
        worstEmrys.mean,
        worstEmrys.cov,
        worstCarouge.mean,
        worstCarouge.cov
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
        Benchmark::printPreamble("Ranging EKF, analytic Jacobian", "FrancoisCarouge/Kalman");
        std::printf("Nanoseconds per predict-update step, %d batches of each filter per case.\n\n", options.samples);
        std::printf("Case: dimension and anchor count. Ratio: EmrysFilter's time over FrancoisCarouge/Kalman's.\n\n");
        Benchmark::printColumns("case     scalar", TimingLabelWidth, "Carouge");
    }

    bool ok = true;
    ok = benchmarkSize<2, 4>(options) && ok;
    ok = benchmarkSize<2, 8>(options) && ok;
    ok = benchmarkSize<2, 16>(options) && ok;
    ok = benchmarkSize<3, 6>(options) && ok;
    ok = benchmarkSize<3, 12>(options) && ok;
    ok = benchmarkSize<3, 24>(options) && ok;
    if (options.count)
    {
        return ok ? 0 : 1;
    }

    std::printf(
        "\nThe EKF in float: the largest distance from the double EmrysFilter EKF over %zu steps, ",
        TrajectoryLength
    );
    std::printf("relative to the size of its mean and of its covariance.\n\n");
    std::printf("%-*s    %-21s    %s\n", LabelWidth, "", "EmrysFilter", "FrancoisCarouge/Kalman");
    std::printf("%-*s    %10s %10s    %10s %10s\n", LabelWidth, "case", "mean", "covariance", "mean", "covariance");
    floatRow<2, 4>();
    floatRow<2, 8>();
    floatRow<2, 16>();
    floatRow<3, 6>();
    floatRow<3, 12>();
    floatRow<3, 24>();
    return ok ? 0 : 1;
}
