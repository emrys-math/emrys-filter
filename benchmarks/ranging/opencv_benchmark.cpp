// The ranging benchmark against OpenCV: times EmrysFilter's EKF of emrys_ranging_kinds.h against
// cv::KalmanFilter run as an EKF in opencv_ranging.h, over the planar and spatial cases of
// benchmark.cpp, with the analytic Jacobian, the only one OpenCV has, and reports the ratio of
// their time per predict-update step.
//
// OpenCV's filter never mirrors its covariance, and the asymmetry that rounding leaves in it grows
// until the filter diverges: in double the pair agrees to 1e-9 for the first 400 to 500 steps of
// each case, and by step 2000 OpenCV's spatial estimate is lost, while EmrysFilter's stays within
// 2e-15 of its own estimate in long double. So each timed run is the first AgreedLength steps,
// over which the pair agrees (opencv_agreement.cpp), and a timed filter starts over from the
// initial estimate at its end, untimed. Each case first runs both filters through that run and
// checks that they agree, and stops if they don't. It then times batches of steps through the pair
// driver of harness.h, alternating between the two filters.
//
// Two tables follow, not timed: OpenCV's distance from EmrysFilter over a long run, beside
// EmrysFilter's largest distance from its long double self, and each library's float EKF's largest
// distance from the double EmrysFilter EKF over the timed run.
//
// With --count, it only runs the agreement pass of each timed pair, one pass of each filter
// through runSteps over the whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_opencv_benchmark --count

#include "../harness.h"
#include "distance.h"
#include "emrys_ranging_kinds.h"
#include "opencv_ranging.h"
#include "scenario.h"

#include <cstddef>
#include <cstdio>

namespace
{

using Benchmark::Options;
using Ranging::Distance;

/// The steps of a timed run, over which OpenCV's EKF agrees with EmrysFilter's in double.
constexpr std::size_t AgreedLength = 300;

/// The steps of the long run, over which OpenCV's EKF diverges.
constexpr int LongLength = 3000;

/// The width of each table's first column: the dimension and the anchor count.
constexpr int LabelWidth = 8;

/// Benchmarks one case in double and prints its row, or reports that the filters disagree and
/// returns false.
template <int D, int A>
[[nodiscard]] bool
benchmarkCase(Options const & options)
{
    using EmrysFilter = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf>;
    using CvFilter = Ranging::OpenCvRanging<D, A>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<double>(Ranging::simulate(scenario, static_cast<int>(AgreedLength), 1));
    auto const makeEmrys = [&] {
        return EmrysFilter{scenario};
    };
    auto const makeCv = [&] {
        return CvFilter{scenario};
    };
    auto const emrysStep = [&](EmrysFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };
    auto const cvStep = [&](CvFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };

    // The agreement pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto cv = makeCv();
        Benchmark::runSteps(emrys, emrysStep, 0, AgreedLength);
        Benchmark::runSteps(cv, cvStep, 0, AgreedLength);
        Distance const d = Ranging::distance(emrys.mean(), emrys.covarianceLower(), cv.mean(), cv.covariance());
        Benchmark::consume(double(emrys.mean().sum() + cv.mean().sum()));
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

    auto emrys = Benchmark::Replay{makeEmrys, emrysStep, AgreedLength};
    auto cv = Benchmark::Replay{makeCv, cvStep, AgreedLength};
    Benchmark::PairTiming const timing = Benchmark::timePair(options, emrys, cv);
    Benchmark::consume(double(emrys.filter().mean().sum() + cv.filter().mean().sum()));

    char label[LabelWidth + 1];
    std::snprintf(label, sizeof label, "%dD %3d", D, A);
    Benchmark::printRow(label, LabelWidth, timing);
    return true;
}

/// Prints the row of the long-run table for one case: OpenCV's distance from EmrysFilter in double
/// at steps 1000, 2000, and 3000, and EmrysFilter's largest distance from its long double self.
template <int D, int A>
void
longRunRow()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, LongLength, 1);
    auto const longDoubleRanges = Ranging::rangesIn<long double>(run);
    auto reference = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, long double>{scenario};
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf>{scenario};
    auto cv = Ranging::OpenCvRanging<D, A>{scenario};
    auto worstEmrys = Distance{0, 0};
    std::printf("%dD %3d ", D, A);
    for (int k = 0; k < LongLength; ++k)
    {
        reference.predict();
        emrys.predict();
        cv.predict();
        reference.update(longDoubleRanges[std::size_t(k)]);
        emrys.update(run.ranges[std::size_t(k)]);
        cv.update(run.ranges[std::size_t(k)]);
        worstEmrys = Ranging::largerOf(
            worstEmrys,
            Ranging::distance(
                emrys.mean(),
                emrys.covarianceLower(),
                reference.mean().template cast<double>().eval(),
                reference.covarianceLower().template cast<double>().eval()
            )
        );
        if ((k + 1) % 1000 == 0)
        {
            Distance const d = Ranging::distance(cv.mean(), cv.covariance(), emrys.mean(), emrys.covarianceLower());
            std::printf("   %9.2e %9.2e", d.mean, d.cov);
        }
    }
    std::printf("    %9.2e %9.2e\n", worstEmrys.mean, worstEmrys.cov);
    std::fflush(stdout);
}

/// Prints the row of the float table for one case: the largest distance of each library's float EKF
/// from the double EmrysFilter EKF over the timed run.
template <int D, int A>
void
floatRow()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, static_cast<int>(AgreedLength), 1);
    auto const floatRanges = Ranging::rangesIn<float>(run);
    auto reference = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf>{scenario};
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, float>{scenario};
    auto cv = Ranging::OpenCvRanging<D, A, float>{scenario};
    auto worstEmrys = Distance{0, 0};
    auto worstCv = Distance{0, 0};
    for (std::size_t k = 0; k < AgreedLength; ++k)
    {
        reference.predict();
        emrys.predict();
        cv.predict();
        reference.update(run.ranges[k]);
        emrys.update(floatRanges[k]);
        cv.update(floatRanges[k]);
        auto const refCov = reference.covarianceLower();
        worstEmrys = Ranging::largerOf(
            worstEmrys,
            Ranging::distance(emrys.mean(), emrys.covarianceLower(), reference.mean(), refCov)
        );
        worstCv = Ranging::largerOf(worstCv, Ranging::distance(cv.mean(), cv.covariance(), reference.mean(), refCov));
    }
    std::printf(
        "%dD %3d    %10.2e %10.2e    %10.2e %10.2e\n",
        D,
        A,
        worstEmrys.mean,
        worstEmrys.cov,
        worstCv.mean,
        worstCv.cov
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
        Benchmark::printPreamble("Ranging EKF in double, analytic Jacobian", "OpenCV");
        std::printf(
            "Nanoseconds per predict-update step, %d batches of each filter per case, over runs of %zu steps.\n\n",
            options.samples,
            AgreedLength
        );
        std::printf("Case: dimension and anchor count. Ratio: EmrysFilter's time over OpenCV's.\n\n");
        Benchmark::printColumns("case", LabelWidth, "OpenCV");
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

    std::printf("\nThe EKF in double over %d steps, not timed: OpenCV's distance from EmrysFilter ", LongLength);
    std::printf("after each thousand steps, and EmrysFilter's largest distance from its long double self, ");
    std::printf("relative to the size of the mean and of the covariance.\n\n");
    std::printf(
        "%-*s    %-19s   %-19s   %-19s    %s\n",
        LabelWidth,
        "",
        "OpenCV at 1000",
        "at 2000",
        "at 3000",
        "EmrysFilter, worst"
    );
    std::printf(
        "%-*s    %9s %9s   %9s %9s   %9s %9s    %9s %9s\n",
        LabelWidth,
        "case",
        "mean",
        "cov",
        "mean",
        "cov",
        "mean",
        "cov",
        "mean",
        "cov"
    );
    longRunRow<2, 4>();
    longRunRow<2, 8>();
    longRunRow<2, 16>();
    longRunRow<3, 6>();
    longRunRow<3, 12>();
    longRunRow<3, 24>();

    std::printf(
        "\nThe EKF in float, not timed: the largest distance from the double EmrysFilter EKF over %zu steps, ",
        AgreedLength
    );
    std::printf("relative to the size of its mean and of its covariance.\n\n");
    std::printf("%-*s    %-21s    %s\n", LabelWidth, "", "EmrysFilter", "OpenCV");
    std::printf("%-*s    %10s %10s    %10s %10s\n", LabelWidth, "case", "mean", "covariance", "mean", "covariance");
    floatRow<2, 4>();
    floatRow<2, 8>();
    floatRow<2, 16>();
    floatRow<3, 6>();
    floatRow<3, 12>();
    floatRow<3, 24>();
    return ok ? 0 : 1;
}
