// The ranging benchmark against mherb/kalman: times the EmrysFilter EKF of emrys_ranging.h against
// mherb/kalman's ExtendedKalmanFilter of mherb_ranging.h, over the planar and spatial cases of
// benchmark.cpp, with the analytic Jacobian, the only one mherb/kalman has, and reports the ratio of
// their time per predict-update step. benchmark.cpp times the same EmrysFilter EKF against the
// plain-Eigen one, so the two tables together place the three.
//
// Each case first runs both filters through the same simulated run and checks that they agree, as
// mherb_agreement.cpp does, and stops if they don't. It then times batches of steps through the
// pair driver of harness.h, alternating between the two filters. The timing is in double only: in
// float the two do not agree, since mherb/kalman forms the gain through an explicit inverse of S,
// whose error the prior covariance amplifies. A second table gives each library's float filter's
// largest distance from the double EmrysFilter EKF over the run instead.
//
// With --count, it only runs the agreement pass, one pass of each filter through runSteps over the
// whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_mherb_benchmark --count

#include "../harness.h"
#include "emrys_ranging.h"
#include "mherb_ranging.h"
#include "scenario.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace
{

using Benchmark::Options;
using Ranging::EmrysRanging;
using Ranging::Jacobian;
using Ranging::MherbRanging;

/// The steps in a simulated run; a timed filter starts over from the initial estimate at its end.
constexpr std::size_t TrajectoryLength = 1000;

/// The width of the tables' first column: the dimension and the anchor count.
constexpr int LabelWidth = 8;

/// The distance of the mean \a mean and the covariance \a cov from the reference's \a refMean and
/// \a refCov, each relative to the size of the reference's, as the agreement tests measure it. The
/// covariances are compared on their lower triangles, the part EmrysFilter guarantees.
struct Distance
{
    double mean;  ///< of the means
    double cov;   ///< of the covariances' lower triangles
};

template <typename Mean, typename Cov, typename RefMean, typename RefCov>
[[nodiscard]] Distance
distance(Mean const & mean, Cov const & cov, RefMean const & refMean, RefCov const & refCov)
{
    auto const covDifference = (cov.template cast<double>() - refCov.template cast<double>()).eval();
    return Distance{
        (mean.template cast<double>() - refMean.template cast<double>()).norm() / std::max(1.0, double(refMean.norm())),
        covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm()
            / std::max(1.0, double(refCov.norm()))
    };
}

/// Benchmarks one case in double and prints its row, or reports that the filters disagree and
/// returns false.
template <int D, int A>
[[nodiscard]] bool
benchmarkCase(Options const & options)
{
    using EmrysEkf = EmrysRanging<D, A, Jacobian::Analytic>;
    using MherbEkf = MherbRanging<D, A>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<double>(Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1));
    auto const mherbRanges = MherbEkf::rangesOf(ranges);
    auto const makeEmrys = [&] {
        return EmrysEkf{scenario};
    };
    auto const makeMherb = [&] {
        return MherbEkf{scenario};
    };
    auto const emrysStep = [&](EmrysEkf & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };
    auto const mherbStep = [&](MherbEkf & filter, std::size_t k) {
        filter.predict();
        filter.update(mherbRanges[k]);
    };

    // The agreement pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto mherb = makeMherb();
        Benchmark::runSteps(emrys, emrysStep, 0, TrajectoryLength);
        Benchmark::runSteps(mherb, mherbStep, 0, TrajectoryLength);
        Distance const d =
            distance(emrys.state().mean(), emrys.state().covarianceLower(), mherb.mean(), mherb.covariance());
        Benchmark::consume(emrys.state().mean().sum() + mherb.mean().sum());
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
    auto mherb = Benchmark::Replay{makeMherb, mherbStep, TrajectoryLength};
    Benchmark::PairTiming const timing = Benchmark::timePair(options, emrys, mherb);
    Benchmark::consume(emrys.filter().state().mean().sum() + mherb.filter().mean().sum());

    char label[LabelWidth + 1];
    std::snprintf(label, sizeof label, "%dD %3d", D, A);
    Benchmark::printRow(label, LabelWidth, timing);
    return true;
}

/// Prints the row of the float table for one case: the largest distance of each library's float
/// filter from the double EmrysFilter EKF over the run.
template <int D, int A>
void
accuracyRow()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1);
    auto const floatRanges = Ranging::rangesIn<float>(run);
    auto const mherbRanges = MherbRanging<D, A, float>::rangesOf(floatRanges);
    auto reference = EmrysRanging<D, A, Jacobian::Analytic>{scenario};
    auto emrys = EmrysRanging<D, A, Jacobian::Analytic, float>{scenario};
    auto mherb = MherbRanging<D, A, float>{scenario};
    auto worstEmrys = Distance{0, 0};
    auto worstMherb = Distance{0, 0};
    for (std::size_t k = 0; k < TrajectoryLength; ++k)
    {
        reference.predict();
        emrys.predict();
        mherb.predict();
        reference.update(run.ranges[k]);
        emrys.update(floatRanges[k]);
        mherb.update(mherbRanges[k]);
        auto const & refMean = reference.state().mean();
        auto const & refCov = reference.state().covarianceLower();
        Distance const e = distance(emrys.state().mean(), emrys.state().covarianceLower(), refMean, refCov);
        Distance const m = distance(mherb.mean(), mherb.covariance(), refMean, refCov);
        worstEmrys = Distance{std::max(worstEmrys.mean, e.mean), std::max(worstEmrys.cov, e.cov)};
        worstMherb = Distance{std::max(worstMherb.mean, m.mean), std::max(worstMherb.cov, m.cov)};
    }
    std::printf(
        "%dD %3d    %10.2e %10.2e    %10.2e %10.2e\n",
        D,
        A,
        worstEmrys.mean,
        worstEmrys.cov,
        worstMherb.mean,
        worstMherb.cov
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
        Benchmark::printPreamble("Ranging EKF, analytic Jacobian, double", "mherb/kalman");
        std::printf("Nanoseconds per predict-update step, %d batches of each filter per case.\n\n", options.samples);
        std::printf("Case: dimension and anchor count. Ratio: EmrysFilter's time over mherb/kalman's.\n\n");
        Benchmark::printColumns("case", LabelWidth, "mherb/kalman");
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

    std::printf("\nIn float: the largest distance from the double EmrysFilter EKF over %zu steps, ", TrajectoryLength);
    std::printf("relative to the size of its mean and of its covariance.\n\n");
    std::printf("%-*s    %-21s    %s\n", LabelWidth, "", "EmrysFilter", "mherb/kalman");
    std::printf("%-*s    %10s %10s    %10s %10s\n", LabelWidth, "case", "mean", "covariance", "mean", "covariance");
    accuracyRow<2, 4>();
    accuracyRow<2, 8>();
    accuracyRow<2, 16>();
    accuracyRow<3, 6>();
    accuracyRow<3, 12>();
    accuracyRow<3, 24>();
    return ok ? 0 : 1;
}
