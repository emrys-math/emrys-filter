// The ranging benchmark against mherb/kalman: times each EmrysFilter filter of
// emrys_ranging_kinds.h against mherb/kalman's of mherb_ranging.h, over the planar and spatial
// cases of benchmark.cpp, with the analytic Jacobian, the only one mherb/kalman has, and reports
// the ratio of their time per predict-update step. benchmark.cpp times the EmrysFilter EKF against
// the plain-Eigen one, so the two tables together place the three.
//
// Each case first runs both filters through the same simulated run and compares them, as
// mherb_agreement.cpp does, and stops if the comparison fails. It then times batches of steps
// through the pair driver of harness.h, alternating between the two filters. The EKF in double and
// the square-root EKF in double and in float must agree (cases.h, MherbAgreementCases). The other
// pairs do not agree and are timed all the same, their rows marked [benchmarks], since
// mherb_agreement.cpp shows each difference to be mherb/kalman's; they must stay within
// disagreementBound. Three tables follow for them. The first gives each library's float EKF's
// largest distance from the double EmrysFilter EKF over the run, since mherb/kalman's explicit
// inverse of S loses its accuracy in float. The second gives the largest distance between the two
// libraries' UKFs in double, in the covariance and the square-root form, which differ by
// mherb/kalman's algorithm: its update uses the sigma points its prediction moved. The third
// gives each library's float UKF's largest distance from its own double UKF.
//
// With --count, it only runs the comparison pass of each pair, one pass of each filter through
// runSteps over the whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_mherb_benchmark --count

#include "../harness.h"
#include "distance.h"
#include "emrys_ranging_kinds.h"
#include "mherb_ranging.h"
#include "scenario.h"

#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace
{

using Benchmark::Options;
using Ranging::Distance;
using Ranging::EmrysRangingKind;
using Ranging::FilterKind;
using Ranging::MherbRanging;

/// The steps in a simulated run; a timed filter starts over from the initial estimate at its end.
constexpr std::size_t TrajectoryLength = 1000;

/// The width of the timing table's first column: the dimension, the anchor count, the filter, the
/// scalar, and the mark of a pair that does not agree.
constexpr int LabelWidth = 27;

/// Whether the two libraries' filters \a Kind agree in \a Scalar (cases.h): the EKF in double, and
/// the square-root EKF in both scalars. The other pairs are timed with a mark.
template <FilterKind Kind, typename Scalar>
constexpr bool Agrees = Kind == FilterKind::SqrtEkf || (Kind == FilterKind::Ekf && std::is_same_v<Scalar, double>);

/// The width of the other tables' first column: the dimension and the anchor count.
constexpr int CaseWidth = 8;

/// The name of a filter, for the tables.
[[nodiscard]] char const *
nameOf(FilterKind kind)
{
    switch (kind)
    {
        case FilterKind::Ekf:
            return "EKF";
        case FilterKind::Ukf:
            return "UKF";
        case FilterKind::SqrtEkf:
            return "sqrt EKF";
        case FilterKind::SqrtUkf:
            return "sqrt UKF";
    }
    return "";
}

/// Benchmarks one case and prints its row, marked if the pair is one that does not agree. Reports
/// that the filters disagree, or that those of a marked pair are further apart than
/// disagreementBound, and returns false.
template <int D, int A, FilterKind Kind, typename Scalar>
[[nodiscard]] bool
benchmarkCase(Options const & options)
{
    using EmrysFilter = EmrysRangingKind<D, A, Kind, Scalar>;
    using MherbFilter = MherbRanging<D, A, Kind, Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1));
    auto const mherbRanges = MherbFilter::rangesOf(ranges);
    auto const makeEmrys = [&] {
        return EmrysFilter{scenario};
    };
    auto const makeMherb = [&] {
        return MherbFilter{scenario};
    };
    auto const emrysStep = [&](EmrysFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };
    auto const mherbStep = [&](MherbFilter & filter, std::size_t k) {
        filter.predict();
        filter.update(mherbRanges[k]);
    };

    // The comparison pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto mherb = makeMherb();
        Benchmark::runSteps(emrys, emrysStep, 0, TrajectoryLength);
        Benchmark::runSteps(mherb, mherbStep, 0, TrajectoryLength);
        Distance const d = Ranging::distance(emrys.mean(), emrys.covarianceLower(), mherb.mean(), mherb.covariance());
        Benchmark::consume(double(emrys.mean().sum() + mherb.mean().sum()));
        double const tolerance =
            Agrees<Kind, Scalar> ? Ranging::agreementTolerance<Scalar> : Ranging::disagreementBound;
        if (!(d.mean < tolerance && d.cov < tolerance))
        {
            std::printf(
                "%dD, %2d anchors, %s, %s: the filters %s (mean %.3g, covariance %.3g relative)\n",
                D,
                A,
                nameOf(Kind),
                Ranging::scalarName<Scalar>(),
                Agrees<Kind, Scalar> ? "disagree" : "are further apart than a marked pair may be",
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
    Benchmark::consume(double(emrys.filter().mean().sum() + mherb.filter().mean().sum()));

    char label[LabelWidth + 1];
    std::snprintf(
        label,
        sizeof label,
        "%dD %3d   %-9s %-6s%s",
        D,
        A,
        nameOf(Kind),
        Ranging::scalarName<Scalar>(),
        Agrees<Kind, Scalar> ? "" : " *"
    );
    Benchmark::printRow(label, LabelWidth, timing);
    return true;
}

/// Benchmarks the filter \a Kind at a dimension and anchor count, in double and in float.
template <int D, int A, FilterKind Kind>
[[nodiscard]] bool
benchmarkKind(Options const & options)
{
    bool const inDouble = benchmarkCase<D, A, Kind, double>(options);
    bool const inFloat = benchmarkCase<D, A, Kind, float>(options);
    return inDouble && inFloat;
}

/// Benchmarks every pair at a dimension and anchor count.
template <int D, int A>
[[nodiscard]] bool
benchmarkSize(Options const & options)
{
    bool const ekf = benchmarkKind<D, A, FilterKind::Ekf>(options);
    bool const sqrtEkf = benchmarkKind<D, A, FilterKind::SqrtEkf>(options);
    bool const ukf = benchmarkKind<D, A, FilterKind::Ukf>(options);
    bool const sqrtUkf = benchmarkKind<D, A, FilterKind::SqrtUkf>(options);
    return ekf && sqrtEkf && ukf && sqrtUkf;
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
    auto const mherbRanges = MherbRanging<D, A, FilterKind::Ekf, float>::rangesOf(floatRanges);
    auto reference = EmrysRangingKind<D, A, FilterKind::Ekf>{scenario};
    auto emrys = EmrysRangingKind<D, A, FilterKind::Ekf, float>{scenario};
    auto mherb = MherbRanging<D, A, FilterKind::Ekf, float>{scenario};
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
        auto const refCov = reference.covarianceLower();
        worstEmrys = Ranging::largerOf(
            worstEmrys,
            Ranging::distance(emrys.mean(), emrys.covarianceLower(), reference.mean(), refCov)
        );
        worstMherb = Ranging::largerOf(
            worstMherb,
            Ranging::distance(mherb.mean(), mherb.covariance(), reference.mean(), refCov)
        );
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

/// The largest distance between the two libraries' filters \a Kind over the run, in double.
template <int D, int A, FilterKind Kind>
[[nodiscard]] Distance
largestDifference()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<double>(Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1));
    auto const mherbRanges = MherbRanging<D, A, Kind>::rangesOf(ranges);
    auto emrys = EmrysRangingKind<D, A, Kind>{scenario};
    auto mherb = MherbRanging<D, A, Kind>{scenario};
    auto worst = Distance{0, 0};
    for (std::size_t k = 0; k < TrajectoryLength; ++k)
    {
        emrys.predict();
        mherb.predict();
        emrys.update(ranges[k]);
        mherb.update(mherbRanges[k]);
        worst = Ranging::largerOf(
            worst,
            Ranging::distance(mherb.mean(), mherb.covariance(), emrys.mean(), emrys.covarianceLower())
        );
    }
    return worst;
}

/// Prints the row of the UKF table for one case: the largest distance between the two libraries'
/// UKFs, in each form.
template <int D, int A>
void
unscentedRow()
{
    Distance const ukf = largestDifference<D, A, FilterKind::Ukf>();
    Distance const sqrtUkf = largestDifference<D, A, FilterKind::SqrtUkf>();
    std::printf("%dD %3d    %10.2e %10.2e    %10.2e %10.2e\n", D, A, ukf.mean, ukf.cov, sqrtUkf.mean, sqrtUkf.cov);
    std::fflush(stdout);
}

/// Prints the row of a float UKF table for one case: the largest distance of each library's float
/// filter \a Kind from its own double one over the run.
template <int D, int A, FilterKind Kind>
void
floatUnscentedRow()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1);
    auto const floatRanges = Ranging::rangesIn<float>(run);
    auto const mherbRanges = MherbRanging<D, A, Kind>::rangesOf(Ranging::rangesIn<double>(run));
    auto const mherbFloatRanges = MherbRanging<D, A, Kind, float>::rangesOf(floatRanges);
    // The step of a filter that takes its ranges from \a ranges.
    auto const stepBy = [](auto const & ranges) {
        return [&ranges](auto & filter, std::size_t k) {
            filter.predict();
            filter.update(ranges[k]);
        };
    };
    auto emrys = EmrysRangingKind<D, A, Kind, float>{scenario};
    auto emrysDouble = EmrysRangingKind<D, A, Kind>{scenario};
    Distance const ofEmrys =
        Ranging::largestDistance(emrys, stepBy(floatRanges), emrysDouble, stepBy(run.ranges), TrajectoryLength);
    auto mherb = MherbRanging<D, A, Kind, float>{scenario};
    auto mherbDouble = MherbRanging<D, A, Kind>{scenario};
    Distance const ofMherb =
        Ranging::largestDistance(mherb, stepBy(mherbFloatRanges), mherbDouble, stepBy(mherbRanges), TrajectoryLength);
    std::printf(
        "%dD %3d    %10.2e %10.2e    %10.2e %10.2e\n",
        D,
        A,
        ofEmrys.mean,
        ofEmrys.cov,
        ofMherb.mean,
        ofMherb.cov
    );
    std::fflush(stdout);
}

/// Prints the float UKF table of the filter \a Kind, headed by \a name.
template <FilterKind Kind>
void
printFloatUnscentedTable(char const * name);

/// Prints the headings of a table of distances whose two column pairs are headed \a first and
/// \a second.
void
printDistanceColumns(char const * first, char const * second)
{
    std::printf("%-*s    %-21s    %s\n", CaseWidth, "", first, second);
    std::printf("%-*s    %10s %10s    %10s %10s\n", CaseWidth, "case", "mean", "covariance", "mean", "covariance");
}

template <FilterKind Kind>
void
printFloatUnscentedTable(char const * name)
{
    std::printf(
        "\nThe %s in float: each library's largest distance from its own %s in double over %zu steps, ",
        name,
        name,
        TrajectoryLength
    );
    std::printf("relative to the size of that one's mean and covariance.\n\n");
    printDistanceColumns("EmrysFilter", "mherb/kalman");
    floatUnscentedRow<2, 4, Kind>();
    floatUnscentedRow<2, 8, Kind>();
    floatUnscentedRow<2, 16, Kind>();
    floatUnscentedRow<3, 6, Kind>();
    floatUnscentedRow<3, 12, Kind>();
    floatUnscentedRow<3, 24, Kind>();
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
        Benchmark::printPreamble("Ranging Kalman filters, analytic Jacobian", "mherb/kalman");
        std::printf("Nanoseconds per predict-update step, %d batches of each filter per case.\n\n", options.samples);
        std::printf("Case: dimension and anchor count. Ratio: EmrysFilter's time over mherb/kalman's.\n");
        std::printf("* The pair does not agree, and the difference is mherb/kalman's: the tables below give it.\n\n");
        Benchmark::printColumns("case     filter    scalar", LabelWidth, "mherb/kalman");
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
    printDistanceColumns("EmrysFilter", "mherb/kalman");
    floatRow<2, 4>();
    floatRow<2, 8>();
    floatRow<2, 16>();
    floatRow<3, 6>();
    floatRow<3, 12>();
    floatRow<3, 24>();

    std::printf("\nThe UKFs: the largest distance between the two libraries' filters over %zu ", TrajectoryLength);
    std::printf("steps, in double, relative to the size of EmrysFilter's mean and covariance.\n");
    std::printf("mherb/kalman updates with the sigma points its prediction moved, before the process noise.\n\n");
    printDistanceColumns("UKF", "square-root UKF");
    unscentedRow<2, 4>();
    unscentedRow<2, 8>();
    unscentedRow<2, 16>();
    unscentedRow<3, 6>();
    unscentedRow<3, 12>();
    unscentedRow<3, 24>();

    printFloatUnscentedTable<FilterKind::Ukf>("UKF");
    printFloatUnscentedTable<FilterKind::SqrtUkf>("square-root UKF");
    return ok ? 0 : 1;
}
