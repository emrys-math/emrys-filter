// The ranging benchmark: times the EmrysFilter EKF of emrys_ranging.h against the plain-Eigen EKF
// of eigen_ranging.h, over planar and spatial cases with growing anchor counts, both
// linearizations, and double and float, and reports the ratio of their time per predict-update
// step.
//
// Each case first runs both filters through the same simulated run and checks that they agree, as
// the agreement test does, and stops if they don't: a timing of two different filters means
// nothing. It then times batches of steps through the pair driver of harness.h, alternating
// between the two filters, and reports the median, 99th percentile, and worst batch as nanoseconds
// per step.
//
// With --count, it only runs the agreement pass, one pass of each filter through runSteps over the
// whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_ranging_benchmark
// --count
//
// Each instantiation of runSteps then carries the instructions of TrajectoryLength steps of one
// filter in one case.

#include "../harness.h"
#include "eigen_ranging.h"
#include "emrys_ranging.h"
#include "scenario.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace
{

using Benchmark::Options;
using Ranging::EigenRanging;
using Ranging::EmrysRanging;
using Ranging::Jacobian;

/// The steps in a simulated run; a timed filter starts over from the initial estimate at its end.
constexpr std::size_t TrajectoryLength = 1000;

/// The width of the table's first column: the dimension, the anchor count, the linearization, and
/// the scalar.
constexpr int LabelWidth = 25;

template <int D, int A, Jacobian K, typename S>
[[nodiscard]] auto const &
meanOf(EmrysRanging<D, A, K, S> const & filter)
{
    return filter.state().mean();
}

template <int D, int A, Jacobian K, typename S>
[[nodiscard]] auto const &
covOf(EmrysRanging<D, A, K, S> const & filter)
{
    return filter.state().covarianceLower();
}

template <int D, int A, Jacobian K, typename S>
[[nodiscard]] auto const &
meanOf(EigenRanging<D, A, K, S> const & filter)
{
    return filter.mean();
}

template <int D, int A, Jacobian K, typename S>
[[nodiscard]] auto const &
covOf(EigenRanging<D, A, K, S> const & filter)
{
    return filter.covarianceLower();
}

/// The name of a linearization, for the table.
[[nodiscard]] char const *
nameOf(Jacobian kind)
{
    return kind == Jacobian::Analytic ? "analytic" : "autodiff";
}

/// The name of a scalar, for the table.
template <typename Scalar>
[[nodiscard]] char const *
nameOf()
{
    return std::is_same_v<Scalar, float> ? "float" : "double";
}

/// Benchmarks one case and prints its row, or reports that the filters disagree and returns false.
template <int D, int A, Jacobian K, typename Scalar>
[[nodiscard]] bool
benchmarkCase(Options const & options)
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, static_cast<int>(TrajectoryLength), 1));
    auto const makeEmrys = [&] {
        return EmrysRanging<D, A, K, Scalar>{scenario};
    };
    auto const makeEigen = [&] {
        return EigenRanging<D, A, K, Scalar>{scenario};
    };
    auto const step = [&](auto & filter, std::size_t k) {
        filter.predict();
        filter.update(ranges[k]);
    };

    // The agreement pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto eigen = makeEigen();
        Benchmark::runSteps(emrys, step, 0, TrajectoryLength);
        Benchmark::runSteps(eigen, step, 0, TrajectoryLength);
        auto const covDifference = (covOf(emrys) - covOf(eigen)).template cast<double>().eval();
        double const meanError = (meanOf(emrys) - meanOf(eigen)).template cast<double>().norm()
                               / std::max(1.0, double(meanOf(eigen).norm()));
        double const covError = covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm()
                              / std::max(1.0, double(covOf(eigen).norm()));
        Benchmark::consume(double(meanOf(emrys).sum() + meanOf(eigen).sum()));
        double const tolerance = Ranging::agreementTolerance<Scalar>;
        if (!(meanError < tolerance && covError < tolerance))
        {
            std::printf(
                "%dD, %2d anchors, %s, %s: the filters disagree (mean %.3g, covariance %.3g relative)\n",
                D,
                A,
                nameOf(K),
                nameOf<Scalar>(),
                meanError,
                covError
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
    Benchmark::PairTiming const timing = Benchmark::timePair(options, emrys, eigen);
    Benchmark::consume(double(meanOf(emrys.filter()).sum() + meanOf(eigen.filter()).sum()));

    char label[LabelWidth + 1];
    std::snprintf(label, sizeof label, "%dD %3d   %-8s  %-6s", D, A, nameOf(K), nameOf<Scalar>());
    Benchmark::printRow(label, LabelWidth, timing);
    return true;
}

/// Benchmarks a dimension and anchor count under both linearizations, in double and in float.
template <int D, int A>
[[nodiscard]] bool
benchmarkSize(Options const & options)
{
    bool const analytic = benchmarkCase<D, A, Jacobian::Analytic, double>(options);
    bool const autodiff = benchmarkCase<D, A, Jacobian::AutoDiff, double>(options);
    bool const analyticFloat = benchmarkCase<D, A, Jacobian::Analytic, float>(options);
    bool const autodiffFloat = benchmarkCase<D, A, Jacobian::AutoDiff, float>(options);
    return analytic && autodiff && analyticFloat && autodiffFloat;
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
        Benchmark::printPreamble("Ranging EKF");
        std::printf("Nanoseconds per predict-update step, %d batches of each filter per case.\n\n", options.samples);
        std::printf("Case: dimension and anchor count. Ratio: EmrysFilter's time over Eigen's.\n\n");
        Benchmark::printColumns("case     Jacobian  scalar", LabelWidth);
    }

    bool ok = true;
    ok = benchmarkSize<2, 4>(options) && ok;
    ok = benchmarkSize<2, 8>(options) && ok;
    ok = benchmarkSize<2, 16>(options) && ok;
    ok = benchmarkSize<3, 6>(options) && ok;
    ok = benchmarkSize<3, 12>(options) && ok;
    ok = benchmarkSize<3, 24>(options) && ok;
    return ok ? 0 : 1;
}
