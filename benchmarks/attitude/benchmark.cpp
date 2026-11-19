// The attitude benchmark: times the EmrysFilter multiplicative EKF of emrys_attitude.h against the
// plain-Eigen one of eigen_attitude.h, under both linearizations of the reference vectors, and
// reports the ratio of their time per step, a prediction through the gyroscope and an update by
// each of the two reference vectors. It does so twice: against the plain-Eigen EKF that
// factorizes the innovation covariance as the library does, which measures the library's
// overhead, and then against the one that factorizes it by Eigen's LLT, as a user of Eigen writes
// it.
//
// Each case first runs the three filters through the same simulated run and checks that they
// agree, as the agreement test does, and stops if they don't: a timing of two different filters
// means nothing. It then times batches of steps through the pair driver of harness.h, alternating
// between the two filters, and reports the median, 99th percentile, and worst batch as nanoseconds
// per step.
//
// With --count, it only runs the agreement pass, one pass of each filter through runSteps over the
// whole run, for Callgrind:
//
//     valgrind --tool=callgrind --toggle-collect='*runSteps*' emrys_filter_attitude_benchmark
// --count
//
// Each instantiation of runSteps then carries the instructions of TrajectoryLength steps of one
// filter in one case.

#include "../harness.h"
#include "eigen_attitude.h"
#include "emrys_attitude.h"
#include "scenario.h"

#include <cstddef>
#include <cstdio>

namespace
{

using Benchmark::Options;
using Orientation::EigenAttitude;
using Orientation::EmrysAttitude;
using Orientation::Factorization;
using Orientation::Jacobian;

/// The steps in a simulated run; a timed filter starts over from the initial estimate at its end.
constexpr std::size_t TrajectoryLength = 1000;

/// The width of the table's first column: the linearization.
constexpr int LabelWidth = 8;

/// The name of a linearization, for the table.
[[nodiscard]] char const *
nameOf(Jacobian kind)
{
    return kind == Jacobian::Analytic ? "analytic" : "autodiff";
}

/// Whether \a eigen ended the run where \a emrys did, to 1e-9; reports the case and the
/// differences when it did not.
template <typename Library, typename Reference>
[[nodiscard]] bool
agree(char const * label, Library const & emrys, Reference const & eigen)
{
    auto const & mean = emrys.state().mean();
    auto const attitude = ::Eigen::Quaterniond{::Eigen::Vector4d{mean.template head<4>()}};
    double const attitudeError = Emrys::UnitQuaternion::log(eigen.attitude().conjugate() * attitude).norm();
    double const biasError = (mean.template tail<3>() - eigen.bias()).norm() / eigen.bias().norm();
    Orientation::Scenario::DifferenceMatrix const difference =
        (emrys.state().covarianceLower() - eigen.covarianceLower()).template triangularView<::Eigen::Lower>();
    double const covError = difference.norm() / eigen.covarianceLower().norm();
    Benchmark::consume(mean.sum() + eigen.attitude().w());
    if (attitudeError < 1e-9 && biasError < 1e-9 && covError < 1e-9)
    {
        return true;
    }
    std::printf(
        "%s: the filters disagree (attitude %.3g rad, bias %.3g, covariance %.3g relative)\n",
        label,
        attitudeError,
        biasError,
        covError
    );
    return false;
}

/// Benchmarks one case, prints its row of the first table and keeps its timing against the
/// plain-Eigen EKF with Eigen's LLT in \a lltTiming, or reports that the filters disagree and
/// returns false.
template <Jacobian K>
[[nodiscard]] bool
benchmarkCase(Options const & options, Benchmark::PairTiming & lltTiming)
{
    auto const scenario = Orientation::makeScenario();
    auto const run = Orientation::simulate(scenario, static_cast<int>(TrajectoryLength), 1);
    auto const makeEmrys = [&] {
        return EmrysAttitude<K>{scenario};
    };
    auto const makeEigen = [&] {
        return EigenAttitude<K>{scenario};
    };
    auto const makeEigenLlt = [&] {
        return EigenAttitude<K, Factorization::EigenLlt>{scenario};
    };
    auto const step = [&](auto & filter, std::size_t k) {
        filter.predict(run.gyro[k]);
        filter.update(run.gravity[k], run.magneticField[k]);
    };

    // The agreement pass: one pass of each filter over the whole run, then a comparison.
    {
        auto emrys = makeEmrys();
        auto eigen = makeEigen();
        auto eigenLlt = makeEigenLlt();
        Benchmark::runSteps(emrys, step, 0, TrajectoryLength);
        Benchmark::runSteps(eigen, step, 0, TrajectoryLength);
        Benchmark::runSteps(eigenLlt, step, 0, TrajectoryLength);
        bool const agrees = agree(nameOf(K), emrys, eigen);
        bool const agreesByLlt = agree(nameOf(K), emrys, eigenLlt);
        if (!(agrees && agreesByLlt))
        {
            return false;
        }
    }
    if (options.count)
    {
        return true;
    }

    auto emrys = Benchmark::Replay{makeEmrys, step, TrajectoryLength};
    auto eigen = Benchmark::Replay{makeEigen, step, TrajectoryLength};
    auto eigenLlt = Benchmark::Replay{makeEigenLlt, step, TrajectoryLength};
    Benchmark::PairTiming const timing = Benchmark::timePair(options, emrys, eigen);
    lltTiming = Benchmark::timePair(options, emrys, eigenLlt);
    Benchmark::consume(
        emrys.filter().state().mean().sum() + eigen.filter().attitude().w() + eigenLlt.filter().attitude().w()
    );
    Benchmark::printRow(nameOf(K), LabelWidth, timing);
    return true;
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
        Benchmark::printPreamble("Attitude EKF");
        std::printf(
            "Nanoseconds per step (a prediction and two reference-vector updates), %d batches of each filter "
            "per case.\n\n",
            options.samples
        );
        std::printf("Ratio: EmrysFilter's time over Eigen's.\n\n");
        Benchmark::printColumns("Jacobian", LabelWidth);
    }

    auto analyticLlt = Benchmark::PairTiming{};
    auto autodiffLlt = Benchmark::PairTiming{};
    bool const analytic = benchmarkCase<Jacobian::Analytic>(options, analyticLlt);
    bool const autodiff = benchmarkCase<Jacobian::AutoDiff>(options, autodiffLlt);
    if (!options.count)
    {
        std::printf("\nAgainst the plain-Eigen EKF that factorizes the innovation covariance by Eigen's LLT.\n\n");
        Benchmark::printColumns("Jacobian", LabelWidth, "Eigen, LLT");
        if (analytic)
        {
            Benchmark::printRow(nameOf(Jacobian::Analytic), LabelWidth, analyticLlt);
        }
        if (autodiff)
        {
            Benchmark::printRow(nameOf(Jacobian::AutoDiff), LabelWidth, autodiffLlt);
        }
    }
    return analytic && autodiff ? 0 : 1;
}
