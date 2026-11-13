// The ranging benchmark's agreement tests (ranging/agreement.cpp) on a Cortex-M [embedded-builds]:
// the float cases, with no heap, no exceptions, and no RTTI, the simulation stepped as the filters
// go since there is no heap to record it in. GoogleTest needs both, so the checks are plain code
// that reports through semihosting, with the tolerances of the hosted tests.

#include "../ranging/eigen_ranging.h"
#include "../ranging/emrys_ranging.h"
#include "../ranging/scenario.h"
#include "semihosting.h"

#include <EmrysFilter/Core/ChiSquared.h>

#include <algorithm>
#include <array>

namespace
{

using Ranging::EigenRanging;
using Ranging::EmrysRanging;
using Ranging::Jacobian;

/// The norm of the lower triangle of \a difference, a difference of covariances, cast to double.
template <typename Derived>
[[nodiscard]] double
lowerNorm(Eigen::MatrixBase<Derived> const & difference)
{
    return difference.template cast<double>().template triangularView<Eigen::Lower>().toDenseMatrix().norm();
}

/// Reports check \a test of case \a name, failed at step \a step or passed when \a step is
/// negative, and returns whether it passed.
bool
report(char const * name, char const * test, int step)
{
    Semihosting::write(step < 0 ? "PASS " : "FAIL ");
    Semihosting::write(name);
    Semihosting::write(": ");
    Semihosting::write(test);
    if (step >= 0)
    {
        Semihosting::write(" at step ");
        Semihosting::write(long{step});
    }
    Semihosting::write("\n");
    return step < 0;
}

/// Both float implementations, driven through the same simulated run, agree at every step to
/// agreementTolerance<float>; returns the first step where they do not, or -1.
template <int D, int A, Jacobian Kind>
[[nodiscard]] int
implementationsAgree()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto simulator = Ranging::Simulator<D, A>{scenario, 1};
    double const tolerance = Ranging::agreementTolerance<float>;
    auto emrys = EmrysRanging<D, A, Kind, float>{scenario};
    auto eigen = EigenRanging<D, A, Kind, float>{scenario};
    for (int k = 0; k < 500; ++k)
    {
        Eigen::Matrix<float, A, 1> const z = simulator.step().template cast<float>();
        emrys.predict();
        eigen.predict();
        emrys.update(z);
        eigen.update(z);
        double const meanScale = std::max(1.0, double(eigen.mean().norm()));
        double const covScale = std::max(1.0, double(eigen.covarianceLower().norm()));
        if ((emrys.state().mean() - eigen.mean()).template cast<double>().norm() >= tolerance * meanScale
            || lowerNorm(emrys.state().covarianceLower() - eigen.covarianceLower()) >= tolerance * covScale)
        {
            return k;
        }
    }
    return -1;
}

/// The float EKF stays within 1e-4, relative to the state's and covariance's size, of the double
/// one over the same run; returns the first step where it does not, or -1.
template <int D, int A, Jacobian Kind>
[[nodiscard]] int
floatTracksDouble()
{
    auto const scenario = Ranging::makeScenario<D, A>();
    auto simulator = Ranging::Simulator<D, A>{scenario, 1};
    auto filter = EmrysRanging<D, A, Kind, float>{scenario};
    auto reference = EmrysRanging<D, A, Kind>{scenario};
    for (int k = 0; k < 500; ++k)
    {
        auto const z = simulator.step();
        filter.predict();
        reference.predict();
        filter.update(z.template cast<float>());
        reference.update(z);
        double const meanScale = std::max(1.0, reference.state().mean().norm());
        double const covScale = std::max(1.0, reference.state().covarianceLower().norm());
        if ((filter.state().mean().template cast<double>() - reference.state().mean()).norm() >= 1e-4 * meanScale
            || lowerNorm(filter.state().covarianceLower().template cast<double>() - reference.state().covarianceLower())
                   >= 1e-4 * covScale)
        {
            return k;
        }
    }
    return -1;
}

/// The float EKF is consistent: over 50 independent runs, the average NEES at every 25th step lies
/// in the two-sided 99.9% interval of a chi-squared with 50 times the state's degrees of freedom,
/// divided by 50; returns the first checked step where it does not, or -1.
template <int D, int A, Jacobian Kind>
[[nodiscard]] int
consistent()
{
    constexpr int runs = 50;
    constexpr int steps = 100;
    constexpr int checkEvery = 25;
    auto const scenario = Ranging::makeScenario<D, A>();
    double const dof = runs * 2.0 * D;
    double const lower = Emrys::chiSquaredQuantile<double>(0.0005, dof) / runs;
    double const upper = Emrys::chiSquaredQuantile<double>(0.9995, dof) / runs;

    auto nees = std::array<double, steps / checkEvery>{};
    for (int r = 0; r < runs; ++r)
    {
        auto simulator = Ranging::Simulator<D, A>{scenario, 100u + unsigned(r)};
        auto filter = EmrysRanging<D, A, Kind, float>{scenario};
        for (int k = 0; k < steps; ++k)
        {
            Eigen::Matrix<float, A, 1> const z = simulator.step().template cast<float>();
            filter.predict();
            filter.update(z);
            if ((k + 1) % checkEvery == 0)
            {
                auto const error = (simulator.truth() - filter.state().mean().template cast<double>()).eval();
                auto const cov = filter.state().covarianceLower().template cast<double>().eval();
                nees[(k + 1) / checkEvery - 1] += error.dot(cov.ldlt().solve(error)) / runs;
            }
        }
    }
    for (int c = 0; c < int(nees.size()); ++c)
    {
        if (!(nees[c] > lower && nees[c] < upper))
        {
            return (c + 1) * checkEvery;
        }
    }
    return -1;
}

/// Runs every check of one case and returns how many failed.
template <int D, int A, Jacobian Kind>
[[nodiscard]] int
runCase(char const * name)
{
    int failures = 0;
    failures += !report(name, "implementations agree", implementationsAgree<D, A, Kind>());
    failures += !report(name, "float tracks double", floatTracksDouble<D, A, Kind>());
    failures += !report(name, "consistent", consistent<D, A, Kind>());
    return failures;
}

}  // namespace

int
program()
{
    int failures = 0;
    failures += runCase<2, 4, Jacobian::Analytic>("2D, 4 anchors, analytic");
    failures += runCase<2, 4, Jacobian::AutoDiff>("2D, 4 anchors, autodiff");
    failures += runCase<3, 6, Jacobian::Analytic>("3D, 6 anchors, analytic");
    failures += runCase<3, 6, Jacobian::AutoDiff>("3D, 6 anchors, autodiff");
    Semihosting::write(failures == 0 ? "all checks passed\n" : "checks failed\n");
    return failures;
}
