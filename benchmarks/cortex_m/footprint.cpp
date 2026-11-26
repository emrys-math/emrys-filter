// One implementation of the ranging EKF on a Cortex-M, for the footprint table of claim 6
// [embedded-builds]: the case of 3D with 6 anchors in float with the analytic Jacobian, stepped
// over a simulated run, by the implementation FOOTPRINT_FILTER names: 1 EmrysFilter, 2 the
// hand-written Eigen EKF of the benchmarks, 3 TinyEKF. 0 runs the simulator alone, which every
// program shares, so that a filter's footprint is its program's less that one's. A program passes
// when its estimate ends within 2 m of the true position, which each filter's does.

#include "../ranging/scenario.h"
#include "semihosting.h"

#include <cmath>

#if FOOTPRINT_FILTER == 1
#    include "../ranging/emrys_ranging.h"
#elif FOOTPRINT_FILTER == 2
#    include "../ranging/eigen_ranging.h"
#elif FOOTPRINT_FILTER == 3
#    include "../ranging/tinyekf_ranging.h"
#elif FOOTPRINT_FILTER != 0
#    error "FOOTPRINT_FILTER is 0, the simulator alone, or 1 to 3, a filter"
#endif

namespace
{

constexpr int Dimension = 3;
constexpr int AnchorCount = 6;
constexpr int Steps = 200;

#if FOOTPRINT_FILTER == 1
using Filter = Ranging::EmrysRanging<Dimension, AnchorCount, Ranging::Jacobian::Analytic, float>;
constexpr char const * Name = "EmrysFilter";
#elif FOOTPRINT_FILTER == 2
using Filter = Ranging::EigenRanging<Dimension, AnchorCount, Ranging::Jacobian::Analytic, float>;
constexpr char const * Name = "hand code";
#elif FOOTPRINT_FILTER == 3
using Filter = Ranging::TinyEkfRanging<Dimension, AnchorCount, float>;
constexpr char const * Name = "TinyEKF";
#else
constexpr char const * Name = "simulator";
#endif

#if FOOTPRINT_FILTER != 0
/// The mean of \a filter, whose accessor differs by implementation.
template <typename F>
[[nodiscard]] Eigen::Matrix<float, 2 * Dimension, 1>
meanOf(F const & filter)
{
    if constexpr (requires { filter.state().mean(); })
    {
        return filter.state().mean();
    }
    else
    {
        return filter.mean();
    }
}
#endif

}  // namespace

int
program()
{
    auto const scenario = Ranging::makeScenario<Dimension, AnchorCount>();
    auto simulator = Ranging::Simulator<Dimension, AnchorCount>{scenario, 1};
#if FOOTPRINT_FILTER == 0
    double sum = 0;
    for (int k = 0; k < Steps; ++k)
    {
        sum += simulator.step().sum();
    }
    bool const passed = std::isfinite(sum);
#else
    auto filter = Filter{scenario};
    for (int k = 0; k < Steps; ++k)
    {
        Eigen::Matrix<float, AnchorCount, 1> const z = simulator.step().template cast<float>();
        filter.predict();
        filter.update(z);
    }
    Eigen::Vector3d const position = meanOf(filter).template head<Dimension>().template cast<double>();
    double const error = (simulator.truth().template head<Dimension>() - position).norm();
    bool const passed = error < 2.0;
#endif
    Semihosting::write(passed ? "PASS " : "FAIL ");
    Semihosting::write(Name);
    Semihosting::write("\n");
    return passed ? 0 : 1;
}
