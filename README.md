# EmrysFilter

**State estimation in C++23 on Eigen 5. One design from the Kalman filter to the
Rao-Blackwellized particle filter, on vectors, angles, quaternions, and discrete modes.
Header-only, allocation-free on bounded states, parallel by policy, and every algorithm cited
to the paper it implements.**

EmrysFilter is a header-only library by Emrys Applied Mathematics for sequential Bayesian
estimation: predict, update, repeat. The same state description, models, and transforms run
under a Kalman filter and its square-root and information forms, a Gaussian mixture filter, an
interacting multiple model filter, a particle filter, a grid filter, their Rao-Blackwellized
combinations, and a hidden Markov filter, with smoothers over all of them. Nothing in it is
invented here.

```cpp
#include <EmrysFilter/EmrysFilter.h>

#include <iomanip>
#include <iostream>
#include <random>

using namespace Emrys;

// The state is named blocks. These two are plain vectors; a heading is an angle.
struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
using State = Composite<Position, Velocity>;

// A radar measures a range and a bearing, an angle that wraps at plus and minus pi.
struct Range : Linear<1> {};
struct Bearing : RadiansPi<1> {};

int main()
{
    // Constant velocity, driven by white acceleration noise of spectral density 0.1.
    double const accelerationDensity = 0.1;
    auto const motion = ConstantVelocity<Position, Velocity>{accelerationDensity};
    auto const initial = GaussianParams<State>{
        ValueVector<State>{-78.0, 12.0, 0.8, -0.3},
        DifferenceVector<State>{25.0, 25.0, 1.0, 1.0}.asDiagonal()};
    auto filter = KalmanFilter{motion, initial};

    // A radar at the origin sees the Position block as range and bearing, which is
    // nonlinear. The model reads the blocks it names, wherever they sit in the state;
    // Linearized differentiates it automatically, or takes the Jacobian a function
    // returns with withJacobian().
    double const rangeVariance = 1.0;
    double const bearingVariance = 1e-4;
    Eigen::Matrix2d const radarNoise = Eigen::Vector2d{rangeVariance, bearingVariance}.asDiagonal();
    auto const radar = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            return Eigen::Matrix<T, 2, 1>{sqrt(p.squaredNorm()), atan2(p(1), p(0))};
        },
        reads(path<Position>),
        writes(path<Range>, path<Bearing>),
        additiveNoise(radarNoise));

    // A GPS fix sees the Position block itself, with correlated noise. It is linear,
    // so it is given by its matrix, and its update is the Kalman filter's exact one.
    Eigen::Matrix2d const gpsNoise{{4.0, 1.0}, {1.0, 3.0}};
    auto const gps = measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(gpsNoise));

    auto rng = std::mt19937_64{42};
    auto truth = ValueVector<State>{-80.0, 10.0, 1.0, -0.5};  // bearing wraps
    auto monitor = NisMonitor{50};

    // The filter takes a measurement from any model the same way, each with its method.
    auto const measure = [&](auto const & m) {
        auto const z = Eigen::Vector2d{simulate<State>(truth, m, rng)};
        auto const updated = filter.update(observe(methods::linearized(), m, z));
        if (updated) {
            monitor.add(*updated);
        }
        return updated.has_value();  // false: innovation covariance not positive definite
    };
    for (int k = 0; k < 50; ++k) {
        truth = simulate<State>(truth, motion, rng, 1.0);  // moves
        filter.predict(1.0);
        if (!measure(radar) || (k % 10 == 9 && !measure(gps))) {
            return 1;
        }
    }
    std::cout << std::setprecision(3)
              << filter.state().marginal<Position>().copy() << std::endl
              << "truth " << truth(idx<State, Position>()).transpose() << std::endl
              << "NIS average " << monitor.average().value() << ", consistent at 99%: "
              << std::boolalpha << (monitor.verdict(0.99) == Consistency::Consistent) << std::endl;
}
```

```
Gaussian of dimension 2
  mean: 47.1 -130
  covariance:
    0.718 0.111
    0.111 0.493
truth 46.8 -129
NIS average 1.01, consistent at 99%: true
```

(Clang 21 on the MSVC standard library; libstdc++ draws different noise from the same seed.)
Write `methods::unscented()`, `methods::iterated(methods::linearized())`, or `methods::gaussHermite<3>()` in place of the radar's
`methods::linearized()` and the same filter is a UKF, an iterated EKF, or a Gauss-Hermite filter. Write
`SqrtKalmanFilter` in place of `KalmanFilter` and it runs in square-root form. Give the noise a
second, wide component and the update is probabilistic data association. The state, the model,
and the measurement don't change.

## How it fits together

Four layers, each a directory of headers, each built on the ones before it.

- **Spaces** say what a state is made of. A state is a `Composite` of named blocks, and each
  block is a vector (`Linear<3>`), an angle (`RadiansPi<>`), a `UnitQuaternion`, a repeated or
  resizable block (a growing map), or a discrete mode. A space knows how to add a difference to
  a value and take the difference of two values, so a residual across an angle wraps, a
  quaternion's covariance is three by three, and a mean of headings is taken in the tangent
  space. Code addresses blocks by name (`marginal<Position>()`), never by index.
- **Distributions** put probability on a space: the Gaussian in four forms, mixtures, particle
  sets, sparse grids, categoricals, hybrids of a discrete mode and a continuous state, and
  Student-t, Gamma, uniform, and products of them.
- **Models** say how a state moves and is observed: process models composed from blocks
  (`ConstantVelocity`, `OrnsteinUhlenbeck`, your own), measurement models that arrive with their
  measurement and choose their own method (`methods::linearized()`, `methods::unscented()`, the
  sigma-point sets, `methods::iterated()`) or, for a linear measurement, give its matrix, and the
  simulation and consistency checks.
- **Filters** are the recursions over all of that, each a state plus `predict()` and `update()`.

## Why EmrysFilter

**One design, every representation.** A filter is a state and a recursion; the state is any
distribution over any space. So the estimators are combinations, not special cases: a mixture
of IMM filters, a particle filter whose particles each carry a Gaussian, a grid of Kalman
filters, a hidden Markov filter beside them. Marginals, conditionals, upkeep, smoothing,
and printing are written once and work on all of them. Manifold states are built in: the
boxplus and boxminus of a space, tangent-space covariances and means, and the covariance reset
are the library's, and angles and SO(3), as the unit quaternion, ship. SE(3) and the extended
pose group come next as spaces, and the invariant EKF is then a space, not a filter.

**Efficient and embedded-friendly.** Nothing sized by a bounded space allocates: not a predict,
an update, a transform, a marginal, or a draw, and the allocation tests build with
`EIGEN_RUNTIME_NO_MALLOC` to prove it for every filter's steps. A type with a run-time number of
elements takes a standard allocator, so a `std::pmr` arena holds a particle filter, and scratch
is owned and reused. The scalar is a template parameter: a `float` filter computes in `float`
throughout, and autodiff and multiprecision scalars run the same code. The library throws
nothing, reports failure through `std::expected`, and builds with `-fno-exceptions -fno-rtti`
and for Cortex-M4F and M7. The `KalmanFilter` costs 1.00 to 1.34 instructions per step relative
to a hand-written Eigen EKF on the same problem, after both pass the same agreement and
consistency tests.

**Parallel by policy.** Every loop over a run-time count (particles, mixture components, grid
cells) takes an `ExecutionPolicy` as its first argument, `execution::sequential` by default. Pass a
standard policy such as `std::execution::par`, or your own type with a `forEachIndex(count, f)`
(a thread pool handle, say, with its own allocator for the scratch), and the same step runs on
threads. Under `execution::sequential` it runs the code it always ran, bitwise, and every overload is
tested against it. The per-element bodies touch nothing
shared, which is the shape the planned GPU backend needs.

**Researched, documented, tested.** Every algorithm cites its source (eighty bibliography
entries) with a sentence on what the code takes and where it departs. Every entity is
documented, internals included, and the reference build fails on one warning. The architecture
is more than ninety named decisions with their reasons, reached through area-by-area design
reviews whose examples quote their compiled output. The tests, each saying what it is
evidence for, run under GCC, Clang, MSVC, and the sanitizers, and a coverage job checks that they
instantiate every public function and execute all but a handful of the headers' regions. Where no
published method fills a need, the library records the problem in
[docs/research.md](docs/research.md) instead of improvising.

**Agent-friendly.** A coding agent in a fresh clone should build a working estimator while
reading little, compiling few times, and checking the result numerically. So: examples of about
a hundred lines, built in CI, to copy; a short usage page with a "which filter?" table;
`simulate()` and the NIS and NEES monitors, which say pass or fail without a plot; one naming
convention without exceptions; [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md), which route an
agent to the right page; and a skill in [skills/](skills/) that an agent working in your own
project installs. Still to come, and listed with the rest of what is not built on the usage
page: the model contracts as concepts, so that a wrong model fails at the filter's entry with a
plain message, and a compact API index.

Also: every update returns its innovation, log-likelihood, and NIS, and the posterior Cramér-Rao
bound says how well any filter could have done. What intervenes in an update is a policy passed
to it (`reductions::gate()`, `select()`, `collapse()`); upkeep (`upkeep::resample(rng)`, `roughen(rng)`,
`pruneBelow()`, ...) runs when the caller says, on a trigger such as `upkeep::whenEss()`: applied to
the filter's state, or held by the filter and run by `maintain()`. Every policy is
made by a factory in the namespace of its kind. No mixins, no virtual dispatch, no hidden state
between steps. The library keeps Eigen's conventions and offers what Eigen lacks upstream,
and it is meant to sit beside a graduate course in state estimation.

## Which filter?

| Your problem | Start with | Then read |
|---|---|---|
| Linear or mildly nonlinear, one mode | `KalmanFilter` with `methods::linearized()`, `unscented()`, `iterated()`, or a sigma-point set | [docs/choosing.md](docs/choosing.md) |
| Ill-conditioned covariances, `float`, long runs | `SqrtKalmanFilter` | [docs/choosing.md](docs/choosing.md), "Can the arithmetic be trusted?" |
| Many sensors, fusion, uninformative priors | `InfoFilter`, `SqrtInfoFilter` | the same |
| Outliers, clutter, ambiguous measurements | mixture measurement noise with `reductions::collapse()`, `select()`, or `gate()`; `GaussianMixtureFilter` to keep the components | [docs/choosing.md](docs/choosing.md), "How is the noise distributed?"; [examples/gaussian_mixture.cpp](examples/gaussian_mixture.cpp) |
| Several motion regimes | `ImmFilter`, `AmmFilter`, `SwitchingModel`, `MarkovChain` | [imm-filter.md](docs/guide/imm-filter.md) |
| Multimodal or strongly nonlinear | `ParticleFilter` with `upkeep::resample(rng)` and `roughen(rng)` | [particle-filter.md](docs/guide/particle-filter.md) |
| Part of the state is conditionally linear | `RbParticleFilter` | [rao-blackwellized-filters.md](docs/guide/rao-blackwellized-filters.md) |
| A low-dimensional nonlinear part worth a grid | `RbPointMassFilter`, `PointMassFilter` | rao-blackwellized-filters.md |
| A discrete state, counts, or arrivals | `HiddenMarkovFilter` with `EmissionTable`, `PoissonEmission`, `ContinuousTimeMarkovChain` | [hidden-markov-filter.md](docs/guide/hidden-markov-filter.md) |
| Estimates after the fact | the histories in `Smoothers/`: fixed-interval and fixed-lag, IMM, mixture, particle, and `mostProbableSequence()` | [smoothing.md](docs/guide/smoothing.md) |
| An angle, a rotation, or your own manifold | `Periodic`, `UnitQuaternion`, or a type that satisfies the `Space` concept | the Spaces module page (`Spaces/Space.h`) |
| How good could any filter be? | `CramerRao.h` | the Models module page (`Models.h`) |

## Install

A C++23 compiler and Eigen 5.0. CI tests GCC 14, Clang 21, Apple Clang 21, and the MSVC of Visual
Studio 2026; GCC 13 builds the examples but is not tested. `std::expected` sets the floor, and a
compiler without it, Clang 18 on libstdc++ among them, stops at the first `std::expected` in
`Core/Weights.h` rather than with a message naming the floor. Eigen 5 is required, not the 3.4
that Linux distributions still ship; if CMake finds no Eigen 5, the build fetches it.

```cmake
include(FetchContent)
FetchContent_Declare(EmrysFilter
  GIT_REPOSITORY https://github.com/emrys-math/emrys-filter.git
  GIT_TAG        dev)  # a release tag, once there is one
FetchContent_MakeAvailable(EmrysFilter)
target_link_libraries(my_estimator PRIVATE EmrysFilter::EmrysFilter)
```

An installed copy is found with `find_package(EmrysFilter 0.1 CONFIG)`. Include
`<EmrysFilter/EmrysFilter.h>`, a layer's umbrella (`<EmrysFilter/Filters.h>`), or one header.
Including everything costs about three seconds and 300 MB of compiler memory, and a program the
size of the examples compiles in 5 to 25 seconds in under a gigabyte, so one file builds without a
CMake project: `clang++ -std=c++23 -O2 -I <checkout>/include -isystem <eigen> estimator.cpp`.
Build and test the library itself with `cmake --workflow --preset clang` (or `gcc`, or `msvc`
from an x64 developer prompt); a test translation unit takes one to four gigabytes, so the
presets limit how many compile at once. [CONTRIBUTING.md](CONTRIBUTING.md) says how a change
is checked.

## Header map

The headers live in `include/EmrysFilter/` in six layers. A header includes only its own layer
and the ones before it, and CI checks that.

| Layer | Holds | Headers |
|---|---|---|
| `Core/` | What everything else builds on | `Version.h`, `Macros.h` (assertions, no exceptions), `Math.h`, `Allocation.h`, `Execution.h` (policies), `Forwarding.h`, `Time.h`, `Stamp.h`, `Print.h`, `NumericError.h` (why a step fails), `Resampling.h`, `Weights.h`, `ChiSquared.h`, `Triangularization.h`, `Multiply.h`, `Solve.h` |
| `Spaces/` | What a state is made of | `Space.h` (the concept, `plus`, `minus`), `Coordinates.h` (`Linear`, `Periodic`), `Composite.h` (`Composite`, `Repeated`), `TagPath.h`, `Subspace.h`, `Indexing.h`, `Discrete.h`, `Reset.h`, `UnitQuaternion.h` |
| `Distributions/` | Probability over a space | `Params.h` (parameters and densities), `Component.h`, `Gaussian.h` and its forms (`SqrtGaussian.h`, `InfoGaussian.h`, `SqrtInfoGaussian.h`, `GaussianForm.h`), `Roots.h` (covariance roots and repairs), `Mixture.h`, `UniformMixture.h`, `SampleSet.h`, `ParticleSet.h`, `KernelDensity.h`, `Hybrid.h`, `Categorical.h`, `ContinuousTimeMarkovChain.h`, `Conditionals.h`, `GridGaussian.h`, `ParticleGaussian.h`, `StudentT.h`, `Gamma.h`, `Uniform.h`, `Product.h`, `Moments.h`, `Statistics.h`, `Sampling.h`, `Fitting.h`, `Divergence.h`, `Upkeep.h` |
| `Models/` | How states move and are observed | `Policies.h` (what a model declares), `ProcessModel.h`, `MeasurementModel.h`, `Placement.h` (placing and composing models), `ResolvedReads.h`, `Measurement.h` (the update), `MeasurementUpdate.h`, `MeasurementReads.h`, `IteratedUpdate.h`, `Likelihood.h`, `Transform.h`, `Methods.h` (the methods and sigma-point sets), `Batched.h` (functions that take every point at once), `TransformArgs.h`, `Linearization.h`, `Condition.h` (the Kalman update), `SwitchingModel.h`, `Emission.h`, `Simulation.h`, `Consistency.h`, `CramerRao.h`, `Reduction.h` (policies), `FormSteps.h` |
| `Filters/` | The recursions | `Recursion.h` (the contract), `FilterUpkeep.h` (the upkeep a filter holds), `KalmanFilter.h`, `SqrtKalmanFilter.h`, `InfoFilter.h`, `SqrtInfoFilter.h`, `MixtureFilter.h`, `ImmFilter.h`, `AmmFilter.h`, `ParticleFilter.h`, `RbUpdate.h`, `RbParticleFilter.h`, `RbPointMassFilter.h`, `HiddenMarkovFilter.h` |
| `Smoothers/` | Estimates of past states from later data | `Lag.h`, `GaussianHistory.h` (with `smoothStep()`), `HybridGaussianHistory.h`, `GaussianMixtureHistory.h`, `ParticleHistory.h`, `CategoricalHistory.h` |

## Documentation

- **Using it:** [docs/usage.md](docs/usage.md), the examples in [examples/](examples/),
  [docs/choosing.md](docs/choosing.md) for the choice of filter, and
  [the reference](https://emrys-math.github.io/emrys-filter/), the public API of every header.
  The `docs` target builds it, or with the internals for the maintainer.
- **Understanding it:** [docs/guide/](docs/guide/), the five design reviews still open, in reading order from
  [imm-filter.md](docs/guide/imm-filter.md) to
  [smoothing.md](docs/guide/smoothing.md), each
  showing real use with its printed results.
- **Why it is this way:** [docs/design/decisions.md](docs/design/decisions.md), the decisions;
  [docs/design/roadmap.md](docs/design/roadmap.md), the ordered steps and their status;
  [parallelism.md](docs/design/parallelism.md) and [benchmarks.md](docs/design/benchmarks.md),
  the plans behind the execution policies, the GPU backend, and the benchmarks.
- **What it cites:** [docs/reference/bibliography.md](docs/reference/bibliography.md).
- **What nobody has published yet:** [docs/research.md](docs/research.md), open problems with
  their hooks and starting points.

## Here, coming, and elsewhere

EmrysFilter does sequential inference on one state over time, continuous, discrete, or hybrid.

**Coming to EmrysFilter** ([roadmap](docs/design/roadmap.md)): a GPU backend for the particle,
grid, and Rao-Blackwellized filters, with counter-based random numbers and batched functions on
the way to it; the ensemble Kalman filter; fusion rules, out-of-sequence updates, and the
consensus step for distributed estimation; SE(3) and the extended pose group as spaces beside
SO(3), with the invariant EKF as their example; hooks for splitting a mixture by nonlinearity; and
Student-t robust updates.

**Planned as separate Emrys projects** that build on this one: a **tracker** (data association
and track management: JPDA, MHT, multi-target), **navigation** (inertial, GNSS, and geodetic
models and frames as a catalog), **analysis** (log readers, noise identification, Monte Carlo
runners, plots, and bindings), and a **controller**, an **optimizer**, a **graph library** for
factor graphs and Bayes nets, and a **sampler** for MCMC. Single-target association stays here,
since clutter is mixture measurement noise.

**Out of scope:** I/O, serialization, plotting, and configuration files (the analysis project's);
Python and ROS bindings (built on the analysis project's catalog of instantiations); batch
optimization and SLAM back-ends (the graph library's; use GTSAM today); model predictive control
(needs a solver the header-only libraries don't take); and inference runtimes for learned
models, which are supported as models through `withJacobian()` and `Unscented` but not linked.

## How it compares

| | EmrysFilter | mherb/kalman | GTSAM | manif | FilterPy |
|---|---|---|---|---|---|
| Language | C++23, header-only | C++11, header-only | C++, library | C++11, header-only | Python |
| Problem | sequential estimation, any representation | EKF, UKF, square-root UKF | factor-graph smoothing and mapping | Lie groups and their Jacobians | teaching and prototyping filters |
| States | vectors, angles, quaternions, discrete modes, composites, growing maps | vectors | Lie groups and vectors in a graph | Lie groups (not a filter) | vectors |
| Multimodal | mixtures, IMM, particles, grids, hybrids | no | no | no | particles, IMM |
| Allocation control, no exceptions, `float` | yes, tested | partly | no | yes | no |
| Every algorithm cited | yes | no | partly | yes | yes, as a book |

A manif group can be wrapped as a `Space` and estimated with any filter here. GTSAM is the tool
when the problem is a graph; this library is the tool when the problem is a recursion.

## Status

Pre-1.0. The algorithm set for 0.1 is frozen and the API is settling through the design
reviews; names may change before the first tagged release, after which `CHANGELOG.md` records
every change. CI runs GCC, Clang, MSVC, the sanitizers, clang-format, the header-layer check,
and the Doxygen build, dispatched by a maintainer for the changes that need them. The repository was eigen-filter and moved to
[emrys-math/emrys-filter](https://github.com/emrys-math/emrys-filter).

## Contributing and citing

The library implements algorithms from the public literature and contributes none of its own,
so a feature request names the paper it would implement; see
[CONTRIBUTING.md](CONTRIBUTING.md). A need that no published method fills goes on
[docs/research.md](docs/research.md) for a researcher to solve and publish, after which the
library implements and cites it. To cite the library, use `CITATION.cff`. To take the library
to another language, or to C++20 or C++17, start from [docs/porting.md](docs/porting.md).

MIT license. Copyright (c) 2026 Jay Moore.
