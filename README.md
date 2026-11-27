# EmrysFilter

[![CI](https://github.com/emrys-math/emrys-filter/actions/workflows/ci.yml/badge.svg?branch=dev&event=push)](https://github.com/emrys-math/emrys-filter/actions/workflows/ci.yml?query=branch%3Adev+event%3Apush)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24195424.svg)](https://doi.org/10.5281/zenodo.24195424)

**State estimation in C++23 on Eigen 5. One design from the Kalman filter to the
Rao-Blackwellized particle filter, on vectors, angles, quaternions, and discrete modes.
Header-only, allocation-free on bounded states, parallel by policy, and every algorithm cited
to the paper it implements.**

A Kalman filter that tracks a target in the plane from position fixes:

```cpp
#include <EmrysFilter/EmrysFilter.h>

#include <print>

using namespace Emrys;

// The state is named blocks: a position and a velocity in the plane.
struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
using State = Composite<Position, Velocity>;

int main()
{
    // The target moves at nearly constant velocity.
    auto const motion = ConstantVelocity<Position, Velocity>{0.1};

    // A GPS fix measures the position, with 1 m^2 of noise on each axis.
    auto const gps = measurementModel(
        affineIn(on<Position>(Eigen::Matrix2d::Identity())),
        additiveNoise(Eigen::Matrix2d::Identity()));

    // Before the first fix: near the origin, moving slowly.
    auto const prior = GaussianParams<State>{
        ValueVector<State>{0.0, 0.0, 1.0, 0.5},
        DifferenceVector<State>{10.0, 10.0, 1.0, 1.0}.asDiagonal()};
    auto filter = KalmanFilter{motion, prior};

    // A fix arrives every half second.
    Eigen::Vector2d const fixes[] = {{0.6, 0.2}, {1.1, 0.4}, {1.4, 0.8}};
    for (auto const & fix : fixes) {
        filter.predict(0.5);
        if (!filter.update(observe(gps, fix))) {
            return 1;  // the innovation covariance was not positive definite
        }
    }
    std::println("{:.3}", filter.state().marginal<Velocity>().copy());
}
```

```
Gaussian of dimension 2
  mean: 0.934 0.532
  covariance:
    0.715     0
        0 0.715
```

The fixes measure the position alone, and the filter estimates the velocity from them.

A model reads the blocks it names, wherever they sit in the state, so the same filter takes a
nonlinear measurement with nothing else changed. A radar at the origin measures a range and a
bearing, an angle that wraps at plus and minus pi:

```cpp
struct Range : Linear<1> {};
struct Bearing : RadiansPi<1> {};

Eigen::Matrix2d const radarNoise = Eigen::Vector2d{1.0, 1e-4}.asDiagonal();
auto const radar = measurementModel(
    []<typename T>(auto const & in, auto const & x) {
        auto const p = block<Position>(in, x);
        return Eigen::Matrix<T, 2, 1>{p.norm(), atan2(p(1), p(0))};
    },
    reads(path<Position>),
    writes(path<Range>, path<Bearing>),
    additiveNoise(radarNoise));

// z is the measured range and bearing.
auto const updated = filter.update(observe(methods::linearized(), radar, z));
```

`methods::linearized()` is the extended Kalman filter's update, and it differentiates the function
automatically. Write `methods::unscented()`, `methods::iterated(methods::linearized())`, or
`methods::gaussHermite<3>()` in its place and the update is that of a UKF, an iterated EKF, or a
Gauss-Hermite filter. Write `SqrtKalmanFilter` in place of `KalmanFilter` and the filter runs in
square-root form. The state, the models, and the measurements don't change. The programs in
[examples/](examples/) are complete estimators, from this filter to a Rao-Blackwellized particle
filter, and each simulates its problem and checks that its filter is consistent.

## How it fits together

Four layers, each built on the ones before it.

- **Spaces** say what a state is made of: a `Composite` of named blocks, each a vector, an
  angle, a unit quaternion, a repeated block such as a growing map, or a discrete mode.
- **Distributions** put probability on a space: the Gaussian in four forms, mixtures, particle
  sets, grids, categoricals, and hybrids of a discrete mode and a continuous state.
- **Models** say how a state moves and is observed, and each measurement chooses its own method.
- **Filters** are the recursions over all of that, each a state with `predict()` and `update()`,
  and the smoothers estimate past states from later data.

## Why EmrysFilter

- **One design, every representation.** A filter's state is any distribution over any space, so
  the estimators are combinations, not special cases: a mixture of IMM filters, a particle
  filter whose particles each carry a Gaussian, a grid of Kalman filters. Marginals,
  conditionals, smoothing, and printing are written once and work on all of them.
- **States on manifolds.** A space knows how to add a difference to a value and how to subtract
  two values, so a residual across an angle wraps, and a quaternion's covariance is three by
  three. Code names a block (`marginal<Position>()`) and never indexes it.
- **Efficient and embedded-friendly.** Nothing sized by a bounded space allocates, and the tests
  build with `EIGEN_RUNTIME_NO_MALLOC` to check it for every filter's steps. The library throws
  nothing, reports failure through `std::expected`, and builds with `-fno-exceptions -fno-rtti`
  and for Cortex-M. The scalar is a template parameter, so a `float` filter computes in `float`
  throughout. On the ranging benchmark, a `KalmanFilter` step takes 1.10 to 1.91 times the
  instructions of a hand-written Eigen EKF ([the benchmark results](docs/benchmarks.md)).
- **Parallel by policy.** Every loop over particles, mixture components, or grid cells takes an
  execution policy: `std::execution::par`, or a type of your own that wraps a thread pool.
- **Cited, documented, tested.** Every algorithm cites the paper it implements, and the library
  invents none. The tests run under GCC, Clang, MSVC, and the sanitizers. Every update returns
  its innovation and its NIS, so `simulate()` and the consistency monitors say whether an
  estimator works, without a plot.
- **Ready for a coding agent.** [AGENTS.md](AGENTS.md) routes an agent to the right page, the
  examples are programs to copy, and a skill in [skills/](skills/) takes an agent in your own
  project through building an estimator and checking it.

## Which filter?

| Your problem | Start with |
|---|---|
| Linear or mildly nonlinear, one mode | `KalmanFilter`, each model with `methods::linearized()` or `unscented()` |
| `float`, ill-conditioned covariances, long runs | `SqrtKalmanFilter` |
| Many sensors, or a prior with no information | `InfoFilter`, `SqrtInfoFilter` |
| Outliers, clutter, ambiguous measurements | mixture measurement noise, `GaussianMixtureFilter` |
| Motion that switches between regimes | `ImmFilter`, `AmmFilter` |
| Multimodal or strongly nonlinear | `ParticleFilter` |
| Part of the state is conditionally linear | `RbParticleFilter`, `RbPointMassFilter` |
| A discrete state, counts, or arrivals | `HiddenMarkovFilter` |
| Estimates of past states from later data | the smoothers in `Smoothers/` |

[docs/usage.md](docs/usage.md#which-filter) has the full table, with an example for each row,
and [docs/choosing.md](docs/choosing.md) gives the reasoning.

## Install

A C++23 compiler and Eigen 5.0. Eigen 5 is required, not the 3.4 that Linux distributions still
ship; if CMake finds no Eigen 5, the build fetches it. CI tests GCC 14, Clang 21, Apple Clang 21,
and the MSVC of Visual Studio 2026.

```cmake
include(FetchContent)
FetchContent_Declare(EmrysFilter
  GIT_REPOSITORY https://github.com/emrys-math/emrys-filter.git
  GIT_TAG        v0.1.0)
FetchContent_MakeAvailable(EmrysFilter)
target_link_libraries(my_estimator PRIVATE EmrysFilter::EmrysFilter)
```

One file also builds without a CMake project:
`clang++ -std=c++23 -O2 -I <checkout>/include -isystem <eigen> estimator.cpp`. A program the size
of the examples compiles in 5 to 25 seconds. To start a project, copy
[examples/consumer/](examples/consumer/). [docs/usage.md](docs/usage.md#getting-it) covers an
installed copy, `find_package`, older compilers, and microcontrollers.

## Documentation

- **Using it:** [docs/usage.md](docs/usage.md), the programs in [examples/](examples/),
  [docs/choosing.md](docs/choosing.md) for the choice of filter,
  [docs/headers.md](docs/headers.md) for what each header holds, and
  [the reference](https://emrys-math.github.io/emrys-filter/), the public API of every header.
- **How fast it is:** [docs/benchmarks.md](docs/benchmarks.md), each benchmark against the same
  filter written by hand in Eigen, and how to run them on your own machine.
- **Why it is this way:** [docs/design/decisions.md](docs/design/decisions.md), the decisions and
  their reasons; [docs/guide/](docs/guide/), the design reviews still open; and
  [docs/design/roadmap.md](docs/design/roadmap.md), the ordered steps and their status.
- **What it cites:** [docs/reference/bibliography.md](docs/reference/bibliography.md), and
  [docs/research.md](docs/research.md) for the problems that no published method solves yet.

## Scope

EmrysFilter does sequential inference on one state over time, continuous, discrete, or hybrid.
When the problem is a graph, as in batch optimization and SLAM back ends, GTSAM is the tool
today. Data association and track management for many targets are planned as a separate
project that builds on this one, as are navigation models and analysis tools. I/O, plotting, and
Python and ROS bindings are out of scope here. The [roadmap](docs/design/roadmap.md) lists what
is coming, including SE(3) and the other pose groups as spaces, noise covariances adapted as
the filter runs, particle proposals that use the measurement, and the ensemble Kalman filter.

| | EmrysFilter | mherb/kalman | GTSAM | manif | FilterPy |
|---|---|---|---|---|---|
| Language | C++23, header-only | C++11, header-only | C++, library | C++11, header-only | Python |
| Problem | sequential estimation, any representation | EKF, UKF, square-root UKF | factor-graph smoothing and mapping | Lie groups and their Jacobians | teaching and prototyping filters |
| States | vectors, angles, quaternions, discrete modes, composites, growing maps | vectors | Lie groups and vectors in a graph | Lie groups (not a filter) | vectors |
| Multimodal | mixtures, IMM, particles, grids, hybrids | no | no | no | particles, IMM |
| Allocation control, no exceptions, `float` | yes, tested | partly | no | yes | no |
| Every algorithm cited | yes | no | partly | yes | yes, as a book |

A manif group can be wrapped as a `Space` and estimated with any filter here.

## Status, contributing, and citing

The library is at 0.1.0. Before 1.0, a minor release may change the API, and
[CHANGELOG.md](CHANGELOG.md) records every change a user can see, with what to write instead
where one breaks code.

The library implements algorithms from the public literature, so a feature request names the
paper it would implement; see [CONTRIBUTING.md](CONTRIBUTING.md). To cite the library, use
`CITATION.cff` or its DOI, [10.5281/zenodo.24195424](https://doi.org/10.5281/zenodo.24195424),
which resolves to the latest release. To take the library to another language, or to C++20 or
C++17, start from [docs/porting.md](docs/porting.md).

MIT license. Copyright (c) 2026 Jay Moore.
