# Using EmrysFilter

EmrysFilter is a header-only C++23 library of state estimators on Eigen 5: Kalman filters in
their extended, unscented, iterated, square-root, and information forms, Gaussian mixture and
interacting multiple model filters, particle and Rao-Blackwellized filters, hidden Markov model
filters, and smoothers for each. This page is for someone, or someone's coding agent, building an
estimator with it. It names the pieces, the conventions they share, and how to check that an
estimator works. The programs in [`examples/`](../examples/) are the worked cases, and each is
built and run with the tests, so their code compiles against this version of the library.

## Getting it

The library needs a C++23 compiler and Eigen 5.0, which its CMake project fetches when no
installed Eigen 5.0 is found. CI tests GCC 14, Clang 21, Apple Clang 21, and the MSVC of Visual
Studio 2026; GCC 13 builds the examples but is not tested. `std::expected` sets the floor, and a
compiler without it, Clang 18 on libstdc++ among them, stops at the first `std::expected` in
`Core/Weights.h` rather than with a message naming the floor. Install it from a checkout with

```sh
cmake -S <checkout> -B <build directory> -DEMRYS_FILTER_BUILD_TESTS=OFF
cmake --install <build directory> --prefix <prefix>
```

which installs the headers and the CMake package, and, when it fetched Eigen, Eigen's headers
and package beside them. Then find it, and link the target `EmrysFilter::EmrysFilter`:

```cmake
find_package(EmrysFilter 0.1 CONFIG REQUIRED)
target_link_libraries(my_estimator PRIVATE EmrysFilter::EmrysFilter)
```

configuring with `-DCMAKE_PREFIX_PATH=<prefix>` unless the prefix is one CMake searches. Or skip
the install and fetch it, which finds an installed one first:

```cmake
include(FetchContent)
FetchContent_Declare(EmrysFilter
  GIT_REPOSITORY <repository URL>
  GIT_TAG <tag or commit>
  FIND_PACKAGE_ARGS 0.1 CONFIG)
FetchContent_MakeAvailable(EmrysFilter)
```

Include `<EmrysFilter/EmrysFilter.h>` for everything, or one layer's header, such as
`<EmrysFilter/Filters.h>`. Everything is in the namespace `Emrys`. Without a CMake project, one
file compiles against the headers alone, as the examples do:

```sh
clang++ -std=c++23 -O2 -I <checkout>/include -isystem <eigen> \
  -o estimator estimator.cpp
```

Including everything costs about three seconds and 300 MB of compiler memory, and a program the
size of the examples compiles in 5 to 25 seconds in under a gigabyte. The library's own tests are
far heavier, since each instantiates many filters; a user's program is not.

To start a new project, copy the directory `examples/consumer/` from the repository. Its
`CMakeLists.txt` finds or fetches the library, builds `main.cpp` in Debug with GCC's and Clang's
template errors cut to their ends, and adds the target that installs the skills below.

On a microcontroller, the library builds with `-fno-exceptions -fno-rtti`, and a filter over
fixed-size spaces never touches the heap. One setting is Eigen's: Eigen puts its small scratch
buffers on the stack with `alloca` only on platforms it recognizes, and on the heap elsewhere.
newlib, the C library of the Arm GNU toolchain, declares `alloca` only outside strict ISO mode,
so with `-std=c++23` define `EIGEN_ALLOCA=__builtin_alloca`. Otherwise a solve over a block whose
size is known only at run time allocates. The repository's `benchmarks/cortex_m/` builds the
ranging filter this way for the Cortex-M4F and M7. Its startup code fails any program that
allocates, and it runs the filter under QEMU.

## Skills for a coding agent

The directory `skills/` holds skills for a coding agent, in the open
[Agent Skills](https://agentskills.io) format. `emrys-filter-build-estimator` takes an agent
through building an estimator and checking it in simulation. Install them from your
`CMakeLists.txt`, so that they match the version you build against:

```cmake
emrys_filter_add_skills_target(DESTINATION .claude/skills)
```

and run `cmake --build <build directory> --target emrys_filter_skills`, which no other build
runs. The destination is your agent's skills directory, relative to the calling directory: Claude
Code reads `.claude/skills/`, Codex and Gemini CLI read `.agents/skills/`, and Copilot and Cursor
read both, so install into one of them. The function comes with the library, whether found or
fetched. Without a CMake project, run
`cmake -DDESTINATION=<directory> -P <EmrysFilter>/tools/install_skills.cmake`, where
`<EmrysFilter>` is a checkout or an install's `share/EmrysFilter`. Each skill is a
directory named `emrys-filter-<name>`, with its own copies of this page, choosing.md, and the
examples, and the install replaces those directories each time and leaves the rest alone.

## Which filter?

One fact drives the choice. A Gaussian through an affine function is exactly a Gaussian, so the
posterior of a linear model with Gaussian noise is one Gaussian forever, and the Kalman filter is
exact. Through a nonlinear function it is not: a Gaussian position through a range becomes a ring,
through a time difference of arrival a curved strip, and through a measurement that two states
explain equally a density with weight in two places. A filter of the Kalman family replaces that
by a Gaussian with approximate moments, which describes it only while the function is nearly
affine across the distribution's spread.

Answer each question below on its own. The first answer is the default; where another applies,
[choosing.md](choosing.md) has the reasoning and the choice.

- **The process model.** The built-in models are affine, so this concerns a model you wrote:
  `methods::linearized()` if it is nearly affine across the state's spread, `methods::unscented()`
  otherwise. The method is the filter's third argument.
- **Each measurement, judged on its own.** A `measurementModel()` given its matrix, if it is
  linear in the state, whose update is exact by any method, so `observe(model, z)` names none.
  Otherwise `methods::linearized()` if
  nearly affine across the spread; `methods::unscented()` if it bends across it or isn't
  differentiable; `methods::iterated(methods::linearized())` or
  `methods::iterated(methods::unscented())` for a precise measurement of a function that bends. The
  method is the first argument of `observe()`.
- **The posterior.** One Gaussian, unless the prior is wide against a precise nonlinear
  measurement, or several states explain the measurements equally. Then particles if the
  measurements are coarse, Gaussian components or a grid if they are precise, with the linear
  part taken out either way; choosing.md says how.
- **The noise.** Gaussian. A Gaussian mixture for outliers, in every filter. Any other measurement
  error distribution as a measurement model known only by its likelihood,
  `measurementModel(reads(...), logLikelihood(g))`, in `ParticleFilter`, and in `PointMassFilter`, `RbPointMassFilter`, and
  `RbParticleFilter` when it reads only the gridded or sampled block. Process noise is Gaussian in
  every filter.
- **The regime.** One process model. `ImmFilter` if the motion, or a bias's or parameter's drift,
  switches during the run;
  `AmmFilter` with `upkeep::floorModes()` if one of several models holds throughout;
  `HiddenMarkovFilter` for a discrete state alone.
- **The arithmetic.** `KalmanFilter`. `SqrtKalmanFilter` for `float`, ill-conditioning, or long
  runs; `InfoFilter` for an uninformative prior or many sensors; `SqrtInfoFilter` for both.
- **After the run.** The smoothers in `Smoothers/` for estimates of past states from
  later data; `PosteriorCramerRaoBound` for the best accuracy any filter could reach.

| The problem | The filter | Example |
|---|---|---|
| Linear dynamics and measurements, Gaussian noise | `KalmanFilter`, with each measurement a `measurementModel()` given its matrix; on a plain vector, the process model too | [constant_velocity.cpp](../examples/constant_velocity.cpp), [linear_kalman.cpp](../examples/linear_kalman.cpp) |
| Mildly nonlinear models, a posterior near one Gaussian | `KalmanFilter`: `methods::linearized()` is the EKF, `methods::unscented()` the unscented filter, chosen per model | [unicycle.cpp](../examples/unicycle.cpp) |
| A state on a manifold, such as an attitude | `KalmanFilter` over a `UnitQuaternion` block | [attitude.cpp](../examples/attitude.cpp) |
| A sensor's bias or a parameter known only from its calibration, to account for without estimating it | `KalmanFilter`, with `Emrys::consider(Emrys::path<Bias>)` in the measurement's `observe()` | [consider_bias.cpp](../examples/consider_bias.cpp) |
| A precise, strongly nonlinear measurement | `methods::iterated(methods::linearized())` (the iterated EKF) or `methods::iterated(methods::unscented())` (the iterated posterior linearization filter) as the measurement's method | — |
| `float`, ill-conditioning, or an uninformative prior | `SqrtKalmanFilter`, `InfoFilter`, or `SqrtInfoFilter`: the same filter over another form of the Gaussian | — |
| Outliers in the measurements | mixture noise on the measurement model, with `reductions::collapse()` or `select()`, or a `reductions::gate()` | [gaussian_mixture.cpp](../examples/gaussian_mixture.cpp) |
| A wide prior with precise nonlinear measurements, such as a position located by ranges or bearings alone, or a posterior that is near-Gaussian piece by piece | `GaussianMixtureFilter`, with your own merge to bound its components; a `MixtureFilter` over `HybridGaussian` components, one IMM per component, if a regime switches as well | — |
| Motion that switches between models | `ImmFilter` over a `SwitchingModel` | [imm.cpp](../examples/imm.cpp) |
| A fixed set of candidate models, one of which is true | `AmmFilter` with `upkeep::floorModes()` | — |
| Any nonlinearity, a posterior of any shape, or non-Gaussian measurement noise, in a few dimensions | `ParticleFilter` | [particle_filter.cpp](../examples/particle_filter.cpp); a measurement that takes every particle at once, [terrain_lookup.cpp](../examples/terrain_lookup.cpp) |
| Part of the state nonlinear, the rest linear-Gaussian given it | `RbParticleFilter` (sampled block that moves), `RbPointMassFilter` (grid over a block that doesn't), each an IMM per particle or cell under a `SwitchingModel`; over Gaussian components, the models' `affineIn()`, which every sigma-point method factors out | [rao_blackwellized.cpp](../examples/rao_blackwellized.cpp) |
| A static block, nonlinear in the measurements, and nothing else | `PointMassFilter` | [point_mass_filter.cpp](../examples/point_mass_filter.cpp) |
| A discrete state only | `HiddenMarkovFilter` | — |
| Estimates of past states from later data | the histories in `Smoothers/`: Rauch-Tung-Striebel, IMM, mixture, particle, and Viterbi | — |
| The best accuracy any filter could reach on a problem | `PosteriorCramerRaoBound` | — |

Rows without an example have tests in `tests/Filters/` and `tests/Models/` that show each in use.
Where more than one row fits, read "Choosing a representation" in [choosing.md](choosing.md).

## The pieces of an estimator

**The simplest filter.** A Kalman filter on a plain vector needs no names: a model given without
`reads()` and `writes()` reads and writes the whole state, and a linear one is given by its matrix.
The textbook filter of a position and velocity in the plane is

```cpp
using State = Emrys::Linear<4>;  // x, y, and their rates
auto const motion =
    Emrys::fixedStep(Emrys::processModel(F, Emrys::additiveNoise(Q)), dt);
auto const fix = Emrys::measurementModel(H, Emrys::additiveNoise(R));
auto const prior = Emrys::GaussianParams<State>{mean, cov};
auto filter = Emrys::KalmanFilter{motion, prior};
filter.predict();
auto const updated = filter.update(Emrys::observe(fix, z));
```

with `F` 4 by 4 and `H` 2 by 4, as [linear_kalman.cpp](../examples/linear_kalman.cpp) runs it.
That program is [constant_velocity.cpp](../examples/constant_velocity.cpp) with its matrices
written out: the same filter, which prints the same results, and only the syntax differs. The
space is named once, by the prior, and the filter binds the step to it. A model given its matrix is
updated exactly by every method, so its measurement needs none. A
nonlinear whole-state model is a function of the whole state, `f(x, dt)` for a process model and
`h(x)` for a measurement model, with its noise over the whole state:
`processModel(f, continuousNoise(Qc))` or `additiveNoise(Q)`, and
`measurementModel(h, additiveNoise(R))`. Such a model takes
no inputs of the call and is neither placed nor composed. Once a state has parts that models read
apart, name them, as the rest of this page does.

**A state space.** A state is a `Composite` of blocks, each a type derived from a coordinate type
so that models can name it:

```cpp
struct Position : Emrys::Linear<2> {};
struct Heading : Emrys::RadiansPi<1> {};   // periodic, in [-pi, pi)
struct Attitude : Emrys::UnitQuaternion {};
using State = Emrys::Composite<Position, Heading>;
```

The coordinate types are `Linear<N>`, the periodic `RadiansPi`, `Radians2Pi`, `Degrees180`, and
`Degrees360`, and `UnitQuaternion`. A fixed-size space is an empty type, so it is named, not
passed: `GaussianParams<State>{mean, cov}`, `Emrys::minus<State>(a, b)`, and
`Emrys::simulate<State>(...)`. A space that holds a count, one with a `Repeated` block, is passed
as an instance, `Emrys::minus(space, a, b)`, wherever its size can't be read from the vectors.

**A process model.** A process model writes some blocks of the state from the blocks it reads, and
`Emrys::compose(models...)` puts models side by side: `ConstantVelocity<Position, Velocity>{q}`,
`ConstantAcceleration`, `RandomWalk<B>{q}`, `OrnsteinUhlenbeck<B>{TimeConstant{tau},
StationaryVariance{s}}`, and `Constant<B>{}` are built in. Their parameters are continuous-time
spectral densities, discretized exactly over each step. Every coordinate of the state must be
written by exactly one model, so a block that doesn't move still needs `Constant` or `RandomWalk`.
Motion the library doesn't provide, such as a unicycle or a gyroscope turning an attitude, is a
model you write with `Emrys::processModel()` (below). Wrap the model in
`Emrys::fixedStep(model, dt)` when the step never changes: the filter it is given computes the
process noise over its state's space once, and the transition where the model is affine, and its
`predict()` then takes no step.

**Measurement models and measurements.** A measurement model predicts one kind of measurement
from what it reads, and `Emrys::observe(method, model, z, inputs...)` makes the measurement a filter
updates on. A linear measurement, $z = H x + v$ with $x$ the blocks read, is given by its matrix,
and its update is the Kalman filter's exact one under every method, so `observe(model, z)` names
none:

```cpp
auto const gps = Emrys::measurementModel(
    Eigen::Matrix2d::Identity(),
    Emrys::reads(Emrys::path<Position>),
    Emrys::additiveNoise(noise));
auto const updated = filter.update(Emrys::observe(gps, z));
```

`reads()` stacks the blocks it names, in that order, and $H$ has a column for each of their
coordinates. The blocks are vectors or angles; a quaternion has no linear measurement. $H$ may
instead be given by its columns, a matrix per block placed with `on<Block>()`, or with
`onEach<Block>()` on a repeated block, which reads each repetition into a repetition of the
measurement, so the measurement grows and shrinks with the block:

```cpp
// each landmark's position relative to the vehicle's, z_i = l_i - p
auto const relative = Emrys::measurementModel(
    Emrys::Columns{Emrys::onEach<Landmarks>(Eigen::Matrix2d::Identity()),
                   Emrys::on<Position>(-Eigen::Matrix2d::Identity())},
    Emrys::additiveNoise(Emrys::onEach<>(offsetNoise)));
```

Its noise is given for one repetition, `onEach<>()` placing it on every one. Any other measurement
is a function of the blocks it reads, as a process model is. It takes `h(in, out, x)`, `h(in, x)`, or
`h(x)`, whichever it needs, told apart by the count of its arguments, and a process model's function
the same with `dt` after:

```cpp
auto const range = Emrys::measurementModel(
    []<typename T>(auto const & in, auto const & x) {
        // a scalar, in the input's scalar
        return T{Emrys::block<Position>(in, x).norm()};
    },
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Range>),  // optional for a model used alone
    Emrys::additiveNoise(rangeNoise));
auto const updated =
    filter.update(Emrys::observe(Emrys::methods::unscented(), range, z));
```

The method, made by its factory in `Emrys::methods` (`linearized()`, `unscented()`, `cubature()`,
`sphericalSimplex()`, `gaussHermite()`, `iterated()`, and `spread()`, which gives a sigma-point
rule other axes), is chosen per measurement, so one filter can take a range by linearization and a
range-bearing measurement by sigma points. The prediction uses the filter's own method, the third
argument of `KalmanFilter{model, prior, method}`. A measurement in a periodic space, such as a
bearing, names that space with `writes()` or `outputSpace()`, and its residuals then wrap.
`Emrys::compose(range, bearing)` measures both side by side, the output `Composite<Range,
Bearing>`. Noise that passes through the function, such as a range's scale error, is an input the
model reads and the call gives, `observe(method, model, z, GaussianParams<ScaleNoise>{...})`; a
model that names it in `affineIn()` has it carried exactly by every sigma-point method.

A measurement model is a value that holds its function and its noise, so a measurement whose
geometry changes, such as one taken from a moving platform or between two moving receivers, is
made by a new model at each time, its function capturing that time's positions:

```cpp
auto tdoa(Eigen::Vector3d const & a1, Eigen::Vector3d const & a2) {
  auto const difference = [a1, a2](auto const & in, auto const & x) {
    /* ... */
  };
  return Emrys::measurementModel(difference,
                                 Emrys::reads(Emrys::path<Position>),
                                 Emrys::additiveNoise(noise));
}
auto const updated = filter.update(
    Emrys::observe(Emrys::methods::unscented(), tdoa(a1, a2), z));
```

A model built once covers a measurement that follows the state's counts:
`outputSpace(CountFrom<Landmarks, RangeBearings>{})` sizes it, a matrix's `Columns` repeat with a
block, and noise placed with `onEach<>()` is placed for each count. What it can't say, it says when
built per measurement, sized and with noise computed from that measurement's own data. A
round-trip range from a pivot with the time differences at the receivers that heard the dwell has
noise correlated through the pivot's timing error, with each error's variance set by its
signal-to-noise ratio, so its covariance is $D \Sigma D^T$ for the independent errors $\Sigma$ and
the differencing $D$, and its size is the number of receivers that heard it:

```cpp
auto const ranging = Emrys::measurementModel(
    dwellFunction(dwell),  // reads the receivers that heard it
    Emrys::reads(Emrys::path<Target>, Emrys::path<Receivers>),
    // Composite<RoundTrip, Tdoas>, sized for the dwell
    Emrys::outputSpace(Reading{RoundTrip{}, Tdoas{Tdoa{}, n}}),
    Emrys::additiveNoise(
        Eigen::MatrixXd{d * sigma.asDiagonal() * d.transpose()}));
filter.update(Emrys::observe(Emrys::methods::linearized(), ranging, dwell.z));
```

A linear measurement with a fixed and a repeated part whose noise is correlated is a matrix built
the same way, for the blocks it reads and that measurement's count. A measurement whose error is
not Gaussian is a model known by its likelihood, `measurementModel(reads(...), logLikelihood(g))`
with `g(in, x, z)` returning $\log p(z \mid x)$, observed without a method, `observe(model, z)`,
by the filters that weight points. Give it `sample(g)` too, with `g(in, x, rng)` returning a draw of
the observation at `x`, and `simulate()` draws its observations; without it `simulate()` refuses
the model.

**A filter.** A filter owns its process model and its state, and has two operations:
`predict(dt, inputs...)` and `update(measurement)`. A filter is named by the distribution of its
state (`KalmanFilter<Gaussian<State>, Model>`), and its constructor deduces it from a model and a
prior. `state()` returns the state's parameters, such as a `GaussianParams` with `mean()` and
`covariance()`, a mixture, a hybrid, or a particle set. A particle set's estimate is its
`moments()`, a `std::expected` that fails only on invalid weights, holding `mean()` and
`covarianceLower()`, of which only the lower triangle is read: take the whole matrix with
`covarianceLower().selfadjointView<Eigen::Lower>()`. A mixture has two estimates: the
`moments()` of its density, `state().toDensity()`, which can sit between its modes, and its most
probable component, `state()[state().mostProbable()].component`.

## Conventions

- **Values and differences.** A space's values and its differences can differ: a heading's
  differences wrap into [-pi, pi), and a unit quaternion's four coordinates have three-coordinate
  rotation vectors as differences. A covariance is over the differences, so an attitude-and-bias
  state has 7 values and a 6 by 6 covariance. Take an error with `Emrys::minus<State>(a, b)`,
  never `a - b`, and apply a correction with `Emrys::plus<State>(x, delta)`; a space with a count
  is passed instead, `Emrys::minus(space, a, b)`. The space sizes its Eigen types, so no
  dimension is written out: `Emrys::ValueVector<State>` is a value, such as a mean or a true
  state, `Emrys::DifferenceVector<State>` a difference, such as an error or a covariance's
  diagonal, and `Emrys::DifferenceMatrix<State>` a covariance; `DifferenceMatrix<Position, State>` is
  a matrix from the state's differences to a measurement's, such as `H`. A block names its own,
  `ValueVector<Position>`, and a resizable space's are dynamic with its bound. A covariance given
  by its variances is `variances.asDiagonal()`, as in
  `GaussianParams<State>{mean, Emrys::DifferenceVector<State>{4.0, 4.0, 1.0, 1.0}.asDiagonal()}`.
- **Differences in your models: recommended, not required.** The examples keep track of which
  vectors are values and which are differences. A block that is a difference of another quantity,
  such as a gyroscope's bias, is declared in that quantity's difference space,
  `struct Bias : Emrys::DifferenceSpace<Gyro> {};`, and the model applies differences with the
  space's operations: `Emrys::less<Gyro>(reading, bias)` for a reading less its bias,
  `Emrys::plus<Heading>(heading, turn)` for a step, and `Emrys::minus<Position>(a, b)` for the
  difference of two values. Doing so keeps a model right on every space and when a space changes,
  so we recommend it. You may leave it out. On a `Linear` space `plus()`, `less()`, and `minus()`
  are `+`, `-`, and `-`, and the filter normalizes the values a model returns, so bare arithmetic is
  also right on an angle that the model returns as a value: `block<Gyro>(in, x) - block<Bias>(in, x)`
  and `heading + dt * turnRate` both work. Bare arithmetic goes wrong on a space whose `plus()` is
  not addition, such as a unit quaternion, and on a difference of angles that the model uses itself,
  such as a steering command `gain * (goal - heading)`, which turns the long way round across the
  seam at pi. Where your spaces are linear, or you have checked those cases, plainer code can be
  the better trade.
- **Parameters and densities.** A distribution that can be invalid, such as a Gaussian whose
  covariance is not positive definite, has two types: its parameters (`GaussianParams`), which hold
  the values and change step by step, and its density (`Gaussian`), which `toDensity()` returns
  after factorizing once. Densities evaluate with `logDensity(x)` and `density(x)`, discrete ones
  with `probability()` and `logProbability()`. A density borrows from its parameters and must not
  outlive them or be read after they change. Hold one by value, `auto const d =
  g.toDensity().value();`: `auto const & d = ...` binds into the temporary `std::expected` and
  dangles at the semicolon. A density of a mixture's or hybrid's component is not to be read after
  the container grows, which may move the component.
- **Failure is a value.** Operations that can fail on valid input return `std::expected`: a
  Gaussian filter's `update()` fails when the innovation covariance is not positive definite, a
  filter's of weighted parts when its weights define no distribution, with a `NumericError` of
  which `GaussianError` and `WeightError` are two names, and `toDensity()` fails on invalid
  parameters. Check each before using its value. Every filter's `update()` returns
  a result with `accepted` and `logLikelihood()`. An update can be rejected, leaving the state as it
  was: by its policy (a `Gate`), with a finite log-likelihood, or because the model gives the
  measurement no density, as when no particle or value explains it, with a log-likelihood of
  $-\infty$. Check `accepted` as well. To score a run, sum `logLikelihood()` over every update,
  rejected ones included.
- **Generic measurement functions.** `Linearized` differentiates a function by automatic
  differentiation, so the function must be generic in its scalar type. Write it as a template
  over the block's type, compute in that type's scalar with `Eigen::numext::sin`, `atan2`, and
  `sqrt`, and return a concrete `Eigen::Matrix`, not an expression, such as
  `Emrys::ValueVector<Measurement, T>` for a named output space. Captured constants and known
  inputs may stay `double`: Eigen mixes them with the dual numbers, so they need no `.cast<T>()`,
  and a scalar such as `dt` needs no `T(dt)`. A `float` filter is the exception: a `double` vector
  does not mix with its `float` ones, so hold its constants in `float`. A function that knows its Jacobian returns it with `withJacobian(value, jacobian)`.
- **Inputs.** An input of a prediction is a block the model names in `reads()` that the state
  lacks, such as a gyroscope's reading `struct Gyro : Linear<3> {}`. The arguments to `predict()`
  after the step give each by its tag: a known control as `Emrys::sure<Gyro>(value)`, or an
  uncertain input, such as an inertial reading with its noise, as a Gaussian over its tag,
  `GaussianParams<Gyro>{...}`, whose uncertainty is carried through the dynamics. The function
  reads either with `block<Gyro>(in, x)`.
- **Upkeep is the caller's.** A particle filter never resamples by itself: after each
  step's updates, apply upkeep such as `Emrys::upkeep::whenEss(std::less{}, 0.5, Emrys::upkeep::resample(rng))` to
  `filter.state()`. Upkeep is called with the state alone; `resample(rng)` and `roughen(rng)` hold a
  reference to the generator, which must outlive them, and an execution policy, when wanted, goes to
  the factory first, as in `resample(std::execution::par, rng)`. A Gaussian mixture is pruned and merged the same way. A
  mixture filter started from a grid of components soon holds many whose weight the measurements
  have driven to nearly zero, and each one still costs an update. `Emrys::upkeep::pruneBelow(1e-6)`
  applied to `filter.state()` removes every component whose normalized weight is below 1e-6 and
  renormalizes the rest, always keeping the most probable; `upkeep::pruneTail(1e-3)` removes the
  lightest components while their total weight stays below 1e-3, and `upkeep::pruneToSize(32)` the
  lightest until 32 remain. Each returns whether it removed any, as a `std::expected` that fails as
  the mixture's weights do. The merge is yours to write: the library gives the merge of two components (`merged()`, `merge()`) and the divergences
  to cost it by, and [gaussian_mixture.cpp](../examples/gaussian_mixture.cpp) is one to start
  from. Resampling copies the particles that explain the measurements and drops the rest, and
  process noise then spreads the copies apart again. A state with little or no process noise, such as a stationary emitter or a
  calibration parameter under `Constant`, gets no such spread, and its cloud collapses onto a
  few points. Follow the resample with a `Roughen`, which moves each particle by a draw from a
  kernel fit to the cloud (the regularized particle filter): `Emrys::upkeep::whenEss(std::less{}, 0.5,
  Emrys::upkeep::resample(rng), Emrys::upkeep::roughen(rng))`. Each roughening widens the cloud a little, so its
  covariance runs slightly large. The trigger holds a reference to the generator, so a class that holds
  both its generator and the trigger must not be copied or moved with the default members: build the
  trigger where it is applied, as `benchmarks/particles/emrys_particles.h` does, since it costs nothing
  to make.
  A Kalman filter takes upkeep too: when rounding leaves its covariance not positive definite, as
  the default update can after a measurement far more precise than the prior,
  `Emrys::upkeep::repairCovariance()(filter.state())` adds a ridge to its diagonal, and returns
  whether it had to. Observing such a measurement with `Emrys::updates::joseph()` avoids it in the
  update, at more cost per update. A filter can also hold its upkeep, given last to its constructor,
  and run it when you call `maintain()`: `auto filter = Emrys::ParticleFilter{model, particles,
  Emrys::upkeep::whenEss(std::less{}, 0.5, Emrys::upkeep::resample(rng))};` and then
  `filter.maintain()` after each step's updates. `Emrys::upkeep::sequence(...)` makes several
  actions one value, run in order.
- **Time.** `predict()` takes a step in any time type: a scalar in the models' unit, usually
  seconds, or a `std::chrono::duration`.
- **Scalars.** Everything takes its scalar from its parameters, `double` by default. A `float`
  filter is spelled with `float` parameters throughout.

## Writing a process model

A process model is a function of what it reads, with policies that name it:

```cpp
auto const unicycle = Emrys::processModel(
    // in and out are the spaces that reads() and writes() define; x is a
    // vector in in, read by block; the result is a vector in out, built from
    // zeros<T>(out) and filled by block. T is the scalar of x, which the call
    // supplies, so that Linearized can differentiate the function.
    []<typename T>(auto const & in, auto const & out, auto const & x,
                   double dt) {
        using Emrys::block;
        T const heading = block<Heading>(in, x)(0);
        T const distance = dt * block<Speed>(in, x)(0);
        auto const displacement = Emrys::DifferenceVector<Position, T>{
            distance * Eigen::numext::cos(heading),
            distance * Eigen::numext::sin(heading)};
        Emrys::DifferenceVector<Heading, T> const turn =
            dt * block<TurnRate>(in, x);
        auto y = Emrys::zeros<T>(out);
        block<Position>(out, y) =
            Emrys::plus<Position>(block<Position>(in, x), displacement);
        block<Heading>(out, y) =
            Emrys::plus<Heading>(block<Heading>(in, x), turn);
        return y;
    },
    Emrys::reads(Emrys::path<Position>, Emrys::path<Heading>,
                 Emrys::path<Speed>, Emrys::path<TurnRate>),
    Emrys::writes(Emrys::path<Position>, Emrys::path<Heading>),
    // or additiveNoise(), or neither
    Emrys::continuousNoise(Emrys::on<Heading>(0.01)));
```

`reads()` names what the function reads, blocks of the state or inputs of the call; `writes()` the
blocks of the state its result fills, in order; `continuousNoise()` white-noise spectral densities
by block of the output, discretized over the step, or `additiveNoise()` a fixed $Q$. A function
that knows its Jacobian over its input's differences returns `withJacobian(y, J)` and is never
differentiated. `affineIn(path<...>)` names the inputs the function is affine in given the rest,
which every sigma-point method then carries exactly rather than spreading points over them.
[unicycle.cpp](../examples/unicycle.cpp) and [attitude.cpp](../examples/attitude.cpp) each write
one, and `processModel()` in `Models/ProcessModel.h` documents the policies.

A model reads its blocks by name wherever they sit in the state, so one model serves every place
that has them. `Emrys::compose(models...)` puts models side by side, each writing its own blocks:
`compose(unicycle, Emrys::RandomWalk<Speed>{0.1}, Emrys::RandomWalk<TurnRate>{0.01})`.
`Emrys::on<Ownship>(vehicle)` places a model on the block `Ownship`, prefixing every path into the
state it reads and writes, so two vehicles are `compose(on<Ownship>(vehicle), on<Target>(vehicle))`.
An input keeps its tag, shared by both, unless renamed: `on<Target>(vehicle,
on<TargetForce>(path<SpecificForce>))` reads the target's force from the argument tagged
`TargetForce`. `Emrys::onEach<Fleet>(vehicle)` places one copy on every repetition of a `Repeated`
block, each with its own noise; an input renamed to a `Repeated` tag has a value per repetition.

## Writing a recursion

A filter is a state and a recursion: the process model, the prediction method, and any scratch,
acting on a state passed in. `KalmanFilter` and `MixtureFilter` take theirs as their last template
argument, the `KalmanRecursion` or `ImmRecursion` their component names unless given one, and
deduce it from a recursion passed to the constructor. So a step the library doesn't ship is a type
of your own that meets `Emrys::Recursion` (`Filters/Recursion.h`), and it runs over one Gaussian or
over each component of a mixture with nothing in the library changed. The shortest one wraps a
library recursion and changes one step. This one inflates every predicted covariance by $\alpha^2$,
a fading-memory filter that forgets old measurements:

```cpp
template <Emrys::Recursion Inner>
class FadingMemory {
public:
  // The members the concept checks. StateType is what the steps act on, here
  // GaussianParams<State>; Reentrant is false for a recursion with scratch.
  using Scalar = typename Inner::Scalar;
  using SpaceType = typename Inner::SpaceType;
  using StateType = typename Inner::StateType;
  using ModelType = typename Inner::ModelType;
  using MethodType = typename Inner::MethodType;
  static constexpr bool Reentrant = Inner::Reentrant;
  template <Emrys::MeasurementModel M>
  using UpdateType = typename Inner::template UpdateType<M>;
  template <Emrys::MeasurementModel M>
  using ResultType = typename Inner::template ResultType<M>;

  FadingMemory(Inner inner, Scalar alpha)
      : m_inner{std::move(inner)}, m_alpha{alpha} {}
  ModelType const & model() const { return m_inner.model(); }
  ModelType & model() { return m_inner.model(); }
  MethodType const & method() const { return m_inner.method(); }

  // predict(state, dt, inputs...), or predict(state, inputs...) with a
  // fixed-step model
  template <typename... Args>
  void predict(StateType & state, Args const &... args) const {
    m_inner.predict(state, args...);
    state.setCovariance(m_alpha * m_alpha * state.covarianceLower());
  }
  // Returns UpdateType<M>, a std::expected whose value says whether the
  // policy accepted the measurement and gives the observation's
  // log-likelihood, by which a mixture reweights.
  template <Emrys::MeasurementModel M, typename Policy>
  auto update(StateType & state, M const & measurement,
              Policy const & policy) const {
    return m_inner.update(state, measurement, policy);
  }
  // The same into a result whose storage is reused.
  template <Emrys::MeasurementModel M, typename Policy, typename Result>
  auto update(StateType & state, M const & measurement, Policy const & policy,
              Result & result) const {
    return m_inner.update(state, measurement, policy, result);
  }

private:
  Inner m_inner;
  Scalar m_alpha;
};

using Inner = Emrys::KalmanRecursion<Emrys::Gaussian<State>, decltype(motion)>;
auto filter = Emrys::KalmanFilter{FadingMemory{Inner{motion}, 1.1}, initial};
// on each component
auto bank = Emrys::MixtureFilter{FadingMemory{Inner{motion}, 1.1}, mixture};
```

A recursion written from scratch conditions the state itself: `Emrys::update(state, measurement)`
gives the branches, one per measurement-noise component, the policy reduces them to the new state
or to nothing, and `update()` returns, in a `std::expected`, a result of its own with `accepted` and
the observation's `logLikelihood()`. An update that returns `accepted` false must not have written
the state, so that no filter copies a state to undo it. Beneath `update()`, `Emrys::condition(prior, predicted, z)`
conditions a prior on one prediction from `Emrys::transform()`, with no noise branches and no
policy. The policy arrives as the filter wraps it: `Emrys::reduceToOne<State>(policy,
std::move(measured.branches))` runs it on the branches and returns the state it keeps, or nothing.
The comment on `Recursion` lists every member a filter may
call, and which it calls only for `predictJoint()`. A mixture filter runs a
Gaussian component's update through the branches rather than the recursion when the policy may
keep several of them (one that adds to the filter's output, below), so a recursion over Gaussians
decides the update only under a policy that returns its state, such as `Collapse` or `Select`, or a
`Gate` over one. `tests/user_recursion.h` is this recursion, and the tests run it inside
both filters.

## Writing a reduction policy

An update gives one branch per measurement-noise component, and a policy reduces them to what
the filter keeps. `reductions::collapse()`, `select()`, `keep()`, and `gate()` over one of them are
the library's, and `update(measurement, policy)` takes any type that meets `Emrys::ReductionPolicy`
(`Models/Reduction.h`). A policy is called with the branches, as an rvalue, and may move their
posteriors out; each has its `logWeight` (its posterior probability among them), `posterior`,
`innovation`, `nis()`, and `logLikelihood()`. A policy sees the branches and nothing else: it never
sees the prior, so a rule that reads the prior is a recursion's (above). It does one of two things:

- returns the one state it keeps in a `std::optional`, or `std::nullopt` to keep none. What it
  returns stands for every branch it was given.
- takes the filter's output as a second argument and adds what it keeps by
  `out.add(logWeight, component)`, where `logWeight` is the log-probability among the branches of
  those the component stands for: a branch's own `logWeight` for a branch kept as it is, the log
  of their summed weights for branches merged. Adding nothing keeps none.

Keeping none rejects the measurement, and the filter keeps its prior. This one collapses, then
keeps the position variance from falling below a floor:

```cpp
struct Floor {
  double variance;
  template <typename Branches>
  auto operator()(Branches && branches) const {
    // Collapse always returns one.
    auto const g =
        *Emrys::reductions::collapse()(std::forward<Branches>(branches));
    auto cov = g.covarianceLower();
    for (int i = 0; i < 2; ++i) cov(i, i) = std::max(cov(i, i), variance);
    using Gaussian = std::remove_const_t<decltype(g)>;
    return std::optional{Gaussian{g.space(), g.mean(), cov}};
  }
};

auto const updated = filter.update(Emrys::observe(gps, z), Floor{0.81});
```

A policy of the wrong result fails the constraint at `update()`, which names what the policy
returned and what it had to. A `MixtureFilter`'s output holds several components
(`Emrys::MixtureReductionPolicy`), so a policy that adds several makes one prior component several,
as `reductions::keep()` does. This one keeps the branches of posterior probability at least 1%,
moving each in:

```cpp
auto keepLikely = [](auto && branches, auto & out) {
  for (auto & b : branches)
    if (b.logWeight >= std::log(0.01))
      out.add(b.logWeight, std::move(b.posterior));
};
```

The filter weights each component added by its prior component's weight, the observation's
likelihood, and `logWeight`, so branches a policy drops are not credited to the ones it keeps, as
with a gate. The output of a `KalmanFilter`, and of a filter that keeps one Gaussian per mode or per
point, holds one component, and a policy that adds a second fails an assertion.

## Writing a sigma-point set

A sigma-point method is a set of weighted points in the coordinates of a root of the covariance,
and `methods::cubature()`, `sphericalSimplex()`, and `gaussHermite()` make the library's. Any type
that meets `Emrys::PointSet` (`Models/Methods.h`) is one too, and so a method that `observe()`, a
filter's prediction, or `Emrys::methods::factored()` takes: two static members, the number of
points over $n$ coordinates, a constant expression so that a fixed-size input's points are sized at
compile time, and the points and weights themselves, filled into matrices the transform has sized.
This one is the symmetric set with a center weight, which `methods::unscented<1, 0, kappa>()`
reproduces:

```cpp
template <double CenterWeight>
struct SymmetricSet {
  static constexpr Eigen::Index pointCount(Eigen::Index n) { return 2 * n + 1; }

  // points is n by pointCount(n), weights has pointCount(n) entries, both
  // in the transform's scalar; returns the index of the point the output
  // mean is taken about, as a tangent-space mean.
  template <typename Points, typename Weights>
  static Eigen::Index unitPoints(Eigen::Index n, Points & points,
                                 Weights & weights) {
    using Scalar = typename Points::Scalar;
    Scalar const center = Scalar(CenterWeight);
    Scalar const radius = Eigen::numext::sqrt(Scalar(n) / (Scalar(1) - center));
    points.setZero();
    for (Eigen::Index i = 0; i < n; ++i) {
      points(i, 1 + i) = radius;
      points(i, 1 + n + i) = -radius;
    }
    weights(0) = center;
    weights.tail(2 * n).setConstant((Scalar(1) - center) / Scalar(2 * n));
    return 0;
  }
};

auto const updated =
    filter.update(Emrys::observe(SymmetricSet<0.5>{}, range, z));
```

The weights must sum to one and give the points mean zero and identity covariance, which is what
makes the points of a root $S$ of $P$ have mean $\mu$ and covariance $P$; the transform handles an
input of no coordinates itself. A fixed-size input arrives in fixed-size matrices, so a set may
compute its constants at compile time, as `methods::Cubature` does in `Models/Methods.h`.
`tests/user_point_set.h` is this set, and the tests run it through every check the library's sets
pass.

## Choosing the unscented parameters by the dimension

`methods::unscented<Alpha, Beta, Kappa>()` uses the same parameters whatever the number $L$ of
coordinates its points spread over. That number is the joint uncertain input of each transform:
the blocks a measurement model reads and its noise, or the state and the process noise, or the coordinates a
`methods::factored()` does not factor out. So a $\kappa$ written for the state's size is wrong for
most of them. `Emrys::methods::fourthMomentUnscented()` uses $\kappa = 3 - L$, which puts the
points at $\sqrt 3$ standard deviations and matches a Gaussian's fourth moment along each axis,
with a negative center weight above three coordinates. A rule of your own is any type that meets
`Emrys::UnscentedRule` (`Models/Methods.h`): one static member that gives $\alpha$, $\beta$, and
$\kappa$ over $n$ coordinates, computed in the transform's scalar so that a `float` filter computes
no `double`. This one keeps the center weight nonnegative:

```cpp
struct ClampedFourthMoment {
  template <typename Scalar>
  static constexpr Emrys::UnscentedParams<Scalar> params(Eigen::Index n) {
    // alpha, beta, kappa
    return {Scalar(1), Scalar(0), n < 3 ? Scalar(3 - n) : Scalar(0)};
  }
};

auto const updated =
    filter.update(Emrys::observe(ClampedFourthMoment{}, range, z));
```

`params()` must be a constant expression for a constant $n$. Over a fixed number of coordinates in
`float` or `double`, the compiler computes the weights from it, and parameters with
$\alpha^2 (n + \kappa) \le 0$ fail to compile. Over a dynamic number, they are computed on each
transform, and an assertion checks them. `tests/user_unscented_rule.h` is this rule.

## Writing axes

A sigma-point rule puts its points along the columns of a root $S$ of the covariance, $S S^T = P$,
and the root is the method's to choose. Every root the library takes uses
`roots::cholesky(roots::pivoted())` unless told otherwise: the Cholesky factor, and the pivoted
root where that fails, so it never fails itself. `Emrys::methods::spread(rule, axes)` gives a rule
other axes, any of `roots::cholesky()`, `pivoted()`, `principal()` (the eigenvectors, scaled),
`symmetric()` (the symmetric square root), and `along(d, ...)` (one column along a direction), and
the rule stays the rule: `spread(cubature(), roots::principal())` is a method wherever
`cubature()` is, as a measurement's method, a filter's prediction method, the points of `factored()`,
or the method `iterated()` relinearizes. The axes apply to the input marked by `withCross()`, the
state, and the other inputs take the default. Axes that can fail, as Cholesky does off the positive
definite cone, take a list of fallbacks tried in order, each a repair of the covariance
(`roots::ridge()`) or other axes: `roots::cholesky(roots::ridge(), roots::pivoted())`. A transform
returns no `std::expected`, so it refuses axes that can fail at compile time, with a message,
unless their list ends in axes that can't. `roots::squareRoot(axes, cov)` takes the root alone, as
a matrix, or a `std::expected` of one when the axes can fail, and `roots::squareRootWithReport()`
adds what each entry did.

Axes of your own are any type that meets `Emrys::Axes` (`Distributions/Roots.h`): `canFail`, and
`root(cov)`, reading only the lower triangle. This one is the reversed Cholesky root, upper
triangular so that the last coordinate's column varies alone:

```cpp
struct ReversedCholesky {
  // Cholesky fails off the positive definite cone.
  static constexpr bool canFail = true;

  // U with U U^T = P, upper triangular: J L J for the Cholesky factor L of
  // J P J, J the reversal.
  template <typename Derived>
  std::expected<Emrys::roots::RootMatrix<Derived>, Emrys::GaussianError>
  root(Eigen::MatrixBase<Derived> const & cov) const {
    using Matrix = Emrys::roots::RootMatrix<Derived>;
    Matrix const full = cov.template selfadjointView<Eigen::Lower>();
    auto const llt = Eigen::LLT<Matrix>{Matrix{full.reverse()}};
    if (llt.info() != Eigen::Success) {
      return std::unexpected{Emrys::GaussianError::NotPositiveDefinite};
    }
    Matrix const lower = llt.matrixL();
    return Matrix{lower.reverse()};
  }
};

// a std::expected
auto const upper = Emrys::roots::squareRoot(ReversedCholesky{}, cov);
auto const axes =
    Emrys::roots::withFallbacks(ReversedCholesky{}, Emrys::roots::pivoted());
// cannot fail
auto const method = Emrys::methods::spread(Emrys::methods::cubature(), axes);
auto const updated = filter.update(Emrys::observe(method, range, z));
```

A repair of your own meets `Emrys::Repair`: `repaired(cov)`, the covariance to retry as a
`std::expected`, such as an eigenvalue floor. Either may add a second overload taking a sink for a
report of what it did. `tests/user_axes.h` is both, and the tests run the axes through the roots
and the transforms.

A repair, the library's or your own, also repairs a covariance that fails to factorize in two other
places, and both refuse axes. A density: `params.toDensity(Emrys::roots::ridge())` keeps the
repaired covariance in `params`. A measurement's innovation covariance: the repair is an argument of the call,
`Emrys::observe(method, model, z, Emrys::roots::ridge())`, in any order with a covariance update
form such as `Emrys::updates::joseph()`. A measurement names its repair by its type and default-constructs it where an update needs one, so such a repair holds no
state.

## Linearizing about another estimate

A step normally linearizes its model at its own state. `Emrys::linearizedAbout(reference)` makes it
linearize about another estimate instead, then applies that linearization to its own state. This is
what a shadow filter does: it runs beside a tracker and is linearized about the tracker's
estimate, as in the linearized Kalman filter of Gelb (1974) about a nominal trajectory. The
reference is a point (a vector, which is copied) or a distribution (such as another filter's
state, which is referred to, not copied):

```cpp
auto tracker = Emrys::KalmanFilter{motion, prior};
auto shadow = Emrys::KalmanFilter{motion, prior};

// a copy, for the reference to refer to
Emrys::GaussianParams<State> const filtered = tracker.state();
// the reference goes before the inputs, if there are any
shadow.predict(dt, Emrys::linearizedAbout(filtered));
tracker.predict(dt);

Emrys::GaussianParams<State> const predicted = tracker.state();
auto const fromShadow =
    shadow.update(measurement, Emrys::linearizedAbout(predicted));
auto const fromTracker = tracker.update(measurement);
if (!fromShadow || !fromTracker) {
  // an innovation covariance was not positive definite, and that filter's
  // state is unchanged
}
```

A linear method (`linearized()`) takes the Jacobian at the reference's mean. A sigma-point method
spreads its points over the reference's distribution, so a point is refused at compile time. The
linearization is carried to the filter's own state, so on a quaternion or an angle the reference
may be in another chart. Linearized about the filter's own state, a step is the same step without a
reference. Linearized about the posterior of an update, it is the next iteration of
`Emrys::methods::iterated()`, which therefore refuses a reference of its own. The free functions
take it too: `Emrys::update(prior, measurement, about)`,
`Emrys::predict(method, prior, model, dt, about, inputs...)`, and
`Emrys::transform(method, f, policies..., about, withCross(x), inputs...)`, where the reference is
for the input right after it. Every form of the Gaussian, square-root and information included, takes it
on both steps.

## Considering a block

Some uncertain parameters of a model are better left unestimated: a bias or a scale factor known
only from its calibration, or one that the measurements observe so weakly that estimating it lets
it absorb errors it doesn't cause. Leaving it out of the state makes the filter overconfident,
since nothing then carries its uncertainty. A measurement can *consider* such a block instead, the
Schmidt-Kalman update of Schmidt (1966): the block stays in the state, the update leaves its mean
and its variance as they were, and the rest of the state, and its covariance with the block, are
updated as they would be if the block were estimated. The block's effect on the estimate stays in
that covariance.

```cpp
struct Bias : Emrys::Linear<1> {};
using State = Emrys::Composite<Heading, Bias>;

auto const fix =
    Emrys::observe(method, compass, z, Emrys::consider(Emrys::path<Bias>));
// the bias's mean and variance are the prior's
auto const updated = filter.update(fix);
```

`consider()` takes one or more paths, each a block of the state, and the model need not read it: a
gyroscope's bias that enters only through the prediction, as in
[consider_bias.cpp](../examples/consider_bias.cpp), is considered through its covariance with the
heading. Each measurement chooses, so one sensor can consider a block that another estimates. The
innovation is the full one, so the NIS, the log-likelihood, and a gate are as without it. It works
with every method and covariance update form, mixture noise, and `Emrys::methods::iterated()`, in
`KalmanFilter`, `MixtureFilter`, and the multiple model filters. The square-root and information
forms, the filters that weight points, and the Rao-Blackwellized filters refuse it at compile time.
A smoother run over a history whose updates considered a block smooths that block as if it had been
estimated.

## Checking an estimator

An estimator is checked in simulation, where the truth is known. `Emrys::simulate` draws the next
true state from the process model, and a measurement of it from a sensor, from the same models
the filter assumes:

```cpp
truth = Emrys::simulate<State>(truth, motion, rng, dt);
auto const z = Emrys::simulate<State>(truth, gps, rng);
```

Two statistics then test the filter's covariances against its errors:

- **NEES**, the normalized estimation error squared, against the truth. Feed `NeesMonitor` the
  final estimate of each of many independent runs: `monitor.add(truth, filter.state())`. A run's
  errors are correlated from one step to the next, so the estimates of one run over time are not
  independent, and the chi-squared test over them rejects a consistent filter far more often than
  its probability says. This is the test that finds a mistuned process noise.
  `NeesMonitor<Emrys::Cumulative>{}` keeps only the sum over every estimate it is given, which
  suits any number of runs; `NeesMonitor{n}` keeps the last `n`. To find which block is
  inconsistent, feed a second monitor one block, `monitor.add<Position>(truth, filter.state())`,
  which reads the block of the estimate and the same block of the truth by the path.
- **NIS**, the normalized innovation squared, which needs no truth and so also runs on a real
  system. The innovations of a consistent filter are independent, so `NisMonitor` can take every
  update of a run: `monitor.add(*updated)`. A particle filter's update has no innovation
  covariance to give it. Before the update, evaluate the measurement function at each particle,
  take the weighted mean and covariance of those values, add the noise covariance, and give the
  innovation's squared Mahalanobis length to `monitor.add(nis, dimension)`. It is approximate
  where the predicted measurement is far from Gaussian, as it is while the particles lie on a
  ring or in separate places.

`verdict(0.999)` tests the sum against two-sided chi-squared bounds and returns an
`Emrys::Consistency`: `Consistent` within them, `Overconfident` above them, and `Underconfident`
below them, each tail with probability 0.0005 for a consistent filter. The two failures differ.
An overconfident filter's covariance is smaller than its errors, so it reports its estimates as
better than they are: an error, which can leave an estimate far off and trusted. An underconfident
filter's covariance is larger than its errors, so its estimates are no worse than it reports: not
an error, only a loss of precision, such as from noise set larger than the system's. To test one
side at level $\alpha$, pass $1 - 2\alpha$ and read that side. `average()` is the sum over its
degrees of freedom, near 1 for a consistent filter. Both return a `std::expected`, which fails
with `Empty` when the monitor holds nothing, as a monitor fed only trusted runs does when none is
trusted, and with `NonFinite` when it holds the NaN that `NeesMonitor` adds for an estimate whose
covariance doesn't factorize. Compare the verdict, `monitor.verdict(0.999) ==
Emrys::Consistency::Consistent`, which is false when there is none; `!monitor.verdict(0.999)`
tests only whether there is one. The test is exact for a Kalman filter on a linear model. For an
extended or unscented filter it is close, and for the estimate of a mixture, an IMM, or particles
only approximate. [constant_velocity.cpp](../examples/constant_velocity.cpp) sets the pattern.
Random draws differ between standard libraries, so a seed that passes with one may not with
another.

A mixture's update has no NIS of its own: `MixtureFilterUpdate::components[i]` is prior component
$i$'s update, with its branches' `nis()` and its `logLikelihood()`, and no published test over them
ships. In a mixture of IMMs, component $i$'s update is an IMM's, one per mode:
`components[i].modes[m].branches.front().nis()` is its NIS given mode $m$, read when
`components[i].modes[m].accepted`. A statistic you keep per component, such as its running NIS,
must follow the components through the merges and prunes between steps, and the state's lineage
carries it.
`startLineage()` marks the components as the parents; each row of the lineage then says what share
of a component came from each parent, and `lineage().carry(parents, children)` gives each component
the share-weighted mean of its parents' values, so a surviving component keeps its sum and a merge
blends two. A filter that records for a smoother marks the parents itself, in `predictJoint()`, so
leave the call out there and carry once per cycle:

```cpp
// one sum per component
Eigen::VectorXd running = Eigen::VectorXd::Zero(filter.state().size());
...
// the components about to be updated are the parents
filter.state().startLineage();
auto const updated = filter.update(Emrys::observe(gps, z));
if (!updated) { /* an innovation covariance was not positive definite */ }
for (Eigen::Index i = 0; i < running.size(); ++i) {
  auto const & component = updated->components[i];
  if (component.accepted) running(i) += component.branches.front().nis();
}
// prune or merge the components through filter.state()
auto carried = Eigen::VectorXd(filter.state().size());
// a dropped component's sum goes; a merge blends two
filter.state().lineage().carry(running, carried);
running = carried;
```

**When to trust an estimate.** A mean and covariance describe a posterior only when it is close
to one Gaussian. A Kalman filter assumes that and can't tell when it fails. A
mixture or particle filter carries the true shape, so it can report alongside its moments whether
they describe it. Read three things from the state at each step, none of which needs the truth:

- that the representation is healthy: the update was accepted, and for particles the effective
  sample size, `state().ess()`, a `std::expected` like the set's other weight functions, is
  large enough to represent the posterior;
- that the posterior is near-Gaussian. For a mixture, an IMM's hybrid state, or a mixture of
  them, `kullbackLeiblerCollapseBounds()` of its density brackets the divergence from it to its
  collapse: the `upper` end is small, such as below 0.1 nats, when the collapse describes it,
  and a `lower` end above that says it does not. The two ends are up to the entropy of the
  weights apart, so the upper end can vouch for a mixture of a few components but never for
  thousands of grid cells read as one. Between the two ends the bracket decides nothing, and a
  converged mixture often lands there: the upper end overstates the divergence of components
  that overlap, giving 0.35 nats for two equal unit-variance Gaussians at ±1 whose divergence
  from their collapse is 0.009, so a few overlapping components left after the mixture converges
  keep it above the threshold. Read that case as undecided, not as untrusted. For a Gaussian
  mixture, decide it by estimating the divergence itself,
  `Emrys::kullbackLeibler(Emrys::MonteCarlo{10'000}, *density, *collapsed, rng)` with `collapsed`
  from `density->collapsed().toDensity()`, and compare the estimate's `value` with the threshold;
  it evaluates both densities at each draw, so run it only when the bracket is undecided. The
  estimate does not take a mixture of IMMs. Merging the components does not decide it either: a
  merge of two lowers the upper end by exactly its cost, which bounds what the merge changed, so
  the upper end read before the merges is still the bound on the state the filter had. For particles,
  test whether the particles, whitened by their own moments, have third moments near 0 and a mean
  fourth power of their length near d(d + 2) in d dimensions, 8 in two. These are Mardia's
  multivariate skewness and kurtosis;
- that the measurements agree with the model: a windowed `NisMonitor`, such as
  `NisMonitor<Eigen::Dynamic>{20}`, whose verdict `v = nis.verdict(0.999)` passes
  `v && *v != Emrys::Consistency::Overconfident`. An underconfident window is no reason to
  withhold trust, since its estimates are no worse than they claim, but a missing verdict is: the
  window is empty, or holds a NaN.

A mixture of IMMs, `MixtureParams<HybridGaussian<State, Mode>>`, has the same two estimates as
a mixture: the `moments()` of `state().toDensity()`, over every component and mode, and the most
probable component, `state()[state().mostProbable()].component`, whose density's `collapsed()`
is its one Gaussian over the modes. The weights of a mixture's components and of a
hybrid's modes are logs, not normalized: `Emrys::logSumExp()` of an Eigen array of them, or
`logTotalWeight()` of the mixture or of `modes()`, is the log of their total. The bracket takes
the mixture of IMMs directly, over the continuous state, with each mode's Gaussian weighted by
its component's weight and its mode's:

```cpp
auto const density = filter.state().toDensity();
if (density) {
  auto const bounds = Emrys::kullbackLeiblerCollapseBounds(*density);
  bool const nearGaussian = bounds.upper < 0.1;
  bool const farFromGaussian = bounds.lower > 0.1;
}
```

Test such an estimator over the estimates it trusts: the NEES of the final estimate of each run
that trusts it, and of the first estimate each run trusts, which tests that trust doesn't come
too early. Count the runs that end untrusted as well, since an estimator that never trusts
itself passes trivially. Expect some: an estimator that withholds trust under faults also
withholds it from some runs in which the world matches the model, as many as half of them in a
test of a mixture of IMMs. Loosening the thresholds trusts more of those runs and more of the
faulted ones too, so set them by the count of trusted-but-far runs the application can bear.
Then run the simulation with the world and the model apart, such as
measurement noise three times the filter's, and check that trust is withheld. Run it too with
the faults the real system will have, such as an outlier while the posterior is still spread
and a gap in the measurements. Count the runs that end far from the truth but trusted. Whether a
filter's failures are flagged or silent depends on its upkeep and thresholds as much as on
its representation, so measure it rather than assume it. Some errors can't
be seen this way: a constant bias that the measurements can't tell from a shift in the state,
such as an uncalibrated delay on one receiver of a time-difference-of-arrival pair, moves the
estimate with the measurements still consistent. A bias like that belongs in the state, with its
calibration uncertainty as its prior. A bias or parameter in the state that drifts at more than
one rate, such as an oscillator's frequency offset that holds steady and then ramps as its
temperature changes, defeats a single random walk: a walk slow enough for the steady periods
lags the ramps, and the lag goes into the rest of the state, where the estimate can end far
off and still trusted, while a walk fast enough for the ramps costs precision throughout. Give
each rate a mode of an `ImmFilter`, or of a mixture of IMMs if the prior is wide as well.

Mixture noise for outliers takes two numbers the library leaves to you and recommends no value
for: the outlier component's probability and its covariance. What it can't change is how an
update tells the branches apart. Each branch's posterior weight is its probability times the
likelihood of the innovation under that branch's innovation covariance, which is the covariance
of the predicted measurement plus the branch's noise covariance. While the state is uncertain,
the predicted measurement's covariance can exceed both noise covariances, and then the branches
give an innovation nearly the same likelihood: an outlier keeps the inlier branch's weight and
moves the estimate. However wide the outlier component, an update can't tell an outlier from an
inlier until the predicted measurement's spread is well below the outlier component's. The
faulted simulation above, with an outlier while the posterior is still spread, measures what
that costs, and is the test of the probability and covariance you choose.

## What is not built yet

- Splitting a Gaussian where a model is too nonlinear for it, fusion of estimates, out-of-sequence
  measurements, and consensus filters.
- The ensemble Kalman filter, deferred past the first release.
- Built-in process models for a coordinated turn and for a gyroscope: the examples write these
  themselves.
- Batched functions throughout. Every method takes a function marked `batched(f)`
  (`Models/Batched.h`), which takes every point at once, but only the particle filter and the
  Rao-Blackwellized reweighting call it once per set; the sigma-point transforms still call it
  once per point.
- Counter-based random numbers and a GPU backend. Every loop over a count already takes an
  execution policy.
- Concepts with short compile errors for the model contracts, and `checkModel<Model, Space>()`.
- A particle filter's NIS, and a test of whether a particle set's moments describe it: the
  caller computes both, as "Checking an estimator" says. There is no such test for a grid yet.
- Non-Gaussian process noise: every filter's prediction, the particle filter's included, draws or
  propagates a Gaussian.
- An update that reports a failing covariance root as its error: a transform refuses axes that can
  fail at compile time, so give them a fallback that can't ("Writing axes").

## Sources

The library implements only algorithms from the public literature. Each filter's documentation
cites its source, and [reference/bibliography.md](reference/bibliography.md) collects them, so a
request for a new method names the paper it would implement.
