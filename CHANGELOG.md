# Changelog

Every change a user can see, newest first. Versions follow [Semantic Versioning](https://semver.org/):
before 1.0, a minor release may break code, and its entry says what breaks and what to write
instead.

## [Unreleased]

### Added

- `KalmanFilter<...>::UpdateType<M>` and `ResultType<M>`, what `update()` with a measurement of
  type `M` returns and fills, as the other filters name them; it was
  `decltype(filter)::RecursionType::UpdateType<M>`.
- `ParticleGaussianParams::forEachParticle(policy, f)`, which calls `f` on each particle as a
  `ParticleSet`'s `forEach()` does, to move the particles in place and keep their Gaussians.
- An update takes a sigma-point method spread by axes that can fail, such as
  `spread(cubature(), roots::cholesky())`, and returns `GaussianError::NotPositiveDefinite` when
  the axes cannot take the root of the blocks the measurement reads; it was a compile error. A
  transform or prediction still refuses such axes, and so does an update whose model's `affineIn()`
  names some of the blocks read.
- `SECURITY.md`: how to report a vulnerability privately, and what counts as one. Among them is
  undefined behavior, or a call that never returns, caused by a floating-point value passed to the
  library, NaN and infinities included, in a build with `NDEBUG`.
- A measurement per value of a mode axis may have noise by another axis:
  `byMode<Maneuver>(when<Cruise>(fix), when<Turn>(biasedFix))` where each fix's noise is
  `additiveNoise(byMode<Sensor>(...))`. An axis that chooses twice along a nesting fails to compile.
- A `MixtureFilter` of hybrids, and an `RbPointMassFilter` or `RbParticleFilter` under a switching
  model, take a measurement per value of a mode axis, as the IMM does.
- `observe(method, byMode<Axis>(when<Value>(model), ...), z)` observes a model per value of an
  axis with one observation, making a measurement per value.
- `Zero<Path>{variance}`, a built-in process model that resets its block to zero with that variance
  on each coordinate, independent of the rest of the state, over any step: the model of a block a
  switching model's mode doesn't use and in which the quantity is zero, such as a
  constant-velocity mode's acceleration beside a constant-acceleration mode, where `Constant`
  hands the old acceleration back at the next mixing.
- `CategoricalParams<Mode>::independent(maneuver, sensor)` builds the distribution on a
  `DiscreteProduct` whose axes are independent from one distribution per axis, as
  `MarkovChain<Mode>::independent()` builds the chain, so an IMM's mode prior is no longer written
  in flat-index order; `independent(space, axes...)` takes the space, for a run-time cardinality.
- `KalmanFilter`, `MixtureFilter`, `ImmFilter`, and `AmmFilter` have `emptyUpdate<M>()`, the empty
  result that `update(measurement, policy, result)` refills, made from the state's allocator with
  room for every component or mode, so a reused result is one line:
  `auto result = filter.emptyUpdate<Fix>();`. The IMM's takes a `byMode` of measurements too. It
  replaces `filter.recursion().emptyUpdate<Fix>(filter.get_allocator(), filter.state().size())`
  and a result built from the allocator and reserved by hand, which still work.

### Changed

- **Breaking:** `Roughen` moves the particles of a Rao-Blackwellized state in place, so it no
  longer allocates a copy of the particle set at each call. A state of the user's own that it
  roughens needs `forEachParticle(policy, f)`, as `ParticleGaussianParams` has, in place of
  `setParticleStates()`.
- [docs/choosing.md](docs/choosing.md), "Starting without a prior", recommends starting a
  `SqrtInfoFilter` from no information, the diffuse initialization of the literature, over a
  least-squares fit to the first measurements, and says when to convert once the state is
  determined. It and [docs/usage.md](docs/usage.md) now say that `InfoFilter` can't predict
  until the measurements determine every direction, where `SqrtInfoFilter` can.
- **Breaking:** a filter over a hybrid takes the mode chain and the motion as its first two
  arguments, `ImmFilter{chain, motion, prior}`, and so do `MixtureFilter`, `RbParticleFilter`,
  and `RbPointMassFilter` (`RbPointMassFilter{chain, motion, grid, conditional}`). The motion is
  one process model every mode follows, or `byMode<Maneuver>(when<Cruise>(cruise),
  when<Turn>(turn))`, a model per value of a mode axis in any order, nested where a model depends
  on two axes. Write `ImmFilter{chain, byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(turn)),
  prior}` in place of `ImmFilter{SwitchingModel{chain, cruise, turn}, prior}` or
  `ImmFilter{switchingModel<Maneuver>(chain, cruise, turn), prior}`; a model per mode of a product
  in flat-index order is `byMode<Mode>(m0, m1, m2, m3)` by the product itself. `AmmFilter` takes
  the motion first and the method and upkeep after the state, as the other filters do:
  `AmmFilter{motion, prior}`, `AmmFilter{motion, prior, method}`, `AmmFilter{motion, prior,
  upkeep}`, in place of `AmmFilter{prior, slow, fast}`, `AmmFilter{method, prior, ...}`, and
  `AmmFilter{upkeep, prior, ...}`. `SwitchingModel<ModeSpace, Scalar, Motion>` replaces
  `SwitchingModel<ModeSpace, Scalar, Path, Models...>`, built `SwitchingModel{chain, motion}`;
  `switchingModel<Path>()`, `model<I>()`, and `modelIndex()` are gone, and `motion()` returns the
  motion, whose `alternative<I>().part()` is a model.
- **Breaking:** a part of a model that depends on the mode is
  `byMode<Axis>(when<Value>(part), ...)`, its alternatives labeled by the value of the mode axis
  each serves, in any order; a value named twice, left out, or not the axis's fails to compile,
  and `otherwise(part)` serves the values not named. Noise by mode axis is
  `additiveNoise(byMode<Sensor>(when<Inlier>(inlier).weight(0.9), when<Outlier>(outlier).weight(0.1)))`,
  one `GaussianParams` per value; write it in place of
  `additiveNoise(mixture).byMode<Sensor>()`. The weights are what a filter without the axis reads
  the noise by, as the mixture's were; without them such a filter refuses the noise at compile
  time. A measurement per value of an axis is `byMode<Sensor>(when<Inlier>(nominal),
  when<Outlier>(biased))`; write it in place of `switchingMeasurement<Sensor>(nominal, biased)`.
  It takes no weights, and only a filter whose state holds hybrids takes it.
  `SwitchingMeasurement`, `switchingMeasurement()`, `AdditiveNoise::byMode()`, and
  `ImmRecursion::SwitchingResultType` are gone; `ImmRecursion::ResultType<M>` takes the `byMode`.
- **Breaking:** `ImmFilter` takes a recursion, as `KalmanFilter` and `MixtureFilter` do:
  `ImmFilter<Component, Model, Method, Allocator, Recursion, Upkeep>`, the `ImmRecursion` unless
  given one, and `ImmFilter{recursion, state}` runs a recursion of your own over the hybrid, such
  as one that wraps the `ImmRecursion` and changes a step. Code that deduces the filter is
  unchanged; code that names its upkeep in the type writes the recursion before it:
  `ImmFilter<C, Model, Method, Allocator, ImmRecursion<C, Model, Method, Allocator>, Upkeep>` for
  `ImmFilter<C, Model, Method, Allocator, Upkeep>`. `AmmFilter` keeps its parameters and runs the
  `ImmRecursion`.

- **Breaking:** `MarkovChain::fromDwellTimes()` and `ContinuousTimeMarkovChain::fromDwellTimes()`
  are renamed `fromSojournTimes()`, with the same arguments: write
  `MarkovChain<Mode>::fromSojournTimes(tau, exits)` for `MarkovChain<Mode>::fromDwellTimes(tau, exits)`.
  Sojourn time is the tracking literature's name for the time spent in a mode, and a dwell is
  already a radar's look at one position.
- **Breaking:** the chain on a `DiscreteProduct` whose axes move independently is built by the
  factory `independent()`, not a constructor, for `MarkovChain` and `ContinuousTimeMarkovChain`:
  write `MarkovChain<Mode>::independent(maneuver, sensor)` for `MarkovChain<Mode>{maneuver, sensor}`,
  and `MarkovChain<Mode>::independent(space, maneuver, sensor)` for
  `MarkovChain<Mode>{space, maneuver, sensor}`. A constructor now takes only what the chain
  stores, and the factories compute it.
- **Breaking:** `MarkovChain` and `ContinuousTimeMarkovChain` are built from a matrix by
  `create()`, which checks it and returns a `std::expected` with a `ChainError`; the constructor
  that took a matrix is private. Write `MarkovChain<Maneuver>::create(matrix).value()` for
  `MarkovChain<Maneuver>{matrix}`, `MarkovChain<Maneuver>::create(matrix, space)` for
  `MarkovChain{space, matrix}`, and the same for `ContinuousTimeMarkovChain`; handle the error
  where the matrix is computed at run time. `MarkovChain::create()` divides each row by its total,
  so it takes weights in any proportion as well as probabilities, which no longer need to sum to
  one to `dummy_precision`. `ContinuousTimeMarkovChain::create()` reads only the rates off the
  diagonal and sets each diagonal entry to minus its row's total. Both cast the matrix to the
  chain's scalar, `double` unless named, where the constructor took the matrix's.
- **Breaking:** `fromSojournTimes()` returns a `std::expected` with a `ChainError` in place of
  asserting: write `.value()` after it, or handle the error. The diagonal of its exit weights is
  no longer read, where it had to be zero.
- **Breaking:** `MarkovChain::persistent()` returns a `std::expected` with
  `ChainError::NotAProbability` when the probability of staying is not in [0, 1], in place of
  asserting: write `MarkovChain<Maneuver>::persistent(0.95).value()` for
  `MarkovChain<Maneuver>::persistent(0.95)`, or handle the error.
- **Breaking:** a `SwitchingModel` moves the mode over the same step as its models. With models
  stepped by `predict(dt)`, its chain is a `ContinuousTimeMarkovChain`, and each prediction moves
  the mode by $e^{Q \Delta t}$, where it moved one step of a `MarkovChain` whatever `dt` was; a
  `MarkovChain` there is now a compile-time error. Write
  `SwitchingModel{ContinuousTimeMarkovChain<Maneuver>::persistent(0.05), cruise, turn}` for
  `SwitchingModel{MarkovChain<Maneuver>::persistent(0.95).value(), cruise, turn}`, with each rate
  about the old probability of leaving divided by the step, or make the models fixed-step with
  `fixedStep(model, dt)` to keep the `MarkovChain`. A `ContinuousTimeMarkovChain` given with
  fixed-step models is taken over their step. `switchingModel<Axis>()` takes either chain the same
  way, `transition(dt)` (or `transition()` with fixed-step models) returns the chain over a
  prediction, and `predict(0.0)` no longer moves the mode. The IMM, the AMM, and the
  Rao-Blackwellized filters follow.
- `ChainError` gains `Negative`, `NonFinite`, `ZeroRow`, `ShortSojourn`, and `NotAProbability`, and
  prints in words.

### Fixed

- `predict()` and `predictJoint()` by a fixed-step model that caches its transition, and an
  `RbParticleFilter`'s prediction of each particle's Gaussian by one, read only the lower triangle
  of the covariance, as every other step does; they multiplied the whole stored matrix, whose
  upper triangle need not hold the covariance.
- `UpdateBranch::component` and `FilterBranch::component` are 0 in a default-constructed branch,
  not indeterminate.

## [0.1.0] - 2026-11-13

The first release: a header-only C++23 library on Eigen 5.0 for sequential Bayesian estimation,
used through `find_package(EmrysFilter 0.1 CONFIG)` or `FetchContent`.

### Added

- **Spaces:** a state as a `Composite` of named blocks, each a vector (`Linear`), an angle
  (`Periodic`, `RadiansPi`), a `UnitQuaternion`, a `Repeated` or resizable block, or a discrete
  mode, with `plus()` and `minus()`, tangent-space covariances and means, the covariance reset,
  and a `Space` concept for a space of your own.
- **Distributions:** the Gaussian in covariance, square-root, information, and square-root
  information forms; Gaussian and uniform mixtures; sample and particle sets; kernel densities;
  categoricals and continuous-time Markov chains; hybrids of a discrete mode and a continuous
  state; grids of Gaussians and particles carrying Gaussians; Student-t, Gamma, uniform, and
  products of them; marginals, conditionals, moments, sampling, fitting, and divergences.
- **Models:** process models composed from blocks (`ConstantVelocity`, `OrnsteinUhlenbeck`, and
  your own); measurement models that read and write the blocks they name and choose their method
  (`methods::linearized()`, `unscented()`, `iterated()`, `gaussHermite<N>()`, and the other
  sigma-point sets), Jacobians by automatic differentiation or returned with `withJacobian()`;
  additive, mixture, and by-mode noise; switching models and measurements; emissions for hidden
  Markov models; functions marked `batched(f)`, which take every point at once; simulation; the
  NIS and NEES consistency monitors; and the posterior Cramér-Rao bound.
- **Filters:** `KalmanFilter`, `SqrtKalmanFilter`, `InfoFilter`, `SqrtInfoFilter`,
  `GaussianMixtureFilter`, `ImmFilter`, `AmmFilter`, `ParticleFilter`, `RbParticleFilter`,
  `RbPointMassFilter`, `PointMassFilter`, and `HiddenMarkovFilter`. Each update returns its
  innovation, log-likelihood, and NIS, or the reason it failed as a `std::expected` error.
  Updates take reductions (`reductions::gate()`, `select()`, `collapse()`), and the filters take
  upkeep (`upkeep::resample()`, `roughen()`, `pruneBelow()`, ...) run on a trigger.
- **Smoothers:** fixed-interval and fixed-lag histories for Gaussian, hybrid (IMM), mixture,
  particle, and categorical filters, and `mostProbableSequence()`.
- **Execution:** every loop over a run-time count takes an execution policy,
  `execution::sequential` by default, a standard policy, or your own.
- **Allocation and embedded builds:** nothing sized by a bounded space allocates, as tested under
  `EIGEN_RUNTIME_NO_MALLOC`; run-time counts take a standard allocator; no exceptions are thrown,
  and the library builds with `-fno-exceptions -fno-rtti` and for Cortex-M4F and M7. The scalar is
  a template parameter, `float` included.
- **Printing:** `operator<<` and `std::format` of the library's types, with a precision
  (`std::println("{:.3}", gaussian)`).
- **Documentation:** the [reference](https://emrys-math.github.io/emrys-filter/), the usage page
  and filter-choosing guide in `docs/`, eleven examples run as tests, and the
  `emrys-filter-build-estimator` skill for a coding agent working in your project.

### Known limitations

- [docs/usage.md](docs/usage.md), "What is not built yet", lists what is missing.
- The design reviews in [docs/guide/](docs/guide/) hold open questions whose answers may break
  code in a later 0.x release, among them the hidden Markov filter's argument order and emissions,
  the particle and Rao-Blackwellized filters, and the smoothers' value sequences.

[Unreleased]: https://github.com/emrys-math/emrys-filter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/emrys-math/emrys-filter/releases/tag/v0.1.0
