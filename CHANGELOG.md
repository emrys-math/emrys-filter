# Changelog

Every change a user can see, newest first. Versions follow [Semantic Versioning](https://semver.org/):
before 1.0, a minor release may break code, and its entry says what breaks and what to write
instead.

## [Unreleased]

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
