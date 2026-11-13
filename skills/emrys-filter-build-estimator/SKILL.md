---
name: emrys-filter-build-estimator
description: Builds a state estimator with EmrysFilter, the header-only C++23 library of Kalman, unscented, particle, interacting multiple model, and Rao-Blackwellized filters on Eigen, and checks it in simulation. Use when writing C++ that estimates a state from noisy measurements with EmrysFilter, such as tracking, navigation, attitude estimation, sensor fusion, an EKF or UKF, a particle filter, or a multiple-model filter, or when checking such an estimator's consistency with NEES and NIS.
license: MIT
metadata:
  emrys-filter-version: "@EMRYS_FILTER_VERSION@"
---

# Building an estimator with EmrysFilter

This skill was installed with EmrysFilter @EMRYS_FILTER_VERSION@, and its files describe that
version. This file is the procedure. The facts about the library are in
[references/usage.md](references/usage.md), about 970 lines: read it once before writing code.
[references/choosing.md](references/choosing.md) is the reasoning behind the choice of filter, read
only when step 2 sends you there. The programs in [examples/](examples/) are complete estimators,
each with the check of step 5.

## 1. Write the problem down

Before any code, write down, in a comment at the top of the program:

- the state: its blocks, their units, and which of them are angles or attitudes;
- the motion of each block, and whether the time step is fixed;
- each measurement: the function of the state it measures, its rate, and its noise covariance;
- the process noise of each moving block, as a continuous-time spectral density;
- the prior: the initial mean and covariance.

Where the task doesn't give a noise level, ask for it. If you must assume one, say so in that
comment and in your reply. A filter whose noise levels are invented can pass every check in
simulation and still be wrong on the real system.

## 2. Choose the filter and its template

Answer the questions of "Which filter?" in usage.md one at a time: how nonlinear the process
model is and how nonlinear each sensor is, judged separately; how far the posterior is from one
Gaussian; how the noise is distributed; whether the motion changes regime; and whether the
arithmetic can be trusted. Take the default answer where it holds. For a question where it
doesn't, read that question's section of choosing.md, and no more of it. Then find the
combination in the table at the end of "Which filter?".

Where the posterior can't be one Gaussian, or more than one row fits, read "Choosing a
representation" in choosing.md before choosing. Carry the blocks the models are affine in exactly,
choose a mixture, a grid, or particles for the rest, keep that choice for the whole run, and gate
measurements before upkeep that discards.

Where the table names an example, read that example in full and use its structure: the space, the
model, the sensors, the prior, the simulation loop, and the checks. Take its structure, not its
numbers.

| Example | Shows |
|---|---|
| [constant_velocity.cpp](examples/constant_velocity.cpp) | The smallest complete estimator: a linear Kalman filter and the checks |
| [linear_kalman.cpp](examples/linear_kalman.cpp) | The textbook Kalman filter on a plain vector, its models given by F, H, Q, and R: constant_velocity.cpp's filter written another way |
| [unicycle.cpp](examples/unicycle.cpp) | An EKF and a UKF side by side, a process model written by the user, range and bearing |
| [attitude.cpp](examples/attitude.cpp) | A unit quaternion state, a gyroscope as a prediction input, vector measurements |
| [particle_filter.cpp](examples/particle_filter.cpp) | A particle filter on a posterior split between two places, and its upkeep |
| [imm.cpp](examples/imm.cpp) | An interacting multiple model filter over a switching model |
| [gaussian_mixture.cpp](examples/gaussian_mixture.cpp) | A Gaussian mixture filter under outliers, and a merge of its components written by the user |
| [rao_blackwellized.cpp](examples/rao_blackwellized.cpp) | A Rao-Blackwellized particle filter |
| [point_mass_filter.cpp](examples/point_mass_filter.cpp) | A grid filter, and a measurement known only by its likelihood |

## 3. Write the space, the models, and the filter

Follow "The pieces of an estimator" in usage.md, in this order:

1. The state space: one tag type per block, derived from a coordinate type, and a `Composite` of
   them. Angles are periodic coordinate types, never `Linear`.
2. The process model: a built-in model for each block of the state where one fits, and a
   `processModel(f, reads(...), writes(...), ...)` of your own where none does ("Writing a process
   model" in usage.md), composed with `compose(models...)`. Every coordinate must be written by
   exactly one model. An input of the call, such as a gyroscope's reading, is a tag the model
   reads and the state lacks, given to `predict()` as `sure<Tag>(value)` or as a Gaussian over the
   tag. Wrap the model in `fixedStep(model, dt)` when the step never changes; the filter binds it
   to its state's space.
3. One measurement model for each kind of measurement: `measurementModel(h, reads(path<Block>...),
   additiveNoise(noise))`, whose function `h(in, out, x)`, `h(in, x)`, or `h(x)` reads each block it
   names with `block<Tag>(in, x)`. A model without `reads()` reads the whole state, which suits a
   plain `Linear<N>` state with nothing to name. A linear one gives its matrix in place of `h`, or its `Columns`, a matrix
   per block, with `onEach<>()` for a repeated block such as landmarks. A function for
   `Linearized` is generic in its scalar (see "Conventions"). A measurement of an angle names
   its periodic space with `outputSpace()`. The noise is additive, `additiveNoise(R)` of its
   covariance alone when it is zero-mean, a Gaussian, or a Gaussian mixture, or an input of the call, named by its tag in `reads()` and passed to `observe()`
   after the observation. A measurement whose error is none of these is known only by its
   likelihood: a model with `reads()` and `logLikelihood(g)`, `g(in, x, z)`, and no function,
   which the particle, grid, and Rao-Blackwellized filters take and the Kalman filters refuse
   (point_mass_filter.cpp). A measurement is `observe(method, model, z)`, the method
   deciding how a Kalman filter carries its uncertainty through `h`, or `observe(model, z)` for
   a model given its matrix, which every method updates exactly, and for a filter that only
   weights points.
4. The prior, as `GaussianParams` (or particles), and then the filter, built from the model and
   the prior.

Build in Debug first: the library's assertions check what the types can't, such as a process model
whose blocks don't cover the state exactly once.

## 4. Mistakes that compile

Check the program for each of these before running it:

- An error taken with `a - b` on a state that has angles or a quaternion. Use `Emrys::minus()`.
- A covariance sized by the state's values rather than its differences: a quaternion has 4 values
  and 3 differences.
- Process noise given as a variance per step rather than as a spectral density, which is wrong by
  a factor of the step.
- An unchecked result: a Gaussian filter's `update()` returns a `std::expected`, and an update
  can also be rejected, which `accepted` reports.
- A particle filter that never resamples: after each step's updates, apply upkeep to
  `filter.state()`, or give the filter its upkeep and call `filter.maintain()`.
- A particle filter over a state with little or no process noise, such as a stationary target,
  that resamples without a `Roughen` after the `Resample`: its cloud collapses onto a few points.
- Degrees where the library expects radians, or the reverse: the coordinate type sets the unit.

When a compile error is long, find the first `error:` line and the first line in your own file
below it. The usual causes are a `Linearized` function written for `double` rather than generic
in its scalar, a function that returns an Eigen expression rather than an `Eigen::Matrix`, and a
`block<Tag>(in, x)` of a tag that `reads()` does not name.

## 5. Check it in simulation

Every estimator gets the check that the examples have, before it is called done:

1. Simulate the truth and the measurements with `Emrys::simulate()`, from the same model and
   sensors the filter uses.
2. Run many independent runs, each with a new filter and a true state drawn from the prior. Give
   `NeesMonitor` the final estimate of each run, and `NisMonitor` every update of every run.
3. Print the RMS error and both averages, and fail unless `verdict(0.999) ==
   Emrys::Consistency::Consistent` for both. Each of `average()` and `verdict()` returns a
   `std::expected`, which fails on a monitor that holds nothing.

"Checking an estimator" in usage.md says why NEES takes one estimate per run, and how to compute
a particle filter's NIS, which its update doesn't give `NisMonitor` directly.

Where the user needs to know when the estimate can be trusted, or the posterior can be far from
one Gaussian, follow "When to trust an estimate" in usage.md: report a verdict with each estimate,
run the NEES test over the estimates the verdict trusts, count the runs that end untrusted, and
run the simulation once more with the measurement noise larger than the filter assumes, to check
that the verdict withholds trust. Run it again with the faults the system will have, such as an
outlier and a gap in the measurements, and count the runs that end far from the truth but
trusted. Name to the user the errors the verdict can't see, such as an unmodeled constant bias.

The simulation draws from exactly the models the filter assumes, so a filter that fails this
check has an error in its code, not in its tuning: a unit, a sign, a block in the wrong place, or
a measurement function that differs from the one simulated. Find that error; don't change the
filter's noise to make the check pass. An average NEES far above 1 means the filter is
overconfident, and far below 1 that it is underconfident. On real data an underconfident filter
is only imprecise, since its estimates are no worse than it reports, but here, where the world is
the filter's own model, either verdict points to the same kind of error. The test is exact only for a Kalman
filter on a linear model. For an extended or unscented filter on a strongly nonlinear problem, or
the estimate of a mixture, an IMM, or particles, it is approximate, and a small excess can come
from the approximation rather than an error: report it with the numbers rather than hiding it.
Random draws differ between standard libraries, so one failing seed is evidence, not proof: try a
few seeds before concluding.

On real data, where there is no truth, NIS is the check that still runs.

## 6. Report

Tell the user what you built, which filter and why, every noise level you assumed, and the check's
numbers: the RMS error, the average NEES, and the average NIS. With a trust verdict, report too
how many runs it trusted and how soon, the NEES over trusted estimates, and what the mismatched
and faulted runs showed.
