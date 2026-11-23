# Benchmarks

Each benchmark solves one problem twice: once with EmrysFilter, and once in plain Eigen as a careful
user would write it by hand. Every number is the library's cost divided by the hand code's, so 1.00
means the same cost, above 1 the library costs more, and below 1 it costs less.
[design/benchmarks.md](design/benchmarks.md) explains which claim each benchmark backs, why the
reference is hand-written code, and what is still to be measured.

## Summary

| Problem | Instructions per step | Time per step, Windows | Time per step, Linux |
|---|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.39–1.50 | 1.15–1.68 | 1.13–1.42 |
| Ranging EKF, 2D, 8 anchors | 1.09–1.15 | 1.02–1.44 | 1.07–1.15 |
| Ranging EKF, 3D, 6 anchors | 1.16–1.21 | 1.11–1.21 | 1.10–1.13 |
| Ranging EKF, 12 or more anchors | 0.89–1.00 | 0.91–0.97 | 0.89–1.07 |
| Attitude EKF, analytic Jacobians | 1.47 | 1.15 | 1.21 |
| Attitude EKF, autodiff | 1.46 | 1.32 | 1.29 |
| Particle filter, 2D | 1.08–1.12 | 1.10–1.18 | 1.00–1.04 |
| Particle filter, 3D | 1.07–1.13 | 1.25–1.40 | 1.02–1.12 |

A range covers every case of its row: `double` and `float`, analytic and automatic Jacobians, and
for the particle filter its likelihoods, and in its times every particle count. Neither side
allocates on the heap during a step, in any case.

The instruction column, and the Linux times of the ranging rows, were measured on 2026-11-23 at the
commit *Stop building a step's Jacobian where nothing reads it*. The other times are of 2026-11-08
and have not been taken again since that commit, which makes the library's steps shorter.

## Method

- **The same algorithm on both sides.** The hand code uses fixed-size matrices, `noalias()`, the
  library's update form $P - K P_{xy}^T$, a Cholesky factorization of the innovation covariance,
  and the library's order of operations. The difference measured is the library's overhead, not a
  difference of algorithm.
- **Agreement before timing.** The two sides must agree in state and covariance to 1e-9 in
  `double` (1e-4 relative in `float`) over hundreds of steps, both must pass a NEES consistency
  test over 50 runs, and neither may allocate during a step. These are tests that run in every CI
  build: `agreement.cpp` and `allocation.cpp` under each problem in
  [benchmarks/](../benchmarks/).
- **A fixed step.** Both sides compute what depends only on the step, such as the process noise,
  once at construction.
- **Instructions.** Callgrind counts the instructions of the filter steps alone
  (`--toggle-collect='*runSteps*'`), in a Release build with Clang 21 and `-march=x86-64-v3` on
  Linux. The counts are deterministic.
- **Time.** The median time per step over 2000 batches, with the two sides' batches interleaved
  and the process pinned to one performance core. Each ratio is of the medians; the p99 ratios
  follow them closely.

## Running them yourself

```sh
cmake --workflow --preset clang-bench   # or gcc-bench, msvc-bench
```

This builds the three runners in Release and runs them; each prints its table. To count
instructions, on Linux:

```sh
valgrind --tool=callgrind --toggle-collect='*runSteps*' \
    build/clang-bench/benchmarks/emrys_filter_ranging_benchmark --count
callgrind_annotate --inclusive=yes callgrind.out.<pid>
```

The Linux numbers below add `-DCMAKE_CXX_FLAGS=-march=x86-64-v3` at configuration.

## Ranging EKF

A receiver moving at nearly constant velocity measures its range to fixed anchors. The state is
position and velocity, 4 numbers in 2D and 6 in 3D, and each step is one prediction and one update
by every range.

| Instructions per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 1.45 | 1.12 | 0.92 | 1.21 | 0.93 | 1.00 |
| `double`, autodiff | 1.50 | 1.15 | 0.93 | 1.21 | 0.93 | 1.00 |
| `float`, analytic | 1.39 | 1.09 | 0.93 | 1.16 | 0.89 | 0.96 |
| `float`, autodiff | 1.43 | 1.10 | 0.93 | 1.16 | 0.89 | 0.97 |

| Time per step, Linux | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 1.13 | 1.10 | 0.95 | 1.10 | 0.93 | 1.07 |
| `double`, autodiff | 1.17 | 1.07 | 0.97 | 1.13 | 0.93 | 1.06 |
| `float`, analytic | 1.17 | 1.10 | 0.95 | 1.10 | 0.89 | 0.98 |
| `float`, autodiff | 1.42 | 1.15 | 0.96 | 1.11 | 0.90 | 1.00 |

At 2D with 4 anchors in `double` with analytic Jacobians, the hand code takes 906 instructions and
147 ns per step, and the library 1,315 instructions and 167 ns; at 3D with 24 anchors they take
1,756 ns and 1,880 ns. Both tables were measured on 2026-11-23.

Until the commit of that day named under the summary, every prediction built the process model's
Jacobian, expanded it to the whole state, and discarded it, and the library took 2,410 instructions
at 2D with 4 anchors, as it did at the public launch (2,399). The ratios this page gave at first,
1.36 at 2D with 4 anchors and 1.32 at 3D with 6 (909 against 1,233 instructions), were of an
earlier state of the library that no commit since the launch reproduces. The Windows times of the
summary are from 2026-11-08: a step took from 104 ns (2D, 4 anchors, `float`, analytic) to 2,150 ns
(3D, 24 anchors, `double`, autodiff).

## Attitude EKF

A unit quaternion attitude and a gyroscope bias, predicted by a 100 Hz gyroscope and updated by
gravity and a magnetic field. The hand code is the multiplicative EKF of Markley (2003) with a
6×6 error covariance and a covariance reset.

| Per step | Hand code, instructions | EmrysFilter, instructions | Ratio | Hand code, Windows | EmrysFilter, Windows |
|---|---|---|---|---|---|
| Analytic measurement Jacobians | 4,089 | 6,030 | 1.47 | 314 ns | 362 ns |
| Autodiff measurement Jacobians | 5,669 | 8,288 | 1.46 | 375 ns | 495 ns |

The instruction counts are of 2026-11-23. Before the commit of that day named under the summary the
library took 7,281 and 9,439; this page gave 4,830 and 7,186 at first (1.18 and 1.27), which no
commit since the launch reproduces, while the hand code's count is unchanged. The times are of
2026-11-08 and have not been taken again.

## Particle filter

The bootstrap particle filter on the ranging problem, with systematic resampling when the effective
sample size falls below half. The two sides, seeded alike, hold the same particles with the same
weights to 1e-9 over 300 steps. Instructions are counted at 1000 particles.

| Per particle and step, Windows | Hand code | EmrysFilter |
|---|---|---|
| 2D | 47–48 ns | 52–57 ns |
| 3D | 65–67 ns | 81–93 ns |

| Instructions per particle and step | Hand code | The noise's likelihood | Its own likelihood | Batched |
|---|---|---|---|---|
| 2D | 833 | 924 | 899 | 932 |
| 3D | 1,345 | 1,517 | 1,468 | 1,433 |

The instructions are of 2026-11-23. Before the commit of that day named under the summary, each
particle's step also built and discarded the process model's Jacobian, and the library took 2,800
to 2,880 instructions in 3D; this page gave 1,397 at first for its own likelihood in 3D, against
the hand code's 1,347. The times are of 2026-11-08 and have not been taken again.

## Compile time

The wall seconds to compile each benchmark as a program uses it, and in parentheses the ratio to
the hand-written program, each translation unit compiled alone on an idle machine.

| Unit | Clang Release | Clang Debug | GCC Release | MSVC Release |
|---|---:|---:|---:|---:|
| Ranging EKF, analytic | 3.4 (1.92) | 5.0 (1.90) | 6.1 (1.86) | 7.0 (1.77) |
| Ranging EKF, autodiff | 3.5 (1.84) | 5.2 (1.87) | 6.0 (1.71) | 9.2 (1.54) |
| Attitude EKF, analytic | 4.4 (1.80) | 6.9 (1.89) | 8.5 (1.84) | 12.3 (2.04) |
| Attitude EKF, autodiff | 5.1 (1.92) | 7.8 (1.81) | 8.9 (1.74) | 13.9 (1.85) |
| Particles, the noise's likelihood | 3.2 (2.22) | 4.3 (2.26) | 6.2 (2.15) | 6.1 (2.17) |
| Particles, its own likelihood | 2.8 (1.96) | 3.9 (2.04) | 5.4 (1.89) | 5.2 (1.86) |
| `EmrysFilter.h` alone, against `Eigen/Dense` | 1.3 (1.88) | 1.3 (1.75) | 2.1 (1.65) | 1.7 (1.91) |
| `examples/consumer/main.cpp` | 2.6 | 3.4 | 4.1 | 5.2 |

Peak memory is 1.5 to 2 times the hand-written program's, at most 2 GB (the attitude EKF, Clang
Debug).

## Against other libraries

The same ranging problem run by the Kalman filters of three other libraries, each written as its
users write it, with the analytic Jacobian, the only one they have. Each pair must agree to 1e-9 in
`double` over the run that is timed, so only the pairs that agree are timed; the runners print the
rest. Here a ratio is EmrysFilter's cost divided by the other library's: below 1, EmrysFilter costs
less. Measured on 2026-11-23 at the commit named under the summary, with Ubuntu 26.04 under WSL2 on
an Intel Core i9-13900F, Clang 21.1.8 with `-march=x86-64-v3`, each runner pinned to one CPU by
`taskset` at normal priority, the median of 2000 batches; instructions by Callgrind as above.

| EKF, `double` | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| mherb/kalman, instructions | 1.08 | 0.25 | 0.31 | 0.36 | 0.45 | 0.28 |
| mherb/kalman, time | 2.19 | 0.71 | 0.32 | 0.80 | 0.42 | 0.31 |
| TinyEKF, instructions | 0.83 | 1.56 | 0.81 | 0.63 | 1.19 | 0.73 |
| TinyEKF, time | 1.68 | 3.24 | 0.79 | 1.23 | 1.00 | 0.60 |
| OpenCV, instructions | 0.03 | 0.07 | 0.11 | 0.06 | 0.12 | 0.08 |
| OpenCV, time | 0.07 | 0.14 | 0.09 | 0.10 | 0.08 | 0.06 |

| Square-root EKF, against mherb/kalman | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, instructions | 0.24 | 0.39 | 0.75 | 0.36 | 0.63 | 0.99 |
| `double`, time | 0.68 | 0.78 | 1.04 | 0.74 | 0.92 | 1.15 |
| `float`, instructions | 0.15 | 0.25 | 0.48 | 0.25 | 0.41 | 0.76 |
| `float`, time | 0.55 | 0.73 | 0.98 | 0.68 | 0.84 | 1.07 |

At 2D with 4 anchors a step of EmrysFilter's EKF takes 166 ns and 1,315 instructions,
mherb/kalman's 76 ns and 1,218, and TinyEKF's 101 ns and 1,575. EmrysFilter is slower than TinyEKF
at three of the six sizes, 3.24 times at 2D with 8 anchors (535 ns against 165 ns), and than
mherb/kalman's EKF at the smallest; its square-root EKF is up to 15% slower than mherb/kalman's at
the largest sizes. It is faster than OpenCV at every size, by 7 to 16 times.

At the small sizes the time does not follow the instructions: the hand-written Eigen filter of the
ranging table takes 906 instructions and 147 ns at 2D with 4 anchors, fewer instructions than
either library and more time. EmrysFilter and the hand code factorize the innovation covariance by
Cholesky and solve for the gain, where mherb/kalman and TinyEKF invert it. Which part of that
takes the time has not been measured.

- **mherb/kalman** (`9f40c2f`, 2018-08-22): its EKF and square-root EKF are timed. Its `float` EKF
  strays up to 4e-2 relative in the mean and 9e-2 in the covariance from the `double` estimate,
  where EmrysFilter's stays within 3e-5, and its two UKFs update with the sigma points of the
  prediction, which leave the process noise out, so they differ from EmrysFilter's by 5e-4 to 3e-3
  relative in the mean and 2e-3 to 4e-3 in the covariance; neither is timed.
- **TinyEKF** (`ba0d2a9`, 2026-04-09): timed in `double`. In `float` it strays up to 5e-3 relative
  in the mean and 3e-2 in the covariance, while the prior is wide, so `float` is not timed.
- **OpenCV** 4.14.0, its core, imgproc, and video modules built from source in Release with its
  default CPU flags: `cv::KalmanFilter` is linear and run as an EKF by the pseudo-measurement
  $z - h(x) + H x$. It never makes its covariance symmetric again, and the asymmetry rounding leaves
  grows until the filter diverges: the two agree to 1e-9 until somewhere between steps 400 and 504,
  and by step 2000 OpenCV's 3D estimates are lost, their covariances 1e7 to 3e11 times the size of
  EmrysFilter's away from it. So it is timed over runs of 300 steps that start over, untimed, at
  their end. With 8 anchors or more it allocates once per step in `double`.

## Not measured yet

- The particle filter's scaling with threads under `std::execution::par`.
- Code size, stack, and cycles per step on Cortex-M4F and Cortex-M7.
- The price of each option (update form, Gaussian form, sigma points) relative to the plain EKF.

## Machines

The commits of the measurements of 2026-11-08 and 2026-11-01 were not recorded; from the next
release on, each table names its commit.

- **Instructions:** 2026-11-23, Clang 21.1.8 with `-march=x86-64-v3`, Callgrind under Ubuntu 26.04
  in WSL2.
- **Time, the ranging EKF on Linux and the other libraries:** 2026-11-23, Intel Core i9-13900F,
  Ubuntu 26.04 under WSL2 with Clang 21.1.8 and `-march=x86-64-v3`, pinned to one CPU at normal
  priority.
- **Time, the rest:** 2026-11-08, Intel Core i9-13900F. Windows 11 with Clang 21.1.0 and no
  `-march`, at high priority; Ubuntu 26.04 under WSL2 with Clang 21.1.8 and `-march=x86-64-v3`, at
  normal priority.
- **Compile time:** 2026-11-01, an Intel Raptor Lake processor under Windows 11, with Clang 21.1.0,
  GCC 14.2.0 (w64devkit), and MSVC 19.50.
