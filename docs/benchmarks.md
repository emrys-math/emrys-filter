# Benchmarks

Each benchmark solves one problem twice: once with EmrysFilter, and once in plain Eigen as a careful
user would write it by hand. Every number is the library's cost divided by the hand code's, so 1.00
means the same cost, above 1 the library costs more, and below 1 it costs less.
[design/benchmarks.md](design/benchmarks.md) explains which claim each benchmark backs, why the
reference is hand-written code, and what is still to be measured.

## Summary

| Problem | Instructions per step | Time per step, Windows | Time per step, Linux |
|---|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.30–1.44 | 1.15–1.68 | 1.22–1.53 |
| Ranging EKF, 2D, 8 anchors | 1.06–1.11 | 1.02–1.44 | 1.06–1.15 |
| Ranging EKF, 3D, 6 anchors | 1.16–1.33 | 1.11–1.21 | 1.14–1.18 |
| Ranging EKF, 12 or more anchors | 0.89–1.01 | 0.91–0.97 | 0.91–1.09 |
| Attitude EKF, analytic Jacobians | 1.18 | 1.15 | 1.21 |
| Attitude EKF, autodiff | 1.27 | 1.32 | 1.29 |
| Particle filter, 2D | 1.03–1.06 | 1.10–1.18 | 1.00–1.04 |
| Particle filter, 3D | 1.04–1.07 | 1.25–1.40 | 1.02–1.12 |

A range covers every case of its row: `double` and `float`, analytic and automatic Jacobians, and
for the particle filter both likelihoods and every particle count. Neither side allocates on the
heap during a step, in any case.

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
| `double`, analytic | 1.36 | 1.09 | 0.91 | 1.32 | 0.94 | 1.01 |
| `double`, autodiff | 1.44 | 1.11 | 0.92 | 1.33 | 0.95 | 1.01 |
| `float`, analytic | 1.30 | 1.06 | 0.93 | 1.16 | 0.89 | 0.96 |
| `float`, autodiff | 1.35 | 1.06 | 0.93 | 1.18 | 0.89 | 0.96 |

At 2D with 4 anchors in `double` with analytic Jacobians, the hand code takes 909 instructions per
step and the library 1,233. On Windows a step takes from 104 ns (2D, 4 anchors, `float`, analytic)
to 2,150 ns (3D, 24 anchors, `double`, autodiff).

## Attitude EKF

A unit quaternion attitude and a gyroscope bias, predicted by a 100 Hz gyroscope and updated by
gravity and a magnetic field. The hand code is the multiplicative EKF of Markley (2003) with a
6×6 error covariance and a covariance reset.

| Per step | Hand code, instructions | EmrysFilter, instructions | Ratio | Hand code, Windows | EmrysFilter, Windows |
|---|---|---|---|---|---|
| Analytic measurement Jacobians | 4,092 | 4,830 | 1.18 | 314 ns | 362 ns |
| Autodiff measurement Jacobians | 5,672 | 7,186 | 1.27 | 375 ns | 495 ns |

## Particle filter

The bootstrap particle filter on the ranging problem, with systematic resampling when the effective
sample size falls below half. The two sides, seeded alike, hold the same particles with the same
weights to 1e-9 over 300 steps. Instructions are counted at 1000 particles.

| Per particle and step, Windows | Hand code | EmrysFilter |
|---|---|---|
| 2D | 47–48 ns | 52–57 ns |
| 3D | 65–67 ns | 81–93 ns |

In 3D, with the measurement's own likelihood, the library takes 1,397 instructions per particle and
step and the hand code 1,347.

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

## Not measured yet

- Against mherb/kalman: its EKF, UKF, square-root EKF, and square-root UKF on the ranging problem.
  The runner is built (`emrys_filter_mherb_benchmark`). It times the EKF in `double` and the
  square-root EKF in both scalars. In `float`, mherb/kalman's EKF loses accuracy, and its two UKFs
  compute a different update from the standard one, so those are compared for accuracy but not
  timed.
- The particle filter's scaling with threads under `std::execution::par`.
- Code size, stack, and cycles per step on Cortex-M4F and Cortex-M7.
- The price of each option (update form, Gaussian form, sigma points) relative to the plain EKF.

## Machines

The commits of these measurements were not recorded; from the next release on, each table names
its commit.

- **Instructions:** November 2026, Clang 21 with `-march=x86-64-v3`, Callgrind under Ubuntu 26.04
  in WSL2.
- **Time:** 2026-11-08, Intel Core i9-13900F. Windows 11 with Clang 21.1.0 and no `-march`, at
  high priority; Ubuntu 26.04 under WSL2 with Clang 21.1.8 and `-march=x86-64-v3`, at normal
  priority.
- **Compile time:** 2026-11-01, an Intel Raptor Lake processor under Windows 11, with Clang 21.1.0,
  GCC 14.2.0 (w64devkit), and MSVC 19.50.
