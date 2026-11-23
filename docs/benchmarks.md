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
  [benchmarks/](../benchmarks/). A pair against another library that cannot agree is timed all the
  same and marked, once a test shows the difference to be that library's
  ([Against other libraries](#against-other-libraries)).
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
users write it, with the analytic Jacobian, the only one they have. Here a ratio is EmrysFilter's
cost divided by the other library's: below 1, EmrysFilter costs less.

Each pair is checked before it is timed. Most agree, to 1e-9 in `double` and 1e-4 in `float` over
the run that is timed. A pair marked \* does not agree: the other library's estimate differs from
EmrysFilter's, and a test shows the difference to be that library's, either because EmrysFilter's
estimate stays with the `double` estimate, on which all five implementations agree to 1e-9, or
because the other library's filter fails a check that involves no EmrysFilter code. A marked
ratio therefore compares two steps that do not compute the same estimate;
[the table after the ratios](#the-marked-pairs) gives how far apart they are, and the notes on
each library give the cause. TinyEKF and OpenCV run in `float` unless told otherwise, so their
marked rows are the ones their users get by default.

Measured on 2026-11-23, with the library as of the commit named under the summary, with Ubuntu 26.04
under WSL2 on an Intel Core i9-13900F, Clang 21.1.8 with `-march=x86-64-v3`, each runner pinned to
one CPU by `taskset` at normal priority, the median of 2000 batches; instructions by Callgrind as
above.

| EKF | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| mherb/kalman, `double`, instructions | 1.08 | 0.25 | 0.31 | 0.36 | 0.45 | 0.28 |
| mherb/kalman, `double`, time | 2.20 | 0.73 | 0.33 | 0.81 | 0.42 | 0.33 |
| mherb/kalman, `float` \*, instructions | 1.08 | 0.25 | 0.36 | 0.34 | 0.44 | 0.36 |
| mherb/kalman, `float` \*, time | 1.58 | 0.62 | 0.35 | 1.14 | 0.41 | 0.37 |
| TinyEKF, `double`, instructions | 0.83 | 1.56 | 0.81 | 0.63 | 1.19 | 0.73 |
| TinyEKF, `double`, time | 1.66 | 3.22 | 0.82 | 1.22 | 1.08 | 0.66 |
| TinyEKF, `float` \*, instructions | 0.75 | 1.51 | 0.88 | 0.54 | 1.22 | 0.89 |
| TinyEKF, `float` \*, time | 1.25 | 2.85 | 0.85 | 1.81 | 1.01 | 0.69 |
| OpenCV, `double`, instructions | 0.03 | 0.07 | 0.11 | 0.06 | 0.12 | 0.08 |
| OpenCV, `double`, time | 0.07 | 0.14 | 0.09 | 0.10 | 0.08 | 0.06 |
| OpenCV, `float` \*, instructions | 0.03 | 0.06 | 0.10 | 0.05 | 0.11 | 0.08 |
| OpenCV, `float` \*, time | 0.04 | 0.12 | 0.07 | 0.14 | 0.07 | 0.06 |

| Against mherb/kalman | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Square-root EKF, `double`, instructions | 0.24 | 0.39 | 0.75 | 0.36 | 0.63 | 0.99 |
| Square-root EKF, `double`, time | 0.71 | 0.84 | 1.04 | 0.75 | 0.89 | 1.16 |
| Square-root EKF, `float`, instructions | 0.15 | 0.25 | 0.48 | 0.25 | 0.41 | 0.76 |
| Square-root EKF, `float`, time | 0.54 | 0.74 | 0.98 | 0.68 | 0.87 | 1.08 |
| UKF, `double` \*, instructions | 1.62 | 0.54 | 0.55 | 0.71 | 0.75 | 0.57 |
| UKF, `double` \*, time | 1.57 | 0.92 | 0.62 | 1.02 | 0.85 | 0.68 |
| UKF, `float` \*, instructions | 1.69 | 0.49 | 0.54 | 0.62 | 0.71 | 0.60 |
| UKF, `float` \*, time | 1.69 | 0.83 | 0.62 | 1.18 | 0.72 | 0.65 |
| Square-root UKF, `double` \*, instructions | 0.97 | 0.97 | 1.20 | 1.04 | 1.15 | 1.51 |
| Square-root UKF, `double` \*, time | 0.89 | 0.89 | 1.03 | 1.17 | 1.09 | 1.37 |
| Square-root UKF, `float` \*, instructions | 0.86 | 0.88 | 0.98 | 0.90 | 1.00 | 1.13 |
| Square-root UKF, `float` \*, time | 0.80 | 0.85 | 0.97 | 0.93 | 0.87 | 0.97 |

At 2D with 4 anchors in `double`, a step of EmrysFilter's EKF takes 167 ns and 1,315 instructions,
mherb/kalman's 76 ns and 1,218, and TinyEKF's 100 ns and 1,575; in `float`, 106 to 111 ns and 1,149,
67 ns and 1,063, and 89 ns and 1,530.

- **Against TinyEKF** the EKF is slower at 2D with 4 and 8 anchors and at 3D with 6, in both
  scalars, by up to 3.22 times (2D with 8 anchors in `double`, 538 ns against 167 ns). It is level
  at 3D with 12 (1.08 and 1.01), and faster with 16 and 24.
- **Against mherb/kalman** the EKF is slower at 2D with 4 anchors in both scalars and at 3D with 6
  in `float`, and faster elsewhere, by up to 3 times. The square-root EKF is slower at 3D with 24
  anchors, by 16% in `double` and 8% in `float`, and at 2D with 16 in `double`, by 4%, and faster
  elsewhere. The UKF is slower at 2D with 4 anchors and 3D with 6 and faster from 8 anchors. The
  square-root UKF is slower in `double` at 2D with 16 anchors and at every 3D size, by up to 37%,
  and faster in `float` at every size.
- **Against OpenCV** the EKF is faster at every size, by 7 to 25 times.

At the small sizes the time does not follow the instructions: the hand-written Eigen filter of the
ranging table takes 906 instructions and 147 ns at 2D with 4 anchors, fewer instructions than
either library and more time. EmrysFilter and the hand code factorize the innovation covariance by
Cholesky and solve for the gain, where mherb/kalman and TinyEKF invert it. Which part of that
takes the time has not been measured.

The instruction ratios against TinyEKF are worst with 8 and 12 anchors, where the innovation
covariance is 8×8 or 12×12: too large for Eigen's unrolled fixed-size code, and small enough that
the overhead of its general kernels is most of their cost. At 2D with 8 anchors, 2,357 of
EmrysFilter's 4,397 instructions factorize that matrix: 940 in Eigen's Cholesky loop, 730 in the
general matrix-vector kernel it calls once per column, and 687 in `LLT::compute()` around them,
507 of which compute a norm that only `rcond()` reads and the filter never asks for. TinyEKF
factorizes and inverts the same matrix in about 1,000. At 3D with 12 anchors, 9,148 of
EmrysFilter's 17,675 are the gain's two triangular solves, which the library solves a column at a
time only up to size 8 and hands to Eigen's blocked solver beyond it, and 4,488 the factorization.
With 16 and 24 anchors the same kernels spread their overhead over more work while TinyEKF's
scalar loops grow with the cube of the anchor count. With 4 and 6, TinyEKF is slower than its own
trend, since Clang leaves its matrix product out of line as a loop with run-time bounds (896 of
its 1,575 instructions at 2D with 4 anchors, 5,648 of 6,484 at 3D with 6). The hand-written Eigen
filter shares the factorization and the solves, and their cost. None of this is changed yet.

At 3D with 6 anchors a step takes longer in `float` than in `double`, for EmrysFilter (573 ns
against 412 ns) and for the hand-written Eigen filter alike, though it takes fewer instructions,
which is why the `float` ratios of that column are the worse ones. The cause has not been
measured.

### The marked pairs

The largest distance over the timed run, relative to the size of the mean and of the covariance of
the estimate it is measured from, as mean / covariance, over the six sizes.

| Marked pair | Measured from | EmrysFilter | The other library |
|---|---|---|---|
| EKF, `float`, mherb/kalman | the `double` estimate | up to 8e-6 / 3e-5 | 2e-4 to 4e-2 / 6e-4 to 9e-2 |
| EKF, `float`, TinyEKF | the `double` estimate | up to 8e-6 / 3e-5 | 2e-4 to 5e-3 / 1e-3 to 3e-2 |
| EKF, `float`, OpenCV | the `double` estimate | up to 8e-6 / 3e-5 | 2e-4 to 2e-3 / 4e-3 to 2e-2 |
| UKF and square-root UKF, `double` | EmrysFilter's filter | | 5e-4 to 3e-3 / 2e-3 to 4e-3 |
| UKF, `float` | each library's own `double` UKF | up to 3e-5 / 9e-5 | 1e-4 to 1e-2 / 1e-3 to 6e-2 |
| Square-root UKF, `float` | each library's own `double` filter | up to 4e-6 / 2e-5 | up to 2e-5 / 1e-4 |

- **mherb/kalman** (`9f40c2f`, 2018-08-22).
  - *Its EKF in `float`* forms the gain with an explicit inverse of the innovation covariance
    (`S.inverse()`), which loses accuracy while the prior is wide. A copy of its header with that
    one line changed, the gain solved by the Cholesky factor of $S$, stays within 6e-5 of the
    `double` estimate at every size; mirroring its covariance instead changes nothing.
  - *Its two UKFs*, in either scalar, compute the innovation covariance and the cross covariance
    from the sigma points their prediction moved, which were drawn before the process noise was
    added, so both leave the process noise out while the covariance includes it. No EmrysFilter
    code is needed to show it: on a linear problem, where a UKF gives the Kalman filter's
    estimate, mherb/kalman's UKF agrees with its own EKF to 1e-12 without process noise and
    differs from it by 2e-3 in the covariance with this problem's. Without process noise the two
    libraries' UKFs agree on the ranging problem to 1e-5 in 2D and 1e-3 in 3D.
  - *Its UKF in `float`* also strays from its own `double` UKF, by up to 1e-2 in the mean, and
    its square-root UKF does not. It forms that gain with an explicit inverse too, which is the
    likely cause and has not been checked.
  - Its square-root UKF allocates on the heap in its prediction, so its times include its
    allocator.
- **TinyEKF** (`ba0d2a9`, 2026-04-09). `float` is the scalar it runs in unless `_float_t` is
  defined. Its estimate strays in `float` over the first steps, while the prior is wide, by three
  of its choices together: the gain through an explicit inverse of $S$, the covariance updated as
  $(I - G H) P$, and no triangle mirrored. A copy of its header with all three changed stays
  within 4e-5 of the `double` estimate; the solve alone, the solve with the update form, and the
  mirroring alone do not.
- **OpenCV** 4.14.0, its core, imgproc, and video modules built from source in Release with its
  default CPU flags: `cv::KalmanFilter` is linear and run as an EKF by the pseudo-measurement
  $z - h(x) + H x$.
  - *In `double`* it never makes its covariance symmetric again, and the asymmetry rounding leaves
    grows until the filter diverges: the two agree to 1e-9 until somewhere between steps 400 and
    504, and by step 2000 OpenCV's 3D estimates are lost, their covariances 1e7 to 3e11 times the
    size of EmrysFilter's away from it. So it is timed over runs of 300 steps that start over,
    untimed, at their end.
  - *In `float`*, the type its filter has unless it is constructed with `CV_64F`, it strays within
    those 300 steps. It solves for the gain by a singular value decomposition of an innovation
    covariance that the unmirrored covariance has left asymmetric: a copy of `correct()`'s
    arithmetic that solves by Cholesky instead stays within 1e-5 of the `double` estimate, and
    OpenCV's own `correct()` within 4e-5 when `cv::completeSymm()` mirrors `errorCovPost` after
    each call.
  - With 8 anchors or more in `double`, and 12 or more in `float`, it allocates once per step.

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
