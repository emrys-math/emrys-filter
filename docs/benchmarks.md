# Benchmarks

Each benchmark solves one problem twice: with EmrysFilter, and in plain Eigen as a careful user
would write it by hand. A ratio is the library's cost divided by the hand code's: above 1 the
library costs more, below 1 it costs less.

The EKFs have two hand-written references, which differ only in how they factorize the innovation
covariance and solve for the gain:

- *Hand code* does both as the library does, with a Cholesky factorization without square roots
  written out a column at a time. Its ratio is the library's overhead.
- *Hand code with Eigen's LLT* writes `Eigen::LLT<Matrix>{S}.solve(Pxy.transpose()).transpose()`,
  as a user of Eigen would. Its ratio is what the library gives that user.

[design/benchmarks.md](design/benchmarks.md) says which claim each benchmark backs and why the
reference is hand-written code.

## Summary

| Problem | Instructions per step | Time per step, Windows | Time per step, Linux | Time per step, Linux, against hand code with Eigen's LLT |
|---|---|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.61–1.86 | 1.35–1.67 | 1.13–1.64 | 0.42–0.64 |
| Ranging EKF, 2D, 8 anchors | 1.32–1.44 | 1.15–1.25 | 1.07–1.46 | 0.35–0.49 |
| Ranging EKF, 3D, 6 anchors | 1.31–1.38 | 1.44–1.51 | 1.35–1.45 | 0.64–0.80 |
| Ranging EKF, 12 or more anchors | 1.10–1.23 | 1.09–1.34 | 1.09–1.29 | 0.37–0.64 |
| Attitude EKF, analytic Jacobians | 1.55 | 1.56 | 1.46 | 1.08 |
| Attitude EKF, autodiff | 1.52 | 1.73 | 1.49 | 1.15 |
| Particle filter, 2D | 1.06–1.10 | 1.30–1.47 | 1.02–1.08 | |
| Particle filter, 3D | 1.04–1.10 | 1.28–1.55 | 1.07–1.15 | |

The library costs more than the hand code in every case. Against hand code with Eigen's LLT, the
ranging EKF takes 0.35 to 0.80 of the time. The attitude EKF takes more time than that hand code,
1.08 and 1.15 times, because its two 3×3 factorizations per step are a small part of the step.

A range covers every case of its row: `double` and `float`, analytic and automatic Jacobians, and
for the particle filter its likelihoods and every particle count. Neither side allocates on the
heap during a step. The Windows column is Clang's; MSVC's is under [With MSVC](#with-msvc).

## Method

- **The same algorithm on both sides.** The hand code uses fixed-size matrices, Eigen's lazy
  products, the library's update form $P - K P_{xy}^T$, the library's factorization and
  substitution, written out, and the library's order of operations, so what is measured is the
  library's overhead. The hand code with Eigen's LLT differs from it in the factorization, the
  solve, and its products, which are `noalias()` ones.
- **Agreement before timing.** The two sides must agree in state and covariance to 1e-9 in
  `double` (1e-4 relative in `float`) over hundreds of steps, both must pass a NEES consistency
  test over 50 runs, and neither may allocate during a step. These are tests in every CI build:
  `agreement.cpp` and `allocation.cpp` under each problem in [benchmarks/](../benchmarks/).
- **A fixed step.** Both sides compute what depends only on the step, such as the process noise,
  once at construction.
- **Instructions.** Callgrind counts the instructions of the filter steps alone
  (`--toggle-collect='*runSteps*'`), in a Release build with Clang 21 and `-march=x86-64-v3` on
  Linux. The counts are deterministic.
- **Time.** The median time per step over 2000 batches, the two sides' batches interleaved, the
  process pinned to one performance core. A ratio is of the medians, and the median of three runs
  of the runner (one for the particle filter).

## Running them yourself

```sh
cmake --workflow --preset clang-bench   # or gcc-bench, msvc-bench
```

This builds the runners in Release and runs them; each prints its table. The Linux numbers below
add `-DCMAKE_CXX_FLAGS=-march=x86-64-v3` at configuration. To count instructions, on Linux:

```sh
valgrind --tool=callgrind --toggle-collect='*runSteps*' \
    build/clang-bench/benchmarks/emrys_filter_ranging_benchmark --count
callgrind_annotate --inclusive=yes callgrind.out.<pid>
```

## Ranging EKF

A receiver moving at nearly constant velocity measures its range to fixed anchors. The state is
position and velocity, 4 numbers in 2D and 6 in 3D, and each step is one prediction and one update
by every range.

| Instructions per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 1.61 | 1.32 | 1.14 | 1.38 | 1.16 | 1.20 |
| `double`, autodiff | 1.76 | 1.34 | 1.16 | 1.36 | 1.18 | 1.19 |
| `float`, analytic | 1.86 | 1.44 | 1.11 | 1.35 | 1.23 | 1.11 |
| `float`, autodiff | 1.76 | 1.42 | 1.12 | 1.31 | 1.23 | 1.10 |

| Time per step, Linux | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 1.40 | 1.19 | 1.09 | 1.38 | 1.19 | 1.22 |
| `double`, autodiff | 1.38 | 1.11 | 1.17 | 1.45 | 1.27 | 1.24 |
| `float`, analytic | 1.13 | 1.07 | 1.09 | 1.43 | 1.21 | 1.12 |
| `float`, autodiff | 1.64 | 1.46 | 1.17 | 1.35 | 1.29 | 1.15 |

Against hand code with Eigen's LLT:

| Instructions per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 0.25 | 0.28 | 0.38 | 0.45 | 0.36 | 0.52 |
| `double`, autodiff | 0.30 | 0.32 | 0.39 | 0.50 | 0.39 | 0.54 |
| `float`, analytic | 0.26 | 0.22 | 0.32 | 0.35 | 0.26 | 0.29 |
| `float`, autodiff | 0.32 | 0.26 | 0.31 | 0.39 | 0.27 | 0.30 |

| Time per step, Linux | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 0.58 | 0.49 | 0.45 | 0.72 | 0.47 | 0.63 |
| `double`, autodiff | 0.58 | 0.49 | 0.47 | 0.80 | 0.49 | 0.64 |
| `float`, analytic | 0.42 | 0.35 | 0.39 | 0.64 | 0.37 | 0.37 |
| `float`, autodiff | 0.64 | 0.48 | 0.40 | 0.65 | 0.41 | 0.39 |

The three runs behind each time ratio differ by at most 0.14 against hand code and 0.05 against
hand code with Eigen's LLT.

| Time per step, Linux, `double`, analytic | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Hand code | 52 ns | 116 ns | 359 ns | 121 ns | 305 ns | 875 ns |
| EmrysFilter | 73 ns | 139 ns | 373 ns | 168 ns | 363 ns | 1,070 ns |
| Hand code with Eigen's LLT | 126 ns | 299 ns | 857 ns | 231 ns | 728 ns | 1,689 ns |

At 2D with 4 anchors in `double` with analytic Jacobians, the hand code takes 394 instructions, the
library 635, and the hand code with Eigen's LLT 2,531. The library's fixed cost per step there is
about 21 ns and 241 instructions, so its ratios are highest at the smallest sizes. On Windows a
step of the library takes from 80 ns (2D, 4 anchors, analytic) to 1,394 ns (3D, 24 anchors,
`double`, autodiff), and the hand code's from 54 ns to 1,250 ns.

At 3D with 6 anchors a step on Linux takes about as long in `float` as in `double` (the library
178 against 168 ns, the hand code 124 against 121 ns), though it takes fewer instructions.

## Attitude EKF

A unit quaternion attitude and a gyroscope bias, predicted by a 100 Hz gyroscope and updated by
gravity and a magnetic field. The hand code is the multiplicative EKF of Markley (2003) with a
6×6 error covariance and a covariance reset.

| Instructions per step | Hand code | Hand code with Eigen's LLT | EmrysFilter | Ratio | Ratio, against Eigen's LLT |
|---|---|---|---|---|---|
| Analytic measurement Jacobians | 3,425 | 7,321 | 5,319 | 1.55 | 0.73 |
| Autodiff measurement Jacobians | 4,985 | 8,889 | 7,585 | 1.52 | 0.85 |

| Time per step, Linux | Hand code | Hand code with Eigen's LLT | EmrysFilter | Ratio | Ratio, against Eigen's LLT |
|---|---|---|---|---|---|
| Analytic measurement Jacobians | 263 ns | 374 ns | 385–405 ns | 1.46 | 1.08 |
| Autodiff measurement Jacobians | 336 ns | 436 ns | 500 ns | 1.49 | 1.15 |

EmrysFilter's time is that of its run against each reference. On Windows the hand code takes 227
and 284 ns and the library 355 and 491 ns.

## Particle filter

The bootstrap particle filter on the ranging problem, with systematic resampling when the effective
sample size falls below half. The two sides, seeded alike, hold the same particles with the same
weights to 1e-9 over 200 steps at every particle count up to a million; with the measurement's own
likelihood they are bitwise equal. It factorizes no innovation covariance, so it has one
reference. Instructions are counted at 1000 particles.

| Per particle and step | Hand code | EmrysFilter |
|---|---|---|
| 2D, Windows | 50–57 ns | 66–80 ns |
| 3D, Windows | 68–71 ns | 89–109 ns |
| 2D, Linux | 141–164 ns | 145–168 ns |
| 3D, Linux | 210–217 ns | 225–248 ns |

On Linux most of each side's step is libstdc++'s `std::normal_distribution`, whose uniform draws
take a `long double` logarithm at every call, so the Linux ratios say less about the library than
the Windows ones.

| Instructions per particle and step | Hand code | The noise's likelihood | Its own likelihood | Batched |
|---|---|---|---|---|
| 2D | 833 | 900 | 885 | 913 |
| 3D | 1,345 | 1,482 | 1,451 | 1,397 |

## With MSVC

Every pair also builds with MSVC 19.50 (the `msvc-bench` preset, `/O2`) and agrees as it does with
Clang, but for `float` at 2D with 8 anchors. There MSVC at `/O2` miscompiles the benchmark's own
measurement function, on both sides, and the estimate is lost within a few steps; the same code at
`/Od` tracks the `double` estimate to 3e-6. So the pairs against mherb/kalman and
FrancoisCarouge/Kalman are not timed there. Times on Windows, at high priority on one core:

| Time per step, MSVC | Against hand code | Against hand code with Eigen's LLT |
|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.34–1.90 | 0.94–1.04 |
| Ranging EKF, 2D, 8 anchors | 1.26–1.62 | 0.54–0.67 |
| Ranging EKF, 3D, 6 anchors, `double` | 1.43–1.56 | 0.79–0.85 |
| Ranging EKF, 12 or more anchors, `double`, and 2D with 16 in `float` | 1.15–1.25 | 0.45–0.81 |
| Ranging EKF, 3D, 6, 12, and 24 anchors, `float` | 0.21–0.66 | 0.48–0.88 |
| Attitude EKF, analytic and autodiff | 1.61, 1.25 | 1.20, 1.15 |
| Particle filter, 2D and 3D | 1.34–1.57 | |

The ratios below 1, in `float` at 3D, are MSVC's code for the hand code, not the library's. With
6, 12, and 24 anchors the hand code's step takes 443, 1,461, and 5,534 ns, against 173, 448, and
1,631 ns in `double`, while the library's takes 259, 419, and 1,178 ns. At 12 and 24 anchors MSVC
compiles the hand code's lazy products in `float`, and at 24 its written-out factorization and
substitution, to code several times slower than the library's for the same operations.

MSVC's steps are longer than Clang's on the same machine: the library's at 2D with 4 anchors takes
132 ns against 83, in `double` with analytic Jacobians. Built by MSVC at `68d4cfd`, before
`19019da`, `c679b0a`, and `ab1b0b9`, which the Clang numbers above include, the EKF took 1.24
times mherb/kalman's time at 2D with 4 anchors in `double` and 1.60 in `float`, and the UKF 1.38
and 1.27 times; at every other size, and against TinyEKF (0.26 to 0.90) and
FrancoisCarouge/Kalman (0.14 to 0.39), it took less.

## Footprint on Cortex-M

The ranging EKF at 3D with 6 anchors in `float` with the analytic Jacobian, built for the
Cortex-M4F and the Cortex-M7 in MinSizeRel with no heap, exceptions, or RTTI. A filter's footprint
is its program's code less that of the simulator that drives it, which every program shares. Each
program runs under QEMU and ends within 2 m of the true position
([benchmarks/cortex_m/](../benchmarks/cortex_m/), the `footprint` target).

| `.text` less the simulator's | Cortex-M4F | Cortex-M7 |
|---|---:|---:|
| EmrysFilter | 19,736 B | 19,576 B |
| Hand code | 6,904 B | 6,696 B |
| TinyEKF | 2,256 B | 2,040 B |

The library's EKF takes 2.9 times the hand-written Eigen EKF's code and 8.7 to 9.6 times TinyEKF's,
whose every product is a loop over arrays. None adds `.data`; the library adds 16 bytes of `.bss`.
The largest stack frame is the one that holds the filter: 7,664 bytes for the library on the M4F,
5,152 for the hand code, and 6,712 for TinyEKF, against the simulator's 4,376. Cycles per step are
not measured yet.

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

The same ranging problem run by the Kalman filters of four other libraries, each written as its
users write it, with the analytic Jacobian, the only one they have. Here a ratio is EmrysFilter's
cost divided by the other library's: below 1, EmrysFilter costs less.

Each pair is checked before it is timed, and most agree to 1e-9 in `double` and 1e-4 in `float`. A
pair marked \* does not agree, and a test shows the difference to be the other library's: either
EmrysFilter's estimate stays with the `double` estimate, on which every implementation agrees, or
the other library's filter fails a check that involves no EmrysFilter code. A marked ratio
compares steps that do not compute the same estimate; [the marked pairs](#the-marked-pairs) gives
how far apart they are and why. TinyEKF and OpenCV run in `float` unless told otherwise, so their
marked rows are what their users get by default.

| EKF | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| mherb/kalman, `double`, instructions | 0.52 | 0.11 | 0.13 | 0.24 | 0.18 | 0.15 |
| mherb/kalman, `double`, time | 1.02 | 0.18 | 0.15 | 0.35 | 0.21 | 0.18 |
| mherb/kalman, `float` \*, instructions | 0.62 | 0.09 | 0.12 | 0.20 | 0.13 | 0.11 |
| mherb/kalman, `float` \*, time | 0.77 | 0.13 | 0.14 | 0.37 | 0.17 | 0.14 |
| TinyEKF, `double`, instructions | 0.40 | 0.66 | 0.33 | 0.43 | 0.46 | 0.38 |
| TinyEKF, `double`, time | 0.78 | 0.85 | 0.38 | 0.54 | 0.50 | 0.36 |
| TinyEKF, `float` \*, instructions | 0.43 | 0.53 | 0.30 | 0.32 | 0.35 | 0.26 |
| TinyEKF, `float` \*, time | 0.60 | 0.58 | 0.35 | 0.58 | 0.46 | 0.30 |
| FrancoisCarouge/Kalman, `double`, instructions | 0.11 | 0.11 | 0.15 | 0.19 | 0.16 | 0.19 |
| FrancoisCarouge/Kalman, `double`, time | 0.27 | 0.17 | 0.15 | 0.27 | 0.17 | 0.19 |
| FrancoisCarouge/Kalman, `float`, instructions | 0.12 | 0.08 | 0.13 | 0.15 | 0.11 | 0.13 |
| FrancoisCarouge/Kalman, `float`, time | 0.18 | 0.11 | 0.13 | 0.25 | 0.14 | 0.13 |
| OpenCV, `double`, instructions | 0.02 | 0.03 | 0.04 | 0.04 | 0.05 | 0.04 |
| OpenCV, `double`, time | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| OpenCV, `float` \*, instructions | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 |
| OpenCV, `float` \*, time | 0.02 | 0.02 | 0.03 | 0.04 | 0.03 | 0.02 |

| Against mherb/kalman | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Square-root EKF, `double`, instructions | 0.11 | 0.14 | 0.27 | 0.18 | 0.26 | 0.53 |
| Square-root EKF, `double`, time | 0.34 | 0.36 | 0.43 | 0.40 | 0.48 | 0.57 |
| Square-root EKF, `float`, instructions | 0.07 | 0.08 | 0.15 | 0.12 | 0.17 | 0.31 |
| Square-root EKF, `float`, time | 0.30 | 0.27 | 0.35 | 0.36 | 0.52 | 0.57 |
| UKF, `double` \*, instructions | 0.86 | 0.32 | 0.31 | 0.50 | 0.44 | 0.40 |
| UKF, `double` \*, time | 0.71 | 0.40 | 0.39 | 0.68 | 0.55 | 0.49 |
| UKF, `float` \*, instructions | 0.91 | 0.26 | 0.27 | 0.41 | 0.37 | 0.30 |
| UKF, `float` \*, time | 0.88 | 0.34 | 0.35 | 0.75 | 0.42 | 0.37 |
| Square-root UKF, `double` \*, instructions | 0.48 | 0.45 | 0.54 | 0.61 | 0.59 | 0.94 |
| Square-root UKF, `double` \*, time | 0.48 | 0.46 | 0.54 | 0.74 | 0.69 | 0.84 |
| Square-root UKF, `float` \*, instructions | 0.41 | 0.37 | 0.39 | 0.51 | 0.51 | 0.57 |
| Square-root UKF, `float` \*, time | 0.48 | 0.42 | 0.44 | 0.63 | 0.58 | 0.59 |

| EKF, time per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| EmrysFilter, `double` | 73 ns | 141 ns | 393 ns | 170 ns | 350 ns | 1,082 ns |
| mherb/kalman, `double` | 72 ns | 788 ns | 2,598 ns | 494 ns | 1,685 ns | 5,938 ns |
| TinyEKF, `double` | 94 ns | 163 ns | 971 ns | 312 ns | 684 ns | 2,937 ns |
| FrancoisCarouge/Kalman, `double` | 279 ns | 841 ns | 2,517 ns | 623 ns | 2,002 ns | 5,730 ns |
| OpenCV, `double` | 2,397 ns | 3,789 ns | 9,117 ns | 3,748 ns | 8,900 ns | 30,428 ns |
| EmrysFilter, `float` | 51 ns | 98 ns | 332 ns | 188 ns | 287 ns | 770 ns |
| mherb/kalman, `float` \* | 67 ns | 768 ns | 2,345 ns | 506 ns | 1,699 ns | 5,483 ns |
| TinyEKF, `float` \* | 86 ns | 160 ns | 908 ns | 308 ns | 600 ns | 2,480 ns |
| FrancoisCarouge/Kalman, `float` | 282 ns | 855 ns | 2,402 ns | 711 ns | 2,063 ns | 5,566 ns |
| OpenCV, `float` \* | 2,459 ns | 3,887 ns | 10,730 ns | 3,994 ns | 9,252 ns | 34,580 ns |

EmrysFilter's times are those of its runs against mherb/kalman; they differ by up to 8% between
runners, so times from different runners compare only to about that.

- **Against TinyEKF** the EKF is faster at every size: 0.36 to 0.85 of TinyEKF's time in `double`,
  0.30 to 0.60 in `float`.
- **Against mherb/kalman** the EKF is level at 2D with 4 anchors in `double` (73 ns against 72 ns)
  and faster everywhere else, at 0.13 to 0.77. The square-root EKF and the square-root UKF are faster
  at every size, and so is the UKF.
- **Against FrancoisCarouge/Kalman** the EKF takes 0.11 to 0.27 of the time at every size.
- **Against OpenCV** the EKF takes 2 to 4% of the time at every size.

### The marked pairs

The largest distance over the timed run, relative to the size of the mean and of the covariance of
the estimate it is measured from, as mean / covariance, over the six sizes.

| Marked pair | Measured from | EmrysFilter | The other library |
|---|---|---|---|
| EKF, `float`, mherb/kalman | the `double` estimate | up to 7e-6 / 5e-5 | 2e-4 to 4e-2 / 6e-4 to 9e-2 |
| EKF, `float`, TinyEKF | the `double` estimate | up to 7e-6 / 5e-5 | 2e-4 to 5e-3 / 1e-3 to 3e-2 |
| EKF, `float`, OpenCV | the `double` estimate | up to 7e-6 / 5e-5 | 2e-4 to 2e-3 / 4e-3 to 2e-2 |
| UKF and square-root UKF, `double` | EmrysFilter's filter | | 5e-4 to 3e-3 / 2e-3 to 4e-3 |
| UKF, `float` | each library's own `double` UKF | up to 2e-5 / 7e-5 | 1e-4 to 1e-2 / 1e-3 to 6e-2 |
| Square-root UKF, `float` | each library's own `double` filter | up to 3e-6 / 1e-5 | up to 2e-5 / 1e-4 |

- **mherb/kalman** (`9f40c2f`, 2018-08-22).
  - *Its EKF in `float`* forms the gain with an explicit inverse of the innovation covariance
    (`S.inverse()`), which loses accuracy while the prior is wide. With that one line changed to
    solve by the Cholesky factor of $S$, it stays within 6e-5 of the `double` estimate at every
    size.
  - *Its two UKFs* compute the innovation and cross covariances from sigma points drawn before the
    process noise was added, so both leave the process noise out. On a linear problem, where a UKF
    gives the Kalman filter's estimate, its UKF agrees with its own EKF to 1e-12 without process
    noise and differs by 2e-3 in the covariance with this problem's. Without process noise the two
    libraries' UKFs agree on the ranging problem to 1e-5 in 2D and 1e-3 in 3D.
  - *Its UKF in `float`* also strays from its own `double` UKF, by up to 1e-2 in the mean, likely
    because it too forms the gain with an explicit inverse.
  - Its square-root UKF allocates on the heap in its prediction, so its times include its
    allocator.
- **TinyEKF** (`ba0d2a9`, 2026-04-09) runs in `float` unless `_float_t` is defined. Its estimate
  strays in `float` while the prior is wide, from three of its choices together: the gain through an
  explicit inverse of $S$, the covariance updated as $(I - G H) P$, and no triangle mirrored. With
  all three changed it stays within 4e-5 of the `double` estimate; no one of them alone suffices.
- **OpenCV** 4.14.0 (core, imgproc, and video built from source in Release with its default CPU
  flags). `cv::KalmanFilter` is linear, and runs as an EKF by the pseudo-measurement
  $z - h(x) + H x$.
  - *In `double`* it never makes its covariance symmetric again, and the asymmetry grows until the
    filter diverges, somewhere between steps 400 and 504. So it is timed over runs of 300 steps
    that start over, untimed, at their end.
  - *In `float`*, its default, it strays within those 300 steps: it solves for the gain by a
    singular value decomposition of an innovation covariance the unmirrored covariance has left
    asymmetric. Solving by Cholesky instead keeps it within 1e-5 of the `double` estimate, and
    mirroring `errorCovPost` by `cv::completeSymm()` after each `correct()` within 4e-5.
  - With 8 anchors or more in `double`, and 12 or more in `float`, it allocates once per step.
- **FrancoisCarouge/Kalman** (`cdd8a93`, 2026-11-23; its last release, 0.5.3, has no extended
  filter with a fixed transition) runs with the Eigen backend of its `support/eigen`. Its EKF
  solves for the gain by a fully pivoted Householder QR of $S$ and updates the covariance by the
  Joseph form, its default. The pair agrees in both scalars, to 9e-14 in `double`. Its filter is
  compiled in a translation unit of its own, since its backend's `operator/` for any two Eigen
  objects takes over Eigen's division of one array by another; the two calls per step this costs
  are not measurable.

## Not measured yet

- The particle filter's scaling with threads under `std::execution::par`.
- Cycles per step on Cortex-M4F and Cortex-M7, which QEMU does not model.
- The price of each option (update form, Gaussian form, sigma points) relative to the plain EKF.

## Machines

- **Instructions:** Clang 21.1.8 with `-march=x86-64-v3`, Callgrind under Ubuntu 26.04 in WSL2.
- **Time on Linux:** Intel Core i9-13900F, Ubuntu 26.04 under WSL2 with Clang 21.1.8 and
  `-march=x86-64-v3`, pinned to one CPU by `taskset` at normal priority.
- **Time on Windows:** the same machine, Windows 11 with Clang 21.1.0 and no `-march`, and with
  MSVC 19.50, at high priority on one CPU.
- **Footprint:** the Arm GNU toolchain 14.2.1, MinSizeRel, and QEMU 10.2.1.
- **Compile time:** an Intel Raptor Lake processor under Windows 11, with Clang 21.1.0, GCC 14.2.0
  (w64devkit), and MSVC 19.50.

Every number but the compile times was measured at `ab1b0b9` on 2026-11-27, but those not measured
again. The particle filter's instructions, the footprint, and the MSVC comparisons with the other
libraries are of `68d4cfd` (2026-11-26), before `19019da`, `c679b0a`, and `ab1b0b9`. The particle
filter's times on Linux are of `2829773`, whose particle steps are the same on both sides, but for
the 3D rows at a million particles, which `9fa89d5` made comparable. The compile times are of
2026-11-01.
