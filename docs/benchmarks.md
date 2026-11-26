# Benchmarks

Each benchmark solves one problem with EmrysFilter and in plain Eigen as a careful user would write
it by hand. Every number is the library's cost divided by the hand code's, so 1.00 means the same
cost, above 1 the library costs more, and below 1 it costs less.

The EKFs have two hand-written references, which differ only in how they factorize the innovation
covariance and solve for the gain. *Hand code* does it as the library does, with a Cholesky
factorization without square roots written out a column at a time, so that the ratio measures the
library's overhead alone. *Hand code with Eigen's LLT* does it as a user of Eigen writes it,
`Eigen::LLT<Matrix>{S}.solve(Pxy.transpose()).transpose()`, so that the ratio measures what the
library gives that user.
[design/benchmarks.md](design/benchmarks.md) explains which claim each benchmark backs, why the
reference is hand-written code, and what is still to be measured.

## Summary

| Problem | Instructions per step | Time per step, Windows | Time per step, Linux | Time per step, Linux, against hand code with Eigen's LLT |
|---|---|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.61–1.91 | 1.36–1.69 | 1.17–1.72 | 0.43–0.65 |
| Ranging EKF, 2D, 8 anchors | 1.27–1.44 | 1.16–1.27 | 1.08–1.40 | 0.34–0.49 |
| Ranging EKF, 3D, 6 anchors | 1.32–1.36 | 1.32–1.49 | 1.37–1.50 | 0.73–0.84 |
| Ranging EKF, 12 or more anchors | 1.10–1.20 | 1.09–1.32 | 1.09–1.25 | 0.38–0.64 |
| Attitude EKF, analytic Jacobians | 1.56 | 1.57 | 1.47 | 1.07 |
| Attitude EKF, autodiff | 1.52 | 1.68 | 1.49 | 1.14 |
| Particle filter, 2D | 1.06–1.10 | 1.29–1.45 | 1.02–1.08 | |
| Particle filter, 3D | 1.04–1.10 | 1.29–1.56 | 1.07–1.15 | |

**The library costs more than the hand code in every case.** Until 2026-11-25 the rows of 12 or
more anchors were below 1, at 0.72 to 0.87 of the hand code's instructions and 0.89 to 1.10 of its
time on Linux, because the hand code they were measured against did more work than the library: it
took its larger products by Eigen's run-time blocked product, where the library takes Eigen's
coefficient-based one, and it zeroed scratch matrices it then overwrote. The hand code now
multiplies and allocates as the library does, and the library no longer zeroes its own
factorization's scratch ([design/benchmarks.md](design/benchmarks.md), section 1).

A range covers every case of its row: `double` and `float`, analytic and automatic Jacobians, and
for the particle filter its likelihoods, and in its times every particle count. Neither side
allocates on the heap during a step, in any case. The particle filter factorizes no innovation
covariance, so it has one reference. The Windows column is Clang's; MSVC's is under
[With MSVC](#with-msvc).

Against hand code with Eigen's LLT the library's ranging EKF takes 0.34 to 0.84 of the time and
0.22 to 0.53 of the instructions. Its attitude EKF takes 0.73 and 0.85 of that hand code's
instructions but 1.07 and 1.14 times its time: there the hand code's two 3×3 factorizations per
step are a small part of it, and the library's overhead, which the column before measures, is the
larger.

Every number on this page but the compile times was measured on 2026-11-26 at the commit
*Measure the footprint of each ranging EKF on the Cortex-M, TinyEKF's included*, and each time
ratio is the median of three runs, except the particle filter's, of one. The particle filter's
times on Linux and Windows and its Windows ratios are of the commit *Keep every header free of
unused names in a Release build*, three before it, whose particle steps are the same on both sides,
but for the 3D rows at a million particles on Linux, which the commit after it made comparable.

Four commits of 2026-11-23 made the steps of both sides shorter: *Stop building a step's Jacobian
where nothing reads it*,
*Substitute through the gain's Cholesky factor in place of Eigen's solve*,
*Factorize the innovation covariance by the library's own Cholesky factorization*, and
*Take a covariance's Cholesky root by the library's own factorization*. At the small sizes
the hand code's step lost a larger part of itself than the library's did, so those ratios rose
while both sides got faster: the nanoseconds are under each table. The attitude EKF's Linux ratios
of 2026-11-08, before them, were 1.21 and 1.29.

## Method

- **The same algorithm on both sides.** The hand code uses fixed-size matrices, Eigen's lazy
  products as the library takes them, scratch matrices left uninitialized where they are written
  before they are read, the library's update form $P - K P_{xy}^T$, the library's Cholesky factorization of the innovation
  covariance and its substitution for the gain, written out, and the library's order of
  operations. The difference measured is the library's overhead, not a difference of algorithm.
  The hand code with Eigen's LLT differs from it in the factorization, the solve, and its
  products, which are `noalias()` ones, as a user of Eigen writes them.
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
| `double`, analytic | 1.61 | 1.27 | 1.13 | 1.35 | 1.15 | 1.20 |
| `double`, autodiff | 1.76 | 1.30 | 1.14 | 1.34 | 1.16 | 1.18 |
| `float`, analytic | 1.91 | 1.44 | 1.10 | 1.36 | 1.19 | 1.10 |
| `float`, autodiff | 1.80 | 1.42 | 1.10 | 1.32 | 1.19 | 1.10 |

| Time per step, Linux | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 1.40 | 1.18 | 1.11 | 1.37 | 1.19 | 1.21 |
| `double`, autodiff | 1.38 | 1.11 | 1.17 | 1.46 | 1.25 | 1.24 |
| `float`, analytic | 1.17 | 1.08 | 1.09 | 1.50 | 1.16 | 1.09 |
| `float`, autodiff | 1.72 | 1.40 | 1.16 | 1.50 | 1.23 | 1.10 |

Against hand code with Eigen's LLT:

| Instructions per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 0.25 | 0.28 | 0.38 | 0.44 | 0.36 | 0.52 |
| `double`, autodiff | 0.30 | 0.32 | 0.38 | 0.50 | 0.39 | 0.53 |
| `float`, analytic | 0.26 | 0.22 | 0.32 | 0.36 | 0.28 | 0.29 |
| `float`, autodiff | 0.32 | 0.26 | 0.31 | 0.40 | 0.30 | 0.30 |

| Time per step, Linux | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 0.58 | 0.49 | 0.45 | 0.73 | 0.47 | 0.63 |
| `double`, autodiff | 0.58 | 0.49 | 0.47 | 0.80 | 0.50 | 0.64 |
| `float`, analytic | 0.43 | 0.34 | 0.38 | 0.84 | 0.45 | 0.45 |
| `float`, autodiff | 0.65 | 0.47 | 0.39 | 0.84 | 0.49 | 0.47 |

Each time ratio is the median of three runs of the runner, each of which times the library against
both references in interleaved batches; the three runs differ by at most 0.07 against hand code
and 0.04 against hand code with Eigen's LLT.

| Time per step, Linux, `double`, analytic | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Hand code | 52 ns | 118 ns | 338 ns | 122 ns | 289 ns | 885 ns |
| EmrysFilter | 73 ns | 139 ns | 374 ns | 167 ns | 344 ns | 1,073 ns |
| Hand code with Eigen's LLT | 126 ns | 299 ns | 833 ns | 230 ns | 733 ns | 1,707 ns |
| Earlier: hand code, with Eigen's factorization | 98 ns | 205 ns | 636 ns | 151 ns | 505 ns | 1,203 ns |
| Earlier: EmrysFilter, with Eigen's factorization | 115 ns | 237 ns | 579 ns | 184 ns | 473 ns | 1,298 ns |
| Earlier: hand code, with Eigen's solve too | 147 ns | 500 ns | 833 ns | 349 ns | 749 ns | 1,756 ns |
| Earlier: EmrysFilter, with Eigen's solve too | 167 ns | 549 ns | 795 ns | 385 ns | 698 ns | 1,880 ns |

At 2D with 4 anchors in `double` with analytic Jacobians, the hand code takes 394 instructions, the
library 635, and the hand code with Eigen's LLT 2,531. The first three rows of nanoseconds are of
2026-11-26, the medians of three runs. The four earlier rows are of the two commits before the one
named next, with Eigen's factorization and then also Eigen's solve, a row at a time, on both
sides, and with the hand code's larger products taken by Eigen's run-time blocked product. The
commit
*Factorize the innovation covariance by the library's own Cholesky factorization*
replaced Eigen's Cholesky factorization of the innovation covariance, on both sides, by one written
out without square roots, and the commit before it replaced Eigen's solve for the Kalman gain by a
substitution through the factor ([Against other libraries](#against-other-libraries) says what was
found). Together they took between 42% and 74% off the time of the library's step. What the
library adds to the hand code's step at 2D with 4 anchors is 21 ns and 241 instructions, where it
was 17 ns and 350 before them, and is now a larger part of a shorter step.

Until the first of those commits, every prediction built the process model's
Jacobian, expanded it to the whole state, and discarded it, and the library took 2,410 instructions
at 2D with 4 anchors, as it did at the public launch (2,399). The ratios this page gave at first,
1.36 at 2D with 4 anchors and 1.32 at 3D with 6 (909 against 1,233 instructions), were of an
earlier state of the library that no commit since the launch reproduces. On Windows a step of the
library takes from 80 ns (2D, 4 anchors, analytic) to 1,394 ns (3D, 24 anchors, `double`,
autodiff), and the hand code's from 54 ns to 1,238 ns. There a step at 3D with 6 anchors takes
less time in `float` than in `double`, 152 ns against 181 ns for the library, unlike on Linux.

## Attitude EKF

A unit quaternion attitude and a gyroscope bias, predicted by a 100 Hz gyroscope and updated by
gravity and a magnetic field. The hand code is the multiplicative EKF of Markley (2003) with a
6×6 error covariance and a covariance reset.

| Instructions per step | Hand code | Hand code with Eigen's LLT | EmrysFilter | Ratio | Ratio, against Eigen's LLT |
|---|---|---|---|---|---|
| Analytic measurement Jacobians | 3,425 | 7,321 | 5,327 | 1.56 | 0.73 |
| Autodiff measurement Jacobians | 4,985 | 8,889 | 7,593 | 1.52 | 0.85 |

| Time per step, Linux | Hand code | Hand code with Eigen's LLT | EmrysFilter | Ratio | Ratio, against Eigen's LLT |
|---|---|---|---|---|---|
| Analytic measurement Jacobians | 263 ns | 361 ns | 387 ns, 387 ns | 1.47 | 1.07 |
| Autodiff measurement Jacobians | 338 ns | 462 ns | 504 ns, 526 ns | 1.49 | 1.14 |

EmrysFilter's two times are those of its run against each reference, each the median of three
runs, whose ratios differ by at most 0.01. On Windows the hand code takes
227 and 284 ns and the library 357 and 477 ns.

The instruction counts are of 2026-11-26; the library's were 5,364 and 7,622 on 2026-11-23, and
5,408 and 7,616 at the commit before *Return what a writesJacobians() function writes without
copying or zeroing it* ([design/benchmarks.md](design/benchmarks.md)). Three commits of 2026-11-23
cut them: before the first the
library took 7,281 and 9,439, the second took about 120 off both sides, and the third, which
replaced the factorization, about 540 off both sides, which raises the ratio. This page gave 4,830 and 7,186 for the library at first (1.18 and 1.27), which no commit
since the launch reproduces. The times on Linux are of 2026-11-26.

## Particle filter

The bootstrap particle filter on the ranging problem, with systematic resampling when the effective
sample size falls below half. The two sides, seeded alike, hold the same particles with the same
weights to 1e-9 over 200 steps, at every particle count up to a million; with the measurement's own
likelihood they are bitwise equal. Instructions are counted at 1000 particles.

| Per particle and step | Hand code | EmrysFilter |
|---|---|---|
| 2D, Windows | 47–50 ns | 61–73 ns |
| 3D, Windows | 66–70 ns | 85–110 ns |
| 2D, Linux | 141–164 ns | 145–168 ns |
| 3D, Linux | 210–217 ns | 225–248 ns |

On Linux most of each side's step is libstdc++'s `std::normal_distribution`, whose uniform draws
take a `long double` logarithm at every call ([design/benchmarks.md](design/benchmarks.md)), so the
part common to both is about three times Windows's and the Linux ratios say less about the library.

| Instructions per particle and step | Hand code | The noise's likelihood | Its own likelihood | Batched |
|---|---|---|---|---|
| 2D | 833 | 900 | 885 | 913 |
| 3D | 1,345 | 1,482 | 1,451 | 1,397 |

The instructions are of 2026-11-26, 14 to 17 fewer for the library than on 2026-11-23. Before the
commit of 2026-11-23 that stopped building unread
Jacobians, each particle's step also built and discarded the process model's Jacobian, and the library took 2,800
to 2,880 instructions in 3D; this page gave 1,397 at first for its own likelihood in 3D, against
the hand code's 1,347. The commit *Make the Cholesky factorization public and the Gaussian's own* took 5 to 19 instructions off the
rows that draw their noise from a Gaussian, whose samplers now multiply by a factor the Gaussian
keeps. The times are of 2026-11-26.

## With MSVC

Every pair also builds with MSVC 19.50 (the `msvc-bench` preset, `/O2`, warnings as errors) and
agrees as it does with Clang, but at 2D with 8 anchors in `float`. There MSVC at `/O2`
miscompiles the benchmark's own measurement function, the same on both sides: its estimate leaves
the `double` one at the first update, 0.0186 relative in the mean, and is lost within a few steps,
so the pairs against mherb/kalman and FrancoisCarouge/Kalman are not timed there. The same code at
`/Od`, or with the ranges taken a column at a time in place of `toAnchor.colwise().norm()`, tracks
the `double` estimate to 3e-6; the library's factorization and solve are right at `/O2`, and the
expression alone in a function of its own is too, so the miscompilation needs the code around it.
MSVC 19.52, which CI runs, was not tried. Its times on Windows, the median of three runs of 2000
batches at high priority on one core, the particle filter's of one run at 1000 particles:

| Time per step, MSVC | Against hand code | Against hand code with Eigen's LLT |
|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.35–1.91 | 0.94–1.04 |
| Ranging EKF, 2D, 8 anchors | 1.25–1.60 | 0.56–0.67 |
| Ranging EKF, 3D, 6 anchors | 1.20–1.59 | 0.80–0.96 |
| Ranging EKF, 12 or more anchors, `double`, and 2D with 16 in `float` | 1.17–1.26 | 0.47–0.81 |
| Ranging EKF, 3D, 12 and 24 anchors, `float` | 0.23–0.43 | 0.52–0.69 |
| Attitude EKF, analytic and autodiff | 1.64, 1.27 | 1.21, 1.17 |
| Particle filter, 2D and 3D | 1.37–1.53 | |

**The ratios below 1 in `float` at 3D with 12 and 24 anchors are MSVC's code for the hand code.**
There its step takes 1,494 and 5,606 ns, against 458 and 1,662 ns in `double` and the library's 459
and 1,269 ns in `float`. Compiled with Eigen's `operator*` for its products in place of the lazy
ones, the hand code's step at 12 anchors takes 463 ns, from 1,980, and at 24 anchors 5,778 ns, from
7,472, where the hand code with Eigen's LLT, which differs from it then in the factorization and
the substitution alone, takes 2,144: so MSVC compiles the hand code's lazy products in `float` at
those sizes, and at 24 anchors its written-out factorization and substitution, to code several times
slower than the library's of the same operations. The library's own step there is where it is with
Clang, relative to `double`.

MSVC's steps are longer than Clang's on the same machine, the library's at 2D with 4 anchors 132 ns
against 80, and against the other libraries built by MSVC the EKF takes 1.24 times mherb/kalman's
time at 2D with 4 anchors in `double` and 1.60 in `float`, and the UKF 1.38 and 1.27 times there; at
every other size and against TinyEKF (0.26 to 0.90) and FrancoisCarouge/Kalman (0.14 to 0.39) it takes
less.

## Footprint on Cortex-M

The ranging EKF at 3D with 6 anchors in `float` with the analytic Jacobian, built for the
Cortex-M4F and the Cortex-M7 in MinSizeRel with no heap, exceptions, or RTTI, by each
implementation, and the simulator that drives it alone. A filter's footprint is its program's
less the simulator's, which every program shares. Each program runs under QEMU and ends within
2 m of the true position ([benchmarks/cortex_m/](../benchmarks/cortex_m/), the `footprint`
target).

| `.text` less the simulator's | Cortex-M4F | Cortex-M7 |
|---|---:|---:|
| EmrysFilter | 19,736 B | 19,576 B |
| Hand code | 6,904 B | 6,696 B |
| TinyEKF | 2,256 B | 2,040 B |

The library's EKF takes 2.9 times the hand-written Eigen EKF's code and 8.7 to 9.6 times
TinyEKF's, whose every product is a loop over arrays. Each adds no `.data`, and the library 16
bytes of `.bss`. The largest stack frame of each program is its `program()`, which holds the filter:
7,664 bytes for the library's on the M4F, 5,152 for the hand code's, and 6,712 for TinyEKF's,
against the simulator's 4,376. Measured on 2026-11-26 with the Arm GNU toolchain 14.2.1 and QEMU
10.2.1. Cycles per step are not measured: QEMU does not model them, and there is no board to
count them on.

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

Each pair is checked before it is timed. Most agree, to 1e-9 in `double` and 1e-4 in `float` over
the run that is timed. A pair marked \* does not agree: the other library's estimate differs from
EmrysFilter's, and a test shows the difference to be that library's, either because EmrysFilter's
estimate stays with the `double` estimate, on which every implementation agrees to 1e-9, or
because the other library's filter fails a check that involves no EmrysFilter code. A marked
ratio therefore compares two steps that do not compute the same estimate;
[the table after the ratios](#the-marked-pairs) gives how far apart they are, and the notes on
each library give the cause. TinyEKF and OpenCV run in `float` unless told otherwise, so their
marked rows are the ones their users get by default.

Measured on 2026-11-26 at the commit *Measure the footprint of each ranging EKF on the Cortex-M,
TinyEKF's included*, with Ubuntu 26.04 under WSL2 on an Intel Core i9-13900F, Clang 21.1.8 with `-march=x86-64-v3`, each
runner pinned to one CPU by `taskset` at normal priority, the median of 2000 batches, and each
ratio the median of three runs; instructions by Callgrind as above.

| EKF | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| mherb/kalman, `double`, instructions | 0.52 | 0.11 | 0.13 | 0.24 | 0.18 | 0.15 |
| mherb/kalman, `double`, time | 1.02 | 0.19 | 0.15 | 0.35 | 0.21 | 0.18 |
| mherb/kalman, `float` \*, instructions | 0.62 | 0.09 | 0.12 | 0.20 | 0.14 | 0.11 |
| mherb/kalman, `float` \*, time | 0.77 | 0.13 | 0.14 | 0.47 | 0.20 | 0.16 |
| TinyEKF, `double`, instructions | 0.40 | 0.66 | 0.33 | 0.43 | 0.46 | 0.38 |
| TinyEKF, `double`, time | 0.78 | 0.85 | 0.38 | 0.54 | 0.51 | 0.37 |
| TinyEKF, `float` \*, instructions | 0.43 | 0.53 | 0.30 | 0.32 | 0.38 | 0.26 |
| TinyEKF, `float` \*, time | 0.60 | 0.58 | 0.35 | 0.74 | 0.55 | 0.34 |
| FrancoisCarouge/Kalman, `double`, instructions | 0.11 | 0.11 | 0.15 | 0.19 | 0.16 | 0.19 |
| FrancoisCarouge/Kalman, `double`, time | 0.27 | 0.17 | 0.15 | 0.27 | 0.18 | 0.19 |
| FrancoisCarouge/Kalman, `float`, instructions | 0.12 | 0.08 | 0.13 | 0.15 | 0.12 | 0.13 |
| FrancoisCarouge/Kalman, `float`, time | 0.19 | 0.11 | 0.14 | 0.32 | 0.16 | 0.15 |
| OpenCV, `double`, instructions | 0.02 | 0.03 | 0.04 | 0.04 | 0.05 | 0.04 |
| OpenCV, `double`, time | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| OpenCV, `float` \*, instructions | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 |
| OpenCV, `float` \*, time | 0.02 | 0.02 | 0.03 | 0.06 | 0.04 | 0.03 |

| Against mherb/kalman | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Square-root EKF, `double`, instructions | 0.11 | 0.14 | 0.27 | 0.18 | 0.27 | 0.53 |
| Square-root EKF, `double`, time | 0.34 | 0.37 | 0.46 | 0.40 | 0.50 | 0.58 |
| Square-root EKF, `float`, instructions | 0.07 | 0.08 | 0.15 | 0.12 | 0.18 | 0.31 |
| Square-root EKF, `float`, time | 0.30 | 0.28 | 0.37 | 0.38 | 0.54 | 0.59 |
| UKF, `double` \*, instructions | 1.25 | 0.39 | 0.35 | 0.59 | 0.50 | 0.43 |
| UKF, `double` \*, time | 0.90 | 0.45 | 0.43 | 0.75 | 0.61 | 0.53 |
| UKF, `float` \*, instructions | 1.30 | 0.33 | 0.30 | 0.50 | 0.44 | 0.33 |
| UKF, `float` \*, time | 1.05 | 0.40 | 0.38 | 0.86 | 0.51 | 0.43 |
| Square-root UKF, `double` \*, instructions | 0.55 | 0.50 | 0.58 | 0.67 | 0.65 | 0.98 |
| Square-root UKF, `double` \*, time | 0.53 | 0.51 | 0.57 | 0.80 | 0.75 | 0.89 |
| Square-root UKF, `float` \*, instructions | 0.47 | 0.41 | 0.43 | 0.57 | 0.57 | 0.60 |
| Square-root UKF, `float` \*, time | 0.50 | 0.44 | 0.48 | 0.66 | 0.65 | 0.64 |

| EKF, time per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| EmrysFilter, `double` | 73 ns | 139 ns | 373 ns | 168 ns | 346 ns | 1,078 ns |
| mherb/kalman, `double` | 72 ns | 743 ns | 2,497 ns | 486 ns | 1,661 ns | 5,839 ns |
| TinyEKF, `double` | 99 ns | 163 ns | 968 ns | 312 ns | 676 ns | 2,888 ns |
| FrancoisCarouge/Kalman, `double` | 292 ns | 828 ns | 2,523 ns | 617 ns | 1,962 ns | 5,676 ns |
| OpenCV, `double` | 2,397 ns | 3,786 ns | 9,190 ns | 3,780 ns | 9,003 ns | 30,368 ns |
| EmrysFilter, `float` | 51 ns | 98 ns | 334 ns | 225 ns | 323 ns | 830 ns |
| mherb/kalman, `float` \* | 67 ns | 762 ns | 2,350 ns | 482 ns | 1,628 ns | 5,331 ns |
| TinyEKF, `float` \* | 86 ns | 160 ns | 909 ns | 305 ns | 584 ns | 2,479 ns |
| FrancoisCarouge/Kalman, `float` | 269 ns | 844 ns | 2,453 ns | 703 ns | 1,969 ns | 5,777 ns |
| OpenCV, `float` \* | 2,459 ns | 3,898 ns | 10,663 ns | 4,011 ns | 9,198 ns | 34,589 ns |

EmrysFilter's times are those of its runs against mherb/kalman; its times in the four runners
differ by up to 5%. Each time is the median of three runs, whose ratios differ by at most 0.18, and
each ratio is of medians taken in interleaved batches of one run, so times of different runners
compare only to about that.

- **Against TinyEKF** the EKF is faster at every size. In `double` it takes 0.78 and 0.85 of
  TinyEKF's time at 2D with 4 and 8 anchors and 0.37 to 0.54 at the other four sizes; in `float`,
  0.34 to 0.74.
- **Against mherb/kalman** the EKF is level at 2D with 4 anchors in `double`, at 1.02 of
  mherb/kalman's time (73 ns against 72 ns), faster there in `float`, at 0.77, and faster
  elsewhere, at 0.13 to 0.47. The square-root EKF is faster at every size, at 0.28 to 0.59. The UKF
  is faster at every size in `double`, by 10% at 2D with 4 anchors, and in `float` it is 5% slower
  there and faster elsewhere. The square-root UKF is faster at every size, at 0.51 to 0.89 in
  `double` and 0.44 to 0.66 in `float`.
- **Against FrancoisCarouge/Kalman** the EKF takes 0.11 to 0.32 of the time and 0.08 to 0.19 of
  the instructions at every size, in both scalars.
- **Against OpenCV** the EKF takes 2 to 6% of the time at every size.

Before those commits EmrysFilter's EKF was slower than both small libraries at the small sizes:
2.20 times mherb/kalman's time at 2D with 4 anchors, and 3.22 times TinyEKF's at 2D with 8. Its
instructions did not explain that, and timing each part of the hand-written filter's step alone
did. At 2D with 4 anchors, Eigen's Cholesky factorization of the innovation covariance took 58 of
the step's 144 ns, and Eigen's solves for the gain, a row at a time, 71, the two together nine
tenths of the step. The solves took 208 instructions, about half an instruction per cycle.
Callgrind's simulation had ruled out memory before that: every filter misses the first-level
cache under 3 times a step.

Both are now written out, in the library and in the hand code, for a matrix of a fixed size up to
24. One commit replaced the solves by a substitution through the factor, every row of the gain
together. The next replaced the factorization. What decided its form is
that a filter's steps wait on each other: the covariance that one step writes is what the next
reads, so a step takes as long as its longest chain of operations that wait on each other, however
few its instructions. Timed that way, each call's input depending on the result of the call
before, factorizing a 4×4 matrix and solving four right-hand sides takes 78 ns with Eigen's
factorization, 54 ns with a Cholesky factorization written out a column at a time, and 36 ns with
the same factorization without square roots, $S = L D L^T$ with $L$ unit triangular. With square
roots each column waits on a root and then a division; without them it waits on one division, and
the substitution scales nothing inside a pass. The pivots, and so what counts as a matrix that is
not positive definite, are the same. Timed as independent calls, which overlap in the processor,
the three take 70, 23, and 12 ns, which is why timing a part alone understates what it costs in a
filter.

The UKF spreads its sigma points by a Cholesky root of the state covariance, which was Eigen's
factorization too. Until the commit *Take a covariance's Cholesky root by the library's own factorization*
the UKF at 2D with 4 anchors took 1.10 times mherb/kalman's time in `double` and 1.33 times in
`float`; it takes 0.90 and 1.05 times now.

What is left at the small sizes is the library's own overhead: its step is longer than the hand
code's by 21 ns at 2D with 4 anchors, against 17 ns before these commits, and where that goes is
not measured yet. Since these measurements the library's factorizations of information matrices
and its local ones have moved to its own factorization too, the posterior Cramér-Rao bound's
last; none of them is in a step timed here.

At 3D with 6 anchors a step takes longer in `float` than in `double`, for EmrysFilter (225 ns
against 167 ns) and for the hand-written Eigen filter (153 ns against 122 ns), though it takes
fewer instructions, and the library's `float` step there is 1.50 times the hand code's, where its
`double` step is 1.37 times. Built for Windows without `-march`, `float` is the faster there, and
on Linux it is the slower with `-march=x86-64-v2 -mavx` too, so it comes with AVX
([design/benchmarks.md](design/benchmarks.md)). One cause was found in the substitution: Eigen vectorizes a column of
six `float`s from wherever its address lets a 16-byte packet start, so that one column's stores
are not the next operation's loads, and the processor waits on each. The substitution and the
factorization pad their columns to whole packets for that reason. Whether the same happens in the
rest of the 6-state step in `float` has not been measured.

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

### FrancoisCarouge/Kalman

FrancoisCarouge/Kalman (`cdd8a93`, 2026-11-23; its last release, 0.5.3, has no extended filter
with a fixed transition) runs with the Eigen backend of its `support/eigen`, unchanged. Its EKF
solves for the gain by a Householder QR decomposition of $S$ with full pivoting and updates the
covariance by the Joseph form, its default. The pair agrees in both scalars, so no row is marked:
to 9e-14 in `double`, over 3000 steps too, and in `float` its estimate stays within 1e-5 of the
`double` one in the mean and 3e-7 in the covariance, closer in the covariance than EmrysFilter's,
which stays within 7e-6 and 5e-5. Its filter is compiled in a translation unit of its own, since
its backend's `operator/` for any two Eigen objects takes over Eigen's division of one array by
another, which EmrysFilter's side of the pair uses; each of its steps is then two calls that
cannot be inlined, which cost nothing measurable against the same filter stepped inline
([design/benchmarks.md](design/benchmarks.md), claim 9).

## Not measured yet

- The particle filter's scaling with threads under `std::execution::par`.
- Cycles per step on Cortex-M4F and Cortex-M7, which QEMU does not model and no board is at hand
  to count.
- The price of each option (update form, Gaussian form, sigma points) relative to the plain EKF.

## Machines

The commit of the compile times of 2026-11-01 was not recorded; every other table is of the
commit the summary names.

- **Instructions:** 2026-11-26, Clang 21.1.8 with `-march=x86-64-v3`, Callgrind under Ubuntu 26.04
  in WSL2.
- **Time on Linux:** 2026-11-26, Intel Core i9-13900F, Ubuntu 26.04 under WSL2 with Clang 21.1.8 and
  `-march=x86-64-v3`, pinned to one CPU at normal priority.
- **Time on Windows:** 2026-11-26, the same machine, Windows 11 with Clang 21.1.0 and no `-march`,
  and with MSVC 19.50, at high priority on one CPU.
- **Footprint:** 2026-11-26, the Arm GNU toolchain 14.2.1, MinSizeRel, and QEMU 10.2.1.
- **Compile time:** 2026-11-01, an Intel Raptor Lake processor under Windows 11, with Clang 21.1.0,
  GCC 14.2.0 (w64devkit), and MSVC 19.50.
