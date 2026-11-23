# Benchmarks

Each benchmark solves one problem twice: once with EmrysFilter, and once in plain Eigen as a careful
user would write it by hand. Every number is the library's cost divided by the hand code's, so 1.00
means the same cost, above 1 the library costs more, and below 1 it costs less.
[design/benchmarks.md](design/benchmarks.md) explains which claim each benchmark backs, why the
reference is hand-written code, and what is still to be measured.

## Summary

| Problem | Instructions per step | Time per step, Windows | Time per step, Linux |
|---|---|---|---|
| Ranging EKF, 2D, 4 anchors | 1.80–1.89 | 1.15–1.68 | 1.17–1.71 |
| Ranging EKF, 2D, 8 anchors | 1.24–1.31 | 1.02–1.44 | 1.10–1.35 |
| Ranging EKF, 3D, 6 anchors | 1.31–1.33 | 1.11–1.21 | 1.24–1.48 |
| Ranging EKF, 12 or more anchors | 0.72–0.87 | 0.91–0.97 | 0.88–1.02 |
| Attitude EKF, analytic Jacobians | 1.56 | 1.15 | 1.21 |
| Attitude EKF, autodiff | 1.52 | 1.32 | 1.29 |
| Particle filter, 2D | 1.08–1.12 | 1.10–1.18 | 1.00–1.04 |
| Particle filter, 3D | 1.07–1.13 | 1.25–1.40 | 1.02–1.12 |

A range covers every case of its row: `double` and `float`, analytic and automatic Jacobians, and
for the particle filter its likelihoods, and in its times every particle count. Neither side
allocates on the heap during a step, in any case.

The instruction column, and the Linux times of the ranging rows, were measured on 2026-11-23 at the
commit *Factorize the innovation covariance by the library's own Cholesky factorization*. The
other times are of 2026-11-08 and have not been taken again since. That commit and two before it on the same day,
*Substitute through the gain's Cholesky factor in place of Eigen's solve* and *Stop building a
step's Jacobian where nothing reads it*, make the steps of both sides shorter. At the small sizes
the hand code's step lost a larger part of itself than the library's did, so those ratios rose
while both sides got faster: the nanoseconds are under each table.

## Method

- **The same algorithm on both sides.** The hand code uses fixed-size matrices, `noalias()`, the
  library's update form $P - K P_{xy}^T$, the library's Cholesky factorization of the innovation
  covariance and its substitution for the gain, written out, and the library's order of operations. The difference measured is the library's overhead, not a
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
| `double`, analytic | 1.80 | 1.24 | 0.82 | 1.33 | 0.83 | 0.86 |
| `double`, autodiff | 1.86 | 1.31 | 0.84 | 1.32 | 0.84 | 0.87 |
| `float`, analytic | 1.89 | 1.31 | 0.82 | 1.33 | 0.72 | 0.81 |
| `float`, autodiff | 1.80 | 1.31 | 0.82 | 1.31 | 0.73 | 0.83 |

| Time per step, Linux | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| `double`, analytic | 1.29 | 1.19 | 0.92 | 1.24 | 0.94 | 1.00 |
| `double`, autodiff | 1.40 | 1.12 | 0.95 | 1.30 | 0.96 | 1.02 |
| `float`, analytic | 1.17 | 1.10 | 0.92 | 1.47 | 0.88 | 0.95 |
| `float`, autodiff | 1.71 | 1.35 | 0.94 | 1.48 | 0.91 | 0.99 |

| Time per step, Linux, `double`, analytic | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Hand code | 55 ns | 128 ns | 454 ns | 137 ns | 402 ns | 1,179 ns |
| EmrysFilter | 71 ns | 152 ns | 417 ns | 170 ns | 376 ns | 1,177 ns |
| Hand code, a commit earlier | 98 ns | 205 ns | 636 ns | 151 ns | 505 ns | 1,203 ns |
| EmrysFilter, a commit earlier | 115 ns | 237 ns | 579 ns | 184 ns | 473 ns | 1,298 ns |
| Hand code, two commits earlier | 147 ns | 500 ns | 833 ns | 349 ns | 749 ns | 1,756 ns |
| EmrysFilter, two commits earlier | 167 ns | 549 ns | 795 ns | 385 ns | 698 ns | 1,880 ns |

At 2D with 4 anchors in `double` with analytic Jacobians, the hand code takes 401 instructions and
the library 723. The tables were measured on 2026-11-23. The commit named under the summary
replaced Eigen's Cholesky factorization of the innovation covariance, on both sides, by one written
out without square roots, and the commit before it replaced Eigen's solve for the Kalman gain by a
substitution through the factor ([Against other libraries](#against-other-libraries) says what was
found). Together they took between 37% and 72% off the time of the library's step. What the
library adds to the hand code's step at 2D with 4 anchors is about what it was, 16 ns and 320
instructions where it was 17 ns and 350, and is now a larger part of a shorter step.

Until the first of the three commits, every prediction built the process model's
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
| Analytic measurement Jacobians | 3,439 | 5,364 | 1.56 | 314 ns | 362 ns |
| Autodiff measurement Jacobians | 5,011 | 7,622 | 1.52 | 375 ns | 495 ns |

The instruction counts are of 2026-11-23. Three commits of that day cut them: before the first the
library took 7,281 and 9,439, the second took about 120 off both sides, and the third, named under
the summary, about 540 off both sides, which raises the ratio. This page gave 4,830 and 7,186 for the library at first (1.18 and 1.27), which no commit
since the launch reproduces. The times are of 2026-11-08 and have not been taken again.

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

The instructions are of 2026-11-23. Before the commit of that day that stopped building unread
Jacobians, each particle's step also built and discarded the process model's Jacobian, and the library took 2,800
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

Measured on 2026-11-23 at the commit named under the summary, with Ubuntu 26.04 under WSL2 on an
Intel Core i9-13900F, Clang 21.1.8 with `-march=x86-64-v3`, each runner pinned to one CPU by
`taskset` at normal priority, the median of 2000 batches; instructions by Callgrind as above.

| EKF | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| mherb/kalman, `double`, instructions | 0.59 | 0.11 | 0.13 | 0.24 | 0.18 | 0.18 |
| mherb/kalman, `double`, time | 0.93 | 0.19 | 0.16 | 0.33 | 0.21 | 0.18 |
| mherb/kalman, `float` \*, instructions | 0.61 | 0.09 | 0.12 | 0.20 | 0.14 | 0.12 |
| mherb/kalman, `float` \*, time | 0.76 | 0.14 | 0.15 | 0.45 | 0.20 | 0.16 |
| TinyEKF, `double`, instructions | 0.46 | 0.66 | 0.34 | 0.43 | 0.48 | 0.45 |
| TinyEKF, `double`, time | 0.72 | 0.86 | 0.39 | 0.50 | 0.52 | 0.38 |
| TinyEKF, `float` \*, instructions | 0.42 | 0.54 | 0.30 | 0.32 | 0.39 | 0.30 |
| TinyEKF, `float` \*, time | 0.61 | 0.63 | 0.35 | 0.75 | 0.53 | 0.32 |
| OpenCV, `double`, instructions | 0.02 | 0.03 | 0.04 | 0.04 | 0.05 | 0.05 |
| OpenCV, `double`, time | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| OpenCV, `float` \*, instructions | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 | 0.03 |
| OpenCV, `float` \*, time | 0.02 | 0.03 | 0.03 | 0.06 | 0.04 | 0.03 |

| Against mherb/kalman | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| Square-root EKF, `double`, instructions | 0.24 | 0.39 | 0.75 | 0.36 | 0.63 | 0.99 |
| Square-root EKF, `double`, time | 0.67 | 0.82 | 1.05 | 0.74 | 0.88 | 1.13 |
| Square-root EKF, `float`, instructions | 0.15 | 0.25 | 0.48 | 0.25 | 0.41 | 0.76 |
| Square-root EKF, `float`, time | 0.55 | 0.75 | 0.97 | 0.69 | 0.85 | 1.04 |
| UKF, `double` \*, instructions | 1.40 | 0.41 | 0.36 | 0.64 | 0.52 | 0.47 |
| UKF, `double` \*, time | 1.10 | 0.49 | 0.44 | 0.77 | 0.62 | 0.53 |
| UKF, `float` \*, instructions | 1.48 | 0.35 | 0.31 | 0.54 | 0.46 | 0.36 |
| UKF, `float` \*, time | 1.33 | 0.44 | 0.40 | 0.84 | 0.52 | 0.42 |
| Square-root UKF, `double` \*, instructions | 0.97 | 0.97 | 1.20 | 1.04 | 1.15 | 1.51 |
| Square-root UKF, `double` \*, time | 0.85 | 0.89 | 1.02 | 1.12 | 1.09 | 1.34 |
| Square-root UKF, `float` \*, instructions | 0.86 | 0.88 | 0.98 | 0.90 | 1.00 | 1.13 |
| Square-root UKF, `float` \*, time | 0.83 | 0.81 | 0.98 | 0.93 | 0.87 | 0.99 |

| EKF, time per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| EmrysFilter, `double` | 70 ns | 155 ns | 431 ns | 175 ns | 398 ns | 1,144 ns |
| mherb/kalman, `double` | 75 ns | 811 ns | 2,743 ns | 538 ns | 1,938 ns | 6,226 ns |
| TinyEKF, `double` | 102 ns | 183 ns | 1,068 ns | 332 ns | 703 ns | 3,078 ns |
| OpenCV, `double` | 2,698 ns | 4,128 ns | 10,034 ns | 4,165 ns | 9,641 ns | 35,095 ns |
| EmrysFilter, `float` | 54 ns | 113 ns | 365 ns | 245 ns | 370 ns | 920 ns |
| mherb/kalman, `float` \* | 71 ns | 821 ns | 2,479 ns | 542 ns | 1,873 ns | 5,875 ns |
| TinyEKF, `float` \* | 92 ns | 175 ns | 986 ns | 315 ns | 699 ns | 2,883 ns |
| OpenCV, `float` \* | 2,755 ns | 4,187 ns | 11,710 ns | 4,311 ns | 10,017 ns | 38,599 ns |

EmrysFilter's times are those of the run against mherb/kalman; they vary by up to 12% between the
three runs. The other libraries' times are 1% to 15% longer than at the measurement before this
one, with no change to their code or its build, so times of different runs compare only to about
that; each ratio is of medians taken in interleaved batches of one run.

- **Against TinyEKF** the EKF is faster at every size. In `double` it takes 0.72 and 0.86 of
  TinyEKF's time at 2D with 4 and 8 anchors and 0.38 to 0.52 at the other four sizes; in `float`,
  0.32 to 0.75.
- **Against mherb/kalman** the EKF is faster at every size: it takes 0.93 of mherb/kalman's time at
  2D with 4 anchors in `double` and 0.76 in `float`, and 0.14 to 0.45 elsewhere. The square-root
  EKF is slower at 3D with 24 anchors, by 13% in `double` and 4% in `float`, and at 2D with 16 in
  `double`, by 5%, and faster elsewhere. The UKF is slower at 2D with 4 anchors, by 10% in
  `double` and 33% in `float`, and faster elsewhere. The square-root UKF is slower in `double` at
  2D with 16 anchors and at every 3D size, by up to 34%, and faster in `float` at every size.
- **Against OpenCV** the EKF takes 2 to 6% of the time at every size.

Three commits earlier EmrysFilter's EKF was slower than both small libraries at the small sizes:
2.20 times mherb/kalman's time at 2D with 4 anchors, and 3.22 times TinyEKF's at 2D with 8. Its
instructions did not explain that, and timing each part of the hand-written filter's step alone
did. At 2D with 4 anchors, Eigen's Cholesky factorization of the innovation covariance took 58 of
the step's 144 ns, and Eigen's solves for the gain, a row at a time, 71, the two together nine
tenths of the step. The solves took 208 instructions, about half an instruction per cycle.
Callgrind's simulation had ruled out memory before that: every filter misses the first-level
cache under 3 times a step.

Both are now written out, in the library and in the hand code, for a matrix of a fixed size up to
24. One commit replaced the solves by a substitution through the factor, every row of the gain
together. The next, named under the summary, replaced the factorization. What decided its form is
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

Two things are left at the small sizes. The library's step is longer than the hand code's by
about what it was before these commits, 16 ns at 2D with 4 anchors, and where that goes is not
measured yet. And mherb/kalman's UKF is ahead at 2D with 4 anchors in the tables above. The UKF
also factorizes the state covariance for its sigma points, which at the commit of the tables was
still Eigen's factorization. The commit after it, *Take a covariance's Cholesky root by the
library's own factorization*, changed that, and in two runs after it the UKF at 2D with 4 anchors
took 0.90 and 0.93 of mherb/kalman's time in `double` and 1.03 and 1.04 in `float`; the tables are
not taken again for it. The library's other Cholesky factorizations are still Eigen's.

At 3D with 6 anchors a step takes longer in `float` than in `double`, for EmrysFilter (245 ns
against 175 ns) and for the hand-written Eigen filter (161 ns against 137 ns), though it takes
fewer instructions, and the library's `float` step there is 1.47 times the hand code's, where its
`double` step is 1.24 times. One cause was found in the substitution: Eigen vectorizes a column of
six `float`s from wherever its address lets a 16-byte packet start, so that one column's stores
are not the next operation's loads, and the processor waits on each. The substitution and the
factorization pad their columns to whole packets for that reason. Whether the same happens in the
rest of the 6-state step in `float` has not been measured.

### The marked pairs

The largest distance over the timed run, relative to the size of the mean and of the covariance of
the estimate it is measured from, as mean / covariance, over the six sizes.

| Marked pair | Measured from | EmrysFilter | The other library |
|---|---|---|---|
| EKF, `float`, mherb/kalman | the `double` estimate | up to 1e-5 / 4e-5 | 2e-4 to 4e-2 / 6e-4 to 9e-2 |
| EKF, `float`, TinyEKF | the `double` estimate | up to 1e-5 / 4e-5 | 2e-4 to 5e-3 / 1e-3 to 3e-2 |
| EKF, `float`, OpenCV | the `double` estimate | up to 1e-5 / 4e-5 | 2e-4 to 2e-3 / 4e-3 to 2e-2 |
| UKF and square-root UKF, `double` | EmrysFilter's filter | | 5e-4 to 3e-3 / 2e-3 to 4e-3 |
| UKF, `float` | each library's own `double` UKF | up to 2e-5 / 5e-5 | 1e-4 to 1e-2 / 1e-3 to 6e-2 |
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
