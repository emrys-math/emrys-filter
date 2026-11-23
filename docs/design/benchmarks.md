# Benchmarks and tests

What the benchmarks are for, which ones the library needs, how each is measured and published, and
how that differs from what the tests are for. The benchmark half is the plan behind the Verification
item "Benchmarks [benchmarks]" in [roadmap.md](roadmap.md) and the benchmark fragments left in steps
13d, 13e, 18, and 19; the test half sorts the rest of Verification into what it is evidence of.
[benchmarks], [coverage], and [embedded-builds] in [decisions.md](decisions.md) are the decisions
this plan rests on. One rule below is proposed as a decision; when adopted it goes into
decisions.md.

## Where we are

**Tests.** 587 test macros, 56 of them typed suites, expand to 789 test cases in three targets: the
unit tests (`tests/`, 725 cases), the allocation tests (`tests/allocation*.cpp`, 18 cases, their own
target because `EIGEN_RUNTIME_NO_MALLOC` must reach every translation unit), and the benchmark
preconditions (`agreement.cpp` and `allocation.cpp` under `benchmarks/ranging/`,
`benchmarks/attitude/`, and `benchmarks/particles/`, 46 cases). Every filter, distribution,
transform, space, and policy overload has a file, and every file and test says what it is evidence
for. Three kinds of test already check performance properties and should be recognized as such: the
allocation tests ([allocation]: a step on a bounded space allocates nothing), the parallel tests
([execution-policies]: every policy overload equals `Sequential` bitwise or to rounding), and the
agreement tests ([benchmarks]: the library's filter and a hand-rolled one agree to 1e-9 over
hundreds of steps and both pass an NEES test).

**Benchmarks.** Three pairs on two problems. `benchmarks/ranging/` has a plain-Eigen simulator and
the EmrysFilter EKF against the hand-rolled Eigen EKF in [benchmarks]'s 2×2 of analytic and autodiff
Jacobians, at six sizes (2D with 4, 8, 16 anchors; 3D with 6, 12, 24), in double and in float.
`benchmarks/particles/` has the bootstrap particle filter against a hand-rolled Eigen loop on the
same scenario, at 2D with 4 anchors and 3D with 6, at 1000, 10,000, and 100,000 particles, with the
likelihood from the noise's Gaussian density or from the measurement's own closed form.
`benchmarks/attitude/` has a unit-quaternion attitude with a gyroscope bias, the EmrysFilter EKF
against the hand-rolled multiplicative EKF, under analytic and autodiff Jacobians of the reference
vectors. All three run through the wall-clock harness of our own (`benchmarks/harness.h`: the
command line, the replay of a run, the interleaved pair driver, and the table) and have a `--count`
mode for Callgrind. At a fixed step the EKF's result is 0.91 to 1.44 instructions per step
relative to the hand-rolled filter in double and 0.89 to 1.35 in float (section 1); the attitude
EKF's, 1.18 and 1.27, and the particle filter's, 1.03 to 1.07, are in sections 2 and 3 below. Nothing runs them in CI, and the numbers are not yet published with their method; the README
quotes the fixed-step ranging ratios.

**Fragments.** Benchmark work is promised in five other places: the attitude benchmark of step
13e (the covariance reset on a quaternion), the fixed-step case of step 13d (a model that builds
$F$ and $Q$ once against hand code that does the same), the block-copy question of step 15b
(`argument()` and `ReadBlocks::split()` copy each block they pass to a measurement, "measured
against the ranging benchmark first"), the thread-scaling curve of step 18, the GPU crossover of
step 19, and the before-and-after measurement the borrow-stamp plan asks for on "the particle
and mixture benchmarks", which do not exist yet. This plan gives each a home.

## The rule: a test checks the implementation, a benchmark backs a claim

A **test** has an oracle known before it runs, passes or fails, runs in every CI build, and
exists once per behavior. That is why there are hundreds, and why there should be more as the
library grows. A performance property with a deterministic oracle is a test, not a benchmark:
"allocates nothing after warm-up", "equals `Sequential` bitwise", and, once the Callgrind job
exists, "takes no more than 1.1 times the hand-rolled filter's instructions".

A **benchmark** produces a number for a reader, and each one exists to support one sentence a
user might doubt. It is published with its method and its machine, re-measured at releases, and
read by people deciding whether to use the library. Its admissibility rests on tests: a pair of
implementations is timed only after the agreement and NEES tests of both pass, which is why
those tests build with the unit tests and run in CI while the timing does not. A pair against
another library that cannot agree is timed too, its numbers marked, once a test shows the
difference to be that library's [benchmarks].

What gets a benchmark, then, is a **claim**, not a feature:

- A new claim gets a new benchmark: a new kind of filter whose cost structure differs (the
  sample-based filters against the Kalman family), a new execution backend or target (threads,
  Cortex-M, a GPU), a new scalar or precision.
- A new option on an existing filter (a transform method, a covariance update form, a Gaussian
  form, a reduction policy) gets at most a row in the options table of the problem it applies
  to, and only if a user would ask what it costs.
- A new distribution, space, upkeep action, divergence, or smoother gets tests and no
  benchmark. Its correctness is the tests' business, and its cost is either covered by the
  allocation tests or is not a claim anyone makes.

The suite this gives is small and stays small: three problems with hand-rolled counterparts,
two within-library tables on the first of them, a footprint table, and a compile-cost table.
That is the whole benchmark plan, and a feature that fits none of these adds to tests only.

*Proposed as the next decision [returned-jacobians] for decisions.md when adopted:* "A test checks the
implementation and a benchmark backs a claim. Performance properties with a deterministic oracle
(allocation, bitwise equality under `Sequential`, an instruction-count ceiling against the
hand-rolled pair) are tests and gate pull requests. A benchmark is added only for a claim a user
might doubt, published with its method and machine, and never one per feature."

## The claims and the benchmark behind each

| # | Claim a user might doubt | Benchmark | Measured | Reference | Status |
|---|---|---|---|---|---|
| 1 | The abstraction costs nothing on a Kalman filter | Ranging EKF, 2×2 | instructions and ns per step, over anchor count and state dimension | hand-rolled Eigen EKF | built; not in CI, not published |
| 2 | A manifold state costs nothing extra | Attitude EKF, 2×2 | the same | hand-rolled multiplicative EKF | the pair is built, not in CI, not published; a step costs 1.18 times the hand code's with analytic measurement Jacobians and 1.27 with autodiff, the prediction 1.07 (section 2) |
| 3 | Sampled filters are as fast as a loop you would write, and scale with threads | Particle filter on the ranging problem | ns per particle-step; ratio of `par` at $T$ threads to `Sequential` | hand-rolled Eigen particle filter (sequential) | the pair is built, not in CI, not published; the thread sweep is step 18's |
| 4 | Each option has a known price | Options table on the ranging problem | ns per step relative to the plain EKF row | the library's own EKF | not started |
| 5 | Composition is additive | Filters table on the ranging problem | ns per step relative to the EKF, divided by the component count | the library's own EKF | not started |
| 6 | It fits and runs on a microcontroller | Footprint and cycles on Cortex-M4F and M7 | `.text`, `.data`, stack per function, DWT cycles per step | hand-rolled EKF and TinyEKF on the same target | after [embedded-builds]'s Cortex-M job |
| 7 | A template library's compile cost is tolerable | Compile cost of each benchmark translation unit | seconds and peak memory per compiler | the hand-rolled translation unit | built and measured on three compilers (section 7), not in CI, not published |
| 8 | A GPU pays above some particle count | Device path against host `par` | crossover particle count | the library's own host path | step 19, not planned here |
| 9 | It is no slower than the C++ Kalman library a user would otherwise pick | mherb/kalman's EKF, UKF, square-root EKF, and square-root UKF, TinyEKF's EKF, and OpenCV's `cv::KalmanFilter` run as an EKF, on the ranging problem | instructions and ns per step, at the ranging sizes and scalars | each library, as its users write it | measured 2026-11-23 on branch `mherb-benchmark`; holds against OpenCV, and fails in time against mherb/kalman's EKF and UKF at 2D with 4 anchors and against TinyEKF at 2D with 4 and 8; the pairs that do not agree are timed with a mark |

**1. Ranging.** What exists, plus the two axes [benchmarks] names and the ranging benchmark lacks: a
constant-acceleration case (state dimension 6 and 9 beside 4 and 6), and the scaling curves over
anchor count and dimension that the six sizes already give once they are plotted. All three problems
step at a fixed interval, so both sides of every pair do the work that depends only on the step
once, at construction: the library through a `FixedStepModel` (step 13d), which holds $Q$, its
factor, and, where the model is affine, $F$, and the hand code by building the same matrices in its
constructor, as a user who knows the rate would. A pair that computed them again at every step would
time work that no user at a fixed rate does. The options table (4) keeps a row for a model given
`dt` at every step, against the fixed-step EKF, which answers what passing the step costs. Sections
1 to 3 trace the counts taken before this change, with both sides given `dt` at every step, and end
with the counts at the fixed step. The step 15b block-copy question is answered on this problem
by Callgrind before and after any change to `argument()`. Compiler Explorer links to both update
steps, as [benchmarks] asks, if Compiler Explorer carries Eigen 5; otherwise the Callgrind job
publishes the disassembly of both update steps as an artifact and the results page quotes the two
instruction counts beside each other.

The scalar is the third axis: both filters take it as a template parameter, `double` by default, and
every case runs in `double` and in `float`, the scalar of an embedded target [embedded-builds]. The
filters convert the scenario once at construction, and the runner converts the simulated ranges
before any filter runs, so no conversion is timed. In float the two filters agree within 4e-6
relative over 500 steps, and the library's float filter stays within 1e-5 of its double one.
Agreement in float needed the hand code to mirror its covariance's lower triangle before subtracting
$K P_{xy}^T$, as the library does: without the mirror its two triangles drift apart by rounding, and
it strayed from double ten times as far as the library did. Measuring float also showed that the
hand code built $S$ by writing a scaled identity over all of it and then adding $H P_{xy}$, where
the library accumulates $H P_{xy}$ into a copy of $R$. At 3D with 24 anchors that cost the hand code
about 8,000 instructions per update, so the library appeared 12% cheaper than the code it is
measured against. The hand code now forms $S$ as the library does; a baseline that loses to the
library understates the library's overhead. For the same reason the hand code factors $S$ by
Cholesky since the library does [innovation-cholesky]: with the pivoted LDLT it had copied from the
library before, the library appeared 12% cheaper at 3D with 24 anchors in float. And where $S$ has a
fixed size of at most eight, the library solves the gain one row at a time
(`internal::transposedSolve()`), since Eigen sends a solve with a matrix right-hand side to its
blocked solver whatever the size; the hand code does the same. That took 2D with 4 anchors from
about 2,600 instructions per step to about 950 in the hand code and from 2,800 to 1,240 in the
library. Since 2026-11-23 both substitute through the factor themselves instead, for a fixed size
of at most 24, since Eigen's solve of a small vector turned out slow in time (claim 9, below).

The library's update also moved its result, with the innovation's residual and $S$, into a
`std::expected` and out again, and each move of a fixed-size matrix is a copy, which at 3D with
24 anchors is a 24×24 $S$ copied several times. The update is now built where it is returned
(section 2), which took 480 instructions per step off 3D with 24 anchors and 50 to 270 off the
smaller cases.

The measurement after these changes (October 2026, Clang 21 on x86-64, Callgrind in WSL, `--count`,
instructions per step relative to the hand code) is 1.00 to 1.34 in double and 1.01 to 1.34 in
float. The highest ratios are the smallest problems, 2D with 4 anchors, at 1.25 to 1.34, where the
library's fixed cost per update (the entry and exit of the functions it does not inline, the two
mirrors of the covariance, and since [innovation-cholesky] the check of the factorization and the
`std::expected` it returns) is about 230 to 330 instructions out of about 1,000. That cost did not
grow with the row-wise solve, which saved both sides the same; it is now most of a small update. 2D
with 4 anchors is above the ceiling of 1.10 that tier 1 plans, and 2D with 8 anchors and 3D with 6
are 1.03 to 1.10. From 2D with 16 anchors and 3D with 12 up, which still solve whole, both scalars
are within 0.026 of the hand code. Float takes more instructions than double at 3D with 24 anchors
on both sides (about 55,000 against 46,000), from Eigen's float kernels; it is not the library's,
and wall-clock time on a target with a single-precision unit is the footprint benchmark's question
(6).

At the fixed step (November 2026, Clang 21 on x86-64 with `-march=x86-64-v3`, Callgrind in WSL,
`--count`) the ratio is 0.91 to 1.44 in double and 0.89 to 1.35 in float:

| Instructions per step | 2D, 4 | 2D, 8 | 2D, 16 | 3D, 6 | 3D, 12 | 3D, 24 |
|---|---|---|---|---|---|---|
| double, analytic | 1.36 | 1.09 | 0.91 | 1.32 | 0.94 | 1.01 |
| double, autodiff | 1.44 | 1.11 | 0.92 | 1.33 | 0.95 | 1.01 |
| float, analytic | 1.30 | 1.06 | 0.93 | 1.16 | 0.89 | 0.96 |
| float, autodiff | 1.35 | 1.06 | 0.93 | 1.18 | 0.89 | 0.96 |

The hand code at 2D with 4 anchors takes 909 instructions per step with analytic Jacobians in
double, and the library 1,233, so the library's fixed cost is still about 320. 3D with 6 anchors is
now as far over the ceiling as 2D with 4: in double the library takes 4,503 against 3,399, and the
1,100 more have not been broken down. With 16 anchors in 2D and 12 in 3D the library takes 5 to 11%
fewer instructions than the hand code, which the wall-clock measurement also shows; that case is
to be found (see "What the measurement shows" below).

*These counts do not reproduce (2026-11-22).* Counted the same way, Callgrind over `--count` with
Clang 21.1.8 and `-march=x86-64-v3` in WSL, the hand code takes the same instructions as above
(906 at 2D with 4 anchors, 3,396 at 3D with 6), but the library takes 2,410 and 6,236, a ratio of
2.66 and 1.84; at the public launch commit (2026-11-12) it took 2,399 at 2D with 4 anchors. The
ratios at 12 anchors or more are 0.98 to 1.06. So the library's fixed cost per step at the small
sizes is about 1,500 instructions, not 320, and it was so at the launch. Whether it grew between the
measurement above and the launch, or the measurement above differed in a way not recorded, is not
known; the history before the launch is in the original repository. `docs/benchmarks.md` now gives
the counts of 2026-11-22. That fixed cost is what loses claim 9 at the small sizes. The attitude
counts below do not reproduce either: the hand code takes 4,089 and 5,669 instructions per step, as
recorded, and the library 7,281 and 9,439, ratios of 1.78 and 1.67 rather than 1.17 and 1.28.

*Found and cut (2026-11-23).* Most of that cost was one thing. `ResolvedProcess::propagate()`
expanded the Jacobian that a kinematic model's function returns to the whole state at every call,
by block assignments whose sizes are run-time values, and the prediction over a cached transition,
each particle's step, and each sigma point of a transform kept only the value. At 2D with 4 anchors
the prediction took 1,155 instructions, 891 of them for that Jacobian, where the hand code's takes
64. The commit *Stop building a step's Jacobian where nothing reads it* gives those callers the
value alone (`propagateValue()`), predicts in place over a cached transition, and copies the
Jacobians that are read in blocks of fixed size. The prediction now takes 83 instructions and the
step 1,315, against the hand code's 906: ratios of 1.45 at 2D with 4 anchors, 1.21 at 3D with 6
(4,096 against 3,396), and 0.89 to 1.00 from 12 anchors, in instructions, and in time on Linux 1.13,
1.10, and 0.89 to 1.07. The attitude EKF takes 6,030 and 8,288 instructions (1.47 and 1.46), and the
particle filter 1.07 to 1.13 times the hand code's per particle and step, where it took 2.1 times.
The 1.25 ceiling of claim 1 is still missed at 2D with 4 anchors. What is left there is in the
update, about 410 instructions over the hand code's: 60 in the gain's solve, the same code compiled
otherwise, 40 gathering the covariance columns of the blocks read, and the rest in the calls of
five functions that are not inlined and in building the update that is returned. Cutting the
last means restructuring the update; whether to is open.

**2. Attitude.** A unit quaternion attitude with a gyroscope bias, `Composite<Attitude, Bias>` over
`UnitQuaternion` and `Linear<3>`, driven by gyroscope inputs and corrected by two reference-vector
measurements (gravity and a magnetic field), the problem of Markley (2003) and Solà (2017), both
already in the bibliography. The hand-rolled filter is the multiplicative EKF as those sources give
it: a 6×6 error covariance, a quaternion mean updated by a small rotation, and Markley's covariance
reset. The 2×2 is the same as ranging's: the measurement Jacobians analytic or by autodiff in the
tangent space. This is the benchmark that prices the reset of [covariance-reset] and the `map()`
view of [unit-quaternion], and it is the one that shows the library's manifold machinery against the
hand code every attitude filter author has written.

The pair is `benchmarks/attitude/`. The body turns at a varying rate. A 100 Hz gyroscope reads that
rate plus a bias, which drifts as a random walk, plus white noise. Each step is a prediction through
the reading and then an update by gravity and by the magnetic field in turn, each a unit vector seen
in the body frame with independent noise on its three coordinates. The library's side is what a user
writes: a `gyroIntegration()` process model that turns the attitude by the reading, a sure input
tagged `Gyro`, less the bias, through `UnitQuaternion::map()` and `exponential()`, composed with
`RandomWalk<Bias>`, and a measurement model on `Attitude` per reference vector. The hand code is Markley's filter in the body-frame form of Solà
(2017): a quaternion mean, a 6×6 covariance, the analytic transition
$F = [R(u)^T, -J_r(u)\,\Delta t; 0, I]$ for the turn $u$, and a reset. Where the two would otherwise
differ to first order, the hand code takes the library's choices: the update form $P - K P_{xy}^T$
with a Cholesky factorization of $S$ [innovation-cholesky] and the gain solved a row at a time, and
the exact reset $G = J_r(\delta)$, where Markley keeps the covariance unchanged and Solà keeps $G$
to first order. The two then agree to 1e-9 over 1000 steps. Without the reset, the hand code differs
from the library by 0.009 rad after the second update of the first step, so the agreement test
detects a missing or wrong reset. The 2×2's linearization axis covers the measurements only. A
fixed-step model caches its transition only when the model is affine in everything it reads, which
a gyroscope-driven attitude is not, so `gyroIntegration()` returns its Jacobian at the state with its
step [returned-jacobians], as the hand code writes it, in both arms.

The first measurement (October 2026, Clang 21 on x86-64, Callgrind in WSL, `--count`, per step of a
prediction and two updates) was 1.61 instructions relative to the hand code with analytic
measurement Jacobians and 1.62 with autodiff. After the copy and allocation cleanup and the change
to sensors of step 17j it was 1.57 for both (11,727 against 7,448, and 14,183 against 9,022). Most
of that difference was in the prediction, 4,317 instructions against 1,000: with no `transition()`,
the library linearized it by forward-mode autodiff over all seven coordinates, with derivative
vectors of six, which took about 3,100 instructions (`plus()` on the composite, the user's
`propagate()` with its quaternion product and exponential, and the composite's walk over its blocks,
all in dual numbers), and about 1,200 more went to $F P F^T + Q$ and copies. [returned-jacobians] lets
a block give its Jacobian at the state, and `GyroIntegration` now does, so the library takes the
hand code's path: one exponential, one rotation matrix, one right Jacobian, and two 6×6 products.
The ratio was then 1.16 with analytic measurement Jacobians and 1.23 with autodiff. The library then
moved to a Cholesky factorization of $S$ that returns an error when it fails [innovation-cholesky],
and the hand code moved with it. That saved the hand code 61 instructions per update and cost the
library 68; the library's 129 more, for checking the factorization and returning a `std::expected`,
have not been broken down. The ratio was then 1.20 with analytic measurement Jacobians and 1.25 with
autodiff. More than half of each update, about 1,760 instructions on both sides, was the solve of
the fixed 3×3 $S$ against the 6×3 cross-covariance, because Eigen sends every solve with a matrix
right-hand side to its blocked level-3 triangular solver, whatever the size. The library now solves
the gain one row at a time where $S$ has a fixed size of at most eight
(`internal::transposedSolve()`), which Eigen unrolls, and the hand code does the same. That saved
the library 1,627 instructions per update and the hand code 1,617, so the step is less than two
thirds of what it was on both sides, and the ratio is now 1.35 with analytic measurement Jacobians
(5,507 against 4,092) and 1.39 with autodiff (7,887 against 5,672): the library's overhead did not
change, and it is a larger share of a smaller step.

A profile of each side alone then showed that the unit quaternion's `plus()`, `normalize()`, and
reset Jacobian were given its rows as blocks of a size known only at run time, because the
composite and the reset cut a block's rows out with `middleRows(offset, rows)`, so Eigen did not
unroll the quaternion's norms and products. They now take a block of the compile-time size
wherever the block's space has one (`internal::middleRowsOf()`), which saved 429 instructions per
step in both arms: the ratio was 1.24 with analytic measurement Jacobians and 1.31 with
autodiff.

The update also moved its branch, with the innovation's residual and 3×3 $S$, which are
fixed-size and so copied by every move, three times: into a `std::expected` of the branch, into
the `FilterUpdate`, and into the `std::expected` of that. The update conditioned in place is now
built where it is returned: `std::expected<void, GaussianError>{}.transform()` initializes its
value from the `FilterUpdate` prvalue, whose branch and innovation are constructed in place, and
the lambda that makes it returns it once, by name. Clang 21 constructs such a named return value
in the caller's object only when the generic lambda's return type is written out rather than
deduced, so it is; without that the whole update was copied by `memcpy` on return. That took the
step to 4,938 instructions with analytic measurement Jacobians, 1.21 times the hand code's
4,092, and to 7,408 with autodiff, 1.31 times 5,672.

The prediction computed the exponential of the turn twice, in `GyroIntegration`'s `propagate()` and
in its `jacobian()` for $R(u)^T$, where the hand code computes it once and takes the rotation from
it, since the old block contract's two functions were called separately and could share nothing. A
model's function now returns both, `withJacobian()` [returned-jacobians], and `gyroIntegration()` does,
which took the prediction from 1,233 instructions to 1,073 and the step to 4,778 with analytic
measurement Jacobians, 1.17 times the hand code's, and to 7,248 with autodiff, 1.28 times.

| Instructions per step | hand code | EmrysFilter |
|---|---|---|
| prediction | 1,000 | 1,073 |
| two updates, analytic | 3,078 | 3,705 |
| two updates, autodiff | 4,658 | 6,175 |

The 73 more in the prediction have not been broken down. Of the 314 more per analytic update, the factorization, the
covariance's $K P_{xy}^T$, the reset Jacobian, and the trigonometry cost the same on both sides.
What remains is the library's plumbing, which has not been broken down again since the update was
built in place: the calls through `updateInto()` and the linearization, the copies around the
linearization, and, estimated from the profile rather than measured apart, about 35 in the gain's
solve and 50 in the reset's congruence over the hand code's products. With autodiff the updates'
ratio is 1.33: differentiating through `plus()` costs the library about 1,230 instructions per
update over the analytic update, against about 790 for the hand code.

At the fixed step the library's prediction went through `FixedStepModel`, whose `predict()` with an
input did not use a block's Jacobian, so it differentiated `GyroIntegration` again, and the step
took 7,468 instructions, 1.83 times the hand code's (1.73 with autodiff). The fixed step now
predicts by the block's Jacobian as a step given at each call does [returned-jacobians]. At the fixed
step (November 2026, as in section 1) the step takes 4,830 instructions with analytic measurement
Jacobians, 1.18 times the hand code's 4,092, and 7,186 with autodiff, 1.27 times 5,672. The
prediction is 1,069 and the two analytic updates 3,706, as in the table. In wall-clock time the
ratio is 1.15 to 1.32 (see "The second wall-clock measurement" below).

**3. Particles.** The ranging problem again, with the bootstrap particle filter in place of the
EKF, so the anchors, motion, and simulator are shared (`benchmarks/particles/scenario.h` takes
them from `ranging/scenario.h`). The scenario's figures change, because the bootstrap filter is
a different instrument: 2 m range noise and a 3 m initial position uncertainty, so that a
thousand particles keep tens within the first likelihood, and ranges every second with an
acceleration density of 0.5 m²/s³, because the filter's only source of diversity after a
resample is the process noise, and under the EKF's 0.1 s step it moves a particle 6 mm, so the
cloud collapses onto one ancestor within a few resamples (the NEES then climbs past a hundred
times the degrees of freedom). The hand-rolled reference is the loop a careful user writes: the
particles in the columns of one fixed-row matrix, log weights, a propagate loop that adds a draw
of the noise, a weight loop with the closed-form likelihood of independent Gaussian ranges, and
systematic resampling into reused scratch when the effective sample size falls below half.

The comparison is particle by particle, not in distribution as first planned. The hand code
takes the library's factor of the process noise (the pivoted LDLT of `semidefiniteFactor()`
rather than a Cholesky factor, as the ranging hand code takes the library's factorization of $S$) and
draws its normals and its resampling offset in the library's order from the same kind of
generator, so the two filters, seeded alike, hold the same particles with the same weights to
1e-9 over 300 steps. That is a stronger precondition than agreement of moments, and it makes
the timing a comparison of the same arithmetic. The NEES test is over 50 runs with the interval
widened by a quarter for the bias of a finite sample; the 3D test case needs 4000 particles to
pass it, where 1000 give a NEES of about twice the degrees of freedom.

The axes are the particle count (1000, 10,000, 100,000) and where the likelihood comes from,
which is the pair's own version of the ranging 2×2: with `Likelihood::Noise` the measurement
has additive Gaussian noise and the library evaluates its density at each particle, which
factorizes $R$ once and then solves a triangular system of the anchor count's size per
particle; with `Likelihood::Own` the measurement supplies `logLikelihood()` in the closed form
the hand code uses, which the particle filter calls instead.

The first measurement (October 2026, Clang 21 on x86-64, Callgrind in WSL at 1000 particles,
`--count --particles 1000`) was 1.26 to 1.41 instructions per step relative to the hand loop:
1.38 and 1.41 in 2D, 1.26 and 1.30 in 3D, the second of each pair the noise-density row. The
whole-step ratio understates the abstraction's cost, because 69% of every step on both sides is
`std::normal_distribution` (about 1,130 instructions per particle in 3D for six normals,
identical in both). Net of the draws, per particle and step in 3D with the own likelihood, the
hand loop spent 168 instructions on the update and 48 on the effective sample size and
resampling together, and the library spent 290 on the update, 91 on `ess()`, and 163 on
`resample()`; its prediction was within 17 instructions of the hand loop's. The difference was
not the per-particle arithmetic but passes over the weights: `update()` took
`logTotalWeight()` of weights its last call had left normalized and then a second log-sum-exp,
`ess()` took `logTotalWeight()` again and exponentiated again, and `resample()` took it a
third time before exponentiating into its scratch, so the library evaluated `exp` about six
times per particle per step where the hand loop does twice.

The particle set now records its log total weight as zero where `normalize()`, `resample()`, or
a filter's update has just scaled the weights to sum to one, and its weight mutators forget it,
so `ess()`, `resample()`, and the next update read it instead of summing the weights again.
That left one `exp` per particle in the update, one in `ess()`, and one in `resample()` on the
steps that resample, and a ratio of 1.16 to 1.25. Most of what remained was the kind of `exp`:
the library called libm's scalar `exp` once per particle, in the streaming log-sum-exp and in
`resample()`'s linear weights, where the hand loop's `.exp()` on a whole array is vectorized.
The log-sum-exp of an array now takes the maximum and then the sum of exponentials relative to
it as whole-array expressions, range by range under a policy, and `resample()` forms its linear
weights the same way. The ratio is now 1.04 to 1.07, 1.04 for the own likelihood and 1.07 for
the noise density in both 2D and 3D, under the ceiling of 1.10. In 3D with the own likelihood
the library spends 1,397 instructions per particle and step against the hand loop's 1,347; the
50 more are a few each in the prediction and in the passes that find the maximum and check the
weights. The noise-density row costs 45 instructions more per particle and step than the
own-likelihood row in 3D and 23 in 2D, the triangular solve against an isotropic $R$; whether `Gaussian::logDensity()` should recognize a diagonal or
scalar covariance is likewise the roadmap's question. In wall-clock time the ratio follows these
counts on Linux and is higher on Windows in 3D (see "The first wall-clock measurement" below).

Thread scaling, `par` at 1, 2, 4, 8, and 16 threads against `Sequential` at $10^3$ to $10^6$
particles, is step 18's curve and still to write on this pair; thread results come from MSVC's
library and from Linux with TBB, and w64devkit's GCC, which runs `par` on one thread, is
excluded by name.

**4 and 5. Within-library tables.** No hand-rolled reference, because the claim is relative:
each row is a configuration of the library on the ranging problem and the number is its time
per step over the plain EKF's. The options table has one row per choice a user makes:
`Linearized`, `Unscented`, `Unscented` over a model with `affineIn()`; `updates::Standard`, `updates::Symmetric`,
`updates::Joseph`; the covariance, square-root, information, and square-root information forms of
the Gaussian; `Collapse` against `Keep` with a two-component mixture noise; a model given `dt`
at every step against the fixed-step model the pairs use. The filters table has one row per recursion over the same state: the
Kalman filter, the Gaussian mixture filter with $K$ components, the IMM with $M$ modes, the
Rao-Blackwellized particle and point-mass filters at a given count, each divided by its count so
that "additive" is a number near one. These two tables are also where the borrow-stamp plan's
"particle and mixture benchmarks before and after" are measured.

**6. Footprint.** [embedded-builds]'s Cortex-M job compiles the ranging and attitude EKFs in `float`
for Cortex-M4F and Cortex-M7 with no heap and records `.text` and `.data` from the linker map and
stack use per function from `-fstack-usage`; the table shows them beside the hand-rolled filter's.
Cycles per step from the DWT counter are measured by hand on one board per core, as
[embedded-builds] says, and published with the board and clock.

**7. Compile cost.** `benchmarks/<problem>/compile_cost.cpp` is the problem's filter as a program
uses it: built, run over a trajectory, and its estimate returned from a function with external
linkage, so the compiler emits all of it. It compiles once with the EmrysFilter side and once with
the plain-Eigen side, for each linearization (each likelihood, for particles, against the one
plain-Eigen loop). Beside them are each umbrella header alone, against `Eigen/Dense` alone, and
`examples/consumer/main.cpp`, where a new user starts. The units are object libraries in a build
with `EMRYS_FILTER_BUILD_BENCHMARKS`, so building them checks that they compile, and
`tools/compile_cost.py` (the `compile_cost` target) times their commands from
`compile_commands.json`: CPU and wall seconds, the least of five runs, and peak memory, with the
ratio of each unit to its reference. `--time-trace` adds Clang's frontend, function instantiation,
and backend totals, and `--match` times any compile command instead, such as the tests'. A
template instantiation that doubles the compile time of a user program is a finding, not a
failure, so nothing gates on these numbers.

The first measurement, on 2026-11-01, on an Intel Core of family 6, model 183 (Raptor
Lake), under Windows 11, with Clang 21.1.0, GCC 14.2.0 (w64devkit), and MSVC 19.50, each unit
compiled alone on an otherwise idle machine: wall seconds, and the ratio to the reference.

| Unit | Clang Release | Clang Debug | GCC Release | MSVC Release |
|---|---:|---:|---:|---:|
| Ranging EKF, analytic | 3.4 (1.92) | 5.0 (1.90) | 6.1 (1.86) | 7.0 (1.77) |
| Ranging EKF, autodiff | 3.5 (1.84) | 5.2 (1.87) | 6.0 (1.71) | 9.2 (1.54) |
| Attitude EKF, analytic | 4.4 (1.80) | 6.9 (1.89) | 8.5 (1.84) | 12.3 (2.04) |
| Attitude EKF, autodiff | 5.1 (1.92) | 7.8 (1.81) | 8.9 (1.74) | 13.9 (1.85) |
| Particles, the noise's likelihood | 3.2 (2.22) | 4.3 (2.26) | 6.2 (2.15) | 6.1 (2.17) |
| Particles, its own likelihood | 2.8 (1.96) | 3.9 (2.04) | 5.4 (1.89) | 5.2 (1.86) |
| `EmrysFilter.h` alone | 1.3 (1.88) | 1.3 (1.75) | 2.1 (1.65) | 1.7 (1.91) |
| `Core.h` alone | 1.0 (1.50) | 1.1 (1.43) | 1.5 (1.18) | 1.3 (1.48) |
| `Spaces.h` alone | 0.7 (1.07) | 0.8 (1.05) | 1.4 (1.06) | 0.9 (1.08) |
| `examples/consumer/main.cpp` | 2.6 | 3.4 | 4.1 | 5.2 |

- A program's compile takes 1.7 to 2.2 times as long with EmrysFilter as by hand, on every
  compiler and in both build types, and its peak memory 1.5 to 2 times as much: at most 2 GB, for
  the attitude EKF in a Clang Debug build. The particle filter, whose hand-rolled loop is the
  smallest, has the largest ratio.
- Clang's trace puts nearly all of the difference in the frontend. Including the library costs
  about 0.6 s more than Eigen alone, and instantiating the filter's functions takes about twice as
  long as the hand code's (ranging, analytic, Release: 1.8 s against 1.0 s). The backend stays
  under 1 s on both sides.
- `Core.h` alone costs 0.3 s more than `Spaces.h`, which includes part of it, on every compiler
  but GCC: some header of `Core/`, or a standard header only it includes, is to be found by
  tracing each alone.
- MSVC generates code on several threads, so its CPU time is 2 to 3 times its wall time, and its
  CPU ratios (up to 2.9) exceed its wall ratios. The table gives wall time, which is what a user
  waits through, and `compile_cost.json` holds both.

**9. mherb/kalman.** The other libraries in the README's comparison table either run no filter
(manif), solve a graph (GTSAM), or run in Python (FilterPy), so mherb/kalman is the one that runs
the same problem. It has vector states only, so the problem is ranging; attitude would need a
multiplicative reset written by hand around it, and that hand code is what would be timed. Every
filter it offers has a counterpart here, so all four are compared: `ExtendedKalmanFilter` against
`KalmanFilter` with `Linearized`, `UnscentedKalmanFilter` against `KalmanFilter` with `Unscented`
at the same α, β, κ, and weights, and the two square-root filters against `SqrtKalmanFilter` with
each method. A filter is dropped only if it does not build against Eigen 5.0 unchanged. A pair
that cannot pass agreement is timed with a mark [benchmarks], and the results page says how far
apart the two are and why the difference is mherb/kalman's. Unlike the hand code,
mherb/kalman's arithmetic is kept as its users get it, update form and factorization included, so
the pair compares what a user of each library runs, not the same operations, and agreement is to
the rounding of two different update forms: its EKF forms the gain with `S.inverse()` and updates
by $P - K H P$, and its UKF with `P_yy.inverse()` and $P - K P_{yy} K^T$, neither mirroring a
triangle afterward, where the library solves by Cholesky and updates by $P - K P_{xy}^T$; the square-root pairs agree on the covariance rebuilt
from each factor, since the factors may differ by an orthogonal transformation. It has no automatic
differentiation, so its side has analytic Jacobians only and the library's autodiff rows stand
without a counterpart. It is fetched at a pinned commit (MIT license) for the benchmarks alone and
never becomes a dependency of the library.

The pairs are `benchmarks/ranging/mherb_ranging.h`, which takes mherb/kalman's filter as a
template argument, `emrys_ranging_kinds.h`, EmrysFilter's counterpart of each, `mherb_agreement.cpp`,
`mherb_allocation.cpp`, and the runner `mherb_benchmark.cpp`. No filter of either library allocates
in its steps but mherb/kalman's square-root UKF, whose prediction downdates its factor by Eigen's
`LLT::rankUpdate()`, which allocates a temporary vector of dynamic size, so that filter's times
include its allocator, as its users' do. The unscented counterparts read the whole state, so that their sigma points spread
over it as mherb/kalman's do, where `emrys_ranging.h` reads only the position.

*The EKF.* In double the two agree to 9e-10 m over 500 steps. In float they cannot agree: mherb/kalman's float EKF strays up
to 0.30 m from the double EmrysFilter EKF in 2D with 4 anchors (0.007 to 0.10 m in the other
cases), where EmrysFilter's float EKF stays within 9e-5 m. In that case $S$ has eigenvalues near 50
and 0.09, so the explicit inverse loses about 3e-5 of its accuracy in float, and $K = P H^T S^{-1}$
multiplies that loss by the prior variance of 25. A scratch copy of its header with that one line
changed, the gain solved by the Cholesky factor of $S$, stays within 6e-5 relative of the double
estimate at every size, where the original strays up to 4e-2 in the mean and 9e-2 in the
covariance; mirroring its covariance instead changes nothing. So the float pair is timed with a
mark, behind a test in `mherb_agreement.cpp` that only mherb/kalman's float EKF leaves the double
estimate, and the runner prints each float filter's largest distance from the double EmrysFilter
EKF over the run, which backs a claim of its own: in float, EmrysFilter stays accurate where
mherb/kalman does not.

*The square-root EKF.* mherb/kalman's predicts its factor by a QR decomposition and updates by
forming $P - K H P$ from the factor and factorizing it again, solving for the gain with the
innovation's factor rather than an inverse. The two agree to 1e-13 in double and to 6e-5 m in
float over 500 steps, so this pair is timed in both scalars.

*The UKFs.* Neither pair agrees, in either form or either scalar: the means differ by 7 to 13 mm
over 500 steps, which is not rounding, since float and double differ alike. mherb/kalman's update
does not draw sigma points from the predicted covariance. It reuses the points its prediction moved
through the model, which were drawn before the process noise was added, so its $P_{yy}$ and
$P_{xy}$ leave out $Q$ while its $P$ includes it. Without process noise the difference falls to
1.2e-6 m in 2D and 1.3 mm in 3D, and `mherb_agreement.cpp` checks that it stays under 1e-5
relative in 2D and 1e-3 in 3D, which shows the pairs
spread the same points with the same weights; what is left is the other half of the same choice,
since the moved points are $F$ times a Cholesky factor of the previous covariance, a different
square root of the predicted covariance than its Cholesky factor. Both libraries' UKFs pass the
NEES test on this problem, whose process noise is small beside the prior. That the difference is
mherb/kalman's takes no EmrysFilter filter to show: on a linear problem the Kalman filter is exact
and a UKF computes the same moments, and with the scenario's motion and its position measured,
mherb/kalman's UKF agrees with its own EKF to 1e-12 without process noise and differs from it by
2e-3 relative in the covariance with the scenario's, the size of its difference from EmrysFilter's
UKF on the ranging problem; `mherb_agreement.cpp` tests both. So the UKF pairs are timed with a
mark, in both forms and both scalars, and the runner prints the largest distance between the
libraries' UKFs in double and each float UKF's distance from its own double one.

**TinyEKF.** The EKF a user on a microcontroller would otherwise pick is TinyEKF
(simondlevy/TinyEKF, MIT license): one C header, sized at compile time by the macros `EKF_N` and
`EKF_M`, which allocates nothing and runs in `float` by default. It backs claim 9 on the desktop
and claim 6 on the target; this pair is the desktop half, and the Cortex-M half waits on the
footprint benchmark. It is fetched at its last commit (`ba0d2a9`, 2026-04-09) and compiled as C++
from a system include, since it is C that warns under the library's warnings. It has an EKF only,
and its caller computes the predicted mean, the predicted ranges, and both Jacobians into row-major
arrays, as its examples do. Its arithmetic is kept as its users get it: $F P F^T + Q$, the gain
$P H^T S^{-1}$ through an explicit inverse of $S$ by its Cholesky factor, and the covariance
$(I - G H) P$, with no triangle mirrored, every product a loop over the arrays.

The pair is `benchmarks/ranging/tinyekf_ranging.h`, against `emrys_ranging_kinds.h`'s EKF, with
`tinyekf_agreement.cpp`, `tinyekf_allocation.cpp`, and the runner `tinyekf_benchmark.cpp`. Its
sizes and scalar are macros fixed for a translation unit, and its header has no include guard, so
each size and scalar is an instance of the header in a namespace of its own
(`tinyekf_instance.h`); its functions are `static`, so the pair has internal linkage too. Neither
side allocates. In double the two agree to 6e-11 relative over 1000 steps, so the pair is timed
in double. In float they do not agree: TinyEKF's float EKF strays from the double EmrysFilter EKF
by 5e-4 to 3e-3 relative in the mean (0.8 to 4 mm) and up to 2e-2 in the covariance, over the
first steps, while the prior is wide; EmrysFilter's float EKF stays within 1e-5. The cause is the
three choices together. A scratch copy of its header with the gain solved by the Cholesky factor,
the covariance updated as $P - G (H P)$, and its triangles mirrored agrees to 4e-5; the solve
alone brings the mean within 3e-5 but not the covariance, and the solve with the form, the
mirroring alone, or the inverse computed in double do not suffice. So the float pair is timed with
a mark, behind a test in `tinyekf_agreement.cpp` that only TinyEKF's float EKF leaves the double
estimate, and the runner prints both libraries' float error as it does for mherb/kalman. Float is
the scalar TinyEKF runs in unless `_float_t` is defined, so the marked rows are the ones its users
get. For claim 6 this is a finding before any cycle is counted: on the scalar of the target TinyEKF
is written for, its estimate is the less accurate.

*Measured* (2026-11-23, at the commit *Substitute through the gain's Cholesky factor in place of Eigen's solve*;
`docs/benchmarks.md` gives the tables and the machine, and the paragraphs below what the
measurement before it found). Against OpenCV, EmrysFilter's EKF takes 3 to 7% of the time and of
the instructions at every size, in either scalar. Against mherb/kalman's EKF it takes 0.22 to 0.54
of the time at every size but 2D with 4 anchors, where it takes 1.50 times in double and 1.39 in
float. Its square-root EKF takes 0.55 to 0.87 of mherb/kalman's time but at 2D with 16 anchors and
3D with 24, where it takes 0.98 to 1.11. Its UKF takes 0.50 to 0.90 but at 2D with 4 anchors (1.32
and 1.63), and its square-root UKF 0.83 to 0.97 in float and 0.87 to 1.33 in double. Against
TinyEKF it takes 1.14 and 1.39 times the time in 2D with 4 and 8 anchors in double, and 1.08 and
1.21 in float, TinyEKF's default, and 0.44 to 0.85 at the other four sizes. So claim 9 holds
against OpenCV, and fails against mherb/kalman's EKF and UKF at 2D with 4 anchors and against
TinyEKF at 2D with 4 and 8. At 2D with 4 anchors the library takes 1,218 instructions and 114 ns,
mherb/kalman 1,218 and 76 ns, and TinyEKF 1,575 and 95 ns, while the hand code takes 864 and 98 ns;
what is left there is Eigen's factorization (below).

*Where the instructions go with 8 and 12 anchors (2026-11-23).* The instruction ratios against
TinyEKF are worst at 2D with 8 anchors (1.56) and 3D with 12 (1.19), between sizes where the
library takes fewer. Callgrind's counts of one case at a time, attributed to source lines, place
the cause on the library's side, in Eigen's code for a matrix of that size, 8×8 or 12×12: too large
for its unrolled fixed-size code, and small enough that the overhead of its general kernels is
most of their cost. At 2D with 8 anchors, 2,357 of the library's 4,397 instructions factorize $S$:
940 in `llt_inplace::unblocked()`, 730 in the general matrix-vector product it calls once per
column, and 687 in `LLT::compute()` around them, of which 507 compute the L1 norm that only
`rcond()` reads. TinyEKF factorizes and inverts the same matrix in about 1,000, and its whole step
takes 2,819. At 3D with 12 anchors, 9,148 of the library's 17,675 are the gain's two triangular
solves, which `transposedSolve()` solves a column at a time only up to `unrolledSolveSize`, 8, and
gives to Eigen's blocked `triangular_solve_matrix` beyond it, and 4,488 are the factorization.
With 16 and 24 anchors the same kernels spread their overhead over more work, while TinyEKF's
loops that invert $S$ grow with the cube of the anchor count (about 12,000 of its 23,264
instructions at 2D with 16). With 4 and 6 anchors TinyEKF is above its own trend, since Clang
leaves its `_mulmat()` out of line as a loop with run-time bounds: 896 of its 1,575 instructions at
2D with 4 anchors and 5,648 of its 6,484 at 3D with 6. The hand code shares the factorization and
the solves, so none of this shows in the ratios to it. Three changes follow from it, none tried:
factorizing without `LLT::compute()`'s norm; a Cholesky factorization for fixed sizes that does
not go through the general matrix-vector kernel; and a solve for sizes from 9 to somewhere under
16 that is not the blocked one. The time at 2D with 8 anchors, 3.22 times TinyEKF's, is further
from it than the instructions, 1.56 times; the next paragraph finds why.

*Where the time goes at the small sizes (2026-11-23).* Callgrind's simulation of the four EKFs
rules out memory, with under 3 first-level data cache misses per step in every filter and size,
and does not single out branches or divisions: at 2D with 4 anchors the library executes 34
divisions per step, the hand code 24, TinyEKF 15, and mherb/kalman 3, but at 2D with 8 the library
and TinyEKF execute 78 and 72. A probe then timed each part of the hand code's step alone, over
the inputs it has in a real run (the median of 4001 passes over 1000 recorded steps, pinned to
one CPU, Clang 21.1.8 with `-march=x86-64-v3` in WSL2):

| Part of the hand code's step | 2D, 4: ns | instructions | 2D, 8: ns | instructions | 3D, 6: ns | instructions |
|---|---|---|---|---|---|---|
| Whole step | 144 | 981 | 526 | 3,919 | 380 | 3,438 |
| Prediction | 8 | 104 | 5 | 104 | 23 | 573 |
| Residual, $H$, $P H^T$, $S$ | 7 | 106 | 29 | 385 | 35 | 446 |
| `LLT::compute()` of $S$ | 58 | 525 | 127 | 2,371 | 50 | 1,414 |
| The gain, a row at a time by `llt.solve()` | 71 | 208 | 354 | 965 | 224 | 626 |
| Mean and covariance update | 7 | 85 | 10 | 125 | 30 | 408 |
| The gain by a substitution written out, dividing | 12 | 116 | 52 | 415 | 38 | 425 |
| The same, multiplying by the diagonal's reciprocals | 8 | 126 | 42 | 427 | 29 | 434 |
| The gain by `S.inverse()`, as mherb/kalman | 10 | 205 | 532 | 12,586 | 308 | 7,221 |

The factorization and the gain's solves are 129 of the step's 144 ns at 2D with 4 anchors, 482 of
526 at 2D with 8, and 273 of 380 at 3D with 6. The solves are the larger part and the one out of
proportion: 208 instructions in 71 ns, about half an instruction per cycle, where the same gain by
a substitution written out, every row of the gain together a column at a time, takes 116
instructions and 12 ns with the divisions kept, and 8 ns multiplying by the reciprocals of the
factor's diagonal. It agrees with `llt.solve()`'s gain to 1e-13. So the divisions are not the
cost; what in Eigen's unrolled solve of a small fixed vector is, is not established (the scalar
stores of one row read back by the vector load of the next, which the processor cannot forward,
is the first suspect). The factorization is out of proportion too at 4x4, 58 ns where a 6x6 takes
50. With the written-out substitution in place of the solves, the hand code's step would take
about 81 ns at 2D with 4 anchors, 214 at 2D with 8, and 185 at 3D with 6, against TinyEKF's 100,
167, and 316 and mherb/kalman's 76, 749, and 507; a factorization written out the same way is the
second change to try. The library's `transposedSolve()` calls the same `llt.solve()` a row at a
time, so both changes are the library's to make, and the hand code's with it.

*The solve, cut (2026-11-23).* The commit *Substitute through the gain's Cholesky factor in place of Eigen's solve*
puts that substitution in `transposedSolve()`, for an `Eigen::LLT` of a fixed size of at most 24
(`substitutionSolveSize`), and in the two hand-written filters. Up to 24 is where it was measured:
against Eigen's blocked solve it took 115 ns in place of 352 at 3D with 12 anchors, 161 in place of
399 at 2D with 16, and 386 in place of 922 at 3D with 24. Two things about its form were measured
too. Solving every row of the gain together, a column at a time, beats solving a row at a time
from 12 anchors (115 ns against 271 at 3D with 12) and is about a fifth faster below. And the columns must be
whole packets: with a 6-state filter in `float`, the unpadded form took 184 ns at 3D with 6
anchors where `double` took 30, and 2,440 ns at 3D with 24 where Eigen's solve took 1,100. Eigen
vectorizes a column of six `float`s from wherever its address lets a 16-byte packet start, so
the stores of one column are not the loads of the next operation on it, and the processor cannot
forward them. Padded with zeros to a multiple of 16 bytes (`smallestPacketSize`), the same solve
takes 51 ns and 450 ns. The same substitution as plain index loops avoids the problem too and is
two to three times as slow in `double`, so the padded columns are what the library uses. That mismatch
of stores and loads is the likeliest reason Eigen's own small solve and factorization run at
half an instruction per cycle, since both work on blocks whose size and alignment are run-time
values; it is not confirmed.

After it, in `double`, a step of the library's EKF takes 114 ns at 2D with 4 anchors (167
before), 225 at 2D with 8 (547), 581 at 2D with 16 (882), 186 at 3D with 6 (412), 482 at 3D with
12 (720), and 1,357 at 3D with 24 (2,018), and the instructions fall from 4,397 to 3,833 at 2D
with 8, from 18,946 to 12,505 at 2D with 16, from 17,675 to 10,439 at 3D with 12, and from 45,099
to 30,752 at 3D with 24. The hand code's fall alike, so the ratios to it move little.

*The factorization, measured and not made (2026-11-23).* The probe also timed two Cholesky
factorizations written out. One that reduces each column by the columns before it over its whole
length, padded to whole packets, with what that leaves above the diagonal never read, beats
`LLT::compute()` at every size in both scalars: 13 ns in place of 58 for a 4×4 matrix in
`double`, 61 in place of 122 for 8×8, 34 in place of 50 for 6×6, 166 in place of 209 for 12×12,
245 in place of 312 for 16×16, and 453 in place of 624 for 24×24. With it and the substitution
the hand code's whole step took 49 ns at 2D with 4 anchors in the probe, where the runner's takes
98 with Eigen's factorization. Two things stand between that and the library. It reads the whole
of each column, and the library promises only a covariance's lower triangle
[borrowed-intermediates], so values above the diagonal that nothing wrote, a denormal among them,
would go through the arithmetic; copying the lower triangle into zeros first, an element at a
time, with the norm that `LLT` keeps for `rcond()`, took 37 ns for the 4×4, most of the gain
gone. And the library's factorizations are `Eigen::LLT` objects, which cannot be built from a
factor computed elsewhere without reaching their protected members. Both are design questions
for the maintainer: whether the innovation covariance may be required whole where it is
factorized, and whether the library keeps `Eigen::LLT` or a factor type of its own, which could
also keep the reciprocals of the diagonal that the substitution takes again.

mherb/kalman's EKF, on Eigen too, has a larger step of its own, between 4 anchors and 6: 1,218
instructions at 2D with 4 anchors, 11,455 at 3D with 6, and 17,467 at 2D with 8, which is why the
ratio to it shows none at 8 and 12. The same attribution places it in `S.inverse()` and in its
products. At 2D with 4 anchors everything is fixed-size code inline, and Eigen inverts a 4x4
matrix by a closed formula. From 5x5 Eigen inverts by `PartialPivLU`, whose inverse solves two
triangular systems by the blocked `triangular_solve_matrix`: at 2D with 8 anchors those two solves
take 7,916 instructions and the factorization 2,351, about 10,500 of the 17,467 with the calls
around them, and at 3D with 6 the inverse takes about 6,150 of 11,455. Its products, written with
`*`, go to the general matrix product with its packing from 8 anchors, about 5,400 at 2D with 8.

*Read from the code, not measured (2026-11-23).* Two more candidates for the fixed cost of a small
update, from a reading of the code that built nothing:

- Whether the calls Clang leaves out of line cost what the profile suggests is unconfirmed.
  `-Rpass-missed=inline` on `benchmarks/ranging/benchmark.cpp` would list them with the reason, and
  the instructions per step with `-mllvm -inline-threshold=2000`, or with `always_inline` put on
  `transposedSolve()` and the factorization for the measurement, would give the cost of the calls.
  What remains of the fixed cost after that is work the hand code skips: the check of the
  factorization, the `std::expected`, and the mirrors of the covariance.
- `kalmanConditionInPlace()` mirrors the covariance's lower triangle into the upper one before it
  subtracts $K \Sigma_{xy}^T$ from the whole matrix (`Models/Condition.h`). The library reads only
  the lower triangle [borrowed-intermediates], so the standard form could subtract into the lower
  triangle alone. Two things hold the mirror in place. `resetCovarianceAt()` reads the whole
  matrix, so a space without an identity reset, such as an attitude, needs the mirror before its
  reset or inside it. And `KalmanFilter.StepsInPlaceFromTheLowerTriangle` requires the posterior
  to be symmetric to 1e-14, which the contract does not promise; the test would compare lower
  triangles. The stored upper triangle would change, which a user who reads `covarianceLower()`
  as a whole matrix can see. The mirror is about six copies for a 4×4 matrix, so removing it
  does not close the gap at the small sizes by itself; it is to be measured before and after.

**OpenCV.** The Kalman filter of OpenCV, the computer-vision library (Apache 2.0 license), is
`cv::KalmanFilter` in its video module, a linear filter whose matrices are `cv::Mat`s sized at run
time. Its users run an EKF through it by setting its measurement matrix to the Jacobian at the
predicted mean after each prediction and passing `correct()` the pseudo-measurement
$z - h(x) + H x$, whose innovation is the EKF's; the motion here is linear, so its transition
matrix is set once. Its arithmetic is kept as its users get it: `predict()` forms $A x$ and
$A P A^T + Q$ and copies them into the posterior, and `correct()` solves $S K^T = H P$ by a singular
value decomposition of $S$ and updates the covariance as $P - K (H P)$, with no triangle mirrored.
OpenCV is too large to fetch and build for a benchmark, and its users install it, so it is found
rather than fetched, behind `EMRYS_FILTER_BENCHMARK_OPENCV`, off by default. The numbers here are
from OpenCV 4.14.0, built from source with its core, imgproc, and video modules; its `kalman.cpp`
is the same in 5.1.0.

The pair is `benchmarks/ranging/opencv_ranging.h`, against `emrys_ranging_kinds.h`'s EKF, with
`opencv_agreement.cpp`, `opencv_allocation.cpp`, and the runner `opencv_benchmark.cpp`. In double
the two agree to 1e-9 relative until they first part, between steps 400 and 504 of each case:
the asymmetry that rounding leaves in OpenCV's covariance grows, from 5e-13 to 1e-11 of its size at
step 100 to 2e-8 to 2e-7 at step 500 and 1e-3 to 7e-3 at step 1000. By step 2000 its spatial estimates
have diverged (in 3D with 6 anchors, a covariance of norm 7e7 where EmrysFilter's is near 0.2), and
its planar covariances are 7 to 40 percent off. Through 3000 steps EmrysFilter's double EKF stays
within 2e-15 of its own estimate in long double, and so does a copy of the OpenCV pair whose
covariance is mirrored after each update, so the drift is OpenCV's unmirrored covariance. The pair
is therefore timed over runs of 300 steps, each timed filter starting over from the initial
estimate, untimed, at the end of one, so that every timed step is one over which the pair agrees;
the runner prints the long run beside the timing. In float, the type `cv::KalmanFilter` has unless
it is constructed with `CV_64F`, the two do not agree even over those 300 steps: OpenCV's strays
from the double estimate by 2e-4 to 2e-3 relative in the mean and 4e-3 to 2e-2 in the covariance,
where EmrysFilter's float EKF stays within 3e-5. The cause is its solve for the gain by a singular
value decomposition of an innovation covariance that its unmirrored covariance has left
asymmetric: a copy of `correct()`'s arithmetic, written from its public matrices, reproduces its
estimate exactly with `DECOMP_SVD` and stays within 1e-5 at every size with `DECOMP_CHOLESKY`, and
OpenCV's own `correct()` stays within 4e-5 when `cv::completeSymm()` mirrors `errorCovPost` after
it. So the float pair is timed with a mark, behind a test in `opencv_agreement.cpp` that only
OpenCV's float EKF leaves the double estimate. With 8 anchors or more in double, and 12 or more in
float, OpenCV's steps make one heap allocation each, and none with 4 or 6, which `opencv_allocation.cpp` records without
requiring, since the OpenCV it builds against is whichever is installed; the timings of those cases
include its allocator, as its users' do.

## How the numbers are measured

Three tiers, by what carries across machines:

1. **Deterministic, in CI, gating.** Instruction counts from Callgrind, heap allocations per step,
   and `.text`, `.data`, and stack on Cortex-M. These are tests in the sense above: the
   allocation count is zero, the stack and sizes have ceilings, and the instruction count is a
   ratio to the hand-rolled filter with a ceiling of 1.10. The ceiling is a constant in the
   checking script with a comment saying why it is what it is. It leaves room for noise between
   compiler releases, and raising it takes a commit that says what was traded for what. Today's
   counts would fail it at 2D with 4 anchors, at 3D with 6, at 2D with 8 with autodiff in double,
   and on the attitude pair (sections 1 and 2), which is work for the library, not a reason to
   raise the ceiling. Absolute counts are uploaded as job artifacts so a trend is visible.
2. **Wall-clock, by hand, published.** The `benchmark` target under the `*-bench` presets, as
   today: median, p99, and worst per step and the EmrysFilter-to-hand ratio of each, interleaving
   the two implementations. Never in CI, where the machine is shared and the numbers mean nothing.
   Every published table carries the commit, compiler and version, CPU, operating system, date,
   and sample count. Re-measured at each release tag, not per pull request.
3. **Curves.** Scaling over anchor count and dimension (1), particle count and threads (3), and
   later the GPU crossover (8). Measured as tier 2, plotted once, published as a table and a
   figure.

**The first wall-clock measurement.** The pairs at a fixed step, October 31,
2026, on an Intel Core i9-13900F with no other work running, each runner pinned to one thread of
a performance core (logical CPU 2) at high priority, 2000 batches of each filter per case. Three
builds of Clang 21 at `-O3`: on Windows 11 with no `-march`, on Windows 11 with
`-march=x86-64-v3`, and on Linux (Ubuntu 26.04 under WSL2) with `-march=x86-64-v3`. Each entry is
the library's median time per step over the hand code's; below 1 the library is faster. The
instruction ratios are Callgrind counts of the same commit, taken later (November 2026, Linux
build with `-march=x86-64-v3`).

| Pair | Windows | Windows, v3 | Linux, v3 | Instructions |
|---|---|---|---|---|
| Ranging, 2D, 4 anchors | 1.11–1.44 | 1.08–1.41 | 1.20–1.49 | 1.40–1.56 |
| Ranging, 2D, 8 anchors | 1.01–1.41 | 1.05–1.16 | 1.06–1.15 | 1.07–1.10 |
| Ranging, 3D, 6 anchors | 1.07–1.20 | 1.08–1.12 | 1.10–1.13 | 1.15–1.26 |
| Ranging, 12 or more anchors | 0.89–0.96 | 0.88–1.00 | 0.90–1.05 | 0.87–0.99 |
| Attitude, analytic | 1.70 | 1.78 | 1.56 | 1.85 |
| Attitude, autodiff | 2.02 | 2.14 | 1.64 | 1.75 |
| Particles, 2D | 1.09–1.17 | 1.04–1.16 | 1.00–1.04 | 1.03–1.06 |
| Particles, 3D | 1.26–1.41 | 1.32–1.47 | 1.02–1.13 | 1.04–1.07 |

The p99 ratios follow the medians closely. Absolute medians on Windows with `-march=x86-64-v3`:
ranging from 160 ns per step (2D, 4 anchors, double) to 1,706 ns (3D, 24 anchors); attitude 527
against 296 ns analytic and 838 against 392 ns with autodiff; particles 41 to 47 ns per particle
and step in 2D and 73 to 85 ns in 3D, against the hand code's 40 to 41 and 55 to 59. On Linux the
particle filters of both sides take about three times as long (142 to 241 ns), most likely in
the standard library's normal draws, which differ between the platforms.

What the measurement shows:

- The instruction set does not explain the ratios: they barely move with `-march=x86-64-v3`. The
  1.41 of 2D with 8 anchors in double without it did not repeat (1.05 with it, 1.06 on Linux)
  and is most likely code placement.
- The attitude pair was the real gap, on both platforms. The counts of the same commit say why:
  under the fixed step the prediction differentiated the step again (section 2), which made the
  step 1.85 times the hand code's in instructions. Since that was fixed the gap is 1.15 to 1.32
  (the second measurement below). A first profile
  on Linux (CPU-clock samples; WSL has no hardware counters) has two calls the library does not
  inline at the top: the gain's solve, `internal::transposedSolve()`, at 10% of all samples, and
  the 3×3 Cholesky factorization, `llt_inplace::unblocked`, at 7.5%. The hand code inlines both
  into its update. The update's plumbing (`kalmanConditionInPlace()`, the linearization,
  `updateInto()`, and the `std::expected` return) is roughly another 15 to 20%. The samples are
  not yet divided between the two filters.
- The particle gap in 3D is particular to Windows: the library's extra time per particle and
  step is about 18 ns there and about 5 ns on Linux.
- From 12 anchors up the library is faster than the hand code, so the hand code does something
  worse there. Under the rule of section 1, a baseline that loses to the library understates the
  library's overhead, and that case is to be found.
- With `-march=x86-64-v3` some `float` cases are slower on both sides on Windows (ranging in 3D
  with 6 anchors, about 300 ns to 560 ns). It does not change the ratios.

**The second wall-clock measurement.** November 8, 2026, for ranging and
particles, and the same code with the fixed step's prediction by the block's Jacobian
(section 2) for attitude, which changes the instructions of no other pair, on the same machine,
pinned as before, with 2000 batches of each filter per case. Two builds: Clang 21.1.0 on Windows 11
with no `-march`, at high priority, and Clang 21.1.8 on Linux (Ubuntu 26.04 under WSL2) with
`-march=x86-64-v3`, at normal priority, since WSL refused a higher one. The instruction ratios are
the counts at the fixed step in sections 1 to 3.

| Pair | Windows | Linux, v3 | Instructions |
|---|---|---|---|
| Ranging, 2D, 4 anchors | 1.15–1.68 | 1.22–1.53 | 1.30–1.44 |
| Ranging, 2D, 8 anchors | 1.02–1.44 | 1.06–1.15 | 1.06–1.11 |
| Ranging, 3D, 6 anchors | 1.11–1.21 | 1.14–1.18 | 1.16–1.33 |
| Ranging, 12 or more anchors | 0.91–0.97 | 0.91–1.09 | 0.89–1.01 |
| Attitude, analytic | 1.15 | 1.21 | 1.18 |
| Attitude, autodiff | 1.32 | 1.29 | 1.27 |
| Particles, 2D | 1.10–1.18 | 1.00–1.04 | 1.03–1.06 |
| Particles, 3D | 1.25–1.40 | 1.02–1.12 | 1.04–1.07 |

Absolute medians on Windows: ranging from 104 ns per step (2D, 4 anchors, float, analytic) to
2,150 ns (3D, 24 anchors, double, autodiff); attitude 362 against 314 ns analytic and 495 against
375 ns with autodiff; particles 52 to 57 ns per particle and step in 2D and 81 to 93 ns in 3D,
against the hand code's 47 to 48 and 65 to 67.

- The ratios of the ranging and particle pairs are within a few hundredths of the first
  measurement's, except one: in 2D with 4 anchors, float, autodiff is 1.68 on Windows and 1.53 on
  Linux, where the first measurement's largest in that row was 1.44 and 1.49. Its instructions fell
  over the same commits, so the time is in how they run.
- The 1.41 of 2D with 8 anchors in double on Windows without `-march` repeated (1.44 analytic, 1.43
  autodiff), so code placement no longer explains it alone; on Linux the case is 1.06 to 1.07.
- The attitude pair now follows its instruction counts on both platforms, 1.15 to 1.32.

**The Callgrind job.** A Linux job on Clang, release with `-g`, that builds the benchmark runners
and runs each with `--count` under `valgrind --tool=callgrind --toggle-collect='*runSteps*'`,
then `tools/instruction_counts.py` reads `callgrind_annotate --inclusive=yes`'s output, matches
each `runSteps` instantiation to its case and implementation by its demangled name, prints the
table, writes it as JSON, and fails the job if any case's ratio exceeds the ceiling. An
instantiation appears once per source file whose code was inlined into it, and the line under
`harness.h`, where `runSteps` is defined, already holds the whole inclusive count: the script reads
that line only, since summing the lines counts the inlined code twice. Valgrind runs no AVX-512 and
the job must build for an instruction set it executes (`-march=x86-64-v3` at most), which the
preset for the job sets. Because the job builds every runner, it also keeps them compiling,
which nothing does today unless someone configures a `*-bench` preset.

**The harness.** Our own, for the reason the roadmap records: neither Google Benchmark nor
nanobench reports a per-sample p99 and worst case or interleaves two implementations' samples.
It grows just enough that a new problem writes only its scenario and its two implementations.
Today `benchmarks/harness.h` has the shared command line (`--samples`, `--count`, and
`--particles` for a runner with a count axis), the compiler line, `Replay`, which runs a filter
over a simulated run a batch of steps at a time and restarts it at the end, `timePair()`, which
sizes the batch and times two replays interleaved, and the table's columns and rows. A runner
writes its agreement pass itself, since what agreement means differs by problem, and the row's
label. Still to come are `--json` and `--markdown`; the `--markdown` output is what gets pasted
into the results page, so a published table is never retyped.

**A problem's directory.** Each problem keeps the ranging layout: `scenario.h` (the problem and a
plain-Eigen simulator, so neither filter produces its own test data; a problem that reuses
another's includes it and changes what it must), `emrys_<problem>.h`, `eigen_<problem>.h`,
`cases.h`, `agreement.cpp` (agreement and NEES), `allocation.cpp`, and `benchmark.cpp`. The
agreement and allocation tests build with the unit tests and run in every CI build; the
allocation tests count the heap through `benchmarks/heap_counter.cpp`, since a program replaces
`operator new` once; the runner builds in the Callgrind job and under the `*-bench` presets.

## Publication

The numbers go where a visitor finds them: [docs/benchmarks.md](../benchmarks.md), a Markdown page
directly under `docs/`, which no CI job reads and Doxygen does not take. It opens with a summary
table of every pair's ratios, then the method as a short list (what the hand-rolled side shares
with the library, what agreement is required, how each number is taken) and how to run the
benchmarks on the reader's own machine, since a ratio on ours is an invitation to check. Then one
section per measured claim with its table, a list of what is not measured yet, and the machine
lines. Its text stops at the method; everything here behind it (why the reference is hand code,
the ceiling, the history of each ratio) stays here, and the page links this plan for it. A table
comes from `--markdown` once the runners have it, so a published table is never retyped.

The README gets a "Performance" paragraph of two or three numbers (the instruction ratio, the
particle-filter thread scaling, the Cortex-M footprint) and a link to the page, and
`docs/reference/mainpage.md` links it from the Doxygen front page. The design doc stays here and
the results stay there; the results page does not explain the plan and this plan quotes no
numbers that will go stale.

## The tests side: what ship-shape means

The tests are already the map of the evidence, and the gaps in that evidence are already
Verification items. This plan changes none of their order and only says what kind of thing each
is, so that none of them is mistaken for a benchmark:

- **Warnings as errors** on all three toolchains: a build property, checked by the build.
- **Coverage [coverage]:** instantiation coverage against the Doxygen entity list, region coverage
  per instantiation, and the axis checklist (plain and manifold spaces, fixed and dynamic sizes,
  `float`, `double`, and autodiff, default and custom allocator) enforced by typed suites. This is
  the one item that may add many tests, and should: a typed suite over an axis is cheap to write and
  is exactly what a header-only library's correctness rests on.
- **No exceptions [embedded-builds]:** a build configuration; credible once instantiation coverage
  holds.
- **Sanitizers:** the ASan and UBSan job CI runs (`clang-sanitize`), and the desktop
  evidence for the Cortex-M target that has none.
- **Cortex-M [embedded-builds]:** a compile-and-QEMU job. Its sizes and stack numbers feed benchmark
  6, but the job itself is a test: the subset runs or it does not.
- **Negative compile tests** for each compile-time error the roadmap claims.
- **Allocation tests:** complete them for the mixture of hybrids that step 15c left out. (The
  resamplers are covered since 2026-10-31, when `resamplers::Multinomial` and `resamplers::Residual`
  began taking the set's scratch.)

Two rules for new tests follow from the benchmark rule. Every behavior gets a test, so the test
count grows with the library and that is right. And because the allocation tests and the
instruction-count ceiling make performance regressions test failures, a new feature never needs
a benchmark to be kept honest; it needs a test, and the allocation target already has a file per
filter family to add to. The test build's own cost is the "Shorter test builds" item under
the roadmap's Launch and is not this plan's concern.

## Steps

In the order each supports the next. Each is a pull request or two.

1. **The Callgrind job and the ceiling.** A `linux-callgrind` CI job, `tools/instruction_counts.py`,
   `--json` and `--markdown` in the runner, and the ceiling at 1.10. The pair driver and shared
   command line move into `benchmarks/harness.h` so that step 4 writes no runner logic.
   *Done when:* the job passes on `dev`, fails on a branch that deliberately adds a second LDLT to
   the update, and uploads the counts as an artifact.
2. **Publish ranging.** `docs/benchmarks.md` with the method and the ranging tables from all
   three compilers, the README paragraph, the mainpage link, and the Compiler Explorer links or
   the disassembly artifact. The constant-acceleration case joins first so the published table
   has the state-dimension axis. *Started:* the page holds the tables measured in October and
   November 2026, whose commits were not recorded, and the README links it.
   *Done when:* a visitor reaches the method and the numbers from the README in two clicks, and
   the page says how to reproduce them.
3. **Attitude (step 13e).** `benchmarks/attitude/` in the ranging layout, the multiplicative EKF
   from Markley (2003) as the hand-rolled side, agreement to 1e-9 over 1000 steps, NEES over 50
   runs, no allocation in either step, and the Callgrind ceiling. *Started:* the pair and its
   tests are built and measured (section 2); the ratio, 1.18 with analytic measurement Jacobians
   and 1.27 with autodiff, is over the ceiling because of the updates' plumbing.
   *Done when:* step 13e's checklist closes and the attitude table is on the results page.
4. **Particles and threads (step 18).** The particle filter and its hand-rolled loop on the
   ranging problem, agreement particle by particle, and the scaling runner over particle count
   and thread count. *Started:* the pair, its tests, and the `Sequential` rows are built
   (section 3); the thread sweep is not.
   *Done when:* the `Sequential` overhead ratio is published and under the ceiling, and the
   scaling curve from MSVC and Linux with TBB is on the results page as step 18's "done when"
   asks.
5. **The within-library tables.** The options and filters tables on the ranging problem, each
   configuration an instantiation of the pair driver against the plain EKF. The row for a model
   given `dt` at every step closes step 13d's leftover. *Done when:* both tables are published and the borrow-stamp work
   has its before measurement.
6. **Footprint (after [embedded-builds]'s Cortex-M job).** The sizes and stack table from the job's
   artifacts, and DWT cycles from one M4F and one M7 board by hand. *Done when:* the table is
   published with the boards and clocks.
7. **Compile cost.** The per-translation-unit time and memory on all three compilers, in CI as
   artifacts and on the results page as a table. *Started:* the units, `tools/compile_cost.py`,
   and the first table are built (section 7); the CI job, Linux's numbers, and the `Core.h` cost
   are not. *Done when:* published, and the numbers are quoted in the contributor notes on build
   cost.

Step 8 of the table, the GPU crossover, waits for step 19 and is specified there.

## Open questions

- Is the ceiling one number for every case, or does the smallest case (2D with 4 anchors, where
  the fixed overhead of about 90 instructions is largest in proportion) get its own? One number
  is simpler to explain; the roadmap's note on what the 90 instructions are is the argument for
  a per-case ceiling.
- Should the results page keep every release's numbers, or only the latest with a link to the
  release that measured it? The latter keeps the page short; the former shows a trend without
  anyone keeping a database.
- Does Compiler Explorer carry Eigen 5? If not, the disassembly artifact is the fallback, and
  the question is how much of it to quote.
- Which boards for the DWT cycle counts? One M4F and one M7 that are common enough for a reader
  to own, with the clock stated.
