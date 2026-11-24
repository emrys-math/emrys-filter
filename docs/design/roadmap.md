# Roadmap

The work left after 0.1.0, the first release, in order. It is written for contributors;
what the library does and why is in [decisions.md](decisions.md), how code is written in
[style.md](style.md), and open design questions in the reviews in `docs/guide/`. The plan behind
steps 18 and 19 is [parallelism.md](parallelism.md), and the plan for the benchmarks is
[benchmarks.md](benchmarks.md).

How it is kept:

- It lists only work not done. A step that is finished becomes a row of the Done table at the end,
  in the commit that finishes it, and its section goes; an item that is finished in an open list is
  deleted. What happened belongs in the commit message, and what a later session must know goes to
  a decision, a header's documentation, or a guide's question.
- An item says what to do, why in a sentence where it isn't plain, and the test or evidence that
  finishes it. A step ends with "Done when".
- Step numbers are never reused, since the headers, tests, and documents cite them; a new step
  takes the next free letter after the step it follows.
- Status is **not started**, **in progress**, **done**, or **deferred**.
- Users don't read this file. Every change a user can see goes into `CHANGELOG.md` in the commit
  that makes it, and the usage page lists what is not built yet.

The advertised launch (Launch, below) waits for every step in this file but step 19 and what
"Deferred past the advertised launch" lists; for every design review finished and every guide
question answered, breaking or not; and for every algorithm checked against its source, each source
read in full (decided 2026-11-16). With no date to meet, a feature that may want a change to the
existing API is built before the launch rather than after it.

## 17t. Sigma-point axes and repairs — in progress

[sigma-point-axes] records the design. The second differences, which use `along(d)`, are 16b's.
`Distributions/Roots.h` holds the axes, `ridge`, the fallback list, `squareRoot()`, and
`squareRootWithReport()`; `methods::spread()` carries a method's axes to `Transform::root()`,
the factored transform's rest [affine-coordinates], and `Iterated`; and every other root goes through
`internal::defaultRoot()`. Under the default the forms
equal the Kalman filter to rounding (`sqrt_filters.cpp`, `info_filter.cpp`, `iterated_forms.cpp`).

An update takes axes that can fail: `transformMeasurement()` takes their root of the blocks read
first and gives the transform that root, and `tests/Models/spread.cpp` checks that
`spread(cubature(), cholesky())` fails the Kalman update on a singular prior, plain, iterated, and
linearized about it, and equals `cubature()` on a positive definite one in every form. Left: the
`pr` checks.

Done when: every root the library takes goes through `squareRoot()`, the forms agree under every
rule, and the `pr` checks pass.

## 17v. Consider states — in progress

[consider-states] records the design: `observe(method, model, z, consider(path<Bias>))`, carried
out by `kalmanConditionInPlace()` and the iterated update, and refused by the square-root and
information forms, the filters that weight points, and the updates given a block.
`tests/Models/consider.cpp`, `tests/Filters/consider_consistency.cpp`, and `example.consider_bias`
are the evidence.

- [ ] Check the bibliography's `Schmidt1966` entry, and what `Consider`'s documentation takes from
      it, against the paper itself (Advances in Control Systems, volume 3), which this session
      could not read.

Done when: the entry is checked and the `land` checks pass, with MSVC.

## 17g. Posterior Cramér-Rao bound — in progress

[cramer-rao-bound]: the recursion and its tests are done.

- [ ] A singular $Q$ (a noiseless block): Section IV of Tichavský, Muravchik, and Nehorai (1998),
      decided with 17zh's singular $Q$.
- [ ] An example of the bound on the Models module page, beside simulate() and nees().

Done when: the singular case is decided and passes its tests on all three compilers.

## 18c. Batched functions — in progress

[batched-functions]: a function marked `batched(f)` takes every point at once and writes one output
`Y`, with Jacobians by block when its mark gives them, and the library calls it once per batch. It
takes each of the function forms of [model-policies], `h(in, out, X, Y)` to `h(X, Y)` and a process
model's with `dt` before `Y`, and a whole-state model's writes its Jacobian whole, `Y.jacobian(j)`
[whole-state-models].

Every call site takes a batched function and calls it once per batch: the particle filter's
predict and update and the Rao-Blackwellized reweighting once per set or chunk; the sigma-point
transforms, the factored one, the iterated posterior linearization filter, the update given a block
(`PinnedMeasurement`), and the Rao-Blackwellized prediction of hybrid particles once per point set,
through `Transform::evaluateEach()` and the function objects the resolvers give; the likelihood
widened by uncertain inputs once per set or chunk; `on()`, `compose()` part by part, and `onEach()`
once per repetition. `Linearized` and the iterated updates call `Batched::operator()`, a batch of
one column, which is their final form. Where a method needs the Jacobian at every point,
`internal::callBatchedWithJacobians()` takes it from the function's mark or by dual numbers over
the batch. `tests/Models/batched.cpp` counts the calls and matches each against the per-point form;
`tests/parallel.cpp` and `tests/allocation_particles.cpp` have their cases, and
`BATCHED_NOT_GENERIC` the negative compile test.

- [ ] The dual numbers over a batch come from the heap where the batch's columns are a run-time
      count, as the likelihood widened by uncertain inputs has them (its chunks): keep them in
      scratch of the filter's, rebound to the dual scalar, and add the case to
      `tests/allocation_particles.cpp`, with the Rao-Blackwellized reweighting's
      `PointLikelihoods::batchData`.
- [ ] Benchmark: the ranging particle filter, per point and batched, at $10^4$ and $10^6$
      particles. The runner has the `batch` likelihood and the $10^6$ count. A first run at $10^4$
      (November 12, Windows, Clang 21, no `-march`, pinned to one core) put the batched row within
      2% of the noise row, as expected of a function with no cost per call, but every row at 2.1 to
      2.6 times the hand loop, against 1.09 to 1.41 on November 8 (benchmarks.md). The commit
      before this step measures the same, and a release probe at $10^4$ found no heap allocation:
      the time is in `predict()`, 3.1 to 3.2 times the hand loop's, with `update()` and
      `housekeep()` near it. Bisect `a7778c4..244c802` for the prediction's slowdown before the
      rows are published; the $10^6$ run was not taken.

Done when: every call site above takes a batched function, the tests pass on all three compilers,
and the benchmark row is published. A chunk size other than 1024, if a user needs one, would be a
policy adaptor such as `chunked(std::execution::par, 4096)`, never a build flag.

## 18. Execution policies — in progress

[execution-policies]; the loops and the reasoning are in parallelism.md. Every loop takes a policy.

- [ ] Benchmark [benchmarks]: thread scaling of the ranging particle filter
      (`benchmarks/particles/`) at $10^3$ to $10^6$ particles, as the ratio to `Sequential`. Under
      `Sequential` it is 1.04 to 1.07 times the hand loop's instructions.
- [ ] `Gaussian::logDensity()` recognizing a diagonal or scalar covariance: with an isotropic $R$,
      the triangular solve costs 45 instructions per particle more than a closed form.

Done when: the scaling curve is published.

## 18b. Counter-based random numbers — deferred

Deferred past the advertised launch (decided 2026-11-21). The step need not bring a generator of
its own: C++26 standardizes `std::philox4x32`, so the work waits for a standard library that has it.

Parallel draws need a generator with no shared state: Philox (Salmon et al. 2011), standardized in
C++26 as `std::philox4x32` (parallelism.md, "Random numbers in parallel"). The engine stays the
caller's and the draws go through the standard distributions, so they are reproducible across
thread counts but not across standard libraries.

- [ ] `Core/Random.h`: a `CounterBasedGenerator` concept (an independent stream at a step and an
      index), which `std::philox4x32` meets where the library has it.
- [ ] Parallel overloads of `predict()`, `Roughen`, `drawSamples()`, and the Monte Carlo
      divergences that draw inside the loop from such a generator; without one they keep drawing
      before the loop.
- [ ] `MetropolisResampler` and `RejectionResampler` (Murray, Lee, and Jacob 2016), which need no
      cumulative sum, and the systematic and stratified resamplers by scan and binary search.
- [ ] Bibliography entries for Salmon et al. (2011) and Murray, Lee, and Jacob (2016).
- [ ] Tests: a parallel prediction equals the sequential one bitwise at any thread count; the new
      resamplers' counts match the weights in expectation.

Done when: parallel draws are bitwise reproducible across thread counts within one standard
library, and the six resamplers pass their tests.

## 17zu. Selections of a repeated block — in progress

A model, a placement, and an accessor name part of a repeated block: one repetition, a run of
repetitions, or a list of them. Today an item of `reads()` is a path, a path is a type, and a model's
reads are resolved from those types and the space alone, so a model that uses one landmark of
`Landmarks` reads the whole block. A sigma-point method then spreads its points over every landmark.
`affineIn()` does not replace the selection: it names whole items, so it is right only where the
function is affine in the repetition it uses, and it still forms the Jacobian and $P_{a|n}$ over
every repetition. Designed on 2026-11-23, its spellings decided that day; the items below are what
is left to build. Every place that names a block takes a selection in this one step, so that no
spelling is left to be found incompatible later.

**The selection** is the subscript of a path value that runs through a repeated block. A repeated
block cannot hold another, so a path takes one subscript and it is never ambiguous. The path may
reach the repeated block through other tags (`path<Map, Landmarks>[k]`) and may go on inside it
(`path<Fleet, Heading>[k]`, vehicle `k`'s heading).

| Path value | Names | Its space |
|---|---|---|
| `path<Landmarks>[k]` | repetition `k` | the repetition's, `Landmark`, of fixed size |
| `path<Landmarks>[Eigen::seqN(first, n)]` | `n` repetitions from `first` | `Landmarks` with count `n` |
| `path<Landmarks>[seen]`, `path<Landmarks>[{2, 5, 9}]` | the repetitions listed, in that order | `Landmarks` with the list's count |

An integer names the block that is repeated. A sequence or a list names a smaller repeated block,
even of one repetition, so a function written for the whole block runs unchanged on part of it. The
selection is held by value in the model, and a model over a selection that changes is built per
measurement, as one whose geometry changes already is (usage.md's `tdoa(a1, a2)`).

```cpp
// range and bearing to landmark k: 2 * 5 + 1 sigma points, whatever the count of landmarks
Emrys::measurementModel(
    sighting,  // reads block<Landmark>(in, x)
    Emrys::reads(Emrys::path<Position>, Emrys::path<Heading>, Emrys::path<Landmarks>[k]),
    Emrys::outputSpace(RangeBearing{}),
    Emrys::additiveNoise(noise));

// every landmark sighted in one scan; scan loops over block<Landmarks>(in, x, i)
Emrys::measurementModel(
    scan,
    Emrys::reads(Emrys::path<Pose>, Emrys::path<Landmarks>[seen]),
    Emrys::outputSpace(RangeBearings{RangeBearing{}, std::ssize(seen)}),
    Emrys::additiveNoise(Emrys::onEach<RangeBearings>(r)));
```

The rules, of which [repetition-selections] holds those built, the rest for it or [model-policies]:

- **Name.** A selected path is read or written under its last tag, and under the repetition's tag
  where the path ends at the repeated block and the subscript is an integer. A space without a tag
  (a repetition declared as `Linear<2>`, or the `Repeated<Heading>` of `path<Fleet, Heading>[seen]`)
  needs `on<Name>(...)`. Two selections of one block take two names, as two paths that end in one
  tag do today.
- **Bound.** A sequence's or a list's space has the bound of the state's block, which sizes fixed
  storage for every landmark where three are read. `on<Sighted>(path<Landmarks>[seen])` with
  `struct Sighted : Repeated<Landmark, 8> {}` reads it under a smaller bound, and the count is
  asserted to fit: the name check accepts a repeated tag of the same repetition and another bound.
- **Index in the function.** `block<Landmarks>(in, x, i)` is the `i`th repetition read, not the
  state's `i`th.
- **Sequences.** Eigen's `seq` and `seqN`. An increment of 1 is one run of rows; any other
  increment is treated as a list. A sequence may count from the block's end, as
  `Eigen::placeholders::lastN(3)` does, and is resolved against the count of the state it meets:
  `Indexing.h` already resolved Eigen's symbols by indexing a vector, which needs none of Eigen's
  internals. `Eigen::placeholders::all` and a bare `last` have no overload. Eigen's types were
  chosen over a `range(first, n)` of the library's own: users know them, and `idx()` returns them.
- **Lists.** Any range of integers, copied into the path value. A braced list or a `std::array` is
  stored at fixed size, so a model that holds one allocates nothing. A selection names each
  repetition once, which bounds a list by the block's `MaxCount`.
- **Checks.** At compile time: a subscript on a path through no repeated block, and a selection
  with no name. Asserted when the model meets a state: every index within the block's count, no
  coordinate named twice among a model's reads or among its writes, and equal selections wherever
  parts of a composition share a name.
- **Counts.** `CountFrom<Landmarks, ...>` keeps counting the state's block. A model over a sequence
  or a list gives its output space sized, since a count is never inferred from what a model reads
  [several-resizable-blocks].

Where a path is a value it takes the subscript. Where a path is a template argument and the call
takes an integer repetition today, that argument takes a sequence or a list too.

- [ ] **`reads()`, the rest.** A plain model reads a selection ([selected-reads],
      `tests/Models/selected_reads.cpp`), with a returned Jacobian and for the likelihoods of
      sampled states, and through each Gaussian form of the Kalman filter, `GaussianMixtureFilter`,
      `ParticleFilter`, and both Rao-Blackwellized filters (`tests/Filters/selected_filters.cpp`),
      and an update that reads one repetition of a bounded block allocates nothing
      (`tests/allocation.cpp`). A
      `SelectedPath` (`Spaces/TagPath.h`) is not a `PathValue`, so that no policy takes one and
      drops its selection: each item here makes its policy take it. `on<Path>()` and `compose()`
      carry the selections of the models they take ([composed-selections],
      `tests/Models/selected_placement.cpp`, which also has process models that read a selection,
      as parts of a composition).
- [ ] **Parts that read a block and one of its repetitions.** Decided by the user on
      2026-11-23: the rule stays in this step, and overlapping reads may come later. The blocks a
      model reads are disjoint: `ReadBlocks` asserts "a
      model must not read a coordinate twice", since the joint of what is read is a marginal of
      the state. A composition's reads are the union of its parts' by name, so
      `compose(steerToward, Constant<Landmarks>{})`, where `steerToward` reads
      `path<Landmarks>[k]` under `Landmark` and `Constant` reads all of `path<Landmarks>`, reads
      landmark `k` twice and asserts. So does any part that reads one repetition beside a part
      that carries the whole block; a process model can instead carry the other repetitions with
      a part that writes their list (`tests/Models/selected_writes.cpp`). Two selections of one
      block that overlap assert too. To allow them, the joint would hold each coordinate once and an item's rows of the
      input would be gathered from it, the input then having more coordinates than the joint,
      and a Jacobian's columns would be summed into the joint's: every method that forms the
      joint (`FormSteps.h`, `IteratedUpdate.h`, `Likelihood.h`, `Measurement.h`,
      `MeasurementReads.h`, `ProcessModel.h`) would change. Until then `writes()` of a selection is
      the way to split the block; writes must stay distinct in any case. The item closes when the
      step does, with the assertion documented in usage.md.
- [ ] **A selection outside the block a model is placed on.** A question for the user. In
      `on<Vehicle>(model)`, an item of the model's `reads()` that the block `Vehicle` lacks is an
      input of the call under its name, so `path<Landmarks>[k]` there would lose its selection,
      and `on<Path>()` refuses it at compile time (`Placement::OuterItemOf`). The other choice
      keeps such an item as it is, path and selection, read from the state the placed model
      meets. It was not built, since an item without a selection would then still become an
      input by name while one with a selection kept its path.
- [ ] **`consider()` and `given()`, the rest.** Both take selections
      ([placed-selected-values], `tests/Models/consider.cpp`). Left: `ImmRecursion::update(state,
      measurement, given)` takes a `GivenBlock` of a whole block, so a selected one is no match
      there; give it a `static_assert` that refuses one, with its negative case, or carry the
      selection through `PinnedMeasurement` as `update(prior, measurement, given)` does. The
      Rao-Blackwellized filters' blocks refuse a path into the repetitions (the audit), so they
      never give a selection.
Done when: every item's test and every negative compile case passes on all three compilers, and the
audit's list has no function without a test or a refusal.

## 19b. A public API ready for the GPU backend — not started

Step 19 is deferred past the advertised launch (decided 2026-11-16), and must then arrive as an
addition: nothing a user writes against 0.x should change when a `Device` policy joins. What it
would otherwise break is found and changed now, with no GPU needed.

- [ ] The device-only syntax check, the first experiment of parallelism.md's "What to find out
      first": one `ParticleFilter` and one `RbParticleFilter` instantiation compiled with
      `clang++ -x cuda --cuda-device-only -fsyntax-only -std=c++23` after a first pass of
      `EMRYS_FILTER_DEVICE` over what a particle's body reaches. An obstacle in a public signature
      or a user's contract is changed before the launch; one inside a function body goes into
      step 19's list.
- [ ] What a device-callable model is, on the user page and in the model contracts: no
      `std::function`, no heap, nothing captured by reference to host memory, and a lambda that
      clang or nvcc's `--extended-lambda` compiles. The built-in models and the examples' models
      meet it.
- [ ] The accessors of the count-sized containers, such as `ParticleSet`'s `Eigen::Map`s, under an
      allocator whose memory the host cannot read: decide now what they become, so that
      `BufferFor<Allocator, T>` and an explicit copy to the host add a type rather than change an
      accessor (parallelism.md, "Memory").
- [ ] The draws: check that the generator signature step 18b settles is the one a device path
      takes, with the normal draws made on the device (parallelism.md's open question on where
      they come from).
- [ ] parallelism.md's open questions that reach a public signature decided: `Parallel{minimumCount}`,
      and who owns the step count that a counter-based generator advances.

Done when: the device-only syntax check passes with only function bodies left to change, those
listed in step 19, and parallelism.md's open questions that reach the API are decided.

## 17zf. Invariant EKF and pose spaces — not started

Barrau and Bonnabel (2017): an EKF on a Lie group whose error is the group error, which converges
from large initial errors on group-affine dynamics ($SE(3)$, $SE_2(3)$). It is `KalmanFilter` with
`Linearized` on a group space whose `plus()` and `minus()` apply the increment on a chosen side, so
it is spaces, an example, and tests, not a filter. The groups go in `Spaces/` [lie-groups]. Its
largest user base is legged robots, by contact-aided estimation (Hartley et al. 2020); the
mechanization and contact models stay EmrysNavigation's.

- [ ] The side of the increment on `UnitQuaternion`: right (the body frame, today's) by default,
      left as the alternative, each with its reset Jacobian [covariance-reset].
- [ ] $SE(2)$, $SE(3)$, and $SE_2(3)$ as spaces with both sides, their exponential and logarithm
      smooth at zero, and closed-form reset Jacobians (Solà, Deray, and Atchuthan 2018). Test:
      each reset Jacobian equals autodiff's, and the exponential and logarithm invert each other
      near zero and at large angles.
- [ ] The typed view. [unit-quaternion] rejected a generic `view(space, block)` until a second
      space wanted a typed form, and a pose is that space. Decide between it and a `map()` per
      space, and whether a `LieGroup` concept (exponential, logarithm, adjoint) stands beside
      `Space`, before the launch, since either could rename `UnitQuaternion::map()`.
- [ ] Tests: the error Jacobian is constant on group-affine dynamics; the invariant filter
      converges from a large initial heading error where the EKF on the same model does not.
- [ ] An example: an IMU with position fixes on $SE_2(3)$, with gyro and accelerometer biases
      beside it (the "imperfect" invariant filter, whose biases break the invariance), its
      mechanization written in the example as `attitude.cpp` writes its own.

Done when: the view is decided, and the spaces, the example, and the tests pass on all three
compilers.

## 17zg. Adaptive noise and robust updates — not started

Noise covariances estimated inside the filter as it runs, and updates that tolerate heavy-tailed
noise, which share the variational Bayes machinery. Estimating noise over a recorded run stays the
analysis project's [analysis-project], and Sage-Husa in an INS/GNSS example EmrysNavigation's.

- [ ] Where an adapted $Q$ lives, decided first, since it changes the existing API. A filter holds
      its process model as `const`, and $Q$ is a policy inside it (`additiveNoise()`,
      `continuousNoise()`), so adapting it needs a noise policy read through a handle, a mutable
      `model()`, or a $Q$ given to `predict()`. An adapted $R$ needs no change: the measurement
      model is given to each `observe()`, so an object the caller holds carries its estimate
      across updates.
- [ ] The variational Bayes estimate of $R$: diagonal by inverse-gamma (Särkkä and Nummenmaa
      2009), and full by inverse-Wishart (Särkkä and Hartikainen 2013), each iterating the update
      on the machinery of [iterated-update].
- [ ] $Q$ and $R$ together by variational Bayes (Huang et al. 2018).
- [ ] Sage-Husa (1969) with fading memory, and a guard that keeps its estimates positive definite
      against its known divergence.
- [ ] The Student-t update by variational Bayes (Piché, Särkkä, and Hartikainen 2012; Roth, Özkan,
      and Gustafsson 2013), and the Huber M-estimator update as an iterated reweighting of $R$,
      its source chosen when it is written. Particle and grid filters take Student-t noise already,
      through a measurement's own `logLikelihood()`.
- [ ] Which Gaussian forms take each: the covariance form first, the square-root and information
      forms refused with a message until written, as consider states are [consider-states].
- [ ] Tests: with the true noise the estimate converges to it and the NEES is consistent; started
      from a wrong $R$ or $Q$ the adaptive filter recovers where the fixed one stays inconsistent;
      Sage-Husa's guard holds where the unguarded recursion goes indefinite; the Student-t update
      moves less than the Kalman update under an outlier, and approaches it as the degrees of
      freedom grow; `tests/allocation.cpp`.

Done when: the decision on $Q$ is recorded, and the estimators and robust updates pass their tests
on all three compilers.

## 17zh. Transition densities — not started

A process model's $\log p(x_{k+1} \mid x_k)$, the prerequisite of the particle proposals and moves
of 17zi and of the particle smoother of 14b. For additive noise it is the Gaussian density of
$x_{k+1} \boxminus f(x_k)$ under $Q$ [tangent-gaussians].

- [ ] A singular $Q$, the usual case (noise that drives the velocity and not the position), makes
      the density degenerate: define it on the subspace the noise reaches, with the rest a
      constraint, decided together with 17g's singular $Q$.
- [ ] A model with an uncertain input through its function [prediction-inputs] has no closed-form
      density: refuse it with a message, or let the model give its own, decided when written.
- [ ] Tests: on a linear Gaussian model it equals the closed form; with a singular $Q$ it is finite
      exactly on the reachable set; on a unit quaternion block it uses the tangent density.

Done when: the density passes its tests on all three compilers.

## 17zi. Particle proposals and moves — not started

Proposals that use the measurement, and moves that rejuvenate a resampled set. Particle flow is
deferred past the advertised launch.

- [ ] The shape of a step that proposes from the measurement, decided first, since it changes the
      filter's API. Today `predict(rng, dt)` and `update(z)` are separate calls, and every proposal
      below needs $z$ while it predicts: a fused step, a proposal given to `predict()`, or another
      shape, and what several measurements at one time do. Decide it with particle-filter.md Q11
      (a recursion for the particle filter).
- [ ] The locally optimal proposal (Doucet, Godsill, and Andrieu 2000): exact for a measurement
      linear in the state with additive Gaussian noise, by the per-point Kalman update the
      Rao-Blackwellized filters already have; linearized (the EKF proposal) or unscented (van der
      Merwe et al. 2000) otherwise.
- [ ] The auxiliary particle filter (Pitt and Shephard 1999).
- [ ] Resample-move with a Metropolis-Hastings move (Gilks and Berzuini 2001), on 17zh's density.
- [ ] An adaptive particle count by KLD-sampling (Fox 2003), within the set's reserve [allocation].
- [ ] Tests: on a linear Gaussian model each proposal's estimate approaches `KalmanFilter`'s as the
      count grows, and the optimal proposal's weights vary less than the bootstrap's; with a
      precise measurement the optimal and auxiliary filters keep an effective sample size where
      the bootstrap filter collapses; resample-move leaves the posterior invariant; KLD-sampling's
      count follows the spread; cases in `tests/parallel.cpp` and `tests/allocation_particles.cpp`.

Done when: the step's shape is decided, and the proposals and moves pass their tests on all three
compilers.

## 17d. Ensemble Kalman filter — not started

Evensen (1994), with perturbed observations (Burgers, van Leeuwen, and Evensen 1998). `ParticleSet`
holds the members and `predict()` already moves them, and a batched measurement function
[batched-functions] is the observation operator over the whole ensemble. It is supported for
whoever chooses it over a sigma-point filter, at any state size the library serves, so a
dynamic-size state of a few thousand coordinates must perform; the operational sizes of weather and
ocean assimilation are not the aim.

- [ ] The members' container: `ParticleSet` with equal weights, or a type of its own, since an
      ensemble is unweighted.
- [ ] The stochastic update, with an execution policy and a case in `tests/parallel.cpp`.
- [ ] The deterministic updates as methods: the ensemble transform filter (Bishop, Etherton, and
      Majumdar 2001) and the ensemble adjustment filter (Anderson 2001).
- [ ] Inflation, multiplicative and adaptive (Anderson 2007; Anderson 2009).
- [ ] Localization by a Schur product with a distance taper (Houtekamer and Mitchell 2001; the
      taper of Gaspari and Cohn 1999). How a space says which coordinates are near which, and a
      measurement near which coordinates, is the one new decision: a taper over coordinates by
      index runs against [tag-addressing], so decide whether the user's taper takes tag paths.
- [ ] Tests: at large $N$ the moments match `KalmanFilter` on an affine model; localization
      recovers the Kalman covariance on a banded problem at small $N$; `tests/allocation.cpp` on a
      bounded space; the time of a dynamic-size run of a few thousand coordinates, recorded.

Done when: the filter, both deterministic updates, inflation, and localization pass on all three
compilers.

## 17zj. Correlated noise — not started

- [ ] Process and measurement noise correlated with each other, $E[w_k v_k^T] = S$ (Simon 2006,
      §7.1), which changes the gain. How a model pair declares $S$ is the decision. Test: equals the
      Kalman filter on the state augmented with the noise, built in plain Eigen.
- [ ] Measurement noise correlated in time, by measurement differencing (Bryson and Henrikson
      1968; Simon 2006, §7.2), which works where augmenting the state with the noise leaves $R$
      singular. Test: with a white part, equals the augmented filter; without one, stays
      consistent where the augmented filter cannot run.

Done when: both pass their tests on all three compilers.

## 17zk. UD factorization — not started

The covariance as $U D U^T$, the form of much aerospace flight software and a well-conditioned form
in `float`: the time update by modified weighted Gram-Schmidt (Thornton 1976; Thornton and Bierman
1977) and the measurement update by Bierman's scalar updates (Bierman 1977), which `FormSteps.h`
already cites for the square-root information form. A fifth form beside the four of
[gaussian-forms], read alike through its density [form-densities].

- [ ] `UdGaussianParams` and a filter over it, with conversions to and from the other forms.
- [ ] The methods it takes: affine and linearized first; a sigma-point method through a conversion,
      or refused, decided when written.
- [ ] Tests: equals `KalmanFilter` to rounding on an affine model; in `float` on an ill-conditioned
      problem it stays positive definite where the covariance form does not, beside
      `SqrtKalmanFilter`; `tests/allocation.cpp`.

Done when: the form passes its tests on all three compilers, or a decision records why the
square-root form serves its users instead.

## 17zl. Unknown inputs — not started

The state estimated through an input that nothing measures and nothing models: Kitanidis (1987),
and the joint input and state estimate of Gillijns and De Moor (2007).

- [ ] The update, on an affine model. Test: on a linear system driven by an arbitrary unknown input,
      the state estimate is unbiased and matches the closed form, and the input estimate converges.

Done when: it passes its test on all three compilers.

## 17zm. Fixed-gain filters — not started

- [ ] The steady-state Kalman filter, its gain found by running the filter's own covariance
      recursion to convergence. Test: equals the Kalman filter once it has converged, and its gain
      the closed-form Riccati solution on a scalar problem.
- [ ] The alpha-beta and alpha-beta-gamma filters, with the closed-form gains of the built-in
      kinematic blocks (Kalata 1984). Test: their gains equal the steady-state filter's on the
      constant-velocity and constant-acceleration models.

Done when: both pass their tests on all three compilers.

## 17zn. Generalized pseudo-Bayesian estimators — not started

GPBn beside the IMM and the AMM, on `SwitchingModel` [switching-models] (Bar-Shalom, Li, and
Kirubarajan 2001), with n a template parameter, since the state's type changes with it. GPBn keeps
a Gaussian per history of the last n−1 modes; a step predicts each through each mode's model,
$M^{n-1} \cdot M$ predictions, updates and reweights the $M^n$ results, and merges by moments
those that differ only in their oldest mode. The IMM, which mixes before predicting, is not in the
family and stays its own filter.

- [ ] The state: weights over the last $\max(n-1, 1)$ modes and Gaussians over the newest $n-1$
      of them, a weight's Gaussian found by dropping its older coordinates. For $n \ge 2$ the two
      agree and the state is a `Hybrid` over a product of $n-1$ copies of the mode space, one per
      lag, with distinct axis tags (a `History<ModeSpace, N>` space, say); for $n = 1$ a partial
      specialization of the state type alone, one Gaussian with weights over the current mode. The
      filter's predict, update, and merge are written once over the mapping: the prior weight of a
      new mode sums the old weights over what the mapping drops, which is empty for $n \ge 2$ and
      $\sum_i w_i \Pi_{ij}$ for $n = 1$, and the merge marginalizes the oldest axis of the
      Gaussians' index and then of the weights'. The merge builds on `Hybrid`'s marginal, and
      GPB2's cross predictions on `HybridPredictedJoint::predictedThrough()`.
- [ ] The estimator. Test: with one mode it equals the Kalman filter; from one Gaussian it equals
      the exact posterior, the mixture over all $M^k$ mode histories by enumeration, for the first
      $n-1$ steps, for $n = 2, 3$; on the IMM's maneuvering target, GPB1's error is at or above
      the IMM's and GPB2's at or below, as the source reports. No test claims the error falls with
      n on a given run, which nothing guarantees.

Done when: GPB1, GPB2, and GPB3 pass their tests on all three compilers.

## 16b. Hooks for splitting by nonlinearity — not started

The published pieces that a splitting action outside the library needs, each useful alone
[splitting-hooks]; the criterion and direction are research.md item 6, and [splitting-hooks] says
what comes back from a transform. The second differences extend 17u's root form, so they follow it.

- [ ] `withSecondDifferences(x)`, marking an input as `withCross()` does, and the result's
      `secondDifferences()`: per axis of the input's root, $\tfrac12 (e_{+j} + e_{-j}) - e_0$ from
      17u's residuals, with no further evaluation of `f`. Rules declare whether they have a centre
      and a symmetric pair per axis; cubature, the spherical simplex, `linearized()`, and a rule
      that declares nothing are refused with a message, each with a negative compile test.
      Test: on a quadratic, $\tfrac{c^2}{2} s_j^T \nabla^2 f_k \, s_j$ exactly; under
      `along(d)`, the first is the second difference along $d$; a rule of the user's own that
      declares the layout gets them.
- [ ] Hessians of a transform's function at the mean, one per output coordinate, in the joint
      input's differences, as a function of their own: by forward-over-forward autodiff, from a
      user-supplied Hessian returned with the value (`withHessian()`), and from about $n^2$
      evaluations; and the overload for the product with a direction, about $n$ duals. Test: on a
      quadratic all three, and the product, match the closed form, and a second-order filter's
      mean correction $\tfrac12 \operatorname{tr}(\nabla^2 f_k P)$ matches the unscented mean.
- [ ] For the factored transform [affine-coordinates], the mixed affine-by-nonlinear block by differencing the per-point
      Jacobians it already evaluates, as a second accessor on its result. Test: equals the
      autodiff block.
- [ ] A split of a Gaussian along a given direction by a univariate split table, returning the
      children for `MixtureParams::split()`. Test: the children keep the parent's moments, and along
      the direction their density approaches the parent's as the table grows.
- [ ] The action itself, outside the library: an [upkeep] function object that probes before the
      step, or a wrapper of the step that probes with the step's own transform along `along(d)`
      and commits it when no split is needed; its home is decided when it is written.

Done when: the second differences, the Hessians, and the split pass their tests on all three compilers with no allocation
on a bounded space, and an outside action built on them splits a component on a range-bearing
example where the unsplit filter is inconsistent.

## 17zo. Second-order and divided-difference filters — not started

Two more rungs of choosing.md's ladder, as methods beside the sigma-point sets [sigma-point-sets]:
the second-order EKF (Athans, Wishner, and Bertolini 1968) on 16b's Hessians, and the
divided-difference filters DD1 and DD2 (Nørgaard, Poulsen, and Ravn 2000).

- [ ] The second-order EKF. Test: on a quadratic measurement its mean correction
      $\tfrac12 \operatorname{tr}(\nabla^2 h \, P)$ is exact.
- [ ] DD1 and DD2. Test: DD1 approaches the EKF as the interval shrinks, and DD2 matches the
      second-order EKF's mean on a quadratic.
- [ ] choosing.md places both on its ladder.

Done when: both pass their tests on all three compilers.

## 17zp. State constraints — not started

Constraints applied as a step after the update; nonlinear ones are the optimizer's, by moving
horizon estimation.

- [ ] Projection onto linear equality constraints (Simon and Chia 2002). Test: the estimate meets
      the constraint, and its error on a constrained simulation is no larger than the unconstrained
      filter's.
- [ ] Truncation at linear inequality constraints (Simon and Simon 2010). Test: the truncated
      moments match a Monte Carlo of the truncated density.

Done when: both pass their tests on all three compilers.

## 17zq. Continuous-discrete filtering — not started

Models given as differential equations, with measurements at discrete times. The Kalman-Bucy
filter, for measurements taken continuously, stays out of scope.

- [ ] A continuous-time block that integrates a user's ODE by fixed-step RK4, differentiates it by
      autodiff, and discretizes its noise by Van Loan (1978), each stage applied by `plus()`. Test:
      on a linear time-invariant system it equals the exact discretization to the integrator's
      order.
- [ ] The moment equations of the continuous-discrete EKF and UKF (Särkkä 2007). Test: on a linear
      system each equals the Kalman filter on the exact discretization.
- [ ] Smoothing between measurement times, with 14b.

Done when: the three pass their tests on all three compilers.

## 14b. More smoothers — not started

The smoothers smoothing.md lists as absent, and the fixed-point smoother.

- [ ] The fixed-point smoother (Simon 2006, §9.2): one past state's estimate, refined as
      measurements arrive. Test: equals the fixed-interval smoother's estimate of that state at
      every step.
- [ ] The two-filter smoother (Fraser and Potter 1969), through `InfoFilter`. Test: equals the
      Rauch-Tung-Striebel smoother on an affine model.
- [ ] The iterated smoothers (Bell 1994; García-Fernández, Svensson, and Särkkä 2017). Test: on an
      affine model one iteration equals the Rauch-Tung-Striebel smoother.
- [ ] Smoothers for the Rao-Blackwellized and grid filters, and smoothing a mixture of IMMs.
- [ ] Particle smoothing by backward simulation (Godsill, Doucet, and West 2004), on 17zh's density.
      Test: on a linear Gaussian model its moments approach the Rauch-Tung-Striebel smoother's as
      the counts grow.

Done when: each passes its test on all three compilers.

## 17zr. Innovation whiteness test — not started

The third consistency test beside the NIS and the NEES [innovation-statistics]: whether a run's
innovations are correlated in time (Bar-Shalom, Li, and Kirubarajan 2001, ch. 5), which a
mistuned filter can show while its NIS passes.

- [ ] The normalized sample autocorrelation of the innovations at lags 1 to $L$, with its
      acceptance region, beside `NeesMonitor`. Test: over Monte Carlo runs of a correctly tuned
      filter it accepts at its stated rate; with $Q$ too small it rejects.

Done when: it passes its test on all three compilers.

## 16c. Fusion rules, out-of-sequence updates, and consensus — not started

The distributed-fusion operations that are pure functions of the library's types
[distributed-fusion]; the node bookkeeping is the tracker's.

- [ ] Divide on `InfoGaussianParams`, and through `toInfo()` on `GaussianParams`: the inverse of
      multiply, after moving both operands to one reference [covariance-reset]. Tests:
      $(a \times b) \div b = a$; a step's posterior divided by its prior and multiplied back gives
      the posterior; on an affine model with no process noise, two sensors' tracklets multiplied
      into a common prior equal the centralized filter. A result that is not positive semidefinite,
      when the divisor holds information the dividend lacks, is an error. On
      `SqrtInfoGaussianParams` divide is a Cholesky downdate by hyperbolic rotations, which can
      fail partway; forming the precision instead gives up the square-root form, so decide which
      when it is written.
- [ ] The product of two independent densities in any two representations, as `multiply()`
      overloads, each giving its pair's natural result: any two Gaussian forms, converted, in
      the form the caller names, since every conversion is exact; a Gaussian mixture times a
      Gaussian, each component updated and its weight scaled by
      $\mathcal{N}(\mu_i; \mu, \Sigma_i + \Sigma)$; two mixtures, $n m$ components; two
      categoricals; and particles or a grid times any density with a log-density, by `reweight()`.
      A result that only approximates, such as particles times a Gaussian returned as a
      Gaussian, stays a separate collapse or fit the caller writes. The product is exact only
      when the operands' errors are independent: no shared prior counted twice and no process
      noise both filters saw. Its documentation says so, and points to divide and covariance
      intersection for the rest. Low or zero process noise is the use case: static parameters,
      calibration, orbit determination. Tests: two Kalman filters with independent priors and
      sensors and no process noise, multiplied, equal one filter given both sensors; each pair
      against its closed form.
- [ ] `multiply()` on two `Uniform`s, the first case needed: the intersection of the boxes, with an
      empty one a `UniformError::Empty`, since the two estimates contradict each other. An
      indicator is idempotent, so a shared prior box counted twice changes nothing and no divide
      is needed; and if each box is sure to hold the truth, so is the intersection, whatever the
      filters share, process noise included. Two arcs of a `Periodic` coordinate can meet in two
      pieces, which no box holds: decide between a new `UniformError` and a result that holds
      both. Tests: the intersection on ordinary and periodic coordinates, the empty case, and
      the split arcs.
- [ ] Fusion with a known cross-covariance (Bar-Shalom and Campo 1986). Test: equals the joint
      Gaussian's conditional, built in plain Eigen.
- [ ] Covariance intersection (Julier and Uhlmann 1997), its weight from a one-dimensional
      minimization of the determinant or the trace, and with a given weight the Kullback-Leibler
      average (Battistelli and Chisci 2014). Tests: weights 0 and 1 return an operand; the fused
      covariance dominates the true error covariance over a grid of cross-covariances.
- [ ] Out-of-sequence updates on `KalmanFilter` over a `GaussianHistory` with a `Lag`: one-step lag
      (Bar-Shalom 2002) and $\ell$ steps (Bar-Shalom, Chen, and Mallick 2004). Test: equals the
      filter rerun with the late measurement in order, for lags up to the window.
- [ ] A consensus step over a node's filter and its neighbors' messages (a density and a weight
      each): consensus on information, the hybrid consensus (Battistelli, Chisci, Mugnai, Farina,
      and Graziano 2015), and the Kalman-consensus filter (Olfati-Saber 2009), with an execution
      policy [execution-policies]. Tests: equal neighbors leave the node unchanged; with
      Metropolis weights the nodes agree as iterations grow; `tests/parallel.cpp`.
- [ ] The channel filter (Grime and Durrant-Whyte 1994) as an example and a test: two nodes
      exchanging every step equal the centralized filter.
- [ ] `tests/allocation.cpp`: none of these allocates on a bounded space.

Done when: the above pass on all three compilers, with the consensus step's parallel case.

## 17za. Mixture arguments of a model — not started

A mixture as an uncertain argument of `predict()` or `observe()` [prediction-inputs], in stages after
the first, which exists (additive mixture noise, [additive-noise]): a mixture through the function;
then several mixtures each naming a mode axis, as `additiveNoise(byMode<Axis>(...))` names one for
additive noise [imm]; then several memoryless mixtures.
The Kalman filter refuses a mixture argument, and the mixture filters take it, with a branch per
component. Each stage is an addition; the first two change no spelling.

- [ ] A mixture through the function: a `GaussianMixtureParams` argument read by the model, each
      component pushed through by the method, the branches weighted by the component weights. Test:
      a `GaussianMixtureFilter` prediction with a two-component input equals the mixture of the two
      predictions with each component as a Gaussian argument.
- [ ] Several mixtures each naming a mode axis, for an `ImmFilter` [imm].
- [ ] Several memoryless mixtures, their components crossed.

Done when: each stage passes its test on all three compilers.

## 17zd. Step models — not started

Not decided whether to build. A process model whose function returns the step over the blocks it
writes rather than their new values, with the library applying `plus()` to each written block. Most
process functions compute new block = old block ⊞ step, so the function reads the block, builds an
output vector, and writes the block back, naming its tag three times; a step model names it in
`writes()` alone. The attitude's prediction becomes the same line as the heading's, since `plus()` on
a unit quaternion is $q \otimes \mathrm{Exp}(d)$:

```cpp
Emrys::processModel(
    Emrys::increment([]<typename T>(auto const & in, auto const &, auto const & x, double dt) {
        return (Emrys::less<Gyro>(Emrys::block<Gyro>(in, x), Emrys::block<Bias>(in, x)) * dt).eval();
    }),
    Emrys::reads(Emrys::path<Heading>, Emrys::path<Bias>, Emrys::path<Gyro>),
    Emrys::writes(Emrys::path<Heading>),
    Emrys::continuousNoise(Emrys::on<Heading>(noiseDensity)));
```

What it needs: every written block is also read; a zero difference vector to fill, beside
`zeros<T>(out)`; and a returned Jacobian of the step chained through ⊞ by the blocks' Jacobians, as
[covariance-reset] does. `increment` is a placeholder name. A model whose new value is not a step
from the old one keeps the present form. It breaks nothing, but is decided before the launch, since
if it is built the examples use it.

Done when: decided, and if built, the examples whose models are steps (`consider_bias.cpp`,
`attitude.cpp`, `unicycle.cpp`) use it, and a test shows a step model predicts as its value form does
under every method.

## 19. GPU backend — deferred

A `Device` policy runs the step 18 primitives on a GPU, with the particles in device memory
through the allocator [allocation] and the per-element bodies compiled as device code. The
experiments of parallelism.md, "What to find out first", decide whether the step is worth it.
Deferred past the advertised launch (decided 2026-11-16); step 19b readies the API for it, and
has taken the device-only syntax check.

- [ ] The experiment: the ranging benchmark under nvc++ `-stdpar=gpu` against
      `std::execution::par`, for the crossover count.
- [ ] `EMRYS_FILTER_DEVICE` on the kernel-reachable surface, and a CI job compiling a device-only
      instantiation with clang.
- [ ] A device-callable fixed-size Cholesky, triangular solves, and log-determinant in `Core/`
      (Upstream to Eigen).
- [ ] `Device` over Thrust (CUDA and rocThrust): the primitives, the reductions, resampling by scan
      and search, and a `UniversalAllocator` alias, on shared memory.
- [ ] A second backend over oneDPL and SYCL, which runs the device path on a CPU device in CI.
- [ ] Later: `BufferFor<Allocator, T>` for device-only memory, with an explicit copy to the host.
- [ ] Tests: predictions equal `Sequential` bitwise from the same Philox seed, reductions within
      tolerance; compile-only in CI, hardware by hand.
- [ ] Benchmark: the device path against the host's parallel path over the particle count.

Done when: a particle filter and a Rao-Blackwellized particle filter run every step on a GPU with
only decision scalars crossing to the host, on one NVIDIA and one other device.

## Documentation — in progress

The aim is documentation that can accompany a graduate course: every algorithm cited
[published-algorithms-only], extensive examples, a well-structured reference, and the design
explained for people and agents alike. The documents for each reader:

| Reader | Now | Once the guides are empty |
|---|---|---|
| Any agent, every session | `AGENTS.md` (CLAUDE.md imports it) | the same |
| Contributor | `style.md`, `decisions.md`, `roadmap.md`, the guides | `style.md`, `decisions.md`, `roadmap.md` |
| User and user's agent | `docs/usage.md`, `examples/`, the public reference, `CHANGELOG.md` | the same |

- [ ] As each guide section settles, move it into the reference: explanation to a module's `.dox`
      page, examples to `examples/` included with `\snippet`, reasons to decisions.md; the section
      becomes a pointer. 0.1.0 was released with the guides as they were, about 8,800 lines and
      93 open questions (decided 2026-11-12), and the move continues. The advertised launch waits
      for every guide reviewed and emptied, every question answered and none moved here (decided
      2026-11-16). A step above whose area no guide covers gets a review of its own, written as
      docs/guide/README.md says, before it is done. The guide README's table tracks both.
- [ ] A user page listing every circumstance in which the library may allocate: a count growing
      past its reserve, the first use of count-sized scratch [allocation], the standard algorithms
      under a policy other than `Sequential` [execution-policies], and anything sized by an
      unbounded space. Audit the headers for it rather than writing it from memory.
- [ ] What is thread-safe, on the user page: which calls on one filter may run at once (none that
      change it), whether two filters sharing a model may run on two threads, and what an
      execution policy's body may touch [execution-policies]. Audit the headers for it, as for
      allocation.
- [ ] An example on recorded data: a small public dataset, chosen for its license, beside an
      example that reads it with `<fstream>` itself, since I/O is not the library's.
- [ ] More examples beside the six of Agent-friendly use, among them learned models: a small
      network's weights embedded as Eigen matrices, under Unscented and under Linearized with its
      own Jacobian [returned-jacobians]; a heteroscedastic network as input noise; a learned
      process noise as an uncertain input [prediction-inputs]; an ensemble as the modes of a
      switching model. And a wide prior located by precise nonlinear measurements, such as ranges
      or Doppler shifts, under a `GaussianMixtureFilter` and as a mixture of IMMs, the row of
      usage.md's table that has none.
- [ ] Contributor skills, in this repository's `.claude/skills/` (and `.agents/skills/` if another
      agent works here; [user-skills] says which agent reads which), apart from the user skills:
      `pre-pr` (the full suite in order, rebase before building, the batch from `next` to `dev`;
      AGENTS.md's CI and Doxygen paragraphs move into it), `new-entity` (test file and CMake entry,
      `\file` and `\ingroup`, deduction guides in `\cond` with a test through the alias, allocation
      and `parallel.cpp` cases, bibliography entry, roadmap row), `record-decision`,
      `guide-review`, `guide-to-reference`, and `benchmark`. Each states a procedure and points to
      style.md and decisions.md.
- [ ] `emptyUpdate<M>()` of a `KalmanFilter`, and so of each filter whose result holds a
      `FilterUpdate` of a fixed number of branches, fails to compile for a measurement without
      mixture noise: the branches are a `std::array` of `FilterBranch`, whose `Innovation` has no
      default constructor, so `FilterUpdate<MeasurementSpace, 1>` has none. Decide what an empty
      innovation is (sized by the measurement space, as `Moments` is), or constrain
      `emptyUpdate()` to a run-time number of branches and say so (found 2026-11-22).

## Agent-friendly use — in progress

Make it cheap for a coding agent, working for someone who has just cloned the repository, to build
a working estimator: few tokens read, few compile loops, and a numeric check that it is right.
`docs/usage.md` and the six programs in `examples/` (constant velocity, unicycle under the EKF and
UKF, attitude, particle, IMM, Rao-Blackwellized particle), each run as a test that fails when its
consistency check does, are the start, with the first user skill, `emrys-filter-build-estimator`,
the CMake target that installs it [user-skills], and `examples/consumer/`, a project to copy that
fetches the library, keeps template errors short, and installs the skills.

- [ ] Friction met writing the examples, for the library. Missing:
      `simulate()` of a `SwitchingModel` with its mode (imm.cpp draws the mode by hand from the
      chain's rows), built-in coordinated-turn and gyroscope-integration blocks
      (unicycle.cpp and attitude.cpp write their own, the largest part of each), the NEES of a
      particle set without building a `GaussianParams` from its moments, one mode's probability in
      fewer than three calls, and the weight of a region by tag path rather than a row index. A
      particle set that degenerates onto one particle reports nothing; only a NEES in the 1e43s
      showed it. GCC 14 warns `-Wmaybe-uninitialized` inside `GaussianParams`' copy (the cached
      `std::optional<Eigen::LLT>`) when `additiveNoise()` copies one, depending on inlining.
- [ ] A filter's description naming its model, its method, and its last update (accepted, NIS),
      after its name and state [printing-and-simulation].
- [ ] Usage page and skill: once the API index and `checkModel` exist, point to them, in usage.md
      and in step 3 of `emrys-filter-build-estimator`.
- [ ] Short compile errors: the model contracts as named concepts, checked by `static_assert`s with
      plain messages at each filter's entry, and `checkModel<Model, Space>()`.
- [ ] A compact API index generated from the Doxygen XML (public signatures only), and a header map
      in the README.
- [ ] More user skills, as the evaluation shows where agents go wrong; candidates are choosing a
      filter for a problem, writing a process model, and checking an estimator on recorded data.
- [ ] An evaluation: fresh agents in a clean copy of `examples/consumer/` given fixed tasks (GPS and odometry
      EKF, quaternion attitude, a multimodal tracker), recording tokens, compile loops, and whether
      the consistency checks pass, with and without the skills, repeated after each change.
- [ ] The evaluation under agents other than Claude Code, each given the task, prompt, and pass
      criteria of the Doppler transmitter run: Codex CLI
      (a different model family, reading AGENTS.md and `.agents/skills/`, installed with
      `AGENT_SKILLS_DIRECTORY`), Copilot agent mode in VS Code and the Copilot coding agent working
      alone from an issue, Gemini CLI (GEMINI.md by default, so whether it finds usage.md
      unprompted), Cursor, and Aider, which loads no instructions file or skills and so tests
      usage.md alone. Record for each whether it found usage.md unprompted, installed and used the
      skills, and chose the filter choosing.md would.

Done when: a fresh agent's estimator passes its consistency check on the evaluation tasks, and its
tokens and compile loops have dropped measurably from a baseline taken before this section.

## Verification — not started

Evidence that the library is efficient and builds where embedded users need it [benchmarks],
[coverage], [embedded-builds], in the order each supports the next.

- [ ] Warnings: the tests, examples, and benchmarks build with `-Werror` and `/WX`
      (`EMRYS_FILTER_WARNINGS`, GCC without `-Wmaybe-uninitialized`, which Eigen's inlined code
      trips), and with `-Wshadow -Wconversion -Wsign-conversion -Wold-style-cast` under GCC and
      Clang since 2026-11-22. Left: a release build of the suite in CI:
      every preset is Debug, so a variable read only by `EMRYS_FILTER_ASSERT` goes unused under
      `NDEBUG` unseen, as `ProcessModel.h`'s `predict()` did until the Cortex-M build found it.
- [ ] Coverage [coverage]: `tools/coverage.py` and the `coverage` job report the headers by
      layer. On 2026-11-06, after 17u, every one of the 1439 public functions was instantiated,
      99.9% of 7055 regions executed, 96.5% of 2082 branch outcomes, and about 88% of 86,500
      instantiations ran. The 10 regions never executed, listed in `coverage_unexecuted.txt` in
      the build directory, are accepted, not worth chasing: guards no input was found to reach,
      the modified Lentz method's two in `ChiSquared.h`, the quantile's bracket doubling to
      infinity (no probability below one at up to 1e9 degrees of freedom, in `float` or
      `double`), and `FormSteps.h:1166`; a factor's conditional failing in `Product.h` and
      `StudentT.h`, which no factor that passed its check does (a product checks its factors when
      it is made); `WithFallbacks::rootFromFactor()`'s assertion that a last fallback that cannot
      fail did not (`Roots.h:1097`); and `leafSupport()` (`Composite.h:690`), evaluated only at
      compile time but `constexpr` for MSVC. The 225 negative compile tests cover the 213 claimed
      compile errors (`tools/check_compile_errors.py`) and pass with Clang 21, GCC 15, and MSVC. Found by the region tests and not fixed:
      `chiSquaredCdf()` in `float` loses its accuracy past about a million degrees of freedom,
      where its log-prefactor cancels. Left: the axis checklist with typed suites, for the branch
      outcomes and instantiations.
- [ ] No exceptions [embedded-builds]: the `gcc-no-exceptions` preset and CI job build the suite
      with `-fno-exceptions -fno-rtti`, leaving out the tests of the standard parallel policies,
      whose libstdc++ implementation catches exceptions (`tests/standard_policies.h`). Credible
      once the instantiation tier of coverage is met.
- [ ] Cortex-M [embedded-builds]: `benchmarks/cortex_m/` and its CI job run the ranging tests in
      `float` on the M4F and M7 under QEMU, with no heap, and report `.text`, `.data`, and stack
      per function. Left for benchmark 6 (benchmarks.md): a program of the filter alone, since the
      sizes now include the tests' double reference filter, simulator, and chi-squared quantiles,
      and the attitude filter, which is `double` only.
- [ ] Benchmarks [benchmarks]: the Callgrind CI job and the 1.10 instruction-count ceiling, then the
      ranging numbers published with their method (benchmarks.md has the plan). The ranging,
      attitude, and particle pairs exist and agree with their hand code. At the fixed step ranging
      is 0.89 to 1.44 times the hand code's instructions, attitude 1.18 and 1.27, and particles
      1.03 to 1.07. Over the ceiling: 2D with 4 anchors, by the library's fixed cost of about 320
      instructions per step (two lower-triangle mirrors, the update's calls that are not inlined,
      the check of the factorization and its `std::expected`), and 3D with 6, by about 1,100 in
      double, which have not been broken down. With 16 anchors in 2D and 12 in 3D the hand code
      takes up to 11% more than the library, and what it does worse is to be found.

Done when: CI is warning-free with warnings as errors, the coverage job reports every public entity
instantiated and every region executed, the no-exceptions and Cortex-M jobs pass, and the ranging
numbers are published.

## Launch — in progress

What a first visitor meets before any feature matters, then what a reviewer looks for.

The launch is in two parts (decided 2026-11-12). The soft launch was 0.1.0, released on
2026-11-13 without an announcement, before the guides are empty (Documentation above). The
advertised launch, the posts under "Getting the word out" below, follows it, and waits for what
the opening of this file says (decided 2026-11-16). Each guide question answered goes into a 0.x
release with its `CHANGELOG.md` entry. The vcpkg and Conan submissions wait for the advertised
launch (decided 2026-11-12), and Discussions are turned on with it.

The repository is public since 2026-11-12, from one squashed commit. A release is tagged on
`main`, which `dev` fast-forwards, and published there as a GitHub release; Zenodo archives each
published release, not a pushed tag, so a release is published only from its tag. The reference is
on Pages, and Discussions are off.

The front door:

- [ ] vcpkg and Conan ports of the CMake package [cmake-package], vcpkg first, submitted at the
      advertised launch. vcpkg ships with
      Visual Studio and is where a Windows C++ developer looks first, so its registry is the
      larger discovery surface, but a port there records no compiler floor: a user on GCC 13
      installs it and meets the failure at the first include. A Conan recipe of package type
      header-library serves every profile from one package and states the floor in `validate()`,
      which rejects an older profile with a plain message before anything compiles; Conan
      Center's CI treats that as an invalid configuration, not a failure. Both registries carry
      Eigen 5.0.1 and no 5.0.0 (2026-11-03), and the library is tested against 5.0.1 while
      accepting any 5.0 or later. The vcpkg overlay port is `packaging/vcpkg/ports/emrys-filter/`,
      and the Conan recipe `packaging/conan/recipes/emrys-filter/` in Conan Center's layout. Built
      from the working tree by `tools/vcpkg_test_port.py` and `tools/conan_test_recipe.py`, each
      passes its own checks, and a consumer built with Clang 21 runs against each; the Conan
      recipe's `validate()` rejects Clang 20 and `cppstd` 17 with a plain message. The `vcpkg`
      and `conan` CI jobs passed on 2026-11-03 (Conan with GCC 14 and Clang 21
      on Linux). Until the submissions, those jobs keep the port and recipe working. Left: the
      `SHA512` and `sha256` of the archive of the tag submitted in the port and
      `conandata.yml`, with a CI step that installs the unmodified port and recipe after that
      tag; and whether Conan Center takes the version range `eigen/[>=5.0.0 <6]` or wants `eigen/5.0.1` exactly.
- [ ] The Eigen 5.0 features the library uses, in the README. Its opening names Eigen 5, and its
      Install section says that 5 is required and 3.4 is not enough; `Core/Version.h` rejects an
      older Eigen.
- [ ] The README's Install section, once the vcpkg and Conan packages are published at the
      advertised launch and not before, since it states only what is true: `vcpkg install
      emrys-filter` and `conan install --requires=emrys-filter/0.1.0` first, then the same
      `find_package` lines, and that vcpkg and Conan Center carry Eigen 5. Say that Conan rejects
      a compiler below the floor with a plain message, and vcpkg does not.
- [ ] The minimum compilers. The launch states only what has been tried, as the README's Install
      section does: CI passes on GCC 14, Clang 21, Apple Clang 21, and the MSVC of Visual Studio
      2026; GCC 13 builds the README example and the examples but isn't tested; Clang 18 on
      libstdc++ 13 fails at the first `std::expected`. Untried: Clang 19 and 20 (which the Conan
      recipe's `validate()` rejects untested), older MSVC and Apple Clang, and each compiler on the
      other standard libraries. Closing it means a table by compiler and standard library,
      `validate()` matching it, and a check naming the floor before the first `std::expected`, the
      only message a vcpkg user would get.
- [ ] Which built-in models ship here, and which in EmrysNavigation: undecided (2026-11-16), as is
      whether EmrysNavigation launches with this library or shortly after with a set of
      essentials. The candidates here are the coordinated-turn and gyroscope blocks the examples
      write themselves, and range, bearing, and range-bearing sensors. Nothing is taken from
      navigation's scope to fill this library.
- [ ] The library on Compiler Explorer, so that a thread can link a snippet that compiles.
- [ ] Build cost for contributors: CONTRIBUTING.md has each file's compile time and memory under
      Clang and MSVC and the jobs a machine's memory allows (measured 2026-11-22). Left: a clean
      build's wall time per workflow preset, GCC's numbers, and `-ftime-trace` numbers per
      header, which needs the trace files read, not only `tools/compile_cost.py`'s totals.
- [ ] Shorter test builds: `-ftime-trace` on the slowest test files (`tools/compile_cost.py
      --match tests/ --time-trace` on a test build finds them), then explicit instantiation of
      the setups the tests repeat (`extern template` in a support header, instantiations in their
      own translation units), and a precompiled header if parsing matters. Record a clean build's
      time per preset before and after.
- [ ] A C++23 module beside the headers [layered-headers], after the profiling above, built if
      the three compiler families import it reliably; if not, a decision records why.
- [ ] What a precondition failure does, on the user's page: `EMRYS_FILTER_ASSERT` is `eigen_assert`,
      which aborts, and `NDEBUG` or `EIGEN_NO_DEBUG` silences it. A user's own definition may be an
      expression or a statement (`tests/assert_override.cpp`); a throwing one terminates inside a
      parallel loop body and across the containers' `noexcept` moves, such as `ParticleSet`'s. The
      other way an exception can come out is `std::bad_alloc` from a count-sized container's
      allocator when exceptions are on.

Getting the word out, once the front door is done: the README's comparison table and its numbers
are what a reader checks in the first hour, and a thread that finds them wrong is the launch.

- [ ] The OpenSSF Best Practices badge at the passing level, beside the DOI in the README, added
      in the 0.2.0 release (decided 2026-11-16). The project is registered on bestpractices.dev
      and its criteria answered before then; the badge waits for 0.2.0 because clang-tidy first
      runs before that release, so the static analysis criterion is true of the release the badge
      appears in, and not of 0.1.0. Every passing criterion the repository can meet was met on
      2026-11-16, private vulnerability reporting and secret scanning with push protection
      included. The answers to `know_secure_design` and `know_common_errors` say that the
      maintainer is not a security specialist and learned the security design from the coding
      agents it was worked out with, and name the checks that hold it: the sanitizers,
      clang-tidy, and [value-preconditions]. Until the audit below is done, they don't claim the
      code was checked against it.
- [ ] The code checked against [value-preconditions], before the advertised launch:
      `SECURITY.md` names a breach as a vulnerability, but nothing has audited for one yet. Read
      each `EMRYS_FILTER_ASSERT` on a floating-point value, and each loop or index computed from
      one, for what follows when the assertion is compiled out; then add tests that pass NaN and
      infinities to each filter's predict and update, each resampler, each distribution's
      factory and draw (the standard rejection samplers, `std::gamma_distribution` among them,
      included), and the mixture reductions, run in a build with `NDEBUG` under the sanitizers.
- [ ] The review of every algorithm against its source, before the advertised launch (decided
      2026-11-16): every entry of `docs/reference/bibliography.md`, those the steps above add
      included, read in full, and every algorithm in the headers checked against the equations it
      cites, as the bibliography's opening says. Each comment cites its source's sections and
      equations where the code follows it, and each departure is said in the entry's note. The 34
      entries marked **Not read.** (2026-11-12), which comments credit while taking their numbers
      from a free source, are the first. The maintainer obtains the paywalled papers and keeps
      them out of the repository, which is public. Done when no entry says **Not read.** and a
      reader who checks a citation finds what the comment says there.
- [ ] Benchmarks against the other libraries that run the same problem, those in the README's
      comparison table and the ones a user would otherwise pick, by the method of benchmarks.md
      (claim 9), since a reader's first question is "how fast against X", and the README's
      numbers are only against hand-written Eigen. That is mherb/kalman, all four of its filters
      (EKF, UKF, square-root EKF, square-root UKF) on the ranging problem, TinyEKF's EKF, the
      C filter a microcontroller user would pick (claims 6 and 9), FrancoisCarouge/Kalman's EKF,
      and OpenCV's `cv::KalmanFilter` run as an EKF, with the results in
      `docs/benchmarks.md`. Started on branch `mherb-benchmark` (2026-11-22): the plan and the results page are written, and its four
      filters, at its last commit (`9f40c2f`, 2018-08-22), build and run unchanged against Eigen
      5.0.1 with Clang 21, warning once (`-Wdeprecated-enum-enum-conversion` in
      `SquareRootExtendedKalmanFilter.hpp`), so its headers come in as system includes. All four
      pairs are written (benchmarks.md, claim 9), and every pair of every library is timed in
      double and in float. A pair that does not agree is timed with a mark [benchmarks], behind a
      test that the difference is the other library's: mherb/kalman's EKF in float, where its
      explicit inverse of S loses its accuracy, and both its UKFs, whose update leaves the
      process noise out of the sigma points; TinyEKF in float, its default, where its explicit
      inverse, its `(I - G H) P`, and its unmirrored covariance together lose its accuracy; and
      OpenCV in float, its default, where it solves for the gain by an SVD of an asymmetric
      innovation covariance. TinyEKF is at its last commit (`ba0d2a9`, 2026-04-09). OpenCV is
      found rather than fetched (`EMRYS_FILTER_BENCHMARK_OPENCV`, off by default), probed with
      4.14.0 built in WSL under `~/opt/opencv-4.14.0`, and timed over runs of 300 steps, since
      in double its unmirrored covariance drifts from step 400 on and diverges by step 2000
      (benchmarks.md, claim 9); with 8 anchors or more it allocates once per step. No CI job
      installs OpenCV. FrancoisCarouge/Kalman's EKF joined on 2026-11-24 (benchmarks.md,
      claim 9), fetched at `cdd8a93` with the Eigen backend of its `support/eigen` in a
      translation unit of its own, since that backend's `operator/` takes over Eigen's array
      division; the pair agrees in both scalars, and the library takes 0.12 to 0.32 of its time.
      Every pair is built and tested with Clang 21 in WSL (2026-11-22, the mherb/kalman EKF also
      with Clang on Windows), and measured there: instructions and time are in
      `docs/benchmarks.md`. The first measurement found the library taking about 1,500
      instructions per step beyond the hand code at the small sizes, most of them for a Jacobian
      that every prediction built and discarded; that is cut (2026-11-23, benchmarks.md, claim 1),
      and every count on the page, with the ranging and library times on Linux, is taken again
      after it. The time at the small sizes was then found in Eigen's `LLT::compute()` and
      its `llt.solve()` of a small fixed vector, nine tenths of the hand code's step at 2D
      with 4 anchors, and the solve is replaced (2026-11-23, benchmarks.md, claim 9):
      `transposedSolve()` substitutes through a Cholesky factor of a fixed size up to 24
      itself, every right-hand side together in columns padded to whole packets, and the
      hand code with it; the library's step in double went from 167 to 114 ns at 2D with 4
      anchors and from 547 to 225 at 2D with 8, and every count and Linux time on the page
      is taken again. The factorization followed (2026-11-23, [own-cholesky]):
      `CholeskyFactorization` factorizes a real matrix of a fixed size up to 24
      itself, without square roots, since a filter's steps wait on each other and a column
      then waits on one division, and the innovation covariance's gain, NIS, and
      log-determinant come from it; the step in double went from 114 to 70 ns at 2D with 4
      anchors and from 225 to 155 at 2D with 8, and the page is measured again at that commit.
      `roots::cholesky()`, which the default sigma points are spread by, takes its root by the
      same type since the commit after. The type is public since the commit after that (the
      user's decision, 2026-11-23), and the Gaussian's own: `Gaussian::cholesky()` replaces
      `llt()`, and `covarianceRoot()` gives the factor $L$ kept with it, which the samplers
      multiply by; the page is measured again at that commit. Claim 9 now holds for the EKF
      against all four libraries at every size and for the UKF in double, and fails for the
      UKF in float at 2D with 4 anchors (1.03 times mherb/kalman's), the square-root EKF in
      double at 3D with 24 (1.06), and the square-root UKF in double at every 3D size (up to
      1.29).
      The information forms' cached factorization and the local ones of the form steps, the
      fitting, the particle sets, the Rao-Blackwellized particle filter, and the conditional
      with its evidence followed, in the commit after the page's. The posterior Cramér-Rao
      bound stays on `Eigen::LLT`: from no information its recursion cancels to exact zeros
      with Eigen's arithmetic and leaves 1e-14 with the library's, which then factorizes as
      information (benchmarks.md, claim 9).
      `Core/Solve.h` no longer substitutes through an `Eigen::LLT`, which nothing called.
      Next: give the Cramér-Rao recursion one consistent way to form its terms, then move it;
      the same substitution in
      `solveInPlace()` and `inverseFrom()`, which still solve a column at a time through
      Eigen; find where the 16 ns and 320 instructions go that the library adds to the hand
      code's step at 2D with 4 anchors (candidates read from the code: the calls left out of
      line, and the covariance mirror in `kalmanConditionInPlace()`); find whether the rest
      of a 6-state step in float has the mismatch of stores and loads the substitution had,
      since 3D with 6 anchors is still slower in float than in double on both sides, and the
      library's float step there is 1.47 times the hand code's;
      take the Windows times, and the
      attitude and particle times, again; MSVC builds of the pairs; the Cortex-M half of
      TinyEKF joins claim 6's footprint benchmark. The whole test suite passed with Clang 21
      and with GCC 15 in WSL (2026-11-23, both with warnings as errors) but
      `Allocation.ParticleFilterUnderAUsersPolicyStaysOffTheHeap`, whose scratch arena `next`
      enlarged before the branch was rebased onto it (2026-11-24). SymForce and manif wait for
      the Lie group spaces (the user's decision, 2026-11-24): a pair on the ranging problem alone
      would compare Eigen with Eigen. Issue #15 tracks this item.
- [ ] GitHub Discussions turned on, so questions have somewhere to go besides issues. The
      repository's 20 topics, GitHub's limit, were chosen on 2026-11-12 by how many public
      repositories use each, so that each has an audience browsing it; a filter named in the
      README is found by search without a topic.
- [ ] The day of the advertised launch: a Show HN (a Hacker News post of one's own work, with a first comment
      from the submitter saying what it is and what feedback is wanted; one attempt, a weekday
      morning US time, since a post climbs in its first hour or sinks and reposts are discouraged)
      and an r/cpp post, both leading with the README's opening and the 50-line example. The
      story for that audience is the design, not another Kalman filter: one recursion over any
      space, allocation-free under `EIGEN_RUNTIME_NO_MALLOC`, no exceptions, Cortex-M under QEMU,
      execution policies, C++23 throughout. The thread's value is its bug and platform reports.
- [ ] The week after: r/robotics, r/ControlTheory, ROS Discourse, and the Eigen mailing list, where
      a serious consumer of Eigen 5.0 is news the maintainers relay; pull requests to the
      awesome-cpp and awesome-robotics lists.
- [ ] The projects in the README's comparison table: an example that wraps a manif group as a
      `Space`, written before the advertised launch and sent to manif's authors; GTSAM's forum is where "a
      recursion, not a graph" questions land.
- [ ] The academic track, in parallel and months long: a JOSS paper beside the `CITATION.cff` and
      its DOI, since "every algorithm cited, nothing invented here" and documentation that can
      accompany a graduate course are what JOSS reviews for, and a paper is what puts a library
      on a syllabus; and `docs/research.md`'s open problems sent to advisors in the ISIF and
      FUSION community directly, since each is scoped for a graduate student.
- [ ] After the advertised launch, an email to each author whose algorithm the library implements,
      inviting a review of the implementation (proposed 2026-11-22). One email per person, not
      per paper, to the corresponding author at a current address; for a source whose authors
      have died, such as Kalman's, a current group in the area instead. Each email names the
      entries it covers, links the code that implements each at a tagged release (not `dev`,
      whose lines move), lists the departures the entry's note records with a question on each,
      and links a program that runs the method: where the paper has an example, that example
      reproduced, built and run as a test (`examples/papers/<anchor>.cpp` or a test file) so it
      keeps compiling, printing the paper's numbers beside the library's, with any that differ
      named as a question rather than left out. It says that coding agents wrote much of the
      code, offers a public place for the answer (a GitHub Discussion, or an "algorithm review"
      issue template), and asks whether the author would like to be named in an
      acknowledgments list. Waits for the review of every algorithm against its source above:
      no email goes out for an entry that says **Not read.** Before the emails, a prioritized list
      of the algorithms for the maintainer's own full review, reading the paper and checking the
      code against it, first those the most users run and those whose notes record departures;
      since the emails go out over time, the maintainer reads as many of each email's papers as
      possible before that email goes out, and each email says which of its papers the
      maintainer read in person. Sent a few at a time, so that what
      the first replies find is fixed before the next. The list of recipients, their addresses,
      and their replies are kept outside this public repository; what an answer changes goes in
      the code and the entry's note.
- [ ] Two README badges beside the DOI and the OpenSSF Best Practices badge, each added only once
      it is true: JOSS's, when the paper above is accepted, since it is the one badge that stands
      for peer review; and Repology's "Packaging status" (`repology.org/badge/vertical-allrepos/
      emrys-filter.svg`), once the vcpkg and Conan submissions are accepted, in place of a badge per
      registry. Repology tracks both Vcpkg and ConanCenter (checked 2026-11-16) and adds any later
      registry or distribution without a README edit; if the two registries name the package
      differently, Repology's maintainers merge them on request.
- [ ] The agent story as a second launch, after the main one so that neither post explains the
      other: the skill in `skills/` submitted to the skills directories and written up on its
      own. It waits for the evaluation under agents other than Claude Code (Agent-friendly use
      above), since "agent-friendly" is then a claim about Codex, Copilot, Gemini CLI, Cursor, and
      Aider with that evaluation's results as the evidence, and the write-up says what each found
      unprompted and whether its estimator passed its consistency check.
- [ ] Not channels: Chocolatey, winget, and Scoop install applications, not libraries a build
      finds; they are where a contributor gets CMake, Ninja, clang-format, and Doxygen.

Allocation [allocation]:

- [ ] A Rao-Blackwellized update under mixture noise allocates each point's branches, since its
      result, which now owns its arrays, has no slot to keep them in.
- [ ] A point-mass filter with a hybrid per cell, and a particle filter with a hybrid per particle,
      allocate at every update the IMM result they reuse across the points
      (rao-blackwellized-filters.md Q26), and run their points only in order (Q25).
- [ ] `drawSamples()` of a gamma, `drawParticleGaussian()`, the fitting, and the samplers'
      batches build owning sample sets, which a bounded space shouldn't need.
- [ ] `GridGaussianParams::subdivide()`, `PointGrid::subdivided()`, and `collapseGivenPoints` allocate
      from the global heap (rao-blackwellized-filters.md Q18); `PoissonEmission`,
      `ExponentialEmission`, and `ArrivalInterval` copy their rates into every measurement
      (hidden-markov-filter.md Q5).
- [ ] The smoothers refactorize each step's predicted covariance on every `smooth()`, and the
      information form builds the covariance where the root suffices. Left as they are, with
      reasons: the histories copy the filtered estimate into the smoothed slot at `push()`;
      `MixtureFitter` allocates per iteration; the measurement steps of FormSteps.h factorize the
      read block again after `Transform::root()`; `Transform` evaluates the function once more to
      size a resizable output.

Decisions and comments that disagree with the code:

- [ ] The page numbers of Särkkä, Bunch, and Godsill (2012) in the bibliography, from the SYSID
      2012 proceedings.
- [ ] GCC 14 at CI's `-Og` with `-Wmaybe-uninitialized` (checked over every test, example, and
      benchmark on 2026-11-22) is silent on the sites `-O1` flagged before: the autodiff reset in
      `Reset.h`, the arrival step, `smoothStep`, the hybrid smoother, `FloorModes`, the hybrid
      collapse, and `additiveNoise()`. It flags Eigen's copy of a `Map`'s empty `Stride` in
      `isApprox()`, which the build's `-Wno-maybe-uninitialized` is for, and one false positive:
      `GivenComponent`'s `pinned{space, measurement}` (`ImmFilter.h`), whose members are declared
      in order, seen in the Rao-Blackwellized filters' tests through the temporary built in
      `ImmRecursion`'s update given a block. The flag stays off while Eigen's warning shows; a
      `GivenComponent` named before the call may silence the other.

Performance with no change to results:

- [ ] The Gauss-Hermite rule runs an eigensolver on every transform; compute it at compile time
      without a stored table (a `consteval` bisection on the Sturm count found the roots exactly). The spherical simplex's steps are computed on every transform too. Decide what a
      multiprecision or autodiff scalar takes.

**Breaking**: guide questions whose likely answer breaks code or changes results, to answer first
of the questions the advertised launch waits for (an answer that keeps today's behavior settles
one too):

- [ ] hidden-markov-filter.md Q1 (the chain first in `HiddenMarkovFilter`), Q3 (an emission in
      another scalar), Q6 (`persistent()` takes a probability in one chain and a rate in the
      other).
- [ ] particle-filter.md Q7, Q9, and Q10; Q3 under its unfavored option.
- [ ] rao-blackwellized-filters.md Q1, Q3, Q4, Q8, Q11, Q13, Q14, Q16, and Q19.
- [ ] smoothing.md Q8 (`ValueSequence::values` holds flat indices), whose favored answer is typed
      queries on a flat index, `has<Turn>(space, v)` (what `internal::ValueLocation::matches` does)
      and `indexIn<Cell>(space, v)`, the reverse of `valueOf` and `at<>()`, not a value type; Q1
      under its unfavored option.

## Interests that shape later steps

- **The project boundary.** This library does sequential inference on one state over time, with
  predict and update, continuous, discrete, or hybrid. Work whose difficulty is which measurement
  belongs to which object, or which variable connects to which, is another project that uses this
  one per node: data association and track management in the tracker, Bayes nets and factor graphs
  in the graph library. Single-target probabilistic data association stays here, as clutter is
  mixture noise [additive-noise]. Distributed fusion splits by [distributed-fusion].
- **The organization's projects.** Each is named `Emrys` and a noun for what it provides, never a
  verb: EmrysFilter, EmrysTracker, EmrysAnalysis (a mass noun, for a pipeline). A name mostly
  appears where a noun goes, and EmrysTrack would take the name of the tracker's central type. The
  repository is the kebab-case form, the include directory the name. In dependency order:
  - **EmrysTracker:** data association and track management: JPDA, MHT, and the random finite set
    filters (PHD, CPHD, multi-Bernoulli, PMBM); the variable-structure IMM, whose mode set depends
    on context such as a road network; the node bookkeeping of distributed fusion
    [distributed-fusion]; and the tracking sensor and motion models: polar, spherical, and
    range-Doppler radar, bearings-only, and time difference of arrival measurements, and the
    coordinated turn and Singer motion models.
  - **EmrysAnalysis** [analysis-project]: noise tuning over recorded runs, by expectation-maximization
    (Shumway and Stoffer 1982), innovation correlation (Mehra 1970), and covariance matching; a
    finite-difference check of Jacobians; and fault detection. The online adaptive updates are this
    library's (step 17zg).
  - **EmrysController:** LQR and LQG, iLQR and DDP, belief-space control and sensor scheduling, up
    to what a Riccati sweep computes; model predictive control would take a user's solver. Also
    H-infinity filtering, the dual of H-infinity control on the same Riccati machinery. Its matrix
    bounds the error rather than being its covariance, so it fits none of this library's
    distributions or consistency checks.
  - **EmrysOptimizer:** Gauss-Newton, Levenberg-Marquardt, and dogleg on the spaces, the engine
    under the graph library, nonlinear MPC, and the analysis project's fitting. Also moving horizon
    estimation and estimation under nonlinear state constraints, each an optimization at every
    step.
  - **EmrysGraph:** factor graphs and Bayes nets on Eigen's sparse module and this library's factor
    operations, with incremental smoothing (iSAM) and the SLAM back ends built on it.
  - **EmrysSampler:** MCMC, SMC samplers, Hamiltonian Monte Carlo on the spaces, quasi-random
    sequences, and particle MCMC (PMMH and particle Gibbs) on this library's particle filter
    likelihood.
  - **EmrysNavigation:** dead reckoning and aiding for a vehicle or robot, as domain models on this
    library's groups [lie-groups]. Propagation: inertial mechanization, IMU preintegration, wheel
    odometry (differential drive, Ackermann, skid steer, with scale factor and slip states), and
    legged contact on the invariant EKF (Hartley et al. 2020), from the user's leg kinematics.
    Aiding: GNSS, magnetometers, and the camera models of visual-inertial odometry. Also gravity,
    geodetic frames, and time scales, and Sage-Husa adaptive noise in an INS/GNSS example.
    Manipulator proprioception (joint torque, friction, momentum observers) is not navigation, and
    is out of scope below.
  - **EmrysOrbit:** orbit determination: two-body motion with zonal gravity, drag, and solar
    radiation pressure; the inertial and Earth-fixed frames; and range, range-rate, angle, and
    GNSS-on-orbit measurements. It takes time scales and Earth frames from EmrysNavigation, and
    consider states (step 17v) come from its literature.
  - **EmrysBattery:** state of charge and state of health from equivalent-circuit models (Thevenin
    and two-RC), open-circuit-voltage curves, and hysteresis, with the circuit's parameters
    estimated jointly. Its users are embedded, which is where this library's fixed sizes, `float`,
    and no allocation [allocation] matter most.

  A project fits when it is a state-space or probabilistic problem, reuses the spaces and models,
  has a literature to cite, and runs on Eigen alone. `Core`, `Spaces`, and `Models` split into a
  base project only when a third consumer appears. Not projects: smoothers, hidden Markov models,
  single-target data association, and the per-library metrics. Set aside: Gaussian processes (the
  name collides with process models), fault detection (a directory of the analysis project),
  classical signal processing (whose "filter" would make EmrysFilter ambiguous), and motion
  planning.

## Out of scope

Estimators with users that neither this library nor any planned project will carry:

- **Set-membership estimators** (ellipsoidal, interval, zonotope). They aren't probabilistic, and
  they need an arithmetic of their own (interval arithmetic, zonotope order reduction).
- **The Kalman-Bucy filter**, for measurements taken continuously. Software receives sampled
  measurements, which the continuous-discrete filters (step 17zq) serve.
- **4D-Var.** It needs adjoint models, at the scale of operational weather and ocean
  assimilation. The ensemble Kalman filter (step 17d) is the part of that field's methods that
  fits here.
- **Learned filters** (KalmanNet, deep Kalman filters). There's no Eigen-only implementation, and
  no settled published algorithm [published-algorithms-only].
- **Manipulator proprioception**: joint torque estimation, friction models, and momentum observers
  for collision detection. They need a rigid-body dynamics library, and a momentum observer is
  deterministic, not Bayesian.
- Gaussian processes and classical signal processing, such as the Wiener filter, are set aside
  above.

## Upstream to Eigen

General-purpose pieces Eigen lacks, to offer once ours have settled [eigen-partner].

- **Inverse regularized incomplete gamma** for `unsupported/Eigen/SpecialFunctions`, $P^{-1}(a, p)$
  and $Q^{-1}(a, q)$; `chiSquaredQuantile()`'s safeguarded Newton iteration on `numext::igamma` is
  the prototype.
- **Allocators for dynamic-size matrices**, which is why [allocation] leaves unbounded spaces on
  the global heap. A separate project.
- **Triangularization in place without the orthogonal factor**: `triangularize()` in
  `Core/Triangularization.h` [gaussian-forms], as a `HouseholderQR::computeInPlace()` that discards
  the reflections or a free function.

## Deferred past the advertised launch

- Step 19, the GPU backend, with its API made ready by step 19b.
- Step 18b, counter-based random numbers (decided 2026-11-21): `std::philox4x32` is standardized in
  C++26, so the step waits for a standard library that has it and brings no generator of its own.
- Particle flow as a proposal (Daum and Huang 2007; Li and Coates 2017).
- Teaching aids (decided 2026-11-16): a course syllabus, an ordered path through the examples and
  the module pages that a graduate course can follow, from the constant-velocity Kalman filter to
  the hybrid filters, with the fixed-gain, GPB, and ensemble filters where a course meets them;
  and a Gemini Gem for students, given the syllabus, usage.md, choosing.md, and the examples as
  its knowledge.
- Maybe before the advertised launch, undecided (2026-11-16): the Rao-Blackwellized point-mass
  filter's prediction between cells (Šmídl and Gašperin 2013; Duník et al. 2019): the block moves
  by a convolution of the cell probabilities with its transition density, the rest entering as
  noise on the block (Pocoma et al. 2023, §III-D), which [rb-point-mass] now rejects. It revives
  pruned cells and costs a pass over cell pairs, so it waits for a use whose fixed block moves;
  until then that use is `RbParticleFilter`'s.

## Done

Finished steps, kept so that the step numbers the headers, tests, and documents cite still
resolve.

| Step | What | Decisions |
|---|---|---|
| 1 | Model interface | [templated-measurements], [method-per-measurement], [additive-noise], [weighted-branches], [sensors], [continuous-time-noise], [prediction-inputs] |
| 2 | Kalman filter | [filters-and-recursions], [reduction-policies] |
| 2b | Chi-squared gating and consistency | [composed-features] |
| 3 | Factored unscented transform | [affine-coordinates] |
| 4 | Gaussian mixture filter | [mixture-filter] |
| 5a | Discrete spaces | [discrete-spaces] |
| 5b | IMM | [imm], [switching-models] |
| 6 | Rao-Blackwellized point-mass filter | [rb-point-mass] |
| 7 | Particle filter | [upkeep] |
| 8a | Rao-Blackwellized core and sparse grid state | [rao-blackwellized-core] |
| 8b | Rao-Blackwellized particle filter | [rb-particle-filter] |
| 9 | Bayesian grid filter | [rb-point-mass] |
| 9a | Likelihood-only measurements in the sample-based filters | [likelihood-measurements] |
| 10 | Allocation control | [allocation] |
| 11 | Sizes and dimensions | [sizes-and-dimensions] |
| 12 | Density and probability names | [density-names] |
| 12a | Stored matrix arguments forward | [forwarded-matrices] |
| 12b | Divergences | [divergences] |
| 12c | Uniform mixtures | [uniform-mixtures] |
| 13 | New name, home, and organization | [eigen-partner], [layered-headers] |
| 13b | Reference documentation for the maintainer | [layered-headers], [published-algorithms-only] |
| 13c | Covariance update forms and statistical linearization | [covariance-update-forms] |
| 13d | Fixed-step models | [fixed-step-models], [returned-jacobians] |
| 13e | Covariance reset | [covariance-reset] |
| 14 | Smoothing | [smoothing] |
| 15 | Hidden Markov models | [emissions] |
| 15b | Unit quaternion space | [unit-quaternion] |
| 15c | Component-parameterized owners and recursions | [component-first], [filters-and-recursions] |
| 16 | Information filter | [gaussian-forms] |
| 16d | A linear measurement is a sensor made from its matrix | [linear-sensors] |
| 17 | Square-root filters | [gaussian-forms] |
| 17a | Views check a modification stamp | [view-stamp] |
| 17b | Iterated extended Kalman filter | [iterated-update] |
| 17c | Autonomous multiple model estimator | [autonomous-multiple-model] |
| 17e | Sigma-point sets and the iterated posterior linearization filter | [sigma-point-sets] |
| 17f | Time as a template parameter | [time-steps] |
| 17h | Several resizable blocks in one space | [several-resizable-blocks] |
| 17i | Plain names that evaluate, parameters, and abbreviations | [params-and-density] |
| 17j | A sensor predicts the measurement | [sensors] |
| 17k | The innovation covariance is factored by Cholesky, and an update can fail | [innovation-cholesky] |
| 17l | Marginals read their source, and iterate the elements of a distribution made of them | [element-marginals] |
| 17m | A categorical distribution has parameters | [categorical-params] |
| 17n | Gamma and Uniform are built by factories that check them | [checked-factories] |
| 17o | A Gaussian conditional can return its evidence |  |
| 17p | `wsse()` is `squaredMahalanobisDistance()` |  |
| 17q | A categorical distribution has an entropy |  |
| 17r | The four forms are read alike through their densities | [form-densities] |
| 17s | Policies are made by factories, in namespaces | [policy-factories], [upkeep] |
| 17u | Linearization forms, and `condition()` | [linearization-forms], [covariance-update-forms] |
| 17z | Models as functions with reads and writes: `processModel()`, `measurementModel()`, policies, composition and placement, affine coordinates | [model-policies], [model-placement], [affine-coordinates], [returned-jacobians], [prediction-inputs] |
| 17zb | Whole-state models, function forms, and `transform()` with a measurement model's policies | [whole-state-models], [transform-policies], [model-policies], [sensors] |
| 17zc | A function takes its scalar as a template parameter, `zeros<T>(out)` replaces `vectorIn()`, and the examples name the steps their models apply | [function-scalars] |
| 17ze | A measurement per value of a mode axis, `switchingMeasurement<Path>()`, for the IMM's update | [imm] |
| 17zs | One spelling for what depends on the mode, `byMode<Axis>(when<Value>(part), ...)`: noise by mode axis, a measurement per value, nested, in every filter that holds hybrids; a hybrid filter takes the chain and the motion, `ImmFilter{chain, motion, prior}` | [by-mode], [switching-models], [imm] |
