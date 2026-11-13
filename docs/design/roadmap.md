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

## 17t. Sigma-point axes and repairs — in progress

[sigma-point-axes] records the design. The second differences, which use `along(d)`, are 16b's.
`Distributions/Roots.h` holds the axes, `ridge`, the fallback list, `squareRoot()`, and
`squareRootWithReport()`; `methods::spread()` carries a method's axes to `Transform::root()`,
the factored transform's rest [affine-coordinates], and `Iterated`; and every other root goes through
`internal::defaultRoot()`. Under the default the forms
equal the Kalman filter to rounding (`sqrt_filters.cpp`, `info_filter.cpp`, `iterated_forms.cpp`).

- [ ] An update takes a spread whose axes can fail and reports a failing root as
      `NumericError::NotPositiveDefinite`. Today `Transform::root()` refuses such axes everywhere
      (`refuseFailingAxes()`, whose message names `pivoted()` and `squareRoot()`), since the root is
      taken inside the transform, which returns no `std::expected`; the update would take the root
      of the marked input first. Test: `spread(cubature(), cholesky())` on a singular prior fails
      the update, and on a positive definite one equals `cubature()`.
- [ ] [sigma-point-sets] gives as a reason for sibling methods that "every existing
      `Unscented{0.5}` would have changed", which the new default changed anyway: reword it.

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

## 17za. Mixture arguments of a model — not started

A mixture as an uncertain argument of `predict()` or `observe()` [prediction-inputs], in stages after
the first, which exists (additive mixture noise, [additive-noise]): a mixture through the function;
then several mixtures each naming a mode axis, as `byMode<Axis>()` names one for additive noise
(imm-filter.md Q10); then several memoryless mixtures.
The Kalman filter refuses a mixture argument, and the mixture filters take it, with a branch per
component. Each stage is an addition, so the step may follow the launch; the first two change no
spelling.

- [ ] A mixture through the function: a `GaussianMixtureParams` argument read by the model, each
      component pushed through by the method, the branches weighted by the component weights. Test:
      a `GaussianMixtureFilter` prediction with a two-component input equals the mixture of the two
      predictions with each component as a Gaussian argument.
- [ ] Several mixtures each naming a mode axis, for an `ImmFilter` (imm-filter.md Q10).
- [ ] Several memoryless mixtures, their components crossed.

Done when: each stage passes its test on all three compilers.

## 17zd. Step models — deferred

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
from the old one keeps the present form. It breaks nothing, so it may follow the launch.

Done when: decided, and if built, the examples whose models are steps (`consider_bias.cpp`,
`attitude.cpp`, `unicycle.cpp`) use it, and a test shows a step model predicts as its value form does
under every method.

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

## 17d. Ensemble Kalman filter — deferred

Past the launch: inflation and localization, without which the filter fails at small ensembles,
made it more than the first release needed. Evensen (1994), with perturbed observations (Burgers,
van Leeuwen, and Evensen 1998). `ParticleSet` holds the members and `predict()` already moves them.

- [ ] The stochastic update, with an execution policy and a case in `tests/parallel.cpp`.
- [ ] Inflation (Anderson 2007).
- [ ] Localization by a Schur product with a distance taper (Houtekamer and Mitchell 2001); how a
      space says which coordinates are near which is the one new decision.
- [ ] The deterministic variants (ensemble square-root and adjustment filters) as methods.
- [ ] Tests: at large $N$ the moments match `KalmanFilter` on an affine model; localization
      recovers the Kalman covariance on a banded problem at small $N$; `tests/allocation.cpp`.

Done when: the filter, inflation, and localization pass on all three compilers.

## 17g. Posterior Cramér-Rao bound — in progress

[cramer-rao-bound]: the recursion and its tests are done.

- [ ] A singular $Q$ (a noiseless block): Section IV of Tichavský, Muravchik, and Nehorai (1998).
- [ ] An example of the bound on the Models module page, beside simulate() and nees().

Done when: the singular case is decided and passes its tests on all three compilers.

## 18. Execution policies — in progress

[execution-policies]; the loops and the reasoning are in parallelism.md. Every loop takes a policy.

- [ ] Benchmark [benchmarks]: thread scaling of the ranging particle filter
      (`benchmarks/particles/`) at $10^3$ to $10^6$ particles, as the ratio to `Sequential`. Under
      `Sequential` it is 1.04 to 1.07 times the hand loop's instructions.
- [ ] `Gaussian::logDensity()` recognizing a diagonal or scalar covariance: with an isotropic $R$,
      the triangular solve costs 45 instructions per particle more than a closed form.

Done when: the scaling curve is published.

## 18b. Counter-based random numbers — not started

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

## 19. GPU backend — not started

A `Device` policy runs the step 18 primitives on a GPU, with the particles in device memory
through the allocator [allocation] and the per-element bodies compiled as device code. The
experiments of parallelism.md, "What to find out first", decide whether the step is worth it.

- [ ] Experiments: a device-only syntax check of one particle filter with clang CUDA; the ranging
      benchmark under nvc++ `-stdpar=gpu` against `std::execution::par`, for the crossover count.
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
      93 open questions (decided 2026-11-12), and the move continues. When the guides are empty, any
      question still open moves here. The guide README's table tracks both.
- [ ] A user page listing every circumstance in which the library may allocate: a count growing
      past its reserve, the first use of count-sized scratch [allocation], the standard algorithms
      under a policy other than `Sequential` [execution-policies], and anything sized by an
      unbounded space. Audit the headers for it rather than writing it from memory.
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
- [ ] `KalmanFilter` names its update's type only through its recursion,
      `decltype(filter)::RecursionType::UpdateType<M>`; give the filter the alias too (from the
      Kalman filter review).
- [ ] `Gate::operator()` declares its result as what the inner policy returns for the update's own
      `Count`, and calls the inner policy on a `std::dynamic_extent` update when it drops
      branches, so a user's policy whose result type depends on `Count` fails inside `Gate`. None
      of the built-in policies does (from the Kalman filter review).

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
      trips). Left: `-Wshadow -Wconversion -Wsign-conversion -Wold-style-cast` in the test flags
      (the only header warnings are the kinematic tables in `ProcessModel.h` and three sites in
      `Reduction.h`, which index `std::array` with `int`), and a release build of the suite in CI:
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
advertised launch, the posts under "Getting the word out" below, follows it. The guide questions
marked **breaking** at the end of this section stayed open past the soft launch (decided
2026-11-13): each answered goes into a 0.x release with its `CHANGELOG.md` entry. The vcpkg and
Conan submissions wait for the advertised launch (decided 2026-11-12); whether GitHub Discussions
does is undecided.

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
- [ ] The DOI Zenodo minted for 0.1.0, in `CITATION.cff` and the README.
- [ ] Build cost for contributors: time and memory per workflow preset and the `-j` warning, with
      `-ftime-trace` numbers per header (`tools/compile_cost.py --match` times any compile
      command, and `--time-trace` adds Clang's totals).
- [ ] Shorter test builds: `-ftime-trace` on the slowest test files (`tools/compile_cost.py
      --match tests/ --time-trace` on a test build finds them), then explicit instantiation of
      the setups the tests repeat (`extern template` in a support header, instantiations in their
      own translation units), and a precompiled header if parsing matters. Record a clean build's
      time per preset before and after. This is also the profiling the deferred module asks for.
- [ ] A ThreadSanitizer run of `tests/parallel.cpp` by hand.
- [ ] What a precondition failure does, on the user's page: `EMRYS_FILTER_ASSERT` is `eigen_assert`,
      which aborts, and `NDEBUG` or `EIGEN_NO_DEBUG` silences it. A user's own definition may be an
      expression or a statement (`tests/assert_override.cpp`); a throwing one terminates inside a
      parallel loop body and across the containers' `noexcept` moves, such as `ParticleSet`'s. The
      other way an exception can come out is `std::bad_alloc` from a count-sized container's
      allocator when exceptions are on.

Getting the word out, once the front door is done: the README's comparison table and its numbers
are what a reader checks in the first hour, and a thread that finds them wrong is the launch.

- [ ] `CODE_OF_CONDUCT.md`, before the advertised launch.
- [ ] `SECURITY.md`, before the advertised launch, with private vulnerability reporting turned on
      (GitHub offers it only on a public repository). The library has no files, sockets, parsing,
      or credentials, so the policy is about the code it compiles into a program: report
      privately undefined behavior that a call meeting every documented precondition reaches, a
      crash on an input documented as returning a `std::expected` error, and packaging that
      fetches or runs something other than it declares; report as a bug a diverging filter, a
      failed `EMRYS_FILTER_ASSERT` precondition (gone under `NDEBUG`, so the caller's to meet),
      and defects in Eigen or a compiler. Fixes go onto `dev` only while the library is pre-1.0.
- [ ] The paywalled sources read, before the advertised launch: the 34 entries that
      `docs/reference/bibliography.md` marks **Not read.** (2026-11-12), which comments credit
      while taking their numbers from a free source. For each, check the code against the original
      as the bibliography's opening says, cite its own sections and equations where the code
      follows it, and delete the sentence. Every algorithm is cited, and a reader who checks a
      citation against the paper should find what the comment says there.
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
      DOI above, since "every algorithm cited, nothing invented here" and documentation that can
      accompany a graduate course are what JOSS reviews for, and a paper is what puts a library
      on a syllabus; and `docs/research.md`'s open problems sent to advisors in the ISIF and
      FUSION community directly, since each is scoped for a graduate student.
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
- [ ] `Roughen` of a Rao-Blackwellized state copies the particle set on every call; `drawSamples()`
      of a gamma, `drawParticleGaussian()`, the fitting, and the samplers' batches build owning
      sample sets, which a bounded space shouldn't need.
- [ ] `GridGaussianParams::subdivide()`, `PointGrid::subdivided()`, and `collapseGivenPoints` allocate
      from the global heap (rao-blackwellized-filters.md Q18); `PoissonEmission`,
      `ExponentialEmission`, and `ArrivalInterval` copy their rates into every measurement
      (hidden-markov-filter.md Q5).
- [ ] `moments()` of a `ParticleSet` allocates count-sized temporaries under every policy: take
      scratch from the policy's allocator, or say that statistics allocate.
- [ ] The smoothers refactorize each step's predicted covariance on every `smooth()`, and the
      information form builds the covariance where the root suffices. Left as they are, with
      reasons: the histories copy the filtered estimate into the smoothed slot at `push()`;
      `MixtureFitter` allocates per iteration; the measurement steps of FormSteps.h factorize the
      read block again after `Transform::root()`; `Transform` evaluates the function once more to
      size a resizable output.

Decisions and comments that disagree with the code:

- [ ] Tests: `LearnsAVelocityFromRanges` doesn't check the position's consistency
      (rao-blackwellized-filters.md §9).
- [ ] The page numbers of Särkkä, Bunch, and Godsill (2012) in the bibliography, from the SYSID
      2012 proceedings.
- [ ] GCC 14 at `-O1 -Wall` warns `-Wmaybe-uninitialized` where Clang is silent: the autodiff reset
      in `Reset.h`, the arrival step, `smoothStep`, the hybrid smoother, `FloorModes` (silent on
      2026-11-04, since it builds the modes from log-weights), the hybrid collapse, and the noise
      built in `additiveNoise()`.
      Check CI's `-Og` build, and silence them at the source if they show.

Performance with no change to results:

- [ ] `predictByTransition()` and the Rao-Blackwellized cached prediction multiply the whole stored
      covariance, where only the lower triangle is guaranteed; mirror first, as
      `predictInPlaceByJacobian()` does.
- [ ] The Gauss-Hermite rule runs an eigensolver on every transform; compute it at compile time
      without a stored table (a `consteval` bisection on the Sturm count found the roots exactly). The spherical simplex's steps are computed on every transform too. Decide what a
      multiprecision or autodiff scalar takes.
- [ ] `internal::printCount()` pluralizes by adding "s"; the first noun that doesn't, such as
      "vertex", adds a defaulted plural argument.

**Breaking**: guide questions whose likely answer breaks code or changes results, open at the soft
launch and to answer first after it (an answer that keeps today's behavior settles one too):

- [ ] hidden-markov-filter.md Q1 (the chain first in `HiddenMarkovFilter`), Q3 (an emission in
      another scalar), Q6 (`persistent()` takes a probability in one chain and a rate in the
      other), and Q7 (the shape of `SwitchingModel` before it takes a continuous chain).
- [ ] particle-filter.md Q7, Q9, and Q10; Q3 under its unfavored option.
- [ ] rao-blackwellized-filters.md Q1, Q3, Q4, Q8, Q11, Q13, Q14, Q16, and Q19.
- [ ] smoothing.md Q8 (`ValueSequence::values` holds flat indices), whose favored answer is typed
      queries on a flat index, `has<Turn>(space, v)` (what `internal::ValueLocation::matches` does)
      and `indexIn<Cell>(space, v)`, the reverse of `valueOf` and `at<>()`, not a value type; Q1
      under its unfavored option.

## Accepted, not yet sequenced

### Invariant extended Kalman filter and pose spaces

Barrau and Bonnabel (2017): an EKF on a Lie group whose error is the group error, which converges
from large initial errors on group-affine dynamics ($SE(3)$, $SE_2(3)$). It is `KalmanFilter` with
`Linearized` on a group space whose `plus()` and `minus()` apply the increment on a chosen side, so
it is an example and a test, not a header. It needs the perturbation side on the unit quaternion
(only this library can add it), the groups as spaces (a `Composite` is a product manifold, not the
group), and a test that the error Jacobian is constant on group-affine dynamics. The groups go in
`Spaces/` [lie-groups]; add the side to the quaternion first. Its largest user base is legged
robots, by contact-aided estimation (Hartley et al. 2020), so sequence it before EmrysNavigation
starts.

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
    library's ("Deferred past the first release").
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
  measurements, which the continuous-discrete filters (deferred, below) serve.
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

## Deferred past the first release

- Student-t robust updates for Kalman-family filters. Student-t noise isn't conjugate, so it needs
  an approximation (a variational update, or reweighting $R$ by the Mahalanobis distance). Particle
  and grid filters take it now through a measurement's own `logLikelihood()`, and mixture noise is
  the robust model until then. Beside it, the Huber M-estimator update, as an iterated reweighting
  of $R$ on the `Iterated` machinery.
- Adaptive noise inside the filter: the variational Bayes estimate of $R$ (Särkkä and Nummenmaa
  2009), and Sage-Husa (1969) with fading memory and a guard that keeps its estimates positive
  definite, against its known divergence. Estimating the noise over a recorded run is the analysis
  project's (Interests, above).
- A process model's transition density $p(x_{k+1} \mid x_k)$, which no process model exposes now.
  It is the one prerequisite of six features: the auxiliary particle filter (Pitt and Shephard
  1999), the locally optimal proposal (Doucet, Godsill, and Andrieu 2000), EKF and UKF proposals
  (van der Merwe et al. 2000), resample-move with a Metropolis-Hastings move (Gilks and Berzuini
  2001), particle flow as a proposal (Daum and Huang 2007; Li and Coates 2017), and particle
  smoothing by backward simulation (Godsill, Doucet, and West 2004).
- The smoothers smoothing.md lists as absent: the two-filter smoother (Fraser and Potter 1969)
  through `InfoFilter`, the iterated smoothers (Bell 1994; García-Fernández, Svensson, and Särkkä
  2017), smoothers for the Rao-Blackwellized and grid filters, and smoothing a mixture of IMMs.
- Two more rungs of choosing.md's ladder: the second-order EKF, on step 16b's Hessians, and the
  divided-difference filters DD1 and DD2 (Nørgaard, Poulsen, and Ravn 2000), as methods beside the
  sigma-point sets.
- Fixed-gain filters: the steady-state Kalman filter, its gain found by running the filter's own
  covariance recursion to convergence, and the alpha-beta and alpha-beta-gamma filters, with the
  closed-form gains of the built-in kinematic blocks (Kalata 1984).
- State constraints: projection onto linear equality constraints (Simon and Chia 2002) and
  truncation at linear inequality constraints (Simon and Simon 2010), as a step after the update.
  Nonlinear constraints are the optimizer's, by moving horizon estimation.
- GPB1 and GPB2, the generalized pseudo-Bayesian estimators beside the IMM and the AMM, for the
  course companion.
- Continuous-discrete filtering: a continuous-time block that integrates a user's ODE by
  fixed-step RK4, differentiates it by autodiff, and discretizes its noise by Van Loan (1978), each
  stage applied by `plus()`; the moment equations of the continuous-discrete EKF and UKF (Särkkä
  2007); and smoothing between measurement times.
- Noise tuning over recorded runs and a finite-difference check of Jacobians belong to the
  analysis project [analysis-project]; the library supplies the log-likelihoods, innovations
  [innovation-statistics], and `withJacobian()`.
- A C++23 module beside the headers [layered-headers], after the test-build profiling above.
- A filter's description naming its model, its method, and its last update (accepted, NIS), after
  its name and state [printing-and-simulation].
- The minimum compilers. The launch states only what has been tried, as the README's Install
  section does: CI passes on GCC 14, Clang 21, Apple Clang 21, and the MSVC of Visual Studio 2026;
  GCC 13 builds the README example and the examples but isn't tested; Clang 18 on libstdc++ 13
  fails at the first `std::expected`. Untried: Clang 19 and 20 (which the Conan recipe's
  `validate()` rejects untested), older MSVC and Apple Clang, and each compiler on the other
  standard libraries. Closing it means a table by compiler and standard library, `validate()`
  matching it, and a check naming the floor before the first `std::expected`, the only message a
  vcpkg user would get.
- The Rao-Blackwellized point-mass filter's prediction between cells (Šmídl and Gašperin 2013;
  Duník et al. 2019): the block moves by a convolution of the cell probabilities with its
  transition density, the rest entering as noise on the block (Pocoma et al. 2023, §III-D), which
  [rb-point-mass] now rejects. It revives pruned cells and costs a pass over cell pairs, so it
  waits for a use whose fixed block moves; until then that use is `RbParticleFilter`'s.

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
