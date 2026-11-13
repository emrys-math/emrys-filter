# Design reviews

The documents here review the library one area at a time, after the first plateau of features.
They are written for the maintainer and Claude, and later serve as source material for user
documentation. The review tends to turn into API changes. Decisions made while reviewing are
recorded in each document's "Decisions so far" section and, where they change the architecture,
in [../design/decisions.md](../design/decisions.md).

## Documents

They are listed in reading order, and each builds on the ones before it.

| # | Document | Status |
|---|---|---|
| 1 | [imm-filter.md](imm-filter.md): mode chains, hybrid states, mixing, Markov outliers, and the AMM | not yet reviewed; refreshed for roadmap 17z (models by `compose()`, error quotes rerun, §11's floor corrected to the 0.001 it printed); Q3, Q5, Q8, Q10 [imm], Q11, Q13, Q17 decided; Q4, Q7, Q9, Q14 done; Q1, Q2, Q6, Q12, Q15, Q16 open; Q18 open |
| 2 | [hidden-markov-filter.md](hidden-markov-filter.md): the forward recursion, emissions, continuous-time chains, and modulated arrivals | written; not yet reviewed; refreshed for roadmap 17z (the `Sensor` tag restored, since the library no longer defines one; examples not rerun); Q4 decided; Q8 half settled (`simulate()` takes a chain); Q9 no longer reproduces; Q1 to Q3, Q5 to Q8, Q10 open |
| 3 | [particle-filter.md](particle-filter.md): sampled prediction, likelihood weighting, resampling, and roughening | not yet reviewed; refreshed for roadmap 17z (models and likelihood-only measurements rewritten, §1, §3, §5, §11, §14 rerun with Clang; §8's tenth roughening not reproduced); Q6 [likelihood-measurements], Q8 [allocation], Q12, Q15 decided; Q1 to Q5, Q7, Q9 to Q11, Q13, Q14, Q16 open |
| 4 | [rao-blackwellized-filters.md](rao-blackwellized-filters.md): point-mass, grid, and particle | not yet reviewed; refreshed for roadmap 17z (§6 and the misuse table rewritten, the prediction ignoring `affineIn()` where it refused `FactoredUnscented`, error quotes rerun with Clang); Q2, Q7, Q15, Q18 [allocation], Q27 [rb-particle-filter] decided; Q11 partly decided; Q1, Q3 to Q6, Q8 to Q10, Q12 to Q14, Q16, Q17, Q19 to Q26 open |
| 5 | [smoothing.md](smoothing.md): the record of the forward pass and the backward passes over it | written; not yet reviewed; refreshed for roadmap 17z (models rewritten, §3 to §5 and §6's IMM columns rerun with Clang); Q4 decided; Q1 to Q3, Q5 to Q11 open; Q12 open |

When you write or finish reviewing a document, update its status here. A finished review moves
out, as the roadmap's Documentation section says, and is deleted: Core and Spaces were the first,
on 2026-11-01, their explanation now in the headers' documentation and module pages, their reasons
in decisions.md, and their one open question in the roadmap. Distributions followed on 2026-11-04,
with no question open, its summaries on the Distributions module page and its unrecorded reasons
in decisions.md ([covariance-repair], [conditional-evidence], [random-draws]). Transforms
followed on 2026-11-09: its methods table and input rules are on the Transforms module page, the
table of the linearization forms on `OutputWithCross`, the axes design in [sigma-point-axes], and
its one open question, Q13 (the square-root parameters as uncertain inputs), decided on
2026-11-12 in [gaussian-forms].
Models followed on 2026-11-11, with every question decided: the explanation is on the Models
module page and in the headers, the unrecorded reasons in decisions.md (Q1 and Q19 in
[continuous-time-noise], Q12 in [weighted-branches], Q17 in [fixed-step-models], Q25 in
[measurement-space-functions], Q28 and Q30 in [affine-coordinates], Q40 in [model-placement], and
Q11 in [transform-policies]), and two observations of its smaller notes in the roadmap.
Mixtures and samples followed on 2026-11-12, with every question decided: the explanation is
in the headers of `Component.h` to `Fitting.h` and on the Distributions module page, and the
unrecorded reasons in decisions.md (Q3 and Q6 in [mixture-entries], Q4 and Q7 in
[params-and-density], and Q5 in [upkeep]).
The Kalman filter and the shape of a filter followed on 2026-11-12, with every question decided:
the explanation is in `KalmanFilter.h`, the form filters' headers, `FormSteps.h`, `Reduction.h`, and
`Consistency.h`, and on the Filters module page, which gained the table of what rejection does in
each filter; the unrecorded reasons in decisions.md (Q6 and Q11 in [composed-features]), and two
observations of its smaller notes in the roadmap. The later filter reviews describe the shape they
share by those headers and decisions.
Divergences followed on 2026-11-12, with every question decided: the explanation is in
`Divergence.h` and on the Divergences module page, which gained the clamping and rounding of the
closed forms and how a merge lowers the collapse bound by its cost, and the rule in
[divergences].
The Gaussian mixture filter followed on 2026-11-12, with every question decided but Q10, splitting
by nonlinearity, deferred to research.md item 6: the explanation is in `MixtureFilter.h`, whose class
comment gained when `Keep` gains over `Collapse`, the collapse bound as a test of the moments, and
the gate as a commitment, and in `Reduction.h`, whose `Select` says why it keeps the whole
likelihood; and the rules in [mixture-filter] and [reduction-policies].
Upkeep followed on 2026-11-12, with every question decided but Q11, `Roughen`'s copy of a
Rao-Blackwellized particle set, which is in the roadmap's allocation items: the explanation is in
`Upkeep.h`, whose module page gained the table of the operations each state takes, what keeping
the state's type means, what each action's return means, and what each allocates, and in
`FilterUpkeep.h`; the rules in [upkeep], [rao-blackwellized-core], [autonomous-multiple-model], and
[covariance-repair], and the reasons for Q6 and Q10 in [upkeep]'s rejections.

## How a document is written

Follow these rules:

- Open with what the document covers, which headers it reviews, and the commit its examples
  were run against.
- A "Decisions so far" section comes next. It starts empty or with decisions already in
  decisions.md that the reader should take as given, and grows as the review settles questions.
- Numbered sections then go through the area piece by piece. Each says what a piece does and
  shows real use.
- Compile and run every example with Clang and GCC (MSVC as well for template-heavy code) before
  quoting it, and show the printed results in comments. Keep example lines to 120 columns, moving
  a result comment above its statement where it doesn't fit. Write scratch programs outside the
  repository rather than adding them to the tests. Quote compiler errors where a misuse should
  fail, since the error messages are part of the API.
- Raise questions inline as quoted **Q*n*** blocks, at the point where a reader would wonder,
  and collect them in a numbered list at the end. Each document numbers its own questions from
  Q1. Once the review has started, keep section and question numbers stable: mark settled
  questions "decided" and strike them through in the list rather than renumbering.
- Describe what the code does now, checked against the headers and tests, not what the roadmap
  intended. Where a document disagrees with decisions.md, say so as a question.
- Cite every algorithm's source from [../reference/bibliography.md](../reference/bibliography.md)
  as a link to its entry, lowercased (`[Runnalls (2007)](../reference/bibliography.md#runnalls2007)`),
  with a sentence on what the code takes and where it departs. Take equation and section numbers
  only from a source that was read; the bibliography's opening gives the form for an original
  that was not. The entries are Doxygen `\anchor`s, so on GitHub the link opens the page but does
  not jump to the entry.
- Avoid repeating another document's material. Cover a piece in one place and point to it from
  the others.

## Scope of each document

Headers are in `include/EmrysFilter/`, in the layer directories of [layered-headers] (`Core/`,
`Spaces/`, `Distributions/`, `Models/`, `Filters/`), and tests in `tests/`, mirrored. The generated
reference (roadmap step 13b: `cmake --build build/docs --target docs`, then
`build/docs/docs/html/index.html`) has a module page per layer, with the subgroups Allocation and
Resamplers under Core, Discrete under Spaces, Upkeep and Divergences under Distributions, and
Transforms under Models; a review starts from the module page and reads the headers in the order
their `\file` comments give. Every test file opens with a scope comment and every test with a
comment saying what it verifies, so the tests are a map of the evidence. The decisions are in
[../design/decisions.md](../design/decisions.md), and the roadmap notes in
[../design/roadmap.md](../design/roadmap.md) give the test evidence for each step.

### 1. Interacting multiple model filter

The hybrid state (`Hybrid`, `HybridGaussian`), switching process models by mode axis
(`SwitchingModel`), `MarkovChain` and its Kronecker-product construction, mixing,
measurement-side outlier modes, and the autonomous multiple model filter, the IMM on the identity
chain. `Spaces/Discrete.h` covers the mode spaces; `FloorModes` is `Upkeep.h`'s, the IMM smoother
smoothing.md's, and mixtures of IMMs `MixtureFilter.h`'s.

- Decisions: [switching-models], [imm], [component-first], [filters-and-recursions],
  [autonomous-multiple-model].
- Headers: `Filters/ImmFilter.h`, `Filters/AmmFilter.h`, `Distributions/Hybrid.h`,
  `Models/SwitchingModel.h`, `MarkovChain` in `Distributions/Categorical.h`.
- Tests: `tests/Filters/imm_filter`, `amm_filter`; `tests/allocation_mixtures`.

### 2. Hidden Markov filter

The forward recursion over a `CategoricalParams`, emission models (`EmissionTable`, `DensityEmission`,
`PoissonEmission`, `ExponentialEmission`), the continuous-time chain and its transition over an
interval, and Markov-modulated arrivals observed by the prediction (`ArrivalInterval`).
`MarkovChain` is imm-filter.md's, `CategoricalParams` `Distributions/Categorical.h`'s, and the smoother
smoothing.md's.

- Decisions: [emissions]; [discrete-spaces], [filters-and-recursions], [smoothing],
  [fixed-step-models], [time-steps] as given.
- Headers: `Filters/HiddenMarkovFilter.h`, `Models/Emission.h`,
  `Distributions/ContinuousTimeMarkovChain.h`.
- Tests: `tests/Filters/hidden_markov_filter`, `tests/Models/emission`,
  `tests/Distributions/continuous_time_markov_chain`; the hidden Markov cases in
  `tests/Smoothers/smoothing`, `tests/Filters/time_steps`, and `tests/allocation`.

### 3. Particle filter

Propagation with sampled process noise, weighting by likelihood, `logLikelihoods()` at many
points, ESS, the resamplers (multinomial, stratified, systematic, residual), roughening, and the
execution-policy overloads. `ParticleSet` itself is `Distributions/ParticleSet.h`'s.

- Decisions: [filters-and-recursions], [upkeep], [allocation], [fixed-step-models],
  [execution-policies], and [time-steps].
- Headers: `Filters/ParticleFilter.h`, `Core/Resampling.h` (the Resamplers subgroup of the Core
  module page).
- Tests: `tests/Filters/particle_filter`, `tests/Core/resampling`, `tests/allocation_particles`,
  `tests/parallel`.
- It covers the statistics of resampling and roughening; `Upkeep.h` covers the `Resample`,
  `Roughen`, and `WhenEss` objects as actions.

### 4. Rao-Blackwellized filters

The shared Rao-Blackwellized core, `PointGrid` and `GridCells`, the sparse `GridGaussianParams`
state, the Rao-Blackwellized point-mass filter, the Bayesian grid filter (`PointMassFilter`) as
its case with nothing left Gaussian, and the Rao-Blackwellized particle filter with
`ParticleGaussianParams`. Why there is no transition kernel between cells; the update given a block;
and these filters' policies, time steps, allocation, result views, and printing.

- Decisions: [rb-point-mass], [rao-blackwellized-core], and [rb-particle-filter]; [allocation],
  [covariance-update-forms], [fixed-step-models], [returned-jacobians], [covariance-reset],
  [execution-policies], [component-first], [filters-and-recursions], [view-stamp], [time-steps],
  [printing-and-simulation].
- Headers: `Distributions/Conditionals.h`, `Distributions/GridGaussian.h`,
  `Distributions/ParticleGaussian.h`, `Filters/RbUpdate.h`, `Filters/RbPointMassFilter.h`,
  `Filters/RbParticleFilter.h`, and `given()` in `Models/Measurement.h`.
- Tests: `tests/Filters/rb_point_mass_filter`, `rb_particle_filter`, and their cases in
  `tests/allocation_particles`, `tests/parallel`, `tests/Filters/time_steps`.
- The prune and `Subdivide` actions are `Upkeep.h`'s.

### 5. Smoothing

The record the caller keeps of the forward pass and the backward pass over it: `predictJoint()`
and `PredictedJoint`, `smoothStep()`, `GaussianHistory` with `Lag` (fixed-interval and
fixed-lag), `HybridGaussianHistory` for the IMM and AMM, `GaussianMixtureHistory` over
`MixtureLineage`, `ParticleHistory` (ancestor tracing), `CategoricalHistory` with
`transitionPosterior()` and `mostProbableSequence()`, and what has no smoother.

- Decisions: [smoothing]; [emissions] for the hidden Markov smoother; [allocation],
  [covariance-reset], [execution-policies].
- Headers: `Smoothers/` (the Smoothers module page), and the `predictJoint()` members and joint types in
  `Models/ProcessModel.h`, `Filters/KalmanFilter.h`, `ImmFilter.h`, `MixtureFilter.h`, and
  `HiddenMarkovFilter.h`.
- Tests: `tests/Smoothers/smoothing`, `mixture_smoothing`, and the smoothing cases in
  `tests/unequal_dimensions_filters`, `allocation`, `allocation_mixtures`,
  `allocation_particles`, and `parallel`.
