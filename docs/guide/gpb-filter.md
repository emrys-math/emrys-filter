# Generalized pseudo-Bayesian filter

Review notes on the filter that keeps a Gaussian for each sequence of the last N modes of a
switching model. `GpbFilter` is the generalized pseudo-Bayesian (GPB) estimator of order N:
GPB1 at order one, GPB2 at order two, which econometrics knows as Kim's filter. The
document covers the space of the sequences, the filter's two steps, how its state is read, how
the filter stands beside the IMM, and what it lacks. Questions are marked **Q*n*** and collected
at the end.

Headers reviewed: `Spaces/ModeSequence.h`, `Filters/GpbFilter.h`, `HybridUpdateSteps`
in `Filters/ImmFilter.h`, and `overModeSequences()` in `Distributions/Hybrid.h`. Tests:
`tests/Filters/gpb_filter.cpp`, and the cases `MODE_SEQUENCE_EMPTY` in
`tests/Spaces/compile_errors.cpp` and `GPB_FIXED_STEP_METHOD` in
`tests/Filters/compile_errors.cpp`. Roadmap step 17zn.

This is the review of a first implementation, written with it, on the branch `mode-sequences`
(2026-11-22), which starts from the branch `cleanup`. Nothing here has landed on `next`, and no
decision of it is in decisions.md. What was checked is in "What was checked" at the end, with
what was not: no workflow preset has run.

Every example below was compiled and run with Clang 21 and GCC 14, in one scratch program outside
the repository, and the outputs in comments are the printed results. The two print the same
digits; no example draws a random number. The compiler errors quoted are Clang's.

The examples share these declarations:

```cpp
struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
using State = Composite<Position, Velocity>;
struct Cruise {};
struct Turn {};
struct Maneuver : Named<Cruise, Turn> {};

double const dt = 1.0;
Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};
auto const cruise = ConstantVelocity<Position, Velocity>{0.01};
auto const turn = ConstantVelocity<Position, Velocity>{4.0};
auto const motion = byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(turn));
auto const chain = ContinuousTimeMarkovChain<Maneuver>::persistent(0.05);
auto const radar = measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(R));
auto const initial = GaussianParams<State>{
    State{}, Eigen::Vector4d{0, 0, 10, 0}, Eigen::Matrix4d{Eigen::Vector4d{10, 10, 4, 4}.asDiagonal()}};
auto const prior = HybridGaussianParams<State, Maneuver>{CategoricalParams<Maneuver>{}, initial};
```

## Decisions so far

The maintainer decided one thing on 2026-11-22, after the first implementation:

- **The filter is `GpbFilter`.** No published name says more about the algorithm than
  "generalized pseudo-Bayesian", so the filter takes the name a reader will search for, as
  `ImmFilter` and `AmmFilter` do, and `Gpb` joins the closed list of abbreviations in
  style.md. Its recursion is `GpbRecursion`, and its template parameter `Order`, the
  literature's word for N. The first implementation called it `ModeSequenceFilter`.

Two things are given:

- Roadmap step 17zn, as the branch `cleanup` words it, asks for GPBn with n a template
  parameter, on `SwitchingModel`, with the IMM left as its own filter. The implementation
  follows that.
- The same step planned a state of the $M^{n-1}$ merged Gaussians, with a state type of its own
  for n = 1. The implementation departs from that, for the reason in §3, and Q2 asks whether
  the departure stands.

The algorithm is the GPB(N) of
[Hashimzade et al. (2024)](../reference/bibliography.md#hashimzade2024), §2.2.2, Algorithm 1,
who credit [Bar-Shalom, Li, and Kirubarajan (2001)](../reference/bibliography.md#barshalom2001);
§2.2 of the same paper says that the filter of [Kim (1994)](../reference/bibliography.md#kim1994)
is GPB(2). The section and algorithm numbers were read from the paper's arXiv page on
2026-11-22. Bar-Shalom, Li, and Kirubarajan was not read.

## 1. What the filter keeps

A system whose mode follows a Markov chain over $M$ modes, each mode with its own linear
model, has an exact posterior after $k$ steps: a mixture with one Gaussian for each of the
$M^k$ sequences of modes the system may have taken. No filter can keep it. Every practical
filter keeps the Gaussians that some recent part of the sequence distinguishes, and replaces
the sequences that agree in that part by one Gaussian with their mean and covariance.

`GpbFilter` keeps the Gaussians that the last $N$ modes distinguish, $M^N$ of them.
Its state is a `Hybrid`, the type an `ImmFilter` holds, over the discrete space
`ModeSequence<ModeSpace, N>`: a weight and a Gaussian for each sequence
$(m_{k-N+1}, \dots, m_k)$.

| Order | Gaussians in the state | Predictions and updates per step | Name in the literature |
|---|---|---|---|
| 1 | $M$ | $M$ | GPB1 |
| 2 | $M^2$ | $M^2$ | GPB2, Kim's filter |
| N | $M^N$ | $M^N$ | GPB(N) |

The literature counts the Gaussians after the merge of §3, $M^{N-1}$, where this table counts
the state's.

## 2. The space of the sequences

`ModeSequence<ModeSpace, N>` is the product of $N$ copies of the mode space, the oldest mode
first and the current one last. The copy of the mode $k$ steps back is tagged
`Lagged<ModeSpace, k>`, and `Current<ModeSpace>` is `Lagged<ModeSpace, 0>`. It derives from the
`DiscreteProduct` of the copies, so everything that reads a product reads it: `valueOf()`, a
categorical's `marginal<Axis>()` and `probability<Value>()`, and a hybrid's `collapse<Axis>()`.

```cpp
using Pairs = ModeSequence<Maneuver, 2>;
auto const pairs = Pairs{};
pairs.cardinality();                                                                       // 4
valueOf<TagPath<Lagged<Maneuver, 1>, Cruise>, TagPath<Current<Maneuver>, Turn>>(pairs);  // 1
pairs.currentOf(1);       // 1: the current mode of sequence 1 is Turn
pairs.following(1, 0);    // 2: after (Cruise, Turn), the mode moving to Cruise gives (Turn, Cruise)
ModeSequence<Maneuver, 3>{}.cardinality();                                                 // 8
```

The current mode varies fastest in the flat index. With $M$ modes, the sequences that differ
only in their oldest mode are $o M^{N-1} + r$ for each oldest mode $o$, and the sequence that
follows them into the mode $j$ is $r M + j$. The prediction's loops are written on these two
facts.

A value's bare name is ambiguous in a sequence of two or more modes, since each copy has it,
and the existing message says what to write:

```
Spaces/Discrete.h:498:9: error: static assertion failed due to requirement
'countValue<Turn, Emrys::ModeSequence<Maneuver, 2>>() < 2': the value names a value of more than one
axis; qualify it with a TagPath to its axis
```

> **Q1.** The names of the space. The filter's name is decided (Decisions so far). Still working
> names are `ModeSequence`, its axis tags `Lagged` and `Current`, and `overModeSequences()`. They
> name a discrete space and not the algorithm, so "GPB" does not suit them. The literature's
> word for the sequence is "history" (Bar-Shalom's "mode history"; Hashimzade et al., "the
> history of regimes in N consecutive periods"). The library already uses `History` for the
> record a smoother reads (`GaussianHistory`, `HybridGaussianHistory`), which is why the space
> is not `ModeHistory`. The alternative is `ModeHistory` with `overModeHistories()`, accepting
> the two senses of the word. I recommend the names as built.

## 3. The steps

`predict()` does two things. It first merges by moments the $M$ Gaussians whose sequences
differ only in their oldest mode, each weighted by its sequence's weight, which leaves
$M^{N-1}$. It then moves each merged Gaussian through every mode's process model, which gives
the $M^N$ Gaussians of the sequences one step on. The sequence that drops $m_{k-N+1}$ and
appends $j$ gets the weight

$$w'(m_{k-N+2}, \dots, m_k, j) = \sum_{m_{k-N+1}} w(m_{k-N+1}, \dots, m_k) \, \Pi_{m_k j},$$

with $\Pi$ the chain over the step. With sequences of one mode the sum runs over the current
mode, and the weight is $\sum_i w_i \Pi_{ij}$, the chain's prediction of the mode. One loop
serves every order.

`update()` updates each sequence's Gaussian by the measurement and multiplies the sequence's
weight by the likelihood. It is the IMM's update, unchanged, now in `HybridUpdateSteps`, a base
class of `ImmRecursion` and of `GpbRecursion`. A measurement that depends on the mode
[by-mode] names the axes of the mode space, and each sequence reads them in its current mode.

```cpp
auto filter = GpbFilter{chain, motion, overModeSequences<2>(prior)};
for (Eigen::Vector2d const & z : {Eigen::Vector2d{10.3, 0.4}, {19.6, 0.1}, {30.2, 2.2}, {39.5, 6.1}})
{
    filter.predict(dt);
    auto const updated = filter.update(observe(radar, z));
    updated->logLikelihood();   // -4.5892, -3.7808, -3.9471, -5.1313
}
```

The source merges after the update and keeps the $M^{N-1}$ merged Gaussians; this filter keeps
the $M^N$ updated ones and merges at the start of the next prediction. The reason is that
`predict()` and `update()` are separate calls. Between them the filter holds the $M^N$
predictions, since each must meet the measurement under its own mode: merging them before the
update would average away the differences between the modes' process models, which is what the
likelihoods tell apart. A state of $M^{N-1}$ Gaussians cannot hold them, so the filter would
keep them in scratch that `state()` does not show, and `state()` after a prediction would not
be the predicted distribution. With the merge moved, the state is the predicted distribution
after `predict()` and the posterior after `update()`, as every filter's is.

Three things follow from it.

- With one update between predictions, the filter and the source produce the same numbers: the
  same merged Gaussians at each prediction, and the same mean and covariance over all sequences
  after each update, since merging by moments in two stages keeps what merging at once keeps.
  `MatchesTheSourceAtOrdersOneToThree` checks it against the source's form written in plain
  Eigen, to 1e-10, for orders 1, 2, and 3.
- A second update before the next prediction conditions each of the $M^N$ Gaussians again,
  under its own sequence. The source's form would have merged between the two.
- Order one needs no state type of its own. The roadmap's plan had one Gaussian with weights
  over the current mode for n = 1, because after the source's merge that is what is left.

> **Q2.** Is the state the $M^N$ updated Gaussians, merged at the start of `predict()`, as
> built? The alternative is the roadmap's plan: a state of $M^{N-1}$ Gaussians, the $M^N$
> predictions held in the filter's scratch between `predict()` and `update()`, and a state type
> of its own for N = 1. I recommend the state as built.

> **Q3.** `HybridUpdateSteps` is a public class template, since `ImmRecursion`'s documented
> `update()` members are now its members, and the published reference leaves out the `internal`
> namespace. Is a public base class the right home, or should the update be a class that both
> recursions hold? Two of its names still say IMM: the result type `ImmFilterUpdate`, whose
> `modes` holds one update per sequence here, and the message of a policy that keeps several
> branches, which this filter's user reads as
> `an IMM filter keeps one component per mode, so its policy must keep one: Collapse, Select, or a Gate over them`.

## 4. The initial state, and reading the state

`overModeSequences<N>(prior)` makes the initial state from a hybrid over the current mode:
each sequence gets its current mode's Gaussian, and its current mode's weight shared equally
among the sequences that end in it. The earlier modes' weights do not matter, since the first
prediction merges over the oldest mode Gaussians that are equal, and no later step reads them.
The filter's order is deduced from the state's type. A hybrid over the mode space itself is
refused, by the message every failed deduction gives:

```
error: no viable constructor or deduction guide for deduction of template arguments of 'GpbFilter'
```

> **Q4.** Should the filter take the prior over the current mode and the order, in place of a
> state the caller spreads with `overModeSequences()`? An order cannot be given to class
> template argument deduction beside deduced arguments, so the choices are a tag argument,
> `GpbFilter{sequencesOf<2>, chain, motion, prior}`, a factory function, or the state
> as now. The message above is the cost of the state as now: it does not say what to write.

The state is read as an IMM's is, through the hybrid:

```cpp
auto const density = filter.state().toDensity();
density->collapsed().mean();                          // 39.5757 5.4513 9.6309 2.9381
auto const sequences = filter.state().modes().toDensity();
sequences->probability(1);                            // 0.0367: (Cruise, Turn)
sequences->probability<TagPath<Lagged<Maneuver, 1>, Cruise>, TagPath<Current<Maneuver>, Turn>>();  // 0.0367
auto const now = filter.state().modes().marginal<Current<Maneuver>>();
now.toDensity()->probability<Turn>();                 // 0.6033
auto const before = filter.state().modes().marginal<Lagged<Maneuver, 1>>();
before.toDensity()->probability<Turn>();              // 0.5963
auto const perMode = density->collapse<Current<Maneuver>>();   // a Gaussian per mode now, as an IMM keeps
perMode.conditional(1).mean()(3);                     // 3.6179, the turn mode's sideways velocity
```

The marginal over `Lagged<Maneuver, 1>` is the probability of the mode one step back given the
measurements up to now, which is a one-step smoothed mode probability. The filter computes it
as a by-product.

Printing names the filter and nests the hybrid, whose lines call each sequence a mode:

```
generalized pseudo-Bayesian filter
  hybrid of 4 modes of dimension 4
    mean: 39.5757 5.45134 9.63086 2.93809
    ...
    mode 1, probability 0.0367425:
```

> **Q5.** Should the filter have accessors for the two things most callers want, the mode
> probabilities now and the Gaussian per mode now, in place of
> `state().modes().marginal<Current<Maneuver>>()` and
> `state().toDensity()->collapse<Current<Maneuver>>()`? No other filter has such accessors;
> the IMM's state gives both directly.

## 5. Beside the IMM

Both filters merge Gaussians by moments, and they differ in which ones and with which weights.

The IMM keeps one Gaussian per mode. Before predicting the mode $j$ it merges all $M$ of them,
weighted by $w_i \Pi_{ij}$: each Gaussian's weight times the chain's probability of moving from
its mode to $j$. It merges once for each $j$, so each mode starts from its own merged Gaussian.

This filter with sequences of one mode merges the same $M$ Gaussians once, weighted by $w_i$
alone, and predicts that one Gaussian through every mode's model. It costs what the IMM costs,
$M$ predictions and $M$ updates, and it uses less: it ignores that a Gaussian whose mode
seldom moves to $j$ should count for less in the start of $j$. The IMM is the better filter of
that size, which is the literature's finding and the reason the IMM replaced GPB1.

With sequences of two or more modes the two rules coincide. The Gaussians this filter merges
share their current mode $m_k$, so the chain's factor $\Pi_{m_k j}$ is the same for all of
them and cancels from the merge's weights. Weighting by it, as the IMM does, changes nothing.
So this filter with sequences of $N \ge 2$ modes is the IMM run on the chain of the last $N$
modes, the chain that moves $(m_{k-N+1}, \dots, m_k)$ to $(m_{k-N+2}, \dots, m_k, j)$ with
probability $\Pi_{m_k j}$. `SequencesOfTwoModesAreTheImmOnTheChainOfPairs` shows it for
$N = 2$ with the library's own `ImmFilter` over a product of two mode axes and that chain: every
Gaussian and probability agrees to 1e-12 over five steps.

That equality is derived here and tested; it is not taken from a source. Hashimzade et al.
define a second family, IMM(N), in §2.2.3, Algorithm 2, and describe the two families as
differing in where the reduction happens ("In the IMM, mixing is done after the state update
and before the measurement update, and in the GPB, collapsing is done after the state and
measurement updates", footnote 3). By the argument above their IMM(N) and GPB(N) are the same
filter for $N \ge 2$ when the mode is a Markov chain, and differ only at $N = 1$. I did not
find that statement in the part of the paper that was read.

The same fixes through the four filters, as each update's log-likelihood:

```cpp
auto one = GpbFilter{chain, motion, overModeSequences<1>(prior)};
auto imm = ImmFilter{chain, motion, prior};
auto two = GpbFilter{chain, motion, overModeSequences<2>(prior)};
auto three = GpbFilter{chain, motion, overModeSequences<3>(prior)};
//  fix   one mode     IMM          two modes    three modes
//  1     -4.589160    -4.589160    -4.589160    -4.589160
//  2     -3.856567    -3.788799    -3.780802    -3.780802
//  3     -4.028011    -3.952627    -3.947060    -3.946875
//  4     -4.570902    -5.086176    -5.131288    -5.131530
```

From one Gaussian every filter is exact at the first fix. Sequences of two modes are exact
through the second and of three through the third, which
`IsExactForAsManyStepsAsASequenceHoldsModes` checks against the enumeration of every mode
history: the filter with sequences of $N$ modes is the exact posterior for its first $N$ steps.
The source's form, merged after the update, is exact for $N - 1$.

On the IMM test's maneuvering target (a cruise, ten steps of sideways acceleration at
2 m/s², another cruise), the root mean square position error over steps 11 to 60, over 2000
runs that give every filter the same truth and noise, with Clang and MSVC's library:

| Chain rate | One mode | IMM | Two modes | Three modes |
|---|---|---|---|---|
| 0.05 | 1.148 | 0.975 | 0.948 | 0.939 |
| 0.2 | 1.163 | 1.076 | 1.009 | 0.976 |

The order is the one the literature reports. The gain is in the cruise before and after the
maneuver. Within the ten steps of the maneuver the four errors at rate 0.05 were 1.263, 1.256,
1.259, and 1.259, which differ by less than the sampling error. The test
`RanksWithTheImmOnAManeuverOnAverage` asserts the order of the first three over 200 runs, with
margins of 10% and 1% where the measured differences are 18% and 3%; it passes with GCC's
library too, which draws other normals from the same seeds.

> **Q6.** Is the filter with sequences of one mode worth keeping? It is GPB1, which the roadmap
> asks for, and it costs no code: it is the general loop at N = 1. It is never the filter to
> choose, since the IMM costs the same and does better, so its use is in a course and in a
> comparison. If it stays, choosing.md should say so in one sentence.

> **Q7.** Should the IMM and this filter be one recursion? One loop with the merge weighted by
> $w \, \Pi$ gives the IMM at N = 1 and this filter at every $N \ge 2$; the same loop weighted
> by $w$ alone gives GPB1 at N = 1. That would make an `ImmFilter` over a `ModeSequence` state
> the filter with longer sequences, and GPB1 a variant of the IMM, in place of a second filter
> type. Against it: the `ImmRecursion` has cases this one lacks (the identity chain that skips
> the merge, the square-root form predicted in the state, the step recorded for the smoother),
> its state type follows its model's mode space, and the roadmap says the IMM stays its own
> filter. I recommend two filters now and this question open until the smoother of Q9 shows
> how much the two share.

## 6. What the filter lacks beside the IMM

> **Q8.** A recursion of the user's own. `ImmFilter` takes its recursion as a template
> parameter and deduces it from `ImmFilter{recursion, state}` [user-extension-points];
> `GpbFilter` always runs `GpbRecursion`. The recursion meets the `Recursion`
> concept, so the parameter is an addition of about sixty lines of deduction guides. The same
> question covers a `MixtureFilter` over these hybrids, which needs one specialization of
> `internal::RecursionFor`, and the Rao-Blackwellized filters' updates given a block, which the
> shared update already has but nothing calls through this recursion. None is built or tested.

> **Q9.** A smoother. The filter has no `predictJoint()`. Kim's smoother is defined on GPB(2)
> (Hashimzade et al., §2.3, Algorithm 3), and `HybridGaussianHistory` already cites Kim (1994)
> for the IMM's mode probabilities, so the smoother of this filter is a known algorithm with a
> home. What a recorded step holds for sequences is not designed.

> **Q10.** Allocation. The steps use the IMM's pattern: scratch from the state's allocator,
> exchanged with the state's components, and a reused result. `tests/allocation_mixtures.cpp`
> has no case for this filter, so the claim that its steps allocate nothing on a bounded space
> is untested. The prediction also builds two arrays of $M^N$ log-weights on the stack for a
> fixed mode space and on the heap for one sized at run time, as the IMM's does.

> **Q11.** The cost has no guard. $M^N$ is computed at compile time as an `int` for a fixed
> mode space, and four modes with sequences of sixteen overflow it. A `static_assert` on the
> product would need a negative compile case; a bound on the order would be arbitrary.

> **Q12.** The measurement reads the current mode only. A measurement that depends on the
> previous mode, such as one taken during the switch, cannot be written, since `byMode<Axis>()`
> resolves against the mode space and not against the sequence. Nothing in the literature read
> asks for it; the question is whether `byMode<Lagged<Maneuver, 1>>()` should mean something
> here or fail as it does now, with the message that the filter's mode space does not have the
> axis.

## What was checked

- The new test file, 11 tests, was compiled and run alone with Clang 21 on Windows (MSVC's
  library), GCC 14 in WSL, and MSVC 19.50, with warnings as errors, and passes with all three.
- A deliberate error in the merge's weights (equal weights in place of the sequences') fails
  four of the tests: the comparison with the source, the comparison with the IMM on the chain
  of pairs, the update by the current mode, and the maneuver.
- The existing test files that reach the restructured `ImmFilter.h`, each compiled and run
  alone with Clang: `imm_filter` (18 tests), `amm_filter` (4), `mixture_filter` (6),
  `rb_particle_filter` (28), `rb_point_mass_filter` (29), `sqrt_filters` (20),
  `Smoothers/smoothing` (24), `update_failure` (19), `time_steps` (8), `filter_upkeep` (10),
  `fixed_step_binding` (6), and `allocation_mixtures` (13). All pass. `imm_filter` also passes
  with GCC 14, and `imm_filter`, `amm_filter`, and `rb_particle_filter` with MSVC.
- The two new negative compile cases, and the seven cases of `ImmFilter.h` about a measurement by
  mode and the policy, four of whose claims moved into the base class, were compiled with Clang,
  and each fails with the message it names.
- clang-format, `check_layers.py`, `check_decisions.py`, and `check_compile_errors.py` (372 of
  372 claims have a case).

- Doxygen 1.18, run with the main checkout's configuration over this branch's headers, gives no
  warning, and `check_policies.py` passes over its XML.

Not checked: no workflow preset has run, so the rest of the suite, the examples, and the
benchmarks have not been built against the changed headers.

## Questions

1. Q1: the names of the space, its axis tags, and `overModeSequences()`; the filter's name is
   decided, `GpbFilter`.
2. Q2: the state as the $M^N$ updated Gaussians, merged at the start of `predict()`, or the
   roadmap's $M^{N-1}$.
3. Q3: `HybridUpdateSteps` as a public base class, and the two names in it that say IMM.
4. Q4: the initial state from `overModeSequences()`, or the order given to the filter.
5. Q5: accessors for the mode probabilities now and the Gaussian per mode now.
6. Q6: whether sequences of one mode, GPB1, stay.
7. Q7: one recursion for the IMM and this filter.
8. Q8: a recursion of the user's own, a `MixtureFilter` over these hybrids, and the updates
   given a block.
9. Q9: a smoother, and what a recorded step holds.
10. Q10: the allocation test.
11. Q11: a guard on $M^N$.
12. Q12: a measurement that depends on an earlier mode.
