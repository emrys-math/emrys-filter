# Smoothing

Review notes on the Smoothers layer: the record of a filter's forward pass that the caller
keeps, and the backward passes that turn it into smoothed estimates. Each section says what a
piece does, shows what using it looks like, and raises questions marked **Q*n***, which are
collected at the end. §1 to §5 cover the Kalman family, which the others are built from; §6 to §9
the IMM, Gaussian mixture, particle, and hidden Markov smoothers; §10 allocation and execution
policies; §11 what has no smoother.

Headers reviewed: `Smoothers/` (`Lag.h`, `GaussianHistory.h`, `HybridGaussianHistory.h`,
`GaussianMixtureHistory.h`, `ParticleHistory.h`, `CategoricalHistory.h`), with the `predictJoint()` members it records from in
`Models/ProcessModel.h` (`PredictedJoint`), `Filters/KalmanFilter.h`, `Filters/ImmFilter.h`
(`HybridPredictedJoint`), `Filters/MixtureFilter.h` (`GaussianMixturePredictedJoint`), and
`Filters/HiddenMarkovFilter.h` (`CategoricalPredictedJoint`), and `MixtureLineage` in
`Distributions/Mixture.h`. Tests: `tests/Smoothers/smoothing.cpp` and `mixture_smoothing.cpp`, with
the quaternion case in `tests/unequal_dimensions_filters.cpp`, the allocation cases in
`tests/allocation.cpp`, `allocation_mixtures.cpp`, and `allocation_particles.cpp`, and the policy
cases in `tests/parallel.cpp`. Decisions: [smoothing], with [emissions] for the hidden Markov
smoother, [allocation] for allocation, [covariance-reset] for the tangent-space covariances, and
[execution-policies] for policies. Roadmap: step 14 (including the mixture smoother, PR #55) and the
smoothing part of step 15.

The examples were refreshed for roadmap 17z (`next`, 2026-11-11), which replaced the
model API: the process models here are the built-in `ConstantVelocity` itself, where they were
`ProcessModel{...}` around it. The examples of §3 to §6 were recompiled and run with Clang 21 (MSVC
STL) and printed what they printed on 2026-11-04, but for three differences of rounding size in §3
and §5, which are measured here over every smoothed mean and covariance entry. The mixture,
particle, and hidden Markov examples of §7 to §9, whose code changed only in the declaration of
`model` they share with §3, were not rerun, nor was GCC. The refresh of 2026-11-04 (`next`), after that of 2026-10-27, compiled and ran them
with Clang (MSVC STL) and GCC (w64devkit); outputs shown in comments are the printed results, the
same on both compilers except where a section says otherwise. That refresh found the Kalman, mixture,
and hidden Markov examples unchanged but for rounding in §3 and §5. The IMM smoother's mode
probabilities are now the expectation correction of Barber (2006) (§6), which needs a
prediction of each mode through every other mode's model, so §6's examples were rerun, the
maneuver one on a track this refresh defines. A kept `HybridPredictedJoint` can no longer be
written `auto joint = HybridPredictedJoint<...>{}` (Q12), and a fixed-step model is spelled
`fixedStep<State>(model, dt)` [space-storage]. The particle example of §8 was rerun on a track and
seed it now states. The filters themselves are reviewed in
[imm-filter.md](imm-filter.md), [particle-filter.md](particle-filter.md), and
[hidden-markov-filter.md](hidden-markov-filter.md), and the mixture filter in `MixtureFilter.h`;
this document covers only what they record and what is done with it.

## Decisions so far

These are in decisions.md [smoothing] and are taken as given here.

- **A smoother is an observer, not a filter feature [smoothing], [composed-features].** The caller
  keeps a history outside the filter and runs the backward pass over it. The filters' states and
  updates are unchanged; the one addition on the filter side is `predictJoint()`, a prediction that
  returns the step's joint instead of discarding where it started.
- **The joint is the currency.** A `PredictedJoint` holds the Gaussian the step started from, the
  prediction, and their cross-covariance $C_k$, which every transform method produces, Unscented
  included. Carrying the starting Gaussian in the joint makes the history one entry per
  prediction, plus the latest posterior at smoothing time.
- **Updates need no recording.** Any number of updates, of any method, between two predictions
  enter the history only through the posterior that the next `predictJoint()` starts from.
- **Fixed-interval and fixed-lag are one type.** A history without a `Lag` keeps every step; with
  `Lag{L}` it keeps the newest $L$ and recycles its oldest node, so a full window allocates
  nothing. The history is a `std::list` from the caller's allocator, since nothing reads it by
  index, and each step holds a slot for its smoothed estimate, so there is no result object.
- **The backward step is Rauch-Tung-Striebel in the tangent space**, written with the
  cross-covariance, so the same step serves every prediction method and manifold states.
- **The IMM smoother is the backward IMM of Nadarajah (2009)**, with mode-matched predictions
  recorded per mode, and its mode probabilities by the expectation correction of Barber (2006)
  under its mean approximation, from a prediction of each mode's filtered Gaussian through every
  mode's model (Q4); the **mixture smoother is the merging-based smoother of Rahmathullah,
  Svensson, and Svensson (2014)** over a lineage the mixture keeps; the **particle smoother is
  ancestor tracing**, told of resamples by `ParticleSet::resampleCount()`; the **hidden Markov
  smoother** is the same backward equation on a finite set, with Viterbi as a free function
  [emissions].
- **Deferred:** particle smoothing by backward simulation (it needs a transition density no
  process model exposes), the Rao-Blackwellized smoothers, and smoothing a mixture of IMMs. All
  three are open questions in decisions.md.

## 1. The record: `predictJoint()`

Every smoother needs two things from the forward pass at each step: the posterior the step
started from and the prediction it made, with their cross-covariance
$C_k = \mathrm{Cov}(x_k, x_{k+1} \mid z_{1:k})$. `PredictedJoint<Space, Scalar>` is that
aggregate:

```cpp
template <Space Space_, RealScalar Scalar_ = double>
struct PredictedJoint
{
    GaussianParams<Space_, Scalar_> filtered;   // p(x_k | z_{1:k}), the state the step started from
    GaussianParams<Space_, Scalar_> predicted;  // p(x_{k+1} | z_{1:k}), process noise included
    MatrixType cross;                     // C_k, a row per difference coordinate of x_k
};
```

It comes from three places:

- the free `predictJoint(method, prior, model, dt, inputs...)`, and `predictJoint(prior,
  fixedStepModel, inputs...)` for a `FixedStepModel`, each input matched by its tag to what the model
  reads [prediction-inputs];
- `KalmanFilter::predictJoint(dt, inputs...)` (or `predictJoint(inputs...)` over a fixed-step
  model), which moves the filter's state exactly as `predict()` does and returns the joint;
- the IMM, mixture, and hidden Markov filters' own `predictJoint()`, which return their own joint
  types (§6, §7, §9).

The cross-covariance comes from `transform()` with the state marked `withCross()`
(the Transforms module page), so it exists for every method, including those with no
Jacobian. For an affine step it is $P_{k \mid k} F^T$; the §3 example checks that it is exactly
that (`cross = P F^T: 0.0e+00`), and the test `PredictJointHoldsThePriorThePredictionAndTheirCross`
checks it under Linearized and Unscented to 1e-12, and `PredictJointWithAFixedStepModelAndAFilter`
through a fixed-step model's cached transition.

## 2. The backward step: `smoothStep()`

```cpp
GaussianParams<Space, Scalar> smoothStep(PredictedJoint<Space, Scalar> const & joint,
                                   GaussianParams<Space, Scalar> const & smoothedNext);
```

returns $p(x_k \mid z_{1:N})$ from the step's joint and $p(x_{k+1} \mid z_{1:N})$. With the gain
$G_k = C_k P_{k+1 \mid k}^{-1}$,

$$\hat x_{k \mid N} = \hat x_{k \mid k} \boxplus G_k \,(\hat x_{k+1 \mid N} \boxminus \hat x_{k+1 \mid k}), \qquad
P_{k \mid N} = P_{k \mid k} + G_k \,(P_{k+1 \mid N} - P_{k+1 \mid k})\, G_k^T.$$

This is the smoother of [Rauch, Tung, and Striebel (1965)](../reference/bibliography.md#rauch1965)
in the form of the Gaussian smoother of [Särkkä (2013)](../reference/bibliography.md#sarkka2013),
chapter 10, which writes the gain with the cross-covariance instead of a transition matrix, so the
affine, linearized, unscented, and other sigma-point predictions all use the same step. The
departure is the manifold: the residual is `minus()` and the step `plus()`, and each covariance is
moved into the tangent space it is combined in, with the reset Jacobians of [covariance-reset]
(`Spaces/Reset.h`). $P_{k+1 \mid N}$ is carried back to the tangent space at the predicted
mean, and the smoothed covariance, formed at the filtered mean, is carried to the smoothed mean. On
a space whose reset is the identity both moves do nothing. $P_{k+1 \mid k}$ is factorized by LDLT,
which tolerates a semidefinite prediction, and the result is symmetrized. The gain depends on the
joint alone, so each history computes it once when it records a step (`internal::rtsGain()`, kept in
the step as `gain`, or `gains` per mode or component) and `smoothStep(joint, gain, smoothedNext)`
applies it at every `smooth()`; the two-argument `smoothStep(joint, smoothedNext)` computes it for
one step.

`smoothStep()` is public because the IMM and mixture smoothers call it once per mode and per
(child, parent) pair, and a caller can use it to build a smoother of their own over another record.

## 3. Fixed-interval smoothing: `GaussianHistory`

`GaussianHistory<Space, Scalar, Allocator>` keeps one `Step` per prediction: the `joint` as
`predictJoint()` returned it and a `smoothed` Gaussian, which is a copy of `joint.filtered` until
`smooth()` runs. `push(joint)` records; `smooth(latest)` takes the filter's current state, which
is its own smoothed estimate, and fills every step's slot from the newest back. The steps are
iterated oldest first.

The example runs a constant-velocity Kalman filter over six position fixes and checks every
smoothed mean and covariance against the textbook RTS smoother written in plain Eigen with the
same $F$, $Q$, $H$, and $R$:

```cpp
struct Position : Linear<1> {};
struct Velocity : Linear<1> {};
using State = Composite<Position, Velocity>;

auto const model = ConstantVelocity<Position, Velocity>{0.2};
auto const initialCov = Eigen::Matrix2d{{4.0, 0.0}, {0.0, 1.0}};
auto const initial = GaussianParams<State>{State{}, Eigen::Vector2d{0.0, 1.0}, initialCov};

auto filter = KalmanFilter{model, initial};
auto history = GaussianHistory<State>{};
static_cast<void>(filter.update(fix(zs[0])));       // zs: 0.1 1.3 1.9 3.2 4.1 4.8; R = 0.5
for (std::size_t k = 1; k < zs.size(); ++k)
{
    history.push(filter.predictJoint(1.0));
    static_cast<void>(filter.update(fix(zs[k])));
}
history.smooth(filter.state());

for (auto const & step : history)
{
    use(step.joint.filtered, step.smoothed);
}
// k=0 filtered p=0.0889  smoothed p=0.1512 v=0.9944  var p 0.4444 -> 0.2906
// k=1 filtered p=1.2475  smoothed p=1.1403 v=0.9793  var p 0.3757 -> 0.1599
// k=2 filtered p=2.0109  smoothed p=2.1110 v=0.9680  var p 0.3802 -> 0.1482
// k=3 filtered p=3.1178  smoothed p=3.0743 v=0.9505  var p 0.3578 -> 0.1496
// k=4 filtered p=4.1092  smoothed p=4.0041 v=0.9095  var p 0.3432 -> 0.1656
// largest difference from plain-Eigen RTS: 5.0e-16
// cross = P F^T: 0.0e+00
```

The history has five steps for six fixes: the last posterior, $p(x_5 \mid z_{0:5})$, is
`filter.state()`, passed to `smooth()` and not stored. The test
`FixedIntervalMatchesBatchLeastSquares` goes further: over eight steps of a 2-D track it matches
the posterior of the whole trajectory by batch least squares, 36 coordinates built and inverted in
plain Eigen, to 1e-9 in every step's mean and marginal covariance.

The record holds no measurements, and several updates between two predictions need nothing
recorded. Running the same track with a second fix at each time (`fix(z + 0.2)`) still records
five steps (`two fixes per step: 5 steps recorded, first smoothed p = 0.2450`).

What `smooth()` cannot tell is whether its record is whole. A `filter.predict(dt)` between two
`push()` calls, or a `latest` that is not the state after the newest record, gives smoothed
estimates without any sign that they are wrong. The mixture history asserts against both through
its lineage's count (§7), and the particle history through `resampleCount()` (§8); the Gaussian
and categorical histories have no such check.

> **Q1.** `GaussianHistory` and `CategoricalHistory` can't tell a record with a missing step, or a
> `latest` that doesn't follow the newest step, from a whole one. Should every joint carry a
> count that the filter advances on each prediction (as `MixtureLineage::count()` does for the
> mixture), so that `push()` and `smooth()` can assert continuity, or is documenting the contract
> enough?

The histories are spelled space first, `GaussianHistory<State>`,
`HybridGaussianHistory<State, Mode>`, `GaussianMixtureHistory<State>`, while [component-first] and
[filters-and-recursions] made the filters and the hybrid and mixture types component first
(`KalmanFilter<Gaussian<State>, ...>`, `MixtureFilter<Component, ...>`). The spelling matches the
`HybridGaussian` and `GaussianMixtureParams` aliases, but nothing generic in the component exists to
alias.

> **Q2.** Should the histories follow [component-first]/[filters-and-recursions], for example
> `History<GaussianParams<State>>` with `GaussianHistory` as the alias, so that a smoother generic
> in its component has a name? Today each history is tied to Gaussian components, so the question is
> only about the spelling, unless Q7 adds other components.

## 4. Fixed-lag smoothing: `Lag`

`GaussianHistory<State> window{Lag{L}}` keeps the newest $L$ steps. Once full, `push()` splices the
oldest node to the back and overwrites it, so the window allocates nothing after its first $L$
steps. `lag()` returns the $L$ as a `std::optional` (empty for a fixed-interval history). Smoothing
after every update gives the lagged estimate $\hat x_{k-L \mid k}$ at `front()`. As
[Särkkä (2013)](../reference/bibliography.md#sarkka2013), chapter 10, describes it, this is the
backward pass over the window at every step, $O(L)$ RTS steps each time. The history keeps no
smoothed estimate from one call to the next, since each new measurement changes all of them.

```cpp
auto lagged = KalmanFilter{model, initial};
auto window = GaussianHistory<State>{Lag{2}};
static_cast<void>(lagged.update(fix(zs[0])));
for (std::size_t k = 1; k < zs.size(); ++k)
{
    window.push(lagged.predictJoint(1.0));
    static_cast<void>(lagged.update(fix(zs[k])));
    window.smooth(lagged.state());
    use(window.front().smoothed);
}
// after z_1: 1 steps, front p = 0.1355
// after z_2: 2 steps, front p = 0.1649
// after z_3: 2 steps, front p = 1.1244
// after z_4: 2 steps, front p = 2.1086
// after z_5: 2 steps, front p = 3.0743
// window equals the fixed-interval pass bitwise: yes; lag() = 2
```

The window's last smoothing is bitwise the fixed-interval pass at the same steps (3.0743 is §3's
smoothed `k=3`), because the backward step at $k$ reads only the steps after it. The tests check
the same for every history (`FixedLagMatchesFixedIntervalOverTheWindow`, its IMM and mixture
counterparts, and `HiddenMarkovFixedLagMatchesFixedIntervalOverTheWindow`).

## 5. Methods, models, and spaces

The history doesn't see the method. A filter under Unscented or a sigma-point set of
[sigma-point-sets], or over a `FixedStepModel`, records the same `PredictedJoint`, and on an affine
model all give the Linearized filter's smoothed estimates to rounding:

```cpp
auto cubature = KalmanFilter{model, initial, methods::cubature()};
auto const cached = fixedStep<State>(model, 1.0);
auto stepped = KalmanFilter{cached, initial};            // records with stepped.predictJoint()
// Unscented vs Linearized: 3.9e-16
// Cubature vs Linearized:  3.9e-16
// FixedStepModel vs the model itself: 1.1e-16
```

An `Iterated` method [iterated-update] and the IPLF [sigma-point-sets] are measurement methods, so
they change only the posteriors the joints start from. A smoother over such a filter is RTS over the
iterated filter. It is not an iterated smoother, which would relinearize the prediction about the
smoothed estimates; the library has none.

On a manifold the step works in the tangent space (§2). `KalmanSmootherStepsInTheTangentSpace` in
`tests/unequal_dimensions_filters.cpp` smooths an attitude on a unit quaternion: every smoothed
quaternion is unit, every covariance has the seven difference coordinates, no smoothed covariance
exceeds its filtered one once both are at the filtered mean, and the first attitude's error falls.

The square-root and information forms [gaussian-forms] have no `predictJoint()`, since the history
holds the covariance form. A history over a form is a constraint failure:

```cpp
auto filter = SqrtKalmanFilter{model, toSqrt(initial).value()};
history.push(filter.predictJoint(1.0));
// clang: error: no matching member function for call to 'predictJoint'
//        note: because '!IsForm' evaluated to false
// gcc:   note: the expression '!Emrys::KalmanFilter<...>::IsForm [...]' evaluated to 'false'
```

> **Q3.** A form filter can't be smoothed at all. Is that the intended scope for 0.1 (the forms
> exist for conditioning and information fusion, and a smoother over them would be a square-root
> RTS, a new algorithm), or should a form filter's `predictJoint()` record in the covariance form,
> converting at the record, so that its user can smooth at the cost of forming covariances?
> Either way, the error could say why: today it names only `!IsForm`.

## 6. The IMM smoother: `HybridGaussianHistory`

`ImmFilter::predictJoint(dt, joint, inputs...)` fills a
`HybridPredictedJoint<State, Mode>`: the mode probabilities before the step, one `PredictedJoint`
per mode, the chain's matrix, which the caller may change between steps, and
`throughOtherModels`, each mode's filtered Gaussian moved through every other mode's model.
`HybridGaussianHistory<State, Mode>` records copies of it, each with a smoothed `HybridGaussian`,
and `smooth(filter.state())` runs the backward pass. The joint is filled in place so that one kept
joint serves the whole run. Each mode's joint comes from a prediction the IMM doesn't otherwise
make: mode $j$'s own filtered Gaussian moved through mode $j$'s model, not the IMM's prediction from
the mixed Gaussian. With the cross predictions the mode probabilities need, that costs $n^2$
predictions for $n$ modes on top of the filter's.

Each backward step runs the IMM in reverse. With $\mu_{t \mid t}$ the filtered mode probabilities
and $\Pi$ the chain:

- the backward transition $b_{ij} = p_{ji}\, \mu^j_{t \mid t} / e_i$, with
  $e_i = \sum_l p_{li}\, \mu^l_{t \mid t}$;
- for each mode $j$ at $t$, the smoothed Gaussians at $t+1$ moment-matched under the backward
  mixing weights $\mu^{i \mid j} \propto b_{ij}\, \mu^i_{t+1 \mid N}$;
- `smoothStep()` from mode $j$'s filtered Gaussian, through its mode-matched joint, to that mixture;
- $\mu^j_{t \mid N} = \sum_i \mu^i_{t+1 \mid N}\, p(r_t = j \mid r_{t+1} = i, z_{1:N})$, with
  $p(r_t = j \mid r_{t+1} = i, z_{1:N}) \propto b_{ij}\, \mathcal{N}(\hat x^i_{t+1 \mid N};
  \hat x^{j \to i}_{t+1 \mid t}, P^{j \to i}_{t+1 \mid t})$ normalized over $j$, where
  $\hat x^{j \to i}_{t+1 \mid t}, P^{j \to i}_{t+1 \mid t}$ is mode $j$'s filtered Gaussian moved
  through mode $i$'s model, which `predictJoint()` records for every pair
  (`HybridPredictedJoint::predictedThrough()`).

This is the backward IMM smoother of [Nadarajah (2009)](../reference/bibliography.md#nadarajah2009),
§8.2.1, equations 8.11 to 8.17 for the Gaussians, published as
[Nadarajah, Tharmarasa, McDonald, and Kirubarajan (2012)](../reference/bibliography.md#nadarajah2012).
It needs no backward-time dynamics, unlike the two-filter IMM smoother of Helmick, Blair, and
Hoffman (1995), and runs one RTS step per mode rather than one per pair of modes. The code departs
from the source in three ways, recorded in the header and in [smoothing]:

- it reads the source's "predicted mean and covariance of mode $j$" as mode $j$'s own prediction,
  which the derivation (8.2) and the gain (8.17) need;
- it takes the RTS covariance update with a plus sign where (8.16) prints a minus;
- it writes the step with the cross-covariance and in the tangent space, as `smoothStep()` does.

The weights are kept as logarithms. A mode at $t$ that no smoothed mode at $t+1$ can have come
from keeps its filtered Gaussian, and where no mode's prediction gives the smoothed mean at $t+1$ a
density, $b_{ij}$ is used unweighted
(`ImmSmootherLeavesTheBackwardTransitionWhereNoPredictionHasADensity`). The final normalizations,
which valid weights always pass, are assertions.

The tests show that with one mode it is the Kalman smoother to 1e-12, and that over two affine
modes every quantity matches a plain-Eigen transcription of 8.11 to 8.17 and of the expectation
correction to 1e-10. That transcription checks that the code is the source. On a target that turns
at step 20, the smoothed maneuver probability rises over steps 17 to 22, 0.998 at the turn's last
step (`ImmSmootherPlacesAManeuverWhereTheFilterLags`), and over nine fixes of a slowly turning
target, 512 mode sequences, it is within 0.03 of enumeration
(`ImmSmootherModeProbabilitiesTrackEnumeration`).

The example below checks the source against the exact posterior on a harder track. A 1-D target
moves at 1 m/s and then at 3 m/s from step 6, seen as $z_k = p_k + 0.4 \sin(2.3 k)$ with
$R = 0.5$, under a quiet and an agile constant-velocity mode with
`MarkovChain<Maneuver>::persistent(0.95)`. Over thirteen steps the exact posterior enumerates all
$2^{13}$ mode sequences, with a Kalman filter and RTS smoother for each sequence, where $r_k$ is
the model of the step into $k$:

```cpp
auto filter = ImmFilter{
    SwitchingModel{MarkovChain<Maneuver>::persistent(0.95),
                   ConstantVelocity<Position, Velocity>{0.01},
                   ConstantVelocity<Position, Velocity>{4.0}},
    HybridParams{CategoricalParams<Maneuver>{}, GaussianParams<State>{State{}, Eigen::Vector2d{0.0, 1.0}, initialCov}}};
auto history = HybridGaussianHistory<State, Maneuver>{};
HybridPredictedJoint<State, Maneuver> joint;               // refilled by every predictJoint() (Q12)
static_cast<void>(filter.update(fix(z(0))));
for (int k = 1; k <= 12; ++k)
{
    filter.predictJoint(1.0, joint);
    history.push(joint);
    static_cast<void>(filter.update(fix(z(k))));
}
history.smooth(filter.state());
// z: 0.00 1.30 1.60 3.23 4.09 4.65 6.38 8.85 11.83 15.38 17.66 21.07 24.25
//  k   filtered (IMM, exact)   smoothed (IMM, exact)   smoothed v (IMM, exact)   Kim (1994)
//  0      0.500  0.500            0.581  0.256            1.00  1.00              0.251
//  2      0.344  0.344            0.636  0.232            0.99  0.99              0.205
//  3      0.242  0.239            0.727  0.257            1.09  1.02              0.202
//  4      0.157  0.155            0.775  0.317            0.90  0.91              0.221
//  5      0.112  0.110            0.720  0.446            1.32  1.06              0.274
//  6      0.086  0.086            0.673  0.710            2.10  1.98              0.382
//  7      0.193  0.213            0.950  0.743            2.80  2.90              0.592
//  8      0.672  0.780            0.999  0.523            3.26  3.13              0.718
//  9      0.887  0.824            0.529  0.412            3.05  3.01              0.707
// 10      0.773  0.688            0.503  0.362            3.03  2.98              0.591
// 12      0.498  0.347            0.498  0.347            3.21  3.11              0.498
```

The last column is Kim's $\sum_i b_{ij}\, \mu^i_{t+1 \mid N}$, computed by hand from the same
record. The filtered IMM is within 0.15 of exact (at the last step, which every smoother starts
from), and the smoothed velocity within 0.26, at step 5. The smoothed mode probabilities are not
close. They put a maneuver at 0.58 to 0.78 over steps 0 to 5, where the exact posterior gives 0.23
to 0.45, and are near certain of it at steps 7 and 8 (0.95, 0.999) against 0.74 and 0.52: 0.47 off
at step 3 and 0.48 at step 8. With noiseless fixes, $z_k = p_k$, the gap is the same (0.49), so the
noise is not the cause. Kim's, which the code rejected, is within 0.33 here (at step 6) and within
0.03 over steps 0 to 2. A likely reading, not yet checked: the mean approximation evaluates mode
$i$'s smoothed mean at $t+1$ under each mode's prediction. Once the quiet mode's filtered covariance
is small its prediction is narrow, and the small pull the later fixes give that mean costs the quiet
mode more density than the wide agile prediction loses, so the agile mode gains at every step (Q12).

The autonomous multiple model filter [autonomous-multiple-model], which inherits `predictJoint()`
from the IMM and so is smoothed by the same history, is the identity chain. Its mode is fixed for
all time, so the exact smoothed mode probability at every step is the final filtered one, and with
$\Pi = I$ the expectation correction gives exactly that, since $b_{ij} = \delta_{ij}$. Over nine
fixes of the same cruise:

```cpp
auto filter = AmmFilter{
    HybridGaussianParams<State, Regime>{CategoricalParams<Regime>{}, initial},
    ConstantVelocity<Position, Velocity>{0.01},    // Slow
    ConstantVelocity<Position, Velocity>{1.0}};    // Fast
// ... eight steps recorded through predictJoint(1.0, joint) as above, z_k = k + 0.4 sin(2.3 k)
// final filtered P(fast) = 0.047
// k=0  P(fast): filtered 0.500  smoothed 0.047
// k=1  P(fast): filtered 0.481  smoothed 0.047
// k=3  P(fast): filtered 0.362  smoothed 0.047
// k=5  P(fast): filtered 0.198  smoothed 0.047
// k=7  P(fast): filtered 0.078  smoothed 0.047
```

With $\Pi = I$ the mixing is trivial too, so each mode's smoothed Gaussian is exactly its own
Kalman smoother's (`ImmSmootherKeepsAFixedModesProbability` checks the probabilities to 1e-12).

Until 2026-11-01 the mode probabilities were Nadarajah's 8.18 and 8.19, which weigh each mode by
$\Lambda_j$, the density of one point, the smoothed mean at $t+1$, under mode $j$'s own prediction,
and never read the smoothed mode probabilities at $t+1$. On an AMM like the one above they ran from
0.269 to 0.003 where every one should have been 0.034, and on the first version of this section's
maneuver track they were certain of the maneuver at steps 5 to 7 where enumeration gave 0.21 to
0.74. Two published replacements were measured against enumeration over the nine fixes of
`ImmSmootherModeProbabilitiesTrackEnumeration` (`smoothing.cpp`), 512 sequences:

| Mode probabilities | Largest error | Identity chain |
|---|---|---|
| Nadarajah (2009), 8.18 and 8.19, $\Lambda_j$ | 0.054 | drifts |
| [Kim (1994)](../reference/bibliography.md#kim1994), $\sum_i b_{ij} \mu^i_{t+1 \mid N}$ | 0.10 | exact |
| [Barber (2006)](../reference/bibliography.md#barber2006), mean approximation | 0.024 | exact |

Kim's needs nothing new, being the total of the backward mixing weights, but it takes the mode at
$t$ given the mode at $t+1$ to depend on the measurements up to $t$ alone, so it loses what the
later fixes say about a maneuver through the continuous state, and the turning-target test fails
under it. Barber's expectation correction weighs that through the smoothed mean at $t+1$, at the
cost of a prediction per pair of modes, and is what the code does. Its error of 0.024 is at step 7,
where the filtered probability the pass starts from is itself 0.010 from exact. Averaging Barber's
(12) over the cubature points of the smoothed Gaussian, which the paper allows, did much worse there
(0.42): the agile mode's wide prediction wins at the outer points. The example above ranks Kim's
and Barber's the other way, so one test case does not settle the choice (Q12).

> **Q4.** *Decided: the mode probabilities are now the expectation correction of Barber (2006) under its mean approximation, exact on the identity chain and within 0.024 of enumeration in `ImmSmootherModeProbabilitiesTrackEnumeration`; 8.18 and 8.19 never read the smoothed probabilities at $t+1$. The labels were not shifted: the filtered and smoothed probabilities at $t$ are both of $r_t$, the model of the step into $t$. Q12 asks whether the mean approximation is the right one, after §6's maneuver example.* ~~The IMM smoother's mode probabilities are far from the exact posterior: overconfident
> and early on a maneuver, and on the identity chain not the constant they must be. Was 8.18 to
> 8.19 read correctly (the thesis could not be fetched for this review), and does the code's
> reading of "mode $j$" (the model of the step out of $t$, where the forward IMM's $r_t$ is the
> model of the step into $t$) shift the labels by one? If the code is the source, the
> documentation should say that the smoothed continuous estimates are the reliable output and the
> smoothed mode probabilities a heuristic, and the maneuver test, which checks only that the
> probability is high near the turn, should be read in that light. A test against enumeration,
> like the example above, would pin down whichever answer is chosen.~~

> **Q12.** The expectation correction's mean approximation is within 0.03 of enumeration on the
> test's slowly turning target, but 0.48 off on §6's speed change, where Kim (1994) is within 0.33:
> it holds the agile mode at 0.58 to 0.78 while the target cruises. One test case chose between the
> two. Should the choice be measured over several tracks, maneuvers among them, with a test against
> enumeration on a maneuver as well, before it is taken as settled? And should the documentation say
> which output to trust: the smoothed continuous estimates are within 0.26 of exact here, the mode
> probabilities not. Separately, a kept joint can't be written as style.md asks, `auto joint =
> HybridPredictedJoint<State, Maneuver>{}`: since `CategoricalParams` became parameters with an
> explicit default constructor ([categorical-params]), the aggregate's empty braces
> copy-initialize its `modes` and fail ("chosen constructor is explicit in copy-initialization ...
> in implicit initialization of field 'modes' with omitted initializer"), and the joint has no
> allocator constructor. `HybridPredictedJoint<State, Maneuver> joint;` works, as the Doxygen writes
> it. Should the joint have a constructor, empty or from an allocator, as the IMM's update result
> has?

`push()` here takes the joint by `const &` and always copies it, into a new node or into the
oldest node of a full window. `GaussianHistory::push()` and `CategoricalHistory::push()` take their
joint by value and move it. `GaussianMixtureHistory` has both a copy and a move overload.

> **Q5.** Three `push()` signatures for one operation. The IMM's is reasonable, since a kept,
> refilled joint is the allocation-free pattern, but a caller who uses `auto joint =
> filter.predictJoint(dt)` and pushes it pays a copy there and not elsewhere. Should every history
> have both overloads, as the mixture's does?

## 7. The mixture smoother: `GaussianMixtureHistory`

The obstacle for a Gaussian mixture filter is lineage, not math. Under `Collapse`, `Select`, or
`Keep` without merging, each component has one parent and RTS per component is exact. A merge
gives a component several parents. The mixture keeps a `MixtureLineage` for this: a row per
component of its log-share from each component the mixture had at the last `startLineage()`, kept
by every operation that changes components (a merge sums two rows by weight, `split()` and
`Keep` copy the parent's row, removal drops rows, `add()` gives none).
`MixtureLineage` in `Mixture.h` documents the lineage as part of the mixture parameters.
`MixtureFilter::predictJoint(dt)` records a `GaussianMixturePredictedJoint`, which holds the
weights, each component's `PredictedJoint`, and the state's lineage, and then starts the lineage
over. Each record therefore says how its components descend from those of the record before.

Each backward step of `GaussianMixtureHistory::smooth()`:

- smooths every pair of a smoothed component $p$ at $k+1$ and a component $n$ at $k$ that $p$ has a
  share from, by `smoothStep()` from $n$'s filtered Gaussian through its joint to $p$'s smoothed
  Gaussian;
- weights the pair by $p$'s smoothed weight times its share from $n$;
- moment-matches the pairs that share a parent $n$, so the smoothed mixture at $k$ has one
  component per filtered component, in the same order. A component nothing descends from (one
  the filter pruned) keeps its filtered Gaussian with zero weight.

This is the forward-backward smoother on merged Gaussian mixtures of
[Rahmathullah, Svensson, and Svensson (2014)](../reference/bibliography.md#rahmathullah2014), §IV
and Algorithm 2. Its f-graph is the lineage and its s-graph the smoothed mixtures. There are four
departures:

- the pair weight is the paper's (15), the share normalized by the merged node's weight, not
  (16) as printed, without which the weights don't sum to one (the test
  `MergedComponentsSplitByTheirShares` shows both);
- the share is kept per parent rather than per merged branch, which is the same once a parent's
  branches are summed;
- each parent's group is merged by a full collapse rather than threshold merging, which bounds
  the smoothed mixture by the filtered one and keeps them aligned;
- the step is `smoothStep()`, in the cross-covariance form and the tangent space.

Under `Collapse` it is the Kalman smoother per component. The example checks that against a
`KalmanFilter` and `GaussianHistory` per prior component. It then runs `Keep` with a fix that is an
outlier one time in five, so the components double at every fix, and `reduce`, the greedy merge
by Runnalls' cost of `examples/gaussian_mixture.cpp` down to four components, brings them back:

```cpp
auto prior = GaussianMixtureParams<State>{};
prior.add(std::log(0.6), Eigen::Vector2d{0.0, 1.0}, cov);
prior.add(std::log(0.4), Eigen::Vector2d{2.0, -1.0}, cov);

auto filter = GaussianMixtureFilter{model, prior};
auto history = GaussianMixtureHistory<State>{};
for (std::size_t k = 1; k < zs.size(); ++k)                  // zs: 0.1 1.3 1.9 6.0 4.1 4.8 6.1
{
    history.push(filter.predictJoint(1.0));
    static_cast<void>(filter.update(outlierFix(zs[k]), reductions::keep()));   // 0.8 N(0, 0.5) + 0.2 N(0, 25)
    static_cast<void>(reduce(filter.state()));
}
history.smooth(filter.state());
// Collapse: largest difference from a Kalman smoother per component 0.0e+00
// k=0  components 2 (filtered 2)  weights sum 1.000000000000  p: filtered 0.800, smoothed 0.377
// k=1  components 4 (filtered 4)  weights sum 1.000000000000  p: filtered 1.251, smoothed 1.244
// k=2  components 4 (filtered 4)  weights sum 1.000000000000  p: filtered 1.843, smoothed 2.151
// k=3  components 4 (filtered 4)  weights sum 1.000000000000  p: filtered 4.225, smoothed 3.097
// k=4  components 4 (filtered 4)  weights sum 1.000000000000  p: filtered 4.411, smoothed 4.044
// k=5  components 4 (filtered 4)  weights sum 1.000000000000  p: filtered 4.913, smoothed 4.999
```

Each smoothed mixture lines up with its filtered one, and the weights sum to one. The outlier at
step 3 (`z = 6.0` on a track near 3) pulls the filtered mean to 4.2; the smoother, which sees the
later fixes, puts it back at 3.1. The combined estimate is read through
`step.smoothed.toDensity()->collapsed()`, since the parameters have no `collapsed()` of their own.
The tests add an exact comparison: with `Keep` and no merging, on a plain vector space, each
smoothed component has the weight and moments of the 16 exact branches through it, to 1e-12 and 1e-9
(`KeepMatchesTheExactSmootherBranchByBranch`).

Records must be one `predictJoint()` apart. A filter state replaced by a copy has no lineage, and
a plain `predict()` between records leaves no mark, so `push()` and `smooth()` assert: "the
lineage must start at the newest record's predictJoint(): one predictJoint() apart, and no copy of
the state between" (`HistoryRefusesALineageThatDoesNotFollow`).

`MixtureFilter` is generic in its component [filters-and-recursions], but `predictJoint()` requires
Gaussian components (`requires isGaussian<ComponentType>`, the trait now public in
`Distributions/Component.h`), and the history holds only Gaussians. A mixture of hybrids, a mixture
of IMMs, keeps a lineage already, but has no joint and no smoother, and no source is chosen (Q10).
Since the square-root form became a component [gaussian-forms], a mixture of square-root Gaussians
has no `predictJoint()` either, for §5's reason. The lineage has a second reader now:
`MixtureLineage::carry()` applies a record's shares to a value kept per component outside the
filter, such as a running NIS [user-extension-points].

> **Q6.** Should the missing `predictJoint()` on a `MixtureFilter` of non-Gaussian components
> fail with a message that names the reason ("smoothing a mixture of hybrids is not provided,
> see Q10"), rather than as an unsatisfied `requires`?

> **Q10.** Smoothing a mixture of IMMs [smoothing], [filters-and-recursions]. `MixtureLineage` is
> kept for any component, but the backward step per pair would be the IMM smoother's, and a merge of
> two hybrids mixes modes. Which source should it follow?

## 8. The particle smoother: `ParticleHistory`

`ParticleHistory<State>` records the filter's particles before each prediction with
`push(filter.state())`, and `smooth(filter.state())` fills each step's smoothed `ParticleSet` by
ancestor tracing. Particle $j$ of a step's smoothed set is the state at that step of the
trajectory ending in the current particle $j$, with the current particle's weight. Prediction
and update keep each particle in its slot, so the only change of lineage between two records is
a resample. The history takes `ParticleSet::ancestors()` when the set's `resampleCount()` has moved
by one since the last record. It asserts against two resamples between records ("at most one
resample may fall between two records of the particles"), since `ancestors()` keeps only the
last, and against a count that went back, which is what a filter state swapped for a copy shows.
`ParticleSet::removeIf()` also renumbers particles and must not fall between records; no
upkeep action calls it on a particle set.

Ancestor tracing is the smoother of [Kitagawa (1996)](../reference/bibliography.md#kitagawa1996),
who stores each particle's path and resamples paths with particles. The departure is storage.
The history keeps each step's particles once, with the parent indices that join them, and traces
the paths when asked. That costs $O(N L)$ index lookups and copies per `smooth()` instead of
copying paths at every resample. Each smoothed set drops the bandwidth the filtered set stored,
so its kernel density uses Silverman's rule.

The example runs 5,000 particles from seed 7 with `upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))` over sixteen fixes of
a cruise, $z_k = k + 0.4 \sin(2.3 k)$, and compares with the RTS smoother on the same affine
problem:

```cpp
auto filter = ParticleFilter{model, ParticleSet<State>{State{}, drawSamples(initial.toDensity().value(), 5000, rng)}};
auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
auto history = ParticleHistory<State>{};
static_cast<void>(filter.update(fix(z(0))));
static_cast<void>(resampling(filter.state()));
for (int k = 1; k <= 15; ++k)
{
    history.push(filter.state());                          // before the prediction
    filter.predict(rng, 1.0);
    static_cast<void>(filter.update(fix(z(k))));
    static_cast<void>(resampling(filter.state()));  // resamples at most once per step
}
history.smooth(filter.state());
// Clang (MSVC STL):
// k= 0  smoothed p: particles 0.063, RTS 0.052 (sd 0.538)   distinct paths 400
// k= 3  smoothed p: particles 3.010, RTS 3.004 (sd 0.379)   distinct paths 558
// k= 6  smoothed p: particles 5.993, RTS 6.004 (sd 0.375)   distinct paths 674
// k= 9  smoothed p: particles 9.025, RTS 9.004 (sd 0.375)   distinct paths 1050
// k=12  smoothed p: particles 11.993, RTS 12.002 (sd 0.383)   distinct paths 1447
// k=14  smoothed p: particles 14.029, RTS 14.038 (sd 0.404)   distinct paths 2568
// GCC (libstdc++), whose random distributions differ:
// k= 0  ... particles 0.068 ... distinct paths 390;  k=14  ... particles 14.030 ... distinct paths 2618
```

The smoothed means stay within a few hundredths of a standard deviation of RTS, but the paths
collapse going back. 5,000 paths pass through about 2,500 distinct particles one step back and
about 400 fifteen steps back, so the smoothed sets lose their spread. This is a fixed-lag tool,
for a lag short enough that the paths have not yet collapsed, and `Lag{L}` gives it the same
recycling window as the Gaussian history. The test `ParticleSmootherIsNearTheKalmanSmootherAtShortLags`
bounds it at 20,000 particles: one to three steps back the means are within 0.2 standard
deviations of RTS and the covariance traces within 10%, while eight steps back fewer than a tenth
of the paths are distinct.

Backward simulation (Godsill, Doucet, and West, 2004) does not degenerate, but it needs the
transition log-density $\log p(x_{k+1} \mid x_k)$, which no process model exposes, and when the
process noise is singular that density lives on the range of $Q$ (Q11). The same interface would
serve a Rao-Blackwellized smoother.

> **Q11.** A transition log-density for backward simulation [smoothing]. A process model gives its
> function and its noise over a step, from `continuousNoise()` or `additiveNoise()` or a built-in
> model's closed form, from which the density follows when $Q$ is nonsingular. When $Q$
> is singular, the density lives on the range of $Q$ at the propagated point, and particles off it
> have zero density, which a density on that range or a density the model supplies would have to
> say. Which?

> **Q7.** `ParticleHistory` takes a `ParticleSet` only. A Rao-Blackwellized particle filter's
> state (`ParticleGaussianParams`) has a sampled block with the same genealogy, and ancestor tracing of
> that block alone needs no new algorithm, but its conditional Gaussians would need the
> Rao-Blackwellized smoother. Should ancestor tracing accept the sampled block of such a state, or
> stay a particle-filter tool until the Rao-Blackwellized smoothers are designed?

## 9. The hidden Markov smoother: `CategoricalHistory` and Viterbi

The hidden Markov filter ([emissions], reviewed in
[hidden-markov-filter.md](hidden-markov-filter.md)) records a `CategoricalPredictedJoint` from
`predictJoint()`: the filtered distribution, the prediction, and the step's transition matrix $K$.
`CategoricalHistory<Space>` has `GaussianHistory`'s interface, and its `smooth()` calls the
categorical `smoothStep()`:

$$s'_i = f_i \sum_j K_{ij}\, \frac{s_j}{\hat p_j},$$

the Bayesian smoothing equation of [Särkkä (2013)](../reference/bibliography.md#sarkka2013),
chapter 8, on a finite set. On a hidden Markov model these are the smoothed marginals $\gamma$ of
the forward-backward algorithm of [Rabiner (1989)](../reference/bibliography.md#rabiner1989),
§III-A, in the form that needs only the filtered distributions: $s_j / \hat p_j$ is Rabiner's
backward variable, normalized. The ratio is taken in logs and scaled by its largest entry. The
step requires `smoothedNext` to have valid weights, which it asserts: weights that are not
reach no fallback, as they would make `transitionPosterior()` NaN.
`transitionPosterior(joint, smoothedNext)` gives the pairwise posteriors $\xi$ of Rabiner's
§III-C, whose sums over a history are the expected transition counts Baum-Welch consumes; the
re-estimation itself is outside the library [analysis-project].

`mostProbableSequence(history, latest)` is the
[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) of
[Viterbi (1967)](../reference/bibliography.md#viterbi1967), in Rabiner's max-product form (§III-B).
The departure is where the likelihoods come from. The history keeps no measurements, so each
step's likelihood is the ratio of the filtered to the predicted probability. That ratio differs
from the product of the updates' likelihoods by their evidence, which is the same for every
sequence, so the argmax is unchanged and the maximum is the sequence's posterior
log-probability. Over a fixed-lag window it is the most probable sequence of the window's values,
with the earlier ones summed out. Ties go to the lowest value. The example checks all three
against enumeration of the 32 weather sequences:

```cpp
struct Weather : Named<Sunny, Rainy> {};
struct Seen : Named<Dry, Umbrella> {};

auto filter = HiddenMarkovFilter{MarkovChain<Weather>{transitions},          // {{0.8, 0.2}, {0.4, 0.6}}
                                 CategoricalParams<Weather>::fromWeights(initial)};  // {0.5, 0.5}
auto const table = EmissionTable{Weather{}, Seen{}, emissions};                         // {{0.9, 0.1}, {0.3, 0.7}}
auto history = CategoricalHistory<Weather>{};
static_cast<void>(filter.update(table.observe(seen[0])));                     // seen: 0 1 1 0 1
for (std::size_t k = 1; k < seen.size(); ++k)
{
    history.push(filter.predictJoint());
    static_cast<void>(filter.update(table.observe(seen[k])));
}
history.smooth(filter.state());
auto const path = mostProbableSequence(history, filter.state());
// k=0  P(rainy): filtered 0.2500  smoothed 0.4492  enumerated 0.4492
// k=1  P(rainy): filtered 0.7500  smoothed 0.8577  enumerated 0.8577
// k=2  P(rainy): filtered 0.8750  smoothed 0.8591  enumerated 0.8591
// k=3  P(rainy): filtered 0.2895  smoothed 0.4600  enumerated 0.4600
// P(x0 = x1 = rainy): 0.4289, enumerated 0.4289
// Viterbi: 0 1 1 1 1  probability 0.1804; enumerated: 0 1 1 1 1  probability 0.1804
```

The tests do the same over 243 sequences of three values with any number of updates between
predictions, and for the Markov-modulated Poisson process over a record of arrivals.

> **Q8.** `ValueSequence::values` holds flat indices, so the path above prints `0 1 1 1 1` rather
> than `Sunny Rainy ...`, and a caller over a `DiscreteProduct` decodes each index. Should it
> hold values of the space, or offer a typed accessor? It is the reverse of `valueOf()` and `at<>()`,
> which name a value by tags and indices; the roadmap favors typed queries on a flat index.

## 10. Allocation and execution policies

Every history is a `std::list` of steps from its allocator [allocation], re-bound to the nodes, with
`allocator_type`, `get_allocator()`, and allocator-extended copy and move constructors
(`HistoryUsesItsAllocator`). The steps of the IMM, mixture, and particle histories build what they
hold from the same allocator by uses-allocator construction. On a bounded space:

- a full fixed-lag window allocates nothing, since `push()` recycles the oldest node. That holds
  for the IMM and mixture histories when one kept joint is refilled by `predictJoint(dt, joint)`,
  and for the particle history at a steady particle count (`FixedLagSmoothingAllocatesNothingOnceFull`,
  `ImmSmoothingStepsWithinItsResource`, `GaussianMixtureSmoothingStepsWithinItsResource`,
  `FixedLagParticleSmoothingStepsWithinItsResource`, `HiddenMarkovFilterAndSmootherStepWithoutAllocating`);
- an unbounded history takes one node per step from its resource and nothing from the heap;
- the mixture smoother's pair scratch is owned by the history and reused;
- `mostProbableSequence()` allocates its back-pointers and result from the history's allocator.

`ParticleHistory::smooth(policy, latest)` and `GaussianMixtureHistory::smooth(policy, latest)` take
an execution policy [execution-policies]. The particle tracing and gather are for-eaches over the
particles, and the mixture's pairs and merges a for-each over each step's components, both bitwise
Sequential's under `par` (`ParticleHistoryMatchesSequential`,
`GaussianMixtureSmoothingMatchesSequential`). `MixtureFilter::predictJoint()` takes a policy too.
The Gaussian, IMM, and categorical histories have none: their work per step is one RTS step, one per
mode, or a product with the transition matrix, none of which grows with a run-time count beyond the
steps, and the steps run in sequence.

## 11. What has no smoother

| Filter | Smoothed by | Notes |
|---|---|---|
| `KalmanFilter` over a `GaussianParams`, any method or fixed-step model | `GaussianHistory` | §3 to §5 |
| `SqrtKalmanFilter`, `InfoFilter`, `SqrtInfoFilter` | none | no `predictJoint()` (Q3) |
| `ImmFilter` | `HybridGaussianHistory` | mode probabilities by the expectation correction, approximate (Q4, Q12) |
| `AmmFilter` | `HybridGaussianHistory`, inherited | exact: per-mode Gaussians and the constant mode probabilities (§6) |
| `MixtureFilter` of Gaussians | `GaussianMixtureHistory` | §7 |
| `MixtureFilter` of hybrids, square-root Gaussians, or other components | none | `requires isGaussian` (Q6, Q10) |
| `ParticleFilter` | `ParticleHistory` | ancestor tracing only; backward simulation deferred |
| `RbParticleFilter`, `RbPointMassFilter`, `PointMassFilter` | none | deferred ([smoothing]; Q7) |
| `HiddenMarkovFilter` | `CategoricalHistory`, `mostProbableSequence()` | §9 |

Two-filter smoothers, iterated smoothers that relinearize about the smoothed estimates, and
smoothers for the continuous-time models between measurement times are not in the library.
[smoothing] records the choice of the RTS form; the others would be new algorithms, so they wait
past the 0.1 algorithm freeze.

Nor is there a consider smoother. A Kalman filter whose updates considered a block
[consider-states] is smoothed by `GaussianHistory` as any other, since the history records the
predictions and never sees a measurement's consider set. The backward pass then moves the
considered block by its covariance with the estimated ones, as if it had been estimated, and its
smoothed covariance is not the error covariance of those estimates, since the pass assumes an
optimal filter. [consider-states] records this; a consider smoother would be
an addition.

> **Q9.** The table above lives only in this review. Should the Smoothers module page, or
> `GaussianHistory.h`'s `\file` comment, carry it, so that a user of a form filter or an RBPF learns
> that there is no smoother from the reference rather than from a failed `predictJoint()`?

## Questions

1. `GaussianHistory` and `CategoricalHistory` can't detect a missing record or a `latest` that
   doesn't follow; add a step count to the joints? (§3)
2. The histories are spelled space first; follow [component-first]/[filters-and-recursions] with a
   component-first `History`? (§3)
3. The square-root and information forms have no `predictJoint()` and can't be smoothed; out of
   scope for 0.1, or record in covariance form? The error names only `!IsForm`. (§5)
4. ~~The IMM smoother's mode probabilities are far from the exact posterior, and on the AMM's
   identity chain they are not constant; check 8.18 to 8.19 and the mode labeling, and document
   or test accordingly. (§6)~~ *Decided: Barber's expectation correction, tested against
   enumeration; Q12 asks whether one test case settles it.*
5. Three `push()` signatures (by value, `const &` only, both); give every history both
   overloads? (§6)
6. `MixtureFilter::predictJoint()` for non-Gaussian components fails as an unsatisfied
   `requires`; say why instead? (§7)
7. Ancestor tracing of a Rao-Blackwellized particle filter's sampled block, or wait for the
   Rao-Blackwellized smoothers? (§8)
8. `mostProbableSequence()` returns flat indices, not values (the reverse of `valueOf()`). (§9)
9. Put the table of what has a smoother in the reference documentation? (§11)
10. Smoothing a mixture of IMMs: which source? (§7)
11. A transition log-density for backward simulation, with a singular $Q$. (§8)
12. The mean approximation is 0.48 off enumeration on §6's speed change, where Kim's is 0.33 off;
    measure the choice over more tracks? And a kept `HybridPredictedJoint` can't be written
    `auto joint = HybridPredictedJoint<...>{}`. (§6)
