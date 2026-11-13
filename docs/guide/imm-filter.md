# Interacting multiple model filter

Review notes on the interacting multiple model (IMM) filter, the autonomous multiple model (AMM)
filter, and the pieces they are built from: `Filters/ImmFilter.h`, `Filters/AmmFilter.h`,
`Distributions/Hybrid.h`, `Models/SwitchingModel.h`, and `MarkovChain` in
`Distributions/Categorical.h`, tested in `tests/Filters/imm_filter.cpp` and
`tests/Filters/amm_filter.cpp` (and, for allocation, `tests/allocation_mixtures.cpp`). The
filter keeps one Gaussian per value of a discrete mode, under a probability for each mode.
Before each prediction it mixes the modes' Gaussians through a Markov chain, then predicts each
through its own process model, and it updates each one by the measurement and reweights the modes
by how well each explained it. A mode axis can also belong to a sensor, so that outliers persist
from one fix to the next, as a Markov chain says. The AMM is the same filter on the identity
chain: the mode never changes, and the filter identifies which one is in force. Each section says
what a piece does and shows it in use. Questions are marked **Q*n*** and collected at the end.

The examples were refreshed for roadmap 17z (`next`, 2026-11-11), which replaced the
model API: a mode's process model is the built-in `ConstantVelocity` itself, a CV/CA pair is a
`compose()` of built-in models, and a fix is a `measurementModel()` that `observe()` takes with its
method. The examples of §1, §3, §4, §6, §7, and §11 were recompiled and run with Clang 21 (MSVC STL)
and printed what they printed on 2026-11-04, but for the floor example of §11, whose code named a
floor that was not the one it ran (corrected there). GCC and MSVC were not rerun for this refresh;
on 2026-11-04 Clang and GCC 14.2 (w64devkit) printed the same results, and the tests are the evidence
that MSVC deduces the same types. The measurements are noise-free or fixed by hand, so no output
depends on a standard library's random distributions. Compiler errors are quoted from Clang, and
were rerun for this refresh. Roadmap 17z also gave `update<Path>()` with Gaussian noise a
`static_assert` that names the cause (§7, Q11), and removed the class template `Emrys::Sensor`, so
a global tag `Sensor` no longer collides with it (Q17). Snippets marked as condensed from the tests
are covered by the test suite. The refresh of 2026-11-04, from that of 2026-10-27, found these changes: a
gate weights a mode by its passing branches, which settles Q8 (§6); `update<Path>()` updates each mode with its one noise component only (§8),
and with Gaussian noise now fails to compile rather than at run time (§7, Q11); mixture noise
allocates nothing into a reused result (§8); the square-root IMM moves its modes in place (§3a);
`HybridParams::conditional(m)` is read-only, with `setConditional()` and `updateConditional()`; the
filters' results are made from an allocator; `CategoricalParams::fromWeights()` and `fixedStep()`
name a fixed-size space rather than take one; the sensor axis's tag, `Sensor` until that refresh,
became `SensorMode`, since the library's class template `Emrys::Sensor` (from 2026-10-31 to roadmap
17z) made a global `Sensor` ambiguous under `using namespace Emrys` (Q17), as hidden-markov-filter.md's
tag also became; and the IMM smoother records each mode's prediction through every other mode's
model (§12). The printed numbers were unchanged but for the printout of §1 and a rounding difference
in §7.

Q10 was decided on 2026-11-12: a measurement's noise names its mode axis,
`additiveNoise(mixture).byMode<SensorMode>()`, and `update<Path>()` is gone. §7 is rewritten for it,
and its code spells the new form; its printed numbers are from 2026-11-04, by the same arithmetic,
and were not rerun. Q9's test came with it. Elsewhere this document's mentions of `update<Path>()`
describe the code before that decision. §7 also gained a measurement per value of an axis,
`switchingMeasurement<Path>()` (roadmap 17ze), with Q19 and Q20; its snippet is condensed from the
tests.

The examples share these declarations, with the library's names unqualified, as under
`using namespace Emrys`:

```cpp
struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
using State = Composite<Position, Velocity>;

struct Cruise {};
struct Turn {};
struct Maneuver : Named<Cruise, Turn> {};
struct Inlier {};
struct Outlier {};
struct SensorMode : Named<Inlier, Outlier> {};
using Mode = DiscreteProduct<Maneuver, SensorMode>;  // flat index 2 m + s

auto cv(double density) { return ConstantVelocity<Position, Velocity>{density}; }
auto const initial = GaussianParams<State>{State{}, Eigen::Vector4d{0, 0, 10, 0},
                                     Eigen::Vector4d{10, 10, 4, 4}.asDiagonal().toDenseMatrix()};

// A position fix z = p + v, by
// observe(methods::linearized(), measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>),
//                                                 additiveNoise(noise)), z).
// fix(z): v ~ N(0, I).
// robustFix(z): v ~ 0.9 N(0, I) + 0.1 N(0, 100 I), a GaussianMixtureParams<Linear<2>>, components in that order.
auto fix(Eigen::Vector2d const & z);
auto robustFix(Eigen::Vector2d const & z);
```

Discrete spaces (`Named`, `DiscreteProduct`, flat indices, `valueOf`) belong to `Spaces/Discrete.h`,
and `CategoricalParams` and `Categorical` to `Distributions/Categorical.h` [categorical-params]. The filter's
shape (`model()`, `method()`, `setState()`, `update(measurement, policy)`, the state and recursion
split of [filters-and-recursions]), the policies (`Collapse`, `Select`, `Keep`, `Gate`), and
`FilterUpdate` belong to `KalmanFilter.h` and `Models/Reduction.h`, and the process models and fixed-step
models to the Models module page. Time steps of any type [time-steps], allocators [allocation], and
printing [printing-and-simulation] are the Core module page's. The floor on mode probabilities,
`FloorModes`, is an upkeep action, documented in `Upkeep.h`; §11 only uses it.
The IMM smoother (`predictJoint()`, `HybridGaussianHistory`) is [smoothing.md](smoothing.md)'s, the
mixture of IMM filters `MixtureFilter.h`'s, and hidden Markov
models over a discrete state alone, with the continuous-time chain,
[hidden-markov-filter.md](hidden-markov-filter.md)'s. This document uses them and doesn't explain
them again.

## Decisions so far

From the review: Q7 is decided by [switching-models] (a method per model, the same one for every
mode by default), and Q14 with it (fixed-step models with different methods are allowed), both
now in the code (§4). These decisions from
[decisions.md](../design/decisions.md) are in place and should be taken as given:

- **The hybrid core** (the Filters module page, `EmrysFilter/Filters.h`; no named decision).
  Mixture noise, the mixture filter, IMM, and the grid of Kalman filters share one structure: a
  discrete variable and a Gaussian conditioned on it. Memoryless mixture noise is the IMM case where
  every row of $\Pi$ is the same, and a persistent outlier process is IMM on the measurement side.
- **[switching-models].** A switching process model is a set of per-mode models plus $\Pi$, owned by
  the filter. Every mode shares one state space; a mode that doesn't use a block holds it. Each
  mode's model may predict by a method of its own, given by `withMethod()` or carried by a
  fixed-step model, and by default every mode predicts by the filter's (Q7). The filter's
  `method()` is that default, `ModelsMethod` when every model is fixed-step (Q14).
- **[discrete-spaces].** Modes are values of a discrete space, stored as flat indices with the last
  axis fastest, so independent per-axis chains combine by Kronecker product.
- **[imm].** The hybrid state keeps exactly one component per mode under the mode weights, a
  `CategoricalParams`;
  `GaussianMixtureParams` stays unlabeled. `SwitchingModel` assigns models one per mode, one per
  value of a mode axis, or one for all. The filter is spelled like `KalmanFilter`, its
  policy must keep one component, a rejected mode keeps its prior component with probability
  zero, and noise by mode axis, `additiveNoise(mixture).byMode<Axis>()`, makes the noise components
  the values of a mode axis where the mode space has it, the memoryless mixture where the state has
  no mode space, and a compile-time error where the mode space lacks it (Q10). A
  `SwitchingMeasurement` holds a measurement per value of a mode axis, and `update()` updates each
  mode by the one its value selects (§7).
- **[component-first].** `HybridParams<Component, ModeSpace, Allocator>` takes the distribution it
  holds per mode first; `HybridGaussianParams<Space, ModeSpace, Scalar, Allocator>` is its alias for
  Gaussians and the spelling for the common case. A hybrid is itself a `Component`, named by its
  density `Hybrid` (`HybridGaussian`), so a `MixtureParams<HybridGaussian<State, Mode>>` is a
  mixture of IMM states.
- **[params-and-density].** The moments and the collapses read the components, so they are the
  density's, the `Hybrid` that `toDensity()` gives after checking the mode weights and factorizing
  every component.
- **[filters-and-recursions].** A filter holds a state and runs a recursion. `ImmRecursion` holds
  the switching model, the method, and the scratch the modes are mixed and updated into, and acts
  on a hybrid passed in; `ImmFilter<Component, Model, Method, Allocator>`, a class with no
  abbreviation behind it, owns the hybrid and runs it. Unlike `KalmanFilter` and `MixtureFilter`, it
  takes no recursion parameter (Q18). An example that deduces a hybrid writes
  `HybridParams{modes, initial}`, not `HybridGaussianParams{...}`, which MSVC rejects.
- **[autonomous-multiple-model].** `AmmFilter` is the IMM on the identity chain, a derived class
  whose constructor builds that chain; the IMM's prediction skips the mixing when the chain is
  exactly the identity; and `FloorModes` is an upkeep action, for which the IMM filter exposes
  a mutable `state()`. A floor as a filter option is rejected.
- **[innovation-cholesky].** A mode whose innovation covariance is not positive definite fails the
  whole update: `update()` returns `GaussianError::NotPositiveDefinite` in its `std::expected`, and
  the overload that fills a result returns `std::expected<void, GaussianError>`. Nothing is copied
  to undo it: the IMM of Gaussians builds its posterior apart, so its state is unchanged, while the
  square-root IMM updates its modes in place and keeps those before the failed one updated (§3a).
- **[gaussian-forms].** The square-root form is a component, so a hybrid of `SqrtGaussian`s is a
  square-root IMM: factors throughout, `Collapse` alone, and a `predict()` that returns a
  `std::expected`; the information forms are not components (§3a).
- The decisions [templated-measurements], [method-per-measurement],
  [weighted-branches], [reduction-policies], [filters-and-recursions], and [composed-features] apply
  here unchanged, as do [allocation] (allocation), [fixed-step-models] (fixed-step models),
  [view-stamp] (no view of a temporary), and [time-steps] (time steps of any type).

## 1. Differences from the Kalman filter

| | `KalmanFilter` | `ImmFilter` |
|---|---|---|
| Params | `<Component, Model, Method, Recursion>` | `<Component, Model, Method, Allocator>`, the allocator the state's; no recursion parameter (Q18) |
| Model | a process model | a `SwitchingModel`: a `MarkovChain` and one or more process models |
| State | a `Component`, such as `GaussianParams<Space, Scalar>` | `HybridParams<Component, ModeSpace, Allocator>`, normally a `HybridGaussianParams<Space, ModeSpace>` |
| Recursion [filters-and-recursions] | `KalmanRecursion` | `ImmRecursion`, with scratch for every mode; not reentrant |
| Constructor | `(model, state, method = {})`, or `(recursion, state)` | `(model, state, method = {})`; asserts that the model and state have the same modes, and checks that every model writes every coordinate of the space once (at compile time where the space is fixed-size) |
| Change the state | mutable `state()`, `setState()` | the same [state-access]: `setState()` also asserts the mode space, and the mutable `state()` takes upkeep such as `FloorModes` [autonomous-multiple-model] |
| `predict(dt, inputs...)` | the Gaussian | mixes through $\Pi$, predicts each mode by its model, steps the chain; with the identity chain, no mixing (§5); `[[nodiscard]]`, returning nothing for Gaussians and a `std::expected` for square-root modes (§3a) |
| `predict(inputs...)`, `predict(Steps{n})` | with fixed-step models [fixed-step-models] | the same, the chain stepping once per step |
| `update(m, policy = reductions::collapse())` | reduces the branches to the new state | reduces each mode's branches, reweights the mode by $p(z \mid m)$, or under a gate by its passing branches' likelihood (§6) |
| `update(m, policy)`, noise `byMode<Axis>()` | the memoryless mixture, by its weights | each mode uses only the noise component its `Axis` value names (§7) |
| `update(switchingMeasurement<Path>(m0, m1, ...), policy)` | | each mode is updated by the measurement its `Path` value selects, all with one result type (§7) |
| `update(m, policy, result)` | refills a result | the same, for a switching measurement too (§6) |
| `update(state, m, given<P>(space, value), policy)` | `update(prior, m, given)` | on `ImmRecursion`: each mode updated given the block, plainly or by mode axis, for a point-mass filter's hybrid cells (rao-blackwellized-filters.md §6) |
| Policies accepted | those returning `std::optional<Component>` | the same; `Keep` is a compile-time error |
| A part rejected | (one part) | the mode keeps its prior component, with probability zero |
| Returns | `FilterUpdate`, in a `std::expected` [innovation-cholesky] | `ImmFilterUpdate`, in a `std::expected`: one `FilterUpdate` per mode |
| With one mode | | the Kalman filter, to $10^{-12}$ (tests, Unscented, including NIS) |

The method serves every mode whose model carries none of its own (§4), and defaults to
`Linearized`; with fixed-step models, which all carry one, it is `ModelsMethod`. Deduction works
through the class's one guide:

```cpp
auto filter = ImmFilter{SwitchingModel{MarkovChain<Maneuver>::persistent(0.95), cv(0.01), cv(4.0)},
                        HybridParams{CategoricalParams<Maneuver>{}, initial}};
// ImmFilter<Gaussian<State>, SwitchingModel<Maneuver, double, Maneuver, CV, CV>, Linearized, std::allocator<std::byte>>
```

`std::cout << filter` prints "interacting multiple model filter" and the state under it
[printing-and-simulation]: the moments over every mode, then each mode's Gaussian with its
probability, nested (`HybridGaussian.PrintsEachModeNested`):

```
interacting multiple model filter
  hybrid of 2 modes of dimension 4
    mean: 0 0 10 0
    covariance:
      10  0  0  0
      ...
    mode 0, probability 0.5:
      Gaussian of dimension 4
        mean: 0 0 10 0
        ...
```

## 2. MarkovChain

`MarkovChain<DiscreteSpace, Scalar = double>` holds a row-stochastic transition matrix,
$\Pi_{ij} = P(j \mid i)$: a row per value moved from, a column per value moved to. Its storage is
fixed-size under the same rule as `CategoricalParams`'s: `MarkovChain<Mode>` is a 4 × 4 row-major
matrix, 144 bytes.

| Built by | Gives |
|---|---|
| `MarkovChain<S>{}`, `MarkovChain{space}` | the identity: never moves |
| `MarkovChain{space, matrix}` | that matrix; asserts its shape, that it is nonnegative, and that rows sum to one (to `dummy_precision`) |
| `MarkovChain<S>::persistent(stay, space = {})` | stays with probability `stay`, otherwise moves to any other value equally |
| `MarkovChain<S>::memoryless(next)` | every row is `next`: the value at each step is independent of the last |
| `MarkovChain<S>::fromDwellTimes(dwell, exits, space = {})` | stays a mean of $\tau_i$ steps at value $i$, $\Pi_{ii} = 1 - 1/\tau_i$, and leaves for $j$ in proportion to `exits(i, j)`; without `exits`, to every other value equally |
| `MarkovChain<Product>{space, chains...}` | one chain per axis, in order: their Kronecker product |
| `ContinuousTimeMarkovChain<S>::transition(dt)` | $e^{Q \, \Delta t}$ from a rate matrix ([emissions]; below), which also has a `fromDwellTimes()`, with dwell times in seconds |

`matrix()` and `transition(from, to)` read it, by flat index. `predict(params, chain)` takes
one step, keeping the total weight, $p'_j = \sum_i p_i \Pi_{ij}$. `stationary(chain)`, for
either kind of chain, is the distribution the chain settles into, the usual mode prior: a
`CategoricalParams`, or `ChainError::Reducible` when some value can't reach some other. It uses
the algorithm of [Grassmann, Taksar, and Heyman (1985)](../reference/bibliography.md#grassmann1985),
which reads only the off-diagonal entries, so that one routine serves a transition matrix and a
generator, and never subtracts, so it keeps every digit for a chain that stays at each value with
probability $1 - 10^{-12}$ and finds reducibility exactly from the zero structure.

```cpp
// a cruise lasts 30 steps, a turn 8, a dive 4; a cruise ends in a turn twice as often as in a dive
auto const exits = Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}};
auto const flight = MarkovChain<Flight>::fromDwellTimes(Eigen::Vector3d{30, 8, 4}, exits);
stationary(flight)->toDensity()->probabilities();  // 0.818 0.145 0.0364: 9/11, 8/55, 2/55
```

A maneuver regime and a sensor state that switch independently:

```cpp
auto const maneuver = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.10, 0.90}}};
auto const chain = MarkovChain<Mode>{maneuver, MarkovChain<SensorMode>::persistent(0.8)};
chain.matrix();
// 0.76 0.19 0.04 0.01      from (Cruise, Inlier)
// 0.19 0.76 0.01 0.04      from (Cruise, Outlier)
// 0.08 0.02 0.72 0.18      from (Turn, Inlier)
// 0.02 0.08 0.18 0.72      from (Turn, Outlier)

auto p = CategoricalParams<Mode>::fromWeights(1, 0, 0, 0);  // in Mode{}, a fixed-size space
p = predict(p, chain);  // 0.76 0.19 0.04 0.01;            P(Turn) 0.05
p = predict(p, chain);  // 0.6171 0.2904 0.0629 0.0296;    P(Turn) 0.0925
```

Because the last axis varies fastest, the Kronecker product's entry $(i, j)$ is the product
of each axis's entry at $i$'s and $j$'s coordinates, which is what the constructor computes
directly. Note that `persistent()` on the product itself is a different chain: it moves to each
of the three other modes with probability $0.2 / 3 = 0.0667$, so a maneuver change and a sensor
change are as likely together as apart. For independent axes, build per axis.

`memoryless(next)` is the bridge between mixture noise and IMM, shown at work in §7:

```cpp
auto const next = CategoricalParams<SensorMode>::fromWeights(0.9, 0.1);
MarkovChain<SensorMode>::memoryless(next.toDensity().value()).matrix();
// 0.9 0.1
// 0.9 0.1
```

### One step per prediction

The chain moves once per `predict()`, whatever `dt` is, including zero:

```cpp
// persistent(0.9), starting in Cruise with probability 1
a.predict(1.0);                   // P(Turn) 0.1
b.predict(0.5); b.predict(0.5);   // P(Turn) 0.18
a.predict(0.0);                   // P(Turn) 0.18: a second step, though no time passed
```

A Kalman filter's `predict(0.0)` changes nothing, so a caller who predicts to a common time before
each of several sensors' fixes gets one chain step per call here. With a fixed sensor rate the
transition matrix is tuned to the rate, and this is the usual discrete-time IMM. With
asynchronous sensors, or a rate that changes, the mode switching rate silently changes with
them. The roadmap (step 5b) notes it, and the `SwitchingModel` comment now says it.

Since step 15 [emissions] the library has the continuous-time chain this calls for,
`ContinuousTimeMarkovChain`, a generator $Q$ whose `transition(dt)` is the `MarkovChain` with
$\Pi = e^{Q \, \Delta t}$. `SwitchingModel` still takes only a `MarkovChain`, but the chain is
mutable through `model().chain()`, so a caller can set it before each prediction:

```cpp
auto const rates = ContinuousTimeMarkovChain<Maneuver>::persistent(0.1);  // per second
auto const step = [&](auto & filter, double dt) {
  filter.model().chain() = rates.transition(dt);
  filter.predict(dt);
};
step(a, 1.0);                  // P(Turn) 0.0906
step(b, 0.5); step(b, 0.5);    // P(Turn) 0.0906: the same, as a rate should give
step(a, 0.0);                  // P(Turn) 0.0906: transition(0) is exactly the identity
```

`transition(0.0)` comes out exactly the identity, so the IMM's identity check (§5) skips the
mixing and leaves the probabilities alone: `predict(0.0)` is a no-op without a special case.

> **Q1.** Give the chain a continuous-time form? Options: (a) document the step-per-predict rule
> in the `predict()` comment and leave it; (b) a chain built from a rate matrix $Q$ (rows
> summing to zero), with `SwitchingModel` taking either kind and computing $\Pi = e^{Q \, \Delta
> t}$ on each prediction; for the 2 to 8 modes IMM uses, the exponential is cheap, and
> $e^{0} = I$ makes `predict(0.0)` a no-op without a special case; (c) skip the chain step when
> `dt` is zero, which fixes the common surprise but not the rate dependence. I'd do (a) now and
> (b) when asynchronous sensors come up. A `persistent` rate chain would take a mean sojourn time,
> the quantity people actually know.
> *Update (refresh):* (a) is done, in the `SwitchingModel` comment and [imm]. The rate chain of (b)
> now exists as `ContinuousTimeMarkovChain` [emissions], and the roadmap's step 15 notes say
> "`SwitchingModel` may later take this chain in place of $\Pi$"; the lambda above is that
> feature written by the caller. What remains is (b)'s `SwitchingModel<..., ContinuousTimeMarkovChain>`,
> whose filter would take `predict(dt)` and step the chain by $e^{Q \, \Delta t}$, as
> `HiddenMarkovFilter` already decides between `predict()` and `predict(dt)` by its chain's kind.
> Its `persistent(rate)` takes a rate, not a sojourn time; the mean sojourn is its reciprocal.
> *Update:* both chains now have `fromDwellTimes(dwell, exits)`, which takes the mean sojourn
> times and where each value goes on leaving. The `SwitchingModel` part is still open
> (rechecked on 2026-11-04: `SwitchingModel` takes only a `MarkovChain`).

## 3. HybridParams and Hybrid

`HybridParams<Component, ModeSpace, Allocator = DefaultAllocator>` [component-first], with
`HybridGaussianParams<Space, ModeSpace, Scalar = double, Allocator = DefaultAllocator>` its alias
for `HybridParams<Gaussian<Space, Scalar>, ModeSpace, Allocator>`, holds the parameters of
$p(x, m) = P(m) \, p_m(x)$, for Gaussians $P(m) \, \mathcal{N}(x; \mu_m, \Sigma_m)$: a
`CategoricalParams` over the modes, whose weights are proportional to $P(m)$ and need not sum to
one, as a mixture's weights need not, and exactly one component per mode, in flat-index order.
Its `toDensity()` checks the weights, factorizes every component, and gives the density, a
`Hybrid` (`HybridGaussian` for Gaussians), which evaluates `logDensity()` and gives the moments and
the collapses [params-and-density].

| Member | Does |
|---|---|
| `HybridParams(modes, component, allocator = {})` | every mode starts at `component` |
| `HybridParams(modes, conditionals)`, `HybridParams(modes, range, allocator)` | one per mode, a `std::vector` from the allocator or any range copied in; asserts the count and that they share a space |
| `HybridParams(other, allocator)` | copied or moved into storage from `allocator` [allocation] |
| `modes()`, `setModes()` | the mode weights, a `CategoricalParams` proportional to $P(m)$; `setModes()` asserts the mode space |
| `conditional(m)`, `conditional<Values...>(at<Axis>(i)...)` | the component given a mode, by flat index or by named values and indices of unnamed axes (via `valueOf`), read-only; deleted on an rvalue [view-stamp] |
| `setConditional(m, component)`, `updateConditional(m, f)` | replaces mode `m`'s component, asserting it is in the hybrid's space and copying into the storage held, or calls `f` on it to change it in place, asserting after that `f` kept the space: how the recursion's steps reach a mode |
| `toDensity()`, `toDensity(roots::ridge())` | the `Hybrid` after `factorize()`, or its `NumericError`; deleted on an rvalue [view-stamp] |
| the density's `collapsed()`, `moments()` | the whole state moment-matched into one component, a tangent-space mean |
| the density's `collapse<Path>()` | a `HybridParams` over one mode axis, each value's modes moment-matched |
| the density's `logDensity(x)`, `density(x)` | $\log \sum_m P(m) \, p_m(x)$, read from the factorizations |
| `toMixture()` | an unlabeled `MixtureParams<Component, Allocator>`, one component per mode, for its `toDensity()`, marginals, and sampling |
| `swapConditionals(buffer)` | exchanges the components with a buffer from an equal allocator: how the recursion writes its result without copying |
| `factorize()`, `factorize(roots::ridge())` | checks the mode weights and factorizes every component; a `NumericError`, the `WeightError` of mode weights that define no distribution or the first `GaussianError`, if one fails |
| `space()`, `modeSpace()`, `size()` (the number of modes), `valueDimension()`, `differenceDimension()`, `get_allocator()` | |

Continuing the run of §7 one fix further, to the end of step 14, mostly in (Cruise, Inlier):

```cpp
auto const & s = filter.state();
s.modes().toDensity()->probabilities();           // 0.961 0.00543 0.033 0.000412
s.conditional<Cruise, Outlier>().mean();          // 140.000 -0.199 10.000 -0.172
auto const density = s.toDensity();
auto const byManeuver = density->collapse<Maneuver>();  // a HybridGaussianParams<State, Maneuver>
byManeuver.modes().toDensity()->probabilities();  // 0.967 0.033
auto const collapsed = density->collapsed();
collapsed.marginal<Position>();                   // a view of a GaussianParams<Position>
auto const mixture = s.toMixture();
mixture.toDensity()->logDensity(collapsed.mean());  // -0.389: the mixture density, for a likelihood or a test
```

`collapse<Path>()` is [reduction-policies]'s "collapse grouped by mode label". [reduction-policies]
says that `Collapse` itself never groups; this is where grouping lives, on the hybrid state
rather than in a policy. A value of an axis with probability zero gets the plain average of its
modes' components, so the result is still one component per value.

The usual starting point is "every mode starts at the same Gaussian", with mode probabilities
that aren't uniform. `CategoricalParams<Mode>{}` is uniform, which for a sensor axis means a 50%
prior on outliers. A product prior from per-axis marginals has to be written out by hand,
in flat-index order:

```cpp
// P(Turn) 0.1, P(Outlier) 0.05, independent
auto const prior =
    CategoricalParams<Mode>::fromWeights(0.9 * 0.95, 0.9 * 0.05, 0.1 * 0.95, 0.1 * 0.05);
// a HybridGaussianParams<State, Mode>; deduce through HybridParams, not the alias [filters-and-recursions]
auto state = HybridParams{prior, initial};
```

> **Q2.** Give `CategoricalParams` on a `DiscreteProduct` a constructor from one `CategoricalParams`
> per dimension, their product, as `MarkovChain` already has for chains:
> `CategoricalParams<Mode>{Mode{}, maneuverPrior, sensorPrior}`? It removes the one place a user
> writes flat-index arithmetic to set up an IMM. A `stationary(chain)` that returns the chain's
> stationary distribution would be the other natural starting prior; I'd add the product
> constructor now and the stationary distribution when someone asks.
> *Update:* `stationary(chain)` exists, for both kinds of chain (§2). For independent axes it
> gives the product prior too, so long as the prior wanted is each axis's stationary one. The
> product constructor is still open.

The hybrid state's `marginal<Position>()` [element-marginals] is a range over the modes: each
element holds the mode's log-weight and its component's `GaussianMarginal`, read in place, and
`copy()` gives a `HybridGaussianParams<Position, Mode>` with the same modes. A named collapse's
`marginal<Position>()` gives the moment-matched one. Since [view-stamp] the view makers are deleted
on an rvalue, so the one-line spellings an earlier draft of this document used no longer compile;
the parameters have to be named first:

```
s.toDensity()->collapsed().marginal<Position>();
error: call to deleted member function 'marginal'
Gaussian.h:847:5: note: candidate function [with Path = Position] has been explicitly deleted
```

The same holds for `s.toMixture().marginal<Position>()`, which before [view-stamp] compiled and
failed a borrow-count assertion in debug builds when the temporary died.

> **Q3 (decided, [element-marginals]).** Add `Hybrid::marginal<Path>()`, returning a `Hybrid` over
> the block's space with the same modes? Added, as a view rather than a copy: a range over the modes
> that gives each mode's log-probability and its component's marginal, with `copy()` for the
> `Hybrid` that keeps the labels. It replaces `s.toMixture().marginal<Position>()`.
> rao-blackwellized-filters.md Q9 asks the same of `GridGaussianParams` and `ParticleGaussianParams`, which need
> their own element type.

The class comment says the hybrid state is also "the state of a grid of Kalman filters, where [the
modes] are cells"; [imm]'s *Why* no longer does ("Every filter that tracks modes keeps one Gaussian
per mode"). Since [rao-blackwellized-core] the grid's state
is `GridGaussianParams`, sparse and separate. The hybrid's users are now the IMM and AMM filters, the IMM
smoother (`HybridGaussianHistory`, [smoothing]), and a `MixtureParams` of hybrids under
`MixtureFilter` ([component-first], [filters-and-recursions]). The `toMixture()` comment, which
offered the mixture "for its logDensity()", now says "for its toDensity(), marginals, and sampling".

> **Q4.** ~~Fix the class comment, and [imm]'s *Why*, to say that the IMM is the user, with the grid
> of Kalman filters having moved to `GridGaussianParams` under [rao-blackwellized-core] because cells
> can't be revived and modes can?~~ *Done:* the `toMixture()` comment, [imm]'s *Why*, and the
> class comment all say so.

### 3a. Hybrids of other components

A `HybridParams` holds any `Component` [component-first]: something with the space and dimension
contract, `factorize()`, and a `ComponentTraits` specialization giving the log-density from the
factorization, the moments, and the collapse of several into one. Three types satisfy it, named by
their densities: `Gaussian`, `SqrtGaussian`, and `Hybrid` itself; the concept is closed to a user's own
types [user-extension-points]. A hybrid of square-root forms is a square-root IMM: the mixing
collapses factors by triangularization, each mode predicts and updates in place as a
`KalmanRecursion` over the form does, `predict()` returns a `std::expected` with the first error
when a mode could not take the step, that mode keeping the component it was to move from while the
others move, and the policy is `Collapse` alone, with no gate (`SQRT_IMM_GATES` in
`tests/Filters/compile_errors.cpp`). Nothing is copied to undo a failure: with the
identity chain each mode moves in place in the state, otherwise the mixed component is moved, not
copied, into the scratch it is predicted in, and a failed update leaves the modes before it
updated. `predictJoint()`, and so the smoother, is the covariance form's. A square-root IMM's
`update(fix)` and `predict(1.0)` compile and run with the declarations above
(`HybridParams{CategoricalParams<Maneuver>{}, toSqrt(initial).value()}`).
The test `SqrtHybridComponentsTrackTheCovarianceForm` (`mixture_filter.cpp`) checks that a mixture
of square-root hybrids tracks the covariance form's to $10^{-9}$. The information forms are not
components, and a hybrid of one fails a `static_assert`:

```
HybridParams<InfoGaussian<State>, Maneuver> h{CategoricalParams<Maneuver>{}, toInfo(initial).value()};
error: static assertion failed: a Gaussian in an information form is not a hybrid's conditional,
       having no collapse; convert each with toGaussian() or toSqrt() and hold a Gaussian or a
       SqrtGaussian (decisions.md [gaussian-forms])
```

[component-first] and [gaussian-forms] say so.

> **Q13.** *Decided: the square-root form is a component, so a square-root IMM compiles: the mixing collapses factors, each mode predicts and updates in place, and `predict()` returns a `std::expected`. The information forms are not.* ~~[component-first] and [gaussian-forms] disagree on whether the forms are components, and
> the code sides with [gaussian-forms]. Amend [component-first]'s sentence to [gaussian-forms]'s
> ("will be, once each has a collapse and a per-component step"), so the decision doesn't promise a
> square-root IMM that doesn't compile? Making the square-root form a component is new work and a
> question of its own; this one is only the text.~~

`Hybrid<HybridGaussian<...>, ...>`, a hybrid of hybrids, satisfies the concept, but nothing runs
a filter over it, and a `MixtureParams` of hybrids is the composition the library uses
(`MixtureFilter.h`).

## 4. SwitchingModel

`SwitchingModel<ModeSpace, Scalar, Path, Models...>` holds the chain and the process models,
the Markovian switching of [Blom and Bar-Shalom (1988)](../reference/bibliography.md#blom1988),
whose jump Markov system lets the mode select the dynamics. The mode space comes first because
the type has no continuous space ([component-first]), and the scalar second because it ends in a
pack. It is normally deduced. Three ways to assign models:

```cpp
auto const chain =
    MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<SensorMode>::persistent(0.8)};

auto perMode = SwitchingModel{chain, cv(0.01), cv(0.01), cv(4.0), cv(4.0)};   // modelIndex 0 1 2 3
auto byManeuver = switchingModel<Maneuver>(chain, cv(0.01), cv(4.0));  // modelIndex 0 0 1 1
auto shared = SwitchingModel{MarkovChain<SensorMode>::persistent(0.9), cv(0.01)};  // every mode: model 0
```

Selecting by axis keeps a two-model maneuver set at two models when a sensor axis is
added to the mode, which is [imm]'s reason for it. `modelIndex(mode)` says which model a mode
follows, `visit(mode, f)` calls `f` on it, and `model<I>()` and `chain()` are mutable, so a
caller can retune between steps (§2 sets the chain that way). `partitions(space)` checks that every
model writes each coordinate of the space once, as the filter's constructor asserts.

The models are all fixed-step models over one step [fixed-step-models], or none are. With fixed-step
models the switching model is fixed-step too, its filter's `predict()` takes no time step, and each
mode predicts by its own model's method:

```cpp
auto const cruise = fixedStep(cv(0.01), 1.0);
auto const turn = fixedStep(cv(4.0), 1.0);
auto filter = ImmFilter{SwitchingModel{MarkovChain<Maneuver>::persistent(0.95), cruise, turn},
                        HybridParams{CategoricalParams<Maneuver>{}, initial}};
filter.predict();            // one step; MethodType is ModelsMethod, each mode predicting by its model's
filter.predict(Steps{3});    // three steps, the chain stepping each time
```

The wrong number of models is caught at compile time for a fixed-size axis:

```
SwitchingModel{chain, cv(0.01), cv(4.0)}       // 2 models for 4 modes
SwitchingModel.h:107:17: error: static assertion failed due to requirement
    'DiscreteProduct<Maneuver, SensorMode>::Cardinality == Eigen::Dynamic || DiscreteProduct<Maneuver, SensorMode>::Cardinality
    == int(sizeof...(Models))': give one process model per value of the axis that selects it, or no path for one model shared by every mode

switchingModel<Maneuver>(chain, cv(0.01), cv(4.0), cv(1.0))   // 3 models for 2 values
SwitchingModel.h:107:17: error: static assertion failed due to requirement 'Maneuver::Cardinality == ... == int(sizeof...(Models))'
```

The first message could suggest `switchingModel<Axis>()`, which is likely what was meant
with a product mode space. One model is shared by every mode only when no path is given, the path
being the whole mode space; with a path that names an axis it is one model per value of that axis,
and its count is checked:

```cpp
auto const m = switchingModel<Maneuver>(chain, cv(0.01));  // 1 model for the 2 maneuvers
SwitchingModel.h:107:17: error: static assertion failed due to requirement 'Maneuver::Cardinality == ... == int(sizeof...(Models))'
```

Modes that differ only in how they are measured, such as a sensor's inlier and outlier states,
share one model by giving no path: `SwitchingModel{chain, cv(0.01)}`, as
`ImmFilter.RidesOutABurstOfOutliers` does. An axis of one value with one model still compiles, the
count being right.

> **Q5.** *Decided: one model is shared only when no path is given; a path that names an axis
> checks the count, and the message says to give no path for one model shared by every mode
> (`SWITCHING_AXIS_ONE_MODEL` in `tests/Models/compile_errors.cpp`).* ~~Bug, I think. Treat one
> model as shared only when no path is given (the path is the whole mode space), and check the
> count as usual when `switchingModel<Path>()` names a dimension? A dimension of one value with
> one model would still compile.~~

### A common state space

[switching-models] puts every mode on one state space, with a mode that doesn't use a block holding
it, "for example, acceleration held at zero with small Q". The library has two built-in models that
can do the holding, and they hold differently. A CV/CA pair on a 1-D position, velocity, and
acceleration, one prediction from acceleration 2:

```cpp
auto const steady = compose(ConstantVelocity<P, V>{0.01}, Constant<A>{});
auto const decaying =
    compose(ConstantVelocity<P, V>{0.01}, OrnsteinUhlenbeck<A>{TimeConstant{0.5}, StationaryVariance{1e-4}});
auto const accelerating = ConstantAcceleration<P, V, A>{1.0};
// steady mode, Constant<A>:            (0, 0, 2)
// steady mode, OrnsteinUhlenbeck<A>:   (0, 0, 0.2707)
// accelerating mode:                   (1, 2, 2)
```

`Constant` keeps the acceleration with no noise, so the CV mode carries 2 m/s² that it ignores, and
mixing hands it back to the CA mode at the next step. An Ornstein-Uhlenbeck model with a short time
constant pulls it toward zero, which is closer to [switching-models]'s "held at zero". Neither sets
it to zero.

> **Q6.** Which is the recommended model for a block a mode doesn't use? I'd say
> `OrnsteinUhlenbeck` with a time constant shorter than the step and a small variance, write that
> into the `SwitchingModel` comment, and amend [switching-models]'s example to name it, since "held
> at zero with small Q" describes no model that exists. A `Zero<Path>` model that resets to zero
> with a given variance is the other option; since roadmap 17z it is a `processModel()` whose
> function returns zero, with `reads(path<A>)`, `writes(path<A>)`, and `additiveNoise()` of the
> variance [model-policies], which predicts $(0, 0, 0)$ from the state above with variance $10^{-4}$
> on $A$. A model that reads nothing, `reads()`, does not compile: its input space would be an empty
> `Composite`, which fails `a composite needs at least one block` deep in the instantiation.

> **Q7.** ~~One method serves every mode, so a CV/coordinated-turn IMM runs the CV mode through
> `Unscented` too (exact for an affine model, at $2n + 1$ sigma points). Leave it?~~ *Decided
> [switching-models]: a method per model, and by default the same method for every mode.* *Done:*
> `withMethod(model, method)` gives a mode's model its own method, below. The library has no
> coordinated-turn model yet, so the classic pair still needs a model of the user's own, a
> `processModel()`.

A mode predicts by the filter's method unless its model carries one of its own. A fixed-step model
always does, and `withMethod()` gives one to a model that isn't fixed-step, as a switching model's
argument only:

```cpp
auto filter = ImmFilter{SwitchingModel{MarkovChain<Maneuver>::persistent(0.95), cv(0.01),
                                       withMethod(turn, methods::unscented())},
                        HybridParams{CategoricalParams<Maneuver>{}, initial}};
// MethodType is Linearized: the cruise mode's; the turn mode predicts by Unscented
```

`model<1>()` returns `turn` itself, so retuning it is unchanged. `AmmFilter{initial, cv(0.01),
withMethod(turn, methods::unscented())}` takes it the same way, and so does a
`RbParticleFilter` or `RbPointMassFilter` over a switching model. `ImmFilter.EachModePredictsByItsOwnMethod`
checks that each mode matches a Kalman filter with its model and method, with a drag model that
`Linearized` and `Unscented` predict 0.2 apart.

Fixed-step models may now differ in their methods, and the filter takes none:

```cpp
auto const cruise = fixedStep(cv(0.01), 1.0);
auto const turn = fixedStep(cv(4.0), 1.0, methods::unscented());
auto filter = ImmFilter{SwitchingModel{MarkovChain<Maneuver>::persistent(0.95), cruise, turn}, ...};
filter.predict();  // cruise by Linearized, turn by Unscented; MethodType is ModelsMethod
```

Naming another method in the type of a filter with fixed-step models is still an error, even the
one every model carries (`IMM_FIXED_STEP_METHOD` in `tests/Filters/compile_errors.cpp`):

```
ImmFilter<Gaussian<State>, SwitchingModel<Maneuver, double, Maneuver, Fixed, Fixed>, methods::Linearized>
ImmFilter.h:314:9: error: static assertion failed due to requirement 'internal::predictsBy<...>':
    a filter with fixed-step models predicts each mode by its model's own method, so it takes
    ModelsMethod, the default, as its method
```

> **Q14.** ~~Either allow fixed-step models with different methods in one switching model (the
> recursion already predicts each mode by its own model's method; `method()` would stay the
> first's, or become meaningless), or check it in `SwitchingModel` with "a switching model's
> fixed-step models must share one method", beside its existing "must share one step"?~~ *Decided
> with Q7: allow them.* *Done:* the error is gone, and the filter's `method()` and `MethodType` are
> `ModelsMethod`, the empty tag, since no mode predicts by the filter's [switching-models].

## 5. predict(): mixing

`predict(dt, inputs...)` does three things, the standard IMM cycle of
[Blom and Bar-Shalom (1988)](../reference/bibliography.md#blom1988), in the form of
[Bar-Shalom, Li, and Kirubarajan (2001)](../reference/bibliography.md#barshalom2001) §11.6.6.
The code takes the four steps of that section (interaction or mixing, mode-matched filtering,
mode probability update, and combination) and departs in two ways: the combined estimate is not
computed each cycle but read on demand as the state's `collapsed()`, and the mixing weights and
mode probabilities are kept as logarithms.

1. **Mixing.** The predicted mode probabilities are $c_j = \sum_i \Pi_{ij} \mu_i$, and the
   probability that the target was in mode $i$ given that it is now in $j$ is
   $\mu_{i \mid j} = \Pi_{ij} \mu_i / c_j$. Mode $j$ starts from the moment-matched collapse of
   every mode's component weighted by $\mu_{i \mid j}$: for Gaussians a tangent-space mean
   $\bar x_j$, and
   $\sum_i \mu_{i \mid j} \left(\Sigma_i + (x_i \boxminus \bar x_j)(x_i \boxminus \bar x_j)^T\right)$.
   The weights are formed in log space; modes of weight zero are left out, and a single
   nonzero weight returns a copy.
2. **Per-mode prediction.** Each mixed component goes through its mode's process model with the
   filter's method, and the call's inputs go to every mode, each matched by its tag to what the
   mode's model reads [prediction-inputs].
3. **Mode probabilities.** They become $c_j$, by `predict(modes, chain)`.

The mixed and predicted components are written into the recursion's scratch, from the state's
allocator, and swapped into the state, so on a bounded space a prediction allocates nothing
(`Allocation.ImmFilterStepsWithinItsResource`).

A test checks all three over five cycles against a plain-Eigen IMM on the four modes of `Mode`,
to $10^{-12}$ in every mode's probability, mean, and covariance (condensed from
`MatchesAHandComputedImmWithMarkovOutliers`).

A mode the chain can't reach ($c_j = 0$) starts from its own component, which then has no weight
anywhere. With $\Pi = \begin{pmatrix} 1 & 0 \\ 0.5 & 0.5 \end{pmatrix}$ from certain cruise,
`Turn` has probability 0 after the step, and its Gaussian was predicted by the agile model
(position variance 15.33, against cruise's 14.00). An update still computes its likelihood, which
contributes nothing.

Since step 17c [autonomous-multiple-model], a chain whose matrix is exactly the identity skips the
mixing: each mode's component is predicted as it is and the mode probabilities are left as they
were, bitwise. The mixing would give the same components, and the same probabilities up to the
rounding of the chain's step through exponentials and logarithms
(`AmmFilter.SkipsTheMixingAndAgreesWithTheFullCollapse`, against a chain with off-diagonal entries
of $10^{-300}$). The check costs a comparison of the matrix per prediction. That is the autonomous
multiple model estimator, §11.

## 6. update(): reweighting by likelihood

`update(measurement, policy = reductions::collapse())` does for each mode $m$, in flat-index order:

1. `Emrys::update(conditional(m), measurement)`, one branch per noise component, or, when an
   innovation covariance is not positive definite, the end of the update, which returns
   `GaussianError::NotPositiveDefinite` with the state unchanged (for the square-root IMM, with
   the modes before it updated) [innovation-cholesky];
2. labels every branch with `mode = m`;
3. passes the branches to the policy, which must keep one of the component type, returned or added
   to an output that holds one;
4. keeps it, with the likelihood of the branches it stands for, $\log p(z \mid m)$ plus the
   log-weight the policy kept it with (zero unless the policy dropped branches, as a gate does), or
   on rejection the mode's prior component with likelihood zero.

The mode probabilities then go through Bayes' rule, $\mu_m \propto c_m \, p(z \mid m)$, by
`CategoricalParams::update()`, which normalizes them. The evidence is
$\log p(z) = \log \sum_m c_m \, p(z \mid m)$, over every branch.

A sideways maneuver of 3 m/s² from step 11 to 16, with exact position fixes, the
`persistent(0.95)` filter of §1, and a cruise-only Kalman filter alongside:

```cpp
for (int k = 1; k <= 20; ++k) {
  // truth: 10 m/s along x; 3 m/s^2 along y for 10 < k <= 16
  filter.predict(1.0);
  auto const update = filter.update(fix(truth));
  filter.state().modes().toDensity()->probability<Turn>();
  filter.state().toDensity()->collapsed().mean();
}
//        P(Turn)  error  KF error  log p(z)  log p(z|m)        nis
// k  9    0.032  0.000     0.000    -2.456   -2.405  -3.385    0.000   0.000
// k 11    0.045  0.869     0.936    -3.060   -3.026  -3.601    1.341   0.494
// k 12    0.950  0.937     3.355    -7.701  -10.605  -5.351   16.517   3.719
// k 14    0.940  0.561     9.996    -4.935   -5.349  -4.902    3.852   2.147
// k 16    0.949  0.569    16.319    -4.998   -5.652  -4.947    4.443   2.227
// k 18    0.833  0.095    17.454    -3.796   -3.479  -3.848    0.275   0.032
// k 20    0.461  0.012    13.449    -3.467   -3.010  -3.824    0.000   0.002
```

One fix into the maneuver, the cruise mode's NIS is 16.5 against the agile mode's 3.7. The
chain predicts a turn probability of 0.09, and the likelihood ratio of
$e^{-5.351 + 10.605} = 191$ takes it to 0.95. After the maneuver the cruise mode wins back
slowly, as the persistent chain says it should. The cruise-only Kalman
filter is 16 m behind at the end of the maneuver. The test suite runs the same scenario with
noisy fixes and checks the maneuver probability (above 0.8 during it, below 0.3 before it) and a
position error at most half the Kalman filter's.

### ImmFilterUpdate

`ImmFilterUpdate<ModeUpdate, Scalar, Allocator>`, generic in one mode's update as
`MixtureFilterUpdate` is in one component's:

| Field | Is |
|---|---|
| `modes` | `internal::Buffer<ModeUpdate, Allocator>` (a `std::vector` from the allocator), one per mode in flat-index order: a `FilterUpdate<MeasurementSpace, 1, Scalar>` for Gaussian noise and a `ReusableFilterUpdate<Space, MeasurementSpace, Scalar, Allocator>` for a mixture, whose scratch keeps the mode's measured branches between fills; its branches before reduction, each labeled with the mode, its `logLikelihood()` $\log p(z \mid m)$, and whether the policy accepted it |
| `logLikelihood` | $\log p(z) = \log \sum_m c_m \, p(z \mid m)$ with the predicted probabilities, over every mode |
| `accepted` | whether any mode accepted; when false, the state is unchanged |

A result is made from an allocator, `ImmFilterUpdate{allocator}`, and gives `get_allocator()`.

Unlike the mixture filter's result (`MixtureFilterUpdate::posteriorIndices()`), this one lines up with the
state: modes are never dropped, so `result.modes[m]` is about `state().conditional(m)`. The
posterior mode probabilities are only in the state. Noise by mode axis gives the same type with
`Count` 1, since each mode keeps one branch.

`update(measurement, policy, result)` refills a result instead of returning one, and returns
`std::expected<void, GaussianError>` [innovation-cholesky], so a loop that passes the same result
allocates nothing once it has been as large. The filter has no maker for that result; the
recursion's does the job:

```cpp
using Fix = decltype(fix({0, 0}));
auto result = filter.recursion().emptyUpdate<Fix>(filter.get_allocator(), filter.state().size());
for (int k = 1; k <= 3; ++k) {
  filter.predict(1.0);
  if (!filter.update(fix({10.0 * k, 0}), reductions::collapse(), result)) { ... }
}
// 2 modes, accepted 1, log p(z) -3.445
```

The allocation tests build it by hand instead, `auto result = Update{allocator};` and a `reserve()`.

> **Q15.** Forward `emptyUpdate<M>()` and `emptyAxisUpdate<M>()` from the filter, defaulting the
> allocator and mode count to the state's, so that reusing a result is one line? Whatever the
> answer, the Kalman and mixture filters should spell it the same way (the mixture filter's
> maker is a private `emptyResult()`); `KalmanFilter.h` is where that convention lives.

### Gating and rejection

A gate tests each mode's branches against that mode's own innovation covariance. Condensed from
`RejectedModesLoseTheirProbability`: a confident mode and a vague one, both at the origin, and a
fix 30 m away:

```cpp
auto const gated = filter.update(fix({30, 0}), reductions::gate(0.997));
// accepted 1; per mode accepted 0, 1; P(Turn) 1
// log p(z|m) -302.2, -8.954; log p(z) -9.647
filter.predict(1.0);
// P(Cruise) 0.1; the cruise Gaussian's mean x 29.93
```

The rejected mode keeps its prior Gaussian with probability zero, and the chain gives it 0.1
back at the next prediction. Its stale Gaussian at the origin is not what comes back, though:
mixing weights mode $i$ by $\mu_i$, which is zero, so the revived cruise mode starts from the turn
mode's Gaussian, near the fix. That is the right outcome, and worth a sentence in the class
comment, which says only that "the chain can bring it back". When every mode rejects, the
measurement is rejected and the state is unchanged, as in every filter [update-outcomes].
On the identity chain nothing brings the mode back (§11).

The reported $\log p(z)$ includes the rejected mode's likelihood ($\log(0.5 e^{-302.2} + 0.5
e^{-8.954}) = -9.647$), as the Kalman and mixture filters report the likelihood before the
gate. When a gate drops some of a mode's noise branches but not all, the mode is reweighted by
the likelihood of the branches that pass, through the log-weight the gate's inner policy keeps its
result with, exactly the mixture filter's case [reduction-policies]. Here the gate drops a whole mode's one branch, so
nothing changes.

> ~~**Q8.** Follow the mixture filter's weighting of a partly gated component here, whatever
> it is? The code path is the same shape: `updateModes()` reads `measured.logLikelihood()` from
> before the policy. I see no reason for the IMM to differ.~~ *Decided with the
> mixture filter:* `updateModes()` adds `internal::logKeptWeight(policy, measured)` to the mode's
> log-likelihood, and the evidence stays over every branch. Since a policy adds what it keeps to
> the filter's output [reduction-policies], it adds the log-weight the policy kept the mode's
> component with.

## 7. Markov outliers: noise by mode axis

Mixture noise in a Kalman filter treats each fix's outlier state as independent of the last
one's. A bias that corrupts several fixes in a row, or a sensor that sticks, is better described
by a chain. The measurement's noise says so itself: `additiveNoise(mixture).byMode<SensorMode>()`
is a `ModeNoise` (`Models/Policies.h`), an `AdditiveNoise` whose components are the values of the
mode axis `SensorMode`, in the axis's order. An IMM whose mode space has that axis updates each
mode with only the component that the mode's value names, with its own likelihood. The
components' weights are ignored there; the chain on that axis supplies the probabilities instead.
Since Q10 was decided, this is how the axis is named; `update<Path>()`, which named it at the call,
is gone. In the examples below, `markovFix(z)` is `robustFix(z)` with its noise by `SensorMode`:

```cpp
// markovFix(z): robustFix(z)'s mixture as additiveNoise(mixture).byMode<SensorMode>().
auto markovFix(Eigen::Vector2d const & z);
```

The same noise means something in every filter, by what the filter's state has [imm]:

| The filter's state | Noise by mode axis is read as |
|---|---|
| no mode space (Kalman, Gaussian mixture, particle, a Rao-Blackwellized filter of Gaussians) | the mixture it is, with its weights: memoryless outliers |
| a mode space with the named axis (IMM, AMM, a mixture of IMMs, a Rao-Blackwellized filter of hybrids) | the axis's values: Markov outliers, with the chain's probabilities |
| a mode space without the named axis | a compile-time error (below) |

So one sensor model serves the outlier-robust Kalman filter and the IMM [imm]. For the two readings
to describe one sensor, the weights should be the chain's stationary probabilities on the axis.

On the full `Mode`, with the prior of §3, a 30 m bias on fixes 8 to 10, and exact fixes
otherwise:

```cpp
auto filter = ImmFilter{switchingModel<Maneuver>(chain, cv(0.01), cv(4.0)), HybridParams{prior, initial}};
for (...) {
  filter.predict(1.0);
  auto const update = filter.update(markovFix(z));
}
//              P(Outlier)  P(Turn)  error  components  labels
// k  7            0.005     0.035   0.000   0 1 0 1    0 1 2 3
// k  8 burst      1.000     0.091   0.332   0 1 0 1    0 1 2 3
// k  9 burst      1.000     0.227   1.664
// k 10 burst      0.997     0.671   9.176
// k 11            0.516     0.230   3.063
// k 12            0.058     0.035   0.212
// k 13            0.009     0.025   0.035
```

(Printed on 2026-11-04 by `update<SensorMode>(robustFix(z))`, which ran the same arithmetic as
`update(markovFix(z))` does now: `updateByAxis()` is unchanged but for finding the axis in the
noise. Not rerun for Q10.)

Each mode's single branch carries the noise component it used (`component`, 0 for the inlier
modes and 1 for the outlier ones) and its mode label. The first biased fix is classified as an
outlier at once, and the estimate stays within 2 m for two fixes. By the third, a persistent
offset is also explained by a maneuver, and the turn probability climbs to 0.67, drifting the
estimate 9 m. That is the model, not the code: a maneuver and a run of biased fixes look alike
after a few steps, and the chain's persistences decide which wins. The test suite's
`RidesOutABurstOfOutliers` uses a sensor axis alone (no maneuver) and holds within 10 m
during a 50 m burst, against more than 30 m for a plain Kalman filter.

### Memoryless outliers are the robust Kalman filter

The Filters module page says memoryless mixture noise is the IMM case where every row of $\Pi$ is
the same. It holds exactly here, and since Q10 the two filters take the same measurement. With a
memoryless chain on `SensorMode` matching the noise weights, a target at 10 m/s along x, and six
exact fixes of which the fourth is 25 m off:

```cpp
auto const rate = CategoricalParams<SensorMode>::fromWeights(0.9, 0.1);
auto const memoryless = MarkovChain<SensorMode>::memoryless(rate.toDensity().value());
auto imm = ImmFilter{SwitchingModel{memoryless, cv(0.01)}, HybridParams{rate, initial}};
auto robust = KalmanFilter{cv(0.01), initial};
// imm.update(markovFix(z)) and robust.update(markovFix(z)) at each fix:
// P(Outlier)        0.0144 0.00599 0.00442 1 0.00559 0.00269
// KF outlier weight 0.0144 0.00599 0.00442 1 0.00559 0.00269
// max difference in collapsed mean and covariance 1.1e-16, in log p(z) 4.4e-16
```

(Printed on 2026-11-04, with `update<SensorMode>(robustFix(z))` for the IMM and `robustFix(z)` for
the Kalman filter; the first difference was 8.9e-16 on 2026-10-27; both are rounding.)

Mixing with identical rows starts both modes from the same collapsed Gaussian, which is exactly
what `Collapse` hands the Kalman filter. The test `NoiseByModeIsMemorylessWhereNoModeSpaceHasItsAxis`
checks this to $10^{-12}$ with one measurement for both filters, and checks that the Kalman filter
updates by `markovFix(z)` exactly as by `robustFix(z)` (Q9).

> ~~**Q9.** Add this equivalence as a test in `tests/Filters/imm_filter.cpp`? It checks that claim,
> and it pins down both `update<Path>()` and the mixing in one comparison with an independent
> filter. I'd add it. (Still absent on 2026-11-04.)~~ *Done with Q10:*
> `ImmFilter.NoiseByModeIsMemorylessWhereNoModeSpaceHasItsAxis`.

### A measurement per value of an axis

Noise by mode axis lets the mode change a measurement's noise and nothing else. A sensor whose
bias, scale factor, or function depends on the mode needs a whole measurement per value of the
axis. `switchingMeasurement<Path>(m0, m1, ...)` holds them, in the axis's order, and `update()`
updates each mode by the one its `Path` value selects, as `switchingModel<Path>()` selects a
process model [imm]. Condensed from `UpdatesEachModeByTheMeasurementItsValueSelects`:

```cpp
// the Inlier modes see the plain fix; the Outlier modes a fix through a function of its own,
// biased by (3, -1) and with four times the noise
auto const update = filter.update(switchingMeasurement<SensorMode>(fix(z), biasedFix(z, bias, 4.0)));
// matches a hand IMM update with each mode's own offset and noise to 1e-12
```

The measurements may differ in model type, function, noise, and method. Every mode's update
fills one `ImmFilterUpdate`, so they need one result type: one measurement space, and noise that
is Gaussian in all of them or a mixture in all of them. Otherwise:

```
static assertion failed: the measurements of a switching measurement update a mode into one type:
give them one measurement space, and noise that is Gaussian in all of them or a mixture in all of them
```

A count that differs from the axis's cardinality fails to compile for a fixed axis ("give one
measurement per value of the axis that selects it") and asserts at run time otherwise. A path the
mode space lacks fails to compile, as noise by a missing axis does, and so does a measurement in it
whose noise is by mode axis, which would select twice (Q20). With Gaussian noise in each,
`switchingMeasurement<SensorMode>(fix(z), outlierFix(z))` is the same Markov outlier model as
`markovFix(z)`, but without the `component` label on each mode's branch, and with a measurement
that a Kalman filter can't read.

### Unmarked mixture noise in an IMM

The same filter takes ordinary updates. With Gaussian noise every mode uses it, which is the
right update for a sensor that has no outlier axis. With mixture noise that is not by mode axis,
each mode runs the memoryless outlier model inside itself, and the sensor axis learns nothing.
Three copies of the filter above, given the same exact fixes of a target at 10 m/s:

```cpp
if (!byMode.update(markovFix(z))) { ... }      // the Markov outlier model
if (!unmarked.update(robustFix(z))) { ... }    // 2 branches per mode
if (!gaussian.update(fix(z))) { ... }          // 1 branch per mode
//       P(Outlier): update(markovFix)  update(robustFix)  update(fix)
// k 1                      0.038             0.230             0.230
// k 2                      0.016             0.338             0.338
// k 4                      0.008             0.442             0.442
// k 8                      0.005             0.492             0.492
// branch weights in each mode of update(robustFix): 0.986 0.014 at k 1, 0.998 0.002 at k 8
```

(Printed on 2026-11-04, the first column by `update<SensorMode>(robustFix(z))`.)

The second column is still legal: an IMM with a `SensorMode` axis that receives unmarked mixture
noise double-models the outliers, and its `P(Outlier)` is then the chain drifting toward its
stationary 0.5, exactly as with no outlier model at all. Nothing warns, because unmarked noise is
the memoryless model in every filter. What Q10 changed is where the choice is made: once, in the
sensor's model, rather than at every call, and a measurement whose noise names an axis always
updates that axis in a filter whose mode space has it.

> ~~**Q10.** Let the measurement say which mode dimension its noise components are, so that
> `update(fix)` does the right thing and the path can't be forgotten? For example
> `additiveNoise(mixture).modes<SensorMode>()`, or a `markovNoise<SensorMode>(mixture)` helper, carried
> in the measurement model's type. A Kalman or mixture filter would ignore it, keeping [imm]'s "one
> measurement type serves both". The IMM would use the named dimension when its mode space has
> one, and fail to compile when it doesn't. Two sensors would name their own dimensions
> (`GpsState`, `RadarState`), which is how a mode space with several sensors already has to be
> written. `update<Path>()` could then go, which also retires its spelling: `update<SensorMode>(fix)`
> reads like a by-tag update of a `SensorMode` *block*, by analogy with `marginal<Position>()`, and
> `update<Position>(fix)` fails with "no member named 'cardinality' in 'Position'" before the
> clearer "the space does not have this axis". I'd do it, keeping `update<Path>()` until the
> tagged form exists.~~ *Decided [imm] (2026-11-12):* the noise is
> `additiveNoise(mixture).byMode<SensorMode>()`, a `ModeNoise`, and the reading depends on the
> state, as the table above says: a filter with no mode space reads the mixture with its weights
> (not "ignores" the tag: it runs the memoryless model of the same sensor), a mode space with the
> axis updates by it, including a mixture of IMMs and the Rao-Blackwellized filters of hybrids,
> which run `ImmRecursion`, and a mode space without the axis is a compile-time error, since a
> filter that could carry the axis but doesn't most likely lost it or misspelled it. `update<Path>()`
> is removed from `ImmRecursion`, `ImmFilter`, `MixtureFilter`, `RbParticleFilter`, and
> `RbPointMassFilter`; `ImmRecursion::ResultType<M>` is one branch per mode for such noise.

Noise by mode axis needs one component per value of the axis, which a mixture's run-time size
can't show at compile time: the wrong count fails at run time, in debug builds, `the noise needs
one component per value of the mode axis` (`ImmFilter.h`, `updateByAxisIn()`). `byMode()` on
anything but one unplaced mixture fails where the noise is built, and a filter whose mode space
lacks the axis fails at its update, each with one error (rerun with Clang 21 for Q10):

```
additiveNoise(GaussianParams<Linear<2>>{...}).byMode<SensorMode>();
Policies.h:463:24: error: static assertion failed due to requirement
    'internal::isGaussianMixture<Emrys::GaussianParams<Emrys::Linear<2, 2>, double>>': byMode() takes additive noise
    that is one mixture over the whole output, with one component per value of the mode axis

filter.update(markovFix(z));  // filter over Maneuver alone
ImmFilter.h:683:13: error: static assertion failed due to requirement 'internal::pathResolves()': the measurement's
    noise is by a mode axis, byMode<Axis>(), that the filter's mode space does not have
```

> **Q11.** *Decided in the code (roadmap 17z): Gaussian noise fails a `static_assert` that names
> the cause, as above; mixture noise keeps the run-time check of its count. The follow-on errors
> remain.* *Since Q10, `byMode()` refuses Gaussian noise where it is built, with no follow-on
> error.* ~~Make that a `static_assert` when both counts are known at compile time? With Gaussian
> noise the branch count is the `std::array`'s 1 (`MeasuredType<M>::BranchCount`), and a `Named`
> dimension's cardinality is a constant, so this misuse is always visible at compile time.
> Mixture noise keeps the run-time check. (If Q10 is adopted, the tagged noise could check its own
> component count where it is built, for a fixed-size dimension.) *Update (2026-11-04):* Gaussian
> noise now fails at compile time, by accident of the code reading the mixture's `size()`; what
> remains is a `static_assert` that says why ("`update<Path>()` needs mixture noise, one component
> per value of the axis"), and the run-time check for mixture noise.~~

## 8. Cost

Per `predict()` with $M$ modes: $M$ mixing collapses, each over up to $M$ Gaussians (modes of weight
zero skipped), none on the identity chain, and $M$ predictions through the modes' models, each a
full transform. Per `update()` with $K$ noise components: $M$ updates of $K$ branches each, with one
transform of the measurement function per mode [additive-noise]. Noise by mode axis updates each
mode with only the noise component its value names (`internal::WithNoiseComponent`), one branch
and one gain per mode; on 2026-10-27 it computed all $K$ branches and kept one.

[switching-models] says "modes whose models differ only in Q reuse one transform of the transition".
That isn't built: every mode transforms through its own model even when two modes differ only in
noise density, as in every example here. With `Linearized` and models whose functions return their
Jacobians, as the built-in models' do, the transform is a matrix product and the cost is negligible;
with `Unscented` it doubles the prediction cost of a two-model IMM. The roadmap lists it under "left
for later".

> **Q12.** Amend [switching-models] to say the shared transform is an optimization not yet written,
> so the decision doesn't describe code that doesn't exist? It would need the switching model to
> know that two models share a transition. Since roadmap 17z a model's function returns its
> Jacobian with each value [returned-jacobians], so the transition is known per model at each call,
> but not before it; telling that two models share a function, such as two `ConstantVelocity` that
> differ only in their spectral density, is the part still missing.

Allocation [allocation]: the recursion keeps two buffers of mode components, the predicted and the
posterior, from the state's allocator, and swaps them with the state's, so neither step builds a
new hybrid. With a reused result (§6), the IMM's and the AMM's steps allocate nothing after a
warm-up on a bounded space, the AMM's floor included (`tests/allocation_mixtures.cpp`). With
mixture noise, each mode's slot in the reused result is a `ReusableFilterUpdate` that keeps its
measured branches as the scratch of its next fill, so those steps allocate nothing after warm-up
either (`Allocation.ImmFilterWithMixtureNoiseStepsWithinItsResource`), as for the mixture filter.

The steps take no execution policy [execution-policies]: the mode count is the mode space's
cardinality, almost always a small constant, and the recursion's scratch makes it non-reentrant,
which is why a `MixtureFilter` of hybrids runs its components in order.

## 9. What misuse looks like

| Misuse | Fails | First error (Clang) |
|---|---|---|
| `Keep` as the policy | at `update()` | `static assertion failed: an IMM filter keeps one component per mode, so its policy must keep one: Collapse, Select, or a Gate over them`, then a follow-on `incompatible operand types` on `ImmFilter.h` line 887 [reduction-policies] |
| wrong model count, fixed-size axis | at construction of the model | `give one process model per value of the axis that selects it, or no path for one model shared by every mode` (§4) |
| one model where a path names an axis | at construction of the model | `give one process model per value of the axis that selects it, or no path for one model shared by every mode` (§4, Q5) |
| a method named in the type of a filter with fixed-step models | at the type | `a filter with fixed-step models predicts each mode by its model's own method, so it takes ModelsMethod, the default, as its method`, from `ImmRecursion` (§4, Q14) |
| `withMethod()` of a fixed-step model | where it is made | `a fixed-step model carries its own method; give the method to fixedStep() in place of withMethod()` (§4) |
| a state over different modes than the model | at construction | `no matching constructor for initialization of 'ImmFilter<...>'` |
| a hybrid of an information form | at the type | `static assertion failed: a Gaussian in an information form is not a hybrid's conditional, having no collapse; ...` (§3a) |
| `byMode<SensorMode>()` on Gaussian noise | where the noise is built | `byMode() takes additive noise that is one mixture over the whole output, with one component per value of the mode axis` (§7, Q11) |
| noise by mode axis with the wrong number of components | at run time, debug only | `the noise needs one component per value of the mode axis` (§7) |
| noise by an axis the mode space lacks, or by a block such as `Position` | at `update()` | `the measurement's noise is by a mode axis, byMode<Axis>(), that the filter's mode space does not have` (§7, Q10) |
| unmarked mixture noise, on a filter with a sensor axis | never | the sensor axis follows the chain alone (§7, Q10) |
| a chain whose rows don't sum to one | at construction, debug only | `each row of the transition matrix must sum to one` |
| `predict(0.0)` to synchronize sensors | never | the chain takes a step (§2, Q1) |
| `s.toDensity()->collapsed().marginal<Position>()`, a view of a temporary | at compile time | `call to deleted member function 'marginal'` (§3, [view-stamp]) |
| an AMM over a product mode with one model per maneuver | at construction of the model | `give one process model per value of the axis that selects it, or no path for one model shared by every mode` (§11) |

The state-mismatch error names no reason. Deduction takes the state type from the model's mode
space, and the given `HybridGaussianParams<State, Maneuver>` then doesn't convert to
`HybridGaussianParams<State, Mode>`. The runtime assertion in the constructor ("the model and the state
have different modes") can only fire for run-time `Finite` axes of different sizes.

## 10. Smaller notes

- Every branch's `mode` is the flat index of its mode's discrete value, an `Eigen::Index` as
  [discrete-spaces] makes a mode [weighted-branches].
- `predict()` takes any `TimeStep<Scalar>` since [time-steps]
  (`filter.predict(std::chrono::milliseconds{500})` moves the chain one step and the Gaussians half
  a second), converted once to the filter's scalar; [time-steps] has the rule. `update()` is
  `[[nodiscard]]` since [innovation-cholesky], as in the other filters that condition a Gaussian.
- ~~`Hybrid` has `moments()` on itself, which `GaussianMixtureParams` lacks.~~ Neither parameters type has them now: the moments and the
  collapses are the density's, `Hybrid` over `HybridParams`, since they read the components
  [params-and-density]. The density's `moments()` still collapses first, so it builds a
  whole component to return its moments.
- `MarkovChain` lives in `Categorical.h` and is documented in the Distributions group, while its
  users are the IMM and `HiddenMarkovFilter`. That is fine as long as `Categorical.h`'s
  `predict(params, chain)` stays with it; no change suggested.
- Transitions are read by flat index (`transition(from, to)`). A by-name form,
  `transition<Cruise, Turn>()` on a `Named` chain, would follow the library's by-tag rule; it is
  the question [tag-addressing] answers for the positional discrete accessors: they stay.
- `MarkovChain{space, matrix}` takes its scalar from the matrix, so an `Eigen::Matrix2f` gives a
  `float` chain, which then fails to combine with a `double` state with an error about
  `predict()`. Rare enough to leave.
- The GCC 14 `-Wmaybe-uninitialized` warning in `GaussianParams`'s move constructor, which every IMM
  example showed on 2026-10-18, is gone, and so is the one in `Categorical::fromLogWeights()` that
  the `FloorModes` examples showed on 2026-10-27: the scratch program of this refresh, every example
  here, builds with GCC 14 at `-O1 -Wall` with no warning.

## 11. The autonomous multiple model filter

When the mode is fixed for all time but unknown, it is a parameter to identify rather than a state
that switches. The estimator for that case is a bank of Kalman filters, one per mode, each running
on its own, and the probability that each mode is the one in force by Bayes' rule over their
measurement likelihoods. `AmmFilter<Component, Model, Method, Allocator>` is that estimator, and [autonomous-multiple-model] makes it the IMM with $\Pi = I$: a class derived from
`ImmFilter` that adds only constructors, which build the identity chain on the state's modes.

It is the multiple model estimator that
[Athans et al. (1977)](../reference/bibliography.md#athans1977), §V.B–C, restate and credit to
Magill (1965): a Kalman filter per model and the posterior model probabilities of their equation
(5.5). Theirs are steady-state filters with constant innovation covariances; here each mode's
component is a full filter's, so its likelihood is exact under the mode.
[Bar-Shalom, Li, and Kirubarajan (2001)](../reference/bibliography.md#barshalom2001), §11.6, set it
beside the IMM. Magill (1965) has no bibliography entry: the roadmap's step 17c notes record that
the paper was not read, so the docs cite it through Athans et al.

```cpp
auto amm = AmmFilter{HybridParams{CategoricalParams<Maneuver>{}, initial}, cv(0.01), cv(4.0)};
amm.model().chain().matrix();   // 1 0 0 1
// derived from ImmFilter<Gaussian<State>, SwitchingModel<Maneuver, double, Maneuver, CV, CV>>

auto const start = HybridParams{CategoricalParams<Maneuver>{}, initial};
auto unscented = AmmFilter{methods::unscented(), start, cv(0.01), cv(4.0)};  // method first
auto shared = AmmFilter{start, cv(0.01)};                                               // one model for every mode
// predict() takes no step
auto fixed = AmmFilter{start, fixedStep<State>(cv(0.01), 1.0), fixedStep<State>(cv(4.0), 1.0)};
```

The state comes first and the models last, because the models are a pack; the method, when
given, comes before the state for the same reason, where the other filters' constructors take it
last. The models are one per mode, in flat-index order, or one shared. A model per value of one
axis of a product mode is not a constructor of the AMM; it is the IMM given
`switchingModel<Maneuver>(MarkovChain<Mode>{}, cruise, turn)`, whose identity chain takes the
same shortcut (the mode probabilities stay at 0.25 each through a prediction). Giving the AMM two
models for four modes fails at the switching model's count check of §4.

`AmmFilter.IsABankOfKalmanFilters` is the evidence for the equivalence: over
six fixes each mode's Gaussian equals a `KalmanFilter`'s with that mode's model, and the mode
probabilities equal the prior times the Kalman filters' likelihoods, normalized, to $10^{-12}$.

### Identification, not tracking

The maneuver of §6, run through the IMM, the AMM, and an AMM floored at 0.01 after each update:

```cpp
auto amm = AmmFilter{HybridParams{CategoricalParams<Maneuver>{}, initial}, cv(0.01), cv(4.0)};
auto floored = AmmFilter{HybridParams{CategoricalParams<Maneuver>{}, initial}, cv(0.01), cv(4.0)};
auto const floor = upkeep::floorModes(0.01);
// each step: predict(1.0), update(fix(truth)), and for floored, floor(floored.state())
//            AMM           floored         IMM persistent(0.95)
//        P(Turn) error  P(Turn) error  P(Turn) error
// k  9    0.000  0.000   0.010  0.000   0.032  0.000
// k 12    0.007  3.334   0.870  0.861   0.950  0.937
// k 16    1.000  0.552   0.990  0.710   0.949  0.569
// k 18    1.000  0.064   0.990  0.238   0.833  0.095
// k 20    1.000  0.017   0.990  0.117   0.461  0.012
// k 25    1.000  0.000   0.990  0.026   0.035  0.018
// k 30    1.000  0.000   0.286  0.499   0.030  0.004
```

The AMM does what it is built for and no more. Through the cruise its turn mode's probability
falls toward zero (it is kept as a logarithm, so it never underflows, but it needs several fixes
of evidence to come back), and one fix into the maneuver it is still at 0.007, 3.3 m off. Once the
maneuver has decided it, it stays in the turn mode for good: with no mixing, the cruise mode's
own filter has drifted with the maneuver and explains nothing better afterwards. The turn mode's
filter tracks the cruise well enough, so the error stays small, but the probabilities no longer
say what the target is doing. The floor brings the turn mode in at once, at the price of a
cruise mode that takes until step 30 to win back, since its filter must reconverge on its own.
The IMM, whose mixing restarts every mode from the current estimate, follows both changes. When
the mode can change, the IMM is the filter; the AMM is for a mode that cannot.

### The floor

The AMM's known failure is a mode probability that falls to zero, or so near it that no run of
measurements brings it back. Under Bayes' rule a probability of exactly zero never returns, and
with probabilities kept as logarithms, zero comes from a gate: a mode whose update the gate
rejects gets probability zero (§6), which the IMM's chain restores at the next prediction and the
AMM's never does. `upkeep::floorModes(probability)`, applied to the mutable `state()` after each update,
raises every mode below the floor to it, after Athans et al. (1977) §V-F; `FloorModes`
in `Upkeep.h` covers the action itself, how it shares the remaining mass, and its return value. Condensed from
`FloorKeepsARuledOutModeRecoverable`, with a confident cruise mode and a vague turn mode at the
origin, one outlier 30 m away, then ten fixes at the origin, all under `reductions::gate(0.997)`:

```cpp
auto const floor = upkeep::floorModes(0.001);  // not the 0.01 of the example above
if (!plain.update(fix({30, 0}), reductions::gate(0.997))) { ... }    // P(Cruise) 0: gated out
if (!floored.update(fix({30, 0}), reductions::gate(0.997))) { ... }
auto const raised = floor(floored.state());               // true; P(Cruise) 0.001
// ten fixes at the origin:
//        plain P(Cruise) |mean|    floored P(Cruise) |mean|
// k  1        0          29.925        0.9990       0.030
// k  5        0          10.136        0.9990       0.010
// k 10        0           0.116        0.9990       0.000
```

The plain AMM never returns to the mode that explains the data far better; only the turn mode's
own filter slowly walks back to the origin. With the floor, one fix restores the cruise mode, and
the floor then holds the turn mode at 0.001.

> **Q16.** Should the AMM floor its modes itself, as an option of the filter, since without a floor
> any gated update can end its usefulness for good? [autonomous-multiple-model] says no: the floor
> changes the distribution on purpose, so it is the caller's [upkeep], and it serves the IMM
> too. I agree, and would only make the class comment's example, which already applies the floor,
> the first thing a reader sees. Recorded here so the choice is visible; I'd mark it decided
> [autonomous-multiple-model] unless the review disagrees.

## 12. What builds on the IMM

The recursion of [filters-and-recursions] lets other pieces reuse the IMM's steps; each is covered
where it lives.

- **Smoothing.** `predictJoint(dt, ...)` records each prediction as a `HybridPredictedJoint`:
  the mode probabilities before the step, each mode's Gaussian moved through its own model (not
  the mixed one), the chain's matrix, and each mode's Gaussian moved through every
  other mode's model (`throughOtherModels`, read by `predictedThrough(from, model)`), $M(M - 1)$
  more predictions per step, from which the smoother weighs the modes by the expectation
  correction of [Barber (2006)](../reference/bibliography.md#barber2006). `HybridGaussianHistory`
  keeps them for the IMM smoother [smoothing]; [smoothing.md](smoothing.md).
- **Mixtures of IMMs.** A `MixtureParams<HybridGaussian<State, Mode>>` is a mixture of IMM states, and
  `MixtureFilter{switchingModel, mixture}` runs the IMM recursion on each component, noise by mode
  axis included; `MixtureFilter.h`.
- **Hidden Markov models.** A discrete state with no Gaussian part is `HiddenMarkovFilter` over a
  `MarkovChain` or a `ContinuousTimeMarkovChain`; [hidden-markov-filter.md](hidden-markov-filter.md).

Merging the hybrid components of a mixture of IMMs needs a cost over hybrids; [upkeep]
decided that the choice of pair, and so the cost, is the caller's, for every
component. `MixtureParams::merged()` and `merge()` work on hybrids, keeping each mode's moments.

> **Q17.** *Settled by roadmap 17z: the class template `Emrys::Sensor` went with `Models/Sensor.h`,
> and nothing in the namespace is named `Sensor`, so the header comments' and the tests' global tag
> `Sensor` compiles under `using namespace Emrys` as written. This document keeps `SensorMode`, which
> is also right.* ~~The examples' sensor axis was a global tag `Sensor` until this refresh, as in the
> tests (which qualify every library name, so they compile) and in the class comments of `ImmFilter`
> and `MixtureFilter` (`update<Sensor>(robustFix)`, `MarkovChain<Sensor>::persistent(0.8)`). Since
> 2026-10-31 the library has a class template `Emrys::Sensor`, so a user who copies those comments
> under `using namespace Emrys` gets `reference to 'Sensor' is ambiguous` at the first use. This
> document and hidden-markov-filter.md now write `SensorMode`. Rename the axis in the header comments
> too, and in the tests for consistency? The class template's name is right ([sensors]), so the tag
> is what gives way.~~

> **Q18.** `KalmanFilter` and `MixtureFilter` take a recursion as their last template parameter and
> a constructor `Filter{recursion, state}` [filters-and-recursions], [user-extension-points];
> `ImmFilter` has neither, so a user's step over a hybrid (a fading-memory IMM, say) has to be run
> on each component of a one-component `MixtureFilter` of hybrids. Give `ImmFilter` the same
> parameter and constructor, so the three filters are spelled alike? It is a breaking change only
> in the parameter count, and nothing is launched.

## Questions

1. The chain steps once per `predict()`, even for `dt` = 0; a continuous-time rate chain? (§2;
   (a) done, the rate chain exists since [emissions], `SwitchingModel` taking it still open)
2. A product `CategoricalParams` from per-dimension marginals, for the initial mode prior (§3).
3. ~~`Hybrid::marginal<Path>()` keeping the modes (§3).~~ Decided: a range over the modes, with
   `copy()` to a `Hybrid` [element-marginals].
4. ~~The class comment still names the grid of Kalman filters as a user (§3).~~ Done: it names
   `GridGaussianParams` for the grid.
5. ~~`switchingModel<Path>(chain, model)` with one model is silently shared (bug?) (§4).~~
   Decided: one model is shared only without a path; with one, the count is checked.
6. The recommended model for a block a mode doesn't use; amend [switching-models]'s example
   (§4).
7. ~~One prediction method for every mode; leave it? (§4; see Q14)~~ *Decided [switching-models]:
   a method per model, the same for every mode by default.* *Done: `withMethod()`.*
8. ~~Follow the mixture filter for partly gated modes (§6).~~ Decided: a mode is
   weighted by its passing branches, now by the log-weight the policy adds [reduction-policies].
9. ~~Test that a memoryless chain reproduces the outlier-robust Kalman filter (§7).~~ *Done with
   Q10.*
10. ~~Let the noise name its mode dimension, so the path can't be forgotten (§7).~~ *Decided [imm]:
    `additiveNoise(mixture).byMode<Axis>()`, read by what the filter's state has.*
11. ~~`update<Path>()` with Gaussian noise: a `static_assert` that names the cause (§7; it now fails
    to compile, with `no member named 'size'`).~~ *Decided in the code (roadmap 17z): a
    `static_assert` names it.*
12. [switching-models] describes a shared transform for modes differing only in Q that isn't built
    (§8).
13. ~~[component-first] says the square-root and information forms are components; [gaussian-forms]
    and the code say not (§3a).~~ *Decided: the square-root form is one.*
14. ~~Fixed-step models with different methods: allow them, or check in `SwitchingModel` with a
    message that names the cause (§4).~~ *Decided with Q7: allow them.* *Done.*
15. Forward `emptyUpdate()` from the filter, so a reused result is one line (§6).
16. The AMM's floor stays the caller's upkeep, not a filter option (§11;
    [autonomous-multiple-model] says so).
17. ~~The tag `Sensor` collides with `Emrys::Sensor` under `using namespace Emrys`; this document now
    writes `SensorMode`. Rename it in the header comments and tests too (§12)?~~ *Settled by roadmap
    17z: `Emrys::Sensor` is gone, so nothing collides.*
18. Give `ImmFilter` a recursion parameter and `ImmFilter{recursion, state}`, as `KalmanFilter` and
    `MixtureFilter` have (§12)?
19. A `SwitchingMeasurement` updates an `ImmFilter`, an `AmmFilter`, and an `ImmRecursion`, but
    not a `MixtureFilter` of hybrids, nor with a block given for a point-mass filter's hybrid cells,
    as `update()` is; add both (§7)?
20. A switching measurement whose measurements have noise by mode axis fails to compile. Support
    it, each value's measurement then reading its noise component by the other axis (§7)?
