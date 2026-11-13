# Hidden Markov filter

Review notes on the hidden Markov model filter: the row of the Filters module page's table with
nothing left Gaussian. The state is a `CategoricalParams`, a `MarkovChain` or a
`ContinuousTimeMarkovChain` moves it, and an emission model updates it. The document covers the
forward recursion, the emission models for discrete and continuous observations, the continuous-time
chain and its transition over an interval, the Markov-modulated Poisson process that the prediction
observes, and where the smoother and the most probable sequence come in. Questions are marked
**Q*n*** and collected at the end.

Headers reviewed: `Filters/HiddenMarkovFilter.h`, `Models/Emission.h`, and
`Distributions/ContinuousTimeMarkovChain.h`. Tests: `tests/Filters/hidden_markov_filter.cpp`,
`tests/Models/emission.cpp`, `tests/Distributions/continuous_time_markov_chain.cpp`, and the
cases that touch these headers in `tests/Smoothers/smoothing.cpp` (four), `tests/Filters/time_steps.cpp`,
and `tests/allocation.cpp`. Roadmap step 15.

Other documents own the pieces this one uses. `CategoricalParams` and `Categorical` are
`Distributions/Categorical.h`'s [categorical-params], the discrete spaces (`Named`, `Finite`, `DiscreteProduct`) are
`Spaces/Discrete.h`'s, and `MarkovChain`, with its persistent and Kronecker-product constructions,
is imm-filter.md §2. The smoother over a `CategoricalHistory` belongs to smoothing.md, and how a
`TimeStep` converts to a scalar [time-steps] to `Core/Time.h`. This document says what the filter
does with them.

Every example was compiled and run with Clang (MSVC STL) and GCC 14 (w64devkit) against
`next` of 2026-11-04, and the outputs in comments are the printed results; the two
compilers print the same digits throughout. The scratch programs are outside the repository. Where
an example has a hand computation, the text gives it. This refresh found every number unchanged
since the refresh of 2026-10-27, and these changes to the API: a fixed-size space is named rather than passed, so
`CategoricalParams<S>::fromWeights(weights)` and `GaussianParams<S>{mean, covariance}` leave it out
[space-storage]; `ContinuousTimeMarkovChain` gained `fromDwellTimes()` and the free `stationary()`
(§4); `EmissionTable::observe()` of a temporary is deleted (§3, Q4); `PoissonEmission`'s count is an
integer (§3); a starting state need not be normalized (§1); and GCC no longer warns in the
arrival step or the smoother (Q9). The document was refreshed for roadmap 17z
(`next`, 2026-11-11): nothing here uses a process or measurement model, so no example changed but
§4's, whose sensor axis is the tag `Sensor` again, since 17z removed the sensor class template
`Emrys::Sensor` that it collided with; Clang compiles it under `using namespace Emrys`. The
examples were not re-run for this refresh.

## Decisions so far

These are recorded in decisions.md and taken as given here.

- **[emissions]: a discrete measurement is an emission model, and a modulated arrival is observed by
  the prediction.** An `EmissionModel` writes $\log p(z \mid v)$ for every value $v$ into a buffer
  the caller sizes by the cardinality. `update(prior, emission)` returns the posterior and the
  log-evidence, or a `WeightError`. A Markov-modulated Poisson process is not an emission, because
  its likelihood depends on the chain's path inside the interval, so it is observed by
  `predict(dt, ArrivalInterval{rates})`.
- **The hidden Markov model is the hybrid core with nothing Gaussian.** The whole state is
  the discrete variable, Markov with $\Pi$ or a generator $Q$, handled by enumeration.
- **[discrete-spaces]: discrete values live in their own family of spaces**, and `CategoricalParams`
  keeps log-weights, which the filter keeps normalized, so a long run of unlikely measurements does
  not underflow.
- **[filters-and-recursions]: the filter is spelled as `KalmanFilter` is**, a class owning its chain
  and its state, deduced from its constructor arguments.
- **The chain's kind decides the prediction**, as [fixed-step-models] does for a fixed-step model:
  `predict()` with a `MarkovChain`, `predict(dt)` with a `ContinuousTimeMarkovChain`, never both.
- **[time-steps]: a time step is any type with a conversion to the filter's scalar**, so
  `predict(250ms)` and `predict(0.25)` are the same step, bit for bit
  (`tests/Filters/time_steps.cpp`).
- **[smoothing]: smoothing is a backward pass over a record the caller keeps.** `predictJoint()`
  returns a `CategoricalPredictedJoint`, and a `CategoricalHistory` keeps them.
- **[space-storage]: a fixed-size space is named, not passed.** A discrete space such as
  `Named<Rain, Dry>` is an empty type, so `CategoricalParams<Load>::fromWeights(weights)` takes the
  weights alone, beside `fromWeights(space, weights)`, which a resizable `Finite` still needs.
- **[allocation]: nothing sized by a bounded space allocates.** The filter's steps under both
  chains, the arrival step included, and a fixed-lag smoother allocate nothing once warm
  (`tests/allocation.cpp`, `HiddenMarkovFilterAndSmootherStepWithoutAllocating`).

## 1. Differences from the Kalman filter

The Filters module page and `KalmanFilter.h` describe the shape every filter shares. The hidden Markov filter differs in
each row:

| | `KalmanFilter` | `HiddenMarkovFilter` |
|---|---|---|
| Params | `<Component, Model, Method>` | `<Space, Chain = MarkovChain<Space>, Scalar = double>` |
| State | a Gaussian in one of its forms | `CategoricalParams<Space, Scalar>` |
| Moved by | a process model and a method | a `MarkovChain` (`predict()`) or a `ContinuousTimeMarkovChain` (`predict(dt)`) |
| Measured by | a measurement [sensors] | an emission model [emissions] |
| `update()` returns | a diagnostic `FilterUpdate` in a `std::expected<..., GaussianError>`, `[[nodiscard]]` [innovation-cholesky] | a `HiddenMarkovFilterUpdate`, with `accepted` and `logLikelihood()`, in a `std::expected<..., WeightError>`, `[[nodiscard]]` [update-outcomes] |
| Running total | none | `logEvidence()`, $\log p(z_{1:k})$, with `resetLogEvidence()` |
| Recursion object [filters-and-recursions] | `KalmanRecursion` | none; the free `predict()` and `update()` |

The filter is a thin holder of its state. Each step calls a free function that a caller can use
without it: `predict(categorical, chain)` (Categorical.h), `predict(categorical, chain, dt)`
(ContinuousTimeMarkovChain.h), and `update(categorical, emission)` and
`predict(categorical, chain, dt, interval)` (Emission.h). What the filter adds is the log-evidence
total and the choice of steps by the chain's type. `chain()` has a non-const overload, so a caller
can change the chain between steps. `state()` has one too, to change the state where it is, and
`setState()` replaces it, asserting that it is on the chain's space [state-access]; neither touches
the log-evidence. A state given to the constructor or to `setState()` need not be normalized: its
weights are written unchecked [categorical-params], and the next update normalizes them or fails
with a `WeightError` if they define no distribution, leaving the state and the log-evidence alone
(`ChecksTheStateWeightsAtTheUpdate`: weights of 5 and 5 give §2's first day; weights of 0 and 0
pass a prediction unchecked and fail the update with `ZeroSum`).

`update()` is `[[nodiscard]]`, as every filter's is, and returns what every filter's returns
[update-outcomes]: the result's `logLikelihood()` is the step's log-evidence, and `accepted` is
false when the measurement was impossible under the state, every value with a nonzero probability
giving it zero likelihood. The state is then left as it was, the result's log-likelihood is
$-\infty$, and so is `logEvidence()` until `resetLogEvidence()`, since the measurements so far have
probability zero under the model (`ImpossibleMeasurementIsRejected`). The update conditions the
state where it is, through `CategoricalParams::update()`, which writes nothing when the weights
after the measurement define no distribution, so the filter copies nothing to keep its prior.

The constructors take a chain and a starting `CategoricalParams`, or a chain alone, which starts from
the uniform distribution. Both deduce the space, the chain, and the scalar from the chain
(`DeducesFromTheChain`). Spelled out, the scalar has to be written twice:

```cpp
auto wrong = HiddenMarkovFilter<Weather, MarkovChain<Weather, float>>{MarkovChain<Weather, float>{}};
// clang: error: static assertion failed due to requirement 'std::is_same_v<float, double>':
//        the chain must be on the filter's space, in its scalar
```

> **Q1.** The `Space` and `Scalar` parameters repeat what the chain already carries, and the
> defaulted `Scalar = double` contradicts a `float` chain unless it is written a second time
> (`HiddenMarkovFilter<Weather, MarkovChain<Weather, float>, float>`). Deduction hides this, but a
> member or an alias has to spell it. Options: default `Scalar_` to `typename Chain_::Scalar`, which
> keeps the space-first order of the conventions; or take the chain first, as [component-first]
> takes a component first for types that hold a distribution per element, and drop the other two.

> **Q2.** The filter has no recursion object, so it does not follow [filters-and-recursions]'s
> split. Nothing composes with it today (a mixture of categoricals is a categorical), so the free
> functions serve. Should [filters-and-recursions] say that the hidden Markov filter is exempt, or
> should a `HiddenMarkovRecursion` exist so that every filter reads alike?

## 2. The forward recursion, by hand

The filter is the forward recursion of a hidden Markov model, kept normalized at every step.
[Rabiner (1989)](../reference/bibliography.md#rabiner1989), §III-A, gives the forward variables
$\alpha_k(j) = p(z_{1:k}, x_k = j)$ and the recursion
$\alpha_{k+1}(j) = \left[\sum_i \alpha_k(i) \Pi_{ij}\right] b_j(z_{k+1})$; its §V-A scales each
$\alpha_k$ to sum to one and recovers $\log p(z_{1:k})$ as the sum of the logs of the scale factors.
The code takes exactly that: the state is the scaled $\alpha_k$, and each `update()` returns the log
of its scale factor, $\log p(z_k \mid z_{1:k-1})$, and adds it to `logEvidence()`. It departs in two
ways. The weights are stored as logs [discrete-spaces], and predict and update are separate calls,
so the caller can predict without measuring or measure twice between predictions.

Three days of the umbrella example. The weather persists with probability 0.7, an umbrella is
seen on 90% of rainy days and 20% of dry ones, and the first day starts at even odds. Rabiner's
unscaled forward variables, by hand:

| Day | Seen | $\alpha_k(\text{rain})$ | $\alpha_k(\text{dry})$ | $\sum \alpha_k = p(z_{1:k})$ |
|---|---|---|---|---|
| 1 | umbrella | $0.5 \cdot 0.9 = 0.45$ | $0.5 \cdot 0.2 = 0.10$ | 0.55 |
| 2 | umbrella | $(0.7 \cdot 0.45 + 0.3 \cdot 0.10) \cdot 0.9 = 0.3105$ | $(0.3 \cdot 0.45 + 0.7 \cdot 0.10) \cdot 0.2 = 0.041$ | 0.3515 |
| 3 | none | $(0.7 \cdot 0.3105 + 0.3 \cdot 0.041) \cdot 0.1 = 0.022965$ | $(0.3 \cdot 0.3105 + 0.7 \cdot 0.041) \cdot 0.8 = 0.09748$ | 0.120445 |

So the filtered probabilities of rain are $0.45/0.55 = 9/11$, $0.3105/0.3515 = 0.883357$, and
$0.022965/0.120445 = 0.190668$; each day's evidence is the ratio of consecutive sums
($0.55$, $0.3515/0.55 = 0.639091$, $0.120445/0.3515 = 0.342660$); and
$\log p(z_{1:3}) = \log 0.120445 = -2.11656$. The filter:

```cpp
struct Rain {};
struct Dry {};
struct Weather : Named<Rain, Dry> {};
struct Umbrella {};
struct NoUmbrella {};
struct Sighting : Named<Umbrella, NoUmbrella> {};

auto const table = EmissionTable{Weather{}, Sighting{}, Eigen::Matrix2d{{0.9, 0.1}, {0.2, 0.8}}};
auto filter = HiddenMarkovFilter{MarkovChain<Weather>::persistent(0.7)};  // uniform start
auto const umbrella = table.observe(Sighting::index<Umbrella>());
auto const none = table.observe(Sighting::index<NoUmbrella>());

auto const day1 = filter.update(umbrella);  // exp: 0.55,     P(rain) 0.818182
filter.predict();                           //               P(rain) 0.627273
auto const day2 = filter.update(umbrella);  // exp: 0.639091, P(rain) 0.883357
filter.predict();
auto const day3 = filter.update(none);      // exp: 0.34266,  P(rain) 0.190668
filter.logEvidence();                       // -2.11656
std::cout << filter;
// hidden Markov filter
//   categorical over 2 values
//     probabilities: 0.190668 0.809332
```

The day-two prediction is $(0.7 \cdot 9 + 0.3 \cdot 2)/11 = 6.9/11 = 0.627273$. The test
`MatchesAHandComputedForwardPass` checks the first two days to `1e-15` against the same fractions,
`RunTimeCardinality` checks four steps on a `Finite<Eigen::Dynamic, 6>` against the recursion
written in plain Eigen, and `IdentityEmissionRecoversTheSequence` checks that an identity table
makes the state certain of the simulated value at every step, with each step's evidence the
predicted probability of that value. A filter prints its name, with its state nested under it
[printing-and-simulation].

## 3. Emission models

### The concept and the update

```cpp
template <DiscreteSpace Space, typename Scalar>
using EmissionBuffer = Eigen::Ref<Eigen::Array<Scalar, Space::Cardinality, 1>>;

template <typename E, typename Space>
concept EmissionModel = DiscreteSpace<Space> &&
    requires(E const & emission,
             Space const & space,
             EmissionBuffer<Space, typename E::Scalar> out) {
    requires RealScalar<typename E::Scalar>;
    emission.logLikelihoods(space, out);
};
```

An emission model has a `Scalar` and fills one log-likelihood per value of the space into the
caller's buffer. It captures its observation, as a continuous measurement does [sensors], so the
model object *is* the measurement. It is the discrete form of a measurement's likelihood: no
innovation, no noise, no method. The buffer is the caller's, sized by the cardinality and inline up
to `MaxCardinality`, so a bounded space allocates nothing. It is an `EmissionBuffer`, whose rows at
compile time are the space's `Cardinality`, so a model over a fixed space writes a fixed-size array
and Eigen unrolls what it does there. A model may take a run-time sized `Eigen::Ref<Eigen::ArrayXd>`
instead, which the fixed buffer binds to (`UserModelTakesEitherBuffer`), at the cost of the size.

`update(prior, emission)` is Bayes' rule on a `CategoricalParams`. It returns a
`CategoricalUpdate{posterior, logEvidence}`, or `WeightError::ZeroSum` when the measurement is
impossible (`ImpossibleMeasurementIsAnError`), and leaves the prior alone, as the Gaussian
`update()` does. The filter's `update()` is this function done in place, plus the running total,
and reads that `ZeroSum` as a rejection [update-outcomes]:

```cpp
auto const step = update(CategoricalParams<Weather>{}, umbrella);
// step->posterior.toDensity()->probability<Rain>() 0.818182, step->logEvidence -0.597837 (log 0.55)
```

The emission's scalar need not be the state's. `update()` fills a buffer in the emission's scalar,
and `CategoricalParams::update()` casts it, so a `double` table updates a `float` filter without a word.

> **Q3.** Should an emission in another scalar than the state's be rejected, as the filter's
> `static_assert` rejects a chain in another scalar? The cast is harmless from `double` to `float`
> for a table, but it is a silent conversion where every other pairing in the library is checked.

### EmissionTable: a discrete observation

`EmissionTable<Space, ObservationSpace, Scalar>` holds the emission matrix $B$ of
[Rabiner (1989)](../reference/bibliography.md#rabiner1989), §II: a right stochastic matrix with a
row per value of the state and a column per observed value, $B_{vz} = P(z \mid v)$, fixed-size and
row-major under the same rule as `MarkovChain`'s. The constructor asserts that it is nonnegative
with rows summing to one (to `dummy_precision`). The table itself is not an emission model
(`static_assert(!EmissionModel<Table, Weather>)` in the tests): `observe(z)` names the observed
value by flat index and returns an `Observation` that borrows the table, so a filter updated
every step does not copy the matrix. Its `logLikelihoods()` is the log of column $z$.

The `Observation` holds a pointer to the table, so it must not outlive it. A table that is a
temporary can't be observed, since `observe()` is `const &` and its `&&` overload is deleted (Q4);
a named table that goes out of scope first is still the caller's to avoid:

```cpp
auto const umbrella =
    EmissionTable{Weather{}, Sighting{}, Eigen::Matrix2d{{0.9, 0.1}, {0.2, 0.8}}}.observe(0);
// clang: error: call to deleted member function 'observe'
//   Emission.h:226:5: note: candidate function has been explicitly deleted
```

> **Q4.** *Decided: yes.* `observe()` is `const &` and its `&&` overload is deleted.
> ~~`observe()` is a view maker by the conventions' definition (it borrows its table), but it
> is a plain `const` member, so calling it on a temporary compiles and dangles. style.md and
> [view-stamp] say a view maker is `const &` with its `&&` overload deleted. Should `observe()`
> follow that? The table has no mutators, so the stamp check of [view-stamp] is not needed; only the
> deleted overload is.~~

### DensityEmission: a continuous observation

`DensityEmission{observation, distributionOf}` evaluates the observation's `logDensity()` under
the distribution `distributionOf(v)` returns for each value, so a hidden Markov model with
continuous observations needs no new type: a `Gaussian` per value, or any distribution
with `logDensity()`. The callable may return a reference into the caller's list or build a
distribution on the spot. The scalar is deduced from `logDensity()`'s result.

```cpp
struct Quiet {};
struct Busy {};
struct Load : Named<Quiet, Busy> {};

auto const quiet = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}};
auto const busy = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{2.0}, Eigen::Matrix<double, 1, 1>{1.0}};
auto const densities = std::array{quiet.toDensity().value(), busy.toDensity().value()};
auto const reading = DensityEmission{Eigen::Matrix<double, 1, 1>{0.5}, [&](Eigen::Index v) -> auto const & {
    return densities[std::size_t(v)];
}};
auto const byDensity = update(CategoricalParams<Load>{}, reading);
// P(busy) 0.268941, log-evidence -1.42382
```

By hand: the ratio of the two densities at 0.5 is $e^{-(1.5^2 - 0.5^2)/2} = e^{-1}$, so
$P(\text{busy}) = 1/(1 + e) = 0.268941$, as `DensityEmissionUsesEachValuesDensity` checks to
`1e-14`. The densities are views of their parameters, so the parameters outlive the emission here,
and a change to the parameters would trip [view-stamp]'s stamp check on the next read.

### PoissonEmission and ExponentialEmission: arrivals with a rate per value

Two emission models take a rate $\lambda_v$ per value of a Poisson process:

- `PoissonEmission<Space>{rates, count, dt}`: a count $n$ over an interval $\Delta t$,
  [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) with mean $\lambda_v \Delta t$:
  $\log p(n \mid v) = n \log(\lambda_v \Delta t) - \lambda_v \Delta t - \log n!$. A zero rate
  gives a positive count zero likelihood and no events certainty.
- `ExponentialEmission<Space>{rates, tau}`: an inter-arrival time $\tau$, with
  [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) density
  $\log p(\tau \mid v) = \log \lambda_v - \lambda_v \tau$.

Both take their interval as any `TimeStep` [time-steps]. `PoissonEmission`'s count is any integer
but `bool`, so a count and an interval given in each other's place fail to compile, where a step
converted to a count would have been truncated:

```cpp
auto const rates = Eigen::Array2d{0.5, 6.0};
auto const byCount = update(CategoricalParams<Load>{}, PoissonEmission<Load>{rates, 3, 0.5});
// P(busy) 0.991029, log-evidence -2.18006
auto const byGap = update(CategoricalParams<Load>{}, ExponentialEmission<Load>{rates, std::chrono::milliseconds{400}});
// P(busy) 0.570749, log-evidence -0.740582
```

By hand: three events in half a second have probability $e^{-0.25} 0.25^3 / 3! = 0.0020281$ when
quiet and $e^{-3} 3^3 / 3! = 0.224042$ when busy, so $P(\text{busy}) = 0.224042 / 0.226070 = 0.991029$
and the evidence is $0.113035 = e^{-2.18006}$. A gap of 0.4 s has density $0.5 e^{-0.2} = 0.409365$
when quiet and $6 e^{-2.4} = 0.544308$ when busy, so $P(\text{busy}) = 0.570749$. The count's
arguments swapped:

```cpp
auto const swapped = PoissonEmission<Load>{rates, 0.5, 3};
// clang: error: no matching constructor for initialization of 'PoissonEmission<Load>'
//   note: candidate template ignored: constraints not satisfied [with ..., Count = double, dt:auto = int]
//   note: because 'double' does not satisfy 'integral'
```

Both hold the value fixed over the interval. Over a `ContinuousTimeMarkovChain`, which may jump
inside it, they are the approximation for intervals short beside the chain's holding times; §5
gives the exact observation and measures the approximation's error.

> **Q5.** `EmissionTable` borrows its matrix through `observe()`, but `PoissonEmission`,
> `ExponentialEmission`, and `ArrivalInterval` copy their rates into every measurement. For a
> bounded space the copy is inline and cheap, but on an unbounded `Finite<Eigen::Dynamic>` it
> allocates once per measurement, against [allocation]'s "scratch sized by a count is owned and
> reused". Should the rates be an object with `count(n, dt)`, `gap(tau)`, and `arrival()` makers
> that borrow them, as the table's `observe()` does?

## 4. The continuous-time chain

`ContinuousTimeMarkovChain<Space, Scalar>` is a time-homogeneous
[continuous-time Markov chain](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain) on a
finite set, held as its generator $Q$: a row per value moved from and a column per value moved
to, $Q_{ij} \ge 0$ for $i \ne j$ the rate of jumping from $i$ to $j$, and each row summing to zero,
so $-Q_{ii}$ is the rate of leaving $i$. Its type is `MarkovChain`'s matrix type, fixed-size and
row-major for a fixed space. It is built in four ways, each with a counterpart on `MarkovChain`
(imm-filter.md §2):

| `ContinuousTimeMarkovChain` | asserts or builds | `MarkovChain` counterpart |
|---|---|---|
| `ContinuousTimeMarkovChain<S>{}` | every rate zero: never moves | `MarkovChain<S>{}`, the identity |
| `{space, generator}` | nonnegative off the diagonal, rows summing to zero (to `dummy_precision`, times the largest leaving rate if above one) | `{space, matrix}`, rows summing to one |
| `::persistent(rate, space = {})` | leaves at `rate`, to each other value at `rate / (n - 1)` | `::persistent(stay)`: stays with probability `stay` |
| `{product, axisChains...}` on a `DiscreteProduct` | the [Kronecker sum](https://en.wikipedia.org/wiki/Kronecker_sum) of the axes' generators | the Kronecker product of the axes' matrices |
| `::fromDwellTimes(tau, [exitWeights], space = {})` | $Q_{ii} = -1/\tau_i$, $Q_{ij} = w_{ij} / (\tau_i \sum_k w_{ik})$; evenly without weights; an infinite $\tau_i$ never leaves | `::fromDwellTimes()`, $\Pi_{ii} = 1 - 1/\tau_i$ in steps |

The dwell times are the chain's mean holding times, in the unit of the steps later given to
`transition()` (seconds for a `std::chrono` duration), and the exit weights, normalized, its jump
probabilities, so `fromDwellTimes(Eigen::Vector2d{1 / a, 1 / b})` is the two-state chain below
exactly; `FromDwellTimes` checks three generators by hand. The free `stationary(chain)` returns the
distribution $\pi$ with $\pi Q = 0$ as a `CategoricalParams`, the prior for a chain that has run
long before the first measurement, or `ChainError::Reducible` when some value can't reach another,
as for a chain that never moves. It is the algorithm of
[Grassmann, Taksar, and Heyman (1985)](../reference/bibliography.md#grassmann1985), which reads only
the off-diagonal rates and never subtracts, shared with `MarkovChain`'s `stationary()`
(imm-filter.md §2); `Stationary` checks it against the closed form, and against the stationary
distribution of the transition over 0.7, to `1e-14`.

The product is a sum, not a product, because two independent axes never jump at the same
instant: the product chain's rate of moving from one mode to another is the rate of one axis
jumping while the others stay. Over any interval its transition is the Kronecker product of the
axes' transitions, which is `MarkovChain`'s product constructor (`ProductIsTheKroneckerSum`, to
`1e-12`).

> **Q6.** `MarkovChain::persistent(0.7)` takes the probability of *staying*, and
> `ContinuousTimeMarkovChain::persistent(2.0)` the rate of *leaving*. Each is the natural
> parameter of its kind, and the scales differ enough (a probability in $[0, 1]$ against a rate)
> that a mix-up is likely to trip the assertion on the discrete side, but not on the continuous
> side, where any nonnegative number is a rate. Is the shared name worth a different spelling,
> such as `persistent(leaveRate)` or `leaving(rate)`?
>
> **Scheduled:** before the launch, since a rename breaks code (roadmap, "Before going public",
> the guide questions to answer before the launch). `fromDwellTimes()`, a mean time per value on
> both chains, is a third spelling with neither reading, so one answer keeps `persistent()` as it
> is and points to it.

### The transition over an interval

`transition(dt)` returns the `MarkovChain` with $\Pi = e^{Q \Delta t}$, from Eigen's
`unsupported/Eigen/MatrixFunctions`, then clears the negative rounding off the zeros and rescales
each row to sum to one. `predict(categorical, chain, dt)` is
`predict(categorical, chain.transition(dt))`, exact over any interval, as a process model with
`continuousNoise()` discretizes its own noise exactly [continuous-time-noise]. The time step is any `TimeStep`
[time-steps].

```cpp
struct Cruise {};
struct Turn {};
struct Maneuver : Named<Cruise, Turn> {};

double const a = 0.3;  // cruise -> turn
double const b = 1.2;  // turn -> cruise
auto const maneuver = ContinuousTimeMarkovChain<Maneuver>{Eigen::Matrix2d{{-a, a}, {b, -b}}};
MarkovChain const step = maneuver.transition(0.5);
// e^{Q/2} =
// 0.894473 0.105527
// 0.422107 0.577893
// closed form of staying in cruise, (b + a e^{-(a+b)/2}) / (a + b): 0.894473

auto const prior = CategoricalParams<Maneuver>::fromWeights(0.1, 0.9);
auto const shortStep = predict(prior, maneuver, 0.5);
auto const longStep = predict(prior, maneuver, 100.0);
shortStep.toDensity()->probabilities();  // 0.469343 0.530657
longStep.toDensity()->probabilities();   // 0.8 0.2, the stationary (b, a) / (a + b)
stationary(maneuver)->toDensity()->probabilities();  // 0.8 0.2

auto filter = HiddenMarkovFilter{maneuver, prior};
filter.predict(std::chrono::milliseconds{500});   // 0.469343 0.530657
```

By hand: $0.1 \cdot 0.894473 + 0.9 \cdot 0.422107 = 0.469343$. The two-state closed form is checked
over four intervals to `1e-13` (`TwoStateTransitionMatchesClosedForm`), the persistent chain's
probability of staying, $1/n + (n-1)/n \cdot e^{-n r t/(n-1)}$, to `1e-13`, and a zero interval
gives the identity (`RunTimeCardinality`).

The product of two axes, a maneuver and a sensor that switches at rate 2:

```cpp
struct Inlier {};
struct Outlier {};
struct Sensor : Named<Inlier, Outlier> {};
using Mode = DiscreteProduct<Maneuver, Sensor>;

auto const sensor = ContinuousTimeMarkovChain<Sensor>::persistent(2.0);
auto const mode = ContinuousTimeMarkovChain<Mode>{maneuver, sensor};
// mode.generator() =
// -2.3    2  0.3    0
//    2 -2.3    0  0.3
//  1.2    0 -3.2    2
//    0  1.2    2 -3.2
```

[Rabiner (1989)](../reference/bibliography.md#rabiner1989) treats only discrete time, and the
header cites no source for the matrix exponential, which is Eigen's.

### A fixed step is a MarkovChain

The filter's chain decides its prediction: a `ContinuousTimeMarkovChain` filter has `predict(dt)`
and no `predict()`, a `MarkovChain` filter the reverse (§7). A filter that always steps by the same
interval is the [fixed-step-models] case, and it needs no fixed-step wrapper: the transition over
that interval is a `MarkovChain`, computed once.

```cpp
auto fixed = HiddenMarkovFilter{maneuver.transition(0.5), prior};  // a MarkovChain filter
fixed.predict();
// fixed.state().toDensity()->probabilities() == filter.state().toDensity()->probabilities(), bit for bit: 1
```

What a fixed-step filter gives up is the arrival step of §5, which needs the generator.

> **Q7.** `SwitchingModel` takes only a `MarkovChain`, so an IMM whose modes switch in continuous
> time assigns `filter.model().chain() = generator.transition(dt)` before each irregular step
> (both accessors have non-const overloads). Roadmap step 15 says "`SwitchingModel`
> may later take this chain in place of $\Pi$". Is that for after 0.1, and should imm-filter.md
> show the workaround?
>
> **Scheduled:** the template's shape is decided before the launch, since adding a chain
> parameter later would break code; the continuous chain itself may follow (roadmap, "Before going
> public", the guide questions to answer before the launch).

## 5. Modulated arrivals

In a [Markov-modulated Poisson process](https://en.wikipedia.org/wiki/Markovian_arrival_process),
events arrive at the rate $\lambda_v$ of the current value of a continuous-time chain, which may
jump between events. Its likelihood is not an emission: an inter-arrival time says that *no* event
arrived while the chain moved, which depends on the path through the interval and not only on the
value at its end. [emissions] records why the roadmap's plan (Poisson emissions after an exact
prediction) does not give the exact filter.

With $\Lambda = \mathrm{diag}(\lambda)$, the interval of length $\tau$ that ends in an event moves
the unnormalized distribution as $p' = p\, e^{(Q - \Lambda)\tau} \Lambda$, and the closing interval
of a record, which ends without one, as $p' = p\, e^{(Q - \Lambda)\tau}$. The product of the steps
over a record is the likelihood $\pi \left[\prod_k e^{(Q - \Lambda)\tau_k} \Lambda\right]
e^{(Q - \Lambda)\tau_{\text{end}}} \mathbf 1$ of
[Fischer and Meier-Hellstern (1993)](../reference/bibliography.md#fischer1993). The code takes
that closed form and applies it one interval at a time, normalizing after each and summing the
logs of the totals, as §2 does for the forward variables. It departs in one place: the exponential
is taken of $Q - \Lambda + \lambda_{\min} I$, and $\lambda_{\min} \tau$ is subtracted from the
log-likelihood, so the slowest value's entries stay of order one however long the interval
(`LongArrivalIntervalDoesNotUnderflow`: an arrival after $\tau = 1000$ gives $\log 0.5 - 1000$, where
the unshifted survival $e^{-1000}$ would underflow).

`ArrivalInterval<Space>{rates, endsInArrival = true}` names the observation, and the filter's
`predict(dt, interval)`, or the free `predict(prior, chain, dt, interval)`, takes it in place of a
prediction and an emission. It returns the interval's log-likelihood, or a `WeightError` with the
state unchanged.

```cpp
auto const generator = Eigen::Matrix2d{{-0.4, 0.4}, {1.1, -1.1}};
auto const rates = Eigen::Array2d{0.5, 6.0};
auto const chain = ContinuousTimeMarkovChain<Load>{generator};
auto const initial = CategoricalParams<Load>::fromWeights(0.7, 0.3);

auto exact = HiddenMarkovFilter{chain, initial};
for (double const tau : {0.37, 1.12, 0.05})
{
    auto const logLikelihood = exact.predict(tau, ArrivalInterval<Load>{rates});
}
// arrival after 0.37: log-likelihood -0.53017326, P(busy) 0.53746652
// arrival after 1.12: log-likelihood -1.6457168,  P(busy) 0.43650952
// arrival after 0.05: log-likelihood 0.7758699,   P(busy) 0.87141386
static_cast<void>(exact.predict(0.63, ArrivalInterval<Load>{rates, false}));
exact.logEvidence();  // -3.0970058

// The closed form in plain Eigen, without the shift:
Eigen::Matrix2d const L = rates.matrix().asDiagonal();
auto forward = Eigen::RowVector2d{0.7, 0.3};
for (double const tau : {0.37, 1.12, 0.05})
{
    forward = forward * Eigen::Matrix2d{((generator - L) * tau).exp()} * L;
}
forward = forward * Eigen::Matrix2d{((generator - L) * 0.63).exp()};
std::log(forward.sum());  // -3.0970058
```

A log-likelihood can be positive, as the third is: an arrival time is a density, and a gap of
0.05 s is likely when events come at 6 per second. `ModulatedPoissonMatchesTheClosedFormLikelihood`
runs the same comparison over eight arrivals and a closing gap to `1e-11`, and
`ArrivalIntervalIsTheModulatedStep` checks one step of each kind to `1e-13`.

### The approximation by counts

Counting arrivals over short steps of $h$ with a `PoissonEmission`, after an exact `predict(h)`,
approximates the exact filter. A count is a probability where an arrival time is a density, so
its log-evidence exceeds the exact one by $\log h$ per arrival. What is left is the error of
splitting $e^{(Q - \Lambda)h}$ into $e^{Qh} e^{-\Lambda h}$, of first order in $h$:

```cpp
for (int const per : {100, 1000})
{
    double const h = 1.0 / per;
    auto counted = HiddenMarkovFilter{chain, initial};
    auto const countStep = [&](Eigen::Index n) {
        counted.predict(h);
        static_cast<void>(counted.update(PoissonEmission<Load>{rates, n, h}));
    };
    // countStep(0) for every step of each gap but its last, countStep(1) for the last,
    // then countStep(0) over the closing gap
}
// h = 0.01:  counted - 3 log h - exact = 0.00047137617
// h = 0.001: counted - 3 log h - exact = 5.8329334e-05
```

A tenfold refinement cuts the error eightfold here. `CountedArrivalsConvergeToTheExactFilter`
measures 5.8e-4 and 8.8e-5 over its eight-arrival record and asserts a ratio above 5. The
approximation costs a prediction and an update per step of $h$, a thousand of them per second of
record at $h = 0.001$, where the exact filter takes one step per arrival.

## 6. Smoothing and the most probable sequence

smoothing.md owns the smoother. What this document's headers supply is `predictJoint()`, in three
overloads matching the three predictions: `predictJoint()` on a `MarkovChain` filter,
`predictJoint(dt)` and `predictJoint(dt, interval)` on a continuous-time one. Each moves the state
as its `predict()` does and returns a `CategoricalPredictedJoint{filtered, predicted, transitions}`
with $\mathrm{predicted} = K^T \mathrm{filtered}$ exactly. $K$ is $\Pi$, or $e^{Q \Delta t}$, or for an
arrival interval the step $e^{(Q - \Lambda)\Delta t}[\Lambda]$ divided by its total, which is no
longer stochastic, since the step observed something. `predictJoint(dt, interval)` returns
`std::expected`, with the same error as `predict(dt, interval)`.

A `CategoricalHistory` records the joints, and `smooth(latest)`, `transitionPosterior()`, and
`mostProbableSequence()` (Viterbi, [Viterbi (1967)](../reference/bibliography.md#viterbi1967) in
the max-product form of Rabiner's §III-B) run over it. The smoother reads filtered and predicted
distributions rather than Rabiner's backward variables, so the history keeps no measurements.
Continuing §2's three days:

```cpp
auto filter = HiddenMarkovFilter{MarkovChain<Weather>::persistent(0.7)};
auto history = CategoricalHistory<Weather>{};
static_cast<void>(filter.update(table.observe(Sighting::index<Umbrella>())));
for (auto const z : {Sighting::index<Umbrella>(), Sighting::index<NoUmbrella>()})
{
    history.push(filter.predictJoint());
    static_cast<void>(filter.update(table.observe(z)));
}
history.smooth(filter.state());
// smoothed P(rain) 0.861929   (day 1)
// smoothed P(rain) 0.799161   (day 2)
// latest   P(rain) 0.190668   (day 3)
auto const path = mostProbableSequence(history, filter.state());
// Viterbi: 0 0 1, probability 0.564905
```

By hand, with Rabiner's backward variables: $\beta_3 = (1, 1)$,
$\beta_2 = \Pi\, (b(\text{none}) \circ \beta_3) = (0.31, 0.59)$, and
$\beta_1 = \Pi\, (b(\text{umbrella}) \circ \beta_2) = (0.2307, 0.1663)$. The smoothed probabilities
of rain are $\alpha_2 \beta_2 = 0.3105 \cdot 0.31 / 0.120445 = 0.799161$ and
$0.45 \cdot 0.2307 / 0.120445 = 0.861929$. Rain, rain, dry has joint probability
$0.5 \cdot 0.9 \cdot 0.7 \cdot 0.9 \cdot 0.3 \cdot 0.8 = 0.06804$, the largest of the eight
sequences, and $0.06804 / 0.120445 = 0.564905$. `tests/Smoothers/smoothing.cpp` checks the smoother,
the transition posteriors, and Viterbi against enumeration of every sequence, over a discrete
chain and over the arrival process.

## 7. What misuse looks like

Predicting with the wrong chain's step fails to compile, with the constraint named:

```cpp
auto discrete = HiddenMarkovFilter{MarkovChain<Weather>::persistent(0.7)};
discrete.predict(0.5);
// clang: error: no matching member function for call to 'predict'
//   note: candidate template ignored: constraints not satisfied [with dt:auto = double]
//   note: because 'internal::IsContinuousTimeChain<Emrys::MarkovChain<Weather>>' evaluated to false
// gcc:   note: the expression 'IsContinuousTimeChain<Chain_> [with Chain_ = Emrys::MarkovChain<Weather, double>]'
//          evaluated to 'false'

auto continuous = HiddenMarkovFilter{ContinuousTimeMarkovChain<Weather>::persistent(1.0)};
continuous.predict();
// clang: note: because '!internal::IsContinuousTimeChain<Emrys::ContinuousTimeMarkovChain<Weather>>' evaluated to false
```

Passing the table instead of an observation:

```cpp
static_cast<void>(discrete.update(table));
// clang: note: because 'EmissionModel<Emrys::EmissionTable<Weather, Sighting>, Weather>' evaluated to false
//        note: because 'emission.logLikelihoods(space, out)' would be invalid:
//              no member named 'logLikelihoods' in 'Emrys::EmissionTable<Weather, Sighting>'
// gcc:   note: the required expression 'emission.logLikelihoods(space, out)' is invalid
```

The Clang messages name `internal::IsContinuousTimeChain`, an internal name, where a reader wants
"this filter's chain is a `MarkovChain`; call `predict()`". The chain on another scalar is the
`static_assert` of §1. A chain or a state on a different instance of a resizable space, an
observation out of range, an emission matrix whose rows do not sum to one, and a generator whose
rows do not sum to zero are assertions at run time.

## 8. Smaller notes

- **Scalars.** The filter, the chains, and the emissions run in `float`: a `float` table and chain
  give §2's first two days (P(rain) 0.883357), and a `float` arrival step works. The matrix exponential is
  Eigen's in the chain's scalar.
- **Execution policies.** No step takes an `ExecutionPolicy` [execution-policies]. The work grows
  with the cardinality, a dimension rather than a count of elements, and is a matrix-vector product
  that Eigen already vectorizes.
- **Simulation.** `simulate(value, chain, rng)` [printing-and-simulation] draws a chain's next
  value, a draw from `chain.next(value)`, the distribution of the value after `value`, with no
  allocation [random-draws]; a continuous-time chain takes the interval after the generator,
  `simulate(value, chain, rng, dt)`. `IdentityEmissionRecoversTheSequence` steps its truth so. An
  emission table still has no draw of its observation.

> **Q8.** *Half settled: `simulate()` takes a chain.* Should the emission models have a draw of
> their observation, so that a hidden Markov example can be written as the continuous ones are?

> **Q9.** *Settled: no longer reproduces.* ~~GCC 14 at `-O1` warns `-Wmaybe-uninitialized` inside
> `Categorical::fromLogWeights()`, on the empty tag space it moves into the result, when inlined
> into `internal::moveByKernel()` (the arrival step) and `smoothStep()`. Every program in §5 and §6
> shows it; Clang is silent.~~ Since [categorical-params] the arrival step builds its posterior as a
> `CategoricalParams` from log-weights, without `fromLogWeights()`, and the program running §2 to
> §8 compiles at `-O1 -Wall` under GCC 14 with no warning on 2026-11-04. The roadmap's list of GCC 14
> `-Wmaybe-uninitialized` warnings still names the arrival step and `smoothStep`, and can drop
> them.

> **Q10.** The Doxygen comment of `predict(prior, chain, dt, interval)` in `Models/Emission.h`
> has a blank line inside the comment block and a line starting `* * one`, which renders as a
> list item, and its sentences break mid-phrase; the comments of `internal::ArrivalKernel` and
> `internal::arrivalKernel()` break the same way. A reformatting slip, presumably from
> clang-format; still there on 2026-11-04, and the roadmap lists it.

## Questions

1. **Q1.** The filter's `Space` and `Scalar` repeat the chain's; default the scalar from the
   chain, or take the chain first? (§1)
2. **Q2.** The filter has no recursion object [filters-and-recursions]; exempt it in
   [filters-and-recursions], or add one? (§1)
3. **Q3.** An emission in another scalar than the state's is cast silently; reject it? (§3)
4. ~~**Q4.** `EmissionTable::observe()` on a temporary compiles and dangles; delete its `&&`
   overload as the view convention says? (§3)~~ *Decided: deleted.*
5. **Q5.** The arrival emissions copy their rates into every measurement, allocating on an
   unbounded space; borrow them as the table does? (§3)
6. **Q6.** `persistent()` takes a probability of staying on `MarkovChain` and a rate of leaving
   on `ContinuousTimeMarkovChain`; rename one? (§4)
7. **Q7.** `SwitchingModel` takes no continuous-time chain; after 0.1, and document the
   workaround in imm-filter.md? (§4)
8. **Q8.** `simulate()` draws a chain's next value; the emission models have no draw of their observation; add one after 0.1? (§8)
9. ~~**Q9.** GCC 14 `-Wmaybe-uninitialized` in `Categorical::fromLogWeights()` through the arrival
   step and the smoother. (§8)~~ *Settled: no longer reproduces as of 2026-11-04.*
10. **Q10.** Doxygen comment formatting in `Models/Emission.h`'s arrival `predict()` and
    `arrivalKernel()`. (§8)
