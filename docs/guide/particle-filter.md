# Particle filter

Review notes on the particle filter: `Filters/ParticleFilter.h` and `Core/Resampling.h`, tested in
`tests/Filters/particle_filter.cpp`, `tests/Core/resampling.cpp`, the particle filter's case of
`tests/allocation_particles.cpp`, and its cases in `tests/parallel.cpp`. It covers propagation with
sampled process noise, weighting by the measurement likelihood, the effective sample size, the four
resamplers, and roughening, and then what later decisions added: execution policies
[execution-policies], time steps of any type and fixed-step models [fixed-step-models],
[time-steps], allocation [allocation], printing [printing-and-simulation], and where the filter
stands against the state/recursion split of [filters-and-recursions]. Each section says what a piece
does and shows it in use. Questions are marked **Q*n*** and collected at the end.

The examples were refreshed for roadmap 17z (`next`, 2026-11-11), compiled and run
with Clang 21 (MSVC STL); GCC was not re-run for this refresh, and the GCC remarks below are from
2026-11-04. Outputs shown in comments are Clang's. GCC's `std::normal_distribution` draws different
numbers from the same seed [random-draws], so its sampled digits differ by about the Monte Carlo
error; the resampler table in §7 draws only uniforms and is identical under both. Compiler errors
are quoted from Clang, trimmed to the lines that matter. What roadmap 17z changed here: a process
model is a function with `reads()` and `writes()`, so the built-in `ConstantVelocity<Position,
Velocity>{0.1}` is itself the model and `compose()` puts models side by side; a measurement model
comes from `measurementModel()`, and a filter that weights points takes it observed with no method,
`observe(model, z)`; a measurement known only by its likelihood is `measurementModel(reads(...),
logLikelihood(g))` [likelihood-measurements]; an uncertain input is a Gaussian over a tagged space
and a known one `sure<Tag>(value)`, both matched to the model's `reads()` by tag (§3, Q15 decided);
and noise that passes through the measurement function has a likelihood at a point only where the
model declares itself affine in it (§4). Separately, since 2026-11-06 the root of a positive
definite noise covariance is its Cholesky factor rather than the pivoted root (§3), which draws
other samples of the same distribution from a seed, so §1's digits after the first prediction are
new; the Kalman filter's numbers in §1 are unchanged. Every other sampled number in §1 to §4 is the
same as on 2026-11-04, §3's Constant example among them, whose singular $Q$ still takes the pivoted
root. Earlier refreshes found: `update()` is
`[[nodiscard]]`; a fixed-size space is named rather than passed [space-storage]; weights
that can't be normalized reject the update in a release build (§4, Q12 decided); a residual
resampler given a negative weight asserts under `Sequential` but not under a policy (§7, Q16); and
the multinomial and residual resamplers and `Roughen` allocate nothing through `resample()` and
under `Sequential` (Q8 decided).

The examples share these declarations, and a helper `fix(z)` that builds a position fix with
unit noise, as `makeFix` does in the tests (with its own noise there). The fix names a method
because §1 gives it to a Kalman filter as well; the particle filter ignores the method, and takes
`observe(model, z)` without one. `diag(...)` in the snippets is shorthand for a diagonal matrix:

```cpp
struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
using State = Composite<Position, Velocity>;

auto const model = ConstantVelocity<Position, Velocity>{0.1};
auto const initial = GaussianParams<State>{Eigen::Vector4d{0, 0, 1, 0.5}, diag(4, 4, 1, 1)};

auto fix(Eigen::Vector2d const & z)
{
    auto const noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    return observe(methods::linearized(),
                   measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(noise)), z);
}
```

Other documents cover what this one uses. `ParticleSet` itself (storage, weights, `moments()`,
the kernel bandwidth, `resample()`'s bookkeeping) is in
`Distributions/ParticleSet.h`. Process and measurement models, their
`reads()`, `writes()`, and inputs, are the Models module page's [model-policies], time steps as a
model sees them `Core/Time.h`'s [time-steps], fixed-step models `FixedStepModel`'s
[fixed-step-models], and `logLikelihoods()` `Models/Likelihood.h`'s [likelihood-measurements]. The filter's spelling, `[[nodiscard]]`, and what "rejected"
means across filters are on the Filters module page and in [update-outcomes]. The execution policies,
allocators, time types, and printing are `Core/Execution.h`'s, `Allocation.h`'s, `Time.h`'s, and
`Print.h`'s. The particle smoother, `ParticleHistory`, is [smoothing.md](smoothing.md) §8. The
`Resample`, `Roughen`, and `WhenEss` function objects belong to `Distributions/Upkeep.h`.
This document covers the statistics behind them and how the particle filter uses them.

## Decisions so far

Nothing has been decided in this review itself. Changes made elsewhere settled Q6
[likelihood-measurements], Q8 [allocation], and Q12. These decisions from
[decisions.md](../design/decisions.md) are in place and should be taken as given:

- **[likelihood-measurements].** A filter that needs $\log p(z \mid x)$ at state points takes it
  from the measurement model's own `logLikelihood(g)` if it has one, and otherwise from the
  additive noise's density at the residual. A model with `reads()` and `logLikelihood(g)` alone,
  no function, noise, or method, is enough, and a model that has both is weighted by its
  likelihood (§4).
- **[filters-and-recursions], [upkeep].** `ParticleFilter<Space, Model, Scalar, Allocator>` is
  spelled like the other filters, without a method. `predict(rng, dt, inputs...)` moves each
  particle through the process model and adds a draw of $Q$; `update(measurement)` multiplies the
  weights by the likelihood and returns the log-likelihood and whether any particle could explain
  the observation, `[[nodiscard]]`. The generator is passed to each call rather than owned.
- **[upkeep].** The filter never resamples on its own. Upkeep is a set of function
  objects the caller applies to `state()`, after the last update before a prediction.
- **[allocation].** The particles come from the set's allocator, and the filter's scratch (the
  likelihoods, and the draws under a policy other than `Sequential`) from the same one, reserved for
  the set's capacity and reused. A resampler that needs storage is handed the set's scratch
  (`ResamplerWithScratch`, §6). Under `Sequential` a prediction draws one particle's normals at a
  time in a vector sized by the noise and the uncertain inputs, so it needs no scratch, and a step
  allocates nothing once it has run at a steady count (§13).
- **[fixed-step-models], [time-steps].** A step of time is any `TimeStep`, converted once to the
  filter's scalar; an integer is not one. With a `FixedStepModel`, `predict(rng, inputs...)` takes
  no step and `predict(rng, Steps{n}, inputs...)` repeats it, draws included (§12).
- **[execution-policies].** `predict()` and `update()` take an execution policy first. A prediction
  draws every particle's normals before it moves any, in Sequential's order, so under `par` it is
  Sequential's bitwise; an update's likelihoods are a for-each and its evidence a reduction in fixed
  chunks, equal to Sequential's to rounding (§11).
- **[filters-and-recursions].** A filter holds a state and runs a recursion. The particle filter has
  the state's half (`state()`, `model()`, the steps) and no separate recursion; §15 and Q11 ask
  whether it should.

## 1. What the filter is

The state is a `ParticleSet`: points $x_i$ with log-weights $\log w_i$, standing for
$\sum_i \hat w_i \, \delta(x - x_i)$ with $\hat w_i = w_i / \sum_j w_j$.

| Step | What it does to particle $i$ |
|---|---|
| `predict(rng, dt, inputs...)` | $x_i \leftarrow f(x_i, u_i) \boxplus L z_i$, with $L L^T = Q(dt)$ and $z_i \sim \mathcal{N}(0, I)$; weights unchanged |
| `update(measurement)` | $\log w_i \leftarrow \log \hat w_i + \log p(z \mid x_i) - \log p(z)$; points unchanged |

With `Resample` applied by the caller this is the bootstrap filter of [Gordon, Salmond, and Smith
(1993)](../reference/bibliography.md#gordon1993), the sequential importance resampling filter of
[Arulampalam et al. (2002)](../reference/bibliography.md#arulampalam2002) §IV.B with the transition
prior as the proposal. The code takes the algorithm whole. It departs in two places: it keeps
log-weights and normalizes them by a log-sum-exp, which the papers leave to the implementer, and it
never resamples on its own [upkeep], where the bootstrap filter resamples at every step and
SIR when the effective sample size falls.

There is no transform method: prediction samples, so the model may be as nonlinear as it likes,
and the update evaluates the likelihood at each point, so a method given to `observe()` is never
used (§4). The filter holds the model, the set, and two scratch buffers (§13), and has two
mutable accessors, `model()` and `state()`, and `get_allocator()`.

A full cycle, next to a Kalman filter on the same linear problem:

```cpp
auto rng = std::mt19937_64{42};
auto particles = ParticleSet<State>{drawSamples(initial.toDensity().value(), 5000, rng)};
auto filter = ParticleFilter{model, particles};         // ParticleFilter<State, Kinematic<double, Position, Velocity>>
auto kalman = KalmanFilter{model, initial};
auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng));

for (int step = 1; step <= 3; ++step) {
  auto const z = Eigen::Vector2d{1.0 * step + 0.3, 0.5 * step - 0.2};
  filter.predict(rng, 1.0);
  kalman.predict(1.0);
  auto const result = filter.update(fix(z));
  auto const exact = kalman.update(fix(z));
  auto const changed = resampling(filter.state());  // std::expected<bool, WeightError>
}
// step 1: logLikelihood -3.648 (Kalman -3.646), ESS 1500, upkeep ran, ESS now 5000
// step 2: logLikelihood -3.065 (Kalman -2.980), ESS 2542, upkeep didn't run
// step 3: logLikelihood -3.037 (Kalman -2.952), ESS 956, upkeep ran, ESS now 5000

filter.state().moments()->mean();  // [3.359, 1.250, 1.042, 0.447]; Kalman [3.317, 1.289, 1.034, 0.477]
// standard deviations:               [0.878, 0.888, 0.661, 0.655]; Kalman [0.819, 0.819, 0.602, 0.602]
```

`Kinematic<double, Position, Velocity>` is the type `ConstantVelocity` names. The means agree to
Monte Carlo error. The spreads are 7 to 10% too wide, which is the cost of `Roughen` (§8): the same
run with `upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))` alone gives $[0.841, 0.830, 0.609, 0.601]$.

Every filter's result spells the log-likelihood `logLikelihood()` and the outcome `accepted`
[update-outcomes]: the particle filter's `result->logLikelihood()` stores the evidence it summed, and
the Kalman filter's `exact->logLikelihood()` derives it from the innovation on request
[innovation-statistics].

The constructor asserts that the set isn't empty and checks that the model writes every
coordinate of the space once, as `KalmanFilter`'s does: at compile time where the types decide it,
and otherwise by an assertion. Handing it a `GaussianParams` instead of a `ParticleSet` fails at
deduction:

```
error: no viable constructor or deduction guide for deduction of template arguments of 'ParticleFilter'
ParticleFilter.h:223:5: note: candidate template ignored: could not match 'ParticleSet' against 'Emrys::GaussianParams'
```

## 2. Building the initial set

Every particle filter starts by drawing a set from a prior, and that still takes a `value()` of
an `expected` and a density view. It no longer takes the space twice: a set of a fixed-size space
takes its draws alone [space-storage].

```cpp
auto particles = ParticleSet<State>{drawSamples(initial.toDensity().value(), 5000, rng)};
```

`tests/Filters/particle_filter.cpp` defines a helper `drawParticles(gaussian, count, rng)` for
exactly this, and the Rao-Blackwellized particle filter has a library function for its own
state, `drawParticleGaussian<Path>(joint, count, rng)`.

> **Q1.** Add `drawParticles(density, count, rng)`, returning an equal-weight `ParticleSet` in
> the density's space, for any density `drawSamples` accepts (a Gaussian, a mixture, a kernel
> density)? It would sit beside `drawParticleGaussian` and make the filter's first line short.
> A `ParticleSet` constructor taking `(density, count, rng)` would do the same, but a
> constructor that consumes a generator reads oddly. An allocator-extended form would also let
> the set be built in its resource rather than moved in (§13).

## 3. Predict

### Process noise

`predict()` asks the model for the discretized $Q(dt)$ over the whole state, factors it once,
and adds $L z_i$ to each propagated particle. The factor is `internal::defaultRoot()` in
`Distributions/Roots.h`: a Cholesky factor, falling back to the pivoted root when $Q$ is only
semidefinite, so $Q$ may be singular. A `Constant` model adds no noise, so the coordinates it writes move only as the
dynamics move them:

```cpp
struct Bias : Linear<1> {};
using Biased = Composite<Position, Velocity, Bias>;
auto const withBias = compose(ConstantVelocity<Position, Velocity>{0.1}, Constant<Bias>{});
auto const start =
    GaussianParams<Biased>{Eigen::Matrix<double, 5, 1>{0, 0, 1, 0.5, 0}, Eigen::Matrix<double, 5, 5>::Identity()};
auto biased = ParticleFilter{withBias, ParticleSet<Biased>{drawSamples(start.toDensity().value(), 4, rng)}};
biased.predict(rng, 1.0);
// std::mt19937_64 rng{42}: bias row before and after: [1.119, 0.015, 0.450, -2.524], unchanged
```

Particles are stored through `setState()`, which normalizes them, so a heading carried past
$\pm\pi$ wraps. The tests check this, and that the tangent-space mean of a cloud straddling the
seam sits just past it.

Each particle's standard normals form one column: the noise's rows, then each uncertain input's,
fixed in size where the spaces are. Under `Sequential` the column is on the stack, drawn and used
before the next particle's, so a prediction takes no scratch (§13). Under another policy every
particle's column is drawn first, in the same order, into scratch the filter owns and reuses, and
the moves then run as a for-each, which is what lets a prediction under a policy match
`Sequential`'s bit for bit (§11). One `std::normal_distribution` serves every draw, so the draws
are those of filling all the columns at once.

### Inputs

Inputs reach the model as in `Emrys::predict()` [prediction-inputs]: each argument after the step is
matched to an item of the model's `reads()` by the tag of its space. An uncertain input (a
Gaussian over a tagged space, the `UncertainInput` concept) is drawn independently for each
particle, and the function reads the draw with `block<Tag>(in, x)` as it reads the state. A known
input, `sure<Tag>(value)`, is gathered into each particle's input beside the draws. A model driven
by a measured acceleration and adding no noise of its own, with 40,000 particles starting from one
exactly known state, $(0, 0, 1, 0.5)$, so only the input spreads them:

```cpp
struct SpecificForce : Linear<2> {};
auto const drivenModel = processModel(
    []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
      auto y = zeros<T>(out);
      Eigen::Matrix<T, 2, 1> const a = block<SpecificForce>(in, x);
      block<Position>(out, y) = block<Position>(in, x) + T(dt) * block<Velocity>(in, x) + T(dt * dt / 2) * a;
      block<Velocity>(out, y) = block<Velocity>(in, x) + T(dt) * a;
      return y;
    },
    reads(path<Position>, path<Velocity>, path<SpecificForce>),
    writes(path<Position>, path<Velocity>));

// std::mt19937_64 rng{42}
auto const accel = GaussianParams<SpecificForce>{Eigen::Vector2d{0.3, -0.1}, diag(0.2, 0.1)};
uncertain.predict(rng, 1.0, accel);                       // sd [0.225, 0.157, 0.451, 0.315]
// sd [0.224, 0.158, 0.447, 0.316]
predict(methods::linearized(), exact, drivenModel, 1.0, accel).covarianceLower();
known.predict(rng, 1.0, sure<SpecificForce>(Eigen::Vector2d{0.3, -0.1}));
// sd [0, 0, 0, 0], mean [1.150, 0.450, 1.300, 0.400]
```

The test suite checks the uncertain input to five standard errors in every mean and covariance
entry with 40,000 particles, with process noise of the model's own added
(`PredictMatchesTheKalmanPredictionInDistribution`). A known input reached the prediction through
an internal `fill()` whose signature had fallen behind the uncertain input's, which no test caught
and which failed to compile from 2026-10-31 until roadmap 17z gathered the sure value into the input
instead (Q15). `tests/Filters/particle_filter.cpp` still has no case with a sure input; the
example above is the check of it here.

> **Q15.** *Decided (roadmap 17z, process models).* A known input is `sure<Tag>(value)`, matched
> to the model's `reads()` by its tag [prediction-inputs], and a model with one reaches no `fill()`:
> the sure value is gathered into each particle's input beside its draws. A plain vector such as
> the `Vector2d` above is refused at compile time with a message naming `sure<Tag>()`, which the
> negative compile test `ARGUMENT_NOT_AN_INPUT` checks, and `tests/Models/models.cpp`,
> `fixed_step_model.cpp`, and `simulation.cpp` predict and simulate with a sure input beside an
> uncertain one. A particle-filter test with a sure input is still owed; the shape of the input is
> no longer the particle filter's to get wrong, and §3's example predicts with one. ~~A particle filter's prediction with a known input
> fails to compile (above). The fix is one parameter on the known input's `fill()`; a test
> predicting with a known input, beside `PredictMatchesTheKalmanPredictionInDistribution`, would
> have caught it, and `tests/parallel.cpp` and `tests/allocation_particles.cpp` have no case with
> one either. Is the fix a bug fix that needs no decision, and should every filter's prediction be
> tested with a known input as well as an uncertain one?~~

### Non-Gaussian process noise

A particle filter's usual reason to exist is noise a Kalman filter can't carry, but both ways
into `predict()` are Gaussian. $Q$ is a covariance. An uncertain input is drawn as
$\mathcal{N}(\mu, \Sigma)$ from its `mean()` and `covarianceLower()`. A `GaussianMixtureParams` doesn't satisfy
`UncertainInput` (checked: `static_assert(!UncertainInput<GaussianMixtureParams<Linear<2>>>)` holds),
and it is not a `sure<Tag>()` input either, so `predict()` refuses it at compile time with the
message for an argument that is neither (§10). Heavy-tailed or multimodal process noise, such as an
occasional maneuver or a wheel slip, can't be written today except by giving the model's function
its own generator outside the library.

> **Q2.** Should the particle filter draw non-Gaussian noise? Options:
> 1. Draw any input that `drawSamples` can draw (a Gaussian, a mixture, a kernel density, and
>    Student-t if it gets a sampler) once per particle, rather than only `UncertainInput`s. The
>    Kalman-family filters would still require a Gaussian. This needs a third kind of argument
>    beside an uncertain and a sure input, and each kind of input would need its rows of the
>    draws (§11) worked out, since a mixture draws a uniform as well as normals.
> 2. Give a process model an optional policy that draws its own noise, used by sample-based
>    filters in place of $Q$.
>
> Option 1 covers process noise that enters as an input, which is most of it, and keeps models free
> of generators. Option 2 covers noise that depends on the state, but a model drawing its own noise
> inside the moves would break [execution-policies]'s rule that the draws come before a parallel
> loop.

The generator comes first in `predict(rng, dt, inputs...)` because the inputs are a trailing
pack; under a policy the policy comes first and the generator second. Elsewhere it comes last
(`drawSamples(density, n, rng)`, `h(state, rng)` and `h(policy, state, rng)` for upkeep,
`r(weights, ancestors, rng)` and `r(policy, weights, ancestors, rng)` for a resampler) or first
again (`ParticleSet::resample(rng, newSize, resampler)`). Passing it in the wrong place fails,
though the first note Clang gives names the fixed-step overload's `Steps`, which a filter with an
ordinary model doesn't have:

```
error: no matching member function for call to 'predict'
ParticleFilter.h:366:5: note: candidate function template not viable: no known conversion from
'std::mt19937_64' (aka 'mersenne_twister_engine<...>') to 'Steps' for 2nd argument
ParticleFilter.h:282:15: note: because 'double' does not satisfy 'uniform_random_bit_generator'
```

> **Q3.** Is there a rule for where the generator goes? Two that fit:
> 1. Last, unless the function ends in a pack, then first. Only `resample()` changes, to
>    `resample(newSize, resampler, rng)`, and its defaulted resampler then needs overloads
>    rather than a default argument.
> 2. First for member functions, last for free functions and function objects. This is what the
>    code does now, so it only needs writing down.
>
> [parallelism.md](../design/parallelism.md) assumes option 2 ("first among the member
> function's arguments, after the policy") for the counter-based generators of step 18b, and the
> upkeep objects put the generator last, so [execution-policies] has made
> option 2 the practice without recording it.

## 4. Update

`update(measurement)` computes $\log \hat w_i + \log p(z \mid x_i)$ for every particle, takes
their log-sum-exp as the evidence $\log p(z) = \log \sum_i \hat w_i \, p(z \mid x_i)$, and
stores the difference, so the weights leave normalized whatever they came in as. The points
don't move. The likelihood is
[Arulampalam et al. (2002)](../reference/bibliography.md#arulampalam2002) equation (48), the
bootstrap filter's weight with the prior as proposal. The result is a
`std::expected<ParticleFilterUpdate<Scalar>, WeightError>`:

| Member | Meaning |
|---|---|
| `logLikelihood()` | $\log p(z)$ under the particles; the Monte Carlo estimate of the Kalman filter's |
| `accepted` | false when every particle has likelihood exactly zero, `logLikelihood()` then $-\infty$; the weights are then unchanged |

Weights that can't be normalized before the update (all zero, or NaN), as a caller can leave them
through the mutable `state()`, return their `WeightError`, as `ParticleSet`'s own functions do, with
the weights unchanged [update-outcomes]. A NaN log-likelihood asserts, once the evidence is summed.
`update()` is `[[nodiscard]]`, so a dropped result is a warning, an error under
`EMRYS_FILTER_WARNINGS_AS_ERRORS`:

```cpp
// From §1's 5,000 draws, std::mt19937_64 rng{42}:
auto const near = filter.update(fix(Eigen::Vector2d{0.5, -0.5}));
// logLikelihood -3.503, accepted, ESS 1718 of 5000, log total weight 0

auto const bounded = filter.update(observe(boundedFix, Eigen::Vector2d{40.0, 0.0}));  // -inf beyond 3 m
// logLikelihood -inf, not accepted, weights unchanged

auto const far = filter.update(fix(Eigen::Vector2d{40.0, 0.0}));   // the same 40 m outlier, Gaussian noise
// logLikelihood -568, accepted, ESS 1.00
filter.state().moments()->mean().head<2>();                         // [7.258, -0.910]
```

`boundedFix` is a measurement model known only by its likelihood [likelihood-measurements], with
`reads()` and `logLikelihood(g)` and no function, noise, or method:

```cpp
auto const boundedFix = measurementModel(
    reads(path<Position>),
    logLikelihood([](auto const & in, auto const & x, auto const & z) {
      Eigen::Vector2d const r = z - block<Position>(in, x);
      return r.norm() <= 3.0 ? -std::log(2 * std::numbers::pi) - 0.5 * r.squaredNorm()
                             : -std::numeric_limits<double>::infinity();
    }));
```

`g(in, x, z)` reads the state as the model's function would, and takes the observation as its
third argument.

`accepted` only catches an observation with zero likelihood everywhere, which Gaussian noise
never gives: in log space nothing underflows. The 40 m outlier above is accepted and puts all
the weight on the one particle nearest it, which is as far out as the cloud reaches. Resampling
then copies that particle 5,000 times, and the filter has lost its distribution. The Kalman
filter guards against this with `Gate` (`Models/Reduction.h`), and the Rao-Blackwellized filters
take a policy, but the particle filter takes none. The test suite's answer is mixture noise: with
an inlier/outlier mixture, 2,000 particles ride out a 60 m outlier every seventh step within 5 m,
and with the inlier noise alone they are dragged past 5 m.

> **Q4.** Should `ParticleFilter::update()` take a gate? Its natural form is on the evidence:
> reject when $\log p(z)$ is below a threshold, or when the observation falls outside a
> probability region of the predicted measurement, using the particles' innovation moments. The
> existing `Gate` works on `MeasurementUpdate` branches and their NIS, which a particle filter
> doesn't produce, so it would be a separate policy type. Or is mixture noise the answer, and
> the documentation should say so?

The result carries less than the Rao-Blackwellized particle filter's, which also has the
per-particle `logLikelihoods`, a per-particle `nis`, and `innovationSize`. A NIS has no meaning
for a bare particle, but the per-particle log-likelihoods are computed anyway, into the filter's
scratch, and not returned.

> **Q5.** Return the per-particle log-likelihoods in `ParticleFilterUpdate`, as the RBPF does?
> They are already in the filter's scratch, so a view of them costs nothing, valid until the next
> update, as the RBPF's are (rao-blackwellized-filters.md §14 asks whether such views should be
> owned instead). They are what a likelihood-based consistency check would read
> and what a caller needs to see which particles an observation favored.
> The alternative is that callers who want them call `logLikelihoods()` themselves, evaluating
> the measurement twice.

Without a NIS, a particle filter has no consistency check that runs on real data, and usage.md
now tells the caller to build one: evaluate the measurement function at each particle before the
update, take the weighted mean and covariance of the values, add the noise covariance, and give
the innovation's squared Mahalanobis length to `NisMonitor::add(nis, dimension)`. A user's agent
building a TDOA geolocator did exactly that, which evaluates the measurement function twice per
particle, once there and once in `update()`.

> **Q13.** Should `ParticleFilterUpdate` carry the predicted measurement's moments, or its NIS,
> so that `NisMonitor::add(update)` takes a particle filter's update as it takes a Kalman
> filter's? The update already evaluates the measurement at every particle, though as a
> likelihood rather than a value, so returning the predicted value's moments would mean a
> measurement function evaluated for its value, which a likelihood-only measurement (Q6) can't
> supply. The alternatives are a free function over the particles and a measurement, or a test
> on the evidence that doesn't assume a Gaussian predicted measurement, such as the probability
> integral transform of each observation under the particles' predictive distribution. The NIS
> is approximate whenever the predicted measurement has several peaks, which is when a particle
> filter is most often chosen.

The two defects the first version of this review found are fixed. The `std::expected`
of the weights' total is no longer dereferenced unchecked: a state whose weights were all set to
$-\infty$ through the mutable `state()` returns its `WeightError` [update-outcomes]. And the NaN
assertion now follows the log-sum-exp, outside the loop that a parallel policy runs.

> **Q12.** *Decided [update-outcomes]: return the `WeightError`.* ~~Should `update()` return
> `accepted = false` when the prior weights can't be normalized, since the `WeightError` is in
> hand, rather than assert and dereference?~~ A rejection with a NaN log-likelihood, which the fix
> first chose, leaves a run's score undefined; `update()` now returns
> `std::expected<ParticleFilterUpdate, WeightError>`, whose error is that one, so that rejected
> means only that no particle explains the observation.

### What a measurement must supply

The likelihood comes from `logLikelihoods(measurement, space, states)` [likelihood-measurements],
written into the filter's scratch by `internal::fillLogLikelihoods()` in `Models/Likelihood.h`: the
model's own `logLikelihood(g)` if it has one, and otherwise the additive noise's density at the
residual, with mixture noise handled exactly. `update()` takes any `Measurement`, an observation of
a measurement model with or without a method (the internal concept `internal::WeighsPoints`).
Nothing else of the measurement is used. A method given to `observe()` is never called, so
`makeRange` in the tests names `methods::unscented()` to no effect; `observe(model, z)` without a
method is the spelling for a filter that weights points, and a Kalman update refuses it at compile
time ("a Kalman update pushes the prior through the model by a method: observe(method, model, z)").

> **Q6.** Should sample-based filters accept a likelihood-only measurement: `Reads` and
> `logLikelihood(blocks...)`, with no method, noise, or innovation? The particle filter and the
> grid filters would take that concept, and `MeasurementModel` would refine it. Today a user
> writing a measurement for a particle filter alone has to invent a Gaussian it never uses.
>
> Decided [likelihood-measurements]: yes. Since roadmap 17z it is a measurement model with
> `reads()` and `logLikelihood(g)` and no function, `measurementModel(reads(path<Position>),
> logLikelihood(g))`, observed without a method; `logLikelihoods()`, `ParticleFilter`,
> `PointMassFilter`, `RbPointMassFilter`, and `RbParticleFilter` take it, the last three only when
> every path it reads lies in the gridded or sampled block. `makeLaplaceFix` in the tests is one,
> with no placeholder noise or method. A model with both a noise and a `logLikelihood(g)` is
> weighted by the likelihood, as before.

Noise that passes through the measurement function, an uncertain input of the call such as a scale
error, has a likelihood at a point only where the model is affine in it: then the input's covariance
carried through the function's Jacobian widens the additive noise's density, which is exact. The
model says so with `affineIn()` [affine-coordinates]; without it, or without additive noise, the
particle filter refuses the measurement with one `static_assert` and no error after it
[likelihood-measurements]. A range scaled by an uncertain `Scale` input, with additive noise and no `affineIn()`:

```
Likelihood.h:126:13: error: static assertion failed due to requirement '...::inputsAffine()': the likelihood at a
point of noise that passes through the function is exact only where the model is affine in it, so name the input in
affineIn(), or give the model logLikelihood(g) (decisions.md [likelihood-measurements])
```

Without the additive noise as well, the first error is the assertion before it, "a measurement
model with no additive noise has no likelihood of its own at a point; give it logLikelihood(g), or
name the inputs it is affine in with affineIn() beside additiveNoise()", followed by the one above
and a third from inside the likelihood.

The crash the first version of this review found, a hand-written measurement whose `noise()`
returned by value with no likelihood of its own, fixed with [view-stamp], went with
the hand-written measurement: a model holds its noise, and `fillLogLikelihoods()` reads it by
reference from the model the measurement holds. The fix of §1 gives $-3.502737$ on the set of §1,
as `ByValueFix` did.

### Scalars

A `ParticleFilter<State, Model, float>` predicts with a `double` model, since [time-steps] converts
the step to the filter's scalar and the built-in models compute in the scalar of the step they are
given. It updates only when its measurements are in `float` too: the noise a
`GaussianParams<Linear<2>, float>` and the observation a `Vector2f`. Given §1's `double` fix it
fails in `Likelihood.h`, at the noise's `double` density evaluated at the `float` residual; the
linear model's `double` matrix, which failed first on 2026-11-04, no longer does. Clang gives four
errors, all from that density:

```
Eigen/src/Core/functors/BinaryFunctors.h:367:86: error: no type named 'ReturnType' in
'Eigen::ScalarBinaryOpTraits<float, double, Eigen::internal::scalar_difference_op<float, double>>'
Gaussian.h:1222:42: note: in instantiation of ...
    'Emrys::Gaussian<Emrys::Linear<2>>::squaredMahalanobisDistance<Eigen::Matrix<float, 2, 1>>'
Likelihood.h:139:38: note: in instantiation of ...
    'Emrys::Gaussian<Emrys::Linear<2>>::logDensity<Eigen::Matrix<float, 2, 1>>'
ParticleFilter.h:416:19: note: in instantiation of ... fillLogLikelihoods<...>
```

> **Q7.** Should a measurement's scalar be allowed to differ from the state's? A `float` filter
> fed by `double` sensor code is a plausible embedded setup, and since [time-steps] the process side
> already accepts it: a `double` model drives a `float` filter. Either `logLikelihoods()` casts
> the innovation to the noise's scalar and the result to the state's, or `update()` checks that
> they agree with a `static_assert` naming both. This is probably a question for every filter,
> and the particle filter is just where it was found.

## 5. Effective sample size

$N_\mathrm{eff} = 1 / \sum_i \hat w_i^2$, from `ParticleSet::ess()`, which
returns `std::expected` like the set's other weight functions. It is the estimate of
[Kong, Liu, and Wong (1994)](../reference/bibliography.md#kong1994) that
[Liu and Chen (1998)](../reference/bibliography.md#liu1998) use to decide when to resample. It
is $N$ for equal weights and 1 when one particle has all the weight. The filter doesn't report
it: [upkeep] leaves it to the state, and `upkeep::whenEss(std::less{}, fraction, actions...)` runs its actions when
$N_\mathrm{eff} < \text{fraction} \cdot N$.

In §1, an update from equal weights left 1,500 of 5,000 effective particles, below half, so
`upkeep::whenEss(std::less{}, 0.5, ...)` resampled. The next update, from equal weights again, left 2,542 and nothing
ran. The third started from those unequal weights and fell to 956. That is the policy working
as intended: resample only when the weights have degenerated, since each resampling adds
Monte Carlo noise of its own.

Where to call it follows from [upkeep]: after the last update before a prediction. With two
sensors reporting at one time, resampling between their updates throws away diversity the second
update could have used, and nothing in the filter prevents it. The `ParticleFilter` class comment
shows the right pattern, and the tests use it. A filter recorded in a `ParticleHistory` must also
resample at most once between records (smoothing.md §8).

The effective sample size says whether the particles represent the posterior. It doesn't say
whether their `moments()` do, which is what a caller reporting a mean and covariance needs: the
moments of a cloud spread along a hyperbola, or over two peaks, describe neither. usage.md's "When
to trust an estimate" now has the caller test it with Mardia's multivariate skewness and kurtosis
of the particles whitened by their moments. The skewness is the sum of the squared third moments
of the whitened coordinates, and the kurtosis the mean fourth power of their length, near 0 and
d(d + 2) for a Gaussian. In the TDOA geolocator the user's agent wrote, both caught the long
early strip (skewness 42 and kurtosis 78 in two dimensions, where a Gaussian gives 0 and 8) and
settled to Gaussian values as the fix formed.

> **Q14.** Should the library provide that test, as a function of a `ParticleSet` (or any
> weighted sample set) returning the skewness and kurtosis, or a verdict at a given tolerance?
> The statistics are those of Mardia (1970), which is not in the bibliography and would need an
> entry. The skewness as Mardia defines it is a double sum over pairs of points, quadratic in the
> count; the sum over the whitened third-moment tensor that the user's agent computed is equal
> and linear, with d^3 terms. A test for mixtures, such as the weight of the heaviest component
> after merging, might belong beside it, with the tests of `Mixture.h`.

## 6. Resampling

`ParticleSet::resample(rng, [newSize], resampler)` turns normalized weights into ancestor
indices with a resampler, copies the chosen particles, sets every weight to $1/\text{newSize}$,
and keeps the indices for `ancestors()`, which the Rao-Blackwellized particle filter reads to
copy each particle's Gaussian and `ParticleHistory` reads to trace lineages, with
`resampleCount()` saying whether they are new; `ParticleSet::resample()` documents the
bookkeeping. The `Resample` action calls it at the current size, or at the size it was given
(`upkeep::resample(rng, n)`), with `resamplers::Systematic` by
default.

A resampler is any function object callable as `r(weights, ancestors, rng)`, with weights a
`std::span<const Scalar>` and ancestors a `std::span<Eigen::Index>` to fill (the `Resampler`
concept). The weights are linear, nonnegative, and need not sum to one; ancestors may be any
length, which is how `resample(rng, newSize)` changes the count. One that also takes a policy
first, `r(policy, weights, ancestors, rng)`, satisfies `ResamplerUnder` (§11). One that needs
storage sized by the counts also takes it as scratch, `r(weights, ancestors, rng, scratch)`, a
`std::span<Scalar>` of at least `weights.size() + ancestors.size()` elements, and satisfies
`ResamplerWithScratch`: `resample()` keeps that scratch and passes it, so `resamplers::Multinomial`
and `resamplers::Residual` allocate nothing there, while their three-argument forms allocate what
they need [allocation]. A function object that doesn't fit fails at the concept:

```
error: no matching member function for call to 'resample'
ParticleSet.h:1180:18: note: because 'Resampler<(lambda ...), Scalar, std::mersenne_twister_engine<...>>' evaluated
    to false
Resampling.h:62:5: note: because 'r(weights, ancestors, rng)' would be invalid: no matching function for call to
    object of type 'const (lambda ...)'
```

## 7. The four resamplers

All four invert the cumulative weights at $M$ positions in $[0, 1)$ (`internal::invertCumulative`)
and differ in how they choose the positions. A zero-weight index is never chosen, including
trailing zeros, which the inversion guards against explicitly. They are the four that
[Hol, Schön, and Gustafsson (2006)](../reference/bibliography.md#hol2006) compare, and each
cites its own source:

| Resampler | Positions | Offspring of index $i$ | Source |
|---|---|---|---|
| `resamplers::Multinomial` | $M$ independent uniforms, sorted | binomial; highest variance | [Gordon et al. (1993)](../reference/bibliography.md#gordon1993) |
| `resamplers::Stratified` | one uniform in each of $M$ equal strata | lower variance | [Kitagawa (1996)](../reference/bibliography.md#kitagawa1996) |
| `resamplers::Systematic` | one uniform offset, then evenly spaced | $\lfloor M \hat w_i \rfloor$ or $\lceil M \hat w_i \rceil$; the default | [Carpenter, Clifford, and Fearnhead (1999)](../reference/bibliography.md#carpenter1999) |
| `resamplers::Residual<R>` | $\lfloor M \hat w_i \rfloor$ copies first, then `R` on the fractional parts | at least $\lfloor M \hat w_i \rfloor$; `R` is multinomial by default | [Liu and Chen (1998)](../reference/bibliography.md#liu1998) |

The code takes each position rule as the header's citation states it. What it adds is shared by
all four: `internal::uniform01` clamps a uniform that `std::uniform_real_distribution` rounds to 1
down to the largest value below it, and the inversion never stops past the last positive weight.

On the weights $(0.8, 0.4, 0, 1.6, 0.2, 0)$, six ancestors, offspring counts over 20,000 draws
(expected $1.6, 0.8, 0, 3.2, 0.4, 0$; indices 2 and 5 are never chosen), the same under Clang and
GCC, from `std::mt19937_64{1}` with the first draw shown and the next 20,000 counted:

```
                               one draw         variance of each count (indices 0 1 3 4)   total
resamplers::Multinomial           0 0 0 1 3 3      1.17 0.68 1.48 0.37                         3.70
resamplers::Stratified            0 0 3 3 3 4      0.24 0.48 0.47 0.24                         1.43
resamplers::Systematic            0 0 1 3 3 3      0.24 0.17 0.17 0.24                         0.81
resamplers::Residual<>            0 3 3 3 0 0      0.42 0.48 0.18 0.32                         1.40
resamplers::Residual<Systematic>  0 3 3 3 0 1      0.24 0.17 0.17 0.24                         0.81
```

Multinomial's variances are the binomial $M \hat w_i (1 - \hat w_i)$. Systematic's are the
least possible for counts restricted to floor or ceiling, and residual with a systematic
remainder is the same here. Three smaller observations:

- The residual resampler's ancestors are not sorted: the deterministic copies come first, then
  the remainder's. The others produce sorted ancestors under `Sequential`; under a parallel
  policy the multinomial resampler leaves its ancestors in the order of its draws (§11). Nothing
  in the library depends on order, but a caller resampling in place might.
- `ParticleSet::resample()` and the `ParticleSet` class comment say repeated resampling at a
  steady size doesn't allocate, and that now holds of all four: the systematic and stratified
  resamplers need no storage, and the multinomial and residual ones take `resample()`'s scratch as
  `ResamplerWithScratch` (§6), where their three-argument forms allocate.
  `Allocation.EveryResamplerStaysWithinTheResource` checks all five configurations of the table.
- The resamplers require `std::floating_point` weights, while `ParticleSet` accepts any
  `RealScalar`, including autodiff types. A set with an autodiff scalar can't resample, and
  since `internal::fillStandardNormal()` requires a floating-point scalar too (a
  `static_assert`), a filter over one can't predict either.

> **Q8.** *Decided [allocation]: scratch from the set.* ~~Make the resamplers allocation-free, so
> the claim in `resample()` holds whichever is passed?~~ A resampler that needs storage takes it
> as `r(weights, ancestors, rng, scratch)`, and `resample()` passes scratch it keeps, so all four
> resample within the set's resource; [allocation] names `ResamplerWithScratch` as the way a
> function object passed in is given scratch. The multinomial resampler still sorts its $M$
> uniforms, in that scratch; generating them in order by exponential spacings, or step 18b's
> resamplers that need no sort, would remove the $O(M \log M)$, and are beyond 0.1.

> **Q9.** Should the residual resampler's default remainder be systematic rather than
> multinomial? It gives the lowest variance of the five above, and `resamplers::Residual<>` is what
> a user who asks for "residual" will get. Multinomial remainder is the textbook definition (Liu
> and Chen), which is the case for keeping it.

### A missing precondition check

The other resamplers assert that every weight is nonnegative, inside `invertCumulative`. The
residual resampler computes its copy counts first, and on 2026-10-27 a negative weight gave a
negative count converted to `std::size_t`, undefined behavior: $(1, -0.5, 1)$ gave ancestors
$0, 0, 1$ with no assertion. Now its sequential form asserts each weight nonnegative
before counting copies and takes no copies for a negative count, so under Clang and GCC the same
weights stop at "weights must be nonnegative". Its form under a parallel policy does neither: it
sums the weights by a reduction, counts copies in a for-each, and places them by a scan, and
$(1, -0.5, 1)$ under `std::execution::par` gives ancestors $0, 2, 2$ with no assertion, under both
compilers. The negative count makes the scanned ends decrease, so index 0's copies and index 2's
both write ancestor 1, a data race when the for-each runs on threads.

> **Q16.** Should the residual resampler's policy form assert each weight nonnegative, as its
> sequential form does, inside the reduction that sums them? A negative weight is a precondition
> violation either way, but under a policy it is also a race on the ancestors, and nothing in
> `tests/parallel.cpp` gives a resampler a bad weight.

## 8. Roughening

Resampling leaves duplicates. If the process noise is small, as for a parameter held constant,
they stay duplicates, and the cloud collapses onto a few points. Roughening breaks ties by moving
every particle by a draw from a Gaussian kernel: `upkeep::roughen(rng, s)` adds $\mathcal{N}(0, s H)$ to each
particle, with $H$ the set's `kdeBandwidth()`. With no bandwidth stored, that is the normal
reference rule of [Silverman (1986)](../reference/bibliography.md#silverman1986) §4.3.2,
$H = c \, \Sigma$ with $c = \left(\frac{4}{N_\mathrm{eff}(d + 2)}\right)^{2/(d+4)}$ and $\Sigma$
the weighted covariance; the code takes Silverman's rule and replaces his $N$ by the effective
sample size, so unequal weights give a wider kernel. Applied after resampling, it is a draw from
the kernel density estimate of the cloud, which makes the pair the regularized particle filter of
[Musso, Oudjane, and Le Gland (2001)](../reference/bibliography.md#musso2001). The code takes
from them the step of jittering every particle by a Gaussian kernel after resampling; the
kernel's bandwidth is Silverman's rule as above, and `scale` multiplies it, which is the code's
own knob. The kernel is drawn in the tangent space and the result normalized, so periodic
coordinates wrap.

A draw from the kernel density has the cloud's covariance plus $s H$, so each `Roughen` inflates
the covariance by a factor $1 + s c$. For four dimensions and 5,000 particles $c = 0.1075$:

```cpp
auto p = ParticleSet<State>{State{}, drawSamples(initial.toDensity().value(), 5000, rng)};
p.kdeBandwidth().value()(0, 0) / p.moments()->covarianceLower()(0, 0);  // 0.1075
auto const roughen = upkeep::roughen(rng);
// covariance diagonal:    [4.062, 3.947, 1.027, 0.998]
(void)roughen(p, rng);  // [4.505, 4.350, 1.122, 1.111], about 10 to 11% larger
// after 10 in a row:      [11.081, 11.167, 2.790, 2.762], 1.1075^10 = 2.78 times
```

In a filter, updates pull the spread back, but not all the way: the 7 to 10% excess in §1's
standard deviations is two roughenings in three steps. The usual correction is Liu and West's
shrinkage: move each particle toward the weighted mean, $x_i \leftarrow a x_i + (1 - a) \bar x$
with $a = \sqrt{1 - s c}$, before adding the kernel draw, which leaves the mean and covariance
unchanged in expectation while still breaking ties.

> **Q10.** Should `Roughen` shrink toward the mean, so that it preserves the first two moments?
> As an option (`upkeep::roughen(.scale = s, .shrink = true)`), as a separate action
> (`Regularize{s}`), or as the default? The plain kernel is what the regularized particle filter
> literature describes, and inflation is a defensible guard against an overconfident filter, but
> it is a bias that grows with how often `WhenEss` fires, which the user doesn't control
> directly. I'd make shrinkage the default and keep the plain kernel as an option. Liu and West
> (2001) is not in the bibliography; it would need an entry before the code could cite it.

`resample()` clears a stored bandwidth, so after a resample `Roughen` takes Silverman's rule of
the cloud it moves unless a bandwidth was stored since. Roughening a Rao-Blackwellized particle state moves the
particles and leaves their Gaussians, which rao-blackwellized-filters.md §9 covers.

## 9. What the tests show

From `tests/Filters/particle_filter.cpp`, `tests/Core/resampling.cpp`, `tests/parallel.cpp`,
`tests/allocation_particles.cpp`, and roadmap step 7, condensed:

- `logLikelihoods()` over a sample matrix equals the Gaussian likelihood by hand, the mixture
  noise's density at each residual, and a model's own `logLikelihood(g)`; a model known only by
  its likelihood weights the particles by it, and so does one with Gaussian noise beside it.
- Predicted particles are a draw from the Kalman prediction, including an uncertain input, to
  five standard errors with 40,000 particles.
- A `FixedStepModel` moves every particle exactly as predicting over the step does, from the
  same particles, input, and seed.
- A `Constant` model composed beside the kinematics makes $Q$ singular, and its coordinate doesn't move. A heading predicted
  across the seam stays in range, and its tangent-space mean sits just past the seam.
- An update's weights and log-likelihood match a hand computation to $10^{-12}$; a measurement no
  particle can explain is rejected with the weights untouched.
- On a constant-velocity run with 20,000 particles and `upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))`, means stay
  within 0.15 standard deviations of the Kalman filter's, covariance entries within 0.15
  relative, and the log-likelihood within 0.15, about twice the worst seen over 30 seeds. The
  fixes there are a fixed offset from the truth: random fixes occasionally land in the tail,
  leave a few hundred effective particles, and made the test seed-dependent. That is the outlier
  collapse of §4 in a milder form.
- Mixture noise keeps 2,000 particles within 5 m through injected 60 m outliers.
- Mirror-image clouds keep equal weight through five range updates, and a position fix resolves
  them.
- Each resampler never chooses a zero weight, gives offspring in proportion to weight within five
  standard deviations, and handles a single nonzero weight. Systematic counts are floor or
  ceiling; residual counts are at least the floor.
- `update()`'s result is checked in the outlier test, since it is `[[nodiscard]]`; an update over
  weights that can't be normalized is rejected with assertions off (`tests/assert_override.cpp`).
- Under `std::execution::par`, each resampler draws as Sequential does and gives its ancestors
  (sorted, for the multinomial ones); a prediction with an uncertain input is Sequential's
  bitwise, also with a fixed-step model under `par_unseq`; an update agrees to $10^{-12}$ with
  Gaussian and with mixture noise.
- Over a pmr set with a capacity, predict, update, and `upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))` allocate
  nothing after warm-up, with a fixed-step model too; every resampler resamples within the
  resource; and `Roughen` under `Sequential` allocates nothing.

Nothing tests a particle filter with a non-`double` scalar, a prediction with a known input
(Q15), roughening inside a filter loop against the Kalman filter, a resample to a different size,
or a `std::chrono` step (the time tests cover the Kalman-family filters).

## 10. What misuse looks like

| Misuse | Fails | First error (Clang) |
|---|---|---|
| `predict(1.0)`, no generator | compile | no matching member function for call to 'predict': 'double' does not satisfy 'uniform_random_bit_generator' |
| `predict(1.0, rng)` | compile | no known conversion from 'std::mt19937_64' to 'Steps' for 2nd argument (§3) |
| `predict(rng, 1)`, an integer step | compile | 'TimeStep<int, double>' evaluated to false |
| a `GaussianParams` as the initial state | compile | could not match 'ParticleSet' against 'Gaussian' |
| `upkeep::resample(rng)(filter)` instead of `filter.state()` | compile | no member named 'resample' in 'Emrys::ParticleFilter<...>' |
| a resampler with the wrong signature | compile | 'Resampler<...>' evaluated to false |
| noise through the measurement function, not declared in `affineIn()`, and no `logLikelihood(g)` | compile | static assertion: the likelihood at a point of noise that passes through the function is exact only where the model is affine in it (§4) |
| a `double` measurement into a `float` filter | compile | no type named 'ReturnType' in 'Eigen::ScalarBinaryOpTraits<float, double, ...>' (Q7) |
| `update()`'s result dropped | compile, warning | ignoring return value of function declared with 'nodiscard' attribute |
| a known input as a plain `Vector2d`, not `sure<Tag>(v)` | compile | static assertion failed: an argument after the observation is an uncertain input over a tagged space, sure<Tag>(value), ... (Q15) |
| an input that the model does not read | compile | static assertion failed: the call was given an input the model does not read ([prediction-inputs], Q15) |
| a `GaussianMixtureParams` as an input | compile | the same assertion as a plain vector, being neither uncertain nor sure (Q15, Q2) |
| an empty initial set | run, debug | assertion: the initial particle set must have a particle |
| weights all $-\infty$ through `state()` | run | debug: assertion; release: rejected, `logLikelihood` NaN (Q12) |
| a 40-sigma outlier with Gaussian noise | never | accepted; ESS 1 (Q4) |
| a negative weight to `resamplers::Residual` | run, debug | `Sequential`: assertion "weights must be nonnegative"; under a policy, none, and a race on the ancestors (Q16) |

## 11. Execution policies

Both steps take a policy first ([execution-policies]; `Core/Execution.h` covers the policies and the
loop primitives), and upkeep takes it in its factories (the Upkeep module page):

```cpp
filter.predict(std::execution::par, rng, 1.0);
auto const updated = filter.update(std::execution::par, fix(z));
auto const inParallel = upkeep::whenEss(std::execution::par, std::less{}, 0.5,
                                        upkeep::resample(std::execution::par, rng),
                                        upkeep::roughen(std::execution::par, rng));
auto const ran = inParallel(filter.state());
```

A prediction draws every particle's standard normals first, in the order the sequential loop
takes them (the noise's rows, then each uncertain input's), into the column of the scratch that
belongs to the particle. The moves are then a for-each that reads only its own column and writes
only its own particle, so a prediction under `par` is Sequential's bitwise from the same seed. The
process model's function runs for several particles at once, so a user's function must not
write shared state. An update's likelihoods are a for-each into the scratch, and its total weight
and evidence are reductions in fixed chunks, so they match Sequential's to rounding and are the
same from run to run:

```cpp
// Two copies of a filter over §1's 5,000 draws from std::mt19937_64{42}; a{7}, b{7}, c{7}; z = (1.3, 0.3)
seconds.predict(a, 1.0);
parallel.predict(std::execution::par, c, 1.0);
// states equal: 1
seconds.update(fix(z))->logLikelihood();                        // -3.648045177606063
parallel.update(std::execution::par, fix(z))->logLikelihood();  // -3.648045177606062
// largest difference in the log-weights: 3.55e-15
```

On 2026-10-27 the two log-likelihoods differed in the last digit under Clang, on 2026-11-04 they
printed the same digits from a setup this document did not record, and GCC's differed by
1.78e-15; from the setup above they differ in the last digit again. Nothing promises more than
rounding.

Under Clang with the MSVC STL the loops run on threads. w64devkit's GCC has no TBB and runs them
on one, so locally only the MSVC library tests real threads (CLAUDE.md).

The resamplers run under a policy as `r(policy, weights, ancestors, rng)`, which is the
`ResamplerUnder` concept; `ParticleSet::resample(policy, ...)` calls a resampler that way if it can
and as a plain `Resampler` otherwise. The draws stay sequential; the cumulative sum becomes a scan
under the policy and each ancestor a binary search into it. That sum is the one collective operation
[Murray, Lee, and Jacob (2016)](../reference/bibliography.md#murray2016) find these resamplers need
in parallel; the code takes that observation and none of their own resamplers, which the roadmap
leaves to step 18b. The ancestors are Sequential's but for rounding at the boundaries between
particles, except that the multinomial resampler under a parallel policy no longer sorts its
uniforms, so its ancestors come in the order of its draws, the same draw of ancestors in another
order. The scan and the multinomial resampler's uniforms allocate, which [execution-policies] allows
away from `Sequential`.

## 12. Time steps and fixed-step models

`predict(rng, dt, inputs...)` takes `dt` as a `TimeStep<Scalar>` ([time-steps]), so a
`std::chrono` duration works and gives the same particles bitwise as the same span in seconds; an
integer does not compile (§10). The step is converted once, with `internal::toScalarTime<Scalar>()`,
and below that the process model sees the filter's scalar:

```cpp
seconds.predict(a, 1.0);
millis.predict(b, 1000ms);   // same seed: states equal, 1
```

With a `FixedStepModel` [fixed-step-models] the step is the model's, so
`predict(rng, inputs...)` takes none and uses the model's cached $Q$, and
`predict(rng, Steps{n}, inputs...)` is $n$ such calls, draws included. The two overloads without a
count exclude a leading `Steps`, as they exclude a leading policy:

```cpp
auto fixed = ParticleFilter{fixedStep<State>(model, 1.0), particles};
for (int k = 0; k < 3; ++k) fixed.predict(d);
stepped.predict(e, Steps{3});      // same seed: states equal, 1
onestep.predict(g);                // equals plain.predict(f, 1.0) from the same seed: 1
```

## 13. Allocation

The filter's two scratch buffers, the log-likelihoods and the draws under a policy, come from the
state's allocator, re-bound to the scalar, and the constructor reserves them for the set's
capacity if it has one ([allocation]), the draws for the noise's rows. `get_allocator()`
returns the set's. A prediction under `Sequential` takes no scratch: it draws each particle's
normals, the noise's rows and then each uncertain input's, into one column on the stack, fixed in
size where the spaces are, and moves the particle before drawing the next, with the same draws as
before. Under another policy it draws every particle's column first into the draws buffer, which
the first prediction with an uncertain input grows to hold the input's rows and every later one
reuses. On a bounded space, with a capacity set and `Sequential`,
predict, update, and `upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))` allocate nothing once they have run at a steady
count, which `Allocation.ParticleFilterStepsWithinItsResource` checks for an ordinary and a
fixed-step model, and a prediction with an uncertain input allocates nothing from construction
on, which `Allocation.ParticleFilterPredictsWithAnUncertainInputFromConstructionOn` checks:

```cpp
auto particles = ParticleSet<State, double, Pmr>{drawSamples(initial.toDensity().value(), n, rng), allocator};
particles.setCapacity(n);
auto filter = ParticleFilter{model, std::move(particles)};   // filter.get_allocator() == allocator
```

The particles are moved in, which saves a copy. A copy would stay in the resource too: against
pmr's rule, [allocation] has a copy of a container keep its source's allocator and capacity, so a
filter built from a copy of a set, or a copy of a whole filter, steps without an allocation from
the first step on (`Allocation.CopiesKeepTheirSourcesResourceAndCapacity`). Nothing in the
particle filter's own orbit allocates per call on `Sequential` any longer: every resampler takes
the set's scratch through `resample()` (§7), and `Roughen` draws and moves one particle at a time
(`Allocation.RoughenKernelDensityAndSingleDrawsAllocateNothing`). `Roughen` of a
`ParticleGaussianParams` still copies the set, which the roadmap's allocation items list.

## 14. Printing

`std::cout << filter` prints "particle filter" and the filter's state under it
([printing-and-simulation]), a summary rather than the particles:

```
particle filter
  particle set of 5000 particles of dimension 4, effective sample size 5000
    mean: 3.35893 1.25007 1.04239 0.447286
    covariance:
          0.770971   -0.0358905      0.38906  -0.00554999
        -0.0358905      0.78931   -0.0211302     0.400068
           0.38906   -0.0211302     0.437312 -2.91252e-05
       -0.00554999     0.400068 -2.91252e-05     0.428428
```

That is §1's filter after its third step. The summary leaves out the bandwidth, whether one is
stored, and the resample count, which a reader debugging a stale bandwidth (§8) would want.

## 15. The shape of the filter after [filters-and-recursions]

[filters-and-recursions] splits the Kalman, mixture, and IMM filters into a state and a recursion,
the model, method, and scratch acting on a state passed in, so a mixture filter can run any
component's recursion on each of its components. The particle filter was not split. It has the
state's surface, `state()`, `model()`, `get_allocator()`, and the steps, and keeps its recursion
inline: the model, the two scratch buffers, and `moveParticles()`. Nothing composes a particle
filter today. The Rao-Blackwellized particle filter keeps its own steps too
(rao-blackwellized-filters.md §16).

> **Q11.** Should the particle filter have a `ParticleRecursion<Space, Model, Scalar, Allocator>`,
> the model and the scratch acting on a `ParticleSet` passed in, as [filters-and-recursions] gives
> the Gaussian filters? It would let one model drive several sets (a bank of filters for multiple
> targets, or a filter run on a copy of its state for a what-if prediction) without copying the
> model and the scratch for each. Or is [filters-and-recursions]'s rule about the filters whose
> state is a component, and should [filters-and-recursions] say so, so that a reader doesn't look
> for the particle filter's recursion?

## Questions

1. A `drawParticles(density, count, rng)` for the initial set (§2).
2. Non-Gaussian process noise: draw any samplable input per particle, or a model's own
   sampling policy (§3).
3. Where the generator argument goes; [execution-policies] has made option 2 the practice (§3).
4. A gate for the particle filter, on the evidence (§4).
5. Return the per-particle log-likelihoods from `update()` (§4).
6. ~~A likelihood-only measurement concept for sample-based filters (§4).~~ Decided
   [likelihood-measurements].
7. Measurements whose scalar differs from the state's, now that models may (§4).
8. ~~Allocation-free resamplers, or a narrower claim in `resample()` (§7).~~ Decided
   [allocation]: they take the set's scratch.
9. Systematic as the residual resampler's default remainder (§7).
10. `Roughen` with Liu–West shrinkage, preserving the covariance (§8).
11. A recursion for the particle filter, or [filters-and-recursions] scoped to component filters
    (§15).
12. ~~`update()` on weights that can't be normalized: reject rather than assert and dereference
    (§4).~~ Decided [update-outcomes]: return the `WeightError`.
13. A particle filter's NIS: the predicted measurement's moments in `ParticleFilterUpdate`, a
    free function, or a test on the evidence (§4).
14. A test of whether a particle set's moments describe it: Mardia's skewness and kurtosis (§5).
15. ~~A prediction with a known input fails to compile since 2026-10-31; fix, and test known inputs
    (§3).~~ *Decided: a known input is `sure<Tag>(value)` (roadmap 17z).*
16. The residual resampler's policy form doesn't assert nonnegative weights, and races on the
    ancestors given a negative one (§7).
