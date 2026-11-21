# Particle proposals, moves, and transition densities

How a particle filter uses the measurement while it moves its particles, how it moves them again
after a resampling, and what the density of a process model's step is. This is the plan behind
roadmap steps 17zh and 17zi, and it says what step 14b's backward simulation will take from them. A
prototype of it is built, on the branch `particle-proposals`, to try the open questions against
code; "Prototype" below says what it holds, what was checked, and what is missing. Every open
question has a proposed answer, which the prototype implements, and every one is still open: nothing
is decided here. Where this plan disagrees with steps 17zh and 17zi, this plan is the newer one.
Status and ordering are in [roadmap.md](roadmap.md).

The branch is on top of `particle-review`, which holds [lazy-draws], [sampled-inputs], and
[generator-arguments] and has not landed, since a step's signature follows the third of them. The
plan was worked out on 2026-11-21.

## Where we are

A particle filter today is `predict(dt, rng, inputs...)`, which draws each particle from the process
model alone, and `update(measurement)`, which weights it: the bootstrap filter. Step 17zi asks for
proposals that use the measurement, the auxiliary particle filter, and resample-move, and says the
shape of the step is to be decided first. Step 17zh asks for a process model's transition density
and calls it the prerequisite of all three.

Five findings from the prototype change that plan:

- **A proposal over a particle's draws needs no transition density.** A particle's next state is
  $x' = f(x) \boxplus F c$, with $F F^T = Q$ the factor a prediction draws with and
  $c \sim \mathcal{N}(0, I)$ the particle's draws. A proposal is a density $q(c)$ over the draws,
  and the weight is multiplied by $p(z \mid x') \, \mathcal{N}(c; 0, I) / q(c)$. Nothing is evaluated
  in the state, so a singular $Q$ needs no special case: a direction the noise does not reach is a
  zero column of $F$. The Gaussian proposal's covariance, $I - G^T S^{-1} G$ with $G = H F$, is
  positive definite whatever the rank of $Q$. So 17zh is not a prerequisite of the proposals or of
  the move as built. Backward simulation needs it, and so does a move made in the state.
- **The measurement's method chooses the proposal.** A Kalman update pushes a Gaussian through the
  model by the method of `observe(method, model, z)`, and a particle filter's `update()` ignores the
  method. A step that uses the measurement pushes each particle's draws through it by that method:
  `Linearized` gives the proposal by local linearization, a sigma-point method the unscented
  particle filter's. A model given its matrix carries `Linearized` without being told
  [linear-sensors], so `filter.step(dt, observe(fix, z), rng)` is the optimal proposal with no name
  for it. A measurement observed without a method draws from the model alone, and the step is then
  `predict()` followed by `update()`, bitwise.
- **The auxiliary filter's resampling can be the upkeep the filter holds.** [upkeep] says a filter
  never resamples on its own. The auxiliary step multiplies each weight by a look-ahead likelihood,
  calls `maintain()`, which runs the upkeep the caller gave the filter, and then moves the particles.
  The caller's upkeep decides whether and how to resample, at every step as Pitt and Shephard do or
  under `whenEss()`. The second stage computes each particle's look-ahead likelihood again at the
  state it moves from, so it reads no ancestor.
- **A move needs the state each particle moved from and its weight factor, and no density.** The
  filter keeps both at each step, one copy of the states, and follows them through one resampling by
  `ancestors()`. The move proposes from the step's own proposal and accepts by the ratio of weight
  factors, an independence sampler.
- **The optimal proposal does not recover what a wide prior loses.** Its weight is
  $p(z \mid x)$ for the state $x$ moved from, whose width is the process noise's. "Evidence" has the
  numbers.

## The calls

```cpp
// The bootstrap filter, as today. Uncertain and sampled inputs are drawn per particle.
filter.predict(dt, rng, inputs...);
auto const updated = filter.update(observe(range, z));

// One step with the measurement in hand. The method builds each particle's proposal.
auto const stepped = filter.step(dt, observe(methods::unscented(), range, z), rng);
auto const optimal = filter.step(dt, observe(fix, z), rng);   // fix is given its matrix

// The auxiliary particle filter: look ahead, run the filter's upkeep, then step.
auto filter = ParticleFilter{model, particles, upkeep::resample(resamplingRng)};
auto const stepped = filter.auxiliaryStep(dt, observe(fix, z), rng);

// Resample-move: after the caller's resampling, move the copies apart.
auto const stepped = filter.step(dt, measurement, rng);
if (filter.maintain().value())
{
    Eigen::Index const moved = filter.move(dt, measurement, rng);
}
```

`step()` and `auxiliaryStep()` return what `update()` returns. Each takes an execution policy first,
and the generator after the measurement, which the call requires [generator-arguments].

## The transition density

`transitionDensity(space, model, dt, inputs...)` returns a `TransitionDensity`, which factorizes $Q$
once and evaluates `logDensity(from, to)`, or `mean(from)` and `logDensityAbout(mean, to)` where one
present state meets many next states. It is the Gaussian density of $x' \boxminus f(x)$ under $Q$
[tangent-gaussians].

With a singular $Q$ it is the degenerate Gaussian's density on the directions the noise reaches,
with the pseudo-determinant, and $-\infty$ at a state the noise cannot reach. It decides which
directions are reached by the factorization a prediction draws with: Cholesky, and where that fails
the pivoted $L D L^T$, whose pivots that are not positive `roots::Pivoted` takes as zero. So a state
that `predict()` or `simulate()` drew has a finite density from the state it was drawn from, which a
rank decided by a threshold on eigenvalues would not promise.

## Evidence

One step from 4,000 particles drawn about a prior with a position spread of 2 and a velocity spread
of 1, a range of variance $R$ from a point 23 away, and a constant-velocity model of acceleration
density $q$; the effective sample size after the step, the mean of 20 seeds (2026-11-21, Clang on
Windows, a scratch program not in the repository):

| $R$ | $q$ | `predict()` then `update()` | `step()`, `Linearized` | `step()`, `Unscented` |
| --- | --- | --- | --- | --- |
| 0.04 | 0.1 | 496 | 673 | 673 |
| 0.04 | 3 | 464 | 2158 | 2193 |
| 0.0004 | 0.1 | 51 | 457 | 457 |
| 0.0004 | 3 | 46 | 359 | 992 |

The four log-likelihoods of each row agree within 0.04. A step with the range observed without a
method gave `predict()` then `update()`'s numbers exactly. The gain is largest where the process
noise is wide against the measurement's, and the last row shows the Jacobian's proposal losing to
the sigma points' where the noise is wide against the bend of the range.

## Prototype

| What is built | Where |
| --- | --- |
| `TransitionDensity` and `transitionDensity()`, with a model's step and a fixed-step model's | `Models/TransitionDensity.h`, `tests/Models/transition_density.cpp` |
| `ParticleFilter::step()`, `auxiliaryStep()`, and `move()`, each with a policy overload | `Filters/ParticleFilter.h`, `tests/Filters/particle_proposals.cpp` |
| The negative compile cases `TRANSITION_DENSITY_UNCERTAIN_INPUT` and `PARTICLE_STEP_UNCERTAIN_INPUT` | `tests/Models/compile_errors.cpp`, `tests/Filters/compile_errors.cpp` |
| Bibliography entries for Doucet, Godsill, and Andrieu (2000), Pitt and Shephard (1999), Gilks and Berzuini (2001), van der Merwe et al. (2000), and Tierney (1994), each marked not read | `docs/reference/bibliography.md` |

**What was checked.** With Clang on Windows, each of `tests/Models/transition_density.cpp` (10
tests), `tests/Filters/particle_proposals.cpp` (8), `tests/Filters/particle_filter.cpp` (17), and
`tests/Models/simulation.cpp` (11) was compiled and run as a program of its own, and all pass. The
two negative cases fail with their messages, and their files compile without them. clang-format,
the layer check, the decisions check, and the check of compile-error claims pass. The `clang`
workflow has not run, and neither have MSVC, GCC, the Doxygen build, or any test file not named
here, the allocation and parallel tests among them.

The tests compare with exact references where there is one: the transition density with a Gaussian
density by Cholesky in plain Eigen, the optimal proposal's weights with
$\mathcal{N}(z; H f(x), H Q H^T + R)$ at each particle to $10^{-9}$, and a fully adapted auxiliary
step with equal weights to $10^{-9}$. The guided step, both auxiliary steps, and the moved filter
stay within 0.15 of the Kalman filter over five steps, the tolerance of the bootstrap filter's own
test.

**What is missing.**

- The steps with a fixed-step model, with an uncertain or a sampled input (refused at compile
  time), and with a batched process function (not tried).
- A guided step under mixture measurement noise or an `Iterated` method. The first fails with the
  square-root forms' message, which is the wrong one here.
- Several measurements at one time. A `step()` followed by an `update()` is correct, but the
  `update()` ends the record, so no `move()` can follow it.
- The same steps in `RbParticleFilter`.
- An adaptive particle count by KLD-sampling.
- A failure that returns an error. An innovation covariance that is not positive definite, and a
  proposal covariance that rounding leaves indefinite, as a measurement far more precise than the
  process noise can, are assertions. The proposal's root should be taken without forming
  $I - G^T S^{-1} G$.
- Scratch reserved at construction, and the cases in `tests/allocation_particles.cpp` and
  `tests/parallel.cpp`. The steps draw every particle's normals into scratch first under every
  policy, where `predict()` under `Sequential` takes none.
- The density of many pairs of states at once under a policy, which backward simulation needs.
- A `ParticleHistory` over an auxiliary step, which resamples between the history's records.
- [usage.md](../usage.md), [choosing.md](../choosing.md), the design review's sections, and the
  decisions.
- The five bibliography entries were written without the sources at hand, and so was the comment
  that says what each call takes from its source. Each is to be checked against the source.

## Open questions

Each question is followed by the answer the prototype implements. The answers are proposals.

1. **What is the density when $Q$ is singular?** Proposed: the degenerate Gaussian's on the
   directions reached, by the prediction's own factorization, and $-\infty$ off them. A next state
   counts as reached when its residual along each direction not reached is within
   $16 n \epsilon$ times the largest coordinate of the two states. That one scale for every
   coordinate is loose for a state whose blocks differ by many orders of magnitude; a scale per
   coordinate, or a tolerance the caller gives, are the alternatives. Step 17g's singular $Q$ can
   take the same split into directions reached and not.
2. **What does the density do with an uncertain input?** Proposed: refuse it at compile time. An
   input the model is `affineIn()` could widen $Q$ exactly, at the cost of a factorization per
   present state, and a model could give its own density as a measurement model gives
   `logLikelihood(g)`. Neither is built.
3. **What is the step called, and what does it take?** Proposed: `step(dt, measurement, rng,
   inputs...)`, returning `update()`'s result. The alternatives were a proposal given to
   `predict()`, which would then return an update's result, and a `proposals` namespace of policies.
4. **Does the measurement's method choose the proposal?** Proposed: yes, as above. The cost is that
   a matrix model cannot ask `step()` for the bootstrap proposal, for which `predict()` and
   `update()` remain, and that a method given to `update()` still does nothing.
5. **How does the auxiliary step resample?** Proposed: `auxiliaryStep()` runs the upkeep the filter
   holds, between its stages, and [upkeep] gains a sentence saying so. The alternatives were a
   resampler given to the call, and a wrapper on the measurement. A held upkeep that roughens moves
   the particles between the two look-ahead likelihoods, which then differ a little.
6. **Who keeps what a move needs?** Proposed: the filter, at every `step()`, one copy of the states
   and a weight factor per particle. A caller that never moves pays for the copy. The alternative
   is a record the caller keeps, as a smoother's history is kept [smoothing], which cannot follow
   an auxiliary step's resampling.
7. **Which moves?** Proposed: the independence sampler from the step's proposal, which needs no
   density. A random walk in the state would take `TransitionDensity`, and is not built.
8. **What do several measurements at one time do?** Proposed: `step()` with one, `update()` with
   the rest. Open: a proposal from all of them, and a `move()` given all of them.
9. **particle-filter.md Q11, a recursion for the particle filter.** Proposed: none, and
   [filters-and-recursions] says it covers the filters whose state is a component. The record a move
   reads belongs to the particles and follows them through a resampling, so a recursion kept apart
   from its state would have to hand the record over at every call.
10. **particle-filter.md Q4, Q5, Q10, and Q13.** Not answered, but each now has something to stand
    on. Q4: an auxiliary step has the look-ahead evidence before any particle moves, so a gate could
    refuse there. Q5: the filter keeps each particle's weight factor from the last step. Q10:
    `move()` separates copies without inflating the covariance, where a transition exists to
    propose from; a parameter held constant has none, and keeps `Roughen`. Q13: a guided step
    computes each particle's innovation and its covariance, so a per-particle NIS costs nothing
    there.
11. **KLD-sampling.** Proposed, not built: an upkeep action that resamples until the count meets
    Fox's bound, within the set's reserve. Its bins are over state coordinates, which
    [tag-addressing] wants named by tag path, the same question as step 17d's localization.
12. **The draws under step 18b.** A step's draws are one column of normals per particle, as a
    prediction's are, so a counter-based generator serves both the same way. A move adds a uniform
    per particle.
