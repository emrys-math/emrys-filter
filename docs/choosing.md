# Choosing a filter

"Which filter?" in [usage.md](usage.md) asks six questions and gives each one's default answer.
This page is the reasoning behind the other answers. Read the section for a question whose
default doesn't hold, and "Choosing a representation" at the end when the posterior can't be one
Gaussian. The questions are independent of one another: how nonlinear the process model is and
how nonlinear each measurement is are separate questions with the same ladder of answers, and the
library keeps them separate, since the prediction method is the filter's and each measurement names
its own. How far the posterior is from one Gaussian decides what represents the state. How the
noise is distributed, whether the motion changes regime, and whether the arithmetic can be
trusted are three more.

Every filter of the Kalman family replaces the output of a nonlinear function by a Gaussian with
an approximation of its moments, which describes it only while the function is nearly affine
across the distribution's spread. Each step down the ladder below either approximates the moments
better or stops forcing one Gaussian.

## How nonlinear is each model?

Judge the process model and each sensor separately, and judge each by how much its function bends
across the state's current spread, not by its formula: a range to a beacon is nearly affine from
10 km away over a 100 m prior, and strongly curved from 50 m away over the same prior. The
built-in process models are all affine, discretized exactly, so the process model raises this
question only when it has a part you wrote, such as a unicycle or a gyroscope turning an
attitude.

| The function is | The method | What its covariance claims |
|---|---|---|
| Affine | A measurement model made from its matrix, `measurementModel(H, reads(...), additiveNoise(noise))`; for a process model you wrote, `methods::linearized()` | Exact. |
| Nearly affine across the spread | `methods::linearized()`, which is the extended Kalman filter (EKF). The Jacobian is taken by forward-mode automatic differentiation, so the function must be generic in its scalar type, or the function returns it with `withJacobian()`. | The covariance of the problem linearized at the mean. It knows nothing of the curvature, so as the curvature grows the covariance understates the error before the mean goes wrong, and the NEES rises first. |
| A precise measurement of a curved function | `methods::iterated(methods::linearized())`, the iterated EKF, as the measurement's method | Relinearizes at the posterior mean until it stops moving, which is Gauss-Newton on the maximum a posteriori estimate, so the mean is where a nonlinear least-squares solve would put it. The covariance is still that of a single linearization, at the last iterate: better placed than the EKF's, but no more aware of the curvature. |
| Bent across the spread, or not differentiable | `methods::unscented()`, `methods::fourthMomentUnscented()`, or a sigma-point set: `methods::cubature()`, `sphericalSimplex()`, `gaussHermite()` | Moments taken from points spread across the distribution, so the covariance sees the curvature and the filter stays consistent further into nonlinearity than the EKF. No Jacobian is taken, so the function needs only to be callable on vectors, and may hold a table lookup or anything else that can't be differentiated. `unscented()`'s defaults put the points at $\sqrt{L}$ standard deviations over $L$ coordinates, far into the tails of a large state; `fourthMomentUnscented()` keeps them at $\sqrt 3$ whatever $L$ is, with a negative center weight above three coordinates. |
| A precise measurement of a strongly bent function | `methods::iterated(methods::unscented())`, or `iterated()` over any set, the iterated posterior linearization filter | Linearizes statistically with respect to the posterior rather than the prior, so the covariance includes the error of the linearization over the posterior's own spread. |

Beyond the ladder no single Gaussian describes the output, and the next question takes over. The
prediction method is the third argument of `KalmanFilter{model, prior, method}`, and a measurement's
method is the first argument of `observe(method, model, z)`, so one filter can predict
by `methods::unscented()`, take a position fix by its matrix, and take a range by
`methods::iterated(methods::unscented())`. `iterated()` is for measurements only: a prediction has
no posterior to relinearize at.
[unicycle.cpp](../examples/unicycle.cpp) runs the EKF and the unscented filter side by side on one
problem, which is the way to find where a problem sits: if the EKF's NEES fails and the unscented
filter's passes, the method was the problem; if both fail, the representation is, and the next
question applies.

## How far is the posterior from one Gaussian?

Every filter above approximates about one mean, so it needs a prior narrow enough that the
functions are nearly affine across it, or at least well sampled by its sigma points, and a
posterior that one Gaussian describes. Two situations break that, and no method on the ladder
repairs them:

- **A wide prior.** A state known only to lie in a region, such as a position anywhere in a 10 km
  box located by ranges or time differences of arrival, gives the functions no point to linearize
  at and the sigma points no spread they can cover. The first precise measurement leaves a
  posterior shaped like the measurement's level set, a ring or a strip, whose mean and covariance
  describe nothing on it.
- **Ambiguity.** Measurements that several states explain equally, such as two ranges that can't
  tell a position from its mirror image, or bearings from one platform, leave the posterior's
  weight in separate places.

A single-Gaussian filter started in either situation does not fail loudly: it may settle in the
wrong place with a small covariance, and no check without the truth sees it. The state then needs
a representation with more than one piece: a `GaussianMixtureFilter`, whose components each cover
a part small enough for the ladder above to hold in it; a `PointMassFilter`, a grid of fixed cells
over the whole state; a `ParticleFilter`; or `RbParticleFilter` and `RbPointMassFilter`, particles
or cells over the nonlinear block with a Kalman filter per point over the rest. "Choosing a
representation" below chooses among them. Each gives, besides its moments, a reading of whether
the moments describe it, which "When to trust an estimate" in [usage.md](usage.md) says how to
use.

## How is the noise distributed?

- **Gaussian**: every filter.
- **Heavy tails and outliers**: a Gaussian mixture as the sensor's noise, such as a narrow
  component with weight 0.95 and a wide one with 0.05. Every filter takes it: a Kalman-family
  filter reduces the branches by its policy, `reductions::collapse()` or `select()`, or keeps them
  as components in a `GaussianMixtureFilter`, and a `reductions::gate()` rejects an observation
  outright; the particle filter
  weights by the mixture's density exactly.
- **Outliers that come in runs**, such as a sensor that sticks or a bias that lasts several
  measurements: the same mixture with `.byMode<Sensor>()` after `additiveNoise()`, and an
  `ImmFilter` whose mode space has a `Sensor` axis with a `MarkovChain` of how long the outliers
  last. The filter learns from each measurement whether the sensor is in its outlier state. Any
  other filter reads that noise as the memoryless mixture of the item above, so one measurement
  model serves both.
- **Any other measurement error distribution**: a measurement model known only by its
  likelihood, `measurementModel(reads(...), logLikelihood(g))`, with `g(in, x, z)` the log of
  p(z | x) at a state point, and no function or noise, observed without a method. `ParticleFilter` weights its
  particles by it, and `PointMassFilter`, `RbPointMassFilter`, and `RbParticleFilter` weight their
  cells or particles by it when every block it reads lies in the gridded or sampled block, leaving
  each point's Gaussian over the rest as it was; a likelihood that reads the rest is a compile
  error, since given it the rest would no longer be Gaussian. A grid weights its cells exactly,
  however the likelihood is shaped, which makes `PointMassFilter` its natural home:
  [point_mass_filter.cpp](../examples/point_mass_filter.cpp) locates a tag by ranges whose
  error, where shelving blocks the path, is Gaussian noise convolved with an exponential delay.
  `StudentT`, `Uniform`, `Gamma`, `KernelDensity`, and the mixtures all have a `logDensity()` to
  write one from. A measurement model with Gaussian noise that also has a `logLikelihood()` is
  weighted by that likelihood in these filters. The Kalman-family filters can't take a likelihood:
  they condition on an innovation and its covariance, and refuse a model known by its likelihood
  alone at compile time.
- **Process noise** is Gaussian in every filter, the particle filter included: a process model's
  noise is a covariance, discretized over the step. Motion that is sometimes one thing and sometimes
  another, such as a maneuver, is modeled by regimes (next question), not by heavy-tailed process
  noise.

## Does the motion change regime?

- **One process model**, whatever its nonlinearity: the filters above.
- **Switching during the run**, where the motion differs qualitatively from time to time, such as
  a target that alternates between straight flight and turns, or a vehicle that stops and starts:
  `ImmFilter` over a `SwitchingModel`, one process model per mode, with a `MarkovChain` of the
  switching probabilities. Its estimate at each step mixes the modes by their probabilities, and
  a mode that was ruled out comes back when the chain lets it. A sensor that sees the state
  differently in some modes, with a bias or through another function, gives one measurement per
  value of a mode axis, `switchingMeasurement<Sensor>(nominal, biased)`, and each mode is updated by
  its own.
- **Several candidate models, one of which holds for the whole run**, such as a target that is
  one of several classes, or a sensor whose bias follows one of several laws: `AmmFilter`, the
  autonomous multiple model estimator, the same bank over the identity chain, so the mode
  probabilities converge on the model the measurements favor instead of mixing. Apply
  `upkeep::floorModes()` after each update so that no mode's probability falls to where it can never
  recover. If the model might change after all, use the IMM with small switching probabilities.
- **A discrete state only**, such as a mode, a count, or an arrival process with no continuous
  state: `HiddenMarkovFilter`.

## Can the arithmetic be trusted?

The covariance form, `KalmanFilter`, serves a `double` state with coordinates of comparable
scale. The same filter over another form of the Gaussian, with the same methods and models,
serves where it doesn't:

- `SqrtKalmanFilter` carries a factor of the covariance, which holds twice the precision and
  stays positive semidefinite where the covariance form's subtraction can lose definiteness.
  Use it in `float`, for coordinates of very different scales, for a measurement far more
  precise than the prior, and for long runs.
- `InfoFilter` carries the precision and information vector, so a prior with no information in
  some direction is a zero, not an enormous variance, and updates add, so many measurements
  between predictions cost little and their order doesn't matter.
- `SqrtInfoFilter` is both at once.

A `MixtureFilter` or an `ImmFilter` over square-root components runs the same way. The
information forms can't be components: a mixture adds densities, the information forms multiply
them, and their collapse fails on a component with no information in some direction, so a mixture
of them is refused at compile time with a message naming `toGaussian()` and `toSqrt()`.

## Estimate a parameter, consider it, or leave it out?

A model's uncertain parameter, such as a sensor's bias or scale factor, can be handled three ways:

- **Estimate it**, as a block of the state with its calibration uncertainty as its prior, when the
  measurements observe it and the model of how it changes is right. This is the default.
- **Consider it**, with `Emrys::consider()` in the measurements that should not move it, when it is
  observed so weakly, or modeled so roughly, that estimating it would let it absorb other errors, or
  when its value must stay the calibrated one. The filter stays consistent: its covariance carries
  the parameter's effect, so it is larger than the estimating filter's.
- **Leave it out** only when its effect is small against the noise. Otherwise the filter is
  overconfident: in [consider_bias.cpp](../examples/consider_bias.cpp), leaving out a gyroscope's
  bias gives an average NEES about 60 times the consistent one.

## After the run

- Estimates of past states from later data: the histories in `Smoothers/`, which record
  the forward pass of a Kalman, IMM, mixture, or particle filter and smooth it backward, fixed
  interval or fixed lag, and the most probable mode sequence by Viterbi.
- The best accuracy any filter could reach on the problem: `PosteriorCramerRaoBound`. A filter
  that is consistent but whose covariance stays far above the bound is paying for its
  approximation, and a step down the ladder may recover some of it.

## Choosing a representation

Where the posterior can't be one Gaussian, choose in two steps: first take out what is linear,
then choose how to carry the rest.

**Take out what is linear.** Find the blocks that the models are affine in, given the rest of the
state, and carry them exactly as Gaussians. Such blocks include a calibration bias, the delay on
one receiver of a time-difference-of-arrival pair, and a velocity under linear dynamics. Over a
grid or particles, that is `RbPointMassFilter` or `RbParticleFilter`, with a Kalman filter per
cell or particle over those blocks. Over Gaussian components, in a `GaussianMixtureFilter` or a
`KalmanFilter`, name those blocks in the models' `affineIn()`. Every sigma-point method then
spreads its points over the other coordinates only, and the named blocks pass through exactly. Either
way the linear blocks cost little, and the hard part of the problem shrinks to the coordinates
that really are nonlinear. A bias belongs in the state even when it seems small: one the
measurements can't tell from a shift in the state moves the estimate and no consistency test sees
it ("When to trust an estimate" in [usage.md](usage.md)).

**Carry the rest.** The sharpness of the measurements decides more than anything else. Mixture
components, and grid cells with a kernel, fill volume. Particles are points. A point is weighted
by the likelihood at that one point, so points must lie densely enough that many of them land
near the set of states consistent with each measurement, its isogram. The count needed grows like
the prior's extent over the measurement's width, raised to the dimension. That is the curse of
dimensionality, and it is acute when measurements are precise: a range accurate to 15 m, against
a prior 10 km across, leaves a strip that few particles of any practical set land on. Precise
measurements cost particles a second time: each resampling draws a new set from the old, which
adds variance of its own, so some of the precision a sharp measurement bought is lost at the next
resample. Coarse measurements, with a likelihood wide enough that many particles fall under it,
are where particles do well. A Gaussian component is updated over its whole volume, so one that
straddles the isogram moves onto it wherever it lies, and carries what a sharp measurement says
exactly. The components need only be small enough that the measurement is nearly linear across
each. The measurement function's curvature sets that scale, not its noise.
- A `GaussianMixtureFilter` suits a posterior that is near-Gaussian at some scale, at any
  dimension. Start it from `uniformComponents(box, counts)`, which covers a box with equal
  components at a spacing you choose. Then remove and merge components as the measurements
  resolve the state.
- A grid, `PointMassFilter` or `RbPointMassFilter`, covers a bounded region deterministically,
  at a resolution that `upkeep::subdivide()` refines and `upkeep::pruneBelow()` trims, and needs no resampling
  or other upkeep to stay healthy. Its cost is the number of cells, which grows exponentially in the
  dimension gridded, so grid a block of one to three dimensions. Its cells don't move, and
  `predict()` doesn't move them: the gridded block must be constant over the run, such as a
  receiver's position for a session, a sensor offset, or a bias, with any motion in the Gaussian
  part. A stationary state's grid can't recover probability it has pruned.
- A `ParticleFilter` suits coarse measurements, a posterior that is far from Gaussian at every
  scale, or measurement noise that isn't Gaussian, in a low dimension. The coarser the
  measurements, the better it does against the other two. Particles fill space poorly above two
  or three dimensions, so with the linear blocks taken out, `RbParticleFilter` keeps the sampled
  dimension low while the block it samples moves freely. Its weights come from Gaussian updates,
  or from a measurement's own likelihood when it reads only the sampled block.

**Keep it for the whole run.** A filter whose posterior has narrowed to one near-Gaussian region
can be left as it is. A mixture merges down to a few components and a grid prunes to a few cells,
so it costs little once converged. And it is still there when a gap in the measurements or a new
ambiguity spreads the posterior again. Handing off to a single Gaussian on convergence saves
little and can't be undone.

**Protect irreversible upkeep.** Pruning, merging, and resampling discard components and particles
that later measurements may need. One outlier while the posterior is still spread can move the
weight off the truth, and a grid or the particles of a stationary state can't move it back. Before
any upkeep that discards, gate each measurement: use a `reductions::gate()` where the filter takes a
policy, or test the measurement against the predicted measurement yourself. Alternatively, give
the sensor mixture noise. Set the thresholds for the worst gap and outlier you expect, not for
the nominal run.
