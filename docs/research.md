# Open problems

EmrysFilter implements algorithms from the public literature and nothing of its own
([published-algorithms-only] in [design/decisions.md](design/decisions.md)). Building it keeps
turning up needs that no published method fills, or that a method fills only without the guarantee
the library would want before relying on it: a bound, an error analysis, or a proof that an
approximation is what it claims. Rather than invent something and ship it, the library records the
need here and leaves the hooks in place.

Each entry is scoped for a graduate student looking for a self-contained publication: address
the need, prototype it against the hooks, prove something, publish. Once it is published, the
library implements it and cites it. To take one, open a **Research project** issue
([CONTRIBUTING.md](../CONTRIBUTING.md)) so that two people don't start the same problem, and ask
there about the hooks. The citations under each entry are starting points found while the need
came up, not a literature review; check them against the sources before building on them, as
the library's own [bibliography](reference/bibliography.md) requires.

Entries come from three places: a design decision that defers something for lack of a reference
[splitting-hooks], the open questions at the end of `decisions.md`, and reviews of a plan against
the literature, like the retrospective outlier rejection that produced the first two entries and the
merge thresholds that produced the eleventh. When an entry is published and implemented, move it to
the "Resolved" list at the end with the citation.

## 1. Retrospective outlier rejection for nonlinear models

**Need.** The smoother histories of [smoothing] are to gain a leave-one-out test on each recorded
measurement and a rejection step that deletes an outlier and re-solves. For a linear-Gaussian
model the test is the standardized smoothed observation disturbance of Harvey and Koopman (1992)
and de Jong and Penzer (1998), and the re-solve is the Kalman smoother on the reduced data. For a
nonlinear model the history holds only the forward pass's statistical linearizations
($H = \Sigma_{xy}^T P^{-1}$ and the residual noise $R + \Omega$, from `statisticalLinearization()`),
not the measurements, so it can test at those linearizations and replay with them frozen, but it
cannot relinearize without the models and the measurement objects.

**Open.** The error of the frozen-linearization replay against the relinearized re-solve, which
Bell (1994) identifies as Gauss-Newton on the smoothing cost. Under what conditions on the
outlier's size and the model's curvature is one frozen replay within a stated fraction of the
relinearized answer, and does the leave-one-out statistic computed at the contaminated
linearization keep its chi-squared calibration to a stated order? A cheap correction, such as
relinearizing only the steps whose displacement exceeds a threshold, would be a result too.

**Starting points.** Harvey and Koopman (1992); de Jong and Penzer (1998); Durbin and Koopman,
*Time Series Analysis by State Space Methods*, 2nd ed. (2012), disturbance smoothing and the
detection of outliers and structural breaks; Bell (1994), *SIAM J. Optim.* 4(3); Lefebvre,
Bruyninckx, and De Schutter (2002) for the statistical linearization the library uses.

**Hooks.** `GaussianHistory` and `smoothStep()` in `Smoothers/GaussianHistory.h`, `PredictedJoint`,
`statisticalLinearization()`, and the planned per-update record (roadmap step 14).

## 2. A calibrated leave-one-out test for grid and mixture posteriors

**Need.** For a point-mass or hidden Markov state the leave-one-out posterior at time $k$ is the
smoothed posterior with the $k$-th likelihood factor divided out, so the leave-one-out predictive
density of each measurement, the conditional predictive ordinate of Geisser and Eddy (1979), is
an $O(K)$ computation from the recorded log-likelihood vector. The library can report it. What it
can't do is turn it into an accept-or-reject decision at a stated probability, as the Gaussian
chi-squared gate does, because under a mixture predictive the ordinate's tail has no closed form.

**Open.** A calibrated test for a measurement against a Gaussian-mixture or grid predictive at a
given probability, cheap enough to run per measurement: the tail probability of the predictive
density, a bound on it, or a transformation with a known reference distribution. A
posterior-weighted average of per-component chi-squared tails is the obvious candidate and has
no justification yet; is it conservative, and by how much?

**Starting points.** Geisser and Eddy (1979), *JASA* 74; Pettit and Smith (1985); Gelfand, Dey,
and Chang (1992) on the ordinate for outlier detection; Bar-Shalom, Li, and Kirubarajan (2001),
§2.3.2, for the gate it would generalize.

**Hooks.** `CategoricalHistory` in `Smoothers/CategoricalHistory.h`, `PointMassFilter`, the `Gate` policy in
`Models/Reduction.h`, and `chiSquaredQuantile()` in `Core/ChiSquared.h`.

## 3. Smoothing a Gaussian mixture through collapses and merges

**Need.** [smoothing]'s mixture smoother follows Rahmathullah, Svensson, and Svensson (2014): the
mixture keeps each component's share from each component at the last prediction through every merge,
and the backward pass smooths every (child, parent) pair and merges the pairs that share a parent.
With `Keep` and no merging it is exact in each component's moments. Through merges it is not, and
the paper's evidence for it is simulation (track loss, RMSE, NEES) on one tracking scenario, with no
bound on the error the forward merges and the backward collapse add.

**Open.** First, the error of the smoothed mixture through merges relative to the exact
Gaussian-sum smoother, as a function of the merge cost Runnalls (2007) bounds going forward. Second,
whether the backward collapse of each parent's pairs to one Gaussian, which the library takes for a
bounded and aligned result, loses much against the paper's threshold merging within the group.
Third, the same backward pass for a mixture of IMMs [filters-and-recursions], where a merge mixes
modes.

**Starting points.** Rahmathullah, Svensson, and Svensson (2014) and Rahmathullah's 2014
licentiate thesis; Kitagawa (1994), *Ann. Inst. Statist. Math.* 46(4), for the two-filter
Gaussian-sum smoother; Runnalls (2007) for the merge cost; Särkkä (2013), chapter 10, for the
Gaussian smoother the per-pair step reuses.

**Hooks.** `GaussianMixtureHistory` in `Smoothers/GaussianMixtureHistory.h`, `smoothStep()` in
`Smoothers/GaussianHistory.h`, `MixtureLineage`
and `MixtureParams::merged()` and `merge()`, `ComponentTraits<GaussianParams>::collapse()`.

## 4. The manifold smoother with covariance resets as a Gauss-Newton step

**Need.** [covariance-reset] moves every covariance to the tangent space of the mean it is combined
at, by each block's reset Jacobian, in the filters and in the smoother. The filter-side reset is in
the error-state literature; the smoother-side use, carrying $P_{k+1 \mid N}$ back to the predicted
mean by the inverse reset and the result forward to the smoothed mean, follows the same rule but is
not written down anywhere the library can cite.

**Open.** Show that the tangent-space RTS pass with both resets is one Gauss-Newton iteration on
the manifold smoothing problem, in the sense of Bell (1994), and characterize what it misses
relative to the converged iterate. A corollary would say when a second backward pass is worth
its cost on a unit quaternion or a pose.

**Starting points.** Bell (1994); Hertzberg, Wagner, Frese, and Schröder (2013) and Solà, Deray,
and Atchuthan (2018) for $\boxplus$ and $\boxminus$; Mueller, Hehn, and D'Andrea (2017) and
Markley (2003) for the reset; Rauch, Tung, and Striebel (1965).

**Hooks.** `smoothStep()`, `internal::resetCovarianceAt()` in `Spaces/Reset.h`, the quaternion
smoother test in `tests/unequal_dimensions_filters.cpp`.

## 5. Student-t measurement updates for the Kalman family

**Need.** The roadmap defers Student-t noise for Kalman-family filters because the update isn't
conjugate; the library's robust model today is the finite mixture of [additive-noise], exact per
component. The literature offers variational updates and iteratively reweighted ones, and the
library's update structure, a residual noise $R + \Omega$ handed to a Kalman conditioning, makes
reweighting $R$ the natural fit.

**Open.** Which approximation fits an update that must run once, allocate nothing on a bounded
space, and compose with the existing reduction policies and smoothers, and what it loses against
the variational posterior. A comparison on the library's own outlier tests, with a bound on the
reweighted update's error as a function of the degrees of freedom, would settle the choice.

**Starting points.** Piché, Särkkä, and Hartikainen (2012), MLSP; Agamennoni, Nieto, and Nebot
(2012), *IEEE Trans. Signal Process.* 60(10); Roth, Özkan, and Gustafsson (2013), ICASSP;
Huang, Zhang, Li, Wu, and Chambers (2017), *IEEE Trans. Aerosp. Electron. Syst.*

**Hooks.** `AdditiveNoise` in `Models/Sensor.h`, `update()` in `Models/Measurement.h`,
`StudentT` in `Distributions/StudentT.h`, the injected-outlier tests in
`tests/Filters/kalman_filter.cpp`.

## 6. Splitting a mixture component by nonlinearity

**Need.** A criterion for whether to split a component before a nonlinear step, and a
direction to split along. Both below are the maintainer's own reasoning, so the action stays
outside the library and only the hooks are inside [splitting-hooks].

*Whether.* With noise passed through the function, or additive noise kept apart, the $\Omega$ of
`statisticalLinearization()` over one component, less the noise, is the part of the output that
the component's best affine fit misses. It is zero when the step is affine over the component's
spread. Weighed against the noise the step's result is compared with, as
$\operatorname{tr}(R^{-1}\Omega)$ or $\log\det(I + R^{-1}\Omega)$, it says how much splitting would
help. It costs nothing beyond the step's own transform only when the decision lags a step behind
or the action wraps the step, transforming each component and keeping the result when it does not
split; an action run before the step transforms each component once more. It needs a method that
matches the fourth moment, since the third-degree rules and `linearized()` see no curvature
(`statisticalLinearization()` in `Models/Transform.h`).

*Which direction.* Not the component's principal axis, which ignores the model, and not the
sigma-point column with the largest second difference: the columns come from a factor whose order
is arbitrary, and $2n+1$ points see only the Hessian's diagonal in that basis. Write the input as
$x = \mu + L u$ for any $L$ with $LL^T = P$, scale the output by $R^{-1/2}$, and form each output's
Hessian in $u$, $G_k = L^T \nabla^2 f_k \, L$. The second-order term the affine fit misses along a
unit direction $v$ is $\tfrac12 (v^T G_k v)_k$, so the direction maximizes $\sum_k (v^T G_k v)^2$.
That is not an eigenproblem: the top eigenvector of $\sum_k G_k^2$ is a starting point, and
power-style iterations refine it, by products $G_k v$. The split direction in the input is $Lv$, the
same for any $L$.

*A direction known up front.* When the model says which way to split, as a range-bearing sensor
does, the transform can spread its points by `along(d)`, and its first second difference
$\tfrac12 [f(\mu + c\,s_1) + f(\mu - c\,s_1)] - f(\mu)$ measures the nonlinearity along $d$ across
the spread, from the step's own points. The objection above to second differences is to an
arbitrary factor's columns, which `along(d)` replaces [splitting-hooks].

*The split.* One-dimensional along $Lv$, by a precomputed univariate split of a standard Gaussian
into weights, offsets, and a shrunken variance, into the component's place through
`GaussianMixtureParams::split()`.

**Open.** Publish the criterion and the direction, or better ones: show what the direction
optimizes, how it relates to the eigenvector of $\sum_k G_k^2$ that starts the iteration, and how
the split's error compares with the principal-axis and sigma-point rules on the library's
benchmark models. A univariate split table with a stated optimality would be part of it, and so
would what to weigh $\Omega$ against before a prediction, which has no $R$: the process noise, or
the component's own predicted covariance.

**Starting points.** Alspach and Sorenson (1972); Huber (2011), "Adaptive Gaussian mixture
filter based on statistical linearization", FUSION; Faubel, McDonough, and Klakow (2009), *IEEE
Signal Process. Lett.* 16(9); DeMars, Bishop, and Jah (2013), *J. Guid. Control Dyn.* 36(4);
Lefebvre, Bruyninckx, and De Schutter (2002); Raitoharju and Ali-Löytty (2012), "An adaptive
derivative free method for Bayesian posterior approximation", *IEEE Signal Process. Lett.*,
which splits along a direction of maximum nonlinearity; Raitoharju, Ali-Löytty, and Piché (2015),
"Binomial Gaussian mixture filter", *EURASIP J. Adv. Signal Process.* 2015:36, whose split takes
binomial weights.

**Hooks.** `statisticalLinearization()`, `GaussianMixtureParams::split()`, the upkeep triggers of
[upkeep], and, once written, the second differences along a sigma-point transform's axes
(`withSecondDifferences()`), the Hessians of a transform's function at the mean and their product
with a direction (by nested forward-mode autodiff, a user-supplied Hessian returned with the value,
the differenced per-point Jacobians of the factored transform [affine-coordinates], or about $n^2$ evaluations), and a split
of a Gaussian along a given direction.

## 7. Selective grid refinement and coarsening

**Need.** The point-mass filters of [rb-point-mass] keep fixed cells; [rao-blackwellized-core]'s
open question is refining a grid selectively. Splitting chosen cells needs per-cell covariances
carried through the split, and keeping the parents would let the children's spread reconsider the
split later. Replaying past measurements against new cells needs the measurement history, which the
library doesn't keep.

**Open.** A refinement rule for a point-mass grid with per-cell Gaussians (a Rao-Blackwellized
grid), with the error of sharing a parent's probability among children against replaying the
measurements, and a coarsening rule that is its inverse. The same shape as splitting a mixture
component, item 6, with the cells' positions fixed.

**Starting points.** Bucy and Senne (1971); Bergman (1999); Šimandl, Královec, and Söderström
(2006), *Automatica* 42(7).

**Hooks.** `PointGrid`, `GridGaussianParams`, `RbPointMassFilter`, and the upkeep
of `Distributions/Upkeep.h`.

## 8. Backward simulation with singular process noise

**Need.** Particle smoothing by backward simulation (Godsill, Doucet, and West, 2004) needs the
transition log-density $\log p(x_{k+1} \mid x_k)$. A process model here gives `propagate()` and
`noise(dt)`, from which the density follows when $Q$ is nonsingular. The common models driven by
acceleration noise alone have a singular $Q$: the density lives on the range of $Q$ at the
propagated point, and particles off it have zero density, so plain backward simulation
degenerates.

**Open.** A backward-simulation smoother for a model with singular $Q$ that needs only what the
model exposes, with its variance against the nonsingular case, and the interface the model would
have to add if one is unavoidable: the density on the range of $Q$, or a model-supplied
transition density. The same interface would serve a Rao-Blackwellized smoother.

**Starting points.** Godsill, Doucet, and West (2004), *JASA* 99(465); Lindsten and Schön
(2013), *Foundations and Trends in Machine Learning* 6(1); Lindsten, Bunch, Särkkä, Schön, and
Godsill (2016), *IEEE J. Sel. Topics Signal Process.* 10(2); Särkkä, Bunch, and Godsill (2012).

**Hooks.** `ParticleSet::ancestors()`, `ProcessModel`'s `propagate()` and `noise(dt)`, the
particle smoother of roadmap step 14.

## 9. Checkpointed smoothing from recorded updates

**Need.** A fixed-interval history stores a predicted joint per step, about $3n^2$ scalars, and
for a mixture that per component. Checkpointing in the adjoint-computation sense (Griewank and
Walther, 2000) would keep the state every $C$ steps and recompute each segment during the
backward pass, trading memory for a second forward pass. Recomputing a prediction exactly needs
the model and the step's inputs; recomputing it from the small per-update records of item 1 is
a frozen linearization.

**Open.** The optimal checkpoint schedule when a segment can be recomputed either exactly, at the
model's cost, or approximately from records, at a lower cost and a stated error, and whether the
revolve schedules remain optimal under that choice. An error bound for the recomputed smoothed
estimates in the approximate case.

**Starting points.** Griewank and Walther (2000), *ACM Trans. Math. Softw.* 26(1); Griewank
(1992) for the logarithmic-growth result; Särkkä (2013), chapter 10.

**Hooks.** `GaussianHistory` and `HybridGaussianHistory`, `predictJoint()`, `FixedStepModel`.

## 10. Highest-density regions for mixtures and kernel densities

**Need.** `Gaussian::confidence()` and `contains()` are exact because a Gaussian's
density depends only on the Mahalanobis distance. A mixture or a kernel density has no closed
form. The roadmap's deferred plan estimates the highest-density threshold from the
$1 - p$ quantile of the log-densities of its own draws.

**Open.** The coverage error of that estimate as a function of the draw count and the dimension,
and a cheaper estimator for the common case of a few well-separated components, with the
condition under which per-component ellipsoids are conservative.

**Starting points.** Hyndman (1996), *The American Statistician* 50(2); Silverman (1986) for the
kernel density.

**Hooks.** `GaussianMixture`, `KernelDensity`, `drawSamplesWithLogDensity()`.

## 11. What a merge costs, and a threshold that follows from it

**Need.** A caller merges mixture components by a cost such as Runnalls' (2007), the weighted
Kullback-Leibler divergence of two components from their moment-matched merge, which
`MixtureParams::merged()` and `kullbackLeibler()` compute, as `examples/gaussian_mixture.cpp`
does until the mixture is down to a count; `Divergence.h` offers the closed-form Kullback-Leibler,
Bhattacharyya, Hellinger, and Bures-Wasserstein distances between Gaussians, the last with a
ground metric. What nobody can say is how many nats a merge may cost. Runnalls' cost bounds the
divergence from the original mixture to the merged one, and Pinsker's inequality turns a
divergence of $\varepsilon$ into a worst-case error of $\sqrt{\varepsilon / 2}$ on any
probability the mixture is used to compute, which is concrete but loose, and it is one step: it
says nothing about what the merge does after the next update.

**Open.** A merge cost with an operational meaning, and the threshold that follows from a stated
tolerance. Two downstreams are candidates. To the inference, the cost is the Kullback-Leibler
divergence between the predictive distributions of the next measurement with and without the
merge, the expected log-evidence lost, which is measured through the measurement model (so a
difference the sensor can't see is free) and takes the weights exactly; it is not bounded by the
state-space divergence, because conditioning on a measurement is not a Markov kernel on the state
and can amplify the difference. To the consumer of the estimate, it is the change in an expected
loss, which Wasserstein with the loss's ground metric bounds through its Lipschitz constant. In
both, a weight is a posterior the next likelihood will rescale, so a light, distant component may
be the one the next measurement confirms, and merging it is irreversible; the cost should be an
expectation over the predictive of the next measurement or the next few, which a sigma-point
quadrature over the predictive could approximate at one transform per candidate pair. Results
wanted: the factor by which the cheap one-step costs bound that expected cost; the
weight-aware Wasserstein merge cost that the transport restricted to Gaussian-mixture couplings
gives when the target is a single component, the weight of each component times its Bures
distance squared to the merge, which has Runnalls' shape with $W_2^2$ in place of the
divergence and is an upper bound on the true $W_2^2$; and a rule for the threshold in either
cost from a tolerance on evidence lost or on the downstream loss.

**Starting points.** Runnalls (2007); Salmond (1990), SPIE 1305, for the joining criterion;
Williams and Maybeck (2003), *Mathematical and Computer Modelling* 43, for the integral squared
error cost; Crouse, Willett, Pattipati, and Svensson (2011), FUSION, for the survey of reduction
algorithms; Ardeshiri, Granström, Özkan, and Orguner (2015), *IEEE Signal Process. Lett.* 22(6);
Assa and Plataniotis (2018), *IEEE Signal Process. Lett.* 25(10), for Wasserstein-based
reduction; Delon and Desolneux (2020), *SIAM J. Imaging Sci.* 13(2), and Chen, Georgiou, and
Tannenbaum (2019), *IEEE Access* 7, for transport between Gaussian mixtures; Kass and Raftery
(1995), *JASA* 90(430), for a scale of what a nat of evidence is worth.

**Hooks.** `GaussianMixtureParams::merged()` and `merge()`, the closed forms in
`Distributions/Divergence.h` with `GroundMetric`, `ComponentTraits<GaussianParams>::collapse()`, the measurement
transforms with `withCross()` for a predictive distribution, and the mixture filter's
injected-outlier and merging tests in `tests/Filters/gaussian_mixture_filter.cpp`.

## 12. Learned models under a Gaussian input: the cross-covariance and the consistency

**Need.** A learned model is a point function to the library already: the unscented methods
propagate it with no derivatives, and Linearized takes the Jacobian it returns [returned-jacobians],
which is what a framework's backward pass gives. A heteroscedastic model, with a mean and a noise
scale that depend on the input, is an innovation with `InputNoise`, and a learned process noise is
an uncertain input [prediction-inputs]. What the library has no method for is a model that knows its
own output moments under a Gaussian input. For a Gaussian process with a squared-exponential or
polynomial kernel they are closed forms, mean, covariance, and the input-output cross-covariance,
which is settled. For a neural network, layer-wise moment propagation (assumed density filtering
through the units) is published but is an approximation stacked per layer with no error bound.

**Open.** Two things the uncertainty-quantification literature does not report because it scores a
model by the predictive likelihood of its outputs. First, the cross-covariance between input and
output, from which the Kalman gain is built: moment propagation through layers drops it unless it is
carried deliberately, and the unscented transform supplies it to second order. Which method gives it
at what accuracy for a network of a given depth and activation, against the exact value by Monte
Carlo? Second, consistency: when a learned model's epistemic uncertainty enters the prediction as
process noise, does the filter's normalized estimation error squared stay within its chi-squared
bounds, and how must the model's uncertainty be calibrated for it to? The library-side question is
the hook: a transform method where the function supplies its own moments and cross-covariance under
a Gaussian input, as the [returned-jacobians] hook lets it supply its Jacobian, so that an analytic
Gaussian process and a moment-propagating network use the same filters.

**Starting points.** Girard, Rasmussen, Quiñonero-Candela, and Murray-Smith (2003), NIPS 15, and
Quiñonero-Candela, Girard, and Rasmussen (2003), Technical University of Denmark report, for the
Gaussian process moments; Ko and Fox (2009), *Autonomous Robots* 27(1), and Deisenroth, Huber, and
Hanebeck (2009), ICML, for filters built on them; Frey and Hinton (1999), *Neural Computation*
11(1), Gast and Roth (2018), CVPR, and Wu, Nowozin, Meeds, Lakshminarayanan, Turner, and
Hernández-Lobato (2019), ICLR, for moment propagation through network layers; Bar-Shalom, Li, and
Kirubarajan (2001), §5.4, for the consistency tests.

**Hooks.** `transform()` and its methods in `Models/Transform.h` and `Models/Methods.h`, `withJacobian()`,
`OutputWithCross`, `InputNoise`, the uncertain inputs of `predict()`, `NisMonitor` in
`Models/Consistency.h` and the NEES check the roadmap plans beside it, and the examples planned
under Documentation in the roadmap.

## Needs without a research question

Some gaps are implementation, with the literature already settled; they belong on the roadmap,
not here, and a pull request citing the source is welcome: a Rao-Blackwellized particle filter
that samples a switching model's mode sequence (Chen and Liu, 2000; Doucet, Gordon, and
Krishnamurthy, 2001); the information filter and square-root filters (roadmap steps 16 and 17);
an inverse regularized incomplete gamma for Eigen; several resizable blocks in one space.

## Resolved

None yet. An entry moves here with its citation when the library implements it.
