# Lie groups and the invariant EKF

How the library represents Lie groups, what an invariant extended Kalman filter needs from it, and
what is still to decide. This is the plan behind roadmap step 17zf. Nothing in it is built. Where it
disagrees with step 17zf or with [lie-groups], [covariance-reset], [model-placement], and
[unit-quaternion] in [decisions.md](decisions.md), this plan is the newer one; those decisions
change in the commits that build it. Status and ordering are in [roadmap.md](roadmap.md).

The plan was worked out on 2026-11-18 and 2026-11-19. Two scratch programs back its claims;
"Evidence" below gives their setups and numbers. Neither program is in the repository.

## Where we are

`UnitQuaternion` is the library's only space with a $\boxplus$ of its own [unit-quaternion]. A pose
today is a `Composite` of a `UnitQuaternion` and `Linear<3>` blocks, which is a correct filter but
not the invariant EKF (Barrau and Bonnabel 2017). Step 17zf and [lie-groups] planned $SE(3)$ and
$SE_2(3)$ as spaces "each with the side its increment is applied on".

Three findings changed that plan:

- **The side of the increment does not change the filter when the covariance is reset.** A filter
  with the increment on the right and one with it on the left give the same means, and covariances
  related exactly by the adjoint, under `Linearized` and a converged `Iterated`. So a side parameter
  on a space would be a choice that changes no estimate.
- **Without the reset the side does matter, and the two published invariant filters omit the
  reset.** In one set of coordinates the left-invariant filter, the library's filter, and the
  right-invariant filter differ only in the matrix that carries the covariance after an update.
  The side therefore belongs to the update, as a policy, not to the space.
- **The filter the roadmap names as the main use needs a group of run-time size.** Contact-aided
  estimation (Hartley et al. 2020) keeps one vector per foot in contact inside the group, and
  right-invariant SLAM keeps the landmarks there. Two fixed types, $SE(3)$ and $SE_2(3)$, cannot
  express either.

## Spaces and groups

A **space** is a set of values where a value can be moved by a small difference,
$x \boxplus \delta$, and two values have a difference, $a \boxminus b$ [difference-spaces]. That is
all the `Space` concept asks for.

A **group** is a space where two values also combine into a third (`compose`), every value has an
inverse, and one value is the identity. In a Lie group the product determines the other operations:

$$x \boxplus \delta = x \cdot \mathrm{Exp}(\delta), \qquad
  a \boxminus b = \mathrm{Log}(b^{-1} \cdot a).$$

"Space" stays the word every user meets. "Group" names the added capability, as a concept
`LieGroup` that refines `Space`.

| Space | A group? | Product |
| --- | --- | --- |
| `Linear` | yes | addition |
| `Periodic` | yes | addition modulo the period |
| `UnitQuaternion` | yes | the quaternion product |
| `Composite`, `Repeated` | when every block is | block by block |
| the group family below | yes | a rotation acting on vectors |
| `Finite`, `Named`, `DiscreteProduct` | no | sets of labels |
| a user's space | not necessarily | a unit direction vector cannot be one |

Two points follow from the table.

- **Not every space is a group.** The 2-sphere has no continuous group operation, and a unit
  direction (a gravity direction, a bearing) is a common state. It still has a $\boxplus$ and a
  $\boxminus$, so `Space` stays the broader concept.
- **The same values can form different groups.** `Composite<Attitude, Position>` and a pose hold
  the same seven numbers. The composite adds a position increment directly; the pose rotates it by
  the attitude first. The invariant EKF is the library's EKF on the second.

`LieGroup` is used by the library, not only by model authors: the adjoint carry below calls
`adjoint` and `exp`. Its members are `identity`, `compose`, `inverse`, `exp`, `log`, `adjoint`, and
the right and left Jacobians; a group that moves points, as a rotation or a pose does, also has the
action of a value on a point. A group whose size is known only at
run time keeps its count in the space instance, so the members belong to the instance and are
reached through free functions, as `plus()` and `minus()` are. They read and write plain Eigen
coefficient vectors. `UnitQuaternion` keeps `map()` for Eigen's quaternion type; no other group has
a typed view, because Eigen has no type for one.

## One side

Every group puts its increment on the right, $X = \hat{X} \cdot \mathrm{Exp}(\xi)$, as
`UnitQuaternion` does today. The differences are then steps in the body frame. In the papers'
vocabulary this is the *left-invariant* error $\hat{X}^{-1} X$; the names are crossed, and this
document says "right side" and "body frame" for the library and keeps "left-invariant" and
"right-invariant" for the published filters.

There is no side parameter on any space, for two reasons.

- With the reset, the choice changes no estimate. The two error coordinates are related exactly,
  $\mathrm{Exp}(\mathrm{Ad}_{\hat{X}} \xi) \cdot \hat{X} = \hat{X} \cdot \mathrm{Exp}(\xi)$, and the
  reset preserves the relation $P_\text{world} = \mathrm{Ad}_{\hat{X}} P_\text{body}
  \mathrm{Ad}_{\hat{X}}^T$ through every step.
- A parameter that changes only the frame the covariance and the process noise are expressed in
  would be chosen wrongly without any visible error.

A user who wants the covariance in the world frame maps it by the adjoint at the mean.

Sigma-point methods are the exception to "changes no estimate". Under `Unscented` and `Cubature`
the two sides differ, by up to 0.9 rad in the first step at a prior of 1.2 rad standard deviation
and by $8 \times 10^{-7}$ rad at a prior 50 times narrower. Neither side is more correct: the points
are spread by a different square root of the same covariance. The documentation of those methods
should say that their results depend on the difference coordinates when the spread is wide.

## The covariance carry

An update computes the posterior covariance in the tangent space at the prior mean and moves the
mean by a step $\delta$. What it then does to the covariance is $P^+ \leftarrow G P^+ G^T$, and
three choices of $G$ give three filters on the same right-side space:

| Carry | $G$ | Filter |
| --- | --- | --- |
| none | $I$ | the published left-invariant EKF |
| reset | the right Jacobian $J_r(\delta) = \sum_k (-\mathrm{ad}_\delta)^k / (k+1)!$ | the library today [covariance-reset]; the same filter on either side |
| adjoint | $\mathrm{Ad}(\mathrm{Exp}(-\delta)) = \sum_k (-\mathrm{ad}_\delta)^k / k!$ | the published right-invariant EKF, with its covariance in body coordinates |

To first order in $\delta$ the reset is halfway between the two published filters.

The carry is a policy chosen per measurement, by an argument of `observe()` as the `updates::` forms
are [covariance-update-forms], because which carry suits depends on the measurement:

| Measurement | Form | Carry that suits it |
| --- | --- | --- |
| something fixed in the body, measured in the world: a position fix, a velocity fix | $y = X \cdot b$ | none |
| something fixed in the world, measured from the body: a landmark, a magnetic field, a gravity direction, a foot through the leg's kinematics | $y = X^{-1} \cdot b$ | adjoint |
| anything else, and any space that is not a group | | reset |

Each published filter pairs a measurement form with the error coordinates in which the
measurement's Jacobian and the covariance recursion do not depend on the estimate, apart from a
rotation of the measurement noise. A reset makes the covariance depend on the innovation, which is
the likely reason the papers omit it. It is not proved here that this explains the numbers in
"Evidence".

- **The default stays the reset.** It needs no knowledge of the measurement and is the only one
  defined for a space that is not a group. In the runs it was between the other two, except in the
  mean with landmarks at the widest prior, where a few runs dominate and its median was still
  between.
- **The adjoint carry on a space that is not a `LieGroup` is a compile error** that names the block
  that is not a group.
- **Biases are unaffected.** A `Composite` of a group and `Linear` bias blocks carries block by
  block, and a `Linear` block's adjoint is the identity. This is what the published filters with
  biases do.
- **Prediction needs no carry.** A prediction's covariance is computed at the predicted mean.

## The group family

One family replaces the planned $SE(3)$ and $SE_2(3)$: a rotation part followed by any number of
vector parts that rotate with it. The parts are the user's tags, as in a `Composite`
[tag-addressing], and a `Repeated` may be among the vectors, so the size may be known only at run
time [several-resizable-blocks].

```cpp
struct Attitude : UnitQuaternion {};
struct Velocity : Linear<3> {};
struct Position : Linear<3> {};
struct Foot     : Linear<3> {};
struct Feet     : Repeated<Foot> { using Repeated::Repeated; };

struct Body : ExtendedPose<Attitude, Velocity, Position, Feet> {};   // SE_{2+n}(3), n at run time
using State = Composite<Body, GyroBias, AccelBias>;
```

`ExtendedPose` is a working name, the literature's term for $SE_k(3)$. With no vectors the family is
$SO(3)$, with one $SE(3)$, with two $SE_2(3)$, and with two plus the feet the group of Hartley et
al. (2020).

The type holds the same values as `Composite<Attitude, Velocity, Position, Feet>` and addresses them
by the same paths. It differs in the product, and so in its $\boxplus$, $\boxminus$, reset, and
adjoint. With $J$ the left Jacobian of the rotation:

$$(R, v_1, \dots, v_k) \boxplus (\phi, \rho_1, \dots, \rho_k) =
  (R \, \mathrm{Exp}(\phi), \; v_1 + R J(\phi) \rho_1, \; \dots, \; v_k + R J(\phi) \rho_k).$$

- **Run-time size is designed for from the start.** It is the hard case, and it decides the API:
  the members belong to the instance, the parts are the user's tags, and inserting and erasing a
  vector must be defined.
- **A separate block cannot stand in for a vector of the group.** The adjoint of the group has
  entries between the rotation's columns and each vector's rows, which a `Composite`'s
  block-by-block carry cannot produce, and in the group a vector's correction is coupled to the
  attitude's at second order. This is from the algebra; no run has checked it.
- **The planar case is the same template**, if the first part may be any rotation that acts on the
  vectors: a `Periodic` heading with `Linear<2>` vectors is $SE_k(2)$.

## Body-frame rows

Each part owns difference rows, as a block of a `Composite` does. On the right side, a step in one
part's rows alone moves only that part:

$$\hat{X} \cdot \mathrm{Exp}(0, \dots, \rho_i, \dots, 0) =
  (\hat{R}, \dots, \hat{v}_i + \hat{R} \rho_i, \dots).$$

So a vector part's rows are steps of that vector expressed in the body frame at the mean, and they
mean that everywhere in the API:

- `covariance<path<Body, Position>>()` is the covariance of the body-frame step. To first order the
  world-frame covariance is $\hat{R} P_{\rho\rho} \hat{R}^T$.
- Noise placed on a part is body-frame noise, which is what an accelerometer's and a gyro's is.
- A model that reads a part reads its values and differentiates over its rows, as on a `Composite`.
  Under `Linearized` this is exact, since the part's value depends at first order only on its own
  rows. Under a sigma-point method it drops only the coupling to parts the model did not read.
- Erasing a vector drops its rows, which is exact. Inserting one takes its covariance and
  cross-covariance in these rows.

A Gaussian over a part's own space, in the world frame, is never what the library returns under the
part's name. Open question 2 is what `marginal<>` returns in its place.

## Noise

- **Noise in the body frame is additive** on the group's rows, by `continuousNoise()` or
  `additiveNoise()` [continuous-time-noise].
- **Noise in another frame is an uncertain input** that the dynamics read [prediction-inputs], not a
  new kind of noise. World-frame noise read as $X' = \mathrm{Exp}(w) \cdot f(X)$ gave the
  estimate-dependent $\mathrm{Ad}(\hat{X}'^{-1}) \, Q \, \mathrm{Ad}(\hat{X}'^{-1})^T$ to
  $2 \times 10^{-14}$.
- **A defect to fix:** `additiveNoise(on<Body>(Q))` does not compile on a block whose values and
  differences differ in size. `placeNoise()` builds a `Product` of the placed parts, and `Product`
  requires equal sizes. Unplaced `additiveNoise(Q)` works, and so does placed `continuousNoise()`.

## Jacobians that do not depend on the estimate

On group-affine dynamics the error's Jacobian does not depend on the estimate, and a measurement of
either form in the carry table has a Jacobian that needs no differentiation. **Where that holds, the
library does not differentiate at each step** (decided 2026-11-19). Today it would: a function that
does not return its Jacobian is differentiated automatically at every call [returned-jacobians].
That gave the right numbers in "Evidence" and costs what the published filters avoid.

What takes the place of differentiation depends on the case. The Jacobians are in the library's
body-frame rows, with $c_i$ the weight of vector $i$ in $b$ and $b_0$ its part that only rotates:

| Case | The Jacobian depends on | The library |
| --- | --- | --- |
| a fixed step and no input | nothing | computes it once and keeps it [fixed-step-models] |
| a motion driven by an input, $X' = X \cdot \mathrm{Exp}(u \, \Delta t)$ | the input and the step | forms $\mathrm{Ad}(\mathrm{Exp}(-u \, \Delta t))$ from them |
| $y = X \cdot b$ | the rotation at the mean | forms $\hat{R} \, [-b_0^\wedge, \; c_1 I, \dots, c_k I]$ |
| $y = X^{-1} \cdot b$ | the predicted measurement | forms $[\hat{y}_0^\wedge, \; -c_1 I, \dots, -c_k I]$, with $\hat{y}_0 = \hat{R}^T b_0$ |

A Jacobian that is the same for the whole run exists only in the first row. An inertial
mechanization's is constant in the world frame, where it depends on gravity alone, and in body-frame
rows it depends on the gyro and accelerometer readings. The world frame is no cheaper: there the
process noise is $\mathrm{Ad}_{\hat{X}} \, Q \, \mathrm{Ad}_{\hat{X}}^T$ at every step (Hartley et
al. 2020), where in body-frame rows it is constant. Neither frame differentiates.

Three mechanisms are proposed, and none is decided:

- **Built-in models for the forms in the table**, which return their Jacobians in closed form
  [returned-jacobians]: the motion by a twist, and the two measurement forms. A measurement model
  of a known form also knows the carry that suits it and can default to that carry, as a model
  given its matrix defaults to `Linearized` [linear-sensors]. A user who names the form then never
  chooses a carry.
- **A user's group-affine model returns its Jacobian** with `withJacobian()`, built from the group's
  `adjoint` and Jacobians. The inertial mechanization is such a model, and it is EmrysNavigation's
  [lie-groups].
- **`affineIn()` extended to a group block**, to mean that $f(x \boxplus \delta) \boxminus f(x) =
  F \delta$ with $F$ independent of $x$. Group-affine dynamics satisfy this exactly, for a
  $\delta$ of any size (the log-linear property of Barrau and Bonnabel 2017). `FixedStepModel` then
  caches $F$ under its present conditions, and a sigma-point method factors the block out exactly.
  Today [affine-coordinates] treats every block that is not linear as nonlinear.

## Sigma-point methods on a group

Nothing has to be built for an unscented filter on a group: `Unscented`, `Cubature`, and the other
point sets [sigma-point-sets] run on a group space as they run on `UnitQuaternion` today. The
literature has the filter (Brossard, Bonnabel, and Condomines 2017; Brossard, Barrau, and Bonnabel
2020, "UKF-M"), and UKF-M's default uncertainty, $\chi = \hat\chi \exp(\xi)$, is the library's
right-side $\boxplus$. Four things follow.

- **The side matters again, and the carry does not cover it.** With the reset, the two sides give
  the same filter under `Linearized` and different ones under a sigma-point method ("Evidence").
  Since $\mathrm{Exp}(\zeta) \hat{X} = \hat{X} \, \mathrm{Exp}(\mathrm{Ad}_{\hat{X}}^{-1} \zeta)$,
  the world-side filter is, in body-frame rows, the same filter with its points spread by another
  square root of the same covariance, $\mathrm{Ad}_{\hat{X}}^{-1} \,
  \mathrm{chol}(\mathrm{Ad}_{\hat{X}} P \mathrm{Ad}_{\hat{X}}^T)$, and with the adjoint carry. So
  the world side needs the root as well as the carry, in a prediction as well as an update. UKF-M
  offers both sides and reports the world side ahead on its inertial navigation example, with no
  general rule. **A sigma-point method gets the world side** (decided 2026-11-19); how is open
  question 9.
- **The library's filter is not UKF-M step for step.** UKF-M propagates the mean through the model
  without noise and uses the points for the covariance alone; the library takes a tangent-space
  mean of the propagated points. UKF-M spreads separate point sets over the state and the noise;
  the library spreads one over both. UKF-M applies no carry after an update and says that the
  transport is left to the user; the library resets. The documentation must name the algorithm the
  library implements and say where it departs from these [published-algorithms-only].
- **On group-affine dynamics the points buy nothing in a prediction.** The map of the differences
  is exactly linear, so every point set returns the linearized result, at one evaluation per point.
  With `affineIn()` extended to a group block, a sigma-point method spends no points on it. What a
  sigma-point method adds is in the update, where $\mathrm{Exp}(\xi) \cdot b$ is not linear in
  $\xi$, and in the blocks beside the group, such as biases.
- **The matching side is a matter of the root too.** For $y = X \cdot b$ the points
  $\hat{X} \cdot \mathrm{Exp}(\xi_i) \cdot b$ are $\hat{X}$ acting on points that do not depend on
  the estimate, so the gain in the body frame does not depend on it, as in the invariant EKF. For
  $y = X^{-1} \cdot b$ the same holds only with the points spread on the world side. This is from
  the algebra; no run has checked it.

## Names

- **`compose(space, a, b)` is the group product**, as in GTSAM and manif, and as "composition"
  means in mathematics: one after the other.
- **The models' `compose()` becomes `blockwise()`**, and `Composition` becomes `Blockwise`.
  `blockwise(range, bearing)` measures `Composite<Range, Bearing>`; the word says how a `Composite`
  works, block by block. It combines the models it is given, however many, and
  one model alone is that model. It never means "this model on every block": that is `onEach`,
  which names the repeated block and also places noise and matrices. `onEach<Fleet>(vehicle)` is
  what `blockwise()` of one placement per repetition would be, for a count known at run time.
- **`UnitQuaternion::exponential()` and `logarithm()` become `exp` and `log`**, the names of the
  concept's members.

Rejected: `times()` for the group product, which is not the field's name and reads oddly on
`Linear`; `sideBySide()`, `stack()` (which may collide with `std::stack`), `join()`, `assemble()`,
and `composeModels()` for the models; `composeGroups()` and `composeSpaces()`, which would mean
building a product space, as `Composite` does.

## Evidence

**Sides (2026-11-18).** Attitude and gyro bias, `Composite<Attitude, Bias>`, with an accelerometer
and a magnetometer reading reference vectors in the body frame, 1000 steps. One run used
`UnitQuaternion`; the other a copy with $\boxplus = \mathrm{Exp}(\delta) \cdot q$, whose reset the
library found by autodiff. The motion was written as a quaternion product, the gyro noise was
isotropic, and the prior was mapped by the rotation at the mean. The numbers are the largest
difference of the mean attitude between the two runs.

| Method | Prior heading $\sigma$ 1.2 rad | Prior 50 times narrower |
| --- | --- | --- |
| `Linearized` | $2 \times 10^{-14}$ rad, covariance $2 \times 10^{-12}$ relative | $3 \times 10^{-14}$ rad |
| `Iterated`, default tolerance | $1 \times 10^{-5}$ rad | $3 \times 10^{-10}$ rad |
| `Iterated`, tolerance $10^{-13}$ | $4 \times 10^{-14}$ rad | $2 \times 10^{-14}$ rad |
| `Unscented` | 0.91 rad at step 0, $5 \times 10^{-4}$ rad at step 999 | $8 \times 10^{-7}$ rad |
| `Cubature` | 0.79 rad at step 0, $4 \times 10^{-4}$ rad at step 999 | $8 \times 10^{-7}$ rad |

The default `Iterated` tolerance is per coordinate in prior standard deviations, which differ
between the frames, so the two sides stop at different iterates.

**Carries (2026-11-19).** $SE(3)$ with a known body-frame twist, $X' = X \cdot \mathrm{Exp}(u \,
\Delta t) \cdot \mathrm{Exp}(w)$, and either a position fix or three landmarks seen from the body,
each with anisotropic noise. The library ran `Linearized` on a right-side $SE(3)$ space written for
the test, with its reset by autodiff or turned off by `identityReset`. Hand-written filters ran on
either side with and without a reset, and on the right side with the adjoint carry too. At full
width the prior is up to 1 rad in attitude and 3 m in position.

| Comparison | Largest difference, mean and covariance |
| --- | --- |
| library without reset, against the paper's matrix form (constant $F$, innovation $\hat{X}^{-1} y$, rotated noise) | $10^{-14}$ |
| library with reset, against the hand-written right-side filter with reset | $10^{-10}$, the accuracy of the hand filter's finite-difference Jacobian |
| right side against left side, both with reset | $10^{-10}$ |
| right side against left side, both without reset | 1.2 at full width, $4 \times 10^{-2}$ at 0.05 of it |
| right side with the adjoint carry, against left side without reset | $10^{-14}$ |

Normalized estimation error squared (NEES; 6 when the covariance matches the error), averaged over
the first five steps of 500 runs. Each cell gives the prior at 0.3, 0.5, and 1.0 of full width:

| Measurement | No carry | Reset | Adjoint carry |
| --- | --- | --- | --- |
| position fix | 6.1, 6.6, 17.5 | 6.8, 10.9, 57.7 | 8.6, 21.3, 164 |
| landmarks from the body | 9.1, 37.4, 2118 | 8.2, 25.8, 33706 | 7.8, 18.3, 498 |

At 0.1 of full width the three agree within 1%. With landmarks at full width a few runs dominate
the means; the medians are 82, 53, and 36, in the same order as at the narrower priors.

## What to find out first

The step is here because the invariant EKF is an established part of the literature, and the
literature's results are not confirmed again before building: that the filter does better than a
conventional EKF from a large initial error (Barrau and Bonnabel 2017), and that contact points and
landmarks belong inside the group (Hartley et al. 2020). The example and tests of step 7 show them.
What the literature does not settle is the library's own:

1. **The carries under `Iterated` and the sigma-point methods.** The published filters are
   linearized, and so is every run in "Evidence".
2. **A prototype of the family at run-time size**, with a vector inserted and erased.

## Steps

1. Decide open questions 1 to 3, which the family's API depends on.
2. `LieGroup`, satisfied by `Linear`, `Periodic`, `UnitQuaternion`, `Composite`, and `Repeated`,
   with a helper that checks a group's members against each other: `plus` against `compose` and
   `exp`, `adjoint` against `compose`, `exp`, and `log`, and each Jacobian against autodiff.
3. The group family, fixed-size and run-time size, with closed forms smooth at zero.
4. The carry policy, in every Kalman-family update, and the world side for the sigma-point
   methods.
5. `blockwise()` in place of the models' `compose()`, and the fix to placed `additiveNoise()`.
6. The models whose Jacobians need no differentiation, and a benchmark
   ([benchmarks.md](benchmarks.md)) of the invariant filter's step against a hand-written one,
   since "does not differentiate at each step" is a claim about cost.
7. An example and tests: the two scratch programs as tests, the error Jacobian the same at every
   estimate on group-affine dynamics, and the group filter against a `Composite` from a large
   initial error.
8. The decisions: [lie-groups] loses the side; [covariance-reset] gains the carry;
   [model-placement] and [unit-quaternion] take the new names; [affine-coordinates] and
   [fixed-step-models] change if `affineIn()` is extended; a decision for the family.

## Open questions

1. **How do separate models write parts of one group?** The contact-aided filter is
   `blockwise(imu, RandomWalk<Feet>{q})`: one model writes the attitude, velocity, and position,
   and another the feet. A foot's rows are body-frame steps at the *new* attitude, which the other
   model writes, so a random walk on a foot has the Jacobian $\hat{R}'^T \hat{R}$ in the foot's
   rows, not the identity that `RandomWalk` assumes. The library has to assemble the group's values
   from all its models before it takes differences. This is required, not optional: it is the
   intended use.
2. **What is a distribution over a part?** `covariance<>()` of a part is settled above. What
   `marginal<>`, a conditional, and an `on<>()` factor return for a part is not: a Gaussian labeled
   with the part's own space would be read as a world-frame one.
3. **Inserting a vector as a function of the state.** A new foot's position comes from the pose and
   the leg's kinematics, so it is correlated with the pose. `insert<Path>()` takes a mean, a
   covariance, and a cross-covariance; nothing forms them from a model and its inputs.
4. **Where does the carry apply?** Every Kalman-family update, presumably; the smoother, a
   Gaussian's conditional, and the information forms' reference are undecided.
5. **Names:** the family, the carry policy and its three values, and the action of a value on a
   point.
6. **What must a rotation provide to head the family?** Enough for a `Periodic` heading to act on
   `Linear<2>` vectors, including one measured in degrees.
7. **The signatures of `LieGroup`'s members**, and whether `UnitQuaternion`'s `exp` returns Eigen's
   quaternion or coefficients.
8. **How does a model say that its Jacobian does not depend on the estimate?** The requirement is
   decided; the three mechanisms in "Jacobians that do not depend on the estimate" are proposals.
   The built-in measurement models also need a spelling for $b$: a constant vector and a weight on
   each vector part, by tag.
9. **How does a sigma-point method get the world side?** That it does is decided. Undecided:
   - **Whether the policy of the update is the side** and not only the carry: under `Linearized` it
     would set the carry, and under a sigma-point method the root that spreads the points as well.
   - **How a prediction chooses**, with no measurement to carry a policy.
   - **Who builds the axes.** The root has a place: `methods::spread(rule, axes)` gives a rule
     other axes [sigma-point-axes], and axes are an object that may hold a matrix, as
     `roots::along()` holds a direction. Axes see the covariance and not the mean, so world-side
     axes are built with the adjoint at the mean for each call, by the caller or by the library.
   - **What a model that reads some parts of a group spreads over.** The adjoint mixes the
     rotation's rows into every vector's, so the world-side root of a part's rows alone is not the
     part's rows of the world-side root.
   - **Which pairs of root and carry compile.** UKF-M has the body root with no carry and the world
     root with the adjoint carry; the library today has the body root with the reset. The body
     root with the adjoint carry is a filter no paper describes [published-algorithms-only].
