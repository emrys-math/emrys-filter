# Lie groups and the invariant EKF

Review notes on Lie groups as spaces, the family of poses, the policy that carries a covariance
through an update, the models that read and write a pose, and the invariant extended Kalman filter
that `KalmanFilter` becomes on them. Each section says what a piece does, shows what using it looks
like, and raises questions marked **Q*n***, which are collected at the end. §1 to §4 cover the
spaces; §5 the carry; §6 and §7 the models; §8 the sigma-point methods; §9 a pose whose size is
known at run time; §10 the example; §11 the evidence gathered before any of this was built; §12
what was checked and what is missing.

This review differs from the others here in what it reviews: a prototype, on the branch
`groups-prototype`, of roadmap step 17zf. Nothing in it has landed. It was built on 2026-11-19 to
try the design's open questions against code, so every question below is followed by the answer the
prototype implements, marked *Proposed*, and every one is still open. Where this document disagrees
with [lie-groups], [covariance-reset], [model-placement], and [unit-quaternion] in
[decisions.md](../design/decisions.md), this document is the newer one; "Decisions so far" lists
what was decided and is not yet written there.

Headers reviewed: `Spaces/Group.h`, `Spaces/GroupChecks.h`, `Spaces/ExtendedPose.h`, the group
members of `Spaces/UnitQuaternion.h`, the carries in `Spaces/Reset.h` and `Models/Condition.h`,
`roots::worldSide()` in `Distributions/Roots.h`, `Models/GroupModels.h`, `Models/Insertion.h`, and
what changed for a pose's parts in `Models/ResolvedReads.h`, `Models/MeasurementReads.h`,
`Models/ProcessModel.h`, `Models/Policies.h`, `Spaces/Indexing.h`, and `Distributions/Gaussian.h`.
Tests: `tests/Spaces/group.cpp` and `tests/Models/carries.cpp`, `group_parts.cpp`,
`group_run_time.cpp`, and `group_models.cpp`. Example: `examples/invariant_ekf.cpp`. Roadmap: step
17zf.

The examples were run against the commit that added this document (`groups-prototype`,
2026-11-21), with Clang 21.1 (MSVC's library), GCC 14.3 (Linux), and MSVC 19.50. Outputs shown in
comments are the printed results, the same on the three compilers except for residuals at rounding
size, which are Clang's, and for §10's example, whose random draws depend on the standard library.
These were the prototype's first compiles with GCC and MSVC, and each rejected code that Clang
takes: GCC a lambda's return type named through an alias inside a function that a member
declaration evaluates, and a lambda repeated by a pack expansion that names the class's members;
MSVC a `requires` expression inside a `static_assert` in a generic lambda. The same commit rewrote
the three. Compiler errors are quoted from Clang.

## Decisions so far

These are in decisions.md and are taken as given: a space is a set with $\boxplus$ and $\boxminus$
[difference-spaces]; `UnitQuaternion` applies its increment on the right [unit-quaternion]; an
update that moves the mean resets the covariance to the tangent space at the new mean
[covariance-reset]; blocks are named by the user's tags [tag-addressing]; a state may hold several
blocks of run-time size [several-resizable-blocks]; a function may return its Jacobian
[returned-jacobians]; a sigma-point rule spreads along axes that are a policy [sigma-point-axes];
and the library implements published algorithms [published-algorithms-only].

These were decided for this step on 2026-11-18 and 2026-11-19 and are not yet in decisions.md.
They go there, and [lie-groups], [covariance-reset], [model-placement], and [unit-quaternion]
change, when the questions below are closed.

- **No space has a side.** Every group applies its increment on the right,
  $X = \hat{X} \cdot \mathrm{Exp}(\xi)$, so every difference is a step in the body frame (§2).
  [lie-groups] says each pose space has "the side its increment is applied on".
- **The side belongs to the update, as a policy** (§5). [covariance-reset] has the reset alone.
- **A group may have a size known only at run time, from the start** (§3, §9). [lie-groups] names
  $SE(3)$ and $SE_2(3)$.
- **`Linear` and `Periodic` are Lie groups**, under addition (§1).
- **The adjoint carry on a space that is not a group is a compile error** (§5).
- **A part's rows of a difference are steps in the body frame, everywhere in the API** (§4).
- **`compose(space, a, b)` is the group product**, as in GTSAM and manif, and as "composition"
  means in mathematics: one after the other. **The models' `compose()` is now `blockwise()`** and
  `Composition` is `Blockwise`: `blockwise(range, bearing)` measures `Composite<Range, Bearing>`,
  block by block. It never means "this model on every block", which is `onEach`.
  **`UnitQuaternion::exponential()` and `logarithm()` are now `exp()` and `log()`**, the names of
  the concept's members. Rejected: `times()` for the product, which is not the field's name and
  reads oddly on `Linear`; `sideBySide()`, `stack()`, `join()`, `assemble()`, and `composeModels()`
  for the models; `composeGroups()` and `composeSpaces()`, which would mean building a product
  space, as `Composite` does.
- **Where a Jacobian does not depend on the estimate, the library does not differentiate at each
  step** (§7).
- **A sigma-point method gets the world side** (§8).
- **Noise in a frame other than the body's is an uncertain input** that the dynamics read
  [prediction-inputs], not a new kind of noise (§6).
- **A published result is not confirmed again before building.** That the invariant filter does
  better than a conventional one from a large initial error, and that contact points and landmarks
  belong inside the group, are the literature's. The example shows the first; no test depends on
  either.

## 1. Spaces and groups

A **space** is a set of values where a value can be moved by a small difference,
$x \boxplus \delta$, and two values have a difference, $a \boxminus b$
([Hertzberg et al. (2013)](../reference/bibliography.md#hertzberg2013), [difference-spaces]). That
is all the `Space` concept asks for.

A **group** is a space where two values also combine into a third, every value has an inverse, and
one value is the identity. In a Lie group the product determines the other two operations
([Solà, Deray, and Atchuthan (2018)](../reference/bibliography.md#sola2018), whose right-side
$\oplus$ and $\ominus$ these are):

$$x \boxplus \delta = x \cdot \mathrm{Exp}(\delta), \qquad
  a \boxminus b = \mathrm{Log}(b^{-1} \cdot a).$$

"Space" stays the word every user meets. "Group" names the added capability, as the concept
`LieGroup`, which refines `Space`.

| Space | A group? | Product |
| --- | --- | --- |
| `Linear` | yes | addition |
| `Periodic` | yes | addition modulo the period |
| `UnitQuaternion` | yes | the quaternion product |
| `Composite`, `Repeated` | when every block is | block by block |
| `ExtendedPose` (§3) | yes | a rotation acting on vectors |
| `Finite`, `Named`, `DiscreteProduct` | no | sets of labels |
| a user's space | not necessarily | a unit direction vector cannot be one |

- **Not every space is a group.** The 2-sphere has no continuous group operation, and a unit
  direction (a gravity direction, a bearing) is a common state. It still has a $\boxplus$ and a
  $\boxminus$, so `Space` stays the broader concept.
- **The same values can form different groups.** `Composite<Attitude, Position>` and a pose hold
  the same seven numbers. The composite adds a position increment directly; the pose rotates it by
  the attitude first (§3).

A space with a `plus()` of its own is a group when it also has the members `compose(a, b, out)`,
`inverse(a, out)`, `identity(out)`, `exp(d, out)`, `log(a, out)`, `adjoint(a, out)`, and
`rightJacobian(d, out)`. Each takes one column per operand and writes coefficients through its last
argument, as `plus()` does. `UnitQuaternion` and `ExtendedPose` have them. `Linear`, `Periodic`,
`Composite`, and `Repeated` have none: the free functions add, or go block by block.

The library and its users call the free functions `compose()`, `inverse()`, `identity()`, `exp()`,
`log()`, `adjoint()`, `rightJacobian()`, and `leftJacobian()`, which return coefficient vectors and
matrices. Each takes the space first, since a group of run-time size keeps its count in the
instance, and a fixed-size group may be named as a template argument instead:

```cpp
struct Attitude : UnitQuaternion {};
struct Bias : Linear<3> {};
struct Heading : RadiansPi<> {};
using AttitudeAndBias = Composite<Attitude, Bias>;

static_assert(LieGroup<Bias> && LieGroup<Heading> && LieGroup<Attitude> && LieGroup<AttitudeAndBias>);

Eigen::Vector4d const a = UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs();
Eigen::Vector4d const b = UnitQuaternion::exp(Eigen::Vector3d{-1.1, 0.3, 0.2}).coeffs();
Eigen::Vector3d const d{0.3, 0.5, -0.4};

// x [+] d is x Exp(d), and a [-] b is Log(b^-1 a).
Eigen::Vector4d const moved = Emrys::compose<Attitude>(a, Emrys::exp<Attitude>(d));
apart(Emrys::plus<Attitude>(a, d), moved);                                      // 0.0e+00
Eigen::Vector3d const between = Emrys::log<Attitude>(Emrys::compose<Attitude>(Emrys::inverse<Attitude>(b), a));
apart(Emrys::minus<Attitude>(a, b), between);                                   // 5.6e-17

// The adjoint moves a step from the right of a value to its left: a Exp(d) = Exp(Ad_a d) a.
Eigen::Vector3d const turned = Emrys::adjoint<Attitude>(a) * d;
apart(Emrys::compose<Attitude>(Emrys::exp<Attitude>(turned), a), moved);        // 1.1e-16

// A heading composes by addition and wraps.
Emrys::compose<Heading>(Eigen::Matrix<double, 1, 1>{3.0}, Eigen::Matrix<double, 1, 1>{1.0});   // -2.2832
```

`apart()` is the scratch program's largest absolute difference of two vectors or matrices. The
adjoint of a `Composite` is its blocks' adjoints on the diagonal, and a `Linear` block's is the
identity. For `Composite<Attitude, Bias>` at `a` with any bias:

```
 0.425 -0.820 -0.383  0.000  0.000  0.000
 0.573  0.571 -0.588  0.000  0.000  0.000
 0.701  0.031  0.713  0.000  0.000  0.000
 0.000  0.000  0.000  1.000  0.000  0.000
 0.000  0.000  0.000  0.000  1.000  0.000
 0.000  0.000  0.000  0.000  0.000  1.000
```

`hasIdentityAdjoint<S>` is true of a space whose adjoint is the identity at every value, a group
under addition or a `Composite` or `Repeated` of such, and the adjoint carry (§5) does nothing
there.

`checkGroup(space, a, b, d)` in `Spaces/GroupChecks.h` checks a group's members against each other
at two values and a difference, for the author of a group written from scratch. It returns the
largest residual of each relation: the group axioms, `plus()` and `minus()` against the two
formulas above, the adjoint against the product, and the right Jacobian against the automatic
derivative of $\mathrm{Log}(\mathrm{Exp}(d)^{-1} \mathrm{Exp}(d + \epsilon))$:

```cpp
Emrys::GroupCheck const check = Emrys::checkGroup(Attitude{}, a, b, d);
// identity 1e-16, inverse 0e+00, associative 2e-16, logOfExp 6e-17
// plus 0e+00, minus 6e-17, adjoint 2e-16, rightJacobian 1e-16
check.worst();   // the largest of the eight
```

The tests call it on every group in the table, on a pose of run-time size, and on a planar pose in
degrees.

`UnitQuaternion` keeps `exp(d)` returning Eigen's quaternion and `log(q)` taking one, beside
`map()`, as overloads of one argument fewer than the members. No other group has a typed view,
because Eigen has no type for one.

> **Q7.** The signatures of `LieGroup`'s members, and whether `UnitQuaternion`'s `exp` returns
> Eigen's quaternion or coefficients.
>
> *Proposed.* As above: members that write through `out`, free functions that return coefficients,
> and `UnitQuaternion`'s typed overloads kept. One cost showed while writing these examples: the
> free functions `exp` and `log` are in the namespace `Emrys`, so code under `using namespace Emrys`
> has them beside `std::exp` and the C library's. Nothing in the library or its tests failed for
> it, since every call there is qualified.

## 2. One side

Every group puts its increment on the right, $X = \hat{X} \cdot \mathrm{Exp}(\xi)$, as
`UnitQuaternion` did before this step. The differences are then steps in the body frame. In the
papers' vocabulary this is the *left-invariant* error $\hat{X}^{-1} X$; the names are crossed, and
this document says "right side" and "body frame" for the library and keeps "left-invariant" and
"right-invariant" for the published filters
([Barrau and Bonnabel (2017)](../reference/bibliography.md#barrau2017), as given in
[Brossard, Barrau, and Bonnabel (2020)](../reference/bibliography.md#brossard2020) and
[Solà, Deray, and Atchuthan (2018)](../reference/bibliography.md#sola2018)).

There is no side parameter on any space, for two reasons.

- With the reset, the choice changes no estimate. The two error coordinates are related exactly,
  $\mathrm{Exp}(\mathrm{Ad}_{\hat{X}} \xi) \cdot \hat{X} = \hat{X} \cdot \mathrm{Exp}(\xi)$, and the
  reset preserves the relation $P_\text{world} = \mathrm{Ad}_{\hat{X}} P_\text{body}
  \mathrm{Ad}_{\hat{X}}^T$ through every step. §11's first table measures it: under `Linearized`
  the two sides' means agree to $10^{-14}$ rad over 1000 steps.
- A parameter that changes only the frame the covariance and the process noise are expressed in
  would be chosen wrongly without any visible error.

Without the reset the side does matter, and the two published invariant filters omit the reset. In
one set of coordinates they and the library's filter differ only in the matrix that carries the
covariance after an update. That is why the side is a policy of the update (§5).

Sigma-point methods are the exception to "changes no estimate". Under `Unscented` and `Cubature`
the two sides differ, by up to 0.9 rad in the first step at a prior of 1.2 rad standard deviation
and by $8 \times 10^{-7}$ rad at a prior 50 times narrower (§11). Neither side is more correct: the
points are spread by a different square root of the same covariance. §8 gives those methods the
other side.

## 3. The pose family

One family stands where [lie-groups] planned $SE(3)$ and $SE_2(3)$: a rotation followed by any
number of vectors that turn with it. The parts are the user's tags, as in a `Composite`
[tag-addressing], and a `Repeated` may be among the vectors, so the size may be known only at run
time (§9).

```cpp
struct Attitude : UnitQuaternion {};
struct Velocity : Linear<3> {};
struct Position : Linear<3> {};

struct Pose : ExtendedPose<Attitude, Position> {};                  // SE(3)
struct Navigation : ExtendedPose<Attitude, Velocity, Position> {};  // SE_2(3)
using Separate = Composite<Attitude, Position>;

static_assert(LieGroup<Pose> && LieGroup<Navigation>);
static_assert(std::is_empty_v<Navigation>);
static_assert(valueDimensionAtCompileTime<Navigation> == 10);
static_assert(valueDimensionAtCompileTime<DifferenceSpace<Navigation>> == 9);
```

With no vectors the family is the rotation's group, with one $SE(3)$, with two $SE_2(3)$, and with
two plus a run of feet the group of
[Hartley et al. (2020)](../reference/bibliography.md#hartley2020). `ExtendedPose` derives from
`Composite` of the same blocks, so it holds the same values at the same paths, and `block<>()`,
`idx<>()`, `insert()`, and `erase()` work on it unchanged. It differs in the product, and so in its
$\boxplus$, $\boxminus$, reset, and adjoint. With $J$ the left Jacobian of the rotation:

$$(R, v_1, \dots, v_k) \boxplus (\phi, \rho_1, \dots, \rho_k) =
  (R \, \mathrm{Exp}(\phi), \; v_1 + R J(\phi) \rho_1, \; \dots, \; v_k + R J(\phi) \rho_k).$$

```cpp
// A body at (10, 0, 0) that has turned a quarter turn about z, so its first axis points along the world's y.
Eigen::Matrix<double, 7, 1> x;
x << UnitQuaternion::exp(Eigen::Vector3d{0.0, 0.0, quarter}).coeffs(), 10.0, 0.0, 0.0;

// One unit in the first row of the position's difference.
Eigen::Matrix<double, 6, 1> ahead;
ahead << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
Emrys::plus<Separate>(x, ahead);   // position 11.0000 0.0000 0.0000: one unit along the world's x
Emrys::plus<Pose>(x, ahead);       // position 10.0000 1.0000 0.0000: one unit along the body's first axis

// A quarter turn and one unit ahead in one difference: the pose moves along an arc of length one.
Eigen::Matrix<double, 6, 1> arc;
arc << 0.0, 0.0, quarter, 1.0, 0.0, 0.0;
Eigen::Matrix<double, 7, 1> const swept = Emrys::plus<Pose>(x, arc);   // position 9.3634 0.6366 0.0000
Emrys::minus<Pose>(swept, x);      // 0.0000 0.0000 1.5708 1.0000 0.0000 0.0000
```

The tests compare the product, inverse, exponential, $\boxplus$, and adjoint of $SE(3)$ with
Eigen's 4 by 4 homogeneous matrices and its matrix exponential, an implementation independent of
the space's.

- **A separate block cannot stand in for a vector of the group.** The adjoint of the group has
  entries between the rotation's columns and each vector's rows, which a `Composite`'s
  block-by-block carry cannot produce, and in the group a vector's correction is coupled to the
  attitude's at second order. This is from the algebra; no run has checked it.
- **The right Jacobian is not a closed form.** The family's reset is
  $J_r(\delta) = \int_0^1 \mathrm{Ad}_{\mathrm{Exp}(-s\delta)} \, ds$, which the prototype
  integrates by a 12-point Gauss-Legendre rule. It agrees with automatic differentiation to
  rounding for steps of a turn or less. `UnitQuaternion`'s own is the closed form it had.

> **Q5.** Names: the family, the carry policy and its three values (§5), and the action of a value
> on a point (§7).
>
> *Proposed for the family.* `ExtendedPose`, the literature's term for $SE_k(3)$. It reads oddly
> for the cases at the ends: `ExtendedPose<Attitude>` is a rotation alone, and
> `ExtendedPose<Compass, Place>` is the plane's plain pose.

### What heads the family

The first part may be any rotation that acts on the vectors. The `Rotation` concept is a fixed-size
`LieGroup` with the size of the vectors it turns, `ActionDimension`, and three members:
`rotationMatrix(q, out)`, the matrix that turns a vector; `integratedRotation(d, out)`,
$V(d) = \int_0^1 R(\mathrm{Exp}(s d)) \, ds$, which for a `UnitQuaternion` is the left Jacobian of
$SO(3)$ and for a heading a 2 by 2 matrix; and `actionJacobian(v, out)`, the derivative of the
turned vector in the rotation's step. The family's product, exponential, and adjoint are written
from these alone. A `Periodic` space of one coordinate is a rotation without members of its own,
its period one turn, so the same template is $SE_k(2)$ in any unit of angle:

```cpp
struct Compass : Degrees360<> {};
struct Place : Linear<2> {};
struct PlanarPose : ExtendedPose<Compass, Place> {};

static_assert(Rotation<Attitude> && Rotation<Compass> && !Rotation<Position>);

// At 90 degrees and (10, 0): a quarter turn and one unit ahead, the arc of the example above.
Emrys::plus<PlanarPose>(Eigen::Vector3d{90.0, 10.0, 0.0}, Eigen::Vector3d{90.0, 1.0, 0.0});
// 180.0000 9.3634 0.6366
```

The heading's column of the adjoint is then per degree, which a test checks.

> **Q6.** What must a rotation provide to head the family? Enough for a `Periodic` heading to act
> on `Linear<2>` vectors, including one measured in degrees.
>
> *Proposed.* The three members above, which sufficed for both rotations. One part of the answer
> is too broad: every `Periodic` space of one coordinate satisfies `Rotation`, so a phase or a time
> of day could head a pose of `Linear<2>` vectors and nothing would object. A heading may need to
> say that it is one.

## 4. Body-frame rows

Each part owns rows of a difference, as a block of a `Composite` does. A step in one vector's rows
alone moves only that vector, by the step turned to the world:

$$\hat{X} \cdot \mathrm{Exp}(0, \dots, \rho_i, \dots, 0) =
  (\hat{R}, \dots, \hat{v}_i + \hat{R} \rho_i, \dots).$$

So a vector's rows are steps of that vector expressed in the body frame at the mean, and they mean
that everywhere in the API:

- `covariance<Position>()` is the covariance of the body-frame step. To first order the covariance
  of the position's own error is $\hat{R} P_{\rho\rho} \hat{R}^T$.
- A prior over a pose gives its vectors' variances in these rows.
- Noise placed on a part is body-frame noise, which is what an accelerometer's and a gyro's is
  (§6).
- Erasing a vector drops its rows, which is exact. Inserting one takes its covariance and
  cross-covariance in these rows (§9).

```cpp
// The body of §3's example, with its position known to 2 along the body's first axis and to 0.5 across it.
Eigen::Matrix<double, 9, 1> variances;
variances << 0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 4.0, 0.25, 0.25;
auto const state = GaussianParams<Navigation>{mean, variances.asDiagonal()};

Eigen::Matrix3d const bodyFrame = state.covariance<Position>();     // diagonal 4.0000 0.2500 0.2500
Eigen::Matrix3d const turn = Eigen::Quaterniond{Eigen::Vector4d{state.mean().head<4>()}}.toRotationMatrix();
Eigen::Matrix3d const worldFrame = turn * bodyFrame * turn.transpose();   // diagonal 0.2500 4.0000 0.2500
```

Three 3 by 3 matrices can be called "the position's covariance", and they differ. The library
returns the first: $P_{\rho\rho}$, in the pose's rows. The second, $\hat{R} P_{\rho\rho}
\hat{R}^T$, is the covariance of $p - \hat{p}$ to first order, which is what a plot or a gate in
the world frame wants; the caller forms it, as above. The third is the position's block of
$\mathrm{Ad}_{\hat{X}} P \mathrm{Ad}_{\hat{X}}^T$, the covariance in world-side differences of the
pose, whose position rows are $\delta p + \hat{p} \times \phi_w$ and so include the attitude's
uncertainty times the distance from the origin.

A Gaussian over a part's own space, in the world frame, is never what the library returns under
the part's name. The marginal of a vector is over the smaller pose of the rotation and that vector:

```cpp
auto const marginal = state.marginal<Position>();
static_assert(std::is_same_v<std::remove_cvref_t<decltype(marginal.space())>, ExtendedPose<Attitude, Position>>);
marginal.mean();              // 7 values: 0.000 0.000 0.707 0.707 10.000 0.000 0.000
marginal.covarianceLower();   // 6 rows: the attitude's and the position's, and their cross-covariance

auto const attitude = state.marginal<Attitude>();   // 4 values, 3 rows: the rotation's marginal is its own
```

It is exact, not first order: dropping the other vectors is a homomorphism of the group, so a
state moved by a difference has the smaller pose moved by that difference's rows, which a test
checks at a large step. A conditional given a part does not compile:

```cpp
auto const given = state.conditional<Position>(Eigen::Vector3d{10.0, 0.5, 0.0});
```

```
error: static assertion failed due to requirement 'posePartOf<Navigation, Position> ==
PosePart::none': a conditional given a part of an ExtendedPose is not defined yet
(docs/guide/groups.md, open question 2): the part's rows are steps in the pose's body frame, so
its value alone gives no residual in them. Condition on the pose whole, or update by a measurement
of the part
```

> **Q2.** What is a distribution over a part? `covariance<>()` of a part is settled above. What
> `marginal<>`, a conditional, and an `on<>()` factor return for a part is not: a Gaussian labeled
> with the part's own space would be read as a world-frame one.
>
> *Proposed.* The marginal as above; the conditional refused, with the message above; an `on<>()`
> factor of a part not addressed, since a `Product` already refuses a space with a quaternion. The
> marginal is exact, and it is probably not what a caller of `marginal<Position>()` wants: it hands
> back seven values and a space the caller did not name, and the world-frame Gaussian over the
> position that a plot or a gate needs still has to be formed by hand.

> **Q10.** Should the library turn a vector's covariance between the pose's rows and the world
> frame? A prior goes in as body-frame variances and a covariance comes out as one, and for equal
> variances on the three axes the frames agree, which is why the example's prior is the same
> numbers on both spaces. For anything else the caller multiplies by $\hat{R}$ on each side, in
> each direction, and nothing in a type says which frame a 3 by 3 matrix is in. A function of the
> state and a path that returns $\hat{R} P_{\rho\rho} \hat{R}^T$, and its inverse for a prior,
> would cover both.

## 5. The covariance carry

An update computes the posterior covariance in the tangent space at the prior mean and moves the
mean by a step $\delta$. What it then does to the covariance is $P^+ \leftarrow G P^+ G^T$, and
three choices of $G$ give three filters on the same right-side space:

| Carry | $G$ | Filter |
| --- | --- | --- |
| `carries::none()` | $I$ | the published left-invariant EKF |
| `carries::reset()` | the right Jacobian $J_r(\delta) = \sum_k (-\mathrm{ad}_\delta)^k / (k+1)!$ | the library before this step [covariance-reset]; the same filter on either side |
| `carries::adjoint()` | $\mathrm{Ad}(\mathrm{Exp}(-\delta)) = \sum_k (-\mathrm{ad}_\delta)^k / k!$ | the published right-invariant EKF, with its covariance in body coordinates |

To first order in $\delta$ the reset is halfway between the two published filters. The carry is an
argument of `observe()`, as the `updates::` forms are [covariance-update-forms], because which
carry suits depends on the measurement:

| Measurement | Form | Carry that suits it |
| --- | --- | --- |
| something fixed in the body, measured in the world: a position fix, a velocity fix | $y = X \cdot b$ | none |
| something fixed in the world, measured from the body: a landmark, a magnetic field, a gravity direction, a foot through the leg's kinematics | $y = X^{-1} \cdot b$ | adjoint |
| anything else, and any space that is not a group | | reset |

Each published filter pairs a measurement form with the error coordinates in which the
measurement's Jacobian and the covariance recursion do not depend on the estimate, apart from a
rotation of the measurement noise. A reset makes the covariance depend on the innovation, which is
the likely reason the papers omit it. It is not proved here that this explains §11's numbers.

```cpp
// Two landmarks seen from the body, y_i = R^T (l_i - p), on the Pose of §3.
auto const sightings = Emrys::measurementModel(
    []<typename T>(auto const & in, auto const & x) {
        Eigen::Quaternion<T> const q{UnitQuaternion::map(Emrys::block<Attitude>(in, x))};
        Eigen::Matrix<T, 6, 1> y;
        y.template head<3>() = q.conjugate() * (landmarkA - Emrys::block<Position>(in, x)).eval();
        y.template tail<3>() = q.conjugate() * (landmarkB - Emrys::block<Position>(in, x)).eval();
        return y;
    },
    Emrys::reads(Emrys::path<Attitude>, Emrys::path<Position>),
    Emrys::outputSpace(Sightings{}),
    Emrys::additiveNoise(Matrix6{Vector6{0.04, 0.09, 0.02, 0.05, 0.03, 0.08}.asDiagonal()})
);
auto const prior = GaussianParams<Pose>{mean, Vector6{0.25, 0.25, 0.25, 2.0, 2.0, 2.0}.asDiagonal()};

auto const none = Emrys::update(prior, Emrys::observe(methods::linearized(), sightings, z, carries::none()));
auto const reset = Emrys::update(prior, Emrys::observe(methods::linearized(), sightings, z));
auto const adjoint = Emrys::update(prior, Emrys::observe(methods::linearized(), sightings, z, carries::adjoint()));

// One mean, three covariances. The step: 0.476 -0.440 0.449 1.452 1.734 -2.441
// means apart: 0.0e+00, 0.0e+00
Matrix6 const p = full(none->branches[0].posterior);
Matrix6 const j = Emrys::rightJacobian<Pose>(step);
Matrix6 const ad = Emrys::adjoint<Pose>(Emrys::exp<Pose>((-step).eval()));
apart(full(reset->branches[0].posterior), j * p * j.transpose());       // 1.4e-16
apart(full(adjoint->branches[0].posterior), ad * p * ad.transpose());   // 3.6e-16
// position variances, none:    1.2584 0.1118 0.0977
// position variances, reset:   0.5665 0.1256 0.0197
// position variances, adjoint: 0.1849 0.1258 0.0256
```

`z` is the sightings from a pose well away from the prior mean, so that the step is large; at a
small step the three covariances agree. A filter's `update()` takes the same argument. The tests
compare `none()` with a left-invariant update written out in plain Eigen, to $10^{-11}$, and
`adjoint()` with a right-invariant update written out in world-frame differences and brought back
by the adjoint at the posterior mean, to $10^{-10}$.

- **The default stays the reset.** It needs no knowledge of the measurement and is the only one
  defined for a space that is not a group. In §11's runs it was between the other two, except in
  the mean with landmarks at the widest prior, where a few runs dominate and its median was still
  between. A measurement model that names its form defaults to the carry that suits it (§7).
- **Biases are unaffected.** A `Composite` of a group and `Linear` bias blocks carries block by
  block, and a `Linear` block's adjoint is the identity. This is what the published filters with
  biases do, and a test checks it.
- **Prediction needs no carry.** A prediction's covariance is computed at the predicted mean.
- **The adjoint carry on a space that is not a `LieGroup` is a compile error** that names the
  block. For a state `Composite<Attitude, Direction>` with `Direction` a space with a `plus()` of
  its own and no group members:

```
error: static assertion failed due to requirement 'isLieGroup<Direction>()': carries::adjoint()
carries a covariance by a group's adjoint, and this block of the state is not a Lie group: its
space has a plus() of its own and no group members (see LieGroup). Use carries::reset()
```

The carries are in the covariance form alone. The square-root and information forms refuse any
carry but the reset on a space where the carry does something:

```cpp
auto rootFilter = Emrys::SqrtKalmanFilter{hold, Emrys::toSqrt(prior).value()};
rootFilter.update(Emrys::observe(methods::linearized(), sightings, z, carries::adjoint()));
```

```
error: static assertion failed due to requirement 'std::is_same_v<Emrys::carries::Adjoint,
Emrys::carries::Reset> || hasIdentityReset<Pose>': a carry other than carries::reset() is the
covariance form's so far (docs/guide/groups.md): filter in a Gaussian
```

That message ends badly: "filter in a Gaussian" means "use `KalmanFilter` over a `GaussianParams`".

> **Q4.** Where does the carry apply? Every Kalman-family update, presumably; the smoother, a
> Gaussian's conditional, and the information forms' reference are undecided.
>
> *Proposed.* The carry is a policy of a measurement's update and of nothing else. A conditional,
> the smoother, and the information forms' reference keep the reset: none of them has a measurement
> whose form a carry could suit, and the published invariant filters define none of them. The
> square-root and information forms' updates should take the carries; the prototype does not have
> them there.

> **Q5, continued.** *Proposed for the carry.* `carries::reset()`, `carries::none()`, and
> `carries::adjoint()`, named for what each does to the covariance. These are the weakest names
> here. Under a sigma-point method `carries::adjoint()` also moves the spread to the world side
> (§8), which the name does not say, so what the argument chooses is the side and the carry is one
> consequence. Names by the published filter (`leftInvariant()`, `rightInvariant()`) or by the
> side would say that, at the cost of the papers' crossed vocabulary (§2).

> **Q12.** What is the carry under `Iterated`? The published filters are linearized, and so is
> every run of the carries in §11. The prototype applies the carry once, to the whole step from the
> prior mean to the last iterate, after the last iterate. No run has examined that and no test
> covers it.

## 6. Models on parts of a pose

The filter the roadmap names as the main use is `blockwise(imu, RandomWalk<Feet>{q})`: one model
writes the attitude, velocity, and position, and another the feet. So separate models must be able
to write parts of one group, and a measurement to read some of them. A model that names a part
sees the part's own values and differences, as it would in a `Composite`, and the library turns
between those and the part's rows. No model is written for a pose, and one model object serves a
pose and a `Composite` of the same blocks:

```cpp
struct Foot : Linear<3> {};
struct Bias : Linear<3> {};
struct Body : ExtendedPose<Attitude, Velocity, Position, Foot> {};
using State = Composite<Body, Bias>;
using Separate = Composite<Attitude, Velocity, Position, Foot, Bias>;

// The attitude turned by a known rate. The model returns its Jacobian over the attitude's own differences.
auto const turning = Emrys::processModel(
    [](auto const & in, auto const & out, auto const & x, double dt) {
        Eigen::Quaterniond const step = UnitQuaternion::exp((dt * rate).eval());
        Eigen::Quaterniond const q{UnitQuaternion::map(block<Attitude>(in, x))};
        auto y = Emrys::zeros<double>(out);
        block<Attitude>(out, y) = (q * step).coeffs();
        Eigen::Matrix3d const jacobian = step.toRotationMatrix().transpose();
        return Emrys::withJacobian(y, jacobian);
    },
    Emrys::reads(path<Attitude>),
    Emrys::writes(path<Attitude>),
    Emrys::continuousNoise(0.01 * Eigen::Matrix3d::Identity())
);
// Four models side by side, each naming the blocks it moves. None of them mentions the pose.
auto const motion = Emrys::blockwise(
    turning,
    Emrys::ConstantVelocity<Position, Velocity>{0.2},
    Emrys::RandomWalk<Foot>{0.05},
    Emrys::RandomWalk<Bias>{0.001}
);

// The prior covariance is the identity but for the foot's rows, which are diag(1, 2, 3).
auto const onPose = Emrys::predict(methods::linearized(), GaussianParams<State>{mean, covariance}, motion, dt);
auto const onSeparate = Emrys::predict(methods::linearized(), GaussianParams<Separate>{mean, covariance}, motion, dt);
apart(onPose.mean(), onSeparate.mean());   // 0.0e+00: the same values

// The foot did not move in the world, and its rows are steps in the body frame, which turned.
onPose.covariance<Emrys::TagPath<Body, Foot>>();
//   1.2013 0.3190 0.3715
//   0.3190 1.9226 0.0261
//   0.3715 0.0261 2.9361     against R'^T R P R^T R' + q dt: 1.8e-15
onSeparate.covariance<Foot>();
//   1.0200 0.0000 0.0000
//   0.0000 2.0200 0.0000
//   0.0000 0.0000 3.0200
```

A foot's rows are body-frame steps at the *new* attitude, which another model writes, so a random
walk on a foot has the Jacobian $\hat{R}'^T \hat{R}$ in the foot's rows, not the identity that
`RandomWalk` returns. What the library does to make that so:

- **A prediction already calls the models side by side as one function of the whole state** and
  takes differences in the state's space. So a model that is differentiated or spread over sigma
  points needed no change.
- **A process model that returns its Jacobian returns it over its blocks' own differences.** The
  library multiplies a vector's columns by $\hat{R}$ and a written vector's rows by $\hat{R}'^T$,
  and gives a vector the step does not write $\hat{R}'^T \hat{R}$ in place of the identity. These
  are products of 3 by 3 blocks, not a differentiation.
- **A measurement's read of a part takes the whole pose** into the joint space it is pushed
  through, once however many parts are read. Differences and sigma points are then the pose's. A
  Jacobian or a matrix the model gives over the part has its columns turned by $\hat{R}$.
- **Such a model's Jacobian over the state's rows depends on the rotation** even where the function
  is affine. So `affineIn()` does not factor a part out, and `FixedStepModel` does not cache the
  transition of a model that touches a part.

A measurement that reads the position alone, as a matrix and as a function:

```cpp
auto const fix = Emrys::measurementModel(
    Eigen::Matrix3d::Identity(),
    Emrys::reads(path<Body, Position>),
    Emrys::additiveNoise(Eigen::Matrix3d{0.04 * Eigen::Matrix3d::Identity()})
);
auto const byMatrix = Emrys::update(onPose, Emrys::observe(fix, z));
// fixByFunction returns block<Position>(in, x), with reads(path<Position>).
auto const byFunction = Emrys::update(onPose, Emrys::observe(methods::linearized(), fixByFunction, z));
// position after the fix: 2.4734 -1.0199 1.1741; matrix and function apart 0.0e+00
```

The tests compare each of these with the same model written over the whole state, which reads and
writes the pose as one block and is differentiated through the pose's own `plus()` and `minus()`.

> **Q1.** How do separate models write parts of one group? This is required, not optional: it is
> the intended use.
>
> *Proposed.* As above. It took the least work of the nine, because a prediction was already one
> function of the whole state. Undecided within it: whether a measurement's read should take only
> the smaller pose of the rotation and the vectors read. That is exact, by §4's homomorphism, and
> cheaper: as built, a position fix on a body with a velocity, a position, and $n$ feet
> differentiates, or spreads points, over all $9 + 3n$ of the body's rows, where 6 would do.

### Noise

- **Noise in the body frame is additive** on the pose's rows, by `continuousNoise()` or
  `additiveNoise()` [continuous-time-noise], placed on the pose or on a part.
- **Noise in another frame is an uncertain input** that the dynamics read. World-frame noise read
  as $X' = \mathrm{Exp}(w) \cdot f(X)$ gave the estimate-dependent
  $\mathrm{Ad}(\hat{X}'^{-1}) \, Q \, \mathrm{Ad}(\hat{X}'^{-1})^T$ to $2 \times 10^{-14}$ in the
  scratch program of §11.
- **A defect is fixed:** `additiveNoise(on<Body>(Q))` did not compile on a block whose values and
  differences differ in size, since `placeNoise()` built a `Product` of the placed parts, which
  requires equal sizes. It now places each covariance over its block's differences.

A model that names a part therefore has two frames. Its function, and any Jacobian it returns, are
in the part's own differences, which for a vector are world-frame. Noise it places on the part is
in the part's rows, which are body-frame. For a scalar density the two are the same, which is why
`RandomWalk<Foot>{0.05}` above and the example's mechanization mean the same on both spaces. For a
matrix they are not:

```cpp
// A position that drifts along the world's first axis only, on a body turned a quarter turn about z.
Eigen::Matrix3d const alongWorldX = Eigen::Vector3d{1.0, 0.0, 0.0}.asDiagonal();
auto const drift = Emrys::processModel(
    []<typename T>(auto const & in, auto const & out, auto const & x, double) {
        auto y = Emrys::zeros<T>(out);
        block<Attitude>(out, y) = block<Attitude>(in, x);
        block<Position>(out, y) = block<Position>(in, x);
        return y;
    },
    Emrys::reads(path<Attitude>, path<Position>),
    Emrys::writes(path<Attitude>, path<Position>),
    Emrys::continuousNoise(Emrys::on<Position>(alongWorldX))
);
auto const separate = Emrys::predict(methods::linearized(), GaussianParams<Separate>{facingY, sure}, drift, 1.0);
auto const pose = Emrys::predict(methods::linearized(), GaussianParams<Pose>{facingY, sure}, drift, 1.0);
// Composite, world-frame variances of the position: 1.00 0.00 0.00
// Pose, world-frame variances of the position:      0.00 1.00 0.00
```

On the pose the drift is along the body's first axis, which points along the world's $y$.

> **Q11.** Should a matrix of noise placed on a vector of a pose be refused, or turned, or left as
> it is? As built, the same model object puts the same matrix in the world frame on a `Composite`
> and in the body frame on a pose, and nothing warns. The decisions cover each half: a part's rows
> are body-frame everywhere, and noise in another frame is an uncertain input. What they do not
> cover is that a model written for a `Composite` changes meaning when the state's space changes,
> which the "one model object serves both" of Q1 invites. Three choices: leave it and document it;
> turn a matrix placed on a vector by $\hat{R}$, so that a model means the same on both spaces and
> body-frame noise is placed on the pose or needs a spelling of its own; or refuse a matrix that is
> not a multiple of the identity there unless the placement names its frame.

## 7. Jacobians that do not depend on the estimate

On group-affine dynamics the error's Jacobian does not depend on the estimate, and a measurement of
either form in §5's table has a Jacobian that needs no differentiation. This is the property the
invariant filters of [Barrau and Bonnabel (2017)](../reference/bibliography.md#barrau2017) are
built on. That paper was not read: the forms below were derived for this step, and the tests check
the measurements' against automatic differentiation and the motion's against the group's adjoint,
which a differentiated twist gives too. Where the property holds, the library does not
differentiate at each step. Before this step it would have: a function that does not return its
Jacobian is differentiated automatically at every call [returned-jacobians]. That gave the right
numbers in §11 and costs what the published filters avoid.

The Jacobians are in the library's body-frame rows, with $c_i$ the weight of vector $i$ in $b$ and
$b_0$ its part that only rotates:

| Case | The Jacobian depends on | The library |
| --- | --- | --- |
| a fixed step and no input | nothing | computes it once and keeps it [fixed-step-models] |
| a motion driven by an input, $X' = X \cdot \mathrm{Exp}(u \, \Delta t)$ | the input and the step | forms $\mathrm{Ad}(\mathrm{Exp}(-u \, \Delta t))$ from them |
| $y = X \cdot b$ | the rotation at the mean | forms $\hat{R} \, [-b_0^\wedge, \; c_1 I, \dots, c_k I]$ |
| $y = X^{-1} \cdot b$ | the predicted measurement | forms $[\hat{y}_0^\wedge, \; -c_1 I, \dots, -c_k I]$, with $\hat{y}_0 = \hat{R}^T b_0$ |

A Jacobian that is the same for the whole run exists only in the first row. An inertial
mechanization's is constant in the world frame, where it depends on gravity alone, and in
body-frame rows it depends on the gyro and accelerometer readings. The world frame is no cheaper:
there the process noise is $\mathrm{Ad}_{\hat{X}} \, Q \, \mathrm{Ad}_{\hat{X}}^T$ at every step,
where in body-frame rows it is constant. Neither frame differentiates.

`Models/GroupModels.h` has the last three rows as models that return their Jacobians
[returned-jacobians]:

```cpp
struct Twist : Linear<9> {};
struct Body : ExtendedPose<Attitude, Velocity, Position> {};
using State = Composite<Body, Bias>;

// X' = X Exp(u dt), with u an input of each prediction: the rate of each row of the body's difference.
auto const motion = Emrys::blockwise(Emrys::bodyVelocity<Body, Twist>(), Emrys::RandomWalk<Bias>{0.001});
Eigen::Matrix<double, 9, 9> const f = Emrys::adjoint<Body>(Emrys::exp<Body>((-dt * twist).eval()));
for (double const scale : {1.0, -2.0})   // two prior means far apart
{
    auto const prior = GaussianParams<State>{meanAt(scale), covariance};
    auto const predicted =
        Emrys::predict(methods::linearized(), prior, motion, dt, GaussianParams<Twist>{twist, twistNoise});
    Eigen::Matrix<double, 9, 3> const cross = full(predicted).topRightCorner<9, 3>();
    apart(cross, f * covariance.topRightCorner<9, 3>());   // 0.0e+00 at both: one F at every estimate
}

// A landmark seen from the body, named by its form: a point fixed in the world, less the position.
auto const sight = Emrys::observedInBody<Body>(
    landmark,
    std::tuple{Emrys::on<Position>(1.0)},
    Emrys::outputSpace(Sighting{}),
    Emrys::additiveNoise(noise)
);
auto const formed = Emrys::update(prior, Emrys::observe(sight, z));   // no method, no carry
auto const differentiated =
    Emrys::update(prior, Emrys::observe(methods::linearized(), byFunction, z, carries::adjoint()));
apart(full(formed->branches[0].posterior), full(differentiated->branches[0].posterior));   // 1.1e-16

auto const withReset = Emrys::update(prior, Emrys::observe(sight, z, carries::reset()));
apart(full(formed->branches[0].posterior), full(withReset->branches[0].posterior));        // 9.8e-02
```

`byFunction` is the same measurement written as a function of the attitude and the position, which
the library differentiates. A model's function type may name a `DefaultCarry` and a
`DefaultMethod`, as a model given its matrix defaults to `Linearized` [linear-sensors], and the two
measurement models do: `observedInBody()` takes the adjoint carry and `observedInWorld()` none, so
`observe(model, z)` with neither is the published filter's update for that form. A carry or a
method named in `observe()` is taken instead, as the last two lines show.

Two more mechanisms were planned. A user's group-affine model returns its Jacobian with
`withJacobian()`, over its blocks' own differences (§6); that works, and the inertial
mechanization, which is EmrysNavigation's [lie-groups], is such a model. `affineIn()` extended to a
group block, to mean that $f(x \boxplus \delta) \boxminus f(x) = F \delta$ with $F$ independent of
$x$, is not built: [affine-coordinates] still treats every block that is not linear as nonlinear.
Group-affine dynamics satisfy that property exactly, for a $\delta$ of any size, so with it
`FixedStepModel` would cache $F$ and a sigma-point method would factor the block out exactly.

> **Q8.** How does a model say that its Jacobian does not depend on the estimate? The requirement
> is decided; the mechanisms are proposals. The built-in measurement models also need a spelling
> for $b$: a constant vector and a weight on each vector, by tag.
>
> *Proposed.* The built-in models and `withJacobian()`, with `DefaultCarry` and `DefaultMethod`.
> The requirement is only partly met. The example's mechanization (§10) is written as a function
> and differentiated at every step; the family's right Jacobian is a quadrature (§3); `affineIn()`
> on a group block is not built; and there is no benchmark of the invariant filter's step against
> a hand-written one, which "does not differentiate at each step" needs, being a claim about cost
> ([benchmarks.md](../design/benchmarks.md)). The spelling of $b$,
> `std::tuple{Emrys::on<Position>(1.0)}`, is clumsy, and a weight on one element of a `Repeated`
> block, such as one foot, has no spelling.

> **Q5, continued.** *Proposed for the action.* No general action of a value on a point. The two
> measurement models stand in its place, named for where the point is observed:
> `observedInWorld<Pose>()` and `observedInBody<Pose>()`.

## 8. Sigma-point methods on a group

Nothing had to be built for an unscented filter on a group: `Unscented`, `Cubature`, and the other
point sets [sigma-point-sets] run on a group space as they ran on `UnitQuaternion`. The literature
has the filter
([Brossard, Barrau, and Bonnabel (2020)](../reference/bibliography.md#brossard2020), "UKF-M"), and
UKF-M's default uncertainty, $\chi = \hat\chi \exp(\xi)$, is the library's right-side $\boxplus$.

- **The side matters again, and the carry does not cover it.** With the reset, the two sides give
  the same filter under `Linearized` and different ones under a sigma-point method (§2). Since
  $\mathrm{Exp}(\zeta) \hat{X} = \hat{X} \, \mathrm{Exp}(\mathrm{Ad}_{\hat{X}}^{-1} \zeta)$, the
  world-side filter is, in body-frame rows, the same filter with its points spread by another
  square root of the same covariance, $\mathrm{Ad}_{\hat{X}}^{-1} \,
  \mathrm{chol}(\mathrm{Ad}_{\hat{X}} P \mathrm{Ad}_{\hat{X}}^T)$, and with the adjoint carry. So
  the world side needs the root as well as the carry, in a prediction as well as an update. UKF-M
  offers both sides and reports the world side ahead on its inertial navigation example, with no
  general rule.
- **The library's filter is not UKF-M step for step.** UKF-M propagates the mean through the model
  without noise and uses the points for the covariance alone; the library takes a tangent-space
  mean of the propagated points. UKF-M spreads separate point sets over the state and the noise;
  the library spreads one over both. UKF-M applies no carry after an update and says that the
  transport is left to the user; the library resets. The reference documentation must name the
  algorithm the library implements and say where it departs from these
  [published-algorithms-only]; it does not yet.
- **On group-affine dynamics the points buy nothing in a prediction.** The map of the differences
  is exactly linear, so every point set returns the linearized result, at one evaluation per point.
  A test checks it for a twist under `Unscented`, to $10^{-9}$. What a sigma-point method adds is
  in the update, where $\mathrm{Exp}(\xi) \cdot b$ is not linear in $\xi$, and in the blocks beside
  the group, such as biases.
- **The matching side is a matter of the root too.** For $y = X \cdot b$ the points
  $\hat{X} \cdot \mathrm{Exp}(\xi_i) \cdot b$ are $\hat{X}$ acting on points that do not depend on
  the estimate, so the gain in the body frame does not depend on it, as in the invariant EKF. For
  $y = X^{-1} \cdot b$ the same holds only with the points spread on the world side. This is from
  the algebra; no run has checked it.

`roots::worldSide(inner)` is axes [sigma-point-axes] whose root is
$\mathrm{Ad}_{\hat{X}}^{-1} S_w$, with $S_w$ the root by `inner` of the covariance in world-frame
differences. Axes see a covariance and not a mean, so these have a second member,
`rootAt(space, mean, cov)`, which the transform calls at the mean of the input it spreads. A
measurement with `carries::adjoint()` under a sigma-point method spreads on the world side without
being told, and a prediction chooses by its method:

```cpp
// The sighting of §7 as a function, on the same prior.
auto const cubature = Emrys::update(prior, Emrys::observe(methods::cubature(), byFunction, z, carries::adjoint()));
auto const named = Emrys::update(
    prior,
    Emrys::observe(methods::spread(methods::cubature(), Emrys::roots::worldSide()), byFunction, z, carries::adjoint())
);
auto const bodySide = Emrys::update(prior, Emrys::observe(methods::cubature(), byFunction, z));
// cubature with carries::adjoint(), against worldSide() named: 0.0e+00
// cubature with the reset, on the body side, apart from it:    1.9e-01

// A motion that is not group-affine, so that the side shows.
auto const onBody = Emrys::predict(methods::cubature(), prior, bend, dt);
auto const onWorld = Emrys::predict(methods::spread(methods::cubature(), Emrys::roots::worldSide()), prior, bend, dt);
apart(full(onBody), full(onWorld));   // 9.5e-02
```

A test checks the identity in the first item of the list above: a cubature filter written out in
plain Eigen in world-frame differences, with no carry there, agrees with the library's under
`carries::adjoint()` to $10^{-9}$. An iterated sigma-point method refuses the adjoint carry, since
its iterates are not spread on the world side:

```cpp
// The prior and the sightings of §5, on its Pose.
Emrys::update(prior, Emrys::observe(methods::iterated(methods::cubature()), sightings, z, carries::adjoint()));
```

```
error: static assertion failed due to requirement '!(isSigmaPointMethod<Emrys::methods::Cubature>
&& std::is_same_v<Emrys::carries::Adjoint, Emrys::carries::Adjoint> &&
!carriesNothing<Emrys::carries::Adjoint, Pose>)': carries::adjoint() is not yet taken by an
iterated sigma-point method (docs/guide/groups.md): iterate methods::linearized(), or use the
sigma-point method without iterating
```

The world side is also refused, at compile time, from a square-root form's factor and in a
transform that factors out `affineIn()` coordinates.

> **Q9.** How does a sigma-point method get the world side? That it does is decided. Undecided:
> whether the policy of the update is the side and not only the carry; how a prediction chooses,
> with no measurement to carry a policy; who builds the axes, which need the mean; what a model
> that reads some parts of a group spreads over, since the adjoint mixes the rotation's rows into
> every vector's; and which pairs of root and carry compile. UKF-M has the body root with no carry
> and the world root with the adjoint carry; the library before this step had the body root with
> the reset; the body root with the adjoint carry is a filter no paper describes
> [published-algorithms-only].
>
> *Proposed.* As built:
> - *The update's policy* is the side: `carries::adjoint()` brings the world root.
> - *A prediction* chooses by its method, `methods::spread(rule, roots::worldSide())`.
> - *The axes* are built by the library, through `rootAt()`. A user's axes may have one too, which
>   widens what axes are without having decided to.
> - *A model that reads some parts* reads the whole pose (§6), so the adjoint is over a group.
> - *Which pairs compile:* the body root with the adjoint carry can no longer be written in an
>   update. The world root with the reset or with no carry still compiles, in an update by naming
>   the axes and in any prediction. No paper describes either.

## 9. A pose of run-time size

Contact-aided estimation keeps one vector per foot in contact inside the group, and
right-invariant SLAM keeps the landmarks there. A `Repeated` among the vectors gives the pose a
count known at run time, and the members follow it:

```cpp
struct Foot : Linear<3> {};
struct Feet : Repeated<Foot> { using Repeated::Repeated; };
struct Leg : Linear<3> {};
struct Body : ExtendedPose<Attitude, Position, Feet> { using ExtendedPose::ExtendedPose; };
struct State : Composite<Body, Bias> { using Composite::Composite; };

// No foot in contact yet.
auto const space = State{Body{Attitude{}, Position{}, Feet{Foot{}, 0}}, Bias{}};
auto state = GaussianParams<State>{space, mean, Eigen::MatrixXd{0.3 * Eigen::MatrixXd::Identity(9, 9)}};
// no feet: 10 values, 9 rows

// A foot touches down at the end of a leg: the position plus the leg's vector, measured in the body frame.
auto const touchdown = Emrys::measurementModel(
    []<typename T>(auto const & in, auto const & x) {
        Eigen::Quaternion<T> const q{UnitQuaternion::map(block<Attitude>(in, x))};
        Eigen::Matrix<T, 3, 1> const leg = block<Leg>(in, x);
        return Eigen::Matrix<T, 3, 1>{block<Position>(in, x) + q * leg};
    },
    Emrys::reads(path<Attitude>, path<Position>, path<Leg>),
    Emrys::outputSpace(Foot{})
);
for (Eigen::Vector3d const & leg : {Eigen::Vector3d{0.2, -0.1, -0.6}, Eigen::Vector3d{-0.2, 0.1, -0.6}})
{
    Eigen::Index const count = state.space().template block<0>().vectorCount() - 1;   // the feet so far
    Emrys::insert<Emrys::TagPath<Body, Feet>>(
        state, count, methods::linearized(), touchdown, GaussianParams<Leg>{leg, legNoise}
    );
}
// two feet: 16 values, 15 rows
// the second foot: 1.5624 -1.7046 -0.0647
// the first foot's rows against the attitude's: minus the leg's cross-product matrix, times 0.3
//    0.0000 -0.1800  0.0300
//    0.1800  0.0000  0.0600
//   -0.0300 -0.0600  0.0000
Emrys::adjoint(state.space(), state.mean());   // 15 by 15
```

`insert<Path>(state, element, method, model, inputs...)` in `Models/Insertion.h` inserts what the
measurement model `model` would measure at the state, with the covariance and cross-covariance the
method gives. Among a pose's vectors the new rows are turned to the body frame: in them the foot's
step is $-l^\wedge \phi + \rho_p$ plus the leg's own error, with no rotation in it, which is the
block printed above. A model placed on each foot predicts beside the body's, a measurement reads
the run of feet, and a foot is erased:

```cpp
auto const motion = Emrys::blockwise(
    turning,   // turns the attitude and holds the position
    Emrys::onEach<Emrys::TagPath<Body, Feet>>(Emrys::RandomWalk<Foot>{0.05}),
    Emrys::RandomWalk<Bias>{0.001}
);
auto const predicted = Emrys::predict(methods::linearized(), state, motion, 0.4);
// the second foot after the prediction: 1.5624 -1.7046 -0.0647

// The feet seen from the body through the legs, y_i = R^T (f_i - p): reads(path<Attitude>, path<Position>,
// path<Body, Feet>), with an output of run-time size.
auto updated =
    Emrys::update(predicted, Emrys::observe(methods::linearized(), kinematics, z, Emrys::carries::adjoint()));
// trace of the covariance before and after the legs' reading: 5.137, 5.039

updated->branches[0].posterior.erase<Emrys::TagPath<Body, Feet>>(0);   // the first foot lifts
// one foot: 13 values, 12 rows
```

The run-time size needed no change to the design that the fixed size had not already needed: the
members belong to the instance, the parts are the user's tags, and `insert()` and `erase()` work on
a `Repeated` among the vectors as they do in a `Composite`. A test checks that erasing an inserted
foot restores the covariance, and that each foot's rows turn by $\hat{R}'^T \hat{R}$ in the
prediction.

The expression for the count of feet, `state.space().template block<0>().vectorCount() - 1`, is
the scratch program's: the pose's `vectorCount()` counts the position too.

> **Q3.** Inserting a vector as a function of the state. A new foot's position comes from the pose
> and the leg's kinematics, so it is correlated with the pose. `insert<Path>()` took a mean, a
> covariance, and a cross-covariance; nothing formed them from a model and its inputs.
>
> *Proposed.* The free function above. It acts on a `GaussianParams`; the other forms and a
> filter's own member are not written, so a `KalmanFilter` in contact-aided use has no way yet to
> gain a foot.

## 10. The example

`examples/invariant_ekf.cpp`: an inertial mechanization at 100 Hz with a gyro bias, and three
landmarks seen from the body at 10 Hz, for 15 s, 40 runs, each from a truth drawn from the prior.
The prior has the heading known to 0.6 rad, the tilt to 0.2 rad, the velocity to 1 m/s, and the
position to 2 m. One filter has the attitude, velocity, and position as an `ExtendedPose` and takes
each sighting by `observedInBody()`, so by the adjoint carry. The other has them as blocks of a
`Composite`, with the same process model object and the same prior values, and takes each sighting
by the same measurement written as a function, which the library differentiates.

```cpp
struct Body : Emrys::ExtendedPose<Attitude, Velocity, Position> {};
using GroupState = Emrys::Composite<Body, Bias>;
using BlockState = Emrys::Composite<Attitude, Velocity, Position, Bias>;

auto const motion = Emrys::blockwise(mechanization(gyroDensity, accelDensity), Emrys::RandomWalk<Bias>{biasDensity});
auto group = Emrys::KalmanFilter{motion, groupPrior};
auto blocks = Emrys::KalmanFilter{motion, blockPrior};

group.predict(dt, Emrys::sure<Gyro>(gyro), Emrys::sure<Accel>(accel));
auto const sight = Emrys::observedInBody<Body>(
    landmark,
    std::tuple{Emrys::on<Position>(1.0)},
    Emrys::outputSpace(Sighting{}),
    Emrys::additiveNoise(sightingVariance * Emrys::DifferenceMatrix<Sighting>::Identity())
);
group.update(Emrys::observe(sight, z));
```

The two libraries draw different samples from the same seed, so each prints its own numbers:

| State | Attitude error, RMS at 15 s | NEES per degree of freedom (1 is ideal) |
| --- | --- | --- |
| `ExtendedPose`, MSVC's library (Clang, MSVC) | 0.252 degrees | 0.95 |
| `Composite`, MSVC's library | 0.602 degrees | 38.0 |
| `ExtendedPose`, libstdc++ (GCC 14, and Clang 21 on Linux) | 0.252 degrees | 0.85 |
| `Composite`, libstdc++ | 1.556 degrees | 146.0 |

This is the literature's result and is shown here, not relied on. The program returns nonzero
when the invariant filter's estimates are not consistent with their covariances at the 99.9%
level, and it is the ctest `example.invariant_ekf`.

The heading prior was 1 rad until this review ran the example on Linux. There it printed a NEES of
4.38 for the pose and failed, where on Windows it had printed 1.12 and passed. The difference was
one draw. Each run's NEES per degree of freedom under libstdc++'s draws, at a heading prior of
1 rad:

| Runs | Initial attitude error | `ExtendedPose` | `Composite` |
| --- | --- | --- | --- |
| 38 of 40 | 0.12 to 1.75 rad | 0.32 to 2.16 | 0.32 to 1240; 10 runs above 5 |
| 1 | 2.26 rad | 3.82 | 106 |
| 1 | 3.07 rad | 138 | 3301 |

From a start that faces nearly backward, the invariant filter's covariance does not match its
error either. With a prior of 1 rad such a start is a 3-sigma draw, which 40 runs contain or not
by the library that draws them, so the example's prior is now 0.6 rad.

## 11. Evidence from before the prototype

Two scratch programs backed the design's first claims, on 2026-11-18 and 2026-11-19. Neither is in
the repository. The tests in `tests/Models/carries.cpp` repeat the second program's comparisons on
the prototype; nothing repeats the first.

**Sides.** Attitude and gyro bias, `Composite<Attitude, Bias>`, with an accelerometer and a
magnetometer reading reference vectors in the body frame, 1000 steps. One run used
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

**Carries.** $SE(3)$ with a known body-frame twist, $X' = X \cdot \mathrm{Exp}(u \, \Delta t) \cdot
\mathrm{Exp}(w)$, and either a position fix or three landmarks seen from the body, each with
anisotropic noise. The library ran `Linearized` on a right-side $SE(3)$ space written for the
test, with its reset by autodiff or turned off by `identityReset`. Hand-written filters ran on
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

## 12. What was checked, and what is missing

**Checked.** With Clang on Windows, on 2026-11-19, 96 test files were each compiled and run as a
program of their own, the allocation tests as one program, and each example was built and run:
every test file but the negative compile tests and `tests/assert_override.cpp`. That run found one
error, in a matrix model on a whole-state filter; after its fix, the two files it broke and
fourteen others near the fix were run again. After the three rewrites that GCC and MSVC asked for
on 2026-11-21, the five group test files and nine test files of `Models/` near the rewrites were
run again with Clang, and the six scratch programs of this review and the example were compiled
and run with Clang, GCC 14, and MSVC. clang-format, the layer check, the decisions check, the
Doxygen build, and the policy check over its XML pass.

**Not checked.** No test file has been compiled with GCC or MSVC, and this review's six programs
found three constructs those compilers reject, so expect more. The `clang` workflow has not run,
nor the negative compile tests, the sanitizers, or libc++.

**Missing**, beside what each question says:

- A closed form for the family's right Jacobian (§3).
- The carries in the square-root and information forms (§5), the world side from a square-root
  form's factor and with `affineIn()` factoring (§8), and both under an iterated sigma-point
  method.
- A negative compile test for each new compile error, which `tools/check_compile_errors.py` lists.
- `affineIn()` on a group block, and the benchmark of the invariant filter's step (§7).
- `insert()` by a model on the other forms and on a filter (§9).
- The messages and comments that still say "composition" for what `blockwise()` makes.
- The reference documentation's account of where the sigma-point filter departs from UKF-M (§8).
- The decisions: [lie-groups] loses the side and gains the family; [covariance-reset] gains the
  carry; [model-placement] and [unit-quaternion] take the new names; [affine-coordinates] and
  [fixed-step-models] change if `affineIn()` is extended.

To land the step: close the questions, finish what is missing, write the decisions, and run the
`land` checks, with GCC and MSVC first, since they have not seen the tests.

## Questions

Each is open. The prototype implements the answer marked *Proposed* where the question is raised.

1. How do separate models write parts of one group, and should a measurement's read of a part
   take only the smaller pose of the rotation and the vectors read? (§6)
2. What do `marginal<>`, a conditional, and an `on<>()` factor return for a part of a pose? (§4)
3. Inserting a vector as a function of the state: a free function on `GaussianParams` alone, or
   also the other forms and a filter's member? (§9)
4. Where does the carry apply, beside a measurement's update in the covariance form? (§5)
5. Names: `ExtendedPose` (§3); `carries::reset()`, `none()`, and `adjoint()`, or names that say
   the side (§5); `observedInWorld()` and `observedInBody()` in place of a general action (§7).
6. What must a rotation provide to head the family, and should every one-coordinate `Periodic`
   space be one? (§3)
7. The signatures of `LieGroup`'s members, and free functions named `exp` and `log` in the
   namespace `Emrys`. (§1)
8. How does a model say that its Jacobian does not depend on the estimate, how is $b$ spelled, and
   what backs the claim about cost? (§7)
9. How does a sigma-point method get the world side: the update's policy, a prediction's method,
   who builds the axes, and which pairs of root and carry compile? (§8)
10. Should the library turn a vector's covariance between the pose's rows and the world frame, for
    a prior going in and a covariance coming out? (§4)
11. A matrix of noise placed on a vector of a pose is body-frame, and on the same block of a
    `Composite` world-frame: refuse it, turn it, or leave it? (§6)
12. What is the carry under `Iterated`? (§5)
