# Rao-Blackwellized filters

Review notes on the filters that keep one block of the state at points and the rest Gaussian:
the Rao-Blackwellized point-mass filter, the Bayesian grid filter, and the Rao-Blackwellized
particle filter. The headers are `Distributions/Conditionals.h` (the covariance forms, the block
split, the stored conditionals, the collapse, and the conditioning on a drawn block),
`Distributions/GridGaussian.h` (`GridAxis`, `GridCells`, `PointCells`, `PointGrid`, and the sparse
`GridGaussianParams` state), `Distributions/ParticleGaussian.h`, `Filters/RbUpdate.h`
(the update given every point, in each form, and the weighting by a likelihood),
`Filters/RbPointMassFilter.h` (`RbPointMassFilter`, its result `RbPointMassFilterUpdate`, and
`PointMassFilter`), and `Filters/RbParticleFilter.h` (`RbParticleFilter` and
`RbParticleFilterUpdate`), tested in
`tests/Filters/rb_point_mass_filter.cpp` and `tests/Filters/rb_particle_filter.cpp`, with their allocation,
execution-policy, and time-step cases in `tests/allocation_particles.cpp`, `tests/parallel.cpp`,
and `tests/Filters/time_steps.cpp`. The update given a block, `update(prior, measurement,
given<Path>(space, value))` in `Models/Measurement.h`, is covered here too. Each section says what a piece does and shows it in use. Questions are marked
**Q*n*** and collected at the end.

The examples were refreshed for roadmap 17z (`next`, 2026-11-11), compiled and run
with Clang (MSVC STL); GCC and MSVC were not re-run for this refresh (GCC last on 2026-11-04, MSVC
on 2026-10-18). What roadmap 17z changed here: a process model is a function with `reads()` and
`writes()`, so a built-in such as `ConstantVelocity<Bias, Drift>{0.01}` is the model itself; a
measurement model comes from `measurementModel()` and is observed with a method,
`observe(method, model, z)`, except one known only by its likelihood, `measurementModel(reads(...),
logLikelihood(g))`, which is observed without one (§6); and `FactoredUnscented` is gone, so the
refusals of it are too, while the update given a block and the RBPF's prediction ignore a model's
`affineIn()` [affine-coordinates] (§2, §10). Every printed result of §2, §6, §8, §9, and §15 is the
same as on 2026-11-04. §10's NaN example, §12's parallel evidence, and §14's refills now state their
setup, which the earlier refreshes did not record, and print new digits for it. Earlier refreshes
found the update results owning their arrays (Q15), both filters taking a measurement
known only by its likelihood ([likelihood-measurements]), `subdivide()` and `collapsed()`
allocating nothing (Q18), and `GivenBlock` holding its space by value (Q2). Outputs in
comments are the printed results, from Clang. The deterministic examples printed the same under
GCC on 2026-11-04. GCC's `std::normal_distribution` draws differently, so the examples with
simulated noise give different numbers under GCC, of the same quality. Snippets marked as condensed
from a test file are covered by the test suite. Compiler errors are quoted from Clang.

The examples share these declarations:

```cpp
struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
struct Bias : Linear<1> {};   // a receiver's clock bias, in meters
struct Drift : Linear<1> {};  // its rate
using State = Composite<Position, Bias, Drift>;
using Rest = Composite<Bias, Drift>;
using Vector1 = Eigen::Matrix<double, 1, 1>;

auto const beacons = std::array<Eigen::Vector2d, 4>{
    {Eigen::Vector2d{-40, -30}, Eigen::Vector2d{60, -20}, Eigen::Vector2d{10, 70}, Eigen::Vector2d{-50, 50}}
};

// A range to a beacon plus the clock bias, 0.5 m noise: nonlinear in position, affine in the bias.
template <typename Method = methods::Linearized>
auto range(double z, Eigen::Vector2d const & beacon, Method const & method = {})
{
    auto const model = measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(d.squaredNorm())) + block<Bias>(in, x)(0)};
        },
        reads(path<Position>, path<Bias>),
        additiveNoise(GaussianParams<Linear<1>>{Vector1::Zero(), Vector1{0.25}})
    );
    return observe(method, model, Vector1{z});
}
// positionRange(z, beacon, sigma) is the same range from Position alone, reads(path<Position>), with
// noise of standard deviation sigma (1 by default), observed by methods::linearized().
```

Process models and measurements are the Models module page's [model-policies]; transforms, `withCross()`, and the
covariance update forms are the Transforms module page's; `ParticleSet`, resamplers, and `Roughen`
belong to the particle filter and upkeep reviews. Reduction policies (`Collapse`,
`Select`, `Gate`) and the shape every filter shares are the Kalman filter review's. Execution
policies, allocation, time, and printing in general are the Core module page's. This document uses
them without explaining them again.

## Decisions so far

Nothing has been decided in this review yet. These decisions from
[decisions.md](../design/decisions.md) are in place and should be taken as given:

- **The hybrid core.** The Rao-Blackwellized filters are the hybrid core's "fixed point of one
  block, enumerated" (the point-mass filter) and "nonlinear substate, sampled" (the RBPF) rows.
- **Likelihoods, [likelihood-measurements].** A measurement's likelihood at a point comes from the
  same measurement object that a Kalman filter uses. A measurement known only by its likelihood (a
  model with `reads()` and `logLikelihood(g)` and no function) is taken too, if every path it reads
  lies in the gridded or sampled block: it multiplies each point's weight and leaves the Gaussians
  alone, and a model that has `logLikelihood(g)` is weighted by it whatever its noise says (§6).
- **[rb-point-mass].** The Rao-Blackwellized point-mass filter fixes one block at points and never
  moves them. No probability passes between cells. Its process model covers only the rest of the
  state. The update for one point is `update(prior, measurement, given<Path>(space, value))`.
  `SharedCovariance` (one transform for every point) and `PerPointCovariance` (a Kalman filter per
  point, with a policy) are chosen by a template parameter. A cell's covariance about its center
  enters each update as an independent uncertain input, and the correlation that leaves between
  updates is accepted.
- **[upkeep].** Upkeep is function objects the caller applies to `filter.state()`.
- **[rao-blackwellized-core].** The Rao-Blackwellized states share one core and differ on three
  axes: what the points are, what kernel they carry, and whether the conditional covariance is
  shared. The store and the collapse are in `Conditionals.h`, the update given every point in
  `RbUpdate.h`. The grid state is sparse: only live cells are stored, and a dead cell is gone for
  good. `Subdivide` splits every live cell of a regular grid at once, so upkeep keeps the
  state's type.
- **[rb-particle-filter].** The RBPF predicts the whole state given each particle, draws the
  particle from the prediction's marginal, and conditions the rest on the draw. Its process model
  covers the whole state. A particle the policy rejects keeps weight zero rather than being removed.
- **[rb-point-mass].** The Bayesian grid filter is the point-mass filter whose block is the whole
  state, with a `Linear<0>` rest. It has no transition kernel, for the same reasons as
  [rb-point-mass].
- **[allocation].** The states, and the filters' scratch, allocate through the grid's or the state's
  allocator, and a step at a steady size allocates nothing (§14). A result sized by a count owns its
  data; the rule that let it be a view into filter scratch is gone.
- **[covariance-update-forms].** The shared update writes its one covariance in the measurement's
  covariance update form, as every other update does (§3).
- **[fixed-step-models], [time-steps].** A filter takes any time step and a fixed-step model, which
  then predicts with no step or with `Steps{n}` (§13).
- **[returned-jacobians].** A measurement that returns its Jacobian works in the update given a
  block; the Jacobian's columns are split between the rest and the given block (§2).
- **[covariance-reset].** The shared forms do not reset their one covariance at each point's mean;
  that is part of the shared approximation, and `PerPointCovariance` resets per point (§3).
- **[execution-policies].** Predict, update, and the RBPF state's resampling and weight reductions
  take an execution policy first; under `Sequential` they run what they ran before (§12).
- **[component-first].** `GridGaussianParams<Grid, Covariance, Component>` takes its component after the
  grid, which carries the space and the scalar, and defaults it to a Gaussian over the grid's rest;
  a `HybridGaussian` makes each cell an IMM filter (§6). `ParticleGaussianParams` takes its component
  last, after the scalar, checked against the rest and the scalar, and a `HybridGaussian` makes each
  particle an IMM filter (§9). `Conditionals` stores either per point (§16).
- **[filters-and-recursions].** The split into a state and a recursion was made for the Kalman,
  mixture, and IMM filters, not for these (§16, Q16).
- **[user-extension-points].** A user's reduction policy meets `ReductionPolicy`, which the
  per-point update's assertion now tests, and a user's likelihood is a measurement model's
  `logLikelihood(g)`; these filters have no recursion for a user to replace (Q16).
- **[view-stamp].** These states hand out no stamped views of their own (only `particles()`, a
  `ParticleSet` with its own); their maps and spans are the "Eigen-shaped surface" left to the
  sanitizer (§14). Their update results own their data (Q15).
- **[printing-and-simulation].** Both filters and both states print their size and their collapse
  (§15).
- **[innovation-cholesky].** The update given a block, and so every filter's `update()`, returns a
  `std::expected` that holds `GaussianError::NotPositiveDefinite`, with the state unchanged, when an
  innovation covariance is not positive definite; a cell or particle whose update fails fails the
  step (§2, §3). `PointMassFilter` is an `RbPointMassFilter`, so its `update()` returns one too.

## 1. One density, three filters

All three filters represent

$$p(x) = \sum_i w_i \, K_i(x_p) \, \mathcal{N}(x_r; \mu_i, \Sigma_i)$$

where the block $x_p$ named by a tag path is held at points $p_i$, $K_i$ is either a point mass at
$p_i$ or a Gaussian about it (the cell's *kernel*), and the rest of the state $x_r$ is Gaussian
given the point. They differ along [rao-blackwellized-core]'s three axes:

| | Points | Kernel | Conditional covariance | State type |
|---|---|---|---|---|
| `RbPointMassFilter` | fixed cells of a `PointGrid` | none, one shared, or one per cell | shared or per cell | `GridGaussianParams<Grid, Covariance>` |
| `PointMassFilter` | fixed cells; the block is the whole state | as above | (nothing is left Gaussian) | `GridGaussianParams<Grid, Covariance>` over `Linear<0>` |
| `RbParticleFilter` | particles, moved and resampled | none: particles are point masses | shared or per particle | `ParticleGaussianParams<Space, Path, Covariance, Scalar, Allocator>` |

What they share is in `Conditionals.h` and `RbUpdate.h` (§3): storing the
conditionals, the update given every point, collapsing to one Gaussian, and splitting a state
into the block and the rest. What differs is how points move (they don't, or by sampling), what
happens to a point whose likelihood is zero (a cell is removed, a particle keeps weight zero),
and how the state is kept in bounds (prune and subdivide, or resample and roughen).

The algorithms are published ones, with two exceptions. The grid is the point-mass filter of
[Bucy and Senne (1971)](../reference/bibliography.md#bucy1971), here over one block with the rest
marginalized: the Rao-Blackwellized point-mass filter of
[Šmídl and Gašperin (2013)](../reference/bibliography.md#smidl2013) and
[Duník et al. (2019)](../reference/bibliography.md#dunik2019), as given in
[Pocoma et al. (2023)](../reference/bibliography.md#pocoma2023), §II-D, a Kalman filter per cell.
Both predict by convolving the grid with a transition density; these filters don't (§7,
[rb-point-mass]). The exceptions have no source found: an IMM filter per cell under a switching
model, which composes two published filters, and a cell's covariance about its center. The particle filter is the Rao-Blackwellized particle filter of
[Doucet et al. (2000)](../reference/bibliography.md#doucet2000) and the marginalized particle
filter of [Schön, Gustafsson, and Nordlund (2005)](../reference/bibliography.md#schon2005), whose
§IV gives the prediction that draws the nonlinear block from the predicted joint and conditions
the linear one on the draw. The code takes that prediction but writes it as one transform of the
whole state rather than through the paper's linear sub-models (§9), and takes the paper's model-1
case, a gain shared by every particle, as the `SharedCovariance` form (§3).

The code uses several words for the block held at points. Public names say *pinned*
(`PinnedSpace`, `pinnedSpace()`, `pinnedRows()`), the overload says *given* (`given<Path>()`),
the Doxygen and decisions.md say *fixed block* for the grid filters and *sampled block* for the
RBPF, and the update functions say *point*. The other part is uniformly the *rest*.

> **Q1.** Should one word name the block across the API and the docs? *Pinned* is in every
> public type name; *fixed* reads better in prose but suggests constant, which a particle isn't.
> *Given* is the right word only at the update, where the block is conditioned on.

## 2. The update given a block

`update(prior, measurement, given<Path>(space, value))` is the per-point update. The prior is a
Gaussian over the rest of a state in `space`. The block named by `Path` comes from `value`: a
vector when it is known exactly (a cell's center, a particle), or a Gaussian, which is an
independent uncertain input (a cell's covariance about its center). The measurement reads blocks
of the whole state by tag path as usual, without knowing that one of them is given. The result
is the ordinary `MeasurementUpdate`, over the rest, in a `std::expected` that holds
`GaussianError::NotPositiveDefinite` instead when the innovation covariance is not positive
definite [innovation-cholesky].

```cpp
auto const prior = GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1.0, 0.1}, Eigen::Matrix2d{{2.0, 0.3}, {0.3, 0.5}}};
auto const p = Eigen::Vector2d{12, -7};
auto const updated = update(prior, range(55.0, beacons[0]), given<Position>(State{}, p));
updated->branches[0].posterior.mean();  // (-1.54176, -0.281263): the Kalman update with H = [1 0], S = 2.25
updated->branches[0].posterior.covarianceLower();   // [[0.222222, 0.0333333], [0.0333333, 0.46]]
updated->logLikelihood();               // -3.141426

// A Gaussian position widens the innovation by the range's sensitivity to it, J C Jᵀ = 0.541:
auto const c = Eigen::Matrix2d{{0.5, 0.1}, {0.1, 0.3}};
auto const position = GaussianParams<Position>{Position{}, p, c};
auto const widened = update(prior, range(55.0, beacons[0]), given<Position>(State{}, position));
widened->branches[0].posterior.mean();  // (-1.04888, -0.207332)
widened->logLikelihood();               // -2.896864
```

`tests/Filters/rb_point_mass_filter.cpp` checks both against the Kalman update by hand, to 1e-12.

Internally the measurement model becomes a function of the rest and the block,
`internal::PinnedMeasurement`, which holds the measurement resolved against the whole space,
assembles a whole state, and reads it by the model's `reads()`. The rest is marked with
`withCross()`; the block is an ordinary argument, known or uncertain, and the call's uncertain
inputs follow it. Because the given block is not conditioned, a later update sees it as
independent again (§6). A model whose function returns its Jacobian with its value
[returned-jacobians] works here too: the Jacobian over the paths read has its columns split into
the rest's and the given block's, and the update matches the one differentiated automatically,
known or Gaussian (`UpdateGivenABlockTakesAReturnedJacobian`).

Three limits. The transform covers the whole rest, not only the paths read (roadmap step 6).
Since [iterated-update], `Iterated` is rejected by a `static_assert` ("an update given a block does
not support Iterated yet"; §10). And a model's `affineIn()` [affine-coordinates] is ignored: under
a sigma-point method the update given a block spreads points over everything uncertain, the rest
and a Gaussian block or input, as the full transform does, which costs evaluations and is never
wrong. Until roadmap 17z
the factored method, `FactoredUnscented`, was refused here; it is gone, and the declaration that
replaced it is ignored instead.

`given()` returns a `GivenBlock`, which holds a copy of the space and a reference to the value,
since a space is empty or holds its sizes. Every use above passes a temporary `State{}`, which is
now safe stored as well, and a temporary value is refused, since its overload is deleted:

```cpp
auto const g = given<Position>(State{}, p);       // the space is a copy
auto const u = update(prior, range(55.0, beacons[0]), g);  // logLikelihood() -3.141426, as above
auto const h = given<Position>(State{}, Eigen::Vector2d{1, 2});
// clang: error: call to deleted function 'given'
```

> **Q2.** *Decided: both.* `GivenBlock` holds the space by value, so `State{}` is safe, and the rvalue
> overloads of `given()`'s value and `withCross()` are deleted (as `affineInput()`'s was, until
> roadmap 17z removed it), so a stored mark can't outlive what it borrows. ~~Should `Given` hold the space by value? Spaces are empty or hold one size, so a copy costs
> nothing, and it would leave only the value borrowed, which is what a reader expects of a function
> named like a view. The roadmap's API-robustness list now proposes the other half: delete the
> rvalue overloads of `withCross()`, `affineInput()`, and `given()`, as [view-stamp] deleted the
> view makers', and give `internal::Given` a public name (now `GivenBlock`). Deleting the rvalue overload for the space
> would reject every example above, which pass `State{}`, so the two go together.~~

## 3. The shared core

`Conditionals.h` holds the code that depends only on how the conditionals are stored, and
`RbUpdate.h` the update given every point. Apart from the two covariance tags,
all of it is in `internal`; users meet it through the two state types and the two filters. (The
first version of this document reviewed both as one header, `Rb.h`, which [layered-headers]'s
layering split: the store belongs to Distributions, the update, which needs measurements, to
Filters.)

### The two covariance forms

`SharedCovariance` and `PerPointCovariance` are empty tag types. As a filter's or state's
`Covariance` parameter they select the store:

| | `SharedCovariance` | `PerPointCovariance` |
|---|---|---|
| Store (`internal::Conditionals`) | one mean per point in a buffer from the allocator, read as a matrix, and one covariance | a buffer of `GaussianParams`s from the allocator |
| Update (`updateGivenShared` / `updateGivenEach`) | one transform at a reference point gives $S$ and the gain; each point costs an innovation and a mean update | each point is its own `update(prior, measurement, given)`, reduced by a policy |
| Noise | one additive Gaussian, and no uncertain input of the call | anything `update()` takes: mixtures, and noise through the function as an uncertain input |
| Policy | none | `Collapse` by default, or `Select`, a `Gate` over them, or a user's policy: any `ReductionPolicy`, which keeps one Gaussian |
| Covariance update form [covariance-update-forms] | the measurement's, applied once | the measurement's, per point |
| Reset at the mean [covariance-reset] | none: part of the shared approximation | per point, as the Kalman update does |
| Exact when | the dynamics and measurement are affine in the rest with the same Jacobian at every point; the measurement may depend on the block arbitrarily | always, as a Kalman filter per point is |

The reference point is the most probable one (ties go to the lowest). The store's `select()` and
`keep()`, which resampling, subdividing, and removing cells use, rebuild into a scratch buffer
and swap it in, so they allocate nothing once the points have been as many (§14).

A rough cost, measured on 2026-10-18 with `-O2` and Clang on this machine, with `Affine` ranges (the
method [returned-jacobians] later removed), no kernels, and the process models of §6 and §9. It has
not been re-run; benchmarks.md is where a cost claim belongs now.

| | `SharedCovariance` | `PerPointCovariance` |
|---|---|---|
| `RbPointMassFilter`, 10,201 cells, rest of size 2, predict + range update | 0.33 ms | 1.47 ms |
| `RbParticleFilter`, 5,000 particles, rest of size 2, predict + range update | 0.31 ms | 1.47 ms |

The same two tags also say whether a `PointGrid`'s cells have one covariance about their centers
or one each (§4). So a grid filter's type can carry both, with different meanings:
`GridGaussianParams<PointGrid<State, Position, Finite<Eigen::Dynamic>, PerPointCovariance>,
PerPointCovariance>` is a grid whose cells have kernels of their own, and whose conditionals
have covariances of their own.

> **Q3.** Should the kernel and the conditional covariance have separate tags? They are different
> axes in [rao-blackwellized-core], and the second implies nothing about the first except one way
> (cells with kernels of their own need a per-point filter, §6). A grid's could be, say, `NoKernel`,
> `SharedKernel`, and `KernelPerCell`, leaving `SharedCovariance` and `PerPointCovariance` for the
> conditionals.

### The shared update in detail

`updateGivenShared` transforms the measurement once, at the reference point with its kernel if any,
to get $\Sigma_{yy}$, the cross-covariance, $S = \Sigma_{yy} + R$, and the gain $K$, from a Cholesky
factorization of $S$; if $S$ is not positive definite, the update returns
`GaussianError::NotPositiveDefinite` before any point changes [innovation-cholesky]. Then, for each
point, it evaluates the measurement function at that point's mean and center, forms the residual
$r_i = z \boxminus h_i$ in the measurement space, and computes the NIS, the log-likelihood, and
$\mu_i \leftarrow \mu_i \boxplus K r_i$. The shared covariance is updated once, in the measurement's
covariance update form [covariance-update-forms]: standard, symmetric, or Joseph, whose $R$ is then
the noise plus the statistical linearization error of the measurement
(`SharedCovarianceUsesTheMeasurementsForm`). This is the model-1 measurement update of [Schön,
Gustafsson, and Nordlund (2005)](../reference/bibliography.md#schon2005), §IV.B, where the linear
substate's gain is the same for every particle; the code takes the one-gain structure and computes
the gain by the measurement's own method rather than from a linear $C$ the caller writes.

Each point's residual is $z \boxminus h(\mu_i, p_i)$, from the measurement at the mean. The
per-point form uses the method's predicted measurement instead, which under `Unscented`, or with a kernel, includes the
sigma points' mean correction. Inside the domain where the shared form is exact the two agree,
since there $f$ is affine in everything that is uncertain. Outside it, as with a range and a box
kernel, the shared form drops the correction at every point, including the reference.

> **Q4.** Is that worth a sentence in the Doxygen, or worth fixing? The reference point's
> correction, $\bar y_\text{ref} - f(\mu_\text{ref}, p_\text{ref})$, could be added to every
> point's residual for one subtraction each. That is no more exact elsewhere, but it makes the
> reference point's update match the per-point form's.

The shared form takes no policy, so it has no gate. Roadmap step 6 notes that a per-point gate on
the NIS would fit it, since gating only zeroes weights and leaves the shared covariance alone.

> **Q5.** Add `update(measurement, reductions::gate(...))` to the shared form, gating each point on its own
> NIS? A gated cell is then removed and a gated particle zeroed, as in the per-point form. The
> gate would be the only policy the shared form takes.

### Collapse, split, and conditioning

`collapseGivenPoints` moment-matches the whole state: the points' weighted spread plus the
weighted kernels for the block, the conditional means' spread plus the conditional covariances
for the rest, and the cross terms between them. Points of weight zero are left out, and means
are tangent-space means (the Spaces module page), so a periodic block collapses across its wrap
(§8).

`BlockSplit` finds the block's rows and the rest's, among the values and among the differences,
and `assemble()` rebuilds a whole state from the two. `pinnedConditioning` and `drawPinned` are
the RBPF's time update (§9): a semidefinite factor of the block's covariance, the gain
$G = P_{rp} P_{pp}^{+}$, and the rest's covariance given the block, from one LDLT of the block's
covariance. `drawPinned` moves the block and the rest by `plus()` in their spaces, and, like the
shared update, doesn't reset the one covariance at each draw [covariance-reset].

## 4. Points: `PointGrid`

`PointGrid<Space, Path, Cells, CellCovariance = SharedCovariance, Scalar = double, Allocator =
DefaultAllocator>` holds the fixed points of the grid filters: the whole space, the path of the
block, a discrete space of cells (`Spaces/Discrete.h`), one center per cell, and the cells' kernels.
The allocator holds the stored centers and kernels, and the `GridGaussianParams` over the grid takes it
(§14). Two functions build one, each with an optional allocator last:

| Builder | Cells | Centers | Subdividable |
|---|---|---|---|
| `regularGrid<Path>(space, first, spacing, counts, [kernel])` | `GridCells<Path>`, one `GridAxis<Block, I>` per coordinate of the block | computed: $\mathit{first} + c \cdot \mathit{spacing}$ | yes |
| `regularGrid<Path>(space, first, spacing, counts, kernels)` | `GridCells<Path>` | computed | no |
| `pointGrid<Path>(space, centers, [kernel])` | `PointCells<Path>`, one value per column | stored | no |
| `pointGrid<Path>(space, centers, kernels)` | `PointCells<Path>` | stored | no |

A single optional kernel gives `CellCovariance = SharedCovariance`; a `std::vector` of kernels,
one per cell, gives `PerPointCovariance`. The constructors are constrained on the regular/stored
split and on the kernel form, so each combination has exactly one.

```cpp
auto const first = Eigen::Vector2d{-10, 5};
auto const spacing = Eigen::Vector2d{2, 0.5};
auto const grid = regularGrid<State, Position>(first, spacing, {3, 4});
// PointGrid<State, Position, GridCells<Position>, SharedCovariance, double>; PinnedSpace Position;
// RestSpace Composite<Bias, Drift>; Regular
grid.size();                                      // 12, the number of cells
grid.center(1);                                   // (-10, 5.5): the last axis varies fastest
grid.center(4);                                   // (-8, 5)
grid.center(valueOf(grid.cells(), at<GridCells<Position>>(2, 3)));  // flat index 11, center (-6, 6.5)
grid.pinnedRows();                                // rows 0 to 1; restIndices() {2, 3}

auto const fine = grid.subdivided(2);             // 48 cells, first (-10.5, 4.875), spacing (1, 0.25)

Eigen::Matrix2d const box = Eigen::Vector2d{4.0 / 12, 0.25 / 12}.asDiagonal();  // a 2 x 0.5 box
auto const boxed = regularGrid<State, Position>(first, spacing, {3, 4}, box);
boxed.cellCovariance(7);                          // box, for every cell
boxed.subdivided(2).cellCovariance(0);            // box / 4: diag(0.0833333, 0.00520833)

auto const centers = Eigen::Matrix<double, 2, 3>{{-5, 0, 8}, {2, 0, -3}};
auto const points = pointGrid<State, Position>(centers);  // CellsType PointCells<Position>, 3 cells
```

The kernel is how a cell stands for an area rather than a point. For a regular grid with spacing
$s$, the box covariance $\mathrm{diag}(s^2 / 12)$ is the covariance of a uniform distribution
over the cell. §6 shows what it does to an update, and §8 what leaving it out costs.

A regular grid computes centers rather than storing them, so a grid of 650,000 cells costs the
same memory as one of 12. Centers are normalized by the block's space, which matters for a
periodic block. Nothing checks that a regular grid over a periodic block doesn't overlap itself:

```cpp
struct Heading : RadiansPi<> {};
struct Speed : Linear<1> {};
auto const heading = regularGrid<Composite<Heading, Speed>, Heading>(Vector1{-pi}, Vector1{pi / 2}, {6});
// centers -3.14159, -1.5708, 0, 1.5708, -3.14159, -1.5708: cells 4 and 5 repeat cells 0 and 1
```

> **Q6.** Should `regularGrid` assert that a periodic axis's span, `counts[i] * spacing[i]`, is
> at most the period? A duplicated cell double-counts its likelihood with no sign that it
> happened. Asserting equality as well would catch a grid that leaves a gap at the wrap, but a
> partial arc is a legitimate grid. Still unchecked on 2026-11-11.

The cells and their axes are tagged by the block the grid covers. A regular grid's cells are
`GridCells<Path>`, and axis $I$ of a block is `GridAxis<Block, I>`, `Block` being the innermost
tagged block that holds the coordinate: `GridAxis<Position, 0>` and `GridAxis<Position, 1>` above,
and `GridAxis<Heading>` and `GridAxis<Speed>` in the heading-and-speed test of §8
(`regularGrid<Motion>` with `Motion : Composite<Heading, Speed>`), where the heading's marginal is
`marginal<GridAxis<Heading>>()`. A point grid's cells are `PointCells<Path>`. A grid whose block
holds two blocks with the same tag would have two axes of one name, and doesn't compile.

> **Q7.** *Decided: the cells and every axis are tagged by the gridded block, as built.* The cells of a regular grid are `GridCells<Path>`, a class derived from the
> `DiscreteProduct` of its axes, and those of a point grid `PointCells<Path>`, derived from
> `Finite<Eigen::Dynamic>`, so a grid can be named in a path wherever its cells go. Axis $I$ of a
> block is `GridAxis<Block, I = 0>`, where `Block` is the innermost tagged block that holds the
> coordinate: `GridAxis<Heading>` and `GridAxis<Speed>` for
> `Motion`, `GridAxis<Position, 0>` and `GridAxis<Position, 1>` for `Position : Linear<2>`. The
> heading's marginal is `marginal<GridAxis<Heading>>()`, not `marginal<Heading>()`: `Heading` names
> the continuous block, and an exact tag keeps the path lookup a type match. Two grids in one product
> are told apart by the cells' tag, `TagPath<GridCells<Position>, GridAxis<Position, 0>>`.
> ~~Should a grid over a composite of one-dimensional tagged blocks name its axes by those
> tags? `GridCells` could be a `DiscreteProduct` of axes that derive from both `GridAxis<I>` and
> a tag wrapper of the block's tag, so both spellings work. [tag-addressing] answers the same about
> `axis<I>()`: positional. For a `Linear<2>` block there is nothing to name the axes by, so
> `GridAxis<I>` stays.~~

The rest's type is the space's `complementSubspace()` of the block (`internal::ComplementOf`), and
when one block is left it is that block's own type, not a composite of it. Over `Composite<X, Bias>`
gridded on `X`, the rest is `Bias`, so `GaussianParams<Composite<Bias>>` is the wrong conditional:
`particleGaussian<X>()` fails to deduce ("deduced type 'const GaussianParams<typename
internal::BlockSplit<...>::RestSpace, [...]>' ... does not match ... 'const
GaussianParams<Composite<Bias>, [...]>'"), and the point-mass filter finds no constructor. The type
to name is the grid's or the state's `RestSpace`, and the value `grid.restSpace()`.

> **Q19.** Should the rest's type be spelled for the user? A free alias such as
> `RestSpaceOf<State, Path>` would let a conditional be declared before the grid exists, and the
> error above would name it rather than `internal::BlockSplit`. Or should the complement of a
> composite always be a composite, so that `Composite<Bias>` is what the rest is?

## 5. The sparse grid state: `GridGaussianParams`

`GridGaussianParams<Grid, Covariance = SharedCovariance>` is the point-mass filter's state: the grid,
the flat indices of the live cells in increasing order, their normalized log-probabilities, and
their Gaussians over the rest. Its allocator is the grid's. Only live cells are stored. A cell
whose probability becomes zero (by a gate, by a prune, or by `setWeights()`) is removed,
since nothing passes probability between cells and so nothing could revive it.

A live cell has two addresses: its *position* $i$ among the live cells, and its *flat index* in
the grid's cells. Positions change whenever cells are removed or split; flat indices change only
when the grid is subdivided.

| By position $i$ | By flat index |
|---|---|
| `conditional(i)`, `center(i)`, `logWeight(i)`, `logWeights()`, `mostProbable()` | `cell(i)` (position to flat index), `find(cell)` (flat index to position, or nothing if dead), `cells()` |

Over the cells as a whole:

| Member | Gives |
|---|---|
| `grid()`, `size()` | the grid, and the number of live cells |
| `toDensity()` | the checked `GridGaussian`, whose `collapsed()` and `moments()` give the whole state as one `GaussianParams` (§3) |
| `marginal<Path>()` | a `CategoricalParams` over one axis of the cells, such as `GridAxis<Position, 0>`, summed over the live cells and normalized |
| `toCategorical()` | a dense, normalized `CategoricalParams` over every cell, dead ones zero |
| `setWeights(params)` | replaces the live cells' weights with a `CategoricalParams`'s, normalized over the live cells, for a prior that isn't uniform |
| `sharedCovariance()` | the conditionals' covariance, with `SharedCovariance` |
| `pruneBelow(logThreshold)`, `pruneTail(logFraction)`, `pruneToSize(n)` | remove the cells below $e^\ell$ (keeping the most probable), the least probable while their total stays below $e^\ell$, or the least probable beyond $n$, and renormalize; each returns the log of the probability removed; `upkeep::pruneBelow(p)`, `pruneTail(f)`, and `pruneToSize(n)` call them |
| `subdivide(k)` | splits every live cell into $k^d$; `upkeep::subdivide(k)` calls it |
| `get_allocator()` | the grid's allocator |

`logWeights()` is an `Eigen::Map` and `cells()` a `std::span` over the state's storage, so they
see later changes and dangle after a reallocation, unchecked (§14).

`toCategorical()` is dense, so its size is the grid's, not the state's. After the refinement run
in §6, 316 live cells sit in a grid of 451,584, and `toCategorical()` allocates all of them.
`marginal<GridAxis<Position, 0>>()` is sparse in what it reads and gives the axis's 672 values.

> **Q8.** Is a dense `toCategorical()` worth keeping once grids are refined? Its uses in the
> tests are comparisons: a probability per cell against a hand computation, and a sparse
> marginal against the dense one. A sparse view (flat indices and probabilities, which `cells()`
> and `logWeights()` already are) covers those.

### Subdividing

`subdivide(k)` replaces the grid by `grid.subdivided(k)`: spacing $s / k$, $k$ times the cells per
axis, and the kernel divided by $k^2$. The live cell with coordinates $c$ becomes the cells
$k c + a$, $0 \le a_j < k$, each with an equal share of its probability and a copy of its
Gaussian. Only a regular grid with no kernel or one shared kernel can subdivide; anything else is
a compile-time error (§10). Splitting chosen cells, which would need a kernel per cell, is an
open question in decisions.md.

The `Subdivide` action, what it does to the collapsed state with and without a box kernel (the
state is unchanged with one, and widens by $\tfrac{s^2}{12}(1 - 1/k^2)$ per axis without), and a
pruned and refined grid filter are on `Subdivide`, `GridGaussianParams::subdivide()`, and
`upkeep::Sequence`. [rao-blackwellized-core] decided that subdividing point masses stays allowed, and the caller decides whether it suits the grid. Unlike the filter's steps, `subdivide()` builds its
list of children in a `std::vector` on the global heap, not in the state's reused scratch (Q18).

> **Q20.** Refining a grid selectively [rao-blackwellized-core]. Splitting chosen cells needs
> per-cell covariances, and so a grid that can carry them through a split. Keeping the parents of
> split cells would let the children's spread reconsider the split later (coarsening), but needs
> more algorithm work. Replaying past measurements against new cells, instead of sharing the
> parent's probability, needs the measurement history and isn't upkeep. Splitting a cell is
> the same shape as splitting a mixture component (research.md item 6); research.md item
> 7 poses it as a research problem.

### Not a distribution

`GridGaussianParams` and `ParticleGaussianParams` give a collapsed Gaussian and per-point access, and nothing
else a distribution has. There is no marginal over continuous blocks (the rest's marginal is a
`GaussianMixtureParams` with the points' weights; the block's is a `CategoricalParams` over cells or a
`ParticleSet`), no `logDensity`, no sampling, no `toDensity()`, and no conversion. They are not
`Component`s [component-first], so a `MixtureParams` of them is not a thing. `Hybrid`, the other
type with one Gaussian per value, has `toMixture()`.

> **Q9.** Should the Rao-Blackwellized states get `marginal<Path>()` over the rest's blocks, as a
> `GaussianMixtureParams`, and the block's marginal as a `ParticleSet` (for particles) or the
> existing `CategoricalParams` (for cells)? A bias estimated by the point-mass filter is often
> multimodal while position is still ambiguous, and `collapsed()` hides that. `toMixture()` over the
> whole state exists only when every point has a kernel; without one, the block's factor is a point
> mass. Since [element-marginals] a mixture's, a hybrid's, and a particle set's marginal is a range
> that gives each element's marginal in turn, copying nothing; the answer here should be the same
> range, over the cells or particles. It needs its own element type, since a cell stores its mean
> beside a covariance it may share rather than a `GaussianParams` a marginal could borrow (roadmap
> 17l).

## 6. The Rao-Blackwellized point-mass filter

`RbPointMassFilter<Grid, Model, Covariance = SharedCovariance, Method = DefaultMethodFor<Model>>`,
with `PerPointCovariance` the default under a `SwitchingModel` (see Hybrid cells below), owns a process model over the rest of the state, a prediction method, a `GridGaussianParams` reached
through `state()`, and scratch for its updates. The default method is `Linearized` for a process
model and a fixed-step model's own for a `FixedStepModel` (§13). Its constructor takes the
model, the grid, one Gaussian over the rest that every cell starts with, and optionally a covariance
tag and a method:

```cpp
auto filter = RbPointMassFilter{model, grid, initial};                        // SharedCovariance, Linearized
auto filter = RbPointMassFilter{model, grid, initial, PerPointCovariance{}};
auto filter = RbPointMassFilter{model, grid, initial, SharedCovariance{}, methods::unscented()};
```

Every cell starts equally probable; `state().setWeights()` changes that. A user's reduction policy
works in the per-cell form if it meets `ReductionPolicy`, keeping one Gaussian
[user-extension-points]; the update's assertion tests that concept (§10).

- **`predict(dt, inputs...)`** moves each cell's Gaussian through the process model, with inputs
  passed as `predict()` takes them [prediction-inputs] and `dt` any time step ([time-steps], §13).
  The model must partition the rest of the state, which the constructor checks at compile time
  where the model's and the space's types decide it, and asserts otherwise (§10). The points and
  their probabilities don't change. With `SharedCovariance`, one prediction at the most probable
  cell gives the covariance, and every mean goes through the model's function at the inputs'
  means. With a fixed-step model it is `predict(inputs...)` or `predict(Steps{n}, inputs...)`. Each
  form takes an execution policy first (§12).
- **`update(measurement)`**, under either covariance form, or **`update(measurement, policy)`**
  (per-cell only, the policy no longer defaulted) is the update given every live cell (§3), followed
  by multiplying each cell's probability by $p(z \mid c)$ and normalizing. Per cell,
  `update(measurement)` collapses each cell's branches, as `Collapse` does. A cell the policy
  rejects is removed. If every cell has likelihood zero, the state is left as it was and the update
  reports not accepted. The Filters module page compares what rejection does in each filter. An
  execution policy goes first, before the measurement:
  `update(std::execution::par, fix, reductions::gate(...))`. A measurement known only by its likelihood takes
  the same `update(measurement)` and no policy (below).
- It returns an `RbPointMassFilterUpdate<Scalar, Allocator>` (the filter's `UpdateType`) in a
  `std::expected`, which holds `GaussianError::NotPositiveDefinite` instead, with the state
  unchanged, when a cell's innovation covariance is not positive definite [innovation-cholesky]
  (with a hybrid per cell, updated where it is, the cells before the failed one keep their update
  and the probabilities are as they were [update-outcomes]). The
  update holds `cells()` (the flat indices of the cells that were alive), and per cell
  `logLikelihoods()` and `nis()` (with mixture noise, the most probable branch's), then the total
  `logLikelihood()`, `innovationSize()`, and `accepted`. The result owns them, in storage from the
  filter's allocator; `update(fix, result)` (or `update(fix, policy, result)` per cell) refills a
  result the caller keeps, which allocates nothing once it has held as many cells (§14).

A receiver at an unknown position in a 100 m square, with a drifting clock bias, ranges to four
beacons at 0.5 m noise, and a coarse grid refined as the probability gathers:

```cpp
auto rng = std::mt19937_64{3};
auto normal = std::normal_distribution<double>{};
double const s = 5.0;
Eigen::Matrix2d const box = (s * s / 12) * Eigen::Matrix2d::Identity();
auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{s, s}, {21, 21}, box);
auto const initial =
    GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, 0}, Eigen::Vector2d{4, 0.01}.asDiagonal().toDenseMatrix()};
auto filter = RbPointMassFilter{ConstantVelocity<Bias, Drift>{0.01}, grid, initial};

auto const truth = Eigen::Vector2d{17.3, -8.6};
double bias = 1.0;
for (int k = 0; k < 12; ++k) {
  bias += 0.05;
  auto const & beacon = beacons[std::size_t(k) % 4];
  double const z = (truth - beacon).norm() + bias + 0.5 * normal(rng);
  filter.predict(1.0);
  auto const result = filter.update(range(z, beacon));  // Linearized; reads Position and Bias
  auto const pruned = upkeep::pruneBelow(1e-9)(filter.state());
  auto const refined = upkeep::whenSize(std::less{}, 100, upkeep::subdivide(2))(filter.state());
}
// k  0  cells updated 441  logL -4.420  after upkeep 161  spacing 5
// k  1  cells updated 161  logL -3.823  after upkeep 140  spacing 2.5
// k  2  cells updated 140  logL -4.679  after upkeep  92  spacing 1.25
// k  3  cells updated  92  logL -2.416  after upkeep 320  spacing 0.625
// ...
// k 11  cells updated 360  logL -0.618  after upkeep 316  spacing 0.15625

auto const & state = filter.state();
state.toDensity()->collapsed();                         // mean (17.744, -8.520, 1.389, 0.034), sd 0.259 0.267 0.378
                                           // truth (17.3, -8.6), bias 1.6
state.cell(state.mostProbable());          // flat index 302008, center (17.734, -8.672), P 0.088
state.marginal<GridAxis<Position, 0>>();   // 672 values, most probable 449, at x = 17.734
state.conditional(state.mostProbable());   // bias 1.326, sd 0.347
```

The numbers are the first version's, unchanged on 2026-11-11. The live count stays in the low
hundreds while the spacing falls from 5 m to 0.16 m, against 10,201 cells for a fixed 1 m grid.
The tests check the same scheme over 20 seeds (roadmap 8a). `PruneBelow` and
`upkeep::whenSize(std::less{}, ..., upkeep::subdivide(2))` are `Upkeep.h`'s, whose `Sequence` runs the same scheme on a grid
filter with nothing left Gaussian.

### Measurements known only by their likelihood

A measurement whose error is neither a Gaussian nor a mixture, such as a quantized range, enters
these filters through its likelihood alone [likelihood-measurements]. Such a measurement model
has `reads()` and `logLikelihood(g)`, and no function, noise, or method, and is observed without a
method, `observe(model, z)`; a model with a function that also has `logLikelihood(g)` is weighted
by the likelihood whatever its noise says. `update(measurement)` evaluates it at each point's
center (`internal::updateGivenLikelihood()`, under the execution policy), multiplies the weights,
and leaves every Gaussian as it was, since a likelihood that reads none of the rest tells it
nothing. Every path it reads must lie in the gridded block (`internal::pathLiesIn`); one that reads
the rest is a compile-time error (§10). A cell's kernel is ignored, so a likelihood narrower than
the cells needs finer cells, from `Subdivide`. This is the point-mass update of
[Bucy and Senne (1971)](../reference/bibliography.md#bucy1971), each cell multiplied by the
likelihood at its center, as [Bergman (1999)](../reference/bibliography.md#bergman1999), chapter 5,
writes it for a terrain map.

The receiver of the example above, with ranges rounded to multiples of 4 m and no other error, so
that $p(z \mid p)$ is $1/4$ within 2 m of the rounded range and zero beyond, on a fixed 1 m grid:

```cpp
// The gridded block only; g(in, x, z) reads the state as a function would, and takes the observation.
auto quantizedRange(double z, Eigen::Vector2d const & beacon)
{
    return observe(
        measurementModel(
            reads(path<Position>),
            logLikelihood([beacon](auto const & in, auto const & x, auto const & observed) {
                return std::abs((block<Position>(in, x) - beacon).norm() - observed(0)) <= 2.0
                         ? -std::log(4.0)
                         : -std::numeric_limits<double>::infinity();
            })
        ),
        Vector1{z}
    );
}

auto grid = regularGrid<State, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{1, 1}, {101, 101});
auto filter = RbPointMassFilter{ConstantVelocity<Bias, Drift>{0.01}, grid, initial};
for (std::size_t b = 0; b < 3; ++b)
{
    auto const result = filter.update(quantizedRange(4 * std::round((truth - beacons[b]).norm() / 4), beacons[b]));
}
// quantized 0: cells updated 10201, logL -4.390, innovationSize() 0, nis() all NaN, live 506
// quantized 1: cells updated 506, logL -3.231, innovationSize() 0, nis() all NaN, live 80
// quantized 2: cells updated 80, logL -3.283, innovationSize() 0, nis() all NaN, live 12
filter.state().toDensity()->collapsed();                  // position (16.750, -9.750), bias 0.500; truth (17.3, -8.6)
filter.state().conditional(0);               // bias 0.500, variance 4.000: the initial Gaussian
static_cast<void>(filter.update(range(z, beacons[3])));  // z with a bias of 1.05: bias 0.652
```

The cells outside every annulus get likelihood zero and are removed, and each cell's Gaussian is the
one it started with until a measurement that reads the rest. The tests check a grid filter against
the Bayes posterior under a quantization triangle convolved with Gaussian noise, by quadrature,
under both covariance forms (`TakesAMeasurementKnownOnlyByItsLikelihood`), that the Gaussians are
untouched (`ALikelihoodOverTheGridLeavesTheRest`), and that the RBPF's Gaussians are too
(`ALikelihoodOverTheParticlesLeavesTheRest`). A likelihood takes no reduction policy, since it has
no branches: `update(measurement, policy)` refuses one with a `static_assert` (§10). The precedence
changed the grid filters: a measurement with Gaussian noise that also supplies a likelihood was
given a Kalman update before 2026-11-02, and is weighted by its likelihood now, or refused if it
reads the rest.

> **Q23.** A likelihood is evaluated at each cell's center and its kernel ignored, while a Kalman
> update widens the innovation by the kernel (below). On a grid with box kernels the same range is
> then weighed two ways: as a likelihood it is as sharp as on point masses, which §8 shows takes all
> the probability to one cell when the cells are coarse against it. Should a cell with a kernel
> average the likelihood over it (by the method's sigma points, say), should a model known only by
> its likelihood be refused on a grid with kernels, or is the Doxygen's advice, finer cells by `Subdivide`, the
> answer?

### The cell's kernel

A kernel enters each update as an independent uncertain input: the method transforms it with the
rest, so the innovation is widened by $J C J^T$, with $J$ the measurement's Jacobian in the block. A
measurement nonlinear in the block therefore needs `Linearized` or `Unscented` (or a sigma-point
set, [sigma-point-sets]); the `Affine` method's unit-step secant, before [returned-jacobians]
removed it, widened it by the wrong amount. Two consequences, both recorded in [rb-point-mass]:

- The offset of the true value from the center is not tracked, so each update treats it as
  new. Successive measurements are treated as independent although the shared offset correlates
  them, so as measurements accumulate the cell weights grow sharper than the cell's extent
  justifies.
- With `SharedCovariance`, the kernel enters only at the reference cell, so the widening is
  exact only if $J$ is the same in every cell. For ranges it isn't, and the most probable cell's
  stands for all. A grid with a kernel per cell requires `PerPointCovariance`, checked at compile
  time.

> **Q10.** Is there a case for tracking the offset instead? Put $\delta = x_p - p_c$ in the rest
> with prior $\mathcal{N}(0, C)$, and evaluate the measurement at $p_c + \delta$. The update
> then conditions $\delta$ along with everything else, so the correlation between updates is
> kept, and cells of different sizes become per-point conditional covariances rather than a
> separate kernel concept. The costs are $d$ more coordinates in every Gaussian, a Gaussian
> rather than uniform shape within the cell, and a `Subdivide` that has to split $\delta$ too.
> If the answer is no, it may be worth a sentence in [rb-point-mass] saying it was considered.

> **Q22.** How should a caller tell whether a grid's collapse describes it? The library offers
> no test. Two that a caller can build both fail on a grid. `kullbackLeiblerCollapseBounds()` over
> the cells, each read as a Gaussian with its kernel's covariance, is loose by up to the entropy
> of the weights, several nats for thousands of cells, so it never vouches for a grid. Mardia's
> skewness and kurtosis over the same Gaussians misread the box cells' shape. In a TDOA
> comparison (2026-11-01, 100 runs), they trusted 15 to 23% of nominal runs that particles and
> a mixture on the same data trusted almost always, and the rate rose with finer refinement. A
> test from a box kernel's exact moments may be the answer. The lower end of the bracket, the
> bound less the weights' entropy, which `kullbackLeiblerCollapseBounds()` now returns, can say
> that a grid is far from its collapse, but not that it is close. The over-sharpening of the
> cell weights (Q10) may also contribute.

### Hybrid cells

Under a `SwitchingModel` over the rest, each cell holds a `HybridGaussian` over the rest, an IMM
filter's state, and the filter is an IMM filter per cell given the cell's center. Condensed from
`tests/Filters/rb_point_mass_filter.cpp`, with `delayed(z, beacon)` a range whose noise has one
component per value of `Clock`:

```cpp
struct Steady {};
struct Wander {};
struct Clock : Named<Steady, Wander> {};
auto const clock = SwitchingModel{
    MarkovChain<Clock>::persistent(0.9), ConstantVelocity<Bias, Drift>{0.01}, ConstantVelocity<Bias, Drift>{1.0}};
auto const prior = HybridGaussian<Rest, Clock>{CategoricalParams<Clock>::fromWeights(Clock{}, Eigen::Vector2d{0.7, 0.3}), initial};
auto filter = RbPointMassFilter{clock, grid, prior};  // PerPointCovariance, deduced from the model
filter.predict(1.0);                        // each cell: mix, predict each mode, step the chain
filter.update(range(z, beacon));            // each cell: an IMM update given its center
filter.update(delayed(z, beacon));          // noise byMode<Clock>(): the components are Clock's values
filter.state().conditional(i);              // HybridGaussian<Rest, Clock>
filter.state().modes();                     // P(Clock), summed over the cells
filter.state().toDensity()->collapsed();                 // GaussianParams over the whole state
```

The model chooses the component: the state is `GridGaussianParams<Grid, PerPointCovariance,
Hybrid<Gaussian<Rest>, Clock, Allocator>>`, from the grid's allocator, and the covariance form
defaults to `PerPointCovariance`, the only form a hybrid takes, since mixing gives each cell
covariances of its own; `SharedCovariance` with a hybrid fails a `static_assert`. The steps are
`ImmRecursion`'s. `predict()` runs `ImmRecursion::predict()` on each cell. `update()` runs
`ImmRecursion::update(state, measurement, given<Path>(space, value), policy)`, the hybrid
counterpart of `update(prior, measurement, given)`, on each cell where it is, with no copy of the
cells, since the recursion leaves a cell no mode accepts as it was [update-outcomes], and multiplies the
cell's probability by the IMM's evidence $p(z \mid c) = \sum_m P(m \mid c) \, p(z \mid m, c)$,
as a `MixtureFilter` of IMMs reweights a component. A cell that no mode accepts gets likelihood zero
and is removed; a cell's NIS is its most probable mode's after the update. A measurement known only
by its likelihood weights the cells and leaves the hybrids alone, and pruning and subdividing carry
each cell's hybrid. The test `HybridCellsMatchAnImmFilterPerCell` checks every cell, over six
updates, plain and by mode axis, against an `ImmFilter` on the rest with the cell's position built
into the measurement.

The cells share the recursion's scratch, so they run in order: `predict()` and every update but one
by a likelihood alone take `Sequential` and refuse any other policy at compile time (Q25). An update
allocates the IMM result that it reuses from cell to cell (Q26).

### Spelling

The filter departs from the other filters' parameter conventions in two ways. Its template
parameters are `<Grid, Model, Covariance, Method>`: the space, the path, the scalar, and the
allocator live in the grid's type, so there is neither a space first nor a scalar last. The
component follows from the model, a `HybridGaussian` under a `SwitchingModel` and a Gaussian
otherwise. `GridGaussianParams<Grid, Covariance, Component>` takes the grid first too. `RbParticleFilter`, by contrast, is
`<Space, Path, Model, Covariance, Method, Scalar, Allocator>`. And the constructors differ: the
RBPF takes `(model, state, [method])`, while the point-mass filter takes `(model, grid,
conditional, [covariance], [method])`, so choosing a method means also spelling the covariance
tag, even the default.

[component-first] has since added a clause to style.md's convention: a type that holds a
distribution per element takes that component first, since it carries the space and the scalar.
The grid carries them in `GridGaussianParams`, so a component first would repeat the rest's space; the
decision now says that a type built over such a structure takes the structure first and defaults
its component from it, `GridGaussianParams<Grid, Covariance, Component = Gaussian<Grid::RestSpace>>`.

> **Q11 (partly decided, [component-first]).** Should the two filters be built the same way? One
> option is for the point-mass filter to take `(model, GridGaussianParams, [method])` as the RBPF takes
> its `ParticleGaussianParams`, with `GridGaussianParams` built from `(grid, conditional, [covariance])` as it
> is now. That lets a non-uniform prior be set before the filter exists, and makes the method the
> only optional argument. The template parameters could stay as they are, since a grid really does
> carry its space and scalar. [component-first]'s clause now records that a type built over a
> component takes it first, and that a type built over a grid takes the grid first; the
> constructor's shape is still open.

## 7. Why there is no transition kernel

The classic point-mass filter of
[Bucy and Senne (1971)](../reference/bibliography.md#bucy1971) predicts by convolving the grid's
probabilities with the transition density of the gridded block, moving probability from cell to
cell, and so does [Bergman (1999)](../reference/bibliography.md#bergman1999)'s terrain filter.
These filters deliberately don't [rb-point-mass]:

- It costs a convolution over the whole grid every step, where the update costs one innovation
  per live cell.
- It smears the density by about one cell spacing per step even when the true motion is
  smaller, so a fixed grid can't hold a sharp estimate of something that moves slowly.
- It revives cells that pruning removed, which defeats the sparse state (§5) and refinement.

Motion goes elsewhere: into the Gaussian part (a bias random walk, a drift) for the point-mass
filter, or into the particles, which move freely, for the RBPF. The point-mass filter then suits a
block that doesn't move: a receiver's position over a session, a fixed sensor offset, a terrain
match's initial position. The grid of Kalman filters, each with exact cell weights for any
measurement nonlinear in the block, is what's left when the convolution is removed.

That leaves out one common use. In terrain-aided navigation, the gridded block is a vehicle's
position, which moves by the dead-reckoned displacement between fixes. The classic answer
translates the whole grid by the known displacement, which costs nothing on a regular grid
(`first` changes) and moves no probability between cells, and widens the cells for the
displacement's uncertainty.

> **Q12.** Should a grid be able to move as a whole? Two pieces would cover terrain-aided
> navigation without a convolution: translating every point by a known displacement (exact,
> $O(1)$ for a regular grid, $O(n)$ for stored centers), and growing a shared kernel by the
> displacement's covariance (a diffusion that moves no probability, and keeps the cells alike so
> `SharedCovariance` stays valid). The alternative that works today is to grid the
> dead-reckoning *error*, which is static or slowly drifting, and capture the dead-reckoned
> position in each measurement. If that is the intended answer, it is worth an example in the
> Doxygen. (Under the 0.1 algorithm freeze, the first answer would wait.)

## 8. The Bayesian grid filter: `PointMassFilter`

`PointMassFilter<Grid, Covariance = SharedCovariance>` is
`RbPointMassFilter<Grid, NoProcessModel, Covariance, methods::Linearized>` for a grid whose path
names the whole state. The rest is `Linear<0>`, a Gaussian over nothing, and `NoProcessModel`
writes nothing, so it partitions that empty rest. Its density is $p(x) = \sum_c P(c) \, \delta(x - p_c)$, `update()` multiplies each cell by $p(z \mid c)$, and
`predict()` changes nothing. It is built from the grid alone, and optionally a covariance tag:

```cpp
auto const first = Eigen::Vector2d{-50, -50};
auto const spacing = Eigen::Vector2d{5, 5};
auto const grid = regularGrid<Composite<Position>, Position>(first, spacing, {21, 21});
auto filter = PointMassFilter{grid};
// decltype(filter): RbPointMassFilter<PointGrid<...>, NoProcessModel, SharedCovariance, Linearized>
// ConditionalType:  GaussianParams<Linear<0>>

auto const truth = Eigen::Vector2d{17.3, -8.6};
for (std::size_t b = 0; b < 4; ++b) {
  filter.predict(1.0);  // does nothing
  if (!filter.update(positionRange((truth - beacons[b]).norm(), beacons[b]))) { ... }  // exact ranges, σ = 1 m
}
filter.state().toDensity()->collapsed();  // (15.015, -9.997), sd 0.274 0.119; truth (17.3, -8.6)
                             // the cell at (15, -10) has P 0.997
```

The estimate sits on the nearest cell with a standard deviation ten times smaller than its
error. With point-mass cells 5 m apart and 1 m noise, one cell explains the ranges much better
than its neighbors, so it takes all the probability. The same grid with a box kernel, and
`Linearized` ranges so that the kernel is widened by the range's Jacobian:

```cpp
Eigen::Matrix2d const box = (25.0 / 12) * Eigen::Matrix2d::Identity();
auto boxed = PointMassFilter{regularGrid<Composite<Position>, Position>(first, spacing, {21, 21}, box)};
// after the same four ranges: (15.931, -9.610), sd 2.423 1.973; the best cell has P 0.807
```

The error is now within one standard deviation. A point-mass grid is consistent only when its
spacing is small against the likelihood's width; a kernel, or `Subdivide` as in §6, is the way
to use a coarse one.

`Periodic` blocks need nothing of their own. Condensed from
`tests/Filters/rb_point_mass_filter.cpp`: a 72 × 20 grid over a heading in $[-\pi, \pi)$ and a speed in
$[0, 10)$, given two fixes at a heading of $\pi - 0.02$, puts most of the probability in the cells
on both sides of the wrap, and `collapsed()` takes the tangent-space mean, so the estimate's
heading is within 0.05 of $\pm\pi$, not the average of the grid's ends.

`PointMassFilter filter{grid}` and `RbPointMassFilter filter{grid}` deduce the same
type (on all three compilers, on 2026-10-18; `DeducesTheDefaultsWithoutACovarianceForm` checks it).
The grid-only constructor requires a `Linear<0>` rest and `NoProcessModel`, so a grid over part of
the state is rejected (§10). A `PointMassFilter` takes no input to `predict()`: an input there is
refused by a `static_assert`, "a point-mass filter over the whole state takes no input"
(`POINT_MASS_PREDICT_INPUT`).

> **Q13.** The type is named `PointMassFilter`, while the roadmap, [rb-point-mass], and the type's
> own Doxygen call it a Bayesian grid filter. In the literature the point-mass filter is the one
> with the convolution prediction this filter lacks (§7), so the name promises the one thing it
> doesn't do. Would `GridFilter` or `BayesianGridFilter` be clearer? Separately, `predict()` exists
> and does nothing. That keeps generic code that calls `predict()` working, but it may also suggest
> that motion is handled. Keep it, or remove it so the omission is visible?

## 9. The Rao-Blackwellized particle filter

### The state: `ParticleGaussianParams`

`ParticleGaussianParams<Space, Path, Covariance = SharedCovariance, Scalar = double, Allocator =
DefaultAllocator>` holds a `ParticleSet` over the block named by `Path`, and for each particle a
Gaussian over the rest. It is the marginalized particle representation of
[Doucet et al. (2000)](../reference/bibliography.md#doucet2000) and
[Schön, Gustafsson, and Nordlund (2005)](../reference/bibliography.md#schon2005). It is built two
ways:

- `particleGaussian<Path>(space, particles, conditional, [covariance])`: given particles, each
  starting with the same Gaussian over the rest. Constructors also take one mean per particle
  with a shared covariance, or one Gaussian per particle. The state's allocator is the
  particles'.
- `drawParticleGaussian<Path, [Covariance]>(joint, count, rng, [covariance], [allocator])`: the
  Rao-Blackwellized form of a Gaussian over the whole state. Each particle is drawn from the
  block's marginal, and its Gaussian is the rest conditioned on the draw, with one covariance for
  all of them.

```cpp
using Motion = Composite<Position, Velocity>;
auto rng = std::mt19937_64{11};
auto const prior =
    GaussianParams<Motion>{Motion{}, Vector4{1, -1, 0, 0}, Vector4{9, 9, 4, 4}.asDiagonal().toDenseMatrix()};
auto initial = drawParticleGaussian<Position>(prior, 500, rng);  // ParticleGaussianParams<Motion, Position>
std::sqrt(initial.sharedCovariance()(0, 0));                     // 2: position and velocity uncorrelated here
```

It offers what `ParticleSet` does for weights (`logWeights()`, `logTotalWeight()`,
`ess()`, `mostProbable()`, the last three also under a policy, §12), per-particle
access (`particle(i)`, `conditional(i)`, `particles()`), `toDensity()` for the collapse, `setCapacity()` (§14), and
`resample([policy], rng, [size], [resampler])`, which copies each particle's Gaussian from its
ancestor. `setParticleStates()` replaces the particles' values and leaves their Gaussians, and is
what `Roughen` uses. So `upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))` and `upkeep::roughen(rng, scale)` apply to it unchanged
[upkeep].

`Roughen` leaves each particle's Gaussian where it was, although the Gaussian's mean was conditioned
on the particle's old value. [rb-particle-filter] records this as an approximation, "as roughening
always is". It prunes as a `ParticleSet` does, each particle's Gaussian removed with it,
though particles are usually kept in bounds by resampling.

### The filter

`RbParticleFilter<Space, Path, Model, Covariance = SharedCovariance, Method =
DefaultMethodFor<Model>, Scalar = double, Allocator = DefaultAllocator>` owns a process model over
the whole state, a prediction method, a `ParticleGaussianParams`, and scratch for the draws and the
updates. It is built as `{model, state, [method]}`, with everything deduced
(`DeducesItsTypeAndHoldsItsParts`). One deduction guide, with the method defaulted and the
parameters in the class's order, is what lets MSVC deduce through the alias.

**`predict(rng, dt, inputs...)`** moves each particle $p_i$ with its Gaussian. It transforms the
Gaussian over the rest through the process model with the particle as a known input, and adds
$Q$, which gives a Gaussian $(m, P)$ over the whole new state. It draws the particle's new value
$p' = m_p \boxplus F z$ with $F F^T = P_{pp}$ (the default root, pivoted where $P_{pp}$ is singular,
so coordinates without noise work), and conditions the rest on the draw:

$$\mu' = m_r \boxplus G F z, \qquad \Sigma' = P_{rr} - G P_{pr}, \qquad G = P_{rp} P_{pp}^{+}$$

The weights don't change. The draw's offset is information about the rest: a velocity is learned
from the positions it moves, which is the marginalized particle filter's time update ([Schön,
Gustafsson, and Nordlund, 2005](../reference/bibliography.md#schon2005), §IV). The paper writes it
through linear sub-models of the nonlinear and linear substates; the code gets the same quantities
from one transform of the whole state, so any process model and method serve [rb-particle-filter].
That transform ignores a model's `affineIn()` [affine-coordinates], as the update given a block
does (§2): under a sigma-point method it spreads points over the whole Gaussian, which costs
evaluations and is never wrong.
With `SharedCovariance`, one transform at the heaviest particle gives $F$, $G$, and $\Sigma'$ for
every particle, and each particle propagates only its mean. The draws are taken first, one standard
normal vector per particle, so the two forms consume the generator identically, and a parallel
prediction draws as the sequential one does (§12). With a fixed-step model whose transition is
cached and no inputs, the prediction given each particle is $F_r P_i F_r^T + Q$, with $F_r$ the
transition's columns for the rest, rather than a transform (§13).

**`update(measurement)`**, under either covariance form, or **`update(measurement, policy)`**
(per-particle only) is the update given every particle (§3), and multiplies each weight by $p(z \mid
i)$; a measurement known only by its likelihood of the sampled block weights the particles and
leaves their Gaussians, as in the point-mass filter (§6). A particle the policy rejects keeps its
Gaussian with weight zero; resampling removes it. If no particle can explain the measurement, the
state is unchanged. It returns a `RbParticleFilterUpdate<Scalar, Allocator>` in a `std::expected`,
which fails as the point-mass filter's does [innovation-cholesky]. The update is like the point-mass
one without `cells()`, since particles have no identity beyond their position in the set; it owns
its arrays too, and the same refill overloads reuse a kept result (§14).

Ranges to three beacons say nothing directly about velocity. Here the particles learn it from
their own motion:

```cpp
auto filter = RbParticleFilter{ConstantVelocity<Position, Velocity>{0.01}, std::move(initial)};
// decltype(filter) is RbParticleFilter<Motion, Position, Kinematic<double, Position, Velocity>>
auto const resample = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
auto truth = Vector4{2, -2, 1.5, -1.0};
for (int k = 1; k <= 20; ++k) {
  truth.head<2>() += truth.tail<2>();
  filter.predict(rng, 1.0);
  for (std::size_t b = 0; b < 3; ++b) {
    double const z = (truth.head<2>() - beacons[b]).norm() + 0.5 * normal(rng);
    if (!filter.update(positionRange(z, beacons[b], 0.5))) { ... }  // Linearized; reads Position only
  }
  auto const changed = resample(filter.state());
}
// k  5  pos err 1.904  vel  1.372 -0.615  vel sd 0.124 0.133    (truth 1.5, -1.0)
// k 10  pos err 0.426  vel  1.756 -1.444  vel sd 0.150 0.157
// k 15  pos err 0.214  vel  1.538 -1.036  vel sd 0.150 0.186
// k 20  pos err 1.470  vel  1.341 -0.769  vel sd 0.159 0.161
// resampled 18 times; each particle's velocity sd (shared) 0.054
```

The velocity's standard deviation falls from 2 m/s to about 0.16. Most of the remaining spread is
between the particles' conditional means, not within each Gaussian.

This run also shows a weakness the first version's seed hid. At $k = 20$ the position error is
1.47 m against a collapsed standard deviation of about 0.31 m, nearly five of them. With
process noise this small on the sampled block and resampling at almost every step, the
particles impoverish: many copies of few positions, whose spread understates the error. Over
seeds 1 to 5 the error at $k = 20$ is 0.10 to 0.77 m against about 0.28; adding
`upkeep::roughen(rng, 0.2)` to the trigger (`upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng, 0.2))`) brings seed 11 to
0.27 m and seeds 1 to 5 to 0.08 to 0.47, all within 1.5 standard deviations. The test
`LearnsAVelocityFromRanges` checks the velocity, within three of its standard deviations, and the
position within 1.5 m, not the position's consistency. The class's Doxygen example and
particle-filter.md's roughening section are where a reader would look for that advice.

### Hybrid particles

Under a `SwitchingModel` over the whole state, each particle holds a `HybridGaussian` over the
rest, an IMM filter's state, and the filter is an IMM filter per particle given its value. The
state is `ParticleGaussianParams<Space, Path, PerPointCovariance, Scalar, Allocator,
HybridGaussian<Rest, Mode>>`, which `particleGaussian<Path>(space, particles, hybrid)` builds and
whose `modes()` sums the particles' modes. Condensed from `tests/Filters/rb_particle_filter.cpp`,
where the modes differ in their process noise and `makeBiasedRangeRate()` has one noise component
per value of `Mode`:

```cpp
auto const motion = SwitchingModel{
    MarkovChain<Mode>::persistent(0.9), ConstantVelocity<Position, Velocity>{0.01}, ConstantVelocity<Position, Velocity>{2.0}};
auto filter = RbParticleFilter{motion, particleGaussian<Position>(State{}, ParticleSet<Position>{positions}, hybrid)};
filter.predict(rng, 1.0);                         // std::expected<void, NumericError>
filter.update(makeRangeRate(z, beacon));          // each particle: an IMM update given its position
filter.update(makeBiasedRangeRate(z, beacon));   // noise byMode<Mode>(): the components are Mode's values
filter.state().conditional(i);                    // HybridGaussianParams<Rest, Mode>
filter.state().modes();                           // P(Mode), summed over the particles
```

The prediction is the IMM's and the marginalized particle filter's together. For each particle it
mixes the modes' Gaussians through the chain as `ImmRecursion::predict()` does, and predicts the
whole state given the particle through each mode's model, which gives the joint predictive a
mixture over the modes, $\sum_j \pi_j \, \mathcal{N}(m_j, P_j)$ with $\pi$ the chain's predicted
mode probabilities. It draws the particle's new value from that mixture, a mode $k$ from $\pi$ by
one uniform draw and then $p' = m_{k,p} \boxplus F_k z$, and conditions every mode's Gaussian over
the rest on $p'$. Since the draw comes from the predictive, the weight doesn't change, and what the
draw says about the mode goes into the mode probabilities,
$P(j \mid p') \propto \pi_j \, \mathcal{N}(p'; m_{j,p}, P_{j,pp})$, as what it says about the
block's motion goes into the rest's mean. A mode the chain reaches whose block covariance $P_{j,pp}$
is singular gives the draw no density, so `predict()` returns `NotPositiveDefinite` with the state
unchanged: the hybrids are built in the posterior store and swapped in at the end. The test
`HybridParticlesMatchAnImmFilterPerParticle` checks every particle over four steps against an
`ImmFilter` over the whole state, started from the particle known and given its draw almost
exactly, and each update against `ImmRecursion`'s with the particle's position built into the
measurement. With one model shared by every mode and every mode starting from the same Gaussian,
the filter matches the filter of one Gaussian per particle, continuous and fixed-step
(`HybridParticlesOfOneDynamicsMatchTheFilterOfOneGaussian`).

`update()` runs `ImmRecursion::update(state, measurement, given<Path>(space, value), policy)` on
each particle where it is and multiplies the weight by the IMM's evidence, as the point-mass filter
does a cell's (§6), through the same loop; a particle no mode accepts keeps its hybrid with weight
zero. The particles share the recursion's scratch, so `predict()` and every update but one by a
likelihood alone take `Sequential` (Q25), and an update allocates the IMM result it reuses (Q26).
Resampling copies each hybrid by its ancestor.

> **Q21.** A Rao-Blackwellized particle filter whose particles are modes: sampling a switching
> model's mode sequence, with a Kalman filter per particle. `RbParticleFilter` samples a
> continuous block only. Which source, and is it this filter with a discrete sampled block or a
> filter of its own?

## 10. What misuse looks like

| Misuse | Fails | First error (Clang) |
|---|---|---|
| a grid with a kernel per cell, in a `SharedCovariance` filter | at the filter | `static assertion failed ... cells with covariances of their own give each cell its own innovation covariance, so the filter needs PerPointCovariance`, then 3 more |
| a policy passed to a `SharedCovariance` filter | at `update()` | `no matching member function for call to 'update'`, then, for the policy overload, `because 'Emrys::Observed<...>' does not satisfy 'ExecutionPolicy'`; `because '!Shared' evaluated to false` comes third |
| mixture noise with `SharedCovariance` | at `update()` | `static assertion failed ... a shared covariance needs additive Gaussian noise and no uncertain input; use PerPointCovariance for mixture noise or noise through the function`, then 2 more |
| `Keep` in a per-point filter | at `update()` | `static assertion failed: a Rao-Blackwellized filter keeps one Gaussian per point, so its policy must keep one: Collapse, Select, or a Gate over them`, then `because 'ReductionPolicy<Emrys::reductions::Keep, ...>' evaluated to false` and 1 more error |
| `Subdivide` on a `pointGrid` state | at the action | `invalid reference to function 'subdivide': constraints not satisfied`, then `because '...::Regular' evaluated to false` |
| `PointMassFilter` over part of the state | at the filter | `no matching constructor`, then `because 'std::is_same_v<typename StateType::RestSpace, Linear<0>>' evaluated to false` |
| an input to a `PointMassFilter`'s `predict()` | at `predict()` | `static assertion failed ... a point-mass filter over the whole state takes no input` |
| a model with a function observed without a method, `observe(model, z)` | at `update()` | `static assertion failed ... a Kalman update pushes the prior through the model by a method: observe(method, model, z)`, then 1 more |
| `Iterated` as a measurement's method [iterated-update] | at `update()` | `static assertion failed ... an update given a block does not support Iterated yet`, then 1 more from `Iterated` itself ("... use its inner method there") |
| a method other than a fixed-step model's own, named in the filter's type [fixed-step-models] | at the filter | `static assertion failed due to requirement 'internal::predictsBy<...>': a filter with a fixed-step model predicts by the model's method` |
| an integer time step, `predict(1)` [time-steps] | at `predict()` | `no matching member function for call to 'predict'`, then `because 'TimeStep<int, double>' evaluated to false` |
| a process model that doesn't write every coordinate of the rest (point-mass) or the state (RBPF) once | at the filter, where the types decide it; otherwise at run time, debug only | `static assertion failed ... the process model must write every coordinate of the state space exactly once: a block it writes twice, or one no model writes, is not a partition`; at run time, `the process model must write every coordinate of the rest of the state once` (or `of the state space`) |
| a likelihood that reads a block of the rest [likelihood-measurements] | at `update()` | `static assertion failed due to requirement 'pathLiesIn<Emrys::TagPath<Velocity>, Position>': logLikelihood() may read only the gridded or sampled block: with the rest conditioned on it, the rest would no longer be Gaussian` |
| a likelihood given a reduction policy | at `update()` | `static assertion failed due to requirement '!internal::hasOwnLikelihood<...>': a measurement weighted by its own logLikelihood() has no branches to reduce, so it takes no policy`, then 5 more, the first "a Kalman update pushes the prior through the model by a method" |
| a temporary value given to `given()` | at `given()` | `call to deleted function 'given'` (§2) |
| a periodic regular grid longer than the period | never | duplicated cells (§4) |
| a measurement whose likelihood is NaN at one point | at run time | point-mass: silently rejected; RBPF: an assertion (below) |

The static assertions say what to do; each is a negative compile test in
`tests/Filters/compile_errors.cpp`, as are the likelihood cases and the measurement without a
method (`RB_UPDATE_WITHOUT_METHOD`). Two rows are gone since 2026-11-04: roadmap 17z removed
`FactoredUnscented`, which the update given a block and the RBPF's prediction refused ("an update
given a block does not support FactoredUnscented yet", "a prediction given a block does not support
FactoredUnscented yet"); both now ignore the `affineIn()` that replaced it (§2, §9). The likelihood
given a policy fails worse than it did: after its own assertion the update goes on to treat the
likelihood as a Kalman update and adds five errors, where on 2026-11-04 it added one. The
policy-with-shared case is still the weakest. Since [execution-policies] added an overload whose
first parameter is an execution policy, the first reason Clang gives is that the measurement is not
an `ExecutionPolicy`, the second that the refill overload `could not match
'RbParticleFilterUpdate<Scalar, ResultAllocator>'` against the policy, and the one that explains
the mistake, `!Shared`, comes third. Two rows went before 2026-11-04: a `given()` stored past its
full expression no longer dangles (Q2), and an update result read after the next update holds its
own values (Q15).

> **Q17.** Add a constrained overload for the shared form, `update(measurement, policy)
> requires Shared`, whose body is a `static_assert` saying "a SharedCovariance filter takes no
> reduction policy; use PerPointCovariance for a gate or a mixture"? It would win overload
> resolution over the policy-first overload, whose constraint fails, and turn the third note into
> the first error. Q5's gate would replace it for `Gate`.

A NaN log-likelihood at a single point is handled differently by the two states. Over a state
`Composite<X, R>` of two `Linear<1>` blocks, with points of `X` at 4, 9, and −1, each with the
Gaussian $\mathcal{N}(0, 1)$ over `R`, a linearized measurement $\sqrt{x} + r$ with noise variance
0.25, observed at 2.5 (the grid with `RandomWalk<R>{0.01}`, the particles with
`compose(RandomWalk<X>{0.01}, RandomWalk<R>{0.01})`):

```
RbPointMassFilter: accepted 0, logLikelihood -nan(ind), per cell -1.13051 -1.13051 -nan(ind), size 3
RbParticleFilter:  Assertion failed: !isNaN(logLikelihood) && "the measurement's log-likelihood must not be NaN"
```

(GCC prints `nan` for `-nan(ind)`.) `GridGaussianParams::reweight` normalizes through
`logTotalWeight`, which reports `NonFinite`, and that path is the one for "no cell explains the
measurement". So a NaN at one cell discards the measurement for every cell, reports a NaN total,
and says nothing. The particle state now asserts on the evidence after the log-sum-exp, not on each
weight inside it, so the assertion no longer runs inside a parallel body.

> **Q14.** This looks like a bug in the point-mass state. Should it assert as the RBPF does, or
> should both treat a NaN point as rejected (log-likelihood $-\infty$), so that a measurement
> undefined at some cells, such as one beyond a map's edge, still updates the others? A
> likelihood measurement [likelihood-measurements] can now return $-\infty$ itself where it is
> undefined, which is the second answer written by the user. The roadmap's request to move the
> RBPF's assertion out of the log-sum-exp loop is done. The point-mass behavior is still so on
> 2026-11-11.

## 11. Smaller notes

- The damaged Doxygen comment the first version found in `Rb.h` (BEL characters
  where `\a` was meant) went with that header's split; `Conditionals.h` and
  `RbUpdate.h` have no control characters. `tests/Distributions/gaussian_mixture.cpp`
  still has one, now at line 253.
- The shared forms use the most probable point as the reference for the one transform, in both
  predict and update. After resampling every weight is equal, so that is particle 0.
- `RbPointMassFilterUpdate::cells()` is what lets a caller line the per-cell arrays up with
  cells after the update has removed some. The result owns it, refilled by an update into a kept
  result.
- `RbPointMassFilter`'s `predict()` with `SharedCovariance` gives the reference cell's mean by
  the model's function too, not by the method's predicted mean. Under `Unscented` with a nonlinear
  user-written model that differs from the per-cell form; inside the shared form's exactness
  domain it doesn't.
- `update(prior, measurement, given)` transforms the whole rest of the state, not only the paths
  the measurement reads (roadmap step 6). With `Linearized` or `Unscented` and a large rest, the
  per-point form pays for that at every point.
- The per-particle predict no longer allocates a Gaussian per particle per step: since step 10 the
  Gaussians are written into the store's reused slots (§14).
- `SharedCovariance` and `PerPointCovariance` are declared in `Conditionals.h`, and `GridAxis`,
  `GridCells`, `PointCells`, and `PointGrid` in `GridGaussian.h`. `GridAxis`, `GridCells`, and
  `PointCells` are in the `Distributions` Doxygen group, not the Discrete subgroup of the Spaces module page.
- `GridGaussianParams`'s prunes return the log of the probability removed, `[[nodiscard]]` as style.md
  asks; the actions report only whether anything changed.

## 12. Execution policies

Every operation of these filters whose work grows with the number of points takes an execution
policy first [execution-policies], and the overload without one runs `Sequential`, the code it ran
before, bitwise:

| Operation | Under a policy |
|---|---|
| `RbPointMassFilter::predict(policy, dt, inputs...)`, and the fixed-step forms | a for-each over the live cells, so the result is Sequential's exactly |
| `RbPointMassFilter::update(policy, measurement[, reduction][, result])`, a likelihood's included | a for-each over the cells into slots, then the evidence as a chunked reduction: the per-cell results are Sequential's, the total matches to rounding; removing dead cells stays sequential |
| `RbParticleFilter::predict(policy, rng, dt, inputs...)`, and the fixed-step forms | the standard normals drawn first, in Sequential's order, then a for-each over the particles: Sequential's exactly from the same seed |
| `RbParticleFilter::update(policy, measurement[, reduction][, result])` | as the point-mass update |
| `ParticleGaussianParams::resample(policy, rng, ...)`, `logTotalWeight(policy)`, `ess(policy)` | the `ParticleSet`'s, with the ancestors' Gaussians gathered as a for-each |

Under the per-point form the reduction policy runs for several points at once, and so does the
process model's function in a prediction, so both must be safe to call concurrently.
Condensed from a 200 × 200 grid of 0.5 m cells over §6's square, its first center at
(−49.75, −49.75), with §6's `ConstantVelocity<Bias, Drift>{0.01}` and `initial`, and 20,000
particles from §9's `prior` and `std::mt19937_64{11}`; `fix` is §6's range to the first beacon from
(17.3, −8.6) plus 1 m:

```cpp
auto parallel = seconds;                                  // an RbPointMassFilter, 40,000 cells, after one predict
auto const sequential = seconds.update(fix);
auto const inParallel = parallel.update(std::execution::par, fix);
// per-cell log-likelihoods bitwise equal; the evidence differs by 1.78e-15

auto one =
    RbParticleFilter{ConstantVelocity<Position, Velocity>{0.1}, drawParticleGaussian<Position>(prior, 20'000, rng)};
auto many = one;
auto first = std::mt19937_64{59};
auto second = std::mt19937_64{59};
one.predict(first, 1.0);
many.predict(std::execution::par, second, 1.0);           // the particles bitwise equal to one's
auto const resampled = many.state().resample(std::execution::par, second);
```

`tests/parallel.cpp` checks both filters in both forms against `Sequential`, with fixed-step
models too. The evidence differs at 40,000 cells because the reduction is chunked by 1024 under
any policy but `Sequential`, whose one chunk is the whole array; on 2026-11-04, from a setup not
recorded, it differed by 2.66e-15, and GCC from w64devkit, which runs `par` on one thread, printed
the same difference. The prunes and `Subdivide` take no policy and run
their single pass sequentially (the Upkeep module page).

## 13. Time steps and fixed-step models

`predict()` takes any time step [time-steps]: a `double`, a `std::chrono::duration`, an integer
clock with its own period, or a user's type with a `TimeTraits` specialization. It is converted
once, to the filter's scalar, before any cell or particle moves. A fixed-step model
[fixed-step-models] carries its step and its method instead; the filter then predicts with no step,
or by a count:

```cpp
auto seconds = RbPointMassFilter{model, grid, initial};
auto milliseconds = RbPointMassFilter{model, grid, initial};
seconds.predict(1.0);
milliseconds.predict(1000ms);                   // the same shared covariance, bitwise

// MethodType Linearized, the model's
auto stepped = RbPointMassFilter{fixedStep<Rest>(model, 1.0), grid, initial};
auto counted = stepped;
stepped.predict();
stepped.predict();
counted.predict(Steps{2});                      // bitwise the two calls
```

An integer is not a step (§10). The point-mass filter's fixed-step model is over the grid's rest,
named as `fixedStep<Rest>()` for a fixed-size one [space-storage] and passed as
`FixedStepModel{model, grid.restSpace(), dt}` otherwise, since its process model covers only the
rest; the particle filter's is over the
whole state. With a cached transition, the RBPF's prediction given each particle is
$F_r P_i F_r^T + Q$ rather than a transform (§9), which matches the continuous model's
prediction to rounding under `Linearized` and exactly under `Unscented`
(`AFixedStepModelMatchesTheStep` in both test files). The RBPF draws in the same order whether
it is stepped one at a time or by `Steps{n}`. `tests/Filters/time_steps.cpp` has both filters'
cases.

## 14. Allocation, and what a step returns

Both filters allocate through one allocator [allocation]: the point-mass filter through the grid's,
which `regularGrid()` and `pointGrid()` take last, and the RBPF through its state's, which
`drawParticleGaussian()` takes last and `particleGaussian()` takes from the particles. The live
cells, their probabilities, and their Gaussians, the particles and theirs, and every filter's
scratch (the likelihoods, the posterior store, the cells updated, the draws) come from it and are
reused, so a step at a steady size allocates nothing. `ParticleGaussianParams::setCapacity(n)` reserves
the state, and the filter reserves its scratch for that capacity when it is built.
`tests/allocation_particles.cpp` runs predict, two updates, and `Resample` or `PruneBelow` over a
`std::pmr` resource in both forms and with fixed-step models, and counts nothing allocated after
warm-up.

The filters take the grid and the state by value, so a `std::pmr` grid copied in takes the
allocator `select_on_container_copy_construction()` gives, the default resource; the test moves
its grid in for that reason. That is the standard containers' rule, not this library's.

Since 2026-10-31 the two exceptions the previous refresh found are gone. `GridGaussianParams::subdivide()`
keeps its list of children in a buffer of the state's, reused, and the subdivided grid's kernel
comes from the grid's allocator; `collapsed()` streams the points through the moments
(`internal::pointMomentsIn()`) and allocates nothing for a bounded space, and the allocation tests
read it every step. `toCategorical()` and `marginal()` return new objects, which is expected.
`Subdivide` grows the state, so the allocation tests don't cover it.

> **Q18.** *Decided [allocation]: both.* ~~Should `subdivide()` build its children in the state's scratch, as removing cells
> does, so that a filter refining under `upkeep::whenSize(std::less{}, n, upkeep::subdivide(2))` allocates nothing once it
> has held as many cells? And should `collapsed()`, which a caller reads every step, reuse
> scratch or take an allocator? The allocation tests cover `Prune` but neither of these.~~

What a step returns owns its data. `RbPointMassFilterUpdate` and
`RbParticleFilterUpdate` keep their cells and per-point arrays in storage from the filter's
allocator, so a result kept past the next update still reads its own values. `update(measurement)`
makes a new result, which allocates; a caller stepping without allocating keeps one and passes it
back:

```cpp
auto result = decltype(filter)::UpdateType{filter.get_allocator()};
for (auto const & fix : fixes)
{
    filter.predict(dt);
    if (auto const updated = filter.update(fix, result); !updated)
    {
        return updated.error();
    }
    // result.logLikelihoods() and result.cells() are this update's, until result is refilled
}
// ranges to beacons 0 to 3 with a 1 m bias and no noise, each after predict(1.0), on the 200 x 200 grid
// of §12, pruned at 1e-9 after each:
// refill 0: cells 40000 logL -4.364;  refill 1: cells 10659;  refill 2: cells 1056;  refill 3: cells 92
```

The same holds for `GridGaussianParams::cells()` and `logWeights()`, and `ParticleGaussianParams::logWeights()`,
which are views of the state and change across an upkeep action.

> **Q24.** `RbPointMassFilterUpdate` and `RbParticleFilterUpdate` declare `allocator_type`, while
> `MixtureFilterUpdate` and `ImmFilterUpdate` have a constructor from an allocator and
> `get_allocator()` but deliberately no `allocator_type`, "which would ask a pmr container of them
> for allocator-extended copies". The Rao-Blackwellized results have no allocator-extended copy
> constructor. Should they drop `allocator_type` to match, or gain the allocator-extended
> constructors?

> **Q15.** *Decided: the results own their arrays, and `update(measurement, result)` and `update(measurement, policy, result)` refill a kept one, so a step allocates nothing.* ~~Should the update results own their arrays? The roadmap's lifetime list offers two
> answers: an owning result with an overload that reuses a caller's `result`, as
> `MixtureFilter::update` has, or a name that says it is a view (`...UpdateView`). An owning result
> would cost an allocation per update unless the caller reuses one, which [allocation] asks of a
> step; the reusable-result overload keeps both.~~

## 15. Printing

Both states and both filters print [printing-and-simulation]: a filter prints its name with its
state nested under it, and the state its size and the moments of its collapse.

```cpp
std::cout << filter;  // the point-mass filter of §6, after the twelve ranges
// Rao-Blackwellized point-mass filter
//   Rao-Blackwellized grid of 316 cells of dimension 4
//     mean: 17.7435 -8.51991 1.38854 0.0339521
//     covariance:
//         0.0671966   0.0132334  -0.0117832 -0.00214544
//         0.0132334   0.0713359   0.0339727  0.00615509
//        -0.0117832   0.0339727    0.142617   0.0410118
//       -0.00214544  0.00615509   0.0410118   0.0281553

std::cout << filter;  // the particle filter of §9
// Rao-Blackwellized particle filter
//   Rao-Blackwellized particle set of 500 particles of dimension 4, effective sample size 500
//     mean: 31.9481 -20.5304 1.34129 -0.768583
//     covariance: ...
```

If the weights are invalid the state prints the weight error instead of the moments. Neither
says how many cells the grid has in all, its spacing, or the covariance form; the first line of
a grid would be more useful with "of 451,584" and the spacing, but that is a matter for
`Core/Print.h`'s printing conventions. The output ends without a newline, as Eigen's does.

## 16. The shape of the filters after [component-first] and [filters-and-recursions]

[filters-and-recursions] split the Kalman, mixture, and IMM filters into a state and a recursion:
the model, the method, and the scratch, acting on a state passed in. The Filters module page calls that
the shape of every filter. These two filters were not split. Each owns its state and its scratch, and
reaches into the state's private members through friendship: the filter reads `m_conditionals` to
update them, swaps its posterior store in, and calls the state's private `reweight()`. There is no
`RbPointMassRecursion` to run a step on a `GridGaussianParams` the caller holds, and so nothing for a
`MixtureFilter`, or a smoother, to hold one of.

[component-first] left `GridGaussianParams` and `ParticleGaussianParams` holding `GaussianParams` until a
second form had a per-point update written for it. The hybrid was the first: `GridGaussianParams` now
takes its component, a Gaussian or a `HybridGaussian` over the rest (§6), and the per-point form of
`internal::Conditionals` stores either. `ParticleGaussianParams` takes one too, a hybrid making each
particle an IMM filter (§9, Q27). The shared
form's store, a matrix of means and one covariance, has no square-root counterpart yet: one factor
shared by every point would be the natural one.

> **Q16.** Should 0.1 split these two filters as [filters-and-recursions] split the others, so that
> every filter has the same shape? The recursions would hold the model, the method, and the scratch,
> and act on a `GridGaussianParams` or `ParticleGaussianParams`; the states would need public (or `internal`)
> access for the reweighting the filters now reach by friendship. Nothing composes them today, so
> the gain is uniformity and a smoother's later use, against an API change before 0.1. If not,
> [filters-and-recursions] should say that it doesn't cover them, and why. [user-extension-points]
> now adds a reason: it lists a recursion among the contracts a user's type meets (`Recursion` in
> `Filters/Recursion.h`, which `KalmanFilter` and `MixtureFilter` take last), and these two filters
> have none to replace, so a user's variant of their steps means a copy of the filter.

## Questions

1. One word for the block held at points: pinned, fixed, given, or sampled (§1).
2. ~~`Given` holds the space by reference, so a stored `given()` dangles; hold it by value? (§2)~~
   *Decided: by value, and a temporary value is rejected.*
3. Separate tags for a grid's kernels and a filter's conditional covariances (§3).
4. The shared update evaluates each innovation at the mean, dropping the method's mean
   correction (§3).
5. A per-point NIS gate for the shared form (§3).
6. A periodic regular grid can overlap itself unchecked (§4).
7. ~~Name a grid's axes by the block's tags where it has them (§4).~~ *Decided: the cells are
   `GridCells<Path>` or `PointCells<Path>`, and each axis `GridAxis<Block, I>`.*
8. Keep the dense `toCategorical()` once grids are refined? (§5)
9. Marginals over continuous blocks, as mixtures or particle sets, for the Rao-Blackwellized
   states (§5).
10. Track the offset from a cell's center in the rest, instead of a kernel? (§6)
11. Build both filters from their state, and record the parameter order of a type built over another
    type (§6). Partly decided: [component-first]'s clause covers a type built over a component.
12. Let a grid translate as a whole, and grow its kernel, for terrain-aided navigation (§7).
13. `PointMassFilter` names the filter by the step it lacks; and its `predict()` does nothing (§8).
14. A NaN likelihood at one cell silently rejects the measurement (bug) (§10). The RBPF's
    assertion has left the log-sum-exp loop.
15. ~~The update results are views over scratch that silently change at the next update; own them,
    or name them views? (§14)~~ *Decided: they own them.*
16. Split the two filters into a state and a recursion, as [filters-and-recursions] split the
    others? [user-extension-points] lists a recursion as a user's contract, and these have none.
    (§16)
17. A `static_assert` overload so that a policy passed to a shared filter says so first (§10).
18. ~~`subdivide()` and `collapsed()` allocate from the global heap, outside the allocator (§14).~~
    *Decided [allocation]: neither does.*
19. Spell the rest's type for the user; a rest of one block is that block, not a composite (§4).
20. Selective refinement and coarsening of a grid. (§5)
21. Particles that are the modes of a switching model. (§9)
22. A test of whether a grid's collapse describes it. (§6)
23. A likelihood ignores a cell's kernel, where a Kalman update widens by it; average it over the
    kernel, refuse it, or advise subdividing? (§6)
24. The Rao-Blackwellized results declare `allocator_type`, which the mixture and IMM results
    leave out on purpose. (§14)
25. Hybrid cells and hybrid particles run one at a time, since every point's update writes the one
    `ImmRecursion`'s scratch. Building each point's mixed and posterior modes in the posterior store,
    which already holds a hybrid per point, would make the step reentrant and let thousands of points
    run in parallel; the particle filter's prediction already builds each particle's hybrid there,
    and takes `Sequential` only for its first failure and its draws in order. (§6, §9)
26. A hybrid grid's or hybrid particles' update allocates the IMM result that it reuses from point
    to point, once per update; the filter could keep it, typed by the measurement, as the IMM's own
    results are kept by the caller. (§6, §9)
27. ~~Hybrid particles: `ParticleGaussianParams` and `RbParticleFilter` still hold one Gaussian per
    particle.~~ *Decided [rb-particle-filter]: under a `SwitchingModel` each particle holds a
    `HybridGaussian`, and the prediction draws from the mixture over the modes and reweights them by
    the draw (§9).*
