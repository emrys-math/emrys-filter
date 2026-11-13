# Decisions

The settled decisions behind EmrysFilter: for each, the rule, the reason, the alternatives worth
not proposing again, and where the code is. It is written for contributors. How code is written
and named is in [style.md](style.md), the order of the work in [roadmap.md](roadmap.md), and open
questions in the review of their area in `docs/guide/`.

How it is kept:

- A decision records a choice: either one with a real alternative that someone could propose
  again, or one whose reason the code and its documentation don't show. A description of what the
  library does is not a decision; it belongs in the header's or the module's documentation.
- A decision is named, in kebab case, for its subject (`view-stamp`, `execution-policies`), and
  cited by its name in brackets, `[view-stamp]`, in headers, tests, and documents. It is added in
  the commit that implements it, with a row in the index under its group. A name changes only with
  its subject, and every citation changes with it.
- A decision that changes is rewritten in place, with no note of the change; the commit message
  gives the reason. A decision that is replaced or no longer records a choice is removed, and its
  citations are changed in the same commit.
- A decision says what and why: **Rule** (about 15 lines at most), **Why** (about 10),
  **Rejected** (alternatives someone is likely to propose again, a sentence each), **Evidence**
  where tests or benchmarks back it, and **Where**. How the code does it belongs in the header's
  documentation, which cites the decision.

## Index

Read the index, then the decisions your change touches.

| Decision | Rule |
|---|---|
| **Core and cross-cutting** | |
| [allocation](#allocation) | Allocation happens only where it's needed, once, and the caller controls it |
| [forwarded-matrices](#forwarded-matrices) | A function that stores a matrix argument forwards it; one that reads it takes `MatrixBase const &` |
| [embedded-builds](#embedded-builds) | The library builds warning-free, without exceptions, and for Cortex-M |
| [borrowed-intermediates](#borrowed-intermediates) | Inside a step, intermediates are borrowed; what a step returns is owned or written in place |
| [execution-policies](#execution-policies) | A loop over a count takes an execution policy as its first argument, `execution::sequential` by default |
| [view-stamp](#view-stamp) | A view checks its parameters' modification stamp when read, in builds with assertions, and no view is made of a temporary; every build keeps the stamp |
| [time-steps](#time-steps) | A time step is any type with a conversion to the filter's scalar, made once, below which time is that scalar |
| [printing-and-simulation](#printing-and-simulation) | Every distribution and filter prints a summary, and simulation draws the truth one step at a time |
| [batched-functions](#batched-functions) | A batched function takes every point at once and writes one output, its Jacobians by block when its mark gives them, and the library calls it once per batch (in progress: step 18c) |
| [params-and-density](#params-and-density) | The plain name evaluates a density, its parameters are `...Params`, and a closed list of abbreviations shortens names |
| [policy-factories](#policy-factories) | A policy is made by a factory in the namespace of its kind, and a parameter that sets how much work is done is a run-time value |
| **Spaces** | |
| [discrete-spaces](#discrete-spaces) | Discrete values live in their own family of spaces |
| [sizes-and-dimensions](#sizes-and-dimensions) | `size()` counts elements, and a dimension says whether it is of values or differences |
| [index-names](#index-names) | The index utilities have short names |
| [covariance-reset](#covariance-reset) | A covariance moves with its mean, by each block's reset Jacobian, and costs nothing where that Jacobian is the identity |
| [unit-quaternion](#unit-quaternion) | The library has a unit quaternion space, stored as Eigen stores a quaternion and viewed through a map |
| [lie-groups](#lie-groups) | Lie groups are spaces in this library, and navigation is domain models written on them (pose groups not built) |
| [linear-space](#linear-space) | The plain space is named Linear, for its structure rather than its set |
| [no-bounded-coordinates](#no-bounded-coordinates) | There is no bounded coordinate type |
| [several-resizable-blocks](#several-resizable-blocks) | A composite may have several resizable blocks, and only a space with at most one takes its sizes from a length |
| [coordinate-parameters](#coordinate-parameters) | A coordinate type's parameters are template arguments, and an interval is spelled by its ends |
| [difference-spaces](#difference-spaces) | A space has at most one normalize(); its differences live in an untagged difference space; updates, residuals, and differences taken away are plus(), minus(), and less(), with defaults |
| [tag-addressing](#tag-addressing) | Blocks and axes are addressed by tag path, never by type or by index |
| [tangent-gaussians](#tangent-gaussians) | A Gaussian on a non-linear space is a Gaussian in the tangent space at its mean; the wrapped normal is not computed |
| [space-storage](#space-storage) | A space costs what its state costs, and a whole distribution keeps its own copy |
| **Distributions** | |
| [upkeep](#upkeep) | Upkeep is a set of function objects the caller applies to a filter's state |
| [rao-blackwellized-core](#rao-blackwellized-core) | Rao-Blackwellized states share one core, and upkeep keeps a state's type |
| [density-names](#density-names) | Continuous distributions evaluate `logDensity()` and `density()`; discrete ones a `probability()` |
| [divergences](#divergences) | Divergences are free functions over factorized views, and only Wasserstein takes a ground metric |
| [uniform-mixtures](#uniform-mixtures) | A uniform is approximated by a mixture on a regular grid, flat by default rather than moment-matched |
| [component-first](#component-first) | A type that holds several Gaussians takes its component as its first parameter, and a hybrid state is a component |
| [mixture-entries](#mixture-entries) | A mixture's entries are addressed by index and handed out read-only, and a reweighting callback is its own member |
| [gaussian-forms](#gaussian-forms) | The Gaussian has four forms, each its own parameters type with the same accessors, and a form folds a linearization in |
| [roots-and-factors](#roots-and-factors) | A root is square and a factor need not be; the Gaussian forms' matrices are named as roots |
| [form-densities](#form-densities) | The four forms are read alike through their densities, where nothing fails, and their parameters keep the failures |
| [element-marginals](#element-marginals) | A marginal reads its source rather than copying it, and a marginal of a distribution made of elements iterates them |
| [categorical-params](#categorical-params) | A categorical distribution has parameters, whose weights are written unchecked and checked once by `toDensity()` |
| [checked-factories](#checked-factories) | A distribution that is its own parameters is built by a factory that checks them, and its constructor is public only where building cannot fail |
| [product-factors](#product-factors) | A product's factors are placed by path with no space, as `Factors`, and one factor may cover every repetition of a block |
| [covariance-repair](#covariance-repair) | A covariance that fails to factorize is repaired only on request, by a repair given as an argument, and the parameters then hold the repaired matrix |
| [sigma-point-axes](#sigma-point-axes) | A sigma-point method is a rule spread by axes the caller may choose, and the default axes are the Cholesky factor, falling back to a pivoted root |
| [conditional-evidence](#conditional-evidence) | `conditional()` returns a plain distribution; the evidence of the given values stays in its weights, or comes beside a single Gaussian |
| [random-draws](#random-draws) | A density draws from the caller's generator through the standard distributions, so a seed fixes the draws within one standard library only |
| **Models** | |
| [method-per-measurement](#method-per-measurement) | Each measurement chooses its transform method |
| [additive-noise](#additive-noise) | Additive noise is explicit, and it may be a mixture |
| [weighted-branches](#weighted-branches) | An update returns weighted branches |
| [switching-models](#switching-models) | A switching process model is a set of per-mode models plus Π |
| [templated-measurements](#templated-measurements) | Heterogeneous sensors use templates, not type erasure |
| [sensors](#sensors) | A measurement model predicts the observation from what it reads, and a measurement is the model observed by a method |
| [linear-sensors](#linear-sensors) | A linear measurement is a model given its matrix, or its matrix's columns by block, which reads blocks whose values add |
| [measurement-space-functions](#measurement-space-functions) | A measurement model's output space may be a function of the state's space, computed once per update and passed to the function as `out` |
| [continuous-time-noise](#continuous-time-noise) | Process noise is continuous-time, discretized exactly by each model |
| [prediction-inputs](#prediction-inputs) | Prediction takes inputs by the tag of their space, uncertain or sure |
| [model-policies](#model-policies) | A model is a function with policies given as calls, and a composition's method is given at the call |
| [function-scalars](#function-scalars) | A function the library calls with a vector may take the vector's scalar as its one template parameter, which the call supplies |
| [model-placement](#model-placement) | A model is placed on a block by prefixing its paths, on every repetition by onEach, and models compose side by side |
| [whole-state-models](#whole-state-models) | A model given without `reads()` and `writes()` reads and writes the whole state, and is neither placed nor composed |
| [transform-policies](#transform-policies) | `transform()` takes a measurement model's policies before its inputs, matches its inputs by tag, and takes no process model |
| [affine-coordinates](#affine-coordinates) | A model declares the inputs it is affine in, and every sigma-point method factors them out |
| [covariance-update-forms](#covariance-update-forms) | The measurement chooses the form of the covariance update, and Joseph rests on the statistical linearization |
| [consider-states](#consider-states) | A measurement can consider blocks of the state rather than estimate them, by an argument of `observe()` |
| [linearization-forms](#linearization-forms) | A transform's result keeps its linearization in the form its method computes, with the parts of its covariance apart |
| [fixed-step-models](#fixed-step-models) | A fixed time step is a model, which caches what the step fixes |
| [innovation-statistics](#innovation-statistics) | An update's statistics are derived from its innovation, when asked for |
| [returned-jacobians](#returned-jacobians) | A function that knows its Jacobian returns it with its value, and nothing differentiates by fixed steps |
| [emissions](#emissions) | A discrete measurement is an emission model, and a modulated arrival is observed by the prediction |
| [iterated-update](#iterated-update) | The iterated update is a method that wraps the method it relinearizes, and iterates in the prior's chart |
| [linearization-reference](#linearization-reference) | A step may be linearized about a point or a distribution the caller gives, in its input's chart |
| [sigma-point-sets](#sigma-point-sets) | Sigma-point sets are methods beside Unscented, and an iterated one linearizes over the posterior |
| [cramer-rao-bound](#cramer-rao-bound) | The posterior Cramér-Rao bound is an information recursion fed by Jacobians at the true states |
| [innovation-cholesky](#innovation-cholesky) | An innovation covariance must be positive definite, and an update that finds one that is not returns an error unless its measurement repairs it |
| [likelihood-measurements](#likelihood-measurements) | A measurement may be known only by its likelihood, which the filters that weight points take, and a measurement's own likelihood wins over its noise |
| **Filters** | |
| [reduction-policies](#reduction-policies) | Filters reduce branches with a policy |
| [composed-features](#composed-features) | Optional features compose with the filter instead of mixing into it |
| [mixture-filter](#mixture-filter) | The Gaussian mixture filter reduces per component and reweights by its likelihood |
| [imm](#imm) | IMM keeps one Gaussian per mode in a hybrid, and switches models and measurements by mode axis |
| [rb-point-mass](#rb-point-mass) | The Rao-Blackwellized point-mass filter fixes one block at points and never moves them |
| [rb-particle-filter](#rb-particle-filter) | The Rao-Blackwellized particle filter predicts each particle with its Gaussian and conditions on the draw |
| [smoothing](#smoothing) | Smoothing is a backward pass over a record the caller keeps of the forward pass |
| [filters-and-recursions](#filters-and-recursions) | A filter holds a state and runs a recursion, and the mixture filter is generic in its component |
| [autonomous-multiple-model](#autonomous-multiple-model) | The autonomous multiple model estimator is the IMM on the identity chain, and a floor on the mode probabilities is upkeep |
| [state-access](#state-access) | Every filter has a mutable `state()` to change its state in place and `setState()` to replace it |
| [update-outcomes](#update-outcomes) | Every filter's update reports one of four outcomes in one shape, and no filter copies its state to undo one |
| **Project** | |
| [layered-headers](#layered-headers) | Headers are grouped in layered directories, under one namespace |
| [analysis-project](#analysis-project) | Analysis tools live in one separate project; the libraries keep the numerics |
| [benchmarks](#benchmarks) | Speed is benchmarked in this repository against hand-rolled Eigen |
| [coverage](#coverage) | Full coverage is instantiation, region, and axis coverage, measured per instantiation (region, entity, and compile-error coverage built; axes Verification) |
| [check-stages](#check-stages) | Checks run in stages chosen by what each can catch, and the diff decides which a change needs |
| [splitting-hooks](#splitting-hooks) | Splitting by nonlinearity is our own work, kept outside the library, with the library supplying the hooks (the action outside the library; hooks not built: step 16b) |
| [distributed-fusion](#distributed-fusion) | Distributed fusion is factor operations and filter steps here, node bookkeeping in the tracker, and no project of its own (not built: step 16c) |
| [eigen-partner](#eigen-partner) | The library is a partner to Eigen, not part of it |
| [published-algorithms-only](#published-algorithms-only) | The library implements only algorithms from the public literature |
| [user-extension-points](#user-extension-points) | What the library won't implement, a user must be able to, against public contracts and without touching it |
| [user-skills](#user-skills) | Skills for a user's agent install from the library, at its version, into the directory the user's project names |
| [cmake-package](#cmake-package) | The install is a CMake package, and installs a fetched Eigen beside it |

## Core and cross-cutting

### allocation

*Allocation happens only where it's needed, once, and the caller controls it.*

**Rule.** Nothing sized by a fixed or bounded space touches the heap, scratch included; an
unbounded dynamic space allocates through Eigen's heap, with no hook. Storage sized by a run-time
count comes from an `Allocator` parameter (style.md, Types).

- A container made from another (a view's `copy()`, a predicted or updated state, a merge, a
  resample) takes the source's allocator, capacity, and scratch reservations, and must not outlive
  its memory resource; copy assignment keeps the destination's.
- An optional capacity is reserved once, and going past it is a precondition violation; without
  one, the container grows. Scratch sized by a count is owned by the container or filter, reused
  across calls, and lent to a function object that accepts it (`ResamplerWithScratch`).
- An update result owns its count-sized arrays; `update(measurement, policy, result)` refills one
  the caller keeps. No result is a view into filter scratch.
- A `SampleSet` is a batch of draws or a temporary, never stored. `drawSample()` and
  `drawSampleWithLogDensity()` allocate nothing for a bounded space; `drawSamples()` does.

**Why.** Eigen takes no allocator, so only a buffer a container holds can be redirected. Callers
need to say where the heap is and to know that allocation happens up front, not every step, and an
allocator admits `std::pmr` at no cost. Copies keep the source's allocator, against pmr's rule,
because a filter built from a copy of a state is where an embedded caller least expects memory to
move. Scratch is reused because a monotonic resource never reclaims what is freed per call.

**Rejected.** A configuration macro: it would not survive the move to a module. A `std::generator`
for a stream of draws: it allocates its frame, and nothing guarantees the allocation is elided.

**Evidence.** `tests/allocation*.cpp`, under `EIGEN_RUNTIME_NO_MALLOC` and `null_memory_resource()`.

**Where.** `Core/Allocation.h`.

### forwarded-matrices

*A function that stores a matrix argument forwards it; one that reads it takes
`MatrixBase const &`.*

**Rule.** A function that keeps a matrix argument whole, as a member, takes it by forwarding
reference constrained by `MatrixExpression` and forwards it, so an rvalue of the stored
type is moved; a stored `Moments` is taken by value. An argument that is only read, or copied
into part of a larger buffer, is `Eigen::MatrixBase<Derived> const &`. Where the stored scalar
may differ, `internal::castScalar<Scalar>()` converts, and forwards unchanged where it matches.

**Why.** Copying a dynamic-size mean and covariance that the caller has just built is an
allocation and a copy that a move avoids; for fixed sizes and expressions, forwarding costs
nothing. Taking the argument by value would need an overload per combination of plain and
expression arguments. The concept, rather than an unconstrained `T &&`, keeps a forwarding
constructor from capturing a copy of its own class.

**Where.** `Core/Forwarding.h`; style.md, Functions.

### embedded-builds

*The library builds warning-free, without exceptions, and for Cortex-M.*

**Rule.** The tests build with `-Wall -Wextra -Wpedantic -Werror` on GCC and Clang and `/W4 /WX`
on MSVC, with Eigen and GoogleTest as system includes, so only the library's warnings count. One
configuration builds the whole suite with `-fno-exceptions -fno-rtti`. The library throws nothing
and calls nothing that throws on a failure it could check (style.md, Functions); failures are a
`std::expected` or `EMRYS_FILTER_ASSERT`, and allocation failure is the allocator's. A Cortex-M
job compiles the ranging benchmark's tests for Cortex-M4F and Cortex-M7 with `float` as the
filter's scalar, runs them under QEMU with a heap that fails the program on first use, and
records code size and stack use. It builds with `EIGEN_ALLOCA=__builtin_alloca`, as an embedded
user must (usage.md): newlib declares `alloca` only outside strict ISO mode, and without it
Eigen puts the scratch of a solve over a run-time-sized block on the heap. A filter in `float`
holds and computes no `double` (style.md): the chi-squared functions and the confidence regions
compute in `Scalar` with its own epsilon, and a method's numbers are converted to it.

**Why.** Embedded users are the ones who weigh a filter library's footprint, and their builds use
these flags. The Arm GNU and Arm LLVM embedded toolchains, ESP-IDF 5.4, and Zephyr's SDK have C++23
with `std::expected`; Arduino AVR (GCC 7) and Teensyduino (GCC 11) do not, and are not targets. A
compiler check is better than searching for `throw`, but it vouches only for instantiated code,
which is why coverage [coverage] comes first. The Cortex-M4F has a single-precision FPU, so the
scalar there is `float`, and a `double` there is emulated in software. The Cortex-M job runs the ranging tests rather than the examples, since
the examples print through iostreams and record their runs in vectors, both of which need the
heap. The ranging filter is the one benchmarks.md measures, and its tests check it against
hand-written Eigen. The heap is checked at run time rather than at link time, because Eigen keeps
a heap branch for scratch over `EIGEN_STACK_ALLOCATION_LIMIT`, which no fixed size takes.

**Where.** `CMakeLists.txt` (`EMRYS_FILTER_WARNINGS`), `CMakePresets.json` (`gcc-no-exceptions`),
`benchmarks/cortex_m/`, `.github/workflows/ci.yml`.

### borrowed-intermediates

*Inside a step, intermediates are borrowed; what a step returns is owned or written in place.*

**Rule.** Within one predict or update, the matrices it reads are referred to, not copied: the
prior's blocks, the cross-covariances, a transform's joint input. A product already computed is
reused, not formed again from its factors. A borrowed block is an Eigen block or expression that
keeps its compile-time sizes, never an `Eigen::Ref`, which drops them and copies when the strides
differ; a lazy expression read twice is evaluated once into a matrix. What a step returns owns its
matrices, and where it has a destination it is written there: a Kalman update with one branch
conditions the state in place, keeping the contract that only a covariance's lower triangle is
read.

**Why.** At the benchmark's small sizes [benchmarks], an update's fixed cost, not its arithmetic,
separated the library from hand-written Eigen: a copy of the prior covariance, a posterior copied
three times on its way into the state, a product computed twice, and zero-filled matrices
overwritten at once. [forwarded-matrices] says how a function takes a matrix; this says what the
internals hold.

### execution-policies

*A loop over a count takes an execution policy as its first argument, `execution::sequential` by
default.*

**Rule.** An operation whose work grows with a run-time count (particles, grid cells, components,
draws) has an overload that takes an execution policy first, before the state it acts on; without one
it runs `execution::sequential` [policy-factories]. Upkeep takes it in its factory [upkeep]. A policy
is `Sequential`, a standard policy (any type `std::for_each` accepts, so the library never includes
`<execution>`), or a user's type with a const `forEachIndex(count, f)` and optionally a const
`get_allocator()` for the scratch; any other policy is consent to allocate outside the container's
allocator. A reduction or scan combines fixed chunks in chunk order, so its result does not depend on
the thread count. A parallel body writes only its own element and, under `par_unseq` and `unseq`,
allocates nothing, which rules out a space of dynamic size; draws, in `Sequential`'s order, and
factorizations are made before the loop [view-stamp]. The Kalman filter, the transforms, the IMM
filter, and reductions over components take no policy: they loop over a dimension or a few items.

**Why.** An argument lets one filter run a five-component update sequentially and a
million-particle prediction in parallel, and leaves the types and deduction guides alone.
`<execution>` stays out because with libstdc++ and oneTBB a program that includes it needs
`-ltbb`, and it quadruples the preprocessed source. Fixed chunks cost nothing against `std::reduce`
and make a parallel result the same on every core count, which debugging a filter needs. Scratch
comes from the policy because it belongs to the execution, as C++26's `std::execution` asks a
receiver for `get_allocator()`. Drawing before the loop keeps a caller's `std::mt19937` usable.

**Rejected.** A chunk size set by a macro: results would depend on the build, and translation
units that disagreed would break the one-definition rule. Step 18c's policy adaptor will set it.

**Evidence.** `tests/parallel.cpp`, every overload against `Sequential`; `tests/allocation*.cpp`.

**Where.** `Core/Execution.h`; [parallelism.md](parallelism.md) for steps 18 and 19.

### view-stamp

*A view checks its parameters' modification stamp when read, in builds with assertions, and no view
is made of a temporary; every build keeps the stamp.*

**Rule.** A view, such as a density from `toDensity()` or a marginal, reads its parameters'
storage, and three things protect it:

- No view of a temporary: a view maker is `const &`, with its `&&` overload deleted. A view of a
  view holds a copy of it and is allowed.
- Each parameters type holds an `internal::Stamp` of two counters: one that every mutator
  advances, and one that only a change of size or structure advances. A density compares the
  first at every read and a marginal the second, so values may change under a marginal. A stale
  read fails `EMRYS_FILTER_ASSERT`; a mutation never fails. Every build keeps the stamp and only
  the checks turn off with the assertions, so no setting changes a type's layout.
- The rest, parameters destroyed or reallocated under a view and the Eigen maps the library hands
  out, is use-after-free, which CI's AddressSanitizer run finds.

**Why.** The stamp checks at the read that would go wrong, at the cost of a load and a compare
beside a triangular solve. Two counters keep a marginal's contract, under which values may change.

**Rejected.** Borrow counts kept by the parameters: they covered only the library's own views,
stored into `const` parameters at every view's construction and destruction (a data race under
shared reads, [execution-policies]), and misfired on a density that outlived an assignment without
being read again. A stamp kept only under a macro: it changed the layout with `NDEBUG`, so
translation units that disagreed broke the one-definition rule without a diagnostic. The names
`Lock` and `Borrow`: the checks detect a change and prevent nothing, and readers took the names to
mean that mutators wait or that threads are made safe.

**Where.** `Core/Stamp.h`.

### time-steps

*A time step is any type with a conversion to the filter's scalar, made once, below which time is
that scalar.*

**Rule.** Every public entry point that takes a step of time takes a `TimeStep<Time, Scalar>` and
converts it once, at the entry, to its own `Scalar`. A step is the scalar itself, or any type with
a `TimeTraits` specialization: a floating-point number, and a `std::chrono::duration`, converted
to seconds by its period. Below that boundary time is the scalar; the built-in models hold their
parameters in a scalar of their own and compute in the step's. The library does no arithmetic
that needs a unit, and the conversion of a duration to seconds is the only one it makes. A
negative or NaN step is a precondition violation. An integer is not a step: a fixed-step filter
advances by `predict(Steps{n}, inputs...)`, which is $n$ calls, bitwise. A history's `Lag{n}`
counts steps.

**Why.** Users keep time as durations or integer clocks, and a library that takes `double` makes
each of them write the conversion, with a chance of the wrong unit. Converting once to the filter's
scalar, rather than to `double`, keeps a `float` filter on a microcontroller [embedded-builds] out
of software double precision. Dividing a duration's count by its period rounds correctly, so `100ms`
is exactly `0.1`, and a run in milliseconds equals the run in seconds bitwise.

**Rejected.** A unit per model: every model would declare one, and a switching model's models
would have to agree. Saying "seconds" for floating-point steps: a claim the library could neither
check nor needed. Integer steps: an integer reads as both a time and a count of steps.

**Where.** `Core/Time.h`.

### printing-and-simulation

*Every distribution and filter prints a summary, and simulation draws the truth one step at a time.*

**Rule.** `operator<<` and `std::formatter` print every distribution, filter, and error code as a
summary of a fixed shape: the name, the element count, the dimensions, the mean and covariance,
then each element nested below with its weight. A particle set prints its count, effective sample
size, and moments, never its particles. A type opts in with an `internal::describe()` overload
found by argument-dependent lookup, so the generic `operator<<` and formatter claim no other type.
`EMRYS_FILTER_NO_IO`, which `EIGEN_NO_IO` implies, removes all of it.

`simulate(space, x, model, rng, dt, inputs...)` draws the next true state, one step per call, and
`simulate(space, x, model, rng, inputs...)` with a measurement model draws a measurement, and
`simulate(value, chain, rng)` a Markov chain's next value, from `chain.next(value)`. `nees()` and `NeesMonitor` test estimates
against the truth as `NisMonitor` tests innovations, after Bar-Shalom, Li, and Kirubarajan (2001),
§5.4.2, but only estimates of independent runs: one run's errors are correlated from step to
step, so a sum over its steps is not chi-squared. A failed estimate adds NaN to the monitor rather
than asserting. `nees()` takes any `GaussianForm`, and `nees<Path>()` one block, reading the truth by
the same path. A monitor's window is sized as an Eigen vector is, or is `Cumulative`.

**Why.** The first thing a user prints is a filter's state, and a particle filter holds thousands
of particles. Printing is opted out of, as in Eigen, because Eigen already includes the stream
headers unless told not to. One step per call, because a simulation interleaves the truth with
the measurements drawn from it and the filter's steps, and its inputs change from step to step.

**Rejected.** A formatter for every type with an `operator<<`: it would collide with Eigen's and
users' own. A filter printed as its state alone, so that a log line from a program with several
filters didn't say which kind it came from; a filter names itself and nests its state.
`simulate(model, x0, steps)`.

**Where.** `Core/Print.h`, `Models/Simulation.h`, `Models/Consistency.h`.

### batched-functions

*A batched function takes every point at once, and the library calls it once per batch.*

**Rule.** `batched(f)` marks a model's, log-likelihood's, or transform's function as taking the
points as the columns of a matrix `X` and writing into a last argument `Y` that the library sizes
once (`h(in, out, X, Y)`); column $j$ of the result depends only on column $j$ of the inputs. The
mark names the Jacobians the function gives, whose storage is sized before the call: none,
`jacobians::perColumn()`, or `jacobians::shared()` for a linear function; `Y.wantsJacobian<Name>()`
says which blocks the method uses. Unmarked, a needed Jacobian comes by autodiff: of one column for
`Linearized` and the iterated updates, and of the whole batch, in duals seeded on the blocks used,
where every point's is needed [affine-coordinates], [likelihood-measurements]. The library calls a
batched function once per sigma-point set and per particle set or chunk [execution-policies].
`on()` keeps its model's mark; `compose()` calls each batched part once over its rows, `onEach()` a
batched model once per repetition, and neither writes a Jacobian, so a method differentiates them.
A user's space whose `normalize()`, `plus()`, or `minus()` can't take columns is not a `Space`.

**Why.** Evaluating at many points can cost more than the arithmetic: a lock, a table lookup, a GPU
dispatch, or a call into another runtime has a fixed cost per call, which a batch pays once. Jacobians
by block compute only the columns used; a shared one keeps $10^6$ particles from storing $10^6$ copies.

**Rejected.** Detecting whether a function accepts a matrix: a generic lambda accepts anything, and a
one-column body can compile for many and be wrong (`x.norm()` of a block). The name `columnwise`,
which reads as mapping a one-column function. Returning the results, which allocates [allocation].
A Jacobian argument, which an argument count can't tell from one more space. One call across
`onEach`'s repetitions or every mixture component, for a gain only a device would see.

**Evidence.** Tests that count calls or compare with the per-point form; negative compile tests.

**Where.** `Models/Batched.h`, `Transform::evaluateEach()`, the resolvers' `ResolvedFunction`,
`ProcessStep`, `ModelTransform`, and `PinnedMeasurement`, and `Composition` in `Models/Placement.h`.

### params-and-density

*The plain name evaluates a density, its parameters are `...Params`, and a closed list of
abbreviations shortens names.*

**Rule.** The type that evaluates a distribution has the plain name. Where the values can be
invalid, `...Params` holds them, caches what evaluation needs, and is checked once by
`toDensity()`, which returns a density borrowing that cache [view-stamp]. A type whose values are
always valid is its own parameters, checked when built [checked-factories]. Filters, mixtures,
and hybrids are parameterized by the density and store its parameters. A statistic that reads the
components, such as a mixture's moments, is the density's; one that reads only weights is also on
the parameters as a `std::expected` (`ess()`, `kdeBandwidth()`). Identifiers use a closed list of
abbreviations, and acronyms are camel-case words (style.md, Naming). `covarianceLower()` is the
stored matrix; `covariance()` is the fallible accessor every Gaussian form shares [gaussian-forms].

**Why.** A factorization is cubic in the dimension and each evaluation after it quadratic, and a
filter changes a covariance once per step and may evaluate it many times. Checking in
`toDensity()` lets `logDensity()` return a plain scalar, and a density writes nothing, so threads
share it. The form every distribution has gets the short name. The list of abbreviations is closed
so that abbreviation does not spread one name at a time. Only the factorization checks that each
component is a distribution, so moments on the parameters could describe none.

**Rejected.** A `toDensity()` on a type that is its own parameters: generic code would still
branch on what it returns. An owning copy that evaluates: it copies every factorization.
`Gaussian` for the parameters: the most common name could not evaluate. All-capital acronyms: a
lowercase-first name can't follow the rule, and adjacent acronyms lose their boundary. A
`covariance()` returning a reference in some forms: generic code needs one return type.
`kernelDensity()` for `ParticleSet`'s `toDensity()`, which builds a `KernelDensity`: one state
would reach its density by its own name.

**Where.** `Distributions/Params.h`; the Distributions module page.

### policy-factories

*A policy is made by a factory in the namespace of its kind, and a parameter that sets how much
work is done is a run-time value.*

**Rule.** A policy chooses how an algorithm runs and holds none of the problem's data. Every one
is made by a factory in a lower-case namespace per kind: `methods` (`linearized()`, `cubature()`,
...), `updates`, `reductions`, `resamplers`, `upkeep`, `execution` (whose `sequential` is a
constant, as `std::execution::par` is), and `jacobians` [batched-functions]; the axes and repairs
are [sigma-point-axes]. A policy's type sits beside its factory (`methods::Cubature`); the
concepts stay in `Emrys`. A parameter that defines the algorithm is a template parameter
(`Unscented<Alpha, Beta, Kappa>`); one that sets when or how much work is done, such as a
threshold, a count, or a budget, is a run-time value (`gate(0.997)`,
`iterated(method, {.tolerance = 1e-6})`).

**Why.** No one has to guess which policy takes braces and which a function, and a factory can name
some template arguments while deducing the rest, refuse an argument with a message, and return what
its arguments call for, where braces need a deduction guide per deduced scalar. A braced argument
deduces nothing, so `iterated()` names its scalar, `double` unless written, and a `float` filter
writes `iterated<float>(...)` [embedded-builds]. A run-time value can come from a configuration
file.

**Rejected.** Braces for the empty tags and factories for the rest, which a reader must tell apart.
Compile-time tolerances and iteration caps, which a configuration file could not set.

**Evidence.** `tests/policies.cpp` asserts each namespace's concept of every member, and
`tools/check_policies.py` checks over the Doxygen XML that every member has such an assertion.

**Where.** `Models/Methods.h`, `Models/Condition.h`, `Models/Reduction.h`, `Core/Resampling.h`,
`Distributions/Upkeep.h`, `Core/Execution.h`.

## Spaces

### discrete-spaces

*Discrete values live in their own family of spaces.*

**Rule.** A discrete space is a finite set of values: `Finite<N>`, `Named<Values...>`, and
`DiscreteProduct<Axes...>`, whose values take one value in each axis. The `DiscreteSpace` concept
shares nothing with `Space`. Axes are tags addressed by tag path, and a named value by its tag;
`valueOf<Turn>(space, at<Cell>(3))` takes a name or an index per leaf axis, checked when compiling.
A value is stored as its flat index, with the last axis varying fastest, so a nested product
flattens as one that lists its axes in place, and a Kronecker product of per-axis matrices is
indexed the same way. A discrete space is not resizable: a new set of values is a new space and a
new distribution. There is no discrete `Repeated`. Probabilities are kept as log-weights.

**Why.** Modes, grid cells, and sensor states need probabilities but no differences or means, so
forcing them into `Space` would make every space-aware algorithm skip them. Products are needed
early: a maneuver regime with a sensor's inlier and outlier state is a mode with two axes. A flat
index keeps a value a plain integer for storage and for an update's mode label, while names keep
user code free of index arithmetic. Log-weights survive a long run of likelihoods that would take a
probability below `double`'s range. A resize can't keep the old values' meaning: changing one
axis's cardinality moves most flat indices, a new value has no default probability (zero can never
recover in log space), and every transition matrix, emission table, and per-mode component needs
new entries only the user can give. So a resize would take as much input as a new distribution.
The variable-structure IMM (Li and Bar-Shalom) also builds its new mode set from the old one.

**Rejected.** A resizable discrete space (above). An index as a template argument. A discrete
`Repeated`, whose cardinality grows exponentially with a run-time count; the usual design is one
filter per track. The name `Product`, which `using namespace Eigen;` beside `using namespace
Emrys;` would make ambiguous with `Eigen::Product`. A nested `Spaces` namespace, which would not
avoid that and would lengthen every tag.

**Where.** `Spaces/Discrete.h`, `Distributions/Categorical.h`.

### sizes-and-dimensions

*`size()` counts elements, and a dimension says whether it is of values or differences.*

**Rule.** A distribution's `size()` counts its elements: mixture components, particles, grid
cells, modes, conditionals. A distribution of one element has none. `valueDimension()` is the
length of a mean or a sample, and `differenceDimension()` the rows of a covariance and the degrees
of freedom; there is no bare `dimension()`. A space implements only the member `valueDimension()`,
and generic code calls the free functions. A discrete space has a cardinality and no dimension; a
`DiscreteProduct`'s parts are axes. The names in full are in style.md, Types.

**Why.** A value and a difference may differ in size (a unit quaternion has 4 coordinates and 3
degrees of freedom), so a Gaussian has two dimensions, and one `size()` can't be both. Neither gets
the bare name: the manifold's dimension is the difference dimension, while most readers would take
`dimension()` to be the length of the mean. `size()` counting elements matches Eigen and the
standard library, where it counts what is iterated, and leaving it off a single density makes a
size trigger on a `GaussianParams` a compile error rather than a no-op. "Axis" frees "dimension"
for its continuous meaning.

**Rejected.** A space's `size()`, `FixedSize`, `Stride`, and `MaxSize`: a space's `size()` counted
coordinates, against the rule that `size()` counts elements. `Stride` (not a dimension) and
`ElementValueDimension`, for what is now `ValueDimensionStep`. `isValidValueDimension`, for what
is now `allowsValueDimension`. A `constexpr` free `valueDimension()`: it can't be constant for a
resizable space or for a user's space without `constexpr` members, so `valueDimensionAtCompileTime<S>`
is the compile-time spelling, as Eigen has `RowsAtCompileTime` beside `rows()`.

**Where.** `Spaces/Space.h`; style.md, Types.

### index-names

*The index utilities have short names.*

**Rule.** The functions that turn a tag path into an Eigen index are `idx()` for values and
`diffIdx()` for differences, and the ones that give an Eigen object's block for a path are `block()`
for values and `diffBlock()` for differences; `diff` is the one exception to full names before the
closed list of [params-and-density]. Each is one name with overloads told apart by their arguments:
`idx<Space, Path>()`, `idx<Path>(space)`, `idx<Path>(space, k)`, and `idx<Space, Path>(k)`, the first
a constant expression where the path names a single coordinate; `block<Path>(space, x)`,
`block<Path>(space, x, k)`, and `block<Space, Path>(x)`, each Eigen's block of the rows, sized at
compile time where the type fixes the size, and `diffBlock` with the same three; and for a matrix
over differences whose rows and columns may be in different spaces, such as a Jacobian,
`diffBlock<Row, Column>(rows, columns, m)`, `diffBlock<Row, Column>(space, m)`, and
`diffBlock<Space, Row, Column>(m)`. A path as a value is `path<Tags...>` [tag-addressing].

**Why.** These names sit inside subscripts and on both sides of an assignment, often two to a line
(`m_cov(diffIdx<S, RowPath>(), diffIdx<S, ColPath>())`, `block<Position>(out, y) = block<Position>(in, x) + dt * v`),
where longer names pushed ordinary lines past the column limit and buried the paths, which are what
the reader needs to see. `diff` reads as "difference" without a lookup; a bare `d` could mean delta,
derivative, or dimension. `block` is Eigen's word for the same thing, and the free function collides
with neither `Composite::block<I>()` nor Eigen's `.block()`, which are members.

**Rejected.** `differenceIdx`, for the length. A variable template for the compile-time form beside
`idxIn()` and `diffIdxIn()`: every form names its space, as a type or an object, so "In" said
nothing the arguments don't, and a reader had to remember which form had it. `tagPath` for the path
value: `path` collides only with `std::filesystem::path`, and only under two using-directives at
once, which model code has no reason to write. `segment` for the accessor, Eigen's name for a
vector's block: the same call takes a matrix of points and a block of a covariance.

**Where.** `Spaces/TagPath.h`, `Spaces/Indexing.h`.

### covariance-reset

*A covariance moves with its mean, by each block's reset Jacobian, and costs nothing where that
Jacobian is the identity.*

**Rule.** An update that moves the mean by a step, $x^+ = x \boxplus \hat\delta$, and computes the
covariance in the tangent space at $x$, moves it to the tangent space at $x^+$ by the reset
Jacobian, $P^+ \leftarrow G P G^T$. The space supplies $G$ block by block: the identity for a block
without its own `plus()` or one that declares `identityReset`, a closed form where it has
`resetJacobian()`, and otherwise autodiff of its `plus()` and `minus()`. Where every block has the
identity, the reset compiles to nothing. It applies to every Kalman-family update, a Gaussian's
conditional, and the information forms' reference [gaussian-forms]; where one covariance is shared
by points that moved by different steps, it is left out as part of that approximation, and
`PerPointCovariance` applies it per point.

**Why.** Without the reset a covariance is labeled with a mean it was not computed at, and a
hand-written error-state filter does it (Solà 2017, §6.3; Mueller, Hehn, and D'Andrea 2017, who
correct its omission in Markley 2003). $G$ belongs to the space, not to the measurement as the
update's form does [covariance-update-forms], because it is geometry that every update moving a
mean needs. It is decided at compile time because most spaces need nothing, and the benchmarks
[benchmarks] compare against hand code that never multiplies by an identity. Autodiff is the
default because it is correct with no work from the space's author.

**Rejected.** A compile error for a space with its own `plus()` and no Jacobian: autodiff makes the
default correct. A dense $n \times n$ $G$, which would multiply mostly by the identity.

**Evidence.** `tests/Spaces/reset.cpp`; the attitude problem in the benchmarks, with the closed
form and the autodiff arms.

**Where.** `Spaces/Reset.h`.

### unit-quaternion

*The library has a unit quaternion space, stored as Eigen stores a quaternion and viewed through a
map.*

**Rule.** `UnitQuaternion` stores (x, y, z, w), the order of `Eigen::Quaternion::coeffs()`, so a
state's attitude block is exactly what an `Eigen::Quaternion` stores, and its product is Eigen's
(Hamilton's). Differences are rotation vectors in the body frame:
$q \boxplus \delta = q \otimes \mathrm{Exp}(\delta)$ and
$q \boxminus p = \mathrm{Log}(p^{-1} \otimes q)$ (Solà, Deray, and Atchuthan 2018). `normalize()`
keeps $w \ge 0$. The exponential and logarithm are the space's own, smooth at zero, and its reset
Jacobian is the closed-form right Jacobian [covariance-reset]. `UnitQuaternion::map()` returns an
`Eigen::Map` over a block, always a view, never a copy; a block it can't view, or a plain matrix
passed as a temporary, is a compile error. `map()` is a member of the space, not part of the `Space`
concept, since nothing generic calls it. It is the library's SO(3) [lie-groups].

**Why.** Attitude is the most common state that is not a plain vector, and a user who writes the
space must get the small-angle series, the double cover, and autodiff-safe branches right. Keeping
Eigen's storage means the user's quaternion and the filter's are one object, not a conversion
with an order to get wrong. A map is the only way to write in place and costs nothing in a debug
build; a map that silently became a copy for an expression would carry a lifetime rule that held
only sometimes.

**Rejected.** (w, x, y, z), which matches Eigen's four-scalar constructor but not its storage, so
every view would be a copy. A `toEigen()` that returns a copy, and a `fromEigen()`: writing through
the map does both. A generic `view(space, block)` customization point, which waits for a second
space that wants a typed form (S², SE(3), geodetic).

**Where.** `Spaces/UnitQuaternion.h`.

### lie-groups

*Lie groups are spaces in this library, and navigation is domain models written on them.*

**Rule.** The Lie groups a state lives on are spaces in `Spaces/`: SO(3) as `UnitQuaternion`
[unit-quaternion], and $SE(3)$ and $SE_2(3)$ beside it when the invariant EKF needs them, each with
the side its increment is applied on. EmrysNavigation holds the domain: inertial mechanization, IMU
preintegration, wheel odometry, legged contact, GNSS, magnetometers, gravity, and geodetic frames,
written on these spaces. It
launches when it has those models, not with the spaces as its content. `Core`, `Spaces`, and
`Models` move to a base project only when a third project consumes them.

**Why.** A group is geometry, not navigation. The optimizer, the graph library, the controller, and
the sampler need $SE(3)$ and none of them navigates, so groups in EmrysNavigation would make each
depend on it. The quaternion is also the one space whose differences have another size than its
values: the tests run the manifold paths through it and the attitude example's state is one, so
outside the library the tests would need a copy. A navigation project holding only spaces would
launch with nothing a navigation user came for.

**Rejected.** The groups in EmrysNavigation, launched alongside this library. A base project for
the spaces now, with one consumer. Leaving every group to the user, which hands each one the
small-angle series and Jacobians that [unit-quaternion] exists to spare them.

**Where.** `Spaces/UnitQuaternion.h`; the pose groups are not built.

### linear-space

*The plain space is named Linear, for its structure rather than its set.*

**Rule.** The space whose values are any real vectors, with $x \boxplus \delta = x + \delta$ and
$a \boxminus b = a - b$, is `Linear<N>`.

**Why.** What the library uses about that space is its structure, not its set: `isLinear<>`,
`hasPlainValues<>`, and the one-pass weighted moments depend on `plus()` and `minus()` being
addition and subtraction, and $\mathbb{R}^n$ with a `plus()` of its own would be the same set and no
longer `Linear`. "Linear space" is the standard synonym for a vector space. That "linear" also
describes models is handled by the rule to say "affine" for $Ax + b$.

**Rejected.** `R<3>` (breaks the rule of full names). `Real<3>` (reads as a number type, and
collides with `Real` in `Eigen::NumTraits`). `Reals<3>` and `RealSpace<3>` (read awkwardly in
`GaussianParams<Reals<3>>`). `Euclidean<3>`, the runner-up, which names the metric the filters
assume but makes no claim about the operations, which are what the code checks.

**Where.** `Spaces/Coordinates.h`.

### no-bounded-coordinates

*There is no bounded coordinate type.*

**Rule.** The coordinate types are `Linear` and `Periodic`. A quantity that can't be negative, such
as a range or a speed, is `Linear`. Where a filter must keep it nonnegative, the user estimates the
vector it is the length of and measures the length, or estimates its log. A distribution with a
bounded support states the support itself: `Gamma` is zero below zero, and `Uniform` takes an
explicit box.

**Why.** A coordinate that reflects a value back across a bound makes $x \boxplus \delta$ nonlinear,
while every Gaussian operation treats it as the identity: samples were folded and the density was
not, so near a bound they disagreed (with mean 0.2 and $\sigma = 1$, 42% of the density's mass lay
below zero, and the samples' mean was 0.817), and a reflected mean kept the old sign of its
cross-covariances. The reset [covariance-reset] could carry that sign through an update but not
through a prediction. Reflection is the physical answer only for the length of a vector that passes
through the origin, and there the standard model is the vector. A bounded type can be added after
the first release without breaking any user; changing what a released one means could not.

**Rejected.** `Bounded<{.lower, .upper}>` and `Magnitude<>`, which were removed for the reasons
above. Folding the density to match the samples, which breaks the quadratic form that conditionals
and updates rely on. Refusing a Gaussian on a bounded space, which would have to reach every place a
Gaussian is built while particle and kernel densities still disagreed with their samples.

**Where.** `Spaces/Coordinates.h`.

### several-resizable-blocks

*A composite may have several resizable blocks, and only a space with at most one takes its sizes
from a length.*

**Rule.** A `Composite` may have any number of resizable blocks, in any position and at any depth,
so a state can hold several growing sets (`Composite<Pose, Tracks, Landmarks>`). A block after the
first resizable one has its offset only at run time. Only a space with at most one resizable block
takes its sizes from a length: it has `resize(n)`, and the constructors and resizes that infer
sizes from a length require it. A space with several is built with its counts, and grows and
shrinks one block at a time through `insert<Path>()` and `erase<Path>()`. A resizable block inside a
`Repeated` stays rejected: ragged data flattens into sibling sets (`Composite<Fleet, Sensors>`),
with each sensor's vehicle kept outside the state as a known association.

**Why.** A state with two growing sets could not be written otherwise. The single-block rule was
also order-dependent where the library builds a composite: `TagPaths<Ownship, Fleet>` compiled and
`TagPaths<Fleet, Ownship>` failed. Inferring the split from a length can't be made safe: the right
total split the wrong way (two tracks and five landmarks against data for three and three) would go
unnoticed, so with several blocks the counts must come from the caller's space, and the check is at
compile time.

**Rejected.** A resizable block inside a repeated one, which would need a run-time index per
repeated block crossed, a space per element, and index vectors in place of strided sequences
everywhere, for a case that sibling sets already express.

**Where.** `Spaces/Composite.h`.

### coordinate-parameters

*A coordinate type's parameters are template arguments, and an interval is spelled by its ends.*

**Rule.** `Periodic<{.lower, .upper}, N>` takes its half-open interval as a structural template
argument, so a tag needs no constructor, a fixed-size coordinate type is an empty type, and the
parameters are part of the type. Both ends must be given; an omitted one is NaN, which a
`static_assert` rejects. The common angles are named by the ends of their interval, as MATLAB's
`wrapToPi` family is: `RadiansPi`, `Radians2Pi`, `Degrees180`, `Degrees360`. They name behavior;
quantities such as a heading are the user's tags. Wrapping keeps a value inside its interval
exactly, for every input. Parameters are `double` whatever the scalar, cast where used.

**Why.** Parameters in the type cost no storage, and a measurement written for a heading in degrees
can't be handed a state in radians. An interval written by its ends shows at the call site what it
is. Values known only at run time (DFT bins of a configured size is the one case known) are left
to a space the user writes.

**Rejected.** Constructor parameters, which made every tag write `using Periodic::Periodic;` and
made a fixed space carry state. A default interval for `Periodic<>`: `Periodic<{.upper = 360.0}>`
would hide its lower end. The spelling `{.period, .center}`: `{.period = 360.0}` silently meant
$[-180, 180)$. Per-repetition parameters in a `Repeated` block: no case needs them. A macro to
skip the exact-bounds checks: they run only for values outside the interval, and translation units
that disagreed on the macro would break the one-definition rule. SIMD wrapping, which waits for a
benchmark that shows normalization matters, since a periodic block in a larger state is a strided
row that Eigen does not vectorize.

**Where.** `Spaces/Coordinates.h`; `PeriodicWrapsValuesNearFarSeams` in `tests/Spaces/space.cpp`.

### difference-spaces

*A space has at most one normalize(); its differences live in an untagged difference space;
updates, residuals, and differences taken away are plus(), minus(), and less(), with defaults.*

**Rule.** A space may define `normalize()` for its values, `difference()` naming the space its
differences live in (itself if absent; every library space has one, so none carries a tag), and its
own `plus()` and `minus()`; a space with any of them must have `difference()`, and one whose
differences have another size must have both operations. The library calls only the free functions,
which default to `normalizeValues(space, x + d)` and `normalizeDifferences(space, a - b)`, and
`less(space, x, d)` is `plus(space, x, -d)`. Linearity is derived: `hasPlainValues<S>`,
`hasPlainDifferences<S>`, and `isLinear<S>` come from which members exist. A space says what it
does through members, which its tags inherit.

**Why.** Each space writes only what it does: an unwrapped heading has a difference space and no
`normalize()`. A $\boxplus$ that is not addition up to canonical form, as on a rotation or a
geodetic position, needs the base point, which a one-argument `normalize()` never sees. A derived
flag can't be wrong; a declared one could skip wrapping silently. A bias in `DifferenceSpace<Gyro>`
is not a `Gyro`. Applied by `less()` it is right on every space; bare `-` is right only where
`plus()` is addition and the library normalizes the result.

**Rejected.** A pair `normalizeValues()` and `normalizeDifferences()` that every non-linear space
defined together, which made one-sided spaces write an empty function. Flags marking a space as a
value or a difference space: a difference space can differ in more than a flag. Typed difference
vectors (`Difference<S>`), which would wrap Eigen's expression templates in every model and
Jacobian. `plus()` and `minus()` with no normalize, which would hide canonicalizing a value from
outside behind `plus(x, 0)`. A declared `isLinear` flag. A specializable `SpaceTraits<S>`: a
specialization doesn't reach a tag, and it would bring back declared flags. A linear tag as its own
difference space. A `minus(x, d)` overload. Requiring these in users' models: a user's trade.

**Where.** `Spaces/Space.h`.

### tag-addressing

*Blocks and axes are addressed by tag path, never by type or by index.*

**Rule.** A path names blocks, and discrete axes and values, by tag: a type derived from a space,
never a library space type such as `RadiansPi<>`, even where it is unique. A path of more than one
tag is a `TagPath`, so a template's arguments are a list of paths
(`marginal<TagPath<Ownship, Position>>()`, `covariance<P, Q>()`); as a function argument it is the
value `path<Ownship, Position>`. Distributions take paths, never index sets. A tag inside a
`Repeated` block is reached only through the block and does not clash with the same tag outside
it. A grid's cells are tagged by its block (`GridCells<Path>`, `PointCells<Path>`), and a regular
grid's axis by the innermost tagged block holding its coordinate (`GridAxis<Position, 0>`). Model
code reads an Eigen object's block with `block()` and `diffBlock()` [index-names], which take the
space and check the size: a vector's space is never inferred from its size. `idx()` and
`diffIdx()` stay for a user's own Eigen work.

**Why.** A bare type doesn't say what it selects. A path names a block, which has rows in both the
value and the difference space; an index set names rows of one, and stops addressing the other
once their sizes differ, as on a unit quaternion. Counting inner tags as ambiguous would make an
ownship beside a fleet of targets reachable only through a path. A vector read without its space
is read wrong the day two spaces share a size.

**Rejected.** Addressing an untagged block by its unique type. A path as a template pack
(`marginal<Ownship, Position>()`), which reads as two paths. An `operator/` on tag instances, which
doesn't work as a template argument. Index forms where the value and difference sizes agree: two
ways to do one thing, which break when a tangent size changes. A by-tag `axis<Cell>()`: generic
code reads an axis by position [discrete-spaces]. An accessor by path per space type: one
addressing call feeds every view, `UnitQuaternion::map(block<Attitude>(space, x))`.

**Where.** `Spaces/TagPath.h`, `Spaces/Indexing.h`, `Spaces/Discrete.h`,
`Distributions/GridGaussian.h`.

### tangent-gaussians

*A Gaussian on a non-linear space is a Gaussian in the tangent space at its mean; the wrapped normal
is not computed.*

**Rule.** The density is $\exp(-\tfrac12 (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu))$ with
the usual normalizer. On a `Periodic` coordinate this is valid while $\sigma$ is small against the
period: integrated over $[-\pi, \pi)$, the density has mass 1.000 at $\sigma = 0.5$, 0.998 at
$\sigma = 1$, and 0.884 at $\sigma = 2$. A spread on the order of the period is a distribution that
a Gaussian doesn't represent: use particles, a grid, or a mixture. A narrow Gaussian on the wrap
point is fine, since residuals take the short way round.

**Why.** Efficient execution comes first, and wherever a Gaussian is the right representation the
wrapped and unwrapped densities agree to that precision.

**Rejected.** The wrapped normal, a sum over periods that breaks the quadratic form that
conditionals and updates rely on, as folding a bounded density did [no-bounded-coordinates].

**Where.** `Distributions/Gaussian.h`; `Spaces/Coordinates.h`.

### space-storage

*A space costs what its state costs, and a whole distribution keeps its own copy.*

**Rule.** A `Composite` or `DiscreteProduct` stores only its parts with state, and an empty part is
one shared instance, so every fixed-size composite or product is an empty type and a distribution
over it is the size of one over `Linear<n>`. A resizable space stores only its counts. A space is
stored once per element wherever the element is a whole distribution: a mixture's components, a
hybrid's modes, and per-point conditionals each hold their own, kept in step by the container's
`insert()` and `erase()` and checked equal by its constructors. A space whose instance holds
nothing, `FixedSizeSpace` (empty and default-constructible), is named as a template argument rather
than passed, by `plus`, `minus`, `normalizeValues`, `normalizeDifferences`, `simulate`, `transform`,
`linearized`, `unscented`, `logLikelihoods`, `logLikelihood`, `valueOf`, `pointGrid`, `regularGrid`
(`regularGrid<State, Position>(first, spacing, counts)`), `CategoricalParams::fromWeights`, and the
constructors of `MarkovChain` and `ContinuousTimeMarkovChain` (`MarkovChain<Maneuver>{matrix}`), as
the constructors that size their space from the vectors already leave it out. A fixed step names no
space: `fixedStep(model, dt)` leaves it to the filter it is given [fixed-step-models].

**Why.** The library is meant to be as lean as Eigen's fixed-size types: a `std::tuple` of empty
leaves added 8 bytes to a `GaussianParams`. A per-element copy is only a resizable space's counts,
8 bytes against the kilobytes of a covariance at its capacity.

**Rejected.** Components that borrow their container's space, through a component concept or a
template parameter on `GaussianParams`: a cost on the most-used type to save a count. A trailing
space defaulted to `S{}`: it still names `S`, and `simulate` ends in its inputs. Leaving out a
resizable space: a function would build it on every call, from counts it can't always read. A named
space for `reads()` and `given()`, whose arguments are the paths they read: a tag is a space, so
`reads(path<Position>, path<RangeBearing>)` reads two blocks, and a wrapper saves nothing.

**Where.** `Spaces/` (`Composite.h`, `Discrete.h`, `Space.h`), `Models/` (`Simulation.h`, `Transform.h`,
`MeasurementModel.h`), `Distributions/` (`Categorical.h`, `GridGaussian.h`, `ContinuousTimeMarkovChain.h`).

## Distributions

### upkeep

*Upkeep is a set of function objects the caller applies to a filter's state.*

**Rule.** A filter never keeps its representation in bounds on its own. The caller applies function objects from
`upkeep` [policy-factories] to any filter's `state()`: actions such as `resample()` and `pruneBelow()`, triggers that
run them when a condition holds (`when()`, and `whenEss()` and `whenSize()` with a comparator first), and
`sequence()`, which runs them all. Every filter holds one, given last to its constructor and `upkeep::None` unless
given, and runs it only when the caller calls `maintain()`. An `UpkeepOperation` is called as `h(state)`, holds what
else it needs, and returns `std::expected<bool, NumericError>`, whether it changed the state or the error that stopped
it: a library action's leaves the state as it was, a trigger's what its earlier actions did. A caller's own action
meets it [user-extension-points]. A state of weighted elements prunes by an absolute threshold on each normalized
weight, by a total below a fraction, and to a size, removing the lightest, the later of equal ones first, and returns
the log of the fraction removed. The caller merges and splits mixture components, from `merged()`, `merge()`,
`split()`, and [divergences]. `roughen()` uses a bandwidth stored since the last `resample()`, else Silverman's.

**Why.** The right moment differs by operation: resampling before a prediction, merging after updates that
multiply components, repairing after updates that rounding left indefinite [covariance-update-forms]; a hook
inside the filter would fix one [composed-features], and `maintain()` fixes none. A held upkeep is one value
of the filter's setup, as its method is. Returning only whether the state changed keeps actions composable,
and a comparator puts its direction at the call. A threshold on each element doesn't bound the total
removed, which grows with the count and each `subdivide()`; a fraction bounds it, and a size the cost. Which
merges to make depends on the use and the hardware; nothing in the literature rules splitting [splitting-hooks].

**Rejected.** Arguments after the state: each action took all its trigger's. Rolling back on an error: a copy of
the state at every call [allocation]. `MergeByCost`, Runnalls' greedy reduction: its search suits some uses and
not others, and an action the library names reads as a recommendation. Silverman's rule alone: `setBandwidth()`
could not steer it. `whenFewer()`: only one of a pair said its direction. `filter.maintain(h)`: no more than
`h(filter.state())`. A setter: the upkeep's type is the filter's. Unconstrained actions: each claimed any state.

**Where.** `Distributions/Upkeep.h`, `Filters/FilterUpkeep.h`, `Core/Weights.h`, `examples/gaussian_mixture.cpp`.

### rao-blackwellized-core

*Rao-Blackwellized states share one core, and upkeep keeps a state's type.*

**Rule.** The Rao-Blackwellized filters represent
$p(x) = \sum_i w_i \, K_i(x_p) \, \mathcal{N}(x_r; \mu_i, \Sigma_i)$ and differ along three
independent axes: the points (fixed grid cells, or particles that move and are resampled), the
kernel about each point (a point mass, or a cell covariance), and the conditional covariance
(shared, or one per point). Code that depends only on how the conditionals are stored is written
once, in `Conditionals.h` and `RbUpdate.h`; prediction and weights stay with each filter. The
point-mass filter's state, `GridGaussianParams`, stores only its live cells, and a cell whose
probability reaches zero is removed. A regular grid is refined only uniformly, by `upkeep::subdivide(k)`,
which splits every live cell; upkeep never changes a state's type. A grid of point masses subdivides
too, though spreading each mass over its cell widens the distribution: whether that suits the grid
is the caller's choice, and a caller whose points mean exactly what they say doesn't subdivide.

**Why.** Nothing revives a cell, so a dense layout only cost memory and a check per dead cell at
every step: after pruning 10,201 cells to under 200, every step still visited all 10,201.
Refinement makes a dense layout impossible: three halvings of a 101 × 101 grid have 650,000 cells.
Writing the update once matters most for the shared-covariance form, the subtle one, which
particles need too. Splitting every cell at once keeps the cells alike, so a shared cell covariance
stays shared and a filter keeps the cost model it was chosen for. Grid and particle states stay
separate types because a cell is never resampled (two copies of a cell would evolve identically)
and a particle is not a value of a discrete space.

**Rejected.** A dense layout over every cell, as `Categorical` has, which suits the IMM, where the
chain revives a mode of probability zero. Splitting chosen cells, which needs a covariance per
cell, a different type (rao-blackwellized-filters.md Q20).

**Where.** `Distributions/Conditionals.h`, `Distributions/GridGaussian.h`,
`Distributions/ParticleGaussian.h`, `Filters/RbUpdate.h`.

### density-names

*Continuous distributions evaluate `logDensity()` and `density()`; discrete ones a `probability()`.*

**Rule.** A continuous distribution evaluates `logDensity(x)` and `density(x)`. A discrete one
evaluates `logProbability()` and `probability()` of a value or an event, and `logProbabilities()`
and `probabilities()` for the whole vector. There is no `pdf()`, `pmf()`, or `operator()`. A
measurement's `logLikelihood()` keeps its name, since it is a function of the state.

**Why.** `probability` already names more than a mass function's value: the probability of an
event, and the weights of modes and cells. "Density" is already the library's word, and it stays
correct for a joint of discrete and continuous parts such as a hybrid's, a density with respect to
counting measure times Lebesgue measure, which `pmf` and `pdf` can't name.

**Rejected.** `pmf()`, which would be a third name beside `probability()`. An `operator()`, which
would return the linear density, the one that underflows, would read as coefficient access in
Eigen, and can't express an event.

**Where.** style.md, Types.

### divergences

*Divergences are free functions over factorized views, and only Wasserstein takes a ground metric.*

**Rule.** Divergences are free functions taking their arguments in the written order,
$D(p \parallel q)$ as `f(p, q)`: `kullbackLeibler()`, `bhattacharyya()`, `hellinger()`,
`jensenShannon()`, and `wasserstein()` (2-Wasserstein). Distances are returned unsquared, in
natural logarithms. Closed forms take two densities of one space, categorical or Gaussian in any
pair of the four forms, and return a plain `Scalar`; the caller's `toDensity()` is the step that
can fail. Every other pair is estimated by passing a method first, `kullbackLeibler(MonteCarlo{n},
p, q, rng)`, which returns the value with its standard error; a call without a method where no
closed form exists fails to compile with one error naming the estimator. Only Wasserstein takes a
ground metric (`GroundMetric`), a pseudometric if singular; `expectedSquaredError()` is its square
to a point mass. `kullbackLeiblerCollapseBounds(p)` brackets the divergence of a Gaussian mixture, a
`HybridGaussian`, or a mixture of them from its collapse. On a manifold, the closed forms compare
covariances in tangent spaces at different means without transporting them.

**Why.** Every binary operation in the library is a free function, and these are cross-type. A
plain number from the closed forms fits a test's assertion or a gate's threshold. No function picks
an estimator for the caller, because an estimate needs a random generator and has an error, and a
call that hid either would read as exact. Distances are unsquared because thresholds are written in
them and the triangle inequality holds for them. The other divergences are unchanged by any
invertible map applied to both distributions, so a weighting of coordinates means nothing to them;
Wasserstein measures in the space's units, so its metric expresses what an error costs. A general
matrix, rather than per-block scales, lets one metric express a mix of blocks, such as the position
a state predicts $T$ ahead.

**Rejected.** Wasserstein between mixtures or particle sets, which needs an optimal-transport
solver.

**Where.** `Distributions/Divergence.h`.

### uniform-mixtures

*A uniform is approximated by a mixture on a regular grid, flat by default rather than
moment-matched.*

**Rule.** `uniformComponents(uniform, counts, relativeWidth = 0.6)` gives equal-weight components at
the centers of equal cells along each coordinate, with standard deviation `relativeWidth` times the
cell's width; periodic coordinates wrap. Given a `Product`, it grids the `Uniform` factors and
gives every component the mean and covariance of each `GaussianParams` factor. The result is a
`UniformGrid`, a random-access range of components made when read, which
`GaussianMixtureParams::assign()` puts in a mixture the caller made.

**Why.** The ripple of the sum depends on the relative width alone, so a width relative to the
spacing keeps it the same on every coordinate whatever the units; 0.6 is flat to 0.16%. A regular
grid tiles both a box and a torus, and periodic coordinates, such as an unknown heading, are the
main use. A state is rarely unknown in every block, and a `Product` already says by tag path which
blocks to grid and which to hold fixed. The caller makes the mixture because it has a capacity and
an allocator, and may already be a filter's state; the range computes each component from its
index, so it allocates nothing.

**Rejected.** The moment-matched width $1/\sqrt{12}$ as default: exact mean and covariance, but a
±39% ripple. Lattices whose duals pack densest, which need 13% to 29% fewer components but clip
raggedly at a box's faces and wrap along at most one periodic coordinate. Rank-1 lattices, which
neither tile a box nor wrap. A `PointGrid`, which is for one block of a state. The other blocks'
Gaussians as extra arguments: the call would need the whole space, and the optional width could not
come last.
A `std::generator`: it allocates its frame, gives no size, and libc++ lacks it. A
`uniformMixture()` returning a new mixture: it takes the allocator but not the capacity.

**Where.** `Distributions/UniformMixture.h`, which documents the ripple for each width.

### component-first

*A type that holds several Gaussians takes its component as its first parameter, and a hybrid state
is a component.*

**Rule.** `MixtureParams<Component, Allocator>` and `HybridParams<Component, ModeSpace, Allocator>`
hold one component per element, with Gaussian aliases (`GaussianMixtureParams`,
`HybridGaussianParams`). A `Component` provides the space, `factorize()`, and through
`ComponentTraits` its log-density, moments, and collapse. `GaussianParams`, `SqrtGaussianParams`
[gaussian-forms], and `Hybrid` are components, so `MixtureParams<HybridGaussian<State, Mode>>` is a
mixture of IMM filters' state [filters-and-recursions]. The information forms are not: a
`static_assert` names `toGaussian()` and `toSqrt()`. What only a Gaussian has (moments in or out,
marginal and conditional views, sampling) stays on the containers under `requires`. A type over a
structure that carries the space takes it first and defaults its component from it:
`GridGaussianParams<Grid, Covariance, Component = Gaussian<Rest>>` [rb-point-mass], or, where the
rest is long to spell, last and checked: `ParticleGaussianParams<..., Scalar, Allocator, Component>`.

**Why.** The atom (a covariance Gaussian, a root, a hybrid state) and the structure over atoms
(weights, modes, points) are independent choices; as a parameter, a mixture of IMMs or a
square-root mixture is a spelling rather than a copy of every structure. The component comes first
because it carries the space and the scalar, which would otherwise be repeated and could disagree;
a grid carries them, so it comes first instead. `Hybrid` is a component, not a case the mixture
filter knows, because the mixture needs nothing beyond the concept. The information forms are for
multiplying densities, which a mixture adds, and their collapse would fail on a component with no
information in a direction, the case they exist for.

**Rejected.** The component as a defaulted parameter after the allocator, which repeats the space
and scalar. The information forms as components with a collapse that can fail.

**Where.** `Distributions/Component.h`, `Distributions/Mixture.h`, `Distributions/Hybrid.h`,
`Distributions/GridGaussian.h`; style.md, Types.

### mixture-entries

*A mixture's entries are addressed by index and handed out read-only, and a reweighting callback is
its own member.*

**Rule.** A `MixtureParams` addresses an entry by its index, as a `ParticleSet` does: `mixture[i]`
reads one, on the parameters and on the density, and `remove()`, `setComponent()`, the setters of a
Gaussian's moments, `setLogWeight()`, `scaleWeight()`, `merged()`, `merge()`, and `split()` take an
`Index`; `add()` and `merge()` return one. `begin()` and `end()` read the entries in order.
Components are handed out `const`: `setComponent()` replaces one and `updateComponent(f)` changes
it in place, each asserting that the space is kept, and a `Hybrid`'s `setConditional()` and
`updateConditional()` do the same for its modes. On a mixture and a particle set, `forEach()` takes
a callback that returns nothing, and `reweight()` one that returns a floating-point log scale; a
callback of the other kind fails a `static_assert` that names the right member.

**Why.** Every caller that addresses an entry, the mixture filter, the IMM, the mixture smoother,
the lineage, and the update branches, holds an index. An index stays valid when `add()`
reallocates, where an iterator does not, and removal moves the last entry into the gap either way,
so an iterator documented no invalidation that an index hides. Through a mutable reference a
caller could change a component's space, which nothing checked, and a callback whose every return
was a log scale took a `bool` as one.

**Rejected.** Entries addressed by `const_iterator`: each caller made one from its index with
`begin() + i`, and the mixture turned it back with `indexOf()`. A mutable `component()`, through
which the IMM updated a hybrid's conditionals in place: `updateConditional()` does that and checks
the space.

**Where.** `Distributions/Mixture.h`, `Distributions/ParticleSet.h`, `Distributions/Hybrid.h`.

### gaussian-forms

*The Gaussian has four forms, each its own parameters type with the same accessors, and a form folds
a linearization in.*

**Rule.** A Gaussian is kept as `GaussianParams` ($\Sigma$), `SqrtGaussianParams` (lower $S$,
$S S^T = \Sigma$), `InfoGaussianParams` ($\Lambda$ and $\eta$ about a reference [covariance-reset]),
or `SqrtInfoGaussianParams` (upper $R$, $R^T R = \Lambda$). The stored matrix is the invariant; the
rest is derived on request through the same fallible accessors and `to*()` conversions, and only the
factorization is cached. A failure is a `GaussianError`, never a silent repair. Each form folds the
one linearization [linearization-forms] in its own way; a prior with no information in a direction
gives an innovation of infinite covariance. The square-root form is a component [component-first]
and an `UncertainInput` by its `root()`; the information forms, with no moments for a flat prior,
are neither. A step returns a `std::expected` exactly when it can fail on valid inputs.

**Why.** Each form has one operation that is free and one that can fail, and they rotate (a flat
prior is impossible as a covariance and $\Lambda = 0$ as information), so the choice belongs to the
problem; shared spaces and accessors make it cheap, and return types differ because the
differences are why the forms exist. Splitting at the linearization puts each inverse where it is
held and lets one recursion serve four states. A flat prior takes the update, since taking it is
the point of the information form. A root as an input admits a user's root-only type.

**Rejected.** A form flag on `GaussianParams`: the invariant differs, and every member would check
it. One triangle for every root, which factorizes twice in the crossing accessors. Views for the
information forms' marginals, which would own their matrices anyway. Caching every derived matrix:
three $n \times n$ caches more in each parameters type.

**Evidence.** `tests/Distributions/gaussian_forms.cpp`, `tests/Filters/info_filter.cpp`,
`tests/Filters/sqrt_filters.cpp`, `tests/allocation_forms.cpp`.

**Where.** `Distributions/GaussianForm.h` and the three form headers, `Models/FormSteps.h`.

### roots-and-factors

*A root is square and a factor need not be; the Gaussian forms' matrices are named as roots.*

**Rule.** A root of a covariance $\Sigma$ is a square $C$ with $C C^T = \Sigma$, and a root of a
precision $\Lambda$ a square $P$ with $P^T P = \Lambda$, so that $P$ whitens. A factor is any $F$,
possibly wide, with $F F^T = \Sigma$, such as the array a square-root step triangularizes;
`rootFromFactor()` takes one. A product's components are also its factors [product-factors], a
separate sense. Every Gaussian form has the fallible pair `covarianceRoot()` and `precisionRoot()`
[gaussian-forms]. Beside them, as `covarianceLower()` is beside `covariance()` [params-and-density],
the square-root form's stored $S$ is `root()` and the square-root information form's stored $R$ is
`infoRoot()`. Internal names follow the same rule, as `lowerTriangularRoot()` gives the
lower-triangular root of $F F^T$. Prose keeps "Cholesky factor" for the triangle of that
factorization, as the literature does.

**Why.** The library already used "root" for the square matrix: `roots::` makes them, axes return
one from `root(cov)`, and the form filters' read blocks and the transform's linearization expose
`root()`. Naming the stored $S$ `factor()` gave it a second name, the same one as $R$ in another form
and a product's component, so the transform could not read $S$ by one name from both the
parameters and the read blocks. $R$ takes `infoRoot()` because it is a root of
the precision, not the covariance: code that reads `root()` as a root of $\Sigma$ would take the
wrong matrix without an error, and `precisionRoot()` is already every form's fallible accessor.

**Rejected.** `factor()` for the stored matrices, which gave "factor" three meanings. `root()` for
$R$, for the reason above. `precisionRoot()` returning $R$ by reference in that form alone: generic
code would have no single return type. `whitener()` for $R$, a term the library uses nowhere else.

**Where.** `Distributions/SqrtGaussian.h`, `Distributions/SqrtInfoGaussian.h`,
`Distributions/GaussianForm.h`, `Distributions/Roots.h`.

### form-densities

*The four forms are read alike through their densities, where nothing fails, and their parameters
keep the failures.*

**Rule.** The `GaussianForm` concept covers the parameters, whose operations fail differently by
form: an information form's mean, moments, and marginal fail while its precision is not positive
definite. Code generic over the forms reads them through `toDensity()`, the one check, after which
nothing fails: `FormGaussian` has what `Gaussian` has but `covarianceLower()`, and its marginals
are parameters of the same form. The covariance form's `GaussianMarginal` view is a `GaussianForm`
too. A `Product` takes any form as a factor, checked by `toDensity()`.

**Why.** At the parameter level a form's failures are part of its meaning: an information form
with no information in some direction is valid, and its marginal over that direction is not. Once
`toDensity()` has succeeded the precision is positive definite, so the four densities share one
contract without a `std::expected` that never holds an error.

**Rejected.** A `marginalCopy<Path>()` on `GaussianParams` so that every form's parameters return
parameters: it adds a copy, and generic code would still unwrap a `std::expected` per form.
Requiring `moments()` of a `Product` factor's parameters, which would reject the forms.

**Evidence.** `tests/Distributions/gaussian_forms.cpp` (`FormDensitiesReadAsTheGaussian`,
`InfoDensityMarginalIsTheSchurComplement`, `GaussianMarginalAccessors`),
`tests/Distributions/product.cpp` (`TakesEveryFormOfTheGaussian`).

**Where.** `Distributions/GaussianForm.h`, `Distributions/Gaussian.h` (`GaussianMarginal`), the
three form headers, `Distributions/Product.h`.

### element-marginals

*A marginal reads its source rather than copying it, and a marginal of a distribution made of
elements iterates them.*

**Rule.** `marginal<Path>()` returns a view of its source wherever it can read the source without
computing anything, and `copy()` on the view gives parameters. A distribution made of elements (a
mixture's components, a hybrid's modes, a particle set's particles) gives a marginal that is a
random-access range over the elements, each made when read and holding its log-weight and its own
marginal. The selection is resolved once, by the range, and every element borrows it, so an element
must not outlive its range. Marginals that must compute their result stay values: the square-root
and information forms, a categorical or grid over one axis, and the small distributions that are
their own parameters.

**Why.** Most marginals were copies, and a marginal of a mixture was a copy of every component. A
range gives what such a marginal is, a weighted set of the elements' marginals, lets the caller
read one component's marginal without copying the others, and allocates nothing even for a space
sized at run time.

**Rejected.** A `std::generator`, whose frame is allocated [allocation]. A composition of standard
views, which can't hold the shared selection, `space()`, and `copy()`, and has no name for error
messages.

**Where.** `Distributions/Mixture.h`, `Distributions/Hybrid.h`, `Distributions/ParticleSet.h`. The
Rao-Blackwellized states don't have element marginals yet (rao-blackwellized-filters.md Q9).

### categorical-params

*A categorical distribution has parameters, whose weights are written unchecked and checked once by
`toDensity()`.*

**Rule.** `CategoricalParams` holds one log-weight per value, which need not sum to one, and is
written without a check, by every value, by one value, or by adding log-likelihoods. Its
`toDensity()` returns a `WeightError` if the weights are all zero or one is NaN or $+\infty$, and
otherwise a `Categorical` that reads each log-probability as a log-weight less their total.
`update(logLikelihoods)` stays on the parameters, checks before and after, and returns the
log-evidence. Everything that held a `Categorical` holds parameters, including a `Hybrid`'s modes,
whose weights need not sum to one. The library's own steps produce normalized weights and read them
without a check that can't fail.

**Why.** The invariant, that the weights don't all vanish, is one that no single write keeps or
breaks alone: filling a histogram, ruling a value out, or adding several sensors' likelihoods before
one normalization passes through states that fail it. A check at every write forced all of that into
plain arrays without the space's indexing. A covariance has the same split for another reason (a
check costs a factorization), and the categorical adds a second: a check that only the finished set
of values can pass. Mixtures and particle sets already stored weights that need not sum to one. The
view keeps the log of the total, rather than the parameters caching it, so that `toDensity()` stays
`const` and writes nothing.

**Rejected.** Normalizing at every write, as `Categorical` did. A `StudentTParams`, until a filter
writes a Student-t after building it; `Gamma` and `Uniform` have no invariant across values that a
write could break part way.

**Where.** `Distributions/Categorical.h`.

### checked-factories

*A distribution that is its own parameters is built by a factory that checks them, and its
constructor is public only where building cannot fail.*

**Rule.** `Gamma`, `Uniform`, and `StudentT` are built by a static `create()` (and named factories
such as `Gamma::exponential()`) that checks the values and returns a `std::expected` with the type's
own error enum; the constructor that takes checked values is private. A constructor that cannot
fail, such as `Uniform(space)` over whole periods, stays public. What the types already decide stays
an assertion or a `static_assert`. Each error prints in words [printing-and-simulation].

**Why.** The constructors asserted on values a user often computes at run time, a rate from data or
bounds from a configuration, where a failure is an outcome to handle and not a programming error
[embedded-builds]. A public constructor beside a factory would leave the unchecked path open. A
constructor that can't fail needs no factory, which keeps `Uniform<Heading>{}` short.

**Rejected.** A `StudentTError` that is a variant of `GaussianError` and an error for the degrees of
freedom: one flat enum lets a caller switch on one type, at the cost of two cases that repeat a
`GaussianError`'s.

**Where.** `Distributions/Gamma.h`, `Distributions/Uniform.h`, `Distributions/StudentT.h`.

### product-factors

*A product's factors are placed by path with no space, as `Factors`, and one factor may cover every
repetition of a block.*

**Rule.** `Factors{on<A>(a), onEach<B>(b)}` holds a product's factors placed by tag path in no
space. It is not a distribution, since it has no sizes. `Product{space, factors}` places them in one
space instance, the only way to place them, as `Product{space, on<A>(a), ...}` does. `onEach<Path>(f)`
takes a factor in the space of one repetition of a repeated block and places it on every repetition
the instance has, as an `EachRepetition`, so a Gaussian factor gives the block $I \otimes R$. Each
`onEach` takes its own block's count, so the factors follow a space with several resizable blocks. A
path may name the whole space, and `onEach<>(f)`, with none, covers every repetition of a space that
has no tag, such as a measurement model's inferred output space. `on()` and `onEach()` belong to the
tag paths, and place a matrix too, in a linear model's `Columns` [linear-sensors], and a model's
noise in `additiveNoise()` and `continuousNoise()`, which take the placed parts themselves rather than
a `Factors` [model-policies].

**Why.** A product resolved its factors against one instance when built, so it could not follow a
space whose counts change, and a factor on a repeated block had to cover the whole block, so
"each landmark independently" could not be said. Additive measurement noise whose size follows the
state's counts needs both [measurement-space-functions].

**Rejected.** Sensors and noise taking `on<>()` arguments directly: a second spelling of a product.
`additiveNoise(Product{...})` for such noise: bound to one count. Inferring the repetition from a
factor's space being the block's element space: the call site would not say what it covers
[tag-addressing]. The name `Unplaced...`: an adjective for what the object lacks.

**Evidence.** `tests/Distributions/product.cpp`: an `onEach` factor's moments, log-density, and
draws, and one `Factors` placed in spaces of two counts.

**Where.** `Distributions/Product.h`, `Spaces/TagPath.h`.

### covariance-repair

*A covariance that fails to factorize is repaired only on request, by a repair given as an argument, and the parameters then hold the repaired matrix.*

**Rule.** `toDensity()` and `factorize()` take no argument, are `const`, never change the
covariance, and return `GaussianError::NotPositiveDefinite`. Their overloads that take a `Repair`
(`g.toDensity(roots::ridge())`), and `upkeep::repairCovariance(repair)` on a filter's `state()`
[upkeep], retry that failure on the repaired matrix and, if it factorizes, store it as the
covariance; otherwise nothing changes. `roots::ridge()` adds to the diagonal the Gershgorin bound on
the most negative eigenvalue plus a relative margin, so a zero matrix still fails. A user's repair
is any type with `repaired(cov)` [user-extension-points]; axes are refused at compile time. A
mixture or a hybrid repairs every component or none. The library's own factorizations of a caller's matrix take
no repair, an innovation covariance's is [innovation-cholesky]'s, and only the covariance form
repairs [gaussian-forms].

**Why.** A repair changes the distribution, so the caller asks for it where it reads, and the
density evaluated afterward is the one the parameters hold. The repairs are the ones a sigma-point
root falls back on [sigma-point-sets], so one concept serves both. Gershgorin's bound needs no
eigensolver. The square-root form is positive semidefinite by construction, and a ridge means
nothing in the information forms.

**Rejected.** A repair inside the plain `toDensity()`: a model error would be ridged at every step
and reported nowhere. A closed list of repairs as a template argument: it took no repair of the
user's own. Axes with fallbacks in place of a repair: they keep no repaired matrix, and axes that
cannot fail would hide a covariance that is not positive definite.

**Evidence.** The ridge and repair tests of `tests/Distributions/gaussian.cpp` and
`gaussian_mixture.cpp`, and the `GAUSSIAN_DENSITY_GIVEN_AXES` negative compile test.

**Where.** `Distributions/Gaussian.h` (`factorize(R const &)`), `Distributions/Mixture.h`,
`Distributions/Roots.h` (`Repair`, `roots::Ridge`).

### sigma-point-axes

*A sigma-point method is a rule spread by axes the caller may choose, and the default axes are the
Cholesky factor, falling back to a pivoted root.*

**Rule.** A sigma-point method is a rule, its unit points and weights, and its axes, the root $S$,
$S S^T = P$, that carries them into the input: the library's in `roots` or a user's own meeting
`Axes` [user-extension-points]. Axes that can fail take a flat list of fallbacks, each a repair or
other axes: `roots::cholesky(roots::ridge(), roots::pivoted())`. A rule meets its axes once, in
`methods::spread(rule, axes)`. Every other root the library takes uses the default,
`cholesky(pivoted())`. In `transform()` and `predict()` a list whose last entry can fail is a
compile error; an update, which returns a `std::expected` already, takes one and reports
`GaussianError::NotPositiveDefinite`.

**Why.** Every root of $P$ is $L Q$ with $Q$ orthogonal, and a rule's moments depend on $Q$ once
$f$ is not integrated exactly, so differing roots made a `KalmanFilter` and a `SqrtKalmanFilter`
disagree. Cholesky is the unique triangular root, analytic in $P$, the cheapest, and the factor the
square-root forms hold. Behind it, `pivoted` gives a semidefinite input an exact root and an
indefinite one the root of its positive semidefinite part, so the default never fails and never
repairs [covariance-repair]. A transform is an integral, which cannot fail on valid inputs, so
axes that can fail there are the caller's explicit choice.

**Rejected.** A pivoted default: its root turns by a large rotation when two pivots swap, moving a
filter's moments for no reason in the data. `symmetric`, the one basis-independent root, at an
eigensolve per root. Nested fallbacks. Reports on transforms and filters: a caller diagnosing one
calls `squareRootWithReport()` on the same covariance with the same axes.

**Where.** `Distributions/Roots.h`, `Models/Methods.h` (`spread()`), `Models/Transform.h`; tests in
`tests/Distributions/roots.cpp`, `tests/Models/spread.cpp`, and `tests/user_axes.h`.

### conditional-evidence

*`conditional()` returns a plain distribution; the evidence of the given values stays in the
conditional's weights, or comes beside a single Gaussian from `conditionalWithEvidence()`.*

**Rule.** A conditional keeps the log-density of the given values, its evidence, wherever its type
has weights to keep it in: a mixture's, a particle set's, and a categorical's conditional leave
their weights unnormalized, so the total weight is the source's times the evidence. A single
Gaussian has no weight, so `conditional<Path>(v)` returns parameters alone, and
`conditionalWithEvidence<Path>(v)` returns a `GaussianConditional`, the same conditional and its
`logEvidence`, in a `std::expected` that fails where $\Sigma_{bb}$ is not positive definite
[innovation-cholesky]. The information forms have no `conditionalWithEvidence()`: their conditional
reads a block of the precision, and the evidence needs the given block's marginal covariance, which
they don't hold.

**Why.** A Rao-Blackwellized update weights each particle or grid point by exactly that number,
which conditioning computes anyway. A second function keeps `conditional()` the same kind of result
in every distribution, and infallible for a single Gaussian, which needs no likelihood.

**Rejected.** `conditional()` returning the pair on the Gaussian alone: generic code over
distributions would branch on the result. Dropping the evidence, as `GaussianParams` once did: the
caller would evaluate the marginal's density again.

**Evidence.** `tests/Distributions/gaussian_subspace.cpp`.

**Where.** `Distributions/Gaussian.h`, `Spaces/Indexing.h` (`internal::conditionMomentsWithEvidence()`).

### random-draws

*A density draws from the caller's generator through the standard distributions, so a seed fixes
the draws within one standard library only.*

**Rule.** Sampling is free functions over densities: `drawSamples()`, `drawSample()`, and their
`WithLogDensity` forms. Parameters don't sample, as they don't evaluate [params-and-density]. The
bit generator is the caller's, and the normal, uniform, gamma, and discrete draws come from
`std::normal_distribution` and its siblings, whose algorithms the standard leaves to the
implementation: one seed gives the same draws under MSVC's library, with Clang on Windows too, and
other draws under libstdc++. Tests compare statistics, never fixed draws. A draw in a scalar that is
not a standard floating-point type fails a `static_assert`. A mixture's draws pick components
independently (`resamplers::Multinomial`), while `ParticleSet::resample()` defaults to the systematic
resampler.

**Why.** Parameters hold values that may be invalid, and the density is the proof that they are
not: a caller makes it once, checks it, and draws, which keeps the factorization out of every draw.
Engines and distributions of the library's own would be a second numerics library, for a
reproducibility across standard libraries that no filter needs. A draw is a sample, whose picks are
independent; a resample replaces a set, which the systematic resampler does with less added
variance.

**Rejected.** `drawSamples()` on parameters, returning `std::expected<SampleSet, GaussianError>`:
it would factorize at each call. A Philox engine and Box–Muller transform of the library's own:
C++26's `std::philox4x32` is the counter-based engine (roadmap step 18b). Eigen's generators:
`Random()` calls `std::rand()`, and `NormalRandomGenerator`, in the unsupported Tensor module, is
built on `internal::` functions and keeps mutable state that every call shares.

**Where.** `Distributions/Sampling.h` and each distribution's header; usage.md tells a user what a
seed fixes.

## Models

### method-per-measurement

*Each measurement chooses its transform method.*

**Rule.** The method (`Linearized`, `Unscented` and the other sigma-point sets, `Iterated`),
made by its factory in `methods` [policy-factories], belongs to
the measurement, not to the filter, so one filter
can take an extended update from one sensor and an unscented update from another. A process model's
method is the filter's, or a fixed-step model's own [fixed-step-models], and each mode's model in a
switching model may have its own [switching-models].

**Why.** How well a linearization serves depends on the sensor against the prior, not on the
filter: a precise range and bearing may need an iterated update where a position fix needs none.

**Rejected.** A method as a parameter of the filter's type, which would make every sensor of a
filter use one method. Aliases named for the textbook filters, such as `ExtendedKalmanFilter` and
`UnscentedKalmanFilter`: the process model, each mode's model in a switching
model, and each measurement choose their methods independently, so no one name describes the
filter. An alias could name only the prediction, the one method the filter owns, and a reader would
take it to apply to the measurements too. The user documentation says which methods make each textbook filter instead.

**Where.** `Models/MeasurementModel.h` (`observe()`), `Models/MeasurementReads.h`, `Models/Methods.h`,
`Models/Transform.h`.

### additive-noise

*Additive noise is explicit, and it may be a mixture.*

**Rule.** A measurement model's noise is `additiveNoise()` ($z = h(x) \boxplus v$) [model-policies],
or an uncertain input of the call that its function reads by tag (noise that passes through $h$)
[sensors], and which it has is part of the measurement's type. For additive noise, $h$ is
transformed once, and each noise component then costs only $S_k = \Sigma_{yy} + R_k$ and a gain.
Additive noise may be a Gaussian mixture, whose components may have nonzero offsets, for outliers
from a separate source (multipath, misassociation). An uncertain input is one Gaussian.
`additiveNoise(R)` of a covariance alone is zero-mean Gaussian noise, unplaced or placed on a block
with `on<Path>(R)` or `onEach<Path>(R)`; it has no space of its own, so a model whose output space
would be its noise's measures in a Linear sized by its function's result [sensors].

**Why.** A two-component outlier model under an unscented transform should cost one pass of sigma
points, not two; without the additive structure visible to the transform, every component needs
its own. The kind in the type lets the update choose its code path at compile time. Additive noise
is nearly always zero-mean, and a mean of zeros written out at every model says nothing.

**Rejected.** Mixture input noise, which would need a transform per component. A kind of noise for
an uncertain input (`inputNoise()`): it is a thing the function reads, like a block of the state.
A scalar variance for every axis, as `continuousNoise()` takes, which reads as a standard deviation
as easily as a variance.

**Where.** `Models/Policies.h` (`internal::placeNoise()`), `Models/MeasurementReads.h`.

### weighted-branches

*An update returns weighted branches.*

**Rule.** The core update is prior plus measurement to a set of branches, each with a normalized
log-weight, a posterior, its innovation, a `mode`, and a `component` (the noise component), plus
the total log-likelihood. A single Gaussian noise is one branch, held in a `std::array` so that
it allocates nothing. The total is read through `logLikelihood()` alone, which is the one branch's
where there is one, and otherwise the evidence the weighting computed, held privately, since
outside the weighting that value means nothing for a single branch (models review). A filter that tracks modes sets `mode` as a label in its result;
`component` stays separate so that an outlier-robust filter can collapse across components while
a Markov outlier model promotes a component to a mode. Both are `Eigen::Index`, the type of a flat
index [discrete-spaces], so a branch's `mode` passes to `conditional(mode)` and
`SwitchingModel::visit(mode, f)` as it is.

**Why.** If the Kalman update returned one Gaussian, mixture noise and the IMM would each need a
second code path in every filter. The branch weights are the per-component probabilities
(inlier or outlier, per mode), worth exposing as diagnostics.

**Rejected.** The word "hypothesis" for a branch, which stays free for the data-association
hypotheses of multiple-hypothesis tracking. A label stored in a mixture's components: a labeled
state is a `HybridGaussian`, and a branch's mode is a flat index into its mode space
[discrete-spaces].

**Where.** `Models/MeasurementUpdate.h`, `Models/Measurement.h`, `Models/Reduction.h`.

### switching-models

*A switching process model is a set of per-mode models plus Π.*

**Rule.** A `SwitchingModel` holds one process model per mode and a transition matrix Π, and is
owned by the filter as a single process model is. All modes share one augmented state space; a
mode that doesn't use a block holds it (for example, an acceleration held at zero with small
noise). Modes whose models differ only in $Q$ reuse one transform of the transition. Each mode's
model may predict by a method of its own (imm-filter.md Q7), and by default every mode predicts by
the same one, the filter's. A model carries its own method as a fixed-step model always does, or
through `withMethod(model, method)`, a wrapper only a switching model takes:
`SwitchingModel{chain, cruise, withMethod(turn, methods::unscented())}`. Fixed-step models may
differ in their methods (imm-filter.md Q14). The filter's `MethodType` and `method()` are the method
of the modes that carry none of their own: `Linearized` by default, and `ModelsMethod`, the empty
tag, when every model is fixed-step and no mode needs one.

**Why.** Mixing Gaussians of different dimensions is messy, and a common space is the standard
approach. A method per model, because how well a linearization serves depends on the model: a
constant-velocity mode is affine and exact under `Linearized`, while a coordinated-turn mode beside
it may need `Unscented`, and one method for both would run the affine mode through $2n + 1$ sigma
points for nothing. The default is one method, because most switching models are variations of one
motion and a user should name a second method only to mean it.

**Rejected.** Resizable spaces for dimensions that depend on the mode. One method for every mode,
the filter's alone, which was the rule until imm-filter.md Q7. `withMethod()` as a process model
that every filter takes, as a fixed-step model without the step: outside a switching model it only
moves the filter's method into the model. The first model's method as the filter's `method()` when
the modes' methods differ, which names one mode's method as if it were all of them.

**Where.** `Models/SwitchingModel.h`; the modes' predictions in `Filters/ImmFilter.h` and
`Filters/RbParticleFilter.h`.

### templated-measurements

*Heterogeneous sensors use templates, not type erasure.*

**Rule.** A filter owns its process model as a type parameter, but measurement models are not part
of its type: a measurement arrives with its data, its model, and its noise, and `update()` is a
function template over it. Different sensors are different measurement types. A user who receives
a mixed stream dispatches with `std::visit` over a variant of their own. The filter never erases a
measurement's type.

**Why.** Systems run a fixed motion model and many sensors whose models and geometry change per
observation. Measurement models stay small and ad hoc, because they are never composed into the
filter's type, and every update compiles to the sensor's own code.

**Rejected.** Registering measurement types in the filter's type, and a type-erased measurement
interface, which would cost a virtual call and an allocation per update and hide the noise kind
from the compiler [additive-noise].

**Where.** `Models/MeasurementModel.h`, `Models/Measurement.h`.

### sensors

*A measurement model predicts the observation from what it reads, and a measurement is the model
observed by a method.*

**Rule.** `measurementModel(h, policies...)` is the model of $z = h(x) \boxplus v$: `h(in, out, x)`
reads the blocks it needs from `x` in the input space that `reads()` defines [model-policies] with
`block<Tag>(in, x)` [tag-addressing], or the whole state without `reads()` [whole-state-models],
and returns the prediction in the output space `out`, generic in the scalar of `x` or with its
Jacobian [returned-jacobians]. The call's inputs are matched to `reads()` by tag, and one the model
does not read is refused. `observe(method, model, z, arguments...)` is the measurement a filter
takes [method-per-measurement], optionally with a covariance form [covariance-update-forms] and a
repair [innovation-cholesky]; `observe(model, z, ...)` is for filters that only weight points. The
residual $z \boxminus \hat z$ is formed in the output space, so a bearing wraps, and a constraint
$g(z, x) = 0$ is a model returning $g$, observed at zero.

**Why.** $h(x)$ is the form every text writes: a hand Jacobian is the textbook $H$, a generic
function captures no observation to cast, and one model serves the filter, the simulation
[printing-and-simulation], and the Cramér-Rao bound [cramer-rao-bound]. Addressing by tag reads a
call's input and a state's block alike, so noise passed through the function is an input, not a
kind of noise. The method is given where the measurement is applied: it is the rule for the prior.

**Rejected.** A measurement that supplies its innovation $z \boxminus h(x)$: each autodiff
innovation cast the observation, each hand Jacobian was $-H$, simulation and the bound needed $h$
again, and an innovation could not live in a space whose values and differences differ. A
hand-written model contract, or a block per parameter of `h` in order, a second place to err.
`observe(method, prior, model, z)` returning the posterior: a measurement is applied to any prior.

**Evidence.** `tests/Models/measurement_model.cpp`, `tests/Models/models.cpp`, `tests/Filters/*`.

**Where.** `Models/MeasurementModel.h`, `Models/MeasurementReads.h`, `Models/Measurement.h`.

### linear-sensors

*A linear measurement is a model given its matrix, or its matrix's columns by block, which reads
blocks whose values add.*

**Rule.** `measurementModel(H, reads(paths...), [outputSpace(space),] noise)` is the model of
$z = H x \boxplus v$, $x$ the blocks read in the order named.
`measurementModel(Columns{onEach<B>(H_B), on<A>(H_A)}, ...)` gives $H$ a matrix per block, placed
by path [product-factors], with no `reads()`. An `onEach` block repeats the measurement
($I \otimes H_B$) and an `on` block is read into every repetition; the space is then computed at
each update [measurement-space-functions].
The function returns $H x$ with $H$ as its Jacobian [returned-jacobians], so every method, form,
and covariance update takes it exactly. Blocks read and the output space have no `plus()` of their
own.

**Why.** A position fix was a lambda returning its argument, which reads as an approximation and
differentiates a matrix; a linear measurement is its $H$. Each landmark's position relative to the
vehicle is linear, but no one matrix fits every count of landmarks.

**Rejected.** A method such as `Exact{}` asserting the function is affine: nothing can check it. A
default method for every model, which would hide an approximation; only a model given its matrix
defaults, to `Linearized`. A separate `linearModel()`, where the matrix already says it is linear.
One container for `Factors` and `Columns`: a product's paths name rows and columns of one space, a
model's only the state's columns of $H$. $H$ on a quaternion's tangent space, where no function of
the value has a constant Jacobian.

**Evidence.** `tests/Models/linear_measurement.cpp`, `tests/allocation.cpp`.

**Where.** `Models/MeasurementModel.h` (`LinearFunction`, `ColumnsFunction`).

### measurement-space-functions

*A measurement model's output space may be a function of the state's space, computed once per update
and passed to the function as `out`.*

**Rule.** `outputSpace()` takes a space, or a function object from the state's space instance to the
output space instance [model-policies]. Every update, likelihood, simulation, and form computes the
space once, at its start, passes it to the function as `out`, and places additive noise given by
path [product-factors] in it. `CountFrom<Path, MeasurementSpace>{}` gives one measurement block per
repetition of a state block, checked at compile time. A linear model whose columns repeat computes
its space the same way [linear-sensors]. A function that returns a vector of run-time size without
`writes()` or `outputSpace()` is refused.

**Why.** A measurement whose size follows the state's resizable blocks had no way to say so: the
caller worked out the sizes again for the space and the noise, and rebuilt the model whenever a
count changed. The function takes the whole state's space so that it can name any block by path,
read or not.

**Rejected.** A function that returns its space with each value: built at every call, and too late
to size the output. A space function given the read blocks' spaces in order: `CountFrom` could not
tell which is its block. A model that maps over the repetitions of one block (`Each<>`): the size
may follow several blocks. Inferring a count from the blocks read [several-resizable-blocks].
A second level of placement (`Rows{on<Fix>(Columns{...}), ...}`) and noise as a function of the
output space: a model built per measurement takes both with an ordinary matrix and noise, and noise
that varies by measurement is built with the measurement anyway.

**Evidence.** `tests/Models/computed_space.cpp`.

**Where.** `Models/MeasurementModel.h` (`CountFrom`), `Models/MeasurementReads.h`
(`internal::ResolvedMeasurement::outputType()`).

### continuous-time-noise

*Process noise is continuous-time, discretized exactly by each model.*

**Rule.** The built-in models take continuous-time parameters and discretize them exactly over the
step: `RandomWalk` takes a white-noise spectral density over its block's differences, `Kinematic`
(two blocks or more, with `ConstantVelocity`, `ConstantAcceleration`) one on its last derivative,
`OrnsteinUhlenbeck` a time constant and a stationary variance, and `Constant` nothing. A model of the
user's own gives `continuousNoise()`, spectral densities by block of its output discretized as
$Q_c \, \Delta t$, or `additiveNoise()`, a fixed discrete $Q$, or neither; never both, and never a
mixture. `OrnsteinUhlenbeck` takes one $\tau$ and $\sigma^2$, by name, for every coordinate of its
block, so coordinates that differ are blocks of their own. `RandomWalk` is a model of its own, which
takes a block whose differences have another size than its values, such as a `UnitQuaternion`; a
`Kinematic` chain has two blocks at least, of one fixed size whose differences are its values'. The models of a composition write every coordinate of the state once, which `predict()`
checks: at compile time where the space is fixed-size, and otherwise in a debug assertion.

**Why.** Continuous parameters keep a model valid when the step varies, which is the usual case with
asynchronous sensors. A filter at a fixed rate shouldn't pay to discretize the same step every time;
[fixed-step-models] caches it. A user's noise is declared, not computed in a member, so that a
composition can place each part's on the blocks it writes without calling anything. A chain adds a
value to a value, which a quaternion cannot take, and a random walk adds only noise on the
differences, which it can. Named parameters cannot swap, as two numbers of one type would.

**Rejected.** A diagonal or a vector of time constants for `OrnsteinUhlenbeck`, which can be added
later without breaking anything; a one-block chain as the random walk.

**Where.** `Models/ProcessModel.h`.

### prediction-inputs

*Prediction takes inputs by the tag of their space, uncertain or sure.*

**Rule.** `predict(method, prior, model, dt, args...)` takes, after the step, the inputs that the
model's `reads()` names and the state lacks, each matched by the tag of its space and given once: an
uncertain input, a Gaussian over a tagged space, independent of the state and the other inputs,
whose uncertainty passes through the dynamics; or a sure one, `sure<Tag>(value)`, such as a control
or a gravity vector, read as it is. The function reads either as a block of its input,
`block<Tag>(in, x)`. An argument the model does not read, an input it reads that the call does not
give, and an untagged input fail to compile. A `linearizedAbout()` reference comes first among the
arguments. `observe()` takes a measurement model's inputs by the same rules.

**Why.** Navigation filters drive the state with inertial measurements rather than modeling the
motion, and the sensors' noise has to go through the same, often nonlinear, mechanization as the
state, which an uncertain input does and an added $Q$ can't. Matching by tag rather than by
position lets each part of a composition read the inputs it names, whatever the others take, and
lets the library refuse an input nothing reads, which a positional pack passed on in silence.

**Where.** `Models/ProcessModel.h`.

### model-policies

*A model is a function with policies given as calls, and a composition's method is given at the call.*

**Rule.** A process model and a measurement model have one shape: a function over Eigen vectors,
the only positional argument, then policies in any order. The function is `f(in, out, x[, dt])`,
`f(in, x[, dt])`, or `f(x[, dt])`, each the full form with leading spaces removed and told apart by
its argument count; a batched function takes its output `Y` last [batched-functions]. Each policy
is a call named for what it declares, given at most once: `reads(items...)`, `writes(paths...)`,
`affineIn(names...)`, `additiveNoise(...)`, `continuousNoise(...)`, `logLikelihood(g)`, and
`outputSpace(x)`. An argument of no policy type is refused with a message naming the policies. The
method is never a policy: it is given at the call, `predict(method, ...)` and
`observe(method, ...)`, since it is the filter's rule for pushing the joint prior through the
model, and a composition is one model. An input of a call is named by the tag of its space, as a
distribution or as `sure<Tag>(value)`.

**Why.** With nothing positional but the function, a policy named by its call reads at the call
site and cannot land in the wrong slot, and a model reads what it names wherever those blocks sit.
The space comes before the vector, as in `block<Tag>(in, x)`, so each short form is the full one
with leading arguments dropped, and the count tells them apart for generic lambdas, since a wrong
count fails overload resolution before a body is instantiated.

**Rejected.** A contract of member types and functions on a hand-written struct, which spelled a
model's shape twice and by position. A `Reads<...>` type the function names, since the function and
the policies sit in one expression. The method as a policy of each part, which made a
composition's parts independent transforms. A mark or a factory per form, or forms told apart by
argument types, which a generic lambda does not declare. A check that a two-argument function's
input and output spaces are equal.

**Where.** `Models/Policies.h`, `Models/Methods.h` (`internal::callForm()`).

### function-scalars

*A function the library calls with a vector may take the vector's scalar as its one template
parameter, which the call supplies.*

**Rule.** A model's function, a `logLikelihood()` or `sample()` function, and a function given to
`transform()` may be written `[]<typename T>(auto const & in, auto const & out, auto const & x)`:
the library calls it with the scalar of `x` (`double`, `float`, or the dual numbers of automatic
differentiation) as its explicit template argument, in each form of [model-policies], and calls a
function without a template parameter list as it is. A function fills its output from
`zeros<T>(out)`, a zero vector in `out`. A function that the user's own code also calls deduces its
scalar instead, `ScalarOf<decltype(x)>`, since nothing supplies `T` at that call, and so does a plain
function, which takes no template argument.

**Why.** A function that is differentiated automatically builds its result in the scalar of its
input, and `using T = ScalarOf<decltype(x)>;` opened most of them. `T` can't be deduced from `x`,
which may be an expression, a block, or a map that keeps its scalar as a member type, and a concept
constrains a parameter without deducing one. The library knows the scalar at every call.

**Rejected.** Deducing `T` from a constrained `x`, which C++ does not do. An argument carrying the
type, which adds a parameter to every function. Mixing `AutoDiffScalar` with `double` operands to
spare the casts, which silently drops the derivatives where a result takes its scalar from a
`double` operand. `vectorIn(out, x)`, which took its scalar from `x` and read as a copy of `x`.

**Evidence.** `tests/Models/whole_state_models.cpp` tells each form with and without the parameter
and checks the scalar the call supplies; `tests/Models/simulation.cpp` and `tests/Models/batched.cpp`
cover `sample()` and `logLikelihood()`.

**Where.** `Models/Methods.h` (`internal::WithScalar`, `internal::callForm()`),
`Models/TransformArgs.h`, `Models/MeasurementReads.h`, `Models/Placement.h`; `Spaces/TagPath.h`
(`zeros()`).

### model-placement

*A model is placed on a block by prefixing its paths, on every repetition by onEach, and models compose side by side.*

**Rule.** `on<Path>(model, renames...)` places a model on the block `Path` names: every item of its
`reads()` that starts in that block, and every block it writes, is prefixed; its function is
untouched. Any other item is an input of the call and keeps its tag, so two placements share it,
unless a rename `on<Name>(path<Tag>)` gives it its own name. `onEach<Path>(model, renames...)`
places one copy of a process model on every repetition of a repeated block, each with its own
noise; an input renamed to a `Repeated` tag has a value per repetition, any other is shared.
`compose(models...)` of measurement models measures the blocks they write side by side: the union
of their reads, their noise on their blocks, and the sum of their log-likelihoods where each has
one. Parts of a composition of process models that state a scalar state the same one; a part that
states none joins any.

**Why.** A model is written once for the block it moves, a vehicle or a sensor, and placed where the
state holds it, so a two-vehicle state is two placements of one model. Sharing an input, gravity
or a common clock, is the common need, so it is the default, and a rename handles each vehicle's
specific force. A `Repeated` tag is the spelling a state already uses for a fleet. A composition is
one model of one state, with one scalar, so a part in another scalar is a slip that would mix
precisions silently.

**Rejected.** Prefixing the inputs too, which makes sharing the special case. Deciding which items
are inputs from the state at the call, which would leave a composition's reads unknown until a
state meets it. A separate spelling for an input every repetition shares. Parts of different
scalars, with the composition in their common type. `Composition::part<I>()`: a part has no tag,
and with no setters it could only assign a whole new part, which composing a new model does too.

**Where.** `Models/Placement.h`, `Models/ProcessModel.h` (`compose()` of process models).

### whole-state-models

*A model given without `reads()` and `writes()` reads and writes the whole state, and is neither
placed nor composed.*

**Rule.** A process model with neither `reads()` nor `writes()`, and a measurement model without
`reads()`, read the whole state of the filter they are given to: the function's input space is the
state's space and its input vector the whole state, and a process model's output space is the
state's too. A process model gives both policies or neither. Its noise is over the whole state,
`continuousNoise(Qc)` of a matrix or `additiveNoise()` of a covariance or a Gaussian;
`processModel(F, noise)` is the matrix model, whose function returns $F x$ with $F$ as its
Jacobian, so that every method predicts it exactly, as `measurementModel(H, noise)` updates exactly [linear-sensors]. Such a model
takes no inputs of the call and no `affineIn()`, and `on()`, `onEach()`, and `compose()` refuse it at
compile time. A batched function of one writes its Jacobian whole, `Y.jacobian(j)` or `Y.jacobian()`.

**Why.** The Kalman filter as a textbook writes it, $F$, $H$, $Q$, and $R$ on `Linear<4>`, is the
first filter most users write, and tags are worth their cost once a state has blocks. The whole
state is the one block every state has, so nothing is inferred from a size. Placement and
composition work through the paths of `reads()` and `writes()`, which such a model has none of, and
two whole-state process models composed would each write every coordinate.

**Rejected.** Inferring `reads()` from the state's blocks; reading `Linear<4>` as a Composite of one
block; a `reads(wholeState)` mark for what is the default; whole-state models in a composition.

**Evidence.** `tests/Models/whole_state_models.cpp`: a constant-velocity filter on `Linear<4>`
equals the tagged one on `Composite<Position, Velocity>`, a fix by function and by matrix equals the
tagged fix, and the matrix models are the textbook Kalman filter.

**Where.** `Models/ResolvedReads.h` (`internal::ReadsWhole`), `Models/ProcessModel.h`
(`internal::LinearStep`), `Models/MeasurementModel.h`, `Models/Batched.h`.

### transform-policies

*`transform()` takes a measurement model's policies before its inputs, matches its inputs by tag,
and takes no process model.*

**Rule.** `transform(method, f, policies..., inputs...)` takes `f` in any of the three forms
[model-policies] and a measurement model's policies, every one before every input: `reads()`,
`writes()` or `outputSpace()`, `affineIn()`, and `additiveNoise()`, one Gaussian. Without `reads()`,
`f` reads its one uncertain input whole. With it, each item is one tag and names an input, a tagged
distribution, `sure<Tag>(v)`, or one of them marked by `withCross()`, matched by tag in any order; an
input no item reads, an item no input gives, and two inputs with one tag are refused. `outputSpace()`
takes an instance or a function of the input space, and a `linearizedAbout()` reference leads the
inputs. `transform(method, model, inputs...)` takes a measurement model for the function and its
policies, so that a model's predicted measurement is `transform()` of it at the blocks it reads.

**Why.** A transform with positional arguments and its output space as an argument was a second
convention beside the models'. With the models' policies, a measurement model is a transform with
its inputs left unbound, and `affineIn()` and `additiveNoise()` work in both. Matched by tag, an
input cannot be in the wrong position, and one that nothing reads is caught.

**Rejected.** Positional inputs beside `reads()`; the output-space argument and
`transform<OutputSpace>()`; a process model, whose step scatters into a state, which is
`predict()`'s. `update(prior, predicted, z)` for the conditioning of a marked input, beside
`update(prior, measurement)`: one name for results of different shapes, so it is
`condition(prior, predicted, z)`, and `update` is left for measurements.

**Evidence.** `tests/Models/transform.cpp`: the predicted measurement of a model and `transform()` of
it agree, with `affineIn()` and `additiveNoise()` as `observe()` takes them.

**Where.** `Models/Transform.h` (`transform()`, `internal::ModelTransform`, `internal::transformBy()`).

### affine-coordinates

*A model declares the inputs it is affine in, and every sigma-point method factors them out.*

**Rule.** `affineIn(names...)` names the inputs a model's function is affine in given the rest,
$y = g(x_n) + H(x_n) \, x_a$. A sigma-point method spreads its points over $x_n$ alone; at each
point $x_a$ given $x_n$ is Gaussian and goes through $f$ exactly, adding $H P_{a|n} H^T$, with $H$
from the returned Jacobian or automatic differentiation in the affine coordinates, one call per
point. A coordinate is factored out where it is a linear block of the state or an uncertain input;
a `sure` input is skipped, and a block that is not linear is treated as nonlinear. A
composition's affine set is the names every part reading them declares. `Linearized`, `Iterated`,
the square-root and information steps, the update given a block, and the Rao-Blackwellized
prediction ignore the declaration.

**Why.** The declaration sits with the function that knows it, and the method stays a choice made
at the call [model-policies]: a bilinear model and noise scaled by the state get exact moments from
every sigma-point method, and a model whose only nonlinear part is a heading spreads its points
over the heading alone. Nothing is inferred, and a coordinate left out still gets sigma points,
which costs evaluations but is never wrong.

**Rejected.** A method naming state paths with a Jacobian of its own, or `affineInput()` marking an
input at the call: both put the model's structure into the method, which a composition's parts
cannot share. An `affineIn<Paths...>(f)` wrapper around the function. Naming the nonlinear blocks
instead, which would treat anything forgotten as affine. A Jacobian of the affine columns alone: a
second Jacobian shape `Linearized` cannot use. Evaluating $f$ along the columns of $P_{a|n}$'s
root: $(2n + 1)(a + 1)$ calls, more than the full transform's $2(n + a) + 1$. A debug-build check
of affinity by second differences: calls and a tolerance beyond the method's work. An opt-out: a
model declared affine in nothing gets the full transform.

**Where.** `Models/Policies.h` (`affineIn()`), `Models/ResolvedReads.h` (the coordinates),
`Models/Transform.h` (the factored transform), `Models/ProcessModel.h`, `Models/MeasurementReads.h`.

### covariance-update-forms

*The measurement chooses the form of the covariance update, and Joseph rests on the statistical
linearization.*

**Rule.** An update's covariance takes one of three forms: `updates::Standard`
($P - K \Sigma_{xy}^T$, the default), `updates::Symmetric` ($P - K S K^T$, made exactly symmetric),
or `updates::Joseph` ($(I - KH) P (I - KH)^T + K R K^T$), each made by its factory
[policy-factories]. A measurement chooses by an argument of `observe()` [sensors], and every
Kalman-family update honors it. The Joseph form's $H$ and $R$ come from the public
`statisticalLinearization()` of a transform's result, read from the parts the result keeps apart
[linearization-forms], with each noise component's $R_k$ added.

**Why.** How well an update is conditioned depends on the sensor against the prior: a precise sensor
needs the Joseph form, and others don't need its cost. With a prior of variance $10^6$ and a fix of
variance $10^{-12}$, the standard and symmetric forms return a posterior variance of exactly zero
and Joseph returns $10^{-12}$. Additive noise is kept apart because recovering $R$ from
$\Sigma_{yy}$ by subtraction cancels the digits that make a precise measurement hard. The standard
form is the default because it is the cheapest and is what hand-written filters do [benchmarks].
`statisticalLinearization()` is public because the information forms [gaussian-forms], the iterated
filters [iterated-update], [sigma-point-sets], and splitting [splitting-hooks] use it too.

**Rejected.** A template parameter on the filters: it adds a parameter to five filter types and
still doesn't let two sensors differ. The cross-covariance form of Joseph,
$P - K\Sigma_{xy}^T - \Sigma_{xy}K^T + KSK^T$: it subtracts rather than adds two positive
semidefinite terms, losing the guarantee that motivates the form. Square-root forms as update forms:
they change what the filter stores [gaussian-forms].

**Where.** `Models/MeasurementModel.h`, `Models/Condition.h`.

### consider-states

*A measurement can consider blocks of the state rather than estimate them, by an argument of
`observe()`, and the covariance form alone carries it out.*

**Rule.** `observe(method, model, z, consider(path<Bias>, ...))` updates the state by the
Schmidt-Kalman update: the gain's rows at the blocks named are zeroed, so their mean and their
covariance block $P_{cc}$ stay the prior's, and every other entry of the posterior, the
cross-covariance $P_{xc}$ included, is the full update's. The paths resolve against the state's
space, not the model's `reads()`, so a block the model never reads is considered through its
covariance with the blocks it does. The innovation is the full one, so the NIS, the
log-likelihood, a gate, and the NIS monitor see the honest $S$. `updates::Standard` and
`updates::Symmetric`, right only for the optimal gain, form the full update and write back the
prior's $P_{cc}$; `updates::Joseph` takes the zeroed gain whole, so the result stays a sum of
positive semidefinite terms. The step has no component in $c$, so the reset [covariance-reset]
leaves $c$'s block alone. Every update that runs `kalmanConditionInPlace()` takes it: the Kalman
filter, mixture noise, the Gaussian mixture and IMM filters, and `Iterated`, whose iterates keep
$c$ at the prior. The square-root and information forms refuse it at compile time, as do the
filters that weight points and the updates given a block. The smoother, whose history never sees
a measurement, smooths $c$ as if estimated, assuming an optimal filter.

**Why.** A sensor's bias known only statistically makes the filter overconfident when dropped and
absorbs errors it doesn't cause when estimated; the consider update keeps its effect in $P_{xc}$
without moving it (see `Consider`). An argument of `observe()` sits beside the covariance
form and the repair, which are also per-measurement choices [method-per-measurement], so one sensor
can consider a block another calibrates, and the filter and recursion types don't change.

**Rejected.** A wrapper over the model, `consider<Paths...>(model)`, which would make the model
carry a choice of the update. A decoupled filter, which drops $P_{xc}$ and is not consistent. A
consider update in the square-root and information forms, since keeping
$P_{cc}$ doesn't come out of a triangularization.

### linearization-forms

*A transform's result keeps its linearization in the form its method computes, with the parts of its
covariance apart.*

**Rule.** A transform with an input marked by `withCross()` returns an `OutputWithCross<InputSpace,
OutputSpace, Form, Scalar>`, whose `Form` the method fixes: `JacobianForm` from `methods::linearized()`
and a Jacobian returned with the value, holding $H$ and $\Sigma_{xy} = P H^T$, and `RootForm` from every
sigma-point method, factoring a model's affine coordinates or not [affine-coordinates], holding the root $S$ that spread the points and
$\Psi = S^{-1} \Sigma_{xy}$. Both hold the output covariance's parts apart, each a product of quantities
of its own size: $H P H^T$, what the marked input explains; $N$, what the other uncertain inputs
explain, split from the joint $J$ or $\Psi$ by input; and $\Omega$, what neither explains, the
covariance of the fit's residuals at the points. The output covariance is their sum. A consumer reads
what it needs and derives the rest as `OutputWithCross`'s table says; no update factorizes the prior
again or subtracts to get back a part the transform computed.

**Why.** $\Sigma_{yy}$ alone makes every consumer that needs more than the moments take it apart again
by a solve against $P$ and a subtraction, which loses about $\epsilon \lVert H P H^T \rVert$: for a fix
of variance $10^{-12}$ against a prior of $10^6$, everything. The residuals' $\Omega$ errs by about
$\epsilon^2$ of that, and keeping $N$ apart treats noise passed as an input as additive noise is.

**Rejected.** A form holding $\Sigma_{xy}$ alone, today's cost with $\Omega$ by subtraction. Accessors
that compute on first use and cache, when the method and the consumer are known at compile time. A
form the caller chooses, which only converts. An `additiveNoise()` argument to the low-level
`update()`, which passes $R$ twice and still subtracts input noise. A solve by the axes that took the
root: the result keeps the root, not the axes, and only a root other than a Cholesky factor gains.

**Evidence.** `tests/Models/linearization_forms.cpp`, `tests/Filters/precise_input_noise.cpp`.

**Where.** `Models/Linearization.h`, `Models/Condition.h`, `Models/Transform.h`,
`Models/Measurement.h`.

### fixed-step-models

*A fixed time step is a model, which caches what the step fixes, bound to the state's space by the
filter given it.*

**Rule.** `FixedStepModel{model, space, dt, [method]}` takes a process model over one step and
computes its $Q$ and $Q$'s factor once; under `Linearized`, for a model that reads only the state,
is `affineIn()` everything it reads, and returns its Jacobian, it also caches $F$ and its inverse. A
filter that owns one has `predict(inputs...)` and `predict(Steps{n}, inputs...)` [time-steps] in
place of `predict(dt, inputs...)`, never both, and refuses a leading number or `TimeStep`, which
would be read as an input. It takes its method from the model, and another method doesn't compile. A
`SwitchingModel` is fixed-step when all its models are, with one shared step. `fixedStep(model, dt,
[method])` names no space: each filter's deduction guide binds it to the state's space and scalar,
and the free `predict()` and `predictJoint()` to the prior's.

**Why.** The cache belongs to the pair of model and step, not to a filter, so every filter uses it
without a time member of its own, sized by the space the prior already names. A per-step prediction
rebuilds $Q$ with `pow` and `exp` per entry and gets $F$ again. The method goes in the model because
it decides what can be cached, and an unused $n \times n$ matrix costs memory on a microcontroller.
The scalar is the state's, as the space is, so neither is named again.

**Rejected.** A model that remembers the last step: mutable state in a `const` model. Deducing the
scalar from the step's type, which a `std::chrono` duration lacks, or from the model's `Scalar`,
which is its parameters' and which a model of `processModel()` lacks. A second filter constructor
constrained on the model's kind: it made Clang's deduction through the aliases ambiguous.
`fixedStep<State>(model, dt)`: it names the space twice. A filter constructor that deduces the
model: it outranks the binding guide, so `ParticleFilter` and `RbPointMassFilter` take
`std::type_identity_t<Model_>`.

**Where.** `Models/ProcessModel.h` (`UnboundFixedStep`, `internal::BoundStep`), the filters'
deduction guides.

### innovation-statistics

*An update's statistics are derived from its innovation, when asked for.*

**Rule.** An update branch carries its `Innovation`, the residual and its covariance $S$. The NIS
and the log-likelihood are member functions, computed when first asked for and kept. An update
computes them itself only where it needs them (to weight the branches of mixture noise, or when a
filter's policy reads them), and then from the factorization of $S$ that the gain already made. A
filter's `update()` returns a `FilterUpdate` whose branches hold innovations and weights but not
posteriors, since the posterior is what went into the state. Under mixture noise it is a
`ReusableFilterUpdate`, which keeps its last fill as scratch, so that refilling allocates nothing
[allocation].

**Why.** Computing the statistics eagerly cost an ordinary extended update a log-determinant and a
triangular solve that most callers never read, about 5% of an update at the benchmark's largest
size [benchmarks], and the returned posterior was a second copy of the state.

**Where.** `Models/MeasurementUpdate.h`, `Models/Measurement.h`.

### returned-jacobians

*A function that knows its Jacobian returns it with its value, and nothing differentiates by fixed
steps.*

**Rule.** A function that knows its Jacobian returns `withJacobian(value, jacobian)`: a model's
function one matrix over its whole input, rows in `out`'s differences and columns in `in`'s; a
function given to `transform()` one matrix for one uncertain input or a `std::tuple` of blocks, one
per uncertain input. It returns all of its Jacobian or none of it. `Linearized` calls it once, at
the means or about a `linearizedAbout()` point. A composition whose parts all return theirs returns
the whole's and is never differentiated. Any other function is differentiated automatically and must
be generic in its scalar; the built-in models return theirs.

**Why.** A value and its Jacobian usually share their work, as a range and its unit direction share
the difference to the anchor, and at the benchmark's small sizes [benchmarks] computing it twice was
the largest cost left in the user's own code: differentiating the attitude benchmark's step took
4,317 instructions, and returning its Jacobian 1,073, against the hand code's 1,000. One function
keeps the value and the Jacobian from drifting apart and receives exactly the value's arguments, and
its return type tells the method at compile time whether there is a Jacobian. All or none, because a
block left out would need a second evaluation with dual numbers on every call.

**Rejected.** `Linearized{jacobian}`, a separate Jacobian function: it received the joint vector
rather than the blocks a measurement reads. The `Affine` method, which differentiated by a unit step
in each coordinate: exact only for an affine function, and on a space with its own `plus()` a step
of one radian is not small. `jacobian()` and `propagateWithJacobian()` members beside `propagate()`,
which spell the step twice. Blocks returned by path, `withJacobian(y, on<Position>(J))`, with the
rest differentiated: a row path and a column path read as one nested path, and a partial Jacobian
costs the second evaluation.

**Where.** `Models/Methods.h`, `Models/Transform.h`, `Models/ProcessModel.h`,
`Models/MeasurementReads.h`.

### emissions

*A discrete measurement is an emission model, and a modulated arrival is observed by the
prediction.*

**Rule.** An emission model writes $\log p(z \mid v)$ for every value $v$ into a buffer sized by the
cardinality, and `update(prior, emission)` on `CategoricalParams` returns the posterior and the
log-evidence, or a `WeightError` when no value can explain the measurement, which
`HiddenMarkovFilter` reports as no density [update-outcomes]. The filter owns a `MarkovChain`, with
`predict()`, or a `ContinuousTimeMarkovChain`, with `predict(dt)`, never both. A Markov-modulated
Poisson arrival is observed by the prediction, `predict(dt, ArrivalInterval{rates})`, through
$p\, e^{(Q - \Lambda) \Delta t} \Lambda$ (Fischer and Meier-Hellstern 1993). Smoothing [smoothing]
reads the filtered and predicted distributions and a matrix $K$ per step, and
`mostProbableSequence()` is Viterbi over the same record.

**Why.** An emission is the discrete form of a likelihood, so it needs no innovation, noise, or
method, only a number per value. A count or an inter-arrival time depends on the chain's path inside
the interval, not only on the value at its end, and only the sub-stochastic step accounts for that;
putting it in the prediction keeps an emission a function of one value. A joint that carries $K$
rather than the chain serves all three kinds of step, including the arrival's, whose $K$ is no
chain's. A smoother over filtered and predicted distributions, rather than the forward and backward
variables of Rabiner (1989), keeps no likelihoods and is the same backward pass as the Gaussian
smoothers'.

**Rejected.** Poisson emissions after an exact prediction as the modulated filter: they hold the
value fixed over the interval, an error of first order in its length, which the tests measure.
Baum-Welch re-estimation here: it is model identification [analysis-project].

**Where.** `Models/Emission.h`, `Distributions/ContinuousTimeMarkovChain.h`,
`Filters/HiddenMarkovFilter.h`.

### iterated-update

*The iterated update is a method that wraps the method it relinearizes, and iterates in the prior's
chart.*

**Rule.** `iterated(linearized())`, an `Iterated<Linearized>`, is the iterated extended Kalman
filter (Bell and Cathey 1993), a method a measurement takes like any other [method-per-measurement]. Each iteration linearizes at the
iterate $x_i = \mu \boxplus \delta_i$ and takes the Kalman update from the unchanged prior under
that linearization, Gauss-Newton on the maximum a posteriori problem in the prior's own chart; the
update under the last linearization is the result. It stops when a step moves no coordinate by more
than its tolerance of its prior standard deviation, or at its iteration cap, both run-time values
given to `iterated()` [policy-factories], and every branch reports `iterations` and `converged`. Every Gaussian form takes it [gaussian-forms]. A transform, a
prediction, and the Rao-Blackwellized given-block updates refuse it at compile time; it refuses a
reference [linearization-reference], and it ignores a model's affineIn() [affine-coordinates].

**Why.** What it changes is how an innovation becomes a linearization, which
[method-per-measurement] puts with the measurement, so one filter can iterate a precise range and
bearing and take a plain update from a position fix, and every filter over a Gaussian gets it
unchanged. The prior's chart keeps the prior's residual and covariance as they are, at the cost of a
reset Jacobian per iteration, nothing on a space whose reset is the identity. A stopping rule scaled
by the prior serves meters and radians with one tolerance. A factored iteration is refused because
the combination is subtler than it looks: posterior linearization over the nonlinear coordinates,
with the affine ones conditioned exactly at each point, has no published analysis
[published-algorithms-only].

**Rejected.** A filter or a reduction policy: a policy acts on branches after they are made, too
late to change where they were linearized. A parameter of `Linearized`, which would not give one
spelling for the iterated form of every method [sigma-point-sets]. The chart at each iterate
(Hertzberg et al. 2013), which would carry the prior's residual and covariance into each chart.

**Where.** `Models/Methods.h`, `Models/Transform.h`, `Models/IteratedUpdate.h`.

### linearization-reference

*A step may be linearized about a point or a distribution the caller gives, in its input's chart.*

**Rule.** `linearizedAbout(reference)` marks a point, copied, or a distribution, borrowed, as where a
step linearizes in place of its input's own mean and spread. A transform takes it right before the
input it is for, a prediction first among its inputs, and an update and a `KalmanFilter`'s update
after the measurement; every Gaussian form takes it on both steps [gaussian-forms]. Linearized takes the
Jacobian at the reference's point, a sigma-point method the statistical linear regression over its
distribution (García-Fernández et al. 2015); either is carried to the input's chart by the reset
Jacobian, its value moved back to the input's mean as an Iterated iterate's is, and applied to the
input's own moments. About the input itself each step is the step without one. A sigma-point
method refuses a point, and an Iterated method any reference, at compile time.

**Why.** One operation serves three uses: a shadow tracker linearized about the main tracker's
estimate, the linearized Kalman filter of Gelb (1974) about a nominal supplied each step; an
iterated update's next iteration; and an iterated smoother's relinearization about the smoothed
estimates. Iteration without a measurement is then a loop over it, and
`Iterated` stays a measurement's method, the one place it has a posterior to converge to. An
argument, so that a step that does not take one fails to compile rather than ignoring it; first
among a prediction's inputs, which the filters pass on unchanged, so `predict()` needs no overload.
The prior's chart, for [iterated-update]'s reasons.

**Rejected.** `Iterated` in `transform()`, which has nothing to converge to: it would be one
linearization or need a second argument, which is this one. A reference carried by the
measurement, which a step that does not linearize, such as a particle filter's, would ignore
silently. A reference that is always a distribution, which would make a Linearized caller invent a
covariance it never reads. An Iterated update started at the reference, until a problem needs it.

**Where.** `Models/TransformArgs.h`, `Models/Transform.h`, `Models/ProcessModel.h`, `Models/Measurement.h`,
`Models/FormSteps.h`, `Filters/KalmanFilter.h`; `tests/Models/linearized_about.cpp`.

### sigma-point-sets

*Sigma-point sets are methods beside Unscented, and an iterated one linearizes over the posterior.*

**Rule.** `Cubature` (Arasaratnam and Haykin 2009), `SphericalSimplex<CenterWeight>` (Julier 2003),
and `GaussHermite<Order>` (Ito and Xiong 2000) are methods beside `Unscented`, each a set of
weighted points on the axes of [sigma-point-axes], accepted wherever a sigma-point method is, as is
any type meeting `PointSet` [user-extension-points]. A type meeting `UnscentedRule` shares
`Unscented`'s code by giving $\alpha$, $\beta$, and $\kappa$ for each number $L$ of coordinates the
points spread over, as `FourthMomentUnscented`'s $\kappa = 3 - L$ (Julier and Uhlmann 2004) does.
`Iterated` over one is the iterated posterior linearization filter (García-Fernández, Svensson,
Morelande, and Särkkä 2015, algorithm 1), stopping by the step in the mean [iterated-update]. A
rule's numbers are compile-time template arguments; `Iterated`'s tolerance and iteration cap, which
set how much work is done, are run-time values [policy-factories].

**Why.** Sibling methods, because the cubature rule's authors distinguish it from the unscented
transform, which it equals only at particular parameters. Posterior linearization, because it is
derived from a criterion and reduces to the iterated extended Kalman filter over `Linearized`, so
one wrapper means one thing. Parameters by $L$, because $L$ is each transform's joint input, which a
parameter chosen by hand for the state misses in every update that reads other blocks or noise.
Compile-time numbers, because a `float` filter holds and computes no `double` [embedded-builds], and
a rule is configuration fixed when the filter is written.

**Rejected.** A point-set parameter, `Unscented<PointSet>`, or a rule parameter, `Unscented<Rule>`.
The iterated sigma-point filters of Sibley, Sukhatme, and Matthies (2006) and Zhan and Wan (2007):
they mix a linearization over the prior with an analytical one, or correct repeatedly under ad hoc
conditions. The paper's stopping rule by the divergence between successive posteriors: one rule
serves every method. A damped iteration (Raitoharju et al. 2018), until a problem oscillates.

**Where.** `Models/Methods.h`, `Models/Transform.h`; `tests/user_point_set.h`,
`tests/user_unscented_rule.h`.

### cramer-rao-bound

*The posterior Cramér-Rao bound is an information recursion fed by Jacobians at the true states.*

**Rule.** `PosteriorCramerRaoBound` holds the Fisher information $J_k$, the inverse of the lower
bound on any estimator's mean squared error, starting from $J_0 = P_0^{-1}$. Its `predict()` and
`update()` implement Proposition 1 of Tichavský, Muravchik, and Nehorai (1998) for additive Gaussian
noise, with each expectation estimated by the mean over true states the caller draws with
`simulate()` and passes as the columns of a `SampleSet`. The Jacobians are the library's, through
`Linearized` at each truth, so a space with its own `plus()` gets the bound in its tangent space
(Chahbazian et al. 2024, Proposition 2). A step whose matrices don't factorize returns
`std::unexpected`.

**Why.** The bound is a pure function of the library's types and allocates nothing beyond its
count-sized averaging, so it belongs here beside NEES, which it complements: NEES says whether a
filter's covariance is honest, the bound how far its error is from the best possible. A recursion on
$J$ rather than its inverse, because the paper's is, $J$ starts at zero for an unknown state, and the
prediction needs $J$. The truths are the caller's because a simulation interleaves them with
measurements and filters, and the average over truths must be formed inside the recursion.

**Rejected.** A Monte Carlo runner or report here, which belong to the analysis project
[analysis-project]. The paper's Jacobian at the pair of states, where the library takes it at
$f(x_k)$: the same on `Linear`, and agreeing to first order elsewhere, the order of the paper's own
truncation.

**Where.** `Models/CramerRao.h`. A singular $Q$ (the paper's Section IV) is roadmap step 17g.

### innovation-cholesky

*An innovation covariance must be positive definite, and an update that finds one that is not
returns an error unless its measurement repairs it.*

**Rule.** Every update factors its innovation covariance $S$ by Cholesky, takes the gain, NIS, and
log-determinant from that factor, and returns `GaussianError::NotPositiveDefinite` when it fails,
leaving the state as [update-outcomes] says. A factorization of the prior (a conditional's
$\Sigma_{bb}$, the smoother's gain) keeps the pivoted LDLT, since a block known exactly makes it
singular and $\Sigma_{ab}$ is zero where its pivot is; a conditional that also needs the density of
its given values [conditional-evidence] factors $\Sigma_{bb}$ by Cholesky and returns the same
error. A measurement may instead repair $S$ by a `Repair` type given to `observe()` [sensors], such
as the ridge [covariance-repair]: a failed factorization is retried once on the repaired $S$, which
the returned innovation holds. Only the covariance form repairs [gaussian-forms].

**Why.** A pivoted LDLT of $S$ hid a singular $S$: it divided by pivots that were only rounding, and
gave a log-likelihood of $+\infty$ for a zero pivot and NaN for a negative one. An $S$ singular in
exact arithmetic needs a noiseless measurement of something the prior knows exactly, which a small
noise, a constrained space, or a projection avoids; one from valid inputs that is not positive
definite is a failure to report. A filter of several components fails the whole step, so no failure
reads as a gate's rejection. The caller cannot reach $S$, so the repair is the measurement's.

**Rejected.** LDLT of $S$ as an option: Cholesky is no slower, and LDLT needs a degenerate
Gaussian's likelihood to be right. Factoring $S$ by axes with fallbacks, `cholesky(ridge())`: their
root keeps no repaired matrix for the gain, posterior, and innovation to share. A repair held by
value: a type is a trait every wrapper forwards. Repairing a caller's own matrices: the caller can,
and a failure there is a mistake in the model. A degenerate Gaussian's density on its support:
membership needs a tolerance, and densities of different rank can't be compared.

**Where.** `Models/Measurement.h`, `Models/Linearization.h`, `Models/FormSteps.h`,
`Filters/RbUpdate.h`, `Spaces/Indexing.h`.

### likelihood-measurements

*A measurement may be known only by its likelihood, which the filters that weight points take, and a
measurement's own likelihood wins over its noise.*

**Rule.** A measurement known only by its likelihood is `measurementModel(reads(...),
logLikelihood(g))`, with `g(in, x, z)` the log-likelihood of `z` at the state `x` [model-policies],
and no function, noise, or method; `simulate()` needs it to carry `sample(g)`, a draw of the
observation at `x`. A model with a function draws through its function and noise, refuses
`sample(g)`, and may carry `logLikelihood(g)`. The filters that weight points take either kind, and
a model's own `logLikelihood(g)` weights them whatever its noise says; the Kalman-family filters
require a function and a method. A likelihood in a Rao-Blackwellized filter reads only the gridded
or sampled block. A model with neither additive noise nor `logLikelihood(g)` has no likelihood at a point, and
where its function's value or Jacobian is not finite the likelihood is NaN.

**Why.** An error distribution that is neither a Gaussian nor a mixture, such as quantization
convolved with Gaussian noise, enters a filter only through its likelihood, and a grid weights its
cells exactly for any likelihood (Bucy and Senne 1971; Bergman 1999); a placeholder noise and method
made users invent a Gaussian nothing used. A likelihood of the pinned block alone leaves the rest's
conditional exact; one that reads the rest has no exact update. The likelihood wins in every filter,
so one rule holds. A draw in the model keeps a simulation from drifting from the likelihood the
filter weights by. Noise through the function gives a likelihood only where the model is affine in
it, so a model needs additive noise or its own likelihood. A NaN says the model is undefined at the
point; $-\infty$ would take a grid cell's weight for good.

**Rejected.** A `LikelihoodModel` concept beside the measurement model's: one shape of model, whose
policies say what it has, serves both. Falling back to the Kalman update in a Rao-Blackwellized
filter when a measurement with noise and `logLikelihood(g)` reads the rest: the same measurement
would be weighted two ways in two filters.

**Where.** `Models/MeasurementModel.h`, `Models/Likelihood.h`, `Filters/RbUpdate.h`.

## Filters

### reduction-policies

*Filters reduce branches with a policy.*

**Rule.** A filter whose representation is poorer than an update's result reduces the branches
with a policy: `Collapse` (moment-match them into one Gaussian, soft outlier rejection), `Select`
(keep the most probable, hard rejection), `Keep` (retain every branch, for a mixture filter), and
`Gate`, which passes those within a chi-squared threshold to another, each made by its factory in
`reductions` [policy-factories]. A policy is given the branches, of an update or of one mode, as an
rvalue, and may move their posteriors out. It returns the one component it keeps in a
`std::optional`, standing for every branch it was given, or adds what it keeps to the filter's
output by `out.add(logWeight, component)`, `logWeight` being the log-probability among the
branches of those the component stands for; keeping none rejects. A filter of several components,
modes, or points weights each kept one by its prior weight, its likelihood, and that log-weight;
the evidence stays over every branch. `Gate` hands its inner policy the passing branches through
their indices, weights unchanged. `ReductionPolicy<P, Update, State>` is the contract for an
output of one component, `MixtureReductionPolicy` for a mixture filter's of several; `update()`
requires its filter's, so `Keep` in a Kalman filter fails at the call, and a policy of the user's
own meets it by its interface [user-extension-points].

**Why.** One update returns weighted branches [weighted-branches], and each filter keeps what its
state can hold; a policy object lets every filter take the same policies. Adding to the output
moves each kept posterior once, into the state, building nothing per component, and its weight
makes a dropped branch impossible; the evidence describes the model before any policy.

**Rejected.** Grouping by mode labels inside the policy: the IMM updates each mode apart. A
returned mixture, built and copied per component; `logPassingWeight()` and `forEachKept()`, hooks
beside the result; a gate that gathered its survivors into a new update, allocating and copying.

**Where.** `Models/Reduction.h`, and the updates of `Filters/KalmanFilter.h`, `MixtureFilter.h`,
`ImmFilter.h`, and `RbUpdate.h`.

### composed-features

*Optional features compose with the filter instead of mixing into it.*

**Rule.** A feature that intervenes in an update is a reduction policy: `reductions::gate(probability, inner)`
or `reductions::gate(NisThreshold{t}, inner)` drops branches whose NIS exceeds the chi-squared quantile for the
innovation's degrees of freedom, hands the survivors to the inner policy, and rejects the
measurement when none survive. A feature that only observes lives outside the filter and reads the
`FilterUpdate` that `update()` returns, as `NisMonitor` and the smoothing histories [smoothing] do.
A gate and mixture noise compose: the gate tests each branch against its own innovation covariance
with no outlier model, while mixture noise compares branches with each other.

**Why.** Policies and outside observers cost nothing when unused and need no extension mechanism.
Policies take only the update, not the prior, since returning nothing already means "keep the
prior".

**Rejected.** A mixin or plugin layer, which would fix the hook points and their order before the
features are known; a run-time one is the type erasure [templated-measurements] rules out. A gate
that caches its quantile: in the gate, which is usually built in the call, it saves nothing, and in a
process-wide table it is shared mutable state in a header-only library; a threshold computed once is
a `NisThreshold`. For a filter of several components, a monitor of the NIS of a branch drawn by its
posterior probability, which stays chi-squared: the caller tests the dominant branch's NIS.

**Where.** `Models/Reduction.h`, `Models/Consistency.h`.

### mixture-filter

*The Gaussian mixture filter reduces per component and reweights by its likelihood.*

**Rule.** `MixtureFilter` updates each component, hands its branches to the policy, and gives each
component the policy keeps the weight $w_i \, p(z \mid i)$ times its share among the branches
[reduction-policies]; the weights are then normalized. With `Keep`
and affine models it carries the exact posterior. A component the policy rejects is dropped; if
every component is rejected, the measurement is rejected and the prior kept. Growth from `Keep` is
controlled by the caller between steps [upkeep], not by the filter.

**Why.** Reducing per component keeps the policies usable unchanged: `Collapse` and `Select` keep
the mixture's size, and a gate tests each component against its own innovation covariance. The
right mixture reduction (truncation, merging by a cost, refitting) depends on the problem, so it
stays outside the filter [composed-features].

**Rejected.** Keeping a gated-out component with its prior, which would leave a component the
measurement contradicts with a weight it can't be given consistently.

**Where.** `Filters/MixtureFilter.h`.

### imm

*IMM keeps one Gaussian per mode in a hybrid, and switches models and measurements by mode axis.*

**Rule.** The IMM's state is a `Hybrid`: mode weights over a discrete mode space and one component
per mode. Its `SwitchingModel` holds a model per value of a mode axis, or one shared
(`switchingModel<Maneuver>(chain, cruise, turn)`), and its chain moves once per prediction.
`predict()` mixes each mode's Gaussian through the chain and predicts it by its mode's model;
`update()` updates each mode, reduces with a policy that keeps one Gaussian (`Keep` doesn't
compile), and reweights by $p(z \mid m)$, a rejected mode keeping its Gaussian at probability zero.
Noise `additiveNoise(mixture).byMode<Axis>()` is the Markov outlier model: in any filter whose mode
space has the axis, each mode updates by its value's component, weighted by the chain alone; with
no mode space the mixture is memoryless; a mode space without the axis fails to compile.
`switchingMeasurement<Sensor>(nominal, biased)` updates each mode by its value's measurement, all of
one update type and none with noise by mode axis. One measurement's update takes a block given as
`update(prior, measurement, given)` does [rb-point-mass], [rb-particle-filter].

**Why.** Every filter that tracks modes keeps one Gaussian per mode, and the mode space fixes the
set, so a rejected mode keeps its place for the chain to restore. Selecting by mode axis keeps a
two-model maneuver set at two models when a two-state sensor joins the mode, and lets the mode change
a measurement's function, not only its noise, without index arithmetic. Naming the axis in the noise
chooses once, where the sensor is modeled. Without a mode space it can mean only the memoryless
model, so one measurement serves the robust filter and the IMM; a mode space lacking it is a mistake.

**Rejected.** A labeled `GaussianMixtureParams`, which would allow several components per mode.
`update<Path>()`, naming the axis at each call: a forgotten axis was left to the chain. An error
without a mode space, which splits one sensor model in two. The mode as an item of `reads()`, whose
items are continuous blocks.

**Where.** `Filters/ImmFilter.h`, `Models/SwitchingModel.h`, `Models/Policies.h`, `Distributions/Hybrid.h`.

### rb-point-mass

*The Rao-Blackwellized point-mass filter fixes one block at points and never moves them.*

**Rule.** `RbPointMassFilter`, the Rao-Blackwellized point-mass filter (Šmídl and Gašperin 2013;
Duník et al. 2019) without its prediction between cells, represents
$p(x) = \sum_c P(c) \, \delta(x_p - p_c) \, \mathcal{N}(x_r; \mu_c, \Sigma_c)$: a block named by a
tag path sits at fixed points, and the rest is a Kalman filter per point. The process model covers
the rest only. A cell may have a covariance about its center (no source), an independent uncertain
input to each update. With `SharedCovariance` (the default), one transform at the most probable cell
serves every cell, exact when the rest is affine with the same Jacobian in every cell;
`PerPointCovariance` makes each cell its own filter. Under a `SwitchingModel` each cell holds a
`HybridGaussian` and is an IMM filter given its center (no source; a composition
[published-algorithms-only]), with `PerPointCovariance`, the cells run in order. `PointMassFilter`,
the Bayesian grid filter, is the same filter with nothing left Gaussian: an update multiplies each
cell's probability by its likelihood, and `predict()` changes nothing.

**Why.** The use is a coarse position with a linear remainder, over thousands to tens of thousands
of cells. With the points fixed, the cell weights are exact for any measurement nonlinear in the
fixed block, and the shared covariance makes the linear case cost little more than evaluating the
measurement once per cell. Motion belongs in the Gaussian part, or in the particles
[rb-particle-filter]. A grid filter is then exactly this filter with an empty rest, so a second
filter would only duplicate it.

**Rejected.** Moving probability between cells by a transition kernel, as the published point-mass
filters predict (roadmap, Deferred): it costs a convolution over the whole grid every step, smears
the density by the cell spacing, and revives the cells that pruning removed. Refining only the cells
near the probability mass, which needs a covariance per cell [rao-blackwellized-core]. Correlating
successive innovations through the untracked offset from a cell's center, left as an approximation.

**Where.** `Filters/RbPointMassFilter.h`, `Distributions/GridGaussian.h`.

### rb-particle-filter

*The Rao-Blackwellized particle filter predicts each particle with its Gaussian and conditions on
the draw.*

**Rule.** `RbParticleFilter`'s state is a particle set over a block named by a tag path with a
Gaussian over the rest per particle [rao-blackwellized-core]. `predict()` transforms each particle's
Gaussian over the rest through the whole process model, with the particle as a known input, draws
the particle's new value from the predictive marginal of the block, and conditions the rest on the
draw. The draws are taken first. `update()` is the update given every point [rao-blackwellized-core]
and reweights as the particle filter does; a rejected particle keeps its Gaussian with weight zero.
Resampling copies each particle's Gaussian by its ancestor. Under a `SwitchingModel` each particle
holds a `HybridGaussian`: `predict()` mixes the modes, draws the particle from the mixture of the
modes' predictions, and conditions every mode on the draw, weighting it by the draw's density;
`update()` is `ImmRecursion`'s given a block [imm], with the particles run in order.

**Why.** Drawing the block from its predictive marginal and conditioning the rest on the draw is the
marginalized particle filter's time update (Schön, Gustafsson, and Nordlund 2005), the "measurement"
of the rest by the block's motion, which is how a velocity is learned from positions. One transform
of the whole state lets any process model and method serve, with the model written once as for the
other filters. Passing the particle as a known input keeps a zero-variance direction out of the
transform, which `Unscented` couldn't factor. A hybrid particle drawn from its predictive mixture
keeps its weight, and what the draw says of the mode goes into the mode probabilities.

**Rejected.** The linear sub-models $f_p$, $A_p$, $f_r$, $A_r$ of the paper, written by the caller.
A pseudo-density for a mode whose block covariance is singular, which compares densities of
different dimensions; that prediction fails.

**Where.** `Filters/RbParticleFilter.h`, `Distributions/ParticleGaussian.h`.

### smoothing

*Smoothing is a backward pass over a record the caller keeps of the forward pass.*

**Rule.** A smoother is an observer [composed-features]. The caller records the forward pass in a
history outside the filter, which owns its buffers with an allocator [allocation], and a backward
pass over it gives the smoothed estimates. The one addition to the filters is `predictJoint()`: the
step's starting Gaussian, its prediction, and their cross-covariance. A history keeps every step, or
the newest $L$ with `Lag{L}`, as a list that recycles its nodes. The backward step is
Rauch-Tung-Striebel in the tangent space, applied by each family as its state allows: the IMM by
Nadarajah (2009, §8.2.1) with Barber's (2006) expectation correction, the mixture filter by
Rahmathullah, Svensson, and Svensson (2014) over its `MixtureLineage`, and the particle filter by
ancestor tracing (Kitagawa 1996), told by `ParticleSet::resampleCount()` when ancestors are new.

**Why.** Every smoother needs the same two things from the filter, the posterior and the predicted
joint at each step. A history inside the filter would fix a window and a storage policy before the
use is known and make every filter pay for smoothing it may not do. The joint holds the
cross-covariance, not a Jacobian, because the sigma-point methods produce it and the per-mode and
per-component steps consume it. The joint carries its starting Gaussian because `filter.state()`
after the prediction is the moved state. A list never moves a recorded step, asks a pool for one
node of known size per step, and makes a full fixed-lag window allocate nothing. Only the mixture
sees every merge, split, and removal, so it keeps the lineage.

**Rejected.** The two-filter IMM smoother (Helmick, Blair, and Hoffman 1995): it needs backward-time
dynamics the process models don't have. Smoothing each mixture component against the next step's
collapsed moments: it has no source and mixes every component's evidence. A flag the caller passes
to the particle history: it can be forgotten without a sign. A growing vector for the history: it
copies on reallocation and leaves abandoned buffers in a monotonic resource. Backward simulation for
particles (Godsill, Doucet, and West 2004): it needs a transition density no process model exposes.

**Where.** `Smoothers/`, whose headers state where each smoother departs from its source.

### filters-and-recursions

*A filter holds a state and runs a recursion, and the mixture filter is generic in its component.*

**Rule.** A recursion is a filter without its state: the model, the method, and any scratch, acting
in place on a state passed in. `KalmanRecursion` acts on one component, `ImmRecursion` on a
`Hybrid`, and a component names its recursion. The filters hold a state and a recursion each:
`KalmanFilter<Component, Model, Method, Recursion>`, `MixtureFilter<Component, Model, Method,
Allocator, Recursion>`, and `ImmFilter`, with the prediction method in the filter's type and deduced
from its constructor, so `predict()` is as short as the process model it drives. The recursion is
the last parameter, the component's unless given one, and `Filter{recursion, state}` deduces it from
any type meeting the public `Recursion` concept, so a step of the user's own runs over one state or
over each component of a mixture [user-extension-points]. A `MixtureFilter` over hybrids is a
mixture of IMM filters, each component mixed, predicted, and updated as an IMM is (the Gaussian sum
filter of Alspach and Sorenson with mode-matched components, as Pasha, Vo, Tuan, and Ma 2009 carry
each target). A recursion that writes scratch is not reentrant, so a mixture of hybrids under a
policy other than `Sequential` doesn't compile.

**Why.** The mixture filter's per-component work was the Kalman step written inline, so a mixture of
IMMs would have needed the IMM written inline too, and every form after it again. With the step an
object, each structure composes with each atom's step once. A recursion, rather than a bank of
filters, because the model, the method, and the scratch are one per filter, and the state belongs to
the mixture. An object, rather than free functions, because the IMM's scratch is owned and reused
[allocation].

**Rejected.** `GaussianMixtureFilter` as an alias: MSVC (19.50) silently mis-deduces through an
alias whose parameters sit inside a nested template argument, so it is a derived class with its own
deduction guides (style.md, Types).

**Where.** `Filters/Recursion.h`, `Filters/KalmanFilter.h`, `Filters/MixtureFilter.h`,
`Filters/ImmFilter.h`; `tests/user_recursion.h`.

### autonomous-multiple-model

*The autonomous multiple model estimator is the IMM on the identity chain, and a floor on the mode
probabilities is upkeep.*

**Rule.** `AmmFilter` derives from `ImmFilter` and adds only a constructor that builds the identity
chain on the state's modes. The IMM's prediction skips mixing when its chain is exactly the
identity. `upkeep::floorModes(probability)` is an upkeep action [upkeep] that raises every mode
below the floor to it and shares the rest in proportion, flooring again any mode the sharing takes
below it. It floors a hybrid's modes, a `CategoricalParams`'s values, and the modes of each
component of a mixture of hybrids, which it checks all before changing any.

**Why.** The estimator of Magill (1965), as Athans et al. (1977) §V restate it, is the IMM with
$\Pi = I$, so a separate filter would repeat the IMM's steps, scratch, and results. A derived class
gives the algorithm its name, supplies the argument that makes it the AMM so that the chain can't be
written wrong, and deduces on every compiler [filters-and-recursions]. The floor is Athans et al.
§V-F: under Bayes' rule a probability of zero never returns, and a tiny one takes as much evidence
to return as it took to fall. The paper doesn't say how the sum is kept at one; proportional sharing
keeps every mode at the floor and the others' ratios unchanged. It is upkeep because it
changes the distribution on purpose, at a moment the caller chooses, and serves the IMM too. A hidden
Markov filter whose chain has an identity block, and a mixture of AMM filters, lose a mode the same
way, and an overload, unlike a loop the caller writes, is a value a filter or a trigger can hold.

**Rejected.** An alias or a factory function for the AMM. A floor as a filter option.

**Where.** `Filters/AmmFilter.h`, `Distributions/Upkeep.h`.

### state-access

*Every filter has a mutable `state()` to change its state in place and `setState()` to replace it.*

**Rule.** Every filter has three members for its state: `state() const` reads it; a mutable
`state()` hands it out to be changed where it is between steps, by upkeep [upkeep] or by the
caller, as when inflating a covariance after a gap in the measurements; and `setState(state)`
replaces it. Through the mutable `state()` the state must keep its space and its allocator, and a
state that must not be empty must stay so. `setState()` takes any state the constructor would:
it makes the constructor's checks again, and the filter's scratch follows the new state's
allocator and capacity.

**Why.** The filters differed only by history: the ones whose states take upkeep had a mutable
`state()` and no `setState()`, the Kalman and hidden Markov filters the reverse, and the IMM gained
both when `floorModes()` arrived. No filter needs to differ. A change in place is what upkeep is,
and without it a Kalman filter's user copies the state out and back to edit it. A replacement can
bring a space, an allocator, or a capacity other than the one the filter's scratch was made for,
and only the filter can make its scratch again; an assignment through `state()` cannot.

**Rejected.** A mutable `state()` alone, with replacement by assignment: the constructor's checks
are skipped, and under an allocator that propagates on assignment the scratch stays on the old
one. `setState()` alone: resampling a particle set would copy it out and back.

**Where.** Every header of `Filters/` that defines a filter; `tests/Filters/`,
`tests/allocation_mixtures.cpp`, `tests/allocation_particles.cpp`.

### update-outcomes

*Every filter's update reports one of four outcomes in one shape, and no filter copies its state to
undo one.*

**Rule.** Every filter's `update()` returns a `std::expected` of a result with `bool accepted` and
`logLikelihood()`, the observation's log-likelihood under the prior. The update *conditions* the
state, on every part or on those its policy passed: `accepted`. Its policy *rejects* every part, or
the model gives the observation *no density* under any part of nonzero weight: not `accepted`, the
state as it was, and `logLikelihood()` finite over every part or $-\infty$. Or the number *cannot be
computed*: a `NumericError` [innovation-cholesky], the state valid. An update decides each part from
quantities sized by the measurement before it writes that part (a `Gate` on the NIS, in every form
[gaussian-forms]) and writes the weights last, so a rejection copies nothing; after an error, parts
updated in place before the failed one stay updated, with the weights as they were. A recursion
whose update is not `accepted` has not written its state [filters-and-recursions]. `NisMonitor`
counts rejected updates.

**Why.** A run's score is the sum of `logLikelihood()`. A rejected update still predicted the
observation, so its log-likelihood is its score; skipping it would make the tightest gate the best
model. A copy to undo a failure would cost as much as the update, on every update. A NIS test that
skips rejections hides a diverged filter behind its gate, and is low even for a consistent one.

**Rejected.** An impossible observation as an error: it is the model's verdict on the data, not a
failure to compute it. Rolling back a failed update of several parts: it needs a copy, or a second
pass that keeps each part's gain. A `logLikelihood` field on some results: generic code needs one
spelling. Undefined weights as a precondition: a caller's matrix that does not factorize is an
error, and so are its weights. A `std::variant` or a class in place of one enumeration with two
names: a comparison with an enumerator would not compile.

**Where.** `Core/NumericError.h`, `Filters/Recursion.h`, each filter's `update()`, the acceptors in
`Models/FormSteps.h` and `Models/Reduction.h`; `tests/Filters/update_outcomes.cpp`.

## Project

### layered-headers

*Headers are grouped in layered directories, under one namespace.*

**Rule.** The headers live in six directories under `include/EmrysFilter/`, each a layer that
includes only from itself and the layers before it: `Core/`, `Spaces/`, `Distributions/`, `Models/`,
`Filters/`, and `Smoothers/`. Each has an umbrella header beside it, and `EmrysFilter.h` includes
the six. A header includes another as `<EmrysFilter/Spaces/Space.h>`, never by a relative path, and
`tools/check_layers.py` fails on an include from a later layer. Within a directory, a parameters
type and its view may include each other. `tests/` mirrors the directories. The library keeps one
namespace, `Emrys`. The headers stay the primary interface; a C++23 module, if one comes, is an
addition beside them.

**Why.** A flat directory of 39 headers and 21,000 lines no longer told a reader where to look or
what depends on what, and directories make a violation of the layering visible in an include path.
Eigen is one namespace with module directories, which is the shape to follow, and the umbrella
headers map onto partitions of a module if one is added. The smoothers are a layer after the
filters because each history reads its filter's joints and no filter reads a history; in one
header among the filters, a Kalman smoother included every filter.

**Rejected.** Nested namespaces: the space customization points (`plus()`, `minus()`, the `has...<>`
traits) are found by lookup from every layer, and users' spaces live in users' namespaces, so a
`Filters` namespace calling a `Spaces` function would need qualified calls throughout, and qualified
names in deduction guides would not help MSVC's fragile deduction through aliases. A module in place
of the headers: a template-heavy build spends its time instantiating and generating code, which a
module doesn't share between translation units, so precompiled headers and explicit instantiation
serve the slow builds better, and the embedded toolchains [embedded-builds] are the furthest from
modules.

**Where.** `tools/check_layers.py`.

### analysis-project

*Analysis tools live in one separate project; the libraries keep the numerics.*

**Rule.** The tools for analyzing data, identifying models, building filters and trackers from
them, and verifying accuracy and consistency live in one analysis project that depends on this
library and the tracker. Each library keeps the numerics a tool needs that are a pure function of
its types: header-only, no I/O, allocating nothing the library doesn't already allocate (innovation
statistics, NEES and NIS, log-likelihoods, `simulate()`, the Cramér-Rao bound). Everything with
I/O, dependencies, or a run-time configuration surface is the analysis project's: readers for
recorded data, noise identification (Allan variance, $Q$ and $R$ from data), a factory from
parameters and configuration to a filter, Monte Carlo runners, reports, plots, and a
finite-difference check of a model's Jacobians. Any Python lives there, never in the libraries.
Execution speed is benchmarked here [benchmarks].

**Why.** The numerics are what the test suites and a coding agent's fastest loop consume, so they
must not sit behind a second repository. The pipeline from data to model to estimator to
verification is one pipeline, and tools split by library would cut across every stage. File
formats, plotting, and Python would blur what the library is and load its CI. Each additional
analysis project would add version coupling along a chain where every API change already ripples.

**Rejected.** An analysis project per library. A project of its own for identification from data,
until an audience that never touches a filter appears.

### benchmarks

*Speed is benchmarked in this repository against hand-rolled Eigen.*

**Rule.** Benchmarks live in `benchmarks/` here. Each implements one problem twice, with
EmrysFilter and from scratch in plain Eigen as a careful user would write it, each with an
automatic and an analytic Jacobian: a 2×2 design that separates the library's overhead from the
cost of differentiation. The hand code uses fixed sizes, `noalias()`, and the library's update form
and factorization. Each pair must agree in state and covariance, and pass an NEES check, before any
timing counts. Reproducible measures (Callgrind instruction counts, heap allocations per step,
`.text` size, and stack use on Cortex-M) can gate a change; wall-clock time is reported as a ratio
to the hand code, for users to run on their own hardware.

**Why.** A library's claim to efficiency is credible only against the code its users would
otherwise write; a one-sided benchmark is a strawman. Instruction and allocation counts are
deterministic; ratios carry across machines better than nanoseconds.

**Where.** `benchmarks/`; [benchmarks.md](benchmarks.md) is the plan, and the rule that a test
checks the implementation while a benchmark backs a claim.

### coverage

*Full coverage is instantiation, region, and axis coverage, measured per instantiation.*

**Rule.** Coverage is measured with Clang's source-based coverage per instantiation, restricted to
the library's headers, and "full" means three things: every public entity is instantiated by the
tests (checked against the Doxygen entity list); every region executes in at least one
instantiation; and each generic component is instantiated across its axes (plain and manifold
spaces, fixed and dynamic sizes, `float`, `double`, and an autodiff scalar, default and custom
allocators), a checklist enforced by typed test suites. Each claimed compile-time error has a
negative compile test, counted separately. A region inside a macro's expansion is not counted: it
is almost always the failure path of an assertion, which correct code never takes. Nor is code the
compiler evaluates and never runs: a function only evaluated at compile time is `consteval`, which
the entity check leaves out, and a `constexpr` variable's initializer is left out. A branch no input reaches is not kept: one that
guards an invariant becomes an assertion, one that a compile-time value rules out `if constexpr`.

**Why.** Line coverage misleads in a template library: a line counts if any instantiation runs it,
so a branch exercised only for `Linear<3>` counts for a quaternion, and a template nothing
instantiates produces no code and is invisible rather than zero. Instantiation coverage is also what
makes the no-exceptions build [embedded-builds] evidence, since a compiler vouches only for what it
instantiates.

**Where.** `tools/coverage.py` runs the tests of a `clang-coverage` build and reports regions,
branch outcomes, lines, functions, and instantiations run, by layer; given the Doxygen XML, it
checks the public functions against the instantiations. Regions merge across programs by where they
start, as the assertion override's ends some elsewhere. A claimed compile error is a static_assert
with a message or a deleted function; its negative compile test is a case in
`tests/<Layer>/compile_errors.cpp`, run by the compile-errors workflows [check-stages] and counted
by the report, and `tools/check_compile_errors.py` checks that each claim has one or a reason no
misuse reaches it. CI's `coverage` job runs the tests and the report, and the `layers` job the
check. The axis checklist is not built yet (roadmap, Verification).

### check-stages

*Checks run in stages chosen by what each can catch, and the diff decides which a change needs.*

**Rule.** A change passes checks at three stages: `land` (onto `next`), `pr` (`next` to `dev`), and
`ci` (before a release). A check's stage is set by what it alone can catch: Clang and the checks
that take seconds at `land`; the other compiler families, the negative compile tests, the
sanitizers, Doxygen, and coverage at `pr`; the builds only CI runs (no exceptions, libc++, arm64,
macOS, Cortex-M) at `ci`. A check moves to an earlier stage when the change touches what only it
tests, as a deduction guide does MSVC, a claimed compile error Clang's negative compile tests, or a
`throw` the build without exceptions, and a check is skipped when the change touches nothing it
tests. `tools/required_checks.py` holds the rules and reads the diff. The negative compile tests
run once per compiler, by the `clang-`, `gcc-`, and `msvc-compile-errors` workflows in those
builds, and in no sanitizer or coverage build, which cannot change a compile's result. A case
compiles again only when its file's object, rebuilt on any change to a header, flag, or compiler
the case depends on, is newer than its last pass. The test presets run eight tests at once.

**Why.** Measured in a Clang Debug build on 2026-11-02, compiling the test translation units takes
about 1,300 CPU-seconds and the 182 negative compile cases 416 more, while the other tests run in
under 40; one test at a time, the per-test processes alone made a run 51 s, and eight at a time
12.7 s. The cost is compiles times configurations, so the stages ration configurations, and the
compiles are skipped where an earlier result still holds; a header change rebuilds every case.

**Rejected.** Levels by run time, which saves almost nothing when running is a few percent of the
cost. Rules in prose, which were applied from memory and let the negative tests go unrun on GCC
and MSVC through six new claims. Running only the tests of the translation units that were rebuilt:
the run is seconds, and the build already skips the rest. The negative tests in every build, as
until 2026-11-12, which made `land` pay for them on each header change.

**Where.** `tools/required_checks.py`, `tests/compile_errors.cmake`, `CMakePresets.json`, and
AGENTS.md's "Build and test".

### splitting-hooks

*Splitting by nonlinearity is our own work, kept outside the library, with the library supplying the
hooks.*

**Rule.** The action that splits a mixture component before a nonlinear step stays outside the
library, in the analysis project or a companion, until its criterion is published or found in the
literature. The library supplies every piece it needs, each published: `statisticalLinearization()`
for whether to split [covariance-update-forms]; the second differences along a sigma-point
transform's axes, kept when an input is marked by `withSecondDifferences()`, which with
`along(d)` measure the direction a split would take; the Hessians of a transform's function at
the mean (the second-order extended Kalman filter's), and their product with a direction; and a
split of a Gaussian along a given direction for `GaussianMixtureParams::split()`. What a transform
evaluates anyway goes on its result, when asked for; what needs more evaluations is a function of
its own, so the Hessians are standalone and serve every method. The action is the caller's: an
upkeep function object [upkeep] that probes before the step, or a wrapper of the step that
transforms each component along its direction, keeps the result when the second difference is
small, and splits and transforms the children when it is large.

**Why.** The library implements only the public literature, and that claim is worth keeping exact.
The hooks are useful alone: the linearization for the Joseph form and the information filter, the
Hessians for a second-order filter, the second differences for a check of linearity along any
direction, and the directional split for any splitting rule. Second differences cost only their
storage, since the points were evaluated for the moments; a Hessian on the transform's result would
save one evaluation of the value against roughly $n$ times the Jacobian's cost.

**Rejected.** Keeping every point output, $(2n + 1) m$ values for $n$ read.
`Linearized` with axes, for directional second derivatives. The name `withCurvature()`.

**Where.** [research.md](../research.md) item 6; roadmap step 16b.

### distributed-fusion

*Distributed fusion is factor operations and filter steps here, node bookkeeping in the tracker, and
no project of its own.*

**Rule.** An operation of distributed fusion that is a pure function of this library's types lives
here; one that keeps state per sensor or object, or decides which track goes with which, lives in
the tracker. Here, as factor operations: divide, the inverse of multiply, which is the
decorrelation step and Drummond's tracklet (1997); fusion of two estimates with a known
cross-covariance (Bar-Shalom and Campo 1986) or by covariance intersection (Julier and Uhlmann
1997); and the Kullback-Leibler average (Battistelli and Chisci 2014). Here, on the filters:
out-of-sequence measurements through the smoother's history (Bar-Shalom 2002; Bar-Shalom, Chen,
and Mallick 2004), a consensus step over neighbors' messages (Battistelli et al. 2015;
Olfati-Saber 2009), and the channel filter (Grime and Durrant-Whyte 1994) as an example. In the
tracker: shadow trackers, tracklet extraction from a track stream, track-to-track association,
handoff, and the topologies.

**Why.** The question fusion adds, which information two estimates already share, is answered by
Gaussian algebra, so it belongs beside the Gaussian. A single-target distributed problem, a sensor
network or a team of robots estimating one state, then uses this library alone. The library has no
notion of a node or a link and needs none, since a message is a distribution and a weight.

**Rejected.** A fusion project between the library and the tracker, which would add a link to the
version chain [analysis-project] argues against.

**Where.** Not built yet (roadmap step 16c).

### eigen-partner

*The library is a partner to Eigen, not part of it.*

**Rule.** EmrysFilter is fully C++23, apart from Eigen's family, and named for its organization:
EmrysFilter, in the namespace `Emrys`, capitalized as `Eigen` is, with the implementation in
`Emrys::internal`. It keeps Eigen's naming conventions, so that client code reads as one, and uses
Eigen's facilities (`Eigen::numext`, `unsupported/Eigen/SpecialFunctions`,
`unsupported/Eigen/MatrixFunctions`) in place of hand-rolled ones. A general-purpose piece that Eigen
lacks is written here so that it can be offered upstream (roadmap, "Upstream to Eigen"). The source
follows its own `.clang-format`, not Eigen's.

**Why.** Joining Eigen would hold the library to Eigen's language support. Matching its conventions
keeps the two reading alike where users mix them.

**Rejected.** The name eigen-filter, which suggested a part of Eigen.

**Where.** style.md; the roadmap's "Upstream to Eigen".

### published-algorithms-only

*The library implements only algorithms from the public literature.*

**Rule.** Every algorithm in the library implements one published in the public literature and cites
it (style.md, Documentation), saying what the code takes from the source and where it departs. The
library invents nothing, and its user documentation says so. A method with no source yet stays
outside, as a problem on the research page, with only its published hooks inside [splitting-hooks].
A foundational concept, such as a distribution or a divergence, links its Wikipedia article.
The point-mass filter has two exceptions, kept for now and marked in its documentation
[rb-point-mass]: an IMM per cell, a composition of published filters that no publication found
states, and a cell's covariance about its center, an approximation with no source.
A citation separates credit from provenance: the original is cited for the method, and equation
and section numbers come only from a source that was read, named beside the original when the
original was not (the bibliography's opening).

**Why.** A user can check every algorithm against its source, and the library and its documentation
can accompany a graduate course in state estimation. A feature request therefore brings its
literature (CONTRIBUTING.md).

**Rejected.** Shipping the maintainer's own methods, such as the criterion for splitting by
nonlinearity, before they are published. Citing only originals, with locators recalled rather
than read: a paywalled original's section numbers could not be checked by anyone.

**Where.** [CONTRIBUTING.md](../../CONTRIBUTING.md), [research.md](../research.md),
`docs/reference/bibliography.md`.

### user-extension-points

*What the library won't implement, a user must be able to, against public contracts and without
touching it.*

**Rule.** [published-algorithms-only] keeps every unpublished method out, so the library owes a user
a place to add one: every kind of piece a filter is built from has a public contract that a user's
own type meets by its interface, with no specialization of an `internal` trait, no edit to a header,
and no reading of a filter's private helpers. The contracts are a process model's function and a
measurement model's function and `logLikelihood(g)`, each with its policies and always in the object
`processModel()` or `measurementModel()` returns, never a struct of the user's own; a space; an
execution policy; a reduction policy; an upkeep operation; a recursion; and a transform method,
among them a sigma-point set and a rule for the unscented parameters. Each is a concept or a
documented set of members, with a user-written instance in `tests/` and a "Writing a ..." section in
usage.md. A contract kept closed, as `Component` is, says so and why. A gap filled from outside the
filter, as a consistency check over a mixture's components, is a recipe in usage.md.

**Why.** The library stops at the published algorithms, and many users' needs sit one step past
them; an academic in particular is here to write that step. A library that invents nothing and also
can't be extended leaves such a user forking it.

**Rejected.** Virtual hook points on the components or the filters: a derived component's fields
don't survive a merge or a collapse, and a hook fixes its moment before the methods that need it are
known [composed-features]. Opening `Component`: its traits encode the collapse, which a wrong
specialization corrupts silently.

**Evidence.** `tests/user_*.h`, run through the filters and transforms; usage.md's reduction policy
and its recipe "Checking an estimator"; the upkeep operation of `examples/gaussian_mixture.cpp`.

**Where.** Every contract's header; the "Writing a ..." sections of usage.md.

### user-skills

*Skills for a user's agent install from the library, at its version, into the directory the user's
project names.*

**Rule.** The skills for a user's coding agent are directories in `skills/`, in the Agent Skills
format (agentskills.io), each named `emrys-filter-<name>` and using only the format's own
frontmatter fields. A project installs them with `emrys_filter_add_skills_target(DESTINATION
<dir>...)`, whose target `emrys_filter_skills` is never part of `ALL`, or with
`tools/install_skills.cmake` under `cmake -P`; both run the script. An installed skill holds its
`SKILL.md` with the library's version, and copies of usage.md, choosing.md, the bibliography, and
the examples, and reads nothing outside its directory. The install owns the `emrys-filter-*`
directories of its destination and nothing else there. A skill states a procedure and points to
usage.md, choosing.md, and the examples for the facts.

**Why.** A skill must describe the version the project builds against, and only an install from
that version's source does. Agents read different directories (in November 2026, Claude Code reads
`.claude/skills/`, Codex and Gemini CLI `.agents/skills/`, and Copilot and Cursor both), so no
default serves every agent, and the project names the one its agents read. Facts kept only in
usage.md and the examples can't diverge between the page and a skill.

**Rejected.** A Claude Code plugin or marketplace: its version is the plugin's rather than the
library's the project fetched, and it serves one agent. Fields only Claude Code reads, which other
agents reject. Committed copies of usage.md and the examples inside `skills/`, which would drift.

**Where.** `skills/`, `tools/install_skills.cmake`, `cmake/EmrysFilterSkills.cmake`, the tests
`skills.install` and `example.consumer`, and `examples/consumer/`, which installs into
`.claude/skills/` unless configured with `AGENT_SKILLS_DIRECTORY`.

### cmake-package

*The install is a CMake package, and installs a fetched Eigen beside it.*

**Rule.** With `EMRYS_FILTER_INSTALL`, on by default only at the top level, `cmake --install`
writes the headers to `include/` and, under `share/EmrysFilter/`, the package
`EmrysFilterConfig.cmake` with `EmrysFilter::EmrysFilter` and `emrys_filter_add_skills_target()`,
and the files the skills install from. A request for 0.1 takes only 0.1.x. The package finds Eigen
5.0 with `find_dependency`, looking first where CMake always looks and then in its own prefix.
When the configure fetched Eigen rather than finding it, the install writes Eigen's headers and
Eigen's own package beside EmrysFilter's; when it found Eigen, it installs none. The fetched
Eigen never enters the user's package registry, and without `EMRYS_FILTER_INSTALL` it installs
nothing.

**Why.** Distributions ship Eigen 3.4, so few machines that install EmrysFilter have Eigen 5.0
installed; a package that found no Eigen would install but not load. Eigen's own package, rather
than a copy under EmrysFilter's name, keeps `Eigen3::Eigen` one target for a project that also
finds Eigen itself. Eigen 5.0 building its package registers its build tree in the user's
registry by default, where it outlives the build and answers other projects' `find_package(Eigen3)`.

**Rejected.** Requiring an installed Eigen before installing EmrysFilter, which turns away most
machines. Eigen's headers inside EmrysFilter's include directory, which shadows another Eigen.

**Where.** The top-level `CMakeLists.txt`, `cmake/`, the test `example.consumer`, which builds
`examples/consumer/` against an install, and the install section of `docs/usage.md`.
