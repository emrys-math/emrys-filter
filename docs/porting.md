# Porting EmrysFilter

EmrysFilter is a C++23 library, and we have no plans to write it in another language. If you
want it in Python, Rust, Julia, or anything else, or in an older C++, this page is for you, and
for the coding agent you may give the work to. It says what to read first, which parts of the
design are the library and which are only C++, how each likely language takes the parts that
are hard to translate, and what moving to C++20 or C++17 costs.

The library is under the MIT license ([LICENSE](../LICENSE)), so a port may be under any license
that keeps the copyright notice. Say in your port's README that it is a port and which commit of
this repository it follows, so that readers can compare the two. If you open an issue about a
port ([CONTRIBUTING.md](../CONTRIBUTING.md)), we will answer questions about the design and
consider a link from this page.

## What to read, and in what order

The code is the least portable thing in the repository. These are the most portable, and a port
should start from them rather than from the headers:

1. **The decisions**, in [design/decisions.md](design/decisions.md). Its decisions
   ([method-per-measurement], [additive-noise], ...) say what the library does and why. Most are
   about the mathematics and how the parts fit together, not about C++. The table in the next
   section sorts them.
2. **The design reviews**, in [guide/](guide/README.md), one per area still under review. Each
   walks through its area with compiled examples and records what was questioned and what was
   decided. A finished review is deleted, and its explanation moves to the module pages of the
   reference and its reasons to the decisions.
3. **The bibliography**, in [reference/bibliography.md](reference/bibliography.md). The library
   implements published algorithms and nothing of its own, and every algorithm cites its
   source, with a sentence on what the code takes from it and where it departs. Implement from
   the paper and the note together, not from the C++ alone.
4. **The tests**, in `tests/`, as a conformance suite. Every test says before it where its
   expected values come from: a closed form, a hand computation, or a reference written in plain
   Eigen. Those values do not depend on the language, so a port can carry the cases over one
   for one and check the same numbers to the same tolerances.
5. **The headers**, in layer order. The library is six layers, each using only the ones before it
   [layered-headers]: `Core/`, `Spaces/`, `Distributions/`, `Models/`, `Filters/`, `Smoothers/`. Port them in that
   order. Inside a layer, each header opens with a comment saying what it holds, the decisions it
   implements, and what to read next.

A port that follows this order has something that runs and passes tests at the end of each
layer: the spaces' arithmetic, then the Gaussian and its forms, then transforms and updates,
then the first filter (`KalmanFilter`), then the others.

## What carries over, and what is only C++

| Kind | Decisions | In a port |
|---|---|---|
| The design | [templated-measurements], [method-per-measurement], [additive-noise], [weighted-branches], [reduction-policies], [switching-models], [sensors], [linear-sensors], [measurement-space-functions], [likelihood-measurements], [continuous-time-noise], [prediction-inputs], [composed-features], [affine-coordinates], [mixture-filter], [discrete-spaces], [imm], [rb-point-mass], [upkeep], [rao-blackwellized-core], [rb-particle-filter], [sizes-and-dimensions], [tag-addressing], [difference-spaces], [tangent-gaussians], [lie-groups], [density-names], [categorical-params], [product-factors], [conditional-evidence], [smoothing], [divergences], [covariance-update-forms], [covariance-repair], [fixed-step-models], [innovation-statistics], [returned-jacobians], [emissions], [covariance-reset], [unit-quaternion], [uniform-mixtures], [filters-and-recursions], [user-extension-points], [gaussian-forms], [form-densities], [iterated-update], [autonomous-multiple-model], [sigma-point-sets], [batched-functions], [linear-space], [no-bounded-coordinates], [cramer-rao-bound], [several-resizable-blocks], [returned-jacobians], and [innovation-cholesky] | Keep. These are what the library is. |
| An intent with a C++ mechanism | [templated-measurements], [filters-and-recursions], [allocation], [index-names], [layered-headers], [embedded-builds], [checked-factories], [borrowed-intermediates], [element-marginals], [execution-policies], [component-first], [view-stamp], [time-steps], [printing-and-simulation], [params-and-density], [coordinate-parameters], [space-storage], [random-draws] | Keep the intent and use the target language's mechanism; see below. |
| Only C++ | [forwarded-matrices] | Drop. |
| How the project is run | [analysis-project], [benchmarks], [coverage], [check-stages], [cmake-package], [user-skills], [eigen-partner], [published-algorithms-only], [splitting-hooks], [distributed-fusion] | Not code. Read them for context. |

The second row needs a sentence each:

- **[templated-measurements], templates instead of type erasure.** The intent is that a filter knows
  each sensor's type, so an update costs no dispatch. Use the language's generics or traits; where
  it has none, dynamic dispatch is acceptable.
- **[filters-and-recursions], [component-first], [params-and-density], the spelling of types.** The
  order of type parameters (the space first, the scalar last, then the allocator; the component
  first for types that hold one per element) is C++. The split between a distribution's parameters
  (`GaussianParams`) and its density (`Gaussian`) is the design: parameters can be invalid and are
  changed step by step, and a density is valid and evaluates.
- **[allocation], allocation.** The intent is that a step allocates nothing sized by a state whose
  size is bounded, and that scratch sized by a count is owned and reused. Keep that where the
  language can (Rust, Julia, C#, Zig); in a language with a garbage collector, keep only the reuse
  of scratch.
- **[index-names], [params-and-density], names.** Names are written in full, except for a closed
  list of words that are always abbreviated: `Params`, `Sqrt`, `Info`, `Rb`, `Imm`, `Amm`, `Ess`,
  and the index functions `idx()` and `diffIdx()`. Map them to the language's case conventions and
  keep the list closed: `log_density`, `sqrt_info_gaussian`, `diff_idx` in Python and Rust;
  `logDensity`, `SqrtInfoGaussian` in Java, Kotlin, and C#.
- **[layered-headers], layers.** Keep the six layers and the rule that a layer uses only the ones
  before it.
- **[embedded-builds], no exceptions.** Every operation that can fail returns a value that holds a
  result or a specific error, and the library throws nothing of its own. Use the language's result
  type, or its convention where it has none (see Python).
- **[checked-factories], construction that can fail.** A distribution built from values a user
  computes at run time (`Gamma`, `Uniform`, `StudentT`) is made by a factory that checks them and
  returns a result, never by a constructor that asserts. Use the language's fallible constructor.
- **[borrowed-intermediates], borrowed intermediates.** Inside one step, matrices are referred to
  rather than copied. It matters only where copies cost something the language lets you avoid.
- **[element-marginals], marginals as views.** A marginal reads its source rather than copying it,
  and a marginal of a mixture or a particle set is a range over its elements. Keep it where views
  are cheap (Rust slices, Julia views); elsewhere a copy gives the same numbers, only slower.
- **[execution-policies], execution policies.** An operation over a run-time count of elements
  (particles, grid cells, mixture components) can run in parallel, and run sequentially by default.
  Use the language's parallel loops.
- **[view-stamp], views.** A view must not outlive the parameters it reads. Rust's borrow checker
  enforces that at compile time and a garbage collector makes it moot; only a language with neither
  needs the stamp check that C++ uses.
- **[time-steps], time steps.** A time step is converted once, at the public entry point, to the
  filter's scalar. Keep that; how the conversion is spelled is the language's.
- **[printing-and-simulation], printing.** Every distribution and filter prints a summary of a fixed
  shape. Use the language's display protocol (`__repr__`, `Display`, `toString()`).
- **[coordinate-parameters], [space-storage], spaces that cost nothing.** In C++ an angle's interval
  is a template argument, so a fixed-size space is an empty type, named rather than passed
  (`minus<State>(a, b)`), and a distribution over it is no larger than one over plain vectors. Where
  a space is an object built once (next section), keep its parameters as members and pass it; keep
  only the rule that a distribution stores a space's counts, never more.
- **[random-draws], random draws.** The caller owns the generator, and the draws come from the
  standard library's distributions, so one seed fixes them within one standard library only, and
  the tests compare statistics, never fixed draws. Use the language's generator, and carry the tests
  over as statistical ones.

## The part that is hard to translate: spaces

Everything in the library is built on a **space**: the set a state's values live in, with
$x \boxplus \delta$ (`plus`) and $a \boxminus b$ (`minus`), a value dimension (the length of a mean)
and a difference dimension (the rows of a covariance), which differ for a unit quaternion. A
`Composite` joins spaces into blocks named by tag types, and the rest of the library addresses state
by tag path (`marginal<Position>()`) rather than by index ([sizes-and-dimensions],
[covariance-reset], [unit-quaternion], [linear-space], [no-bounded-coordinates], [cramer-rao-bound],
[several-resizable-blocks]).

In C++, a space is a type, and its layout, its dimensions, and the index of every tag path are
computed at compile time. Fixed sizes are what let a step run without allocating, and a tag
that is missing is a compile error. A port has to decide where that work happens:

- **At compile time**, if the language can compute a composite's dimensions from its blocks'
  dimensions: C++, Zig, Julia (through its types and specialization), and Rust only partly
  (see below).
- **Once, at construction**, otherwise: the space is an object that resolves each tag path to a
  range of indices when it is built, and every later use reads that range. A missing tag is
  then an error at construction rather than at compile time.

Either way, resolve paths once and never search for them inside a step. Keep the arithmetic in
the free functions (`plus(space, x, d)`, `minus(space, a, b)`, `normalizeValues(space, x)`),
never `a - b` on a value, so that a port of the unit quaternion and the periodic coordinates
works everywhere a plain vector does.

## Language notes

Each sketch below is the C++ example

```cpp
auto filter = KalmanFilter{motion, initial};
filter.predict(dt);
// std::expected; an error if S is not positive definite [innovation-cholesky]
auto const stats = filter.update(rangeAt(z));
```

as it might look in the target language, where `rangeAt(z)` makes a measurement from a
sensor's observation `z` [sensors]. They are sketches of a design, not a working API.

### Python

Use **JAX**, not NumPy alone. A filter step on a state of ten or so dimensions is a few dozen
small matrix operations, and in plain NumPy the interpreter's overhead can be a hundred times
the arithmetic. JAX compiles a whole step with `jit`, and three parts of the design become
features of JAX:

- `vmap` is the execution policy [execution-policies] and the batched function [batched-functions]
  at once.
- `jax.jacfwd` gives any user function its Jacobian, which is what [returned-jacobians] asks a
  function to supply with its value.
- The scalar type (`float32`, `float64`) is a dtype rather than a type parameter.

A space is an object built once, and a tag path is a field name or a string. Under `jit`, a
step must be a pure function, so steps return a new filter rather than changing it, and a
failure cannot raise: a fallible step returns an `ok` flag beside its result, as JAX's own
solvers do. Outside `jit`, a port may raise exceptions instead, which is the Python convention.

```python
State = Composite(
    position=Linear(3), velocity=Linear(3), attitude=UnitQuaternion()
)
estimator = KalmanFilter(motion, initial)
estimator = estimator.predict(dt)
estimator, stats, ok = estimator.update(range_at(z))
```

Python bindings to the C++ library are the alternative, and a weaker one. A template library can
only be bound for a fixed set of instantiations, and a model written in Python would be called back
once per point. Batched functions [batched-functions] reduce that to one call per batch, which makes
bindings usable for states whose size is set at run time.

### Rust

Rust is the closest match to the C++. `Result` is `std::expected`, `no_std` is the embedded build
[embedded-builds], traits are concepts, `rayon` gives the execution policies, `nalgebra` is Eigen,
and `num-dual` gives automatic differentiation for the scalar. The borrow checker enforces
[view-stamp] at compile time, so the stamp is not needed.

The hard part is a composite's dimension. On stable Rust, code generic over a space `S` cannot
write `SMatrix<T, { S::DIM }, { S::DIM }>`, because that needs the unstable
`generic_const_exprs`. There are two ways through:

- `nalgebra`'s type-level dimensions (`DimName`, `DimAdd`), which work and give long error
  messages.
- Associated storage types on the space (`type Value; type Covariance;`), filled in by a derive
  macro. A concrete struct can use its dimension as an array length; only generic code cannot.

The derive macro also gives tag paths a better form than in C++, since a field is its own tag:

```rust
#[derive(Space)]
struct State {
    position: Linear<3>,
    velocity: Linear<3>,
    attitude: UnitQuaternion,
}

let mut filter = KalmanFilter::new(motion, initial);
filter.predict(dt);
let stats = filter.update(&range_at(z))?;
let position = filter.state().marginal::<State::Position>();
```

Rust has no variadic generics, so `Product<Space, Scalar, Factors...>` needs implementations for
tuples up to some length, generated by a macro. It has no overloading, so the overloads of
`idx()` become functions with different names. The allocator parameter becomes `allocator-api2`
or is dropped in favor of fixed capacity (`heapless`, `arrayvec`) or `Vec`.

### Julia

Julia may be the most natural target. Multiple dispatch does what the C++ templates do,
`StaticArrays` gives fixed sizes that do not allocate, `ForwardDiff` dual numbers give a generic
scalar and Jacobians, `ComponentArrays` is tagged blocks over a flat vector, and `Manifolds.jl`
has the retractions a space needs. Test for allocations with `@allocated` where the C++ tests use
a counting allocator. The cost is compile latency at the first call. Julia already has filtering
packages, such as `LowLevelParticleFilters.jl`; a port adds the parts they lack, such as the
hybrid filters and the Rao-Blackwellized filters.

### C\#

C# fits better than its reputation in numerical work suggests. Generics over value types are
compiled once per type, as C++ templates are. Static abstract members of interfaces let a space
give its dimension as a static member, generic math (`INumber<T>`) makes the scalar a type
parameter, and `Span<T>` and `stackalloc` keep a step from allocating.

### Java and Kotlin

The users are on Android and in robotics; WPILib's Java filters, with dimensions as `Nat<N>` type
parameters, show how far the type system goes. Generics are erased and there are no value types yet,
so a space is an object built once, and only `double` is practical (EJML handles `float` by
generating a second copy of its code). There is no automatic differentiation, so a user supplies a
Jacobian or chooses a sigma-point method; the library itself never takes a derivative by finite
differences [returned-jacobians]. [allocation] becomes the reuse of preallocated workspaces, as EJML
does it. In Kotlin, sealed classes and data classes fit update branches and their labels
[weighted-branches], operator overloading makes $\boxplus$ and $\boxminus$ readable, and a result
type replaces the exception.

```kotlin
val filter = KalmanFilter(motion, initial)
filter.predict(dt)
when (val stats = filter.update(rangeAt(z))) {
    is Ok -> log(stats.value.nis())
    // NotPositiveDefinite; the state is unchanged [innovation-cholesky]
    is Err -> skip(z)
}
```

### Others

- **Zig.** Its compile-time evaluation can compute a composite's layout and dimensions, which is
  closer to C++ than Rust gets. The language is not yet stable.
- **MATLAB.** Many aerospace users work in it, but its own toolboxes already cover most of the
  filters. A port adds the most where they are thin: spaces other than vectors, and the hybrid
  filters.
- **Go.** Not recommended. Without operator overloading the arithmetic is hard to read, and its
  numerical libraries are few.

## C++20

**Cost: small, and mechanical.** The library uses one thing from C++23, and nothing else:
`std::expected` and `std::unexpected`, in 40 headers, with `transform()` on an expected and
`std::unexpect`. It used `std::forward_like` too, and no longer does: `Core/Forwarding.h` spells
its own `internal::forwardLike()`, since libstdc++ 14.2 deduces the standard one's return type in
a way Clang 21 rejects, and that one is C++20.

Replace `std::expected` with an implementation that runs on C++20 and has `transform()`, such as
[tl::expected](https://github.com/TartanLlama/expected).

We checked this on 2026-10-31, when `std::forward_like` still needed
replacing too: with both replaced, `EmrysFilter.h` and the tests of the Kalman, IMM,
Rao-Blackwellized particle, and update-failure filters compiled as C++20 with Clang 21. We
compiled them; we did not run them, and the check has not been repeated since. Eigen 5.0 needs
only C++14. The standard library must have `<format>`, since `Core/Print.h` specializes
`std::formatter`, unless printing is left out with `EMRYS_FILTER_NO_IO` (which Eigen's
`EIGEN_NO_IO` implies).

## C++17

**Cost: large.** The library's generic interfaces are written with concepts throughout, and C++17
has none, so a move to C++17 rewrites how every template states what it accepts. The arithmetic,
the algorithms, and the tests' expected values stay the same. If a C++17 compiler is the only
reason, check first whether the toolchain has a C++20 mode; most embedded toolchains from GCC 10
on do.

The C++20 features in the headers, as counted on 2026-10-31 (a count, not a C++17 build, so
expect the compiler to find more):

| Feature | Uses | Replacement in C++17 |
|---|---|---|
| Concepts | 49 | `constexpr bool` traits, written with `std::void_t` |
| `requires` clauses | 267 | `std::enable_if_t` on a template parameter |
| Constrained template parameters (`template <Space S>`) | about 300 | `typename S` with an `enable_if_t` or a `static_assert` |
| Abbreviated function templates (`auto const & x`) | about 275 | Explicit template parameters |
| Template lambdas (`[&]<std::size_t... I>(...)`) | 67 | Generic lambdas taking `std::integral_constant`, or helper functions |
| `consteval` | 29 | `constexpr` |
| `std::span` | 70 | A C++17 span such as `tcb::span`, or `Eigen::Map` |
| `std::ranges` and views | about 20 | Algorithms on iterator pairs |
| `std::remove_cvref_t`, `std::type_identity` | 143 | Two one-line aliases |
| `std::uniform_random_bit_generator` | 91 | A trait |
| Class types as template parameters | 3 declarations | See below |
| `std::numbers` | 4 | Constants in `Core/Math.h` |
| Designated initializers | 6 | Positional initializers |
| Defaulted `operator<=>` | 1 | The six comparison operators |
| `std::formatter` | 1 | Remove, or use {fmt}; `operator<<` stays |
| `[[likely]]` | 2 | Remove |

Four of these need more than a substitution:

- **Overloads that rely on one constraint subsuming another.** Where two overloads differ by a
  concept that refines another, C++20 picks the more constrained one, and `enable_if` cannot.
  Make the conditions mutually exclusive, or rank the overloads with tag types.
- **Constraints on members that are not templates.** Roughly 90 of the `requires` clauses
  constrain a member function of a class template that has no template parameters of its own.
  `enable_if` needs one, so each becomes a template with a defaulted parameter; a constrained
  copy or move constructor cannot become a template and needs a conditional base class instead.
- **Class types as template parameters.** `Periodic<PeriodicInterval{.lower, .upper}>` takes its
  interval as a template argument of class type, with members of type `double`, and C++17 allows
  neither. Pass a type whose static members are the bounds (`Periodic<RadiansPi>`). The two
  masks in `Distributions/Product.h`, of type `std::array<bool, N>`, become a pack of `bool`s.
  Compile-time `std::ranges::count`, `find`, and `sort` in the same file become loops.
- **Deduction through aliases.** C++20 deduces template arguments through an alias template
  (`GaussianMixture{...}`, `PointMassFilter{...}`) and for aggregates; C++17 does neither.
  Users then write the template arguments, or call a function that makes the object.

The tests use the same features, so they need the same work. `[[no_unique_address]]` is already
behind a macro that checks whether the compiler has it.
