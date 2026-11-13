# Style

How EmrysFilter's code, tests, and documentation are written and named. It binds everyone who
contributes, people and agents alike. Read it before editing anything under `include/`, `tests/`,
`examples/`, or `benchmarks/`. What the library does and why is in
[decisions.md](decisions.md); the rules here cite a decision (D*n*) where one gives the reason.

A rule goes in here in the commit whose code first follows it, citing its decision if it has one.
Each rule states what to write, with at most a sentence of reason; reasons that need more belong in
the decision, and history belongs in the commit message. An open question goes in the review in
`docs/guide/` for its area, not here.

## Naming

- Names are written in full except for the words in a closed list, which are always abbreviated, in
  every identifier that contains them (types, functions, variables, files, internals), and never
  mixed with the full word [params-and-density]: `Params` (parameters), `Sqrt` (square root), `Info`
  (information), `Rb` (Rao-Blackwellized), `Imm` and `Amm` (interacting and autonomous multiple
  model), `Ess` (effective sample size), and the index utilities `idx()` and `block()` for values and
  `diffIdx()` and `diffBlock()` for differences [index-names]. An abbreviation is a camel-case word: `RbParticleFilter`,
  `SqrtInfoGaussian`, `toSqrtInfo()`. So is an acronym in our own identifiers (`LltType`,
  `noiseLlt`), while Eigen's names keep Eigen's spelling (`Eigen::LLT`, `matrixLLT()`). Don't
  shorten anything else; it is `differenceDimension()`, not `diffDimension()`. A word joins the list
  only by changing [params-and-density]. Prose and the first sentence of a Doxygen comment write the
  words out. The index utilities are overloaded on their arguments: `idx<Space, Path>()` from the
  type, `idx<Path>(space)`, and `idx<Path>(space, k)` or `idx<Space, Path>(k)` for one repetition,
  and `block<Path>(space, x)`, `block<Path>(space, x, k)`, and `block<Space, Path>(x)` the same way.
- A name that a protocol looks up keeps the protocol's spelling: `allocator_type`, `value_type`,
  `const_iterator`, `begin()`, and `get_allocator()` for the standard library; `Real` and
  `Literal` in a `Eigen::NumTraits` specialization. A trait is a variable template or a concept,
  not a struct with a `value`. One that computes a type is a struct whose member is `type` (named
  `...Of`, or `...Impl` where the alias reads as a verb), followed by an alias that the code
  outside the struct's definitions uses (`DifferenceSpace<S>`, not
  `typename DifferenceSpaceOf<S>::type`). The exception is a trait used in the concept that
  constrains its own alias, as `DifferenceSpaceOf` is in `Space`.
- Implementation details live in `internal`. Namespaces inside `Emrys` are lower case, as Eigen's
  (`Eigen::internal`, `Eigen::numext`) and the standard's are, and the library's own namespace
  keeps its capital, as `Eigen` does [policy-factories]. Tests may reach into `internal` by
  qualified name; user code never does.

## Types and their template parameters

- Every type is parameterized by a space first and a scalar last, but before an allocator,
  defaulting to `double` (`Gaussian<State>`, `ParticleFilter<State, Model, Scalar, Allocator>`). A
  plain vector is `Linear<N>` (`Gaussian<Linear<3>>`); there are no size-taking aliases. The Eigen
  types a space sizes are `ValueVector<S, Scalar>`, `DifferenceVector<S, Scalar>`, and
  `DifferenceMatrix<Rows, Cols = Rows, Scalar>` (`Spaces/Space.h`), and a type's own
  `VectorType`, `DifferenceVectorType`, and `MatrixType` are these. An example or a guide whose
  state has blocks writes them in place of a dimension wherever the vector belongs to a space it
  names, a block's own (`ValueVector<Position>`) included; one whose state is a plain `Linear<N>`
  may use Eigen's fixed-size types, as `linear_kalman.cpp` does, to show that the aliases are a
  convenience. A covariance given by its variances is written `variances.asDiagonal()`, which
  `GaussianParams`, `Moments`, `additiveNoise()`, `on()`, and `onEach()` accept
  (`CovarianceExpression`). A type
  ending in a parameter pack takes the scalar right after the space (`Product<Space, Scalar,
  Factors...>`). A type that holds a distribution per element takes that component first instead,
  since it carries the space and scalar (`MixtureParams<Gaussian<State>, Allocator>`,
  `HybridParams<Gaussian<State>, Mode, Allocator>`, `KalmanFilter<Gaussian<State>, Model, Method>`,
  `MixtureFilter<Component, Model, Method, Allocator>`), a filter's recursion comes last, defaulting
  to the one its component names (`KalmanFilter<Gaussian<State>, Model, Method, Recursion>`), and
  an alias keeps the space-first spelling
  for Gaussian components (`GaussianMixtureParams<State, Scalar, Allocator>`,
  `GaussianMixture<State, Scalar, Allocator>`, `HybridGaussianParams<State, Mode, Scalar, Allocator>`,
  `GaussianMixtureFilter<State, Model, Method, Scalar, Allocator>`) [component-first],
  [filters-and-recursions], [params-and-density].
- A policy, which chooses how an algorithm runs and holds none of the problem's data, is made by
  its factory and named by its type, and the factories live in a namespace per kind
  [policy-factories]: `methods` (`methods::cubature()` makes a `methods::Cubature`), `updates`,
  `reductions`, `resamplers`, `upkeep`, `execution`, whose `sequential` is a constant as
  `std::execution::par` is, and `jacobians` (`jacobians::perColumn()`) [batched-functions]; `roots`
  comes with roadmap step 17t. No policy is built with braces
  outside its own header, and no policy has a deduction guide. A test aliases a namespace it uses
  unqualified (`namespace methods = Emrys::methods;`), and an example writes
  `Emrys::methods::linearized()`. Named as a type, a template with its defaults is written
  `Unscented<>`.
- A filter in `float` holds and computes no `double` [embedded-builds]. A parameter that defines
  the algorithm a policy names is a compile-time template argument, as a `Periodic` interval's are
  (`unscented<0.5>()`, `gaussHermite<3>()`; [sigma-point-sets]), converted to `Scalar` where used;
  one that sets when or how much work is done is a run-time value (`gate(0.997)`, `pruneBelow(1e-6)`,
  `iterated(method, {.tolerance = 1e-6})`) [policy-factories]. A policy that holds such a value
  takes a `RealScalar Scalar_ = double` for it, deduced from the value through
  `internal::DeducedScalar` (`reductions::gate(0.997f)` is a `Gate<Collapse, float>`,
  `upkeep::roughen(rng, 0)` a `Roughen<std::mt19937_64, double>`), and computes in the scalar of what it
  is applied to.
  A braced argument deduces nothing, so `iterated()` names its scalar as a template argument
  instead, `double` unless written, and a `float` filter writes
  `iterated<float>(method, {.tolerance = 1e-6f})`.
- A type that holds a run-time number of elements, and a filter or result over one, takes an
  `Allocator` right after the scalar, defaulting to `DefaultAllocator` (`ParticleSet<Space, Scalar,
  Allocator>`, `ParticleFilter<Space, Model, Scalar, Allocator>`). It rebinds it for each buffer,
  has `allocator_type`, `get_allocator()`, and allocator-extended constructors, and passes it on to
  the types made from it. Nothing sized by a bounded space allocates, and scratch sized by a count
  is owned and reused [allocation]; `tests/allocation*.cpp` checks every filter's steps. A policy
  other than `execution::sequential` is consent to allocate outside the type's allocator: the primitives in
  `Core/Execution.h` take their scratch through `internal::scratchVector()`, from a user's policy's
  `get_allocator()` where it has one and from the heap otherwise, since the standard algorithms
  allocate with no hook anyway [execution-policies].
- An alias template repeats the parameters of its class, so deduction works through it
  (`PointMassFilter`, `GaussianMixture`). MSVC deduces through an alias only if each deduction
  guide lists its template parameters in the class's order, defaulting rather than fixing any it
  can't deduce, and tests deduce through the alias.
- A distribution has parameters and a density [params-and-density]. The type that evaluates, its
  density, has the plain name (`Gaussian`, `GaussianMixture`, `Uniform`, `StudentT`). Where the
  distribution's values can be invalid, such as a covariance that is not positive definite, its
  parameters are a second type, `...Params` (`GaussianParams`, `GaussianMixtureParams`), which is
  changed step by step and caches what evaluation needs; `toDensity()` on it returns the type that
  evaluates, which borrows that cache. Every type that evaluates names its parameters as
  `ParamsType`, read through `Params<D>`, and every parameters type names what `toDensity()` returns
  as `DensityType`, read through `Density<P>` (`Distributions/Params.h`); a type whose values are
  always valid is its own parameters and has no `toDensity()`. Filters, mixtures, and hybrids are
  parameterized by the type that evaluates and store `Params<C>` (`KalmanFilter<Gaussian<State>,
  Model>`), and their deduction guides map the parameters they are given through `Density<P>`.
  Subsets are `...Marginal`. Don't say "owner"; the type a view borrows from is its parameters
  [params-and-density]. `ParticleSet`, a sample set, is the parameters of `KernelDensity`. A view
  must not outlive its parameters: a view maker is `const &` with its `&&` overload deleted, and
  every mutator of parameters advances their `internal::Stamp`, which the views check when read
  [view-stamp].
- `size()` counts elements: mixture components, particles, grid cells, modes. A distribution of
  one element has none. A dimension says its kind: `valueDimension()` is the length of a mean or
  a sample, `differenceDimension()` the rows of a covariance and the degrees of freedom, with
  `ValueDimensionAtCompileTime` and `DifferenceDimensionAtCompileTime` and their `Max...`
  bounds. They differ where a space's differences have other sizes than its values, as a unit
  quaternion's do, so size covariances, normalizers, and noise draws by the difference
  dimension and index them with `diffIdx<Path>(space)`. A space has no `size()` either: it
  implements `valueDimension()`, with `FixedValueDimension`, `ValueDimensionStep`, and
  `MaxValueDimension`, and generic code calls the free `valueDimension(space)` and
  `differenceDimension(space)`, not the member. A `DiscreteProduct`'s parts are axes
  (`axis<I>()`, `AxisCount`) [sizes-and-dimensions].
- A continuous distribution evaluates `logDensity(x)` and `density(x)`; a discrete one
  `logProbability()` and `probability()` of a value or an event, and `logProbabilities()` and
  `probabilities()` for the whole vector. There is no `pdf()`, `pmf()`, or `operator()`
  [density-names].

## Functions

- Fallible operations return `std::expected` with a specific error type. Precondition violations
  use `EMRYS_FILTER_ASSERT`. The library throws nothing of its own and calls nothing that throws
  on a failure it could have checked (`.at()`, `std::expected::value()`, `std::stoi`, an empty
  `std::function`, `dynamic_cast` to a reference), so it builds with `-fno-exceptions`;
  allocation failure is the allocator's [embedded-builds].
- `[[nodiscard]]` on every non-void function, except assignment operators and convenience
  returns such as the index `merge()` returns.
- A function that stores a matrix argument whole takes it as a `MatrixExpression`
  forwarding reference and forwards it, so an rvalue is moved [forwarded-matrices]; a `Moments` it
  stores is taken by value. One that only reads its argument, or copies it into part of a larger
  buffer, takes `Eigen::MatrixBase<Derived> const &`.
- A function template whose body never names a parameter type uses abbreviated syntax
  (`void f(auto const & x)`). When the body needs the type, declare the template parameter and
  use it, never `decltype(x)`.
- An operation whose work grows with a run-time count takes an `ExecutionPolicy` in an overload with
  the policy first, and the overload without one forwards `execution::sequential`
  [execution-policies]. Under it the operation must run the code it ran before, bitwise. Loops go through the `Core/Execution.h`
  primitives, never `std::reduce` or a raw `std::for_each`. A parallel loop body writes only its own
  element or slot: draw from a generator before the loop, factorize before it and read
  factorizations through `internal::FactorizedGaussian` (never `toDensity()` inside it), and never
  `push_back`. Each overload gets a case in `tests/parallel.cpp` against `execution::sequential`.
- Space-aware arithmetic: residuals are `minus(space, a, b)` ($a \boxminus b$), updates are
  `plus(space, x, d)` ($x \boxplus \delta$), a difference taken away is `less(space, x, d)`
  ($x \boxplus (-\delta)$), and weighted means are tangent-space means. Values
  and differences that arrive from outside (a user's mean, a model's output) are canonicalized
  with `normalizeValues(space, x)` and `normalizeDifferences(space, d)`. Never write `a - b` or
  `x + d` followed by a normalize. Call the free functions, never the members: a space has at
  most a `normalize()` for its values, a `difference()` naming its difference space, and its own
  `plus()` and `minus()`, both or neither, and the free functions fall back or do nothing where
  those are absent (`hasPlainValues<>`, `hasPlainDifferences<>`, `isLinear<>`). Inside the library
  the calls are qualified, `Emrys::plus(space, x, d)` and `Emrys::minus(space, a, b)`, so that a
  free function in a user's namespace is never found by argument-dependent lookup; customization
  is by the members only.
- Address state by tag path (`marginal<Position>()`) rather than raw indices wherever a space
  names its blocks, and in model code read and write a vector's block with `block<Path>(space, x)`
  and a covariance's or a Jacobian's with `diffBlock<...>(space, m)`, never by a segment counted by
  hand [tag-addressing].
- A function the library calls that names its input's scalar takes it as its one template
  parameter, `[]<typename T>(auto const & in, auto const & out, auto const & x)`, and fills its
  output from `zeros<T>(out)` [function-scalars]; one that other code also calls deduces it,
  `ScalarOf<decltype(x)>`.
- In the examples and in the documentation's models, which users copy, a block that is a
  difference of another quantity, such as a sensor's bias, is declared in that quantity's
  difference space, `struct Bias : DifferenceSpace<Gyro> {};`, and applied with the space's
  operations, not bare `+` or `-`, even where the space is linear: a reading less its bias is
  `less<Gyro>(reading, bias)`, a reading predicted with its bias is `plus<Compass>(heading, bias)`,
  a step is applied by `plus()`, and `minus()` is kept for the difference of two values
  [difference-spaces]. Bare arithmetic is left for the physics that carries one space into
  another, such as a rate times the step, whose result is named in the space it lands in,
  `DifferenceVector<Heading, T> const turn = rate * dt;`, and for sums of linear quantities such as
  forces. A step built from several reads is named before it is applied. This is a recommendation
  to users, not a requirement, and usage.md says so; tests and benchmarks may write bare arithmetic.

## Expressions and statements

- Scalar math goes through `Eigen::numext::` (`numext::floor`, `numext::log`, `numext::atan2`,
  ...), which dispatches to custom scalars' own overloads, so write no `using std::f;` lines for
  ADL. What Eigen or its `AutoDiffScalar` lacks goes through `Core/Math.h`, whose functions are
  public so that users' functions can call them too: `logGamma<Scalar>()` (`numext::lgamma`
  takes only `float` and `double`), `logOnePlus<Scalar>()`, `isFinite()`, `isNaN()`, and
  `allFinite()` (in place of `numext::lgamma`, `numext::log1p`, `numext::isfinite`,
  `numext::isnan`, and `.allFinite()`). Arithmetic on autodiff scalars yields expression types,
  so pass `numext` a plain `Scalar` (`numext::log(Scalar(a * b))`); the `Math.h` functions that
  return a scalar take it as an explicit template argument for the same reason.
  `tests/Distributions/autodiff_scalars.cpp` evaluates the densities in `AutoDiffScalar`.
- Mathematical constants come from `std::numbers`, never `EIGEN_PI` or `M_PI`. Where the type
  depends on `Scalar`, use `pi<Scalar>()` and `log2Pi<Scalar>()` from
  `Core/Math.h`, which work for custom scalars such as autodiff and multiprecision types
  (`std::numbers::pi_v<Scalar>` is ill-formed for them). Plain `double` contexts, such as the
  `Coordinates.h` intervals, use `std::numbers::pi` directly.
- Initialize with `=` when the initializer already is a value of the type: an expression result, a
  copy, a returned value, a literal, or a lambda (`Eigen::Vector3d d = a - b;`, `double sum = 0;`,
  `for (int i = 0; ...)`, `Scalar m_total = 0;` for a member). The type stays on the left there,
  since the conversion is the point; an Eigen expression is never `auto`, which would keep the
  expression rather than its value. `auto` always takes `=`, since its type comes from the
  initializer: `auto const g = f.marginal<Position>().copy();`,
  `auto const [mean, cov] = moments();`, `if (auto const * k = points.kernel(i))`,
  `auto const draw = [&] { return normal(engine); };`, never `auto const g{...}`.
- A local built from parts (coefficients, constructor arguments, or nothing) names its type on the
  right, so that a block's names line up on the left: `auto const z = Eigen::Vector2d{1.0, 2.0};`,
  `auto filter = KalmanFilter{model, initial};`, `auto mixture = GaussianMixture<State>{State{}};`,
  `auto rng = std::mt19937_64{42};`, `auto const ldlt = Eigen::LDLT<Matrix>{cov};`,
  `auto const space = Unicycle{};`, `auto logWeights = Eigen::ArrayXd{count};`. No local is
  declared bare (`Type name;`), even one a later line fills or resizes: it is
  `auto result = Matrix{};`, `auto sum = LogSumExp<Scalar>{};`. Deduction from
  constructor arguments works through it, and guaranteed copy elision lets it build a type that
  can't be moved. A member, which can't be `auto`, keeps braces
  (`internal::Buffer<Scalar, Allocator> m_scratch{allocator};`), as does a variable at namespace
  scope.
- Exceptions to both: functional casts for conversions (`Scalar(x)`), which `=` would let narrow
  silently, and constructors whose meaning braces change: any class with an initializer-list
  constructor, such as a standard container or `internal::Buffer`, takes parentheses
  (`auto v = std::vector<T>(n);`, `Base(allocator)`).
- Write `*` directly before the object it dereferences (`*density`, `*m_llt`). Where the operand
  is a call or a member chain, examples and tests spell out the accessor instead:
  `initial.toDensity().value()`, not `*initial.toDensity()`. The library can't, since `value()`
  throws, so it parenthesizes the operand, which `*` applies to as a whole against the reading
  order of the chain: `*(covariance())`, `*(other.m_capacity)`.
- Increment and decrement are prefix (`++i`, `--count`) wherever the old value goes unused;
  postfix only where it is read (`result(k++) = i;`).
- An integer literal of five or more digits groups them in threes with `'`, as does the integer
  part of a floating literal (`10'000`, `100'000`, `std::ratio<1, 90'000>`, `20'000.0`); four
  digits stay plain (`1000`, `5000`), and digits after the point are never grouped. Prose groups
  with commas (40,000 cells), and quoted program output stays as printed.
- Format with clang-format 20 or later, which reads the project's own style from `.clang-format`.

## Tests

- Tests are not Doxygen input. Every test file opens with a plain comment saying what it covers,
  the headers under test, and the roadmap step and decisions it is the evidence for, and every
  `TEST` has a plain comment before it saying what it verifies and, where the name does not say,
  where the expected values come from (a hand computation, a closed form, a plain-Eigen
  reference) or why the tolerance is what it is.
- A new test file goes in `tests/CMakeLists.txt`, under the directory that mirrors the header's.
- A static_assert with a message and a deleted function are claimed compile errors, and each gets
  a case in its layer's `tests/<Layer>/compile_errors.cpp`: an `#if defined(CASE)` block whose
  comment names the header and the message (or the declaration), as the file's others do
  [coverage]. A static_assert that only checks the library has no message.

## Documentation

- Every entity is documented, internals included, since the Doxygen build fails on any warning.
  Every header opens with a `\file` comment (what it holds, the decisions it implements, what to
  read next); every public entity has `\ingroup` with its layer's group (`Core`, `Spaces`,
  `Distributions`, `Models`, `Filters`, `Smoothers`) or a subgroup of it; deduction guides sit inside
  `/// \cond DEDUCTION_GUIDES` ... `/// \endcond` rather than being documented.
- Math is in `\f$ ... \f$`. The first sentence is the brief (`JAVADOC_AUTOBRIEF`), and parameters
  are described in prose, not `\param` lists.
- Every algorithm cites its source: an entry in `docs/reference/bibliography.md`, anchored by
  first author and year and cited as `\ref Runnalls2007 "Runnalls (2007)"`, with a sentence on
  what the code takes from it and where it departs. Equation and section numbers come from a
  source that was read; when the original was not, the comment credits it and adds the source
  read (`as given in \ref Sarkka2013 "Särkkä (2013)", §8.2`), as the bibliography's opening
  says. The library implements algorithms from the public literature and has nothing of its own
  to cite. A foundational concept, such as a distribution or a divergence, links its Wikipedia
  article inline.
- A file read in the repository, by people on GitHub or by agents, is Markdown that uses only
  what GitHub renders, with no Doxygen commands: `AGENTS.md`, the files in `docs/design/`, and
  the user's usage page. Long explanation that belongs to a module of the reference is a `.dox`
  file in `docs/reference/`, which adds to the group with `\addtogroup` and can use `\ref`,
  `\snippet`, and math freely; the `\defgroup` in the header stays a paragraph that points to it,
  so that someone editing the header doesn't read it.
- Examples in user documentation model good behavior: they check every `std::expected` before
  using its value, or say in a comment why it cannot fail there (a covariance built positive
  definite a line above). Examples in `docs/design/`, `docs/guide/`, and the reference
  documentation are for the maintainer and needn't.
- In the examples, a full-line comment has a blank line above it, unless it is the first line of a
  block or continues a comment.
- Example code in the user's documents (`README.md`, `docs/usage.md`, `docs/choosing.md`, and
  `docs/porting.md`) keeps to 80 columns, so that GitHub shows each line whole beside its sidebar;
  elsewhere it keeps the library's 120. clang-format doesn't reach either. A comment that shows a result or an error follows its statement where the line fits and
  goes on its own line above the statement where it doesn't. A code block of compiler or program
  output is quoted as printed, whatever its length.
- Prose is in American spelling, as the identifiers are (*meters*, *normalize*, *behavior*), and
  plain enough for a reader whose first language isn't English or who reads through a
  translator: no idioms, no regional or sports metaphors, and literal wording. Titles and
  quotations keep their own spelling (*Mathematical and Computer Modelling*).

## Commits

- An imperative subject line, then prose describing the behavior and API changes. The reason for
  changing a decision goes here, not in the decision's text.
