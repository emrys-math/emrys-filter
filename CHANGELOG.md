# Changelog

Every change a user can see, newest first. Versions follow [Semantic Versioning](https://semver.org/):
before 1.0, a minor release may break code, and its entry says what breaks and what to write
instead.

## [Unreleased]

### Added

- `shared(path<Name>)` in `affineIn()`: the input `Name`, in which the function is affine with a
  Jacobian that is the same at every point of a call, such as one that depends on a sure input or
  on a process model's step. A function marked `writesJacobians()` is asked for it at the first
  point of a factored transform alone, and a batched one writes it once, `jacobian<Name>(y)`.
- An example of `PosteriorCramerRaoBound` on the Models page of the reference: the bound stepped
  along true states that `simulate()` draws, a column per Monte Carlo run.
- Models without a function, `affineIn()` alone as the whole map, updated and predicted exactly:
  `measurementModel(affineIn(on<A>(H_A), onEach<B>(H_B)), noise)`, as `Columns` gives it, and with
  `offset(...)` its constant term; `processModel(affineIn(on<A>(F_A)...), writes(...), noise)`,
  which holds one step and whose fixed step caches its `F`; `affineIn(F)` of one matrix over the
  whole state, as `processModel(F, noise)` and `measurementModel(H, noise)` are; and
  `transform(method, affineIn(...), inputs...)`.
- Constant items in `affineIn()`: `on<Name>(H)`, a matrix over every row of the output for an input
  that the function then does not read, and `offset(on<Block>(c)...)` or `offset(c)`, the constant
  term by block of the output or for all of it: `affineIn(path<Velocity>, on<Bias>(Hb),
  offset(on<Range>(cr)))`. The model reads each such input as an affine one and adds `H x + c` to
  the function's value by `plus()`, and `H` to its Jacobian, in `measurementModel()`,
  `processModel()`, and `transform()` alike. A process model with one holds one step, so only
  `fixedStep()` predicts it. A batched function takes them too: it writes the Jacobians of its own
  blocks, and the model adds the constant columns.
- `OrnsteinUhlenbeck<B, RevertingTo<M>>{TimeConstant{tau}, StationaryVariance{s}}`, an
  Ornstein-Uhlenbeck block that reverts to the mean `M` instead of zero: `x' = phi x + (1 - phi) m`
  with the noise of the form about zero, so that `s` is the stationary variance about `M`. The mean
  is read from the state where the state has it, such as a `Constant<M>` or `RandomWalk<M>` block,
  and is otherwise an input of the call, `sure<M>(m)` or a Gaussian. Both blocks are linear, of one
  size, under different tags. `OrnsteinUhlenbeck<B, Scalar>` is unchanged; the form about a mean
  takes its scalar third.
- [docs/benchmarks.md](docs/benchmarks.md), the benchmark results: each problem's instructions and
  time per step against the same filter written by hand in Eigen, the compile times, the method,
  and how to run the benchmarks yourself.
- The ranging Kalman filters benchmarked against mherb/kalman's four (`emrys_filter_mherb_benchmark`),
  with their agreement and allocation tests, each timed in `double` and in `float`. The pairs
  that do not agree, the EKF in `float` and both UKFs, are timed with a mark, behind tests that
  the difference is mherb/kalman's. Building the tests or the benchmarks now fetches mherb/kalman;
  configure with `-DEMRYS_FILTER_BENCHMARK_MHERB=OFF` to build without it.
- The ranging EKF benchmarked against TinyEKF's (`emrys_filter_tinyekf_benchmark`), with its
  agreement and allocation tests: timed in `double` and in `float`, TinyEKF's default, in which the
  two do not agree and the pair is marked, behind a test that only TinyEKF's estimate strays.
  Building the tests or the benchmarks now fetches TinyEKF; configure with
  `-DEMRYS_FILTER_BENCHMARK_TINYEKF=OFF` to build without it.
- The ranging EKF benchmarked against FrancoisCarouge/Kalman's (`emrys_filter_carouge_benchmark`),
  with its Eigen backend from the library's `support/eigen`, and with its agreement and allocation
  tests: timed in `double` and in `float`, in both of which the two agree. Building the tests or the
  benchmarks now fetches FrancoisCarouge/Kalman; configure with
  `-DEMRYS_FILTER_BENCHMARK_CAROUGE=OFF` to build without it.
- The ranging EKF benchmarked against OpenCV's `cv::KalmanFilter` (`emrys_filter_opencv_benchmark`),
  with its agreement and allocation tests, built with `-DEMRYS_FILTER_BENCHMARK_OPENCV=ON` against
  an installed OpenCV 4 or 5: timed over runs of 300 steps, over which the two agree in `double`,
  beside how far OpenCV's estimate drifts over 3000, and in `float`, its default, in which the pair
  is marked, behind a test that only OpenCV's estimate strays.
- A path value takes a subscript that selects repetitions of the repeated block it runs through,
  and gives a `SelectedPath`: `path<Landmarks>[k]` is one repetition,
  `path<Landmarks>[Eigen::seqN(first, n)]` or `path<Landmarks>[Eigen::placeholders::lastN(n)]`
  several, and `path<Landmarks>[{2, 5, 9}]`, or a `std::array`, a `std::vector`, a `std::span`, or
  an Eigen vector of indices, those listed, in that order.
- `reads()` takes a selected path, so a model reads part of a repeated block:
  `reads(path<Position>, path<Heading>, path<Landmarks>[k])` reads landmark `k` as a `Landmark`,
  and a sigma-point method spreads over those coordinates alone, whatever the count of landmarks.
  A sequence or a list is read as the repeated block with their count, so a function written for
  the whole block runs on part of it, and `on<Name>(path<Landmarks>[seen])` reads it under a name
  that may bound the count more tightly. The model holds the selection, so it is built for each
  measurement whose selection differs.
- A process model's `writes()` takes a selected path and `on<Name>(path)`, as `reads()` does:
  `writes(path<Fleet>[k])` writes vehicle `k`, and the parts of a composition may write different
  repetitions of one block. A measurement model's `writes()` takes neither.
- `on(path<Fleet>[k], model)` places a process or a measurement model on one repetition of a
  repeated block, and `onEach(path<Fleet>[repetitions], model)` places a process model on each
  repetition of a sequence or a list, an input renamed to a `Repeated` tag having a value per
  repetition selected.
- `on(path<Landmarks>[k], x)` and `onEach(path<Landmarks>[repetitions], x)` place a factor of
  `Factors` or `Product`, a block of `Columns`, or a part of `additiveNoise()` on selected
  repetitions, and `on()` a part of `continuousNoise()`. `Columns{onEach(path<Landmarks>[seen], h),
  on<Position>(-h)}` measures the landmarks listed, in a space of their count.
- `consider()` takes a selected path, `consider(path<Landmarks>[k])`, and
  `given<Path>(space, value, selection)` gives selected repetitions to `update()` as one block, and
  to `ImmRecursion::update()`, which updates each mode given them.
- `conditional<Path>(values, repetitions)` conditions a distribution on selected repetitions of a
  repeated block, and its result is the space with them erased. `conditional<Path>(values)` of a
  block inside each repetition, `conditional<TagPath<Fleet, Heading>>()`, is now a compile error,
  where it failed in Eigen before.
- `compose()` and `on<Path>()` take a model whose `reads()` selects repetitions. A composition reads
  each name once, with the selection of the first part that reads it; parts that read a block
  under one name select the same repetitions, which is asserted when the composition meets a
  state. `on<Path>(model)` keeps the selections of the items in the model's block, and
  `on<Path>()` and `onEach<Path>()` keep those of the items outside it, which are read from the
  state: a model placed on vehicle 1 reads `path<Fleet>[0]`. A selected item the state lacks is a
  compile error, since an input of the call has no repetitions.
- `idx()`, `diffIdx()`, `block()`, and `diffBlock()` take an Eigen sequence or a list of
  repetitions where they take one repetition's index: `block<Fleet>(space, x, Eigen::seqN(1, 2))`
  is vehicles 1 and 2, and `idx<TagPath<Fleet, Heading>>(space, std::array{3, 0})` the headings of
  vehicles 3 and 0. They are in `Spaces/Selection.h`.
- `marginal<Path>(k)`, on every distribution that has it, and a Gaussian's `mean<Path>(k)` and
  `covariance<Path>(k)` take an Eigen sequence or a list of repetitions where they take one:
  `g.marginal<Fleet>(std::array{2, 0})` is the marginal of vehicles 2 and 0, in that order, over a
  `Fleet` of two.
- A Gaussian's `covariance<RowPath, ColPath>()` takes a selection of repetitions for each of its two
  paths that runs through a repeated block, in order: `covariance<Position, Landmarks>(k)` is the
  position's cross-covariance with landmark `k`, and `covariance<Landmarks, Landmarks>(i, j)` that
  of landmarks `i` and `j`.
- `nees<Path>(truth, estimate, repetitions)` tests the repetitions selected of a repeated block:
  `nees<Landmarks>(truth, filter.state(), k)`.
- `NeesMonitor::add<Path>(truth, estimate, repetitions)` adds the NEES of the repetitions selected.
- `examples/landmarks.cpp`: a vehicle mapping landmarks, each sighting a model of one landmark,
  `path<Landmarks>[k]`.
- The forms of `idx()`, `diffIdx()`, `block()`, `diffBlock()`, and `marginal()` that take a
  repetition take a path that reaches its repeated block through other tags:
  `block<TagPath<Theater, Fleet>>(space, x, k)`. The path had to start at the repeated block.
- `KalmanFilter<...>::UpdateType<M>` and `ResultType<M>`, what `update()` with a measurement of
  type `M` returns and fills, as the other filters name them; it was
  `decltype(filter)::RecursionType::UpdateType<M>`.
- `ParticleGaussianParams::forEachParticle(policy, f)`, which calls `f` on each particle as a
  `ParticleSet`'s `forEach()` does, to move the particles in place and keep their Gaussians.
- An update takes a sigma-point method spread by axes that can fail, such as
  `spread(cubature(), roots::cholesky())`, and returns `GaussianError::NotPositiveDefinite` when
  the axes cannot take the root of the blocks the measurement reads; it was a compile error. A
  transform or prediction still refuses such axes, and so does an update whose model's `affineIn()`
  names some of the blocks read.
- `SECURITY.md`: how to report a vulnerability privately, and what counts as one. Among them is
  undefined behavior, or a call that never returns, caused by a floating-point value passed to the
  library, NaN and infinities included, in a build with `NDEBUG`.
- A measurement per value of a mode axis may have noise by another axis:
  `byMode<Maneuver>(when<Cruise>(fix), when<Turn>(biasedFix))` where each fix's noise is
  `additiveNoise(byMode<Sensor>(...))`. An axis that chooses twice along a nesting fails to compile.
- A `MixtureFilter` of hybrids, and an `RbPointMassFilter` or `RbParticleFilter` under a switching
  model, take a measurement per value of a mode axis, as the IMM does.
- `observe(method, byMode<Axis>(when<Value>(model), ...), z)` observes a model per value of an
  axis with one observation, making a measurement per value.
- `Zero<Path>{variance}`, a built-in process model that resets its block to zero with that variance
  on each coordinate, independent of the rest of the state, over any step: the model of a block a
  switching model's mode doesn't use and in which the quantity is zero, such as a
  constant-velocity mode's acceleration beside a constant-acceleration mode, where `Constant`
  hands the old acceleration back at the next mixing.
- `CategoricalParams<Mode>::independent(maneuver, sensor)` builds the distribution on a
  `DiscreteProduct` whose axes are independent from one distribution per axis, as
  `MarkovChain<Mode>::independent()` builds the chain, so an IMM's mode prior is no longer written
  in flat-index order; `independent(space, axes...)` takes the space, for a run-time cardinality.
- `KalmanFilter`, `MixtureFilter`, `ImmFilter`, and `AmmFilter` have `emptyUpdate<M>()`, the empty
  result that `update(measurement, policy, result)` refills, made from the state's allocator with
  room for every component or mode, so a reused result is one line:
  `auto result = filter.emptyUpdate<Fix>();`. The IMM's takes a `byMode` of measurements too. It
  replaces `filter.recursion().emptyUpdate<Fix>(filter.get_allocator(), filter.state().size())`
  and a result built from the allocator and reserved by hand, which still work.

### Changed

- The square-root form's update triangularizes its joint root by reflections over the state's
  columns and one column of the noise's root at a time, not over the whole root, so its cost grows
  with the measurement's size times the state's rather than with the cube of their sum: at 3D with
  24 anchors the square-root EKF's step takes 0.62 of mherb/kalman's time, from 1.08, and the
  square-root UKF's 0.89, from 1.27. Its results differ from before by rounding.
- `CholeskyFactorization` no longer zeroes its scratch before writing it, 4.6 KB at a size of 24
  in `double` at every factorization: the ranging EKF's step at 3D with 24 anchors takes
  23,400 instructions, from 28,200. Its results are unchanged.
- `triangularize()` of a column-major matrix whose columns are contiguous works through a map of
  dynamic size, so that it is instantiated once per scalar and bound on the columns, whatever the
  rows and the expression; the square-root forms instantiated it once per stacked shape. A test
  file of the square-root filters compiles about a fifth faster with Clang.
- `onEach()` places a model whose `affineIn()` names `shared(path<Name>)`, which it refused: each
  repetition writes the shared block once, and where the model shares every block of a repetition
  it reads, the placed model shares the repeated block.
- A function marked `writesJacobians()` is asked for the Jacobians a method reads alone in more
  places: `transform()` asks for the factored blocks, and for a `shared(path<Name>)` block at the
  first point alone, as an update does; the update given a block, `update(prior, measurement,
  given<P>(space, value))`, asks Linearized for the blocks outside the given block alone, and for
  those inside it too when it is given as a Gaussian, and a sigma-point method for none; and a
  composition at a point, and the function of a model with constant items in `affineIn()`, pass on
  to the function they call the blocks wanted, or none for a value alone.
- The factored transform reads a block whose Jacobian is the same at every point, a
  `shared(path<Name>)` block or a constant item `on<Name>(H)`, at its first point alone: its mean is
  that column, and it adds nothing to the curvature term.
- `compose()` and `onEach()` forward the Jacobians their batched parts write, each part's or
  repetition's into its rows, so a method no longer differentiates a composition whose parts are all
  marked `batched(f, jacobians::perColumn())`, or a part at a point that writes its Jacobian. A
  composition with any part that writes none is differentiated, as before.
- The mark `jacobians::shared()` is gone: a Jacobian the same at every point of a call is a block's,
  named `shared(path<Name>)` in `affineIn()`. Write `batched(f, jacobians::perColumn())` with
  `affineIn(shared(path<Name>)...)` for `batched(f, jacobians::shared())`, and `jacobian<Name>(y)`,
  written once, for each shared block; a whole-state batched function writes `jacobian(y, j)` per
  column, and a linear one is `affineIn(F)`.
- A function that knows its Jacobian writes it into its output: mark it `writesJacobians(f)`, take
  the output `y` last, `h(in, out, x, y)` or `f(in, out, x, dt, y)`, and write `value(y) = v` and,
  when `wantsJacobians(y)`, `jacobian(y) = J`, or by block `jacobian<Name>(y)` when
  `wantsJacobian<Name>(y)`, in place of `return withJacobian(v, J)`, each block wanted written
  whole, zeros included. `withJacobian()` and `WithJacobian` are internal. A step that reads the
  value alone, such as a particle's, wants no Jacobian, so it is not computed. The output's space must be known before the call: give a
  measurement model whose space came from its returned size `outputSpace()`.
- A batched function writes its output by free functions, as `block<Name>(in, x)` reads its input:
  write `values(y)` for `y.values()`, `wantsJacobian<Name>(y)` for `y.template wantsJacobian<Name>()`,
  `jacobian<Name>(y, j)` and `jacobian<Name>(y)` for `y.template jacobian<Name>(j)` and
  `y.template jacobian<Name>()`, `wantsJacobians(y)` for `y.wantsJacobians()`, and `jacobian(y, j)`
  and `jacobian(y)` for `y.jacobian(j)` and `y.jacobian()`. The members are private.
- The matrix forms are gone; a linear model is one without a function, `affineIn()` alone. Write
  `measurementModel(affineIn(on<A>(H_A), on<B>(H_B)), noise)` for
  `measurementModel(H, reads(path<A>, path<B>), noise)`, splitting `H` into the columns of each
  block, as `h.leftCols<2>()` and `h.rightCols<2>()`, and `on(path<Landmarks>[k], H)` for a selected
  item; `measurementModel(affineIn(items...), noise)` for `measurementModel(Columns{items...},
  noise)`; `measurementModel(affineIn(H), noise)` for `measurementModel(H, noise)`; and
  `fixedStep(processModel(affineIn(F), noise), dt)` for `processModel(F, noise)`, whose `F` is of one
  step, so that `predict(dt)` refuses it. `Columns` is internal.
- A process model with `additiveNoise(Q)`, or a composition or placement with a part that has it,
  holds the `Q` of one step, so `predict(dt)`, `predictJoint(dt)`, a filter's `predict(dt)`, and
  every other prediction over a step it is given refuse it at compile time; it took that `Q` at any
  step, whatever `dt` said. Write `fixedStep(model, dt)` for the step the `Q` is of, or
  `continuousNoise(Qc)` for a step that varies.
- `Gaussian::llt()` is `Gaussian::cholesky()`, and returns a `CholeskyFactorization<MatrixType>`,
  the library's own factorization, where it returned an `Eigen::LLT`; `Gaussian::LltType` is
  `Gaussian::CholeskyType`. Write `g.covarianceRoot()`, new on the density, for
  `g.llt().matrixL()`; `g.cholesky().solve(b)` for `g.llt().solve(b)`; and
  `g.cholesky().logDeterminant()` for the sum over the diagonal of `g.llt().matrixLLT()`.
  `CholeskyFactorization` (`EmrysFilter/Core/Cholesky.h`) also gives `lower()`,
  `inverseQuadraticForm()`, `whitenInPlace()`, and `isPositiveDefinite()`, and can factorize a
  matrix of your own.
- The information form of the Gaussian, the iterated update and the updates of the information
  and square-root forms, the fitting of a mixture, a particle set's conditional, a
  Rao-Blackwellized particle filter's weights, and a conditional with its evidence factorize by
  `CholeskyFactorization` too, so their results differ from before by rounding.
- An update factorizes its innovation covariance by a Cholesky factorization of the library's own
  where the measurement has a fixed size of at most 24, in place of `Eigen::LLT`: without square
  roots, a column at a time, which is what a filter's step waits on. With the substitution of the
  next entry, a step of the ranging EKF in `double` takes 70 ns where it took 167 with 4 anchors
  in 2D, 155 where it took 547 with 8, and 1,144 where it took 2,018 with 24 in 3D
  ([docs/benchmarks.md](docs/benchmarks.md)), and the EKF is now faster than mherb/kalman's,
  TinyEKF's, and OpenCV's at every size measured, by 7% where it is closest. The gain, the NIS,
  and the log-likelihood differ from before by rounding.
- `roots::cholesky()`, by which the default sigma points are spread, takes the root of a
  covariance of a fixed size of at most 24 by the same factorization, so a sigma-point filter's
  points differ from before by rounding. In two runs the ranging UKF with 4 anchors in 2D took
  0.90 and 0.93 of mherb/kalman's time in `double`, where it took 1.10 before.
- The Kalman gain, and every other solve through a Cholesky factorization of a fixed size of at
  most 24 that gives a gain or a statistical linearization, is formed by a substitution the
  library writes out, in place of Eigen's solve, which took most of a small update's time. The
  result differs from before by rounding.
- A solve through a triangular root of a fixed size of at most 24, in the square-root and
  information forms, the divergences, and the square-root update's whitening, is a substitution
  the library writes out too, in place of Eigen's triangular solve, so the results differ from
  before by rounding. A step takes fewer instructions: the ranging UKF's 2,254 at 2D with 4 anchors
  where it took 3,255, and 9,799 at 3D with 6 where it took 11,613.
- A step in `float` at six measurements with AVX takes less time: the ranging EKF's at 3D with 6
  anchors 178 ns where it took 238, and with 12 and 24 anchors 281 and 740 ns where it took 327 and
  860. The factorization and its substitutions read each padded column whole, and an innovation
  covariance whose columns are padded is formed in one pass, so that no load waits on stores of
  another width.
- An update whose innovation covariance holds a NaN returns
  `GaussianError::NotPositiveDefinite` and leaves the state as it was. Before, `Eigen::LLT`
  reported success for such a matrix, and the update wrote NaN into the state, or, in a
  Rao-Blackwellized point-mass filter with a covariance per point, dropped the cell. A measurement
  function whose Jacobian is undefined at a point, as a range's is at the point it is measured
  from, gives such a covariance there.
- A step no longer builds a Jacobian that nothing reads. A process model whose function returns
  its Jacobian, as `ConstantVelocity` and the other kinematic models do, had that Jacobian
  expanded to the whole state at every call, also where only the next state was read:
  `KalmanFilter::predict()` over a `fixedStep()` whose transition is cached, `propagate()`,
  `simulate()`, each particle's step, and each sigma point of a prediction. Those now compute the
  value alone. Where the Jacobian is read, it is copied in blocks of fixed size, and so is a
  measurement's. `KalmanFilter::predict()` over a cached transition forms $F P F^T + Q$ in place,
  so its covariance differs from the earlier one, and from `predictJoint()`'s, by rounding. The
  ranging EKF's step in 2D with 4 anchors takes 1,315 instructions where it took 2,410
  ([docs/benchmarks.md](docs/benchmarks.md)).
- `on<Path>(model)` and `onEach<Path>(model)` keep the path of an item of the model's `reads()`
  outside the block they place it on, which is then read from the state where the state has it and
  is an input by its one tag otherwise. An item of several tags, `path<Leader, Position>`, was an
  input under its last tag, `Position`; write `path<Position>` in the model's `reads()` to keep it
  an input.
- `erase<Path>(element, count)` of `GaussianParams`, of a mixture's parameters, and of `ParticleSet`
  is `erase<Path>(elements)`: one element's index, an Eigen sequence, or a list of indices, each
  named once and all erased together, so the indices are those before the call. Write
  `erase<Path>(Eigen::seqN(element, count))` for `erase<Path>(element, count)`; `erase<Path>(element)`
  is unchanged. A list erases elements that are not next to each other in one call:
  `erase<Landmarks>(std::array{0, 2})`.
- **Breaking:** `Roughen` moves the particles of a Rao-Blackwellized state in place, so it no
  longer allocates a copy of the particle set at each call. A state of the user's own that it
  roughens needs `forEachParticle(policy, f)`, as `ParticleGaussianParams` has, in place of
  `setParticleStates()`.
- `ParticleSet::moments(policy)` under a policy other than `Sequential` takes its chunks' weights
  and deviations from scratch for the policy, from a user's policy's allocator where it has one,
  in place of a heap allocation per chunk.
- The headers compile without warnings under GCC's and Clang's `-Wshadow -Wconversion
  -Wsign-conversion -Wold-style-cast`, which the tests, examples, and benchmarks now build with.
- [docs/choosing.md](docs/choosing.md), "Starting without a prior", recommends starting a
  `SqrtInfoFilter` from no information, the diffuse initialization of the literature, over a
  least-squares fit to the first measurements, and says when to convert once the state is
  determined. It and [docs/usage.md](docs/usage.md) now say that `InfoFilter` can't predict
  until the measurements determine every direction, where `SqrtInfoFilter` can.
- **Breaking:** a filter over a hybrid takes the mode chain and the motion as its first two
  arguments, `ImmFilter{chain, motion, prior}`, and so do `MixtureFilter`, `RbParticleFilter`,
  and `RbPointMassFilter` (`RbPointMassFilter{chain, motion, grid, conditional}`). The motion is
  one process model every mode follows, or `byMode<Maneuver>(when<Cruise>(cruise),
  when<Turn>(turn))`, a model per value of a mode axis in any order, nested where a model depends
  on two axes. Write `ImmFilter{chain, byMode<Maneuver>(when<Cruise>(cruise), when<Turn>(turn)),
  prior}` in place of `ImmFilter{SwitchingModel{chain, cruise, turn}, prior}` or
  `ImmFilter{switchingModel<Maneuver>(chain, cruise, turn), prior}`; a model per mode of a product
  in flat-index order is `byMode<Mode>(m0, m1, m2, m3)` by the product itself. `AmmFilter` takes
  the motion first and the method and upkeep after the state, as the other filters do:
  `AmmFilter{motion, prior}`, `AmmFilter{motion, prior, method}`, `AmmFilter{motion, prior,
  upkeep}`, in place of `AmmFilter{prior, slow, fast}`, `AmmFilter{method, prior, ...}`, and
  `AmmFilter{upkeep, prior, ...}`. `SwitchingModel<ModeSpace, Scalar, Motion>` replaces
  `SwitchingModel<ModeSpace, Scalar, Path, Models...>`, built `SwitchingModel{chain, motion}`;
  `switchingModel<Path>()`, `model<I>()`, and `modelIndex()` are gone, and `motion()` returns the
  motion, whose `alternative<I>().part()` is a model.
- **Breaking:** a part of a model that depends on the mode is
  `byMode<Axis>(when<Value>(part), ...)`, its alternatives labeled by the value of the mode axis
  each serves, in any order; a value named twice, left out, or not the axis's fails to compile,
  and `otherwise(part)` serves the values not named. Noise by mode axis is
  `additiveNoise(byMode<Sensor>(when<Inlier>(inlier).weight(0.9), when<Outlier>(outlier).weight(0.1)))`,
  one `GaussianParams` per value; write it in place of
  `additiveNoise(mixture).byMode<Sensor>()`. The weights are what a filter without the axis reads
  the noise by, as the mixture's were; without them such a filter refuses the noise at compile
  time. A measurement per value of an axis is `byMode<Sensor>(when<Inlier>(nominal),
  when<Outlier>(biased))`; write it in place of `switchingMeasurement<Sensor>(nominal, biased)`.
  It takes no weights, and only a filter whose state holds hybrids takes it.
  `SwitchingMeasurement`, `switchingMeasurement()`, `AdditiveNoise::byMode()`, and
  `ImmRecursion::SwitchingResultType` are gone; `ImmRecursion::ResultType<M>` takes the `byMode`.
- **Breaking:** `ImmFilter` takes a recursion, as `KalmanFilter` and `MixtureFilter` do:
  `ImmFilter<Component, Model, Method, Allocator, Recursion, Upkeep>`, the `ImmRecursion` unless
  given one, and `ImmFilter{recursion, state}` runs a recursion of your own over the hybrid, such
  as one that wraps the `ImmRecursion` and changes a step. Code that deduces the filter is
  unchanged; code that names its upkeep in the type writes the recursion before it:
  `ImmFilter<C, Model, Method, Allocator, ImmRecursion<C, Model, Method, Allocator>, Upkeep>` for
  `ImmFilter<C, Model, Method, Allocator, Upkeep>`. `AmmFilter` keeps its parameters and runs the
  `ImmRecursion`.

- **Breaking:** `MarkovChain::fromDwellTimes()` and `ContinuousTimeMarkovChain::fromDwellTimes()`
  are renamed `fromSojournTimes()`, with the same arguments: write
  `MarkovChain<Mode>::fromSojournTimes(tau, exits)` for `MarkovChain<Mode>::fromDwellTimes(tau, exits)`.
  Sojourn time is the tracking literature's name for the time spent in a mode, and a dwell is
  already a radar's look at one position.
- **Breaking:** the chain on a `DiscreteProduct` whose axes move independently is built by the
  factory `independent()`, not a constructor, for `MarkovChain` and `ContinuousTimeMarkovChain`:
  write `MarkovChain<Mode>::independent(maneuver, sensor)` for `MarkovChain<Mode>{maneuver, sensor}`,
  and `MarkovChain<Mode>::independent(space, maneuver, sensor)` for
  `MarkovChain<Mode>{space, maneuver, sensor}`. A constructor now takes only what the chain
  stores, and the factories compute it.
- **Breaking:** `MarkovChain` and `ContinuousTimeMarkovChain` are built from a matrix by
  `create()`, which checks it and returns a `std::expected` with a `ChainError`; the constructor
  that took a matrix is private. Write `MarkovChain<Maneuver>::create(matrix).value()` for
  `MarkovChain<Maneuver>{matrix}`, `MarkovChain<Maneuver>::create(matrix, space)` for
  `MarkovChain{space, matrix}`, and the same for `ContinuousTimeMarkovChain`; handle the error
  where the matrix is computed at run time. `MarkovChain::create()` divides each row by its total,
  so it takes weights in any proportion as well as probabilities, which no longer need to sum to
  one to `dummy_precision`. `ContinuousTimeMarkovChain::create()` reads only the rates off the
  diagonal and sets each diagonal entry to minus its row's total. Both cast the matrix to the
  chain's scalar, `double` unless named, where the constructor took the matrix's.
- **Breaking:** `fromSojournTimes()` returns a `std::expected` with a `ChainError` in place of
  asserting: write `.value()` after it, or handle the error. The diagonal of its exit weights is
  no longer read, where it had to be zero.
- **Breaking:** `MarkovChain::persistent()` returns a `std::expected` with
  `ChainError::NotAProbability` when the probability of staying is not in [0, 1], in place of
  asserting: write `MarkovChain<Maneuver>::persistent(0.95).value()` for
  `MarkovChain<Maneuver>::persistent(0.95)`, or handle the error.
- **Breaking:** a `SwitchingModel` moves the mode over the same step as its models. With models
  stepped by `predict(dt)`, its chain is a `ContinuousTimeMarkovChain`, and each prediction moves
  the mode by $e^{Q \Delta t}$, where it moved one step of a `MarkovChain` whatever `dt` was; a
  `MarkovChain` there is now a compile-time error. Write
  `SwitchingModel{ContinuousTimeMarkovChain<Maneuver>::persistent(0.05), cruise, turn}` for
  `SwitchingModel{MarkovChain<Maneuver>::persistent(0.95).value(), cruise, turn}`, with each rate
  about the old probability of leaving divided by the step, or make the models fixed-step with
  `fixedStep(model, dt)` to keep the `MarkovChain`. A `ContinuousTimeMarkovChain` given with
  fixed-step models is taken over their step. `switchingModel<Axis>()` takes either chain the same
  way, `transition(dt)` (or `transition()` with fixed-step models) returns the chain over a
  prediction, and `predict(0.0)` no longer moves the mode. The IMM, the AMM, and the
  Rao-Blackwellized filters follow.
- `ChainError` gains `Negative`, `NonFinite`, `ZeroRow`, `ShortSojourn`, and `NotAProbability`, and
  prints in words.

### Fixed

- The factored unscented transform of a bounded input affine in many coordinates compiles: it held
  the Jacobians of every sigma point on the stack at their largest, which over a repeated block of
  at most four 4-coordinate repetitions passed Eigen's limit. It holds them on the heap where they
  would.
- A program that includes `Models/Methods.h` builds with `-Wunused-parameter -Werror` and
  `NDEBUG` again: the sigma-point axes given a root read their argument only in an assertion.
- Every header builds with `-Wall -Wextra -Werror` and `NDEBUG`: seventeen parameters and
  variables that only an assertion reads, in `ContinuousTimeMarkovChain`, `Divergence.h`, `Gamma`,
  `GridGaussian`, `Uniform`, `UniformMixture.h`, `RbUpdate.h`, `PosteriorCramerRaoBound`, the
  emissions of `Emission.h`, and `ParticleFilter::predict()` without inputs, warned as unused.
- `ParticleFilter::predict()` under a parallel policy no longer races: each particle's move
  recorded the change in the set's stamp, a write from every thread. It moves them through the
  set's `forEach()`, which records it once. Found by a ThreadSanitizer run of the parallel tests.
- `predict()` and `predictJoint()` by a fixed-step model that caches its transition, and an
  `RbParticleFilter`'s prediction of each particle's Gaussian by one, read only the lower triangle
  of the covariance, as every other step does; they multiplied the whole stored matrix, whose
  upper triangle need not hold the covariance.
- `PointMassFilter{grid}` and its other whole-state forms deduce under MSVC 19.52, which deduced
  the model's type as the covariance form too through the alias's deduction guides.
- `UpdateBranch::component` and `FilterBranch::component` are 0 in a default-constructed branch,
  not indeterminate.
- `PosteriorCramerRaoBound::covariance()` reports `GaussianError::Singular` for a direction whose
  information is within rounding of zero. From no information a prediction's terms cancel to about
  `1e-14` rather than to zero, which it took for information, returning a covariance of about
  `1e14` for a state it knew nothing of. The bound now factorizes by `CholeskyFactorization` in
  place of `Eigen::LLT`, so its values differ from before by rounding.

## [0.1.0] - 2026-11-13

The first release: a header-only C++23 library on Eigen 5.0 for sequential Bayesian estimation,
used through `find_package(EmrysFilter 0.1 CONFIG)` or `FetchContent`.

### Added

- **Spaces:** a state as a `Composite` of named blocks, each a vector (`Linear`), an angle
  (`Periodic`, `RadiansPi`), a `UnitQuaternion`, a `Repeated` or resizable block, or a discrete
  mode, with `plus()` and `minus()`, tangent-space covariances and means, the covariance reset,
  and a `Space` concept for a space of your own.
- **Distributions:** the Gaussian in covariance, square-root, information, and square-root
  information forms; Gaussian and uniform mixtures; sample and particle sets; kernel densities;
  categoricals and continuous-time Markov chains; hybrids of a discrete mode and a continuous
  state; grids of Gaussians and particles carrying Gaussians; Student-t, Gamma, uniform, and
  products of them; marginals, conditionals, moments, sampling, fitting, and divergences.
- **Models:** process models composed from blocks (`ConstantVelocity`, `OrnsteinUhlenbeck`, and
  your own); measurement models that read and write the blocks they name and choose their method
  (`methods::linearized()`, `unscented()`, `iterated()`, `gaussHermite<N>()`, and the other
  sigma-point sets), Jacobians by automatic differentiation or returned with `withJacobian()`;
  additive, mixture, and by-mode noise; switching models and measurements; emissions for hidden
  Markov models; functions marked `batched(f)`, which take every point at once; simulation; the
  NIS and NEES consistency monitors; and the posterior Cramér-Rao bound.
- **Filters:** `KalmanFilter`, `SqrtKalmanFilter`, `InfoFilter`, `SqrtInfoFilter`,
  `GaussianMixtureFilter`, `ImmFilter`, `AmmFilter`, `ParticleFilter`, `RbParticleFilter`,
  `RbPointMassFilter`, `PointMassFilter`, and `HiddenMarkovFilter`. Each update returns its
  innovation, log-likelihood, and NIS, or the reason it failed as a `std::expected` error.
  Updates take reductions (`reductions::gate()`, `select()`, `collapse()`), and the filters take
  upkeep (`upkeep::resample()`, `roughen()`, `pruneBelow()`, ...) run on a trigger.
- **Smoothers:** fixed-interval and fixed-lag histories for Gaussian, hybrid (IMM), mixture,
  particle, and categorical filters, and `mostProbableSequence()`.
- **Execution:** every loop over a run-time count takes an execution policy,
  `execution::sequential` by default, a standard policy, or your own.
- **Allocation and embedded builds:** nothing sized by a bounded space allocates, as tested under
  `EIGEN_RUNTIME_NO_MALLOC`; run-time counts take a standard allocator; no exceptions are thrown,
  and the library builds with `-fno-exceptions -fno-rtti` and for Cortex-M4F and M7. The scalar is
  a template parameter, `float` included.
- **Printing:** `operator<<` and `std::format` of the library's types, with a precision
  (`std::println("{:.3}", gaussian)`).
- **Documentation:** the [reference](https://emrys-math.github.io/emrys-filter/), the usage page
  and filter-choosing guide in `docs/`, eleven examples run as tests, and the
  `emrys-filter-build-estimator` skill for a coding agent working in your project.

### Known limitations

- [docs/usage.md](docs/usage.md), "What is not built yet", lists what is missing.
- The design reviews in [docs/guide/](docs/guide/) hold open questions whose answers may break
  code in a later 0.x release, among them the hidden Markov filter's argument order and emissions,
  the particle and Rao-Blackwellized filters, and the smoothers' value sequences.

[Unreleased]: https://github.com/emrys-math/emrys-filter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/emrys-math/emrys-filter/releases/tag/v0.1.0
