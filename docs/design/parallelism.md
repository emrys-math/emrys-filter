# Parallelism and GPU execution

Where the library takes an execution policy, how the per-element work would run on a GPU, and what
to find out before building the second. Step 18's choices are decided: [execution-policies] in
[decisions.md](decisions.md) records them, and the section on the execution policy below keeps only
what [execution-policies] doesn't say. The GPU half is still a plan; when roadmap step 19 starts,
the choices that survive go into decisions.md too. Status and ordering are in
[roadmap.md](roadmap.md).

## Where we are

Before step 18 the library had no parallelism of its own: no header named a thread, and the only
parallel work was Eigen's SIMD inside the small fixed-size expressions. Since step 18
[execution-policies] the loops in the table below take an execution policy; still no header includes
`<execution>`, which with libstdc++ and oneTBB installed would make every user link TBB
[execution-policies]. The guides saw this coming twice: the distributions review noted that
`GaussianParams`'s lazy factorization behind `const` and its plain `int` lock counters raced under
concurrent use, "which a particle filter evaluating likelihoods in parallel will meet" (settled by
[view-stamp]), and the Kalman filter review
rejected a process-wide quantile cache because it would be shared mutable state in a header-only
library.

Two things already in place turn out to be most of the groundwork:

- **Allocation control [allocation].** Every type that holds a run-time number of elements keeps its
  storage in `internal::Buffer<T, Allocator>`, rebinding the caller's allocator, and every filter
  owns its count-sized scratch from the same allocator and reuses it. Device memory is a question of
  which allocator the caller passes, not a new storage design (see "Memory" below).
- **Per-element bodies that are already kernels.** A particle's prediction is `propagate()`,
  `plus()`, and a factor times a draw; its weighting is `measure()`, `minus()`, and a factorized
  density's `logDensity()`; a Rao-Blackwellized point's update is one Kalman update on fixed-size
  matrices. These bodies touch nothing shared, size nothing by the count, and are templates on the
  user's model, which is exactly what a GPU thread runs. The library's prediction of what parallel
  code looks like has, so far, been right by accident of [allocation] and [sizes-and-dimensions];
  the work is to give the loops around those bodies a policy.

## The loops that take a policy

Every loop over a run-time count of elements, in the headers today. "Shape" is the parallel
primitive each loop reduces to, which is what a backend has to supply (next section).

| Where | Loop | Shape | Count | Notes |
| --- | --- | --- | --- | --- |
| `ParticleFilter::moveParticles()` | draw noise, draw inputs, propagate, `plus` per particle | for-each | particles | the draws are inside the loop today; the RBPF draws its normals first and so can go parallel as it stands |
| `ParticleFilter::update()` | `fillLogLikelihoods()`, add prior weights, log-sum-exp, renormalize | for-each, reduce, for-each | particles | the density is factorized once before the loop already |
| `ParticleSet::resample()` | exp the weights, cumulative sum and inversion, gather by ancestors, reset weights | for-each, scan + search, gather, fill | particles | the four resamplers all invert a cumulative sum; multinomial sorts its uniforms, which a binary search per draw avoids |
| `ParticleSet::forEach()` | the caller's body per particle | for-each | particles | parallel only when the body is independent per particle, which the policy argument states |
| `ParticleSet::removeIf()`, `compact()`, `GridGaussianParams::normalizeAndRemoveDead()` | keep the elements passing a predicate, in order | compaction (scan + scatter) | particles, cells | |
| `internal::logTotalWeight()`, `ess()`, `LogSumExp` | sums over the weights | reduce | particles, components, cells | the one-pass log-sum-exp is sequential; the parallel form reduces pairs (see "Reductions") |
| `internal::weightedMomentsIn()`, `mixtureMomentsIn()` | weighted mean and covariance, two passes in a curved space | reduce with a vector, then a matrix, as the value | particles, components | the value type is a fixed-size Eigen object; the pass structure is unchanged |
| `KernelDensity::logDensity()` | one kernel per particle at a point | reduce | particles | parallel over particles for one point, or over points for many |
| `Roughen`, `internal::fillStandardNormal()` | a draw per particle | for-each with a generator | particles | needs the counter-based generator |
| `RbParticleFilter::predict()` | transform, condition, draw pinned per particle | for-each | particles | normals drawn first already |
| `RbPointMassFilter::predict()` | `predict()` per cell | for-each | cells | |
| `internal::updateGivenEach()`, `updateShared()`, `reweight()` | a Kalman update per point, then log-sum-exp and renormalize | for-each, reduce, for-each | particles, cells | the shared-covariance form does one covariance and a for-each over means |
| `GaussianMixtureFilter::update()` | `update()` and the policy per component, log-sum-exp, add to the posterior | for-each, reduce, for-each | components | `m_posterior.add()` is a `push_back`; parallel writes need preallocated slots indexed by component |
| `GaussianMixtureParams::logDensity()`, `moments()`, `ess()` | per component | reduce | components | `logDensity()` calls `toDensity()` per component inside the loop; factorize all first |
| `MixtureFit` E-step and M-step | residuals of all points under each component; log-sum-exp per point; weighted moments per component | for-each over components × points, reduce per point, reduce per component | components × points | the largest regular workload in the mixture family |
| `kullbackLeibler(MonteCarlo, ...)` and the other estimates | evaluate log p and log q per draw, mean and variance | for-each with a generator, reduce | draws | the draws come from a mixture or KDE, so they need the generator too |

Not on the list: the Kalman filter, the transforms, and the IMM filter. Their loops are over a
state's dimension or a handful of modes, where Eigen's SIMD is the right parallelism and a
thread or a kernel launch costs more than the work. Nothing sized by a space gets a policy.

Two kinds of count differ in scale. Particles and grid cells run from $10^3$ to $10^7$; mixture
components from 1 to perhaps $10^3$. The mixture loops take the policy for uniformity and for
the E-step, but a five-component mixture update should stay `Sequential`, and the default says
so.

## The execution policy

[execution-policies] decides it: a leading argument, `Emrys::Sequential` by default, any standard
policy through `std::for_each` over a counting iterator, and every reduction and scan in fixed
chunks combined in order. Where the build departs from the plan above it, and what the decision
leaves out:

- **One standard algorithm, not five.** The plan forwarded each primitive to its standard
  counterpart (`std::transform_reduce`, `std::inclusive_scan`, ...). Their partition, and so their
  rounding, is the implementation's, which made a parallel result reproducible only to rounding,
  across runs as well as thread counts. Built on `std::for_each` over chunks of a fixed size, a
  reduction is the same every run and at every thread count, and a device backend has one algorithm
  to supply. `std::is_execution_policy_v` is not the test either: libstdc++ names it only in
  `<execution>`, and the library keeps that header out so that a user who never passes a policy
  never links TBB [execution-policies]. `ExecutionPolicy` asks whether `std::for_each` takes the
  type instead.
- **Sequential is the old code, not a plain-loop rewrite of it.** Under `Sequential` each
  reduction is one chunk of the whole range, so the whole-array Eigen expressions stay, with
  their vectorization, and every result is what it was.
- **Cheap reductions of dear elements are bitwise.** The EM log-likelihood, the Monte Carlo
  estimates, and the mixture filter's evidence compute their elements as a for-each into scratch
  and reduce in order, which matches Sequential exactly, at the cost of the scratch.
- **The multinomial resampler's ancestors come in draw order.** Under a parallel policy it
  inverts each draw by binary search and does not sort, as the plan proposed; the ancestors are
  the same draw, not the same sequence. The other three resamplers give Sequential's ancestors.
- **Some loops in the table run sequentially whatever the policy:** a mixture's reductions over
  its components (moments, effective sample size, log-density), which rarely reach one chunk;
  `GridGaussianParams`'s removal of dead cells; and the prunes and Subdivide, which take a policy only so a
  trigger can pass one on.
- **The tests' tolerances.** A reduction under `par` differs from Sequential's by some ulps of
  the sum: `tests/parallel.cpp` allows $10^{-12}$ on totals, moments, and evidences of order one,
  and $10^{-9}$ on a fitted mixture, whose EM iterations amplify the moments' rounding.

### Random numbers in parallel

Two phases, and the first stands alone.

**Hoist the draws** (done in step 18). `RbParticleFilter::predict()` already
filled a matrix of standard normals, one column per particle, before its loop.
`ParticleFilter::moveParticles()` now does the same: each particle's column holds its noise's
rows and then each uncertain input's, filled particle by particle in the order the old loop drew
them, so the body is a pure function of the particle and its column and the caller's
`std::mt19937` and the rest of the API are unchanged. The draws stay
sequential: at $10^6$ particles in six dimensions that is $6 \times 10^6$ normal draws per step,
a tenth of a second or so, which by Amdahl's law caps the speedup of a parallel prediction at a
few times however many cores there are. Enough for a first step, not for a GPU.

**A counter-based generator.** Philox (Salmon, Moraes, Dror, and Shaw, 2011, "Parallel random
numbers: as easy as 1, 2, 3") makes draw $j$ of particle $i$ at step $k$ a pure function
$\Phi_{\text{seed}}(k, i, j)$ with no state to share, so a parallel loop draws for its own index,
the result is bitwise independent of the thread count, and a GPU produces the same numbers as
the CPU. C++26 adds it as `std::philox4x32` (feature-test macro `__cpp_lib_philox_engine`). The
library writes no engine of its own [random-draws], so on the C++23 toolchains the
engine is the caller's. A concept `CounterBasedGenerator` names what the parallel overloads
need: a generator that yields an independent stream at `(step, index)`. `std::philox4x32` meets
it where the standard library has it; until then a caller brings a counter-based engine, or the
parallel overloads keep drawing before the loop. The sequential overloads keep accepting any
`std::uniform_random_bit_generator`.

`std::normal_distribution` is stateful and implementation-defined (libstdc++ and MSVC give
different numbers for the same bits). The parallel path still uses it, one per index, so a
result is the same at every thread count within one standard library, and differs between
standard libraries as the sequential path's does. The standard distributions are not callable
on a device, which leaves the device's draws an open question below.

The generator's place in the argument list (particle-filter.md Q3) doesn't change: first
among the member function's arguments, after the policy.

## GPU execution

The user's four questions, then the architecture.

### Can we use Eigen on a GPU?

Yes, for fixed-size objects inside kernels, which is the only kind the library's bounded spaces
produce. Eigen's page "Using Eigen in CUDA kernels" states the scope: fixed-size matrices,
vectors, and arrays, with most Core functions marked `__host__ __device__` when the compiler is
nvcc or hipcc, meant for "numerous but small problems". Eigen 5.0 unifies CUDA and HIP behind
one GPU backend (`EIGEN_GPUCC`); its SYCL backend serves the Tensor module only. It also warns
that host SIMD is off in a `.cu` file, and that a device build should set
`EIGEN_DEFAULT_DENSE_INDEX_TYPE` to `int` throughout.

What the Eigen 5.0.0 tree we fetch marks device-callable, by counting `EIGEN_DEVICE_FUNC`:
products, triangular and self-adjoint views, triangular solves, and fixed-size `inverse()` are
annotated; `LLT.h` has no annotation at all, `LDLT.h` only its `rows()` and `cols()`, and
`AutoDiffScalar.h` none. So on a device the library cannot call `Eigen::LLT`, which every
density evaluation and Kalman update does, and cannot run a `Linearized` method on autodiff.
The consequences:

- A device-callable unblocked Cholesky, forward and back substitution, and log-determinant for
  fixed-size matrices go into `Core/` behind a `EMRYS_FILTER_DEVICE` annotation. Twenty lines,
  exact for $d \le 16$ or so, and a candidate for "Upstream to Eigen" in the roadmap, since
  Eigen's lack of a device `LLT` is the gap.
- On a device the methods are `Affine`, `Unscented`, and `Linearized` with an analytic
  Jacobian; autodiff stays on the host. Dynamic-size and unbounded spaces stay on the host,
  which [allocation] already says about their allocation.
- Every library function a kernel body reaches carries `EMRYS_FILTER_DEVICE`, which expands to
  `__host__ __device__` under `EIGEN_GPUCC` and to nothing otherwise: the spaces' `plus()`,
  `minus()`, and `normalize()`, `Moments`, the Gaussian density, the process models'
  `propagate()`, the sensor's `measure()`, and the Kalman update. Mechanical but
  cross-cutting, and a CI job can hold the line (below). The user's model and measurement must
  be device-callable too, which for a lambda means nvcc's `--extended-lambda` or clang.
- `std::expected`, `std::span`, `std::tuple`, and `std::optional` are header-only types and compile
  in device code (nvcc wants `--expt-relaxed-constexpr`); `EMRYS_FILTER_ASSERT` maps to the device
  `assert`. The library's no-exceptions, no-`std::function` rules [embedded-builds] are the device
  rules already.

Compiler: CUDA 13's language-support page lists `--std=c++23`, but its nvcc option tables
disagree from one 13.x release to the next, and nvcc with Visual Studio is still listed as
unsupported for Eigen. Clang compiles CUDA, HIP, and SYCL (through DPC++ or AdaptiveCpp) in
C++23, and `clang++ -x cuda --cuda-device-only -fsyntax-only` checks that a kernel instantiation
compiles with no GPU present, which is how CI holds the annotation line. nvcc is a target to
test against, not the one the design leans on.

### Write kernels ourselves, or use a GPU linear algebra package?

Neither, in the usual sense of either. The library's device workload is $N$ independent small
dense problems with $d \lesssim 12$ and $m \lesssim 6$, not one large matrix. cuBLAS, cuSOLVER,
rocSOLVER, and oneMKL are built for the large matrix; their batched small-matrix routines
(`potrfBatched`, `gemmBatched`, MAGMA's and Kokkos Kernels' batched sets) are closer, but each
launch performs one operation across the batch, so a Kalman update becomes ten launches with
every intermediate in global memory. One fused kernel in which each thread holds its particle's
matrices in registers does the update in one pass, and that fused kernel is what fixed-size Eigen
in device code gives us: the existing per-element bodies, compiled for the device. The library
writes no kernels in the CUDA sense; it writes the per-index body once, which it has, and a
backend launches it. Batched BLAS becomes worth revisiting only for a state too large for
registers, say $d > 32$, which is outside the bounded-space design anyway.

The question is then which runtime launches the body and supplies the five primitives:

| Runtime | Primitives it gives | Reach | Fit |
| --- | --- | --- | --- |
| C++17 parallel algorithms offloaded: nvc++ `-stdpar=gpu` | the standard ones, over unified memory, no annotations | NVIDIA | a zero-code-change experiment once the policy argument exists; NVHPC 26.3 also ships a draft C++26 `std::execution` |
| Thrust / rocThrust | `for_each`, `transform_reduce`, `inclusive_scan`, `gather`, `copy_if`, with `thrust::device` and `universal_allocator` | NVIDIA, AMD | the same shapes as our primitives and the standard's signatures; ships with CUDA and ROCm |
| oneDPL device policies over SYCL | the standard algorithms over a `sycl::queue`, `usm_allocator` | Intel, and NVIDIA and AMD through DPC++ or AdaptiveCpp; runs on a CPU device in CI | vendor-neutral, same signatures again |
| Kokkos | its own views, `parallel_for`, `parallel_reduce`, batched kernels | all | a second storage model beside the Allocator design; heavy for a header-only partner to Eigen |
| Raw CUDA / HIP / SYCL kernels | everything, by hand | one vendor each | only where a fused step beats the primitives measurably |

**Recommendation:** define the five primitives on our own policy types, so that `Sequential` and
the standard policies work now, and implement the first `Device` policy over Thrust, since it
mirrors the standard algorithms, covers two vendors, and brings the memory side with it. Before
that, the nvc++ `-stdpar=gpu` experiment measures what a GPU gives for the ranging benchmark
with no backend written at all; the measurement decides whether step 19 is worth its cost.
oneDPL is the second backend, for vendor neutrality and for running the device code path on a
CPU device in CI, where no GPU exists. A fused kernel is a later optimization, taken only where
a profile shows the launch count dominating.

### Reductions on the device, without moving the particles back

Yes, entirely. After a prediction every count-sized step in the table is one of the five
primitives, and Thrust, CUB, and oneDPL run all five on the device:

- **Reweighting** is a for-each (add the log-likelihoods), a reduce (the log-sum-exp pair), and a
  for-each (subtract the evidence). The evidence is one scalar in device memory; the
  normalization kernel reads it there.
- **ESS** is one reduce; the **mean and covariance** are a reduce with a fixed-size vector, then
  one with a fixed-size matrix, as the value type, with Eigen's `operator+` as the combine. The
  two-pass tangent-space mean of a curved space is two such reduces.
- **Resampling** by the systematic, stratified, or multinomial scheme is an inclusive scan of
  the weights, a binary search per output position (`upper_bound`, vectorized), and a gather of
  the ancestors' columns into the scratch buffer, which is then swapped in as today. No sort.
- **Compaction** (`removeIf`, dead cells) is a scan of the keep flags and a scatter.

What crosses to the host is the handful of scalars a decision needs: the log-likelihood and
`accepted` the update returns, and the ESS that `WhenEss` compares, a few bytes per step. The
estimate comes back as $d + d^2$ scalars when asked. The particles cross only when the caller
reads them.

Murray, Lee, and Jacob (2016) analyze exactly this step. The standard
resamplers need a collective (the scan), which is the one synchronization in an otherwise
embarrassingly parallel step; their Metropolis and rejection resamplers draw each ancestor
independently from the weights, with no collective, and run faster on a GPU. They also show that
in single precision the standard schemes are biased above a few hundred thousand particles while
the two alternatives are not, which matters because consumer GPUs run `double` at a thirty-second
of `float` rate and the device path will mostly be `float`. Both fit the `Resampler` concept as
written (weights, ancestors, generator) and need a per-ancestor random stream, which the
counter-based generator supplies; they join `Core/Resampling.h` in step 18b so that the CPU path
has them first.

### Memory

The caller chooses device memory by choosing the allocator, through the parameter [allocation] put
on every container:

```cpp
using DeviceParticles = ParticleSet<State, float, thrust::universal_allocator<std::byte>>;
ParticleFilter filter{model, DeviceParticles{space, states, weights}};
filter.predict(Emrys::Device{stream}, rng, dt);
```

CUDA managed memory (`universal_allocator`) and SYCL shared USM (`usm_allocator<...,
alloc::shared>`) are host-accessible, so `internal::Buffer`, a `std::vector` over the re-bound
allocator, works unchanged: `std::vector` constructs and moves its elements through host
pointers, and the `Eigen::Map` accessors keep working on the host, at the cost of page migration
if the host touches a buffer the device just wrote. The filter's count-sized scratch comes from
the same allocator and so is device memory too, with nothing added. A backend's own scratch (CUB
reduction storage) is sized at the container's capacity once, as [allocation] wants.

Device-only memory (`cudaMalloc`, `alloc::device`) is faster and never migrates, but
`std::vector` cannot hold it and a host-side `Map` over it would be a footgun that the current
accessors offer freely. That path needs `internal::Buffer` to become a customization point
(`BufferFor<Allocator, T>`, defaulting to `std::vector`) and an explicit `copyToHost()`. It is a
later optimization; the first device path requires shared memory, where every existing accessor
stays correct.

### Where the GPU pays, and where it doesn't

Each primitive is a kernel launch of several microseconds, and a particle filter step is six to
ten of them, so a step costs on the order of a hundred microseconds before any particle moves.
Against the CPU's parallel path the device wins from roughly $10^4$ to $10^5$ particles upward,
earlier for an expensive model, and never for a mixture filter with a hundred components. The
Rao-Blackwellized particle filter, with a Kalman update per particle, is the best case: more
work per element, the same launch count. The grid filters are next: a regular grid's prediction
is a convolution, the most GPU-shaped operation there is, though the library's point-mass filter
moves Gaussians through the model rather than convolving probabilities. The benchmark of step 19
measures the crossover rather than guessing it.

A longer-term direction, not a plan: C++26's `std::execution` (senders and receivers) expresses
a filter step as a pipeline whose stages a scheduler may fuse, which is how a prediction and the
update that follows it would become one launch without a `predictAndUpdate()` in the API. NVHPC
26.3 ships a draft implementation; the library can look again when two toolchains do.

## What to find out first

Cheap experiments that would change the plan, in the order to run them:

1. Compile one particle filter instantiation with `clang++ -x cuda --cuda-device-only
   -fsyntax-only -std=c++23` after a first pass of `EMRYS_FILTER_DEVICE` annotations, to learn
   what else in the reachable surface the device rejects (standard headers, `std::expected`,
   the asserts, the `Reads` machinery of `MeasurementReads.h`).
2. Build the ranging benchmark's particle filter with nvc++ `-stdpar=gpu` once the policy
   argument exists, and time it against `std::execution::par` on the host at $10^3$ to $10^6$
   particles. This is the crossover measurement with no backend written.
3. Check the three CI toolchains' parallel algorithms: libstdc++ needs TBB linked for `par`
   (`libtbb-dev` in CI), MSVC's are built in, and libc++'s coverage has to be checked at the
   LLVM version CI uses, since the search above found no current status page.
4. Confirm which Eigen 5.0.x point release CI will have and recount the device annotations
   there; an `LLT` annotation upstream would delete a piece of step 19.

## Steps

Roadmap steps 18 (execution policies), 18b (counter-based random numbers), and 19 (GPU backend) hold
the checklists and the "done when" criteria. Step 18's decisions are [execution-policies]. When step
18b starts: the caller's counter-based generator, `std::philox4x32` where it exists, with the
standard distributions, and no engine or transform of the library's own. When step 19 starts: Thrust
as the first device backend, shared memory as its precondition, and the device-callable Cholesky in
`Core/`. The bibliography gains Salmon et al. (2011) and Murray, Lee, and Jacob (2016) when the
first comment cites them.

## Open questions

- Should `Sequential` have a sibling `Parallel{minimumCount}` that runs `par` above a count and
  sequentially below it, so that a caller who doesn't know the count ahead of time can pass one
  policy? A reduction below one chunk already runs sequentially; a for-each over a few mixture
  components does not.
- `Roughen` after a parallel resample needs the counter-based generator's step counter advanced
  in step with the filter's; who owns the step count, the filter or the generator?
- Where does the device path get its normal draws, when the standard distributions are not
  callable on a device and the library writes no transform of its own? Drawing on the host
  before the loop is the fallback, at the Amdahl cost above.
- The device path's `Scalar` is `float`. Is the pair log-sum-exp in `float` over $10^7$ particles
  accurate enough, or does the evidence want a `double` accumulator while the weights stay
  `float`?
