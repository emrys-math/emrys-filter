# Header map

The headers live in `include/EmrysFilter/` in six layers. A header includes only its own layer
and the ones before it, and CI checks that.

| Layer | Holds | Headers |
|---|---|---|
| `Core/` | What everything else builds on | `Version.h`, `Macros.h` (assertions, no exceptions), `Math.h`, `Allocation.h`, `Execution.h` (policies), `Forwarding.h`, `Time.h`, `Stamp.h`, `Print.h`, `NumericError.h` (why a step fails), `Resampling.h`, `Weights.h`, `ChiSquared.h`, `Triangularization.h`, `Multiply.h`, `Cholesky.h` (the factorization the gains come from), `Solve.h` |
| `Spaces/` | What a state is made of | `Space.h` (the concept, `plus`, `minus`), `Coordinates.h` (`Linear`, `Periodic`), `Composite.h` (`Composite`, `Repeated`), `TagPath.h`, `Selection.h` (several repetitions of a repeated block), `Subspace.h`, `Indexing.h`, `Discrete.h`, `Reset.h`, `UnitQuaternion.h` |
| `Distributions/` | Probability over a space | `Params.h` (parameters and densities), `Component.h`, `Gaussian.h` and its forms (`SqrtGaussian.h`, `InfoGaussian.h`, `SqrtInfoGaussian.h`, `GaussianForm.h`), `Roots.h` (covariance roots and repairs), `Mixture.h`, `UniformMixture.h`, `SampleSet.h`, `ParticleSet.h`, `KernelDensity.h`, `Hybrid.h`, `Categorical.h`, `ContinuousTimeMarkovChain.h`, `Conditionals.h`, `GridGaussian.h`, `ParticleGaussian.h`, `StudentT.h`, `Gamma.h`, `Uniform.h`, `Product.h`, `Moments.h`, `Statistics.h`, `Sampling.h`, `Fitting.h`, `Divergence.h`, `Upkeep.h` |
| `Models/` | How states move and are observed | `Policies.h` (what a model declares), `ProcessModel.h`, `MeasurementModel.h`, `AffineConstants.h` (the constant part of `affineIn()`), `Placement.h` (placing and composing models), `ResolvedReads.h`, `Measurement.h` (the update), `MeasurementUpdate.h`, `MeasurementReads.h`, `IteratedUpdate.h`, `Likelihood.h`, `Transform.h`, `Methods.h` (the methods and sigma-point sets), `Batched.h` (functions that take every point at once), `PointOutput.h` (functions that write their own Jacobians), `TransformArgs.h`, `Linearization.h`, `Condition.h` (the Kalman update), `ByMode.h` (parts per value of a mode axis), `SwitchingModel.h`, `Emission.h`, `Simulation.h`, `Consistency.h`, `CramerRao.h`, `Reduction.h` (policies), `FormSteps.h` |
| `Filters/` | The recursions | `Recursion.h` (the contract), `FilterUpkeep.h` (the upkeep a filter holds), `KalmanFilter.h`, `SqrtKalmanFilter.h`, `InfoFilter.h`, `SqrtInfoFilter.h`, `MixtureFilter.h`, `ImmFilter.h`, `AmmFilter.h`, `ParticleFilter.h`, `RbUpdate.h`, `RbParticleFilter.h`, `RbPointMassFilter.h`, `HiddenMarkovFilter.h` |
| `Smoothers/` | Estimates of past states from later data | `Lag.h`, `GaussianHistory.h` (with `smoothStep()`), `HybridGaussianHistory.h`, `GaussianMixtureHistory.h`, `ParticleHistory.h`, `CategoricalHistory.h` |

Include `<EmrysFilter/EmrysFilter.h>` for everything, a layer's umbrella header, such as
`<EmrysFilter/Filters.h>`, or one header. [The reference](https://emrys-math.github.io/emrys-filter/)
documents each header, and its module pages are the place to start reading a layer.
