#pragma once

/** \file
 * Everything in the Filters layer: the Kalman, %Gaussian mixture, interacting multiple model,
 * particle, Rao-Blackwellized, and hidden Markov model filters.
 */

/** \defgroup Filters Filters
 *
 * The fifth layer (decisions.md [layered-headers]): a filter owns its process model and a state, predicts with
 * `predict(dt, inputs...)`, and updates with `update(measurement)` [filters-and-recursions]. A filter is named by
 * the density of its state and holds the state's parameters, which each step changes in place
 * (\ref ParamsAndDensities "Parameters and densities"). KalmanFilter.h is the
 * shape the others follow, and over a form of the %Gaussian it is the filter that
 * InfoFilter.h, SqrtKalmanFilter.h, and SqrtInfoFilter.h spell [gaussian-forms];
 * MixtureFilter.h, ImmFilter.h, ParticleFilter.h, RbPointMassFilter.h, and RbParticleFilter.h
 * differ from it in their state, with RbUpdate.h shared by the last two, while HiddenMarkovFilter.h
 * has nothing left %Gaussian [emissions]. AmmFilter.h spells the IMM on the identity chain [autonomous-multiple-model].
 * Upkeep between steps is the caller's [upkeep]: a filter holds the upkeep it was given, FilterUpkeep.h's, and
 * runs it on its state when the caller calls maintain(). So is the record of the forward pass
 * that the histories of \ref Smoothers smooth [smoothing]. Recursion.h spells what KalmanFilter.h's and
 * MixtureFilter.h's filters ask of the recursion that runs their steps, so that one of the user's own runs inside them
 * [user-extension-points].
 *
 * Most of the filters share one structure, a discrete variable and a %Gaussian state conditioned
 * on it, and differ in what the discrete part is and how they handle it:
 *
 * | Filter | Discrete part | Handled by |
 * |---|---|---|
 * | Kalman filter with mixture noise | inlier or outlier, memoryless | enumeration, collapsed each step |
 * | %Gaussian mixture filter | component index, unlabeled | enumeration, reduced |
 * | IMM | maneuver regime, Markov with transition matrix \f$ \Pi \f$ | enumeration, mixed through \f$ \Pi \f$ |
 * | Rao-Blackwellized point mass | fixed point of one block | enumeration, no mixing |
 * | Rao-Blackwellized particle | a sampled block | sampling |
 * | Hidden Markov model | the whole state, Markov through \f$ \Pi \f$ or a generator | enumeration, nothing %Gaussian |
 *
 * Memoryless mixture noise is the IMM's case where every row of \f$ \Pi \f$ is the same, and a
 * persistent outlier process (bursts, a stuck sensor) is an IMM on the measurement side: the
 * noise <tt>additiveNoise(mixture).byMode<Sensor>()</tt> (ModeNoise) names the mode axis whose
 * values its components are, which an IMM whose mode space has that axis updates by, and which a
 * filter with no mode space reads as memoryless mixture noise.
 *
 * A filter owns a state distribution and a recursion: a process model and a prediction method,
 * with any scratch the steps need, acting on a state passed in (decisions.md [filters-and-recursions]). It moves the
 * state forward with predict() and conditions it on measurements with update(), which takes any
 * measurement (see MeasurementModel): each measurement carries its own model, noise, and transform
 * method, so one filter can take a position fix by an affine update and a range-bearing
 * measurement by an unscented one. Different sensors are different measurement types; a stream
 * that mixes them dispatches with \c std::visit over the user's own variant.
 *
 * \code
 * auto filter = KalmanFilter{ConstantVelocity<Position, Velocity>{0.1}, initial};
 * filter.predict(dt);
 * auto const updated = filter.update(fix);     // GaussianParams noise: one branch
 * auto const robust = filter.update(robustFix);  // mixture noise, collapsed
 * auto const selected = filter.update(robustFix, reductions::select());  // mixture noise, the most probable kept
 * auto const gated = filter.update(fix, reductions::gate(0.997));  // rejected when its NIS is improbable
 * if (!gated) { ... }                          // S was not positive definite (decisions.md [innovation-cholesky])
 * if (!gated->accepted) { ... }
 * \endcode
 *
 * An update gives one branch per measurement-noise component (see MeasurementUpdate). A
 * filter whose state is poorer than that reduces them with a policy: Collapse moment-matches,
 * Select keeps the most probable, and Gate drops branches that fail a chi-squared test before
 * passing the rest to another policy. A policy returns the new state, or adds what it keeps to the
 * filter's output; keeping nothing rejects the measurement and keeps the prior (see
 * ReductionPolicy).
 *
 * Every filter is parameterized by its component first [component-first], [filters-and-recursions]: KalmanFilter over a
 * Gaussian, MixtureFilter over a Mixture of components, ImmFilter over a Hybrid of components. The recursion a
 * component's steps run is the IMM's for a Hybrid and the Kalman one for any other, so a MixtureFilter whose components
 * are hybrid states is a mixture of IMM filters.
 *
 * Every filter's update reports its outcome in one shape [update-outcomes]. A part of the state whose
 * branches the policy rejects gets probability zero; whether the zero is stored follows from the
 * state, and when every part is rejected the state keeps its prior:
 *
 * | Filter | A part rejected |
 * |---|---|
 * | KalmanFilter and the form filters | (one part) |
 * | GaussianMixtureFilter | the component is dropped |
 * | ImmFilter, AmmFilter | the mode keeps its prior %Gaussian with probability 0, which the chain can revive |
 * | RbPointMassFilter | the cell is removed, since nothing moves probability between cells |
 * | RbParticleFilter | the particle keeps its %Gaussian with weight 0, until resampled |
 * | ParticleFilter, HiddenMarkovFilter | (no policy); an observation nothing explains leaves the weights |
 *
 * Under the AMM's identity chain a mode of probability zero can't come back, which is what FloorModes is for
 * [autonomous-multiple-model].
 *
 * Features that only watch the filter stay outside it: a NisMonitor is fed the result of each
 * update, so a filter without one pays nothing.
 *
 * Every filter here is an implementation of one published in the literature, cited in its own
 * documentation; \ref Sarkka2013 "Särkkä (2013)" covers the Kalman family and the particle
 * filter in one notation, and \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)" the
 * tracking practice around them.
 *
 * Read \ref Smoothers next.
 */

#include <EmrysFilter/Filters/AmmFilter.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/HiddenMarkovFilter.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Filters/RbUpdate.h>
#include <EmrysFilter/Filters/RbParticleFilter.h>
#include <EmrysFilter/Filters/RbPointMassFilter.h>
