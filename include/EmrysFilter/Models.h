#pragma once

/** \file
 * Everything in the Models layer: transforms of a distribution through a function, process and
 * measurement models, mode-switching models, the branch reduction policies, the consistency
 * statistics, and simulation.
 */

/** \defgroup Models Models
 *
 * The fourth layer (decisions.md [layered-headers]): how a distribution moves through a function and what a
 * filter is given to work with. A model's noise is parameters, which a step reads through their
 * density (\ref ParamsAndDensities "Parameters and densities"). Transform.h propagates a
 * distribution with the methods of Methods.h, over the arguments of TransformArgs.h, into the
 * results of Linearization.h, which Condition.h conditions on an observation; ProcessModel.h and
 * Measurement.h, with the measurement models of MeasurementModel.h, the policies of Policies.h, and the placements
 * and compositions of Placement.h, are the models [model-policies], [model-placement], [affine-coordinates],
 * [templated-measurements], [method-per-measurement], [additive-noise], [sensors], [continuous-time-noise],
 * [weighted-branches], and [prediction-inputs]; SwitchingModel.h adds a mode chain [switching-models], [imm];
 * Emission.h measures a discrete state [emissions]; Reduction.h and Consistency.h reduce an update's branches and test
 * them [weighted-branches], [reduction-policies], [composed-features], and Simulation.h draws the true states that
 * Consistency.h's nees() tests a filter against, and CramerRao.h's posterior Cramér-Rao bound bounds its error along
 * them [cramer-rao-bound]; FormSteps.h holds the steps of the square-root and information forms of the %Gaussian
 * [gaussian-forms]; and Batched.h marks a function that takes every point at once [batched-functions].
 *
 * A process model moves the whole state forward in time; a measurement model relates part
 * of the state to an observation. They are separate kinds of object: a filter owns its
 * process model, and a measurement arrives carrying its own model and noise.
 *
 * Both kinds of model are functions with reads and writes (decisions.md [model-policies]): a
 * function <tt>f(in, out, x[, dt])</tt>, <tt>f(in, x[, dt])</tt>, or <tt>f(x[, dt])</tt> over Eigen
 * vectors, and policies given as calls in any order. A model without reads() and writes() reads and
 * writes the whole state (decisions.md [whole-state-models]), so the textbook Kalman filter on
 * <tt>Linear<4></tt> is <tt>processModel(F, additiveNoise(q))</tt> and
 * <tt>measurementModel(H, additiveNoise(r))</tt>. A process model is processModel(), a built-in model, or compose() of
 * process models; a measurement model is measurementModel(). The built-in process models, RandomWalk, Constant,
 * ConstantVelocity and ConstantAcceleration (Kinematic), and OrnsteinUhlenbeck, take their tags
 * as template arguments and the scalar of their parameters last, \c double by default:
 *
 * \code
 * struct Position : Linear<2> {};
 * struct Velocity : Linear<2> {};
 * struct Acceleration : Linear<2> {};
 * struct Bias : Linear<1> {};
 * using State = Composite<Position, Velocity, Acceleration, Bias>;
 *
 * auto const model = compose(ConstantAcceleration<Position, Velocity, Acceleration>{0.1},
 *                            OrnsteinUhlenbeck<Bias>{TimeConstant{60.0}, StationaryVariance{0.25}});
 * auto const predicted = predict(methods::linearized(), state, model, dt);
 * \endcode
 *
 * A model of the user's own reads the blocks it names with <tt>block<Name>(in, x)</tt>, writes the
 * blocks it names into a vector in \c out, and names the inputs of the call it reads by tag; the
 * call gives each as an uncertain input, a %Gaussian over the tag's space whose uncertainty goes
 * through the dynamics, or as <tt>sure<Tag>(value)</tt>, a value known exactly:
 *
 * \code
 * // Velocity driven by a measured specific force plus known gravity, rather than modeled.
 * struct SpecificForce : Linear<2> {};
 * struct Gravity : Linear<2> {};
 * auto const driven = processModel(
 *     []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
 *       auto y = zeros<T>(out);
 *       auto const a = block<SpecificForce>(in, x) + block<Gravity>(in, x);
 *       block<Position>(out, y) = block<Position>(in, x) + dt * block<Velocity>(in, x) + (dt * dt / 2) * a;
 *       block<Velocity>(out, y) = block<Velocity>(in, x) + dt * a;
 *       return y;
 *     },
 *     reads(path<Position>, path<Velocity>, path<SpecificForce>, path<Gravity>),
 *     writes(path<Position>, path<Velocity>));
 * auto const predicted = predict(methods::unscented(), state, compose(driven, RandomWalk<Bias>{1e-8}), dt,
 *                                accelerometer, sure<Gravity>(g));
 * \endcode
 *
 * The built-in models take continuous-time parameters and discretize them exactly over the step
 * \c dt, as \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)" discretize the
 * continuous-time kinematic models; a model of the user's own gives continuousNoise(), a spectral
 * density over the blocks it writes that the step discretizes as \f$ Q_c \, \Delta t \f$, or
 * additiveNoise(), a fixed \f$ Q \f$, or neither where its noise comes through an input. The models
 * of a composition must write every coordinate of the state exactly once, so a coordinate that does
 * not move still needs a Constant or RandomWalk model; the check is at compile time where the space
 * is fixed-size. The composition's noise is block-diagonal.
 *
 * \c dt reaches a model's function in the filter's scalar, converted once from whatever TimeStep
 * the caller passed (see \ref Time), so a \c float filter's models compute in \c float. It is in the
 * unit of the models' continuous-time parameters, seconds when it was a \c std::chrono duration. The
 * function is generic in the scalar of \c x, which the Linearized method differentiates
 * automatically, or returns its Jacobian with its value (withJacobian(), [returned-jacobians]), as
 * the built-in models do, so that Linearized predicts them with no differentiation.
 *
 * A model reads its blocks by name wherever they sit in the state, so one model serves every place
 * that has them: on<Ownship>(vehicle) places a model on a block, prefixing every path into the state
 * it reads and writes, onEach<Fleet>(vehicle) places one copy on every repetition of a repeated
 * block, and compose() of measurement models measures their blocks side by side (Placement.h). A
 * model that names the inputs it is affine in with affineIn() has them carried exactly by every
 * sigma-point method, which spreads its points over the rest [affine-coordinates].
 *
 * A measurement model takes one of three forms, from the convenient to the general: a model built
 * once, a function with a fixed output space or one computed from the state's, such as CountFrom,
 * or a matrix, or Columns that repeat with a block, which serves every count of the state's blocks;
 * a model built per measurement, its function, space, and noise computed from that measurement's
 * own data, such as the receivers that heard a signal or noise whose variances follow each
 * reading's signal-to-noise ratio; and a model known only by its likelihood,
 * <tt>measurementModel(reads(...), logLikelihood(g))</tt>, for the filters that weight points
 * [likelihood-measurements]. observe() makes a measurement of it, and update() transforms only the
 * blocks the model reads and corrects the rest of the state through their cross-covariance, which is
 * exact under a %Gaussian prior; it returns a branch per component of the additive noise, each with
 * its Innovation [weighted-branches], [innovation-statistics]. The filters that weight points take
 * \f$ \log p(z \mid x) \f$ from the same measurement, by logLikelihoods() (Likelihood.h).
 *
 * A function that costs more per call than per point, such as one that looks up a table, holds a
 * lock, or calls into another runtime or a device, can take every point at once: batched(f) marks it
 * as taking the points as the columns of a matrix \c X and writing their values into the output
 * \c Y the library provides, <tt>h(in, out, X, Y)</tt>, or for a process model
 * <tt>f(in, out, X, dt, Y)</tt>, with \c Y.values() a column per point (Batched.h,
 * [batched-functions]). Column \f$ j \f$ of what it writes depends only on column \f$ j \f$ of \c X.
 * The library calls it once per particle set or chunk and once per sigma-point set, and through
 * on(), compose(), and onEach() as through the model itself. A method that needs its Jacobian takes
 * it from the function, marked batched(f, jacobians::perColumn()) or batched(f, jacobians::shared())
 * and writing <tt>Y.jacobian<Name>(j)</tt> for the blocks <tt>Y.wantsJacobian<Name>()</tt> names,
 * or differentiates it in dual numbers, for which the function takes \c X as <tt>auto const &</tt>:
 *
 * \code
 * // The terrain's height below the aircraft, looked up for every particle in one pass.
 * auto const elevation = measurementModel(
 *     batched([&terrain](auto const & in, auto const &, auto const & X, auto & Y) {
 *       auto const position = block<Position>(in, X);
 *       for (Eigen::Index j = 0; j < X.cols(); ++j)
 *         Y.values()(0, j) = terrain.height(position.col(j));
 *     }),
 *     reads(path<Position>), additiveNoise(elevationNoise));
 * \endcode
 *
 * The program examples/terrain_lookup.cpp filters such a measurement with a particle filter.
 *
 * Read \ref Filters next.
 */

#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Condition.h>
#include <EmrysFilter/Models/Consistency.h>
#include <EmrysFilter/Models/CramerRao.h>
#include <EmrysFilter/Models/Emission.h>
#include <EmrysFilter/Models/FormSteps.h>
#include <EmrysFilter/Models/IteratedUpdate.h>
#include <EmrysFilter/Models/Likelihood.h>
#include <EmrysFilter/Models/Linearization.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/MeasurementReads.h>
#include <EmrysFilter/Models/MeasurementUpdate.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/Simulation.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Models/TransformArgs.h>
