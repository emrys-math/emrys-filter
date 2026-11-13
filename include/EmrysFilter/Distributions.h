#pragma once

/** \file
 * Everything in the Distributions layer: the distributions, their parameters and densities, the
 * read-only views over them, their moments, samples, statistics, sampling, fitting,
 * upkeep, and the divergences between them.
 */

/** \defgroup Distributions Distributions
 *
 * The third layer (decisions.md [layered-headers]): the distributions, each stored as parameters and evaluated
 * through a density, and what is computed from them.
 *
 * \anchor ParamsAndDensities
 * <b>Parameters and densities.</b> A distribution whose values can fail to define it, such as a
 * %Gaussian whose covariance is not positive definite, has two types [params-and-density]: its parameters, which
 * hold the values and change step by step, and its density, which evaluates it. The split exists
 * so that the expensive work is done once and the cheap work many times. Evaluating a %Gaussian's
 * density needs a Cholesky factorization of its covariance, whose cost is cubic in the dimension,
 * while each evaluation after that is a triangular solve, whose cost is quadratic. A filter
 * changes a covariance once in a step but may evaluate it many times in that step: a particle
 * filter's update evaluates one measurement noise at the residual of every particle, and a
 * %Gaussian mixture evaluates each of its components at every value. So GaussianParams holds the
 * mean and covariance and caches their factorization; toDensity() factorizes only if the
 * covariance has changed since the last factorization, and returns a Gaussian, which borrows that
 * factorization and copies nothing:
 *
 * \code
 * auto const cov = Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}};
 * auto noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), cov};
 * auto const density = noise.toDensity();  // factorizes, or fails with a GaussianError
 * if (!density)
 * {
 *     return;
 * }
 * for (Eigen::Index j = 0; j < residuals.cols(); ++j)
 * {
 *     logLikelihoods(j) = density->logDensity(residuals.col(j));  // a triangular solve each
 * }
 * noise.setCovariance(widened);  // density must not be read again; the next toDensity() factorizes
 * \endcode
 *
 * The split also puts the check of the values in one place. Parameters that fail to define a
 * distribution fail in toDensity(), once, as a \c std::expected, so logDensity() returns a plain
 * scalar and a loop over particles or components has no error to check. A density writes
 * nothing, so any number of threads may evaluate one at once (see below). In exchange, a density
 * must not outlive its parameters, and reading it after they have changed is a programmer error
 * that \c EMRYS_FILTER_ASSERT catches [view-stamp].
 *
 * Two cases the check can't see, since the stamp it reads is gone. Hold a density by value: the
 * \c std::expected that toDensity() returns is a temporary, and a reference into it dangles at
 * the end of the statement. And a density of a component inside a container, a mixture's or a
 * hybrid's, must not be read after the container grows, which may move the component it
 * borrows; only AddressSanitizer catches either.
 *
 * \code
 * auto const density = noise.toDensity().value();          // a Gaussian, held by value
 * auto const & dangling = noise.toDensity().value();       // dangles at the semicolon
 * auto const first = mixture[0].component.toDensity().value();
 * mixture.add(0.0, moments);  // may move the components; first must not be read again
 * \endcode
 *
 * The split also lets values be written one at a time where the check is of all of them
 * together. A categorical distribution's weights must not all be zero, and no single write makes
 * or breaks that: filling a histogram, setting a weight by name, or ruling a value out passes
 * through states that would fail. CategoricalParams takes those writes unchecked, and its
 * toDensity() checks the weights once and returns the Categorical that reads the probabilities.
 *
 * A distribution whose values are always valid, such as Uniform, Gamma, StudentT, or Product,
 * has nothing to cache, so it is one type: it checks its values when it is built, as
 * StudentT::create() does, and is its own parameters and its own density. Params<D> and
 * Density<P> (Params.h) map each type to the other, and a type of the first kind to itself.
 * Filters, mixtures, and hybrids are named by the density and store its parameters:
 * <tt>KalmanFilter<Gaussian<State>, Model></tt> holds a GaussianParams, which each step changes
 * in place, and its state's toDensity() gives the Gaussian. Generic code is written against the
 * density, since every distribution has one, and the HasLogDensity concept (Product.h) says what
 * a continuous density can do.
 *
 * A statistic that reads the components is the density's: a mixture's or a hybrid's moments and
 * collapse, and a Rao-Blackwellized state's collapse. It needs no factorization to compute, but
 * only the factorization checks that each component is a distribution, so on the parameters it
 * could describe none. One that reads only the weights, such as MixtureParams::ess(), is on the
 * parameters too, a \c std::expected of the weights' error (decisions.md [params-and-density]).
 *
 * <b>The headers.</b> Params.h pairs parameters with their densities. Moments.h holds the mean and
 * covariance that every distribution is matched to, SampleSet.h a matrix of samples, and Roots.h
 * the square roots of a covariance by the axes and repairs of \c Emrys::roots. The
 * Gaussian (Gaussian.h) comes first, with its draws (Sampling.h) and its square-root and
 * information forms SqrtGaussian, InfoGaussian, and SqrtInfoGaussian, which GaussianForm.h ties
 * together [gaussian-forms]. The distributions that are their own parameters are Uniform, Gamma, StudentT,
 * and Product. CategoricalParams and Categorical are the distribution over a discrete space [discrete-spaces], and
 * ContinuousTimeMarkovChain.h moves one over an interval [emissions]. The containers of weighted
 * elements share the statistics in Statistics.h and, where the elements are Gaussians, the
 * Component concept (Component.h, [component-first]): GaussianMixture (Mixture.h), with uniformComponents()
 * giving, one at a time, the components that approximate a Uniform, or the Uniform factors of a
 * Product, to fill one (UniformMixture.h, [uniform-mixtures]),
 * HybridGaussian (Hybrid.h, [imm]), and
 * ParticleSet with its KernelDensity. Fitting.h fits a mixture by expectation-maximization;
 * Upkeep.h holds the actions a caller applies to a state between steps [upkeep]; Divergence.h
 * compares two distributions [divergences]; and Conditionals.h, GridGaussian.h, and ParticleGaussian.h are
 * the Rao-Blackwellized states [rao-blackwellized-core], [rb-particle-filter]. Read Params.h first, and \ref Models
 * next.
 *
 * <b>Sharing between threads.</b> Parameters are for changing step by step, not for sharing. Their
 * \c const members fill the cache that evaluation needs, so \c const does not make them safe to
 * read from several threads at once [execution-policies]: factorize(), toDensity(), and the accessors that need a
 * factorization, such as GaussianParams::precision(), write the cache the first time after a
 * change, and the square-root and information forms' toDensity() writes it each time. What threads
 * should share is what toDensity() returns: a density writes nothing, so any number of threads may
 * evaluate one at once, while its parameters stay alive and unchanged and nothing calls their
 * \c const members that write the cache. Code that needs only the mean and covariance shares a copy
 * of the Moments instead, which owns its values and needs neither.
 *
 * <b>What a view's check catches.</b> A view's check of its parameters detects a change after the
 * fact and prevents nothing: it blocks no mutator and makes nothing wait. It records the
 * parameters' modification stamp when the view is made, and reading the view after the parameters
 * have changed is a programmer error that \c EMRYS_FILTER_ASSERT catches. The stamp is a plain
 * counter, so changing parameters on one thread while another reads a view of them is a data race
 * that no check catches.
 *
 * The views differ in what they tolerate. Parameters keep two counters: every mutator advances
 * the first, and a change of size or structure (a resize, an insertion or removal, a resample, an
 * assignment, a move from) advances the second too. A density reads the cached factorization or
 * bandwidth, so it checks the first and fails after any change. A marginal of parameters reads
 * their values by indices resolved when it was made, so it checks only the second: it sees new
 * values, and fails after a change of size or structure. A marginal of a view holds a copy of the
 * view and fails as the view would. An element of a mixture's or a hybrid's marginal reads its
 * component through the range's indices, so it must not outlive the range. A copy of the
 * parameters is new parameters, and changing it leaves the views of the original alone.
 *
 * <b>Marginals and conditionals.</b> Every continuous distribution has marginal<Path>() and
 * conditional<GivenPath>(values) by tag path [tag-addressing]. A marginal reads its source wherever
 * it can without computing, and copy() gives parameters [element-marginals]: the %Gaussian's is a
 * GaussianMarginal view, and a mixture's, a hybrid's, or a particle set's a range over the elements'
 * marginals. A marginal that must compute is a value: the square-root and information forms',
 * a categorical's over an axis, and those of the distributions that are their own parameters. A
 * conditional is always a value. One whose type has weights (a mixture's, a particle set's, a
 * categorical's) leaves them unnormalized, so its total weight is the source's times the density of
 * the given values; a single %Gaussian returns that evidence from conditionalWithEvidence()
 * [conditional-evidence]. A KernelDensity only evaluates: condition the ParticleSet, whose
 * conditional stores the exact conditional bandwidth.
 *
 * <b>Random draws.</b> A density draws, and parameters don't: drawSamples() and drawSample(), with
 * their WithLogDensity forms, take the density and the caller's generator [random-draws]. The
 * draws go through \c std::normal_distribution and its siblings, whose algorithms the standard
 * leaves to the implementation, so a seed gives the same draws under one standard library and
 * other draws under another. drawSample() takes its own draws from the generator at each call, so a
 * stream of single draws is not the sequence drawSamples() gives from the same seed.
 */

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Conditionals.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Divergence.h>
#include <EmrysFilter/Distributions/Fitting.h>
#include <EmrysFilter/Distributions/Gamma.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/KernelDensity.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Distributions/Roots.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Distributions/StudentT.h>
#include <EmrysFilter/Distributions/Uniform.h>
#include <EmrysFilter/Distributions/UniformMixture.h>
