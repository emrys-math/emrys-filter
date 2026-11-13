#pragma once

/** \file
 * Everything in the Spaces layer: continuous spaces and their tag paths, the index utilities,
 * the unit quaternion, and discrete spaces.
 */

/** \defgroup Spaces Spaces
 *
 * The second layer (decisions.md [layered-headers]): where values live. A continuous space says how long its
 * values and differences are and how to add and subtract them (Space.h, [sizes-and-dimensions]). Its coordinate
 * types are Linear and Periodic (Coordinates.h, [no-bounded-coordinates]), Composite and Repeated build spaces from
 * spaces (Composite.h, [several-resizable-blocks]), tag paths name their blocks and idx() and block() index them
 * (TagPath.h, [index-names]), and Subspace.h builds the space a path names and grows and shrinks blocks; a discrete
 * space is a finite set of named values (Discrete.h, [discrete-spaces]); Reset.h carries a covariance to the tangent
 * space at a moved mean [covariance-reset]; UnitQuaternion.h is an attitude stored as Eigen stores a quaternion
 * [unit-quaternion]; and Indexing.h holds what marginals and conditionals by path are built from. Read Space.h first.
 *
 * A space describes how the coordinates of a vector behave: whether they wrap or are plain
 * linear values, and how many there are. Every
 * distribution type, and Moments and SampleSet, is parameterized by a space and then a
 * scalar, which defaults to \c double (<tt>GaussianParams<State></tt>), and takes its size and
 * storage bound from the space. A fixed-size space is an empty type and costs nothing. A
 * plain vector of dimension \c N is <tt>Linear<N></tt>, so <tt>GaussianParams<Linear<3>></tt> is a
 * %Gaussian on a 3-vector.
 *
 * Values and differences each live in a space. A space whose values need bringing into
 * canonical form has a normalize() member, applied to blocks whose rows are its coordinates
 * and whose columns are samples. Its differences live in its difference space, from
 * difference(), which is another space: an angle in <tt>[0, 360)</tt> has differences in
 * <tt>[-180, 180)</tt>, and a Linear coordinate's differences are plain vectors. The library works
 * with them through four free functions, which do nothing where there is nothing to do:
 *  - plus() applies an update, \f$ x \boxplus \delta \f$, by default
 *    <tt>normalizeValues(space, x + δ)</tt>;
 *  - minus() takes a residual, \f$ a \boxminus b \f$, by default
 *    <tt>normalizeDifferences(space, a - b)</tt>;
 *  - normalizeValues() and normalizeDifferences() bring a value or a difference that came
 *    from outside, such as a user's mean or a model's output, into canonical form.
 * A space whose updates and residuals aren't addition and subtraction, such as latitude and
 * longitude with differences in meters, defines its own plus() and minus() members. A space is
 * linear when none of this does anything (isLinear<>).
 *
 * The coordinate types are Linear and Periodic. Their parameters are template arguments, so a
 * fixed-size coordinate type is always an empty type, and the common periods are named:
 * RadiansPi, Radians2Pi, Degrees180, and Degrees360. There is no bounded coordinate type [no-bounded-coordinates]:
 * a quantity that can't be negative is estimated as the vector it is the length of, or as a log,
 * and a distribution that has a bounded support, such as Gamma, states it itself.
 *
 * Spaces compose: Composite concatenates them, and Repeated repeats a fixed-size block
 * a runtime number of times. Named subspaces are tag types derived from a space
 * (<tt>struct Heading : Degrees360<> {};</tt>).
 *
 * A space's valueDimension() is the number of coordinates of a value. Its differences may have
 * fewer, as a unit quaternion's four coordinates have three-coordinate rotation vectors as
 * differences, so a distribution has two dimensions: valueDimension(), the length of a mean or a
 * sample, and differenceDimension(), the rows of a covariance and the degrees of freedom. Generic
 * code calls the free functions valueDimension(space) and differenceDimension(space); a space
 * implements only the member valueDimension(), and its difference dimension is its difference
 * space's [sizes-and-dimensions]. A fixed-size space is an empty type whose value dimension is a compile-time
 * constant. A resizable space (a Repeated block, a coordinate type with \c Eigen::Dynamic
 * dimension, or a Composite with one) stores its value dimension. Any number of a Composite's
 * blocks may be resizable, at any depth of nested composites, such as a set of tracks and a set of
 * landmarks. Distributions keep their space's value dimension in step with their data, and grow or
 * shrink each resizable block in place with insert() and erase().
 *
 * A space with at most one resizable block can take its sizes from a length alone, so it has
 * resize(n), and a distribution can be built in it from a mean without a space. A space with
 * several can't: the same length splits among the blocks in more than one way. A distribution in
 * one is built from a space that carries its counts, and the space is grown block by block.
 *
 * idx<Path>(space) turns a path (a tag or a TagPath) into an Eigen index set from the
 * space's sizes. idx<S, Path>() gives the same index as a compile-time constant. Model code reads
 * and writes the block a path names with <tt>block<Path>(space, x)</tt> on a vector of values, or a
 * matrix whose columns are values, and <tt>diffBlock<Path>(space, d)</tt> on one of differences,
 * each Eigen's block with the rows the space's type fixes, and the sizes checked against the space;
 * <tt>diffBlock<Row, Column>(rows, columns, m)</tt> is a block of a covariance or a Jacobian, and
 * zeros() the vector such code fills block by block. Where a call takes paths as arguments, a
 * path is the value <tt>path<Ownship, Position></tt> (decisions.md [tag-addressing]).
 *
 * Every space exposes \c FixedValueDimension (the coordinates of its fixed part),
 * \c ValueDimensionStep (nonzero if it is resizable: the coordinates of one element of its
 * resizable part), and \c MaxValueDimension (the largest valid value dimension, or
 * \c Eigen::Dynamic if unbounded). Like Eigen's matrices, the coordinate types take a dimension
 * and a maximum dimension, and Repeated takes a maximum number of repetitions. Distributions use the space's \c
 * MaxValueDimension as their storage bound, so a capped space gets fixed-capacity storage.
 *
 * The \f$ \boxplus \f$ and \f$ \boxminus \f$ operators, and the idea of writing an estimator
 * once over any state space that provides them, are those of
 * \ref Hertzberg2013 "Hertzberg, Wagner, Frese, and Schröder (2013)"; a Periodic coordinate is
 * the circle \f$ S^1 \f$ of their examples. See also
 * [directional statistics](https://en.wikipedia.org/wiki/Directional_statistics) for why a mean
 * of angles is not the mean of their values.
 *
 * Read \ref Distributions next.
 */

#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>
