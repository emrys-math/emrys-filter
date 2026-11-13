#pragma once

/** \file
 * SampleSet: a matrix whose columns are samples in a space, which the samplers return and the
 * unscented transform uses for its sigma points, and SampleSetMap and ConstSampleSetMap, the
 * same layout over memory the matrix does not own, which ParticleSet and the other containers
 * return over their allocators' buffers [allocation]. Read Gaussian.h next.
 */

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Distributions
 * A matrix whose columns are samples in \a Space (see \ref Spaces), with the space's size
 * and storage bound. It is a plain Eigen matrix, so the space is not part of its type and
 * the samples are not normalized by it.
 *
 * \tparam Space   space of the samples; plain vectors of dimension \c N are \c Linear<N>
 * \tparam Scalar  scalar type, \c double by default
 *
 * The number of samples is not bounded, so a SampleSet allocates its storage from Eigen's heap,
 * which takes no allocator. Where that storage must come from an allocator or a buffer of one's
 * own, as on a system with no heap, use a SampleSetMap over it: the containers keep their samples
 * that way and return them as maps [allocation], and drawSample() takes one sample at a time.
 *
 * Eigen requires a single-row matrix to be row-major, so one-dimensional sample sets are
 * stored row-major; all others are column-major.
 */
template <typename Space, typename Scalar = double>
using SampleSet = Eigen::Matrix<
    Scalar,
    valueDimensionAtCompileTime<Space>,
    Eigen::Dynamic,
    (maxValueDimensionAtCompileTime<Space> == 1 ? Eigen::RowMajor : Eigen::ColMajor),
    maxValueDimensionAtCompileTime<Space>,
    Eigen::Dynamic>;

/** \ingroup Distributions
 * A writable view of samples in \a Space, one per column, over memory it does not own, laid out
 * as a SampleSet is: what to wrap a buffer from an allocator in, so that nothing is taken from the
 * heap.
 *
 * \tparam Space   space of the samples
 * \tparam Scalar  scalar type, \c double by default
 */
template <typename Space, typename Scalar = double>
using SampleSetMap = Eigen::Map<SampleSet<Space, Scalar>>;

/** \ingroup Distributions
 * A read-only view of samples in \a Space, one per column, over memory it does not own, laid out
 * as a SampleSet is: what the containers return for their states, means, and centers.
 *
 * \tparam Space   space of the samples
 * \tparam Scalar  scalar type, \c double by default
 */
template <typename Space, typename Scalar = double>
using ConstSampleSetMap = Eigen::Map<SampleSet<Space, Scalar> const>;

}  // namespace Emrys
