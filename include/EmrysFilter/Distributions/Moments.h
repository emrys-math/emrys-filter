#pragma once

/** \file
 * Moments: the mean and covariance that a %Gaussian carries and that a mixture or a sample set
 * is moment-matched to, with MomentsError and the ScalarOf alias. The mean is a value in the
 * space and the covariance is over its differences (\ref Spaces). Read SampleSet.h next.
 */

#include <type_traits>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Distributions
 * The scalar type of the Eigen object or distribution \a X, for writing functions generic
 * in it.
 */
template <typename X>
using ScalarOf = typename std::remove_cvref_t<X>::Scalar;

/** \ingroup Distributions
 * Reasons a distribution's moments() can fail.
 */
enum class MomentsError
{
    Infinite,  ///< the distribution has no finite mean or covariance, such as a Student-t with \f$ \nu \le 2 \f$
};

/** \ingroup Distributions
 * \brief First and second moments (mean and covariance) of a distribution.
 *
 * The mean is a value in the space, with valueDimension() coordinates, and the covariance is
 * over its differences, with differenceDimension() rows and columns (see \ref Spaces). See
 * [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix).
 *
 * \tparam Space_   space of the mean (see \ref Spaces), which sets the dimensions and the
 *                  storage bounds. The space is part of the type only: none is stored,
 *                  and the mean is not normalized.
 * \tparam Scalar_  scalar type, \c double by default
 *
 * No invariants are enforced: the covariance is not required to be symmetric or
 * positive definite. Like Eigen matrices, fixed-size Moments are uninitialized
 * by default. The covariance is read through covarianceLower(), the name every type with a
 * stored covariance gives it, since what reads it reads only the lower triangle.
 */
template <Space Space_, RealScalar Scalar_ = double>
class Moments
{
public:

    using Scalar = Scalar_;      ///< the scalar type
    using Index = Eigen::Index;  ///< Eigen's index type
    using SpaceType = Space_;    ///< the space the mean lives in

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = valueDimensionAtCompileTime<Space_>;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = maxValueDimensionAtCompileTime<Space_>;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = valueDimensionAtCompileTime<DifferenceSpace<Space_>>;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = maxValueDimensionAtCompileTime<DifferenceSpace<Space_>>;

    using VectorType = ValueVector<Space_, Scalar>;                 ///< a value, such as the mean
    using DifferenceVectorType = DifferenceVector<Space_, Scalar>;  ///< a difference, such as a residual
    using MatrixType = DifferenceMatrix<Space_, Space_, Scalar>;  ///< a matrix over differences, such as the covariance

    Moments() = default;

    /** Constructs moments with \a valueDimension coordinates in the mean and \a differenceDimension
     * in the covariance, with uninitialized contents.
     */
    Moments(Index valueDimension, Index differenceDimension)
    {
        resize(valueDimension, differenceDimension);
    }

    /** Constructs moments sized for \a space, with uninitialized contents. */
    explicit Moments(Space_ const & space)
        : Moments{Emrys::valueDimension(space), Emrys::differenceDimension(space)}
    {}

    /** Constructs moments from \a mean and \a cov, moving either one passed as an rvalue of the
     * stored type. \a cov may be a diagonal, its variances.
     */
    template <MatrixExpression Mean, CovarianceExpression Cov>
    Moments(Mean && mean, Cov && cov)
        : m_mean{std::forward<Mean>(mean)}
        , m_cov{std::forward<Cov>(cov)}
    {
        EMRYS_FILTER_ASSERT(m_cov.rows() == m_cov.cols() && "the covariance must be square");
        EMRYS_FILTER_ASSERT(
            (!internal::hasEqualSizes<Space_> || m_cov.rows() == m_mean.size())
            && "the covariance must have a row per coordinate of the mean"
        );
    }

    /** \returns the number of coordinates of the mean */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_mean.size();
    }

    /** \returns the number of rows and columns of the covariance */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return m_cov.rows();
    }

    /** Resizes the mean to \a valueDimension coordinates and the covariance to \a differenceDimension
     * rows and columns. Contents are unspecified afterward.
     */
    void
    resize(Index valueDimension, Index differenceDimension)
    {
        m_mean.resize(valueDimension);
        m_cov.resize(differenceDimension, differenceDimension);
    }

    /** Resizes the mean and covariance for \a space. Contents are unspecified afterward. */
    void
    resize(Space_ const & space)
    {
        resize(Emrys::valueDimension(space), Emrys::differenceDimension(space));
    }

    /** \returns the mean */
    [[nodiscard]] VectorType const &
    mean() const
    {
        return m_mean;
    }

    /** \returns the covariance, over differences */
    [[nodiscard]] MatrixType const &
    covarianceLower() const
    {
        return m_cov;
    }

    /** \returns the mean, to change it */
    [[nodiscard]] VectorType &
    mean()
    {
        return m_mean;
    }

    /** \returns the covariance, to change it */
    [[nodiscard]] MatrixType &
    covarianceLower()
    {
        return m_cov;
    }

    /** \returns the entries of the mean in the block named by the path \a Path (a tag or a
     * TagPath): a lazy expression, or a coefficient for a single coordinate. Moments store no
     * space, so the path resolves at compile time, as idx() does: no resizable block may precede it.
     */
    template <typename Path>
    [[nodiscard]] auto
    mean() const
    {
        return m_mean(idx<Space_, Path>());
    }

    /** \returns the covariance of the block named by \a Path (see mean<Path>()), at its rows of
     * differences (see diffIdx()). Unlike GaussianParams::covariance<Path>(), it reads the entries as
     * stored, since Moments needn't be symmetric.
     */
    template <typename Path>
    [[nodiscard]] auto
    covarianceLower() const
    {
        return m_cov(diffIdx<Space_, Path>(), diffIdx<Space_, Path>());
    }

    /** \returns the cross-covariance of the blocks named by \a RowPath and \a ColPath, as stored. */
    template <typename RowPath, typename ColPath>
    [[nodiscard]] auto
    covarianceLower() const
    {
        return m_cov(diffIdx<Space_, RowPath>(), diffIdx<Space_, ColPath>());
    }

    /** \returns the writable entries of the mean in the block named by \a Path; see mean<Path>(). */
    template <typename Path>
    [[nodiscard]] decltype(auto)
    mean()
    {
        return m_mean(idx<Space_, Path>());
    }

    /** \returns the writable covariance of the block named by \a Path; see covarianceLower<Path>(). */
    template <typename Path>
    [[nodiscard]] decltype(auto)
    covarianceLower()
    {
        return m_cov(diffIdx<Space_, Path>(), diffIdx<Space_, Path>());
    }

private:

    VectorType m_mean;  ///< the mean
    MatrixType m_cov;   ///< the covariance
};

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes a MomentsError in words. */
inline void
describe(std::ostream & stream, MomentsError error, Indent indent)
{
    stream << indent;
    switch (error)
    {
        case MomentsError::Infinite:
            stream << "the moments are infinite";
            break;
    }
}

/** Describes \a moments: their dimension, mean, and covariance. */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, Moments<Space_, Scalar_> const & moments, Indent indent)
{
    stream << indent << "moments";
    printDimension(stream, moments.valueDimension(), moments.differenceDimension());
    printMoments(stream, indent.deeper(), moments);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO
