// The ranging EKF of carouge_ranging.h with FrancoisCarouge/Kalman, in a translation unit of its own
// (the header says why). Declared with an output model H and an observation h but no transition
// function, the library's filter is its extended filter with a fixed transition: predict() forms
// F x and F P F^T + Q, and update() forms S = H P H^T + R, the gain P H^T / S, and the Joseph form
// (I - K H) P (I - K H)^T + K R K^T, its default. The backend's division solves S^T K^T = H P by a
// Householder QR decomposition with full pivoting, and no policy mirrors the covariance, since none
// is declared by default. Like mherb_ranging.h, the pair keeps the library's arithmetic as its
// users get it. The transition is set once, after construction, by f(). There is no automatic
// differentiation, so the Jacobian is analytic only.
//
// The process noise Q is declared by its rows, as the library's samples declare a matrix, not as
// an Eigen matrix: the library tells a matrix Q from a function of the state by whether it can be
// called with the state, and an Eigen matrix can, by the operator() that indexes it by an array of
// indices, so an Eigen matrix Q matches no extended filter without a transition function.

#include "carouge_ranging.h"

#include "scenario.h"

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <utility>

#include <fcarouge/kalman.hpp>
#include <fcarouge/linalg.hpp>

namespace Ranging
{

namespace
{

/// The ranges h(x) to the anchors, the library's observation.
template <int Dimension, int AnchorCount, typename Scalar>
struct RangeObservation
{
    using State = Eigen::Matrix<Scalar, 2 * Dimension, 1>;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;

    Eigen::Matrix<Scalar, Dimension, AnchorCount> anchors;  ///< one anchor per column

    [[nodiscard]] Ranges
    operator()(State const & x) const
    {
        return (anchors.colwise() - x.template head<Dimension>()).colwise().norm().transpose();
    }
};

/// The Jacobian H of the ranges at x, the library's output model: each row is the unit direction
/// from its anchor to the position, and the velocity columns are zero.
template <int Dimension, int AnchorCount, typename Scalar>
struct RangeJacobian
{
    using State = Eigen::Matrix<Scalar, 2 * Dimension, 1>;
    using Matrix = Eigen::Matrix<Scalar, AnchorCount, 2 * Dimension>;

    Eigen::Matrix<Scalar, Dimension, AnchorCount> anchors;  ///< one anchor per column

    [[nodiscard]] Matrix
    operator()(State const & x) const
    {
        Matrix h = Matrix::Zero();
        for (int i = 0; i < AnchorCount; ++i)
        {
            Eigen::Matrix<Scalar, Dimension, 1> const d = x.template head<Dimension>() - anchors.col(i);
            h.row(i).template head<Dimension>() = d.transpose() / d.norm();
        }
        return h;
    }
};

/// The rows of a square matrix of size N, as the library's declarations take a matrix.
template <typename Scalar, std::size_t N>
struct Rows
{
    Scalar values[N][N];
};

/// The process noise of \a rows, declared by its rows.
template <typename Scalar, std::size_t N, std::size_t... Row>
[[nodiscard]] auto
processUncertainty(Rows<Scalar, N> const & rows, std::index_sequence<Row...>)
{
    return fcarouge::process_uncertainty{rows.values[Row]...};
}

/// The library's filter at \a scenario's initial estimate in \a Scalar, before its transition is
/// set.
template <int Dimension, int AnchorCount, typename Scalar>
[[nodiscard]] auto
makeFilter(Scenario<Dimension, AnchorCount> const & scenario)
{
    constexpr int N = 2 * Dimension;
    using State = Eigen::Matrix<Scalar, N, 1>;
    using Covariance = Eigen::Matrix<Scalar, N, N>;
    using RangeMatrix = Eigen::Matrix<Scalar, AnchorCount, AnchorCount>;
    Eigen::Matrix<Scalar, Dimension, AnchorCount> const anchors = scenario.anchors.template cast<Scalar>();
    Scalar const dt = Scalar(scenario.timeStep);
    Scalar const q = Scalar(scenario.accelDensity);
    auto noise = Rows<Scalar, std::size_t(N)>{};
    for (std::size_t i = 0; i < std::size_t(Dimension); ++i)
    {
        std::size_t const p = i;                           // the position coordinate
        std::size_t const v = std::size_t(Dimension) + i;  // its velocity
        noise.values[p][p] = q * dt * dt * dt / 3;
        noise.values[p][v] = q * dt * dt / 2;
        noise.values[v][p] = q * dt * dt / 2;
        noise.values[v][v] = q * dt;
    }
    Scalar const variance = Scalar(scenario.rangeStdDev * scenario.rangeStdDev);
    return fcarouge::kalman{
        fcarouge::state{State{scenario.initialMean.template cast<Scalar>()}},
        fcarouge::output<Eigen::Matrix<Scalar, AnchorCount, 1>>,
        fcarouge::estimate_uncertainty{Covariance{scenario.initialCov.template cast<Scalar>()}},
        processUncertainty(noise, std::make_index_sequence<std::size_t(N)>{}),
        fcarouge::output_uncertainty{RangeMatrix{variance * RangeMatrix::Identity()}},
        fcarouge::output_model{RangeJacobian<Dimension, AnchorCount, Scalar>{anchors}},
        fcarouge::observation{RangeObservation<Dimension, AnchorCount, Scalar>{anchors}},
    };
}

}  // namespace

template <int Dimension, int AnchorCount, typename Scalar>
struct CarougeRanging<Dimension, AnchorCount, Scalar>::Filter
{
    decltype(makeFilter<Dimension, AnchorCount, Scalar>(std::declval<ScenarioType const &>())) filter;
};

template <int Dimension, int AnchorCount, typename Scalar>
CarougeRanging<Dimension, AnchorCount, Scalar>::CarougeRanging(ScenarioType const & scenario)
    : m_filter{std::make_unique<Filter>(Filter{makeFilter<Dimension, AnchorCount, Scalar>(scenario)})}
{
    Covariance transition = Covariance::Identity();
    transition.template topRightCorner<Dimension, Dimension>().diagonal().setConstant(Scalar(scenario.timeStep));
    m_filter->filter.f(transition);
}

template <int Dimension, int AnchorCount, typename Scalar>
CarougeRanging<Dimension, AnchorCount, Scalar>::CarougeRanging(CarougeRanging &&) noexcept = default;

template <int Dimension, int AnchorCount, typename Scalar>
CarougeRanging<Dimension, AnchorCount, Scalar> &
CarougeRanging<Dimension, AnchorCount, Scalar>::operator=(CarougeRanging &&) noexcept = default;

template <int Dimension, int AnchorCount, typename Scalar>
CarougeRanging<Dimension, AnchorCount, Scalar>::~CarougeRanging() = default;

template <int Dimension, int AnchorCount, typename Scalar>
void
CarougeRanging<Dimension, AnchorCount, Scalar>::predict()
{
    m_filter->filter.predict();
}

template <int Dimension, int AnchorCount, typename Scalar>
void
CarougeRanging<Dimension, AnchorCount, Scalar>::update(Ranges const & z)
{
    m_filter->filter.update(z);
}

template <int Dimension, int AnchorCount, typename Scalar>
auto
CarougeRanging<Dimension, AnchorCount, Scalar>::mean() const -> State
{
    return m_filter->filter.x();
}

template <int Dimension, int AnchorCount, typename Scalar>
auto
CarougeRanging<Dimension, AnchorCount, Scalar>::covariance() const -> Covariance
{
    return m_filter->filter.p();
}

template class CarougeRanging<2, 4, double>;
template class CarougeRanging<2, 8, double>;
template class CarougeRanging<2, 16, double>;
template class CarougeRanging<3, 6, double>;
template class CarougeRanging<3, 12, double>;
template class CarougeRanging<3, 24, double>;
template class CarougeRanging<2, 4, float>;
template class CarougeRanging<2, 8, float>;
template class CarougeRanging<2, 16, float>;
template class CarougeRanging<3, 6, float>;
template class CarougeRanging<3, 12, float>;
template class CarougeRanging<3, 24, float>;

}  // namespace Ranging
