#pragma once

// The ranging problem solved with FrancoisCarouge/Kalman (https://github.com/FrancoisCarouge/Kalman),
// another implementation of [benchmarks]: a C++ library whose filter type is deduced from the
// characteristics it is declared with, over a linear algebra backend its user supplies. The pair
// uses the Eigen backend of its support/eigen directory, the one its README points an Eigen user
// to. carouge_ranging.cpp declares the filter and says what it computes.
//
// The library lives in that translation unit alone, and this header declares the filter without it,
// because the backend declares, in namespace Eigen, an operator/ for any two Eigen objects, which
// solves a linear system. Argument-dependent lookup then finds it for every division of one Eigen
// array by another in a translation unit that includes the backend, Eigen's own included, and
// emrys_ranging_kinds.h's Jacobian, which divides the anchors' offsets rowwise by their distances,
// no longer compiles beside it. So each step of this filter is one call into that translation unit
// for the prediction and one for the update, which the compiler cannot inline;
// docs/design/benchmarks.md gives what that costs. Its state lives on the heap, allocated when it is
// constructed, never in a step. Compare emrys_ranging_kinds.h's EKF.

#include "scenario.h"

#include <Eigen/Core>

#include <memory>

namespace Ranging
{

/// The ranging EKF with FrancoisCarouge/Kalman in \a Scalar, double or float, with the analytic
/// Jacobian, for the dimensions and anchor counts that carouge_ranging.cpp instantiates: 2D with
/// 4, 8, and 16 anchors, and 3D with 6, 12, and 24.
template <int Dimension, int AnchorCount, typename Scalar = double>
class CarougeRanging
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    static constexpr int StateSize = ScenarioType::StateSize;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;
    using State = Eigen::Matrix<Scalar, StateSize, 1>;
    using Covariance = Eigen::Matrix<Scalar, StateSize, StateSize>;

    /// The filter at \a scenario's initial estimate, with the transition over its fixed step.
    explicit CarougeRanging(ScenarioType const & scenario);
    CarougeRanging(CarougeRanging &&) noexcept;
    CarougeRanging &
    operator=(CarougeRanging &&) noexcept;
    ~CarougeRanging();

    /// Moves the estimate forward by the scenario's time step.
    void
    predict();

    /// Conditions the estimate on the ranges \a z, one per anchor.
    void
    update(Ranges const & z);

    /// The current mean, position then velocity.
    [[nodiscard]] State
    mean() const;

    /// The current covariance; its two triangles can differ by rounding.
    [[nodiscard]] Covariance
    covariance() const;

private:

    struct Filter;

    std::unique_ptr<Filter> m_filter;  ///< the library's filter, of a type only carouge_ranging.cpp names
};

}  // namespace Ranging
