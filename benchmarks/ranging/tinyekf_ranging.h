#pragma once

// The ranging problem solved with TinyEKF (https://github.com/simondlevy/TinyEKF), a C EKF for
// microcontrollers and another implementation of [benchmarks], written as its own examples write
// it: the caller computes the predicted mean, the measurement prediction, and their Jacobians
// into arrays, row-major, and passes them with the noise covariances to ekf_predict() and
// ekf_update(). Like mherb_ranging.h, it keeps the library's own arithmetic, which is what its
// users run: the prediction forms F P F^T + Q, and the update forms the gain P H^T S^-1 through an
// explicit inverse of S by its Cholesky factor, then the covariance (I - G H) P, with no triangle
// mirrored; every product is a loop over the arrays. There is no automatic differentiation, so the
// Jacobian is analytic only. Compare emrys_ranging_kinds.h's EKF.
//
// TinyEKF's sizes and scalar are macros, fixed for the translation unit that includes it, so each
// size and scalar of the benchmark is an instance of its header in a namespace of its own
// (tinyekf_instance.h), and TinyEkf<N, M, Scalar> picks one. Its functions are static, so
// everything here that calls them is in an unnamed namespace too: the same definition in two
// translation units would otherwise name different functions.

#include "scenario.h"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdlib>

// TinyEKF's header includes these. Included here first, at global scope, they are skipped by their
// include guards when each instance includes them inside its namespace.
#include <math.h>
#include <stdbool.h>
#include <string.h>

namespace Ranging
{

// NOLINTNEXTLINE(cert-dcl59-cpp,google-build-namespaces): TinyEKF's functions are static (above)
namespace
{

/// TinyEKF with \a N states and \a M measurements in \a Scalar, one of the instances below.
template <int N, int M, typename Scalar>
struct TinyEkf;

/// The namespace of the instance with \a n states and \a m measurements in \a scalar.
#define TINYEKF_INSTANCE_NAMESPACE(n, m, scalar) TINYEKF_INSTANCE_NAMESPACE_EXPANDED(n, m, scalar)
#define TINYEKF_INSTANCE_NAMESPACE_EXPANDED(n, m, scalar) tinyekf_##n##_##m##_##scalar

// The planar cases: 4 states, and 4, 8, or 16 anchors.
#define TINYEKF_INSTANCE_N 4
#define TINYEKF_INSTANCE_M 4
#define TINYEKF_INSTANCE_SCALAR double
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 4
#define TINYEKF_INSTANCE_M 4
#define TINYEKF_INSTANCE_SCALAR float
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 4
#define TINYEKF_INSTANCE_M 8
#define TINYEKF_INSTANCE_SCALAR double
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 4
#define TINYEKF_INSTANCE_M 8
#define TINYEKF_INSTANCE_SCALAR float
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 4
#define TINYEKF_INSTANCE_M 16
#define TINYEKF_INSTANCE_SCALAR double
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 4
#define TINYEKF_INSTANCE_M 16
#define TINYEKF_INSTANCE_SCALAR float
#include "tinyekf_instance.h"

// The spatial cases: 6 states, and 6, 12, or 24 anchors.
#define TINYEKF_INSTANCE_N 6
#define TINYEKF_INSTANCE_M 6
#define TINYEKF_INSTANCE_SCALAR double
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 6
#define TINYEKF_INSTANCE_M 6
#define TINYEKF_INSTANCE_SCALAR float
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 6
#define TINYEKF_INSTANCE_M 12
#define TINYEKF_INSTANCE_SCALAR double
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 6
#define TINYEKF_INSTANCE_M 12
#define TINYEKF_INSTANCE_SCALAR float
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 6
#define TINYEKF_INSTANCE_M 24
#define TINYEKF_INSTANCE_SCALAR double
#include "tinyekf_instance.h"
#define TINYEKF_INSTANCE_N 6
#define TINYEKF_INSTANCE_M 24
#define TINYEKF_INSTANCE_SCALAR float
#include "tinyekf_instance.h"

#undef TINYEKF_INSTANCE_NAMESPACE
#undef TINYEKF_INSTANCE_NAMESPACE_EXPANDED

/// The ranging EKF with TinyEKF in \a Scalar, with the analytic Jacobian.
template <int Dimension, int AnchorCount, typename Scalar = double>
class TinyEkfRanging
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    static constexpr int StateSize = ScenarioType::StateSize;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;
    using State = Eigen::Matrix<Scalar, StateSize, 1>;
    using Covariance = Eigen::Matrix<Scalar, StateSize, StateSize>;

    /// The filter at \a scenario's initial estimate. The step is fixed, so F, Q, and R are set once
    /// here, and the velocity columns of H, which stay zero.
    explicit TinyEkfRanging(ScenarioType const & scenario)
        : m_timeStep{Scalar(scenario.timeStep)}
    {
        for (int i = 0; i < AnchorCount; ++i)
        {
            for (int j = 0; j < Dimension; ++j)
            {
                m_anchors[i * Dimension + j] = Scalar(scenario.anchors(j, i));
            }
        }
        Scalar const dt = m_timeStep;
        Scalar const q = Scalar(scenario.accelDensity);
        for (int i = 0; i < Dimension; ++i)
        {
            int const p = i;              // the position coordinate
            int const v = Dimension + i;  // its velocity
            m_F[p * StateSize + p] = 1;
            m_F[p * StateSize + v] = dt;
            m_F[v * StateSize + v] = 1;
            m_Q[p * StateSize + p] = q * dt * dt * dt / 3;
            m_Q[p * StateSize + v] = q * dt * dt / 2;
            m_Q[v * StateSize + p] = q * dt * dt / 2;
            m_Q[v * StateSize + v] = q * dt;
        }
        Scalar const rangeStdDev = Scalar(scenario.rangeStdDev);
        for (int i = 0; i < AnchorCount; ++i)
        {
            m_R[i * AnchorCount + i] = rangeStdDev * rangeStdDev;
        }
        for (int i = 0; i < StateSize; ++i)
        {
            m_ekf.x[i] = Scalar(scenario.initialMean(i));
            for (int j = 0; j < StateSize; ++j)
            {
                m_ekf.P[i * StateSize + j] = Scalar(scenario.initialCov(i, j));
            }
        }
    }

    /// Moves the estimate forward by the scenario's time step.
    void
    predict()
    {
        Scalar fx[std::size_t(StateSize)];
        for (int i = 0; i < Dimension; ++i)
        {
            fx[i] = m_ekf.x[i] + m_timeStep * m_ekf.x[Dimension + i];
            fx[Dimension + i] = m_ekf.x[Dimension + i];
        }
        Api::predict(m_ekf, fx, m_F, m_Q);
    }

    /// Conditions the estimate on the ranges \a z, one per anchor. The update fails only if the
    /// innovation covariance has no Cholesky factor, which a positive range noise rules out, so a
    /// failure stops the program.
    void
    update(Ranges const & z)
    {
        Scalar hx[std::size_t(AnchorCount)];
        for (int i = 0; i < AnchorCount; ++i)
        {
            Scalar squared = 0;
            for (int j = 0; j < Dimension; ++j)
            {
                Scalar const d = m_ekf.x[j] - m_anchors[i * Dimension + j];
                squared += d * d;
            }
            hx[i] = std::sqrt(squared);
            for (int j = 0; j < Dimension; ++j)
            {
                m_H[i * StateSize + j] = (m_ekf.x[j] - m_anchors[i * Dimension + j]) / hx[i];
            }
        }
        if (!Api::update(m_ekf, z.data(), hx, m_H, m_R))
        {
            std::abort();
        }
    }

    /// The current mean, position then velocity.
    [[nodiscard]] State
    mean() const
    {
        return Eigen::Map<State const>{m_ekf.x};
    }

    /// The current covariance; its two triangles can differ by rounding.
    [[nodiscard]] Covariance
    covariance() const
    {
        return Eigen::Map<Eigen::Matrix<Scalar, StateSize, StateSize, Eigen::RowMajor> const>{m_ekf.P};
    }

private:

    using Api = TinyEkf<StateSize, AnchorCount, Scalar>;

    Scalar m_timeStep;  ///< the fixed time between measurements
    // The extents are converted, since GCC warns of a sign conversion at an int one.
    Scalar m_anchors[std::size_t(AnchorCount * Dimension)]{};  ///< one anchor per row
    Scalar m_F[std::size_t(StateSize * StateSize)]{};          ///< the state transition, row-major
    Scalar m_Q[std::size_t(StateSize * StateSize)]{};          ///< the process noise covariance
    Scalar m_H[std::size_t(AnchorCount * StateSize)]{};        ///< the measurement Jacobian
    Scalar m_R[std::size_t(AnchorCount * AnchorCount)]{};      ///< the range noise covariance
    typename Api::Filter m_ekf{};                              ///< TinyEKF's mean and covariance
};

}  // namespace

}  // namespace Ranging
