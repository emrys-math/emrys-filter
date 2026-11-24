#pragma once

// The ranging problem solved from scratch in plain Eigen, as a careful user would write it [benchmarks]:
// fixed sizes, noalias() products, only the position columns of the covariance in the update,
// fixed-size derivative vectors for autodiff, S formed by accumulating H Pxy into a copy of R as
// the library forms it rather than into a zeroed matrix, and the library's covariance update form
// and factorization (P - K Pxy^T on a covariance mirrored from its lower triangle, with a Cholesky
// factorization of S without square roots, by whole columns, and the gain by substitution), so the two agree to
// rounding, in double or in float. Without the mirror the two triangles drift
// apart by rounding, which in float costs a tenth of the filter's accuracy. With
// Factorization::EigenLlt it factorizes S by Eigen's LLT and solves for the gain by its solve()
// instead, as a user of Eigen writes it, and agrees with the library to rounding all the same.
// Compare emrys_ranging.h.

#include "scenario.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace Ranging
{

/// The ranging EKF in plain Eigen in \a Scalar, linearizing the ranges as \a Kind says and
/// factorizing the innovation covariance as \a F says.
template <
    int Dimension,
    int AnchorCount,
    Jacobian Kind,
    typename Scalar = double,
    Factorization F = Factorization::WrittenOut>
class EigenRanging
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    static constexpr int StateSize = ScenarioType::StateSize;
    using StateVector = Eigen::Matrix<Scalar, StateSize, 1>;
    using StateMatrix = Eigen::Matrix<Scalar, StateSize, StateSize>;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;

    explicit EigenRanging(ScenarioType const & scenario)
        : m_anchors{scenario.anchors.template cast<Scalar>()}
        , m_rangeNoise{Scalar(scenario.rangeStdDev * scenario.rangeStdDev) * RangeMatrix::Identity()}
        , m_timeStep{Scalar(scenario.timeStep)}
        , m_transition{transitionOver(m_timeStep)}
        , m_processNoise{processNoiseOver(m_timeStep, Scalar(scenario.accelDensity))}
        , m_mean{scenario.initialMean.template cast<Scalar>()}
        , m_cov{scenario.initialCov.template cast<Scalar>()}
    {}

    /// Moves the estimate forward by the scenario's time step: x = F x, P = F P F^T + Q, with F and
    /// Q built once at construction, since the step is fixed.
    void
    predict()
    {
        m_mean.template head<Dimension>() += m_timeStep * m_mean.template tail<Dimension>();
        auto propagated = StateMatrix{};
        propagated.noalias() = m_transition * m_cov;
        m_cov.noalias() = propagated * m_transition.transpose();
        m_cov += m_processNoise;
    }

    /// Conditions the estimate on the ranges \a z, one per anchor.
    void
    update(Ranges const & z)
    {
        // The residual z - h(x) and the Jacobian of h in position; h does not depend on velocity.
        auto residual = Ranges{};
        auto h = Eigen::Matrix<Scalar, AnchorCount, Dimension>{};
        linearize(z, residual, h);

        // Pxy = P H^T, where H is h's Jacobian padded with zero velocity columns.
        auto cross = Eigen::Matrix<Scalar, StateSize, AnchorCount>{};
        cross.noalias() = m_cov.template leftCols<Dimension>() * h.transpose();
        RangeMatrix innovationCov = m_rangeNoise;
        innovationCov.noalias() += h * cross.template topRows<Dimension>();

        auto gain = Eigen::Matrix<Scalar, StateSize, AnchorCount>{};
        if constexpr (F == Factorization::EigenLlt)
        {
            // The gain Pxy S^-1 as a user of Eigen writes it: S^-1 Pxy^T by Eigen's LLT, transposed.
            gain = Eigen::LLT<RangeMatrix>{innovationCov}.solve(cross.transpose()).transpose();
        }
        else
        {
            // The Cholesky factorization of S without square roots, S = L D L^T with L unit lower
            // triangular, a column at a time, as the library takes it
            // (CholeskyFactorization), since Eigen's LLT of a matrix this small takes
            // several times as long, and since each step of the filter waits on this one: column k of
            // L D is column k of S less L(k, j) times each column j before it, and the column after
            // it waits on one division, where with square roots it would wait on a root and then a
            // division. Each step is one operation on a whole column, so what it leaves above the
            // diagonal is never read. The columns are padded with zeros to a multiple of 16 bytes,
            // Eigen's smallest packet: it vectorizes a column of six floats from wherever its address
            // lets a packet start, and one column's stores are then not the next one's loads.
            constexpr int Lanes = int(16 / sizeof(Scalar));
            constexpr int PaddedAnchors = (AnchorCount + Lanes - 1) / Lanes * Lanes;
            using FactorColumn = Eigen::Matrix<Scalar, PaddedAnchors, 1>;
            auto unscaled = Eigen::Matrix<Scalar, PaddedAnchors, AnchorCount>{};
            auto unit = Eigen::Matrix<Scalar, PaddedAnchors, AnchorCount>{};
            auto reciprocal = Ranges{};
            for (int k = 0; k < AnchorCount; ++k)
            {
                FactorColumn column = FactorColumn::Zero();
                column.template head<AnchorCount>() = innovationCov.col(k);
                for (int j = 0; j < k; ++j)
                {
                    column -= (unscaled(k, j) * reciprocal(j)) * unscaled.col(j);
                }
                reciprocal(k) = 1 / column(k);
                unscaled.col(k) = column;
                unit.col(k) = column * reciprocal(k);
            }

            // The gain Pxy S^-1 by substitution, as the library solves it: forward through L, a
            // multiplication by the reciprocals of the pivots, and backward through L^T. Column r of
            // the gain holds unknown r of every row, so each step is one operation on whole columns,
            // padded as the factor's are.
            constexpr int Padded = (StateSize + Lanes - 1) / Lanes * Lanes;
            using PaddedColumn = Eigen::Matrix<Scalar, Padded, 1>;
            auto work = Eigen::Matrix<Scalar, Padded, AnchorCount>{};
            for (int r = 0; r < AnchorCount; ++r)
            {
                PaddedColumn unknown = PaddedColumn::Zero();
                unknown.template head<StateSize>() = cross.col(r);
                for (int c = 0; c < r; ++c)
                {
                    unknown -= unit(r, c) * work.col(c);
                }
                work.col(r) = unknown;
            }
            for (int r = AnchorCount - 1; r >= 0; --r)
            {
                PaddedColumn unknown = work.col(r) * reciprocal(r);
                for (int c = r + 1; c < AnchorCount; ++c)
                {
                    unknown -= unit(c, r) * work.col(c);
                }
                work.col(r) = unknown;
            }
            gain = work.template topRows<StateSize>();
        }
        m_mean.noalias() += gain * residual;
        m_cov.template triangularView<Eigen::StrictlyUpper>() = m_cov.transpose();
        m_cov.noalias() -= gain * cross.transpose();
    }

    /// The current mean, position then velocity.
    [[nodiscard]] StateVector const &
    mean() const
    {
        return m_mean;
    }

    /// The current covariance.
    [[nodiscard]] StateMatrix const &
    covarianceLower() const
    {
        return m_cov;
    }

private:

    using PositionVector = Eigen::Matrix<Scalar, Dimension, 1>;
    using PositionMatrix = Eigen::Matrix<Scalar, Dimension, Dimension>;
    using RangeMatrix = Eigen::Matrix<Scalar, AnchorCount, AnchorCount>;

    /// The constant-velocity transition F over the step \a dt.
    [[nodiscard]] static StateMatrix
    transitionOver(Scalar dt)
    {
        StateMatrix transition = StateMatrix::Identity();
        transition.template topRightCorner<Dimension, Dimension>() = dt * PositionMatrix::Identity();
        return transition;
    }

    /// The process noise Q over the step \a dt, the exact discretization of white-noise
    /// acceleration of spectral density \a accelDensity.
    [[nodiscard]] static StateMatrix
    processNoiseOver(Scalar dt, Scalar accelDensity)
    {
        PositionMatrix const identity = PositionMatrix::Identity();
        auto processNoise = StateMatrix{};
        processNoise << dt * dt * dt / 3 * identity, dt * dt / 2 * identity, dt * dt / 2 * identity, dt * identity;
        return accelDensity * processNoise;
    }

    /// Writes the residual z - h(x) at the mean to \a residual and h's Jacobian in position to \a
    /// h.
    void
    linearize(Ranges const & z, Ranges & residual, Eigen::Matrix<Scalar, AnchorCount, Dimension> & h) const
    {
        PositionVector const p = m_mean.template head<Dimension>();
        if constexpr (Kind == Jacobian::Analytic)
        {
            // h's Jacobian is each anchor's unit line of sight to p: minus its direction from p.
            Eigen::Matrix<Scalar, Dimension, AnchorCount> const toAnchor = m_anchors.colwise() - p;
            Eigen::Matrix<Scalar, 1, AnchorCount> const ranges = toAnchor.colwise().norm();
            residual = z - ranges.transpose();
            h = -(toAnchor.array().rowwise() / ranges.array()).matrix().transpose();
        }
        else
        {
            // Forward mode over position only, with a fixed-size derivative vector, so nothing
            // allocates.
            using Dual = Eigen::AutoDiffScalar<PositionVector>;
            auto pDual = Eigen::Matrix<Dual, Dimension, 1>{};
            for (int j = 0; j < Dimension; ++j)
            {
                pDual(j) = Dual{p(j), Dimension, j};
            }
            Eigen::Matrix<Dual, 1, AnchorCount> const ranges =
                (m_anchors.template cast<Dual>().colwise() - pDual).colwise().norm();
            for (int i = 0; i < AnchorCount; ++i)
            {
                residual(i) = z(i) - ranges(i).value();
                h.row(i) = ranges(i).derivatives().transpose();
            }
        }
    }

    Eigen::Matrix<Scalar, Dimension, AnchorCount> m_anchors;  ///< one anchor per column
    RangeMatrix m_rangeNoise;                                 ///< the ranges' noise covariance, R
    Scalar m_timeStep;                                        ///< the fixed time between measurements
    StateMatrix m_transition;                                 ///< F over the step
    StateMatrix m_processNoise;                               ///< Q over the step
    StateVector m_mean;                                       ///< the estimate's mean
    StateMatrix m_cov;                                        ///< the estimate's covariance
};

}  // namespace Ranging
