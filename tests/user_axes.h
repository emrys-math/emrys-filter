#pragma once

// Axes and a repair of the user's own, for tests/Distributions/roots.cpp, the transforms that take a method's
// axes, and the repairs of a density (tests/Distributions/gaussian.cpp) and of an innovation covariance
// (tests/Filters/update_failure.cpp): reversed Cholesky axes, the upper-triangular root U with U U^T = P that a
// Cholesky factorization of P with its coordinates in reverse order gives, so that the last coordinate's column varies
// alone; and an eigenvalue floor, which raises every eigenvalue of a covariance to a floor before the root is retried.
// Each meets its concept by its members alone, with nothing of the library's specialized, and neither takes a sink,
// so each reports roots::NoReport [user-extension-points]. usage.md, "Writing axes", shows the first.

#include <EmrysFilter/Distributions/Roots.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <expected>

namespace EmrysTest
{

// U with U U^T = P and U upper triangular: J L J for the Cholesky factor L of J P J, J the reversal.
struct ReversedCholesky
{
    static constexpr bool canFail = true;

    template <typename Derived>
    [[nodiscard]] std::expected<Emrys::roots::RootMatrix<Derived>, Emrys::GaussianError>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        using Matrix = Emrys::roots::RootMatrix<Derived>;
        Matrix const full = cov.template selfadjointView<Eigen::Lower>();
        Matrix const reversed = full.reverse();
        auto const llt = Eigen::LLT<Matrix>{reversed};
        if (llt.info() != Eigen::Success)
        {
            return std::unexpected{Emrys::GaussianError::NotPositiveDefinite};
        }
        Matrix const lower = llt.matrixL();
        return Matrix{lower.reverse()};
    }
};

// The covariance with every eigenvalue raised to at least Floor.
template <double Floor>
struct EigenvalueFloor
{
    template <typename Derived>
    [[nodiscard]] std::expected<typename Derived::PlainObject, Emrys::GaussianError>
    repaired(Eigen::MatrixBase<Derived> const & cov) const
    {
        using Matrix = typename Derived::PlainObject;
        auto const solver = Eigen::SelfAdjointEigenSolver<Matrix>{Matrix{cov}};
        auto const values = solver.eigenvalues().cwiseMax(typename Derived::Scalar(Floor)).eval();
        return Matrix{solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose()};
    }
};

}  // namespace EmrysTest
