// One instance of TinyEKF, included by tinyekf_ranging.h once per state size, measurement size, and
// scalar, and so without #pragma once. TinyEKF fixes its sizes and its scalar by the macros EKF_N,
// EKF_M, and _float_t, which its header reads, and has no include guard, so each instance is its
// header included again inside a namespace of its own, after these macros. The includer defines
// TINYEKF_INSTANCE_N, TINYEKF_INSTANCE_M, and TINYEKF_INSTANCE_SCALAR, inside the namespace that
// declares TinyEkf and with its TINYEKF_INSTANCE_NAMESPACE, and this file undefines the three. The
// functions are TinyEKF's own, unchanged, and TinyEkf<N, M, Scalar> names them.

#if !defined(TINYEKF_INSTANCE_N) || !defined(TINYEKF_INSTANCE_M) || !defined(TINYEKF_INSTANCE_SCALAR)
#    error "define TINYEKF_INSTANCE_N, TINYEKF_INSTANCE_M, and TINYEKF_INSTANCE_SCALAR before this file"
#endif

#define EKF_N TINYEKF_INSTANCE_N
#define EKF_M TINYEKF_INSTANCE_M
// TinyEKF's own name for its scalar, which its header reads.
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#define _float_t TINYEKF_INSTANCE_SCALAR
// The namespace of this instance.
#define TINYEKF_INSTANCE TINYEKF_INSTANCE_NAMESPACE(TINYEKF_INSTANCE_N, TINYEKF_INSTANCE_M, TINYEKF_INSTANCE_SCALAR)

namespace TINYEKF_INSTANCE
{
#include <tinyekf.h>
}

/// TinyEKF with EKF_N states and EKF_M measurements in _float_t.
template <>
struct TinyEkf<EKF_N, EKF_M, _float_t>
{
    using Filter = TINYEKF_INSTANCE::ekf_t;

    /// TinyEKF's ekf_predict(): the mean becomes \a fx, and the covariance F P F^T + Q.
    static void
    predict(Filter & ekf, _float_t const * fx, _float_t const * F, _float_t const * Q)
    {
        TINYEKF_INSTANCE::ekf_predict(&ekf, fx, F, Q);
    }

    /// TinyEKF's ekf_update(), false if the innovation covariance has no Cholesky factor.
    [[nodiscard]] static bool
    update(Filter & ekf, _float_t const * z, _float_t const * hx, _float_t const * H, _float_t const * R)
    {
        return TINYEKF_INSTANCE::ekf_update(&ekf, z, hx, H, R);
    }
};

#undef EKF_N
#undef EKF_M
#undef _float_t
#undef TINYEKF_INSTANCE
#undef TINYEKF_INSTANCE_N
#undef TINYEKF_INSTANCE_M
#undef TINYEKF_INSTANCE_SCALAR
