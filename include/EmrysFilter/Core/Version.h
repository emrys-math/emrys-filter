#pragma once

/** \file
 * The library's version, the same numbers as the CMake project's, and the Eigen version it
 * requires. It includes only Eigen/Version, and Macros.h includes it, so every header makes the
 * version available. Read Macros.h next.
 */

// Eigen/Version is new in Eigen 5.0, and holds nothing but the version macros, so an older Eigen
// fails here with a message rather than with a missing file.
#if !__has_include(<Eigen/Version>)
#    error "EmrysFilter requires Eigen 5.0 or newer"
#endif
#include <Eigen/Version>

// The library is tested against Eigen 5.0.1 alone. Any 5.0 or later is accepted, 5.0.0 included,
// but untested.
static_assert(EIGEN_MAJOR_VERSION >= 5, "EmrysFilter requires Eigen 5.0 or newer");

namespace Emrys
{

/** \ingroup Core
 * The library's version, as major.minor.patch; the same numbers as the CMake project's.
 */
inline constexpr int versionMajor = 0;

/** \ingroup Core
 * The minor version. \sa versionMajor
 */
inline constexpr int versionMinor = 1;

/** \ingroup Core
 * The patch version. \sa versionMajor
 */
inline constexpr int versionPatch = 0;

}  // namespace Emrys
