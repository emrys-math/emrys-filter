#pragma once

/** \file
 * The macros every header includes first. Includes Version.h, which checks the Eigen version, and
 * then Eigen/Core. Nothing here depends on the rest of the library; read Math.h next.
 */

#include <EmrysFilter/Core/Version.h>

#include <Eigen/Core>

/** \ingroup Core
 * Assertion used for programmer errors (misuse of the API).
 *
 * Defaults to \c eigen_assert, so it is controlled by the same switches as Eigen's
 * own checks (\c NDEBUG, \c EIGEN_NO_DEBUG). Define it before including any
 * EmrysFilter header to override, as an expression like \c assert() or as a statement such as
 * <tt>do { ... } while (false)</tt>: the library uses it only as a statement of its own.
 */
#ifndef EMRYS_FILTER_ASSERT
#    define EMRYS_FILTER_ASSERT(x) eigen_assert(x)
#endif

// EMRYS_FILTER_NO_IO leaves out printing (see Print.h). Eigen's EIGEN_NO_IO implies it, since
// Eigen then has no operator<< for the matrices a description prints.
#if defined(EIGEN_NO_IO) && !defined(EMRYS_FILTER_NO_IO)
#    define EMRYS_FILTER_NO_IO
#endif

/** \ingroup Core
 * Lets an empty member, such as the default Linear space, take no storage.
 */
#if defined(__has_cpp_attribute) && __has_cpp_attribute(msvc::no_unique_address)
#    define EMRYS_FILTER_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif defined(__has_cpp_attribute) && __has_cpp_attribute(no_unique_address)
#    define EMRYS_FILTER_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#    define EMRYS_FILTER_NO_UNIQUE_ADDRESS
#endif
