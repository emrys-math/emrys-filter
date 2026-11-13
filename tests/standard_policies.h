#pragma once

// Whether the standard library has parallel execution policies the tests can run: libc++ has them
// only as an experimental feature, and libstdc++'s parallel algorithms catch exceptions, so they
// don't compile under -fno-exceptions. Where there are none, the tests that call
// std::execution::par and its kin are left out; Sequential and the users' policies are still
// tested.

#include <version>

#if defined(__cpp_lib_execution) && (defined(__cpp_exceptions) || !defined(__GLIBCXX__))
#    define EMRYS_FILTER_TEST_STANDARD_POLICIES 1
#else
#    define EMRYS_FILTER_TEST_STANDARD_POLICIES 0
#endif
