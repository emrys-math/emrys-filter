cmake_minimum_required(VERSION 3.28)

project(EmrysFilter
  VERSION 0.1.0
  DESCRIPTION "State estimation utilities in C++23 using Eigen"
  LANGUAGES CXX
)

option(EMRYS_FILTER_BUILD_TESTS "Build EmrysFilter unit tests" ${PROJECT_IS_TOP_LEVEL})
option(EMRYS_FILTER_BUILD_EXAMPLES "Build the examples and run them as tests" ${EMRYS_FILTER_BUILD_TESTS})
option(EMRYS_FILTER_BUILD_DOCS "Add the docs target (needs Doxygen)" OFF)
option(EMRYS_FILTER_BUILD_BENCHMARKS "Build the benchmark runners and the benchmark target" OFF)
option(EMRYS_FILTER_INSTALL "Install the headers, the CMake package, and a fetched Eigen"
  ${PROJECT_IS_TOP_LEVEL})

# Only set global language settings when we're the top-level project; when
# consumed via FetchContent, the C++23 requirement propagates through the target.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

include(FetchContent)

# --- Eigen ------------------------------------------------------------------
# Use a system Eigen 5.0+ if available, otherwise fetch it.  The content name
# matches the package name (Eigen3) so a parent project that declares Eigen
# first takes precedence.  When EmrysFilter installs, a fetched Eigen installs
# beside it with its CMake package, which the installed EmrysFilterConfig.cmake
# finds [cmake-package]; otherwise it is excluded from all, which also drops its
# install rules.  The build tree never goes in the user's package registry.
set(EMRYS_FILTER_EIGEN_VERSION 5.0)
set(EMRYS_FILTER_EIGEN_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
if(EMRYS_FILTER_INSTALL)
  set(EMRYS_FILTER_EIGEN_EXCLUDE_FROM_ALL)
endif()

# Let the normal variables below override Eigen's option() defaults.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_BLAS OFF)
set(EIGEN_BUILD_LAPACK OFF)
set(EIGEN_BUILD_DEMOS OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_CMAKE_PACKAGE ${EMRYS_FILTER_INSTALL})
set(CMAKE_EXPORT_PACKAGE_REGISTRY OFF)

FetchContent_Declare(Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
  SYSTEM
  ${EMRYS_FILTER_EIGEN_EXCLUDE_FROM_ALL}
  FIND_PACKAGE_ARGS ${EMRYS_FILTER_EIGEN_VERSION} CONFIG
)
# Eigen 5.0's own CMakeLists.txt sets policy CMP0146 to OLD, which newer CMake warns is
# deprecated; that warning is Eigen's to fix, so silence it for Eigen's directory only.
set(CMAKE_WARN_DEPRECATED OFF)
FetchContent_MakeAvailable(Eigen3)
unset(CMAKE_WARN_DEPRECATED)

# --- Library ----------------------------------------------------------------
include(GNUInstallDirs)

add_library(EmrysFilter INTERFACE)
add_library(EmrysFilter::EmrysFilter ALIAS EmrysFilter)

target_include_directories(EmrysFilter INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(EmrysFilter INTERFACE cxx_std_23)
target_link_libraries(EmrysFilter INTERFACE Eigen3::Eigen)

# --- User skills ------------------------------------------------------------
include(cmake/EmrysFilterSkills.cmake)

# --- Install ----------------------------------------------------------------
# The headers, and under share/EmrysFilter the CMake package and what the skills install from:
# the script, the skills, usage.md and choosing.md, the bibliography, the examples, and the version
# in a file.
if(EMRYS_FILTER_INSTALL)
  include(CMakePackageConfigHelpers)
  set(EMRYS_FILTER_DATA_DIR ${CMAKE_INSTALL_DATADIR}/EmrysFilter)
  set(EMRYS_FILTER_PACKAGE_DIR ${EMRYS_FILTER_DATA_DIR}/cmake)

  install(TARGETS EmrysFilter EXPORT EmrysFilterTargets)
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT EmrysFilterTargets NAMESPACE EmrysFilter:: DESTINATION ${EMRYS_FILTER_PACKAGE_DIR})

  configure_package_config_file(cmake/EmrysFilterConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/EmrysFilterConfig.cmake
    INSTALL_DESTINATION ${EMRYS_FILTER_PACKAGE_DIR}
    NO_SET_AND_CHECK_MACRO
  )
  # Before 1.0 a minor version may break, so a request for 0.1 takes only 0.1.x.
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/EmrysFilterConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
    ARCH_INDEPENDENT
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/EmrysFilterConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/EmrysFilterConfigVersion.cmake
    cmake/EmrysFilterSkills.cmake
    DESTINATION ${EMRYS_FILTER_PACKAGE_DIR}
  )

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/VERSION "${PROJECT_VERSION}\n")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VERSION DESTINATION ${EMRYS_FILTER_DATA_DIR})
  install(FILES tools/install_skills.cmake DESTINATION ${EMRYS_FILTER_DATA_DIR}/tools)
  install(DIRECTORY skills/ DESTINATION ${EMRYS_FILTER_DATA_DIR}/skills
    PATTERN CMakeLists.txt EXCLUDE
    PATTERN test_install.cmake EXCLUDE
  )
  install(FILES docs/usage.md docs/choosing.md DESTINATION ${EMRYS_FILTER_DATA_DIR}/docs)
  install(FILES docs/reference/bibliography.md DESTINATION ${EMRYS_FILTER_DATA_DIR}/docs/reference)
  file(GLOB EMRYS_FILTER_EXAMPLE_SOURCES examples/*.cpp)
  install(FILES ${EMRYS_FILTER_EXAMPLE_SOURCES} DESTINATION ${EMRYS_FILTER_DATA_DIR}/examples)
endif()

# --- Warnings ---------------------------------------------------------------
# The tests, examples, and benchmarks build with these, warnings as errors unless turned off
# (decisions.md [embedded-builds]). Eigen and GoogleTest are system includes, so only the library's
# warnings and the programs' own count, except where GCC inlines Eigen into them: its
# -Wmaybe-uninitialized then flags the copy of a Map's empty Stride in Quaternion::isApprox(), so
# GCC builds without that warning.
option(EMRYS_FILTER_WARNINGS_AS_ERRORS "Build the tests, examples, and benchmarks with -Werror or /WX" ON)
set(EMRYS_FILTER_WARNINGS
  $<IF:$<CXX_COMPILER_ID:MSVC>,/W4 /permissive- /Zc:__cplusplus /bigobj,-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
  $<$<BOOL:${EMRYS_FILTER_WARNINGS_AS_ERRORS}>:$<IF:$<CXX_COMPILER_ID:MSVC>,/WX,-Werror>>
)

# --- Tests ------------------------------------------------------------------
if(EMRYS_FILTER_BUILD_TESTS)
  # Build settings for everything below (GoogleTest, the tests, the benchmarks' agreement tests).
  # Each test translation unit instantiates heavy Eigen templates and takes over a gigabyte, so
  # Ninja runs at most EMRYS_FILTER_TEST_COMPILE_JOBS of them at once whatever its -j; beyond
  # that they only swap. GCC and Clang Debug builds keep line tables but not full debug info, which
  # is most of a test's compile time, and build with -Og: at -O0 every Eigen helper is emitted as
  # its own function, tens of thousands per test, and MinGW's assembler takes minutes over the
  # sections that makes. Turn EMRYS_FILTER_TEST_FULL_DEBUG_INFO on to step through a test. MSVC
  # embeds debug info in each object (/Z7), so parallel compiles don't queue on one PDB, and links
  # with 8 KB PDB pages: an incremental link grows the PDB each time, and the main test program's
  # passed the 4 GB that the default 4 KB pages allow after many relinks, though one link makes 2 GB.
  set(EMRYS_FILTER_TEST_COMPILE_JOBS 12 CACHE STRING "Test translation units Ninja compiles at once")
  option(EMRYS_FILTER_TEST_FULL_DEBUG_INFO "Build the tests unoptimized with full debug info" OFF)
  # The negative compile tests compile once per case, minutes in all. The clang, gcc, and msvc presets
  # register them, and only their <compiler>-compile-errors presets build and run them [check-stages].
  option(EMRYS_FILTER_TEST_COMPILE_ERRORS "Register the negative compile tests (tests/*/compile_errors.cpp)" OFF)
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS emrys_filter_test_compile=${EMRYS_FILTER_TEST_COMPILE_JOBS})
  set(CMAKE_JOB_POOL_COMPILE emrys_filter_test_compile)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
  add_link_options("$<$<CXX_COMPILER_ID:MSVC>:/PDBPAGESIZE:8192>")
  if(NOT EMRYS_FILTER_TEST_FULL_DEBUG_INFO)
    add_compile_options(
      "$<$<CONFIG:Debug>:$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Og;-gline-tables-only>$<$<CXX_COMPILER_ID:GNU>:-Og;-g1>>"
    )
  endif()

  enable_testing()
  add_subdirectory(tests)
endif()

# --- Examples ---------------------------------------------------------------
if(EMRYS_FILTER_BUILD_EXAMPLES)
  enable_testing()
  add_subdirectory(examples)
  add_subdirectory(skills)
endif()

# --- Benchmarks -------------------------------------------------------------
# The benchmarks' agreement tests build with the tests, reusing their GoogleTest; the runners only
# on request, and mean something only in a release build (the *-bench presets).
if(EMRYS_FILTER_BUILD_TESTS OR EMRYS_FILTER_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# --- Documentation ----------------------------------------------------------
if(EMRYS_FILTER_BUILD_DOCS)
  add_subdirectory(docs)
endif()
