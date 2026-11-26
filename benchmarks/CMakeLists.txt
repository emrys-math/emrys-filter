# Benchmarks: each problem implemented with EmrysFilter and in plain Eigen.
#
# The agreement, consistency, and allocation tests that must pass before any timing counts build
# and run with the unit tests, so the two implementations can't drift apart. They are built with
# EIGEN_RUNTIME_NO_MALLOC, which the allocation test needs in every translation unit.
#
# The runners build with EMRYS_FILTER_BUILD_BENCHMARKS, and the benchmark target runs them; the
# *-bench presets configure a release build that does both.

# mherb/kalman, the other C++ library that runs the ranging problem, a third implementation of it
# (docs/design/benchmarks.md, claim 9). It is fetched at its last commit, for the benchmarks alone,
# and never becomes a dependency of the library. Its own CMakeLists.txt builds its tests and
# examples, so only its headers are used: SOURCE_SUBDIR names a directory without one, and the
# headers come in as system includes, since one of them warns under Clang.
option(EMRYS_FILTER_BENCHMARK_MHERB "Build the ranging benchmark against mherb/kalman, which is fetched" ON)
if(EMRYS_FILTER_BENCHMARK_MHERB)
  FetchContent_Declare(mherb_kalman
    URL https://github.com/mherb/kalman/archive/9f40c2f7bedb08e964297a00a2b9360036590896.tar.gz
    URL_HASH SHA256=6fe66ac2128ce73ffd98457827b63af33efbd6690a90ff23ba51035d4c2d6603
    SOURCE_SUBDIR headers-only
  )
  FetchContent_MakeAvailable(mherb_kalman)
  add_library(emrys_filter_mherb_kalman INTERFACE)
  target_include_directories(emrys_filter_mherb_kalman SYSTEM INTERFACE "${mherb_kalman_SOURCE_DIR}/include")
  target_link_libraries(emrys_filter_mherb_kalman INTERFACE Eigen3::Eigen)
endif()

# TinyEKF, a C EKF for microcontrollers, another implementation of the ranging EKF (claims 6 and 9).
# It is fetched at its last commit, for the benchmarks alone, like mherb/kalman. It is one header,
# src/tinyekf.h, which comes in as a system include, since it is C that warns as C++ under these
# warnings; the repository has no CMakeLists.txt, so nothing of it is built.
option(EMRYS_FILTER_BENCHMARK_TINYEKF "Build the ranging benchmark against TinyEKF, which is fetched" ON)
if(EMRYS_FILTER_BENCHMARK_TINYEKF)
  FetchContent_Declare(tinyekf
    URL https://github.com/simondlevy/TinyEKF/archive/ba0d2a90e22f2f4e4120a91cb05ca2ca6a8e8da3.tar.gz
    URL_HASH SHA256=78ed05682c5863b452741cfe07c10eee7896c4def8d43d5c07128a1407f1e6ed
  )
  FetchContent_MakeAvailable(tinyekf)
  add_library(emrys_filter_tinyekf INTERFACE)
  target_include_directories(emrys_filter_tinyekf SYSTEM INTERFACE "${tinyekf_SOURCE_DIR}/src")
endif()

# FrancoisCarouge/Kalman, a C++ Kalman filter library over a linear algebra backend its user
# supplies, another implementation of the ranging EKF (claim 9). It is fetched at the commit it was
# first benchmarked at, for the benchmarks alone, like mherb/kalman; its last release, 0.5.3, has no
# extended filter with a fixed transition. Only its headers are used: those of include/, the
# library, and those of support/eigen/, the Eigen backend its README points an Eigen user to.
# SOURCE_SUBDIR names a directory without a CMakeLists.txt, so none of its own build runs, and the
# headers come in as system includes. Its project builds as C++26, but these headers build as C++23.
option(EMRYS_FILTER_BENCHMARK_CAROUGE "Build the ranging benchmark against FrancoisCarouge/Kalman, which is fetched" ON)
if(EMRYS_FILTER_BENCHMARK_CAROUGE)
  FetchContent_Declare(carouge_kalman
    URL https://github.com/FrancoisCarouge/Kalman/archive/cdd8a938794319d171faa89b71aa0bb858c622b6.tar.gz
    URL_HASH SHA256=f30a5beb22b25c3788f0ca6043cd4ffa2a2dc23e278f884d38472a9833078cb1
    SOURCE_SUBDIR headers-only
  )
  FetchContent_MakeAvailable(carouge_kalman)
  add_library(emrys_filter_carouge_kalman INTERFACE)
  target_include_directories(emrys_filter_carouge_kalman SYSTEM INTERFACE
    "${carouge_kalman_SOURCE_DIR}/include"
    "${carouge_kalman_SOURCE_DIR}/support/eigen"
  )
  target_link_libraries(emrys_filter_carouge_kalman INTERFACE Eigen3::Eigen)
  # Its headers name types with letters outside ASCII, in UTF-8 with no byte-order mark, which MSVC
  # reads in the code page unless told otherwise.
  target_compile_options(emrys_filter_carouge_kalman INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
endif()

# OpenCV's cv::KalmanFilter, the Kalman filter of the computer-vision library, another
# implementation of the ranging EKF (claim 9). OpenCV is too large to fetch and build for a
# benchmark, and its users install it, so it is found rather than fetched: the option is off by
# default, and turning it on needs an OpenCV 4 or 5 with its core and video modules where
# find_package() looks (OpenCV_DIR). Its kalman.cpp is the same in 4.14.0 and 5.1.0.
option(EMRYS_FILTER_BENCHMARK_OPENCV "Build the ranging benchmark against OpenCV, which must be installed" OFF)
if(EMRYS_FILTER_BENCHMARK_OPENCV)
  find_package(OpenCV REQUIRED COMPONENTS core video)
  add_library(emrys_filter_opencv INTERFACE)
  target_include_directories(emrys_filter_opencv SYSTEM INTERFACE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(emrys_filter_opencv INTERFACE opencv_core opencv_video)
endif()

if(EMRYS_FILTER_BUILD_TESTS AND EMRYS_FILTER_TEST_SHARD_programs)
  add_executable(emrys_filter_benchmark_tests
    heap_counter.cpp
    ranging/agreement.cpp
    ranging/allocation.cpp
    particles/agreement.cpp
    particles/allocation.cpp
    attitude/agreement.cpp
    attitude/allocation.cpp
  )
  target_link_libraries(emrys_filter_benchmark_tests PRIVATE
    EmrysFilter::EmrysFilter
    GTest::gtest_main
  )
  if(EMRYS_FILTER_BENCHMARK_MHERB)
    target_sources(emrys_filter_benchmark_tests PRIVATE
      ranging/mherb_agreement.cpp
      ranging/mherb_allocation.cpp
    )
    target_link_libraries(emrys_filter_benchmark_tests PRIVATE emrys_filter_mherb_kalman)
  endif()
  if(EMRYS_FILTER_BENCHMARK_TINYEKF)
    target_sources(emrys_filter_benchmark_tests PRIVATE
      ranging/tinyekf_agreement.cpp
      ranging/tinyekf_allocation.cpp
    )
    target_link_libraries(emrys_filter_benchmark_tests PRIVATE emrys_filter_tinyekf)
  endif()
  if(EMRYS_FILTER_BENCHMARK_CAROUGE)
    target_sources(emrys_filter_benchmark_tests PRIVATE
      ranging/carouge_agreement.cpp
      ranging/carouge_allocation.cpp
      ranging/carouge_ranging.cpp
    )
    target_link_libraries(emrys_filter_benchmark_tests PRIVATE emrys_filter_carouge_kalman)
  endif()
  if(EMRYS_FILTER_BENCHMARK_OPENCV)
    target_sources(emrys_filter_benchmark_tests PRIVATE
      ranging/opencv_agreement.cpp
      ranging/opencv_allocation.cpp
    )
    target_link_libraries(emrys_filter_benchmark_tests PRIVATE emrys_filter_opencv)
  endif()
  target_compile_options(emrys_filter_benchmark_tests PRIVATE ${EMRYS_FILTER_WARNINGS})
  target_compile_definitions(emrys_filter_benchmark_tests PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  gtest_discover_tests(emrys_filter_benchmark_tests)
endif()

if(EMRYS_FILTER_BUILD_BENCHMARKS)
  # One runner per problem, each timing its EmrysFilter implementation against its plain-Eigen one.
  foreach(problem ranging attitude particles)
    add_executable(emrys_filter_${problem}_benchmark
      ${problem}/benchmark.cpp
    )
    target_link_libraries(emrys_filter_${problem}_benchmark PRIVATE EmrysFilter::EmrysFilter)
    target_compile_options(emrys_filter_${problem}_benchmark PRIVATE ${EMRYS_FILTER_WARNINGS})
  endforeach()

  set(benchmark_commands
    COMMAND emrys_filter_ranging_benchmark
    COMMAND emrys_filter_attitude_benchmark
    COMMAND emrys_filter_particles_benchmark
  )
  # The ranging EKF against mherb/kalman's.
  if(EMRYS_FILTER_BENCHMARK_MHERB)
    add_executable(emrys_filter_mherb_benchmark ranging/mherb_benchmark.cpp)
    target_link_libraries(emrys_filter_mherb_benchmark PRIVATE EmrysFilter::EmrysFilter emrys_filter_mherb_kalman)
    target_compile_options(emrys_filter_mherb_benchmark PRIVATE ${EMRYS_FILTER_WARNINGS})
    list(APPEND benchmark_commands COMMAND emrys_filter_mherb_benchmark)
  endif()
  # The ranging EKF against TinyEKF's.
  if(EMRYS_FILTER_BENCHMARK_TINYEKF)
    add_executable(emrys_filter_tinyekf_benchmark ranging/tinyekf_benchmark.cpp)
    target_link_libraries(emrys_filter_tinyekf_benchmark PRIVATE EmrysFilter::EmrysFilter emrys_filter_tinyekf)
    target_compile_options(emrys_filter_tinyekf_benchmark PRIVATE ${EMRYS_FILTER_WARNINGS})
    list(APPEND benchmark_commands COMMAND emrys_filter_tinyekf_benchmark)
  endif()
  # The ranging EKF against FrancoisCarouge/Kalman's, whose filter is in a translation unit of its
  # own (ranging/carouge_ranging.h says why).
  if(EMRYS_FILTER_BENCHMARK_CAROUGE)
    add_executable(emrys_filter_carouge_benchmark ranging/carouge_benchmark.cpp ranging/carouge_ranging.cpp)
    target_link_libraries(emrys_filter_carouge_benchmark PRIVATE EmrysFilter::EmrysFilter emrys_filter_carouge_kalman)
    target_compile_options(emrys_filter_carouge_benchmark PRIVATE ${EMRYS_FILTER_WARNINGS})
    list(APPEND benchmark_commands COMMAND emrys_filter_carouge_benchmark)
  endif()
  # The ranging EKF against OpenCV's.
  if(EMRYS_FILTER_BENCHMARK_OPENCV)
    add_executable(emrys_filter_opencv_benchmark ranging/opencv_benchmark.cpp)
    target_link_libraries(emrys_filter_opencv_benchmark PRIVATE EmrysFilter::EmrysFilter emrys_filter_opencv)
    target_compile_options(emrys_filter_opencv_benchmark PRIVATE ${EMRYS_FILTER_WARNINGS})
    list(APPEND benchmark_commands COMMAND emrys_filter_opencv_benchmark)
  endif()

  add_custom_target(benchmark
    ${benchmark_commands}
    USES_TERMINAL
    COMMENT "Timing each problem, EmrysFilter against plain Eigen, mherb/kalman, TinyEKF, FrancoisCarouge/Kalman, and OpenCV"
  )
endif()

if(EMRYS_FILTER_BUILD_BENCHMARKS)
  # The compile-cost table: tools/compile_cost.py times the compile commands of the translation
  # units below, which compile_cost_units.json lists. Each is an object library that nothing links,
  # so building it checks only that it compiles. They are, for each problem and linearization, the
  # EmrysFilter side and the plain-Eigen one, whose ratio is the claim; each umbrella header alone,
  # against Eigen/Dense alone; and the consumer project's main.cpp, where a new user starts.
  set(compile_cost_units)
  set(compile_cost_targets)
  function(emrys_filter_compile_cost_unit target source label reference)
    add_library(${target} OBJECT ${source})
    target_link_libraries(${target} PRIVATE EmrysFilter::EmrysFilter)
    target_compile_options(${target} PRIVATE ${EMRYS_FILTER_WARNINGS})
    target_compile_definitions(${target} PRIVATE ${ARGN})
    # Without module scanning, a unit's compile command depends on no scan step's output.
    set_target_properties(${target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
    set(compile_cost_units ${compile_cost_units}
      "{\"target\": \"${target}\", \"label\": \"${label}\", \"reference\": \"${reference}\"}" PARENT_SCOPE)
    set(compile_cost_targets ${compile_cost_targets} ${target} PARENT_SCOPE)
  endfunction()

  foreach(problem IN ITEMS ranging attitude)
    foreach(kind IN ITEMS Analytic AutoDiff)
      string(TOLOWER "${problem}_${kind}" name)
      emrys_filter_compile_cost_unit(emrys_filter_compile_cost_${name}_eigen
        ${problem}/compile_cost.cpp "${problem}, ${kind}, plain Eigen" ""
        COMPILE_COST_EMRYS=0 COMPILE_COST_KIND=${kind}
      )
      emrys_filter_compile_cost_unit(emrys_filter_compile_cost_${name}_emrys
        ${problem}/compile_cost.cpp "${problem}, ${kind}, EmrysFilter" emrys_filter_compile_cost_${name}_eigen
        COMPILE_COST_EMRYS=1 COMPILE_COST_KIND=${kind}
      )
    endforeach()
  endforeach()
  # The plain-Eigen particle filter computes its likelihood one way, and is the reference for both
  # of EmrysFilter's.
  emrys_filter_compile_cost_unit(emrys_filter_compile_cost_particles_eigen
    particles/compile_cost.cpp "particles, plain Eigen" "" COMPILE_COST_EMRYS=0
  )
  foreach(kind IN ITEMS Noise Own)
    string(TOLOWER "${kind}" name)
    emrys_filter_compile_cost_unit(emrys_filter_compile_cost_particles_${name}_emrys
      particles/compile_cost.cpp "particles, ${kind} likelihood, EmrysFilter" emrys_filter_compile_cost_particles_eigen
      COMPILE_COST_EMRYS=1 COMPILE_COST_KIND=${kind}
    )
  endforeach()

  set(reference)
  foreach(header IN ITEMS Eigen/Dense EmrysFilter/Core.h EmrysFilter/Spaces.h EmrysFilter/Distributions.h
                          EmrysFilter/Models.h EmrysFilter/Filters.h EmrysFilter/EmrysFilter.h)
    string(MAKE_C_IDENTIFIER "${header}" name)
    string(TOLOWER "${name}" name)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/compile_cost/${name}.cpp")
    file(CONFIGURE OUTPUT "${source}" CONTENT "#include <${header}>\n")
    if(NOT header STREQUAL "Eigen/Dense")
      set(reference emrys_filter_compile_cost_eigen_dense)
    endif()
    emrys_filter_compile_cost_unit(emrys_filter_compile_cost_${name} "${source}" "#include <${header}>" "${reference}")
  endforeach()

  emrys_filter_compile_cost_unit(emrys_filter_compile_cost_consumer
    "${PROJECT_SOURCE_DIR}/examples/consumer/main.cpp" "examples/consumer/main.cpp" ""
  )

  list(JOIN compile_cost_units ",\n  " units)
  file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/compile_cost_units.json" CONTENT "[\n  ${units}\n]\n")
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_custom_target(compile_cost
      COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/tools/compile_cost.py" "${CMAKE_BINARY_DIR}"
      USES_TERMINAL
      COMMENT "Timing the compile-cost units"
    )
    add_dependencies(compile_cost ${compile_cost_targets})
  endif()
endif()
