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

if(EMRYS_FILTER_BUILD_TESTS)
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

  add_custom_target(benchmark
    ${benchmark_commands}
    USES_TERMINAL
    COMMENT "Timing each problem, EmrysFilter against plain Eigen, mherb/kalman, and TinyEKF"
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
