# Benchmarks: each problem implemented with EmrysFilter and in plain Eigen.
#
# The agreement, consistency, and allocation tests that must pass before any timing counts build
# and run with the unit tests, so the two implementations can't drift apart. They are built with
# EIGEN_RUNTIME_NO_MALLOC, which the allocation test needs in every translation unit.
#
# The runners build with EMRYS_FILTER_BUILD_BENCHMARKS, and the benchmark target runs them; the
# *-bench presets configure a release build that does both.

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

  add_custom_target(benchmark
    COMMAND emrys_filter_ranging_benchmark
    COMMAND emrys_filter_attitude_benchmark
    COMMAND emrys_filter_particles_benchmark
    USES_TERMINAL
    COMMENT "Timing each problem, EmrysFilter against plain Eigen"
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
