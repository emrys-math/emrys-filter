# Use a system GoogleTest if available, otherwise fetch it.
set(INSTALL_GTEST OFF)
# Match the default MSVC runtime (/MD) used by the rest of the build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
  URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
  SYSTEM
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(emrys_filter_tests
  smoke.cpp
  unequal_dimensions_distributions.cpp
  unequal_dimensions_filters.cpp
  unequal_dimensions_transforms.cpp
  parallel.cpp
  printing.cpp
  no_io.cpp
  policies.cpp
  Core/resampling.cpp
  Core/math.cpp
  Core/execution.cpp
  Core/triangularization.cpp
  Core/solve.cpp
  Core/time.cpp
  Spaces/space.cpp
  Spaces/blocks.cpp
  Spaces/reset.cpp
  Spaces/unit_quaternion.cpp
  Distributions/moments.cpp
  Distributions/gaussian.cpp
  Distributions/gaussian_mixture.cpp
  Distributions/particle_set.cpp
  Distributions/sampling.cpp
  Distributions/roots.cpp
  Models/spread.cpp
  Distributions/kernel_density.cpp
  Distributions/gaussian_subspace.cpp
  Distributions/mixture_subspace.cpp
  Distributions/particle_subspace.cpp
  Distributions/fitting.cpp
  Distributions/gaussian_space.cpp
  Distributions/space_distributions.cpp
  Distributions/resizable.cpp
  Distributions/densities.cpp
  Distributions/uniform_mixture.cpp
  Distributions/product.cpp
  Distributions/categorical.cpp
  Distributions/continuous_time_markov_chain.cpp
  Distributions/divergence.cpp
  Distributions/autodiff_scalars.cpp
  Distributions/upkeep.cpp
  Distributions/component.cpp
  Distributions/gaussian_forms.cpp
  Distributions/info_gaussian.cpp
  Models/transform.cpp
  Models/affine_coordinates.cpp
  Models/covariance_updates.cpp
  Models/consider.cpp
  Models/linearization_forms.cpp
  Models/iterated.cpp
  Models/iterated_posterior_linearization.cpp
  Models/iterated_forms.cpp
  Models/iterated_forms_attitude.cpp
  Models/linearized_about.cpp
  Models/sigma_points.cpp
  Models/models.cpp
  Models/model_policies.cpp
  Models/batched.cpp
  Models/measurement_model.cpp
  Models/whole_state_models.cpp
  Models/placement.cpp
  Models/computed_space.cpp
  Models/linear_measurement.cpp
  Models/per_measurement.cpp
  Models/fixed_step_model.cpp
  Models/consistency.cpp
  Models/cramer_rao.cpp
  Models/simulation.cpp
  Models/emission.cpp
  Filters/kalman_filter.cpp
  Filters/consider_consistency.cpp
  Filters/info_filter.cpp
  Filters/sqrt_filters.cpp
  Filters/precise_input_noise.cpp
  Filters/gaussian_mixture_filter.cpp
  Filters/imm_filter.cpp
  Filters/amm_filter.cpp
  Filters/particle_filter.cpp
  Filters/rb_point_mass_filter.cpp
  Filters/rb_particle_filter.cpp
  Filters/hidden_markov_filter.cpp
  Filters/mixture_filter.cpp
  Filters/time_steps.cpp
  Filters/fixed_step_binding.cpp
  Filters/filter_upkeep.cpp
  Filters/update_failure.cpp
  Filters/update_outcomes.cpp
  Filters/large_fixed.cpp
  Smoothers/smoothing.cpp
  Smoothers/mixture_smoothing.cpp
)
target_link_libraries(emrys_filter_tests PRIVATE
  EmrysFilter::EmrysFilter
  GTest::gtest_main
)
# Test support shared between files, such as the unit quaternion space in Spaces/.
target_include_directories(emrys_filter_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(emrys_filter_tests PRIVATE
  ${EMRYS_FILTER_WARNINGS}
)

# libstdc++ runs the parallel algorithms on TBB whenever its headers are found, and then the program
# must link it; without TBB it runs them on one thread. MSVC's are built in.
find_package(TBB QUIET)
if(TBB_FOUND)
  target_link_libraries(emrys_filter_tests PRIVATE TBB::tbb)
endif()

gtest_discover_tests(emrys_filter_tests)

# The allocation tests forbid Eigen's heap at run time, which needs EIGEN_RUNTIME_NO_MALLOC in
# every translation unit of the target, and replace the global operator new, so they have their own.
add_executable(emrys_filter_allocation_tests
  allocation.cpp
  allocation_forms.cpp
  allocation_mixtures.cpp
  allocation_particles.cpp
)
target_link_libraries(emrys_filter_allocation_tests PRIVATE
  EmrysFilter::EmrysFilter
  GTest::gtest_main
)
target_include_directories(emrys_filter_allocation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(emrys_filter_allocation_tests PRIVATE EIGEN_RUNTIME_NO_MALLOC)
target_compile_options(emrys_filter_allocation_tests PRIVATE
  ${EMRYS_FILTER_WARNINGS}
)

gtest_discover_tests(emrys_filter_allocation_tests)

# An EMRYS_FILTER_ASSERT override changes the body of every inline function that asserts, which a
# program's translation units must agree on, so the test that defines one is a program of its own.
add_executable(emrys_filter_assert_override_tests
  assert_override.cpp
)
target_link_libraries(emrys_filter_assert_override_tests PRIVATE
  EmrysFilter::EmrysFilter
  GTest::gtest_main
)
target_compile_options(emrys_filter_assert_override_tests PRIVATE
  ${EMRYS_FILTER_WARNINGS}
)

gtest_discover_tests(emrys_filter_assert_override_tests)

# The negative compile tests [coverage]. Each `#if defined(CASE)` block of these files is a misuse
# that must fail to compile with the static_assert or deleted function its comment names, and is a
# test of its own, run by compile_errors.cmake with the file's compile command and CASE defined.
# Without a case each file compiles, as the target emrys_filter_compile_errors, which checks what its
# cases share, and a case compiles again only when that object has been rebuilt since it last passed
# [check-stages]. tools/check_compile_errors.py checks that every compile error the headers claim
# has a case. They are opt-in, with EMRYS_FILTER_TEST_COMPILE_ERRORS, which the clang, gcc, and msvc
# presets set. The target is not part of the default build, and those presets' test runs leave the
# cases out (label compile_error): the <compiler>-compile-errors workflows build the target and run
# them, since a case checks a compiler's result, which no sanitizer or coverage flag changes.
if(EMRYS_FILTER_TEST_COMPILE_ERRORS)
  set(EMRYS_FILTER_COMPILE_ERROR_SOURCES
    Spaces/compile_errors.cpp
    Distributions/compile_errors.cpp
    Models/compile_errors.cpp
    Filters/compile_errors.cpp
  )
  add_library(emrys_filter_compile_errors OBJECT EXCLUDE_FROM_ALL ${EMRYS_FILTER_COMPILE_ERROR_SOURCES})
  target_link_libraries(emrys_filter_compile_errors PRIVATE EmrysFilter::EmrysFilter)
  # The cases rerun the compile command alone, which a module scan's flags would tie to the build.
  set_target_properties(emrys_filter_compile_errors PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  target_compile_options(emrys_filter_compile_errors PRIVATE
    ${EMRYS_FILTER_WARNINGS}
  )
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${EMRYS_FILTER_COMPILE_ERROR_SOURCES})
  foreach(source IN LISTS EMRYS_FILTER_COMPILE_ERROR_SOURCES)
    cmake_path(REMOVE_EXTENSION source OUTPUT_VARIABLE prefix)
    string(REPLACE "/" "." prefix "${prefix}")
    string(REPLACE "." "\\." pattern "/${source}.")
    set(object "$<FILTER:$<TARGET_OBJECTS:emrys_filter_compile_errors>,INCLUDE,${pattern}>")
    # Only the openings are kept, since a bracket in a comment would join list elements.
    file(READ ${source} text)
    string(REGEX MATCHALL "\n#if defined\\([A-Za-z_0-9]+\\)" openings "${text}")
    foreach(opening IN LISTS openings)
      string(REGEX MATCH "defined\\(([A-Za-z_0-9]+)\\)" _ "${opening}")
      add_test(NAME ${prefix}.${CMAKE_MATCH_1}
        COMMAND ${CMAKE_COMMAND}
          -DCOMMANDS=${CMAKE_BINARY_DIR}/compile_commands.json
          -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/${source}
          -DCASE=${CMAKE_MATCH_1}
          "-DSYNTAX_ONLY=$<IF:$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},MSVC>,/Zs,-fsyntax-only>"
          "-DOBJECT=${object}"
          -DSTAMP=${CMAKE_CURRENT_BINARY_DIR}/compile_errors/${prefix}.${CMAKE_MATCH_1}.passed
          -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_errors.cmake
      )
      # A case is a compile and takes a test translation unit's memory, so it counts as two of the
      # test presets' eight jobs: four compile at once.
      set_tests_properties(${prefix}.${CMAKE_MATCH_1} PROPERTIES LABELS compile_error PROCESSORS 2)
    endforeach()
  endforeach()
endif()
