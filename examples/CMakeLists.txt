# Each example is one program, built and run as a test: it simulates its system, filters it, and
# returns nonzero when the filter's consistency check fails.
set(EMRYS_FILTER_EXAMPLES
  constant_velocity
  linear_kalman
  unicycle
  attitude
  consider_bias
  particle_filter
  imm
  gaussian_mixture
  rao_blackwellized
  point_mass_filter
  terrain_lookup
)

foreach(example IN LISTS EMRYS_FILTER_EXAMPLES)
  add_executable(emrys_filter_example_${example} ${example}.cpp)
  target_link_libraries(emrys_filter_example_${example} PRIVATE EmrysFilter::EmrysFilter)
  target_compile_options(emrys_filter_example_${example} PRIVATE ${EMRYS_FILTER_WARNINGS})
  add_test(NAME example.${example} COMMAND emrys_filter_example_${example})
endforeach()

# The project in consumer/ is the start of a user's own, so it builds as one, against this checkout
# fetched and then installed, with the Eigen already fetched and the same compiler.
add_test(NAME example.consumer
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
    "-DLIBRARY=${PROJECT_SOURCE_DIR}"
    "-DBINARY=${CMAKE_CURRENT_BINARY_DIR}/consumer"
    "-DGENERATOR=${CMAKE_GENERATOR}"
    "-DCOMPILER=${CMAKE_CXX_COMPILER}"
    "-DEIGEN=${eigen3_SOURCE_DIR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_consumer.cmake"
)
# It takes most of a test run's time, and only a change to the build or the install can break it,
# so a run while working leaves it out with -E example.consumer [check-stages].
set_tests_properties(example.consumer PROPERTIES LABELS package)
