# The user skills install as a test: into a directory beside an agent's own skill, twice, which
# checks each skill against the Agent Skills specification and its links, that a second install
# replaces the first, and that the agent's own skill is left alone.
add_test(NAME skills.install
  COMMAND "${CMAKE_COMMAND}" "-DDESTINATION=${CMAKE_CURRENT_BINARY_DIR}/agent-skills" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/test_install.cmake"
)
