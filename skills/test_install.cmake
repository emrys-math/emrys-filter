# Runs tools/install_skills.cmake twice into DESTINATION, which holds a skill of the agent's own and
# a stale emrys-filter-* skill, and fails unless the install replaced the stale skill, kept the
# agent's, and filled in the version.
cmake_minimum_required(VERSION 3.28)

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH root)
file(REMOVE_RECURSE "${DESTINATION}")
file(WRITE "${DESTINATION}/own-skill/SKILL.md" "an agent's own skill\n")
file(WRITE "${DESTINATION}/emrys-filter-removed/SKILL.md" "a skill of an earlier version\n")

foreach(pass 1 2)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "-DDESTINATION=${DESTINATION}" -P "${root}/tools/install_skills.cmake"
    RESULT_VARIABLE result
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "install ${pass} failed")
  endif()
endforeach()

if(NOT EXISTS "${DESTINATION}/own-skill/SKILL.md")
  message(FATAL_ERROR "the install removed a skill it doesn't own")
endif()
if(EXISTS "${DESTINATION}/emrys-filter-removed")
  message(FATAL_ERROR "the install left a skill of an earlier version")
endif()
file(GLOB skills "${DESTINATION}/emrys-filter-*/SKILL.md")
if(NOT skills)
  message(FATAL_ERROR "the install wrote no skill")
endif()
foreach(skill IN LISTS skills)
  file(READ "${skill}" text)
  if(text MATCHES "@EMRYS_FILTER_VERSION@")
    message(FATAL_ERROR "${skill}: the version is not filled in")
  endif()
endforeach()
