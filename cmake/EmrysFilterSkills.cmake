# emrys_filter_add_skills_target(DESTINATION <dir>...) adds the target emrys_filter_skills, never
# part of ALL, which installs the skills in skills/ for a coding agent: each destination is an
# agent's skills directory, relative to the calling directory. tools/install_skills.cmake does the
# work and says what it writes. The top-level CMakeLists.txt and the installed
# EmrysFilterConfig.cmake both include this file, and in a checkout and an install alike tools/ is
# beside its directory.
function(emrys_filter_add_skills_target)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "" "DESTINATION")
  if(NOT arg_DESTINATION)
    message(FATAL_ERROR "emrys_filter_add_skills_target: give DESTINATION, an agent's skills directory")
  endif()
  set(destinations)
  foreach(destination IN LISTS arg_DESTINATION)
    cmake_path(ABSOLUTE_PATH destination BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
    list(APPEND destinations "${destination}")
  endforeach()
  string(JOIN "|" destinations ${destinations})
  cmake_path(GET CMAKE_CURRENT_FUNCTION_LIST_DIR PARENT_PATH root)
  add_custom_target(emrys_filter_skills
    COMMAND "${CMAKE_COMMAND}" "-DDESTINATION=${destinations}" -P
      "${root}/tools/install_skills.cmake"
    COMMENT "Installing EmrysFilter's skills"
    VERBATIM
  )
endfunction()
