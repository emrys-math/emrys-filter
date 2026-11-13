# Installs EmrysFilter's user skills into an agent's skills directory:
#
#   cmake -DDESTINATION=<dir> -P tools/install_skills.cmake
#
# DESTINATION is one directory, or several separated by |, relative to the working directory. Each
# skill in skills/ becomes a directory of its own name, emrys-filter-<name>, holding its SKILL.md
# with the library's version filled in, references/usage.md, references/choosing.md, and
# references/bibliography.md, and examples/ with the example programs, so that it reads nothing
# outside its directory. The script owns the directories named emrys-filter-* in each destination:
# it removes them and writes them again, and leaves everything else there alone. emrys_filter_add_skills_target(), in
# cmake/EmrysFilterSkills.cmake, runs this script as a target. It runs from a checkout or from an
# install's share/EmrysFilter, which holds the same paths.
cmake_minimum_required(VERSION 3.28)

if(NOT DESTINATION)
  message(FATAL_ERROR "install_skills.cmake: set DESTINATION, the agent's skills directory")
endif()

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH root)

# The version is the project's: in a checkout, read from the line project() gives it on, and in an
# install, from the file VERSION the install writes beside tools/.
if(EXISTS "${root}/CMakeLists.txt")
  file(STRINGS "${root}/CMakeLists.txt" versionLine REGEX "^  VERSION [0-9.]+$" LIMIT_COUNT 1)
  string(REGEX REPLACE "^  VERSION " "" version "${versionLine}")
else()
  file(STRINGS "${root}/VERSION" version LIMIT_COUNT 1)
endif()
if(NOT version)
  message(FATAL_ERROR "install_skills.cmake: no version in ${root}")
endif()

# usage.md and choosing.md link the examples as ../examples/, which from references/ is the
# skill's examples/, and the bibliography as reference/bibliography.md, which is beside them in the
# copy.
file(READ "${root}/docs/usage.md" usage)
string(REPLACE "reference/bibliography.md" "bibliography.md" usage "${usage}")
file(READ "${root}/docs/choosing.md" choosing)
string(REPLACE "reference/bibliography.md" "bibliography.md" choosing "${choosing}")
file(GLOB examples "${root}/examples/*.cpp")
file(GLOB skills LIST_DIRECTORIES true "${root}/skills/emrys-filter-*")

string(REPLACE "|" ";" destinations "${DESTINATION}")
foreach(destination IN LISTS destinations)
  cmake_path(ABSOLUTE_PATH destination BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
  file(GLOB previous LIST_DIRECTORIES true "${destination}/emrys-filter-*")
  foreach(directory IN LISTS previous)
    if(IS_DIRECTORY "${directory}")
      file(REMOVE_RECURSE "${directory}")
    endif()
  endforeach()

  foreach(skill IN LISTS skills)
    cmake_path(GET skill FILENAME name)
    set(target "${destination}/${name}")
    file(COPY "${skill}/" DESTINATION "${target}")
    file(READ "${skill}/SKILL.md" text)

    # The Agent Skills specification: the name matches the directory, and the description is at
    # most 1024 characters.
    string(REGEX MATCH "\nname: ([^\n]*)\n" match "${text}")
    if(NOT CMAKE_MATCH_1 STREQUAL name)
      message(FATAL_ERROR "${skill}/SKILL.md: its name is \"${CMAKE_MATCH_1}\", not its directory's, ${name}")
    endif()
    string(REGEX MATCH "\ndescription: ([^\n]*)\n" match "${text}")
    string(LENGTH "${CMAKE_MATCH_1}" length)
    if(length EQUAL 0 OR length GREATER 1024)
      message(FATAL_ERROR "${skill}/SKILL.md: its description is ${length} characters, not 1 to 1024")
    endif()

    string(REPLACE "@EMRYS_FILTER_VERSION@" "${version}" text "${text}")
    file(WRITE "${target}/SKILL.md" "${text}")
    file(WRITE "${target}/references/usage.md" "${usage}")
    file(WRITE "${target}/references/choosing.md" "${choosing}")
    file(COPY "${root}/docs/reference/bibliography.md" DESTINATION "${target}/references")
    file(COPY ${examples} DESTINATION "${target}/examples")

    # Every relative link in the skill's Markdown must reach a file in the copy.
    file(GLOB_RECURSE pages "${target}/*.md")
    foreach(page IN LISTS pages)
      # A list doesn't split inside square brackets, so the link's ] is replaced first. A target
      # has no space, which tells a link from a lambda's `[](auto const & in, ...)` in a snippet.
      file(READ "${page}" pageText)
      string(REPLACE "](" "<link>(" pageText "${pageText}")
      string(REGEX MATCHALL "<link>\\([^):# ]+[)#]" links "${pageText}")
      cmake_path(GET page PARENT_PATH pageDirectory)
      foreach(link IN LISTS links)
        string(REGEX REPLACE "^<link>\\(|[)#]$" "" link "${link}")
        if(NOT EXISTS "${pageDirectory}/${link}")
          message(FATAL_ERROR "${page}: the link ${link} reaches no file in the installed skill")
        endif()
      endforeach()
    endforeach()
  endforeach()
  message(STATUS "Installed EmrysFilter ${version} skills in ${destination}")
endforeach()
