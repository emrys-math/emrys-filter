# Builds examples/consumer as a project of its own would, twice: once fetching this checkout, and
# once finding it installed, with the Eigen the install put beside it [cmake-package]. Each time it
# runs the program and installs the skills from it. Run as
#   cmake -DSOURCE=<consumer> -DLIBRARY=<checkout> -DBINARY=<directory> -DGENERATOR=<generator>
#         -DCOMPILER=<C++ compiler> [-DEIGEN=<Eigen source>] -P test_consumer.cmake
# where EIGEN, when given, saves fetching Eigen again.

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    list(JOIN ARGN " " command)
    message(FATAL_ERROR "${command} failed: ${result}")
  endif()
endfunction()

# Configures, builds, and runs the consumer in <directory> with the extra cache settings given.
function(build_consumer directory)
  run("${CMAKE_COMMAND}" -S "${SOURCE}" -B "${directory}" -G "${GENERATOR}"
    "-DCMAKE_CXX_COMPILER=${COMPILER}"
    "-DAGENT_SKILLS_DIRECTORY=${directory}/skills"
    ${ARGN}
  )
  run("${CMAKE_COMMAND}" --build "${directory}")
  file(GLOB program "${directory}/estimator" "${directory}/estimator.exe")
  run(${program})
  if(EXISTS "${directory}/skills")
    message(FATAL_ERROR "the build installed the skills without being asked")
  endif()
  run("${CMAKE_COMMAND}" --build "${directory}" --target emrys_filter_skills)
  if(NOT EXISTS "${directory}/skills/emrys-filter-build-estimator/SKILL.md")
    message(FATAL_ERROR "emrys_filter_skills installed no skills in ${directory}/skills")
  endif()
endfunction()

# Reads <variable> from the CMake cache in <directory> into <out>.
function(cached directory variable out)
  file(STRINGS "${directory}/CMakeCache.txt" line REGEX "^${variable}:[A-Z]+=" LIMIT_COUNT 1)
  string(REGEX REPLACE "^[^=]*=" "" value "${line}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

file(REMOVE_RECURSE "${BINARY}")
# Given Eigen's source, the library fetches it rather than finding an installed one, so that the
# install below puts Eigen beside it.
set(eigen)
if(EIGEN)
  set(eigen "-DFETCHCONTENT_SOURCE_DIR_EIGEN3=${EIGEN}" -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER)
endif()

# Fetched: the checkout as the consumer's FetchContent source, never an installed EmrysFilter.
build_consumer("${BINARY}/fetched"
  -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
  "-DFETCHCONTENT_SOURCE_DIR_EMRYSFILTER=${LIBRARY}"
  ${eigen}
)

# Installed: the checkout configured on its own, installed into a prefix, and found there. The
# user's package registry, which may hold other builds' Eigen, is ignored here and below: an Eigen
# found through it is installed nowhere, so the consumer, which ignores it too, would find none.
set(prefix "${BINARY}/prefix")
run("${CMAKE_COMMAND}" -S "${LIBRARY}" -B "${BINARY}/library" -G "${GENERATOR}"
  "-DCMAKE_CXX_COMPILER=${COMPILER}"
  -DEMRYS_FILTER_BUILD_TESTS=OFF
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  ${eigen}
)
run("${CMAKE_COMMAND}" --install "${BINARY}/library" --prefix "${prefix}")
# Disconnected, a consumer that doesn't find the install fails rather than cloning the repository.
build_consumer("${BINARY}/installed"
  -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  "-DCMAKE_PREFIX_PATH=${prefix}"
)
cached("${BINARY}/installed" EmrysFilter_DIR emrysFilterDir)
cmake_path(IS_PREFIX prefix "${emrysFilterDir}" NORMALIZE found)
if(NOT found)
  message(FATAL_ERROR "the consumer found EmrysFilter in ${emrysFilterDir}, not in ${prefix}")
endif()
if(EIGEN)
  cached("${BINARY}/installed" Eigen3_DIR eigenDir)
  cmake_path(IS_PREFIX prefix "${eigenDir}" NORMALIZE found)
  if(NOT found)
    message(FATAL_ERROR "the consumer found Eigen in ${eigenDir}, not beside EmrysFilter")
  endif()
endif()
