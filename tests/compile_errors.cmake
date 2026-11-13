# Runs one negative compile test [coverage]: compiles SOURCE with the macro CASE defined, checking
# syntax only, and passes when the compile fails with the diagnostic the case names.
#
#   cmake -DCOMMANDS=<compile_commands.json> -DSOURCE=<file> -DCASE=<macro> -DSYNTAX_ONLY=<flag>
#         -DOBJECT=<SOURCE's object file> -DSTAMP=<file> -P compile_errors.cmake
#
# The compile command is the one the build uses for SOURCE, from compile_commands.json, so the
# flags, definitions, and include directories are a test's. A case is a block of SOURCE opened by
# `#if defined(CASE)` and a comment naming its claim, one of:
#
#   // static_assert: <header>: <text>   a static_assert of <header> fails with a message holding <text>
#   // deleted: <header>: <declaration>  a call selects the deleted <declaration> of <header>
#   // constraint: <header>: <concept>    a call to a function of <header> fails the constraint <concept>
#
# A compiler names the header in the diagnostic's location, so the header names the claim's site,
# and tools/check_compile_errors.py checks that each claim in the headers has a case.
#
# A case that passes writes STAMP, and passes again without compiling while STAMP is newer than
# OBJECT, SOURCE, and this script [check-stages]. A case includes nothing SOURCE doesn't, so the
# build recompiles OBJECT whenever a header, flag, or compiler the case depends on changes.

if(EXISTS "${STAMP}" AND NOT "${OBJECT}" IS_NEWER_THAN "${STAMP}"
   AND NOT "${SOURCE}" IS_NEWER_THAN "${STAMP}"
   AND NOT "${CMAKE_CURRENT_LIST_FILE}" IS_NEWER_THAN "${STAMP}")
  message(STATUS "${CASE} passed when last compiled, and nothing it depends on has changed")
  return()
endif()
file(REMOVE "${STAMP}")

file(READ "${COMMANDS}" commands)
string(JSON count LENGTH "${commands}")
math(EXPR last "${count} - 1")
file(REAL_PATH "${SOURCE}" source)
set(command "")
foreach(index RANGE ${last})
  string(JSON file GET "${commands}" ${index} file)
  file(REAL_PATH "${file}" file)
  if(file STREQUAL source)
    string(JSON command GET "${commands}" ${index} command)
    string(JSON directory GET "${commands}" ${index} directory)
    break()
  endif()
endforeach()
if(command STREQUAL "")
  message(FATAL_ERROR "${SOURCE} is not in ${COMMANDS}")
endif()

# The object file, its dependency file, and its PDB are the build's; checking syntax writes none.
separate_arguments(arguments NATIVE_COMMAND "${command}")
set(compile "")
set(skip OFF)
foreach(argument IN LISTS arguments)
  if(skip)
    set(skip OFF)
  elseif(argument MATCHES "^-(o|MF|MT|MQ)$")
    set(skip ON)
  elseif(argument MATCHES "^(-c|-MD|-MMD|/showIncludes)$" OR argument MATCHES "^[-/](Fo|Fd)")
  else()
    list(APPEND compile "${argument}")
  endif()
endforeach()

file(READ "${SOURCE}" text)
string(REGEX MATCH "\n#if defined\\(${CASE}\\)[^\r\n]*" opening "${text}")
if(NOT opening MATCHES "// (static_assert|deleted|constraint): ([^:]+): (.+)$")
  message(FATAL_ERROR "${SOURCE} has no case ${CASE} naming its claim")
endif()
set(kind "${CMAKE_MATCH_1}")
set(header "${CMAKE_MATCH_2}")
set(claim "${CMAKE_MATCH_3}")
cmake_path(GET header FILENAME header_name)

execute_process(
  COMMAND ${compile} "${SYNTAX_ONLY}" "-D${CASE}"
  WORKING_DIRECTORY "${directory}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output
)
if(result EQUAL 0)
  message(FATAL_ERROR "${CASE} compiled; it should fail with the ${kind} ${header}: ${claim}")
endif()

# The diagnostic names the header and the message, the concept, or the deleted function's name.
string(FIND "${output}" "${header_name}" at_header)
if(kind STREQUAL "static_assert" OR kind STREQUAL "constraint")
  string(FIND "${output}" "${claim}" at_claim)
else()
  # MSVC writes `operator =` where the declaration has `operator=`.
  string(REGEX MATCH "[A-Za-z_~][A-Za-z_0-9=]*" name "${claim}")
  string(REPLACE " " "" squashed "${output}")
  string(FIND "${squashed}" "${name}" at_claim)
  if(NOT output MATCHES "deleted")
    set(at_claim -1)
  endif()
endif()
if(at_header EQUAL -1 OR at_claim EQUAL -1)
  message(FATAL_ERROR "${CASE} failed without the ${kind} ${header}: ${claim}\n${output}")
endif()
file(WRITE "${STAMP}" "")
