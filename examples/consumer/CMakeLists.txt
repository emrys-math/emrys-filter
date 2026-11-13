# A project that uses EmrysFilter, to copy and build on. It finds an installed EmrysFilter or
# fetches the library, builds main.cpp with short template errors, and installs EmrysFilter's
# skills for a coding agent.
cmake_minimum_required(VERSION 3.28)
project(Estimator LANGUAGES CXX)

# The library's assertions catch what its types can't, so build in Debug until the estimator
# passes its checks.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# An EmrysFilter 0.1 installed where find_package looks, or under CMAKE_PREFIX_PATH, comes first;
# without one, the library is fetched. To build against a local checkout instead, configure with
# -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER -DFETCHCONTENT_SOURCE_DIR_EMRYSFILTER=<checkout>.
include(FetchContent)
FetchContent_Declare(EmrysFilter
  GIT_REPOSITORY https://github.com/emrys-math/emrys-filter.git
  GIT_TAG v0.1.0
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS 0.1 CONFIG
)
FetchContent_MakeAvailable(EmrysFilter)

add_executable(estimator main.cpp)
target_link_libraries(estimator PRIVATE EmrysFilter::EmrysFilter)

# A template error's first lines name the cause and its last the line of yours that caused it, so
# GCC and Clang stop after three errors and print only the ends of each instantiation chain. At
# -O0 they compile and run Eigen slowly, so Debug builds with -Og, which keeps the assertions.
set(gnu_like "$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},GNU>>")
target_compile_options(estimator PRIVATE
  "$<${gnu_like}:-ftemplate-backtrace-limit=4;$<$<CONFIG:Debug>:-Og>>"
  "$<$<AND:${gnu_like},$<CXX_COMPILER_ID:GNU>>:-fmax-errors=3>"
  "$<$<AND:${gnu_like},$<NOT:$<CXX_COMPILER_ID:GNU>>>:-ferror-limit=3>"
)

# cmake --build <build directory> --target emrys_filter_skills installs the skills, and no other
# build does. Claude Code reads .claude/skills; Codex and Gemini CLI read .agents/skills.
set(AGENT_SKILLS_DIRECTORY .claude/skills CACHE STRING
  "Where to install EmrysFilter's skills: a coding agent's skills directory, or a list of them")
emrys_filter_add_skills_target(DESTINATION ${AGENT_SKILLS_DIRECTORY})
