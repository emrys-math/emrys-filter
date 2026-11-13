# The reference documentation: Doxygen over every header (roadmap step 13b). `cmake --build <dir>
# --target docs` writes HTML to <dir>/docs/html and the warnings to <dir>/docs/warnings.txt.
#
# With EMRYS_FILTER_DOCS_INTERNALS on, the default, it is the maintainer's reference: private
# members, anonymous namespaces, and the `internal` namespace are included, since it is for reading
# the library, not only using it, and its XML is what tools/check_policies.py and tools/coverage.py
# read. Off, it is the published reference on GitHub Pages (.github/workflows/pages.yml): the
# public API alone.

find_package(Doxygen REQUIRED)

# The comments use their first sentence as the brief and describe parameters in prose, so the
# brief is automatic and a missing \param is not a warning.
set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
set(DOXYGEN_PROJECT_NUMBER "${PROJECT_VERSION}")
set(DOXYGEN_PROJECT_BRIEF "${PROJECT_DESCRIPTION}")
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}/include")
set(DOXYGEN_STRIP_FROM_INC_PATH "${PROJECT_SOURCE_DIR}/include")
set(DOXYGEN_FILE_PATTERNS *.h *.md)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_EXTRACT_STATIC YES)
if(EMRYS_FILTER_DOCS_INTERNALS)
  set(DOXYGEN_EXTRACT_PRIVATE YES)
  set(DOXYGEN_EXTRACT_ANON_NSPACES YES)
  set(DOXYGEN_INTERNAL_DOCS YES)
  # The main page's \if maintainer text.
  set(DOXYGEN_ENABLED_SECTIONS maintainer)
else()
  set(DOXYGEN_EXTRACT_PRIVATE NO)
  set(DOXYGEN_EXTRACT_ANON_NSPACES NO)
  set(DOXYGEN_INTERNAL_DOCS NO)
  set(DOXYGEN_EXCLUDE_SYMBOLS internal Emrys::internal)
endif()
set(DOXYGEN_HIDE_UNDOC_MEMBERS NO)
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF NO)
set(DOXYGEN_PREDEFINED "EMRYS_FILTER_NO_UNIQUE_ADDRESS=" "EMRYS_FILTER_ASSERT(x)=")
set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_IF_INCOMPLETE_DOC YES)
set(DOXYGEN_WARN_NO_PARAMDOC NO)
set(DOXYGEN_WARN_LOGFILE "${CMAKE_CURRENT_BINARY_DIR}/warnings.txt")
# Every entity is documented, so a warning is a regression: the build fails after writing them all.
set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_LATEX NO)
# The XML lists the entities that tools/coverage.py checks the tests instantiate [coverage].
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_HAVE_DOT NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/reference/mainpage.md")

doxygen_add_docs(docs
  "${PROJECT_SOURCE_DIR}/include/EmrysFilter"
  "${CMAKE_CURRENT_SOURCE_DIR}/reference"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Generating the reference documentation with Doxygen"
)
