set(VCPKG_BUILD_TYPE release) # header-only: nothing to build twice

# The source is fetched at the release tag. The SHA512 is the tag's archive, filled in when the
# tag exists. Until then, a test copy of this port replaces this call with the working tree.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO emrys-math/emrys-filter
    REF "v${VERSION}"
    SHA512 0
    HEAD_REF main
)

# The library fetches Eigen when it finds none. vcpkg's eigen3 is the one it must find, so
# a fetch fails the build here rather than downloading a second Eigen.
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DEMRYS_FILTER_INSTALL=ON
        -DEMRYS_FILTER_BUILD_TESTS=OFF
        -DEMRYS_FILTER_BUILD_EXAMPLES=OFF
        -DEMRYS_FILTER_BUILD_DOCS=OFF
        -DEMRYS_FILTER_BUILD_BENCHMARKS=OFF
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON
        -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS
)
vcpkg_cmake_install()

# The package installs to share/EmrysFilter/cmake, where find_package(EmrysFilter) looks without
# help, so vcpkg_cmake_config_fixup would only move it. share/EmrysFilter also holds the skills,
# usage.md, and the examples [cmake-package].

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
