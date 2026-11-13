import os

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout, CMakeDeps
from conan.tools.files import copy, get, rm, rmdir
from conan.tools.scm import Version

required_conan_version = ">=2.1"


class EmrysFilterConan(ConanFile):
    name = "emrys-filter"
    description = (
        "Header-only C++23 state estimation library on Eigen 5.0: Kalman, particle, and grid "
        "filters over typed state spaces."
    )
    license = "MIT"
    url = "https://github.com/conan-io/conan-center-index"
    homepage = "https://github.com/emrys-math/emrys-filter"
    topics = ("state-estimation", "kalman-filter", "particle-filter", "eigen", "header-only")
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    no_copy_source = True

    @property
    def _min_cppstd(self):
        return 23

    # The compilers that build the library's tests. `std::expected`, `std::forward_like`, and the
    # rest of C++23 that it uses set the floor (README.md). "msvc" 194 is Visual Studio 2022.
    @property
    def _compilers_minimum_version(self):
        return {
            "gcc": "14",
            "clang": "21",
            "apple-clang": "21",
            "msvc": "194",
        }

    def layout(self):
        cmake_layout(self, src_folder="src")

    def requirements(self):
        # The library accepts any Eigen 5.0 or later and rejects 3.4 at the first include.
        self.requires("eigen/[>=5.0.0 <6]", transitive_headers=True)

    def package_id(self):
        self.info.clear()

    def validate(self):
        check_min_cppstd(self, self._min_cppstd)
        minimum = self._compilers_minimum_version.get(str(self.settings.compiler))
        if minimum and Version(self.settings.compiler.version) < minimum:
            raise ConanInvalidConfiguration(
                f"{self.ref} requires {self.settings.compiler} {minimum} or later for C++23, "
                f"and this profile has {self.settings.compiler.version}."
            )

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        # Eigen comes from the dependency, so the project must not fetch its own.
        tc = CMakeToolchain(self)
        tc.variables["EMRYS_FILTER_INSTALL"] = True
        tc.variables["EMRYS_FILTER_BUILD_TESTS"] = False
        tc.variables["EMRYS_FILTER_BUILD_EXAMPLES"] = False
        tc.variables["EMRYS_FILTER_BUILD_DOCS"] = False
        tc.variables["EMRYS_FILTER_BUILD_BENCHMARKS"] = False
        tc.variables["FETCHCONTENT_FULLY_DISCONNECTED"] = True
        tc.variables["FETCHCONTENT_TRY_FIND_PACKAGE_MODE"] = "ALWAYS"
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()
        # The CMake package is for CMake users without Conan; CMakeDeps writes this package's own.
        # The skills function stays, as a build module, and so do the files the skills install from.
        package_dir = os.path.join(self.package_folder, "share", "EmrysFilter", "cmake")
        rm(self, "EmrysFilterConfig*.cmake", package_dir)
        rm(self, "EmrysFilterTargets*.cmake", package_dir)
        rmdir(self, os.path.join(self.package_folder, "lib"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "EmrysFilter")
        self.cpp_info.set_property("cmake_target_name", "EmrysFilter::EmrysFilter")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.set_property(
            "cmake_build_modules",
            [os.path.join("share", "EmrysFilter", "cmake", "EmrysFilterSkills.cmake")],
        )
