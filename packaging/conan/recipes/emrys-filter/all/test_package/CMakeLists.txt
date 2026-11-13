cmake_minimum_required(VERSION 3.28)
project(test_package LANGUAGES CXX)

find_package(EmrysFilter CONFIG REQUIRED)

add_executable(test_package test_package.cpp)
target_link_libraries(test_package PRIVATE EmrysFilter::EmrysFilter)
set_target_properties(test_package PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

# The package's build module defines this function, so a Conan user can install the skills too.
emrys_filter_add_skills_target(DESTINATION skills)
