# The Cortex-M build [embedded-builds]: the ranging benchmark's float tests for the Cortex-M4F and
# the Cortex-M7, with no heap, no exceptions, and no RTTI, run under QEMU, and their code size and
# stack use. A project of its own, built with the Arm GNU toolchain (arm-none-eabi.cmake):
#
#   cmake -S benchmarks/cortex_m -B build/cortex-m -G Ninja
#   cmake --build build/cortex-m
#   ctest --test-dir build/cortex-m --output-on-failure
#   cmake --build build/cortex-m --target footprint
cmake_minimum_required(VERSION 3.28)

if(NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/arm-none-eabi.cmake)
endif()

project(EmrysFilterCortexM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE MinSizeRel)
endif()

set(EMRYS_FILTER_BUILD_TESTS OFF)
set(EMRYS_FILTER_INSTALL OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. EmrysFilter EXCLUDE_FROM_ALL)

find_program(EMRYS_QEMU qemu-system-arm)
find_program(EMRYS_ARM_SIZE arm-none-eabi-size REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Each core: its compiler flags, then the QEMU machine that has it. The M4F's FPU is single
# precision, so double runs in software there; the M7's is double precision.
set(EMRYS_CORES m4 m7)
set(EMRYS_m4_FLAGS -mcpu=cortex-m4 -mfpu=fpv4-sp-d16)
set(EMRYS_m4_MACHINE mps2-an386)
set(EMRYS_m7_FLAGS -mcpu=cortex-m7 -mfpu=fpv5-d16)
set(EMRYS_m7_MACHINE mps2-an500)

enable_testing()
set(footprint_arguments)
foreach(core IN LISTS EMRYS_CORES)
  set(target ranging_${core})
  set(flags ${EMRYS_${core}_FLAGS} -mthumb -mfloat-abi=hard)
  add_executable(${target} ranging.cpp startup.cpp)
  set_target_properties(${target} PROPERTIES SUFFIX .elf)
  target_link_libraries(${target} PRIVATE EmrysFilter::EmrysFilter)
  # Eigen puts small scratch buffers on the stack with alloca where it knows the platform has it,
  # and otherwise on the heap; newlib declares alloca only outside strict ISO mode, so without this
  # a solve on a block whose size is known only at run time allocates. usage.md tells embedded
  # users the same.
  target_compile_definitions(${target} PRIVATE EIGEN_ALLOCA=__builtin_alloca)
  target_compile_options(${target} PRIVATE
    ${flags} -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections -fstack-usage
    -Wall -Wextra -Wpedantic -Werror -Wno-psabi
  )
  # No startup files: startup.cpp is the startup, and its _sbrk fails any program that allocates.
  target_link_options(${target} PRIVATE
    ${flags} -nostartfiles --specs=nano.specs -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
    -Wl,--gc-sections -Wl,-Map=$<TARGET_FILE_DIR:${target}>/${target}.map
  )
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/link.ld)

  if(EMRYS_QEMU)
    add_test(NAME cortex_m.ranging.${core}
      COMMAND ${EMRYS_QEMU} -machine ${EMRYS_${core}_MACHINE} -nographic -monitor none -serial none
        -semihosting-config enable=on,target=native -kernel $<TARGET_FILE:${target}>
    )
    set_tests_properties(cortex_m.ranging.${core} PROPERTIES TIMEOUT 900)
  endif()
  list(APPEND footprint_arguments ${core} $<TARGET_FILE:${target}> ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir)
endforeach()

# Code size and stack use per core, as a Markdown table.
add_custom_target(footprint
  COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/footprint.py ${EMRYS_ARM_SIZE} ${footprint_arguments}
  DEPENDS ranging_m4 ranging_m7
  VERBATIM
)
