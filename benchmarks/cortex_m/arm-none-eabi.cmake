# The Arm GNU toolchain for bare-metal Cortex-M (decisions.md [embedded-builds]). The core's flags
# are per target, in CMakeLists.txt beside this file, so one build makes every core.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
# A test program can't link without the startup code, so the compiler checks build a library.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
