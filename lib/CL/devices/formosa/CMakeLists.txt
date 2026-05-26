#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2024 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

if(MSVC)
  set_source_files_properties( pocl-formosa.c pocl-formosa.h 
                               PROPERTIES LANGUAGE C )
  set_source_files_properties( formosa-util.cc formosa-util.h
                               formosa-llvm-util.cc formosa-llvm-util.h 
                               PROPERTIES LANGUAGE CXX )
endif(MSVC)

# this is required, because we're compiling the C++ files of this directory
# with LLVM-provided flags, which usually contain their own setting of
# C++ standard, which may be in conflict with pocl's default C++ standard.
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
set(CMAKE_CXX_STANDARD ${POCL_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FORMOSA_CLANG_PATH "" CACHE FILEPATH
    "Path to the FORMOSA kernel compiler wrapper")
if(NOT FORMOSA_CLANG_PATH)
  find_program(FORMOSA_CLANG_PATH NAMES formosa-clang)
endif()

add_compile_definitions(FORMOSA_LLVM="${LLVM_PREFIX}")
if(FORMOSA_CLANG_PATH)
  add_compile_definitions(FORMOSA_CLANG_PATH="${FORMOSA_CLANG_PATH}")
endif()
add_pocl_device_library(pocl-devices-formosa
                        pocl-formosa.c pocl-formosa.h
                        pocl-formosa-graph.c
                        formosa-llvm-util.cc formosa-llvm-util.h
                        formosa-util.cc formosa-util.h)

if(ENABLE_LOADABLE_DRIVERS)
  # Add runtime library
  target_link_libraries(pocl-devices-formosa PRIVATE ${PTHREAD_LIBRARY})
endif()

set(FORMOSA_PLATFORM "lv" CACHE STRING "Platform for FORMOSA HAL, default to lv")
message(STATUS "FORMOSA_PLATFORM: ${FORMOSA_PLATFORM}")

if(CMAKE_BUILD_TYPE STREQUAL Release)
  find_package(Formosa REQUIRED)
  target_link_libraries(pocl-devices-formosa PUBLIC Formosa::formosa-hal.${FORMOSA_PLATFORM})
else()
  if(NOT FORMOSA_SOURCE_DIR)
    message(FATAL_ERROR "FORMOSA_SOURCE_DIR is not set.\nUse -D FORMOSA_SOURCE_DIR=/absolute/path/to/formosa to define the path.")
  endif()
  set(ENABLE_PROJECTS hal)
  add_subdirectory(${FORMOSA_SOURCE_DIR} formosa-build)

  # Prioritize source headers
  target_include_directories(pocl-devices-formosa BEFORE PRIVATE
                             "${FORMOSA_SOURCE_DIR}/hal/include"
                             "${FORMOSA_SOURCE_DIR}/fw")

  target_link_libraries(pocl-devices-formosa PUBLIC formosa-hal.${FORMOSA_PLATFORM})
endif()

target_include_directories(pocl-devices-formosa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)
target_compile_options(pocl-devices-formosa PRIVATE -I${LLVM_INCLUDE_DIRS})
