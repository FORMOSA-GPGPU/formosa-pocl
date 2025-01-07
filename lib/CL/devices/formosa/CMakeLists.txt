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
                               falloc/fsa_mem_allocator.c
                               falloc/fsa_mem_allocator.h
                               PROPERTIES LANGUAGE C )
  set_source_files_properties( pocl-formosa-util.cc
                               pocl-formosa-util.h PROPERTIES LANGUAGE CXX )
endif(MSVC)

# this is required, because we're compiling the C++ files of this directory
# with LLVM-provided flags, which usually contain their own setting of
# C++ standard, which may be in conflict with pocl's default C++ standard.
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
set_source_files_properties(pocl-formosa-util.cc pocl-formosa-util.h
                            PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
include_directories(${LLVM_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)
set(CMAKE_CXX_STANDARD ${POCL_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(FORMOSA_LLVM="${LLVM_PREFIX}")
add_pocl_device_library(pocl-devices-formosa
                        pocl-formosa.c pocl-formosa.h
                        pocl-formosa-util.cc pocl-formosa-util.h
                        falloc/fsa_mem_allocator.c falloc/fsa_mem_allocator.h)

if(ENABLE_LOADABLE_DRIVERS)
  # Add runtime library
  target_link_libraries(pocl-devices-formosa PRIVATE ${PTHREAD_LIBRARY})
endif()

message(STATUS "CASVP_INSTALL_DIR: ${CASVP_INSTALL_DIR}")
target_include_directories(pocl-devices-formosa PRIVATE
                           ${CASVP_INSTALL_DIR}/include)

target_link_libraries(pocl-devices-formosa PRIVATE comm)

install(FILES "kernel_util.cl" "start.S"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/formosa"
        COMPONENT "lib")
install(DIRECTORY "${CASVP_INSTALL_DIR}/include"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/formosa"
        COMPONENT "lib")
