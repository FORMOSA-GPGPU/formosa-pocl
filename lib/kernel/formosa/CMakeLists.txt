#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2024 pocl developers
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

include("bitcode_rules")

set(KERNEL_SOURCES ${SOURCES_GENERIC})

foreach(FILE printf.c printf_base.c get_local_id.c get_local_size.c get_group_id.c
    get_global_size.c get_num_groups.c get_work_dim.c get_global_id.c
    get_global_offset.c barrier.ll)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE get_local_id.cl get_local_size.cl get_group_id.cl get_global_size.cl
  barrier.cl get_num_groups.cl get_work_dim.cl get_global_id.c get_global_offset.cl)
  list(APPEND KERNEL_SOURCES "formosa/${FILE}")
endforeach()

set(CLANG_FLAGS "-emit-llvm" "-target" "riscv64-unknown-unknown-elf" "-mcpu=formosa-gpgpu")
set(LLC_FLAGS "")

set(KERNEL_CL_FLAGS "-Xclang"
                    "-cl-std=CL${FORMOSA_DEVICE_CL_STD}"
                    "-Xclang" 
                    "-cl-ext=-all,+cl_khr_byte_addressable_store,+cl_khr_int64,+cl_khr_fp64"
                    "-D__OPENCL_C_VERSION__=${FORMOSA_DEVICE_CL_VERSION}"
                    ${KERNEL_CL_FLAGS})

set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${FORMOSA_DEVICE_CL_VERSION}")

make_kernel_bc(KERNEL_BC "riscv64-formosa" "formosa" 0 0 0 ${KERNEL_SOURCES})

# just debug
message(STATUS "FORMOSA Kernel BC: ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_riscv64-formosa" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_riscv64-formosa")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")

install(FILES "kernel_util.cl" "start.S" "link.ld"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/formosa"
        COMPONENT "lib")
