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

foreach(FILE atomics.cl printf.c printf_base.c get_local_id.c get_local_size.c get_group_id.c
    get_global_size.c get_num_groups.c get_work_dim.c get_global_id.c
    get_global_offset.c barrier.ll erf.cl erfc.cl kernel_clock.cl)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE get_local_id.cl get_local_size.cl get_group_id.cl get_global_size.cl
  barrier.cl get_num_groups.cl get_work_dim.cl get_global_id.cl get_global_offset.cl
  erf.cl erfc.cl subgroup_function.cl kernel_clock.cl)
  list(APPEND KERNEL_SOURCES "formosa/${FILE}")
endforeach()

if(ENABLE_FORMOSA_WORKGRAPH)
  list(APPEND KERNEL_SOURCES "formosa/workgraph/work_graph.cl")
endif()

list(APPEND KERNEL_SOURCES "formosa/printf_globals.c")
list(APPEND KERNEL_SOURCES "formosa/printf_alloc.c")
list(APPEND KERNEL_SOURCES "formosa/atomics.cl")

if(NOT DEFINED FORMOSA_DEVICE_EXTENSIONS)
  set(FORMOSA_DEVICE_EXTENSIONS "")
endif()

if(NOT DEFINED FORMOSA_DEVICE_FEATURES_30)
  set(FORMOSA_DEVICE_FEATURES_30 "")
endif()

set(FORMOSA_DEVICE_EXTENSIONS_LIST "${FORMOSA_DEVICE_EXTENSIONS}")
set(FORMOSA_DEVICE_FEATURES_LIST "${FORMOSA_DEVICE_FEATURES_30}")

separate_arguments(FORMOSA_DEVICE_EXTENSIONS_LIST)
separate_arguments(FORMOSA_DEVICE_FEATURES_LIST)

# -----------------------------------------------------------------------------
# Build -cl-ext flag
# -----------------------------------------------------------------------------

set(FORMOSA_CL_EXT_FLAGS "-all")

foreach(EXT IN LISTS FORMOSA_DEVICE_EXTENSIONS_LIST)
  if(NOT EXT STREQUAL "")
    string(APPEND FORMOSA_CL_EXT_FLAGS ",+${EXT}")
  endif()
endforeach()

foreach(FEATURE IN LISTS FORMOSA_DEVICE_FEATURES_LIST)
  if(NOT FEATURE STREQUAL "")
    string(APPEND FORMOSA_CL_EXT_FLAGS ",+${FEATURE}")
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Build -D flags
# -----------------------------------------------------------------------------

set(FORMOSA_DEVICE_EXTENSION_DEFINES "")
set(FORMOSA_DEVICE_FEATURE_DEFINES "")

foreach(EXT IN LISTS FORMOSA_DEVICE_EXTENSIONS_LIST)
  if(NOT EXT STREQUAL "")
    list(APPEND FORMOSA_DEVICE_EXTENSION_DEFINES "-D${EXT}=1")
  endif()
endforeach()

foreach(FEATURE IN LISTS FORMOSA_DEVICE_FEATURES_LIST)
  if(NOT FEATURE STREQUAL "")
    list(APPEND FORMOSA_DEVICE_FEATURE_DEFINES "-D${FEATURE}=1")
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------

set(CLANG_FLAGS
    "-emit-llvm"
    "-target" "riscv64-unknown-unknown-elf"
    "-mcpu=formosa-gpgpu"
    "-Xclang" "-target-feature"
    "-Xclang" "+zaamo")
set(LLC_FLAGS "")

set(KERNEL_CL_FLAGS
    "-Xclang"
    "-cl-std=CL${FORMOSA_DEVICE_CL_STD}"
    "-Xclang"
    "-cl-ext=${FORMOSA_CL_EXT_FLAGS}"
    "-D__OPENCL_C_VERSION__=${FORMOSA_DEVICE_CL_VERSION}"
    "-DPOCL_DEVICE_ADDRESS_BITS=64"
)

list(APPEND KERNEL_CL_FLAGS ${FORMOSA_DEVICE_EXTENSION_DEFINES})
list(APPEND KERNEL_CL_FLAGS ${FORMOSA_DEVICE_FEATURE_DEFINES})

set(DEVICE_CL_FLAGS
    "-D__OPENCL_VERSION__=${FORMOSA_DEVICE_CL_VERSION}"
    "-DPOCL_DEVICE_ADDRESS_BITS=64"
)

list(APPEND DEVICE_CL_FLAGS ${FORMOSA_DEVICE_EXTENSION_DEFINES})
list(APPEND DEVICE_CL_FLAGS ${FORMOSA_DEVICE_FEATURE_DEFINES})


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

install(FILES "kernel_util.cl" "wg_info.h" "start.S" "link.ld"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/formosa"
        COMPONENT "lib")
