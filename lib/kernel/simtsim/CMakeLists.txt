include("bitcode_rules")

set(KERNEL_SOURCES ${SOURCES_GENERIC})

foreach(FILE atomics.cl printf.c printf_base.c get_local_id.c get_local_size.c get_group_id.c
    get_global_size.c get_num_groups.c get_work_dim.c get_global_id.c
    get_global_offset.c barrier.ll erf.cl erfc.cl)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE get_local_id.cl get_local_size.cl get_group_id.cl get_global_size.cl
  barrier.cl get_num_groups.cl get_work_dim.cl get_global_id.cl get_global_offset.cl
  erf.cl erfc.cl)
  list(APPEND KERNEL_SOURCES "simtsim/${FILE}")
endforeach()

list(APPEND KERNEL_SOURCES "simtsim/printf_globals.c")
list(APPEND KERNEL_SOURCES "simtsim/printf_alloc.c")
list(APPEND KERNEL_SOURCES "simtsim/atomics.cl")

set(CLANG_FLAGS
    "-emit-llvm"
    "-target" "riscv64-unknown-unknown-elf"
    "-mcpu=formosa-gpgpu"
    "-Xclang" "-target-feature"
    "-Xclang" "+zaamo")
set(LLC_FLAGS "")

set(KERNEL_CL_FLAGS "-Xclang"
                    "-cl-std=CL1.2"
                    "-Xclang"
                    "-cl-ext=-all,+cl_khr_byte_addressable_store,+cl_khr_global_int32_base_atomics,+cl_khr_global_int32_extended_atomics,+cl_khr_int64,+cl_khr_int64_base_atomics,+cl_khr_int64_extended_atomics,+cl_khr_fp64"
                    "-D__OPENCL_C_VERSION__=120"
                    "-Dcl_khr_global_int32_base_atomics"
                    "-Dcl_khr_global_int32_extended_atomics"
                    "-Dcl_khr_int64"
                    "-Dcl_khr_int64_base_atomics"
                    "-Dcl_khr_int64_extended_atomics"
                    "-Dcl_khr_fp64"
                    "-DPOCL_DEVICE_ADDRESS_BITS=64"
                    ${KERNEL_CL_FLAGS})

set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=120")

make_kernel_bc(KERNEL_BC "riscv64-simtsim" "simtsim" 0 0 0 ${KERNEL_SOURCES})

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

add_custom_target("kernel_riscv64-simtsim" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_riscv64-simtsim")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")

install(FILES "kernel_util.cl" "wg_info.h" "link.ld" "simtsim_start.S" "errno_stub.c"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/simtsim"
        COMPONENT "lib")
