void _CL_OVERLOADABLE barrier(cl_mem_fence_flags flags) {
  // The 'flags' argument is not used here because the FSA barrier
  // instruction does not distinguish between global and local memory
  // fences.
  __builtin_riscv_fsa_barrier(0, 0);
}

// The SM implements all requested fence orders by draining its memory stream.
// The memory clobber also keeps the compiler from moving accesses across it.
void _CL_OVERLOADABLE mem_fence(cl_mem_fence_flags flags) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

void _CL_OVERLOADABLE read_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}

void _CL_OVERLOADABLE write_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}

void _CL_OVERLOADABLE atomic_work_item_fence(cl_mem_fence_flags flags,
                                             memory_order order,
                                             memory_scope scope) {
  mem_fence(flags);
}

void _CL_OVERLOADABLE work_group_barrier(cl_mem_fence_flags flags) {
  barrier(flags);
}

void _CL_OVERLOADABLE work_group_barrier(cl_mem_fence_flags flags,
                                         memory_scope scope) {
  barrier(flags);
}
