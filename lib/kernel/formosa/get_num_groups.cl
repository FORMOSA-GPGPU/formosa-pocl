size_t _CL_OVERLOADABLE get_num_groups(uint dim) {
  size_t global_size, local_size;
  switch (dim) {
  case 0:
    global_size = __builtin_riscv_fsa_global_size_x();
    local_size = __builtin_riscv_fsa_local_size_x();
    break;
  case 1:
    global_size = __builtin_riscv_fsa_global_size_y();
    local_size = __builtin_riscv_fsa_local_size_y();
    break;
  case 2:
    global_size = __builtin_riscv_fsa_global_size_z();
    local_size = __builtin_riscv_fsa_local_size_z();
    break;
  default:
    return 1;
  }
  return (global_size + local_size - 1) / local_size;
}
