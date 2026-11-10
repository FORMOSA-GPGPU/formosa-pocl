size_t _CL_OVERLOADABLE get_local_size(uint);
size_t _CL_OVERLOADABLE get_group_id(uint);
size_t _CL_OVERLOADABLE get_local_id(uint);
size_t _CL_OVERLOADABLE get_global_size(uint);

size_t _CL_OVERLOADABLE get_global_linear_id(void) {
  size_t x = get_local_size(0) * get_group_id(0) + get_local_id(0);
  size_t y = get_local_size(1) * get_group_id(1) + get_local_id(1);
  size_t z = get_local_size(2) * get_group_id(2) + get_local_id(2);
  return (z * get_global_size(1) + y) * get_global_size(0) + x;
}

/* mhartid holds the local linear id, with x varying fastest. */
size_t _CL_OVERLOADABLE get_local_linear_id(void) {
  return __builtin_riscv_fsa_local_linear_id();
}
