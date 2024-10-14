size_t _CL_OVERLOADABLE get_num_groups(uint dim);
size_t _CL_OVERLOADABLE get_local_size(uint dim);


size_t _CL_OVERLOADABLE get_global_size(uint dim) {
  return get_num_groups(dim) * get_local_size(dim);
}

