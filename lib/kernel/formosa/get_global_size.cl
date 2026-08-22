size_t _CL_OVERLOADABLE get_local_size(uint dim);
size_t _CL_OVERLOADABLE get_num_groups(uint dim);

size_t _CL_OVERLOADABLE get_global_size(uint dim) {
  size_t local_size = get_local_size(dim);
  size_t num_groups = get_num_groups(dim);
  return local_size * num_groups;
}
