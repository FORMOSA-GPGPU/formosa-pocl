uint64_t _CL_OVERLOADABLE get_local_size(uint);
uint64_t _CL_OVERLOADABLE get_group_id(uint);
uint64_t _CL_OVERLOADABLE get_local_id(uint);
uint64_t _CL_OVERLOADABLE get_global_offset(uint);

uint64_t _CL_OVERLOADABLE get_global_id(unsigned int dim) {
  switch (dim) {
  case 0:
    return get_local_size(0) * get_group_id(0) + get_local_id(0) +
           get_global_offset(0);
  case 1:
    return get_local_size(1) * get_group_id(1) + get_local_id(1) +
           get_global_offset(1);
  case 2:
    return get_local_size(2) * get_group_id(2) + get_local_id(2) +
           get_global_offset(2);
  default:
    return 0;
  }
}
