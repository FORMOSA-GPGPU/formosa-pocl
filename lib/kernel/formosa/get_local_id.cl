size_t _CL_OVERLOADABLE get_local_size(uint dim);

size_t _CL_OVERLOADABLE get_local_id(uint dim) {
  size_t mhartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(mhartid));

  size_t local_size_x = get_local_size(0);
  size_t local_size_y = get_local_size(1);
  size_t local_size_xy = local_size_x * local_size_y;
  size_t remaining = mhartid % local_size_xy;

  switch (dim) {
  case 0:
    return remaining % local_size_x;
  case 1:
    return remaining / local_size_x;
  case 2:
    return mhartid / local_size_xy;
  default:
    return 1;
  }
}
