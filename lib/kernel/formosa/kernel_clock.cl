ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_device(void) {
  ulong clk;
  __asm__ volatile("csrr %0, cycle" : "=r"(clk));
  return clk;
}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_work_group(void) {
  ulong clk;
  __asm__ volatile("csrr %0, cycle" : "=r"(clk));
  return clk;
}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_sub_group(void) {
  ulong clk;
  __asm__ volatile("csrr %0, cycle" : "=r"(clk));
  return clk;
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_device(void) {
  ulong clk;
  __asm__ volatile("csrr %0, cycle" : "=r"(clk));
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_work_group(void) {
  ulong clk;
  __asm__ volatile("csrr %0, cycle" : "=r"(clk));
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_sub_group(void) {
  ulong clk;
  __asm__ volatile("csrr %0, cycle" : "=r"(clk));
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}
