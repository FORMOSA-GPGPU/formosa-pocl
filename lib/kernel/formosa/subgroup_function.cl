uint _CL_OVERLOADABLE get_sub_group_id() {
    size_t xlanes;
    __asm__ volatile("csrr %0, xlanes" : "=r"(xlanes));
    return __builtin_riscv_fsa_local_linear_id() / xlanes;
}

uint _CL_OVERLOADABLE get_sub_group_local_id() {
    return __builtin_riscv_fsa_lane_id();
}

uint _CL_OVERLOADABLE get_max_sub_group_size() {
    size_t xlanes;
    __asm__ volatile("csrr %0, xlanes" : "=r"(xlanes));
    return xlanes;
}

uint _CL_OVERLOADABLE get_sub_group_size() {
    return __builtin_riscv_fsa_sub_group_size();
}

uint _CL_OVERLOADABLE get_num_sub_groups() {
    size_t local_size = get_local_size(0) * get_local_size(1) * get_local_size(2);
    size_t xlanes;
    __asm__ volatile("csrr %0, xlanes" : "=r"(xlanes));
    return (local_size + xlanes - 1) / xlanes;
}

uint _CL_OVERLOADABLE get_enqueued_num_sub_groups() {
    // Return the same value as get_num_sub_groups() if the kernel is executed
    // with a uniform work-group size.
    // If the kernel is executed with a non-uniform work-group size, return the
    // number of sub-groups in each of the work-groups, that make up the
    // uniform region of the global range.
    return get_num_sub_groups();
}