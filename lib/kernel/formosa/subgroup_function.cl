uint _CL_OVERLOADABLE get_sub_group_id() {
    size_t mhartid, xlanes;
    __asm__ volatile("csrr %0, mhartid" : "=r"(mhartid));
    __asm__ volatile("csrr %0, xlanes" : "=r"(xlanes));
    return mhartid / xlanes;
}

uint _CL_OVERLOADABLE get_sub_group_local_id() {
    size_t lane_id;
    __asm__ volatile("csrr %0, xlaneid" : "=r"(lane_id));
    return lane_id;
}

uint _CL_OVERLOADABLE get_max_sub_group_size() {
    size_t xlanes;
    __asm__ volatile("csrr %0, xlanes" : "=r"(xlanes));
    return xlanes;
}

uint _CL_OVERLOADABLE get_sub_group_size() {
    size_t mhartid, xlanes;
    __asm__ volatile("csrr %0, mhartid" : "=r"(mhartid));
    __asm__ volatile("csrr %0, xlanes" : "=r"(xlanes));
    size_t local_size = get_local_size(0) * get_local_size(1) * get_local_size(2);
    size_t sgid = mhartid / xlanes;
    size_t sg_begin = sgid * xlanes;
    size_t remaining = local_size - sg_begin;
    return remaining < xlanes ? remaining : xlanes;
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