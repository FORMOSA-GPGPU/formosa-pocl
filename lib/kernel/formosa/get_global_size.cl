size_t _CL_OVERLOADABLE get_global_size(uint dim) {
    switch(dim) {
        case 0:
            return __builtin_riscv_fsa_global_size_x();
        case 1:
            return __builtin_riscv_fsa_global_size_y();
        case 2:
            return __builtin_riscv_fsa_global_size_z();
        default:
            return 1;
    }
}

