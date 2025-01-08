.section .text

.org 0x0
_start:
  .global _start
  /* ensure all thread within a warp has the same priority */
  fsa.pri.reset

  /* set all registers to zero */
  mv  x1, x0
  mv  x2, x0
  mv  x3, x0
  mv  x4, x0
  mv  x5, x0
  mv  x6, x0
  mv  x7, x0
  mv  x8, x0
  mv  x9, x0
  mv x10, x0
  mv x11, x0
  mv x12, x0
  mv x13, x0
  mv x14, x0
  mv x15, x0
  mv x16, x0
  mv x17, x0
  mv x18, x0
  mv x19, x0
  mv x20, x0
  mv x21, x0
  mv x22, x0
  mv x23, x0
  mv x24, x0
  mv x25, x0
  mv x26, x0
  mv x27, x0
  mv x28, x0
  mv x29, x0
  mv x30, x0
  mv x31, x0

  /*
    calculate workitem local id
    workgroupThreadID = warpSize * warpID + warpThreadID
    workgroupThreadID.z = workgroupThreadID / (workgroupDim.x * workgroupDim.y)
    remaining = workgroupThreadID % (workgroupDim.x * workgroupDim.y)
    workgroupThreadID.y = remaining / workgroupDim.x
    workgroupThreadID.x = remaining % workgroupDim.x
  */
  # csrr t0, xwid     /* get warp id */
  # csrr t1, xnw      /* get number of warps */
  mul  t0, t0, t1
  csrr t1, xtid     /* get thread id in a warp */
  add  t0, t0, t1   /* thread id */
  csrr t1, xclsx    /* get workgroup size x */
  csrr t2, xclsy    /* get workgroup size y */
  mul  t2, t1, t2   /* workgroupDim.x * workgroupDim.y */
  div  t3, t0, t2   /* thread id x */
  csrw xcllz, t3    /* set thread id z */
  rem  t0, t0, t2   /* remaining */
  div  t3, t0, t1   /* thread id y */
  csrw xclly, t3    /* set thread id y */
  rem  t0, t0, t1   /* thread id x */
  csrw xcllx, t0    /* set thread id x */
  
  /* stack initilization */
  la sp, _stack_start
  csrr a0, mhartid  /* global thread id */
  slli a1, a0, 16   /* stack for each thread is 64 KiB */
  sub sp, sp, a1

main_entry:
  /* set argc and argv to 0 */
  # csrr t0, xkernelpc
  # csrr a0, xkernelarg
  li a1, 0

  /* jump to main program entry point */
  fsa.pri.raise.f
  jalr ra, t0, 0
  fsa.pri.lower.f

  /* use ecall to terminate kernel programs */
  ecall

/* If execution ends up here just put the core to sleep */
dead_loop:
  nop
  j dead_loop
