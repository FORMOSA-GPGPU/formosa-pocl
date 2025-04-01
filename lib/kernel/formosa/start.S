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
    local linear thread id = mhartid - mhartidbase
    local id z (xcllz) = linear thread id / (group size x * group size y)
    remaining = linear thread id % (group size x * group size y)
    local id y (xclly) = remaining / group size x
    local id x (xcllx) = remaining % group size x
  */
  csrr t0, mhartid        /* global linear thread id */
  csrr t1, xmhartidbase   /* global linear thread id base */
  sub  t0, t0, t1         /* local linear thread id */
  csrr t1, xclsx          /* get workgroup size x */
  csrr t2, xclsy          /* get workgroup size y */
  mul  t2, t1, t2         /* group size x * group size y */
  div  t3, t0, t2         /* local id x */
  csrw xcllz, t3          /* set local id z */
  rem  t0, t0, t2         /* remaining */
  div  t3, t0, t1         /* local id y */
  csrw xclly, t3          /* set local id y */
  rem  t0, t0, t1         /* local id x */
  csrw xcllx, t0          /* set local id x */

  /* stack initilization */
  la sp, _stack_start
  csrr a0, mhartid  /* global thread id */
  slli a1, a0, 10   /* stack for each thread is 1 KiB */
  sub sp, sp, a1

main_entry:
  /* set argc and argv to 0 */
  csrr t0, xkernelpc
  csrr a0, xkernelarg
  li a1, 0

  /* jump to main program entry point */
  fsa.pri.raise.f
  jalr ra, t0, 0
  fsa.pri.lower.f

  /* flush all caches */
  // fence  // TODO: uncomment this line when fence instruction is supported

  /* use ecall to terminate kernel programs */
  ecall

/* If execution ends up here just put the core to sleep */
dead_loop:
  nop
  j dead_loop
