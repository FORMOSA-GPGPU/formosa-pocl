#include "casvp-config/casvp-config.h"
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

  /* stack initilization */
  # la sp, _stack_start
  csrr a0, mhartid  /* global thread id */
  slli a1, a0, 16   /* stack for each thread is 64 KiB */
  sub sp, sp, a1

  /* clear BSS */
  fsa.pri.raise.t
  bnez a0, zero_loop_end  /* thread 0 is designated to clear BSS */

  # la a0, _bss_start
  # la a1, _bss_end

  bge a0, a1, zero_loop_end

zero_loop:
  sd x0, 0(a0)
  addi a0, a0, 8
  ble a0, a1, zero_loop

zero_loop_end:
  fsa.pri.lower.t
  fsa.bar 0, 0  /* wait until all warps reach here */

main_entry:
  /* set argc and argv to 0 */
  csrr t0, CASVP_FORMOSA_CSR_KERNEL_PC
  csrr a0, CASVP_FORMOSA_CSR_KERNEL_ARG
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
