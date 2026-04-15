#define WG_INFO_DIM 0x0
#define WG_INFO_OFFSET_WGID 0x4
#define WG_INFO_OFFSET_LOCAL_SIZE 0x10
#define WG_INFO_OFFSET_NUM_GROUPS 0x1C
#define WG_INFO_OFFSET_GLOBAL_OFFSET 0x28
#define WG_INFO_OFFSET_NUM_THREADS 0x34
#define WG_INFO_OFFSET_TRAMPOLINE 0x38
#define WG_INFO_OFFSET_KARG 0x40
#define WG_INFO_OFFSET_STACK_BASE 0x48
#define WG_INFO_OFFSET_STACK_SIZE 0x50
#define WG_INFO_OFFSET_LOCAL_MEM_BASE 0x58
#define WG_INFO_OFFSET_LOCAL_MEM_SIZE 0x60
#define WG_INFO_OFFSET_PRINTF_BUFFER 0x68
#define WG_INFO_OFFSET_PRINTF_BUFFER_POSITION 0x70
#define WG_INFO_OFFSET_PRINTF_BUFFER_CAPACITY 0x78

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
    Stack Initialization:

    Each thread is allocated 1 KiB of stack space.
    The stack pointer (sp) is calculated as:
        sp = global_mem_base + (mhartid + 1) * 1 KiB

    This ensures each thread gets a unique, non-overlapping stack region.
  */

  // Read the wg_info
  csrr t0, mscratch
  /* stack initilization */
  csrr t1, mhartid
  addi t1, t1, 1
  lw t2, WG_INFO_OFFSET_STACK_SIZE(t0)
  mul t2, t1, t2  // t2 = (mhartid + 1) * stack_size
  ld t1, WG_INFO_OFFSET_STACK_BASE(t0)  // load stack base address
  add sp, t1, t2

main_entry:
  // Read trampoline
  ld t1, WG_INFO_OFFSET_TRAMPOLINE(t0)
  // Read kernel argument
  ld a0, WG_INFO_OFFSET_KARG(t0)

  // Seed the generic PoCL device-side printf globals from WGInfo.
  la t2, _printf_buffer
  ld t3, WG_INFO_OFFSET_PRINTF_BUFFER(t0)
  sd t3, 0(t2)

  la t2, _printf_buffer_position
  ld t3, WG_INFO_OFFSET_PRINTF_BUFFER_POSITION(t0)
  sd t3, 0(t2)

  la t2, _printf_buffer_capacity
  lw t3, WG_INFO_OFFSET_PRINTF_BUFFER_CAPACITY(t0)
  sw t3, 0(t2)

  /* jump to main program entry point */
  jalr ra, t1, 0

  /* flush all caches */
  // fence  // TODO: uncomment this line when fence instruction is supported

  /* use ecall to terminate kernel programs */
  ecall

/* If execution ends up here just put the core to sleep */
dead_loop:
  nop
  j dead_loop
