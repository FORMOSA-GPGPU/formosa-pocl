OUTPUT_ARCH(riscv)

MEMORY
{
  read_only  (rx): ORIGIN = 0x10000000, LENGTH = 0x08000000  /* 128 MiB */
  kargs      (r):  ORIGIN = 0x18000000, LENGTH = 0x08000000  /* 128 MiB */
  local_mem  (rw): ORIGIN = 0x20000000, LENGTH = 0x08000000  /* 128 MiB */
  global_mem (rw): ORIGIN = 0x40000000, LENGTH = 0x80000000  /*   2 GiB */
}

/* Stack information variables */
_min_stack   = 0x10000;   /* 64K - minimum stack space to reserve */
_stack_start = ORIGIN(local_mem) + LENGTH(local_mem);  /* the end of local_mem */

/* Kernel arguments */
_kargs = ORIGIN(kargs);

ENTRY(_start)

SECTIONS {

.text : {
  . = ALIGN(8);
  _start = .;
  *(.text)
  *(.text.*)
  . = ALIGN(8);
} > global_mem

.rodata : {
  . = ALIGN(8);
  /* Small RO data before large RO data */
  *(.srodata)
  *(.srodata.*)
  *(.rodata)
  *(.rodata.*)
  . = ALIGN(8);
} > global_mem

.data : {
  . = ALIGN(8);
  /* Small data before large data */
  *(.sdata)
  *(.sdata.*)
  *(.data);
  *(.data.*)
  . = ALIGN(8);
} > global_mem

.bss : {
  . = ALIGN(8);
  _bss_start = .;
  /* Small BSS before large BSS */
  *(.sbss)
  *(.sbss.*)
  *(.bss)
  *(.bss.*)
  *(COMMON)
  _bss_end = .;
  . = ALIGN(8);
} > global_mem

/* ensure there is enough room for stack */
.stack (NOLOAD): {
  . = ALIGN(8);
  . = . + _min_stack * 1024;  /* 32W-32T */
  . = ALIGN(8);
  stack = . ;
  _stack = . ;
} > local_mem

}

