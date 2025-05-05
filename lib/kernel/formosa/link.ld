OUTPUT_ARCH(riscv)

MEMORY
{
  global_mem (rwx): ORIGIN = 0x00000000, LENGTH = 0x80000000  /*   2 GiB */
}

/* Stack information variables */
_min_stack   = 0x400;   /* 1K - minimum stack space to reserve */

/* Kernel arguments */

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

}