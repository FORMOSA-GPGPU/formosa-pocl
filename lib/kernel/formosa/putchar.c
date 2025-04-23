#include <stdint.h>

#define HART_CONSOLE_BASE 0x00002000

void _putchar(char character) {
    // Use mhartid to determine which output address to write to
    uint64_t mhartid;
    __asm__ volatile ("csrr %0, mhartid" : "=r" (mhartid));
    volatile char *output = (char *) (HART_CONSOLE_BASE + mhartid);
    *output = character;
}
