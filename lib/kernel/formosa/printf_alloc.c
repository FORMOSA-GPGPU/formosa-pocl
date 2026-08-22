/* Formosa-specific device-side printf allocator.
 *
 * Derived from PoCL's lib/kernel/printf.c.
 *
 * Formosa modification:
 * - Use a RISC-V atomic fetch-add so multiple work-items can reserve
 *   disjoint buffer ranges concurrently.
 *
 * Copyright (c) 2024 Michal Babej / Intel Finland Oy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <stdint.h>

#ifdef PRINTF_BUFFER_AS_ID
#define PRINTF_BUFFER_AS __attribute__((address_space(PRINTF_BUFFER_AS_ID)))
#else
#define PRINTF_BUFFER_AS
#endif

#define ATTRS __attribute__((noinline)) __attribute__((optnone))

extern PRINTF_BUFFER_AS char *_printf_buffer;
extern PRINTF_BUFFER_AS uint32_t *_printf_buffer_position;
extern uint32_t _printf_buffer_capacity;

#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
extern void pocl_flush_printf_buffer(PRINTF_BUFFER_AS void *buffer,
                                     uint32_t bytes);
#endif

static inline uint32_t
formosa_atomic_fetch_add_u32(PRINTF_BUFFER_AS uint32_t *ptr, uint32_t value) {
  uint32_t old;
  __asm__ volatile("amoadd.w.aqrl %0, %2, (%1)"
                   : "=r"(old)
                   : "r"(ptr), "r"(value)
                   : "memory");
  return old;
}

PRINTF_BUFFER_AS void *ATTRS
pocl_printf_alloc(PRINTF_BUFFER_AS char *__buffer,
                  PRINTF_BUFFER_AS uint32_t *__buffer_position,
                  uint32_t __buffer_capacity, uint32_t bytes) {
  uint32_t old = formosa_atomic_fetch_add_u32(__buffer_position, bytes);
  uint64_t end = (uint64_t)old + (uint64_t)bytes;
  if (end > __buffer_capacity)
    return (void *)0;

  return __buffer + old;
}

PRINTF_BUFFER_AS void *ATTRS pocl_printf_alloc_stub(uint32_t bytes) {
  PRINTF_BUFFER_AS void *retval = pocl_printf_alloc(
      _printf_buffer, _printf_buffer_position, _printf_buffer_capacity, bytes);

#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
  pocl_flush_printf_buffer(_printf_buffer, bytes);
#endif

  return retval;
}
