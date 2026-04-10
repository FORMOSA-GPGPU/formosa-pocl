/* Formosa-specific device-side printf allocator.
 *
 * This mirrors lib/kernel/printf.c, but uses an atomic fetch-add so multiple
 * work-items can reserve disjoint buffer ranges concurrently.
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
