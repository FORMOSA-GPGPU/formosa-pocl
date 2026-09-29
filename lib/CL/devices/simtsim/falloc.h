#ifndef FALLOC_H
#define FALLOC_H

#include <stdint.h>
#include <stdio.h>

void fsa_init(uintptr_t start, size_t size, FILE *log);
int fsa_malloc(void **devPtr, size_t size);
int fsa_addr_malloc(uintptr_t addr, size_t size);
int fsa_free(void *devPtr);
void fsa_clean(void);

#endif
