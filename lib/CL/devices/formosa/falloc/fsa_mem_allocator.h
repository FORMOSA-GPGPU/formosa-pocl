#pragma once

#include <stdint.h>
#include <stdlib.h>

void fsaMemAllocInit(uintptr_t start, size_t size, int log);
int fsaMalloc(void **devPtr, size_t size);
int fsaFree(void *devPtr);
void fsaMemAllocClean();