#include "fsa_mem_allocator.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct Block {
  /* Size of the block */
  size_t size;
  struct Block* next;
  /* Start address of the data */
  uintptr_t dptr;
};

static struct Block free_list = {.size = -1, .next = NULL, .dptr = 0};
static struct Block allocated_list = {.size = -1, .next = NULL, .dptr = 0};
static FILE* logf = NULL;

void log_list(struct Block* list_head) {
  struct Block* curr = list_head->next;
  fprintf(logf, "[");
  while (curr) {
    if (curr != list_head->next) {
      fprintf(logf, ",");
    }
    fprintf(logf, "{\"start_addr\": %ld, \"size\": %ld}\n", curr->dptr,
            curr->size);
    curr = curr->next;
  }
  fprintf(logf, "]");
}

void fsaMemLog(int op, uintptr_t value) {
  fprintf(logf, ",{");
  fprintf(logf, "\"operation\": \"%s\",", op ? "malloc" : "free");
  fprintf(logf, "\"value\": %ld,", value);
  fprintf(logf, "\"free_list\":");
  log_list(&free_list);
  fprintf(logf, ",\"allocated_list\":");
  log_list(&allocated_list);
  fprintf(logf, "}");
}

/* Align number of ssize_t, 64-bit in our case */
static inline uintptr_t align(size_t n) {
  return (n + sizeof(size_t) - 1) & ~(sizeof(size_t) - 1);
}

void fsaMemAllocInit(uintptr_t start, size_t size, int log) {
  /* Init full size free block */
  struct Block* temp = malloc(sizeof(*temp));
  temp->size = size;
  temp->next = NULL;
  temp->dptr = start;
  free_list.dptr = start;
  free_list.next = temp;
  /* Init log file */
  if (log) {
    logf = fopen("log.json", "w");
    fprintf(logf, "[");
    fprintf(logf, "{");
    fprintf(logf, "\"start\": %ld,", start);
    fprintf(logf, "\"size\": %ld", size);
    fprintf(logf, "}");
  }
}

int fsaMalloc(void** devPtr, size_t size) {
  /* Align to word size */
  size_t org = size;
  size = align(size);
  struct Block *free = free_list.next, *prev = &free_list;
  while (free) {
    if (free->size >= size) {
      break;
    }
    prev = free;
    free = free->next;
  }
  if (!free) {
    /* OOM error */
    return 1;
  }
  /* Split if possible */
  if (free->size > size) {
    struct Block* temp = malloc(sizeof(*temp));
    temp->size = free->size - size;
    temp->next = free->next;
    temp->dptr = free->dptr + size;
    free->next = temp;
    free->size = size;
  }
  /* Remove from free list and insert to allocated list head */
  prev->next = free->next;
  free->next = allocated_list.next;
  allocated_list.next = free;
  *devPtr = (void*)free->dptr;
  if (logf) {
    fsaMemLog(1, org);
  }
  return 0;
}

int fsaAddrMalloc(uintptr_t addr, size_t size) {
  /* Align to word size */
  size_t org = size;
  size = align(size);
  struct Block *free = free_list.next, *prev = &free_list;
  while (free) {
    if (free->dptr <= addr && addr < free->dptr + free->size &&
        addr + size <= free->dptr + free->size) {
      break;
    }
    prev = free;
    free = free->next;
  }
  if (!free) {
    /* OOM error */
    return 1;
  }
  /* Split if possible (right) */
  if (addr + size < free->dptr + free->size) {
    struct Block* temp = malloc(sizeof(*temp));
    temp->size = free->dptr + free->size - (addr + size);
    temp->next = free->next;
    temp->dptr = addr + size;
    free->next = temp;
    free->size = size;
  }
  /* Split if possible (left) */
  if (free->dptr < addr) {
    struct Block* temp = malloc(sizeof(*temp));
    temp->size = addr - free->dptr;
    prev->next = temp;
    temp->next = free;
    temp->dptr = free->dptr;
    free->dptr = addr;
    free->size = size;
    prev = temp;
  }
  /* Remove from free list and insert to allocated list head */
  prev->next = free->next;
  free->next = allocated_list.next;
  allocated_list.next = free;
  if (logf) {
    fsaMemLog(1, org);
  }

  return 0;
}

int fsaFree(void* devPtr) {
  uintptr_t ptr = (uintptr_t)devPtr;
  /* Search for allocated block */
  struct Block *alloc = allocated_list.next, *alloc_prev = &allocated_list;
  while (alloc) {
    if (ptr == alloc->dptr) {
      break;
    }
    alloc_prev = alloc;
    alloc = alloc->next;
  }
  if (!alloc) {
    /* Invalid pointer */
    return 1;
  }
  alloc_prev->next = alloc->next;
  /* Search insertion location in free list */
  struct Block *curr = free_list.next, *insert = &free_list;
  while (curr) {
    if (ptr < curr->dptr) {
      break;
    }
    insert = curr;
    curr = curr->next;
  }
  /* Merge if possible */
  if (insert->dptr + insert->size == alloc->dptr) {
    insert->size += alloc->size;
    free(alloc);
  } else {
    alloc->next = insert->next;
    insert->next = alloc;
  }
  if (logf) {
    fsaMemLog(0, ptr);
  }
  return 0;
}

void fsaMemAllocClean() {
  struct Block* curr = free_list.next;
  while (curr) {
    struct Block* temp = curr;
    curr = curr->next;
    free(temp);
  }
  curr = allocated_list.next;
  while (curr) {
    struct Block* temp = curr;
    curr = curr->next;
    free(temp);
  }
  if (logf) {
    fprintf(logf, "]");
    fclose(logf);
  }
}
