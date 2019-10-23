#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 

__device__ int vm_swap(int phyPage, int virtPage, VirtualMemory* vm) {
	(*vm->pagefault_num_ptr)++;
	for (int i = 0; i < vm->PAGESIZE; i++) {
		uchar tempBuffer;
		tempBuffer = vm->storage[virtPage*vm->PAGESIZE + i];
		vm->buffer[phyPage*vm->PAGESIZE + i] = tempBuffer;
	}
	return 0;
}

__device__ u32 leastUsed(VirtualMemory* vm) {
	int tempSmallest = vm->invert_page_table[vm->PAGE_ENTRIES];
	int tempAddr = 0;
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i + vm->PAGE_ENTRIES] < tempSmallest) {
			tempSmallest = vm->invert_page_table[i + vm->PAGE_ENTRIES];
			tempAddr = i;
		}
	}
	
	//vm->invert_page_table[tempAddr + vm->PAGE_ENTRIES] += 1024;
	//printf("least1 should be: %d\n", vm->invert_page_table[1024]);
	vm->invert_page_table[tempAddr + vm->PAGE_ENTRIES] = tempSmallest + vm->PAGE_ENTRIES;
	//printf("least used index is: %d\n", tempAddr);
	return (u32) tempAddr;
}


__device__ int getPhyAddr(int addr, VirtualMemory* vm) {
	int physicalPage = 0;
	int virtualPage = addr / vm->PAGESIZE;
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if ((vm->invert_page_table[i]) == (virtualPage)) { // todo: 这个地方应该是addr/page? 1024个page entry怎么存？这样判断virtual addr 肯定不对应
			physicalPage = (u32)i;
			return physicalPage;
		}
	}
	// if not in physical memory, then must be in swap space
	return 0xFFFFFFFF;
}

__device__ void init_invert_page_table(VirtualMemory *vm) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		// 1000_0000_0000_0000_0000_0000_0000_0000
		// 0000_0000_0000_0000_0001_1111_1111_1111 
		/*only 13 bits are required for storing the 128kb virtual address*/ 
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
	// init variables
	vm->buffer = buffer;
	// buffer is the physical memory
	vm->storage = storage;
	// virtual memory
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;

	// init constants
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;

	// before first vm_write or vm_read
	init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
	/* Complate vm_read function to read single element from data buffer */
	//todo 
	uchar toReturn;
	int pageIdx = addr / vm->PAGESIZE;  // virtualPage index
	int offsetIdx = addr % vm->PAGESIZE;

	int memAddr = getPhyAddr(addr, vm);
	if (memAddr == 0xFFFFFFFF) {
		// do swap
		u32 leastIdx = leastUsed(vm);
		vm_swap(leastIdx,pageIdx,vm);
		memAddr = leastIdx;
	}
	printf("inside read, char get is %c\n", vm->buffer[memAddr*vm->PAGESIZE + offsetIdx]);
	return vm->buffer[memAddr*vm->PAGESIZE+offsetIdx];
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	/* Complete vm_write function to write value into data buffer */
	int pageIdx = addr/vm->PAGESIZE;
	int offsetIdx = addr%vm->PAGESIZE;
	int physicalPage = 0xFFFFFFFF;
	
	physicalPage = getPhyAddr(addr,vm);
	// unsuccessful get will return 0xFFFFFFFF, which means needs swapping (or needs to be initialized).
	printf("getPhyAddr success , addr is: %d \n",addr);

	// if the page has been found in the pagetable
	printf("the physicalPage is: %d \n",physicalPage);
	if (physicalPage != 0xFFFFFFFF) {
		printf("found\n");
		int phyAddr = physicalPage * vm->PAGESIZE + offsetIdx;
		vm->buffer[phyAddr] = value;
		vm->invert_page_table[physicalPage + vm->PAGE_ENTRIES]+=vm->PAGE_ENTRIES;
		printf("normal write done \n");
	}
	else {
		// do swap before write
		int leastIdx = leastUsed(vm);
		if (vm->invert_page_table[leastIdx] == 0x80000000) {
			vm->buffer[leastIdx] = value;
			printf("value1 is: %c\n",value);
			vm->invert_page_table[leastIdx] = addr;
		}
		else {
			printf("the least index for swapping is: %d\n",leastIdx);
			printf("swap\n");
			vm_swap(leastIdx, pageIdx, vm);
			vm->buffer[leastIdx] = value;
			printf("value2 is: %c\n", value);
			vm->invert_page_table[leastIdx] = addr;
		}
	}
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
	/* Complete snapshot function togther with vm_read to load elements from data
	* to result buffer */
	printf("inside snapshot");
	for (int i = 0; i < vm->STORAGE_SIZE; i++) {
		results[offset + i] = vm_read(vm, ((u32)i));
	}
}

