#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
	//printf("break1\n");
	//printf("the input size is: %d", input_size);
	for (int i = 0; i < input_size; i++) {
		vm_write(vm, i, input[i]);
	}
	/*
	printf("break 1 finish, trying first element retrieval");
	for (int i = 0; i < vm->PHYSICAL_MEM_SIZE; i++) {
		printf("the retrieved element is : %c\n", vm->buffer[i]);
	}
	*/
	printf("break2\n");
	
	/*
	for (int i = input_size - 1; i >= input_size - 32769; i--) {
		//printf("vm read addr is: %d",i);
		int value = vm_read(vm, i);
		printf("value is : %d",value);
	}
	*/
	printf("break3\n");

	vm_snapshot(vm, results, 0, input_size);
}
