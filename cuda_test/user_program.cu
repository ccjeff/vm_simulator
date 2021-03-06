﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
	printf("break1\n");
	//printf("the current executing thread is: %d");
	for (int i = 0; i < input_size; i+=4) {
		int index = i + threadIdx.x;
		//printf("the index is: %d\n",index);
		vm_write(vm, index, input[index]);
		__syncthreads();
	}
	
	/*
	printf("break 1 finish, trying first element retrieval");
	for (int i = 0; i < vm->PHYSICAL_MEM_SIZE; i++) {
		printf("the retrieved element is : %c\n", vm->buffer[i]);
	}
	*/
	printf("break2\n");
	/*
	for (int i = 0; i < input_size; i+=4) {
		int thread_num = threadIdx.x;
		uchar a = vm_read(vm, i + threadIdx.x);
		//printf("the threadnum is: %d", threadIdx.x);
		//printf("the threadnum is: %d", threadIdx.x);
		//printf("the value from vm_read is: %c \n", a);
	}
	*/

	
	for (int i = input_size - 1; i >= input_size - 32769; i-=4) {
		//printf("vm read addr is: %d",i);
		int value = vm_read(vm, i);
		//printf("value is : %d",value);
		__syncthreads();
	}

	
	printf("break3\n");

	vm_snapshot(vm, results, 0, input_size);
	__syncthreads();
}
