﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

// page size is 32bytes
#define PAGE_SIZE (1 << 5)
// 16 KB in page table
#define INVERT_PAGE_TABLE_SIZE (1 << 14)
// 32 KB in shared memory
#define PHYSICAL_MEM_SIZE (1 << 15)
// 128 KB in global memory
#define STORAGE_SIZE (1 << 17)

#define NUM_THREADS (1<<2)
// newly created 4 threads for parallel execution of the program

/*managed: memory space specifier*/

//// count the pagefault times
__device__ __managed__ int pagefault_num = 0;

// data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

// memory allocation for virtual_memory
// secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];
// page table
extern __shared__ u32 pt[];

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size);

/*the use of global: to tell the compiler that the following function will run on GPU
 whereas in cuda, functions are called as kernels*/
__global__ void mykernel(int input_size) {

	// memory allocation for virtual_memory
	// take shared memory as physical memory
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	VirtualMemory vm;
	vm_init(&vm, data, storage, pt, &pagefault_num, PAGE_SIZE,
			INVERT_PAGE_TABLE_SIZE, PHYSICAL_MEM_SIZE, STORAGE_SIZE,
			PHYSICAL_MEM_SIZE / PAGE_SIZE);
	// PHYSICAL_MEM_SIZE/PAGE_SIZE = entrie numbers

	// user program the access pattern for testing paging
	user_program(&vm, input, results, input_size);
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize) {
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize) {
	FILE *fp;

	fp = fopen(fileName, "rb");
	if (!fp) {
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	// Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize) {
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	// Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);

	return fileLen;
}

int main() {
	cudaError_t cudaStatus;
	cudaSetDevice(0);
	// input file's memory size
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	float elapsed_time;
	/* Launch kernel function in GPU, with single thread
	and dynamically allocate INVERT_PAGE_TABLE_SIZE bytes of share memory,
	which is used for variables declared as "extern __shared__" */
	mykernel<<<1, NUM_THREADS, INVERT_PAGE_TABLE_SIZE>>>(input_size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// Parallel execution in 4 threads
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n",
			cudaGetErrorString(cudaStatus));
		return 0;
	}

	printf("input size: %d\n", input_size);  // input_size = 131,072 in test case

	cudaDeviceSynchronize();
	cudaDeviceReset();
	
	write_binaryFile(OUTFILE, results, input_size);
	printf("GPU elapse time is: %f seconds\n", elapsed_time/1000.0);
	printf("pagefault number is %d\n", pagefault_num);

	return 0;
}
