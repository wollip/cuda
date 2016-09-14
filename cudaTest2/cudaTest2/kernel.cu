
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <stdio.h>

#define NUM_BLOCKS 10
#define BLOCK_SIZE 1

__global__ void hello(){
	for (int i = 0; i < 10; i++){
		printf("this is block: %d, thread %d\n",i,  blockIdx.x);
		__syncthreads();
		printf("hi: %d, %d\n", i, blockIdx.x);
		__syncthreads();
	}

}

int main(void){
	hello<<<NUM_BLOCKS, BLOCK_SIZE>>>();
	
	printf("hello\n");
	cudaDeviceSynchronize();

	printf("done\n");
	system("pause");
	return 0;
}