
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <cstdlib>
#include <math.h>

#define ARRAY_SIZE 1001

__device__ void swap(float* a, int index1, int index2){
	float temp = a[index1];
	a[index1] = a[index2];
	a[index2] = temp;
}

__global__ void oddEvenSort(float* d_in, float* d_out){
	__shared__ float s[ARRAY_SIZE];
	
	int index = 2*threadIdx.x;
	int swapIndex = index + 1;

	s[index] = d_in[index];
	s[swapIndex] = d_in[swapIndex];
	__syncthreads();
	
	bool even = true;
	for (int i = 0; i < ARRAY_SIZE; i++){
		if (swapIndex < ARRAY_SIZE && index >= 0){
			if (s[index] > s[swapIndex]){
				swap(s, index, swapIndex);
			}			
		}
		if (even){
			index++;
			swapIndex++;
			even = false;
		}
		else{
			index--;
			swapIndex--;
			even = true;
		}
		__syncthreads();
	}
	

	d_out[index] = s[2*threadIdx.x];
	d_out[swapIndex] = s[2*threadIdx.x + 1];
}

int main(void){
	float h_in[ARRAY_SIZE] , h_out[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++){
		h_in[i] = rand() % 10000;
	}

	const size_t BYTE_SIZE = ARRAY_SIZE*sizeof(float);

	float* d_in, *d_out;
	cudaMalloc((void**)&d_in, BYTE_SIZE);
	cudaMalloc((void**)&d_out, BYTE_SIZE);
	cudaMemcpy(d_in, h_in, BYTE_SIZE, cudaMemcpyHostToDevice);

	oddEvenSort << <1, ceil(ARRAY_SIZE/2) >> >(d_in, d_out);

	cudaMemcpy(h_out, d_out, BYTE_SIZE, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE; i++){
		printf("%d:%lf\n", i, h_out[i]);
	}
	
	cudaFree(d_in);
	cudaFree(d_out);

	system("pause");
	return 0;
}