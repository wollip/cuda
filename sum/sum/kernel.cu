
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>

#define ARRAY_SIZE 1000

__global__ void sum(float* d_in, float* d_sum){
	__shared__ float s[ARRAY_SIZE];
	int index = threadIdx.x;

	s[index] = d_in[index];
	__syncthreads();

	for (int i = 1; i < ARRAY_SIZE; i <<= 1){
		if (index%(i*2) == 0){
			s[index] += s[index + i];
		}
		__syncthreads();
	}
	if (index == 0){
		d_sum[0] = s[index];
	}
}


int main(void){
	const size_t BYTE_SIZE = ARRAY_SIZE *  sizeof(float);

	float h_in[ARRAY_SIZE];
	float h_sum[1];

	for (int i = 0; i < ARRAY_SIZE; i++){
		h_in[i] = float(rand() % 10 + 1);
		//printf("%lf\n", h_in[i]);
	}

	float* d_in;
	float* d_sum;

	cudaMalloc((void**)&d_in, BYTE_SIZE);
	cudaMalloc((void**)&d_sum, sizeof(float));

	cudaMemcpy(d_in, h_in, BYTE_SIZE, cudaMemcpyHostToDevice);

	if(cudaDeviceSynchronize() == cudaSuccess)
		printf("memory copy is sucessful\n");

	sum << <1, ARRAY_SIZE >> >(d_in, d_sum);

	if (cudaDeviceSynchronize() == cudaSuccess)
		printf("sums is sucessful\n");

	cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

	printf("sum: %lf\n", h_sum[0]);
	
	cudaFree(d_in);
	cudaFree(d_sum);

	system("pause");
	return 0;
}