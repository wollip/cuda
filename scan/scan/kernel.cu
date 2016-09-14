
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <cstdlib>

#define ARRAY_SIZE 1000

__global__ void scan(float* d_in, float* d_out){
	__shared__ float s[ARRAY_SIZE];
	int index = threadIdx.x;
	
	s[index] = d_in[index];
	__syncthreads();

	float local = 0;
	for(int add = 1; add < ARRAY_SIZE; add <<= 1){
		local = s[index];
		__syncthreads();

		if (add + index < ARRAY_SIZE)
			s[index + add] += local;	
		__syncthreads();
	}
	d_out[index] = s[index];
}



int main(void){

	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];

	for (int i = 0; i < ARRAY_SIZE; i++){
		h_in[i] = float( rand() % 10 + 1 );
		//printf("%lf\n", h_in[i]);
	}
	
	const size_t BYTE_SIZE = ARRAY_SIZE *  sizeof(float);

	float* d_in;
	float* d_out;

	cudaMalloc((void**)&d_in, BYTE_SIZE);
	cudaMalloc((void**)&d_out, BYTE_SIZE);


	cudaMemcpy(d_in, h_in, BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	scan << <1, ARRAY_SIZE >> >(d_in, d_out);

	cudaMemcpy(h_out, d_out, BYTE_SIZE, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < ARRAY_SIZE; i++){
		printf("%d, %lf: %lf\n", i, h_in[i], h_out[i]);
	}
	
	cudaFree(d_in);
	cudaFree(d_out);
	
	system("pause");
	return 0;
}