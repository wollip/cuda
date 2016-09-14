
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>


__global__ void matrixMultiply(cudaPitchedPtr matrix1, cudaExtent extent){
	//printf("matrixMultiply is called from: %d, %d", threadIdx.x, threadIdx.y);

	char* devPtr = (char*)matrix1.ptr;
	size_t pitch = matrix1.pitch;
	size_t slicePitch = pitch*extent.height;

	int x = threadIdx.x;
	int y = threadIdx.y;
	int z = threadIdx.z;

	char* slice = devPtr + z * slicePitch;
	float* row = (float*)(slice + y * pitch);
	printf("%d,%d,%d : %f\n", x, y, z, row[x]);

}

int main(void){
	static const size_t ROWNUM = 10;
	static const size_t COLNUM = 5;
	static const size_t Z = 2;

	float* h_data = new float[ROWNUM*COLNUM*Z];
	for (int i = 0; i < ROWNUM*COLNUM*Z; i++){
		h_data[i] = (float)i;
	}

	cudaPitchedPtr h_dataPtr = make_cudaPitchedPtr(h_data, ROWNUM*sizeof(float), ROWNUM, COLNUM);

	cudaPitchedPtr d_matrix1Ptr;
	cudaExtent extent = make_cudaExtent(ROWNUM*sizeof(float), COLNUM, Z);

	cudaMalloc3D(&d_matrix1Ptr, extent);

	printf("%d\n", d_matrix1Ptr.pitch);

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = h_dataPtr;
	params.dstPtr = d_matrix1Ptr;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;

	cudaMemcpy3D(&params);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));


	dim3 dimen = dim3(ROWNUM, COLNUM, Z);
	matrixMultiply << <1, dimen >> >(d_matrix1Ptr, extent);
	
	delete[] h_data;

	cudaFree(d_matrix1Ptr.ptr);
	system("pause");
	return 0;
}