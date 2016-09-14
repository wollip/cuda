// made by jason

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include <stdio.h>

class GpuTimer{
public:
	GpuTimer();
	~GpuTimer();

	void startTimer();
	void stopTimer();
	void printTime();
private:
	void checkStatus(int i);
public:
	cudaEvent_t start, stop;
	bool bstart, bstop;
	cudaError_t cudaStatus;
};

void GpuTimer::checkStatus(int i){
	if(cudaStatus != cudaSuccess){
		printf("GpuTimer: cuda failed in %d\n", i);
		printf("%s\n", cudaGetErrorString(cudaStatus));
	}
}

GpuTimer::GpuTimer(){
	cudaStatus = cudaSetDevice(0);

	cudaEventCreate(&start);
	checkStatus(0);

	cudaEventCreate(&stop);
	checkStatus(1);

	bstart = false;
	bstop = false;
}

GpuTimer::~GpuTimer(){
	cudaEventDestroy(start);
	checkStatus(2);
	cudaEventDestroy(stop);
	checkStatus(3);
}

void GpuTimer::startTimer(void){
	bstart = true;
	cudaEventRecord(start, 0);
}

void GpuTimer::stopTimer(void){
	if(bstart){
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		bstart = false;
		bstop = true;
	}else{
		printf("GpuTimer: start has not been called\n");
	}
}


void GpuTimer::printTime(void){
	if(bstop){
		float time;
		cudaEventElapsedTime(&time, start, stop);
		printf("time = %f ms\n", time);
		bstop = false;
	}else{
		printf("GpuTImer: stop has not been called\n");
	}
}