#include <stdio.h>
void CPUFunction(){
    printf("this function is defined to run on the CPU.\n");
}

__global__ void GPUFunction(){
    printf("This function is defined to run on the GPU \n");
}

int main(){
    CPUFunction();

    GPUFunction()<<<1, 1>>>();
    cudaDeviceSynchronize();
}