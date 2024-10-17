// !nvcc -arch = sm_70 -o GPU, CPU processing ./ GPU, CPU processing.cu -run

#include <stdio.h>

__global__ void GPUFunction(){
    printf("gpu function");
}

void CPUFunction(){
    printf("CPUFunction")
}

int main(){
    CPUFunction();
    GPUFunction<<<1, 1>>>(); //gpu,cpu,gpu 로 나오게 하려면 각 gpu함수뒤에 cudaDeviceSynchronize를 쓰면된다
    cudaDeviceSynchronize(); 
}