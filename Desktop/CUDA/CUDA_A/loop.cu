#include <cuda_runtime.h>
#include <stdio.h>

__global__ void loop(){
    printf("loop count: %d \n",threadIdx.x+1);
}

int main(){
    loop<<<1,10>>>();
    cudaDeviceSynchronize();
}

