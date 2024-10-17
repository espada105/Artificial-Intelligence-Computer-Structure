#include <stdio.h>

__global__ void loop(){
    printf("parrel processing");
}

int main(){
    loop<<<5,5>>>();
    cudaDeviceSynchronize();
}