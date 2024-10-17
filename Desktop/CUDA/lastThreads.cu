#include <cuda_runtime.h>
#include <stdio.h>

__global__ void last(){
    if(threadIdx.x == 1023, blockIdx.x ==255){
        printf("Last thread completed");
    }
}

int main(){
    last <<<256, 1024>>>(); //block과 thread 위치를 잘 확인해야됨 
    cudaDeviceSynchronize(); //커널 실행이 비동기 이므로 반드시 완료후에 싱크를 해야된다. cDS
}