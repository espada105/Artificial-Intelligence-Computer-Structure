#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>  // C에서 true/false 사용을 위한 헤더

// void init 함수
void init(int *a, int N) {
    int i;
    for (i = 0; i < N; ++i) {  // ++1을 ++i로 수정
        a[i] = i;
    }
}

// global 함수
__global__ void doubleElements(int *a, int N) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;  // thread 인덱스를 계산하는 공식
    if (i < N) {
        a[i] *= 2;  
    }
}

// bool 2를 곱한 결과인지를 확인하는 check 함수
bool checkElementsAreDoubled(int *a, int N) {
    int i;
    for (i = 0; i < N; i++) {
        if (a[i] != i * 2) return false;
    }
    return true;  
}

// int main() 함수
int main() {
    int N = 1000;
    int *a;

    size_t size = N * sizeof(int);  // 1000개의 정수를 저장할 메모리 크기 계산 (4000 바이트)

    cudaMallocManaged(&a, size);  // 통합 메모리 할당

    init(a, N);  // CPU에서 배열 초기화

    size_t threads_per_block = 256;  // 블록당 스레드 수
    size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;  // 필요한 블록 수 계산

    // GPU에서 커널 실행
    doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
    cudaDeviceSynchronize();  // GPU 작업 완료 대기

    // 결과 확인
    bool areDoubled = checkElementsAreDoubled(a, N);  

    // 결과 출력
    printf("모든 배열 원소에 2를 곱한 결과가 들어갔는가? %s\n", areDoubled ? "Y" : "N");

    cudaFree(a);  // 메모리 해제

    return 0;
}
