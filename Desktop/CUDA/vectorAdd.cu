#include <stdio.h>

/*__global__ gpu에서 먼저실행하려면 3줄추가하고 global로 변경한다*/ void initWith(float num, float *a, int N) {
    for (int i = 0; i < N; ++i) {  
        a[i] = num;
    }
}벡

__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;   
    int stride = blockDim.x * gridDim.x;   
    //전체 그리드의 스레드 수를 계산, 모든 스레드가 동시에 작업을 처리하는 것을 방지하기 위해 사용, 큰 벡터에서 작업할 때 하나의 스레드가 작업한 후에 stride만큼 이동해 다른 작업을 수행할 수 있게 함
    
    
    for (int i = index; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}


void checkElementsAre(float target, float *vector, int N){
    for(int i = 0; i < N; ++i){
        if(vector[i] != target){
            printf("Fail: vector[%d] - %0.0f 는 % 0.0f값과 다릅니다. \n",i, vector[i], target)
            exit(1);
        }
    }
    printf("Success: 모든 값이 올바르게 계산되었습니다.\n");
}

int main(){
    const int N = 2<<24; // N은 2^24 크기의 터를 의미(스칼라값 2^24)
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;
// cudaMallocManaged를 통해 호스트(CPU)와 디바이스(GPU) 모두 접근할 수 있는 관리 메모리(a, b, c)를 할당
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;
    printf("블록으로 설정된 총 개수: %d\n", numberOfBlocks);

// a,b,c 벡터 초기화
    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

//각각 블록당 스레드 수와 전체 블록 수를 설정
    size_t threadsPerBlock;
    size_t numberOfBlocks;

//초기화 작업을 CPU에서 먼저 하는 경우
    // threadsPerBlock = 1;
    // numberOfBlocks = 1;
    // printf("블록으로 설정된 총 개수: %d\n", numberOfBlocks);

//초기화 작업을 GPU에서 먼저 하는 경우
    // initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
    // initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
    // initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);


    // cudaDeviceSynchronize();

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c,a,b,N);

// 커널실행할때 에러 확인 
    addVectorsErr = cudaGetError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize(); //cudaDeviceSynchronize 써서 GPU 작업 끝날때까지 기다리기 -> 동기화에러 확인
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    checkElementsAre(7, c, N); //c 벡터의 각 요소가 7(3 + 4)이 되었는지 확인

//할당한 메모리를 해제
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
