__global__ void addVectors(float *result, float *a, float *b, int N){
    int index = threadIdx.x + blockIdx.x + *blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}