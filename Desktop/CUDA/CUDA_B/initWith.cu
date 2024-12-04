__global__
void initWith(float num, float *a, int N){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride)
    {
        a[i] = num; 
    }
}