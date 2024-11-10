//통합 메모리(Unified Memory) 할당
float *a;
float *b;
float *c;

cudaMallocManaged(&a, size);
cudaMallocManaged(&b, size);
cudaMallocManaged(&c, size);

cudaMemPrefetchAsync(a, size, deviceId);
cudaMemPrefetchAsync(b, size, deviceId);
cudaMemPrefetchAsync(c, size, deviceId);

//커널을 이용한 초기화
threadsPerBlock = 256;
numberOfBlocks = 32 * numberOfSMs;
initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
//에러 확인, GPU 기다림, 실행여부 점검
addVectorsErr = cudaGetLastError();
if(addVectorsErr != cudaSuccess) printf("Error: %s\n",cudaGetErrorString(addVectorsErr));

asyncErr = cudaDeviceSynchronize();
if(asyncErr != cudaSuccess) printf("Error: %s\n",cudaGetErrorString(asyncErr));

checkElementsAre(7, c, N);
cudaFree(a);
cudaFree(b);
cudaFree(c);
