//스트림 생성 및 제거
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaStreamDestroy(stream);

//스트림을 사용한 커널 실행
someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>();