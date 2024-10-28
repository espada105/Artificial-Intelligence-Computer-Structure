#incldue <stdio.h>

int main(){
    int deviceId; 
    cudaGetDevice(&deviceId); //현재 GPU 장치의 ID를 가져옴  

    cudaDeviceProp props; //cudaDeviceProp 구조체를 선언하여, GPU의 속성을 저장할 공간을 만듦
    cudaGetDeviceProperties(&props, deviceId); // 해당 디바이스 ID를 가진 GPU의 속성 정보를 props에 저장

    int computeCapabilityMajor = props.major; //현재 GPU의 "컴퓨팅 기능"을 나타내는 주 버전(major)을 가져옴
    int computeCapabilityMajor = props.minor; //현재 GPU의 "컴퓨팅 기능"을 나타내는 부 버전(minor)을 가져옴
    int multiprocessorCount = props.multiprocessorCount; //  GPU에 있는 스트리밍 멀티프로세서(SM)의 개수를 가져옴
    int warpSize = props.warpSize; // 워프 크기(즉, 한 번에 몇 개의 스레드가 그룹으로 처리되는지)를 가져옵니다. 일반적으로 이 값은 32

    printf("Device ID: %d\n SM 개수: %d\n Compute CapabilityMajor: %d\n Compute CapabilityMinor:%d\n워프크기: %d\n",deviceId,multiprocessorCount,computeCapabilityMajor,cudaDevAttrComputeCapabilityMinor,warpSize);
}