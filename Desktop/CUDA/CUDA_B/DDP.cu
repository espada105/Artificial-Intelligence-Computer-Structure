if __name__ == "__main__":
    args = prepare()  // Argument 파싱
    time_start = time.time()
    
    //DP로 바꾸려면 지우기
    mp.spawn(main, args=(args,), nprocs=torch.cuda.device_count())  // Multi-processing 시작
    
    time_elapsed = time.time() - time_start
    print(f"\nTime elapsed: {time_elapsed:.2f} seconds")

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3", help="Comma-separated list of GPU ids")
    parser.add_argument("-e", "--epochs", default=3, type=int, metavar="N", help="Number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=32, type=int, metavar="N", help="Number of batch size")
    args = parser.parse_args()

    
    // 환경 변수 설정
    os.environ["MASTER_ADDR"] = "localhost"  // 마스터 노드의 IP 주소        //DP로 바꾸려면 지우기
    os.environ["MASTER_PORT"] = "19198"     // 마스터 노드의 포트 번호       //DP로 바꾸려면 지우기
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  // 사용하려는 GPU 설정
    world_size = torch.cuda.device_count()  // 사용 가능한 GPU 개수         //DP로 바꾸려면 지우기
    os.environ["WORLD_SIZE"] = str(world_size)                           //DP로 바꾸려면 지우기
    
    return args


//DP로 바꾸려면 지우기
def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)  // 현재 프로세스에 맞는 GPU 선택
    os.environ["RANK"] = str(local_rank)  // 현재 프로세스의 순위
    dist.init_process_group(backend="nccl", init_method="env://")  // DDP 초기화
def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()  // 현재 프로세스의 rank
    g = torch.Generator()         // PyTorch 랜덤 시드 생성기
    g.manual_seed(seed + local_rank)
    return g


def test(model, test_dataloader):
    local_rank = dist.get_rank()  // 현재 프로세스의 rank    //DP로 바꾸려면 지우기
    model.eval()
    size = torch.tensor(0.0).cuda()                         //DP로 바꾸려면 지우기
    correct = torch.tensor(0.0).cuda()                      //DP로 바꾸려면 지우기

    for images, labels in test_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            size += images.size(0)
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()

    // 모든 GPU에서 결과를 병합
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)              //DP로 바꾸려면 지우기
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)           //DP로 바꾸려면 지우기

    if local_rank == 0:  // 마스터 프로세스에서 결과 출력
        acc = correct / size
        print(f"Accuracy is {acc:.2%}")

def main(local_rank, args):
    init_ddp(local_rank)  // DDP 초기화                      //DP로 바꾸려면 지우기
    model = ConvNet().cuda()  // 모델 초기화

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  // BatchNorm 동기화  //DP로 바꾸려면 지우기
    // DDP 래핑, DP로 바꾸려면 지우기        
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)//DDP 전용 샘플러
g = get_ddp_generator()  // 랜덤 시드 생성기
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,  // DDP 샘플러와 shuffle 동시 사용 불가
    num_workers=4,
    pin_memory=True,
    sampler=train_sampler,
    generator=g
)
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
for epoch in range(args.epochs):
    if local_rank == 0:  // 출력은 마스터 프로세스에서만 수행
        print(f"Begin training of epoch {epoch + 1}/{args.epochs}")

    train_dloader.sampler.set_epoch(epoch)  // DDP 샘플러에서 에포크를 설정
    train(model, train_dloader, criterion, optimizer, scaler)  // 학습 함수 호출
if local_rank == 0:
    print("Begin testing")

test(model, test_dloader)  // 테스트 함수 호출

if local_rank == 0:  // 모델 및 스케일러 저장은 마스터 프로세스에서만 수행
    torch.save(
        {
            "model": model.state_dict(),
            "scaler": scaler.state_dict(),
        },
        "ddp_checkpoint.pt",
    )

dist.destroy_process_group()  // DDP 프로세스 그룹 종료
