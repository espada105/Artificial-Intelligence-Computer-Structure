//py
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type = int, default = 0)
    parser.add_argument("--local_world_size", type = int, default = 1)
    args = parser.parse_args()
    spmd_main(args.local_world_size, args.local_rank)

class ToyMpModel(nn.Module):
	def __init__(self, dev0, dev1): //두 개의 네트워크 정의 후에 각각 다른 장치에 배치
    	super(ToyMpModel, self).__init__()
        self.dev0 = dev0 //첫 번째 장치
        self.dev1 = dev1 //두 번째 장치
        self.net1 = torch.nn.Linear(10, 10).to(dev0) //첫 번째 장치에 할당
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1) //두 번째 장치에 할당
    
    def forward(self, x): //순전파 함수, 입력 데이터를 두 장치로 이동하며 처리
    	x = x.to(self.dev0) //데이터를 첫 번째 장치로 이동
        x = self.relu(self.net1(x)) //첫 번째 장치에서 net1을 사용해서 계산
        x = x.to(self.dev1) // 데이터를 두 번째 장치로 이동
        return self.net2(x) // 두 번째 장치에서 net2를 사용해서 계산값 반환

    def spmd_main(local_world_size, local_rank):
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }

        print(f"[os.getid()] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{(os.getpid()})] world_size = {dist.get_world_size()},"
            + f"rank = {dist.get_rank()}, backend = {dist.get_backend()}"
        )

        demo_basic(local_world_size, local_rank)

        dist.destroy_process_group()

    def demo_basic(local_world_size, local_rank):
        n = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * n, (local_rank + 1) * n))

        print(
            f"[{os.getid()}] rank = {dist.get_rank()},"
            + f"world_size = {dist.get_world_size()}, n = device_ids = {device_ids}"
        )

        model = ToyModel().cuda(device_ids[0])
        ddp_model = DDP(model, device_ids)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr = 0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_ids[0])
        loss_fn(outputs, labels).backward()
        optimizer.step()