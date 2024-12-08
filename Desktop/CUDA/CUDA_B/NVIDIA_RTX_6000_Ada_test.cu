import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os

if torch.cuda.device_count() > 1:
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    print("Multiple GPUs not available")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def generate_data(samples = 1600000, features = 1024):
        x = torch.randn(samples, features)
        y = torch.randint(0, 10, (samples,))
        return TensorDataset(x, y)
    
    def train_model(batch_size = 160000, epochs = 5):
    dataset = generate_data()
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    model = SimpleModel()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = torch.device("cudda" if torch.cuda.is_available() else "cpu")

    print(f"사용하는 GPU 목록: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model()


