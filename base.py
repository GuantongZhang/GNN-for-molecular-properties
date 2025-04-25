import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set

class QM_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(root=".")
        self.data = torch.load(path)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

class Net(torch.nn.Module):
    def __init__(self, num_features=11, dim=64):
        super().__init__()
        self.lin0 = Linear(num_features, dim)
        nn = Sequential(Linear(4, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean') # replace with your own convolutional layers here
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = Linear(2 * dim, dim)
        self.lin2 = Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        for _ in range(3):
            out = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        out = self.set2set(out, data.batch)
        return self.lin2(F.relu(self.lin1(out))).view(-1)

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.l1_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            total_loss += F.l1_loss(model(data), data.y).item()
    return total_loss / len(loader)

# Data setup
train_data = QM_Dataset("./data/train.pt")
train_data, validate_data = torch.utils.data.random_split(train_data, [19000, 1000])
test_data = QM_Dataset("./data/test.pt")

train_loader = DataLoader(train_data, batch_size=128)
validate_loader = DataLoader(validate_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=8)

# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

# Training loop
for epoch in range(1, 101):
    loss = train(train_loader)
    val_mae = eval(validate_loader)
    print(f"Epoch {epoch}: Loss: {loss:.4f}, Validation MAE: {val_mae:.4f}")
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"base_model/model_epoch_{epoch}.pt")