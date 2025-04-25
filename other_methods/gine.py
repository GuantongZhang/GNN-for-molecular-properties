import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool

class QM_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(root=".")
        self.data = torch.load(path)
        self.graph_ids = [d.name for d in self.data]

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

class ImprovedNet(torch.nn.Module):
    def __init__(self, num_features=11, edge_dim=4, dim=128):
        super().__init__()
        self.atom_encoder = Linear(num_features, dim)
        
        # GINEConv with proper edge dimension handling
        nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINEConv(nn, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn, edge_dim=edge_dim)
        self.conv3 = GINEConv(nn, edge_dim=edge_dim)
        
        # Global pooling
        self.pool = global_add_pool
        
        # Prediction head
        self.head = Sequential(
            Linear(dim, dim//2),
            ReLU(),
            Linear(dim//2, 1)
        )

    def forward(self, data):
        x = F.relu(self.atom_encoder(data.x))
        edge_attr = data.edge_attr.float()  # Ensure edge features are float
        
        x = F.relu(self.conv1(x, data.edge_index, edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, edge_attr))
        x = F.relu(self.conv3(x, data.edge_index, edge_attr))
        x = self.pool(x, data.batch)
        return self.head(x).view(-1)

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.l1_loss(model(data), data.y.float())  # Ensure target is float
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
            total_loss += F.l1_loss(model(data), data.y.float()).item()
    return total_loss / len(loader)

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = QM_Dataset("./data/train.pt")
train_data, validate_data = torch.utils.data.random_split(train_data, [19000, 1000])

train_loader = DataLoader(train_data, batch_size=128)
validate_loader = DataLoader(validate_data, batch_size=128)

model = ImprovedNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training loop
for epoch in range(1, 101):
    loss = train(train_loader)
    val_mae = eval(validate_loader)
    print(f"Epoch {epoch}: Loss: {loss:.4f}, Val MAE: {val_mae:.4f}")
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")