import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, global_mean_pool
from torch_geometric.nn.aggr import DegreeScalerAggregation

class QM_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(root=".")
        self.data = torch.load(path)
        self.graph_ids = [d.name for d in self.data]

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

class PNAModel(torch.nn.Module):
    def __init__(self, num_features=11, edge_dim=4, dim=128):
        super().__init__()
        
        # Calculate degree statistics (important for PNA)
        self.degrees = torch.tensor([1, 2, 3, 4])
            # Degree distribution: {4: 90613, 2: 35111, 3: 38518, 1: 195952}
            # See deg_distr.py
        
        # Input embedding
        self.atom_encoder = Linear(num_features, dim)
        self.edge_encoder = Linear(edge_dim, dim)
        
        # PNA Convolutions with edge features
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv1 = PNAConv(
            in_channels=dim,
            out_channels=dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=self.degrees,
            edge_dim=dim,
            towers=4
        )
        self.conv2 = PNAConv(
            in_channels=dim,
            out_channels=dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=self.degrees,
            edge_dim=dim,
            towers=4
        )
        self.conv3 = PNAConv(
            in_channels=dim,
            out_channels=dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=self.degrees,
            edge_dim=dim,
            towers=4
        )
        
        # Batch normalization
        self.bn1 = BatchNorm1d(dim)
        self.bn2 = BatchNorm1d(dim)
        self.bn3 = BatchNorm1d(dim)
        
        # Prediction head
        self.head = Sequential(
            Linear(dim, dim//2),
            ReLU(),
            BatchNorm1d(dim//2),
            Linear(dim//2, 1)
        )

    def forward(self, data):
        x = F.relu(self.atom_encoder(data.x))
        edge_attr = F.relu(self.edge_encoder(data.edge_attr.float()))
        
        x = F.relu(self.bn1(self.conv1(x, data.edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, data.edge_index, edge_attr)))
        x = F.relu(self.bn3(self.conv3(x, data.edge_index, edge_attr)))
        
        x = global_mean_pool(x, data.batch)
        return self.head(x).view(-1)

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y.float())  # Using MSE for smoother gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            total_loss += F.l1_loss(out, data.y.float()).item()  # Report MAE
    return total_loss / len(loader)

if __name__ == "__main__":

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = QM_Dataset("./data/train.pt")
    train_data, val_data = torch.utils.data.random_split(train_data, [19000, 1000])
    test_data = QM_Dataset("./data/test.pt")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=128)

    # Model and optimizer
    model = PNAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training loop
    best_val_mae = float('inf')
    epochs = list(range(1, 101))
    train_loss = []
    val_mae = []

    for epoch in epochs:
        train_loss = train(train_loader)
        val_mae = eval(val_loader)
        scheduler.step(val_mae)
        
        train_loss.append(train_loss)
        val_mae.append(val_mae)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_pna_model.pt")
            print(f"New best model saved (Val MAE: {val_mae:.4f})")

    # Plot
    import matplotlib.pyplot as plt

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_loss, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(train_loss)*1.1)

    # Create secondary axis for validation MAE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation MAE', color=color)
    ax2.plot(epochs, val_mae, color=color, label='Validation MAE', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(val_mae)*1.1)

    # Add title and grid
    plt.title('Training Loss and Validation MAE over Epochs')
    fig.tight_layout()
    plt.grid(True)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    #plt.show()
    plt.savefig("training_validation_plot.png", dpi=300)