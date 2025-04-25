import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from improved import QM_Dataset, PNAModel



test_data = QM_Dataset("./data/test.pt")
test_loader = DataLoader(test_data, batch_size=8)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PNAModel().to(device)
model.load_state_dict(torch.load("./pna_model/1234.pt"))
model.eval()

y_pred = []
#graph_ids = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        y_pred.extend(out.cpu().numpy().flatten())
        
        batch_indices = data.batch.cpu().numpy()
        unique_batch_ids = torch.unique(data.batch).cpu().numpy()
        
        #for graph_num in unique_batch_ids:
        #    graph_ids.append(test_data.graph_ids[graph_num])

df = pd.DataFrame({"Idx": test_data.graph_ids, "labels": y_pred})
df.to_csv("./data/submission.csv", index=False)
print("Submission file created successfully.")