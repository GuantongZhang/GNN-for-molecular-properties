from collections import defaultdict
import torch

dataset = torch.load("./data/train.pt")

# Count degrees across all molecules
degree_counts = defaultdict(int)

for graph in dataset:
    # Compute degrees for each graph
    degrees = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
    for deg in degrees:
        degree_counts[int(deg)] += 1

print("Degree distribution:", dict(degree_counts))