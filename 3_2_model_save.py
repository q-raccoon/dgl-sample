from dgl.data.utils import save_graphs
from dgl.data import CoraFullDataset
import os

dataset = CoraFullDataset()
graph = dataset[0]

model_dir = './model'

save_graphs(os.path.join(model_dir, 'dgl-citation-network-graph.bin'), graph)