from dgl.data.utils import load_graphs
import os

model_dir = './model'

glist, label_dict = load_graphs(os.path.join(model_dir, 'dgl-citation-network-graph.bin'))
graph = glist[0]

print(graph)