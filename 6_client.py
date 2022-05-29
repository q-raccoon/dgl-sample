from dgl.data import CoraFullDataset
dataset = CoraFullDataset()
graph = dataset[0]

print(graph)

labels = graph.ndata['label']
features = graph.ndata['feat']

first = features[0].tolist()

print(first)