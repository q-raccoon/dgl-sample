# https://datachef.co/blog/a-graph-convolution-network-in-sagemaker/

from dgl.data import CoraFullDataset
dataset = CoraFullDataset()
graph = dataset[0]

print(graph)

labels = graph.ndata['label']
features = graph.ndata['feat']

print(len(features))
print(len(features[0].tolist()))
print(len(labels))

print(dataset.num_classes)
print(features.shape, features.shape[0], features.shape[1])