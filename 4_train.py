import os, json

import torch as th
import torch.nn.functional as F 

from dgl.data import CoraFullDataset
from dgl.data.utils import split_dataset, save_graphs, load_graphs

import torch.nn as nn
from dgl.nn import GraphConv

class GraphConvolutionalNetwork(nn.Module):
  def __init__(self, in_feat, h_feat, out_feat):
    super().__init__()

    self.gcl1 = GraphConv(in_feat, h_feat)
    self.relu = nn.ReLU()
    self.gcl2 = GraphConv(h_feat, out_feat)

  def forward(self, graph, feat_data):
    x = self.gcl1(graph, feat_data)
    x = self.relu(x)
    x = self.gcl2(graph, x)

    return x

def main():
    # Setup Variables
    config_dir = './config'
    model_dir = './model'

    with open(os.path.join(config_dir, 'hyperparameters.json'), 'r') as file:
      parameters_dict = json.load(file)

      learning_rate = float(parameters_dict['learning-rate'])
      epochs = int(parameters_dict['epochs'])

    # Getting dataset
    dataset = CoraFullDataset()
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']

    # Splitting dataset
    train_mask, val_mask = split_dataset(graph, [0.8, 0.2])

    # Creating Model
    model = GraphConvolutionalNetwork(features.shape[1], 16, dataset.num_classes)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
      pred = model(graph, features)
      loss = F.cross_entropy(pred[train_mask.indices], labels[train_mask.indices].to(th.long))

      train_acc = (labels[train_mask.indices] == pred[train_mask.indices].argmax(1)).float().mean()
      val_acc = (labels[val_mask.indices] == pred[val_mask.indices].argmax(1)).float().mean()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'Epoch {epoch}/{epochs} | Loss: {loss.item()}, train_accuracy: {train_acc}, val_accuracy: {val_acc}')

    
    # Saving Graph
    save_graphs(os.path.join(model_dir, 'dgl-citation-network-graph.bin'), graph)

    # Saving Model
    th.save(model, os.path.join(model_dir, 'dgl-citation-network-model.pt'))

if __name__ == '__main__':
    main()