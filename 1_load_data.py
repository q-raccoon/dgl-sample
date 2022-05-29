import torch.nn.functional as F
import pandas as pd

import torch
import dgl

import matplotlib.pyplot as plt
import networkx as nx

nodes_data = pd.read_csv('./data/nodes.csv')
edges_data = pd.read_csv('./data/edges.csv')

src = edges_data['Src'].to_numpy()
dst = edges_data['Dst'].to_numpy()

g = dgl.graph((src, dst))

age = torch.tensor(nodes_data['Age'].to_numpy()).float() / 100
# print(age[0: 10])   # 0~9번째까지 slice
# print(age[[0, 10]]) # 0, 10 번째 값
g.ndata['age'] = age

club = nodes_data['Club'].to_list()
club = torch.tensor([c == 'Officer' for c in club]).long()
club_onehot = F.one_hot(club)

g.ndata.update({
  'club': club,
  'club_onehot': club_onehot
})
del g.ndata['age']

dege_weight = torch.tensor(edges_data['Weight'].to_numpy())
g.edata['weight'] = dege_weight

print(g)

g.srcdata['src_h'] = torch.randn(34,4)
g.dstdata['dst_h'] = torch.randn(34,7)

print(g)
nx_g = g.to_networkx().to_undirected()
pos = nx.kamada_kawai_layout(nx_g)

nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])

plt.show()

