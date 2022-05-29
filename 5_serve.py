import os
from dgl.convert import graph 
from dgl.data.utils import load_graphs
import torch as th

import torch.nn as nn
from dgl.nn import GraphConv

from fastapi import FastAPI
from brotli_asgi import BrotliMiddleware
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from pydantic import BaseModel
from typing import List

class Data(BaseModel) :
  data: List[int]

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


model_dir = './model'
graph_dir = './model'

glist, label_dict = load_graphs(os.path.join(model_dir, 'dgl-citation-network-graph.bin'))
graph = glist[0]
model = th.load(os.path.join(model_dir, 'dgl-citation-network-model.pt'))
features = graph.ndata['feat']
pred = model(graph, features)

app = FastAPI()

app.add_middleware(
  BrotliMiddleware,  
  quality=4,
  mode="generic",
  lgwin=22,
  lgblock=0,
  minimum_size=400,
  gzip_fallback=True
)
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/") 
def ping():
  return 'hello world'

@app.post("/gcn") 
def predict(data: Data):
  return pred[data.data].tolist()

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)