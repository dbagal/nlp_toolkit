from sequence_models import *
import torch

d,n = 2,64
dmodel, dq,dk,dv = 100, 50,50,50
heads=2
feedfwd = 200
dqk = 50
num_buckets = 32

transformer_encoder = TransformerEncoder(dmodel,dq,dk,dv,heads,feedfwd)
reformer_encoder = ReformerEncoder(dmodel,dqk,dv,heads,feedfwd,num_buckets)

# Input of shape (d,n,dmodel)
# d => Number of documents
# n => Sequence length
# dmodel => Number of features
x = torch.randn(d,n,dmodel) + positionEncoding(d,n,dmodel)
print(transformer_encoder(x).shape)
print(reformer_encoder(x).shape)