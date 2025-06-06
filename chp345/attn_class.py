import torch
import torch.nn as nn
from attn_class_v2 import SelfAttentionV2, selfattn_v2

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys =  x @ self.W_key
        values = x @ self.W_value

        attn_score = queries @ keys.T
        attn_weights = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec
    
torch.manual_seed(123)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2

sa_V1 = SelfAttentionV1(d_in, d_out)
# print(sa_V1(inputs))

# Exercise 3.1
sa_V1.W_key = nn.Parameter(selfattn_v2.W_key.weight.T)
sa_V1.W_value = nn.Parameter(selfattn_v2.W_value.weight.T)
sa_V1.W_query = nn.Parameter(selfattn_v2.W_query.weight.T)
print(sa_V1(inputs))