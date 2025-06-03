import torch
import torch.nn as nn
# from attn_class import SelfAttentionV1, sa_V1

class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values= self.W_value(x)

        attn_scores = queries @  keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)

        context_vec  = attn_weights @ values
        return context_vec
    
torch.manual_seed(789)
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

selfattn_v2 = SelfAttentionV2(d_in, d_out)
# print(selfattn_v2(inputs))

queries = selfattn_v2.W_query(inputs)
keys = selfattn_v2.W_key(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

print(attn_weights)

context_length = attn_weights.shape[0]
mask = torch.tril(torch.ones(context_length, context_length))
print(mask)
masked_attn_weights = attn_weights*mask
print(masked_attn_weights)

# Using simple normalisation over causally masked attention weight matrix
row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
print(row_sums)

renorm_attn_weights = masked_attn_weights/row_sums
print(renorm_attn_weights)

# row_sum_renorm = renorm_attn_weights.sum(dim=-1)
# print(row_sum_renorm)
