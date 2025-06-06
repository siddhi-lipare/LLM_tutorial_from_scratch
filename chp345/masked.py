from attn_class_v2 import selfattn_v2, SelfAttentionV2
import torch

torch.manual_seed(789)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

queries = selfattn_v2.W_query(inputs)
keys = selfattn_v2.W_key(inputs)
values = selfattn_v2.W_value(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

# print(attn_weights)

context_length = attn_weights.shape[0]

# mask = torch.tril(torch.ones(context_length, context_length))
# print(mask)
# masked_attn_weights = attn_weights*mask
# print(masked_attn_weights)

# # Using simple normalisation over causally masked attention weight matrix
# row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
# print(row_sums)

# renorm_attn_weights = masked_attn_weights/row_sums
# print(renorm_attn_weights)

# context_vec = renorm_attn_weights @ values
# print(context_vec)


mask2 = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask2.bool(), -torch.inf)
print(masked)
attn_weights_2 = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights_2) # Now sum is already 1

context_vec = attn_weights_2 @ values
print(context_vec)

