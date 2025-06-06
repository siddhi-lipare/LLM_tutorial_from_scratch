import torch
from attn_class_v2 import selfattn_v2, SelfAttentionV2
from masked import attn_weights_2, inputs

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)

# print(dropout(example))
print(dropout(attn_weights_2))

batch = torch.stack((inputs, inputs), dim=0)

print(batch.shape)
