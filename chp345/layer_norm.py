import torch
import torch.nn as nn
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
# print("Mean: \n", mean)
# print("Variance: \n", var)

out_norm = (out - mean)/torch.sqrt(var)
print(out_norm)
mean_norm = out_norm.mean(dim=-1, keepdim = True)
var_norm = out_norm.var(dim=-1, keepdim = True)
torch.set_printoptions(sci_mode = False)
print("Mean norm: \n", mean_norm)
print("variance norm: \n", var_norm)