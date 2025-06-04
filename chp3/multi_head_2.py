import torch 
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out  % num_heads ==0), \
        "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = (attn_weights @ values).transpose(1,2)

        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)
        context_vecs = self.out_proj(context_vecs)

        return context_vecs
    
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                    [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print(a @ a.transpose(2,3))
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)     

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("Second head: \n", second_res)

torch.manual_seed(123)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
batch_size, context_length, d_in = batch.shape
d_out = 2
print(batch.shape)

mha = MultiHeadSelfAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)

context_vecs = mha(batch)
print(context_vecs)
print(context_vecs.shape)


# # Exercise 3.3  

# torch.manual_seed(123)
# inputs = torch.tensor(
#     [[0.43, 0.15, 0.89], # Your     (x^1)
#      [0.55, 0.87, 0.66], # journey  (x^2)
#      [0.57, 0.85, 0.64], # starts   (x^3)
#      [0.22, 0.58, 0.33]] # with     (x^4)
# )

# batch = torch.stack((inputs, inputs), dim=0)
# batch2 = batch.repeat(1,1,256)
# batch2 = batch2.repeat(1, 256, 1)
# print(batch2.shape)
# batch_size, context_length, d_in = batch2.shape

# d_out = 768
# mha = MultiHeadSelfAttention(d_in, d_out, context_length, dropout=0.0, num_heads=12)

# context_vecs = mha(batch2)
# print(context_vecs)
# print(context_vecs.shape)
