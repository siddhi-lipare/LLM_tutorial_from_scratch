import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i,query) # dot product gives similarity
print(attn_scores_2)

# # normalising attention scores
# attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum() # Normalises

# print("Attentiion weights Normalised: ", attn_scores_2_tmp)
# # print("Sum: ", attn_scores_2_tmp.sum()) # will sum up to one

# # using softmax naively for normalising attention scores

# def softmax_naive(x):
#     return torch.exp(x)/ torch.exp(x).sum(dim=0)

# attn_scores_2_softmax_naive= softmax_naive(attn_scores_2)
# print("Attention weights (Softmax naive): ", attn_scores_2_softmax_naive)
# # print("Sum: ", attn_scores_2_softmax_naive.sum())

# using torch.softax

attn_scores_2_softmax = torch.softmax(attn_scores_2, dim=0)
print("Attention weights (torch.softmax): ", attn_scores_2_softmax)
print("Sum: ", attn_scores_2_softmax.sum())

query = inputs[1]
contex_vec_2 = torch.empty(query.shape)
for i, x_i in enumerate(inputs):
    contex_vec_2 += attn_scores_2_softmax[i]*x_i 

print(contex_vec_2)