import torch

input_ids = [2, 3, 5, 1]

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight) # weight matrix has 6 rows and 3 columns.
# 1 row for each of the the possible six tokens in the vocabulary, 1 column for each of the three embedding dimensions

print(embedding_layer(torch.tensor([3])))

