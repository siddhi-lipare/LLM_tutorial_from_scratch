from functools import partial
from dataset_prep import custom_collate_fn, InstructionDataset, tokenizer, train_data, val_data, test_data
import torch
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = custom_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn = custom_collate_fn,
    shuffle=True
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn = custom_collate_fn,
    shuffle=False,
    drop_last= True,
    num_workers=num_workers
)

print("Train loader: ")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)


