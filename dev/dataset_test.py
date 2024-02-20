import torch.nn as nn
from dataset import *
dataset = LibriSpeechDataset(SPLITS[0], "char")

# print(len(dataset))
# print(dataset[0][0].shape, dataset.tokenizer.decode(dataset[0][1].tolist()))

loss_fn = nn.CTCLoss()

dataloader = get_dataloader(SPLITS[0], 3)
for x, y, lx, ly in dataloader:
    print(x.shape, y.shape, lx.shape, ly.shape)
    print(lx, ly)
    # print(dataset.tokenizer.decode(y.tolist()))
    x = x.log_softmax(2).transpose(0, 1)
    loss = loss_fn(x, y, lx, ly)
    print(loss)
    break