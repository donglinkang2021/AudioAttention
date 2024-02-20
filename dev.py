import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import CTCModel
from dataset import get_dataloader, SPLITS
from decode import GreedyDecoder
import jiwer

np.random.seed(1337)
torch.manual_seed(1337)

# ---------config----------
## model
conformer_kwargs = {
    "input_dim": 80,
    "num_heads": 4,
    "ffn_dim": 128,
    "num_layers": 6,
    "depthwise_conv_kernel_size": 31,
    "dropout": 0.1,
}

ctcmodel_kwargs = {
    "vocab_type": "char",
    "vocab_size": 30,
    "conformer_kwargs": conformer_kwargs,
}

model_name = "conformer_ctc"

## train
batch_size = 128
save_begin = 30
learning_rate = 1e-2
num_epochs = 1
eval_interval = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------config----------

model = CTCModel(**ctcmodel_kwargs)
model.to(device)
valloader = get_dataloader(SPLITS[0], batch_size)
trainloader = get_dataloader(SPLITS[1], batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CTCLoss()
decoder = GreedyDecoder(vocab_type = "char")

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    losses = []
    split = "val"

    # compute loss
    for x, y, lx, ly in tqdm(valloader, ncols=100, desc="Eval processing", leave=False):
        x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
        y_pred, y_pred_lengths = model(x, lx)
        loss = criterion(y_pred, y, y_pred_lengths, ly)
        losses.append(loss.item())
    metrics[split + '_loss'] = np.mean(losses)

    # compute wer for the last batch
    wer = 0
    batch_size = x.size(0)
    for i in tqdm(range(batch_size), ncols=100, desc="WER computing", leave=False):
        transcript = decoder(y_pred[i])
        target = decoder.tokenizer.decode(y[i].tolist())
        wer += jiwer.wer(target, transcript)
    print(f"transcript: {transcript}")
    print(f"target: {target}")
    wer /= batch_size
    metrics[split + '_wer'] = wer

    model.train()
    return metrics

best_loss = 1000
n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
for epoch in range(num_epochs):
    for i, (data, target, data_lengths, target_lengths) in enumerate(trainloader):
        iter = epoch * n_batches + i
        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"--- eval batch {iter}:", end=' ')
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}", end=' ')
            print("---")

            if iter > save_begin and metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                torch.save(model.state_dict(), f'checkpoint/best_{model_name}.pth')
        
        data, target, data_lengths, target_lengths = data.to(device), target.to(device), data_lengths.to(device), target_lengths.to(device)

        out, out_lengths = model(data, data_lengths)
        loss = criterion(out, target, out_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"batch {iter}, loss: {loss.item()}")

