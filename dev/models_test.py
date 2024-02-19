import torch
import torch.nn as nn
from torchaudio.models import Conformer
from einops import rearrange

torch.manual_seed(2024)

input_dim = 80
num_heads = 4
ffn_dim = 128
num_layers = 4
depthwise_conv_kernel_size = 31
batch_size = 32
vocab_size = 29
dropout = 0.1

conformer = Conformer(
    input_dim=input_dim,
    num_heads=num_heads,
    ffn_dim=ffn_dim,
    num_layers=num_layers,
    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
    dropout=dropout,
)

lengths = torch.randint(200, 400, (batch_size,))  # (batch,)
T = int(lengths.max())
input = torch.rand(batch_size, T, input_dim)  # (batch, num_frames, input_dim)
out_frames, out_lengths = conformer(input, lengths)

class LossCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(input_dim, vocab_size)
        self.loss = nn.CTCLoss()

    def forward(self, x, lengths, targets, target_lengths):
        logits = self.fc(x)
        # logits = rearrange(logits, 'B T C -> T B C')
        logits = logits.log_softmax(-1).transpose(0, 1)
        loss = self.loss(logits, targets, lengths, target_lengths)
        return loss


loss_fn = LossCTC()
S = 100
targets = torch.randint(low=1, high=vocab_size, size=(batch_size, S))
target_lengths = torch.randint(1, 100, (batch_size,))
loss = loss_fn(out_frames, out_lengths, targets, target_lengths)
loss.backward()
        

print("conformer input", end = " ")
print(input.shape, lengths.shape) 
print("conformer output", end = " ")
print(out_frames.shape, out_lengths.shape)
print("target", end = " ") 
print(targets.shape, target_lengths.shape)
print("loss", end = " ")
print(loss.item())