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
