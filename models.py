import torch
import torch.nn as nn
from torchaudio.models import Conformer

class CTCModel(nn.Module):
    def __init__(self, vocab_type, vocab_size, conformer_kwargs):
        super().__init__()
        self.conformer = Conformer(**conformer_kwargs)
        self.fc = nn.Linear(conformer_kwargs["input_dim"], vocab_size)

    def forward(self, x, x_lengths):
        logits, logits_lengths = self.conformer(x, x_lengths)
        logits = self.fc(logits)
        logits = logits.log_softmax(-1).transpose(0, 1)
        return logits, logits_lengths
