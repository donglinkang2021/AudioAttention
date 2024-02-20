import torch
import torch.nn as nn
from torchaudio.models import Conformer
from tokenizer import Tokenizer

class ASRModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_lengths):
        raise NotImplementedError
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class CTCModel(ASRModel):
    def __init__(self, vocab_type, vocab_size, conformer_kwargs):
        super().__init__()
        self.tokenizer = Tokenizer(vocab_type)
        self.conformer = Conformer(**conformer_kwargs)
        self.fc = nn.Linear(conformer_kwargs["input_dim"], vocab_size)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def forward(self, x, x_lengths):
        logits, logits_lengths = self.conformer(x, x_lengths)
        logits = self.fc(logits)
        logits = logits.log_softmax(-1)
        logits = logits.transpose(0, 1) # BxTxC -> TxBxC
        return logits, logits_lengths

class LinearModel(ASRModel):
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(input_dim, vocab_size)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def forward(self, x, x_lengths):
        x = self.fc(x)
        x = x.log_softmax(-1)
        x = x.transpose(0, 1) # BxTxC -> TxBxC
        return x, x_lengths