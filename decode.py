import torch
import torch.nn as nn
from typing import List
from tokenizer import Tokenizer


class GreedyDecoder(torch.nn.Module):
    def __init__(self, vocab_type: str):
        super().__init__()
        self.tokenizer = Tokenizer(vocab_type)

    def forward(self, emission: torch.Tensor) -> List[str]:
        indices = torch.argmax(emission, dim=-1)  # (num_seq, vocab_size) -> (num_seq,)
        return self.tokenizer.decode(indices.tolist())

# vocab_type = "char"
# greedy_decoder = GreedyDecoder(vocab_type)
# emission = torch.rand(50, 29)
# transcript = greedy_decoder(emission)
# print(transcript)
