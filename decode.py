import torch
import torch.nn as nn
from typing import List
from tokenizer import Tokenizer

# torch.manual_seed(1337)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, vocab_type: str):
        super().__init__()
        self.tokenizer = Tokenizer(vocab_type)

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]

        # remove consecutive repeating elements
        indices = torch.unique_consecutive(indices, dim=-1) 

        indices = indices.tolist()

        # remove pad and unk tokens
        indices = [idx for idx in indices if idx not in [0, 2]]

        # remove the token after eos
        if indices[-1] == 1:
            indices = indices[:-1]

        return self.tokenizer.decode(indices)

vocab_type = "char"
greedy_decoder = GreedyCTCDecoder(vocab_type = "char")
emission = torch.rand(50, 29)
transcript = greedy_decoder(emission)
print(transcript)
