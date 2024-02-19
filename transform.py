# kaldi.py and transform.py are equivalent. The difference is that kaldi.py does not support batch processing but transform.py does.

import torch
import torch.nn as nn
import torchaudio.functional as F
from torchaudio.transforms import MelSpectrogram, MFCC

def cmvn(feature:torch.Tensor, dim_T=0, eps=1e-10):
    return (feature - feature.mean(dim=dim_T, keepdim=True)) / (feature.std(dim=dim_T, keepdim=True) + eps)

# fbank80
melspec_kwargs = {
    "sample_rate": 16000,
    "n_mels": 80,
    "n_fft": 400,
    "hop_length": 160,
    "center": False,
}

# mfcc39
melkwargs = {
    "n_mels": 23,
    "n_fft": 400,
    "hop_length": 160,
    "center": False,
}

mfcc_kwargs = {
    "sample_rate": 16000,
    "n_mfcc": 13,
    "log_mels": True,
    "melkwargs": melkwargs,
}

class FBANK80(nn.Module):
    def __init__(self, melspec_kwargs):
        super().__init__()
        self.transform = MelSpectrogram(**melspec_kwargs)
        self.log_offset = 1e-6

    def forward(self, x):
        x = self.transform(x) + self.log_offset # (B, n_mels, T)
        x = x.log()                             # (B, n_mels, T)
        x = x.transpose(1, 2)                   # (B, T, n_mels)
        x = cmvn(x, dim_T=1)                    # (B, T, n_mels)
        return x

class MFCC39(nn.Module):
    def __init__(self, mfcc_kwargs):
        super().__init__()
        self.transform = MFCC(**mfcc_kwargs)

    def forward(self, x):
        x = self.transform(x)                   # (B, n_mfcc, T)
        d1 = F.compute_deltas(x)                # (B, n_mfcc, T)
        d2 = F.compute_deltas(d1)               # (B, n_mfcc, T)
        x = torch.cat([x, d1, d2], dim=1)       # (B, n_mfcc*3, T)
        x = x.transpose(1, 2)                   # (B, T, n_mfcc*3)
        x = cmvn(x, dim_T=1)
        return x
