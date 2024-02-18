import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.compliance.kaldi as K

def cmvn(feature:torch.Tensor, dim_T=0, eps=1e-10):
    return (feature - feature.mean(dim=dim_T, keepdim=True)) / (feature.std(dim=dim_T, keepdim=True) + eps)

class Fbank(nn.Module):
    def __init__(self, n_mels=80, frame_length=25, frame_shift=10):
        super().__init__()
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def forward(self, waveform, sample_rate):
        fbank = K.fbank(
            waveform=waveform,
            sample_frequency=sample_rate,
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )
        return cmvn(fbank, dim_T=1) 
    

class MFCC(nn.Module):
    def __init__(self, n_mfcc=13, frame_length=25, frame_shift=10, use_delta=True):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.use_delta = use_delta

    def forward(self, waveform, sample_rate):
        mfcc = K.mfcc(
            waveform=waveform,
            sample_frequency=sample_rate,
            num_ceps=self.n_mfcc,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )
        if self.use_delta:
            mfcc_delta1 = F.compute_deltas(mfcc)
            mfcc_delta2 = F.compute_deltas(mfcc_delta1)
            mfcc = torch.cat([mfcc, mfcc_delta1, mfcc_delta2], dim=-1)
        return cmvn(mfcc, dim_T=1)
    
