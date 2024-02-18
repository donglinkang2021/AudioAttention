import torchaudio
import torch

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

wave = torch.rand(1, 400)
sample_rate = 16000


# transforms fbank
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=23,
    n_fft=400,
    hop_length=160,
)
tr_fbank = transform(wave).squeeze(0).transpose(0, 1)

# kaldi compliance fbank
kaldi_fbank = torchaudio.compliance.kaldi.fbank(
    wave,
    num_mel_bins = 23,
    use_energy = False,
    window_type="hanning"
)

ft = torch.cat([tr_fbank, kaldi_fbank]).transpose(0, 1)

print(tr_fbank.shape)
print(kaldi_fbank.shape)
print(ft.shape)
print(ft)