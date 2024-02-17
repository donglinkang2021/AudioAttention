import torch
import numpy as np
from torchaudio.datasets import LIBRISPEECH
from pathlib import Path

SPLITS = [
    "dev-clean",
    "train-clean-360",
]

LS_ROOT = "/opt/data/private/linkdom/data/LibriSpeech"
out_root = "/opt/data/private/linkdom/data/libri_features3"
# label_dir = "/opt/data/private/linkdom/data/libri_kmeans/lab" # pseudo labels


out_root = Path(out_root).absolute()
feature_root = out_root / "fbank80"

for split in SPLITS:
    dataset = LIBRISPEECH(
        root = out_root.as_posix(), 
        url = split, 
        folder_in_archive = LS_ROOT, 
        download = False
    )
    for wav, sample_rate, utter, spk_id, chapter_no, utt_no in dataset:
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        print(wav.shape, sample_rate, utter.lower(), sample_id)

        fbank_path = feature_root / f"{sample_id}.npy"
        fbank = np.load(fbank_path, allow_pickle=True) 
        fbank = torch.from_numpy(fbank).float() 
        print(fbank.shape)

        break
    break