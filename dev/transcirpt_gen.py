# generate the transcript file of all my LibriSpeech data
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from pathlib import Path

SPLITS = [
    "dev-clean",
    "train-clean-360",
]

LS_ROOT = "/opt/data/private/linkdom/data/LibriSpeech"
out_root = "/opt/data/private/linkdom/data/libri_features3"
out_root = Path(out_root).absolute()
transcript_file = out_root / "transcript.txt"

# open the transcript file
with transcript_file.open("w") as f:
    for split in SPLITS:
        dataset = LIBRISPEECH(
            root = out_root.as_posix(), 
            url = split, 
            folder_in_archive = LS_ROOT, 
            download = False
        )
        print(f"Processing {split}...")
        for wav, sample_rate, utter, spk_id, chapter_no, utt_no in tqdm(dataset):
            f.write(f"{utter.lower()}\n")
