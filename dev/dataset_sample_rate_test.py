# generate the transcript file of all my LibriSpeech data
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from pathlib import Path

SPLITS = [
    "dev-clean",
    "train-clean-360",
]

LS_ROOT = "/opt/data/private/linkdom/data/"

# open the transcript file
for split in SPLITS:
    dataset = LIBRISPEECH(
        root = LS_ROOT, 
        url = split, 
        download = False
    )
    print(f"Processing {split}...")
    for waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id in tqdm(dataset):
        sample_id = f"{speaker_id}-{chapter_id}-{utterance_id}"
        if sample_rate != 16000:
            print(f"Sample rate is {sample_rate} for {sample_id}")
            break
    print(f"{split} done.")

# all the sample rates are 16000
# (GPT) root@asr:~/AudioAttention# python dev/dataset_sample_rate_test.py 
# Processing dev-clean...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2703/2703 [00:27<00:00, 99.66it/s]
# dev-clean done.
# Processing train-clean-360...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 104014/104014 [39:05<00:00, 44.35it/s]
# train-clean-360 done.