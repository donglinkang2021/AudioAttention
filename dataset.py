import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tokenizer import Tokenizer

SPLITS = [
    "dev-clean",
    "train-clean-360",
]

DATA_ROOT = "/opt/data/private/linkdom/data/"

class LibriSpeechDataset(Dataset):
    def __init__(self, split: str, vocab_type: str = "char"):
        self.split = split
        self.data = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
        self.vocab_type = vocab_type
        self.tokenizer = Tokenizer(vocab_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, _, utterance, _, _, _ = self.data[idx]
        utterance = torch.LongTensor(self.tokenizer.encode(utterance))
        return waveform, utterance


def collate_fn(batch):
    # Sorting sequences by lengths
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

    # Pad data sequences
    data = [item[0].squeeze() for item in sorted_batch]
    data_lengths = torch.tensor([len(d) for d in data],dtype=torch.long) 
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    # Pad labels
    target = [item[1] for item in sorted_batch]
    target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

    return data, target, data_lengths, target_lengths



batch_size = 32
dataloader = DataLoader(LibriSpeechDataset(SPLITS[0]), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
for i, (data, target, data_lengths, target_lengths) in enumerate(dataloader):
    print(data.shape, target.shape, data_lengths.shape, target_lengths.shape)
    if i == 0:
        break

