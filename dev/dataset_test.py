from dataset import LibriSpeechDataset, SPLITS
dataset = LibriSpeechDataset(SPLITS[0], "char")

print(len(dataset))
print(dataset[0][0].shape, dataset[0][1])