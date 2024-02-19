from decode import GreedyCTCDecoder
import jiwer
from dataset import get_dataloader, SPLITS
from tokenizer import Tokenizer

batch_size = 64
valloader = get_dataloader(SPLITS[0], batch_size)

for x, y, lx, ly in valloader:
    break

tokenizer = Tokenizer("char")

# compute wer for the last batch
wer = 0
batch_size = x.size(0)
for i in range(batch_size):
    transcript = tokenizer.decode(y[i].tolist())
    target = tokenizer.decode(y[i].tolist())
    wer += jiwer.wer(target, transcript)
print(f"transcript: {transcript}")
print(f"target: {target}")
wer /= batch_size
print(f"wer: {wer:.4f}")