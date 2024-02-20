import numpy as np
from fast_ctc_decode import beam_search, viterbi_search
from tokenizer import Tokenizer
tokenizer = Tokenizer("char")
vocab_list = ['<pad>', '<eos>', '<unk>', '▁', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', 'm', 'c', 'w', 'f', 'g', 'y', 'p', 'b', 'v', 'k', "'", 'x', 'j', 'q']
alphabet = "NACGT"
posteriors = np.random.rand(100, len(vocab_list)).astype(np.float32)
seq, path = viterbi_search(posteriors, vocab_list)
print(seq)
seq, path = beam_search(posteriors, vocab_list, beam_size=3, beam_cut_threshold=0.03)
print(seq)