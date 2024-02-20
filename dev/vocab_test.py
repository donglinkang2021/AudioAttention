import sentencepiece as spm

text_file = '/opt/data/private/linkdom/data/libri_features3/transcript.txt'
model_type = 'char'
model_prefix = f'/root/AudioAttention/vocab/librispeech_{model_type}'

# Load the sentencepiece model
sp = spm.SentencePieceProcessor()

# Load the trained model
sp.load(f'{model_prefix}.model')

# Encode the text
encoded_text = sp.encode_as_ids('hello world')
print(encoded_text)

# Decode the text
decoded_text = sp.decode_ids(encoded_text)
print(decoded_text)

# Get the size of the vocabulary
vocab_size = sp.get_piece_size()
print(vocab_size)

print(sp.bos_id(), sp.pad_id(), sp.eos_id(), sp.unk_id())

vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]
print(vocab_list)