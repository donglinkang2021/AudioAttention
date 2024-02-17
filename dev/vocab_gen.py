import sentencepiece as spm

text_file = '/opt/data/private/linkdom/data/libri_features3/transcript.txt'
model_type = 'bpe'
model_prefix = f'/root/AudioAttention/vocab/librispeech_{model_type}'
idx_setting = '--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --eos_piece=<eos>'

# Train the sentencepiece model
spm.SentencePieceTrainer.train(f'--input={text_file} --model_prefix={model_prefix} --model_type={model_type} {idx_setting} --vocab_size=1000')