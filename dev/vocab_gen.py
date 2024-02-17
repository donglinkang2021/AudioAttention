import sentencepiece as spm

text_file = '/opt/data/private/linkdom/data/libri_features3/transcript.txt'
model_type = 'char'
model_prefix = f'/root/AudioAttention/vocab/librispeech_{model_type}'

# Train the sentencepiece model
spm.SentencePieceTrainer.train(f'--input={text_file} --model_prefix={model_prefix} --model_type={model_type}')
