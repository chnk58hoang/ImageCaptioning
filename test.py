import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=corpus.txt --model_prefix=m --vocab_size=200')