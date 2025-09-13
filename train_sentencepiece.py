# train_sentencepiece.py
import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    '--input=nsmc_train.txt --model_prefix=spm_bpe32k '
    '--vocab_size=32000 --model_type=bpe '
    '--character_coverage=0.9995 --byte_fallback=true'
)
print("SentencePiece 모델(spm_bpe32k.model) 생성 완료.")