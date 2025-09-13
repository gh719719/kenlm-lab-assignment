# tokenize_for_kenlm.py
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='spm_bpe32k.model')
with open('nsmc_train.txt', 'r', encoding='utf-8') as f_in, \
     open('corpus.txt', 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if line.strip():
            f_out.write(' '.join(sp.encode(line.strip(), out_type=str)) + '\n')
print("KenLM 학습용 corpus.txt 생성 완료.")