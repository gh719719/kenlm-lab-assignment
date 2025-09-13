import sentencepiece as spm
import kenlm
import numpy as np
import time

# 평가할 모델 목록 (모델 이름, 파일 경로)
models_to_evaluate = {
    "Tri-gram": "nsmc_3gram.bin",
    "5-gram": "nsmc_5gram.bin"
}

# SentencePiece 토크나이저 로드
sp = spm.SentencePieceProcessor(model_file='spm_bpe32k.model')
print("토크나이저 로딩 완료.")

# 각 모델에 대해 Perplexity 계산
for name, model_path in models_to_evaluate.items():
    print(f"\n===== {name} 모델 Perplexity 계산 시작 =====")
    start_time = time.time()
    model = kenlm.Model(model_path)
    
    total_log_prob = 0
    total_tokens = 0

    # nsmc_test.txt 파일을 한 줄씩 읽어서 처리
    with open('nsmc_test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 문장을 토큰화
            tokens = sp.encode(line, out_type=str)
            
            # KenLM 모델로 문장의 로그 확률(log probability) 계산
            log_prob = model.score(' '.join(tokens), bos=True, eos=True)
            
            total_log_prob += log_prob
            # 토큰 개수는 문장 시작(BOS)과 끝(EOS) 토큰을 포함하여 계산
            total_tokens += len(tokens) + 2

    # Perplexity 최종 계산 (PPL = e^(-1/N * sum(log(P))))
    ppl = np.exp(-total_log_prob / total_tokens)
    end_time = time.time()

    print(f"모델: {name}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"계산 시간: {end_time - start_time:.2f}초")
    print("=" * 40)