# prepare_data.py
import pandas as pd
# 'nsmc' 폴더 안의 데이터를 읽도록 경로 지정
train_df = pd.read_csv('nsmc/ratings_train.txt', sep='\t')
test_df = pd.read_csv('nsmc/ratings_test.txt', sep='\t')
with open('nsmc_train.txt', 'w', encoding='utf-8') as f:
    for doc in train_df['document'].dropna():
        f.write(doc + '\n')
with open('nsmc_test.txt', 'w', encoding='utf-8') as f:
    for doc in test_df['document'].dropna():
        f.write(doc + '\n')
print("nsmc_train.txt와 nsmc_test.txt 생성 완료.")