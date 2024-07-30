'''
전체적인 구조
pre-trained된 모델이 사용자 입력을 encode한 값과  
wellness_dataset과 코사인 유사도를 계산하여 가장 높은 것과 매칭
'''

import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from prompt_toolkit import prompt

# 모델 로드
start_time = time.time()
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print(f"모델 로딩 시간: {time.time() - start_time:.2f}초")

# 데이터셋 로드
start_time = time.time()
df = pd.read_csv('wellness_dataset.csv')
df['embedding'] = df['embedding'].apply(json.loads)
print(f"데이터셋 로딩 시간: {time.time() - start_time:.2f}초")

def get_answer(user_input, context):
    # 입력 문장과 문맥을 결합
    combined_input = context + " " + user_input
    
    # 사용자 입력 인코딩
    start_time = time.time()
    embedding = model.encode(combined_input)
    print(f"인코딩 시간: {time.time() - start_time:.2f}초")

    # 유사도 계산
    start_time = time.time()
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    print(f"유사도 계산 시간: {time.time() - start_time:.2f}초")

    # 가장 유사한 답변 찾기
    answer = df.loc[df['distance'].idxmax()]
    return answer['챗봇'], combined_input

def main():
    print("심리상담 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)")
    context = ''
    while True:
        user_input = prompt("당신: ")
        if user_input.lower() == 'exit':
            print("챗봇: 대화를 종료합니다. 좋은 하루 되세요!")
            break
        response, context = get_answer(user_input, context)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main()

