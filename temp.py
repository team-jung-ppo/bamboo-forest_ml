import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from prompt_toolkit import prompt

# 모델 로드
start_time = time.time()
model = SentenceTransformer('jhgan/ko-sroberta-multitask', device='cpu')
print(f"모델 로딩 시간: {time.time() - start_time:.2f}초")

# 데이터셋 로드
start_time = time.time()
df = pd.read_csv('wellness_dataset.csv')
df['embedding'] = df['embedding'].apply(json.loads)
print(f"데이터셋 로딩 시간: {time.time() - start_time:.2f}초")

def get_answer(user_input):
    start_time = time.time()
    embedding = model.encode(user_input)
    print(f"인코딩 시간: {time.time() - start_time:.2f}초")

    start_time = time.time()
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    print(f"유사도 계산 시간: {time.time() - start_time:.2f}초")

    answer = df.loc[df['distance'].idxmax()]
    return answer['챗봇']

def main():
    print("심리상담 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)")
    while True:
        user_input = prompt("당신: ")
        if user_input.lower() == 'exit':
            print("챗봇: 대화를 종료합니다. 좋은 하루 되세요!")
            break
        response = get_answer(user_input)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main()
