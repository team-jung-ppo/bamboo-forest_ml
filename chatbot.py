'''
전체적인 구조
pre-trained된 모델이 사용자 입력을 encode한 값과  
wellness_dataset과 코사인 유사도를 계산하여 가장 높은 것과 매칭
'''

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def load_dataset(chatBotType):
    # chatBotType에 따라 다른 CSV 파일을 로드합니다.
    csv_files = {
        "아저씨": "oldman_dataset.csv",
        "아줌마": "oldgirl_dataset.csv"
        # 필요한 다른 타입의 데이터셋 파일들을 여기에 추가
    }
    
    # 기본값은 '아저씨'로 설정
    csv_file = csv_files.get(chatBotType, "oldman_dataset.csv")
    
    # 데이터셋 로드
    df = pd.read_csv(csv_file)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def get_answer(user_input, chatBotType):
    df = load_dataset(chatBotType)
    embedding = model.encode(user_input)
    embeddings = list(df['embedding'])
    distances = cosine_similarity([embedding], embeddings).squeeze()
    answer = df.iloc[distances.argmax()]
    return answer['챗봇']

def main():
    print("심리상담 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)")
    chatBotType = input("챗봇 타입을 입력하세요 (아저씨, 아줌마): ")
    
    while True:
        user_input = input("당신: ")
        if user_input.lower() == 'exit':
            print("챗봇: 대화를 종료합니다. 좋은 하루 되세요!")
            break
        
        response = get_answer(user_input, chatBotType)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main()
