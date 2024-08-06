from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)

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

@app.route('/get-response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data.get("message")
    chatBotType = data.get("chatBotType", "아저씨")
    response = get_answer(user_input, chatBotType)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
