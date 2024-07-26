from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import math

app = Flask(__name__)

# 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 데이터셋 로드
df = pd.read_csv('wellness_dataset.csv')
df['embedding'] = df['embedding'].apply(json.loads)

def get_answer(user_input):
    embedding = model.encode(user_input)
    embeddings = list(df['embedding'])
    distances = cosine_similarity([embedding], embeddings).squeeze()
    answer = df.iloc[distances.argmax()]
    return answer['챗봇']

@app.route('/get-response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data.get("message")
    response = get_answer(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
