import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 검색 모델 로드 (DPR)
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# 생성 모델 로드 (T5)
generation_model = T5ForConditionalGeneration.from_pretrained("t5-small")
generation_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 데이터셋 로드
df = pd.read_csv('wellness_dataset.csv')
df['context_embedding'] = df['context'].apply(lambda x: context_encoder(**context_tokenizer(x, return_tensors='pt')).pooler_output)

def get_relevant_contexts(question, top_k=5):
    question_embedding = question_encoder(**question_tokenizer(question, return_tensors='pt')).pooler_output
    similarities = df['context_embedding'].apply(lambda x: cosine_similarity(question_embedding, x).squeeze())
    top_k_indices = similarities.nlargest(top_k).index
    return df.loc[top_k_indices, 'context']

def generate_answer(question):
    relevant_contexts = get_relevant_contexts(question)
    input_text = question + " " + " ".join(relevant_contexts)
    input_ids = generation_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = generation_model.generate(input_ids)
    answer = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    print("심리상담 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)")
    while True:
        user_input = input("당신: ")
        if user_input.lower() == 'exit':
            print("챗봇: 대화를 종료합니다. 좋은 하루 되세요!")
            break
        response = generate_answer(user_input)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main()
