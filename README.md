# Bamboo-Forest-ML-Chatbot
***
## 프로젝트 소개
***
아줌마/아저씨 버전으로 채팅 기능 제공하는 챗봇입니다.
***
## 주요 기능
***
공통적으로 'jhgan/ko-sroberta-multitask'으로 SentenceTransformer으로 임베딩을 진행하고 챗봇 타입에 
맞게 csv파일을 로드해서 코사인 유사도를 구해서 가장 높은 값을 가지는 챗봇의 응답을 도출합니다.
(챗봇타입을 설정하지 않는다면 자동적으로 아저씨 버전으로 진행됩니다)
### 아저씨 챗봇
oldman_dataset.csv파일 사용
### 아줌마 챗봇
wellness_dataset.csv파일 사용
