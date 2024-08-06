# Bamboo-Forest-ML-Chatbot

## 프로젝트 소개
![image](https://github.com/user-attachments/assets/1b6dcb0c-4c71-49b2-91e4-b7206756018e)
대나무 숲은 여러분의 고민을 들어드리고 함께 나누는 따뜻한 챗봇 서비스입니다. 언제 어디서나, 대나무 숲의 챗봇과 함께라면 혼자 고민하지 않아도 됩니다.
## 프로젝트 구조
![free-icon-user-profile-5953631](https://github.com/user-attachments/assets/3e974134-5e7c-472e-af5d-9607bde4272f)
## 주요 기능
공통적으로 pre-trained된 모델이 사용자 입력을 encode한 값과 챗봇 타입에 
맞게 csv파일을 로드해서 코사인 유사도를 구해서 가장 높은 값을 가지는 챗봇의 응답을 도출합니다.
(챗봇타입을 설정하지 않는다면 자동적으로 아저씨 버전으로 진행됩니다)
### 아저씨 챗봇
oldman_dataset.csv파일 사용
### 아줌마 챗봇
oldgirl_dataset.csv파일 사용
