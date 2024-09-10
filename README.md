# 대화 맥락 추론 Baseline
본 리포지토리는 2024.07.01 ~ 2024.08.23에 시행된 '2024년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '대화 맥락 추론 (가형)'에 참가한 소스코드 및 후기를 포함하고 있습니다.

|Model|Accuracy(%)|
|:---|---:|
|google/gemma-2-9b-it (Qlora)|94.38|

|Parameter|Value|
|---|---|
|epoch|3|
|rank|32|
|alpha|64|
|lr|2e-4|
|dropout|0.01|

총 62개 팀 중 19위를 기록하였습니다.

<img width="933" alt="image" src="https://github.com/user-attachments/assets/c23224b2-5bd3-4eca-bc4b-c4064ed85947">


## 리포지토리 구조 (Repository Structure)
```
# 추론 결과 보관 디렉토리
answer

# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py
└── evaluate_accuracy.py
└── generate_answers.py

# 학습에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
{
    "id": "nikluge-2024-대화 맥락 추론-train-000001",
    "input": {
        "conversation": [
            {
                "speaker": 2,
                "utterance": "진짜 신의 한수",
                "utterance_id": "MDRW2100003410.1.1"
            },
            {
                "speaker": 1,
                "utterance": "이사하자마자 비 많이 와서 베란다 물 많이 새는 거 알았잖아",
                "utterance_id": "MDRW2100003410.1.2"
            },
            {
                "speaker": 2,
                "utterance": "글치 계속 해떴으면 몰랐겠지",
                "utterance_id": "MDRW2100003410.1.3"
            },
            ...
            ...
            ...
        ],
        "reference_id": [
            "MDRW2100003410.1.11"
        ],
        "category": "원인",
        "inference_1": "화자2가 사는 곳 근처에서 베란다 보수 공사가 진행되고 있다.",
        "inference_2": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 진행되고 있다.",
        "inference_3": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 중단되었다."
    },
    "output": "inference_2" # The Correct answer is inference_2
}
```

## Reference 
공식 Baseline (Teddysum) (https://github.com/teddysum/Korean_CCI_2024)

국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)

대회 리더보드 (https://huggingface.co/spaces/Korean-AI-Malpyeong-Leaderboard/Korean-AI-Malpyeong-Leaderboard)

## 후기
1. 경진대회 참가 환경
   
대회 초기에는 runpod에서 3090 * 1 서버를 임대하여 진행하였고 중반 이후부터 4090 * 2 서버로 진행하였습니다.

2. 테스트 모델

|model|
|---|
|MLP-KTLim/llama-3-Korean-Bllossom-8B|
|**google/gemma-2-9b-it**|
|google/gemma-2-27b-it|

베이스라인 모델인 llama-3-Korean-Bllossom-8B로는 90점을 넘지 못해 gemma-2-9b-it 모델로 변경하였고 gemma-2-27b-it 모델은 파인튜닝 후 추론 과정에서 결과가 제대로 나오지 않는 오류를 수정하지 못해 gemma-2-9b-it 모델로 대회를 마무리했습니다.

3. 자체평가

추론 결과 파일은 하루에 5번 국립국어원 서버에 업로드 하는 제한이 있고 특정 시간대에는 점수가 측정되는 텀이 길어 대회 중반부터 gpt-4o API를 사용하여 자체 답안지를 생성(generate_answers.py)했습니다.
그 후 파인튜닝 모델 추론 결과 파일과 gpt 답안지를 비교(evaluate_accuracy.py)하여 가장 점수가 높은 추론 파일을 업로드하는 방식을 사용하였습니다.

 
 
