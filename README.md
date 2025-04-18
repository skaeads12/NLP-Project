# NLP-Project

## 사용 방법

```shell
python main.py
```

**파라미터**
|param|full param|type|default|description|
|---:|---:|:---:|:---:|:---|
|-d|--data_dir|str|/workspace/nlp-project/data/humaneval.jsonl|입력 데이터 경로(HumanEval 파이썬 파일 기준)|
|-s|--save_dir|str|/workspace/nlp-project/outputs|모델 응답 출력 파일 저장 경로|
|-m|--model|str|bigcode/starcoderbase-1b|모델 이름 or 경로(huggingface 기준)|
|-b|--batch_size|int|32|배치 크기, 메모리 터지면 숫자 줄이기|
|-ml|--max_length|int|1024|모델이 출력할 수 있는 최대 길이(입력 길이 포함), HumanEval 파이썬 파일 기준으로 프롬프트의 최대 길이가 400이므로, 넉넉하게 512이상|

**사용 예시**
```shell
cd /workspace/nlp-project
python main.py \
-d data/humaneval-cpp.jsonl \
-s result/cpp \
-m bigcode/starcoderbase-1b \
-b 16 \
-ml 512
```

> StarCoder 1B 모델을 로드하는 데 1분 이상 소요될 수 있습니다.  
> 경우에 따라 huggingface 로그인이 필요할 수 있습니다. 이는 (```pip install -U "huggingface_hub[cli]"```) 설치가 되어있다면 ```huggingface-cli login```로 수행할 수 있습니다.

## 파일 구조

**dataset.py**

- 데이터 세트를 로드하고 모델 입력으로 가공하는 클래스 정의

**main.py**

- 모델 추론하는 핵심 코드

**data/**

- 데이터 세트 저장 경로
