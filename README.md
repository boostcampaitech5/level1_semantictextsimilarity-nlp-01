# STS (Semantic Textual Similarity) 대회
- boostcamp AI Tech 5 NLP 트랙 레벨1 프로젝트
- 문장 간 유사도 (STS; Semantic Textual Similarity) 측정 대회

## 일정 (Schedule)
프로젝트 전체 기간 (10일+) : 4월 10일 (월) 10:00 ~ 4월 20일 (목) 19:00

## 프로젝트 구조 (Project Structure)
<img src="https://user-images.githubusercontent.com/37128004/233752744-9becc593-3457-4d00-bb6c-79a0423edcdc.png" width="700" height="500"/>


## 데이터 (Data)
- 총 데이터 개수 : 10,974 문장 쌍
- Train 데이터 개수: 9,324
- Test 데이터 개수: 1,100
- Dev 데이터 개수: 550
- 데이터셋의 train:dev:test 비율 = 85:5:10
- Label 점수: 0 ~ 5사이의 실수
  - 5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함
  - 4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음
  - 3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
  - 2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함
  - 1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음
  - 0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음

## 프로젝트 구조 (Project structure)
```
level1_semantictextsimilarity-nlp-01/
│
├── train.py                // 학습 시작을 위한 메인 스크립트
├── inference.py            // 학습 모델의 평가 및 추론을 위한 스크립트
│
├── config.json             // 학습 설정 관리를 위한 JSON
├── args.py                 // 학습을 위한 메인 스크립트와 학습 설정 연결
│
├── lr_scheduler.py         // 학습률 스케줄러
├── sweep.py                // 하이퍼파라미터 서치를 위한 스크립트
│
├── dataloader/             // 데이터 불러오기에 관한 모든 것
│   ├── dataset.py
│   └── dataloader.py
│
├── data/                   // 입력 데이터 저장을 위한 기본 저장소
│
├── lightning_logs/         // PyTorch lightning 자동 생성 로깅 출력
│
├── models/                 // 모델, 손실 함수, 최적화 알고리즘, 평가 지표
│   ├── gru_model.py
│   └── model.py            // 베이스라인 모델
│
├── saved/
│   └── models/             // 학습 모델의 저장소
│  
└── utils/                  // 유틸리티 함수
    ├── back_translate.py   // 역번역을 위한 모듈
    └── ...
```

## 사용법 (Usage)
- 환경 설치

  `pip install -r requirements.txt`
- 학습

  `python train.py`
- 추론

  `python inference.py`
- 하이퍼파라미터 튜닝

  `python sweep.py`
- 역번역 (실행 파일의 권한을 부여할 수 있는 로컬 환경에서)

  `python utils/translate.py`

### 설정 파일 형식 (Config file format)
```
{
    "name": "exp-name",
    "n_gpu": 1,
    "arch": {
        "type": "snunlp/KR-ELECTRA-discriminator",
        "args": {

        }
    },
    "dataloader": {
        "type": "STSDataModule",
        "args": {
            "batch_size": 16,
            "shuffle": true,
            "dataset_commit_hash": "00e11acc1364f0f222777f8d828e0a8cf0de265c",
            "num_workers": -1,
            "k_fold": {
                "enabled": false,
                "k": 5
            }
        }
    },
    "optimizer": {
        "type": "AdamW",
        "args": {
            "lr": 1e-5,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": {
        "type": "SmoothL1Loss",
        "args": {
            "bce": false,
            "beta": 0.5
        }
    },
    "metrics": ["val_pearson"],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,          
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 1,
        "save_dir": "saved/models/",
        "save_freq": 1,
        "monitor": "max val_pearson",
        "early_stop": 3
    },
    "wandb": {
        "entity": "salmons",
        "project_name": "sts",
        "sweep_count": 10
    },
    "sweep_config": {
        "method": "bayes",
        "metric": {
            "name": "val_pearson",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "values": [3e-6, 5e-6, 7e-6, 1e-5, 2e-5, 3e-5, 5e-5]
            },
            "epochs": {
                "values": [4, 5, 6, 7]
            },
            "batch_size" : {
                "values" : [16]
            },
            "loss_function" :{
                "values" : ["SmoothL1Loss"]
            },
            "beta":{
                "values" : [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            "optimizer":{
                "values" : ["AdamW"]
            }
        }
    }
}
```
