# Version 1.0

## folder structure
vla_project/   
│   
├── configs/              # 실험 설정   
│   └── base.yaml   
│   
├── data/                 # 데이터 관련   
│   ├── dataset.py   
│   └── transforms.py   
│   
├── models/               # 모델 정의   
│   ├── vision_encoder.py   
│   ├── language_encoder.py   
│   ├── fusion.py   
│   └── policy.py        # 최종 VLA 모델   
│   
├── train/                # 학습 관련   
│   ├── train.py   
│   └── loss.py   
│   
├── eval/                 # 평가 코드   
│   └── evaluate.py   
│   
├── utils/                # 공통 유틸   
│   ├── logger.py   
│   └── misc.py   
│   
├── checkpoints/          # 모델 저장   
│   
└── main.py               # 실행 entry point   