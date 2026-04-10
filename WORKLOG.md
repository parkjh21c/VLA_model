# VLA 작업 기록

## 1. 현재 프로젝트 구조

현재 프로젝트는 초기 프로토타입 단계이므로, 폴더 구조를 크게 쪼개지 않고 단순한 형태를 유지하고 있다.

```text
VLA/
├── configs/
├── data/
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── fusion.py
│   ├── language_encoder.py
│   ├── policy.py
│   └── vision_encoder.py
├── train/
│   ├── loss.py
│   └── train.py
├── external/
│   └── boxbrown-mydataset/
├── checkpoints/
├── .cache/
└── WORKLOG.md
```

프로토타입 단계에서는 이 정도 구조로도 충분하다고 판단했고, 구조 자체는 유지하기로 했다.

## 2. 데이터셋 개요

사용 중인 데이터셋:

- 로컬 경로: [external/boxbrown-mydataset](/home/user/background/project/VLA/external/boxbrown-mydataset)
- 원본: Hugging Face `boxbrown/mydataset`

확인한 내용:

- LeRobot v3 스타일 구조를 따름
- 멀티모달 로봇 데이터셋
- action 차원: `[6]`
- observation.state 차원: `[6]`
- 카메라 2개:
  - `camera1`
  - `camera2`

현재 학습 전제:

- 총 270개 episode만 사용
- `episode 0`, `episode 271`은 제외
- 유효 episode는 사실상 `1 ~ 270`
- 기본 카메라는 `camera1`

## 3. 왜 `lerobot`를 직접 쓰지 않았는가

처음에는 데이터 기록 버전에 맞추기 위해 `lerobot==0.4.3` 사용을 시도했다.

하지만 `vla` conda 환경에서 아래 문제가 있었다.

- `accelerate` 등 필수 의존성 누락
- `torch`, `torchvision`, `datasets`, `huggingface_hub` 버전 충돌
- `LeRobotDataset` import 불안정
- 현재 학습 환경을 유지하면서 맞추기 어려움

결론:

- 학습 파이프라인은 `lerobot`에 직접 의존하지 않음
- 대신 `pandas + parquet + ffmpeg`로 로컬 데이터셋을 직접 읽는 방식으로 전환

이 방식이 현재 `vla` 환경을 덜 깨고, 디버깅도 더 단순했다.

## 4. 현재 데이터 로더 동작 방식

파일:

- [data/dataset.py](/home/user/background/project/VLA/data/dataset.py)

현재 `VLADataset`은 다음을 수행한다.

- `meta/info.json` 읽기
- `meta/tasks.parquet` 읽기
- `meta/episodes/**/*.parquet` 읽기
- `data/**/*.parquet` 읽기
- `episode 0`, `episode 271` 제거
- 각 frame을 다음 형태로 반환:

```python
image, text, action
```

세부 내용:

- `image`: 비디오에서 해당 시점 frame을 `ffmpeg`로 추출 후 transform 적용
- `text`: `task_index`에 대응되는 task 문장
- `action`: shape `[6]`의 `torch.float32` tensor

중요한 점:

- 프레임을 샘플마다 `ffmpeg` subprocess로 뽑기 때문에 느릴 수 있다.
- 하지만 `PyAV`, `lerobot`, `torchvision` 비디오 backend 문제를 피할 수 있다는 장점이 있다.

## 5. 모델 관련 수정 사항

### `models/policy.py`

- `action_dim`을 `7`에서 `6`으로 수정
- 데이터셋 action 차원과 맞춤

### `models/vision_encoder.py`

- `forward` 들여쓰기 버그 수정
- `ViT-B/16` backbone 사용
- cache 경로를 프로젝트 내부 `.cache/torch`로 변경
- pretrained 가중치 다운로드 실패 시 random init fallback 사용

### `models/language_encoder.py`

- 모델명을 `distilbert-base-uncased`로 수정
- cache 경로를 프로젝트 내부 `.cache/huggingface`로 변경
- 먼저 로컬 pretrained transformer를 로드 시도
- 실패 시 `EmbeddingBag` 기반 fallback text encoder 사용

### `models/fusion.py`

- vision feature와 language feature를 단순 concat
- 현재 프로토타입 단계에서는 이 정도로 유지

## 6. DistilBERT pretrained 사용 여부

처음에는 DistilBERT가 실제로 로드되지 않았고, fallback text encoder가 동작하고 있었다.

원인:

- [`.cache/huggingface`](/home/user/background/project/VLA/.cache/huggingface) 안에 pretrained 모델이 없었음
- `local_files_only=True`로 로컬 캐시만 찾도록 되어 있었음

이후 조치:

- `distilbert-base-uncased` tokenizer + model을 프로젝트 로컬 캐시에 직접 다운로드함

현재 확인된 상태:

- [`.cache/huggingface`](/home/user/background/project/VLA/.cache/huggingface)에 DistilBERT 캐시 파일 존재
- `LanguageEncoder()` 생성 시:
  - `use_transformer = True`
  - tokenizer 존재
  - model 존재
- `LanguageEncoder.forward()`에서 실제 출력 확인:
  - 출력 shape: `(2, 768)`
  - 서로 다른 문장에 대해 서로 다른 embedding 생성
- `Policy.forward()`까지 정상 연결 확인

즉, 현재는 fallback이 아니라 pretrained DistilBERT 경로가 실제로 사용되고 있다.

## 7. pretrained 가중치 관련 문제 정리

### 1) 캐시 경로 문제

초기에는 pretrained weight를 아래 경로에 저장하려고 시도했다.

- `~/.cache/torch/hub`

이 경로는 현재 환경에서 쓰기 권한 문제가 있었다.

해결:

- `.cache/torch`
- `.cache/huggingface`

로컬 프로젝트 내부 캐시를 사용하도록 변경했다.

### 2) 네트워크 / DNS 문제

다음과 같은 에러를 확인했다.

- `Could not resolve host: huggingface.co`
- `Temporary failure in name resolution`

이건 모델 코드 문제라기보다, 외부 네트워크 또는 DNS가 일시적으로 불안정한 상태에 가깝다.

현재 대응:

- vision encoder는 다운로드 실패 시 random init fallback
- language encoder는 pretrained 없으면 fallback text encoder 사용

즉 인터넷이 불안정해도 학습은 계속 가능하게 만들어 두었다.

## 8. 학습 파이프라인

파일:

- [train/train.py](/home/user/background/project/VLA/train/train.py)
- [train/loss.py](/home/user/background/project/VLA/train/loss.py)

현재 구현된 내용:

- `VLADataset` 로드
- transform 적용
- `collate_fn` 정의
- train/val dataloader 생성
- training loop
- validation loop
- checkpoint 저장
- optional `wandb` logging

loss는 현재 `MSELoss`를 사용한다.

## 9. episode 기준 split

초기에는 frame 기준 split을 사용했는데, 이 경우 같은 episode의 거의 동일한 장면이 train/val에 동시에 들어갈 수 있다.

그래서 현재는 episode 기준 split으로 변경했다.

동작 방식:

- 먼저 unique `episode_index`를 섞음
- episode 단위로 train/val 분리
- 각 episode에 속한 frame들을 해당 split에 할당

현재 split 결과:

- train: `243 episodes`
- val: `27 episodes`
- train frames: `127163`
- val frames: `14303`
- train/val episode overlap: `0`

이 방식이 validation을 더 정직하게 만들어 준다.

## 10. 학습 실행 및 상태 확인

기본 실행 명령:

```bash
conda run -n vla python -m train.train
```

짧은 smoke test에서 확인된 내용:

- dataset length 확인 완료
- 첫 샘플 로드 완료
- dataloader batch 생성 완료
- model forward 완료
- loss 계산 완료
- 1-step 수준 학습 smoke run 완료

관측된 예시 출력:

```text
Epoch [1/1] train_loss=2379.991455 val_loss=2232.539307
```

## 11. epoch 수와 학습 시간 관련 메모

현재 설정은 다음과 같다.

- batch size: `8`
- train frames: `127163`
- val frames: `14303`

즉 한 epoch는 대략:

- train batches: `15896`
- val batches: `1788`

이 구조는 frame 수가 적지 않고, 샘플마다 `ffmpeg`로 frame을 뽑기 때문에 꽤 느릴 수 있다.

짧은 time probe 결과:

- `train 2 step + val 1 step` 실행 시간: 약 `5.27초`

정확한 전체 epoch 시간은 하드웨어와 캐시 상태에 따라 달라지지만, 현재 구조에서는 1 epoch가 꽤 오래 걸릴 가능성이 높다.

따라서 `epoch=5`는 "가벼운 실험"은 아니며, 실제로 충분한지는 `val_loss` 추이를 보고 판단해야 한다.

## 12. checkpoint 정리 규칙

checkpoint는 실험별로 분리해서 관리하도록 정리했다.

현재 구조:

- [checkpoints/exp_fallback_text](/home/user/background/project/VLA/checkpoints/exp_fallback_text)
  - fallback text encoder 상태에서 돌렸던 기존 5 epoch 결과
- [checkpoints/exp_smoke](/home/user/background/project/VLA/checkpoints/exp_smoke)
  - smoke test 결과
- [checkpoints/exp_time_probe](/home/user/background/project/VLA/checkpoints/exp_time_probe)
  - 짧은 시간 측정용 결과
- [checkpoints/exp_pretrained_distilbert](/home/user/background/project/VLA/checkpoints/exp_pretrained_distilbert)
  - pretrained DistilBERT 기준 실험 결과 저장 폴더

현재 [train/train.py](/home/user/background/project/VLA/train/train.py)의 기본 `save_dir`는 다음으로 설정되어 있다.

- `checkpoints/exp_pretrained_distilbert`

즉, 지금 다시 학습을 돌리면 pretrained DistilBERT 실험 결과가 이 폴더에 저장된다.

## 13. wandb 사용 준비 상태

현재 [train/train.py](/home/user/background/project/VLA/train/train.py)에는 `wandb`가 optional 하게 붙어 있다.

지원되는 옵션:

- `use_wandb`
- `wandb_project`
- `wandb_run_name`
- `wandb_mode`

epoch마다 기록되는 값:

- `train_loss`
- `val_loss`
- `epoch_time_sec`

현재 확인한 상태:

- `wandb` 패키지는 아직 `vla` 환경에 설치되지 않음

사용하려면:

```bash
conda activate vla
pip install wandb
wandb login
```

예시 실행:

```bash
conda run -n vla python -c "from train.train import train; train(use_wandb=True, wandb_project='vla', wandb_run_name='exp_pretrained_distilbert')"
```

## 14. 현재 한계

현재 프로토타입은 end-to-end로 동작하지만, 아직 아래 한계가 있다.

- frame extraction이 느림
- vision encoder pretrained는 네트워크 상태에 따라 random init fallback 가능
- DistilBERT는 현재 pretrained로 동작하지만, 네트워크/캐시 상태에 의존할 수 있음
- state input은 아직 policy에 직접 넣지 않음
- test split 없음
- config 기반 실험 관리가 아직 약함

## 15. 다음에 하면 좋은 일

우선순위 기준으로 보면 다음이 유용하다.

1. `wandb` 설치 후 실제 학습 로그 기록
2. `test split` 추가
3. frame 캐싱 또는 사전 이미지 추출로 학습 속도 개선
4. config/YAML 기반으로 하이퍼파라미터 관리
5. 필요하면 `observation.state`도 policy 입력에 추가
6. 배포용 inference script 별도 구성
