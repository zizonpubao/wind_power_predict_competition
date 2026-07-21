# 새 PC에서 이어서 작업하기

Claude Code 데스크톱 앱은 세션이 실행 중인 물리 머신을 바꾸는 기능이 없다 (환경은 세션
시작 시 Local/Remote/SSH/WSL 중 하나로 고정). 다른 PC에서 작업하려면 그 PC에서 **새
세션**을 열어야 한다.

## 1. Claude Code 데스크톱 앱에서

새 세션 시작 → 환경 선택에서 **Local** 선택 → 아래 순서로 이 프로젝트를 준비한다.

## 2. 코드 가져오기 (git에 있는 것)

```
git clone https://github.com/zizonpubao/wind_power_predict_competition.git
```

`CLAUDE.md`, `src/`, `docs/`, `reports/`, `.claude/agents/` 등 코드·문서는 전부 여기 있다.
새 세션은 `CLAUDE.md`를 읽는 것만으로 지금까지의 맥락(대회 규정, 데이터 함정, 진행 상황,
현재 최고 스코어 등)을 파악할 수 있다.

## 3. git에 없는 것 (수동으로 옮겨야 함)

- **원본 대회 데이터** (`open (1)` 폴더 내용물, 필수): 재배포 금지 우려로 git에 올리지
  않았다. 원래 PC에서 USB/클라우드 드라이브 등으로 직접 복사해 와야 한다.
  - `configs/paths.py`는 여러 머신을 오가며 작업하는 상황을 감안해 **경로를 자동으로
    찾는다**: `_KNOWN_RAW_DATA_ROOTS` 목록에 있는 후보 경로들 중 `sample_submission.csv`가
    실제로 있는 첫 번째 경로를 씀 (현재 등록된 후보: 노트북의 `C:\Users\aica_\Desktop\open (1)`,
    heelo PC의 `C:\Users\heelo\Desktop\claude_code\wind_power_predicet\datasets`
    /`...\datasets\open (1)`).
  - **완전히 새로운 머신이면**: `configs/paths.py`의 `_KNOWN_RAW_DATA_ROOTS`에 그 머신의
    실제 데이터 경로를 한 줄 추가하면 된다. 또는 코드 수정 없이 `BARAM_RAW_DATA_ROOT`
    환경변수로 덮어쓸 수도 있음 (`_KNOWN_RAW_DATA_ROOTS`보다 항상 우선).
- **생성된 산출물** (`data/interim/`, `data/processed/`, `experiments/`, `submissions/`):
  전부 `.gitignore` 대상이라 git에 없다. 아래 4번 순서대로 다시 돌리면 몇 분 안에
  복구된다 (원본 데이터만 있으면 어디서든 재생성 가능).

## 4. 환경 설치 + 파이프라인 재실행

```
conda env create -f environment.yml
conda activate baram2026

python -m src.features.build_features       # data/processed/ 재생성
python -m src.training.train_baseline        # 베이스라인 재학습 -> experiments/
python -m src.training.tune_hyperparams       # 하이퍼파라미터 재튜닝 (선택, 시간 걸림)
```

(모듈 실행 경로/스크립트명은 시점에 따라 바뀔 수 있으니, 새 세션에서 `src/training/`
디렉토리 실제 파일을 먼저 확인할 것.)

## 5. 확인

```
pytest tests/ -q
```

이게 통과하면 새 PC 환경이 노트북과 동일하게 동작한다는 뜻이다.

## GPU 관련 참고

이 프로젝트는 현재 LightGBM(CPU) 기반이라 GPU 이득이 크지 않다 (2.6만 행 데이터,
튜닝 40trial×3그룹도 CPU로 수 분 내 완료). GPU는 나중에 딥러닝 모델을 추가로 시도할 때나
의미가 있다 — 그 전까지는 굳이 GPU 환경에 맞춰 별도 설정할 필요는 없다.
