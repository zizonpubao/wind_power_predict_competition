# BARAM 2026 — 풍력발전량 예측 AI 경진대회

## 1. 대회 개요

- **주제**: 기상예보(LDAPS/GFS) + 터빈 SCADA 실측 데이터를 활용해 3개 KPX 그룹(풍력단지)의
  **시간 단위 발전량(kWh)**을 예측.
- **평가**: 1차 온라인 리더보드 = 평균 예측오차율(**1-NMAE**) + **정산금획득률(FICR)**.
  1차 상위팀은 2차 발표평가(과제이해도/기술우수성/문제해결력/적용가능성) 진행.
- **예측 대상 기간**: 2025-01-01 01:00 ~ 2026-01-01 00:00 (`sample_submission.csv`와 동일, 8,760행).
- **팀**: `kpx_group_1`(21.6MW), `kpx_group_2`(21.6MW), `kpx_group_3`(21.0MW).

## 2. 원본 데이터 — 읽기 전용 원칙

원본 데이터는 `C:\Users\aica_\Desktop\open (1)\` 에 있으며 **절대 수정/이동/덮어쓰기 금지**.
모든 코드는 이 경로를 절대경로로 읽기만 하고, 가공 결과는 프로젝트 내부
(`data/interim/`, `data/processed/`)에 별도 파일로 저장한다. 원본 폴더 안에 파일을 새로 쓰지 않는다.

```
C:\Users\aica_\Desktop\open (1)\
├── train\ldaps_train.csv, gfs_train.csv, train_labels.csv,
│        scada_vestas_train.csv, scada_unison_train.csv
├── test\ldaps_test.csv, gfs_test.csv
├── sample_submission.csv
├── info.xlsx
└── data_description.md
```

경로는 코드에서 하드코딩하지 말고 반드시 `configs/paths.py`를 통해 참조한다.

## 3. 데이터 핵심 지식 (실수하기 쉬운 지점)

- **인코딩/시간대**: 모든 CSV는 `utf-8-sig`, 모든 시각은 KST `YYYY-MM-DD HH:MM:SS`.
- **예보 발표/사용가능 시각 분리**: `forecast_kst_dtm`(예측 대상 시각) ≠ `data_available_kst_dtm`
  (그 예보를 실제로 쓸 수 있게 된 시각, 전날 09시 발표분이 13시부터 가용).
  같은 `data_available_kst_dtm`을 공유하는 24개 시각이 한 "발표 묶음"이다.
  **CV(교차검증) 분할 시 이 발표 묶음 단위로 시계열을 끊어야 하며**, 랜덤 셔플로 미래 정보가
  섞여 들어가는 leakage를 절대 만들지 않는다. (test 데이터 자체는 이미 이 규칙을 지켜서
  제공되므로, leakage는 주로 우리가 직접 만드는 학습/검증 분할 과정에서 발생한다.)
- **격자 구조**: LDAPS = 16개 격자(`grid_id` 1~16, ~1.5km 해상도), GFS = 9개 격자
  (`grid_id` 1~9, 0.25도 해상도). `forecast_kst_dtm` 1개당 grid_id 개수만큼 행이 반복되는
  long format이므로, 모델 입력으로 쓰려면 wide로 pivot하거나 격자 통계(평균/최근접 격자 등)로
  집계해야 한다.
- **터빈 ↔ KPX 그룹 매핑** (`info.xlsx`, 상세는 [docs/turbine_kpx_mapping.md](docs/turbine_kpx_mapping.md)):
  - VESTAS 1~6호기 → `kpx_group_1` (21.6MW)
  - VESTAS 7~12호기 → `kpx_group_2` (21.6MW)
  - UNISON 1~5호기 → `kpx_group_3` (21.0MW)
  - `info.xlsx`는 인코딩 문제로 터미널에서 바로 열지 말고 `docs/info_raw.csv`(재저장본, 원본 미수정)를 사용.
- **SCADA는 test 기간에 없다**: `scada_*_train.csv`는 학습 기간에만 존재하는 보조 데이터이며,
  평가 기간 예측에는 절대 직접 입력으로 쓸 수 없다 (data_description.md 12번 항목,
  "비공개 운영 데이터... 예측에 사용 불가"와 궤를 같이함). SCADA는 ① 라벨 검증
  (그룹 합산치 vs `train_labels.csv` 비교), ② 결측치 보간/이상치 탐지, ③ 파워커브 등
  물리 기반 피처의 파라미터 추정 등 **학습 파이프라인 설계 보조용**으로만 활용한다.
- **SCADA 단위**: `*_power_kw10m`은 10분 단위 순시평균출력(kW). 라벨은 1시간 kWh 집계이므로
  6개 10분 값 평균 → 1시간 kWh로 환산 시 시간 경계(0분/10분.../50분) 정렬을 검증해야 한다.
- **결측 라벨**: `kpx_group_3`은 2022년 값이 전부 결측 (2023년부터 제공). 그룹별로 학습 가능
  기간이 다르므로, 그룹 3개를 하나의 멀티아웃풋 모델로 묶을지 개별 모델로 분리할지는
  이 결측 구조를 반영해서 결정해야 한다.
- **설비용량 환산**: 평가에서 발전량-설비용량을 함께 쓸 때 1시간 기준 kWh로 환산
  (21.6MW → 21,600kWh, 21.0MW → 21,000kWh). 발전량이 설비용량을 초과하는 이상치가 있는지
  EDA에서 반드시 확인.

## 4. 아직 확인이 필요한 사항 (사용자 확인 요청)

- **1-NMAE / FICR 정확한 산식**: `data_description.md`에는 지표 이름만 있고 계산식이 없다.
  특히 FICR(정산금획득률)은 국내 재생에너지 발전량예측제도(오차율 기반 인센티브/패널티)와
  연동된 지표로 추정되는데, 이건 대회 공식 규정 페이지(DACON 등)를 봐야 정확히 알 수 있다.
  **대회 공식 페이지 URL을 알려주면** domain-researcher 에이전트가 정확한 산식과 대회 규정
  (제출 횟수 제한, 팀 규정, 마감일, 발표평가 일정 등)을 조사해서 `reports/domain_research/`에
  정리하도록 하겠다.
- **개발 환경 확인 결과**: GPU 미검출(`nvidia-smi` 없음) → 기본 전략은 CPU 친화적인 gradient
  boosting(LightGBM/XGBoost/CatBoost) 계열을 우선하고, 딥러닝(LSTM/TFT 등)은 필요성이
  확인되면 추후 검토. 현재 conda(25.11.1)는 있으나 lightgbm/xgboost/torch/optuna 등은
  미설치 상태 — 설치 진행해도 되는지 확인 필요.

## 5. 디렉토리 구조

```
claude_ai/
├── CLAUDE.md                 # 이 파일 — 프로젝트 규칙/도메인 지식
├── README.md
├── environment.yml           # conda 환경 정의
├── configs/
│   └── paths.py              # 원본 데이터 절대경로 등 중앙 경로 설정
├── docs/                     # 대회 자료 사본/파생 문서 (원본 미수정)
│   ├── data_description.md
│   ├── info_raw.csv
│   └── turbine_kpx_mapping.md
├── data/
│   ├── interim/               # 정제/병합된 중간 산출물 (parquet), git 추적 안 함
│   └── processed/             # 모델 입력용 feature table, git 추적 안 함
├── notebooks/eda/             # 탐색적 분석 노트북
├── src/
│   ├── data/                  # 원본 CSV 로더, 파서
│   ├── features/              # 피처 엔지니어링 (풍속벡터, 파워커브, lag/rolling 등)
│   ├── validation/            # 발표묶음 단위 time-series CV 분할기
│   ├── models/                # 모델 wrapper (LightGBM/XGBoost/CatBoost 등)
│   ├── evaluation/             # 1-NMAE, FICR 메트릭 구현
│   ├── training/               # 학습 파이프라인/실험 실행 스크립트
│   └── inference/              # test 예측 + submission 생성
├── experiments/<run_id>/      # 실험별 config, metrics, model artifact, log
├── submissions/                # 제출용 CSV (git 추적 안 함)
├── reports/
│   ├── eda/                    # EDA 분석 리포트
│   └── domain_research/        # 도메인 조사 리포트 (제도, 파워커브 이론 등)
└── tests/                      # 단위 테스트 (leakage 방지 로직, 피처 함수 등)
```

## 6. 에이전트 팀 & 워크플로우

메인 세션(사용자와 직접 대화)이 기획/설계를 담당하고, 아래 서브에이전트(`.claude/agents/`)에게
백그라운드로 실행을 위임한다. 각 에이전트 정의는 `.claude/agents/*.md` 참고.

| 에이전트 | 역할 |
|---|---|
| (메인 세션) | 사용자와 기능 기획, 문제 탐색, 구조 설계, 작업 분배 |
| `domain-researcher` | 대회 규정/평가 산식 조사, 풍력발전 도메인 지식(파워커브, 웨이크 효과, 재생에너지 정산제도) 리서치 |
| `eda-analyst` | 데이터 탐색, 통계/시각화, 라벨-SCADA 정합성 검증, 이상치/결측 리포트 |
| `feature-engineer` | `src/features/` 전담 — 파워커브/풍향벡터/격자집계 등 도메인 기반 피처 구현 |
| `code-writer` | 설계안을 바탕으로 그 외 `src/` 코드(로더, 모델, 학습/추론 파이프라인) 구현 |
| `code-reviewer` | 작성된 코드의 버그, 구조적 문제, leakage 위험, 기존 코드와 충돌 여부 검수 |
| `trainer` | 실제 데이터로 학습 실행, 실험 결과를 `experiments/`에 기록 |
| `evaluator` | `src/evaluation/` 전담 — 1-NMAE/FICR 정확 구현, 실험 간 성능 비교/리더보드 관리 |
| `ensembler` | 여러 학습 완료 모델의 예측을 블렌딩/스태킹해 최종 제출 산출 (대회 후반 투입) |

일반적인 작업 순서: 사용자 ↔ 메인 세션에서 설계 합의 → `eda-analyst`/`domain-researcher`가
탐색·조사 → `feature-engineer`/`code-writer`가 구현 → `code-reviewer`가 검수(문제 있으면
반려) → `trainer`가 학습 실행 → `evaluator`가 정확한 지표로 검증/랭킹 → 여러 유효 모델이
쌓이면 `ensembler`가 블렌딩. 각 에이전트 산출물은 설계 단계로 계속 피드백된다.

## 7. 코딩 컨벤션

- 모든 시간 처리는 KST 그대로 유지 (UTC 변환 불필요, 변환 시 실수 유발).
- 원본 CSV 재저장 금지. 원본에서 파생된 파일은 전부 `data/interim` 또는 `data/processed`에 parquet로.
- 랜덤 시드 고정, 실험 설정은 `experiments/<run_id>/config.yaml`로 남겨 재현 가능하게 유지.
- 검증 분할은 반드시 `src/validation`의 발표묶음 기준 분할기를 통해서만 수행 (직접 `train_test_split` 금지).
