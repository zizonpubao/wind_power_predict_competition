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
- **SCADA 단위 (정정, code-writer 실측 검증 완료)**: `*_power_kw10m`을 1시간 kWh로 환산할 때
  **평균이 아니라 6개 10분 값을 그대로 합산(sum)**해야 `train_labels.csv`와 거의 완벽히
  일치한다 (r=0.999, 그룹 3개 전부). 평균을 쓰면 6배 과소 추정된다 — 컬럼명이
  `kw10m`(순시 kW)처럼 보이지만 실제로는 "10분 구간 에너지량"에 가까운 값으로 보인다.
  `src/data/loaders.py`의 `aggregate_scada_to_hourly`가 이미 sum으로 구현되어 있으니
  이 함수를 통해서만 시간 집계할 것, 직접 재구현하지 말 것.
  또한 `kst_dtm`은 **집계 구간의 종료 시각**이 확인됨 (HH:00~HH:50 6개 값 → 라벨의
  HH+1:00 시각과 상관관계 0.999, 같은 시간대(HH:00) 라벨과는 0.954로 더 낮음) —
  `data_description.md`의 "집계 구간의 종료 시각" 문구와 일치.
  단, 이상치/센티널 값(예: ±4000kW를 넘는 값 등, 실측 약 868/190만 행)이 섞여 있어 그대로
  합산하면 왜곡될 수 있음 — 이상치 마스킹은 아직 loader에 구현되지 않았으므로
  `eda-analyst`/`feature-engineer`가 후속으로 처리해야 한다.
- **결측 라벨**: `kpx_group_3`은 2022년 값이 전부 결측 (2023년부터 제공). 그룹별로 학습 가능
  기간이 다르므로, 그룹 3개를 하나의 멀티아웃풋 모델로 묶을지 개별 모델로 분리할지는
  이 결측 구조를 반영해서 결정해야 한다.
- **설비용량 환산**: 평가에서 발전량-설비용량을 함께 쓸 때 1시간 기준 kWh로 환산
  (21.6MW → 21,600kWh, 21.0MW → 21,000kWh). 발전량이 설비용량을 초과하는 이상치가 있는지
  EDA에서 반드시 확인.

## 4. EDA 확정 사실 (`reports/eda/eda_summary.md`, 2026-07-20 전수 조사 완료)

- vestas SCADA에 물리적으로 불가능한 극단값이 0.48%(759/157,819행) 존재 (`|값|` 최대
  ~5×10⁷) — SCADA를 어디에 쓰든 `|값| > ~700` 임계치로 마스킹 필수. unison은 이런 문제 없음.
- 보정 후에도 라벨-SCADA 합산치 gap은 p5~p95 ±14~16%p, p1~p99 ±23~29%p로 크고 계절성(겨울↑
  여름↓)까지 있음 — **SCADA는 부드러운 참고 신호일 뿐 QA pass/fail 기준으로 쓰지 말 것.**
- 결측은 라벨(타깃)에만 존재하고 규모도 작음 (그룹1/2 103~104시간, 그룹3 2023~2024 구간 내
  6시간) — 결측 행은 단순 drop으로 충분, 임퓨테이션 불필요. 기상(LDAPS/GFS)·SCADA 파일은
  시간축 결측 0건.
- group3은 SCADA 상관계수도 그룹1/2보다 낮음(0.949 vs 0.965~0.969) — 짧은 학습기간과 함께
  개별 모델/손실 가중치 조정을 고려할 근거가 하나 더 늘어남.
- group3 설비용량 초과 38건(최대 +0.62%)은 무시 가능하나, 추론 후처리에
  `clip(0, capacity×1.01)` 안전장치 권장.
- **LDAPS가 주 피처, GFS는 보조** — 풍속-발전량 상관계수가 LDAPS 0.73~0.74, GFS 0.54~0.55로
  뚜렷하게 차이남 (해상도 1.5km vs 0.25도 차이가 그대로 반영).
- lead-hour(예보 시차)에 따른 뚜렷한 성능 저하 신호는 발견되지 않음 (대리 지표 기반 한계 있음).
- 원본 데이터 자체에는 leakage 0건 (4개 기상 파일 전수 확인) — leakage 리스크는 전적으로
  우리가 만드는 `src/validation` 분할기 구현에 달려있음.

## 5. 확정된 평가 산식 & 대회 규정 (DACON 공식 페이지 조사 완료, 2026-07-20)

출처: https://dacon.io/competitions/official/236727 (overview/evaluation, rules, schedule).
상세 근거/인용은 `reports/domain_research/`(nmae_formula.md, ficr_formula.md,
competition_rules.md) 참고. 이 파일들은 로컬에만 있고 원본 데이터 사본
(`docs/data_description.md`, `docs/info_raw.csv`)과 달리 우리가 직접 조사·작성한 문서라
git에도 포함되어 있다.

**평가 산식**:
- 그룹별 NMAE = 평균( |예측−실제| / 그룹설비용량(kWh) ), 1-NMAE = 1 − (3개 그룹 NMAE 평균)
- 그룹별 FICR = 획득 정산금 / 이론상 최대 정산금 (시간대별 NMAE 기준 정산 단가 적용), FICR = 3개 그룹 평균
- **최종 점수 = 0.5 × (1-NMAE) + 0.5 × FICR**
- 두 지표 모두 **실제 발전량이 설비용량의 10% 이상인 시간대에만** 적용 (그 미만은 평가 제외)
- **FICR 정산 단가표 확정** (대회 공식 페이지 확인, `ficr_formula.md` 참고):
  시간대별 개별 오차율 `nmae_h = |예측-실제|/그룹설비용량` 기준으로

  | nMAE 구간 | 정산 단가 |
  |---|---|
  | 6% 이하 | 4원/kWh |
  | 6% 초과 ~ 8% 이하 | 3원/kWh |
  | 8% 초과 | 0원 (정산금 없음) |

  그룹별 FICR = Σ(단가_h × 실제발전량_h) / Σ(4원 × 실제발전량_h) (오차 0일 때 매 시간 최고단가
  4원 적용을 "이론상 최대"로 가정). FICR = 3개 그룹 평균.
  **FICR은 계단식(불연속) 지표** — 오차율이 6%/8% 문턱을 넘느냐가 3%→5% 개선보다 훨씬 큰
  영향을 줄 수 있음. 단순 MAE 최소화만으로는 FICR을 직접 최적화하지 못할 수 있으므로
  모델링/후처리 전략에 반영할 것. `evaluator` 에이전트가 이 정확한 산식을
  `src/evaluation/`에 구현한다.
- **Public/Private Score**: Public = 평가 데이터 중 사전 샘플링된 40% (대회 기간 중 리더보드
  실시간 반영), Private = 나머지 60% (최종 순위 산정 기준).

**절대 규정 — 로컬 모델만 허용 (매우 중요)**:
- **외부 원격 추론 API 사용 금지** (OpenAI API, Gemini API, Hugging Face Inference API 등).
  **로컬에서 직접 실행되는 모델만 허용.** 즉 실제 예측 파이프라인(학습/추론)은 반드시 로컬에서
  돌아가는 모델(LightGBM/XGBoost/CatBoost, 로컬 실행 딥러닝 모델 등)만 써야 하며, 어떤 형태로든
  Claude/GPT/Gemini 등 원격 LLM API를 예측값 산출 로직에 끼워 넣으면 안 된다. (Claude Code를
  개발 보조 도구로 쓰는 것 자체는 코드 작성 과정이지 제출 파이프라인의 일부가 아니므로 별개.)
- 만약 딥러닝 모델을 쓴다면, **2026-07-06(대회 시작일) 이전에 가중치가 공개된 오픈소스 +
  상업적 이용 가능 라이선스** 모델만 사용 가능.

**일정 (오늘 2026-07-20 기준)**:
- 대회 기간(제출 가능): ~2026-08-14(금) 10:00 (약 25일 남음)
- 팀 병합 마감: 2026-08-07(금) 23:59
- 2차 평가 대상자(리더보드 상위 30팀) 산출물 제출: 2026-08-14 12:00 ~ 08-17 10:00, 검증
  08-17~08-21, 검증 통과 상위 20팀 오프라인 발표평가 2026-08-28(금)
- 최종 순위: 리더보드 성과 50% + 발표평가 50%

**제출/팀 규정**: 1일 최대 제출 5회, 팀 최대 3명, 중복 참가 금지, 부정행위 적발 시 해당 팀
평가 제외.

**여전히 미확인** (필요시 domain-researcher 추가 조사): FICR 정확한 오차율 구간 경계·단가
(DACON 미공개), 상금 배분 상세(`/overview/prize`), 2차 순위 산정 시 "리더보드 성과"가 1차
총점과 동일 재계산인지 여부.

## 6. 개발 환경

GPU 미검출(`nvidia-smi` 없음) → CPU 친화적인 gradient boosting(LightGBM/XGBoost/CatBoost)
계열을 기본 전략으로 하고, 딥러닝(LSTM/TFT 등)은 필요성이 확인되면 추후 검토 — 단, 위 4절의
"로컬 실행 + 대회 시작 전 공개 오픈소스 + 상업적 이용 가능 라이선스" 제약을 반드시 지킬 것.
conda 환경 `baram2026`은 `environment.yml`로 이미 구성/설치 완료
(pandas/numpy/scikit-learn/lightgbm/xgboost/catboost/optuna 등).

## 7. 디렉토리 구조

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

## 8. 에이전트 팀 & 워크플로우

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

## 9. 코딩 컨벤션

- 모든 시간 처리는 KST 그대로 유지 (UTC 변환 불필요, 변환 시 실수 유발).
- 원본 CSV 재저장 금지. 원본에서 파생된 파일은 전부 `data/interim` 또는 `data/processed`에 parquet로.
- 랜덤 시드 고정, 실험 설정은 `experiments/<run_id>/config.yaml`로 남겨 재현 가능하게 유지.
- 검증 분할은 반드시 `src/validation`의 발표묶음 기준 분할기를 통해서만 수행 (직접 `train_test_split` 금지).
