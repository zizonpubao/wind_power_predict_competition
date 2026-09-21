# BARAM 2026 — 풍력발전량 예측 AI 경진대회

기상예보(NWP)와 터빈 SCADA 데이터로 3개 풍력단지(KPX 그룹)의 **시간 단위 발전량(kWh)** 을 예측하는
DACON 대회 참가 기록입니다. 리더보드 점수 **0.6091 → 0.63330** (Public, 최종 선택 기준)까지의
모든 시도 — 성공과 실패 전부 — 를 이 리포지토리에서 볼 수 있습니다.

> **이 프로젝트의 전 과정을 정리한 자습용 슬라이드(58장)**:
> [reports/review/baram2026_review.pptx](reports/review/baram2026_review.pptx)
> — 문제 정의부터 피처 엔지니어링, 분위수 GBM + 결정최적화, CV vs 리더보드 괴리까지.
> 팩트체크 리포트([ppt_review.md](reports/review/ppt_review.md)) 포함.

## 대회 요약

| | |
|---|---|
| 과제 | 3개 풍력단지(21.6 / 21.6 / 21.0 MW)의 2025년 시간당 발전량 예측 (8,760시간) |
| 입력 | LDAPS(1.5km)·GFS(0.25°) 기상예보 격자, 터빈 SCADA(학습 기간만) |
| 평가 | **0.5 × (1-NMAE) + 0.5 × FICR** |
| 제약 | 로컬 실행 모델만 허용, 1일 제출 5회 |

**FICR(정산금획득률)** 이 전략 전체를 지배했습니다: 시간대별 오차율이 6% 이하면 4원/kWh,
6~8%면 3원/kWh, 8% 초과면 **0원**인 계단식 지표라서, 평균 오차(NMAE)를 줄이는 것과
문턱 안쪽으로 시간대를 밀어 넣는 것은 다른 문제입니다. 이 구조가 아래의 "분위수 예측 +
결정최적화" 설계로 이어졌습니다.

## 파이프라인

```
LDAPS/GFS 격자 (+ 외부 NWP: ECMWF·GEM 백필)
   │  격자 집계(IDW/최근접/분산) · 풍향 벡터분해 · 파워커브 변환(v³)
   │  rolling 통계(발표묶음 내) · SCADA 나셀풍속 quantile mapping · 교차모델 불일치 피처
   ▼
피처 테이블 (그룹별 76~89개, gain 기반 선택)
   │
   ├─ LightGBM 9-분위수 모델 ──► 기대효용 결정최적화 (FICR 계단 단가에 맞춰 제출값 선택)
   ├─ LSTM
   └─ Transformer
   ▼
가중 블렌드 (GBM 0.35 / LSTM 0.325 / Transformer 0.325)
   ▼
그룹별 배수 재보정 (과소예측 편향 교정, half 강도) → clip → 제출
```

핵심 설계 결정 4가지:

1. **점예측이 아니라 분포 예측** — LightGBM으로 9개 분위수를 뽑고, FICR의 계단 단가표에 대해
   기대 정산금을 최대화하는 값을 후처리로 선택 (`src/features/decision_optimize.py`). 단독으로
   +0.0035.
2. **발표묶음 단위 시계열 CV** — 같은 발표 시각을 공유하는 24시간을 한 블록으로 묶어 분할
   (`src/validation/`). 리키지는 데이터가 아니라 우리가 만드는 분할에서 생깁니다.
3. **외부 NWP를 리키지-세이프로 백필** — Open-Meteo Previous Runs API의 `previous_day2`
   오프셋으로 대회 정보 컷오프(D-1 14:00 KST)를 수학적으로 만족시키며 ECMWF/ICON/GEM/JMA를
   추가 (`src/data/fetch_*.py`, 검증 테스트 포함). ECMWF는 실전 +0.0028로 후반 최대 도약.
4. **하루 5제출 = 5번의 실전 실험** — CV가 ±0.002를 못 가르는 상황에서, 변수 1개씩만 바꾼
   프로브를 매일 5개 제출해 리더보드 자체를 실험 장치로 사용. 운영 프로토콜과 전체 결과
   장부는 [reports/experiment_queue.md](reports/experiment_queue.md).

## 점수 여정 (Public 리더보드)

| 날짜 | 시도 | 점수 | Δ |
|---|---|---|---|
| 07-22 | LightGBM 단독 (튜닝+피처선택) | 0.6091 | 기준점 |
| 07-23 | 9-분위수 GBM + 결정최적화 | 0.6126 | +0.0035 |
| 07-23 | GBM/LSTM/Transformer 블렌드 | 0.6227 | +0.0101 |
| 07-24 | 배수 재보정 (half 강도) | 0.6280 | +0.0053 |
| 08-07 | ECMWF 피처 (g1 한정) | 0.6302 | +0.0022 |
| 08-11 | ECMWF 전그룹 | 0.6330 | +0.0028 |
| 08-12 | GBM 블렌드 가중 0.30→0.35 | 0.63325 | +0.0003 |
| 08-13 | 재보정 강도 0.5→0.45 | **0.63330** | +0.0001 |

## 실패도 기록입니다

전체 장부는 [experiment_queue.md](reports/experiment_queue.md)에 있고, 골라 보면:

- **ECMWF 사건 (최대 교훈)** — CV는 g2에서 -0.0229 "해로움"을 경고했지만 리더보드에선 신기록.
  원인: ECMWF 커버리지가 학습 기간의 ~25%뿐이라 CV 폴드 대부분이 이 피처를 본 적이 없음.
  이후 "부분 커버리지 피처는 CV로 기각하지 않고 리더보드로 판정"을 룰로 채택.
- **top3avg 사건** — 근접 동률 제출 3개를 평균했더니 1-NMAE는 멀쩡한데 FICR이 붕괴(-0.0022).
  평균(스무딩)은 시간대별 오차를 6%/8% 문턱 밖으로 밀어냅니다. 계단식 지표에서 앙상블
  상식이 통하지 않는 사례.
- **NN에 ECMWF 주입 → 3그룹 전부 붕괴(-0.031)** — 트리는 결측/희소 피처를 분기에서 무시할 수
  있지만 소형 NN은 그러지 못함. 같은 피처, 다른 모델 계열, 정반대 결과.
- **ICON(-0.0067)·JMA(중립) 기각** — 같은 "새 NWP 추가"라도 다 통하는 게 아님. 특히 JMA는
  전 기간 커버라 CV를 신뢰할 수 있었기에, 위 룰을 적용하지 않고 정직하게 기각.
- 그 외: 19-분위수(해상도↑가 실전 미전이), 월별 재보정(계절 분해 역효과), 시드 배깅(CV
  개선이 실전 미전이), 재보정 강도 0.6(과보정) 등 — 각각 장부에 수치와 함께 기록.

## 리포지토리 구조

```
├── src/
│   ├── data/          # 원본 로더 + 외부 NWP 백필 (fetch_ecmwf/gem/jma/icon.py)
│   ├── features/      # 피처 엔지니어링, quantile mapping, 결정최적화
│   ├── validation/    # 발표묶음 단위 시계열 CV 분할기
│   ├── models/        # LightGBM 분위수 / LSTM / Transformer 래퍼
│   ├── training/      # 학습 스크립트 (실험별 experiments/<run_id>/ 기록)
│   ├── evaluation/    # 1-NMAE / FICR 공식 산식 구현
│   └── inference/     # test 예측 → 제출 CSV
├── tests/             # 리키지 방지 로직·피처 함수 단위 테스트
├── reports/
│   ├── experiment_queue.md   # ★ 모든 실험의 대기열·프로토콜·결과 장부
│   ├── review/               # 프로젝트 리뷰 슬라이드(58장) + 팩트체크 리포트
│   ├── pipeline_explained.md # 파이프라인 상세 해설
│   ├── eda/                  # EDA 리포트 (SCADA 함정, 라벨 gap 등)
│   └── domain_research/      # 평가 산식·파워커브·웨이크 효과 조사
└── CLAUDE.md          # 프로젝트 규칙·도메인 지식·에이전트 팀 구성
```

## 재현 방법

```bash
conda env create -f environment.yml
conda activate baram2026
python -m src.features.build_features        # 피처 테이블 생성
python -m src.training.train_gbm_quantile    # GBM 학습 (experiments/에 기록)
python -m src.inference.predict <run_id>     # 제출 CSV 생성
```

원본 대회 데이터는 리포지토리에 포함되지 않습니다 (로컬 경로는 [configs/paths.py](configs/paths.py)
에서 설정). 외부 NWP 백필은 `python -m src.data.fetch_ecmwf` 등으로 재수행할 수 있습니다.

## 참고 문서

- [docs/data_description.md](docs/data_description.md) — 원본 데이터 명세서 사본
- [docs/turbine_kpx_mapping.md](docs/turbine_kpx_mapping.md) — 터빈 ↔ KPX 그룹 매핑
- [reports/pipeline_explained.md](reports/pipeline_explained.md) — 파이프라인 해설
- [reports/weekly_review.md](reports/weekly_review.md) — 주간 리뷰
