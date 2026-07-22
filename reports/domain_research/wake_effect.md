# 터빈 웨이크(후류) 효과 — 그룹1→그룹2 이상반응 가설 검증용 이론/피처 조사

**작성 배경**: `code-writer`의 최근 두 실험(OOF-isotonic bias calibration, 비대칭 손실함수)에서
`kpx_group_2`만 다른 두 그룹과 반대 방향으로 반응했고, 지리적으로 `kpx_group_1`(태백가덕산
1~6호기)이 `kpx_group_2`(7~12호기)의 바로 상류(풍상, 방위각 295° 방향에서 바람이 불 때)에
위치한다는 사실이 확인되어, "그룹1의 웨이크가 그룹2 발전량에 자유대기 예보만으로 설명 안 되는
편향을 만드는가"라는 가설을 조사했다. 결론부터: **웨이크는 물리적으로 실재하는 효과이고, 방향
정렬 기반 피처는 표준적인 실무 기법이지만, 이 프로젝트의 예보 입력(LDAPS/GFS)이 웨이크라는
서브그리드 현상을 직접 담고 있지 않다는 점이 피처 설계의 핵심 제약이다.**

## 1. 좌표 재계산 — 그룹1→그룹2 방위각/거리 독립 검증

`docs/info_raw.csv`의 개별 터빈 DMS 좌표를 `src/features/weather_features.py`의
`_parse_google_coord`/`_group_centroid`와 동일한 방식(단순 산술평균 centroid)으로 직접
수동 재계산해 사용자가 제시한 값을 재현/검증했다 (코드 실행 없이 수기 계산, 소수점 5자리
DMS→십진 변환 후 평면근사 bearing 공식 `atan2(Δlon·cos(lat), Δlat)` 사용):

| 관계 | 중심좌표 (lat, lon) | 방위각(bearing) | 거리 |
|---|---|---|---|
| kpx_group_1 centroid | (37.28713, 128.95202) | — | — |
| kpx_group_2 centroid | (37.28225, 128.96515) | — | — |
| kpx_group_3 centroid | (37.27520, 128.97144) | — | — |
| **group1 → group2** | | **115.0°** (ESE) | **1.28 km** |
| group2 → group3 (참고, 이번 조사에서 추가 확인) | | ≈145° (SE) | ≈0.96 km |
| group1 → group3 (참고) | | ≈128° (SE) | ≈2.17 km |

group1→group2는 사용자가 제시한 115°/1.28km와 정확히 일치 — 독립 재계산으로 확증됨.
group2→group3, group1→group3은 이번에 추가로 계산한 참고값으로, 그룹3에서 유사한 이상반응이
나타날 경우 확장 가설(그룹2가 그룹3을 웨이크)을 검토할 때 바로 쓸 수 있도록 남겨둔다 (이번
조사 범위인 그룹1→그룹2 가설과는 별개이며 검증되지 않은 참고용).

**방위각의 의미(중요, 혼동 주의)**: "bearing 115°"는 *그룹1에서 그룹2 쪽으로 향하는 지리적
방향*(플룸이 흘러가는 방향)이다. 기상학적 풍향(`weather_features.py`의 `dir_deg`, "바람이
불어오는 방향")과는 반대 개념이므로, 그룹1이 그룹2를 웨이크하려면 바람이 **295°
(115°+180°, WNW)에서 불어와야** 한다. §4의 피처 수식은 이 부호를 명시적으로 반영한다.

## 2. Jensen(=PARK, N.O. Jensen 1983) 웨이크 모델 — 실무 표준 해석적 모델

가장 널리 쓰이는 단순 해석적 웨이크 모델(WAsP/WindPRO의 PARK 모델과 동일 계열, FLORIS·PyWake
등 오픈소스 웨이크 모델링 툴체인의 기본 모델)은 원뿔형(top-hat) 속도결손을 가정한다.

**웨이크 반경** (하류로 갈수록 선형 확장):

```
r(x) = R + k·x
```

- `R`: 로터 반경 (V126 = 63m, U136 = 68m)
- `x`: 상류 터빈으로부터 하류로의 거리
- `k`: wake decay/entrainment constant — 지표 거칠기·대기 안정도·난류강도에 의존.
  실무에서 **onshore ≈ 0.075, offshore ≈ 0.04~0.05**를 기본값으로 쓰는 것이 통념(해상은 대기
  난류가 약해 웨이크가 더 오래 남기 때문에 k가 작음). 산악 능선처럼 복잡지형(complex
  terrain)은 일반적으로 평지보다 주변 난류강도가 높아 웨이크가 더 빨리 회복되는 경향이 있어
  실무적으로 k를 0.075보다 다소 높게(예 0.09~0.12) 잡는 경우도 있으나, 이는 **이 프로젝트
  사이트에 대해 확인된 값이 아니라 일반적 경향**이다.

**속도결손(velocity deficit) 비율**:

```
ΔU/U0 = (1 − √(1 − Ct)) / (1 + 2·k·x/D)²
```

- `Ct`: 추력계수(thrust coefficient), 대략 0.7~0.9(정격 부근에서 높고 고풍속에서 감소) —
  V126/U136의 확인된 Ct(v) 곡선은 없음(`reports/domain_research/turbine_power_curves.md`가
  이미 3.6MW 버전 정격풍속 자체도 "미확인/근사치"라고 명시한 것과 같은 한계), 문헌 상 대표값
  0.8 정도를 근사 상수로만 쓸 수 있다.
- `D`: 로터직경 (V126=126m)
- `x/D`: 다이아미터 단위 하류거리 — **group1→group2 거리 1.28km ÷ 126m ≈ 10.2D로 고정 상수**
  (지리적 관계라 시간에 따라 변하지 않음).

**대입 시뮬레이션(참고용, 정확한 계수 미확인이므로 크기감 파악용)**: `Ct=0.8`, `x/D=10.2`
가정 시 — `k=0.075`(전형적 육상값) → 속도결손 ≈**8.7%**, `k=0.10`(복잡지형 가정) →
≈**6.0%**. 출력이 풍속의 대략 세제곱에 비례하는 파워커브 램프구간에서는 작은 풍속결손도
`ΔP/P ≈ 3·ΔU/U`로 증폭되므로, 바람이 정확히 정렬됐을 때 개별 터빈 쌍 기준 **출력결손이
18~27%대까지 커질 수 있다**는 정성적 크기감을 준다 (그룹 전체 합산 효과는 터빈별 정렬도 분산,
웨이크 폭 대비 하류 그룹의 가로 퍼짐 등으로 이보다는 완만해질 것).

**"그룹" 단위 근사**: 개별 터빈-터빈 쌍의 웨이크를 다 계산하는 것(전형적 웨이크 배치최적화
방식, 다중 웨이크는 흔히 RSS 또는 선형 합산)은 이 프로젝트 스케일(그룹당 5~6기, 격자 간격
1.5km인 예보 입력)에서는 과한 정밀도다. 실무에서도 팜 단위/캠퍼스 단위 웨이크 스크리닝을 할 때
**대표 좌표(중심 or 무게중심) 간 벡터로 근사**하는 것이 표준적으로 쓰이는 단순화이며(예:
inter-farm wake 연구에서 팜 중심 간 방위각·거리로 30° 섹터 단위 스크리닝), 이 프로젝트에서도
그룹 centroid를 이미 `_group_centroid()`가 계산해두었으므로 그대로 재사용 가능하다.

**웨이크 발생 조건(방향 정렬 허용폭)**: 웨이크는 상류-하류 벡터와 풍향이 완전히 정렬될 때만
최대이고, 벗어날수록 급격히 약해진다. 문헌/표준에서 통용되는 정렬 허용폭:
- IEC 61400-12-1(터빈 성능시험 표준)은 계측탑/터빈이 이웃 터빈이나 장애물의 웨이크 영향을
  받는 "wake-affected sector"를 정의해 시험 데이터에서 제외하도록 요구하며, 실무 적용 시
  섹터 폭은 보통 대상 터빈 간 상대위치·로터직경 비율에 따라 정해진다(표준 자체는 정확한
  각도를 하나로 못박지 않고 절차만 규정 — 이 부분은 표준 원문 유료본을 확보하지 못해 정확한
  숫자는 **미확인**).
- Inter-farm/단지간 웨이크 연구에서는 흔히 **±15°(단일 섹터 30°폭)** 단위로 방향을 나눠
  영향권을 스크리닝하는 사례가 확인됨.
- Wake steering(요잉 제어) 연구에서는 하류 터빈에 유의미한 영향을 주는 정렬 범위로 **±20°**
  수준의 오차/편차가 자주 인용됨.

종합하면 "**±15~30° 이내를 웨이크 영향권으로 본다**"는 실무적 경험칙은 여러 출처에서 정성적으로
뒷받침되지만, 이 프로젝트의 V126/U136·1.28km 거리에 대해 확정된 단일 숫자는 없다 — §4에서는
±25°를 기본값으로 제안하되 `feature-engineer`/`evaluator`가 이후 SCADA로 보정할 수 있는
튜닝 가능한 상수로 명시한다.

## 3. 웨이크가 예측 성능에 미치는 영향 — 기존 연구/사례

- 파키스탄 소재 팜 인접 단지 사례(WRF 기반): **단지간(inter-farm) 웨이크 효과를 고려**하자
  풍속 예측 MAE가 계절에 따라 **7.7%(여름)~14%(겨울)** 감소했고, 출력 예측 NMAE는
  **15~26%** 개선됨 [Adaramola 계열, ScienceDirect 요약].
- CFD 사전계산 유동장 기반 발전량 예측 연구에서도 "웨이크 효과를 고려하면 예측 정확도가 추가로
  향상된다"는 결론 [ResearchGate, "Impact of wake effect on wind power prediction"].
- "단기 풍력발전량 예측 모델에 웨이크 효과가 반영되는 경우는 드물다"는 문제의식에서 출발해,
  해석적 웨이크 모델(엔지니어링 웨이크 모델)을 신경망 입력/기반함수로 결합한
  physics-informed 접근이 순수 신경망 대비 풍속·출력 예측 정확도를 유의하게 개선했다는
  최근 사례 [ScienceDirect, "A physics-inspired neural network model for short-term wind
  power prediction considering wake effects"].
- 실측(SCADA) 기반 웨이크 손실 평가 연구에서는 단지 내 터빈의 **32.8%가 웨이크에 심하게
  영향받고**, 최저 손실월 실제출력/이론출력 비가 84.2%까지 떨어진 사례가 보고됨 [MDPI
  Sustainability, "Research on Evaluation Method of Wind Farm Wake Energy Efficiency Loss
  Based on SCADA Data Analysis"] — 이 프로젝트도 `scada_vestas_train.csv`로 동일한 방법론
  (풍속을 고정하고 풍향만 waked/unwaked로 나눠 터빈별 평균출력 비교)을 그대로 적용해 볼 수
  있다는 방법론적 시사점이 있다 (SCADA는 학습기간 보조신호로만 쓸 수 있다는 CLAUDE.md 제약과
  일치 — 계수 보정용으로 쓰고 test 입력으로는 쓰지 않음).

## 4. 이 프로젝트 데이터의 핵심 제약 — 웨이크는 예보 입력에 원래 없다

**가장 중요한 사실**: LDAPS는 격자간격 ~1.5km, GFS는 ~0.25°(≈20km 이상)다. group1↔group2
거리(1.28km)는 **LDAPS 격자 한 칸보다도 작다.** 즉 원본 기상예보(LDAPS/GFS) 자체는 이런
서브그리드 규모의 터빈-터빈 웨이크 물리를 절대 담고 있지 않으며, 그룹1 중심과 그룹2 중심에서
집계한 풍속/풍향은 사실상 거의 같은 값(같거나 인접 격자)일 가능성이 높다. 따라서:

- 웨이크로 인한 그룹2의 "설명 안 되는 편향"은 **예보 데이터가 아니라 실제 발전(라벨) 쪽에만
  존재하는 효과**이고, 모델이 이를 학습하려면 "예보상 풍향이 웨이크 정렬 방향에 얼마나
  가까운가"라는 **간접 프록시 피처**를 통해서만 접근할 수 있다 — 직접적인 웨이크 후 풍속을
  관측/예보하는 값은 없다.
- 이런 구조상 이 피처들은 "물리적으로 정확한 웨이크 결손"이 아니라 "웨이크가 걸릴 확률이 높은
  기상 상태를 알려주는 신호"로 취급해야 하며, 트리 기반 모델(LightGBM 등)이 그룹2 학습 시
  이 신호와 잔차 편향 사이의 관계를 직접 학습하도록 놓아주는 것이 물리 계수(Ct, k)를 정밀하게
  맞추려는 시도보다 실용적이다.

## 5. feature-engineer가 바로 구현할 수 있는 구체적 피처 제안

**사전 계산 가능한 상수** (지리적 관계, 시간 불변):

```
BEARING_G1_TO_G2_DEG = 115.0          # group1 centroid -> group2 centroid, 지리적 방위각
WAKE_FROM_DIR_DEG      = (BEARING_G1_TO_G2_DEG + 180) % 360 = 295.0
                         # group1이 group2를 웨이크하려면 바람이 이 방향("에서") 불어야 함
DIST_G1_G2_KM          = 1.28
ROTOR_D_V126_M         = 126.0
X_OVER_D_G1_G2         = (DIST_G1_G2_KM * 1000) / ROTOR_D_V126_M ≈ 10.2   # 시간불변 상수
WAKE_SECTOR_HALF_WIDTH_DEG = 25.0     # §2 문헌 경험칙(±15~30°) 중간값, 추후 SCADA로 보정 권장
```

입력 컬럼: `wind_speed_direction()`이 group1(또는 두 그룹이 사실상 같은 LDAPS 격자를 보므로
group1 집계본을 그대로 사용해도 무방, §4 참고)에 대해 만들어내는
`ldaps_10m_dir_sin`/`ldaps_10m_dir_cos`/`ldaps_10m_dir_deg`, `ldaps_10m_speed_mean`
(또는 `spatial_aggregate`가 만드는 group1의 `_idw`/`_nearest` 변형) — 기존 파이프라인 산출물을
그대로 재사용, 새 원본 컬럼 파싱 불필요.

**① `wake_alignment_cos_g1g2`** — 방향 정렬도 (연속값, [-1, 1]):

```
wake_alignment_cos_g1g2 = cos(dir_deg − WAKE_FROM_DIR_DEG)
                        = dir_cos·cos(WAKE_FROM_DIR_DEG) + dir_sin·sin(WAKE_FROM_DIR_DEG)
```

(우변은 이미 계산되어 있는 `dir_sin`/`dir_cos`만으로 `atan2` 재계산 없이 바로 구현 가능 — 각도
차를 명시적으로 구하지 않는 삼각함수 항등식 이용.) `+1`에 가까울수록 바람이 정확히 295°에서
불어와 그룹1이 그룹2를 정면으로 웨이크하는 상태, `-1`은 정반대(그룹2가 오히려 상류), `0`은
측풍(웨이크 축과 무관).

**② `wake_sector_exposure_g1g2`** — 웨이크 영향권 여부의 완만한(0 나눗셈 없는) 지표, §2의
±25° 경험칙 기반, 하드컷 대신 램프로 트리모델이 경계에서 급단절을 다시 학습할 필요 없게:

```
angle_diff = min(|dir_deg − WAKE_FROM_DIR_DEG|, 360 − |dir_deg − WAKE_FROM_DIR_DEG|)
wake_sector_exposure_g1g2 = max(0, 1 − angle_diff / WAKE_SECTOR_HALF_WIDTH_DEG)
```

`angle_diff=0`(완전 정렬)일 때 1, `angle_diff≥25°`일 때 0. 이진 플래그가 필요하면
`wake_sector_exposure_g1g2 > 0`으로 파생.

**③ `wake_deficit_proxy_g1g2`** — Jensen 모델 속도결손을 근사한 스칼라(단위: m/s, "그룹2가
그룹1 웨이크로 인해 잃을 것으로 추정되는 풍속"), §2의 대입 시뮬레이션 값을 상수로 고정하고
정렬도·풍속과 곱한 형태:

```
WAKE_DEFICIT_FRAC_CONST ≈ 0.06~0.09   # §2 계산: Ct=0.8, k=0.075~0.10, X_OVER_D_G1_G2=10.2 가정
wake_deficit_proxy_g1g2 = ldaps_10m_speed_mean(group1)
                          * max(0, wake_alignment_cos_g1g2)
                          * WAKE_DEFICIT_FRAC_CONST
```

`max(0, ·)`로 클리핑하는 이유: cos<0(그룹2가 상류)일 때는 웨이크가 물리적으로 존재하지
않으므로 결손이 0이어야 함. `WAKE_DEFICIT_FRAC_CONST`는 미확인 Ct·k에 의존하는 **근사
상수**이므로, 여유가 되면 `scada_vestas_train.csv`에서 7~12호기(그룹2側 터빈)의 실측
출력을 "정렬됨(`wake_sector_exposure_g1g2` 높음)" vs "정렬 안 됨" 구간으로 나눠 평균출력비를
비교하는 방식(§3의 SCADA 기반 웨이크 손실 평가 방법론)으로 경험적으로 재추정하는 것을 권장—
이 프로젝트의 turbine_power_curves.md가 이미 채택한 "제조사 미확인값은 SCADA로 보정"이라는
관례와 일치한다.

파생 활용: `wake_deficit_proxy_g1g2`를 그룹1의 원 풍속에서 빼(`speed − proxy`) 기존
`power_curve_transform()`에 넣으면 "웨이크를 반영한 그룹2 기대 파워커브 비율"과 "웨이크
미반영 파워커브 비율"의 차이를 4번째 피처로 추가할 수도 있다(옵션, 우선순위는 ①~③보다 낮음).

## 출처

- [WindPRO/PARK 웨이크 모델 소개 PDF (EMD)](https://help.emd.dk/knowledgebase/content/ReferenceManual/Wake_Model.pdf) — PARK/Jensen 모델 소개 문서(표지·개요만 확인, 본문 수식 페이지는 접근 실패)
- [PyWake 공식 문서 — Wake Deficit Models](https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/WakeDeficitModels.html) — Jensen(NOJ) 모델의 정확한 속도결손 수식, entrainment constant 표기 확인
- [FLORIS Wake Models 문서](https://nrel.github.io/floris/wake_models.html) — Jensen 모델 계열 교차확인용 검색결과
- 웨이크 감쇠상수 k 관련 종합 검색 결과 — WAsP 권장값 onshore k≈0.075, offshore k≈0.04~0.05,
  근사식 `k=0.5/ln(h/z0)` (검색엔진 스니펫 기반, 1차 출처인 Peña 2016 Wind Energy 논문
  ([DOI 10.1002/we.1863](https://onlinelibrary.wiley.com/doi/full/10.1002/we.1863))과
  WindEurope 2016 발표자료는 결제장벽/PDF 파싱 실패로 직접 인용문 확보는 못했음 — **k 값
  자체는 2차 출처 스니펫 신뢰도로 취급할 것**)
- [Jensen wake model 리뷰 (ResearchGate, "A Review of Wind Turbine Wake Models for Microscale Wind Park Simulation")](https://www.researchgate.net/publication/334330334_A_REVIEW_OF_WIND_TURBINE_WAKE_MODELS_FOR_MICROSCALE_WIND_PARK_SIMULATION)
- [IEC 61400-12-1 표준 개요 (ANSI 블로그/webstore 미리보기)](https://blog.ansi.org/ansi/iec-61400-12-1-performance-measurement-wind-turbines/) — wake-affected sector 제외 절차 존재 확인, 정확한 각도 수치는 원문 유료본 필요(미확인)
- Inter-farm wake 30° 섹터 스크리닝, wake steering ±20° 정렬 오차 — 검색 결과 종합
  ([ScienceDirect, Inter-farm wake effect on layout optimization](https://www.sciencedirect.com/science/article/pii/S2950601824000216), [PNAS, Wind farm power optimization through wake steering](https://www.pnas.org/doi/10.1073/pnas.1903680116))
- [ScienceDirect, "On the wake effect in wind farm power forecasting: a new data-driven approach" (E3S Conferences)](https://www.e3s-conferences.org/articles/e3sconf/pdf/2020/57/e3sconf_ati2020_08016.pdf) — inter-farm 웨이크 반영 시 풍속 MAE 7.7~14%, 출력 NMAE 15~26% 개선 (검색 스니펫 기반, PDF 원문 직접 파싱은 실패)
- [ResearchGate, "Impact of wake effect on wind power prediction"](https://www.researchgate.net/publication/304011733_Impact_of_wake_effect_on_wind_power_prediction) — 웨이크 반영 시 예측 정확도 추가 향상 (초록 수준 확인, 상세 수치는 페이지 접근 제한으로 미확인)
- [ScienceDirect, "A physics-inspired neural network model for short-term wind power prediction considering wake effects"](https://www.sciencedirect.com/science/article/abs/pii/S0360544222020989) — 해석적 웨이크 모델을 신경망에 결합해 순수 NN 대비 정확도 개선 (검색 스니펫 기반)
- [MDPI Sustainability, "Research on Evaluation Method of Wind Farm Wake Energy Efficiency Loss Based on SCADA Data Analysis"](https://www.mdpi.com/2071-1050/16/5/1813) — 실측 SCADA 기반 웨이크 손실 평가 방법론(팜 내 32.8% 터빈이 웨이크 심각 영향, 최저월 실제/이론 출력비 84.2%) — 이 프로젝트의 SCADA 보정 방법론 제안 근거
- `docs/info_raw.csv`, `docs/turbine_kpx_mapping.md`, `src/features/weather_features.py` —
  좌표/방위각 재계산 및 기존 피처 파이프라인(`wind_speed_direction`, `spatial_aggregate`,
  `power_curve_transform`)과의 연동점 확인용 (프로젝트 내부 자료)
- `reports/domain_research/turbine_power_curves.md` — V126/U136 로터직경·Ct 미확인 상태에
  대한 기존 조사(§4의 "확인된 값 vs 근사치" 구분 관례를 이 리포트에서도 따름)

## 미확인 — 추가 조사/검증 필요 시

- V126-3.6MW(사이트 특화 모드)의 실제 Ct(v) 곡선 — 제조사 미공개, §2/§4의 `Ct≈0.8`은 일반
  문헌값 근사치일 뿐 이 터빈 고유값이 아님.
- 이 사이트(태백가덕산 능선, 복잡지형)에 맞는 wake decay constant k의 실측/보정값 — 문헌은
  onshore 평지 기준 0.075를 표준으로 제시하나 능선 지형에서의 실제값은 미확인.
  `scada_vestas_train.csv`로 경험적 보정 시도 시 얻을 수 있음 (§5 권장사항 참고).
  IEC 61400-12-1 "wake-affected sector" 제외 각도 원문 수치 — 표준 원문(유료) 미확보로
  정확한 각도 기준 미확인, ±15~30° 범위는 여러 2차 출처 종합 추정치.
