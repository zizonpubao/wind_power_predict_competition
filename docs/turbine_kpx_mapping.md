# 터빈 ↔ KPX 그룹 매핑 (info.xlsx 파싱 결과)

`info.xlsx`는 인코딩 문제로 터미널에서 직접 열면 한글이 깨진다. `docs/info_raw.csv`(UTF-8-SIG로 재저장, 원본 미수정)와 아래 요약을 대신 참고할 것.

## 요약 매핑

| 단계 | 명칭 | 제작사 | 모델명 | 호기 범위 | SCADA 컬럼 접두사 | KPX 그룹 | 그룹설비용량 |
|---|---|---|---|---|---|---|---|
| 1 | 태백가덕산 | VESTAS | V126 | 1~6 | `vestas_wtg01`~`vestas_wtg06` | `kpx_group_1` | 21.6 MW (3.6MW × 6) |
| 1 | 태백가덕산 | VESTAS | V126 | 7~12 | `vestas_wtg07`~`vestas_wtg12` | `kpx_group_2` | 21.6 MW (3.6MW × 6) |
| 2 | 태백가덕산/태백원동 | UNISON | U136 | 1~5 | `unison_wtg01`~`unison_wtg05` | `kpx_group_3` | 21.0 MW (4.2MW × 5) |

- Hub Height: VESTAS V126 = 117 m, UNISON U136 = 117 m
- Rotor Diameter: VESTAS V126 = 126 m, UNISON U136 = 136 m
- `KPX그룹`, `그룹설비용량(MW)` 컬럼은 각 그룹의 대표(첫 호기) 행에만 값이 채워져 있고 나머지는 병합 셀이라 NaN — 위 표처럼 그룹 단위로 broadcast해서 사용해야 함.

## 왜 중요한가

- `scada_vestas_train.csv` / `scada_unison_train.csv`의 개별 터빈 컬럼을 위 매핑대로 그룹핑해서 합산하면 `train_labels.csv`의 `kpx_group_*`와 (원칙적으로) 대응되어야 함. 이 관계를 **실제로 검증**하는 것이 EDA 1순위 작업 (SCADA 합산치 vs 공식 라벨 간 오차/스케일 확인 — 정격 개수, 가동정지, 손실분 등으로 완전히 일치하지 않을 수 있음).
- SCADA `power_kw10m`은 10분 단위 순시 평균 출력(kW)이다. `train_labels.csv`의 라벨은 1시간 집계 `kWh`. 10분 값 6개를 평균해 kW → kWh(×1h) 환산하는 절차가 필요 (10분 평균 파워의 시간 평균 = 그 시간대 평균 출력이므로 그대로 kWh 취급 가능하지만, 시간 경계 정렬은 별도 검증 필요).
- 이 매핑은 SCADA를 보조 피처(터빈 단위 결측 대체, 이상치 탐지, 발전기 가동상태)로 쓸 때만 필요하고, **test 기간에는 SCADA가 없으므로** 예측 모델의 직접 입력으로는 쓸 수 없음. 기상예보 → 그룹 발전량 매핑 함수를 학습하는 데 SCADA는 학습기간 한정 보조 신호로만 활용 가능.
