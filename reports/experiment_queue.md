# 일일 실험 파이프라인 — 대기열·프로토콜·결과 장부

목표: **매일 제출 후보 5개**를 자동 생산하고, 리더보드 점수를 피드백 삼아 대기열을 갱신한다.
핵심 논리: 이 대회에서 CV는 ±0.002 차이를 못 가르지만 리더보드(Public 40%)는 0.0001 단위로
가른다 → **하루 5회 제출 = 하루 5번의 실전 실험**. 배수 재보정(+0.0053)도 이 방식으로 찾았다.

## 운영 프로토콜 (매일)

1. **아침 배치 (자동, cron)**: 대기열 상위 항목부터 후보 5개 CSV를 `submissions/`에 생성.
   각 후보는 "무엇을 검증하는가" 한 줄과 예상 결과를 브리핑에 명시.
2. **사용자 제출**: 5개를 DACON에 제출하고 점수 스크린샷/숫자를 대화에 붙임.
3. **저녁 정산 (자동 또는 사용자 신호 시)**: 결과를 아래 장부에 기록, 대기열 갱신,
   ** 갱신**(점수 여정·장부·시도 보드 반영)
   (이긴 방향은 후속 실험 파생, 진 방향은 폐기). 최고 기록 갱신 시 메모리·제출선택 갱신.

## 하드 룰

- **안전망**: 현재 최고(`submission_20260724_calib_multcalib_half.csv`, 0.6280)는 항상
  리더보드 "제출선택"에 유지. 프로브가 넘어설 때만 교체 권고.
- 후보 1개당 검증 변수는 1개 (변수 하나만 바꿔 원인 식별 가능하게).
- 컴퓨트 예산: 하루 배치 전체가 세션 사용한도(5시간) 안에 여유 있게 끝나야 함.
  한도 임박 시 새 작업 시작 금지, 만들어진 후보까지만 브리핑.
- 결과 판정은 정직하게: 노이즈(±0.001)면 "무승부"로 기록, 억지 해석 금지.
- 환경: conda 삭제됨 → venv `C:\Users\heelo\.venv` (pandas/pyarrow 설치됨, 필요시
  lightgbm/scikit-learn/joblib 추가). torch 없음 → 신경망 재학습 불가, 기존 모델의 test 예측이
  담긴 기존 제출 CSV를 산술 조작(스왑/스케일)하는 방식 우선.

## 대기열 (우선순위순 — 매일 갱신)

| # | 실험 | 검증하는 것 | 방법/비용 | 상태 |
|---|---|---|---|---|
| 1 | 물리승자 GBM 스왑 블렌드 | 물리 피처(격자분산·ρv³)가 실전에서 통하나 (CV +0.0012 노이즈 → 리더보드로 해상) | 기존 run `20260724_162111` GBM test 예측 → 블렌드에서 GBM만 교체 → half 재보정. lightgbm 설치 필요 | 생성완료 (`submission_20260724_probe_physicsgbm_halfrecal.csv`) |
| 2 | 결정최적화 w_ficr=0.6 | FICR 가중 상향이 실전 FICR을 더 버나 (CV 평탄 → 리더보드로) | 기존 GBM 모델 로드 → w_ficr=0.6로 결정단계만 재실행 → 스왑+재보정 | 생성완료 (`submission_20260724_probe_wficr06_halfrecal.csv`) |
| 3 | ECMWF IFS 피처 (부분 커버리지) | 제3 NWP가 신호를 더하나. 커버리지 2024-04+뿐이지만 **LightGBM은 NaN 네이티브 처리** → 결측 그대로 두고 학습 가능 | Open-Meteo Previous Runs API `previous_day2`(D-2 12z, 리키지 검증 tests/test_ecmwf.py) 백필 → `data/interim/ecmwf_ifs.parquet` → 피처 11개 병합 → GBM 재학습 `20260807_142800` | 생성완료 (`submission_20260807_probe_ecmwf_halfrecal.csv`). CV 전체 0.6018 vs 기준 0.6038 (-0.0020, 노이즈권). 커버 구간(2024-04+) OOF: g1 +0.0038 / g2 **-0.0229** / g3 -0.0006 → 커버 구간에서도 순증 없음(g2 악화 주도). ecmwf_ws100-target corr 0.773 > ldaps 0.744로 원신호는 강함 — 기대 낮게 리더보드로 최종 판정 |
| 4 | 19-분위수 GBM | 분포 해상도↑ → 결정최적화 정밀도↑ | 분위수 19개로 재학습(CPU) | 생성완료 (`submission_20260807_probe_q19_halfrecal.csv`). 원본(기준) 피처셋으로 재학습(run `20260807_192239_gbm_quantile_pruned_q19`). CV overall 0.6051 vs 기준 0.6038 (+0.0013), 3그룹 전부 동일 방향 개선(g1 0.6102 +0.0009 / g2 0.6344 +0.0007 / g3 0.5706 +0.0022) — 노이즈 경계선(±0.001)이지만 방향이 일관돼 실전 프로브로 판정 |
| 5 | 월별 재보정 배수 (절반 강도) | 편향의 계절성(겨울↑여름↓, ficr_gap_diagnosis 근거) 반영 | OOF에서 월별 factor fit(연간 배수로 50% 수축) → 절반 강도 → base에 적용 | 생성완료 (`submission_20260807_probe_monthlyrecal_half.csv`) |
| 6 | 블렌드 가중 GBM 0.25 프로브 | 0.20/0.30 사이 미확인 지점 (half 재보정 위에서 비교) | base·gbm20 CSV 행별 평균 → half 배수 → clip | 생성완료 (`submission_20260807_probe_gbm25_halfrecal.csv`) |
| 6b | 재보정 강도 0.6 | 강도 곡선 0.5(0.6280)↔1.0(0.6205) 사이, 피크가 0.5 오른쪽인지 | base × (1+0.6×(full배수−1)) = g1 1.072/g2 1.036/g3 1.069 → clip | 생성완료 (`submission_20260807_probe_strength06.csv`) |
| 7 | 결정최적화 n_grid 201 | 후보 해상도 1%→0.5%가 문턱 근처를 더 잘 잡나 | 결정단계만 재실행 | 대기 |
| 8 | GBM 시드 배깅 (3시드) | 분위수 추정 분산 축소 | seed 3개 재학습 후 분위수 평균(CPU) | 생성완료 (`submission_20260809_probe_seedbag_halfrecal.csv`). `src/training/train_gbm_seedbag.py` 신규 구현(3개 시드 각각 fit → 분위수 배열 평균 → 결정최적화 1회). 191416과 동일 피처셋(canonical, exp A/B 미적용) 기준 seed={42,202,777}: g1 0.6116(+0.0012)/g2 0.6337(±0.0000)/g3 0.5708(+0.0024), overall 0.6054 vs 0.6042(**+0.0012**) — 3그룹 전부 악화 없이 flat~개선, 일관된 방향이라 프로브 채택. selected_features.json은 변경 없음(모델 앙상블 방식이라 피처 선택과 무관) |
| 9 | 풍속 quantile-mapping 편향보정 | LDAPS 풍속의 분포 편향을 SCADA 실측 풍속으로 교정(학습기간 fit, test 적용 가능 함수) | 매핑 fit → 피처 추가 → 재학습 | 생성완료 (`submission_20260809_probe_qmwind_halfrecal.csv`). SCADA에 나셀 풍속 컬럼(`*_ws`, 이상치 없음) 실존 확인 → "데이터 없음" 기각 아님. `src/features/wind_qm.py`(백분위 매칭 bias correction, train쌍으로만 fit, frozen 함수로 test 적용 → 리키지 없음) 신규 구현 + 단위테스트(`tests/test_wind_qm.py`, 5건). 전그룹 추가 run `20260809_103230`(exp A 위에 적용): g1 0.6101(-0.0003, 중립) / g2 **0.6383(+0.0018, exp A 위에 추가 개선)** / g3 0.5675(-0.0009, 소폭악화). g2-단독 재학습(`20260809_104004`)으로 격리 확인: g2 0.6383(전그룹run과 동일) & g1/g3 test 예측 byte-identical(exp A 기준 대비 오염 없음). **selected_features.json에 g2만 `ldaps_ws_qm`/`ldaps_ws_qm_power_curve` 영구 반영(manual_overrides 기록)** |
| 10 | ICON g1-한정 (제4 NWP) | ECMWF g1 수확과 동일 패턴이 제4 NWP에도 통하나 | `src/data/fetch_icon.py`(DWD icon_global, Open-Meteo Previous Runs API, offset_days=2) 백필 → g1 피처 11개 추가 → GBM 재학습 | **기각** (`20260807_233518` 계열 run `20260807_232548_gbm_quantile_pruned`). 리키지 검증: `tests/test_icon.py`(20 테스트 통과, ECMWF와 동일 offset=2 산술 재사용, 보수적으로 ECMWF의 7h34m 지연을 그대로 차용 — DWD 실제 배포시각은 별도 확인 못 함). 백필 16,824행, 실제 커버리지는 2024-02-17+(기대했던 2022-11-24+는 offset 미적용 원본 아카이브 한정이며 리키지-세이프 API로는 확인 안 됨). CV: g1 0.6081 vs 기준 0.6104 (**-0.0023**), 커버구간 OOF 0.6303 vs 0.6370 (**-0.0067**) — ECMWF와 반대로 g1을 악화시킴. g2/g3 불변 확인(byte-identical). **selected_features.json 원복 완료(groups 섹션 canonical과 byte-identical), 프로브 미생성** |
| 12 | ECMWF g1 한정 재시도 | ECMWF 원신호는 target 상관 0.773으로 LDAPS(0.744)보다 강한데 g2가 통합을 망침(-0.0229) — g1만(+0.0038) 쓰면 순증인가 | 그룹별 선택 포함으로 재학습(CPU) | 생성완료 (`submission_20260807_probe_ecmwfg1_halfrecal.csv`). `configs/selected_features.json`에 g1만 ECMWF 11개 피처 영구 반영(manual_overrides 기록, 2026-08-07). run `20260807_191416_gbm_quantile_pruned`: CV overall 0.6042 vs 기준 0.6038 (+0.0004), g1 전체 CV 0.6104(+0.0011)/커버구간(2024-04+) OOF 0.6411 vs 기준 0.6373 (**+0.0038, 이전 전체그룹 실험과 정확히 일치**) — g2/g3 예측값이 기준과 완전히 동일함을 직접 확인(byte-identical, 버그 없음). 가설 확인: g1 단독 적용 시 g2 오염 없이 순증 |
| 11 | torch 재설치 + 신경망 재통합 | 신경망 트랙 복구 (현재 블렌드는 기존 CSV 산술로만 유지 가능) | ~2.5GB 설치, 필요 시점에 | 보류 |
| 13 | 재보정 배수 재적합 (g1 GBM 교체 반영) | 새 g1 GBM(191416)으로 블렌드 OOF가 바뀌었으니 half 배수(1.06/1.03/1.0575)가 여전히 최적인가 | 새 블렌드(0.30×gbm_191416+0.35×lstm+0.35×transformer) OOF로 `fit_group_factor` 그리드([0.85,1.12]) 재적합 | **완료, 무승부**: 재적합 full-strength 배수 = g1 1.12(그리드 상한 포화)/g2 1.06/g3 1.115 — 기존과 **소수점까지 완전 동일**(half 적용해도 1.06/1.03/1.0575 그대로). g1이 두 번 다 그리드 상한(1.12)에서 포화되는 구조적 한계이지 신호 변화가 아님. 프로브 불필요(생성 안 함) |
| 14 | ECMWF g3 추가 | g3 커버구간이 전그룹 실험에서 -0.0006(중립)이었음 — g3 단독으로도 그런가 | g1 유지, g3에만 ECMWF 11피처 추가(g2 제외) 재학습 | **기각** (`20260807_233511_gbm_quantile_pruned`). g1 byte-identical(0.6104, 격리 확인) / g3 CV 0.5680 vs 기준 0.5684(**-0.0004**), 커버구간 OOF 0.5897 vs 0.5903(**-0.0006**) — 원 전그룹 실험의 g3 중립~소폭악화 결론과 정확히 일치. selected_features.json 원복 완료(canonical과 byte-identical), 프로브 미생성 |
| 15 | g2 물리 피처 (격자분산·ρv³) 한정 | g1은 ECMWF 신소재를 얻었지만(#12) g2는 아직 없음 — 과거 물리 게이트 최강 신호(+0.0027 추정)가 g2 단독으로 실제 개선인가 | g1 ECMWF 유지, g2에만 `ldaps_10m_grid_range`/`ldaps_10m_grid_std`/`ldaps_rho_v3_10m` 3개 추가(g3 제외) 재학습(CPU) | 생성완료 (`submission_20260809_probe_g2physics_halfrecal.csv`). run `20260809_101906_gbm_quantile_pruned`: g2 CV 0.6365 vs 기준 0.6337 (**+0.0028**, 과거 게이트 추정치와 거의 일치), overall 0.6051 vs 0.6042(+0.0009). g1/g3 test 예측 byte-identical(격리 확인, 오염 없음). **selected_features.json에 g2만 3개 피처 영구 반영(manual_overrides 기록)** — 이후 #9(qm풍속)이 이 위에 추가로 얹어짐(g2 최종 0.6383) |
| 16 | 전부 결합 (A+B+C) | 오늘 채택된 g2 물리(#15)+g2 QM풍속(#9)이 이미 반영된 canonical selected_features(g1=76/g2=71/g3=67) 그대로 시드 배깅(#8, seed 42/202/777)까지 얹으면 누적 효과가 실제로 합산되나 | canonical 그대로 `train_gbm_seedbag.py` 재학습(추가 피처 변경 없음) → 스왑 산식 프로브 | 생성완료 (`submission_20260809_probe_allcombo_halfrecal.csv`). run `20260809_112118_gbm_seedbag_pruned`: g1 0.6116(+0.0012)/g2 0.6381(+0.0044)/g3 0.5708(+0.0024), **overall 0.6068 vs 기준 0.6042(+0.0026, 오늘 CV 최고)** — A+B+C 효과가 서로 상쇄 없이 누적됨(g2는 단독 QM run 0.6383과 거의 동일, 시드배깅이 g2엔 중립이라는 #8 관찰과 일치). 학습 중 `train_gbm_seedbag.py`의 `SeedBaggedGBMQuantileModel.__module__` 재작성 로직이 `-m` 실행 시 `PicklingError`를 일으키는 버그 발견·수정(`sys.modules` identity aliasing 추가 + OOF를 모델 dump보다 먼저 저장하도록 순서 변경, 스모크테스트로 fix 검증 후 재학습). selected_features.json 변경 없음(이미 canonical에 A/B 반영됨) |

## 결과 장부

| 날짜 | 제출 파일 | 검증한 것 | 점수 | 판정 |
|---|---|---|---|---|
| 07-22 | lgbm_tuned_pruned | LightGBM 단독 트랙 | 0.6091 | 기준점 |
| 07-23 | gbm_quantile | 분위수+결정최적화 | 0.6126 | ✅ +0.0035 |
| 07-23 | nested_lofo | CV최적 블렌드 가중 | 0.6172 | ✅ (단, 고정에 짐) |
| 07-23 | v14_fixed | 고정 균등 블렌드 | 0.6227 | ✅ +0.0055 |
| 07-23 | pointblend gbm20 | 신경망-heavy 가중 | 0.6214 | ❌ 피크 확인용 |
| 07-24 | multcalib_full | 배수 재보정 full | 0.6205 | ❌ 과보정 |
| 07-24 | multcalib_half | 배수 재보정 half | **0.6280** | ✅ **현재 최고** |
| 07-24 | g3fix | g3 배수만 상향 | 0.6271 | ❌ 과보정 |
| 08-07 | probe_q19 | 19-분위수 분포 해상도 | 0.6274 | ❌ CV방향 미전이 |
| 08-07 | **probe_ecmwfg1** | **ECMWF g1 수확** | **0.6302** | ✅ **신기록 +0.0022** |
| 08-07 | probe_monthlyrecal | 계절별 재보정 배수 | 0.6252 | ❌ 계절분해 역효과 |
| 08-07 | probe_strength06 | 재보정 강도 0.6 | 0.6266 | ❌ 피크=0.5 확정 |
| 08-09 | probe_g2physics | g2 물리 피처(#15) | (제출 대기) | CV +0.0028 (격리 검증 완료) |
| 08-09 | probe_qmwind | g2 SCADA 풍속 QM(#9) | (제출 대기) | CV +0.0018 (g2physics 위, 격리 검증 완료) |
| 08-09 | probe_seedbag | 시드 배깅 3개(#8) | (제출 대기) | CV +0.0012 (3그룹 전부 flat~개선) |
| 08-09 | **probe_allcombo** | A+B+C 전부 결합(#16) | (제출 대기) | **CV +0.0026 (오늘 CV 최고)** |
| 08-07 | probe_wficr06 | 결정단계 FICR 가중 0.6 | 0.6277 | ❌ 무승부(0.5 유지) |
| 08-11 | probe_allcombo | g2재료+QM+시드배깅 결합 | 0.6305 | ✅ +0.0003 (미세) |
| 08-11 | probe_g2physics | g2 물리 재료 분리 | 0.6311 | ✅ +0.0009 |
| 08-11 | probe_qmwind | g2 물리+풍속QM | 0.6316 | ✅ +0.0014 |
| 08-11 | probe_seedbag | 시드배깅 분리 | 0.6295 | ❌ 소폭 하회 |
| 08-11 | **probe_ecmwf(전그룹)** | **ECMWF 3그룹 전부** | **0.6330** | ✅ **신기록 +0.0028 — CV의 g2 -0.0229 경고가 실전에서 역전!** |
