# 외부 데이터 후보 조사 — LDAPS/GFS 너머 신호 (BARAM 2026)

조사일: 2026-07-24 · 담당: domain-researcher

## 결론 (먼저)

**가져올 가치 있는 top 1개: ECMWF IFS HRES 과거예보 (Open-Meteo Historical Forecast API 경유).**
우리의 두 제약을 실제로 통과하는 **유일한** 후보다.
- 2022-2025 전 기간(학습+평가) 커버 (IFS 아카이브 2017-01-01~).
- LDAPS(KMA)·GFS(NCEP)와 **독립적인 제3의 NWP** → 진짜 새 신호 가능성. EDA에서 LDAPS 0.73 / GFS 0.55였는데 ECMWF IFS는 통상 GFS보다 스킬이 높아 앙상블/보정 재료로 상방이 있음.
- 리키지: **D-2 12 UTC 런**을 쓰면 대상일 전날 14:00 KST 이전 확정 → 규칙 통과.
- API로 init 시각 고정 재현 가능, CC-BY-4.0(상업/재배포 가능, 출처표기).

**second: ICON (DWD)** — 독립 모델이나 아카이브가 2022-11-24부터라 2022 대부분 결손. 2023-2025만 커버. ECMWF 붙인 뒤 여력 있으면 추가 검토.

나머지(ERA5·인근 AWS·KPX curtailment·GEFS reforecast·KMA 동네예보)는 **커버리지 또는 리키지에서 탈락**. 상세는 아래.

---

## 후보별 평가표

| 후보 | 접근성/라이선스 | 2022-2024 (학습) | 2025 (평가) | 리키지 규칙 | 시간해상도 | 실무난이도 | 예상 신호가치 | 판정 |
|---|---|---|---|---|---|---|---|---|
| **ECMWF IFS HRES** (Open-Meteo Hist. Forecast API) | 공개, CC-BY-4.0, 상업/재배포 가능 | ✅ 2017~ 전체 | ✅ 커버 | ✅ D-2 12z 런 사용 시 통과 | 1h(보간), 원 9km(2024 이후)/0.4°(이전) | 낮음(REST API, 좌표+init) | **높음** — GFS/LDAPS와 독립 NWP | **채택 1순위** |
| **ICON (DWD)** (Open-Meteo) | 공개, CC-BY | ⚠️ 2022-11-24~ (2022 대부분 결손) | ✅ | ✅ 동일 | 1h | 낮음 | 중 — 독립 글로벌/유럽 모델 | 2순위(보조) |
| **KMA GDPS 10km** (Open-Meteo) | 공개, CC-BY | ❓ 시작일 미확인 | ✅ | ✅ | 1h | 낮음 | 낮음 — LDAPS와 같은 KMA 계열, 상관 높음 | 보류 |
| **KMA LDPS 1.5km** (Open-Meteo) | 공개 | — | — | — | — | — | **없음 — 대회 제공 LDAPS와 동일 모델** | 중복, 기각 |
| **GEFS reforecast v12** (NOAA AWS) | 공개, PD | ❌ 2000-2019만 | ❌ | — | 0.25°, 1일1회 | 중 | — | 커버리지 탈락 |
| **GEFS 운영앙상블** (NODD, 2017~) | 공개, PD | ✅ | ✅ | ✅ | 0.25° | 중(grib2 대용량) | 낮음 — GFS와 강상관 | 저우선 |
| **ERA5 재분석** (CDS) | 공개, Copernicus | ✅ | ⚠️ ERA5T ~5일 지연 | ❌ **재분석=대상시각 관측 반영** | 1h, 0.25° | 중 | 학습보정용만 | **평가 피처 불가** |
| **인근 산악 AWS** (태백/정선/삼척) | 공개(KMA) | ✅ 실측 | ❌ 2025 실측 사용 불가 | ❌ 프록시화 | 1h | 낮음 | 낮음 — 능선 허브 대표성·프록시 한계(석포/대관령 교훈) | 기각 |
| **KPX curtailment/계통** | 부분공개 | 사후 실적 | 사후 | ❌ 대개 사후·전날14시 前 미확정·단지별 아님 | — | 중 | 라벨 급락 원인 설명용 | 예측피처 불가 |

---

## 상세 근거

### 1. ECMWF IFS HRES — 채택 1순위
- **아카이브**: Open-Meteo Historical Forecast API가 IFS를 2017-01-01부터 보관(운영 런을 연속 시계열로 stitch). 2024-02 이후 원해상도 0.25°/9km, 이전은 0.4°. 우리 사이트(37.28N,128.96E) 좌표 질의 가능.
- **독립성**: 대회 제공은 KMA LDAPS + NCEP GFS. ECMWF IFS는 유럽센터 독립 모델로 중기 스킬 최상위. 두 기존 모델과 상관이 완전하지 않아 앙상블/오차보정에 새 정보 기여 여지 큼.
- **리키지 계산**: 컷오프 = 대상일 D 전날 14:00 KST(=05:00 UTC D-1).
  - IFS 00z(D-1) 배포 05:45–07:34 UTC = 14:45–16:34 KST D-1 → **컷오프 이후, 사용 불가.**
  - IFS **12z(D-2)** 배포 17:45–19:34 UTC D-2 = 02:45–04:34 KST D-1 → **컷오프 이전, 사용 가능.** 대상 D 00–23시까지 리드 ~27–51h. (대회 LDAPS도 사실상 day-ahead라 리드타임 급 유사.)
  - 재현: Open-Meteo Single Runs/Previous Runs API로 init 시각을 12z(D-2)로 고정해 leakage-safe 하게 재추출 가능.
- **라이선스**: Open-Meteo 데이터 CC-BY-4.0(상업·재배포 허용, 출처표기). ECMWF open data도 2025-10 이후 CC-BY-4.0 완전개방 → 원천에서 직접 받을 수도 있으나 ECMWF 자체 open-data 포털은 rolling 아카이브(최근분)만 → **과거 2022-2024는 Open-Meteo 경유가 실무적**.
- **주의**: Open-Meteo는 재배포가 아니라 API 사용이므로 대회의 "비유출·재현가능·누구나 접근" 요건 충족. 무료 tier rate-limit(비상업 소량)만 유의, 백필은 좌표 1점이라 부담 작음.

### 2. ICON (DWD) — 2순위
- Open-Meteo 아카이브 2022-11-24~. 2022년 대부분 결손이라 학습 전기간 커버 실패지만 2023-2025는 완전. 독립 모델(비-KMA/비-NCEP)이라 보조 앙상블 멤버로 가치. ECMWF 먼저 검증 후 여력 시.

### 3. ERA5 — 평가 피처로는 불가, 학습보정용만
- 재분석은 대상시각의 실제 관측을 동화 → 예측시점엔 알 수 없는 정보. 2025 평가에 live 피처로 쓰면 리키지. 또 ERA5T도 ~5일 지연이라 실시간 부재.
- **유일한 합법 용법**: 2022-2024 학습구간에서 LDAPS/GFS의 계통편향을 ERA5 기준으로 offline 진단·보정 규칙 학습(피처가 아니라 전처리 파라미터). 단 2025엔 ERA5를 못 넣으므로 "LDAPS→진실"의 보정계수만 이전 적용 → 이득 제한적, 우선순위 낮음.

### 4. 인근 산악 AWS — 기각 (기존 교훈 반영)
- 석포(831) 골짜기·대관령 프록시 실패와 동일 구조: (a) 능선 허브 117m 풍속 대표성 부족, (b) 2025 실측 사용 불가 → LDAPS/GFS 정보만 담긴 프록시로 축소되어 순증 신호 없음. 태백/정선/삼척 고지 AWS를 추가로 파도 (b) 한계는 불변이라 근본 무용.

### 5. KPX curtailment/계통 — 예측 피처 불가
- 출력제한 실적·정산 계통데이터는 사후 공개가 일반적이고, 하루전 발전계획도 단지별 재생에너지 curtailment를 D-1 14:00 KST 이전에 확정·공개하지 않음 → 리키지/재현성 탈락. 다만 **라벨 급락 구간의 원인 해석(EDA)** 용도로는 참고 가치. 예측 입력에는 못 씀.

### 6. GEFS — reforecast는 2000-2019(커버 탈락). 운영앙상블(2017~)은 커버·리키지 통과하나 GFS와 강상관이라 순증 낮고 grib2 용량부담. ECMWF 대비 후순위.

### 7. 기상청 API허브 단기예보(동네예보) — 5km 격자, 3h 주기, 2008~ 보유. 그러나 이는 KMA 후처리 산물로 LDAPS/GDPS와 같은 KMA 계열 → 독립 신호 약함. 과거 발표분 재현(특정 tmFc 조회)은 가능하나 ECMWF 대비 이점 없음.

---

## 종합 순위 (실현가능성 × 잠재력)

1. **ECMWF IFS HRES (Open-Meteo)** — 두 제약 모두 통과 + 독립 NWP. 유일한 "진짜" 후보.
2. ICON (DWD) — 2022 결손 감수 시 보조 앙상블.
3. (그 외 전부 커버리지/리키지 탈락 또는 중복)

**권고**: ECMWF IFS HRES(D-2 12z 런)를 좌표 1점 백필로 확보해 LDAPS/GFS와 나란히 피처화하고, 우선 CV에서 순증 여부를 확인. 이때 반드시 발표묶음 단위 CV + D-2 12z init 고정으로 leakage 재현. 여기서 이득이 없으면 외부데이터로 갭을 좁힐 여지는 사실상 없다고 판단.

---

## 출처
- Open-Meteo Historical Forecast API (모델별 아카이브 시작일: IFS 2017-01-01, GFS 2021-03-23, ICON 2022-11-24): https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Previous/Single Runs API (init 고정 재현): https://open-meteo.com/en/docs/previous-runs-api
- Open-Meteo KMA API (GDPS 10km + LDPS 1.5km 구성): https://open-meteo.com/en/docs/kma-api
- Open-Meteo Licence (CC-BY-4.0, 원천 라이선스): https://open-meteo.com/en/licence
- ECMWF open data 전면개방(CC-BY-4.0, 2025-10): https://www.ecmwf.int/node/29328 · https://registry.opendata.aws/ecmwf-forecasts/
- ECMWF dissemination schedule (00z 05:45-07:34 UTC, 12z 17:45-19:34 UTC): https://confluence.ecmwf.int/display/DAC/Dissemination+schedule
- NOAA GEFS reforecast v12 (2000-2019, 0.25°): https://registry.opendata.aws/noaa-gefs-reforecast/ · https://psl.noaa.gov/forecasts/reforecast2/
- 기상청 API허브 단기예보(5km/3h, 2008~): https://apihub.kma.go.kr/apiList.do?seqApi=10
- KPX 공공데이터/종합자료실: https://www.kpx.or.kr/menu.es?mid=a10107020000
