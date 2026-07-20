# 터빈 파워커브 스펙 — VESTAS V126 / UNISON U136

`feature-engineer`가 파워커브 형태(비선형 풍속→발전량 매핑, cut-in/rated/cut-out 기반)의
풍속 변환 피처를 만들 때 사용할 실제 제조사 스펙 조사 결과. 두 모델 모두 컷인/정격/컷아웃
풍속 3개 값은 제조사 공식 자료로 확인됨. 다만 "풍속별 출력(kW) 전체 테이블/차트"는 두 모델
모두 프로젝트가 쓰는 정확한 정격출력(3.6MW, 4.2MW) 버전으로는 공개된 곳을 찾지 못했다
(근접 변형 모델의 공식 차트는 확보). 아래에서 "제조사 원본 확인"과 "동일 플랫폼 근접 모델
기반 추정/미확인"을 명확히 구분했다.

## 요약 표

| 항목 | VESTAS V126 (프로젝트: 3.6MW, hub 117m, 로터 126m) | UNISON U136 (프로젝트: 4.2MW, hub 117m, 로터 136m) |
|---|---|---|
| Cut-in 풍속 | **3 m/s** (제조사 확인, 3.45MW/3.0MW 등 V126 全 변형 동일) | **3 m/s** (제조사 공식 페이지 확인) |
| Rated 풍속 | **미확인** (3.6MW 정확한 값은 공개 안 됨). 근접 참고값: V126-3.0MW 12 m/s(제조사 원본), V126-3.45MW 11.5 m/s(2차 출처) | **11.3 m/s** (제조사 공식 페이지 확인) |
| Cut-out 풍속 | **22.5 m/s** (제조사 확인, V126 全 변형 동일) | **22 m/s** (제조사 공식 페이지의 "운전풍속 3~22m/s"에서 도출) |
| Rated 출력 | **3.6 MW = 3,600 kW** — Vestas 공식 브로슈어에 "V126-3.45 MW의 Power Optimised Mode, 최대 3.6 MW (site specific)"로 명시 확인. 별도 "V126-3.6MW" 독립 모델 데이터시트는 못 찾음 | **4.2 MW = 4,200 kW** (제조사 공식 페이지 확인, 프로젝트 값과 정확히 일치) |

## 1. VESTAS V126 (3.6 MW)

### 확인된 사실 — Vestas 공식 자료

Vestas 공식 홈페이지에서 내려받은 "4 MW platform" 브로슈어
(`https://www.vestas.com/content/dam/vestas-com/global/en/brochures/onshore/4MW_Platform_Brochure.pdf`,
문서 하단 표기 "03/2024-EN", 즉 2024년 3월판 공식 최신 브로슈어) 7페이지 "V126-3.45 MW® IEC
IIB/IEC IIA — Facts & figures"에 다음이 명시되어 있음(원문 그대로):

- Rated power: **3,450 kW**
- Cut-in wind speed: **3 m/s**
- Cut-out wind speed: **22.5 m/s**
- Re cut-in wind speed: 20 m/s
- Wind class: IEC IIB/IEC IIA
- Rotor diameter: 126 m / Swept area 12,469 m²
- Hub heights: **87m / 117m / 137m / 147m / 149m / 166m** ← 프로젝트 스펙(hub 117m)과 정확히 일치
- **Turbine options** 목록 중: **"Power Optimised Mode up to 3.6 MW (site specific)"**

즉 이 브로슈어가 **프로젝트에서 쓰는 "V126, 3.6MW, hub 117m" 조합의 근거를 직접 설명한다**:
V126은 기본형이 3.45MW이고, 사이트별로 컨버터/발전기를 소프트웨어적으로 "Power Optimised
Mode"로 3.6MW까지 끌어올려 운용하는 것이 Vestas가 공식 제공하는 옵션이다. 태백가덕산 1단계
(코오롱글로벌, 2020년 9월 가동, V126 12기 = 43.2MW = 3.6MW×12)가 바로 이 조합과 정확히
일치한다 (뉴스 검색 결과, 아래 출처 참고). 다만 **"3.6MW 모드"만 따로 뽑은 독립 데이터시트
(정격풍속·파워커브 차트 포함)는 Vestas 공식 사이트에서 찾지 못했다** — 3.45MW 기본형 스펙만
공개되어 있고, 3.6MW는 "site specific" 옵션이라 표준 브로슈어에 곡선이 없는 것으로 보인다.

**Cut-in(3 m/s)·Cut-out(22.5 m/s)은 V126 로터/공력설계가 동일한 모든 정격출력 변형에서
일관되게 동일하게 확인됨** (아래 3개 변형 모두 제조사 자료로 개별 확인):

| V126 변형 | Rated power | Cut-in | Cut-out | Rated 풍속 | 출처 |
|---|---:|---:|---:|---:|---|
| V126-3.0 MW | 3,000 kW | 3 m/s | 22.5 m/s | **12 m/s** | Vestas 원본 "Facts & figures" PDF (2012년 작성, 이탈리아 환경부 환경영향평가 아카이브에 사본 존재) |
| V126-3.3 MW | 3,300 kW | 3 m/s | 22.5 m/s | 미확인 | wind-turbine-models.com (2차 출처) |
| V126-3.45 MW | 3,450 kW | 3 m/s | 22.5 m/s | 11.5 m/s (2차 출처만) | Vestas 공식 4MW platform 브로슈어(2024) + thewindpower.net |
| **V126-3.6 MW (site-specific 모드)** | **3,600 kW** | **3 m/s로 추정(확인 안 됨)** | **22.5 m/s로 추정(확인 안 됨)** | **미확인** | Vestas 브로슈어에 옵션으로만 언급, 별도 스펙시트 없음 |

같은 로터·나셀·기어박스를 공유하는 플랫폼이라 3.6MW 모드도 cut-in/cut-out이 3.45MW와
동일할 가능성이 높지만, **이는 추정이며 Vestas가 3.6MW 전용으로 명시한 값은 아니다.**
Rated 풍속은 출력이 클수록(같은 로터에서 더 많은 전력을 뽑아내므로) 약간 더 높은 풍속에서
정격에 도달할 것으로 예상되나(3.0MW: 12 m/s → 3.45MW: ~11.5 m/s처럼 오히려 낮아지는 경향도
보여 단순 외삽이 어려움), **feature-engineer가 3.6MW 전용 rated 풍속 숫자를 임의로 만들어
쓰면 안 되고, 3.45MW의 근사값(11~12 m/s 범위) 또는 3.0MW 확인값(12 m/s)을 "동일 플랫폼
근사치"로 명시하고 사용해야 한다.**

### 파워커브 형태 (V126-3.0MW 공식 차트 기반, 참고용)

프로젝트가 쓰는 3.6MW 곡선 자체는 아니지만, **동일 로터(126m)의 Vestas 공식 파워커브 차트
(V126-3.0MW, 2012 Facts & Figures PDF)에서 읽은 대략적인 곡선 형태**는 다음과 같다(차트에서
육안으로 읽은 근사치이며 표 형태의 정밀 수치가 아님에 주의):

| 풍속 (m/s) | 출력 (kW, 근사) | 비고 |
|---:|---:|---|
| 3 | 0 | cut-in |
| 5 | ~250–350 | |
| 7 | ~800–1,000 | |
| 9 | ~1,800–2,000 | 정격(3,000kW)의 절반 부근 |
| 12 | 3,000 (정격) | rated |
| 12–22.5 | 3,000 (평탄) | rated~cut-out 사이 정격 유지 |
| 22.5 | 0 | cut-out |

전형적인 pitch-regulated 풍력터빈의 S자형 곡선(cut-in 이후 완만히 증가 → 정격 부근에서
가파르게 상승 → 정격풍속부터 cut-out까지 평탄)이며, 3.6MW 버전도 로터·공력설계가 동일하므로
곡선의 "모양"(S자, 변곡점 위치 비율)은 유사할 것으로 보이나 세로축 스케일(정격출력)과 정격
풍속 도달 지점만 다를 것으로 추정된다.

## 2. UNISON U136 (4.2 MW)

### 확인된 사실 — Unison 공식 홈페이지

Unison 공식 제품 페이지 `https://www.unison.co.kr/product/4MW_Platform_U136`에 명시된
값(한글 원문 라벨 그대로):

- 정격출력(Rated power): **4,200 kW = 4.2 MW** ← 프로젝트 값과 정확히 일치
- 로터직경(Rotor diameter): 136 m
- 허브높이(Hub height): **95 m, 117 m** ← 프로젝트 값(117m)과 정확히 일치
- 운전풍속(Operating wind speed): **3 ~ 22 m/s**
- 컷인풍속(Cut-in): **3 m/s**
- 정격풍속(Rated wind speed): **11.3 m/s**
- 한계풍속(Survival/extreme wind speed, IEC 3초 돌풍 극한내풍속): 70 m/s — **이것은 cut-out이
  아니라 별도의 구조 설계 극한풍속**이므로 혼동하지 않도록 주의. 실제 운전상 cut-out은
  "운전풍속 3~22m/s" 범위의 상한인 **22 m/s**로 판단됨.
- 설계등급(IEC Design Class): IEC IA
- 설계수명: 20년

2차 출처(wind-turbine-models.com 검색 스니펫)도 cut-in 3 m/s, cut-out 22 m/s로 동일하게
확인되어 정합성 있음.

### 파워커브 형태 — 상세 테이블/차트는 공개된 곳을 찾지 못함

Unison 공식 사이트, 2025 대한민국 에너지대전 제품 페이지, 코머신(komachine.com) 제조사
소개, thewindpower.net, wind-turbine-models.com을 모두 확인했으나 **"풍속별 출력(kW) 테이블
또는 차트" 원본은 어디에서도 찾지 못했다.** wind-turbine-models.com은 U136에 대해 "파워
데이터가 시스템에 없다(Power data ... are not stored)"고 명시. 따라서 U136의 세부 파워커브
포인트는 **미확인**이며, feature-engineer가 필요하다면:

1. cut-in(3 m/s)/rated(11.3 m/s)/cut-out(22 m/s) 3개 확정값으로 일반적인 파워커브 함수형태
   (예: cut-in~rated 구간 3차 다항식 또는 sigmoid 근사, rated~cut-out 구간 상수 4,200kW,
   그 외 구간 0)를 직접 구성하거나,
2. 프로젝트 내부 SCADA 데이터(`scada_unison_train.csv`)의 실측 풍속-출력 관계를 경험적으로
   피팅하는 방식(CLAUDE.md에 이미 "SCADA는 파워커브 등 물리 기반 피처의 파라미터 추정에
   활용 가능"이라 명시됨)을 권장한다 — 이쪽이 실제 국내 설치 조건(공기밀도, 웨이크 등)까지
   반영되어 공개 데이터시트보다 더 정확할 수 있다.

## 3. 프로젝트 맥락과의 교차검증 (참고)

뉴스 검색 결과(코오롱글로벌 태백가덕산 관련 기사, 2차 출처)에 따르면 태백가덕산 풍력단지는
1단계 2020년 9월 가동 **베스타스 3.6MW급 12기(43.2MW)**, 2단계 2022년 12월 완공
**4.2MW급 5기(21MW, 유니슨)**로 구성되어, `docs/turbine_kpx_mapping.md`의 VESTAS
V126×12(kpx_group_1/2, 3.6MW×12=21.6MW×2) / UNISON U136×5(kpx_group_3, 4.2MW×5=21MW)
구성과 정확히 부합한다. 이는 두 모델 식별(V126=3.6MW 사이트, U136=4.2MW 사이트)이 맞다는
간접 확증이지만, 이 자체가 정격풍속 등 세부 파워커브 수치를 제공하지는 않는다.

## 출처

- Vestas 공식 4 MW platform 브로슈어(2024년 3월판): `https://www.vestas.com/content/dam/vestas-com/global/en/brochures/onshore/4MW_Platform_Brochure.pdf.coredownload.inline.pdf`
  (V126-3.45 MW Facts & figures 페이지: rated power, cut-in/cut-out/re-cut-in 풍속, hub
  heights, "Power Optimised Mode up to 3.6 MW (site specific)" 옵션 문구 확인)
- Vestas V126-3.0 MW "Facts & figures" 원본 데이터시트(2012년 Vestas 작성, PDF 메타데이터
  "Vestas_2012_3 MW Turbines"), 이탈리아 환경부(MITE) 환경영향평가 문서 아카이브 미러:
  `https://va.mite.gov.it/File/Documento/394411` (rated power, cut-in/rated/cut-out/re
  cut-in 풍속, hub height, 파워커브 차트(이미지) 확인)
- Vestas V126-3.45 MW 공식 제품 페이지: `https://www.vestas.com/en/energy-solutions/onshore-wind-turbines/4-mw-platform/V126-3-45-MW`
  (cut-in 3m/s, cut-out 22.5m/s 재확인)
- Unison 공식 제품 페이지(U136): `https://www.unison.co.kr/product/4MW_Platform_U136`
  (정격출력 4,200kW, 로터직경 136m, 허브높이 95/117m, 운전풍속 3~22m/s, 컷인 3m/s, 정격풍속
  11.3m/s, 한계풍속 70m/s, IEC IA 확인)
- wind-turbine-models.com (2차 출처, 검색 스니펫으로만 확인, 원문 페이지는 403으로 직접 접근
  불가): V126 계열 cut-in/cut-out 교차검증, U136 cut-in 3m/s·cut-out 22m/s 교차검증,
  "U136 파워 데이터 시스템에 없음" 명시
- thewindpower.net (2차 출처, 검색 스니펫): V126/3450 rated wind speed 11.5 m/s (원문 페이지
  자체는 유료 구독 필요로 직접 열람 불가, 검색 엔진 스니펫으로만 확인된 값이라 신뢰도 보통)
- 코오롱글로벌 태백가덕산 풍력 관련 뉴스 기사(2차 출처, 프로젝트 구성 교차검증용):
  전기신문 `https://www.electimes.com/news/articleView.html?idxno=361590`,
  국토일보 `https://www.ikld.kr/news/articleView.html?idxno=236508` 등 검색 결과 종합
  (1단계 베스타스 3.6MW×12기, 2단계 유니슨 4.2MW×5기 시기·용량 일치 확인)
- Unison U136 개발/인증 이력(2차 출처, 검색 결과 종합): 2018.10 시제품 설치(정암풍력,
  4.3MW 용량), 2019.05 독일 DEWI-OCC 국제인증, 2019.08 국내 KS인증

## 미확인 — 추가 조사 필요 시

- **V126-3.6MW 전용 정격풍속(rated wind speed)과 세부 파워커브 수치**: Vestas가 "site
  specific Power Optimised Mode"로만 제공하고 표준 데이터시트를 공개하지 않아 확인 불가.
  근사치로 3.45MW(~11.5 m/s, 2차 출처) 또는 3.0MW(12 m/s, 제조사 원본)를 참고할 수 있으나
  3.6MW 고유값은 아님.
- **UNISON U136의 풍속별 출력(kW) 상세 테이블/차트**: 제조사 공식 자료 어디에도 공개된 곳을
  찾지 못함. cut-in/rated/cut-out 3개 값만 확정.
- 두 모델 모두 **공기밀도 보정 기준(표준 1.225 kg/m³ 대비 실제 사이트 고도/기온에 따른 보정
  방식)은 이번 조사 범위에 포함하지 않음** — 필요 시 별도 조사 요청 필요.
