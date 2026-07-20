# BARAM 2026 — 풍력발전량 예측 AI 경진대회

프로젝트 규칙, 데이터 도메인 지식, 디렉토리 구조, 에이전트 팀 구성은 [CLAUDE.md](CLAUDE.md)를 참고.

## 빠른 시작

```bash
conda env create -f environment.yml
conda activate baram2026
```

원본 데이터는 `C:\Users\aica_\Desktop\open (1)\` (읽기 전용). 경로는 항상
[configs/paths.py](configs/paths.py)를 통해 참조한다.

## 참고 문서

- [docs/data_description.md](docs/data_description.md) — 원본 데이터 명세서 사본
- [docs/turbine_kpx_mapping.md](docs/turbine_kpx_mapping.md) — 터빈 ↔ KPX 그룹 매핑 정리
- `reports/eda/` — EDA 분석 결과
- `reports/domain_research/` — 대회 규정/도메인 리서치 결과
