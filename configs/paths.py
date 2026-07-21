"""Central path configuration. Never hardcode data paths elsewhere.

원본 데이터는 읽기 전용이며 이 파일을 통해서만 참조한다.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 원본 데이터 (읽기 전용, 수정 금지) 위치는 머신마다 다르다 (예: 노트북은
# Desktop\open (1), 다른 PC는 프로젝트 안 datasets\ 등). 여러 머신을 오가며
# 작업하므로, 하드코딩된 단일 경로 대신:
#   1) BARAM_RAW_DATA_ROOT 환경변수가 있으면 그걸 그대로 사용,
#   2) 없으면 알려진 후보 경로들 중 실제로 sample_submission.csv가 있는
#      첫 번째 경로를 사용,
#   3) 그것도 못 찾으면(예: 새 머신에 데이터를 아직 안 옮긴 상태) 첫 번째
#      후보를 기본값으로 삼는다 -- 이후 실제 로딩 시점에 pandas가 명확한
#      FileNotFoundError를 내주므로 여기서 억지로 막지 않는다.
# 새 머신을 추가할 때는 아래 _KNOWN_RAW_DATA_ROOTS에 그 머신의 절대경로를
# 한 줄 추가하면 된다 (코드 수정은 그거면 끝).
_KNOWN_RAW_DATA_ROOTS = [
    Path(r"C:\Users\aica_\Desktop\open (1)"),
    Path(r"C:\Users\heelo\Desktop\claude_code\wind_power_predicet\datasets"),
    # in case the whole "open (1)" folder was copied as-is into datasets\
    Path(r"C:\Users\heelo\Desktop\claude_code\wind_power_predicet\datasets\open (1)"),
]


def _resolve_raw_data_root() -> Path:
    env_override = os.environ.get("BARAM_RAW_DATA_ROOT")
    if env_override:
        return Path(env_override)
    for candidate in _KNOWN_RAW_DATA_ROOTS:
        if (candidate / "sample_submission.csv").exists():
            return candidate
    return _KNOWN_RAW_DATA_ROOTS[0]


RAW_DATA_ROOT = _resolve_raw_data_root()
RAW_TRAIN_DIR = RAW_DATA_ROOT / "train"
RAW_TEST_DIR = RAW_DATA_ROOT / "test"

LDAPS_TRAIN_CSV = RAW_TRAIN_DIR / "ldaps_train.csv"
GFS_TRAIN_CSV = RAW_TRAIN_DIR / "gfs_train.csv"
TRAIN_LABELS_CSV = RAW_TRAIN_DIR / "train_labels.csv"
SCADA_VESTAS_TRAIN_CSV = RAW_TRAIN_DIR / "scada_vestas_train.csv"
SCADA_UNISON_TRAIN_CSV = RAW_TRAIN_DIR / "scada_unison_train.csv"

LDAPS_TEST_CSV = RAW_TEST_DIR / "ldaps_test.csv"
GFS_TEST_CSV = RAW_TEST_DIR / "gfs_test.csv"

SAMPLE_SUBMISSION_CSV = RAW_DATA_ROOT / "sample_submission.csv"
INFO_XLSX = RAW_DATA_ROOT / "info.xlsx"
DATA_DESCRIPTION_MD = RAW_DATA_ROOT / "data_description.md"

# 프로젝트 내부 산출물 (쓰기 가능)
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"

# KPX 그룹 설비용량 (kWh, 1시간 기준)
GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21_600,
    "kpx_group_2": 21_600,
    "kpx_group_3": 21_000,
}

# 터빈 -> KPX 그룹 매핑 (docs/turbine_kpx_mapping.md 참고)
VESTAS_GROUP_1 = [f"vestas_wtg{i:02d}" for i in range(1, 7)]
VESTAS_GROUP_2 = [f"vestas_wtg{i:02d}" for i in range(7, 13)]
UNISON_GROUP_3 = [f"unison_wtg{i:02d}" for i in range(1, 6)]

TURBINE_GROUP_MAP = {
    "kpx_group_1": VESTAS_GROUP_1,
    "kpx_group_2": VESTAS_GROUP_2,
    "kpx_group_3": UNISON_GROUP_3,
}

for _d in (DATA_INTERIM_DIR, DATA_PROCESSED_DIR, EXPERIMENTS_DIR, SUBMISSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
