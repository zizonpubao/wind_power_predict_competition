"""Central path configuration. Never hardcode data paths elsewhere.

원본 데이터는 읽기 전용이며 이 파일을 통해서만 참조한다.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 원본 데이터 (읽기 전용, 수정 금지)
RAW_DATA_ROOT = Path(r"C:\Users\aica_\Desktop\open (1)")
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
