"""경로와 공통 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "vehicles.csv"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ROOT / "models"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlruns").as_uri()
EXPERIMENT = "used-vehicle-price"

SEED = 42
TEST_SIZE = 0.2        # 평가용으로 떼어 두는 매물 비율 (차량 단위)
CALIB_SIZE = 0.1       # 학습용 중 예측 구간 보정에 쓰는 비율
VALID_SIZE = 0.1       # 학습용 중 조기 종료·튜닝 검증에 쓰는 비율

# 데이터는 2021년 4~5월에 수집된 매물이다. 차령 = 기준 연도 - 연식 (2022년식은 0년)
REF_YEAR = 2022

# 정제 기준 (근거는 README와 리포트 1절)
PRICE_MIN = 1_000      # 0원, 1달러 같은 가격 미기재·자리표시 제외
PRICE_MAX = 150_000    # 1,234,567달러 같은 입력 오류와 초고가 수집차 제외
YEAR_MIN = 1995        # 클래식카는 시세 구조가 달라 범위에서 제외
YEAR_MAX = 2022
ODO_MAX = 400_000      # 1,111,111 같은 자리표시 제외
ODO_PLACEHOLDER = 100  # 2년 이상 된 차의 100마일 미만 주행거리는 미기재로 본다
MODEL_MIN_COUNT = 30   # 이보다 적게 나온 모델은 제조사별 '기타'로 묶는다

# 트럭 계열 (아이트럭 업무와 가장 가까운 세그먼트)
TRUCK_TYPES = {"truck", "pickup"}

SEGMENT_KO = {"truck": "트럭·픽업", "suv": "SUV", "sedan": "세단", "other": "기타 차종", "unknown": "차종 미기재"}
