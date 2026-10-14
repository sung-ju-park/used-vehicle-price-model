`python src/run_all.py`를 실행하면 여기에 서빙용 모델이 만들어집니다 (용량이 커서 저장소에는 포함하지 않음).

- `price_model.ubj`: 시세 예측 XGBoost 모델
- `quantile_model.ubj`: 80% 시세 구간용 분위수 모델
- `meta.json`: 범주 목록, 모델명 목록, 구간 보정값(CQR)
