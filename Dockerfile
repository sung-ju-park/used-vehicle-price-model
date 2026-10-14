# 시세 예측 API 컨테이너. 먼저 로컬에서 python src/run_all.py로 models/ 폴더를 만든 뒤 빌드한다.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy xgboost fastapi uvicorn
COPY src/clean.py src/config.py src/features.py ./src/
COPY serve/ ./serve/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
