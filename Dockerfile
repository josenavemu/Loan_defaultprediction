FROM python:3.14.3-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY api-requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r api-requirements.txt

RUN mkdir -p /app/models

COPY api/ ./api/

COPY models/best_model.pkl ./models/
COPY models/imputer.pkl ./models/
COPY models/scaler.pkl ./models/
COPY models/ohe.pkl ./models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
