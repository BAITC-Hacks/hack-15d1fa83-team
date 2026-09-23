FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml requirements.txt requirements-runtime.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home appuser
COPY manage.py ./
COPY config ./config
COPY common ./common
COPY forecasting ./forecasting
COPY weather_archive ./weather_archive
COPY templates ./templates
COPY static ./static
COPY inference ./inference
COPY models/production ./models/production
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data
USER appuser
ENV MODEL_PATH=/app/models/production/model.json
EXPOSE 8000
CMD ["python", "inference/serve.py", "--host", "0.0.0.0", "--port", "8000"]
