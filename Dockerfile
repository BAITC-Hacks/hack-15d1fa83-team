FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY requirements-runtime.txt .
COPY src ./src
RUN pip install --no-cache-dir -r requirements-runtime.txt . && useradd --create-home appuser
COPY models/production /model
USER appuser
ENV MODEL_PATH=/model/model.json
EXPOSE 8000
CMD ["uvicorn", "windpower.api:app", "--host", "0.0.0.0", "--port", "8000"]
