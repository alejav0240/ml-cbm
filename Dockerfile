FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1
ENV ML_PRODUCTION_MODEL=/app/artifacts/production/model.json
EXPOSE 8010

CMD ["python", "service.py", "--host", "0.0.0.0", "--port", "8010"]
