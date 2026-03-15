FROM python:3.12-slim-bookworm
RUN pip install fastapi==0.115.0 uvicorn==0.32.0 scikit-learn==1.5.1 pandas==2.2.2 numpy==1.26.4 joblib==1.4.2 prometheus_client==0.20.0
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
