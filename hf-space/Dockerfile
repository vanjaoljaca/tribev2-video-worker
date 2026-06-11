FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1

WORKDIR /app
COPY app.py /app/app.py
COPY start.py /app/start.py
COPY pilot-report.html /app/pilot-report.html
COPY pilot-results.zip /app/pilot-results.zip

EXPOSE 7860
CMD ["python", "-u", "/app/start.py"]
