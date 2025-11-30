# Simple Dockerfile for running the Linear Regression pipeline
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any) and copy project files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command runs the pipeline
CMD ["python", "app/main.py"]
