# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy both requirements files
COPY requirements.txt gpu-requirements.txt ./

# Install everything in one go
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r gpu-requirements.txt

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the server
CMD ["uvicorn", "backend_api_main:app", "--host", "0.0.0.0", "--port", "8000"]