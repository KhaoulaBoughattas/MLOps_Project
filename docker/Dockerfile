# Use a stable Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ api/
COPY mlruns/ mlruns/
COPY api/main.py .  

# Set environment variables
ENV MLRUNS_PATH=/app/mlruns
ENV PYTHONPATH=/app

# Expose FastAPI default port
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
