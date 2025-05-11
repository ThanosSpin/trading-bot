# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Streamlit will run on port 8080 for Cloud Run
ENV PORT 8080
EXPOSE 8080

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.enableCORS=false"]