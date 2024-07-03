FROM python:3.9-slim

WORKDIR /app

# Get system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

# Get all files from repo
COPY . .

# Get package dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for FastAPI and Streamlit
EXPOSE 1313 8501

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 1313 & streamlit run app.py"]
