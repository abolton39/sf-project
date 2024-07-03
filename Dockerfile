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

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "1313"]
