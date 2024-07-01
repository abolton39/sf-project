FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir numpy
RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY data data
COPY model.pkl model.pkl
COPY variables.pkl variables.pkl

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "1313"]
