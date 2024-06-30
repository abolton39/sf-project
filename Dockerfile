FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy --upgrade

COPY src src
COPY data data
COPY model.pkl model.pkl
COPY variables.pkl variables.pkl

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "1313"]
