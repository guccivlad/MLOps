FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip build && \
    python3 -m build -v && \
    pip install dist/*.whl

CMD ["python3", "python/gauss.py"]