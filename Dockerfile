FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch (much smaller than the default build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .
EXPOSE 5000

# 1 worker to save RAM; long timeout because each request trains the GNN
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "app:app"]
