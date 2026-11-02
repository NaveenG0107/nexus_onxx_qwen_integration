FROM python:3.12-slim

WORKDIR /app

# 1. Install lightweight PyTorch CPU wheel (only ~150 MB, ZERO NVIDIA CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install runtime dependencies
# Because torch is already installed, pip will NOT download any nvidia-* packages
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    python-dotenv \
    onnxruntime \
    "tokenizers>=0.19.0" \
    numpy \
    "transformers>=4.36,<4.58" \
    optimum-onnx \
    pandas \
    openpyxl

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
