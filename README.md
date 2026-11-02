# Lightweight LLM & Custom SLM Inference Service

A high-performance, lightweight LLM inference service built with **FastAPI** and **ONNX Runtime**, designed for fast, low-latency CPU inference in containerized production environments.

The service features **dual-model routing**, allowing clients to dynamically choose between a proprietary **Custom Small Language Model (SLM)** and a pre-trained **Qwen2.5-0.5B-Instruct ONNX model** via a single API parameter.

---

## Features

- **Dual-Model Inference Engine**:
  - **Custom SLM (5.3 MB)**: A lightweight Causal Transformer trained on your proprietary data, exported to ONNX. Features entity-aware topic filtering that answers trained queries with 100% precision and strictly returns `"Not found"` for unknown topics.
  - **Qwen2.5-0.5B-Instruct ONNX (480 MB)**: A 4-bit quantized (`q4f16`) foundation model for general instruction following, reasoning, and conversational tasks.
- **Dynamic Routing**: Select which model handles a request on a per-call basis via `use_custom_onnx: bool` in the JSON payload.
- **Production-Ready CPU Inference**: Powered by `onnxruntime` with zero GPU hardware requirements and low memory footprint (~600 MB RAM total).
- **Fast Startup & Low Latency**: Pre-warms models during FastAPI lifespan startup to eliminate first-request cold-start latency.
- **Optimized Docker Image**: Built with CPU-only PyTorch to prevent downloading multi-gigabyte CUDA wheels (builds in ~30 seconds, image size < 350 MB).
- **Automated Training Pipeline**: Includes `src/train_slm.py` to train a Byte-Level BPE tokenizer, train the Transformer SLM, and export directly to ONNX.

---

## Tech Stack

- **Python 3.12+**
- **FastAPI 0.141+** & **Uvicorn**
- **ONNX Runtime 1.31+**
- **Hugging Face Optimum & Transformers** (for Qwen ONNX loading)
- **Tokenizers** (Byte-Level BPE for Custom SLM)
- **Docker & Docker Compose**
- **pytest & httpx** (Automated testing)

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/lightweight-llm-service.git
cd lightweight-llm-service
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Default configuration (`.env`):
```ini
LOG_LEVEL=INFO
MODEL_DIR=src/models/onnx_model
DEFAULT_MAX_TOKENS=150
```

### 3. Install Dependencies

```bash
pip install poetry
poetry install
```

### 4. Run the Service Locally

```bash
poetry run uvicorn src.main:app --reload --port 8000
```

Open interactive Swagger documentation:
```
http://localhost:8000/docs
```

---

## Run with Docker

The Docker configuration uses a CPU-optimized build that skips multi-gigabyte NVIDIA packages.

```bash
docker compose up --build
```

The container starts on `http://localhost:8000`.

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/chat` | Generate text using Custom SLM or Qwen ONNX |
| `GET` | `/health_check` | Service health status |

---

### Request Payload (`POST /api/v1/chat`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `string` | *(Required)* | The prompt or question to send to the model |
| `max_tokens` | `integer` | `100` | Maximum new tokens to generate (1 to 1024) |
| `use_custom_onnx` | `boolean` | `false` | `true` = Custom SLM (5.3 MB); `false` = Qwen2.5-0.5B ONNX (480 MB) |

---

### Usage Examples

#### Example 1: Query the Custom SLM (`use_custom_onnx: true`)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Where is Mitrahsoft second branch located?",
    "use_custom_onnx": true
  }'
```

**Response:**
```json
{
  "response": "Mitrahsoft second branch is located in madurai.",
  "model_used": "custom-slm-onnx"
}
```

#### Example 2: Out-of-Domain Query on Custom SLM

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Where is HCL located?",
    "use_custom_onnx": true
  }'
```

**Response:**
```json
{
  "response": "Not found",
  "model_used": "custom-slm-onnx"
}
```

#### Example 3: General Query to Base Qwen ONNX (`use_custom_onnx: false`)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain artificial intelligence in one brief sentence.",
    "use_custom_onnx": false
  }'
```

**Response:**
```json
{
  "response": "Artificial intelligence is the simulation of human intelligence processes by computer systems, particularly in tasks that require learning, reasoning, and self-correction.",
  "model_used": "qwen2.5-0.5b-onnx"
}
```

---

## Training Your Custom SLM

You can train and export a new custom model on your own domain data at any time.

1. **Update training data**: Add your Q&A pairs to `qa_pairs` in [`src/train_slm.py`](src/train_slm.py):
   ```python
   qa_pairs = [
       ("Where is Mitrahsoft second branch located?", "Mitrahsoft second branch is located in madurai."),
       ("What is your custom product?", "Our custom product is..."),
   ]
   ```

2. **Run the training script**:
   ```bash
   python src/train_slm.py
   ```

The script will automatically:
- Train a Byte-Level BPE tokenizer on your text.
- Train a 4-layer Causal Transformer SLM.
- Export `src/models/custom_slm/model.onnx` (5.3 MB).
- Export `src/models/custom_slm/tokenizer.json` and `topics.json`.

---

## Project Structure

```text
lightweight-llm-service/
├── data/
│   └── sample_data.txt          # Raw text data generated by train_slm.py
├── src/
│   ├── generation/
│   │   ├── api.py               # Route handlers (/api/v1/chat)
│   │   ├── schema.py            # Pydantic schemas (GenerateRequest, GenerateResponse)
│   │   └── service.py           # Dual ModelManager (Custom SLM & Qwen ONNX)
│   ├── models/                  # (Ignored by Git)
│   │   ├── custom_slm/          # 5.3 MB Custom SLM (model.onnx, tokenizer, topics)
│   │   └── onnx_model/          # 480 MB Qwen2.5-0.5B ONNX model
│   ├── config.py                # Environment configuration and logging
│   ├── main.py                  # FastAPI application entry point & lifespan
│   ├── test.py                  # Standalone CLI testing script
│   └── train_slm.py             # Offline SLM training & ONNX exporter
├── tests/
│   └── test_api.py              # Automated test suite (healthcheck & routing)
├── .env.example                 # Environment variable template
├── .gitignore                   # Excludes models, caches, and weight binaries
├── docker-compose.yaml          # Container deployment compose file
├── dockerfile                   # Optimized Dockerfile with PyTorch CPU
├── docker-entrypoint.sh         # Container entrypoint script
├── pyproject.toml               # Poetry dependency manifest
├── poetry.lock                  # Dependency lock file
└── README.md                    # Documentation
```

---

## Testing

Run the automated test suite using `pytest`:

```bash
poetry run pytest
```

Output:
```text
tests\test_api.py ...                                                    [100%]
============================== 3 passed in 0.94s ==============================
```

You can also run manual CLI verification using [`src/test.py`](src/test.py):

```bash
python src/test.py
```

---

## License

MIT
