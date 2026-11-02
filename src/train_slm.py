import os
import sys
import json
import math
import ast
import hashlib
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----------------------------------------------------
# 1. Config & Hyperparameters
# ----------------------------------------------------
VOCAB_SIZE = 2048       # Compact vocabulary
EMBED_DIM = 128         # Embedding dimension
NUM_HEADS = 4           # Multi-head attention heads
NUM_LAYERS = 4          # Number of transformer blocks
MAX_SEQ_LEN = 128       # Context window (supports multi-line DBML schemas)
BATCH_SIZE = 8
EPOCHS = 35
LEARNING_RATE = 1e-3
OUTPUT_DIR = "src/models/custom_slm"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------
# 2. Tokenizer Training
# ----------------------------------------------------
def build_and_train_tokenizer(data_path: str, save_path: str) -> Tokenizer:
    print("Training Byte-Level BPE Tokenizer...", flush=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tokenizer.train([data_path], trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}", flush=True)
    return tokenizer


# ----------------------------------------------------
# 3. Small Language Model (SLM) Architecture
# ----------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ----------------------------------------------------
# 4. Discrete Q&A Dataset (Each sample is prompt -> answer)
# ----------------------------------------------------
class QADataset(Dataset):
    def __init__(self, qa_pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_seq_len: int):
        self.samples = []
        pad_id = tokenizer.token_to_id("<pad>")

        # Balance class counts so minority Q&As (e.g. Mitrahsoft) aren't drowned out by 50 DBML tables
        from collections import Counter
        ans_counts = Counter(a for _, a in qa_pairs)
        max_count = max(ans_counts.values()) if ans_counts else 1

        balanced_pairs = []
        for q, a in qa_pairs:
            multiplier = min(8, max(1, max_count // max(1, ans_counts[a])))
            for _ in range(multiplier):
                balanced_pairs.append((q, a))

        for q, a in balanced_pairs:
            text = f"Prompt: {q}\nAnswer: {a}<eos>"
            enc = tokenizer.encode(text)
            ids = enc.ids
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            padded = ids + [pad_id] * (max_seq_len - len(ids))
            x = torch.tensor(padded[:-1], dtype=torch.long)
            y = torch.tensor(padded[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_single_file(file_path: str) -> list[tuple[str, str]]:
    qa_pairs = []
    if not os.path.exists(file_path):
        return qa_pairs

    ext = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if ext == ".jsonl":
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q = item.get("prompt") or item.get("question") or item.get("instruction")
                a = item.get("response") or item.get("answer") or item.get("output")
                if q and a:
                    qa_pairs.append((q.strip(), a.strip()))
        elif ext == ".json":
            data = json.loads(content)
            for item in data:
                q = item.get("prompt") or item.get("question")
                a = item.get("response") or item.get("answer")
                if q and a:
                    qa_pairs.append((q.strip(), a.strip()))
        else:
            # 1. Parse standard 'Prompt: ...\nAnswer: ...<eos>' format
            if "Prompt:" in content and "Answer:" in content:
                raw_blocks = content.split("Prompt:")
                for block in raw_blocks:
                    block = block.strip()
                    if not block or "Answer:" not in block:
                        continue
                    parts = block.split("Answer:", 1)
                    q = parts[0].strip()
                    a = parts[1].replace("<eos>", "").strip()
                    if q and a:
                        qa_pairs.append((q, a))
            else:
                # 2. Parse python tuple lines e.g. ("question", "answer")
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        try:
                            pair = ast.literal_eval(line)
                            if isinstance(pair, tuple) and len(pair) == 2:
                                qa_pairs.append((str(pair[0]).strip(), str(pair[1]).strip()))
                        except Exception:
                            continue
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}", flush=True)

    return qa_pairs


def load_qa_pairs(target_path: str) -> list[tuple[str, str]]:
    """
    Loads QA pairs from a single file or scans an entire folder for all
    .txt, .json, and .jsonl files (ignoring temp/cache files).
    """
    qa_pairs = []
    if not os.path.exists(target_path):
        return qa_pairs

    if os.path.isdir(target_path):
        # Scan all dataset files in directory
        all_files = sorted(os.listdir(target_path))
        dataset_files = [
            f for f in all_files
            if f.endswith((".txt", ".jsonl", ".json"))
            and not f.startswith((".", "_"))
        ]
        print(f"Found {len(dataset_files)} dataset file(s) in '{target_path}': {dataset_files}", flush=True)
        for f in dataset_files:
            file_full = os.path.join(target_path, f)
            file_pairs = load_single_file(file_full)
            print(f"  -> {f}: Loaded {len(file_pairs)} pairs", flush=True)
            qa_pairs.extend(file_pairs)
    else:
        qa_pairs = load_single_file(target_path)

    # Deduplicate while preserving order
    seen = set()
    unique_pairs = []
    for q, a in qa_pairs:
        key = (q.strip(), a.strip())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)

    return unique_pairs


# ----------------------------------------------------
# 5. Training & Export Execution
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune Custom SLM across dataset folder or files")
    parser.add_argument("--data", type=str, default="data", help="Path to data folder (e.g. 'data') or single dataset file")
    parser.add_argument("--fine-tune", action="store_true", help="Load existing checkpoint.pt and fine-tune with lower learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 30 for scratch, 10 for fine-tune)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Base learning rate")
    args = parser.parse_args()

    # Determine default epochs based on mode
    if args.epochs is None:
        epochs = 10 if args.fine_tune else 30
    else:
        epochs = args.epochs

    cache_dir = "data/.cache"
    os.makedirs(cache_dir, exist_ok=True)
    temp_train_file = os.path.join(cache_dir, "tokenizer_train.txt")

    # 1. Load QA pairs across folder or file
    qa_pairs = load_qa_pairs(args.data)
    if not qa_pairs:
        print(f"Warning: No pairs found in {args.data}. Using fallback pairs.", flush=True)
        qa_pairs = [
            ("Where is Mitrahsoft second branch located?", "Mitrahsoft second branch is located in madurai."),
            ("Where is Mitrahsoft located?", "Mitrahsoft has its second branch located in madurai."),
            ("What is Mitrahsoft?", "Mitrahsoft is an IT services and software development company."),
            ("What is artificial intelligence?", "Artificial intelligence is intelligence demonstrated by machines."),
            ("What is machine learning?", "Machine learning algorithms build a model based on sample data."),
            ("What is deep learning?", "Deep learning is part of a broader family of machine learning methods."),
            ("What is natural language processing?", "Natural language processing helps computers understand human text."),
        ]
    print(f"Total active training samples: {len(qa_pairs)} unique QA pairs.", flush=True)

    # Save known topics metadata for the inference guardrail
    topics_file = os.path.join(OUTPUT_DIR, "topics.json")
    with open(topics_file, "w", encoding="utf-8") as f:
        json.dump([q for q, _ in qa_pairs], f, indent=2)
    print(f"Known topics saved to {topics_file} ({len(qa_pairs)} questions registered)", flush=True)

    # Build raw training text stream for tokenizer
    formatted_qa = [f"Prompt: {q}\nAnswer: {a}<eos>\n" for q, a in qa_pairs]
    with open(temp_train_file, "w", encoding="utf-8") as f:
        f.write("".join(formatted_qa) * 20)

    # 2. Build or Load Tokenizer
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    if args.fine_tune and os.path.exists(tokenizer_path):
        print(f"Fine-tuning mode: Loading existing tokenizer from {tokenizer_path}...", flush=True)
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = build_and_train_tokenizer(temp_train_file, tokenizer_path)

    # 3. Build Dataset & Dataloader
    pad_id = tokenizer.token_to_id("<pad>")
    dataset = QADataset(qa_pairs, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Initialize or Load Model
    model = SmallLanguageModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(device)
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")

    def get_model_stats(m: nn.Module):
        total_p = sum(p.numel() for p in m.parameters())
        total_layers = len(list(m.named_parameters()))
        l2_norm = math.sqrt(sum(p.norm().item()**2 for p in m.parameters()))
        hasher = hashlib.sha256()
        for p in m.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        fingerprint = hasher.hexdigest()[:12]
        return total_p, total_layers, l2_norm, fingerprint

    effective_lr = args.lr
    is_fine_tune = bool(args.fine_tune and os.path.exists(checkpoint_path))

    if is_fine_tune:
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        saved_state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(saved_state)
        effective_lr = args.lr * 0.1  # Reduce learning rate for fine-tuning

        total_p, total_layers, initial_norm, initial_hash = get_model_stats(model)

        print("\n" + "=" * 70, flush=True)
        print("[MODE] FINE-TUNING ON EXISTING WEIGHTS", flush=True)
        print(f"  * Checkpoint File       : {checkpoint_path} ({ckpt_size_mb:.2f} MB)", flush=True)
        print(f"  * Active Parameters     : {total_p:,} across {total_layers} parameter tensors", flush=True)
        print(f"  * Base Weight Hash      : SHA256:{initial_hash}", flush=True)
        print(f"  * Base Weight L2 Norm   : {initial_norm:.4f}", flush=True)
        print(f"  * Learning Rate         : {effective_lr} (Scaled for preservation)", flush=True)
        print("=" * 70 + "\n", flush=True)
    else:
        total_p, total_layers, initial_norm, initial_hash = get_model_stats(model)

        print("\n" + "=" * 70, flush=True)
        print("[MODE] TRAINING FROM SCRATCH", flush=True)
        print(f"  * Weight Initialization : Fresh random Gaussian/uniform weights", flush=True)
        print(f"  * Active Parameters     : {total_p:,} across {total_layers} parameter tensors", flush=True)
        print(f"  * Initial Weight Hash   : SHA256:{initial_hash}", flush=True)
        print(f"  * Initial Weight L2 Norm: {initial_norm:.4f}", flush=True)
        print(f"  * Learning Rate         : {effective_lr}", flush=True)
        print("=" * 70 + "\n", flush=True)

    # Snapshot initial parameters tensor-by-tensor for exact delta auditing
    initial_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    print(f"Training on {len(dataset)} balanced samples for {epochs} epochs...", flush=True)
    model.train()
    first_loss = None
    last_loss = None

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        if first_loss is None:
            first_loss = avg_loss
        last_loss = avg_loss
        print(f"Epoch {epoch:2d}/{epochs} - Loss: {avg_loss:.4f}", flush=True)

    # Dynamically compute post-training weights metrics
    total_p, total_layers, final_norm, final_hash = get_model_stats(model)
    delta_norm = math.sqrt(sum((p.detach() - initial_params[n]).norm().item()**2 for n, p in model.named_parameters()))
    percent_shift = (delta_norm / initial_norm) * 100 if initial_norm > 0 else 0.0

    # Dynamically evaluate the model state and status
    if delta_norm == 0:
        audit_status = "UNCHANGED: No gradient updates occurred."
    elif percent_shift < 25.0:
        audit_status = f"FINE-TUNED: Preserved prior knowledge with a targeted {percent_shift:.2f}% adaptation."
    else:
        audit_status = f"TRAINED FROM SCRATCH: Underwent full {percent_shift:.2f}% structural transformation."

    print("\n" + "=" * 70, flush=True)
    print("[DYNAMIC WEIGHT VERIFICATION AUDIT]", flush=True)
    print(f"  * Pre-Training Hash   : SHA256:{initial_hash}", flush=True)
    print(f"  * Post-Training Hash  : SHA256:{final_hash}", flush=True)
    print(f"  * Initial Weight Norm : {initial_norm:.4f}", flush=True)
    print(f"  * Final Weight Norm   : {final_norm:.4f}", flush=True)
    print(f"  * Weight Delta (Norm) : {delta_norm:.4f}", flush=True)
    print(f"  * Relative Shift      : {percent_shift:.2f}%", flush=True)
    print(f"  * Loss Trajectory     : {first_loss:.4f} -> {last_loss:.4f} (Drop: {first_loss - last_loss:.4f})", flush=True)
    print(f"  * Audit Status        : {audit_status}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # 5. Save PyTorch Checkpoint (for future fine-tuning)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"PyTorch checkpoint saved to: {checkpoint_path}", flush=True)

    # 6. Export to ONNX
    print("Exporting trained SLM to ONNX...", flush=True)
    model.eval()
    model.cpu()

    dummy_input = torch.randint(0, VOCAB_SIZE, (1, 16), dtype=torch.long)
    onnx_file_path = os.path.join(OUTPUT_DIR, "model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"Success! Model exported to: {onnx_file_path}", flush=True)
    print(f"Tokenizer exported to: {tokenizer_path}", flush=True)


if __name__ == "__main__":
    main()
