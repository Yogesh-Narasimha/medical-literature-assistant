"""Fine-tune a small embedding model on synthetic (question, passage) pairs.

Loss: in-batch-negatives contrastive loss (MultipleNegativesRankingLoss). For a batch of B pairs, each
question must score its own passage higher than the other B-1 passages (cosine similarity * scale,
cross-entropy over the batch). Only TRAIN-split documents are used; the test split is never touched.

Run:  python -m src.finetune.train
Out:  models/bge-small-medical/
"""
import json
import random
import sys
import time

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, ROOT

BASE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge's query instruction
OUT_DIR = ROOT / "models" / "bge-small-medical"
PAIRS = DATA_DIR / "finetune" / "pairs.jsonl"

EPOCHS, BATCH, LR, SCALE, MAX_LEN, SEED = 3, 32, 2e-5, 20.0, 256, 42


MIXED = len(sys.argv) > 1 and sys.argv[1] == "mixed"     # python -m src.finetune.train mixed
if MIXED:                                                  # long + short question styles
    OUT_DIR = OUT_DIR.with_name("bge-small-medical-mixed")
    PAIR_FILES = [PAIRS, PAIRS.with_name("pairs_short.jsonl")]
else:
    PAIR_FILES = [PAIRS]


def load_train_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for path in PAIR_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == "train":
                    pairs += [(QUERY_PREFIX + q, r["text"], r["chunk_id"]) for q in r["questions"]]
    return pairs


def batches(pairs, batch_size, rng):
    """Shuffle, and never put two questions about the same passage in one batch
    (they would be false negatives for each other)."""
    order = pairs[:]
    rng.shuffle(order)
    while order:
        batch, seen, rest = [], set(), []
        for p in order:
            if len(batch) < batch_size and p[2] not in seen:
                batch.append(p)
                seen.add(p[2])
            else:
                rest.append(p)
        yield batch
        order = rest


def embed(model, texts, device):
    feats = model.tokenize(texts)
    feats = {k: v.to(device) for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    model = SentenceTransformer(BASE_MODEL, device=device)
    model.max_seq_length = MAX_LEN
    pairs = load_train_pairs()
    print(f"{len(pairs)} training pairs from {len({p[2] for p in pairs})} passages", flush=True)

    steps_per_epoch = sum(1 for _ in batches(pairs, BATCH, random.Random(SEED)))
    total = steps_per_epoch * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / (0.1 * total), max(0.0, (total - s) / (0.9 * total))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    model.train()
    step, t0 = 0, time.time()
    for epoch in range(EPOCHS):
        losses = []
        for batch in batches(pairs, BATCH, rng):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                q = embed(model, [b[0] for b in batch], device)
                d = embed(model, [b[1] for b in batch], device)
                sims = q @ d.T * SCALE
                labels = torch.arange(len(batch), device=device)
                loss = (F.cross_entropy(sims, labels) + F.cross_entropy(sims.T, labels)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 20 == 0:
                print(f"  epoch {epoch + 1} step {step}/{total} loss {sum(losses[-20:]) / 20:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        print(f"epoch {epoch + 1} mean loss {sum(losses) / len(losses):.4f}", flush=True)

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_DIR))
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
