"""
Experiment B: Best-of-N sweep on Qwen2.5-1.5B-Instruct.

Sweeps N in {1, 2, 4, 8, 16} on the same 30 GSM8K problems used in
huginn_sweep.py. For each problem we draw max(N_VALUES) independent samples
at temperature 0.7 in a single batched generate() call, then take majority
vote over the first N of those samples for each budget -- this reproduces
best-of-N for every N without regenerating from scratch at each budget
(16 generations per problem instead of 1+2+4+8+16=31).

Usage:
    python bon_sweep.py --smoke-test      # 2 problems, N in {1,2}
    python bon_sweep.py                   # full sweep (30 problems x 16 samples = 480 generations)
    python bon_sweep.py --device cpu      # run on CPU (model is 1.5B, CPU-feasible)
    python bon_sweep.py --resume          # skip problems already in the output file

Requires: torch, transformers, datasets (already installed for huginn_sweep.py)
"""

import os

# Fixes a ~64x CPU inference slowdown on AMD CPUs (Intel MKL's default CPU
# dispatcher takes a much slower path on non-Intel chips). Must be set before
# torch is imported. See backend/server.py for the benchmark that found this.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

import argparse
import json
import time
from collections import Counter
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
N_VALUES = [1, 2, 4, 8, 16]
MAX_N = max(N_VALUES)
N_GSM8K = 30
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
OUTPUT_PATH = Path(__file__).parent / "data" / "bon_sweep_raw.json"

SYSTEM_PROMPT = "You are a helpful assistant that solves problems step by step."
GSM8K_SUFFIX = (
    "\n\nSolve this step by step. On the final line, write your answer as:\n"
    "The answer is <number>."
)


def load_model_and_tokenizer(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID} onto {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_gsm8k(n: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i in range(n):
        row = ds[i]
        gold_raw = row["answer"].split("####")[-1].strip()
        gold = gold_raw.replace(",", "")
        problems.append(
            {
                "task": "gsm8k",
                "problem_id": f"gsm8k_{i}",
                "prompt": row["question"] + GSM8K_SUFFIX,
                "gold": gold,
            }
        )
    return problems


def extract_last_number(text: str):
    import re

    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def grade(prediction, gold: str) -> bool:
    if prediction is None:
        return False
    try:
        return float(prediction) == float(gold)
    except ValueError:
        return prediction.strip() == gold.strip()


def majority_vote(predictions):
    """Group predictions by numeric value (so '1400' and '1400.0' count together),
    return (winning_raw_string, vote_counts) or (None, {}) if all predictions are None."""
    groups = {}  # canonical_key -> [raw_strings]
    for p in predictions:
        if p is None:
            continue
        try:
            key = float(p)
        except ValueError:
            key = p
        groups.setdefault(key, []).append(p)

    if not groups:
        return None, {}

    counts = Counter({k: len(v) for k, v in groups.items()})
    winning_key, _ = counts.most_common(1)[0]
    winning_raw = groups[winning_key][0]
    return winning_raw, {str(k): len(v) for k, v in groups.items()}


def sample_completions(model, tokenizer, device, prompt: str, n: int, seed: int):
    import torch

    torch.manual_seed(seed)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt", add_special_tokens=False).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=1.0,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    total_new_tokens = 0
    for seq in outputs:
        new_tokens = seq[prompt_len:]
        total_new_tokens += new_tokens.shape[0]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return completions, latency, total_new_tokens


def load_existing(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="2 problems, N in {1,2} only")
    parser.add_argument("--resume", action="store_true", help="skip problem_ids already recorded")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    n_problems = 2 if args.smoke_test else N_GSM8K
    n_values = [1, 2] if args.smoke_test else N_VALUES
    max_n = max(n_values)

    problems = load_gsm8k(n_problems)
    print(f"Loaded {len(problems)} problems, sweeping N in {n_values} (max_n={max_n})")
    print(f"Total base generations: {len(problems) * max_n}")

    model, tokenizer = load_model_and_tokenizer(args.device)

    output_path = Path(args.output)
    records = load_existing(output_path) if args.resume else []
    done_ids = {rec["problem_id"] for rec in records}

    for idx, p in enumerate(problems):
        if p["problem_id"] in done_ids:
            continue

        seed = 2000 + idx
        try:
            completions, latency, total_tokens = sample_completions(
                model, tokenizer, args.device, p["prompt"], max_n, seed
            )
            per_sample_predictions = [extract_last_number(c) for c in completions]

            budgets = {}
            for n in n_values:
                prefix_preds = per_sample_predictions[:n]
                majority_pred, vote_counts = majority_vote(prefix_preds)
                budgets[str(n)] = {
                    "majority_prediction": majority_pred,
                    "correct": grade(majority_pred, p["gold"]),
                    "vote_counts": vote_counts,
                    "individual_predictions": prefix_preds,
                }

            record = {
                "problem_id": p["problem_id"],
                "task": p["task"],
                "gold": p["gold"],
                "budgets": budgets,
                "total_latency_s": latency,
                "total_new_tokens": total_tokens,
                "raw_completions": completions,
            }
        except Exception as e:
            record = {
                "problem_id": p["problem_id"],
                "task": p["task"],
                "gold": p["gold"],
                "budgets": {},
                "total_latency_s": None,
                "total_new_tokens": None,
                "raw_completions": None,
                "error": str(e),
            }
            print(f"  ERROR on {p['problem_id']}: {e}")

        records.append(record)
        save(output_path, records)

        status = "OK" if record.get("error") is None else "ERR"
        print(f"[{idx + 1}/{len(problems)}] {status} {p['problem_id']} gold={p['gold']!r}")
        if record.get("error") is None:
            for n in n_values:
                b = record["budgets"][str(n)]
                print(f"    N={n:>2}  pred={b['majority_prediction']!r}  correct={b['correct']}")

    print(f"Done. Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
