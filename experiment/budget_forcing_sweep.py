"""
Experiment C: budget forcing (Muennighoff et al. 2025, "s1: Simple test-time
scaling", arXiv:2501.19393, Section 3.1) on Qwen2.5-1.5B-Instruct.

This is OUR OWN implementation of the technique on our own small model --
NOT a reproduction of s1-32B, which was fine-tuned on s1K with an explicit
end-of-thinking delimiter. Qwen2.5-1.5B-Instruct has no such delimiter, so
we approximate the same mechanism structurally: the model is asked to end
its reasoning with "The answer is <number>."; when it tries to stop there,
we suppress that stop and append "Wait," to force it to keep reasoning, up
to `budget` times, before taking its final answer. This is the same
minimum-token-forcing idea as the paper (Sec 3.1: "we suppress the
generation of the end-of-thinking token delimiter and optionally append the
string 'Wait' ... to encourage the model to reflect on its current
generation"), adapted to a model without a trained delimiter.

Usage:
    python budget_forcing_sweep.py --smoke-test      # 2 problems, budgets {0,1}
    python budget_forcing_sweep.py                   # full sweep (30 problems x budgets {0,1,2,4})
    python budget_forcing_sweep.py --device cpu
    python budget_forcing_sweep.py --resume
"""

import os

# Fixes a ~64x CPU inference slowdown on AMD CPUs (Intel MKL's default CPU
# dispatcher takes a much slower path on non-Intel chips). Must be set before
# torch is imported. See backend/server.py for the benchmark that found this.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BUDGETS = [0, 1, 2, 4]  # number of forced "Wait" continuations
N_GSM8K = 30
MAX_NEW_TOKENS_PER_CHUNK = 200
OUTPUT_PATH = Path(__file__).parent / "data" / "budget_forcing_raw.json"

SYSTEM_PROMPT = "You are a helpful assistant that solves problems step by step."
GSM8K_SUFFIX = (
    "\n\nSolve this step by step. On the final line, write your answer as:\n"
    "The answer is <number>."
)
WAIT_INJECTION = "\nWait, let me double check that."


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
        problems.append({"task": "gsm8k", "problem_id": f"gsm8k_{i}", "prompt": row["question"] + GSM8K_SUFFIX, "gold": gold})
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


def run_budget_forcing(model, tokenizer, device, prompt: str, budget: int):
    """Generate greedily; each time the model stops (hits EOS) and we still
    have forced continuations left, append 'Wait, let me double check that.'
    and keep generating. Returns (full_text, n_forced_used, total_new_tokens, latency)."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    running_text = chat_input
    total_new_tokens = 0
    forced_used = 0

    start = time.perf_counter()
    for step in range(budget + 1):
        inputs = tokenizer(running_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_PER_CHUNK,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        total_new_tokens += new_tokens.shape[0]
        chunk_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        running_text = running_text + chunk_text

        stopped_naturally = new_tokens.shape[0] < MAX_NEW_TOKENS_PER_CHUNK
        if step < budget:
            running_text += WAIT_INJECTION
            forced_used += 1
        elif not stopped_naturally:
            pass  # ran out of budget AND out of tokens -- take what we have

    latency = time.perf_counter() - start
    generated_only = running_text[len(chat_input):]
    return generated_only, forced_used, total_new_tokens, latency


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
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--budgets", default=None, help="comma-separated override, e.g. 0,1,2,4")
    parser.add_argument("--n-problems", type=int, default=None)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    budgets = [int(x) for x in args.budgets.split(",")] if args.budgets else ([0, 1] if args.smoke_test else BUDGETS)
    n_problems = args.n_problems if args.n_problems is not None else (2 if args.smoke_test else N_GSM8K)

    problems = load_gsm8k(n_problems)
    print(f"Loaded {len(problems)} problems, sweeping budgets in {budgets}")
    print(f"Total generations (approx, chunked): {len(problems) * sum(b + 1 for b in budgets)}")

    model, tokenizer = load_model_and_tokenizer(args.device)

    output_path = Path(args.output)
    records = load_existing(output_path) if args.resume else []
    records = [r for r in records if r.get("error") is None]
    done_keys = {(r["problem_id"], r["budget"]) for r in records}

    total = len(problems) * len(budgets)
    count = len(records)

    for budget in budgets:
        for idx, p in enumerate(problems):
            key = (p["problem_id"], budget)
            if key in done_keys:
                continue
            try:
                text, forced_used, tokens, latency = run_budget_forcing(model, tokenizer, args.device, p["prompt"], budget)
                prediction = extract_last_number(text)
                correct = grade(prediction, p["gold"])
                record = {
                    "problem_id": p["problem_id"],
                    "task": p["task"],
                    "budget": budget,
                    "gold": p["gold"],
                    "prediction": prediction,
                    "correct": correct,
                    "forced_continuations_used": forced_used,
                    "total_new_tokens": tokens,
                    "latency_s": latency,
                    "raw_output": text,
                }
            except Exception as e:
                record = {
                    "problem_id": p["problem_id"], "task": p["task"], "budget": budget, "gold": p["gold"],
                    "prediction": None, "correct": False, "forced_continuations_used": None,
                    "total_new_tokens": None, "latency_s": None, "raw_output": None, "error": str(e),
                }
                print(f"  ERROR on {p['problem_id']} budget={budget}: {e}")

            records.append(record)
            count += 1
            save(output_path, records)

            status = "OK" if record.get("error") is None else "ERR"
            print(f"[{count}/{total}] {status} {p['problem_id']} budget={budget} pred={record['prediction']!r} gold={p['gold']!r} correct={record['correct']}")

            import gc
            torch.cuda.empty_cache() if args.device == "cuda" else None
            gc.collect()

    print(f"Done. Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
