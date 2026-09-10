"""
Experiment A: Huginn-0125 latent compute sweep.

Sweeps the recurrence depth r (the `num_steps` kwarg to generate()) over
{1, 4, 8, 16, 32, 64} on 30 GSM8K + 30 OpenBookQA problems, greedy decoding,
and records per-problem prediction / correctness / latency / raw output.

Usage:
    python huginn_sweep.py --smoke-test          # 2 problems x 2 r-values, sanity check
    python huginn_sweep.py                       # full sweep (360 generations)
    python huginn_sweep.py --resume              # skip (problem, r) pairs already in the output file

Requires: torch, transformers, datasets, accelerate
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate

Model card usage confirmed (tomg-group-umd/huginn-0125, 2026-09-08):
    - trust_remote_code=True is required (custom recurrent-depth architecture).
    - `num_steps` is passed directly to generate(), NOT into GenerationConfig
      ("num_steps and other model arguments CANNOT be included in the
      GenerationConfig" -- per the model card).
    - The model expects its own chat template via tokenizer.apply_chat_template.
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

# Reduces CUDA allocator fragmentation -- must be set before torch is imported.
# 8GB VRAM leaves very little headroom for a 3.5B bf16 model; r=4 OOM'd
# without this on the first smoke test. (Superseded for the default path by
# 4-bit quantization below, which fits the full r sweep in ~3.5-4.3GB; this
# still matters for --no-quantize.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Fixes a ~64x CPU inference slowdown on AMD CPUs (Intel MKL's default CPU
# dispatcher takes a much slower path on non-Intel chips). Only relevant for
# --device cpu, but harmless to set unconditionally.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

MODEL_ID = "tomg-group-umd/huginn-0125"
R_VALUES = [1, 4, 8, 16, 32, 64]
N_GSM8K = 30
N_OPENBOOKQA = 30
MAX_NEW_TOKENS = 200
OUTPUT_PATH = Path(__file__).parent / "data" / "huginn_sweep_raw.json"

SYSTEM_PROMPT = "You are a helpful assistant that solves problems step by step."

GSM8K_SUFFIX = (
    "\n\nSolve this step by step. On the final line, write your answer as:\n"
    "The answer is <number>."
)

OBQA_SUFFIX = (
    "\n\nThink through the options, then on the final line write your answer as:\n"
    "The answer is <letter>."
)


def load_model_and_tokenizer(device: str, quantize: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if device == "cuda" and quantize:
        # 4-bit (nf4) loading: confirmed on an 8GB GPU this drops Huginn-0125
        # from ~7.9GB to ~3.5GB resident, leaving enough headroom to run the
        # full r in {1,4,8,16,32,64} sweep (measured up to ~4.3GB at r=32)
        # instead of OOM'ing past r=1 in bf16.
        from transformers import BitsAndBytesConfig

        print(f"Loading {MODEL_ID} (4-bit nf4, trust_remote_code=True) onto {device} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, trust_remote_code=True, device_map=device
        )
    else:
        print(f"Loading {MODEL_ID} (bf16, trust_remote_code=True) onto {device} ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
    model.eval()
    return model, tokenizer


def build_generation_config(tokenizer):
    from transformers import GenerationConfig

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 65505
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 65504
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    return GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        stop_strings=["<|end_text|>", "<|end_turn|>"],
        use_cache=True,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        return_dict_in_generate=True,
        eos_token_id=eos_id,
        bos_token_id=bos_id,
        pad_token_id=pad_id,
    )


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


def load_openbookqa(n: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split="test")
    problems = []
    for i in range(n):
        row = ds[i]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choice_lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        prompt = f"{row['question_stem']}\n{choice_lines}{OBQA_SUFFIX}"
        problems.append(
            {
                "task": "openbookqa",
                "problem_id": f"openbookqa_{i}",
                "prompt": prompt,
                "gold": row["answerKey"].strip(),
            }
        )
    return problems


def extract_last_number(text: str):
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def extract_first_letter(text: str):
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else None


def grade(task: str, prediction, gold: str) -> bool:
    if prediction is None:
        return False
    if task == "gsm8k":
        try:
            return float(prediction) == float(gold)
        except ValueError:
            return prediction.strip() == gold.strip()
    return prediction.strip().upper() == gold.strip().upper()


def run_one(model, tokenizer, gen_config, device, prompt: str, r: int, task: str, seed: int):
    import torch

    torch.manual_seed(seed)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(
        chat_input, return_tensors="pt", add_special_tokens=False
    ).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids, gen_config, tokenizer=tokenizer, num_steps=r
        )
    latency = time.perf_counter() - start

    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    new_tokens = sequences[0][input_ids.shape[1] :]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if task == "gsm8k":
        prediction = extract_last_number(raw_output)
    else:
        prediction = extract_first_letter(raw_output)

    return raw_output, prediction, latency


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
    parser.add_argument("--smoke-test", action="store_true", help="2 problems x 2 r-values only")
    parser.add_argument("--resume", action="store_true", help="skip (problem_id, r) pairs already recorded")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--r-values", default=None, help="comma-separated override, e.g. 1,4,8,16"
    )
    parser.add_argument("--n-problems", type=int, default=None, help="override problems per task")
    parser.add_argument("--no-quantize", action="store_true", help="load in bf16 instead of 4-bit (needs more VRAM, only r=1 fits on 8GB)")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be very slow).")
        args.device = "cpu"

    if args.r_values:
        r_values = [int(x) for x in args.r_values.split(",")]
    else:
        r_values = R_VALUES[:2] if args.smoke_test else R_VALUES

    if args.n_problems is not None:
        n_gsm8k = n_obqa = args.n_problems
    else:
        n_gsm8k = 2 if args.smoke_test else N_GSM8K
        n_obqa = 2 if args.smoke_test else N_OPENBOOKQA

    problems = load_gsm8k(n_gsm8k) + load_openbookqa(n_obqa)
    print(f"Loaded {len(problems)} problems, sweeping r in {r_values}")
    print(f"Total generations: {len(problems) * len(r_values)}")

    model, tokenizer = load_model_and_tokenizer(args.device, quantize=not args.no_quantize)
    gen_config = build_generation_config(tokenizer)

    output_path = Path(args.output)
    records = load_existing(output_path) if args.resume else []
    done_keys = {
        (rec["problem_id"], rec["r"]) for rec in records if rec.get("error") is None
    }
    records = [rec for rec in records if rec.get("error") is None]

    total = len(problems) * len(r_values)
    count = len(records)

    for r in r_values:
        for idx, p in enumerate(problems):
            key = (p["problem_id"], r)
            if key in done_keys:
                continue

            seed = 1000 + idx
            try:
                raw_output, prediction, latency = run_one(
                    model, tokenizer, gen_config, args.device, p["prompt"], r, p["task"], seed
                )
                correct = grade(p["task"], prediction, p["gold"])
                record = {
                    "problem_id": p["problem_id"],
                    "task": p["task"],
                    "r": r,
                    "gold": p["gold"],
                    "prediction": prediction,
                    "correct": correct,
                    "latency_s": latency,
                    "raw_output": raw_output,
                }
            except Exception as e:
                record = {
                    "problem_id": p["problem_id"],
                    "task": p["task"],
                    "r": r,
                    "gold": p["gold"],
                    "prediction": None,
                    "correct": False,
                    "latency_s": None,
                    "raw_output": None,
                    "error": str(e),
                }
                print(f"  ERROR on {p['problem_id']} r={r}: {e}")

            records.append(record)
            count += 1
            save(output_path, records)

            import torch

            torch.cuda.empty_cache()
            gc.collect()

            status = "OK" if record.get("error") is None else "ERR"
            print(
                f"[{count}/{total}] {status} {p['problem_id']} r={r} "
                f"pred={record['prediction']!r} gold={p['gold']!r} "
                f"correct={record['correct']} latency={record['latency_s']}"
            )

    print(f"Done. Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
