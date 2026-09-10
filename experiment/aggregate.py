"""
Step 3: turn the two raw sweep outputs into frontend-ready JSON.

Reads:
    experiment/data/huginn_sweep_raw.json  (from huginn_sweep.py)
    experiment/data/bon_sweep_raw.json     (from bon_sweep.py)

Writes:
    frontend/public/data/results.json

Only covers the two cells we ran ourselves:
    - transformer x best-of-n   (bon_sweep.py, Qwen2.5-1.5B-Instruct, GSM8K)
    - huginn-0125 x latent compute (huginn_sweep.py, GSM8K + OpenBookQA)

The other two results cells (Snell et al. beam search, BDH-CQ Table 5) are
published numbers, hand-entered into frontend/public/data/content.json --
this script does not touch them.

Output shape:
{
  "curves": [
    {"system": ..., "method": ..., "task": ..., "budget": ..., "accuracy": ...,
     "n_problems": ..., "avg_latency_s": ...}
  ],
  "per_problem": {
    "<system>__<method>__<task>": [
      {"problem_id": ..., "budget": ..., "prediction": ..., "correct": ...,
       "gold": ..., "latency_s": ...}
    ]
  }
}
"""

import json
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DATA = Path(__file__).parent / "data"
HUGINN_RAW = EXPERIMENT_DATA / "huginn_sweep_raw.json"
BON_RAW = EXPERIMENT_DATA / "bon_sweep_raw.json"
BUDGET_FORCING_RAW = EXPERIMENT_DATA / "budget_forcing_raw.json"
OUTPUT_PATH = Path(__file__).parents[1] / "frontend" / "public" / "data" / "results.json"

HUGINN_SYSTEM = "huginn-0125"
HUGINN_METHOD = "latent_compute"
TRANSFORMER_SYSTEM = "transformer_qwen2.5-1.5b-instruct"
BON_METHOD = "best_of_n"
BUDGET_FORCING_METHOD = "budget_forcing"

MAX_RAW_CHARS = 1500  # cap stored generation text so results.json stays a reasonable size


def build_question_map(gsm8k_n=30, openbookqa_n=30):
    """Re-derive the actual question text per problem_id from the source
    datasets (already cached locally from the sweeps) so the frontend can
    show a real question, not a fabricated one, when a learner picks one."""
    questions = {}
    try:
        from datasets import load_dataset

        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        for i in range(gsm8k_n):
            questions[f"gsm8k_{i}"] = {"task": "gsm8k", "question": gsm8k[i]["question"]}

        obqa = load_dataset("allenai/openbookqa", "main", split="test")
        for i in range(openbookqa_n):
            row = obqa[i]
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            choice_lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
            questions[f"openbookqa_{i}"] = {
                "task": "openbookqa",
                "question": f"{row['question_stem']}\n{choice_lines}",
            }
    except Exception as e:
        print(f"WARNING: could not rebuild question map ({e}); real-question picker will be limited.")
    return questions


def trunc(s):
    if s is None:
        return None
    return s if len(s) <= MAX_RAW_CHARS else s[:MAX_RAW_CHARS] + "…"


def load(path: Path):
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_huginn(records):
    """records: list of {problem_id, task, r, gold, prediction, correct, latency_s, raw_output, [error]}"""
    curves = []
    per_problem = defaultdict(list)

    by_task_r = defaultdict(list)
    for rec in records:
        if rec.get("error"):
            continue
        by_task_r[(rec["task"], rec["r"])].append(rec)

    for (task, r), recs in sorted(by_task_r.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(recs)
        n_correct = sum(1 for x in recs if x["correct"])
        latencies = [x["latency_s"] for x in recs if x["latency_s"] is not None]
        curves.append(
            {
                "system": HUGINN_SYSTEM,
                "method": HUGINN_METHOD,
                "task": task,
                "budget": r,
                "accuracy": n_correct / n if n else None,
                "n_problems": n,
                "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
            }
        )

    for rec in records:
        if rec.get("error"):
            continue
        key = f"{HUGINN_SYSTEM}__{HUGINN_METHOD}__{rec['task']}"
        per_problem[key].append(
            {
                "problem_id": rec["problem_id"],
                "budget": rec["r"],
                "prediction": rec["prediction"],
                "gold": rec["gold"],
                "correct": rec["correct"],
                "latency_s": rec["latency_s"],
                "raw_output": trunc(rec.get("raw_output")),
            }
        )

    return curves, per_problem


def aggregate_bon(records):
    """records: list of {problem_id, task, gold, budgets: {N: {majority_prediction, correct, ...}}, total_latency_s, [error]}"""
    curves = []
    per_problem = defaultdict(list)

    by_task_n = defaultdict(list)
    for rec in records:
        if rec.get("error"):
            continue
        for n_str, budget_data in rec["budgets"].items():
            by_task_n[(rec["task"], int(n_str))].append((rec, budget_data))

    for (task, n), pairs in sorted(by_task_n.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        total = len(pairs)
        n_correct = sum(1 for _, b in pairs if b["correct"])
        curves.append(
            {
                "system": TRANSFORMER_SYSTEM,
                "method": BON_METHOD,
                "task": task,
                "budget": n,
                "accuracy": n_correct / total if total else None,
                "n_problems": total,
                "avg_latency_s": None,
            }
        )

    for rec in records:
        if rec.get("error"):
            continue
        key = f"{TRANSFORMER_SYSTEM}__{BON_METHOD}__{rec['task']}"
        raw_completions = rec.get("raw_completions") or []
        for n_str, budget_data in rec["budgets"].items():
            n = int(n_str)
            individual = budget_data["individual_predictions"]
            samples = [
                {"prediction": individual[i], "raw_text": trunc(raw_completions[i]) if i < len(raw_completions) else None}
                for i in range(len(individual))
            ]
            per_problem[key].append(
                {
                    "problem_id": rec["problem_id"],
                    "budget": n,
                    "prediction": budget_data["majority_prediction"],
                    "gold": rec["gold"],
                    "correct": budget_data["correct"],
                    "vote_counts": budget_data["vote_counts"],
                    "samples": samples,
                }
            )

    return curves, per_problem


def aggregate_budget_forcing(records):
    """records: list of {problem_id, task, budget, gold, prediction, correct,
    forced_continuations_used, total_new_tokens, latency_s, raw_output, [error]}"""
    curves = []
    per_problem = defaultdict(list)

    by_task_budget = defaultdict(list)
    for rec in records:
        if rec.get("error"):
            continue
        by_task_budget[(rec["task"], rec["budget"])].append(rec)

    for (task, budget), recs in sorted(by_task_budget.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(recs)
        n_correct = sum(1 for x in recs if x["correct"])
        latencies = [x["latency_s"] for x in recs if x["latency_s"] is not None]
        curves.append(
            {
                "system": TRANSFORMER_SYSTEM,
                "method": BUDGET_FORCING_METHOD,
                "task": task,
                "budget": budget,
                "accuracy": n_correct / n if n else None,
                "n_problems": n,
                "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
            }
        )

    for rec in records:
        if rec.get("error"):
            continue
        key = f"{TRANSFORMER_SYSTEM}__{BUDGET_FORCING_METHOD}__{rec['task']}"
        per_problem[key].append(
            {
                "problem_id": rec["problem_id"],
                "budget": rec["budget"],
                "prediction": rec["prediction"],
                "gold": rec["gold"],
                "correct": rec["correct"],
                "latency_s": rec["latency_s"],
                "raw_output": trunc(rec.get("raw_output")),
            }
        )

    return curves, per_problem


def main():
    all_curves = []
    all_per_problem = {}

    huginn_records = load(HUGINN_RAW)
    if huginn_records:
        curves, per_problem = aggregate_huginn(huginn_records)
        all_curves.extend(curves)
        all_per_problem.update(per_problem)
        print(f"Aggregated {len(huginn_records)} Huginn records -> {len(curves)} curve points")

    bon_records = load(BON_RAW)
    if bon_records:
        curves, per_problem = aggregate_bon(bon_records)
        all_curves.extend(curves)
        all_per_problem.update(per_problem)
        print(f"Aggregated {len(bon_records)} best-of-N records -> {len(curves)} curve points")

    bf_records = load(BUDGET_FORCING_RAW)
    if bf_records:
        curves, per_problem = aggregate_budget_forcing(bf_records)
        all_curves.extend(curves)
        all_per_problem.update(per_problem)
        print(f"Aggregated {len(bf_records)} budget-forcing records -> {len(curves)} curve points")

    questions = build_question_map()
    print(f"Built question map for {len(questions)} problems")

    output = {"curves": all_curves, "per_problem": all_per_problem, "questions": questions}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
