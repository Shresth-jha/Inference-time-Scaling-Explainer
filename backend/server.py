"""
Live inference backend for Recurrence Lab.

Real models, run live, streamed token-by-token as they're actually generated
-- not a replay of pre-baked sweep data, not a canned animation. The frontend
diagram lights up in lockstep with these SSE events; nothing animates unless
a real generate() call is actually producing that token right now.

Endpoints (SSE streaming, primary):
  - GET /stream/best_of_n?question=...&n=...       -> Qwen2.5-1.5B-Instruct, CPU
  - GET /stream/budget_forcing?question=...&budget=... -> Qwen2.5-1.5B-Instruct, CPU
  - GET /stream/huginn?question=...&r=...&task=...  -> Huginn-0125, GPU, 4-bit quantized

Endpoints (non-streaming, kept for simple curl testing / fallback):
  - POST /generate/best_of_n, /generate/budget_forcing, /generate/huginn

There is no gold answer for an arbitrary user question, so these endpoints
never claim correct/incorrect -- only real-sweep replay (driven by our GSM8K
data, in the frontend's offline dropdown) can do that.

Run:
    python backend/server.py
Then open frontend/public/index.html via `python -m http.server 5173
--directory frontend/public` -- the page detects this server at
http://localhost:8000 and enables the real live-model controls.
"""

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

# Fixes a ~64x CPU inference slowdown on AMD CPUs: Intel MKL's default CPU
# dispatcher takes a much slower code path on non-Intel chips even when the
# hardware fully supports the fast instructions. Confirmed via isolated
# benchmark on this machine (Ryzen 7 7840HS): 33.7s/token -> 0.53s/token.
# Must be set before torch/transformers are imported.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parents[1] / "experiment"))

import bon_sweep
import budget_forcing_sweep
import huginn_sweep

# Live requests trade some generation length for responsiveness. Shorter
# generations still show the real mechanism; they just cut off long-winded
# reasoning sooner. (These affect the non-streaming endpoints and the prompt
# construction shared with the streaming ones.)
LIVE_MAX_NEW_TOKENS = 120
LIVE_HUGINN_MAX_NEW_TOKENS = 100

app = FastAPI(title="Recurrence Lab live inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"qwen_model": None, "qwen_tokenizer": None, "huginn_model": None, "huginn_tokenizer": None, "huginn_gen_config": None}


@app.on_event("startup")
def load_models():
    print("Loading Qwen2.5-1.5B-Instruct onto cpu ...")
    STATE["qwen_model"], STATE["qwen_tokenizer"] = bon_sweep.load_model_and_tokenizer("cpu")
    print("Qwen ready.")

    try:
        import torch

        if os.environ.get("SKIP_HUGINN") == "1":
            print("SKIP_HUGINN=1 set -- not loading Huginn (e.g. GPU busy with an offline sweep).")
        elif torch.cuda.is_available():
            print("Loading Huginn-0125 onto cuda (4-bit nf4, full r range) ...")
            STATE["huginn_model"], STATE["huginn_tokenizer"] = huginn_sweep.load_model_and_tokenizer("cuda", quantize=True)
            STATE["huginn_gen_config"] = huginn_sweep.build_generation_config(STATE["huginn_tokenizer"])
            print("Huginn ready.")
        else:
            print("No CUDA GPU visible -- live Huginn endpoint disabled (would be too slow on CPU).")
    except Exception as e:
        print(f"Huginn failed to load, live Huginn endpoint disabled: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "qwen_loaded": STATE["qwen_model"] is not None,
        "huginn_loaded": STATE["huginn_model"] is not None,
    }


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ---------- streaming: best-of-n (N real samples, streamed concurrently) ----------


def _stream_one_sample(model, tokenizer, prompt: str, sample_idx: int, seed: int, out_q: "queue.Queue"):
    import torch
    from transformers import TextIteratorStreamer

    torch.manual_seed(seed)
    messages = [{"role": "system", "content": bon_sweep.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_input, return_tensors="pt", add_special_tokens=False)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=LIVE_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=bon_sweep.TEMPERATURE,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    full_text = ""
    for piece in streamer:
        full_text += piece
        out_q.put({"type": "token", "sample": sample_idx, "text": piece})
    thread.join()

    prediction = bon_sweep.extract_last_number(full_text)
    out_q.put({"type": "sample_done", "sample": sample_idx, "prediction": prediction, "raw_text": full_text})


@app.get("/stream/best_of_n")
def stream_best_of_n(question: str, n: int = 3):
    n = max(1, min(n, 5))
    prompt = question.strip() + bon_sweep.GSM8K_SUFFIX

    def event_stream():
        yield sse({"type": "start", "question": question, "n": n})
        out_q: "queue.Queue" = queue.Queue()
        threads = []
        for i in range(n):
            seed = int(time.time() * 1000) % 100000 + i
            t = threading.Thread(target=_stream_one_sample, args=(STATE["qwen_model"], STATE["qwen_tokenizer"], prompt, i, seed, out_q))
            t.start()
            threads.append(t)

        results = {}
        finished = 0
        while finished < n:
            item = out_q.get()
            if item["type"] == "sample_done":
                results[item["sample"]] = item["prediction"]
                finished += 1
            yield sse(item)
        for t in threads:
            t.join()

        predictions = [results[i] for i in range(n)]
        majority_pred, vote_counts = bon_sweep.majority_vote(predictions)
        yield sse({"type": "done", "majority_prediction": majority_pred, "vote_counts": vote_counts})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- streaming: budget forcing (sequential, real forced continuations) ----------


@app.get("/stream/budget_forcing")
def stream_budget_forcing(question: str, budget: int = 2):
    import torch
    from transformers import TextIteratorStreamer

    budget = max(0, min(budget, 2))
    prompt = question.strip() + budget_forcing_sweep.GSM8K_SUFFIX
    model, tokenizer = STATE["qwen_model"], STATE["qwen_tokenizer"]

    def event_stream():
        yield sse({"type": "start", "question": question, "budget": budget})
        messages = [{"role": "system", "content": budget_forcing_sweep.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        running_text = chat_input
        forced_used = 0

        for step in range(budget + 1):
            inputs = tokenizer(running_text, return_tensors="pt", add_special_tokens=False)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=budget_forcing_sweep.MAX_NEW_TOKENS_PER_CHUNK,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            chunk_text = ""
            for piece in streamer:
                chunk_text += piece
                yield sse({"type": "token", "segment": step, "text": piece})
            thread.join()
            running_text += chunk_text

            if step < budget:
                running_text += budget_forcing_sweep.WAIT_INJECTION
                forced_used += 1
                yield sse({"type": "wait_injected", "segment": step})

        generated_only = running_text[len(chat_input):]
        prediction = budget_forcing_sweep.extract_last_number(generated_only)
        yield sse({"type": "done", "prediction": prediction, "forced_continuations_used": forced_used})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- streaming: huginn (single real pass at the requested r) ----------


@app.get("/stream/huginn")
def stream_huginn(question: str, r: int = 4, task: str = "gsm8k"):
    if STATE["huginn_model"] is None:
        def err_stream():
            yield sse({"type": "error", "message": "Huginn is not loaded on this server (no CUDA GPU detected)."})
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    r = max(1, min(r, 32))
    suffix = huginn_sweep.GSM8K_SUFFIX if task == "gsm8k" else huginn_sweep.OBQA_SUFFIX
    prompt = question.strip() + suffix
    model, tokenizer, gen_config = STATE["huginn_model"], STATE["huginn_tokenizer"], STATE["huginn_gen_config"]

    def event_stream():
        import torch
        from transformers import TextIteratorStreamer

        yield sse({"type": "start", "question": question, "r": r})
        messages = [{"role": "system", "content": huginn_sweep.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to("cuda")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(
            target=model.generate,
            args=(input_ids, gen_config),
            kwargs=dict(tokenizer=tokenizer, num_steps=r, streamer=streamer),
        )
        thread.start()
        full_text = ""
        for piece in streamer:
            full_text += piece
            yield sse({"type": "token", "text": piece})
        thread.join()

        prediction = huginn_sweep.extract_last_number(full_text) if task == "gsm8k" else huginn_sweep.extract_first_letter(full_text)
        yield sse({"type": "done", "prediction": prediction, "r": r})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- non-streaming fallback endpoints (simple curl testing) ----------


class BestOfNRequest(BaseModel):
    question: str
    n: int = 3


@app.post("/generate/best_of_n")
def generate_best_of_n(req: BestOfNRequest):
    n = max(1, min(req.n, 5))
    prompt = req.question.strip() + bon_sweep.GSM8K_SUFFIX
    seed = int(time.time() * 1000) % 100000
    old_tokens = bon_sweep.MAX_NEW_TOKENS
    bon_sweep.MAX_NEW_TOKENS = LIVE_MAX_NEW_TOKENS
    try:
        completions, latency, total_tokens = bon_sweep.sample_completions(
            STATE["qwen_model"], STATE["qwen_tokenizer"], "cpu", prompt, n, seed
        )
    finally:
        bon_sweep.MAX_NEW_TOKENS = old_tokens
    predictions = [bon_sweep.extract_last_number(c) for c in completions]
    majority_pred, vote_counts = bon_sweep.majority_vote(predictions)
    return {
        "question": req.question,
        "n": n,
        "samples": [{"prediction": predictions[i], "raw_text": completions[i]} for i in range(len(completions))],
        "majority_prediction": majority_pred,
        "vote_counts": vote_counts,
        "latency_s": latency,
        "total_tokens": total_tokens,
    }


class BudgetForcingRequest(BaseModel):
    question: str
    budget: int = 2


@app.post("/generate/budget_forcing")
def generate_budget_forcing(req: BudgetForcingRequest):
    budget = max(0, min(req.budget, 2))
    prompt = req.question.strip() + budget_forcing_sweep.GSM8K_SUFFIX
    old_tokens = budget_forcing_sweep.MAX_NEW_TOKENS_PER_CHUNK
    budget_forcing_sweep.MAX_NEW_TOKENS_PER_CHUNK = LIVE_MAX_NEW_TOKENS
    try:
        text, forced_used, tokens, latency = budget_forcing_sweep.run_budget_forcing(
            STATE["qwen_model"], STATE["qwen_tokenizer"], "cpu", prompt, budget
        )
    finally:
        budget_forcing_sweep.MAX_NEW_TOKENS_PER_CHUNK = old_tokens
    prediction = budget_forcing_sweep.extract_last_number(text)
    return {
        "question": req.question,
        "budget": budget,
        "forced_continuations_used": forced_used,
        "prediction": prediction,
        "raw_output": text,
        "total_tokens": tokens,
        "latency_s": latency,
    }


class HuginnRequest(BaseModel):
    question: str
    r: int = 4
    task: str = "gsm8k"


@app.post("/generate/huginn")
def generate_huginn(req: HuginnRequest):
    if STATE["huginn_model"] is None:
        return {"error": "Huginn is not loaded on this server (no CUDA GPU detected)."}

    r = max(1, min(req.r, 32))
    suffix = huginn_sweep.GSM8K_SUFFIX if req.task == "gsm8k" else huginn_sweep.OBQA_SUFFIX
    prompt = req.question.strip() + suffix
    seed = int(time.time() * 1000) % 100000
    raw_output, prediction, latency = huginn_sweep.run_one(
        STATE["huginn_model"], STATE["huginn_tokenizer"], STATE["huginn_gen_config"], "cuda", prompt, r, req.task, seed
    )
    return {
        "question": req.question,
        "r": r,
        "prediction": prediction,
        "raw_output": raw_output,
        "latency_s": latency,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
