"""Stage 1: Harvest Teacher Completions & Top-K Token Logprobs from DeepSeek API
and align with local DeepSeek-V4-Flash / Qwen3.8-27B (248,320-token vocabulary).
"""

from __future__ import annotations

import glob
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

PROMPTS = [
    "Derive the exact floating-point threshold where log(1 + exp(-m)) rounds to 0 in IEEE-754 binary32 and bfloat16.",
    "Write a concise Python function implementing Sinkhorn-Knopp log-domain matrix scaling without underflow.",
    "Explain why zero-order polytope optimization can train non-differentiable Heaviside spiking neurons without surrogate gradients.",
    "Prove that for any prime p, the discrete Fourier transform diagonalizes circular convolution modulo p.",
    "Give a compact PyTorch implementation of Group Query Attention (GQA) with RMSNorm on Q and K.",
    "Solve for the eigenvalues of a regular simplex vertex matrix in R^d centered at the origin.",
    "Explain why soft-target knowledge distillation suffers from gradient cancellation when student logits approach teacher logits.",
    "Write a fast bit-packing routine in Python/PyTorch that unpacks 4-bit affine-quantized U32 weights into bfloat16.",
]


def query_deepseek_api(api_key: str) -> Dict:
    """Query DeepSeek API for available models and harvest reasoning/technical completions with top-20 logprobs."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    models_list: List[str] = []
    try:
        req = urllib.request.Request("https://api.deepseek.com/models", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_list = [m.get("id", "") for m in data.get("data", [])]
            print(f"[DeepSeek API] Available models: {models_list}", flush=True)
    except Exception as e:
        print(f"[DeepSeek API] Model list query warning: {e}", flush=True)

    # Prefer flash / v4 / chat model supporting logprobs
    preferred = [
        m for m in models_list if any(k in m.lower() for k in ["flash", "v4", "chat"])
    ]
    model_id = preferred[0] if preferred else (models_list[0] if models_list else "deepseek-chat")
    print(f"[DeepSeek API] Using teacher model: {model_id}", flush=True)

    traces = []
    for idx, prompt in enumerate(PROMPTS):
        t0 = time.time()
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a precise mathematical and systems ML expert. Be concise."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 96,
            "temperature": 0.2,
            "logprobs": True,
            "top_logprobs": 20,
        }
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            choice = res["choices"][0]
            content = choice["message"].get("content", "")
            lp_list = (choice.get("logprobs") or {}).get("content") or []
            token_records = []
            for tok_item in lp_list:
                top_candidates = [
                    {"token": c["token"], "logprob": float(c["logprob"])}
                    for c in tok_item.get("top_logprobs", [])
                ]
                token_records.append(
                    {
                        "token": tok_item.get("token", ""),
                        "logprob": float(tok_item.get("logprob", 0.0)),
                        "top_logprobs": top_candidates,
                    }
                )
            elapsed = time.time() - t0
            print(
                f"  [{idx+1}/{len(PROMPTS)}] Harvested {len(token_records)} tokens with top-{len(token_records[0]['top_logprobs']) if token_records else 0} logprobs ({elapsed:.2f}s)",
                flush=True,
            )
            traces.append(
                {
                    "prompt": prompt,
                    "response": content,
                    "model": res.get("model", model_id),
                    "tokens": token_records,
                    "elapsed_sec": elapsed,
                }
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(PROMPTS)}] API call warning: {e}", flush=True)

    return {
        "api_models": models_list,
        "selected_model": model_id,
        "num_traces": len(traces),
        "traces": traces,
    }


def main():
    out_path = Path("data/deepseek_teacher_traces.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    result = {"api_models": [], "selected_model": "local-deepseek-v4-flash", "num_traces": 0, "traces": []}
    if api_key:
        result = query_deepseek_api(api_key)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {result['num_traces']} teacher traces to {out_path}", flush=True)


if __name__ == "__main__":
    main()
