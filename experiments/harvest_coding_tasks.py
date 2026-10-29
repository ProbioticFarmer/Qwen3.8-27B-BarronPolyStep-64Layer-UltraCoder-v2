"""Harvest live DeepSeek-V4 (`deepseek-flash` / `deepseek-v4-pro`) coding & bug-repair
traces with top-20 token logprobs for the Qwen3.8-27B Execution-Verified Coding Benchmark.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

CODING_PROMPTS = [
    "Write a Python function `barron_log_softmax_1d(z: list[float]) -> list[float]` that computes log_softmax excluding the maximum index from logsumexp to prevent 1 + exp(-m) rounding to 1. Return only code.",
    "Fix the off-by-one bug in binary search `bisect_left_range(arr, target)` so it returns the first index i where arr[i] >= target, or len(arr) if none exists. Return concise Python code.",
    "Write a Python function `fenwick_prefix_sum(nums: list[int], idx: int) -> int` using a 1-indexed Fenwick tree (Binary Indexed Tree) supporting O(log n) prefix queries. Return concise Python code.",
    "Write a Python function `kahn_toposort(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]` that returns the lexicographically smallest topological order or [] if a cycle exists.",
    "Write a Python function `mod_inverse_prime(a: int, p: int) -> int` using Fermat's little theorem `pow(a, p - 2, p)` and handle `a % p == 0` by raising ValueError.",
    "Write a Python function `edit_distance_dp(s1: str, s2: str) -> int` that computes Levenshtein distance in O(min(len(s1), len(s2))) space.",
    "Write a Python function `sinkhorn_row_norm(log_alpha: list[list[float]]) -> list[list[float]]` that normalizes each row in log-domain so exp of each row sums to 1.0.",
    "Write a Python function `max_subarray_circular(nums: list[int]) -> int` using Kadane's algorithm to handle both non-wrapping and circular wrapping subarrays.",
]


def harvest_deepseek_coding(api_key: str) -> Dict:
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
            print(f"[DeepSeek Coding API] Models available: {models_list}", flush=True)
    except Exception as e:
        print(f"[DeepSeek Coding API] Model list warning: {e}", flush=True)

    model_id = "deepseek-flash" if "deepseek-flash" in models_list else (models_list[0] if models_list else "deepseek-chat")
    print(f"[DeepSeek Coding API] Harvesting coding solutions from `{model_id}`...", flush=True)

    traces = []
    for idx, prompt in enumerate(CODING_PROMPTS):
        t0 = time.time()
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python algorithm engineer. Output only clean Python code without markdown prose.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "logprobs": True,
            "top_logprobs": 15,
        }
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            choice = res["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            lp_obj = choice.get("logprobs") or {}
            lp_list = lp_obj.get("content") or lp_obj.get("reasoning_content") or []
            token_records = []
            for tok_item in lp_list[:64]:
                top_candidates = [
                    {"token": c["token"], "logprob": float(c["logprob"])}
                    for c in tok_item.get("top_logprobs", [])[:10]
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
                f"  [{idx+1}/{len(CODING_PROMPTS)}] Harvested {len(lp_list)} tokens "
                f"(content={len(content)} chars, reasoning={len(reasoning)} chars) in {elapsed:.2f}s",
                flush=True,
            )
            traces.append(
                {
                    "prompt": prompt,
                    "content": content,
                    "reasoning_preview": reasoning[:300],
                    "num_logprob_tokens": len(lp_list),
                    "sampled_tokens": token_records,
                    "elapsed_sec": elapsed,
                }
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(CODING_PROMPTS)}] Warning: {e}", flush=True)

    return {
        "api_models": models_list,
        "selected_model": model_id,
        "num_traces": len(traces),
        "traces": traces,
    }


def main():
    out_path = Path("data/deepseek_coding_traces.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    result = {"api_models": [], "selected_model": "local-deepseek-v4-flash", "num_traces": 0, "traces": []}
    if api_key:
        result = harvest_deepseek_coding(api_key)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {result['num_traces']} coding traces to {out_path}", flush=True)


if __name__ == "__main__":
    main()
