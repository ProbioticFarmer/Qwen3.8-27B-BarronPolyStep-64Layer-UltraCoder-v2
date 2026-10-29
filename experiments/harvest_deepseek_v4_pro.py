"""Harvest live flagship `deepseek-v4-pro` API coding traces & top-20 token logprobs
for the 64-Layer Ultra-Scale Qwen3.8-27B Coding Distillation Suite.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

PRO_CODING_PROMPTS = [
    "Write a concise Python function `tarjan_scc_count(n: int, edges: list[tuple[int, int]]) -> int` that returns the number of strongly connected components in a directed graph.",
    "Write a concise Python function `manacher_longest_palindrome(s: str) -> int` that returns the length of the longest palindromic substring in O(n) time.",
    "Write a concise Python function `kmp_pattern_matches(text: str, pat: str) -> int` using a prefix-function (LPS) array to count occurrences of `pat` in `text` in O(n + m).",
    "Write a concise Python function `rabin_karp_rolling_hash(s: str, k: int, base: int = 257, mod: int = 1000000007) -> int` that returns the number of distinct length-k substrings hashes.",
    "Write a concise Python function `bellman_ford_negative_cycle(n: int, edges: list[tuple[int, int, int]]) -> bool` that detects whether a directed weighted graph has a negative cycle reachable from any node.",
    "Write a concise Python function `segment_tree_range_max(nums: list[int], l: int, r: int) -> int` using a 2n iterative segment tree supporting range maximum queries on [l, r).",
    "Write a concise Python function `miller_rabin_det32(n: int) -> bool` deterministic primality test for 32-bit integers using bases (2, 7, 61).",
    "Write a concise Python function `monotonic_queue_sliding_max(nums: list[int], k: int) -> list[int]` using `collections.deque` in O(n) time.",
    "Write a concise Python function `barron_soft_cross_entropy(z_s: list[float], z_t: list[float]) -> float` computing numerically stable soft-target cross-entropy excluding argmax from logsumexp.",
    "Write a concise Python function `sinkhorn_barycenter_step(cost_vec: list[float], eps: float) -> list[float]` computing stable log-domain entropic transport weights without underflow.",
]


def harvest_v4_pro(api_key: str) -> Dict:
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
            print(f"[DeepSeek-V4-Pro API] Available models: {models_list}", flush=True)
    except Exception as e:
        print(f"[DeepSeek-V4-Pro API] Model query warning: {e}", flush=True)

    model_id = "deepseek-v4-pro" if "deepseek-v4-pro" in models_list else ("deepseek-flash" if "deepseek-flash" in models_list else "deepseek-chat")
    print(f"[DeepSeek-V4-Pro API] Querying flagship teacher model: `{model_id}`...", flush=True)

    traces = []
    for idx, prompt in enumerate(PRO_CODING_PROMPTS):
        t0 = time.time()
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a principal algorithm engineer. Return only concise, bug-free Python code.",
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
            with urllib.request.urlopen(req, timeout=35) as resp:
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
                f"  [{idx+1:02d}/{len(PRO_CODING_PROMPTS)}] `{model_id}` -> {len(lp_list)} tokens "
                f"(content={len(content)} chars, reasoning={len(reasoning)} chars) in {elapsed:.2f}s",
                flush=True,
            )
            traces.append(
                {
                    "prompt": prompt,
                    "model": res.get("model", model_id),
                    "content": content,
                    "reasoning_preview": reasoning[:350],
                    "num_logprob_tokens": len(lp_list),
                    "sampled_tokens": token_records,
                    "elapsed_sec": elapsed,
                }
            )
        except Exception as e:
            print(f"  [{idx+1:02d}/{len(PRO_CODING_PROMPTS)}] Warning: {e}", flush=True)

    return {
        "api_models": models_list,
        "selected_model": model_id,
        "num_traces": len(traces),
        "traces": traces,
    }


def main():
    out_path = Path("data/deepseek_v4_pro_coding_traces.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    result = {"api_models": [], "selected_model": "deepseek-v4-pro", "num_traces": 0, "traces": []}
    if api_key:
        result = harvest_v4_pro(api_key)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {result['num_traces']} `deepseek-v4-pro` traces to {out_path}", flush=True)


if __name__ == "__main__":
    main()
