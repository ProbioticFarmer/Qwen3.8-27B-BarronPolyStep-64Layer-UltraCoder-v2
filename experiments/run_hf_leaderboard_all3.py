"""Deploy and Run the 3-Step Public Leaderboard Execution, Standalone 27B Merge Verification
& Submission Pipeline (`hf_leaderboard_all3_pipeline.py`) on Hugging Face `a100-large` (80GB).
"""

from __future__ import annotations

import argparse
import os
import tarfile
import time
import huggingface_hub as hf

PIQWEN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3"
SYNC_REPO_ID = "ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2"
BUNDLE_TAR = "/tmp/piqwen_v4_patch.tar.gz"
RESULTS_TAR_NAME = "piqwen_leaderboard_all3_results.tar.gz"


def build_bundle(force: bool = True) -> str:
    if os.path.exists(BUNDLE_TAR):
        os.remove(BUNDLE_TAR)
    include_items = [
        ("cuda_qwen38_gpu.py", os.path.join(PIQWEN_DIR, "cuda_qwen38_gpu.py")),
        ("hf_side_by_side_runner.py", os.path.join(PIQWEN_DIR, "hf_side_by_side_runner.py")),
        ("hf_competitions_suite_runner.py", os.path.join(PIQWEN_DIR, "hf_competitions_suite_runner.py")),
        ("hf_full_upstream_evalplus_runner.py", os.path.join(PIQWEN_DIR, "hf_full_upstream_evalplus_runner.py")),
        ("hf_heldout_code_math_runner.py", os.path.join(PIQWEN_DIR, "hf_heldout_code_math_runner.py")),
        ("hf_leaderboard_all3_pipeline.py", os.path.join(PIQWEN_DIR, "hf_leaderboard_all3_pipeline.py")),
        ("polystep_v5_280x_trainer_cuda.py", os.path.join(PIQWEN_DIR, "polystep_v5_280x_trainer_cuda.py")),
        ("qwen38_compact_72tasks_cache.npz", "/tmp/qwen38_compact_72tasks_cache.npz"),
        ("docstring_transition_vecs.npz", "/tmp/docstring_transition_vecs.npz"),
        ("v4_active_6tensors.safetensors", "/tmp/v4_active_6tensors.safetensors"),
        ("leaderboard_all3_hf_results.json", os.path.join(PIQWEN_DIR, "leaderboard_all3_hf_results.json")),
        ("full_upstream_evalplus_hf_results.json", os.path.join(PIQWEN_DIR, "full_upstream_evalplus_hf_results.json")),
        ("submissions", os.path.join(PIQWEN_DIR, "submissions")),
        (
            "checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/adapter_config.json",
            os.path.join(
                PIQWEN_DIR,
                "checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/adapter_config.json",
            ),
        ),
    ]
    with tarfile.open(BUNDLE_TAR, "w:gz") as tar:
        for arcname, full_p in include_items:
            if os.path.exists(full_p):
                tar.add(full_p, arcname=arcname)
            else:
                raise FileNotFoundError(f"Missing bundle file: {full_p}")
    size_mb = os.path.getsize(BUNDLE_TAR) / (1024 * 1024)
    print(f"[Bundle] Created lightweight v4 patch {BUNDLE_TAR} ({size_mb:.2f} MB)", flush=True)
    return BUNDLE_TAR


REMOTE_SCRIPT = r"""
set -euo pipefail
echo "================================================================================"
echo "BOOTING HUGGING FACE A100-LARGE (80GB VRAM) 3-STEP PUBLIC LEADERBOARD PIPELINE"
echo "Step 1: Official Upstream bigcode/bigcodebench-hard (All 148 Tasks) + HF PRs"
echo "Step 2: Complete Official evalplus/mbppplus (All 378 Tasks) + humanevalplus (164)"
echo "Step 3: Standalone 27B Merge + 3-Part Functional Verification Gate + Queueing"
echo "================================================================================"
nvidia-smi

echo "[1/4] Installing Python dependencies..."
pip install -q datasets safetensors tokenizers "huggingface_hub>=0.26.0" gguf \
    pandas scipy scikit-learn matplotlib seaborn beautifulsoup4 lxml statsmodels \
    nltk psutil flask flask-wtf flask-login flask-mail wtforms opencv-python-headless \
    rsa pycryptodome cryptography wordcloud shapely geopandas soundfile librosa \
    regex openpyxl xlwt pyquery gensim python-Levenshtein pytesseract
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" || true

echo "[2/4] Downloading 3-Step Leaderboard bundle + v4 patch to /Users/joshua/evaluation/PiQwen..."
mkdir -p /Users/joshua/evaluation/PiQwen
python3 - <<'PY'
import huggingface_hub as hf, tarfile
from safetensors.torch import load_file, save_file

path = hf.hf_hub_download(
    repo_id="ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
    filename="piqwen_leaderboard_all3_bundle.tar.gz",
    repo_type="model",
)
with tarfile.open(path, "r:gz") as tar:
    tar.extractall("/Users/joshua/evaluation/PiQwen")

patch_path = hf.hf_hub_download(
    repo_id="ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
    filename="piqwen_v4_patch.tar.gz",
    repo_type="model",
    force_download=True,
)
with tarfile.open(patch_path, "r:gz") as tar:
    tar.extractall("/Users/joshua/evaluation/PiQwen")

st_full = "/Users/joshua/evaluation/PiQwen/checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/adapter_model.safetensors"
st_patch = "/Users/joshua/evaluation/PiQwen/v4_active_6tensors.safetensors"
sd = load_file(st_full)
sd_patch = load_file(st_patch)
sd.update(sd_patch)
save_file(sd, st_full)
print(f"Applied v4 multi-module patch ({len(sd_patch)} active tensors) into {st_full} ({len(sd)} total tensors).", flush=True)
PY

echo "[3/4] Downloading Qwen/Qwen3.8-27B (55.6 GB bfloat16, 18 shards)..."
python3 - <<'PY'
import time, huggingface_hub as hf
t0 = time.time()
snap = hf.snapshot_download("Qwen/Qwen3.8-27B")
print(f"Downloaded Qwen/Qwen3.8-27B to {snap} in {time.time()-t0:.1f}s", flush=True)
PY

echo "[4/4] Running 3-Step Public Leaderboard Pipeline on NVIDIA A100-80GB..."
cd /Users/joshua/evaluation/PiQwen
python3 /Users/joshua/evaluation/PiQwen/hf_leaderboard_all3_pipeline.py

echo "Uploading 3-Step Leaderboard Results & Official Submissions to HF Hub..."
python3 - <<'PY'
import os, tarfile, huggingface_hub as hf
root = "/Users/joshua/evaluation/PiQwen"
out_tar = "/tmp/piqwen_leaderboard_all3_results.tar.gz"
with tarfile.open(out_tar, "w:gz") as tar:
    for rel in [
        "leaderboard_all3_hf_results.json",
        "LEADERBOARD_ALL3_REPORT.md",
        "submissions",
        "checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/adapter_model.safetensors",
    ]:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            tar.add(p, arcname=rel)

api = hf.HfApi()
for repo in [
    "ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
    "ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2",
]:
    api.upload_file(
        path_or_fileobj=out_tar,
        path_in_repo="piqwen_leaderboard_all3_results.tar.gz",
        repo_id=repo,
        repo_type="model",
        commit_message="Upload 3-Step Public Leaderboard A100-80GB results, BCB-Hard 148, MBPP+ 378, and Merge Verification",
    )
    for rel in [
        "leaderboard_all3_hf_results.json",
        "LEADERBOARD_ALL3_REPORT.md",
        "submissions/humanevalplus_164_polystep_samples.jsonl",
        "submissions/mbppplus_378_polystep_samples.jsonl",
        "submissions/EVALPLUS_LEADERBOARD_SUBMISSION.md",
        "submissions/ProbioticFarmer_Qwen3.8-27B-BarronPolyStep-RealCoder-v3_ProbioticFarmer.json",
        "submissions/ProbioticFarmer--Qwen3.8-27B-BarronPolyStep-RealCoder-v3--main--bigcodebench-hard-complete--custom-0-1-sanitized_calibrated.jsonl",
        "submissions/ProbioticFarmer--Qwen3.8-27B-BarronPolyStep-RealCoder-v3--main--bigcodebench-hard-complete--custom-0-1-sanitized_calibrated_eval_results.json",
    ]:
        full_p = os.path.join(root, rel)
        if os.path.exists(full_p):
            api.upload_file(
                path_or_fileobj=full_p,
                path_in_repo=rel,
                repo_id=repo,
                repo_type="model",
                commit_message=f"Upload {rel} from 3-Step Leaderboard A100-80GB pipeline",
            )
print("Successfully uploaded 3-Step Leaderboard results and official submissions to HF Hub!", flush=True)
PY
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-bundle-only", action="store_true")
    args = parser.parse_args()

    bundle_path = build_bundle(force=args.build_bundle_only)
    if args.build_bundle_only:
        return

    api = hf.HfApi()
    token = hf.get_token()
    assert token, "HF token not found"

    active_job_file = os.path.join(PIQWEN_DIR, "active_hf_leaderboard_all3_job.txt")
    if os.path.exists(active_job_file):
        try:
            old_jid = open(active_job_file, encoding="utf-8").read().strip()
            if old_jid:
                api.cancel_job(job_id=old_jid)
                print(f"[Cleanup] Canceled previous HF job {old_jid}", flush=True)
        except Exception as e_can:
            print(f"[Cleanup Note] {e_can}", flush=True)

    print(f"[Upload] Uploading {bundle_path} to {REPO_ID}...", flush=True)
    api.upload_file(
        path_or_fileobj=bundle_path,
        path_in_repo="piqwen_v4_patch.tar.gz",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload lightweight v5 280x CUDA SVD patch for 3-Step Leaderboard A100-80GB execution",
    )
    print("[Upload] Bundle uploaded.", flush=True)

    job = api.run_job(
        image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        command=["bash", "-c", REMOTE_SCRIPT],
        flavor="a100-large",
        timeout="2h",
        env={
            "PIQWEN_BACKEND": "cuda",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        secrets={"HF_TOKEN": token},
    )
    print(
        f"[HF Job Launched] id={job.id} | flavor=a100-large | url={getattr(job, 'url', '')}",
        flush=True,
    )
    with open(os.path.join(PIQWEN_DIR, "active_hf_leaderboard_all3_job.txt"), "w", encoding="utf-8") as f:
        f.write(f"{job.id}\n")

    while True:
        try:
            for line in api.fetch_job_logs(job_id=job.id):
                print(line, flush=True)
        except Exception as e:
            print(f"[Log stream info] {e}", flush=True)

        info = api.inspect_job(job_id=job.id)
        stage = info.status.stage
        print(f"[HF Job Status] id={job.id} stage={stage}", flush=True)
        if stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            break
        time.sleep(5)

    if stage == "COMPLETED":
        print("[Download] Fetching piqwen_leaderboard_all3_results.tar.gz from HF Hub...", flush=True)
        res_tar = hf.hf_hub_download(
            repo_id=REPO_ID,
            filename=RESULTS_TAR_NAME,
            repo_type="model",
            force_download=True,
        )
        with tarfile.open(res_tar, "r:gz") as tar:
            tar.extractall(PIQWEN_DIR)
        print(f"[Complete] Extracted 3-Step Leaderboard results into {PIQWEN_DIR}!", flush=True)


if __name__ == "__main__":
    main()
