"""Deploy and Run the Clean `240x` `<think>` Chain-of-Thought Evaluation Pipeline
(`hf_clean_240x_think_eval.py`) on Hugging Face `a100-large` (NVIDIA A100-SXM4-80GB).
"""

from __future__ import annotations

import argparse
import os
import tarfile
import time
import huggingface_hub as hf

PIQWEN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3"
BUNDLE_TAR = "/tmp/clean_240x_think_bundle.tar.gz"
RESULTS_TAR_NAME = "clean_240x_think_results.tar.gz"


def build_bundle() -> str:
    if os.path.exists(BUNDLE_TAR):
        os.remove(BUNDLE_TAR)
    include_items = [
        ("cuda_qwen38_gpu.py", os.path.join(PIQWEN_DIR, "cuda_qwen38_gpu.py")),
        ("hf_clean_240x_think_eval.py", os.path.join(PIQWEN_DIR, "hf_clean_240x_think_eval.py")),
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
    print(f"[Bundle] Created lightweight clean <think> bundle {BUNDLE_TAR} ({size_mb:.2f} MB)", flush=True)
    return BUNDLE_TAR


REMOTE_SCRIPT = r"""
set -euo pipefail
echo "================================================================================"
echo "BOOTING HUGGING FACE A100-LARGE (80GB VRAM) CLEAN 240x <think> CoT EVALUATION"
echo "  * Benchmarks: HumanEval+ (164), MBPP+ (378), BigCodeBench-Hard (148)"
echo "  * Zero Train-on-Test (Train-40, 0% overlap) & Zero Test-Suite Oracle"
echo "================================================================================"
nvidia-smi

echo "[1/4] Installing Python & BigCodeBench scientific dependencies..."
pip install -q datasets safetensors tokenizers "huggingface_hub[hf_transfer]>=0.26.0" hf_transfer gguf \
    pandas scipy scikit-learn matplotlib seaborn beautifulsoup4 lxml statsmodels \
    nltk psutil flask flask-wtf flask-login flask-mail wtforms opencv-python-headless \
    rsa pycryptodome cryptography wordcloud shapely geopandas soundfile librosa \
    regex openpyxl xlwt pyquery gensim python-Levenshtein pytesseract
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" || true

echo "[2/4] Downloading clean_240x_think_bundle.tar.gz & clean 240x adapter_model.safetensors..."
mkdir -p /Users/joshua/evaluation/PiQwen/checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'PY'
import shutil, tarfile, huggingface_hub as hf

b_path = hf.hf_hub_download(
    repo_id="ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
    filename="clean_240x_think_bundle.tar.gz",
    repo_type="model",
    force_download=True,
)
with tarfile.open(b_path, "r:gz") as tar:
    tar.extractall("/Users/joshua/evaluation/PiQwen")

ad_path = hf.hf_hub_download(
    repo_id="ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
    filename="adapter_model.safetensors",
    repo_type="model",
)
dst_ad = "/Users/joshua/evaluation/PiQwen/checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/adapter_model.safetensors"
shutil.copy2(ad_path, dst_ad)
print("Extracted bundle and clean 240x adapter_model.safetensors!", flush=True)
PY

echo "[3/4] Downloading Qwen/Qwen3.8-27B (55.6 GB bfloat16, 18 shards)..."
python3 - <<'PY'
import time, huggingface_hub as hf
t0 = time.time()
snap = hf.snapshot_download("Qwen/Qwen3.8-27B")
print(f"Downloaded Qwen/Qwen3.8-27B to {snap} in {time.time()-t0:.1f}s", flush=True)
PY

echo "[4/4] Running Clean 240x <think> Chain-of-Thought Evaluation on NVIDIA A100-80GB..."
cd /Users/joshua/evaluation/PiQwen
python3 /Users/joshua/evaluation/PiQwen/hf_clean_240x_think_eval.py

echo "Uploading Clean 240x <think> Evaluation Results & Submissions to HF Hub..."
python3 - <<'PY'
import os, tarfile, huggingface_hub as hf
root = "/Users/joshua/evaluation/PiQwen"
out_tar = "/tmp/clean_240x_think_results.tar.gz"
with tarfile.open(out_tar, "w:gz") as tar:
    for rel in [
        "clean_240x_think_eval_results.json",
        "submissions/humanevalplus_164_polystep_think_samples.jsonl",
        "submissions/mbppplus_378_polystep_think_samples.jsonl",
        "submissions/ProbioticFarmer--Qwen3.8-27B-BarronPolyStep-RealCoder-v3--main--bigcodebench-hard-instruct--custom-0-1-sanitized_calibrated.jsonl",
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
        path_in_repo="clean_240x_think_results.tar.gz",
        repo_id=repo,
        repo_type="model",
        commit_message="Upload Clean 240x <think> Chain-of-Thought A100-80GB evaluation results",
    )
    for rel in [
        "clean_240x_think_eval_results.json",
        "submissions/humanevalplus_164_polystep_think_samples.jsonl",
        "submissions/mbppplus_378_polystep_think_samples.jsonl",
        "submissions/ProbioticFarmer--Qwen3.8-27B-BarronPolyStep-RealCoder-v3--main--bigcodebench-hard-instruct--custom-0-1-sanitized_calibrated.jsonl",
    ]:
        full_p = os.path.join(root, rel)
        if os.path.exists(full_p):
            api.upload_file(
                path_or_fileobj=full_p,
                path_in_repo=rel,
                repo_id=repo,
                repo_type="model",
                commit_message=f"Upload {rel} from Clean 240x <think> A100-80GB evaluation",
            )
print("Successfully uploaded Clean 240x <think> evaluation results to HF Hub!", flush=True)
PY
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-bundle-only", action="store_true")
    args = parser.parse_args()

    bundle_path = build_bundle()
    if args.build_bundle_only:
        return

    api = hf.HfApi()
    token = hf.get_token()
    assert token, "HF token not found"

    active_job_file = os.path.join(PIQWEN_DIR, "active_hf_think_eval_job.txt")
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
        path_in_repo="clean_240x_think_bundle.tar.gz",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload lightweight Clean 240x <think> CoT evaluation bundle for A100-80GB",
    )
    print("[Upload] Bundle uploaded.", flush=True)

    job = api.run_job(
        image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        command=["bash", "-c", REMOTE_SCRIPT],
        flavor="a100-large",
        timeout="2h",
        env={
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
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
    with open(os.path.join(PIQWEN_DIR, "active_hf_think_eval_job.txt"), "w", encoding="utf-8") as f:
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
        print("[Download] Fetching clean_240x_think_results.tar.gz from HF Hub...", flush=True)
        res_tar = hf.hf_hub_download(
            repo_id=REPO_ID,
            filename=RESULTS_TAR_NAME,
            repo_type="model",
            force_download=True,
        )
        with tarfile.open(res_tar, "r:gz") as tar:
            tar.extractall(PIQWEN_DIR)
        print(f"[Complete] Extracted Clean 240x <think> results into {PIQWEN_DIR}!", flush=True)


if __name__ == "__main__":
    main()
