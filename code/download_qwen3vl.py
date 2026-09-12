#!/usr/bin/env python3
"""Fetch the frozen Qwen3-VL-32B-Instruct revision authorised in DECISIONS.md D-003.1.

Pinned revision only. Weight files are restricted to bf16 safetensors; GGUF/FP8 variants
are never fetched because they would change numerics relative to the bf16 baselines.
Re-running is safe: the hub cache resumes and already-complete files are not re-fetched.
"""
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"

path = snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
    max_workers=8,
)
print("SNAPSHOT:", path)
