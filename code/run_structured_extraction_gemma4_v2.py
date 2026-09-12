#!/usr/bin/env python3
"""
MILU Stage-6 cross-model replication runner: Gemma-4-31B-IT.

Controls preserved from the frozen InternVL experiment:
- same MILU-1062 manifest
- same Pilot-100 manifest
- same prompt files (v2_concepts.txt / v2_triples.txt)
- same MM / TX / IMG definitions
- same concepts / triples schemas
- same JSON extraction and schema validation
- same deterministic decoding protocol
- same max_new_tokens=512
- same seed
- raw generations preserved without repair

Only the model-specific serving adapter changes.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any


# ============================================================
# Import frozen/common experiment machinery
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
BASE_RUNNER = THIS_DIR / "run_structured_extraction_v2.py"

spec = importlib.util.spec_from_file_location("milu_v2_base_gemma4", BASE_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import base runner: {BASE_RUNNER}")

base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


EXPERIMENT_VERSION = "experiment_v2_stage6_gemma4_31b"
PROMPT_VERSION = base.PROMPT_VERSION

DEFAULT_MODEL_ID = "google/gemma-4-31b-it"
DEFAULT_MODEL_REVISION = "145dc2508c480a64b47242f160d286cff94a2343"

DEFAULT_EXPERIMENT_ROOT = Path(
    "external_artifacts/experiment_v2"
)

DEFAULT_MANIFEST = (
    DEFAULT_EXPERIMENT_ROOT / "manifests/milu1062_manifest.csv"
)


# Save original constructor BEFORE monkey-patching it.
_original_base_record = base.base_record


def stage6_base_record(*args, **kwargs):
    rec = _original_base_record(*args, **kwargs)
    rec["experiment_version"] = EXPERIMENT_VERSION
    rec["prompt_version"] = PROMPT_VERSION
    rec["model_adapter"] = "Gemma4Adapter"
    rec["serving_family"] = "transformers_AutoModelForMultimodalLM"
    rec["serving_protocol"] = "deterministic_controlled_extraction"
    rec["native_model_default_sampling_used"] = False
    return rec


base.base_record = stage6_base_record


# ============================================================
# Gemma-4 adapter
# ============================================================

class Gemma4Adapter:
    def __init__(
        self,
        model_id: str,
        revision: str,
        dtype_name: str,
        device_map: str,
    ) -> None:
        import torch
        import transformers
        from transformers import (
            AutoProcessor,
            AutoModelForMultimodalLM,
        )

        self.torch = torch
        self.transformers_version = transformers.__version__

        if dtype_name == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype_name == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

        print("Loading Gemma-4 processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
        )

        print("Loading Gemma-4 model...")
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            dtype=self.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        ).eval()

        print("Gemma-4 loaded.")
        print("Transformers:", self.transformers_version)
        print("Model class:", self.model.__class__.__name__)
        print("Processor class:", self.processor.__class__.__name__)
        print("Requested dtype:", self.dtype)
        print("Model device:", self.model.device)

    def generate(
        self,
        prompt: str,
        generation_config: dict[str, Any],
        image_path: str | None = None,
    ) -> str:

        content = []

        if image_path is not None:
            # Transformers multimodal chat templates support local image paths.
            content.append({
                "type": "image",
                "path": str(Path(image_path).resolve()),
            })

        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [{
            "role": "user",
            "content": content,
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Match the official direct-generation pattern.
        inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(generation_config)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        generated = outputs[0][input_len:]

        return self.processor.decode(
            generated,
            skip_special_tokens=True,
        )


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=["health", "validation5", "pilot100", "full1062"],
        required=True,
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--selection-manifest",
        type=Path,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "results_stage6_gemma4_31b",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
    )

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )

    parser.add_argument(
        "--device-map",
        default="auto",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260815,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    # Frozen controlled-extraction protocol.
    # We intentionally do NOT use Gemma-4's native sampling defaults.
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    manifest_hash = base.manifest_sha256(args.manifest)

    print("=" * 90)
    print("MILU STAGE-6 GEMMA-4-31B RUNNER")
    print("=" * 90)
    print("Experiment version:", EXPERIMENT_VERSION)
    print("Prompt version:", PROMPT_VERSION)
    print("Model:", args.model_id)
    print("Revision:", args.revision)
    print("dtype:", args.dtype)
    print("Manifest SHA256:", manifest_hash)
    print("Generation config:", generation_config)
    print("Serving interpretation: controlled deterministic protocol")
    print("=" * 90)

    adapter = Gemma4Adapter(
        model_id=args.model_id,
        revision=args.revision,
        dtype_name=args.dtype,
        device_map=args.device_map,
    )

    # --------------------------------------------------------
    # Basic health prompts
    # --------------------------------------------------------

    health_ok = base.run_health(
        adapter,
        args,
        manifest_hash,
        generation_config,
    )

    # --------------------------------------------------------
    # One-slide six-condition health gate
    # --------------------------------------------------------

    if args.phase == "health":
        if not health_ok:
            print("BASIC HEALTH: FAIL")
            return 2

        slide = base.load_selection(args)[0]

        records = base.run_extraction_records(
            adapter,
            args,
            [slide],
            args.output_root / "health_test",
            manifest_hash,
            generation_config,
        )

        bad = [
            r for r in records
            if r.get("generation_error")
            or r.get("degenerate_repetition")
            or not r.get("json_parse_success")
            or not r.get("schema_valid")
        ]

        print()
        print("=" * 90)
        print("HEALTH SLIDE SUMMARY")
        print("=" * 90)

        for r in records:
            parsed = r.get("parsed_output") or {}

            vals = (
                parsed.get(r["task"], [])
                if isinstance(parsed, dict)
                else []
            )

            print(
                f"{r['mode']:3} {r['task']:8} "
                f"parse={str(r['json_parse_success']):5} "
                f"schema={str(r['schema_valid']):5} "
                f"count={len(vals):2d} "
                f"degenerate={str(r.get('degenerate_repetition', False)):5} "
                f"error={'YES' if r.get('generation_error') else 'NO'}"
            )

        print()
        print("Health extraction records:", len(records), "/ 6")
        print("Failures:", len(bad), "/ 6")

        status = (
            "PASS"
            if len(records) == 6 and len(bad) == 0
            else "FAIL"
        )

        print("STAGE-6 GEMMA-4 TECHNICAL HEALTH:", status)

        return 0 if status == "PASS" else 1

    # --------------------------------------------------------
    # validation5 / pilot100
    # --------------------------------------------------------

    if not health_ok:
        return 2

    slides = base.load_selection(args)
    out_dir = args.output_root / args.phase

    records = base.run_extraction_records(
        adapter,
        args,
        slides,
        out_dir,
        manifest_hash,
        generation_config,
    )

    bad = [
        r for r in records
        if r.get("generation_error")
        or r.get("degenerate_repetition")
        or not r.get("json_parse_success")
        or not r.get("schema_valid")
    ]

    print()
    print("TOTAL EXTRACTION RECORDS:", len(records))
    print("INVALID/FAILED:", len(bad))

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
