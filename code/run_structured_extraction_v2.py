#!/usr/bin/env python3
"""Clean structured extraction runner for MILU experiment_v2.

This runner intentionally does not modify or depend on the historical R1C3
scripts. It preserves every raw generation and stores parsing/schema status as
separate fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_VERSION = "experiment_v2_stage3"
PROMPT_VERSION = "v2"
DEFAULT_MODEL_ID = "OpenGVLab/InternVL3-14B"
DEFAULT_MODEL_REVISION = "419aa10d2db7da6c64382ad3124f79a60cec42aa"
DEFAULT_MANIFEST_HASH = "818bfc2900b6da99604499a55c2a294988ee8ee66a251a4ce3675dd41bed1e3e"
VALID_MODES = ("TX", "MM", "IMG")
VALID_TASKS = ("concepts", "triples")


@dataclass(frozen=True)
class Slide:
    lecture: str
    slide_id: str
    image_path: str
    text_path: str
    image_sha256: str
    text_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def read_manifest(path: Path) -> list[Slide]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [Slide(**row) for row in rows]


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True)


def extract_json(raw: str) -> tuple[bool, Any]:
    text = (raw or "").strip()
    if not text:
        return False, None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        return True, json.loads(candidate)
    except Exception:
        return False, None


def schema_valid(task: str, parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if task == "concepts":
        values = parsed.get("concepts")
        if not isinstance(values, list):
            return False
        return all(
            isinstance(item, dict)
            and isinstance(item.get("term"), str)
            and isinstance(item.get("category"), str)
            for item in values
        )
    if task == "triples":
        values = parsed.get("triples")
        if not isinstance(values, list):
            return False
        return all(
            isinstance(item, dict)
            and isinstance(item.get("subject"), str)
            and isinstance(item.get("predicate"), str)
            and isinstance(item.get("object"), str)
            for item in values
        )
    return False


def mode_description(mode: str) -> str:
    if mode == "TX":
        return "transcript only"
    if mode == "IMG":
        return "slide image only"
    if mode == "MM":
        return "slide image and transcript"
    raise ValueError(f"unknown mode: {mode}")


def build_prompt(prompt_template: str, mode: str, transcript: str) -> str:
    if mode == "TX":
        block = "Transcript:\n" + transcript.strip()
    elif mode == "IMG":
        block = "Slide image is provided. No transcript content is provided."
    elif mode == "MM":
        block = "Slide image is provided.\n\nTranscript:\n" + transcript.strip()
    else:
        raise ValueError(f"unknown mode: {mode}")
    return (
        prompt_template.replace("<<MODE_DESCRIPTION>>", mode_description(mode))
        .replace("<<INPUT_BLOCK>>", block)
        .strip()
    )


def load_prompt(root: Path, task: str) -> str:
    return (root / "prompts" / f"v2_{task}.txt").read_text(encoding="utf-8")


def repetition_degenerate(raw: str) -> bool:
    if not raw:
        return False
    if "!!!!!!!!!!" in raw:
        return True
    tokens = re.findall(r"\S+", raw)
    if len(tokens) >= 30:
        most_common = max(tokens.count(tok) for tok in set(tokens))
        return most_common / len(tokens) >= 0.6
    return False


class InternVL3Adapter:
    def __init__(
        self,
        model_id: str,
        revision: str,
        dtype_name: str,
        device_map: str,
        max_tiles: int,
        use_flash_attn: bool,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dtype_name = dtype_name
        self.dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        self.max_tiles = max_tiles
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        kwargs = {
            "revision": revision,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if use_flash_attn:
            kwargs["use_flash_attn"] = True
        self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()

    def _load_image(self, image_path: str):
        import torchvision.transforms as T
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        image_size = 448
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        tiles = self._dynamic_preprocess(image, image_size=image_size, max_num=self.max_tiles)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = self.torch.stack(pixel_values).to(self.dtype)
        if self.torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        return pixel_values

    @staticmethod
    def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @classmethod
    def _dynamic_preprocess(cls, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = cls._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def generate(self, prompt: str, generation_config: dict[str, Any], image_path: str | None = None) -> str:
        pixel_values = self._load_image(image_path) if image_path else None
        query = "<image>\n" + prompt if image_path else prompt
        with self.torch.inference_mode():
            return self.model.chat(self.tokenizer, pixel_values, query, generation_config)


def base_record(
    slide: Slide | None,
    manifest_hash: str,
    model_id: str,
    model_revision: str,
    model_dtype: str,
    mode: str,
    task: str,
    generation_config: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_manifest_hash": manifest_hash,
        "lecture": slide.lecture if slide else "",
        "slide_id": slide.slide_id if slide else "",
        "image_path": slide.image_path if slide else "",
        "text_path": slide.text_path if slide else "",
        "model_id": model_id,
        "model_revision": model_revision,
        "model_dtype": model_dtype,
        "mode": mode,
        "task": task,
        "prompt_version": PROMPT_VERSION,
        "generation_config": generation_config,
        "random_seed": random_seed,
        "raw_output": "",
        "json_parse_success": False,
        "schema_valid": False,
        "parsed_output": None,
        "generation_error": "",
        "timestamp": utc_now(),
    }


def write_record(out_dir: Path, record: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name_parts = [
        record["lecture"] or "health",
        record["slide_id"] or record["task"],
        record["mode"],
        record["task"],
    ]
    path = out_dir / ("__".join(name_parts) + ".json")
    path.write_text(json_dumps(record) + "\n", encoding="utf-8")
    with (out_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def run_health(adapter: InternVL3Adapter, args, manifest_hash: str, generation_config: dict[str, Any]) -> bool:
    out_dir = args.output_root / "health_test"
    ok = True
    health_prompts = [
        ("hello", "Respond with exactly: HELLO"),
        ("arithmetic", "What is 2 + 2? Answer briefly."),
    ]
    for task, prompt in health_prompts:
        rec = base_record(
            None,
            manifest_hash,
            args.model_id,
            args.revision,
            args.dtype,
            "TX",
            task,
            generation_config,
            args.seed,
        )
        rec["exact_prompt"] = prompt
        try:
            raw = adapter.generate(prompt, generation_config, None)
            rec["raw_output"] = raw
            rec["degenerate_repetition"] = repetition_degenerate(raw)
            if not raw.strip() or rec["degenerate_repetition"]:
                ok = False
        except Exception as exc:
            rec["generation_error"] = repr(exc)
            ok = False
        write_record(out_dir, rec)
    return ok


def run_extraction_records(
    adapter: InternVL3Adapter,
    args,
    slides: list[Slide],
    out_dir: Path,
    manifest_hash: str,
    generation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    prompt_root = args.experiment_root
    records = []
    for slide in slides:
        transcript = read_text(slide.text_path)
        for mode in VALID_MODES:
            for task in VALID_TASKS:
                template = load_prompt(prompt_root, task)
                prompt = build_prompt(template, mode, transcript)
                rec = base_record(
                    slide,
                    manifest_hash,
                    args.model_id,
                    args.revision,
                    args.dtype,
                    mode,
                    task,
                    generation_config,
                    args.seed,
                )
                rec["exact_prompt"] = prompt
                image_path = slide.image_path if mode in ("MM", "IMG") else None
                try:
                    raw = adapter.generate(prompt, generation_config, image_path)
                    parsed_ok, parsed = extract_json(raw)
                    rec["raw_output"] = raw
                    rec["json_parse_success"] = parsed_ok
                    rec["schema_valid"] = schema_valid(task, parsed)
                    rec["parsed_output"] = parsed
                    rec["degenerate_repetition"] = repetition_degenerate(raw)
                except Exception as exc:
                    rec["generation_error"] = repr(exc)
                    rec["degenerate_repetition"] = False
                write_record(out_dir, rec)
                records.append(rec)
    return records


def load_selection(args) -> list[Slide]:
    manifest = read_manifest(args.selection_manifest or args.manifest)
    if args.phase == "health":
        return manifest[:1]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["health", "validation5", "pilot100", "full1062"], required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("external_artifacts/experiment_v2"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/milu1062_manifest.csv"))
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--no-flash-attn", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    manifest_hash = manifest_sha256(args.manifest)

    adapter = InternVL3Adapter(
        model_id=args.model_id,
        revision=args.revision,
        dtype_name=args.dtype,
        device_map=args.device_map,
        max_tiles=args.max_tiles,
        use_flash_attn=not args.no_flash_attn,
    )

    health_ok = run_health(adapter, args, manifest_hash, generation_config)
    if args.phase == "health":
        if not health_ok:
            return 2
        slide = load_selection(args)[0]
        run_extraction_records(adapter, args, [slide], args.output_root / "health_test", manifest_hash, generation_config)
        return 0
    if not health_ok:
        return 2

    slides = load_selection(args)
    out_dir = args.output_root / args.phase
    records = run_extraction_records(adapter, args, slides, out_dir, manifest_hash, generation_config)
    bad = [
        rec
        for rec in records
        if rec.get("generation_error") or rec.get("degenerate_repetition") or not rec.get("json_parse_success") or not rec.get("schema_valid")
    ]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
