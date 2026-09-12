#!/usr/bin/env python3
"""Unified V3 extension runner for the ARR study.

DESIGN CONSTRAINT: the copied frozen runners
  code/run_structured_extraction_v2.py        (InternVL3 adapter + all shared machinery)
  code/run_structured_extraction_gemma4_v2.py (Gemma-4 adapter)
are NOT edited. This module imports them and reuses their frozen functions verbatim:
  extract_json, schema_valid, repetition_degenerate, build_prompt, mode_description,
  base_record, write_record, read_manifest, manifest_sha256, read_text, InternVL3Adapter.
Their SHA256 values are re-verified at startup and recorded in every output record.

What this module adds, and nothing else:
  * --prompt-prefix    select the frozen P0 / P1 / P2 prompt files (content unchanged)
  * --modes / --tasks  restrict the condition grid (e.g. MM-only for Stages 2 and 4)
  * stochastic serving (--do-sample/--temperature/--top-p) with explicit, fully documented
    per-record RNG seeding
  * resumability       completed records are detected and never regenerated or duplicated
  * a Qwen3-VL adapter for the new model family
  * --max-tiles / --image-longest-edge for the Stage 9 visual-budget audit

No prompt text, no parser, no schema validator and no metric is altered here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

# SHA256 of the path-sanitized public runner copies. Original execution hashes are recorded under provenance/.
FROZEN_RUNNER_HASHES = {
    "run_structured_extraction_v2.py":
        "730838cc4a25172a97ea9007d3fef8d8e5f70ade3df29f529262531046cb07a3",
    "run_structured_extraction_gemma4_v2.py":
        "7ab3dcc87a6ef6b3f675808d06915803bb5f6960834f9a8f9ee6091061ce7512",
}

FROZEN_PROMPT_HASHES = {
    "v2_concepts.txt": "ba990022f28121aa42a5ce4ef737df9f8cdc67314b32e70df54482de601d1611",
    "v2_triples.txt":  "347f7ba57df62c2b92a1e402c8b387dbbbe29c133a9528c322b8b913e9697374",
    "p1_concepts.txt": "b69e893c07821dd4b7d2c8523f85c7a3833eb9316c64632d193632555d47c3ac",
    "p1_triples.txt":  "b391990215bb830ef5e44ce926c87699b1fc547cb4702b4ce28c20ca0fd8ea37",
    "p2_concepts.txt": "1d0ae98764193128f48e837b4e96ab883205b27f49bf82baf11ce90a4c0e3949",
    "p2_triples.txt":  "3f3fca0b19bf7ac4600b29823419c3c80a39e37a499f5d9fd1f6132b77633ae9",
}


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def verify_frozen_artifacts() -> dict[str, str]:
    """Abort rather than run against a mutated runner or prompt."""
    seen = {}
    for name, expect in FROZEN_RUNNER_HASHES.items():
        got = sha256_file(THIS_DIR / name)
        if got != expect:
            raise SystemExit(f"FROZEN RUNNER MODIFIED: {name}\n  expect {expect}\n  got    {got}")
        seen[name] = got
    for name, expect in FROZEN_PROMPT_HASHES.items():
        got = sha256_file(ROOT / "prompts" / name)
        if got != expect:
            raise SystemExit(f"FROZEN PROMPT MODIFIED: {name}\n  expect {expect}\n  got    {got}")
        seen[name] = got
    return seen


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module(THIS_DIR / "run_structured_extraction_v2.py", "milu_v3_base")


# ----------------------------------------------------------------------------------
# RNG seeding. Every RNG on the serving path is seeded explicitly and recorded.
# ----------------------------------------------------------------------------------
def seed_everything(seed: int) -> list[str]:
    seeded = []
    random.seed(seed);                      seeded.append("python:random.seed")
    os.environ["PYTHONHASHSEED"] = str(seed); seeded.append("env:PYTHONHASHSEED")
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32));   seeded.append("numpy:np.random.seed")
    except Exception:
        pass
    import torch
    torch.manual_seed(seed);                seeded.append("torch:manual_seed(CPU generator)")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed);   seeded.append("torch:cuda.manual_seed_all(all devices)")
    try:
        import transformers
        transformers.set_seed(seed)
        seeded.append("transformers:set_seed(random+numpy+torch+torch.cuda)")
    except Exception:
        pass
    return seeded


def derive_seed(base_seed: int, lecture: str, slide_id: str, mode: str, task: str) -> int:
    """Deterministic per-record seed.

    Sampling is reseeded per record from a digest of (base seed, slide, mode, task) so that
    a record's sample depends ONLY on its own identity and the run's seed -- never on how
    many records happened to be generated before it. This makes a stochastic run exactly
    reproducible and exactly resumable, and makes each record independent across conditions.
    A single global seed with sequential sampling would be order-dependent and would silently
    change every downstream record whenever a run were resumed.
    """
    key = f"{base_seed}|{lecture}|{slide_id}|{mode}|{task}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 31 - 1)


# ----------------------------------------------------------------------------------
# Qwen3-VL adapter (new family). Frozen prompts, frozen parser, frozen decoding protocol.
# ----------------------------------------------------------------------------------
class Qwen3VLAdapter:
    ADAPTER_NAME = "Qwen3VLAdapter"
    SERVING_FAMILY = "transformers_Qwen3VLForConditionalGeneration"

    def __init__(self, model_id: str, revision: str, dtype_name: str, device_map: str,
                 image_longest_edge: int | None = None, **_ignored) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        self.image_longest_edge = image_longest_edge

        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, local_files_only=True,
            dtype=self.dtype, device_map=device_map, low_cpu_mem_usage=True).eval()
        print("Qwen3-VL loaded. transformers:", self.transformers_version,
              "| class:", self.model.__class__.__name__,
              "| processor:", self.processor.__class__.__name__,
              "| device:", self.model.device)

    def _image(self, image_path: str):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if self.image_longest_edge:
            w, h = img.size
            s = self.image_longest_edge / max(w, h)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        return img

    def generate(self, prompt: str, generation_config: dict[str, Any],
                 image_path: str | None = None) -> str:
        content = []
        images = None
        if image_path is not None:
            content.append({"type": "image"})
            images = [self._image(image_path)]
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        in_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **dict(generation_config))
        return self.processor.decode(out[0][in_len:], skip_special_tokens=True)


class InternVLWrapper:
    """Thin wrapper around the FROZEN base.InternVL3Adapter (its code is untouched)."""
    ADAPTER_NAME = "InternVL3Adapter"
    SERVING_FAMILY = "transformers_AutoModel_trust_remote_code"

    def __init__(self, model_id: str, revision: str, dtype_name: str, device_map: str,
                 max_tiles: int = 12, use_flash_attn: bool = False,
                 image_longest_edge: int | None = None, **_ignored) -> None:
        self.image_longest_edge = image_longest_edge
        self.max_tiles = max_tiles
        self.inner = base.InternVL3Adapter(
            model_id=model_id, revision=revision, dtype_name=dtype_name,
            device_map=device_map, max_tiles=max_tiles, use_flash_attn=use_flash_attn)
        import transformers
        self.transformers_version = transformers.__version__
        if image_longest_edge:
            self._install_resize_hook(image_longest_edge)

    def _install_resize_hook(self, longest_edge: int) -> None:
        """Stage-9 only: pre-resize the source image before the frozen tiling runs."""
        from PIL import Image as PILImage
        original_open = PILImage.open
        inner = self.inner

        def _load_image(image_path: str):
            img = original_open(image_path).convert("RGB")
            w, h = img.size
            s = longest_edge / max(w, h)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), PILImage.BICUBIC)
            tmp = Path(os.environ.get("V3_TMPDIR", "/tmp")) / "v3_resized.png"
            img.save(tmp)
            return type(inner)._load_image(inner, str(tmp))

        self.inner._load_image = _load_image  # type: ignore[method-assign]

    def generate(self, prompt, generation_config, image_path=None):
        return self.inner.generate(prompt, generation_config, image_path)


def build_adapter(family: str, args):
    if family == "internvl":
        return InternVLWrapper(
            model_id=args.model_id, revision=args.revision, dtype_name=args.dtype,
            device_map=args.device_map, max_tiles=args.max_tiles,
            use_flash_attn=False, image_longest_edge=args.image_longest_edge)
    if family == "gemma4":
        gm = load_module(THIS_DIR / "run_structured_extraction_gemma4_v2.py", "milu_v3_gemma4")
        ad = gm.Gemma4Adapter(model_id=args.model_id, revision=args.revision,
                              dtype_name=args.dtype, device_map=args.device_map)
        ad.ADAPTER_NAME = "Gemma4Adapter"
        ad.SERVING_FAMILY = "transformers_AutoModelForMultimodalLM"
        return ad
    if family == "qwen3vl":
        return Qwen3VLAdapter(
            model_id=args.model_id, revision=args.revision, dtype_name=args.dtype,
            device_map=args.device_map, image_longest_edge=args.image_longest_edge)
    raise ValueError(f"unknown family: {family}")


# ----------------------------------------------------------------------------------
# Resume support
# ----------------------------------------------------------------------------------
def completed_keys(out_dir: Path) -> set[tuple[str, str, str, str]]:
    """Records already generated WITHOUT a generation error. Errors are retried but the
    original failed record is preserved on disk as an audit artifact."""
    path = out_dir / "records.jsonl"
    done: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("generation_error"):
                continue
            done.add((r.get("lecture", ""), r.get("slide_id", ""), r.get("mode", ""),
                      r.get("task", "")))
    return done


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["internvl", "gemma4", "qwen3vl"], required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--condition-name", required=True,
                   help="label for this run, e.g. p1_mm or stoch_seed20260821")
    p.add_argument("--prompt-prefix", default="v2", choices=["v2", "p1", "p2"])
    p.add_argument("--modes", default="MM,TX,IMG")
    p.add_argument("--tasks", default="concepts,triples")
    p.add_argument("--selection-manifest", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/milu1062_manifest.csv")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-tiles", type=int, default=12)
    p.add_argument("--image-longest-edge", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    frozen = verify_frozen_artifacts()
    print("Frozen artifact verification: PASS (%d files)" % len(frozen))

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    assert all(m in base.VALID_MODES for m in modes)
    assert all(t in base.VALID_TASKS for t in tasks)

    if args.do_sample:
        generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": True,
                             "temperature": args.temperature, "top_p": args.top_p}
    else:
        generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                             "temperature": 0.0, "top_p": 1.0}

    manifest_hash = base.manifest_sha256(args.manifest)
    slides = base.read_manifest(args.selection_manifest)
    if args.limit:
        slides = slides[: args.limit]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else completed_keys(out_dir)

    todo = [(s, m, t) for s in slides for m in modes for t in tasks
            if (s.lecture, s.slide_id, m, t) not in done]

    print("=" * 88)
    print("MILU V3 ARR EXTENSION RUNNER")
    print("=" * 88)
    print("condition        :", args.condition_name)
    print("family / model   :", args.family, "/", args.model_id)
    print("revision         :", args.revision)
    print("prompt prefix    :", args.prompt_prefix)
    print("modes / tasks    :", modes, "/", tasks)
    print("generation config:", generation_config)
    print("base seed        :", args.seed)
    print("manifest sha256  :", manifest_hash)
    print("selection        :", args.selection_manifest, f"({len(slides)} slides)")
    print("output dir       :", out_dir)
    print("already complete :", len(done), "| to generate:", len(todo))
    print("=" * 88, flush=True)

    if not todo:
        print("Nothing to do; run already complete.")
        return 0

    seeded_rngs = seed_everything(args.seed)
    print("RNGs seeded at startup:", seeded_rngs, flush=True)

    adapter = build_adapter(args.family, args)

    templates = {t: (ROOT / "prompts" / f"{args.prompt_prefix}_{t}.txt").read_text(encoding="utf-8")
                 for t in tasks}
    prompt_hashes = {t: hashlib.sha256(v.encode()).hexdigest() for t, v in templates.items()}
    print("prompt hashes    :", prompt_hashes, flush=True)

    transcripts: dict[str, str] = {}
    n_done = 0
    for slide, mode, task in todo:
        if slide.text_path not in transcripts:
            transcripts[slide.text_path] = base.read_text(slide.text_path)
        prompt = base.build_prompt(templates[task], mode, transcripts[slide.text_path])

        rec_seed = derive_seed(args.seed, slide.lecture, slide.slide_id, mode, task)
        if args.do_sample:
            seed_everything(rec_seed)

        rec = base.base_record(slide, manifest_hash, args.model_id, args.revision, args.dtype,
                               mode, task, generation_config, args.seed)
        rec["experiment_version"] = "experiment_v3_arr"
        rec["condition_name"] = args.condition_name
        rec["prompt_version"] = args.prompt_prefix
        rec["prompt_file"] = f"{args.prompt_prefix}_{task}.txt"
        rec["prompt_sha256"] = prompt_hashes[task]
        rec["model_adapter"] = getattr(adapter, "ADAPTER_NAME", type(adapter).__name__)
        rec["serving_family"] = getattr(adapter, "SERVING_FAMILY", "")
        rec["transformers_version"] = getattr(adapter, "transformers_version", "")
        rec["decoding"] = "stochastic" if args.do_sample else "deterministic"
        rec["record_seed"] = rec_seed
        rec["seeded_rngs"] = seeded_rngs
        rec["max_tiles"] = args.max_tiles
        rec["image_longest_edge"] = args.image_longest_edge
        rec["frozen_runner_sha256"] = FROZEN_RUNNER_HASHES
        rec["exact_prompt"] = prompt

        image_path = slide.image_path if mode in ("MM", "IMG") else None
        try:
            raw = adapter.generate(prompt, generation_config, image_path)
            parsed_ok, parsed = base.extract_json(raw)
            rec["raw_output"] = raw
            rec["json_parse_success"] = parsed_ok
            rec["schema_valid"] = base.schema_valid(task, parsed)
            rec["parsed_output"] = parsed
            rec["degenerate_repetition"] = base.repetition_degenerate(raw)
        except Exception as exc:
            rec["generation_error"] = repr(exc)
            rec["degenerate_repetition"] = False
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()

        base.write_record(out_dir, rec)
        n_done += 1
        if n_done % 100 == 0:
            print(f"[{datetime.now(timezone.utc).isoformat()}] {n_done}/{len(todo)}", flush=True)

    print("COMPLETED:", n_done, "records ->", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
