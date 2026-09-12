# Recorded execution environments

The study used NVIDIA H100 80 GB GPUs.

## InternVL / Qwen execution environment

Recorded environment:

- Python 3.13.11
- PyTorch 2.9.1
- CUDA 12.8 build
- Transformers 4.57.6
- torchvision 0.24.1
- Pillow 11.3.0
- NumPy 2.2.6
- Accelerate 1.12.0

Qwen3-VL completion logs recorded:

- model class: `Qwen3VLForConditionalGeneration`
- processor class: `Qwen3VLProcessor`
- Transformers 4.57.6

## Gemma-4 environment

Gemma-4 was executed in a separately frozen environment using the Transformers 5.x multimodal API.

The model revision, prompts, generation parameters, schemas, and output records are pinned independently of the environment location.

## Hardware note

Absolute machine paths and host-specific environment locations are intentionally omitted from the anonymous public release.
