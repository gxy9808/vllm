# Step-4 minimal inference

> 中文版本：[README.zh.md](README.zh.md)

This directory contains the standalone Step-4 inference path validated on one
node with **8 x NVIDIA H200** and tensor parallelism **TP=8**.

The default sliding-attention path is PyTorch SDPA. QKNorm/RoPE, DSA sparse
attention, block-scaled FP8 expert GEMM, and the critical MoE primitives use
Triton/CUDA; model orchestration and NCCL collectives remain PyTorch.

The same eight ranks are used in two roles: the dense backbone is tensor
parallel (`TP=8`), while the 352 routed experts are expert-sharded across those
ranks (`EP=8`, 44 experts/rank). This is a co-located TP/EP topology, not an
independent `TP × EP` process grid.

## Contents

| File | What it is |
|------|------------|
| `config.json` | The validated TP=8 reference model configuration (Step-4's own field names, `gqa-provider-shared-tp-v2` layout marker). |
| `convert.py` | Converts an original Step-4 checkpoint into per-rank TP shards: `tp8/model-r{0..7}.safetensors` + matching `config-r{0..7}.json`, plus a shared `tokenizer_files/`. |
| `generate.py` | The greedy inference entry point. Loads one shard per `torchrun` rank and runs `Step4ForCausalLM.generate_greedy`; takes a text `--prompt`, or exact token ids via `--prompt-json` / `--prompt-json-batch`. |
| `model.py` | The Step-4 model definition — configuration, decoder layers, attention (DSA on the full-attention layers, SDPA on the sliding layers), and the TP/EP layer wiring. Imports its compute kernels from `kernel.py`. Generated file; do not edit by hand. |
| `kernel.py` | The Triton/CUDA compute kernels: the sparse-attention (DSA) indexer + decode/prefill, block-scaled FP8 expert GEMM, the MoE primitives, and fused QKNorm/RoPE. Generated file; do not edit by hand. |
| `requirements.txt` | Pinned runtime dependencies (torch 2.10.0, transformers 4.57.6, safetensors 0.7.0, triton 3.6.0). |

Typical flow: `convert.py` (once, to shard a checkpoint) → `generate.py` (to run it).
`model.py` + `kernel.py` are the model itself and are imported by `generate.py`.

## Install

Use a CUDA-enabled PyTorch build matching the host driver. The pinned baseline
was PyTorch 2.10.0+cu128, Triton 3.6.0, Transformers 4.57.6, and Safetensors
0.7.0; the release was also regression-tested on the clean cu129 image recorded
in the [evaluation environment manifest](../evaluation/ENVIRONMENT.md).

The release loads the fast tokenizer serialized in local `tokenizer.json` directly.
Transformers 5.15 otherwise maps the legacy tokenizer class metadata to an incompatible
Llama implementation that can collapse long prompts. Direct loading preserves the
byte-level BPE input IDs and output decoding validated with Transformers 4.57.6.

```bash
pip install -r requirements.txt
```

## Convert the checkpoint

Run from this directory. Conversion writes one weight and config file per rank
under `SAVE_PATH/tp8` and copies tokenizer assets to
`SAVE_PATH/tokenizer_files`.

```bash
export HF_CKPT_PATH=/path/to/Step-4
export SAVE_PATH=/path/to/Step-4-TP8

python3 convert.py \
  --checkpoint "$HF_CKPT_PATH" \
  --out-dir "$SAVE_PATH" \
  --tp-size 8 \
  --ep-size 8
```

`config.json` is the validated TP=8 reference configuration. `convert.py`
reads the source checkpoint configuration and writes matching
`config-r{0..7}.json` files with the required
`gqa-provider-shared-tp-v2` layout marker and
`expert_parallel_size=8`; do not mix them with weights from a different
checkpoint or an older TP layout. The byte-pinned reference `config.json`
predates that explicit EP metadata, so `generate.py` treats a missing
`expert_parallel_size` as equal to `tp_size`.

## Generate

The tokenizer copied by conversion is discovered automatically.

```bash
torchrun --standalone --nproc-per-node=8 generate.py \
  --tp-dir "$SAVE_PATH/tp8" \
  --ep-size 8 \
  --prompt "请介绍一下张量并行。" \
  --max-new-tokens 128
```

For exact token-id input, pass a JSON list or `{"prompt_ids": [...]}`:

```bash
torchrun --standalone --nproc-per-node=8 generate.py \
  --tp-dir "$SAVE_PATH/tp8" \
  --ep-size 8 \
  --prompt-json /path/to/prompt.json \
  --max-new-tokens 128
```

## Limitations

- Only single-node, co-located `TP=EP=8` on 8 x H200 has been validated.
- `--ep-size` defaults to 8 and currently must equal TP/world size; independent
  TP and EP process groups such as `TP=4, EP=2` are not implemented.
  `generate.py` reads TP from the shard config (and checks it against
  `WORLD_SIZE`); it intentionally has no separate `--tp-size` switch.
- Sliding attention uses the validated SDPA path; the experimental pure-Triton
  sliding backend is intentionally not included in this minimal release.
- Generation is greedy. Continuous dynamic batching, paged KV cache, prefix
  caching, CUDA graphs, and multi-node execution are not provided.
- The MTP layer is not loaded, so speculative decoding is unavailable.
- The checkpoint's factory FP8/BF16 expert layout is consumed directly; the
  converter and generated TP shards are checkpoint-layout specific.
