#!/usr/bin/env python3
"""Measure batch-one, uncached next-position TurnGPT scoring latency on CUDA.

Inputs are fixed-length real cached windows and are resident on the GPU before
timing. Each request includes embeddings, a full-context forward pass, the last
position's vocabulary projection, and a scalar <ts> probability returned to CPU.
Loading, tokenization, input transfer and warmup are excluded. No memory metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/processed/qwen3_turngpt_verified_8dataset_v2"
TIMING_SCOPE = "gpu_resident_inputs_to_cpu_next_ts_probability_no_kv_cache"


def select_samples(data_dir, context_length, windows_per_dataset, seed):
    """Choose real full-length windows, preserving corpus identity and content hash."""
    if context_length < 1 or windows_per_dataset < 1:
        raise ValueError("Context length and windows per dataset must be positive")
    raw_manifest = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    expected = manifest["splits"]["test"]
    groups = {name: [] for name in sorted(expected)}
    counts = Counter()
    token_ids = manifest["special_token_ids"]
    with (data_dir / "test.jsonl").open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            name = row["dataset"]
            if row["source_split"] != "test" or name not in groups:
                raise ValueError(f"Unexpected corpus/split at line {line_number}")
            counts[name] += 1
            ids, states = row["input_ids"], row["speaker_ids"]
            if len(ids) != len(states) or not ids:
                raise ValueError(f"Misaligned input/speaker sequence at line {line_number}")
            if any(t < 0 or t > token_ids["ts"] for t in ids):
                raise ValueError(f"Unexpected text token at line {line_number}")
            if not set(states) <= {token_ids["speaker1"], token_ids["speaker2"]}:
                raise ValueError(f"Unexpected speaker token at line {line_number}")
            if len(ids) == context_length:
                groups[name].append({"dataset": name, "jsonl_line": line_number,
                                     "input_ids": ids, "speaker_ids": states})
    if dict(counts) != {n: d["windows"] for n, d in expected.items()}:
        raise ValueError("Test window counts do not match the manifest")
    selected = []
    for name, rows in groups.items():
        rng = random.Random(f"{seed}:{name}:latency:{context_length}")
        indices = sorted(rng.sample(range(len(rows)), min(windows_per_dataset, len(rows))))
        selected.extend(rows[i] for i in indices)
    if not selected:
        raise ValueError(f"No real test windows have exactly {context_length} tokens")
    random.Random(f"{seed}:latency:order").shuffle(selected)
    for sample_id, row in enumerate(selected):
        row["sample_id"] = sample_id
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return selected, {
        "data_dir": str(data_dir.resolve()),
        "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "input_samples_sha256": digest,
        "sample_count": len(selected),
        "eligible_windows_by_corpus": {n: len(rows) for n, rows in groups.items()},
        "sampled_windows_by_corpus": dict(sorted(Counter(r["dataset"] for r in selected).items())),
        "special_token_ids": token_ids,
    }


def percentile(values, q):
    """Linear interpolation at (n - 1) * q; identical to NumPy's linear method."""
    if not values or not 0 <= q <= 1:
        raise ValueError("Percentile needs nonempty values and q in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(latencies):
    if not latencies or any(not math.isfinite(x) or x <= 0 for x in latencies):
        raise ValueError("Latency samples must be finite positive milliseconds")
    return {"requests": len(latencies), "mean_ms": statistics.fmean(latencies),
            "p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95)}


def next_ts_probability(model, input_ids, speaker_ids, attention_mask, ts_id):
    embeddings = model.get_input_embeddings()
    # All context tokens pass through the transformer. Only the last hidden state
    # needs a vocabulary projection for a single next-position boundary decision.
    logits = model(inputs_embeds=embeddings(input_ids) + embeddings(speaker_ids),
                   attention_mask=attention_mask, use_cache=False,
                   logits_to_keep=1).logits[:, -1, :].float()
    return (logits[:, ts_id] - torch.logsumexp(logits, dim=-1)).exp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--windows-per-dataset", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("Warmup and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this latency benchmark on the CUDA worker")
    samples, data_info = select_samples(args.data_dir, args.context_length,
                                        args.windows_per_dataset, args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    token_ids = {"ts": tok.convert_tokens_to_ids("<ts>"),
                 "speaker1": tok.convert_tokens_to_ids("<speaker1>"),
                 "speaker2": tok.convert_tokens_to_ids("<speaker2>")}
    if token_ids != data_info["special_token_ids"]:
        raise ValueError("Checkpoint tokenizer does not match cached inputs")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    if model.config.eos_token_id != token_ids["ts"]:
        raise ValueError("Expected a saved TurnGPT checkpoint, not a Base model")
    inputs = [(torch.tensor(r["input_ids"], device=device).unsqueeze(0),
               torch.tensor(r["speaker_ids"], device=device).unsqueeze(0),
               torch.ones((1, args.context_length), dtype=torch.long, device=device))
              for r in samples]
    print(json.dumps({"event": "warmup", "checkpoint": str(args.checkpoint),
                      "samples": len(inputs), "warmup_requests": args.warmup}), flush=True)
    measurements = []
    with torch.inference_mode():
        for i in range(args.warmup):
            next_ts_probability(model, *inputs[i % len(inputs)], token_ids["ts"]).item()
        torch.cuda.synchronize(device)
        for repeat in range(args.repeats):
            for sample, tensors in zip(samples, inputs):
                # Previous work must be complete before the host clock starts.
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                score = next_ts_probability(model, *tensors, token_ids["ts"]).item()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise RuntimeError("Non-finite or invalid TurnGPT score")
                measurements.append({"repeat": repeat + 1, "sample_id": sample["sample_id"],
                                     "dataset": sample["dataset"], "latency_ms": elapsed_ms,
                                     "ts_probability": score})
            print(json.dumps({"event": "repeat_complete", "repeat": repeat + 1,
                              **summarize([m["latency_ms"] for m in measurements
                                           if m["repeat"] == repeat + 1])}), flush=True)
    gpu = torch.cuda.get_device_properties(device)
    report = {
        "status": "ok", "schema_version": 1, "benchmark": "turngpt_next_position_latency",
        "checkpoint": str(args.checkpoint.resolve()), **data_info,
        "context_length": args.context_length, "batch_size": 1,
        "windows_per_dataset": args.windows_per_dataset, "seed": args.seed,
        "warmup_requests": args.warmup, "repeats": args.repeats,
        "dtype": "bfloat16", "device": str(device), "gpu_name": gpu.name,
        "gpu_uuid": str(getattr(gpu, "uuid", "unavailable")),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda, "attention_implementation": model.config._attn_implementation,
        "torch_num_threads": torch.get_num_threads(), "tf32": False,
        "use_cache": False, "logits_to_keep": 1,
        "timing_scope": TIMING_SCOPE, "timer": "perf_counter_ns; synchronize before; scalar .item() before stop",
        "excluded": ["model_loading", "tokenization", "disk_io", "input_host_to_device", "warmup"],
        "percentile_method": "linear_interpolation_at_(n-1)*q",
        "summary": summarize([m["latency_ms"] for m in measurements]),
        "per_repeat": [dict(repeat=i, **summarize([m["latency_ms"] for m in measurements
                                                  if m["repeat"] == i]))
                       for i in range(1, args.repeats + 1)],
        "sample_refs": [{k: r[k] for k in ("sample_id", "dataset", "jsonl_line")} for r in samples],
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({"event": "saved", "output": str(args.output), **report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
