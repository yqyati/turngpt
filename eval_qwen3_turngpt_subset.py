#!/usr/bin/env python3
"""Evaluate per-corpus Qwen3 TurnGPT language-model loss and perplexity.

Use scripts/eval_loss.sh for the eight-corpus, 500-step baseline inputs.
Turn-boundary bAcc is evaluated separately by eval_qwen3_turngpt_paper_metrics.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen3_turngpt_subset import (
    DEFAULT_DATA, forward, initialize_turngpt_embeddings, make_collator,
)
from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS, resolve_dtype


ROOT = Path(__file__).parent
# Preserve the historical CLI default; scripts/ selects the 500-step checkpoint.
DEFAULT_CHECKPOINT = None


def load_groups(path: Path, manifest: dict, split: str):
    groups = {name: [] for name in manifest["splits"][split]}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["source_split"] != split or row["dataset"] not in groups:
                raise ValueError("Unexpected split or dataset in evaluation data")
            groups[row["dataset"]].append((row["input_ids"], row["speaker_ids"]))
    for name, rows in groups.items():
        if len(rows) != manifest["splits"][split][name]["windows"]:
            raise ValueError(f"Window count mismatch for {name}/{split}")
    return groups


@torch.no_grad()
def evaluate_group(model, rows, batch_size, collate, device):
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    weighted_loss = 0.0
    target_tokens = 0
    turn_targets = 0
    for batch in loader:
        labels = batch[3]
        # Causal LM shifts labels internally; position zero has no target.
        valid = labels[:, 1:] != -100
        n_targets = int(valid.sum())
        if n_targets == 0:
            continue
        turn_targets += int(((labels[:, 1:] == model.config.eos_token_id) & valid).sum())
        loss = forward(model, batch, device).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss")
        weighted_loss += float(loss.detach().cpu()) * n_targets
        target_tokens += n_targets
    if target_tokens == 0:
        raise RuntimeError("No target tokens to evaluate")
    mean_loss = weighted_loss / target_tokens
    return {
        "windows": len(rows), "target_tokens": target_tokens,
        "turn_targets": turn_targets, "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=None,
                        help="Evaluate freshly initialized TurnGPT tokens on this Base model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--windows-per-dataset", type=int, default=10000000,
                        help="Seeded sample per corpus; 0 evaluates every window.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-seed", type=int, default=1,
                        help="Base initialization seed; match the training run's seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.checkpoint or args.base_model):
        parser.error("Provide --checkpoint or --base-model")
    if args.windows_per_dataset < 0 or args.batch_size < 1:
        raise ValueError("windows-per-dataset must be nonnegative and batch-size positive")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    raw_manifest = (args.data_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["scope"] != "three_corpus_written_subset":
        raise ValueError("Expected a compatible prepared cache")
    groups = load_groups(args.data_dir / f"{args.split}.jsonl", manifest, args.split)
    model_path = args.base_model if args.base_model is not None else args.checkpoint
    if args.base_model is not None:
        torch.manual_seed(args.init_seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.base_model is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
    ids = {"ts": tokenizer.convert_tokens_to_ids("<ts>"),
           "speaker1": tokenizer.convert_tokens_to_ids(SPEAKER1),
           "speaker2": tokenizer.convert_tokens_to_ids(SPEAKER2)}
    if ids != manifest["special_token_ids"]:
        raise ValueError("Checkpoint tokenizer does not match prepared data")
    collate = make_collator(tokenizer.pad_token_id, ids["speaker1"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=resolve_dtype(args.dtype)
    )
    if args.base_model is not None:
        initialize_turngpt_embeddings(model, tokenizer, ids)
    model.to(device)
    if model.config.eos_token_id != ids["ts"]:
        raise ValueError("Checkpoint model EOS does not match <ts>")
    model.eval()

    results = {}
    total_weighted_loss = 0.0
    total_targets = 0
    for name, rows in groups.items():
        if args.windows_per_dataset:
            rng = random.Random(f"{args.seed}:{name}:{args.split}")
            indices = sorted(rng.sample(range(len(rows)), min(args.windows_per_dataset, len(rows))))
            rows = [rows[i] for i in indices]
        result = evaluate_group(model, rows, args.batch_size, collate, device)
        results[name] = result
        total_weighted_loss += result["mean_loss"] * result["target_tokens"]
        total_targets += result["target_tokens"]
        print(json.dumps({"dataset": name, **result}), flush=True)

    report = {
        "status": "ok", "model": str(model_path.resolve()),
        "model_kind": "initialized_base" if args.base_model is not None else "checkpoint",
        "init_seed": args.init_seed if args.base_model is not None else None,
        "split": args.split, "windows_per_dataset": args.windows_per_dataset,
        "batch_size": args.batch_size, "seed": args.seed, "device": device,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "datasets": results,
        "overall_token_weighted_loss": total_weighted_loss / total_targets,
        "overall_target_tokens": total_targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "saved", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
