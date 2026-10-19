#!/usr/bin/env python3
"""Measure next-token <ts> AP, ROC-AUC and Brier on sampled held-out windows.

Use scripts/eval_turn_shift.sh for the eight-corpus, 500-step test inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_qwen3_turngpt_subset import DEFAULT_CHECKPOINT, load_groups
from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS, resolve_dtype
from train_qwen3_turngpt_subset import DEFAULT_DATA, initialize_turngpt_embeddings, make_collator


def average_precision(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    positives = sum(labels)
    if not positives:
        raise ValueError("No positive turn boundaries")
    seen_positive = 0
    precision_sum = 0.0
    rank = 0
    while rank < len(pairs):
        end = rank + 1
        while end < len(pairs) and pairs[end][0] == pairs[rank][0]:
            end += 1
        new_positive = sum(label for _, label in pairs[rank:end])
        seen_positive += new_positive
        precision_sum += new_positive * (seen_positive / end)
        rank = end
    return precision_sum / positives


def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Need both positive and negative labels")
    positive_rank_sum = 0.0
    rank = 1
    while rank <= len(pairs):
        end = rank
        while end < len(pairs) and pairs[end][0] == pairs[rank - 1][0]:
            end += 1
        mean_rank = (rank + end) / 2
        positive_rank_sum += mean_rank * sum(label for _, label in pairs[rank - 1:end])
        rank = end + 1
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@torch.no_grad()
def evaluate_group(model, rows, collate, batch_size, device, ts_id):
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    scores = []
    labels = []
    model.eval()
    for input_ids, speaker_ids, attention_mask, _ in loader:
        input_ids = input_ids.to(device)
        speaker_ids = speaker_ids.to(device)
        attention_mask = attention_mask.to(device)
        embeddings = model.get_input_embeddings()
        logits = model(
            inputs_embeds=embeddings(input_ids) + embeddings(speaker_ids),
            attention_mask=attention_mask, use_cache=False,
        ).logits[:, :-1].float()
        valid = attention_mask[:, 1:].bool()
        ts_logits = logits[..., ts_id]
        probabilities = (ts_logits - torch.logsumexp(logits, dim=-1)).exp()
        scores.extend(probabilities[valid].cpu().tolist())
        labels.extend((input_ids[:, 1:][valid] == ts_id).int().cpu().tolist())
    positives = sum(labels)
    threshold_positive = sum(score >= 0.5 for score in scores)
    true_positive = sum(score >= 0.5 and label for score, label in zip(scores, labels))
    return {
        "windows": len(rows),
        "prediction_positions": len(labels),
        "turn_boundaries": positives,
        "turn_rate": positives / len(labels),
        "average_precision": average_precision(scores, labels),
        "roc_auc": roc_auc(scores, labels),
        "brier_score": sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels),
        "precision_at_0_5": true_positive / threshold_positive if threshold_positive else 0.0,
        "recall_at_0_5": true_positive / positives,
        "predicted_boundaries_at_0_5": threshold_positive,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--windows-per-dataset", type=int, default=10000000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.checkpoint or args.base_model):
        parser.error("Provide --checkpoint or --base-model")
    if args.windows_per_dataset < 1 or args.batch_size < 1:
        raise ValueError("windows-per-dataset and batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    raw_manifest = (args.data_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["scope"] != "three_corpus_written_subset":
        raise ValueError("Unexpected dataset scope")
    groups = load_groups(args.data_dir / f"{args.split}.jsonl", manifest, args.split)
    model_path = args.base_model if args.base_model is not None else args.checkpoint
    if args.base_model is not None:
        torch.manual_seed(args.init_seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.base_model is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
    ids = {
        "ts": tokenizer.convert_tokens_to_ids("<ts>"),
        "speaker1": tokenizer.convert_tokens_to_ids(SPEAKER1),
        "speaker2": tokenizer.convert_tokens_to_ids(SPEAKER2),
    }
    if ids != manifest["special_token_ids"]:
        raise ValueError("Tokenizer and data special token IDs differ")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=resolve_dtype(args.dtype)
    )
    if args.base_model is not None:
        initialize_turngpt_embeddings(model, tokenizer, ids)
    if model.config.eos_token_id != ids["ts"]:
        raise ValueError("Model EOS does not match <ts>")
    model.to(device)
    collate = make_collator(tokenizer.pad_token_id, ids["speaker1"])

    results = {}
    for name, rows in groups.items():
        rng = random.Random(f"{args.seed}:{name}:{args.split}")
        indices = sorted(rng.sample(range(len(rows)), min(args.windows_per_dataset, len(rows))))
        results[name] = evaluate_group(
            model, [rows[i] for i in indices], collate, args.batch_size, device, ids["ts"]
        )
        print(json.dumps({"dataset": name, **results[name]}), flush=True)

    report = {
        "status": "ok", "metric_scope": "token_level_next_ts_on_sampled_windows",
        "model_kind": "initialized_base" if args.base_model is not None else "checkpoint",
        "model": str(model_path.resolve()), "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "split": args.split, "windows_per_dataset": args.windows_per_dataset,
        "batch_size": args.batch_size, "seed": args.seed,
        "init_seed": args.init_seed if args.base_model is not None else None,
        "device": device, "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "saved", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
