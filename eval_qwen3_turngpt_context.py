#!/usr/bin/env python3
"""Evaluate one checkpoint on matched targets at 128/256/512 history limits."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_qwen3_turngpt_latency import next_ts_probability, summarize, TIMING_SCOPE
from eval_qwen3_turngpt_paper_metrics import best_threshold
from turngpt_context_data import LENGTHS, ROOT, file_hash, object_hash
from turngpt_metrics import classification_metrics, aggregate_classification_metrics

DEFAULT_CHECKPOINT = None
DEFAULT_DATA = ROOT / "data/processed/qwen3_turngpt_context_seed1_v1"


def load_samples(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["experiment"] != "matched_context_next_token" or manifest["context_lengths"] != list(LENGTHS):
        raise ValueError("Unexpected context dataset protocol")
    tokens = manifest["special_token_ids"]
    groups = {}
    for split in ("validation", "test", "latency"):
        path = data_dir / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"Context input checksum mismatch: {path}")
        rows = [json.loads(line) for line in path.open()]
        if len(rows) != manifest["files"][path.name]["rows"]:
            raise ValueError("Context row count mismatch")
        seen = set()
        for row in rows:
            ids, states = row["input_ids"], row["speaker_ids"]
            actual_split = "test" if split == "latency" else split
            if row["source_split"] != actual_split or row["dataset"] not in manifest["splits"][actual_split]:
                raise ValueError("Unexpected context corpus/split")
            if row["sample_id"] != object_hash([row["dataset"], actual_split, row["source_id"], row["target_position"]]):
                raise ValueError("Target identity mismatch")
            if row["sample_id"] in seen:
                raise ValueError("Duplicate context target")
            seen.add(row["sample_id"])
            if row["target_position"] != row["available_history_tokens"] or not 1 <= len(ids) <= 512:
                raise ValueError("Invalid target/history length")
            if len(ids) != min(row["available_history_tokens"], 512) or len(ids) != len(states):
                raise ValueError("Truncated history/state alignment error")
            if row["label"] not in (0, 1) or any(not isinstance(t, int) or not 0 <= t <= tokens["ts"] for t in ids):
                raise ValueError("Invalid context label/tokens")
            if not set(states) <= {tokens["speaker1"], tokens["speaker2"]}:
                raise ValueError("Invalid context speaker states")
            if split == "latency" and len(ids) != 512:
                raise ValueError("Latency requires full 512-token source histories")
        groups[split] = rows
    return manifest, groups


def make_batch(rows, length, pad_id, speaker1_id, device):
    """Left pad so the last output predicts the same held-out token in each row."""
    width = max(min(length, len(row["input_ids"])) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    speakers = torch.full_like(ids, speaker1_id)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        text, states = row["input_ids"][-length:], row["speaker_ids"][-length:]
        ids[i, -len(text):] = torch.tensor(text, device=device)
        speakers[i, -len(states):] = torch.tensor(states, device=device)
        mask[i, -len(text):] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return ids, speakers, mask, positions


def batch_probabilities(model, batch, ts_id):
    ids, speakers, mask, positions = batch
    embedding = model.get_input_embeddings()
    logits = model(inputs_embeds=embedding(ids) + embedding(speakers), attention_mask=mask,
                   position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    return (logits[:, ts_id] - torch.logsumexp(logits, dim=-1)).exp()


@torch.inference_mode()
def score_rows(model, rows, length, batch_size, pad_id, token_ids, device):
    # Sorting is only for padding efficiency. Return scores in target-file order.
    order = sorted(range(len(rows)), key=lambda i: min(length, len(rows[i]["input_ids"])))
    scores = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = make_batch([rows[i] for i in indices], length, pad_id, token_ids["speaker1"], device)
        values = batch_probabilities(model, batch, token_ids["ts"]).cpu().tolist()
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise RuntimeError("Invalid context prediction")
        for i, value in zip(indices, values):
            scores[i] = value
    return scores


def metrics_for_subset(rows, scores, thresholds, subset):
    reports, eligible, excluded = {}, [], {}
    for name in thresholds:
        indices = [i for i, row in enumerate(rows) if row["dataset"] == name and subset in row["subsets"]]
        labels = [rows[i]["label"] for i in indices]
        selected_scores = [scores[i] for i in indices]
        positives = sum(labels)
        if not 0 < positives < len(labels):
            excluded[name] = {"targets": len(labels), "positives": positives,
                              "reason": "balanced accuracy requires both classes"}
            continue
        metrics, confusion = classification_metrics(selected_scores, labels, thresholds[name]["threshold"])
        reports[name] = {"prediction_positions": len(labels), "turn_boundaries": positives,
                         "threshold": thresholds[name]["threshold"], "test_confusion_matrix": confusion,
                         **{f"test_{key}": value for key, value in metrics.items()}}
        eligible.append(confusion)
    summary = aggregate_classification_metrics(eligible) if eligible else {}
    return {"datasets": reports, "included_corpora": list(reports), "excluded_corpora": excluded, **summary}


def classification_report(rows, scores):
    thresholds = {}
    for name in sorted({r["dataset"] for r in rows["validation"]}):
        indices = [i for i, row in enumerate(rows["validation"]) if row["dataset"] == name and "primary" in row["subsets"]]
        values = [scores["validation"][i] for i in indices]
        labels = [rows["validation"][i]["label"] for i in indices]
        threshold, bacc = best_threshold(values, labels)
        thresholds[name] = {"threshold": threshold, "validation_bacc": bacc,
                            "targets": len(labels), "positives": sum(labels)}
    return {"thresholds": thresholds,
            "primary": metrics_for_subset(rows["test"], scores["test"], thresholds, "primary"),
            "history_gt256": metrics_for_subset(rows["test"], scores["test"], thresholds, "history_gt256")}


def measure_latency(model, rows, length, ts_id, device, warmup, repeats):
    inputs = [(torch.tensor(row["input_ids"][-length:], device=device).unsqueeze(0),
               torch.tensor(row["speaker_ids"][-length:], device=device).unsqueeze(0),
               torch.ones((1, length), dtype=torch.long, device=device)) for row in rows]
    measurements = []
    with torch.inference_mode():
        for i in range(warmup):
            next_ts_probability(model, *inputs[i % len(inputs)], ts_id).item()
        for repeat in range(1, repeats + 1):
            for row, tensors in zip(rows, inputs):
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                score = next_ts_probability(model, *tensors, ts_id).item()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise RuntimeError("Invalid latency prediction")
                measurements.append({"repeat": repeat, "sample_id": row["sample_id"],
                                     "dataset": row["dataset"], "latency_ms": elapsed, "ts_probability": score})
            print(json.dumps({"event": "latency_repeat", "context_length": length, "repeat": repeat,
                              **summarize([m["latency_ms"] for m in measurements if m["repeat"] == repeat])}), flush=True)
    return {"summary": summarize([m["latency_ms"] for m in measurements]),
            "per_repeat": [dict(repeat=i, **summarize([m["latency_ms"] for m in measurements if m["repeat"] == i]))
                           for i in range(1, repeats + 1)],
            "measurements": measurements, "sample_count": len(rows),
            "sampled_targets_by_corpus": dict(Counter(r["dataset"] for r in rows)),
            "target_refs_sha256": object_hash([r["sample_id"] for r in rows]),
            "input_sha256": object_hash([(r["sample_id"], r["input_ids"][-length:], r["speaker_ids"][-length:]) for r in rows]),
            "batch_size": 1, "warmup_requests": warmup, "repeats": repeats,
            "timing_scope": TIMING_SCOPE,
            "timer": "perf_counter_ns; synchronize before; scalar .item() before stop",
            "percentile_method": "linear_interpolation_at_(n-1)*q",
            "excluded": ["model_loading", "tokenization", "disk_io", "input_host_to_device", "warmup"]}


def comparison_text(reports):
    lines = ["Matched-target context analysis: fixed checkpoint, validation-bAcc thresholds"]
    for subset in ("primary", "history_gt256"):
        lines.extend(["", f"Classification subset: {subset} (corpus-macro)",
                      f'{"Context":>8}{"bAcc":>12}{"F1":>12}{"Precision":>12}{"Recall":>12}{"Accuracy":>12}{"Targets":>10}'])
        anchor = reports[0]["classification"][subset]
        for report in reports:
            result = report["classification"][subset]
            if result["included_corpora"] != anchor["included_corpora"]:
                raise ValueError("Comparison corpus mismatch")
            for name in result["included_corpora"]:
                for key in ("prediction_positions", "turn_boundaries"):
                    if result["datasets"][name][key] != anchor["datasets"][name][key]:
                        raise ValueError("Comparison target counts differ")
            lines.append(f'{report["context_length"]:8d}' + ''.join(
                f'{result["macro_test_" + key]:12.5f}' for key in ("bacc", "f1", "precision", "recall", "accuracy"))
                + f'{result["test_prediction_positions"]:10d}')
        lines.append("Included corpora: " + ", ".join(anchor["included_corpora"]))
    lines.extend(["", "Fixed-length next-position latency, batch 1, no KV cache",
                  f'{"Context":>8}{"Mean ms":>12}{"P50 ms":>12}{"P95 ms":>12}{"Requests":>12}'])
    for report in reports:
        if report["latency"]["target_refs_sha256"] != reports[0]["latency"]["target_refs_sha256"]:
            raise ValueError("Latency target identities differ across contexts")
        result = report["latency"]["summary"]
        lines.append(f'{report["context_length"]:8d}{result["mean_ms"]:12.3f}{result["p50_ms"]:12.3f}{result["p95_ms"]:12.3f}{result["requests"]:12d}')
    lines.extend(["", "Classification uses matched sampled targets, not the earlier full-window test protocol.",
                  "The long-history subset retains primary validation thresholds; it uses only corpora with both classes.",
                  "512-token inference extends beyond the checkpoint's 256-token task-training windows."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.batch_size, args.warmup, args.repeats) < 1:
        raise ValueError("Batch size, warmup and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the single-GPU worker in conda environment verl")
    manifest, rows = load_samples(args.data_dir)
    torch.manual_seed(manifest["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    token_ids = {name: tokenizer.convert_tokens_to_ids(text) for name, text in
                 (("ts", "<ts>"), ("speaker1", "<speaker1>"), ("speaker2", "<speaker2>"))}
    if token_ids != manifest["special_token_ids"] or tokenizer.pad_token_id is None:
        raise ValueError("Checkpoint tokenizer differs from context data")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    if model.config.eos_token_id != token_ids["ts"]:
        raise ValueError("Expected a trained TurnGPT checkpoint")
    gpu = torch.cuda.get_device_properties(device)
    metadata = {"status": "ok", "schema_version": 1, "experiment": manifest["experiment"],
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_training_context_tokens": 256,
                "data_dir": str(args.data_dir.resolve()), "data_manifest_sha256": file_hash(args.data_dir / "manifest.json"),
                "target_files": manifest["files"], "seed": manifest["seed"],
                "threshold_selection": "validation_only_per_corpus_and_context_length",
                "threshold_objective": "balanced_accuracy", "test_subsets_use_same_threshold": True,
                "classification_batch_size": args.batch_size, "dtype": "bfloat16", "gpu_name": gpu.name,
                "gpu_uuid": str(getattr(gpu, "uuid", "unavailable")), "torch_version": torch.__version__,
                "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
                "attention_implementation": model.config._attn_implementation, "torch_num_threads": 1,
                "tf32": False, "use_cache": False, "logits_to_keep": 1}
    args.output_dir.mkdir(parents=True)
    reports = []
    for length in LENGTHS:
        scores = {}
        for split in ("validation", "test"):
            scores[split] = [None] * len(rows[split])
            for name in manifest["splits"][split]:
                indices = [i for i, row in enumerate(rows[split]) if row["dataset"] == name]
                print(json.dumps({"event": "score_corpus", "context_length": length, "split": split,
                                  "dataset": name, "targets": len(indices)}), flush=True)
                values = score_rows(model, [rows[split][i] for i in indices], length, args.batch_size,
                                    tokenizer.pad_token_id, token_ids, device)
                for i, value in zip(indices, values):
                    scores[split][i] = value
        classification = classification_report(rows, scores)
        score_path = args.output_dir / f"context_{length}_scores.jsonl"
        with score_path.open("x") as handle:
            for split in ("validation", "test"):
                for row, value in zip(rows[split], scores[split]):
                    handle.write(json.dumps({"sample_id": row["sample_id"], "split": split,
                                             "ts_probability": value}, separators=(",", ":")) + "\n")
        latency = measure_latency(model, rows["latency"], length, token_ids["ts"], device, args.warmup, args.repeats)
        effective_context = {}
        for split in ("validation", "test"):
            primary = [r for r in rows[split] if "primary" in r["subsets"]]
            effective_context[split] = {
                "targets": len(primary),
                "mean_input_tokens": sum(min(length, len(r["input_ids"])) for r in primary) / len(primary),
                "full_length_targets": sum(len(r["input_ids"]) >= length for r in primary),
            }
        report = {**metadata, "context_length": length, "classification": classification, "latency": latency,
                  "classification_context_policy": "up to L immediately preceding tokens; shorter histories retained",
                  "effective_primary_context": effective_context,
                  "scores_file": score_path.name, "scores_sha256": file_hash(score_path)}
        (args.output_dir / f"context_{length}.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(json.dumps({"event": "context_complete", "context_length": length,
                          "macro_test_bacc": classification["primary"]["macro_test_bacc"],
                          "macro_test_f1": classification["primary"]["macro_test_f1"], **latency["summary"]}), flush=True)
    comparison = comparison_text(reports)
    (args.output_dir / "comparison.log").write_text(comparison)
    print("\n" + comparison, flush=True)
    print(f"Results and raw scores: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
