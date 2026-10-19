#!/usr/bin/env python3
"""Select per-corpus thresholds on validation, then evaluate test classification.

The bAcc definition follows TurnGPT, but this is not a full reproduction of
the paper's protocol. scripts/eval_bacc.sh selects the current baseline inputs.
Precision, recall, F1 and accuracy use the same validation-bAcc-selected threshold.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_qwen3_turngpt_subset import DEFAULT_CHECKPOINT, load_groups
from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS, resolve_dtype
from train_qwen3_turngpt_subset import DEFAULT_DATA, initialize_turngpt_embeddings, make_collator
from turngpt_metrics import classification_metrics, aggregate_classification_metrics

@torch.no_grad()
def collect(model, rows, collate, batch_size, device, ts_id):
    scores, labels = [], []
    for ids, sp, mask, _ in DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0):
        ids, sp, mask = ids.to(device), sp.to(device), mask.to(device)
        emb = model.get_input_embeddings()
        logits = model(inputs_embeds=emb(ids) + emb(sp), attention_mask=mask, use_cache=False).logits[:, :-1].float()
        valid = mask[:, 1:].bool()
        prob = (logits[..., ts_id] - torch.logsumexp(logits, dim=-1)).exp()
        scores.extend(prob[valid].cpu().tolist())
        labels.extend((ids[:, 1:][valid] == ts_id).int().cpu().tolist())
    return scores, labels

def bacc(scores, labels, threshold):
    pred = [s >= threshold for s in scores]
    pos = sum(labels); neg = len(labels) - pos
    tp = sum(p and y for p, y in zip(pred, labels)); tn = sum((not p) and (not y) for p, y in zip(pred, labels))
    return (tp / pos + tn / neg) / 2 if pos and neg else 0.0

def best_threshold(scores, labels):
    """Find the threshold with maximum bAcc in one sorted sweep."""
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5, 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    best = (0.0, 0.5)
    i = 0
    while i < len(order):
        score = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == score:
            if labels[order[j]]:
                tp += 1
            else:
                fp += 1
            j += 1
        value = (tp / positives + (negatives - fp) / negatives) / 2
        if value > best[0] or (value == best[0] and score < best[1]):
            best = (value, score)
        i = j
    return best[1], best[0]

def sample_groups(data_dir, split, manifest, limit, seed):
    groups = load_groups(data_dir / f"{split}.jsonl", manifest, split)
    out = {}
    for name, rows in groups.items():
        rng = random.Random(f"{seed}:{name}:{split}")
        out[name] = [rows[i] for i in sorted(rng.sample(range(len(rows)), min(limit, len(rows))))]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT); ap.add_argument("--base-model", type=Path)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--windows-per-dataset", type=int, default=100000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--init-seed", type=int, default=1)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda"); ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    if not (args.checkpoint or args.base_model):
        ap.error("Provide --checkpoint or --base-model")
    if args.windows_per_dataset < 1 or args.batch_size < 1:
        raise ValueError("windows-per-dataset and batch-size must be positive")
    if args.output.exists(): raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    raw_manifest = (args.data_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["scope"] != "three_corpus_written_subset": raise ValueError("Unexpected dataset scope")
    model_path = args.base_model or args.checkpoint
    if args.base_model: torch.manual_seed(args.init_seed)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.base_model:
        tok.pad_token = tok.pad_token or tok.eos_token; tok.add_special_tokens(SPECIAL_TOKENS)
    ids = {"ts": tok.convert_tokens_to_ids("<ts>"), "speaker1": tok.convert_tokens_to_ids(SPEAKER1), "speaker2": tok.convert_tokens_to_ids(SPEAKER2)}
    if ids != manifest["special_token_ids"]: raise ValueError("Tokenizer/data token IDs differ")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=resolve_dtype(args.dtype))
    if args.base_model: initialize_turngpt_embeddings(model, tok, ids)
    model.to(device).eval(); collate = make_collator(tok.pad_token_id, ids["speaker1"])
    val = sample_groups(args.data_dir, "validation", manifest, args.windows_per_dataset, args.seed)
    test = sample_groups(args.data_dir, "test", manifest, args.windows_per_dataset, args.seed)
    reports, thresholds = {}, {}
    for name in val:
        print(json.dumps({"event": "evaluate_corpus", "dataset": name,
                          "validation_windows": len(val[name]), "test_windows": len(test[name])}), flush=True)
        vs, vy = collect(model, val[name], collate, args.batch_size, device, ids["ts"])
        threshold, validation_bacc = best_threshold(vs, vy)
        thresholds[name] = {"threshold": threshold, "validation_bacc": validation_bacc, "windows": len(val[name])}
        ts, ty = collect(model, test[name], collate, args.batch_size, device, ids["ts"])
        metrics, confusion = classification_metrics(ts, ty, threshold)
        reports[name] = {"windows": len(test[name]), "prediction_positions": len(ty),
                         "turn_boundaries": sum(ty), "threshold": threshold,
                         "validation_bacc": validation_bacc,
                         "test_confusion_matrix": confusion,
                         **{f"test_{key}": value for key, value in metrics.items()}}
        print(json.dumps({"dataset": name, **reports[name]}), flush=True)
    summary = aggregate_classification_metrics([x["test_confusion_matrix"] for x in reports.values()])
    report = {"status": "ok", "schema_version": 2,
              "metric_scope": "paper_threshold_tuned_balanced_accuracy",
              "threshold_selection": "validation_only", "threshold_objective": "balanced_accuracy",
              "classification_threshold": "same_validation_bacc_threshold_for_all_metrics",
              "model_kind": "initialized_base" if args.base_model else "checkpoint",
              "model": str(model_path.resolve()), "data_dir": str(args.data_dir.resolve()),
              "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
              "windows_per_dataset": args.windows_per_dataset, "batch_size": args.batch_size,
              "seed": args.seed, "init_seed": args.init_seed if args.base_model else None,
              "speaker_embedding_initialization": getattr(model.config, "turngpt_speaker_initialization", "zero_mean_normal"),
              "device": device, "dtype": args.dtype, "torch_version": torch.__version__,
              "datasets": reports, **summary, "thresholds": thresholds}
    print(json.dumps({"event": "test_summary", **summary}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps({"event": "saved", "output": str(args.output)}), flush=True)
if __name__ == "__main__": main()
