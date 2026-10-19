#!/usr/bin/env python3
"""Fine-tune Qwen3 Base or GPT-2 on compatible prepared TurnGPT text windows.

The historical filename refers to the initial three-corpus subset. Defaults
now use the formal eight-corpus Qwen3 cache and the 500-step recipe. Supply
--model explicitly, and --data-dir for GPT-2's native-token cache. The generic
scripts/train_eval.sh runs training and both classification evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS, resolve_dtype


ROOT = Path(__file__).parent
# Preserve the historical CLI default; scripts/ selects the eight-corpus cache.
DEFAULT_DATA = ROOT / "data" / "processed" / "qwen3_turngpt_verified_8dataset_v2"


class WindowDataset(Dataset):
    def __init__(self, path: Path, manifest: dict, split: str):
        self.rows = []
        ts_id = manifest["special_token_ids"]["ts"]
        speaker_ids = {
            manifest["special_token_ids"]["speaker1"],
            manifest["special_token_ids"]["speaker2"],
        }
        counts = {name: 0 for name in manifest["splits"][split]}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                ids, states = row["input_ids"], row["speaker_ids"]
                if row["source_split"] != split or not 2 <= len(ids) <= 256:
                    raise ValueError(f"Invalid split or window length in {path}")
                if len(ids) != len(states) or not set(states) <= speaker_ids:
                    raise ValueError(f"Misaligned speaker states in {path}")
                if row["dataset"] not in counts:
                    raise ValueError(f"Unexpected dataset in {path}: {row['dataset']}")
                if any(token < 0 or token > ts_id for token in ids):
                    raise ValueError(f"Invalid token ID in {path}")
                counts[row["dataset"]] += 1
                self.rows.append((ids, states))
        expected = {name: values["windows"] for name, values in manifest["splits"][split].items()}
        if counts != expected:
            raise ValueError(f"Window count mismatch for {split}: {counts} != {expected}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collator(pad_id: int, speaker1_id: int):
    def collate(rows):
        length = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
        speaker_ids = torch.full((len(rows), length), speaker1_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for i, (ids, states) in enumerate(rows):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            speaker_ids[i, :n] = torch.tensor(states)
            attention_mask[i, :n] = 1
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return input_ids, speaker_ids, attention_mask, labels

    return collate


def forward(model, batch, device):
    input_ids, speaker_ids, attention_mask, labels = (x.to(device) for x in batch)
    embeddings = model.get_input_embeddings()
    return model(
        inputs_embeds=embeddings(input_ids) + embeddings(speaker_ids),
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )


def initialize_turngpt_embeddings(model, tokenizer, token_ids):
    """Initialize TurnGPT rows in the input and, when untied, output weights."""
    # GPT-2's pretrained vocabulary has a nonzero centroid. Zero-centred new
    # output rows can dominate its softmax (even though speaker IDs are never
    # targets). Centre their random initialization on the old vocabulary mean.
    # The Qwen3 branch keeps its original operations and RNG sequence unchanged.
    speaker_centres = None
    if model.config.model_type == "gpt2":
        with torch.no_grad():
            speaker_centres = (
                model.get_input_embeddings().weight.float().mean(0),
                model.get_output_embeddings().weight.float().mean(0),
            )
        model.config.turngpt_speaker_initialization = "pretrained_vocabulary_mean_plus_normal_noise"
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    embeddings = model.get_input_embeddings().weight
    punctuation_ids = [tokenizer.encode(mark, add_special_tokens=False)[0]
                       for mark in ("!", "?", ".")]
    with torch.no_grad():
        embeddings[token_ids["ts"]].copy_(embeddings[punctuation_ids].mean(0))
        for name in ("speaker1", "speaker2"):
            embeddings[token_ids[name]].normal_(mean=0.0, std=model.config.initializer_range)
            if speaker_centres is not None:
                embeddings[token_ids[name]].add_(speaker_centres[0].to(embeddings))
        # Qwen3-8B has a separate output head. The smaller tied models already
        # share the rows initialized above; preserve their initialization/RNG.
        output_embeddings = model.get_output_embeddings().weight
        if output_embeddings.data_ptr() != embeddings.data_ptr():
            output_embeddings[token_ids["ts"]].copy_(output_embeddings[punctuation_ids].mean(0))
            for name in ("speaker1", "speaker2"):
                output_embeddings[token_ids[name]].normal_(mean=0.0, std=model.config.initializer_range)
                if speaker_centres is not None:
                    output_embeddings[token_ids[name]].add_(speaker_centres[1].to(output_embeddings))
    model.config.eos_token_id = token_ids["ts"]
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = token_ids["ts"]
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False


def validation_samples(dataset, manifest, split, per_dataset, seed):
    """Select fixed windows from every corpus without changing train order."""
    selected = {}
    offset = 0
    for name, stats in manifest["splits"][split].items():
        count = stats["windows"]
        rng = random.Random(f"{seed}:{name}:{split}")
        indices = sorted(rng.sample(range(count), min(per_dataset, count)))
        selected[name] = [dataset.rows[offset + index] for index in indices]
        offset += count
    if offset != len(dataset):
        raise ValueError("Manifest window order does not match dataset")
    return selected


@torch.no_grad()
def evaluate(model, groups, collate, batch_size, device):
    model.eval()
    results = {}
    total_loss = 0.0
    total_targets = 0
    for name, rows in groups.items():
        loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
        weighted_loss = 0.0
        targets = 0
        for batch in loader:
            n = int((batch[3][:, 1:] != -100).sum())
            if n == 0:
                continue
            loss = forward(model, batch, device).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            weighted_loss += float(loss.detach().cpu()) * n
            targets += n
        if targets == 0:
            raise RuntimeError(f"No validation targets for {name}")
        results[name] = {"windows": len(rows), "target_tokens": targets,
                         "mean_loss": weighted_loss / targets}
        total_loss += weighted_loss
        total_targets += targets
    return {"datasets": results, "overall_token_weighted_loss": total_loss / total_targets,
            "overall_target_tokens": total_targets}


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.val_windows_per_dataset < 1:
        raise ValueError("steps, batch-size, grad-accum, and val-windows-per-dataset must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    manifest_path = args.data_dir / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["scope"] != "three_corpus_written_subset" or manifest["settings"] != {
        "max_length": 256, "keep_length": 64, "overlap_length": 10,
        "limit_dialogues_per_split": None,
    }:
        raise ValueError("Expected a compatible 256/64/10 prepared cache")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    token_ids = {
        "ts": tokenizer.convert_tokens_to_ids("<ts>"),
        "speaker1": tokenizer.convert_tokens_to_ids(SPEAKER1),
        "speaker2": tokenizer.convert_tokens_to_ids(SPEAKER2),
    }
    if token_ids != manifest["special_token_ids"]:
        raise ValueError("Tokenizer special tokens do not match prepared data")

    collate = make_collator(tokenizer.pad_token_id, token_ids["speaker1"])
    train_data = WindowDataset(args.data_dir / "train.jsonl", manifest, "train")
    val_data = WindowDataset(args.data_dir / "validation.jsonl", manifest, "validation")
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    val_groups = validation_samples(
        val_data, manifest, "validation", args.val_windows_per_dataset, args.seed
    )

    dtype = resolve_dtype(args.dtype)
    model_kwargs = {"local_files_only": True}
    if dtype != "auto":
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    initialize_turngpt_embeddings(model, tokenizer, token_ids)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    initial_val = evaluate(model, val_groups, collate, args.batch_size, device)
    print(json.dumps({"event": "initial_validation", **initial_val}), flush=True)
    iterator = iter(loader)
    losses = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss = forward(model, batch, device).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            micro_losses.append(float(loss.detach().cpu()))
            (loss / args.grad_accum).backward()
        optimizer.step()
        step_loss = sum(micro_losses) / len(micro_losses)
        losses.append(step_loss)
        print(json.dumps({"event": "train_step", "step": step, "loss": step_loss}), flush=True)

    final_val = evaluate(model, val_groups, collate, args.batch_size, device)
    print(json.dumps({"event": "final_validation", **final_val}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    report = {
        "status": "ok", "scope": manifest["scope"], "base_model": str(args.model.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "device": device, "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "dtype": args.dtype, "seed": args.seed, "optimizer": "AdamW",
        "learning_rate": args.learning_rate, "steps": args.steps,
        "batch_size": args.batch_size, "gradient_accumulation": args.grad_accum,
        "no_train_first_n": -1, "validation_windows_per_dataset": args.val_windows_per_dataset,
        "speaker_embedding_initialization": getattr(model.config, "turngpt_speaker_initialization", "zero_mean_normal"),
        "train_windows": len(train_data), "validation_windows": len(val_data),
        "train_step_losses": losses, "initial_validation": initial_val,
        "final_validation": final_val, "checkpoint": str(model_dir.resolve()),
    }
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "saved", "report": str(args.output_dir / "train_report.json"),
                      "checkpoint": str(model_dir)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--val-windows-per-dataset", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=6.25e-5)
    parser.add_argument("--seed", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
