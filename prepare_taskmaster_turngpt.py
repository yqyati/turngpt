#!/usr/bin/env python3
"""Prepare official Taskmaster 2/3 JSON files with checked-in splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "taskmaster"
SPLIT_ROOT = ROOT / "data" / "splits"


def normalize(text):
    text = re.sub(r"[,.:;]+(\w+)", r" \1", text)
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s\s+", " ", re.sub(r'[.,!?:;()\[\]"-]', "", text)).strip()


def alternating(turns):
    result = []
    previous = None
    for item in turns:
        text = item["text"].strip()
        if not text:
            continue
        speaker = item["speaker"]
        if speaker == previous:
            result[-1] += " " + text
        else:
            result.append(text)
        previous = speaker
    return result


def load_dataset_files():
    records = {"taskmaster2": [], "taskmaster3": []}
    for name in ("flights", "food-ordering", "restaurant-search"):
        data = json.loads((RAW / "taskmaster2" / f"{name}.json").read_text())
        records["taskmaster2"].extend((f"{name}:{i}", alternating(x["utterances"])) for i, x in enumerate(data))
    for path in sorted((RAW / "taskmaster3").glob("data_*.json")):
        data = json.loads(path.read_text())
        for i, item in enumerate(data):
            records["taskmaster3"].append((f"{path.stem}:{i}", alternating(item["utterances"])))
    return records


def split_records(records):
    manifests = {
        "taskmaster2": json.loads((SPLIT_ROOT / "taskmaster2_splits.json").read_text()),
        "taskmaster3": json.loads((SPLIT_ROOT / "taskmaster3_splits.json").read_text()),
    }
    result = {split: [] for split in ("train", "validation", "test")}
    for dataset, values in records.items():
        manifest = manifests[dataset]
        for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
            for index in manifest[key]:
                source_id, turns = values[index]
                result[split].append((dataset, source_id, turns))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--keep-length", type=int, default=64)
    parser.add_argument("--overlap-length", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    ids = {n: tokenizer.convert_tokens_to_ids(t) for n, t in (("ts", "<ts>"), ("speaker1", SPEAKER1), ("speaker2", SPEAKER2))}
    records = split_records(load_dataset_files())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: (args.output_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in records}
    reports = {s: Counter() for s in records}
    try:
        for split, items in records.items():
            texts, accepted = [], []
            for dataset, source_id, turns in items:
                reports[split][f"{dataset}_source_dialogues"] += 1
                if len(turns) < 2 or any(not re.search(r"\w", t) for t in turns):
                    reports[split]["filtered_dialogues"] += 1
                    continue
                text = normalize(turns[0]) + "".join("<ts> " + normalize(t) for t in turns[1:]) + "<ts>"
                texts.append(text)
                accepted.append((dataset, source_id, turns))
            tokenized = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            for (dataset, source_id, turns), input_ids in zip(accepted, tokenized):
                speakers, state = [], ids["speaker1"]
                for token in input_ids:
                    speakers.append(state)
                    if token == ids["ts"]:
                        state = ids["speaker2"] if state == ids["speaker1"] else ids["speaker1"]
                reports[split][f"{dataset}_accepted_dialogues"] += 1
                reports[split][f"{dataset}_turns"] += len(turns)
                reports[split][f"{dataset}_tokens"] += len(input_ids)
                reports[split][f"{dataset}_ts_tokens"] += input_ids.count(ids["ts"])
                stride = args.max_length - args.overlap_length
                for start in range(0, len(input_ids), stride):
                    window = input_ids[start:start + args.max_length]
                    if len(window) < args.keep_length:
                        break
                    handles[split].write(json.dumps({"dataset": dataset, "source_split": split, "source_id": source_id, "start_token": start, "input_ids": window, "speaker_ids": speakers[start:start + args.max_length]}, separators=(",", ":")) + "\n")
                    reports[split][f"{dataset}_windows"] += 1
    finally:
        for h in handles.values():
            h.close()
    manifest = {"status": "ok", "scope": "taskmaster2_3", "source_manifest": str((RAW / "manifest.json").resolve()), "split_manifests": [str((SPLIT_ROOT / "taskmaster2_splits.json").resolve()), str((SPLIT_ROOT / "taskmaster3_splits.json").resolve())], "settings": {"max_length": args.max_length, "keep_length": args.keep_length, "overlap_length": args.overlap_length}, "special_token_ids": ids, "splits": {s: dict(v) for s, v in reports.items()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
