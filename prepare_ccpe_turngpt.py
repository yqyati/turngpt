#!/usr/bin/env python3
"""Prepare the author-released CCPE corpus with deterministic 90/5/5 splits."""

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
DEFAULT_RAW = ROOT / "data" / "raw" / "ccpe" / "data.json"


def normalize_reference(text: str) -> str:
    text = re.sub(r"[,.:;]+(\w+)", r" \1", text)
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[.,!?:;()\[\]"-]', "", text)
    return re.sub(r"\s\s+", " ", text).strip()


def split_name(index: int, seed: int) -> str:
    digest = hashlib.sha256(f"ccpe:{seed}:{index}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def dialog_text(turns):
    text = normalize_reference(turns[0])
    for turn in turns[1:]:
        text += "<ts> " + normalize_reference(turn)
    return text + "<ts>"


def encode_ids(tokenizer, text, ids):
    input_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    speakers = []
    state = ids["speaker1"]
    for token in input_ids:
        speakers.append(state)
        if token == ids["ts"]:
            state = ids["speaker2"] if state == ids["speaker1"] else ids["speaker1"]
    return input_ids, speakers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--keep-length", type=int, default=64)
    parser.add_argument("--overlap-length", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.max_length <= args.overlap_length or args.keep_length > args.max_length:
        raise ValueError("Invalid window settings")
    raw = args.data.read_bytes()
    conversations = json.loads(raw)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    ids = {name: tokenizer.convert_tokens_to_ids(token)
           for name, token in (("ts", "<ts>"), ("speaker1", SPEAKER1), ("speaker2", SPEAKER2))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
               for split in ("train", "validation", "test")}
    reports = {split: Counter() for split in handles}
    records = []
    for index, conversation in enumerate(conversations):
            split = split_name(index, args.seed)
            turns = []
            previous = None
            for utterance in conversation["utterances"]:
                text = utterance["text"].strip()
                if not text:
                    continue
                speaker = utterance["speaker"]
                if speaker == previous:
                    turns[-1] += " " + text
                else:
                    turns.append(text)
                previous = speaker
            reports[split]["source_dialogues"] += 1
            if len(turns) < 2 or any(not re.search(r"\w", turn) for turn in turns):
                reports[split]["filtered_dialogues"] += 1
                continue
            records.append((index, split, turns, dialog_text(turns)))
    texts = [record[3] for record in records]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
    try:
        for record, input_ids in zip(records, encoded["input_ids"]):
            index, split, turns, _ = record
            speakers = []
            state = ids["speaker1"]
            for token in input_ids:
                speakers.append(state)
                if token == ids["ts"]:
                    state = ids["speaker2"] if state == ids["speaker1"] else ids["speaker1"]
            reports[split]["accepted_dialogues"] += 1
            reports[split]["turns"] += len(turns)
            reports[split]["tokens"] += len(input_ids)
            reports[split]["ts_tokens"] += input_ids.count(ids["ts"])
            stride = args.max_length - args.overlap_length
            for start in range(0, len(input_ids), stride):
                window = input_ids[start : start + args.max_length]
                if len(window) < args.keep_length:
                    break
                handles[split].write(json.dumps({
                    "dataset": "ccpe", "source_split": split,
                    "source_id": str(index), "start_token": start,
                    "input_ids": window,
                    "speaker_ids": speakers[start : start + args.max_length],
                }, separators=(",", ":")) + "\n")
                reports[split]["windows"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "status": "ok", "scope": "ccpe_single_corpus",
        "source": "google-research-datasets/ccpe author release",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_path": str(args.data.resolve()), "seed": args.seed,
        "split_rule": "sha256(ccpe:{seed}:{source_index}) mod 100: 0-89 train, 90-94 validation, 95-99 test",
        "settings": {"max_length": args.max_length, "keep_length": args.keep_length,
                     "overlap_length": args.overlap_length},
        "special_token_ids": ids,
        "splits": {split: dict(reports[split]) for split in reports},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
