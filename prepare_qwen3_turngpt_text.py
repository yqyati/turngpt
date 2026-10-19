#!/usr/bin/env python3
"""Prepare the three locally verified written corpora for Qwen3 TurnGPT.

The output is deliberately named a three-corpus subset. It is not the full
eight-corpus paper dataset. Each JSONL row is one reference-style token window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from audit_dailydialog import read_split as read_daily_dialog_split
from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS


ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
SPLITS = ("train", "validation", "test")
DATASETS = ("daily_dialog", "personachat", "multiwoz21")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persona_dialogues(path: Path):
    current = None
    index = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("1 your persona:"):
            if current is not None:
                yield str(index), current
            index += 1
            current = []
        elif current is not None and "\t" in line:
            first, second, *_ = line.split("\t")
            first = first.split(" ", 1)[1].strip()
            second = second.strip()
            current.extend((first, second))
    if current is not None:
        yield str(index), current


def load_sources():
    sources = {name: {} for name in DATASETS}
    hashes = {}
    for split in SPLITS:
        path = RAW / "dailydialog" / f"{split}.zip"
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        sources["daily_dialog"][split] = list(
            (str(i), turns) for i, turns in enumerate(read_daily_dialog_split(path, split))
        )

        parlai_split = "valid" if split == "validation" else split
        path = RAW / "personachat_parlai" / "personachat" / f"{parlai_split}_both_original.txt"
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        sources["personachat"][split] = list(persona_dialogues(path))

    path = RAW / "multiwoz21" / "MultiWOZ_2.1.zip"
    hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    with zipfile.ZipFile(path) as archive:
        root = "MultiWOZ_2.1/"
        data = json.loads(archive.read(root + "data.json"))
        test_ids = set(archive.read(root + "testListFile.txt").decode().split())
        valid_ids = set(archive.read(root + "valListFile.txt").decode().split())
    if test_ids & valid_ids or not (test_ids | valid_ids) <= data.keys():
        raise ValueError("MultiWOZ 2.1 split IDs overlap or are missing")
    split_ids = {
        "train": sorted(data.keys() - test_ids - valid_ids),
        "validation": sorted(valid_ids),
        "test": sorted(test_ids),
    }
    for split, ids in split_ids.items():
        sources["multiwoz21"][split] = [
            (key, [utterance["text"] for utterance in data[key]["log"]])
            for key in ids
        ]
    return sources, hashes


def normalize_reference(text: str) -> str:
    """Mirror SpokenNormalizer, including space insertion before punctuation removal."""
    import unicodedata

    text = re.sub(r"[,.:;]+(\w+)", r" \1", text)
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r'[.,!?:;()\[\]"-]', "", text)
    return re.sub(r"\s\s+", " ", text).strip()


def tokenize_dialog(tokenizer, turns, ts_id, speaker1_id, speaker2_id):
    text = normalize_reference(turns[0])
    for turn in turns[1:]:
        text += "<ts> " + normalize_reference(turn)
    text += "<ts>"
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    speakers = []
    speaker = speaker1_id
    for token in ids:
        speakers.append(speaker)
        if token == ts_id:
            speaker = speaker2_id if speaker == speaker1_id else speaker1_id
    return ids, speakers


def prepare(args):
    if args.max_length <= args.overlap_length or args.keep_length > args.max_length:
        raise ValueError("Require max_length > overlap_length and keep_length <= max_length")
    if args.limit_dialogues is not None and args.limit_dialogues < 1:
        raise ValueError("--limit-dialogues must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    original_eos = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = original_eos
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    ts_id = tokenizer.convert_tokens_to_ids("<ts>")
    speaker1_id = tokenizer.convert_tokens_to_ids(SPEAKER1)
    speaker2_id = tokenizer.convert_tokens_to_ids(SPEAKER2)
    sources, hashes = load_sources()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "scope": "three_corpus_written_subset",
        "reference": "TurnGPT DialogTextDM 256/64/10 defaults and SpokenDialogTokenizer",
        "model": str(args.model.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "settings": {
            "max_length": args.max_length,
            "keep_length": args.keep_length,
            "overlap_length": args.overlap_length,
            "limit_dialogues_per_split": args.limit_dialogues,
        },
        "special_token_ids": {"ts": ts_id, "speaker1": speaker1_id, "speaker2": speaker2_id},
        "raw_sha256": hashes,
        "splits": {},
    }
    for split in SPLITS:
        out = args.output_dir / f"{split}.jsonl"
        split_report = {}
        with out.open("w", encoding="utf-8") as handle:
            for dataset in DATASETS:
                dialogues = sources[dataset][split]
                selected = dialogues[: args.limit_dialogues] if args.limit_dialogues else dialogues
                counts = Counter(source_dialogues=len(dialogues), selected_dialogues=len(selected))
                for source_id, turns in selected:
                    # Match reference filter_empty_turns before tokenization.
                    if any(not re.search(r"\w", turn) for turn in turns):
                        counts["filtered_dialogues"] += 1
                        continue
                    ids, speakers = tokenize_dialog(
                        tokenizer, turns, ts_id, speaker1_id, speaker2_id
                    )
                    if len(ids) != len(speakers):
                        raise AssertionError(f"speaker alignment failed: {dataset}/{source_id}")
                    counts["accepted_dialogues"] += 1
                    counts["turns"] += len(turns)
                    counts["tokens"] += len(ids)
                    counts["ts_tokens"] += ids.count(ts_id)
                    stride = args.max_length - args.overlap_length
                    for start in range(0, len(ids), stride):
                        window_ids = ids[start : start + args.max_length]
                        if len(window_ids) < args.keep_length:
                            break
                        row = {
                            "dataset": dataset,
                            "source_split": split,
                            "source_id": source_id,
                            "start_token": start,
                            "input_ids": window_ids,
                            "speaker_ids": speakers[start : start + args.max_length],
                        }
                        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                        counts["windows"] += 1
                split_report[dataset] = dict(counts)
        report["splits"][split] = split_report
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--keep-length", type=int, default=64)
    parser.add_argument("--overlap-length", type=int, default=10)
    parser.add_argument("--limit-dialogues", type=int, default=None)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
