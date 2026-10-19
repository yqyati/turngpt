#!/usr/bin/env python3
"""Audit the raw DailyDialog split archives using the upstream parser format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile


SPLITS = ("train", "validation", "test")


def read_split(path: Path, split: str):
    with ZipFile(path) as outer:
        dialog_path = f"{split}/dialogues_{split}.txt"
        acts_path = f"{split}/dialogues_act_{split}.txt"
        emotions_path = f"{split}/dialogues_emotion_{split}.txt"
        with (
            outer.open(dialog_path) as dialogs_file,
            outer.open(acts_path) as acts_file,
            outer.open(emotions_path) as emotions_file,
        ):
            dialogs = dialogs_file.read().decode("utf-8").splitlines()
            acts = acts_file.read().decode("utf-8").splitlines()
            emotions = emotions_file.read().decode("utf-8").splitlines()

    if not (len(dialogs) == len(acts) == len(emotions)):
        raise ValueError(
            f"{split} line counts differ: dialogs={len(dialogs)}, "
            f"acts={len(acts)}, emotions={len(emotions)}"
        )

    records = []
    for row, (dialog_line, act_line, emotion_line) in enumerate(
        zip(dialogs, acts, emotions)
    ):
        # Match the upstream DailyDialog loader's delimiter handling.
        dialog = dialog_line.split("__eou__")[:-1]
        act_ids = act_line.split(" ")[:-1]
        emotion_ids = emotion_line.split(" ")[:-1]
        if not (len(dialog) == len(act_ids) == len(emotion_ids)):
            raise ValueError(
                f"{split} row {row} token counts differ: "
                f"turns={len(dialog)}, acts={len(act_ids)}, "
                f"emotions={len(emotion_ids)}"
            )
        records.append(dialog)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "raw" / "dailydialog",
    )
    args = parser.parse_args()

    all_hashes: dict[str, set[str]] = {}
    report = {"data_dir": str(args.data_dir.resolve()), "splits": {}}
    for split in SPLITS:
        dialogs = read_split(args.data_dir / f"{split}.zip", split)
        lengths = [len(dialog) for dialog in dialogs]
        empty_dialogs = sum(not dialog for dialog in dialogs)
        empty_turns = sum(not turn.strip() for dialog in dialogs for turn in dialog)
        hashes = {
            hashlib.sha256("\x1f".join(dialog).encode("utf-8")).hexdigest()
            for dialog in dialogs
        }
        all_hashes[split] = hashes
        report["splits"][split] = {
            "dialogues": len(dialogs),
            "turns": sum(lengths),
            "turns_per_dialog_median": median(lengths),
            "turns_per_dialog_min": min(lengths),
            "turns_per_dialog_max": max(lengths),
            "empty_dialogues": empty_dialogs,
            "empty_turns": empty_turns,
            "unique_dialogues": len(hashes),
        }

    overlap = {}
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1 :]:
            overlap[f"{split_a}:{split_b}"] = len(
                all_hashes[split_a] & all_hashes[split_b]
            )
    report["exact_dialogue_overlap"] = overlap
    report["status"] = "ok"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
