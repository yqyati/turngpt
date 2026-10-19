#!/usr/bin/env python3
"""Check GPT-2/Qwen3 corpus provenance and retained dialogue bytes, not token equality."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retained_dialogues(path, tokenizer, manifest, split):
    """Reassemble overlapping token windows and convert reversible byte-BPE to bytes.

    Comparing bytes avoids Unicode replacement characters at truncated multibyte
    endpoints. Special <ts> markers remain literal ASCII in both representations.
    """
    decoder = {value: key for key, value in bytes_to_unicode().items()}
    ids = manifest["special_token_ids"]
    token_bytes = {}
    def decode(tokens):
        for token in tokens:
            if token not in token_bytes:
                token_bytes[token] = (b"<ts>" if token == ids["ts"] else
                                      bytes(decoder[c] for c in tokenizer.convert_ids_to_tokens(token)))
        return b"".join(token_bytes[t] for t in tokens)

    dialogs = {}
    counts = {name: {"windows": 0, "tokens": 0, "ts_tokens": 0}
              for name in manifest["splits"][split]}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            name = row["dataset"]
            if row["source_split"] != split or name not in counts:
                raise ValueError(f"Invalid corpus or split: {path}")
            key = (name, str(row["source_id"]))
            tokens, states, start = row["input_ids"], row["speaker_ids"], row["start_token"]
            if len(tokens) != len(states):
                raise ValueError(f"Misaligned speaker states: {key}")
            if key not in dialogs:
                if start != 0:
                    raise ValueError(f"Missing dialogue prefix: {key}")
                dialogs[key] = ([], [])
            previous, previous_states = dialogs[key]
            overlap = len(previous) - start
            if overlap < 0 or previous[start:] != tokens[:overlap] or previous_states[start:] != states[:overlap]:
                raise ValueError(f"Inconsistent overlapping windows: {key}")
            previous.extend(tokens[overlap:])
            previous_states.extend(states[overlap:])
            counts[name]["windows"] += 1
            counts[name]["tokens"] += len(tokens)
            counts[name]["ts_tokens"] += tokens.count(ids["ts"])
    for name in counts:
        if counts[name] != manifest["splits"][split][name]:
            raise ValueError(f"Cached counts disagree: {path}/{name}")
    result = {}
    for key, (tokens, states) in dialogs.items():
        speaker = ids["speaker1"]
        for token, state in zip(tokens, states):
            if state != speaker:
                raise ValueError(f"Unexpected speaker transition: {key}")
            if token == ids["ts"]:
                speaker = ids["speaker2"] if speaker == ids["speaker1"] else ids["speaker1"]
        result[key] = decode(tokens)
    return result


def source_manifest(data_dir, manifest, name, family):
    """Read packaged provenance without following archived absolute paths."""
    formal = ROOT / "data/processed" / ("gpt2_turngpt_8dataset_v1/merged" if family == "gpt2" else "qwen3_turngpt_verified_8dataset_v2")
    source = Path(manifest["sources"][name])
    if data_dir.resolve() == formal.resolve():
        return ROOT / "data/provenance" / family / (source.name + ".json")
    # Newly rebuilt caches use source paths written by the current merger.
    return (source if source.is_absolute() else ROOT / source) / "manifest.json"


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    gm = json.loads((args.gpt2_data / "manifest.json").read_text())
    qm = json.loads((args.qwen_data / "manifest.json").read_text())
    if gm["datasets"] != qm["datasets"] or gm["settings"] != qm["settings"]:
        raise ValueError("Corpus order or window settings differ")
    gt = AutoTokenizer.from_pretrained(args.gpt2_model, local_files_only=True)
    qt = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True)
    raw_checks = {}
    for name in gm["datasets"]:
        gp = source_manifest(args.gpt2_data, gm, name, "gpt2")
        qp = source_manifest(args.qwen_data, qm, name, "qwen3")
        g, q = json.loads(gp.read_text()), json.loads(qp.read_text())
        # Pre-tokenization filtering and source/split selection must be the same.
        for key in ("seed", "split_rule", "raw_sha256"):
            if g.get(key) != q.get(key):
                raise ValueError(f"Source preparation differs: {name}/{key}")
        selected = {}
        for split in SPLITS:
            a, b = g["splits"][split], q["splits"][split]
            if name in a:
                a, b = a[name], b[name]
            for key in a:
                if any(term in key for term in ("dialogues", "turns")):
                    if a[key] != b.get(key):
                        raise ValueError(f"Source/filter counts differ: {name}/{split}/{key}")
                    selected[f"{split}/{key}"] = a[key]
        raw_checks[name] = {"gpt2_source_manifest_sha256": file_hash(gp),
                            "qwen_source_manifest_sha256": file_hash(qp), "source_counts": selected}
    reports = {}
    for split in SPLITS:
        print(f"Checking retained {split} dialogue text and speaker states", flush=True)
        gpath, qpath = args.gpt2_data / f"{split}.jsonl", args.qwen_data / f"{split}.jsonl"
        for path, manifest in ((gpath, gm), (qpath, qm)):
            if file_hash(path) != manifest["totals"][split]["sha256"]:
                raise ValueError(f"Cached split hash mismatch: {path}")
        g = retained_dialogues(gpath, gt, gm, split)
        q = retained_dialogues(qpath, qt, qm, split)
        shared = g.keys() & q.keys()
        for key in shared:
            if not (g[key].startswith(q[key]) or q[key].startswith(g[key])):
                raise ValueError(f"Retained text differs beyond the truncated tail: {split}/{key}")
        reports[split] = {
            "shared_retained_dialogues": len(shared),
            "gpt2_only_retained_dialogues": len(g.keys() - q.keys()),
            "qwen_only_retained_dialogues": len(q.keys() - g.keys()),
            "shared_text_prefixes_match": True,
            "gpt2": gm["totals"][split], "qwen3": qm["totals"][split],
            "corpora": {name: {"gpt2": gm["splits"][split][name], "qwen3": qm["splits"][split][name]}
                        for name in gm["datasets"]},
        }
    result = {"status": "ok", "protocol": "same_source_dialogues_and_splits_native_token_windows",
              "gpt2_data": str(args.gpt2_data.resolve()), "qwen_data": str(args.qwen_data.resolve()),
              "gpt2_manifest_sha256": file_hash(args.gpt2_data / "manifest.json"),
              "qwen_manifest_sha256": file_hash(args.qwen_data / "manifest.json"),
              "source_checks": raw_checks, "splits": reports,
              "limitation": "Native tokenizers change window endpoints, retained short dialogues, prediction positions and class prevalence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Verified matching source/filter counts, raw hashes where recorded, shared dialogue prefixes and speaker states.")
    for split, row in reports.items():
        print(f"{split}: GPT-2 {row['gpt2']['windows']:,} windows; Qwen3 {row['qwen3']['windows']:,} windows")
    print(f"Saved {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2-data", type=Path, required=True)
    parser.add_argument("--gpt2-model", type=Path, required=True)
    parser.add_argument("--qwen-data", type=Path, default=ROOT / "data/processed/qwen3_turngpt_verified_8dataset_v2")
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
