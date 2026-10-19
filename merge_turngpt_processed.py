#!/usr/bin/env python3
"""Merge verified TurnGPT JSONL caches without changing examples."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

DEFAULT_SOURCES = {
    "daily_dialog": Path("data/processed/qwen3_turngpt_3corpus_v1"),
    "personachat": Path("data/processed/qwen3_turngpt_3corpus_v1"),
    "multiwoz21": Path("data/processed/qwen3_turngpt_3corpus_v1"),
    "ccpe": Path("data/processed/ccpe_turngpt_v1"),
    "taskmaster1": Path("data/processed/taskmaster1_turngpt_v1"),
    "taskmaster2": Path("data/processed/taskmaster2_3_turngpt_v1"),
    "taskmaster3": Path("data/processed/taskmaster2_3_turngpt_v1"),
    "metawoz": Path("data/processed/metawoz_turngpt_v2"),
}

def validate_sources(sources):
    """Infer token IDs from the source manifests; never mix tokenizers."""
    token_ids = None
    provenance = {}
    for dataset, directory in sources.items():
        path = directory / "manifest.json"
        raw = path.read_bytes()
        manifest = json.loads(raw)
        current = manifest["special_token_ids"]
        if set(current) != {"ts", "speaker1", "speaker2"} or len(set(current.values())) != 3:
            raise ValueError(f"Invalid special-token IDs: {path}")
        if token_ids is not None and current != token_ids:
            raise ValueError(f"Cannot merge different tokenizers: {path}")
        token_ids = current
        for key, expected in (("max_length", 256), ("keep_length", 64), ("overlap_length", 10)):
            if manifest["settings"][key] != expected:
                raise ValueError(f"Window settings differ: {path}/{key}")
        provenance[dataset] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    return token_ids, provenance

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--include", nargs="+", choices=sorted(DEFAULT_SOURCES), default=list(DEFAULT_SOURCES))
    ap.add_argument("--source-map", type=Path, required=True, help="JSON mapping each corpus to its cache directory; paths relative to this file")
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if len(set(args.include)) != len(args.include):
        raise ValueError("Duplicate corpus names")
    source_map = DEFAULT_SOURCES
    if args.source_map:
        source_map = {k: args.source_map.resolve().parent / v
                      for k, v in json.loads(args.source_map.read_text()).items()}
    sources = {k: source_map[k] for k in args.include}
    token_ids, source_manifests = validate_sources(sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: (args.output_dir / f"{s}.jsonl").open("w") for s in ("train", "validation", "test")}
    stats = {s: {d: Counter() for d in sources} for s in handles}
    seen = {s: set() for s in handles}
    try:
        for dataset, directory in sources.items():
            for split, handle in handles.items():
                path = directory / f"{split}.jsonl"
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open() as inp:
                    for line_no, line in enumerate(inp, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("dataset") != dataset:
                            # The shared TM2/TM3 cache is split by its dataset field.
                            continue
                        ids, states = row["input_ids"], row["speaker_ids"]
                        if (row["source_split"] != split or not 64 <= len(ids) <= 256 or
                                len(ids) != len(states) or
                                not set(states) <= {token_ids["speaker1"], token_ids["speaker2"]} or
                                any(x < 0 or x > token_ids["ts"] for x in ids)):
                            raise ValueError(f"Invalid window: {path}:{line_no}")
                        key = (dataset, str(row.get("source_id")), int(row.get("start_token", -1)))
                        if key in seen[split]:
                            raise ValueError(f"duplicate example {split} {key}")
                        seen[split].add(key)
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        stats[split][dataset]["windows"] += 1
                        stats[split][dataset]["tokens"] += len(row.get("input_ids", []))
                        stats[split][dataset]["ts_tokens"] += ids.count(token_ids["ts"])
    finally:
        for handle in handles.values():
            handle.close()
    files = {}
    for split in handles:
        p = args.output_dir / f"{split}.jsonl"
        files[split] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "windows": sum(x["windows"] for x in stats[split].values()), "tokens": sum(x["tokens"] for x in stats[split].values()), "ts_tokens": sum(x["ts_tokens"] for x in stats[split].values())}
    manifest = {"status": "ok", "scope": "three_corpus_written_subset", "dataset_scope": "merged_verified_turngpt", "datasets": args.include, "sources": {k: str(v) for k, v in sources.items()}, "source_manifests": source_manifests, "splits": {s: {d: dict(c) for d, c in stats[s].items() if c} for s in handles}, "totals": files, "settings": {"max_length": 256, "keep_length": 64, "overlap_length": 10, "limit_dialogues_per_split": None}, "special_token_ids": token_ids}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
