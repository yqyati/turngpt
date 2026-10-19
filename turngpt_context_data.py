"""Build matched next-token targets from verified, overlapping cached windows.

Only retained contiguous dialogue prefixes are recovered. Discarded short tails
are not reconstructed. Sampling is independent of labels and model scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data/processed/qwen3_turngpt_verified_8dataset_v2"
LENGTHS = (128, 256, 512)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def recover_dialogues(path, manifest, split):
    """Verify file hash, counts, overlaps and state alignment before stitching."""
    if file_hash(path) != manifest["totals"][split]["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path}")
    expected = manifest["splits"][split]
    chunks = {name: {} for name in expected}
    counts = Counter()
    tokens = manifest["special_token_ids"]
    max_length = manifest["settings"]["max_length"]
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            name, sid, start = row["dataset"], str(row["source_id"]), row["start_token"]
            ids, states = row["input_ids"], row["speaker_ids"]
            if row["source_split"] != split or name not in expected:
                raise ValueError("Unexpected source split/corpus")
            if not isinstance(start, int) or start < 0 or not 2 <= len(ids) <= max_length:
                raise ValueError("Invalid source offset/length")
            if len(ids) != len(states) or any(not isinstance(t, int) or not 0 <= t <= tokens["ts"] for t in ids):
                raise ValueError("Invalid text tokens/state alignment")
            if not set(states) <= {tokens["speaker1"], tokens["speaker2"]}:
                raise ValueError("Invalid speaker states")
            offsets = chunks[name].setdefault(sid, {})
            if start in offsets:
                raise ValueError("Duplicate source window")
            offsets[start] = (ids, states)
            counts[name] += 1
    if dict(counts) != {name: info["windows"] for name, info in expected.items()}:
        raise ValueError("Source window counts differ from manifest")
    groups = {name: [] for name in expected}
    for name, dialogues in chunks.items():
        for sid, offsets in sorted(dialogues.items()):
            ids, states = [], []
            for start, (part_ids, part_states) in sorted(offsets.items()):
                if start > len(ids):
                    raise ValueError(f"Gap in retained dialogue: {name}/{sid}/{start}")
                overlap = min(len(ids) - start, len(part_ids))
                if ids[start:start + overlap] != part_ids[:overlap] or states[start:start + overlap] != part_states[:overlap]:
                    raise ValueError(f"Conflicting overlapping windows: {name}/{sid}/{start}")
                ids.extend(part_ids[overlap:])
                states.extend(part_states[overlap:])
            for i in range(1, len(ids)):
                expected_state = states[i - 1]
                if ids[i - 1] == tokens["ts"]:
                    expected_state = tokens["speaker2"] if expected_state == tokens["speaker1"] else tokens["speaker1"]
                if states[i] != expected_state:
                    raise ValueError(f"Speaker transition mismatch: {name}/{sid}/{i}")
            groups[name].append({"source_id": sid, "input_ids": ids, "speaker_ids": states})
    return groups


def sample_refs(dialogues, minimum_history, limit, seed_key):
    counts = [max(0, len(row["input_ids"]) - minimum_history) for row in dialogues]
    total = sum(counts)
    chosen = sorted(random.Random(seed_key).sample(range(total), min(limit, total)))
    refs = []
    offset = index = 0
    for dialogue_index, count in enumerate(counts):
        while index < len(chosen) and chosen[index] < offset + count:
            refs.append((dialogue_index, minimum_history + chosen[index] - offset))
            index += 1
        offset += count
    return refs, total


def make_sample(name, split, dialogue, position, ts_id, memberships):
    start = max(0, position - max(LENGTHS))
    return {
        "sample_id": object_hash([name, split, dialogue["source_id"], position]),
        "dataset": name, "source_split": split, "source_id": dialogue["source_id"],
        "target_position": position, "available_history_tokens": position,
        "input_ids": dialogue["input_ids"][start:position],
        "speaker_ids": dialogue["speaker_ids"][start:position],
        "label": int(dialogue["input_ids"][position] == ts_id),
        "subsets": sorted(memberships),
    }


def describe(rows):
    return {"targets": len(rows), "positives": sum(row["label"] for row in rows),
            "history_gt128": sum(row["available_history_tokens"] > 128 for row in rows),
            "history_gt256": sum(row["available_history_tokens"] > 256 for row in rows),
            "history_ge512": sum(row["available_history_tokens"] >= 512 for row in rows)}


def build_samples(groups, split, ts_id, targets_per_corpus=4096, long_targets_per_corpus=2048,
                  latency_per_corpus=32, seed=1):
    samples, latency, stats = [], [], {}
    for name, dialogues in groups.items():
        primary, total = sample_refs(dialogues, 1, targets_per_corpus, f"{seed}:{name}:{split}:primary")
        selected = {ref: {"primary"} for ref in primary}
        long_refs, long_total = sample_refs(dialogues, 257, long_targets_per_corpus, f"{seed}:{name}:{split}:long")
        if split == "test":
            for ref in long_refs:
                selected.setdefault(ref, set()).add("history_gt256")
        rows = [make_sample(name, split, dialogues[d], p, ts_id, memberships)
                for (d, p), memberships in sorted(selected.items())]
        primary_rows = [r for r in rows if "primary" in r["subsets"]]
        if not 0 < sum(r["label"] for r in primary_rows) < len(primary_rows):
            raise ValueError(f"Primary sample must have both classes: {name}/{split}")
        samples.extend(rows)
        stats[name] = {"retained_dialogues": len(dialogues), "eligible_primary_targets": total,
                       "eligible_history_gt256_targets": long_total,
                       "primary": describe(primary_rows),
                       "history_gt256": describe([r for r in rows if "history_gt256" in r["subsets"]])}
        if split == "test":
            refs, eligible = sample_refs(dialogues, 512, latency_per_corpus, f"{seed}:{name}:latency")
            latency.extend(make_sample(name, split, dialogues[d], p, ts_id, ["latency"]) for d, p in refs)
            stats[name]["latency"] = {"eligible_targets": eligible, "sampled_targets": len(refs)}
    random.Random(f"{seed}:context:latency:order").shuffle(latency)
    return samples, latency, stats


def write_jsonl(path, rows):
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def prepare(source, output, targets_per_corpus=4096, long_targets_per_corpus=2048, latency_per_corpus=32, seed=1):
    if output.exists():
        raise FileExistsError(output)
    if min(targets_per_corpus, long_targets_per_corpus, latency_per_corpus) < 1:
        raise ValueError("Sample counts must be positive")
    manifest_path = source / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True)
    info = {"status": "ok", "schema_version": 1, "experiment": "matched_context_next_token",
            "source_data": str(source.resolve()), "source_manifest_sha256": file_hash(manifest_path),
            "source_split_sha256": {}, "context_lengths": list(LENGTHS), "seed": seed,
            "targets_per_corpus": targets_per_corpus, "long_targets_per_corpus": long_targets_per_corpus,
            "latency_per_corpus": latency_per_corpus, "special_token_ids": source_manifest["special_token_ids"],
            "sampling": "uniform unique retained next-token positions per corpus, independent of labels",
            "history": "verified contiguous cached prefixes only; discarded tails are excluded",
            "splits": {}, "files": {}}
    for split in ("validation", "test"):
        groups = recover_dialogues(source / f"{split}.jsonl", source_manifest, split)
        rows, latency, stats = build_samples(groups, split, info["special_token_ids"]["ts"],
                                             targets_per_corpus, long_targets_per_corpus, latency_per_corpus, seed)
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        info["files"][path.name] = {"sha256": file_hash(path), "rows": len(rows)}
        info["splits"][split] = stats
        info["source_split_sha256"][split] = source_manifest["totals"][split]["sha256"]
        if split == "test":
            if not latency:
                raise ValueError("No retained targets with at least 512 history tokens for latency")
            path = output / "latency.jsonl"
            write_jsonl(path, latency)
            info["files"][path.name] = {"sha256": file_hash(path), "rows": len(latency)}
    (output / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets-per-corpus", type=int, default=4096)
    parser.add_argument("--long-targets-per-corpus", type=int, default=2048)
    parser.add_argument("--latency-per-corpus", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    info = prepare(args.source, args.output_dir, args.targets_per_corpus,
                   args.long_targets_per_corpus, args.latency_per_corpus, args.seed)
    print(json.dumps({"output": str(args.output_dir), "files": info["files"], "splits": info["splits"]}, indent=2))


if __name__ == "__main__":
    main()
