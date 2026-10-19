"""Verify cache checksums, window counts, labels, and speaker alignment on CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULTS = [ROOT / "data/processed/qwen3_turngpt_verified_8dataset_v2",
            ROOT / "data/processed/gpt2_turngpt_8dataset_v1/merged"]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    token_ids = manifest["special_token_ids"]
    ts = token_ids["ts"]
    speakers = {token_ids["speaker1"], token_ids["speaker2"]}
    result = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        if sha256(path) != manifest["totals"][split]["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        counts = {name: Counter() for name in manifest["datasets"]}
        keys = set()
        positions = positives = 0
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                ids, states = row["input_ids"], row["speaker_ids"]
                name = row["dataset"]
                key = (name, str(row["source_id"]), row["start_token"])
                if (name not in counts or row["source_split"] != split or key in keys or
                        not 64 <= len(ids) <= 256 or len(ids) != len(states) or
                        not set(states) <= speakers or any(not isinstance(i, int) or not 0 <= i <= ts for i in ids)):
                    raise ValueError(f"Invalid window: {path}:{line_no}")
                for i in range(1, len(ids)):
                    expected = (next(s for s in speakers if s != states[i-1])
                                if ids[i-1] == ts else states[i-1])
                    if states[i] != expected:
                        raise ValueError(f"Invalid speaker transition: {path}:{line_no}")
                keys.add(key)
                counts[name].update(windows=1, tokens=len(ids), ts_tokens=ids.count(ts))
                positions += len(ids) - 1
                positives += ids[1:].count(ts)
        if {k: dict(v) for k, v in counts.items()} != manifest["splits"][split]:
            raise ValueError(f"Corpus counts differ: {path}")
        totals = {k: sum(c[k] for c in counts.values()) for k in ("windows", "tokens", "ts_tokens")}
        if any(totals[k] != manifest["totals"][split][k] for k in totals):
            raise ValueError(f"Total counts differ: {path}")
        result[split] = {**totals, "prediction_positions": positions, "positive_targets": positives}
    return {"data_dir": str(directory.resolve()), "status": "ok", "splits": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, action="append", help="Repeat for multiple caches; defaults to both packaged main caches.")
    args = parser.parse_args()
    for directory in args.data_dir or DEFAULTS:
        print(json.dumps(verify(directory), indent=2), flush=True)


if __name__ == "__main__":
    main()
