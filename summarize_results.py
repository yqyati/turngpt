"""Read and verify the packaged main table without loading models or old paths."""
import argparse
import hashlib
import json
from pathlib import Path
from turngpt_metrics import aggregate_classification_metrics

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/main")
    args = parser.parse_args()
    caches = {"gpt2": ROOT / "data/processed/gpt2_turngpt_8dataset_v1/merged/manifest.json",
              "qwen3": ROOT / "data/processed/qwen3_turngpt_verified_8dataset_v2/manifest.json"}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in caches.items()}
    print("Model                  Base bAcc   Tuned bAcc     Base F1    Tuned F1")
    for name in ("gpt2", "qwen3_0p6b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b"):
        results = []
        for kind in ("base", "500step"):
            report = json.loads((args.results_dir / name / f"{kind}_full_metrics.json").read_text())
            if report["data_manifest_sha256"] != hashes["gpt2" if name == "gpt2" else "qwen3"]:
                raise ValueError(f"Data manifest differs: {name}/{kind}")
            if report["threshold_selection"] != "validation_only":
                raise ValueError(f"Unexpected threshold protocol: {name}/{kind}")
            metrics = aggregate_classification_metrics([r["test_confusion_matrix"] for r in report["datasets"].values()])
            for key, value in metrics.items():
                if key.startswith("macro_test_") and abs(value - report[key]) > 1e-12:
                    raise ValueError(f"Metric differs from confusion counts: {name}/{kind}/{key}")
            results.append(metrics)
        base, tuned = results
        print(f'{name:<22} {base["macro_test_bacc"]:10.5f} {tuned["macro_test_bacc"]:12.5f} {base["macro_test_f1"]:11.5f} {tuned["macro_test_f1"]:11.5f}')


if __name__ == "__main__":
    main()
