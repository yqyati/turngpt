# Packaging changes

This directory is a curated copy of the main TurnGPT project as of 18 October 2026. The source directory was read, not moved or edited. Regular files were copied with independent inodes; the package does not use symbolic links or hard links to the source.

## Included artifacts

- Original text sources actually consumed by the eight-corpus preparation pipeline.
- Preserved Taskmaster 2/3 and MetaLWOZ split files in `data/splits/`.
- Complete Qwen3 and GPT-2 train/validation/test caches, with their original manifests.
- The matched-history data and evaluation code for the supplementary 128/256/512-token experiment.
- Source manifests and the historical cross-tokenizer data audit, without duplicate intermediate JSONL caches.
- Main training, classification, loss, diagnostic, and latency implementations.
- Main-result metrics and training reports for GPT-2 and four Qwen3 sizes; historical latency and context results.
- Five small CPU test modules and the original TurnGPT paper/license.

Model weights, full reference repositories, old training trials, duplicate token caches, BNC scene experiments, and MultiWOZ task-state pilots are excluded. The original project retains all of these.

## Code changes in this copy

1. Extracted shared special-token constants and dtype conversion into `turngpt_common.py`, removing the runtime dependency on the old smoke-test program.
2. Made model paths explicit: training and all preparers require `--model`; evaluators require a checkpoint or Base path. The history evaluator also requires an explicit checkpoint.
3. Changed the default main cache from the historical three-corpus cache to the formal eight-corpus Qwen3 cache.
4. Changed training defaults from 5 to 500 steps and from 32 to 64 validation-loss windows per corpus. Classification, loss, and diagnostic evaluation now default to a per-corpus cap of 100,000, covering the full current validation/test caches. Loss and diagnostic evaluation default to test.
5. Relocated stored split indices to `data/splits/`. The merger now requires an explicit source map. The deeper data auditor resolves formal-cache source manifests through `data/provenance/` instead of archived absolute paths.
6. Replaced experiment-specific shell scripts with generic wrappers. `train_eval.sh` receives Base model, data directory, and a new run directory. `prepare_data.sh` receives tokenizer directory and a new output directory, prepares all corpora, merges them, and verifies the cache.
7. Added CPU-only cache verification and a result summarizer that does not follow historical filesystem paths.

The model initialization math, speaker-conditioned forward pass, training loss, optimizer implementation, threshold search, metric aggregation, and scoring implementations are retained. The CPU regression suite checks those behaviors. Saved data and result JSONs preserve their original bytes and hashes; they are not rewritten to disguise the historical run locations.

## Provenance records

`copy_manifest.json` maps every copied artifact to its original path and checksum, records the packaged checksum, and identifies modified code. Files newly authored for this package are listed separately. `verification.json` records completed checks. The original working directory's file sizes and modification timestamps are compared before and after packaging.

The recorded main-result JSONs retain absolute paths to original models, manifests, and runs as provenance. Those paths are not runtime dependencies of the curated workflow. Model weights may be stored anywhere and supplied explicitly. Full environment recreation remains subject to the dependency and hardware qualifications in the README.
