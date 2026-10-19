# TurnGPT with Qwen3

This graduation project adapts text-based TurnGPT turn-boundary prediction to Qwen3 Base and compares Qwen3-0.6B, 1.7B, 4B, and 8B with a retrained GPT-2 small baseline. The main experiments examine classification performance under a fixed fine-tuning budget, model size, and inference latency. A supplementary experiment compares inference history lengths with a fixed Qwen3-1.7B checkpoint.

This directory is the curated submission package: it contains the data, preprocessing, training, evaluation, CPU checks, and saved main results. It is an independent copy of the relevant files from the original working project. The original `workspace/kth` directory is unchanged.

The five trained **500-step checkpoints are available separately as [GitHub Release assets](https://github.com/yqyati/turngpt/releases/tag/v1.0-500step)**. Model weights are excluded from the Git checkout. Supply a local Base-model or downloaded checkpoint path when running training or inference. Data preparation needs only tokenizer files. No code or data is loaded from the old working directory; old paths inside preserved experimental JSONs are historical metadata.

The project builds on TurnGPT's task formulation and the authors' later simplified implementation, adapting the approach to Qwen3 and a common experimental framework for model comparison. Section 8 describes the methodological relationship and the evaluation settings used in this study.

Package snapshot: 18 October 2026.

## 1. Directory layout

```text
turngpt/
├── README.md
├── requirements.txt
├── scripts/                      # Portable preprocessing, training, and evaluation entry points
├── train_qwen3_turngpt_subset.py  # Shared Qwen3 / GPT-2 fine-tuning
├── eval_qwen3_turngpt_*.py        # Classification, loss, and history-length evaluation
├── eval_qwen3_turn_shift.py       # AP / ROC-AUC / Brier diagnostics
├── prepare_*.py                  # Corpus preparation
├── merge_turngpt_processed.py
├── turngpt_common.py             # Shared task tokens and dtype conversion
├── turngpt_metrics.py
├── benchmark_qwen3_turngpt_latency.py
├── turngpt_context_data.py
├── audit_*.py
├── verify_data.py                # CPU-only checksum, count, and alignment verification
├── summarize_results.py          # Read and check the saved main table without models
├── data/
│   ├── raw/                     # Selected original text data for all eight corpus entries
│   ├── splits/                  # Preserved Taskmaster 2/3 and MetaLWOZ split indices
│   ├── processed/               # Formal Qwen3, GPT-2, and matched-history caches
│   └── provenance/              # Original source manifests and cross-tokenizer audit
├── results/
│   ├── main/                    # Base / 500-step metrics and training reports for five models
│   ├── latency/                 # Per-model historical latency records
│   └── context/                 # Completed 128/256/512-token history analysis
├── tests/                       # Small CPU regression checks
└── docs/                        # Original paper, attribution, and packaging records
```

`data/processed/` contains only the two formal native-token main caches and the supplementary matched-history cache. Intermediate source-cache JSONL files and early trial caches are excluded; their source manifests remain in `data/provenance/`. The preparation entry point can rebuild the intermediate caches from the included raw sources and split indices.

BNC scene-prefix experiments, MultiWOZ task-state pilots, old smoke runs, model checkpoints, and the full reference repositories are outside this package's scope. Shared constants previously imported from a smoke-test file are extracted into `turngpt_common.py`.

## 2. Prediction task and model adaptation

### 2.1 What is predicted?

The input is a sequence of dialogue turns. After normalization, turns are joined with a special `<ts>` token, and a final `<ts>` is appended to the dialogue. For example:

```text
Original turns:
  Speaker A: Hi, can you help me?
  Speaker B: Yes, what do you need?

Normalized sequence:
  hi can you help me<ts> yes what do you need<ts>
```

At each valid next-token prediction position, the model estimates

```text
score(t) = P(next token = <ts> | dialogue prefix through position t).
label(t) = 1 if the actual next token is <ts>, otherwise 0.
```

This is text-based turn-boundary classification using a causal language model. The main benchmark uses neither audio nor acoustic timing. Its labels come from corpus turn segmentation; they are not direct annotations of all possible places where a listener could take the floor.

### 2.2 Speaker state

Each text token has an aligned speaker-state ID, either `<speaker1>` or `<speaker2>`. The state switches after `<ts>`; the boundary token itself retains the preceding speaker's state. The model input is constructed as:

```text
inputs_embeds = token_embedding(input_ids) + token_embedding(speaker_ids)
```

The speaker rows belong to the model's embedding table. They are input states, rather than extra text positions or classification labels. Padding is hidden by the attention mask, and its training labels are set to `-100`.

### 2.3 Fine-tuning objective and initialization

Training uses the standard causal next-token language-model loss over valid sequence positions, including ordinary text and `<ts>`. There is no separate binary classifier in the main model, and this implementation does not use the reference repository's optional future-turn projection head. All model parameters are fine-tuned; LoRA is not used.

The tokenizer is extended with `<ts>`, `<speaker1>`, and `<speaker2>`. The `<ts>` embedding is initialized from the mean of the embeddings of `!`, `?`, and `.`. Qwen3 speaker rows use a zero-mean normal initialization with the model's configured initialization standard deviation.

GPT-2 speaker rows use the pretrained vocabulary embedding mean plus normal noise with standard deviation `0.02`. This separate initialization prevents newly added, zero-centered output rows from disproportionately dominating GPT-2's softmax. The difference is recorded in the saved reports and should be acknowledged when interpreting the cross-family comparison.

Qwen3-0.6B, 1.7B, and 4B share input/output embedding weights in the local configurations. Qwen3-8B has an independent output head; the initialization routine therefore initializes its new output rows separately as well.

The Base condition in the results means the pretrained model **after the same task-token extension and initialization, but before task fine-tuning**. It is not an untouched pretrained vocabulary evaluated without adaptation.

## 3. Code and entry points

| Purpose | Entry point | Python implementation |
|---|---|---|
| Prepare all eight corpus entries with either tokenizer | `bash scripts/prepare_data.sh TOKENIZER_DIR NEW_OUTPUT_DIR` | `prepare_qwen3_turngpt_text.py`, `prepare_ccpe_turngpt.py`, `prepare_taskmaster1_turngpt.py`, `prepare_taskmaster_turngpt.py`, `prepare_metawoz_turngpt.py`, `merge_turngpt_processed.py` |
| Train from a Base model | `bash scripts/train.sh --model ... --output-dir ...` | `train_qwen3_turngpt_subset.py` |
| Train, evaluate fine-tuned model, evaluate Base | `bash scripts/train_eval.sh BASE_MODEL DATA_DIR NEW_RUN_DIR` | Same trainer and classification evaluator |
| Full validation-selected classification | `bash scripts/eval_bacc.sh --checkpoint ... --output ...` | `eval_qwen3_turngpt_paper_metrics.py` |
| Language-model loss / perplexity | `bash scripts/eval_loss.sh --checkpoint ... --output ...` | `eval_qwen3_turngpt_subset.py` |
| AP / ROC-AUC / Brier diagnostics | `bash scripts/eval_turn_shift.sh --checkpoint ... --output ...` | `eval_qwen3_turn_shift.py` |
| One checkpoint's scoring latency | `bash scripts/benchmark_latency.sh --checkpoint ... --output ...` | `benchmark_qwen3_turngpt_latency.py` |
| Matched inference history evaluation | `bash scripts/eval_context_lengths.sh --checkpoint ... --output-dir ...` | `eval_qwen3_turngpt_context.py` |
| Verify data on CPU | `python verify_data.py` | Validates both formal main caches without model weights |
| Review the saved main table | `python summarize_results.py` | Recomputes metrics from saved confusion counts |

The main training functions are `initialize_turngpt_embeddings`, `make_collator`, `forward`, and `train`. Classification evaluation uses `collect` to obtain probabilities and `best_threshold` to select validation thresholds. `turngpt_metrics.py` defines metrics and aggregation.

The five preparation scripts correspond to the three initial corpora, CCPE, Taskmaster 1, Taskmaster 2/3, and MetaLWOZ. They share task constants through `turngpt_common.py`; the three-corpus parser imports `read_split` from `audit_dailydialog.py`. The GPT-2/Qwen3 provenance and retained-text comparison is implemented in `audit_gpt2_turngpt_data.py`.

Historical filenames containing `subset` are retained to keep the computational implementation recognizable. They handle the formal eight-corpus experiment. `paper_metrics` implements TurnGPT's bAcc measure using the validation-based evaluation protocol in Section 5. Training uses Hugging Face Transformers directly, without the old reference Lightning trainer.

## 4. Datasets and splits

### 4.1 Main benchmark corpus inventory

The main table uses eight **text corpus entries**. Taskmaster 1, 2, and 3 are counted separately. This counting convention differs from the original paper.

| Corpus | Local source under `data/raw/` | Split policy and selected configuration |
|---|---|---|
| DailyDialog | `dailydialog/{train,validation,test}.zip` | Supplied train/validation/test archives. |
| PersonaChat | `personachat_parlai/personachat/{train,valid,test}_both_original.txt` | Official ParlAI `both_original` splits; only conversational turns are used, without persona descriptions or candidate responses. |
| MultiWOZ 2.1 | `multiwoz21/MultiWOZ_2.1.zip` | Supplied `valListFile.txt` and `testListFile.txt`; the remaining dialogues form train. The main benchmark uses turn text, without dialogue goals or task-state prefixes. |
| CCPE | `ccpe/data.json` | Author release with 502 source dialogues. Seed-1 SHA-256 buckets implement a nominal 90/5/5 dialogue split, producing 461/18/23 dialogues. |
| Taskmaster 1 | `taskmaster/taskmaster1/` | `self-dialogs.json` with `train.csv`, `dev.csv`, and `test.csv`; this is the one-person/self-dialogue configuration. |
| Taskmaster 2 | `taskmaster/taskmaster2/` | Flights, food-ordering, and restaurant-search JSON files, using `taskmaster2_splits.json` from `data/splits/`. |
| Taskmaster 3 | `taskmaster/taskmaster3/` | Sorted `data_*.json` files, using the stored `taskmaster3_splits.json` indices. |
| MetaLWOZ | `metawoz/metalwoz-v1.zip`, `metalwoz-test-v1.zip` | Stored `metawoz_splits.json` train/validation indices; test comes from the nested official held-out archive. |

Taskmaster 2/3 stored indices approximate 90/5/5 on the selected source configurations. Their exact membership depends on the preserved source ordering and split files; it should not be regenerated from an arbitrary new shuffle. MetaLWOZ's training/validation indices split the original training archive approximately 90/10, while test is separate held-out data.

Neither **MapTask nor Switchboard** is included in the main training or test cache. BNC2014 scene data is also separate from the main benchmark. The presence of spoken-dialogue loaders in the reference repository does not mean those corpora were used in the main table.

### 4.2 Dialogue counts before token windows

The following counts are accepted dialogues after the preparation scripts' filtering, before short token-window tails are discarded. An accepted dialogue can produce multiple windows or no retained window.

| Corpus | Train dialogues | Validation dialogues | Test dialogues |
|---|---:|---:|---:|
| DailyDialog | 11,117 | 1,000 | 1,000 |
| PersonaChat | 8,938 | 1,000 | 968 |
| MultiWOZ 2.1 | 8,438 | 1,000 | 1,000 |
| CCPE | 461 | 18 | 23 |
| Taskmaster 1 | 6,165 | 769 | 769 |
| Taskmaster 2 | 6,126 | 340 | 341 |
| Taskmaster 3 | 21,374 | 1,188 | 1,188 |
| MetaLWOZ | 34,038 | 3,783 | 2,318 |

Filtering is corpus-specific. The scripts remove invalid or empty-turn dialogues; some also skip empty utterances before checking the remaining dialogue. Adjacent utterances are merged where the CCPE and Taskmaster 2/3 speaker labels identify the same speaker. Taskmaster 1 follows its self-dialogue turn list. The exact rules are in the preparation files listed in Section 3.

### 4.3 Text normalization and window construction

The normalization mirrors the reference `SpokenNormalizer`: lowercase text, remove Unicode combining accents, insert spacing around selected punctuation patterns, remove the configured punctuation set, and collapse repeated whitespace. The removed characters include `. , ! ? : ; ( ) [ ] " -`; this is not a blanket removal of every punctuation character. The implementation is in `normalize_reference` and equivalent corpus-specific helpers.

Each dialogue is tokenized independently. Its token sequence is cut into windows with:

| Setting | Value | Meaning |
|---|---:|---|
| `max_length` | 256 | Maximum token count in one window. |
| `keep_length` | 64 | Minimum retained window length; shorter final tails are discarded. |
| `overlap_length` | 10 | Overlap between neighboring full windows. |
| Stride | 246 | `256 - 10`. |

The value `64` is a minimum retained length, not a fixed 64-token history for the main benchmark. No window combines different source dialogues. Overlap is retained during evaluation, so some source positions appear in more than one window; the published totals are window-based prediction counts rather than counts of unique source positions.

### 4.4 Formal processed data

Qwen3 cache: [`data/processed/qwen3_turngpt_verified_8dataset_v2`](data/processed/qwen3_turngpt_verified_8dataset_v2/).

GPT-2 cache: [`data/processed/gpt2_turngpt_8dataset_v1/merged`](data/processed/gpt2_turngpt_8dataset_v1/merged/).

| Corpus | Qwen3 train windows | Qwen3 validation windows | Qwen3 test windows | GPT-2 test windows |
|---|---:|---:|---:|---:|
| DailyDialog | 7,531 | 721 | 679 | 680 |
| PersonaChat | 8,956 | 1,000 | 969 | 969 |
| MultiWOZ 2.1 | 10,052 | 1,253 | 1,234 | 1,163 |
| CCPE | 752 | 26 | 39 | 39 |
| Taskmaster 1 | 7,156 | 886 | 883 | 858 |
| Taskmaster 2 | 6,998 | 389 | 378 | 369 |
| Taskmaster 3 | 28,044 | 1,547 | 1,586 | 1,517 |
| MetaLWOZ | 32,075 | 3,555 | 2,200 | 2,202 |
| **Total** | **101,564** | **9,377** | **7,968** | **7,797** |

GPT-2 has 98,998 train windows and 9,143 validation windows. Its data is reconstructed from the same source dialogues and split policies with its native tokenizer and the same 256/64/10 settings. It does not use Qwen3 token IDs or windows.

For Qwen3, the test manifest contains 1,339,818 stored token occurrences and 106,304 `<ts>` occurrences. Classification excludes each window's first token, which has no preceding position inside that window, as well as padding. The resulting test evaluation contains **1,331,850 valid next-token positions and 106,259 positive boundary targets**. GPT-2 contains **1,290,544 valid test positions and 106,392 positive targets**. These are the actual denominators for the main classification results.

### 4.5 Cache format and provenance

Each cache contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`. One JSONL row has this schema:

```json
{
  "dataset": "multiwoz21",
  "source_split": "test",
  "source_id": "<original dialogue identifier>",
  "start_token": 0,
  "input_ids": ["<integer token IDs>"],
  "speaker_ids": ["<aligned integer speaker-state IDs>"]
}
```

The placeholders above illustrate the schema; actual arrays contain integers. `start_token` gives the window offset in its source dialogue. `input_ids` and `speaker_ids` must have equal lengths.

Qwen3's added IDs are `<ts>=151669`, `<speaker1>=151670`, and `<speaker2>=151671`. GPT-2's are `50257`, `50258`, and `50259`, respectively. The four local Qwen3 Base tokenizers were checked for matching tokenizer files, allowing them to share the Qwen3 cache. The merger validates special-token IDs before combining sources.

Manifests record corpus counts, window settings, special-token IDs, and cache hashes. Source manifests additionally record available raw-file hashes and source information; the GPT-2 [data comparison report](data/provenance/data_comparison.json) checks source consistency across tokenizers. The Qwen3 v2 manifest's `scope=three_corpus_written_subset` is a legacy reader-compatibility field. Its `datasets`, `dataset_scope`, and `splits` describe the actual eight-corpus content.

The source manifests copied to `data/provenance/qwen3/` and `data/provenance/gpt2/` retain their original bytes. The formal cache manifests are also unchanged, preserving the hashes referenced by historical results. The audit code resolves these packaged source manifests locally; it does not follow their archived absolute source paths.

## 5. Training and evaluation protocol

### 5.1 Main training recipe

| Setting | Main experiment |
|---|---|
| Models | GPT-2 small, Qwen3-0.6B/1.7B/4B/8B Base. |
| Fine-tuning | All parameters; 500 optimizer updates from each initialized Base. |
| Microbatch / accumulation | 2 windows / 2 microbatches; effective batch of 4 windows. |
| Optimizer / learning rate | AdamW; fixed learning rate `6.25e-5`. |
| Random seed | 1. |
| Training order | Shuffled window-level DataLoader; no equal-per-corpus sampling. |
| Validation during training | Initial and final language-model loss on at most 64 fixed sampled windows per corpus. |
| Hardware / precision | Single NVIDIA H200; bfloat16. |
| Checkpoint saved | Final model and tokenizer; no optimizer-state resume interface. |

500 updates process approximately 2,000 training-window presentations. The full train cache is available, but the fixed budget does not traverse all 101,564 Qwen3 training windows or establish convergence. The experiments compare a common update budget, not equal FLOPs, equal wall-clock time, or independently optimized training schedules.

The older 100/200/500-step main checkpoints were independently trained from Base. Passing an existing fine-tuned model to the main trainer is not seamless continuation: it reinitializes task-token embeddings and creates a fresh optimizer. 

### 5.2 Main classification evaluation

For **each model and each corpus**, the evaluator selects the threshold maximizing balanced accuracy on validation. It then fixes that threshold for test:

```text
bAcc = 0.5 × [TP / (TP + FN) + TN / (TN + FP)]
prediction = 1 when P(<ts>) >= validation-selected threshold
```

Precision, recall, F1, and accuracy use that same bAcc-selected threshold. F1 is the positive-boundary F1 and is not independently maximized.

The main table is **corpus-macro**: compute each metric per corpus, then take the unweighted mean across the eight entries. Thus, CCPE and MetaLWOZ receive equal corpus weight despite their different sizes. Saved reports also provide position-weighted averages and pooled metrics obtained after adding confusion counts. These aggregation methods should not be interchanged.

Formal main-table evaluation uses all 9,377 validation and 7,968 test Qwen3 windows, or all 9,143/7,797 GPT-2 windows. **The cleaned classification, loss, and diagnostic entry points default to `--windows-per-dataset 100000`, covering the full current caches.** The old working-directory wrappers defaulted to 1,000; this package changes that default to make main-table reproduction the standard behavior. For classification, this cap applies to both validation and test. Classification and diagnostic evaluation require a positive cap; the loss evaluator separately supports zero for all windows. Using `100000` works for all three evaluators on the current caches.

## 6. Environment and usage

### 6.0 Clone the repository and download data

Raw data and processed JSONL files are stored with Git LFS. Install [Git LFS](https://git-lfs.com/) before cloning, then run:

```bash
git lfs install
git clone https://github.com/yqyati/turngpt.git
cd turngpt
git lfs pull
```

The data files occupy approximately 1.1 GiB. `git lfs pull` ensures that the actual datasets are downloaded rather than only their pointer files. Model weights are supplied separately as described below.

### 6.0.1 Download the trained checkpoints

The [v1.0-500step release](https://github.com/yqyati/turngpt/releases/tag/v1.0-500step) contains the exact seed-1, 500-update checkpoints used in Section 7: GPT-2 small and Qwen3-0.6B/1.7B/4B/8B. Each model package includes its weights, configuration, tokenizer, training report, full evaluation results, and upstream model documentation. The five checkpoints occupy approximately 27.35 GiB in total. Optimizer state is not included.

GPT-2 and Qwen3-0.6B use a single `.tar` asset each. Larger archives are split into numbered parts of at most 1,900 MiB to fit GitHub's per-asset limit. Download every part for a selected model, verify its SHA-256 checksums, and concatenate the parts in numerical order before extraction. The release includes `SHA256SUMS`, `checkpoint_manifest.json`, upstream license texts, and [DOWNLOAD.md](https://github.com/yqyati/turngpt/releases/download/v1.0-500step/DOWNLOAD.md) with full instructions.

For example, using the GitHub CLI from the repository root:

```bash
gh release download v1.0-500step --repo yqyati/turngpt \
  --dir checkpoints/downloads \
  --pattern 'turngpt-qwen3-1p7b-500step.tar.part*' \
  --pattern 'SHA256SUMS' --pattern 'checkpoint_manifest.json' --pattern 'DOWNLOAD.md' \
  --pattern 'LICENSE-QWEN3.txt'
cd checkpoints/downloads
sha256sum --check --ignore-missing SHA256SUMS
mkdir -p ../extracted
cat turngpt-qwen3-1p7b-500step.tar.part* | tar -xf - -C ../extracted
cd ../..
TG_CHECKPOINT="$PWD/checkpoints/extracted/turngpt-qwen3-1p7b-500step/model"
```

For a selective download, confirm that every asset for that model listed in `checkpoint_manifest.json` is present; `--ignore-missing` skips other models' files. Pass the extracted `model/` path to the evaluation and latency commands below. The project code implements the speaker-conditioned TurnGPT forward pass used for these checkpoints.

### 6.1 Set up the runtime

The recorded experiments used PyTorch `2.8.0+cu128`, Transformers `4.56.2`, and a single NVIDIA H200 with bfloat16. The current machine already provides these libraries in the `verl` Conda environment:

```bash
cd /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/turngpt
conda activate verl
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

On another machine, create a Python 3.12 environment, install an appropriate PyTorch 2.8.0 build for its CUDA runtime, then install the dependencies in `requirements.txt`. This file records direct dependencies; it is not a complete lockfile for the historical environment.

```bash
python -m pip install -r requirements.txt
```

All following commands assume the package root as the working directory. Wrappers accept `TG_PYTHON=/path/to/python` if a specific interpreter is needed. GPU experiments are manually started in an existing GPU environment; these scripts do not allocate cluster resources. Project operation rules are in [AGENTS.md](AGENTS.md).

Before using examples with `TG_BASE` or `TG_CHECKPOINT`, assign real local paths. They must point to Hugging Face-format directories with the model configuration, tokenizer files, and required weight files. Loading is local-only.

```bash
TG_BASE=/path/to/Qwen3-0.6B-Base
TG_CHECKPOINT=/path/to/turngpt_checkpoint/model
```

The original machine's 1.7B/4B/8B Base models are under `/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/model/`. The original 0.6B Base and GPT-2 Base are under `workspace/kth/model/`. Existing fine-tuned weights also remain in the original project. These may be supplied explicitly as model inputs; the submission does not contain or automatically select them.

### 6.2 Review the data and completed results without a GPU

```bash
python verify_data.py
python summarize_results.py
```

`verify_data.py` checks the full Qwen3 and GPT-2 main caches: SHA-256 hashes, per-corpus window/token counts, split labels, duplicate window keys, and speaker alignment. `summarize_results.py` verifies the copied main-result manifest hashes and recomputes macro metrics from confusion counts. Neither command loads a model.

### 6.3 Train and evaluate a Qwen3 model

```bash
bash scripts/train_eval.sh \
  "$TG_BASE" \
  data/processed/qwen3_turngpt_verified_8dataset_v2 \
  runs/qwen3_reproduction
```

This runs 500 optimizer updates, full classification evaluation of the resulting checkpoint, and full evaluation of the initialized Base. The output directory must not exist. It creates:

```text
runs/qwen3_reproduction/
├── train/model/                  # Final checkpoint and tokenizer
├── train/train_report.json
├── train.log
├── 500step_full_metrics.json
├── 500step_eval.log
├── base_full_metrics.json
└── base_eval.log
```

Use the same entry point for 0.6B, 1.7B, 4B, or 8B by changing `TG_BASE`. The local Qwen3 tokenizers used in the original experiments were identical; a replacement tokenizer must match the cache's vocabulary and special-token IDs.

For individual stages or parameter overrides:

```bash
bash scripts/train.sh \
  --model "$TG_BASE" \
  --output-dir runs/qwen3_individual/train \
  --steps 500 --batch-size 2 --grad-accum 2 \
  --learning-rate 6.25e-5 --seed 1

bash scripts/eval_bacc.sh \
  --checkpoint runs/qwen3_individual/train/model \
  --windows-per-dataset 100000 \
  --output runs/qwen3_individual/500step_full_metrics.json

bash scripts/eval_bacc.sh \
  --base-model "$TG_BASE" --init-seed 1 \
  --windows-per-dataset 100000 \
  --output runs/qwen3_individual/base_full_metrics.json
```

Training defaults are 500 steps, batch size 2, accumulation 2, learning rate `6.25e-5`, seed 1, and up to 64 validation-loss windows per corpus. The default data is the included Qwen3 eight-corpus cache. A model path is required. Evaluators require `--checkpoint` or `--base-model`; `--base-model` selects fresh task-token initialization and takes precedence if both are supplied.

Training requires an empty or absent output directory; classification output JSONs must not already exist. Use new output names for repeated runs.

### 6.4 Train and evaluate GPT-2

```bash
TG_GPT2_BASE=/path/to/gpt2
bash scripts/train_eval.sh \
  "$TG_GPT2_BASE" \
  data/processed/gpt2_turngpt_8dataset_v1/merged \
  runs/gpt2_reproduction
```

The original baseline used `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Use a compatible local GPT-2 small model and tokenizer. The shared trainer automatically selects the GPT-2 speaker initialization described in Section 2. GPT-2 must use its own processed cache, rather than the default Qwen3 cache.

For standalone GPT-2 evaluation:

```bash
bash scripts/eval_bacc.sh \
  --checkpoint runs/gpt2_reproduction/train/model \
  --data-dir data/processed/gpt2_turngpt_8dataset_v1/merged \
  --output runs/gpt2_reproduction/independent_metrics.json
```

### 6.5 Loss, diagnostics, latency, and history length

```bash
bash scripts/eval_loss.sh \
  --checkpoint "$TG_CHECKPOINT" \
  --output runs/diagnostics/loss.json

bash scripts/eval_turn_shift.sh \
  --checkpoint "$TG_CHECKPOINT" \
  --output runs/diagnostics/turn_shift.json

bash scripts/benchmark_latency.sh \
  --checkpoint "$TG_CHECKPOINT" \
  --context-length 256 --windows-per-dataset 32 --warmup 20 --repeats 3 \
  --output runs/latency/model.json
```

The examples default to Qwen3 data. Add `--data-dir data/processed/gpt2_turngpt_8dataset_v1/merged` for GPT-2 loss, diagnostics, or latency. The latency wrapper measures the explicitly supplied checkpoint once; repeat with other checkpoints and new output paths for a size comparison.

The original Qwen3 latency protocol uses batch size 1, bfloat16, SDPA, no KV-cache reuse, 20 warm-up requests, and three repetitions over 184 retained real 256-token windows. Timing includes the forward pass, probability calculation, and returning the scalar result to CPU. It excludes loading, tokenization, input transfer, and warm-up. GPT-2 uses native 256-token windows, so equal token counts do not imply equal text spans. These are scoring latencies, rather than complete conversational-system response times.

The supplementary matched-history experiment uses the original Qwen3-1.7B 500-step checkpoint:

```bash
TG_1P7B_CHECKPOINT=/path/to/qwen3_1p7b_500step/train/model
bash scripts/eval_context_lengths.sh \
  --checkpoint "$TG_1P7B_CHECKPOINT" \
  --output-dir runs/context_lengths
```

Its included data is `data/processed/qwen3_turngpt_context_seed1_v1`. The evaluator compares the same targets at 128/256/512 tokens. To reconstruct this data from the formal Qwen3 windows:

```bash
python turngpt_context_data.py --output-dir data/processed/rebuilt_context
```

### 6.6 Rebuild either native-token cache

Both tokenized main caches are already present. To reproduce preprocessing from the included raw sources, supply a local tokenizer directory and a new output directory:

```bash
bash scripts/prepare_data.sh "$TG_BASE" data/processed/rebuilt_qwen3
bash scripts/prepare_data.sh "$TG_GPT2_BASE" data/processed/rebuilt_gpt2
```

Each run executes the five corpus-preparation pipelines with 256/64/10 settings, writes `source_map.json`, merges the eight entries into `merged/`, and runs cache verification. CCPE uses its default seed 1. The pipeline uses `data/splits/` for the stored Taskmaster 2/3 and MetaLWOZ indices. Tokenization runs on CPU and does not load model weights.

A rebuilt cache is used by passing `--data-dir data/processed/rebuilt_qwen3/merged` or its GPT-2 counterpart to training/evaluation. Intermediate source caches are left in the new output directory for inspection.

Each preparer can also be run individually:

```bash
python prepare_ccpe_turngpt.py \
  --model "$TG_BASE" --seed 1 \
  --output-dir data/processed/rebuilt_ccpe
```

The merger requires an explicit `--source-map` JSON mapping corpus names to cache directories; relative values are resolved against that JSON file's directory. Use the generated `source_map.json` as an example.

For the deeper original-source and retained-text comparison between the included Qwen3/GPT-2 caches:

```bash
python audit_gpt2_turngpt_data.py \
  --gpt2-data data/processed/gpt2_turngpt_8dataset_v1/merged \
  --gpt2-model "$TG_GPT2_BASE" \
  --qwen-model "$TG_BASE" \
  --output runs/data_audit.json
```

This CPU audit uses the copied source manifests in `data/provenance/` for the formal caches, and the actual source-cache paths for newly rebuilt caches. The preserved historical audit is [data/provenance/data_comparison.json](data/provenance/data_comparison.json).

## 7. Main results and saved evidence

### 7.1 Full-test corpus-macro results

All values use validation-selected bAcc thresholds and the full native-token test caches.

| Model | Base bAcc | 500-step bAcc | Base F1 | 500-step F1 |
|---|---:|---:|---:|---:|
| GPT-2 small | 0.76897 | 0.87700 | 0.31918 | 0.47207 |
| Qwen3-0.6B | 0.85327 | 0.91087 | 0.42270 | 0.56280 |
| Qwen3-1.7B | 0.88345 | 0.92062 | 0.47589 | 0.58435 |
| Qwen3-4B | 0.87813 | 0.92136 | 0.47161 | 0.58044 |
| Qwen3-8B | 0.89336 | 0.92536 | 0.49612 | 0.59435 |

Every model benefits from task fine-tuning under this protocol. Qwen3's fine-tuned macro bAcc increases with size; F1 improves overall with a local decrease at 4B. These are single-seed, fixed-budget results without confidence intervals or proof of convergence. Cross-family differences do not isolate architecture, parameter count, pretraining data, tokenization, or initialization as the sole cause.

### 7.2 Included result files

Each directory below contains `base_full_metrics.json`, `500step_full_metrics.json`, and `train_report.json`:

| Model | Result directory |
|---|---|
| GPT-2 small | [results/main/gpt2/](results/main/gpt2/) |
| Qwen3-0.6B | [results/main/qwen3_0p6b/](results/main/qwen3_0p6b/) |
| Qwen3-1.7B | [results/main/qwen3_1p7b/](results/main/qwen3_1p7b/) |
| Qwen3-4B | [results/main/qwen3_4b/](results/main/qwen3_4b/) |
| Qwen3-8B | [results/main/qwen3_8b/](results/main/qwen3_8b/) |

The JSONs preserve the original run metadata and bytes. Their old model/data paths document where those experiments ran; the local summarizer reads only the packaged files and does not follow those paths. They describe historical experiments, rather than claiming that GPU training has been rerun after packaging.

[results/latency/](results/latency/) contains the five model-specific latency records, including raw timings and environment details. Mean latencies for GPT-2 / Qwen3-0.6B / 1.7B / 4B / 8B were `2.620 / 12.507 / 12.657 / 15.678 / 16.079 ms`. GPT-2 was measured separately from the four Qwen3 models; interpret the historical comparison with that limitation.

[results/context/context_length_metrics.json](results/context/context_length_metrics.json) records the fixed-1.7B matched-target history analysis. Increasing inference history from 128 to 512 tokens did not produce a consistent classification improvement; the checkpoint itself was task-trained on 256-token windows.

## 8. Relationship to TurnGPT and experimental design

This study builds on the causal language-model approach introduced by Erik Ekstedt and Gabriel Skantze (2020) in *TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog*. It retains text normalization, speaker-conditioned inputs, and probabilistic turn-boundary scoring, and extends the experimental comparison to four Qwen3 Base sizes alongside a retrained GPT-2 baseline.

The implementation draws on the authors' later simplified TurnGPT branch. The reference code and data-repository revisions are `64f7d8c3b00e2a8332bca9267e6252462cbd0407` and `8236f3188798381a4815250a0337a7a090137540`, respectively. The paper is included as [`TurnGPT_2020.pdf`](docs/TurnGPT_2020.pdf).

The following settings define the relationship between the original study and the experiments reported here:

| Aspect | TurnGPT reference | Settings used in this study |
|---|---|---|
| Language model | GPT-2 / DialoGPT-based models. | Qwen3-0.6B, 1.7B, 4B, and 8B, with a retrained GPT-2 small baseline. |
| Boundary representation | The paper scores the maximum probability of two speaker tokens. The later simplified implementation uses a turn-shift token. | A single `<ts>` probability, with speaker identity supplied through aligned speaker embeddings. |
| Corpus coverage | Assistant: Taskmaster, MultiWOZ, MetaLWOZ, CCPE; Written: Persona, DailyDialog; Spoken: MapTask, Switchboard. | Eight text corpus entries drawn from Assistant and Written sources. Taskmaster 1/2/3 are separate entries; MapTask and Switchboard are outside this evaluation set. Exact releases and splits are specified in Section 4. |
| Text processing | ASR-style normalization and corpus-specific preparation; the later repository provides 256/64/10 window defaults. | The repository's normalization and windowing approach, using the preserved source files and split indices described in Section 4. |
| Tokenization | GPT-2-family BPE. | Each model family uses its native tokenizer. Qwen3 sizes share one cache; GPT-2 uses a separately tokenized cache from the same source dialogues and splits. |
| Decision threshold | Paper Section 6.2 describes selecting a threshold for the highest test score. | Per-model, per-corpus thresholds selected on validation and fixed for test. |
| Metrics and aggregation | bAcc reported for Assistant, Spoken, and Written test groups. | The same bAcc formula, plus precision, recall, F1, and accuracy; corpus-macro results form the main table, with weighted and pooled summaries available. |
| Training budget | Training configurations of the original study. | A common 500-update fine-tuning recipe, fixed learning rate, and seed 1 for all five models. |

The retrained GPT-2 model provides the empirical reference for the main table under this study's source-data and training-budget settings. Within Qwen3, all four sizes share the tokenized data and evaluation procedure. Across GPT-2 and Qwen3, native tokenization changes window endpoints and prediction positions; the model-specific speaker initialization is described in Section 2.

Quantitative conclusions in this README therefore refer to the models evaluated under the protocol in Section 5. Published TurnGPT results provide the methodological and literature context, while the locally retrained GPT-2 baseline anchors the reported performance comparison.

## 9. CPU checks, provenance, and package scope

Run the included small-model and metric checks without a GPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

These checks cover tied/untied embedding initialization, GPT-2 compatibility, native-token cache merging, classification aggregation, latency-score equivalence, and causal matched-history construction. They use small randomly initialized models or synthetic inputs. They do not start full model training or establish new experimental results.

The raw files, formal processed data, stored split indices, and copied result JSONs are byte-for-byte copies. [docs/copy_manifest.json](docs/copy_manifest.json) records the original relative path, source checksum, destination path, and packaged checksum of each copied artifact. Modified code is marked separately. [docs/packaging_notes.md](docs/packaging_notes.md) describes changes to defaults and paths. [docs/verification.json](docs/verification.json) records packaging checks.

Only direct workflow dependencies are retained. Excluded items include model weights, duplicate intermediate token caches, abandoned dataset candidates, exploratory BNC/task-state experiments, old checkpoints, and full reference repository checkouts. The failed Context pilots are not included as improvement claims. VAP/audio-text fusion is not implemented or evaluated in this package.

The reference implementation's license is preserved in [docs/TurnGPT_LICENSE.txt](docs/TurnGPT_LICENSE.txt), with attribution in [docs/attribution.md](docs/attribution.md). Dataset sources and model weights retain their providers' licenses and access conditions.

When reporting results, identify the cache/tokenizer, checkpoint, training budget, threshold source, aggregation method, environment, and seed. Distinguish full-test from sampled metrics, token occurrences from valid prediction targets, and corpus-macro from pooled results.
