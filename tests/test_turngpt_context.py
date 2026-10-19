"""Check paired targets, causal input construction, padding and threshold isolation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turngpt_context_data import recover_dialogues, build_samples, make_sample, file_hash
from eval_qwen3_turngpt_context import (
    make_batch, batch_probabilities, score_rows, classification_report,
)
from benchmark_qwen3_turngpt_latency import next_ts_probability


class ContextTests(unittest.TestCase):
    def test_recovery_deduplicates_overlap_and_rejects_gaps_conflicts_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            rows = [dict(dataset="a", source_id="dialogue", source_split="test", start_token=0,
                         input_ids=[2, 3, 12, 4], speaker_ids=[13, 13, 13, 14]),
                    dict(dataset="a", source_id="dialogue", source_split="test", start_token=2,
                         input_ids=[12, 4, 5, 12], speaker_ids=[13, 14, 14, 14])]
            manifest = {"totals": {"test": {}}, "splits": {"test": {"a": {"windows": 2}}},
                        "settings": {"max_length": 4},
                        "special_token_ids": {"ts": 12, "speaker1": 13, "speaker2": 14}}

            def write(values):
                path.write_text("".join(json.dumps(r) + "\n" for r in values))
                manifest["totals"]["test"]["sha256"] = file_hash(path)

            write(rows)
            recovered = recover_dialogues(path, manifest, "test")["a"][0]
            self.assertEqual(recovered["input_ids"], [2, 3, 12, 4, 5, 12])
            self.assertEqual(recovered["speaker_ids"], [13, 13, 13, 14, 14, 14])
            damaged = copy.deepcopy(rows)
            damaged[1]["input_ids"][0] = 11
            write(damaged)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                recover_dialogues(path, manifest, "test")
            damaged = copy.deepcopy(rows)
            damaged[1]["start_token"] = 5
            write(damaged)
            with self.assertRaisesRegex(ValueError, "Gap"):
                recover_dialogues(path, manifest, "test")
            manifest["totals"]["test"]["sha256"] = "incorrect"
            with self.assertRaisesRegex(ValueError, "checksum"):
                recover_dialogues(path, manifest, "test")

    def test_contexts_end_before_target_and_preserve_speaker_history(self):
        dialogue = {"source_id": "d", "input_ids": list(range(700)),
                    "speaker_ids": [13 if i < 400 else 14 for i in range(700)]}
        row = make_sample("a", "test", dialogue, 600, 600, {"primary"})
        self.assertEqual(row["label"], 1)
        for length in (128, 256, 512):
            self.assertEqual(row["input_ids"][-length:], list(range(600 - length, 600)))
            self.assertEqual(row["speaker_ids"][-length:], dialogue["speaker_ids"][600-length:600])
            self.assertNotIn(600, row["input_ids"][-length:])
        dialogue["input_ids"][600:] = [-1] * 100
        altered = make_sample("a", "test", dialogue, 600, 600, {"primary"})
        self.assertEqual(row["input_ids"], altered["input_ids"])
        self.assertEqual(row["sample_id"], altered["sample_id"])
        self.assertEqual(altered["label"], 0)

    def test_sampling_is_reproducible_and_does_not_depend_on_labels(self):
        ids = [12 if i % 10 == 0 else 2 for i in range(720)]
        groups = {"a": [{"source_id": "a", "input_ids": ids, "speaker_ids": [13] * len(ids)}]}
        original = build_samples(groups, "test", 12, 256, 100, 10, 1)
        self.assertEqual(original, build_samples(groups, "test", 12, 256, 100, 10, 1))
        changed = copy.deepcopy(groups)
        changed["a"][0]["input_ids"] = [2 if t == 12 else 12 for t in ids]
        modified = build_samples(changed, "test", 12, 256, 100, 10, 1)
        self.assertEqual([r["sample_id"] for r in original[0]], [r["sample_id"] for r in modified[0]])
        self.assertEqual(len(original[0]), len({r["sample_id"] for r in original[0]}))
        self.assertEqual(sum("primary" in r["subsets"] for r in original[0]), 256)
        self.assertEqual(sum("history_gt256" in r["subsets"] for r in original[0]), 100)
        self.assertTrue(all(len(r["input_ids"]) == 512 for r in original[1]))

    def test_batched_left_padding_matches_individual_predictions(self):
        torch.set_num_threads(1)
        for tied in (True, False):
            with self.subTest(tied=tied), torch.inference_mode():
                torch.manual_seed(1)
                config = Qwen3Config(vocab_size=20, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                     head_dim=8, tie_word_embeddings=tied, attention_dropout=0.0)
                config._attn_implementation = "sdpa"
                model = Qwen3ForCausalLM(config).eval()
                rows = [{"input_ids": [2, 3, 12, 4, 5], "speaker_ids": [13, 13, 13, 14, 14]},
                        {"input_ids": [6], "speaker_ids": [13]},
                        {"input_ids": [7, 8, 9], "speaker_ids": [14, 14, 14]}]
                tokens = {"ts": 12, "speaker1": 13, "speaker2": 14}
                for length in (2, 4, 8):
                    actual = batch_probabilities(model, make_batch(rows, length, 0, 13, "cpu"), 12)
                    expected = []
                    for row in rows:
                        ids = torch.tensor([row["input_ids"][-length:]])
                        states = torch.tensor([row["speaker_ids"][-length:]])
                        expected.append(next_ts_probability(model, ids, states, torch.ones_like(ids), 12).item())
                    torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=1e-7)
                    ordered = score_rows(model, rows, length, 2, 0, tokens, "cpu")
                    torch.testing.assert_close(torch.tensor(ordered), torch.tensor(expected), rtol=1e-5, atol=1e-7)

    def test_test_labels_and_long_subset_cannot_change_validation_threshold(self):
        rows = {split: [{"dataset": "a", "label": label, "subsets": ["primary", "history_gt256"]}
                        for label in (0, 1, 0, 1)] for split in ("validation", "test")}
        scores = {"validation": [0.1, 0.6, 0.4, 0.8], "test": [0.2, 0.7, 0.3, 0.9]}
        report = classification_report(rows, scores)
        self.assertEqual(report["thresholds"]["a"]["threshold"], 0.6)
        self.assertEqual(report["primary"]["macro_test_bacc"], 1.0)
        for row in rows["test"]:
            row["label"] = 1 - row["label"]
        changed = classification_report(rows, scores)
        self.assertEqual(report["thresholds"], changed["thresholds"])
        self.assertEqual(changed["history_gt256"]["datasets"]["a"]["threshold"], 0.6)


if __name__ == "__main__":
    unittest.main()
