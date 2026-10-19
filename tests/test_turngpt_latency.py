"""Validate latency workload semantics and input selection without a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_qwen3_turngpt_latency import next_ts_probability, select_samples, summarize


class LatencyTests(unittest.TestCase):
    def test_gpt2_score_matches_full_logits_and_projects_only_last_position(self):
        torch.set_num_threads(1)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                torch.manual_seed(1)
                config = GPT2Config(vocab_size=20, n_embd=16, n_layer=1, n_head=2,
                                    n_positions=256, attn_pdrop=0, embd_pdrop=0, resid_pdrop=0)
                config._attn_implementation = "sdpa"
                model = GPT2LMHeadModel(config).to(dtype=dtype).eval()
                ids = torch.tensor([[2, 3, 12, 4, 5]])
                speakers = torch.tensor([[13, 13, 13, 14, 14]])
                mask = torch.ones_like(ids)
                weight = model.get_input_embeddings()
                full = model(inputs_embeds=weight(ids) + weight(speakers),
                             attention_mask=mask, use_cache=False).logits.float()
                expected = full[:, -1, :].softmax(-1)[:, 12]
                projected_shapes = []
                hook = model.lm_head.register_forward_pre_hook(
                    lambda module, args: projected_shapes.append(tuple(args[0].shape)))
                try:
                    actual = next_ts_probability(model, ids, speakers, mask, 12)
                finally:
                    hook.remove()
                self.assertEqual(projected_shapes, [(1, 1, 16)])
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_last_position_score_matches_full_logits_for_both_head_types(self):
        for tied in (True, False):
            with self.subTest(tied=tied), torch.inference_mode():
                torch.manual_seed(1)
                config = Qwen3Config(vocab_size=20, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=8,
                                     tie_word_embeddings=tied, attention_dropout=0.0)
                config._attn_implementation = "sdpa"
                model = Qwen3ForCausalLM(config).eval()
                ids = torch.tensor([[2, 3, 12, 4, 5]])
                speakers = torch.tensor([[13, 13, 13, 14, 14]])
                mask = torch.ones_like(ids)
                weight = model.get_input_embeddings()
                full = model(inputs_embeds=weight(ids) + weight(speakers),
                             attention_mask=mask, use_cache=False).logits.float()
                expected = full[:, -1, :].softmax(-1)[:, 12]
                actual = next_ts_probability(model, ids, speakers, mask, 12)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_fixed_length_selection_preserves_content_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [dict(dataset="a", source_split="test", input_ids=[2, 3, 4, 12],
                         speaker_ids=[13, 13, 14, 14]),
                    dict(dataset="a", source_split="test", input_ids=[2, 12], speaker_ids=[13, 13]),
                    dict(dataset="b", source_split="test", input_ids=[5, 6, 7, 12],
                         speaker_ids=[13, 14, 14, 14])]
            manifest = {"splits": {"test": {"a": {"windows": 2}, "b": {"windows": 1}}},
                        "special_token_ids": {"ts": 12, "speaker1": 13, "speaker2": 14}}
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            samples, info = select_samples(path, 4, 32, 1)
            self.assertEqual((samples, info), select_samples(path, 4, 32, 1))
            self.assertEqual(info["sample_count"], 2)
            self.assertEqual(info["sampled_windows_by_corpus"], {"a": 1, "b": 1})
            for sample in samples:
                original = rows[sample["jsonl_line"] - 1]
                self.assertEqual(sample["input_ids"], original["input_ids"])
                self.assertEqual(sample["speaker_ids"], original["speaker_ids"])
            rows[0]["input_ids"][0] = 8
            (path / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertNotEqual(info["input_samples_sha256"],
                                select_samples(path, 4, 32, 1)[1]["input_samples_sha256"])
            with self.assertRaisesRegex(ValueError, "No real test windows"):
                select_samples(path, 5, 32, 1)
            manifest["splits"]["test"]["a"]["windows"] = 99
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "counts"):
                select_samples(path, 4, 32, 1)

    def test_summary_interpolates_percentiles_and_rejects_invalid_timings(self):
        result = summarize([40, 10, 30, 20])
        self.assertEqual(result, {"requests": 4, "mean_ms": 25, "p50_ms": 25, "p95_ms": 38.5})
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                summarize(values)


if __name__ == "__main__":
    unittest.main()
