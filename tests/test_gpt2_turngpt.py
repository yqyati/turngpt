"""GPT-2 compatibility and tokenizer-independent cache merging."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from eval_qwen3_turngpt_paper_metrics import collect
from merge_turngpt_processed import validate_sources
from train_qwen3_turngpt_subset import forward, initialize_turngpt_embeddings, make_collator

ROOT = Path(__file__).resolve().parents[1]


class ToyTokenizer:
    pad_token_id = 0
    def __len__(self):
        return 23
    def encode(self, mark, add_special_tokens=False):
        return [{"!": 1, "?": 2, ".": 3}[mark]]


class GPT2CompatibilityTests(unittest.TestCase):
    def test_speaker_input_matches_native_gpt2_and_gradients_are_finite(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        config = GPT2Config(vocab_size=20, n_embd=16, n_layer=1, n_head=2, n_positions=256,
                            attn_pdrop=0, embd_pdrop=0, resid_pdrop=0)
        model = GPT2LMHeadModel(config)
        # Nonzero centroid exposes the GPT-2 vocabulary-expansion issue.
        with torch.no_grad():
            model.get_input_embeddings().weight.add_(0.3)
        original = model.get_input_embeddings().weight.detach().clone()
        initialize_turngpt_embeddings(model, ToyTokenizer(), {"ts": 20, "speaker1": 21, "speaker2": 22})
        weight = model.get_input_embeddings().weight
        torch.testing.assert_close(weight[:20], original, rtol=0, atol=0)
        torch.testing.assert_close(weight[20], original[[1, 2, 3]].mean(0), rtol=0, atol=0)
        self.assertLess((weight[21] - original.mean(0)).abs().mean().item(), 0.04)
        self.assertEqual(weight.data_ptr(), model.get_output_embeddings().weight.data_ptr())
        self.assertEqual(model.config.turngpt_speaker_initialization, "pretrained_vocabulary_mean_plus_normal_noise")
        rows = [([2, 3, 20, 4, 20], [21, 21, 21, 22, 22]), ([5, 20, 6], [21, 21, 22])]
        batch = make_collator(0, 21)(rows)
        model.eval()
        native = model(input_ids=batch[0], token_type_ids=batch[1], attention_mask=batch[2],
                       labels=batch[3], use_cache=False)
        adapted = forward(model, batch, "cpu")
        # GPT-2 adds position embeddings before token_type embeddings; the
        # shared path adds speaker embeddings first. Allow FP32 sum-order error.
        torch.testing.assert_close(adapted.logits, native.logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(adapted.loss, native.loss)
        scores, labels = collect(model, rows, make_collator(0, 21), 2, "cpu", 20)
        self.assertEqual(labels, [0, 1, 0, 1, 1, 0])
        expected = native.logits[:, :-1].float().softmax(-1)[..., 20][batch[2][:, 1:].bool()]
        torch.testing.assert_close(torch.tensor(scores), expected, rtol=1e-4, atol=1e-6)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6.25e-5)
        forward(model, batch, "cpu").loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = GPT2LMHeadModel.from_pretrained(directory, local_files_only=True).eval()
            model.eval()
            torch.testing.assert_close(forward(loaded, batch, "cpu").logits, forward(model, batch, "cpu").logits)

    def test_merge_reads_special_ids_and_preserves_corpus_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("daily_dialog", "personachat"):
                source = root / name
                source.mkdir()
                manifest = {"settings": {"max_length": 256, "keep_length": 64, "overlap_length": 10},
                            "special_token_ids": {"ts": 20, "speaker1": 21, "speaker2": 22}}
                (source / "manifest.json").write_text(json.dumps(manifest))
                for split in ("train", "validation", "test"):
                    row = {"dataset": name, "source_split": split, "source_id": "0", "start_token": 0,
                           "input_ids": [2] * 63 + [20], "speaker_ids": [21] * 64}
                    (source / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            mapping = root / "sources.json"
            mapping.write_text(json.dumps({name: name for name in ("daily_dialog", "personachat")}))
            subprocess.run([sys.executable, str(ROOT / "merge_turngpt_processed.py"), "--source-map", str(mapping),
                            "--include", "daily_dialog", "personachat", "--output-dir", str(root / "merged")],
                           check=True, capture_output=True, text=True)
            merged = json.loads((root / "merged/manifest.json").read_text())
            self.assertEqual(merged["special_token_ids"]["ts"], 20)
            self.assertEqual(merged["totals"]["test"]["ts_tokens"], 2)
            rows = [json.loads(line) for line in (root / "merged/test.jsonl").read_text().splitlines()]
            self.assertEqual([r["dataset"] for r in rows], ["daily_dialog", "personachat"])
            manifest["special_token_ids"] = {"ts": 30, "speaker1": 31, "speaker2": 32}
            (root / "personachat/manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different tokenizers"):
                validate_sources({"daily_dialog": root / "daily_dialog", "personachat": root / "personachat"})


if __name__ == "__main__":
    unittest.main()
