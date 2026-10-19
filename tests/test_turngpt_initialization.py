"""CPU regression checks for tied Qwen3 weights and the separate 8B output head."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_qwen3_turngpt_subset import initialize_turngpt_embeddings, forward, make_collator


class TinyTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 15

    def encode(self, mark, add_special_tokens=False):
        return [{"!": 1, "?": 2, ".": 3}[mark]]


TOKEN_IDS = {"ts": 12, "speaker1": 13, "speaker2": 14}


def tiny_model(tied):
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, tie_word_embeddings=tied, attention_dropout=0.0,
        pad_token_id=0, eos_token_id=0,
    ))


def legacy_initialization(model, tokenizer):
    """The tied-weight initialization used in the completed 0.6B/1.7B/4B runs."""
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        weight[12].copy_(weight[[1, 2, 3]].mean(0))
        weight[13].normal_(mean=0.0, std=model.config.initializer_range)
        weight[14].normal_(mean=0.0, std=model.config.initializer_range)


class InitializationTests(unittest.TestCase):
    def test_tied_weights_preserve_legacy_parameters_and_rng(self):
        torch.manual_seed(10)
        model = tiny_model(tied=True)
        reference = copy.deepcopy(model)
        torch.manual_seed(1)
        legacy_initialization(reference, TinyTokenizer())
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(1)
        initialize_turngpt_embeddings(model, TinyTokenizer(), TOKEN_IDS)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, reference.state_dict()[name]), name)
        self.assertEqual(model.get_input_embeddings().weight.data_ptr(),
                         model.get_output_embeddings().weight.data_ptr())

    def test_untied_output_initialization_and_training(self):
        torch.manual_seed(10)
        model = tiny_model(tied=False)
        with torch.no_grad():
            model.get_input_embeddings().weight[[1, 2, 3]] = 0.25
            model.get_output_embeddings().weight[[1, 2, 3]] = 0.75
            model.get_input_embeddings().weight[12:15] = 99
            model.get_output_embeddings().weight[12:15] = -99
        original_input = model.get_input_embeddings().weight[:12].detach().clone()
        original_output = model.get_output_embeddings().weight[:12].detach().clone()
        initialize_turngpt_embeddings(model, TinyTokenizer(), TOKEN_IDS)
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        self.assertNotEqual(input_weight.data_ptr(), output_weight.data_ptr())
        self.assertEqual(tuple(input_weight.shape), (15, 16))
        self.assertEqual(tuple(output_weight.shape), (15, 16))
        self.assertTrue(torch.equal(input_weight[:12], original_input))
        self.assertTrue(torch.equal(output_weight[:12], original_output))
        self.assertTrue(torch.equal(input_weight[12], original_input[[1, 2, 3]].mean(0)))
        self.assertTrue(torch.equal(output_weight[12], original_output[[1, 2, 3]].mean(0)))
        for weight in (input_weight, output_weight):
            self.assertTrue(torch.isfinite(weight[13:15]).all())
            self.assertGreater(weight[13:15].std().item(), 0)
            self.assertLess(weight[13:15].abs().max().item(), 1)
        self.assertEqual(model.config.eos_token_id, 12)
        self.assertEqual(model.generation_config.eos_token_id, 12)
        self.assertFalse(model.config.use_cache)

        # Include padding and both speaker states in a real Qwen3 backward pass.
        batch = make_collator(0, 13)([([4, 5, 12, 6, 12], [13, 13, 13, 14, 14]),
                                     ([7, 12], [13, 13])])
        loss = forward(model, batch, "cpu").loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for weight in (input_weight, output_weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
