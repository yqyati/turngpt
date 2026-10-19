"""Shared task tokens and dtype conversion for training and preprocessing."""
import torch

SPECIAL_TOKENS = {
    "eos_token": "<ts>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}
SPEAKER1 = "<speaker1>"
SPEAKER2 = "<speaker2>"


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[name]
