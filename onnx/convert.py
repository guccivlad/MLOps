import os
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.export import Dim
from transformers import AutoModel, AutoTokenizer


class BertWithProj(nn.Module):
    def __init__(
        self, model_name: str = "ai-forever/ruBert-base", output_dim: int = 32
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            return_dict=True,
        )

        hidden = self.backbone.config.hidden_size
        self.proj = torch.nn.Linear(hidden, output_dim)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids, attention_mask)
        proj = self.proj(out.pooler_output)

        return proj


def encode(tokenizer: Any, text: str, max_len: int):
    encoded = tokenizer.encode_plus(
        text, padding="max_length", max_length=max_len, truncation=True
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.int32, device="cpu")
    attention_mask = torch.tensor(
        encoded["attention_mask"], dtype=torch.int64, device="cpu"
    )
    return input_ids, attention_mask


if __name__ == "__main__":
    model_name = "ai-forever/ruBert-base"

    model = BertWithProj()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    texts = ["Привет", "Дз по млопс"]
    tokens_tensor, att_mask_tensor = [], []
    for text in texts:
        _tokens, _mask = encode(tokenizer, text, 16)
        tokens_tensor.append(_tokens)
        att_mask_tensor.append(_mask)

    tokens_tensor = torch.stack(tokens_tensor)
    att_mask_tensor = torch.stack(att_mask_tensor)

    batch = Dim("BATCH_SIZE", min=2)
    dynamic_shapes = {
        "input_ids": {0: batch},
        "attention_mask": {0: batch},
    }

    os.makedirs("model_repository/rubert_onnx/1", exist_ok=True)
    onnx_path = "model_repository/rubert_onnx/1/ruBert_base_embedder.onnx"
    torch.onnx.export(
        model,
        (tokens_tensor, att_mask_tensor),
        onnx_path,
        opset_version=19,
        input_names=["INPUT_IDS", "ATTENTION_MASK"],
        output_names=["EMBEDDINGS"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )

    model_out = model(tokens_tensor, att_mask_tensor).detach().numpy()

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        "INPUT_IDS": tokens_tensor.numpy(),
        "ATTENTION_MASK": att_mask_tensor.numpy(),
    }
    onnx_out = ort_session.run(["EMBEDDINGS"], ort_inputs)[0]

    diff = model_out - onnx_out
    max_abs = float(np.max(np.abs(diff)))
    min_abs = float(np.min(np.abs(diff)))

    print("max_diff =", max_abs)
    print("min_diff =", min_abs)
