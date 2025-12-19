import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        tok_dir = os.path.join(Path(__file__).resolve().parent, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

    def execute(self, requests):
        responses = []

        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "TEXT")
            arr = inp.as_numpy()

            flat = arr.reshape(-1)
            texts = []
            for x in flat:
                if isinstance(x, (bytes, bytearray)):
                    texts.append(x.decode("utf-8"))
                else:
                    texts.append(str(x))

            enc = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=16,
                return_tensors="np",
            )

            input_ids = enc["input_ids"].astype(np.int32)
            attention_mask = enc["attention_mask"].astype(np.int64)

            out_ids = pb_utils.Tensor("INPUT_IDS", input_ids)
            out_mask = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_ids, out_mask])
            )

        return responses
