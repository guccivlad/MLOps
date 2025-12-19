import numpy as np


def gemm_intensity(tokens: int, out_features: int, in_features: int):
    flops = 2 * tokens * out_features * in_features
    bytes_ = 4 * (
        tokens * in_features + in_features * out_features + tokens * out_features
    )

    return flops / bytes_


def layernorm_intensity(numel: int):
    flops = 10 * numel
    bytes_ = 8 * numel

    return flops / bytes_


def table_intensity(
    batch_list: list[int], tokens: int, hidden: int = 768, heads: int = 12
):
    d = hidden // heads
    rows = []
    for batch in batch_list:
        all_tokens = batch * tokens
        lin_intensity = gemm_intensity(all_tokens, hidden, hidden)
        ffn_intensity = gemm_intensity(all_tokens, 4 * hidden, hidden)

        Mq = batch * heads * tokens
        attn_intensity = gemm_intensity(Mq, tokens, d)

        ln_intensity = layernorm_intensity(batch * tokens * hidden)

        rows.append((batch, lin_intensity, ffn_intensity, attn_intensity, ln_intensity))

    return rows


if __name__ == "__main__":
    batch_list = [4, 32, 128, 512]

    print("Tokens in batch=16")
    for r in table_intensity(batch_list, tokens=16):
        print(f"Intensity for batch={r[0]}: {r[1:]}")
