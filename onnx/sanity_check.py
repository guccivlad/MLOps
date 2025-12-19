import numpy as np
import tritonclient.http as httpclient

if __name__ == "__main__":
    text = "Какой-то текст. Который нужен. Для проверки вызова тритона."

    client = httpclient.InferenceServerClient(url="localhost:8000")
    meta = client.get_model_metadata("rubert_ensemble")
    output_names = [o["name"] for o in meta["outputs"]]

    text_bytes = np.array([[text.encode("utf-8")]], dtype=object)

    input = httpclient.InferInput("TEXT", text_bytes.shape, "BYTES")
    input.set_data_from_numpy(text_bytes)

    res = client.infer(model_name="rubert_ensemble", inputs=[input], outputs=None)

    for name in output_names:
        out = res.as_numpy(name)
        print(f"{name}:")
        print(out.reshape(-1))
