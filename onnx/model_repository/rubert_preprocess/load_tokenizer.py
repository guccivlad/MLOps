from transformers import AutoTokenizer

AutoTokenizer.from_pretrained("ai-forever/ruBert-base").save_pretrained(
    "model_repository/preprocess/tokenizer"
)
