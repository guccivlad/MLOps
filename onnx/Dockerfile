FROM nvcr.io/nvidia/tritonserver:25.08-py3

RUN pip3 install --no-cache-dir transformers sentencepiece numpy

COPY model_repository /model_repository

CMD ["tritonserver", "--model-repository=/model_repository"]
