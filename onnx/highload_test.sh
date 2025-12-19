TRITON_CONTAINER=triton-rubert

docker run --rm -it \
  --network container:$TRITON_CONTAINER \
  nvcr.io/nvidia/tritonserver:25.08-py3-sdk \
  perf_analyzer -m rubert_ensemble -i http -u localhost:8000 \
    -b 1 --concurrency-range 1:4 --measurement-interval 20000
