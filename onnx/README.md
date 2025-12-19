Перед запуском тритон сервера нужно выполнить скрипт для конвертации - `convert.py`

## Анализ модели (`ai-forever/ruBert-base`)

Для подсчета FLOPS использовался скрипт `flops.py`. Результаты:

```
Intensity for batch=4:
 - linear: 27.428571428571427
 - fnn: 28.9811320754717
 - attention: 6.295081967213115
 - layer norm: 1.25

Intensity for batch=32:
 - linear: 109.71428571428571
 - fnn: 139.63636363636363
 - attention: 6.386694386694387
 - layer norm: 1.25

Intensity for batch=128:
 - linear: 161.68421052631578
 - fnn: 236.30769230769232
 - attention: 6.396668401874024
 - layer norm: 1.25

Intensity for batch=512:
 - linear: 183.40298507462686
 - fnn: 285.7674418604651
 - attention: 6.399166775159484
 - layer norm: 1.25
```

Характеристики `NVIDIA A10`:
```
Peak FP32: 31.2 TFLOPS
Пропускная способность памяти: 600 GB/s
```
Порог `52 FLOP/byte`. Тогда

```
Attention ~ 6.3 => memory-bound
LayerNorm ~1.25 => memory-bound
Linear:
 - batch=4: 27.4 => memory-bound
 - batch=32: 109.7 => compute-bound
 - batch=128: 161.7 => compute-bound
 - batch=512: 183.4 => compute-bound
FFN:
 - batch=4: 29.0 => memory-bound
 - batch=32: 139.6 => compute-bound
 - batch=128: 236.3 => compute-bound
 - batch=512: 285.8 => compute-bound
```


## Тест тритона под нагрузкой

Для запуска контейнера с тритон сервером нужно выполнить команды:

```
docker build -t rubert-triton .
docker run --rm --name triton-rubert -p 8000:8000 rubert-triton:latest
```

Для запуска `perf-analyzer` нужно воспользоваться скриптом `highload_test.sh`

Результаты тестирования:

```
Concurrency: 1, throughput: 38.9739 infer/sec, latency 25641 usec
Concurrency: 2, throughput: 57.3975 infer/sec, latency 34824 usec
Concurrency: 3, throughput: 59.13 infer/sec, latency 50708 usec
Concurrency: 4, throughput: 59.9507 infer/sec, latency 66696 usec
```