# MLOps

Реализован метод Гаусса-Жордана для поиска обратной матрицы на C++(`src/main.cpp`)

Тестовая функция написана на питоне, в которой сравниваетмся реализация с `numpy`(`python/gauss.py`)

## Запуск

```
docker build -t gauss .
docker run --rm gauss
```