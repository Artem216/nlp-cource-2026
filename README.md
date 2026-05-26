# NLP

Репозиторий для домашних заданий по курсу NLP.

## Сборка окружения

```
uv sync
```

## LLM reranker через vLLM

По умолчанию `--provider vllm` ходит в локальный OpenAI-compatible сервер
`http://127.0.0.1:8000/v1`. URL можно переопределить через `--base-url` или
`VLLM_BASE_URL`, ключ API - через `--api-key` или `VLLM_API_KEY`.

```
uv run python project/run_rumteb_llm_reranker.py \
  --provider vllm \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --strategy pointwise-graded \
  --profile quick
```

Для `listwise-rankgpt` через удалённый vLLM лучше ограничивать размер ответа и
начинать с одного параллельного запроса:

```
uv run python project/run_rumteb_llm_reranker.py \
  --provider vllm \
  --base-url https://api.ai.gnivc.ru/v1 \
  --model <served-model> \
  --strategy listwise-rankgpt \
  --profile quick \
  --concurrency 1 \
  --max-tokens 256 \
  --timeout 300
```

Самый устойчивый вариант для долгого прогона - простой последовательный
`pointwise/doc` runner. Он делает один короткий LLM-запрос на один документ,
пишет подробный прогресс в `run.log` и `events.jsonl`, а при окончательном сбое
одного документа оставляет исходный порядок документов для этого запроса:

```
uv run python project/simple_llm_reranker/run.py \
  --base-url https://api.ai.gnivc.ru/v1 \
  --model <served-model> \
  --max-queries-per-task 25 \
  --rerank-top-k 10
```

## rusBEIR для LLM reranker

rusBEIR добавлен отдельным runner'ом, потому что это BEIR/Hugging Face
бенчмарк, а не MTEB-задача. Для честного полного прогона нужен first-stage
run-файл с кандидатами в TREC/TSV/JSON/JSONL формате. По умолчанию
`--candidate-source auto`: если указан `--first-stage-run`, runner использует
его, иначе построит локальные TF-IDF кандидаты.

```
uv run python project/run_rusbeir_llm_reranker.py \
  --provider vllm \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --strategy pointwise-graded \
  --tasks rus-scifact \
  --first-stage-run path/to/rus-scifact.run
```

Если передать директорию в `--first-stage-run`, runner будет искать файлы вида
`<task>.run`, например `rus-scifact.run`. Для полного набора датасетов используйте
`--tasks all`.

Для быстрого smoke-прогона без отдельного ретривера есть локальный TF-IDF источник
кандидатов:

```
uv run python project/run_rusbeir_llm_reranker.py \
  --provider vllm \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --strategy listwise-rankgpt \
  --tasks rus-scifact \
  --profile quick \
  --max-corpus-docs 5000 \
  --skip-preflight
```

Кросс-энкодеры запускаются на том же rusBEIR слое, но отдельным runner'ом:

```
uv run python project/run_rusbeir_reranker.py \
  BAAI/bge-reranker-v2-m3 \
  --tasks rus-scifact \
  --first-stage-run path/to/rus-scifact.run \
  --device cuda \
  --batch-size 32
```

Для быстрого локального прогона без готового run-файла можно так же включить
TF-IDF кандидатов:

```
uv run python project/run_rusbeir_reranker.py \
  BAAI/bge-reranker-v2-m3 \
  --tasks rus-scifact \
  --profile quick \
  --max-corpus-docs 5000 \
  --device cpu
```
