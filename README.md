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
