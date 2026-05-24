#!/usr/bin/env python3
"""Run LLM-as-reranker strategies on ruMTEB reranking tasks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import mteb

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.llm_reranker.benchmark import (
    ALLOWED_TASKS,
    OVERWRITE_STRATEGIES,
    PROFILE_DEFAULTS,
    build_summary,
    load_tasks,
    positive_float,
    positive_int,
    print_summary,
    resolve_output_dir,
    subset_task_queries,
)
from project.llm_reranker.cache import RequestCache
from project.llm_reranker.logging_utils import add_logging_arguments, configure_logging
from project.llm_reranker.providers import OllamaProvider, OpenRouterProvider, VLLMProvider
from project.llm_reranker.search_adapter import LLMRerankerSearchModel
from project.llm_reranker.strategies import (
    ListwiseRankGPTRerankStrategy,
    PairwisePRPRerankStrategy,
    PointwiseGradedRerankStrategy,
)


logger = logging.getLogger(__name__)

PROVIDERS = ("ollama", "openrouter", "vllm")
STRATEGIES = ("pointwise-graded", "pairwise-prp", "listwise-rankgpt")
PROMPT_LANGUAGES = ("ru", "en")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-reranker strategies on ruMTEB reranking tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", required=True, choices=PROVIDERS)
    parser.add_argument("--model", required=True, help="Provider model identifier")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALLOWED_TASKS,
        default=list(ALLOWED_TASKS),
        help="Subset of ruMTEB reranking tasks to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cache, predictions, and summaries",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="full",
        help="Benchmark profile",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=positive_int,
        default=None,
        help="Number of top-ranked candidates to rerank with the LLM",
    )
    parser.add_argument(
        "--max-queries-per-task",
        type=positive_int,
        default=None,
        help="Optional query cap per task",
    )
    parser.add_argument(
        "--overwrite-strategy",
        choices=OVERWRITE_STRATEGIES,
        default="only-missing",
        help="MTEB cache overwrite strategy",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the provider",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=120.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=4,
        help="Number of query-level reranking workers",
    )
    parser.add_argument(
        "--max-retries",
        type=positive_int,
        default=3,
        help="Number of provider retries for malformed/transient responses",
    )
    parser.add_argument(
        "--prompt-language",
        choices=PROMPT_LANGUAGES,
        default="ru",
        help="Language used in reranking prompts",
    )
    parser.add_argument(
        "--query-max-chars",
        type=positive_int,
        default=500,
        help="Maximum query length included in prompts",
    )
    parser.add_argument(
        "--doc-max-chars",
        type=positive_int,
        default=None,
        help="Override strategy-specific document truncation limit",
    )
    parser.add_argument(
        "--window-size",
        type=positive_int,
        default=10,
        help="Sliding window size for listwise reranking",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=5,
        help="Sliding window stride for listwise reranking",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL. Defaults to OLLAMA_HOST, OpenRouter API, or local vLLM API.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter/vLLM API key. Defaults to OPENROUTER_API_KEY or VLLM_API_KEY.",
    )
    parser.add_argument(
        "--keep-alive",
        default=None,
        help="Optional Ollama keep_alive value",
    )
    parser.add_argument(
        "--site-url",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter requests",
    )
    parser.add_argument(
        "--app-name",
        default="ruMTEB LLM Reranker",
        help="Optional X-Title header for OpenRouter requests",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip provider connectivity validation before evaluation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable MTEB progress bars",
    )
    add_logging_arguments(parser)
    return parser.parse_args(argv)


def resolve_profile_args(args: argparse.Namespace) -> tuple[int, int | None]:
    defaults = PROFILE_DEFAULTS[args.profile]
    rerank_top_k = args.rerank_top_k or defaults["rerank_top_k"]
    if args.max_queries_per_task is not None:
        max_queries_per_task = args.max_queries_per_task
    else:
        max_queries_per_task = defaults["max_queries_per_task"]
    logger.info(
        "Resolved profile limits: profile=%s, rerank_top_k=%s, max_queries_per_task=%s",
        args.profile,
        rerank_top_k,
        max_queries_per_task,
    )
    return rerank_top_k, max_queries_per_task


def create_provider(args: argparse.Namespace, request_cache: RequestCache):
    if args.provider == "ollama":
        base_url = args.base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        logger.info(
            "Creating Ollama provider: model=%s, base_url=%s, timeout=%s, "
            "temperature=%s, max_retries=%s, keep_alive=%s",
            args.model,
            base_url,
            args.timeout,
            args.temperature,
            args.max_retries,
            args.keep_alive,
        )
        return OllamaProvider(
            model=args.model,
            request_cache=request_cache,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            base_url=base_url,
            keep_alive=args.keep_alive,
        )

    if args.provider == "vllm":
        base_url = args.base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = args.api_key or os.environ.get("VLLM_API_KEY")
        logger.info(
            "Creating vLLM provider: model=%s, base_url=%s, timeout=%s, "
            "temperature=%s, max_retries=%s, api_key_present=%s",
            args.model,
            base_url,
            args.timeout,
            args.temperature,
            args.max_retries,
            api_key is not None,
        )
        return VLLMProvider(
            model=args.model,
            request_cache=request_cache,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            base_url=base_url,
            api_key=api_key,
        )

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OpenRouter API key is missing")
        raise ValueError("OpenRouter requires --api-key or OPENROUTER_API_KEY")
    base_url = args.base_url or "https://openrouter.ai/api/v1"
    logger.info(
        "Creating OpenRouter provider: model=%s, base_url=%s, timeout=%s, "
        "temperature=%s, max_retries=%s, site_url=%s, app_name=%s",
        args.model,
        base_url,
        args.timeout,
        args.temperature,
        args.max_retries,
        args.site_url,
        args.app_name,
    )
    return OpenRouterProvider(
        model=args.model,
        request_cache=request_cache,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_url=base_url,
        api_key=api_key,
        site_url=args.site_url,
        app_name=args.app_name,
    )


def strategy_doc_max_chars(strategy_name: str, override: int | None) -> int:
    if override is not None:
        logger.debug("Using explicit doc_max_chars=%s for strategy=%s", override, strategy_name)
        return override
    defaults = {
        "pointwise-graded": 1400,
        "pairwise-prp": 900,
        "listwise-rankgpt": 500,
    }
    resolved = defaults[strategy_name]
    logger.debug("Using default doc_max_chars=%s for strategy=%s", resolved, strategy_name)
    return resolved


def create_strategy(args: argparse.Namespace, provider):
    doc_max_chars = strategy_doc_max_chars(args.strategy, args.doc_max_chars)
    common_kwargs = {
        "provider": provider,
        "prompt_language": args.prompt_language,
        "query_max_chars": args.query_max_chars,
        "doc_max_chars": doc_max_chars,
    }
    logger.info(
        "Creating rerank strategy: strategy=%s, prompt_language=%s, "
        "query_max_chars=%s, doc_max_chars=%s",
        args.strategy,
        args.prompt_language,
        args.query_max_chars,
        doc_max_chars,
    )
    if args.strategy == "pointwise-graded":
        return PointwiseGradedRerankStrategy(**common_kwargs)
    if args.strategy == "pairwise-prp":
        return PairwisePRPRerankStrategy(**common_kwargs)
    logger.info(
        "Listwise window settings: window_size=%s, stride=%s",
        args.window_size,
        args.stride,
    )
    return ListwiseRankGPTRerankStrategy(
        **common_kwargs,
        window_size=args.window_size,
        stride=args.stride,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)
    logger.info(
        "Starting LLM ruMTEB run: provider=%s, model=%s, strategy=%s, tasks=%s, "
        "profile=%s, overwrite_strategy=%s, concurrency=%s",
        args.provider,
        args.model,
        args.strategy,
        args.tasks,
        args.profile,
        args.overwrite_strategy,
        args.concurrency,
    )
    rerank_top_k, max_queries_per_task = resolve_profile_args(args)

    output_dir = resolve_output_dir(
        provider=args.provider,
        model_name_or_path=args.model,
        strategy=args.strategy,
        output_dir=args.output_dir,
    )
    cache_dir = output_dir / "cache"
    prediction_dir = output_dir / "predictions"
    request_cache_dir = output_dir / "request_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    request_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Prepared artifact directories: output=%s, cache=%s, predictions=%s, "
        "request_cache=%s",
        output_dir.resolve(),
        cache_dir.resolve(),
        prediction_dir.resolve(),
        request_cache_dir.resolve(),
    )

    request_cache = RequestCache(request_cache_dir)
    provider = create_provider(args, request_cache)
    if not args.skip_preflight:
        logger.info("Running provider preflight")
        provider.preflight()
        logger.info("Provider preflight completed")
    else:
        logger.info("Provider preflight skipped")
    strategy = create_strategy(args, provider)
    model = LLMRerankerSearchModel(
        provider_name=args.provider,
        provider_model_name=args.model,
        strategy=strategy,
        rerank_top_k=rerank_top_k,
        concurrency=args.concurrency,
    )

    tasks = load_tasks(args.tasks)
    subset_task_queries(tasks, max_queries_per_task)

    logger.info("Starting MTEB evaluation")
    results = mteb.evaluate(
        model,
        tasks=tasks,
        raise_error=True,
        encode_kwargs={},
        cache=mteb.ResultCache(cache_path=cache_dir),
        overwrite_strategy=args.overwrite_strategy,
        prediction_folder=prediction_dir,
        show_progress_bar=not args.quiet,
    )
    logger.info("MTEB evaluation finished")

    run_stats = {
        "provider": provider.stats(),
        "strategy": strategy.stats(),
        "search_model": model.stats(),
        "request_cache": request_cache.stats(),
    }
    run_stats_path = output_dir / "run_stats.json"
    run_stats_path.write_text(
        json.dumps(run_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote run stats: %s", run_stats_path.resolve())
    logger.info("Run stats snapshot: %s", json.dumps(run_stats, ensure_ascii=False))

    summary = build_summary(
        provider=args.provider,
        model_name_or_path=args.model,
        strategy=args.strategy,
        profile=args.profile,
        rerank_top_k=rerank_top_k,
        max_queries_per_task=max_queries_per_task,
        temperature=args.temperature,
        timeout=args.timeout,
        concurrency=args.concurrency,
        prompt_language=args.prompt_language,
        tasks=tasks,
        output_dir=output_dir,
        results=results,
        run_stats_path=run_stats_path,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote summary: %s", summary_path.resolve())
    print_summary(summary)
    logger.info("LLM ruMTEB run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
