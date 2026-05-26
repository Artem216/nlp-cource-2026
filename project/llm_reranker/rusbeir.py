from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import make_descending_scores, normalize_whitespace


logger = logging.getLogger(__name__)

RUSBEIR_BENCHMARK_NAME = "rusBEIR"
RUSBEIR_BENCHMARK_VERSION = "hf-2025"
RUSBEIR_K_VALUES = (1, 3, 5, 10, 100, 1000)


@dataclass(frozen=True)
class RusBeirDatasetSpec:
    name: str
    corpus_repo: str
    qrels_repo: str
    default_split: str
    display_name: str
    corpus_config: str = "corpus"
    queries_config: str = "queries"
    corpus_split: str = "train"


@dataclass(frozen=True)
class RusBeirTaskData:
    spec: RusBeirDatasetSpec
    split: str
    corpus: Dataset
    queries: Dataset
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True)
class RunEntry:
    query_id: str
    doc_id: str
    score: float | None
    rank: int | None
    order: int


RUSBEIR_DATASETS: dict[str, RusBeirDatasetSpec] = {
    "rus-nfcorpus": RusBeirDatasetSpec(
        name="rus-nfcorpus",
        corpus_repo="kaengreg/rus-nfcorpus",
        qrels_repo="kaengreg/rus-nfcorpus-qrels",
        default_split="test",
        display_name="rus-NFCorpus",
    ),
    "rus-arguana": RusBeirDatasetSpec(
        name="rus-arguana",
        corpus_repo="kaengreg/rus-arguana",
        qrels_repo="kaengreg/rus-arguana-qrels",
        default_split="test",
        display_name="rus-ArguAna",
    ),
    "rus-scifact": RusBeirDatasetSpec(
        name="rus-scifact",
        corpus_repo="kaengreg/rus-scifact",
        qrels_repo="kaengreg/rus-scifact-qrels",
        default_split="test",
        display_name="rus-SciFact",
    ),
    "rus-scidocs": RusBeirDatasetSpec(
        name="rus-scidocs",
        corpus_repo="kaengreg/rus-scidocs",
        qrels_repo="kaengreg/rus-scidocs-qrels",
        default_split="test",
        display_name="rus-SCiDOCS",
    ),
    "rus-trec-covid": RusBeirDatasetSpec(
        name="rus-trec-covid",
        corpus_repo="kaengreg/rus-trec-covid",
        qrels_repo="kaengreg/rus-trec-covid-qrels",
        default_split="test",
        display_name="rus-TREC-COVID",
    ),
    "rus-fiqa": RusBeirDatasetSpec(
        name="rus-fiqa",
        corpus_repo="kaengreg/rus-fiqa",
        qrels_repo="kaengreg/rus-fiqa-qrels",
        default_split="test",
        display_name="rus-FiQA",
    ),
    "rus-quora": RusBeirDatasetSpec(
        name="rus-quora",
        corpus_repo="kaengreg/rus-quora",
        qrels_repo="kaengreg/rus-quora-qrels",
        default_split="test",
        display_name="rus-Quora",
    ),
    "rus-cqadupstack": RusBeirDatasetSpec(
        name="rus-cqadupstack",
        corpus_repo="kaengreg/rus-cqadupstack",
        qrels_repo="kaengreg/rus-cqadupstack-qrels",
        default_split="test",
        display_name="rus-CQADupstack",
    ),
    "rus-touche": RusBeirDatasetSpec(
        name="rus-touche",
        corpus_repo="kaengreg/rus-touche",
        qrels_repo="kaengreg/rus-touche-qrels",
        default_split="test",
        display_name="rus-Touche",
    ),
    "rus-mmarco": RusBeirDatasetSpec(
        name="rus-mmarco",
        corpus_repo="kaengreg/rus-mmarco-google",
        qrels_repo="kaengreg/rus-mmarco-qrels",
        default_split="dev",
        display_name="rus-MMARCO",
    ),
    "rus-miracl": RusBeirDatasetSpec(
        name="rus-miracl",
        corpus_repo="kaengreg/rus-miracl",
        qrels_repo="kaengreg/rus-miracl-qrels",
        default_split="dev",
        display_name="rus-MIRACL",
    ),
    "rus-xquad": RusBeirDatasetSpec(
        name="rus-xquad",
        corpus_repo="kaengreg/rus-xquad",
        qrels_repo="kaengreg/rus-xquad-qrels",
        default_split="dev",
        display_name="rus-XQuAD",
    ),
    "rus-xquad-sentences": RusBeirDatasetSpec(
        name="rus-xquad-sentences",
        corpus_repo="kaengreg/rus-xquad-sentences",
        qrels_repo="kaengreg/rus-xquad-sentences-qrels",
        default_split="dev",
        display_name="rus-XQuAD-sentences",
    ),
    "rus-tydiqa": RusBeirDatasetSpec(
        name="rus-tydiqa",
        corpus_repo="kaengreg/rus-tydiqa",
        qrels_repo="kaengreg/rus-tydiqa-qrels",
        default_split="dev",
        display_name="rus-TyDi QA",
    ),
    "sberquad-retrieval": RusBeirDatasetSpec(
        name="sberquad-retrieval",
        corpus_repo="kaengreg/sberquad-retrieval",
        qrels_repo="kaengreg/sberquad-retrieval-qrels",
        default_split="test",
        display_name="SberQUAD-retrieval",
    ),
    "russcibench-retrieval": RusBeirDatasetSpec(
        name="russcibench-retrieval",
        corpus_repo="kaengreg/ruSciBench-retrieval",
        qrels_repo="kaengreg/ruSciBench-retrieval-qrels",
        default_split="dev",
        display_name="rusSciBench-retrieval",
    ),
    "ru-facts": RusBeirDatasetSpec(
        name="ru-facts",
        corpus_repo="kaengreg/ru-facts",
        qrels_repo="kaengreg/ru-facts-qrels",
        default_split="dev",
        display_name="ru-facts",
    ),
    "rubq": RusBeirDatasetSpec(
        name="rubq",
        corpus_repo="kaengreg/rubq",
        qrels_repo="kaengreg/rubq-qrels",
        default_split="test",
        display_name="RuBQ",
    ),
    "ria-news": RusBeirDatasetSpec(
        name="ria-news",
        corpus_repo="kaengreg/ria-news",
        qrels_repo="kaengreg/ria-news-qrels",
        default_split="test",
        display_name="Ria-News",
    ),
    "wikifacts-articles": RusBeirDatasetSpec(
        name="wikifacts-articles",
        corpus_repo="kaengreg/wikifacts-articles_v0",
        qrels_repo="kaengreg/wikifacts-articles_v0-qrels",
        default_split="dev",
        display_name="wikifacts-articles",
    ),
    "wikifacts-para": RusBeirDatasetSpec(
        name="wikifacts-para",
        corpus_repo="kaengreg/wikifacts-para_v0",
        qrels_repo="kaengreg/wikifacts-para_v0-qrels",
        default_split="dev",
        display_name="wikifacts-para",
    ),
    "wikifacts-sents": RusBeirDatasetSpec(
        name="wikifacts-sents",
        corpus_repo="kaengreg/wikifacts-sents_v0",
        qrels_repo="kaengreg/wikifacts-sents_v0-qrels",
        default_split="dev",
        display_name="wikifacts-sents",
    ),
    "wikifacts-sliding_para2": RusBeirDatasetSpec(
        name="wikifacts-sliding_para2",
        corpus_repo="kaengreg/wikifacts-window_2_v0",
        qrels_repo="kaengreg/wikifacts-window_2_v0-qrels",
        default_split="dev",
        display_name="wikifacts-sliding_para2",
    ),
    "wikifacts-sliding_para3": RusBeirDatasetSpec(
        name="wikifacts-sliding_para3",
        corpus_repo="kaengreg/wikifacts-window_3_v0",
        qrels_repo="kaengreg/wikifacts-window_3_v0-qrels",
        default_split="dev",
        display_name="wikifacts-sliding_para3",
    ),
    "wikifacts-sliding_para4": RusBeirDatasetSpec(
        name="wikifacts-sliding_para4",
        corpus_repo="kaengreg/wikifacts-window_4_v0",
        qrels_repo="kaengreg/wikifacts-window_4_v0-qrels",
        default_split="dev",
        display_name="wikifacts-sliding_para4",
    ),
    "wikifacts-sliding_para5": RusBeirDatasetSpec(
        name="wikifacts-sliding_para5",
        corpus_repo="kaengreg/wikifacts-window_5_v0",
        qrels_repo="kaengreg/wikifacts-window_5_v0-qrels",
        default_split="dev",
        display_name="wikifacts-sliding_para5",
    ),
    "wikifacts-sliding_para6": RusBeirDatasetSpec(
        name="wikifacts-sliding_para6",
        corpus_repo="kaengreg/wikifacts-window_6_v0",
        qrels_repo="kaengreg/wikifacts-window_6_v0-qrels",
        default_split="dev",
        display_name="wikifacts-sliding_para6",
    ),
}

RUSBEIR_ALIASES = {
    "all": "all",
    "rus-scidocs": "rus-scidocs",
    "rus-tydi-qa": "rus-tydiqa",
    "rus-tydi qa": "rus-tydiqa",
    "russcibench": "russcibench-retrieval",
    "ruscibench-retrieval": "russcibench-retrieval",
    "rus-scibench-retrieval": "russcibench-retrieval",
    "ruscibench": "russcibench-retrieval",
    "rubq": "rubq",
    "rubqreranking": "rubq",
    "ria-news": "ria-news",
}
RUSBEIR_ALIASES.update({name.lower(): name for name in RUSBEIR_DATASETS})
RUSBEIR_ALIASES.update({spec.display_name.lower(): name for name, spec in RUSBEIR_DATASETS.items()})


def available_rusbeir_task_names() -> tuple[str, ...]:
    return tuple(RUSBEIR_DATASETS)


def resolve_rusbeir_dataset_specs(selected_names: Sequence[str]) -> list[RusBeirDatasetSpec]:
    if not selected_names:
        selected_names = ["rus-scifact"]
    resolved_names: list[str] = []
    for raw_name in selected_names:
        key = raw_name.strip().lower()
        if key == "all":
            for name in RUSBEIR_DATASETS:
                if name not in resolved_names:
                    resolved_names.append(name)
            continue
        canonical = RUSBEIR_ALIASES.get(key)
        if canonical is None or canonical == "all":
            allowed = ", ".join(available_rusbeir_task_names())
            raise ValueError(f"Unknown rusBEIR task {raw_name!r}. Allowed tasks: {allowed}, all")
        if canonical not in resolved_names:
            resolved_names.append(canonical)
    return [RUSBEIR_DATASETS[name] for name in resolved_names]


def split_for_spec(spec: RusBeirDatasetSpec, requested_split: str) -> str:
    if requested_split == "auto":
        return spec.default_split
    return requested_split


def load_rusbeir_queries_and_qrels(
    *,
    spec: RusBeirDatasetSpec,
    split: str,
    max_queries_per_task: int | None,
) -> tuple[Dataset, dict[str, dict[str, int]]]:
    logger.info(
        "Loading rusBEIR queries and qrels: task=%s, split=%s, query_repo=%s, qrels_repo=%s",
        spec.name,
        split,
        spec.corpus_repo,
        spec.qrels_repo,
    )
    raw_queries = _load_dataset_split_with_train_fallback(
        repo=spec.corpus_repo,
        config=spec.queries_config,
        split=split,
        role="queries",
        task_name=spec.name,
    )
    all_queries = normalize_query_dataset(raw_queries)
    raw_qrels = _load_dataset_split_with_train_fallback(
        repo=spec.qrels_repo,
        config=None,
        split=split,
        role="qrels",
        task_name=spec.name,
    )
    all_qrels = normalize_qrels(raw_qrels)
    available_query_ids = set(all_queries["id"])
    qrels = {
        query_id: doc_scores
        for query_id, doc_scores in all_qrels.items()
        if query_id in available_query_ids
    }
    if not qrels:
        query_sample = ", ".join(all_queries["id"][:5])
        qrel_sample = ", ".join(list(all_qrels)[:5])
        raise ValueError(
            "rusBEIR qrels do not overlap loaded queries: "
            f"task={spec.name!r}, split={split!r}, "
            f"query_sample=[{query_sample}], qrel_sample=[{qrel_sample}]"
        )
    queries = _select_queries_with_qrels(
        all_queries,
        qrels,
        max_queries_per_task=max_queries_per_task,
    )
    allowed_query_ids = set(queries["id"])
    qrels = {
        query_id: doc_scores
        for query_id, doc_scores in qrels.items()
        if query_id in allowed_query_ids
    }
    logger.info(
        "Loaded rusBEIR queries and qrels: task=%s, split=%s, queries=%s, qrels_queries=%s",
        spec.name,
        split,
        len(queries),
        len(qrels),
    )
    return queries, qrels


def _select_queries_with_qrels(
    queries: Dataset,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    max_queries_per_task: int | None,
) -> Dataset:
    qrel_query_ids = set(qrels)
    selected_indices: list[int] = []
    for index, query_id in enumerate(queries["id"]):
        if str(query_id) not in qrel_query_ids:
            continue
        selected_indices.append(index)
        if max_queries_per_task is not None and len(selected_indices) >= max_queries_per_task:
            break
    selected = queries.select(selected_indices)
    if max_queries_per_task is not None and len(selected) < len(qrel_query_ids):
        logger.info(
            "Limited rusBEIR queries with qrels: queries=%s, max_queries=%s",
            len(selected),
            max_queries_per_task,
        )
    return selected


def load_rusbeir_corpus(
    *,
    spec: RusBeirDatasetSpec,
    text_type: str,
    needed_doc_ids: set[str] | None = None,
    max_corpus_docs: int | None = None,
) -> Dataset:
    logger.info(
        "Loading rusBEIR corpus: task=%s, repo=%s, split=%s, needed_docs=%s, max_corpus_docs=%s",
        spec.name,
        spec.corpus_repo,
        spec.corpus_split,
        len(needed_doc_ids) if needed_doc_ids is not None else None,
        max_corpus_docs,
    )
    raw_corpus = load_dataset(spec.corpus_repo, spec.corpus_config, split=spec.corpus_split)
    if needed_doc_ids is not None:
        raw_corpus = raw_corpus.filter(lambda row: _row_id(row) in needed_doc_ids)
        logger.info(
            "Filtered rusBEIR corpus to candidate docs: task=%s, docs=%s",
            spec.name,
            len(raw_corpus),
        )
    if max_corpus_docs is not None and len(raw_corpus) > max_corpus_docs:
        raw_corpus = raw_corpus.select(range(max_corpus_docs))
        logger.info(
            "Limited rusBEIR corpus: task=%s, max_corpus_docs=%s",
            spec.name,
            max_corpus_docs,
        )
    corpus = normalize_corpus_dataset(raw_corpus, text_type=text_type)
    logger.info("Loaded rusBEIR corpus: task=%s, docs=%s", spec.name, len(corpus))
    return corpus


def load_rusbeir_task(
    *,
    spec: RusBeirDatasetSpec,
    split: str,
    text_type: str,
    max_queries_per_task: int | None = None,
    needed_doc_ids: set[str] | None = None,
    max_corpus_docs: int | None = None,
) -> RusBeirTaskData:
    queries, qrels = load_rusbeir_queries_and_qrels(
        spec=spec,
        split=split,
        max_queries_per_task=max_queries_per_task,
    )
    corpus = load_rusbeir_corpus(
        spec=spec,
        text_type=text_type,
        needed_doc_ids=needed_doc_ids,
        max_corpus_docs=max_corpus_docs,
    )
    return RusBeirTaskData(spec=spec, split=split, corpus=corpus, queries=queries, qrels=qrels)


def _load_dataset_split_with_train_fallback(
    *,
    repo: str,
    config: str | None,
    split: str,
    role: str,
    task_name: str,
) -> Dataset:
    try:
        return _load_dataset_split(repo=repo, config=config, split=split)
    except ValueError as exc:
        if split == "train" or not _is_unknown_split_error(exc):
            raise
        logger.warning(
            "rusBEIR %s split %r is unavailable for task=%s, repo=%s, config=%s; "
            "loading physical HF split 'train' instead",
            role,
            split,
            task_name,
            repo,
            config,
        )
        try:
            return _load_dataset_split(repo=repo, config=config, split="train")
        except Exception as fallback_exc:
            raise ValueError(
                f"Could not load rusBEIR {role} for task={task_name!r} from repo={repo!r}, "
                f"config={config!r}: split {split!r} is unavailable and fallback split "
                "'train' also failed"
            ) from fallback_exc


def _load_dataset_split(*, repo: str, config: str | None, split: str) -> Dataset:
    if config is None:
        return load_dataset(repo, split=split)
    return load_dataset(repo, config, split=split)


def _is_unknown_split_error(exc: ValueError) -> bool:
    message = str(exc)
    return "Unknown split" in message and "Should be one of" in message


def normalize_query_dataset(dataset: Dataset) -> Dataset:
    rows = []
    for row in dataset:
        rows.append(
            {
                "id": _row_id(row),
                "text": _first_present_text(
                    row,
                    ("text", "query", "question", "processed_text", "processed_query"),
                ),
            }
        )
    return Dataset.from_list(rows)


def normalize_corpus_dataset(dataset: Dataset, *, text_type: str) -> Dataset:
    rows = []
    for row in dataset:
        text = _first_present_text(
            row,
            (text_type, "text", "processed_text", "passage", "content", "document"),
        )
        title = _first_present_text(
            row,
            (
                "processed_title" if text_type == "processed_text" else "title",
                "title",
                "processed_title",
            ),
            default="",
        )
        rows.append({"id": _row_id(row), "title": title, "text": text})
    return Dataset.from_list(rows)


def normalize_qrels(
    dataset: Dataset,
    *,
    allowed_query_ids: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in dataset:
        query_id = _first_present_value(row, ("query-id", "query_id", "qid", "query"))
        doc_id = _first_present_value(
            row,
            ("corpus-id", "corpus_id", "doc_id", "document_id", "pid", "passage_id"),
        )
        if query_id is None or doc_id is None:
            raise ValueError(f"Could not read qrels row ids: {row!r}")
        query_id = str(query_id)
        if allowed_query_ids is not None and query_id not in allowed_query_ids:
            continue
        score_value = _first_present_value(row, ("score", "relevance", "label"))
        score = int(float(score_value if score_value is not None else 1))
        qrels.setdefault(query_id, {})[str(doc_id)] = score
    return qrels


def load_first_stage_run(path: Path, *, top_k: int | None = None) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    logger.info("Loading first-stage run: path=%s, top_k=%s", path, top_k)
    if suffix == ".json":
        return _limit_top_ranked(_parse_json_run(json.loads(path.read_text(encoding="utf-8"))), top_k)
    if suffix == ".jsonl":
        entries = []
        with path.open("r", encoding="utf-8") as handle:
            for order, line in enumerate(handle):
                if line.strip():
                    entries.extend(_record_to_entries(json.loads(line), order=order))
        return _entries_to_top_ranked(entries, top_k=top_k)
    return _limit_top_ranked(_parse_text_run(path), top_k)


def resolve_first_stage_run_path(root: Path, spec: RusBeirDatasetSpec) -> Path:
    if root.is_file():
        return root
    candidates = [
        root / f"{spec.name}{suffix}"
        for suffix in (".run", ".trec", ".txt", ".tsv", ".json", ".jsonl")
    ]
    candidates.extend(
        root / f"{spec.display_name}{suffix}"
        for suffix in (".run", ".trec", ".txt", ".tsv", ".json", ".jsonl")
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(str(candidate.name) for candidate in candidates)
    raise FileNotFoundError(f"No first-stage run file for {spec.name} in {root}. Tried: {names}")


def limit_top_ranked_to_queries(
    top_ranked: Mapping[str, Sequence[str]],
    query_ids: Iterable[str],
    *,
    top_k: int | None,
) -> dict[str, list[str]]:
    query_id_set = set(query_ids)
    limited = {
        str(query_id): _dedupe_doc_ids(doc_ids, top_k=top_k)
        for query_id, doc_ids in top_ranked.items()
        if str(query_id) in query_id_set
    }
    logger.info(
        "Limited first-stage candidates to loaded queries: queries_with_candidates=%s",
        len(limited),
    )
    return limited


def needed_prefix_doc_ids(top_ranked: Mapping[str, Sequence[str]], *, rerank_top_k: int) -> set[str]:
    doc_ids: set[str] = set()
    for ranked_doc_ids in top_ranked.values():
        doc_ids.update(str(doc_id) for doc_id in ranked_doc_ids[:rerank_top_k])
    return doc_ids


def build_tfidf_first_stage(
    *,
    corpus: Dataset,
    queries: Dataset,
    top_k: int,
    max_features: int | None = 200_000,
) -> dict[str, list[str]]:
    logger.info(
        "Building TF-IDF first-stage candidates: corpus=%s, queries=%s, top_k=%s, max_features=%s",
        len(corpus),
        len(queries),
        top_k,
        max_features,
    )
    corpus_ids = [str(row["id"]) for row in corpus]
    corpus_texts = [_joined_document_text(row) for row in corpus]
    query_texts = [str(row["text"]) for row in queries]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform(query_texts)
    score_matrix = query_matrix @ corpus_matrix.T

    top_ranked: dict[str, list[str]] = {}
    for query_index, query_row in enumerate(queries):
        score_row = score_matrix.getrow(query_index)
        if score_row.nnz:
            scored_indices = sorted(
                zip(score_row.indices.tolist(), score_row.data.tolist(), strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            selected_indices = [index for index, _ in scored_indices[:top_k]]
        else:
            selected_indices = []
        if len(selected_indices) < min(top_k, len(corpus_ids)):
            selected_set = set(selected_indices)
            for index in range(len(corpus_ids)):
                if index not in selected_set:
                    selected_indices.append(index)
                    if len(selected_indices) >= top_k:
                        break
        top_ranked[str(query_row["id"])] = [corpus_ids[index] for index in selected_indices[:top_k]]
    logger.info("Built TF-IDF first-stage candidates for %s queries", len(top_ranked))
    return top_ranked


def evaluate_retrieval(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    *,
    k_values: Sequence[int] = RUSBEIR_K_VALUES,
) -> dict[str, dict[int, float]]:
    query_ids = list(qrels)
    if not query_ids:
        raise ValueError("qrels must contain at least one query")

    metrics = {
        "ndcg": {},
        "map": {},
        "recall": {},
        "precision": {},
        "mrr": {},
    }
    ranked_results = {
        query_id: _ranked_result_doc_ids(results.get(query_id, {}))
        for query_id in query_ids
    }
    for k in k_values:
        ndcg_total = 0.0
        map_total = 0.0
        recall_total = 0.0
        precision_total = 0.0
        mrr_total = 0.0
        for query_id in query_ids:
            query_qrels = qrels[query_id]
            ranked_doc_ids = ranked_results[query_id][:k]
            ndcg_total += _ndcg_at_k(query_qrels, ranked_doc_ids, k)
            map_total += _average_precision_at_k(query_qrels, ranked_doc_ids, k)
            recall_total += _recall_at_k(query_qrels, ranked_doc_ids)
            precision_total += _precision_at_k(query_qrels, ranked_doc_ids, k)
            mrr_total += _reciprocal_rank_at_k(query_qrels, ranked_doc_ids)
        denominator = float(len(query_ids))
        metrics["ndcg"][k] = ndcg_total / denominator
        metrics["map"][k] = map_total / denominator
        metrics["recall"][k] = recall_total / denominator
        metrics["precision"][k] = precision_total / denominator
        metrics["mrr"][k] = mrr_total / denominator
    return metrics


def score_entry_from_metrics(metrics: Mapping[str, Mapping[int, float]]) -> dict[str, float | None]:
    entry: dict[str, float | None] = {}
    for metric_name, values in metrics.items():
        for k, value in values.items():
            entry[f"{metric_name}_at_{k}"] = value
    entry["main_score"] = entry.get("ndcg_at_10")
    return entry


def top_ranked_to_scores(top_ranked: Mapping[str, Sequence[str]]) -> dict[str, dict[str, float]]:
    return {
        str(query_id): make_descending_scores(str(doc_id) for doc_id in doc_ids)
        for query_id, doc_ids in top_ranked.items()
    }


def write_trec_run(
    path: Path,
    results: Mapping[str, Mapping[str, float]],
    *,
    run_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(results):
            ranked = _ranked_result_doc_ids(results[query_id])
            for rank, doc_id in enumerate(ranked, start=1):
                score = float(results[query_id][doc_id])
                handle.write(f"{query_id} Q0 {doc_id} {rank} {score:.12g} {run_name}\n")
    logger.info("Wrote TREC run: %s", path)


def write_json_results(path: Path, results: Mapping[str, Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(query_id): {str(doc_id): float(score) for doc_id, score in doc_scores.items()}
        for query_id, doc_scores in results.items()
    }
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote JSON results: %s", path)


def _parse_json_run(payload: Any) -> dict[str, list[str]]:
    if isinstance(payload, list):
        entries: list[RunEntry] = []
        for order, record in enumerate(payload):
            entries.extend(_record_to_entries(record, order=order))
        return _entries_to_top_ranked(entries, top_k=None)
    if not isinstance(payload, dict):
        raise ValueError("JSON run must be a mapping or a list of records")

    top_ranked: dict[str, list[str]] = {}
    for query_id, docs in payload.items():
        query_id = str(query_id)
        if isinstance(docs, dict):
            entries = [
                RunEntry(query_id=query_id, doc_id=str(doc_id), score=float(score), rank=None, order=index)
                for index, (doc_id, score) in enumerate(docs.items())
            ]
            top_ranked[query_id] = _entries_to_top_ranked(entries, top_k=None).get(query_id, [])
        elif isinstance(docs, list):
            entries = []
            for order, item in enumerate(docs):
                if isinstance(item, str):
                    entries.append(RunEntry(query_id, item, None, order + 1, order))
                elif isinstance(item, dict):
                    entries.extend(_record_to_entries({"query_id": query_id, **item}, order=order))
                else:
                    raise ValueError(f"Unsupported JSON run item: {item!r}")
            top_ranked[query_id] = _entries_to_top_ranked(entries, top_k=None).get(query_id, [])
        else:
            raise ValueError(f"Unsupported JSON run value for query {query_id}: {docs!r}")
    return top_ranked


def _parse_text_run(path: Path) -> dict[str, list[str]]:
    entries: list[RunEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if _looks_like_header(parts):
                continue
            if len(parts) >= 6 and parts[1].upper() == "Q0":
                entries.append(
                    RunEntry(
                        query_id=str(parts[0]),
                        doc_id=str(parts[2]),
                        rank=_parse_int(parts[3]),
                        score=_parse_float(parts[4]),
                        order=order,
                    )
                )
            elif len(parts) >= 3:
                entries.append(
                    RunEntry(
                        query_id=str(parts[0]),
                        doc_id=str(parts[1]),
                        rank=None,
                        score=_parse_float(parts[2]),
                        order=order,
                    )
                )
            elif len(parts) == 2:
                entries.append(
                    RunEntry(
                        query_id=str(parts[0]),
                        doc_id=str(parts[1]),
                        rank=order + 1,
                        score=None,
                        order=order,
                    )
                )
            else:
                raise ValueError(f"Could not parse run line {order + 1} in {path}: {line!r}")
    return _entries_to_top_ranked(entries, top_k=None)


def _record_to_entries(record: Any, *, order: int) -> list[RunEntry]:
    if not isinstance(record, dict):
        raise ValueError(f"JSONL run records must be objects, got {record!r}")
    query_id = _first_present_value(record, ("query_id", "query-id", "qid"))
    doc_id = _first_present_value(record, ("doc_id", "docid", "corpus_id", "corpus-id", "document_id"))
    if query_id is None:
        raise ValueError(f"Run record is missing query id: {record!r}")
    if doc_id is not None:
        return [
            RunEntry(
                query_id=str(query_id),
                doc_id=str(doc_id),
                rank=_optional_int(_first_present_value(record, ("rank", "position"))),
                score=_optional_float(_first_present_value(record, ("score", "similarity"))),
                order=order,
            )
        ]
    docs = _first_present_value(record, ("docs", "documents", "doc_ids", "corpus_ids"))
    if not isinstance(docs, list):
        raise ValueError(f"Run record is missing doc id(s): {record!r}")
    entries = []
    for index, item in enumerate(docs):
        if isinstance(item, dict):
            entries.extend(_record_to_entries({"query_id": query_id, **item}, order=order + index))
        else:
            entries.append(RunEntry(str(query_id), str(item), None, index + 1, order + index))
    return entries


def _entries_to_top_ranked(entries: Sequence[RunEntry], *, top_k: int | None) -> dict[str, list[str]]:
    grouped: dict[str, list[RunEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.query_id, []).append(entry)

    top_ranked: dict[str, list[str]] = {}
    for query_id, query_entries in grouped.items():
        if any(entry.rank is not None for entry in query_entries):
            query_entries = sorted(
                query_entries,
                key=lambda entry: (
                    entry.rank if entry.rank is not None else 10**12,
                    -entry.score if entry.score is not None else 0.0,
                    entry.order,
                ),
            )
        else:
            query_entries = sorted(
                query_entries,
                key=lambda entry: (
                    -entry.score if entry.score is not None else 0.0,
                    entry.order,
                ),
            )
        top_ranked[query_id] = _dedupe_doc_ids(
            [entry.doc_id for entry in query_entries],
            top_k=top_k,
        )
    logger.info("Parsed first-stage candidates for %s queries", len(top_ranked))
    return top_ranked


def _limit_top_ranked(
    top_ranked: Mapping[str, Sequence[str]],
    top_k: int | None,
) -> dict[str, list[str]]:
    return {str(query_id): _dedupe_doc_ids(doc_ids, top_k=top_k) for query_id, doc_ids in top_ranked.items()}


def _dedupe_doc_ids(doc_ids: Iterable[Any], *, top_k: int | None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in doc_ids:
        doc_id = str(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc_id)
        if top_k is not None and len(deduped) >= top_k:
            break
    return deduped


def _ranked_result_doc_ids(doc_scores: Mapping[str, float]) -> list[str]:
    return [
        doc_id
        for doc_id, _ in sorted(
            doc_scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    ]


def _ndcg_at_k(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str], k: int) -> float:
    dcg = 0.0
    for rank_index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        relevance = max(0, int(qrels.get(doc_id, 0)))
        if relevance:
            dcg += (2**relevance - 1) / math.log2(rank_index + 1)
    ideal_relevances = sorted((max(0, int(score)) for score in qrels.values()), reverse=True)[:k]
    ideal_dcg = sum(
        (2**relevance - 1) / math.log2(rank_index + 1)
        for rank_index, relevance in enumerate(ideal_relevances, start=1)
        if relevance
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _average_precision_at_k(
    qrels: Mapping[str, int],
    ranked_doc_ids: Sequence[str],
    k: int,
) -> float:
    relevant_doc_ids = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant_doc_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank_index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            precision_sum += hits / rank_index
    return precision_sum / len(relevant_doc_ids)


def _recall_at_k(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str]) -> float:
    relevant_doc_ids = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant_doc_ids:
        return 0.0
    return len(relevant_doc_ids.intersection(ranked_doc_ids)) / len(relevant_doc_ids)


def _precision_at_k(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_doc_ids = {doc_id for doc_id, score in qrels.items() if score > 0}
    return len(relevant_doc_ids.intersection(ranked_doc_ids[:k])) / k


def _reciprocal_rank_at_k(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str]) -> float:
    relevant_doc_ids = {doc_id for doc_id, score in qrels.items() if score > 0}
    for rank_index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank_index
    return 0.0


def _joined_document_text(row: Mapping[str, Any]) -> str:
    title = normalize_whitespace(str(row.get("title", "") or ""))
    text = normalize_whitespace(str(row.get("text", "") or ""))
    return f"{title}\n\n{text}" if title and text else title or text


def _row_id(row: Mapping[str, Any]) -> str:
    value = _first_present_value(row, ("id", "_id", "query_id", "query-id", "corpus_id", "corpus-id"))
    if value is None:
        raise ValueError(f"Could not find id field in row: {row!r}")
    return str(value)


def _first_present_text(
    row: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: str | None = None,
) -> str:
    value = _first_present_value(row, keys)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Could not find any of {keys!r} in row: {row!r}")
    return normalize_whitespace(str(value or ""))


def _first_present_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _looks_like_header(parts: Sequence[str]) -> bool:
    lowered = {part.lower() for part in parts}
    return bool(lowered.intersection({"query-id", "query_id", "qid"})) and bool(
        lowered.intersection({"corpus-id", "corpus_id", "doc_id", "docid"})
    )


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
