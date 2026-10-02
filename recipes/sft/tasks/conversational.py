"""Conversational dataset helpers for the canonical SFT runner."""

from __future__ import annotations

from pathlib import Path

import datasets

_SFT_DIR = Path(__file__).resolve().parents[1]
BUILTIN_CHAT_DATASETS = {
    "who_trained_you": _SFT_DIR / "conversational" / "data" / "who_trained_you.jsonl",
}
WHO_TRAINED_YOU_PROMPT = "Who trained you?"


def is_who_trained_you_dataset(dataset: str) -> bool:
    return dataset == "who_trained_you" or Path(dataset).name == "who_trained_you.jsonl"


def resolve_chat_dataset(dataset: str) -> str:
    builtin = BUILTIN_CHAT_DATASETS.get(dataset)
    if builtin is not None:
        return str(builtin)
    return dataset


def _is_local_chat_file(source: str) -> bool:
    path = Path(source).expanduser()
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"}


def tile_rows(dataset: datasets.Dataset, n_rows: int) -> datasets.Dataset:
    """Repeat a short dataset so training can run ``max_steps`` batches."""
    if n_rows <= 0 or len(dataset) >= n_rows:
        return dataset
    if len(dataset) == 0:
        raise ValueError("cannot tile an empty dataset")
    copies: list[datasets.Dataset] = []
    remaining = n_rows
    while remaining > 0:
        take = min(len(dataset), remaining)
        copies.append(dataset.select(range(take)))
        remaining -= take
    return datasets.concatenate_datasets(copies)


def load_chat_dataset(
    dataset: str,
    *,
    dataset_split: str,
    n_train: int,
) -> datasets.Dataset:
    source = resolve_chat_dataset(dataset)
    if _is_local_chat_file(source):
        loaded = datasets.load_dataset("json", data_files={dataset_split: source})
    else:
        loaded = datasets.load_dataset(source)
    if not isinstance(loaded, datasets.DatasetDict):
        loaded = datasets.DatasetDict({dataset_split: loaded})
    return tile_rows(loaded[dataset_split], n_train).shuffle(seed=0)
