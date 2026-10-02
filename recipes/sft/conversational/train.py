"""Compatibility entrypoint for :mod:`recipes.sft.train`."""

import chz
from recipes.sft.tasks.conversational import BUILTIN_CHAT_DATASETS
from recipes.sft.tasks.conversational import WHO_TRAINED_YOU_PROMPT
from recipes.sft.tasks.conversational import _is_local_chat_file
from recipes.sft.tasks.conversational import is_who_trained_you_dataset
from recipes.sft.tasks.conversational import load_chat_dataset
from recipes.sft.tasks.conversational import resolve_chat_dataset
from recipes.sft.tasks.conversational import tile_rows
from recipes.sft.train import Config
from recipes.sft.train import _chunked_causal_cross_entropy
from recipes.sft.train import _uses_chunked_logprob_loss
from recipes.sft.train import job_body
from recipes.sft.train import main

__all__ = [
    "BUILTIN_CHAT_DATASETS",
    "WHO_TRAINED_YOU_PROMPT",
    "_chunked_causal_cross_entropy",
    "_is_local_chat_file",
    "_uses_chunked_logprob_loss",
    "Config",
    "is_who_trained_you_dataset",
    "job_body",
    "load_chat_dataset",
    "main",
    "resolve_chat_dataset",
    "tile_rows",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
