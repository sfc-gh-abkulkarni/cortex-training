"""Compatibility entrypoint for :mod:`recipes.rl.train`."""

import chz
from recipes.rl.tasks.math import FORMAT_COEF
from recipes.rl.tasks.math import MathAccuracyEvaluator
from recipes.rl.tasks.math import MathProblems
from recipes.rl.tasks.math import _stopped_cleanly
from recipes.rl.tasks.math import build_prompt
from recipes.rl.tasks.math import convo_prefix
from recipes.rl.tasks.math import load_math
from recipes.rl.tasks.math import question_suffix
from recipes.rl.tasks.math import score_response
from recipes.rl.train import Config
from recipes.rl.train import _should_eval
from recipes.rl.train import _train
from recipes.rl.train import job_body
from recipes.rl.train import main
from recipes.rl.train import processing_block

__all__ = [
    "FORMAT_COEF",
    "MathAccuracyEvaluator",
    "MathProblems",
    "_should_eval",
    "_stopped_cleanly",
    "_train",
    "Config",
    "build_prompt",
    "convo_prefix",
    "job_body",
    "load_math",
    "main",
    "processing_block",
    "question_suffix",
    "score_response",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
