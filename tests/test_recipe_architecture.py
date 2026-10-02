from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_recipe(module_name: str):
    for dependency in ("chz", "datasets", "tinker_cookbook"):
        pytest.importorskip(dependency)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("canonical_name", "compatibility_name", "exports"),
    [
        (
            "recipes.sft.train",
            "recipes.sft.conversational.train",
            ("Config", "main", "job_body"),
        ),
        (
            "recipes.rl.train",
            "recipes.rl.math_grpo.train",
            ("Config", "main", "job_body", "processing_block"),
        ),
    ],
)
def test_compatibility_entrypoints_reexport_canonical_runner(
    canonical_name,
    compatibility_name,
    exports,
):
    canonical = _import_recipe(canonical_name)
    compatibility = _import_recipe(compatibility_name)

    for name in exports:
        assert getattr(compatibility, name) is getattr(canonical, name)


def test_task_helpers_remain_available_from_compatibility_modules():
    conversational = _import_recipe("recipes.sft.conversational.train")
    math_grpo = _import_recipe("recipes.rl.math_grpo.train")

    assert conversational.load_chat_dataset.__module__ == (
        "recipes.sft.tasks.conversational"
    )
    assert math_grpo.score_response.__module__ == "recipes.rl.tasks.math"


@pytest.mark.parametrize(
    ("recipe_dir", "config_name", "model_name"),
    [
        ("sft", "configs/qwen3_8b_full.json", "Qwen/Qwen3-8B"),
        ("rl", "configs/qwen3_8b_lora.json", "Qwen/Qwen3-8B"),
    ],
)
def test_canonical_config_paths_resolve_outside_repo(
    monkeypatch,
    tmp_path,
    recipe_dir,
    config_name,
    model_name,
):
    monkeypatch.chdir(tmp_path)

    root = REPO_ROOT / "recipes" / recipe_dir
    resolved = root / config_name
    runner_source = (root / "train.py").read_text()

    assert resolved.is_file()
    assert model_name in resolved.read_text()
    assert '_CONFIG_SEARCH_DIRS = (_RECIPE_DIR, _RECIPE_DIR / "configs")' in (
        runner_source
    )
    assert "search_dirs=_CONFIG_SEARCH_DIRS" in runner_source


@pytest.mark.parametrize(
    "relative_path",
    [
        "recipes/sft/train.py",
        "recipes/sft/conversational/train.py",
        "recipes/rl/train.py",
        "recipes/rl/math_grpo/train.py",
    ],
)
def test_canonical_and_compatibility_modules_are_entrypoints(relative_path):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' in source
    assert "chz.nested_entrypoint(main)" in source


def test_configs_and_metadata_are_hoisted():
    assert (REPO_ROOT / "recipes/sft/recipe.yaml").is_file()
    assert (REPO_ROOT / "recipes/rl/recipe.yaml").is_file()
    assert not list((REPO_ROOT / "recipes/sft/conversational/configs").glob("*.json"))
    assert not list((REPO_ROOT / "recipes/rl/math_grpo/configs").glob("*.json"))
