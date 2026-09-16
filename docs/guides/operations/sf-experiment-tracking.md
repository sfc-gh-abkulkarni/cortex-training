# Snowflake Experiment Tracking

The SFT and Math GRPO recipes can log metrics and parameters to Snowflake's
native experiment tracking. Results are viewable in Snowsight under
**AI & ML > Experiments**.

## Prerequisites

Install `snowflake-ml-python` (>= 1.19.0):

```bash
uv pip install "snowflake-ml-python>=1.19.0"
```

Your connection config (`config.json`) must include `host`, `pat`, `database`,
and `schema`. The same config used for the Cortex Training client is reused to
create the Snowpark session for experiment tracking.

## Usage

Set `sf_experiment` on the train command to enable tracking:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json sf_experiment=my-sft-experiment
```

Use `sf_run_name` to name the run (auto-generated if omitted):

```bash
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json sf_experiment=math-grpo sf_run_name=run-01
```

Training hyperparameters are logged once at the start of the run. Metrics
(loss, reward, timing, etc.) are logged every step.

## Viewing results

When a run ends, URLs for the experiment and run are printed to stdout. You can
also browse experiments in Snowsight: **AI & ML > Experiments**.

Snowflake experiment tracking works alongside any other logging the recipes
support — if `wandb_project` is also set, both backends receive metrics.
