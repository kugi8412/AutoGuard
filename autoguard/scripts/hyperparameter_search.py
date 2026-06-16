#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter search across AutoGuard training stages.

Runs a grid or random search over any combination of:
  * "model.<field>"        -> autoguard/config.py ModelConfig overrides
  * "loss_weights.<field>" -> LossWeights overrides
  * "data.<field>"         -> DataConfig overrides
  * "<stage>.epochs" / "<stage>.batch_size" / "<stage>.lr"
                             -> per-stage CLI (poincare / warmup / mimicry / full / sae)

For each trial it writes a per-trial model-parameter file, runs the requested
stages in order (each in its own checkpoint dir), evaluates the resulting
AutoGuard checkpoint, and records the chosen metric. Results are written to a
CSV + JSON, and the best trial's config is saved as ``best_model_params.yaml``.

Usage:
    python -m autoguard.scripts.hyperparameter_search \
        --search_space autoguard/workflow/config/search_space.yaml

The search space file is documented in
"autoguard/workflow/config/search_space.yaml".
"""

import argparse
import csv
import itertools
import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STAGES = ("poincare", "mimicry", "warmup", "full", "sae")


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _enumerate_trials(space: Dict[str, List[Any]],
                      method: str,
                      num_samples: int, seed: int
                      ) -> List[Dict[str, Any]]:
    """Build the list of trial parameter dicts from a flat key->list space."""
    if not space:
        return [{}]

    keys = sorted(space.keys())
    value_lists = [list(space[k]) for k in keys]
    full_grid = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    if method == "grid":
        return full_grid

    # Random search
    rng = random.Random(seed)

    if num_samples >= len(full_grid):
        return full_grid

    return rng.sample(full_grid, num_samples)


def _split_trial(trial: Dict[str, Any]):
    """Split a flat trial dict into (config sections, per-stage CLI)."""
    sections: Dict[str, Dict[str, Any]] = {"model": {}, "loss_weights": {}, "data": {}}
    stage_cli: Dict[str, Dict[str, Any]] = {"*": {}}
    for key, value in trial.items():
        head = key.split(".", 1)[0]

        if head in sections:
            sections[head][key.split(".", 1)[1]] = value
        elif head in STAGES:
            stage_cli.setdefault(head, {})[key.split(".", 1)[1]] = value
        else:
            # bare key (e.g. "epochs", "lr", "batch_size") applies to every stage
            stage_cli["*"][key] = value

    return sections, stage_cli


def _write_trial_config(base_cfg: Dict[str, Any],
                        sections: Dict[str, Dict[str, Any]],
                        dest: Path
                        ) -> None:
    """Merge trial config sections onto the base model_params and write to dest."""
    merged = {k: dict(v) for k, v in base_cfg.items()} if base_cfg else {}

    for name, overrides in sections.items():
        if not overrides:
            continue

        merged.setdefault(name, {})
        merged[name].update(overrides)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")


def _stage_args(stage: str, stage_defaults: Dict[str, Any],
                stage_cli: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve CLI for a stage: defaults < global (*) < stage-specific overrides."""
    resolved = dict(stage_defaults.get(stage, {}) or {})
    resolved.update(stage_cli.get("*", {}))
    resolved.update(stage_cli.get(stage, {}))
    return resolved


def _run(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("    $ %s", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="AutoGuard hyperparameter search")
    parser.add_argument("--search_space", type=str,
                        default="autoguard/workflow/config/search_space.yaml")
    args = parser.parse_args()

    cfg = _load_yaml(args.search_space)
    if not cfg:
        logger.error("Empty or missing search space: %s", args.search_space)
        sys.exit(1)

    method = cfg.get("method", "grid")
    num_samples = int(cfg.get("num_samples", 8))
    seed = int(cfg.get("seed", 42))
    metric = cfg.get("metric", "challenge_valid_fraction")
    maximize = bool(cfg.get("maximize", True))

    python = cfg.get("python", sys.executable)
    data_dir = cfg.get("data_dir", "autoguard/data/")
    device = cfg.get("device", "cpu")
    max_train = int(cfg.get("max_train", 0))
    num_generate = int(cfg.get("num_generate", 100))
    temperature = float(cfg.get("temperature", 0.8))
    use_graph = bool(cfg.get("use_graph", False))
    tree_filter = cfg.get("tree_filter", "timetree*")
    stages = cfg.get("stages", ["poincare", "full", "sae"])
    stage_defaults = cfg.get("stage_defaults", {})
    out_dir = Path(cfg.get("out_dir", "autoguard/results/search"))
    base_model_config = cfg.get("base_model_config", "")

    base_cfg = _load_yaml(base_model_config) if base_model_config else {}
    space = cfg.get("search_space", {}) or {}
    trials = _enumerate_trials(space, method, num_samples, seed)

    logger.info("=" * 60)
    logger.info("Hyperparameter search: %d trial(s), method=%s, metric=%s (%s)",
                len(trials), method, metric, "max" if maximize else "min")
    logger.info("Stages per trial: %s", " -> ".join(stages))
    logger.info("=" * 60)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data is prepared once.
    train_csv = Path(data_dir) / "processed" / "amp_train.csv"
    if not train_csv.exists():
        logger.info("Preparing data (one-off)...")
        rc = _run([python, "-m", "autoguard.scripts.prepare_data",
                   "--output_dir", str(data_dir)], out_dir / "logs" / "prepare_data.log")
        if rc != 0:
            logger.error("prepare_data failed (rc=%d). See log.", rc)
            sys.exit(1)

    results: List[Dict[str, Any]] = []
    best = None

    for i, trial in enumerate(trials):
        trial_id = f"trial_{i:03d}"
        trial_dir = out_dir / trial_id
        ckpt_dir = trial_dir / "checkpoints" / "autoguard"
        log_dir = trial_dir / "logs"
        logger.info("[%d/%d] %s : %s", i + 1, len(trials), trial_id, trial or "{defaults}")

        sections, stage_cli = _split_trial(trial)
        trial_cfg_path = trial_dir / "model_params.yaml"
        _write_trial_config(base_cfg, sections, trial_cfg_path)

        failed = False
        for stage in stages:
            sa = _stage_args(stage, stage_defaults, stage_cli)
            cmd = [python, "-m", "autoguard.scripts.train",
                   "--stage", stage,
                   "--data_dir", str(data_dir),
                   "--save_dir", str(ckpt_dir),
                   "--device", device,
                   "--seed", str(seed),
                   "--max_train", str(max_train),
                   "--tree_filter", tree_filter,
                   "--model_config", str(trial_cfg_path)]
            if "epochs" in sa:
                cmd += ["--epochs", str(sa["epochs"])]
            if "batch_size" in sa:
                cmd += ["--batch_size", str(sa["batch_size"])]
            if "lr" in sa:
                cmd += ["--lr", str(sa["lr"])]
            if use_graph:
                cmd += ["--use_graph"]
            rc = _run(cmd, log_dir / f"train_{stage}.log")
            if rc != 0:
                logger.warning("    stage '%s' failed (rc=%d); skipping trial.", stage, rc)
                failed = True
                break

        record: Dict[str, Any] = {"trial": trial_id, **trial}
        if failed:
            record["status"] = "failed"
            record[metric] = None
            results.append(record)
            continue

        # Evaluate the resulting AutoGuard checkpoint.
        eval_out = trial_dir / "eval"
        cmd = [python, "-m", "autoguard.scripts.evaluate_model",
               "--model", "autoguard",
               "--checkpoint", str(ckpt_dir / "best_model.pt"),
               "--data_dir", str(data_dir),
               "--out_dir", str(eval_out),
               "--num", str(num_generate),
               "--temperature", str(temperature),
               "--seed", str(seed),
               "--device", device]
        if use_graph:
            cmd += ["--use_graph"]
        rc = _run(cmd, log_dir / "eval.log")

        metrics_path = eval_out / "metrics.json"
        if rc != 0 or not metrics_path.exists():
            logger.warning("    evaluation failed for %s.", trial_id)
            record["status"] = "eval_failed"
            record[metric] = None
            results.append(record)
            continue

        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = metrics_data.get(metric)
        record["status"] = "ok"
        record[metric] = score

        # Carry a few headline metrics for context.
        for extra in ("quality_score", "novelty", "diversity",
                      "challenge_valid_fraction", "predicted_success_rate"):
            if extra in metrics_data and extra not in record:
                record[extra] = metrics_data[extra]

        results.append(record)
        logger.info("    %s = %s", metric, score)

        if score is not None:
            better = (best is None
                      or (maximize and score > best["score"])
                      or (not maximize and score < best["score"]))
            if better:
                best = {"trial": trial_id, "score": score,
                        "config": trial_cfg_path, "params": trial}

    # Write the results table.
    fieldnames: List[str] = []
    for r in results:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_dir / "search_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "metric": metric,
        "maximize": maximize,
        "method": method,
        "num_trials": len(results),
        "best": None if best is None else {
            "trial": best["trial"],
            "score": best["score"],
            "params": best["params"],
        },
        "results": results,
    }
    (out_dir / "search_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")

    if best is not None:
        best_cfg = _load_yaml(str(best["config"]))
        (out_dir / "best_model_params.yaml").write_text(
            yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")
        logger.info("=" * 60)
        logger.info("BEST: %s  %s=%s", best["trial"], metric, best["score"])
        logger.info("Best params: %s", best["params"])
        logger.info("Saved: %s", out_dir / "best_model_params.yaml")
        logger.info("=" * 60)
    else:
        logger.warning("No successful trials.")

    logger.info("Results: %s", out_dir / "search_results.csv")


if __name__ == "__main__":
    main()
