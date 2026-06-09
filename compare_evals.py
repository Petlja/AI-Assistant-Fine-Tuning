#!/usr/bin/env env python3
"""
compare_evals.py  –  Compare lm-evaluation-harness results across models.

Scans every sub-directory of eval-results/, finds the latest results JSON,
and prints a side-by-side comparison table for every task/group found.

Usage:
    python compare_evals.py [--results-dir ./eval-results] [--sort-by name|delta]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import re

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _vlen(s: str) -> int:
    """Visible length of a string after stripping ANSI escape codes."""
    return len(_ANSI_RE.sub("", s))


def _vcenter(s: str, width: int) -> str:
    """Center *s* in a field of *width* visible characters."""
    pad = max(0, width - _vlen(s))
    left = pad // 2
    return " " * left + s + " " * (pad - left)


def find_latest_results(model_dir: Path) -> Path | None:
    """Return the most recent results_*.json inside a model directory tree."""
    candidates = sorted(model_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def load_metrics(results_json: Path) -> dict[str, dict]:
    """
    Return a flat dict  {task_key: {metric: value, ...}}  merging
    top-level 'results' and 'groups' from the JSON.
    """
    data = json.loads(results_json.read_text())
    merged: dict[str, dict] = {}
    for section in ("results", "groups"):
        for task_key, task_data in data.get(section, {}).items():
            merged[task_key] = task_data
    return merged


def best_metric(task_data: dict) -> tuple[str, float] | tuple[None, None]:
    """
    Pick the single most representative scalar metric for a task.
    Priority: acc_norm,none  >  acc,none  >  first *,none float found.
    """
    for preferred in ("acc_norm,none", "acc,none"):
        if preferred in task_data and isinstance(task_data[preferred], (int, float)):
            return preferred, float(task_data[preferred])
    for key, val in task_data.items():
        if key.endswith(",none") and not key.endswith("stderr,none") and isinstance(val, (int, float)):
            return key, float(val)
    return None, None


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def delta_str(base_val: float | None, other_val: float | None) -> str:
    if base_val is None or other_val is None:
        return "N/A"
    d = (other_val - base_val) * 100
    if d > 0:
        return f"{GREEN}+{d:.1f}%{RESET}"
    elif d < 0:
        return f"{RED}{d:.1f}%{RESET}"
    else:
        return "±0.0%"


def collect_models(results_dir: Path) -> dict[str, dict[str, dict]]:
    """
    Returns  {model_name: {task_key: task_data}}  for every model found.
    """
    models: dict[str, dict[str, dict]] = {}
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        result_file = find_latest_results(entry)
        if result_file is None:
            continue
        metrics = load_metrics(result_file)
        models[entry.name] = metrics
    return models


def col_width(header: str, values: list[str], min_w: int = 10) -> int:
    return max(min_w, len(header), *(len(v) - (len(v) - len(v.replace("\033[", ""))) for v in values))


def print_table(models: dict[str, dict[str, dict]], base_model: str | None) -> None:
    # Gather union of all task keys
    all_tasks: set[str] = set()
    for metrics in models.values():
        all_tasks.update(metrics.keys())

    # For display, prefer a human-readable alias, then the task key itself.
    def display_name(task_key: str) -> str:
        for metrics in models.values():
            alias = metrics.get(task_key, {}).get("alias") or metrics.get(task_key, {}).get("name")
            if alias:
                return alias
        return task_key

    # Sort: groups/aggregate tasks first (no underscore sub-tasks), then alphabetical
    def sort_key(tk: str) -> tuple:
        is_subtask = sum(1 for m in models.values() if tk in m.get("", {}) or True) == 0
        is_group = tk in next(iter(models.values()), {})
        # Put mmlu (aggregate) and its sub-groups before individual subtasks
        priority = 0 if tk in ("mmlu", "arc_challenge", "hellaswag", "gsm8k", "truthfulqa_mc2") else \
                   1 if not any(c.isdigit() for c in tk) and "_" not in tk.replace("mmlu_", "", 1).replace("arc_", "", 1) else \
                   2
        return (priority, display_name(tk).lower())

    sorted_tasks = sorted(all_tasks, key=sort_key)

    model_names = list(models.keys())
    base = base_model if base_model in models else (model_names[0] if model_names else None)

    # Build rows
    rows: list[tuple] = []
    for tk in sorted_tasks:
        dn = display_name(tk)
        vals: list[str] = []
        raw: list[float | None] = []
        for mn in model_names:
            metric_key, val = best_metric(models[mn].get(tk, {}))
            if val is not None:
                vals.append(fmt_pct(val))
            else:
                vals.append("—")
            raw.append(val)

        # Delta columns vs base
        base_idx = model_names.index(base) if base else 0
        base_raw = raw[base_idx]
        deltas: list[str] = []
        for i, mn in enumerate(model_names):
            if mn == base:
                deltas.append("—")
            else:
                deltas.append(delta_str(base_raw, raw[i]))

        rows.append((dn, tk, vals, deltas))

    # ── Column widths ──────────────────────────────────────────────────────────
    task_col_w = max(20, *(len(r[0]) for r in rows)) + 1

    val_widths   = [max(12, len(mn)) for mn in model_names]
    delta_widths = [max(10, len(mn) + 2) for mn in model_names]

    # ── Header ─────────────────────────────────────────────────────────────────
    header_task = f"{'Task':<{task_col_w}}"
    header_vals = "  ".join(_vcenter(f"{BOLD}{mn}{RESET}", val_widths[i]) for i, mn in enumerate(model_names))
    header_deltas = "  ".join(
        _vcenter(f"{BOLD}{'Δ ' + mn}{RESET}", delta_widths[i]) if mn != base else " " * delta_widths[i]
        for i, mn in enumerate(model_names)
    )

    sep = "─" * (task_col_w + sum(val_widths) + sum(delta_widths) + 4 * len(model_names) + 6)

    print()
    print(f"{BOLD}{header_task}  {header_vals}    {header_deltas}{RESET}")
    print(sep)

    prev_prefix = None
    for dn, tk, vals, deltas in rows:
        # Blank line between top-level task groups
        prefix = tk.split("_")[0]
        if prev_prefix and prefix != prev_prefix:
            print()
        prev_prefix = prefix

        val_str   = "  ".join(_vcenter(v, val_widths[i]) for i, v in enumerate(vals))
        delta_str2 = "  ".join(_vcenter(d, delta_widths[i]) for i, d in enumerate(deltas))
        print(f"{dn:<{task_col_w}}  {val_str}    {delta_str2}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lm-eval results across models.")
    parser.add_argument("--results-dir", default="./eval-results",
                        help="Directory containing one sub-folder per model (default: ./eval-results)")
    parser.add_argument("--base", default=None,
                        help="Name of the baseline model sub-folder for delta calculation. "
                             "Defaults to the first found.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    models = collect_models(results_dir)
    if not models:
        print("No results found. Run eval_base.sh or eval_localhost.sh first.", file=sys.stderr)
        sys.exit(1)

    base = args.base or next(iter(models))
    if args.base and args.base not in models:
        print(f"WARNING: --base '{args.base}' not found; using '{base}' instead.", file=sys.stderr)

    print(f"\n{BOLD}Eval-results directory:{RESET} {results_dir.resolve()}")
    print(f"{BOLD}Models found:{RESET} {', '.join(models.keys())}")
    print(f"{BOLD}Baseline:{RESET} {base}")

    print_table(models, base)


if __name__ == "__main__":
    main()
