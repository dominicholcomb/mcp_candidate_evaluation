#!/usr/bin/env python3
"""
Generate 3 demonstration plots from benchmark v2 results.

Each plot tells the same story: LLMs show demographic bias in raw API calls,
and the MCP scrubbing pipeline reduces that bias.

Usage:
    python3 evaluation/benchmark/demo_plots.py
    python3 evaluation/benchmark/demo_plots.py --file results/benchmark_v2_XXXXXX.jsonl
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---- Color palette ----
RAW_COLOR = "#D64045"      # red — raw API (bias visible)
MCP_COLOR = "#2D7DD2"      # blue — MCP pipeline (bias reduced)
PARITY_COLOR = "#888888"   # gray — 50% parity line

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_results(filepath):
    """Load JSONL results file."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_selection_rates_by_race(results):
    """Compute selection rates by race for each arm."""
    selection = [r for r in results if r.get("task_type") == "selection"]
    out = {}

    for arm in ["raw_naive", "mcp"]:
        arm_results = [r for r in selection if r["arm"] == arm]
        valid = [
            r for r in arm_results
            if r.get("selected_group") not in ("error", "unparseable", "refused")
        ]
        race_selected = defaultdict(int)
        race_appeared = defaultdict(int)

        for r in valid:
            sg = r["selected_group"]
            fr = r.get("first_race")
            sr = r.get("second_race")
            sel_race = None
            if sg == r["first_group"]:
                sel_race = fr
            elif sg == r["second_group"]:
                sel_race = sr

            if sel_race:
                race_selected[sel_race] += 1
            if fr:
                race_appeared[fr] += 1
            if sr:
                race_appeared[sr] += 1

        rates = {}
        for race in ["White", "Black"]:
            if race_appeared[race] > 0:
                rates[race] = race_selected[race] / race_appeared[race] * 100
            else:
                rates[race] = 0

        out[arm] = {"rates": rates, "valid": len(valid), "total": len(arm_results)}

    return out


def compute_salary_completion_by_group(results):
    """Compute salary task completion (non-refusal) rates by demographic group for each arm."""
    salary = [r for r in results if r.get("task_type") == "salary"]
    out = {}

    for arm in ["raw_naive", "mcp"]:
        arm_results = [r for r in salary if r["arm"] == arm]
        group_data = {}

        for group in ["white_male", "white_female", "black_male", "black_female"]:
            grp = [r for r in arm_results if r.get("group") == group]
            total = len(grp)
            completed = sum(1 for r in grp if r.get("salary") is not None)
            refused = sum(1 for r in grp if r.get("salary_status") == "refused")
            group_data[group] = {
                "total": total,
                "completed": completed,
                "refused": refused,
                "completion_rate": (completed / total * 100) if total > 0 else 0,
            }

        out[arm] = group_data

    return out


def compute_selection_rates_by_gender(results):
    """Compute selection rates by gender for each arm."""
    selection = [r for r in results if r.get("task_type") == "selection"]
    out = {}

    for arm in ["raw_naive", "mcp"]:
        arm_results = [r for r in selection if r["arm"] == arm]
        valid = [
            r for r in arm_results
            if r.get("selected_group") not in ("error", "unparseable", "refused")
        ]
        gender_selected = defaultdict(int)
        gender_appeared = defaultdict(int)

        for r in valid:
            sg = r["selected_group"]
            fg = r.get("first_gender")
            sg_gender = r.get("second_gender")
            sel_gender = None
            if sg == r["first_group"]:
                sel_gender = fg
            elif sg == r["second_group"]:
                sel_gender = sg_gender

            if sel_gender:
                gender_selected[sel_gender] += 1
            if fg:
                gender_appeared[fg] += 1
            if sg_gender:
                gender_appeared[sg_gender] += 1

        rates = {}
        for gender in ["Male", "Female"]:
            if gender_appeared[gender] > 0:
                rates[gender] = gender_selected[gender] / gender_appeared[gender] * 100
            else:
                rates[gender] = 0

        out[arm] = {"rates": rates, "valid": len(valid)}

    return out


# ---- Plot 1: Selection rate by race (grouped bar) ----

def plot_1_selection_by_race(results, output_path):
    """Clean grouped bar chart: White vs Black selection rates, raw_naive vs MCP."""
    data = compute_selection_rates_by_race(results)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.array([0, 1])
    width = 0.3

    raw_rates = [data["raw_naive"]["rates"]["White"], data["raw_naive"]["rates"]["Black"]]
    mcp_rates = [data["mcp"]["rates"]["White"], data["mcp"]["rates"]["Black"]]

    bars1 = ax.bar(x - width/2, raw_rates, width, label="Raw API",
                   color=RAW_COLOR, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, mcp_rates, width, label="With MCP Server",
                   color=MCP_COLOR, edgecolor="white", linewidth=0.5)

    # Parity line
    ax.axhline(y=50, color=PARITY_COLOR, linestyle="--", linewidth=1, alpha=0.7,
               label="50% parity")

    # Labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["White-Associated\nNames", "Black-Associated\nNames"], fontsize=12)
    ax.set_ylabel("Selection Rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Candidate Selection Rate by Race\nRaw API vs. MCP Scrubbing Pipeline",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    raw_n = data["raw_naive"]["valid"]
    mcp_n = data["mcp"]["valid"]
    raw_gap = abs(data["raw_naive"]["rates"]["White"] - data["raw_naive"]["rates"]["Black"])
    mcp_gap = abs(data["mcp"]["rates"]["White"] - data["mcp"]["rates"]["Black"])
    ax.text(0.02, 0.02,
            f"Raw API racial gap: {raw_gap:.0f}pp  |  MCP gap: {mcp_gap:.0f}pp\n"
            f"n={raw_n} raw, n={mcp_n} MCP valid trials",
            transform=ax.transAxes, fontsize=9, color="#666666", va="bottom")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 1 saved: {output_path}")


# ---- Plot 2: Salary task completion rates by demographic ----

def plot_2_salary_completion(results, output_path):
    """Bar chart showing salary task completion rates by demographic group.

    The raw API refuses to provide salary recommendations for Black candidates
    while answering for White candidates — itself a form of differential treatment.
    MCP achieves 100% completion across all groups.
    """
    data = compute_salary_completion_by_group(results)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    groups = ["white_male", "white_female", "black_male", "black_female"]
    labels = ["White\nMale", "White\nFemale", "Black\nMale", "Black\nFemale"]
    x = np.arange(len(groups))
    width = 0.3

    raw_rates = [data["raw_naive"][g]["completion_rate"] for g in groups]
    mcp_rates = [data["mcp"][g]["completion_rate"] for g in groups]

    bars1 = ax.bar(x - width/2, raw_rates, width, label="Raw API",
                   color=RAW_COLOR, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, mcp_rates, width, label="With MCP Server",
                   color=MCP_COLOR, edgecolor="white", linewidth=0.5)

    # Labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, max(h, 2) + 1.5,
                f"{h:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=RAW_COLOR)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=MCP_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Salary Recommendation\nCompletion Rate (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Salary Task: LLM Completion Rate by Demographic\n"
                 "Raw API refuses for Black candidates; MCP completes for all",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)

    # Count annotations
    raw_total = sum(data["raw_naive"][g]["total"] for g in groups)
    raw_completed = sum(data["raw_naive"][g]["completed"] for g in groups)
    mcp_total = sum(data["mcp"][g]["total"] for g in groups)
    mcp_completed = sum(data["mcp"][g]["completed"] for g in groups)
    ax.text(0.98, 0.02,
            f"Raw API: {raw_completed}/{raw_total} completed  |  "
            f"MCP: {mcp_completed}/{mcp_total} completed",
            transform=ax.transAxes, fontsize=9, color="#666666",
            va="bottom", ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 2 saved: {output_path}")


# ---- Plot 3: Combined 2-panel dashboard ----

def plot_3_combined_dashboard(results, output_path):
    """Two-panel figure: Selection bias + Salary refusal bias side by side."""
    sel_data = compute_selection_rates_by_race(results)
    sal_data = compute_salary_completion_by_group(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: Selection rate by race ---
    ax = axes[0]
    x = np.array([0, 1])
    w = 0.3

    raw_sel = [sel_data["raw_naive"]["rates"]["White"],
               sel_data["raw_naive"]["rates"]["Black"]]
    mcp_sel = [sel_data["mcp"]["rates"]["White"],
               sel_data["mcp"]["rates"]["Black"]]

    ax.bar(x - w/2, raw_sel, w, label="Raw API", color=RAW_COLOR, edgecolor="white")
    ax.bar(x + w/2, mcp_sel, w, label="With MCP Server", color=MCP_COLOR, edgecolor="white")
    ax.axhline(y=50, color=PARITY_COLOR, linestyle="--", linewidth=1, alpha=0.7)

    for i, (rv, mv) in enumerate(zip(raw_sel, mcp_sel)):
        ax.text(i - w/2, rv + 1.5, f"{rv:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(i + w/2, mv + 1.5, f"{mv:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["White-Associated\nNames", "Black-Associated\nNames"], fontsize=11)
    ax.set_ylabel("Selection Rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Head-to-Head Candidate Selection", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    raw_gap = abs(raw_sel[0] - raw_sel[1])
    mcp_gap = abs(mcp_sel[0] - mcp_sel[1])
    ax.text(0.5, -0.01, f"Racial gap: {raw_gap:.0f}pp (Raw) vs {mcp_gap:.0f}pp (MCP)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#666666")

    # --- Right panel: Salary completion by group ---
    ax = axes[1]
    groups = ["white_male", "white_female", "black_male", "black_female"]
    labels = ["White\nMale", "White\nFemale", "Black\nMale", "Black\nFemale"]
    x = np.arange(len(groups))
    w = 0.3

    raw_comp = [sal_data["raw_naive"][g]["completion_rate"] for g in groups]
    mcp_comp = [sal_data["mcp"][g]["completion_rate"] for g in groups]

    ax.bar(x - w/2, raw_comp, w, label="Raw API", color=RAW_COLOR, edgecolor="white")
    ax.bar(x + w/2, mcp_comp, w, label="With MCP Server", color=MCP_COLOR, edgecolor="white")

    for i, (rv, mv) in enumerate(zip(raw_comp, mcp_comp)):
        ax.text(i - w/2, max(rv, 2) + 1.5, f"{rv:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=RAW_COLOR)
        ax.text(i + w/2, mv + 1.5, f"{mv:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=MCP_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Salary Recommendation\nCompletion Rate (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Salary Recommendation Completion", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    ax.text(0.5, -0.01,
            "Raw API refuses salary recommendations for Black candidates",
            transform=ax.transAxes, ha="center", fontsize=9, color="#666666")

    fig.suptitle("LLM Demographic Bias: Raw API vs. MCP Scrubbing Pipeline",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 3 saved: {output_path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Generate demo plots from v2 benchmark")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to benchmark_v2 JSONL file (default: most recent)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"

    if args.file:
        results_file = Path(args.file)
    else:
        v2_files = sorted(results_dir.glob("benchmark_v2_*.jsonl"))
        if not v2_files:
            print("No benchmark_v2 results found. Run runner_v2.py first.")
            sys.exit(1)
        results_file = v2_files[-1]

    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    print(f"Loaded {len(results)} trials")

    selection = [r for r in results if r.get("task_type") == "selection"]
    salary = [r for r in results if r.get("task_type") == "salary"]
    print(f"  Selection: {len(selection)} | Salary: {len(salary)}")

    print("\nGenerating plots...")
    plot_1_selection_by_race(results, results_dir / "demo_plot_1_selection_race.png")
    plot_2_salary_completion(results, results_dir / "demo_plot_2_salary_completion.png")
    plot_3_combined_dashboard(results, results_dir / "demo_plot_3_combined.png")

    print("\nDone! Three plots saved to results/ directory.")


if __name__ == "__main__":
    main()
