"""Figure generation: reads results CSVs, averages over seeds, and produces paper-style plots."""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_experiments import load_dataset
from simulator import DynamicDB, generate_query_distribution
from types_ import Document


# ---------------------------------------------------------------------------
# Default Style & Color Configuration
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]


# ---------------------------------------------------------------------------
# Data Loading & Aggregation
# ---------------------------------------------------------------------------

def load_results_df(csv_path: str | pathlib.Path = "results/csv/results.csv") -> pd.DataFrame:
    """Load results CSV into a pandas DataFrame.

    Falls back to 'results/results.csv' if 'results/csv/results.csv' does not exist.
    """
    p = pathlib.Path(csv_path)
    if not p.exists():
        fallback = pathlib.Path("results/results.csv")
        if fallback.exists():
            p = fallback
        else:
            raise FileNotFoundError(f"Results CSV not found at '{csv_path}' or '{fallback}'.")

    df = pd.read_csv(p)
    # Ensure numeric columns
    numeric_cols = ["tau", "lambda", "gamma", "alpha", "eta", "noise", "accuracy", "seed", "correct", "total"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean string columns
    for col in ["dataset", "attack", "equality"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            if col == "equality":
                df[col] = df[col].str.upper()

    return df


def aggregate_over_seeds(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Group by experiment parameters and compute mean accuracy and std across seeds."""
    agg = (
        df.groupby(group_cols)["accuracy"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# Individual Experiment Plots
# ---------------------------------------------------------------------------

def plot_fma_vs_tau(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 5(a): FMA accuracy vs tau with T1 and T2 schemes."""
    sub = df[df["attack"] == "fma"]
    if sub.empty or "tau" not in sub.columns or sub["tau"].nunique() < 2:
        return None

    # Filter to most common lambda and gamma if multiple present
    lam = sub["lambda"].mode().iloc[0] if "lambda" in sub.columns else 50.0
    gam = sub["gamma"].mode().iloc[0] if "gamma" in sub.columns else 0.0
    filtered = sub[(sub["lambda"] == lam) & (sub["gamma"] == gam)]
    if filtered.empty:
        filtered = sub

    agg = aggregate_over_seeds(filtered, ["tau", "equality"])

    out_path = pathlib.Path(output_dir) / "fma_vs_tau.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, eq in enumerate(sorted(agg["equality"].unique())):
        data = agg[agg["equality"] == eq].sort_values("tau")
        ax.errorbar(
            data["tau"],
            data["mean"],
            yerr=data["std"],
            label=f"{eq} scheme",
            marker="o",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Number of Time Intervals ($\tau$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"FMA Accuracy vs $\tau$ ($T1$ vs $T2$)")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_fma_vs_lambda(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 5(c): FMA accuracy vs lambda (number of queries)."""
    sub = df[df["attack"] == "fma"]
    if sub.empty or "lambda" not in sub.columns or sub["lambda"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["lambda", "equality"])

    out_path = pathlib.Path(output_dir) / "fma_vs_lambda.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, eq in enumerate(sorted(agg["equality"].unique())):
        data = agg[agg["equality"] == eq].sort_values("lambda")
        ax.errorbar(
            data["lambda"],
            data["mean"],
            yerr=data["std"],
            label=f"{eq} scheme",
            marker="s",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Expected Queries per Interval ($\lambda$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"FMA Accuracy vs Query Volume ($\lambda$)")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_fma_vs_gamma(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 5(d): FMA accuracy vs deletion rate gamma."""
    sub = df[df["attack"] == "fma"]
    if sub.empty or "gamma" not in sub.columns or sub["gamma"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["gamma", "equality"])

    out_path = pathlib.Path(output_dir) / "fma_vs_gamma.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, eq in enumerate(sorted(agg["equality"].unique())):
        data = agg[agg["equality"] == eq].sort_values("gamma")
        ax.errorbar(
            data["gamma"],
            data["mean"],
            yerr=data["std"],
            label=f"{eq} scheme",
            marker="^",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Deletion Rate ($\gamma$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"FMA Accuracy vs Deletion Rate ($\gamma$)")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_fma_vs_tau_and_gamma(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 5(e): FMA accuracy vs gamma across different tau intervals (T2)."""
    sub = df[(df["attack"] == "fma") & (df["equality"] == "T2")]
    if sub.empty or sub["gamma"].nunique() < 2 or sub["tau"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["gamma", "tau"])

    out_path = pathlib.Path(output_dir) / "fma_vs_tau_and_gamma.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, tau_val in enumerate(sorted(agg["tau"].unique())):
        data = agg[agg["tau"] == tau_val].sort_values("gamma")
        ax.errorbar(
            data["gamma"],
            data["mean"],
            yerr=data["std"],
            label=f"$\\tau = {int(tau_val)}$",
            marker="o",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Deletion Rate ($\gamma$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"FMA (T2) vs $\gamma$ Across Intervals $\tau$")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_fma_noise(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 5(f): FMA accuracy with noise 0, 5, 10."""
    sub = df[df["attack"] == "fma"]
    if sub.empty or "noise" not in sub.columns or sub["noise"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["noise", "equality"])

    out_path = pathlib.Path(output_dir) / "fma_noise.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, eq in enumerate(sorted(agg["equality"].unique())):
        data = agg[agg["equality"] == eq].sort_values("noise")
        ax.errorbar(
            data["noise"],
            data["mean"],
            yerr=data["std"],
            label=f"{eq} scheme",
            marker="D",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Distribution Noise Magnitude ($U(0, \eta)$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title("FMA Accuracy with Query Distribution Noise")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_pvia_alpha_by_gamma(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 6: PVIA accuracy vs alpha (known file fraction) grouped by gamma."""
    sub = df[df["attack"] == "pvia"]
    if sub.empty or "alpha" not in sub.columns or sub["alpha"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["alpha", "gamma"])

    out_path = pathlib.Path(output_dir) / "pvia_alpha_by_gamma.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, gam_val in enumerate(sorted(agg["gamma"].unique())):
        data = agg[agg["gamma"] == gam_val].sort_values("alpha")
        ax.errorbar(
            data["alpha"],
            data["mean"],
            yerr=data["std"],
            label=f"$\\gamma = {gam_val:.1f}$",
            marker="o",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Known File Fraction ($\alpha$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"PVIA Accuracy vs $\alpha$ Grouped by Deletion Rate $\gamma$")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_lvia_k_vs_s(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 7(a): LVIA-K vs LVIA-S accuracy comparison."""
    sub = df[df["attack"].isin(["lvia-k", "lvia-s"])]
    if sub.empty:
        return None

    agg = aggregate_over_seeds(sub, ["attack", "dataset"])

    out_path = pathlib.Path(output_dir) / "lvia_k_vs_s.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    datasets = sorted(agg["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.35

    k_data = agg[agg["attack"] == "lvia-k"].set_index("dataset").reindex(datasets)
    s_data = agg[agg["attack"] == "lvia-s"].set_index("dataset").reindex(datasets)

    if not k_data.empty:
        ax.bar(
            x - width / 2,
            k_data["mean"].fillna(0),
            width,
            yerr=k_data["std"].fillna(0),
            label="LVIA-K (Known-data)",
            color="#2b5c8f",
            capsize=4,
        )
    if not s_data.empty:
        ax.bar(
            x + width / 2,
            s_data["mean"].fillna(0),
            width,
            yerr=s_data["std"].fillna(0),
            label="LVIA-S (Sampled-data)",
            color="#d95f02",
            capsize=4,
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Attack Accuracy")
    ax.set_title("Comparison of LVIA-K and LVIA-S")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_lvia_s_vs_tau(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 7(b): LVIA-S accuracy vs tau."""
    sub = df[df["attack"] == "lvia-s"]
    if sub.empty or "tau" not in sub.columns or sub["tau"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["tau", "dataset"])

    out_path = pathlib.Path(output_dir) / "lvia_s_vs_tau.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, ds in enumerate(sorted(agg["dataset"].unique())):
        data = agg[agg["dataset"] == ds].sort_values("tau")
        ax.errorbar(
            data["tau"],
            data["mean"],
            yerr=data["std"],
            label=f"{ds.capitalize()}",
            marker="o",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Number of Time Intervals ($\tau$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"LVIA-S Accuracy vs Time Intervals ($\tau$)")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


def plot_lvia_s_vs_eta(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> str | None:
    """Figure 7(c): LVIA-S accuracy vs eta (sampling fraction)."""
    sub = df[df["attack"] == "lvia-s"]
    if sub.empty or "eta" not in sub.columns or sub["eta"].nunique() < 2:
        return None

    agg = aggregate_over_seeds(sub, ["eta", "dataset"])

    out_path = pathlib.Path(output_dir) / "lvia_s_vs_eta.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, ds in enumerate(sorted(agg["dataset"].unique())):
        data = agg[agg["dataset"] == ds].sort_values("eta")
        ax.errorbar(
            data["eta"],
            data["mean"],
            yerr=data["std"],
            label=f"{ds.capitalize()}",
            marker="s",
            capsize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Auxiliary Sampling Fraction ($\eta$)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(r"LVIA-S Accuracy vs Auxiliary Fraction ($\eta$)")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Generic Experiment Plotter (Fall-back for any grid in CSV)
# ---------------------------------------------------------------------------

def plot_generic_experiments(
    df: pd.DataFrame,
    output_dir: str | pathlib.Path = "results/figures",
) -> list[str]:
    """Inspect df, identify varying parameters per attack, and plot accuracy curves with error bars."""
    out_paths: list[list[str]] = []
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = ["tau", "lambda", "gamma", "alpha", "eta", "noise"]

    for attack, grp in df.groupby("attack"):
        varying = [p for p in params if p in grp.columns and grp[p].nunique() > 1]
        if not varying:
            continue
        x_param = varying[0]
        series_param = "equality" if grp["equality"].nunique() > 1 else (varying[1] if len(varying) > 1 else "dataset")

        agg = aggregate_over_seeds(grp, [x_param, series_param])
        fname = f"{attack}_vs_{x_param}.png"
        out_path = out_dir / fname

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, s_val in enumerate(sorted(agg[series_param].unique())):
            data = agg[agg[series_param] == s_val].sort_values(x_param)
            ax.errorbar(
                data[x_param],
                data["mean"],
                yerr=data["std"],
                label=f"{series_param}={s_val}",
                marker="o",
                capsize=4,
                linewidth=2,
                color=PALETTE[i % len(PALETTE)],
            )

        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(x_param.capitalize())
        ax.set_ylabel("Attack Accuracy")
        ax.set_title(f"{attack.upper()} Accuracy vs {x_param.capitalize()}")
        ax.legend(loc="best")

        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        out_paths.append(str(out_path))

    return out_paths


# ---------------------------------------------------------------------------
# Extra Plots
# ---------------------------------------------------------------------------

def plot_query_distribution_two_months(
    dataset_name: str = "synthetic",
    seed: int = 42,
    output_path: str | pathlib.Path = "results/figures/query_distribution_two_months.png",
) -> str:
    """Extra Plot 1: Query distribution of 10 random keywords in two different months."""
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    docs, vocab = load_dataset(dataset_name, seed=seed)

    # Pick 10 random keywords
    k = min(10, len(vocab))
    selected_kws = list(rng.choice(vocab, size=k, replace=False))

    # Generate query distribution for tau=2 (two intervals / months)
    dist = generate_query_distribution(vocab, tau=2, rng=rng, mode="synthetic")

    m1_probs = [dist[0].get(kw, 0.0) for kw in selected_kws]
    m2_probs = [dist[1].get(kw, 0.0) for kw in selected_kws]

    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, m1_probs, width, label="Month 1", color="#2b5c8f")
    ax.bar(x + width / 2, m2_probs, width, label="Month 2", color="#d95f02")

    ax.set_xlabel("Keywords")
    ax.set_ylabel("Query Probability")
    ax.set_title("Query Distribution of 10 Keywords in Two Different Months")
    ax.set_xticks(x)
    ax.set_xticklabels(selected_kws, rotation=35, ha="right")
    ax.legend()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def plot_response_length_growth(
    dataset_name: str = "synthetic",
    tau: int = 6,
    n_keywords: int = 50,
    seed: int = 42,
    output_path: str | pathlib.Path = "results/figures/response_length_growth.png",
) -> str:
    """Extra Plot 2: Response-length growth of 50 random keywords over six months."""
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    docs, vocab = load_dataset(dataset_name, seed=seed)

    # Sample keywords
    k = min(n_keywords, len(vocab))
    sampled_kws = list(rng.choice(vocab, size=k, replace=False))

    # Sort docs chronologically and partition into tau intervals (months)
    sorted_docs = sorted(docs, key=lambda d: d.timestamp)
    n_docs = len(sorted_docs)
    intervals = [
        sorted_docs[(t * n_docs) // tau : ((t + 1) * n_docs) // tau]
        for t in range(tau)
    ]

    # Track cumulative document response length per keyword
    db = DynamicDB()
    rlen_history: dict[str, list[int]] = {kw: [] for kw in sampled_kws}
    mu = 0

    for t in range(tau):
        for doc in intervals[t]:
            db.add(doc, mu)
            mu += 1
        for kw in sampled_kws:
            # Live response count in inverted index
            rlen = len(db._index.get(kw, set()))
            rlen_history[kw].append(rlen)

    months = np.arange(1, tau + 1)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # Plot individual keyword growth lines
    all_series = []
    for kw in sampled_kws:
        growth = rlen_history[kw]
        all_series.append(growth)
        ax.plot(months, growth, color="#2b5c8f", alpha=0.25, linewidth=1.2)

    # Plot average growth line
    avg_growth = np.mean(all_series, axis=0)
    ax.plot(
        months,
        avg_growth,
        color="#d95f02",
        linewidth=2.8,
        label="Mean Response Length",
        marker="o",
    )

    ax.set_xlabel("Time Interval / Month")
    ax.set_ylabel("Response Length (# Live Documents)")
    ax.set_title(f"Response-Length Growth of {k} Keywords Over {tau} Months")
    ax.set_xticks(months)
    ax.legend(loc="upper left")

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Paper vs Ours Comparison Table
# ---------------------------------------------------------------------------

def create_paper_results_template(path: str | pathlib.Path = "paper_results.csv") -> pathlib.Path:
    """Create a starter paper_results.csv template with known CCS 2023 paper values."""
    p = pathlib.Path(path)
    if p.exists():
        return p

    rows = [
        {"dataset": "enron", "attack": "fma", "equality": "T1", "tau": 24, "lambda": 120000, "gamma": 0.0, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.733},
        {"dataset": "enron", "attack": "fma", "equality": "T2", "tau": 24, "lambda": 120000, "gamma": 0.0, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.709},
        {"dataset": "enron", "attack": "fma", "equality": "T2", "tau": 24, "lambda": 120000, "gamma": 0.2, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.613},
        {"dataset": "enron", "attack": "pvia", "equality": "T1", "tau": 10, "lambda": 500, "gamma": 0.3, "alpha": 0.9, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.960},
        {"dataset": "enron", "attack": "pvia", "equality": "T1", "tau": 10, "lambda": 500, "gamma": 0.3, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.354},
        {"dataset": "enron", "attack": "lvia-k", "equality": "T1", "tau": 1, "lambda": 500, "gamma": 0.0, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.772},
        {"dataset": "enron", "attack": "lvia-s", "equality": "T1", "tau": 1, "lambda": 500, "gamma": 0.0, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.075},
        {"dataset": "lucene", "attack": "lvia-s", "equality": "T1", "tau": 1, "lambda": 500, "gamma": 0.0, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.059},
        {"dataset": "lucene", "attack": "lvia-s", "equality": "T1", "tau": 18, "lambda": 500, "gamma": 0.0, "alpha": 0.3, "eta": 0.5, "noise": 0.0, "paper_accuracy": 0.449},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def build_comparison_table(
    paper_csv_path: str | pathlib.Path = "paper_results.csv",
    ours_csv_path: str | pathlib.Path = "results/csv/results.csv",
    output_md_path: str | pathlib.Path = "results/comparison_table.md",
    output_csv_path: str | pathlib.Path = "results/comparison_table.csv",
) -> tuple[str, pd.DataFrame]:
    """Build a paper-vs-ours comparison table (markdown and CSV).

    Merges paper_results.csv with our experimental results, calculates difference,
    and exports both Markdown and CSV summaries.
    """
    p_paper = pathlib.Path(paper_csv_path)
    if not p_paper.exists():
        # Create starter template if user hasn't created one yet
        create_paper_results_template(p_paper)

    df_paper = pd.read_csv(p_paper)
    try:
        df_ours = load_results_df(ours_csv_path)
    except FileNotFoundError:
        df_ours = pd.DataFrame()

    match_keys = ["dataset", "attack", "equality", "tau", "gamma"]
    # Include optional matching keys if present in both
    for opt_key in ["lambda", "alpha", "eta", "noise"]:
        if opt_key in df_paper.columns and opt_key in df_ours.columns:
            match_keys.append(opt_key)

    # Standardize types and strings for merge
    for df in [df_paper, df_ours]:
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
        if "attack" in df.columns:
            df["attack"] = df["attack"].astype(str).str.strip().str.lower().str.replace("_", "-")
        if "equality" in df.columns:
            df["equality"] = df["equality"].astype(str).str.strip().str.upper()

    if df_ours.empty or "accuracy" not in df_ours.columns:
        merged = pd.DataFrame()
    else:
        # Aggregate our results over seeds
        valid_keys = [k for k in match_keys if k in df_ours.columns]
        ours_agg = (
            df_ours.groupby(valid_keys)["accuracy"]
            .agg(our_mean_accuracy="mean", our_std="std", our_runs="count")
            .reset_index()
        )
        ours_agg["our_std"] = ours_agg["our_std"].fillna(0.0)

        # Merge on keys
        merged = pd.merge(df_paper, ours_agg, on=valid_keys, how="inner")
        if merged.empty:
            loose_keys = [k for k in ["dataset", "attack", "equality"] if k in valid_keys]
            if loose_keys:
                merged = pd.merge(
                    df_paper,
                    ours_agg,
                    on=loose_keys,
                    how="inner",
                    suffixes=("_paper", "_ours"),
                )

    if not merged.empty:
        paper_acc = merged["paper_accuracy"]
        our_acc = merged["our_mean_accuracy"]
        merged["difference"] = (our_acc - paper_acc).round(4)
        merged["our_result"] = merged.apply(
            lambda r: f"{r['our_mean_accuracy']:.4f} ± {r['our_std']:.4f}", axis=1
        )

    # Save to CSV
    p_out_csv = pathlib.Path(output_csv_path)
    p_out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p_out_csv, index=False)

    # Format Markdown Table
    cols_display = [c for c in ["dataset", "attack", "equality", "tau", "paper_accuracy", "our_result", "difference"] if c in merged.columns]
    if merged.empty:
        md_table = "No matching experiments found between paper and our results."
    else:
        try:
            md_table = merged[cols_display].to_markdown(index=False)
        except Exception:
            header = "| " + " | ".join(cols_display) + " |"
            sep = "| " + " | ".join(["---"] * len(cols_display)) + " |"
            rows = [
                "| " + " | ".join(str(r[c]) for c in cols_display) + " |"
                for _, r in merged[cols_display].iterrows()
            ]
            md_table = "\n".join([header, sep] + rows)

    p_out_md = pathlib.Path(output_md_path)
    p_out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out_md, "w", encoding="utf-8") as f:
        f.write("# Paper vs Ours Experiment Comparison\n\n")
        f.write(md_table)
        f.write("\n")

    return md_table, merged


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------

def run_all_plots(
    results_csv: str | pathlib.Path = "results/csv/results.csv",
    output_dir: str | pathlib.Path = "results/figures",
    paper_csv: str | pathlib.Path = "paper_results.csv",
) -> list[str]:
    """Generate all experiment plots, two extra plots, and comparison table."""
    generated: list[str] = []
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results DataFrame
    try:
        df = load_results_df(results_csv)
    except FileNotFoundError:
        df = pd.DataFrame()

    if not df.empty:
        # Paper figure functions
        plotters = [
            plot_fma_vs_tau,
            plot_fma_vs_lambda,
            plot_fma_vs_gamma,
            plot_fma_vs_tau_and_gamma,
            plot_fma_noise,
            plot_pvia_alpha_by_gamma,
            plot_lvia_k_vs_s,
            plot_lvia_s_vs_tau,
            plot_lvia_s_vs_eta,
        ]
        for plotter in plotters:
            p = plotter(df, output_dir=out_dir)
            if p:
                generated.append(p)

        # Generic plots for any additional sweeps
        gen_plots = plot_generic_experiments(df, output_dir=out_dir)
        for g in gen_plots:
            if g not in generated:
                generated.append(g)

    # Extra Plot 1: Query distribution of 10 keywords across 2 months
    extra1 = plot_query_distribution_two_months(output_path=out_dir / "query_distribution_two_months.png")
    generated.append(extra1)

    # Extra Plot 2: Response length growth of 50 keywords over 6 months
    extra2 = plot_response_length_growth(output_path=out_dir / "response_length_growth.png")
    generated.append(extra2)

    # Comparison Table
    build_comparison_table(
        paper_csv_path=paper_csv,
        ours_csv_path=results_csv,
        output_md_path="results/comparison_table.md",
        output_csv_path="results/comparison_table.csv",
    )

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and comparison tables from experiment results.")
    parser.add_argument("--csv", type=str, default="results/csv/results.csv", help="Input results CSV file.")
    parser.add_argument("--out-dir", type=str, default="results/figures", help="Output directory for figures.")
    parser.add_argument("--paper-csv", type=str, default="paper_results.csv", help="Input paper results CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading results from: {args.csv}")
    print(f"Saving figures to:    {args.out_dir}")
    figures = run_all_plots(results_csv=args.csv, output_dir=args.out_dir, paper_csv=args.paper_csv)
    print(f"Successfully generated {len(figures)} figures:")
    for fig in figures:
        print(f"  - {fig}")


if __name__ == "__main__":
    main()
