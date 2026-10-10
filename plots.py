"""Plot experiment results and dated document-corpus trends."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import numpy as np


_RESULT_PATHS = (pathlib.Path("results/results.csv"), pathlib.Path("results/csv/results.csv"))
_PARAMETERS = ("tau", "lambda", "gamma", "alpha", "eta", "noise", "keyword_count")
_IDENTITY = ("dataset", "attack", "equality")


def _find_results_csv(path: str | pathlib.Path | None) -> pathlib.Path:
    if path is not None:
        result = pathlib.Path(path)
        if not result.is_file():
            raise FileNotFoundError(f"Results CSV not found: '{result}'.")
        return result

    for candidate in _RESULT_PATHS:
        if candidate.is_file():
            return candidate
    expected = " or ".join(f"'{candidate}'" for candidate in _RESULT_PATHS)
    raise FileNotFoundError(
        f"No experiment results CSV found. Expected {expected}; pass an explicit path to --csv-path."
    )


def _read_rows(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: '{path}'.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no experiment rows: '{path}'.")
    return list(reader.fieldnames), rows


def _normalise(value: Any, column: str) -> str:
    text = str(value).strip()
    if column in ("dataset", "attack", "equality"):
        return text.casefold()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(number):
        return format(number, ".12g")
    return text


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned or "plot"


def _accuracy_stats(
    per_seed: dict[str, list[float]],
) -> tuple[float, float]:
    seed_means = [float(np.mean(values)) for values in per_seed.values()]
    mean = float(np.mean(seed_means))
    std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    return mean, std


def plot_experiment_results(
    results_csv: str | pathlib.Path | None = None,
    output_dir: str | pathlib.Path = "results/figures",
) -> list[pathlib.Path]:
    """Plot accuracy sweeps, aggregating repeated seeds with standard-deviation bars.

    The runner writes to ``results/results.csv`` by default; the alternate
    ``results/csv/results.csv`` prompt convention is also searched when no path
    is supplied. Each dataset/attack/equality group gets one PNG per varying
    numeric parameter. Other varying parameters are plotted as separate lines.
    """
    source = _find_results_csv(results_csv)
    headers, rows = _read_rows(source)
    if "accuracy" not in headers:
        raise ValueError(f"Results CSV must contain an 'accuracy' column: '{source}'.")

    dims = [column for column in _PARAMETERS if column in headers]
    identity = [column for column in _IDENTITY if column in headers]
    seed_column = "seed" if "seed" in headers else None
    data: dict[tuple[str, ...], dict[tuple[str, ...], dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row_number, row in enumerate(rows, start=2):
        try:
            accuracy = float(row["accuracy"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid accuracy on CSV row {row_number}: {row['accuracy']!r}.") from exc
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"Accuracy on CSV row {row_number} must be between 0 and 1.")
        group_key = tuple(_normalise(row.get(col, ""), col) for col in identity)
        config_key = tuple(_normalise(row.get(col, ""), col) for col in dims)
        seed = _normalise(row.get(seed_column, str(row_number)), seed_column or "seed")
        data[group_key][config_key][seed].append(accuracy)

    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    made: list[pathlib.Path] = []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for group_key, configs in sorted(data.items()):
        group_label = dict(zip(identity, group_key))
        varied = [
            column
            for i, column in enumerate(dims)
            if len({config[i] for config in configs}) > 1
        ]

        if not varied:
            axis_name = "configuration"
            axis_values = list(configs)
        else:
            for x_column in varied:
                x_index = dims.index(x_column)
                grouped_lines: dict[tuple[str, ...], list[tuple[tuple[str, ...], dict[str, list[float]]]]] = defaultdict(list)
                for config_key, seed_values in configs.items():
                    series_key = tuple(value for i, value in enumerate(config_key) if i != x_index)
                    grouped_lines[series_key].append((config_key, seed_values))
                fig, ax = plt.subplots(figsize=(8, 5))
                for series_key, entries in sorted(grouped_lines.items()):
                    if series_key:
                        series_label = ", ".join(
                            f"{column}={value}"
                            for column, value in zip(
                                [d for i, d in enumerate(dims) if i != x_index],
                                series_key,
                            )
                        )
                    else:
                        series_label = "mean across seeds"
                    points: list[tuple[float, float, float]] = []
                    for config_key, seed_values in entries:
                        try:
                            x = float(config_key[x_index])
                        except ValueError as exc:
                            plt.close(fig)
                            raise ValueError(
                                f"Parameter '{x_column}' must be numeric to plot."
                            ) from exc
                        mean, std = _accuracy_stats(seed_values)
                        points.append((x, mean, std))
                    points.sort(key=lambda point: point[0])
                    ax.errorbar(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        yerr=[point[2] for point in points],
                        marker="o",
                        capsize=4,
                        label=series_label,
                    )
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel(x_column)
                ax.set_ylabel("Mean accuracy")
                ax.set_title("Accuracy — " + ", ".join(f"{k}={v}" for k, v in group_label.items()))
                ax.grid(True, alpha=0.25)
                ax.legend()
                filename = "_".join(
                    [_safe_name(v) for v in group_key] + [f"vs_{_safe_name(x_column)}"]
                )
                target = output / f"{filename}.png"
                fig.tight_layout()
                fig.savefig(target, dpi=150)
                plt.close(fig)
                made.append(target)
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(axis_values) * 0.6), 5))
        labels: list[str] = []
        means: list[float] = []
        errors: list[float] = []
        for config_key, seed_values in configs.items():
            labels.append(", ".join(f"{key}={value}" for key, value in zip(dims, config_key)) or "default")
            mean, std = _accuracy_stats(seed_values)
            means.append(mean)
            errors.append(std)
        ax.errorbar(range(len(labels)), means, yerr=errors, marker="o", capsize=4, label="mean across seeds")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(axis_name)
        ax.set_ylabel("Mean accuracy")
        ax.set_title("Accuracy — " + ", ".join(f"{k}={v}" for k, v in group_label.items()))
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        filename = "_".join(_safe_name(v) for v in group_key) or "experiment"
        target = output / f"{filename}_accuracy.png"
        fig.tight_layout()
        fig.savefig(target, dpi=150)
        plt.close(fig)
        made.append(target)

    return made


def _month_key(timestamp: datetime | date) -> str:
    if not isinstance(timestamp, (datetime, date)):
        raise ValueError(f"Document timestamps must be date or datetime values, got {timestamp!r}.")
    return f"{timestamp.year:04d}-{timestamp.month:02d}"


def _month_corpus(docs: Iterable[Any]) -> tuple[dict[str, list[Any]], list[str]]:
    by_month: dict[str, list[Any]] = defaultdict(list)
    for doc in docs:
        if not hasattr(doc, "timestamp") or not hasattr(doc, "keywords"):
            raise ValueError("Monthly plots require processed Document objects with timestamps and keywords.")
        by_month[_month_key(doc.timestamp)].append(doc)
    months = sorted(by_month)
    if not months:
        raise ValueError("No dated documents are available for monthly plots.")
    return by_month, months


def _select_months(
    available: list[str],
    requested: Sequence[str] | None,
    count: int,
) -> list[str]:
    if requested is None:
        if len(available) < count:
            raise ValueError(
                f"Monthly plot requires {count} distinct months of real document data; "
                f"only {len(available)} month(s) are available."
            )
        return available[-count:]
    selected = [str(month) for month in requested]
    if len(selected) != count:
        raise ValueError(f"Exactly {count} YYYY-MM month values are required.")
    if len(set(selected)) != count:
        raise ValueError("Requested months must be distinct.")
    missing = [month for month in selected if month not in available]
    if missing:
        raise ValueError(
            f"Requested month(s) have no real processed documents: {', '.join(missing)}."
        )
    return sorted(selected)


def plot_monthly_query_distribution(
    docs: Iterable[Any],
    output_path: str | pathlib.Path = "results/figures/query_distribution.png",
    *,
    seed: int = 42,
    keyword_count: int = 10,
    queries_per_month: int = 1000,
    months: Sequence[str] | None = None,
) -> pathlib.Path:
    """Plot simulated monthly query shares for random keywords.

    Query probabilities are proxied by each month's observed document
    frequencies and sampled with a fixed query count. This is an explicit
    simulation, not a claim that processed documents contain client query logs.
    """
    by_month, available = _month_corpus(docs)
    selected_months = _select_months(available, months, 2)
    if keyword_count < 1 or queries_per_month < 1:
        raise ValueError("keyword_count and queries_per_month must be positive.")

    vocabulary = sorted({
        str(keyword)
        for month in selected_months
        for doc in by_month[month]
        for keyword in doc.keywords
    })
    if len(vocabulary) < keyword_count:
        raise ValueError(
            f"Query-distribution plot needs {keyword_count} distinct keywords; "
            f"the selected months contain only {len(vocabulary)}."
        )

    rng = np.random.default_rng(seed)
    keywords = sorted(str(word) for word in rng.choice(vocabulary, keyword_count, replace=False))
    query_shares: dict[str, list[float]] = {keyword: [] for keyword in keywords}
    for month in selected_months:
        month_docs = by_month[month]
        month_counts = Counter(
            str(keyword)
            for doc in month_docs
            for keyword in doc.keywords
        )
        all_keywords = sorted(month_counts)
        counts = np.array([month_counts[keyword] for keyword in all_keywords], dtype=float)
        if counts.sum() == 0:
            raise ValueError(f"Month {month} contains no indexed keywords for the query simulation.")
        probabilities = counts / counts.sum()
        simulated = rng.multinomial(queries_per_month, probabilities)
        frequency = dict(zip(all_keywords, simulated))
        for keyword in keywords:
            query_shares[keyword].append(float(frequency.get(keyword, 0)) / queries_per_month)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = pathlib.Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(keyword_count)
    width = 0.8 / len(selected_months)
    for i, month in enumerate(selected_months):
        ax.bar(
            x + (i - (len(selected_months) - 1) / 2) * width,
            [query_shares[keyword][i] for keyword in keywords],
            width=width,
            label=month,
        )
    ax.set_xticks(x, keywords, rotation=35, ha="right")
    ax.set_xlabel("Randomly selected keyword")
    ax.set_ylabel("Simulated query share")
    ax.set_title(f"Monthly query distribution for {keyword_count} random keywords (document-frequency proxy)")
    ax.legend(title="Observed document month")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_monthly_response_growth(
    docs: Iterable[Any],
    output_path: str | pathlib.Path = "results/figures/response_length_growth.png",
    *,
    seed: int = 42,
    keyword_count: int = 50,
    months: Sequence[str] | None = None,
) -> pathlib.Path:
    """Plot cumulative response lengths for random keywords at six real month snapshots."""
    by_month, available = _month_corpus(docs)
    selected_months = _select_months(available, months, 6)
    if keyword_count < 1:
        raise ValueError("keyword_count must be positive.")
    eligible_counts: Counter[str] = Counter(
        str(keyword)
        for month in available
        if month <= selected_months[-1]
        for doc in by_month[month]
        for keyword in doc.keywords
    )
    vocabulary = sorted(eligible_counts)
    if len(vocabulary) < keyword_count:
        raise ValueError(
            f"Response-growth plot needs {keyword_count} distinct keywords; "
            f"the dated corpus contains only {len(vocabulary)}."
        )

    rng = np.random.default_rng(seed)
    keywords = sorted(str(word) for word in rng.choice(vocabulary, keyword_count, replace=False))
    response_lengths: dict[str, list[int]] = {keyword: [] for keyword in keywords}
    cumulative_counts: Counter[str] = Counter()
    snapshots: dict[str, Counter[str]] = {}
    selected_set = set(selected_months)
    for month in available:
        if month > selected_months[-1]:
            break
        cumulative_counts.update(
            str(keyword)
            for doc in by_month[month]
            for keyword in doc.keywords
        )
        if month in selected_set:
            snapshots[month] = cumulative_counts.copy()
    for month in selected_months:
        for keyword in keywords:
            response_lengths[keyword].append(snapshots[month][keyword])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = pathlib.Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for keyword in keywords:
        ax.plot(selected_months, response_lengths[keyword], marker="o", linewidth=1, label=keyword)
    ax.set_xlabel("Document month")
    ax.set_ylabel("Cumulative response length (documents)")
    ax.set_title(f"Response-length growth for {keyword_count} random keywords")
    ax.legend(title="Keyword", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def _accuracy_column(headers: Sequence[str]) -> str:
    for candidate in ("accuracy", "paper_accuracy", "mean_accuracy"):
        if candidate in headers:
            return candidate
    raise ValueError("CSV must contain an 'accuracy', 'paper_accuracy', or 'mean_accuracy' column.")


def build_paper_comparison_table(
    paper_results_csv: str | pathlib.Path,
    ours_results_csv: str | pathlib.Path | None = None,
    *,
    markdown_path: str | pathlib.Path = "results/figures/paper_comparison.md",
    csv_path: str | pathlib.Path = "results/figures/paper_comparison.csv",
) -> list[dict[str, Any]]:
    """Compare user-provided paper accuracies against ours and write Markdown/CSV.

    Rows are matched on shared experiment dimensions. Repeated runs/seeds are
    averaged per matching configuration; ours also reports sample standard
    deviation across seed means.
    """
    paper_path = pathlib.Path(paper_results_csv)
    if not paper_path.is_file():
        raise FileNotFoundError(f"Paper results CSV not found: '{paper_path}'.")
    ours_path = _find_results_csv(ours_results_csv)
    paper_headers, paper_rows = _read_rows(paper_path)
    ours_headers, ours_rows = _read_rows(ours_path)
    paper_accuracy = _accuracy_column(paper_headers)
    ours_accuracy = _accuracy_column(ours_headers)
    dimensions = [column for column in _IDENTITY + _PARAMETERS if column in paper_headers and column in ours_headers]
    if not dimensions:
        raise ValueError("Paper and ours CSVs must share at least one experiment-dimension column.")

    def aggregate(rows: list[dict[str, str]], accuracy_key: str, source: pathlib.Path) -> dict[tuple[str, ...], list[float]]:
        grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row_number, row in enumerate(rows, start=2):
            try:
                value = float(row[accuracy_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid accuracy in '{source}' row {row_number}: {row[accuracy_key]!r}."
                ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Accuracy in '{source}' row {row_number} must be between 0 and 1.")
            key = tuple(_normalise(row.get(column, ""), column) for column in dimensions)
            seed = _normalise(row.get("seed", str(row_number)), "seed")
            grouped[key][seed].append(value)
        return {
            key: [float(np.mean(values)) for values in by_seed.values()]
            for key, by_seed in grouped.items()
        }

    paper_values = aggregate(paper_rows, paper_accuracy, paper_path)
    ours_values = aggregate(ours_rows, ours_accuracy, ours_path)
    common = sorted(set(paper_values) & set(ours_values))
    if not common:
        raise ValueError("No matching experiment configurations were found between paper and ours CSVs.")

    output_rows: list[dict[str, Any]] = []
    for key in common:
        paper_mean = float(np.mean(paper_values[key]))
        ours_mean = float(np.mean(ours_values[key]))
        ours_std = float(np.std(ours_values[key], ddof=1)) if len(ours_values[key]) > 1 else 0.0
        output_rows.append({
            **dict(zip(dimensions, key)),
            "paper_accuracy": paper_mean,
            "ours_mean_accuracy": ours_mean,
            "ours_std_accuracy": ours_std,
            "accuracy_delta": ours_mean - paper_mean,
        })

    csv_target = pathlib.Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = dimensions + [
        "paper_accuracy", "ours_mean_accuracy", "ours_std_accuracy", "accuracy_delta",
    ]
    with csv_target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    md_target = pathlib.Path(markdown_path)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    headings = dimensions + ["Paper accuracy", "Ours mean accuracy", "Ours std. dev.", "Difference (ours - paper)"]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for row in output_rows:
        values = [str(row[column]).replace("|", "\\|") for column in dimensions]
        values.extend(f"{row[column]:.4f}" for column in (
            "paper_accuracy", "ours_mean_accuracy", "ours_std_accuracy", "accuracy_delta",
        ))
        lines.append("| " + " | ".join(values) + " |")
    md_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_rows


def _load_documents(dataset: str, seed: int) -> list[Any]:
    from run_experiments import load_dataset

    docs, _ = load_dataset(dataset, seed=seed)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment results and dated corpus trends.")
    commands = parser.add_subparsers(dest="command", required=True)

    results = commands.add_parser("results", help="Plot accuracy sweeps from an experiment results CSV.")
    results.add_argument("--csv-path", default=None, help="CSV path; defaults to results/results.csv, then results/csv/results.csv.")
    results.add_argument("--output-dir", default="results/figures")

    query = commands.add_parser("query-distribution", help="Plot two months of simulated query shares from dated documents.")
    query.add_argument("--dataset", required=True, help="Processed dataset name or pickle path.")
    query.add_argument("--output", default="results/figures/query_distribution.png")
    query.add_argument("--seed", type=int, default=42)
    query.add_argument("--keyword-count", type=int, default=10)
    query.add_argument("--queries-per-month", type=int, default=1000)
    query.add_argument("--months", nargs=2, metavar=("YYYY-MM", "YYYY-MM"))

    growth = commands.add_parser("response-growth", help="Plot cumulative response lengths across six observed months.")
    growth.add_argument("--dataset", required=True, help="Processed dataset name or pickle path.")
    growth.add_argument("--output", default="results/figures/response_length_growth.png")
    growth.add_argument("--seed", type=int, default=42)
    growth.add_argument("--keyword-count", type=int, default=50)
    growth.add_argument("--months", nargs=6, metavar=("M1", "M2", "M3", "M4", "M5", "M6"))

    compare = commands.add_parser("compare", help="Write paper-versus-ours Markdown and CSV tables.")
    compare.add_argument("--paper-results-csv", required=True)
    compare.add_argument("--ours-results-csv", default=None)
    compare.add_argument("--markdown-output", default="results/figures/paper_comparison.md")
    compare.add_argument("--csv-output", default="results/figures/paper_comparison.csv")

    args = parser.parse_args()
    if args.command == "results":
        paths = plot_experiment_results(args.csv_path, args.output_dir)
        print(f"Saved {len(paths)} plot(s) to '{args.output_dir}'.")
    elif args.command == "query-distribution":
        path = plot_monthly_query_distribution(
            _load_documents(args.dataset, args.seed),
            args.output,
            seed=args.seed,
            keyword_count=args.keyword_count,
            queries_per_month=args.queries_per_month,
            months=args.months,
        )
        print(f"Saved '{path}'.")
    elif args.command == "response-growth":
        path = plot_monthly_response_growth(
            _load_documents(args.dataset, args.seed),
            args.output,
            seed=args.seed,
            keyword_count=args.keyword_count,
            months=args.months,
        )
        print(f"Saved '{path}'.")
    else:
        rows = build_paper_comparison_table(
            args.paper_results_csv,
            args.ours_results_csv,
            markdown_path=args.markdown_output,
            csv_path=args.csv_output,
        )
        print(f"Wrote {len(rows)} comparison row(s) to '{args.markdown_output}' and '{args.csv_output}'.")


if __name__ == "__main__":
    main()
