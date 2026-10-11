"""Streamlit UI for running and inspecting the leakage-abuse attacks.

Launch from the repository root:

    streamlit run app.py

Every run goes through :func:`run_experiments.run_pipeline`, the same code path
used by the experiment grids, so what the UI shows is what the CLI computes.
"""

from __future__ import annotations

import collections
import pathlib

import pandas as pd
import streamlit as st

import leakage
from equality import equality_quality
from run_experiments import load_dataset, run_pipeline
import evaluate

st.set_page_config(page_title="DPS · FP/BP-DSSE attacks", layout="wide")

PROCESSED_DIR = pathlib.Path("datasets/processed")
RESULTS_CSVS = [pathlib.Path("results/csv/results.csv"), pathlib.Path("results/results.csv")]

ATTACKS = {
    "fma": "FMA · Frequency Matching (§5): query distribution per interval",
    "via": "VIA · Volumetric Inference (§6.1): all inserted files known",
    "pvia": "PVIA · Partial VIA (§6.2): a fraction α of inserted files known",
    "lvia-k": "LVIA-K (§6.2): exact keyword distribution",
    "lvia-s": "LVIA-S (§6.2): keyword distribution sampled with fraction η",
    "no_update": "No update knowledge (§6.3): snapshot before μ₀",
    "no_update_pvia": "No update knowledge + partial snapshot (§6.3)",
    "no_update_lvia": "No update knowledge + snapshot counts (§6.3)",
}
EQUALITY = {
    "T1": "T1 · token relations leak query equality",
    "T2": "T2 · rebuild equality from response similarity (rsp ≥ δ)",
}


def available_datasets() -> list[str]:
    names = ["synthetic", "obvious", "trends"]
    if PROCESSED_DIR.exists():
        names += sorted(p.stem for p in PROCESSED_DIR.glob("*.pkl"))
    return names


@st.cache_data(show_spinner=False)
def cached_dataset(name: str):
    return load_dataset(name, seed=42)


# ---------------------------------------------------------------------------
# Sidebar: experiment configuration
# ---------------------------------------------------------------------------

st.sidebar.title("Experiment")
dataset = st.sidebar.selectbox("Dataset", available_datasets(), key="dataset")
attack = st.sidebar.selectbox("Attack", list(ATTACKS), format_func=lambda a: ATTACKS[a], key="attack")
equality_mode = st.sidebar.radio("Query equality", list(EQUALITY), format_func=lambda e: EQUALITY[e], key="equality")
delta = st.sidebar.slider("δ (response-similarity threshold)", 0.1, 1.0, 0.6, 0.05, key="delta",
                          disabled=equality_mode != "T2")

st.sidebar.subheader("Dynamics")
tau = st.sidebar.slider("τ · time intervals", 1, 24, 10, key="tau")
lam = st.sidebar.number_input("λ · expected total queries", min_value=0, value=500, step=100, key="lam",
                              help="Paper uses 120K–480K for FMA; large values on real datasets take a while.")
gamma = st.sidebar.slider("γ · deletion rate", 0.0, 0.5, 0.1, 0.05, key="gamma")

st.sidebar.subheader("Attacker knowledge")
alpha = st.sidebar.slider("α · known-file fraction", 0.05, 0.95, 0.8, 0.05, key="alpha",
                          disabled=attack not in ("pvia", "no_update_pvia"))
eta = st.sidebar.slider("η · sampled fraction for LVIA-S", 0.05, 1.0, 0.5, 0.05, key="eta",
                        disabled=attack != "lvia-s")
noise = st.sidebar.slider("Noise U(0, n) on query distribution", 0, 10, 0, key="noise",
                          disabled=attack != "fma")
tolerance = st.sidebar.number_input("FMA candidate tolerance", min_value=0.0, value=0.0, step=0.5, key="tol",
                                    disabled=attack != "fma",
                                    help="Slack added to the best frequency-match score; use > 0 with noise.")

st.sidebar.subheader("Run")
seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1, key="seed")
log_csv = st.sidebar.checkbox("Append result to results/csv/results.csv", value=False, key="log")
run_clicked = st.sidebar.button("Run attack", type="primary", width="stretch")

if run_clicked:
    cfg = {
        "dataset": dataset, "attack": attack, "equality": equality_mode, "tau": tau,
        "lambda": float(lam), "gamma": gamma, "alpha": alpha, "eta": eta, "noise": float(noise),
        "seed": int(seed), "delta": delta, "tolerance": tolerance or None,
    }
    with st.spinner(f"Simulating {tau} intervals and running {attack.upper()}…"):
        try:
            out = run_pipeline(cfg)
        except Exception as exc:  # surface config/data errors in the UI
            st.sidebar.error(f"Run failed: {exc}")
            out = None
    if out is not None:
        out["config"] = cfg
        st.session_state["run"] = out
        if log_csv:
            RESULTS_CSVS[0].parent.mkdir(parents=True, exist_ok=True)
            evaluate.log_result(out["row"], RESULTS_CSVS[0])
            st.sidebar.success(f"Logged to {RESULTS_CSVS[0]}")


# ---------------------------------------------------------------------------
# Helpers for the run views
# ---------------------------------------------------------------------------

def records_frame(run: dict) -> pd.DataFrame:
    gt, preds, eq = run["ground_truth"], run["predictions"], run["equality"]
    rows = []
    for r in run["records"]:
        a, d = leakage.derive_insert_delete(r.ulen, r.rlen) if (r.ulen - r.rlen) % 2 == 0 else (None, None)
        top_vols = ", ".join(f"{v}×{c}" for v, c in sorted(r.fvol.items())[:6])
        rows.append({
            "query_id": r.query_id,
            "interval": r.interval,
            "probe": r.is_probe,
            "token": str(r.token)[:28],
            "ulen (u)": r.ulen,
            "rlen (n)": r.rlen,
            "ilen a=(u+n)/2": a,
            "dlen d=(u−n)/2": d,
            "fvol (volume×count)": top_vols,
            "group": eq.group_of.get(r.query_id),
            "true keyword": gt.get(r.query_id),
            "predicted": preds.get(r.query_id),
            "correct": preds.get(r.query_id) is not None and preds.get(r.query_id) == gt.get(r.query_id),
        })
    return pd.DataFrame(rows)


def show_run(run: dict) -> None:
    row, cfg = run["row"], run["config"]
    df = records_frame(run)
    sampled = df[~df["probe"]]

    st.subheader(f"{row['attack'].upper()} on {row['dataset']} · {row['equality']}")
    st.caption(
        f"τ={cfg['tau']} · λ={int(cfg['lambda'])} · γ={cfg['gamma']} · α={cfg['alpha']} · "
        f"η={cfg['eta']} · noise={cfg['noise']} · seed={cfg['seed']}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recovery rate", f"{row['accuracy'] * 100:.1f}%")
    c2.metric("Correct / client queries", f"{row['correct']} / {row['total']}")
    c3.metric("Keywords in DB", row["keyword_count"])
    c4.metric("Equality groups", len(run["equality"].groups))
    c5.metric("Runtime", f"{row['runtime_s']:.2f}s")

    n = len(run["records"])
    if n <= 3000:
        prec, rec = equality_quality(run["equality"], run["ground_truth"])
        st.caption(f"Query-equality inference: pairwise precision {prec:.3f}, recall {rec:.3f} over {n} queries "
                   f"(probe queries included).")
    else:
        st.caption(f"Query-equality precision/recall skipped for {n} queries (pairwise check is O(n²)).")

    tab_int, tab_rec, tab_grp, tab_kw = st.tabs(
        ["Per interval", "Leakage records", "Equality groups", "Per keyword"])

    with tab_int:
        if sampled.empty:
            st.info("No client queries were sampled (λ = 0). Raise λ to see per-interval recovery.")
        else:
            per_int = sampled.groupby("interval").agg(queries=("query_id", "size"), recovered=("correct", "mean"))
            per_int["recovered"] *= 100
            st.markdown("**Recovery rate per interval (%)**")
            st.bar_chart(per_int["recovered"])
            st.dataframe(per_int.rename(columns={"recovered": "recovered %"}), width="stretch")

    with tab_rec:
        st.markdown("What the server observes for each search token, plus the volumes the attacker derives "
                    "with Eq. 2. The last three columns are ground truth and the attack's guess.")
        f1, f2, f3 = st.columns(3)
        show_probes = f1.checkbox("Include probe queries", value=False,
                                  help="One query per keyword per interval, used to observe update lengths (§7.3).")
        only_wrong = f2.checkbox("Only unrecovered", value=False)
        kw_filter = f3.text_input("Filter by true keyword")
        view = df if show_probes else sampled
        if only_wrong:
            view = view[~view["correct"]]
        if kw_filter:
            view = view[view["true keyword"].astype(str).str.contains(kw_filter, case=False)]
        st.dataframe(view.head(5000), width="stretch", hide_index=True)
        st.caption(f"Showing {min(len(view), 5000)} of {len(view)} rows.")
        st.download_button("Download records as CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name=f"records_{row['attack']}_{row['dataset']}.csv", mime="text/csv")

    with tab_grp:
        gt = run["ground_truth"]
        grp_rows = []
        for gid, qids in run["equality"].groups.items():
            kws = collections.Counter(gt.get(q) for q in qids)
            top_kw, top_n = kws.most_common(1)[0]
            grp_rows.append({"group": gid, "size": len(qids), "distinct true keywords": len(kws),
                             "majority keyword": top_kw, "purity": round(top_n / len(qids), 3),
                             "predicted": run["predictions"].get(qids[0]) if qids else None})
        gdf = pd.DataFrame(grp_rows).sort_values("size", ascending=False)
        pure = (gdf["distinct true keywords"] == 1).mean() * 100 if len(gdf) else 0
        st.markdown(f"**{len(gdf)} groups**, {pure:.1f}% contain a single true keyword. "
                    "T1 groups are exact by construction; T2 groups can merge or split keywords.")
        st.dataframe(gdf.head(2000), width="stretch", hide_index=True)

    with tab_kw:
        if sampled.empty:
            st.info("No client queries to summarise.")
        else:
            per_kw = (sampled.groupby("true keyword")
                      .agg(queries=("query_id", "size"), recovered=("correct", "mean"))
                      .sort_values("queries", ascending=False))
            per_kw["recovered"] = (per_kw["recovered"] * 100).round(1)
            st.markdown(f"**{(per_kw['recovered'] == 100).sum()}** of {len(per_kw)} queried keywords fully recovered.")
            st.dataframe(per_kw.rename(columns={"recovered": "recovered %"}), width="stretch")


def show_leakage_explorer(run: dict) -> None:
    st.markdown(
        "Pick a keyword to follow its leakage across intervals. The scheme reveals **u** (update length) and "
        "**n** (response length); the attacker derives insertions **a** and deletions **d** with Eq. 2. "
        "The right-hand columns recompute a and d directly from the update log to show the derivation is exact."
    )
    probes = [r for r in run["records"] if r.is_probe]
    if not probes:
        st.info("This run has no probe queries (λ = 0 skips querying). Run again with λ > 0.")
        return
    gt = run["ground_truth"]
    kws = sorted({gt[r.query_id] for r in probes})
    kw = st.selectbox("Keyword", kws, key="leak_kw")
    rows = []
    for r in sorted((p for p in probes if gt[p.query_id] == kw), key=lambda p: p.interval):
        a, d = leakage.derive_insert_delete(r.ulen, r.rlen)
        rows.append({
            "interval": r.interval, "u (ulen)": r.ulen, "n (rlen)": r.rlen,
            "a derived": a, "d derived": d,
            "a from log": leakage.ilen(run["events"], kw, r.timestamp),
            "d from log": leakage.dlen(run["events"], kw, r.timestamp),
            "files returned": sum(r.fvol.values()),
        })
    ldf = pd.DataFrame(rows).set_index("interval")
    st.line_chart(ldf[["u (ulen)", "n (rlen)", "a derived", "d derived"]])
    match = (ldf["a derived"] == ldf["a from log"]).all() and (ldf["d derived"] == ldf["d from log"]).all()
    (st.success if match else st.warning)(
        "Derived volumes match the update log in every interval." if match
        else "Derived volumes differ from the update log in some intervals.")
    st.dataframe(ldf, width="stretch")


def show_dataset(name: str) -> None:
    docs, vocab = cached_dataset(name)
    vols = [len(d.keywords) for d in docs]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", f"{len(docs):,}")
    c2.metric("Keywords", f"{len(vocab):,}")
    c3.metric("Avg keywords / doc", f"{sum(vols) / max(len(vols), 1):.1f}")
    ts = sorted(d.timestamp for d in docs)
    c4.metric("Date range", f"{ts[0]:%Y-%m-%d} → {ts[-1]:%Y-%m-%d}" if ts else "–")

    left, right = st.columns(2)
    with left:
        df_counts = collections.Counter(kw for d in docs for kw in d.keywords)
        top = pd.DataFrame(df_counts.most_common(25), columns=["keyword", "documents"]).set_index("keyword")
        st.markdown("**Top 25 keywords by document frequency**")
        st.bar_chart(top)
    with right:
        try:
            months = pd.Series([pd.Timestamp(t).strftime("%Y-%m") for t in ts])
            st.markdown("**Documents per month**")
            st.bar_chart(months.value_counts().sort_index())
        except Exception:
            st.caption("Timestamps cannot be grouped by month for this dataset.")
    st.markdown("**File volume distribution** (keywords per document — the `fvol` leakage unit)")
    st.bar_chart(pd.Series(vols).value_counts().sort_index())


def show_results() -> None:
    path = next((p for p in RESULTS_CSVS if p.exists()), None)
    if path is None:
        st.info("No results CSV yet. Run grids with "
                "`python run_experiments.py --grid grids/<name>.yaml --csv-path results/csv/results.csv`, "
                "or tick “Append result” in the sidebar.")
    else:
        res = pd.read_csv(path)
        st.markdown(f"**{len(res)} runs** in `{path}`")
        cols = st.columns(3)
        f_attack = cols[0].multiselect("Attack", sorted(res["attack"].unique()))
        f_ds = cols[1].multiselect("Dataset", sorted(res["dataset"].unique()))
        f_eq = cols[2].multiselect("Equality", sorted(res["equality"].unique()))
        view = res
        if f_attack:
            view = view[view["attack"].isin(f_attack)]
        if f_ds:
            view = view[view["dataset"].isin(f_ds)]
        if f_eq:
            view = view[view["equality"].isin(f_eq)]
        keys = ["dataset", "attack", "equality", "tau", "lambda", "gamma", "alpha", "eta", "noise"]
        agg = (view.groupby(keys)["accuracy"].agg(["mean", "std", "count"]).reset_index()
               .rename(columns={"mean": "mean accuracy", "std": "std", "count": "seeds"}))
        st.dataframe(agg, width="stretch", hide_index=True)

    comp = pathlib.Path("results/comparison_table.csv")
    if comp.exists():
        st.markdown("**Paper vs ours**")
        cdf = pd.read_csv(comp, encoding="utf-8")
        cdf = cdf.drop(columns=[c for c in ("our_result",) if c in cdf.columns])
        st.dataframe(cdf, width="stretch", hide_index=True)

    figs = sorted(pathlib.Path("results/figures").glob("*.png")) if pathlib.Path("results/figures").exists() else []
    if figs:
        st.markdown("**Figures** (from `python plots.py`)")
        cols = st.columns(2)
        for i, fig in enumerate(figs):
            cols[i % 2].image(str(fig), caption=fig.stem, width="stretch")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("Leakage-abuse attacks on forward & backward private DSSE")
st.caption("Re-implementation of Xu et al., CCS 2023. Configure a run in the sidebar.")

t_run, t_leak, t_data, t_res = st.tabs(["Attack run", "Leakage explorer", "Dataset", "Experiment results"])

with t_run:
    if "run" in st.session_state:
        show_run(st.session_state["run"])
    else:
        st.info("Choose a dataset, attack and parameters in the sidebar, then click **Run attack**. "
                "`synthetic` runs in a second; the Enron/Lucene sets take longer at large λ.")

with t_leak:
    if "run" in st.session_state:
        show_leakage_explorer(st.session_state["run"])
    else:
        st.info("Run an attack first; the explorer follows keywords through that run's update log.")

with t_data:
    show_dataset(dataset)

with t_res:
    show_results()
