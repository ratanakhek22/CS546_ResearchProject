"""
Reads all experiment result JSONs from a results/{dataset}/{run}/ directory,
averages across trials, and generates a comparison HTML report.

Usage:
    python visualize.py --dataset scifact
    python visualize.py --dataset scifact --run 2026-04-24_14-32-01
"""
import os
import json
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

COLORS = {
    "random":        "#4A90D9",
    "kmeans":        "#E8744A",
    "hilbert_curve": "#5BBF8A",
}
DEFAULT_COLORS = ["#4A90D9", "#E8744A", "#5BBF8A", "#A87FD4", "#F0C040"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
TOP_K = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--run-id", type=str, default=None, help="Timestamp dir name. Defaults to latest run.")
    return parser.parse_args()

def resolve_run_dir(dataset, run):
    base = os.path.join(RESULTS_DIR, dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No results directory found for dataset '{dataset}' at {base}")
    if run:
        path = os.path.join(base, run)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run directory not found: {path}")
        return path
    runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return os.path.join(base, runs[-1])

def load_and_average(run_dir):
    """
    Loads all trial JSONs from run_dir, groups by strategy, and averages
    numeric metrics across trials. Returns dict of {strategy: averaged_metrics}.
    """
    trial_files = [f for f in os.listdir(run_dir) if f.endswith(".json")]
    if not trial_files:
        raise FileNotFoundError(f"No JSON files found in {run_dir}")

    grouped = {}
    for fname in trial_files:
        if fname.startswith("eval_"):
            continue
        with open(os.path.join(run_dir, fname)) as f:
            data = json.load(f)
        strategy = data.get("strategy", fname.split("_")[0])
        grouped.setdefault(strategy, []).append(data)

    averaged = {}
    for strategy, trials in grouped.items():
        avg = {"strategy": strategy, "num_trials": len(trials)}
        numeric_keys = [
            "build_time_seconds",
            "preprocess_time_seconds",
            "total_time_seconds",
            "avg_retrieval_ms",
        ]
        for key in numeric_keys:
            vals = [t[key] for t in trials if key in t]
            if vals:
                avg[key]          = round(float(np.mean(vals)), 4)
                avg[key + "_std"] = round(float(np.std(vals)),  4)

        # load eval file for this strategy if it exists
        eval_path = os.path.join(run_dir, f"eval_{strategy}.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                eval_data = json.load(f)
            for key in [f"recall@{TOP_K}", f"ndcg@{TOP_K}"]:
                if key in eval_data:
                    avg[key] = eval_data[key]
        averaged[strategy] = avg

    return averaged

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def color_for(strategy, i):
    return COLORS.get(strategy, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

def plot_bar(strategies, values, stds, title, ylabel, colors):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    x = np.arange(len(strategies))
    ax.bar(x, values, color=colors, width=0.5, zorder=3,
           yerr=stds if any(s > 0 for s in stds) else None,
           error_kw={"ecolor": "#ffffff55", "capsize": 4, "capthick": 1.5})

    for xi, val in zip(x, values):
        ax.text(xi, val + (max(values) * 0.04 if max(values) > 0 else 0.01),
                f"{val:.4f}", ha="center", va="bottom",
                color="#e0e0e0", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in strategies],
                       color="#e0e0e0", fontsize=11)
    ax.set_ylabel(ylabel, color="#a0a0a0", fontsize=10)
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#a0a0a0")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#2a2a2a", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig

def plot_grouped_bar(strategies, groups, stds, title, ylabel, colors):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    group_labels = list(groups.keys())
    n_strats = len(strategies)
    width = 0.6 / n_strats
    x = np.arange(len(group_labels))

    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offsets = x + (i - n_strats / 2 + 0.5) * width
        vals = [groups[g][i] for g in group_labels]
        errs = [stds[g][i]   for g in group_labels]
        ax.bar(offsets, vals, width=width * 0.9, color=color,
               label=strategy.replace("_", " ").title(), zorder=3,
               yerr=errs if any(e > 0 for e in errs) else None,
               error_kw={"ecolor": "#ffffff55", "capsize": 3})

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, color="#e0e0e0", fontsize=11)
    ax.set_ylabel(ylabel, color="#a0a0a0", fontsize=10)
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#a0a0a0")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#2a2a2a", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.legend(facecolor="#1a1a1a", labelcolor="#e0e0e0", edgecolor="#333333", fontsize=9)
    fig.tight_layout()
    return fig

def generate_report(all_results, dataset, run_id, output_file):
    strategies = list(all_results.keys())
    colors = [color_for(s, i) for i, s in enumerate(strategies)]

    def get_vals(key):
        return [all_results[s].get(key, 0) for s in strategies]

    def get_stds(key):
        return [all_results[s].get(key + "_std", 0) for s in strategies]

    # build_time metric
    fig_build = plot_bar(strategies, get_vals("build_time_seconds"),
                         get_stds("build_time_seconds"),
                         "HNSW build time", "seconds", colors)
    b64_build = fig_to_base64(fig_build); plt.close(fig_build)

    # retrieval metric
    fig_ret = plot_bar(strategies, get_vals("avg_retrieval_ms"),
                       get_stds("avg_retrieval_ms"),
                       "Average retrieval time per query", "milliseconds", colors)
    b64_ret = fig_to_base64(fig_ret); plt.close(fig_ret)

    # recall@k metric
    has_recall = any(f"recall@{TOP_K}" in all_results[s] for s in strategies)
    b64_recall = None
    if has_recall:
        fig_recall = plot_bar(strategies,
                            [all_results[s].get(f"recall@{TOP_K}", 0) for s in strategies],
                            [0] * len(strategies),
                            f"Recall@{TOP_K}", "score", colors)
        b64_recall = fig_to_base64(fig_recall); plt.close(fig_recall)

    # ndcg@k metric
    has_ndcg = any(f"ndcg@{TOP_K}" in all_results[s] for s in strategies)
    b64_ndcg = None
    if has_ndcg:
        fig_ndcg = plot_bar(strategies,
                            [all_results[s].get(f"ndcg@{TOP_K}", 0) for s in strategies],
                            [0] * len(strategies),
                            f"NDCG@{TOP_K}", "score", colors)
        b64_ndcg = fig_to_base64(fig_ndcg); plt.close(fig_ndcg)

    # preprocess metric
    has_preprocess = any("preprocess_time_seconds" in all_results[s] for s in strategies)
    b64_pre = None
    if has_preprocess:
        fig_pre = plot_bar(strategies, get_vals("preprocess_time_seconds"),
                           get_stds("preprocess_time_seconds"),
                           "Preprocessing time (0 = none)", "seconds", colors)
        b64_pre = fig_to_base64(fig_pre); plt.close(fig_pre)

    # total_time metric
    has_total = any("total_time_seconds" in all_results[s] for s in strategies)
    b64_total = None
    if has_total:
        groups = {
            "Build":      get_vals("build_time_seconds"),
            "Preprocess": get_vals("preprocess_time_seconds"),
            "Total":      [all_results[s].get("total_time_seconds",
                           all_results[s].get("build_time_seconds", 0))
                           for s in strategies],
        }
        group_stds = {
            "Build":      get_stds("build_time_seconds"),
            "Preprocess": get_stds("preprocess_time_seconds"),
            "Total":      get_stds("total_time_seconds"),
        }
        fig_total = plot_grouped_bar(strategies, groups, group_stds,
                                     "Time breakdown by strategy", "seconds", colors)
        b64_total = fig_to_base64(fig_total); plt.close(fig_total)

    table_rows = ""
    for s, color in zip(strategies, colors):
        d = all_results[s]
        def fmt(key, unit="s"):
            v   = d.get(key)
            std = d.get(key + "_std")
            if v is None:
                return "—"
            base = f"{v:.4f}{unit}"
            if std and std > 0:
                base += f" <span style='color:#555'>±{std:.4f}</span>"
            return base
        table_rows += f"""
        <tr>
            <td><span class="dot" style="background:{color}"></span>{s.replace('_',' ').title()}</td>
            <td>{d.get('num_trials', 1)}</td>
            <td>{fmt('build_time_seconds')}</td>
            <td>{fmt(f"recall@{TOP_K}", "")}</td>
            <td>{fmt(f"ndcg@{TOP_K}", "")}</td>
            <td>{fmt('preprocess_time_seconds')}</td>
            <td>{fmt('total_time_seconds')}</td>
            <td>{fmt('avg_retrieval_ms', 'ms')}</td>
        </tr>"""

    # HTML Metric Sections
    recall_section = f"""
    <div class="card">
    <h2>Recall@{TOP_K}</h2>
    <img src="data:image/png;base64,{b64_recall}" />
    </div>""" if b64_recall else ""

    ndcg_section = f"""
    <div class="card">
    <h2>NDCG@{TOP_K}</h2>
    <img src="data:image/png;base64,{b64_ndcg}" />
    </div>""" if b64_ndcg else ""

    preprocess_section = f"""
    <div class="card">
        <h2>Preprocessing time</h2>
        <img src="data:image/png;base64,{b64_pre}" />
    </div>""" if b64_pre else ""

    total_section = f"""
    <div class="card">
        <h2>Time breakdown</h2>
        <img src="data:image/png;base64,{b64_total}" />
    </div>""" if b64_total else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HNSW Report — {dataset}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #0a0a0a; color: #d0d0d0;
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 15px; line-height: 1.6; padding: 0 0 4rem;
        }}
        header {{
            border-bottom: 1px solid #1e1e1e;
            padding: 2.5rem 3rem 2rem;
            display: flex; flex-direction: column; gap: 6px;
        }}
        header h1 {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.5rem; font-weight: 600;
            color: #ffffff; letter-spacing: -0.02em;
        }}
        header p {{ font-size: 13px; color: #555; font-family: 'IBM Plex Mono', monospace; }}
        .strategies {{
            display: flex; gap: 10px;
            padding: 1.5rem 3rem; border-bottom: 1px solid #1a1a1a;
        }}
        .badge {{
            font-family: 'IBM Plex Mono', monospace; font-size: 11px;
            padding: 4px 12px; border-radius: 2px; font-weight: 600; letter-spacing: 0.04em;
        }}
        .grid {{
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 1px; background: #1a1a1a; margin: 1px;
        }}
        .card {{ background: #0a0a0a; padding: 2rem 2.5rem; }}
        .card.full {{ grid-column: 1 / -1; }}
        .card h2 {{
            font-family: 'IBM Plex Mono', monospace; font-size: 11px;
            font-weight: 600; color: #555; letter-spacing: 0.1em;
            text-transform: uppercase; margin-bottom: 1.5rem;
        }}
        .card img {{ width: 100%; border-radius: 2px; }}
        table {{ width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
        th {{
            text-align: left; color: #444; font-size: 11px;
            letter-spacing: 0.08em; text-transform: uppercase;
            padding: 0 1rem 1rem 0; border-bottom: 1px solid #1e1e1e;
        }}
        td {{
            padding: 0.85rem 1rem 0.85rem 0;
            border-bottom: 1px solid #141414;
            color: #c0c0c0; vertical-align: middle;
        }}
        td:first-child {{ color: #ffffff; font-weight: 600; }}
        .dot {{
            display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; margin-right: 8px; vertical-align: middle;
        }}
    </style>
</head>
<body>
    <header>
        <h1>HNSW Insertion Strategy — {dataset.title()} Report</h1>
        <p>CS546 Research Project · Run: {run_id} · {len(strategies)} strategies compared</p>
    </header>
    <div class="strategies">
    {''.join(f'<span class="badge" style="background:{color_for(s,i)}22;color:{color_for(s,i)}">{s.replace("_"," ").upper()}</span>' for i,s in enumerate(strategies))}
    </div>
    <div class="grid">
        <div class="card">
            <h2>Build time</h2>
            <img src="data:image/png;base64,{b64_build}" />
        </div>
        <div class="card">
            <h2>Avg retrieval time per query</h2>
            <img src="data:image/png;base64,{b64_ret}" />
        </div>
        {preprocess_section}
        {total_section}
        {recall_section}
        {ndcg_section}
        <div class="card full">
            <h2>Summary — averaged across trials</h2>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Trials</th>
                        <th>Build time</th>
                        <th>Recall@{TOP_K}</th>
                        <th>NDCG@{TOP_K}</th>
                        <th>Preprocess time</th>
                        <th>Total time</th>
                        <th>Avg retrieval</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(html)
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    run_dir = resolve_run_dir(args.dataset, args.run_id)
    run_id = os.path.basename(run_dir)
    print(f"Loading results from: {run_dir}")

    all_results = load_and_average(run_dir)
    print(f"Strategies found: {list(all_results.keys())}")

    output_file = os.path.join(SCRIPT_DIR, "reports", f"{args.dataset}_{run_id}.html")
    generate_report(all_results, args.dataset, run_id, output_file)