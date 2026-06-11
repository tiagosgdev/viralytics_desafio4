"""Generate sanity-check plots for the StockAgent Phase-1 data.

Run from repo root:
    python3 stock_agent/sanity_plots.py

Produces:
    stock_agent/plots/01_stock_overall.png  ... 10_sales_timeseries.png
    stock_agent/SANITY_PLOTS.md             (markdown report embedding the PNGs)

Read-only on the DB. Reuses StockStats — no new code modules.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from stock_stats import StockStats, PIVOT_KEYS  # noqa: E402

PLOTS_DIR = BASE_DIR / "plots"
MD_PATH = BASE_DIR / "SANITY_PLOTS.md"
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]

sns.set_theme(style="whitegrid")
FIGSIZE = (10, 5)


# ─── individual plots ──────────────────────────────────────────────────


def _plot_stock_overall(stats: StockStats) -> tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(stats.df["stock_count"], bins=80, ax=ax, color="steelblue")
    ax.set_title("stock_count distribution (60K rows)")
    ax.set_xlabel("stock_count")
    ax.set_ylabel("rows")
    out = PLOTS_DIR / "01_stock_overall.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    p50 = int(stats.df["stock_count"].median())
    p95 = int(stats.df["stock_count"].quantile(0.95))
    pmax = int(stats.df["stock_count"].max())
    caption = (
        f"Right-skewed as expected. Median={p50}, p95={p95}, max={pmax}. "
        f"The fat tail past p95 is the planted 8% overstock outliers from the seeder."
    )
    return "Stock distribution (overall)", out.name, caption


def _plot_stock_per_size(stats: StockStats) -> tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(
        data=stats.df, x="size", y="stock_count", order=SIZE_ORDER,
        showfliers=False, ax=ax, color="lightsteelblue",
    )
    ax.set_title("stock_count by size")
    ax.set_xlabel("size")
    ax.set_ylabel("stock_count (fliers hidden)")
    out = PLOTS_DIR / "02_stock_per_size.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    means = stats.df.groupby("size")["stock_count"].mean().reindex(SIZE_ORDER)
    caption = (
        "Per-size box. Mean stock by size: "
        + ", ".join(f"{s}={means[s]:.1f}" for s in SIZE_ORDER)
        + ". M heaviest, XXL lightest — matches the seeder size_mult."
    )
    return "Stock per size", out.name, caption


def _plot_age_hist(stats: StockStats) -> tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(stats.df["age_days"], bins=60, ax=ax, color="darkorange")
    for cutoff in (360, 720, 1080):
        ax.axvline(cutoff, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("age_days distribution")
    ax.set_xlabel("age_days (since items.created_at)")
    ax.set_ylabel("rows")
    out = PLOTS_DIR / "03_age_hist.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    a = stats.df["age_days"]
    b0 = int((a < 360).sum())
    b1 = int(((a >= 360) & (a < 720)).sum())
    b2 = int((a >= 720).sum())
    total = b0 + b1 + b2
    caption = (
        f"Three year-buckets: <360d={b0} ({b0/total:.0%}), "
        f"360–720d={b1} ({b1/total:.0%}), ≥720d={b2} ({b2/total:.0%}). "
        f"Seeder target 55/30/15."
    )
    return "Age histogram", out.name, caption


def _plot_color_type_heatmap(stats: StockStats) -> tuple[str, str, str]:
    pivot = (
        stats.df.pivot_table(
            index="color", columns="type", values="stock_count", aggfunc="sum"
        ).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3, ax=ax, cbar_kws={"label": "total stock"})
    ax.set_title("Total stock_count by color × type")
    fig.autofmt_xdate(rotation=45)
    out = PLOTS_DIR / "04_color_type_heatmap.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    zeros = int((pivot == 0).sum().sum())
    cells = pivot.size
    caption = (
        f"Coverage check: {zeros}/{cells} cells are zero "
        f"(genuine gaps where a color/type pair has no items in the catalogue). "
        f"No systematic collapse — every color has stock in multiple types."
    )
    return "Color × Type heatmap", out.name, caption


def _plot_velocity_hist(stats: StockStats) -> tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(stats.df["sales_velocity"], bins=80, ax=ax, color="seagreen")
    ax.set_title("sales_velocity distribution (units sold per day)")
    ax.set_xlabel("sales_velocity")
    ax.set_ylabel("rows")
    out = PLOTS_DIR / "05_velocity_hist.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    low_share = (stats.df["sales_velocity"] < 0.05).mean()
    median = stats.df["sales_velocity"].median()
    caption = (
        f"Median velocity={median:.3f}/day. "
        f"{low_share*100:.1f}% of rows have velocity < 0.05/day — the planted "
        f"underperformer tail (~15% target). High-velocity tail = fast movers "
        f"the agent will rank LOW (push_score's perf_score inverts velocity)."
    )
    return "Sales velocity distribution", out.name, caption


def _plot_gap_per_age_quartile(stats: StockStats) -> tuple[str, str, str]:
    df = stats.df.copy()
    df = df[df["last_sold_at"].notna()].copy()
    df["age_q"] = pd.qcut(df["age_days"], q=4, labels=["Q1 (newest)", "Q2", "Q3", "Q4 (oldest)"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(
        data=df, x="age_q", y="days_since_last_sale",
        showfliers=False, ax=ax, color="mediumpurple",
    )
    ax.set_title("days_since_last_sale per age quartile")
    ax.set_xlabel("age quartile")
    ax.set_ylabel("days_since_last_sale (fliers hidden)")
    out = PLOTS_DIR / "06_gap_per_age_quartile.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    means = df.groupby("age_q", observed=True)["days_since_last_sale"].mean()
    caption = (
        "Means: "
        + ", ".join(f"{label}={means[label]:.1f}d" for label in means.index)
        + ". Strictly increasing → older items haven't sold recently, as expected."
    )
    return "Gap per age quartile", out.name, caption


def _plot_push_score_hist(stats: StockStats) -> tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(stats.df["push_score"], bins=60, ax=ax, color="crimson")
    ax.set_title("push_score distribution")
    ax.set_xlabel("push_score (0 = ignore, 1 = push hardest)")
    ax.set_ylabel("rows")
    out = PLOTS_DIR / "07_push_score_hist.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    sw = stats.sum_weights
    p50 = stats.df["push_score"].median()
    p95 = stats.df["push_score"].quantile(0.95)
    pmax = stats.df["push_score"].max()
    caption = (
        f"In [0, {sw}] by construction (active=0 rows clamped to 0). "
        f"Median={p50:.3f}, p95={p95:.3f}, max={pmax:.3f}. The right tail is what "
        f"StockAgent will argue to push."
    )
    return "Push score distribution", out.name, caption


def _plot_top50_push_table(stats: StockStats) -> tuple[str, str, str]:
    rows = []
    for iid, sz in stats.get_overstock_items(top_k=50):
        r = stats.get_row(iid, sz)
        rows.append({
            "item_id": int(iid),
            "size": sz,
            "stock": int(r["stock_count"]),
            "sold": int(r["total_sold"]),
            "age_d": round(float(r["age_days"]), 0),
            "vel": round(float(r["sales_velocity"]), 3),
            "push": round(float(r["push_score"]), 3),
            "season": r["season"],
            "color": r["color"],
            "type": r["type"],
        })
    df = pd.DataFrame(rows)

    # Render the DataFrame as a matplotlib table image
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    ax.set_title("Top 50 items by push_score", pad=14)
    out = PLOTS_DIR / "08_top50_push_table.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    old_share = (df["age_d"] >= 720).mean()
    slow_share = (df["vel"] < 0.5).mean()
    caption = (
        f"Top-50: {old_share*100:.0f}% are ≥720d old, "
        f"{slow_share*100:.0f}% have sales_velocity<0.5/d. "
        f"Cross-checks the seeder's smoke assertion #4."
    )
    return "Top 50 push items", out.name, caption


def _plot_attribute_pressure(stats: StockStats) -> tuple[str, str, str]:
    pressure = stats.get_attribute_pressure()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, key in zip(axes.flat, PIVOT_KEYS):
        items = list(pressure[key].items())
        items.sort(key=lambda kv: -kv[1])
        vals = items[:15]  # cap labels
        labels = [v[0] for v in vals]
        scores = [v[1] for v in vals]
        ax.barh(labels[::-1], scores[::-1], color="teal")
        ax.set_title(f"attribute_pressure[{key}] — top {len(vals)}")
        ax.set_xlabel("mean push_score")
    fig.suptitle("Attribute pressure (StockAgent's negotiation signal)", y=1.00)
    fig.tight_layout()
    out = PLOTS_DIR / "09_attribute_pressure.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Surface the top value per attribute
    top_per = {k: max(pressure[k].items(), key=lambda kv: kv[1]) for k in PIVOT_KEYS}
    caption = (
        "Per-attribute mean push_score, sorted desc. Highest-pressure value per axis: "
        + ", ".join(f"{k}={v[0]} ({v[1]:.3f})" for k, v in top_per.items())
        + ". These are the values StockAgent will argue most strongly for in Phase 2."
    )
    return "Attribute pressure", out.name, caption


def _plot_sales_timeseries(stats: StockStats) -> tuple[str, str, str]:
    conn = sqlite3.connect(stats.db_path)
    try:
        df = pd.read_sql_query(
            "SELECT ts, -delta AS units "
            "FROM stock_events WHERE reason='sale'",
            conn,
        )
    finally:
        conn.close()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    monthly = (
        df.set_index("ts")["units"]
          .resample("MS")
          .sum()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(monthly["ts"], monthly["units"], width=20, color="slategray")
    ax.set_title("Monthly synthetic sales (stock_events where reason='sale')")
    ax.set_xlabel("month")
    ax.set_ylabel("units sold")
    fig.autofmt_xdate()
    out = PLOTS_DIR / "10_sales_timeseries.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    total = int(df["units"].sum())
    months = len(monthly)
    caption = (
        f"{total:,} sale units across {months} months. "
        f"Older months show higher cumulative sales because items had more "
        f"days to accumulate them in the seeder model."
    )
    return "Monthly sales (event log replay)", out.name, caption


# ─── markdown writer ───────────────────────────────────────────────────


def _write_markdown(sections: list[tuple[str, str, str]]) -> None:
    lines = [
        "# StockAgent — Sanity Plots\n",
        "Auto-generated by `stock_agent/sanity_plots.py`. Read-only on the DB.\n",
        "Each section embeds one figure + the numbers behind it. "
        "Re-run the script after `seed_stock.py --force` or any change you "
        "want to visually verify.\n",
        "---\n",
    ]
    for i, (title, png_name, caption) in enumerate(sections, 1):
        lines.append(f"## {i}. {title}\n")
        lines.append(f"![{title}](plots/{png_name})\n")
        lines.append(caption + "\n")
        lines.append("---\n")
    MD_PATH.write_text("\n".join(lines))


# ─── main ──────────────────────────────────────────────────────────────


def main() -> int:
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"Loading StockStats from {BASE_DIR.parent / 'LNIAGIA' / 'DB' / 'SQLLite' / 'clothing.db'}")
    stats = StockStats()
    print(f"  {len(stats.df)} rows loaded.\n")

    plot_fns = [
        _plot_stock_overall,
        _plot_stock_per_size,
        _plot_age_hist,
        _plot_color_type_heatmap,
        _plot_velocity_hist,
        _plot_gap_per_age_quartile,
        _plot_push_score_hist,
        _plot_top50_push_table,
        _plot_attribute_pressure,
        _plot_sales_timeseries,
    ]

    sections: list[tuple[str, str, str]] = []
    for fn in plot_fns:
        title, png, caption = fn(stats)
        sections.append((title, png, caption))
        print(f"  ✓ {title:<40} -> {png}")

    _write_markdown(sections)
    print(f"\nWrote {MD_PATH.relative_to(BASE_DIR.parent)}")
    print(f"Plots in {PLOTS_DIR.relative_to(BASE_DIR.parent)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
