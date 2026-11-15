"""Matplotlib figures. Each function returns a Figure; nothing is shown or saved here."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from .config import BANK_NAMES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
# Fixed colour per bank, so a bank keeps its colour when others are filtered out.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BANK_COLOURS = dict(zip(BANK_NAMES, PALETTE))
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", [BLUE, "#eeeeec", RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.8, "legend.frameon": False,
})


def _years(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _end_labels(ax, last: pd.Series, colours: dict[str, str], fmt: str) -> None:
    """Label each line at its right end, spreading labels apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.048
    y = None
    for name, value in last.sort_values().items():
        y = value if y is None else max(value, y + gap)
        ax.annotate(
            f"{name}  {fmt.format(value)}", xy=(1, y), xycoords=("axes fraction", "data"),
            xytext=(14, 0), textcoords="offset points", va="center", fontsize=9, color=INK,
        )
        ax.plot([1.005], [y], marker="o", ms=5, color=colours[name], clip_on=False,
                transform=ax.get_yaxis_transform())


def performance_chart(rebased: pd.DataFrame, currency: str = "USD") -> Figure:
    """Banks indexed to 100 at the start of the sample."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name in rebased.columns:
        ax.plot(rebased.index, rebased[name], color=BANK_COLOURS.get(name, MUTED))
    ax.axhline(100, color=MUTED, linewidth=0.8)
    ax.set_title(f"ASEAN banks: total return in {currency}, start = 100")
    ax.set_xlim(rebased.index[0], rebased.index[-1])
    _years(ax)
    _end_labels(ax, rebased.iloc[-1], BANK_COLOURS, "{:.0f}")
    fig.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.08)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str) -> Figure:
    """Correlation matrix: blue negative, grey zero, red positive."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.72 * n + 2.6, 0.62 * n + 1.6))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(n), corr.index)
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_title(title, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).outline.set_visible(False)
    fig.tight_layout()
    return fig


def macro_chart(yields: pd.DataFrame, factors: pd.DataFrame) -> Figure:
    """Yield-curve spread, VIX, and gold and Brent, stacked on a shared time axis."""
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)

    spread = yields["Spread"].dropna()
    a.plot(spread.index, spread, color=BLUE)
    a.fill_between(spread.index, spread, 0, where=spread < 0, color=RED, alpha=0.25, linewidth=0)
    a.axhline(0, color=MUTED, linewidth=0.8)
    a.set_title("US yield curve: 10-year minus 2-year (percentage points). Red = inverted")

    vix = factors["VIX"].dropna()
    b.plot(vix.index, vix, color=BLUE)
    b.axhline(20, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    b.annotate("20: the usual line between calm and stressed", xy=(vix.index[-1], 20),
               xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8.5, color=MUTED,
               annotation_clip=False)
    b.set_title("VIX: expected S&P 500 volatility")

    idx = factors[["Gold", "Brent"]].ffill().dropna()
    idx = idx / idx.iloc[0] * 100
    for name, colour in (("Gold", "#eda100"), ("Brent", "#6250d6")):
        c.plot(idx.index, idx[name], color=colour)
        c.annotate(name, xy=(idx.index[-1], idx[name].iloc[-1]), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=9, annotation_clip=False)
    c.axhline(100, color=MUTED, linewidth=0.8)
    c.set_title("Gold and Brent crude, start = 100")
    c.set_xlim(spread.index[0], max(spread.index[-1], idx.index[-1]))
    _years(c)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.95, bottom=0.05, hspace=0.3)
    return fig


def regime_chart(table: pd.DataFrame, title: str, fmt: str = "{:+.0%}") -> Figure:
    """Two bars per row: the first column in blue, the second in orange."""
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(table) + 1.8))
    rows = range(len(table))
    weeks = table.attrs.get("weeks", {})
    for k, (col, colour) in enumerate(zip(table.columns, (BLUE, ORANGE))):
        offset = (k - 0.5) * 0.36
        label = f"{col} ({weeks[col]} weeks)" if col in weeks else col
        bars = ax.barh([r + offset for r in rows], table[col], height=0.32, color=colour, label=label)
        ax.bar_label(bars, labels=[fmt.format(v) for v in table[col]], padding=3, fontsize=8.5, color=INK)
    ax.set_yticks(list(rows), table.index)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    return fig


def drawdown_chart(dd: pd.DataFrame) -> Figure:
    """One small panel per bank, same scale, so the depth of each fall is comparable."""
    names = list(dd.columns)
    cols = 4
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.3 * rows + 0.6), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, name in zip(flat, names):
        s = dd[name].dropna()
        ax.fill_between(s.index, s, 0, color=BANK_COLOURS.get(name, MUTED), alpha=0.85, linewidth=0)
        ax.set_title(f"{name}  (worst {s.min():.0%})", fontsize=9.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in flat[len(names):]:
        ax.set_visible(False)
    for ax in flat[max(len(names) - cols, 0):len(names)]:
        ax.xaxis.set_tick_params(labelbottom=True)  # bottom panel of a column with a gap below
    fig.suptitle("Drawdowns: fall from the previous peak", x=0.06, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def attribution_chart(attr: pd.DataFrame) -> Figure:
    """Price + dividends + currency = total return in USD, one panel per step, banks as rows."""
    attr = attr.sort_values("Total in USD")
    steps = [("Price", "Share price", BLUE), ("Dividends", "+ Dividends", "#1baf7a"),
             ("Currency", "+ Currency", ORANGE), ("Total in USD", "= Total in USD", INK)]
    lo = min(0.0, attr[[s for s, *_ in steps]].min().min())
    hi = max(0.0, attr[[s for s, *_ in steps]].max().max())
    pad = (hi - lo) * 0.45
    fig, axes = plt.subplots(1, 4, figsize=(12, 0.5 * len(attr) + 1.7), sharey=True)
    rows = range(len(attr))
    for ax, (col, label, colour) in zip(axes, steps):
        vals = attr[col]
        bars = ax.barh(list(rows), vals, height=0.6, color=colour)
        ax.bar_label(bars, labels=[f"{v:+.0%}" for v in vals], padding=3, fontsize=8.5, color=INK)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(lo - pad * (lo < 0), hi + pad)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_title(label, fontsize=10.5)
    axes[0].set_yticks(list(rows), attr.index)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    first, last = attr.attrs.get("period", (None, None))
    when = f", {first:%b %Y} to {last:%b %Y}" if first is not None else ""
    fig.suptitle(f"Where a US-dollar investor's return came from{when}", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def valuation_chart(valuation: pd.DataFrame) -> Figure:
    """Price-to-book against return on equity, with the straight line across the banks."""
    v = valuation.dropna(subset=["P/B", "ROE"])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fit = valuation.attrs.get("fit", {})
    if v.empty:
        ax.text(0.5, 0.5, "Valuation data unavailable", ha="center", transform=ax.transAxes)
        return fig
    if pd.notna(fit.get("slope")):
        xs = pd.Series([v["ROE"].min() * 0.9, v["ROE"].max() * 1.05])
        ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=MUTED, linewidth=1,
                linestyle=(0, (4, 3)))
        ax.annotate("Peer line: P/B the group pays\nfor each level of ROE",
                    xy=(xs.iloc[0], fit["intercept"] + fit["slope"] * xs.iloc[0]),
                    xytext=(0, -10), textcoords="offset points", ha="left", va="top",
                    fontsize=8.5, color=MUTED)
    for name, row in v.iterrows():
        ax.scatter(row["ROE"], row["P/B"], s=70, color=BANK_COLOURS.get(name, MUTED),
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(name, xy=(row["ROE"], row["P/B"]), xytext=(7, -3), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.set_xlabel("Return on equity (trailing earnings / book value)")
    ax.set_ylabel("Price / book value")
    ax.set_ylim(0, v["P/B"].max() * 1.15)
    ax.margins(x=0.15)
    ax.set_title("Valuation: what investors pay for each bank's profitability")
    fig.tight_layout()
    return fig


def interval_chart(table: pd.DataFrame, value: str, half: str, title: str, fmt: str = "{:+.0%}",
                   axis_fmt: FuncFormatter | PercentFormatter | None = None, colours: dict | None = None) -> Figure:
    """Dot with a 95% interval per row. Grey when the interval crosses zero (no clear effect)."""
    t = table[[value, half]].dropna()
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.6))
    rows = list(range(len(t)))
    for r, (name, row) in zip(rows, t.iterrows()):
        lo, hi = row[value] - row[half], row[value] + row[half]
        clear = lo > 0 or hi < 0
        colour = (colours or {}).get(name, BLUE) if clear else "#a3a29d"
        ax.plot([lo, hi], [r, r], color=colour, linewidth=2.5, solid_capstyle="round")
        ax.scatter(row[value], r, s=60, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(fmt.format(row[value]), xy=(hi, r), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(rows, t.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(axis_fmt or PercentFormatter(1, decimals=0))
    ax.margins(x=0.12, y=0.08)
    ax.set_title(title)
    fig.text(0.01, 0.01, "Dot = estimate, line = 95% interval. Grey = interval includes zero.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def rolling_beta_chart(rolling: pd.DataFrame, band: pd.DataFrame, full: pd.Series, title: str) -> Figure:
    """One small panel per bank: one-year rolling beta, its 95% band, and the full-sample value."""
    names = [n for n in rolling.columns if rolling[n].notna().any()]
    cols = 4
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.3 * rows + 0.7), sharex=True, sharey=True,
                             squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, names):
        s, b = rolling[name].dropna(), band[name].reindex(rolling[name].dropna().index)
        colour = BANK_COLOURS.get(name, MUTED)
        ax.fill_between(s.index, s - b, s + b, color=colour, alpha=0.18, linewidth=0)
        ax.plot(s.index, s, color=colour, linewidth=1.6)
        ax.axhline(full.get(name, float("nan")), color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_title(f"{name}  (full sample {full.get(name, float('nan')):.2f})", fontsize=9.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in flat[len(names):]:
        ax.set_visible(False)
    for ax in flat[max(len(names) - cols, 0):len(names)]:
        ax.xaxis.set_tick_params(labelbottom=True)  # bottom panel of a column with a gap below
    fig.suptitle(title, x=0.06, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.06, 0.005, "Line = trailing 52-week estimate, band = 95% interval, dashed = whole sample.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig
