"""Per-finding figures for the restructured paper (Figures 2-4) and the compact judge-human table.

Everything is read from figures/final_results.json (the hashed extract of stored results); no model calls.
Run from the repository root:  python scripts/build_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "results/final_results.json").read_text())
OUT = ROOT / "figures"

# One colour per model type, used in every figure.
TYPE_COLOR = {"fine-tuned": "#333333", "open": "#0072B2", "api": "#D55E00", "human": "#6A3D9A"}
WIDTH = 5.5  # inches, \textwidth of the ICLR template

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def panel_letter(ax, letter, x=-0.02, y=1.10):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom", ha="right")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------- Figure 2: information access
def figure2():
    info = DATA["information"]
    models = [("RoBERTa", "fine-tuned"), ("Qwen3-4B", "open"), ("Qwen3-32B", "open"),
              ("Gemini Flash", "api"), ("Gemini Pro", "api")]
    conds = [("claim_only", "Claim\nonly"), ("note_only", "Note\nonly"), ("random_note", "Random\nnote"),
             ("same_mechanism_wrong_note", "Same-mech.\nnote"), ("trigger_masked_note", "Masked\nnote")]
    grid = []
    for name, _ in models:
        row = []
        for key, _ in conds:
            if name == "RoBERTa":
                row.append(info[name]["conditions"][key]["delta_pts_mean"])
            else:
                row.append(info[name]["deltas"][key]["delta_points"])
        grid.append(row)

    fig = plt.figure(figsize=(WIDTH, 1.95))
    ax = fig.add_axes([0.13, 0.12, 0.47, 0.70])
    norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=6)
    ax.imshow(grid, cmap="RdBu", norm=norm, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < -15 else "#1A1A1A")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[1] for c in conds])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[0] for m in models])
    for tick, (_, kind) in zip(ax.get_yticklabels(), models):
        tick.set_color(TYPE_COLOR[kind])
        tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(-0.02, 1.27, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=8, ha="right")
    ax.text(0.5, -0.10, "Change in T1 macro-F1 from each model's full input (points)", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5)
    handles = [Line2D([], [], marker="s", ls="", color=TYPE_COLOR[k], label=l)
               for k, l in (("fine-tuned", "fine-tuned"), ("open", "open-weight"), ("api", "API-only"))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.26, -0.16), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)

    # (b) evidence-only change across benchmarks
    bx = fig.add_axes([0.73, 0.14, 0.26, 0.66])
    ext = DATA["external_deltas"]
    benches = [("veribench_wild(note_only)", "VeriBench-Wild"), ("pubhealth", "PubHealth"),
               ("averitec", "AVeriTeC"), ("fever", "FEVER")]
    for off, (model, label, color) in zip((-0.18, 0.18), (("qwen3-32b-fp8", "Qwen3-32B", "#0072B2"),
                                                          ("qwen3-4b", "Qwen3-4B", "#56A0C7"))):
        for i, (key, _) in enumerate(benches):
            v = ext[model][key]["delta_points"]
            lo, hi = ext[model][key]["paired_ci95_points"]
            bx.barh(i + off, v, height=0.34, color=color, edgecolor="none",
                    label=label if i == 0 else None)
            bx.errorbar(v, i + off, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="#1A1A1A", elinewidth=0.6, capsize=1.2)
    bx.set_yticks(range(len(benches)))
    bx.set_yticklabels([b[1] for b in benches])
    bx.get_yticklabels()[0].set_fontweight("bold")
    bx.invert_yaxis()
    bx.axvline(0, color="#7F7F7F", lw=0.6)
    bx.set_xlabel("Note-only change (points)")
    bx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handlelength=1.0, borderaxespad=0.1)
    bx.text(-0.02, 1.13, "(b)", transform=bx.transAxes, fontweight="bold", fontsize=8, ha="right")
    bx.text(0.0, 1.13, "Evidence alone, across benchmarks", transform=bx.transAxes, fontsize=7)
    save(fig, "fig2_information")
    return grid


# ----------------------------------------------------------------- Figure 3: source x adaptation
def figure3():
    src = DATA["source"]
    panels = [("unmatched", "B_icl16", "In-context (16-shot)", "unmatched"),
              ("unmatched", "C_qlora", "QLoRA fine-tuning", "unmatched"),
              ("matched", "B_icl16", "In-context (16-shot)", "strict-matched"),
              ("matched", "C_qlora", "QLoRA fine-tuning", "strict-matched")]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.75))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.12, wspace=0.18)
    for n, (ax, (pool, regime, title, pool_name)) in enumerate(zip(axes, panels)):
        c = src[pool][regime]
        cells = [[c["upfd->upfd"]["mean"], c["upfd->cn"]["mean"]], [c["cn->upfd"]["mean"], c["cn->cn"]["mean"]]]
        ax.imshow(cells, cmap="Blues", vmin=0.30, vmax=0.85)
        for i in range(2):
            for j in range(2):
                v = cells[i][j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        fontweight="bold" if i == j else "normal", color="white" if v > 0.66 else "#1A1A1A")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["UPFD", "CN"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["UPFD", "CN"] if n == 0 else ["", ""])
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        g = c["G"]
        lo, hi = g["t_ci95"]
        ax.set_title(f"{title}\n{pool_name}\n$G$ = {g['mean']:.1f} [{lo:.1f}, {hi:.1f}]", fontsize=6.5, pad=3,
                     color="#D55E00" if regime == "C_qlora" else "#0072B2")
        if n == 0:
            ax.set_ylabel("Adapted on", fontsize=6.5)
        ax.set_xlabel("Tested on", fontsize=6.5, labelpad=1)
    save(fig, "fig3_source")


# ----------------------------------------------------------------- Figure 4: evaluation
JUDGE_TYPE = {"Mistral-7B": "open", "Qwen2.5-0.5B": "open", "Phi-3-medium": "open", "Qwen3-4B": "open",
              "Qwen3-32B": "open", "Gemini-3.5-Flash": "api", "Gemini-3.1-Pro": "api", "Grok-4.7": "api"}
SHORT = {"Gemini-3.5-Flash": "Gemini Flash", "Gemini-3.1-Pro": "Gemini Pro", "Grok-4.7": "Grok 4.7"}
# Fixed audit order (original three judges, then the five added judges); not a ranking.
ORDER = ["Mistral-7B", "Qwen2.5-0.5B", "Phi-3-medium", "Qwen3-4B", "Qwen3-32B", "Gemini-3.5-Flash", "Gemini-3.1-Pro", "Grok-4.7"]


def shown(j):
    """The scoring rule used in the paper's figure: lenient where it exists, stored strict otherwise."""
    return j["lenient"] or j["strict"]


def figure4():
    """(a) acceptance vs. agreement with the human majority, with the parser shift; (b) generator gap."""
    judges = DATA["judges"]
    human = DATA["human"]
    fig = plt.figure(figsize=(WIDTH, 2.05))

    # (a) YES rate against Cohen's kappa with the human majority, same 476 judgments.
    ax = fig.add_axes([0.075, 0.17, 0.46, 0.72])
    hum = 100 * human["majority_yes_rate"]
    ax.axvline(hum, color=TYPE_COLOR["human"], lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(hum + 0.8, 0.622, f"human-majority acceptance {hum:.1f}%", color=TYPE_COLOR["human"],
            fontsize=5.8, va="top", ha="left")
    # (label, dx, dy, ha) offsets in data units, placed by hand so no label covers a point
    place = {"Mistral-7B": (0, 0.035, "center"), "Qwen2.5-0.5B": (0, 0.035, "center"),
             "Phi-3-medium": (1.6, 0.012, "left"), "Qwen3-4B": (1.6, -0.028, "left"),
             "Qwen3-32B": (1.6, 0.0, "left"), "Gemini-3.5-Flash": (-1.4, -0.01, "right"),
             "Gemini-3.1-Pro": (-1.4, 0.0, "right"), "Grok-4.7": (1.4, 0.022, "left")}
    for name in ORDER:
        s = shown(judges[name])
        x, y, col = 100 * s["yes_rate"], s["kappa"], TYPE_COLOR[JUDGE_TYPE[name]]
        ax.plot(x, y, "o", color=col, ms=4.6, mec="white", mew=0.5, zorder=3)
        dx, dy, ha = place[name]
        label = SHORT.get(name, name) + (" (lenient)" if name == "Qwen3-4B" else "")
        ax.text(x + dx, y + dy, label, fontsize=6.0, color=col, ha=ha, va="center")
    q_strict, q_len = judges["Qwen3-4B"]["strict"], judges["Qwen3-4B"]["lenient"]
    x0, y0, x1, y1 = 100 * q_strict["yes_rate"], q_strict["kappa"], 100 * q_len["yes_rate"], q_len["kappa"]
    ax.plot(x0, y0, "o", mfc="white", mec=TYPE_COLOR["open"], ms=4.6, mew=1.0, zorder=3)
    ax.annotate("", xy=(x1 - 0.9, y1 - 0.012), xytext=(x0 + 0.8, y0 + 0.012),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8, shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=-0.18", mutation_scale=7), zorder=2)
    ax.text(x0 + 1.6, y0 + 0.004, "Qwen3-4B (strict)", fontsize=6.0, color=TYPE_COLOR["open"], va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
    ax.text(21.5, 0.165, "same replies,\nre-parsed", fontsize=5.8, color="#555555", ha="left", va="center",
            style="italic")
    ax.set_xlim(-1.5, 70)
    ax.set_ylim(-0.02, 0.63)
    ax.set_xlabel("Acceptance: YES rate on the same 476 judgments (%)")
    ax.set_ylabel(r"Agreement with human majority ($\kappa$)")
    ax.grid(color="#EDEDED", lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", ms=4.6, color=TYPE_COLOR["open"], label="open-weight judge"),
               Line2D([], [], marker="o", ls="", ms=4.6, color=TYPE_COLOR["api"], label="API-only judge")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False, handletextpad=0.2,
              borderaxespad=0.2)
    ax.text(-0.02, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")
    ax.text(0.0, 1.04, "Acceptance is not agreement", transform=ax.transAxes, fontsize=7, va="bottom")

    # (b) Mistral - Qwen acceptance difference per judge and for the human majority.
    bx = fig.add_axes([0.73, 0.17, 0.25, 0.72])
    names = ["Human majority"] + ORDER
    for i, name in enumerate(names):
        if name == "Human majority":
            g, col = human["majority_gap_pts"], TYPE_COLOR["human"]
            lo, hi = human["majority_gap_ci95"]
            bx.plot([lo, hi], [i, i], color=col, lw=1.0, solid_capstyle="butt")
            bx.plot(g, i, "D", color=col, ms=4.0, zorder=3)
        else:
            by = judges[name]["by_generator"]
            g, col = 100 * (by["Mistral"] - by["Qwen"]), TYPE_COLOR[JUDGE_TYPE[name]]
            bx.plot([0, g], [i, i], color=col, lw=0.9, alpha=0.45)
            bx.plot(g, i, "o", color=col, ms=4.2, mec="white", mew=0.5, zorder=3)
    bx.axhspan(-0.5, 0.5, color="#F2EEF7", zorder=0)
    bx.axvline(0, color="#7F7F7F", lw=0.6)
    bx.set_yticks(range(len(names)))
    bx.set_yticklabels([SHORT.get(n, n) for n in names])
    for tick, name in zip(bx.get_yticklabels(), names):
        tick.set_color(TYPE_COLOR["human" if name == "Human majority" else JUDGE_TYPE[name]])
    bx.invert_yaxis()
    bx.set_xlim(-2, 20)
    bx.set_ylim(len(names) - 0.4, -0.6)
    bx.set_xlabel("Mistral \u2212 Qwen (points)")
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="x", color="#EDEDED", lw=0.5)
    bx.set_axisbelow(True)
    bx.text(-0.02, 1.04, "(b)", transform=bx.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")
    bx.text(0.0, 1.04, "Generator gap", transform=bx.transAxes, fontsize=7, va="bottom")
    save(fig, "fig4_scoring")


# ----------------------------------------------------------------- Table 4: human agreement by mechanism
MECHANISMS = [("factualError", "Factual error"), ("manipulatedMedia", "Manipulated media"),
              ("outdated", "Outdated"), ("missingContext", "Missing context"),
              ("unverified", "Unverified claim"), ("satire", "Satire")]


def table4():
    human = DATA["human"]
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Mechanism & $n$ & $\kappa_2$ & $\kappa_4$ \\", r"\midrule"]
    for key, label in MECHANISMS:
        m = human["per_mechanism"][key]
        lines.append(f"{label} & {m['n']} & {m['binary']['fleiss_kappa']:.3f} & {m['four_label']['fleiss_kappa']:.3f} \\\\")
    lines += [r"\midrule",
              f"All judgments & {human['n_judgments']} & {human['binary']['fleiss_kappa']:.3f} & "
              f"{human['four_label']['fleiss_kappa']:.3f} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    (ROOT / "tables" / "table4_human_agreement.tex").write_text("\n".join(lines))

# ----------------------------------------------------------------- Table 3: judges against the human majority
def table3():
    """Compact judge table; a dagger marks the three original judges (stored strict verdicts)."""
    judges = DATA["judges"]
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Judge & Type & YES\% & $\kappa$ & BA & F1 \\", r"\midrule"]
    for k, name in enumerate(ORDER):
        s = shown(judges[name])
        typ = "open" if JUDGE_TYPE[name] == "open" else "API"
        mark = r"$^\dagger$" if k < 3 else ""
        lines.append(f"{SHORT.get(name, name)}{mark} & {typ} & {100*s['yes_rate']:.1f} & {s['kappa']:.3f} & "
                     f"{s['balanced_acc']:.3f} & {s['yes_f1']:.3f} \\\\")
    q = judges["Qwen3-4B"]["strict"]
    lines += [r"\midrule",
              f"Qwen3-4B, strict & open & {100*q['yes_rate']:.1f} & {q['kappa']:.3f} & "
              f"{q['balanced_acc']:.3f} & {q['yes_f1']:.3f} \\\\",
              f"Human majority & --- & {100*DATA['human']['majority_yes_rate']:.1f} & --- & --- & --- \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    (ROOT / "tables" / "table3_judges.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    g = figure2()
    figure3()
    figure4()
    table3()
    table4()
    print("figure 2 grid:", [[round(v, 2) for v in r] for r in g])
    print("written:", sorted(p.name for p in OUT.glob("fig[234]_*")))
