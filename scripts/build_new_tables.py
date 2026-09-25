"""Appendix tables for the panel, supervision, format, generator, and fresh-data experiments.

Reads results/new_results.json (aggregate numbers only) and writes tables/*.tex.
Run from the repository root:  python scripts/build_new_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = json.loads((ROOT / "results/new_results.json").read_text())
GEN = ROOT / "tables"
ORDER = ["Gemini Flash", "Gemini Pro", "Qwen3-32B", "Qwen3-4B", "Qwen2.5-0.5B", "Mistral-7B", "Llama-3.1-8B",
         "Gemma-3-12B", "Phi-3-medium", "Phi-4", "OLMo-2-13B", "Granite-3.3-8B"]
TAG = {"gemini-3.5-flash": "Gemini Flash", "gemini-3.1-pro": "Gemini Pro", "qwen3-32b-fp8": "Qwen3-32B",
       "qwen3-4b": "Qwen3-4B", "qwen2.5-0.5b": "Qwen2.5-0.5B", "mistral-7b": "Mistral-7B", "llama-3.1-8b": "Llama-3.1-8B",
       "phi-3-medium": "Phi-3-medium", "phi-4": "Phi-4", "olmo-2-13b": "OLMo-2-13B", "gemma-3-12b": "Gemma-3-12B",
       "granite-3.3-8b": "Granite-3.3-8B"}
T1C = [("claim_plus_note", "Full"), ("claim_only", "Claim"), ("note_only", "Note"), ("claim_plus_random_note", "Random"),
       ("claim_plus_same_axis_wrong_note", "Same-mech."), ("claim_plus_trigger_masked_note", "Masked")]
AVC = [("claim_plus_evidence", "Full"), ("claim_only", "Claim"), ("evidence_only", "Evidence"),
       ("claim_plus_random_evidence", "Random"), ("claim_plus_same_label_evidence", "Same-verdict"),
       ("claim_plus_masked_evidence", "Masked")]


def ci(v):
    return f"$[{v[0]:+.1f},{v[1]:+.1f}]$"


def write(name, lines):
    (GEN / name).write_text("\n".join(lines + [""]))


P = N["fx12_panel_rankings"]
fresh = {TAG[t]: v for t, v in N.get("fx11_fresh_models", {}).items()}

# ---- per-model scores: T1 frozen, AVeriTeC, fresh
lines = [r"\begin{tabular}{@{}l" + "r" * 6 + "@{\hspace{8pt}}" + "r" * 6 + "@{\hspace{8pt}}rr@{}}", r"\toprule",
         r" & \multicolumn{6}{c}{T1, frozen CN test} & \multicolumn{6}{c}{AVeriTeC} & \multicolumn{2}{c}{T1, fresh 2026} \\",
         r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(l){14-15}",
         "Model & " + " & ".join(l for _, l in T1C) + " & " + " & ".join(l for _, l in AVC) + r" & Full & Same-mech. \\",
         r"\midrule"]
for m in ORDER:
    a = [P["t1_frozen"]["conditions"][k]["macro_f1"][m] for k, _ in T1C]
    b = [P["averitec"]["conditions"][k]["macro_f1"][m] for k, _ in AVC]
    f = fresh.get(m, {})
    fr = [f"{100 * f[k]:.1f}" if k in f else "--" for k in ("claim_plus_note", "claim_plus_same_axis_wrong_note")]
    lines.append(f"{m} & " + " & ".join(f"{x:.1f}" for x in a) + " & " + " & ".join(f"{x:.1f}" for x in b)
                 + " & " + " & ".join(fr) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("panel_scores.tex", lines)

# ---- ranking summary per condition
lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Data & Condition & $\tau$ vs.\ full & Pairs flipped (sig.) & API $-$ open mean & API $-$ best open \\", r"\midrule"]
blocks = [("T1 frozen", "t1_frozen", T1C), ("AVeriTeC", "averitec", AVC)]
if "t1_fresh" in P:
    blocks.append(("T1 fresh", "t1_fresh", T1C))
for name, key, conds in blocks:
    for i, (c, lab) in enumerate(conds):
        r = P[key]["conditions"][c]
        tau = "---" if i == 0 else f"${r['kendall_tau_vs_full']:+.2f}$ {ci(r['kendall_tau_ci95'])}".replace("+.1f", "")
        if i:
            lo, hi = r["kendall_tau_ci95"]
            tau = f"${r['kendall_tau_vs_full']:+.2f}$ $[{lo:+.2f},{hi:+.2f}]$"
        fl = "---" if i == 0 else f"{r['pairs_flipped']}/{r['pairs_total']} ({r['pairs_flipped_significant']})"
        a, b = r["api_minus_open_mean"], r["api_mean_minus_best_open"]
        lines.append(f"{name if i == 0 else ''} & {lab} & {tau} & {fl} & ${a[0]:+.1f}$ {ci(a[1:3])} & ${b[0]:+.1f}$ {ci(b[1:3])} \\\\")
    lines.append(r"\midrule" if key != blocks[-1][1] else r"\bottomrule")
lines += [r"\end{tabular}"]
write("panel_rankings.tex", lines)

# ---- supervision scaling (Qwen3-4B; Llama-3.1-8B when present)
S = N["fx9_supervision_scaling"]
L = N.get("fx10_second_backbone", {})
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Backbone, pool & Examples & $G$ [95\% CI] & Within & Cross & $\Delta$within vs.\ ICL & $\Delta$cross vs.\ ICL \\", r"\midrule"]
LAB = {"icl": "ICL, 16", "16": "16", "64": "64", "256": "256", "1024": "1,024", "4078": "4,078", "full": "full",
       "k16": "16", "k256": "256"}


def rowset(title, table):
    out = []
    for k, r in table.items():
        g = r["G"]
        mi = r.get("minus_icl")
        d = (f"${mi['within']['mean']:+.1f}$ & ${mi['cross']['mean']:+.1f}$") if mi else "--- & ---"
        out.append(f"{title if not out else ''} & {LAB.get(k, k)} & {g['mean']:.1f} $[{g['t_ci95'][0]:.1f},{g['t_ci95'][1]:.1f}]$"
                   f" & {r['within_mean']:.1f} & {r['cross_mean']:.1f} & {d} \\\\")
    return out


lines += rowset("Qwen3-4B, unmatched", S["unmatched"]) + [r"\midrule"] + rowset("Qwen3-4B, matched", S["matched"])
for pool in ("unmatched", "matched"):
    if pool in L and L[pool]:
        lines += [r"\midrule"] + rowset(f"Llama-3.1-8B, {pool}", L[pool])
lines += [r"\bottomrule", r"\end{tabular}"]
write("supervision_scaling.tex", lines)

# ---- prompt formats (12 judges x 4 formats)
F = N["fx8_prompt_formats"]
FM = ["frozen", "template", "json", "label"]
lines = [r"\begin{tabular}{@{}l" + "rrr" * 4 + "@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{f}}}" for f in ("Frozen", "Template", "JSON", "Label")) + r" \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}",
         "Judge" + " & str. & len. & $\\kappa$" * 4 + r" \\", r"\midrule"]
for j in F["judges"]:
    cells = []
    for f in FM:
        s, l = F["pipelines"][f"{j}|{f}|strict"], F["pipelines"][f"{j}|{f}|lenient"]
        cells.append(f"{s['yes_rate']:.1f} & {l['yes_rate']:.1f} & {l['kappa']:.2f}")
    lines.append(f"{j} & " + " & ".join(cells) + r" \\")
lines += [r"\midrule", r"Spearman $\rho$ (YES, $\kappa$) & \multicolumn{3}{c}{" + " / ".join(
    f"{F['by_format'][f'frozen|{p}']['spearman_acceptance_vs_kappa']:+.2f}" for p in ("strict", "lenient")) + "}" +
          "".join(r" & \multicolumn{3}{c}{" + " / ".join(f"{F['by_format'][f'{f}|{p}']['spearman_acceptance_vs_kappa']:+.2f}"
                                                         for p in ("strict", "lenient")) + "}" for f in FM[1:]) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
write("prompt_formats.tex", lines)

# ---- generator gap per pipeline (FX7)
G7 = N["fx7_generator_ranking"]["pipelines"]
judges = ["Mistral-7B", "Qwen2.5-0.5B", "Phi-3-medium", "Qwen3-4B", "Qwen3-32B", "Gemini Flash", "Gemini Pro", "Grok 4.7"]
lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Judge & Gap, strict & Gap, lenient & $\tau_{10}$ strict & $\tau_{10}$ lenient \\", r"\midrule"]
for j in judges:
    s, l = G7[f"{j} | strict"], G7[f"{j} | lenient"]
    lines.append(f"{j} & ${s['gap_mistral_minus_qwen']:+.1f}$ {ci(s['gap_ci95'])} & ${l['gap_mistral_minus_qwen']:+.1f}$ {ci(l['gap_ci95'])}"
                 f" & ${s['tau_systems_vs_human']:+.2f}$ & ${l['tau_systems_vs_human']:+.2f}$ \\\\")
h = G7["Human majority"]
lines += [r"\midrule", f"Human majority & \\multicolumn{{2}}{{c}}{{${h['gap_mistral_minus_qwen']:+.1f}$ {ci(h['gap_ci95'])}}} & --- & --- \\\\",
          r"\bottomrule", r"\end{tabular}"]
write("generator_gap.tex", lines)
print("wrote panel_scores, panel_rankings, supervision_scaling, prompt_formats, generator_gap")
