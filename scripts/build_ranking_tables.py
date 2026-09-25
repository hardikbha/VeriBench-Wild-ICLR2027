"""Appendix tables for ranking consequences and parser sensitivity (all eight judges).

Reads results/ranking_results.json (FX6: stored predictions, no model calls) and
results/parser_results.json (FX5: both parsers for all eight T0 judges). Writes tables/*.tex.
Run from the repository root:  python scripts/build_ranking_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results/ranking_results.json").read_text())
GEN = ROOT / "tables"


def fmt_ci(ci):
    return f"$[{ci[0]:+.1f},{ci[1]:+.1f}]$"


# ---- information audit: macro-F1 by condition, API-minus-open level gap, Kendall tau vs full input
A = R["A_information"]
models = ["Qwen3-4B", "Qwen3-32B", "Gemini Flash", "Gemini Pro"]
rows = [("claim_plus_note", "Claim + note (full)"), ("claim_only", "Claim only"), ("note_only", "Note only"),
        ("claim_plus_random_note", "Random note"), ("claim_plus_same_axis_wrong_note", "Same-mechanism note"),
        ("claim_plus_trigger_masked_note", "Masked note")]
lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Condition & Qwen3-4B & Qwen3-32B & Gem.\ Flash & Gem.\ Pro & API $-$ open & $\tau$ \\", r"\midrule"]
for key, label in rows:
    s = A["scores"][key]
    lv = A["api_minus_open_level"][key]
    lines.append(f"{label} & " + " & ".join(f"{100 * s[m]:.1f}" for m in models)
                 + f" & ${lv['pts']:+.1f}$ {fmt_ci(lv['ci95'])} & ${A['kendall_vs_full'][key]:+.2f}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
(GEN / "ranking_information.tex").write_text("\n".join(lines))

# ---- generalization audit: within- vs cross-source macro-F1 by regime, seed-paired QLoRA - ICL
B = R["B_generalization"]
lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"Pool & Scores & ICL (16-shot) & QLoRA & QLoRA $-$ ICL \\", r"\midrule"]
for pool, name in (("unmatched", "Unmatched"), ("matched", "Strict-matched")):
    b = B[pool]
    w, c = b["qlora_minus_icl_within_pts"], b["qlora_minus_icl_cross_pts"]
    lines.append(f"{name} & within source & {100 * b['within_mean']['icl']:.1f} & {100 * b['within_mean']['qlora']:.1f}"
                 f" & ${w['mean']:+.1f}$ {fmt_ci(w['t_ci95'])} \\\\")
    lines.append(f" & cross source & {100 * b['cross_mean']['icl']:.1f} & {100 * b['cross_mean']['qlora']:.1f}"
                 f" & ${c['mean']:+.1f}$ {fmt_ci(c['t_ci95'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
(GEN / "ranking_generalization.tex").write_text("\n".join(lines))

# ---- scoring audit: both parsers for all eight judges
P_path = ROOT / "results/parser_results.json"
if P_path.exists():
    P = json.loads(P_path.read_text())
    order = ["Mistral-7B", "Qwen2.5-0.5B", "Phi-3-medium", "Qwen3-4B", "Qwen3-32B", "Gemini Flash", "Gemini Pro", "Grok 4.7"]
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{YES (\%)} & & \multicolumn{2}{c}{$\kappa$ with majority} & $\kappa$ rank \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
             r"Judge & Templ.\ (\%) & strict & lenient & Shift & strict & lenient & strict$\to$len. \\", r"\midrule"]
    for j in order:
        v = P["judges"][j]
        mark = r"$^\ast$" if v["source"] == "rerun" else ""
        arrow = r"$\to$"
        end = r" \\"
        lines.append(f"{j}{mark} & {100 * v['template_compliance']:.1f} & "
                     f"{100 * v['strict']['yes_rate']:.1f} & {100 * v['lenient']['yes_rate']:.1f} & "
                     f"${v['parser_shift_pts']:+.1f}$ & {v['strict']['kappa']:.3f} & {v['lenient']['kappa']:.3f} & "
                     f"{P['kappa_rank_strict'][j]}{arrow}{P['kappa_rank_lenient'][j]}{end}")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (GEN / "parser_all_judges.tex").write_text("\n".join(lines))
print("written:", sorted(p.name for p in GEN.glob("ranking_*.tex")) + (["parser_all_judges.tex"] if P_path.exists() else []))
