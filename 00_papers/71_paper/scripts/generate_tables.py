
import os, pandas as pd
os.makedirs("tables", exist_ok=True)

def to_tex_table(df, path, caption, label):
    cols = list(df.columns)
    header = " \\toprule\n" + " & ".join(cols) + " \\\\ \n\\midrule\n"
    body = ""
    for _, row in df.iterrows():
        body += " & ".join(str(row[c]) if row[c] is not None else "--" for c in cols) + " \\\\ \n"
    tex = "\\begin{table}[!t]\n\\centering\n\\caption{%s}\n\\label{%s}\n\\begin{tabular}{%s}\n%s%s\\bottomrule\n\\end{tabular}\n\\end{table}\n" % (
        caption, label, "l" + "c"*(len(cols)-1), header, body
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)

liar6 = pd.read_csv("metrics/liar6_results.csv").fillna("--")
to_tex_table(liar6[["method","acc","macro_f1","auc_macro_ovr","ece","brier"]], "tables/liar6_overall.tex",
             "LIAR 6-way (content-only).", "tab:liar6_overall")

liar2 = pd.read_csv("metrics/liar2_results.csv").fillna("--")
to_tex_table(liar2, "tables/liar2_overall.tex", "LIAR binary (content-only).", "tab:liar2_overall")

pf = pd.read_csv("metrics/politifact_results.csv").fillna("--")
to_tex_table(pf, "tables/politifact_overall.tex", "PolitiFact results (content-only vs. published baselines).", "tab:pf_overall")

gc = pd.read_csv("metrics/gossipcop_results.csv").fillna("--")
to_tex_table(gc, "tables/gossipcop_overall.tex", "GossipCop results (content-only vs. published baselines).", "tab:gc_overall")

faith = pd.read_csv("metrics/faithfulness.csv")
to_tex_table(faith, "tables/faithfulness.tex", "Faithfulness and auditability metrics.", "tab:faithfulness")

transfer = pd.read_csv("metrics/cross_domain.csv")
to_tex_table(transfer, "tables/transfer.tex", "Cross-domain transfer.", "tab:transfer")

ablate_liar6 = pd.read_csv("metrics/ablation_liar6.csv")
to_tex_table(ablate_liar6, "tables/ablation_liar6.tex", "Ablation on LIAR (6-way).", "tab:ablation_liar6")

ablate_pf = pd.read_csv("metrics/ablation_politifact.csv")
to_tex_table(ablate_pf, "tables/ablation_politifact.tex", "Ablation on PolitiFact (binary).", "tab:ablation_pf")

print("Tables written to tables/")
