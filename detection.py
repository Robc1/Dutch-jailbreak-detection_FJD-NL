"""
Detection.py summary:

Folder layout (Experiment 1):
  [affirmative prompt folder] / [Run N] / [Temperature 1.0] / *.json (with N being 1-5)

  Folder layout (Experiment 2):
  [Je bent een goede assistent/No prompt] / [Run N] / [Temperature T] / *.json (with N being 1-5 
  and with T being 0, 0.1, 0.3, 0.5, 0.7 or 1.0)

Each temperature folder holds, per language:
  - one benign file:        [benign language].json
  - several jailbreak files: [jailbreak category 1].json, [jailbreak category 2].json, ...

Each JSON file = a list of "groups" (one "group" contains the input and output prompt, token probability scores
of the first 32 tokens, a label (jailbreak or benign) and a refusal classification (true or false). 
Every prompt is scored from a single generation, not resampled).

Each entry's 'scores'[0] = per-token probability for the first 32 generated
tokens. Only scores[0][0] (the FIRST token's probability) is used. Tokens 1-31 are loaded but unused.

Per [Run N], per language:
  - ALL benign files for that language are pooled once -> "benign pool"
  - EACH jailbreak category is compared individually against that SAME pool:
      benign pool vs [jailbreak category 1]
      benign pool vs [jailbreak category 2]
      ...
  - each pairing -> ROC AUC, bootstrap CI, permutation p-value, best-threshold
    TPR/FPR/TNR/FNR, stored keyed by [jailbreak category] x [Run N]
  - refusal counts tracked per file (benign + jailbreak) x [Run N]

This repeats for every Run in [Temperature X], and for every
[affirmative prompt folder] (Exp 1) / every [Temperature X] (Exp 2).

Once all runs for one prompt/temperature are done:
  --> one .tex table per [affirmative prompt folder] / [Temperature X]:
     rows = jailbreak categories, columns = [Run 1], [Run 2], ...

Only after everything above has finished, across all prompts/temperatures:
  --> one heatmap .png: rows = jailbreak categories,
     columns = [affirmative prompt folder 1], [affirmative prompt folder 2], ...
     (or [Temperature 1], [Temperature 2], ... in Exp 2)
     each cell = mean AUC over runs where p < 0.05 (else NaN/grey)
  --> one refusal .tex table, same row/column shape, cell = mean refusal count
"""

import json
import numpy as np
from sklearn import metrics
import os
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fjd_utils import (
    IGNORE_FOLDERS,
    is_benign_file,
    collect_subfolders,
    find_json_files,
    find_exact_temperature_folder,
    find_temperature_folders,
    temperature_sort_key,
)

# ===========================================================================
# Utility helpers 
# ===========================================================================


#Prints to console and optionally write to a file.
def log(message, file=None):

    print(message)
    if file is not None:
        file.write(message + '\n')

#Partition a list of .json file paths into two (benign, jail) pairs, one per language, based on filename conventions.
def split_by_language(all_json):
    dutch_benign, english_benign = [], []
    dutch_jail,   english_jail   = [], []

    for f in all_json:
        base = os.path.basename(f).lower()
        if is_benign_file(f):
            if 'english' in base:
                english_benign.append(f)
            else:
                dutch_benign.append(f)
        else:
            if 'english' in base:
                english_jail.append(f)
            else:
                dutch_jail.append(f)

    return dutch_benign, english_benign, dutch_jail, english_jail



# ===========================================================================
# Statistical helpers
# ===========================================================================


#Reads results and labels from a list of JSON file paths.
def read_result(data_list):
    dataset = []
    labels  = []
    names   = []

    for path in data_list:
        basename = os.path.splitext(os.path.basename(path))[0]
        name = basename.split('-')[-2]
        names.append(name)
        print(f"DEBUG path parsing: full path={path}, extracted name={name}")

        with open(path, 'r', encoding='utf-8') as file:
            res = json.load(file)
            dataset.append(res)

        if is_benign_file(path):
            labels.append([1] * len(res))
        else:
            labels.append([0] * len(res))

    return dataset, labels, names

#Extracts transition score arrays from the loaded dataset.
def get_score_from_result(dataset):
    scores = []
    for datas in dataset:
        score = []
        for data in datas:
            tmp = []
            for d in data:
                tmp.append(np.array(d['scores'][0][0]))
            score.append(tmp)
        scores.append(score)
    return scores

#Calculates the Confidence Interval (CI) of the AUC scores.
def bootstrap_auc_ci(logits, labels, n_bootstraps=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    logits = np.array(logits)
    benign_idx = np.where(labels == 1)[0]
    jail_idx   = np.where(labels == 0)[0]
    auc_scores = []
    for _ in range(n_bootstraps):
        b_idx = rng.choice(benign_idx, size=len(benign_idx), replace=True)
        j_idx = rng.choice(jail_idx,   size=len(jail_idx),   replace=True)
        indices = np.concatenate([b_idx, j_idx])
        boot_logits = logits[indices]
        boot_labels = labels[indices]
        fpr, tpr, _ = metrics.roc_curve(boot_labels, boot_logits, pos_label=0)
        auc_scores.append(metrics.auc(fpr, tpr))
    alpha    = (1.0 - ci) / 2.0
    auc_low  = float(np.percentile(auc_scores, 100 * alpha))
    auc_high = float(np.percentile(auc_scores, 100 * (1 - alpha)))
    return auc_low, auc_high

#Calculates the p-value of the AUC scores.
def permutation_pvalue(logits, labels, n_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    fpr, tpr, _ = metrics.roc_curve(labels, logits, pos_label=0)
    observed_auc = metrics.auc(fpr, tpr)
    perm_aucs = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        f, t, _ = metrics.roc_curve(perm_labels, logits, pos_label=0)
        perm_aucs.append(metrics.auc(f, t))
    perm_aucs = np.array(perm_aucs)
    p_value = (np.sum(perm_aucs >= observed_auc) + 1) / (n_permutations + 1)
    return p_value


# ===========================================================================
# Output generators — LaTeX table 
# ===========================================================================
 

#Extracts the display label for a jailbreak filename stem. 
#e.g. 'jailbreak-obfuscated-natural' -> 'obfuscated-natural'
def jail_display_name(jail_filename):
    prefix = 'jailbreak-'
    if jail_filename.startswith(prefix):
        return jail_filename[len(prefix):]
    return jail_filename

#Generates the LaTeX tables (.tex file) for in the Appendix
def generate_latex_table(latex_data, title, output_dir, ci=0.95, n_runs=5):
    ci_pct = int(ci * 100)
    jail_files = sorted(latex_data.keys())
    all_runs = sorted({
        run
        for jail in jail_files
        for run in latex_data[jail].keys()
    })

    run_cols = ' & '.join([f'\\textbf{{{r}}}' for r in all_runs])
    col_spec = 'llccccc' if len(all_runs) == 5 else ('ll' + 'c' * len(all_runs))

    lines = []
    lines.append(r'\definecolor{lightgray}{gray}{0.95}')
    lines.append(r'\definecolor{decisiongreen}{RGB}{198,224,180}')
    lines.append(r'\definecolor{decisionred}{RGB}{244,204,204}')
    lines.append('')
    lines.append(r'\begin{table*}')
    lines.append(r'	\centering')
    lines.append(r'	\scriptsize')
    lines.append(f'	\\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'		\toprule')
    lines.append(f'		\\textbf{{Jailbreak}} & \\textbf{{Metric}} & {run_cols} \\\\')
    lines.append(r'		\midrule')

    for jail_idx, jail_filename in enumerate(jail_files):
        display  = jail_display_name(jail_filename)
        run_data = latex_data[jail_filename]

        def fmt_runs(key, fmt='.4f'):
            return ' & '.join(
                f'{run_data.get(r, {}).get(key, float("nan")):{fmt}}'
                if run_data.get(r, {}).get(key) is not None else 'N/A'
                for r in all_runs
            )

        def fmt_ci(low_key, high_key):
            parts = []
            for r in all_runs:
                entry = run_data.get(r, {})
                lo = entry.get(low_key)
                hi = entry.get(high_key)
                parts.append(f'[{lo:.4f},{hi:.4f}]' if (lo is not None and hi is not None) else 'N/A')
            return ' & '.join(parts)
        
        def fmt_pvalues():
            return ' & '.join(
                f'{run_data.get(r, {}).get("p_value", float("nan")):.4g}'
                if run_data.get(r, {}).get("p_value") is not None else 'N/A'
                for r in all_runs
        )

        lines.append(f'		% ---------------- {display.upper()} ----------------')
        lines.append(f'		{display}')
        lines.append(f'		& AUC & {fmt_runs("auc")} \\\\')
        lines.append(f'		\\rowcolor{{lightgray}}')
        lines.append(f'		& AUC {ci_pct}\\% CI & {fmt_ci("auc_low", "auc_high")} \\\\')
        lines.append(f'		& TPR & {fmt_runs("tpr")} \\\\')
        lines.append(f'		\\rowcolor{{lightgray}}')
        lines.append(f'		& TNR & {fmt_runs("tnr")} \\\\')
        lines.append(f'		& FPR & {fmt_runs("fpr")} \\\\')
        lines.append(f'		\\rowcolor{{lightgray}}')
        lines.append(f'		& FNR & {fmt_runs("fnr")} \\\\')
        lines.append(f'     & p-value & {fmt_pvalues()} \\\\')
        lines.append(f'		\\rowcolor{{lightgray}}')

        decision_cells = []
        for r in all_runs:
            dec   = run_data.get(r, {}).get('decision', 'N/A')
            color = 'decisiongreen' if dec == 'Reject' else 'decisionred'
            decision_cells.append(f'\\cellcolor{{{color}}}{dec}')
        lines.append(f'		& Decision & {" & ".join(decision_cells)} \\\\')

        if jail_idx < len(jail_files) - 1:
            lines.append(r'		\midrule')

    safe_title  = title.replace(' ', '_').replace('/', '-')
    label_name  = safe_title.lower()

    lines.append(r'		\bottomrule')
    lines.append(r'	\end{tabular}')
    lines.append(
        f'	\\caption{{Complete results across Runs 1--{len(all_runs)} for all jailbreak types'
        f' — {title}.}}'
    )
    lines.append(f'	\\label{{tab:{label_name}_full}}')
    lines.append(r'\end{table*}')

    tex_content = '\n'.join(lines) + '\n'
    out_path    = os.path.join(output_dir, f"{safe_title}_table.tex")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"[LATEX] Saved table to: {out_path}")


# ===========================================================================
# Output generators — heatmap
# ===========================================================================

#Computes mean AUC and a significance mask over runs for every (x_label, jailbreak) pair.
#Only AUC values from significant runs contribute to the mean. 
#If no runs are significant the cell is set to NaN, which causes generate_heatmap to display it 
#as grey "N/A" via cmap.set_bad. 
#sig_mask is True for cells where at least one significant run exists 
#(i.e. a real mean could be computed); False where the cell is NaN.

def build_mean_auc_matrix(auc_data, sort_x_key=None):
    x_labels   = sorted(auc_data.keys(), key=sort_x_key)
    jail_names = sorted({
        jail
        for x in auc_data.values()
        for jail in x.keys()
    })

    matrix   = np.full((len(jail_names), len(x_labels)), np.nan)
    sig_mask = np.zeros((len(jail_names), len(x_labels)), dtype=bool)

    for col, x in enumerate(x_labels):
        for row, jail in enumerate(jail_names):
            run_entries = auc_data[x].get(jail, {})
            if run_entries:
                sig_auc_vals = []
                for entry in run_entries.values():
                    if isinstance(entry, dict):
                        if entry['significant']:
                            sig_auc_vals.append(entry['auc'])
                    else:
                        sig_auc_vals.append(float(entry))

                if sig_auc_vals:
                    # Mean over significant runs only
                    matrix[row, col]   = float(np.mean(sig_auc_vals))
                    sig_mask[row, col] = True
                

    return x_labels, jail_names, matrix, sig_mask


#Variant of build_mean_auc_matrix for the without-refusals heatmaps. 
#Mirrors build_mean_auc_matrix exactly: mean over significant runs only, 
#NaN if no significant runs — but additionally tracks whether a cell has NO stored entries 
#at all (all prompts refused) via all_refused_mask. 
#This lets generate_heatmap_without_refusals distinguish two NaN causes: 
#   1.all_refused_mask True --> black "All refused (no data)"
#   2.all_refused_mask False --> grey  "Not significant (p >= 0.05)" (same as original)

def build_mean_auc_matrix_no_refusals(auc_data, sort_x_key=None):
    x_labels   = sorted(auc_data.keys(), key=sort_x_key)
    jail_names = sorted({
        jail
        for x in auc_data.values()
        for jail in x.keys()
    })

    matrix           = np.full((len(jail_names), len(x_labels)), np.nan)
    sig_mask         = np.zeros((len(jail_names), len(x_labels)), dtype=bool)
    all_refused_mask = np.ones((len(jail_names), len(x_labels)), dtype=bool)

    for col, x in enumerate(x_labels):
        for row, jail in enumerate(jail_names):
            run_entries = auc_data[x].get(jail, {})
            if run_entries:
                all_refused_mask[row, col] = False  # entries exist --> not all refused
                sig_auc_vals = []
                for entry in run_entries.values():
                    if isinstance(entry, dict):
                        if entry['significant']:
                            sig_auc_vals.append(entry['auc'])
                    else:
                        sig_auc_vals.append(float(entry))

                if sig_auc_vals:
                    matrix[row, col]   = float(np.mean(sig_auc_vals))
                    sig_mask[row, col] = True

    return x_labels, jail_names, matrix, sig_mask, all_refused_mask



#Generates and saves the AUC heatmap. Non-significant cells (sig_mask == False) are overlaid with 
#a grey rectangle so they stand out from significant results.  
#A legend entry explains the grey fill.

def generate_heatmap(x_labels, jail_names, matrix, output_dir, filename,
                     x_axis_label, title, sig_mask=None):
    n_jails = len(jail_names)
    n_x     = len(x_labels)

    fig_width  = max(8,  n_x     * 1.8 + 3)
    fig_height = max(6,  n_jails * 0.7 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='lightgrey')   # NaN cells shown in light grey

    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest',
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Mean AUC (over runs)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(n_x))
    ax.set_xticklabels(x_labels, fontsize=9, rotation=35, ha='right')

    ax.set_yticks(np.arange(n_jails))
    ax.set_yticklabels(jail_names, fontsize=9)

    _INSIG_COLOR = '#AAAAAA' 

    for row in range(n_jails):
        for col in range(n_x):
            val = matrix[row, col]

            if np.isnan(val):
                # Grey background via cmap.set_bad --> write "N/A" in white
                ax.text(col, row, 'N/A', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold', zorder=3)
            else:
                # Significant cell, pick text colour for readability
                text_color = 'black' if 0.35 < val < 0.75 else 'white'
                ax.text(
                    col, row, f'{val:.3f}',
                    ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold',
                    zorder=3,
                )

    ax.set_xticks(np.arange(n_x)     - 0.5, minor=True)
    ax.set_yticks(np.arange(n_jails) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xlabel(x_axis_label, fontsize=12, labelpad=10)
    ax.set_ylabel('Jailbreak File',  fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)

    #Legend
    legend_handles = [
        mpatches.Patch(
            facecolor=_INSIG_COLOR,
            edgecolor='white',
            label='Not significant\n(p \u2265 0.05)',
        )
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.18, 1.0), 
        fontsize=9,
        framealpha=0.9,
        title='Significance',
        title_fontsize=9,
    )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[HEATMAP] Saved heatmap to: {out_path}")


# ===========================================================================
# Heatmap orchestration — called once per experiment after all evaluations
# ===========================================================================


   
# Builds and saves the Experiment 1 heatmap. 
# heatmap_auc_data[prompt_name][jail_stem][run_key] = {'auc': float, 'significant': bool} Dutch 
# and English jailbreaks are pooled (as in heatmap.py).
 
def generate_heatmap_exp1(heatmap_auc_data, root_name, reporting_dir):
    if not heatmap_auc_data:
        print("[HEATMAP] No data collected — heatmap skipped.")
        return

    x_labels, jail_names, matrix, sig_mask = build_mean_auc_matrix(
        heatmap_auc_data, sort_x_key=None
    )
    print(f"[HEATMAP] Exp1 matrix shape: {matrix.shape}  "
          f"({len(jail_names)} jailbreaks \u00d7 {len(x_labels)} prompts)")

    generate_heatmap(
        x_labels     = x_labels,
        jail_names   = jail_names,
        matrix       = matrix,
        sig_mask     = sig_mask,
        output_dir   = reporting_dir,
        filename     = f"{root_name}_experiment1_heatmap",
        x_axis_label = 'Affirmative Prompt',
        title        = f'Mean AUC per Affirmative Prompt \u00d7 Jailbreak\n({root_name})',
    )




# Builds and saves the Experiment 2 heatmap. 
# heatmap_auc_data[temp_name][jail_stem][run_key] = {'auc': float, 'significant': bool}
# Dutch and English jailbreaks are pooled (as in heatmap.py).

def generate_heatmap_exp2(heatmap_auc_data, root_name, reporting_dir):
    if not heatmap_auc_data:
        print("[HEATMAP] No data collected — heatmap skipped.")
        return

    x_labels, jail_names, matrix, sig_mask = build_mean_auc_matrix(
        heatmap_auc_data, sort_x_key=temperature_sort_key
    )
    print(f"[HEATMAP] Exp2 matrix shape: {matrix.shape}  "
          f"({len(jail_names)} jailbreaks \u00d7 {len(x_labels)} temperatures)")

    generate_heatmap(
        x_labels     = x_labels,
        jail_names   = jail_names,
        matrix       = matrix,
        sig_mask     = sig_mask,
        output_dir   = reporting_dir,
        filename     = f"{root_name}_experiment2_heatmap",
        x_axis_label = 'Temperature',
        title        = f'Mean AUC per Temperature \u00d7 Jailbreak\n({root_name})',
    )

# ===========================================================================
# Heatmap orchestration, without-refusals variants
# ===========================================================================


# Generate and save the without-refusals AUC heatmap. Identical to generate_heatmap except NaN cells
# are rendered in two distinct colours depending on their cause: 
# - lightgrey : entries exist but none were significant  ("Not significant") 
# - black     : no entries at all — every prompt was refused ("All refused") 
# Since matplotlib's cmap.set_bad only supports one bad colour, NaN cells are rendered 
# lightgrey via cmap.set_bad, and black cells are drawn on top as filled rectangles using all_refused_mask.

def generate_heatmap_with_refused(x_labels, jail_names, matrix, all_refused_mask,
                                  output_dir, filename, x_axis_label, title):
    n_jails = len(jail_names)
    n_x     = len(x_labels)

    fig_width  = max(8,  n_x     * 1.8 + 3)
    fig_height = max(6,  n_jails * 0.7 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='lightgrey')   # NaN cells: lightgrey = not significant

    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest',
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Mean AUC (over runs)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(n_x))
    ax.set_xticklabels(x_labels, fontsize=9, rotation=35, ha='right')

    ax.set_yticks(np.arange(n_jails))
    ax.set_yticklabels(jail_names, fontsize=9)

    _INSIG_COLOR = '#AAAAAA'

    for row in range(n_jails):
        for col in range(n_x):
            val = matrix[row, col]

            if all_refused_mask[row, col]:
                # Draw a black rectangle over the lightgrey NaN background
                ax.add_patch(plt.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    color='black', zorder=2,
                ))
                ax.text(col, row, 'N/A', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold', zorder=3)
            elif np.isnan(val):
                # lightgrey via cmap.set_bad — not significant, write "N/A" in white
                ax.text(col, row, 'N/A', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold', zorder=3)
            else:
                text_color = 'black' if 0.35 < val < 0.75 else 'white'
                ax.text(
                    col, row, f'{val:.3f}',
                    ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold',
                    zorder=3,
                )

    ax.set_xticks(np.arange(n_x)     - 0.5, minor=True)
    ax.set_yticks(np.arange(n_jails) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xlabel(x_axis_label, fontsize=12, labelpad=10)
    ax.set_ylabel('Jailbreak File',  fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)

    #Legend
    legend_handles = [
        mpatches.Patch(
            facecolor=_INSIG_COLOR,
            edgecolor='white',
            label='Not significant\n(p \u2265 0.05)',
        ),
        mpatches.Patch(
            facecolor='black',
            edgecolor='white',
            label='All refused\n(no data)',
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.18, 1.0),
        fontsize=9,
        framealpha=0.9,
        title='Significance',
        title_fontsize=9,
    )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[HEATMAP] Saved heatmap to: {out_path}")



# Build and save the Experiment 1 heatmap with refusals excluded.
# heatmap_auc_data[prompt_name][jail_stem][run_key] = {'auc': float, 'significant': bool} 
# Cells where all prompts were refused appear black with white "N/A".
# Cells with entries but no significant runs appear lightgrey with white "N/A".

def generate_heatmap_exp1_without_refusals(heatmap_auc_data, root_name, reporting_dir):
    if not heatmap_auc_data:
        print("[HEATMAP] No data collected — without-refusals heatmap skipped.")
        return

    x_labels, jail_names, matrix, sig_mask, all_refused_mask = \
        build_mean_auc_matrix_no_refusals(heatmap_auc_data, sort_x_key=None)
    print(f"[HEATMAP] Exp1 (no refusals) matrix shape: {matrix.shape}  "
          f"({len(jail_names)} jailbreaks \u00d7 {len(x_labels)} prompts)")

    generate_heatmap_with_refused(
        x_labels         = x_labels,
        jail_names       = jail_names,
        matrix           = matrix,
        all_refused_mask = all_refused_mask,
        output_dir       = reporting_dir,
        filename         = f"{root_name}_experiment1_heatmap_without_refusals",
        x_axis_label     = 'Affirmative Prompt',
        title            = f'Mean AUC per Affirmative Prompt \u00d7 Jailbreak\n({root_name}) \u2014 Refusals Excluded',
    )


# Builds and saves the Experiment 2 heatmap with refusals excluded.
# heatmap_auc_data[temp_name][jail_stem][run_key] = {'auc': float, 'significant': bool}
# Cells where all prompts were refused appear black with white "N/A". 
# Cells with entries but no significant runs appear lightgrey with white "N/A".

def generate_heatmap_exp2_without_refusals(heatmap_auc_data, root_name, reporting_dir):

    if not heatmap_auc_data:
        print("[HEATMAP] No data collected — without-refusals heatmap skipped.")
        return

    x_labels, jail_names, matrix, sig_mask, all_refused_mask = \
        build_mean_auc_matrix_no_refusals(heatmap_auc_data, sort_x_key=temperature_sort_key)
    print(f"[HEATMAP] Exp2 (no refusals) matrix shape: {matrix.shape}  "
          f"({len(jail_names)} jailbreaks \u00d7 {len(x_labels)} temperatures)")

    generate_heatmap_with_refused(
        x_labels         = x_labels,
        jail_names       = jail_names,
        matrix           = matrix,
        all_refused_mask = all_refused_mask,
        output_dir       = reporting_dir,
        filename         = f"{root_name}_experiment2_heatmap_without_refusals",
        x_axis_label     = 'Temperature',
        title            = f'Mean AUC per Temperature \u00d7 Jailbreak\n({root_name}) \u2014 Refusals Excluded',
    )

# ===========================================================================
# Output helper: merge Dutch + English results, then save combined outputs
# ===========================================================================

# Merges Dutch and English result dicts into a single combined dict, then generate one LaTeX table 
# for both languages together.
def merge_and_save_outputs(latex_data_dutch, latex_data_english,
                           title, output_dir, ci, log_file=None):

    os.makedirs(output_dir, exist_ok=True)

    combined_latex_data = {**latex_data_dutch, **latex_data_english}

    print(f"\n[DEBUG] '{title}' combined latex_data keys: {list(combined_latex_data.keys())}")

    if combined_latex_data:
        generate_latex_table(
            latex_data=combined_latex_data,
            title=title,
            output_dir=output_dir,
            ci=ci,
            n_runs=len({rk for jail in combined_latex_data for rk in combined_latex_data[jail]}),
        )
    else:
        msg = f"[WARN] latex_data empty for '{title}' — table skipped."
        print(msg)
        if log_file:
            log_file.write(msg + '\n')

# ===========================================================================
# Refusal counting helper
# ===========================================================================

# Count the number of entries with ""refused": true" across all JSON files.
# Each file contains a JSON array of groups, where each group is itself a list of prompt-result 
# dicts (matching the nested structure used by "read_result" / "get_score_from_result"):
#        file → [ [ {entry}, {entry}, ... ], [ {entry}, ... ], ... ]
# Each entry dict may optionally carry a "refused" boolean field.
# Entries that lack the field entirely are treated as not refused.

def count_refusals_from_files(json_file_list):
    total = 0
    for path in json_file_list:
        with open(path, 'r', encoding='utf-8') as fh:
            groups = json.load(fh)
        # groups is a list of groups; each group is a list of entry dicts.
        for group in groups:
            for entry in group:
                if entry.get('refused', False):
                    total += 1
    return total

# ===========================================================================
# Refusal table for Experiment 1 
# ===========================================================================

# Break a prompt name into a ``\\makecell`` header with at most "words_per_line" words per line.
def _make_makecell_header(prompt_name, words_per_line=3):

    safe = prompt_name.replace('-', r'\mbox{-}')
    words = safe.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        chunk = ' '.join(words[i:i + words_per_line])
        lines.append(f'\\textbf{{{chunk}}}')
    inner = r'\\'.join(lines)
    return f'\\makecell{{{inner}}}'




#Return the lines for one ``tabular`` block wrapped in "\\resizebox{\\textwidth}{!}".
def _build_subtable(prompt_subset, jail_stems, refusal_data, all_run_keys,
                    jail_prompt_counts=None):
    t1 = '\t'
    t2 = '\t\t'
    t3 = '\t\t\t'

    n_cols   = len(prompt_subset)
    col_spec = 'l' + 'c' * n_cols

    header_parts = [
        _make_makecell_header(p, words_per_line=3)
        for p in prompt_subset
    ]
    header_row = (
        t3 + '\\textbf{Jailbreak} &\n'
        + ' &\n'.join(t3 + h for h in header_parts)
        + ' \\\\'
    )

    lines = []
    lines.append(t1 + '\\resizebox{\\textwidth}{!}{%')
    lines.append(t2 + '\\begin{tabular}{' + col_spec + '}')
    lines.append(t3 + '\\toprule')
    lines.append(header_row)
    lines.append(t3 + '\\midrule')

    for jail_idx, jail_stem in enumerate(jail_stems):
        base_display = jail_display_name(jail_stem).replace('-', r'\mbox{-}')
        if jail_prompt_counts and jail_stem in jail_prompt_counts:
            display = base_display + ' (' + str(jail_prompt_counts[jail_stem]) + ')'
        else:
            display = base_display

        cells = []
        for prompt in prompt_subset:
            run_counts = refusal_data.get(prompt, {}).get(jail_stem, {})
            if all(rk in run_counts for rk in all_run_keys):
                avg = sum(run_counts[rk] for rk in all_run_keys) / len(all_run_keys)
                cells.append(str(round(avg)))
            else:
                # At least one run is missing for this cell --> whole cell is N/A
                cells.append('N/A')

        if jail_idx % 2 == 1:
            lines.append(t3 + '\\rowcolor{lightgray}')
        lines.append(t3 + display + ' & ' + ' & '.join(cells) + ' \\\\')

    lines.append(t3 + '\\bottomrule')
    lines.append(t2 + '\\end{tabular}%')
    lines.append(t1 + '}')

    return lines


    
# Generate a LaTeX table that shows, for every (jailbreak file, affirmative prompt) pair, 
# the number of refused prompts in each run.
def generate_refusal_table_exp1(refusal_data, all_run_keys, root_name, reporting_dir,
                                jail_prompt_counts=None):

    if not refusal_data:
        print("[REFUSAL TABLE] No refusal data collected — table skipped.")
        return

    prompt_names = sorted(refusal_data.keys())
    jail_stems   = sorted({
        jail
        for prompt in refusal_data.values()
        for jail in prompt.keys()
    })

    if not prompt_names or not jail_stems:
        print("[REFUSAL TABLE] Empty prompt or jailbreak list — table skipped.")
        return

    split      = (len(prompt_names) + 1) // 2
    top_half   = prompt_names[:split]
    bot_half   = prompt_names[split:]

    safe_root  = root_name.replace(' ', '_').replace('/', '-')
    label_name = safe_root.lower()
    n_runs     = len(all_run_keys)

    lines = []
    lines.append(r'\definecolor{lightgray}{gray}{0.95}')
    lines.append('')
    lines.append(r'\begin{table*}')
    lines.append(r'	\centering')
    lines.append(r'	\footnotesize')
    lines.append(r'	\setlength{\tabcolsep}{2.5pt}')
    lines.append('')
    lines.append('\t% --- Top half ---')
    lines.extend(_build_subtable(top_half, jail_stems, refusal_data, all_run_keys,
                                 jail_prompt_counts=jail_prompt_counts))

    if bot_half:
        lines.append('')
        lines.append(r'	\vspace{1em}')
        lines.append('')
        lines.append('\t% --- Bottom half ---')
        lines.extend(_build_subtable(bot_half, jail_stems, refusal_data, all_run_keys,
                                     jail_prompt_counts=jail_prompt_counts))

    lines.append('')
    n_runs_str = f'Run 1--{n_runs}' if n_runs > 1 else 'Run 1'
    lines.append(
        f'\t\\caption{{Average refusal counts for every jailbreak~$\\times$~affirmative-prompt'
        f' combination. Each cell shows the mean number of refused prompts across'
        f' {n_runs_str}, rounded to the nearest whole number;'
        f' \\texttt{{N/A}} indicates that at least one run was missing.}}'
    )
    lines.append(f'\t\\label{{tab:{label_name}_exp1_refusals}}')
    lines.append(r'\end{table*}')

    tex_content = '\n'.join(lines) + '\n'
    out_path    = os.path.join(reporting_dir, f"{safe_root}_experiment1_refusal_table.tex")
    os.makedirs(reporting_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(tex_content)
    print(f"[REFUSAL TABLE] Saved to: {out_path}")


# ===========================================================================
# Refusal table for Experiment 2
# ===========================================================================


# Generate a LaTeX table that shows, for every (jailbreak file, temperature)pair, 
# the number of refused prompts in each run.
def generate_refusal_table_exp2(refusal_data, all_run_keys, root_name, reporting_dir,
                                jail_prompt_counts=None):
    if not refusal_data:
        print("[REFUSAL TABLE EXP2] No refusal data collected -- table skipped.")
        return

    temp_names = sorted(refusal_data.keys(), key=temperature_sort_key)
    jail_stems = sorted({
        jail
        for temp in refusal_data.values()
        for jail in temp.keys()
    })

    if not temp_names or not jail_stems:
        print("[REFUSAL TABLE EXP2] Empty temperature or jailbreak list -- table skipped.")
        return

    split    = (len(temp_names) + 1) // 2
    top_half = temp_names[:split]
    bot_half = temp_names[split:]

    safe_root  = root_name.replace(' ', '_').replace('/', '-')
    label_name = safe_root.lower()
    n_runs     = len(all_run_keys)
    n_runs_str = 'Run 1--' + str(n_runs) if n_runs > 1 else 'Run 1'

    t = '\t'

    lines = []
    lines.append(r'\definecolor{lightgray}{gray}{0.95}')
    lines.append('')
    lines.append(r'\begin{table*}')
    lines.append(t + r'\centering')
    lines.append(t + r'\footnotesize')
    lines.append(t + r'\setlength{\tabcolsep}{2.5pt}')
    lines.append('')
    lines.append(t + '% --- Top half ---')
    lines.extend(_build_subtable(top_half, jail_stems, refusal_data, all_run_keys,
                                 jail_prompt_counts=jail_prompt_counts))

    if bot_half:
        lines.append('')
        lines.append(t + r'\vspace{1em}')
        lines.append('')
        lines.append(t + '% --- Bottom half ---')
        lines.extend(_build_subtable(bot_half, jail_stems, refusal_data, all_run_keys,
                                     jail_prompt_counts=jail_prompt_counts))

    lines.append('')
    lines.append(
        t + r'\caption{Average refusal counts for every jailbreak~$\times$~temperature'
        + ' combination (' + root_name + '). Each cell shows the mean number of refused'
        + ' prompts across ' + n_runs_str + r', rounded to the nearest whole number;'
        + r' \texttt{N/A} indicates that at least one run was missing.}'
    )
    lines.append(t + r'\label{tab:' + label_name + r'_exp2_refusals}')
    lines.append(r'\end{table*}')

    tex_content = '\n'.join(lines) + '\n'
    out_path    = os.path.join(reporting_dir, safe_root + '_experiment2_refusal_table.tex')
    os.makedirs(reporting_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(tex_content)
    print('[REFUSAL TABLE EXP2] Saved to: ' + out_path)


# ===========================================================================
# Core evaluation
# ===========================================================================


# Reads results from a list of JSON file paths, skipping any prompt entry that has ""refused": true". 
# Each JSON file is a list of prompts; each prompt is a single-element wrapper list whose only element 
# is the entry dict.  An entry is skipped entirely when "entry.get('refused', False)"" is True. 
# For non-refused entries, "entry['scores'][0][0]"" (the first token's probability) is extracted and 
# wrapped as "np.array(value)", matching exactly what "get_score_from_result" produces for a single prompt. 
# The label is derived from the filename basename (lowercased): 
#   - "benign" in basename --> label 1 
#   - otherwise            --> label 0

def read_result_without_refusals(data_list):
    dataset = []
    labels  = []
    names   = []

    for path in data_list:
        basename = os.path.splitext(os.path.basename(path))[0]
        name = basename.split('-')[-2]
        names.append(name)
        print(f"DEBUG path parsing: full path={path}, extracted name={name}")

        with open(path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)

        filtered = [entry for entry in raw if not entry[0].get('refused', False)]

        dataset.append(filtered)

        if is_benign_file(path):
            labels.append([1] * len(filtered))
        else:
            labels.append([0] * len(filtered))

    return dataset, labels, names



# Evaluate one (benign set, jailbreak set) pair with refused prompts excluded, storing results into 
# the heatmap data structure only. No LaTeX-table storage is performed.  Cells where all prompts in 
# a jailbreak file were refused (zero entries remaining) are simply not stored; "build_mean_auc_matrix" 
# will produce NaN for those cells, rendered as black "N/A" by "generate_heatmap" when "black_nan=True".

def run_evaluation_without_refusals(benign_data_list, jail_data_list,
                                    n_bootstraps, ci, label="",
                                    log_file=None,
                                    heatmap_auc_data=None,
                                    heatmap_x_key=None,
                                    run_key=None):
    if not benign_data_list:
        log(f"  [SKIP] No benign files found for: {label}", log_file)
        return
    if not jail_data_list:
        log(f"  [SKIP] No jailbreak files found for: {label}", log_file)
        return

    benign_dataset, benign_labels_data, _ = read_result_without_refusals(benign_data_list)
    jail_dataset,   jail_labels_data,   _ = read_result_without_refusals(jail_data_list)

    benign_scores = get_score_from_result(benign_dataset)
    jail_scores   = get_score_from_result(jail_dataset)

    LENGTH     = len(benign_scores[0][0])
    BENIGN_NUM = len(benign_scores)
    JAIL_NUM   = len(jail_scores)

    for l in range(LENGTH):
        logits, labels = [], []

        for bn in range(BENIGN_NUM):
            for score in benign_scores[bn]:
                logits.append(score[l])
            labels += benign_labels_data[bn]

        benign_logits      = logits.copy()
        benign_labels_copy = labels.copy()

        for an in range(JAIL_NUM):
            jail_file_scores = jail_scores[an]

            if not jail_file_scores:
                jail_filename = os.path.splitext(
                    os.path.basename(jail_data_list[an]))[0]
                log(f"  [SKIP] All prompts refused in {jail_filename} for: {label}",
                    log_file)
                continue

            logits = benign_logits.copy()
            labels = benign_labels_copy.copy()

            for score in jail_file_scores:
                logits.append(score[l])
            labels += jail_labels_data[an]

            log(f"\n=== [{label}] — no refusals ===", log_file)
            log(f"++++++++++++++++++++++++++++++++++++++++++++++", log_file)

            jail_filename = os.path.splitext(
                os.path.basename(jail_data_list[an]))[0]
            log(f"Jailbreak: {jail_filename}", log_file)

            logits = -np.array(logits)

            fpr, tpr, _ = metrics.roc_curve(labels, logits, pos_label=0)
            auc         = metrics.auc(fpr, tpr)

            log(f"Benign scores:    {int(np.sum(np.array(labels) == 1))}", log_file)
            log(f"Jailbreak scores: {int(np.sum(np.array(labels) == 0))}", log_file)
            log(f"AUC: {auc}", log_file)

            auc_low, auc_high = bootstrap_auc_ci(
                logits, labels, n_bootstraps=n_bootstraps, ci=ci)
            log(f"AUC {int(ci * 100)}% CI: [{auc_low:.4f}, {auc_high:.4f}]", log_file)

            p_value = permutation_pvalue(logits, labels,
                                         n_permutations=n_bootstraps, seed=42)
            log(f"p-value (H_0: AUC = 0.5): {p_value:.4f}", log_file)

            reject = p_value < 0.05
            log("Reject Null Hypothesis" if reject else "Fail To Reject Null Hypothesis",
                log_file)

            if heatmap_auc_data is not None and heatmap_x_key is not None and run_key is not None:
                heatmap_auc_data[heatmap_x_key][jail_filename][run_key] = {
                    'auc':         float(auc),
                    'significant': reject,
                }
                print(f"[HEATMAP/no-refusals] Stored x={heatmap_x_key}, "
                      f"jail={jail_filename}, run={run_key}, "
                      f"auc={auc:.4f}, significant={reject}")

  
#Evaluates one (benign set, jailbreak set) pair and stores metrics.
def run_evaluation(benign_data_list, jail_data_list, n_bootstraps, ci,
                   label="", log_file=None, run_key=None,
                   latex_data=None, heatmap_auc_data=None, heatmap_x_key=None,
                   jail_prompt_counts=None):
    if not benign_data_list:
        log(f"  [SKIP] No benign files found for: {label}", log_file)
        return
    if not jail_data_list:
        log(f"  [SKIP] No jailbreak files found for: {label}", log_file)
        return

    benign_dataset, benign_labels_data, benign_names = read_result(benign_data_list)
    jail_dataset,   jail_labels_data,   jail_names   = read_result(jail_data_list)

    benign_scores = get_score_from_result(benign_dataset)
    jail_scores   = get_score_from_result(jail_dataset)

    LENGTH     = len(benign_scores[0][0])
    BENIGN_NUM = len(benign_scores)
    JAIL_NUM   = len(jail_scores)

    for l in range(LENGTH):
        logits, labels = [], []

        for bn in range(BENIGN_NUM):
            for score in benign_scores[bn]:
                logits.append(score[l])
            labels += benign_labels_data[bn]

        benign_logits      = logits.copy()
        benign_labels_copy = labels.copy()

        for an in range(JAIL_NUM):
            logits = benign_logits.copy()
            labels = benign_labels_copy.copy()

            for score in jail_scores[an]:
                logits.append(score[l])
            labels += jail_labels_data[an]

            log(f"\n=== [{label}] ===", log_file)
            log(f"++++++++++++++++++++++++++++++++++++++++++++++", log_file)
            log(f"Jailbreak: {jail_names[an]}", log_file)

            logits = -np.array(logits)   # negate: higher score = more likely jailbreak

            fpr, tpr, _ = metrics.roc_curve(labels, logits, pos_label=0)
            auc         = metrics.auc(fpr, tpr)

            dist        = fpr ** 2 + (1 - tpr) ** 2
            best        = np.argmin(dist)
            tpr_at_best = tpr[best]
            fpr_at_best = fpr[best]
            tnr_at_best = 1 - fpr[best]
            fnr_at_best = 1 - tpr[best]

            log(f"Benign scores:    {np.sum(np.array(labels) == 1)}", log_file)
            jail_score_count = int(np.sum(np.array(labels) == 0))
            log(f"Jailbreak scores: {jail_score_count}", log_file)
            log(f"Best TPR: {tpr_at_best}", log_file)
            log(f"Best FPR: {fpr_at_best}", log_file)
            log(f"Best TNR: {tnr_at_best}", log_file)
            log(f"Best FNR: {fnr_at_best}", log_file)
            log(f"Best AUC: {auc}",         log_file)

            auc_low, auc_high = bootstrap_auc_ci(logits, labels,
                                                  n_bootstraps=n_bootstraps, ci=ci)
            log(f"AUC {int(ci * 100)}% CI: [{auc_low:.4f}, {auc_high:.4f}]", log_file)

            p_value = permutation_pvalue(logits, labels,
                                         n_permutations=n_bootstraps, seed=42)
            log(f"p-value (H_0: AUC = 0.5): {p_value:.4f}", log_file)

            reject = p_value < 0.05
            log("Reject Null Hypothesis" if reject else "Fail To Reject Null Hypothesis",
                log_file)

            jail_filename = os.path.splitext(os.path.basename(jail_data_list[an]))[0]

            #prompt count storage
            if jail_prompt_counts is not None and jail_score_count > 0:
                jail_prompt_counts.setdefault(jail_filename, jail_score_count)

            #LaTeX table storage (all jailbreaks included)
            if latex_data is not None and run_key is not None:
                latex_data[jail_filename][run_key] = {
                    'auc':         float(auc),
                    'auc_low':     float(auc_low),
                    'auc_high':    float(auc_high),
                    'tpr':         float(tpr_at_best),
                    'tnr':         float(tnr_at_best),
                    'fpr':         float(fpr_at_best),
                    'fnr':         float(fnr_at_best),
                    'p_value':     float(p_value),
                    'decision':    'Reject' if reject else 'Fail',
                }
                print(f"[LATEX] Stored jail={jail_filename}, run={run_key}")

            #heatmap storage (Dutch + English pooled, all jailbreaks)
            if heatmap_auc_data is not None and heatmap_x_key is not None and run_key is not None:
                heatmap_auc_data[heatmap_x_key][jail_filename][run_key] = {
                    'auc':         float(auc),
                    'significant': reject,
                }
                print(f"[HEATMAP] Stored x={heatmap_x_key}, jail={jail_filename}, "
                      f"run={run_key}, auc={auc:.4f}, significant={reject}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser("FJD Detection")
    parser.add_argument("--model",        type=str,   default='llama2-7b')
    parser.add_argument("--data",         type=str,   default='./data/result/llama2-7b',
                        help="Experiment 1: root folder containing all affirmative-prompt "
                             "subfolders.  Experiment 2: one specific affirmative-prompt folder.")
    parser.add_argument("--n_bootstraps", type=int,   default=1000)
    parser.add_argument("--ci",           type=float, default=0.95)
    parser.add_argument("--experiment",   type=int,   default=1,
                        help="1 = all prompts, fixed temperature → one combined report; "
                             "2 = one prompt, all temperatures → one report per temperature")
    parser.add_argument("--temperature",  type=str,   default='Temperature 1.0',
                        help="(Experiment 1 only) Name of the temperature subfolder to read "
                             "from inside each Run X directory. Default: 'Temperature 1.0'")
    parser.add_argument("--exclude-refusals", action="store_true",
                        help="Also generate heatmaps with refused entries excluded from "
                             "AUC computation.")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data)
    is_experiment_1 = (args.experiment == 1)
    is_experiment_2 = (args.experiment == 2)

    # Reporting directory
    reporting_dir    = os.path.join(data_root, 'Reporting')
    os.makedirs(reporting_dir, exist_ok=True)
    root_name        = os.path.basename(data_root)
    output_file_path = os.path.join(reporting_dir, f"{root_name}_Results.txt")

    print(f"[DEBUG] data_root        = {data_root}")
    print(f"[DEBUG] experiment       = {args.experiment}")
    print(f"[DEBUG] temperature      = {args.temperature!r}  (used by Experiment 1 only)")
    print(f"[DEBUG] reporting_dir    = {reporting_dir}")
    print(f"[DEBUG] results txt      = {output_file_path}")
    print(f"[DEBUG] exclude_refusals = {args.exclude_refusals}")

    with open(output_file_path, 'w', encoding='utf-8') as log_file:

        log(f"\n{'='*60}", log_file)
        log(f"Data root:     {data_root}", log_file)
        log(f"Experiment:    {args.experiment}", log_file)
        if is_experiment_1:
            log(f"Temperature:   {args.temperature}", log_file)
        log(f"Reporting dir: {reporting_dir}", log_file)
        log(f"Results file:  {output_file_path}", log_file)
        log(f"{'='*60}", log_file)

        # ==================================================================
        # EXPERIMENT 1
        # ==================================================================
        if is_experiment_1:

            heatmap_auc_data_exp1 = defaultdict(lambda: defaultdict(dict))
            heatmap_auc_data_exp1_no_refusals = defaultdict(lambda: defaultdict(dict))
            refusal_data_exp1     = defaultdict(lambda: defaultdict(dict))
            all_run_keys_exp1     = []
            jail_prompt_counts_exp1 = {}

            prompt_folders = collect_subfolders(data_root)
            print(f"[DEBUG] affirmative prompt folders: "
                  f"{[os.path.basename(p) for p in prompt_folders]}")

            if not prompt_folders:
                log("[WARN] No affirmative-prompt subfolders found under data_root. "
                    "For Experiment 1, --data must point to the root that contains "
                    "all affirmative-prompt subfolders.", log_file)

            for prompt_folder in prompt_folders:
                prompt_name          = os.path.basename(prompt_folder)
                prompt_reporting_dir = os.path.join(prompt_folder, 'Reporting')
                os.makedirs(prompt_reporting_dir, exist_ok=True)
                prompt_log_path      = os.path.join(prompt_reporting_dir,
                                                    f"{prompt_name}_Results.txt")

                latex_data_dutch   = defaultdict(dict)
                latex_data_english = defaultdict(dict)

                with open(prompt_log_path, 'w', encoding='utf-8') as prompt_log:

                    log(f"\n{'='*60}", prompt_log)
                    log(f"Affirmative prompt: {prompt_name}", prompt_log)
                    log(f"Temperature:        {args.temperature}", prompt_log)
                    log(f"Reporting dir:      {prompt_reporting_dir}", prompt_log)
                    log(f"{'='*60}", prompt_log)

                    log(f"\n{'='*60}", log_file)
                    log(f"Processing prompt: {prompt_name}", log_file)
                    log(f"{'='*60}", log_file)

                    run_folders = collect_subfolders(prompt_folder)
                    print(f"\n[DEBUG] Prompt '{prompt_name}' — "
                          f"run folders: {[os.path.basename(r) for r in run_folders]}")

                    if not run_folders:
                        log(f"  [WARN] No Run subfolders found in '{prompt_name}' — skipping.",
                            prompt_log)
                        continue

                    for run_folder in run_folders:
                        run_name    = os.path.basename(run_folder)
                        temp_folder = find_exact_temperature_folder(run_folder, args.temperature)

                        if temp_folder is None:
                            log(f"  [WARN] '{args.temperature}' not found in "
                                f"'{prompt_name} / {run_name}' — skipping.", prompt_log)
                            continue

                        run_key = run_name

                        if run_key not in all_run_keys_exp1:
                            all_run_keys_exp1.append(run_key)

                        all_json = find_json_files(temp_folder)
                        dutch_benign, english_benign, dutch_jail, english_jail = \
                            split_by_language(all_json)

                        log(f"\n>>> Processing: {prompt_name} / {run_name}", prompt_log)
                        log(f"    Temperature folder:   {os.path.basename(temp_folder)}",
                            prompt_log)
                        log(f"    Dutch benign:         "
                            f"{[os.path.basename(f) for f in dutch_benign]}", prompt_log)
                        log(f"    Dutch jailbreaks:     "
                            f"{[os.path.basename(f) for f in dutch_jail]}", prompt_log)
                        log(f"    English benign:       "
                            f"{[os.path.basename(f) for f in english_benign]}", prompt_log)
                        log(f"    English jailbreaks:   "
                            f"{[os.path.basename(f) for f in english_jail]}", prompt_log)

                        # Dutch
                        run_evaluation(
                            dutch_benign, dutch_jail,
                            args.n_bootstraps, args.ci,
                            label=f"{prompt_name} / {run_name} [Dutch]",
                            log_file=prompt_log,
                            run_key=run_key,
                            latex_data=latex_data_dutch,
                            heatmap_auc_data=heatmap_auc_data_exp1,
                            heatmap_x_key=prompt_name,
                            jail_prompt_counts=jail_prompt_counts_exp1,
                        )

                        # English
                        run_evaluation(
                            english_benign, english_jail,
                            args.n_bootstraps, args.ci,
                            label=f"{prompt_name} / {run_name} [English]",
                            log_file=prompt_log,
                            run_key=run_key,
                            latex_data=latex_data_english,
                            heatmap_auc_data=heatmap_auc_data_exp1,
                            heatmap_x_key=prompt_name,
                            jail_prompt_counts=jail_prompt_counts_exp1,
                        )

                        # Dutch (no refusals)
                        if args.exclude_refusals:
                            run_evaluation_without_refusals(
                                dutch_benign, dutch_jail,
                                args.n_bootstraps, args.ci,
                                label=f"{prompt_name} / {run_name} [Dutch, no refusals]",
                                log_file=prompt_log,
                                heatmap_auc_data=heatmap_auc_data_exp1_no_refusals,
                                heatmap_x_key=prompt_name,
                                run_key=run_key,
                            )

                        # English (no refusals)
                        if args.exclude_refusals:
                            run_evaluation_without_refusals(
                                english_benign, english_jail,
                                args.n_bootstraps, args.ci,
                                label=f"{prompt_name} / {run_name} [English, no refusals]",
                                log_file=prompt_log,
                                heatmap_auc_data=heatmap_auc_data_exp1_no_refusals,
                                heatmap_x_key=prompt_name,
                                run_key=run_key,
                            )

                        # Refusal counts (Dutch + English pooled)
                        all_jail_files = dutch_jail + english_jail
                        all_benign_files = dutch_benign + english_benign
                        for jail_path in all_jail_files + all_benign_files:
                            jail_stem = os.path.splitext(os.path.basename(jail_path))[0]
                            n_refused = count_refusals_from_files([jail_path])
                            existing  = refusal_data_exp1[prompt_name][jail_stem].get(run_key, 0)
                            refusal_data_exp1[prompt_name][jail_stem][run_key] = existing + n_refused

                            print(
                                f"[REFUSAL] prompt={prompt_name}, jail={jail_stem}, "
                                f"run={run_key}, refused={n_refused} "
                                f"(total so far: {existing + n_refused})"
                            )

                # Per-prompt: LaTeX table
                merge_and_save_outputs(
                    latex_data_dutch=latex_data_dutch,
                    latex_data_english=latex_data_english,
                    title=prompt_name,
                    output_dir=prompt_reporting_dir,
                    ci=args.ci,
                    log_file=log_file,
                )

            # After all prompts: heatmap + refusal table
            generate_heatmap_exp1(
                heatmap_auc_data=heatmap_auc_data_exp1,
                root_name=root_name,
                reporting_dir=reporting_dir,
            )

            if args.exclude_refusals:
                generate_heatmap_exp1_without_refusals(
                    heatmap_auc_data=heatmap_auc_data_exp1_no_refusals,
                    root_name=root_name,
                    reporting_dir=reporting_dir,
                )

            generate_refusal_table_exp1(
                refusal_data=refusal_data_exp1,
                all_run_keys=all_run_keys_exp1,
                root_name=root_name,
                reporting_dir=reporting_dir,
                jail_prompt_counts=jail_prompt_counts_exp1,
            )

        # ==================================================================
        # EXPERIMENT 2
        # ==================================================================
        elif is_experiment_2:

            affirmative_name = root_name

            heatmap_auc_data_exp2 = defaultdict(lambda: defaultdict(dict))
            heatmap_auc_data_exp2_no_refusals = defaultdict(lambda: defaultdict(dict))
            refusal_data_exp2     = defaultdict(lambda: defaultdict(dict))
            all_run_keys_exp2     = []
            jail_prompt_counts_exp2 = {}

            temp_latex_data_dutch   = defaultdict(lambda: defaultdict(dict))
            temp_latex_data_english = defaultdict(lambda: defaultdict(dict))

            run_folders = collect_subfolders(data_root)
            print(f"[DEBUG] run folders: {[os.path.basename(r) for r in run_folders]}")

            if not run_folders:
                log("[WARN] No Run subfolders found. For Experiment 2, --data must point "
                    "to a single affirmative-prompt folder.", log_file)

            for run_folder in run_folders:
                run_name     = os.path.basename(run_folder)
                temp_folders = find_temperature_folders(run_folder)

                if not temp_folders:
                    log(f"  [WARN] No temperature subfolders found in '{run_name}' — skipping.",
                        log_file)
                    continue

                for temp_folder in temp_folders:
                    temp_name = os.path.basename(temp_folder)
                    run_key   = run_name

                    all_json = find_json_files(temp_folder)
                    dutch_benign, english_benign, dutch_jail, english_jail = \
                        split_by_language(all_json)

                    log(f"\n>>> Processing: {run_name} / {temp_name}", log_file)
                    log(f"    Dutch benign:         {[os.path.basename(f) for f in dutch_benign]}",
                        log_file)
                    log(f"    Dutch jailbreaks:     {[os.path.basename(f) for f in dutch_jail]}",
                        log_file)
                    log(f"    English benign:       {[os.path.basename(f) for f in english_benign]}",
                        log_file)
                    log(f"    English jailbreaks:   {[os.path.basename(f) for f in english_jail]}",
                        log_file)

                    # Dutch
                    run_evaluation(
                        dutch_benign, dutch_jail,
                        args.n_bootstraps, args.ci,
                        label=f"{run_name} / {temp_name} [Dutch]",
                        log_file=log_file,
                        run_key=run_key,
                        latex_data=temp_latex_data_dutch[temp_name],
                        heatmap_auc_data=heatmap_auc_data_exp2,
                        heatmap_x_key=temp_name,
                        jail_prompt_counts=jail_prompt_counts_exp2,
                    )

                    # English
                    run_evaluation(
                        english_benign, english_jail,
                        args.n_bootstraps, args.ci,
                        label=f"{run_name} / {temp_name} [English]",
                        log_file=log_file,
                        run_key=run_key,
                        latex_data=temp_latex_data_english[temp_name],
                        heatmap_auc_data=heatmap_auc_data_exp2,
                        heatmap_x_key=temp_name,
                        jail_prompt_counts=jail_prompt_counts_exp2,
                    )

                    # Dutch (no refusals)
                    if args.exclude_refusals:
                        run_evaluation_without_refusals(
                            dutch_benign, dutch_jail,
                            args.n_bootstraps, args.ci,
                            label=f"{run_name} / {temp_name} [Dutch, no refusals]",
                            log_file=log_file,
                            heatmap_auc_data=heatmap_auc_data_exp2_no_refusals,
                            heatmap_x_key=temp_name,
                            run_key=run_key,
                        )

                    # English (no refusals)
                    if args.exclude_refusals:
                        run_evaluation_without_refusals(
                            english_benign, english_jail,
                            args.n_bootstraps, args.ci,
                            label=f"{run_name} / {temp_name} [English, no refusals]",
                            log_file=log_file,
                            heatmap_auc_data=heatmap_auc_data_exp2_no_refusals,
                            heatmap_x_key=temp_name,
                            run_key=run_key,
                        )

                    # Refusal counts (Dutch + English pooled)
                    if run_key not in all_run_keys_exp2:
                        all_run_keys_exp2.append(run_key)

                    all_jail_files = dutch_jail + english_jail
                    all_benign_files = dutch_benign + english_benign

                    for jail_path in all_jail_files + all_benign_files:
                        jail_stem = os.path.splitext(os.path.basename(jail_path))[0]
                        n_refused = count_refusals_from_files([jail_path])
                        existing  = refusal_data_exp2[temp_name][jail_stem].get(run_key, 0)
                        refusal_data_exp2[temp_name][jail_stem][run_key] = existing + n_refused

                        print(
                            f"[REFUSAL EXP2] temp={temp_name}, jail={jail_stem}, "
                            f"run={run_key}, refused={n_refused} "
                            f"(total so far: {existing + n_refused})"
                        )

            # Per-temperature: LaTeX table
            all_temp_names = sorted(set(
                list(temp_latex_data_dutch.keys())
                + list(temp_latex_data_english.keys())
            ))
            print(f"[DEBUG] temperatures found: {all_temp_names}")

            if not all_temp_names:
                print("[WARN] No data collected — nothing generated.")

            for temp_name in all_temp_names:
                temp_reporting_dir = os.path.join(reporting_dir, temp_name)

                merge_and_save_outputs(
                    latex_data_dutch=temp_latex_data_dutch.get(temp_name, defaultdict(dict)),
                    latex_data_english=temp_latex_data_english.get(temp_name, defaultdict(dict)),
                    title=f"{affirmative_name}_{temp_name}",
                    output_dir=temp_reporting_dir,
                    ci=args.ci,
                    log_file=log_file,
                )

            # After all temperatures: heatmap + refusal table
            generate_heatmap_exp2(
                heatmap_auc_data=heatmap_auc_data_exp2,
                root_name=root_name,
                reporting_dir=reporting_dir,
            )

            if args.exclude_refusals:
                generate_heatmap_exp2_without_refusals(
                    heatmap_auc_data=heatmap_auc_data_exp2_no_refusals,
                    root_name=root_name,
                    reporting_dir=reporting_dir,
                )

            generate_refusal_table_exp2(
                refusal_data=refusal_data_exp2,
                all_run_keys=all_run_keys_exp2,
                root_name=root_name,
                reporting_dir=reporting_dir,
                jail_prompt_counts=jail_prompt_counts_exp2,
            )

        else:
            log(f"[ERROR] Unknown experiment number: {args.experiment}. Use 1 or 2.", log_file)

    print(f"\nResults saved to: {output_file_path}")
