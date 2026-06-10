"""
===============================================================================
File: generate_pdf_for_bloxplot.py
Description: Standalone script to generate JCR-ready publication-quality boxplot PDFs comparing multiple algorithmic metrics.
Role: Plotting Script
Execution: python src/latex_figures_generators/generate_pdf_for_bloxplot.py
Inputs: global_metrics.pkl files from different experiments.
Outputs: boxplot_metrics_[A|B|C].pdf files in open_results/article_data.

Authors: Carlos Guerrero and Isaac Lera
Date: June 2026
License: MIT License
===============================================================================
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ensure we can import the project's local modules from src if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_dir, "..")))

from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD

def get_non_dominated(points):
    if not points:
        return []
    is_dominated = np.zeros(len(points), dtype=bool)
    for i, p1 in enumerate(points):
        f1 = p1['fitness']
        for j, p2 in enumerate(points):
            if i == j: continue
            f2 = p2['fitness']
            if f2[0] <= f1[0] and f2[1] <= f1[1] and (f2[0] < f1[0] or f2[1] < f1[1]):
                is_dominated[i] = True
                break
    return [p for i, p in enumerate(points) if not is_dominated[i]]

def calculate_spacing(fitness_list):
    if len(fitness_list) < 2: return 0.0
    arr = np.array(fitness_list)
    distances = []
    for i in range(len(arr)):
        dists = np.sum(np.abs(arr - arr[i]), axis=1)
        dists[i] = np.inf
        distances.append(np.min(dists))
    d_mean = np.mean(distances)
    s = np.sqrt(np.sum((distances - d_mean)**2) / (len(distances) - 1))
    return s
    
def calculate_spacing_to_extent(fitness_list, s_val):
    if len(fitness_list) < 2: return float('inf')
    arr = np.array(fitness_list)
    max_dists = np.max(arr, axis=0) - np.min(arr, axis=0)
    extent = np.sum(max_dists)
    if extent == 0: return float('inf')
    return s_val / extent
    
def calculate_set_coverage(front_A, front_B):
    if len(front_A) == 0 or len(front_B) == 0: return 0.0
    dominated_count = 0
    for b in front_B:
        for a in front_A:
            if a[0] <= b[0] and a[1] <= b[1]:
                dominated_count += 1
                break
    return dominated_count / len(front_B)

def generate_boxplots_for_exp(exp_name, exp_dir, results_article_dir):
    pkl_path = os.path.join(exp_dir, "global_metrics.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return
        
    print(f"Processing {exp_name} from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
    all_decomp_fronts = data.get('all_decomp_fronts', [])
    all_trad_fronts = data.get('all_trad_fronts', [])
    
    # 1. Compute global reference front
    all_solutions = []
    for df in all_decomp_fronts:
        if df: all_solutions.extend(df)
    for tf in all_trad_fronts:
        if tf: all_solutions.extend(tf)
        
    global_ref_front = get_non_dominated(all_solutions)
    if not global_ref_front:
        print("No solutions found for global reference front.")
        return
        
    global_ref_fit = [ind['fitness'] for ind in global_ref_front]
    all_fit_arr = np.array(global_ref_fit)
    f_min = np.min(all_fit_arr, axis=0)
    f_max = np.max(all_fit_arr, axis=0)
    if np.all(f_min == f_max):
        f_max = f_min + 1.0
        
    def normalize(fitness_list):
        return [(np.array(fit) - f_min) / (f_max - f_min) for fit in fitness_list]
        
    norm_ref = normalize(global_ref_fit)
    ref_point = np.ones(2) * 1.1
    
    hv_ind = HV(ref_point=ref_point)
    gd_ind = GD(np.array(norm_ref))
    igd_ind = IGD(np.array(norm_ref))
    
    stats = {
        'hv_decomp': [], 'hv_trad': [],
        'spacing_decomp': [], 'spacing_trad': [],
        'ste_decomp': [], 'ste_trad': [],
        'gd_decomp': [], 'gd_trad': [],
        'igd_decomp': [], 'igd_trad': [],
        'c_decomp_trad': [], 'c_trad_decomp': []
    }
    
    for i in range(len(all_decomp_fronts)):
        df = all_decomp_fronts[i]
        tf = all_trad_fronts[i]
        
        df_fit = [ind['fitness'] for ind in df] if df else []
        tf_fit = [ind['fitness'] for ind in tf] if tf else []
        
        norm_df = normalize(df_fit) if df_fit else []
        norm_tf = normalize(tf_fit) if tf_fit else []
        
        stats['hv_decomp'].append(hv_ind(np.array(norm_df)) if norm_df else 0.0)
        stats['hv_trad'].append(hv_ind(np.array(norm_tf)) if norm_tf else 0.0)
        
        stats['gd_decomp'].append(gd_ind(np.array(norm_df)) if norm_df else float('inf'))
        stats['gd_trad'].append(gd_ind(np.array(norm_tf)) if norm_tf else float('inf'))
        
        stats['igd_decomp'].append(igd_ind(np.array(norm_df)) if norm_df else float('inf'))
        stats['igd_trad'].append(igd_ind(np.array(norm_tf)) if norm_tf else float('inf'))
        
        s_d = calculate_spacing(df_fit)
        stats['spacing_decomp'].append(s_d)
        stats['ste_decomp'].append(calculate_spacing_to_extent(df_fit, s_d))
        
        s_t = calculate_spacing(tf_fit)
        stats['spacing_trad'].append(s_t)
        stats['ste_trad'].append(calculate_spacing_to_extent(tf_fit, s_t))
        
        stats['c_decomp_trad'].append(calculate_set_coverage(df_fit, tf_fit))
        stats['c_trad_decomp'].append(calculate_set_coverage(tf_fit, df_fit))
        
    # 2. Setup Figure and Subplots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('Hypervolume (HV) ↑', 'hv_decomp', 'hv_trad'),
        ('Spacing (S) ↓', 'spacing_decomp', 'spacing_trad'),
        ('Spacing-to-Extent (STE) ↓', 'ste_decomp', 'ste_trad'),
        ('Generational Distance (GD) ↓', 'gd_decomp', 'gd_trad'),
        ('Inverted Generational Distance (IGD) ↓', 'igd_decomp', 'igd_trad'),
        ('Set Coverage C(A,B) ↑', 'c_decomp_trad', 'c_trad_decomp')
    ]
    
    # Load colormaps to get the exact same soft colors as in the scatter plot
    cmap_dec = plt.get_cmap('Blues')
    cmap_trad = plt.get_cmap('Reds')
    # Use a softer, lighter shade (0.6) from the colormaps to match the scatter plots
    colors = [cmap_dec(0.6), cmap_trad(0.6)]
    labels = ['DC-GA', 'Global']      # Consistent text labels
    
    for idx, (title, key_dec, key_trad) in enumerate(metrics_to_plot):
        ax = axes[idx]
        data = [stats[key_dec], stats[key_trad]]
        
        # Remove inf values for clean plotting
        data = [[x for x in arr if x != float('inf')] for arr in data]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.45)
        
        # Premium styling
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)  # Slightly transparent for a softer, premium look
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.2)
            
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.2)
            
        for median in bp['medians']:
            median.set_color('black')  # Clean black line for JCR editorial standards
            median.set_linewidth(1.8)
            
        for flier in bp['fliers']:
            flier.set_marker('o')
            flier.set_markerfacecolor('gray')
            flier.set_alpha(0.4)
            flier.set_markersize(5)
            
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        
    plt.tight_layout()
    
    # Save the output PDF in results_article
    output_pdf = os.path.join(results_article_dir, f"boxplot_metrics_{exp_name}.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot PDF successfully at: {output_pdf}")

def main():
    # Configurable variable for the results directory path
    RESULTS_DIR_NAME = "../../open_results/article_data"
    
    results_article_dir = os.path.abspath(os.path.join(base_dir, RESULTS_DIR_NAME))
    
    for exp_name in ['A', 'B', 'C']:
        exp_dir = os.path.join(results_article_dir, exp_name)
        if os.path.exists(exp_dir):
            generate_boxplots_for_exp(exp_name, exp_dir, results_article_dir)

if __name__ == "__main__":
    main()
