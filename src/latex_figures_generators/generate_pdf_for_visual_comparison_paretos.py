"""
===============================================================================
File: generate_pdf_for_visual_comparison_paretos.py
Description: Standalone script to generate a horizontal multi-panel PDF comparing the Pareto front evolution across experiments.
Role: Plotting Script
Execution: python src/latex_figures_generators/generate_pdf_for_visual_comparison_paretos.py
Inputs: aggregation_history.pkl and history_submatrix_traditional.pkl files.
Outputs: pareto_comparison_horizontal.pdf file in open_results/article_data.

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
from matplotlib.lines import Line2D

# Ensure we can import the project's local modules from src if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_dir, "..")))

def main():
    # Configurable variable for the results directory path
    RESULTS_DIR_NAME = "../../open_results/article_data"
    
    # Paths to the experiments directories
    results_article_dir = os.path.abspath(os.path.join(base_dir, RESULTS_DIR_NAME))
    
    experiments = {
        'A': {
            'dir': os.path.join(results_article_dir, "A"),
            'title': "Experiment A"
        },
        'B': {
            'dir': os.path.join(results_article_dir, "B"),
            'title': "Experiment B"
        },
        'C': {
            'dir': os.path.join(results_article_dir, "C"),
            'title': "Experiment C"
        }
    }
    
    # Check if the directories exist
    for exp_name, exp_info in experiments.items():
        if not os.path.exists(exp_info['dir']):
            print(f"Error: Directory not found: {exp_info['dir']}")
            return
            
    # Setup Figure and Subplots
    # 1 row, 3 columns
    # We want a wide figure suitable for paper width
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Styling details: high-contrast colors and publication-quality palette
    # Use Blues colormap for Decomposition (DC-GA) and Reds colormap for Traditional (Global)
    cmap_dec = plt.get_cmap('Blues')
    cmap_trad = plt.get_cmap('Reds')
    
    # We will loop through the experiments and plot each in a subplot
    for idx, exp_name in enumerate(['A', 'B', 'C']):
        exp_info = experiments[exp_name]
        ax = axes[idx]
        
        # Load the pickle files
        agg_file = os.path.join(exp_info['dir'], "aggregation_history.pkl")
        trad_file = os.path.join(exp_info['dir'], "history_submatrix_traditional.pkl")
        
        if not os.path.exists(agg_file) or not os.path.exists(trad_file):
            print(f"Error: Missing pickle files in {exp_info['dir']}")
            continue
            
        with open(agg_file, "rb") as f:
            aggregation_history = pickle.load(f)
            
        with open(trad_file, "rb") as f:
            traditional_history = pickle.load(f)
            
        n_steps = len(aggregation_history)
        generations_per_step = (len(traditional_history) - 1) // n_steps if len(traditional_history) > 1 else 1
        
        # Determine the root node ID dynamically (it will be the maximum integer key in the last step)
        last_step_fronts = aggregation_history[-1]
        root_node_id = max(k for k in last_step_fronts.keys() if isinstance(k, int))
        
        # Plot Decomposition (DC-GA) steps
        for step_idx, step_fronts in enumerate(aggregation_history):
            pareto_front = step_fronts.get(root_node_id, [])
            if not pareto_front:
                continue
            x = [ind['fitness'][0] for ind in pareto_front]
            y = [ind['fitness'][1] for ind in pareto_front]
            
            # Map steps to colormap range [0.35, 1.0] for visibility
            c_val = 0.35 + 0.65 * (step_idx / max(1, n_steps - 1))
            color = cmap_dec(c_val)
            
            # For the last step, make it slightly larger and solid
            is_last = (step_idx == n_steps - 1)
            size = 75 if is_last else 30
            alpha = 0.95 if is_last else 0.4
            linewidth = 0.8 if is_last else 0.0
            edgecolor = 'black' if is_last else 'none'
            label = 'DC-GA' if (is_last and idx == 0) else None
            
            ax.scatter(x, y, color=color, marker='o', alpha=alpha, s=size, 
                       edgecolors=edgecolor, linewidths=linewidth, zorder=3, label=label)
            
        # Plot Traditional (Global) steps
        for step_idx in range(n_steps):
            gen_idx = (step_idx + 1) * generations_per_step
            if gen_idx < len(traditional_history):
                pareto_front = traditional_history[gen_idx]['pareto_front']
                if not pareto_front:
                    continue
                x = [ind['fitness'][0] for ind in pareto_front]
                y = [ind['fitness'][1] for ind in pareto_front]
                
                # Map steps to colormap range [0.35, 1.0] for visibility
                c_val = 0.35 + 0.65 * (step_idx / max(1, n_steps - 1))
                color = cmap_trad(c_val)
                
                is_last = (step_idx == n_steps - 1)
                size = 75 if is_last else 30
                alpha = 0.95 if is_last else 0.4
                linewidth = 0.8 if is_last else 0.0
                edgecolor = 'black' if is_last else 'none'
                label = 'Global' if (is_last and idx == 0) else None
                
                ax.scatter(x, y, color=color, marker='^', alpha=alpha, s=size,
                           edgecolors=edgecolor, linewidths=linewidth, zorder=2, label=label)
                
        # Set titles and label sizes (large for JCR article readability)
        ax.set_title(exp_info['title'], fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('Mean Response Time (latency)', fontsize=14, labelpad=10)
        if idx == 0:
            ax.set_ylabel('Total Price (units)', fontsize=14, labelpad=10)
        
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='major', labelsize=12, labelrotation=90)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        ax.margins(0.08)
        
    # Extract the legend handles and labels directly from the first subplot to ensure perfect color/alpha/shape match
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Place the legend at the bottom of the figure, in a single horizontal row
    fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=2, fontsize=14, frameon=True, facecolor='white', edgecolor='lightgray')
    
    # Adjust layout to make room for the bottom legend and reduce spacing between subplots
    plt.tight_layout()
    # wspace=0.09 is the perfect sweet spot for close but separated subplots!
    plt.subplots_adjust(bottom=0.18, wspace=0.09)
    
    # Save the output PDF in results_article
    output_pdf = os.path.join(results_article_dir, "pareto_comparison_horizontal.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Comparison PDF successfully saved at: {output_pdf}")

if __name__ == "__main__":
    main()
