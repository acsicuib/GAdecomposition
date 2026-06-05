"""
===============================================================================
File: results_manager.py
Description: Handles the storage, retrieval, plotting, and statistical analysis (e.g., Wilcoxon tests, Pareto comparisons) of the simulation metrics.
Role: Class Definition (Utility)
Execution: Imported as a module
Inputs: Simulation runs metrics, Fitness lists.
Outputs: Pickle logs (global_metrics.pkl), PDFs (boxplots, pareto fronts).

Authors: Carlos Guerrero and Isaac Lera
Date: June 2026
License: MIT License
===============================================================================
"""
import os
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import scipy.stats

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

class ResultsManager:
    def __init__(self, config, load_dir=None):
        self.config = config
        self.experiment_id = getattr(config, 'experiment_id', 'experiment')
        
        if load_dir:
            self.base_output_dir = load_dir
            self.output_dir = load_dir
        else:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.base_output_dir = f"../results/{self.experiment_id}_{self.timestamp}"
            self.output_dir = self.base_output_dir
            
            os.makedirs(self.base_output_dir, exist_ok=True)
            
            config_path = getattr(config, '_config_path', '../config/simulation_config.json')
            if os.path.exists(config_path):
                import shutil
                shutil.copy(config_path, os.path.join(self.base_output_dir, "simulation_config.json"))
                # If custom name was used, also backup with the original filename
                orig_name = os.path.basename(config_path)
                if orig_name != "simulation_config.json":
                    shutil.copy(config_path, os.path.join(self.base_output_dir, orig_name))

            
        self.all_decomp_fronts = []
        self.all_trad_fronts = []
        self.all_metrics = []
        
    def set_current_run(self, run_index):
        self.current_run = run_index
        self.output_dir = os.path.join(self.base_output_dir, f"run_{run_index}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def collect_run_data(self, decomp_front, trad_front, local_nsga_times, aggregation_merge_times, traditional_nsga_time):
        self.all_decomp_fronts.append(decomp_front)
        self.all_trad_fronts.append(trad_front)
        
        max_local_nsga_time = max(local_nsga_times.values()) if local_nsga_times else 0.0
        max_agg_time = max(aggregation_merge_times.values()) if aggregation_merge_times else 0.0
        total_parallel_decomp_time = max_local_nsga_time + max_agg_time
        speedup = traditional_nsga_time / total_parallel_decomp_time if total_parallel_decomp_time > 0 else 0.0
        
        self.all_metrics.append({
            'run': self.current_run,
            'max_local_nsga_time': max_local_nsga_time,
            'max_agg_time': max_agg_time,
            'total_parallel_decomp_time': total_parallel_decomp_time,
            'traditional_nsga_time': traditional_nsga_time,
            'speedup': speedup
        })
        
    def _compute_global_reference_front(self):
        all_solutions = []
        for df in self.all_decomp_fronts:
            if df: all_solutions.extend(df)
        for tf in self.all_trad_fronts:
            if tf: all_solutions.extend(tf)
        return get_non_dominated(all_solutions)
        
    def save_global_metrics(self):
        file_path = os.path.join(self.base_output_dir, "global_metrics.pkl")
        with open(file_path, "wb") as f:
            pickle.dump({
                'all_metrics': self.all_metrics,
                'all_decomp_fronts': self.all_decomp_fronts,
                'all_trad_fronts': self.all_trad_fronts
            }, f)
            
    def load_global_metrics(self):
        file_path = os.path.join(self.base_output_dir, "global_metrics.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.all_metrics = data.get('all_metrics', [])
                self.all_decomp_fronts = data.get('all_decomp_fronts', [])
                self.all_trad_fronts = data.get('all_trad_fronts', [])
        else:
            print(f"Warning: No global_metrics.pkl found in {self.base_output_dir}")
        
    def save_history(self, node_id, nsga_instance):
        """Guarda el histórico de poblaciones de una instancia NSGA2 usando pickle."""
        file_path = os.path.join(self.output_dir, f"history_submatrix_{node_id}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(nsga_instance.history, f)
            
    def save_aggregation_history(self, agg_history):
        """Guarda el histórico de las agregaciones del HC-PA usando pickle."""
        file_path = os.path.join(self.output_dir, "aggregation_history.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(agg_history, f)
            
    def plot_global_pareto_evolution(self, aggregation_history, root_node_id, traditional_history=None):
        """
        Genera un scatter plot mostrando la evolución del frente de Pareto
        del nodo raíz (solución global) a lo largo de los steps del HC-PA,
        y opcionalmente la del modo tradicional.
        """
        plt.figure(figsize=(12, 9))
        n_steps = len(aggregation_history)
        
        # Colormap para Decomposition (tonos fríos/azulados/verdosos)
        cmap_dec = plt.get_cmap('viridis')
        # Colormap para Traditional (tonos cálidos/rojizos/anaranjados)
        cmap_trad = plt.get_cmap('plasma')
        
        # Primero, Decomposition
        for step_idx, step_fronts in enumerate(aggregation_history):
            pareto_front = step_fronts.get(root_node_id, [])
            if not pareto_front:
                continue
            x = [ind['fitness'][0] for ind in pareto_front] # Obj 1: Mean Response Time
            y = [ind['fitness'][1] for ind in pareto_front] # Obj 2: Total Price
            
            color = cmap_dec(step_idx / max(1, n_steps - 1))
            label = f'Paso {step_idx + 1} (Decomp.)'
            plt.scatter(x, y, color=color, marker='o', alpha=0.7, edgecolors='none', s=50, label=label)
            
        # Segundo, Traditional
        if traditional_history is not None:
            for step_idx in range(n_steps):
                gen_idx = (step_idx + 1) * self.config.generations
                if gen_idx < len(traditional_history):
                    pareto_front = traditional_history[gen_idx]['pareto_front']
                    if not pareto_front:
                        continue
                    x = [ind['fitness'][0] for ind in pareto_front]
                    y = [ind['fitness'][1] for ind in pareto_front]
                    
                    color = cmap_trad(step_idx / max(1, n_steps - 1))
                    label = f'Paso {step_idx + 1} (Trad.)'
                    plt.scatter(x, y, color=color, marker='^', alpha=0.7, edgecolors='none', s=50, label=label)
                    
        plt.xlabel('Mean Response Time (latency)')
        plt.ylabel('Total Price (units)')
        plt.title('Global Pareto Front Evolution: Decomposition vs Traditional Mode')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        pdf_path = os.path.join(self.output_dir, "pareto_evolution_global.pdf")
        plt.savefig(pdf_path, format='pdf')
        plt.close()
        
    def generate_global_solutions_reports(self, aggregation_history, root_node_id, visualizer):
        """Genera archivos de texto legibles y PDFs para el frente de Pareto global del último step."""
        if not aggregation_history:
            return
            
        last_step_fronts = aggregation_history[-1]
        pareto_front = last_step_fronts.get(root_node_id, [])
        
        if not pareto_front:
            return
            
        # 1. Generar archivo de texto legible
        txt_path = os.path.join(self.output_dir, "global_pareto_solutions.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Frente Pareto Global - Último Step\n")
            f.write("="*50 + "\n\n")
            
            for idx, ind in enumerate(pareto_front):
                sol = ind['solution']
                temp = ind['template']
                fit = ind['fitness']
                
                f.write(f"Solución {idx + 1}:\n")
                f.write(f"  Objetivo 1 (Mean Response Time): {fit[0]:.4f}\n")
                f.write(f"  Objetivo 2 (Total Price): {fit[1]:.4f}\n")
                f.write("  Asignaciones:\n")
                
                for i, node in enumerate(temp['rows']):
                    hosted = []
                    for j, srv in enumerate(temp['services']):
                        if sol[i, j] == 1:
                            hosted.append(srv)
                    
                    if hosted:
                        srv_str = ", ".join([f"App_{s}" for s in hosted])
                        f.write(f"    - Nodo {node}: {srv_str}\n")
                        
                f.write("\n" + "-"*50 + "\n\n")
                
        # 2. Generar un PDF para cada solución
        solutions_dir = os.path.join(self.output_dir, "global_solutions_pdfs")
        os.makedirs(solutions_dir, exist_ok=True)
        
        for idx, ind in enumerate(pareto_front):
            sol = ind['solution']
            temp = ind['template']
            fit = ind['fitness']
            
            pdf_path = os.path.join(solutions_dir, f"solution_{idx + 1}.pdf")
            title = f"Global Solution {idx + 1} | Latency: {fit[0]:.2f} | Price: {fit[1]:.2f}"
            visualizer.draw_solution(sol, temp, pdf_path, title)
            
    def plot_pareto_evolution(self, node_id, history, subtitle=""):
        """
        Genera un scatter plot mostrando la evolución del frente de Pareto
        a lo largo de las generaciones para una submatriz específica.
        El resultado se guarda como PDF.
        """
        n_generations = len(history)
        
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('viridis')
        
        for gen_idx, gen_data in enumerate(history):
            pareto_front = gen_data['pareto_front']
                    
            if not pareto_front:
                continue
                
            x = [ind['fitness'][0] for ind in pareto_front] # Obj 1: Mean Response Time
            y = [ind['fitness'][1] for ind in pareto_front] # Obj 2: Total Price
            
            # Gradiente de color en función de la generación actual
            color = cmap(gen_idx / max(1, n_generations - 1))
            
            # Añadir labels solo para algunas generaciones en la leyenda
            label = ""
            if gen_idx % max(1, n_generations // 5) == 0 or gen_idx == n_generations - 1:
                label = f'Gen {gen_idx}'
                
            plt.scatter(x, y, color=color, alpha=0.7, label=label)
            
        plt.xlabel('Mean Response Time (latency)')
        plt.ylabel('Total Price (units)')
        
        plt.title(f'Pareto Front Evolution - Submatrix {node_id}\n{subtitle}')
        
        # Eliminar posibles etiquetas duplicadas en la leyenda
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True)
        pdf_path = os.path.join(self.output_dir, f"pareto_evolution_submatrix_{node_id}.pdf")
        plt.savefig(pdf_path, format='pdf')
        plt.close()

    def generate_markdown_summary(self, infra, workload):
        md = "# Simulation Summary\n\n"
        
        md += "## Node Capacities and Costs\n"
        md += "| Node ID | Available Capacity | Cost |\n"
        md += "|---|---|---|\n"
        for node in sorted(infra.graph.nodes()):
            cap = infra.node_capacities[node]
            price = infra.node_prices.get(node, 0)
            cap_str = "∞ (Unlimited)" if cap == float('inf') else f"{cap} units"
            tag = "☁️ CLOUD" if node == infra.cloud_node else ""
            md += f"| Node {node} | {cap_str} | {price} {tag} |\n"
            
        md += "\n## Application Requirements and Loads\n"
        md += "| Application ID | Resource Demand (Capacity) | Execution Load (Cost) |\n"
        md += "|---|---|---|\n"
        for app_id, demand in sorted(workload.app_demands.items()):
            exec_load = workload.app_execution_loads.get(app_id, 0.0)
            md += f"| App_{app_id} | {demand} | {exec_load:.2f} |\n"
            
        md += "\n## Connected Users\n"
        md += "| User | Associated Node | Requested Application | Request Ratio |\n"
        md += "|---|---|---|---|\n"
        user_count = 1
        for node, user_requests in workload.user_assignments.items():
            for req in user_requests:
                md += f"| User_{user_count} | Node {node} | App_{req['app_id']} | {req['request_ratio']:.2f} |\n"
                user_count += 1
                
        md += "\n## Affinity Vectors (V) per Node\n"
        md += "| Node ID | V Vector (App Presence) |\n"
        md += "|---|---|\n"
        for node in sorted(infra.graph.nodes()):
            vector = workload.node_vectors[node]
            v_str = "[" + " ".join([str(int(x)) for x in vector]) + "]"
            tag = "☁️ CLOUD" if node == infra.cloud_node else ""
            md += f"| Node {node} | {v_str} {tag} |\n"
            
        file_path = os.path.join(self.output_dir, "simulation_summary.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)


    def generate_global_statistical_analysis(self):
        """
        Ejecuta el análisis estadístico usando el frente de referencia global sobre
        todas las ejecuciones, genera gráficas (boxplots) y un reporte Markdown.
        """
        self.output_dir = self.base_output_dir # Volvemos al directorio raíz
        
        global_ref_front = self._compute_global_reference_front()
        if not global_ref_front:
            print("No solutions found to generate the global front.")
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
        
        from pymoo.indicators.hv import HV
        from pymoo.indicators.gd import GD
        from pymoo.indicators.igd import IGD
        
        hv_ind = HV(ref_point=ref_point)
        gd_ind = GD(np.array(norm_ref))
        igd_ind = IGD(np.array(norm_ref))
        
        stats = {
            'hv_decomp': [], 'hv_trad': [],
            'spacing_decomp': [], 'spacing_trad': [],
            'ste_decomp': [], 'ste_trad': [],
            'gd_decomp': [], 'gd_trad': [],
            'igd_decomp': [], 'igd_trad': [],
            'c_decomp_trad': [], 'c_trad_decomp': [],
            'speedup': [],
            'speedup_nsga_only': []
        }
        
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
            
        for i in range(len(self.all_decomp_fronts)):
            df = self.all_decomp_fronts[i]
            tf = self.all_trad_fronts[i]
            
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
            
            stats['speedup'].append(self.all_metrics[i]['speedup'])
            
            t_nsga = self.all_metrics[i]['max_local_nsga_time']
            t_trad = self.all_metrics[i]['traditional_nsga_time']
            speedup_nsga = t_trad / t_nsga if t_nsga > 0 else 0.0
            stats['speedup_nsga_only'].append(speedup_nsga)
            
        self._generate_boxplots(stats)
        self._generate_statistical_markdown(stats)
        self._generate_optimization_evolution_plot()

    def _generate_boxplots(self, stats):
        metrics_to_plot = [
            ('Hypervolume (HV) ↑', 'hv_decomp', 'hv_trad'),
            ('Spacing (S) ↓', 'spacing_decomp', 'spacing_trad'),
            ('Spacing-to-Extent (STE) ↓', 'ste_decomp', 'ste_trad'),
            ('Generational Distance (GD) ↓', 'gd_decomp', 'gd_trad'),
            ('Inverted Generational Distance (IGD) ↓', 'igd_decomp', 'igd_trad'),
            ('Set Coverage C(A,B) ↑', 'c_decomp_trad', 'c_trad_decomp')
        ]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        axes = axes.flatten()
        
        for idx, (title, key_dec, key_trad) in enumerate(metrics_to_plot):
            ax = axes[idx]
            data = [stats[key_dec], stats[key_trad]]
            labels = ['Decomposition', 'Traditional']
            
            data = [[x for x in arr if x != float('inf')] for arr in data]
            
            ax.boxplot(data, labels=labels, patch_artist=True)
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, "statistical_boxplots_metrics.pdf"))
        plt.close()
        
        plt.figure(figsize=(8, 6))
        data_speedup = [stats['speedup'], stats['speedup_nsga_only']]
        labels_speedup = ['Total Speedup (with Aggregation)', 'Genetic Speedup (Only NSGA-II)']
        plt.boxplot(data_speedup, labels=labels_speedup, patch_artist=True)
        plt.title("Simulated Parallel Speedup")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.base_output_dir, "statistical_boxplot_speedup.pdf"))
        plt.close()

    def _generate_statistical_markdown(self, stats):
        md = f"# Global Statistical Analysis Report ({self.config.n_runs} Runs)\n\n"
        md += "This report statistically analyzes the performance of Decomposition mode vs Traditional mode.\n\n"
        
        md += "## Performance Metrics (Mean, Median, Standard Deviation and Wilcoxon)\n\n"
        md += "| Metric | Mean Decomp. | Median Decomp. | Std. Dev. Decomp. | Mean Trad. | Median Trad. | Std. Dev. Trad. | p-value (Wilcoxon) | Significance (α=0.05) |\n"
        md += "|---|---|---|---|---|---|---|---|---|\n"
        
        comparisons = [
            ('Hypervolume', 'hv_decomp', 'hv_trad', 'greater'),
            ('Spacing', 'spacing_decomp', 'spacing_trad', 'less'),
            ('STE', 'ste_decomp', 'ste_trad', 'less'),
            ('GD', 'gd_decomp', 'gd_trad', 'less'),
            ('IGD', 'igd_decomp', 'igd_trad', 'less'),
            ('Coverage', 'c_decomp_trad', 'c_trad_decomp', 'greater')
        ]
        
        for name, k_dec, k_trad, _ in comparisons:
            d_arr = np.array(stats[k_dec])
            t_arr = np.array(stats[k_trad])
            
            valid_idx = (d_arr != float('inf')) & (t_arr != float('inf'))
            d_valid = d_arr[valid_idx]
            t_valid = t_arr[valid_idx]
            
            mean_d = np.mean(d_valid) if len(d_valid) > 0 else 0.0
            median_d = np.median(d_valid) if len(d_valid) > 0 else 0.0
            std_d = np.std(d_valid) if len(d_valid) > 0 else 0.0
            
            mean_t = np.mean(t_valid) if len(t_valid) > 0 else 0.0
            median_t = np.median(t_valid) if len(t_valid) > 0 else 0.0
            std_t = np.std(t_valid) if len(t_valid) > 0 else 0.0
            
            try:
                res = scipy.stats.wilcoxon(d_valid, t_valid)
                p_val = res.pvalue
                sig = "✅ Significant" if p_val < 0.05 else "❌ Not Signif."
            except Exception as e:
                p_val = float('nan')
                sig = "Error (equal vals)"
                
            md += f"| **{name}** | {mean_d:.6f} | {median_d:.6f} | {std_d:.6f} | {mean_t:.6f} | {median_t:.6f} | {std_t:.6f} | {p_val:.4e} | {sig} |\n"
            
        md += "\n## Execution Times Analysis and Statistical Tests\n\n"
        md += "The following details the total execution times (in seconds), the obtained Speedup "
        md += "and the statistical comparisons using the Wilcoxon signed-rank test.\n\n"
        
        # Extracción y cálculo de estadísticas de tiempos
        local_nsga_times = np.array([m['max_local_nsga_time'] for m in self.all_metrics])
        agg_times = np.array([m['max_agg_time'] for m in self.all_metrics])
        decomp_total_times = np.array([m['total_parallel_decomp_time'] for m in self.all_metrics])
        trad_total_times = np.array([m['traditional_nsga_time'] for m in self.all_metrics])
        
        mean_nsga, median_nsga, std_nsga = np.mean(local_nsga_times), np.median(local_nsga_times), np.std(local_nsga_times)
        mean_agg, median_agg, std_agg = np.mean(agg_times), np.median(agg_times), np.std(agg_times)
        mean_dec, median_dec, std_dec = np.mean(decomp_total_times), np.median(decomp_total_times), np.std(decomp_total_times)
        mean_trad, median_trad, std_trad = np.mean(trad_total_times), np.median(trad_total_times), np.std(trad_total_times)
        
        md += "### Total Times Summary (seconds)\n\n"
        md += "| Component | Mean (s) | Median (s) | Std. Dev. (s) |\n"
        md += "|---|---|---|---|\n"
        md += f"| **Decomposition (Parallel Total)** | {mean_dec:.4f} s | {median_dec:.4f} s | {std_dec:.4f} s |\n"
        md += f"| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest NSGA-II* | {mean_nsga:.4f} s | {median_nsga:.4f} s | {std_nsga:.4f} s |\n"
        md += f"| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest Aggregation* | {mean_agg:.4f} s | {median_agg:.4f} s | {std_agg:.4f} s |\n"
        md += f"| **Traditional (Sequential Total)** | {mean_trad:.4f} s | {median_trad:.4f} s | {std_trad:.4f} s |\n\n"
        
        # Pruebas de Wilcoxon para tiempos
        try:
            res_total = scipy.stats.wilcoxon(decomp_total_times, trad_total_times)
            p_val_total = res_total.pvalue
            sig_total = "✅ Significant" if p_val_total < 0.05 else "❌ Not Signif."
        except Exception as e:
            p_val_total = float('nan')
            sig_total = "Error (equal vals / small samples)"
            
        try:
            res_nsga = scipy.stats.wilcoxon(local_nsga_times, trad_total_times)
            p_val_nsga = res_nsga.pvalue
            sig_nsga = "✅ Significant" if p_val_nsga < 0.05 else "❌ Not Signif."
        except Exception as e:
            p_val_nsga = float('nan')
            sig_nsga = "Error (equal vals / small samples)"
            
        # Speedups
        speedups_total = stats['speedup']
        speedups_nsga_only = stats['speedup_nsga_only']
        
        mean_sp_tot, median_sp_tot, std_sp_tot = np.mean(speedups_total), np.median(speedups_total), np.std(speedups_total)
        mean_sp_nsga, median_sp_nsga, std_sp_nsga = np.mean(speedups_nsga_only), np.median(speedups_nsga_only), np.std(speedups_nsga_only)
        
        md += "### Speedup and Wilcoxon Comparison (Decomp vs Traditional)\n\n"
        md += "Two statistical comparisons for execution time are presented:\n\n"
        
        md += "#### 1. Total Times Comparison (Parallel Total with Aggregation vs Traditional Sequential)\n"
        md += "Compares the full parallel time of decomposition (slowest NSGA-II + slowest aggregation step) against the total traditional sequential time:\n\n"
        md += "| Metric | Value |\n"
        md += "|---|---|\n"
        md += f"| **p-value (Wilcoxon)** | {p_val_total:.4e} |\n"
        md += f"| **Significance (α=0.05)** | {sig_total} |\n"
        md += f"| **Mean Speedup** | {mean_sp_tot:.2f}x |\n"
        md += f"| **Median Speedup** | {median_sp_tot:.2f}x |\n"
        md += f"| **Speedup Standard Deviation** | {std_sp_tot:.2f} |\n\n"
        
        md += "#### 2. Genetic Algorithm Comparison (Only Slowest NSGA-II from Decomp vs Traditional)\n"
        md += "Compares only the times dedicated to NSGA-II evolution, isolating the impact of the hierarchical aggregation operator:\n\n"
        md += "| Metric | Value |\n"
        md += "|---|---|\n"
        md += f"| **p-value (Wilcoxon)** | {p_val_nsga:.4e} |\n"
        md += f"| **Significance (α=0.05)** | {sig_nsga} |\n"
        md += f"| **Mean Speedup** | {mean_sp_nsga:.2f}x |\n"
        md += f"| **Median Speedup** | {median_sp_nsga:.2f}x |\n"
        md += f"| **Speedup Standard Deviation** | {std_sp_nsga:.2f} |\n\n"
        
        file_path = os.path.join(self.base_output_dir, "wilcoxon_statistical_report.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
            
        print(f"\nStatistical report generated at: {file_path}\n")

    def _generate_optimization_evolution_plot(self):
        """
        Genera una gráfica con la evolución de la latencia y coste a lo largo de las generaciones,
        comparando NSGA-II tradicional y Decomposition. Pinta el mínimo, la mediana y el máximo,
        promediados sobre todas las ejecuciones, junto con la desviación estándar (sombreado).
        """
        generations_per_step = self.config.generations
        k_steps = self.config.k_steps
        n_runs = self.config.n_runs
        root_node_id = 2 * self.config.n_nodes - 2
        
        total_gens = generations_per_step * k_steps
        
        # Estructuras para almacenar los valores por generación/paso para cada repetición
        trad_lat = {'min': [[] for _ in range(total_gens + 1)], 'med': [[] for _ in range(total_gens + 1)], 'max': [[] for _ in range(total_gens + 1)]}
        trad_cost = {'min': [[] for _ in range(total_gens + 1)], 'med': [[] for _ in range(total_gens + 1)], 'max': [[] for _ in range(total_gens + 1)]}
        
        dec_lat = {'min': [[] for _ in range(k_steps)], 'med': [[] for _ in range(k_steps)], 'max': [[] for _ in range(k_steps)]}
        dec_cost = {'min': [[] for _ in range(k_steps)], 'med': [[] for _ in range(k_steps)], 'max': [[] for _ in range(k_steps)]}
        
        for run_idx in range(n_runs):
            run_dir = os.path.join(self.base_output_dir, f"run_{run_idx}")
            
            # Cargar histórico de Traditional
            trad_file = os.path.join(run_dir, "history_submatrix_traditional.pkl")
            if os.path.exists(trad_file):
                import pickle
                with open(trad_file, "rb") as f:
                    trad_history = pickle.load(f)
                    
                for gen_idx, gen_data in enumerate(trad_history):
                    if gen_idx > total_gens:
                        break
                    front = gen_data['pareto_front']
                    if front:
                        lats = [ind['fitness'][0] for ind in front]
                        costs = [ind['fitness'][1] for ind in front]
                        trad_lat['min'][gen_idx].append(np.min(lats))
                        trad_lat['med'][gen_idx].append(np.median(lats))
                        trad_lat['max'][gen_idx].append(np.max(lats))
                        
                        trad_cost['min'][gen_idx].append(np.min(costs))
                        trad_cost['med'][gen_idx].append(np.median(costs))
                        trad_cost['max'][gen_idx].append(np.max(costs))
                        
            # Cargar histórico de Decomposition
            dec_file = os.path.join(run_dir, "aggregation_history.pkl")
            if os.path.exists(dec_file):
                import pickle
                with open(dec_file, "rb") as f:
                    dec_history = pickle.load(f)
                    
                for step_idx, step_data in enumerate(dec_history):
                    if step_idx >= k_steps:
                        break
                    front = step_data.get(root_node_id, [])
                    if front:
                        lats = [ind['fitness'][0] for ind in front]
                        costs = [ind['fitness'][1] for ind in front]
                        dec_lat['min'][step_idx].append(np.min(lats))
                        dec_lat['med'][step_idx].append(np.median(lats))
                        dec_lat['max'][step_idx].append(np.max(lats))
                        
                        dec_cost['min'][step_idx].append(np.min(costs))
                        dec_cost['med'][step_idx].append(np.median(costs))
                        dec_cost['max'][step_idx].append(np.max(costs))
                        
        x_trad = np.arange(total_gens + 1)
        x_dec = np.array([(s + 1) * generations_per_step for s in range(k_steps)])
        
        def get_stats(data_list):
            means, stds = [], []
            for arr in data_list:
                if arr:
                    means.append(np.mean(arr))
                    stds.append(np.std(arr))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            return np.array(means), np.array(stds)
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfica de Latencia
        for key, color, label in [('min', 'green', 'Min'), ('med', 'blue', 'Median'), ('max', 'red', 'Max')]:
            t_m, t_s = get_stats(trad_lat[key])
            ax1.plot(x_trad, t_m, color=color, linestyle='-', alpha=0.7, label=f"Trad {label}")
            ax1.fill_between(x_trad, t_m - t_s, t_m + t_s, color=color, alpha=0.1)
            
            d_m, d_s = get_stats(dec_lat[key])
            ax1.plot(x_dec, d_m, color=color, linestyle='--', marker='o', alpha=0.9, label=f"Decomp {label}")
            ax1.fill_between(x_dec, d_m - d_s, d_m + d_s, color=color, alpha=0.2)
            
        ax1.set_title("Latency Evolution vs Generations")
        ax1.set_xlabel("Generations")
        ax1.set_ylabel("Latency Value")
        ax1.set_yscale('log', nonpositive='clip')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')
        
        # Gráfica de Coste
        for key, color, label in [('min', 'green', 'Min'), ('med', 'blue', 'Median'), ('max', 'red', 'Max')]:
            t_m, t_s = get_stats(trad_cost[key])
            ax2.plot(x_trad, t_m, color=color, linestyle='-', alpha=0.7, label=f"Trad {label}")
            ax2.fill_between(x_trad, t_m - t_s, t_m + t_s, color=color, alpha=0.1)
            
            d_m, d_s = get_stats(dec_cost[key])
            ax2.plot(x_dec, d_m, color=color, linestyle='--', marker='o', alpha=0.9, label=f"Decomp {label}")
            ax2.fill_between(x_dec, d_m - d_s, d_m + d_s, color=color, alpha=0.2)
            
        ax2.set_title("Cost Evolution vs Generations")
        ax2.set_xlabel("Generations")
        ax2.set_ylabel("Cost Value")
        ax2.set_yscale('log', nonpositive='clip')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, "optimization_evolution_comparison.pdf"))
        plt.close()
        
        # Generar gráfica idéntica solo con la serie tradicional (NSGA-II global)
        fig_trad, (ax1_t, ax2_t) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfica de Latencia Tradicional
        for key, color, label in [('min', 'green', 'Min'), ('med', 'blue', 'Median'), ('max', 'red', 'Max')]:
            t_m, t_s = get_stats(trad_lat[key])
            ax1_t.plot(x_trad, t_m, color=color, linestyle='-', alpha=0.7, label=f"Trad {label}")
            ax1_t.fill_between(x_trad, t_m - t_s, t_m + t_s, color=color, alpha=0.1)
            
        ax1_t.set_title("Latency Evolution vs Generations (Global Traditional)")
        ax1_t.set_xlabel("Generations")
        ax1_t.set_ylabel("Latency Value")
        ax1_t.set_yscale('log', nonpositive='clip')
        ax1_t.grid(True, linestyle='--', alpha=0.7)
        ax1_t.legend(loc='upper right')
        
        # Gráfica de Coste Tradicional
        for key, color, label in [('min', 'green', 'Min'), ('med', 'blue', 'Median'), ('max', 'red', 'Max')]:
            t_m, t_s = get_stats(trad_cost[key])
            ax2_t.plot(x_trad, t_m, color=color, linestyle='-', alpha=0.7, label=f"Trad {label}")
            ax2_t.fill_between(x_trad, t_m - t_s, t_m + t_s, color=color, alpha=0.1)
            
        ax2_t.set_title("Cost Evolution vs Generations (Global Traditional)")
        ax2_t.set_xlabel("Generations")
        ax2_t.set_ylabel("Cost Value")
        ax2_t.set_yscale('log', nonpositive='clip')
        ax2_t.grid(True, linestyle='--', alpha=0.7)
        ax2_t.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, "optimization_evolution_traditional.pdf"))
        plt.close()

