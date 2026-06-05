"""
===============================================================================
File: main.py
Description: Orchestrates the entire simulation process, including setup, genetic algorithm execution, recovery mechanisms, and statistical reporting.
Role: Main Entry Point
Execution: python src/main.py [optional_config_path]
Inputs: JSON Configuration File (default: ../config/simulation_config.json)
Outputs: Simulation logs, pickled metrics, Submatrices summaries, Wilcoxon statistical reports.

Authors: Carlos Guerrero and Isaac Lera
Date: June 2026
License: MIT License
===============================================================================
"""
import networkx as nx
import numpy as np
import random
import time
import os
import pickle
import glob
from scipy.spatial.distance import pdist, squareform, jaccard
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import to_tree
from infrastructure import Infrastructure
from workload import Workload
from hierarchical_scheduler import HierarchicalScheduler
from simulation_config import SimulationConfig
from random_manager import RandomManager
from visualizer import Visualizer
from nsga2 import NSGA2
from results_manager import ResultsManager
from aggregation_operator import HierarchicalAggregator

def print_submatrices_summary(submatrices, config, rm=None):
    from collections import Counter
    import os
    natural_sizes = []
    pruned_sizes = []
    
    for node_id, template in submatrices.items():
        n_rows = len(template['rows'])
        n_services = len(template['services'])
        if template.get('is_pruned', False):
            theo = template.get('theoretical_rows', n_rows)
            pruned_sizes.append((theo, n_rows, n_services))
        else:
            natural_sizes.append((n_rows, n_services))
            
    output_lines = []
    output_lines.append("\n" + "="*70)
    output_lines.append(f"{'CALCULATED SUBMATRICES SIZE SUMMARY (DECOMPOSITION)':^70}")
    output_lines.append("="*70)
    
    output_lines.append(f"\n--- Natural Submatrices ({len(natural_sizes)}) ---")
    if natural_sizes:
        output_lines.append(f" {'Dimensions (Rows x Services)':<35} | {'Total Cells':<12} | {'Count':<8}")
        output_lines.append("-" * 70)
        c_nat = Counter(natural_sizes)
        sorted_nat = sorted(c_nat.items(), key=lambda x: (x[0][0]*x[0][1], x[0][0], x[0][1]), reverse=True)
        for (n_rows, n_services), count in sorted_nat:
            dims = f"{n_rows} x {n_services}"
            area = n_rows * n_services
            output_lines.append(f" {dims:<35} | {area:<12} | {count:<8}")
    else:
        output_lines.append(" None")
        
    output_lines.append(f"\n--- Pruned Submatrices ({len(pruned_sizes)}) ---")
    if pruned_sizes:
        output_lines.append(f" {'Theoretical -> Final (Rows) x Serv.':<35} | {'Cells (Final)':<12} | {'Count':<8}")
        output_lines.append("-" * 70)
        c_pruned = Counter(pruned_sizes)
        sorted_pruned = sorted(c_pruned.items(), key=lambda x: (x[0][1]*x[0][2], x[0][0]), reverse=True)
        for (t_rows, f_rows, n_servs), count in sorted_pruned:
            dims = f"{t_rows}->{f_rows} x {n_servs}"
            area = f_rows * n_servs
            output_lines.append(f" {dims:<35} | {area:<12} | {count:<8}")
    else:
        output_lines.append(" None")
        
    output_lines.append("="*70)
    output_lines.append(f"{'GLOBAL OPTIMIZATION MATRIX (TRADITIONAL)':^70}")
    output_lines.append("="*70)
    global_rows = config.n_nodes
    global_services = config.n_apps
    global_area = global_rows * global_services
    output_lines.append(f" {'Dimensions (Rows x Services)':<35} | {'Total Cells':<12} | {'Count':<8}")
    output_lines.append("-" * 70)
    dims_global = f"{global_rows} x {global_services}"
    output_lines.append(f" {dims_global:<35} | {global_area:<12} | {'1':<8}")
    
    full_output = "\n".join(output_lines)
    print(full_output)
    
    if rm is not None and hasattr(rm, 'base_output_dir'):
        file_path = os.path.join(rm.base_output_dir, "submatrices_summary.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Submatrices Size Summary\n\n```text\n")
            f.write(full_output)
            f.write("\n```\n")
    print("="*70 + "\n")

def get_latest_results_dir():
    dirs = glob.glob("../results/*_*")
    if not dirs:
        return None
    # Sort by timestamp (the end of the dir name)
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]

def run_simulation_phase(config, rm):
    # Instanciamos generadores independientes para cada componente
    rng_infra = RandomManager(config.seed_infra)
    rng_apps = RandomManager(config.seed_apps)
    rng_users = RandomManager(config.seed_users)
    rng_clustering = RandomManager(config.seed_clustering)
    rng_nsga2 = RandomManager(config.seed_nsga2)

    # Inyectamos el gestor en los componentes
    infra = Infrastructure(config, rng_infra)
    infra.setup_cloud()
    infra.assign_latencies()
    infra.assign_capacities()
    infra.assign_prices()
    infra.precompute_shortest_paths()

    workload = Workload(infra, config, rng_apps, rng_users)
    workload.generate_demand()
    workload.calculate_vi_vectors()

    scheduler = HierarchicalScheduler(infra, workload, config)
    Z = scheduler.run_clustering()
    submatrices = scheduler.get_submatrices()

    print(f"\nSaving all results of this execution to: {rm.output_dir}\n")
    print(f"Configuration: Infra_Seed={config.seed_infra}, Apps_Seed={config.seed_apps}, Users_Seed={config.seed_users}")
    for node_id in range(config.n_nodes):
        if node_id in submatrices:
            print(f"Submatrix Node {node_id}: {submatrices[node_id]['rows']} x {submatrices[node_id]['services']}")
            print('\n')
            
    print_submatrices_summary(submatrices, config, rm)
            
    # Guardar contexto de la simulación antes de iniciar las ejecuciones
    ctx_file = os.path.join(rm.base_output_dir, "simulation_context.pkl")
    with open(ctx_file, "wb") as f:
        pickle.dump({
            'infra': infra, 
            'workload': workload, 
            'Z': Z, 
            'submatrices': submatrices
        }, f)

    runs_to_execute = list(range(config.n_runs))
    execute_runs(runs_to_execute, config, rm, infra, workload, Z, submatrices)
    
    # Finalizadas todas las runs, guardamos las métricas globales
    rm.save_global_metrics()
    print("\nSimulation finished. Data saved successfully.")
    return infra, workload, Z, submatrices

def execute_runs(runs_to_execute, config, rm, infra, workload, Z, submatrices):
    for run_idx in runs_to_execute:
        print("\n" + "="*80)
        print(f"STARTING RUN {run_idx + 1}/{config.n_runs}")
        print("="*80)
        
        rm.set_current_run(run_idx)
        
        # Generar semilla específica para el run
        run_nsga_seed = config.seed_nsga2 + run_idx
        rng_nsga2 = RandomManager(run_nsga_seed)
        
        print(f"\nInitializing NSGA-II instances for each submatrix (Population size: {config.pop_size})...")
        nsga2_instances = {}
        for node_id, template in submatrices.items():
            nsga_seed = rng_nsga2.py_rng.randint(0, 2**31 - 1)
            nsga_rng = RandomManager(nsga_seed)
            nsga = NSGA2(rng=nsga_rng, template=template, infra=infra, workload=workload)
            nsga.initialize_population(pop_size=config.pop_size)
            nsga2_instances[node_id] = nsga
            
        print(f"Initializing traditional NSGA-II instance (Population size: {config.pop_size})...")
        traditional_nsga_seed = rng_nsga2.py_rng.randint(0, 2**31 - 1)
        traditional_nsga_rng = RandomManager(traditional_nsga_seed)
        traditional_template = {
            "rows": list(infra.graph.nodes()),
            "services": list(range(config.n_apps))
        }
        traditional_nsga = NSGA2(rng=traditional_nsga_rng, template=traditional_template, infra=infra, workload=workload)
        traditional_nsga.initialize_population(pop_size=config.pop_size)
        
        K_STEPS = config.k_steps
        print(f"Running {K_STEPS} steps of {config.generations} generations for the {len(nsga2_instances)} submatrices and traditional mode...")
        
        aggregator = HierarchicalAggregator(infra, workload, config, nsga2_instances, Z)
        aggregation_history = []
        
        local_nsga_times = {node_id: 0.0 for node_id in submatrices.keys()}
        aggregation_merge_times = {}
        traditional_nsga_time = 0.0
        
        for step in range(K_STEPS):
            print(f"  Running step {step + 1}/{K_STEPS}...")
            # Modo Decomposition
            for node_id, nsga in nsga2_instances.items():
                t_start = time.perf_counter()
                nsga.run_generations(n_generations=config.generations)
                local_nsga_times[node_id] += time.perf_counter() - t_start
                
            print(f"  -> Running Hierarchical Aggregation Operator (HC-PA)...")
            step_fronts, step_merge_times = aggregator.aggregate_step()
            aggregation_history.append(step_fronts)
            for parent_id, t_merge in step_merge_times.items():
                aggregation_merge_times[parent_id] = aggregation_merge_times.get(parent_id, 0.0) + t_merge
            
            # Modo Tradicional
            print(f"  -> Running traditional mode...")
            t_start = time.perf_counter()
            traditional_nsga.run_generations(n_generations=config.generations)
            traditional_nsga_time += time.perf_counter() - t_start
                
        print(f"NSGA-II and HC-PA executed successfully. History: {K_STEPS * config.generations + 1} populations per instance.")
        
        # Guardar resultados evolutivos
        for node_id, nsga in nsga2_instances.items():
            rm.save_history(node_id, nsga)
            
        print("\nSaving traditional mode results...")
        rm.save_history("traditional", traditional_nsga)
        rm.save_aggregation_history(aggregation_history)
        
        root_node_id = 2 * config.n_nodes - 2
        
        rm.collect_run_data(
            aggregation_history[-1].get(root_node_id, []),
            traditional_nsga.history[-1]['pareto_front'],
            local_nsga_times,
            aggregation_merge_times,
            traditional_nsga_time
        )
        
        # Save run_metrics to avoid data loss if it crashes
        run_metrics_file = os.path.join(rm.output_dir, "run_metrics.pkl")
        with open(run_metrics_file, "wb") as f:
            pickle.dump({
                'local_nsga_times': local_nsga_times,
                'aggregation_merge_times': aggregation_merge_times,
                'traditional_nsga_time': traditional_nsga_time
            }, f)

def run_plotting_phase(config, rm, infra, workload, Z, submatrices):
    print("\n" + "="*80)
    print(f"{'STARTING PLOT AND REPORT GENERATION':^80}")
    print("="*80)
    
    # 1. Visualización General
    rm.output_dir = rm.base_output_dir
    n_nodes = config.n_nodes
    fig_width = max(12, min(50, n_nodes * 0.20))
    plt.figure(figsize=(fig_width, 10))
    
    # Substantially larger leaf labels for complete legibility
    dynamic_font_size = max(8.0, min(14.0, 1000.0 / n_nodes))
    
    dendrogram(
        Z, 
        labels=list(infra.graph.nodes()),
        leaf_rotation=90.0,
        leaf_font_size=dynamic_font_size
    )
    plt.title("Hybrid Hierarchical Clustering (Latency + App Affinity)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Node ID", fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel("Hybrid Distance", fontsize=14, fontweight='bold', labelpad=10)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(rm.base_output_dir, "clustering_dendrogram.pdf"), format='pdf')
    plt.close()



    viz = Visualizer(infra, workload, config)
    
    # Listado por consola de las infraestructuras y reporte Markdown
    viz.list_node_capacities()
    viz.list_app_requirements()
    viz.list_users_human_readable()
    viz.list_node_vectors()
    
    viz.draw_infrastructure(output_path=os.path.join(rm.base_output_dir, "infrastructure_graph.pdf"))
    viz.draw_infrastructure_detailed(output_path=os.path.join(rm.base_output_dir, "infrastructure_graph_detailed.pdf"))
    
    rm.generate_markdown_summary(infra, workload)
    
    # 2. Generación de Gráficos Evolutivos por Run
    root_node_id = 2 * config.n_nodes - 2
    for run_idx in range(config.n_runs):
        print(f"\nGenerating plots for run {run_idx + 1}/{config.n_runs}")
        rm.set_current_run(run_idx)
        
        # Para submatrices y tradicional, necesitamos el "history"
        for node_id, template in submatrices.items():
            hist_file = os.path.join(rm.output_dir, f"history_submatrix_{node_id}.pkl")
            if os.path.exists(hist_file):
                with open(hist_file, "rb") as f:
                    history = pickle.load(f)
                n_users = sum(len(workload.user_assignments.get(n, [])) for n in template['rows'])
                subtitle = f"Nodes: {len(template['rows'])} | Users: {n_users} | Apps: {len(template['services'])}"
                rm.plot_pareto_evolution(node_id, history, subtitle=subtitle)
                
        trad_file = os.path.join(rm.output_dir, f"history_submatrix_traditional.pkl")
        if os.path.exists(trad_file):
            with open(trad_file, "rb") as f:
                trad_history = pickle.load(f)
            n_users = sum(len(workload.user_assignments.get(n, [])) for n in infra.graph.nodes())
            subtitle = f"Nodes: {config.n_nodes} | Users: {n_users} | Apps: {config.n_apps}"
            rm.plot_pareto_evolution("traditional", trad_history, subtitle=subtitle)
        else:
            trad_history = None
            
        agg_file = os.path.join(rm.output_dir, "aggregation_history.pkl")
        if os.path.exists(agg_file):
            with open(agg_file, "rb") as f:
                aggregation_history = pickle.load(f)
            rm.plot_global_pareto_evolution(aggregation_history, root_node_id, trad_history)
            rm.generate_global_solutions_reports(aggregation_history, root_node_id, viz)
            
    # 3. Análisis Estadístico Global
    print("\n" + "="*80)
    print(f"{'CALCULATING GLOBAL STATISTICAL ANALYSIS':^80}")
    print("="*80)
    
    rm.generate_global_statistical_analysis()
        
    print("All results and PDFs have been generated successfully.")

if __name__ == "__main__":
    import sys
    config_file = "../config/simulation_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        
    config = SimulationConfig(config_file)

    
    if config.run_simulation:
        rm = ResultsManager(config)
        infra, workload, Z, submatrices = run_simulation_phase(config, rm)
        
    elif config.run_recovery:
        load_dir = config.load_results_dir
        if not load_dir or load_dir == "latest":
            load_dir = get_latest_results_dir()
            if not load_dir:
                print("Error: No results directory found in '../results/'")
                exit(1)
            print(f"Recovering from the latest results directory: {load_dir}")
        else:
            if not os.path.exists(load_dir):
                print(f"Error: The directory {load_dir} does not exist.")
                exit(1)
                
        rm = ResultsManager(config, load_dir=load_dir)
        
        # Leemos el config original copiado
        copied_json = os.path.join(load_dir, "simulation_config.json")
        if os.path.exists(copied_json):
            import json
            with open(copied_json, "r") as f:
                old_data = json.load(f)
            original_runs = old_data.get("algorithm", {}).get("n_runs", config.n_runs)
            config.n_runs = original_runs
            
        ctx_file = os.path.join(load_dir, "simulation_context.pkl")
        if not os.path.exists(ctx_file):
            print(f"Error: simulation_context.pkl not found in {load_dir}. Cannot recover.")
            exit(1)
            
        with open(ctx_file, "rb") as f:
            ctx = pickle.load(f)
        infra, workload, Z, submatrices = ctx['infra'], ctx['workload'], ctx['Z'], ctx['submatrices']
        
        print_submatrices_summary(submatrices, config, rm)
        
        pending_runs = []
        root_node_id = 2 * config.n_nodes - 2
        for run_idx in range(config.n_runs):
            run_dir = os.path.join(load_dir, f"run_{run_idx}")
            agg_file = os.path.join(run_dir, "aggregation_history.pkl")
            trad_file = os.path.join(run_dir, "history_submatrix_traditional.pkl")
            metrics_file = os.path.join(run_dir, "run_metrics.pkl")
            
            if os.path.isdir(run_dir) and os.path.exists(agg_file) and os.path.exists(trad_file) and os.path.exists(metrics_file):
                # Run ya había finalizado: recuperar métricas y meterlas al rm
                rm.set_current_run(run_idx)
                with open(agg_file, "rb") as f:
                    agg_hist = pickle.load(f)
                with open(trad_file, "rb") as f:
                    trad_hist = pickle.load(f)
                with open(metrics_file, "rb") as f:
                    run_mets = pickle.load(f)
                    
                rm.collect_run_data(
                    agg_hist[-1].get(root_node_id, []),
                    trad_hist[-1]['pareto_front'],
                    run_mets['local_nsga_times'],
                    run_mets['aggregation_merge_times'],
                    run_mets['traditional_nsga_time']
                )
            else:
                pending_runs.append(run_idx)
                
        if pending_runs:
            print(f"\nDetected {len(pending_runs)} pending runs: {pending_runs}")
            execute_runs(pending_runs, config, rm, infra, workload, Z, submatrices)
            rm.save_global_metrics()
            print("\nRecovery finished. Data saved successfully.")
        else:
            print(f"\nNo pending runs. (Total {config.n_runs} runs)")
            rm.save_global_metrics()
            
    else:
        load_dir = config.load_results_dir
        if not load_dir or load_dir == "latest":
            load_dir = get_latest_results_dir()
            if not load_dir:
                print("Error: No results directory found in '../results/'")
                exit(1)
            print(f"Using the latest results directory: {load_dir}")
        else:
            if not os.path.exists(load_dir):
                print(f"Error: The directory {load_dir} does not exist.")
                exit(1)
                
        rm = ResultsManager(config, load_dir=load_dir)
        rm.load_global_metrics()
        
        ctx_file = os.path.join(rm.base_output_dir, "simulation_context.pkl")
        if not os.path.exists(ctx_file):
            print(f"Error: simulation_context.pkl not found in {rm.base_output_dir}. Needed for plotting without simulating.")
            exit(1)
            
        with open(ctx_file, "rb") as f:
            ctx = pickle.load(f)
        infra = ctx['infra']
        workload = ctx['workload']
        Z = ctx['Z']
        submatrices = ctx['submatrices']
        
        print_submatrices_summary(submatrices, config, rm)
        
