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

def print_submatrices_summary(submatrices, config):
    from collections import Counter
    sizes = []
    for node_id, template in submatrices.items():
        n_rows = len(template['rows'])
        n_services = len(template['services'])
        sizes.append((n_rows, n_services))
    
    counter = Counter(sizes)
    # Ordenar por area (filas * columnas) descendente, luego por filas y columnas descendente
    sorted_sizes = sorted(counter.items(), key=lambda x: (x[0][0] * x[0][1], x[0][0], x[0][1]), reverse=True)
    
    print("\n" + "="*70)
    print(f"{'RESUMEN DE TAMAÑOS DE SUBMATRICES CALCULADAS (DESCOMPOSICIÓN)':^70}")
    print("="*70)
    print(f" {'Dimensiones (Filas x Servicios)':<35} | {'Total Celdas':<12} | {'Cantidad':<8}")
    print("-"*70)
    for (n_rows, n_services), count in sorted_sizes:
        dims = f"{n_rows} x {n_services}"
        area = n_rows * n_services
        print(f" {dims:<35} | {area:<12} | {count:<8}")
        
    print("="*70)
    print(f"{'MATRIZ DE OPTIMIZACIÓN GLOBAL (TRADICIONAL)':^70}")
    print("="*70)
    global_rows = config.n_nodes
    global_services = config.n_apps
    global_area = global_rows * global_services
    print(f" {'Dimensiones (Filas x Servicios)':<35} | {'Total Celdas':<12} | {'Cantidad':<8}")
    print("-"*70)
    dims_global = f"{global_rows} x {global_services}"
    print(f" {dims_global:<35} | {global_area:<12} | {'1':<8}")
    print("="*70 + "\n")

def get_latest_results_dir():
    dirs = glob.glob("results/*_*")
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

    print(f"\nGuardando todos los resultados de esta ejecución en: {rm.output_dir}\n")
    print(f"Configuración: Infra_Seed={config.seed_infra}, Apps_Seed={config.seed_apps}, Users_Seed={config.seed_users}")
    for node_id in range(config.n_nodes):
        if node_id in submatrices:
            print(f"Submatriz Nodo {node_id}: {submatrices[node_id]['rows']} x {submatrices[node_id]['services']}")
            print('\n')
            
    print_submatrices_summary(submatrices, config)
            
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
    print("\nSimulación finalizada. Datos guardados correctamente.")
    return infra, workload, Z, submatrices

def execute_runs(runs_to_execute, config, rm, infra, workload, Z, submatrices):
    for run_idx in runs_to_execute:
        print("\n" + "="*80)
        print(f"INICIANDO EJECUCIÓN {run_idx + 1}/{config.n_runs}")
        print("="*80)
        
        rm.set_current_run(run_idx)
        
        # Generar semilla específica para el run
        run_nsga_seed = config.seed_nsga2 + run_idx
        rng_nsga2 = RandomManager(run_nsga_seed)
        
        print(f"\nInicializando instancias de NSGA-II para cada submatriz (Tamaño de población: {config.pop_size})...")
        nsga2_instances = {}
        for node_id, template in submatrices.items():
            nsga_seed = rng_nsga2.py_rng.randint(0, 2**31 - 1)
            nsga_rng = RandomManager(nsga_seed)
            nsga = NSGA2(rng=nsga_rng, template=template, infra=infra, workload=workload)
            nsga.initialize_population(pop_size=config.pop_size)
            nsga2_instances[node_id] = nsga
            
        print(f"Inicializando la instancia tradicional de NSGA-II (Tamaño de población: {config.pop_size})...")
        traditional_nsga_seed = rng_nsga2.py_rng.randint(0, 2**31 - 1)
        traditional_nsga_rng = RandomManager(traditional_nsga_seed)
        traditional_template = {
            "rows": list(infra.graph.nodes()),
            "services": list(range(config.n_apps))
        }
        traditional_nsga = NSGA2(rng=traditional_nsga_rng, template=traditional_template, infra=infra, workload=workload)
        traditional_nsga.initialize_population(pop_size=config.pop_size)
        
        K_STEPS = config.k_steps
        print(f"Ejecutando {K_STEPS} pasos de {config.generations} generaciones para las {len(nsga2_instances)} submatrices y modo tradicional...")
        
        aggregator = HierarchicalAggregator(infra, workload, config, nsga2_instances, Z)
        aggregation_history = []
        
        local_nsga_times = {node_id: 0.0 for node_id in submatrices.keys()}
        aggregation_merge_times = {}
        traditional_nsga_time = 0.0
        
        for step in range(K_STEPS):
            print(f"  Ejecutando paso {step + 1}/{K_STEPS}...")
            # Modo Decomposition
            for node_id, nsga in nsga2_instances.items():
                t_start = time.perf_counter()
                nsga.run_generations(n_generations=config.generations)
                local_nsga_times[node_id] += time.perf_counter() - t_start
                
            print(f"  -> Ejecutando Operador de Agregación Jerárquica (HC-PA)...")
            step_fronts, step_merge_times = aggregator.aggregate_step()
            aggregation_history.append(step_fronts)
            for parent_id, t_merge in step_merge_times.items():
                aggregation_merge_times[parent_id] = aggregation_merge_times.get(parent_id, 0.0) + t_merge
            
            # Modo Tradicional
            print(f"  -> Ejecutando modo tradicional...")
            t_start = time.perf_counter()
            traditional_nsga.run_generations(n_generations=config.generations)
            traditional_nsga_time += time.perf_counter() - t_start
                
        print(f"NSGA-II y HC-PA ejecutados exitosamente. Histórico: {K_STEPS * config.generations + 1} poblaciones por instancia.")
        
        # Guardar resultados evolutivos
        for node_id, nsga in nsga2_instances.items():
            rm.save_history(node_id, nsga)
            
        print("\nGuardando resultados del modo tradicional...")
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
    print(f"{'INICIANDO GENERACIÓN DE GRÁFICAS Y REPORTES':^80}")
    print("="*80)
    
    # 1. Visualización General
    rm.output_dir = rm.base_output_dir
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=list(infra.graph.nodes()))
    plt.title(f"Clustering Jerárquico Híbrido (Latencia + Afinidad de App)")
    plt.xlabel("ID del Nodo")
    plt.ylabel("Distancia Híbrida")
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
        print(f"\nGenerando gráficas para la ejecución {run_idx + 1}/{config.n_runs}")
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
    print(f"{'CALCULANDO ANÁLISIS ESTADÍSTICO GLOBAL':^80}")
    print("="*80)
    
    rm.generate_global_statistical_analysis()
        
    print("Todos los resultados y PDFs han sido generados correctamente.")

if __name__ == "__main__":
    config = SimulationConfig()
    
    if config.run_simulation:
        rm = ResultsManager(config)
        infra, workload, Z, submatrices = run_simulation_phase(config, rm)
        
    elif config.run_recovery:
        load_dir = config.load_results_dir
        if not load_dir or load_dir == "latest":
            load_dir = get_latest_results_dir()
            if not load_dir:
                print("Error: No se encontró ningún directorio de resultados en 'results/'")
                exit(1)
            print(f"Recuperando desde el último directorio de resultados: {load_dir}")
        else:
            if not os.path.exists(load_dir):
                print(f"Error: El directorio {load_dir} no existe.")
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
            print(f"Error: No se encontró simulation_context.pkl en {load_dir}. No se puede recuperar.")
            exit(1)
            
        with open(ctx_file, "rb") as f:
            ctx = pickle.load(f)
        infra, workload, Z, submatrices = ctx['infra'], ctx['workload'], ctx['Z'], ctx['submatrices']
        
        print_submatrices_summary(submatrices, config)
        
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
            print(f"\nDetectadas {len(pending_runs)} repeticiones pendientes: {pending_runs}")
            execute_runs(pending_runs, config, rm, infra, workload, Z, submatrices)
            rm.save_global_metrics()
            print("\nRecuperación finalizada. Datos guardados correctamente.")
        else:
            print(f"\nNo hay repeticiones pendientes. (Total {config.n_runs} runs)")
            rm.save_global_metrics()
            
    else:
        load_dir = config.load_results_dir
        if not load_dir or load_dir == "latest":
            load_dir = get_latest_results_dir()
            if not load_dir:
                print("Error: No se encontró ningún directorio de resultados en 'results/'")
                exit(1)
            print(f"Usando el último directorio de resultados: {load_dir}")
        else:
            if not os.path.exists(load_dir):
                print(f"Error: El directorio {load_dir} no existe.")
                exit(1)
                
        rm = ResultsManager(config, load_dir=load_dir)
        rm.load_global_metrics()
        
        ctx_file = os.path.join(rm.base_output_dir, "simulation_context.pkl")
        if not os.path.exists(ctx_file):
            print(f"Error: No se encontró simulation_context.pkl en {rm.base_output_dir}. Necesario para graficar sin simular.")
            exit(1)
            
        with open(ctx_file, "rb") as f:
            ctx = pickle.load(f)
        infra = ctx['infra']
        workload = ctx['workload']
        Z = ctx['Z']
        submatrices = ctx['submatrices']
        
        print_submatrices_summary(submatrices, config)
        
    if config.generate_plots:
        run_plotting_phase(config, rm, infra, workload, Z, submatrices)
