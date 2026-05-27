import json
import os

class SimulationConfig:
    """Configuración cargada desde un archivo JSON con valores por defecto."""
    def __init__(self, config_path="simulation_config.json"):
        self._config_path = config_path
        
        # 1. Definir valores por defecto (fallback)
        self.experiment_id = "development"
        self.generation_mode = "uniform_generation"
        
        # Modos de ejecución
        self.run_simulation = True
        self.run_recovery = False
        self.generate_plots = True
        self.load_results_dir = "latest"
        
        # Semillas
        self.seed_infra = 101
        self.seed_apps = 150
        self.seed_users = 202
        self.seed_workload = 202
        self.seed_clustering = 303
        self.seed_nsga2 = 404
        
        # Infraestructura
        self.n_nodes = 30
        self.m_edges = 2
        self.min_node_cap = 20
        self.max_node_cap = 100
        self.min_node_price = 1
        self.max_node_price = 5
        self.min_latency = 1.0
        self.max_latency = 10.0
        
        # Tiered Zones Defaults
        self.tiered_zones = {
            "cloud": {"capacity": 10000, "price": 0.5},
            "fog": {
                "min_capacity": 50, "max_capacity": 200,
                "min_price": 2, "max_price": 4,
                "min_latency_to_cloud": 20.0, "max_latency_to_cloud": 40.0,
                "fog_percentage": 0.6
            },
            "edge": {
                "min_capacity": 10, "max_capacity": 30,
                "min_price": 5, "max_price": 10,
                "min_latency_to_fog": 1.0, "max_latency_to_fog": 5.0,
                "mesh_prob": 0.3,
                "min_mesh_latency": 2.0, "max_mesh_latency": 8.0
            }
        }
        
        # Workload
        self.n_apps = 8
        self.n_users = 25
        self.edge_nodes_with_users_percentage = 0.3
        self.min_app_demand = 2
        self.max_app_demand = 10
        self.min_app_execution_load = 1.0
        self.max_app_execution_load = 10.0
        # User
        self.min_request_ratio = 0.1
        self.max_request_ratio = 10.0
        
        # Algoritmo
        self.alpha = 0.4
        self.beta = 0.6
        self.k_levels = 2
        self.k_steps = 5
        self.n_runs = 10
        
        # Genético
        self.pop_size = 50
        self.generations = 100
        self.aggregator_front_size = self.pop_size

        # 2. Cargar desde el archivo JSON si existe
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Asignar experiment_id si existe
                if "experiment_id" in data:
                    self.experiment_id = data["experiment_id"]
                if "generation_mode" in data:
                    self.generation_mode = data["generation_mode"]
                if "run_simulation" in data:
                    self.run_simulation = data["run_simulation"]
                if "run_recovery" in data:
                    self.run_recovery = data["run_recovery"]
                if "generate_plots" in data:
                    self.generate_plots = data["generate_plots"]
                if "load_results_dir" in data:
                    self.load_results_dir = data["load_results_dir"]
                elif "target_results_dir" in data:
                    self.load_results_dir = data["target_results_dir"]
                
                # Mapeo de grupos jerárquicos
                if "seeds" in data:
                    seeds = data["seeds"]
                    self.seed_infra = seeds.get("infra", self.seed_infra)
                    self.seed_workload = seeds.get("workload", self.seed_workload)
                    self.seed_apps = seeds.get("apps", seeds.get("workload", self.seed_apps))
                    self.seed_users = seeds.get("users", seeds.get("workload", self.seed_users))
                    self.seed_clustering = seeds.get("clustering", self.seed_clustering)
                    self.seed_nsga2 = seeds.get("nsga2", self.seed_nsga2)
                
                if "infrastructure" in data:
                    infra = data["infrastructure"]
                    self.n_nodes = infra.get("n_nodes", self.n_nodes)
                    self.m_edges = infra.get("m_edges", self.m_edges)
                    self.min_node_cap = infra.get("min_node_cap", self.min_node_cap)
                    self.max_node_cap = infra.get("max_node_cap", self.max_node_cap)
                    self.min_node_price = infra.get("min_node_price", self.min_node_price)
                    self.max_node_price = infra.get("max_node_price", self.max_node_price)
                    self.min_latency = infra.get("min_latency", self.min_latency)
                    self.max_latency = infra.get("max_latency", self.max_latency)
                
                if "tiered_zones" in data:
                    tz = data["tiered_zones"]
                    if "cloud" in tz:
                        self.tiered_zones["cloud"].update(tz["cloud"])
                    if "fog" in tz:
                        self.tiered_zones["fog"].update(tz["fog"])
                    if "edge" in tz:
                        self.tiered_zones["edge"].update(tz["edge"])
                
                if "workload" in data:
                    wl = data["workload"]
                    self.n_apps = wl.get("n_apps", self.n_apps)
                    self.n_users = wl.get("n_users", self.n_users)
                    self.edge_nodes_with_users_percentage = wl.get(
                        "edge_nodes_with_users_percentage",
                        wl.get("edge_nodes_percentage", self.edge_nodes_with_users_percentage)
                    )
                    self.min_app_demand = wl.get("min_app_demand", self.min_app_demand)
                    self.max_app_demand = wl.get("max_app_demand", self.max_app_demand)
                    self.min_app_execution_load = wl.get("min_app_execution_load", self.min_app_execution_load)
                    self.max_app_execution_load = wl.get("max_app_execution_load", self.max_app_execution_load)
                
                if "user" in data:
                    usr = data["user"]
                    self.min_request_ratio = usr.get("min_request_ratio", self.min_request_ratio)
                    self.max_request_ratio = usr.get("max_request_ratio", self.max_request_ratio)
                
                if "algorithm" in data:
                    alg = data["algorithm"]
                    self.alpha = alg.get("alpha", self.alpha)
                    self.beta = alg.get("beta", self.beta)
                    self.k_levels = alg.get("k_levels", self.k_levels)
                    self.k_steps = alg.get("k_steps", self.k_steps)
                    self.n_runs = alg.get("n_runs", self.n_runs)
                
                if "genetic" in data:
                    gen = data["genetic"]
                    self.pop_size = gen.get("pop_size", self.pop_size)
                    self.generations = gen.get("generations", self.generations)
                    self.aggregator_front_size = gen.get("aggregator_front_size", self.pop_size)
            except Exception as e:
                print(f"Error al cargar el archivo de configuración {config_path}: {e}. Usando valores por defecto.")