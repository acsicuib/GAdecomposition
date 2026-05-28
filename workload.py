import networkx as nx
import random
import numpy as np
from infrastructure import Infrastructure
from simulation_config import SimulationConfig
from random_manager import RandomManager


class Workload:
    def __init__(self, infrastructure: Infrastructure, config: SimulationConfig, rng_apps: RandomManager, rng_users: RandomManager):
        self.infra = infrastructure
        self.config = config
        self.rng_apps = rng_apps
        self.rng_users = rng_users
        
        # Demandas de recursos para límites de capacidad (ej: RAM, Disco)
        self.app_demands = {
            i: self.rng_apps.py_rng.randint(config.min_app_demand, config.max_app_demand) 
            for i in range(config.n_apps)
        }
        
        # Cargas de ejecución para calcular el coste facturado (ej: CPU time)
        self.app_execution_loads = {
            i: self.rng_apps.py_rng.uniform(config.min_app_execution_load, config.max_app_execution_load)
            for i in range(config.n_apps)
        }

        # Popularidad de aplicaciones
        if getattr(self.config, 'generation_mode', 'uniform_generation') in ['geographical_aware', 'tiered_zones']:
            self.app_popularities = [self.rng_apps.py_rng.uniform(0.1, 1.0) for _ in range(config.n_apps)]
        else:
            self.app_popularities = [1.0 for _ in range(config.n_apps)]
        
        self.user_assignments = {}
        self.node_vectors = {node: np.zeros(config.n_apps) for node in infrastructure.graph.nodes()}
 
    def generate_demand(self):
        mode = getattr(self.config, 'generation_mode', 'uniform_generation')
        if mode == 'tiered_zones':
            edge_nodes = [n for n, d in self.infra.graph.nodes(data=True) if d.get('layer') == 'edge']
            if not edge_nodes:
                allowed_nodes = list(self.infra.graph.nodes())
            else:
                edge_nodes_sorted = sorted(edge_nodes, key=lambda k: self.infra.centrality.get(k, 0.0))
                pct = getattr(self.config, 'edge_nodes_with_users_percentage', getattr(self.config, 'edge_nodes_percentage', 0.3))
                
                n_allowed = int(self.config.n_nodes * pct)
                if n_allowed > len(edge_nodes_sorted):
                    print(f"Advertencia: El porcentaje configurado ({pct*100}%) pide {n_allowed} nodos edge, pero solo existen {len(edge_nodes_sorted)}. Se limitará a los disponibles.")
                    n_allowed = len(edge_nodes_sorted)
                n_allowed = max(1, n_allowed)
                allowed_nodes = edge_nodes_sorted[:n_allowed]
        elif mode == 'geographical_aware':
            # Nodos con menor centralidad (edge nodes)
            # Ordenamos los nodos por su centralidad (ascendente, menor centralidad primero)
            sorted_nodes = sorted(self.infra.centrality.keys(), key=lambda k: self.infra.centrality[k])
            pct = getattr(self.config, 'edge_nodes_with_users_percentage', getattr(self.config, 'edge_nodes_percentage', 0.3))
            n_edge_nodes = max(1, int(self.config.n_nodes * pct))
            allowed_nodes = sorted_nodes[:n_edge_nodes]
        else:
            allowed_nodes = list(self.infra.graph.nodes())

        for _ in range(self.config.n_users):
            # Usamos el generador de usuarios para que no interfiera con otros componentes
            user_node = self.rng_users.py_rng.choice(allowed_nodes)
            app_id = self.rng_users.py_rng.choices(range(self.config.n_apps), weights=self.app_popularities, k=1)[0]
            request_ratio = self.rng_users.py_rng.uniform(self.config.min_request_ratio, self.config.max_request_ratio)
            
            if user_node not in self.user_assignments:
                self.user_assignments[user_node] = []
            self.user_assignments[user_node].append({'app_id': app_id, 'request_ratio': request_ratio})

    def calculate_vi_vectors(self):
        for user_node, user_requests in self.user_assignments.items():
            path = nx.shortest_path(self.infra.graph, source=user_node, 
                                    target=self.infra.cloud_node, weight='latency')
            for node in path:
                for req in user_requests:
                    self.node_vectors[node][req['app_id']] = 1