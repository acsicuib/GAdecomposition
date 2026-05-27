import networkx as nx
import random
from simulation_config import SimulationConfig
from random_manager import RandomManager

class Infrastructure:
    def __init__(self, config: SimulationConfig, rng: RandomManager):
        self.config = config
        self.rng = rng
        
        mode = getattr(self.config, 'generation_mode', 'uniform_generation')
        
        if mode == 'tiered_zones':
            self.graph = nx.Graph()
            
            # 1. Crear el CORE (Cloud)
            self.cloud_node = 0
            self.graph.add_node(self.cloud_node, layer='cloud')
            
            # 2. Crear capa FOG (Regionales / Routers intermedios)
            fog_pct = config.tiered_zones["fog"].get("fog_percentage", 0.6)
            n_fog = max(1, int(config.n_nodes * fog_pct))
            fog_nodes = list(range(1, n_fog + 1))
            for f_node in fog_nodes:
                self.graph.add_node(f_node, layer='fog')
                lat = self.rng.py_rng.uniform(
                    config.tiered_zones["fog"]["min_latency_to_cloud"],
                    config.tiered_zones["fog"]["max_latency_to_cloud"]
                )
                self.graph.add_edge(self.cloud_node, f_node, latency=lat)
            
            # 3. Crear capa EDGE (Nodos locales donde se conectan usuarios)
            edge_nodes = list(range(n_fog + 1, config.n_nodes))
            for e_node in edge_nodes:
                self.graph.add_node(e_node, layer='edge')
                parent_fog = self.rng.py_rng.choice(fog_nodes)
                lat = self.rng.py_rng.uniform(
                    config.tiered_zones["edge"]["min_latency_to_fog"],
                    config.tiered_zones["edge"]["max_latency_to_fog"]
                )
                self.graph.add_edge(parent_fog, e_node, latency=lat)
                
            # 4. Enlaces horizontales (Mesh local entre Edges vecinos)
            for i in range(len(edge_nodes)):
                for j in range(i + 1, len(edge_nodes)):
                    if self.graph.nodes[edge_nodes[i]]['layer'] == self.graph.nodes[edge_nodes[j]]['layer']:
                        if self.rng.py_rng.random() < config.tiered_zones["edge"]["mesh_prob"]:
                            lat = self.rng.py_rng.uniform(
                                config.tiered_zones["edge"]["min_mesh_latency"],
                                config.tiered_zones["edge"]["max_mesh_latency"]
                            )
                            self.graph.add_edge(edge_nodes[i], edge_nodes[j], latency=lat)
        else:
            # NetworkX acepta una instancia de random.Random o un entero
            self.graph = nx.barabasi_albert_graph(
                n=config.n_nodes, 
                m=config.m_edges, 
                seed=self.rng.np_rng # Usamos el generador de números aleatorios de infra para numpy
            )
            for node in self.graph.nodes():
                self.graph.nodes[node]['layer'] = 'default'
        
        self.cloud_node = None
        self.node_capacities = {}
        self.node_prices = {}
        self.centrality = {}
        self.c_norm = {}

    def setup_cloud(self):
        self.centrality = nx.closeness_centrality(self.graph)
        self.cloud_node = max(self.centrality, key=self.centrality.get)
        
        c_values = list(self.centrality.values())
        c_min = min(c_values)
        c_max = max(c_values)
        
        for node, c_val in self.centrality.items():
            if c_max > c_min:
                self.c_norm[node] = (c_val - c_min) / (c_max - c_min)
            else:
                self.c_norm[node] = 0.5

    def assign_latencies(self):
        mode = getattr(self.config, 'generation_mode', 'uniform_generation')
        if mode == 'tiered_zones':
            return # Latencias ya asignadas en la creación del grafo
            
        for u, v in self.graph.edges():
            if mode == 'geographical_aware':
                c_edge = (self.c_norm[u] + self.c_norm[v]) / 2.0
                # Nodos periféricos (bajo c_edge) -> Alta latencia
                # Nodos centrales (alto c_edge) -> Baja latencia
                base_lat = self.config.max_latency - c_edge * (self.config.max_latency - self.config.min_latency)
                noise = base_lat * 0.2
                lat = self.rng.py_rng.uniform(base_lat - noise, base_lat + noise)
                lat = max(self.config.min_latency, min(lat, self.config.max_latency))
                self.graph[u][v]['latency'] = lat
            else:
                self.graph[u][v]['latency'] = self.rng.py_rng.uniform(
                    self.config.min_latency, self.config.max_latency
                )

    def assign_capacities(self):
        for node in self.graph.nodes():
            if node == self.cloud_node:
                self.node_capacities[node] = float('inf')
            else:
                mode = getattr(self.config, 'generation_mode', 'uniform_generation')
                if mode == 'tiered_zones':
                    layer = self.graph.nodes[node]['layer']
                    if layer == 'cloud':
                        self.node_capacities[node] = float('inf')
                    else:
                        tz_config = self.config.tiered_zones[layer]
                        self.node_capacities[node] = self.rng.py_rng.randint(
                            tz_config["min_capacity"], tz_config["max_capacity"]
                        )
                elif mode == 'geographical_aware':
                    # Mayor centralidad -> Mayor capacidad
                    base_cap = self.config.min_node_cap + self.c_norm[node] * (self.config.max_node_cap - self.config.min_node_cap)
                    noise = base_cap * 0.2
                    cap = self.rng.py_rng.uniform(base_cap - noise, base_cap + noise)
                    cap = max(self.config.min_node_cap, min(int(cap), self.config.max_node_cap))
                    self.node_capacities[node] = cap
                else:
                    self.node_capacities[node] = self.rng.py_rng.randint(
                        self.config.min_node_cap, self.config.max_node_cap
                    )

    def assign_prices(self):
        for node in self.graph.nodes():
            if node == self.cloud_node:
                self.node_prices[node] = 0
            else:
                mode = getattr(self.config, 'generation_mode', 'uniform_generation')
                if mode == 'tiered_zones':
                    layer = self.graph.nodes[node]['layer']
                    if layer == 'cloud':
                        self.node_prices[node] = self.config.tiered_zones["cloud"]["price"]
                    else:
                        tz_config = self.config.tiered_zones[layer]
                        self.node_prices[node] = self.rng.py_rng.uniform(
                            tz_config["min_price"], tz_config["max_price"]
                        )
                elif mode == 'geographical_aware':
                    # Mayor centralidad -> Menor coste
                    base_price = self.config.max_node_price - self.c_norm[node] * (self.config.max_node_price - self.config.min_node_price)
                    noise = base_price * 0.2
                    price = self.rng.py_rng.uniform(base_price - noise, base_price + noise)
                    price = max(self.config.min_node_price, min(price, self.config.max_node_price))
                    self.node_prices[node] = price
                else:
                    self.node_prices[node] = self.rng.py_rng.randint(
                        self.config.min_node_price, self.config.max_node_price
                    )

    def precompute_shortest_paths(self):
        """
        Calcula previamente las distancias mínimas entre todos los nodos del grafo 
        según el atributo 'latency' para optimizar las consultas posteriores.
        """
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='latency'))
