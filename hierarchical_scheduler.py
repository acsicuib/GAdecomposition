import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform, jaccard
from scipy.cluster.hierarchy import linkage, to_tree
import matplotlib.pyplot as plt
from infrastructure import Infrastructure
from workload import Workload
from simulation_config import SimulationConfig


class HierarchicalScheduler:
    def __init__(self, infrastructure: Infrastructure, workload: Workload, config: SimulationConfig):
        self.infra = infrastructure
        self.workload = workload
        self.config = config
        self.linkage_matrix = None

    def hybrid_distance(self, node_u, node_v):
        dist_lat = self.infra.shortest_paths[node_u][node_v]
        vec_u = self.workload.node_vectors[node_u]
        vec_v = self.workload.node_vectors[node_v]
        
        # Jaccard de scipy/numpy
        if np.all(vec_u == 0) and np.all(vec_v == 0):
            dist_jaccard = 0
        else:
            dist_jaccard = jaccard(vec_u, vec_v)
            
        return self.config.alpha * dist_lat + self.config.beta * dist_jaccard

    def run_clustering(self):
        nodes = list(self.infra.graph.nodes())
        n = len(nodes)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i, j] = dist_matrix[j, i] = self.hybrid_distance(nodes[i], nodes[j])
        
        self.linkage_matrix = linkage(squareform(dist_matrix), method='complete')
        return self.linkage_matrix

    def get_submatrices(self):
        Z = self.linkage_matrix
        n_nodes = self.config.n_nodes
        parent_map = {}
        for i, row in enumerate(Z):
            parent_id = n_nodes + i
            parent_map[int(row[0])] = parent_map[int(row[1])] = parent_id

        def get_leaves(cluster_id):
            if cluster_id < n_nodes: return [cluster_id]
            row = Z[cluster_id - n_nodes]
            return get_leaves(int(row[0])) + get_leaves(int(row[1]))

        submatrices = {}
        for node in self.infra.graph.nodes():
            if node == self.infra.cloud_node:
                continue
            ancestor = node
            for _ in range(self.config.k_levels):
                if ancestor in parent_map: ancestor = parent_map[ancestor]
            
            cluster_nodes = get_leaves(ancestor)
            services = set()
            for n_idx in cluster_nodes:
                if n_idx != self.infra.cloud_node:
                    services.update(np.where(self.workload.node_vectors[n_idx] == 1)[0])
            
            rows = list(cluster_nodes)
            if self.infra.cloud_node not in rows: rows.append(self.infra.cloud_node)
            
            submatrices[node] = {
                "rows": rows,
                "services": sorted(list(services))
            }
        return submatrices