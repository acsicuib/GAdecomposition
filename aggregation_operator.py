import numpy as np
from evaluate_solution import EvaluateSolution
from nsga2 import NSGA2
from random_manager import RandomManager

class HierarchicalAggregator:
    def __init__(self, infra, workload, config, nsga2_instances, linkage_matrix):
        self.infra = infra
        self.workload = workload
        self.config = config
        self.nsga2_instances = nsga2_instances
        self.Z = linkage_matrix
        
        # Precalcular la demanda total de peticiones por app (para desempatar)
        self.app_total_requests = {j: 0.0 for j in range(self.config.n_apps)}
        for node, reqs in self.workload.user_assignments.items():
            for req in reqs:
                self.app_total_requests[req['app_id']] += req['request_ratio']
                
        # Definir una plantilla global
        self.global_template = {
            'rows': list(self.infra.graph.nodes()),
            'services': list(range(self.config.n_apps))
        }
        
        # Instancia de ayuda para reutilizar funciones NSGA2
        dummy_rng = RandomManager(0)
        self.nsga2_helper = NSGA2(dummy_rng, self.global_template, self.infra, self.workload)

    def combine_submatrices(self, sol1, temp1, sol2, temp2):
        """
        Combina dos submatrices creando un nuevo template que es la unión de sus filas y servicios.
        Retorna la matriz conjunta y el nuevo template asociado.
        """
        new_rows = sorted(list(set(temp1['rows']).union(set(temp2['rows']))))
        new_services = sorted(list(set(temp1['services']).union(set(temp2['services']))))
        
        new_template = {'rows': new_rows, 'services': new_services}
        
        # Mapeos inversos para la nueva matriz
        row_map = {r: i for i, r in enumerate(new_rows)}
        srv_map = {s: i for i, s in enumerate(new_services)}
        
        # Iniciar con -1 (zona sin jurisdicción)
        G_joint = np.full((len(new_rows), len(new_services)), -1)
        
        # Volcar sol1
        for i, r in enumerate(temp1['rows']):
            for j, s in enumerate(temp1['services']):
                G_joint[row_map[r], srv_map[s]] = sol1[i, j]
                
        # Superponer sol2 y detectar conflictos
        for i, r in enumerate(temp2['rows']):
            for j, s in enumerate(temp2['services']):
                new_i, new_j = row_map[r], srv_map[s]
                v1 = G_joint[new_i, new_j]
                v2 = sol2[i, j]
                
                if v1 == -1:
                    G_joint[new_i, new_j] = v2
                elif v1 != v2:
                    G_joint[new_i, new_j] = -2 # Conflicto (1 vs 0)
                    
        # Resolver colisiones evaluando capacidad del nodo
        for i in range(G_joint.shape[0]):
            conflicts = np.where(G_joint[i] == -2)[0]
            if len(conflicts) > 0:
                # Demanda actual de los servicios ya aceptados (1s)
                current_demand = sum(self.workload.app_demands.get(new_services[j], 0) 
                                     for j in range(G_joint.shape[1]) if G_joint[i, j] == 1)
                
                # Ordenar aplicaciones en conflicto de más solicitada a menos solicitada
                conflicts = sorted(conflicts, key=lambda j_idx: self.app_total_requests[new_services[j_idx]], reverse=True)
                
                node_id = new_rows[i]
                capacity = self.infra.node_capacities.get(node_id, 0)
                
                for j in conflicts:
                    app_demand = self.workload.app_demands.get(new_services[j], 0)
                    if current_demand + app_demand <= capacity:
                        G_joint[i, j] = 1
                        current_demand += app_demand
                    else:
                        G_joint[i, j] = 0
                        
        # Cualquier celda que siga en -1 significa que ninguno de los hijos la proponía (zonas exclusivas fuera de su radar)
        G_joint[G_joint == -1] = 0
                        
        return G_joint, new_template

    def aggregate_step(self):
        """Ejecuta todos los pasos de agregación escalable y retorna el diccionario de frentes y los tiempos de fusión."""
        import time
        fronts = {}
        merge_times = {}
        n_nodes = self.config.n_nodes
        
        # Mapear frentes de hojas conservando sus dimensiones locales
        for i in range(n_nodes):
            if i == self.infra.cloud_node:
                fronts[i] = []
                continue
            nsga = self.nsga2_instances[i]
            pf = nsga.history[-1]['pareto_front']
            local_pf = []
            for ind in pf:
                sol = ind['solution'].copy()
                temp = nsga.template
                fit = EvaluateSolution.evaluate(sol, temp, self.infra, self.workload)
                local_pf.append({
                    'solution': sol,
                    'template': temp,
                    'fitness': fit
                })
            
            if local_pf:
                self.nsga2_helper._fast_non_dominated_sort(local_pf)
                local_pf = [ind for ind in local_pf if ind.get('rank', 0) == 0]
            fronts[i] = local_pf
            
        # Combinar según dendrograma
        for k, row in enumerate(self.Z):
            c1, c2 = int(row[0]), int(row[1])
            c_parent = n_nodes + k
            
            t_merge_start = time.perf_counter()
            
            if c1 == self.infra.cloud_node:
                fronts[c_parent] = fronts[c2]
                t_merge_end = time.perf_counter()
                merge_times[c_parent] = t_merge_end - t_merge_start
                print(f"    [HC-PA] Agregando {c1} (Cloud Bypass) y {c2} -> {c_parent} | Front size: {len(fronts[c_parent])} | Time: {merge_times[c_parent]:.4f}s")
                continue
            elif c2 == self.infra.cloud_node:
                fronts[c_parent] = fronts[c1]
                t_merge_end = time.perf_counter()
                merge_times[c_parent] = t_merge_end - t_merge_start
                print(f"    [HC-PA] Agregando {c1} y {c2} (Cloud Bypass) -> {c_parent} | Front size: {len(fronts[c_parent])} | Time: {merge_times[c_parent]:.4f}s")
                continue
            
            # Step 1: Pre-filtrar paretos hijos si superan aggregator_front_size
            front_c1 = fronts[c1]
            front_c2 = fronts[c2]
            
            if len(front_c1) > self.config.aggregator_front_size:
                front_c1 = self.nsga2_helper._survival_selection(front_c1, self.config.aggregator_front_size, allow_random_fill=False)
            
            if len(front_c2) > self.config.aggregator_front_size:
                front_c2 = self.nsga2_helper._survival_selection(front_c2, self.config.aggregator_front_size, allow_random_fill=False)
                
            # Step 2: Combinatorial Cross-Product
            P_joint = []
            for ind1 in front_c1:
                for ind2 in front_c2:
                    # Step 2: Spatial Conflict Resolution con crecimiento de la submatriz
                    G_joint, new_template = self.combine_submatrices(
                        ind1['solution'], ind1['template'],
                        ind2['solution'], ind2['template']
                    )
                    
                    # Step 3: Evaluar fitness con la nueva plantilla fusionada
                    fit = EvaluateSolution.evaluate(G_joint, new_template, self.infra, self.workload)
                    P_joint.append({
                        'solution': G_joint,
                        'template': new_template,
                        'fitness': fit
                    })
                    
            # Step 3 & 4: Deduplicación absoluta, Pareto Dominance Filtering & Crowding Distance Truncation
            if len(P_joint) > 0:
                P_joint = self.nsga2_helper._survival_selection(P_joint, self.config.pop_size, allow_random_fill=False)
                
            fronts[c_parent] = [ind for ind in P_joint if ind.get('rank', 0) == 0]
            
            t_merge_end = time.perf_counter()
            merge_times[c_parent] = t_merge_end - t_merge_start
            
            print(f"    [HC-PA] Agregando {c1} y {c2} -> {c_parent} | Front size: {len(fronts[c_parent])} | Time: {merge_times[c_parent]:.4f}s")
            
        return fronts, merge_times
