import numpy as np
import networkx as nx

class EvaluateSolution:
    """Clase estática para evaluar las soluciones."""
    
    @staticmethod
    def evaluate(solution: np.ndarray, template: dict, infra, workload) -> list:
        """
        Evalúa los objetivos reales de la solución.
        Asume minimización de los objetivos.
        
        :param solution: Matriz binaria de la solución.
        :param template: Template de la submatriz (contiene las filas/nodos).
        :param infra: Instancia de Infrastructure.
        :param workload: Instancia de Workload.
        :return: Lista con los valores de los objetivos.
        """
        # Verificación estricta de capacidad (las soluciones ahora DEBEN ser siempre viables)
        capacity_penalty = EvaluateSolution._calculate_capacity_penalty(solution, template, infra, workload)
        
        if capacity_penalty > 0:
            raise ValueError(f"Solución inválida detectada: la carga excede la capacidad de al menos un nodo. Con los nuevos operadores de reparación, esto no debería ocurrir nunca.")
            
        # obj1: Mean response time
        mean_response_time = EvaluateSolution._calculate_mean_response_time(solution, template, infra, workload)
        
        # obj2: Total price
        total_price = EvaluateSolution._calculate_total_price(solution, template, infra, workload)
        
        return [mean_response_time, total_price]

    @staticmethod
    def _calculate_mean_response_time(solution: np.ndarray, template: dict, infra, workload) -> float:
        """
        Calcula el tiempo medio de respuesta para cada usuario consultando la menor distancia
        hacia los nodos que tienen el servicio solicitado.
        """
        rows = template.get("rows", [])
        services = template.get("services", [])
        
        total_weighted_response_time = 0.0
        total_request_ratio = 0.0
        
        for user_node in rows:
            if user_node in workload.user_assignments:
                # Cada elemento en la lista corresponde a la petición de un usuario
                for req in workload.user_assignments[user_node]:
                    app_id = req['app_id']
                    ratio = req['request_ratio']
                    total_request_ratio += ratio
                    
                    if app_id not in services:
                        # Si el servicio no está en la plantilla, obligatoriamente se provee desde el cloud
                        total_weighted_response_time += (infra.shortest_paths[user_node][infra.cloud_node] * ratio)
                        continue
                        
                    service_idx = services.index(app_id)
                    
                    # Índices de las filas (nodos) donde está instanciado este servicio (donde hay un 1)
                    hosting_indices = np.where(solution[:, service_idx] == 1)[0]
                    
                    if len(hosting_indices) == 0:
                        # Si ninguna instancia del servicio está en esta solución, 
                        # suponemos que está en el cloud y sumamos la latencia hasta él.
                        total_weighted_response_time += (infra.shortest_paths[user_node][infra.cloud_node] * ratio)
                        continue
                        
                    min_distance = float('inf')
                    for h_idx in hosting_indices:
                        provider_node = rows[h_idx]
                        if provider_node in infra.shortest_paths[user_node]:
                            dist = infra.shortest_paths[user_node][provider_node]
                            if dist < min_distance:
                                min_distance = dist
                            
                    if min_distance == float('inf'):
                        total_weighted_response_time += (infra.shortest_paths[user_node][infra.cloud_node] * ratio)
                    else:
                        total_weighted_response_time += (min_distance * ratio)
                        
        if total_request_ratio == 0:
            return 0.0
            
        return total_weighted_response_time / total_request_ratio

    @staticmethod
    def _calculate_total_price(solution: np.ndarray, template: dict, infra, workload) -> float:
        """
        Calcula el precio total basándose en el coste computacional por instancia.
        Precio = Suma para cada (nodo, app) de (sol[nodo, app] * coste_unidad_nodo * carga_app).
        Esto elimina automáticamente los servicios basura, ya que cada instancia encarece el coste total proporcionalmente.
        """
        total_price = 0.0
        rows = template.get("rows", [])
        services = template.get("services", [])
        
        for i, node_id in enumerate(rows):
            node_unit_price = infra.node_prices.get(node_id, 0.0)
            for j, app_id in enumerate(services):
                if solution[i, j] == 1:
                    app_exec_load = workload.app_execution_loads.get(app_id, 0.0)
                    total_price += (node_unit_price * app_exec_load)
                
        return total_price

    @staticmethod
    def _calculate_capacity_penalty(solution: np.ndarray, template: dict, infra, workload) -> float:
        """
        Calcula una penalización si los servicios alojados en un nodo exceden su capacidad.
        """
        rows = template.get("rows", [])
        services = template.get("services", [])
        
        penalty = 0.0
        
        for i, node_id in enumerate(rows):
            node_capacity = infra.node_capacities.get(node_id, 0)
            if node_capacity == float('inf'):
                continue
                
            # Calcular la demanda total de las aplicaciones instanciadas en este nodo
            node_demand = 0.0
            for j, app_id in enumerate(services):
                if solution[i, j] == 1:
                    node_demand += workload.app_demands.get(app_id, 0)
                    
            if node_demand > node_capacity:
                # Penalización proporcional al exceso de demanda
                penalty += (node_demand - node_capacity) * 1000.0
                
        return penalty