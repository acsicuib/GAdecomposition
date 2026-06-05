"""
===============================================================================
File: nsga2.py
Description: Implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II) logic for optimizing the placement of applications in the infrastructure.
Role: Class Definition (Algorithm)
Execution: Imported as a module
Inputs: Infrastructure definitions, Workload data, Population templates.
Outputs: Pareto fronts and non-dominated individuals (solutions).

Authors: Carlos Guerrero and Isaac Lera
Date: June 2026
License: MIT License
===============================================================================
"""
import numpy as np

from evaluate_solution import EvaluateSolution


class NSGA2:
    def __init__(self, rng, template, infra, workload):
        """
        Inicializa la clase encargada de ejecutar el algoritmo NSGA-II.
        
        :param rng: Generador de números aleatorios (instancia de RandomManager).
        :param template: Diccionario con la plantilla de la solución ('rows', 'services').
        :param infra: Instancia de Infrastructure para evaluar las soluciones.
        :param workload: Instancia de Workload para evaluar las soluciones.
        """
        self.rng = rng
        self.template = template
        self.infra = infra
        self.workload = workload
        self.nodes = template.get("rows", [])
        self.services = template.get("services", [])
        self.num_nodes = len(self.nodes)
        self.num_services = len(self.services)
        
        # Histórico: Lista de poblaciones por generación. 
        # Cada población es una lista de diccionarios: {'solution': matriz, 'fitness': lista_objetivos}
        self.history = []
        self.population = []

    def initialize_population(self, pop_size: int):
        """
        Inicializa la población aleatoriamente basándose en el template.
        """
        # Calculamos el tamaño máximo posible de la población (si es pequeño, limitamos)
        total_bits = self.num_nodes * self.num_services
        max_possible = 2 ** total_bits if total_bits < 30 else float('inf')
        actual_pop_size = min(pop_size, max_possible)
        
        self.population = []
        seen_matrices = set()
        attempts = 0
        max_attempts = actual_pop_size * 50 # Límite para no bloquearse buscando las últimas combinaciones
        
        while len(self.population) < actual_pop_size and attempts < max_attempts:
            sol = self._generate_random_feasible_solution()
            mat_bytes = sol.tobytes()
            attempts += 1
            if mat_bytes not in seen_matrices:
                try:
                    fitness = EvaluateSolution.evaluate(sol, self.template, self.infra, self.workload)
                    seen_matrices.add(mat_bytes)
                    self.population.append({
                        'solution': sol,
                        'fitness': fitness
                    })
                except ValueError:
                    pass
            
        # Guardamos la población inicial como la Generación 0
        self._save_to_history()

    def run_generations(self, n_generations: int):
        """
        Ejecuta N generaciones del algoritmo NSGA-II y guarda los resultados en el histórico.
        """
        pop_size = len(self.population)
        if pop_size == 0:
            raise ValueError("La población no ha sido inicializada. Llama a initialize_population primero.")

        for _ in range(n_generations):
            # 1. Generar descendencia (Crossover y Mutación)
            offspring = self._generate_offspring(pop_size)
            
            # 2. Combinar padres e hijos (R_t = P_t U Q_t)
            combined_population = self.population + offspring
            
            # 3. Selección de supervivientes (P_{t+1}) usando clasificación no dominada
            self.population = self._survival_selection(combined_population, pop_size)
            
            # 4. Añadir la población resultante al histórico
            self._save_to_history()

    def _save_to_history(self):
        """Calcula el Pareto front actual y guarda copias profundas de la población y el frente."""
        # Calculamos los frentes para asegurarnos de que la población tiene la clave 'rank'
        self._fast_non_dominated_sort(self.population)
        
        pop_copy = []
        pareto_front = []
        for ind in self.population:
            # Copia profunda para que el histórico sea inmutable
            ind_copy = {
                'solution': ind['solution'].copy(),
                'fitness': list(ind['fitness']),
                'rank': ind.get('rank', 0),
                'crowding_distance': ind.get('crowding_distance', 0.0)
            }
            pop_copy.append(ind_copy)
            
            # Si es del primer frente (no dominado), va al pareto front
            if ind_copy['rank'] == 0:
                pareto_front.append(ind_copy)
                
        self.history.append({
            'population': pop_copy,
            'pareto_front': pareto_front
        })

    def _generate_offspring(self, num_offspring: int):
        """
        Genera hijos mediante torneo binario simple, cruce uniforme y mutación bit-flip.
        """
        offspring = []
        pop_size = len(self.population)
        
        # Obtenemos generadores del random_manager
        rnd = self.rng.py_rng if hasattr(self.rng, 'py_rng') else __import__('random')
        np_rnd = self.rng.np_rng if hasattr(self.rng, 'np_rng') else np.random
        
        while len(offspring) < num_offspring:
            # Selección por torneo binario aleatorio
            p1_idx = rnd.randint(0, pop_size - 1)
            p2_idx = rnd.randint(0, pop_size - 1)
            
            parent1 = self.population[p1_idx]['solution']
            parent2 = self.population[p2_idx]['solution']
            
            # Cruce uniforme (mascara booleana)
            mask = np_rnd.integers(0, 2, size=(self.num_nodes, self.num_services)).astype(bool)
            child_sol = np.where(mask, parent1, parent2)
            
            # Reparación tras cruce y cálculo de cargas
            child_sol, loads = self._repair_solution(child_sol)
            
            # Mutación inteligente para asegurar viabilidad
            total_elements = self.num_nodes * self.num_services
            mutation_prob = 1.0 / total_elements if total_elements > 0 else 0.1
            
            child_sol = self._smart_mutation(child_sol, mutation_prob, loads)
            
            try:
                # Evaluar hijo
                child_fitness = EvaluateSolution.evaluate(child_sol, self.template, self.infra, self.workload)
                offspring.append({
                    'solution': child_sol,
                    'fitness': child_fitness
                })
            except ValueError:
                # Descartar la solución inválida y continuar buscando
                continue
            
        return offspring

    def _survival_selection(self, combined_population, target_size, allow_random_fill=True):
        """
        Selecciona los 'target_size' mejores individuos de la población combinada
        usando fast non-dominated sorting y crowding distance. Prioriza eliminar duplicados absolutos.
        """
        unique_pop = []
        seen_matrices = set()
        for ind in combined_population:
            mat_bytes = ind['solution'].tobytes()
            if mat_bytes not in seen_matrices:
                seen_matrices.add(mat_bytes)
                unique_pop.append(ind)
                
        # Rellenar con individuos aleatorios si nos quedamos sin suficientes por borrar duplicados
        if allow_random_fill:
            total_bits = self.num_nodes * self.num_services
            max_possible = 2 ** total_bits if total_bits < 30 else float('inf')
            effective_target = min(target_size, max_possible)
            
            attempts = 0
            max_attempts = effective_target * 50
            
            while len(unique_pop) < effective_target and attempts < max_attempts:
                sol = self._generate_random_feasible_solution()
                mat_bytes = sol.tobytes()
                attempts += 1
                if mat_bytes not in seen_matrices:
                    try:
                        fit = EvaluateSolution.evaluate(sol, self.template, self.infra, self.workload)
                        seen_matrices.add(mat_bytes)
                        unique_pop.append({
                            'solution': sol,
                            'fitness': fit
                        })
                    except ValueError:
                        pass
                    
        # Actualizamos target_size si unique_pop es menor y no rellenamos
        target_size = min(target_size, len(unique_pop))
        
        fronts = self._fast_non_dominated_sort(unique_pop)
        
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= target_size:
                # Si todo el frente cabe, se añade directamente
                for idx in front:
                    new_population.append(unique_pop[idx])
            else:
                # Si el frente actual no cabe entero, calculamos crowding distance para desempate
                self._calculate_crowding_distance(unique_pop, front)
                # Ordenamos el frente actual en base a la crowding distance de mayor a menor
                front.sort(key=lambda x: unique_pop[x].get('crowding_distance', 0), reverse=True)
                
                # Extraemos solo los que necesitamos para rellenar la población
                needed = target_size - len(new_population)
                for i in range(needed):
                    new_population.append(unique_pop[front[i]])
                break
                
        return new_population

    def _fast_non_dominated_sort(self, population):
        """
        Algoritmo de clasificación de frentes no dominados.
        Retorna una lista de listas de índices, donde la primera lista 
        es el primer frente (los no dominados), etc.
        """
        fronts = [[]]
        S = [[] for _ in range(len(population))]
        n = [0] * len(population)
        
        for p in range(len(population)):
            for q in range(len(population)):
                if self._dominates(population[p]['fitness'], population[q]['fitness']):
                    S[p].append(q)
                elif self._dominates(population[q]['fitness'], population[p]['fitness']):
                    n[p] += 1
            if n[p] == 0:
                population[p]['rank'] = 0
                fronts[0].append(p)
                
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        population[q]['rank'] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)
            
        return fronts[:-1] # El último siempre será una lista vacía

    def _dominates(self, fit1, fit2):
        """
        Verifica si la solución con fitness 'fit1' domina a 'fit2'.
        Asume que TODOS los objetivos se deben MINIMIZAR.
        """
        and_condition = True
        or_condition = False
        for f1, f2 in zip(fit1, fit2):
            if f1 > f2:
                and_condition = False
                break
            if f1 < f2:
                or_condition = True
        return and_condition and or_condition

    def _calculate_crowding_distance(self, population, front):
        """
        Calcula la métrica de Crowding Distance para mantener la diversidad 
        en el frente de Pareto.
        """
        l = len(front)
        if l == 0: return
        
        # Inicializar a 0
        for i in front:
            population[i]['crowding_distance'] = 0
            
        num_objectives = len(population[front[0]]['fitness'])
        
        # Para cada objetivo
        for m in range(num_objectives):
            # Ordenar individuos del frente respecto al objetivo m
            front.sort(key=lambda x: population[x]['fitness'][m])
            
            # Los extremos tienen distancia infinita (siempre se mantienen)
            population[front[0]]['crowding_distance'] = float('inf')
            population[front[-1]]['crowding_distance'] = float('inf')
            
            f_min = population[front[0]]['fitness'][m]
            f_max = population[front[-1]]['fitness'][m]
            
            if f_max - f_min == 0:
                continue
                
            # Distancia para los puntos intermedios
            for i in range(1, l - 1):
                population[front[i]]['crowding_distance'] += (
                    (population[front[i + 1]]['fitness'][m] - population[front[i - 1]]['fitness'][m]) / (f_max - f_min)
                )

    def _generate_random_feasible_solution(self):
        """Genera una solución 100% viable utilizando un esquema similar a FFD,
        pero introduciendo aleatoriedad para la población inicial."""
        sol = np.zeros((self.num_nodes, self.num_services), dtype=int)
        # Ordenamos índices de servicios por demanda (descendente)
        services_sorted_indices = sorted(
            range(self.num_services), 
            key=lambda idx: self.workload.app_demands.get(self.services[idx], 0), 
            reverse=True
        )
        
        rnd = self.rng.py_rng if hasattr(self.rng, 'py_rng') else __import__('random')
        
        for i, node in enumerate(self.nodes):
            capacity = self.infra.node_capacities.get(node, 0)
            current_load = 0
            for j in services_sorted_indices:
                if rnd.random() < 0.5:
                    demand = self.workload.app_demands.get(self.services[j], 0)
                    if current_load + demand <= capacity:
                        sol[i, j] = 1
                        current_load += demand
        return sol

    def _repair_solution(self, sol):
        """Repara una solución pos-cruce para asegurar que no excede capacidades."""
        rnd = self.rng.py_rng if hasattr(self.rng, 'py_rng') else __import__('random')
        
        # Calcular cargas actuales
        loads = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            for j in range(self.num_services):
                if sol[i, j] == 1:
                    loads[i] += self.workload.app_demands.get(self.services[j], 0)
                    
        for i in range(self.num_nodes):
            node = self.nodes[i]
            capacity = self.infra.node_capacities.get(node, 0)
            
            while loads[i] > capacity:
                present_services = [j for j in range(self.num_services) if sol[i, j] == 1]
                if not present_services:
                    break
                    
                j_to_move = rnd.choice(present_services)
                demand_j = self.workload.app_demands.get(self.services[j_to_move], 0)
                
                # Descartar de i
                sol[i, j_to_move] = 0
                loads[i] -= demand_j
                
                other_nodes = list(range(self.num_nodes))
                other_nodes.remove(i)
                rnd.shuffle(other_nodes)
                
                placed = False
                for k in other_nodes:
                    if sol[k, j_to_move] == 0:
                        node_k = self.nodes[k]
                        cap_k = self.infra.node_capacities.get(node_k, 0)
                        if loads[k] + demand_j <= cap_k:
                            sol[k, j_to_move] = 1
                            loads[k] += demand_j
                            placed = True
                            break
                            
                if not placed:
                    # Intentar intercambio (swap)
                    for k in other_nodes:
                        node_k = self.nodes[k]
                        cap_k = self.infra.node_capacities.get(node_k, 0)
                        services_k = [l for l in range(self.num_services) if sol[k, l] == 1 and sol[i, l] == 0]
                        for l_to_move in services_k:
                            demand_l = self.workload.app_demands.get(self.services[l_to_move], 0)
                            if loads[k] + demand_j - demand_l <= cap_k and loads[i] + demand_l <= capacity:
                                sol[k, j_to_move] = 1
                                sol[k, l_to_move] = 0
                                sol[i, l_to_move] = 1
                                loads[k] = loads[k] + demand_j - demand_l
                                loads[i] = loads[i] + demand_l
                                placed = True
                                break
                        if placed:
                            break
                            
        return sol, loads

    def _smart_mutation(self, sol, mutation_prob, loads):
        """Aplica mutaciones asegurando que no se rompe la viabilidad de capacidad."""
        rnd = self.rng.py_rng if hasattr(self.rng, 'py_rng') else __import__('random')
        np_rnd = self.rng.np_rng if hasattr(self.rng, 'np_rng') else np.random
        
        mut_mask = np_rnd.random(size=(self.num_nodes, self.num_services)) < mutation_prob
        
        for i in range(self.num_nodes):
            node = self.nodes[i]
            capacity = self.infra.node_capacities.get(node, 0)
            
            for j in range(self.num_services):
                if mut_mask[i, j]:
                    demand_j = self.workload.app_demands.get(self.services[j], 0)
                    if sol[i, j] == 1:
                        sol[i, j] = 0
                        loads[i] -= demand_j
                    else:
                        if loads[i] + demand_j <= capacity:
                            sol[i, j] = 1
                            loads[i] += demand_j
                        else:
                            other_nodes = list(range(self.num_nodes))
                            other_nodes.remove(i)
                            rnd.shuffle(other_nodes)
                            
                            placed = False
                            for k in other_nodes:
                                if sol[k, j] == 0:
                                    node_k = self.nodes[k]
                                    cap_k = self.infra.node_capacities.get(node_k, 0)
                                    if loads[k] + demand_j <= cap_k:
                                        sol[k, j] = 1
                                        loads[k] += demand_j
                                        placed = True
                                        break
                                        
                            if not placed:
                                for k in other_nodes:
                                    node_k = self.nodes[k]
                                    cap_k = self.infra.node_capacities.get(node_k, 0)
                                    services_k = [l for l in range(self.num_services) if sol[k, l] == 1 and sol[i, l] == 0]
                                    for l_to_move in services_k:
                                        demand_l = self.workload.app_demands.get(self.services[l_to_move], 0)
                                        if loads[k] + demand_j - demand_l <= cap_k and loads[i] + demand_l <= capacity:
                                            sol[k, j] = 1
                                            sol[k, l_to_move] = 0
                                            sol[i, l_to_move] = 1
                                            loads[k] = loads[k] + demand_j - demand_l
                                            loads[i] = loads[i] + demand_l
                                            placed = True
                                            break
                                    if placed:
                                        break
        return sol

