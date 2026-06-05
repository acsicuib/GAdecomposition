"""
===============================================================================
File: visualizer.py
Description: Provides plotting capabilities to visualize network topologies, application clusters, and deployment schemas.
Role: Class Definition (Utility)
Execution: Imported as a module
Inputs: Network graphs, Dendrogram data.
Outputs: Matplotlib figures and visual schema images.

Authors: Carlos Guerrero and Isaac Lera
Date: June 2026
License: MIT License
===============================================================================
"""
import matplotlib.pyplot as plt
import networkx as nx
from infrastructure import Infrastructure
from workload import Workload
from simulation_config import SimulationConfig

class Visualizer:
    def __init__(self, infrastructure, workload, config):
        self.infra = infrastructure
        self.workload = workload
        self.config = config

    def draw_infrastructure(self, output_path=None):
        """Dibuja el grafo con un layout consistente basado en la semilla de infra (vista simple)."""
        G = self.infra.graph
        n_nodes = len(G.nodes())
        
        # Dynamic spring layout distance k to avoid overlaps
        k_val = 2.5 / (n_nodes ** 0.5) if n_nodes > 0 else 0.1
        pos = nx.spring_layout(G, seed=self.config.seed_infra, k=k_val, iterations=100)
        
        # Dynamic figure size
        fig_width = max(12, min(36, int(n_nodes * 0.15)))
        fig_height = max(8, min(24, int(n_nodes * 0.10)))
        plt.figure(figsize=(fig_width, fig_height))
        
        nodes = list(G.nodes())
        user_nodes = list(self.workload.user_assignments.keys())
        
        # Categorizar nodos
        cloud_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'cloud']
        if not cloud_nodes:
            cloud_nodes = [self.infra.cloud_node]
            
        fog_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'fog']
        edge_no_users = [n for n, d in G.nodes(data=True) if d.get('layer') == 'edge' and n not in user_nodes]
        
        # Fallback para modos no-tiered
        if not fog_nodes and not edge_no_users:
            edge_no_users = [n for n in nodes if n not in cloud_nodes and n not in user_nodes]
            
        # Scale node size based on number of nodes
        size_scale = max(0.15, min(1.0, 50.0 / n_nodes)) if n_nodes > 0 else 1.0
        c_size = int(800 * size_scale)
        f_size = int(600 * size_scale)
        e_size = int(500 * size_scale)
        u_size = int(500 * size_scale)
        f_font = max(4, min(10, int(300 / n_nodes))) if n_nodes > 0 else 10

        if cloud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cloud_nodes, 
                                   node_color='red', node_size=c_size, label='Cloud')
        if fog_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fog_nodes, 
                                   node_color='orange', node_size=f_size, label='Fog')
        if edge_no_users:
            nx.draw_networkx_nodes(G, pos, nodelist=edge_no_users, 
                                   node_color='skyblue', node_size=e_size, label='Edge (No users)')
        if user_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                   node_color='lightgreen', node_size=u_size, 
                                   edgecolors='green', linewidths=max(0.5, 2.0 * size_scale), label='Edge (With users)')

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=f_font)
        
        plt.legend(scatterpoints=1)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, format='pdf')
            plt.close()
        else:
            plt.show()

    def draw_infrastructure_detailed(self, output_path=None):
        """Dibuja el grafo mostrando parámetros detallados de nodos, enlaces y usuarios."""
        G = self.infra.graph
        n_nodes = len(G.nodes())
        
        # Dynamic spring layout distance k to avoid overlaps
        k_val = 3.5 / (n_nodes ** 0.5) if n_nodes > 0 else 0.1
        pos = nx.spring_layout(G, seed=self.config.seed_infra, k=k_val, iterations=100)
        
        # Dynamic figure size (much larger for detailed view with text!)
        fig_width = max(16, min(48, int(n_nodes * 0.20)))
        fig_height = max(12, min(36, int(n_nodes * 0.15)))
        plt.figure(figsize=(fig_width, fig_height))
        
        nodes = list(G.nodes())
        user_nodes = list(self.workload.user_assignments.keys())
        
        # Categorizar nodos
        cloud_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'cloud']
        if not cloud_nodes:
            cloud_nodes = [self.infra.cloud_node]
            
        fog_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'fog']
        edge_no_users = [n for n, d in G.nodes(data=True) if d.get('layer') == 'edge' and n not in user_nodes]
        
        # Fallback para modos no-tiered
        if not fog_nodes and not edge_no_users:
            edge_no_users = [n for n in nodes if n not in cloud_nodes and n not in user_nodes]

        # Scale node size based on number of nodes
        size_scale = max(0.15, min(1.0, 50.0 / n_nodes)) if n_nodes > 0 else 1.0
        c_size = int(2200 * size_scale)
        f_size = int(1800 * size_scale)
        e_size = int(1500 * size_scale)
        u_size = int(1700 * size_scale)
        f_font = max(3, min(7, int(250 / n_nodes))) if n_nodes > 0 else 7

        if cloud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cloud_nodes, 
                                   node_color='red', node_size=c_size, label='Cloud')
        if fog_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fog_nodes, 
                                   node_color='orange', node_size=f_size, label='Fog')
        if edge_no_users:
            nx.draw_networkx_nodes(G, pos, nodelist=edge_no_users, 
                                   node_color='skyblue', node_size=e_size, label='Edge (No users)')
        if user_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                   node_color='lightgreen', node_size=u_size, 
                                   edgecolors='green', linewidths=max(0.5, 2.0 * size_scale), label='Edge (With users)')

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Dibujar etiquetas de enlaces (latencias) - Omitir si hay demasiados enlaces para evitar colapso visual
        if n_nodes <= 60:
            edge_labels = nx.get_edge_attributes(G, 'latency')
            edge_labels = {k: f"{v:.1f}ms" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(
                G, pos, 
                edge_labels=edge_labels, 
                font_size=max(3, min(6, int(180 / n_nodes))) if n_nodes > 0 else 6, 
                font_color='dimgrey',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )
        
        # Custom labels para nodos
        labels = {}
        for n in G.nodes():
            cap = self.infra.node_capacities[n]
            cap_str = "inf" if cap == float('inf') else f"{cap}"
            price = self.infra.node_prices[n]
            price_str = f"{price:.1f}" if isinstance(price, (int, float)) else str(price)
            
            label_parts = [f"N{n}"]
            label_parts.append(f"C:{cap_str} P:{price_str}")
            
            # Usuarios y aplicaciones asociadas
            if n in self.workload.user_assignments and self.workload.user_assignments[n]:
                from collections import defaultdict
                app_groups = defaultdict(list)
                for req in self.workload.user_assignments[n]:
                    app_groups[req['app_id']].append(req['request_ratio'])
                
                group_strings = []
                for app_id in sorted(app_groups.keys()):
                    ratios = app_groups[app_id]
                    # Formato: A{app_id}:{count}({total_ratio:.1f})
                    group_strings.append(f"A{app_id}:{len(ratios)}({sum(ratios):.1f})")
                
                # Dividir en líneas de máximo 2 elementos para legibilidad
                usr_lines = []
                for i in range(0, len(group_strings), 2):
                    usr_lines.append(" ".join(group_strings[i:i+2]))
                label_parts.append("Usrs:")
                label_parts.extend(usr_lines)
            
            labels[n] = "\n".join(label_parts)
            
        nx.draw_networkx_labels(G, pos, labels, font_size=f_font)
        
        plt.legend(scatterpoints=1)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, format='pdf')
            plt.close()
        else:
            plt.show()

    def draw_solution(self, solution, template, output_path, title="Solution"):
        """Dibuja el grafo mostrando los servicios alojados en cada nodo según la solución."""
        G = self.infra.graph
        n_nodes = len(G.nodes())
        
        # Dynamic spring layout distance k to avoid overlaps
        k_val = 3.0 / (n_nodes ** 0.5) if n_nodes > 0 else 0.1
        pos = nx.spring_layout(G, seed=self.config.seed_infra, k=k_val, iterations=100)
        
        # Dynamic figure size
        fig_width = max(14, min(42, int(n_nodes * 0.18)))
        fig_height = max(10, min(30, int(n_nodes * 0.13)))
        plt.figure(figsize=(fig_width, fig_height))
        
        # Mapear nodo a servicios alojados
        node_to_services = {}
        for i, node in enumerate(template['rows']):
            hosted = []
            for j, srv in enumerate(template['services']):
                if solution[i, j] == 1:
                    hosted.append(srv)
            node_to_services[node] = hosted
            
        nodes = list(G.nodes())
        edge_nodes = [n for n in nodes if n != self.infra.cloud_node]
        
        # Custom labels
        labels = {}
        for n in nodes:
            srvs = node_to_services.get(n, [])
            label_parts = [f"{n}"]
            if srvs:
                srv_str = ",".join([str(s) for s in srvs])
                label_parts.append(f"Apps:{srv_str}")
                
            # Usuarios conectados y aplicaciones solicitadas
            if n in self.workload.user_assignments and self.workload.user_assignments[n]:
                req_list = self.workload.user_assignments[n]
                n_users = len(req_list)
                requested_apps = sorted(list(set(req['app_id'] for req in req_list)))
                req_app_str = ",".join([str(a) for a in requested_apps])
                label_parts.append(f"Usr:{n_users}({req_app_str})")
                
            labels[n] = "\n".join(label_parts)
                
        user_nodes = list(self.workload.user_assignments.keys())
        
        # Categorizar nodos
        cloud_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'cloud']
        if not cloud_nodes:
            cloud_nodes = [self.infra.cloud_node]
            
        fog_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'fog']
        edge_no_users = [n for n, d in G.nodes(data=True) if d.get('layer') == 'edge' and n not in user_nodes]
        
        # Fallback para modos no-tiered
        if not fog_nodes and not edge_no_users:
            edge_no_users = [n for n in nodes if n not in cloud_nodes and n not in user_nodes]

        # Scale node size based on number of nodes
        size_scale = max(0.15, min(1.0, 50.0 / n_nodes)) if n_nodes > 0 else 1.0
        c_size = int(1200 * size_scale)
        f_size = int(1000 * size_scale)
        e_size = int(800 * size_scale)
        u_size = int(800 * size_scale)
        f_font = max(3, min(8, int(250 / n_nodes))) if n_nodes > 0 else 8

        if cloud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cloud_nodes, 
                                   node_color='red', node_size=c_size, label='Cloud')
        if fog_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fog_nodes, 
                                   node_color='orange', node_size=f_size, label='Fog')
        if edge_no_users:
            nx.draw_networkx_nodes(G, pos, nodelist=edge_no_users, 
                                   node_color='skyblue', node_size=e_size, label='Edge (No users)')
        if user_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                   node_color='lightgreen', node_size=u_size, 
                                   edgecolors='green', linewidths=max(0.5, 2.0 * size_scale), label='Edge (With users)')

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, labels, font_size=f_font)
        
        plt.legend(scatterpoints=1)
        plt.axis('off')
        
        plt.savefig(output_path, format='pdf')
        plt.close()

    def list_users_human_readable(self):
        """Lista usuarios, sus apps y dónde están conectados."""
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║ {'CONNECTED USERS REPORT':^56} ║")
        print("╠" + "═"*15 + "╦" + "═"*20 + "╦" + "═"*21 + "╣")
        print(f"║ {'USER':<13} ║ {'ASSOCIATED NODE':<18} ║ {'APPLICATION':<19} ║")
        print("╠" + "═"*15 + "╬" + "═"*20 + "╬" + "═"*21 + "╣")
        
        user_count = 1
        for node, reqs in self.workload.user_assignments.items():
            for req in reqs:
                print(f"║ User_{user_count:<8} ║ Node {node:<13} ║ App_{req['app_id']:<14} ║")
                user_count += 1
        print("╚" + "═"*15 + "╩" + "═"*20 + "╩" + "═"*21 + "╝")

    def list_node_capacities(self):
        """Muestra la capacidad de cómputo disponible en cada nodo."""
        print("\n" + "╔" + "═"*45 + "╗")
        print(f"║ {'NODE COMPUTE CAPACITIES':^43} ║")
        print("╠" + "═"*18 + "╦" + "═"*24 + "╣")
        print(f"║ {'NODE ID':<16} ║ {'AVAILABLE CAPACITY':<22} ║")
        print("╠" + "═"*18 + "╬" + "═"*24 + "╣")
        
        for node in sorted(self.infra.graph.nodes()):
            cap = self.infra.node_capacities[node]
            cap_str = "∞ (Unlimited)" if cap == float('inf') else f"{cap} units"
            tag = " [CLOUD]" if node == self.infra.cloud_node else ""
            print(f"║ Node {node:<11} ║ {cap_str:<22} ║{tag}")
            
        print("╚" + "═"*18 + "╩" + "═"*24 + "╝")

    def list_app_requirements(self):
        """Muestra los requerimientos de recursos y carga de cada aplicación."""
        print("\n" + "╔" + "═"*70 + "╗")
        print(f"║ {'APPLICATION REQUIREMENTS AND LOAD':^68} ║")
        print("╠" + "═"*20 + "╦" + "═"*22 + "╦" + "═"*24 + "╣")
        print(f"║ {'APPLICATION ID':<18} ║ {'RESOURCE DEMAND':<20} ║ {'EXECUTION LOAD':<22} ║")
        print("╠" + "═"*20 + "╬" + "═"*22 + "╬" + "═"*24 + "╣")
        
        for app_id, demand in sorted(self.workload.app_demands.items()):
            exec_load = self.workload.app_execution_loads.get(app_id, 0.0)
            print(f"║ App_{app_id:<14} ║ {demand:<20} ║ {exec_load:<22.2f} ║")
            
        print("╚" + "═"*20 + "╩" + "═"*22 + "╩" + "═"*24 + "╝")

    def list_node_vectors(self):
        """Muestra el vector V de cada nodo (Apps que pasan por él)."""
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║ {'AFFINITY VECTORS (V) PER NODE':^56} ║")
        print("╠" + "═"*15 + "╦" + "═"*40 + "╣")
        print(f"║ {'NODE ID':<13} ║ {'V VECTOR (App Presence)':<38} ║")
        print("╠" + "═"*15 + "╬" + "═"*40 + "╬")
        
        for node in sorted(self.infra.graph.nodes()):
            vector = self.workload.node_vectors[node]
            # Formateamos el vector para que sea legible: [0 1 0 0 1...]
            v_str = "[" + " ".join([str(int(x)) for x in vector]) + "]"
            tag = " [CLOUD]" if node == self.infra.cloud_node else ""
            print(f"║ Node {node:<8} ║ {v_str:<38} ║{tag}")
            
        print("╚" + "═"*15 + "╩" + "═"*40 + "╝")