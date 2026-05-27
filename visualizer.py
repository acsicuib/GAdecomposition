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
        pos = nx.spring_layout(G, seed=self.config.seed_infra)
        
        plt.figure(figsize=(12, 8))
        
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

        if cloud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cloud_nodes, 
                                   node_color='red', node_size=800, label='Cloud')
        if fog_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fog_nodes, 
                                   node_color='orange', node_size=600, label='Fog')
        if edge_no_users:
            nx.draw_networkx_nodes(G, pos, nodelist=edge_no_users, 
                                   node_color='skyblue', node_size=500, label='Edge (Sin usuarios)')
        if user_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                   node_color='lightgreen', node_size=500, 
                                   edgecolors='green', linewidths=2, label='Edge (Con usuarios)')

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title(f"Topología del Continuum (Semilla Infra: {self.config.seed_infra})")
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
        pos = nx.spring_layout(G, seed=self.config.seed_infra)
        
        plt.figure(figsize=(16, 12))
        
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

        if cloud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cloud_nodes, 
                                   node_color='red', node_size=2200, label='Cloud')
        if fog_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fog_nodes, 
                                   node_color='orange', node_size=1800, label='Fog')
        if edge_no_users:
            nx.draw_networkx_nodes(G, pos, nodelist=edge_no_users, 
                                   node_color='skyblue', node_size=1500, label='Edge (Sin usuarios)')
        if user_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                   node_color='lightgreen', node_size=1700, 
                                   edgecolors='green', linewidths=2, label='Edge (Con usuarios)')

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Dibujar etiquetas de enlaces (latencias)
        edge_labels = nx.get_edge_attributes(G, 'latency')
        edge_labels = {k: f"{v:.1f}ms" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels, 
            font_size=6, 
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
            
        nx.draw_networkx_labels(G, pos, labels, font_size=7)
        
        plt.title(f"Topología Detallada (Modo: {getattr(self.config, 'generation_mode', 'uniform')}, Semilla Infra: {self.config.seed_infra})")
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
        pos = nx.spring_layout(G, seed=self.config.seed_infra)
        
        plt.figure(figsize=(14, 10))
        
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

        if cloud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cloud_nodes, 
                                   node_color='red', node_size=1200, label='Cloud')
        if fog_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fog_nodes, 
                                   node_color='orange', node_size=1000, label='Fog')
        if edge_no_users:
            nx.draw_networkx_nodes(G, pos, nodelist=edge_no_users, 
                                   node_color='skyblue', node_size=800, label='Edge (Sin usuarios)')
        if user_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                   node_color='lightgreen', node_size=800, 
                                   edgecolors='green', linewidths=2, label='Edge (Con usuarios)')

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(title)
        plt.legend(scatterpoints=1)
        plt.axis('off')
        
        plt.savefig(output_path, format='pdf')
        plt.close()

    def list_users_human_readable(self):
        """Lista usuarios, sus apps y dónde están conectados."""
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║ {'REPORTE DE USUARIOS CONECTADOS':^56} ║")
        print("╠" + "═"*15 + "╦" + "═"*20 + "╦" + "═"*21 + "╣")
        print(f"║ {'USUARIO':<13} ║ {'NODO ASOCIADO':<18} ║ {'APLICACIÓN':<19} ║")
        print("╠" + "═"*15 + "╬" + "═"*20 + "╬" + "═"*21 + "╣")
        
        user_count = 1
        for node, reqs in self.workload.user_assignments.items():
            for req in reqs:
                print(f"║ User_{user_count:<8} ║ Nodo {node:<13} ║ App_{req['app_id']:<14} ║")
                user_count += 1
        print("╚" + "═"*15 + "╩" + "═"*20 + "╩" + "═"*21 + "╝")

    def list_node_capacities(self):
        """Muestra la capacidad de cómputo disponible en cada nodo."""
        print("\n" + "╔" + "═"*45 + "╗")
        print(f"║ {'CAPACIDADES DE CÓMPUTO POR NODO':^43} ║")
        print("╠" + "═"*18 + "╦" + "═"*24 + "╣")
        print(f"║ {'ID NODO':<16} ║ {'CAPACIDAD DISPONIBLE':<22} ║")
        print("╠" + "═"*18 + "╬" + "═"*24 + "╣")
        
        for node in sorted(self.infra.graph.nodes()):
            cap = self.infra.node_capacities[node]
            cap_str = "∞ (Unlimited)" if cap == float('inf') else f"{cap} units"
            tag = " [CLOUD]" if node == self.infra.cloud_node else ""
            print(f"║ Nodo {node:<11} ║ {cap_str:<22} ║{tag}")
            
        print("╚" + "═"*18 + "╩" + "═"*24 + "╝")

    def list_app_requirements(self):
        """Muestra los requerimientos de recursos y carga de cada aplicación."""
        print("\n" + "╔" + "═"*70 + "╗")
        print(f"║ {'REQUERIMIENTOS Y CARGA DE APLICACIONES':^68} ║")
        print("╠" + "═"*20 + "╦" + "═"*22 + "╦" + "═"*24 + "╣")
        print(f"║ {'ID APLICACIÓN':<18} ║ {'DEMANDA RECURSOS':<20} ║ {'CARGA EJECUCIÓN':<22} ║")
        print("╠" + "═"*20 + "╬" + "═"*22 + "╬" + "═"*24 + "╣")
        
        for app_id, demand in sorted(self.workload.app_demands.items()):
            exec_load = self.workload.app_execution_loads.get(app_id, 0.0)
            print(f"║ App_{app_id:<14} ║ {demand:<20} ║ {exec_load:<22.2f} ║")
            
        print("╚" + "═"*20 + "╩" + "═"*22 + "╩" + "═"*24 + "╝")

    def list_node_vectors(self):
        """Muestra el vector V de cada nodo (Apps que pasan por él)."""
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║ {'VECTORES DE AFINIDAD (V) POR NODO':^56} ║")
        print("╠" + "═"*15 + "╦" + "═"*40 + "╣")
        print(f"║ {'ID NODO':<13} ║ {'VECTOR V (Presencia de Apps)':<38} ║")
        print("╠" + "═"*15 + "╬" + "═"*40 + "╬")
        
        for node in sorted(self.infra.graph.nodes()):
            vector = self.workload.node_vectors[node]
            # Formateamos el vector para que sea legible: [0 1 0 0 1...]
            v_str = "[" + " ".join([str(int(x)) for x in vector]) + "]"
            tag = " [CLOUD]" if node == self.infra.cloud_node else ""
            print(f"║ Nodo {node:<8} ║ {v_str:<38} ║{tag}")
            
        print("╚" + "═"*15 + "╩" + "═"*40 + "╝")