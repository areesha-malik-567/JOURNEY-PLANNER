import heapq
import sys

# Graph Class
class Graph:
    def _init_(self):
        self.graph = {}
    
    # Add a location (node)
    def add_location(self, location):
        if location not in self.graph:
            self.graph[location] = {}
    
    # Add a route (edge with weights)
    def add_route(self, start, end, distance, cost=0, time=0):
        self.add_location(start)
        self.add_location(end)
        self.graph[start][end] = {'distance': distance, 'cost': cost, 'time': time}
        self.graph[end][start] = {'distance': distance, 'cost': cost, 'time': time}  # Undirected graph

    # Display the graph (for debugging)
    def display_graph(self):
        for location, routes in self.graph.items():
            print(f"{location}: {routes}")

# Dijkstra's Algorithm for Shortest Path
def dijkstra(graph, start, destination, weight='distance'):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    previous_nodes = {node: None for node in graph}
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weights in graph[current_node].items():
            distance = current_distance + weights[weight]
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Reconstruct the path
    path = []
    node = destination
    while previous_nodes[node]:
        path.insert(0, node)
        node = previous_nodes[node]
    if distances[destination] != float('inf'):
        path.insert(0, node)
    return distances[destination], path

# Floyd-Warshall Algorithm for All-Pairs Shortest Paths
def floyd_warshall(graph, weight='distance'):
    nodes = list(graph.keys())
    distances = {node: {neighbor: float('inf') for neighbor in nodes} for node in nodes}
    
    # Initialize distances with graph data
    for node in nodes:
        distances[node][node] = 0
        for neighbor, weights in graph[node].items():
            distances[node][neighbor] = weights[weight]
    
    # Compute shortest paths
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if distances[i][j] > distances[i][k] + distances[k][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    
    return distances

# Breadth-First Search (BFS)
def bfs(graph, start, destination):
    visited = set()
    queue = [(start, [start])]  # Node and path
    
    while queue:
        current_node, path = queue.pop(0)
        
        if current_node == destination:
            return path
        
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

# Depth-First Search (DFS)
def dfs(graph, start, destination, path=None, visited=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    
    path.append(start)
    visited.add(start)
    
    if start == destination:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            new_path = dfs(graph, neighbor, destination, path[:], visited)
            if new_path:
                return new_path
    return None

# Main Function to Use the Journey Planner
def journey_planner():
    g = Graph()
    
    # Example Data
    g.add_route("A", "B", 4, cost=10, time=5)
    g.add_route("A", "C", 2, cost=5, time=3)
    g.add_route("B", "C", 5, cost=7, time=4)
    g.add_route("B", "D", 10, cost=15, time=8)
    g.add_route("C", "D", 3, cost=6, time=2)
    
    print("Graph Representation:")
    g.display_graph()
    
    print("\n1. Dijkstra's Algorithm (Shortest Path)")
    start = input("Enter starting location: ")
    end = input("Enter destination: ")
    weight = input("Choose weight (distance/cost/time): ")
    shortest_distance, path = dijkstra(g.graph, start, end, weight)
    if path:
        print(f"Shortest {weight} from {start} to {end}: {shortest_distance}")
        print(f"Path: {' -> '.join(path)}")
    else:
        print("No path found!")
    
    print("\n2. All-Pairs Shortest Paths (Floyd-Warshall)")
    weight = input("Choose weight (distance/cost/time): ")
    all_pairs = floyd_warshall(g.graph, weight)
    print("Shortest paths between all locations:")
    for source, targets in all_pairs.items():
        print(f"{source}: {targets}")
    
    print("\n3. BFS (Find Path)")
    bfs_path = bfs(g.graph, start, end)
    if bfs_path:
        print(f"Path found using BFS: {' -> '.join(bfs_path)}")
    else:
        print("No path found using BFS!")
    
    print("\n4. DFS (Find Path)")
    dfs_path = dfs(g.graph, start, end)
    if dfs_path:
        print(f"Path found using DFS: {' -> '.join(dfs_path)}")
    else:
        print("No path found using DFS!")

# Run the Journey Planner
if _name_ == "_main_":
    journey_planner()
Convert this code into python
