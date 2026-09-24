import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },
    "Main_Corridor": {
        "Nursing_Station": 2.2
    },
    "Patient_Wing": {
        "Laboratory": 5.0
    },
    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },
    "Laboratory": {
        "Emergency_Ward": 3.2
    },
    "Emergency_Ward": {}
}


# Heuristic

def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Path reconstruction

def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


# GBFS

def gbfs(start, goal):
    frontier = []
    counter = 0

    heapq.heappush(
        frontier,
        (heuristic(start, goal), counter, start)
    )

    came_from = {}
    visited = set()
    expansion_order = []

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = sum(
                hospital_graph[path[i]][path[i + 1]]
                for i in range(len(path) - 1)
            )
            return path, cost, expansion_order

        for neighbor in hospital_graph[current]:
            if neighbor not in visited:
                counter += 1
                came_from.setdefault(neighbor, current)
                heapq.heappush(
                    frontier,
                    (heuristic(neighbor, goal), counter, neighbor)
                )

    return None, float("inf"), expansion_order


# A*

def a_star(start, goal):
    frontier = []
    counter = 0

    g_cost = {start: 0.0}
    came_from = {}
    expansion_order = []
    expanded_best_g = {}

    heapq.heappush(
        frontier,
        (heuristic(start, goal), counter, start)
    )

    while frontier:
        _, _, current = heapq.heappop(frontier)
        current_g = g_cost[current]

        if current in expanded_best_g and current_g > expanded_best_g[current] + 1e-9:
            continue

        expanded_best_g[current] = current_g
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, current_g, expansion_order

        for neighbor, edge_cost in hospital_graph[current].items():
            tentative_g = current_g + edge_cost

            if tentative_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current

                counter += 1
                f_value = tentative_g + heuristic(neighbor, goal)

                heapq.heappush(
                    frontier,
                    (f_value, counter, neighbor)
                )

    return None, float("inf"), expansion_order


##########################################
# Streamlit GUI Code

st.set_page_config(
    page_title="Hospital Emergency Supply Robot",
    page_icon="🏥",
    layout="wide"
)

st.title("Emergency Supply Robot")
st.write(
    "Select an initial node, a goal node, and a search algorithm "
    "to visualize the resulting solution path."
)

# define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# create a selectbox for the user to choose the start and goal nodes
start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)

# create a selectbox for the user to choose the search algorithm
algorithm = st.selectbox(
    "Select Search Algorithm",
    ["GBFS", "A*"]
)


if st.button("Run Search"):

    if algorithm == "GBFS":
        path, cost, expansion_order = gbfs(start, goal)
    else:
        path, cost, expansion_order = a_star(start, goal)

    if path is None:
        st.error("No path was found between the selected nodes.")

    else:
        # Display result
        st.subheader("Search Result")

        st.write(f"Algorithm: **{algorithm}**")
        st.write(f"Solution Path: **{' → '.join(path)}**")
        st.write(f"Total Path Cost: **{cost:.2f}**")
        st.write(f"Expansion Order: **{' → '.join(expansion_order)}**")

        # Visualize NetworkX graph

        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():
            G.add_node(node, pos=locations[node])

            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = locations

        fig, ax = plt.subplots(figsize=(10, 6))

        # Draw the complete graph.
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="lightblue",
            node_size=1800,
            edgecolors="black"
        )

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=9,
            font_weight="bold"
        )

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="gray",
            width=1.5,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.03"
        )

        edge_labels = nx.get_edge_attributes(G, "weight")

        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=9
        )

        # Highlight the solution path.
        solution_edges = list(zip(path[:-1], path[1:]))

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=solution_edges,
            edge_color="red",
            width=4,
            arrows=True,
            arrowsize=22,
            connectionstyle="arc3,rad=0.03"
        )

        # Highlight start and goal.
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=[start],
            node_color="lightgreen",
            node_size=1900,
            edgecolors="black"
        )

        if goal != start:
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=[goal],
                node_color="gold",
                node_size=1900,
                edgecolors="black"
            )

        ax.set_title(f"{algorithm} Solution Path")
        ax.axis("off")

        st.pyplot(fig)
