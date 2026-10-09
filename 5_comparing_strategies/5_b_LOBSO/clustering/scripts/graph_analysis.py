import pandas as pd 
import networkx as nx

#graph = pd.read_csv('../output/out_proteins_graph.tsv', sep = '\t')

graph = pd.read_csv('../output/out_prot_type_2_graph.tsv', sep = '\t')


G = nx.from_pandas_edgelist(graph,
                            source='complex_a',
                            target='complex_b')

print(G.number_of_nodes())
print(G.number_of_edges())

components = sorted(
    nx.connected_components(G),
    key=len,
    reverse=True
)

component_id = {}

for i, component in enumerate(components, start=1):
    for node in component:
        component_id[node] = f'{i}'
print(component_id)

tot = 0
for i, component in enumerate(components, start=1):
    print(f'{i}: {len(component)} nodes')
    tot += len(component)
print(tot)