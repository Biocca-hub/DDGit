import pandas as pd 

data = pd.read_csv('../input/final_data.tsv', sep = '\t')
del data['Unnamed: 0']

complexes = data.Complex.to_list()
old_out = data.out_protein.to_list()

out = {}

for complex_, old in zip(data.Complex, data.out_protein):
    out.setdefault(complex_, []).extend(s for s in old.split(',') if s in set(complexes))

for c in set(complexes):
    out[c] = list(set(out[c]))

# Costruzione diretta degli archi
edges = set()

nodes_in_edges = []
for complex_a, neighbours in out.items():
    for complex_b in neighbours:
        # tuple ordinata per evitare sia A-B che B-A
        edges.add(tuple(sorted((complex_a, complex_b))))
        nodes_in_edges.extend([complex_a, complex_b])

nodes_in_edges = set(nodes_in_edges)
not_edge = set(complexes) - nodes_in_edges
for node in not_edge:
    edges.add(tuple(sorted((node,node))))

connected = pd.DataFrame(
    edges,
    columns=['complex_a', 'complex_b']
)
nodes_in_edges = set(connected['complex_a']) | set(connected['complex_b'])

print("Total complexes:", len(set(complexes)))
print("Nodes in graph:", len(nodes_in_edges))
print("Isolated complexes:", len(set(complexes) - nodes_in_edges))
print("Edges:", len(connected))

connected.to_csv(
    '../output/out_proteins_graph.tsv',
    sep='\t',
    index=False
)


