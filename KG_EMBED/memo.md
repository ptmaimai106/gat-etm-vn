
### Learning medical code embedding from knowledge graph

We leverage an ICD–ATC knowledge graph to learn code embedding ρ^(icd), ρ^(atc). As shown in Fig. 1b, there are **3 types of relations** in this knowledge graph:

1. **ICD hierarchy**
   ([https://icdlist.com/icd-9/index](https://icdlist.com/icd-9/index))
   augmented by linking each pair of ancestral nodes and child nodes.

2. **ATC hierarchy**
   ([https://www.whocc.no/atc_ddd_index/](https://www.whocc.no/atc_ddd_index/))
   augmented by linking each pair of descendants and ancestors.

3. **ICD–ATC relations**
   ([http://hulab.rxnfinder.org/mia/](http://hulab.rxnfinder.org/mia/))

We extracted these relations from their corresponding websites and constructed an **undirected knowledge graph**
(\mathcal{G} = {\mathcal{V}, \mathcal{E}}), where:

* (\mathcal{V}) contains all of the ICD and ATC codes as the nodes, and
* (\mathcal{E}) contains ICD–ICD, ATC–ATC, and ICD–ATC relations as the edges.

The resulting knowledge graph is **sparsely connected** because of the tree-structure of both ICD and ATC taxonomy.
To further improve the information flow, we augmented the knowledge graph by **connecting each node to all of its ancestral nodes** (Fig. 1b).

