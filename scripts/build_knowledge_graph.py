"""
Build Knowledge Graph for GAT-ETM with VN Data
===============================================

Script xây dựng Knowledge Graph kết hợp:
1. ICD-10 hierarchy (đã có từ Step 1)
2. ATC hierarchy (build từ WHO ATC-DDD)
3. Disease-Drug links (từ co-occurrence trong dữ liệu VN)

Output:
- embed_vn/atc_hierarchy.pkl - ATC hierarchy graph
- embed_vn/knowledge_graph.pkl - Merged ICD-10 + ATC + disease-drug links
- embed_vn/node2vec_embeddings.pkl - Node2Vec embeddings

Author: Claude Code
Date: 2026-03-30
"""

import pickle
import os
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# PART 1: BUILD ATC HIERARCHY GRAPH
# ============================================

def build_atc_graph(atc_csv_path):
    """
    Build ATC hierarchy graph từ WHO ATC-DDD CSV.

    ATC có 5 levels:
    - Level 1: Anatomical main group (1 letter, e.g., A)
    - Level 2: Therapeutic subgroup (3 chars, e.g., A01)
    - Level 3: Pharmacological subgroup (4 chars, e.g., A01A)
    - Level 4: Chemical subgroup (5 chars, e.g., A01AA)
    - Level 5: Chemical substance (7 chars, e.g., A01AA01)

    Args:
        atc_csv_path: Path to WHO ATC-DDD CSV file

    Returns:
        nx.DiGraph: ATC hierarchy graph
    """
    print("\n" + "=" * 50)
    print("Building ATC Hierarchy Graph")
    print("=" * 50)

    # Load ATC data
    atc_df = pd.read_csv(atc_csv_path)
    print(f"Loaded {len(atc_df)} ATC codes")

    G = nx.DiGraph()

    # Add all nodes with attributes
    for _, row in atc_df.iterrows():
        code = row['atc_code']
        name = row['atc_name']

        # Determine level based on code length
        code_len = len(code)
        if code_len == 1:
            level = 1
            node_type = 'anatomical'
        elif code_len == 3:
            level = 2
            node_type = 'therapeutic'
        elif code_len == 4:
            level = 3
            node_type = 'pharmacological'
        elif code_len == 5:
            level = 4
            node_type = 'chemical_subgroup'
        else:  # 7 characters
            level = 5
            node_type = 'substance'

        G.add_node(
            code,
            name=name,
            level=level,
            type=node_type,
            code_type='ATC',
            ddd=row.get('ddd', None),
            uom=row.get('uom', None),
            adm_r=row.get('adm_r', None)
        )

    print(f"Added {G.number_of_nodes()} nodes")

    # Build hierarchy edges based on code prefixes
    print("Building hierarchy edges...")
    edge_count = 0

    all_codes = set(G.nodes())

    for code in all_codes:
        # Find parent based on code prefix
        parent = None
        code_len = len(code)

        if code_len == 7:  # Level 5 -> Level 4
            parent = code[:5]
        elif code_len == 5:  # Level 4 -> Level 3
            parent = code[:4]
        elif code_len == 4:  # Level 3 -> Level 2
            parent = code[:3]
        elif code_len == 3:  # Level 2 -> Level 1
            parent = code[0]

        if parent and parent in all_codes:
            G.add_edge(parent, code, relation='hierarchy')
            edge_count += 1

    print(f"Added {edge_count} hierarchy edges")

    # Graph augmentation: connect to all ancestors
    print("Performing graph augmentation...")
    augmented_count = 0

    for code in all_codes:
        # Get all ancestors
        ancestors = []
        current = code

        while True:
            code_len = len(current)
            if code_len == 7:
                parent = current[:5]
            elif code_len == 5:
                parent = current[:4]
            elif code_len == 4:
                parent = current[:3]
            elif code_len == 3:
                parent = current[0]
            else:
                break

            if parent in all_codes:
                ancestors.append(parent)
                current = parent
            else:
                break

        # Add augmented edges
        for ancestor in ancestors:
            if not G.has_edge(ancestor, code):
                G.add_edge(ancestor, code, relation='augmented')
                augmented_count += 1

    print(f"Added {augmented_count} augmented edges")

    return G


def print_atc_statistics(G):
    """Print ATC graph statistics."""
    print("\nATC Graph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Count by level
    level_counts = defaultdict(int)
    for node, attrs in G.nodes(data=True):
        level_counts[attrs.get('level', 0)] += 1

    print("\n  Nodes by level:")
    for level in sorted(level_counts.keys()):
        print(f"    Level {level}: {level_counts[level]}")

    # Sample path
    print("\n  Sample hierarchy path:")
    sample_leaf = None
    for node in G.nodes():
        if len(node) == 7:  # Level 5
            sample_leaf = node
            break

    if sample_leaf:
        path = [sample_leaf]
        current = sample_leaf
        while True:
            parents = [p for p in G.predecessors(current)
                      if G.edges[p, current].get('relation') == 'hierarchy']
            if not parents:
                break
            current = parents[0]
            path.append(current)

        print(f"    {' -> '.join(reversed(path))}")
        for code in reversed(path):
            name = G.nodes[code].get('name', '')[:40]
            print(f"      {code}: {name}")


# ============================================
# PART 2: EXTRACT DISEASE-DRUG LINKS FROM VN DATA
# ============================================

def extract_disease_drug_links(medicine_data_path, drug_mapping_path, min_cooccurrence=5):
    """
    Extract disease-drug co-occurrence links từ dữ liệu VN.

    Args:
        medicine_data_path: Path to thuoc.xlsx
        drug_mapping_path: Path to drug_atc_mapping.csv
        min_cooccurrence: Minimum co-occurrence count to include link

    Returns:
        list: List of (icd_code, atc_code, count) tuples
    """
    print("\n" + "=" * 50)
    print("Extracting Disease-Drug Links from VN Data")
    print("=" * 50)

    # Load drug mapping
    drug_mapping = pd.read_csv(drug_mapping_path)
    drug_to_atc = {}
    for _, row in drug_mapping.iterrows():
        if pd.notna(row['atc_code']) and row['atc_code'] != '':
            drug_name = str(row['drug_name']).lower()
            drug_to_atc[drug_name] = row['atc_code']

    print(f"Loaded {len(drug_to_atc)} drug-to-ATC mappings")

    # Load medicine data
    print(f"Loading medicine data from {medicine_data_path}...")
    df = pd.read_excel(medicine_data_path)
    print(f"Loaded {len(df)} records")

    # Import parse function from parse_drug_names
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from parse_drug_names import parse_drug_text, normalize_drug_name

    # Count ICD-ATC co-occurrences
    cooccurrence = defaultdict(int)
    records_processed = 0
    links_found = 0

    for idx, row in df.iterrows():
        if pd.isna(row.get('MAICD')) or pd.isna(row.get('Thuoc')):
            continue

        # Parse ICD codes (may be multiple, separated by ';')
        icd_str = str(row['MAICD'])
        icd_codes = [c.strip().upper() for c in icd_str.split(';') if c.strip()]

        # Normalize ICD codes (add dot if needed)
        normalized_icds = []
        for icd in icd_codes:
            if len(icd) > 3 and '.' not in icd:
                icd = icd[:3] + '.' + icd[3:]
            normalized_icds.append(icd)

        # Parse drug names
        drugs = parse_drug_text(row['Thuoc'])

        # Find ATC codes for drugs
        atc_codes = []
        for drug in drugs:
            drug_lower = normalize_drug_name(drug)
            if drug_lower in drug_to_atc:
                atc_codes.append(drug_to_atc[drug_lower])
            else:
                # Try partial matching
                for key, atc in drug_to_atc.items():
                    if key in drug_lower or drug_lower in key:
                        atc_codes.append(atc)
                        break

        # Count co-occurrences
        for icd in normalized_icds:
            for atc in set(atc_codes):  # Use set to avoid duplicates
                cooccurrence[(icd, atc)] += 1
                links_found += 1

        records_processed += 1
        if records_processed % 10000 == 0:
            print(f"  Processed {records_processed} records...")

    print(f"Processed {records_processed} records")
    print(f"Found {len(cooccurrence)} unique ICD-ATC pairs")

    # Filter by minimum co-occurrence
    filtered_links = [
        (icd, atc, count)
        for (icd, atc), count in cooccurrence.items()
        if count >= min_cooccurrence
    ]

    print(f"Links with >= {min_cooccurrence} co-occurrences: {len(filtered_links)}")

    # Print top links
    print("\nTop 20 disease-drug links:")
    sorted_links = sorted(filtered_links, key=lambda x: x[2], reverse=True)
    for icd, atc, count in sorted_links[:20]:
        print(f"  {icd} - {atc}: {count}")

    return filtered_links


# ============================================
# PART 3: MERGE GRAPHS
# ============================================

def merge_knowledge_graph(icd_graph, atc_graph, disease_drug_links):
    """
    Merge ICD-10 and ATC graphs, add disease-drug links.

    Args:
        icd_graph: ICD-10 hierarchy graph
        atc_graph: ATC hierarchy graph
        disease_drug_links: List of (icd, atc, count) tuples

    Returns:
        nx.Graph: Merged undirected knowledge graph
    """
    print("\n" + "=" * 50)
    print("Merging Knowledge Graph")
    print("=" * 50)

    # Start with ICD graph
    print(f"ICD-10 graph: {icd_graph.number_of_nodes()} nodes, {icd_graph.number_of_edges()} edges")
    print(f"ATC graph: {atc_graph.number_of_nodes()} nodes, {atc_graph.number_of_edges()} edges")

    # Compose graphs (combine nodes and edges)
    G = nx.compose(icd_graph, atc_graph)
    print(f"After compose: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Add disease-drug links
    print(f"\nAdding {len(disease_drug_links)} disease-drug links...")
    links_added = 0
    links_skipped = 0

    for icd, atc, count in disease_drug_links:
        # Check if both nodes exist
        icd_exists = icd in G
        atc_exists = atc in G

        # For ATC codes that are partial (like A12AA), try to find in graph
        if not atc_exists:
            # Try adding as a new node if it's a valid ATC prefix
            if len(atc) >= 1 and len(atc) <= 7:
                G.add_node(atc, code_type='ATC', level=len(atc), type='mapped')
                atc_exists = True

        if icd_exists and atc_exists:
            G.add_edge(icd, atc, relation='disease_drug', weight=count)
            links_added += 1
        else:
            links_skipped += 1

    print(f"Links added: {links_added}")
    print(f"Links skipped (missing nodes): {links_skipped}")

    # Convert to undirected graph (như paper)
    print("\nConverting to undirected graph...")
    G_undirected = G.to_undirected()

    print(f"\nFinal Knowledge Graph:")
    print(f"  Nodes: {G_undirected.number_of_nodes()}")
    print(f"  Edges: {G_undirected.number_of_edges()}")

    # Count edge types
    edge_types = defaultdict(int)
    for u, v, data in G_undirected.edges(data=True):
        rel = data.get('relation', 'unknown')
        edge_types[rel] += 1

    print("\n  Edges by type:")
    for rel, count in sorted(edge_types.items()):
        print(f"    {rel}: {count}")

    # Count node types
    node_types = defaultdict(int)
    for node, data in G_undirected.nodes(data=True):
        code_type = data.get('code_type', 'unknown')
        node_types[code_type] += 1

    print("\n  Nodes by code type:")
    for t, count in sorted(node_types.items()):
        print(f"    {t}: {count}")

    return G_undirected


# ============================================
# PART 4: GENERATE NODE2VEC EMBEDDINGS
# ============================================

def generate_node2vec_embeddings(G, dimensions=256, walk_length=20, num_walks=10, window=8):
    """
    Generate Node2Vec embeddings cho knowledge graph.

    Args:
        G: Knowledge graph (undirected)
        dimensions: Embedding dimension (paper: 256)
        walk_length: Length of random walks (paper: 20)
        num_walks: Number of walks per node (paper: 10)
        window: Context window size (paper: 8)

    Returns:
        dict: {node: embedding_vector}
    """
    print("\n" + "=" * 50)
    print("Generating Node2Vec Embeddings")
    print("=" * 50)

    try:
        from node2vec import Node2Vec
    except ImportError:
        print("ERROR: node2vec package not installed.")
        print("Install with: pip install node2vec")
        return None

    print(f"Parameters:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Walk length: {walk_length}")
    print(f"  Num walks: {num_walks}")
    print(f"  Window: {window}")

    # Initialize Node2Vec
    print("\nInitializing Node2Vec (this may take a while)...")
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=4,
        quiet=False
    )

    # Fit model
    print("Training Word2Vec model...")
    model = node2vec.fit(window=window, min_count=1, batch_words=4)

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            # Node not in vocabulary (rare)
            embeddings[node] = np.zeros(dimensions)

    print(f"Generated embeddings for {len(embeddings)} nodes")
    print(f"Embedding shape: {embeddings[list(embeddings.keys())[0]].shape}")

    return embeddings


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to build complete knowledge graph."""
    print("=" * 60)
    print("BUILD KNOWLEDGE GRAPH FOR GAT-ETM")
    print("=" * 60)

    # Paths
    embed_dir = PROJECT_ROOT / 'embed_vn'
    embed_dir.mkdir(exist_ok=True)

    icd_graph_path = embed_dir / 'icd10_hierarchy.pkl'
    atc_csv_path = PROJECT_ROOT / 'data_vn' / 'atc_reference' / 'who_atc_ddd.csv'
    drug_mapping_path = PROJECT_ROOT / 'data_vn' / 'drug_atc_mapping.csv'
    medicine_data_path = Path('/Users/m001938/Documents/CS_UIT/LuanVan/data-bv/thuoc.xlsx')

    # Output paths
    atc_graph_path = embed_dir / 'atc_hierarchy.pkl'
    kg_path = embed_dir / 'knowledge_graph.pkl'
    embeddings_path = embed_dir / 'node2vec_embeddings.pkl'

    # ========================================
    # Step 1: Load ICD-10 graph
    # ========================================
    print("\n[Step 1] Loading ICD-10 hierarchy graph...")
    with open(icd_graph_path, 'rb') as f:
        icd_graph = pickle.load(f)
    print(f"Loaded ICD-10 graph: {icd_graph.number_of_nodes()} nodes")

    # ========================================
    # Step 2: Build ATC hierarchy graph
    # ========================================
    print("\n[Step 2] Building ATC hierarchy graph...")
    atc_graph = build_atc_graph(atc_csv_path)
    print_atc_statistics(atc_graph)

    # Save ATC graph
    with open(atc_graph_path, 'wb') as f:
        pickle.dump(atc_graph, f)
    print(f"Saved: {atc_graph_path}")

    # ========================================
    # Step 3: Extract disease-drug links
    # ========================================
    print("\n[Step 3] Extracting disease-drug links...")
    disease_drug_links = extract_disease_drug_links(
        medicine_data_path,
        drug_mapping_path,
        min_cooccurrence=3  # Lower threshold to get more links
    )

    # ========================================
    # Step 4: Merge graphs
    # ========================================
    print("\n[Step 4] Merging knowledge graph...")
    knowledge_graph = merge_knowledge_graph(icd_graph, atc_graph, disease_drug_links)

    # Save knowledge graph
    with open(kg_path, 'wb') as f:
        pickle.dump(knowledge_graph, f)
    print(f"Saved: {kg_path}")

    # ========================================
    # Step 5: Generate Node2Vec embeddings
    # ========================================
    print("\n[Step 5] Generating Node2Vec embeddings...")
    embeddings = generate_node2vec_embeddings(
        knowledge_graph,
        dimensions=256,
        walk_length=20,
        num_walks=10,
        window=8
    )

    if embeddings:
        # Save embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Saved: {embeddings_path}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ICD-10 hierarchy: {icd_graph.number_of_nodes()} nodes")
    print(f"ATC hierarchy: {atc_graph.number_of_nodes()} nodes")
    print(f"Disease-drug links: {len(disease_drug_links)}")
    print(f"Knowledge graph: {knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges")
    if embeddings:
        print(f"Embeddings: {len(embeddings)} nodes x 256 dimensions")

    print("\nOutput files:")
    print(f"  - {atc_graph_path}")
    print(f"  - {kg_path}")
    if embeddings:
        print(f"  - {embeddings_path}")

    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
