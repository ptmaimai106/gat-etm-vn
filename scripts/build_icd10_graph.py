"""
Build ICD-10 Hierarchy Graph for GAT-ETM
========================================

Script này xây dựng ICD-10 knowledge graph từ WHO ICD-10 classification.
Sử dụng package simple-icd-10 để lấy hierarchy.

Output: embed_vn/icd10_hierarchy.pkl (NetworkX DiGraph)

Author: Claude Code
Date: 2026-03-25
"""

import pickle
import sys
from pathlib import Path

import networkx as nx
import simple_icd_10 as icd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_icd10_statistics():
    """Print statistics about ICD-10 codes."""
    all_codes = icd.get_all_codes(with_dots=True)

    # Count by type
    chapters = []
    blocks = []
    categories = []
    subcategories = []

    for code in all_codes:
        if icd.is_chapter(code):
            chapters.append(code)
        elif icd.is_block(code):
            blocks.append(code)
        elif icd.is_category(code):
            categories.append(code)
        else:
            subcategories.append(code)

    print("=" * 50)
    print("ICD-10 WHO Statistics")
    print("=" * 50)
    print(f"Total codes: {len(all_codes)}")
    print(f"  - Chapters: {len(chapters)}")
    print(f"  - Blocks: {len(blocks)}")
    print(f"  - Categories (3-char): {len(categories)}")
    print(f"  - Subcategories (4+ char): {len(subcategories)}")
    print()

    # Sample chapters
    print("Sample chapters:")
    for ch in chapters[:5]:
        desc = icd.get_description(ch)
        print(f"  {ch}: {desc[:60]}...")

    return all_codes


def build_icd10_graph(include_augmentation=True):
    """
    Build ICD-10 hierarchy as NetworkX DiGraph.

    Structure:
    - Nodes: ICD-10 codes (chapters, blocks, categories, subcategories)
    - Edges: Parent -> Child relationships
    - Node attributes: type, level, description

    Graph augmentation (như paper):
    - Connect each node to ALL ancestors (not just direct parent)

    Args:
        include_augmentation: If True, add edges to all ancestors (paper method)

    Returns:
        nx.DiGraph: ICD-10 hierarchy graph
    """
    print("\nBuilding ICD-10 hierarchy graph...")

    G = nx.DiGraph()
    all_codes = icd.get_all_codes(with_dots=True)

    # Add all nodes with attributes
    print(f"Adding {len(all_codes)} nodes...")
    for code in all_codes:
        # Determine node type and level
        if icd.is_chapter(code):
            node_type = 'chapter'
            level = 0
        elif icd.is_block(code):
            node_type = 'block'
            level = 1
        elif icd.is_category(code):
            node_type = 'category'
            level = 2
        else:
            node_type = 'subcategory'
            # Level depends on specificity (A00.0 = 3, A00.01 = 4, etc.)
            level = 2 + len(code.split('.')[-1]) if '.' in code else 3

        # Get description
        try:
            description = icd.get_description(code)
        except Exception:
            description = ""

        G.add_node(
            code,
            type=node_type,
            level=level,
            description=description,
            code_type='ICD10'  # For compatibility with merged graph
        )

    # Add edges (parent -> child)
    print("Adding hierarchy edges...")
    edge_count = 0
    for code in all_codes:
        try:
            children = icd.get_children(code)
            for child in children:
                G.add_edge(code, child, relation='hierarchy')
                edge_count += 1
        except Exception:
            pass

    print(f"Added {edge_count} hierarchy edges")

    # Graph augmentation: connect to all ancestors
    if include_augmentation:
        print("Performing graph augmentation (connecting to all ancestors)...")
        augmented_edges = 0
        for code in all_codes:
            try:
                ancestors = icd.get_ancestors(code)
                for ancestor in ancestors:
                    if not G.has_edge(ancestor, code):
                        G.add_edge(ancestor, code, relation='augmented')
                        augmented_edges += 1
            except Exception:
                pass
        print(f"Added {augmented_edges} augmented edges")

    return G


def validate_with_vn_codes(G, sample_vn_codes):
    """
    Check how many VN ICD codes exist in the graph.

    Args:
        G: ICD-10 graph
        sample_vn_codes: List of ICD codes from VN data

    Returns:
        dict: Validation statistics
    """
    print("\n" + "=" * 50)
    print("Validating with VN ICD codes")
    print("=" * 50)

    found = []
    not_found = []

    for code in sample_vn_codes:
        # Normalize code (ensure has dot)
        normalized = code.strip().upper()
        if len(normalized) > 3 and '.' not in normalized:
            normalized = normalized[:3] + '.' + normalized[3:]

        if normalized in G:
            found.append(normalized)
        else:
            # Try without dot
            no_dot = normalized.replace('.', '')
            if no_dot in G:
                found.append(no_dot)
            else:
                not_found.append(code)

    print(f"Found in graph: {len(found)}/{len(sample_vn_codes)} ({100*len(found)/len(sample_vn_codes):.1f}%)")

    if not_found:
        print(f"\nNot found ({len(not_found)} codes):")
        for code in not_found[:10]:
            print(f"  - {code}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    return {
        'found': found,
        'not_found': not_found,
        'coverage': len(found) / len(sample_vn_codes) if sample_vn_codes else 0
    }


def print_graph_statistics(G):
    """Print graph statistics."""
    print("\n" + "=" * 50)
    print("Graph Statistics")
    print("=" * 50)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Count by type
    type_counts = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print("\nNodes by type:")
    for t, count in sorted(type_counts.items()):
        print(f"  - {t}: {count}")

    # Edge types
    edge_types = {}
    for u, v, attrs in G.edges(data=True):
        rel = attrs.get('relation', 'unknown')
        edge_types[rel] = edge_types.get(rel, 0) + 1

    print("\nEdges by relation:")
    for t, count in sorted(edge_types.items()):
        print(f"  - {t}: {count}")

    # Sample paths
    print("\nSample hierarchy path:")
    sample_leaf = None
    for node in G.nodes():
        if G.out_degree(node) == 0:  # Leaf node
            sample_leaf = node
            break

    if sample_leaf:
        # Get path to root
        path = [sample_leaf]
        current = sample_leaf
        while True:
            parents = list(G.predecessors(current))
            hierarchy_parents = [p for p in parents
                               if G.edges[p, current].get('relation') == 'hierarchy']
            if not hierarchy_parents:
                break
            current = hierarchy_parents[0]
            path.append(current)

        print(f"  {' -> '.join(reversed(path))}")
        for code in reversed(path):
            desc = G.nodes[code].get('description', '')[:50]
            print(f"    {code}: {desc}...")


def save_graph(G, output_path):
    """Save graph to pickle file."""
    print(f"\nSaving graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print("Done!")


def main():
    """Main function."""
    # Output path
    output_dir = PROJECT_ROOT / 'embed_vn'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'icd10_hierarchy.pkl'

    # Get statistics
    get_icd10_statistics()

    # Build graph with augmentation
    G = build_icd10_graph(include_augmentation=True)

    # Print statistics
    print_graph_statistics(G)

    # Sample VN codes from plan (top 10 từ dữ liệu thực)
    sample_vn_codes = [
        'I10',      # Tăng huyết áp vô căn
        'E78.2',    # Tăng lipid máu hỗn hợp
        'E11.9',    # ĐTĐ type 2
        'K21',      # Trào ngược dạ dày-thực quản
        'I25.0',    # Bệnh tim thiếu máu xơ vữa
        'J20',      # Viêm phế quản cấp
        'R10.0',    # Đau bụng cấp
        'K76.0',    # Gan nhiễm mỡ
        'I25',      # Bệnh tim thiếu máu cục bộ mạn
        'M10.0',    # Gout vô căn
    ]

    validate_with_vn_codes(G, sample_vn_codes)

    # Save graph
    save_graph(G, output_path)

    print("\n" + "=" * 50)
    print("COMPLETED!")
    print("=" * 50)
    print(f"Output: {output_path}")

    return G


if __name__ == '__main__':
    main()
