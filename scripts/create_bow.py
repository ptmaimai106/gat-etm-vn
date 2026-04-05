"""
Create Bag-of-Words Data for GAT-ETM Model

Script để tạo BoW matrices và metadata cho training GAT-ETM model.

Input:
    - data-bv/thuoc.xlsx - Dữ liệu đơn thuốc VN
    - embed_vn/knowledge_graph.pkl - Knowledge graph (ICD-10 + ATC)
    - embed_vn/node2vec_embeddings.pkl - Node2Vec embeddings
    - data_vn/drug_atc_mapping.csv - Drug name → ATC mapping

Output:
    - data_vn/bow_train.npy - Training BoW matrix (sparse)
    - data_vn/bow_test.npy - Test BoW matrix (sparse)
    - data_vn/bow_test_1.npy - First half of test documents (sparse)
    - data_vn/bow_test_2.npy - Second half of test documents (sparse)
    - data_vn/metadata.txt - Metadata file for model
    - data_vn/vocab_info.pkl - Vocabulary information
    - embed_vn/icd_embeddings.npy - ICD embeddings (ordered by vocab)
    - embed_vn/atc_embeddings.npy - ATC embeddings (ordered by vocab)
    - embed_vn/graph_by_vocab.pkl - Graph with nodes renumbered by vocab

Usage:
    python scripts/create_bow.py [--full]
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
import argparse
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split
import networkx as nx


def normalize_icd_code(code):
    """Normalize ICD-10 code to standard format."""
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    # Remove special markers (*, +, !)
    code = re.sub(r'[*+!]', '', code)
    # Remove trailing dots
    code = code.rstrip('.')
    # Validate format: letter followed by digits, optionally with .digit(s)
    if re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', code):
        return code
    return None


def parse_icd_codes(maicd_text):
    """Parse MAICD column to extract list of ICD codes."""
    if pd.isna(maicd_text):
        return []

    # Split by semicolon
    parts = str(maicd_text).split(';')
    codes = []
    for part in parts:
        code = normalize_icd_code(part.strip())
        if code:
            codes.append(code)
    return list(set(codes))  # Return unique codes


def parse_drug_text(text, drug_atc_mapping):
    """
    Parse drug text and return list of ATC codes.
    Uses the drug_atc_mapping dictionary for lookup.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    # Pattern để split - kết thúc bằng (số) + đơn vị đóng gói
    unit_pattern = (
        r'\(\d+\)\s*(?:'
        r'Viên|viên|VIÊN|'
        r'Chai[^\(]*?|chai[^\(]*?|CHAI[^\(]*?|'
        r'Ống|ống|ONG|'
        r'Gói|gói|GOI|'
        r'Túi|túi|TUI|'
        r'Hộp|hộp|HOP|'
        r'Lọ[^\(]*?|lọ[^\(]*?|LO|'
        r'Tuýp|tuýp|TUYP|'
        r'Vỉ|vỉ|VI|'
        r'Miếng|miếng|MIENG|'
        r'Bánh|bánh|BANH|'
        r'Bình|bình|BINH|'
        r'Bộ|bộ|BO|'
        r'Cái|cái|CAI|'
        r'Kít|kít|KIT|'
        r'mL|ML|ml'
        r')'
    )

    # Split text theo unit pattern
    parts = re.split(unit_pattern, text, flags=re.IGNORECASE)

    atc_codes = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 2:
            continue
        if re.match(r'^\d', part):
            continue

        # Normalize drug name
        drug_name = normalize_drug_name(part)

        # Lookup ATC code
        if drug_name in drug_atc_mapping:
            atc_code = drug_atc_mapping[drug_name]
            if atc_code and pd.notna(atc_code):
                atc_codes.append(atc_code)

    return list(set(atc_codes))  # Return unique ATC codes


def normalize_drug_name(text):
    """Normalize drug name for lookup."""
    # Remove dosage
    name = re.sub(r'\s+\d+[\d\.,/\s]*(?:mg|ml|g|mcg|IU|UI|%|/\d+\w*).*$', '', text, flags=re.IGNORECASE)
    name = re.sub(r'\s*\((?:tương ứng|chứa|bao gồm)[^)]*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[;,]\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'\s*\+\s*$', '', name)
    name = re.sub(r'\s*\(\s*$', '', name)
    return name.strip().lower()


def load_drug_atc_mapping(filepath):
    """Load drug to ATC mapping from CSV."""
    df = pd.read_csv(filepath)
    mapping = {}
    for _, row in df.iterrows():
        drug_name = str(row['drug_name']).lower().strip()
        atc_code = row['atc_code']
        if pd.notna(atc_code) and atc_code != '':
            mapping[drug_name] = atc_code
    return mapping


def process_patient_data(df, drug_atc_mapping):
    """
    Process patient data to extract (ICD codes, ATC codes) per patient.
    Returns list of dicts: {'icd_codes': [...], 'atc_codes': [...]}
    """
    patients = []

    for idx, row in df.iterrows():
        icd_codes = parse_icd_codes(row.get('MAICD'))
        atc_codes = parse_drug_text(row.get('Thuoc'), drug_atc_mapping)

        # Only include patients with both ICD and ATC codes
        if icd_codes and atc_codes:
            patients.append({
                'icd_codes': icd_codes,
                'atc_codes': atc_codes
            })

    return patients


def create_vocabulary(patients, knowledge_graph):
    """
    Create vocabulary from patient data, filtering to codes in KG.

    Returns:
        icd_vocab: List of ICD codes in vocab
        atc_vocab: List of ATC codes in vocab
        icd2idx: Dict mapping ICD code to index
        atc2idx: Dict mapping ATC code to index
    """
    # Get all unique codes from patients
    all_icd_codes = set()
    all_atc_codes = set()

    for patient in patients:
        all_icd_codes.update(patient['icd_codes'])
        all_atc_codes.update(patient['atc_codes'])

    # Get nodes in knowledge graph
    kg_nodes = set(knowledge_graph.nodes())

    # Filter codes: must be in KG
    # For ICD codes, check both exact match and parent codes
    icd_vocab = []
    for code in sorted(all_icd_codes):
        if code in kg_nodes:
            icd_vocab.append(code)
        # Also check parent code (e.g., J44 for J44.0)
        elif '.' in code:
            parent = code.split('.')[0]
            if parent in kg_nodes:
                # Use parent code instead
                icd_vocab.append(parent)
    icd_vocab = sorted(list(set(icd_vocab)))

    # For ATC codes, check exact match (level 5) or parent levels
    atc_vocab = []
    for code in sorted(all_atc_codes):
        if code in kg_nodes:
            atc_vocab.append(code)
        else:
            # Try different ATC levels
            for length in [5, 4, 3]:
                parent = code[:length] if len(code) >= length else None
                if parent and parent in kg_nodes:
                    atc_vocab.append(parent)
                    break
    atc_vocab = sorted(list(set(atc_vocab)))

    # Create index mappings
    icd2idx = {code: idx for idx, code in enumerate(icd_vocab)}
    atc2idx = {code: idx for idx, code in enumerate(atc_vocab)}

    return icd_vocab, atc_vocab, icd2idx, atc2idx


def create_bow_matrix(patients, icd2idx, atc2idx):
    """
    Create sparse BoW matrix from patient data.

    Matrix shape: (N_patients, V_icd + V_atc)
    Columns 0 to V_icd-1: ICD codes
    Columns V_icd to V_total-1: ATC codes
    """
    N = len(patients)
    V_icd = len(icd2idx)
    V_atc = len(atc2idx)
    V = V_icd + V_atc

    # Use lil_matrix for efficient incremental construction
    bow = lil_matrix((N, V), dtype=np.float32)

    for p_idx, patient in enumerate(patients):
        # Add ICD codes
        for code in patient['icd_codes']:
            if code in icd2idx:
                bow[p_idx, icd2idx[code]] = 1
            # Also try parent code
            elif '.' in code:
                parent = code.split('.')[0]
                if parent in icd2idx:
                    bow[p_idx, icd2idx[parent]] = 1

        # Add ATC codes
        for code in patient['atc_codes']:
            if code in atc2idx:
                bow[p_idx, V_icd + atc2idx[code]] = 1
            else:
                # Try parent levels
                for length in [5, 4, 3]:
                    parent = code[:length] if len(code) >= length else None
                    if parent and parent in atc2idx:
                        bow[p_idx, V_icd + atc2idx[parent]] = 1
                        break

    # Convert to CSR for efficient storage and operations
    return bow.tocsr()


def split_document(bow_row, vocab_cum):
    """
    Split a document BoW vector into two halves.
    For document completion evaluation.

    vocab_cum: cumulative vocab sizes [0, V_icd, V_total]

    For each code type, randomly assign codes to first or second half.
    """
    indices = bow_row.indices
    data = bow_row.data

    indices_1 = []
    indices_2 = []
    data_1 = []
    data_2 = []

    for i, idx in enumerate(indices):
        # Randomly assign to first or second half
        if np.random.random() < 0.5:
            indices_1.append(idx)
            data_1.append(data[i])
        else:
            indices_2.append(idx)
            data_2.append(data[i])

    return (np.array(indices_1), np.array(data_1)), (np.array(indices_2), np.array(data_2))


def create_test_splits(bow_test, vocab_cum):
    """
    Create bow_test_1 and bow_test_2 by splitting each test document.
    """
    N = bow_test.shape[0]
    V = bow_test.shape[1]

    bow_1 = lil_matrix((N, V), dtype=np.float32)
    bow_2 = lil_matrix((N, V), dtype=np.float32)

    np.random.seed(42)  # For reproducibility

    for i in range(N):
        row = bow_test.getrow(i)
        (idx_1, data_1), (idx_2, data_2) = split_document(row, vocab_cum)

        for j, idx in enumerate(idx_1):
            bow_1[i, idx] = data_1[j]
        for j, idx in enumerate(idx_2):
            bow_2[i, idx] = data_2[j]

    return bow_1.tocsr(), bow_2.tocsr()


def create_embeddings_by_vocab(node2vec_embeddings, icd_vocab, atc_vocab, embedding_dim=256):
    """
    Create embedding arrays ordered by vocabulary.
    """
    V_icd = len(icd_vocab)
    V_atc = len(atc_vocab)

    # ICD embeddings
    icd_embeddings = np.zeros((V_icd, embedding_dim), dtype=np.float32)
    for idx, code in enumerate(icd_vocab):
        if code in node2vec_embeddings:
            icd_embeddings[idx] = node2vec_embeddings[code]
        else:
            # Try to find parent or similar code
            found = False
            if '.' in code:
                parent = code.split('.')[0]
                if parent in node2vec_embeddings:
                    icd_embeddings[idx] = node2vec_embeddings[parent]
                    found = True
            if not found:
                # Random initialization
                icd_embeddings[idx] = np.random.randn(embedding_dim) * 0.01

    # ATC embeddings
    atc_embeddings = np.zeros((V_atc, embedding_dim), dtype=np.float32)
    for idx, code in enumerate(atc_vocab):
        if code in node2vec_embeddings:
            atc_embeddings[idx] = node2vec_embeddings[code]
        else:
            # Try parent levels
            found = False
            for length in [5, 4, 3, 1]:
                parent = code[:length] if len(code) >= length else None
                if parent and parent in node2vec_embeddings:
                    atc_embeddings[idx] = node2vec_embeddings[parent]
                    found = True
                    break
            if not found:
                # Random initialization
                atc_embeddings[idx] = np.random.randn(embedding_dim) * 0.01

    return icd_embeddings, atc_embeddings


def create_graph_by_vocab(knowledge_graph, icd_vocab, atc_vocab):
    """
    Create a subgraph containing only vocab nodes, with indices renumbered.
    Ensures all vocab nodes are in graph and adds edges for connectivity.

    Returns:
        graph: NetworkX graph with nodes renumbered (0 to V-1)
        node_mapping: Dict mapping vocab code to new index
    """
    # Create node mapping
    node_mapping = {}
    for idx, code in enumerate(icd_vocab):
        node_mapping[code] = idx
    for idx, code in enumerate(atc_vocab):
        node_mapping[code] = len(icd_vocab) + idx

    # Create reverse mapping
    idx_to_code = {v: k for k, v in node_mapping.items()}

    # Create new graph
    G = nx.Graph()

    # Add ALL vocab nodes first (important!)
    total_vocab = len(icd_vocab) + len(atc_vocab)
    for idx in range(total_vocab):
        G.add_node(idx)

    # Add edges (only between vocab nodes)
    for u, v in knowledge_graph.edges():
        if u in node_mapping and v in node_mapping:
            G.add_edge(node_mapping[u], node_mapping[v])

    # Add hierarchy edges within ICD and ATC to improve connectivity
    # For ICD: connect subcodes to parent codes (e.g., J44.0 -> J44)
    for i, code in enumerate(icd_vocab):
        if '.' in code:
            parent = code.split('.')[0]
            if parent in node_mapping:
                G.add_edge(node_mapping[code], node_mapping[parent])

    # For ATC: connect lower levels to higher levels
    for i, code in enumerate(atc_vocab):
        for length in [5, 4, 3, 1]:
            if len(code) > length:
                parent = code[:length]
                if parent in node_mapping:
                    G.add_edge(node_mapping[code], node_mapping[parent])
                    break

    # Ensure graph is connected by adding edges between disconnected components
    # Connect each component to the largest one
    components = list(nx.connected_components(G))
    if len(components) > 1:
        # Find largest component
        largest = max(components, key=len)
        largest_node = list(largest)[0]

        # Connect other components
        for comp in components:
            if comp != largest:
                comp_node = list(comp)[0]
                G.add_edge(comp_node, largest_node)

    return G, node_mapping


def create_metadata_file(icd_vocab, atc_vocab, output_dir):
    """Create metadata.txt file for model."""
    content = f"icd atc\n"
    content += f"{len(icd_vocab)} {len(atc_vocab)}\n"
    content += "1 1\n"  # train_embeddings flag
    content += f"embed_vn/icd_embeddings.npy embed_vn/atc_embeddings.npy\n"

    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(content)

    print(f"Saved: {output_dir}/metadata.txt")


def main():
    parser = argparse.ArgumentParser(description='Create BoW data for GAT-ETM')
    parser.add_argument('--full', action='store_true',
                        help='Use full data from data-bv/thuoc.xlsx')
    parser.add_argument('--input', type=str, default=None,
                        help='Custom input file path')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test set ratio (default: 0.3)')
    parser.add_argument('--min_codes', type=int, default=2,
                        help='Minimum codes per patient (default: 2)')
    args = parser.parse_args()

    print("=" * 60)
    print("CREATE BOW DATA FOR GAT-ETM")
    print("=" * 60)
    print()

    # Paths
    if args.input:
        input_file = args.input
    elif args.full:
        input_file = '../data-bv/thuoc.xlsx'
    else:
        input_file = 'raw_vn_data/df_medicine_sample.xlsx'

    output_dir = 'data_vn'
    embed_dir = 'embed_vn'

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input: {input_file}")
    print(f"Output: {output_dir}/")
    print(f"Test size: {args.test_size}")
    print()

    # Step 1: Load data
    print("Step 1: Loading data...")

    # Load patient data
    print(f"  Loading patient data from {input_file}...")
    df = pd.read_excel(input_file)
    print(f"  Loaded {len(df)} records")

    # Load drug mapping
    print("  Loading drug-ATC mapping...")
    drug_atc_mapping = load_drug_atc_mapping(f'{output_dir}/drug_atc_mapping.csv')
    print(f"  Loaded {len(drug_atc_mapping)} drug mappings")

    # Load knowledge graph
    print("  Loading knowledge graph...")
    with open(f'{embed_dir}/knowledge_graph.pkl', 'rb') as f:
        knowledge_graph = pickle.load(f)
    print(f"  KG nodes: {knowledge_graph.number_of_nodes()}, edges: {knowledge_graph.number_of_edges()}")

    # Load node2vec embeddings
    print("  Loading Node2Vec embeddings...")
    with open(f'{embed_dir}/node2vec_embeddings.pkl', 'rb') as f:
        node2vec_embeddings = pickle.load(f)
    print(f"  Loaded {len(node2vec_embeddings)} embeddings")
    print()

    # Step 2: Process patient data
    print("Step 2: Processing patient data...")
    patients = process_patient_data(df, drug_atc_mapping)
    print(f"  Patients with both ICD and drugs: {len(patients)}")

    # Filter patients with minimum codes
    patients = [p for p in patients if len(p['icd_codes']) + len(p['atc_codes']) >= args.min_codes]
    print(f"  Patients after filtering (min {args.min_codes} codes): {len(patients)}")
    print()

    # Step 3: Create vocabulary
    print("Step 3: Creating vocabulary...")
    icd_vocab, atc_vocab, icd2idx, atc2idx = create_vocabulary(patients, knowledge_graph)
    print(f"  ICD vocabulary size: {len(icd_vocab)}")
    print(f"  ATC vocabulary size: {len(atc_vocab)}")
    print(f"  Total vocabulary: {len(icd_vocab) + len(atc_vocab)}")
    print()

    # Step 4: Create BoW matrix
    print("Step 4: Creating BoW matrix...")
    bow = create_bow_matrix(patients, icd2idx, atc2idx)
    print(f"  BoW shape: {bow.shape}")
    print(f"  Non-zero entries: {bow.nnz}")
    print(f"  Density: {100 * bow.nnz / (bow.shape[0] * bow.shape[1]):.4f}%")
    print()

    # Step 5: Train/test split
    print("Step 5: Splitting train/test...")
    indices = np.arange(bow.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)

    bow_train = bow[train_idx]
    bow_test = bow[test_idx]

    print(f"  Train size: {bow_train.shape[0]}")
    print(f"  Test size: {bow_test.shape[0]}")
    print()

    # Step 6: Create test splits for document completion
    print("Step 6: Creating test splits for document completion...")
    vocab_cum = [0, len(icd_vocab), len(icd_vocab) + len(atc_vocab)]
    bow_test_1, bow_test_2 = create_test_splits(bow_test, vocab_cum)
    print(f"  bow_test_1 non-zero: {bow_test_1.nnz}")
    print(f"  bow_test_2 non-zero: {bow_test_2.nnz}")
    print()

    # Step 7: Save BoW matrices
    print("Step 7: Saving BoW matrices...")
    np.save(f'{output_dir}/bow_train.npy', bow_train)
    np.save(f'{output_dir}/bow_test.npy', bow_test)
    np.save(f'{output_dir}/bow_test_1.npy', bow_test_1)
    np.save(f'{output_dir}/bow_test_2.npy', bow_test_2)
    print(f"  Saved: {output_dir}/bow_train.npy")
    print(f"  Saved: {output_dir}/bow_test.npy")
    print(f"  Saved: {output_dir}/bow_test_1.npy")
    print(f"  Saved: {output_dir}/bow_test_2.npy")
    print()

    # Step 8: Create embeddings by vocab
    print("Step 8: Creating embeddings ordered by vocab...")
    embedding_dim = 256
    icd_embeddings, atc_embeddings = create_embeddings_by_vocab(
        node2vec_embeddings, icd_vocab, atc_vocab, embedding_dim
    )
    np.save(f'{embed_dir}/icd_embeddings.npy', icd_embeddings)
    np.save(f'{embed_dir}/atc_embeddings.npy', atc_embeddings)
    print(f"  ICD embeddings shape: {icd_embeddings.shape}")
    print(f"  ATC embeddings shape: {atc_embeddings.shape}")
    print(f"  Saved: {embed_dir}/icd_embeddings.npy")
    print(f"  Saved: {embed_dir}/atc_embeddings.npy")
    print()

    # Step 9: Create graph by vocab
    print("Step 9: Creating graph with vocab indices...")
    graph_by_vocab, node_mapping = create_graph_by_vocab(knowledge_graph, icd_vocab, atc_vocab)
    with open(f'{embed_dir}/graph_by_vocab.pkl', 'wb') as f:
        pickle.dump(graph_by_vocab, f)
    print(f"  Graph nodes: {graph_by_vocab.number_of_nodes()}")
    print(f"  Graph edges: {graph_by_vocab.number_of_edges()}")
    print(f"  Saved: {embed_dir}/graph_by_vocab.pkl")
    print()

    # Step 10: Create combined embeddings
    print("Step 10: Creating combined embeddings...")
    combined_embeddings = np.vstack([icd_embeddings, atc_embeddings])
    with open(f'{embed_dir}/embeddings_by_vocab.pkl', 'wb') as f:
        pickle.dump(combined_embeddings, f)
    print(f"  Combined embeddings shape: {combined_embeddings.shape}")
    print(f"  Saved: {embed_dir}/embeddings_by_vocab.pkl")
    print()

    # Step 11: Save vocabulary info
    print("Step 11: Saving vocabulary info...")
    vocab_info = {
        'icd_vocab': icd_vocab,
        'atc_vocab': atc_vocab,
        'icd2idx': icd2idx,
        'atc2idx': atc2idx,
        'vocab_cum': vocab_cum,
        'node_mapping': node_mapping,
        'train_indices': train_idx,
        'test_indices': test_idx
    }
    with open(f'{output_dir}/vocab_info.pkl', 'wb') as f:
        pickle.dump(vocab_info, f)
    print(f"  Saved: {output_dir}/vocab_info.pkl")
    print()

    # Step 12: Create metadata file
    print("Step 12: Creating metadata file...")
    create_metadata_file(icd_vocab, atc_vocab, output_dir)
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total patients processed: {len(patients)}")
    print(f"Training samples: {bow_train.shape[0]}")
    print(f"Test samples: {bow_test.shape[0]}")
    print(f"ICD vocabulary: {len(icd_vocab)}")
    print(f"ATC vocabulary: {len(atc_vocab)}")
    print(f"Total vocabulary: {len(icd_vocab) + len(atc_vocab)}")
    print()
    print("Output files:")
    print(f"  {output_dir}/bow_train.npy")
    print(f"  {output_dir}/bow_test.npy")
    print(f"  {output_dir}/bow_test_1.npy")
    print(f"  {output_dir}/bow_test_2.npy")
    print(f"  {output_dir}/metadata.txt")
    print(f"  {output_dir}/vocab_info.pkl")
    print(f"  {embed_dir}/icd_embeddings.npy")
    print(f"  {embed_dir}/atc_embeddings.npy")
    print(f"  {embed_dir}/graph_by_vocab.pkl")
    print(f"  {embed_dir}/embeddings_by_vocab.pkl")
    print()
    print("DONE!")


if __name__ == '__main__':
    main()
