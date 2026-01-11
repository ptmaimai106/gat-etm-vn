#!/usr/bin/env python3
"""
Build Simple Knowledge Graph based on GAT-ETM Paper (for Manual Verification)

This is a simplified version that only uses a few ICD and ATC codes
to allow manual verification of connections.

Key features:
- Only 1-3 ICD codes (selected manually or top frequent)
- Only 5-10 ATC codes (selected manually or top frequent)
- Full hierarchy structure for these codes
- ICD-ATC relations from mapping file or co-occurrence
- Augmentation: skip connections to all ancestors

Use case: Manual verification of graph structure before building full KG
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Try to import node2vec, install if not available
try:
    from node2vec import Node2Vec
except ImportError:
    print("Warning: node2vec not installed. Install with: pip install node2vec")
    print("Will use random embeddings as fallback.")
    Node2Vec = None


class Simple_Paper_KG_Builder:
    def __init__(self, 
                 mimic_path='mimic-iii-clinical-database-demo-1.4',
                 output_dir='embed_paper_simple',
                 embedding_dim=256,
                 window=8,
                 walk_length=20,
                 num_walks=10,
                 icd_atc_mapping_file=None,
                 num_icd=1,
                 num_atc=5):
        """
        Initialize Simple KG Builder
        
        Args:
            mimic_path: Path to MIMIC-III data directory
            output_dir: Directory to save output files
            embedding_dim: Dimension of node embeddings
            window: Node2Vec window size
            walk_length: Node2Vec walk length
            num_walks: Node2Vec number of walks
            icd_atc_mapping_file: Path to ICD-ATC mapping file (CSV format)
            num_icd: Number of ICD codes to use (default: 1)
            num_atc: Number of ATC codes to use (default: 5)
        """
        self.mimic_path = mimic_path
        self.output_dir = output_dir
        self.embedding_dim = embedding_dim
        self.window = window
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.icd_atc_mapping_file = icd_atc_mapping_file
        self.num_icd = num_icd
        self.num_atc = num_atc

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize graph (undirected as per paper)
        self.G = nx.Graph()

        # Code vocabularies (only ICD and ATC)
        self.vocab_icd = []
        self.vocab_atc = []

        # ICD-ATC mapping
        self.icd_atc_relations = []

    def load_icd_codes_from_mimic(self):
        """
        Load a small number of ICD-9 codes from MIMIC-III
        Select top N most frequent codes
        """
        print("\n" + "=" * 80)
        print("Step 1: Loading ICD-9 codes from MIMIC-III (Simple Mode)")
        print("=" * 80)
        
        diagnoses_file = os.path.join(self.mimic_path, 'DIAGNOSES_ICD.csv')
        if not os.path.exists(diagnoses_file):
            raise FileNotFoundError(f"DIAGNOSES_ICD.csv not found at {diagnoses_file}")
        
        diagnoses_df = pd.read_csv(diagnoses_file)
        
        # Find ICD code column (handle different column name variations)
        icd_column = None
        for col in ['ICD9_CODE', 'icd9_code', 'ICD_CODE', 'icd_code', 'ICD9', 'icd9']:
            if col in diagnoses_df.columns:
                icd_column = col
                break
        
        if icd_column is None:
            print(f"  ERROR: Cannot find ICD code column. Available columns:")
            print(f"    {list(diagnoses_df.columns)}")
            raise ValueError("Cannot find ICD code column in DIAGNOSES_ICD.csv")
        
        print(f"  Using column: {icd_column}")
        
        # Get ICD code frequencies
        icd_counts = diagnoses_df[icd_column].value_counts()
        
        # Select top N most frequent codes
        top_icd = icd_counts.head(self.num_icd).index.tolist()
        self.vocab_icd = sorted([str(code).strip() for code in top_icd if pd.notna(code)])
        
        print(f"Selected {len(self.vocab_icd)} ICD-9 codes (top {self.num_icd} most frequent):")
        for i, icd in enumerate(self.vocab_icd, 1):
            count = icd_counts[icd] if icd in icd_counts.index else 0
            print(f"  {i}. {icd} (appears {count} times)")
        
        return self.vocab_icd

    def build_icd_hierarchy(self):
        """
        Build ICD-9 hierarchy for selected codes
        """
        print("\n" + "=" * 80)
        print("Step 2: Building ICD-9 Hierarchy")
        print("=" * 80)
        
        # Add root node
        self.G.add_node('ICD9_ROOT', type='ICD9', level=0)
        
        icd_nodes = set()
        icd_nodes.add('ICD9_ROOT')
        
        # Helper to normalize ICD code
        def normalize_icd(icd_code):
            if pd.isna(icd_code):
                return None
            code_str = str(icd_code).strip().replace('.', '')
            return code_str if code_str else None
        
        # Build hierarchy for each ICD code
        for icd_code in tqdm(self.vocab_icd, desc="  Processing ICD codes"):
            icd_normalized = normalize_icd(icd_code)
            if not icd_normalized:
                continue
            
            # Create prefix hierarchy: 3-digit → 4-digit → 5-digit
            prefixes = []
            if len(icd_normalized) >= 3:
                prefixes.append(icd_normalized[:3])  # 3-digit
            if len(icd_normalized) >= 4:
                prefixes.append(icd_normalized[:4])  # 4-digit
            if len(icd_normalized) >= 5:
                prefixes.append(icd_normalized[:5])  # 5-digit
            
            # Add all prefix nodes and leaf node
            all_nodes = ['ICD9_ROOT'] + prefixes + [icd_normalized]
            
            # Add nodes to graph
            for node in all_nodes:
                if node not in icd_nodes:
                    level = 0 if node == 'ICD9_ROOT' else len(node)
                    self.G.add_node(node, type='ICD9', level=level, 
                                   code=icd_code if node == icd_normalized else None)
                    icd_nodes.add(node)
            
            # Create hierarchical edges: parent → child
            for i in range(len(all_nodes) - 1):
                parent = all_nodes[i]
                child = all_nodes[i + 1]
                if not self.G.has_edge(parent, child):
                    self.G.add_edge(parent, child, edge_type='hierarchical')
        
        print(f"  ICD-9 hierarchy: {len(icd_nodes)} nodes")
        print(f"  Nodes: {sorted(icd_nodes)}")
        hierarchical_edges = [e for e in self.G.edges(data=True) if e[2].get('edge_type') == 'hierarchical']
        print(f"  Hierarchical edges: {len(hierarchical_edges)}")
        
        return icd_nodes

    def load_atc_codes(self):
        """
        Load a small number of ATC codes from MIMIC-III
        Select top N most frequent drugs
        """
        print("\n" + "=" * 80)
        print("Step 3: Loading ATC Codes (Simple Mode)")
        print("=" * 80)
        
        prescriptions_file = os.path.join(self.mimic_path, 'PRESCRIPTIONS.csv')
        if os.path.exists(prescriptions_file):
            print("  Loading from PRESCRIPTIONS.csv...")
            prescriptions_df = pd.read_csv(prescriptions_file, nrows=50000)  # Sample for speed
            
            # Find drug column
            drug_column = None
            for col in ['drug_name_generic', 'drug', 'drug_name', 'DRUG']:
                if col in prescriptions_df.columns:
                    drug_column = col
                    break
            
            if drug_column:
                # Get drug frequencies
                drug_counts = prescriptions_df[drug_column].dropna().value_counts()
                
                # Select top N most frequent drugs
                top_drugs = drug_counts.head(self.num_atc).index.tolist()
                
                # Create ATC codes from drug names (hash-based placeholder)
                self.vocab_atc = []
                for drug in top_drugs:
                    if pd.notna(drug):
                        drug_str = str(drug).strip().upper()
                        if drug_str:
                            atc_code = f"DRUG_{abs(hash(drug_str)) % 100000:05d}"
                            if atc_code not in self.vocab_atc:
                                self.vocab_atc.append(atc_code)
                
                print(f"Selected {len(self.vocab_atc)} ATC codes (top {self.num_atc} most frequent drugs):")
                for i, (drug, atc) in enumerate(zip(top_drugs[:len(self.vocab_atc)], self.vocab_atc), 1):
                    count = drug_counts[drug] if drug in drug_counts.index else 0
                    print(f"  {i}. {drug} → {atc} (appears {count} times)")
            else:
                print("  WARNING: No drug column found, using sample ATC codes")
                self.vocab_atc = [f"DRUG_{i:05d}" for i in range(self.num_atc)]
        else:
            print("  PRESCRIPTIONS.csv not found, using sample ATC codes")
            self.vocab_atc = [f"DRUG_{i:05d}" for i in range(self.num_atc)]
        
        return self.vocab_atc

    def build_atc_hierarchy(self):
        """
        Build ATC hierarchy for selected codes
        """
        print("\n" + "=" * 80)
        print("Step 4: Building ATC Hierarchy")
        print("=" * 80)
        
        # Add root node
        self.G.add_node('ATC_ROOT', type='ATC', level=0)
        
        atc_nodes = set()
        atc_nodes.add('ATC_ROOT')
        
        # Build hierarchy for each ATC code
        for atc_code in tqdm(self.vocab_atc, desc="  Processing ATC codes"):
            # Remove DRUG_ prefix if present
            code_clean = atc_code.replace('DRUG_', '').replace('ATC_', '')
            
            # For placeholder codes, create simple hierarchy
            if code_clean.isdigit():
                # Create simple hierarchy: ATC_ROOT → ATC_GROUP → ATC_XX → code
                prefixes = ['ATC_ROOT', 'ATC_GROUP', f'ATC_{code_clean[:2]}', atc_code]
            else:
                # Real ATC code structure
                prefixes = ['ATC_ROOT']
                if len(code_clean) >= 1:
                    prefixes.append(code_clean[0])
                if len(code_clean) >= 3:
                    prefixes.append(code_clean[:3])
                if len(code_clean) >= 4:
                    prefixes.append(code_clean[:4])
                if len(code_clean) >= 5:
                    prefixes.append(code_clean[:5])
                prefixes.append(atc_code)
            
            # Add all nodes
            all_nodes = prefixes
            
            for node in all_nodes:
                if node not in atc_nodes:
                    level = 0 if node == 'ATC_ROOT' else len([c for c in node if c.isalnum()])
                    self.G.add_node(node, type='ATC', level=level, 
                                   code=atc_code if node == atc_code else None)
                    atc_nodes.add(node)
            
            # Create hierarchical edges: parent → child
            for i in range(len(all_nodes) - 1):
                parent = all_nodes[i]
                child = all_nodes[i + 1]
                if not self.G.has_edge(parent, child):
                    self.G.add_edge(parent, child, edge_type='hierarchical')
        
        print(f"  ATC hierarchy: {len(atc_nodes)} nodes")
        print(f"  Nodes: {sorted(atc_nodes)}")
        return atc_nodes

    def load_icd_atc_relations(self):
        """
        Load ICD-ATC relations from mapping file or create from co-occurrence
        """
        print("\n" + "=" * 80)
        print("Step 5: Loading ICD-ATC Relations")
        print("=" * 80)
        
        if self.icd_atc_mapping_file and os.path.exists(self.icd_atc_mapping_file):
            print(f"  Loading from file: {self.icd_atc_mapping_file}")
            mapping_df = pd.read_csv(self.icd_atc_mapping_file)
            
            # Filter to only include our selected ICD and ATC codes
            icd_set = set(self.vocab_icd)
            atc_set = set(self.vocab_atc)
            
            # Normalize ICD codes for matching
            def normalize_icd_for_match(icd):
                if pd.isna(icd):
                    return None
                return str(icd).strip().replace('.', '')
            
            for _, row in tqdm(mapping_df.iterrows(), desc="  Processing relations", total=len(mapping_df)):
                icd_code = str(row.iloc[0]).strip()
                atc_code = str(row.iloc[1]).strip()
                relation_type = row.iloc[2] if len(row) > 2 else 'treats'
                
                # Normalize ICD code
                icd_normalized = normalize_icd_for_match(icd_code)
                
                # Check if both codes are in our selected sets
                icd_in_set = any(normalize_icd_for_match(icd) == icd_normalized for icd in icd_set)
                atc_in_set = atc_code in atc_set
                
                if icd_in_set and atc_in_set:
                    # Find the actual normalized ICD node in graph
                    icd_node = None
                    for node in self.G.nodes():
                        if self.G.nodes[node].get('type') == 'ICD9' and normalize_icd_for_match(node) == icd_normalized:
                            icd_node = node
                            break
                    
                    if icd_node and atc_code in self.G.nodes():
                        if not self.G.has_edge(icd_node, atc_code):
                            self.G.add_edge(icd_node, atc_code, 
                                           edge_type='icd_atc_relation',
                                           relation=relation_type)
                            self.icd_atc_relations.append((icd_node, atc_code, relation_type))
        else:
            print("  WARNING: ICD-ATC mapping file not provided")
            print("  Creating relations from co-occurrence in MIMIC-III...")
            
            # Load co-occurrence from MIMIC-III
            self._create_relations_from_cooccurrence()
        
        print(f"  ICD-ATC relations: {len(self.icd_atc_relations)} edges")
        if len(self.icd_atc_relations) > 0:
            print("  Relations:")
            for icd, atc, rel in self.icd_atc_relations[:10]:  # Show first 10
                print(f"    {icd} <-> {atc} ({rel})")
        
        return self.icd_atc_relations

    def _create_relations_from_cooccurrence(self):
        """Create ICD-ATC relations from co-occurrence in MIMIC-III"""
        print("  Extracting co-occurrence from MIMIC-III...")
        
        # Load tables
        diagnoses_file = os.path.join(self.mimic_path, 'DIAGNOSES_ICD.csv')
        prescriptions_file = os.path.join(self.mimic_path, 'PRESCRIPTIONS.csv')
        
        if not os.path.exists(diagnoses_file) or not os.path.exists(prescriptions_file):
            print("    WARNING: Cannot load tables, creating placeholder relations")
            # Create some placeholder relations
            icd_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('type') == 'ICD9' 
                        and self.G.nodes[n].get('level', 999) > 3]
            atc_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('type') == 'ATC' 
                        and self.G.nodes[n].get('level', 999) > 3]
            
            # Connect each ICD to a few ATCs
            for icd in icd_nodes[:1]:  # Only first ICD
                for atc in atc_nodes[:3]:  # First 3 ATCs
                    if not self.G.has_edge(icd, atc):
                        self.G.add_edge(icd, atc, edge_type='icd_atc_relation', relation='treats')
                        self.icd_atc_relations.append((icd, atc, 'treats'))
            return
        
        # Load data
        diagnoses_df = pd.read_csv(diagnoses_file)
        prescriptions_df = pd.read_csv(prescriptions_file, nrows=10000)  # Sample
        
        # Normalize ICD codes
        def normalize_icd(icd):
            if pd.isna(icd):
                return None
            return str(icd).strip().replace('.', '')
        
        # Get our selected ICD codes (normalized)
        selected_icd_normalized = set()
        for icd in self.vocab_icd:
            normalized = normalize_icd(icd)
            if normalized:
                selected_icd_normalized.add(normalized)
        
        # Find HADM_ID column (handle different column name variations)
        hadm_column = None
        for col in ['HADM_ID', 'hadm_id', 'HADMID', 'hadmid']:
            if col in diagnoses_df.columns:
                hadm_column = col
                break
        
        if hadm_column is None:
            print("    WARNING: Cannot find HADM_ID column in DIAGNOSES_ICD.csv")
            print(f"    Available columns: {list(diagnoses_df.columns)}")
            return
        
        # Find ICD column (should already be found in load_icd_codes_from_mimic, but check again)
        icd_column = None
        for col in ['ICD9_CODE', 'icd9_code', 'ICD_CODE', 'icd_code', 'ICD9', 'icd9']:
            if col in diagnoses_df.columns:
                icd_column = col
                break
        
        if icd_column is None:
            print("    WARNING: Cannot find ICD code column in DIAGNOSES_ICD.csv")
            return
        
        # Group by HADM_ID
        hadm_to_icd = defaultdict(set)
        for _, row in diagnoses_df.iterrows():
            hadm_id = row[hadm_column]
            icd_code = row[icd_column]
            if pd.notna(icd_code) and pd.notna(hadm_id):
                icd_norm = normalize_icd(icd_code)
                if icd_norm in selected_icd_normalized:
                    hadm_to_icd[hadm_id].add(icd_norm)
        
        # Group ATC by HADM_ID
        hadm_to_atc = defaultdict(set)
        # Find HADM_ID column in prescriptions
        hadm_column_pres = None
        for col in ['HADM_ID', 'hadm_id', 'HADMID', 'hadmid']:
            if col in prescriptions_df.columns:
                hadm_column_pres = col
                break
        
        if hadm_column_pres is None:
            print("    WARNING: Cannot find HADM_ID column in PRESCRIPTIONS.csv")
            print(f"    Available columns: {list(prescriptions_df.columns)}")
            return
        
        drug_column = None
        for col in ['drug_name_generic', 'drug', 'drug_name', 'DRUG']:
            if col in prescriptions_df.columns:
                drug_column = col
                break
        
        if drug_column:
            # Create drug to ATC mapping
            drug_to_atc = {}
            for drug in prescriptions_df[drug_column].dropna().unique():
                drug_str = str(drug).strip().upper()
                atc_code = f"DRUG_{abs(hash(drug_str)) % 100000:05d}"
                drug_to_atc[drug_str] = atc_code
            
            for _, row in prescriptions_df.iterrows():
                hadm_id = row[hadm_column_pres]
                drug = row[drug_column]
                if pd.notna(drug) and pd.notna(hadm_id):
                    drug_str = str(drug).strip().upper()
                    if drug_str in drug_to_atc:
                        atc_code = drug_to_atc[drug_str]
                        if atc_code in self.vocab_atc:
                            hadm_to_atc[hadm_id].add(atc_code)
        else:
            print("    WARNING: Cannot find drug column in PRESCRIPTIONS.csv")
            print(f"    Available columns: {list(prescriptions_df.columns)}")
        
        # Find co-occurrences
        common_admissions = set(hadm_to_icd.keys()) & set(hadm_to_atc.keys())
        
        for hadm_id in common_admissions:
            icd_codes = hadm_to_icd[hadm_id]
            atc_codes = hadm_to_atc[hadm_id]
            
            for icd_norm in icd_codes:
                # Find ICD node in graph
                icd_node = None
                for node in self.G.nodes():
                    if (self.G.nodes[node].get('type') == 'ICD9' and 
                        normalize_icd(node) == icd_norm):
                        icd_node = node
                        break
                
                for atc_code in atc_codes:
                    if icd_node and atc_code in self.G.nodes():
                        if not self.G.has_edge(icd_node, atc_code):
                            self.G.add_edge(icd_node, atc_code, 
                                           edge_type='icd_atc_relation',
                                           relation='treats')
                            self.icd_atc_relations.append((icd_node, atc_code, 'treats'))

    def augment_graph(self):
        """
        Augment graph by connecting each node to all of its ancestral nodes
        """
        print("\n" + "=" * 80)
        print("Step 6: Augmenting Graph (Skip Connections)")
        print("=" * 80)
        
        augmented_edges = 0
        
        # For each node, find all ancestors via hierarchical edges
        for node in tqdm(self.G.nodes(), desc="  Adding skip connections"):
            node_type = self.G.nodes[node].get('type', '')
            node_level = self.G.nodes[node].get('level', 999)
            
            if node_type not in ['ICD9', 'ATC']:
                continue
            
            # Find all ancestors by traversing up hierarchical edges
            ancestors = []
            current = node
            visited = set()
            
            while current and current not in visited:
                visited.add(current)
                
                # Find parent via hierarchical edge
                parent = None
                parent_level = node_level
                
                for neighbor in self.G.neighbors(current):
                    edge_data = self.G.get_edge_data(current, neighbor, {})
                    if edge_data.get('edge_type') == 'hierarchical':
                        neighbor_level = self.G.nodes[neighbor].get('level', 999)
                        neighbor_type = self.G.nodes[neighbor].get('type', '')
                        
                        # Parent should have lower level and same type (or be root)
                        if (neighbor_level < parent_level and 
                            (neighbor_type == node_type or neighbor.endswith('_ROOT'))):
                            parent = neighbor
                            parent_level = neighbor_level
                
                if parent and parent not in ancestors:
                    ancestors.append(parent)
                    current = parent
                else:
                    break
            
            # Add edges to all ancestors (skip connections)
            for ancestor in ancestors:
                if not self.G.has_edge(node, ancestor):
                    distance = abs(self.G.nodes[node].get('level', 999) - 
                                  self.G.nodes[ancestor].get('level', 0))
                    weight = 0.9 ** distance
                    
                    self.G.add_edge(node, ancestor, 
                                   edge_type='augmented',
                                   weight=weight)
                    augmented_edges += 1
        
        print(f"  Augmented edges added: {augmented_edges}")
        return augmented_edges

    def generate_embeddings(self):
        """Generate node embeddings using Node2Vec"""
        print("\n" + "=" * 80)
        print("Step 7: Generating Node2Vec Embeddings")
        print("=" * 80)
        
        if Node2Vec is None:
            print("  WARNING: node2vec not available, using random embeddings")
            num_nodes = self.G.number_of_nodes()
            embeddings = np.random.randn(num_nodes, self.embedding_dim)
            node_list = list(self.G.nodes())
            return embeddings, node_list
        
        # Convert to undirected if not already
        if self.G.is_directed():
            self.G = self.G.to_undirected()
        
        print(f"  Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(self.G, 
                           dimensions=self.embedding_dim,
                           walk_length=self.walk_length,
                           num_walks=self.num_walks,
                           p=1, q=1,
                           workers=1)
        
        # Train model
        print("  Training Node2Vec model...")
        model = node2vec.fit(window=self.window, min_count=1, batch_words=4)
        
        # Extract embeddings
        node_list = list(self.G.nodes())
        embeddings = np.zeros((len(node_list), self.embedding_dim))
        
        for i, node in enumerate(node_list):
            try:
                embeddings[i] = model.wv[str(node)]
            except KeyError:
                embeddings[i] = np.random.randn(self.embedding_dim)
        
        print(f"  Embeddings generated: shape {embeddings.shape}")
        return embeddings, node_list

    def renumber_nodes_by_vocab(self, embeddings):
        """Renumber nodes according to vocabulary order"""
        print("\n" + "=" * 80)
        print("Step 8: Renumbering Nodes by Vocabulary Order")
        print("=" * 80)
        
        renumber = {}
        new_index = 0
        
        def normalize_icd_for_graph(icd):
            if pd.isna(icd):
                return None
            code_str = str(icd).strip().replace('.', '')
            return code_str if code_str else None
        
        # First, add ICD vocabulary nodes
        for icd in self.vocab_icd:
            icd_normalized = normalize_icd_for_graph(icd)
            if icd_normalized and icd_normalized in self.G.nodes():
                if icd_normalized not in renumber:
                    renumber[icd_normalized] = new_index
                    new_index += 1
        
        # Then, add ATC vocabulary nodes
        for atc in self.vocab_atc:
            if atc in self.G.nodes():
                if atc not in renumber:
                    renumber[atc] = new_index
                    new_index += 1
        
        # Add remaining nodes
        remaining_nodes = set(self.G.nodes()) - set(renumber.keys())
        for node in sorted(remaining_nodes):
            renumber[node] = new_index
            new_index += 1
        
        # Create reverse mapping
        graphnode_vocab = {v: k for k, v in renumber.items()}
        
        # Relabel graph
        G_renumbered = nx.relabel_nodes(self.G, renumber)
        nx.set_node_attributes(G_renumbered, name='code', values=graphnode_vocab)
        
        # Reorder embeddings
        node_list_old = list(self.G.nodes())
        embeddings_renumbered = np.zeros_like(embeddings)
        
        for old_node, new_idx in renumber.items():
            old_idx = node_list_old.index(old_node)
            embeddings_renumbered[new_idx] = embeddings[old_idx]
        
        print(f"  Renumbered: {len(renumber)} nodes")
        print(f"    Vocab nodes: {len(self.vocab_icd) + len(self.vocab_atc)}")
        print(f"    Other nodes: {len(remaining_nodes)}")
        
        return G_renumbered, embeddings_renumbered, graphnode_vocab

    def save_outputs(self, G_renumbered, embeddings, graphnode_vocab):
        """Save graph and embeddings"""
        print("\n" + "=" * 80)
        print("Step 9: Saving Outputs")
        print("=" * 80)
        
        # Create vocab_info
        vocab_info = {
            'icd': self.vocab_icd,
            'atc': self.vocab_atc
        }
        
        # Save graph
        graph_filename = os.path.join(self.output_dir, 
                                     f'icdatc_graph_simple_{self.window}_{self.walk_length}_{self.num_walks}_{self.embedding_dim}_renumbered_by_vocab.pkl')
        with open(graph_filename, 'wb') as f:
            pickle.dump(G_renumbered, f)
        print(f"  Graph saved: {graph_filename}")
        
        # Save embeddings
        embed_filename = os.path.join(self.output_dir,
                                     f'icdatc_embed_simple_{self.window}_{self.walk_length}_{self.num_walks}_{self.embedding_dim}_by_vocab.pkl')
        with open(embed_filename, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"  Embeddings saved: {embed_filename}")
        
        # Save vocab mapping
        vocab_filename = os.path.join(self.output_dir, 'graphnode_vocab.pkl')
        with open(vocab_filename, 'wb') as f:
            pickle.dump(graphnode_vocab, f)
        print(f"  Vocab mapping saved: {vocab_filename}")
        
        # Save vocab info
        vocab_info_filename = os.path.join(self.output_dir, 'vocab_info.pkl')
        with open(vocab_info_filename, 'wb') as f:
            pickle.dump(vocab_info, f)
        print(f"  Vocab info saved: {vocab_info_filename}")
        
        return graph_filename, embed_filename

    def build(self):
        """Main build function"""
        print("=" * 80)
        print("Building Simple Knowledge Graph based on GAT-ETM Paper")
        print(f"Using {self.num_icd} ICD codes and {self.num_atc} ATC codes")
        print("=" * 80)
        
        # Step 1: Load ICD codes
        self.load_icd_codes_from_mimic()
        
        # Step 2: Build ICD hierarchy
        self.build_icd_hierarchy()
        
        # Step 3: Load ATC codes
        self.load_atc_codes()
        
        # Step 4: Build ATC hierarchy
        self.build_atc_hierarchy()
        
        print(f"\nGraph after hierarchies: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Step 5: Load ICD-ATC relations
        self.load_icd_atc_relations()
        
        print(f"\nGraph after ICD-ATC relations: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Step 6: Augment graph
        self.augment_graph()
        
        print(f"\nGraph after augmentation: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Step 7: Generate embeddings
        embeddings, node_list = self.generate_embeddings()
        
        # Step 8: Renumber nodes
        G_renumbered, embeddings_renumbered, graphnode_vocab = self.renumber_nodes_by_vocab(embeddings)
        
        # Step 9: Save outputs
        graph_file, embed_file = self.save_outputs(G_renumbered, embeddings_renumbered, graphnode_vocab)
        
        print("\n" + "=" * 80)
        print("Simple Knowledge Graph built successfully!")
        print("=" * 80)
        print(f"Final graph: {G_renumbered.number_of_nodes()} nodes, {G_renumbered.number_of_edges()} edges")
        print(f"Embeddings shape: {embeddings_renumbered.shape}")
        
        # Print detailed statistics
        node_types = Counter([G_renumbered.nodes[node].get('type', 'UNKNOWN') for node in G_renumbered.nodes()])
        edge_types = Counter([G_renumbered.edges[edge].get('edge_type', 'unknown') for edge in G_renumbered.edges()])
        
        print(f"\nNode types: {dict(node_types)}")
        print(f"Edge types: {dict(edge_types)}")
        print(f"Vocabulary sizes:")
        print(f"  ICD: {len(self.vocab_icd)}")
        print(f"  ATC: {len(self.vocab_atc)}")
        
        # Print all nodes and edges for manual verification
        print(f"\n" + "=" * 80)
        print("Graph Structure (for Manual Verification)")
        print("=" * 80)
        print("\nAll Nodes:")
        for node in sorted(G_renumbered.nodes()):
            node_type = G_renumbered.nodes[node].get('type', 'UNKNOWN')
            level = G_renumbered.nodes[node].get('level', 999)
            code = G_renumbered.nodes[node].get('code', None)
            print(f"  {node}: type={node_type}, level={level}, code={code}")
        
        print("\nAll Edges:")
        for u, v, data in sorted(G_renumbered.edges(data=True)):
            edge_type = data.get('edge_type', 'unknown')
            weight = data.get('weight', 1.0)
            print(f"  {u} <-> {v}: type={edge_type}, weight={weight:.3f}")
        
        return G_renumbered, embeddings_renumbered


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Simple Knowledge Graph based on GAT-ETM Paper')
    parser.add_argument('--mimic_path', type=str, 
                       default='mimic-iii-clinical-database-demo-1.4',
                       help='Path to MIMIC-III data directory')
    parser.add_argument('--output_dir', type=str, default='embed_paper_simple',
                       help='Output directory for graph and embeddings')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--window', type=int, default=8,
                       help='Node2Vec window size')
    parser.add_argument('--walk_length', type=int, default=20,
                       help='Node2Vec walk length')
    parser.add_argument('--num_walks', type=int, default=10,
                       help='Node2Vec number of walks')
    parser.add_argument('--icd_atc_mapping', type=str, default=None,
                       help='Path to ICD-ATC mapping file (CSV format)')
    parser.add_argument('--num_icd', type=int, default=1,
                       help='Number of ICD codes to use (default: 1)')
    parser.add_argument('--num_atc', type=int, default=5,
                       help='Number of ATC codes to use (default: 5)')
    
    args = parser.parse_args()
    
    # Create builder
    builder = Simple_Paper_KG_Builder(
        mimic_path=args.mimic_path,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        window=args.window,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        icd_atc_mapping_file=args.icd_atc_mapping,
        num_icd=args.num_icd,
        num_atc=args.num_atc
    )
    
    # Build graph
    G, embeddings = builder.build()
    
    print("\n" + "=" * 80)
    print("Build complete!")
    print("=" * 80)
    print(f"\nTo visualize:")
    print(f"  python visualize/visualize_graph.py \\")
    print(f"    --graph_file {args.output_dir}/icdatc_graph_simple_{args.window}_{args.walk_length}_{args.num_walks}_{args.embedding_dim}_renumbered_by_vocab.pkl \\")
    print(f"    --vocab_info_file {args.output_dir}/vocab_info.pkl \\")
    print(f"    --output visualize/graph_visualization_paper_simple.png")


if __name__ == '__main__':
    main()
