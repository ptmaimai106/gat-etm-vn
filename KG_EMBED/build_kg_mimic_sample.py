#!/usr/bin/env python3
"""
Script to build Knowledge Graph from MIMIC-III for GAT-ETM training

This script:
1. Extracts ICD9, CPT, ATC (drugs), and Lab codes from MIMIC-III
2. Builds hierarchical edges (ICD9, CPT, ATC trees)
3. Builds co-occurrence edges from admissions
4. Generates Node2Vec embeddings
5. Renumbers nodes according to vocab order
6. Saves graph and embeddings as pickle files
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


class MIMIC_KG_Builder:
    def __init__(self, mimic_path='mimic-iii-clinical-database-demo-1.4', 
                 output_dir='embed', 
                 embedding_dim=256,
                 window=8,
                 walk_length=20,
                 num_walks=10,
                 augmented=True):
        """
        Initialize KG Builder
        
        Args:
            mimic_path: Path to MIMIC-III data directory
            output_dir: Directory to save output files
            embedding_dim: Dimension of node embeddings
            window: Node2Vec window size
            walk_length: Node2Vec walk length
            num_walks: Node2Vec number of walks
            augmented: Whether to add augmented edges
        """
        self.mimic_path = mimic_path
        self.output_dir = output_dir
        self.embedding_dim = embedding_dim
        self.window = window
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.augmented = augmented
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize graph
        self.G = nx.Graph()
        
        # Code vocabularies
        self.vocab_icd = []
        self.vocab_cpt = []
        self.vocab_atc = []
        self.vocab_lab = []
        
        # Code mappings
        self.icd_code_to_node = {}
        self.cpt_code_to_node = {}
        self.atc_code_to_node = {}
        self.lab_code_to_node = {}
        
    def load_mimic_tables(self):
        """Load all necessary MIMIC-III tables"""
        print("Loading MIMIC-III tables...")
        
        self.d_icd_diagnoses = pd.read_csv(
            os.path.join(self.mimic_path, 'D_ICD_DIAGNOSES.csv')
        )
        self.diagnoses_icd = pd.read_csv(
            os.path.join(self.mimic_path, 'DIAGNOSES_ICD.csv')
        )
        
        self.d_icd_procedures = pd.read_csv(
            os.path.join(self.mimic_path, 'D_ICD_PROCEDURES.csv')
        )
        self.procedures_icd = pd.read_csv(
            os.path.join(self.mimic_path, 'PROCEDURES_ICD.csv')
        )
        
        # Try to load CPT tables if available
        try:
            self.d_cpt = pd.read_csv(
                os.path.join(self.mimic_path, 'D_CPT.csv')
            )
            self.cpt_events = pd.read_csv(
                os.path.join(self.mimic_path, 'CPTEVENTS.csv')
            )
        except FileNotFoundError:
            print("Warning: CPT tables not found, skipping CPT codes")
            self.d_cpt = None
            self.cpt_events = None
        
        self.prescriptions = pd.read_csv(
            os.path.join(self.mimic_path, 'PRESCRIPTIONS.csv')
        )
        
        self.lab_events = pd.read_csv(
            os.path.join(self.mimic_path, 'LABEVENTS.csv')
        )
        self.d_labitems = pd.read_csv(
            os.path.join(self.mimic_path, 'D_LABITEMS.csv')
        )
        
        self.admissions = pd.read_csv(
            os.path.join(self.mimic_path, 'ADMISSIONS.csv')
        )
        print("Tables loaded successfully!")
        print("After that, choose the sampling data")

        # ===== SAMPLE MODE =====
        # Tăng số subjects để có đủ dữ liệu nhưng vẫn giữ nhỏ
        # Lấy 3-5 subjects để có đủ ICD9 codes cho hierarchy
        NUM_SAMPLE_SUBJECTS = 5
        SAMPLE_SUBJECTS = set(self.admissions['subject_id'].dropna().unique()[:NUM_SAMPLE_SUBJECTS])
        print(f"Sampling {len(SAMPLE_SUBJECTS)} subjects for small KG")
        
        self.admissions = self.admissions[self.admissions['subject_id'].isin(SAMPLE_SUBJECTS)]

        self.diagnoses_icd = self.diagnoses_icd[self.diagnoses_icd['subject_id'].isin(SAMPLE_SUBJECTS)]
        self.procedures_icd = self.procedures_icd[self.procedures_icd['subject_id'].isin(SAMPLE_SUBJECTS)]
        self.prescriptions = self.prescriptions[self.prescriptions['subject_id'].isin(SAMPLE_SUBJECTS)]
        self.lab_events = self.lab_events[self.lab_events['subject_id'].isin(SAMPLE_SUBJECTS)]
        if self.cpt_events is not None:
            self.cpt_events = self.cpt_events[self.cpt_events['subject_id'].isin(SAMPLE_SUBJECTS)]


    def build_icd9_hierarchy(self):
        """Build ICD9 diagnosis hierarchy - SIMPLE MODE: only 1 ICD-9 code"""
        print("Building ICD9 hierarchy (SIMPLE MODE: 1 ICD-9 code)...")
        
        # Get ICD9 codes with frequency
        icd9_counts = Counter(self.diagnoses_icd['icd9_code'].dropna())
        
        # Select top 1 most frequent ICD9 code
        if len(icd9_counts) == 0:
            print("No ICD9 codes found, skipping ICD9 hierarchy")
            return
        
        top_icd9_code = icd9_counts.most_common(1)[0][0]
        print(f"  Selected ICD-9 code: {top_icd9_code} (frequency: {icd9_counts[top_icd9_code]})")
        
        # Normalize ICD9 codes (handle both with/without dots)
        def normalize_icd9(code, keep_format=False):
            if pd.isna(code):
                return None
            code_str = str(code).strip()
            # Remove dots for internal representation
            code_str_no_dot = code_str.replace('.', '')
            # Return format based on keep_format flag
            if keep_format and '.' in code_str:
                # Keep original format with dot
                return code_str
            else:
                # Return without dot (for hierarchy building)
                return code_str_no_dot
        
        # Build hierarchy for selected ICD9 code only
        normalized = normalize_icd9(top_icd9_code)
        if not normalized or len(normalized) < 3:
            print("  Invalid ICD9 code format, skipping")
            return
        
        # Add all prefix levels for this code
        icd9_nodes = set()
        icd9_edges = []
        
        for i in range(3, len(normalized) + 1):
            prefix = normalized[:i]
            icd9_nodes.add(prefix)
            # Add edge to parent (if exists)
            if i > 3:
                parent = normalized[:i-1]
                if parent in icd9_nodes:
                    icd9_edges.append((parent, prefix))
        
        # Add root node
        root_icd = 'ICD9_ROOT'
        
        # Rebuild edges cho selected nodes, đảm bảo hierarchy đúng
        icd9_edges = []
        for node in sorted(icd9_nodes, key=lambda x: len(x)):  # Xử lý từ ngắn đến dài
            if len(node) == 3:  # Top-level 3-digit codes
                icd9_edges.append((root_icd, node))
            elif len(node) > 3:
                # Tìm parent node trực tiếp (bỏ 1 ký tự cuối)
                parent = node[:-1]
                if parent in icd9_nodes:
                    icd9_edges.append((parent, node))
                else:
                    # Tìm parent gần nhất
                    found_parent = False
                    for i in range(len(node) - 1, 2, -1):  # Từ độ dài hiện tại về 3
                        parent = node[:i]
                        if parent in icd9_nodes:
                            icd9_edges.append((parent, node))
                            found_parent = True
                            break
                    if not found_parent:
                        # Nếu không tìm thấy parent, kết nối với root
                        icd9_edges.append((root_icd, node))
        
        # Add nodes and edges to graph
        self.G.add_node(root_icd, type='ICD9', level=0)
        for node in icd9_nodes:
            self.G.add_node(node, type='ICD9', level=len(node))
        for parent, child in icd9_edges:
            self.G.add_edge(parent, child, edge_type='hierarchical')
        
        # Store vocabulary (only the selected ICD9 code)
        # Keep original format for vocab (with dots if present)
        self.vocab_icd = []
        orig_str = str(top_icd9_code).strip()
        if '.' in orig_str:
            self.vocab_icd.append(orig_str)
        else:
            self.vocab_icd.append(normalized)
        self.vocab_icd = sorted(list(set(self.vocab_icd)))
        
        print(f"ICD9 hierarchy: {len(icd9_nodes)} nodes, {len(icd9_edges)} edges")
        print(f"ICD9 vocabulary size: {len(self.vocab_icd)}")
        # Log chi tiết về hierarchy
        level_dist = {}
        for node in icd9_nodes:
            level = len(node)
            level_dist[level] = level_dist.get(level, 0) + 1
        print(f"  ICD9 nodes by level: {level_dist}")
        print(f"  ICD9 hierarchical edges: {len(icd9_edges)}")
    
    def build_cpt_hierarchy(self):
        """Build CPT procedure hierarchy"""
        if self.d_cpt is None:
            print("Skipping CPT hierarchy (tables not available)")
            return
        
        print("Building CPT hierarchy...")
        
        # Get unique CPT codes from procedures
        cpt_codes = set()
        if self.cpt_events is not None:
            cpt_codes.update(self.cpt_events['cpt_cd'].dropna().unique())
        
        # Also get from ICD procedures (some may be CPT-like)
        # For now, we'll build a simple hierarchy based on CPT sections
        
        if len(cpt_codes) == 0:
            print("No CPT codes found, skipping CPT hierarchy")
            return
        
        # CPT codes are typically 5-digit numeric codes
        # Build hierarchy: section -> subsection -> code
        root_cpt = 'CPT_ROOT'
        self.G.add_node(root_cpt, type='CPT', level=0)
        
        cpt_nodes = set()
        cpt_edges = []
        
        for code in cpt_codes:
            code_str = str(code).strip()
            if len(code_str) >= 1:
                cpt_nodes.add(code_str)
                # For CPT, we can group by first digit or use D_CPT structure
                if len(code_str) >= 1:
                    section = code_str[0]
                    section_node = f'CPT_{section}'
                    if section_node not in cpt_nodes:
                        cpt_nodes.add(section_node)
                        cpt_edges.append((root_cpt, section_node))
                    cpt_edges.append((section_node, code_str))
        
        # Add nodes and edges
        for node in cpt_nodes:
            if node.startswith('CPT_'):
                self.G.add_node(node, type='CPT', level=1)
            else:
                self.G.add_node(node, type='CPT', level=2)
        
        for parent, child in cpt_edges:
            self.G.add_edge(parent, child, edge_type='hierarchical')
        
        self.vocab_cpt = sorted([str(c) for c in cpt_codes])
        
        print(f"CPT hierarchy: {len(cpt_nodes)} nodes, {len(cpt_edges)} edges")
        print(f"CPT vocabulary size: {len(self.vocab_cpt)}")
    
    def extract_drugs_and_map_to_atc(self):
        """Extract drugs from PRESCRIPTIONS and attempt ATC mapping - SIMPLE MODE: top 5 ATC codes"""
        print("Extracting drugs and mapping to ATC (SIMPLE MODE: top 5 ATC codes)...")
        
        # Get drugs with frequency
        drug_counts = Counter(self.prescriptions['drug_name_generic'].dropna())
        
        if len(drug_counts) == 0:
            print("No drugs found, skipping ATC hierarchy")
            return
        
        # Select top 5 most frequent drugs
        top_drugs = [drug for drug, count in drug_counts.most_common(5)]
        print(f"  Selected top 5 drugs: {top_drugs}")
        print(f"  Frequencies: {[drug_counts[drug] for drug in top_drugs]}")
        
        # For now, we'll create a simple mapping
        # In production, you would use RxNorm API or ATC mapping file
        # Here we create placeholder ATC codes based on drug names
        
        atc_codes = set()
        drug_to_atc = {}
        
        # Simple heuristic: create ATC-like codes from drug names
        # This is a placeholder - in real scenario, use proper ATC mapping
        for drug in top_drugs:
            if pd.isna(drug):
                continue
            drug_str = str(drug).strip().upper()
            # Create a simple hash-based code (not real ATC, but for structure)
            # In production, use proper RxNorm -> ATC mapping
            atc_code = f"DRUG_{hash(drug_str) % 100000:05d}"
            atc_codes.add(atc_code)
            drug_to_atc[drug_str] = atc_code
        
        # Build ATC hierarchy (simplified)
        root_atc = 'ATC_ROOT'
        self.G.add_node(root_atc, type='ATC', level=0)
        
        atc_nodes = set([root_atc])
        atc_edges = []
        
        # Group ATC codes by first character (simplified hierarchy)
        for atc_code in atc_codes:
            atc_nodes.add(atc_code)
            if len(atc_code) > 5:
                # Create intermediate levels
                level1 = atc_code[:6]  # DRUG_
                level2 = atc_code[:7]  # DRUG_X
                if level1 not in atc_nodes:
                    atc_nodes.add(level1)
                    atc_edges.append((root_atc, level1))
                if level2 not in atc_nodes and level2 != atc_code:
                    atc_nodes.add(level2)
                    atc_edges.append((level1, level2))
                atc_edges.append((level2, atc_code))
            else:
                atc_edges.append((root_atc, atc_code))
        
        # Add nodes and edges
        for node in atc_nodes:
            if node == root_atc:
                continue
            level = len([e for e in atc_edges if e[1] == node])
            self.G.add_node(node, type='ATC', level=level)
        
        for parent, child in atc_edges:
            self.G.add_edge(parent, child, edge_type='hierarchical')
        
        self.vocab_atc = sorted(list(atc_codes))
        self.drug_to_atc = drug_to_atc
        
        print(f"ATC hierarchy: {len(atc_nodes)} nodes, {len(atc_edges)} edges")
        print(f"ATC vocabulary size: {len(self.vocab_atc)}")
        print("Note: Using placeholder ATC codes. For production, use proper RxNorm->ATC mapping.")
    
    def extract_lab_codes(self):
        """Extract lab item codes - SIMPLE MODE: top 10 lab codes"""
        print("Extracting lab codes (SIMPLE MODE: top 10 lab codes)...")
        
        # Get lab itemids with frequency
        lab_counts = Counter(self.lab_events['itemid'].dropna())
        
        if len(lab_counts) == 0:
            print("No lab codes found, skipping lab hierarchy")
            return
        
        # Select top 10 most frequent lab codes
        top_labs = [itemid for itemid, count in lab_counts.most_common(10)]
        print(f"  Selected top 10 lab codes: {top_labs}")
        print(f"  Frequencies: {[lab_counts[itemid] for itemid in top_labs]}")
        
        root_lab = 'LAB_ROOT'
        self.G.add_node(root_lab, type='LAB', level=0)
        
        lab_nodes = set([root_lab])
        lab_edges = []
        
        # Group labs by category from D_LABITEMS (only for selected labs)
        lab_categories = {}
        for _, row in self.d_labitems.iterrows():
            itemid = row['itemid']
            if itemid in top_labs:
                category = str(row.get('category', 'UNKNOWN'))
                lab_categories[itemid] = category
                lab_node = str(itemid)
                lab_nodes.add(lab_node)
                
                category_node = f'LAB_{category}'
                if category_node not in lab_nodes:
                    lab_nodes.add(category_node)
                    lab_edges.append((root_lab, category_node))
                
                lab_edges.append((category_node, lab_node))
        
        # Add nodes and edges
        for node in lab_nodes:
            if node == root_lab:
                continue
            if node.startswith('LAB_'):
                self.G.add_node(node, type='LAB', level=1)
            else:
                self.G.add_node(node, type='LAB', level=2)
        
        for parent, child in lab_edges:
            self.G.add_edge(parent, child, edge_type='hierarchical')
        
        self.vocab_lab = sorted([str(itemid) for itemid in top_labs])
        
        print(f"Lab hierarchy: {len(lab_nodes)} nodes, {len(lab_edges)} edges")
        print(f"Lab vocabulary size: {len(self.vocab_lab)}")
    
    def build_cooccurrence_edges(self, min_cooccurrence=1):
        """Build co-occurrence edges from admissions"""
        print("Building co-occurrence edges from admissions...")
        
        # Group by HADM_ID (hospital admission ID)
        hadm_to_icd = defaultdict(set)
        hadm_to_atc = defaultdict(set)
        hadm_to_cpt = defaultdict(set)
        hadm_to_lab = defaultdict(set)
        
        # Map ICD9 codes
        # Normalize to match graph nodes (without dots)
        def normalize_icd9_for_mapping(code):
            if pd.isna(code):
                return None
            code_str = str(code).strip().replace('.', '')
            return code_str if code_str and code_str != 'nan' else None
        
        for _, row in self.diagnoses_icd.iterrows():
            hadm_id = row['hadm_id']
            icd9 = normalize_icd9_for_mapping(row['icd9_code'])
            if icd9 and icd9 in self.G.nodes():
                hadm_to_icd[hadm_id].add(icd9)
        
        # Map ATC codes (from prescriptions)
        for _, row in self.prescriptions.iterrows():
            hadm_id = row['hadm_id']
            drug = str(row.get('drug_name_generic', '')).strip()
            if drug and drug != 'nan' and drug in self.drug_to_atc:
                atc_code = self.drug_to_atc[drug]
                hadm_to_atc[hadm_id].add(atc_code)
        
        # Map CPT codes
        if self.procedures_icd is not None:
            for _, row in self.procedures_icd.iterrows():
                hadm_id = row['hadm_id']
                cpt = str(row['icd9_code']).strip()  # ICD9 procedure code
                if cpt and cpt != 'nan':
                    hadm_to_cpt[hadm_id].add(cpt)
        
        # Map Lab codes
        # Lab events don't have hadm_id directly, need to use ICUSTAYS or ADMISSIONS
        # Try to load ICUSTAYS for better mapping
        try:
            icustays = pd.read_csv(
                os.path.join(self.mimic_path, 'ICUSTAYS.csv')
            )
            # Create mapping: subject_id + charttime -> hadm_id
            # For simplicity, map subject_id to most recent hadm_id
            subject_to_hadm = {}
            for _, row in self.admissions.iterrows():
                subject_id = row['subject_id']
                hadm_id = row['hadm_id']
                # Keep most recent admission for each subject
                if subject_id not in subject_to_hadm:
                    subject_to_hadm[subject_id] = hadm_id
        except FileNotFoundError:
            # Fallback: use ADMISSIONS only
            subject_to_hadm = {}
            for _, row in self.admissions.iterrows():
                subject_id = row['subject_id']
                hadm_id = row['hadm_id']
                if subject_id not in subject_to_hadm:
                    subject_to_hadm[subject_id] = hadm_id
        
        # Map lab events to hadm_id
        for _, row in self.lab_events.iterrows():
            subject_id = row['subject_id']
            if subject_id in subject_to_hadm:
                hadm_id = subject_to_hadm[subject_id]
                itemid = str(row['itemid'])
                if itemid and itemid != 'nan' and itemid != 'nan':
                    hadm_to_lab[hadm_id].add(itemid)
        
        # Build co-occurrence edges
        cooccurrence_edges = []
        cooccurrence_counts = Counter()
        
        print("  Processing admissions for co-occurrence...")
        # Tăng số admissions được xử lý để có đủ co-occurrence patterns
        # Nhưng vẫn giữ nhỏ để KG không quá lớn
        MAX_ADMISSIONS = 10  # Tăng từ 1 lên 10
        hadm_list = list(hadm_to_icd.keys())[:MAX_ADMISSIONS]
        print(f"  Processing {len(hadm_list)} admissions for co-occurrence edges")
        for hadm_id in tqdm(hadm_list, desc="  Admissions"):
            # ICD9 <-> ATC
            for icd in hadm_to_icd[hadm_id]:
                for atc in hadm_to_atc[hadm_id]:
                    edge = (icd, atc)
                    cooccurrence_edges.append(edge)
                    cooccurrence_counts[edge] += 1
            
            # ICD9 <-> CPT
            for icd in hadm_to_icd[hadm_id]:
                for cpt in hadm_to_cpt[hadm_id]:
                    edge = (icd, cpt)
                    cooccurrence_edges.append(edge)
                    cooccurrence_counts[edge] += 1
            
            # ICD9 <-> Lab
            for icd in hadm_to_icd[hadm_id]:
                for lab in hadm_to_lab[hadm_id]:
                    edge = (icd, lab)
                    cooccurrence_edges.append(edge)
                    cooccurrence_counts[edge] += 1
            
            # ICD9 <-> ICD9 (self co-occurrence)
            icd_list = list(hadm_to_icd[hadm_id])
            for i, icd1 in enumerate(icd_list):
                for icd2 in icd_list[i+1:]:
                    edge = (icd1, icd2)
                    cooccurrence_edges.append(edge)
                    cooccurrence_counts[edge] += 1
        
        # Add edges to graph (only if nodes exist and meet threshold)
        added_edges = 0
        for edge, count in cooccurrence_counts.items():
            if count >= min_cooccurrence:
                node1, node2 = edge
                if node1 in self.G.nodes() and node2 in self.G.nodes():
                    self.G.add_edge(node1, node2, 
                                  edge_type='cooccurrence', 
                                  weight=count)
                    added_edges += 1
        
        print(f"Co-occurrence edges: {added_edges} edges added")
        print(f"  Total co-occurrence pairs: {len(cooccurrence_counts)}")
    
    def create_ego_graph(self):
        """Create ego-graph from selected nodes (1 ICD-9, top 10 lab, top 5 ATC)"""
        print("Creating ego-graph from selected nodes...")
        
        # Get selected nodes (vocab nodes)
        selected_nodes = set()
        
        # Helper function to normalize ICD for graph matching
        def normalize_icd_for_graph(icd):
            """Normalize ICD code to match graph node format"""
            if pd.isna(icd):
                return None
            code_str = str(icd).strip().replace('.', '')
            return code_str if code_str else None
        
        # Add ICD-9 nodes (including hierarchy)
        for icd in self.vocab_icd:
            # Normalize to match graph nodes
            icd_normalized = normalize_icd_for_graph(icd)
            if icd_normalized and icd_normalized in self.G.nodes():
                selected_nodes.add(icd_normalized)
                # Add all neighbors (hierarchy nodes)
                selected_nodes.update(self.G.neighbors(icd_normalized))
        
        # Add Lab nodes (including hierarchy)
        for lab in self.vocab_lab:
            lab_str = str(lab)
            if lab_str in self.G.nodes():
                selected_nodes.add(lab_str)
                # Add all neighbors (hierarchy nodes)
                selected_nodes.update(self.G.neighbors(lab_str))
        
        # Add ATC nodes (including hierarchy)
        for atc in self.vocab_atc:
            if atc in self.G.nodes():
                selected_nodes.add(atc)
                # Add all neighbors (hierarchy nodes)
                selected_nodes.update(self.G.neighbors(atc))
        
        # Also ensure root nodes are included
        root_nodes = ['ICD9_ROOT', 'LAB_ROOT', 'ATC_ROOT', 'CPT_ROOT']
        for root in root_nodes:
            if root in self.G.nodes():
                selected_nodes.add(root)
        
        # Create ego-graph
        ego_graph = self.G.subgraph(selected_nodes).copy()
        
        print(f"  Original graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  Ego-graph: {ego_graph.number_of_nodes()} nodes, {ego_graph.number_of_edges()} edges")
        
        # Replace original graph with ego-graph
        self.G = ego_graph
        
        # Log node types in ego-graph
        node_types = Counter([self.G.nodes[node].get('type', 'UNKNOWN') for node in self.G.nodes()])
        print(f"  Node types in ego-graph: {dict(node_types)}")
    
    def augment_graph(self):
        """Add augmented edges (skip connections in hierarchy)"""
        if not self.augmented:
            return
        
        print("Augmenting graph with skip connections...")
        
        e = 0.9  # Decay factor
        added_edges = 0
        
        # For each leaf node, add edges to ancestors
        for node in list(self.G.nodes()):
            node_type = self.G.nodes[node].get('type', '')
            if node_type in ['ICD9', 'ATC', 'CPT']:
                # Find all ancestors
                ancestors = []
                current = node
                while True:
                    predecessors = list(self.G.predecessors(current))
                    if not predecessors:
                        break
                    parent = predecessors[0]  # Take first parent
                    if parent != current:
                        ancestors.append(parent)
                        current = parent
                    else:
                        break
                
                # Add edges to ancestors with decaying weights
                for i, ancestor in enumerate(ancestors):
                    if not self.G.has_edge(node, ancestor):
                        weight = e ** (i + 1)
                        self.G.add_edge(node, ancestor, 
                                      edge_type='augmented', 
                                      weight=weight)
                        added_edges += 1
        
        print(f"Augmented edges: {added_edges} edges added")
    
    def generate_embeddings(self):
        """Generate Node2Vec embeddings"""
        print("Generating Node2Vec embeddings...")
        
        if Node2Vec is None:
            print("Warning: node2vec not available, using random embeddings")
            num_nodes = self.G.number_of_nodes()
            embeddings = np.random.randn(num_nodes, self.embedding_dim).astype(np.float32)
            # Normalize
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            return embeddings
        
        # Convert to undirected for Node2Vec
        G_undirected = self.G.to_undirected()
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            G_undirected,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=4,
            p=1,  # Return parameter
            q=1   # In-out parameter
        )
        
        # Fit model
        print("  Training Node2Vec model...")
        model = node2vec.fit(window=self.window, min_count=1, batch_words=4)
        
        # Get embeddings for all nodes
        node_list = list(self.G.nodes())
        embeddings = np.zeros((len(node_list), self.embedding_dim), dtype=np.float32)
        
        for i, node in enumerate(node_list):
            embeddings[i] = model.wv[str(node)]
        
        print(f"Embeddings generated: shape {embeddings.shape}")
        return embeddings
    
    def renumber_nodes_by_vocab(self, embeddings):
        """Renumber nodes according to vocab order"""
        print("Renumbering nodes by vocabulary order...")
        
        # Create mapping: old_node -> new_index
        renumber = {}
        new_index = 0
        
        # Helper function to normalize ICD for graph matching
        def normalize_icd_for_graph(icd):
            """Normalize ICD code to match graph node format"""
            if pd.isna(icd):
                return None
            code_str = str(icd).strip().replace('.', '')
            return code_str if code_str else None
        
        # First, add vocab nodes in order
        # ICD9 - need to normalize to match graph nodes
        # Create mapping from vocab codes to graph nodes
        vocab_to_graph_icd = {}
        for icd in self.vocab_icd:
            icd_normalized = normalize_icd_for_graph(icd)
            if icd_normalized and icd_normalized in self.G.nodes():
                vocab_to_graph_icd[icd] = icd_normalized
        
        # Renumber ICD nodes
        for icd in self.vocab_icd:
            if icd in vocab_to_graph_icd:
                graph_node = vocab_to_graph_icd[icd]
                if graph_node not in renumber:
                    renumber[graph_node] = new_index
                    new_index += 1
        
        # CPT
        for cpt in self.vocab_cpt:
            if cpt in self.G.nodes():
                renumber[cpt] = new_index
                new_index += 1
        
        # ATC
        for atc in self.vocab_atc:
            if atc in self.G.nodes():
                renumber[atc] = new_index
                new_index += 1
        
        # Lab
        for lab in self.vocab_lab:
            if lab in self.G.nodes():
                renumber[lab] = new_index
                new_index += 1
        
        # Add remaining nodes (hierarchical nodes, roots, etc.)
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
        
        print(f"Renumbered: {len(renumber)} nodes")
        print(f"  Vocab nodes: {len(self.vocab_icd) + len(self.vocab_cpt) + len(self.vocab_atc) + len(self.vocab_lab)}")
        print(f"  Other nodes: {len(remaining_nodes)}")
        
        return G_renumbered, embeddings_renumbered, graphnode_vocab
    
    def save_outputs(self, G_renumbered, embeddings, graphnode_vocab):
        """Save graph and embeddings"""
        print("Saving outputs...")
        
        # Determine filename prefix
        prefix = 'augmented_' if self.augmented else ''
        graph_filename = os.path.join(
            self.output_dir,
            f'{prefix}icdatc_graph_{self.window}_{self.walk_length}_{self.num_walks}_{self.embedding_dim}_renumbered_by_vocab.pkl'
        )
        embed_filename = os.path.join(
            self.output_dir,
            f'{prefix}icdatc_embed_{self.window}_{self.walk_length}_{self.num_walks}_{self.embedding_dim}_by_vocab.pkl'
        )
        vocab_filename = os.path.join(self.output_dir, 'graphnode_vocab.pkl')
        
        # Save graph
        with open(graph_filename, 'wb') as f:
            pickle.dump(G_renumbered, f)
        print(f"  Graph saved: {graph_filename}")
        
        # Save embeddings
        with open(embed_filename, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"  Embeddings saved: {embed_filename}")
        
        # Save vocab mapping
        with open(vocab_filename, 'wb') as f:
            pickle.dump(graphnode_vocab, f)
        print(f"  Vocab mapping saved: {vocab_filename}")
        
        # Save vocabularies
        vocab_info = {
            'icd': self.vocab_icd,
            'cpt': self.vocab_cpt,
            'atc': self.vocab_atc,
            'lab': self.vocab_lab
        }
        vocab_info_filename = os.path.join(self.output_dir, 'vocab_info.pkl')
        with open(vocab_info_filename, 'wb') as f:
            pickle.dump(vocab_info, f)
        print(f"  Vocab info saved: {vocab_info_filename}")
        
        return graph_filename, embed_filename
    
    def build(self):
        """Main build function"""
        print("=" * 80)
        print("Building Knowledge Graph from MIMIC-III")
        print("=" * 80)
        
        # Step 1: Load tables
        self.load_mimic_tables()
        
        # Step 2: Build hierarchies
        self.build_icd9_hierarchy()
        self.build_cpt_hierarchy()
        self.extract_drugs_and_map_to_atc()
        self.extract_lab_codes()
        
        # Log chi tiết về edge types
        edge_types = {}
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print(f"\nGraph after hierarchies: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  Edge types: {edge_types}")
        
        # Step 3: Build co-occurrence edges
        self.build_cooccurrence_edges(min_cooccurrence=1)
        
        # Log chi tiết về edge types sau co-occurrence
        edge_types = {}
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print(f"\nGraph after co-occurrence: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  Edge types: {edge_types}")
        
        # Step 3.5: Create ego-graph from selected nodes (SIMPLE MODE)
        self.create_ego_graph()
        
        # Log chi tiết về edge types sau ego-graph
        edge_types = {}
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print(f"\nGraph after ego-graph creation: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  Edge types: {edge_types}")
        
        # Step 4: Augment graph
        self.augment_graph()
        
        # Log chi tiết về edge types sau augmentation
        edge_types = {}
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print(f"\nGraph after augmentation: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  Edge types: {edge_types}")
        
        # Step 5: Generate embeddings
        embeddings = self.generate_embeddings()
        
        # Step 6: Renumber nodes
        G_renumbered, embeddings_renumbered, graphnode_vocab = self.renumber_nodes_by_vocab(embeddings)
        
        # Step 7: Save outputs
        graph_file, embed_file = self.save_outputs(G_renumbered, embeddings_renumbered, graphnode_vocab)
        
        print("\n" + "=" * 80)
        print("Knowledge Graph built successfully!")
        print("=" * 80)
        print(f"Final graph: {G_renumbered.number_of_nodes()} nodes, {G_renumbered.number_of_edges()} edges")
        print(f"Embeddings shape: {embeddings_renumbered.shape}")
        print(f"\nOutput files:")
        print(f"  Graph: {graph_file}")
        print(f"  Embeddings: {embed_file}")
        
        return G_renumbered, embeddings_renumbered


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build KG from MIMIC-III')
    parser.add_argument('--mimic_path', type=str, 
                       default='mimic-iii-clinical-database-demo-1.4',
                       help='Path to MIMIC-III data directory')
    parser.add_argument('--output_dir', type=str, default='embed',
                       help='Output directory for saved files')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--window', type=int, default=8,
                       help='Node2Vec window size')
    parser.add_argument('--walk_length', type=int, default=20,
                       help='Node2Vec walk length')
    parser.add_argument('--num_walks', type=int, default=10,
                       help='Node2Vec number of walks')
    parser.add_argument('--augmented', action='store_true', default=True,
                       help='Add augmented edges')
    parser.add_argument('--no_augmented', dest='augmented', action='store_false',
                       help='Do not add augmented edges')
    
    args = parser.parse_args()
    
    # Create builder and build
    builder = MIMIC_KG_Builder(
        mimic_path=args.mimic_path,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        window=args.window,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        augmented=args.augmented
    )
    
    builder.build()


if __name__ == '__main__':
    main()

