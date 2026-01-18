#!/usr/bin/env python3
"""
Script to prepare MIMIC-III data for GAT-ETM training

This script:
1. Loads vocab info from KG embeddings (must be built first)
2. Extracts ICD and ATC codes from MIMIC-III admissions
3. Creates BoW matrices (train/test/test_1/test_2)
4. Creates metadata.txt file

Prerequisites:
- KG embeddings must be built first using build_kg_mimic.py
- MIMIC-III data must be available in mimic_path

Usage:
    python prepare_mimic_data.py --mimic_path mimic-iii-clinical-database-demo-1.4 \
                                  --kg_embed_dir KG_EMBED/embed_augmented \
                                  --output_dir data \
                                  --train_ratio 0.8
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings('ignore')


class MIMIC_Data_Preparer:
    def __init__(self, mimic_path, kg_embed_dir, output_dir='data',
                 train_ratio=0.8, random_seed=42, binary_bow=True):
        """
        Initialize data preparer
        
        Args:
            mimic_path: Path to MIMIC-III data directory
            kg_embed_dir: Directory containing KG embeddings (vocab_info.pkl)
            output_dir: Output directory for BoW files and metadata
            train_ratio: Ratio of training data (default: 0.8)
            random_seed: Random seed for train/test split
            binary_bow: If True, use binary encoding (0/1), else use frequency
        """
        self.mimic_path = mimic_path
        self.kg_embed_dir = kg_embed_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.binary_bow = binary_bow
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.vocab_info = None
        self.drug_to_atc = {}
        self.patient_codes = {}
        
    def load_kg_vocab(self):
        """Load vocabulary info from KG embeddings"""
        print("\n" + "=" * 80)
        print("Step 1: Loading Vocabulary from KG Embeddings")
        print("=" * 80)
        
        vocab_info_path = os.path.join(self.kg_embed_dir, 'vocab_info.pkl')
        if not os.path.exists(vocab_info_path):
            raise FileNotFoundError(
                f"Vocab info not found at {vocab_info_path}.\n"
                f"Please run KG builder first:\n"
                f"  python KG_EMBED/build_kg_mimic.py --mimic_path {self.mimic_path} --output_dir {self.kg_embed_dir}"
            )
        
        with open(vocab_info_path, 'rb') as f:
            self.vocab_info = pickle.load(f)
        
        print(f"Loaded vocabularies:")
        for code_type, vocab_list in self.vocab_info.items():
            if len(vocab_list) > 0:
                print(f"  {code_type.upper()}: {len(vocab_list)} codes")
        
        # Filter to only ICD and ATC (as expected by model)
        available_types = [ct for ct in ['icd', 'atc'] if ct in self.vocab_info and len(self.vocab_info[ct]) > 0]
        if len(available_types) == 0:
            raise ValueError("No ICD or ATC vocabularies found in KG embeddings")
        
        print(f"\nUsing code types: {available_types}")
        return self.vocab_info
    
    def load_drug_to_atc_mapping(self):
        """
        Load or create drug-to-ATC mapping
        
        This should match the mapping used in build_kg_mimic.py
        """
        print("\n" + "=" * 80)
        print("Step 2: Loading Drug-to-ATC Mapping")
        print("=" * 80)
        
        # Try to load from KG builder if saved
        mapping_file = os.path.join(self.kg_embed_dir, 'drug_to_atc.pkl')
        if os.path.exists(mapping_file):
            print(f"  Loading from {mapping_file}...")
            with open(mapping_file, 'rb') as f:
                self.drug_to_atc = pickle.load(f)
            print(f"  Loaded {len(self.drug_to_atc)} drug-to-ATC mappings")
        else:
            # Create mapping from prescriptions (same logic as build_kg_mimic.py)
            print("  Creating drug-to-ATC mapping from prescriptions...")
            prescriptions_file = os.path.join(self.mimic_path, 'PRESCRIPTIONS.csv')
            
            if not os.path.exists(prescriptions_file):
                print(f"  WARNING: {prescriptions_file} not found")
                return {}
            
            # Load prescriptions
            print("  Loading PRESCRIPTIONS.csv...")
            try:
                prescriptions_df = pd.read_csv(prescriptions_file, nrows=100000)  # Sample for speed
            except Exception as e:
                print(f"  Error loading prescriptions: {e}")
                return {}
            
            # Find drug column
            drug_column = None
            for col in ['drug_name_generic', 'drug', 'drug_name', 'DRUG']:
                if col in prescriptions_df.columns:
                    drug_column = col
                    break
            
            if drug_column is None:
                print("  WARNING: No drug name column found")
                return {}
            
            # Create hash-based ATC codes (same as build_kg_mimic.py)
            unique_drugs = prescriptions_df[drug_column].dropna().unique()
            print(f"  Found {len(unique_drugs)} unique drugs")
            
            for drug in tqdm(unique_drugs, desc="  Creating mappings"):
                if pd.notna(drug):
                    drug_str = str(drug).strip().upper()
                    if drug_str:
                        atc_code = f"DRUG_{abs(hash(drug_str)) % 100000:05d}"
                        self.drug_to_atc[drug_str] = atc_code
            
            print(f"  Created {len(self.drug_to_atc)} mappings")
            print("  NOTE: Using hash-based placeholder ATC codes")
            print("        For production, use RxNorm API → WHO ATC mapping")
        
        return self.drug_to_atc
    
    def normalize_icd_code(self, icd_code, keep_format=False):
        """
        Normalize ICD code
        
        Args:
            icd_code: ICD code (can be string with or without dots)
            keep_format: If True, keep original format (with dots if present)
        
        Returns:
            Normalized ICD code
        """
        if pd.isna(icd_code):
            return None
        code_str = str(icd_code).strip()
        if keep_format:
            return code_str
        else:
            # Remove dots for matching
            return code_str.replace('.', '')
    
    def extract_codes_from_mimic(self):
        """
        Extract ICD and ATC codes from MIMIC-III for each patient/admission
        """
        print("\n" + "=" * 80)
        print("Step 3: Extracting Codes from MIMIC-III")
        print("=" * 80)
        
        # Load necessary tables
        print("  Loading MIMIC-III tables...")
        
        # Load DIAGNOSES_ICD
        diagnoses_file = os.path.join(self.mimic_path, 'DIAGNOSES_ICD.csv')
        if not os.path.exists(diagnoses_file):
            raise FileNotFoundError(f"DIAGNOSES_ICD.csv not found at {diagnoses_file}")
        
        diagnoses_df = pd.read_csv(diagnoses_file)
        print(f"    Loaded {len(diagnoses_df)} diagnosis records")
        
        # Load PRESCRIPTIONS
        prescriptions_file = os.path.join(self.mimic_path, 'PRESCRIPTIONS.csv')
        if not os.path.exists(prescriptions_file):
            raise FileNotFoundError(f"PRESCRIPTIONS.csv not found at {prescriptions_file}")
        
        print("    Loading PRESCRIPTIONS.csv (this may take a while)...")
        try:
            prescriptions_df = pd.read_csv(prescriptions_file)
        except MemoryError:
            print("    File too large, reading in chunks...")
            chunks = []
            chunk_size = 100000
            for chunk in pd.read_csv(prescriptions_file, chunksize=chunk_size):
                chunks.append(chunk)
            prescriptions_df = pd.concat(chunks, ignore_index=True)
        print(f"    Loaded {len(prescriptions_df)} prescription records")
        
        # Create vocab sets for fast lookup
        vocab_icd_set = set(self.vocab_info.get('icd', []))
        vocab_atc_set = set(self.vocab_info.get('atc', []))
        
        # Create bidirectional mapping for ICD code matching
        # Handle both formats: with dots (401.9) and without (4019)
        vocab_icd_normalized = {}
        vocab_icd_to_normalized = {}  # Original -> normalized
        
        for icd in vocab_icd_set:
            normalized = self.normalize_icd_code(icd, keep_format=False)
            # Map normalized -> original (keep first occurrence if duplicates)
            if normalized not in vocab_icd_normalized:
                vocab_icd_normalized[normalized] = icd
            vocab_icd_to_normalized[icd] = normalized
        
        print(f"    ICD vocab size: {len(vocab_icd_set)}")
        print(f"    ATC vocab size: {len(vocab_atc_set)}")
        
        # Group by HADM_ID (hospital admission)
        print("  Grouping codes by admission...")
        
        # Extract ICD codes
        hadm_to_icd = defaultdict(set)
        icd_column = None
        for col in ['ICD9_CODE', 'icd9_code', 'ICD_CODE', 'icd_code']:
            if col in diagnoses_df.columns:
                icd_column = col
                break
        
        hadm_column = None
        for col in ['HADM_ID', 'hadm_id', 'HADMID', 'hadmid']:
            if col in diagnoses_df.columns:
                hadm_column = col
                break
        
        if icd_column is None or hadm_column is None:
            raise ValueError("Cannot find ICD_CODE or HADM_ID columns in DIAGNOSES_ICD.csv")
        
        print("    Extracting ICD codes...")
        matched_count = 0
        unmatched_count = 0
        
        for _, row in tqdm(diagnoses_df.iterrows(), total=len(diagnoses_df), desc="    Processing diagnoses"):
            hadm_id = row[hadm_column]
            icd_code = row[icd_column]
            
            if pd.notna(icd_code) and pd.notna(hadm_id):
                # Try both original and normalized formats
                icd_orig = str(icd_code).strip()
                icd_normalized = self.normalize_icd_code(icd_code, keep_format=False)
                
                matched = False
                # Check if in vocab (try both formats)
                if icd_orig in vocab_icd_set:
                    hadm_to_icd[hadm_id].add(icd_orig)
                    matched = True
                elif icd_normalized in vocab_icd_normalized:
                    # Use original format from vocab
                    vocab_code = vocab_icd_normalized[icd_normalized]
                    hadm_to_icd[hadm_id].add(vocab_code)
                    matched = True
                
                if matched:
                    matched_count += 1
                else:
                    unmatched_count += 1
        
        print(f"    Matched ICD codes: {matched_count}")
        if unmatched_count > 0:
            print(f"    Unmatched ICD codes: {unmatched_count} (these will be skipped)")
        
        print(f"    Found {len(hadm_to_icd)} admissions with ICD codes")
        
        # Extract ATC codes from prescriptions
        hadm_to_atc = defaultdict(set)
        drug_column = None
        for col in ['drug_name_generic', 'drug', 'drug_name', 'DRUG']:
            if col in prescriptions_df.columns:
                drug_column = col
                break
        
        hadm_column_pres = None
        for col in ['HADM_ID', 'hadm_id', 'HADMID', 'hadmid']:
            if col in prescriptions_df.columns:
                hadm_column_pres = col
                break
        
        if drug_column is None or hadm_column_pres is None:
            print("    WARNING: Cannot find drug or HADM_ID columns in PRESCRIPTIONS.csv")
        else:
            print("    Extracting ATC codes...")
            for _, row in tqdm(prescriptions_df.iterrows(), total=len(prescriptions_df), desc="    Processing prescriptions"):
                hadm_id = row[hadm_column_pres]
                drug = row[drug_column]
                
                if pd.notna(drug) and pd.notna(hadm_id):
                    drug_str = str(drug).strip().upper()
                    if drug_str in self.drug_to_atc:
                        atc_code = self.drug_to_atc[drug_str]
                        if atc_code in vocab_atc_set:
                            hadm_to_atc[hadm_id].add(atc_code)
            
            print(f"    Found {len(hadm_to_atc)} admissions with ATC codes")
        
        # Combine into patient_codes dictionary
        # Use HADM_ID as patient identifier (each admission = one document)
        print("  Creating patient code dictionaries...")
        all_hadm_ids = set(list(hadm_to_icd.keys()) + list(hadm_to_atc.keys()))
        
        for hadm_id in tqdm(all_hadm_ids, desc="    Processing admissions"):
            self.patient_codes[hadm_id] = {
                'icd': list(hadm_to_icd.get(hadm_id, set())),
                'atc': list(hadm_to_atc.get(hadm_id, set()))
            }
        
        print(f"  Total admissions processed: {len(self.patient_codes)}")
        
        # Print statistics
        total_icd = sum(len(codes['icd']) for codes in self.patient_codes.values())
        total_atc = sum(len(codes['atc']) for codes in self.patient_codes.values())
        print(f"  Total ICD codes: {total_icd}")
        print(f"  Total ATC codes: {total_atc}")
        print(f"  Average ICD codes per admission: {total_icd / len(self.patient_codes):.2f}")
        print(f"  Average ATC codes per admission: {total_atc / len(self.patient_codes):.2f}")
        
        return self.patient_codes
    
    def create_code_to_idx_mapping(self):
        """
        Create mapping from (code_type, code) to column index in BoW matrix
        """
        code_to_idx = {}
        vocab_cum = [0]
        
        # Get code types in order (ICD first, then ATC)
        code_types = []
        if 'icd' in self.vocab_info and len(self.vocab_info['icd']) > 0:
            code_types.append('icd')
        if 'atc' in self.vocab_info and len(self.vocab_info['atc']) > 0:
            code_types.append('atc')
        
        for code_type in code_types:
            vocab_list = sorted(self.vocab_info[code_type])
            vocab_size = len(vocab_list)
            
            for idx, code in enumerate(vocab_list):
                code_to_idx[(code_type, code)] = vocab_cum[-1] + idx
            
            vocab_cum.append(vocab_cum[-1] + vocab_size)
        
        return code_to_idx, vocab_cum, code_types
    
    def create_bow_matrix(self, code_to_idx, vocab_cum):
        """
        Create BoW matrix from patient codes
        """
        print("\n" + "=" * 80)
        print("Step 4: Creating BoW Matrix")
        print("=" * 80)
        
        num_patients = len(self.patient_codes)
        total_vocab_size = vocab_cum[-1]
        
        print(f"  Number of patients/admissions: {num_patients}")
        print(f"  Total vocabulary size: {total_vocab_size}")
        
        rows = []
        cols = []
        data = []
        
        print("  Building sparse matrix...")
        for patient_idx, (hadm_id, codes_dict) in enumerate(tqdm(self.patient_codes.items(), desc="  Processing patients")):
            code_counts = Counter()
            
            for code_type in ['icd', 'atc']:
                if code_type in codes_dict:
                    for code in codes_dict[code_type]:
                        key = (code_type, code)
                        if key in code_to_idx:
                            if self.binary_bow:
                                code_counts[code_to_idx[key]] = 1  # Binary encoding
                            else:
                                code_counts[code_to_idx[key]] += 1  # Frequency encoding
            
            for col_idx, count in code_counts.items():
                rows.append(patient_idx)
                cols.append(col_idx)
                data.append(count)
        
        bow_matrix = csr_matrix((data, (rows, cols)),
                                shape=(num_patients, total_vocab_size))
        
        print(f"  BoW matrix shape: {bow_matrix.shape}")
        print(f"  Non-zero entries: {bow_matrix.nnz}")
        print(f"  Sparsity: {(1 - bow_matrix.nnz / (num_patients * total_vocab_size)) * 100:.2f}%")
        
        return bow_matrix
    
    def split_train_test(self, bow_matrix):
        """
        Split BoW matrix into train and test sets
        Also create test_1 and test_2 for document completion task
        """
        print("\n" + "=" * 80)
        print("Step 5: Splitting Train/Test Data")
        print("=" * 80)
        
        np.random.seed(self.random_seed)
        num_patients = bow_matrix.shape[0]
        indices = np.random.permutation(num_patients)
        
        train_size = int(num_patients * self.train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        bow_train = bow_matrix[train_indices]
        bow_test = bow_matrix[test_indices]
        
        # Split test into two halves for document completion
        test_size = len(test_indices)
        test_1_indices = test_indices[:test_size // 2]
        test_2_indices = test_indices[test_size // 2:]
        
        bow_test_1 = bow_matrix[test_1_indices]
        bow_test_2 = bow_matrix[test_2_indices]
        
        print(f"  Train size: {bow_train.shape[0]}")
        print(f"  Test size: {bow_test.shape[0]}")
        print(f"  Test_1 size: {bow_test_1.shape[0]}")
        print(f"  Test_2 size: {bow_test_2.shape[0]}")
        
        return bow_train, bow_test, bow_test_1, bow_test_2
    
    def save_bow_files(self, bow_train, bow_test, bow_test_1, bow_test_2):
        """Save BoW matrices to .npy files"""
        print("\n" + "=" * 80)
        print("Step 6: Saving BoW Files")
        print("=" * 80)
        
        files = {
            'bow_train.npy': bow_train,
            'bow_test.npy': bow_test,
            'bow_test_1.npy': bow_test_1,
            'bow_test_2.npy': bow_test_2
        }
        
        for filename, matrix in files.items():
            filepath = os.path.join(self.output_dir, filename)
            np.save(filepath, matrix, allow_pickle=True)
            print(f"  Saved: {filepath} (shape: {matrix.shape}, nnz: {matrix.nnz})")
    
    def create_metadata_file(self, code_types, vocab_sizes):
        """Create metadata.txt file"""
        print("\n" + "=" * 80)
        print("Step 7: Creating Metadata File")
        print("=" * 80)
        
        metadata_path = os.path.join(self.output_dir, 'metadata.txt')
        
        # Default values
        train_embeddings = [1] * len(code_types)  # Train embeddings from graph
        embedding_files = ['*'] * len(code_types)  # Use KG embeddings
        
        with open(metadata_path, 'w') as f:
            f.write(str(code_types) + '\n')
            f.write(str(vocab_sizes) + '\n')
            f.write(str(train_embeddings) + '\n')
            f.write(str(embedding_files) + '\n')
        
        print(f"  Saved: {metadata_path}")
        print(f"  Content:")
        print(f"    Code types: {code_types}")
        print(f"    Vocab sizes: {vocab_sizes}")
        print(f"    Train embeddings: {train_embeddings}")
        print(f"    Embedding files: {embedding_files}")
        
        return metadata_path
    
    def prepare(self):
        """Main preparation function"""
        print("=" * 80)
        print("Preparing MIMIC-III Data for GAT-ETM Training")
        print("=" * 80)
        
        # Step 1: Load vocab from KG
        self.load_kg_vocab()
        
        # Step 2: Load drug-to-ATC mapping
        self.load_drug_to_atc_mapping()
        
        # Step 3: Extract codes from MIMIC-III
        self.extract_codes_from_mimic()
        
        # Step 4: Create code-to-index mapping
        code_to_idx, vocab_cum, code_types = self.create_code_to_idx_mapping()
        vocab_sizes = [len(self.vocab_info[ct]) for ct in code_types]
        
        print(f"\nCode types: {code_types}")
        print(f"Vocab sizes: {vocab_sizes}")
        print(f"Vocab cumulative: {vocab_cum}")
        
        # Step 5: Create BoW matrix
        bow_matrix = self.create_bow_matrix(code_to_idx, vocab_cum)
        
        # Step 6: Split train/test
        bow_train, bow_test, bow_test_1, bow_test_2 = self.split_train_test(bow_matrix)
        
        # Step 7: Save BoW files
        self.save_bow_files(bow_train, bow_test, bow_test_1, bow_test_2)
        
        # Step 8: Create metadata file
        self.create_metadata_file(code_types, vocab_sizes)
        
        print("\n" + "=" * 80)
        print("Data Preparation Complete!")
        print("=" * 80)
        print(f"\nOutput files saved to: {self.output_dir}/")
        print("  - bow_train.npy")
        print("  - bow_test.npy")
        print("  - bow_test_1.npy")
        print("  - bow_test_2.npy")
        print("  - metadata.txt")
        print("\nNext steps:")
        print("  1. Verify metadata.txt matches your KG vocab sizes")
        print("  2. Run training:")
        print(f"     python main_getm_mimic.py --data_path {self.output_dir} --kg_embed_dir {self.kg_embed_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MIMIC-III data for GAT-ETM training'
    )
    parser.add_argument('--mimic_path', type=str,
                       default='mimic-iii-clinical-database-demo-1.4',
                       help='Path to MIMIC-III data directory')
    parser.add_argument('--kg_embed_dir', type=str,
                       default='KG_EMBED/embed_augmented',
                       help='Directory containing KG embeddings (vocab_info.pkl)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for BoW files and metadata')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for train/test split')
    parser.add_argument('--frequency_bow', action='store_true',
                       help='Use frequency encoding instead of binary')
    
    args = parser.parse_args()
    
    # Create preparer
    preparer = MIMIC_Data_Preparer(
        mimic_path=args.mimic_path,
        kg_embed_dir=args.kg_embed_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        binary_bow=not args.frequency_bow
    )
    
    # Prepare data
    preparer.prepare()


if __name__ == '__main__':
    main()
