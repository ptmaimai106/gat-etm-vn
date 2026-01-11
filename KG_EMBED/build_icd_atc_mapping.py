#!/usr/bin/env python3
"""
Build ICD-ATC Mapping from MIMIC-III Data

This script creates ICD-ATC relations based on co-occurrence in MIMIC-III admissions.
While the paper uses external knowledge base (MIA), this script creates an approximation
based on actual EHR co-occurrence patterns.

Output format: CSV file with columns:
  ICD_CODE,ATC_CODE,RELATION,FREQUENCY

Where:
  - ICD_CODE: ICD-9 diagnosis code (normalized, no dots)
  - ATC_CODE: ATC drug code
  - RELATION: 'treats' (default, based on co-occurrence)
  - FREQUENCY: Number of times this pair appeared together
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings('ignore')


class ICD_ATC_Mapping_Builder:
    def __init__(self, mimic_path, output_file='data/icd_atc_mapping.csv', 
                 min_frequency=1, max_relations=None):
        """
        Initialize mapping builder
        
        Args:
            mimic_path: Path to MIMIC-III data directory
            output_file: Output CSV file path
            min_frequency: Minimum co-occurrence frequency to include
            max_relations: Maximum number of relations to output (None = all)
        """
        self.mimic_path = mimic_path
        self.output_file = output_file
        self.min_frequency = min_frequency
        self.max_relations = max_relations
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Data storage
        self.diagnoses_df = None
        self.prescriptions_df = None
        self.drug_to_atc = {}
        self.icd_atc_pairs = Counter()
        
    def load_mimic_tables(self):
        """Load necessary MIMIC-III tables"""
        print("\n" + "=" * 80)
        print("Step 1: Loading MIMIC-III Tables")
        print("=" * 80)
        
        # Load DIAGNOSES_ICD
        diagnoses_file = os.path.join(self.mimic_path, 'DIAGNOSES_ICD.csv')
        if not os.path.exists(diagnoses_file):
            raise FileNotFoundError(f"DIAGNOSES_ICD.csv not found at {diagnoses_file}")
        
        print("  Loading DIAGNOSES_ICD.csv...")
        self.diagnoses_df = pd.read_csv(diagnoses_file)
        print(f"    Loaded {len(self.diagnoses_df)} diagnosis records")
        
        # Load PRESCRIPTIONS
        prescriptions_file = os.path.join(self.mimic_path, 'PRESCRIPTIONS.csv')
        if not os.path.exists(prescriptions_file):
            raise FileNotFoundError(f"PRESCRIPTIONS.csv not found at {prescriptions_file}")
        
        print("  Loading PRESCRIPTIONS.csv...")
        # Read in chunks if file is too large
        try:
            self.prescriptions_df = pd.read_csv(prescriptions_file)
            print(f"    Loaded {len(self.prescriptions_df)} prescription records")
        except MemoryError:
            print("    File too large, reading in chunks...")
            chunks = []
            chunk_size = 100000
            for chunk in pd.read_csv(prescriptions_file, chunksize=chunk_size):
                chunks.append(chunk)
            self.prescriptions_df = pd.concat(chunks, ignore_index=True)
            print(f"    Loaded {len(self.prescriptions_df)} prescription records")
        
    def create_drug_to_atc_mapping(self):
        """
        Create mapping from drug names to ATC codes
        
        For now, uses hash-based placeholder (similar to build_kg_mimic.py)
        In production, should use RxNorm API → WHO ATC mapping
        """
        print("\n" + "=" * 80)
        print("Step 2: Creating Drug to ATC Mapping")
        print("=" * 80)
        
        # Get unique drug names
        drug_column = None
        for col in ['drug_name_generic', 'drug', 'drug_name', 'DRUG']:
            if col in self.prescriptions_df.columns:
                drug_column = col
                break
        
        if drug_column is None:
            print("  WARNING: No drug name column found. Available columns:")
            print(f"    {list(self.prescriptions_df.columns)}")
            # Try to use first text column
            for col in self.prescriptions_df.columns:
                if self.prescriptions_df[col].dtype == 'object':
                    drug_column = col
                    print(f"  Using column: {drug_column}")
                    break
        
        if drug_column is None:
            raise ValueError("Cannot find drug name column in PRESCRIPTIONS.csv")
        
        unique_drugs = self.prescriptions_df[drug_column].dropna().unique()
        print(f"  Found {len(unique_drugs)} unique drugs")
        
        # Create ATC codes from drug names (hash-based placeholder)
        # In production, use RxNorm API → WHO ATC mapping
        print("  Creating ATC codes (hash-based placeholder)...")
        print("  NOTE: For production, use RxNorm API → WHO ATC mapping")
        
        for drug in tqdm(unique_drugs, desc="  Processing drugs"):
            if pd.notna(drug):
                drug_str = str(drug).strip().upper()
                if drug_str:
                    # Create ATC-like code from hash
                    # Format: DRUG_XXXXX (similar to build_kg_mimic.py)
                    atc_code = f"DRUG_{abs(hash(drug_str)) % 100000:05d}"
                    self.drug_to_atc[drug_str] = atc_code
        
        print(f"  Created {len(self.drug_to_atc)} drug-to-ATC mappings")
        return self.drug_to_atc
    
    def normalize_icd_code(self, icd_code):
        """
        Normalize ICD code: remove dots for internal use
        
        Args:
            icd_code: ICD code (can be string with or without dots)
        
        Returns:
            Normalized ICD code (no dots)
        """
        if pd.isna(icd_code):
            return None
        code_str = str(icd_code).strip()
        # Remove dots
        normalized = code_str.replace('.', '')
        return normalized if normalized else None
    
    def extract_cooccurrence_pairs(self):
        """
        Extract ICD-ATC co-occurrence pairs from admissions
        
        For each admission (HADM_ID), find all ICD codes and ATC codes,
        then create pairs for all combinations.
        """
        print("\n" + "=" * 80)
        print("Step 3: Extracting ICD-ATC Co-occurrence Pairs")
        print("=" * 80)
        
        # Group ICD codes by HADM_ID
        print("  Grouping ICD codes by admission...")
        hadm_to_icd = defaultdict(set)
        
        icd_column = None
        for col in ['ICD9_CODE', 'icd9_code', 'ICD_CODE', 'icd_code']:
            if col in self.diagnoses_df.columns:
                icd_column = col
                break
        
        if icd_column is None:
            raise ValueError("Cannot find ICD code column in DIAGNOSES_ICD.csv")
        
        hadm_column = None
        for col in ['HADM_ID', 'hadm_id', 'HADMID', 'hadmid']:
            if col in self.diagnoses_df.columns:
                hadm_column = col
                break
        
        if hadm_column is None:
            raise ValueError("Cannot find HADM_ID column in DIAGNOSES_ICD.csv")
        
        for _, row in tqdm(self.diagnoses_df.iterrows(), desc="  Processing diagnoses", 
                          total=len(self.diagnoses_df)):
            hadm_id = row[hadm_column]
            icd_code = row[icd_column]
            
            if pd.notna(icd_code) and pd.notna(hadm_id):
                icd_normalized = self.normalize_icd_code(icd_code)
                if icd_normalized:
                    hadm_to_icd[hadm_id].add(icd_normalized)
        
        print(f"    Found {len(hadm_to_icd)} admissions with ICD codes")
        
        # Group ATC codes by HADM_ID
        print("  Grouping ATC codes by admission...")
        hadm_to_atc = defaultdict(set)
        
        drug_column = None
        for col in ['drug_name_generic', 'drug', 'drug_name', 'DRUG']:
            if col in self.prescriptions_df.columns:
                drug_column = col
                break
        
        hadm_column_pres = None
        for col in ['HADM_ID', 'hadm_id', 'HADMID', 'hadmid']:
            if col in self.prescriptions_df.columns:
                hadm_column_pres = col
                break
        
        if hadm_column_pres is None:
            raise ValueError("Cannot find HADM_ID column in PRESCRIPTIONS.csv")
        
        for _, row in tqdm(self.prescriptions_df.iterrows(), desc="  Processing prescriptions",
                          total=len(self.prescriptions_df)):
            hadm_id = row[hadm_column_pres]
            drug = row[drug_column] if drug_column else None
            
            if pd.notna(drug) and pd.notna(hadm_id):
                drug_str = str(drug).strip().upper()
                if drug_str in self.drug_to_atc:
                    atc_code = self.drug_to_atc[drug_str]
                    hadm_to_atc[hadm_id].add(atc_code)
        
        print(f"    Found {len(hadm_to_atc)} admissions with ATC codes")
        
        # Find common admissions
        common_admissions = set(hadm_to_icd.keys()) & set(hadm_to_atc.keys())
        print(f"    Found {len(common_admissions)} admissions with both ICD and ATC codes")
        
        # Create co-occurrence pairs
        print("  Creating ICD-ATC pairs...")
        for hadm_id in tqdm(common_admissions, desc="  Processing admissions"):
            icd_codes = hadm_to_icd[hadm_id]
            atc_codes = hadm_to_atc[hadm_id]
            
            # Create all pairs
            for icd in icd_codes:
                for atc in atc_codes:
                    pair = (icd, atc)
                    self.icd_atc_pairs[pair] += 1
        
        print(f"    Created {len(self.icd_atc_pairs)} unique ICD-ATC pairs")
        print(f"    Total co-occurrences: {sum(self.icd_atc_pairs.values())}")
        
        return self.icd_atc_pairs
    
    def filter_and_sort_pairs(self):
        """
        Filter pairs by minimum frequency and optionally limit number
        """
        print("\n" + "=" * 80)
        print("Step 4: Filtering and Sorting Pairs")
        print("=" * 80)
        
        # Filter by minimum frequency
        filtered_pairs = {
            pair: freq for pair, freq in self.icd_atc_pairs.items() 
            if freq >= self.min_frequency
        }
        
        print(f"  Pairs with frequency >= {self.min_frequency}: {len(filtered_pairs)}")
        
        # Sort by frequency (descending)
        sorted_pairs = sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # Limit number if specified
        if self.max_relations and len(sorted_pairs) > self.max_relations:
            print(f"  Limiting to top {self.max_relations} pairs")
            sorted_pairs = sorted_pairs[:self.max_relations]
        
        print(f"  Final number of pairs: {len(sorted_pairs)}")
        
        return sorted_pairs
    
    def save_to_csv(self, pairs):
        """
        Save ICD-ATC mapping to CSV file
        
        Format: ICD_CODE,ATC_CODE,RELATION,FREQUENCY
        """
        print("\n" + "=" * 80)
        print("Step 5: Saving to CSV")
        print("=" * 80)
        
        # Create DataFrame
        data = []
        for (icd_code, atc_code), frequency in pairs:
            data.append({
                'ICD_CODE': icd_code,
                'ATC_CODE': atc_code,
                'RELATION': 'treats',  # Default relation type
                'FREQUENCY': frequency
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        print(f"  Saved {len(df)} relations to {self.output_file}")
        
        # Print statistics
        print(f"\n  Statistics:")
        print(f"    Total relations: {len(df)}")
        print(f"    Unique ICD codes: {df['ICD_CODE'].nunique()}")
        print(f"    Unique ATC codes: {df['ATC_CODE'].nunique()}")
        print(f"    Average frequency: {df['FREQUENCY'].mean():.2f}")
        print(f"    Median frequency: {df['FREQUENCY'].median():.2f}")
        print(f"    Max frequency: {df['FREQUENCY'].max()}")
        print(f"    Min frequency: {df['FREQUENCY'].min()}")
        
        # Show top 10 most frequent pairs
        print(f"\n  Top 10 most frequent pairs:")
        for i, row in df.head(10).iterrows():
            print(f"    {i+1}. {row['ICD_CODE']} <-> {row['ATC_CODE']} (freq: {row['FREQUENCY']})")
        
        return self.output_file
    
    def build(self):
        """Main build function"""
        print("=" * 80)
        print("Building ICD-ATC Mapping from MIMIC-III")
        print("=" * 80)
        
        # Step 1: Load tables
        self.load_mimic_tables()
        
        # Step 2: Create drug-to-ATC mapping
        self.create_drug_to_atc_mapping()
        
        # Step 3: Extract co-occurrence pairs
        self.extract_cooccurrence_pairs()
        
        # Step 4: Filter and sort
        filtered_pairs = self.filter_and_sort_pairs()
        
        # Step 5: Save to CSV
        output_file = self.save_to_csv(filtered_pairs)
        
        print("\n" + "=" * 80)
        print("Build complete!")
        print("=" * 80)
        print(f"Output file: {output_file}")
        print("\nNote: This mapping is based on co-occurrence in MIMIC-III.")
        print("For production, consider using external knowledge base:")
        print("  - MIA: http://hulab.rxnfinder.org/mia/")
        print("  - RxNorm API → WHO ATC mapping")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Build ICD-ATC Mapping from MIMIC-III Data'
    )
    parser.add_argument('--mimic_path', type=str,
                       default='mimic-iii-clinical-database-demo-1.4',
                       help='Path to MIMIC-III data directory')
    parser.add_argument('--output', type=str,
                       default='data/icd_atc_mapping.csv',
                       help='Output CSV file path')
    parser.add_argument('--min_frequency', type=int, default=1,
                       help='Minimum co-occurrence frequency to include (default: 1)')
    parser.add_argument('--max_relations', type=int, default=None,
                       help='Maximum number of relations to output (default: all)')
    
    args = parser.parse_args()
    
    # Create builder
    builder = ICD_ATC_Mapping_Builder(
        mimic_path=args.mimic_path,
        output_file=args.output,
        min_frequency=args.min_frequency,
        max_relations=args.max_relations
    )
    
    # Build mapping
    output_file = builder.build()
    
    print(f"\n✓ Mapping saved to: {output_file}")
    print(f"\nTo use in build_kg_paper.py:")
    print(f"  python KG_EMBED/build_kg_paper.py --icd_atc_mapping {output_file}")


if __name__ == '__main__':
    main()
