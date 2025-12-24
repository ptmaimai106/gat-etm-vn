#!/usr/bin/env python3
"""
Template script để extract codes từ EHR Việt Nam
Adapt từ extract_codes_from_mimic() nhưng customize cho VN data

Cần customize các phần:
1. Data loading (tùy format của bạn)
2. Code extraction logic
3. Normalization cho tiếng Việt
4. Mapping strategies
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class VN_EHR_Extractor:
    def __init__(self, data_path, output_dir='data_vn'):
        """
        Initialize extractor
        
        Args:
            data_path: Path to VN EHR data (CSV, SQL, Excel, etc.)
            output_dir: Output directory
        """
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Patient codes: {patient_id: {code_type: [codes]}}
        self.patient_codes = defaultdict(lambda: defaultdict(list))
        
        # Code mappings (nếu có)
        self.drug_to_atc = {}  # Tên thuốc VN -> ATC code
        self.diagnosis_to_icd = {}  # Tên bệnh VN -> ICD code
        
    def load_data(self):
        """
        Load VN EHR data
        TODO: Customize theo format dữ liệu của bạn
        """
        print("Loading VN EHR data...")
        
        # Option 1: Load từ CSV files
        # self.diagnoses = pd.read_csv(os.path.join(self.data_path, 'diagnoses.csv'))
        # self.prescriptions = pd.read_csv(os.path.join(self.data_path, 'prescriptions.csv'))
        # self.labs = pd.read_csv(os.path.join(self.data_path, 'labs.csv'))
        
        # Option 2: Load từ SQL database
        # import sqlite3
        # conn = sqlite3.connect(os.path.join(self.data_path, 'ehr.db'))
        # self.diagnoses = pd.read_sql_query("SELECT * FROM diagnoses", conn)
        # ...
        
        # Option 3: Load từ Excel
        # self.diagnoses = pd.read_excel(os.path.join(self.data_path, 'diagnoses.xlsx'))
        # ...
        
        # TODO: Implement data loading
        raise NotImplementedError("Please implement load_data() based on your data format")
    
    def normalize_vietnamese_text(self, text):
        """
        Normalize Vietnamese text
        Options:
        1. Keep original (recommended)
        2. Remove accents
        3. Lowercase
        """
        if pd.isna(text):
            return None
        
        text = str(text).strip()
        
        # Option 1: Keep original
        return text
        
        # Option 2: Remove accents (uncomment if needed)
        # from unidecode import unidecode
        # return unidecode(text)
        
        # Option 3: Lowercase
        # return text.lower()
    
    def extract_diagnosis_codes(self):
        """
        Extract diagnosis codes từ VN EHR
        TODO: Customize theo cấu trúc dữ liệu của bạn
        """
        print("Extracting diagnosis codes...")
        
        # Example structure (adapt to your data):
        # for _, row in self.diagnoses.iterrows():
        #     patient_id = row['patient_id']  # or 'subject_id', etc.
        #     visit_id = row['visit_id']  # or 'hadm_id', etc.
        #     
        #     # Option 1: Có ICD code sẵn
        #     icd_code = row.get('icd_code', None)
        #     if pd.notna(icd_code):
        #         # Normalize code
        #         icd_code = str(icd_code).strip().replace('.', '')
        #         self.patient_codes[patient_id]['icd'].append(icd_code)
        #     
        #     # Option 2: Có tên bệnh tiếng Việt, cần map sang ICD
        #     diagnosis_name = row.get('diagnosis_name', None)
        #     if pd.notna(diagnosis_name):
        #         diagnosis_name = self.normalize_vietnamese_text(diagnosis_name)
        #         # Map tên bệnh -> ICD code
        #         if diagnosis_name in self.diagnosis_to_icd:
        #             icd_code = self.diagnosis_to_icd[diagnosis_name]
        #             self.patient_codes[patient_id]['icd'].append(icd_code)
        
        # TODO: Implement extraction logic
        raise NotImplementedError("Please implement extract_diagnosis_codes()")
    
    def extract_drug_codes(self):
        """
        Extract drug codes từ VN EHR
        TODO: Customize theo cấu trúc dữ liệu của bạn
        """
        print("Extracting drug codes...")
        
        # Example structure (adapt to your data):
        # for _, row in self.prescriptions.iterrows():
        #     patient_id = row['patient_id']
        #     visit_id = row['visit_id']
        #     
        #     # Option 1: Có ATC code sẵn
        #     atc_code = row.get('atc_code', None)
        #     if pd.notna(atc_code):
        #         self.patient_codes[patient_id]['atc'].append(atc_code)
        #     
        #     # Option 2: Có tên thuốc tiếng Việt, cần map sang ATC
        #     drug_name = row.get('drug_name', None)
        #     if pd.notna(drug_name):
        #         drug_name = self.normalize_vietnamese_text(drug_name)
        #         # Map tên thuốc -> ATC code
        #         if drug_name in self.drug_to_atc:
        #             atc_code = self.drug_to_atc[drug_name]
        #             self.patient_codes[patient_id]['atc'].append(atc_code)
        #     
        #     # Option 3: Có generic name, map sang ATC
        #     generic_name = row.get('generic_name', None)
        #     if pd.notna(generic_name):
        #         # Map generic -> ATC
        #         ...
        
        # TODO: Implement extraction logic
        raise NotImplementedError("Please implement extract_drug_codes()")
    
    def extract_lab_codes(self):
        """
        Extract lab codes từ VN EHR
        TODO: Customize theo cấu trúc dữ liệu của bạn
        """
        print("Extracting lab codes...")
        
        # Example structure (adapt to your data):
        # for _, row in self.labs.iterrows():
        #     patient_id = row['patient_id']
        #     visit_id = row['visit_id']
        #     
        #     # Option 1: Có lab code sẵn
        #     lab_code = row.get('lab_code', None)
        #     if pd.notna(lab_code):
        #         self.patient_codes[patient_id]['lab'].append(str(lab_code))
        #     
        #     # Option 2: Có tên xét nghiệm, cần normalize
        #     lab_name = row.get('lab_name', None)
        #     if pd.notna(lab_name):
        #         lab_name = self.normalize_vietnamese_text(lab_name)
        #         # Use lab_name as code hoặc map sang standard code
        #         self.patient_codes[patient_id]['lab'].append(lab_name)
        
        # TODO: Implement extraction logic
        raise NotImplementedError("Please implement extract_lab_codes()")
    
    def load_code_mappings(self, mapping_file=None):
        """
        Load code mappings (drug VN -> ATC, diagnosis VN -> ICD, etc.)
        TODO: Load từ file hoặc database
        """
        if mapping_file and os.path.exists(mapping_file):
            print(f"Loading code mappings from {mapping_file}...")
            # Load mappings
            # Example:
            # with open(mapping_file, 'rb') as f:
            #     mappings = pickle.load(f)
            #     self.drug_to_atc = mappings.get('drug_to_atc', {})
            #     self.diagnosis_to_icd = mappings.get('diagnosis_to_icd', {})
            pass
        else:
            print("No mapping file provided. Will use codes as-is or create mappings manually.")
    
    def create_manual_mappings(self):
        """
        Create manual mappings cho top codes
        TODO: Fill in mappings cho các codes phổ biến nhất
        """
        print("Creating manual code mappings...")
        
        # Example: Manual mapping cho top drugs
        # self.drug_to_atc = {
        #     'paracetamol': 'N02BE01',
        #     'amoxicillin': 'J01CA04',
        #     # ... add more mappings
        # }
        
        # Example: Manual mapping cho top diagnoses
        # self.diagnosis_to_icd = {
        #     'tiểu đường': 'E11',
        #     'cao huyết áp': 'I10',
        #     # ... add more mappings
        # }
        
        pass
    
    def extract_all(self):
        """
        Extract tất cả codes
        """
        print("=" * 80)
        print("Extracting codes from VN EHR")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Load mappings (nếu có)
        self.load_code_mappings()
        self.create_manual_mappings()
        
        # Extract codes
        self.extract_diagnosis_codes()
        self.extract_drug_codes()
        self.extract_lab_codes()
        
        # Convert lists to sets (remove duplicates)
        for patient_id in self.patient_codes:
            for code_type in self.patient_codes[patient_id]:
                self.patient_codes[patient_id][code_type] = list(
                    set(self.patient_codes[patient_id][code_type])
                )
        
        # Statistics
        num_patients = len(self.patient_codes)
        code_counts = defaultdict(int)
        for patient_id, codes_dict in self.patient_codes.items():
            for code_type, codes in codes_dict.items():
                code_counts[code_type] += len(codes)
        
        print(f"\nExtraction complete!")
        print(f"Number of patients: {num_patients}")
        print(f"Code counts: {dict(code_counts)}")
        
        # Save
        output_file = os.path.join(self.output_dir, 'vn_patient_codes.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(dict(self.patient_codes), f)
        print(f"Saved to: {output_file}")
        
        return self.patient_codes


def main():
    """
    Main function
    TODO: Update paths và parameters
    """
    # Paths
    data_path = 'path/to/vn_ehr_data'  # TODO: Update
    output_dir = 'data_vn'  # TODO: Update if needed
    mapping_file = None  # TODO: Path to mapping file if available
    
    # Create extractor
    extractor = VN_EHR_Extractor(data_path, output_dir)
    
    # Load mappings (nếu có)
    if mapping_file:
        extractor.load_code_mappings(mapping_file)
    
    # Extract
    patient_codes = extractor.extract_all()
    
    print("\nNext steps:")
    print("1. Review extracted codes")
    print("2. Create vocabulary")
    print("3. Build Knowledge Graph")


if __name__ == '__main__':
    main()

