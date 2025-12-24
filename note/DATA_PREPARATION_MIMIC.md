# Hướng dẫn Chuẩn bị Dataset MIMIC-III cho GAT-ETM

Tài liệu này giải thích cách chuẩn bị dataset MIMIC-III để train mô hình GAT-ETM, dựa trên cách PopHR dataset được xử lý trong `main_getm.py`.

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Các file cần chuẩn bị](#các-file-cần-chuẩn-bị)
3. [Chi tiết từng file](#chi-tiết-từng-file)
4. [Quy trình chuẩn bị dữ liệu](#quy-trình-chuẩn-bị-dữ-liệu)
5. [Ví dụ code](#ví-dụ-code)
6. [Kiểm tra dữ liệu](#kiểm-tra-dữ-liệu)

---

## Tổng quan

Để train GAT-ETM với dataset MIMIC-III, bạn cần chuẩn bị:

1. **Knowledge Graph và Embeddings** (đã có từ `build_kg_mimic.py`)
2. **Bag-of-Words (BoW) matrices** cho train và test
3. **Metadata file** chứa thông tin về code types và vocab sizes
4. **Embedding files** (optional) cho từng code type

---

## Các file cần chuẩn bị

### 1. Knowledge Graph Files (từ KG Builder)

Đã được tạo từ `build_kg_mimic.py` hoặc `build_kg_mimic_sample.py`:

- `embed/augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl` - Graph pickle file
- `embed/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl` - Embeddings pickle file
- `embed/graphnode_vocab.pkl` - Vocab mapping
- `embed/vocab_info.pkl` - Vocab info

### 2. Data Files (cần tạo mới)

#### 2.1. Bag-of-Words Matrices

**Format:** Sparse matrix (scipy.sparse.csr_matrix) được lưu dưới dạng `.npy` với `allow_pickle=True`

- `data/bow_train.npy` - Training data (CSR matrix)
- `data/bow_test.npy` - Full test data (CSR matrix)
- `data/bow_test_1.npy` - First half of test data (cho document completion)
- `data/bow_test_2.npy` - Second half of test data (cho document completion)

**Cấu trúc:**
- Shape: `(num_patients, total_vocab_size)`
- `total_vocab_size = vocab_size[0] + vocab_size[1] + ...` (tổng vocab của tất cả code types)
- Mỗi hàng là một patient, mỗi cột là một code
- Giá trị: frequency hoặc binary (0/1)

**Ví dụ với 2 code types (ICD9, ATC):**
```
Row 0: [icd_0_freq, icd_1_freq, ..., icd_N_freq, atc_0_freq, atc_1_freq, ..., atc_M_freq]
Row 1: [icd_0_freq, icd_1_freq, ..., icd_N_freq, atc_0_freq, atc_1_freq, ..., atc_M_freq]
...
```

#### 2.2. Metadata File

**File:** `data/metadata.txt`

**Format:** 4 dòng, mỗi dòng là một list được convert sang string

**Nội dung:**
```
code_types vocab_sizes train_embeddings embedding_files
```

**Ví dụ:**
```
['icd', 'atc'] [1000, 500] [1, 1] ['*', '*']
```

**Giải thích:**
- **Dòng 1 - `code_types`**: List các loại codes (ví dụ: `['icd', 'atc']` hoặc `['icd', 'cpt', 'atc', 'lab']`)
- **Dòng 2 - `vocab_sizes`**: List kích thước vocab cho từng code type (ví dụ: `[1000, 500]` nghĩa là ICD có 1000 codes, ATC có 500 codes)
- **Dòng 3 - `train_embeddings`**: List 0/1 cho biết có train embeddings hay không (1 = train, 0 = fix)
- **Dòng 4 - `embedding_files`**: List đường dẫn đến embedding files cho từng code type (`'*'` = không có, sử dụng graph embeddings)

**Lưu ý:**
- Số phần tử trong mỗi list phải bằng nhau (bằng số code types)
- `vocab_sizes` phải khớp với vocab sizes từ KG builder
- Thứ tự code types phải khớp với thứ tự trong BoW matrices

#### 2.3. Embedding Files (Optional)

**Format:** NumPy array files (`.npy`)

**Ví dụ:**
- `data/icd_embeddings.npy` - Shape: `(vocab_size[0], embedding_dim)`
- `data/atc_embeddings.npy` - Shape: `(vocab_size[1], embedding_dim)`

**Lưu ý:**
- Nếu không có, sử dụng `'*'` trong metadata và model sẽ sử dụng graph embeddings
- Embeddings phải có cùng `embedding_dim` với graph embeddings (thường là 256)

---

## Chi tiết từng file

### 1. Bag-of-Words Matrix Format

#### Cấu trúc

```python
import numpy as np
from scipy.sparse import csr_matrix

# Ví dụ: 1000 patients, 1500 total vocab (1000 ICD + 500 ATC)
bow_matrix = csr_matrix((1000, 1500))

# Lưu file
np.save('data/bow_train.npy', bow_matrix, allow_pickle=True)
```

#### Vocab Ordering

Vocab order trong BoW matrix phải khớp với thứ tự trong KG:

```
[ICD9 codes: 0 to vocab_size[0]-1]
[CPT codes: vocab_size[0] to vocab_size[0]+vocab_size[1]-1]
[ATC codes: vocab_size[0]+vocab_size[1] to vocab_size[0]+vocab_size[1]+vocab_size[2]-1]
[Lab codes: ...]
```

**Ví dụ với vocab_cum:**
```python
vocab_cum = [0, 1000, 1200, 1700, 1800]
# ICD9: 0-999
# CPT: 1000-1199
# ATC: 1200-1699
# Lab: 1700-1799
```

#### Test Data Split

Test data được chia thành 2 phần cho document completion task:

```python
# bow_test.npy: Full test data
# bow_test_1.npy: First half (dùng để predict theta)
# bow_test_2.npy: Second half (dùng để evaluate reconstruction)
```

**Cách chia:**
```python
test_size = len(test_indices)
test_1_indices = test_indices[:test_size//2]
test_2_indices = test_indices[test_size//2:]

bow_test_1 = bow_matrix[test_1_indices]
bow_test_2 = bow_matrix[test_2_indices]
```

### 2. Metadata File Format

#### Ví dụ đầy đủ

```
['icd', 'atc']
[1000, 500]
[1, 1]
['*', '*']
```

#### Giải thích từng dòng

**Dòng 1: Code Types**
```python
code_types = ['icd', 'atc']  # hoặc ['icd', 'cpt', 'atc', 'lab']
```

**Dòng 2: Vocab Sizes**
```python
vocab_sizes = [1000, 500]  # ICD có 1000 codes, ATC có 500 codes
```

**Lưu ý:** Vocab sizes phải khớp với:
- Vocab sizes từ `vocab_info.pkl` (từ KG builder)
- Số cột trong BoW matrices cho mỗi code type

**Dòng 3: Train Embeddings**
```python
train_embeddings = [1, 1]  # 1 = train embeddings, 0 = fix embeddings
```

**Dòng 4: Embedding Files**
```python
embedding_files = ['*', '*']  # '*' = không có file, sử dụng graph embeddings
# hoặc
embedding_files = ['data/icd_embeddings.npy', 'data/atc_embeddings.npy']
```

### 3. Vocab Mapping với KG

Vocab order trong BoW phải khớp với vocab order trong KG (sau khi renumber):

```python
# Load vocab info từ KG
with open('embed/vocab_info.pkl', 'rb') as f:
    vocab_info = pickle.load(f)

# vocab_info = {
#     'icd': ['250.10', '401.9', ...],  # 1000 codes
#     'atc': ['DRUG_12345', ...],        # 500 codes
#     ...
# }

# Vocab order trong BoW:
# Column 0-999: ICD codes theo thứ tự vocab_info['icd']
# Column 1000-1499: ATC codes theo thứ tự vocab_info['atc']
```

---

## Quy trình chuẩn bị dữ liệu

### Bước 1: Extract Codes từ MIMIC-III

```python
import pandas as pd
import numpy as np
from collections import defaultdict

def extract_codes_from_mimic(mimic_path):
    """
    Extract codes từ MIMIC-III tables
    
    Returns:
        patient_codes: dict {patient_id: {code_type: [codes]}}
    """
    # Load tables
    diagnoses_icd = pd.read_csv(f'{mimic_path}/DIAGNOSES_ICD.csv')
    prescriptions = pd.read_csv(f'{mimic_path}/PRESCRIPTIONS.csv')
    # ... load other tables
    
    patient_codes = defaultdict(lambda: defaultdict(list))
    
    # Extract ICD9 codes
    for _, row in diagnoses_icd.iterrows():
        patient_id = row['subject_id']
        hadm_id = row['hadm_id']
        icd9 = row['icd9_code']
        if pd.notna(icd9):
            patient_codes[patient_id]['icd'].append(icd9)
    
    # Extract ATC codes (từ prescriptions)
    # Cần map drug_name_generic -> ATC code (sử dụng mapping từ KG builder)
    for _, row in prescriptions.iterrows():
        patient_id = row['subject_id']
        drug = row['drug_name_generic']
        # Map drug -> ATC (sử dụng drug_to_atc từ KG builder)
        # ...
    
    return patient_codes
```

### Bước 2: Tạo Vocabulary

```python
def create_vocabulary(patient_codes, vocab_info_from_kg):
    """
    Tạo vocabulary từ vocab_info của KG
    
    Args:
        patient_codes: dict từ extract_codes_from_mimic
        vocab_info_from_kg: vocab_info từ vocab_info.pkl
    
    Returns:
        vocab_icd, vocab_atc, code_to_idx
    """
    # Sử dụng vocab từ KG để đảm bảo thứ tự khớp
    vocab_icd = vocab_info_from_kg['icd']
    vocab_atc = vocab_info_from_kg['atc']
    
    # Tạo mapping: code -> index
    code_to_idx = {}
    
    # ICD codes
    for idx, code in enumerate(vocab_icd):
        code_to_idx[('icd', code)] = idx
    
    # ATC codes
    vocab_cum = [0, len(vocab_icd), len(vocab_icd) + len(vocab_atc)]
    for idx, code in enumerate(vocab_atc):
        code_to_idx[('atc', code)] = vocab_cum[1] + idx
    
    return vocab_icd, vocab_atc, code_to_idx
```

### Bước 3: Tạo BoW Matrix

```python
from scipy.sparse import csr_matrix

def create_bow_matrix(patient_codes, code_to_idx, vocab_icd, vocab_atc):
    """
    Tạo Bag-of-Words matrix từ patient codes
    
    Args:
        patient_codes: dict {patient_id: {code_type: [codes]}}
        code_to_idx: dict {('icd', code): idx} hoặc {('atc', code): idx}
        vocab_icd, vocab_atc: vocab lists
    
    Returns:
        bow_matrix: CSR matrix (num_patients, total_vocab_size)
    """
    num_patients = len(patient_codes)
    total_vocab_size = len(vocab_icd) + len(vocab_atc)
    
    # Initialize sparse matrix
    rows = []
    cols = []
    data = []
    
    for patient_idx, (patient_id, codes_dict) in enumerate(patient_codes.items()):
        # Count codes
        code_counts = defaultdict(int)
        
        # Count ICD codes
        for code in codes_dict.get('icd', []):
            # Normalize code (bỏ dấu chấm nếu có)
            code_normalized = str(code).strip().replace('.', '')
            # Tìm code trong vocab (có thể cần normalize)
            # ...
            if ('icd', code) in code_to_idx:
                code_counts[code_to_idx[('icd', code)]] += 1
        
        # Count ATC codes
        for code in codes_dict.get('atc', []):
            if ('atc', code) in code_to_idx:
                code_counts[code_to_idx[('atc', code)]] += 1
        
        # Add to sparse matrix
        for col_idx, count in code_counts.items():
            rows.append(patient_idx)
            cols.append(col_idx)
            data.append(count)
    
    # Create CSR matrix
    bow_matrix = csr_matrix((data, (rows, cols)), 
                           shape=(num_patients, total_vocab_size))
    
    return bow_matrix
```

### Bước 4: Split Train/Test

```python
def split_train_test(bow_matrix, train_ratio=0.8, random_seed=42):
    """
    Chia dữ liệu thành train và test
    """
    np.random.seed(random_seed)
    num_patients = bow_matrix.shape[0]
    indices = np.random.permutation(num_patients)
    
    train_size = int(num_patients * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    bow_train = bow_matrix[train_indices]
    bow_test = bow_matrix[test_indices]
    
    # Chia test thành 2 nửa cho document completion
    test_size = len(test_indices)
    test_1_indices = test_indices[:test_size//2]
    test_2_indices = test_indices[test_size//2:]
    
    bow_test_1 = bow_matrix[test_1_indices]
    bow_test_2 = bow_matrix[test_2_indices]
    
    return bow_train, bow_test, bow_test_1, bow_test_2
```

### Bước 5: Tạo Metadata File

```python
def create_metadata_file(code_types, vocab_sizes, train_embeddings, embedding_files, output_path):
    """
    Tạo metadata.txt file
    """
    with open(output_path, 'w') as f:
        # Dòng 1: code_types
        f.write(str(code_types) + '\n')
        # Dòng 2: vocab_sizes
        f.write(str(vocab_sizes) + '\n')
        # Dòng 3: train_embeddings
        f.write(str(train_embeddings) + '\n')
        # Dòng 4: embedding_files
        f.write(str(embedding_files) + '\n')
```

### Bước 6: Lưu Files

```python
# Lưu BoW matrices
np.save('data/bow_train.npy', bow_train, allow_pickle=True)
np.save('data/bow_test.npy', bow_test, allow_pickle=True)
np.save('data/bow_test_1.npy', bow_test_1, allow_pickle=True)
np.save('data/bow_test_2.npy', bow_test_2, allow_pickle=True)

# Tạo metadata
code_types = ['icd', 'atc']
vocab_sizes = [len(vocab_icd), len(vocab_atc)]
train_embeddings = [1, 1]
embedding_files = ['*', '*']  # Sử dụng graph embeddings

create_metadata_file(code_types, vocab_sizes, train_embeddings, 
                     embedding_files, 'data/metadata.txt')
```

---

## Ví dụ code

### Script hoàn chỉnh

```python
#!/usr/bin/env python3
"""
Script để chuẩn bị MIMIC-III data cho GAT-ETM training
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter


def load_kg_vocab(kg_embed_dir='KG_EMBED/embed'):
    """Load vocab info từ KG builder"""
    with open(f'{kg_embed_dir}/vocab_info.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    return vocab_info


def extract_codes_from_mimic(mimic_path, vocab_info, drug_to_atc_mapping=None):
    """
    Extract codes từ MIMIC-III và map về vocab từ KG
    """
    patient_codes = defaultdict(lambda: defaultdict(set))

    # Load tables
    diagnoses_icd = pd.read_csv(f'{mimic_path}/DIAGNOSES_ICD.csv')
    prescriptions = pd.read_csv(f'{mimic_path}/PRESCRIPTIONS.csv')

    # Normalize ICD9 codes (bỏ dấu chấm)
    def normalize_icd(code):
        if pd.isna(code):
            return None
        code_str = str(code).strip()
        # Giữ cả format có và không có dấu chấm
        return code_str, code_str.replace('.', '')

    # Tạo vocab sets để lookup nhanh
    vocab_icd_set = set(vocab_info['icd'])
    vocab_icd_normalized = {v.replace('.', '') if '.' in v else v: v for v in vocab_info['icd']}

    # Extract ICD9 codes
    for _, row in diagnoses_icd.iterrows():
        patient_id = row['subject_id']
        icd9 = row['icd9_code']
        if pd.notna(icd9):
            orig, normalized = normalize_icd(icd9)
            # Tìm trong vocab
            if orig in vocab_icd_set:
                patient_codes[patient_id]['icd'].add(orig)
            elif normalized in vocab_icd_normalized:
                patient_codes[patient_id]['icd'].add(vocab_icd_normalized[normalized])

    # Extract ATC codes từ prescriptions
    if drug_to_atc_mapping:
        for _, row in prescriptions.iterrows():
            patient_id = row['subject_id']
            drug = str(row.get('drug_name_generic', '')).strip().upper()
            if drug and drug in drug_to_atc_mapping:
                atc_code = drug_to_atc_mapping[drug]
                if atc_code in vocab_info['atc']:
                    patient_codes[patient_id]['atc'].add(atc_code)

    # Convert sets to lists
    patient_codes_list = {}
    for pid, codes_dict in patient_codes.items():
        patient_codes_list[pid] = {
            'icd': list(codes_dict['icd']),
            'atc': list(codes_dict['atc'])
        }

    return patient_codes_list


def create_code_to_idx(vocab_info):
    """Tạo mapping từ code -> index trong BoW matrix"""
    code_to_idx = {}

    vocab_cum = [0]
    for code_type in ['icd', 'atc']:  # Có thể thêm 'cpt', 'lab'
        if code_type in vocab_info:
            vocab_cum.append(vocab_cum[-1] + len(vocab_info[code_type]))

    idx = 0
    for code_type in ['icd', 'atc']:
        if code_type in vocab_info:
            for code in vocab_info[code_type]:
                code_to_idx[(code_type, code)] = idx
                idx += 1

    return code_to_idx, vocab_cum


def create_bow_matrix(patient_codes, code_to_idx):
    """Tạo BoW matrix"""
    num_patients = len(patient_codes)
    total_vocab_size = max(code_to_idx.values()) + 1

    rows = []
    cols = []
    data = []

    for patient_idx, (patient_id, codes_dict) in enumerate(patient_codes.items()):
        code_counts = Counter()

        for code_type in ['icd', 'atc']:
            for code in codes_dict.get(code_type, []):
                key = (code_type, code)
                if key in code_to_idx:
                    code_counts[code_to_idx[key]] += 1

        for col_idx, count in code_counts.items():
            rows.append(patient_idx)
            cols.append(col_idx)
            data.append(count)

    bow_matrix = csr_matrix((data, (rows, cols)),
                            shape=(num_patients, total_vocab_size))
    return bow_matrix


def split_train_test(bow_matrix, train_ratio=0.8, random_seed=42):
    """Chia train/test"""
    np.random.seed(random_seed)
    num_patients = bow_matrix.shape[0]
    indices = np.random.permutation(num_patients)

    train_size = int(num_patients * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    bow_train = bow_matrix[train_indices]
    bow_test = bow_matrix[test_indices]

    test_size = len(test_indices)
    test_1_indices = test_indices[:test_size // 2]
    test_2_indices = test_indices[test_size // 2:]

    bow_test_1 = bow_matrix[test_1_indices]
    bow_test_2 = bow_matrix[test_2_indices]

    return bow_train, bow_test, bow_test_1, bow_test_2


def create_metadata_file(code_types, vocab_sizes, train_embeddings,
                         embedding_files, output_path):
    """Tạo metadata.txt"""
    with open(output_path, 'w') as f:
        f.write(str(code_types) + '\n')
        f.write(str(vocab_sizes) + '\n')
        f.write(str(train_embeddings) + '\n')
        f.write(str(embedding_files) + '\n')


def main():
    # Paths
    mimic_path = '../mimic-iii-clinical-database-demo-1.4'
    kg_embed_dir = '../KG_EMBED/embed'
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load vocab từ KG
    print("Loading vocab from KG...")
    vocab_info = load_kg_vocab(kg_embed_dir)
    print(f"ICD vocab size: {len(vocab_info['icd'])}")
    print(f"ATC vocab size: {len(vocab_info['atc'])}")

    # Step 2: Load drug_to_atc mapping (nếu có từ KG builder)
    drug_to_atc = None
    # Có thể load từ KG builder nếu đã lưu

    # Step 3: Extract codes từ MIMIC-III
    print("Extracting codes from MIMIC-III...")
    patient_codes = extract_codes_from_mimic(mimic_path, vocab_info, drug_to_atc)
    print(f"Number of patients: {len(patient_codes)}")

    # Step 4: Create code_to_idx mapping
    code_to_idx, vocab_cum = create_code_to_idx(vocab_info)
    print(f"Total vocab size: {vocab_cum[-1]}")
    print(f"Vocab cumulative: {vocab_cum}")

    # Step 5: Create BoW matrix
    print("Creating BoW matrix...")
    bow_matrix = create_bow_matrix(patient_codes, code_to_idx)
    print(f"BoW matrix shape: {bow_matrix.shape}")
    print(f"Non-zero entries: {bow_matrix.nnz}")

    # Step 6: Split train/test
    print("Splitting train/test...")
    bow_train, bow_test, bow_test_1, bow_test_2 = split_train_test(bow_matrix)
    print(f"Train size: {bow_train.shape[0]}")
    print(f"Test size: {bow_test.shape[0]}")

    # Step 7: Save BoW matrices
    print("Saving BoW matrices...")
    np.save(f'{output_dir}/bow_train.npy', bow_train, allow_pickle=True)
    np.save(f'{output_dir}/bow_test.npy', bow_test, allow_pickle=True)
    np.save(f'{output_dir}/bow_test_1.npy', bow_test_1, allow_pickle=True)
    np.save(f'{output_dir}/bow_test_2.npy', bow_test_2, allow_pickle=True)

    # Step 8: Create metadata file
    print("Creating metadata file...")
    code_types = ['icd', 'atc']
    vocab_sizes = [len(vocab_info['icd']), len(vocab_info['atc'])]
    train_embeddings = [1, 1]
    embedding_files = ['*', '*']  # Sử dụng graph embeddings

    create_metadata_file(code_types, vocab_sizes, train_embeddings,
                         embedding_files, f'{output_dir}/metadata.txt')

    print("Data preparation complete!")
    print(f"\nFiles created in {output_dir}/:")
    print("  - bow_train.npy")
    print("  - bow_test.npy")
    print("  - bow_test_1.npy")
    print("  - bow_test_2.npy")
    print("  - metadata.txt")


if __name__ == '__main__':
    main()
```

---

## Kiểm tra dữ liệu

### 1. Kiểm tra BoW Matrices

```python
import numpy as np
from scipy.sparse import csr_matrix

# Load BoW matrix
bow_train = np.load('data/bow_train.npy', allow_pickle=True).item()

print(f"Shape: {bow_train.shape}")
print(f"Type: {type(bow_train)}")
print(f"Non-zero entries: {bow_train.nnz}")
print(f"Density: {bow_train.nnz / (bow_train.shape[0] * bow_train.shape[1]):.4f}")

# Kiểm tra vocab ranges
print(f"\nFirst 10 columns (should be ICD): {bow_train[:5, :10].toarray()}")
print(f"Columns 1000-1010 (should be ATC): {bow_train[:5, 1000:1010].toarray()}")
```

### 2. Kiểm tra Metadata

```python
# Load metadata
metadata = np.loadtxt('data/metadata.txt', dtype=str)
code_types, vocab_sizes, train_embeddings, embedding_files = metadata

print(f"Code types: {code_types}")
print(f"Vocab sizes: {vocab_sizes}")
print(f"Train embeddings: {train_embeddings}")
print(f"Embedding files: {embedding_files}")

# Kiểm tra vocab sizes khớp với BoW
bow_train = np.load('data/bow_train.npy', allow_pickle=True).item()
total_vocab = bow_train.shape[1]
expected_total = sum([int(v) for v in vocab_sizes])

print(f"\nBoW total vocab: {total_vocab}")
print(f"Expected total vocab: {expected_total}")
assert total_vocab == expected_total, "Vocab sizes don't match!"
```

### 3. Kiểm tra Vocab Order

```python
import pickle

# Load vocab info từ KG
with open('KG_EMBED/embed/vocab_info.pkl', 'rb') as f:
    vocab_info = pickle.load(f)

# Load BoW
bow_train = np.load('data/bow_train.npy', allow_pickle=True).item()

# Kiểm tra một vài codes
print("Checking vocab order...")
print(f"First ICD code in vocab: {vocab_info['icd'][0]}")
print(f"First ATC code in vocab: {vocab_info['atc'][0]}")

# Kiểm tra có patients nào có codes này không
# (cần implement lookup dựa trên code_to_idx)
```

### 4. Test với main_getm.py

```bash
# Chạy với mode eval để kiểm tra data loading
python main_getm.py \
    --data_path data/ \
    --meta_file metadata \
    --mode eval \
    --load_from <checkpoint_path> \
    --graph_path KG_EMBED/embed/augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl
```

---

## Lưu ý quan trọng

### 1. Vocab Order Phải Khớp

- Vocab order trong BoW matrix **phải khớp** với vocab order trong KG (sau khi renumber)
- Sử dụng `vocab_info.pkl` từ KG builder để đảm bảo thứ tự đúng

### 2. Code Normalization

- ICD9 codes có thể có hoặc không có dấu chấm (ví dụ: "250.10" vs "25010")
- Cần normalize để match với vocab từ KG
- KG builder thường normalize bằng cách bỏ dấu chấm

### 3. Drug to ATC Mapping

- Cần map `drug_name_generic` từ PRESCRIPTIONS → ATC codes
- Sử dụng mapping từ KG builder (nếu có) hoặc tạo mapping riêng
- Có thể sử dụng RxNorm API hoặc ATC mapping file

### 4. Sparse Matrix Format

- BoW matrices phải là **scipy.sparse.csr_matrix**
- Lưu dưới dạng `.npy` với `allow_pickle=True`
- Khi load, sử dụng `.item()` để lấy matrix object

### 5. Test Data Split

- Test data được chia thành 2 phần cho document completion task
- `bow_test_1`: Dùng để predict theta (encoder input)
- `bow_test_2`: Dùng để evaluate reconstruction (decoder target)

### 6. Memory Considerations

- Với dataset lớn, BoW matrices có thể rất lớn
- Sử dụng sparse matrices để tiết kiệm memory
- Có thể cần batch processing khi extract codes

---

## Troubleshooting

### Lỗi: "Vocab sizes don't match"

**Nguyên nhân:** Vocab sizes trong metadata không khớp với BoW matrix

**Giải pháp:**
- Kiểm tra lại `vocab_sizes` trong metadata
- Đảm bảo tổng `vocab_sizes` = số cột trong BoW matrix

### Lỗi: "Index out of range" khi load data

**Nguyên nhân:** Vocab order trong BoW không khớp với vocab order trong KG

**Giải pháp:**
- Sử dụng `vocab_info.pkl` từ KG builder để tạo code_to_idx
- Đảm bảo thứ tự code types giống nhau

### Lỗi: "Sparse matrix format error"

**Nguyên nhân:** BoW matrix không phải CSR format

**Giải pháp:**
- Convert sang CSR: `bow_matrix = csr_matrix(bow_matrix)`
- Lưu với `allow_pickle=True`

### Lỗi: "Drug codes not found in vocab"

**Nguyên nhân:** Drug to ATC mapping không đúng hoặc ATC codes không có trong vocab

**Giải pháp:**
- Kiểm tra drug_to_atc mapping
- Đảm bảo ATC codes trong vocab từ KG builder

---

## Tóm tắt Checklist

- [ ] Knowledge Graph files từ KG builder
- [ ] Extract codes từ MIMIC-III tables
- [ ] Map codes về vocab từ KG (đảm bảo thứ tự đúng)
- [ ] Tạo BoW matrices (train, test, test_1, test_2)
- [ ] Tạo metadata.txt với vocab sizes đúng
- [ ] Kiểm tra vocab order khớp với KG
- [ ] Test data loading với main_getm.py

---

## Tài liệu tham khảo

- `main_getm.py`: Main training script
- `dataset.py`: Dataset class implementation
- `KG_EMBED/README.md`: Knowledge Graph builder documentation

