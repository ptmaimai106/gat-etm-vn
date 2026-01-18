# Phân tích Input và Hướng dẫn Chi tiết Training GAT-ETM

## 1. PHÂN TÍCH CÁC INPUT ĐÃ CUNG CẤP

### ✅ 1.1. Dữ liệu EHR (Bag-of-Words) - **ĐÚNG VÀ ĐỦ**

**Mô tả của bạn:**
- Dạng túi từ bag-of-word: hồ sơ bệnh nhân được coi như một tài liệu, trong đó mã y tế(ICD, ATC) được coi như là các từ
- Vector tần suất: Vp = Vicd, Vatc
- Xử lý dữ liệu phi cấu trúc lâm sàng -> trích xuất mã bệnh, mã thuốc -> chuyển về mã chuẩn quốc tế -> đưa vào vector tần suất

**Phân tích:**
✅ **ĐÚNG** - Mô tả này chính xác với cách GAT-ETM xử lý dữ liệu.

**Chi tiết kỹ thuật cần bổ sung:**

#### Format cụ thể của BoW files:
- **File cần có:**
  1. `bow_train.npy` - Training data
  2. `bow_test.npy` - Full test data  
  3. `bow_test_1.npy` - First half của test data (cho document completion task)
  4. `bow_test_2.npy` - Second half của test data (cho document completion task)

#### Cấu trúc dữ liệu:
```python
# BoW matrix format: scipy.sparse.csr_matrix
# Shape: (num_patients, total_vocab_size)
# total_vocab_size = vocab_size[0] + vocab_size[1] + ...
# Ví dụ: nếu có ICD (61210 codes) + ATC (82020 codes)
#        thì total_vocab_size = 61210 + 82020 = 143230

# Cấu trúc columns:
# [ICD codes (0:61210), ATC codes (61210:143230)]
# Thứ tự phải khớp với metadata file!

# Giá trị trong matrix:
# - Frequency: số lần xuất hiện của code trong hồ sơ
# - Hoặc Binary: 0/1 (có/không có code)
```

#### Code để tạo BoW từ raw EHR data:
```python
import numpy as np
from scipy.sparse import csr_matrix

def create_bow_from_ehr(patient_records, vocab_mapping):
    """
    patient_records: list of dicts, mỗi dict chứa codes của một patient
    vocab_mapping: dict mapping code -> column index
    """
    num_patients = len(patient_records)
    num_codes = len(vocab_mapping)
    
    rows, cols, data = [], [], []
    
    for patient_idx, record in enumerate(patient_records):
        # record = {'icd': ['4019', '250'], 'atc': ['A10BA02', 'C07AB02']}
        for code_type in ['icd', 'atc']:
            if code_type in record:
                for code in record[code_type]:
                    if code in vocab_mapping:
                        col_idx = vocab_mapping[code]
                        rows.append(patient_idx)
                        cols.append(col_idx)
                        data.append(1)  # hoặc frequency
    
    bow_matrix = csr_matrix((data, (rows, cols)), 
                            shape=(num_patients, num_codes))
    
    # Lưu file
    np.save('bow_train.npy', bow_matrix, allow_pickle=True)
    return bow_matrix
```

---

### ✅ 1.2. Embedded Knowledge Graph - **ĐÚNG NHƯNG CẦN BỔ SUNG CHI TIẾT**

**Mô tả của bạn:**
- Embedded KG được tạo ra từ các liên kết:
  - ICD-ICD
  - ATC-ATC  
  - ICD-ATC
- Todo:
  - Thu thập các mã ICD, ATC từ dataset
  - Nếu dữ liệu gốc là mã thuốc thương mại -> ánh xạ sang mã ATC
  - Taxonomy: thu thập ICD, ATC để build Augmentation
  - Augmentation: kết nối mỗi node với tất cả tổ tiên

**Phân tích:**
✅ **ĐÚNG** - Mô tả đúng về cấu trúc KG, nhưng cần bổ sung:

#### Các loại edges cần có:

1. **Hierarchical edges (Taxonomy):**
   - ICD parent-child: `401` → `4019` (3-digit → 5-digit)
   - ATC hierarchy: `A` → `A10` → `A10B` → `A10BA` → `A10BA02`
   - Đây là edges từ ontology/taxonomy structure

2. **Co-occurrence edges:**
   - ICD-ATC: Nếu bệnh nhân có ICD code `4019` và dùng thuốc ATC `C07AB02` trong cùng visit → edge
   - ICD-ICD: Nếu 2 ICD codes xuất hiện cùng nhau trong cùng visit → edge
   - ATC-ATC: Nếu 2 ATC codes được kê cùng nhau → edge

3. **Augmented edges (Skip connections):**
   - Kết nối mỗi node với TẤT CẢ ancestors (không chỉ parent trực tiếp)
   - Ví dụ: `4019` → `401` → `40` → `ROOT`
   - Weight decay: `0.9^distance` (càng xa càng nhẹ)

#### Output files cần có:

1. **Graph file:** `augmented_icdatc_graph_{params}_renumbered_by_vocab.pkl`
   - Format: NetworkX Graph object (pickle)
   - Nodes: Tất cả ICD và ATC codes
   - Edges: Tất cả các loại edges trên
   - **QUAN TRỌNG:** Nodes phải được renumber theo thứ tự vocab trong BoW!

2. **Embeddings file:** `augmented_icdatc_embed_{window}_{walk_length}_{num_walks}_{dim}_by_vocab.pkl`
   - Format: numpy array (pickle)
   - Shape: `(num_nodes, embedding_dim)`
   - Được tạo bằng Node2Vec hoặc DeepWalk
   - **QUAN TRỌNG:** Thứ tự rows phải khớp với graph nodes (đã renumber)

#### Code để build KG (tóm tắt):
```python
import networkx as nx
import pickle
from node2vec import Node2Vec

# 1. Tạo graph với hierarchical edges
G = nx.DiGraph()
# Add ICD hierarchy
# Add ATC hierarchy

# 2. Add co-occurrence edges từ EHR data
for patient in patients:
    icd_codes = patient['icd']
    atc_codes = patient['atc']
    # Add edges giữa tất cả pairs trong cùng visit

# 3. Augment: Add skip connections
def augment_graph(G):
    for node in G.nodes():
        ancestors = get_all_ancestors(node, G)
        for ancestor in ancestors:
            distance = get_distance(node, ancestor)
            weight = 0.9 ** distance
            G.add_edge(node, ancestor, weight=weight)

# 4. Generate embeddings
node2vec = Node2Vec(G, dimensions=256, walk_length=20, num_walks=10)
model = node2vec.fit(window=8)
embeddings = model.wv.vectors  # numpy array

# 5. Renumber nodes theo vocab order
vocab_order = ['401', '4019', ..., 'A10BA02', ...]  # Thứ tự trong BoW
node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(vocab_order)}
G_renumbered = nx.relabel_nodes(G, node_mapping)
embeddings_renumbered = embeddings[vocab_order]

# 6. Save
pickle.dump(G_renumbered, open('graph.pkl', 'wb'))
pickle.dump(embeddings_renumbered, open('embeddings.pkl', 'wb'))
```

---

### ⚠️ 1.3. Metadata File - **THIẾU THÔNG TIN QUAN TRỌNG**

**Mô tả của bạn:**
```
code1 code2 ...
num_code1 num_code2 ...
train_init_emb_flag
init_emb_path
```

**Phân tích:**
⚠️ **THIẾU CHI TIẾT** - Format này đúng nhưng cần làm rõ:

#### Format chính xác:

File `metadata.txt` (hoặc `metadata_icd61210_atc82020_fine.txt`) có **4 dòng**, mỗi dòng là một Python list được convert sang string:

```python
# Dòng 1: Code types (tên các loại medical codes)
['icd', 'atc']

# Dòng 2: Vocab sizes (số lượng codes của mỗi loại)
[61210, 82020]

# Dòng 3: Train init embeddings flag (1 = train, 0 = freeze)
[1, 1]

# Dòng 4: Initial embedding paths (đường dẫn file embeddings, '*' = không có)
['*', '*']
```

#### Ví dụ file metadata.txt thực tế:
```
['icd', 'atc']
[61210, 82020]
[1, 1]
['*', '*']
```

#### Lưu ý quan trọng:

1. **Thứ tự code types PHẢI khớp với:**
   - Thứ tự columns trong BoW matrix
   - Thứ tự trong vocab_info.pkl (từ KG builder)
   - Thứ tự trong graph nodes

2. **Vocab sizes PHẢI chính xác:**
   - Phải bằng số lượng unique codes trong dataset
   - Phải khớp với vocab_size trong vocab_info.pkl

3. **Train embeddings flag:**
   - `1`: Embeddings sẽ được train (fine-tune) trong quá trình training
   - `0`: Embeddings được freeze (không update)
   - Thường set `[1, 1]` để train end-to-end

4. **Init emb paths:**
   - `'*'`: Không có pre-trained embeddings, sẽ dùng từ KG embeddings
   - Hoặc đường dẫn đến file `.npy` chứa embeddings: `'embed/icd_embeddings.npy'`

---

## 2. CÁC INPUT CÒN THIẾU HOẶC CẦN LÀM RÕ

### ❌ 2.1. Vocab Mapping và Alignment

**Vấn đề:** Cần đảm bảo vocab trong BoW khớp với vocab trong KG!

**Cần có:**
- Mapping giữa code strings và column indices trong BoW
- Mapping giữa code strings và node indices trong KG
- Đảm bảo thứ tự codes trong BoW = thứ tự nodes trong KG (sau renumbering)

**File cần tạo:**
```python
# vocab_mapping.pkl
{
    'icd': {
        '401': 0,
        '4019': 1,
        ...
    },
    'atc': {
        'A10BA02': 61210,
        ...
    }
}

# graphnode_vocab.pkl (từ KG builder)
{
    0: '401',      # node 0 trong graph = code '401'
    1: '4019',     # node 1 trong graph = code '4019'
    ...
}
```

### ❌ 2.2. Test Data Split

**Vấn đề:** Cần split test data thành 2 phần cho document completion task.

**Cần có:**
- `bow_test_1.npy`: First half của codes (để infer theta)
- `bow_test_2.npy`: Second half của codes (để evaluate reconstruction)

**Code để split:**
```python
def split_test_data(bow_test, code_types, vocab_cum):
    """
    Split test data theo từng code type
    """
    bow_test_1 = []
    bow_test_2 = []
    
    for i, code_type in enumerate(code_types):
        start_idx = vocab_cum[i]
        end_idx = vocab_cum[i+1]
        mid_idx = (start_idx + end_idx) // 2
        
        # Split mỗi code type thành 2 phần
        bow_test_1.append(bow_test[:, start_idx:mid_idx])
        bow_test_2.append(bow_test[:, mid_idx:end_idx])
    
    bow_test_1 = scipy.sparse.hstack(bow_test_1)
    bow_test_2 = scipy.sparse.hstack(bow_test_2)
    
    return bow_test_1, bow_test_2
```

---

## 3. HƯỚNG DẪN CHI TIẾT TỪNG BƯỚC TRAIN MODEL

### BƯỚC 1: Chuẩn bị Môi trường

```bash
# 1.1. Cài đặt dependencies
pip install torch torch-geometric
pip install numpy scipy scikit-learn
pip install networkx node2vec
pip install tqdm pandas
pip install python-louvain leidenalg

# 1.2. Kiểm tra GPU (nếu có)
python -c "import torch; print(torch.cuda.is_available())"
```

### BƯỚC 2: Chuẩn bị Dữ liệu EHR → BoW Format

#### 2.1. Extract và Normalize Medical Codes

```python
# extract_codes.py
import pandas as pd
import re

def extract_icd_codes(clinical_text, icd_mapping):
    """
    Trích xuất ICD codes từ text lâm sàng
    """
    # Sử dụng regex hoặc NER model để extract codes
    # Ví dụ: "Patient has hypertension (401.9)"
    codes = re.findall(r'\b\d{3}\.?\d*\b', clinical_text)
    normalized_codes = []
    for code in codes:
        # Normalize: remove dots, pad zeros
        normalized = code.replace('.', '').ljust(5, '0')
        if normalized in icd_mapping:
            normalized_codes.append(normalized)
    return normalized_codes

def map_drug_to_atc(drug_name, drug_mapping):
    """
    Map tên thuốc thương mại → ATC code
    """
    # Sử dụng RxNorm API hoặc mapping file
    if drug_name in drug_mapping:
        return drug_mapping[drug_name]
    return None

# Process từng patient record
patient_records = []
for patient_id, record in raw_ehr_data.items():
    icd_codes = extract_icd_codes(record['diagnosis_text'], icd_mapping)
    atc_codes = [map_drug_to_atc(drug, drug_mapping) 
                 for drug in record['prescriptions']]
    atc_codes = [c for c in atc_codes if c is not None]
    
    patient_records.append({
        'patient_id': patient_id,
        'icd': icd_codes,
        'atc': atc_codes
    })
```

#### 2.2. Tạo Vocabulary và Mapping

```python
# create_vocab.py
from collections import Counter

def create_vocab(patient_records, code_types):
    """
    Tạo vocabulary từ patient records
    """
    vocab = {ct: [] for ct in code_types}
    
    for record in patient_records:
        for code_type in code_types:
            if code_type in record:
                vocab[code_type].extend(record[code_type])
    
    # Get unique codes và sort
    vocab_unique = {}
    vocab_mapping = {}
    vocab_cum = [0]
    
    for code_type in code_types:
        unique_codes = sorted(set(vocab[code_type]))
        vocab_unique[code_type] = unique_codes
        vocab_size = len(unique_codes)
        vocab_cum.append(vocab_cum[-1] + vocab_size)
        
        # Create mapping: code -> column index
        vocab_mapping[code_type] = {
            code: vocab_cum[-2] + idx 
            for idx, code in enumerate(unique_codes)
        }
    
    return vocab_unique, vocab_mapping, vocab_cum

# Usage
code_types = ['icd', 'atc']
vocab_unique, vocab_mapping, vocab_cum = create_vocab(patient_records, code_types)

# Save vocab info
import pickle
with open('vocab_info.pkl', 'wb') as f:
    pickle.dump({
        'vocab_unique': vocab_unique,
        'vocab_mapping': vocab_mapping,
        'vocab_cum': vocab_cum,
        'vocab_size': [len(vocab_unique[ct]) for ct in code_types]
    }, f)
```

#### 2.3. Tạo BoW Matrices

```python
# create_bow.py
import numpy as np
from scipy.sparse import csr_matrix

def create_bow_matrix(patient_records, vocab_mapping, vocab_cum):
    """
    Tạo BoW matrix từ patient records
    """
    num_patients = len(patient_records)
    total_vocab_size = vocab_cum[-1]
    
    rows, cols, data = [], [], []
    
    for patient_idx, record in enumerate(patient_records):
        for code_type in ['icd', 'atc']:
            if code_type in record:
                for code in record[code_type]:
                    if code in vocab_mapping[code_type]:
                        col_idx = vocab_mapping[code_type][code]
                        rows.append(patient_idx)
                        cols.append(col_idx)
                        data.append(1)  # Binary encoding
                        # Hoặc data.append(frequency) nếu dùng frequency
    
    bow_matrix = csr_matrix((data, (rows, cols)), 
                            shape=(num_patients, total_vocab_size))
    return bow_matrix

# Create train và test BoW
train_records = patient_records[:train_size]
test_records = patient_records[train_size:]

bow_train = create_bow_matrix(train_records, vocab_mapping, vocab_cum)
bow_test = create_bow_matrix(test_records, vocab_mapping, vocab_cum)

# Split test data cho document completion
bow_test_1, bow_test_2 = split_test_data(bow_test, code_types, vocab_cum)

# Save files
np.save('data/bow_train.npy', bow_train, allow_pickle=True)
np.save('data/bow_test.npy', bow_test, allow_pickle=True)
np.save('data/bow_test_1.npy', bow_test_1, allow_pickle=True)
np.save('data/bow_test_2.npy', bow_test_2, allow_pickle=True)
```

### BƯỚC 3: Xây dựng Knowledge Graph

#### 3.1. Build Graph Structure

```python
# build_kg.py (tóm tắt các bước chính)
import networkx as nx
import pickle

def build_icd_hierarchy(icd_codes):
    """
    Build ICD hierarchy từ codes
    """
    G = nx.DiGraph()
    
    for code in icd_codes:
        # Add node
        G.add_node(code, type='ICD')
        
        # Add parent-child edges
        if len(code) >= 3:
            parent_3 = code[:3]
            G.add_node(parent_3, type='ICD')
            G.add_edge(code, parent_3)
        
        if len(code) >= 4:
            parent_4 = code[:4]
            G.add_node(parent_4, type='ICD')
            G.add_edge(code, parent_4)
    
    return G

def build_atc_hierarchy(atc_codes):
    """
    Build ATC hierarchy từ codes
    """
    G = nx.DiGraph()
    
    for code in atc_codes:
        G.add_node(code, type='ATC')
        
        # ATC hierarchy: A -> A10 -> A10B -> A10BA -> A10BA02
        for i in range(1, len(code)):
            parent = code[:i]
            G.add_node(parent, type='ATC')
            G.add_edge(code, parent)
    
    return G

def add_cooccurrence_edges(G, patient_records):
    """
    Add co-occurrence edges từ EHR data
    """
    # ICD-ATC co-occurrence
    for record in patient_records:
        icd_codes = record.get('icd', [])
        atc_codes = record.get('atc', [])
        
        for icd in icd_codes:
            for atc in atc_codes:
                if icd in G and atc in G:
                    if not G.has_edge(icd, atc):
                        G.add_edge(icd, atc, weight=1, type='cooccurrence')
                    else:
                        G[icd][atc]['weight'] += 1
    
    # ICD-ICD co-occurrence
    for record in patient_records:
        icd_codes = record.get('icd', [])
        for i, icd1 in enumerate(icd_codes):
            for icd2 in icd_codes[i+1:]:
                if icd1 in G and icd2 in G:
                    if not G.has_edge(icd1, icd2):
                        G.add_edge(icd1, icd2, weight=1, type='cooccurrence')
                    else:
                        G[icd1][icd2]['weight'] += 1
    
    return G

def augment_graph(G):
    """
    Add skip connections (connect node với tất cả ancestors)
    """
    G_aug = G.copy()
    
    for node in G.nodes():
        ancestors = get_all_ancestors(node, G)
        for ancestor in ancestors:
            distance = get_shortest_path_length(G, node, ancestor)
            if distance > 1:  # Skip direct parent
                weight = 0.9 ** distance
                if not G_aug.has_edge(node, ancestor):
                    G_aug.add_edge(node, ancestor, weight=weight, type='augmented')
    
    return G_aug

# Main build process
icd_codes = vocab_unique['icd']
atc_codes = vocab_unique['atc']

G_icd = build_icd_hierarchy(icd_codes)
G_atc = build_atc_hierarchy(atc_codes)
G = nx.compose(G_icd, G_atc)

G = add_cooccurrence_edges(G, patient_records)
G = augment_graph(G)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

#### 3.2. Generate Node Embeddings

```python
# generate_embeddings.py
from node2vec import Node2Vec
import numpy as np

# Convert to undirected for Node2Vec
G_undirected = G.to_undirected()

# Generate embeddings
node2vec = Node2Vec(
    G_undirected,
    dimensions=256,      # embedding dimension
    walk_length=20,     # length of each random walk
    num_walks=10,       # number of walks per node
    workers=4
)

model = node2vec.fit(window=8, min_count=1)

# Get embeddings
node_list = list(G.nodes())
embeddings = np.array([model.wv[node] for node in node_list])

print(f"Embeddings shape: {embeddings.shape}")
```

#### 3.3. Renumber Nodes theo Vocab Order

```python
# renumber_nodes.py
def renumber_nodes_by_vocab(G, embeddings, vocab_unique, code_types, vocab_cum):
    """
    Renumber nodes để khớp với vocab order trong BoW
    """
    # Tạo mapping: old_node -> new_index
    node_mapping = {}
    new_node_list = []
    
    for i, code_type in enumerate(code_types):
        codes = vocab_unique[code_type]
        for code in codes:
            if code in G:
                new_idx = len(new_node_list)
                node_mapping[code] = new_idx
                new_node_list.append(code)
    
    # Relabel graph
    G_renumbered = nx.relabel_nodes(G, node_mapping)
    
    # Reorder embeddings
    embeddings_renumbered = np.array([
        embeddings[node_list.index(node)] 
        for node in new_node_list
    ])
    
    # Create vocab mapping
    graphnode_vocab = {idx: node for idx, node in enumerate(new_node_list)}
    
    return G_renumbered, embeddings_renumbered, graphnode_vocab

# Usage
G_renumbered, embeddings_renumbered, graphnode_vocab = renumber_nodes_by_vocab(
    G, embeddings, vocab_unique, code_types, vocab_cum
)

# Save
pickle.dump(G_renumbered, open('embed/augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl', 'wb'))
pickle.dump(embeddings_renumbered, open('embed/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl', 'wb'))
pickle.dump(graphnode_vocab, open('embed/graphnode_vocab.pkl', 'wb'))
```

### BƯỚC 4: Tạo Metadata File

```python
# create_metadata.py
code_types = ['icd', 'atc']
vocab_sizes = [len(vocab_unique['icd']), len(vocab_unique['atc'])]
train_flags = [1, 1]  # Train embeddings
emb_paths = ['*', '*']  # Use KG embeddings

metadata_content = f"""['icd', 'atc']
{vocab_sizes}
{train_flags}
{emb_paths}
"""

with open('data/metadata.txt', 'w') as f:
    f.write(metadata_content)
```

### BƯỚC 5: Cấu trúc Thư mục

```
GAT-ETM/
├── data/
│   ├── metadata.txt                    # Metadata file
│   ├── bow_train.npy                   # Training BoW
│   ├── bow_test.npy                    # Test BoW
│   ├── bow_test_1.npy                  # Test split 1
│   └── bow_test_2.npy                  # Test split 2
├── embed/
│   ├── augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl
│   ├── augmented_icdatc_embed_8_20_10_256_by_vocab.pkl
│   ├── graphnode_vocab.pkl
│   └── vocab_info.pkl
├── results/                            # (sẽ được tạo khi train)
├── main_getm.py
├── graph_etm.py
├── graph_etm_DI.py
├── dataset.py
└── utils.py
```

### BƯỚC 6: Train Model

```bash
python main_getm.py \
    --data_path data/ \
    --save_path results/ \
    --meta_file metadata \
    --mode train \
    --gpu_device 0 \
    --num_topics 50 \
    --t_hidden_size 128 \
    --rho_size 256 \
    --emb_size 256 \
    --epochs 50 \
    --lr 0.01 \
    --batch_size 512 \
    --tq \
    --optimizer adam \
    --clip 2.0 \
    --theta_act relu \
    --bow_norm True
```

### BƯỚC 7: Kiểm tra Kết quả

Sau khi train xong, check các files trong `results/`:
- `beta_icd.npy`, `beta_atc.npy`: Topic-word distributions
- `rho_icd.npy`, `rho_atc.npy`: Learned code embeddings
- `alpha_icd.npy`, `alpha_atc.npy`: Topic embeddings
- `theta_train/`, `theta_test/`: Document-topic distributions

---

## 4. CHECKLIST TRƯỚC KHI TRAIN

### ✅ Dữ liệu:
- [ ] BoW matrices đã được tạo và format đúng (CSR sparse matrix)
- [ ] Vocab sizes trong metadata khớp với BoW matrices
- [ ] Test data đã được split thành 2 phần

### ✅ Knowledge Graph:
- [ ] Graph đã được build với đầy đủ edges (hierarchical + co-occurrence + augmented)
- [ ] Nodes đã được renumber theo vocab order
- [ ] Embeddings đã được generate và reorder theo vocab
- [ ] Graph và embeddings files đã được save đúng format

### ✅ Metadata:
- [ ] Metadata file format đúng (4 dòng)
- [ ] Code types khớp với BoW columns
- [ ] Vocab sizes chính xác
- [ ] Train flags và emb paths đã set đúng

### ✅ Code:
- [ ] Đã update `main_getm.py` với đúng graph_path và embed_path
- [ ] Đã kiểm tra các imports và dependencies

---

## 5. TÓM TẮT CÁC INPUT CẦN THIẾT

### Bắt buộc:
1. **BoW Matrices** (4 files):
   - `bow_train.npy`
   - `bow_test.npy`
   - `bow_test_1.npy`
   - `bow_test_2.npy`

2. **Knowledge Graph** (2 files):
   - Graph pickle file
   - Embeddings pickle file

3. **Metadata File**:
   - `metadata.txt` với format 4 dòng

### Tùy chọn:
- Pre-trained embeddings cho từng code type
- Vocab mapping files (để debug)

---

## 6. LƯU Ý QUAN TRỌNG

1. **Thứ tự vocab PHẢI khớp** giữa:
   - BoW matrix columns
   - Graph nodes (sau renumbering)
   - Metadata file

2. **Vocab sizes PHẢI chính xác**:
   - Phải bằng số lượng unique codes trong dataset
   - Phải khớp giữa metadata và vocab_info

3. **Graph nodes PHẢI bao gồm TẤT CẢ codes** trong BoW:
   - Nếu có code trong BoW nhưng không có trong graph → lỗi
   - Nếu có code trong graph nhưng không có trong BoW → OK (nhưng không được sử dụng)

4. **Test split PHẢI đúng format**:
   - `bow_test_1` và `bow_test_2` phải có cùng số patients
   - Mỗi code type được split thành 2 phần
