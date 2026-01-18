# Tóm Tắt: Phân Tích Input và Các Bước Train GAT-ETM

## 📋 PHÂN TÍCH CÁC INPUT ĐÃ CUNG CẤP

### ✅ 1. Dữ liệu EHR (Bag-of-Words) - **ĐÚNG VÀ ĐỦ**

**Mô tả của bạn:** ✅ Chính xác
- Dạng túi từ bag-of-word
- Vector tần suất: Vp = Vicd, Vatc
- Xử lý dữ liệu phi cấu trúc → trích xuất mã → chuẩn hóa → vector tần suất

**Bổ sung chi tiết kỹ thuật:**

#### Files cần có:
```
data/
├── bow_train.npy      # CSR sparse matrix: (num_patients, total_vocab_size)
├── bow_test.npy       # Full test data
├── bow_test_1.npy     # First half (cho document completion)
└── bow_test_2.npy     # Second half (cho document completion)
```

#### Format:
- **Type:** `scipy.sparse.csr_matrix` được lưu dưới dạng `.npy` với `allow_pickle=True`
- **Shape:** `(num_patients, vocab_size[0] + vocab_size[1] + ...)`
- **Columns order:** `[ICD codes (0:vocab_size[0]), ATC codes (vocab_size[0]:vocab_size[0]+vocab_size[1]), ...]`
- **Values:** Binary (0/1) hoặc frequency

---

### ✅ 2. Embedded Knowledge Graph - **ĐÚNG NHƯNG CẦN BỔ SUNG**

**Mô tả của bạn:** ✅ Đúng về cấu trúc, nhưng thiếu chi tiết format

#### Các loại edges cần có:

1. **Hierarchical edges (Taxonomy):**
   - ICD: `401` → `4019` (parent-child)
   - ATC: `A` → `A10` → `A10B` → `A10BA02`

2. **Co-occurrence edges:**
   - ICD-ATC: Cùng xuất hiện trong cùng visit
   - ICD-ICD: Cùng xuất hiện trong cùng visit
   - ATC-ATC: Cùng được kê đơn

3. **Augmented edges (Skip connections):**
   - Kết nối mỗi node với TẤT CẢ ancestors
   - Weight: `0.9^distance`

#### Output files cần có:

```
embed/
├── augmented_icdatc_graph_{params}_renumbered_by_vocab.pkl
│   └── NetworkX Graph object (pickle)
│
└── augmented_icdatc_embed_{window}_{walk_length}_{num_walks}_{dim}_by_vocab.pkl
    └── numpy array: (num_nodes, embedding_dim)
```

**QUAN TRỌNG:** Nodes phải được **renumber** theo thứ tự vocab trong BoW!

---

### ⚠️ 3. Metadata File - **THIẾU CHI TIẾT FORMAT**

**Mô tả của bạn:** ⚠️ Đúng nhưng cần làm rõ format chính xác

#### Format chính xác (4 dòng):

File `data/metadata.txt`:
```
['icd', 'atc']
[61210, 82020]
[1, 1]
['*', '*']
```

**Giải thích:**
- **Dòng 1:** Code types (tên các loại mã y tế)
- **Dòng 2:** Vocab sizes (số lượng codes của mỗi loại)
- **Dòng 3:** Train init embeddings flag (`1` = train, `0` = freeze)
- **Dòng 4:** Initial embedding paths (`'*'` = dùng KG embeddings)

**Lưu ý:**
- Thứ tự code types PHẢI khớp với thứ tự columns trong BoW
- Vocab sizes PHẢI chính xác với số lượng unique codes

---

## ❌ CÁC INPUT CÒN THIẾU

### 1. Vocab Alignment
- Cần đảm bảo vocab trong BoW = vocab trong KG (sau renumbering)
- Cần file mapping: `graphnode_vocab.pkl`

### 2. Test Data Split
- Cần split test data thành 2 phần cho document completion task
- Mỗi code type được chia đôi

---

## 🚀 CÁC BƯỚC CHI TIẾT ĐỂ TRAIN MODEL

### **BƯỚC 1: Chuẩn bị Môi trường**

```bash
pip install torch torch-geometric numpy scipy scikit-learn networkx node2vec tqdm pandas
```

### **BƯỚC 2: Xử lý Dữ liệu EHR → BoW**

#### 2.1. Extract và Normalize Codes

```python
# Từ raw EHR data:
# - Trích xuất ICD codes từ diagnosis text
# - Map tên thuốc thương mại → ATC codes
# - Normalize format (remove dots, pad zeros)

patient_records = [
    {'patient_id': 'P001', 'icd': ['4019', '250'], 'atc': ['A10BA02', 'C07AB02']},
    ...
]
```

#### 2.2. Tạo Vocabulary

```python
# Tạo vocab từ tất cả codes trong dataset
vocab_unique = {
    'icd': ['401', '4019', '250', ...],  # Sorted unique codes
    'atc': ['A10BA02', 'C07AB02', ...]
}

vocab_mapping = {
    'icd': {'401': 0, '4019': 1, ...},
    'atc': {'A10BA02': 61210, ...}
}

vocab_cum = [0, 61210, 143230]  # Cumulative sizes
```

#### 2.3. Tạo BoW Matrices

```python
from scipy.sparse import csr_matrix
import numpy as np

# Create sparse matrix
rows, cols, data = [], [], []
for patient_idx, record in enumerate(patient_records):
    for code_type in ['icd', 'atc']:
        for code in record[code_type]:
            col_idx = vocab_mapping[code_type][code]
            rows.append(patient_idx)
            cols.append(col_idx)
            data.append(1)

bow_train = csr_matrix((data, (rows, cols)), 
                       shape=(num_patients, total_vocab_size))

# Save
np.save('data/bow_train.npy', bow_train, allow_pickle=True)
```

#### 2.4. Split Test Data

```python
# Split mỗi code type thành 2 phần
# Ví dụ: ICD codes [0:61210] → [0:30605] và [30605:61210]
bow_test_1, bow_test_2 = split_test_data(bow_test, code_types, vocab_cum)
```

### **BƯỚC 3: Xây dựng Knowledge Graph**

#### 3.1. Build Graph Structure

```python
import networkx as nx

# 1. Build hierarchical edges
G = nx.DiGraph()
# Add ICD hierarchy
# Add ATC hierarchy

# 2. Add co-occurrence edges từ EHR data
for patient in patient_records:
    for icd in patient['icd']:
        for atc in patient['atc']:
            G.add_edge(icd, atc, type='cooccurrence')

# 3. Augment: Add skip connections
for node in G.nodes():
    ancestors = get_all_ancestors(node)
    for ancestor in ancestors:
        G.add_edge(node, ancestor, weight=0.9**distance)
```

#### 3.2. Generate Embeddings

```python
from node2vec import Node2Vec

node2vec = Node2Vec(G.to_undirected(), dimensions=256, 
                   walk_length=20, num_walks=10)
model = node2vec.fit(window=8)
embeddings = np.array([model.wv[node] for node in G.nodes()])
```

#### 3.3. Renumber Nodes theo Vocab Order

```python
# QUAN TRỌNG: Renumber để khớp với BoW columns
node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(vocab_order)}
G_renumbered = nx.relabel_nodes(G, node_mapping)
embeddings_renumbered = embeddings[vocab_order]

# Save
pickle.dump(G_renumbered, open('embed/graph.pkl', 'wb'))
pickle.dump(embeddings_renumbered, open('embed/embeddings.pkl', 'wb'))
```

### **BƯỚC 4: Tạo Metadata File**

```python
# data/metadata.txt
metadata = """['icd', 'atc']
[61210, 82020]
[1, 1]
['*', '*']
"""
```

### **BƯỚC 5: Cấu trúc Thư mục**

```
GAT-ETM/
├── data/
│   ├── metadata.txt
│   ├── bow_train.npy
│   ├── bow_test.npy
│   ├── bow_test_1.npy
│   └── bow_test_2.npy
├── embed/
│   ├── augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl
│   └── augmented_icdatc_embed_8_20_10_256_by_vocab.pkl
├── main_getm.py
├── graph_etm.py
└── ...
```

### **BƯỚC 6: Update main_getm.py**

```python
# Dòng 114: Update graph path
args.graph_path = 'embed/augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl'

# Dòng 117: Update embeddings path
args.graph_embed = pickle.load(open('embed/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl', 'rb'))
```

### **BƯỚC 7: Train Model**

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
    --epochs 50 \
    --lr 0.01 \
    --batch_size 512 \
    --tq \
    --optimizer adam
```

---

## ✅ CHECKLIST TRƯỚC KHI TRAIN

### Dữ liệu:
- [ ] BoW matrices đã tạo (4 files)
- [ ] Format đúng (CSR sparse matrix)
- [ ] Vocab sizes khớp với metadata

### Knowledge Graph:
- [ ] Graph đã build với đủ edges
- [ ] Nodes đã renumber theo vocab order
- [ ] Embeddings đã generate và reorder
- [ ] Files đã save đúng format

### Metadata:
- [ ] Format đúng (4 dòng)
- [ ] Code types khớp với BoW columns
- [ ] Vocab sizes chính xác

### Code:
- [ ] Đã update graph_path và embed_path trong main_getm.py
- [ ] Đã kiểm tra dependencies

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Thứ tự vocab PHẢI khớp** giữa:
   - BoW matrix columns
   - Graph nodes (sau renumbering)
   - Metadata file

2. **Vocab sizes PHẢI chính xác**:
   - Bằng số lượng unique codes trong dataset
   - Khớp giữa metadata và vocab_info

3. **Graph nodes PHẢI bao gồm TẤT CẢ codes** trong BoW

4. **Test split PHẢI đúng format**:
   - Mỗi code type được chia đôi

---

## 📚 TÀI LIỆU THAM KHẢO

Xem file chi tiết: `INPUT_ANALYSIS_AND_TRAINING_STEPS.md` để biết:
- Code examples đầy đủ
- Giải thích chi tiết từng bước
- Troubleshooting guide
