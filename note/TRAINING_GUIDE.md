# Hướng dẫn Train GAT-ETM trên Dữ liệu EHR Public

Hướng dẫn chi tiết từng bước để train mô hình GAT-ETM (Graph Attention-Embedded Topic Model) trên dữ liệu EHR công khai, tương tự như dataset PopEHR được sử dụng trong paper gốc.

## Mục lục

1. [Cài đặt Môi trường](#1-cài-đặt-môi-trường)
2. [Chuẩn bị Dữ liệu EHR](#2-chuẩn-bị-dữ-liệu-ehr)
3. [Xây dựng Knowledge Graph (ICD-ATC)](#3-xây-dựng-knowledge-graph-icd-atc)
4. [Tạo Code Embeddings (Node2Vec)](#4-tạo-code-embeddings-node2vec)
5. [Chuẩn bị Dữ liệu BoW Format](#5-chuẩn-bị-dữ-liệu-bow-format)
6. [Tạo Metadata File](#6-tạo-metadata-file)
7. [Train Model](#7-train-model)
8. [Evaluate Model](#8-evaluate-model)

---

## 1. Cài đặt Môi trường

### 1.1. Cài đặt Python Dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\Scripts\activate  # Windows

# Cài đặt các package cần thiết
pip install torch torch-geometric
pip install numpy scipy scikit-learn
pip install networkx node2vec
pip install tqdm pandas
pip install python-louvain leidenalg
pip install icdcodex  # Để xử lý ICD codes
```

### 1.2. Cài đặt các Dependencies Bổ sung

```bash
# Nếu cần xử lý dữ liệu MIMIC-III
pip install pyarrow  # Để đọc parquet files

# Để visualize (tùy chọn)
pip install matplotlib seaborn
```

### 1.3. Tạo các File Utils Cần thiết

Tạo file `utils.py` với các hàm hỗ trợ:

```python
# utils.py
import numpy as np
from scipy.sparse import coo_matrix
from collections import Counter

def nearest_neighbors(X, k=10):
    """Tìm k nearest neighbors"""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices[:, 1:], distances[:, 1:]

def get_topic_coherence(beta, train_data, top_n=10):
    """
    Tính Topic Coherence
    
    Args:
        beta: topic-word distribution (K x V)
        train_data: scipy sparse matrix (N x V)
        top_n: số từ top để tính coherence
    """
    K, V = beta.shape
    coherence_scores = []
    
    for k in range(K):
        # Lấy top_n từ có xác suất cao nhất
        top_words = np.argsort(beta[k])[-top_n:][::-1]
        
        # Tính coherence cho topic k
        coherence_k = 0
        count = 0
        
        for i, word_i in enumerate(top_words):
            for j, word_j in enumerate(top_words[i+1:], start=i+1):
                # Đếm số documents chứa cả word_i và word_j
                docs_with_i = train_data[:, word_i].toarray().flatten() > 0
                docs_with_j = train_data[:, word_j].toarray().flatten() > 0
                docs_with_both = docs_with_i & docs_with_j
                
                if docs_with_j.sum() > 0:
                    coherence_k += np.log((docs_with_both.sum() + 1) / docs_with_j.sum())
                    count += 1
        
        if count > 0:
            coherence_scores.append(coherence_k / count)
        else:
            coherence_scores.append(0)
    
    return np.mean(coherence_scores)

def get_topic_diversity(beta, top_n=10):
    """
    Tính Topic Diversity
    
    Args:
        beta: topic-word distribution (K x V)
        top_n: số từ top để tính diversity
    """
    K, V = beta.shape
    unique_words = set()
    
    for k in range(K):
        top_words = np.argsort(beta[k])[-top_n:][::-1]
        unique_words.update(top_words)
    
    diversity = len(unique_words) / (K * top_n)
    return diversity
```

---

## 2. Chuẩn bị Dữ liệu EHR

### 2.1. Download Public EHR Dataset

Có thể sử dụng các dataset công khai sau:

- **MIMIC-III**: https://mimic.mit.edu/ (cần đăng ký và hoàn thành training)
- **eICU**: https://eicu-crd.mit.edu/ (cần đăng ký)
- **MIMIC-IV**: https://mimic.mit.edu/iv/ (phiên bản mới hơn)
- **Synthetic EHR data**: Có thể tạo từ các công cụ như Synthea

**Lưu ý**: Các dataset này thường yêu cầu đăng ký và training về privacy. Đảm bảo tuân thủ các quy định.

### 2.2. Extract ICD và ATC Codes

Tạo script `prepare_ehr_data.py`:

```python
# prepare_ehr_data.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from collections import defaultdict

def extract_codes_from_mimic(data_path):
    """
    Extract ICD và ATC codes từ MIMIC-III dataset
    
    Args:
        data_path: đường dẫn đến thư mục chứa MIMIC-III data
    """
    # Đọc diagnoses (ICD codes)
    diagnoses_df = pd.read_csv(f'{data_path}/DIAGNOSES_ICD.csv')
    
    # Đọc prescriptions (ATC codes)
    prescriptions_df = pd.read_csv(f'{data_path}/PRESCRIPTIONS.csv')
    
    # Tạo mapping từ HADM_ID đến codes
    patient_codes = defaultdict(lambda: {'icd': [], 'atc': []})
    
    # Extract ICD codes
    for _, row in diagnoses_df.iterrows():
        hadm_id = row['HADM_ID']
        icd_code = row['ICD9_CODE']  # hoặc ICD10_CODE
        if pd.notna(icd_code):
            patient_codes[hadm_id]['icd'].append(str(icd_code))
    
    # Extract ATC codes (cần map từ drug name sang ATC)
    # Có thể sử dụng thư viện như pyMedTermino hoặc manual mapping
    for _, row in prescriptions_df.iterrows():
        hadm_id = row['HADM_ID']
        drug_name = row['DRUG']
        # Map drug_name to ATC code (cần implement mapping logic)
        atc_code = map_drug_to_atc(drug_name)
        if atc_code:
            patient_codes[hadm_id]['atc'].append(atc_code)
    
    return patient_codes

def create_vocabulary(patient_codes):
    """
    Tạo vocabulary từ patient codes
    """
    all_icd = set()
    all_atc = set()
    
    for codes in patient_codes.values():
        all_icd.update(codes['icd'])
        all_atc.update(codes['atc'])
    
    vocab_icd = sorted(list(all_icd))
    vocab_atc = sorted(list(all_atc))
    
    # Tạo mapping từ code sang index
    code_to_idx = {}
    idx = 0
    for code in vocab_icd:
        code_to_idx[code] = idx
        idx += 1
    for code in vocab_atc:
        code_to_idx[code] = idx
        idx += 1
    
    return vocab_icd, vocab_atc, code_to_idx

def create_bow_matrix(patient_codes, code_to_idx, vocab_icd, vocab_atc):
    """
    Tạo Bag-of-Words matrix từ patient codes
    """
    num_patients = len(patient_codes)
    vocab_size = len(vocab_icd) + len(vocab_atc)
    
    rows, cols, data = [], [], []
    
    for patient_idx, (hadm_id, codes) in enumerate(patient_codes.items()):
        # ICD codes
        for icd in codes['icd']:
            if icd in code_to_idx:
                rows.append(patient_idx)
                cols.append(code_to_idx[icd])
                data.append(1.0)
        
        # ATC codes
        for atc in codes['atc']:
            if atc in code_to_idx:
                rows.append(patient_idx)
                cols.append(code_to_idx[atc])
                data.append(1.0)
    
    bow_matrix = csr_matrix((data, (rows, cols)), shape=(num_patients, vocab_size))
    return bow_matrix

def split_train_test(bow_matrix, train_ratio=0.8):
    """
    Chia dữ liệu thành train và test
    """
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

# Main execution
if __name__ == '__main__':
    # Điều chỉnh đường dẫn theo dataset của bạn
    data_path = 'path/to/mimic/data'
    
    # Extract codes
    print("Extracting codes from EHR data...")
    patient_codes = extract_codes_from_mimic(data_path)
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab_icd, vocab_atc, code_to_idx = create_vocabulary(patient_codes)
    
    # Save vocabularies
    with open('icd_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_icd, f)
    with open('atc_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_atc, f)
    
    print(f"ICD vocabulary size: {len(vocab_icd)}")
    print(f"ATC vocabulary size: {len(vocab_atc)}")
    
    # Create BoW matrices
    print("Creating BoW matrices...")
    bow_matrix = create_bow_matrix(patient_codes, code_to_idx, vocab_icd, vocab_atc)
    
    # Split train/test
    bow_train, bow_test, bow_test_1, bow_test_2 = split_train_test(bow_matrix)
    
    # Save BoW matrices
    np.save('data/bow_train.npy', bow_train, allow_pickle=True)
    np.save('data/bow_test.npy', bow_test, allow_pickle=True)
    np.save('data/bow_test_1.npy', bow_test_1, allow_pickle=True)
    np.save('data/bow_test_2.npy', bow_test_2, allow_pickle=True)
    
    print("Data preparation completed!")
```

**Lưu ý**: Cần implement hàm `map_drug_to_atc()` để map drug names sang ATC codes. Có thể sử dụng:
- WHO ATC/DDD Index: https://www.whocc.no/atc_ddd_index/
- RxNorm API: https://www.nlm.nih.gov/research/umls/rxnorm/
- Manual mapping file

---

## 3. Xây dựng Knowledge Graph (ICD-ATC)

### 3.1. Download ICD và ATC Hierarchies

```bash
# Tạo thư mục để lưu graph data
mkdir -p graph_data
```

### 3.2. Tạo Script Build Knowledge Graph

Tạo file `build_knowledge_graph.py`:

```python
# build_knowledge_graph.py
import networkx as nx
import pickle
import pandas as pd
from icdcodex import hierarchy

def build_icd_tree():
    """
    Xây dựng ICD-9 hierarchy tree
    """
    icd_tree = hierarchy.icd9()[0]  # Sử dụng icdcodex library
    
    # Hoặc tự build từ file:
    # icd_tree = nx.DiGraph()
    # # Load ICD hierarchy từ file
    # with open('icd9_hierarchy.csv', 'r') as f:
    #     for line in f:
    #         parent, child = line.strip().split(',')
    #         icd_tree.add_edge(parent, child)
    
    return icd_tree

def build_atc_tree():
    """
    Xây dựng ATC hierarchy tree từ WHO ATC classification
    """
    atc_tree = nx.DiGraph()
    atc_tree.add_node('root')
    
    # Load ATC hierarchy từ file
    # Format: parent_code,child_code
    # Ví dụ: A,A01 (A01 là con của A)
    try:
        with open('atc_hierarchy.csv', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    parent, child = parts
                    atc_tree.add_edge(parent, child)
    except FileNotFoundError:
        print("Warning: atc_hierarchy.csv not found. Creating minimal ATC tree.")
        # Tạo minimal tree từ ATC codes trong vocabulary
        with open('atc_vocab.pkl', 'rb') as f:
            vocab_atc = pickle.load(f)
        
        for code in vocab_atc:
            # ATC codes có format: A01AA01 (level 1-5)
            if len(code) >= 1:
                atc_tree.add_edge('root', code[0])
            if len(code) >= 3:
                atc_tree.add_edge(code[0], code[:3])
            if len(code) >= 4:
                atc_tree.add_edge(code[:3], code[:4])
            if len(code) >= 5:
                atc_tree.add_edge(code[:4], code[:5])
            if len(code) >= 7:
                atc_tree.add_edge(code[:5], code)
    
    return atc_tree

def load_atc_to_icd_mapping():
    """
    Load mapping từ ATC sang ICD (drug-disease relationships)
    """
    # Có thể tạo từ:
    # 1. Medical knowledge bases (DrugBank, PharmGKB)
    # 2. Literature mining
    # 3. Manual curation
    
    atc2icd = []
    
    try:
        # Nếu có file mapping
        df = pd.read_csv('atc_to_icd_mapping.csv')
        for _, row in df.iterrows():
            atc2icd.append([row['ATC'], row['ICD'], row['ICD_DESC']])
    except FileNotFoundError:
        print("Warning: atc_to_icd_mapping.csv not found.")
        print("Creating empty mapping. You may need to add drug-disease relationships manually.")
    
    return atc2icd

def augment_graph(G, icd_tree, atc_tree):
    """
    Augment graph với các edges từ leaf nodes đến ancestors
    """
    e = 0.9  # Decay factor
    
    # Augment ICD tree
    for node in icd_tree.nodes():
        if len(list(icd_tree.neighbors(node))) == 0:  # Leaf node
            ancestors = list(icd_tree.predecessors(node))
            if ancestors:
                parent = ancestors[0]
                w = e
                while len(list(icd_tree.predecessors(parent))) > 0:
                    grandparent = list(icd_tree.predecessors(parent))[0]
                    G.add_edge(node, grandparent, weight=w)
                    w *= e
                    parent = grandparent
    
    # Augment ATC tree
    for node in atc_tree.nodes():
        if len(list(atc_tree.neighbors(node))) == 0:  # Leaf node
            ancestors = list(atc_tree.predecessors(node))
            if ancestors:
                parent = ancestors[0]
                w = e
                while len(list(atc_tree.predecessors(parent))) > 0:
                    grandparent = list(atc_tree.predecessors(parent))[0]
                    G.add_edge(node, grandparent, weight=w)
                    w *= e
                    parent = grandparent
    
    return G

def build_knowledge_graph(augmented=True):
    """
    Xây dựng knowledge graph kết hợp ICD và ATC
    """
    print("Building ICD tree...")
    icd_tree = build_icd_tree()
    nx.set_node_attributes(icd_tree, name='type', values='ICD9')
    
    print("Building ATC tree...")
    atc_tree = build_atc_tree()
    nx.set_node_attributes(atc_tree, name='type', values='ATC')
    
    print("Combining trees...")
    G = nx.compose(icd_tree, atc_tree)
    
    print("Loading ATC-ICD mappings...")
    atc2icd = load_atc_to_icd_mapping()
    
    print("Adding ATC-ICD edges...")
    for row in atc2icd:
        atc_code, icd_code, icd_desc = row[0], row[1], row[2] if len(row) > 2 else None
        
        # Try to find ICD code in graph
        if icd_code in G:
            if atc_code in G:
                G.add_edge(atc_code, icd_code)
        elif icd_desc and icd_desc in G:
            if atc_code in G:
                G.add_edge(atc_code, icd_desc)
    
    # Convert to undirected
    G = G.to_undirected()
    
    # Augment graph if needed
    if augmented:
        print("Augmenting graph...")
        G = augment_graph(G, icd_tree, atc_tree)
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

if __name__ == '__main__':
    # Build graph
    G = build_knowledge_graph(augmented=True)
    
    # Save graph
    with open('graph_data/icdatc_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    
    print("Knowledge graph saved to graph_data/icdatc_graph.pkl")
```

### 3.3. Tạo các File Input Cần thiết

**atc_hierarchy.csv**: Format
```
root,A
root,B
A,A01
A01,A01A
A01A,A01AA
A01AA,A01AA01
```

**atc_to_icd_mapping.csv**: Format
```
ATC,ICD,ICD_DESC
A10BA02,250,Diabetes mellitus
A10BA02,250.0,Diabetes mellitus without mention of complication
```

---

## 4. Tạo Code Embeddings (Node2Vec)

### 4.1. Tạo Script Generate Embeddings

Tạo file `generate_embeddings.py`:

```python
# generate_embeddings.py
import networkx as nx
import numpy as np
import pickle
from node2vec import Node2Vec
from sklearn.decomposition import PCA

def generate_node2vec_embeddings(G, dimensions=256, walk_length=20, num_walks=10, window=8):
    """
    Tạo embeddings cho các nodes trong graph sử dụng Node2Vec
    """
    print("Initializing Node2Vec...")
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=4,  # Số CPU cores
        p=1,  # Return parameter
        q=1   # In-out parameter
    )
    
    print("Training Node2Vec model...")
    model = node2vec.fit(window=window, min_count=1, batch_words=4)
    
    print("Extracting embeddings...")
    embeddings = np.zeros((len(G.nodes()), dimensions))
    node_to_idx = {}
    
    for idx, node in enumerate(G.nodes()):
        node_to_idx[node] = idx
        if node in model.wv:
            embeddings[idx] = model.wv[node]
        else:
            # Random initialization nếu node không có trong model
            embeddings[idx] = np.random.normal(0, 0.1, dimensions)
    
    return embeddings, node_to_idx

def align_embeddings_with_vocab(embeddings, node_to_idx, vocab_icd, vocab_atc):
    """
    Align embeddings với vocabulary order
    """
    import pickle
    
    # Load vocabularies
    with open('icd_vocab.pkl', 'rb') as f:
        vocab_icd = pickle.load(f)
    with open('atc_vocab.pkl', 'rb') as f:
        vocab_atc = pickle.load(f)
    
    # Tạo renumbering mapping
    renumber = {}
    idx = 0
    
    # Map ICD codes
    for code in vocab_icd:
        # Normalize ICD code format
        normalized_code = normalize_icd_code(code)
        if normalized_code in node_to_idx:
            renumber[normalized_code] = idx
        idx += 1
    
    # Map ATC codes
    for code in vocab_atc:
        if code in node_to_idx:
            renumber[code] = idx
        idx += 1
    
    # Tạo aligned embeddings
    V = len(vocab_icd) + len(vocab_atc)
    aligned_embeddings = np.zeros((V, embeddings.shape[1]))
    
    for node, graph_idx in node_to_idx.items():
        if node in renumber:
            vocab_idx = renumber[node]
            aligned_embeddings[vocab_idx] = embeddings[graph_idx]
    
    return aligned_embeddings, renumber

def normalize_icd_code(code):
    """
    Normalize ICD code format (thêm/xóa dots)
    """
    # Remove dots
    code = str(code).replace('.', '')
    
    # Add dots at appropriate positions (ICD-9 format: XXX.XX)
    if len(code) >= 3:
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code
    return code

if __name__ == '__main__':
    # Load graph
    print("Loading knowledge graph...")
    with open('graph_data/icdatc_graph.pkl', 'rb') as f:
        G = pickle.load(f)
    
    # Generate embeddings
    print("Generating Node2Vec embeddings...")
    embeddings, node_to_idx = generate_node2vec_embeddings(
        G,
        dimensions=256,
        walk_length=20,
        num_walks=10,
        window=8
    )
    
    # Align with vocabulary
    print("Aligning embeddings with vocabulary...")
    aligned_embeddings, renumber = align_embeddings_with_vocab(
        embeddings, node_to_idx, None, None
    )
    
    # Save embeddings
    np.save('embed/icdatc_embed_8_20_10_256_by_vocab.npy', aligned_embeddings)
    
    # Save renumbered graph
    G_renumbered = nx.relabel_nodes(G, {v: k for k, v in renumber.items()})
    with open('embed/icdatc_graph_256_renumbered_by_vocab.pkl', 'wb') as f:
        pickle.dump(G_renumbered, f)
    
    print("Embeddings saved to embed/icdatc_embed_8_20_10_256_by_vocab.npy")
    print("Renumbered graph saved to embed/icdatc_graph_256_renumbered_by_vocab.pkl")
```

---

## 5. Chuẩn bị Dữ liệu BoW Format

Đảm bảo bạn đã chạy script `prepare_ehr_data.py` ở bước 2 để tạo các file:
- `data/bow_train.npy`
- `data/bow_test.npy`
- `data/bow_test_1.npy`
- `data/bow_test_2.npy`

Các file này phải là scipy sparse matrices (CSR format) được lưu dưới dạng numpy array với `allow_pickle=True`.

---

## 6. Tạo Metadata File

Tạo file metadata trong thư mục `data/` với format:

**data/metadata.txt**:
```
icd atc
61210 82020
1 1
embed/icdatc_embed_8_20_10_256_by_vocab.npy embed/icdatc_embed_8_20_10_256_by_vocab.npy
```

Giải thích:
- Dòng 1: Tên các loại codes (space-separated)
- Dòng 2: Kích thước vocabulary của mỗi loại code
- Dòng 3: Flag train embeddings (1 = train, 0 = freeze)
- Dòng 4: Đường dẫn đến file embeddings cho mỗi loại code (hoặc `*` nếu không có)

**Lưu ý**: Điều chỉnh các số liệu theo dataset của bạn:
- `61210`: số lượng ICD codes trong vocabulary
- `82020`: số lượng ATC codes trong vocabulary

---

## 7. Train Model

### 7.1. Cấu trúc Thư mục

Đảm bảo cấu trúc thư mục như sau:

```
GAT-ETM/
├── data/
│   ├── metadata.txt
│   ├── bow_train.npy
│   ├── bow_test.npy
│   ├── bow_test_1.npy
│   └── bow_test_2.npy
├── embed/
│   ├── icdatc_embed_8_20_10_256_by_vocab.npy
│   └── icdatc_graph_256_renumbered_by_vocab.pkl
├── results/
│   └── (sẽ được tạo khi train)
├── main_getm.py
├── graph_etm.py
├── dataset.py
└── utils.py
```

### 7.2. Chạy Training

```bash
python3 main_getm.py \
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
    --theta_act relu
```

### 7.3. Các Hyperparameters Quan trọng

- `--num_topics`: Số lượng topics (50-100 thường tốt)
- `--t_hidden_size`: Kích thước hidden layer cho encoder (128-256)
- `--rho_size`: Kích thước embeddings (256-512)
- `--epochs`: Số epochs (50-150)
- `--lr`: Learning rate (0.001-0.01)
- `--batch_size`: Batch size (512-1024 tùy GPU memory)
- `--tq`: Tính Topic Quality (coherence + diversity)

### 7.4. Train với Drug Imputation

Nếu muốn train với drug imputation task:

```bash
python3 main_getm.py \
    --data_path data/ \
    --save_path results/ \
    --meta_file metadata \
    --mode train \
    --drug_imputation \
    --dc_thr 5 \
    --impute_k 5 \
    --loss wkl \
    --gamma 1.0 \
    --gpu_device 0 \
    --num_topics 50 \
    --epochs 50 \
    --lr 0.01
```

---

## 8. Evaluate Model

### 8.1. Evaluate Trained Model

```bash
python3 main_getm.py \
    --data_path data/ \
    --save_path results/ \
    --meta_file metadata \
    --mode eval \
    --load_from results/getm_UKPD_K_50_Htheta_128_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.01_Bsz_512_RhoSize_256_trainEmbeddings_[1, 1].mdl \
    --gpu_device 0 \
    --num_topics 50 \
    --tq
```

### 8.2. Kết quả Evaluation

Model sẽ output:
- **Negative Log Likelihood (NLL)**: Loss trên test set
- **Topic Coherence (TC)**: Độ nhất quán của topics
- **Topic Diversity (TD)**: Độ đa dạng của topics
- **Topic Quality (TQ)**: TC × TD
- **Drug Imputation Metrics** (nếu có): Accuracy@k, Precision@k, Recall@k, F1@k

### 8.3. Xem Embeddings và Topics

Sau khi train/eval, các file sau sẽ được lưu trong `results/`:
- `beta_icd.npy`: Topic-word distribution cho ICD codes
- `beta_atc.npy`: Topic-word distribution cho ATC codes
- `rho_icd.npy`: Embeddings của ICD codes
- `rho_atc.npy`: Embeddings của ATC codes
- `alpha_icd.npy` / `alpha_atc.npy`: Topic embeddings
- `theta_train/`: Topic distributions cho training set
- `theta_test/`: Topic distributions cho test set

---

## Troubleshooting

### Lỗi thường gặp:

1. **File not found errors**: Đảm bảo tất cả các file đã được tạo đúng đường dẫn
2. **Memory errors**: Giảm `batch_size` hoặc `num_topics`
3. **CUDA out of memory**: Giảm `batch_size` hoặc sử dụng CPU
4. **Graph nodes không match vocabulary**: Kiểm tra lại việc align embeddings với vocabulary

### Tips:

- Bắt đầu với dataset nhỏ để test pipeline
- Monitor training loss để điều chỉnh learning rate
- Sử dụng `--tq` để đánh giá chất lượng topics
- Thử các hyperparameters khác nhau để tối ưu performance

---

## Tài liệu Tham khảo

- Paper gốc: https://www.nature.com/articles/s41598-022-22956-w
- MIMIC-III: https://mimic.mit.edu/
- ICD-9 Codes: https://icdlist.com/icd-9/index
- ATC Codes: https://www.whocc.no/atc_ddd_index/
- Node2Vec: https://github.com/eliorc/node2vec

---

## Kết luận

Hướng dẫn này cung cấp các bước chi tiết để train GAT-ETM từ đầu. Quá trình bao gồm:
1. Chuẩn bị dữ liệu EHR
2. Xây dựng knowledge graph
3. Tạo embeddings
4. Train và evaluate model

Chúc bạn thành công!

