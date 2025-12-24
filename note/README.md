# Knowledge Graph Builder và Visualizer cho MIMIC-III

Tài liệu này giải thích các script để xây dựng và trực quan hóa Knowledge Graph từ dữ liệu MIMIC-III cho mô hình GAT-ETM.

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [build_kg_mimic.py](#build_kg_mimicpy)
3. [build_kg_mimic_sample.py](#build_kg_mimic_samplepy)
4. [visualize_graph.py](#visualize_graphpy)
5. [Cách sử dụng](#cách-sử-dụng)
6. [Output files](#output-files)

---

## Tổng quan

Hệ thống này xây dựng Knowledge Graph từ dữ liệu MIMIC-III với các thành phần:

- **ICD9 codes**: Mã chẩn đoán (diagnosis codes)
- **CPT codes**: Mã thủ thuật (procedure codes)
- **ATC codes**: Mã thuốc (drug codes)
- **Lab codes**: Mã xét nghiệm (laboratory codes)

Knowledge Graph bao gồm:
- **Hierarchical edges**: Các cạnh phân cấp (cây phân cấp)
- **Co-occurrence edges**: Các cạnh đồng xuất hiện (từ cùng một admission)
- **Augmented edges**: Các cạnh bổ sung (skip connections)

---

## build_kg_mimic.py

### Mục đích

Script chính để xây dựng Knowledge Graph từ toàn bộ dữ liệu MIMIC-III.

### Class: `MIMIC_KG_Builder`

#### Khởi tạo

```python
builder = MIMIC_KG_Builder(
    mimic_path='mimic-iii-clinical-database-demo-1.4',
    output_dir='embed',
    embedding_dim=256,
    window=8,
    walk_length=20,
    num_walks=10,
    augmented=True
)
```

**Tham số:**
- `mimic_path`: Đường dẫn đến thư mục chứa dữ liệu MIMIC-III
- `output_dir`: Thư mục lưu output files
- `embedding_dim`: Kích thước embedding vector (mặc định: 256)
- `window`: Kích thước window cho Node2Vec (mặc định: 8)
- `walk_length`: Độ dài mỗi random walk (mặc định: 20)
- `num_walks`: Số lượng walks cho mỗi node (mặc định: 10)
- `augmented`: Có thêm augmented edges hay không (mặc định: True)

#### Các phương thức chính

##### 1. `load_mimic_tables()`

**Chức năng:** Load tất cả các bảng cần thiết từ MIMIC-III

**Các bảng được load:**
- `D_ICD_DIAGNOSES.csv`: Dictionary cho ICD9 diagnoses
- `DIAGNOSES_ICD.csv`: Dữ liệu chẩn đoán
- `D_ICD_PROCEDURES.csv`: Dictionary cho ICD9 procedures
- `PROCEDURES_ICD.csv`: Dữ liệu thủ thuật
- `D_CPT.csv`: Dictionary cho CPT codes (nếu có)
- `CPTEVENTS.csv`: Dữ liệu CPT events (nếu có)
- `PRESCRIPTIONS.csv`: Dữ liệu đơn thuốc
- `LABEVENTS.csv`: Dữ liệu xét nghiệm
- `D_LABITEMS.csv`: Dictionary cho lab items
- `ADMISSIONS.csv`: Dữ liệu nhập viện

**Lưu ý:** Script này sử dụng **toàn bộ dữ liệu**, không có sampling.

##### 2. `build_icd9_hierarchy()`

**Chức năng:** Xây dựng cây phân cấp cho ICD9 codes

**Cách hoạt động:**
1. Lấy tất cả các ICD9 codes unique từ `DIAGNOSES_ICD`
2. Normalize codes (bỏ dấu chấm nếu có)
3. Xây dựng hierarchy dựa trên prefix:
   - Code "25010" → tạo nodes: "250", "2501", "25010"
   - Tạo edges: `ICD9_ROOT` → "250" → "2501" → "25010"
4. Lưu vocabulary: tất cả các ICD9 codes (leaf nodes)

**Ví dụ:**
```
ICD9_ROOT
  └── 250 (3-digit)
       └── 2501 (4-digit)
            └── 25010 (5-digit)
```

##### 3. `build_cpt_hierarchy()`

**Chức năng:** Xây dựng cây phân cấp cho CPT codes

**Cách hoạt động:**
1. Lấy các CPT codes từ `CPTEVENTS` (nếu có)
2. Nhóm theo ký tự đầu tiên
3. Tạo hierarchy: `CPT_ROOT` → `CPT_X` → code cụ thể

**Ví dụ:**
```
CPT_ROOT
  └── CPT_1
       └── 12345
```

##### 4. `extract_drugs_and_map_to_atc()`

**Chức năng:** Trích xuất thuốc và map sang ATC codes

**Cách hoạt động:**
1. Lấy tất cả các drugs unique từ `PRESCRIPTIONS`
2. Tạo placeholder ATC codes (hash-based, không phải ATC thật)
3. Xây dựng hierarchy: `ATC_ROOT` → intermediate levels → ATC code

**Lưu ý:** 
- Script sử dụng placeholder ATC codes (hash-based)
- Trong production, nên sử dụng RxNorm API hoặc ATC mapping file thật

**Ví dụ:**
```
ATC_ROOT
  └── DRUG_
       └── DRUG_1
            └── DRUG_12345
```

##### 5. `extract_lab_codes()`

**Chức năng:** Trích xuất lab codes và xây dựng hierarchy

**Cách hoạt động:**
1. Lấy tất cả các lab itemids unique từ `LABEVENTS`
2. Nhóm theo category từ `D_LABITEMS`
3. Tạo hierarchy: `LAB_ROOT` → `LAB_{category}` → lab code

**Ví dụ:**
```
LAB_ROOT
  └── LAB_CHEMISTRY
       └── 50809
```

##### 6. `build_cooccurrence_edges()`

**Chức năng:** Xây dựng các cạnh đồng xuất hiện từ admissions

**Cách hoạt động:**
1. Nhóm tất cả codes theo `hadm_id` (hospital admission ID)
2. Với mỗi admission, tạo edges giữa:
   - ICD9 ↔ ATC (chẩn đoán và thuốc)
   - ICD9 ↔ CPT (chẩn đoán và thủ thuật)
   - ICD9 ↔ Lab (chẩn đoán và xét nghiệm)
   - ICD9 ↔ ICD9 (nhiều chẩn đoán cùng lúc)
3. Edge weight = số lần đồng xuất hiện
4. Chỉ thêm edge nếu `count >= min_cooccurrence`

**Tham số:**
- `min_cooccurrence`: Ngưỡng tối thiểu để tạo edge (mặc định: 1)

##### 7. `augment_graph()`

**Chức năng:** Thêm augmented edges (skip connections)

**Cách hoạt động:**
1. Với mỗi leaf node trong hierarchy (ICD9, ATC, CPT)
2. Tìm tất cả ancestors
3. Thêm edge trực tiếp từ leaf node đến mỗi ancestor
4. Weight giảm dần: `weight = 0.9^(i+1)` với i là khoảng cách

**Mục đích:** Giúp mô hình học được các mối quan hệ xa trong hierarchy

**Ví dụ:**
```
ICD9_ROOT
  └── 250
       └── 2501
            └── 25010
```

Augmented edges:
- `25010` → `ICD9_ROOT` (weight = 0.9^3)
- `25010` → `250` (weight = 0.9^2)
- `2501` → `ICD9_ROOT` (weight = 0.9^2)

##### 8. `generate_embeddings()`

**Chức năng:** Sinh embeddings bằng Node2Vec

**Cách hoạt động:**
1. Convert graph sang undirected
2. Khởi tạo Node2Vec với các tham số:
   - `dimensions`: embedding_dim
   - `walk_length`: walk_length
   - `num_walks`: num_walks
   - `p=1, q=1`: return và in-out parameters
3. Train model với window size
4. Trích xuất embeddings cho tất cả nodes

**Fallback:** Nếu Node2Vec không có, sử dụng random embeddings (normalized)

##### 9. `renumber_nodes_by_vocab()`

**Chức năng:** Đánh số lại nodes theo thứ tự vocabulary

**Cách hoạt động:**
1. Đánh số lại nodes theo thứ tự:
   - ICD9 vocab nodes
   - CPT vocab nodes
   - ATC vocab nodes
   - Lab vocab nodes
   - Các nodes khác (hierarchical nodes, roots, etc.)
2. Tạo mapping: `graphnode_vocab` (new_index → old_node_name)
3. Relabel graph và reorder embeddings

**Mục đích:** Đảm bảo vocab nodes có index liên tục, dễ sử dụng trong training

##### 10. `save_outputs()`

**Chức năng:** Lưu graph, embeddings, và vocab mappings

**Output files:**
- Graph pickle file
- Embeddings pickle file
- `graphnode_vocab.pkl`: mapping từ index → node name
- `vocab_info.pkl`: vocabularies cho ICD9, CPT, ATC, Lab

##### 11. `build()`

**Chức năng:** Hàm chính điều phối toàn bộ quy trình

**Quy trình:**
1. Load tables
2. Build hierarchies (ICD9, CPT, ATC, Lab)
3. Build co-occurrence edges
4. Augment graph (nếu enabled)
5. Generate embeddings
6. Renumber nodes
7. Save outputs

---

## build_kg_mimic_sample.py

### Mục đích

Script tương tự `build_kg_mimic.py` nhưng sử dụng **sample mode** - chỉ sử dụng một phần nhỏ dữ liệu để tạo Knowledge Graph nhỏ hơn, phù hợp cho testing và development.

### Khác biệt so với `build_kg_mimic.py`

#### 1. Sampling dữ liệu (`load_mimic_tables()`)

**Dòng 128-142:**
```python
# SAMPLE MODE: Chỉ lấy 5 subjects đầu tiên
NUM_SAMPLE_SUBJECTS = 5
SAMPLE_SUBJECTS = set(self.admissions['subject_id'].dropna().unique()[:NUM_SAMPLE_SUBJECTS])

# Filter tất cả tables theo SAMPLE_SUBJECTS
self.admissions = self.admissions[self.admissions['subject_id'].isin(SAMPLE_SUBJECTS)]
self.diagnoses_icd = self.diagnoses_icd[self.diagnoses_icd['subject_id'].isin(SAMPLE_SUBJECTS)]
# ... tương tự cho các tables khác
```

**Kết quả:** Chỉ sử dụng dữ liệu từ 5 subjects đầu tiên.

#### 2. Simple Mode cho ICD9 (`build_icd9_hierarchy()`)

**Dòng 145-245:**
- **Chỉ chọn 1 ICD-9 code phổ biến nhất** (thay vì tất cả)
- Xây dựng hierarchy chỉ cho code này

**Ví dụ:**
```python
# Lấy top 1 ICD9 code
top_icd9_code = icd9_counts.most_common(1)[0][0]
# Chỉ xây hierarchy cho code này
```

#### 3. Simple Mode cho ATC (`extract_drugs_and_map_to_atc()`)

**Dòng 303-377:**
- **Chỉ chọn top 5 drugs phổ biến nhất** (thay vì tất cả)

```python
top_drugs = [drug for drug, count in drug_counts.most_common(5)]
```

#### 4. Simple Mode cho Lab (`extract_lab_codes()`)

**Dòng 379-433:**
- **Chỉ chọn top 10 lab codes phổ biến nhất** (thay vì tất cả)

```python
top_labs = [itemid for itemid, count in lab_counts.most_common(10)]
```

#### 5. Giới hạn Co-occurrence (`build_cooccurrence_edges()`)

**Dòng 516-518:**
```python
MAX_ADMISSIONS = 10  # Chỉ xử lý 10 admissions đầu tiên
hadm_list = list(hadm_to_icd.keys())[:MAX_ADMISSIONS]
```

#### 6. Ego-graph Creation (`create_ego_graph()`)

**Dòng 563-619:**
- Tạo ego-graph từ các nodes đã chọn (vocab nodes)
- Bao gồm: vocab nodes + neighbors (hierarchy nodes) + root nodes
- **Mục đích:** Giảm kích thước graph, chỉ giữ phần liên quan

**Cách hoạt động:**
1. Lấy tất cả vocab nodes (ICD9, Lab, ATC đã chọn)
2. Thêm tất cả neighbors của chúng (hierarchy nodes)
3. Thêm root nodes
4. Tạo subgraph từ các nodes này

**Kết quả:** Graph nhỏ hơn nhiều, chỉ chứa các nodes liên quan đến vocab đã chọn.

### Khi nào sử dụng

- **build_kg_mimic.py**: Khi cần Knowledge Graph đầy đủ từ toàn bộ dữ liệu
- **build_kg_mimic_sample.py**: Khi cần Knowledge Graph nhỏ để:
  - Testing và development
  - Debugging
  - Visualization (graph nhỏ dễ visualize hơn)
  - Quick experiments

---

## visualize_graph.py

### Mục đích

Script để trực quan hóa Knowledge Graph đã được tạo từ `build_kg_mimic.py` hoặc `build_kg_mimic_sample.py`.

### Class: `GraphVisualizer`

#### Khởi tạo

```python
visualizer = GraphVisualizer(
    graph_file='embed/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl',
    vocab_file='embed/graphnode_vocab.pkl',
    vocab_info_file='embed/vocab_info.pkl'
)
```

**Tham số:**
- `graph_file`: Đường dẫn đến graph pickle file
- `vocab_file`: Đường dẫn đến `graphnode_vocab.pkl` (optional)
- `vocab_info_file`: Đường dẫn đến `vocab_info.pkl` (optional)

#### Các phương thức chính

##### 1. `load_graph()`

**Chức năng:** Load graph và vocab files từ pickle

##### 2. `get_node_statistics()` và `print_statistics()`

**Chức năng:** In thống kê về graph

**Thông tin hiển thị:**
- Tổng số nodes và edges
- Phân bố node types (ICD9, CPT, ATC, LAB)
- Phân bố edge types (hierarchical, cooccurrence, augmented)
- Degree statistics (min, max, average, median)
- Vocabulary sizes

##### 3. `get_relationships()` và `print_relationships()`

**Chức năng:** Trích xuất và in các mối quan hệ giữa các node types

**Các loại relationships:**
- ICD9-ATC: Chẩn đoán và thuốc
- ICD9-LAB: Chẩn đoán và xét nghiệm
- ICD9-CPT: Chẩn đoán và thủ thuật
- ICD9-ICD9: Nhiều chẩn đoán cùng lúc
- ATC-LAB, ATC-CPT, LAB-CPT

**Output:** Danh sách các connections, sắp xếp theo weight (nếu có)

##### 4. `create_subgraph()`

**Chức năng:** Tạo subgraph để visualize (nếu graph quá lớn)

**Strategies:**
- `'random'`: Chọn ngẫu nhiên
- `'high_degree'`: Chọn nodes có degree cao nhất
- `'hierarchical'`: Chọn root nodes và neighbors của chúng

##### 5. `visualize_graph()`

**Chức năng:** Tạo visualization chính của graph

**Tham số:**
- `output_file`: Tên file output (mặc định: 'graph_visualization.png')
- `max_nodes`: Số nodes tối đa để visualize (mặc định: 500)
- `layout`: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'hierarchical')
- `node_size`: Kích thước nodes (mặc định: 50)
- `font_size`: Kích thước font cho labels (mặc định: 8)
- `show_labels`: Có hiển thị labels hay không

**Màu sắc:**
- **Node types:**
  - ICD9: Đỏ (#FF6B6B)
  - CPT: Teal (#4ECDC4)
  - ATC: Xanh dương (#45B7D1)
  - LAB: Light salmon (#FFA07A)
- **Edge types:**
  - Hierarchical: Dark blue-gray (#2C3E50), solid line
  - Co-occurrence: Đỏ (#E74C3C), dashed line
  - Augmented: Tím (#9B59B6), dotted line

**Labels:**
- Luôn hiển thị cho root nodes (level 0)
- Hiển thị cho vocab nodes (nodes trong vocabulary)
- Hiển thị cho high-degree nodes (degree > 3)

**Legend:**
- Hiển thị màu sắc cho node types và edge types

**Summary box:**
- Hiển thị summary về relationships ở góc dưới bên phải

##### 6. `visualize_by_type()`

**Chức năng:** Tạo các visualization riêng biệt cho từng node type

**Output:** Các file PNG riêng cho ICD9, CPT, ATC, LAB

##### 7. `_visualize_subgraph()`

**Chức năng:** Helper function để visualize một subgraph cụ thể

---

## Cách sử dụng

### 1. Build Knowledge Graph (Full Data)

```bash
cd KG_EMBED
python build_kg_mimic.py \
    --mimic_path ../mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed \
    --embedding_dim 256 \
    --window 8 \
    --walk_length 20 \
    --num_walks 10 \
    --augmented
```

### 2. Build Knowledge Graph (Sample Mode)

```bash
cd KG_EMBED
python build_kg_mimic_sample.py \
    --mimic_path ../mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed_simple \
    --embedding_dim 256 \
    --window 8 \
    --walk_length 20 \
    --num_walks 10 \
    --augmented
```

### 3. Visualize Graph

#### Chỉ in statistics:

```bash
cd visualize
python visualize_graph.py \
    --graph_file ../KG_EMBED/embed/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl \
    --vocab_file ../KG_EMBED/embed/graphnode_vocab.pkl \
    --vocab_info_file ../KG_EMBED/embed/vocab_info.pkl \
    --stats_only
```

#### Tạo visualization:

```bash
cd visualize
python visualize_graph.py \
    --graph_file ../KG_EMBED/embed/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl \
    --vocab_file ../KG_EMBED/embed/graphnode_vocab.pkl \
    --vocab_info_file ../KG_EMBED/embed/vocab_info.pkl \
    --output graph_visualization.png \
    --max_nodes 500 \
    --layout spring \
    --node_size 50 \
    --show_labels
```

#### Tạo visualization cho từng node type:

```bash
cd visualize
python visualize_graph.py \
    --graph_file ../KG_EMBED/embed/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl \
    --vocab_file ../KG_EMBED/embed/graphnode_vocab.pkl \
    --vocab_info_file ../KG_EMBED/embed/vocab_info.pkl \
    --by_type
```

---

## Output Files

### Từ `build_kg_mimic.py` hoặc `build_kg_mimic_sample.py`

#### 1. Graph Pickle File

**Tên file:** `[augmented_]icdatc_graph_{window}_{walk_length}_{num_walks}_{embedding_dim}_renumbered_by_vocab.pkl`

**Nội dung:** NetworkX Graph object đã được renumber

**Sử dụng:**
```python
import pickle
import networkx as nx

with open('embed/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl', 'rb') as f:
    G = pickle.load(f)

# G là NetworkX Graph với:
# - Nodes: integer indices (0, 1, 2, ...)
# - Node attributes: 'type', 'level', 'code'
# - Edge attributes: 'edge_type', 'weight'
```

#### 2. Embeddings Pickle File

**Tên file:** `[augmented_]icdatc_embed_{window}_{walk_length}_{num_walks}_{embedding_dim}_by_vocab.pkl`

**Nội dung:** NumPy array với shape `(num_nodes, embedding_dim)`

**Sử dụng:**
```python
import pickle
import numpy as np

with open('embed/icdatc_embed_8_20_10_256_by_vocab.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# embeddings.shape = (num_nodes, 256)
# embeddings[i] là embedding vector cho node i
```

#### 3. Graphnode Vocab (`graphnode_vocab.pkl`)

**Nội dung:** Dictionary mapping từ node index → node name

**Sử dụng:**
```python
import pickle

with open('embed/graphnode_vocab.pkl', 'rb') as f:
    graphnode_vocab = pickle.load(f)

# graphnode_vocab[0] = 'ICD9_ROOT' (ví dụ)
# graphnode_vocab[1] = '250' (ví dụ)
```

#### 4. Vocab Info (`vocab_info.pkl`)

**Nội dung:** Dictionary chứa vocabularies cho từng loại

**Sử dụng:**
```python
import pickle

with open('embed/vocab_info.pkl', 'rb') as f:
    vocab_info = pickle.load(f)

# vocab_info = {
#     'icd': ['250.10', '401.9', ...],
#     'cpt': ['12345', '67890', ...],
#     'atc': ['DRUG_12345', ...],
#     'lab': ['50809', '50810', ...]
# }
```

---

## Lưu ý quan trọng

### 1. ATC Codes

- Script sử dụng **placeholder ATC codes** (hash-based)
- Trong production, nên sử dụng:
  - RxNorm API để map drugs → RxNorm codes
  - ATC mapping file để map RxNorm → ATC codes

### 2. Sampling vs Full Data

- **build_kg_mimic.py**: Sử dụng toàn bộ dữ liệu → Graph lớn, mất nhiều thời gian
- **build_kg_mimic_sample.py**: Sử dụng sample → Graph nhỏ, nhanh hơn, phù hợp cho testing

### 3. Node2Vec

- Cần cài đặt: `pip install node2vec`
- Nếu không có, script sẽ sử dụng random embeddings (fallback)

### 4. Graph Size

- Với full data, graph có thể rất lớn (hàng nghìn nodes và edges)
- Nên sử dụng `max_nodes` khi visualize để tránh quá tải

### 5. Memory

- Với full data, cần đủ RAM (có thể cần > 8GB)
- Nếu thiếu RAM, nên sử dụng sample mode

---

## Ví dụ Workflow

### Workflow 1: Development và Testing

```bash
# 1. Build sample KG
python build_kg_mimic_sample.py \
    --mimic_path ../mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed_simple

# 2. Visualize sample KG
cd ../visualize
python visualize_graph.py \
    --graph_file ../KG_EMBED/embed_simple/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl \
    --vocab_file ../KG_EMBED/embed_simple/graphnode_vocab.pkl \
    --vocab_info_file ../KG_EMBED/embed_simple/vocab_info.pkl \
    --output graph_sample.png
```

### Workflow 2: Production

```bash
# 1. Build full KG (có thể mất vài giờ)
python build_kg_mimic.py \
    --mimic_path ../mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed \
    --embedding_dim 256

# 2. Kiểm tra statistics
cd ../visualize
python visualize_graph.py \
    --graph_file ../KG_EMBED/embed/icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl \
    --vocab_file ../KG_EMBED/embed/graphnode_vocab.pkl \
    --vocab_info_file ../KG_EMBED/embed/vocab_info.pkl \
    --stats_only
```

---

## Troubleshooting

### Lỗi: "FileNotFoundError: D_CPT.csv"

**Nguyên nhân:** CPT tables không có trong MIMIC-III demo

**Giải pháp:** Script sẽ tự động skip CPT hierarchy, không ảnh hưởng đến các phần khác

### Lỗi: "Memory Error"

**Nguyên nhân:** Graph quá lớn

**Giải pháp:** 
- Sử dụng `build_kg_mimic_sample.py` thay vì `build_kg_mimic.py`
- Giảm số lượng subjects trong sample mode

### Lỗi: "Node2Vec not installed"

**Nguyên nhân:** Chưa cài đặt node2vec

**Giải pháp:** 
```bash
pip install node2vec
```

Hoặc script sẽ tự động sử dụng random embeddings (fallback)

### Visualization quá lớn, không load được

**Giải pháp:** 
- Giảm `--max_nodes` (ví dụ: 200 thay vì 500)
- Sử dụng sample KG thay vì full KG

---

## Tài liệu tham khảo

- [MIMIC-III Documentation](https://mimic.mit.edu/)
- [NetworkX Documentation](https://networkx.org/)
- [Node2Vec Paper](https://arxiv.org/abs/1607.00653)
- [GAT-ETM Paper](https://arxiv.org/abs/...)

---

## Tác giả

Script được phát triển cho dự án GAT-ETM (Graph Attention Network - Embedded Topic Model) sử dụng dữ liệu MIMIC-III.

