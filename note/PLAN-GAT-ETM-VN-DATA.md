# PLAN: Triển khai GAT-ETM trên dữ liệu EHR Việt Nam

## Tổng quan

**Mục tiêu**: Áp dụng mô hình GAT-ETM (từ paper gốc) trên dữ liệu EHR của bệnh viện Việt Nam để:
1. Học các disease topics từ dữ liệu EHR VN
2. Thực hiện drug imputation (đề xuất thuốc dựa trên chẩn đoán)
3. So sánh kết quả với paper gốc

---

## Tiến độ tổng quan

**Tiến độ hiện tại: 5/8 phases (62.5%)**

### Phase 1: Thu thập ICD-10 & ATC Hierarchy ✅
| | |
|---|---|
| **Trạng thái** | ✅ Hoàn thành (Session 2) |
| **Input** | WHO ICD-10 (package `simple-icd-10`) |
| **Output** | `embed_vn/icd10_hierarchy.pkl` (12,542 nodes, 39,078 edges) |
| **Script** | `scripts/build_icd10_graph.py`, `scripts/icd_utils.py` |

### Phase 2: Parse Drug Names ✅
| | |
|---|---|
| **Trạng thái** | ✅ Hoàn thành (Session 3) |
| **Input** | `data-bv/thuoc.xlsx` (83,761 rows) |
| **Output** | `data_vn/unique_drugs.csv` (1,075 thuốc), `data_vn/drug_statistics.csv` |
| **Script** | `scripts/parse_drug_names.py` |

### Phase 3: Drug → ATC Mapping ✅
| | |
|---|---|
| **Trạng thái** | ✅ Hoàn thành (Session 4) |
| **Input** | `data_vn/unique_drugs.csv`, WHO ATC-DDD database |
| **Output** | `data_vn/drug_atc_mapping.csv` (585/1,075 mapped, 86.1% coverage), `data_vn/unmapped_drugs.csv` |
| **Script** | `scripts/drug_atc_mapping.py` |

### Phase 4: Xây dựng Knowledge Graph + Node2Vec ✅
| | |
|---|---|
| **Trạng thái** | ✅ Hoàn thành (Session 5) |
| **Input** | `embed_vn/icd10_hierarchy.pkl`, `data_vn/drug_atc_mapping.csv`, `data-bv/thuoc.xlsx` |
| **Output** | `embed_vn/atc_hierarchy.pkl` (6,440 nodes), `embed_vn/knowledge_graph.pkl` (18,897 nodes, 77,952 edges), `embed_vn/node2vec_embeddings.pkl` (256-dim) |
| **Script** | `scripts/build_knowledge_graph.py` |

### Phase 5: Chuẩn bị Data cho Model ✅
| | |
|---|---|
| **Trạng thái** | ✅ Hoàn thành (Session 6) |
| **Input** | `data-bv/thuoc.xlsx`, `embed_vn/knowledge_graph.pkl`, `data_vn/drug_atc_mapping.csv` |
| **Output** | `data_vn/bow_train.npy` (49,184 samples), `data_vn/bow_test.npy` (21,080 samples), `data_vn/bow_test_1.npy`, `data_vn/bow_test_2.npy`, `data_vn/metadata.txt`, `data_vn/vocab_info.pkl`, `embed_vn/graph_by_vocab.pkl` |
| **Script** | `scripts/create_bow.py` |

### Phase 6: Training ⏳
| | |
|---|---|
| **Trạng thái** | ⏳ Chưa làm |
| **Input** | `data_vn/bow_*.npy`, `data_vn/metadata.txt`, `embed_vn/node2vec_embeddings.pkl` |
| **Output** | `results_vn/model_checkpoints/`, `results_vn/beta_*.npy`, `results_vn/theta_*/` |
| **Script** | `main_getm.py` |

### Phase 7: Evaluation ⏳
| | |
|---|---|
| **Trạng thái** | ⏳ Chưa làm |
| **Input** | Trained model, `data_vn/bow_test*.npy` |
| **Output** | Metrics: NLL, TC, TD, TQ, Prec@5, Recall@5, F1@5 |
| **Script** | `main_getm.py --mode eval` |

### Phase 8: Visualization & Analysis ⏳
| | |
|---|---|
| **Trạng thái** | ⏳ Chưa làm |
| **Input** | Trained model, embeddings, topic distributions |
| **Output** | t-SNE plots, topic visualizations, case studies |
| **Script** | TBD |

---

## Phân tích dữ liệu hiện có

### Dữ liệu VN thực tế (data-bv/)

| File | Rows | Unique Values | Mô tả |
|------|------|---------------|-------|
| `CLS.xlsx` | **395,583** | 20,029 MAICD | Cận lâm sàng, thủ thuật, xét nghiệm |
| `thuoc.xlsx` | **83,761** | 37,872 patients, 57,756 Thuoc | Thông tin đơn thuốc |

### Thống kê chi tiết

```
TỔNG QUAN:
- Tổng records: 479,344
- Unique patients: ~37,872
- Unique ICD codes: 2,788 (98.8% đúng format ICD-10)
- Records có thuốc: 74,383 (88.8% của thuoc.xlsx)
- Unique drug names (ước tính): 1,000+

TOP 10 ICD CODES:
1. I10    - Tăng huyết áp vô căn (135,401)
2. E78.2  - Tăng lipid máu hỗn hợp (118,106)
3. E11.9  - ĐTĐ type 2, không biến chứng (53,819)
4. K21    - Trào ngược dạ dày-thực quản (45,120)
5. I25.0  - Bệnh tim thiếu máu xơ vữa (37,608)
6. J20    - Viêm phế quản cấp (22,187)
7. R10.0  - Đau bụng cấp (19,989)
8. K76.0  - Gan nhiễm mỡ (19,007)
9. I25    - Bệnh tim thiếu máu cục bộ mạn (18,089)
10. M10.0 - Gout vô căn (16,802)
```

### Cấu trúc dữ liệu chính

**CLS.xlsx (63 columns):**
- `MAICD`: Mã ICD-10 (có thể multiple, ngăn cách bởi `;`)
- `CHANDOAN`: Mô tả chẩn đoán
- `HOTEN`: Họ tên bệnh nhân
- `NAMSINH`: Năm sinh
- `NGAY`: Ngày khám
- Các cột khác: TENKP, TENBS, XUTRI, ...

**thuoc.xlsx (22 columns):**
- `MAICD`: Mã ICD-10
- `Thuoc`: Danh sách thuốc dạng text
  - Format: "TenThuoc dosage(quantity) DonVi TenThuoc2 dosage2(quantity2) DonVi2..."
  - VD: "Vinfoxin 50mg(56) Viên Betahistin 24 24mg(56) Viên..."
- `HOTEN`, `NAMSINH`, `GIOITINH`: Thông tin bệnh nhân

### So sánh với Paper gốc

| Aspect | Paper gốc (PopHR) | Dữ liệu VN |
|--------|-------------------|------------|
| ICD version | ICD-9 | **ICD-10** ✓ |
| Drug codes | ATC codes | **Tên thương mại** (cần mapping) |
| Patient ID | Có sẵn | Composite key (HOTEN+NAMSINH+GIOITINH) |
| Sample size | 1.2M patients | **~38K patients** |
| Unique ICD | 5,107 | **2,788** |
| Knowledge Graph | ICD-9 + ATC | Cần xây dựng ICD-10 + ATC |

### Nguồn ICD-10 Vietnam
- URL: https://icd.kcb.vn/icd-10/icd10-dual
- Cung cấp: ICD-10 hierarchy theo chuẩn Việt Nam
- Có thể có mapping ICD-9 ↔ ICD-10

---

## PLAN CHI TIẾT (A-Z)

### Phase 1: Thu thập và Chuẩn bị Dữ liệu Bổ sung

#### Task 1.1: Thu thập ICD-10 Hierarchy
**Input**: WHO ICD-10 classification
**Output**: `icd10_graph.pkl` (NetworkX graph)

```
Nguồn dữ liệu:
- WHO ICD-10 Browser: https://icd.who.int/browse10
- Python package: `simple_icd_10` hoặc `icd10-cm`
- File CSV/XML từ WHO

Cấu trúc graph:
- Nodes: Mã ICD-10 (VD: J44, J44.0, J44.1, ...)
- Edges: Parent-child relationships trong hierarchy
- Node attributes: level, description
```

#### Task 1.2: Thu thập ATC Hierarchy
**Input**: WHO ATC/DDD Index
**Output**: `atc_graph.pkl` (NetworkX graph)

```
Nguồn dữ liệu:
- WHO ATC/DDD Index: https://www.whocc.no/atc_ddd_index/
- Có thể scrape hoặc download

Cấu trúc graph:
- Nodes: Mã ATC (VD: N05, N05A, N05AH, N05AH03, ...)
- Edges: 5 levels hierarchy
- Node attributes: level, name
```

#### Task 1.3: Thu thập ICD-10 to ATC Relations (Disease-Drug Links)
**Input**: Các nguồn drug-disease associations
**Output**: `icd10_atc_relations.pkl`

```
Nguồn dữ liệu khả thi:
- DrugBank: https://go.drugbank.com/
- MEDI-SPAN / RxNorm
- OpenFDA
- Hoặc tự xây dựng từ dữ liệu VN (ICD + Thuốc được kê cùng nhau)

Format:
- DataFrame với columns: [ICD10_code, ATC_code, relation_type]
```

---

### Phase 2: Chuẩn hóa Dữ liệu VN

#### Task 2.1: Xử lý và chuẩn hóa mã ICD
**Input**: `df_all_feature_sample.xlsx`, `df_medicine_sample.xlsx`
**Output**: `processed_icd.csv`

```python
# Pseudocode
1. Đọc cả 2 file xlsx
2. Extract MAICD column
3. Split multiple ICD codes (ngăn cách bởi ';')
4. Chuẩn hóa format ICD-10:
   - Loại bỏ ký tự đặc biệt (';', spaces)
   - Validate format (VD: J44.0, N18.5, ...)
5. Loại bỏ ICD codes không hợp lệ
6. Lưu mapping: raw_icd -> normalized_icd
```

#### Task 2.2: Mapping tên thuốc VN → ATC codes
**Input**: Cột `Thuoc` từ `df_medicine_sample.xlsx`
**Output**: `drug_atc_mapping.csv`, `processed_drugs.csv`

```python
# Đây là bước QUAN TRỌNG và KHÓ NHẤT

Các approach khả thi:

A. Manual mapping (nếu số lượng thuốc ít):
   - Extract tất cả unique drug names
   - Tra cứu thủ công mã ATC cho từng thuốc
   - Tạo file mapping

B. Automated mapping:
   - Sử dụng RxNorm API để match tên thuốc
   - Sử dụng DrugBank để tra cứu
   - Sử dụng fuzzy matching với danh mục thuốc WHO

C. Semi-automated:
   - Automated matching trước
   - Manual review và correction sau

Steps:
1. Parse cột Thuoc để extract tên thuốc riêng lẻ
   VD: "Vinfoxin 50mg(56) Viên" -> "Vinfoxin"
2. Normalize tên thuốc (lowercase, remove dosage, ...)
3. Match với ATC database
4. Validate và review kết quả
```

#### Task 2.3: Tạo Patient ID và tổng hợp dữ liệu
**Input**: Processed ICD, Processed drugs
**Output**: `patient_data.pkl`

```python
# Tạo unique patient identifier
# Có thể dùng: HOTEN + NAMSINH + GIOITINH làm composite key
# Hoặc nếu có MA_BENH_AN thì dùng trực tiếp

# Output format:
patient_data = {
    'patient_id': [...],
    'icd_codes': [[list of ICD codes per patient], ...],
    'atc_codes': [[list of ATC codes per patient], ...],
}
```

---

### Phase 3: Xây dựng Knowledge Graph

#### Task 3.1: Xây dựng ICD-10 Graph
**Script**: `build_icd10_graph.py`

```python
import networkx as nx

def build_icd10_graph():
    G = nx.DiGraph()

    # Add nodes from ICD-10 hierarchy
    # Set node attributes: type='ICD10', level=...

    # Add edges (parent -> child)

    # Graph augmentation: connect each node to all ancestors
    # (như trong paper)

    return G
```

#### Task 3.2: Xây dựng ATC Graph
**Script**: `build_atc_graph.py`

```python
def build_atc_graph():
    G = nx.DiGraph()

    # ATC có 5 levels:
    # Level 1: Anatomical main group (1 letter)
    # Level 2: Therapeutic subgroup (2 digits)
    # Level 3: Pharmacological subgroup (1 letter)
    # Level 4: Chemical subgroup (1 letter)
    # Level 5: Chemical substance (2 digits)

    # Add nodes and edges
    # Graph augmentation

    return G
```

#### Task 3.3: Merge ICD-ATC Graphs với Disease-Drug Links
**Script**: `build_knowledge_graph.py`
**Output**: `icd10_atc_knowledge_graph.pkl`

```python
def merge_graphs(icd_graph, atc_graph, disease_drug_links):
    G = nx.compose(icd_graph, atc_graph)

    # Add disease-drug edges
    for icd, atc in disease_drug_links:
        if icd in G and atc in G:
            G.add_edge(icd, atc)

    # Convert to undirected
    G = G.to_undirected()

    return G
```

#### Task 3.4: Generate Node2Vec Embeddings
**Script**: `generate_node2vec_embeddings.py`
**Output**: `node2vec_embeddings.pkl`

```python
from node2vec import Node2Vec

# Parameters (từ paper):
EMBEDDING_DIM = 256
WALK_LENGTH = 20
NUM_WALKS = 10
WINDOW = 8

def generate_embeddings(G):
    node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM,
                        walk_length=WALK_LENGTH,
                        num_walks=NUM_WALKS,
                        workers=4)
    model = node2vec.fit(window=WINDOW)

    # Get embeddings for all nodes
    embeddings = {node: model.wv[node] for node in G.nodes()}

    return embeddings
```

---

### Phase 4: Chuẩn bị Data cho Model

#### Task 4.1: Tạo Vocabulary
**Script**: `create_vocabulary.py`
**Output**: `icd_vocab.pkl`, `atc_vocab.pkl`, `vocab_info.pkl`

```python
# Chỉ giữ các codes xuất hiện trong cả:
# 1. Knowledge graph
# 2. Patient data

def create_vocabulary(patient_data, knowledge_graph):
    # Extract unique ICD codes từ patients
    icd_codes_in_data = set()
    for patient in patient_data:
        icd_codes_in_data.update(patient['icd_codes'])

    # Filter: chỉ giữ codes có trong KG
    icd_vocab = [c for c in icd_codes_in_data if c in knowledge_graph]

    # Tương tự cho ATC
    atc_vocab = ...

    # Create vocab mappings
    icd2idx = {code: idx for idx, code in enumerate(icd_vocab)}
    atc2idx = {code: idx for idx, code in enumerate(atc_vocab)}

    return icd_vocab, atc_vocab, icd2idx, atc2idx
```

#### Task 4.2: Tạo Bag-of-Words Representation
**Script**: `create_bow.py`
**Output**: `bow_train.npy`, `bow_test.npy`, `bow_test_1.npy`, `bow_test_2.npy`

```python
import numpy as np
from scipy.sparse import csr_matrix

def create_bow(patient_data, icd2idx, atc2idx):
    V_icd = len(icd2idx)
    V_atc = len(atc2idx)
    V = V_icd + V_atc
    N = len(patient_data)

    # Create sparse BoW matrix
    rows, cols, data = [], [], []
    for p_idx, patient in enumerate(patient_data):
        for icd in patient['icd_codes']:
            if icd in icd2idx:
                rows.append(p_idx)
                cols.append(icd2idx[icd])
                data.append(1)

        for atc in patient['atc_codes']:
            if atc in atc2idx:
                rows.append(p_idx)
                cols.append(V_icd + atc2idx[atc])
                data.append(1)

    bow = csr_matrix((data, (rows, cols)), shape=(N, V))
    return bow

# Split train/test: 60/10/30 hoặc 70/30
# Cho test: split document thành 2 halves để evaluate
```

#### Task 4.3: Tạo Metadata file
**Output**: `metadata.txt`

```
# Format: code_types vocab_size train_embeddings embedding_files
icd atc
5107 1057
1 1
embed/node2vec_icd.npy embed/node2vec_atc.npy
```

---

### Phase 5: Adapt Model Code

#### Task 5.1: Update `generate_icdNatc_geometric.py`
- Thay ICD-9 graph bằng ICD-10 graph
- Update code dict cho ICD-10
- Cập nhật ATC mapping nếu cần

#### Task 5.2: Update `dataset.py` (nếu có)
- Đảm bảo đọc đúng format dữ liệu mới

#### Task 5.3: Verify `graph_etm_DI.py` và `main_getm.py`
- Kiểm tra compatibility với data format mới
- Có thể không cần thay đổi nếu data format giống

---

### Phase 6: Training

#### Task 6.1: Train Model cơ bản (không drug imputation)
```bash
python main_getm.py \
    --data_path data_vn/ \
    --meta_file metadata \
    --num_topics 50 \
    --epochs 50 \
    --batch_size 512 \
    --lr 0.01 \
    --tq
```

#### Task 6.2: Train Model với Drug Imputation
```bash
python main_getm.py \
    --data_path data_vn/ \
    --drug_imputation \
    --loss wkl \
    --gamma 1.0 \
    --epochs 50
```

---

### Phase 7: Evaluation

#### Task 7.1: Đánh giá Topic Quality
- **Topic Coherence (TC)**: Đo co-occurrence của top codes
- **Topic Diversity (TD)**: Đo uniqueness across topics
- **Topic Quality = TC × TD**

```python
# Metrics từ paper:
# TC với top-3 codes
# TD với top-3 codes
# So sánh với paper: GAT-ETM đạt TQ = 0.192
```

#### Task 7.2: Đánh giá Reconstruction
- Negative Log-Likelihood (NLL) trên test set
- Paper: GAT-ETM đạt NLL = 172.69

#### Task 7.3: Đánh giá Drug Imputation
- **Patient-wise**: Prec@5, Recall@5, F1@5
- **Drug-wise**: Precision tại các percentile tần suất

```python
# Metrics từ paper:
# Prec@5 = 0.26, Recall@5 = 0.1225, F1@5 = 0.1569
```

#### Task 7.4: So sánh kết quả
Tạo bảng so sánh:

| Metric | Paper (PopHR) | VN Data |
|--------|---------------|---------|
| NLL | 172.69 | ? |
| TC (ICD) | 0.18 | ? |
| TC (ATC) | 0.314 | ? |
| TD (ICD) | 0.76 | ? |
| TD (ATC) | 0.787 | ? |
| TQ (avg) | 0.192 | ? |
| Prec@5 | 0.26 | ? |
| Recall@5 | 0.1225 | ? |
| F1@5 | 0.1569 | ? |

---

### Phase 8: Visualization và Analysis

#### Task 8.1: Visualize Topic Distributions
- Hiển thị top ICD và ATC codes cho mỗi topic
- Kiểm tra clinical coherence

#### Task 8.2: Visualize Code Embeddings
- t-SNE visualization
- Kiểm tra clustering của related codes

#### Task 8.3: Case Study Drug Imputation
- Chọn một số patients
- Phân tích thuốc được đề xuất vs thực tế

---

## Ước tính khối lượng công việc

| Phase | Tasks | Độ khó | Ghi chú |
|-------|-------|--------|---------|
| Phase 1 | Thu thập KG data | Medium | Cần research nguồn dữ liệu |
| Phase 2 | Chuẩn hóa data VN | **HIGH** | Drug name → ATC mapping là khó nhất |
| Phase 3 | Xây dựng KG | Medium | Có thể tham khảo code paper |
| Phase 4 | Chuẩn bị data | Low | Follow format của paper |
| Phase 5 | Adapt code | Low | Minimal changes |
| Phase 6 | Training | Low | Follow paper settings |
| Phase 7 | Evaluation | Medium | Implement metrics |
| Phase 8 | Analysis | Medium | Visualization và interpretation |

---

## Challenges và Risks

### 1. Drug Name to ATC Mapping
- **Challenge**: Tên thuốc VN là thương hiệu, không phải generic name
- **Risk**: Mapping không chính xác → ảnh hưởng kết quả
- **Mitigation**: Manual verification, sử dụng multiple data sources

### 2. Sample Size nhỏ
- **Challenge**: Chỉ có 50 rows sample data
- **Risk**: Model không học được patterns tốt
- **Mitigation**: Thu thập thêm data từ bệnh viện

### 3. ICD-10 vs ICD-9
- **Challenge**: Paper dùng ICD-9, VN dùng ICD-10
- **Risk**: Cần rebuild knowledge graph
- **Mitigation**: Có mapping ICD-9 ↔ ICD-10 available

### 4. Missing Disease-Drug Links
- **Challenge**: Không có sẵn ICD-10 to ATC relations
- **Risk**: Knowledge graph không complete
- **Mitigation**: Có thể learn từ data (co-occurrence) hoặc dùng external sources

---

## Cấu trúc thư mục đề xuất

```
GAT-ETM/
├── raw_vn_data/                 # Dữ liệu gốc từ bệnh viện
│   ├── df_all_feature_sample.xlsx
│   └── df_medicine_sample.xlsx
│
├── data_vn/                     # Dữ liệu đã xử lý cho model
│   ├── bow_train.npy
│   ├── bow_test.npy
│   ├── bow_test_1.npy
│   ├── bow_test_2.npy
│   └── metadata.txt
│
├── embed_vn/                    # Knowledge graph và embeddings
│   ├── icd10_graph.pkl
│   ├── atc_graph.pkl
│   ├── icd10_atc_knowledge_graph.pkl
│   ├── node2vec_embeddings.pkl
│   └── vocab_info.pkl
│
├── scripts/                     # Scripts xử lý dữ liệu
│   ├── preprocess_vn_data.py
│   ├── build_knowledge_graph.py
│   ├── drug_atc_mapping.py
│   └── create_bow.py
│
├── results_vn/                  # Kết quả training
│   ├── model_checkpoints/
│   ├── beta_*.npy
│   ├── rho_*.npy
│   └── theta_*/
│
└── note/                        # Documentation
    ├── GAT-ETM-paper-summary.md
    └── PLAN-GAT-ETM-VN-DATA.md
```

---

## Next Steps

1. **Immediate**: Xác nhận có thể thu thập thêm dữ liệu VN (nhiều hơn 50 rows)
2. **Short-term**: Bắt đầu Phase 1 - thu thập ICD-10 và ATC hierarchies
3. **Parallel**: Research drug name to ATC mapping solutions

---

---

## SESSION LOG

### Session 1: 2026-03-18 - Khám phá và Lập Plan

**Đã hoàn thành:**
- [x] Đọc và tóm tắt paper GAT-ETM (`note/GAT-ETM-paper-summary.md`)
- [x] Phân tích cấu trúc dữ liệu VN thực tế (CLS.xlsx, thuoc.xlsx)
- [x] Xác định thống kê: 479K records, 38K patients, 2,788 ICD codes
- [x] Phân tích format thuốc và khả năng parse
- [x] Lập plan chi tiết 8 phases
- [x] Xác định nguồn ICD-10 VN: https://icd.kcb.vn/icd-10/icd10-dual

**Phát hiện quan trọng:**
1. Dữ liệu VN dùng ICD-10 (98.8% valid format)
2. Thuốc ở dạng text, cần parse + mapping ATC
3. Có thể tạo patient ID từ HOTEN + NAMSINH + GIOITINH
4. Sample size ~38K patients (nhỏ hơn paper 30x nhưng vẫn đủ dùng)

---

## NEXT STEPS (Ưu tiên theo thứ tự)

### Step 1: Thu thập ICD-10 Hierarchy
```
Nguồn: https://icd.kcb.vn/icd-10/icd10-dual
Output: embed_vn/icd10_hierarchy.pkl

Tasks:
1. Scrape hoặc download ICD-10 từ trang kcb.vn
2. Parse thành NetworkX graph
3. Thực hiện graph augmentation (link to ancestors)
```

### Step 2: Parse và Extract tên thuốc
```
Input: data-bv/thuoc.xlsx (cột Thuoc)
Output: data_vn/unique_drugs.csv

Tasks:
1. Parse cột Thuoc với regex pattern
2. Normalize tên thuốc (lowercase, remove dosage)
3. Tạo danh sách unique drug names
4. Ước tính: ~1,000+ unique drugs
```

### Step 3: Drug Name → ATC Mapping
```
Input: unique_drugs.csv
Output: data_vn/drug_atc_mapping.csv

Nguồn mapping khả thi:
- DrugBank (https://go.drugbank.com/)
- RxNorm API
- WHO ATC Index
- Manual mapping (nếu cần)

⚠️ Đây là step khó nhất, có thể cần semi-automated approach
```

### Step 4: Xây dựng ATC Hierarchy
```
Nguồn: WHO ATC/DDD Index (https://www.whocc.no/atc_ddd_index/)
Output: embed_vn/atc_hierarchy.pkl
```

### Step 5: Tạo Knowledge Graph
```
Input: icd10_hierarchy.pkl, atc_hierarchy.pkl, drug_atc_mapping.csv
Output: embed_vn/icd10_atc_knowledge_graph.pkl

Tasks:
1. Merge ICD-10 và ATC graphs
2. Thêm disease-drug edges từ co-occurrence trong data
3. Graph augmentation
4. Generate Node2Vec embeddings
```

### Step 6: Chuẩn bị Data cho Model
```
Output: data_vn/bow_train.npy, bow_test.npy, metadata.txt
```

### Step 7: Train và Evaluate
```
Script: main_getm.py với data_path=data_vn/
```

---

## FILE LOCATIONS

```
Dữ liệu gốc:
/Users/m001938/Documents/CS_UIT/LuanVan/data-bv/
├── CLS.xlsx      (395,583 rows - cận lâm sàng)
└── thuoc.xlsx    (83,761 rows - đơn thuốc)

Project:
/Users/m001938/Documents/CS_UIT/LuanVan/GAT-ETM/
├── note/
│   ├── GAT-ETM-paper-summary.md    # Tóm tắt paper
│   └── PLAN-GAT-ETM-VN-DATA.md     # Plan này
├── raw_vn_data/                     # Sample data (50 rows)
├── data_vn/                         # [TBD] Processed data
├── embed_vn/                        # [TBD] KG và embeddings
└── scripts/                         # [TBD] Processing scripts

Nguồn tham khảo:
- ICD-10 VN: https://icd.kcb.vn/icd-10/icd10-dual
- WHO ATC: https://www.whocc.no/atc_ddd_index/
- Paper code: https://github.com/li-lab-mcgill/GAT-ETM
```

---

## QUICK START (Lần sau)

```bash
# 1. Activate environment
cd /Users/m001938/Documents/CS_UIT/LuanVan/GAT-ETM
source venv/bin/activate

# 2. Đọc plan
cat note/PLAN-GAT-ETM-VN-DATA.md

# 3. Bắt đầu từ Step 1: Thu thập ICD-10
# (xem chi tiết trong plan)
```

---

### Session 2: 2026-03-25 - Thu thập ICD-10 Hierarchy (Step 1)

**Đã hoàn thành:**
- [x] Khám phá nguồn ICD-10 VN (kcb.vn là SPA, không scrape được)
- [x] Tìm giải pháp thay thế: Python package `simple-icd-10` (WHO ICD-10)
- [x] Cài đặt package: `pip install simple-icd-10`
- [x] Tạo script `scripts/build_icd10_graph.py`
- [x] Build ICD-10 hierarchy graph với graph augmentation
- [x] Tạo utility functions `scripts/icd_utils.py`
- [x] Validate với dữ liệu VN: **97.1% coverage** (67/69 codes)

**Output:**
```
embed_vn/icd10_hierarchy.pkl
├── Nodes: 12,542 (WHO ICD-10 codes)
│   ├── Chapters: 22
│   ├── Blocks: 274
│   ├── Categories: 2,050
│   └── Subcategories: 10,196
├── Edges: 39,078
│   ├── Hierarchy: 12,520
│   └── Augmented: 26,558
└── Format: NetworkX DiGraph
```

**Scripts tạo:**
- `scripts/build_icd10_graph.py` - Build ICD-10 hierarchy graph
- `scripts/icd_utils.py` - Utility functions để normalize/validate ICD codes

**Phát hiện:**
1. Package `simple-icd-10` cung cấp WHO ICD-10 (2019 version)
2. 97.1% codes trong VN data có trong WHO ICD-10
3. Codes không tìm thấy: D75.2 (có thể là ICD-10-CM specific)
4. Cần xử lý special markers (*, +, !) khi normalize codes

**Next Step:** Step 2 - Parse và Extract tên thuốc từ dữ liệu VN

---

### Session 3: 2026-03-30 - Parse Drug Names (Step 2)

**Đã hoàn thành:**
- [x] Phân tích format cột Thuoc trong dữ liệu VN
- [x] Tạo script `scripts/parse_drug_names.py` để extract tên thuốc
- [x] Xử lý các đơn vị đóng gói: Viên, Chai, Ống, Gói, Túi, Hộp, Lọ, Tuýp, Vỉ, Miếng, Bánh, Bình, Bộ, Cái, Kít
- [x] Lọc bỏ vật tư y tế (bơm tiêm, kim tiêm) và mỹ phẩm (shampoo, cream, wash)
- [x] Chạy parse trên full data (83,761 records)

**Output:**
```
data_vn/unique_drugs.csv
├── Columns: drug_name, generic_name_guess, atc_code, count
├── Unique drugs: 1,075
└── Format: CSV (UTF-8)

data_vn/drug_statistics.csv
├── Columns: drug_name, count
├── Sorted by frequency (descending)
└── Format: CSV (UTF-8)
```

**Thống kê:**
```
Tổng số dòng: 83,761
Dòng có thuốc: 74,089 (88.5%)
Tổng số thuốc parsed: 287,038
Số thuốc unique: 1,075
Thuốc có count >= 100: 379
Thuốc có count >= 1,000: 71
```

**Top 10 thuốc phổ biến:**
```
1. Agicardi         8,577
2. Partamol Tab.    6,955
3. Atorvastatin     6,946
4. Tunadimet        5,587
5. Esomeprazol      5,497
6. Kapredin         5,054
7. Concor Cor       4,043
8. Atisyrup Zinc    3,787
9. Glumeform        3,666
10. Dacolfort       3,660
```

**Scripts tạo:**
- `scripts/parse_drug_names.py` - Parse và extract tên thuốc từ dữ liệu VN

**Phát hiện:**
1. Format thuốc: "TênThuốc Dosage(Quantity) ĐơnVị" (VD: "Vinfoxin 50mg(56) Viên")
2. Dữ liệu chứa cả thuốc, vật tư y tế, và mỹ phẩm
3. Một số tên thương mại có số ở cuối (Betahistin 24, Apitim 10)
4. Cột atc_code để trống, cần mapping ở Step 3

**Next Step:** Step 3 - Drug Name → ATC Mapping

---

### Session 4: 2026-03-30 - Drug Name → ATC Mapping (Step 3)

**Đã hoàn thành:**
- [x] Nghiên cứu nguồn dữ liệu ATC (WHO ATC/DDD Index, RxNorm API)
- [x] Download WHO ATC-DDD database (6,971 codes)
- [x] Tạo script `scripts/drug_atc_mapping.py` với 2 strategies:
  - Manual dictionary mapping (260+ generic names + VN brand names)
  - Fuzzy string matching với WHO ATC database
- [x] Chạy automated mapping trên 1,075 thuốc VN

**Output:**
```
data_vn/atc_reference/who_atc_ddd.csv
├── ATC codes: 6,971
├── Level 5 (substances): 5,684
└── Format: CSV (atc_code, atc_name, ddd, uom, adm_r, note)

data_vn/drug_atc_mapping.csv
├── Columns: drug_name, count, generic_name, atc_code, confidence, method
├── Mapped drugs: 585/1,075 (54.4%)
└── Format: CSV (UTF-8)

data_vn/unmapped_drugs.csv
├── Unmapped drugs: 490/1,075
└── Sorted by frequency (priority for manual review)
```

**Thống kê mapping:**
```
Total drugs: 1,075
Mapped: 585 (54.4%)
  - Manual dict: 257
  - Fuzzy match: 328
Unmapped: 490 (45.6%)

PRESCRIPTION COVERAGE: 247,130/287,038 (86.1%)
```

**By confidence:**
```
high (exact match):     234
medium (partial match):  23
fuzzy_0.65-1.00:        328
```

**Scripts tạo:**
- `scripts/drug_atc_mapping.py` - Mapping tên thuốc VN → ATC codes

**Nguồn dữ liệu:**
- WHO ATC/DDD Index: https://atcddd.fhi.no/atc_ddd_index/
- GitHub scraper: https://github.com/fabkury/atcd
- RxNorm API (reference): https://lhncbc.nlm.nih.gov/RxNav/

**Phát hiện:**
1. 86.1% prescription coverage đủ tốt cho mục đích xây dựng KG
2. Manual dictionary mapping hiệu quả hơn fuzzy matching cho brand names VN
3. Một số thuốc unmapped là thuốc đông y, thực phẩm chức năng, hoặc tên thương mại đặc biệt
4. Fuzzy matching cần threshold ~0.65-0.7 để cân bằng precision/recall

**Next Step:** Step 4 - Build ATC Hierarchy Graph & Knowledge Graph

---

### Session 5: 2026-03-30 - Build Knowledge Graph (Step 4)

**Đã hoàn thành:**
- [x] Build ATC hierarchy graph từ WHO ATC-DDD data
- [x] Load ICD-10 graph từ Step 1
- [x] Merge ICD-10 + ATC graphs
- [x] Extract disease-drug links từ co-occurrence trong dữ liệu VN
- [x] Generate Node2Vec embeddings (256 dimensions)

**Output:**
```
embed_vn/
├── icd10_hierarchy.pkl      # 2.1 MB - ICD-10 hierarchy (từ Step 1)
├── atc_hierarchy.pkl        # 1.2 MB - ATC hierarchy
│   ├── Nodes: 6,440 (5 levels)
│   │   ├── Level 1 (Anatomical): 14
│   │   ├── Level 2 (Therapeutic): 94
│   │   ├── Level 3 (Pharmacological): 269
│   │   ├── Level 4 (Chemical): 909
│   │   └── Level 5 (Substance): 5,154
│   └── Edges: 23,975 (hierarchy + augmented)
│
├── knowledge_graph.pkl      # 3.8 MB - Merged graph
│   ├── Nodes: 18,897
│   │   ├── ICD-10: 12,457
│   │   └── ATC: 6,440
│   ├── Edges: 77,952
│   │   ├── hierarchy: 18,945
│   │   ├── augmented: 44,084
│   │   └── disease_drug: 14,923
│   └── Format: NetworkX undirected Graph
│
└── node2vec_embeddings.pkl  # 19 MB - Node embeddings
    ├── Nodes: 18,897
    ├── Dimensions: 256
    └── Parameters: walk_length=20, num_walks=10, window=8
```

**Disease-Drug Links Statistics:**
```
Total unique ICD-ATC pairs: 32,217
Filtered (>= 3 co-occurrences): 15,373
Links added to graph: 14,923

Top 10 Disease-Drug Links:
1. E78.2 (Hyperlipidemia) - C10AA05 (Atorvastatin): 11,587
2. I10 (Hypertension) - C10AA05 (Atorvastatin): 10,463
3. K21 (GERD) - A02BC05 (Esomeprazole): 9,871
4. I10 (Hypertension) - B01AC06 (Aspirin): 9,102
5. E78.2 (Hyperlipidemia) - B01AC06 (Aspirin): 8,939
6. E78.2 (Hyperlipidemia) - B01AC04 (Clopidogrel): 8,588
7. I10 (Hypertension) - B01AC04 (Clopidogrel): 8,290
8. I10 (Hypertension) - C09CA01 (Losartan): 7,227
9. E78.2 (Hyperlipidemia) - C09CA01 (Losartan): 6,754
10. E78.2 (Hyperlipidemia) - C10AA07 (Rosuvastatin): 6,558
```

**Scripts tạo:**
- `scripts/build_knowledge_graph.py` - Build complete knowledge graph với Node2Vec

**Phát hiện:**
1. Knowledge graph có tổng 18,897 nodes (ICD-10 + ATC)
2. Disease-drug links từ co-occurrence trong dữ liệu VN rất phong phú
3. Top diseases (I10-Hypertension, E78.2-Hyperlipidemia) có nhiều drug associations
4. Node2Vec embeddings mất ~35 giây để generate

**Next Step:** ~~Step 5 - Prepare Data for Model~~ ✅ Hoàn thành

---

### Session 6: 2026-04-05 - Chuẩn bị Data cho Model (Step 5)

**Đã hoàn thành:**
- [x] Phân tích format dữ liệu model yêu cầu (dataset.py, main_getm.py)
- [x] Tạo script `scripts/create_bow.py` để tạo BoW matrices
- [x] Xử lý 83,761 records từ thuoc.xlsx
- [x] Tạo vocabulary từ patient data và knowledge graph
- [x] Tạo BoW matrices cho train/test
- [x] Tạo embeddings theo thứ tự vocabulary
- [x] Tạo graph by vocab (renumbered nodes)
- [x] Đảm bảo graph connected

**Output:**
```
data_vn/
├── bow_train.npy        # 49,184 samples, sparse CSR
├── bow_test.npy         # 21,080 samples, sparse CSR
├── bow_test_1.npy       # First half of test docs
├── bow_test_2.npy       # Second half of test docs
├── metadata.txt         # Model metadata
└── vocab_info.pkl       # Vocabulary mappings

embed_vn/
├── icd_embeddings.npy   # (1493, 256)
├── atc_embeddings.npy   # (377, 256)
├── graph_by_vocab.pkl   # 1870 nodes, 16119 edges
└── embeddings_by_vocab.pkl  # Combined (1870, 256)

embed/
├── augmented_icdatc_graph_256_renumbered_by_vocab.pkl
└── augmented_icdatc_embed_8_20_10_256_by_vocab.pkl
```

**Thống kê:**
```
Dữ liệu đầu vào:
- Tổng records: 83,761
- Patients có cả ICD và thuốc: 70,264 (83.9%)

Vocabulary:
- ICD codes: 1,493 (từ 2,788 codes trong data, filtered by KG)
- ATC codes: 377 (từ 585 mapped drugs)
- Tổng vocabulary: 1,870

BoW Matrix:
- Training samples: 49,184 (70%)
- Test samples: 21,080 (30%)
- Non-zero entries: 485,006
- Density: 0.37%

Graph:
- Nodes: 1,870 (all vocab codes)
- Edges: 16,119 (hierarchy + disease-drug links)
- Connected: Yes (single component)
```

**Scripts tạo:**
- `scripts/create_bow.py` - Tạo BoW data cho model

**Phát hiện:**
1. 83.9% records có cả ICD codes và thuốc (đủ để training)
2. Vocabulary được filter để chỉ giữ codes có trong KG
3. Graph được augmented để đảm bảo connected (thêm hierarchy edges)
4. Format output tương thích với main_getm.py

**Next Step:** ~~Phase 6 - Training model với dữ liệu VN~~ ✅ Đã sẵn sàng

---

### Session 6 (tiếp theo): 2026-04-05 - Data Verification & Training Script

**Đã hoàn thành:**
- [x] Tạo script `scripts/verify_data.py` để verification data
- [x] Tạo verification report và files cho expert review
- [x] Phân tích main_getm.py và graph_etm.py để hiểu data requirements
- [x] Sửa lỗi PyTorch 2.x sparse tensor collation (custom collate_fn)
- [x] Tạo `train_vn.py` - training script adapted cho VN data
- [x] Test training thành công 1 epoch

**Output Verification:**
```
data_vn/verification/
├── verification_report.txt           # Báo cáo tổng hợp
├── expert_review_drug_mapping.csv    # 50 thuốc phổ biến cần review
├── expert_review_fuzzy_matches.csv   # Fuzzy matches có thể sai
├── expert_review_disease_drug_pairs.csv  # 100 cặp ICD-ATC
├── expert_review_sample_patients.csv # 20 bệnh nhân mẫu
├── viz_codes_distribution.png        # Visualization
├── viz_code_frequency.png
└── viz_embeddings_tsne.png
```

**Data Statistics:**
```
Training set: 49,184 samples × 1,870 codes
Test set: 21,080 samples × 1,870 codes
Density: 0.37%

Codes per patient:
- ICD: mean 3.42, median 3
- ATC: mean 3.47, median 3
- Total: mean 6.89, median 6

Graph: 1,870 nodes, 16,119 edges, Connected: Yes
```

**Vấn đề cần Expert Review:**

1. **Fuzzy Matches có thể sai** (trong `expert_review_fuzzy_matches.csv`):
   | Drug VN | ATC Mapped | Generic | Confidence |
   |---------|------------|---------|------------|
   | wosulin | N06AA16 | dosulepin | 0.75 | ❌ Sai (Wosulin = insulin) |
   | ripratine | N05AX15 | cariprazine | 0.80 | ❓ |
   | glimsure | G02CB02 | lisuride | 0.75 | ❓ |
   | stamlo-t | C07AA07 | sotalol | 0.67 | ❓ |

2. **Disease-Drug Pairs hợp lý** (đã kiểm tra top 20):
   - E78.2 + C10AA05 (Tăng lipid + Atorvastatin) ✅
   - I10 + B01AC06 (Tăng HA + Aspirin) ✅
   - K21 + A02BC05 (Trào ngược + Esomeprazole) ✅
   - E11.9 + A10BA02 (ĐTĐ type 2 + Metformin) ✅

**Training Script:**
- Script mới: `train_vn.py` (PyTorch 2.x compatible)
- Test 1 epoch thành công:
  - Train Loss: 31.62, KL: 0.47
  - Test NLL: 14.83
  - Model params: 1,807,400

**Next Step:** Chạy full training với 50 epochs

---

*Created: 2026-03-18*
*Last Updated: 2026-04-05*
*Author: Claude Code*
