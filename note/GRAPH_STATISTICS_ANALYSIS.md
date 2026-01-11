# Phân Tích Chi Tiết Graph Statistics - Simple Ego Graph

## Tổng Quan

Graph Statistics cho thấy sự khác biệt quan trọng giữa **số lượng nodes trong graph** và **vocabulary sizes** (số codes thực sự được sử dụng trong training).

---

## 1. Total Nodes và Edges

### **Total nodes: 28**
**Ý nghĩa**: Tổng số nodes trong knowledge graph visualization

**Bao gồm**:
- ✅ **Vocabulary nodes** (leaf nodes): Các codes thực sự xuất hiện trong EHR data
- ✅ **Hierarchical nodes** (intermediate nodes): Parent nodes trong hierarchy trees
- ✅ **Root nodes**: Các node gốc (ví dụ: `ICD9_ROOT`, `LAB_ROOT`, `ATC_ROOT`)

**Trong visualization**:
- Tất cả 28 nodes đều được hiển thị
- Mỗi node có thể là một code cụ thể hoặc một node trung gian

---

### **Total edges: 28**
**Ý nghĩa**: Tổng số connections giữa các nodes

**Phân loại**:
- **Cooccurrence edges**: 10 edges (dashed red lines)
- **Hierarchical edges**: 18 edges (solid black lines)

**Tổng**: 10 + 18 = 28 edges ✓

---

## 2. Node Types - Phân Loại Nodes Theo Type

### **ATC: 11 nodes**
**Ý nghĩa**: Có 11 nodes thuộc loại ATC (drug codes) trong graph

**Bao gồm**:
- ✅ **Vocabulary ATC nodes**: Các ATC codes thực sự trong vocabulary (5 nodes)
- ✅ **Hierarchical ATC nodes**: Parent nodes trong ATC hierarchy (6 nodes)

**Ví dụ trong visualization**:
- `DRUG_24`, `DRUG_59`, `DRUG_69`, `DRUG_24276`, `DRUG_44419`, `DRUG_59178`, `DRUG_64655`, `DRUG_69067` (8 nodes có prefix DRUG_)
- Có thể có thêm 3 nodes ATC khác (có thể là parent nodes hoặc nodes không có prefix)

**Lưu ý**: 
- Vocabulary size cho ATC = **5** (chỉ 5 codes thực sự được dùng)
- 11 - 5 = **6 nodes** là hierarchical/parent nodes

---

### **CPT: 1 node**
**Ý nghĩa**: Có 1 node thuộc loại CPT (procedure code) trong graph

**Bao gồm**:
- Có thể là 1 CPT code trong vocabulary
- Hoặc 1 CPT node trong hierarchy

**Lưu ý**:
- Vocabulary size cho CPT = **11** (có 11 CPT codes trong vocabulary)
- Nhưng trong graph này chỉ có 1 CPT node được hiển thị
- **Giải thích**: Đây là ego-graph (subgraph), chỉ hiển thị 1 CPT node có liên kết với các nodes khác

---

### **ICD9: 3 nodes**
**Ý nghĩa**: Có 3 nodes thuộc loại ICD9 (diagnosis codes) trong graph

**Bao gồm**:
- ✅ **Vocabulary ICD9 nodes**: Có thể là 1 ICD9 code trong vocabulary
- ✅ **Hierarchical ICD9 nodes**: 2 parent nodes trong ICD9 hierarchy

**Ví dụ trong visualization**:
- `42731` (được xác định là ICD9 node trong Relationships Summary)
- Có thể có 2 nodes khác là parent nodes (ví dụ: `4273`, `427`)

**Lưu ý**:
- Vocabulary size cho ICD = **1** (chỉ 1 ICD code trong vocabulary)
- 3 - 1 = **2 nodes** là hierarchical/parent nodes

---

### **LAB: 13 nodes**
**Ý nghĩa**: Có 13 nodes thuộc loại LAB (laboratory codes) trong graph

**Bao gồm**:
- ✅ **Vocabulary LAB nodes**: Các LAB codes thực sự trong vocabulary (10 nodes)
- ✅ **Hierarchical LAB nodes**: Parent nodes trong LAB hierarchy (3 nodes)

**Ví dụ trong visualization**:
- `LAB_Chemistry`, `LAB_Hematology` (2 category nodes)
- `50882`, `50902`, `51275` (được xác định là LAB nodes trong Relationships Summary)
- Có thể có thêm 8 LAB nodes khác (có thể là parent nodes hoặc category nodes)

**Lưu ý**:
- Vocabulary size cho LAB = **10** (10 LAB codes trong vocabulary)
- 13 - 10 = **3 nodes** là hierarchical/parent nodes

---

## 3. Edge Types - Phân Loại Edges Theo Type

### **Cooccurrence: 10 edges**
**Ý nghĩa**: 10 edges đại diện cho mối quan hệ đồng xuất hiện (cùng xuất hiện trong cùng một admission/patient)

**Trong visualization**: 
- Được hiển thị bằng **dashed red lines** (đường nét đứt màu đỏ)
- Kết nối các nodes cùng xuất hiện trong dữ liệu thực tế

**Ví dụ từ Relationships Summary**:
- `42731 <-> 50882` (ICD9-LAB cooccurrence)
- `42731 <-> 50902` (ICD9-LAB cooccurrence)
- `42731 <-> 51275` (ICD9-LAB cooccurrence)
- Và 7 connections khác

**Ý nghĩa**:
- Phản ánh mối quan hệ thực tế trong dữ liệu EHR
- Giúp model học được các patterns đồng xuất hiện

---

### **Hierarchical: 18 edges**
**Ý nghĩa**: 18 edges đại diện cho mối quan hệ phân cấp (parent-child relationships)

**Trong visualization**:
- Được hiển thị bằng **solid black lines** (đường liền màu đen)
- Kết nối parent nodes với child nodes trong hierarchy

**Ví dụ**:
- `DRUG_59 → DRUG_59178` (ATC hierarchy)
- `DRUG_69067 → 4273` (ATC-ICD9 hierarchy)
- `DRUG_69 → DRUG_44419` (ATC hierarchy)
- `LAB_Chemistry → DRUG_24` (LAB-ATC hierarchy)
- `LAB_Chemistry → 50912` (LAB-ICD9 hierarchy)

**Ý nghĩa**:
- Phản ánh cấu trúc phân cấp của medical codes
- Giúp model hiểu được semantic relationships giữa các codes

---

## 3. Chi Tiết Về Ý Nghĩa Của 2 Loại Edges

### **3.1. Cooccurrence Edges (Đồng Xuất Hiện)**

#### **Định nghĩa**:
Cooccurrence edges đại diện cho **mối quan hệ thực tế** giữa các medical codes khi chúng **cùng xuất hiện** trong cùng một **hospital admission** (`hadm_id`).

#### **Cách tạo ra**:
```python
# Từ code: build_cooccurrence_edges()
1. Nhóm tất cả codes theo hadm_id (hospital admission ID):
   - hadm_to_icd: {hadm_id: [icd1, icd2, ...]}
   - hadm_to_atc: {hadm_id: [atc1, atc2, ...]}
   - hadm_to_lab: {hadm_id: [lab1, lab2, ...]}

2. Với mỗi admission, tạo edges giữa:
   - ICD9 ↔ ATC (chẩn đoán và thuốc được kê cùng lúc)
   - ICD9 ↔ CPT (chẩn đoán và thủ thuật được thực hiện)
   - ICD9 ↔ Lab (chẩn đoán và xét nghiệm được chỉ định)
   - ICD9 ↔ ICD9 (nhiều chẩn đoán cùng lúc)

3. Đếm tần suất đồng xuất hiện (weight = count)
4. Chỉ thêm edge nếu count >= min_cooccurrence (default=1)
```

#### **Ví dụ cụ thể**:
```
Patient A, Admission #12345:
  - Diagnoses: 42731 (Atrial fibrillation)
  - Lab tests: 50882 (Sodium), 50902 (Potassium), 51275 (WBC)
  
→ Tạo cooccurrence edges:
  42731 <-> 50882 (weight=1)
  42731 <-> 50902 (weight=1)
  42731 <-> 51275 (weight=1)
```

#### **Ý nghĩa trong GAT-ETM**:
1. **Phản ánh patterns thực tế**: 
   - Cho biết các codes nào thường xuất hiện cùng nhau trong thực tế
   - Ví dụ: Bệnh nhân rung nhĩ (42731) thường được xét nghiệm điện giải (50882, 50902)

2. **Học được clinical associations**:
   - ICD9 ↔ ATC: Bệnh nào thường dùng thuốc nào
   - ICD9 ↔ Lab: Bệnh nào cần xét nghiệm gì
   - ICD9 ↔ ICD9: Các bệnh thường đi kèm (comorbidities)

3. **Edge weight quan trọng**:
   - Weight = số lần đồng xuất hiện
   - Weight cao → mối quan hệ mạnh hơn
   - Model có thể sử dụng weight để ưu tiên các connections quan trọng

4. **Khác với causal relationships**:
   - Cooccurrence ≠ causation
   - Chỉ cho biết "cùng xuất hiện", không phải "gây ra"
   - Ví dụ: `42731 <-> 50882` không có nghĩa là rung nhĩ gây ra xét nghiệm Sodium

#### **Trong visualization**:
- **Màu**: Đỏ (red)
- **Style**: Dashed lines (nét đứt)
- **Vị trí**: Thường ở **central cluster**, kết nối nhiều nodes với nhau
- **Đặc điểm**: Tạo ra **dense connections** giữa các nodes khác loại

---

### **3.2. Hierarchical Edges (Phân Cấp)**

#### **Định nghĩa**:
Hierarchical edges đại diện cho **cấu trúc phân cấp** (parent-child relationships) của medical codes theo **ontology/hierarchy** của chúng.

#### **Cách tạo ra**:
```python
# Từ code: build_icd9_hierarchy(), build_cpt_hierarchy(), etc.

1. ICD9 Hierarchy:
   - Với mỗi ICD9 code (ví dụ: "40190"):
     * Tạo 3-digit parent: "401"
     * Tạo 4-digit parent: "4019"
     * Kết nối: ICD9_ROOT → 401 → 4019 → 40190

2. ATC Hierarchy:
   - Với mỗi ATC code:
     * Tạo intermediate levels
     * Kết nối: ATC_ROOT → DRUG_ → DRUG_X → DRUG_XXXXX

3. LAB Hierarchy:
   - Với mỗi lab code:
     * Lấy category từ D_LABITEMS
     * Kết nối: LAB_ROOT → LAB_Chemistry → 50868

4. Tất cả edges có edge_type='hierarchical'
```

#### **Ví dụ cụ thể**:
```
ICD9 Hierarchy:
  ICD9_ROOT (level 0)
    └── 427 (level 3) - "Diseases of circulatory system"
        └── 4273 (level 4) - "Other cardiac dysrhythmias"
            └── 42731 (level 5) - "Atrial fibrillation"

→ Hierarchical edges:
  ICD9_ROOT → 427 (hierarchical)
  427 → 4273 (hierarchical)
  4273 → 42731 (hierarchical)
```

```
LAB Hierarchy:
  LAB_ROOT (level 0)
    └── LAB_Chemistry (level 1) - Category
        ├── 50882 (level 2) - Sodium test
        ├── 50902 (level 2) - Potassium test
        └── 51275 (level 2) - WBC test

→ Hierarchical edges:
  LAB_ROOT → LAB_Chemistry (hierarchical)
  LAB_Chemistry → 50882 (hierarchical)
  LAB_Chemistry → 50902 (hierarchical)
  LAB_Chemistry → 51275 (hierarchical)
```

#### **Ý nghĩa trong GAT-ETM**:
1. **Semantic structure**:
   - Cho biết codes nào thuộc cùng một nhóm/category
   - Ví dụ: `50882`, `50902`, `51275` đều thuộc `LAB_Chemistry`

2. **Generalization và specialization**:
   - Parent nodes = general concepts (ví dụ: "Diseases of circulatory system")
   - Child nodes = specific concepts (ví dụ: "Atrial fibrillation")
   - Giúp model hiểu được mối quan hệ "is-a" (là một loại của)

3. **Knowledge transfer**:
   - Model có thể học được rằng các codes cùng parent có semantic similarity
   - Ví dụ: Nếu model biết `42731` (Atrial fibrillation) liên quan đến một thuốc, nó có thể suy luận về `4273` (Other cardiac dysrhythmias)

4. **Hierarchical embeddings**:
   - Parent nodes có embeddings đại diện cho toàn bộ subtree
   - Child nodes có embeddings kế thừa từ parent + specific features

#### **Trong visualization**:
- **Màu**: Đen/Xám (black/gray)
- **Style**: Solid lines (đường liền)
- **Vị trí**: Thường ở **peripheral area** hoặc tạo thành **hierarchical chains**
- **Đặc điểm**: Tạo ra **tree-like structure** cho mỗi code type

---

### **3.3. So Sánh 2 Loại Edges**

| Đặc điểm | Cooccurrence Edges | Hierarchical Edges |
|----------|-------------------|-------------------|
| **Nguồn dữ liệu** | Từ EHR data (admissions) | Từ ontology/hierarchy structure |
| **Ý nghĩa** | "Cùng xuất hiện" (co-occur) | "Là một loại của" (is-a) |
| **Tính chất** | Data-driven (từ dữ liệu thực tế) | Knowledge-based (từ ontology) |
| **Weight** | Tần suất đồng xuất hiện | Thường = 1.0 (structural) |
| **Direction** | Undirected (bidirectional) | Directed (parent → child) |
| **Ví dụ** | `42731 <-> 50882` (bệnh và xét nghiệm cùng lúc) | `4273 → 42731` (parent → child) |
| **Mục đích** | Học patterns thực tế | Học semantic structure |
| **Visualization** | Red dashed lines | Black solid lines |

---

### **3.4. Tại Sao Cả 2 Loại Đều Quan Trọng?**

#### **1. Bổ sung cho nhau**:
- **Hierarchical**: Cung cấp semantic structure (codes nào liên quan về mặt ý nghĩa)
- **Cooccurrence**: Cung cấp empirical patterns (codes nào thường xuất hiện cùng nhau)

#### **2. Ví dụ minh họa**:
```
Scenario: Model cần hiểu về code 42731 (Atrial fibrillation)

Hierarchical edges cho biết:
  - 42731 là một loại của 4273 (Other cardiac dysrhythmias)
  - 4273 là một loại của 427 (Diseases of circulatory system)
  → Semantic: "Atrial fibrillation is a type of cardiac dysrhythmia"

Cooccurrence edges cho biết:
  - 42731 thường xuất hiện cùng với 50882 (Sodium test)
  - 42731 thường xuất hiện cùng với DRUG_24 (một loại thuốc)
  → Empirical: "Atrial fibrillation patients often get Sodium tests and Drug_24"

Kết hợp cả 2:
  - Model hiểu được cả semantic structure VÀ empirical patterns
  - Có thể suy luận: Nếu thấy 4273 (parent), có thể cần xét nghiệm 50882
```

#### **3. Trong GAT-ETM Model**:
- **Graph Attention**: Model sử dụng cả 2 loại edges để tính attention weights
- **Embedding learning**: Node embeddings được học từ cả hierarchical structure và cooccurrence patterns
- **Topic modeling**: Topics được học từ cả semantic relationships và empirical co-occurrences

---

### **3.5. Kết Luận**

**Cooccurrence edges**:
- ✅ Phản ánh **thực tế lâm sàng** (clinical reality)
- ✅ Học được **patterns từ dữ liệu** (data-driven)
- ✅ Quan trọng cho **prediction tasks** (dự đoán codes nào sẽ xuất hiện)

**Hierarchical edges**:
- ✅ Phản ánh **cấu trúc tri thức** (knowledge structure)
- ✅ Học được **semantic relationships** (knowledge-based)
- ✅ Quan trọng cho **generalization** (tổng quát hóa sang codes mới)

**Cả 2 loại đều cần thiết** để GAT-ETM có thể:
- Hiểu được cả semantic structure và empirical patterns
- Học được embeddings tốt cho cả codes phổ biến và rare codes
- Tạo ra topics có ý nghĩa cả về mặt semantic và clinical

---

## 4. Vocabulary Sizes - Số Codes Thực Sự Trong Vocabulary

### **ICD: 1**
**Ý nghĩa**: Chỉ có **1 ICD code** thực sự được sử dụng trong vocabulary (training data)

**So sánh với Node Types**:
- Node types: **3 ICD9 nodes** trong graph
- Vocabulary: **1 ICD code** trong vocabulary
- **Chênh lệch**: 3 - 1 = **2 nodes** là hierarchical/parent nodes

**Giải thích**:
- Graph chứa cả leaf node (code thực tế) và parent nodes (intermediate nodes)
- Vocabulary chỉ chứa leaf node (code thực sự xuất hiện trong EHR)
- 2 parent nodes (`4273`, `427`) được thêm vào để tạo hierarchy structure

---

### **CPT: 11**
**Ý nghĩa**: Có **11 CPT codes** trong vocabulary

**So sánh với Node Types**:
- Node types: **1 CPT node** trong graph visualization
- Vocabulary: **11 CPT codes** trong vocabulary
- **Chênh lệch**: 11 - 1 = **10 CPT codes** không được hiển thị trong graph này

**Giải thích**:
- Đây là **ego-graph** (subgraph), chỉ hiển thị nodes có liên kết với selected nodes
- Chỉ 1 CPT code có liên kết với các nodes khác được hiển thị
- 10 CPT codes khác không có liên kết nên không được hiển thị trong visualization này

---

### **ATC: 5**
**Ý nghĩa**: Có **5 ATC codes** thực sự được sử dụng trong vocabulary

**So sánh với Node Types**:
- Node types: **11 ATC nodes** trong graph
- Vocabulary: **5 ATC codes** trong vocabulary
- **Chênh lệch**: 11 - 5 = **6 nodes** là hierarchical/parent nodes

**Giải thích**:
- 5 ATC codes là leaf nodes (codes thực tế)
- 6 nodes còn lại là parent nodes trong ATC hierarchy
- Parent nodes giúp tạo semantic relationships giữa các ATC codes

---

### **LAB: 10**
**Ý nghĩa**: Có **10 LAB codes** thực sự được sử dụng trong vocabulary

**So sánh với Node Types**:
- Node types: **13 LAB nodes** trong graph
- Vocabulary: **10 LAB codes** trong vocabulary
- **Chênh lệch**: 13 - 10 = **3 nodes** là hierarchical/parent nodes

**Giải thích**:
- 10 LAB codes là leaf nodes (codes thực tế)
- 3 nodes còn lại có thể là:
  - Category nodes: `LAB_Chemistry`, `LAB_Hematology`
  - Parent nodes trong LAB hierarchy

---

## 5. Tổng Kết và So Sánh

### **Tổng số nodes**:
- **Graph nodes**: 28 nodes
- **Vocabulary nodes**: 1 + 11 + 5 + 10 = **27 nodes**
- **Chênh lệch**: 28 - 27 = **1 node**

**Giải thích chênh lệch**:
- 1 node có thể là:
  - Root node (ví dụ: `ICD9_ROOT`, `LAB_ROOT`, `ATC_ROOT`)
  - Hoặc một parent node không được tính vào vocabulary

---

### **Bảng So Sánh Chi Tiết**:

| Code Type | Graph Nodes | Vocabulary Size | Chênh Lệch | Giải Thích |
|-----------|-------------|-----------------|------------|------------|
| **ICD** | 3 | 1 | +2 | 2 parent nodes trong hierarchy |
| **CPT** | 1 | 11 | -10 | Ego-graph chỉ hiển thị 1 node có liên kết |
| **ATC** | 11 | 5 | +6 | 6 parent nodes trong ATC hierarchy |
| **LAB** | 13 | 10 | +3 | 3 category/parent nodes |
| **TOTAL** | **28** | **27** | **+1** | 1 root/parent node |

---

## 6. Ý Nghĩa Trong Context của GAT-ETM

### **Tại sao có sự khác biệt?**

1. **Graph chứa cả hierarchy structure**:
   - Parent nodes giúp model hiểu semantic relationships
   - Ví dụ: `DRUG_59` (parent) → `DRUG_59178` (child) cho thấy mối quan hệ phân cấp

2. **Vocabulary chỉ chứa leaf nodes**:
   - Chỉ các codes thực sự xuất hiện trong EHR data
   - Được sử dụng để tạo BoW matrices cho training

3. **Ego-graph chỉ hiển thị subset**:
   - Chỉ hiển thị nodes có liên kết với selected nodes
   - Không hiển thị isolated nodes (nodes không có edges)

---

### **Cách sử dụng trong Training**:

1. **Graph embeddings**:
   - Tất cả 28 nodes đều có embeddings (từ Node2Vec)
   - Embeddings được renumber theo vocabulary order

2. **Vocabulary order**:
   - Chỉ 27 vocabulary nodes được sử dụng trong BoW matrices
   - Thứ tự: ICD (0-0), CPT (1-11), ATC (12-16), LAB (17-26)

3. **Metadata file**:
   ```
   ['icd', 'cpt', 'atc', 'lab']
   [1, 11, 5, 10]
   [1, 1, 1, 1]
   ['*', '*', '*', '*']
   ```

---

## 7. Kết Luận

### **Key Takeaways**:

1. **Graph nodes (28) ≠ Vocabulary nodes (27)**:
   - Graph chứa cả hierarchy structure
   - Vocabulary chỉ chứa leaf nodes thực tế

2. **Node types count ≠ Vocabulary sizes**:
   - Node types: Đếm tất cả nodes trong graph
   - Vocabulary sizes: Đếm chỉ leaf nodes trong vocabulary

3. **Ego-graph chỉ hiển thị subset**:
   - Chỉ hiển thị nodes có liên kết
   - Không hiển thị isolated nodes

4. **Hierarchical nodes quan trọng**:
   - Giúp model hiểu semantic relationships
   - Tạo structure cho knowledge graph

### **Khi tạo Metadata File**:

✅ **Sử dụng Vocabulary Sizes** (không phải Node Types):
```
['icd', 'cpt', 'atc', 'lab']
[1, 11, 5, 10]  # ← Vocabulary sizes, không phải node counts
[1, 1, 1, 1]
['*', '*', '*', '*']
```

❌ **Không sử dụng Node Types**:
```
[3, 1, 11, 13]  # ← Sai! Đây là node counts, không phải vocab sizes
```

---

## 8. Ví Dụ Cụ Thể Từ Visualization

### **ICD9 Nodes**:
- **Graph**: 3 nodes (`42731`, `4273`, `427`)
- **Vocabulary**: 1 node (`42731` - leaf node)
- **Parent nodes**: `4273` (4-digit), `427` (3-digit)

### **ATC Nodes**:
- **Graph**: 11 nodes (8 có prefix DRUG_, 3 không có prefix)
- **Vocabulary**: 5 nodes (leaf nodes)
- **Parent nodes**: 6 nodes (hierarchical structure)

### **LAB Nodes**:
- **Graph**: 13 nodes
- **Vocabulary**: 10 nodes (leaf nodes)
- **Category/Parent nodes**: 3 nodes (`LAB_Chemistry`, `LAB_Hematology`, và 1 node khác)

### **CPT Nodes**:
- **Graph**: 1 node (chỉ hiển thị node có liên kết)
- **Vocabulary**: 11 nodes (tất cả CPT codes)
- **Giải thích**: 10 CPT codes khác không có liên kết nên không hiển thị trong ego-graph
