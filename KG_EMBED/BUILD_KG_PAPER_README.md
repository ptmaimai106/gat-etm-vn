# Build Knowledge Graph Based on GAT-ETM Paper

## Tổng Quan

File `build_kg_paper.py` được tạo để build Knowledge Graph **chính xác theo paper gốc** của GAT-ETM:

**Paper**: "Modeling electronic health record data using an end-to-end knowledge-graph-informed topic model"  
**Journal**: Nature Scientific Reports (2022) 12:17868  
**Link**: https://www.nature.com/articles/s41598-022-22956-w

---

## So Sánh 2 Cách Build KG

### **1. build_kg_paper.py** (Theo Paper Gốc)

#### **Node Types**:
- ✅ **ICD-9 codes** (diagnosis codes)
- ✅ **ATC codes** (drug codes)
- ❌ **KHÔNG có** CPT codes
- ❌ **KHÔNG có** LAB codes

#### **Edge Types**:
1. **ICD Hierarchy** (ICD-ICD edges)
   - Parent-child relationships trong ICD-9 taxonomy
   - Augmented: connect each node to all ancestral nodes

2. **ATC Hierarchy** (ATC-ATC edges)
   - Parent-child relationships trong WHO ATC classification
   - Augmented: connect each node to all ancestral nodes

3. **ICD-ATC Relations** (ICD-ATC edges)
   - Từ external knowledge base: http://hulab.rxnfinder.org/mia/
   - Drug-disease relationships (treats, contraindicated, etc.)
   - **KHÔNG phải** từ co-occurrence trong EHR data

#### **Augmentation**:
- Connect each node to **all of its ancestral nodes** (skip connections)
- Weight decay: `0.9^distance`

#### **Không có**:
- ❌ Co-occurrence edges từ EHR data
- ❌ CPT codes
- ❌ LAB codes

---

### **2. build_kg_mimic.py** (Mở Rộng cho MIMIC-III)

#### **Node Types**:
- ✅ **ICD-9 codes**
- ✅ **ATC codes**
- ✅ **CPT codes** (procedure codes)
- ✅ **LAB codes** (laboratory codes)

#### **Edge Types**:
1. **Hierarchical Edges**:
   - ICD-9 hierarchy
   - CPT hierarchy
   - ATC hierarchy
   - LAB hierarchy

2. **Co-occurrence Edges** (từ EHR data):
   - ICD ↔ ATC (cùng admission)
   - ICD ↔ CPT (cùng admission)
   - ICD ↔ LAB (cùng admission)
   - ICD ↔ ICD (nhiều diagnoses cùng lúc)

3. **Augmented Edges** (optional):
   - Skip connections trong hierarchy

#### **Augmentation**:
- Skip connections trong hierarchy trees
- Co-occurrence edges từ thực tế EHR data

---

## Sự Khác Biệt Chính

| Đặc điểm | build_kg_paper.py | build_kg_mimic.py |
|----------|-------------------|-------------------|
| **Node Types** | ICD + ATC only | ICD + ATC + CPT + LAB |
| **ICD-ATC Relations** | External KB (MIA) | Co-occurrence từ EHR |
| **Co-occurrence** | ❌ Không có | ✅ Có (từ admissions) |
| **CPT Codes** | ❌ Không có | ✅ Có |
| **LAB Codes** | ❌ Không có | ✅ Có |
| **Augmentation** | Skip to ancestors | Skip + co-occurrence |

---

## Cách Sử Dụng build_kg_paper.py

### **1. Cài Đặt Dependencies**:
```bash
pip install networkx pandas numpy node2vec tqdm
```

### **2. Chuẩn Bị Dữ Liệu**:

#### **Bắt buộc**:
- MIMIC-III `DIAGNOSES_ICD.csv` (để lấy ICD-9 codes)

#### **Tùy chọn**:
- ICD-ATC mapping file (CSV format):
  ```
  ICD_CODE,ATC_CODE,RELATION
  250,A10BA02,treats
  4019,C02CA01,treats
  ...
  ```
  
  Nếu không có, script sẽ tạo placeholder relations.

### **3. Chạy Script**:

#### **Cơ bản**:
```bash
python KG_EMBED/build_kg_paper.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper
```

#### **Với ICD-ATC mapping file**:
```bash
python KG_EMBED/build_kg_paper.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper \
  --icd_atc_mapping data/icd_atc_mapping.csv
```

#### **Với custom parameters**:
```bash
python KG_EMBED/build_kg_paper.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper \
  --embedding_dim 256 \
  --window 8 \
  --walk_length 20 \
  --num_walks 10
```

---

## Output Files

Sau khi chạy, các file sau sẽ được tạo trong `output_dir`:

1. **Graph file**: `icdatc_graph_{window}_{walk_length}_{num_walks}_{embedding_dim}_renumbered_by_vocab.pkl`
   - NetworkX graph object
   - Nodes đã được renumber theo vocabulary order

2. **Embeddings file**: `icdatc_embed_{window}_{walk_length}_{num_walks}_{embedding_dim}_by_vocab.pkl`
   - NumPy array: `(num_nodes, embedding_dim)`
   - Node embeddings từ Node2Vec

3. **Vocab mapping**: `graphnode_vocab.pkl`
   - Dictionary: `{node_index: code_string}`
   - Mapping từ node index về code gốc

4. **Vocab info**: `vocab_info.pkl`
   - Dictionary: `{'icd': [...], 'atc': [...]}`
   - Vocabulary lists cho ICD và ATC

---

## Lưu Ý Quan Trọng

### **1. ICD-ATC Relations**:
- Paper sử dụng external knowledge base: http://hulab.rxnfinder.org/mia/
- Script hiện tại có placeholder nếu không có mapping file
- **Để production**: Cần download/parse MIA database hoặc sử dụng RxNorm API

### **2. ATC Codes**:
- Paper sử dụng WHO ATC classification thực tế
- Script hiện tại tạo placeholder từ drug names (hash-based)
- **Để production**: Cần mapping từ RxNorm → WHO ATC codes

### **3. Vocabulary Order**:
- ICD vocabulary nodes được đánh số trước (0 → V_icd-1)
- ATC vocabulary nodes tiếp theo (V_icd → V_icd+V_atc-1)
- Hierarchical nodes được đánh số sau

### **4. Graph Structure**:
- Graph là **undirected** (như paper)
- Edge types: `hierarchical`, `icd_atc_relation`, `augmented`
- Augmented edges có weight decay: `0.9^distance`

---

## Sử Dụng Output trong Training

Sau khi build xong, sử dụng trong `main_getm_mimic.py`:

```bash
python main_getm_mimic.py \
  --data_path data/ \
  --kg_embed_dir embed_paper \
  --epochs 50
```

**Metadata file** sẽ tự động được tạo với:
```
['icd', 'atc']
[vocab_size_icd, vocab_size_atc]
[1, 1]
['*', '*']
```

---

## So Sánh Kết Quả

### **build_kg_paper.py**:
- Graph nhỏ hơn (chỉ ICD + ATC)
- Phù hợp với paper gốc
- Cần external ICD-ATC mapping
- Không có co-occurrence từ EHR

### **build_kg_mimic.py**:
- Graph lớn hơn (ICD + ATC + CPT + LAB)
- Phù hợp với MIMIC-III dataset
- Co-occurrence từ EHR data
- Không cần external mapping (nhưng có thể thêm)

---

## Khi Nào Dùng Cái Nào?

### **Dùng build_kg_paper.py khi**:
- ✅ Muốn reproduce kết quả paper gốc
- ✅ Chỉ có ICD và ATC codes
- ✅ Có access đến external ICD-ATC knowledge base
- ✅ Muốn so sánh với baseline paper

### **Dùng build_kg_mimic.py khi**:
- ✅ Làm việc với MIMIC-III dataset đầy đủ
- ✅ Cần CPT và LAB codes
- ✅ Muốn sử dụng co-occurrence từ EHR data
- ✅ Muốn mở rộng model với nhiều code types

---

## References

1. **Paper**: Zou, Y. et al. Modeling electronic health record data using an end-to-end knowledge-graph-informed topic model. *Sci Rep* **12**, 17868 (2022). https://doi.org/10.1038/s41598-022-22956-w

2. **ICD-9**: https://icdlist.com/icd-9/index

3. **WHO ATC**: https://www.whocc.no/atc_ddd_index/

4. **MIA Database**: http://hulab.rxnfinder.org/mia/
