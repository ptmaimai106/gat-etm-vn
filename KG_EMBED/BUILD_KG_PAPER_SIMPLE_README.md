# Build Simple Knowledge Graph for Manual Verification

## Tổng Quan

Script `build_kg_paper_simple.py` tạo một Knowledge Graph đơn giản với chỉ vài ICD và ATC codes để:
- ✅ **Manual verification**: Kiểm tra thủ công cấu trúc graph
- ✅ **Debug**: Xác nhận các connections đúng
- ✅ **Testing**: Test workflow trước khi build full KG

**Khác biệt với `build_kg_paper.py`**:
- Chỉ sử dụng **1-3 ICD codes** (top frequent)
- Chỉ sử dụng **5-10 ATC codes** (top frequent)
- In ra **tất cả nodes và edges** để manual verification
- Visualization chi tiết hơn

---

## Cách Sử Dụng

### **1. Build Simple KG**

#### **Cơ bản (1 ICD, 5 ATC)**:
```bash
python KG_EMBED/build_kg_paper_simple.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper_simple
```

#### **Với nhiều codes hơn**:
```bash
python KG_EMBED/build_kg_paper_simple.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper_simple \
  --num_icd 3 \
  --num_atc 10
```

#### **Với ICD-ATC mapping file**:
```bash
python KG_EMBED/build_kg_paper_simple.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper_simple \
  --num_icd 1 \
  --num_atc 5 \
  --icd_atc_mapping data/icd_atc_mapping.csv
```

---

### **2. Visualize Simple Graph**

#### **Sử dụng script visualize riêng**:
```bash
python visualize/visualize_graph_simple.py \
  --graph_file embed_paper_simple/icdatc_graph_simple_8_20_10_256_renumbered_by_vocab.pkl \
  --vocab_info_file embed_paper_simple/vocab_info.pkl \
  --output visualize/graph_visualization_paper_simple.png
```

#### **Hoặc sử dụng script visualize chung**:
```bash
python visualize/visualize_graph.py \
  --graph_file embed_paper_simple/icdatc_graph_simple_8_20_10_256_renumbered_by_vocab.pkl \
  --vocab_info_file embed_paper_simple/vocab_info.pkl \
  --output visualize/graph_visualization_paper_simple.png
```

---

## Output

### **1. Console Output**

Script sẽ in ra:
- **Tất cả nodes** với type, level, code, degree
- **Tất cả edges** với type, weight, relation
- **Statistics** chi tiết

Ví dụ:
```
Nodes by Type:
  ICD9 (4 nodes):
    - ICD9_ROOT: level=0, code=None, degree=1
    - 427: level=3, code=None, degree=2
    - 4273: level=4, code=None, degree=2
    - 42731: level=5, code=42731, degree=3

Edges by Type:
  HIERARCHICAL (3 edges):
    - ICD9_ROOT <-> 427
    - 427 <-> 4273
    - 4273 <-> 42731
  
  ICD_ATC_RELATION (2 edges):
    - 42731 <-> DRUG_12345, relation=treats
    - 42731 <-> DRUG_67890, relation=treats
```

### **2. Visualization**

- **Nodes**: 
  - ICD9: Hình vuông đỏ với prefix `ICD_`
  - ATC: Hình tam giác xanh dương với prefix `ATC_` hoặc `DRUG_`
- **Edges**:
  - Hierarchical: Đường liền đen
  - ICD-ATC relation: Đường đứt đỏ
  - Augmented: Đường chấm tím
- **Labels**: Tất cả nodes đều có labels
- **Statistics box**: Hiển thị thống kê chi tiết

---

## Manual Verification Checklist

Sau khi build và visualize, kiểm tra:

### **1. ICD Hierarchy**:
- [ ] ICD9_ROOT → 3-digit → 4-digit → 5-digit (leaf node)
- [ ] Tất cả hierarchical edges được tạo đúng
- [ ] Level được assign đúng

### **2. ATC Hierarchy**:
- [ ] ATC_ROOT → intermediate levels → leaf ATC code
- [ ] Tất cả hierarchical edges được tạo đúng
- [ ] Level được assign đúng

### **3. ICD-ATC Relations**:
- [ ] Các ICD codes có connections đến ATC codes
- [ ] Relation type được set đúng (treats, etc.)
- [ ] Không có duplicate edges

### **4. Augmentation**:
- [ ] Mỗi node có skip connections đến tất cả ancestors
- [ ] Weight decay đúng (0.9^distance)
- [ ] Không có cycles

### **5. Vocabulary Order**:
- [ ] ICD vocabulary nodes được đánh số trước (0 → V_icd-1)
- [ ] ATC vocabulary nodes tiếp theo (V_icd → V_icd+V_atc-1)
- [ ] Hierarchical nodes được đánh số sau

---

## Ví Dụ Workflow

### **Step 1: Build Simple KG**
```bash
python KG_EMBED/build_kg_paper_simple.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper_simple \
  --num_icd 1 \
  --num_atc 5
```

**Output**:
- Graph file: `embed_paper_simple/icdatc_graph_simple_8_20_10_256_renumbered_by_vocab.pkl`
- Embeddings: `embed_paper_simple/icdatc_embed_simple_8_20_10_256_by_vocab.pkl`
- Vocab info: `embed_paper_simple/vocab_info.pkl`

### **Step 2: Visualize**
```bash
python visualize/visualize_graph_simple.py \
  --graph_file embed_paper_simple/icdatc_graph_simple_8_20_10_256_renumbered_by_vocab.pkl \
  --vocab_info_file embed_paper_simple/vocab_info.pkl \
  --output visualize/graph_visualization_paper_simple.png
```

### **Step 3: Manual Verification**
1. Mở visualization image
2. Kiểm tra console output (tất cả nodes và edges)
3. Verify:
   - ICD hierarchy structure
   - ATC hierarchy structure
   - ICD-ATC connections
   - Augmentation edges

### **Step 4: Nếu OK, build full KG**
```bash
python KG_EMBED/build_kg_paper.py \
  --mimic_path mimic-iii-clinical-database-demo-1.4 \
  --output_dir embed_paper \
  --icd_atc_mapping data/icd_atc_mapping.csv
```

---

## Expected Output Example

Với `--num_icd 1 --num_atc 5`, bạn sẽ thấy:

**Nodes**:
- ICD9: ~4-5 nodes (ICD9_ROOT + prefixes + leaf)
- ATC: ~8-10 nodes (ATC_ROOT + intermediate + 5 leaf codes)
- **Total**: ~12-15 nodes

**Edges**:
- Hierarchical: ~8-12 edges
- ICD-ATC relations: ~1-5 edges (tùy co-occurrence)
- Augmented: ~10-20 edges (skip connections)
- **Total**: ~20-40 edges

**Visualization**:
- Graph nhỏ, dễ nhìn
- Tất cả nodes có labels rõ ràng
- Dễ dàng verify connections manually

---

## Troubleshooting

### **Không có ICD-ATC relations**:
- Kiểm tra xem có mapping file không
- Kiểm tra xem ICD và ATC codes có xuất hiện cùng nhau trong admissions không
- Tăng `--num_icd` và `--num_atc` để có nhiều codes hơn

### **Graph quá nhỏ**:
- Tăng `--num_icd` và `--num_atc`
- Ví dụ: `--num_icd 3 --num_atc 10`

### **Visualization không rõ**:
- Tăng `node_size` trong script
- Tăng `font_size` trong script
- Kiểm tra output resolution (dpi=300)

---

## Next Steps

Sau khi verify simple KG:
1. ✅ Build full KG với `build_kg_paper.py`
2. ✅ Build ICD-ATC mapping với `build_icd_atc_mapping.py`
3. ✅ Train model với `main_getm_mimic.py`
