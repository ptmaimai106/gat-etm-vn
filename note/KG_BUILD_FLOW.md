# Knowledge Graph Build Flow - MIMIC-III

### Chi tiết cách lấy dữ liệu:

1. **ICD9 Codes** (dòng 133):
   ```python
   icd9_codes = set(self.diagnoses_icd['icd9_code'].dropna().unique())
   ```
   → Lấy **TẤT CẢ** unique ICD9 codes từ toàn bộ table `DIAGNOSES_ICD`

2. **CPT Codes** (dòng 210):
   ```python
   cpt_codes.update(self.cpt_events['cpt_cd'].dropna().unique())
   ```
   → Lấy **TẤT CẢ** unique CPT codes từ table `CPTEVENTS`

3. **ATC/Drugs** (dòng 261):
   ```python
   drugs = self.prescriptions['drug_name_generic'].dropna().unique()
   ```
   → Lấy **TẤT CẢ** unique drugs từ toàn bộ table `PRESCRIPTIONS`

4. **Lab Codes** (dòng 328):
   ```python
   lab_itemids = set(self.lab_events['itemid'].dropna().unique())
   ```
   → Lấy **TẤT CẢ** unique lab itemids từ toàn bộ table `LABEVENTS`

5. **Co-occurrence Edges** (dòng 388-442, 449):
   ```python
   for _, row in self.diagnoses_icd.iterrows():  # TẤT CẢ rows
   for _, row in self.prescriptions.iterrows():  # TẤT CẢ rows
   for hadm_id in tqdm(set(list(hadm_to_icd.keys()) + list(hadm_to_atc.keys()))):
   ```
   → Xử lý **TẤT CẢ** admissions để tạo co-occurrence edges

---

## 🔄 Flow xây dựng Knowledge Graph

### **STEP 1: Load MIMIC-III Tables** (`load_mimic_tables()`)
```
📂 Load các tables:
   ├── D_ICD_DIAGNOSES.csv (dictionary)
   ├── DIAGNOSES_ICD.csv (actual diagnoses)
   ├── D_ICD_PROCEDURES.csv (dictionary)
   ├── PROCEDURES_ICD.csv (actual procedures)
   ├── D_CPT.csv (dictionary, optional)
   ├── CPTEVENTS.csv (CPT events, optional)
   ├── PRESCRIPTIONS.csv (drug prescriptions)
   ├── LABEVENTS.csv (lab test results)
   ├── D_LABITEMS.csv (lab items dictionary)
   └── ADMISSIONS.csv (admission records)
```

### **STEP 2: Build Hierarchical Structures**

#### 2.1 ICD9 Hierarchy (`build_icd9_hierarchy()`)
```
Input: Tất cả unique ICD9 codes từ DIAGNOSES_ICD
Process:
  1. Normalize codes (remove dots for internal use)
  2. Build prefix hierarchy: 3-digit → 4-digit → 5-digit
     Example: "250" → "2501" → "25010"
  3. Add root node: ICD9_ROOT
  4. Create edges: parent → child (hierarchical)
  5. Store vocab (leaf nodes only, keep original format)

Output:
  - Nodes: ICD9_ROOT + all prefix levels + leaf codes
  - Edges: hierarchical edges between parent-child
  - Vocab: sorted list of unique ICD9 codes (original format)

? Tại sao phải "Build prefix hierarchy", hiện tai data ở cột icd9_code khá đa dạng:
   - 3 chữ số
   - 4 chữ số
   - 5 chữ số
   - có cả ký tự  
```

#### 2.2 CPT Hierarchy (`build_cpt_hierarchy()`)
```
Input: Tất cả unique CPT codes từ CPTEVENTS
Process:
  1. Group by first digit (section)
  2. Build hierarchy: CPT_ROOT → CPT_X → code
  3. Create edges: section → code

Output:
  - Nodes: CPT_ROOT + sections + codes
  - Edges: hierarchical edges
  - Vocab: sorted CPT codes
```

#### 2.3 ATC/Drug Hierarchy (`extract_drugs_and_map_to_atc()`)
```
Input: Tất cả unique drugs từ PRESCRIPTIONS
Process:
  1. Map drug names → ATC codes (placeholder, hash-based)
  2. Build hierarchy: ATC_ROOT → DRUG_ → DRUG_X → DRUG_XXXXX
  3. Create edges: hierarchical

Output:
  - Nodes: ATC_ROOT + intermediate levels + ATC codes
  - Edges: hierarchical edges
  - Vocab: sorted ATC codes
  - Mapping: drug_to_atc dictionary
```

#### 2.4 Lab Hierarchy (`extract_lab_codes()`)
```
Input: Tất cả unique lab itemids từ LABEVENTS
Process:
  1. Get categories from D_LABITEMS
  2. Build hierarchy: LAB_ROOT → LAB_CATEGORY → itemid
  3. Create edges: category → itemid

Output:
  - Nodes: LAB_ROOT + categories + itemids
  - Edges: hierarchical edges
  - Vocab: sorted lab itemids
```

**Sau Step 2: Graph có hierarchical structure cho 4 loại codes**

---

### **STEP 3: Build Co-occurrence Edges** (`build_cooccurrence_edges()`)
```
Input: Tất cả admissions từ các tables
Process:
  1. Group codes by HADM_ID (hospital admission ID):
     - hadm_to_icd: map admission → ICD9 codes
     - hadm_to_atc: map admission → ATC codes  
     - hadm_to_cpt: map admission → CPT codes
     - hadm_to_lab: map admission → Lab codes
  
  2. For each admission, create co-occurrence pairs:
     - ICD9 ↔ ATC (diagnosis-drug)
     - ICD9 ↔ CPT (diagnosis-procedure)
     - ICD9 ↔ Lab (diagnosis-lab test)
     - ICD9 ↔ ICD9 (diagnosis-diagnosis)
  
  3. Count co-occurrence frequency
  4. Add edges if count >= min_cooccurrence (default=1)

Output:
  - Additional edges: co-occurrence edges with weights
  - Edge attributes: edge_type='cooccurrence', weight=count
```

**Sau Step 3: Graph có thêm edges thể hiện quan hệ đồng xuất hiện**

---

### **STEP 4: Augment Graph** (`augment_graph()`, optional)
```
Input: Graph từ Step 3
Process:
  1. For each leaf node (ICD9, ATC, CPT):
     - Find all ancestors in hierarchy
     - Add skip connections to ancestors
     - Weight = 0.9^(distance)
  
  2. Example: 
     Node "25010" → ancestors: ["2501", "250", "ICD9_ROOT"]
     Add edges: 25010 → 2501 (weight=0.9)
               25010 → 250 (weight=0.81)
               25010 → ICD9_ROOT (weight=0.729)

Output:
  - Additional edges: augmented/skip edges
  - Edge attributes: edge_type='augmented', weight=decay_factor
```

**Sau Step 4: Graph có skip connections giữa leaf nodes và ancestors**

---

### **STEP 5: Generate Node2Vec Embeddings** (`generate_embeddings()`)
```
Input: Complete graph (hierarchical + co-occurrence + augmented edges)
Process:
  1. Convert to undirected graph
  2. Initialize Node2Vec:
     - dimensions: 256 (default)
     - walk_length: 20
     - num_walks: 10
     - window: 8
     - p=1, q=1
  
  3. Train Node2Vec model
  4. Extract embeddings for all nodes

Output:
  - Embeddings matrix: (num_nodes, embedding_dim)
  - Each node has a vector representation
```

**Sau Step 5: Mỗi node có embedding vector**

---

### **STEP 6: Renumber Nodes by Vocab** (`renumber_nodes_by_vocab()`)
```
Input: Graph + embeddings
Process:
  1. Create renumbering mapping:
     - Index 0-N: ICD9 vocab nodes (sorted)
     - Index N-M: CPT vocab nodes (sorted)
     - Index M-K: ATC vocab nodes (sorted)
     - Index K-L: Lab vocab nodes (sorted)
     - Index L+: Other nodes (roots, intermediate levels)
  
  2. Relabel graph nodes: old_node → new_index
  3. Reorder embeddings accordingly
  4. Create reverse mapping: graphnode_vocab (index → original_code)

Output:
  - Renumbered graph: nodes are now indices 0, 1, 2, ...
  - Reordered embeddings: embeddings[i] corresponds to node i
  - Mapping: graphnode_vocab dictionary
```

**Sau Step 6: Nodes được đánh số lại theo thứ tự vocab**

---

### **STEP 7: Save Outputs** (`save_outputs()`)
```
Save files:
  1. Graph: icdatc_graph_{params}_renumbered_by_vocab.pkl
     - NetworkX graph with renumbered nodes
  
  2. Embeddings: icdatc_embed_{params}_by_vocab.pkl
     - NumPy array: (num_nodes, embedding_dim)
  
  3. Vocab mapping: graphnode_vocab.pkl
     - Dictionary: {node_index: original_code}
  
  4. Vocab info: vocab_info.pkl
     - Dictionary: {'icd': [...], 'cpt': [...], 'atc': [...], 'lab': [...]}
```

---

## 📊 Tóm tắt Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Load Tables                                         │
│ → Load tất cả CSV files từ MIMIC-III                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Build Hierarchies                                   │
│ → ICD9: Root → 3-digit → 4-digit → 5-digit                 │
│ → CPT: Root → Section → Code                                │
│ → ATC: Root → Level → Drug code                             │
│ → Lab: Root → Category → Itemid                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Co-occurrence Edges                                 │
│ → Group by HADM_ID                                          │
│ → Create edges: ICD9↔ATC, ICD9↔CPT, ICD9↔Lab, ICD9↔ICD9    │
│ → Weight = co-occurrence count                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Augment Graph (optional)                            │
│ → Add skip connections: leaf → ancestors                    │
│ → Weight = 0.9^(distance)                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Generate Embeddings                                 │
│ → Node2Vec training                                         │
│ → Extract embeddings for all nodes                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Renumber Nodes                                      │
│ → Reorder: ICD9 → CPT → ATC → Lab → Others                  │
│ → Relabel graph nodes                                       │
│ → Reorder embeddings                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Save Files                                           │
│ → Graph.pkl, Embeddings.pkl, Vocab.pkl, VocabInfo.pkl      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Points

1. **Không có sampling**: Script xử lý **TẤT CẢ** records từ tables
2. **Hierarchical structure**: Mỗi code type có cây phân cấp
3. **Co-occurrence**: Edges được tạo từ đồng xuất hiện trong cùng admission
4. **Augmentation**: Skip connections giúp model học tốt hơn
5. **Node2Vec**: Embeddings capture graph structure
6. **Renumbering**: Nodes được sắp xếp theo vocab order để dễ sử dụng trong training


