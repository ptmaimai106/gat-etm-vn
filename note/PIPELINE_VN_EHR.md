# Pipeline Áp dụng GAT-ETM trên Hồ sơ Bệnh án Điện tử Việt Nam
## Timeline: 2-3 tháng

---

## 📋 Tổng quan

Pipeline này hướng dẫn cách áp dụng GAT-ETM (Graph Attention-Embedded Topic Model) trên dữ liệu hồ sơ bệnh án điện tử (EHR) của Việt Nam, dựa trên:
- Codebase GAT-ETM đã có (train trên PopHR)
- Kinh nghiệm test trên MIMIC-III
- Xử lý dữ liệu tiếng Việt và format đặc thù của Việt Nam

---

## 🎯 Mục tiêu

**Mục tiêu cuối cùng:** Có một hệ thống GAT-ETM hoạt động trên EHR Việt Nam với:
- Knowledge Graph được build từ dữ liệu Việt Nam
- BoW matrices được chuẩn bị đúng format
- Model train được và cho kết quả tốt
- Có thể evaluate và visualize kết quả

---

## 📅 Timeline Chi tiết

### **THÁNG 1: Data Understanding & Preparation**

#### **Tuần 1-2: Data Exploration & Mapping**

**Mục tiêu:** Hiểu rõ cấu trúc dữ liệu EHR Việt Nam

**Tasks:**

1. **Data Inventory (3-5 ngày)**
   ```python
   # Checklist cần thu thập:
   - [ ] Các bảng dữ liệu có sẵn (tương tự MIMIC-III)
   - [ ] Format dữ liệu (CSV, SQL, JSON, Excel?)
   - [ ] Số lượng records (patients, visits, codes)
   - [ ] Cấu trúc schema database
   - [ ] Sample data (10-20 records để test)
   ```

2. **Code System Mapping (5-7 ngày)**
   ```python
   # Xác định các loại codes trong EHR VN:
   
   # A. Diagnosis Codes (Chẩn đoán)
   - [ ] ICD-10 VN? ICD-9? Hay hệ thống riêng?
   - [ ] Format: "A00.0" hay "A000"?
   - [ ] Có hierarchy không? (parent-child relationships)
   - [ ] Mapping sang ICD-10 quốc tế? (nếu cần)
   
   # B. Drug Codes (Thuốc)
   - [ ] Tên thuốc tiếng Việt hay tiếng Anh?
   - [ ] Có mã thuốc (ATC, NDC, hay mã nội bộ)?
   - [ ] Cần map sang ATC không?
   - [ ] Có thông tin generic name không?
   
   # C. Procedure Codes (Thủ thuật)
   - [ ] CPT? ICD-10-PCS? Hay mã riêng?
   - [ ] Có hierarchy không?
   
   # D. Lab Codes (Xét nghiệm)
   - [ ] Tên xét nghiệm tiếng Việt?
   - [ ] Có mã LOINC không?
   - [ ] Có category/group không?
   
   # E. Other Codes
   - [ ] Vitals (dấu hiệu sinh tồn)?
   - [ ] Demographics?
   - [ ] Temporal information (visits, dates)?
   ```

3. **Data Quality Assessment (2-3 ngày)**
   ```python
   # Đánh giá chất lượng dữ liệu:
   - [ ] Missing values rate
   - [ ] Duplicate records
   - [ ] Data consistency
   - [ ] Temporal coverage (số năm dữ liệu)
   - [ ] Patient coverage (số lượng patients)
   - [ ] Code frequency distribution
   ```

**Deliverables:**
- Document: `DATA_INVENTORY.md` mô tả cấu trúc dữ liệu
- Document: `CODE_MAPPING.md` mapping các code systems
- Script: `explore_data.py` để explore dữ liệu

---

#### **Tuần 3-4: Data Extraction & Preprocessing**

**Mục tiêu:** Extract và preprocess dữ liệu thành format chuẩn

**Tasks:**

1. **Build Data Extraction Script (5-7 ngày)**
   ```python
   # File: extract_vn_ehr.py
   
   def extract_patient_codes(vn_ehr_path):
       """
       Extract codes từ EHR Việt Nam
       Tương tự extract_codes_from_mimic() nhưng adapt cho VN
       """
       # 1. Load tables từ database/file
       # 2. Extract diagnosis codes (có thể cần normalize tiếng Việt)
       # 3. Extract drug codes (map tên thuốc VN -> codes)
       # 4. Extract procedure codes
       # 5. Extract lab codes
       # 6. Group by patient/visit
       pass
   
   # Xử lý tiếng Việt:
   - [ ] Normalize text (lowercase, remove accents?)
   - [ ] Map tên thuốc VN -> generic names
   - [ ] Map tên bệnh VN -> ICD codes (nếu có)
   - [ ] Handle abbreviations
   ```

2. **Code Normalization & Mapping (5-7 ngày)**
   ```python
   # File: normalize_vn_codes.py
   
   # A. Diagnosis Normalization
   def normalize_diagnosis_vn(diagnosis_text):
       """
       Normalize diagnosis từ tiếng Việt
       - Remove accents? (tùy chọn)
       - Map tên bệnh -> ICD code
       - Handle variations
       """
       pass
   
   # B. Drug Mapping
   def map_drug_vn_to_atc(drug_name_vn):
       """
       Map tên thuốc tiếng Việt -> ATC code
       Options:
       1. Sử dụng drug dictionary VN (nếu có)
       2. Map qua English name -> ATC
       3. Sử dụng embedding similarity
       4. Manual mapping cho top drugs
       """
       pass
   
   # C. Lab Normalization
   def normalize_lab_vn(lab_name_vn):
       """
       Normalize lab test names
       - Map tên VN -> standard codes
       - Group similar tests
       """
       pass
   ```

3. **Create Vocabulary (3-5 ngày)**
   ```python
   # File: create_vocab_vn.py
   
   def create_vocabulary_vn(patient_codes):
       """
       Tạo vocabulary từ extracted codes
       - Filter rare codes (min frequency threshold)
       - Sort codes (theo frequency hoặc alphabetical)
       - Create code_to_idx mapping
       - Save vocab files
       """
       pass
   ```

**Deliverables:**
- Script: `extract_vn_ehr.py`
- Script: `normalize_vn_codes.py`
- Script: `create_vocab_vn.py`
- Output: `vn_patient_codes.pkl` (extracted codes)
- Output: `vn_vocab_info.pkl` (vocabulary)

---

### **THÁNG 2: Knowledge Graph Construction & Data Preparation**

#### **Tuần 5-6: Build Knowledge Graph cho EHR VN**

**Mục tiêu:** Xây dựng KG từ dữ liệu Việt Nam

**Tasks:**

1. **Adapt KG Builder cho VN (5-7 ngày)**
   ```python
   # File: build_kg_vn_ehr.py
   # Adapt từ build_kg_mimic.py
   
   class VN_EHR_KG_Builder:
       def __init__(self, vn_ehr_path, output_dir='embed_vn'):
           # Similar to MIMIC_KG_Builder
           pass
       
       def build_diagnosis_hierarchy(self):
           """
           Build hierarchy cho diagnosis codes VN
           - ICD-10 VN có hierarchy không?
           - Nếu không có, tạo hierarchy dựa trên prefix
           - Hoặc sử dụng ICD-10 quốc tế hierarchy
           """
           pass
       
       def build_drug_hierarchy(self):
           """
           Build hierarchy cho drugs
           - Map drugs -> ATC codes
           - Build ATC hierarchy (nếu có ATC)
           - Hoặc tạo hierarchy dựa trên drug categories
           """
           pass
       
       def build_lab_hierarchy(self):
           """
           Build hierarchy cho lab tests
           - Group by category (nếu có)
           - Hoặc tạo categories dựa trên tên
           """
           pass
       
       def build_cooccurrence_edges(self):
           """
           Build co-occurrence edges từ visits
           - Group codes by visit/patient
           - Create edges: diagnosis <-> drug, etc.
           """
           pass
   ```

2. **Handle Vietnamese-specific Challenges (3-5 ngày)**
   ```python
   # Challenges:
   
   # A. Code Systems khác nhau
   - [ ] ICD-10 VN vs ICD-10 quốc tế
   - [ ] Drug codes: ATC hay mã nội bộ?
   - [ ] Lab codes: LOINC hay tên tiếng Việt?
   
   # B. Missing Hierarchy
   - [ ] Nếu không có hierarchy sẵn, tạo dựa trên:
       * Prefix matching (như ICD-9)
       * Semantic similarity (embeddings)
       * Manual curation (cho top codes)
   
   # C. Text Processing
   - [ ] Normalize tiếng Việt (remove accents?)
   - [ ] Handle abbreviations
   - [ ] Map tên -> codes
   ```

3. **Validate KG (2-3 ngày)**
   ```python
   # File: validate_kg_vn.py
   
   def validate_kg():
       """
       Validate KG quality:
       - Number of nodes/edges
       - Connectivity (isolated nodes?)
       - Edge types distribution
       - Compare với MIMIC-III KG (nếu có thể)
       """
       pass
   ```

**Deliverables:**
- Script: `build_kg_vn_ehr.py`
- Output: `embed_vn/icdatc_graph_*.pkl`
- Output: `embed_vn/icdatc_embed_*.pkl`
- Output: `embed_vn/vocab_info.pkl`
- Document: `KG_VN_SPECIFICS.md` (các điểm khác biệt với MIMIC-III)

---

#### **Tuần 7-8: Create BoW Matrices**

**Mục tiêu:** Tạo BoW matrices từ extracted codes

**Tasks:**

1. **Create BoW Matrix Script (5-7 ngày)**
   ```python
   # File: create_bow_vn.py
   # Adapt từ DATA_PREPARATION_MIMIC.md
   
   def create_bow_matrix_vn(patient_codes, vocab_info):
       """
       Tạo BoW matrix từ patient codes VN
       - Map codes -> vocab indices
       - Create sparse matrix
       - Handle missing codes (không có trong vocab)
       """
       pass
   
   def split_train_test_vn(bow_matrix, train_ratio=0.8):
       """
       Split train/test
       - Random split
       - Hoặc temporal split (nếu có temporal info)
       """
       pass
   ```

2. **Create Metadata File (1-2 ngày)**
   ```python
   # File: create_metadata_vn.py
   
   def create_metadata_vn():
       """
       Tạo metadata.txt cho VN data
       Format:
       ['icd', 'atc']  # hoặc ['icd', 'drug', 'lab']
       [vocab_size_icd, vocab_size_atc]
       [1, 1]  # train embeddings
       ['*', '*']  # use graph embeddings
       """
       pass
   ```

3. **Data Validation (2-3 ngày)**
   ```python
   # File: validate_data_vn.py
   
   def validate_bow_matrices():
       """
       Validate BoW matrices:
       - Shape consistency
       - Vocab order matches KG
       - Non-zero entries reasonable
       - Train/test split correct
       """
       pass
   ```

**Deliverables:**
- Script: `create_bow_vn.py`
- Script: `create_metadata_vn.py`
- Output: `data_vn/bow_train.npy`
- Output: `data_vn/bow_test.npy`
- Output: `data_vn/bow_test_1.npy`
- Output: `data_vn/bow_test_2.npy`
- Output: `data_vn/metadata.txt`

---

### **THÁNG 3: Training & Evaluation**

#### **Tuần 9-10: Initial Training & Debugging**

**Mục tiêu:** Train model lần đầu và fix bugs

**Tasks:**

1. **Setup Training Environment (1-2 ngày)**
   ```bash
   # Update main_getm.py paths:
   - [ ] Update graph_path to VN KG
   - [ ] Update data_path to data_vn/
   - [ ] Update metadata file path
   - [ ] Check GPU availability
   - [ ] Install dependencies
   ```

2. **Small-scale Test Training (3-5 ngày)**
   ```bash
   # Train với subset nhỏ trước:
   python main_getm.py \
       --data_path data_vn/ \
       --meta_file metadata \
       --graph_path embed_vn/icdatc_graph_*.pkl \
       --save_path results_vn_test/ \
       --mode train \
       --num_topics 20 \
       --epochs 10 \
       --batch_size 256
   
   # Debug issues:
   - [ ] Data loading errors
   - [ ] Vocab mismatch errors
   - [ ] Memory issues
   - [ ] Graph format issues
   ```

3. **Fix Issues & Optimize (3-5 ngày)**
   ```python
   # Common issues và fixes:
   
   # Issue 1: Vocab mismatch
   # Fix: Đảm bảo vocab order trong BoW = vocab order trong KG
   
   # Issue 2: Memory issues
   # Fix: 
   - Reduce batch size
   - Use sparse matrices efficiently
   - Reduce graph size (filter rare codes)
   
   # Issue 3: Training instability
   # Fix:
   - Adjust learning rate
   - Add gradient clipping
   - Check data normalization
   ```

**Deliverables:**
- Updated: `main_getm.py` với paths cho VN
- Script: `train_vn_test.py` (wrapper script)
- Results: `results_vn_test/` (test training results)
- Document: `TRAINING_ISSUES.md` (issues và solutions)

---

#### **Tuần 11-12: Full Training & Evaluation**

**Mục tiêu:** Train model đầy đủ và evaluate

**Tasks:**

1. **Full-scale Training (5-7 ngày)**
   ```bash
   # Train với full data:
   python main_getm.py \
       --data_path data_vn/ \
       --meta_file metadata \
       --graph_path embed_vn/icdatc_graph_*.pkl \
       --save_path results_vn/ \
       --mode train \
       --num_topics 50 \
       --epochs 100 \
       --batch_size 512 \
       --lr 0.01 \
       --tq
   
   # Monitor training:
   - [ ] Loss curves
   - [ ] Topic coherence/diversity
   - [ ] Perplexity on test set
   - [ ] Training time
   ```

2. **Evaluation & Analysis (3-5 ngày)**
   ```python
   # File: evaluate_vn.py
   
   def evaluate_model():
       """
       Evaluate trained model:
       1. Document completion (perplexity)
       2. Topic coherence
       3. Topic diversity
       4. Topic quality (coherence * diversity)
       5. Visualize topics
       6. Compare với baseline (nếu có)
       """
       pass
   
   # Visualizations:
   - [ ] Topic-word distributions (top words per topic)
   - [ ] Topic-patient distributions
   - [ ] Graph visualization
   - [ ] Topic coherence/diversity plots
   ```

3. **Interpretation & Documentation (2-3 ngày)**
   ```python
   # File: interpret_topics_vn.py
   
   def interpret_topics():
       """
       Interpret learned topics:
       - Top codes per topic
       - Clinical meaning
       - Compare với medical knowledge
       - Identify interesting patterns
       """
       pass
   ```

**Deliverables:**
- Trained model: `results_vn/getm_*.mdl`
- Evaluation results: `results_vn/evaluation_results.txt`
- Visualizations: `results_vn/visualizations/`
- Document: `EVALUATION_REPORT.md`
- Document: `TOPIC_INTERPRETATION.md`

---

## 🔧 Technical Details

### 1. Handling Vietnamese Text

```python
# Option 1: Keep Vietnamese (recommended)
# - Preserve original text
# - Use Vietnamese NLP tools if needed
# - Map to codes using dictionary

# Option 2: Normalize (if needed)
from unidecode import unidecode

def normalize_vietnamese(text):
    """
    Remove Vietnamese accents
    Use only if necessary for code matching
    """
    return unidecode(text)

# Option 3: Use embeddings
# - Train word embeddings on Vietnamese medical text
# - Use similarity for code matching
```

### 2. Code Mapping Strategies

```python
# A. Diagnosis Codes
# Option 1: Use ICD-10 VN directly (if available)
# Option 2: Map VN -> ICD-10 quốc tế
# Option 3: Create custom hierarchy

# B. Drug Codes
# Option 1: Manual mapping cho top drugs (100-200 drugs)
# Option 2: Use drug dictionary VN -> ATC
# Option 3: Use embedding similarity
# Option 4: Keep as-is nếu có mã nội bộ

# C. Lab Codes
# Option 1: Group by category manually
# Option 2: Use LOINC mapping (nếu có)
# Option 3: Create categories từ tên
```

### 3. Knowledge Graph Construction

```python
# Hierarchy Building:
# - Nếu có hierarchy sẵn: sử dụng trực tiếp
# - Nếu không: tạo dựa trên prefix (như ICD-9)
# - Hoặc: sử dụng semantic similarity

# Co-occurrence Edges:
# - Group by visit/patient
# - Create edges: diagnosis <-> drug, etc.
# - Weight by frequency

# Augmented Edges:
# - Skip connections trong hierarchy
# - Weight decay theo distance
```

### 4. Data Quality Considerations

```python
# Filtering:
# - Remove rare codes (frequency < threshold)
# - Remove patients với quá ít codes
# - Handle missing values

# Normalization:
# - BoW normalization (per code type)
# - Frequency vs binary encoding
```

---

## 📊 Success Metrics

### Phase 1: Data Preparation
- [ ] Successfully extract codes từ EHR VN
- [ ] Create vocabulary với reasonable size (1000-5000 codes)
- [ ] Build KG với >1000 nodes, >5000 edges

### Phase 2: Training
- [ ] Model trains without errors
- [ ] Loss decreases over epochs
- [ ] No memory issues

### Phase 3: Evaluation
- [ ] Perplexity < baseline (nếu có)
- [ ] Topic coherence > 0.3 (reasonable)
- [ ] Topic diversity > 0.5 (reasonable)
- [ ] Topics are interpretable

---

## 🚨 Risk Mitigation

### Risk 1: Data Format Khác Biệt
**Mitigation:**
- Start với sample data nhỏ
- Build flexible extraction scripts
- Test từng component riêng

### Risk 2: Code Mapping Khó Khăn
**Mitigation:**
- Manual mapping cho top codes (80/20 rule)
- Use fuzzy matching cho remaining
- Accept some codes unmapped

### Risk 3: Vietnamese Text Processing
**Mitigation:**
- Keep original text nếu có thể
- Use simple normalization
- Focus on code extraction, not text understanding

### Risk 4: Training Issues
**Mitigation:**
- Test trên MIMIC-III trước (đã có)
- Start với small model
- Debug incrementally

### Risk 5: Timeline Overrun
**Mitigation:**
- Prioritize: KG building > BoW > Training
- Can skip some features (augmented edges, etc.)
- Focus on getting it working first, optimize later

---

## 📝 Checklist Summary

### Month 1
- [ ] Data inventory complete
- [ ] Code mapping strategy defined
- [ ] Data extraction script working
- [ ] Vocabulary created

### Month 2
- [ ] KG built successfully
- [ ] BoW matrices created
- [ ] Metadata file created
- [ ] Data validated

### Month 3
- [ ] Model trains successfully
- [ ] Evaluation completed
- [ ] Results documented
- [ ] Topics interpreted

---

## 🔗 Resources & References

### Code References
- `build_kg_mimic.py`: KG building template
- `main_getm.py`: Training script
- `DATA_PREPARATION_MIMIC.md`: Data prep guide

### External Resources
- ICD-10 VN: [Ministry of Health Vietnam]
- ATC Classification: https://www.whocc.no/atc_ddd_index/
- Vietnamese NLP: [pyvi, underthesea, etc.]

### Documentation
- GAT-ETM Paper: [Nature Scientific Reports]
- MIMIC-III: https://mimic.mit.edu/

---

## 💡 Tips & Best Practices

1. **Start Small**: Test với 100-1000 patients trước
2. **Iterate Quickly**: Fix issues as they come
3. **Document Everything**: Keep notes on decisions
4. **Validate Early**: Check data quality từ đầu
5. **Compare với MIMIC-III**: Use as reference
6. **Focus on Core**: Skip advanced features nếu cần
7. **Get Help**: Consult domain experts khi cần

---

## 📞 Support & Questions

Nếu gặp vấn đề:
1. Check existing documentation
2. Compare với MIMIC-III implementation
3. Debug incrementally
4. Ask for help từ team/community

---

**Good luck với project! 🚀**

