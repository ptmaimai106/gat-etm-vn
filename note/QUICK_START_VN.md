# Quick Start Guide: GAT-ETM cho EHR Việt Nam

Hướng dẫn nhanh để bắt đầu apply GAT-ETM trên dữ liệu hồ sơ bệnh án điện tử Việt Nam.

---

## 🚀 Bước 1: Chuẩn bị Dữ liệu (Tuần 1-2)

### 1.1. Khảo sát Dữ liệu

```python
# Checklist cần trả lời:
- [ ] Dữ liệu ở format nào? (CSV, SQL, Excel, JSON?)
- [ ] Có những bảng nào? (diagnoses, prescriptions, labs, procedures?)
- [ ] Codes ở format nào? (ICD-10 VN? Tên tiếng Việt? Mã nội bộ?)
- [ ] Có bao nhiêu patients? Visits? Records?
- [ ] Có temporal information không? (dates, visits?)
```

### 1.2. Extract Codes

```bash
# Sử dụng template script
cd scripts/
python extract_vn_ehr_template.py

# Customize:
# 1. Update load_data() - load dữ liệu của bạn
# 2. Update extract_*_codes() - extract logic
# 3. Update normalize_vietnamese_text() - nếu cần
```

**Output:** `data_vn/vn_patient_codes.pkl`

---

## 🏗️ Bước 2: Build Knowledge Graph (Tuần 3-4)

### 2.1. Adapt KG Builder

```python
# Copy và modify build_kg_mimic.py
cp KG_EMBED/build_kg_mimic.py KG_EMBED/build_kg_vn_ehr.py

# Thay đổi:
# 1. load_mimic_tables() -> load_vn_ehr_tables()
# 2. build_icd9_hierarchy() -> build_diagnosis_hierarchy_vn()
# 3. extract_drugs_and_map_to_atc() -> extract_drugs_vn()
# 4. build_cooccurrence_edges() -> adapt cho VN data structure
```

### 2.2. Build KG

```bash
cd KG_EMBED/
python build_kg_vn_ehr.py \
    --vn_ehr_path ../data_vn/ \
    --output_dir embed_vn/ \
    --embedding_dim 256 \
    --augmented
```

**Output:**
- `embed_vn/icdatc_graph_*.pkl`
- `embed_vn/icdatc_embed_*.pkl`
- `embed_vn/vocab_info.pkl`

---

## 📊 Bước 3: Tạo BoW Matrices (Tuần 5-6)

### 3.1. Create BoW

```python
# Sử dụng script từ DATA_PREPARATION_MIMIC.md
# Adapt cho VN data

python create_bow_vn.py \
    --patient_codes data_vn/vn_patient_codes.pkl \
    --vocab_info embed_vn/vocab_info.pkl \
    --output_dir data_vn/
```

**Output:**
- `data_vn/bow_train.npy`
- `data_vn/bow_test.npy`
- `data_vn/bow_test_1.npy`
- `data_vn/bow_test_2.npy`

### 3.2. Create Metadata

```python
# Tạo metadata.txt
# Format:
['icd', 'atc']  # code types
[1000, 500]     # vocab sizes
[1, 1]          # train embeddings
['*', '*']      # use graph embeddings
```

**Output:** `data_vn/metadata.txt`

---

## 🎯 Bước 4: Train Model (Tuần 7-8)

### 4.1. Update main_getm.py

```python
# Update paths trong main_getm.py:
args.graph_path = 'embed_vn/augmented_icdatc_graph_*.pkl'
args.graph_embed = pickle.load(open('embed_vn/augmented_icdatc_embed_*.pkl', 'rb'))
args.data_path = 'data_vn/'
args.meta_file = 'metadata'
```

### 4.2. Test Training (Small Scale)

```bash
python main_getm.py \
    --data_path data_vn/ \
    --meta_file metadata \
    --graph_path embed_vn/augmented_icdatc_graph_*.pkl \
    --save_path results_vn_test/ \
    --mode train \
    --num_topics 20 \
    --epochs 10 \
    --batch_size 256
```

### 4.3. Full Training

```bash
python main_getm.py \
    --data_path data_vn/ \
    --meta_file metadata \
    --graph_path embed_vn/augmented_icdatc_graph_*.pkl \
    --save_path results_vn/ \
    --mode train \
    --num_topics 50 \
    --epochs 100 \
    --batch_size 512 \
    --lr 0.01 \
    --tq
```

---

## ✅ Checklist Nhanh

### Phase 1: Data (Tuần 1-2)
- [ ] Khảo sát dữ liệu xong
- [ ] Extract codes thành công
- [ ] Có `vn_patient_codes.pkl`

### Phase 2: KG (Tuần 3-4)
- [ ] Build KG thành công
- [ ] Có graph và embeddings files
- [ ] Vocab info đã tạo

### Phase 3: BoW (Tuần 5-6)
- [ ] BoW matrices đã tạo
- [ ] Metadata file đã tạo
- [ ] Data validated

### Phase 4: Training (Tuần 7-8)
- [ ] Test training thành công
- [ ] Full training hoàn thành
- [ ] Evaluation done

---

## 🆘 Troubleshooting Nhanh

### Lỗi: "Vocab mismatch"
**Fix:** Đảm bảo vocab order trong BoW = vocab order trong KG

### Lỗi: "Memory error"
**Fix:** Giảm batch_size, filter rare codes

### Lỗi: "Code not found"
**Fix:** Check code normalization, mapping

### Lỗi: "Training unstable"
**Fix:** Giảm learning rate, thêm gradient clipping

---

## 📚 Tài liệu Chi tiết

- **Pipeline đầy đủ:** `PIPELINE_VN_EHR.md`
- **Data preparation:** `DATA_PREPARATION_MIMIC.md`
- **KG building:** `KG_EMBED/README.md`
- **Template script:** `scripts/extract_vn_ehr_template.py`

---

## 💡 Tips

1. **Start small**: Test với 100-1000 patients trước
2. **Iterate quickly**: Fix issues ngay khi phát hiện
3. **Document decisions**: Ghi lại mọi quyết định
4. **Compare với MIMIC-III**: Dùng làm reference
5. **Get help**: Hỏi khi cần

---

**Good luck! 🚀**

