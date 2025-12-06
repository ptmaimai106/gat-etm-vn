# Hướng dẫn Build Knowledge Graph từ MIMIC-III

## Tổng quan

Script `build_kg_mimic.py` được thiết kế để xây dựng Knowledge Graph từ dữ liệu MIMIC-III, phục vụ cho việc training model GAT-ETM.

## Yêu cầu

### Python packages
```bash
pip install pandas numpy networkx node2vec tqdm
```

### Dữ liệu MIMIC-III
Đảm bảo thư mục `mimic-iii-clinical-database-demo-1.4` chứa các file CSV cần thiết:
- `D_ICD_DIAGNOSES.csv`
- `DIAGNOSES_ICD.csv`
- `D_ICD_PROCEDURES.csv`
- `PROCEDURES_ICD.csv`
- `D_CPT.csv` (optional)
- `CPTEVENTS.csv` (optional)
- `PRESCRIPTIONS.csv`
- `LABEVENTS.csv`
- `D_LABITEMS.csv`
- `ADMISSIONS.csv`

## Cách sử dụng

### Cơ bản
```bash
python build_kg_mimic.py
```

### Với các tham số tùy chỉnh
```bash
python build_kg_mimic.py \
    --mimic_path mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed \
    --embedding_dim 256 \
    --window 8 \
    --walk_length 20 \
    --num_walks 10 \
    --augmented
```

### Không sử dụng augmented edges
```bash
python build_kg_mimic.py --no_augmented
```

## Các tham số

- `--mimic_path`: Đường dẫn đến thư mục chứa dữ liệu MIMIC-III (mặc định: `mimic-iii-clinical-database-demo-1.4`)
- `--output_dir`: Thư mục lưu output files (mặc định: `embed`)
- `--embedding_dim`: Chiều của embedding vector (mặc định: 256)
- `--window`: Node2Vec window size (mặc định: 8)
- `--walk_length`: Node2Vec walk length (mặc định: 20)
- `--num_walks`: Node2Vec number of walks (mặc định: 10)
- `--augmented` / `--no_augmented`: Có thêm augmented edges hay không (mặc định: có)

## Output files

Script sẽ tạo các file sau trong thư mục `output_dir`:

1. **Graph file**: `augmented_icdatc_graph_{window}_{walk_length}_{num_walks}_{embedding_dim}_renumbered_by_vocab.pkl`
   - NetworkX graph object đã được renumber theo vocab order
   - Format: pickle file

2. **Embeddings file**: `augmented_icdatc_embed_{window}_{walk_length}_{num_walks}_{embedding_dim}_by_vocab.pkl`
   - Numpy array chứa node embeddings
   - Shape: (num_nodes, embedding_dim)
   - Format: pickle file

3. **Vocab mapping**: `graphnode_vocab.pkl`
   - Dictionary mapping node index -> code string
   - Format: pickle file

4. **Vocab info**: `vocab_info.pkl`
   - Dictionary chứa vocabularies cho từng code type
   - Format: pickle file

## Cấu trúc Knowledge Graph

### Node types
- **ICD9**: Diagnosis codes từ DIAGNOSES_ICD
- **CPT**: Procedure codes từ PROCEDURES_ICD/CPTEVENTS
- **ATC**: Drug codes (mapped từ PRESCRIPTIONS)
- **LAB**: Lab item codes từ LABEVENTS

### Edge types
1. **Hierarchical edges**: 
   - ICD9 parent-child relationships
   - CPT hierarchy
   - ATC hierarchy
   - Lab category hierarchy

2. **Co-occurrence edges**:
   - ICD9 ↔ ATC (cùng admission)
   - ICD9 ↔ CPT (cùng admission)
   - ICD9 ↔ Lab (cùng admission)
   - ICD9 ↔ ICD9 (cùng admission)

3. **Augmented edges** (nếu enabled):
   - Skip connections trong hierarchy trees

## Lưu ý quan trọng

### ATC Mapping
Script hiện tại sử dụng **placeholder ATC codes** dựa trên hash của drug names. Để có ATC codes chính xác, bạn cần:

1. Sử dụng RxNorm API để map drug names → RxNorm codes
2. Sử dụng WHO ATC index để map RxNorm → ATC codes
3. Hoặc sử dụng file mapping có sẵn

Có thể cập nhật function `extract_drugs_and_map_to_atc()` để sử dụng mapping thực tế.

### ICD9 Format
Script tự động normalize ICD9 codes (remove dots). Đảm bảo format nhất quán với vocab trong training data.

### Performance
- Với dataset lớn, quá trình build có thể mất vài phút đến vài giờ
- Co-occurrence edge building là bước tốn thời gian nhất
- Có thể giảm `min_cooccurrence` threshold để tăng số edges

## Sử dụng output trong training

Sau khi build xong, update `main_getm.py`:

```python
args.graph_path = 'embed/augmented_icdatc_graph_8_20_10_256_renumbered_by_vocab.pkl'
args.graph_embed = pickle.load(open('embed/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl', 'rb'))
```

## Troubleshooting

### Lỗi: "node2vec not installed"
```bash
pip install node2vec
```
Nếu không cài được, script sẽ tự động dùng random embeddings.

### Lỗi: "File not found"
Kiểm tra đường dẫn `--mimic_path` có đúng không.

### Memory error
- Giảm số walks: `--num_walks 5`
- Giảm walk length: `--walk_length 10`
- Tăng `min_cooccurrence` threshold trong code

## Ví dụ output

```
================================================================================
Building Knowledge Graph from MIMIC-III
================================================================================
Loading MIMIC-III tables...
Tables loaded successfully!
Building ICD9 hierarchy...
ICD9 hierarchy: 1234 nodes, 2345 edges
ICD9 vocabulary size: 567
Building CPT hierarchy...
...
Graph after hierarchies: 5000 nodes, 8000 edges
Building co-occurrence edges from admissions...
  Processing admissions for co-occurrence...
Co-occurrence edges: 15000 edges added
Graph after co-occurrence: 5000 nodes, 23000 edges
Augmenting graph with skip connections...
Augmented edges: 5000 edges added
Generating Node2Vec embeddings...
  Training Node2Vec model...
Embeddings generated: shape (5000, 256)
Renumbering nodes by vocabulary order...
Saving outputs...
================================================================================
Knowledge Graph built successfully!
================================================================================
Final graph: 5000 nodes, 28000 edges
Embeddings shape: (5000, 256)
```

