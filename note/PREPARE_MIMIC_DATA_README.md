# Hướng dẫn Chuẩn bị Dữ liệu MIMIC-III cho GAT-ETM

Script này khai thác dữ liệu MIMIC-III public dataset để tạo các file input cần thiết cho việc train model GAT-ETM.

## Tổng quan

Script `prepare_mimic_data.py` sẽ:
1. Load vocabulary từ KG embeddings (đã được build trước đó)
2. Extract ICD và ATC codes từ MIMIC-III admissions
3. Tạo BoW matrices (train/test/test_1/test_2)
4. Tạo metadata.txt file

## Prerequisites

### 1. Build Knowledge Graph trước

Bạn **PHẢI** build KG embeddings trước khi chạy script này:

```bash
cd KG_EMBED
python build_kg_mimic.py \
    --mimic_path ../mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed_augmented \
    --embedding_dim 256 \
    --window 8 \
    --walk_length 20 \
    --num_walks 10 \
    --augmented
```

Điều này sẽ tạo ra:
- `KG_EMBED/embed_augmented/vocab_info.pkl` - Vocabulary info
- `KG_EMBED/embed_augmented/augmented_icdatc_graph_*.pkl` - Graph
- `KG_EMBED/embed_augmented/augmented_icdatc_embed_*.pkl` - Embeddings

### 2. MIMIC-III Data

Đảm bảo bạn có các file CSV sau trong thư mục `mimic-iii-clinical-database-demo-1.4/`:
- `DIAGNOSES_ICD.csv` - ICD diagnosis codes
- `PRESCRIPTIONS.csv` - Prescription data (drug names)

## Usage

### Basic Usage

```bash
python prepare_mimic_data.py \
    --mimic_path mimic-iii-clinical-database-demo-1.4 \
    --kg_embed_dir KG_EMBED/embed_augmented \
    --output_dir data
```

### Full Options

```bash
python prepare_mimic_data.py \
    --mimic_path mimic-iii-clinical-database-demo-1.4 \
    --kg_embed_dir KG_EMBED/embed_augmented \
    --output_dir data \
    --train_ratio 0.8 \
    --random_seed 42 \
    --frequency_bow  # Optional: use frequency instead of binary encoding
```

### Arguments

- `--mimic_path`: Đường dẫn đến thư mục chứa MIMIC-III CSV files (default: `mimic-iii-clinical-database-demo-1.4`)
- `--kg_embed_dir`: Đường dẫn đến thư mục chứa KG embeddings (default: `KG_EMBED/embed_augmented`)
- `--output_dir`: Thư mục output cho BoW files và metadata (default: `data`)
- `--train_ratio`: Tỷ lệ dữ liệu training (default: 0.8)
- `--random_seed`: Random seed cho train/test split (default: 42)
- `--frequency_bow`: Sử dụng frequency encoding thay vì binary (default: binary)

## Output Files

Script sẽ tạo các file sau trong thư mục `output_dir`:

### 1. BoW Matrices

- `bow_train.npy`: Training data (scipy.sparse.csr_matrix)
- `bow_test.npy`: Full test data
- `bow_test_1.npy`: First half của test data (cho document completion)
- `bow_test_2.npy`: Second half của test data (cho document completion)

**Format:**
- Type: `scipy.sparse.csr_matrix` (saved as numpy array with `allow_pickle=True`)
- Shape: `(num_patients, vocab_size[0] + vocab_size[1])`
- Columns: `[ICD codes (0:vocab_size[0]), ATC codes (vocab_size[0]:vocab_size[0]+vocab_size[1])]`
- Values: Binary (0/1) hoặc frequency (nếu dùng `--frequency_bow`)

### 2. Metadata File

`metadata.txt` - File chứa thông tin về code types và vocab sizes:

```
['icd', 'atc']           # Code types
[61210, 82020]          # Vocab sizes
[1, 1]                  # Train embeddings flags (1 = train, 0 = freeze)
['*', '*']              # Embedding file paths ('*' = use KG embeddings)
```

## Workflow Hoàn chỉnh

### Step 1: Build Knowledge Graph

```bash
cd KG_EMBED
python build_kg_mimic.py \
    --mimic_path ../mimic-iii-clinical-database-demo-1.4 \
    --output_dir embed_augmented \
    --augmented
```

### Step 2: Prepare Data

```bash
python prepare_mimic_data.py \
    --mimic_path mimic-iii-clinical-database-demo-1.4 \
    --kg_embed_dir KG_EMBED/embed_augmented \
    --output_dir data
```

### Step 3: Train Model

```bash
python main_getm_mimic.py \
    --data_path data \
    --kg_embed_dir KG_EMBED/embed_augmented \
    --epochs 50 \
    --num_topics 50
```

## Lưu ý quan trọng

1. **Vocab Matching**: Script sẽ tự động match ICD codes từ MIMIC-III với vocab trong KG. Nếu có codes không match được, chúng sẽ bị bỏ qua.

2. **Drug-to-ATC Mapping**: Script sử dụng hash-based mapping (giống như `build_kg_mimic.py`). Để có kết quả tốt hơn trong production, nên sử dụng:
   - RxNorm API → WHO ATC mapping
   - Hoặc external knowledge base như MIA

3. **Memory Usage**: Với dataset lớn, script có thể tốn nhiều memory. Nếu gặp lỗi MemoryError:
   - Giảm số lượng admissions xử lý
   - Hoặc xử lý theo batches

4. **Train/Test Split**: Split được thực hiện ở mức admission (HADM_ID), không phải patient. Mỗi admission được coi như một document độc lập.

## Troubleshooting

### Lỗi: "Vocab info not found"

**Nguyên nhân**: Chưa build KG embeddings.

**Giải pháp**: Chạy `build_kg_mimic.py` trước.

### Lỗi: "Cannot find ICD_CODE or HADM_ID columns"

**Nguyên nhân**: Tên cột trong CSV file không đúng.

**Giải pháp**: Kiểm tra tên cột trong file CSV. Script tự động tìm các tên cột phổ biến nhưng có thể cần điều chỉnh.

### Lỗi: MemoryError

**Nguyên nhân**: Dataset quá lớn.

**Giải pháp**: 
- Xử lý theo chunks nhỏ hơn
- Hoặc giảm số lượng admissions xử lý

### Warning: "No drug name column found"

**Nguyên nhân**: PRESCRIPTIONS.csv không có cột drug name.

**Giải pháp**: Script sẽ skip ATC extraction nhưng vẫn có thể tạo BoW với ICD codes.

## Kiểm tra Output

Sau khi chạy script, kiểm tra:

1. **File sizes**: Các file .npy phải có kích thước hợp lý
2. **Metadata**: Kiểm tra `metadata.txt` có đúng format
3. **Vocab sizes**: Đảm bảo vocab sizes khớp với KG embeddings

```python
import numpy as np
import pickle

# Check BoW matrix
bow_train = np.load('data/bow_train.npy', allow_pickle=True).item()
print(f"Train shape: {bow_train.shape}")
print(f"Non-zero entries: {bow_train.nnz}")

# Check metadata
with open('data/metadata.txt', 'r') as f:
    lines = f.readlines()
    print("Metadata:")
    for line in lines:
        print(line.strip())
```

## Next Steps

Sau khi chuẩn bị xong data, bạn có thể:

1. **Train model**: Chạy `main_getm_mimic.py`
2. **Evaluate**: Sử dụng các metrics như topic coherence, topic diversity
3. **Visualize**: Visualize topics và embeddings

Xem thêm trong `note/TRAINING_GUIDE.md` để biết chi tiết về training.
