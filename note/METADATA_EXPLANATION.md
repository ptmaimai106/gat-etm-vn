# Giải thích Chi tiết về Metadata File trong GAT-ETM

## Tổng quan

Metadata file (`metadata.txt`) là file cấu hình quan trọng nhất trong quá trình training GAT-ETM. File này chứa thông tin về cấu trúc dữ liệu, vocabulary sizes, và cách xử lý embeddings cho từng loại medical code.

## Vị trí và Format

- **Vị trí**: `data/metadata.txt` (trong thư mục `data_path`)
- **Format**: Text file với 4 dòng, mỗi dòng là một Python list được convert sang string
- **Encoding**: UTF-8

## Cấu trúc Metadata File

### Format tổng quát:
```
['code_type_1', 'code_type_2', ...]
[vocab_size_1, vocab_size_2, ...]
[train_flag_1, train_flag_2, ...]
['embedding_file_1', 'embedding_file_2', ...]
```

### Ví dụ thực tế cho MIMIC-III:
```
['icd', 'atc']
[61210, 82020]
[1, 1]
['*', '*']
```

---

## Chi tiết từng dòng

### **Dòng 1: `code_types` - Danh sách các loại Medical Codes**

**Ý nghĩa**: Định nghĩa các loại medical codes được sử dụng trong dataset.

**Format**: Python list string
```python
['icd', 'atc']  # Ví dụ với 2 loại codes
```

**Giải thích**:
- Mỗi phần tử là tên của một loại medical code
- Tên này phải **khớp chính xác** với tên được sử dụng trong:
  - Knowledge Graph (KG) vocab info (`vocab_info.pkl`)
  - BoW matrices (thứ tự columns)
  - Model code (khi truy cập `beta[code_type]`)

**Các loại codes phổ biến trong MIMIC-III**:
- `'icd'` hoặc `'icd9'`: ICD-9 diagnosis codes (bệnh lý)
- `'atc'`: ATC drug codes (thuốc)
- `'cpt'`: CPT procedure codes (thủ thuật)
- `'lab'`: Laboratory codes (xét nghiệm)

**Lưu ý quan trọng**:
1. **Thứ tự quan trọng**: Thứ tự trong list này quyết định thứ tự trong BoW matrix
   - Ví dụ: `['icd', 'atc']` nghĩa là trong BoW matrix:
     - Columns 0 → vocab_size[0]-1: ICD codes
     - Columns vocab_size[0] → vocab_size[0]+vocab_size[1]-1: ATC codes

2. **Phải khớp với KG**: Tên code types phải khớp với keys trong `vocab_info.pkl` từ KG builder

3. **Không được thay đổi sau khi train**: Nếu thay đổi, model sẽ không load được

---

### **Dòng 2: `vocab_sizes` - Kích thước Vocabulary cho mỗi Code Type**

**Ý nghĩa**: Số lượng unique codes trong vocabulary của mỗi code type.

**Format**: Python list of integers (as string)
```python
[61210, 82020]  # ICD có 61210 codes, ATC có 82020 codes
```

**Giải thích**:
- Mỗi số đại diện cho số lượng unique codes của code type tương ứng
- `vocab_sizes[0]` = số ICD codes
- `vocab_sizes[1]` = số ATC codes
- ...

**Cách xác định vocab_size**:

1. **Từ KG Builder** (khuyến nghị):
   ```python
   # Load từ vocab_info.pkl
   with open('KG_EMBED/embed_augmented/vocab_info.pkl', 'rb') as f:
       vocab_info = pickle.load(f)
   
   vocab_sizes = [len(vocab_info['icd']), len(vocab_info['atc'])]
   # Output: [61210, 82020]
   ```

2. **Từ BoW Matrix**:
   ```python
   # Vocab size = số columns cho mỗi code type trong BoW matrix
   # Phải khớp với vocab_cum
   vocab_cum = [0, 61210, 143230]  # Cumulative sum
   vocab_sizes = [61210, 82020]  # Differences
   ```

3. **Từ dữ liệu thô MIMIC-III**:
   ```python
   # Đếm số unique codes trong DIAGNOSES_ICD.csv và PRESCRIPTIONS.csv
   unique_icd = diagnoses_df['ICD9_CODE'].nunique()
   unique_atc = prescriptions_df['ATC_CODE'].nunique()
   ```

**Lưu ý quan trọng**:
1. **Phải khớp chính xác** với:
   - Vocab sizes trong `vocab_info.pkl` từ KG builder
   - Số columns trong BoW matrices cho mỗi code type
   - Vocab sizes được sử dụng khi build KG

2. **Tính toán `vocab_cum`**:
   ```python
   vocab_cum = np.cumsum([0] + vocab_sizes)
   # Ví dụ: [0, 61210, 143230]
   # ICD: columns 0-61209
   # ATC: columns 61210-143229
   ```

3. **Không được nhỏ hơn số codes thực tế**: Nếu nhỏ hơn, một số codes sẽ bị mất

---

### **Dòng 3: `train_embeddings` - Flag quyết định có Train Embeddings hay không**

**Ý nghĩa**: Cho biết model có được phép train (fine-tune) embeddings cho mỗi code type hay không.

**Format**: Python list of integers (0 or 1, as string)
```python
[1, 1]  # Train embeddings cho cả ICD và ATC
[0, 1]  # Fix ICD embeddings, train ATC embeddings
```

**Giải thích**:
- `1` = **Train embeddings**: Model sẽ cập nhật embeddings trong quá trình training
- `0` = **Fix embeddings**: Giữ nguyên embeddings ban đầu, không train

**Khi nào dùng `1` (train)**:
- Khi muốn model học embeddings từ dữ liệu EHR
- Khi embeddings ban đầu chỉ là initialization
- Khi sử dụng graph embeddings làm starting point và muốn fine-tune

**Khi nào dùng `0` (fix)**:
- Khi embeddings đã được pre-train tốt và muốn giữ nguyên
- Khi muốn giảm số parameters cần train
- Khi embeddings từ external source (như Word2Vec, GloVe) và muốn giữ nguyên

**Lưu ý**:
- Trong `main_getm_mimic.py`, mặc định là `[1, 1, ...]` (train tất cả)
- Nếu dùng `'*'` trong dòng 4 (embedding files), thường nên dùng `1` để train từ graph embeddings

---

### **Dòng 4: `embedding` - Đường dẫn đến Embedding Files**

**Ý nghĩa**: Đường dẫn đến file embeddings cho mỗi code type, hoặc `'*'` nếu sử dụng graph embeddings.

**Format**: Python list of strings
```python
['*', '*']  # Sử dụng graph embeddings cho cả ICD và ATC
['data/icd_embeddings.npy', 'data/atc_embeddings.npy']  # Sử dụng file riêng
```

**Giải thích**:

1. **`'*'` - Sử dụng Graph Embeddings**:
   - Model sẽ sử dụng embeddings từ `graph_embed` (đã load từ KG)
   - Embeddings được lấy từ phần tương ứng trong graph embeddings:
     ```python
     # Ví dụ với vocab_cum = [0, 61210, 143230]
     icd_embeddings = graph_embed[0:61210]      # ICD embeddings
     atc_embeddings = graph_embed[61210:143230]  # ATC embeddings
     ```
   - **Khuyến nghị cho MIMIC-III**: Dùng `'*'` vì đã có graph embeddings từ KG builder

2. **Đường dẫn file - Sử dụng Pre-trained Embeddings**:
   - File phải là NumPy array (`.npy`)
   - Shape: `(vocab_size[i], embedding_dim)`
   - `embedding_dim` phải khớp với `rho_size` (thường là 256)
   - Ví dụ:
     ```python
     # data/icd_embeddings.npy
     # Shape: (61210, 256)
     ```

**Workflow trong code**:
```python
embeddings = {}
for i, c in enumerate(args.code_types):
    if args.embedding[i] == '*': 
        embeddings[c] = None  # Sẽ dùng graph embeddings
    else:
        embed_file = os.path.join(args.data_path, args.embedding[i])
        embeddings[c] = torch.from_numpy(np.load(embed_file)).to(device)
```

**Lưu ý**:
- Nếu dùng `'*'`, đảm bảo `args.graph_embed` đã được load
- Nếu dùng file riêng, đảm bảo file tồn tại và có shape đúng
- Đường dẫn có thể là relative (từ `data_path`) hoặc absolute

---

## Mối quan hệ giữa các dòng

### 1. Số lượng phần tử phải bằng nhau:
```python
len(code_types) == len(vocab_sizes) == len(train_embeddings) == len(embedding)
```

### 2. Thứ tự phải khớp:
- `code_types[0]` ↔ `vocab_sizes[0]` ↔ `train_embeddings[0]` ↔ `embedding[0]`
- Tất cả đều cho cùng một code type

### 3. Vocab sizes phải khớp với BoW matrix:
```python
total_vocab_size = sum(vocab_sizes)
# Phải bằng số columns trong BoW matrices
assert bow_train.shape[1] == total_vocab_size
```

### 4. Vocab sizes phải khớp với KG:
```python
# Từ vocab_info.pkl
vocab_info = {'icd': [...], 'atc': [...]}
assert len(vocab_info['icd']) == vocab_sizes[0]
assert len(vocab_info['atc']) == vocab_sizes[1]
```

---

## Ví dụ đầy đủ cho MIMIC-III

### Scenario 1: Sử dụng Graph Embeddings (Khuyến nghị)

**metadata.txt**:
```
['icd', 'atc']
[61210, 82020]
[1, 1]
['*', '*']
```

**Giải thích**:
- 2 loại codes: ICD và ATC
- ICD có 61210 unique codes, ATC có 82020 unique codes
- Train embeddings cho cả hai (fine-tune từ graph embeddings)
- Sử dụng graph embeddings từ KG (không có file riêng)

**BoW Matrix structure**:
```
Total columns: 61210 + 82020 = 143230
- Columns 0-61209: ICD codes
- Columns 61210-143229: ATC codes
```

---

### Scenario 2: Sử dụng Pre-trained Embeddings

**metadata.txt**:
```
['icd', 'atc']
[61210, 82020]
[0, 1]
['data/icd_pretrained.npy', '*']
```

**Giải thích**:
- ICD: Sử dụng pre-trained embeddings từ file, **không train** (fix)
- ATC: Sử dụng graph embeddings, **có train** (fine-tune)

---

### Scenario 3: Nhiều Code Types (ICD, CPT, ATC, Lab)

**metadata.txt**:
```
['icd', 'cpt', 'atc', 'lab']
[61210, 5000, 82020, 3000]
[1, 1, 1, 1]
['*', '*', '*', '*']
```

**BoW Matrix structure**:
```
Total columns: 61210 + 5000 + 82020 + 3000 = 150230
vocab_cum = [0, 61210, 66210, 148230, 151230]
- Columns 0-61209: ICD
- Columns 61210-66209: CPT
- Columns 66210-148229: ATC
- Columns 148230-151229: Lab
```

---

## So sánh với Script Gốc (main_getm.py)

### Khác biệt chính:

1. **main_getm.py** (Private dataset):
   - Metadata file **bắt buộc phải có**
   - Load trực tiếp từ file, không có fallback
   - Format có thể khác (space-separated hoặc list string)

2. **main_getm_mimic.py** (MIMIC-III):
   - Metadata file **tự động tạo** nếu không có
   - Load từ `vocab_info.pkl` làm fallback
   - Format: Python list string (dùng `eval()`)

### Code comparison:

**main_getm.py**:
```python
# Bắt buộc phải có metadata
metadata = np.loadtxt(os.path.join(args.data_path, args.meta_file+'.txt'), dtype=str)
args.code_types, vocab_size, train_embeddings, args.embedding = metadata
```

**main_getm_mimic.py**:
```python
# Tự động tạo nếu không có
if os.path.exists(metadata_path):
    metadata = np.loadtxt(metadata_path, dtype=str)
    # Parse metadata...
else:
    # Tạo metadata từ vocab_info.pkl
    args.train_embeddings = [1] * len(args.code_types)
    args.embedding = ['*'] * len(args.code_types)
    # Tự động tạo file...
```

---

## Cách tạo Metadata File cho MIMIC-III

### Script Python để tạo metadata:

```python
import pickle
import os

def create_metadata_for_mimic(kg_embed_dir, data_path, output_file='metadata.txt'):
    """
    Tạo metadata.txt từ vocab_info.pkl
    
    Args:
        kg_embed_dir: Thư mục chứa KG embeddings (ví dụ: 'KG_EMBED/embed_augmented')
        data_path: Thư mục data (ví dụ: 'data/')
        output_file: Tên file metadata (mặc định: 'metadata.txt')
    """
    # Load vocab info từ KG
    vocab_info_path = os.path.join(kg_embed_dir, 'vocab_info.pkl')
    with open(vocab_info_path, 'rb') as f:
        vocab_info = pickle.load(f)
    
    # Xác định code types (ví dụ: chỉ lấy 'icd' và 'atc')
    code_types = ['icd', 'atc']  # Hoặc lấy tất cả: list(vocab_info.keys())
    code_types = [ct for ct in code_types if ct in vocab_info and len(vocab_info[ct]) > 0]
    
    # Tính vocab sizes
    vocab_sizes = [len(vocab_info[ct]) for ct in code_types]
    
    # Mặc định: train embeddings, dùng graph embeddings
    train_embeddings = [1] * len(code_types)
    embedding_files = ['*'] * len(code_types)
    
    # Tạo metadata file
    metadata_path = os.path.join(data_path, output_file)
    os.makedirs(data_path, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        f.write(str(code_types) + '\n')
        f.write(str(vocab_sizes) + '\n')
        f.write(str(train_embeddings) + '\n')
        f.write(str(embedding_files) + '\n')
    
    print(f"Metadata saved to {metadata_path}")
    print(f"Code types: {code_types}")
    print(f"Vocab sizes: {vocab_sizes}")
    
    return metadata_path

# Sử dụng
if __name__ == '__main__':
    create_metadata_for_mimic(
        kg_embed_dir='KG_EMBED/embed_augmented',
        data_path='data/',
        output_file='metadata.txt'
    )
```

---

## Kiểm tra Metadata File

### Script validation:

```python
import numpy as np
import pickle

def validate_metadata(metadata_path, kg_embed_dir, bow_train_path):
    """Kiểm tra metadata file có hợp lệ không"""
    
    # 1. Load metadata
    metadata = np.loadtxt(metadata_path, dtype=str)
    code_types = eval(metadata[0])
    vocab_sizes = eval(metadata[1])
    train_embeddings = eval(metadata[2])
    embedding_files = eval(metadata[3])
    
    # 2. Kiểm tra số lượng phần tử
    assert len(code_types) == len(vocab_sizes) == len(train_embeddings) == len(embedding_files), \
        "Số lượng phần tử không khớp!"
    
    # 3. Kiểm tra với vocab_info.pkl
    vocab_info_path = os.path.join(kg_embed_dir, 'vocab_info.pkl')
    with open(vocab_info_path, 'rb') as f:
        vocab_info = pickle.load(f)
    
    for i, ct in enumerate(code_types):
        assert ct in vocab_info, f"Code type '{ct}' không có trong vocab_info!"
        assert len(vocab_info[ct]) == vocab_sizes[i], \
            f"Vocab size cho '{ct}' không khớp: {len(vocab_info[ct])} vs {vocab_sizes[i]}"
    
    # 4. Kiểm tra với BoW matrix
    bow_train = np.load(bow_train_path, allow_pickle=True)
    total_vocab = sum(vocab_sizes)
    assert bow_train.shape[1] == total_vocab, \
        f"BoW matrix columns ({bow_train.shape[1]}) không khớp với total vocab ({total_vocab})"
    
    print("✓ Metadata file hợp lệ!")
    return True
```

---

## Tóm tắt

Metadata file chứa 4 thông tin quan trọng:

1. **`code_types`**: Các loại medical codes (ICD, ATC, ...)
2. **`vocab_sizes`**: Số lượng codes trong mỗi vocabulary
3. **`train_embeddings`**: Có train embeddings hay không (0/1)
4. **`embedding`**: Đường dẫn embedding files hoặc `'*'` để dùng graph embeddings

**Quy tắc vàng**:
- Số phần tử trong mỗi list phải bằng nhau
- Vocab sizes phải khớp với KG và BoW matrices
- Thứ tự code types quyết định thứ tự trong BoW matrix
- Với MIMIC-III, khuyến nghị dùng `['*', '*']` và `[1, 1]` để tận dụng graph embeddings

