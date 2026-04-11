# Hướng dẫn Train GAT-ETM trên Máy Mới

## Tổng quan

Toàn bộ dữ liệu đã được xử lý và commit vào git (phases 1–5 hoàn thành).
Khi pull code về máy mới, chỉ cần cài môi trường và chạy training — **không cần chạy lại pipeline xử lý dữ liệu**.

---

## Trạng thái hiện tại

| Phase | Mô tả | Trạng thái |
|-------|-------|-----------|
| 1 | Thu thập ICD-10 & ATC Hierarchy | ✅ Hoàn thành |
| 2 | Parse Drug Names | ✅ Hoàn thành |
| 3 | Drug → ATC Mapping | ✅ Hoàn thành |
| 4 | Xây dựng Knowledge Graph + Node2Vec | ✅ Hoàn thành |
| 5 | Tạo BoW matrices | ✅ Hoàn thành |
| 6 | Training | ✅ Đã có model K=20 (`results_vn/model_K20.pt`) |
| 7 | Evaluation | ⏳ Chưa làm |
| 8 | Visualization & Analysis | ⏳ Chưa làm |

---

## Các file quan trọng trong repo

```
data_vn/
├── bow_train.npy          # 49,184 training samples
├── bow_test.npy           # 21,080 test samples
├── bow_test_1.npy         # Test split 1 (document completion)
├── bow_test_2.npy         # Test split 2 (document completion)
├── metadata.txt           # 1493 ICD + 377 ATC codes
└── vocab_info.pkl         # Vocabulary metadata

embed_vn/
├── augmented_icdatc_graph_256_renumbered_by_vocab.pkl   # Knowledge graph
├── augmented_icdatc_embed_8_20_10_256_by_vocab.pkl      # Node2Vec embeddings (1870x256)
├── icd_embeddings.npy     # ICD embeddings (1493x256)
└── atc_embeddings.npy     # ATC embeddings (377x256)

results_vn/
└── model_K20.pt           # Model đã train sẵn với K=20 topics
```

---

## Bước 1 — Clone code

```bash
git clone <repo_url>
cd GAT-ETM
```

---

## Bước 2 — Cài môi trường Python

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
pip install simple-icd-10 fuzzywuzzy python-Levenshtein
```

Nếu có GPU (khuyến nghị), cài PyTorch với CUDA:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Bước 3 — Train model

> ⚠️ **Quan trọng**: `train_vn.py` mặc định trỏ `--graph_path` và `--embed_path` vào `embed/` (data MIMIC-III).
> Phải **ghi đè thủ công** sang `embed_vn/` để dùng đúng data Việt Nam.

### 3.1 Train cơ bản — topic model

```bash
python train_vn.py \
  --num_topics 50 \
  --epochs 100 \
  --graph_path embed_vn/augmented_icdatc_graph_256_renumbered_by_vocab.pkl \
  --embed_path embed_vn/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl \
  --tq
```

### 3.2 Train với drug imputation

```bash
python train_vn.py \
  --num_topics 50 \
  --epochs 50 \
  --graph_path embed_vn/augmented_icdatc_graph_256_renumbered_by_vocab.pkl \
  --embed_path embed_vn/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl \
  --drug_imputation --loss wkl --gamma 1.0
```

### 3.3 Các tham số hay dùng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--num_topics` | 50 | Số topics K |
| `--epochs` | 50 | Số epochs |
| `--batch_size` | 512 | Batch size |
| `--lr` | 0.01 | Learning rate |
| `--tq` | False | Tính Topic Quality metrics |
| `--gpu_device` | `0` | GPU ID (bỏ qua nếu dùng CPU) |
| `--save_path` | `results_vn/` | Thư mục lưu kết quả |

Model checkpoint lưu tại: `results_vn/model_K{num_topics}.pt`

---

## Bước 4 — Evaluate model

### 4.1 Eval model vừa train

```bash
python train_vn.py \
  --mode eval \
  --num_topics 50 \
  --load_from results_vn/model_K50.pt \
  --graph_path embed_vn/augmented_icdatc_graph_256_renumbered_by_vocab.pkl \
  --embed_path embed_vn/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl \
  --tq
```

### 4.2 Eval model K=20 có sẵn trong repo (baseline)

```bash
python train_vn.py \
  --mode eval \
  --num_topics 20 \
  --load_from results_vn/model_K20.pt \
  --graph_path embed_vn/augmented_icdatc_graph_256_renumbered_by_vocab.pkl \
  --embed_path embed_vn/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl \
  --tq
```

### 4.3 Metrics đầu ra

| Metric | Mô tả | Paper gốc (GAT-ETM) |
|--------|-------|---------------------|
| NLL | Negative Log-Likelihood (thấp hơn = tốt hơn) | 172.69 |
| TC | Topic Coherence (top-3 codes) | — |
| TD | Topic Diversity (top-3 codes) | — |
| TQ = TC × TD | Topic Quality | 0.192 |

---

## Lưu ý quan trọng

1. **Đường dẫn embed**: Luôn chỉ rõ `--graph_path` và `--embed_path` vào `embed_vn/`, không dùng default.
2. **Vocabulary size**: 1,493 ICD + 377 ATC = 1,870 codes (nhỏ hơn paper gốc: 5,107 + 1,057).
3. **Data size**: 49,184 train / 21,080 test (~38K patients từ bệnh viện VN).
4. **Model đã có sẵn**: `results_vn/model_K20.pt` có thể dùng làm baseline để so sánh.
