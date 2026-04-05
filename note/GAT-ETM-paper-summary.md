# Tóm tắt Paper: GAT-ETM (Graph ATtention-Embedded Topic Model)

**Paper**: "Modeling electronic health record data using an end-to-end knowledge-graph-informed topic model"
**Tác giả**: Yuesong Zou, Ahmad Pesaranghader, Ziyang Song, Aman Verma, David L. Buckeridge & Yue Li
**Nguồn**: Scientific Reports (2022) 12:17868
**Link**: https://doi.org/10.1038/s41598-022-22956-w
**Code**: https://github.com/li-lab-mcgill/GAT-ETM

---

## 1. Vấn đề nghiên cứu

- Dữ liệu EHR (Electronic Health Records) tăng trưởng nhanh, mở ra cơ hội hiểu bệnh tật một cách hệ thống
- **Thách thức**: Dữ liệu EHR thưa (sparse) và nhiễu (noisy), đặc biệt với các mã bệnh/thuốc hiếm gặp
- Các phương pháp trước đó (MixEHR, ETM, GETM) chưa tận dụng tốt knowledge graph hoặc không học end-to-end

---

## 2. Đóng góp chính của GAT-ETM

1. **End-to-end framework**: Học đồng thời embedding của mã y tế từ knowledge graph và topic embedding từ dữ liệu EHR
2. **Linear decoder**: Giúp trích xuất topic có ý nghĩa và diễn giải được
3. **Graph augmentation**: Kết nối node với tất cả ancestor nodes + liên kết ICD-ATC qua quan hệ bệnh-thuốc

---

## 3. Kiến trúc mô hình

### 3.1 Input
- **ICD-9 codes**: Mã chẩn đoán bệnh (5107 mã)
- **ATC codes**: Mã thuốc/hoạt chất (1057 mã)
- **Knowledge Graph**: Kết hợp ICD hierarchy + ATC hierarchy + ICD-ATC relations

### 3.2 Các thành phần chính

| Thành phần | Chức năng |
|------------|-----------|
| **VAE Encoder** | Suy luận topic mixture θ_p của bệnh nhân từ bag-of-words |
| **GAT (Graph Attention Network)** | Học code embedding ρ từ knowledge graph, dùng multi-head attention |
| **Linear Decoder** | Tái tạo EHR data từ θ_p và β (topic distribution) |

### 3.3 Ký hiệu chính

| Ký hiệu | Mô tả |
|---------|-------|
| D | Số lượng bệnh nhân |
| K | Số lượng topics |
| V_icd, V_atc | Kích thước từ vựng ICD và ATC |
| v_p | Vector tần suất mã của bệnh nhân p |
| θ_p | Topic mixture của bệnh nhân p (tổng = 1) |
| ρ^(icd), ρ^(atc) | KG-informed embedding của ICD/ATC codes |
| α^(icd), α^(atc) | Embedding của topics cho ICD/ATC |
| β_k | Phân phối topic thứ k |

### 3.4 Quá trình sinh (Generative Process)

```
Với mỗi bệnh nhân p:
1. Lấy mẫu topic mixture:
   δ_p ~ N(0, I)
   θ_p = softmax(δ_p)  # Logistic-Normal distribution

2. Với mỗi mã EHR c_pn (t ∈ {icd, atc}):
   c_pn ~ Categorical(β^(t) × θ_p)

Trong đó:
   β_k^(t) = softmax(ρ^(t)^T × α_k)
   # inner product của code embedding và topic embedding
```

### 3.5 Learning Procedure

1. **Khởi tạo code embedding**: Dùng Node2Vec trên knowledge graph (256 chiều)
2. **GAT layers**: 3 layers, 4 heads attention
3. **Encoder**: 2 input layers (ICD + ATC) → element-wise addition → NN_μ, NN_σ
4. **Optimization**: Adam optimizer, lr=0.01, batch size=512
5. **Loss**: Maximize ELBO (Evidence Lower Bound)

---

## 4. Xây dựng Knowledge Graph

### 4.1 Nguồn dữ liệu
- ICD hierarchy: https://icdlist.com/icd-9/index
- ATC hierarchy: https://www.whocc.no/atc_ddd_index/
- ICD-ATC relations: http://hulab.rxnfinder.org/mia/

### 4.2 Graph Augmentation Strategy
- Kết nối mỗi node với TẤT CẢ ancestor nodes (không chỉ parent trực tiếp)
- Merge ICD graph và ATC graph thông qua disease-drug links
- Mục đích: Tăng information flow trong graph thưa (tree structure)

---

## 5. Dữ liệu thực nghiệm

- **PopHR dataset**: 1.2 triệu bệnh nhân từ Quebec, Canada
- Theo dõi longitudinal lên đến 20 năm
- **Xử lý**: Collapse time series → frequency vector cho mỗi bệnh nhân
- **Chuyển đổi**: DIN codes (>10,000) → ATC codes (1057)

### 5.1 Phenotypes được đánh giá (12 bệnh)
- AMI, Asthma, CHF, COPD, Diabetes, Hypertension
- IHD, Epilepsy, Schizophrenia, ADHD, HIV, Autism

---

## 6. Các task đánh giá

### 6.1 Reconstruction Task
- Split: 60% train, 30% validation, 10% test
- Chia mỗi test document làm 2 nửa: 1 nửa để infer θ, 1 nửa để evaluate

### 6.2 Topic Quality
- **Topic Coherence (TC)**: Đo co-occurrence của top codes trong cùng topic
- **Topic Diversity (TD)**: Đo tính độc nhất của codes across topics
- **Topic Quality = TC × TD**

### 6.3 Phenotype Classification
- Train unsupervised model → infer topic mixture
- Train LASSO classifier dùng θ_p làm features
- Evaluate bằng AUROC

### 6.4 Drug Imputation
- Input: Chỉ ICD codes của bệnh nhân
- Output: Dự đoán ATC codes
- Metrics: Prec@5, Recall@5, F1@5, Drug-wise precision

---

## 7. Kết quả

### 7.1 So sánh với baselines

| Model | Recon. NLL | Topic Quality |
|-------|------------|---------------|
| MixEHR | 203.97 | 0.0673 |
| ETM | 198.26 | 0.0704 |
| GETM | 184.32 | 0.1843 |
| **GAT-ETM** | **172.69** | **0.1920** |

### 7.2 Drug Imputation (Patient-wise)

| Model | Prec@5 | Recall@5 | F1@5 |
|-------|--------|----------|------|
| Frequency-based | 0.1049 | 0.0432 | 0.0577 |
| KNN | 0.1606 | 0.0713 | 0.0930 |
| ETM | 0.1823 | 0.0833 | 0.1075 |
| GETM | 0.2378 | 0.1101 | 0.1418 |
| **GAT-ETM** | **0.2600** | **0.1225** | **0.1569** |

### 7.3 Phenotype Classification
- GAT-ETM đạt AUROC cao nhất trên cả 12 bệnh mạn tính

---

## 8. Ablation Study

| Model | Recon. NLL | TQ (ave.) |
|-------|------------|-----------|
| GAT-ETM (full) | 172.69 | 0.1920 |
| w/o Node2Vec init | 179.59 | 0.0830 |
| w/o Graph augmentation | 181.63 | 0.1651 |
| GETM w/ augmentation | 180.44 | 0.1768 |

**Kết luận**:
- Graph augmentation cải thiện reconstruction nhiều nhất
- Node2Vec initialization quan trọng nhất cho Topic Quality
- GAT (end-to-end) giúp balance cả hai metrics

---

## 9. Kết quả định tính

### 9.1 Topic Coherence
Các topic học được có ý nghĩa lâm sàng:
- Topic 15: Pneumonia
- Topic 25: Cystic Fibrosis
- Topic 61: Congenital Heart Defects
- Topic 72: Thyroiditis
- Topic 78: Connective Tissue Diseases

### 9.2 Code Embedding Visualization (t-SNE)
- ICD và ATC cùng nhóm bệnh lý cluster gần nhau
- Ví dụ: ICD "Skin diseases" + ATC "Dermatologicals" cluster cùng nhau

### 9.3 Drug Imputation Case Study
- Thuốc được đề xuất có khoảng cách ngắn đến các mã ICD quan sát được trong knowledge graph

---

## 10. Hướng phát triển tương lai

1. **Mở rộng Knowledge Graph**: Thêm UMLS, Gene Ontology, multi-relational graphs
2. **Drug-Drug Interactions**: Tránh đề xuất thuốc có tương tác bất lợi
3. **Guided Topic Models**: Dùng PheCodes, CCS làm anchor topics
4. **Attention Analysis**: Phân tích attention weights để hiểu disease comorbidity
5. **Dynamic Topic Model**: Mô hình hóa sự tiến triển bệnh theo thời gian

---

## 11. Ý nghĩa cho nghiên cứu EHR Việt Nam

### 11.1 Yêu cầu dữ liệu
1. **Mã ICD-10**: Việt Nam sử dụng ICD-10 (khác với ICD-9 trong paper)
2. **Mã thuốc**: Cần mapping từ mã thuốc Việt Nam sang ATC codes
3. **Knowledge Graph**: Xây dựng quan hệ bệnh-thuốc phù hợp thực hành lâm sàng VN

### 11.2 Các bước triển khai
1. Thu thập và chuẩn hóa dữ liệu EHR từ bệnh viện
2. Xây dựng ICD-10 hierarchy và drug hierarchy
3. Thiết lập disease-drug relationships
4. Chuyển đổi dữ liệu sang bag-of-words representation
5. Train GAT-ETM model
6. Đánh giá trên các phenotype phổ biến tại Việt Nam

### 11.3 Ứng dụng tiềm năng
- **Computational phenotyping**: Tự động xác định phenotype bệnh nhân
- **Drug recommendation**: Đề xuất thuốc dựa trên chẩn đoán
- **Patient stratification**: Phân tầng bệnh nhân cho nghiên cứu
- **Comorbidity analysis**: Phân tích bệnh đồng mắc

---

## 12. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of topics (K) | 100 |
| Embedding dimension (L) | 256 |
| Encoder hidden size | 128 |
| GAT layers | 3 |
| GAT attention heads | 4 |
| Learning rate | 0.01 |
| Batch size | 512 |
| Weight decay | 1.2 × 10^-6 |

---

*Tóm tắt bởi: Claude Code*
*Ngày: 2026-03-18*
