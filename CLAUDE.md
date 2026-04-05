# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAT-ETM (Graph ATtention-Embedded Topic Model) là một mô hình topic neural end-to-end học đồng thời knowledge graph của medical codes và topic distributions từ Electronic Health Records (EHR). Paper gốc: [Nature Scientific Reports 2022](https://www.nature.com/articles/s41598-022-22956-w).

**Mục tiêu hiện tại**: Adapt mô hình cho dữ liệu EHR Việt Nam (ICD-10 + ATC drug codes).

## Common Commands

### Training
```bash
# Train model
python main_getm.py --data_path data_vn/ --meta_file metadata --num_topics 50 --epochs 100 --batch_size 512 --lr 0.01 --save_path results_vn/ --gpu_device 0

# Evaluate trained model
python main_getm.py --data_path data_vn/ --meta_file metadata --mode eval --load_from results_vn/trained_model --num_topics 50
```

### Vietnamese Data Pipeline
```bash
# Phase 1: Parse drug names from Excel
cd scripts/ && python parse_drug_names.py

# Phase 2: Map drugs to ATC codes
python drug_atc_mapping.py

# Phase 3: Build ICD-10 hierarchy graph
python build_icd10_graph.py

# Phase 4: Build complete knowledge graph
python build_knowledge_graph.py
```

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install simple-icd-10 fuzzywuzzy python-Levenshtein  # For Vietnamese data
```

## Architecture

### Core Model Components

**graph_etm.py** - Main GAT-ETM architecture:
- `GCNet`: Graph Attention Network (3 layers, 4 attention heads) để học code embeddings từ knowledge graph
- `GETM`: Model chính kết hợp VAE encoder (infers θ từ BoW) + GAT layers + linear decoder

**dataset.py** - Data handling:
- `NontemporalDataset`: PyTorch Dataset cho EHR data với train/test splits
- Hỗ trợ multiple code types (ICD, ATC) với separate normalization

**main_getm.py** - Training script với các arguments quan trọng:
- `--data_path`: Thư mục chứa BoW matrices và metadata
- `--num_topics`: Số topics (thường 20-100)
- `--tq`: Enable topic-code embeddings
- `--drug_imputation`: Enable drug imputation mode

### Data Flow

```
Raw EHR Data → [parse_drug_names.py] → Drug List
                     ↓
            [drug_atc_mapping.py]
                     ↓
ICD-10 + ATC codes → [build_knowledge_graph.py] → Knowledge Graph + Node2Vec embeddings
                     ↓
Patient BoW matrices → [main_getm.py] → Topics + Code Embeddings
```

### Key Data Formats

**Metadata file** (data_vn/metadata.txt):
```
icd atc                    # code types
2788 1075                  # vocab sizes
1 1                        # train embeddings flag (0=freeze, 1=train)
embed_path embed_path      # embedding file paths
```

**BoW matrices**: `bow_train.npy`, `bow_test.npy`, `bow_test_1.npy`, `bow_test_2.npy`

### Directory Structure

- `scripts/` - Data processing pipeline cho Vietnamese EHR
- `data_vn/` - Processed BoW matrices và metadata
- `embed_vn/` - Knowledge graphs và node embeddings
- `KG_EMBED/` - MIMIC-III knowledge graph builders (reference)
- `note/` - Documentation và implementation plans

## Vietnamese Data Specifics

**ICD-10 Processing**:
- Sử dụng `simple-icd-10` package (WHO ICD-10 2019)
- Graph augmentation: connect mỗi node tới ALL ancestors (không chỉ parent)
- Normalization: xử lý special markers (*, +, !)

**ATC Mapping**:
- Manual dictionary (260+ entries) + fuzzy matching (threshold 0.65-0.70)
- Current coverage: 585/1075 drugs (54.4%), 86.1% prescriptions

**Knowledge Graph**:
- ICD-10 hierarchy + ATC hierarchy + disease-drug co-occurrence edges
- Node2Vec: walk_length=20, walks=10, dim=256

## Key Documentation

- `note/GAT-ETM-paper-summary.md` - Paper summary (Vietnamese)
- `note/PLAN-GAT-ETM-VN-DATA.md` - 8-phase implementation plan với session logs
- `note/TRAINING_GUIDE.md` - Training instructions
