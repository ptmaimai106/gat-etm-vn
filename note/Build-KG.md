Input:
- Embedded Knowledge Graph được tạo ra từ các liên kết: ICD-ICD, ATC-ATC, ICD-ATC
	+ Trích xuất các mã ICD, ATC từ dataset 
	+ Nếu dữ liệu gốc là mã thuốc thương mại -> ánh xạ sang mã ATC 
	+ Hierarchical edges (Taxonomy): tạo cây phân cấp
		* ICD: `401` → `4019` (parent-child)
		* ATC: `A` → `A10` → `A10B` → `A10BA02`
	+ Augmentation: kết nối mỗi node với tất cả tổ tiên 
	#### Output files cần có:
	```
	embed/
	├── augmented_icdatc_graph_{params}_renumbered_by_vocab.pkl
	│   └── NetworkX Graph object (pickle)
	│
	└── augmented_icdatc_embed_{window}_{walk_length}_{num_walks}_{dim}_by_vocab.pkl
		└── numpy array: (num_nodes, embedding_dim)
	```

- Dữ liệu EHR dưới dạng bag-of-word:
	+ Dạng túi từ bag-of-word: hồ sơ bệnh nhân được coi như một tài liệu, trong đó mã y tế(ICD, ATC) được coi như là các từ
	+ Vector tần suất: Vp = Vicd, Vatc
	+ Xử lý dữ liệu phi cấu trúc → trích xuất mã → chuẩn hóa → vector tần suất
	-> tạo các file BoW Format
	-> output step này: 
		data/bow_train.npy
		data/bow_test.npy
		data/bow_test_1.npy
		data/bow_test_2.npy
		#### Format:
			- type: `scipy.sparse.csr_matrix` được lưu dưới dạng `.npy` với `allow_pickle=True`
			- shape: `(num_patients, vocab_size[0] + vocab_size[1] + ...)`
			- column `[ICD codes (0:vocab_size[0]), ATC codes (vocab_size[0]:vocab_size[0]+vocab_size[1]), ...]`
			- value: Binary (0/1) hoặc frequency

- Metadata File: data/metadata.txt
	+ ví dụ: 
		```
		['icd', 'atc'] --> Code types (tên các loại mã y tế)
		[61210, 82020] -->Vocab sizes (số lượng codes của mỗi loại)
		[1, 1] --> Train init embeddings flag (`1` = train, `0` = freeze)
		['*', '*'] -->  Initial embedding paths (`'*'` = dùng KG embeddings)
		```
	+ note:
		- Thứ tự code types PHẢI khớp với thứ tự columns trong BoW
		- Vocab sizes PHẢI chính xác với số lượng unique codes







==============




ATC_ROOT (level 0, structural)
  └── ATC_GROUP (level 1, structural)
      └── ATC_65 (level 2, structural - từ 2 chữ số đầu của DRUG_65944)
          └── DRUG_65944 (level 3, VOCABULARY ITEM - mã thuốc thực tế)


ICD9_ROOT (level 0, structural)
  └── 3-digit prefix (level 3, structural)
      └── 4-digit prefix (level 4, structural)
          └── 5-digit prefix (level 5, structural - nếu có)
              └── full_code (level N, VOCABULARY ITEM - mã ICD thực tế)



1. SEMANTIC STRUCTURE (Cấu trúc ngữ nghĩa):
           - Hierarchy cung cấp cấu trúc "is-a" (là một loại của)
           - Ví dụ: "4019" (hypertension) là một loại của "401" (circulatory diseases)
           - Giúp GAT hiểu được semantic relationships giữa các codes
        
2. GENERALIZATION (Tổng quát hóa):
	- Parent nodes đại diện cho concepts tổng quát
	- Child nodes đại diện cho concepts cụ thể
	- Model có thể học từ parent và áp dụng cho child nodes
	- Quan trọng cho rare codes (codes hiếm gặp trong training data)

3. KNOWLEDGE TRANSFER (Chuyển giao tri thức):
	- Codes cùng parent có semantic similarity
	- Nếu model biết "4019" liên quan đến một thuốc, có thể suy luận về "401" (parent)
	- Giúp model học được patterns từ ít dữ liệu hơn

4. GRAPH ATTENTION NETWORK (GAT):
	- GAT sử dụng hierarchy để tính attention weights
	- Attention mechanism tập trung vào các nodes liên quan trong hierarchy
	- 3-layer GAT: mỗi layer học từ neighbors ở các levels khác nhau
	- Final embedding ρ = kết hợp embeddings từ tất cả layers

5. EMBEDDING LEARNING:
	- Node embeddings được học từ cả hierarchical structure và co-occurrence
	- Parent nodes có embeddings đại diện cho toàn bộ subtree
	- Child nodes kế thừa features từ parent + specific features

6. AUGMENTATION (Skip Connections):
	- Kết nối mỗi node với tất cả ancestors (skip connections)
	- Weight decay: 0.9^distance (càng xa càng nhẹ)
	- Giúp information flow nhanh hơn trong GNN
	- Cho phép model học được long-range dependencies
	


KG của GAT-ETM được build từ: 
    - Hierarchical knowledge (ICD tree, ATC tree)
    - Medical ontologies (SNOMED/ICD mappings, drug–disease links)
    - Co-occurrence graph (tự build từ PopHR)
    - Additional curated medical relations
Các thông tin có trong private dataset PopHR:
	ICD diagnosis
		+ ATC drug codes
		+ Lab tests
		+ Procedures
		+ Temporal visits
		+ Internal knowledge-base links
		+ Drug–disease causal info
		+ Local hierarchical coding system
KG trong GAT-ETM gồm những loại node/edge nào ? 
	Node types:
		+ ICD diagnosis codes
		+ ATC drug codes
		+ Procedure codes (CPT)
		+ Lab item codes
		+ Clinical concepts (optional)
	Edge types:
		A. Ontology-based hierarchical edges
			ICD9 ← parent-child → ICD9
			CPT hierarchy
			ATC drug hierarchy	
		B. Co-occurrence edges
			ICD9 (disease) ↔ ATC (drug)
			ICD9 ↔ CPT
			ICD9 ↔ lab test
		C. Knowledge-based edges (từ UMLS, DrugBank, RxNorm)
			drug → treats → disease
			disease → causes → lab abnormality
			drug → has side effect → code
Tóm tắt cấu trúc KG họ build:
	ICD --hier--> ICD
	CPT --hier--> CPT
	ATC --hier--> ATC

	ICD --co-occurrence--> ATC
	ICD --co-occurrence--> CPT
	ICD --co-occurrence--> Lab

	ATC --side-effect--> ICD
	ATC --treats--> ICD

Với MIMIC-III nếu muốn build KG tương tự cần: 
    (1) ICD9 diagnosis hierarchy: D_ICD_DIAGNOSES -> (icd_child → icd_parent)
    (2) CPT procedure hierarchy: D_CPT + PROCEDURES_ICD / CPTEVENTS -> (cpt_child → cpt_parent)
    (3) ATC drug codes: khai thác bảng PRESCRIPTIONS (drug names → RxNorm → ATC)
    (4) Co-occurrence edges (cực kỳ quan trọng): ADMISSIONS + DIAGNOSES + PRESCRIPTIONS:
        ICD9 ↔ DRUG (ATC)
        ICD9 ↔ CPT
        ICD9 ↔ LAB
        ICD9 ↔ ICD9
    Co-occurrence rule: 
        If ICD9_a and DRUG_b appear in the same visit → add edge (a,b)
        If ICD9_a and CPT_c appear in same stay → add edge (a,c)

STEP build KG từ MIMIC-III
	Step 1. Extract code sets
		ICD9 codes  ← DIAGNOSES_ICD + D_ICD_DIAGNOSES
		CPT codes   ← PROCEDURES_ICD, CPTEVENTS, D_CPT
		DRUG codes  ← PRESCRIPTIONS → map to ATC (external)
		LAB codes   ← LABEVENTS → D_LABITEMS
	Step 2: Build edges
		(A) ICD9 hierarchy
			From D_ICD_DIAGNOSES: child → parent mapping using ICD9 prefix rules
		(B) CPT hierarchy
			From D_CPT, grouping by section.
		(C) DRUG → ATC mapping
			Use: RxNorm API (free) and WHO ATC index (free .csv)
		(D) Co-occurrence edges	
			Using DIAGNOSES_ICD, PRESCRIPTIONS, PROCEDURES_ICD, LABEVENTS.
			For each hospital admission (HADM_ID):
				Diagnoses ↔ Drugs
				Diagnoses ↔ Procedures
				Diagnoses ↔ Labs
		(E) Self co-occurrence among ICDs:
			If ICD_a and ICD_b appear in same visit → edge(a,b)
Sau khi build được KG, để đưa vào training GAT-ETM thì cần: 
    -> Convert KG to PyTorch Geometric format
    -> Generate ebedding similar to "augmented_icdatc_embed...pkl"



