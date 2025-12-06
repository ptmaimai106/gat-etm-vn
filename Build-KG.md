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

