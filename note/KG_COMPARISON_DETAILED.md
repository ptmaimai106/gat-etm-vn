# Giải Thích Chi Tiết và So Sánh Các Knowledge Graph

## Tổng Quan

Tài liệu này giải thích chi tiết 4 Knowledge Graph và so sánh từng cặp:
1. **Cặp 1:** Normal KG vs Augmented KG (full demo data)
2. **Cặp 2:** Simple Ego KG vs Simple Augmented KG (simple data)

---

## KIẾN THỨC NỀN TẢNG: Nodes và Edges trong Knowledge Graph

### 1. Ý Nghĩa của Nodes (Đỉnh)

Nodes trong Knowledge Graph đại diện cho các thực thể y tế từ dữ liệu MIMIC-III. Mỗi node có:
- **Type**: Loại thực thể (ICD9, CPT, ATC, LAB)
- **Level**: Mức độ trong hierarchy (0 = root, 1+ = các levels con)
- **Code**: Mã định danh thực tế (ví dụ: "401.9", "50868")

#### **1.1. ICD9 Nodes (Màu đỏ/Red)**

**Nguồn dữ liệu:** `DIAGNOSES_ICD.csv` từ MIMIC-III

**Ý nghĩa:**
- Đại diện cho **chẩn đoán bệnh** (diagnoses) theo hệ thống phân loại ICD-9
- ICD-9 là hệ thống mã hóa bệnh tật quốc tế phiên bản 9

**Cấu trúc Hierarchy:**
```
ICD9_ROOT (level 0)
  └── 401 (level 3) - 3-digit code: "Diseases of circulatory system"
      └── 4019 (level 4) - 4-digit code: "Essential hypertension, unspecified"
          └── 40190 (level 5) - 5-digit code: Specific subtype
```

**Ví dụ cụ thể:**
- `4019` = "Essential hypertension, unspecified" (Tăng huyết áp cần thiết, không xác định)
- `42731` = "Atrial fibrillation" (Rung nhĩ)
- `5849` = "Acute kidney failure, unspecified" (Suy thận cấp, không xác định)

**Cách build (từ code):**
1. Lấy tất cả ICD9 codes từ `DIAGNOSES_ICD.csv`
2. Normalize (bỏ dấu chấm): `"401.9"` → `"4019"`
3. Tạo hierarchy: với mỗi code, tạo tất cả prefix levels (3-digit, 4-digit, 5-digit)
4. Kết nối parent-child: `ICD9_ROOT → 401 → 4019 → 40190`

**Trong graph:**
- **Root nodes** (level 0): `ICD9_ROOT` - node gốc của toàn bộ ICD9 hierarchy
- **Intermediate nodes** (level 3-4): Các mã 3-digit, 4-digit - đại diện cho nhóm bệnh
- **Leaf nodes** (level 5): Các mã 5-digit cụ thể - đại diện cho chẩn đoán chi tiết
- **Vocab nodes**: Chỉ các leaf nodes thực sự xuất hiện trong dữ liệu (dùng cho training)

#### **1.2. CPT Nodes (Màu teal/Cyan hoặc Light Blue)**

**Nguồn dữ liệu:** `CPTEVENTS.csv` hoặc `PROCEDURES_ICD.csv`

**Ý nghĩa:**
- Đại diện cho **thủ thuật y tế** (procedures) theo hệ thống CPT (Current Procedural Terminology)
- CPT là hệ thống mã hóa các thủ thuật, dịch vụ y tế

**Cấu trúc Hierarchy:**
```
CPT_ROOT (level 0)
  └── CPT_2 (level 1) - Section node
      └── 20000 (level 2) - Specific CPT code
```

**Ví dụ cụ thể:**
- `CPT_2` = Section 2 (có thể là "Surgery" hoặc nhóm thủ thuật cụ thể)
- `20000` = Một thủ thuật cụ thể trong section đó

**Cách build (từ code):**
1. Lấy CPT codes từ `CPTEVENTS.csv` hoặc `PROCEDURES_ICD.csv`
2. Tạo section nodes dựa trên ký tự đầu: `"20000"` → section `"2"` → node `"CPT_2"`
3. Kết nối: `CPT_ROOT → CPT_2 → 20000`

**Trong graph:**
- **Root nodes**: `CPT_ROOT` - node gốc
- **Section nodes**: `CPT_1`, `CPT_2`, `CPT_3`, ... - nhóm các thủ thuật
- **Leaf nodes**: Các mã CPT cụ thể - thủ thuật thực tế được thực hiện

#### **1.3. ATC Nodes (Màu xanh nhạt/Light Blue hoặc Dark Blue)**

**Nguồn dữ liệu:** `PRESCRIPTIONS.csv` (drugs)

**Ý nghĩa:**
- Đại diện cho **thuốc** (drugs/medications) theo hệ thống ATC (Anatomical Therapeutic Chemical)
- ATC phân loại thuốc theo: Anatomical (cơ quan), Therapeutic (điều trị), Chemical (hóa học)

**Cấu trúc Hierarchy:**
```
ATC_ROOT (level 0)
  └── DRUG_ (level 1) - Intermediate level
      └── DRUG_5 (level 2) - Group level
          └── DRUG_55390 (level 3) - Specific drug code
```

**Lưu ý quan trọng:**
- Trong code hiện tại, ATC codes được tạo bằng **hash-based placeholder** (không phải ATC thực tế)
- Format: `DRUG_{hash(drug_name) % 100000:05d}`
- Ví dụ: `"Aspirin"` → hash → `DRUG_12345`
- **Trong production**, cần dùng RxNorm API hoặc ATC mapping file thực tế

**Cách build (từ code):**
1. Lấy drugs từ `PRESCRIPTIONS.csv` (cột `drug_name_generic`)
2. Tạo ATC-like codes bằng hash: `hash(drug_name) % 100000`
3. Tạo hierarchy: `ATC_ROOT → DRUG_ → DRUG_X → DRUG_XXXXX`
4. Lưu mapping: `drug_name → DRUG_XXXXX` trong `self.drug_to_atc`

**Trong graph:**
- **Root nodes**: `ATC_ROOT` - node gốc
- **Intermediate nodes**: `DRUG_`, `DRUG_5`, ... - các nhóm thuốc
- **Leaf nodes**: `DRUG_55390`, `DRUG_86785`, ... - các thuốc cụ thể
- **Đặc điểm**: Nhiều ATC nodes ở **peripheral area** (ngoại vi) vì chúng ít co-occurrence hơn

#### **1.4. LAB Nodes (Màu cam/Orange)**

**Nguồn dữ liệu:** `LABEVENTS.csv` và `D_LABITEMS.csv`

**Ý nghĩa:**
- Đại diện cho **xét nghiệm phòng thí nghiệm** (laboratory tests)
- Mỗi lab test có `itemid` và `category` (ví dụ: "Chemistry", "Hematology")

**Cấu trúc Hierarchy:**
```
LAB_ROOT (level 0)
  └── LAB_Chemistry (level 1) - Category node
      └── 50868 (level 2) - Specific lab itemid
  └── LAB_Hematology (level 1) - Category node
      └── 51275 (level 2) - Specific lab itemid
```

**Ví dụ cụ thể:**
- `LAB_Chemistry` = Category "Chemistry" (Hóa học)
- `50868` = Một xét nghiệm cụ thể trong category Chemistry (có thể là "Sodium", "Potassium", etc.)
- `LAB_Hematology` = Category "Hematology" (Huyết học)
- `51275` = Một xét nghiệm cụ thể trong category Hematology (có thể là "White Blood Cell Count", etc.)

**Cách build (từ code):**
1. Lấy lab itemids từ `LABEVENTS.csv`
2. Lấy category từ `D_LABITEMS.csv` (match theo `itemid`)
3. Tạo category nodes: `LAB_{category}` (ví dụ: `LAB_Chemistry`, `LAB_Hematology`)
4. Kết nối: `LAB_ROOT → LAB_Chemistry → 50868`

**Trong graph:**
- **Root nodes**: `LAB_ROOT` - node gốc
- **Category nodes**: `LAB_Chemistry`, `LAB_Hematology`, `LAB_Blood Gas`, ... - các nhóm xét nghiệm
- **Leaf nodes**: Các `itemid` cụ thể - xét nghiệm thực tế được chỉ định
- **Đặc điểm**: LAB nodes thường ở **central cluster** vì chúng có nhiều co-occurrence với ICD9 codes

---

### 2. Ý Nghĩa của Edges (Cạnh/Kết Nối)

Edges đại diện cho **mối quan hệ** giữa các nodes. Mỗi edge có:
- **edge_type**: Loại quan hệ (hierarchical, co-occurrence, augmented)
- **weight**: Trọng số (tần suất, độ quan trọng)

#### **2.1. Hierarchical Edges (Màu xám/Gray, Solid lines)**

**Ý nghĩa:**
- Đại diện cho **quan hệ phân cấp** (parent-child relationship)
- Mô tả cấu trúc tree/hierarchy của các mã y tế

**Ví dụ cụ thể:**
```
ICD9 Hierarchy:
  ICD9_ROOT → 401 (3-digit: "Diseases of circulatory system")
  401 → 4019 (4-digit: "Essential hypertension")
  4019 → 40190 (5-digit: Specific subtype)

LAB Hierarchy:
  LAB_ROOT → LAB_Chemistry
  LAB_Chemistry → 50868 (Sodium test)
  LAB_Chemistry → 50983 (Potassium test)

ATC Hierarchy:
  ATC_ROOT → DRUG_
  DRUG_ → DRUG_5
  DRUG_5 → DRUG_55390 (Specific drug)
```

**Cách build (từ code):**
1. **ICD9**: Với mỗi code, tạo prefix levels và kết nối parent-child
   ```python
   for i in range(3, len(normalized) + 1):
       prefix = normalized[:i]  # "4019" → "401", "4019"
       if i > 3:
           parent = normalized[:i-1]  # "401"
           icd9_edges.append((parent, prefix))  # ("401", "4019")
   ```

2. **LAB**: Kết nối category nodes với specific itemids
   ```python
   category_node = f'LAB_{category}'  # "LAB_Chemistry"
   lab_edges.append((root_lab, category_node))  # (LAB_ROOT, LAB_Chemistry)
   lab_edges.append((category_node, lab_node))  # (LAB_Chemistry, 50868)
   ```

3. **ATC**: Kết nối các levels trong drug hierarchy
   ```python
   atc_edges.append((root_atc, level1))  # (ATC_ROOT, DRUG_)
   atc_edges.append((level1, level2))   # (DRUG_, DRUG_5)
   atc_edges.append((level2, atc_code))  # (DRUG_5, DRUG_55390)
   ```

**Trong graph:**
- **Vị trí**: Thường ở **peripheral area** hoặc trong các hierarchical chains
- **Số lượng**: Ít hơn nhiều so với co-occurrence edges
- **Mục đích**: Giúp mô hình hiểu được cấu trúc phân cấp (ví dụ: "4019" là một loại của "401")

#### **2.2. Co-occurrence Edges (Màu đỏ/Red, Dashed lines)**

**Ý nghĩa:**
- Đại diện cho **sự đồng xuất hiện** (co-occurrence) của các codes trong cùng một admission
- Nếu hai codes xuất hiện cùng nhau trong cùng một `hadm_id` (hospital admission ID), chúng được kết nối

**Các loại Co-occurrence:**

**A. ICD9 ↔ LAB (Diagnosis ↔ Lab Test)**
- **Ý nghĩa lâm sàng**: Khi bệnh nhân có chẩn đoán X, họ thường được chỉ định xét nghiệm Y
- **Ví dụ**: 
  - `4019` (Hypertension) ↔ `50868` (Sodium test)
  - `42731` (Atrial fibrillation) ↔ `51275` (White Blood Cell Count)
- **Weight**: Số lần xuất hiện cùng nhau trong các admissions

**B. ICD9 ↔ ICD9 (Diagnosis ↔ Diagnosis)**
- **Ý nghĩa lâm sàng**: **Comorbidities** - các bệnh thường đi kèm nhau
- **Ví dụ**:
  - `4019` (Hypertension) ↔ `2724` (Hyperlipidemia) - Bệnh nhân tăng huyết áp thường có rối loạn lipid
  - `5849` (Kidney failure) ↔ `4019` (Hypertension) - Suy thận và tăng huyết áp thường đi kèm
- **Weight**: Số lần hai diagnoses xuất hiện cùng nhau

**C. ICD9 ↔ ATC (Diagnosis ↔ Drug)**
- **Ý nghĩa lâm sàng**: **Treatment patterns** - thuốc thường được kê cho chẩn đoán nào
- **Ví dụ**:
  - `4019` (Hypertension) ↔ `DRUG_55390` (có thể là thuốc hạ huyết áp)
  - `42731` (Atrial fibrillation) ↔ `DRUG_86785` (có thể là thuốc chống đông)
- **Weight**: Số lần diagnosis và drug xuất hiện cùng nhau

**D. ICD9 ↔ CPT (Diagnosis ↔ Procedure)**
- **Ý nghĩa lâm sàng**: **Procedure patterns** - thủ thuật thường được thực hiện cho chẩn đoán nào
- **Ví dụ**:
  - `4019` (Hypertension) ↔ `CPT_2` (có thể là các thủ thuật liên quan)
- **Weight**: Số lần diagnosis và procedure xuất hiện cùng nhau

**Cách build (từ code):**
```python
# Group codes by HADM_ID
hadm_to_icd = defaultdict(set)  # {hadm_id: {icd9_codes}}
hadm_to_atc = defaultdict(set)  # {hadm_id: {atc_codes}}
hadm_to_lab = defaultdict(set)  # {hadm_id: {lab_codes}}

# For each admission, create co-occurrence pairs
for hadm_id in admissions:
    # ICD9 ↔ LAB
    for icd in hadm_to_icd[hadm_id]:
        for lab in hadm_to_lab[hadm_id]:
            edge = (icd, lab)
            cooccurrence_counts[edge] += 1
    
    # ICD9 ↔ ICD9 (comorbidities)
    icd_list = list(hadm_to_icd[hadm_id])
    for i, icd1 in enumerate(icd_list):
        for icd2 in icd_list[i+1:]:
            edge = (icd1, icd2)
            cooccurrence_counts[edge] += 1
```

**Trong graph:**
- **Vị trí**: Chiếm đa số trong **central cluster**
- **Số lượng**: Rất nhiều (ví dụ: 65,583 ICD9-LAB connections)
- **Mục đích**: Giúp mô hình học được patterns lâm sàng (ví dụ: "Bệnh nhân tăng huyết áp thường có xét nghiệm sodium")

#### **2.3. Augmented Edges (Màu tím/Purple, Dotted lines)**

**Ý nghĩa:**
- Đại diện cho **skip connections** (kết nối bỏ qua) trong hierarchy
- Cho phép leaf nodes kết nối trực tiếp với ancestors (tổ tiên) của chúng, bỏ qua các levels trung gian

**Ví dụ cụ thể:**
```
Hierarchy truyền thống:
  ICD9_ROOT → 401 → 4019 → 40190

Augmented edges được thêm:
  40190 → 4019 (weight = 0.9^1 = 0.9)      # Skip 1 level
  40190 → 401 (weight = 0.9^2 = 0.81)      # Skip 2 levels
  40190 → ICD9_ROOT (weight = 0.9^3 = 0.729)  # Skip 3 levels
```

**Cách build (từ code):**
```python
# For each leaf node, find all ancestors
for node in graph.nodes():
    if node_type in ['ICD9', 'ATC', 'CPT']:
        ancestors = []  # List of ancestors
        current = node
        # Traverse up hierarchy
        while True:
            parent = find_parent(current)  # Find parent via hierarchical edge
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        
        # Add skip connections with decaying weights
        for i, ancestor in enumerate(ancestors):
            weight = 0.9 ** (i + 1)  # Decay factor
            graph.add_edge(node, ancestor, 
                          edge_type='augmented', 
                          weight=weight)
```

**Weight Decay:**
- Weight giảm dần khi khoảng cách tăng: `weight = 0.9^(distance)`
- **Lý do**: Mối quan hệ gần hơn (parent) quan trọng hơn mối quan hệ xa hơn (grandparent, root)
- Ví dụ:
  - Distance 1 (parent): weight = 0.9
  - Distance 2 (grandparent): weight = 0.81
  - Distance 3 (root): weight = 0.729

**Trong graph:**
- **Vị trí**: Kết nối các nodes trong cùng hierarchy
- **Số lượng**: Phụ thuộc vào số lượng leaf nodes và độ sâu của hierarchy
- **Mục đích**: 
  - Cải thiện **information flow** trong Graph Neural Networks
  - Cho phép thông tin truyền nhanh hơn qua graph
  - Đặc biệt hữu ích cho GAT-ETM với attention mechanism

**Lưu ý**: Trong một số visualizations, augmented edges **không hiển thị** (có thể do visualization code chưa render purple edges)

---

### 3. Isolated Nodes (Nodes Tự Do)

**Định nghĩa:**
- Nodes **không có edges nào** (degree = 0)
- Hoặc nodes chỉ có hierarchical edges nhưng không có co-occurrence edges

**Nguyên nhân:**

**A. Nodes chỉ có trong Hierarchy:**
- Các intermediate nodes trong hierarchy (ví dụ: `401`, `4019`) nhưng không xuất hiện trong dữ liệu thực tế
- Chúng chỉ tồn tại để tạo cấu trúc hierarchy, nhưng không có co-occurrence với nodes khác

**B. Nodes ít xuất hiện:**
- Một số codes xuất hiện rất ít trong dữ liệu
- Không đủ `min_cooccurrence` threshold để tạo co-occurrence edges
- Ví dụ: Một drug hiếm chỉ xuất hiện 1 lần, không có co-occurrence với diagnoses khác

**C. Nodes không match với dữ liệu:**
- Một số nodes được tạo trong hierarchy nhưng không có trong vocab (không xuất hiện trong dữ liệu thực tế)
- Ví dụ: `DRUG_64655` trong Simple KG - có thể là một drug code được tạo nhưng không có trong prescriptions thực tế

**Ví dụ trong Simple KG:**
- `DRUG_64655`: Isolated node ở bottom-center - không có connections nào
- Có thể do:
  1. Drug này không xuất hiện trong admissions được sample
  2. Không có co-occurrence với ICD9/LAB codes khác
  3. Chỉ có trong hierarchy nhưng không có trong ego-graph extraction

**Ý nghĩa:**
- **Trong training**: Isolated nodes vẫn có thể hữu ích vì chúng có hierarchical connections
- **Trong visualization**: Chúng xuất hiện như các nodes riêng lẻ, không kết nối với central cluster
- **Trong analysis**: Có thể chỉ ra các codes hiếm hoặc không phổ biến

**Cách xử lý:**
- **Option 1**: Giữ lại - chúng vẫn có thông tin từ hierarchy
- **Option 2**: Loại bỏ - nếu không có edges nào, có thể filter ra
- **Option 3**: Kết nối với root - thêm edge với root node để đảm bảo connectivity

---

### 4. Hub Nodes (Nodes Trung Tâm)

**Định nghĩa:**
- Nodes có **rất nhiều connections** (high degree)
- Thường là các codes phổ biến trong dữ liệu

**Ví dụ:**
- **`4019`** (Hypertension): Hub node với 65,583 ICD9-LAB connections
  - Lý do: Hypertension là bệnh rất phổ biến
  - Xuất hiện cùng với nhiều lab tests và diagnoses khác
  
- **`LAB_Chemistry`**: Hub node trong Simple KG
  - Lý do: Chemistry tests được chỉ định cho nhiều diagnoses khác nhau
  - Kết nối với nhiều ICD9 nodes

**Đặc điểm:**
- Thường ở **central cluster** của graph
- Có nhiều co-occurrence edges (đỏ)
- Đóng vai trò quan trọng trong information flow

**Ý nghĩa:**
- Phản ánh **patterns lâm sàng thực tế**
- Giúp mô hình học được các relationships quan trọng
- Có thể được sử dụng để identify common diseases/tests

---

## CẶP 1: Normal KG vs Augmented KG (Full Demo Data)

### 1.1. Normal Knowledge Graph (Full Demo Data)

#### **Thống Kê Tổng Quan**
- **Số nodes:** 500 nodes
- **Số edges:** 3,907 edges
- **Mật độ:** ~7.8 edges/node
- **Loại dữ liệu:** Full demo data (không có sampling)

#### **Cấu Trúc Node Types**

**ICD9 Nodes (Màu đỏ/Red)**
- **Số lượng:** Chiếm phần lớn trong central cluster
- **Ví dụ nodes:** `4019`, `5849`, `2724`, `42731`, `4280`, `25000`, `5990`, `2449`
- **Ý nghĩa chi tiết:**
  - `4019` = "Essential hypertension, unspecified" (Tăng huyết áp cần thiết) - **Hub node** với 65,583 connections
  - `42731` = "Atrial fibrillation" (Rung nhĩ) - Một diagnosis phổ biến
  - `5849` = "Acute kidney failure, unspecified" (Suy thận cấp)
  - `2724` = "Hyperlipidemia" (Rối loạn lipid máu)
  - Các nodes này có cả **hierarchical structure** (3-digit → 4-digit → 5-digit) và **co-occurrence** với LAB/ATC codes
- **Đặc điểm:**
  - Node `4019` là một hub node quan trọng, có nhiều connections
  - Các ICD9 codes xuất hiện trong nhiều admissions khác nhau
  - Có cả 3-digit, 4-digit, và 5-digit codes trong hierarchy

**CPT Nodes (Màu teal/Cyan)**
- **Số lượng:** Ít hơn ICD9, tập trung ở central cluster
- **Ví dụ nodes:** `CPT_2`, `CPT_3`, `CPT_4`, `CPT_9`
- **Ý nghĩa chi tiết:**
  - `CPT_2`, `CPT_3`, `CPT_4`, `CPT_9` = Các section nodes (level 1) trong CPT hierarchy
  - Đại diện cho các nhóm thủ thuật y tế (procedures)
  - Kết nối với ICD9 nodes qua **co-occurrence edges** (khi diagnosis và procedure xuất hiện cùng nhau)
- **Đặc điểm:**
  - Các CPT section nodes (level 1) - intermediate nodes trong hierarchy
  - Kết nối với ICD9 nodes qua co-occurrence edges (đỏ)
  - Có hierarchical structure: `CPT_ROOT → CPT_X → specific_CPT_code`

**ATC Nodes (Màu xanh nhạt/Light Blue)**
- **Số lượng:** Nhiều, phân bố ở peripheral area
- **Ví dụ nodes:** `DRUG_86785`, `DRUG_17446`, `DRUG_71607`, `DRUG_62939`, `DRUG_04999`, `DRUG_85260`
- **Ý nghĩa chi tiết:**
  - Đại diện cho **thuốc** (drugs/medications) được kê cho bệnh nhân
  - Codes được tạo bằng hash-based placeholder (không phải ATC thực tế)
  - Mỗi `DRUG_XXXXX` tương ứng với một `drug_name_generic` từ `PRESCRIPTIONS.csv`
  - Ví dụ: `DRUG_86785` có thể là "Aspirin" hoặc một thuốc khác (dựa trên hash)
- **Đặc điểm:**
  - Chủ yếu ở **peripheral area** (ngoại vi) vì ít co-occurrence hơn ICD9/LAB
  - Kết nối với central cluster qua **co-occurrence edges** (khi drug được kê cho diagnosis)
  - Có **hierarchical structure**: `ATC_ROOT → DRUG_ → DRUG_X → DRUG_XXXXX`
  - Một số DRUG nodes có thể là **isolated nodes** nếu không có co-occurrence

**LAB Nodes (Màu cam/Orange)**
- **Số lượng:** Nhiều trong central cluster
- **Ví dụ nodes:** `51041` (Blood Gas), `50983`, `50882`, `51275`, `50971`
- **Ý nghĩa chi tiết:**
  - Đại diện cho **xét nghiệm phòng thí nghiệm** (laboratory tests)
  - Mỗi node là một `itemid` từ `LABEVENTS.csv`
  - Có **category nodes**: `LAB_Chemistry`, `LAB_Hematology`, `LAB_Blood Gas`, ...
  - Ví dụ cụ thể:
    - `51041` = Blood Gas test (xét nghiệm khí máu)
    - `50868` = Sodium test (xét nghiệm natri) - thuộc category Chemistry
    - `51275` = White Blood Cell Count (đếm bạch cầu) - thuộc category Hematology
- **Đặc điểm:**
  - Tập trung ở **central cluster** vì có nhiều co-occurrence với ICD9 codes
  - Có nhiều **co-occurrence edges** (đỏ) với ICD9 nodes (khi lab test được chỉ định cho diagnosis)
  - Có **hierarchical structure**: `LAB_ROOT → LAB_Category → itemid`
  - Category nodes như `LAB_Chemistry` có thể là **hub nodes** nếu kết nối với nhiều ICD9 codes

#### **Cấu Trúc Edge Types**

**Hierarchical Edges (Màu xám/Gray lines, Solid)**
- **Số lượng:** Ít hơn nhiều so với co-occurrence
- **Ý nghĩa:** Đại diện cho quan hệ phân cấp (parent-child) trong hierarchy
- **Ví dụ cụ thể:**
  - ICD9: `ICD9_ROOT → 401 → 4019` (root → 3-digit → 4-digit)
  - LAB: `LAB_ROOT → LAB_Chemistry → 50868` (root → category → specific test)
  - ATC: `ATC_ROOT → DRUG_ → DRUG_5 → DRUG_55390` (root → intermediate → group → specific drug)
  - `DRUG_36175 → DRUG_36` (child → parent trong ATC hierarchy)
- **Đặc điểm:**
  - Tạo cấu trúc **tree-like** cho các node types
  - Kết nối các **levels** trong hierarchy (level 0 → 1 → 2 → ...)
  - Thường ở **peripheral area** hoặc trong các hierarchical chains
  - **Không có weight** (hoặc weight = 1.0) vì đây là structural relationships

**Co-occurrence Edges (Màu đỏ/Red lines, Dashed)**
- **Số lượng:** Chiếm đa số (~3,800+ edges)
- **Ý nghĩa:** Đại diện cho sự đồng xuất hiện của codes trong cùng một admission (`hadm_id`)
- **Các loại co-occurrence:**

  1. **ICD9 ↔ LAB (65,583 connections)**
     - **Ý nghĩa lâm sàng:** Khi bệnh nhân có diagnosis X, họ thường được chỉ định lab test Y
     - **Ví dụ cụ thể:**
       - `4019` (Hypertension) ↔ `50983` (Potassium test) - Bệnh nhân tăng huyết áp thường được kiểm tra kali
       - `4019` ↔ `50882` (Sodium test) - Kiểm tra natri cho bệnh nhân tăng huyết áp
       - `4019` ↔ `51275` (WBC count) - Đếm bạch cầu
     - **Weight:** Số lần xuất hiện cùng nhau trong các admissions
     - Node `4019` là **hub node** với rất nhiều LAB connections

  2. **ICD9 ↔ ICD9 (10,576 connections)**
     - **Ý nghĩa lâm sàng:** **Comorbidities** - các bệnh thường đi kèm nhau
     - **Ví dụ cụ thể:**
       - `5849` (Kidney failure) ↔ `4019` (Hypertension) - Suy thận và tăng huyết áp thường đi kèm
       - `4019` ↔ `2724` (Hyperlipidemia) - Tăng huyết áp và rối loạn lipid máu
       - `42731` (Atrial fibrillation) ↔ `4280` (Heart failure) - Rung nhĩ và suy tim
     - **Weight:** Số lần hai diagnoses xuất hiện cùng nhau
     - Phản ánh **real-world comorbidities** trong dân số bệnh nhân

  3. **ICD9 ↔ ATC (Nhiều connections)**
     - **Ý nghĩa lâm sàng:** **Treatment patterns** - thuốc thường được kê cho diagnosis nào
     - **Ví dụ:** `4019` (Hypertension) ↔ `DRUG_55390` (có thể là thuốc hạ huyết áp)
     - **Weight:** Số lần diagnosis và drug xuất hiện cùng nhau trong prescriptions

  4. **ICD9 ↔ CPT (Connections)**
     - **Ý nghĩa lâm sàng:** **Procedure patterns** - thủ thuật thường được thực hiện cho diagnosis nào
     - **Ví dụ:** `4019` ↔ `CPT_2` (có thể là các thủ thuật liên quan đến tăng huyết áp)
     - **Weight:** Số lần diagnosis và procedure xuất hiện cùng nhau

#### **Cấu Trúc Graph**

**Central Cluster (Dense Core)**
- **Thành phần:** ICD9 (red) + LAB (orange) + CPT (teal)
- **Mật độ:** Cực kỳ cao với nhiều co-occurrence edges
- **Đặc điểm:**
  - Node `4019` là một super-hub
  - Nhiều numerical labels: `4019`, `50983`, `51041`, `51419`, `2724`, `51275`, `50882`
  - Các nodes này xuất hiện cùng nhau trong nhiều admissions

**Peripheral Area**
- **Thành phần:** Chủ yếu là ATC nodes (light blue)
- **Kết nối:**
  - Một số DRUG nodes kết nối với central cluster qua **co-occurrence edges** (đỏ)
    - Ví dụ: `DRUG_55390` ↔ `4019` (drug được kê cho diagnosis)
  - Các DRUG nodes tạo **hierarchical chains** với nhau (xám)
    - Ví dụ: `DRUG_53555 → DRUG_29807 → DRUG_33747 → DRUG_60320`
    - Đây là parent-child relationships trong ATC hierarchy
- **Isolated Nodes:**
  - Một số DRUG nodes có thể là **isolated** (chỉ có hierarchical edges, không có co-occurrence)
  - Lý do: Drugs hiếm hoặc không phổ biến, không xuất hiện cùng với diagnoses trong sample data

#### **Ý Nghĩa Lâm Sàng**

1. **Hub Node `4019`:**
   - Đây là một ICD9 code rất phổ biến (có thể là "Essential hypertension" - 401.9)
   - Xuất hiện cùng với nhiều lab tests và diagnoses khác
   - Phản ánh thực tế: bệnh nhân tăng huyết áp thường có nhiều xét nghiệm và comorbidities

2. **ICD9-LAB Co-occurrence:**
   - 65,583 connections cho thấy mối quan hệ mạnh mẽ giữa diagnoses và lab tests
   - Các lab tests được chỉ định dựa trên diagnoses

3. **ICD9-ICD9 Co-occurrence:**
   - 10,576 connections cho thấy nhiều bệnh nhân có multiple diagnoses
   - Phản ánh comorbidities trong dân số bệnh nhân

---

### 1.2. Augmented Knowledge Graph (Full Demo Data)

#### **Thống Kê Tổng Quan**
- **Số nodes:** 500 nodes (giống Normal KG)
- **Số edges:** 3,820 edges (ít hơn Normal KG 87 edges)
- **Mật độ:** ~7.6 edges/node
- **Loại dữ liệu:** Full demo data với augmented edges

#### **Sự Khác Biệt Chính: Augmented Edges**

**Augmented Edges (Skip Connections)**
- **Mục đích:** Tạo skip connections giữa leaf nodes và ancestors trong hierarchy
- **Cơ chế:**
  - Với mỗi leaf node (ICD9, ATC, CPT), tìm tất cả ancestors
  - Thêm edge trực tiếp từ leaf node đến mỗi ancestor
  - Weight = `0.9^(distance)` với distance là số levels
- **Màu sắc:** Purple/Violet (màu tím) - **NHƯNG không hiển thị trong visualization này**
- **Style:** Dotted lines

**Ví dụ Augmented Edge:**
```
Hierarchy: ICD9_ROOT → 401 → 4019

Augmented edges được thêm:
- 4019 → 401 (weight = 0.9^1 = 0.9)
- 4019 → ICD9_ROOT (weight = 0.9^2 = 0.81)
```

#### **Cấu Trúc Node Types (Tương tự Normal KG)**

**ICD9 Nodes (Red)**
- Ví dụ: `1944`, `2500`, `273`, `536`, `782`, `2054`, `349`, `999`, `289`, `230`, `530`, `251`, `1622`, `493`, `4932`, `7810`, `99631`, `410`, `284`, `332`, `707`, `998`, `568`, `V12`, `29`, `78120`, `57800`, `3404`, `958`, `25062`, `1977`, `18`, `416`, `8638`, `20289`, `20292`, `20303`, `553`, `20692`, `20694`, `196`, `443`, `571`, `820`, `2102`, `1953`, `1459`, `50876`, `G458`, `51069`, `551`, `51529`, `50805`, `51091`, `51030`, `51096`, `32659`, `32422`, `432`, `37486`, `44150`, `62270`, `99251`, `36620`, `99223`, `9926`, `99232`, `5151426`, `51450`, `51046`, `1028`, `5133`, `51424`, `51515`, `51516`, `9741`, `8796`, `51333`, `5121`, `51519`, `6519`, `51054`, `5135`, `5115`, `51355`, `51001`, `288`, `79`, `950`, `V1204`, `693`, `51347`, `51263`, `51235`, `51517`, `51460`, `51428`, `5131`, `5156`, `529`, `303`, `682`, `556`, `528`, `426`, `1980`, `1976`, `22`, `7`

**CPT Nodes (Orange)**
- Ví dụ: `CPT_2`, `CPT_3`, `CPT_4`, `CPT_9`
- Lưu ý: Trong visualization này, CPT được hiển thị màu orange thay vì teal

**ATC Nodes (Light Blue)**
- Ví dụ: `DRUG_55390`, `DRUG_93350`, `DRUG_18148`, `DRUG_08568`, `DRUG_49299`, `DRUG_05001`, `DRUG_05509`, `DRUG_33615`, `DRUG_33603`, `DRUG_03002`, `DRUG_12592`, `DRUG_61418`, `DRUG_26988`, `DRUG_40782`, `DRUG_54494`, `DRUG_04460`, `DRUG_42629`, `DRUG_68135`, `DRUG_03269`, `DRUG_02222`, `DRUG_05723`, `DRUG_77619`, `DRUG_23732`, `DRUG_25815`, `DRUG_53555`, `DRUG_29807`, `DRUG_33747`, `DRUG_60320`, `DRUG_04591`, `DRUG_95773`, `DRUG_79461`, `DRUG_27026`, `DRUG_33067`, `DRUG_95159`, `DRUG_18741`, `DRUG_58740`, `DRUG_69932`, `DRUG_21499`, `DRUG_63012`, `DRUG_56476`, `DRUG_56961`, `DRUG_56378`, `DRUG_56`, `DRUG_45895`, `DRUG_19801`, `DRUG_66502`, `DRUG_05800`, `DRUG_33273`, `DRUG_97150`, `DRUG_68874`, `DRUG_07557`, `DRUG_62098`, `DRUG_68353`, `DRUG_68113`, `DRUG_57725`, `DRUG_74317`, `DRUG_99830`, `DRUG_90038`, `DRUG_8`, `DRUG_81979`, `DRUG_50459`, `DRUG_46782`, `DRUG_90198`, `DRUG_90646`, `DRUG_31693.02`, `DRUG_31`, `DRUG_90`, `DRUG_68408`, `DRUG_05564`, `DRUG_16433`, `DRUG_79463`, `DRUG_37103`, `DRUG_75113`, `DRUG_15465`, `DRUG_30691`, `DRUG_68443`, `DRUG_72730`, `DRUG_82476`, `DRUG_06254`, `DRUG_24595`, `DRUG_97414`, `DRUG_53182`, `DRUG_74143`, `DRUG_04683`, `DRUG_68216`, `DRUG_42245`, `DRUG_66415`, `DRUG_40978`, `DRUG_04453`, `DRUG_63697`, `DRUG_76465`, `DRUG_47311`, `DRUG_30492`, `DRUG_86190`, `DRUG_01026`, `DRUG_45389`, `DRUG_32817`, `DRUG_69167`, `DRUG_08192`, `DRUG_63219`, `DRUG_97781`, `DRUG_77890`, `DRUG_7503`

**LAB Nodes (Black/Dark)**
- Ví dụ: `LAB_CHEMISTRY` (single node, kết nối với central cluster)

#### **Cấu Trúc Edge Types**

**Hierarchical Edges (Gray)**
- Tương tự Normal KG
- Ví dụ: `DRUG_53555 → DRUG_29807 → DRUG_33747 → DRUG_60320` (hierarchical chain)
- Kết nối `LAB_CHEMISTRY` với central cluster

**Co-occurrence Edges (Red)**
- Tương tự Normal KG
- **ICD9-LAB:** 65,583 connections
  - Ví dụ: `4019 <-> 51006`, `4019 <-> 51275`, `4019 <-> 50971`
- **ICD9-ICD9:** 11,683 connections (nhiều hơn Normal KG)
  - Ví dụ: `42731 <-> 5990`, `4280 <-> 25000`, `5849 <-> 2449`

**Augmented Edges (Purple/Violet, Dotted)**
- **Ý nghĩa:** Skip connections cho phép leaf nodes kết nối trực tiếp với ancestors
- **Ví dụ cụ thể:**
  ```
  Hierarchy: ICD9_ROOT → 401 → 4019 → 40190
  
  Augmented edges được thêm:
  - 40190 → 4019 (weight = 0.9)      # Skip 1 level
  - 40190 → 401 (weight = 0.81)     # Skip 2 levels  
  - 40190 → ICD9_ROOT (weight = 0.729)  # Skip 3 levels
  ```
- **Weight decay:** `weight = 0.9^(distance)` - giảm dần khi khoảng cách tăng
- **Lưu ý:** Mặc dù được thêm vào graph, nhưng **KHÔNG hiển thị** trong visualization này
- **Có thể do:**
  - Visualization code chưa render purple edges (cần check `visualize_graph.py`)
  - Augmented edges bị filter trong quá trình visualization
  - Augmented edges quá ít hoặc không đủ nổi bật so với co-occurrence edges

#### **So Sánh với Normal KG**

| Đặc điểm | Normal KG | Augmented KG |
|----------|-----------|--------------|
| **Số nodes** | 500 | 500 |
| **Số edges** | 3,907 | 3,820 |
| **Augmented edges** | **KHÔNG** | **CÓ** (nhưng không hiển thị) |
| **ICD9-ICD9 connections** | 10,576 | 11,683 (nhiều hơn) |
| **ICD9-LAB connections** | 65,583 | 65,583 (giống nhau) |
| **Cấu trúc** | Dense central cluster | Tương tự, nhưng có skip connections |

**Nhận xét:**
- Augmented KG có ít edges hơn (3,820 vs 3,907) - có thể do:
  - Một số edges bị merge hoặc filter
  - Augmented edges thay thế một số hierarchical edges
  - Hoặc do sampling/filtering khác nhau trong quá trình build

---

### 1.3. So Sánh Chi Tiết Cặp 1: Normal vs Augmented

#### **A. Về Số Lượng Edges**

**Normal KG:** 3,907 edges
- Chủ yếu là co-occurrence edges
- Một số hierarchical edges
- **KHÔNG có** augmented edges

**Augmented KG:** 3,820 edges
- Có thêm augmented edges (skip connections)
- Ít hơn Normal KG 87 edges
- **Giải thích:** Có thể do:
  1. Augmented edges được thêm nhưng một số edges khác bị loại bỏ
  2. Hoặc do quá trình build/filter khác nhau
  3. Hoặc augmented edges không được đếm riêng trong visualization

#### **B. Về Cấu Trúc Graph**

**Normal KG:**
- Chỉ có 2 loại edges: hierarchical (gray) và co-occurrence (red)
- Cấu trúc tuân theo hierarchy truyền thống
- Information flow phải đi qua các levels trong hierarchy

**Augmented KG:**
- Có 3 loại edges: hierarchical (gray), co-occurrence (red), và augmented (purple)
- Có skip connections cho phép information flow trực tiếp
- Leaf nodes có thể kết nối trực tiếp với ancestors

#### **C. Về Ứng Dụng**

**Normal KG:**
- Phù hợp cho:
  - Exploratory analysis
  - Hiểu cấu trúc hierarchy truyền thống
  - Training models không cần skip connections

**Augmented KG:**
- Phù hợp cho:
  - Graph Neural Networks (GNN)
  - GAT-ETM với attention mechanism
  - Models cần information flow nhanh hơn
  - Training với skip connections

#### **D. Về Performance**

**Normal KG:**
- Information flow chậm hơn (phải đi qua nhiều levels)
- Có thể mất thông tin khi đi qua hierarchy
- Phù hợp cho models không cần long-range dependencies

**Augmented KG:**
- Information flow nhanh hơn (skip connections)
- Giữ được thông tin từ ancestors xa
- Phù hợp cho models cần long-range dependencies
- Có thể cải thiện performance của GNN models

---

## CẶP 2: Simple Ego KG vs Simple Augmented KG

### 2.1. Simple Ego Knowledge Graph

#### **Thống Kê Tổng Quan**
- **Số nodes:** 28 nodes
- **Số edges:** 28 edges
- **Mật độ:** 1 edge/node (rất sparse)
- **Loại dữ liệu:** Simple data với ego-graph extraction

#### **Cấu Trúc Node Types**

**LAB Nodes (Orange/Dark Orange)**
- **Ví dụ:** `LAB_Chemistry`, `LAB_Hematology`
- **Vai trò:** Central hubs của graph
- **Kết nối:** Nhiều connections với ICD9 nodes

**ICD9 Nodes (Red/Black text)**
- **Ví dụ:** `42731`, `50868`, `50912`, `50983`, `51274`, `51237`, `50971`, `51221`, `51275`, `50882`, `4273`, `50902`, `17`, `24`, `18`, `27`
- **Đặc điểm:**
  - Node `42731` là một hub quan trọng
  - Kết nối với cả `LAB_Chemistry` và `LAB_Hematology`
  - Có nhiều co-occurrence với các LAB nodes khác

**ATC Nodes (Light Orange/DRUG_)**
- **Ví dụ:** `DRUG_20`, `DRUG_20439`, `DRUG_66273`, `DRUG_09718`, `DRUG_66`, `DRUG_06615`, `DRUG_51533`, `DRUG_59`, `DRUG_59178`, `DRUG_69067`, `DRUG_69`, `DRUG_44419`, `DRUG_24`, `DRUG_24276`, `DRUG_64655`
- **Ý nghĩa chi tiết:**
  - Đại diện cho **thuốc** được kê cho bệnh nhân
  - Codes được tạo bằng hash-based placeholder từ `drug_name_generic`
  - Mỗi `DRUG_XXXXX` tương ứng với một thuốc cụ thể trong prescriptions
- **Đặc điểm:**
  - Tạo **hierarchical chains** (xám): `DRUG_09718 → DRUG_66 → DRUG_06615 → DRUG_51533`
    - Đây là parent-child relationships trong ATC hierarchy
  - Một số kết nối với ICD9 nodes qua **co-occurrence edges** (đỏ)
    - Khi drug được kê cho diagnosis cụ thể
  - **Isolated node:** `DRUG_64655` ở bottom-center
    - Chỉ có hierarchical edges, không có co-occurrence
    - Có thể là drug hiếm hoặc không xuất hiện trong sample data

**CPT Nodes (Teal/Light Blue)**
- **Số lượng:** Rất ít hoặc không có trong visualization này
- Có thể có nhưng không được hiển thị rõ

#### **Cấu Trúc Edge Types**

**Hierarchical Edges (Dark Gray, Solid)**
- **Ví dụ:**
  - `DRUG_09718 → DRUG_66 → DRUG_06615 → DRUG_51533` (hierarchical chain)
  - `DRUG_20439 → 24` (hierarchical)
  - `DRUG_66273 → 18` (hierarchical)
  - `DRUG_69067 → 4273` (hierarchical)
  - `DRUG_69 → 4273` (hierarchical)

**Co-occurrence Edges (Reddish-Orange, Dotted)**
- **ICD9 ↔ LAB (10 connections)**
  - **Ý nghĩa:** Lab tests được chỉ định cho diagnoses
  - **Ví dụ cụ thể:**
    - `42731` (Atrial fibrillation) ↔ `50868` (Sodium test)
    - `42731` ↔ `50912` (có thể là Potassium hoặc lab test khác)
    - `42731` ↔ `50983` (có thể là Creatinine hoặc lab test khác)
  - **Hub nodes:**
    - `LAB_Chemistry` kết nối với nhiều ICD9 nodes - đây là **hub node**
    - `LAB_Hematology` kết nối với một số ICD9 nodes
- **ICD9 ↔ ICD9 (Một số connections)**
  - **Ý nghĩa:** Comorbidities - các bệnh đi kèm nhau
  - Ví dụ: `42731` có thể kết nối với các ICD9 codes khác nếu chúng xuất hiện cùng nhau

#### **Cấu Trúc Graph**

**Central Cluster:**
- **Thành phần:** `LAB_Chemistry`, `LAB_Hematology`, `42731`
- **Kết nối:** Dense network của co-occurrence edges
- **Đặc điểm:**
  - `LAB_Chemistry` là hub chính
  - `42731` kết nối với cả hai LAB nodes
  - Nhiều ICD9 nodes xung quanh

**Peripheral Nodes:**
- **Thành phần:** DRUG nodes và một số ICD9 nodes
- **Kết nối:** Hierarchical chains và một số co-occurrence với central cluster
- **Ví dụ:**
  - `DRUG_09718 → DRUG_66 → DRUG_06615 → DRUG_51533` (chain ở bottom-left)
  - `DRUG_59 → DRUG_59178 → 18` (chain ở top-left)
  - `DRUG_69067 → 4273` và `DRUG_69 → 4273` (connections)

#### **Ý Nghĩa Lâm Sàng**

1. **LAB Hubs:**
   - `LAB_Chemistry` và `LAB_Hematology` là các category nodes quan trọng
   - Chúng kết nối với nhiều specific ICD9 codes
   - Phản ánh: các lab tests được chỉ định cho nhiều diagnoses khác nhau

2. **ICD9 Code `42731`:**
   - Là một diagnosis quan trọng (có thể là "Atrial fibrillation")
   - Xuất hiện cùng với nhiều lab tests
   - Kết nối với cả Chemistry và Hematology labs

3. **DRUG Hierarchical Chains:**
   - Cho thấy cấu trúc phân cấp của drugs
   - Các drugs có thể được nhóm theo categories

---

### 2.2. Simple Augmented Knowledge Graph

#### **Thống Kê Tổng Quan**
- **Số nodes:** 28 nodes (giống Simple Ego KG)
- **Số edges:** 28 edges (giống Simple Ego KG)
- **Mật độ:** 1 edge/node
- **Loại dữ liệu:** Simple data với augmented edges

#### **Sự Khác Biệt: Augmented Edges**

**Augmented Edges (Skip Connections)**
- **Mục đích:** Tạo skip connections trong hierarchy
- **Cơ chế:** Tương tự như Augmented KG (full data)
- **Weight:** `0.9^(distance)` với distance là số levels
- **Lưu ý:** Trong visualization này, **KHÔNG thấy purple edges**
- **Giải thích:** Có thể do:
  1. Augmented edges không được render trong visualization
  2. Hoặc augmented edges quá ít và không nổi bật
  3. Hoặc visualization chỉ hiển thị một số edge types

#### **Cấu Trúc Node Types (Tương tự Simple Ego KG)**

**LAB Nodes:**
- `LAB_Chemistry`, `LAB_Hematology` (central hubs)

**ICD9 Nodes:**
- `42731`, `50868`, `50912`, `50983`, `51274`, `51237`, `50971`, `51221`, `51275`, `50882`, `4273`, `50902`, `17`, `24`, `18`, `27`

**ATC Nodes:**
- `DRUG_20`, `DRUG_20439`, `DRUG_66273`, `DRUG_09718`, `DRUG_66`, `DRUG_06615`, `DRUG_51533`, `DRUG_59`, `DRUG_59178`, `DRUG_69067`, `DRUG_69`, `DRUG_44419`, `DRUG_24`, `DRUG_24276`, `DRUG_64655`

#### **Cấu Trúc Edge Types**

**Hierarchical Edges (Dark Gray):**
- Tương tự Simple Ego KG
- Ví dụ: `DRUG_09718 → DRUG_66 → DRUG_06615 → DRUG_51533`

**Co-occurrence Edges (Reddish-Orange, Dotted):**
- Tương tự Simple Ego KG
- **ICD9-LAB:** 10 connections
  - Ví dụ: `42731 <-> 50868`, `42731 <-> 50912`, `42731 <-> 50983`

**Augmented Edges (Purple):**
- **Lưu ý:** Mặc dù được thêm vào graph, nhưng **KHÔNG hiển thị** trong visualization
- Có thể do visualization không render purple edges hoặc augmented edges quá ít

#### **So Sánh với Simple Ego KG**

| Đặc điểm | Simple Ego KG | Simple Augmented KG |
|----------|---------------|---------------------|
| **Số nodes** | 28 | 28 |
| **Số edges** | 28 | 28 |
| **Augmented edges** | **KHÔNG** | **CÓ** (nhưng không hiển thị) |
| **Cấu trúc** | Chỉ hierarchical + co-occurrence | Có thêm skip connections |
| **Information flow** | Phải đi qua hierarchy | Có thể skip levels |

**Nhận xét:**
- Cả hai có cùng số nodes và edges
- Augmented version có thêm skip connections nhưng không thấy trong visualization
- Có thể augmented edges đã được thêm nhưng visualization không render chúng

---

### 2.3. So Sánh Chi Tiết Cặp 2: Simple Ego vs Simple Augmented

#### **A. Về Quy Mô**

**Simple Ego KG:**
- 28 nodes, 28 edges
- Rất sparse (1 edge/node)
- Được extract từ larger graph bằng ego-graph method

**Simple Augmented KG:**
- 28 nodes, 28 edges (giống nhau)
- Cùng sparse
- Có thêm augmented edges nhưng không làm tăng số edges hiển thị

#### **B. Về Cấu Trúc**

**Simple Ego KG:**
- Chỉ có 2 loại edges: hierarchical và co-occurrence
- Cấu trúc tuân theo hierarchy
- Information flow chậm hơn

**Simple Augmented KG:**
- Có 3 loại edges: hierarchical, co-occurrence, và augmented
- Có skip connections
- Information flow nhanh hơn

#### **C. Về Visualization**

**Simple Ego KG:**
- Hiển thị rõ 2 loại edges
- Dễ nhìn và phân tích

**Simple Augmented KG:**
- Augmented edges không hiển thị (có thể do visualization issue)
- Cần cải thiện visualization để hiển thị purple edges

#### **D. Về Ứng Dụng**

**Simple Ego KG:**
- Phù hợp cho:
  - Understanding structure
  - Debugging
  - Small-scale experiments

**Simple Augmented KG:**
- Phù hợp cho:
  - GNN training với skip connections
  - Testing augmented edge functionality
  - Small-scale GAT-ETM experiments

---

## Tổng Kết So Sánh Tất Cả 4 KG

### Bảng So Sánh Tổng Quan

| KG | Nodes | Edges | Augmented | Data Type | Use Case |
|----|-------|-------|-----------|-----------|----------|
| **Normal (Full)** | 500 | 3,907 | ❌ | Full demo | Analysis, Training |
| **Augmented (Full)** | 500 | 3,820 | ✅ | Full demo | GNN, GAT-ETM |
| **Simple Ego** | 28 | 28 | ❌ | Simple | Debug, Small-scale |
| **Simple Augmented** | 28 | 28 | ✅ | Simple | GNN testing |

### Key Insights

1. **Augmented Edges:**
   - Được thêm vào graph nhưng không luôn hiển thị trong visualization
   - Cần cải thiện visualization để render purple edges
   - Có thể cải thiện performance của GNN models

2. **Full Data vs Simple Data:**
   - Full data: 500 nodes, ~3,800-3,900 edges
   - Simple data: 28 nodes, 28 edges
   - Simple data là subgraph được extract từ full data

3. **Edge Count Discrepancy:**
   - Augmented KG có ít edges hơn Normal KG (3,820 vs 3,907)
   - Có thể do filtering hoặc merging trong quá trình build
   - Cần investigate thêm

4. **Visualization Issues:**
   - Augmented edges không hiển thị trong visualizations
   - Cần cải thiện visualization code để render purple edges
   - Hoặc cần verify xem augmented edges có thực sự được thêm vào không

---

## Khuyến Nghị

### 1. Cải Thiện Visualization
- Thêm code để render purple augmented edges
- Đảm bảo tất cả edge types được hiển thị
- Có thể thêm toggle để show/hide các edge types

### 2. Verify Augmented Edges
- Kiểm tra xem augmented edges có thực sự được thêm vào graph không
- Log số lượng augmented edges được thêm
- Verify trong graph object sau khi build

### 3. Experiment với Augmented KG
- So sánh performance của models trained trên Normal vs Augmented KG
- Đánh giá impact của skip connections
- Tune decay factor `e = 0.9` nếu cần

### 4. Documentation
- Document rõ sự khác biệt giữa các KG versions
- Giải thích khi nào nên dùng version nào
- Provide examples và use cases

---

## HƯỚNG DẪN ĐỌC VÀ PHÂN TÍCH KNOWLEDGE GRAPH

### 1. Cách Đọc Graph Visualization

**Bước 1: Xác định Node Types**
- **Đỏ (Red)**: ICD9 nodes - Chẩn đoán bệnh
- **Teal/Cyan**: CPT nodes - Thủ thuật y tế
- **Light Blue**: ATC nodes - Thuốc
- **Orange**: LAB nodes - Xét nghiệm phòng thí nghiệm
- **Gray**: UNKNOWN nodes - Không phân loại

**Bước 2: Xác định Edge Types**
- **Gray solid lines**: Hierarchical edges - Quan hệ phân cấp (parent-child)
- **Red dashed lines**: Co-occurrence edges - Đồng xuất hiện trong cùng admission
- **Purple dotted lines**: Augmented edges - Skip connections (nếu có)

**Bước 3: Phân tích Cấu Trúc**
- **Central cluster**: Dense area với nhiều co-occurrence edges
  - Thường chứa hub nodes (nodes có nhiều connections)
  - Phản ánh patterns lâm sàng phổ biến
- **Peripheral area**: Sparse area với ít connections
  - Thường chứa ATC nodes hoặc isolated nodes
  - Có nhiều hierarchical edges

**Bước 4: Tìm Hub Nodes**
- Nodes có nhiều connections (high degree)
- Thường ở central cluster
- Ví dụ: `4019` với 65,583 ICD9-LAB connections

**Bước 5: Tìm Isolated Nodes**
- Nodes không có edges hoặc chỉ có hierarchical edges
- Thường ở peripheral area
- Có thể là codes hiếm hoặc không phổ biến

### 2. Phân Tích Ý Nghĩa Lâm Sàng

**A. Co-occurrence Patterns:**
- **ICD9 ↔ LAB**: Cho thấy lab tests thường được chỉ định cho diagnoses nào
  - Ví dụ: Hypertension → Sodium test (kiểm tra natri cho bệnh nhân tăng huyết áp)
- **ICD9 ↔ ICD9**: Cho thấy comorbidities (bệnh đi kèm)
  - Ví dụ: Hypertension ↔ Hyperlipidemia (tăng huyết áp và rối loạn lipid thường đi kèm)
- **ICD9 ↔ ATC**: Cho thấy treatment patterns (thuốc thường được kê cho diagnosis nào)
  - Ví dụ: Hypertension → Antihypertensive drugs
- **ICD9 ↔ CPT**: Cho thấy procedure patterns (thủ thuật thường được thực hiện cho diagnosis nào)

**B. Hierarchical Structure:**
- Cho thấy cấu trúc phân cấp của các mã y tế
- Giúp hiểu mối quan hệ giữa general và specific codes
- Ví dụ: `401` (general) → `4019` (specific) → `40190` (very specific)

**C. Augmented Edges:**
- Cho phép skip connections trong hierarchy
- Giúp information flow nhanh hơn trong GNN
- Weight giảm dần khi khoảng cách tăng

### 3. Ví Dụ Phân Tích Cụ Thể

**Ví dụ 1: Hub Node `4019` (Hypertension)**
```
Node: 4019 (ICD9, Red)
Connections:
  - ICD9 ↔ LAB: 65,583 connections
    → 4019 ↔ 50983 (Potassium test)
    → 4019 ↔ 50882 (Sodium test)
    → 4019 ↔ 51275 (WBC count)
  - ICD9 ↔ ICD9: 10,576 connections
    → 4019 ↔ 2724 (Hyperlipidemia)
    → 4019 ↔ 5849 (Kidney failure)
  - ICD9 ↔ ATC: Nhiều connections
    → 4019 ↔ DRUG_55390 (có thể là thuốc hạ huyết áp)

Ý nghĩa:
- Hypertension là bệnh rất phổ biến
- Bệnh nhân tăng huyết áp thường có nhiều xét nghiệm
- Có nhiều comorbidities (hyperlipidemia, kidney failure)
- Được điều trị bằng nhiều loại thuốc khác nhau
```

**Ví dụ 2: LAB Hub `LAB_Chemistry`**
```
Node: LAB_Chemistry (LAB, Orange)
Connections:
  - LAB ↔ ICD9: Nhiều connections
    → LAB_Chemistry ↔ 42731 (Atrial fibrillation)
    → LAB_Chemistry ↔ 4019 (Hypertension)
    → LAB_Chemistry ↔ 5849 (Kidney failure)
  - Hierarchical:
    → LAB_ROOT → LAB_Chemistry → 50868 (Sodium)
    → LAB_ROOT → LAB_Chemistry → 50983 (Potassium)

Ý nghĩa:
- Chemistry tests được chỉ định cho nhiều diagnoses khác nhau
- Đây là category node (level 1) kết nối với nhiều specific tests
- Phản ánh: Chemistry tests là xét nghiệm phổ biến trong clinical practice
```

**Ví dụ 3: Isolated Node `DRUG_64655`**
```
Node: DRUG_64655 (ATC, Light Blue)
Connections:
  - Không có co-occurrence edges
  - Chỉ có hierarchical edges (nếu có)

Ý nghĩa:
- Drug này hiếm hoặc không phổ biến
- Không xuất hiện cùng với diagnoses trong sample data
- Có thể là:
  - Drug mới hoặc ít được sử dụng
  - Không có trong admissions được sample
  - Hoặc chỉ có trong hierarchy nhưng không có trong prescriptions thực tế
```

### 4. Checklist Khi Phân Tích Graph

- [ ] Xác định được các node types (ICD9, CPT, ATC, LAB)
- [ ] Xác định được các edge types (hierarchical, co-occurrence, augmented)
- [ ] Tìm được hub nodes (nodes có nhiều connections)
- [ ] Tìm được isolated nodes (nodes ít hoặc không có connections)
- [ ] Hiểu được cấu trúc central cluster vs peripheral area
- [ ] Phân tích được co-occurrence patterns (ICD9-LAB, ICD9-ICD9, etc.)
- [ ] Hiểu được hierarchical structure (parent-child relationships)
- [ ] Xác định được augmented edges (nếu có)
- [ ] Phân tích được ý nghĩa lâm sàng của các patterns

---

## TÓM TẮT: Ý Nghĩa Nodes và Edges

### Nodes (Đỉnh)
- **ICD9 (Red)**: Chẩn đoán bệnh - có hierarchy 3-digit → 4-digit → 5-digit
- **CPT (Teal)**: Thủ thuật y tế - có hierarchy section → specific code
- **ATC (Light Blue)**: Thuốc - có hierarchy root → intermediate → specific drug
- **LAB (Orange)**: Xét nghiệm - có hierarchy category → specific test

### Edges (Cạnh)
- **Hierarchical (Gray)**: Quan hệ phân cấp (parent-child) - structural relationships
- **Co-occurrence (Red)**: Đồng xuất hiện trong cùng admission - clinical patterns
- **Augmented (Purple)**: Skip connections - giúp information flow nhanh hơn

### Node States
- **Hub Nodes**: Nhiều connections - phản ánh patterns phổ biến
- **Isolated Nodes**: Ít hoặc không có connections - codes hiếm hoặc không phổ biến
- **Intermediate Nodes**: Nodes trong hierarchy (không phải leaf) - structural nodes

---

*Tài liệu này được tạo dựa trên code trong `build_kg_mimic.py`, `build_kg_mimic_sample.py`, `visualize_graph.py`, và phân tích các visualization images.*

