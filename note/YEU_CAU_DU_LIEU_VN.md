# Yêu Cầu Dữ Liệu Thực Tế Tại Việt Nam cho GAT-ETM

Tài liệu này mô tả các yêu cầu về dữ liệu cần thiết để áp dụng GAT-ETM với dữ liệu EHR thực tế tại Việt Nam.

---

## 📋 Tổng Quan

Để áp dụng GAT-ETM với dữ liệu Việt Nam, bạn cần chuẩn bị các loại dữ liệu sau:

1. **Dữ liệu chẩn đoán (ICD codes)** - Bắt buộc
2. **Dữ liệu thuốc (ATC codes)** - Bắt buộc  
3. **Dữ liệu thủ thuật/phẫu thuật** - Tùy chọn
4. **Dữ liệu xét nghiệm** - Tùy chọn
5. **Thông tin bệnh nhân và lần nhập viện** - Bắt buộc

---

## 🔍 Chi Tiết Từng Loại Dữ Liệu

### 1. Dữ Liệu Chẩn Đoán (ICD Codes) - BẮT BUỘC

#### 1.1. Bảng Chẩn Đoán (Diagnoses Table)

**Tên file:** `CHAN_DOAN.csv` hoặc `DIAGNOSES.csv`

**Các cột bắt buộc:**

| Tên cột | Kiểu dữ liệu | Mô tả | Ví dụ |
|---------|--------------|-------|-------|
| `ma_nhap_vien` | Integer/String | Mã nhập viện (tương đương HADM_ID) | `12345` |
| `ma_benh_nhan` | Integer/String | Mã bệnh nhân (tương đương SUBJECT_ID) | `67890` |
| `ma_icd` | String | Mã ICD-10 (hoặc ICD-9) | `I10`, `E11.9`, `A00` |
| `ngay_chan_doan` | Date/Datetime | Ngày chẩn đoán (tùy chọn) | `2024-01-15` |

**Lưu ý:**
- Mã ICD có thể có hoặc không có dấu chấm (`.`) - hệ thống sẽ tự động chuẩn hóa
- Ví dụ: `I10` và `I10.0` đều được chấp nhận
- Nên sử dụng ICD-10 (chuẩn quốc tế) hoặc ICD-10-VN (phiên bản Việt Nam)

#### 1.2. Bảng Từ Điển ICD (ICD Dictionary) - Tùy chọn nhưng khuyến nghị

**Tên file:** `D_ICD_CHAN_DOAN.csv` hoặc `ICD_DICTIONARY.csv`

**Các cột:**

| Tên cột | Kiểu dữ liệu | Mô tả |
|---------|--------------|-------|
| `ma_icd` | String | Mã ICD |
| `ten_benh` | String | Tên bệnh (tiếng Việt hoặc tiếng Anh) |
| `cap_do` | Integer | Cấp độ trong cây phân cấp (3-digit, 4-digit, 5-digit) |

**Mục đích:** Giúp xây dựng cấu trúc phân cấp (hierarchy) của ICD codes

---

### 2. Dữ Liệu Thuốc (ATC Codes) - BẮT BUỘC

#### 2.1. Bảng Kê Đơn Thuốc (Prescriptions Table)

**Tên file:** `KE_DON_THUOC.csv` hoặc `PRESCRIPTIONS.csv`

**Các cột bắt buộc:**

| Tên cột | Kiểu dữ liệu | Mô tả | Ví dụ |
|---------|--------------|-------|-------|
| `ma_nhap_vien` | Integer/String | Mã nhập viện | `12345` |
| `ma_benh_nhan` | Integer/String | Mã bệnh nhân | `67890` |
| `ten_thuoc` | String | Tên thuốc (generic name) | `Paracetamol`, `Amoxicillin` |
| `ma_atc` | String | Mã ATC (nếu có) | `N02BE01`, `J01CA04` |
| `ngay_ke_don` | Date/Datetime | Ngày kê đơn (tùy chọn) | `2024-01-15` |

**Lưu ý quan trọng:**

1. **Mapping Thuốc → ATC:**
   - Nếu đã có mã ATC trong dữ liệu → sử dụng trực tiếp
   - Nếu chỉ có tên thuốc → cần mapping từ tên thuốc sang mã ATC
   - Có thể sử dụng:
     - **WHO ATC Index** (https://www.whocc.no/atc_ddd_index/)
     - **RxNorm API** (nếu có)
     - **Bảng mapping thủ công** từ Bộ Y Tế Việt Nam

2. **Tên thuốc nên chuẩn hóa:**
   - Sử dụng tên generic (tên chung) thay vì tên thương mại
   - Ví dụ: `Paracetamol` thay vì `Panadol`, `Tylenol`
   - Có thể cần mapping từ tên thương mại → tên generic

#### 2.2. Bảng Mapping Thuốc → ATC (Drug-to-ATC Mapping)

**Tên file:** `DRUG_TO_ATC_MAPPING.csv` hoặc `THUOC_ATC.csv`

**Các cột:**

| Tên cột | Kiểu dữ liệu | Mô tả |
|---------|--------------|-------|
| `ten_thuoc` | String | Tên thuốc (generic name) |
| `ma_atc` | String | Mã ATC tương ứng |
| `nhom_atc` | String | Nhóm ATC (level 1-5) |

**Ví dụ:**

```
ten_thuoc,ma_atc,nhom_atc
Paracetamol,N02BE01,N02BE
Amoxicillin,J01CA04,J01CA
Metformin,A10BA02,A10BA
```

---

### 3. Dữ Liệu Thủ Thuật/Phẫu Thuật (Procedures) - TÙY CHỌN

#### 3.1. Bảng Thủ Thuật

**Tên file:** `THU_THUAT.csv` hoặc `PROCEDURES.csv`

**Các cột:**

| Tên cột | Kiểu dữ liệu | Mô tả |
|---------|--------------|-------|
| `ma_nhap_vien` | Integer/String | Mã nhập viện |
| `ma_benh_nhan` | Integer/String | Mã bệnh nhân |
| `ma_thu_thuat` | String | Mã thủ thuật (ICD-10-PCS hoặc mã nội bộ) |
| `ten_thu_thuat` | String | Tên thủ thuật |

**Lưu ý:**
- Có thể sử dụng mã ICD-10-PCS hoặc mã phân loại nội bộ của bệnh viện
- Nếu không có, có thể bỏ qua phần này

---

### 4. Dữ Liệu Xét Nghiệm (Lab Tests) - TÙY CHỌN

#### 4.1. Bảng Xét Nghiệm

**Tên file:** `XET_NGHIEM.csv` hoặc `LABEVENTS.csv`

**Các cột:**

| Tên cột | Kiểu dữ liệu | Mô tả |
|---------|--------------|-------|
| `ma_nhap_vien` | Integer/String | Mã nhập viện |
| `ma_benh_nhan` | Integer/String | Mã bệnh nhân |
| `ma_xet_nghiem` | String/Integer | Mã xét nghiệm (itemid) |
| `ten_xet_nghiem` | String | Tên xét nghiệm |
| `nhom_xet_nghiem` | String | Nhóm/category xét nghiệm |

**Ví dụ:**

```
ma_xet_nghiem,ten_xet_nghiem,nhom_xet_nghiem
LAB001,Công thức máu,Huyết học
LAB002,Đường huyết,Sinh hóa
LAB003,Chức năng gan,Sinh hóa
```

---

### 5. Thông Tin Bệnh Nhân và Nhập Viện - BẮT BUỘC

#### 5.1. Bảng Nhập Viện (Admissions Table)

**Tên file:** `NHAP_VIEN.csv` hoặc `ADMISSIONS.csv`

**Các cột bắt buộc:**

| Tên cột | Kiểu dữ liệu | Mô tả |
|---------|--------------|-------|
| `ma_nhap_vien` | Integer/String | Mã nhập viện (PRIMARY KEY) |
| `ma_benh_nhan` | Integer/String | Mã bệnh nhân |
| `ngay_nhap_vien` | Date/Datetime | Ngày nhập viện |
| `ngay_xuat_vien` | Date/Datetime | Ngày xuất viện (tùy chọn) |

**Mục đích:** 
- Liên kết các mã ICD, ATC với cùng một lần nhập viện
- Tạo co-occurrence edges trong Knowledge Graph

---

## 📊 Cấu Trúc Dữ Liệu Tối Thiểu

### Tối thiểu để chạy được:

1. ✅ **Bảng chẩn đoán** với: `ma_nhap_vien`, `ma_icd`
2. ✅ **Bảng kê đơn thuốc** với: `ma_nhap_vien`, `ten_thuoc` (hoặc `ma_atc`)
3. ✅ **Bảng mapping thuốc → ATC** (nếu chỉ có tên thuốc)

### Khuyến nghị để có kết quả tốt:

1. ✅ Tất cả các bảng trên
2. ✅ Bảng từ điển ICD (để xây dựng hierarchy)
3. ✅ Bảng nhập viện (để liên kết dữ liệu)
4. ✅ Dữ liệu xét nghiệm (nếu có)

---

## 🔄 Quy Trình Xử Lý Dữ Liệu Việt Nam

### Bước 1: Chuẩn bị dữ liệu thô

```
1. Xuất dữ liệu từ hệ thống HIS/EHR của bệnh viện
2. Chuẩn hóa tên cột theo format trên
3. Làm sạch dữ liệu (xử lý missing values, duplicates)
4. Mapping thuốc → ATC (nếu chưa có mã ATC)
```

### Bước 2: Xây dựng Knowledge Graph

```bash
# Tạo script tương tự build_kg_mimic.py nhưng cho dữ liệu VN
python KG_EMBED/build_kg_vn.py \
    --data_path ./data_vn \
    --output_dir KG_EMBED/embed_vn \
    --embedding_dim 256
```

**Script này sẽ:**
- Đọc các bảng CSV của Việt Nam
- Xây dựng hierarchy cho ICD codes
- Xây dựng hierarchy cho ATC codes (nếu có)
- Tạo co-occurrence edges từ các lần nhập viện
- Generate embeddings bằng Node2Vec
- Lưu graph và embeddings

### Bước 3: Chuẩn bị BoW matrices

```bash
# Tạo script tương tự prepare_mimic_data.py nhưng cho dữ liệu VN
python prepare_vn_data.py \
    --data_path ./data_vn \
    --kg_embed_dir KG_EMBED/embed_vn \
    --output_dir data_vn \
    --train_ratio 0.8
```

**Script này sẽ:**
- Load vocab từ KG embeddings
- Extract ICD và ATC codes từ các bảng
- Tạo BoW matrices (train/test/test_1/test_2)
- Tạo metadata.txt

### Bước 4: Training

```bash
python main_getm_mimic.py \
    --data_path data_vn \
    --kg_embed_dir KG_EMBED/embed_vn \
    --epochs 50 \
    --num_topics 50
```

---

## ⚠️ Các Vấn Đề Đặc Biệt Cho Việt Nam

### 1. Mã ICD

**Vấn đề:**
- Việt Nam có thể sử dụng ICD-10-VN (phiên bản Việt Nam)
- Một số bệnh viện có thể dùng mã nội bộ

**Giải pháp:**
- Mapping mã nội bộ → ICD-10 chuẩn (nếu có thể)
- Hoặc sử dụng trực tiếp mã nội bộ và xây dựng hierarchy thủ công

### 2. Mapping Thuốc → ATC

**Vấn đề:**
- Nhiều bệnh viện Việt Nam chỉ có tên thuốc (tiếng Việt hoặc tên thương mại)
- Không có mã ATC sẵn

**Giải pháp:**
- Tạo bảng mapping thủ công từ:
  - Danh mục thuốc của Bộ Y Tế Việt Nam
  - WHO ATC Index
  - RxNorm (nếu có)
- Hoặc sử dụng hash-based codes như trong code hiện tại (tạm thời)

### 3. Định dạng ngày tháng

**Vấn đề:**
- Có thể có nhiều format: `dd/mm/yyyy`, `yyyy-mm-dd`, `dd-mm-yyyy`

**Giải pháp:**
- Chuẩn hóa về format `yyyy-mm-dd` hoặc `yyyy/mm/dd` trước khi xử lý

### 4. Encoding và ký tự đặc biệt

**Vấn đề:**
- Tên thuốc/bệnh có thể có tiếng Việt có dấu
- Encoding có thể là UTF-8, Windows-1258, VISCII

**Giải pháp:**
- Đảm bảo tất cả file CSV sử dụng UTF-8 encoding
- Xử lý normalize Unicode nếu cần

### 5. Dữ liệu thiếu

**Vấn đề:**
- Một số bệnh nhân có thể không có chẩn đoán hoặc không có thuốc
- Một số lần nhập viện có thể thiếu thông tin

**Giải pháp:**
- Bỏ qua các records thiếu dữ liệu quan trọng
- Hoặc xử lý missing values tùy theo từng trường hợp

---

## 📝 Template CSV Files

### Template: CHAN_DOAN.csv

```csv
ma_nhap_vien,ma_benh_nhan,ma_icd,ngay_chan_doan
12345,67890,I10,2024-01-15
12345,67890,E11.9,2024-01-15
12346,67891,A00,2024-01-16
```

### Template: KE_DON_THUOC.csv

```csv
ma_nhap_vien,ma_benh_nhan,ten_thuoc,ma_atc,ngay_ke_don
12345,67890,Paracetamol,N02BE01,2024-01-15
12345,67890,Amoxicillin,J01CA04,2024-01-15
12346,67891,Metformin,A10BA02,2024-01-16
```

### Template: DRUG_TO_ATC_MAPPING.csv

```csv
ten_thuoc,ma_atc,nhom_atc
Paracetamol,N02BE01,N02BE
Amoxicillin,J01CA04,J01CA
Metformin,A10BA02,A10BA
```

### Template: NHAP_VIEN.csv

```csv
ma_nhap_vien,ma_benh_nhan,ngay_nhap_vien,ngay_xuat_vien
12345,67890,2024-01-15,2024-01-20
12346,67891,2024-01-16,2024-01-22
```

---

## 🔧 Scripts Cần Tạo Mới

### 1. `KG_EMBED/build_kg_vn.py`

Script này tương tự `build_kg_mimic.py` nhưng:
- Đọc các file CSV của Việt Nam
- Xử lý encoding UTF-8
- Mapping thuốc Việt Nam → ATC
- Xử lý ICD-10-VN hoặc mã nội bộ

### 2. `prepare_vn_data.py`

Script này tương tự `prepare_mimic_data.py` nhưng:
- Đọc các file CSV của Việt Nam
- Xử lý tên cột tiếng Việt
- Mapping thuốc → ATC từ bảng mapping

---

## ✅ Checklist Trước Khi Bắt Đầu

- [ ] Có dữ liệu chẩn đoán với mã ICD (hoặc mã nội bộ có thể mapping)
- [ ] Có dữ liệu thuốc với tên thuốc hoặc mã ATC
- [ ] Có bảng mapping thuốc → ATC (nếu chỉ có tên thuốc)
- [ ] Có mã nhập viện để liên kết các dữ liệu
- [ ] Dữ liệu đã được làm sạch (không có duplicates, missing values quan trọng)
- [ ] File CSV sử dụng UTF-8 encoding
- [ ] Tên cột đã được chuẩn hóa theo format trên

---

## 📚 Tài Liệu Tham Khảo

1. **ICD-10:** https://icd.who.int/browse10/2019/en
2. **WHO ATC Index:** https://www.whocc.no/atc_ddd_index/
3. **ICD-10-VN:** Tài liệu của Bộ Y Tế Việt Nam
4. **Danh mục thuốc Việt Nam:** Tài liệu của Bộ Y Tế Việt Nam

---

## 💡 Gợi Ý Cải Thiện

1. **Sử dụng mã chuẩn quốc tế:**
   - ICD-10 thay vì mã nội bộ
   - ATC codes thay vì chỉ tên thuốc

2. **Xây dựng bảng mapping chất lượng:**
   - Mapping thuốc → ATC từ nguồn chính thống
   - Cập nhật định kỳ khi có thuốc mới

3. **Chuẩn hóa dữ liệu:**
   - Sử dụng tên generic thay vì tên thương mại
   - Chuẩn hóa format ngày tháng
   - Xử lý encoding đúng cách

4. **Bảo mật dữ liệu:**
   - Anonymize mã bệnh nhân trước khi xử lý
   - Tuân thủ quy định về bảo vệ dữ liệu cá nhân

---

---

## 🔗 Khai Thác Liên Kết Bệnh-Thuốc (Disease-Drug Relationships)

Phần này mô tả chi tiết cách khai thác và sử dụng các liên kết bệnh-thuốc từ dữ liệu EHR để cung cấp thông tin hữu ích cho bác sĩ.

---

### 📊 Các Loại Liên Kết Bệnh-Thuốc Có Thể Khai Thác

#### 1. **Co-occurrence Relationships (Đồng Xuất Hiện)**

**Định nghĩa:** Liên kết giữa một mã chẩn đoán (ICD) và một mã thuốc (ATC) khi chúng **cùng xuất hiện** trong cùng một lần nhập viện.

**Cách khai thác:**
```
Với mỗi lần nhập viện (ma_nhap_vien):
  - Lấy tất cả mã ICD từ bảng CHAN_DOAN
  - Lấy tất cả mã ATC từ bảng KE_DON_THUOC
  - Tạo liên kết: ICD_i ↔ ATC_j (với mọi i, j)
  - Đếm tần suất: weight = số lần xuất hiện cùng nhau
```

**Ví dụ:**
```
Lần nhập viện #12345:
  - Chẩn đoán: I10 (Tăng huyết áp), E11.9 (Đái tháo đường type 2)
  - Thuốc: N02BE01 (Paracetamol), A10BA02 (Metformin)
  
→ Tạo các liên kết:
  I10 ↔ N02BE01 (weight=1)
  I10 ↔ A10BA02 (weight=1)
  E11.9 ↔ N02BE01 (weight=1)
  E11.9 ↔ A10BA02 (weight=1)
```

**Ý nghĩa lâm sàng:**
- Cho biết thuốc nào thường được kê cho bệnh nào trong thực tế
- Phản ánh **treatment patterns** (mô hình điều trị) tại bệnh viện
- Có thể khác với guidelines do:
  - Điều kiện thực tế của bệnh nhân
  - Sự sẵn có của thuốc
  - Kinh nghiệm của bác sĩ

---

#### 2. **Temporal Relationships (Quan Hệ Thời Gian)**

**Định nghĩa:** Liên kết dựa trên **thứ tự thời gian** giữa chẩn đoán và kê đơn thuốc.

**Cách khai thác:**
```
Với mỗi lần nhập viện:
  - So sánh ngay_chan_doan và ngay_ke_don
  - Phân loại:
    * Thuốc được kê TRƯỚC chẩn đoán → Preventive/Prophylactic
    * Thuốc được kê SAU chẩn đoán → Treatment
    * Thuốc được kê CÙNG NGÀY → Acute treatment
```

**Ví dụ:**
```
Lần nhập viện #12345:
  - 2024-01-15: Chẩn đoán I10 (Tăng huyết áp)
  - 2024-01-15: Kê đơn C09AA02 (ACE inhibitor)
  
→ Liên kết: I10 → C09AA02 (type='treatment', delay_days=0)
```

**Các loại temporal relationships:**

| Loại | Định nghĩa | Ví dụ |
|------|-------------|-------|
| **Preventive** | Thuốc kê trước khi có chẩn đoán | Aspirin cho bệnh nhân nguy cơ tim mạch |
| **Acute** | Thuốc kê cùng ngày với chẩn đoán | Kháng sinh cho nhiễm trùng cấp |
| **Chronic** | Thuốc kê sau chẩn đoán và tiếp tục | Metformin cho đái tháo đường |
| **Prophylactic** | Thuốc phòng ngừa | Heparin cho bệnh nhân nằm viện |

---

#### 3. **Frequency-Based Relationships (Quan Hệ Dựa Trên Tần Suất)**

**Định nghĩa:** Phân loại liên kết dựa trên **tần suất xuất hiện** trong dữ liệu.

**Cách khai thác:**
```
Với mỗi cặp (ICD, ATC):
  - Đếm số lần xuất hiện cùng nhau: count
  - Tính tỷ lệ: 
    * support = count / total_admissions
    * confidence = count / count(ICD)
  - Phân loại:
    * High-frequency: count >= threshold (ví dụ: 100)
    * Medium-frequency: 10 <= count < 100
    * Low-frequency: 1 <= count < 10
```

**Ví dụ:**
```
ICD I10 (Tăng huyết áp) ↔ ATC C09AA02 (ACE inhibitor):
  - Xuất hiện cùng nhau: 500 lần
  - Tổng số lần nhập viện: 1000
  - Tổng số lần chẩn đoán I10: 600
  
→ support = 500/1000 = 0.5 (50%)
→ confidence = 500/600 = 0.83 (83%)
→ Loại: High-frequency relationship
```

**Ý nghĩa:**
- **High-frequency:** Treatment pattern phổ biến, có thể là guideline
- **Medium-frequency:** Treatment pattern thường gặp
- **Low-frequency:** Treatment pattern hiếm, có thể là:
  - Trường hợp đặc biệt
  - Điều trị thay thế
  - Cần xem xét kỹ

---

#### 4. **Comorbidity-Based Relationships (Quan Hệ Dựa Trên Bệnh Đồng Mắc)**

**Định nghĩa:** Liên kết giữa thuốc và **nhiều bệnh cùng lúc** (comorbidities).

**Cách khai thác:**
```
Với mỗi lần nhập viện có nhiều chẩn đoán:
  - Lấy tất cả mã ICD: {I10, E11.9, I25.9}
  - Lấy tất cả mã ATC: {A10BA02, C09AA02}
  - Tạo liên kết:
    * Single-disease: I10 → A10BA02
    * Multi-disease: {I10, E11.9} → A10BA02
```

**Ví dụ:**
```
Lần nhập viện #12345:
  - Chẩn đoán: I10 (Tăng huyết áp), E11.9 (Đái tháo đường)
  - Thuốc: A10BA02 (Metformin)
  
→ Liên kết:
  Single: E11.9 → A10BA02
  Multi: {I10, E11.9} → A10BA02
```

**Ý nghĩa:**
- Cho biết thuốc nào được dùng khi có nhiều bệnh cùng lúc
- Giúp hiểu về **drug interactions** và **contraindications**
- Hữu ích cho bác sĩ khi điều trị bệnh nhân có nhiều bệnh

---

### 📋 Dữ Liệu Cần Thu Thập Để Khai Thác Liên Kết

#### 1. **Dữ Liệu Tối Thiểu**

| Bảng | Cột Bắt Buộc | Cột Khuyến Nghị |
|------|--------------|-----------------|
| `CHAN_DOAN` | `ma_nhap_vien`, `ma_icd` | `ngay_chan_doan`, `ma_benh_nhan` |
| `KE_DON_THUOC` | `ma_nhap_vien`, `ma_atc` (hoặc `ten_thuoc`) | `ngay_ke_don`, `so_luong`, `don_vi` |
| `NHAP_VIEN` | `ma_nhap_vien`, `ma_benh_nhan` | `ngay_nhap_vien`, `ngay_xuat_vien` |

#### 2. **Dữ Liệu Bổ Sung (Để Khai Thác Tốt Hơn)**

| Bảng | Mục Đích |
|------|----------|
| `THONG_TIN_BENH_NHAN` | Tuổi, giới tính, tiền sử bệnh |
| `XET_NGHIEM` | Kết quả xét nghiệm trước khi kê đơn |
| `TAC_DUNG_PHU` | Tác dụng phụ của thuốc (nếu có) |
| `CHI_DINH_THUOC` | Lý do kê đơn (nếu có) |

---

### 🔍 Các Metrics và Thống Kê Có Thể Cung Cấp

#### 1. **Treatment Frequency Statistics**

**Mục đích:** Cho biết thuốc nào thường được kê cho bệnh nào.

**Metrics:**
```
Với mỗi cặp (ICD, ATC):
  - Frequency: Số lần xuất hiện cùng nhau
  - Support: Tỷ lệ trong tổng số admissions
  - Confidence: Tỷ lệ trong số lần chẩn đoán ICD
  - Lift: Độ mạnh của mối quan hệ (so với ngẫu nhiên)
```

**Ví dụ output:**
```
ICD: I10 (Tăng huyết áp)
  Top 5 thuốc thường được kê:
    1. C09AA02 (ACE inhibitor): frequency=500, confidence=83%, support=50%
    2. C07AB02 (Beta blocker): frequency=300, confidence=50%, support=30%
    3. C08CA01 (Calcium channel blocker): frequency=200, confidence=33%, support=20%
    4. C09AA03 (ARB): frequency=150, confidence=25%, support=15%
    5. C09DA01 (ACE + Diuretic): frequency=100, confidence=17%, support=10%
```

---

#### 2. **Drug-Disease Co-occurrence Matrix**

**Mục đích:** Ma trận cho biết tần suất đồng xuất hiện giữa các bệnh và thuốc.

**Format:**
```
        | ATC_1 | ATC_2 | ATC_3 | ...
ICD_1   |  100  |   50  |   20  | ...
ICD_2   |   30  |  200  |   10  | ...
ICD_3   |   15  |   25  |  150  | ...
```

**Ứng dụng:**
- Tìm thuốc phù hợp cho một bệnh cụ thể
- Phát hiện treatment patterns bất thường
- So sánh với guidelines

---

#### 3. **Temporal Analysis**

**Mục đích:** Phân tích thứ tự thời gian giữa chẩn đoán và kê đơn.

**Metrics:**
```
Với mỗi cặp (ICD, ATC):
  - Average delay: Thời gian trung bình từ chẩn đoán đến kê đơn
  - Median delay: Thời gian trung vị
  - Min/Max delay: Thời gian ngắn nhất/dài nhất
  - Same-day ratio: Tỷ lệ kê đơn cùng ngày với chẩn đoán
```

**Ví dụ:**
```
ICD: I10 (Tăng huyết áp) → ATC: C09AA02 (ACE inhibitor)
  - Average delay: 0.5 ngày
  - Median delay: 0 ngày
  - Same-day ratio: 80%
  - Interpretation: Hầu hết được kê đơn ngay khi chẩn đoán
```

---

#### 4. **Comorbidity Patterns**

**Mục đích:** Phân tích thuốc được dùng khi có nhiều bệnh cùng lúc.

**Metrics:**
```
Với mỗi thuốc (ATC):
  - Single-disease usage: Số lần dùng cho 1 bệnh
  - Multi-disease usage: Số lần dùng cho ≥2 bệnh
  - Common disease pairs: Các cặp bệnh thường đi kèm
```

**Ví dụ:**
```
ATC: A10BA02 (Metformin)
  - Single-disease (E11.9): 300 lần
  - Multi-disease ({I10, E11.9}): 200 lần
  - Common pairs:
    * {I10, E11.9}: 150 lần
    * {E11.9, I25.9}: 50 lần
```

---

### 💡 Ứng Dụng Thực Tế Cho Bác Sĩ

#### 1. **Treatment Recommendation**

**Mục đích:** Gợi ý thuốc phù hợp dựa trên chẩn đoán và dữ liệu lịch sử.

**Cách sử dụng:**
```
Input: Mã ICD (ví dụ: I10 - Tăng huyết áp)
Output: 
  - Top 5 thuốc thường được kê (kèm confidence)
  - Treatment patterns từ các bác sĩ khác
  - So sánh với guidelines
```

**Ví dụ:**
```
Bác sĩ chẩn đoán: I10 (Tăng huyết áp)

Hệ thống gợi ý:
  1. C09AA02 (ACE inhibitor)
     - Được kê trong 83% trường hợp
     - Phù hợp với guidelines
     - Ít tác dụng phụ
  
  2. C07AB02 (Beta blocker)
     - Được kê trong 50% trường hợp
     - Phù hợp cho bệnh nhân có nhịp tim nhanh
  
  3. C08CA01 (Calcium channel blocker)
     - Được kê trong 33% trường hợp
     - Phù hợp cho bệnh nhân lớn tuổi
```

---

#### 2. **Drug-Drug Interaction Warning**

**Mục đích:** Cảnh báo khi kê nhiều thuốc có thể tương tác.

**Cách sử dụng:**
```
Input: Danh sách thuốc đang kê
Output:
  - Các cặp thuốc thường được dùng cùng nhau (an toàn)
  - Các cặp thuốc hiếm khi dùng cùng nhau (cảnh báo)
  - Tác dụng phụ tiềm ẩn
```

**Ví dụ:**
```
Bác sĩ đang kê:
  - C09AA02 (ACE inhibitor)
  - A10BA02 (Metformin)

Hệ thống cảnh báo:
  ⚠️ Cặp thuốc này hiếm khi được dùng cùng nhau (chỉ 5% trường hợp)
  ⚠️ Có thể gây hạ đường huyết
  ✅ Tuy nhiên, an toàn nếu theo dõi đường huyết định kỳ
```

---

#### 3. **Treatment Pattern Analysis**

**Mục đích:** Phân tích và so sánh treatment patterns.

**Cách sử dụng:**
```
Input: Mã ICD hoặc mã ATC
Output:
  - Treatment patterns phổ biến
  - Treatment patterns hiếm gặp
  - So sánh với guidelines
  - Phân tích theo thời gian (trends)
```

**Ví dụ:**
```
Phân tích: I10 (Tăng huyết áp)

Treatment patterns:
  ✅ Phổ biến (theo guidelines):
     - ACE inhibitor: 83%
     - Beta blocker: 50%
  
  ⚠️ Hiếm gặp (cần xem xét):
     - Diuretic đơn độc: 5%
     - Chỉ dùng thuốc giảm đau: 2%
  
  📈 Trends:
     - ACE inhibitor: Tăng từ 70% → 83% (2020-2024)
     - Beta blocker: Giảm từ 60% → 50% (2020-2024)
```

---

#### 4. **Comorbidity Management**

**Mục đích:** Gợi ý điều trị khi bệnh nhân có nhiều bệnh.

**Cách sử dụng:**
```
Input: Danh sách mã ICD (nhiều bệnh)
Output:
  - Thuốc phù hợp cho nhiều bệnh cùng lúc
  - Tránh thuốc có thể làm nặng bệnh khác
  - Ưu tiên thuốc có thể điều trị nhiều bệnh
```

**Ví dụ:**
```
Bệnh nhân có:
  - I10 (Tăng huyết áp)
  - E11.9 (Đái tháo đường type 2)
  - I25.9 (Bệnh tim thiếu máu cục bộ)

Hệ thống gợi ý:
  ✅ Metformin (A10BA02):
     - Điều trị đái tháo đường
     - Có thể giúp giảm huyết áp
     - An toàn cho bệnh nhân tim mạch
  
  ✅ ACE inhibitor (C09AA02):
     - Điều trị tăng huyết áp
     - Bảo vệ thận (quan trọng cho đái tháo đường)
     - Cải thiện chức năng tim
  
  ⚠️ Tránh:
     - Beta blocker không chọn lọc (có thể che dấu triệu chứng hạ đường huyết)
```

---

### 📊 Bảng Thống Kê Mẫu Cho Bác Sĩ

#### Bảng 1: Top Thuốc Cho Mỗi Bệnh

```
Bệnh: I10 (Tăng huyết áp)
┌─────────────────────┬───────────┬────────────┬─────────────┐
│ Thuốc (ATC)         │ Tần suất  │ Confidence │ Support     │
├─────────────────────┼───────────┼────────────┼─────────────┤
│ C09AA02 (ACE)       │    500    │   83.3%    │   50.0%     │
│ C07AB02 (Beta)      │    300    │   50.0%    │   30.0%     │
│ C08CA01 (CCB)       │    200    │   33.3%    │   20.0%     │
│ C09AA03 (ARB)       │    150    │   25.0%    │   15.0%     │
│ C09DA01 (ACE+Diu)   │    100    │   16.7%    │   10.0%     │
└─────────────────────┴───────────┴────────────┴─────────────┘
```

#### Bảng 2: Treatment Patterns Theo Thời Gian

```
Bệnh: I10 (Tăng huyết áp)
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Năm      │ ACE (%)      │ Beta (%)     │ CCB (%)      │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 2020     │    70.0      │    60.0      │    25.0      │
│ 2021     │    75.0      │    55.0      │    28.0      │
│ 2022     │    78.0      │    52.0      │    30.0      │
│ 2023     │    81.0      │    51.0      │    32.0      │
│ 2024     │    83.3      │    50.0      │    33.3      │
└──────────┴──────────────┴──────────────┴──────────────┘
```

#### Bảng 3: Comorbidity Patterns

```
Bệnh chính: E11.9 (Đái tháo đường type 2)
┌─────────────────────┬───────────┬────────────┬─────────────┐
│ Bệnh đồng mắc       │ Tần suất  │ Tỷ lệ (%)  │ Thuốc phổ biến│
├─────────────────────┼───────────┼────────────┼─────────────┤
│ I10 (Tăng huyết áp) │    200    │   66.7%    │ Metformin   │
│ I25.9 (Bệnh tim)    │    100    │   33.3%    │ Metformin   │
│ N18.9 (Bệnh thận)   │     50    │   16.7%    │ Insulin     │
└─────────────────────┴───────────┴────────────┴─────────────┘
```

---

### 🔧 Cách Thu Thập và Lưu Trữ Liên Kết

#### 1. **Tạo Bảng Liên Kết**

**Tên bảng:** `LIEN_KET_BENH_THUOC.csv`

**Cấu trúc:**
```csv
ma_icd,ma_atc,frequency,support,confidence,lift,ngay_bat_dau,ngay_ket_thuc
I10,N02BE01,500,0.5,0.833,1.2,2020-01-01,2024-12-31
I10,C09AA02,300,0.3,0.5,1.5,2020-01-01,2024-12-31
E11.9,A10BA02,400,0.4,0.8,2.0,2020-01-01,2024-12-31
```

**Các cột:**
- `ma_icd`: Mã chẩn đoán
- `ma_atc`: Mã thuốc
- `frequency`: Số lần xuất hiện cùng nhau
- `support`: Tỷ lệ trong tổng số admissions
- `confidence`: Tỷ lệ trong số lần chẩn đoán ICD
- `lift`: Độ mạnh của mối quan hệ
- `ngay_bat_dau`, `ngay_ket_thuc`: Khoảng thời gian phân tích

---

#### 2. **Tạo Bảng Temporal Relationships**

**Tên bảng:** `QUAN_HE_THOI_GIAN.csv`

**Cấu trúc:**
```csv
ma_icd,ma_atc,avg_delay_days,median_delay_days,same_day_ratio,type
I10,C09AA02,0.5,0,0.8,treatment
E11.9,A10BA02,1.0,0,0.6,treatment
I25.9,N02BE01,-2.0,-1,0.3,preventive
```

**Các cột:**
- `avg_delay_days`: Thời gian trung bình từ chẩn đoán đến kê đơn (âm = trước chẩn đoán)
- `median_delay_days`: Thời gian trung vị
- `same_day_ratio`: Tỷ lệ kê đơn cùng ngày
- `type`: Loại quan hệ (treatment, preventive, chronic, etc.)

---

#### 3. **Script Khai Thác Liên Kết**

**Tên file:** `extract_disease_drug_relationships.py`

**Chức năng:**
```python
def extract_disease_drug_relationships():
    """
    Khai thác các loại liên kết bệnh-thuốc từ dữ liệu EHR
    """
    # 1. Load dữ liệu
    diagnoses = load_diagnoses()
    prescriptions = load_prescriptions()
    admissions = load_admissions()
    
    # 2. Khai thác co-occurrence
    cooccurrence = extract_cooccurrence(diagnoses, prescriptions, admissions)
    
    # 3. Khai thác temporal relationships
    temporal = extract_temporal(diagnoses, prescriptions)
    
    # 4. Tính toán metrics
    metrics = calculate_metrics(cooccurrence)
    
    # 5. Lưu kết quả
    save_relationships(cooccurrence, temporal, metrics)
```

---

### 📝 Checklist Cho Bác Sĩ Thu Thập Dữ Liệu

- [ ] **Dữ liệu chẩn đoán:**
  - [ ] Mã ICD chính xác và đầy đủ
  - [ ] Ngày chẩn đoán (để phân tích temporal)
  - [ ] Mã nhập viện để liên kết với thuốc

- [ ] **Dữ liệu thuốc:**
  - [ ] Mã ATC hoặc tên thuốc generic
  - [ ] Ngày kê đơn (để phân tích temporal)
  - [ ] Liên kết với mã nhập viện

- [ ] **Dữ liệu bổ sung:**
  - [ ] Thông tin bệnh nhân (tuổi, giới tính)
  - [ ] Tiền sử bệnh
  - [ ] Kết quả xét nghiệm (nếu có)
  - [ ] Tác dụng phụ (nếu có)

- [ ] **Chất lượng dữ liệu:**
  - [ ] Không có missing values quan trọng
  - [ ] Mã ICD và ATC đúng format
  - [ ] Ngày tháng hợp lệ
  - [ ] Liên kết giữa các bảng chính xác

---

### 🎯 Kết Luận

Việc khai thác liên kết bệnh-thuốc từ dữ liệu EHR thực tế có thể cung cấp:

1. **Treatment patterns** thực tế tại bệnh viện
2. **Gợi ý điều trị** dựa trên dữ liệu lịch sử
3. **Cảnh báo tương tác thuốc** và điều trị không phù hợp
4. **Phân tích trends** và thay đổi trong điều trị
5. **Hỗ trợ quyết định lâm sàng** dựa trên evidence từ dữ liệu

Để khai thác tốt nhất, cần:
- Dữ liệu đầy đủ và chính xác
- Mapping thuốc → ATC đúng
- Liên kết chính xác giữa các bảng
- Phân tích định kỳ để cập nhật patterns

---

**Tác giả:** Generated for GAT-ETM Project  
**Ngày:** 2024  
**Phiên bản:** 1.1 (Updated with Disease-Drug Relationship Extraction)
