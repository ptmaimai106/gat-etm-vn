# Cải thiện Knowledge Graph để thể hiện "Tri thức"

## Vấn đề ban đầu

Sau khi build KG và visualize, thống kê cho thấy:
- **Total nodes**: 255
- **Total edges**: 3213
- **Node types**: ATC: 60, CPT: 4, ICD9: 2, LAB: 137, UNKNOWN: 52
- **Edge types**: cooccurrence: 2982 (chiếm 93% tổng số edges)

**Vấn đề chính**:
1. ICD9 chỉ có 2 nodes → **Không có hierarchy rõ ràng**
2. Chủ yếu là co-occurrence edges → **Thiếu hierarchical edges** (tri thức cấu trúc)
3. UNKNOWN nodes chiếm 52 nodes → Không rõ ràng về loại

## Các cải thiện đã thực hiện

### 1. Tăng số lượng subjects (Line 129-137)
- **Trước**: Chỉ lấy 1 subject
- **Sau**: Lấy 5 subjects (`NUM_SAMPLE_SUBJECTS = 5`)
- **Lý do**: Cần đủ ICD9 codes để xây dựng hierarchy đầy đủ

### 2. Cải thiện ICD9 hierarchy (Line 185-250)
- **Trước**: Chỉ lấy 1 ICD9 node → Không có hierarchy
- **Sau**: 
  - Lấy 15 leaf nodes (5-digit codes)
  - Tự động thêm tất cả parent nodes (3-digit, 4-digit)
  - Đảm bảo hierarchy đầy đủ: `ICD9_ROOT → 3-digit → 4-digit → 5-digit`
  - Rebuild edges đúng với hierarchy

### 3. Tăng số admissions được xử lý (Line 466-467)
- **Trước**: Chỉ xử lý 1 admission
- **Sau**: Xử lý 10 admissions (`MAX_ADMISSIONS = 10`)
- **Lý do**: Cần đủ co-occurrence patterns nhưng vẫn giữ KG nhỏ

### 4. Thêm logging chi tiết
- Log số lượng nodes theo level (3-digit, 4-digit, 5-digit)
- Log số lượng edges theo type sau mỗi bước
- Giúp kiểm tra và debug dễ dàng hơn

## Kết quả mong đợi

Sau khi rebuild với các cải thiện trên, KG sẽ có:

1. **Hierarchical edges rõ ràng**:
   - ICD9 hierarchy: `ICD9_ROOT → 3-digit → 4-digit → 5-digit`
   - ATC hierarchy: `ATC_ROOT → Level → Drug code`
   - CPT hierarchy: `CPT_ROOT → Section → Code`
   - LAB hierarchy: `LAB_ROOT → Category → Itemid`

2. **Cân bằng giữa edge types**:
   - Hierarchical edges: ~50-100 edges (thể hiện tri thức cấu trúc)
   - Co-occurrence edges: ~2000-3000 edges (thể hiện quan hệ đồng xuất hiện)
   - Augmented edges: ~50-100 edges (skip connections)

3. **ICD9 nodes đầy đủ**:
   - Khoảng 20-30 ICD9 nodes (bao gồm root, 3-digit, 4-digit, 5-digit)
   - Hierarchy rõ ràng với nhiều level

## Cách điều chỉnh thêm (nếu cần)

### Nếu muốn KG nhỏ hơn:
```python
MAX_ICD9_LEAF_NODES = 10  # Giảm từ 15 xuống 10
NUM_SAMPLE_SUBJECTS = 3   # Giảm từ 5 xuống 3
MAX_ADMISSIONS = 5        # Giảm từ 10 xuống 5
```

### Nếu muốn KG lớn hơn (nhưng vẫn "sample"):
```python
MAX_ICD9_LEAF_NODES = 30  # Tăng từ 15 lên 30
NUM_SAMPLE_SUBJECTS = 10  # Tăng từ 5 lên 10
MAX_ADMISSIONS = 20       # Tăng từ 10 lên 20
```

## Các bước tiếp theo để kiểm tra

1. **Rebuild KG**:
   ```bash
   python KG_EMBED/build_kg_mimic_sample.py --output_dir embed_simple_2
   ```

2. **Visualize và kiểm tra**:
   - Kiểm tra số lượng hierarchical edges
   - Kiểm tra ICD9 hierarchy có đầy đủ các level không
   - Kiểm tra tỷ lệ giữa hierarchical và co-occurrence edges

3. **Nếu vẫn chưa đủ**:
   - Tăng `MAX_ICD9_LEAF_NODES`
   - Tăng `NUM_SAMPLE_SUBJECTS`
   - Kiểm tra xem có đủ ICD9 codes trong data không

## Lưu ý

- Các tham số có thể điều chỉnh dễ dàng ở đầu các hàm
- Logging chi tiết giúp debug và hiểu rõ quá trình build
- Đảm bảo hierarchy được build đúng trước khi thêm co-occurrence edges

