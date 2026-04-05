"""
Parse Drug Names from VN EHR Data

Script để extract và normalize tên thuốc từ dữ liệu EHR Việt Nam.

Input:
    - raw_vn_data/df_medicine_sample.xlsx (sample)
    - hoặc data-bv/thuoc.xlsx (full data)

Output:
    - data_vn/unique_drugs.csv - Danh sách thuốc unique
    - data_vn/drug_statistics.csv - Thống kê tần suất thuốc

Usage:
    python scripts/parse_drug_names.py [--full]

    --full: Sử dụng full data từ data-bv/thuoc.xlsx
"""

import pandas as pd
import re
import os
import argparse
from collections import Counter


def parse_drug_text(text):
    """
    Parse chuỗi thuốc thành danh sách tên thuốc.

    Format input: "TenThuoc1 dosage(qty) Unit TenThuoc2 dosage(qty) Unit ..."
    VD: "Vinfoxin 50mg(56) Viên Betahistin 24 24mg(56) Viên"

    Returns:
        List of unique drug names (lowercase, normalized)
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    # Pattern để split - kết thúc bằng (số) + đơn vị đóng gói
    # Bao gồm: Viên, Chai, Ống, Gói, Túi, Hộp, Lọ, Tuýp, Vỉ, Miếng
    # Thêm: bánh, bình, bộ, cái, kít (vật tư y tế)
    unit_pattern = (
        r'\(\d+\)\s*(?:'
        r'Viên|viên|VIÊN|'
        r'Chai[^\(]*?|chai[^\(]*?|CHAI[^\(]*?|'
        r'Ống|ống|ONG|'
        r'Gói|gói|GOI|'
        r'Túi|túi|TUI|'
        r'Hộp|hộp|HOP|'
        r'Lọ[^\(]*?|lọ[^\(]*?|LO|'
        r'Tuýp|tuýp|TUYP|'
        r'Vỉ|vỉ|VI|'
        r'Miếng|miếng|MIENG|'
        r'Bánh|bánh|BANH|'
        r'Bình|bình|BINH|'
        r'Bộ|bộ|BO|'
        r'Cái|cái|CAI|'
        r'Kít|kít|KIT|'
        r'mL|ML|ml'
        r')'
    )

    # Split text theo unit pattern
    parts = re.split(unit_pattern, text, flags=re.IGNORECASE)

    drugs = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 2:
            continue

        # Skip nếu bắt đầu bằng số (like '120 ml')
        if re.match(r'^\d', part):
            continue

        # Loại bỏ dosage ở cuối tên thuốc
        name = remove_dosage(part)

        # Clean up
        name = clean_drug_name(name)

        # Skip các sản phẩm không phải thuốc
        if is_non_drug_product(name):
            continue

        if name and len(name) > 1:
            drugs.append(name)

    # Return unique drugs (preserve first occurrence)
    seen = set()
    unique_drugs = []
    for d in drugs:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            unique_drugs.append(d)

    return unique_drugs


def is_non_drug_product(name):
    """
    Kiểm tra xem sản phẩm có phải là vật tư y tế/mỹ phẩm không phải thuốc.
    """
    name_lower = name.lower()

    # Vật tư y tế
    medical_supplies = [
        'bơm tiêm', 'kim tiêm', 'kim bánh', 'insupen',
        'huyết thanh kháng độc', 'path tezt kit',
    ]

    # Mỹ phẩm / sản phẩm chăm sóc
    cosmetics = [
        'shampoo', 'body wash', 'bubble bath', 'shower',
        'soap', 'cleanser', 'cleansing', 'moisturis',
        'barrier cream', 'wash gel', 'emulsion',
        'hygiene wash', 'skin repair',
    ]

    for item in medical_supplies + cosmetics:
        if item in name_lower:
            return True

    return False


def remove_dosage(text):
    """
    Loại bỏ thông tin dosage từ tên thuốc.

    VD: "Vinfoxin 50mg" -> "Vinfoxin"
    VD: "Calciferat 750mg/200IU 750mg (tương ứng với 300mg Calci) + 200 IU" -> "Calciferat"
    """
    # Pattern 1: số + đơn vị ở cuối
    # Pattern bao gồm: mg, ml, g, mcg, IU, UI, %, số/số
    dosage_pattern = r'\s+\d+[\d\.,/\s]*(?:mg|ml|g|mcg|IU|UI|%|/\d+\w*).*$'
    name = re.sub(dosage_pattern, '', text, flags=re.IGNORECASE)

    # Pattern 2: (tương ứng với...) và các mô tả trong ngoặc
    name = re.sub(r'\s*\((?:tương ứng|chứa|bao gồm)[^)]*\)', '', name, flags=re.IGNORECASE)

    return name.strip()


def clean_drug_name(name):
    """
    Normalize và clean tên thuốc.
    """
    # Loại bỏ dấu ; hoặc , ở cuối
    name = re.sub(r'[;,]\s*$', '', name)

    # Loại bỏ khoảng trắng thừa
    name = re.sub(r'\s+', ' ', name)

    # Loại bỏ + ở cuối (từ pattern kết hợp thuốc)
    name = re.sub(r'\s*\+\s*$', '', name)

    # Loại bỏ ngoặc mở không đóng
    name = re.sub(r'\s*\(\s*$', '', name)

    return name.strip()


def normalize_drug_name(name):
    """
    Chuẩn hóa tên thuốc để so sánh và matching.

    - Lowercase
    - Loại bỏ ký tự đặc biệt
    - Giữ nguyên số (có thể là phần tên thương mại)
    """
    # Lowercase
    normalized = name.lower()

    # Loại bỏ dấu gạch ngang kép
    normalized = re.sub(r'-+', '-', normalized)

    # Loại bỏ khoảng trắng thừa
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized.strip()


def extract_generic_name(brand_name):
    """
    Thử extract tên generic từ tên thương mại.

    Một số pattern phổ biến:
    - "Venlafaxine STELLA 75mg" -> "Venlafaxine"
    - "Betahistin 24" -> "Betahistin"
    """
    # Loại bỏ số ở cuối (strength indicator)
    name = re.sub(r'\s+\d+$', '', brand_name)

    # Loại bỏ các brand suffix phổ biến
    brand_suffixes = ['STELLA', 'DHG', 'STADA', 'TV', 'PHARMA', 'PHARM', 'PLUS', 'FORTE',
                      'CAP', 'TAB', 'DT', 'DR', 'SR', 'XR', 'ER', 'CR', 'OD']
    for suffix in brand_suffixes:
        name = re.sub(rf'\s+{suffix}$', '', name, flags=re.IGNORECASE)

    return name.strip()


def process_drug_data(df, thuoc_column='Thuoc'):
    """
    Xử lý toàn bộ DataFrame thuốc.

    Returns:
        - all_drugs: List tất cả thuốc parsed (có thể trùng)
        - unique_drugs: Set thuốc unique
        - drug_counts: Counter tần suất mỗi thuốc
    """
    all_drugs = []
    drug_counts = Counter()

    total_rows = len(df)
    rows_with_drugs = 0

    for idx, row in df.iterrows():
        drugs = parse_drug_text(row.get(thuoc_column))
        if drugs:
            rows_with_drugs += 1
            all_drugs.extend(drugs)
            drug_counts.update([normalize_drug_name(d) for d in drugs])

    unique_drugs = set(normalize_drug_name(d) for d in all_drugs)

    print(f"Tổng số dòng: {total_rows}")
    print(f"Dòng có thuốc: {rows_with_drugs} ({100*rows_with_drugs/total_rows:.1f}%)")
    print(f"Tổng số thuốc parsed: {len(all_drugs)}")
    print(f"Số thuốc unique: {len(unique_drugs)}")

    return all_drugs, unique_drugs, drug_counts


def save_results(unique_drugs, drug_counts, output_dir='data_vn'):
    """
    Lưu kết quả ra file CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save unique drugs list
    unique_list = sorted(unique_drugs)
    df_unique = pd.DataFrame({
        'drug_name': unique_list,
        'generic_name_guess': [extract_generic_name(d) for d in unique_list],
        'atc_code': [''] * len(unique_list),  # Để trống, cần mapping sau
        'count': [drug_counts.get(d, 0) for d in unique_list]
    })

    unique_file = os.path.join(output_dir, 'unique_drugs.csv')
    df_unique.to_csv(unique_file, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu danh sách thuốc unique: {unique_file}")

    # 2. Save drug statistics (sorted by frequency)
    df_stats = pd.DataFrame([
        {'drug_name': drug, 'count': count}
        for drug, count in drug_counts.most_common()
    ])

    stats_file = os.path.join(output_dir, 'drug_statistics.csv')
    df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"Đã lưu thống kê tần suất: {stats_file}")

    # 3. Print top drugs
    print(f"\n=== TOP 20 THUỐC PHỔ BIẾN ===")
    for drug, count in drug_counts.most_common(20):
        print(f"  {count:5d}x  {drug}")

    return df_unique, df_stats


def main():
    parser = argparse.ArgumentParser(description='Parse drug names from VN EHR data')
    parser.add_argument('--full', action='store_true',
                        help='Use full data from data-bv/thuoc.xlsx')
    parser.add_argument('--input', type=str, default=None,
                        help='Custom input file path')
    parser.add_argument('--output', type=str, default='data_vn',
                        help='Output directory')
    args = parser.parse_args()

    # Determine input file
    if args.input:
        input_file = args.input
    elif args.full:
        input_file = '../data-bv/thuoc.xlsx'
    else:
        input_file = 'raw_vn_data/df_medicine_sample.xlsx'

    print(f"=== PARSE DRUG NAMES ===")
    print(f"Input: {input_file}")
    print(f"Output: {args.output}/")
    print()

    # Check file exists
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return

    # Read data
    print("Đang đọc dữ liệu...")
    df = pd.read_excel(input_file)
    print(f"Đã đọc {len(df)} dòng")
    print()

    # Process
    print("Đang parse tên thuốc...")
    all_drugs, unique_drugs, drug_counts = process_drug_data(df)

    # Save
    save_results(unique_drugs, drug_counts, args.output)

    print("\n=== HOÀN TẤT ===")


if __name__ == '__main__':
    main()
