"""
Data Verification Script for GAT-ETM

Tạo báo cáo verification để kiểm tra data trước khi training.

Output:
    - data_vn/verification_report.txt - Text report
    - data_vn/verification/ - Visualizations và CSV files cho manual review

Usage:
    python scripts/verify_data.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_data():
    """Load all necessary data files."""
    print("Loading data files...")

    data = {}

    # BoW matrices
    data['bow_train'] = np.load('data_vn/bow_train.npy', allow_pickle=True).item()
    data['bow_test'] = np.load('data_vn/bow_test.npy', allow_pickle=True).item()

    # Vocab info
    with open('data_vn/vocab_info.pkl', 'rb') as f:
        data['vocab_info'] = pickle.load(f)

    # Drug mapping
    data['drug_mapping'] = pd.read_csv('data_vn/drug_atc_mapping.csv')

    # Knowledge graph
    with open('embed_vn/knowledge_graph.pkl', 'rb') as f:
        data['kg'] = pickle.load(f)

    # Graph by vocab
    with open('embed_vn/graph_by_vocab.pkl', 'rb') as f:
        data['graph_vocab'] = pickle.load(f)

    # Embeddings
    data['icd_emb'] = np.load('embed_vn/icd_embeddings.npy')
    data['atc_emb'] = np.load('embed_vn/atc_embeddings.npy')

    return data


def verify_bow_matrices(data, report_lines):
    """Verify BoW matrices statistics."""
    report_lines.append("\n" + "="*60)
    report_lines.append("1. BOW MATRICES VERIFICATION")
    report_lines.append("="*60)

    bow_train = data['bow_train']
    bow_test = data['bow_test']
    vocab_info = data['vocab_info']

    V_icd = len(vocab_info['icd_vocab'])
    V_atc = len(vocab_info['atc_vocab'])

    # Basic stats
    report_lines.append(f"\nTraining set:")
    report_lines.append(f"  Shape: {bow_train.shape}")
    report_lines.append(f"  Non-zero entries: {bow_train.nnz}")
    report_lines.append(f"  Density: {100*bow_train.nnz/(bow_train.shape[0]*bow_train.shape[1]):.4f}%")

    report_lines.append(f"\nTest set:")
    report_lines.append(f"  Shape: {bow_test.shape}")
    report_lines.append(f"  Non-zero entries: {bow_test.nnz}")
    report_lines.append(f"  Density: {100*bow_test.nnz/(bow_test.shape[0]*bow_test.shape[1]):.4f}%")

    # Codes per patient
    train_codes_per_patient = np.array(bow_train.sum(axis=1)).flatten()
    test_codes_per_patient = np.array(bow_test.sum(axis=1)).flatten()

    report_lines.append(f"\nCodes per patient (train):")
    report_lines.append(f"  Min: {train_codes_per_patient.min():.0f}")
    report_lines.append(f"  Max: {train_codes_per_patient.max():.0f}")
    report_lines.append(f"  Mean: {train_codes_per_patient.mean():.2f}")
    report_lines.append(f"  Median: {np.median(train_codes_per_patient):.0f}")

    # ICD vs ATC distribution
    train_icd = bow_train[:, :V_icd]
    train_atc = bow_train[:, V_icd:]

    icd_per_patient = np.array(train_icd.sum(axis=1)).flatten()
    atc_per_patient = np.array(train_atc.sum(axis=1)).flatten()

    report_lines.append(f"\nICD codes per patient (train):")
    report_lines.append(f"  Mean: {icd_per_patient.mean():.2f}, Median: {np.median(icd_per_patient):.0f}")

    report_lines.append(f"\nATC codes per patient (train):")
    report_lines.append(f"  Mean: {atc_per_patient.mean():.2f}, Median: {np.median(atc_per_patient):.0f}")

    # Check for empty rows
    empty_rows = np.sum(train_codes_per_patient == 0)
    report_lines.append(f"\n⚠️ Empty rows in training: {empty_rows}")

    return train_codes_per_patient, icd_per_patient, atc_per_patient


def verify_vocabulary(data, report_lines):
    """Verify vocabulary coverage and validity."""
    report_lines.append("\n" + "="*60)
    report_lines.append("2. VOCABULARY VERIFICATION")
    report_lines.append("="*60)

    vocab_info = data['vocab_info']
    icd_vocab = vocab_info['icd_vocab']
    atc_vocab = vocab_info['atc_vocab']

    report_lines.append(f"\nVocabulary sizes:")
    report_lines.append(f"  ICD codes: {len(icd_vocab)}")
    report_lines.append(f"  ATC codes: {len(atc_vocab)}")
    report_lines.append(f"  Total: {len(icd_vocab) + len(atc_vocab)}")

    # ICD code format check
    import re
    valid_icd = [c for c in icd_vocab if re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', c)]
    report_lines.append(f"\nICD format validation:")
    report_lines.append(f"  Valid format: {len(valid_icd)}/{len(icd_vocab)}")

    invalid_icd = [c for c in icd_vocab if c not in valid_icd]
    if invalid_icd[:5]:
        report_lines.append(f"  Sample invalid: {invalid_icd[:5]}")

    # ATC code format check
    valid_atc = [c for c in atc_vocab if re.match(r'^[A-Z]\d{2}[A-Z]{0,2}\d{0,2}$', c)]
    report_lines.append(f"\nATC format validation:")
    report_lines.append(f"  Valid format: {len(valid_atc)}/{len(atc_vocab)}")

    # ICD category distribution
    icd_categories = Counter([c[0] for c in icd_vocab])
    report_lines.append(f"\nICD categories distribution:")
    for cat, count in sorted(icd_categories.items()):
        report_lines.append(f"  {cat}: {count}")

    # ATC level 1 distribution
    atc_categories = Counter([c[0] for c in atc_vocab])
    report_lines.append(f"\nATC level 1 distribution:")
    for cat, count in sorted(atc_categories.items()):
        report_lines.append(f"  {cat}: {count}")

    return icd_vocab, atc_vocab


def verify_drug_mapping(data, report_lines, output_dir):
    """Verify drug to ATC mapping - FOR EXPERT REVIEW."""
    report_lines.append("\n" + "="*60)
    report_lines.append("3. DRUG-ATC MAPPING (CẦN CHUYÊN GIA Y TẾ KIỂM TRA)")
    report_lines.append("="*60)

    df = data['drug_mapping']
    mapped = df[df['atc_code'].notna() & (df['atc_code'] != '')]

    report_lines.append(f"\nMapping statistics:")
    report_lines.append(f"  Total drugs: {len(df)}")
    report_lines.append(f"  Mapped: {len(mapped)} ({100*len(mapped)/len(df):.1f}%)")

    # Method distribution
    report_lines.append(f"\nBy mapping method:")
    for method, count in mapped['method'].value_counts().items():
        report_lines.append(f"  {method}: {count}")

    # Export TOP 50 mapped drugs for expert review
    top_mapped = mapped.nlargest(50, 'count')[['drug_name', 'atc_code', 'generic_name', 'count', 'confidence', 'method']]
    top_mapped.to_csv(f'{output_dir}/expert_review_drug_mapping.csv', index=False, encoding='utf-8-sig')

    report_lines.append(f"\n📋 FILE CHO CHUYÊN GIA: {output_dir}/expert_review_drug_mapping.csv")
    report_lines.append("   → Kiểm tra 50 thuốc phổ biến nhất, xác nhận ATC code đúng")

    # Export fuzzy matches for review (potentially wrong)
    fuzzy = mapped[mapped['method'] == 'fuzzy_match'].nlargest(30, 'count')
    fuzzy[['drug_name', 'atc_code', 'generic_name', 'count', 'confidence']].to_csv(
        f'{output_dir}/expert_review_fuzzy_matches.csv', index=False, encoding='utf-8-sig'
    )
    report_lines.append(f"📋 FILE CHO CHUYÊN GIA: {output_dir}/expert_review_fuzzy_matches.csv")
    report_lines.append("   → Kiểm tra fuzzy matches (có thể sai)")

    return mapped


def extract_disease_drug_pairs(data, report_lines, output_dir):
    """Extract and verify disease-drug co-occurrences - FOR EXPERT REVIEW."""
    report_lines.append("\n" + "="*60)
    report_lines.append("4. DISEASE-DRUG PAIRS (CẦN CHUYÊN GIA Y TẾ KIỂM TRA)")
    report_lines.append("="*60)

    bow_train = data['bow_train']
    vocab_info = data['vocab_info']
    icd_vocab = vocab_info['icd_vocab']
    atc_vocab = vocab_info['atc_vocab']
    V_icd = len(icd_vocab)

    # Count co-occurrences
    report_lines.append("\nĐang tính toán disease-drug co-occurrences...")

    # Convert to dense for easier computation (may be slow for large data)
    # Only sample if too large
    if bow_train.shape[0] > 10000:
        sample_idx = np.random.choice(bow_train.shape[0], 10000, replace=False)
        bow_sample = bow_train[sample_idx].toarray()
    else:
        bow_sample = bow_train.toarray()

    icd_bow = bow_sample[:, :V_icd]
    atc_bow = bow_sample[:, V_icd:]

    # Co-occurrence matrix
    cooccur = icd_bow.T @ atc_bow  # (V_icd, V_atc)

    # Get top pairs
    pairs = []
    for i in range(cooccur.shape[0]):
        for j in range(cooccur.shape[1]):
            if cooccur[i, j] > 0:
                pairs.append({
                    'icd_code': icd_vocab[i],
                    'atc_code': atc_vocab[j],
                    'count': int(cooccur[i, j])
                })

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('count', ascending=False)

    report_lines.append(f"\nTotal disease-drug pairs: {len(pairs_df)}")
    report_lines.append(f"Pairs with count >= 100: {len(pairs_df[pairs_df['count'] >= 100])}")

    # Add ICD descriptions (common ones)
    icd_descriptions = {
        'I10': 'Tăng huyết áp vô căn',
        'E78.2': 'Tăng lipid máu hỗn hợp',
        'E11.9': 'ĐTĐ type 2, không biến chứng',
        'K21': 'Trào ngược dạ dày-thực quản',
        'I25.0': 'Bệnh tim thiếu máu xơ vữa',
        'J20': 'Viêm phế quản cấp',
        'K29': 'Viêm dạ dày-tá tràng',
        'M54.2': 'Đau cột sống cổ',
        'H81': 'Rối loạn chức năng tiền đình',
        'F41': 'Rối loạn lo âu',
    }

    # Add ATC descriptions (common ones)
    atc_descriptions = {
        'C10AA05': 'Atorvastatin (hạ lipid)',
        'C10AA07': 'Rosuvastatin (hạ lipid)',
        'B01AC06': 'Aspirin (chống kết tập TC)',
        'B01AC04': 'Clopidogrel (chống kết tập TC)',
        'A02BC05': 'Esomeprazole (PPI)',
        'A02BC02': 'Pantoprazole (PPI)',
        'C08CA01': 'Amlodipine (CCB)',
        'C09CA01': 'Losartan (ARB)',
        'A10BA02': 'Metformin (ĐTĐ)',
        'N02BE01': 'Paracetamol (giảm đau)',
    }

    pairs_df['icd_desc'] = pairs_df['icd_code'].map(icd_descriptions).fillna('')
    pairs_df['atc_desc'] = pairs_df['atc_code'].map(atc_descriptions).fillna('')

    # Export top 100 pairs for expert review
    top_pairs = pairs_df.head(100)
    top_pairs.to_csv(f'{output_dir}/expert_review_disease_drug_pairs.csv', index=False, encoding='utf-8-sig')

    report_lines.append(f"\n📋 FILE CHO CHUYÊN GIA: {output_dir}/expert_review_disease_drug_pairs.csv")
    report_lines.append("   → Kiểm tra 100 cặp bệnh-thuốc phổ biến nhất")
    report_lines.append("   → Xác nhận các cặp này có clinically meaningful không")

    # Print top 20 in report
    report_lines.append("\nTOP 20 DISEASE-DRUG PAIRS:")
    report_lines.append("-" * 70)
    for _, row in top_pairs.head(20).iterrows():
        icd_desc = f" ({row['icd_desc']})" if row['icd_desc'] else ""
        atc_desc = f" ({row['atc_desc']})" if row['atc_desc'] else ""
        report_lines.append(f"  {row['icd_code']}{icd_desc} - {row['atc_code']}{atc_desc}: {row['count']}")

    return pairs_df


def extract_sample_patients(data, report_lines, output_dir):
    """Extract sample patient records - FOR EXPERT REVIEW."""
    report_lines.append("\n" + "="*60)
    report_lines.append("5. SAMPLE PATIENT RECORDS (CẦN CHUYÊN GIA Y TẾ KIỂM TRA)")
    report_lines.append("="*60)

    bow_train = data['bow_train']
    vocab_info = data['vocab_info']
    icd_vocab = vocab_info['icd_vocab']
    atc_vocab = vocab_info['atc_vocab']
    V_icd = len(icd_vocab)

    # Sample 20 patients with varying code counts
    n_codes = np.array(bow_train.sum(axis=1)).flatten()

    # Get patients with different numbers of codes
    samples = []
    for target in [3, 5, 7, 10, 15]:
        candidates = np.where((n_codes >= target-1) & (n_codes <= target+1))[0]
        if len(candidates) > 0:
            idx = np.random.choice(candidates, min(4, len(candidates)), replace=False)
            samples.extend(idx)

    # Extract patient records
    patient_records = []
    for i, idx in enumerate(samples[:20]):
        row = bow_train.getrow(idx).toarray().flatten()
        icd_codes = [icd_vocab[j] for j in range(V_icd) if row[j] > 0]
        atc_codes = [atc_vocab[j] for j in range(len(atc_vocab)) if row[V_icd + j] > 0]

        patient_records.append({
            'patient_id': i + 1,
            'n_icd': len(icd_codes),
            'n_atc': len(atc_codes),
            'icd_codes': '; '.join(icd_codes),
            'atc_codes': '; '.join(atc_codes)
        })

    df_patients = pd.DataFrame(patient_records)
    df_patients.to_csv(f'{output_dir}/expert_review_sample_patients.csv', index=False, encoding='utf-8-sig')

    report_lines.append(f"\n📋 FILE CHO CHUYÊN GIA: {output_dir}/expert_review_sample_patients.csv")
    report_lines.append("   → Kiểm tra 20 bệnh nhân mẫu")
    report_lines.append("   → Xác nhận ICD codes và ATC codes có hợp lý cho cùng bệnh nhân")

    # Print sample in report
    report_lines.append("\nSAMPLE PATIENTS:")
    report_lines.append("-" * 70)
    for _, row in df_patients.head(5).iterrows():
        report_lines.append(f"\nPatient {row['patient_id']}:")
        report_lines.append(f"  ICD ({row['n_icd']}): {row['icd_codes']}")
        report_lines.append(f"  ATC ({row['n_atc']}): {row['atc_codes']}")

    return df_patients


def verify_graph(data, report_lines, output_dir):
    """Verify graph structure."""
    report_lines.append("\n" + "="*60)
    report_lines.append("6. GRAPH STRUCTURE VERIFICATION")
    report_lines.append("="*60)

    import networkx as nx

    graph = data['graph_vocab']

    report_lines.append(f"\nGraph statistics:")
    report_lines.append(f"  Nodes: {graph.number_of_nodes()}")
    report_lines.append(f"  Edges: {graph.number_of_edges()}")
    report_lines.append(f"  Connected: {nx.is_connected(graph)}")

    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        report_lines.append(f"  ⚠️ Number of components: {len(components)}")

    # Degree distribution
    degrees = [d for n, d in graph.degree()]
    report_lines.append(f"\nDegree distribution:")
    report_lines.append(f"  Min: {min(degrees)}")
    report_lines.append(f"  Max: {max(degrees)}")
    report_lines.append(f"  Mean: {np.mean(degrees):.2f}")
    report_lines.append(f"  Median: {np.median(degrees):.0f}")

    # Nodes with degree 0 or 1
    low_degree = sum(1 for d in degrees if d <= 1)
    report_lines.append(f"  Nodes with degree ≤ 1: {low_degree}")

    return graph


def create_visualizations(data, output_dir, train_codes, icd_codes, atc_codes):
    """Create visualization plots."""
    print("Creating visualizations...")

    # 1. Codes per patient distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(train_codes, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Total codes per patient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Total Codes per Patient')

    axes[1].hist(icd_codes, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('ICD codes per patient')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of ICD Codes per Patient')

    axes[2].hist(atc_codes, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[2].set_xlabel('ATC codes per patient')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of ATC Codes per Patient')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/viz_codes_distribution.png', dpi=150)
    plt.close()

    # 2. Code frequency distribution
    bow_train = data['bow_train']
    code_freq = np.array(bow_train.sum(axis=0)).flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(code_freq, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Code frequency (number of patients)')
    ax.set_ylabel('Number of codes')
    ax.set_title('Distribution of Code Frequencies')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/viz_code_frequency.png', dpi=150)
    plt.close()

    # 3. t-SNE of embeddings (sample)
    try:
        from sklearn.manifold import TSNE

        icd_emb = data['icd_emb']
        atc_emb = data['atc_emb']

        # Sample if too large
        n_sample = min(500, len(icd_emb))
        icd_sample = icd_emb[np.random.choice(len(icd_emb), n_sample, replace=False)]

        n_sample_atc = min(300, len(atc_emb))
        atc_sample = atc_emb[np.random.choice(len(atc_emb), n_sample_atc, replace=False)]

        combined = np.vstack([icd_sample, atc_sample])
        labels = ['ICD'] * len(icd_sample) + ['ATC'] * len(atc_sample)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))

        icd_mask = np.array(labels) == 'ICD'
        atc_mask = np.array(labels) == 'ATC'

        ax.scatter(embedded[icd_mask, 0], embedded[icd_mask, 1],
                   c='blue', alpha=0.5, label='ICD codes', s=20)
        ax.scatter(embedded[atc_mask, 0], embedded[atc_mask, 1],
                   c='red', alpha=0.5, label='ATC codes', s=20)

        ax.set_title('t-SNE Visualization of Code Embeddings')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/viz_embeddings_tsne.png', dpi=150)
        plt.close()

    except ImportError:
        print("  sklearn not available, skipping t-SNE visualization")


def main():
    print("="*60)
    print("DATA VERIFICATION FOR GAT-ETM")
    print("="*60)
    print()

    # Create output directory
    output_dir = 'data_vn/verification'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data()

    # Initialize report
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("DATA VERIFICATION REPORT FOR GAT-ETM")
    report_lines.append("="*60)
    report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")

    # Run verifications
    train_codes, icd_codes, atc_codes = verify_bow_matrices(data, report_lines)
    verify_vocabulary(data, report_lines)
    verify_drug_mapping(data, report_lines, output_dir)
    extract_disease_drug_pairs(data, report_lines, output_dir)
    extract_sample_patients(data, report_lines, output_dir)
    verify_graph(data, report_lines, output_dir)

    # Summary for expert review
    report_lines.append("\n" + "="*60)
    report_lines.append("SUMMARY: FILES CẦN CHUYÊN GIA Y TẾ KIỂM TRA")
    report_lines.append("="*60)
    report_lines.append(f"""
1. {output_dir}/expert_review_drug_mapping.csv
   → 50 thuốc phổ biến nhất và ATC mapping
   → Kiểm tra: Tên thuốc VN có đúng với mã ATC không?

2. {output_dir}/expert_review_fuzzy_matches.csv
   → Các thuốc được mapping bằng fuzzy matching (có thể sai)
   → Kiểm tra: Fuzzy match có đúng không? Cần sửa không?

3. {output_dir}/expert_review_disease_drug_pairs.csv
   → 100 cặp bệnh-thuốc phổ biến nhất
   → Kiểm tra: Các cặp ICD-ATC này có hợp lý về mặt lâm sàng không?

4. {output_dir}/expert_review_sample_patients.csv
   → 20 bệnh nhân mẫu với ICD và ATC codes
   → Kiểm tra: Mỗi bệnh nhân có ICD codes và thuốc phù hợp không?
""")

    # Create visualizations
    create_visualizations(data, output_dir, train_codes, icd_codes, atc_codes)

    report_lines.append("\n" + "="*60)
    report_lines.append("VISUALIZATIONS")
    report_lines.append("="*60)
    report_lines.append(f"""
1. {output_dir}/viz_codes_distribution.png
   → Phân bố số codes mỗi bệnh nhân

2. {output_dir}/viz_code_frequency.png
   → Phân bố tần suất các codes

3. {output_dir}/viz_embeddings_tsne.png
   → t-SNE visualization của embeddings
""")

    # Save report
    report_text = '\n'.join(report_lines)
    with open(f'{output_dir}/verification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\nReport saved to: {output_dir}/verification_report.txt")
    print(f"Expert review files saved to: {output_dir}/")


if __name__ == '__main__':
    main()
