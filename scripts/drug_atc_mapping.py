"""
Drug Name to ATC Code Mapping

Script để mapping tên thuốc thương mại Việt Nam sang mã ATC quốc tế.

Strategies:
1. Direct keyword matching: Tìm generic name trong tên thuốc VN
2. Fuzzy matching: So sánh với ATC database bằng fuzzy string matching
3. Manual mapping dictionary: Mapping thủ công cho các thuốc phổ biến

Input:
    - data_vn/unique_drugs.csv - Danh sách thuốc VN
    - data_vn/atc_reference/who_atc_ddd.csv - WHO ATC database

Output:
    - data_vn/drug_atc_mapping.csv - Kết quả mapping
    - data_vn/unmapped_drugs.csv - Thuốc không map được

Usage:
    python scripts/drug_atc_mapping.py
"""

import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict
import os


# ============================================
# MANUAL MAPPING DICTIONARY
# Mapping từ keyword trong tên thuốc VN → (generic_name, ATC_code)
# ============================================

MANUAL_DRUG_MAPPING = {
    # === Cardiovascular (C) ===
    'atorvastatin': ('atorvastatin', 'C10AA05'),
    'rosuvastatin': ('rosuvastatin', 'C10AA07'),
    'simvastatin': ('simvastatin', 'C10AA01'),
    'pravastatin': ('pravastatin', 'C10AA03'),
    'lovastatin': ('lovastatin', 'C10AA02'),
    'amlodipine': ('amlodipine', 'C08CA01'),
    'losartan': ('losartan', 'C09CA01'),
    'valsartan': ('valsartan', 'C09CA03'),
    'irbesartan': ('irbesartan', 'C09CA04'),
    'telmisartan': ('telmisartan', 'C09CA07'),
    'candesartan': ('candesartan', 'C09CA06'),
    'lisinopril': ('lisinopril', 'C09AA03'),
    'enalapril': ('enalapril', 'C09AA02'),
    'perindopril': ('perindopril', 'C09AA04'),
    'ramipril': ('ramipril', 'C09AA05'),
    'captopril': ('captopril', 'C09AA01'),
    'metoprolol': ('metoprolol', 'C07AB02'),
    'bisoprolol': ('bisoprolol', 'C07AB07'),
    'carvedilol': ('carvedilol', 'C07AG02'),
    'atenolol': ('atenolol', 'C07AB03'),
    'propranolol': ('propranolol', 'C07AA05'),
    'nebivolol': ('nebivolol', 'C07AB12'),
    'clopidogrel': ('clopidogrel', 'B01AC04'),
    'aspirin': ('acetylsalicylic acid', 'B01AC06'),
    'warfarin': ('warfarin', 'B01AA03'),
    'rivaroxaban': ('rivaroxaban', 'B01AF01'),
    'apixaban': ('apixaban', 'B01AF02'),
    'dabigatran': ('dabigatran etexilate', 'B01AE07'),
    'nitroglycerin': ('glyceryl trinitrate', 'C01DA02'),
    'isosorbide': ('isosorbide dinitrate', 'C01DA08'),
    'digoxin': ('digoxin', 'C01AA05'),
    'furosemide': ('furosemide', 'C03CA01'),
    'hydrochlorothiazide': ('hydrochlorothiazide', 'C03AA03'),
    'spironolactone': ('spironolactone', 'C03DA01'),
    'indapamide': ('indapamide', 'C03BA11'),
    'nifedipine': ('nifedipine', 'C08CA05'),
    'diltiazem': ('diltiazem', 'C08DB01'),
    'verapamil': ('verapamil', 'C08DA01'),
    'felodipine': ('felodipine', 'C08CA02'),
    'trimetazidine': ('trimetazidine', 'C01EB15'),
    'nicorandil': ('nicorandil', 'C01DX16'),
    'ivabradine': ('ivabradine', 'C01EB17'),

    # === Gastrointestinal (A) ===
    'omeprazole': ('omeprazole', 'A02BC01'),
    'esomeprazole': ('esomeprazole', 'A02BC05'),
    'pantoprazole': ('pantoprazole', 'A02BC02'),
    'lansoprazole': ('lansoprazole', 'A02BC03'),
    'rabeprazole': ('rabeprazole', 'A02BC04'),
    'ranitidine': ('ranitidine', 'A02BA02'),
    'famotidine': ('famotidine', 'A02BA03'),
    'metoclopramide': ('metoclopramide', 'A03FA01'),
    'domperidone': ('domperidone', 'A03FA03'),
    'loperamide': ('loperamide', 'A07DA03'),
    'mebeverine': ('mebeverine', 'A03AA04'),
    'sucralfate': ('sucralfate', 'A02BX02'),
    'bismuth': ('bismuth subcitrate', 'A02BX05'),
    'lactulose': ('lactulose', 'A06AD11'),
    'macrogol': ('macrogol', 'A06AD15'),

    # === Diabetes (A10) ===
    'metformin': ('metformin', 'A10BA02'),
    'glimepiride': ('glimepiride', 'A10BB12'),
    'gliclazide': ('gliclazide', 'A10BB09'),
    'glibenclamide': ('glibenclamide', 'A10BB01'),
    'glipizide': ('glipizide', 'A10BB07'),
    'pioglitazone': ('pioglitazone', 'A10BG03'),
    'sitagliptin': ('sitagliptin', 'A10BH01'),
    'vildagliptin': ('vildagliptin', 'A10BH02'),
    'linagliptin': ('linagliptin', 'A10BH05'),
    'empagliflozin': ('empagliflozin', 'A10BK03'),
    'dapagliflozin': ('dapagliflozin', 'A10BK01'),
    'insulin': ('insulin', 'A10AB01'),
    'acarbose': ('acarbose', 'A10BF01'),

    # === Antibiotics (J01) ===
    'amoxicillin': ('amoxicillin', 'J01CA04'),
    'ampicillin': ('ampicillin', 'J01CA01'),
    'penicillin': ('phenoxymethylpenicillin', 'J01CE02'),
    'cephalexin': ('cefalexin', 'J01DB01'),
    'cefuroxime': ('cefuroxime', 'J01DC02'),
    'ceftriaxone': ('ceftriaxone', 'J01DD04'),
    'cefixime': ('cefixime', 'J01DD08'),
    'cefpodoxime': ('cefpodoxime', 'J01DD13'),
    'azithromycin': ('azithromycin', 'J01FA10'),
    'clarithromycin': ('clarithromycin', 'J01FA09'),
    'erythromycin': ('erythromycin', 'J01FA01'),
    'ciprofloxacin': ('ciprofloxacin', 'J01MA02'),
    'levofloxacin': ('levofloxacin', 'J01MA12'),
    'ofloxacin': ('ofloxacin', 'J01MA01'),
    'moxifloxacin': ('moxifloxacin', 'J01MA14'),
    'norfloxacin': ('norfloxacin', 'J01MA06'),
    'metronidazole': ('metronidazole', 'J01XD01'),
    'doxycycline': ('doxycycline', 'J01AA02'),
    'tetracycline': ('tetracycline', 'J01AA07'),
    'gentamicin': ('gentamicin', 'J01GB03'),
    'amikacin': ('amikacin', 'J01GB06'),
    'clindamycin': ('clindamycin', 'J01FF01'),
    'vancomycin': ('vancomycin', 'J01XA01'),
    'linezolid': ('linezolid', 'J01XX08'),
    'trimethoprim': ('trimethoprim', 'J01EA01'),
    'sulfamethoxazole': ('sulfamethoxazole', 'J01EC01'),
    'nitrofurantoin': ('nitrofurantoin', 'J01XE01'),
    'fosfomycin': ('fosfomycin', 'J01XX01'),

    # === Pain/Anti-inflammatory (M/N) ===
    'paracetamol': ('paracetamol', 'N02BE01'),
    'acetaminophen': ('paracetamol', 'N02BE01'),
    'ibuprofen': ('ibuprofen', 'M01AE01'),
    'diclofenac': ('diclofenac', 'M01AB05'),
    'naproxen': ('naproxen', 'M01AE02'),
    'meloxicam': ('meloxicam', 'M01AC06'),
    'piroxicam': ('piroxicam', 'M01AC01'),
    'celecoxib': ('celecoxib', 'M01AH01'),
    'etoricoxib': ('etoricoxib', 'M01AH05'),
    'indomethacin': ('indometacin', 'M01AB01'),
    'ketoprofen': ('ketoprofen', 'M01AE03'),
    'tramadol': ('tramadol', 'N02AX02'),
    'codeine': ('codeine', 'N02AA59'),
    'morphine': ('morphine', 'N02AA01'),
    'fentanyl': ('fentanyl', 'N02AB03'),
    'pregabalin': ('pregabalin', 'N03AX16'),
    'gabapentin': ('gabapentin', 'N03AX12'),

    # === Muscle relaxants ===
    'methocarbamol': ('methocarbamol', 'M03BA03'),
    'thiocolchicoside': ('thiocolchicoside', 'M03BX05'),
    'eperisone': ('eperisone', 'M03BX09'),
    'tolperisone': ('tolperisone', 'M03BX04'),
    'baclofen': ('baclofen', 'M03BX01'),

    # === Respiratory (R) ===
    'salbutamol': ('salbutamol', 'R03AC02'),
    'terbutaline': ('terbutaline', 'R03AC03'),
    'formoterol': ('formoterol', 'R03AC13'),
    'salmeterol': ('salmeterol', 'R03AC12'),
    'ipratropium': ('ipratropium bromide', 'R03BB01'),
    'tiotropium': ('tiotropium bromide', 'R03BB04'),
    'budesonide': ('budesonide', 'R03BA02'),
    'fluticasone': ('fluticasone', 'R03BA05'),
    'beclometasone': ('beclometasone', 'R03BA01'),
    'montelukast': ('montelukast', 'R03DC03'),
    'theophylline': ('theophylline', 'R03DA04'),
    'aminophylline': ('aminophylline', 'R03DA05'),
    'acetylcysteine': ('acetylcysteine', 'R05CB01'),
    'ambroxol': ('ambroxol', 'R05CB06'),
    'bromhexine': ('bromhexine', 'R05CB02'),
    'dextromethorphan': ('dextromethorphan', 'R05DA09'),
    'codeine': ('codeine', 'R05DA04'),

    # === Antihistamines ===
    'loratadine': ('loratadine', 'R06AX13'),
    'cetirizine': ('cetirizine', 'R06AE07'),
    'desloratadine': ('desloratadine', 'R06AX27'),
    'levocetirizine': ('levocetirizine', 'R06AE09'),
    'fexofenadine': ('fexofenadine', 'R06AX26'),
    'chlorpheniramine': ('chlorphenamine', 'R06AB04'),
    'diphenhydramine': ('diphenhydramine', 'R06AA02'),
    'promethazine': ('promethazine', 'R06AD02'),

    # === Corticosteroids ===
    'prednisolone': ('prednisolone', 'H02AB06'),
    'prednisone': ('prednisone', 'H02AB07'),
    'methylprednisolone': ('methylprednisolone', 'H02AB04'),
    'dexamethasone': ('dexamethasone', 'H02AB02'),
    'hydrocortisone': ('hydrocortisone', 'H02AB09'),
    'betamethasone': ('betamethasone', 'H02AB01'),
    'triamcinolone': ('triamcinolone', 'H02AB08'),

    # === Psychotropic (N05/N06) ===
    'diazepam': ('diazepam', 'N05BA01'),
    'lorazepam': ('lorazepam', 'N05BA06'),
    'alprazolam': ('alprazolam', 'N05BA12'),
    'clonazepam': ('clonazepam', 'N03AE01'),
    'midazolam': ('midazolam', 'N05CD08'),
    'zolpidem': ('zolpidem', 'N05CF02'),
    'zopiclone': ('zopiclone', 'N05CF01'),
    'amitriptyline': ('amitriptyline', 'N06AA09'),
    'sertraline': ('sertraline', 'N06AB06'),
    'fluoxetine': ('fluoxetine', 'N06AB03'),
    'paroxetine': ('paroxetine', 'N06AB05'),
    'escitalopram': ('escitalopram', 'N06AB10'),
    'citalopram': ('citalopram', 'N06AB04'),
    'venlafaxine': ('venlafaxine', 'N06AX16'),
    'duloxetine': ('duloxetine', 'N06AX21'),
    'mirtazapine': ('mirtazapine', 'N06AX11'),
    'trazodone': ('trazodone', 'N06AX05'),
    'quetiapine': ('quetiapine', 'N05AH04'),
    'olanzapine': ('olanzapine', 'N05AH03'),
    'risperidone': ('risperidone', 'N05AX08'),
    'aripiprazole': ('aripiprazole', 'N05AX12'),
    'haloperidol': ('haloperidol', 'N05AD01'),

    # === Anticonvulsants ===
    'carbamazepine': ('carbamazepine', 'N03AF01'),
    'valproic': ('valproic acid', 'N03AG01'),
    'valproate': ('valproic acid', 'N03AG01'),
    'phenytoin': ('phenytoin', 'N03AB02'),
    'levetiracetam': ('levetiracetam', 'N03AX14'),
    'lamotrigine': ('lamotrigine', 'N03AX09'),
    'topiramate': ('topiramate', 'N03AX11'),
    'oxcarbazepine': ('oxcarbazepine', 'N03AF02'),

    # === Antivirals ===
    'acyclovir': ('aciclovir', 'J05AB01'),
    'valacyclovir': ('valaciclovir', 'J05AB11'),
    'oseltamivir': ('oseltamivir', 'J05AH02'),
    'ribavirin': ('ribavirin', 'J05AB04'),
    'tenofovir': ('tenofovir disoproxil', 'J05AF07'),
    'lamivudine': ('lamivudine', 'J05AF05'),
    'entecavir': ('entecavir', 'J05AF10'),

    # === Antifungals ===
    'fluconazole': ('fluconazole', 'J02AC01'),
    'itraconazole': ('itraconazole', 'J02AC02'),
    'ketoconazole': ('ketoconazole', 'J02AB02'),
    'terbinafine': ('terbinafine', 'D01BA02'),
    'clotrimazole': ('clotrimazole', 'D01AC01'),
    'miconazole': ('miconazole', 'D01AC02'),
    'nystatin': ('nystatin', 'A07AA02'),

    # === Gout ===
    'allopurinol': ('allopurinol', 'M04AA01'),
    'febuxostat': ('febuxostat', 'M04AA03'),
    'colchicine': ('colchicine', 'M04AC01'),

    # === Thyroid ===
    'levothyroxine': ('levothyroxine sodium', 'H03AA01'),
    'thiamazole': ('thiamazole', 'H03BB02'),
    'propylthiouracil': ('propylthiouracil', 'H03BA02'),

    # === Vitamins/Minerals ===
    'calcium': ('calcium', 'A12AA'),
    'vitamin d': ('colecalciferol', 'A11CC05'),
    'vitamin b': ('vitamin b complex', 'A11EA'),
    'vitamin c': ('ascorbic acid', 'A11GA01'),
    'vitamin e': ('tocopherol', 'A11HA03'),
    'folic acid': ('folic acid', 'B03BB01'),
    'iron': ('ferrous sulfate', 'B03AA07'),
    'zinc': ('zinc', 'A12CB'),
    'magnesium': ('magnesium', 'A12CC'),
    'potassium': ('potassium chloride', 'A12BA01'),

    # === Others ===
    'diosmin': ('diosmin', 'C05CA03'),
    'silymarin': ('silymarin', 'A05BA03'),
    'betahistine': ('betahistine', 'N07CA01'),
    'cinnarizine': ('cinnarizine', 'N07CA02'),
    'piracetam': ('piracetam', 'N06BX03'),
    'ginkgo': ('ginkgo biloba', 'N06DX02'),
    'donepezil': ('donepezil', 'N06DA02'),
    'memantine': ('memantine', 'N06DX01'),
    'domperidone': ('domperidone', 'A03FA03'),

    # === VN Brand Names - Common mappings ===
    'agicardi': ('acetylsalicylic acid', 'B01AC06'),  # Aspirin
    'partamol': ('paracetamol', 'N02BE01'),
    'concor': ('bisoprolol', 'C07AB07'),
    'lipitor': ('atorvastatin', 'C10AA05'),
    'plavix': ('clopidogrel', 'B01AC04'),
    'nexium': ('esomeprazole', 'A02BC05'),
    'glucophage': ('metformin', 'A10BA02'),
    'augmentin': ('amoxicillin and enzyme inhibitor', 'J01CR02'),
    'klacid': ('clarithromycin', 'J01FA09'),
    'zithromax': ('azithromycin', 'J01FA10'),
    'voltaren': ('diclofenac', 'M01AB05'),
    'arcoxia': ('etoricoxib', 'M01AH05'),
    'ventolin': ('salbutamol', 'R03AC02'),
    'singulair': ('montelukast', 'R03DC03'),
    'zyrtec': ('cetirizine', 'R06AE07'),
    'claritin': ('loratadine', 'R06AX13'),
    'xanax': ('alprazolam', 'N05BA12'),
    'lexapro': ('escitalopram', 'N06AB10'),
    'lyrica': ('pregabalin', 'N03AX16'),
    'neurontin': ('gabapentin', 'N03AX12'),
    'diamicron': ('gliclazide', 'A10BB09'),
    'amaryl': ('glimepiride', 'A10BB12'),
    'januvia': ('sitagliptin', 'A10BH01'),

    # === VN Brand Names - Extended ===
    # PPIs
    'stadnex': ('esomeprazole', 'A02BC05'),
    'gastevin': ('esomeprazole', 'A02BC05'),
    'rablet': ('rabeprazole', 'A02BC04'),
    'prazopro': ('omeprazole', 'A02BC01'),

    # Cardiovascular
    'valsgim': ('valsartan', 'C09CA03'),
    'apitim': ('amlodipine', 'C08CA01'),
    'lacisartan': ('losartan', 'C09CA01'),
    'lisoril': ('lisinopril', 'C09AA03'),
    'mibetel': ('telmisartan', 'C09CA07'),
    'rotinvast': ('rosuvastatin', 'C10AA07'),
    'lowsta': ('atorvastatin', 'C10AA05'),
    'lipanthyl': ('fenofibrate', 'C10AB05'),
    'tunadimet': ('clopidogrel', 'B01AC04'),
    'dogrelsavi': ('clopidogrel', 'B01AC04'),

    # Diabetes
    'glumeform': ('metformin', 'A10BA02'),
    'glumeben': ('glibenclamide', 'A10BB01'),
    'novomix': ('insulin aspart', 'A10AD05'),
    'scilin': ('insulin', 'A10AB01'),
    'glaritus': ('insulin glargine', 'A10AE04'),

    # Pain/Inflammation
    'greatcet': ('paracetamol', 'N02BE01'),
    'pharbacol': ('paracetamol', 'N02BE01'),
    'toricam': ('piroxicam', 'M01AC01'),
    'meyerproxen': ('naproxen', 'M01AE02'),
    'afenacol': ('diclofenac', 'M01AB05'),
    'etoricoxib': ('etoricoxib', 'M01AH05'),

    # Antibiotics
    'acigmentin': ('amoxicillin and enzyme inhibitor', 'J01CR02'),
    'curam': ('amoxicillin and enzyme inhibitor', 'J01CR02'),
    'midantin': ('amoxicillin and enzyme inhibitor', 'J01CR02'),
    'cepmaxlox': ('cefpodoxime', 'J01DD13'),
    'cefcenat': ('ceftriaxone', 'J01DD04'),
    'bicelor': ('cefaclor', 'J01DC04'),
    'kaflovo': ('levofloxacin', 'J01MA12'),
    'kapredin': ('cefuroxime', 'J01DC02'),
    'momencef': ('cefixime', 'J01DD08'),
    'zaromax': ('azithromycin', 'J01FA10'),

    # Antihistamines
    'apixodin': ('desloratadine', 'R06AX27'),
    'cetirizin': ('cetirizine', 'R06AE07'),
    'fexofenadine': ('fexofenadine', 'R06AX26'),

    # Respiratory
    'atisyrup': ('zinc preparations', 'A12CB'),
    'hoastex': ('herbal cough preparation', 'R05X'),

    # Muscle relaxants
    'nakibu': ('thiocolchicoside', 'M03BX05'),

    # Eye drops
    'sanlein': ('hyaluronic acid', 'S01XA'),
    'laci-eye': ('artificial tears', 'S01XA20'),

    # GI
    'sucrafil': ('sucralfate', 'A02BX02'),

    # Probiotics (A07FA)
    'bolabio': ('lactobacillus', 'A07FA01'),
    'biosubtyl': ('bacillus subtilis', 'A07FA'),
    'enterobella': ('probiotics', 'A07FA'),

    # Urology
    'harnal': ('tamsulosin', 'G04CA02'),

    # Others
    'dacolfort': ('diosmin', 'C05CA03'),
    'imerixx': ('trimebutine', 'A03AA05'),
    'b12 ankermann': ('cyanocobalamin', 'B03BA01'),
    'calciferat': ('calcium and vitamin d', 'A12AX'),
    'vinfoxin': ('sertraline', 'N06AB06'),
    'mirzaten': ('mirtazapine', 'N06AX11'),
    'lupipezil': ('donepezil', 'N06DA02'),
    'ozanta': ('olanzapine', 'N05AH03'),

    # === More VN Brand Names ===
    # Antacids/GI
    'hantacid': ('aluminium hydroxide', 'A02AB01'),
    'phostaligel': ('aluminium phosphate', 'A02AB03'),

    # Antibiotics
    'qcozetax': ('cefotaxime', 'J01DD01'),
    'imefed': ('cefixime', 'J01DD08'),
    'wizosone': ('cefpodoxime', 'J01DD13'),

    # Pain
    'nefolin': ('tramadol', 'N02AX02'),
    'ketovital': ('ketoprofen', 'M01AE03'),
    'ketoproxin': ('ketoprofen', 'M01AE03'),

    # Cardiovascular
    'savitelmihct': ('telmisartan and diuretics', 'C09DA07'),
    'huntelaar': ('losartan', 'C09CA01'),

    # Neurological
    'pegaset': ('pregabalin', 'N03AX16'),
    'gaptinew': ('gabapentin', 'N03AX12'),
    'letarid': ('levetiracetam', 'N03AX14'),
    'epilona': ('valproic acid', 'N03AG01'),

    # Cough/Cold
    'drotusc': ('dextromethorphan', 'R05DA09'),
    'dixirein': ('ambroxol', 'R05CB06'),

    # ENT
    'betahistin': ('betahistine', 'N07CA01'),

    # Urology
    'alfuzosin': ('alfuzosin', 'G04CA01'),

    # Dermatology
    'asosalic': ('salicylic acid', 'D01AE12'),
    'golheal': ('collagenase', 'D03BA'),

    # Traditional medicine (categorize as herbal)
    'hoạt huyết': ('herbal preparation', 'V'),

    # Supplements
    'mivifort': ('multivitamins', 'A11AA'),

    # === HIGH FREQUENCY UNMAPPED - Extended ===
    # Pain/Fever
    'hapacol': ('paracetamol', 'N02BE01'),
    'atirin': ('paracetamol', 'N02BE01'),
    'siro ho haspan': ('cough preparation', 'R05'),
    'meyerexcold': ('cold preparation', 'R05X'),

    # Cardiovascular
    'kavasdin': ('amlodipine', 'C08CA01'),
    'ridolip': ('rosuvastatin', 'C10AA07'),
    'amaloris': ('amlodipine', 'C08CA01'),
    'tracardis': ('telmisartan', 'C09CA07'),
    'vastanic': ('atorvastatin', 'C10AA05'),
    'auroliza': ('rosuvastatin', 'C10AA07'),
    'atoris': ('atorvastatin', 'C10AA05'),
    'atelec': ('cilnidipine', 'C08CA14'),
    'mitiquapril': ('quinapril', 'C09AA06'),
    'bluecan': ('candesartan', 'C09CA06'),

    # Diabetes
    'jardiance': ('empagliflozin', 'A10BK03'),
    'gikanin': ('gliclazide', 'A10BB09'),
    'bividiac': ('vildagliptin', 'A10BH02'),

    # GI
    'capesto': ('esomeprazole', 'A02BC05'),
    'isoday': ('pantoprazole', 'A02BC02'),
    'lacikez': ('lansoprazole', 'A02BC03'),

    # Antibiotics
    'usaralphar': ('amoxicillin', 'J01CA04'),
    'antikans': ('amoxicillin and enzyme inhibitor', 'J01CR02'),
    'biocemet': ('cefixime', 'J01DD08'),

    # Antihistamines/Allergy
    'fefasdin': ('fexofenadine', 'R06AX26'),

    # Anti-inflammatory
    'etofride': ('etoricoxib', 'M01AH05'),
    'tiphapred': ('methylprednisolone', 'H02AB04'),

    # ENT/Eye
    'bluemint': ('menthol preparation', 'R01AX'),

    # Vitamins
    'calci sac': ('calcium', 'A12AA'),
    'b-coenzyme': ('vitamin b complex', 'A11EA'),
}


def normalize_drug_name(name):
    """Chuẩn hóa tên thuốc để matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower()
    # Remove special characters but keep spaces and hyphens
    name = re.sub(r'[^\w\s\-]', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def extract_keywords(drug_name):
    """Extract các từ khóa từ tên thuốc."""
    normalized = normalize_drug_name(drug_name)
    # Split by space and hyphen
    words = re.split(r'[\s\-]+', normalized)
    # Filter short words and numbers
    keywords = [w for w in words if len(w) > 2 and not w.isdigit()]
    return keywords


def match_with_manual_dict(drug_name):
    """
    Matching tên thuốc với manual dictionary.
    Returns: (generic_name, atc_code, confidence) or None
    """
    normalized = normalize_drug_name(drug_name)
    keywords = extract_keywords(drug_name)

    # Try exact match first
    for keyword in keywords:
        if keyword in MANUAL_DRUG_MAPPING:
            generic, atc = MANUAL_DRUG_MAPPING[keyword]
            return (generic, atc, 'high')

    # Try partial match
    for key, (generic, atc) in MANUAL_DRUG_MAPPING.items():
        if key in normalized:
            return (generic, atc, 'medium')

    return None


def fuzzy_match_atc(drug_name, atc_df, threshold=0.7):
    """
    Fuzzy matching với ATC database.
    Returns: (atc_code, atc_name, similarity) or None
    """
    normalized = normalize_drug_name(drug_name)
    keywords = extract_keywords(drug_name)

    best_match = None
    best_score = 0

    # Only check level 5 codes (actual drugs)
    level5 = atc_df[atc_df['atc_code'].str.len() == 7]

    for _, row in level5.iterrows():
        atc_name = normalize_drug_name(row['atc_name'])

        # Check if any keyword matches
        for keyword in keywords:
            if len(keyword) > 3:  # Only check meaningful keywords
                # Direct substring match
                if keyword in atc_name or atc_name in normalized:
                    score = SequenceMatcher(None, keyword, atc_name).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = (row['atc_code'], row['atc_name'], score)

        # Full name similarity
        similarity = SequenceMatcher(None, normalized, atc_name).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = (row['atc_code'], row['atc_name'], similarity)

    if best_match and best_score >= threshold:
        return best_match
    return None


def map_drugs_to_atc(drugs_df, atc_df):
    """
    Map danh sách thuốc VN sang ATC codes.
    """
    results = []

    for idx, row in drugs_df.iterrows():
        drug_name = row['drug_name']
        count = row.get('count', 0)

        result = {
            'drug_name': drug_name,
            'count': count,
            'generic_name': '',
            'atc_code': '',
            'confidence': '',
            'method': ''
        }

        # Strategy 1: Manual dictionary matching
        manual_match = match_with_manual_dict(drug_name)
        if manual_match:
            result['generic_name'] = manual_match[0]
            result['atc_code'] = manual_match[1]
            result['confidence'] = manual_match[2]
            result['method'] = 'manual_dict'
        else:
            # Strategy 2: Fuzzy matching
            fuzzy_match = fuzzy_match_atc(drug_name, atc_df, threshold=0.65)
            if fuzzy_match:
                result['generic_name'] = fuzzy_match[1]
                result['atc_code'] = fuzzy_match[0]
                result['confidence'] = f'fuzzy_{fuzzy_match[2]:.2f}'
                result['method'] = 'fuzzy_match'

        results.append(result)

    return pd.DataFrame(results)


def main():
    print("=== DRUG NAME TO ATC MAPPING ===")
    print()

    # Load data
    print("Loading data...")
    drugs_df = pd.read_csv('data_vn/unique_drugs.csv')
    atc_df = pd.read_csv('data_vn/atc_reference/who_atc_ddd.csv')

    print(f"  Drugs to map: {len(drugs_df)}")
    print(f"  ATC codes available: {len(atc_df)}")
    print()

    # Run mapping
    print("Running mapping...")
    results_df = map_drugs_to_atc(drugs_df, atc_df)

    # Statistics
    mapped = results_df[results_df['atc_code'] != '']
    unmapped = results_df[results_df['atc_code'] == '']

    print()
    print("=== MAPPING STATISTICS ===")
    print(f"Total drugs: {len(results_df)}")
    print(f"Mapped: {len(mapped)} ({100*len(mapped)/len(results_df):.1f}%)")
    print(f"Unmapped: {len(unmapped)} ({100*len(unmapped)/len(results_df):.1f}%)")
    print()

    # By method
    print("By method:")
    print(results_df['method'].value_counts())
    print()

    # By confidence
    print("By confidence:")
    print(results_df['confidence'].value_counts().head(10))
    print()

    # Coverage by count
    total_prescriptions = drugs_df['count'].sum()
    mapped_prescriptions = results_df[results_df['atc_code'] != '']['count'].sum()
    print(f"Prescription coverage: {mapped_prescriptions}/{total_prescriptions} ({100*mapped_prescriptions/total_prescriptions:.1f}%)")
    print()

    # Save results
    os.makedirs('data_vn', exist_ok=True)

    # Save all mappings
    results_df.to_csv('data_vn/drug_atc_mapping.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: data_vn/drug_atc_mapping.csv")

    # Save unmapped drugs (sorted by count for priority review)
    unmapped_sorted = unmapped.sort_values('count', ascending=False)
    unmapped_sorted.to_csv('data_vn/unmapped_drugs.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: data_vn/unmapped_drugs.csv")

    # Print top unmapped drugs
    print()
    print("=== TOP 30 UNMAPPED DRUGS (by frequency) ===")
    for _, row in unmapped_sorted.head(30).iterrows():
        print(f"  {row['count']:5d}x  {row['drug_name']}")

    # Print sample mapped drugs
    print()
    print("=== SAMPLE MAPPED DRUGS ===")
    for _, row in mapped.head(20).iterrows():
        print(f"  {row['drug_name'][:30]:30s} -> {row['atc_code']} ({row['generic_name']}) [{row['method']}]")


if __name__ == '__main__':
    main()
