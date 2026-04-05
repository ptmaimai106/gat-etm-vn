"""
ICD-10 Utilities for VN Data
============================

Utility functions để normalize và validate ICD codes từ dữ liệu VN.
Xử lý các edge cases như:
- Dấu `*` (secondary code marker)
- Format không có dấu `.`
- Codes không có trong WHO ICD-10 (fallback về parent)

Author: Claude Code
Date: 2026-03-25
"""

import re
from typing import Optional, List, Tuple
import simple_icd_10 as icd


def normalize_icd_code(code: str) -> str:
    """
    Normalize ICD-10 code to standard format.

    - Remove special markers (*, +, !)
    - Ensure proper dot format (A00.0)
    - Uppercase
    - Strip whitespace

    Args:
        code: Raw ICD code from VN data

    Returns:
        Normalized ICD code
    """
    if not code:
        return ""

    # Strip and uppercase
    normalized = str(code).strip().upper()

    # Remove special markers (*, +, ! are ICD-10 dagger/asterisk markers)
    normalized = re.sub(r'[*+!†‡]', '', normalized)

    # Remove any trailing/leading non-alphanumeric (except dot)
    normalized = re.sub(r'^[^A-Z0-9]+|[^A-Z0-9.]+$', '', normalized)

    # Ensure dot format for codes > 3 chars
    if len(normalized) > 3 and '.' not in normalized:
        normalized = normalized[:3] + '.' + normalized[3:]

    return normalized


def validate_and_map_icd(code: str, fallback_to_parent: bool = True) -> Tuple[str, bool, str]:
    """
    Validate ICD code and optionally map to parent if not found.

    Args:
        code: ICD code to validate
        fallback_to_parent: If True, map unknown codes to nearest valid parent

    Returns:
        Tuple of (mapped_code, is_exact_match, original_code)
    """
    original = code
    normalized = normalize_icd_code(code)

    if not normalized:
        return ("", False, original)

    # Check exact match
    if icd.is_valid_item(normalized):
        return (normalized, True, original)

    # Try without dot
    no_dot = normalized.replace('.', '')
    if icd.is_valid_item(no_dot):
        return (no_dot, True, original)

    # Fallback to parent
    if fallback_to_parent:
        # Try progressively shorter codes
        test_code = normalized
        while len(test_code) > 1:
            # Remove last character (or dot + character)
            if test_code.endswith('.'):
                test_code = test_code[:-1]
            elif '.' in test_code:
                parts = test_code.rsplit('.', 1)
                if len(parts[1]) > 1:
                    test_code = parts[0] + '.' + parts[1][:-1]
                else:
                    test_code = parts[0]
            else:
                test_code = test_code[:-1]

            if icd.is_valid_item(test_code):
                return (test_code, False, original)

    return ("", False, original)


def extract_icd_codes_from_string(text: str) -> List[str]:
    """
    Extract all ICD codes from a string (may contain multiple codes).

    Handles formats:
    - Single code: "I10"
    - Multiple with semicolon: "I10; E78.2; K21"
    - Multiple with comma: "I10, E78.2, K21"

    Args:
        text: String containing ICD code(s)

    Returns:
        List of extracted ICD codes
    """
    if not text or str(text).lower() in ('nan', 'none', ''):
        return []

    text = str(text)

    # Split by common delimiters
    codes = re.split(r'[;,\s]+', text)

    # Filter and normalize
    result = []
    for code in codes:
        normalized = normalize_icd_code(code)
        if normalized and len(normalized) >= 2:  # At least 2 chars (chapter code)
            result.append(normalized)

    return result


def get_icd_description(code: str) -> str:
    """Get description for ICD code, handling edge cases."""
    normalized = normalize_icd_code(code)
    if not normalized:
        return ""

    try:
        if icd.is_valid_item(normalized):
            return icd.get_description(normalized)
    except Exception:
        pass

    return ""


def get_icd_ancestors(code: str) -> List[str]:
    """Get all ancestors for ICD code."""
    normalized = normalize_icd_code(code)
    if not normalized:
        return []

    try:
        if icd.is_valid_item(normalized):
            return icd.get_ancestors(normalized)
    except Exception:
        pass

    return []


def batch_validate_codes(codes: List[str], fallback_to_parent: bool = True) -> dict:
    """
    Validate a batch of ICD codes.

    Args:
        codes: List of ICD codes to validate
        fallback_to_parent: If True, map unknown codes to parent

    Returns:
        dict with validation results
    """
    results = {
        'exact_matches': [],
        'parent_fallbacks': [],
        'not_found': [],
        'mapping': {}  # original -> mapped
    }

    for code in codes:
        mapped, is_exact, original = validate_and_map_icd(code, fallback_to_parent)

        if mapped:
            results['mapping'][original] = mapped
            if is_exact:
                results['exact_matches'].append(original)
            else:
                results['parent_fallbacks'].append((original, mapped))
        else:
            results['not_found'].append(original)

    return results


# Test
if __name__ == '__main__':
    print("Testing ICD utilities...")
    print("=" * 50)

    # Test cases
    test_codes = [
        'I10',      # Normal
        'E78.2',    # With dot
        'E789',     # Without dot
        'D63.8*',   # With asterisk
        'D75.2',    # Not in WHO (should fallback to D75)
        'J44.0+',   # With plus
        ' k21 ',    # Lowercase with spaces
        'INVALID',  # Invalid
    ]

    print("\nNormalization test:")
    for code in test_codes:
        normalized = normalize_icd_code(code)
        print(f"  {code!r:15} -> {normalized!r}")

    print("\nValidation test:")
    for code in test_codes:
        mapped, is_exact, original = validate_and_map_icd(code)
        status = "exact" if is_exact else ("fallback" if mapped else "NOT FOUND")
        desc = get_icd_description(mapped) if mapped else ""
        print(f"  {code!r:15} -> {mapped!r:10} [{status:8}] {desc[:40]}")

    print("\nBatch validation:")
    results = batch_validate_codes(test_codes)
    print(f"  Exact matches: {len(results['exact_matches'])}")
    print(f"  Fallbacks: {len(results['parent_fallbacks'])}")
    print(f"  Not found: {len(results['not_found'])}")
