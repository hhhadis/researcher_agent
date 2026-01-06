"""
PDF Text Extractor Module
Extracts raw text from PDF files using PyMuPDF (fitz)
Supports section-aware extraction to prioritize important content
"""

import fitz  # PyMuPDF
import re
from typing import Optional, Dict, List, Tuple


def extract_text_from_pdf(pdf_path: str, section_aware: bool = True, max_chars: int = 15000) -> str:
    """
    Extract text from a PDF file with optional section-aware prioritization.
    
    Args:
        pdf_path: Path to the PDF file
        section_aware: If True, intelligently extract priority sections (default: True)
        max_chars: Maximum characters to extract (default: 15000)
        
    Returns:
        Extracted text as a string, optimized for LLM analysis
        
    Raises:
        ValueError: If PDF contains no extractable text (e.g., image-based PDF)
    """
    try:
        doc = fitz.open(pdf_path)
        
        if section_aware:
            text, _ = _extract_priority_sections(doc, max_chars)
        else:
            # Legacy mode: extract all text
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
        
        doc.close()
        
        # Check if extraction yielded meaningful text
        text_stripped = text.strip()
        if not text_stripped or len(text_stripped) < 50:
            raise ValueError(
                f"PDF appears to contain no extractable text (only {len(text_stripped)} characters). "
                f"This PDF may be:\n"
                f"  - An image-based/scanned document without OCR\n"
                f"  - Protected or encrypted\n"
                f"  - Using non-standard text encoding\n"
                f"To use this PDF, you may need to:\n"
                f"  - Run OCR (Optical Character Recognition) on it first\n"
                f"  - Convert it to a text-searchable format\n"
                f"  - Use a different version of the document"
            )
        
        return text
    
    except ValueError:
        # Re-raise ValueError with our custom message
        raise
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def _extract_priority_sections(doc, max_chars: int) -> Tuple[str, int]:
    """
    Extract priority sections from a research paper.
    
    Prioritizes: Abstract, Introduction, Method/Approach, Results, Conclusion
    Skips: References, Acknowledgments, Appendices
    
    Args:
        doc: PyMuPDF document object
        max_chars: Maximum characters to extract (soft limit - may slightly exceed for complete sections)
        
    Returns:
        Tuple of (extracted text, full text length)
    """
    # Extract all text first
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text() + "\n"
    
    full_text_length = len(full_text)
    
    # Detect sections
    sections = _detect_sections(full_text)
    
    if not sections:
        # Fallback: if no sections detected, just take the beginning
        extracted = full_text[:max_chars] + "\n\n[...truncated...]" if len(full_text) > max_chars else full_text
        return (extracted, full_text_length)
    
    # Define section priorities (higher = more important)
    priority_map = {
        'abstract': 10,
        'introduction': 9,
        'method': 10,
        'methodology': 10,
        'approach': 10,
        'methods': 10,
        'results': 7,
        'experiments': 7,
        'conclusion': 9,
        'conclusions': 9,
        'discussion': 6,
        'related work': 3,
        'background': 5,
        'references': 0,
        'acknowledgments': 0,
        'acknowledgements': 0,
        'appendix': 0,
        'supplementary': 0
    }
    
    # Assign priorities and sort sections
    prioritized_sections = []
    for section_name, start_pos, end_pos in sections:
        section_key = section_name.lower().strip()
        # Find matching priority (handle partial matches)
        priority = 4  # default priority for unknown sections
        for key, prio in priority_map.items():
            if key in section_key:
                priority = prio
                break
        
        if priority > 0:  # Skip zero-priority sections
            section_text = full_text[start_pos:end_pos].strip()
            prioritized_sections.append((priority, section_name, section_text))
    
    # Sort by priority (descending)
    prioritized_sections.sort(key=lambda x: x[0], reverse=True)
    
    # Build output text within character limit (soft limit - allow slight overflow for complete sections)
    output_parts = []
    current_length = 0
    soft_limit = max_chars * 1.1  # Allow 10% overflow to complete sections
    
    for priority, section_name, section_text in prioritized_sections:
        section_header = f"\n\n=== {section_name} ===\n\n"
        section_with_header = section_header + section_text
        
        # Use soft limit for adding complete sections, hard limit for partial
        if current_length + len(section_with_header) <= soft_limit:
            output_parts.append((priority, section_name, section_with_header))
            current_length += len(section_with_header)
        elif current_length + len(section_with_header) <= max_chars:
            # Still within hard limit, add it
            output_parts.append((priority, section_name, section_with_header))
            current_length += len(section_with_header)
        else:
            # Try to fit partial section (only if we have meaningful space)
            remaining_chars = max_chars - current_length - len(section_header) - 50
            if remaining_chars > 500:  # Only include if we can get meaningful content
                partial_text = section_text[:remaining_chars] + "\n\n[...section truncated...]"
                output_parts.append((priority, section_name, section_header + partial_text))
            break
    
    # Re-sort by appearance order for readability (but keep priority selection)
    # Actually, let's keep priority order to emphasize important content
    result_text = "".join([text for _, _, text in output_parts])
    
    if not result_text.strip():
        # Fallback if something went wrong
        result_text = full_text[:max_chars]
    
    return (result_text, full_text_length)


def _detect_sections(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect section boundaries in research paper text.
    
    Args:
        text: Full text of the paper
        
    Returns:
        List of (section_name, start_position, end_position) tuples
    """
    sections = []
    
    # Common section header patterns in academic papers
    # Matches: "1. Introduction", "2.1 Method", "Abstract", "INTRODUCTION", etc.
    patterns = [
        r'^(\d+\.?\s+)?([A-Z][A-Za-z\s]+)$',  # "1. Introduction" or "Introduction"
        r'^([A-Z][A-Z\s]+)$',  # "INTRODUCTION" (all caps)
        r'^(\d+\.?\d*\.?\s+)?([A-Z][A-Za-z\s]+)$',  # "2.1 Method"
    ]
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Skip very short or very long lines (unlikely to be section headers)
        if len(line_stripped) < 3 or len(line_stripped) > 100:
            continue
        
        # Check if line matches section header patterns
        is_section = False
        section_name = None
        
        for pattern in patterns:
            match = re.match(pattern, line_stripped)
            if match:
                # Extract section name (without numbering)
                groups = match.groups()
                section_name = groups[-1].strip() if groups else line_stripped
                
                # Validate: common section names or reasonably short
                if len(section_name.split()) <= 5:  # Section names typically short
                    is_section = True
                    break
        
        if is_section and section_name:
            # Calculate character position
            char_pos = sum(len(lines[j]) + 1 for j in range(i))  # +1 for \n
            sections.append((section_name, char_pos, None))
    
    # Set end positions (start of next section or end of document)
    for i in range(len(sections)):
        if i < len(sections) - 1:
            sections[i] = (sections[i][0], sections[i][1], sections[i+1][1])
        else:
            sections[i] = (sections[i][0], sections[i][1], len(text))
    
    return sections


def extract_text_from_pdf_bytes(pdf_bytes: bytes, section_aware: bool = True, max_chars: int = 15000) -> Tuple[str, int]:
    """
    Extract text from PDF file bytes (useful for uploaded files).
    
    Args:
        pdf_bytes: PDF file content as bytes
        section_aware: If True, intelligently extract priority sections (default: True)
        max_chars: Maximum characters to extract (default: 15000)
        
    Returns:
        Tuple of (extracted text, full text length)
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if section_aware:
            text, full_length = _extract_priority_sections(doc, max_chars)
        else:
            # Legacy mode: extract all text
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            full_length = len(text)
        
        doc.close()
        return (text, full_length)
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF bytes: {str(e)}")


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = len(doc)
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "page_count": page_count
        }
    
    except Exception as e:
        raise Exception(f"Error extracting PDF metadata: {str(e)}")


if __name__ == "__main__":
    # Test the extractor
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        print("=" * 60)
        print("Testing Section-Aware Extraction (Default)")
        print("=" * 60)
        text = extract_text_from_pdf(pdf_path, section_aware=True, max_chars=15000)
        print(f"Extracted {len(text)} characters")
        print("\nExtracted content preview (first 800 chars):")
        print(text[:800])
        print("\n...")
        
        print("\n" + "=" * 60)
        print("Testing Legacy Extraction (All Text)")
        print("=" * 60)
        text_legacy = extract_text_from_pdf(pdf_path, section_aware=False)
        print(f"Extracted {len(text_legacy)} characters (full paper)")
        print("\nFirst 500 characters:")
        print(text_legacy[:500])
    else:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        print("\nThis will demonstrate:")
        print("  1. Section-aware extraction (smart, prioritized)")
        print("  2. Legacy extraction (all text)")


