import re


def normalize_medqa(s: str) -> str:
    """
    Extract the answer letter (A, B, C, or D) from the model output.
    Handles various formats like "A", "The answer is A", "A.", "Answer: A", etc.
    
    Updated to prioritize single-token responses (with max_new_tokens=1, 
    the model should generate only A, B, C, or D).
    
    Note: If the model generates partial words (e.g., "The" instead of "B"), 
    this function cannot extract the answer and will return an empty string.
    In such cases, consider using logit-based prediction or constrained decoding
    to force the model to generate only A, B, C, or D tokens.
    """
    if not s:
        return ""
    
    # Remove leading/trailing whitespace
    s = s.strip()
    
    # For single-token responses (max_new_tokens=1), the output should be
    # just the letter, possibly with whitespace. Check this first.
    # Match exactly A, B, C, or D (case insensitive) as standalone or with punctuation
    single_token_pattern = r'^\s*([ABCD])[\.\)\s]*$'
    match = re.match(single_token_pattern, s, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in ['A', 'B', 'C', 'D']:
            return letter
    
    # Try to find a single letter A, B, C, or D (case insensitive)
    # Look for the pattern at the start, after "Answer:", "answer:", or standalone
    patterns = [
        r'^[Aa]nswer\s*:?\s*([ABCD])',  # "Answer: A" or "answer: A"
        r'^([ABCD])[\.\)]?\s*$',  # "A." or "A)" or just "A"
        r'^[Tt]he\s+[Aa]nswer\s+is\s+([ABCD])',  # "The answer is A"
        r'^([ABCD])\b',  # A, B, C, or D at the start (word boundary)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in ['A', 'B', 'C', 'D']:
                return letter
    
    # If no pattern matches, try to extract the first A, B, C, or D found
    # (fallback for edge cases)
    match = re.search(r'\b([ABCD])\b', s.upper())
    if match:
        return match.group(1)
    
    # Handle common partial words that might indicate the model is trying to
    # generate a full sentence but got cut off. Unfortunately, we can't extract
    # the answer from partial words like "The", "Answer", etc.
    # These cases should ideally be handled by using constrained decoding or
    # logit-based prediction instead of text-based extraction.
    
    # Return empty string if no valid answer found
    return ""


def process_output_top1_medqa(output: str) -> str:
    """Process top-1 output for MedQA - extract just the answer letter."""
    return normalize_medqa(output)


def process_output_topk_medqa(output: str) -> str:
    """Process top-k output for MedQA - extract just the answer letter."""
    return normalize_medqa(output)


def process_output_cot_medqa(output: str) -> str:
    """Process chain-of-thought output for MedQA - extract just the answer letter."""
    return normalize_medqa(output)









