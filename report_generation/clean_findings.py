import json
import os
import shutil
import time
import re
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
INPUT_FILE = r"d:\workspace\ResearchAgent\report_generation\extracted_hierarchical.jsonl"
BACKUP_FILE = r"d:\workspace\ResearchAgent\report_generation\extracted_hierarchical.jsonl.bak"
OUTPUT_FILE = r"d:\workspace\ResearchAgent\report_generation\extracted_hierarchical_cleaned.jsonl"

# Load API Keys (Simplified for this script)
def load_api_keys():
    keys = {}
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        key_file_path = os.path.join(project_root, 'API_KEY.md')
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match_zhipu = re.search(r'ZHIPUAI_API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match_zhipu:
                    keys['zhipu'] = match_zhipu.group(1)
    except:
        pass
    if 'zhipu' not in keys:
        keys['zhipu'] = os.environ.get("ZHIPUAI_API_KEY")
    return keys

keys = load_api_keys()
if not keys.get('zhipu'):
    print("Error: ZHIPUAI_API_KEY not found. Please set it in API_KEY.md or env vars.")
    exit(1)

zhipu_client = ZhipuAI(api_key=keys['zhipu'])

def evaluate_finding_quality(finding_item):
    """
    Uses LLM to evaluate the quality of a research finding (Question-Argument-Evidence).
    Returns a score (0-10) and a boolean 'keep' decision.
    """
    question = finding_item.get('question', '')
    # Check both 'arguments' and 'claim' fields as the structure might vary
    arguments = finding_item.get('arguments', [])
    claim = finding_item.get('claim', '')
    
    # Construct a readable representation for the LLM
    text_repr = f"Question: {question}\n"
    
    if arguments:
        for i, arg in enumerate(arguments):
            if isinstance(arg, dict):
                text_repr += f"Argument {i+1}: {arg.get('claim', '')}\n"
                text_repr += f"Evidence {i+1}: {json.dumps(arg.get('evidence', []), ensure_ascii=False)}\n"
            else:
                text_repr += f"Argument {i+1}: {arg}\n"
    elif claim:
         text_repr += f"Claim: {claim}\n"
         evidence = finding_item.get('evidence', [])
         text_repr += f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
    else:
        return 0, False

    prompt = f"""
    You are a strict Research Quality Auditor. Evaluate the following research finding for quality, specificity, and usefulness.

    Finding to Evaluate:
    {text_repr}

    Criteria for "High Quality":
    1. **Specificity**: The Question is specific and well-defined, not vague (e.g., "What is X?" is bad; "How does X affect Y under Z conditions?" is good).
    2. **Substance**: The Argument/Claim provides a meaningful, non-trivial answer.
    3. **Evidence**: The Evidence consists of concrete quotes or data from the text, not just general statements.
    4. **Completeness**: The finding actually answers the question posed.

    Criteria for rejection (Low Quality):
    - The answer is "not mentioned", "unknown", or "no information".
    - The question is trivial or structural (e.g., "What is the introduction?").
    - The evidence is missing or irrelevant.

    Task:
    1. Assign a Quality Score (0-10).
    2. Decide whether to KEEP (Score >= 7) or DISCARD the finding.
    
    Output JSON ONLY:
    {{
        "score": 8,
        "keep": true,
        "reason": "Specific question with strong evidential support."
    }}
    """

    for attempt in range(3):
        try:
            response = zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return result.get('score', 0), result.get('keep', False)
        except Exception as e:
            time.sleep(1)
            
    return 0, False

def process_batch(findings_list):
    """
    Processes a list of findings in parallel.
    """
    valid_results = []
    
    # Limit max_workers to avoid rate limits
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_finding = {executor.submit(evaluate_finding_quality, item): item for item in findings_list}
        
        for future in as_completed(future_to_finding):
            original_item = future_to_finding[future]
            try:
                score, keep = future.result()
                if keep:
                    valid_results.append(original_item)
            except Exception as e:
                # print(f"Error processing finding: {e}")
                pass
                
    return valid_results

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    # print(f"Backing up original file to {BACKUP_FILE}...")
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    
    print(f"Processing {INPUT_FILE} with LLM Semantic Evaluation...")
    
    all_findings_to_process = []
    
    # Read all findings first to process them properly
    # The file structure is one JSON object per line, which contains a "findings" list
    # We need to flatten this structure for processing and then reconstruct it or save as flat list
    
    file_structure_map = [] # To reconstruct original file structure if needed
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Reading {len(lines)} files from input...")
    
    processed_lines = []
    total_findings_count = 0
    kept_findings_count = 0
    
    for line in tqdm(lines, desc="Processing Files"):
        try:
            data = json.loads(line)
            findings = data.get('findings', [])
            
            if not findings:
                processed_lines.append(data)
                continue
                
            total_findings_count += len(findings)
            
            # Process findings for this file
            valid_findings = process_batch(findings)
            kept_findings_count += len(valid_findings)
            
            data['findings'] = valid_findings
            processed_lines.append(data)
                
        except json.JSONDecodeError:
            continue
            
    # Write back cleaned data
    print(f"Writing cleaned data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in processed_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # print(f"Replacing original file with cleaned version...")
    # shutil.move(OUTPUT_FILE, INPUT_FILE)
    
    print("-" * 30)
    print(f"Semantic Filtering Complete.")
    print(f"Total Findings Evaluated: {total_findings_count}")
    print(f"Total Findings Kept: {kept_findings_count}")
    print(f"Discarded: {total_findings_count - kept_findings_count} low-quality findings")

if __name__ == "__main__":
    main()