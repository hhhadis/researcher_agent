
import json
import os
import re

def parse_ris_like_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    papers = []
    current_paper = {
        'title': '',
        'abstract': '',
        'keywords': [],
        'year': ''
    }
    
    # Simple state machine or line-by-line processing
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('TI  - '):
            current_paper['title'] = line[6:].strip()
        elif line.startswith('AB  - '):
            current_paper['abstract'] = line[6:].strip()
        elif line.startswith('KW  - '):
            current_paper['keywords'].append(line[6:].strip())
        elif line.startswith('PY  - '):
            current_paper['year'] = line[6:].strip()
        elif line.startswith('ER  -'):
            # End of record, save and reset
            # Generate a simple ID based on index
            current_paper['paperId'] = str(len(papers))
            papers.append(current_paper)
            current_paper = {
                'title': '',
                'abstract': '',
                'keywords': [],
                'year': ''
            }

    return papers

def main():
    input_dir = r'd:\workspace\ResearchAgent\data\paper'
    output_dir = r'd:\workspace\ResearchAgent\core concept_tree_2'
    output_file = os.path.join(output_dir, 'extracted_data.json')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_papers = []
    
    # Iterate over all .txt files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            print(f"Reading from {file_path}...")
            papers = parse_ris_like_file(file_path)
            all_papers.extend(papers)

    # Re-assign unique PaperIds
    for idx, paper in enumerate(all_papers):
        paper['paperId'] = str(idx)

    print(f"Extracted total {len(all_papers)} papers.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
