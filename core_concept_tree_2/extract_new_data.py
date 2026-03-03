
import json
import os
import re
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# API Key
def load_zhipuai_api_key():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        key_file_path = os.path.join(project_root, 'API_KEY.md')
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'ZHIPUAI_API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return os.environ.get("ZHIPUAI_API_KEY")

ZHIPUAI_API_KEY = load_zhipuai_api_key()

def extract_keywords_glm(text, client):
    """
    Use GLM to extract Technical and Ethical keywords.
    Returns a dict: {'technical': [], 'ethical': []}
    """
    if not text.strip():
        return {'technical': [], 'ethical': []}
    
    prompt = f"""
    Please extract keywords from the following research paper text (Title + Abstract).
    You need to extract two categories of keywords:
    1. Technical Keywords: Algorithms, models, methods, or technologies used or proposed (e.g., "Deep Learning", "Transformer", "Optimization").
    2. Ethical Keywords: Ethical issues, societal impacts, or values discussed (e.g., "Privacy", "Bias", "Fairness", "Accountability").
    
    Return the result in strictly valid JSON format with keys "technical" and "ethical".
    Example: {{"technical": ["keyword1", "keyword2"], "ethical": ["keyword3", "keyword4"]}}
    
    Text: {text}
    """
    
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "You are an expert researcher helper."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        # Clean up markdown if present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        
        result = json.loads(content)
        return {
            'technical': result.get('technical', []),
            'ethical': result.get('ethical', [])
        }
    except Exception as e:
        # print(f"Error extracting keywords: {e}")
        return {'technical': [], 'ethical': []}

def parse_ris_like_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    papers = []
    current_paper = {
        'title': '',
        'abstract': '',
        'year': ''
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('TI  - '):
            current_paper['title'] = line[6:].strip()
        elif line.startswith('AB  - '):
            current_paper['abstract'] = line[6:].strip()
        elif line.startswith('PY  - '):
            current_paper['year'] = line[6:].strip()
        elif line.startswith('ER  -'):
            # End of record
            if current_paper['title'] or current_paper['abstract']:
                current_paper['paperId'] = str(len(papers))
                papers.append(current_paper)
            current_paper = {
                'title': '',
                'abstract': '',
                'year': ''
            }

    return papers

def process_paper(paper, client):
    text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
    # Limit length
    text = text[:8000]
    
    extracted = extract_keywords_glm(text, client)
    paper['technical_keywords'] = extracted['technical']
    paper['ethical_keywords'] = extracted['ethical']
    # Remove raw keywords if any, or just keep them? The user said "not use original keywords"
    if 'keywords' in paper:
        del paper['keywords']
    return paper

def main():
    input_dir = r'd:\workspace\ResearchAgent\data\paper'
    output_dir = r'd:\workspace\ResearchAgent\core_concept_tree_2'
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

    print(f"Parsed total {len(all_papers)} papers. Starting GLM extraction...")
    
    # Initialize Client
    client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
    
    enhanced_papers = []
    batch_size = 10
    
    # Use ThreadPool
    completed_count = 0
    save_interval = 50
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_paper, p, client) for p in all_papers[:300]] # Limit to 300 for faster execution in this session
        
        for future in tqdm(futures, total=len(futures)):
            try:
                res = future.result()
                enhanced_papers.append(res)
                completed_count += 1
                
                if completed_count % save_interval == 0:
                     with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(enhanced_papers, f, indent=4, ensure_ascii=False)
                        
            except Exception as e:
                print(f"Error processing paper: {e}")

    print(f"Extracted data for {len(enhanced_papers)} papers.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_papers, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
