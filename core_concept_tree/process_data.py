
import json
import os
from zhipuai import ZhipuAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 请在这里填入你的 API Key，或者设置环境变量 ZHIPUAI_API_KEY
ZHIPUAI_API_KEY = "77e73e22741f4b45854c777f4763236f.311YBhRcPuEQk3ZQ"

def extract_keywords_glm(text, client):
    """
    使用 GLM-4-Flash 提取关键词
    """
    if not text.strip():
        return []
    
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for extracting academic keywords from research papers."},
                {"role": "user", "content": f"Please extract 3-5 representative technical keywords from the following text (Title + Abstract). Return ONLY the keywords separated by commas, no numbering or other text.\n\nText: {text}"}
            ],
            temperature=0.1,
            max_tokens=100
        )
        content = response.choices[0].message.content
        # 处理返回结果，分割逗号并去除空白
        keywords = [k.strip() for k in content.split(',') if k.strip()]
        return keywords
    except Exception as e:
        # print(f"Error extracting keywords: {e}") # 减少报错输出，避免刷屏
        return []

def process_item(item, client):
    title = item.get('title', '')
    abstract = item.get('abstract', '')
    text_to_extract = f"Title: {title}\nAbstract: {abstract}"
    
    # 限制长度，防止超出 Token 限制（虽然 Flash 支持长文本，但摘要通常不长）
    text_to_extract = text_to_extract[:10000]

    keywords = extract_keywords_glm(text_to_extract, client)
    
    # 如果提取失败（比如网络问题），保留空列表或者回退到简单的规则（这里暂留空）
    
    return {
        'paperId': item.get('paperId'),
        'title': title,
        'abstract': abstract,
        'year': item.get('year'),
        'keywords': keywords
    }

def process_json(file_path, output_dir):
    """
    Processes a JSON file to extract paper details, generates keywords using GLM-4-Flash,
    and saves the enhanced data to a new JSON file.
    """
    if not os.path.exists(file_path):
        print(f"Input file not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查 API Key
    api_key = os.environ.get("ZHIPUAI_API_KEY", ZHIPUAI_API_KEY)
    if "YOUR_API_KEY" in api_key or not api_key:
        print("Error: ZhipuAI API Key is missing.")
        print("Please set the ZHIPUAI_API_KEY environment variable or modify the 'ZHIPUAI_API_KEY' variable in this script.")
        return

    client = ZhipuAI(api_key=api_key)

    print(f"Processing {len(data)} papers with GLM-4-Flash...")
    
    # Check for existing data to resume progress
    output_path = os.path.join(output_dir, 'extracted_data.json')
    extracted_data = []
    processed_ids = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                extracted_data.extend(existing_data)
                for item in existing_data:
                    processed_ids.add(item.get('paperId'))
            print(f"Resuming from existing file. {len(processed_ids)} papers already processed.")
        except json.JSONDecodeError:
            print("Existing file is corrupted or empty. Starting from scratch.")

    # Filter items that need processing
    items_to_process = [item for item in data if item.get('paperId') not in processed_ids]
    print(f"Remaining papers to process: {len(items_to_process)}")

    if not items_to_process:
        print("All papers have been processed.")
        return output_path

    # Batch processing to save progress frequently
    batch_size = 50
    # Use ThreadPoolExecutor for concurrency
    max_workers = 10
    
    # Process in chunks
    for i in range(0, len(items_to_process), batch_size):
        batch_items = items_to_process[i : i + batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, item, client) for item in batch_items]
            
            for future in tqdm(futures, desc=f"Batch {i//batch_size + 1}/{(len(items_to_process)+batch_size-1)//batch_size}", leave=False):
                try:
                    result = future.result()
                    extracted_data.append(result)
                    
                    # Print sample log occasionally
                    if len(extracted_data) % 50 == 0:
                        title = result.get('title', 'No Title')[:50] + "..."
                        keywords = result.get('keywords', [])
                        tqdm.write(f"Processed {len(extracted_data)}/{len(data)}: {title} -> {keywords}")
                        
                except Exception as e:
                    print(f"Task failed: {e}")
        
        # Save after each batch
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        # print(f"Progress saved. Total processed: {len(extracted_data)}")

    print(f"Data extracted and saved to {output_path}")
    return output_path

if __name__ == '__main__':
    # 假设输入文件是原始的 train.json
    input_file = r'd:/workspace/ResearchAgent/Research-14K/data/train.json'
    
    # 修正输出目录为当前脚本所在目录的上一级或同级，这里使用 absolute path 确保正确
    output_dir = r'd:/workspace/ResearchAgent/core_concept_tree'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    process_json(input_file, output_dir)
