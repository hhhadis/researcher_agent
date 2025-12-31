
import json
from tqdm import tqdm

def extract_articles():
    source_path = 'd:/workspace/ResearchAgent/matched_references_train.json'
    train_data_path = 'd:/workspace/ResearchAgent/Research-14K/data/train.json'
    output_path = 'd:/workspace/ResearchAgent/found_article_titles.json'

    print(f"正在加载 {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    source_titles = set(source_data.get('matched_titles', []))
    print("加载完成。")

    print(f"正在加载 {train_data_path}...")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print("加载完成。")

    found_titles = set()
    print("开始匹配文章...")
    for article in tqdm(train_data, desc="正在处理文章"):
        if 'messages' in article and article['messages']:
            user_message = None
            for message in article['messages']:
                if message.get('role') == 'user':
                    user_message = message.get('content')
                    break
            
            if user_message:
                # 检查fuxi_titles中的任何一个标题是否存在于参考文献中
                if any(title in user_message for title in source_titles):
                    found_titles.add(article['title'])

    results = {
        "total_matches": len(found_titles),
        "matched_titles": list(found_titles)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n总共匹配到 {len(found_titles)} 篇文章。")
    print(f"匹配到的文章标题已保存到 {output_path} 文件中。")

if __name__ == '__main__':
    extract_articles()
