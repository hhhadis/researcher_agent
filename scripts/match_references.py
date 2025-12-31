
import json
from tqdm import tqdm

def match_references(use_test_file=True):
    fuxi_references_path = 'd:/workspace/ResearchAgent/data/fuxi_extreme_references.json'
    if use_test_file:
        data_path = 'd:/workspace/ResearchAgent/Research-14K/data/test.json'
        output_path = 'd:/workspace/ResearchAgent/matched_references_test.json'
    else:
        data_path = 'd:/workspace/ResearchAgent/Research-14K/data/train.json'
        output_path = 'd:/workspace/ResearchAgent/matched_references_train.json'

    print("正在加载 fuxi_extreme_references.json...")
    with open(fuxi_references_path, 'r', encoding='utf-8') as f:
        fuxi_references = json.load(f)
    fuxi_titles = {ref['title'] for ref in fuxi_references}
    print("加载完成。")

    print(f"正在加载 {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("加载完成。")

    matched_articles = []
    print("开始匹配文章...")
    for article in tqdm(data, desc="正在处理文章"):
        if 'messages' in article and article['messages']:
            user_message = None
            for message in article['messages']:
                if message.get('role') == 'user':
                    user_message = message.get('content')
                    break
            
            if user_message:
                # 检查fuxi_titles中的任何一个标题是否存在于参考文献中
                if any(title in user_message for title in fuxi_titles):
                    matched_articles.append(article['title'])

    results = {
        "total_matches": len(matched_articles),
        "matched_titles": matched_articles
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n总共匹配到 {len(matched_articles)} 篇文章。")
    print(f"匹配到的文章标题已保存到 {output_path} 文件中。")

if __name__ == '__main__':
    # 首先使用 test.json 进行测试
    match_references(use_test_file=False)
