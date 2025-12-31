import json
from collections import defaultdict, deque
from tqdm import tqdm

def build_citation_chain_bfs():
    """
    使用广度优先搜索（BFS）构建引用链。
    - 第0层: Fuxi参考文献。
    - 第1层: 引用了第0层文献的train.json文章。
    - 第n层: 引用了第n-1层文献的train.json文章。
    """
    print("开始使用BFS构建引用链...")

    # 1. 加载数据
    print("加载 fuxi_extreme_references.json...")
    with open('d:/workspace/ResearchAgent/data/fuxi_extreme_references.json', 'r', encoding='utf-8') as f:
        fuxi_references = json.load(f)
    layer_0_titles = {ref['title'].lower().strip() for ref in fuxi_references}

    print("加载 train.json...")
    with open('d:/workspace/ResearchAgent/Research-14K/data/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 2. 预处理，构建train.json内部的引用地图 (citing -> cited)
    print("预处理train.json，构建引用地图...")
    all_train_titles = {article['title'].lower().strip() for article in train_data}
    forward_citation_map = defaultdict(set)
    train_article_messages = {}

    for article in tqdm(train_data, desc="构建引用地图"):
        title = article['title'].lower().strip()
        user_message = ""
        if 'messages' in article and article['messages']:
            for message in article['messages']:
                if message.get('role') == 'user':
                    user_message = message.get('content', '').lower()
                    break
        train_article_messages[title] = user_message
        
        if user_message:
            # 这是一个耗时的操作，但对于精确构建图是必要的
            cited_titles_in_message = {t for t in all_train_titles if t in user_message and t != title}
            if cited_titles_in_message:
                forward_citation_map[title].update(cited_titles_in_message)

    # 在此处添加保存地图的逻辑
    map_output_path = 'd:/workspace/ResearchAgent/citation_map.json'
    print(f"引用地图构建完成，正在保存到 {map_output_path}...")
    # 将 set 转换为 list 以便 JSON 序列化
    serializable_map = {k: list(v) for k, v in forward_citation_map.items()}
    with open(map_output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_map, f, indent=2, ensure_ascii=False)
    print("引用地图已保存。")

    # 3. BFS构建层级
    print("开始通过BFS构建层级...")
    layers = defaultdict(list)
    layers[0] = list(layer_0_titles)
    processed_titles = set(layer_0_titles)
    
    # 构建第1层 (特殊处理，连接Fuxi文献和train.json)
    layer_1_titles = set()
    for article_title, user_message in tqdm(train_article_messages.items(), desc="构建第1层"):
        if any(l0_title in user_message for l0_title in layers[0]):
            if article_title not in processed_titles:
                layer_1_titles.add(article_title)
    
    layers[1] = list(layer_1_titles)
    for title in layers[1]:
        processed_titles.add(title)

    # 构建后续层级
    current_layer_num = 1
    max_layers = 10 # 防止无限循环
    while current_layer_num < max_layers:
        prev_layer_titles = layers[current_layer_num]
        if not prev_layer_titles:
            print(f"第{current_layer_num}层没有文献，停止构建。")
            break
        
        next_layer_titles = set()
        for title in tqdm(prev_layer_titles, desc=f"构建第{current_layer_num + 1}层"):
            # 从预处理的地图中查找下一层
            cited_by_current = forward_citation_map.get(title, set())
            for cited_title in cited_by_current:
                if cited_title not in processed_titles:
                    next_layer_titles.add(cited_title)
        
        if not next_layer_titles:
            print(f"第{current_layer_num + 1}层没有新文献，停止构建。")
            break
            
        current_layer_num += 1
        layers[current_layer_num] = list(next_layer_titles)
        for title in next_layer_titles:
            processed_titles.add(title)

    # 4. 准备并保存结果
    layer_counts = {f"layer_{i}": len(titles) for i, titles in layers.items()}
    result = {
        "total_layers": len(layers),
        "total_articles_in_chain": len(processed_titles),
        "layer_counts": layer_counts
    }

    output_path = 'd:/workspace/ResearchAgent/reference_chain_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n--- 引用链分析完成 ---")
    for layer, count in layer_counts.items():
        print(f"{layer}: {count} 篇文献")
    print(f"结果已保存到 {output_path}")

if __name__ == '__main__':
    build_citation_chain_bfs()
