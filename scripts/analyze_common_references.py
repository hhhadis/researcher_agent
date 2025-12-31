import json
from tqdm import tqdm
from collections import defaultdict, Counter

def analyze_common_references():
    # 文件路径
    fuxi_path = 'd:/workspace/ResearchAgent/data/fuxi_extreme_references.json'
    train_path = 'd:/workspace/ResearchAgent/Research-14K/data/train.json'
    
    print("正在加载fuxi_extreme_references.json...")
    with open(fuxi_path, 'r', encoding='utf-8') as f:
        fuxi_references = json.load(f)
    
    # 提取fuxi文献的标题和作者信息
    fuxi_titles = {}
    fuxi_authors = {}
    for ref in fuxi_references:
        title = ref['title'].lower().strip()
        fuxi_titles[title] = ref
        fuxi_authors[title] = ref.get('authors', [])
    
    print(f"fuxi文献总数: {len(fuxi_titles)}")
    
    print("正在加载train.json...")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"train文章总数: {len(train_data)}")
    
    # 存储匹配结果
    matched_articles = []  # 存储包含匹配文献的文章
    common_references = []  # 存储共同文献的详细信息
    reference_usage_count = Counter()  # 统计每个文献被引用的次数
    
    print("开始分析共同文献...")
    
    for article in tqdm(train_data, desc="正在处理文章"):
        article_matched_refs = []  # 这篇文章匹配到的文献
        
        if 'messages' in article and article['messages']:
            # 获取第一个user消息作为参考文献内容
            user_message = None
            for message in article['messages']:
                if message.get('role') == 'user':
                    user_message = message.get('content', '').lower()
                    break
            
            if user_message:
                # 检查每个fuxi文献是否出现在参考文献中
                for title, ref_info in fuxi_titles.items():
                    if title in user_message:
                        article_matched_refs.append(ref_info)
                        reference_usage_count[title] += 1
        
        # 如果这篇文章有匹配的文献
        if article_matched_refs:
            matched_articles.append({
                'article_title': article['title'],
                'matched_references': article_matched_refs,
                'matched_count': len(article_matched_refs)
            })
            
            # 将匹配的文献添加到共同文献列表
            for ref in article_matched_refs:
                if ref not in common_references:
                    common_references.append(ref)
    
    # 保存详细结果
    results = {
        'statistics': {
            'total_fuxi_references': len(fuxi_titles),
            'total_train_articles': len(train_data),
            'articles_with_common_references': len(matched_articles),
            'total_common_references': len(common_references),
            'matching_rate': f"{len(matched_articles)/len(train_data)*100:.2f}%"
        },
        'common_references': common_references,
        'matched_articles': matched_articles,
        'reference_usage_stats': dict(reference_usage_count)
    }
    
    # 保存详细结果到文件
    output_file = 'd:/workspace/ResearchAgent/common_references_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成专门的文献引用次数统计文件
    reference_citation_stats = []
    for title, count in reference_usage_count.items():
        # 找到对应的完整文献信息
        full_ref = next((ref for ref in common_references if ref['title'].lower() == title), None)
        if full_ref:
            citation_info = {
                'title': full_ref['title'],
                'authors': full_ref.get('authors', []),
                'year': full_ref.get('year', 'N/A'),
                'venue': full_ref.get('venue', 'N/A'),
                'citation_count': count,
                'percentage_of_total_articles': f"{count/len(train_data)*100:.2f}%"
            }
            reference_citation_stats.append(citation_info)
    
    # 按引用次数排序
    reference_citation_stats.sort(key=lambda x: x['citation_count'], reverse=True)
    
    # 保存文献引用次数统计
    citation_stats_file = 'd:/workspace/ResearchAgent/reference_citation_statistics.json'
    with open(citation_stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_train_articles': len(train_data),
            'total_common_references': len(common_references),
            'reference_citation_stats': reference_citation_stats
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n文献引用次数统计已保存到: {citation_stats_file}")
    
    # 生成统计表
    print("\n=== 文献匹配统计表 ===")
    print(f"fuxi_extreme_references.json中的文献总数: {len(fuxi_titles)}")
    print(f"train.json中的文章总数: {len(train_data)}")
    print(f"包含共同文献的文章数: {len(matched_articles)}")
    print(f"共同文献总数: {len(common_references)}")
    print(f"文章匹配率: {len(matched_articles)/len(train_data)*100:.2f}%")
    print(f"文献匹配率: {len(common_references)/len(fuxi_titles)*100:.2f}%")
    
    # 最常用的前10个文献
    print("\n=== 被引用最多的前10个文献 ===")
    top_references = reference_usage_count.most_common(10)
    for i, (title, count) in enumerate(top_references, 1):
        print(f"{i}. {title[:80]}...")
        print(f"   被引用次数: {count}")
        # 找到完整的文献信息
        full_ref = next((ref for ref in common_references if ref['title'].lower() == title), None)
        if full_ref:
            authors = ', '.join(full_ref.get('authors', [])[:3])
            if len(full_ref.get('authors', [])) > 3:
                authors += ' et al.'
            print(f"   作者: {authors}")
            print(f"   年份: {full_ref.get('year', 'N/A')}")
            print(f"   期刊: {full_ref.get('venue', 'N/A')}")
        print()
    
    # 匹配文章最多的文献
    print("\n=== 匹配文章数统计 ===")
    match_count_stats = Counter(article['matched_count'] for article in matched_articles)
    for count, num_articles in sorted(match_count_stats.items()):
        print(f"包含{count}个共同文献的文章数: {num_articles}")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    results = analyze_common_references()