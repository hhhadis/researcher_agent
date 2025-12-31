import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

def visualize_citation_graph():
    """
    可视化引用网络图。
    1. 读取保存的引用地图。
    2. 构建完整的图。
    3. 筛选出度大于等于3的节点，并创建一个子图。
    4. 使用matplotlib绘制子图并保存。
    """
    map_path = 'd:/workspace/ResearchAgent/citation_map.json'
    print(f"正在从 {map_path} 加载引用地图...")
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            citation_map = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到引用地图文件 {map_path}。请先运行build_reference_chain_simple.py生成它。")
        return

    # 构建完整的图
    G = nx.Graph()
    for citing, cited_list in citation_map.items():
        for cited in cited_list:
            G.add_edge(citing, cited)

    print(f"完整的图构建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")

    # 筛选度大于等于3的节点
    core_nodes = [node for node, degree in G.degree() if degree >= 3]
    if not core_nodes:
        print("图中没有找到度大于等于3的节点，无法生成可视化结果。")
        return
    
    # 创建只包含核心节点及其连接的子图
    core_graph = G.subgraph(core_nodes)
    print(f"已筛选出核心网络，包含 {core_graph.number_of_nodes()} 个节点和 {core_graph.number_of_edges()} 条边。")

    # 可视化
    print("开始绘制核心网络图...")
    plt.figure(figsize=(20, 20))
    
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    pos = nx.spring_layout(core_graph, k=0.6, iterations=50)
    
    nx.draw_networkx_nodes(core_graph, pos, node_size=50, node_color='skyblue')
    nx.draw_networkx_edges(core_graph, pos, alpha=0.5, edge_color='gray')
    
    # 为了避免标签重叠，只显示部分节点的标签或使用更小的字体
    # 在这里，我们选择不显示标签，因为节点太多会导致混乱
    # 如果需要显示，可以使用下面的代码并调整参数
    # nx.draw_networkx_labels(core_graph, pos, font_size=8)

    plt.title("Citation Network (Nodes with Degree >= 3)", size=25)
    plt.axis('off')
    
    output_path = 'd:/workspace/ResearchAgent/citation_graph.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"网络图已成功保存到 {output_path}")

if __name__ == '__main__':
    visualize_citation_graph()
