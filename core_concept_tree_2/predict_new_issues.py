
import os
import re
import json
import networkx as nx
from zhipuai import ZhipuAI
from collections import defaultdict
from tqdm import tqdm
import create_network

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

def get_client():
    return ZhipuAI(api_key=ZHIPUAI_API_KEY)

# Deprecated: extract_paper_ids is no longer needed as build_concept_graph returns structured data
# def extract_paper_ids(node_papers_dict): ...

def get_linked_nodes(source_node, source_pid_map, target_pid_map):
    """
    Finds nodes in the target set that share papers with the source_node.
    Returns list of (target_node, overlap_count) sorted by count.
    """
    if source_node not in source_pid_map:
        return []
        
    source_pids = source_pid_map[source_node]
    links = []
    
    for target_node, target_pids in target_pid_map.items():
        overlap = len(source_pids.intersection(target_pids))
        if overlap > 0:
            links.append((target_node, overlap))
            
    # Sort by overlap count desc
    links.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in links]

def get_lineage(G, node):
    """
    Returns a path from a root to the given node.
    Since the graph is constructed as a tree (each node has at most 1 parent in our logic),
    we can just traverse predecessors.
    Returns list: [Root, ..., Parent, Node]
    """
    path = [node]
    curr = node
    visited = set([node])
    
    while True:
        preds = list(G.predecessors(curr))
        if not preds:
            break
        # Take the first parent (should be only one or strongest)
        parent = preds[0]
        if parent in visited: # Cycle protection
            break
        visited.add(parent)
        path.append(parent)
        curr = parent
        
    return list(reversed(path))

def predict_issues_for_topic(topic_dir, client, output_file):
    topic_name = os.path.basename(topic_dir)
    print(f"Processing topic: {topic_name}")
    
    # 1. Build Trees
    print("  Building Technical Tree...")
    G_tech, _, tech_papers, tech_pid_map = create_network.build_concept_graph(topic_dir, 'technical')
    print("  Building Ethical Tree...")
    G_eth, _, eth_papers, eth_pid_map = create_network.build_concept_graph(topic_dir, 'ethical')
    
    if not G_tech or G_tech.number_of_nodes() == 0:
        print("  No technical graph found. Skipping.")
        return
    if not G_eth or G_eth.number_of_nodes() == 0:
        print("  No ethical graph found. Skipping.")
        return

    # 2. Map PIDs (No longer need regex extraction)
    # tech_pid_map and eth_pid_map are already dicts of {keyword: set(paper_ids)}
    
    # 3. Identify Frontiers (Leaf nodes in Tech Tree)
    # Filter for leaves that have reasonable significance (e.g., node count or papers)
    # For now, take top 5 leaves by paper_count
    leaves = [n for n in G_tech.nodes() if G_tech.out_degree(n) == 0]
    
    # Sort leaves by paper count (stored in G_tech nodes)
    # Note: create_network stores 'paper_count' in node attributes
    leaves.sort(key=lambda n: G_tech.nodes[n].get('paper_count', 0), reverse=True)
    
    # Limit to top 5 to avoid excessive API usage
    top_leaves = leaves[:5]
    print(f"  Selected top {len(top_leaves)} technical frontiers: {top_leaves}")
    
    results = []
    
    for leaf in tqdm(top_leaves, desc="Generating predictions"):
        # Get lineage
        lineage = get_lineage(G_tech, leaf)
        lineage_str = " -> ".join(lineage)
        
        # Build Context String
        context_desc = []
        
        # For each step in lineage, find associated ethical issues
        # We assume the lineage represents time/evolution
        for tech_node in lineage:
            year = G_tech.nodes[tech_node].get('year', 'Unknown')
            linked_ethics = get_linked_nodes(tech_node, tech_pid_map, eth_pid_map)
            
            # Filter linked_ethics to those present in G_eth to ensure they are valid nodes
            valid_ethics = [e for e in linked_ethics if e in G_eth]
            
            if valid_ethics:
                top_ethics = valid_ethics[:3] # Top 3
                eth_context_str = ", ".join(top_ethics)
                context_desc.append(f"- **{tech_node}** ({year}): Associated with [{eth_context_str}]")
                
                # Optional: Add lineage of these ethical nodes?
                # For brevity, we stick to the co-occurrence.
            else:
                context_desc.append(f"- **{tech_node}** ({year}): No strong ethical associations found.")
                
        context_text = "\n".join(context_desc)
        
        # Construct Prompt
        prompt = f"""
You are an expert researcher in Artificial Intelligence and Ethics.
We have analyzed a large corpus of research papers and constructed "Concept Trees" for both Technical Concepts and Ethical Issues.

We have identified a specific evolutionary path of technology:
{lineage_str}

Here is the history of Ethical Issues associated with this technical path:
{context_text}

Task:
Based on this trajectory, please infer and predict:
1. **New Ethical Issues**: What NEW ethical challenges might arise from the latest technology '{leaf}' or its immediate successors? (Issues that are not just repetitions of the past, but emergent problems).
2. **New Technical Solutions**: What NEW technical concepts or methods might emerge to address the ethical issues associated with '{leaf}'?

Please provide the output in the following JSON format:
{{
  "technical_concept": "{leaf}",
  "predicted_new_ethical_issues": [
    {{
      "issue": "Name of the issue",
      "description": "Brief explanation of why this arises from {leaf}"
    }}
  ],
  "predicted_new_technical_solutions": [
    {{
      "solution": "Name of the solution",
      "description": "Brief explanation of how this solves ethical problems of {leaf}"
    }}
  ]
}}
"""
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            # Clean json block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            data = json.loads(content.strip())
            results.append(data)
            
        except Exception as e:
            print(f"Error processing {leaf}: {e}")
            
    # Save results
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved predictions to {output_file}")
        
        # Also append to a master markdown file for the user
        md_file = output_file.replace('.json', '.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Predictions for Topic: {topic_name}\n\n")
            for item in results:
                f.write(f"## Tech Frontier: {item['technical_concept']}\n\n")
                f.write("### Predicted New Ethical Issues\n")
                for issue in item.get('predicted_new_ethical_issues', []):
                    f.write(f"- **{issue['issue']}**: {issue['description']}\n")
                f.write("\n### Predicted New Technical Solutions\n")
                for sol in item.get('predicted_new_technical_solutions', []):
                    f.write(f"- **{sol['solution']}**: {sol['description']}\n")
                f.write("\n---\n")

def main():
    base_dir = r"d:\workspace\ResearchAgent\core_concept_tree_2"
    client = get_client()
    
    # Find all topics
    topic_dirs = [d for d in os.listdir(base_dir) if d.startswith("topic_") and os.path.isdir(os.path.join(base_dir, d))]
    topic_dirs.sort()
    
    print(f"Found {len(topic_dirs)} topics.")
    
    output_dir = os.path.join(base_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    for d in topic_dirs:
        topic_path = os.path.join(base_dir, d)
        output_file = os.path.join(output_dir, f"prediction_{d}.json")
        predict_issues_for_topic(topic_path, client, output_file)

if __name__ == "__main__":
    main()
