import json
import os
import re
from openai import OpenAI
from proposal_visualization import visualize_proposal_graph

# 设置 API Key 和 Base URL
# 在实际运行时，请确保环境变量 OPENROUTER_API_KEY 已设置
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
YOUR_SITE_URL = "https://trae.ai" 
YOUR_SITE_NAME = "ResearchAgent"

if not API_KEY:
    print("Warning: OPENROUTER_API_KEY not found in environment variables.")
    print("Please set it via: $env:OPENROUTER_API_KEY='sk-or-...'")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }
)

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_profile(researcher_data):
    """提取学者的关键研究画像 (L3, L2 & L1)"""
    hlg = researcher_data.get('hlg_data', {})
    return {
        "goals": hlg.get('Level3', [])[:10],       # Level 3: Problems/Goals
        "frameworks": hlg.get('Level2', [])[:10],  # Level 2: Frameworks
        "techniques": hlg.get('Level1', [])[:10]   # Level 1: Techniques
    }

def step1_propose_topics(r1_profile, r2_profile, cross_relations):
    """
    第一步：分析共同兴趣并提出合作议题 (Returns List of Proposals)
    """
    
    prompt = f"""
    作为一位资深的交叉学科研究顾问，请分析以下两位学者（R1 和 R2）的研究画像及他们之间潜在的关联。
    
    【学者 R1 档案】
    - 核心关注问题 (Level 3): {', '.join(r1_profile['goals'])}
    - 主要技术框架 (Level 2): {', '.join(r1_profile['frameworks'])}
    - 具体技术手段 (Level 1): {', '.join(r1_profile['techniques'])}
    
    【学者 R2 档案】
    - 核心关注问题 (Level 3): {', '.join(r2_profile['goals'])}
    - 主要技术框架 (Level 2): {', '.join(r2_profile['frameworks'])}
    - 具体技术手段 (Level 1): {', '.join(r2_profile['techniques'])}
    
    【已识别的潜在关联 (Cross-Researcher Relations)】
    {json.dumps(cross_relations, indent=2, ensure_ascii=False)}
    
    请基于上述关联提出 2-3 个合作研究议题。
    
    **重要：请务必以严格的 JSON 格式输出，不要包含 Markdown 代码块标记（如 ```json），直接输出 JSON 字符串。**
    
    JSON 格式要求如下：
    [
      {{
        "title": "议题名称（尽量简短，适合做文件夹名）",
        "description": "议题的详细描述，包括研究背景、必要性和预期创新点。",
        "sub_concepts": ["子概念1", "子概念2"], 
        "r1_nodes": [
          {{"name": "R1相关节点1", "level": "Level3"}},
          {{"name": "R1相关节点2", "level": "Level1"}}
        ],
        "r2_nodes": [
          {{"name": "R2相关节点1", "level": "Level2"}},
          {{"name": "R2相关节点2", "level": "Level1"}}
        ],
        "connections": [
          {{"source": "R1相关节点1", "target": "子概念1"}},
          {{"source": "R2相关节点1", "target": "子概念1"}},
          {{"source": "子概念1", "target": "议题名称"}},
          {{"source": "R1相关节点2", "target": "议题名称"}} 
        ]
      }}
    ]
    
    要求：
    1. "sub_concepts" 必须严格选自提供的【已识别的潜在关联 (Cross-Researcher Relations)】列表。
    2. **关键规则（必须严格遵守）：**
       - 列表中的**每一个** "sub_concept" 都必须充当连接桥梁。
       - 对于每个子概念，**必须**找出至少一个 R1 节点和一个 R2 节点与其相连。
       - 连接方向必须体现为：`R1_Node -> Sub_concept` 和 `R2_Node -> Sub_concept`，最终汇聚到 `Sub_concept -> Title`。
       - **禁止**出现悬空的子概念（即只连接了一方或者都没有连接）。
    3. **增加直接支持节点：** 除了上述连接子概念的节点外，请为 R1 和 R2 各自额外找出 **4-6 个** 重要的支持节点（Level1/2/3均可）。
       - 这些节点不通过子概念，而是**直接连接**到主议题：Researcher Node -> Title。
       - 目的是展示双方在该议题上广泛的技术储备和多维度的贡献。
    4. "r1_nodes" 和 "r2_nodes" 必须是上述档案中存在的真实节点，"level" 必须是 Level1, Level2 或 Level3。
    """
    
    print("\n--- Step 1: Generating Research Proposals (JSON) ---\n")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "你是一位专业的交叉学科研究顾问。请只输出 JSON 数据，不要输出任何其他文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up if markdown blocks are present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        print("Received response length:", len(content))
        
        try:
            proposals = json.loads(content)
            return proposals
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print("Raw Content:", content)
            return None
            
    except Exception as e:
        print(f"Error in Step 1: {e}")
        return None

def step2_personalized_intro(proposal, r1_profile, r2_profile, output_dir):
    """
    第二步：分别为 R1 和 R2 生成个性化介绍 (针对单个议题)
    """
    
    print(f"\n--- Step 2: Generating Personalized Introductions for '{proposal['title']}' ---\n")
    
    # Generate R1 Intro
    _generate_single_intro("R1", r1_profile, "R2", r2_profile, proposal, output_dir)
    
    # Generate R2 Intro
    _generate_single_intro("R2", r2_profile, "R1", r1_profile, proposal, output_dir)

def _generate_single_intro(target_r, target_profile, partner_r, partner_profile, proposal, output_dir):
    prompt = f"""
    基于以下合作研究提案：
    
    **标题**: {proposal['title']}
    **描述**: {proposal['description']}
    
    请为学者 {target_r} 写一份个性化的备忘录。
    
    【{target_r} 的背景】
    - 关注: {', '.join(target_profile['goals'])}
    - 擅长: {', '.join(target_profile['frameworks'])}
    
    【合作伙伴 {partner_r} 的优势】
    - 擅长: {', '.join(partner_profile['frameworks'])}
    
    请包含以下内容：
    1. **核心价值**：为什么这个特定的议题对 {target_r} 如此重要？
    2. **合作优势**：重点介绍 {partner_r} 的技术优势如何在这个议题中弥补 {target_r} 的短板。
    3. **行动建议**：建议下一步的具体技术对接点。
    
    语气要专业、鼓舞人心且具有洞察力。
    """
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "你是一位专业的学术合作经纪人，擅长向学者推销跨学科合作机会。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        
        # Save to file
        filename = f"Intro_for_{target_r}.md"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Personalized Introduction for {target_r}\n\n")
            f.write(f"## Proposal: {proposal['title']}\n\n")
            f.write(content + "\n\n")
        print(f"Saved introduction to {file_path}")
            
    except Exception as e:
        print(f"Error generating intro for {target_r}: {e}")

def main():
    json_path = 'd:/workspace/ResearchAgent/cross_paper_analysis/cross_于洋XiChen_1222.json'
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # Generate base output directory
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_root = os.path.join(os.path.dirname(json_path), f"{base_name}_Output")
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")
    
    # 1. Load Data
    data = load_data(json_path)
    r1_profile = extract_profile(data['researcher_1'])
    r2_profile = extract_profile(data['researcher_2'])
    cross_relations = data.get('cross_researcher_relations', [])
    
    # 2. Step 1: Propose Topics (Get JSON List)
    proposals = step1_propose_topics(r1_profile, r2_profile, cross_relations)
    
    if proposals:
        for i, p in enumerate(proposals):
            # Clean title for folder name
            safe_title = re.sub(r'[\\/*?:"<>|]', "", p['title'])
            topic_dir = os.path.join(output_root, f"Topic_{i+1}_{safe_title}")
            
            if not os.path.exists(topic_dir):
                os.makedirs(topic_dir)
                
            print(f"\nProcessing Topic {i+1}: {p['title']}")
            print(f"Directory: {topic_dir}")
            
            # Save Proposal Details
            with open(os.path.join(topic_dir, "Proposal_Details.md"), 'w', encoding='utf-8') as f:
                f.write(f"# {p['title']}\n\n")
                f.write(f"{p['description']}\n\n")
                f.write(f"**Sub-concepts**: {', '.join(p['sub_concepts'])}\n")
            
            # Generate Visualization
            img_path = os.path.join(topic_dir, "Structure_Graph.png")
            visualize_proposal_graph(p, img_path)
            
            # 3. Step 2: Personalized Intro (Per Topic)
            step2_personalized_intro(p, r1_profile, r2_profile, topic_dir)

if __name__ == "__main__":
    main()
