import json
import os
import re
from openai import OpenAI
from paper_visualization import visualize_paper_graph

# Setup API
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
YOUR_SITE_URL = "https://trae.ai" 
YOUR_SITE_NAME = "ResearchAgent"

if not API_KEY:
    print("Warning: OPENROUTER_API_KEY not found.")

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

def extract_paper_summaries(papers_data):
    """
    Extracts concise profiles for each paper to fit in context.
    """
    summaries = []
    for p in papers_data:
        hlg = p.get('hlg_data', {})
        summary = {
            "id": p.get('id'),
            "name": p.get('name'),
            "goals_l3": hlg.get('Level3', [])[:5],
            "frameworks_l2": hlg.get('Level2', [])[:5],
            "techniques_l1": hlg.get('Level1', [])[:5]
        }
        summaries.append(summary)
    return summaries

def step1_synthesize_topics(paper_summaries, cross_relations):
    """
    Step 1: Analyze papers and relations to propose synthesized research themes.
    """
    
    # Format papers for prompt
    papers_text = ""
    for p in paper_summaries:
        papers_text += f"""
        [Paper ID: {p['id']}]
        - Name: {p['name']}
        - Core Problems (L3): {', '.join(p['goals_l3'])}
        - Frameworks (L2): {', '.join(p['frameworks_l2'])}
        - Techniques (L1): {', '.join(p['techniques_l1'])}
        """
        
    # Format relations
    relations_text = json.dumps(cross_relations, indent=2, ensure_ascii=False)
    
    prompt = f"""
    As a Senior Research Editor for a top-tier review journal, analyze the following collection of research papers and their inter-relations.
    
    Your task is to identify 2-3 **Synthesized Research Themes** that connect multiple papers.
    A "Synthesized Research Theme" is not just a summary of one paper, but a broader topic that integrates contributions from at least 2-3 papers to address a common high-level challenge.
    
    【Papers Collection】
    {papers_text}
    
    【Cross-Paper Relations】
    {relations_text}
    
    Please output the result in strict JSON format.
    
    JSON Format Requirements:
    [
      {{
        "title": "Theme Title (Concise, e.g., 'Robust High-Frequency Market Modeling')",
        "description": "A comprehensive description of this research theme and why it matters.",
        "related_papers": ["paper_1", "paper_2", ...],
        "sub_concepts": ["Concept A", "Concept B"],
        "paper_nodes": {{
          "paper_1": [{{"name": "Node from Paper 1", "level": "Level3"}}],
          "paper_2": [{{"name": "Node from Paper 2", "level": "Level2"}}]
        }},
        "connections": [
          {{"source": "Node from Paper 1", "target": "Concept A"}},
          {{"source": "Concept A", "target": "Theme Title"}}
        ]
      }}
    ]
    
    Rules:
    1. **"sub_concepts"**: Identify 3-5 key concepts that act as bridges between the papers. These can be from the "Cross-Paper Relations" or synthesized concepts.
    2. **"paper_nodes"**: For each related paper, select 2-3 specific nodes (L1/L2/L3) from their profile that contribute to this theme.
    3. **"connections"**: explicit connections showing how paper nodes support the sub-concepts, and how sub-concepts support the main theme.
    4. Ensure strict JSON validity. Do not include markdown formatting like ```json.
    """
    
    print("\n--- Step 1: Synthesizing Research Themes (JSON) ---\n")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a Research Editor. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        themes = json.loads(content)
        return themes
    except Exception as e:
        print(f"Error in Step 1: {e}")
        print("Raw content:", content if 'content' in locals() else "No content")
        return None

def step2_generate_report(theme, paper_summaries, output_dir):
    """
    Step 2: Generate a comprehensive markdown report for the theme.
    """
    print(f"\n--- Step 2: Generating Report for '{theme['title']}' ---\n")
    
    # Filter summaries for related papers
    related_ids = theme.get('related_papers', [])
    related_summaries = [p for p in paper_summaries if p['id'] in related_ids]
    
    papers_context = "\n".join([f"- {p['id']}: {p['name']} (Focus: {', '.join(p['goals_l3'][:3])})" for p in related_summaries])
    
    prompt = f"""
    Write a comprehensive **Research Synthesis Report** for the following theme:
    
    **Theme**: {theme['title']}
    **Description**: {theme['description']}
    
    **Involved Papers**:
    {papers_context}
    
    Please write a structured report (Markdown) covering:
    1. **Executive Summary**: Why is this theme important?
    2. **Integrated Methodology**: How do the different papers approach this problem? Compare and contrast their methods (Level 2/Level 1).
    3. **Key Contributions**: What specific value does each paper add to this theme?
    4. **Synthesis & Future Outlook**: Based on these papers, what are the emerging trends? What is missing? What should be the next research step?
    
    Tone: Academic, insightful, and forward-looking.
    """
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a senior academic researcher writing a review report."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        
        filename = "Theme_Report.md"
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(f"# Research Theme: {theme['title']}\n\n")
            f.write(content)
        print(f"Saved report to {os.path.join(output_dir, filename)}")
        
    except Exception as e:
        print(f"Error generating report: {e}")

def main():
    json_path = 'd:/workspace/ResearchAgent/outputs/multi-paper_analysis/multi_paper_analysis.json'
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # Output directory
    output_root = 'd:/workspace/ResearchAgent/multi_paper_analysis/Analysis_Output'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 1. Load Data
    data = load_data(json_path)
    papers = data.get('papers', [])
    relations = data.get('cross_paper_relations', [])
    
    print(f"Loaded {len(papers)} papers and {len(relations)} relations.")
    
    # 2. Extract Summaries
    summaries = extract_paper_summaries(papers)
    
    # 3. Step 1: Synthesize Topics
    themes = step1_synthesize_topics(summaries, relations)
    
    if themes:
        for i, theme in enumerate(themes):
            safe_title = re.sub(r'[\\/*?:"<>|]', "", theme['title'])
            theme_dir = os.path.join(output_root, f"Theme_{i+1}_{safe_title}")
            
            if not os.path.exists(theme_dir):
                os.makedirs(theme_dir)
                
            print(f"\nProcessing Theme {i+1}: {theme['title']}")
            
            # Save JSON details
            with open(os.path.join(theme_dir, "theme_details.json"), 'w', encoding='utf-8') as f:
                json.dump(theme, f, indent=2)
                
            # 4. Visualization
            img_path = os.path.join(theme_dir, "Theme_Graph.png")
            visualize_paper_graph(theme, img_path)
            
            # 5. Step 2: Report
            step2_generate_report(theme, summaries, theme_dir)

if __name__ == "__main__":
    main()
