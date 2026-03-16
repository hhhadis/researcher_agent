import os
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pypdf import PdfReader
from zhipuai import ZhipuAI
from openai import OpenAI

# API Key Loading
def load_api_keys():
    keys = {}
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        key_file_path = os.path.join(project_root, 'API_KEY.md')
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Load Zhipu Key
                match_zhipu = re.search(r'ZHIPUAI_API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match_zhipu:
                    keys['zhipu'] = match_zhipu.group(1)
                
                # Load OpenRouter/Gemini Key
                match_or = re.search(r'API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match_or:
                    keys['gemini'] = match_or.group(1)
    except Exception:
        pass
    
    # Fallback to env vars
    if 'zhipu' not in keys:
        keys['zhipu'] = os.environ.get("ZHIPUAI_API_KEY")
    if 'gemini' not in keys:
        keys['gemini'] = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
        
    return keys

keys = load_api_keys()
if not keys.get('zhipu'):
    print("Error: ZHIPUAI_API_KEY not found.")
    exit(1)
if not keys.get('gemini'):
    print("Error: GEMINI_API_KEY (or API_KEY) not found.")
    exit(1)

# Clients
zhipu_client = ZhipuAI(api_key=keys['zhipu'])
gemini_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=keys['gemini'],
)

def extract_text_from_pdf(pdf_path, max_pages=5):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Limit to first few pages to capture abstract and introduction
        num_pages = min(len(reader.pages), max_pages)
        for i in range(num_pages):
            page = reader.pages[i]
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        # print(f"Error reading {pdf_path}: {e}")
        return ""

def extract_hierarchical_findings(text, filename):
    if not text or not text.strip():
        return None
    
    # Step 1: Identify Research Questions
    step1_prompt = f"""
    You are an expert research analyst. Analyze the following text (from a research report titled "{filename}") and identify the core research questions being addressed.
    
    Text content (excerpt):
    {text[:8000]} 
    
    Output Format:
    Return a JSON list of strings, where each string is a specific research question.
    Example:
    ["What is the impact of X on Y?", "How does Z affect W?"]
    
    Do not include any other text, only the JSON list.
    """
    
    questions = []
    # Retry logic for Step 1
    for attempt in range(3):
        try:
            response = zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "user", "content": step1_prompt}
                ],
                stream=False
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            if "[" in content and "]" in content:
                 content = content[content.find("["):content.rfind("]")+1]
            questions = json.loads(content)
            break
        except Exception as e:
            # print(f"Error extracting questions for {filename} (Attempt {attempt+1}): {e}")
            time.sleep(1)
            
    if not questions or not isinstance(questions, list):
        return None

    # Step 2: Answer Questions with Direct Quotes
    step2_prompt = f"""
    You are a rigorous research assistant. You have a text and a list of research questions.
    
    Text content (excerpt):
    {text[:8000]}
    
    Research Questions:
    {json.dumps(questions, ensure_ascii=False)}
    
    Task:
    For each research question, determine the answer (Argument) based *strictly* on the provided text.
    Crucially, you must extract the **EXACT SENTENCES** (Direct Quotes) from the text that serve as evidence for the argument.
    
    Output Format:
    Return a JSON list of objects matching this structure:
    [
      {{
        "question": "The research question from the list",
        "arguments": [
          {{
            "claim": "The answer/argument derived from the text.",
            "evidence": [
              "Direct quote 1 from text.",
              "Direct quote 2 from text."
            ]
          }}
        ]
      }}
    ]
    
    Constraints:
    1. **Evidence** must be verbatim quotes from the text. Do not summarize or paraphrase the evidence.
    2. If a question is not answered in the text, do not include it in the output.
    3. Ensure the 'claim' directly answers the 'question'.
    
    Do not include any other text, only the JSON list.
    """
    
    # Retry logic for Step 2
    for attempt in range(3):
        try:
            # Step 2: Use ZhipuAI (GLM-4-Flash)
            response = zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "user", "content": step2_prompt}
                ],
                stream=False
            )
            content = response.choices[0].message.content.strip()
            # Clean up code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            if "[" in content and "]" in content:
                 content = content[content.find("["):content.rfind("]")+1]
            # Validate JSON
            json.loads(content)
            return content
        except Exception as e:
            # print(f"Error extracting findings for {filename} (Attempt {attempt+1}): {e}")
            time.sleep(2) # Wait a bit before retry
            
    return None

def cluster_documents(filenames):
    """Clusters documents based on their filenames/titles using Gemini."""
    prompt = f"""
    You are a research librarian. I have a list of {len(filenames)} research report filenames.
    Please group them into 5-10 meaningful semantic clusters (themes) based on their titles.
    
    Filenames:
    {json.dumps(filenames, ensure_ascii=False)}
    
    Output Format:
    Return a JSON object where keys are "Cluster Name" and values are lists of filenames belonging to that cluster.
    Example:
    {{
      "Renewable Energy Trends": ["report1.pdf", "report2.pdf"],
      "Energy Storage": ["report3.pdf"]
    }}
    Do not include any other text, only the JSON.
    """
    response = call_gemini_with_retry(prompt)
    if not response:
        return None
        
    # Attempt to extract JSON using regex
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return response
    except:
        return response

def merge_cluster_findings(cluster_name, cluster_findings):
    """Merges semantic duplicates within a cluster using Gemini."""
    # cluster_findings is a list of {"filename": ..., "findings": [question_objects]}
    
    prompt = f"""
    You are a Knowledge Graph expert. I have extracted research findings from multiple reports related to the theme "{cluster_name}".
    
    Your task is to MERGE semantically identical or highly similar findings into a single consolidated knowledge graph structure.
    
    Input Data (List of findings per file):
    {json.dumps(cluster_findings, ensure_ascii=False)}
    
    Instructions:
    1. **Merge Questions**: If two reports ask the same research question (even if phrased differently), merge them into a single "Global Question Node".
    2. **Merge Arguments**: If the answers/arguments to a merged question are semantically identical, merge them.
    3. **Track Sources**: For every merged Question and Argument, maintain a list of "source_files".
    4. **Highlight Consensus**: If an Argument is supported by 3 or more sources, mark it as "consensus": true.
    5. **EVIDENCE**: Select the top 2 strongest pieces of evidence. Preserve the original phrasing/quotes where possible.
    
    Output Format:
    Return a valid JSON object. ensure the JSON is complete and valid.
    {{
      "cluster_name": "{cluster_name}",
      "questions": [
        {{
          "id": "Q1",
          "text": "Merged Question Text",
          "source_files": ["file1.pdf", "file2.pdf"],
          "arguments": [
            {{
              "id": "A1",
              "text": "Merged Argument Text",
              "consensus": true/false,
              "source_files": ["file1.pdf", "file2.pdf"],
              "evidence": [
                {{
                  "text": "Evidence 1",
                  "source": "file1.pdf"
                }}
              ]
            }}
          ]
        }}
      ]
    }}
    
    Do not include any other text, only the JSON.
    """
    # Use Gemini 2.0 Flash for its large context window
    return call_gemini_with_retry(prompt)

import matplotlib.pyplot as plt
import networkx as nx

def generate_static_cluster_images(all_clusters_data, output_dir):
    """Generates static PNG images for each cluster using NetworkX and Matplotlib."""
    
    # Create images directory
    images_dir = os.path.join(output_dir, "cluster_images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    print(f"Generating images in {images_dir}...")
    
    # Color scheme
    colors = {
        "Cluster": "#FFD700", # Gold
        "Question": "#87CEFA", # LightSkyBlue
        "Argument": "#90EE90", # LightGreen
        "Consensus": "#FF6347", # Tomato
        "Evidence": "#D3D3D3"  # LightGray
    }
    
    for i, cluster_data in enumerate(all_clusters_data):
        if not cluster_data: continue
        
        try:
            data = json.loads(cluster_data) if isinstance(cluster_data, str) else cluster_data
            cluster_name = data.get("cluster_name", f"Cluster_{i+1}")
            safe_name = re.sub(r'[\\/*?:"<>|]', "", cluster_name).replace(" ", "_")
            
            G = nx.DiGraph()
            node_colors = []
            node_sizes = []
            labels = {}
            
            # Cluster Node
            c_id = "root"
            G.add_node(c_id, layer=0)
            node_colors.append(colors["Cluster"])
            node_sizes.append(3000)
            labels[c_id] = cluster_name
            
            for q_idx, q in enumerate(data.get("questions", [])):
                q_id = f"Q_{q_idx}"
                q_text = q.get("text", "Question")
                q_label = (q_text[:20] + '...') if len(q_text) > 20 else q_text
                
                G.add_node(q_id, layer=1)
                G.add_edge(c_id, q_id)
                node_colors.append(colors["Question"])
                node_sizes.append(2000)
                labels[q_id] = q_label
                
                for a_idx, arg in enumerate(q.get("arguments", [])):
                    a_id = f"A_{q_idx}_{a_idx}"
                    a_text = arg.get("text", "Argument")
                    consensus = arg.get("consensus", False)
                    a_label = (a_text[:20] + '...') if len(a_text) > 20 else a_text
                    
                    G.add_node(a_id, layer=2)
                    G.add_edge(q_id, a_id)
                    node_colors.append(colors["Consensus"] if consensus else colors["Argument"])
                    node_sizes.append(1500 if consensus else 1000)
                    labels[a_id] = a_label
                    
                    for e_idx, ev in enumerate(arg.get("evidence", [])):
                        e_id = f"E_{q_idx}_{a_idx}_{e_idx}"
                        G.add_node(e_id, layer=3)
                        G.add_edge(a_id, e_id)
                        node_colors.append(colors["Evidence"])
                        node_sizes.append(300)
                        labels[e_id] = "" # No label for evidence to save space

            plt.figure(figsize=(12, 8))
            
            # Try multipartite layout for hierarchy
            try:
                pos = nx.multipartite_layout(G, subset_key="layer")
            except:
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
            nx.draw(G, pos, 
                    node_color=node_colors, 
                    node_size=node_sizes, 
                    with_labels=True, 
                    labels=labels,
                    font_size=8,
                    edge_color="gray", 
                    arrows=True,
                    alpha=0.9)
            
            plt.title(f"Cluster: {cluster_name}", fontsize=15)
            plt.axis('off')
            
            output_path = os.path.join(images_dir, f"{safe_name}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {output_path}")
            
        except Exception as e:
            print(f"Error generating image for cluster {cluster_name}: {e}")
            continue





def call_gemini_with_retry(prompt, model="google/gemini-2.0-flash-001", retries=3, delay=60):
    # Skip Gemini if blocked (region issue)
    # return call_zhipu_fallback(prompt)

    # Try Gemini first
    for attempt in range(retries):
        try:
            # Step 3 & 4: Use Gemini via OpenRouter
            response = gemini_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg and "region" in error_msg.lower():
                print("Gemini blocked in region. Switching to ZhipuAI...")
                return call_zhipu_fallback(prompt)
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                print(f"Gemini Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{retries}...")
                time.sleep(delay)
            elif "401" in error_msg or "User not found" in error_msg:
                print("Gemini API Error: Unauthorized (401). Falling back to ZhipuAI...")
                return call_zhipu_fallback(prompt)
            else:
                print(f"Error calling Gemini API: {e}")
                time.sleep(5)
    
    print("Gemini failed after retries. Falling back to ZhipuAI...")
    return call_zhipu_fallback(prompt)

def call_zhipu_fallback(prompt):
    try:
        # Add delay to avoid rate limits
        time.sleep(2)
        response = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling ZhipuAI fallback: {e}")
        return None

def construct_knowledge_graph(all_triples):
    # Use full context for Gemini 2.0 Flash (supports 1M tokens)
    context_text = all_triples
    print(f"Sending {len(context_text)} characters to Gemini for KG construction...")
    
    prompt = f"""
    You are constructing a research Knowledge Graph based on extracted findings.
    Here is a list of research triples (Question, Argument, Evidence) extracted from various reports:
    
    {context_text}
    
    (List might be truncated if extremely long)
    
    Based on the method of "Semantic Aggregation" (inspired by LeanRAG), your task is to:
    1. **Cluster Triples**: Group these triples into semantic clusters (high-level themes or domains).
    2. **Identify Relations**: Identify explicit relationships between these clusters (e.g., "Supports", "Contradicts", "Extends", "Is a prerequisite for").
    3. **Construct Hierarchy**: Organize the knowledge into a hierarchy (e.g., Domain -> Theme -> Specific Finding).
    
    Output the Knowledge Graph structure in Markdown format.
    
    Output Format:
    # Knowledge Graph Structure

    ## Domain: [Domain Name]
    ### Theme: [Theme Name]
    - **Core Concept**: [Brief description]
    - **Key Findings**:
      - [Question]: [Argument] (Evidence: [Brief Evidence]) - [Source Report Filename if available]
    - **Relations**:
      - Connected to [Other Theme] via [Relationship Type] because [Reason]
    
    ... (Repeat for all major themes)
    
    ## Global Relationships
    - [Theme A] -> [Theme B]: [Relationship Description]
    """
    
    return call_gemini_with_retry(prompt)

def generate_landscape_report(kg_structure):
    prompt = f"""
    You are writing a comprehensive "Research Landscape Report" based on a constructed Knowledge Graph.
    
    The Knowledge Graph structure is provided below:
    {kg_structure}
    
    Please write a structured report that:
    1. **Executive Summary**: Overview of the main research domains and key themes identified.
    2. **Detailed Analysis**: For each major domain/theme, synthesize the arguments and evidence. Highlight consensus and debates.
    3. **Knowledge Gaps**: Identify areas where the graph is sparse or where questions remain unanswered (based on the lack of evidence or conflicting arguments).
    4. **Interconnections**: Discuss how different research areas relate to each other (e.g., how policy impacts technology, or how economic factors drive adoption).
    5. **Conclusion**: Summary of the state of the field.

    The report should be in Markdown format.
    """
    
    return call_gemini_with_retry(prompt)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate Research Report")
    parser.add_argument("--base_dir", default=r"d:\workspace\ResearchAgent\智慧体研究-报告", help="Directory containing PDF files")
    parser.add_argument("--output_dir", default=r"d:\workspace\ResearchAgent\report_generation", help="Directory for output files")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir
    
    hierarchical_jsonl = os.path.join(output_dir, "extracted_hierarchical.jsonl")
    clusters_file = os.path.join(output_dir, "clusters.json")
    merged_graph_file = os.path.join(output_dir, "merged_graph.json")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Find PDF files
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    all_pdf_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                all_pdf_files.append(os.path.join(root, file))
    
    print(f"Found total {len(all_pdf_files)} PDF files.")
    
    # 2. Cluster Documents
    clusters = {}
    if os.path.exists(clusters_file):
        print("Loading existing clusters...")
        with open(clusters_file, "r", encoding="utf-8") as f:
            clusters = json.load(f)
    else:
        print("Clustering documents by title...")
        filenames = [os.path.basename(f) for f in all_pdf_files]
        # Batch clustering if too many files (Gemini Flash can handle 325 easily though)
        cluster_json_str = cluster_documents(filenames)
        if cluster_json_str:
            try:
                # Cleanup json string
                cluster_json_str = cluster_json_str.replace("```json", "").replace("```", "").strip()
                clusters = json.loads(cluster_json_str)
                with open(clusters_file, "w", encoding="utf-8") as f:
                    json.dump(clusters, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error parsing clusters: {e}")
                return

    print(f"Identified {len(clusters)} clusters: {list(clusters.keys())}")

    # 3. Process Files (Hierarchical Extraction)
    processed_files = set()
    extracted_data_map = {} # filename -> data
    
    if os.path.exists(hierarchical_jsonl):
        print("Loading existing extracted data...")
        with open(hierarchical_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_files.add(data['filename'])
                    extracted_data_map[data['filename']] = data['findings']
                except:
                    pass
    
    # Map filenames to full paths
    filename_to_path = {os.path.basename(f): f for f in all_pdf_files}
    
    files_to_process = []
    for cluster_name, file_list in clusters.items():
        for fname in file_list:
            if fname not in processed_files and fname in filename_to_path:
                files_to_process.append(filename_to_path[fname])
    
    # Remove duplicates
    files_to_process = list(set(files_to_process))
    print(f"Files remaining to process: {len(files_to_process)}")

    if files_to_process:
        print("Step 1: Extracting text from PDFs...")
        newly_extracted_text = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(extract_text_from_pdf, pdf): pdf for pdf in files_to_process}
            
            for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Extracting text"):
                pdf_path = future_to_file[future]
                try:
                    text = future.result()
                    if text:
                        newly_extracted_text.append({
                            "filename": os.path.basename(pdf_path),
                            "text": text
                        })
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
        
        print("Step 2: Extracting hierarchical findings...")
        with open(hierarchical_jsonl, "a", encoding="utf-8") as jsonl_file:
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_data = {executor.submit(extract_hierarchical_findings, item["text"], item["filename"]): item for item in newly_extracted_text}
                
                for future in tqdm(as_completed(future_to_data), total=len(newly_extracted_text), desc="Extracting findings"):
                    data = future_to_data[future]
                    try:
                        findings_str = future.result()
                        if findings_str:
                            findings_json = json.loads(findings_str)
                            result_entry = {
                                "filename": data['filename'],
                                "findings": findings_json
                            }
                            
                            # Write to file
                            jsonl_file.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            
                            extracted_data_map[data['filename']] = findings_json
                    except Exception as e:
                        print(f"Error extracting/parsing findings for {data['filename']}: {e}")

    # 4. Merge Findings per Cluster
    print("Step 3: Merging findings within clusters...")
    merged_results = []
    
    # Load existing merged results if any
    if os.path.exists(merged_graph_file):
        try:
            with open(merged_graph_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    merged_results = json.loads(content)
                    print(f"Loaded {len(merged_results)} existing merged clusters.")
        except Exception as e:
            print(f"Warning: Could not load existing merged graph ({e}). Starting fresh.")
            merged_results = []

    for cluster_name, file_list in clusters.items():
        cluster_findings = []
        for fname in file_list:
            if fname in extracted_data_map:
                cluster_findings.append({
                    "filename": fname,
                    "findings": extracted_data_map[fname]
                })
        
        if not cluster_findings:
            continue
        
        # Check if already processed
        cluster_exists = False
        for m in merged_results:
            if m.get("cluster_name") == cluster_name:
                cluster_exists = True
                break
        if cluster_exists:
            print(f"Skipping already merged cluster: {cluster_name}")
            continue

        print(f"Merging cluster: {cluster_name} ({len(cluster_findings)} files)...")
        merged_json_str = merge_cluster_findings(cluster_name, cluster_findings)
        if merged_json_str:
            try:
                merged_json_str = merged_json_str.replace("```json", "").replace("```", "").strip()
                # Simple fix for unclosed JSON if needed
                if not merged_json_str.endswith("}"):
                    # Attempt to close simple structures
                    if merged_json_str.endswith("]"): merged_json_str += "}"
                    elif merged_json_str.endswith('"'): merged_json_str += "]}"
                    else: merged_json_str += '"}]}' # Hope for the best
                
                merged_data = json.loads(merged_json_str)
                merged_results.append(merged_data)
                
                # Incremental save
                with open(merged_graph_file, "w", encoding="utf-8") as f:
                    json.dump(merged_results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"Error parsing merged results for {cluster_name}: {e}")
        
        time.sleep(5) # Delay to be nice to API
    
    # Save merged graph
    with open(merged_graph_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    # # 5. Visualization
    # print("Step 4: Generating Visualization...")
    # generate_static_cluster_images(merged_results, output_dir)
    # print("Done!")

if __name__ == "__main__":
    main()
