import json
import os
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from math import pi
from zhipuai import ZhipuAI
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_FILE = r"d:\workspace\ResearchAgent\report_generation\extracted_hierarchical.jsonl"
OUTPUT_DIR = r"d:\workspace\ResearchAgent\report_generation\analysis_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load API Keys
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
                
                # Load OpenRouter Key (API_KEY)
                match_or = re.search(r'API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match_or:
                    keys['openrouter'] = match_or.group(1)
    except:
        pass
    
    if 'zhipu' not in keys:
        keys['zhipu'] = os.environ.get("ZHIPUAI_API_KEY")
    if 'openrouter' not in keys:
        keys['openrouter'] = os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        
    return keys

keys = load_api_keys()
if not keys.get('zhipu'):
    print("Error: ZHIPUAI_API_KEY not found.")
    
if not keys.get('openrouter'):
    print("Error: OpenRouter API_KEY not found.")
    exit(1)

zhipu_client = ZhipuAI(api_key=keys['zhipu']) if keys.get('zhipu') else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=keys['openrouter'],
)

def get_embedding_with_retry(text, retries=3):
    """Get embedding with retry logic."""
    for attempt in range(retries):
        try:
            # Use the multimodal input format as per user example
            response = openrouter_client.embeddings.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/TraeAI", 
                    "X-OpenRouter-Title": "ResearchAgent"
                },
                model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
                input=[
                    {
                        "content": [
                            {"type": "text", "text": text}
                        ]
                    }
                ],
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                pass
            time.sleep(1)
    return None

def load_data(filepath):
    """Load Q&A pairs from JSONL file."""
    findings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                filename = data.get('filename', 'Unknown')
                for item in data.get('findings', []):
                    question = item.get('question', '').strip()
                    if not question: continue
                    
                    for arg in item.get('arguments', []):
                        claim = arg.get('claim', '').strip()
                        if claim:
                            findings.append({
                                'filename': filename,
                                'question': question,
                                'claim': claim,
                                'evidence': arg.get('evidence', [])
                            })
            except json.JSONDecodeError:
                continue
    return findings

def get_representative_question(questions):
    """Identify the single most critical research question from a list."""
    prompt = f"""
    You are a Senior Editor. Below is a list of research questions from a cluster of papers.
    Identify or synthesize the SINGLE most critical, central, and controversial scientific question that covers the core debate of these questions.
    
    Questions:
    {json.dumps(questions[:30], ensure_ascii=False)}
    
    Return ONLY the question string, nothing else.
    """
    try:
        response = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return questions[0] if questions else "Unknown Question"

def extract_key_viewpoints(claims, question):
    """Identify key distinct viewpoints for a given question from a sample of claims."""
    prompt = f"""
    Question: {question}
    
    Below is a sample of research claims/findings related to this question:
    {json.dumps(claims[:50], ensure_ascii=False)}
    
    Task:
    Identify the DISTINCT, MAINSTREAM viewpoints or schools of thought (perspectives) that summarize the debate or discussion.
    There should be between 3 and 6 viewpoints depending on the complexity of the topic.
    These viewpoints should be distinct dimensions (e.g., Efficiency vs. Equity, Government-led vs. Market-led, Optimistic vs. Pessimistic).
    
    Return JSON ONLY:
    [
        {{"id": "V1", "label": "Short Label 1 (3-5 words)", "description": "Brief description of viewpoint 1"}},
        {{"id": "V2", "label": "Short Label 2 (3-5 words)", "description": "Brief description of viewpoint 2"}},
        ...
    ]
    """
    try:
        response = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error extracting viewpoints: {e}")
        # Fallback
        return [
            {"id": "V1", "label": "Viewpoint 1", "description": "Generic Viewpoint 1"},
            {"id": "V2", "label": "Viewpoint 2", "description": "Generic Viewpoint 2"},
            {"id": "V3", "label": "Viewpoint 3", "description": "Generic Viewpoint 3"}
        ]

def analyze_multidimensional_stance(claim, question, viewpoints):
    """
    Analyze the stance of a claim against multiple key viewpoints.
    Returns a vector [score_1, score_2, ...] where score is -1.0 to 1.0.
    """
    viewpoint_desc = "\n".join([f"{v['id']}: {v['label']} - {v['description']}" for v in viewpoints])
    ids = [v['id'] for v in viewpoints]
    
    # Construct expected JSON structure for prompt
    json_structure = "{\n" + ",\n".join([f'        "{vid}": <float -1.0 to 1.0>' for vid in ids]) + "\n    }"
    
    prompt = f"""
    Question: {question}
    Claim: {claim}
    
    Key Mainstream Viewpoints:
    {viewpoint_desc}
    
    Task:
    Evaluate the stance of the Claim towards EACH of the Key Viewpoints.
    Assign a score from -1.0 (Strongly Oppose) to +1.0 (Strongly Support) for each viewpoint.
    0.0 means Neutral or Irrelevant.
    
    Return JSON ONLY:
    {json_structure}
    """
    try:
        response = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return [float(result.get(v['id'], 0.0)) for v in viewpoints]
    except Exception as e:
        return [0.0] * len(viewpoints)

def plot_radar_chart(cluster_id, rep_question, viewpoints, stance_matrix, output_dir):
    """
    Generates a Radar Chart (Spider Plot) summarizing the stance distribution.
    Uses K-Means to find representative stance patterns (centroids) to plot.
    """
    categories = [v['label'] for v in viewpoints]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Cluster the stance vectors to find distinct "Schools of Thought"
    num_items = len(stance_matrix)
    
    if num_items >= 10:
        # Determine optimal number of clusters (2 to 5) based on Silhouette Score
        best_k = 3
        best_score = -1
        
        # Try different k values
        possible_ks = range(2, min(6, num_items // 5 + 1))
        
        if len(possible_ks) > 0:
            for k in possible_ks:
                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                labels_temp = kmeans_temp.fit_predict(stance_matrix)
                
                # Check if valid clustering (more than 1 cluster, less than N-1)
                if len(set(labels_temp)) > 1 and len(set(labels_temp)) < num_items:
                    score = silhouette_score(stance_matrix, labels_temp)
                    if score > best_score:
                        best_score = score
                        best_k = k
        
        n_clusters = best_k
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(stance_matrix)
        centroids = kmeans.cluster_centers_
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    else:
        centroids = [np.mean(stance_matrix, axis=0)]
        cluster_sizes = [num_items]
        
    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-0.5, 0, 0.5], ["-0.5", "0", "0.5"], color="grey", size=7)
    plt.ylim(-1.0, 1.0)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each centroid
    for i, centroid in enumerate(centroids):
        values = centroid.tolist()
        values += values[:1] # Close the loop
        
        label = f"Pattern {i+1} (N={cluster_sizes[i]})"
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
        
    plt.title(f"Stance Patterns: Cluster {cluster_id}\nQ: {rep_question[:60]}...", size=11, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    output_path = os.path.join(output_dir, f"cluster_{cluster_id}_radar_viz.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved Radar Chart to {output_path}")


def main():
    print("Loading data...")
    all_findings = load_data(INPUT_FILE)
    if not all_findings:
        print("No findings found.")
        return

    print(f"Loaded {len(all_findings)} findings. Generating embeddings for questions...")
    
    # 1. Embed Questions
    unique_questions = list(set(f['question'] for f in all_findings))
    question_embeddings = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_question = {executor.submit(get_embedding_with_retry, q): q for q in unique_questions}
        
        for future in tqdm(as_completed(future_to_question), total=len(unique_questions), desc="Generating Question Embeddings"):
            q = future_to_question[future]
            try:
                emb = future.result()
                if emb:
                    question_embeddings[q] = emb
            except Exception as e:
                pass
            
    valid_findings = [f for f in all_findings if f['question'] in question_embeddings]
    
    if not valid_findings:
        print("No valid embeddings generated.")
        return

    # Prepare matrix for clustering
    X_questions = np.array([question_embeddings[f['question']] for f in valid_findings])
    
    # 2. Cluster Questions
    print("Clustering questions...")
    num_clusters = min(10, len(unique_questions) // 5 + 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_questions)
    
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(valid_findings[idx])
        
    # 3. Analyze each cluster with Multidimensional Stance
    print("Analyzing clusters with Multidimensional Stance...")
    
    analysis_report = []
    
    for cluster_id, items in clusters.items():
        print(f"Processing Cluster {cluster_id} ({len(items)} items)...")
        
        questions = list(set(item['question'] for item in items))
        claims = [item['claim'] for item in items]
        
        # Get Representative Question
        rep_question = get_representative_question(questions)
        print(f"  Representative Question: {rep_question[:60]}...")
        
        # Extract Key Viewpoints (Dynamic)
        print("  Extracting key viewpoints...")
        key_viewpoints = extract_key_viewpoints(claims, rep_question)
        print(f"  Key Viewpoints ({len(key_viewpoints)}): {[v['label'] for v in key_viewpoints]}")
        
        # Calculate Stance Vectors
        print(f"  Calculating stance vectors for {len(items)} claims...")
        stance_vectors = []
        valid_claims_for_plot = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_item = {executor.submit(analyze_multidimensional_stance, item['claim'], rep_question, key_viewpoints): item for item in items}
            
            for future in tqdm(as_completed(future_to_item), total=len(items), desc=f"Cluster {cluster_id} Vectors", leave=False):
                item = future_to_item[future]
                try:
                    vec = future.result()
                    stance_vectors.append(vec)
                    valid_claims_for_plot.append(item['claim'])
                except Exception as e:
                    pass
        
        if not stance_vectors:
            continue
            
        stance_matrix = np.array(stance_vectors)
        num_vars = stance_matrix.shape[1]
        
        # Visualization: Radar Chart (Spider Plot)
        try:
            if num_vars >= 3:
                plot_radar_chart(cluster_id, rep_question, key_viewpoints, stance_matrix, OUTPUT_DIR)
            else:
                # Fallback to 2D Scatter if less than 3 vars (Radar chart needs at least 3)
                print(f"  Only {num_vars} viewpoints, skipping Radar Chart (needs 3+). Using 2D Scatter.")
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                xs = stance_matrix[:, 0]
                ys = stance_matrix[:, 1]
                distances = np.sqrt(xs**2 + ys**2)
                
                sc = ax.scatter(xs, ys, c=distances, cmap='viridis', s=50, alpha=0.7)
                ax.set_xlabel(key_viewpoints[0]['label'])
                ax.set_ylabel(key_viewpoints[1]['label'])
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.axhline(0, color='gray', alpha=0.5)
                ax.axvline(0, color='gray', alpha=0.5)
                
                plt.colorbar(sc, label='Strength of Stance')
                plt.title(f"2D Viewpoint Distribution: Cluster {cluster_id}\nQ: {rep_question[:60]}...")
                output_path = os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}_scatter_viz.png")
                plt.savefig(output_path)
                plt.close()
                print(f"  Saved 2D Scatter to {output_path}")

        except Exception as e:
            print(f"Error plotting cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            
        analysis_report.append({
            "cluster_id": int(cluster_id),
            "representative_question": rep_question,
            "key_viewpoints": key_viewpoints,
            "num_items": len(items)
        })

    # Save Report
    json_path = os.path.join(OUTPUT_DIR, "vector_analysis_report.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        
    # Generate Markdown Report with Radar Charts
    md_path = os.path.join(OUTPUT_DIR, "vector_analysis_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Research Landscape Analysis: Stance Patterns & Viewpoints\n\n")
        f.write("This report analyzes the key research questions and the distribution of academic stances using vector-based clustering.\n\n")
        
        for cluster in analysis_report:
            cid = cluster['cluster_id']
            question = cluster['representative_question']
            num_items = cluster['num_items']
            viewpoints = cluster['key_viewpoints']
            
            f.write(f"## Cluster {cid}: {question}\n\n")
            f.write(f"**Number of Findings:** {num_items}\n\n")
            
            f.write("**Key Dimensions of Debate:**\n")
            for v in viewpoints:
                f.write(f"- **{v['label']}**: {v['description']}\n")
            f.write("\n")
            
            # Embed Radar Chart
            image_filename = f"cluster_{cid}_radar_viz.png"
            if os.path.exists(os.path.join(OUTPUT_DIR, image_filename)):
                f.write(f"### Stance Distribution (Radar Chart)\n")
                f.write(f"![Stance Distribution - Cluster {cid}]({image_filename})\n\n")
                f.write("> **Interpretation:** The radar chart visualizes the dominant stance patterns within this research cluster. "
                        "Each colored area represents a distinct 'school of thought' or pattern of viewpoints found in the literature. "
                        "The axes represent the key dimensions of debate identified above.\n\n")
            else:
                # Fallback for 2D scatter or missing image
                scatter_filename = f"cluster_{cid}_scatter_viz.png"
                if os.path.exists(os.path.join(OUTPUT_DIR, scatter_filename)):
                    f.write(f"### Stance Distribution (2D Scatter)\n")
                    f.write(f"![Stance Distribution - Cluster {cid}]({scatter_filename})\n\n")
            
            f.write("---\n\n")
            
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    print(f"Markdown report generated at {md_path}")

if __name__ == "__main__":
    main()
