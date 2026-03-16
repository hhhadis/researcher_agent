import json
import os
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from zhipuai import ZhipuAI
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm
import warnings

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
    # exit(1) # Don't exit yet, might just use OpenRouter for everything later
    
if not keys.get('openrouter'):
    print("Error: OpenRouter API_KEY not found.")
    exit(1)

zhipu_client = ZhipuAI(api_key=keys['zhipu']) if keys.get('zhipu') else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=keys['openrouter'],
)

def get_embedding(text):
    """Get embedding for a text string using OpenRouter (nvidia/llama-nemotron-embed-vl-1b-v2:free)."""
    try:
        # Use the multimodal input format as per user example, even for text-only
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
        print(f"Error getting embedding: {e}")
        # Fallback to standard text input if the complex structure fails
        try:
             response = openrouter_client.embeddings.create(
                model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
                input=text
            )
             return response.data[0].embedding
        except Exception as e2:
             print(f"Error getting embedding (fallback): {e2}")
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

def analyze_cluster_consistency(claims):
    """Analyze if claims in a cluster are consistent or divergent using LLM."""
    if not claims:
        return "No claims to analyze."
    
    prompt = f"""
    You are a Research Analyst. Analyze the following list of claims/viewpoints regarding a specific research topic.
    Determine if these viewpoints are **Consistent** (agreeing with each other) or **Divergent** (showing different perspectives or conflicts).
    
    Claims:
    {json.dumps(claims[:20], ensure_ascii=False)}  # Limit to 20 for context window
    
    Output a brief summary (max 100 words) describing the consensus or debate.
    Start with "CONSISTENT:" or "DIVERGENT:".
    """
    try:
        response = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing consistency: {e}"

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

def analyze_stance(claim, question):
    """Determine the stance score (-10 to 10) of a claim towards a question."""
    prompt = f"""
    Question: {question}
    Claim: {claim}
    
    Evaluate the stance of the claim regarding the question.
    Assign a **Score** as an integer from -10 to 10, where:
    - -10 represents extremely strong opposition.
    - 0 represents completely neutral or no clear stance.
    - +10 represents extremely strong support.
    
    You can use ANY integer between -10 and 10 to reflect the nuance of the stance (e.g., -2 for slight skepticism, 7 for strong but not extreme support).
    
    Return JSON ONLY:
    {{
        "score": <int>,
        "reason": "brief reason"
    }}
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
        # Ensure score is an integer within range
        score = int(result.get('score', 0))
        return {"score": max(-10, min(10, score)), "reason": result.get('reason', '')}
    except:
        return {"score": 0, "reason": "Error"}

def main():
    print("Loading data...")
    all_findings = load_data(INPUT_FILE)
    if not all_findings:
        print("No findings found.")
        return

    print(f"Loaded {len(all_findings)} findings. Generating embeddings for questions...")
    
    # 1. Embed Questions
    # To save API calls, unique questions first
    unique_questions = list(set(f['question'] for f in all_findings))
    question_embeddings = {}
    
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ... (keep existing imports)

# Improve embedding function with retry logic and parallel execution support
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
                # print(f"Failed to get embedding after {retries} attempts: {e}")
                pass
            time.sleep(1)
    return None

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
    
    # Parallel processing for embeddings
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
            
    # Filter findings with valid embeddings
    valid_findings = [f for f in all_findings if f['question'] in question_embeddings]
    
    if not valid_findings:
        print("No valid embeddings generated.")
        return

    # Prepare matrix for clustering
    X_questions = np.array([question_embeddings[f['question']] for f in valid_findings])
    
    # 2. Cluster Questions
    print("Clustering questions...")
    num_clusters = min(10, len(unique_questions) // 5 + 1) # Heuristic for K
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_questions)
    
    # Organize by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(valid_findings[idx])
        
    # 3. Analyze each cluster
    print("Analyzing clusters...")
    
    analysis_report = []
    
    for cluster_id, items in clusters.items():
        print(f"Processing Cluster {cluster_id} ({len(items)} items)...")
        
        # Get cluster topic (using LLM on questions)
        questions = list(set(item['question'] for item in items))
        claims = [item['claim'] for item in items]
        
        # Generate Topic Name
        topic_prompt = f"Summarize these research questions into a single short topic label (3-5 words):\n{json.dumps(questions[:10], ensure_ascii=False)}"
        try:
            topic_response = zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": topic_prompt}]
            )
            topic_label = topic_response.choices[0].message.content.strip()
        except:
            topic_label = f"Cluster {cluster_id}"
            
        # Analyze Consistency
        consistency_analysis = analyze_cluster_consistency(claims)
        
        # Get Representative Question for Stance Analysis
        rep_question = get_representative_question(questions)
        print(f"  Representative Question: {rep_question[:60]}...")
        
        # Dimensionality Reduction for Visualization (Claims)
        claim_embeddings = []
        valid_claims = []
        claim_scores = [] # To store stance scores
        
        # Use parallel processing for claims too
        with ThreadPoolExecutor(max_workers=5) as executor:
             future_to_item = {}
             # Stance Analysis (Parallelize)
             future_to_stance = {}
             
             # We can skip embedding if we only need scores for the 1D plot as requested.
             # However, let's keep embedding code if we want to revert or use it later, 
             # BUT for the user's specific request "draw on a number axis", we mainly need the SCORE.
             # To save time/cost, let's prioritize STANCE analysis.
             
             print(f"  Analyzing stance for {len(items)} claims...")
             for item in items:
                 future = executor.submit(analyze_stance, item['claim'], rep_question)
                 future_to_stance[future] = item
             
             for future in tqdm(as_completed(future_to_stance), total=len(items), desc=f"Cluster {cluster_id} Stance", leave=False):
                 item = future_to_stance[future]
                 try:
                    stance_result = future.result()
                    score = stance_result.get('score', 0)
                    claim_scores.append(score)
                    valid_claims.append(item['claim'])
                 except Exception as e:
                    pass

        if len(claim_scores) > 2:
            try:
                # 1D Visualization with Jitter
                plt.figure(figsize=(12, 6))
                
                # Add random jitter to Y-axis to separate points
                y_jitter = np.random.normal(0, 0.05, len(claim_scores))
                
                # Map scores to colors (Red -> Gray -> Green)
                # Normalize score -10 to 10 -> 0 to 1 for colormap
                norm_scores = [(s + 10) / 20 for s in claim_scores]
                cmap = plt.get_cmap('RdYlGn')
                colors = [cmap(s) for s in norm_scores]
                
                scatter = plt.scatter(claim_scores, y_jitter, c=colors, alpha=0.7, s=100, edgecolors='grey')
                
                # Axis formatting
                plt.xlim(-11, 11)
                plt.ylim(-0.5, 0.5)
                plt.yticks([]) # Hide Y axis
                plt.xlabel("Stance Score (-10: Oppose <---> +10: Support)", fontsize=12)
                
                # Title with Question (Wrapped)
                wrapped_title = "\n".join([rep_question[i:i+80] for i in range(0, len(rep_question), 80)])
                plt.title(f"Viewpoint Distribution: {topic_label}\nQ: {wrapped_title}", fontsize=10)
                
                # Annotate a few representative points (min, max, median)
                # Sort by score to find extremes easily
                sorted_indices = np.argsort(claim_scores)
                indices_to_annotate = [sorted_indices[0], sorted_indices[-1], sorted_indices[len(sorted_indices)//2]]
                
                for idx in indices_to_annotate:
                    txt = valid_claims[idx][:40] + "..."
                    plt.annotate(txt, (claim_scores[idx], y_jitter[idx]), 
                                 xytext=(0, 10), textcoords='offset points', 
                                 ha='center', fontsize=8,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

                plt.grid(axis='x', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}_viz.png"))
                plt.close()
            except Exception as e:
                print(f"Error plotting cluster {cluster_id}: {e}")
        
        # Calculate stats
        positive_count = sum(1 for s in claim_scores if s > 2)
        negative_count = sum(1 for s in claim_scores if s < -2)
        neutral_count = len(claim_scores) - positive_count - negative_count
        avg_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0
        
        analysis_report.append({
            "cluster_id": int(cluster_id),
            "topic": topic_label,
            "representative_question": rep_question,
            "num_items": len(items),
            "questions": questions[:5],
            "consistency_analysis": consistency_analysis,
            "stance_stats": {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "average_score": avg_score
            }
        })


    # Save Report
    with open(os.path.join(OUTPUT_DIR, "analysis_report.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        
    # Generate Markdown Summary
    md_report = "# Research Viewpoint Analysis\n\n"
    for item in analysis_report:
        md_report += f"## Topic: {item['topic']}\n"
        md_report += f"- **Items**: {item['num_items']}\n"
        md_report += f"- **Representative Question**: {item['representative_question']}\n"
        if 'stance_stats' in item:
            md_report += f"- **Stance Distribution**: Positive (>2): {item['stance_stats']['positive_count']}, Negative (<-2): {item['stance_stats']['negative_count']}, Neutral: {item['stance_stats']['neutral_count']}\n"
            md_report += f"- **Average Score**: {item['stance_stats']['average_score']:.2f}\n"
        md_report += f"- **Key Questions**: {', '.join(item['questions'][:3])}\n"
        md_report += f"- **Analysis**: {item['consistency_analysis']}\n"
        md_report += f"![Visualization](cluster_{item['cluster_id']}_viz.png)\n\n"
        
    with open(os.path.join(OUTPUT_DIR, "analysis_report.md"), 'w', encoding='utf-8') as f:
        f.write(md_report)
        
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()