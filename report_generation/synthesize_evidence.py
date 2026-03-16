import json
import os
import argparse
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zhipuai import ZhipuAI
from openai import OpenAI
from collections import defaultdict
import random

# --- API Loading (Reused from generate_report.py) ---
def load_api_keys():
    keys = {}
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        key_file_path = os.path.join(project_root, 'API_KEY.md')
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match_zhipu = re.search(r'ZHIPUAI_API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match_zhipu: keys['zhipu'] = match_zhipu.group(1)
                match_or = re.search(r'API_KEY\s*=\s*["\'](.+?)["\']', content)
                if match_or: keys['gemini'] = match_or.group(1)
    except Exception: pass
    if 'zhipu' not in keys: keys['zhipu'] = os.environ.get("ZHIPUAI_API_KEY")
    if 'gemini' not in keys: keys['gemini'] = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
    return keys

keys = load_api_keys()
if not keys.get('zhipu'): print("Error: ZHIPUAI_API_KEY not found."); exit(1)
if not keys.get('gemini'): print("Error: GEMINI_API_KEY not found."); exit(1)

zhipu_client = ZhipuAI(api_key=keys['zhipu'])
gemini_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=keys['gemini'])

def call_zhipu_fallback(prompt):
    try:
        time.sleep(2)
        response = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling ZhipuAI fallback: {e}")
        return None

def call_gemini_with_retry(prompt, model="google/gemini-2.0-flash-001", retries=3, delay=60):
    for attempt in range(retries):
        try:
            response = gemini_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg and "region" in error_msg.lower():
                print("Gemini blocked in region. Switching to ZhipuAI...")
                return call_zhipu_fallback(prompt)
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                print(f"Gemini Rate limit hit. Waiting {delay} seconds...")
                time.sleep(delay)
            elif "401" in error_msg:
                return call_zhipu_fallback(prompt)
            else:
                time.sleep(5)
    return call_zhipu_fallback(prompt)

# --- Core Logic ---

def extract_evidence(jsonl_path):
    """Extracts all evidence items from the hierarchical JSONL file."""
    evidence_list = []
    print(f"Reading {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                filename = data.get('filename', 'Unknown')
                findings = data.get('findings', [])
                if not findings: continue
                
                for q in findings:
                    q_text = q.get('question', '')
                    for arg in q.get('arguments', []):
                        arg_text = arg.get('claim', '')
                        ev_items = arg.get('evidence', [])
                        
                        # Handle if evidence is string or list
                        if isinstance(ev_items, str):
                            ev_items = [ev_items]
                        
                        for ev in ev_items:
                            # Context string to help understanding
                            full_text = f"{ev}"
                            evidence_list.append({
                                "text": full_text,
                                "source": filename,
                                "context_arg": arg_text,
                                "context_q": q_text
                            })
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    return evidence_list

def identify_core_topics(evidence_list, num_topics=20):
    """
    Identifies core topics by sampling evidence and asking the LLM to generate potential research themes.
    """
    # Sample a diverse set of evidence to find topics
    sample_size = min(len(evidence_list), 100)
    sample_evidence = random.sample(evidence_list, sample_size)
    
    evidence_text = "\n".join([f"- {item['text']}" for item in sample_evidence])
    
    prompt = f"""
    You are a Research Topic Analyzer. I have a large dataset of extracted evidence from various research papers.
    Here is a random sample of that evidence:
    
    {evidence_text}
    
    Based on this sample, please identify {num_topics} distinct, high-level research themes or topics that are likely present in the full dataset.
    The topics should be broad enough to capture multiple pieces of evidence but specific enough to form a coherent argument.
    Examples of good topics: "Impact of AI on Labor Markets", "Climate Change Mitigation Strategies", "Educational Inequality Factors".
    
    Output Format (JSON List of Strings):
    ["Topic 1", "Topic 2", ...]
    """
    
    response = call_gemini_with_retry(prompt)
    try:
        json_str = response.replace("```json", "").replace("```", "").strip()
        if "[" in json_str:
            json_str = json_str[json_str.find("["):json_str.rfind("]")+1]
        return json.loads(json_str)
    except:
        print("Error parsing topics, using fallback list.")
        return ["Economic Inequality", "Climate Policy", "Technological Innovation", "Labor Market Trends", "Education Reform"]

def retrieve_relevant_evidence(topic, evidence_list, vectorizer, tfidf_matrix, top_k=50):
    """
    Retrieves the most relevant evidence for a given topic using cosine similarity.
    This allows the same evidence to be selected for multiple different topics.
    """
    topic_vec = vectorizer.transform([topic])
    cosine_similarities = cosine_similarity(topic_vec, tfidf_matrix).flatten()
    
    # Get top_k indices
    related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]
    
    relevant_evidence = []
    for idx in related_docs_indices:
        if cosine_similarities[idx] > 0.1: # Minimum similarity threshold
            relevant_evidence.append(evidence_list[idx])
            
    return relevant_evidence

def synthesize_topic_argument(topic, evidence_items):
    """Synthesizes an argument for a specific topic using relevant evidence."""
    
    if not evidence_items:
        return None

    # Limit context size
    subset = evidence_items[:40] 
    
    evidence_text = ""
    for i, item in enumerate(subset):
        evidence_text += f"{i+1}. {item['text']} (Source: {item['source']})\n"
        
    prompt = f"""
    You are a Senior Research Synthesizer. 
    Target Research Topic: "{topic}"
    
    I have retrieved a set of evidence that is semantically relevant to this topic.
    Note: Some evidence might be tangentially related or irrelevant outliers due to the retrieval process.
    
    Your task is to:
    1. Filter and analyze the evidence that genuinely supports the Target Research Topic.
    2. Formulate a comprehensive **Research Argument (Thesis Statement)** about "{topic}" based on this evidence.
    3. If the evidence supports multiple distinct perspectives on this topic, you can formulate a complex argument or note the debate.
    4. List the **Key Evidence Points** used.
    5. Cite the **Sources** (filenames).
    
    Input Evidence:
    {evidence_text}
    
    Output Format (JSON):
    {{
      "topic": "{topic}",
      "synthesized_argument": "The main claim...",
      "key_evidence_points": [
        "Point 1...",
        "Point 2..."
      ],
      "sources": ["file1.pdf", "file2.pdf"]
    }}
    
    Return ONLY valid JSON.
    """
    
    response = call_gemini_with_retry(prompt)
    if not response: return None
    
    try:
        json_str = response.replace("```json", "").replace("```", "").strip()
        if "{" in json_str:
            json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
        return json.loads(json_str)
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to extracted_hierarchical.jsonl")
    parser.add_argument("--output_file", required=True, help="Path to save synthesized arguments")
    parser.add_argument("--num_topics", type=int, default=30, help="Number of topics to generate")
    args = parser.parse_args()
    
    # 1. Extract Evidence
    evidence_list = extract_evidence(args.input_file)
    print(f"Total evidence items extracted: {len(evidence_list)}")
    
    if not evidence_list:
        print("No evidence found. Exiting.")
        return

    # 2. Vectorize Evidence (Pre-compute for retrieval)
    print("Vectorizing evidence for retrieval...")
    texts = [item['text'] for item in evidence_list]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # 3. Identify Core Topics
    print("Identifying core research topics from data...")
    topics = identify_core_topics(evidence_list, num_topics=args.num_topics)
    print(f"Identified {len(topics)} topics: {topics}")
    
    # 4. Synthesize per Topic (Iterative Retrieval)
    synthesized_results = []
    print("Synthesizing arguments for each topic...")
    
    for i, topic in enumerate(topics):
        print(f"Processing Topic {i+1}/{len(topics)}: {topic}...")
        
        # Dynamic Retrieval: Find evidence relevant to THIS topic
        # This allows the same evidence to be used for multiple topics
        relevant_evidence = retrieve_relevant_evidence(topic, evidence_list, vectorizer, tfidf_matrix)
        
        if len(relevant_evidence) < 3:
            print(f"  Skipping topic '{topic}' - insufficient relevant evidence found.")
            continue
            
        print(f"  Found {len(relevant_evidence)} relevant evidence items.")
        result = synthesize_topic_argument(topic, relevant_evidence)
        
        if result:
            synthesized_results.append(result)
            
        # Incremental save
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(synthesized_results, f, indent=2, ensure_ascii=False)
            
    print(f"Done! Saved {len(synthesized_results)} synthesized arguments to {args.output_file}")

if __name__ == "__main__":
    main()
