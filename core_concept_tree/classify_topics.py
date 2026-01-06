
import os
import json
import pandas as pd
from bertopic import BERTopic
from zhipuai import ZhipuAI

# API Key
ZHIPUAI_API_KEY = "77e73e22741f4b45854c777f4763236f.311YBhRcPuEQk3ZQ"

def generate_topic_label(keywords, client):
    if not keywords:
        return "Unknown Topic"
    
    prompt = f"""
    You are an expert researcher. Based on the following keywords describing a research topic, please provide ONE specific and concise academic concept name (e.g., "Deep Learning", "Gene Editing", "Supply Chain Management") that best represents this topic.
    
    Strict Rules:
    1. Return ONLY the concept name. No explanation.
    2. Do not use quotes.
    3. Prefer singular form unless plural is standard.
    
    Keywords: {', '.join(keywords)}
    """
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Labeling Error: {e}")
        return keywords[0] if keywords else "Topic"

def classify_with_bertopic(data_path, output_dir):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter data with abstracts
    data_with_abstracts = [item for item in data if item.get('abstract')]
    abstracts = [item['abstract'] for item in data_with_abstracts]
    
    print(f"Training BERTopic on {len(abstracts)} documents...")
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=True)
    topics, _ = topic_model.fit_transform(abstracts)
    
    # Save model and info
    topic_model.save(os.path.join(output_dir, "bertopic_model"))
    
    doc_info = topic_model.get_document_info(abstracts)
    doc_info.to_csv(os.path.join(output_dir, "document_info.csv"), index=False)
    
    topic_info = topic_model.get_topic_info()
    
    # --- Integrate LLM Labeling ---
    print("Generating Topic Labels using LLM...")
    client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
    
    llm_labels = []
    for index, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            llm_labels.append("Outliers")
            continue
            
        # Get top keywords for this topic
        # Representation column is usually a list of strings
        # But in dataframe it might be stringified list or actual list
        # Let's use topic_model.get_topic(topic_id) to be safe
        top_words_scores = topic_model.get_topic(topic_id)
        if top_words_scores:
            top_words = [w[0] for w in top_words_scores[:10]] # Top 10 words
            label = generate_topic_label(top_words, client)
            llm_labels.append(label)
            print(f"Topic {topic_id}: {label}")
        else:
            llm_labels.append(f"Topic {topic_id}")
            
    topic_info['CustomLabel'] = llm_labels
    topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
    # ------------------------------

    # Organize files into directories
    # Use zip to ensure strict alignment between topic info and original data
    if len(doc_info) != len(data_with_abstracts):
        print(f"Warning: Mismatch in counts! doc_info: {len(doc_info)}, data: {len(data_with_abstracts)}")

    for (index, row), original_item in zip(doc_info.iterrows(), data_with_abstracts):
        topic_id = row['Topic']
        topic_dir = os.path.join(output_dir, f"topic_{topic_id}")
        os.makedirs(topic_dir, exist_ok=True)

        # original_item corresponds to the current row in doc_info
        title = original_item.get('title', 'No Title Provided')
        keywords = original_item.get('keywords', [])
        
        # Explicitly handle year
        year = original_item.get('year')
        if not year:
            year = 'Unknown'
            # print(f"Warning: Missing year for document index {index}")

        # Format the content to be saved
        content_to_save = f"Title: {title}\nYear: {year}\nKeywords: {', '.join(keywords)}"

        # Use the document's original index as the filename
        file_path = os.path.join(topic_dir, f"doc_{index}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_to_save)

if __name__ == '__main__':
    base_dir = r'd:/workspace/ResearchAgent/core_concept_tree'
    input_file = os.path.join(base_dir, 'extracted_data.json')
    classify_with_bertopic(input_file, base_dir)
