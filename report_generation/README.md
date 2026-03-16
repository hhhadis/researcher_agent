# Research Landscape & Viewpoint Analysis Pipeline

This repository contains a comprehensive pipeline for automated research synthesis, from raw PDF extraction to advanced multi-dimensional viewpoint analysis. The system is designed to process large collections of academic papers, identify core debates, and visualize the distribution of scholarly stances.

## 🚀 Pipeline Overview

The workflow consists of three main stages:
1.  **Extraction**: Mining hierarchical findings (Question -> Argument -> Evidence) from PDFs.
2.  **Cleaning**: Using LLM-based semantic filtering to ensure data quality.
3.  **Analysis**: Vector-based clustering, viewpoint extraction, and radar chart visualization.

---

## 🛠️ Key Scripts & Modules

### 1. `generate_report.py` (Data Extraction)
The entry point for processing raw PDF documents.
*   **Input**: A directory of PDF files.
*   **Process**:
    *   Extracts text from the first 5 pages (Abstract/Intro focus).
    *   Uses **ZhipuAI (GLM-4)** to identify key research questions.
    *   Extracts hierarchical "Findings": *Question* -> *Claims/Arguments* -> *Supporting Evidence*.
*   **Output**: `extracted_hierarchical.jsonl` (JSON Lines format).

### 2. `clean_findings.py` (Quality Control)
A semantic filter to remove noise and low-quality extractions.
*   **Process**:
    *   Uses an LLM "Research Auditor" persona to score each finding (0-10).
    *   **Criteria**:
        *   **Specificity**: Is the question well-defined?
        *   **Substance**: Does the claim provide a meaningful answer?
        *   **Evidence**: Is there concrete data or citations?
*   **Output**: Updates `extracted_hierarchical.jsonl` with only high-quality findings.

### 3. `analyze_viewpoints_vectors.py` (Landscape Analysis)
The core analysis engine that maps the "shape" of academic debate.
*   **Step 1: Question Clustering**
    *   Embeds all research questions using **NVIDIA Llama-Nemotron Embeddings**.
    *   Uses **K-Means** to group similar questions into thematic clusters (Topics).
    *   Identifies a "Representative Question" for each cluster.

*   **Step 2: Dynamic Viewpoint Extraction**
    *   For each cluster, uses LLM to identify **3-5 distinct dimensions of debate** (e.g., "Efficiency vs. Equity", "Government-led vs. Market-led").
    *   These dimensions become the axes of the analysis.

*   **Step 3: Multi-dimensional Stance Scoring**
    *   Maps every single research claim to a continuous vector (e.g., `[-0.8, 0.5, 0.2]`) representing its stance on the identified dimensions (-1.0 to +1.0).

*   **Step 4: Pattern Recognition & Visualization**
    *   Uses **Silhouette Score** to automatically detect the optimal number of "Schools of Thought" (Patterns) within the topic.
    *   Generates **Radar Charts (Spider Plots)** showing the distribution of these schools of thought.
    *   **Interpretation**:
        *   **Axes**: The key dimensions of the debate.
        *   **Colored Areas**: Distinct groups of papers with similar stance combinations.
        *   **N**: The number of papers falling into that specific school of thought.

*   **Output**: 
    *   `analysis_results/vector_analysis_report.md`: Full Markdown report with embedded charts.
    *   `analysis_results/*.png`: High-resolution radar charts.

---

## 📂 Output Structure

```text
report_generation/
├── extracted_hierarchical.jsonl       # The raw knowledge base of findings
├── analysis_results/
│   ├── vector_analysis_report.md      # 📄 The FINAL READABLE REPORT
│   ├── vector_analysis_report.json    # Structured analysis data
│   ├── cluster_0_radar_viz.png        # 📊 Radar chart for Topic 0
│   ├── cluster_1_radar_viz.png        # 📊 Radar chart for Topic 1
│   └── ...
```

## ⚡ Usage Guide

### Prerequisites
*   Python 3.8+
*   API Keys for **ZhipuAI** and **OpenRouter** (configured in `API_KEY.md` or environment variables).

### Step-by-Step Execution

1.  **Run Extraction** (Process PDFs):
    ```bash
    python generate_report.py
    ```

2.  **Run Quality Filter** (Optional but recommended):
    ```bash
    python clean_findings.py
    ```

3.  **Generate Analysis Report**:
    ```bash
    python analyze_viewpoints_vectors.py
    ```

4.  **View Results**:
    Open `analysis_results/vector_analysis_report.md` in any Markdown viewer (e.g., VS Code, Typora).

---

## 📊 Understanding the Radar Charts

*   **The Polygon**: Represents a "School of Thought" or a cluster of papers with similar views.
*   **The Vertices (Corners)**: Represent the key dimensions of the debate (e.g., "Economic Cost", "Environmental Impact").
*   **The Area**: A larger area towards a corner means that group of papers **strongly emphasizes or supports** that dimension.
*   **Overlap**: Where colors overlap, it indicates shared ground between different schools of thought.
