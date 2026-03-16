# Research Report Generation

This directory contains the pipeline for automated research report generation, knowledge extraction, and landscape analysis.

## Key Scripts

### 1. `generate_report.py`
The main entry point for the pipeline.
- **Function**: Processes PDF documents to extract hierarchical findings (Research Questions -> Claims -> Evidence).
- **Output**: `extracted_hierarchical.jsonl`

### 2. `clean_findings.py`
Quality control module.
- **Function**: Uses LLM to evaluate the quality of extracted findings.
- **Logic**: Filters out low-quality, vague, or unsupported claims based on "Specificity", "Substance", and "Evidence" criteria.
- **Output**: Cleaned version of `extracted_hierarchical.jsonl`.

### 3. `analyze_viewpoints_vectors.py` (New!)
Advanced landscape analysis tool using vector embeddings and clustering.
- **Function**:
  1.  **Clustering**: Groups research findings into thematic clusters using K-Means on question embeddings.
  2.  **Viewpoint Extraction**: Identifies 3-5 distinct, mainstream viewpoints (dimensions of debate) for each cluster.
  3.  **Stance Analysis**: Maps each claim to a multi-dimensional stance vector (-1 to 1) against these viewpoints.
  4.  **Pattern Recognition**: Uses Silhouette Score to automatically detect the optimal number of "Schools of Thought" (2-5 patterns).
  5.  **Visualization**: Generates **Radar Charts (Spider Plots)** to visualize the shape of academic debate.
- **Output**: 
  - `analysis_results/vector_analysis_report.md`: A comprehensive Markdown report with embedded Radar Charts.
  - `analysis_results/*.png`: Individual Radar Chart images.

## Output Files

- `extracted_hierarchical.jsonl`: Raw hierarchical findings from each report.
- `analysis_results/`: Directory containing the final analysis reports and visualizations.
  - `vector_analysis_report.md`: **The main readable report.**
  - `cluster_X_radar_viz.png`: Visualization of stance patterns for Cluster X.

## Usage

To run the full analysis pipeline:

1.  **Extract Data**:
    ```bash
    python generate_report.py
    ```

2.  **Clean & Filter (Optional)**:
    ```bash
    python clean_findings.py
    ```

3.  **Generate Landscape Analysis & Radar Charts**:
    ```bash
    python analyze_viewpoints_vectors.py
    ```
    *Check `analysis_results/vector_analysis_report.md` for the results.*
