# ResearchAgent: AI-Powered Research Analysis & Collaboration Platform

ResearchAgent is a comprehensive toolkit designed to analyze research papers, extract core concepts, visualize knowledge evolution, and predict future research directions. By leveraging Large Language Models (LLM) like GLM-4-Flash and advanced clustering techniques (BERTopic), it transforms static paper collections into dynamic, interactive knowledge graphs and actionable insights.

## 🚀 Key Features

### 1. Dual-Category Concept Extraction & Visualization
- **Automated Keyword Extraction**: Uses GLM-4-Flash to extract and categorize keywords from papers into two distinct streams:
  - **Technical Concepts**: Algorithms, models, and methodologies.
  - **Ethical/Social Issues**: Challenges, biases, and societal impacts.
- **Interactive Knowledge Graphs**: visualization of concept evolution over time using PyVis.
  - **Dual Trees**: Generates separate but linked concept trees for Technical and Ethical domains.
  - **Paper Association**: Hover over nodes to see relevant papers (with auto-line-wrapping tooltips).
  - **Connection Preservation**: Maintains the semantic link between technical advancements and their associated ethical considerations.

### 2. Intelligent Trend Prediction
- **Lineage Analysis**: Traces the evolutionary path of concepts from their roots to the latest frontiers.
- **Issue Prediction**: Based on historical technical-ethical pairings, the system predicts:
  - **Emerging Ethical Issues**: Potential future challenges arising from current frontier technologies.
  - **Proposed Technical Solutions**: Innovative approaches to mitigate these predicted risks.
- **Automated Reporting**: Generates detailed JSON and Markdown reports on future research opportunities.

### 3. Cross-Paper Collaboration Analysis
- **Collaboration Proposal**: Analyzes pairs of papers to identify complementary strengths.
- **Workflow Generation**: Proposes concrete collaboration workflows and joint research directions.

### 4. Multi-Paper & Dataset Analysis
- **Research-14K Dataset Integration**: Tools for analyzing large-scale datasets.
- **Reference Analysis**: Scripts for matching references and building citation chains.

## 📂 Project Structure

```
ResearchAgent/
├── core_concept_tree_2/       # [LATEST] Core logic for Concept Tree & Prediction
│   ├── extract_new_data.py    # GLM-based keyword extraction (Tech & Ethical)
│   ├── classify_topics.py     # BERTopic clustering and topic assignment
│   ├── create_network.py      # NetworkX & PyVis graph generation (Dual Trees)
│   └── predict_new_issues.py  # Trend prediction and issue forecasting script
├── core_concept_tree/         # [STABLE] Previous version of concept analysis
├── cross_paper_analysis/      # Logic for analyzing paper collaboration
│   └── collaboration_workflow.py
├── multi_paper_analysis/      # Tools for analyzing multiple papers simultaneously
├── research_14k_diffusion/    # Analysis specifically for Research-14K dataset
├── scripts/                   # Utility scripts for data processing and visualization
└── README.md                  # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/hhhadis/researcher_agent.git
    cd ResearchAgent
    ```

2.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Install the required packages:
    ```bash
    pip install zhipuai bertopic pandas networkx pyvis tqdm matplotlib scikit-learn
    ```

3.  **API Key Configuration**:
    This project uses ZhipuAI (GLM-4). You need to configure your API key.
    - Create a `API_KEY.md` file or set it in your environment variables as required by the scripts.
    - *Note: Ensure your API key is kept secure.*

## 📖 Usage

### 1. Concept Tree Generation (End-to-End)
Navigate to `core_concept_tree_2` and run the pipeline:

```bash
cd core_concept_tree_2

# Step 1: Extract Keywords (Tech & Ethical)
python extract_new_data.py

# Step 2: Classify Topics
python classify_topics.py

# Step 3: Generate Interactive Network Graphs
python create_network.py
```
*Output*: Interactive HTML files in `core_concept_tree_2/concept_network/`.

### 2. Predict Future Issues
Generate predictions for new ethical issues based on the latest technical frontiers:

```bash
cd core_concept_tree_2
python predict_new_issues.py
```
*Output*: Prediction reports in `core_concept_tree_2/predictions/`.

### 3. Cross-Paper Analysis
Run the collaboration workflow analysis:

```bash
cd cross_paper_analysis
python collaboration_workflow.py
```


