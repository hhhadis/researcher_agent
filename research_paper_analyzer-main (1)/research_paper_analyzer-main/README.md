# 🧠 Research Logic Graph Extractor

An AI-powered application that extracts and visualizes hierarchical logic graphs from research papers using LLMs with a sophisticated multi-pass analysis system.

## 🎯 Overview

This tool automatically analyzes research papers (PDF) and generates a **three-level Hierarchical Logic Graph (HLG)** following a problem-solving narrative:
- **Level 3**: Research field problems and challenges
- **Level 2**: Mathematical/conceptual formulations of those problems
- **Level 1**: Specific techniques and algorithms that solve the formulations

The structure follows: **Problem → Formulation → Solution**

### 🆕 **Multi-Paper Comparison Mode**
Compare multiple research papers (2-5) and automatically identify cross-paper relations! Perfect for literature reviews, approach comparison, and understanding how papers relate to each other. See [MULTI_PAPER_MODE.md](MULTI_PAPER_MODE.md) for details.

### 🔍 Multi-Pass Analysis System

The application uses an intelligent **two-pass (or three-pass) approach**:

1. **Pass 1 - Node Extraction**: Extracts all concepts (Level 1, 2, 3) from the paper
2. **Pass 2 - Relation Finding**: Identifies ALL relations between nodes, grounded in paper text, with confidence scores and explanations
3. **Pass 3 - Contextual Inference** (Optional): Infers additional context nodes and relations beyond what's explicitly stated in the paper
   - Adds concepts relevant to research context but not mentioned in paper
   - Infers logical connections and broader research context
   - Visually distinguished with dashed borders/lines

Every relation includes:
- **Confidence score** (1-10 scale)
- **Detailed explanation** grounding the relation in paper content
- **Overall analysis confidence** for quality assessment

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
research-logic-graph/
│
├── app.py                          # Main Streamlit application
├── pdf_extractor.py                # Section-aware PDF text extraction
├── llm_parser.py                   # Multi-pass LLM API integration
├── graph_builder.py                # Graph construction and visualization
├── requirements.txt                # Python dependencies
├── .env                            # API keys (create this file)
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── MULTI_PAPER_MODE.md             # Multi-paper comparison guide
├── IMPROVEMENTS.md                 # Development notes
├── prompts/
│   ├── paper_to_logicgraph.txt    # Pass 1: Node extraction prompt
│   ├── find_relations.txt          # Pass 2: Relation finding prompt
│   ├── infer_context.txt           # Pass 3: Contextual inference prompt
│   ├── cross_paper_relations.txt   # Pass 4: Cross-paper relation finding (multi-paper mode)
│   └── README.md                   # Prompt documentation
├── examples/
│   └── BETag_output.json          # Example output
└── papers/                         # Sample research papers (optional)
```

## 🔧 Features

### 📄 Smart PDF Extraction
- Upload research papers in PDF format
- **Section-aware extraction** prioritizing key content:
  - Prioritizes: Abstract, Introduction, Methods, Results, Conclusions
  - Skips: References, Acknowledgments, Appendices
- Intelligent character limit management
- Text preview functionality

### 🔍 Multi-Pass LLM Analysis
- **Pass 1**: Extract all nodes (concepts) from paper
- **Pass 2**: Find ALL relations with confidence scores and explanations
- **Pass 3** (Optional): Infer contextual nodes and relations
- Multiple LLM model support (Claude 3.5 Sonnet, GPT-4, Llama)
- Advanced structured prompt engineering (3 specialized prompts)
- Automatic JSON parsing and validation
- **Confidence scoring system** (1-10 scale) for all relations
- **Detailed explanations** grounding each relation in paper text
- **Token usage tracking** with per-pass breakdowns

### 🎨 Interactive Visualization
- Color-coded nodes by level and type
- **Visual distinction for inferred elements**:
  - Inferred nodes: dashed borders
  - Inferred relations: dashed lines
- Interactive graph manipulation (drag, zoom, pan)
- Edge labels showing relation types
- Hierarchical layout (Level 3 → Level 2 → Level 1)
- Hover tooltips with confidence scores and explanations

### 💾 Export Options
- Download HLG as JSON (with all metadata)
- Export interactive graph as HTML
- Export graph structure as JSON
- All exports include confidence scores and explanations

## 🧩 Output Format

The output includes comprehensive metadata and confidence scoring:

```json
{
  "Level3": ["Data Imbalance", "Model Generalization"],
  "Level2": {
    "Problem": ["Loss Function Formulation", "Parameter Efficiency Constraint"],
    "Method": ["Weighted Optimization", "Low-rank Factorization"],
    "Relations": [
      {
        "source": "Data Imbalance",
        "target": "Loss Function Formulation",
        "relation": "formulated-as",
        "confidence": 9,
        "explanation": "Paper frames the imbalance problem as a weighted loss optimization"
      },
      {
        "source": "Weighted Optimization",
        "target": "Focal Loss",
        "relation": "implemented-via",
        "confidence": 8,
        "explanation": "Focal loss implements the weighted optimization approach"
      }
    ]
  },
  "Level1": ["LoRA", "Focal Loss", "Class Weights"],
  "overall_confidence": 8,
  "overall_explanation": "High confidence in extraction. Paper clearly describes methodology.",
  
  "InferredNodes": [
    {
      "node": "Cross-Entropy Loss",
      "level": "Level1",
      "confidence": 7,
      "explanation": "Standard baseline for classification, contextually relevant"
    }
  ],
  "InferredRelations": [
    {
      "source": "Focal Loss",
      "target": "Cross-Entropy Loss",
      "relation": "extends",
      "confidence": 9,
      "explanation": "Focal Loss is a direct extension of Cross-Entropy with focus parameter"
    }
  ],
  "inference_confidence": 7,
  "inference_explanation": "Inferred elements add valuable research context",
  
  "_token_usage": {
    "pass1_prompt_tokens": 3245,
    "pass1_completion_tokens": 423,
    "pass2_prompt_tokens": 2876,
    "pass2_completion_tokens": 856,
    "pass3_prompt_tokens": 1654,
    "pass3_completion_tokens": 542,
    "prompt_tokens": 7775,
    "completion_tokens": 1821,
    "total_tokens": 9596
  }
}
```

**Note**: `InferredNodes`, `InferredRelations`, `inference_confidence`, and Pass 3 token usage only appear when Pass 3 (Contextual Inference) is enabled.

## 📊 Understanding Confidence Scores

Every relation and inferred element includes a confidence score (1-10 scale) and detailed explanation:

### Confidence Levels:
- **9-10**: Very high confidence - explicitly stated in paper with clear evidence
- **7-8**: High confidence - strongly implied or well-supported by paper content
- **5-6**: Moderate confidence - reasonable inference from paper context
- **3-4**: Low confidence - weak connection or speculative inference
- **1-2**: Very low confidence - highly uncertain or tangential

### Color Coding in UI:
- 🟢 **Green** (8-10): High confidence
- 🟡 **Yellow** (6-7): Moderate confidence
- 🟠 **Orange** (4-5): Low confidence
- 🔴 **Red** (1-3): Very low confidence

### Pass-Specific Scoring:
- **Pass 2 Relations**: Confidence based on how explicitly the relation is stated in the paper
- **Pass 3 Inferred Elements**: Confidence based on logical soundness and contextual relevance
- **Overall Confidence**: Aggregate score for the entire analysis quality

Use these scores to filter or prioritize connections when working with the extracted logic graph.

## 🎨 Graph Visualization

### Node Colors:
- **🔵 Blue nodes**: Level 3 (Research field problems)
- **🟣 Purple nodes**: Level 2 Problem Formulations
- **🟢 Green nodes**: Level 2 Solution Approaches
- **⚫ Gray nodes**: Level 1 (Technical implementations)

### Visual Distinctions:
- **Solid borders**: Nodes extracted from paper (Pass 1)
- **Dashed borders**: Inferred context nodes (Pass 3)
- **Solid lines**: Relations grounded in paper text (Pass 2)
- **Dashed lines**: Inferred relations (Pass 3)
- **Edge colors**: Different colors for different relation types
- **Hover tooltips**: Show confidence scores and detailed explanations

## 🔗 Supported Relations

### Problem-to-Problem (L3→L3)
- `causes` - One research problem leads to or creates another
- `contributes-to` - One problem contributes to a broader problem
- `related-to` - Problems are connected or complementary
- `extends` - One problem builds upon another
- `conflicts` - Mutually exclusive problem framings

### Cross-Level (Problem-Solving Flow)
- `formulated-as`, `reduced-to`, `modeled-as` - Problem → Formulation (L3→L2)
- `solved-by`, `implemented-via`, `optimized-by` - Formulation → Solution (L2→L1)
- `implements`, `approximates` - Solution → Formulation (L1→L2)

### Within-Level (L2→L2, L1→L1)
- `related-to` - Concepts are connected
- `improves`, `extends` - Incremental improvements
- `conflicts` - Contradictory approaches
- `is-part-of` - Component relationship

### Dependencies & Validation
- `requires`, `enables` - Prerequisites
- `validates`, `supports` - Evidence and backing

## 🛠️ Configuration

Edit settings in the sidebar:
- **LLM Model**: Choose from supported models (Claude 3.5 Sonnet, GPT-4o, GPT-4 Turbo, Llama 3.1 70B)
- **Max Characters**: Adjust text processing limit (5,000 - 30,000 characters)
- **Enable Pass 3: Contextual Inference**: Toggle to enable/disable inference of additional context
  - When enabled: Adds inferred nodes and relations (uses extra tokens)
  - When disabled: Only uses content explicitly stated in the paper

### Customizing Prompts

The system uses three specialized prompts that can be customized:

1. **`prompts/paper_to_logicgraph.txt`**: Controls Pass 1 (node extraction)
2. **`prompts/find_relations.txt`**: Controls Pass 2 (relation finding)
3. **`prompts/infer_context.txt`**: Controls Pass 3 (contextual inference)

Modify these files to adjust how the LLM analyzes papers and extracts logic graphs.

## 📊 Example Usage

### Single Paper Mode

1. **Upload**: Upload a research paper PDF
2. **Extract**: Click "Extract Text from PDF" (uses smart section-aware extraction)
3. **Configure** (Optional): In sidebar, adjust settings:
   - Select LLM model
   - Adjust max characters
   - Enable/disable Pass 3 (Contextual Inference)
4. **Analyze**: Navigate to "Analysis" tab and click "Analyze Paper with LLM"
   - Pass 1: Extracts all nodes
   - Pass 2: Finds all relations with confidence scores
   - Pass 3 (if enabled): Infers additional context
5. **Review**: View extracted nodes, relations, confidence scores, and token usage
6. **Visualize**: View interactive graph in "Visualization" tab
   - Hierarchical layout shows problem → formulation → solution flow
   - Hover over nodes/edges to see confidence scores and explanations
7. **Export**: Download JSON, HTML, or graph structure from "Export" tab

### Multi-Paper Mode

1. **Select Mode**: Click "📚 Multi-Paper Mode" at the top
2. **Upload**: Upload 2-5 research papers (PDFs)
3. **Extract**: Click "Extract Text from All Papers"
4. **Configure**: Adjust settings in sidebar (same as single paper mode)
5. **Analyze**: Click "Analyze All Papers & Find Cross-Paper Relations"
   - Phase 1: Analyzes each paper individually
   - Phase 2: Finds cross-paper relations
6. **Review**: View per-paper results and cross-paper relations
7. **Visualize**: View combined graph with color-coded papers and magenta cross-paper edges
8. **Export**: Download multi-paper analysis JSON or HTML

For detailed multi-paper mode instructions, see [MULTI_PAPER_MODE.md](MULTI_PAPER_MODE.md)

## 🔐 API Keys

This project uses OpenRouter for LLM access. Get your API key at:
https://openrouter.ai/

## 📈 Features & Roadmap

### ✅ Implemented Features:
- ✅ Multi-pass analysis system (2-pass or 3-pass)
- ✅ Multi-paper comparison mode (2-5 papers)
- ✅ Cross-paper relation detection
- ✅ Confidence scoring for all relations
- ✅ Contextual inference (Pass 3)
- ✅ Section-aware PDF extraction
- ✅ Token usage tracking
- ✅ Visual distinction for inferred elements
- ✅ Level 3→Level 3 problem relations

### 🔮 Future Extensions:
- **Graph Diffusion Mode**: Generate variations of logic graphs
- **Cross-paper Comparison**: Compare multiple papers
- **Ontology Layer**: Standardized concept vocabulary
- **Embedding Layer**: Convert graphs to vectors for clustering
- **Batch Processing**: Analyze multiple papers at once
- **Citation Analysis**: Extract and visualize paper citations
- **Custom Relation Types**: User-defined relation categories

## 🐛 Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### API Key Issues
- Ensure `.env` file exists in project root
- Check API key is valid at https://openrouter.ai/
- Verify environment variable is loaded: `OPENROUTER_API_KEY=your_key_here`
- Ensure you have API credits available

### PDF Extraction Issues
- Ensure PDF is not password-protected
- Check if PDF contains extractable text (not image-based)
- Try re-uploading the file
- If extraction yields very few characters, the PDF may need OCR processing
- Large PDFs: Use the max characters slider to process within limits

### Analysis Issues
- **Low confidence scores**: Paper may be unclear or highly technical; try different sections
- **Missing relations**: Enable Pass 3 for contextual inference
- **Incorrect relations**: Adjust prompts in `prompts/` directory to refine extraction
- **Token limits exceeded**: Reduce max characters or use shorter papers

### Visualization Issues
- **Graph not displaying**: Ensure analysis completed successfully
- **Overlapping nodes**: Drag nodes to rearrange; hierarchical layout minimizes overlap
- **Inferred elements not showing**: Ensure Pass 3 is enabled in settings
- **Tooltips not appearing**: Hover directly over nodes or edges

## 💡 Best Practices & Tips

### For Best Results:
1. **Use well-structured papers**: Papers with clear sections (Abstract, Methods, Results) work best
2. **Start without Pass 3**: First analyze with 2-pass mode to see paper-grounded results, then enable Pass 3 for context
3. **Adjust max characters**: For very long papers, increase the limit; for focused analysis, decrease it
4. **Check confidence scores**: Focus on high-confidence relations (7+) for core paper contributions
5. **Customize prompts**: Modify prompts in `prompts/` directory to focus on specific aspects of papers
6. **Export early**: Save JSON outputs for each analysis to compare different models or settings

### Recommended Workflow:
1. **Quick analysis** (2-pass, 10k chars, Claude 3.5 Sonnet) - Fast, paper-focused
2. **Deep analysis** (3-pass, 20k chars, Claude 3.5 Sonnet) - Comprehensive with context
3. **Compare results** - Look at differences between 2-pass and 3-pass outputs

### Model Selection:
- **Claude 3.5 Sonnet**: Best overall performance, excellent reasoning (recommended)
- **GPT-4o**: Fast and cost-effective, good for quick analyses
- **GPT-4 Turbo**: Balanced performance and quality
- **Llama 3.1 70B**: Open source option, decent results

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional relation types and categorization
- Improved prompt engineering
- Better visualization layouts
- Support for more LLM providers
- Batch processing capabilities

## 📧 Support

For issues and questions, please open an issue on GitHub.

**Tip**: When reporting issues, please include:
- Error messages or screenshots
- Paper characteristics (length, field, structure)
- Settings used (model, max chars, Pass 3 enabled/disabled)
- Token usage statistics if available


