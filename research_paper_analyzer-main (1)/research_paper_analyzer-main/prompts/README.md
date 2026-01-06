# Prompt Templates

This directory contains prompt templates used by the Multi-Pass LLM Parser.

The system uses **three specialized prompts** for a comprehensive analysis:
- **Pass 1**: Extract nodes (concepts) from paper
- **Pass 2**: Find relations between nodes with confidence scores
- **Pass 3** (Optional): Infer contextual nodes and relations

## 📁 Files Overview

### `paper_to_logicgraph.txt` - Pass 1: Node Extraction

**Purpose**: Extracts all concepts from the research paper and organizes them into three levels.

**Output Structure**:
```json
{
  "Level3": ["Problem1", "Problem2"],
  "Level2": {
    "Problem": ["Formulation1", "Formulation2"],
    "Method": ["Approach1", "Approach2"]
  },
  "Level1": ["Technique1", "Technique2"]
}
```

**What it does**:
- Identifies research field problems (Level 3)
- Extracts mathematical/conceptual formulations (Level 2 - Problem)
- Captures solution approaches (Level 2 - Method)
- Lists concrete techniques and algorithms (Level 1)

**Key sections to customize**:
- **Level definitions**: Clarify what belongs in each level for your domain
- **Node naming conventions**: Adjust conciseness vs. descriptiveness
- **Domain focus**: Add field-specific terminology or concepts to prioritize
- **Extraction strategy**: Guide how to identify key concepts

### `find_relations.txt` - Pass 2: Relation Finding

**Purpose**: Identifies ALL relations between the extracted nodes, grounded in paper text.

**Input**: List of nodes from Pass 1 + original paper text

**Output Structure**:
```json
{
  "Relations": [
    {
      "source": "Node A",
      "target": "Node B",
      "relation": "formulated-as",
      "confidence": 9,
      "explanation": "Paper explicitly states..."
    }
  ],
  "overall_confidence": 8,
  "overall_explanation": "High quality extraction..."
}
```

**What it does**:
- Finds connections between nodes at all levels
- Assigns confidence scores (1-10) based on evidence strength
- Provides detailed explanations grounding each relation in paper text
- Evaluates overall analysis quality

**Key sections to customize**:
- **Relation types**: Add or remove relation categories
- **Confidence criteria**: Adjust what constitutes high vs. low confidence
- **Evidence requirements**: Specify how explicit paper text must be
- **Cross-level focus**: Emphasize certain types of connections (e.g., L3→L2→L1)

### `infer_context.txt` - Pass 3: Contextual Inference

**Purpose**: Infers additional context beyond what's explicitly in the paper.

**Input**: Complete graph from Pass 1 & 2 (nodes + relations)

**Output Structure**:
```json
{
  "InferredNodes": [
    {
      "node": "Related Concept",
      "level": "Level1",
      "confidence": 7,
      "explanation": "Relevant to research context because..."
    }
  ],
  "InferredRelations": [
    {
      "source": "Node A",
      "target": "Node B",
      "relation": "extends",
      "confidence": 8,
      "explanation": "Logical connection based on..."
    }
  ],
  "overall_confidence": 7,
  "overall_explanation": "Inferences add valuable context..."
}
```

**What it does**:
- Identifies concepts relevant to the research but not in paper
- Infers logical connections and transitive relations
- Adds broader research context
- Maintains confidence scoring for all inferences

**Key sections to customize**:
- **Inference scope**: Control how much context to add
- **Domain knowledge**: Guide what background concepts are relevant
- **Confidence thresholds**: Set minimum confidence for inferences
- **Inference types**: Focus on certain kinds of logical connections

## 🛠️ How to Modify Prompts

### General Guidelines:

1. **Test incrementally**: Make small changes and test with sample papers
2. **Preserve JSON structure**: Keep the output schema consistent
3. **Be specific**: Clear instructions yield better results
4. **Use examples**: Add examples for complex concepts
5. **Domain-specific**: Tailor prompts to your research field

### Making Changes:

1. **Edit the prompt file** directly in a text editor
2. **Restart the application** (or reinitialize parser) for changes to take effect
3. **Test with a sample paper** to validate output quality
4. **Compare results** with previous version to measure impact

### Common Customizations:

#### For Better Node Extraction (Pass 1):
- Add examples of good vs. bad node names
- Specify domain-specific terminology
- Adjust node count targets (e.g., "Extract 5-10 Level 3 concepts")
- Guide handling of mathematical notation

#### For Better Relations (Pass 2):
- Add new relation types for your domain
- Adjust confidence scoring criteria
- Require more/less explicit evidence
- Focus on specific relation patterns (e.g., causal chains)

#### For Better Inference (Pass 3):
- Limit or expand inference scope
- Specify important background concepts
- Control inference confidence thresholds
- Guide inference reasoning style

## 📊 Confidence Scoring Guidelines

All prompts use a **1-10 confidence scale**:

- **9-10**: Explicitly stated with clear evidence
- **7-8**: Strongly implied or well-supported
- **5-6**: Reasonable inference from context
- **3-4**: Weak connection or speculation
- **1-2**: Highly uncertain or tangential

When modifying prompts, maintain these confidence standards for consistency.

## 🔄 Pass Interaction

The three passes work together:

```
Pass 1 (Node Extraction)
    ↓ nodes extracted
Pass 2 (Relation Finding) ← uses Pass 1 nodes + paper text
    ↓ nodes + paper-based relations
Pass 3 (Contextual Inference) ← uses complete graph from Pass 1 & 2
    ↓ adds inferred nodes + relations
Final Graph (complete)
```

Understanding this flow helps when customizing prompts for your use case.

## 💡 Tips for Best Results

1. **Start with Pass 1**: Get node extraction right before worrying about relations
2. **Be specific about confidence**: Clear criteria yield consistent scoring
3. **Test across domains**: Prompts may work differently for different research fields
4. **Iterate**: Prompt engineering is iterative; refine based on outputs
5. **Compare 2-pass vs 3-pass**: Sometimes less is more; Pass 3 adds tokens and complexity
6. **Document changes**: Keep notes on what modifications improve results

## 🐛 Troubleshooting

### Issue: LLM returns malformed JSON
- **Solution**: Simplify prompt, add explicit JSON format examples, check for special characters

### Issue: Too many/few nodes extracted
- **Solution**: Add explicit count guidelines (e.g., "Extract 3-7 concepts at each level")

### Issue: Low confidence scores across the board
- **Solution**: Relax confidence criteria in Pass 2 prompt, or paper may lack clear structure

### Issue: Inferred nodes not relevant
- **Solution**: Narrow Pass 3 scope, add domain constraints, increase confidence threshold

### Issue: Missing obvious relations
- **Solution**: In Pass 2, emphasize comprehensive relation finding, lower evidence requirements slightly

## 📚 Additional Resources

- See main `README.md` for overall application usage
- Check `IMPROVEMENTS.md` for development notes
- Example outputs in `examples/` directory show expected format

---

**Remember**: These prompts are the brain of the system. Thoughtful customization can dramatically improve extraction quality for your specific use case!


