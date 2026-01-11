# Persona Interview System with Bridging Inference

A multi-agent system for persona inference using **bridging discourse analysis** and **embedding-based evaluation**. The system employs a GPT-4 Tool Agent to interview Hugging Face language models, extracting implicit persona attributes through linguistic bridging relations.

## 🎯 Overview

This system demonstrates a novel approach to persona inference by:
1. **Conducting strategic interviews** with target LLMs to elicit persona-revealing responses
2. **Extracting bridging inference relations** from conversation patterns (linguistic analysis)
3. **Constructing discourse graphs** to map implicit connections between concepts
4. **Predicting persona attributes** across four dimensions: Social Role, Personality, Background, and Interests
5. **Evaluating predictions** using embedding-based semantic similarity

## 🏗️ Architecture

```
┌─────────────────┐
│   Tool Agent    │  GPT-4 conducts interview & analyzes responses
│     (GPT-4)     │  - Generates strategic questions
└────────┬────────┘  - Extracts bridging relations
         │           - Constructs persona graph
         ▼
┌─────────────────┐
│   Target LLM    │  Hugging Face model (Qwen, Llama, etc.)
│  (Qwen/Llama)   │  - Responds as persona
└────────┬────────┘  - Answers filtered to prevent leakage
         │
         ▼
┌─────────────────┐
│ Bridging Graph  │  Linguistic discourse structure
│   & Analysis    │  - Part-of, Instrument, Theme relations
└────────┬────────┘  - Importance scoring via centrality
         │
         ▼
┌─────────────────┐
│ Persona Predict │  Final inference with similarity eval
│  + Evaluation   │  - Qwen embedding-based comparison
└─────────────────┘
```

## 📁 File Structure

### Core Scripts

- **`test_qwen.py`** ⭐ **Recommended**
  - Complete pipeline with embedding-based evaluation
  - Automatic persona extraction from agent's output
  - Qwen embedding similarity scoring (cosine similarity)
  - Comprehensive evaluation reports

### Configuration Files

- **`persona_schema.json`** - Defines persona attribute structure
  - Social roles, personality traits, backgrounds, interests
  
- **`bridging_relationships.json`** - Linguistic bridging relation definitions
  - Relation types: part-of, member-of, instrument, theme, cause-of, temporal, in

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers langchain langchain-openai langgraph
pip install matplotlib networkx numpy
```

### Setup

1. Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
HF_TOKEN=your-huggingface-token-here  # Optional, for gated models
```

2. Prepare configuration files:
   - `persona_schema.json` - Define your persona structure
   - `bridging_relationships.json` - Define bridging relation types

### Run the System

```bash
python test_qwen3.py
```

The system will:
1. Generate a random target persona (ground truth)
2. Conduct an interview with the target LLM
3. Extract bridging relations from conversation
4. Build and visualize discourse graph
5. Predict persona attributes
6. Evaluate predictions with embedding similarity

## 📊 Output Files

After execution, find results in the `outputs/` directory:

```
outputs/
├── interview_results_Qwen_Qwen3-1.7B.json  # Complete interview log
├── bridging_results.json                    # Extracted bridging relations
├── graph_structure.json                     # Discourse graph data
├── graph_visualization.png                  # Graph visualization
└── persona_similarity.json                  # Evaluation scores
```

## 🔬 Evaluation Metrics

The system evaluates persona predictions using **cosine similarity** of Qwen embeddings:

```
📊 GT vs Predicted - Qwen Embedding Similarity (Cosine)
═══════════════════════════════════════════════════════
1️⃣  Social Role:     0.8523
2️⃣  Personality:     0.7891
3️⃣  Background:      0.8234
4️⃣  Interests:       0.7645
────────────────────────────────────────────────────────
📈 OVERALL AVERAGE:  0.8073
════════════════════════════════════════════════════════
```

## 🧠 Key Concepts

### Bridging Inference
Bridging inference captures **implicit connections** between discourse elements that require world knowledge or semantic understanding. Unlike surface-level coreference, bridging relations reveal deeper cognitive reasoning patterns.

**Example:**
```
Q: What do you do for work?
A: I spend most of my time optimizing pipelines and ensuring data quality.

Bridging Relation:
- Anchor: "work"
- Anaphor: "pipelines"
- Relation: instrument (pipelines are instruments for work)
- Inference: Reveals data engineering role without explicit statement
```

### Persona Leakage Prevention (test_qwen2.py, test_qwen3.py)

To ensure fair evaluation, target LLM responses are **redacted** to remove ground truth persona keywords:

```python
# Original response
"As a data engineer, I work with ML pipelines..."

# Redacted response (sent to Tool Agent)
"As a [REDACTED], I work with ML pipelines..."
```

This prevents the Tool Agent from trivially extracting persona attributes through keyword matching.

## ⚙️ Configuration

### Persona Schema Example

```json
{
  "structure": {
    "social_role": {
      "categories": {
        "professional": {
          "examples": ["data engineer", "teacher", "nurse"]
        }
      }
    },
    "personality": {
      "categories": [
        {
          "openness": {
            "description": "Creative, curious, open to new experiences"
          }
        }
      ]
    }
  }
}
```

### Bridging Relations Example

```json
{
  "relations": {
    "part-of": {
      "description": "Anaphor is a component of anchor",
      "example": "room → ceiling"
    },
    "instrument": {
      "description": "Anaphor is a tool/method for anchor",
      "example": "murder → knife"
    }
  }
}
```

## 🎛️ Customization

### Change Target Model

```python
# In test_qwen3.py
TARGET_MODEL = "Qwen/Qwen3-1.7B"  # Change to any HF model
# Examples: "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1"
```

### Adjust Interview Length

```python
run_interview_system(
    openai_api_key=OPENAI_API_KEY,
    target_model_name=TARGET_MODEL,
    num_questions=5,  # Change number of questions
    device="cpu"
)
```

### Modify Tool Agent Model

```python
# In test_qwen3.py
TOOL_MODEL = 'gpt-4'  # Change to 'gpt-4-turbo', 'gpt-3.5-turbo', etc.
```

## 📈 Research Applications

This system is designed for research in:
- **Persona inference** from conversational data
- **Bridging discourse analysis** in dialogue systems
- **Multi-agent LLM interaction** patterns
- **Implicit reasoning** in language models
- **Persona consistency** evaluation

## 🛠️ Technical Details

### Bridging Extraction Pipeline

1. **Conversation Collection**: Agent asks strategic questions
2. **Utterance Analysis**: Parse Q&A pairs into discourse units
3. **Relation Identification**: Detect implicit connections requiring inference
4. **Graph Construction**: Build directed graph with weighted edges
5. **Centrality Scoring**: Compute PageRank/betweenness for importance ranking

### Embedding Similarity Computation

```python
# For each persona attribute
embedding_gt = encode_with_qwen(ground_truth_value)
embedding_pred = encode_with_qwen(predicted_value)
similarity = cosine_similarity(embedding_gt, embedding_pred)
```

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{persona_interview_system,
  title={Persona Interview System with Bridging Inference},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/persona-interview-system}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional bridging relation types
- Multi-turn dialogue strategies
- Alternative evaluation metrics
- Support for multilingual personas

## 📄 License

[MIT License](LICENSE)

## 🔍 Troubleshooting

### Common Issues

**Issue**: `OPENAI_API_KEY not found`
```bash
# Solution: Create .env file with your API key
echo "OPENAI_API_KEY=sk-..." > .env
```

**Issue**: Model download fails
```bash
# Solution: Set HF_TOKEN for gated models
export HF_TOKEN=hf_...
```

**Issue**: Out of memory on GPU
```bash
# Solution: Force CPU execution
device="cpu"  # In run_interview_system()
```

**Issue**: Thread warnings on macOS
```bash
# Already handled in code via:
os.environ["TOKENIZERS_PARALLELISM"] = "false"
matplotlib.use('Agg')
```

## 📧 Contact

For questions or collaboration: [your.email@example.com]

---

**Note**: This system requires OpenAI API access for the Tool Agent (GPT-4). The target LLM runs locally via Hugging Face Transformers.