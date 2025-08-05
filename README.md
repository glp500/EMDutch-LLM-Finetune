# EMDutch-LLM-Finetune

## Fine-Tuning Large Language Models for Early Modern Dutch Translation

This repository contains the complete experimental framework and research code for the paper **"Fine-Tuning Large-Language Models for Early Modern Dutch Translation"**. The project addresses the challenging task of translating Early Modern Dutch (circa 1550-1750) historical texts to contemporary Dutch using state-of-the-art large language models.

## 🎯 Purpose and Research Motivation

Early Modern Dutch represents a critical period in Dutch linguistic history, characterized by significant grammatical, lexical, and orthographic differences from contemporary Dutch. Historical documents from this era—including testimonials, legal texts, and religious manuscripts—contain invaluable cultural and historical information that remains largely inaccessible to modern readers.

**Key Challenges Addressed:**
- **Linguistic Evolution**: Archaic vocabulary, obsolete grammatical structures, and historical spelling variations
- **Limited Training Data**: Scarcity of parallel Early Modern Dutch to Modern Dutch translation pairs
- **Domain Specificity**: Historical texts often contain specialized terminology and cultural references
- **Preservation of Meaning**: Maintaining historical context while making content accessible to contemporary readers

## 🔬 Research Methodology

This research employs **ORPO (Odds Ratio Preference Optimization)** fine-tuning to adapt large language models for historical Dutch translation. The methodology combines:

1. **Base Model Selection**: Meta-Llama-3-8B-Instruct as the foundation model
2. **Preference-Based Training**: ORPO training using curated translation pairs with quality rankings
3. **Parameter-Efficient Fine-Tuning**: LoRA (Low-Rank Adaptation) for memory-efficient training
4. **Comprehensive Evaluation**: BERT and METEOR score analysis across multiple model variants

## 📊 Model Comparisons

The research evaluates multiple approaches:
- **Base Models**: Mixtral-8x7B-Instruct, Llama-3-8B-Instruct, Aya-23-8B, Phi-3-medium
- **Fine-Tuned Models**: ORPO-Llama-3-8B, Unsloth-Llama-3-8B
- **Evaluation Metrics**: BERT scores (semantic similarity) and METEOR scores (alignment-based evaluation)

## 🗂️ Repository Structure

```
EMDutch-LLM-Finetune/
├── Model Training/
│   ├── Thesis_ORPO_Training.ipynb          # ORPO fine-tuning implementation
│   └── Thesis_Unsloth_Training.ipynb       # Alternative Unsloth training approach
├── Inference/
│   ├── Thesis_Inference_Template.ipynb     # Translation inference pipeline
│   └── Thesis_Inference_Template_2.ipynb   # Alternative inference template
├── Data Preprocessing/
│   ├── model_responses_processing_final.ipynb    # Model output processing
│   └── testimonial_data_processing.ipynb         # Training data preparation
├── Translation Evaluation/
│   └── Model_Evaluation.ipynb              # BERT/METEOR score computation
├── Text Data Analysis/
│   ├── test_data_analysis.ipynb            # Test dataset analysis
│   ├── test_results_wordcount.ipynb        # Word count analysis
│   └── test_score_analysis.ipynb           # Score distribution analysis
├── Training Data/
│   ├── ORPO/orpo_train.csv                 # ORPO training dataset
│   └── Unsloth/unsloth_train.csv           # Unsloth training dataset
├── Test Data/
│   └── test.csv                            # Evaluation test set
└── Findings:Results from paper/
    ├── BERT:METEOR Scores/                 # Evaluation metrics results
    └── Initial Translations from all Models/    # Model translation outputs
```

## 🚀 Getting Started

### Prerequisites

The notebooks are designed to run in Google Colab with GPU acceleration (recommended: L4 GPU or higher). Required libraries are installed automatically within each notebook.

### Key Dependencies
- `transformers`: Hugging Face model library
- `trl`: TRL library for ORPO training
- `peft`: Parameter-efficient fine-tuning
- `bert-score`: BERT-based evaluation metrics
- `evaluate`: Hugging Face evaluation library
- `bitsandbytes`: Quantization for memory efficiency
- `wandb`: Experiment tracking

## 📋 Usage Instructions

### 1. Model Training

#### ORPO Training
```python
# Open Model Training/Thesis_ORPO_Training.ipynb
# The notebook handles:
# - Hugging Face authentication
# - Weights & Biases login
# - Dataset loading from glp500/testimonyORPO
# - ORPO fine-tuning with preference optimization
# - Model pushing to Hugging Face Hub
```

**Training Configuration:**
- Learning rate: 8e-6
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Training epochs: 8
- LoRA rank: 16, alpha: 32

#### Alternative Training with Unsloth
```python
# Open Model Training/Thesis_Unsloth_Training.ipynb
# Alternative training approach using Unsloth framework
```

### 2. Translation Inference

```python
# Open Inference/Thesis_Inference_Template.ipynb
# Configure model loading and generation parameters
# Input: Early Modern Dutch text
# Output: Contemporary Dutch translation
```

**Inference Pipeline:**
1. Load fine-tuned model and tokenizer
2. Format input text with appropriate chat templates
3. Generate translation with controlled parameters
4. Post-process output for readability

### 3. Model Evaluation

```python
# Open Translation Evaluation/Model_Evaluation.ipynb
# The notebook computes:
# - BERT scores for semantic similarity
# - METEOR scores for alignment-based evaluation
# - Statistical analysis across all model variants
```

**Evaluation Process:**
1. Load translation results CSV
2. Compute BERT scores using `bert-base-uncased`
3. Calculate METEOR scores with NLTK
4. Export results for statistical analysis

## 📈 Results and Findings

The research demonstrates significant improvements in Early Modern Dutch translation quality through preference-based fine-tuning. Key findings include:

- **ORPO Fine-tuning**: Substantial improvements over base models in translation accuracy
- **Semantic Preservation**: High BERT scores indicating maintained semantic meaning
- **Fluency Enhancement**: Improved METEOR scores showing better alignment with reference translations
- **Domain Adaptation**: Successful adaptation to historical Dutch linguistic patterns

## 🔄 Reproducibility

This repository enables complete reproduction of the research:

1. **Training Reproduction**: Run training notebooks with identical hyperparameters
2. **Evaluation Reproduction**: Use provided test datasets and evaluation scripts
3. **Result Verification**: Compare outputs with provided result files
4. **Methodology Extension**: Adapt notebooks for related historical translation tasks

## 📝 Citation

If you use this repository in your research, please cite:

```bibtex
@software{EMDutch_LLM_Finetune,
  title = {Fine-Tuning Large-Language Models for Early Modern Dutch Translation},
  author = {Anonymous},
  url = {https://anonymous.4open.science/r/EMDutch-LLM-Finetune-744E/README.md},
  year = {2024}
}
```

## 🤝 Contributing

This repository serves as a research artifact for reproducibility. For questions about the methodology or implementation details, please refer to the accompanying research paper.

## 📄 License

This repository is provided for academic research purposes. Please refer to individual model licenses for fine-tuned models published on Hugging Face Hub.
