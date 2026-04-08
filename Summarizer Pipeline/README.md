# NLP Abstractive Summarization Pipeline

## Overview

Built a robust NLP summarization pipeline using state-of-the-art Sequence-to-Sequence (Seq2Seq) Transformer models (BART & T5) to condense extensive legal documents and CNN/DailyMail articles. This pipeline streamlines information consumption, generating accurate, concise, and context-aware summaries dynamically.

## Key Concepts

- **Abstractive Text Summarization**: Generating cohesive summaries using generative AI rather than simple extractive techniques.
- **Hugging Face Ecosystem**: Using pre-trained Transformers, Datasets, and Trainer APIs.
- **NLP Evaluation**: Multi-metric evaluation protocols combining precision, recall, and semantic similarity.

## Architecture / Approach

1. **Data Pre-Processing**: Extracted structured text components, grouped legal clauses (FACT, JUDGMENT, ORDER), and chunked long documents into overlapping sequences (1024 chunks) using greedy splitting techniques to bypass LLM context constraints limit.
2. **Abstractive Inferencing (BART)**: Initialized zero-shot summary generation utilizing `facebook/bart-large-cnn`.
3. **Fine-Tuning (T5-Small)**: Trained and updated a `t5-small` model on the CNN/DailyMail dataset using targeted hyperparameter sweeps, minimizing validation loss.
4. **Evaluation**: Leveraged standard metrics (ROUGE, BLEU) and neural-embedding metrics (BERTScore) to assess summary fidelity.

## Dataset

- **Legal Cases**: Custom JSON records extracted from standard raw formats. Included specific tagging logic for categorizing legal discourse.
- **CNN/DailyMail Dataset**: Utilized for fine-tuning the T5 model, mapping article bodies to highly distilled highlight reference summaries.

## Implementation Details

- **Libraries**: `PyTorch`, `Transformers`, `Datasets`, `Evaluate`, `Rouge-Score`, `SacreBLEU`, `BERT-Score`.
- **Design Choices**: Extracted JSON structures for scalability instead of CSVs. Applied custom text chunking iterators to bypass Transformer maximum sequence token limits for massive legal records without causing arbitrary clipping.
- **Optimizations**: Used FP16 mixed-precision training and `num_proc` multiprocessing in `Dataset.map` functions to massively accelerate tokenization.

## Results & Metrics

- Successfully processed un-truncated document sequences, creating high-quality abstractive textual summaries.
- Evaluated models demonstrating high multi-gram overlap (ROUGE-L & ROUGE-1/2) alongside high continuous similarity representations via BERTScore F1 metric (averaging above 0.85+ semantic similarity).

## Key Learnings

- **Chunking Complexities**: Handling documents that exceed standard 512/1024 token limits requires intelligent sliding window or chunk-stitching methods to avoid losing vital contextual endpoints.
- **Generative Metric Trade-offs**: Recognized that exact-match metrics like BLEU often penalize good abstractive generation, making contextual equivalents like BERTScore pivotal in modern LLM evaluations.

## Improvements / Future Work

- **Long-Context Transformers**: Migrate from sliding window BART implementation to sparse architectures like Longformer/LED to capture end-to-end global document context spanning 4k+ tokens.
- **RLHF**: Incorporate a human-feedback layer to fine-tune the strictness and tonality of the output legal summaries.

## How to Run

1. Install dependencies: `pip install transformers datasets torch evaluate rouge-score bert-score sacrebleu`
2. Launch `Sumarization Pipeline.ipynb` module.
3. Trigger the data extraction logic cell to pull the CNN/DailyMail arrays for training processes.
