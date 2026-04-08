# Multi-Disciplinary Transformer Architectures

## Overview
This comprehensive repository showcases the implementation of diverse Transformer and Autoencoder models to tackle distinct challenges in Computer Vision, Sequence-to-Sequence NLP, Text Classification, and Sequential Anomaly tracking. The methodologies demonstrate full-stack deep learning proficiency from building foundational attention mechanisms from scratch to orchestrating transfer learning via powerful Hugging Face backbones.

## Key Projects & Concepts
1. **Vision Transformer (ViT)**
   - **Concept**: Applied transfer learning using `google/vit-base-patch16-224` to classify Cats vs Dogs.
   - **Impact**: Achieved **>99% Validation Accuracy** after just three epochs, highlighting the immense pattern recognition capabilities of large patch-based vision models. Handled deployment via a highly scalable `FastAPI` endpoint mimicking production CV pipelines.
2. **Spam Classification via LLM Linear Probing (DistilBERT vs ALBERT)**
   - **Concept**: Froze underlying high-parameter LLM weights and trained a rapid, lightweight MLP Classification Head across custom Enron datasets. 
   - **Impact**: Engineered the DistilBERT pipeline to hit **97.96% F1 Score and 97.95% Accuracy**, drastically outperforming expectations (>85% baseline) while saving 90% of standard fine-tuning compute requirements.
3. **Seq2Seq NLP Abstractive Summarizer**
   - **Concept**: Fine-tuned a BART foundation model on congressional `BillSum` datasets leveraging custom `Seq2SeqTrainer` pipelines with FP16 precision.
   - **Impact**: Engineered high-throughput text summarizations, effectively optimizing training logic across evaluation callbacks. Extensively analyzed metrics via ROUGE and Semantic-BERTScore.
4. **Foundational Transformer Decoder (From Scratch)**
   - **Concept**: Developed an **Encoder-Only Transformer Classifier** natively via standard PyTorch matrix primitives, crafting isolated Positional Encodings, multi-head self-attention paths, and dense layer normalizations.
5. **Autoencoders (Dense, LSTM, Conv1D)**
   - **Concept**: Constructed sequential Autoencoders targeting robust representations. Designed extensive grid-searching loops iteratively testing hyper-parameter boundaries and plotting comparative Loss trajectories.

## Architecture / Approach
- **Transfer Learning**: Seamlessly orchestrated Hugging Face `Trainer` loops, adjusting learning rates, batch sizes, and weight decay to maximize convergence paths.
- **API Deployment**: Deployed PyTorch models directly into robust `FastAPI` backends processing payload image byte-streams and returning pure JSON inference targets, effectively bridging ML logic and software engineering endpoints.
- **Probing**: Capitalized on existing dense representation vector spaces via linear probing, preserving base model integrity while extracting extremely high predictive capacity with minimal gradient updates.

## Implementation Details
- **Frameworks**: PyTorch, Hugging Face `transformers` & `datasets`, `FastAPI`, Scikit-Learn.
- **Design Choices**: Extensively leveraged mixed precision (FP16) training to manage computational overhead during Transformer model adjustments and managed custom PyTorch backward loops.

## Results & Metrics
- **Spam Filtering**: **98.21% Precision**, **97.95% Accuracy** utilizing DistilBERT encoded features.
- **Computer Vision (ViT)**: Reached peak accuracy with under 0.015 Val Loss in 3 epochs.
- **NLP Text Classification**: Natively mapped raw sequences directly against dynamic memory matrices in the custom 'Transformer From Scratch' execution.

## Key Learnings
- **Probing Over Fine-Tuning**: Realized the tangible value of freezing base configurations on specialized tasks. Fine-tuning ALBERT yielded under 70% metrics while DistilBERT scaled immediately, representing the importance of backbone selection.
- **Inference Latency**: Using robust CV platforms within standard REST endpoints proved extremely performant provided batch mapping and tokenizations were strictly optimized in the deployment stack.

## Improvements / Future Work
- Distill customized heavy transformer models using Knowledge Distillation mechanisms to compress deployment architecture latency further.
- Explore cross-attention bridging between CV models and Auto-regressive LLMs to begin mapping VQA (Visual Question Answering) networks.

## 🛠️ How to Run
1. Install requirements: `pip install torch transformers datasets scikit-learn fastapi uvicorn evaluate`
2. For the ViT API endpoint: Run `uvicorn model_deploy:app --host 127.0.0.1 --port 8000 --reload`
3. Traverse target notebooks, ensuring appropriate access to standard GPU acceleration via CUDA.
