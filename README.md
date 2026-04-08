# 🧠 Deep Learning & Machine Learning Portfolio

## Overview
This repository contains a comprehensive collection of advanced machine learning and deep learning projects. Each module demonstrates hands-on expertise in formulating machine learning logic, architecting complex neural networks from scratch, and leveraging state-of-the-art foundation models via transfer learning. The domains span Computer Vision (CV), Natural Language Processing (NLP), and Reinforcement Learning (RL).

The emphasis represents an engineering-first mentality targeting hyper-parameter optimization, deep observability (e.g. gradient tracing), model evaluation across nuanced matrices, and application deployments natively bridging Python ML backends to scalable API pipelines.

---

## 📂 Projects

| Project / Module | Domain | Frameworks & Techniques | Highlights & Impact |
|------------------|--------|-------------------------|---------------------|
| **[Transformer Architectures](./Transformer/)** | NLP & CV | Hugging Face, PyTorch, DistilBERT, ViT, BART, FastAPI | Outperformed baselines utilizing DistilBERT for SPAM mapping (**97.96% F1**). Reached **99%+ accuracy** via pre-trained Vision Transformers (ViT) within 3 epochs and designed a deployment-ready FastAPI backend. Built an Encoder-Only architecture entirely from scratch. |
| **[Summarization Pipeline](./Summarizer%20Pipeline/)** | NLP | PyTorch, `transformers`, `sacrebleu`, T5, BART-Large | Engineered a comprehensive Seq2Seq pipeline extracting overlapping text sequences. Built abstractive representations of legal articles & CNN datasets leveraging zero-shot generation and fine-tuning. Evaluated semantic similarity scoring through BERTScore F1 metric. |
| **[Reinforcement Learning Engines](./RL/)** | RL | OpenAI Gymnasium, NumPy, Matplotlib, SARSA, Q-Learning | Implemented advanced logic modeling (Double Q-Learning and N-Step trajectories) avoiding maximization bias sub-optimality. Explored and mapped stochastic grids finding convergence parameters iteratively testing variable `Epsilon Decay` and `Discount Factors`. |
| **[Various Model Architectures](./Various%20Model%20Architectures/)** | CV, NLP, Time Series | PyTorch, VGG, ResNet, Bidirectional LSTM, Scikit-Learn | Extracted hard metric proofs of Vanishing Gradient collapses on deep un-normalized VGG architectures using native PyTorch backward hooks to track L2 norms. Mapped temporal tracking Bidirectional LSTMs predicting sequence sliding windows with rigorous RMSE evaluation. |
| **[Capsule Networks](./Capsule%20Networks%20/)** | Computer Vision | PyTorch, Dynamic Routing, Custom Image Processing vectors | Executed advanced spatial-preservation architecture leveraging Vector mapping compared to standard numerical mappings. Recreated original classification structures preserving dynamic image bounds and reconstructing images from embedded caps matrices utilizing Custom PyTorch functions. |

---

## 🛠️ Global Technical Environment
- **Core ML Frameworks**: PyTorch, Hugging Face Transformers, Hugging Face Datasets
- **Scientific Computing**: NumPy, Pandas, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Deployment Structure**: FastAPI, Uvicorn
- **Evaluation Utilities**: Evaluate, Rouge-Score, SacreBLEU, BERT-Score

## 🚀 Execution & Setup
To run any of the specific project modules:
1. Ensure your local environment is mapped appropriately via `pip install -r requirements.txt` (Assume standardized libraries listed in technical environments).
2. For specific API models, `uvicorn` and FastAPI deployment handlers run synchronously bridging endpoint logic. 
3. Datasets utilized can be instantiated mostly through `datasets` loaders mapping directly into automated cache hierarchies.

Leverage the internal `README.md` documents enclosed inside each project tracking specific architectural workflows and tuning results. 
