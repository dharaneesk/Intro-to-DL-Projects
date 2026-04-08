# Advanced Deep Learning Architectures: Analyzing Gradients & Sequential Models

## Overview

Engineered and evaluated an array of classic multi-layer architectures to directly analyze and tackle persistent Neural Network phenomenons, specifically solving sequential forecasting/sentiment mapping challenges utilizing Bidirectional LSTMs alongside analyzing Vanishing Gradients on Deep VGG-Style CNN paradigms.

## Key Concepts

- **Gradient Flow Analysis**: Deep interrogation of sequential backpropagation by hooking directly into layer gradients and recording norm decays in hyper-deep CNNs.
- **Sequential Recurrences (LSTMs)**: Resolving context dependencies across textual inputs leveraging variable gating.
- **Time Series Regressions**: Formulating sliding window abstractions atop normalized numerical sets for iterative predictive tracking.

## Architecture / Approach

1. **Vanishing Gradient Deep CNN Analysis ([VGG & ResNet Architecture])**:
   - Constructed a highly deep, synthetic VGG-style CNN.
   - Deployed PyTorch `register_full_backward_hook` triggers to intercept backpropagation calculations and cache the precise `L2 Gradient Norms` layer over layer.
   - Proved mathematically and visually how earlier layers in unstructured deep models exponentially decay to gradients approximating `0`, starving updates. Formatted ResNet equivalents utilizing skip connections to bypass and resolve this failure pattern.
2. **Sentiment Analysis LSTM**:
   - Utilized a 3-Layered Deep LSTM taking mapped vocabulary integers, ending in Sigmoid evaluations mapped across PyTorch's `BCE Loss` logic.
3. **Time Series Dual-LSTM Forecaster**:
   - Engineered Bidirectional LSTMs predicting temporal sequences windowed into 48-step rolling context structures. Implemented targeted MSE tracking.

## Dataset

- **Textual Processing**: Amazon Product Reviews Dataset (`__label__` extractions). Designed custom vocabulary tokenizers mapping strings to padding-enforced vectors.
- **Image Sets**: Reshaped visual image datasets (VGG tasks).
- **Time Series**: Standardized feature engineering logic and numerical normalization via inverse scalar transformation to present comprehensible RMSE tracking.

## Implementation Details

- **Libraries**: `PyTorch`, `NLTK`, Scikit-Learn, `Seaborn`.
- **System Memory Optimizations**: Employed explicit `gc.collect()` and `torch.cuda.empty_cache()` with `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` flags to effectively stream high-volume memory traces without CUDA OOM limits.
- **Design Choices**: Extracted exact statistical reports across Precision, Recall, and F1 to interpret multi-class boundaries comprehensively. Used extensive early stopping loops bound to custom validation tracking routines.

## Results & Metrics

- **Time Series Analytics**: Extracted solid R² metrics backed by rigorous manual Hyper-parameter tuning loops stabilizing sequence duration impacts between sizes `24` and `48`.
- **Vanishing Gradients**: Produced compelling visual mappings tracking exponential gradient decay down layer mappings, mathematically grounding the absolute necessity of non-linear standardizations (Batch Norms) and Residual pathways in un-bounded depth engineering.
- **NLP Sentiment**: Mapped review tokenization mapping sequences effectively bridging natural language structures into fixed computational matrix formats.

## Key Learnings

- **Hook Mapping & Internal Tracing**: Mastering internal `backward_hooks` in PyTorch provides unparalleled deep observability into tensor degradation otherwise completely obscured in standard black-box network execution.
- **Manual Backprop Management**: Validated that `Sigmoid` limits inherently suppress variance in extensive networks significantly more than standard `ReLU` / `He Initialization`, driving fundamental design decisions in all consequent CV architectures.

## Improvements / Future Work

- Implement explicit gradient clipping features to evaluate Gradient Exploding patterns on unbounded sequence datasets.
- Adopt modern Layer Normalization and RMSNorm tracking over standard Batch Normalization to test temporal independence scaling in long-tier sequences.

## How to Run

1. Install dependencies: `pip install torch torchvision nltk torchinfo scikit-learn`
2. Download target datasets using external links outlined in notebook cells.
3. Open target architecture files (e.g. `Vanishing Gradient Analysis.ipynb`) to explore visual gradient decays mapping dynamically in output cells.
