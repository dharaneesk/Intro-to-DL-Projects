# Capsule Networks for Image Classification

## Overview

This project implements a custom Capsule Network architecture using PyTorch for advanced image classification. By modeling hierarchical visual structures differently from traditional CNNs, Capsule Networks preserve spatial relationships and part-whole dependencies, improving performance on specific edge cases in image recognition by dynamically routing information.

## Key Concepts

- **Capsule Networks**: Utilizing vector-based feature representations to maintain the properties of entities.
- **Dynamic Routing**: An algorithm that determines the strength of the connection between lower-level capsules and higher-level capsules based on agreement.
- **Margin Loss & Reconstruction**: A dedicated loss function coupled with a reconstruction mechanism acting as a regularizer.

## Architecture / Approach

1. **Input Processing**: Standardized data ingestion and encoding via convolutional layers.
2. **Primary Capsules**: Extracts scalar features and transforms them into multidimensional capsules.
3. **Digit Capsules**: Applies dynamic routing by agreement to accurately classify inputs into 10 distinct classes.
4. **Decoder**: Reconstructs the input image from the capsule representation to compute the reconstruction loss, enforcing representation robustness.

## Dataset

- **Source & Preprocessing**: Standardized baseline 10-class dataset (e.g., MNIST/Fashion-MNIST). Data is normalized, one-hot encoded for custom loss function targets, and batched using PyTorch `DataLoaders`.

## Implementation Details

- **Framework**: Developed entirely in PyTorch.
- **Optimization**: Adam optimizer with custom dynamic learning rates.
- **Design Decisions**: Implemented custom `plot_metric_curve` utilities and automated misclassification visualizers to better interpret model failure cases.

## Results & Metrics

- **Performance**: High fidelity classification accuracy with stable progression in validation metrics across 10 epochs.
- **Insight**: Reconstructed images mapped accurately to original inputs, validating the representational power of the capsules. Visualizing misclassified images exposed specific morphological ambiguities within the dataset.

## Key Learnings

- **Vector Operations in PyTorch**: Mastered tensor permutations and custom dynamic routing algorithms which differ significantly from scalar neuron activations in typical deep networks.
- **Regularization through Reconstruction**: Learned how using a secondary task (reconstruction) effectively restrains overfitting in complex networks.

## Improvements / Future Work

- **Performance Scaling**: Investigate matrix-routing (CapsNet v2) to improve computational speed.
- **Complex Datasets**: Applying the Capsule architecture to overlapping digits or affine transformed image datasets to benchmark translation invariance.

## 🛠️ How to Run

1. Install dependencies: `pip install torch torchvision matplotlib`
2. Open and run the Jupyter notebook `Capsule_Networks_Intro.ipynb`.
3. Model artifacts and metrics automatically save to `model.pt` and `model_values.pkl`.
