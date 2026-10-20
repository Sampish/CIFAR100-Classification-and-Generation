# Efficient CIFAR-100 Modelling

How well can deep learning models perform on CIFAR-100 (60,000 32×32 images, 100 classes) when parameters and training steps are strictly limited? This project tackles two tasks under fixed budgets.

## The problem

| Task | Constraints | Target |
|------|-------------|--------|
| **Classification** | < 100,000 trainable parameters, 10,000 optimisation steps | ≥ 50% test accuracy |
| **Generation** | < 1,000,000 parameters, 50,000 optimisation steps | Realistic, diverse unconditional samples |

At this scale a standard CNN quickly runs out of budget, so every design choice is about parameter efficiency.

## Results

| Task | Model | Parameters used | Result |
|------|-------|-----------------|--------|
| Classification | Depthwise separable residual CNN | 99,416 | **57.7%** test accuracy |
| Generation | Residual GAN | 822,324 | **FID 46.3** |

## Approach

**Classifier:**
- Depthwise separable convolutions keep the parameter cost low.
- Residual connections and BatchNorm keep training stable.
- Global average pooling replaces the fully connected layers.
- I tuned hyperparameters with a 216-configuration grid search.

**GAN:**
- The generator and discriminator are both built from residual blocks.
- Training uses hinge loss and differentiable augmentation (DiffAugment).
- Sampling uses an EMA of the generator weights.

The full methods, results and limitations are in [`hqcb32-paper.pdf`](hqcb32-paper.pdf).

## Files

- `hqcb32-classifier.ipynb` / `.py`: classifier
- `hqcb32-classifier-parameter_optimisation.ipynb`: hyperparameter search
- `hp_search_results_20260103_153953.json`: search results
- `hqcb32-generative-model.ipynb`: GAN
- `hqcb32-paper.pdf`: report

## Run

```bash
pip install torch torchvision numpy matplotlib jupyter
jupyter notebook
```

CIFAR-100 downloads automatically via `torchvision`.