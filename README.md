## Motivation

Optimization algorithms are fundamental to deep learning training.
This project aims to better understand how different optimizers affect convergence behavior and training stability by implementing them from scratch.
# Optimization from Scratch

This project compares three optimizers implemented from scratch on MNIST:

- SGD
- Momentum
- Adam

## Files

- `models.py`: neural network model
- `optimizers.py`: optimizer implementations
- `train.py`: trains the model with SGD, Momentum, and Adam
- `plot.py`: plots loss and accuracy curves from CSV results
- `results/`: generated CSV files
- `figures/`: generated curve images

## Run

```bash
source .venv/bin/activate
python train.py
python plot.py
```

For a longer run:

```bash
python train.py --epochs 10 --train-limit 60000 --test-limit 10000
python plot.py
```

For SGD and Momentum, I use the same learning rate to isolate the effect of the momentum term.
For Adam, I keep the standard beta values beta1=0.9 and beta2=0.999, because these are part of the Adam algorithm's default design.

In this experiment, SGD with a well-chosen learning rate performs very competitively.
Adam reduces training loss quickly, but does not always produce the best test accuracy.
Momentum can improve convergence over vanilla SGD, but its performance depends strongly on the learning rate.