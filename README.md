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
