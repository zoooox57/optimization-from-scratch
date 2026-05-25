import csv
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(project_dir / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metric(path, metric):
    epochs = []
    values = []
    with path.open() as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["epoch"]))
            values.append(float(row[metric]))
    return epochs, values


def plot_metric(results_dir, figures_dir, metric, output_name, ylabel):
    plt.figure(figsize=(8, 5))

    for name in ["sgd", "momentum", "adam"]:
        path = results_dir / f"{name}.csv"
        epochs, values = read_metric(path, metric)
        plt.plot(epochs, values, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / output_name, dpi=150)
    plt.close()


def main():
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_metric(results_dir, figures_dir, "test_loss", "loss_curve.png", "Test Loss")
    plot_metric(
        results_dir,
        figures_dir,
        "test_acc",
        "accuracy_curve.png",
        "Test Accuracy",
    )


if __name__ == "__main__":
    main()
