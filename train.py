import argparse
import csv
import ssl
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import MnistMLP
from optimizers import Adam, Momentum, SGD


ssl._create_default_https_context = ssl._create_unverified_context


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(data_dir, train, batch_size, limit=None):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def accuracy(outputs, labels):
    predictions = outputs.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(outputs, labels) * batch_size
            total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


def make_optimizer(name, params):
    if name == "sgd":
        return SGD(params, lr=0.1)
    if name == "momentum":
        return Momentum(params, lr=0.05, momentum=0.9)
    if name == "adam":
        return Adam(params, lr=0.001)
    raise ValueError(f"Unknown optimizer: {name}")


def train_optimizer(name, train_loader, test_loader, device, epochs, results_dir):
    torch.manual_seed(0)

    model = MnistMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(name, model.parameters())
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_acc += accuracy(outputs, labels) * batch_size
            total_samples += batch_size

        train_loss /= total_samples
        train_acc /= total_samples
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        rows.append(row)

        print(
            f"{name:8s} epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    output_path = results_dir / f"{name}.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=2000)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    device = get_device()
    print("device:", device)

    train_loader = build_loader(data_dir, True, args.batch_size, args.train_limit)
    test_loader = build_loader(data_dir, False, args.batch_size, args.test_limit)

    for optimizer_name in ["sgd", "momentum", "adam"]:
        train_optimizer(
            optimizer_name,
            train_loader,
            test_loader,
            device,
            args.epochs,
            results_dir,
        )


if __name__ == "__main__":
    main()
