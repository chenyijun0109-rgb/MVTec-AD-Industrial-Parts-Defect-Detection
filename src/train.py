import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecBottleDataset, class_counts
from model import build_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 for MVTec bottle binary defect detection.")
    parser.add_argument("--metadata_path", type=Path, default=Path("results/metadata.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MVTecBottleDataset(args.metadata_path, split="train", image_size=args.image_size)
    val_dataset = MVTecBottleDataset(args.metadata_path, split="val", image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    counts = class_counts(train_dataset)
    total = counts[0] + counts[1]
    weights = torch.tensor([total / max(counts[0], 1), total / max(counts[1], 1)], dtype=torch.float32, device=device)

    model = build_resnet18(num_classes=2, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    log_path = args.output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                }
            )
            csv_file.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "image_size": args.image_size,
                        "val_acc": best_val_acc,
                        "class_counts": counts,
                    },
                    args.output_dir / "best_model.pth",
                )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to {args.output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
