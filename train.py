import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
import wandb
torch.set_float32_matmul_precision('high')

from transformer import SwinTransformer
from utils import learning_rate_schedule

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train(run, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_dataset = CIFAR100("./data", train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100("./data", train=False, download=True, transform=test_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Prepare SwinTransformer arguments
    depths = [int(x) for x in args.depths.split(",")] if isinstance(args.depths, str) else args.depths
    num_heads = [int(x) for x in args.num_heads.split(",")] if isinstance(args.num_heads, str) else args.num_heads
    d_ff_ratio = args.d_ff_ratio
    model = SwinTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        window_size=args.window_size,
        in_channels=args.in_channels,
        embed_dim=args.d_model,
        depths=depths,
        num_heads=num_heads,
        d_ff_ratio=d_ff_ratio,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    if args.use_compile:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    best_epoch = 0
    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        current_lr = learning_rate_schedule(
            t=epoch,
            lr_max=args.lr,
            lr_min=args.min_lr,
            t_warm_up=args.t_warm_up,
            t_cos_anneal=args.t_cos_anneal,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_samples += images.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        val_loss, val_acc = evaluate(model, test_dataloader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} "
            f"LR: {current_lr:.6e}"
        )

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": current_lr,
                "best/val_acc": best_acc,
                "best/epoch": best_epoch,
            }
        )

        if val_acc > best_acc + args.min_delta:
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}."
            )
            break

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {args.save_path}")
    run.summary["best_val_acc"] = best_acc
    run.summary["best_epoch"] = best_epoch
    run.summary["best_model_path"] = args.save_path
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Swin Transformer on CIFAR-100")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=96, help="Embedding dimension (embed_dim)")
    parser.add_argument("--depths", type=str, default="2,2,6,2", help="Comma-separated list, e.g. 2,2 for two stages")
    parser.add_argument("--num_heads", type=str, default="3,6,12,24", help="Comma-separated list, e.g. 4,8 for two stages")
    parser.add_argument("--d_ff_ratio", type=int, default=4, help="Feedforward ratio for SwiGLU")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--t_warm_up", type=int, default=10)
    parser.add_argument("--t_cos_anneal", type=int, default=120)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_delta", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--save_path", type=str, default="swin_cifar100_best.pt")
    return parser.parse_args()


if __name__ == "__main__":
    """
    • Swin-T: C = 96, layer numbers = {2, 2, 6, 2}  
    • Swin-S: C = 96, layer numbers ={2, 2, 18, 2}  
    • Swin-B: C = 128, layer numbers ={2, 2, 18, 2}  
    • Swin-L: C = 192, layer numbers ={2, 2, 18, 2}
    """
    args = parse_args()
    run = wandb.init(project="Swin Transformer", config=vars(args))
    train(run, args)


    