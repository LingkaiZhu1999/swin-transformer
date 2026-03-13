import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from transformers import SwinForImageClassification, AutoImageProcessor


def evaluate(model, dataloader, criterion, device):
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for images, labels in dataloader:
			images = images.to(device)
			labels = labels.to(device)

			logits = model(pixel_values=images).logits
			loss = criterion(logits, labels)

			total_loss += loss.item() * images.size(0)
			total_correct += (logits.argmax(dim=1) == labels).sum().item()
			total_samples += images.size(0)

	return total_loss / total_samples, total_correct / total_samples


def train(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	processor = AutoImageProcessor.from_pretrained(args.model_name)
	
	size_dict = processor.size
	image_size = size_dict.get("height", size_dict.get("shortest_edge", 224))
	
	normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

	train_transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]
	)
	test_transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			normalize,
		]
	)

	train_dataset = CIFAR100(args.data_dir, train=True, download=True, transform=train_transform)
	val_dataset = CIFAR100(args.data_dir, train=False, download=True, transform=test_transform)

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=(device == "cuda"),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=(device == "cuda"),
	)

	id2label = {idx: name for idx, name in enumerate(train_dataset.classes)}
	label2id = {name: idx for idx, name in id2label.items()}

	# CHANGED: Use SwinForImageClassification
	model = SwinForImageClassification.from_pretrained(
		args.model_name,
		num_labels=100,
		id2label=id2label,
		label2id=label2id,
		ignore_mismatched_sizes=True,
	).to(device)

	if args.linear_probe:
		# CHANGED: model.vit.parameters() -> model.swin.parameters()
		# Freeze the entire Swin backbone; only the classifier head remains trainable
		for param in model.swin.parameters():
			param.requires_grad = False
		trainable_params = model.classifier.parameters()
		print("Linear probe mode: backbone frozen, training classifier head only")
	else:
		trainable_params = model.parameters()

	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
	optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

	best_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		model.train()
		running_loss = 0.0
		running_correct = 0
		running_samples = 0

		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad(set_to_none=True)
			logits = model(pixel_values=images).logits
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			running_correct += (logits.argmax(dim=1) == labels).sum().item()
			running_samples += images.size(0)

		train_loss = running_loss / running_samples
		train_acc = running_correct / running_samples
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)

		print(
			f"Epoch [{epoch}/{args.epochs}] "
			f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
			f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
		)

		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict(), args.save_path)
			print(f"Saved best model to {args.save_path} (val_acc={best_acc:.4f})")

	print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


def parse_args():
	parser = argparse.ArgumentParser(description="Fine-tune pretrained Swin on CIFAR-100")
	# CHANGED: Default model to a Swin Transformer
	parser.add_argument("--model_name", type=str, default="microsoft/swin-tiny-patch4-window7-224")
	parser.add_argument("--data_dir", type=str, default="./data")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=3e-5)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--label_smoothing", type=float, default=0.1)
	parser.add_argument("--num_workers", type=int, default=2)
	# CHANGED: Default save path
	parser.add_argument("--save_path", type=str, default="swin_cifar100_finetuned.pt")
	parser.add_argument("--linear_probe", action="store_true", help="Freeze backbone and only train the classification head")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(args)