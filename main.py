import argparse
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from src.config import Config
from src.data_loader import get_cifar100_loaders
from src.engine import evaluate, train_one_epoch
from src.models import get_model
from src.utils import (
    ExperimentTracker,
    apply_pruning,
    apply_quantization,
    get_model_size_mb,
    log_to_csv,
    set_seed,
)


def run_experiment(args):
    set_seed(Config.SEED)
    device = Config.get_device(args.technique)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    print(f"STARTING: {args.model} | {args.technique} | device={device}")
    use_aug = args.technique != "quantization"
    train_loader, val_loader, test_loader = get_cifar100_loaders(augment=use_aug)

    model = get_model(args.model, num_classes=Config.NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    if args.technique == "quantization":
        candidates = [
            f"{args.model}_pruning_{args.pruning_amount}_best.pth",
            f"{args.model}_baseline_best.pth",
        ]
        load_path = next((os.path.join(Config.RESULT_DIR, p) for p in candidates if os.path.exists(os.path.join(Config.RESULT_DIR, p))), None)
        if load_path is None:
            raise FileNotFoundError("Train a baseline or pruning checkpoint before quantization.")

        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        if "pruning" in load_path:
            model = apply_pruning(model, amount=args.pruning_amount)

        tracker = ExperimentTracker(f"{args.model}_quantization")
        tracker.start()
        quantized_model = apply_quantization(model, train_loader)
        _, test_acc = evaluate(quantized_model, test_loader, criterion, "cpu")
        elapsed, energy = tracker.stop()

        save_path = os.path.join(Config.RESULT_DIR, f"{args.model}_quantized.pth")
        torch.save(quantized_model.state_dict(), save_path)
        log_to_csv({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": args.model,
            "technique": "quantization",
            "pruning_amount": args.pruning_amount,
            "best_epoch": 0,
            "val_accuracy": test_acc,
            "test_accuracy": test_acc,
            "total_time_min": elapsed,
            "total_energy_kwh": energy,
            "model_size_mb": get_model_size_mb(save_path),
        })
        return

    if args.technique == "pruning":
        base_path = os.path.join(Config.RESULT_DIR, f"{args.model}_baseline_best.pth")
        if os.path.exists(base_path):
            model.load_state_dict(torch.load(base_path, map_location=device))
        model = apply_pruning(model, amount=args.pruning_amount)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        experiment_name = f"{args.model}_pruning_{args.pruning_amount}"
    else:
        optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        experiment_name = f"{args.model}_baseline"

    scheduler = MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    tracker = ExperimentTracker(experiment_name)
    best_acc = -1.0
    best_epoch = 0
    save_path = os.path.join(Config.RESULT_DIR, f"{experiment_name}_best.pth")

    tracker.start()
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} | loss={train_loss:.4f} | val_acc={val_acc:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)

    elapsed, energy = tracker.stop()
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)

    log_to_csv({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model,
        "technique": args.technique,
        "pruning_amount": args.pruning_amount if args.technique == "pruning" else 0,
        "best_epoch": best_epoch,
        "val_accuracy": best_acc,
        "test_accuracy": test_acc,
        "total_time_min": elapsed,
        "total_energy_kwh": energy,
        "model_size_mb": get_model_size_mb(save_path),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Green AI experiment runner")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v2"], default="resnet50")
    parser.add_argument("--technique", choices=["baseline", "pruning", "quantization"], default="baseline")
    parser.add_argument("--pruning_amount", type=float, default=0.3)
    run_experiment(parser.parse_args())
