"""
Quick test: HGM discovered architecture vs Transformer baseline
Fast trainability comparison on MNIST
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from architectures import ArchitectureWrapper, create_transformer_baseline
import json
import time

def get_quick_data(n_samples=1000):
    """Get small MNIST subset for fast testing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    indices = torch.randperm(len(dataset))[:n_samples]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=64, shuffle=True)

class QuickClassifier(nn.Module):
    """Wrap backbone for classification"""
    def __init__(self, backbone, input_size=784, hidden=64, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden)
        self.backbone = backbone
        self.output = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_proj(x).unsqueeze(1)
        x = self.backbone(x).squeeze(1)
        return self.output(x)

def quick_train(model, loader, device, steps=50):
    """Train for N steps and measure performance"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = 0.0
    correct = 0
    total = 0
    step = 0

    start_time = time.time()

    for epoch in range(10):  # Multiple passes
        for data, target in loader:
            if step >= steps:
                break

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            step += 1

        if step >= steps:
            break

    train_time = time.time() - start_time
    accuracy = 100. * correct / total
    avg_loss = total_loss / steps

    return accuracy, avg_loss, train_time

def quick_eval(model, loader, device):
    """Quick evaluation"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100. * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    print("Loading MNIST data...")
    train_loader = get_quick_data(1000)
    test_loader = get_quick_data(500)

    # Load discovered architecture
    print("Loading discovered architecture...")
    with open('ultimate.json', 'r') as f:
        result = json.load(f)
        discovered_spec = result['best_spec']

    # Create baseline
    print("Creating transformer baseline...")
    baseline_spec = create_transformer_baseline(dim=64, heads=4, layers=2)

    # Build models
    discovered_backbone = ArchitectureWrapper(discovered_spec).to(device)
    baseline_backbone = ArchitectureWrapper(baseline_spec).to(device)

    discovered_model = QuickClassifier(discovered_backbone, hidden=64).to(device)
    baseline_model = QuickClassifier(baseline_backbone, hidden=64).to(device)

    # Count parameters
    discovered_params = sum(p.numel() for p in discovered_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())

    print(f"\nDiscovered params: {discovered_params:,}")
    print(f"Baseline params:   {baseline_params:,}")
    print(f"Ratio: {discovered_params/baseline_params:.2f}x")

    print("\n" + "=" * 60)
    print("QUICK TRAINING TEST (50 steps)")
    print("=" * 60)

    # Test discovered
    print("\n[1/2] Training HGM Discovered Architecture...")
    disc_acc, disc_loss, disc_time = quick_train(discovered_model, train_loader, device, steps=50)
    print(f"  Accuracy: {disc_acc:.2f}%")
    print(f"  Loss: {disc_loss:.4f}")
    print(f"  Time: {disc_time:.2f}s")

    # Test baseline
    print("\n[2/2] Training Transformer Baseline...")
    base_acc, base_loss, base_time = quick_train(baseline_model, train_loader, device, steps=50)
    print(f"  Accuracy: {base_acc:.2f}%")
    print(f"  Loss: {base_loss:.4f}")
    print(f"  Time: {base_time:.2f}s")

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    disc_eval = quick_eval(discovered_model, test_loader, device)
    base_eval = quick_eval(baseline_model, test_loader, device)

    print(f"\nHGM Discovered: {disc_eval:.2f}%")
    print(f"Transformer:    {base_eval:.2f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTraining Performance (50 steps):")
    print(f"  HGM: {disc_acc:.2f}% accuracy in {disc_time:.2f}s")
    print(f"  TRF: {base_acc:.2f}% accuracy in {base_time:.2f}s")
    print(f"  Speed: {base_time/disc_time:.2f}x {'faster' if disc_time < base_time else 'slower'}")

    print(f"\nEval Accuracy:")
    print(f"  HGM: {disc_eval:.2f}%")
    print(f"  TRF: {base_eval:.2f}%")
    print(f"  Delta: {disc_eval - base_eval:+.2f}%")

    print(f"\nParameters:")
    print(f"  HGM: {discovered_params:,}")
    print(f"  TRF: {baseline_params:,}")
    print(f"  Ratio: {discovered_params/baseline_params:.2f}x")

    if disc_eval > base_eval:
        print("\n✅ HGM architecture outperforms baseline!")
    elif abs(disc_eval - base_eval) < 1.0:
        print("\n⚖️  Comparable performance")
    else:
        print("\n⚠️  Baseline outperforms (search needs more time)")

if __name__ == "__main__":
    main()
