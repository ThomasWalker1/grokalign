# mnist_train_script.py
import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import wandb as wb

from utils import GrokAlign, Centroids

def get_data(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.flatten())
    ])
    train_ds = torchvision.datasets.MNIST(root=config.data_dir, train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(root=config.data_dir, train=False, transform=transform, download=True)
    train_ds = torch.utils.data.Subset(train_ds, range(config.train_points))
    test_ds = torch.utils.data.Subset(test_ds, range(config.test_points))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def build_model(config):
    layers = [nn.Linear(784, config.width, bias=False), nn.ReLU()]
    for _ in range(config.depth - 2):
        layers += [nn.Linear(config.width, config.width, bias=False), nn.ReLU()]
    layers += [nn.Linear(config.width, 10, bias=False)]
    model = nn.Sequential(*layers)
    for p in model.parameters():
        p.data = 8.0 * p.data
    return model

def compute_accuracy(model, loader, device):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += x.size(0)
    return correct / total

def centroid_statistics(centroids, x):
    centroids(x)
    norm = centroids.get_norms()
    alignment = centroids.get_alignments()
    return {'centroid_norm': norm.mean().item(), 'centroid_alignment': alignment}

def train(config):
    run_name = f"Jr_at_{config.jac_level}"
    wb.init(project='inhibiting_generalisation', config=config, name=run_name)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if 'cuda' in config.device:
        torch.cuda.manual_seed_all(config.seed)

    model = build_model(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    one_hots = torch.eye(10).to(config.device)
    grokalign = GrokAlign(model) if config.jac_level > 0 else None
    centroids = Centroids(model)

    train_loader, test_loader = get_data(config)
    logged_steps = np.unique(np.append(np.logspace(0, np.log10(config.steps), config.num_logs, dtype=int), [0, config.steps]))

    for step in tqdm(range(config.steps + 1)):
        model.train()
        for x, labels in train_loader:
            x, labels = x.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, one_hots[labels])
            if grokalign:
                loss += 1e-3 * torch.abs(config.jac_level - grokalign(x))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step in logged_steps:
            stats = {'step': step}
            for name, loader in {'train': train_loader, 'test': test_loader}.items():
                stats[f'{name}_accuracy'] = compute_accuracy(model, loader, config.device)
            stats.update(centroid_statistics(centroids, x))
            wb.log(stats)

    wb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_points', type=int, default=1024)
    parser.add_argument('--test_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=196)
    parser.add_argument('--steps', type=int, default=32000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--jac_level', type=float, default=0.0)
    parser.add_argument('--width', type=int, default=196)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_logs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    train(args)